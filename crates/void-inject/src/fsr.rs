//! FSR 4 in place of the engine TAA (PLAN G1, docs/fsr4-integration.md): native AA (FSR
//! anti-aliases at the engine's render size) or upscale (`fsr_mode = upscale`: the engine
//! renders at `fsr_scale`, FSR writes display size, `fsr_scale.rs` switches the post chain).
//!
//! Engine-level design (no command-list split, docs/fsr4-integration.md "Engine-level
//! alternative without a split"):
//! - Record time (worker thread): the TAA `call Ctx_Dispatch` at 0x140327817 inside
//!   `PostFX_TemporalAA_and_Post` (0x140327270, pass `ARK_RENDERPASS_POSTPROC`) goes to
//!   [`taa_dispatch`] through a stub that adds `r14` (the pass). With FSR ready it reads the
//!   TAA inputs from the engine context's parameter block and the view, queues a [`Job`]
//!   for the pass, and records nothing (the dispatch is skipped). TAA is the first GPU work
//!   of that pass's command list.
//!   Our reflections (`ssr.rs`) are captured at the same site: FSR carries them to the
//!   submit hook (run before the pack); with the engine TAA they are recorded into the
//!   pass's deferred context before its dispatch. `ssr` also installs this site alone
//!   ([`install_taa_site`]) when FSR is off.
//! - Submit time (whichever worker submits, inside the engine's immediate-context lock):
//!   the `call [rax+1D0h]` (`ExecuteCommandList(immediate, pass+0x118, FALSE)`) at
//!   0x1402C6067 in `sub_1402C5F60` goes to [`submit`] (stub adds `rbx` = pass). For a
//!   queued job it packs the inputs into shared textures, signals a shared fence, runs FSR
//!   on our D3D12 queue (GPU waits only, between command lists, never inside a driver
//!   replay: E047), waits for it on the D3D11 side, writes the result into `taaResult`
//!   (`hdrBuffer2`), then executes the engine's list.
//!
//! Units (d2enhanced TAA shader analysis, lead): `hdrBuffer` and `hdrBuffer2` hold the same
//! "stored" HDR units; the exposure passed to FSR is the engine TAA's tonemap scale
//! `exp(-customEVBias) * ELF * (compressedHdr ? ELF * viewWhite : 1)`, ELF from the
//! auto-exposure buffer element `exposureIndex.y`.
//!
//! Engine scheduler without command lists (`r_schedulerUseDeferredCtx 0`, or -1 on a
//! driver without driver command lists; TaaSiteRE, docs/performance-baseline.md): every
//! GPU pass records on worker 0 directly on the immediate context and nothing goes through
//! 0x1402C6067. The TAA site then runs the same work inline ([`execute_job`] on the
//! immediate context) and invalidates the engine context's shader caches afterwards.
//!
//! Jitter, render scale, post-chain switch, mip bias and the OIT mask inputs live in
//! `fsr_scale.rs`.
//!
//! Fallbacks: until FSR is ready (or after a failure) the engine TAA runs unchanged; a
//! job whose FSR dispatch fails gets a plain copy of the input so the frame is not
//! garbage. Turning FSR off re-arms the engine's TAA history reset (`rs+8`).

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_11_0, D3D11_SRV_DIMENSION_BUFFER, D3D11_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE, IDXGIAdapter1,
    IDXGIDevice, IDXGIFactory4, IDXGIResource1,
};
use windows::core::{Interface, PCSTR, s};

use hex_literal::hex;
use iced_x86::code_asm::{r14, rbx};

use crate::{ffx, log, mem, settings};

// ------------------------------------------------------------------ engine sites --

/// `call Ctx_Dispatch` of the TAA pass (O, IDB); registers at the call:
/// rcx = engine context, edx/r8d = groups, r9d = 1, r14 = pass, r15 = view.
const TAA_CALL_RVA: u64 = 0x32_7817;
const TAA_CALL_BYTES: [u8; 5] = hex!("E8 44 3C F9 FF");
/// `Ctx_Dispatch(engineCtx, x, y, z)` = `ID3D11DeviceContext::Dispatch(ctx->d3d, x, y, z)`.
const CTX_DISPATCH_RVA: u64 = 0x2B_B460;
/// `call [rax+1D0h]` = `ExecuteCommandList(immediate, pass+0x118, FALSE)` in
/// `sub_1402C5F60`, inside `RS_LockImmediate` / `RS_UnlockImmediate`; rbx = pass.
const SUBMIT_CALL_RVA: u64 = 0x2C_6067;
const SUBMIT_CALL_BYTES: [u8; 6] = hex!("FF 90 D0 01 00 00");
/// `PostFX_TemporalAA_and_Post`, the POSTPROC pass callback (pass +0xD8).
const POSTPROC_CALLBACK_RVA: u64 = 0x32_7270;
/// Immediate `ID3D11DeviceContext*` used by the submit path (`qword_1432D4EF8`).
const IMMEDIATE_CTX_RVA: u64 = 0x32D_4EF8;
/// `idRenderSystemLocal` (`unk_1432FC580`): +8 TAA history-reset counter (int, consumed
/// twice per TAA frame), +0xCC frame counter (TAA parity, jitter index).
pub(crate) const RS_RVA: u64 = 0x32F_C580;
const RS_TAA_RESET: u64 = 8;
pub(crate) const RS_FRAME: u64 = 0xCC;
/// `ARK_RENDERPASS_POST_TONEMAP` callback (pass +0xD8).
const POST_TONEMAP_CALLBACK_RVA: u64 = crate::fsr_scale::POST_TONEMAP_RVA;
/// Engine render context: cached input layout / VS / PS / CS pointers (cleared at pass
/// start; docs "Engine context state caches"). Zeroed after inline work so the next
/// program bind sets every shader again.
const ENGINE_CTX_SHADER_CACHE: u64 = 0x12B88;
/// `dword_14397A6F8`: TAA material variant; `useCompressedHdrBuffers = (value != 1)`
/// (`SetPerViewShaderParms` 0x1403587B7).
const HDR_MODE_RVA: u64 = 0x397_A6F8;

// Parameter handles (globals pointing at parameter declarations; slot index at +0x48).
const H_VIEW_COLOR: u64 = 0x22B_25D8;
const H_MOTION: u64 = 0x22B_27E0;
const H_TAA_RESULT: u64 = 0x22B_3500;
const H_DEPTH: u64 = 0x22B_1F18;
const H_EXPOSURE_BUFFER: u64 = 0x22B_28E0;
const H_EXPOSURE_INDEX: u64 = 0x22B_2900;
const H_EV_BIAS: u64 = 0x22B_2920;
/// `PostFX/Luminance/white_level`; taken as d2enhanced's `view_white_level` (H).
const H_WHITE_LEVEL: u64 = 0x22B_2960;

// Pass, view and image-view layouts (docs/fsr4-integration.md, reflection + code).
const PASS_VIEW: u64 = 0x70;
const PASS_CALLBACK: u64 = 0xD8;
const VIEW_JITTER: u64 = 30800;
/// `m_previousFrameNumber`: the frame id `idRenderView_Setup` used for this view.
const VIEW_FRAME: u64 = 30644;
const VIEW_VIEWPORT: u64 = 30816;
const VIEW_UNSCALED_VIEWPORT: u64 = 30840;
/// `m_projectionMatrix` element [11] (engine memory is the transpose of the CB copy):
/// the near plane of the reversed-Z infinite projection.
const VIEW_PROJ_NEAR: u64 = 29584 + 11 * 4;
const VIEW_FOV_Y: u64 = 14960;
/// `m_previousSubPixelJitter` (copied from the current value in `idRenderView_Setup`).
const VIEW_PREV_JITTER: u64 = 30808;
/// `m_viewMatrix` / `m_previousViewMatrix` (camera motion test) and
/// `m_inverseViewProjectionMatrix` / `m_previousViewProjectionMatrix` (sky motion
/// vectors); 64-byte `idRenderMatrix` each (d2enhanced NOTES, reflection corroborated).
const VIEW_VIEW: u64 = 29776;
const VIEW_PREV_VIEW: u64 = 29904;
const VIEW_INV_VP: u64 = 30032;
const VIEW_PREV_VP: u64 = 30096;
const IMAGE_RESOURCE: u64 = 0x130;

/// Engine -> FFX conventions (d2enhanced, verified there with the debug view and
/// measured image shifts): jitter in render pixels = (-x, +y) * render size, motion
/// vectors are (current - previous) in viewport UV.
const JITTER_SIGN: [f32; 2] = [-1.0, 1.0];
const MOTION_SIGN: f32 = -1.0;
const FRAMES_IN_FLIGHT: usize = 3;
const FENCE_TIMEOUT_MS: u32 = 2000;
/// FSR dispatches longer than this are logged one by one (white-flash diagnostics).
const GPU_SPIKE_MS: f32 = 20.0;

// ---------------------------------------------------------------------- state --

/// User toggle (F6); FSR runs only when this is set and the runtime is ready.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// FSR debug view, index into [`DEBUG_VIEWS`].
static DEBUG_MODE: AtomicU8 = AtomicU8::new(0);
/// Display size the ready runtime was built for (w << 32 | h), 0 = not ready.
static READY_SIZE: AtomicU64 = AtomicU64::new(0);
static INIT_BUSY: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicBool = AtomicBool::new(false);
/// Calls of the TAA site (to tell the user when anti-aliasing is not TAA).
static TAA_CALLS: AtomicU64 = AtomicU64::new(0);
static TAA_SEEN: AtomicU64 = AtomicU64::new(0);
/// `fsr_scale::POSTPROC_CALLS` at the previous tick.
static POSTPROC_SEEN: AtomicU64 = AtomicU64::new(0);
/// Consecutive ticks in which the world was post-processed (POSTPROC ran) without the
/// TAA site: menus and loading screens run neither and do not count.
static TAA_IDLE_FRAMES: AtomicU32 = AtomicU32::new(0);
static TAA_HINT_LOGGED: AtomicBool = AtomicBool::new(false);
static CAPTURE_ERRORS: AtomicU32 = AtomicU32::new(0);
/// Sky motion-vector convention: 0 = not decided yet, 1..=6 = `SKY_VARIANTS` index + 1,
/// `u32::MAX` = no convention reproduced the engine's vectors (sky vectors stay 0).
/// Survives runtime re-creation (resize).
static SKY_VARIANT: AtomicU32 = AtomicU32::new(0);
/// Record-time CPU cost of `capture` (ns, count), reported with the FSR stats.
static CAPTURE_NS: AtomicU64 = AtomicU64::new(0);
static CAPTURE_N: AtomicU64 = AtomicU64::new(0);

/// Both call sites are redirected (`install` succeeded).
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// FSR GPU time per dispatch, exponential moving average (f32 bits; 0 = none yet).
static GPU_MS_AVG: AtomicU32 = AtomicU32::new(0);

static RUNTIME: Mutex<Option<Box<Runtime>>> = Mutex::new(None);
static JOBS: Mutex<Vec<(usize, Job)>> = Mutex::new(Vec::new());
/// FFX loader from the game directory (set before any FSR initialisation runs).
static LOADER: LazyLock<Result<ffx::Loader, String>> = LazyLock::new(|| {
    crate::game_dir()
        .ok_or_else(|| "no game directory".to_owned())
        .and_then(|d| ffx::Loader::load(d))
});

pub(crate) fn base() -> u64 {
    mem::image_base().unwrap_or(0x1_4000_0000)
}

fn size_key(w: u32, h: u32) -> u64 {
    (u64::from(w) << 32) | u64::from(h)
}

/// FSR is on, has not failed and has a runtime.
pub(crate) fn running() -> bool {
    ACTIVE.load(Ordering::Relaxed)
        && !FAILED.load(Ordering::Relaxed)
        && READY_SIZE.load(Ordering::Acquire) != 0
}

/// [`running`] with a runtime built for `display`.
pub(crate) fn running_for(display: [u32; 2]) -> bool {
    running() && READY_SIZE.load(Ordering::Acquire) == size_key(display[0], display[1])
}

// ------------------------------------------------------------------------ jobs --

/// TAA inputs of one POSTPROC recording, captured at record time.
struct Job {
    frame: u32,
    color: ID3D11Resource,
    motion: ID3D11Resource,
    depth: ID3D11Resource,
    output: ID3D11Resource,
    exposure: Option<ID3D11Buffer>,
    exposure_index: u32,
    ev_bias: f32,
    view_white: f32,
    compressed_hdr: bool,
    render: [u32; 2],
    display: [u32; 2],
    jitter: [f32; 2],
    near: f32,
    fov_y: f32,
    /// `m_justTeleported` was set for this view's frame (camera cut).
    cut: bool,
    /// OIT revealage targets (`OpacityBuffer`, `hairOpacity`) for the masks.
    opacity: Option<ID3D11Resource>,
    hair: Option<ID3D11Resource>,
    /// Upscale with the post chain switched: `distoVectors` to stretch to full size.
    disto: Option<ID3D11Resource>,
    /// Camera reprojection for the sky motion vectors (raw engine memory, 16 floats each;
    /// the shader tests both matrix conventions, see `SKY_VARIANTS`).
    camera: Camera,
    /// Our reflections for this frame (`ssr.rs`): run on the immediate context right before
    /// FSR packs `color`.
    ssr: Option<crate::ssr::Job>,
}

#[derive(Clone, Copy, Default)]
struct Camera {
    inv_vp: [f32; 16],
    prev_vp: [f32; 16],
    /// `2 * (previous - current)` sub-pixel jitter in NDC, as the engine's
    /// `jitterRelativeToPreviousFrame`.
    jitter_rel: [f32; 2],
    /// The view matrix changed since the previous frame (rotation or translation).
    moving: bool,
    /// The matrices were readable and finite (otherwise no sky vectors this frame).
    valid: bool,
}

fn read_camera(view: u64, jitter: [f32; 2]) -> Option<Camera> {
    let inv_vp = mem::f32s::<16>(view + VIEW_INV_VP)?;
    let prev_vp = mem::f32s::<16>(view + VIEW_PREV_VP)?;
    let cur = mem::f32s::<16>(view + VIEW_VIEW)?;
    let prev = mem::f32s::<16>(view + VIEW_PREV_VIEW)?;
    let prev_jitter = mem::f32s::<2>(view + VIEW_PREV_JITTER)?;
    let finite = inv_vp.iter().chain(&prev_vp).all(|v| v.is_finite());
    finite.then(|| Camera {
        inv_vp,
        prev_vp,
        jitter_rel: [
            2.0 * (prev_jitter[0] - jitter[0]),
            2.0 * (prev_jitter[1] - jitter[1]),
        ],
        moving: cur.iter().zip(&prev).any(|(a, b)| (a - b).abs() > 1e-5),
        valid: true,
    })
}

// SAFETY: the D3D11 objects are free-threaded; a job is moved from the recording worker
// to the submitting worker and used there only.
unsafe impl Send for Job {}

/// Parameter-block slot of handle global `h_rva` in engine context `ctx`:
/// `ctx + 0x10 + 16 * (decl.index + 1)`.
pub(crate) fn slot(ctx: u64, h_rva: u64) -> Result<u64, String> {
    let decl = mem::u64_at(base() + h_rva).ok_or("param handle unreadable")?;
    let index = mem::u32_at(decl + 0x48).ok_or("param decl unreadable")? as i32;
    if !(0..4096).contains(&index) {
        return Err(format!("param slot index {index} out of range"));
    }
    Ok(ctx + 0x10 + 16 * (index as u64 + 1))
}

/// The `ID3D11Resource` of an `arkImageView` (`m_Image` -> `ID3D11Resource*` at +0x130).
pub(crate) fn view_resource(view: u64) -> Option<ID3D11Resource> {
    let image = mem::u64_at(view)?;
    let raw = mem::u64_at(image + IMAGE_RESOURCE)? as *mut c_void;
    // SAFETY: `raw` is the engine's live ID3D11Resource pointer for this image (O: read
    // by every view creator); `cloned` takes our own reference.
    unsafe { ID3D11Resource::from_raw_borrowed(&raw) }.cloned()
}

/// The `ID3D11Resource` behind an image parameter (`{arkImageView*, 0}`).
fn image_resource(ctx: u64, h_rva: u64, what: &str) -> Result<ID3D11Resource, String> {
    let view = mem::u64_at(slot(ctx, h_rva)?).ok_or_else(|| format!("{what}: slot"))?;
    view_resource(view).ok_or_else(|| format!("{what}: no image resource"))
}

/// The engine's `motionVectors` of the TAA pass recorded on engine context `ctx` (`None` if
/// the parameter is unreadable).
pub(crate) fn motion_vectors(ctx: u64) -> Option<ID3D11Resource> {
    image_resource(ctx, H_MOTION, "motionVectors").ok()
}

/// The `ID3D11Buffer` behind a buffer parameter (`{0, arkBufferView*}` -> arkBuffer ->
/// `ID3D11Buffer*` at +0).
fn buffer_param(ctx: u64, h_rva: u64) -> Option<ID3D11Buffer> {
    let view = mem::u64_at(slot(ctx, h_rva).ok()? + 8)?;
    let buffer = mem::u64_at(mem::u64_at(view)?)?;
    let raw = buffer as *mut c_void;
    // SAFETY: live engine buffer pointer; `cloned` takes our own reference.
    unsafe { ID3D11Buffer::from_raw_borrowed(&raw) }.cloned()
}

pub(crate) fn rect_size(addr: u64) -> Option<[u32; 2]> {
    let mut b = [0u8; 16];
    if !mem::read_into(addr, &mut b) {
        return None;
    }
    let i = |k: usize| i32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
    let (w, h) = (i(8) - i(0) + 1, i(12) - i(4) + 1);
    (w > 0 && h > 0 && w <= 16384 && h <= 16384).then_some([w as u32, h as u32])
}

fn capture(ctx: u64, pass: u64) -> Result<Job, String> {
    let view = mem::u64_at(pass + PASS_VIEW).ok_or("pass view")?;
    let render = rect_size(view + VIEW_VIEWPORT).ok_or("viewport")?;
    let display = rect_size(view + VIEW_UNSCALED_VIEWPORT).ok_or("unscaled viewport")?;
    let jitter = mem::f32s::<2>(view + VIEW_JITTER).ok_or("jitter")?;
    let near = mem::f32s::<1>(view + VIEW_PROJ_NEAR).ok_or("near")?[0];
    let fov_y = mem::f32s::<1>(view + VIEW_FOV_Y).ok_or("fov")?[0];
    let frame = mem::u32_at(base() + RS_RVA + RS_FRAME).ok_or("frame counter")?;
    let cut = mem::u32_at(view + VIEW_FRAME).is_some_and(crate::fsr_scale::teleported);
    let exposure_index = mem::u32_at(slot(ctx, H_EXPOSURE_INDEX)? + 4).unwrap_or(0);
    let ev_bias = mem::f32s::<1>(slot(ctx, H_EV_BIAS)?).map_or(0.0, |v| v[0]);
    let view_white = mem::f32s::<1>(slot(ctx, H_WHITE_LEVEL)?).map_or(1.0, |v| v[0]);
    let compressed_hdr = mem::u32_at(base() + HDR_MODE_RVA).is_some_and(|v| v != 1);
    let (opacity, hair) = if settings::get().fsr_masks {
        crate::fsr_scale::oit_targets()
    } else {
        (None, None)
    };
    Ok(Job {
        frame,
        color: image_resource(ctx, H_VIEW_COLOR, "viewColorMap")?,
        motion: image_resource(ctx, H_MOTION, "motionVectors")?,
        depth: image_resource(ctx, H_DEPTH, "fx/depthFull")?,
        output: image_resource(ctx, H_TAA_RESULT, "taaResult")?,
        exposure: buffer_param(ctx, H_EXPOSURE_BUFFER),
        exposure_index: exposure_index.min(1 << 16),
        ev_bias: if ev_bias.is_finite() { ev_bias } else { 0.0 },
        view_white: if view_white.is_finite() && view_white > 0.0 {
            view_white
        } else {
            1.0
        },
        compressed_hdr,
        render,
        display,
        jitter,
        near: if near > 0.0 && near.is_finite() {
            near
        } else {
            0.01
        },
        fov_y: if fov_y > 0.01 && fov_y < 3.1 {
            fov_y
        } else {
            1.0
        },
        cut,
        opacity,
        hair,
        disto: None,
        ssr: None,
        camera: read_camera(view, jitter).unwrap_or_default(),
    })
}

// ----------------------------------------------------------------- record hook --

type CtxDispatchFn = unsafe extern "system" fn(*mut c_void, u32, u32, u32);

fn original_dispatch(ctx: *mut c_void, x: u32, y: u32) {
    // SAFETY: `Ctx_Dispatch` of the verified executable with its own signature; the
    // arguments are the ones the engine passed (z is always 1 at this site).
    unsafe {
        let f = std::mem::transmute::<usize, CtxDispatchFn>((base() + CTX_DISPATCH_RVA) as usize);
        f(ctx, x, y, 1);
    }
}

/// Replacement for the TAA `Ctx_Dispatch` call (stub: r9 = r14 = pass). Our reflections
/// (`ssr.rs`) are captured first: FSR takes them along to its submit-time work; with the
/// engine TAA they are recorded into this deferred context right before the dispatch.
unsafe extern "system" fn taa_dispatch(ctx: *mut c_void, x: u32, y: u32, pass: *mut c_void) {
    TAA_CALLS.fetch_add(1, Ordering::Relaxed);
    let taken = std::panic::catch_unwind(|| {
        let mut ssr = crate::ssr::capture(pass as u64, motion_vectors(ctx as u64));
        let taken = record(ctx as u64, pass as u64, &mut ssr);
        if let Some(job) = ssr {
            crate::ssr::record_engine(ctx as u64, job);
        }
        taken
    })
    .unwrap_or(false);
    if !taken {
        original_dispatch(ctx, x, y);
    }
}

fn log_capture_error(e: &str) {
    if CAPTURE_ERRORS.fetch_add(1, Ordering::Relaxed) < 5 {
        log::line(&format!("fsr: {e}; engine TAA used"));
    }
}

/// Takes the frame for FSR; false = run the engine TAA. Deferred contexts queue a job for
/// the submit hook; a context wrapping the immediate one (scheduler without command
/// lists, or a pass recorded at submit) runs the job inline. A taken frame also takes
/// `ssr` (run before FSR); otherwise it stays for the engine TAA.
fn record(ctx: u64, pass: u64, ssr: &mut Option<crate::ssr::Job>) -> bool {
    if !ACTIVE.load(Ordering::Relaxed) || FAILED.load(Ordering::Relaxed) {
        return false;
    }
    let d3d = mem::u64_at(ctx).unwrap_or(0);
    if d3d == 0 {
        return false;
    }
    let inline = Some(d3d) == mem::u64_at(base() + IMMEDIATE_CTX_RVA);
    let display =
        match mem::u64_at(pass + PASS_VIEW).and_then(|v| rect_size(v + VIEW_UNSCALED_VIEWPORT)) {
            Some(d) => d,
            None => return false,
        };
    if READY_SIZE.load(Ordering::Acquire) != size_key(display[0], display[1]) {
        request_init(d3d, display);
        return false;
    }
    // A scaled frame needs the post-chain switch; without it the engine TAA (which
    // upscales by itself) keeps the image correct.
    let switch = match crate::fsr_scale::prepare_switch(pass) {
        Ok(s) => s,
        Err(e) => {
            log_capture_error(&format!("upscale switch not possible ({e})"));
            return false;
        }
    };
    let t_capture = Instant::now();
    let captured = capture(ctx, pass);
    CAPTURE_NS.fetch_add(t_capture.elapsed().as_nanos() as u64, Ordering::Relaxed);
    CAPTURE_N.fetch_add(1, Ordering::Relaxed);
    let mut job = match captured {
        Ok(job) => job,
        Err(e) => {
            log_capture_error(&format!("input capture failed ({e})"));
            return false;
        }
    };
    if switch.is_some() {
        job.disto = crate::fsr_scale::disto_vectors(pass);
    }
    job.ssr = ssr.take();
    if inline {
        // SAFETY: the engine context's live immediate ID3D11DeviceContext.
        let raw = d3d as *mut c_void;
        let Some(imm) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&raw) }).cloned() else {
            return false;
        };
        let ok = execute_job(&imm, job);
        invalidate_engine_caches(ctx);
        if !ok {
            // The fallback copy left the render-size input in taaResult: keep the post
            // chain at render scale.
            return true;
        }
    } else {
        let mut jobs = JOBS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A job still queued for this pass was never submitted: that frame's TAA was
        // skipped and nothing wrote `taaResult` (white-flash diagnostics: logged each time).
        if let Some(i) = jobs.iter().position(|(p, _)| *p == pass as usize) {
            let old = jobs.swap_remove(i).1;
            static DROPPED: AtomicU64 = AtomicU64::new(0);
            let n = DROPPED.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 20 || n.is_multiple_of(100) {
                log::line(&format!(
                    "fsr: frame {} job dropped: its pass was recorded again (frame {}) before the submit hook ran it; taaResult not written for that frame (#{n})",
                    old.frame, job.frame
                ));
            }
        }
        jobs.push((pass as usize, job));
    }
    if let Some(sw) = switch
        && !crate::fsr_scale::apply_switch(ctx, &sw)
    {
        log_capture_error("post-chain switch failed after FSR took the frame");
    }
    true
}

/// After D3D11 work on the immediate context in the middle of an engine recording: the
/// engine skips `*SSetShader` / `IASetInputLayout` when its cached pointer matches, so the
/// cached pointers are cleared.
fn invalidate_engine_caches(engine_ctx: u64) {
    let _ = mem::write_bytes(engine_ctx + ENGINE_CTX_SHADER_CACHE, &[0u8; 32]);
}

// ----------------------------------------------------------------- submit hook --

type EclFn = unsafe extern "system" fn(*mut c_void, *mut c_void, i32);

/// Replacement for the engine's `ExecuteCommandList` call (stub: r9 = rbx = pass); runs
/// inside the engine's immediate-context lock.
unsafe extern "system" fn submit(
    imm: *mut c_void,
    list: *mut c_void,
    restore: i32,
    pass: *mut c_void,
) {
    let _ = std::panic::catch_unwind(|| before_execute(imm, pass as u64));
    // SAFETY: `imm` is the immediate ID3D11DeviceContext the engine was calling; slot
    // 58 (0x1D0 / 8) is ExecuteCommandList, called with the engine's own arguments.
    unsafe {
        let vtbl = *(imm as *const *const usize);
        let f = std::mem::transmute::<usize, EclFn>(*vtbl.add(0x1D0 / 8));
        f(imm, list, restore);
    }
}

fn before_execute(imm: *mut c_void, pass: u64) {
    let callback = mem::u64_at(pass + PASS_CALLBACK);
    let tonemap = callback == Some(base() + POST_TONEMAP_CALLBACK_RVA);
    if !tonemap && callback != Some(base() + POSTPROC_CALLBACK_RVA) {
        return;
    }
    // SAFETY: the engine's live immediate context.
    let Some(ctx) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&imm) }).cloned() else {
        return;
    };
    if tonemap {
        let job = DEPTH_JOBS.lock().ok().and_then(|mut j| {
            j.iter()
                .position(|(p, _)| *p == pass as usize)
                .map(|i| j.swap_remove(i).1)
        });
        if let Some(job) = job {
            run_depth(&ctx, &job);
        }
        return;
    }
    let job = {
        let mut jobs = JOBS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(i) = jobs.iter().position(|(p, _)| *p == pass as usize) else {
            return;
        };
        jobs.swap_remove(i).1
    };
    execute_job(&ctx, job);
}

/// The FSR work of one frame on the immediate context (submit hook or inline at the TAA
/// site), preceded by the frame's reflections (they change `color` before FSR reads it).
/// On failure: input copy into `taaResult`, FSR disabled. Returns success.
fn execute_job(ctx: &ID3D11DeviceContext, mut job: Job) -> bool {
    begin_check(ctx, &job);
    if let Some(ssr) = job.ssr.take() {
        crate::ssr::execute(ctx, ssr);
    }
    let mut rt = RUNTIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = match rt.as_mut() {
        Some(r) if r.display == job.display => r.run(ctx, &job),
        _ => Err("runtime not ready for this size".into()),
    };
    match result {
        Ok(()) => true,
        Err(e) => {
            fallback_copy(ctx, &job);
            log::line(&format!(
                "fsr: frame {} failed: {e}; FSR disabled",
                job.frame
            ));
            crate::notify::error(format!("FSR failed ({e}); engine TAA restored"));
            FAILED.store(true, Ordering::Relaxed);
            set_active(false);
            false
        }
    }
}

/// Starts the frame's value check (`fsr_check.rs`) and scans the colour before our
/// reflections change it (stage 0, only when SSR runs this frame). Only with
/// `diagnostics = 1`; otherwise the check stays closed and every later scan is a no-op.
fn begin_check(ctx: &ID3D11DeviceContext, job: &Job) {
    if !settings::get().diagnostics {
        return;
    }
    let mut rt = RUNTIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(r) = rt.as_mut().filter(|r| r.display == job.display) else {
        return;
    };
    let before = if job.ssr.is_some() {
        match r.view(&job.color, false) {
            Ok(View::Srv(v)) => Some(v),
            _ => None,
        }
    } else {
        None
    };
    if let Some(c) = r.check.as_mut()
        && c.begin(ctx)
        && let Some(v) = &before
    {
        c.scan(ctx, v, job.render, 0);
    }
}

/// Depth resample before the post-tonemap surface draws (upscale mode): the render-scaled
/// depth buffer is stretched to display size in place (`Runtime::resample_depth`).
pub(crate) struct DepthJob {
    pub target: ID3D11Resource,
    pub render: [u32; 2],
    pub display: [u32; 2],
}
// SAFETY: D3D11 resources are free-threaded; moved from the recording to the submitting
// worker and used there only.
unsafe impl Send for DepthJob {}

static DEPTH_JOBS: Mutex<Vec<(usize, DepthJob)>> = Mutex::new(Vec::new());

/// Called by the POST_TONEMAP hook at record time: inline on the immediate context, or
/// queued for the pass's submission.
pub(crate) fn depth_resample(engine_ctx: u64, pass: u64, job: DepthJob) {
    let Some(d3d) = mem::u64_at(engine_ctx).filter(|&d| d != 0) else {
        return;
    };
    if Some(d3d) == mem::u64_at(base() + IMMEDIATE_CTX_RVA) {
        let raw = d3d as *mut c_void;
        // SAFETY: the engine context's live immediate ID3D11DeviceContext.
        if let Some(imm) = unsafe { ID3D11DeviceContext::from_raw_borrowed(&raw) }.cloned() {
            run_depth(&imm, &job);
            invalidate_engine_caches(engine_ctx);
        }
    } else if let Ok(mut j) = DEPTH_JOBS.lock() {
        j.retain(|(p, _)| *p != pass as usize);
        j.push((pass as usize, job));
    }
}

fn run_depth(ctx: &ID3D11DeviceContext, job: &DepthJob) {
    static FAILED_ONCE: AtomicBool = AtomicBool::new(false);
    let mut rt = RUNTIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(r) = rt.as_mut() else {
        return;
    };
    if let Err(e) = r.resample_depth(ctx, &job.target, job.render, job.display)
        && !FAILED_ONCE.swap(true, Ordering::Relaxed)
    {
        log::line(&format!(
            "fsr: depth resample for the post-tonemap draws failed: {e} (in-world UI depth test uses render-scaled depth)"
        ));
    }
}

/// The skipped TAA must still leave an image in `taaResult`: copy the input.
fn fallback_copy(ctx: &ID3D11DeviceContext, job: &Job) {
    let b = D3D11_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: job.render[0],
        bottom: job.render[1],
        back: 1,
    };
    // SAFETY: both are live engine textures of the same format family (hdrBuffer /
    // hdrBuffer2); the box lies inside both.
    unsafe { ctx.CopySubresourceRegion(&job.output, 0, 0, 0, 0, &job.color, 0, Some(&b)) };
}

// ------------------------------------------------------------------- controls --

/// F6: FSR on/off (when `fsr = 1`). Off re-arms the engine TAA history reset.
pub fn toggle() {
    let on = !ACTIVE.load(Ordering::Relaxed);
    if on {
        FAILED.store(false, Ordering::Relaxed);
    }
    set_active(on);
    log::line(&format!(
        "fsr: {}",
        if on { "on" } else { "off (engine TAA)" }
    ));
    crate::notify::info(if on {
        "FSR 4 on"
    } else {
        "FSR off (engine TAA)"
    });
}

fn set_active(on: bool) {
    let was = ACTIVE.swap(on, Ordering::Relaxed);
    if was && !on {
        // Both consumers in the TAA block decrement: >= 2 makes the next engine TAA frame
        // clear its stale history and take the current frame fully.
        let _ = mem::write_u32(base() + RS_RVA + RS_TAA_RESET, 2);
    }
}

/// Runtime numbers for the overlay (`status`).
#[derive(Clone, Debug, Default)]
pub struct RuntimeStatus {
    pub provider: String,
    pub display: [u32; 2],
    pub render: [u32; 2],
    pub frames: u64,
    pub resets: u64,
    pub cuts: u64,
}

/// State for the overlay (read-only; no behaviour change).
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// Call sites redirected (at startup with `fsr = 1`, or when `fsr` was turned on).
    pub installed: bool,
    /// F6 state.
    pub active: bool,
    /// Index into [`DEBUG_VIEWS`].
    pub debug_view: u8,
    pub failed: bool,
    /// None while the runtime is busy (submission holds it) or not created yet.
    pub runtime: Option<RuntimeStatus>,
    /// Rolling FSR GPU time per dispatch (ms).
    pub gpu_ms: Option<f32>,
    /// Sky motion-vector decision.
    pub sky: String,
    /// Live mip bias: (bias in the engine's sampler cells, biased samplers, tracked).
    pub mip_bias: (f32, usize, usize),
}

pub fn status() -> Status {
    let runtime = RUNTIME.try_lock().ok().and_then(|g| {
        g.as_ref().map(|r| RuntimeStatus {
            provider: r.upscaler.provider().version.clone(),
            display: r.display,
            render: r.last_render,
            frames: r.frames,
            resets: r.resets,
            cuts: r.cuts,
        })
    });
    let gpu = f32::from_bits(GPU_MS_AVG.load(Ordering::Relaxed));
    let sky = match SKY_VARIANT.load(Ordering::Relaxed) {
        0 => "undecided".to_owned(),
        u32::MAX => "off (no convention matched)".to_owned(),
        n => SKY_VARIANTS
            .get(n as usize - 1)
            .map_or_else(|| format!("variant {n}"), |v| (*v).to_owned()),
    };
    Status {
        installed: INSTALLED.load(Ordering::Relaxed),
        active: ACTIVE.load(Ordering::Relaxed),
        debug_view: DEBUG_MODE.load(Ordering::Relaxed),
        failed: FAILED.load(Ordering::Relaxed),
        runtime,
        gpu_ms: (gpu > 0.0).then_some(gpu),
        sky,
        mip_bias: crate::fsr_scale::mip_bias_status(),
    }
}

/// FSR debug views (Shift+F6 cycles): FFX's own view (drawn by FSR into its output, linear
/// scales) and our motion-vector view (`fsr_motion.rs`, log scale, drawn at present).
pub const DEBUG_VIEWS: [&str; 3] = ["off", "FFX", "motion vectors"];
pub const DEBUG_FFX: u8 = 1;
pub const DEBUG_MOTION: u8 = 2;

/// Shift+F6: next FSR debug view.
pub fn toggle_debug_view() {
    let next = (DEBUG_MODE.load(Ordering::Relaxed) + 1) % DEBUG_VIEWS.len() as u8;
    set_debug_view(next);
}

pub fn set_debug_view(view: u8) {
    let view = view.min(DEBUG_VIEWS.len() as u8 - 1);
    DEBUG_MODE.store(view, Ordering::Relaxed);
    log::line(&format!("fsr: debug view {}", DEBUG_VIEWS[view as usize]));
}

pub fn debug_view() -> u8 {
    DEBUG_MODE.load(Ordering::Relaxed)
}

/// Main thread before the engine present, immediate context under the engine lock
/// (`overlay.rs`, debug view "motion vectors"): FSR's last packed motion vectors.
pub fn draw_debug(
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    size: [u32; 2],
) -> Result<(), String> {
    let Ok(g) = RUNTIME.try_lock() else {
        return Ok(());
    };
    let Some(r) = g.as_ref() else {
        return Ok(());
    };
    let Some(view) = r.motion_view.as_ref() else {
        return Ok(());
    };
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(|e| format!("device: {e}"))?;
    let same = |a: &windows::core::IUnknown, b: &windows::core::IUnknown| a.as_raw() == b.as_raw();
    if !same(
        &dev.cast().map_err(|e| e.to_string())?,
        &r.d11.cast().map_err(|e| e.to_string())?,
    ) {
        return Ok(());
    }
    view.draw(ctx, &r.gfx_state, &r.raster, rtv, r.last_render, size)
}

/// Main thread, every frame (called whether or not FSR is installed): applies the live
/// FSR settings (`fsr` on = install at run time if needed, off = deactivate; mip bias,
/// mode toasts and auto-exposure context changes) and hints when the TAA pass does not
/// run (anti-aliasing not TAA).
pub fn tick() {
    let s = settings::get();
    apply_enabled(s.fsr);
    if !INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    crate::fsr_scale::frame();
    // `fsr_auto_exposure` is fixed in the FFX context: a change re-creates it (as a resize
    // does: the next TAA frame requests the init, the engine TAA runs meanwhile).
    if READY_SIZE.load(Ordering::Acquire) != 0
        && !INIT_BUSY.load(Ordering::Acquire)
        && let Ok(rt) = RUNTIME.try_lock()
        && let Some(r) = rt.as_ref()
        && r.auto_exposure != s.fsr_auto_exposure
    {
        READY_SIZE.store(0, Ordering::Release);
        log::line(&format!(
            "fsr: fsr_auto_exposure = {}: re-creating the FSR context",
            s.fsr_auto_exposure
        ));
        crate::notify::info(format!(
            "FSR auto exposure {}: context re-created",
            if s.fsr_auto_exposure { "on" } else { "off" }
        ));
    }
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let calls = TAA_CALLS.load(Ordering::Relaxed);
    let posts = crate::fsr_scale::POSTPROC_CALLS.load(Ordering::Relaxed);
    let world = POSTPROC_SEEN.swap(posts, Ordering::Relaxed) != posts;
    if TAA_SEEN.swap(calls, Ordering::Relaxed) != calls {
        TAA_IDLE_FRAMES.store(0, Ordering::Relaxed);
        return;
    }
    // The menu can post-process a scene without TAA, so the engine's preset decides
    // (`r_AntiAliasingPreset`: 3 = Temporal AA; read once, at the 600th idle frame).
    if world
        && TAA_IDLE_FRAMES.fetch_add(1, Ordering::Relaxed) == 600
        && let Some(aa) = crate::cvars::info("r_AntiAliasingPreset")
        && aa.int != 3
        && !TAA_HINT_LOGGED.swap(true, Ordering::Relaxed)
    {
        log::line(&format!(
            "fsr: no engine TAA pass for 600 rendered frames, r_AntiAliasingPreset = {} (3 = Temporal AA)",
            aa.int
        ));
        crate::notify::warn("FSR idle: set Anti-aliasing to TAA");
    }
}

/// `fsr` as last seen by [`tick`]: 0 = not seen yet, 1 = off, 2 = on.
static FSR_SETTING: AtomicU32 = AtomicU32::new(0);
/// A run-time installation was attempted (it is not retried after a failure).
static INSTALL_TRIED: AtomicBool = AtomicBool::new(false);

/// Applies a change of the `fsr` setting: on installs FSR now if needed (call sites
/// patched with threads suspended, `mem::patch_code`; MinHook hooks) and activates it;
/// off deactivates it like F6 (everything stays installed, inert).
fn apply_enabled(on: bool) {
    let now = if on { 2 } else { 1 };
    let was = FSR_SETTING.swap(now, Ordering::Relaxed);
    if was == now {
        return;
    }
    let first = was == 0;
    if on && !INSTALLED.load(Ordering::Relaxed) {
        // `first`: fsr = 1 at startup and the startup installation failed (logged there).
        // A failed run-time installation is not retried either.
        if first || INSTALL_TRIED.swap(true, Ordering::Relaxed) {
            return;
        }
        match install() {
            Ok(()) => {
                log::line("fsr: installed at run time (fsr = 1)");
                crate::notify::info("FSR 4 installed and on");
            }
            Err(e) => {
                log::line(&format!(
                    "fsr: run-time installation failed: {e}; engine TAA used"
                ));
                crate::notify::error(format!("FSR could not be installed: {e}"));
            }
        }
        return;
    }
    if first || !INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    if on {
        FAILED.store(false, Ordering::Relaxed);
        set_active(true);
        log::line("fsr: on (fsr = 1)");
        crate::notify::info("FSR 4 on");
    } else {
        set_active(false);
        log::line("fsr: off (fsr = 0; hooks stay installed, inert; engine TAA)");
        crate::notify::info("FSR off (engine TAA; hooks stay installed, inert)");
    }
}

/// FSR is installed (the overlay enables F6 and the FSR controls).
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// Display size of the ready FSR runtime.
pub(crate) fn ready_display() -> Option<[u32; 2]> {
    let k = READY_SIZE.load(Ordering::Acquire);
    (k != 0).then_some([(k >> 32) as u32, k as u32])
}

// ---------------------------------------------------------------- installation --

/// The TAA call site is redirected (kept across a failed submit-site patch); serialised by
/// `TAA_SITE_LOCK` (FSR and SSR both install it).
static TAA_SITE: AtomicBool = AtomicBool::new(false);
static TAA_SITE_LOCK: Mutex<()> = Mutex::new(());

/// Redirects the TAA call site only (`taa_dispatch`; the stub passes the pass object as a
/// fourth argument, r9). Used by FSR and by the reflections (`ssr.rs`, which need the site
/// also without FSR: with FSR not installed `record` declines and the engine TAA runs).
pub(crate) fn install_taa_site() -> Result<(), String> {
    let _guard = TAA_SITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if TAA_SITE.load(Ordering::Relaxed) {
        return Ok(());
    }
    mem::redirect_call(
        base() + TAA_CALL_RVA,
        &TAA_CALL_BYTES,
        r14,
        taa_dispatch as *const () as u64,
    )
    .map_err(|e| format!("TAA call site: {e}"))?;
    TAA_SITE.store(true, Ordering::Relaxed);
    Ok(())
}

/// Installs FSR: every `fsr_scale` hook (inert until FSR runs), then both call sites;
/// FSR starts active. Called at startup (`fsr = 1`) or from [`tick`] when `fsr` is
/// turned on while the game runs: the call-site patches suspend the other threads and
/// wait until none executes the patched bytes (`mem::patch_code`), MinHook does the same
/// for its hooks.
pub fn install() -> Result<(), String> {
    if INSTALLED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let b = base();
    crate::fsr_scale::install();
    install_taa_site()?;
    mem::redirect_call(
        b + SUBMIT_CALL_RVA,
        &SUBMIT_CALL_BYTES,
        rbx,
        submit as *const () as u64,
    )
    .map_err(|e| format!("submit call site: {e}"))?;
    FAILED.store(false, Ordering::Relaxed);
    ACTIVE.store(true, Ordering::Relaxed);
    INSTALLED.store(true, Ordering::Relaxed);
    FSR_SETTING.store(2, Ordering::Relaxed);
    log::line("fsr: TAA and submit call sites redirected; F6 toggles FSR, Shift+F6 the debug view");
    Ok(())
}

// ------------------------------------------------------------------- runtime --

struct SendDevice(ID3D11Device);
// SAFETY: D3D11 devices are free-threaded.
unsafe impl Send for SendDevice {}

fn request_init(d3d_ctx: u64, display: [u32; 2]) {
    if FAILED.load(Ordering::Relaxed) || INIT_BUSY.swap(true, Ordering::AcqRel) {
        return;
    }
    let raw = d3d_ctx as *mut c_void;
    // SAFETY: the engine context's live ID3D11DeviceContext1.
    let Some(device) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&raw) })
        .and_then(|c| unsafe { c.GetDevice() }.ok())
    else {
        INIT_BUSY.store(false, Ordering::Release);
        return;
    };
    let device = SendDevice(device);
    READY_SIZE.store(0, Ordering::Release);
    let spawned = std::thread::Builder::new()
        .name("void-fsr-init".into())
        .spawn(move || {
            let device = device;
            let t = Instant::now();
            // Drop the old runtime first (waits for its GPU work).
            *RUNTIME
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            let made = std::panic::catch_unwind(|| {
                let loader = LOADER.as_ref().map_err(Clone::clone)?;
                Runtime::new(&device.0, display, loader)
            });
            match made {
                Ok(Ok(rt)) => {
                    log::line(&format!(
                        "fsr: ready, provider {} at {}x{} ({:.0} ms)",
                        rt.upscaler.provider().version,
                        display[0],
                        display[1],
                        t.elapsed().as_secs_f64() * 1000.0
                    ));
                    *RUNTIME
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::new(rt));
                    READY_SIZE.store(size_key(display[0], display[1]), Ordering::Release);
                    crate::notify::info(format!(
                        "FSR {} ready at {}x{}",
                        RUNTIME
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .as_ref()
                            .map_or_else(String::new, |r| r.upscaler.provider().version.clone()),
                        display[0],
                        display[1]
                    ));
                }
                Ok(Err(e)) => {
                    log::line(&format!("fsr: initialisation failed: {e}; engine TAA used"));
                    crate::notify::error(format!("FSR could not start: {e}"));
                    FAILED.store(true, Ordering::Relaxed);
                }
                Err(_) => {
                    log::line("fsr: initialisation panicked; engine TAA used");
                    crate::notify::error("FSR could not start (internal error)");
                    FAILED.store(true, Ordering::Relaxed);
                }
            }
            INIT_BUSY.store(false, Ordering::Release);
        });
    if spawned.is_err() {
        INIT_BUSY.store(false, Ordering::Release);
    }
}

/// A texture created by D3D11 (shared, NT handle) and opened by D3D12.
pub(crate) struct Shared {
    pub(crate) res: ID3D12Resource,
    pub(crate) uav: Option<ID3D11UnorderedAccessView>,
    pub(crate) srv: Option<ID3D11ShaderResourceView>,
}

struct FrameSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Fence value signalled after this slot's last submission (0 = unused).
    fence: u64,
    /// The slot's timestamps are waiting to be read.
    timed: bool,
}

/// Cached views of engine resources, keyed by resource pointer.
enum View {
    Srv(ID3D11ShaderResourceView),
    Uav(ID3D11UnorderedAccessView),
}

impl View {
    fn dup(&self) -> Self {
        match self {
            View::Srv(s) => View::Srv(s.clone()),
            View::Uav(u) => View::Uav(u.clone()),
        }
    }
}

struct Runtime {
    display: [u32; 2],
    d11: ID3D11Device5,
    fence11: ID3D11Fence,
    queue: ID3D12CommandQueue,
    fence12: ID3D12Fence,
    fence_value: u64,
    event: HANDLE,
    slots: Vec<FrameSlot>,
    next: usize,
    timestamps: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ts_freq: u64,
    color: Shared,
    motion: Shared,
    depth: Shared,
    exposure: Shared,
    output: Shared,
    /// Masks (R8_UNORM): reactive, transparency & composition.
    reactive: Shared,
    tc: Shared,
    pack: ID3D11ComputeShader,
    unpack: ID3D11ComputeShader,
    stretch: ID3D11ComputeShader,
    linear: ID3D11SamplerState,
    vs_full: ID3D11VertexShader,
    ps_depth: ID3D11PixelShader,
    depth_always: ID3D11DepthStencilState,
    raster: ID3D11RasterizerState,
    gfx_state: ID3DDeviceContextState,
    /// Scratch textures keyed by (width, height, format): stretch target / depth copy.
    scratch: HashMap<(u32, u32, i32, bool), (ID3D11Texture2D, View)>,
    params: ID3D11Buffer,
    views: HashMap<(usize, bool), (View, u32)>,
    upscaler: ffx::Upscaler,
    last_frame: Option<u32>,
    last_render: [u32; 2],
    last_dispatch: Option<Instant>,
    frames: u64,
    resets: u64,
    /// Resets caused by `m_justTeleported`.
    cuts: u64,
    /// FSR dispatches over [`GPU_SPIKE_MS`].
    spikes: u64,
    gpu_ms: Vec<f32>,
    /// CPU ms per FSR frame on the submitting thread (inside the engine's immediate
    /// lock, so on the frame's critical path): pack + D3D11 signal + flush, D3D12
    /// record + submit, D3D11 wait + unpack + resamples, total.
    cpu_ms: Vec<[f32; 4]>,
    logged_inputs: bool,
    /// Depth-stencil views of engine depth targets (depth resample).
    dsvs: HashMap<usize, ID3D11DepthStencilView>,
    /// Depth + stencil resample (`SV_StencilRef`; None when the device lacks
    /// `PSSpecifiedStencilRefSupported` or the shader did not compile).
    ps_depth_stencil: Option<ID3D11PixelShader>,
    stencil_replace: ID3D11DepthStencilState,
    /// Stencil-plane SRVs of depth copies, keyed by texture pointer.
    stencil_srvs: HashMap<usize, ID3D11ShaderResourceView>,
    /// Depth-bindable scratch targets for engine depth textures without a DSV binding.
    ds_scratch: HashMap<(u32, u32, i32), (ID3D11Texture2D, ID3D11DepthStencilView)>,
    sky: Sky,
    /// NaN / Inf / huge-value check of the FSR path (None if it could not be created).
    check: Option<crate::fsr_check::ValueCheck>,
    /// Engine motion-vector coverage per depth band on static-camera frames.
    motion_stats: Option<crate::fsr_motion::MotionStats>,
    /// The "motion vectors" debug view.
    motion_view: Option<crate::fsr_motion::MotionView>,
    /// Provider phase count for the last (render, display) widths.
    jitter_auto: Option<((u32, u32), u32)>,
    /// `fsr_auto_exposure` the context was created with (a change re-creates it).
    auto_exposure: bool,
}

// SAFETY: D3D11/D3D12 devices and queues are free-threaded; the runtime is used under
// `RUNTIME`'s mutex by one thread at a time (the engine also serialises submission).
unsafe impl Send for Runtime {}

impl Drop for Runtime {
    fn drop(&mut self) {
        // The upscaler and shared resources must outlive their GPU work.
        let _ = self.wait(self.fence_value);
        // SAFETY: our own event handle.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

const PACK_HLSL: &str = r"
struct AutoExposure
{
    float EngineLuminanceFactor;
    float LuminanceFactor;
    float MinLuminanceLDR;
    float MaxLuminanceLDR;
    float MiddleGreyLuminanceLDR;
    float EV;
    float Fstop;
    uint  PeakHistogramValue;
};
Texture2D<float4> g_color : register(t0);
Texture2D<float2> g_motion : register(t1);
Texture2D<float> g_depth : register(t2);
StructuredBuffer<AutoExposure> g_exposure : register(t3);
RWTexture2D<float4> o_color : register(u0);
RWTexture2D<float2> o_motion : register(u1);
RWTexture2D<float> o_depth : register(u2);
RWTexture2D<float> o_exposure : register(u3);
cbuffer Params : register(b0)
{
    uint2 renderSize;
    uint exposureIndex;
    uint compressedHdr;
    float viewWhite;
    float evBias;
    uint hasExposure;
    uint flags;
    uint2 area;
    uint2 last;
    float reactiveScale;
    float pad1;
    uint2 srcSize;
    uint2 dstSize;
    float2 depthScale;
};
Texture2D<float4> g_opacity : register(t4);
Texture2D<float4> g_hair : register(t5);
RWTexture2D<float> o_reactive : register(u4);
RWTexture2D<float> o_tc : register(u5);
// Sky motion vectors. The engine's matrices are passed as raw memory; which multiplication
// order and jitter term reproduce its own vectors is measured on geometry pixels (stats:
// inlier counts per variant at [0..5], samples at [6]) and then used where the engine
// leaves 0 (sky: reversed-Z depth 0, vector 0). Variant v: conv = v / 3 (0: mul(M, x),
// 1: mul(x, M)), k = v % 3 (jitter term 0, +jitterRel, -jitterRel).
cbuffer Sky : register(b1)
{
    row_major float4x4 invVP;
    row_major float4x4 prevVP;
    float2 jitterRel;
    uint skyVariant;  // 0 = off, else variant + 1
    uint skyValidate;
};
RWByteAddressBuffer o_stats : register(u6);
float2 PrevNdc(float2 ndc, float d, uint conv)
{
    float4 x = float4(ndc, d, 1.0);
    float4 wp = conv == 0 ? mul(invVP, x) : mul(x, invVP);
    float4 p = conv == 0 ? mul(prevVP, wp) : mul(wp, prevVP);
    return abs(p.w) > 1e-20 ? p.xy / p.w : float2(1e9, 1e9);
}
float2 VariantMv(float2 ndc, float2 prevNdc, uint k)
{
    float2 j = k == 0 ? float2(0.0, 0.0) : (k == 1 ? jitterRel : -jitterRel);
    return (ndc - prevNdc - j) * float2(0.5, -0.5);
}
// White-flash guard (docs/fsr4-integration.md, Value check): non-finite channels become 0,
// the rest is clamped to [0, 65000] (below the FP16 and R11G11B10 maxima), so neither FSR's
// history nor the engine's bloom / exposure after it can be poisoned by NaN or Inf.
float3 Sanitize(float3 c)
{
    bool3 bad = (asuint(c) & 0x7f800000) == 0x7f800000;
    return bad ? 0.0 : clamp(c, 0.0, 65000.0);
}
// flags: 1 = masks, 2 = hair revealage present, 4 = repair the last column/row (bright
// edge at reduced render scale, d2enhanced NOTES Bright edge).
[numthreads(8, 8, 1)]
void pack(uint2 id : SV_DispatchThreadID)
{
    if (id.x == 0 && id.y == 0)
    {
        float e = 1.0;
        if (hasExposure != 0)
        {
            float elf = g_exposure[exposureIndex].EngineLuminanceFactor;
            e = exp(-evBias) * elf;
            if (compressedHdr != 0)
                e *= elf * viewWhite;
            if (!(e > 1e-8 && e < 1e8))
                e = 1.0;
        }
        o_exposure[uint2(0, 0)] = e;
    }
    if (id.x >= renderSize.x || id.y >= renderSize.y)
        return;
    uint2 src = (flags & 4) != 0 ? min(id, max(renderSize, 2) - 2) : id;
    o_color[id] = float4(Sanitize(g_color[src].rgb), 1.0);
    float d = g_depth[id];
    float2 e = g_motion[id];
    float2 ndc = (float2(id) + 0.5) / float2(renderSize) * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float2 mv = e;
    if (skyVariant != 0 && d <= 0.0 && e.x == 0.0 && e.y == 0.0)
    {
        uint v = skyVariant - 1;
        float2 s = VariantMv(ndc, PrevNdc(ndc, 0.0, v / 3), v % 3);
        if (all(abs(s) < 4.0))
            mv = s;
    }
    o_motion[id] = mv;
    o_depth[id] = d;
    if (skyValidate != 0 && d > 0.0 && (id.x & 15) == 0 && (id.y & 15) == 0)
    {
        uint ignored;
        o_stats.InterlockedAdd(24, 1, ignored);
        [unroll] for (uint conv = 0; conv < 2; ++conv)
        {
            float2 pn = PrevNdc(ndc, d, conv);
            [unroll] for (uint k = 0; k < 3; ++k)
            {
                float err = length((VariantMv(ndc, pn, k) - e) * float2(renderSize));
                if (err < 0.5)
                    o_stats.InterlockedAdd((conv * 3 + k) * 4, 1, ignored);
            }
        }
    }
    if ((flags & 1) != 0)
    {
        float rev = g_opacity[id].x;
        if ((flags & 2) != 0)
            rev *= g_hair[id].x;
        float tc = saturate(1.0 - rev);
        o_tc[id] = tc;
        o_reactive[id] = saturate(tc * reactiveScale);
    }
}
Texture2D<float4> g_out : register(t0);
RWTexture2D<float4> o_result : register(u0);
[numthreads(8, 8, 1)]
void unpack(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= area.x || id.y >= area.y)
        return;
    o_result[id] = float4(Sanitize(g_out[min(id, last)].rgb), 1.0);
}
Texture2D<float4> g_src : register(t0);
SamplerState s_linear : register(s0);
RWTexture2D<float4> o_dst : register(u0);
// Stretches the render-scaled top-left region (srcSize pixels) of a texture of dstSize
// pixels over the whole texture.
[numthreads(8, 8, 1)]
void stretch(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= dstSize.x || id.y >= dstSize.y)
        return;
    float2 p = (float2(id) + 0.5) * float2(srcSize) / float2(dstSize);
    p = clamp(p, 0.5, float2(srcSize) - 0.5);
    o_dst[id] = g_src.SampleLevel(s_linear, p / float2(dstSize), 0);
}
float4 vs_full(uint i : SV_VertexID) : SV_Position
{
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
Texture2D<float> g_zsrc : register(t0);
Texture2D<uint2> g_ssrc : register(t1);
float ps_depth(float4 pos : SV_Position) : SV_Depth
{
    uint2 p = min(uint2(pos.xy * depthScale), srcSize - 1);
    return g_zsrc.Load(int3(p, 0));
}
struct DepthStencil
{
    float depth : SV_Depth;
    uint stencil : SV_StencilRef;
};
// Depth and stencil (stencil plane read through an X24_G8 / X32_G8X24 view, written with
// SV_StencilRef + stencil op REPLACE).
DepthStencil ps_depth_stencil(float4 pos : SV_Position)
{
    uint2 p = min(uint2(pos.xy * depthScale), srcSize - 1);
    DepthStencil o;
    o.depth = g_zsrc.Load(int3(p, 0));
    o.stencil = g_ssrc.Load(int3(p, 0)).y;
    return o;
}
";

const FLAG_MASKS: u32 = 1;
const FLAG_HAIR: u32 = 2;
const FLAG_REPAIR_EDGE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    render: [u32; 2],
    exposure_index: u32,
    compressed_hdr: u32,
    view_white: f32,
    ev_bias: f32,
    has_exposure: u32,
    flags: u32,
    area: [u32; 2],
    last: [u32; 2],
    reactive_scale: f32,
    pad1: f32,
    src_size: [u32; 2],
    dst_size: [u32; 2],
    depth_scale: [f32; 2],
}

/// `cbuffer Sky` (b1).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SkyParams {
    inv_vp: [f32; 16],
    prev_vp: [f32; 16],
    jitter_rel: [f32; 2],
    variant: u32,
    validate: u32,
}

/// Human-readable names of the sky motion-vector variants (shader `SKY_VARIANTS`).
const SKY_VARIANTS: [&str; 6] = [
    "mul(M, x), no jitter term",
    "mul(M, x), - jitterRel",
    "mul(M, x), + jitterRel",
    "mul(x, M), no jitter term",
    "mul(x, M), - jitterRel",
    "mul(x, M), + jitterRel",
];
/// Decision thresholds: enough moving frames and samples, a clear winner over the other
/// multiplication order, and a give-up limit.
const SKY_MIN_FRAMES: u32 = 30;
const SKY_MIN_SAMPLES: u64 = 2000;
const SKY_MIN_INLIERS: f64 = 0.5;
const SKY_MARGIN: f64 = 1.5;
const SKY_GIVE_UP_FRAMES: u32 = 900;

/// Picks the convention from accumulated statistics (`counts[0..6]` inliers per variant,
/// `counts[6]` samples, over `frames` moving frames): `Some(variant + 1)` when the best
/// variant reproduces at least half of the engine's geometry vectors and beats the best
/// variant of the other multiplication order by `SKY_MARGIN`; `Some(u32::MAX)` after
/// `SKY_GIVE_UP_FRAMES` without a decision; `None` = keep measuring.
fn sky_decision(counts: &[u64; 7], frames: u32) -> Option<u32> {
    if frames < SKY_MIN_FRAMES || counts[6] < SKY_MIN_SAMPLES {
        return (frames >= SKY_GIVE_UP_FRAMES).then_some(u32::MAX);
    }
    let ratio = |v: usize| counts[v] as f64 / counts[6] as f64;
    let best = (0..6).max_by(|&a, &b| ratio(a).total_cmp(&ratio(b)))?;
    let other = (0..6)
        .filter(|v| v / 3 != best / 3)
        .map(ratio)
        .fold(0.0, f64::max);
    if ratio(best) >= SKY_MIN_INLIERS && ratio(best) >= SKY_MARGIN * other {
        Some(best as u32 + 1)
    } else {
        (frames >= SKY_GIVE_UP_FRAMES).then_some(u32::MAX)
    }
}

/// Sky motion-vector resources: constants, the statistics buffer and its read-back ring.
struct Sky {
    params: ID3D11Buffer,
    stats: ID3D11Buffer,
    stats_uav: ID3D11UnorderedAccessView,
    staging: Vec<ID3D11Buffer>,
    /// Staging indices with a copy in flight, oldest first.
    pending: VecDeque<usize>,
    next: usize,
    counts: [u64; 7],
    frames: u32,
}

impl Sky {
    fn new(d11: &ID3D11Device5) -> Result<Self, String> {
        let buffer = |desc: D3D11_BUFFER_DESC| -> Result<ID3D11Buffer, String> {
            let mut b = None;
            // SAFETY: valid descriptor / out pointer.
            unsafe { d11.CreateBuffer(&desc, None, Some(&mut b)) }
                .map_err(|e| format!("sky buffer: {e}"))?;
            b.ok_or_else(|| "no sky buffer".to_owned())
        };
        let params = buffer(D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<SkyParams>() as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            ..Default::default()
        })?;
        let stats = buffer(D3D11_BUFFER_DESC {
            ByteWidth: 32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
            MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
            ..Default::default()
        })?;
        let mut uav = None;
        // SAFETY: raw view of the whole 8-word buffer.
        unsafe {
            d11.CreateUnorderedAccessView(
                &stats,
                Some(&D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D11_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: 8,
                            Flags: D3D11_BUFFER_UAV_FLAG_RAW.0 as u32,
                        },
                    },
                }),
                Some(&mut uav),
            )
        }
        .map_err(|e| format!("sky stats UAV: {e}"))?;
        let staging = (0..4)
            .map(|_| {
                buffer(D3D11_BUFFER_DESC {
                    ByteWidth: 32,
                    Usage: D3D11_USAGE_STAGING,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            params,
            stats,
            stats_uav: uav.ok_or("no sky stats UAV")?,
            staging,
            pending: VecDeque::new(),
            next: 0,
            counts: [0; 7],
            frames: 0,
        })
    }

    /// Reads finished statistics copies (never waits) and decides when possible.
    fn poll(&mut self, ctx: &ID3D11DeviceContext) {
        while let Some(&i) = self.pending.front() {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            // SAFETY: our staging buffer; DO_NOT_WAIT returns an error while in flight;
            // the mapped range is the buffer's 32 bytes.
            let words = unsafe {
                if ctx
                    .Map(
                        &self.staging[i],
                        0,
                        D3D11_MAP_READ,
                        D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                        Some(&mut m),
                    )
                    .is_err()
                {
                    break;
                }
                let w = std::slice::from_raw_parts(m.pData.cast::<u32>(), 7).to_vec();
                ctx.Unmap(&self.staging[i], 0);
                w
            };
            self.pending.pop_front();
            for (c, w) in self.counts.iter_mut().zip(&words) {
                *c += u64::from(*w);
            }
            self.frames += 1;
            if SKY_VARIANT.load(Ordering::Relaxed) != 0 {
                continue;
            }
            let Some(d) = sky_decision(&self.counts, self.frames) else {
                continue;
            };
            SKY_VARIANT.store(d, Ordering::Relaxed);
            let ratios: Vec<String> = (0..6)
                .map(|v| {
                    format!(
                        "{:.2}",
                        self.counts[v] as f64 / self.counts[6].max(1) as f64
                    )
                })
                .collect();
            log::line(&if d == u32::MAX {
                format!(
                    "fsr: sky motion vectors off: no matrix convention reproduces the engine's vectors (inlier ratios {ratios:?} over {} samples, {} moving frames)",
                    self.counts[6], self.frames
                )
            } else {
                format!(
                    "fsr: sky motion vectors on: {} (inlier ratios {ratios:?} over {} samples, {} moving frames)",
                    SKY_VARIANTS[d as usize - 1],
                    self.counts[6],
                    self.frames
                )
            });
            if d == u32::MAX {
                crate::notify::warn("FSR sky motion vectors off (no matching convention)");
            } else {
                crate::notify::info("FSR sky motion vectors on");
            }
        }
    }
}

fn compile_shader(entry: PCSTR, target: PCSTR) -> Result<Vec<u8>, String> {
    crate::hlsl::compile(PACK_HLSL, entry, target)
}

fn compile_cs(device: &ID3D11Device5, entry: PCSTR) -> Result<ID3D11ComputeShader, String> {
    let code = compile_shader(entry, s!("cs_5_0"))?;
    let mut cs = None;
    // SAFETY: valid bytecode / out pointer.
    unsafe { device.CreateComputeShader(&code, None, Some(&mut cs)) }
        .map_err(|e| format!("CreateComputeShader: {e}"))?;
    cs.ok_or_else(|| "no compute shader".into())
}

fn typed(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R24G8_TYPELESS => DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
        DXGI_FORMAT_R32_TYPELESS => DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32G8X24_TYPELESS => DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        DXGI_FORMAT_R16_TYPELESS => DXGI_FORMAT_R16_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R16G16_TYPELESS => DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R8_TYPELESS => DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        f => f,
    }
}

impl Runtime {
    fn new(device: &ID3D11Device, display: [u32; 2], loader: &ffx::Loader) -> Result<Self, String> {
        let d11: ID3D11Device5 = device.cast().map_err(|e| format!("ID3D11Device5: {e}"))?;
        // SAFETY (whole block): plain DXGI / D3D12 object creation with valid descriptors and
        // out pointers on live devices; handles are closed after opening.
        unsafe {
            let dxgi: IDXGIDevice = device.cast().map_err(|e| e.to_string())?;
            let luid = dxgi
                .GetAdapter()
                .and_then(|a| a.GetDesc())
                .map_err(|e| format!("adapter: {e}"))?
                .AdapterLuid;
            let factory: IDXGIFactory4 = CreateDXGIFactory1().map_err(|e| e.to_string())?;
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(luid)
                .map_err(|e| format!("EnumAdapterByLuid: {e}"))?;
            let mut d12: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut d12)
                .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
            let d12 = d12.ok_or("no D3D12 device")?;
            let queue: ID3D12CommandQueue = d12
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(|e| format!("queue: {e}"))?;
            let fence12: ID3D12Fence = d12
                .CreateFence(0, D3D12_FENCE_FLAG_SHARED)
                .map_err(|e| format!("fence: {e}"))?;
            let h = d12
                .CreateSharedHandle(&fence12, None, GENERIC_ALL.0, None)
                .map_err(|e| format!("fence handle: {e}"))?;
            let mut fence11: Option<ID3D11Fence> = None;
            let opened = d11.OpenSharedFence(h, &mut fence11);
            let _ = CloseHandle(h);
            opened.map_err(|e| format!("OpenSharedFence: {e}"))?;
            let fence11 = fence11.ok_or("no D3D11 fence")?;
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)
                .map_err(|e| e.to_string())?;

            let mut slots = Vec::new();
            for _ in 0..FRAMES_IN_FLIGHT {
                let allocator: ID3D12CommandAllocator = d12
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .map_err(|e| e.to_string())?;
                let list: ID3D12GraphicsCommandList = d12
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .map_err(|e| e.to_string())?;
                list.Close().map_err(|e| e.to_string())?;
                slots.push(FrameSlot {
                    allocator,
                    list,
                    fence: 0,
                    timed: false,
                });
            }
            let mut timestamps: Option<ID3D12QueryHeap> = None;
            d12.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2 * FRAMES_IN_FLIGHT as u32,
                    NodeMask: 0,
                },
                &mut timestamps,
            )
            .map_err(|e| e.to_string())?;
            let timestamps = timestamps.ok_or("no query heap")?;
            let mut readback: Option<ID3D12Resource> = None;
            d12.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 16 * FRAMES_IN_FLIGHT as u64,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
            .map_err(|e| e.to_string())?;
            let readback = readback.ok_or("no readback buffer")?;
            let ts_freq = queue.GetTimestampFrequency().map_err(|e| e.to_string())?;

            let [w, h] = display;
            let color = shared(
                &d11,
                &d12,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                w,
                h,
                true,
                false,
            )?;
            let motion = shared(&d11, &d12, DXGI_FORMAT_R16G16_FLOAT, w, h, true, false)?;
            let depth = shared(&d11, &d12, DXGI_FORMAT_R32_FLOAT, w, h, true, false)?;
            let exposure = shared(&d11, &d12, DXGI_FORMAT_R32_FLOAT, 1, 1, true, false)?;
            let output = shared(
                &d11,
                &d12,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                w,
                h,
                false,
                true,
            )?;

            let reactive = shared(&d11, &d12, DXGI_FORMAT_R8_UNORM, w, h, true, false)?;
            let tc = shared(&d11, &d12, DXGI_FORMAT_R8_UNORM, w, h, true, false)?;

            let pack = compile_cs(&d11, s!("pack"))?;
            let unpack = compile_cs(&d11, s!("unpack"))?;
            let stretch = compile_cs(&d11, s!("stretch"))?;
            let vs_code = compile_shader(s!("vs_full"), s!("vs_5_0"))?;
            let ps_code = compile_shader(s!("ps_depth"), s!("ps_5_0"))?;
            let (mut vs_full, mut ps_depth) = (None, None);
            d11.CreateVertexShader(&vs_code, None, Some(&mut vs_full))
                .map_err(|e| format!("vertex shader: {e}"))?;
            d11.CreatePixelShader(&ps_code, None, Some(&mut ps_depth))
                .map_err(|e| format!("pixel shader: {e}"))?;
            let mut linear = None;
            d11.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                Some(&mut linear),
            )
            .map_err(|e| format!("sampler: {e}"))?;
            let mut depth_always = None;
            d11.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut depth_always),
            )
            .map_err(|e| format!("depth state: {e}"))?;
            // Depth + stencil resample: stencil REPLACE with the shader's SV_StencilRef.
            let replace = D3D11_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D11_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D11_STENCIL_OP_REPLACE,
                StencilPassOp: D3D11_STENCIL_OP_REPLACE,
                StencilFunc: D3D11_COMPARISON_ALWAYS,
            };
            let mut stencil_replace = None;
            d11.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: true.into(),
                    StencilReadMask: 0xFF,
                    StencilWriteMask: 0xFF,
                    FrontFace: replace,
                    BackFace: replace,
                },
                Some(&mut stencil_replace),
            )
            .map_err(|e| format!("stencil state: {e}"))?;
            let mut opts2 = D3D11_FEATURE_DATA_D3D11_OPTIONS2::default();
            let stencil_ref = d11
                .CheckFeatureSupport(
                    D3D11_FEATURE_D3D11_OPTIONS2,
                    (&raw mut opts2).cast(),
                    std::mem::size_of_val(&opts2) as u32,
                )
                .is_ok()
                && opts2.PSSpecifiedStencilRefSupported.as_bool();
            let ps_depth_stencil = if stencil_ref {
                match compile_shader(s!("ps_depth_stencil"), s!("ps_5_0")) {
                    Ok(code) => {
                        let mut ps = None;
                        d11.CreatePixelShader(&code, None, Some(&mut ps))
                            .map_err(|e| format!("stencil pixel shader: {e}"))?;
                        ps
                    }
                    Err(e) => {
                        log::line(&format!(
                            "fsr: depth resample keeps stencil as rendered: SV_StencilRef shader: {e}"
                        ));
                        None
                    }
                }
            } else {
                log::line(
                    "fsr: depth resample keeps stencil as rendered (no PSSpecifiedStencilRefSupported)",
                );
                None
            };
            let mut raster = None;
            d11.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut raster),
            )
            .map_err(|e| format!("rasterizer state: {e}"))?;
            let mut gfx_state = None;
            d11.CreateDeviceContextState(
                0,
                &[d11.GetFeatureLevel()],
                D3D11_SDK_VERSION,
                &ID3D11Device::IID,
                None,
                Some(&mut gfx_state),
            )
            .map_err(|e| format!("context state: {e}"))?;
            let mut params = None;
            d11.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: std::mem::size_of::<Params>().next_multiple_of(16) as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut params),
            )
            .map_err(|e| format!("constant buffer: {e}"))?;
            let params = params.ok_or("no constant buffer")?;

            let s = settings::get();
            let auto_exposure = s.fsr_auto_exposure;
            let upscaler = ffx::Upscaler::new(
                loader,
                &d12,
                &ffx::UpscalerDesc {
                    max_render: display,
                    max_upscale: display,
                    hdr: true,
                    depth_inverted: true,
                    depth_infinite: true,
                    auto_exposure,
                    dynamic_resolution: true,
                    provider: None,
                    debug_checker: false,
                },
            )?;
            let sky = Sky::new(&d11)?;
            let check = crate::fsr_check::ValueCheck::new(&d11, color.uav.as_ref())
                .map_err(|e| log::line(&format!("fsr: value check unavailable: {e}")))
                .ok();
            let motion_stats = crate::fsr_motion::MotionStats::new(&d11)
                .map_err(|e| log::line(&format!("fsr: motion statistics unavailable: {e}")))
                .ok();
            let motion_view = motion
                .uav
                .as_ref()
                .ok_or_else(|| "no motion UAV".to_owned())
                .and_then(|u| crate::fsr_motion::MotionView::new(&d11, u))
                .map_err(|e| log::line(&format!("fsr: motion debug view unavailable: {e}")))
                .ok();
            Ok(Self {
                display,
                d11,
                fence11,
                queue,
                fence12,
                fence_value: 0,
                event,
                slots,
                next: 0,
                timestamps,
                readback,
                ts_freq,
                color,
                motion,
                depth,
                exposure,
                output,
                reactive,
                tc,
                pack,
                unpack,
                stretch,
                linear: linear.ok_or("no sampler")?,
                vs_full: vs_full.ok_or("no vertex shader")?,
                ps_depth: ps_depth.ok_or("no pixel shader")?,
                depth_always: depth_always.ok_or("no depth state")?,
                raster: raster.ok_or("no rasterizer state")?,
                gfx_state: gfx_state.ok_or("no context state")?,
                scratch: HashMap::new(),
                params,
                views: HashMap::new(),
                upscaler,
                last_frame: None,
                last_render: [0, 0],
                last_dispatch: None,
                frames: 0,
                resets: 0,
                cuts: 0,
                spikes: 0,
                gpu_ms: Vec::new(),
                cpu_ms: Vec::new(),
                logged_inputs: false,
                dsvs: HashMap::new(),
                ps_depth_stencil,
                stencil_replace: stencil_replace.ok_or("no stencil state")?,
                stencil_srvs: HashMap::new(),
                ds_scratch: HashMap::new(),
                sky,
                check,
                motion_stats,
                motion_view,
                jitter_auto: None,
                auto_exposure,
            })
        }
    }

    /// CPU wait until the D3D12 fence reached `value` (normally already true).
    fn wait(&self, value: u64) -> Result<(), String> {
        // SAFETY: live fence and our event.
        unsafe {
            if self.fence12.GetCompletedValue() >= value {
                return Ok(());
            }
            self.fence12
                .SetEventOnCompletion(value, self.event)
                .map_err(|e| e.to_string())?;
            let t = Instant::now();
            if windows::Win32::System::Threading::WaitForSingleObject(self.event, FENCE_TIMEOUT_MS)
                != windows::Win32::Foundation::WAIT_OBJECT_0
            {
                return Err(format!(
                    "GPU fence {value} not reached in {FENCE_TIMEOUT_MS} ms"
                ));
            }
            // White-flash diagnostics: the slot's previous FSR work was still running.
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            if ms > GPU_SPIKE_MS {
                log::line(&format!(
                    "fsr: CPU waited {ms:.1} ms for D3D12 fence {value} (FSR frame {})",
                    self.frames
                ));
            }
        }
        Ok(())
    }

    /// A cached SRV (`uav = false`) or UAV of an engine texture.
    fn view(&mut self, res: &ID3D11Resource, uav: bool) -> Result<View, String> {
        let key = (res.as_raw() as usize, uav);
        let frame = self.frames as u32;
        if let Some((v, used)) = self.views.get_mut(&key) {
            *used = frame;
            return Ok(v.dup());
        }
        let tex: ID3D11Texture2D = res.cast().map_err(|e| format!("not a 2D texture: {e}"))?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter / view creation on a live texture with valid descriptors.
        let v = unsafe {
            tex.GetDesc(&mut desc);
            let format = typed(desc.Format);
            if uav {
                let d = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
                    },
                };
                let mut u = None;
                self.d11
                    .CreateUnorderedAccessView(res, Some(&d), Some(&mut u))
                    .map_err(|e| format!("UAV ({format:?}): {e}"))?;
                View::Uav(u.ok_or("no UAV")?)
            } else {
                let d = D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                };
                let mut s = None;
                self.d11
                    .CreateShaderResourceView(res, Some(&d), Some(&mut s))
                    .map_err(|e| format!("SRV ({format:?}): {e}"))?;
                View::Srv(s.ok_or("no SRV")?)
            }
        };
        // Views of textures the engine released are dropped after a while (they keep the
        // texture alive).
        if self.views.len() > 32 {
            self.views
                .retain(|_, (_, used)| frame.wrapping_sub(*used) < 600);
        }
        let copy = v.dup();
        self.views.insert(key, (copy, frame));
        Ok(v)
    }

    fn exposure_srv(&mut self, buf: &ID3D11Buffer) -> Option<ID3D11ShaderResourceView> {
        let key = (buf.as_raw() as usize, false);
        if let Some((View::Srv(s), _)) = self.views.get(&key) {
            return Some(s.clone());
        }
        let mut desc = D3D11_BUFFER_DESC::default();
        // SAFETY: plain getter / view creation on a live buffer.
        unsafe {
            buf.GetDesc(&mut desc);
            if desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32 == 0
                || desc.StructureByteStride != 32
            {
                return None;
            }
            let d = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D11_SRV_DIMENSION_BUFFER,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D11_BUFFER_SRV {
                        Anonymous1: D3D11_BUFFER_SRV_0 { FirstElement: 0 },
                        Anonymous2: D3D11_BUFFER_SRV_1 {
                            NumElements: desc.ByteWidth / desc.StructureByteStride,
                        },
                    },
                },
            };
            let mut s = None;
            self.d11
                .CreateShaderResourceView(buf, Some(&d), Some(&mut s))
                .ok()?;
            let s = s?;
            self.views
                .insert(key, (View::Srv(s.clone()), self.frames as u32));
            Some(s)
        }
    }

    /// Engine SRV for an optional mask input; failures only drop the masks.
    fn optional_srv(&mut self, res: Option<&ID3D11Resource>) -> Option<ID3D11ShaderResourceView> {
        match self.view(res?, false) {
            Ok(View::Srv(s)) => Some(s),
            _ => None,
        }
    }

    /// Caches the provider's jitter sequence for (render, display) widths for the
    /// jitter-table writer (`fsr_scale`).
    fn update_jitter_sequence(&mut self, render_w: u32, display_w: u32) {
        let formula = crate::fsr_scale::phase_count(render_w, display_w);
        // Provider phase count per size pair, cached (the query is per context).
        let auto = match self.jitter_auto {
            Some((key, n)) if key == (render_w, display_w) => n,
            _ => {
                let n = self
                    .upscaler
                    .jitter_phase_count(render_w, display_w)
                    .ok()
                    .filter(|n| (1..=4096).contains(n))
                    .map_or(formula, |n| n as u32);
                self.jitter_auto = Some(((render_w, display_w), n));
                n
            }
        };
        let wanted = settings::get().fsr_jitter_phases.unwrap_or(auto);
        if crate::fsr_scale::jitter_sequence_known(render_w, display_w, wanted as usize) {
            return;
        }
        let queried = (0..wanted as i32)
            .map(|i| self.upscaler.jitter_offset(i, wanted as i32))
            .collect::<Result<Vec<_>, _>>();
        let (offsets, source) = match queried {
            Ok(v) => (v, "provider"),
            Err(e) => {
                log::line(&format!(
                    "fsr: jitter query failed ({e}); FFX 3.x Halton formula used"
                ));
                (
                    (0..wanted)
                        .map(|i| crate::fsr_scale::halton_offset(i, wanted))
                        .collect(),
                    "formula",
                )
            }
        };
        log::line(&format!(
            "fsr: jitter sequence {render_w}->{display_w}: {} phases ({}; AMD default {auto}) from the {source}, first {:?}",
            offsets.len(),
            if wanted == auto {
                "auto"
            } else {
                "fsr_jitter_phases"
            },
            offsets.first()
        ));
        crate::fsr_scale::set_jitter_sequence(render_w, display_w, offsets);
    }

    fn run(&mut self, ctx: &ID3D11DeviceContext, job: &Job) -> Result<(), String> {
        let t0 = Instant::now();
        let (t_pack_out, t_d12);
        let ctx4: ID3D11DeviceContext4 = ctx.cast().map_err(|e| format!("context4: {e}"))?;
        let [rw, rh] = job.render;
        if rw > self.display[0] || rh > self.display[1] || rw < 2 || rh < 2 {
            return Err(format!("render size {rw}x{rh} outside the context"));
        }
        let upscale = job.render != job.display;
        let (View::Srv(color), View::Srv(motion), View::Srv(depth), View::Uav(out)) = (
            self.view(&job.color, false)?,
            self.view(&job.motion, false)?,
            self.view(&job.depth, false)?,
            self.view(&job.output, true)?,
        ) else {
            return Err("view kinds".into());
        };
        let exposure = job.exposure.as_ref().and_then(|b| self.exposure_srv(b));
        let opacity = self.optional_srv(job.opacity.as_ref());
        let hair = self.optional_srv(job.hair.as_ref());
        let masks = opacity.is_some();
        if let Some(c) = &self.check {
            c.scan(ctx, &color, job.render, 1);
        }
        if let Some(m) = self
            .motion_stats
            .as_mut()
            .filter(|_| settings::get().diagnostics)
        {
            m.frame(
                ctx,
                &motion,
                &depth,
                job.render,
                job.near,
                job.camera.valid && !job.camera.moving,
            );
        }
        self.update_jitter_sequence(rw, job.display[0]);
        if !self.logged_inputs {
            self.logged_inputs = true;
            log::line(&format!(
                "fsr: first frame {}: render {rw}x{rh}, display {}x{} ({}), jitter {:?}, near {}, fov_y {:.4}, exposure buffer {} index {}, evBias {}, viewWhite {}, compressedHdr {}, masks {} (hair {}), distoVectors {}",
                job.frame,
                job.display[0],
                job.display[1],
                if upscale { "upscale" } else { "native" },
                job.jitter,
                job.near,
                job.fov_y,
                if exposure.is_some() {
                    "ok"
                } else {
                    "MISSING (1.0 used)"
                },
                job.exposure_index,
                job.ev_bias,
                job.view_white,
                job.compressed_hdr,
                masks,
                hair.is_some(),
                job.disto.is_some()
            ));
        }
        // Output: native = render size, written over whole 16-pixel tiles as the engine
        // TAA does; upscale = the whole display-size target.
        let (area, out_size) = if upscale {
            (job.display, job.display)
        } else {
            (
                [
                    rw.next_multiple_of(16).min(self.display[0]),
                    rh.next_multiple_of(16).min(self.display[1]),
                ],
                job.render,
            )
        };
        let mut flags = 0;
        if masks {
            flags |= FLAG_MASKS;
            if hair.is_some() {
                flags |= FLAG_HAIR;
            }
        }
        if upscale {
            flags |= FLAG_REPAIR_EDGE;
        }
        let p = Params {
            render: job.render,
            exposure_index: job.exposure_index,
            compressed_hdr: u32::from(job.compressed_hdr),
            view_white: job.view_white,
            ev_bias: job.ev_bias,
            has_exposure: u32::from(exposure.is_some()),
            flags,
            area,
            last: [out_size[0] - 1, out_size[1] - 1],
            reactive_scale: settings::get().fsr_reactive,
            ..Default::default()
        };
        // Sky motion vectors: measure the convention while the camera moves, then apply.
        self.sky.poll(ctx);
        let sky_on = settings::get().fsr_sky_motion && job.camera.valid;
        let decided = SKY_VARIANT.load(Ordering::Relaxed);
        let validate = sky_on
            && decided == 0
            && job.camera.moving
            && !job.cut
            && self.sky.pending.len() < self.sky.staging.len();
        let sp = SkyParams {
            inv_vp: job.camera.inv_vp,
            prev_vp: job.camera.prev_vp,
            jitter_rel: job.camera.jitter_rel,
            variant: if sky_on && (1..=6).contains(&decided) {
                decided
            } else {
                0
            },
            validate: u32::from(validate),
        };

        // SAFETY (whole block): D3D11 calls on the engine's immediate context (inside its
        // lock, or inline in its recording) with live views; every binding we make is
        // cleared before returning.
        unsafe {
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.UpdateSubresource(&self.sky.params, 0, None, (&raw const sp).cast(), 0, 0);
            if validate {
                ctx.ClearUnorderedAccessViewUint(&self.sky.stats_uav, &[0; 4]);
            }
            ctx.CSSetShader(&self.pack, None);
            ctx.CSSetConstantBuffers(
                0,
                Some(&[Some(self.params.clone()), Some(self.sky.params.clone())]),
            );
            ctx.CSSetShaderResources(
                0,
                Some(&[
                    Some(color),
                    Some(motion),
                    Some(depth),
                    exposure.clone(),
                    opacity,
                    hair,
                ]),
            );
            let uavs = [
                self.color.uav.clone(),
                self.motion.uav.clone(),
                self.depth.uav.clone(),
                self.exposure.uav.clone(),
                self.reactive.uav.clone(),
                self.tc.uav.clone(),
                Some(self.sky.stats_uav.clone()),
            ];
            ctx.CSSetUnorderedAccessViews(0, 7, Some(uavs.as_ptr()), None);
            ctx.Dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
            let none7: [Option<ID3D11UnorderedAccessView>; 7] = Default::default();
            ctx.CSSetUnorderedAccessViews(0, 7, Some(none7.as_ptr()), None);
            ctx.CSSetShaderResources(0, Some(&[None, None, None, None, None, None]));
            ctx.CSSetConstantBuffers(0, Some(&[None, None]));
            if validate {
                let i = self.sky.next;
                self.sky.next = (i + 1) % self.sky.staging.len();
                ctx.CopyResource(&self.sky.staging[i], &self.sky.stats);
                self.sky.pending.push_back(i);
            }

            self.fence_value += 1;
            let inputs_ready = self.fence_value;
            ctx4.Signal(&self.fence11, inputs_ready)
                .map_err(|e| format!("D3D11 Signal: {e}"))?;
            if settings::get().fsr_flush {
                ctx.Flush();
            }
            let t_pack = t0.elapsed();

            let done = self.dispatch_fsr(job, inputs_ready, masks)?;
            t_d12 = t0.elapsed() - t_pack;
            t_pack_out = t_pack;

            ctx4.Wait(&self.fence11, done)
                .map_err(|e| format!("D3D11 Wait: {e}"))?;
            if let Some(c) = &self.check {
                if let Some(input) = &c.fsr_input {
                    c.scan(ctx, input, job.render, 2);
                }
                if let Some(o) = &self.output.srv {
                    c.scan(ctx, o, out_size, 3);
                }
            }
            ctx.CSSetShader(&self.unpack, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetShaderResources(0, Some(std::slice::from_ref(&self.output.srv)));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(out)), None);
            ctx.Dispatch(area[0].div_ceil(8), area[1].div_ceil(8), 1);
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.CSSetShaderResources(0, Some(&[None]));
            ctx.CSSetConstantBuffers(0, Some(&[None]));
            ctx.CSSetShader(None, None);
            if let Some(c) = self.check.as_mut() {
                c.end(ctx, job.frame);
            }
        }
        if let Some(d) = &job.disto
            && let Err(e) = self.stretch_to_full(ctx, d, job.render, job.display)
        {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::line(&format!("fsr: distoVectors stretch failed: {e}"));
            }
        }
        // Upscale: post-TAA effects read `fx/depthFull` with display-size coordinates; the
        // pack above already took the render-scaled copy FSR needed.
        if upscale
            && settings::get().fsr_depth_resample
            && let Err(e) = self.resample_depth(ctx, &job.depth, job.render, job.display)
        {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::line(&format!(
                    "fsr: fx/depthFull resample failed: {e} (post effects read render-scaled depth)"
                ));
            }
        }
        let total = t0.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        self.cpu_ms.push([
            ms(t_pack_out),
            ms(t_d12),
            ms(total.saturating_sub(t_pack_out + t_d12)),
            ms(total),
        ]);
        Ok(())
    }

    /// A scratch texture (SRV, plus UAV if `uav`) with the given size and format.
    fn scratch(
        &mut self,
        w: u32,
        h: u32,
        format: DXGI_FORMAT,
        uav: bool,
    ) -> Result<(ID3D11Texture2D, View), String> {
        let key = (w, h, format.0, uav);
        if let Some((t, v)) = self.scratch.get(&key) {
            return Ok((t.clone(), v.dup()));
        }
        let mut bind = D3D11_BIND_SHADER_RESOURCE.0;
        if uav {
            bind |= D3D11_BIND_UNORDERED_ACCESS.0;
        }
        // SAFETY: plain texture / view creation with valid descriptors and out pointers.
        let (tex, view) = unsafe {
            let mut t = None;
            self.d11
                .CreateTexture2D(
                    &D3D11_TEXTURE2D_DESC {
                        Width: w,
                        Height: h,
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: format,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: bind as u32,
                        ..Default::default()
                    },
                    None,
                    Some(&mut t),
                )
                .map_err(|e| format!("scratch {format:?}: {e}"))?;
            let t: ID3D11Texture2D = t.ok_or("no scratch texture")?;
            let typed = typed(format);
            let v = if uav {
                let mut u = None;
                let d = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: typed,
                    ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_UAV { MipSlice: 0 },
                    },
                };
                self.d11
                    .CreateUnorderedAccessView(&t, Some(&d), Some(&mut u))
                    .map_err(|e| format!("scratch UAV {typed:?}: {e}"))?;
                View::Uav(u.ok_or("no UAV")?)
            } else {
                let mut s = None;
                let d = D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: typed,
                    ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                };
                self.d11
                    .CreateShaderResourceView(&t, Some(&d), Some(&mut s))
                    .map_err(|e| format!("scratch SRV {typed:?}: {e}"))?;
                View::Srv(s.ok_or("no SRV")?)
            };
            (t, v)
        };
        self.scratch.insert(key, (tex.clone(), view.dup()));
        Ok((tex, view))
    }

    /// Stretches the render-scaled top-left region of engine texture `res` (e.g. the
    /// half-size `distoVectors`) over the whole texture: CS into a scratch copy, then
    /// `CopyResource` back. Compute only, no graphics state is touched.
    fn stretch_to_full(
        &mut self,
        ctx: &ID3D11DeviceContext,
        res: &ID3D11Resource,
        render: [u32; 2],
        display: [u32; 2],
    ) -> Result<(), String> {
        let tex: ID3D11Texture2D = res.cast().map_err(|e| format!("not a 2D texture: {e}"))?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter on a live texture.
        unsafe { tex.GetDesc(&mut desc) };
        let src = [
            crate::fsr_scale::scaled_extent(desc.Width, render[0] as f32 / display[0] as f32)
                .min(desc.Width),
            crate::fsr_scale::scaled_extent(desc.Height, render[1] as f32 / display[1] as f32)
                .min(desc.Height),
        ];
        let View::Srv(srv) = self.view(res, false)? else {
            return Err("view kind".into());
        };
        let (scratch, View::Uav(uav)) = self.scratch(desc.Width, desc.Height, desc.Format, true)?
        else {
            return Err("scratch view kind".into());
        };
        let p = Params {
            src_size: src,
            dst_size: [desc.Width, desc.Height],
            ..Default::default()
        };
        // SAFETY: compute work on live views; bindings cleared afterwards; the copy is
        // between textures of identical description.
        unsafe {
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.CSSetShader(&self.stretch, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetShaderResources(0, Some(&[Some(srv)]));
            ctx.CSSetSamplers(0, Some(&[Some(self.linear.clone())]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(uav)), None);
            ctx.Dispatch(desc.Width.div_ceil(8), desc.Height.div_ceil(8), 1);
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.CSSetShaderResources(0, Some(&[None]));
            ctx.CSSetSamplers(0, Some(&[None]));
            ctx.CSSetConstantBuffers(0, Some(&[None]));
            ctx.CSSetShader(None, None);
            ctx.CopyResource(res, &scratch);
        }
        Ok(())
    }

    /// Stretches the render-scaled depth (top-left `render` region) of `target` over the
    /// whole display-size texture, in place: copy, then a full-screen `SV_Depth` draw
    /// (depth ALWAYS) under a swapped-in device context state, so the engine's pipeline
    /// state is exactly restored. Stencil is resampled the same way with `SV_StencilRef`
    /// where supported, else left as rendered. Targets without a depth-stencil binding
    /// (e.g. `decompressedDepth`) are drawn in a depth-bindable scratch copy and copied back.
    fn resample_depth(
        &mut self,
        ctx: &ID3D11DeviceContext,
        target: &ID3D11Resource,
        render: [u32; 2],
        display: [u32; 2],
    ) -> Result<(), String> {
        let tex: ID3D11Texture2D = target
            .cast()
            .map_err(|e| format!("not a 2D texture: {e}"))?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter on a live texture.
        unsafe { tex.GetDesc(&mut desc) };
        let (dsv_format, stencil_format) = match desc.Format {
            DXGI_FORMAT_R24G8_TYPELESS => (
                DXGI_FORMAT_D24_UNORM_S8_UINT,
                Some(DXGI_FORMAT_X24_TYPELESS_G8_UINT),
            ),
            DXGI_FORMAT_R32G8X24_TYPELESS => (
                DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                Some(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT),
            ),
            DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_D32_FLOAT, None),
            DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_D16_UNORM, None),
            f => return Err(format!("depth format {f:?}")),
        };
        let size = [display[0].min(desc.Width), display[1].min(desc.Height)];
        let (copy, View::Srv(srv)) = self.scratch(desc.Width, desc.Height, desc.Format, false)?
        else {
            return Err("scratch view kind".into());
        };
        // Stencil plane of the copy (only with the SV_StencilRef shader).
        let stencil = match (stencil_format, &self.ps_depth_stencil) {
            (Some(f), Some(_)) => Some(self.stencil_srv(&copy, f)?),
            _ => None,
        };
        // Draw target: the texture itself when it is depth-bindable, else a scratch copy.
        let direct = desc.BindFlags & D3D11_BIND_DEPTH_STENCIL.0 as u32 != 0;
        let (draw_tex, dsv): (ID3D11Resource, ID3D11DepthStencilView) = if direct {
            let key = target.as_raw() as usize;
            let dsv = match self.dsvs.get(&key) {
                Some(d) => d.clone(),
                None => {
                    let d = self.create_dsv(target, dsv_format)?;
                    if self.dsvs.len() > 8 {
                        self.dsvs.clear();
                    }
                    self.dsvs.insert(key, d.clone());
                    d
                }
            };
            (target.clone(), dsv)
        } else {
            let key = (desc.Width, desc.Height, desc.Format.0);
            let (t, d) = match self.ds_scratch.get(&key) {
                Some(e) => e.clone(),
                None => {
                    // Same layout as the target (CopyResource needs identical shape).
                    let mut d2 = desc;
                    d2.BindFlags =
                        (D3D11_BIND_DEPTH_STENCIL.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32;
                    d2.MiscFlags = 0;
                    d2.CPUAccessFlags = 0;
                    d2.Usage = D3D11_USAGE_DEFAULT;
                    let mut t = None;
                    // SAFETY: valid descriptor / out pointer.
                    unsafe { self.d11.CreateTexture2D(&d2, None, Some(&mut t)) }
                        .map_err(|e| format!("depth scratch: {e}"))?;
                    let t: ID3D11Texture2D = t.ok_or("no depth scratch")?;
                    let r: ID3D11Resource = t.cast().map_err(|e| e.to_string())?;
                    let d = self.create_dsv(&r, dsv_format)?;
                    self.ds_scratch.insert(key, (t.clone(), d.clone()));
                    (t, d)
                }
            };
            (t.cast().map_err(|e| e.to_string())?, d)
        };
        let ctx1: ID3D11DeviceContext1 = ctx.cast().map_err(|e| format!("context1: {e}"))?;
        let p = Params {
            src_size: render,
            depth_scale: [
                render[0] as f32 / display[0] as f32,
                render[1] as f32 / display[1] as f32,
            ],
            ..Default::default()
        };
        // SAFETY (whole block): the copies are between textures of identical description;
        // the draw runs under our own device context state, swapped back afterwards
        // (restores every binding of the engine's state); our bindings are cleared before
        // the swap.
        unsafe {
            ctx.CopyResource(&copy, target);
            if !direct {
                // Keeps the stencil as rendered when the shader cannot write it.
                ctx.CopyResource(&draw_tex, target);
            }
            let mut prev = None;
            ctx1.SwapDeviceContextState(&self.gfx_state, Some(&mut prev));
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            ctx.IASetInputLayout(None);
            ctx.VSSetShader(&self.vs_full, None);
            match (&stencil, &self.ps_depth_stencil) {
                (Some(_), Some(ps)) => {
                    ctx.PSSetShader(ps, None);
                    ctx.OMSetDepthStencilState(&self.stencil_replace, 0);
                }
                _ => {
                    ctx.PSSetShader(&self.ps_depth, None);
                    ctx.OMSetDepthStencilState(&self.depth_always, 0);
                }
            }
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.PSSetShaderResources(0, Some(&[Some(srv), stencil]));
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size[0] as f32,
                Height: size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(None, None, u32::MAX);
            ctx.OMSetRenderTargets(None, &dsv);
            ctx.Draw(3, 0);
            ctx.OMSetRenderTargets(None, None);
            ctx.PSSetShaderResources(0, Some(&[None, None]));
            ctx.PSSetConstantBuffers(0, Some(&[None]));
            if let Some(p) = prev {
                ctx1.SwapDeviceContextState(&p, None);
            }
            if !direct {
                ctx.CopyResource(target, &draw_tex);
            }
        }
        Ok(())
    }

    fn create_dsv(
        &self,
        res: &ID3D11Resource,
        format: DXGI_FORMAT,
    ) -> Result<ID3D11DepthStencilView, String> {
        let mut d = None;
        let dd = D3D11_DEPTH_STENCIL_VIEW_DESC {
            Format: format,
            ViewDimension: D3D11_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // SAFETY: view creation on a live depth-bindable texture.
        unsafe {
            self.d11
                .CreateDepthStencilView(res, Some(&dd), Some(&mut d))
        }
        .map_err(|e| format!("DSV {format:?}: {e}"))?;
        d.ok_or_else(|| "no DSV".into())
    }

    /// Stencil-plane SRV of a depth copy (`X24_TYPELESS_G8_UINT` / `X32_TYPELESS_G8X24_UINT`).
    fn stencil_srv(
        &mut self,
        tex: &ID3D11Texture2D,
        format: DXGI_FORMAT,
    ) -> Result<ID3D11ShaderResourceView, String> {
        let key = tex.as_raw() as usize;
        if let Some(s) = self.stencil_srvs.get(&key) {
            return Ok(s.clone());
        }
        let d = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                },
            },
        };
        let mut s = None;
        // SAFETY: view creation on our live scratch texture.
        unsafe {
            self.d11
                .CreateShaderResourceView(tex, Some(&d), Some(&mut s))
        }
        .map_err(|e| format!("stencil SRV {format:?}: {e}"))?;
        let s = s.ok_or("no stencil SRV")?;
        self.stencil_srvs.insert(key, s.clone());
        Ok(s)
    }

    /// Records and submits the FSR dispatch after `inputs_ready`; returns the fence value
    /// signalled when it is done.
    fn dispatch_fsr(&mut self, job: &Job, inputs_ready: u64, masks: bool) -> Result<u64, String> {
        let i = self.next;
        self.next = (self.next + 1) % self.slots.len();
        self.wait(self.slots[i].fence)?;
        if self.slots[i].timed {
            self.read_timestamps(i);
            self.slots[i].timed = false;
        }
        let reset = self
            .last_frame
            .is_none_or(|f| job.frame != f.wrapping_add(1))
            || self.last_render != job.render
            || job.cut;
        self.last_frame = Some(job.frame);
        self.last_render = job.render;
        self.resets += u64::from(reset);
        self.cuts += u64::from(job.cut);
        let now = Instant::now();
        let frame_ms = self
            .last_dispatch
            .map_or(16.7, |t| (now - t).as_secs_f32() * 1000.0)
            .clamp(0.1, 1000.0);
        self.last_dispatch = Some(now);
        let [rw, rh] = job.render;
        let s = settings::get();
        let sharpening = (s.fsr_sharpness > 0.0).then_some(s.fsr_sharpness.min(1.0));
        let common = D3D12_RESOURCE_STATE_COMMON;
        let res = |r: &ID3D12Resource| -> ID3D12Resource { r.clone() };
        let (color, depth, motion, exposure, output, reactive, tc) = (
            res(&self.color.res),
            res(&self.depth.res),
            res(&self.motion.res),
            res(&self.exposure.res),
            res(&self.output.res),
            res(&self.reactive.res),
            res(&self.tc.res),
        );
        let d = ffx::UpscaleDispatch {
            color: ffx::Res {
                resource: &color,
                state: common,
            },
            depth: ffx::Res {
                resource: &depth,
                state: common,
            },
            motion_vectors: ffx::Res {
                resource: &motion,
                state: common,
            },
            exposure: (!self.auto_exposure).then_some(ffx::Res {
                resource: &exposure,
                state: common,
            }),
            reactive: masks.then_some(ffx::Res {
                resource: &reactive,
                state: common,
            }),
            transparency_composition: masks.then_some(ffx::Res {
                resource: &tc,
                state: common,
            }),
            output: ffx::Res {
                resource: &output,
                state: common,
            },
            jitter: [
                JITTER_SIGN[0] * job.jitter[0] * rw as f32,
                JITTER_SIGN[1] * job.jitter[1] * rh as f32,
            ],
            motion_vector_scale: [MOTION_SIGN * rw as f32, MOTION_SIGN * rh as f32],
            render_size: job.render,
            upscale_size: job.display,
            sharpening,
            frame_time_delta_ms: frame_ms,
            pre_exposure: 1.0,
            reset,
            camera_near: job.near,
            camera_far: 1.0e6,
            camera_fov_y: job.fov_y,
            view_space_to_meters: 1.0,
            debug_view: DEBUG_MODE.load(Ordering::Relaxed) == DEBUG_FFX,
        };
        let slot = &self.slots[i];
        // SAFETY (whole block): the slot's previous work completed (waited above), so its
        // allocator and list can be reset; every resource recorded stays alive in `self`
        // until a later wait on its fence.
        unsafe {
            slot.allocator.Reset().map_err(|e| e.to_string())?;
            slot.list
                .Reset(&slot.allocator, None)
                .map_err(|e| e.to_string())?;
            // Hybrid ray-traced reflections (`ssr/rt_reflect.rs`): composited into the colour
            // input before FSR reads it; nothing is recorded when off or not ready.
            crate::ssr::rt_reflect::record(&crate::ssr::rt_reflect::FsrFrame {
                list: &slot.list,
                color: &self.color.res,
                depth: &self.depth.res,
                motion: &self.motion.res,
                render: job.render,
                fence: &self.fence12,
                fence_next: self.fence_value + 1,
                ts_freq: self.ts_freq,
            });
            let q0 = 2 * i as u32;
            slot.list
                .EndQuery(&self.timestamps, D3D12_QUERY_TYPE_TIMESTAMP, q0);
            self.upscaler.dispatch(&slot.list, &d)?;
            slot.list
                .EndQuery(&self.timestamps, D3D12_QUERY_TYPE_TIMESTAMP, q0 + 1);
            slot.list.ResolveQueryData(
                &self.timestamps,
                D3D12_QUERY_TYPE_TIMESTAMP,
                q0,
                2,
                &self.readback,
                16 * i as u64,
            );
            slot.list.Close().map_err(|e| format!("Close: {e}"))?;
            self.queue
                .Wait(&self.fence12, inputs_ready)
                .map_err(|e| format!("D3D12 Wait: {e}"))?;
            let list: ID3D12CommandList = slot.list.cast().map_err(|e| e.to_string())?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.fence_value += 1;
            let done = self.fence_value;
            self.queue
                .Signal(&self.fence12, done)
                .map_err(|e| format!("D3D12 Signal: {e}"))?;
            self.slots[i].fence = done;
            self.slots[i].timed = true;
            self.frames += 1;
            if self.frames.is_multiple_of(1200) {
                self.report();
            }
            Ok(done)
        }
    }

    fn read_timestamps(&mut self, i: usize) {
        let mut p: *mut c_void = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 16 * i,
            End: 16 * i + 16,
        };
        // SAFETY: readback buffer written by a completed list; mapped range in bounds.
        unsafe {
            if self.readback.Map(0, Some(&range), Some(&mut p)).is_err() || p.is_null() {
                return;
            }
            let t = std::slice::from_raw_parts(p.cast::<u8>().add(16 * i).cast::<u64>(), 2);
            let (a, b) = (t[0], t[1]);
            self.readback
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            if b > a && self.ts_freq > 0 {
                let ms = ((b - a) as f64 * 1000.0 / self.ts_freq as f64) as f32;
                self.gpu_ms.push(ms);
                // White-flash diagnostics: a single long FSR dispatch (user saw 74 ms once).
                if ms > GPU_SPIKE_MS {
                    self.spikes += 1;
                    if self.spikes <= 20 || self.spikes.is_multiple_of(100) {
                        log::line(&format!(
                            "fsr: GPU spike {ms:.1} ms in the dispatch of FSR frame {} (slot {i}, spike #{})",
                            self.frames, self.spikes
                        ));
                    }
                }
                let avg = f32::from_bits(GPU_MS_AVG.load(Ordering::Relaxed));
                let avg = if avg > 0.0 {
                    avg * 0.95 + ms * 0.05
                } else {
                    ms
                };
                GPU_MS_AVG.store(avg.to_bits(), Ordering::Relaxed);
            }
        }
    }

    fn report(&mut self) {
        let mut v = std::mem::take(&mut self.gpu_ms);
        let cpu = std::mem::take(&mut self.cpu_ms);
        let (ns, n) = (
            CAPTURE_NS.swap(0, Ordering::Relaxed),
            CAPTURE_N.swap(0, Ordering::Relaxed),
        );
        if !settings::get().diagnostics {
            return;
        }
        v.sort_by(f32::total_cmp);
        let q = |v: &[f32], f: f64| {
            v.get(((v.len() as f64 - 1.0).max(0.0) * f).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        log::line(&format!(
            "fsr: {} frames ({} resets, {} camera cuts), {} GPU ms p50 {:.3} p99 {:.3} max {:.3} (n {})",
            self.frames,
            self.resets,
            self.cuts,
            self.upscaler.provider().version,
            q(&v, 0.5),
            q(&v, 0.99),
            v.last().copied().unwrap_or(0.0),
            v.len()
        ));
        let part = |i: usize| {
            let mut c: Vec<f32> = cpu.iter().map(|x| x[i]).collect();
            c.sort_by(f32::total_cmp);
            format!("{:.3}/{:.3}", q(&c, 0.5), q(&c, 0.99))
        };
        log::line(&format!(
            "fsr: CPU ms p50/p99 on the submit path: pack+flush {}, D3D12 submit {}, wait+unpack+resample {}, total {}; record-time capture avg {:.3}",
            part(0),
            part(1),
            part(2),
            part(3),
            ns as f64 / n.max(1) as f64 / 1e6
        ));
        if let Some(c) = self.check.as_mut() {
            log::line(&format!("fsr: {}", c.summary()));
        }
    }
}

/// A D3D11-created shared texture opened by D3D12 (native D3D11 cannot open D3D12-created
/// textures: d2enhanced NOTES "FSR native-AA results").
pub(crate) fn shared(
    d11: &ID3D11Device5,
    d12: &ID3D12Device,
    format: DXGI_FORMAT,
    w: u32,
    h: u32,
    uav: bool,
    srv: bool,
) -> Result<Shared, String> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0) as u32,
    };
    // SAFETY: valid descriptors / out pointers on live devices; the handle is closed after
    // D3D12 opened its own reference.
    unsafe {
        let mut tex = None;
        d11.CreateTexture2D(&desc, None, Some(&mut tex))
            .map_err(|e| format!("shared texture {format:?}: {e}"))?;
        let tex: ID3D11Texture2D = tex.ok_or("no texture")?;
        let r1: IDXGIResource1 = tex.cast().map_err(|e| e.to_string())?;
        let handle = r1
            .CreateSharedHandle(
                None,
                DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                None,
            )
            .map_err(|e| format!("shared handle: {e}"))?;
        let mut res: Option<ID3D12Resource> = None;
        let opened = d12.OpenSharedHandle(handle, &mut res);
        let _ = CloseHandle(handle);
        opened.map_err(|e| format!("OpenSharedHandle: {e}"))?;
        let res = res.ok_or("no D3D12 resource")?;
        let mut u = None;
        if uav {
            d11.CreateUnorderedAccessView(&tex, None, Some(&mut u))
                .map_err(|e| format!("UAV: {e}"))?;
        }
        let mut s = None;
        if srv {
            d11.CreateShaderResourceView(&tex, None, Some(&mut s))
                .map_err(|e| format!("SRV: {e}"))?;
        }
        Ok(Shared {
            res,
            uav: u,
            srv: s,
        })
    }
}

// ---------------------------------------------------------------- offline check --

/// Offline device with engine-like textures for [`offline_check`].
struct Offline {
    device: ID3D11Device,
    ctx: ID3D11DeviceContext,
}

impl Offline {
    fn texture(
        &self,
        size: [u32; 2],
        format: DXGI_FORMAT,
        bind: i32,
        data: Option<(&[u8], u32)>,
    ) -> Result<ID3D11Texture2D, String> {
        let d = D3D11_TEXTURE2D_DESC {
            Width: size[0],
            Height: size[1],
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind as u32,
            ..Default::default()
        };
        let init = data.map(|(b, pitch)| D3D11_SUBRESOURCE_DATA {
            pSysMem: b.as_ptr().cast(),
            SysMemPitch: pitch,
            SysMemSlicePitch: 0,
        });
        let mut t = None;
        // SAFETY: valid descriptor; the initial data covers the declared size.
        unsafe {
            self.device
                .CreateTexture2D(&d, init.as_ref().map(|i| i as *const _), Some(&mut t))
        }
        .map_err(|e| format!("texture {format:?}: {e}"))?;
        t.ok_or_else(|| "no texture".to_owned())
    }

    /// Reads a texture back; `f(x, y, texel bytes)` per texel of `texel` bytes.
    fn read(
        &self,
        t: &ID3D11Texture2D,
        texel: usize,
        mut f: impl FnMut(u32, u32, &[u8]),
    ) -> Result<(), String> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY (whole block): staging copy of a live texture of the same description,
        // mapped for reading within its row pitch.
        unsafe {
            t.GetDesc(&mut desc);
            desc.Usage = D3D11_USAGE_STAGING;
            desc.BindFlags = 0;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
            desc.MiscFlags = 0;
            let mut st = None;
            self.device
                .CreateTexture2D(&desc, None, Some(&mut st))
                .map_err(|e| format!("staging: {e}"))?;
            let st: ID3D11Texture2D = st.ok_or("no staging")?;
            self.ctx.CopyResource(&st, t);
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            self.ctx
                .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
                .map_err(|e| format!("map: {e}"))?;
            for y in 0..desc.Height {
                let row = std::slice::from_raw_parts(
                    m.pData.cast::<u8>().add(y as usize * m.RowPitch as usize),
                    desc.Width as usize * texel,
                );
                for x in 0..desc.Width {
                    f(x, y, &row[x as usize * texel..(x as usize + 1) * texel]);
                }
            }
            self.ctx.Unmap(&st, 0);
        }
        Ok(())
    }
}

fn f16_bits(v: f32) -> u16 {
    // Positive normal range only (offline test data).
    if v <= 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    if e <= 0 {
        return 0;
    }
    ((e.min(30) as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}

fn f16_value(h: u16) -> f32 {
    let e = i32::from((h >> 10) & 0x1F);
    let m = f32::from(h & 0x3FF) / 1024.0;
    match e {
        0 => m * 2f32.powi(-14),
        31 => f32::NAN,
        e => (1.0 + m) * 2f32.powi(e - 15),
    }
}

/// Offline check of the FSR work (`examples/fsr_offline.rs`): a stand-in "game" D3D11
/// device owns engine-like textures (R11G11B10 colour, R16G16 motion, R24G8 depth, a
/// 32-byte structured auto-exposure buffer, R11G11B10 output, RG16F `OpacityBuffer`, R8
/// `hairOpacity`, half-size RGBA16F `distoVectors`), and `Runtime::run` packs them, runs
/// FSR on D3D12 with cross-API fences and unpacks:
/// 1. native (render = display), alternating the two entry styles: inline on the
///    immediate context (scheduler without command lists) and before a deferred
///    context's command list that reads the output (submit hook; checks ordering);
/// 2. upscale at `scale` (render snapped to multiples of 8): jitter through the engine
///    table round trip, masks, the `distoVectors` stretch and the depth resample.
///
/// Returns a report. Covers everything except engine memory, hooks and call sites.
pub fn offline_check(
    ffx_dir: &std::path::Path,
    size: [u32; 2],
    frames: u32,
    scale: f32,
) -> Result<String, String> {
    use crate::fsr_scale::{halton_offset, phase_count, snap_render, table_entry};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_1};
    // The value check and motion statistics are part of what this harness verifies.
    settings::edit(|s| s.diagnostics = true);
    let loader = ffx::Loader::load(ffx_dir)?;
    let [w, h] = size;
    let (mut device, mut ctx) = (None, None);
    // SAFETY: device creation with valid arguments and out pointers.
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            Some(&[D3D_FEATURE_LEVEL_11_1]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut ctx),
        )
    }
    .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
    let o = Offline {
        device: device.ok_or("no device")?,
        ctx: ctx.ok_or("no context")?,
    };
    let srv = D3D11_BIND_SHADER_RESOURCE.0;
    let render_up = [snap_render(w, scale), snap_render(h, scale)];
    let pattern = |x: u32, y: u32| {
        let line = if x.is_multiple_of(7) || y.is_multiple_of(11) {
            2.0
        } else {
            0.0
        };
        0.25 + 0.2 * ((x as f32 * 0.05).sin() * (y as f32 * 0.03).cos()) + line
    };
    let mut color = Vec::with_capacity((w * h * 4) as usize);
    let (mut native_sum, mut up_sum) = (0f64, 0f64);
    for y in 0..h {
        for x in 0..w {
            let v = pattern(x, y);
            native_sum += f64::from(v);
            if x < render_up[0] && y < render_up[1] {
                up_sum += f64::from(v);
            }
            color.extend_from_slice(&pack_r11g11b10([v, v * 0.8, v * 0.6]).to_le_bytes());
        }
    }
    let c = o.texture(
        size,
        DXGI_FORMAT_R11G11B10_FLOAT,
        srv,
        Some((&color, w * 4)),
    )?;
    let zeros = vec![0u8; (w * h * 4) as usize];
    let mv = o.texture(size, DXGI_FORMAT_R16G16_FLOAT, srv, Some((&zeros, w * 4)))?;
    // Depth (R24G8): 0.5 with stencil 7 inside the upscale render region, 0 (far, stencil
    // 0) elsewhere.
    let mut zdata = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let d: u32 = if x < render_up[0] && y < render_up[1] {
                0x7F_FFFF | (7 << 24)
            } else {
                0
            };
            zdata.extend_from_slice(&d.to_le_bytes());
        }
    }
    // The same data in a texture without a depth-stencil binding (like `decompressedDepth`
    // may be): resampled through the scratch path.
    let depth_srv_only = o.texture(size, DXGI_FORMAT_R24G8_TYPELESS, srv, Some((&zdata, w * 4)))?;
    let depth = o.texture(
        size,
        DXGI_FORMAT_R24G8_TYPELESS,
        srv | D3D11_BIND_DEPTH_STENCIL.0,
        Some((&zdata, w * 4)),
    )?;
    let out = o.texture(
        size,
        DXGI_FORMAT_R11G11B10_FLOAT,
        srv | D3D11_BIND_UNORDERED_ACCESS.0,
        None,
    )?;
    let check = o.texture(size, DXGI_FORMAT_R11G11B10_FLOAT, srv, None)?;
    // OIT revealage: 0.25 (coverage 0.75) in a square, 1 elsewhere; hair revealage 1.
    let mut op = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let rev = if (w / 8..w / 4).contains(&x) && (h / 8..h / 4).contains(&y) {
                0.25
            } else {
                1.0
            };
            op.extend_from_slice(&f16_bits(rev).to_le_bytes());
            op.extend_from_slice(&0u16.to_le_bytes());
        }
    }
    let opacity = o.texture(size, DXGI_FORMAT_R16G16_FLOAT, srv, Some((&op, w * 4)))?;
    let ones = vec![255u8; (w * h) as usize];
    let hair = o.texture(size, DXGI_FORMAT_R8_UNORM, srv, Some((&ones, w)))?;
    // distoVectors: half size, x = source column + 1 inside the scaled region.
    let half = [w / 2, h / 2];
    let disto_src = [
        crate::fsr_scale::scaled_extent(half[0], render_up[0] as f32 / w as f32),
        crate::fsr_scale::scaled_extent(half[1], render_up[1] as f32 / h as f32),
    ];
    let mut dd = Vec::with_capacity((half[0] * half[1] * 8) as usize);
    for _y in 0..half[1] {
        for x in 0..half[0] {
            let v = if x < disto_src[0] {
                x as f32 + 1.0
            } else {
                0.0
            };
            for c in [v, 0.0, 0.0, 1.0] {
                dd.extend_from_slice(&f16_bits(c).to_le_bytes());
            }
        }
    }
    let disto = o.texture(
        half,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        srv | D3D11_BIND_RENDER_TARGET.0,
        Some((&dd, half[0] * 8)),
    )?;
    // Auto exposure: 2 elements, EngineLuminanceFactor 1.
    let mut ae = [0f32; 16];
    ae[0] = 1.0;
    ae[8] = 1.0;
    let mut buf = None;
    // SAFETY: buffer creation with initial data of the declared size.
    unsafe {
        o.device.CreateBuffer(
            &D3D11_BUFFER_DESC {
                ByteWidth: 64,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: srv as u32,
                MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
                StructureByteStride: 32,
                ..Default::default()
            },
            Some(&D3D11_SUBRESOURCE_DATA {
                pSysMem: ae.as_ptr().cast(),
                ..Default::default()
            }),
            Some(&mut buf),
        )
    }
    .map_err(|e| format!("exposure buffer: {e}"))?;
    let res = |t: &ID3D11Texture2D| -> Result<ID3D11Resource, String> {
        t.cast().map_err(|e| e.to_string())
    };
    let t0 = Instant::now();
    let mut rt = Runtime::new(&o.device, size, &loader)?;
    let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let job = |f: u32, render: [u32; 2], up: bool| -> Result<Job, String> {
        // Jitter as the engine produces it from our table entry (round trip).
        let n = phase_count(render[0], w);
        let e = table_entry(halton_offset(f, n), size, render);
        Ok(Job {
            frame: 100 + f,
            color: res(&c)?,
            motion: res(&mv)?,
            depth: res(&depth)?,
            output: res(&out)?,
            exposure: buf.clone(),
            exposure_index: 1,
            ev_bias: 0.0,
            view_white: 1.0,
            compressed_hdr: true,
            render,
            display: size,
            jitter: [e[0] / w as f32, e[1] / h as f32],
            near: 0.01,
            fov_y: 1.0,
            cut: false,
            opacity: up.then(|| res(&opacity)).transpose()?,
            hair: up.then(|| res(&hair)).transpose()?,
            disto: up.then(|| res(&disto)).transpose()?,
            ssr: None,
            camera: Camera::default(),
        })
    };
    // 1. Native, both entry styles.
    let mut deferred_ok = true;
    let t1 = Instant::now();
    for f in 0..frames {
        let j = job(f, size, false)?;
        if f % 2 == 0 {
            rt.run(&o.ctx, &j)?;
            continue;
        }
        // SAFETY: a deferred context records a copy of the output; the list runs after
        // the FSR work, as the engine's POSTPROC list does after the submit hook.
        unsafe {
            let mut dc = None;
            o.device
                .CreateDeferredContext(0, Some(&mut dc))
                .map_err(|e| format!("deferred context: {e}"))?;
            let dc: ID3D11DeviceContext = dc.ok_or("no deferred context")?;
            dc.CopyResource(&check, &out);
            let mut list = None;
            dc.FinishCommandList(false, Some(&mut list))
                .map_err(|e| format!("FinishCommandList: {e}"))?;
            rt.run(&o.ctx, &j)?;
            o.ctx.ExecuteCommandList(list.as_ref(), false);
        }
        if f + 2 >= frames {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            o.read(&out, 4, |_, _, t| a.extend_from_slice(t))?;
            o.read(&check, 4, |_, _, t| b.extend_from_slice(t))?;
            deferred_ok &= a == b;
        }
    }
    let native_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let red_stats = |area: [u32; 4]| -> Result<(f64, f32, u64), String> {
        let (mut sum, mut lo, mut bad, mut n) = (0f64, f32::MAX, 0u64, 0u64);
        o.read(&out, 4, |x, y, t| {
            if x < area[0] || y < area[1] || x >= area[2] || y >= area[3] {
                return;
            }
            let r = unpack_r11g11b10(u32::from_le_bytes([t[0], t[1], t[2], t[3]]))[0];
            if r.is_finite() {
                sum += f64::from(r);
                lo = lo.min(r);
                n += 1;
            } else {
                bad += 1;
            }
        })?;
        Ok((sum / n.max(1) as f64, lo, bad))
    };
    let (native_mean, native_min, native_bad) = red_stats([0, 0, w, h])?;
    let native_resets = rt.resets;
    // 2. Upscale.
    let t2 = Instant::now();
    for f in 0..frames {
        let mut j = job(frames + f, render_up, true)?;
        // The engine rewrites distoVectors every frame; the stand-in is written once, so
        // only the last frame stretches it.
        if f + 1 < frames {
            j.disto = None;
        }
        rt.run(&o.ctx, &j)?;
    }
    let up_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let (up_mean, up_min, up_bad) = red_stats([0, 0, w, h])?;
    let (corner_mean, corner_min, _) = red_stats([w * 3 / 4, h * 3 / 4, w, h])?;
    // Transparency mask (shared texture on the same device), read through a copy.
    let mut tc_hi = 0u64;
    {
        let uav = rt.tc.uav.clone().ok_or("no tc UAV")?;
        // SAFETY: getter on our own live view.
        let r = unsafe { uav.GetResource() }.map_err(|e| format!("tc resource: {e}"))?;
        let tex: ID3D11Texture2D = r.cast().map_err(|e| e.to_string())?;
        o.read(&tex, 1, |x, y, t| {
            if x < render_up[0] && y < render_up[1] && t[0] > 128 {
                tc_hi += 1;
            }
        })?;
    }
    let tc_expected =
        u64::from((w / 4).min(render_up[0]) - w / 8) * u64::from((h / 4).min(render_up[1]) - h / 8);
    // distoVectors: the right column must hold the last source column after the stretch.
    let (mut disto_right, mut disto_mid) = (0f32, 0f32);
    o.read(&disto, 8, |x, y, t| {
        if y == half[1] / 2 {
            let v = f16_value(u16::from_le_bytes([t[0], t[1]]));
            if x == half[0] - 1 {
                disto_right = v;
            } else if x == half[0] / 2 {
                disto_mid = v;
            }
        }
    })?;
    // Depth resample (the upscale frames above already resampled `depth` as fx/depthFull):
    // every texel should hold 0.5 and, with SV_StencilRef, stencil 7.
    rt.resample_depth(&o.ctx, &res(&depth)?, render_up, size)?;
    rt.resample_depth(&o.ctx, &res(&depth_srv_only)?, render_up, size)?;
    let stencil_ref = rt.ps_depth_stencil.is_some();
    let depth_stats = |t: &ID3D11Texture2D| -> Result<(u64, u64), String> {
        let (mut half, mut st) = (0u64, 0u64);
        o.read(t, 4, |_, _, b| {
            let d = u32::from_le_bytes([b[0], b[1], b[2], 0]);
            half += u64::from(d.abs_diff(0x7F_FFFF) < 4);
            st += u64::from(b[3] == 7);
        })?;
        Ok((half, st))
    };
    let (depth_half, depth_stencil) = depth_stats(&depth)?;
    let (srv_only_half, srv_only_stencil) = depth_stats(&depth_srv_only)?;
    let sky = [
        offline_sky(&o, &mut rt, size, false)?,
        offline_sky(&o, &mut rt, size, true)?,
    ]
    .join("\n");
    // 3. Value check and guard (white-flash diagnostics): a colour input with 4 NaN, 4 Inf
    // and 4 huge (65504) texels. The check must report them at the pack input (stage 1);
    // the guard must keep FSR's input clean (stage 2) and the final output finite.
    let poison = {
        let mut px = Vec::with_capacity((w * h * 8) as usize);
        for y in 0..h {
            for x in 0..w {
                let r: u16 = match (x, y) {
                    (0..4, 0) => 0x7E00,  // NaN
                    (4..8, 0) => 0x7C00,  // +Inf
                    (8..12, 0) => 0x7BFF, // 65504
                    _ => 0x3800,          // 0.5
                };
                for c in [r, 0x3800, 0x3800, 0x3C00] {
                    px.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        o.texture(
            size,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            srv,
            Some((&px, w * 8)),
        )?
    };
    for f in 0..4 {
        let mut j = job(3 * frames + f, size, false)?;
        j.color = res(&poison)?;
        if let Some(c) = rt.check.as_mut() {
            c.begin(&o.ctx);
        }
        rt.run(&o.ctx, &j)?;
    }
    let (_, _, poison_bad) = red_stats([0, 0, w, h])?;
    let check = rt.check.as_mut().ok_or("value check unavailable")?;
    check.drain(&o.ctx, 2000);
    let (bad_frame, st) = check.last_bad.ok_or("value check reported no bad frame")?;
    let checked = check.checked;
    let ok = (st[1].nan, st[1].inf, st[1].huge) == (4, 4, 4)
        && !st[0].bad()
        && !st[2].bad()
        && st[2].max <= crate::fsr_check::HUGE
        && st[3].nan + st[3].inf == 0
        && poison_bad == 0;
    let value_check = format!(
        "value check: {checked} frames read back, last bad frame {bad_frame}: {}; final output non-finite {poison_bad}: {}",
        crate::fsr_check::describe(&st),
        if ok { "OK" } else { "FAILED" }
    );
    let motion = offline_motion(&o, &mut rt, size, |render, mv, z| {
        let mut j = job(4 * frames, render, false)?;
        j.motion = res(mv)?;
        j.depth = res(z)?;
        Ok(j)
    })?;
    let bias = offline_mip_bias(&o)?;
    rt.report();
    Ok(format!(
        "offline FSR: provider {}, display {w}x{h}, init {init_ms:.0} ms\n\
         native: {frames} frames in {native_ms:.1} ms CPU (inline + deferred-list styles); red mean {native_mean:.4} (input {:.4}), min {native_min:.4}, non-finite {native_bad}; deferred list saw the FSR output: {deferred_ok}; resets {native_resets}\n\
         upscale {}x{} -> {w}x{h} (scale {scale}): {frames} frames in {up_ms:.1} ms CPU; red mean {up_mean:.4} (render-region input {:.4}), min {up_min:.4}, non-finite {up_bad}; bottom-right quarter mean {corner_mean:.4} min {corner_min:.4}\n\
         masks: transparency texels > 0.5 = {tc_hi} (expected {tc_expected}); distoVectors stretch: right column {disto_right} (last source column {}), middle {disto_mid}; resets total {}\n\
         depth resample (of {} texels; SV_StencilRef {stencil_ref}): depth target {depth_half} at 0.5, {depth_stencil} stencil 7; SRV-only target {srv_only_half} at 0.5, {srv_only_stencil} stencil 7\n\
         {sky}\n\
         {value_check}\n\
         {motion}\n\
         {bias}",
        rt.upscaler.provider().version,
        native_sum / f64::from(w * h),
        render_up[0],
        render_up[1],
        up_sum / f64::from(render_up[0] * render_up[1]),
        disto_src[0],
        rt.resets,
        w * h,
    ))
}

/// Motion-vector diagnostics (`fsr_motion.rs`): synthetic engine vectors and depth in four
/// depth-band columns (2, 10, 40, 200 m; top quarter sky): rows h/2..3h/4 move 0.5 px, rows
/// 3h/4..h 0.1 px (below the 0.25 px threshold). Checks the per-band counts against a CPU
/// count, then runs one FSR frame with them and draws the debug view into an RGBA8 target:
/// brightness of both motions, black where the vectors are 0, the legend bar.
fn offline_motion(
    o: &Offline,
    rt: &mut Runtime,
    size: [u32; 2],
    job: impl Fn([u32; 2], &ID3D11Texture2D, &ID3D11Texture2D) -> Result<Job, String>,
) -> Result<String, String> {
    use crate::fsr_motion::{THRESHOLD_PX, view_brightness};
    let [w, h] = size;
    let near = 0.01f32;
    let band_z = [2.0f32, 10.0, 40.0, 200.0];
    let texel = |x: u32, y: u32| -> (f32, f32) {
        let d = if y < h / 4 {
            0.0
        } else {
            near / band_z[(x * 4 / w) as usize]
        };
        let px = if y >= h * 3 / 4 {
            0.1
        } else if y >= h / 2 {
            0.5
        } else {
            0.0
        };
        (d, px / w as f32)
    };
    let (mut zd, mut md) = (Vec::new(), Vec::new());
    let mut expected = [(0u64, 0u64); 5];
    for y in 0..h {
        for x in 0..w {
            let (d, u) = texel(x, y);
            zd.extend_from_slice(&d.to_le_bytes());
            md.extend_from_slice(&u.to_le_bytes());
            md.extend_from_slice(&0f32.to_le_bytes());
            let band = if d > 0.0 { (x * 4 / w) as usize } else { 4 };
            expected[band].1 += 1;
            expected[band].0 += u64::from(u * w as f32 > THRESHOLD_PX);
        }
    }
    let srv = D3D11_BIND_SHADER_RESOURCE.0;
    let ztex = o.texture(size, DXGI_FORMAT_R32_FLOAT, srv, Some((&zd, w * 4)))?;
    let mtex = o.texture(size, DXGI_FORMAT_R32G32_FLOAT, srv, Some((&md, w * 8)))?;
    let res = |t: &ID3D11Texture2D| -> Result<ID3D11Resource, String> {
        t.cast().map_err(|e| e.to_string())
    };
    let (View::Srv(mv_srv), View::Srv(z_srv)) =
        (rt.view(&res(&mtex)?, false)?, rt.view(&res(&ztex)?, false)?)
    else {
        return Err("view kinds".into());
    };
    let stats = rt
        .motion_stats
        .as_mut()
        .ok_or("motion statistics unavailable")?;
    stats.sample(&o.ctx, &mv_srv, &z_srv, size, near);
    stats.drain(&o.ctx, 2000);
    let bands = stats.last.ok_or("no motion statistics read back")?;
    let stats_ok = bands == expected;

    rt.run(&o.ctx, &job(size, &mtex, &ztex)?)?;
    let target = o.texture(
        size,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        srv | D3D11_BIND_RENDER_TARGET.0,
        None,
    )?;
    let mut rtv = None;
    // SAFETY: default view of our own render-target texture.
    unsafe {
        o.device
            .CreateRenderTargetView(&target, None, Some(&mut rtv))
    }
    .map_err(|e| format!("debug-view RTV: {e}"))?;
    let rtv = rtv.ok_or("no debug-view RTV")?;
    let view = rt
        .motion_view
        .as_ref()
        .ok_or("motion debug view unavailable")?;
    view.draw(
        &o.ctx,
        &rt.gfx_state,
        &rt.raster,
        &rtv,
        rt.last_render,
        size,
    )?;
    let probes = [
        (w / 8, h * 5 / 8, view_brightness(0.5)),
        (w * 5 / 8, h * 7 / 8 - 40, view_brightness(0.1)),
        (w * 3 / 8, h * 3 / 8, 0.0),
        (w / 2, h - 6, 0.5),
    ];
    let mut got = [0f32; 4];
    o.read(&target, 4, |x, y, t| {
        for (i, &(px, py, _)) in probes.iter().enumerate() {
            if (x, y) == (px, py) {
                got[i] = f32::from(t[0].max(t[1]).max(t[2])) / 255.0;
            }
        }
    })?;
    let view_ok = probes.iter().zip(&got).all(|(p, g)| (p.2 - g).abs() < 0.02);
    Ok(format!(
        "motion statistics: {} (expected {}); debug view brightness 0.5 px {:.3} (expected {:.3}), 0.1 px {:.3} ({:.3}), zero {:.3}, legend middle {:.3}: {}",
        crate::fsr_motion::describe(&bands),
        if stats_ok { "match" } else { "MISMATCH" },
        got[0],
        probes[0].2,
        got[1],
        probes[1].2,
        got[2],
        got[3],
        if stats_ok && view_ok { "OK" } else { "FAILED" }
    ))
}

/// Live mip bias on stand-in engine sampler cells (`fsr_scale::samplers`): an anisotropic
/// wrap sampler (options bias 0.25) and a clamp sampler (not biased) in heap cells, biased
/// to -1.5, then restored; checks the cells' objects, their descs and the reference counts.
fn offline_mip_bias(o: &Offline) -> Result<String, String> {
    use crate::fsr_scale::samplers;
    use std::sync::atomic::AtomicU64 as Cell;
    /// A stand-in engine cell: owns one sampler reference; unregistered and released on
    /// drop (also on an early error return).
    struct OfflineCell(Box<Cell>);
    impl OfflineCell {
        fn addr(&self) -> u64 {
            self.0.as_ptr() as u64
        }
    }
    impl Drop for OfflineCell {
        fn drop(&mut self) {
            samplers::forget(self.addr());
            let raw = self.0.load(Ordering::Acquire) as *mut c_void;
            if !raw.is_null() {
                // SAFETY: the cell's own reference.
                drop(unsafe { ID3D11SamplerState::from_raw(raw) });
            }
        }
    }
    let make = |address: D3D11_TEXTURE_ADDRESS_MODE| -> Result<OfflineCell, String> {
        let mut s = None;
        // SAFETY: valid descriptor and out pointer.
        unsafe {
            o.device.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_ANISOTROPIC,
                    AddressU: address,
                    AddressV: address,
                    AddressW: address,
                    MipLODBias: 0.25,
                    MaxAnisotropy: 16,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                Some(&mut s),
            )
        }
        .map_err(|e| format!("sampler: {e}"))?;
        let s: ID3D11SamplerState = s.ok_or("no sampler")?;
        Ok(OfflineCell(Box::new(Cell::new(s.into_raw() as u64))))
    };
    let cells = [
        make(D3D11_TEXTURE_ADDRESS_WRAP)?,
        make(D3D11_TEXTURE_ADDRESS_CLAMP)?,
    ];
    let (wrap, clamp) = (cells[0].addr(), cells[1].addr());
    // Options bits: anisotropic min/mag/mip (0x2A), U clamp = 3 << 6.
    samplers::note(wrap, 0x2A, 0.25);
    samplers::note(clamp, 0x2A | (3 << 6), 0.25);
    let bias_of = |cell: u64| -> Result<(f32, u64), String> {
        let raw = mem::u64_at(cell).ok_or("cell")? as *mut c_void;
        // SAFETY: the cell holds a live sampler (ours or the swapped-in one).
        let s = unsafe { ID3D11SamplerState::from_raw_borrowed(&raw) }.ok_or("null cell")?;
        let mut d = D3D11_SAMPLER_DESC::default();
        // SAFETY: plain getter.
        unsafe { s.GetDesc(&mut d) };
        Ok((d.MipLODBias, raw as u64))
    };
    let before = bias_of(wrap)?;
    let mut applied = None;
    for _ in 0..16 {
        applied = applied.or(samplers::frame(-1.5));
    }
    let (biased, biased_obj) = bias_of(wrap)?;
    let clamp_bias = bias_of(clamp)?.0;
    let mut restored = None;
    for _ in 0..16 {
        restored = restored.or(samplers::frame(0.0));
    }
    let (back, _) = bias_of(wrap)?;
    // Let the grace period release the swapped-out objects; the cells release theirs.
    for _ in 0..(samplers::GRACE_FRAMES + 2) {
        samplers::frame(0.0);
    }
    drop(cells);
    let ok = (biased + 1.25).abs() < 1e-4
        && (back - 0.25).abs() < 1e-4
        && (clamp_bias - 0.25).abs() < 1e-4
        && biased_obj != before.1
        && applied.is_some_and(|(b, n)| b == -1.5 && n == 1)
        && restored.is_some_and(|(b, _)| b == 0.0);
    Ok(format!(
        "live mip bias: wrap sampler {:.2} -> {biased:.2} (new object {}) -> {back:.2}; clamp sampler {clamp_bias:.2} (untouched); applied {applied:?}, restored {restored:?}: {}",
        before.0,
        biased_obj != before.1,
        if ok { "OK" } else { "MISMATCH" }
    ))
}

/// Row-major 4x4 product `a * b` (row-vector convention, as the offline camera uses).
fn mat_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut o = [0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            o[r * 4 + c] = (0..4).map(|k| a[r * 4 + k] * b[k * 4 + c]).sum();
        }
    }
    o
}

/// Row vector times row-major matrix.
fn vec_mul(v: [f32; 4], m: &[f32; 16]) -> [f32; 4] {
    std::array::from_fn(|c| (0..4).map(|k| v[k] * m[k * 4 + c]).sum())
}

fn f16_signed(v: f32) -> u16 {
    if v < 0.0 {
        0x8000 | f16_bits(-v)
    } else {
        f16_bits(v)
    }
}

fn f16_signed_value(h: u16) -> f32 {
    let v = f16_value(h & 0x7FFF);
    if h & 0x8000 != 0 { -v } else { v }
}

/// Sky motion vectors offline: a camera turning 0.01 rad per frame with a reversed-Z
/// infinite projection, matrices in row-vector convention, passed as is (expected
/// decision `mul(x, M)`) or transposed (`mul(M, x)`); no jitter term expected, with a
/// 1.5-pixel jitter delta that the engine-style vectors exclude. Geometry (left half,
/// depths 2..50 m) carries engine-style vectors; the sky (right half, depth 0, vector 0)
/// must receive the reprojected direction's vector.
fn offline_sky(
    o: &Offline,
    rt: &mut Runtime,
    size: [u32; 2],
    transposed: bool,
) -> Result<String, String> {
    let [w, h] = size;
    SKY_VARIANT.store(0, Ordering::Relaxed);
    rt.sky = Sky::new(&rt.d11)?;
    let near = 0.01f32;
    let fy = 1.0 / 0.5f32.tan();
    let fx = fy * h as f32 / w as f32;
    // clip = v * P: (x fx, y fy, near, -z); depth = near / -z.
    let proj = [
        fx, 0.0, 0.0, 0.0, //
        0.0, fy, 0.0, 0.0, //
        0.0, 0.0, 0.0, -1.0, //
        0.0, 0.0, near, 0.0,
    ];
    let proj_inv = [
        1.0 / fx,
        0.0,
        0.0,
        0.0, //
        0.0,
        1.0 / fy,
        0.0,
        0.0, //
        0.0,
        0.0,
        0.0,
        1.0 / near, //
        0.0,
        0.0,
        -1.0,
        0.0,
    ];
    let rot = |yaw: f32| {
        let (s, c) = yaw.sin_cos();
        [
            c, 0.0, -s, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            s, 0.0, c, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    };
    let rot_t = |yaw: f32| rot(-yaw);
    let jitter_rel = [2.0 * 1.5 / w as f32, 0.0];
    let ndc = |x: u32, y: u32| {
        [
            (x as f32 + 0.5) / w as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / h as f32 * 2.0,
        ]
    };
    let prev_ndc = |n: [f32; 2], d: f32, inv: &[f32; 16], prev: &[f32; 16]| {
        let p = vec_mul(vec_mul([n[0], n[1], d, 1.0], inv), prev);
        [p[0] / p[3], p[1] / p[3]]
    };
    let srv = D3D11_BIND_SHADER_RESOURCE.0;
    let zero = vec![0u8; (w * h * 4) as usize];
    let depth = o.texture(
        size,
        DXGI_FORMAT_R24G8_TYPELESS,
        srv | D3D11_BIND_DEPTH_STENCIL.0,
        Some((&zero, w * 4)),
    )?;
    let mv = o.texture(size, DXGI_FORMAT_R16G16_FLOAT, srv, Some((&zero, w * 4)))?;
    let color = o.texture(size, DXGI_FORMAT_R11G11B10_FLOAT, srv, Some((&zero, w * 4)))?;
    let out = o.texture(
        size,
        DXGI_FORMAT_R11G11B10_FLOAT,
        srv | D3D11_BIND_UNORDERED_ACCESS.0,
        None,
    )?;
    let res = |t: &ID3D11Texture2D| -> Result<ID3D11Resource, String> {
        t.cast().map_err(|e| e.to_string())
    };
    let (mut zbuf, mut mbuf) = (vec![0u8; zero.len()], vec![0u8; zero.len()]);
    let mut last = ([0f32; 16], [0f32; 16]);
    let mut decided_at = None;
    let mut f = 0u32;
    while f < 200 && decided_at.is_none_or(|d| f < d + 3) {
        let (yaw_prev, yaw) = (f as f32 * 0.01, (f + 1) as f32 * 0.01);
        let vp_prev = mat_mul(&rot(yaw_prev), &proj);
        let inv = mat_mul(&proj_inv, &rot_t(yaw));
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let (d, m) = if x < w / 2 {
                    let d = near / (2.0 + 48.0 * y as f32 / h as f32);
                    let n = ndc(x, y);
                    let p = prev_ndc(n, d, &inv, &vp_prev);
                    (d, [(n[0] - p[0]) * 0.5, (n[1] - p[1]) * -0.5])
                } else {
                    (0.0, [0.0, 0.0])
                };
                let du = (d * 16_777_215.0).round() as u32;
                zbuf[i..i + 4].copy_from_slice(&du.to_le_bytes());
                mbuf[i..i + 2].copy_from_slice(&f16_signed(m[0]).to_le_bytes());
                mbuf[i + 2..i + 4].copy_from_slice(&f16_signed(m[1]).to_le_bytes());
            }
        }
        // SAFETY: full-texture updates with data of the declared size and pitch.
        unsafe {
            o.ctx
                .UpdateSubresource(&depth, 0, None, zbuf.as_ptr().cast(), w * 4, 0);
            o.ctx
                .UpdateSubresource(&mv, 0, None, mbuf.as_ptr().cast(), w * 4, 0);
        }
        let job = Job {
            frame: 10_000 + f,
            color: res(&color)?,
            motion: res(&mv)?,
            depth: res(&depth)?,
            output: res(&out)?,
            exposure: None,
            exposure_index: 0,
            ev_bias: 0.0,
            view_white: 1.0,
            compressed_hdr: false,
            render: size,
            display: size,
            jitter: [0.0, 0.0],
            near,
            fov_y: 1.0,
            cut: false,
            opacity: None,
            hair: None,
            disto: None,
            ssr: None,
            camera: Camera {
                inv_vp: if transposed { transpose(&inv) } else { inv },
                prev_vp: if transposed {
                    transpose(&vp_prev)
                } else {
                    vp_prev
                },
                jitter_rel,
                moving: true,
                valid: true,
            },
        };
        rt.run(&o.ctx, &job)?;
        last = (inv, vp_prev);
        if decided_at.is_none() && SKY_VARIANT.load(Ordering::Relaxed) != 0 {
            decided_at = Some(f);
        }
        f += 1;
    }
    let variant = SKY_VARIANT.load(Ordering::Relaxed);
    // The packed motion vectors FSR received (shared texture, same device).
    let uav = rt.motion.uav.clone().ok_or("no motion UAV")?;
    // SAFETY: getter on our own live view.
    let r = unsafe { uav.GetResource() }.map_err(|e| format!("motion resource: {e}"))?;
    let packed: ID3D11Texture2D = r.cast().map_err(|e| e.to_string())?;
    let (mut sky_err, mut geo_err, mut sky_n) = (0f32, 0f32, 0u64);
    o.read(&packed, 4, |x, y, t| {
        let m = [
            f16_signed_value(u16::from_le_bytes([t[0], t[1]])),
            f16_signed_value(u16::from_le_bytes([t[2], t[3]])),
        ];
        let i = ((y * w + x) * 4) as usize;
        let input = [
            f16_signed_value(u16::from_le_bytes([mbuf[i], mbuf[i + 1]])),
            f16_signed_value(u16::from_le_bytes([mbuf[i + 2], mbuf[i + 3]])),
        ];
        let px =
            |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]) * w as f32).hypot((a[1] - b[1]) * h as f32);
        if x < w / 2 {
            geo_err = geo_err.max(px(m, input));
        } else {
            let n = ndc(x, y);
            let p = prev_ndc(n, 0.0, &last.0, &last.1);
            sky_err = sky_err.max(px(m, [(n[0] - p[0]) * 0.5, (n[1] - p[1]) * -0.5]));
            sky_n += 1;
        }
    })?;
    let name = match variant {
        1..=6 => SKY_VARIANTS[variant as usize - 1],
        0 => "undecided",
        _ => "gave up",
    };
    let expected = if transposed {
        SKY_VARIANTS[0]
    } else {
        SKY_VARIANTS[3]
    };
    Ok(format!(
        "sky motion vectors ({} matrices): decision '{name}' after {} frames (expected '{expected}'); sky pixels {sky_n}: max error {sky_err:.3} px; geometry vectors passed through: max change {geo_err:.4} px",
        if transposed {
            "transposed"
        } else {
            "row-vector"
        },
        decided_at.map_or(f, |d| d + 1)
    ))
}

fn transpose(m: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|i| m[(i % 4) * 4 + i / 4])
}

fn f32_to_small_float(v: f32, mant_bits: u32) -> u32 {
    // Unsigned float with 5 exponent bits (bias 15) and `mant_bits` mantissa bits.
    let v = v.max(0.0);
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    if exp <= 0 {
        return 0;
    }
    if exp >= 31 {
        return (30 << mant_bits) | ((1 << mant_bits) - 1);
    }
    ((exp as u32) << mant_bits) | ((bits >> (23 - mant_bits)) & ((1 << mant_bits) - 1))
}

fn small_float_to_f32(v: u32, mant_bits: u32) -> f32 {
    let exp = (v >> mant_bits) & 0x1F;
    let mant = v & ((1 << mant_bits) - 1);
    match exp {
        0 => mant as f32 / (1 << mant_bits) as f32 * 2f32.powi(-14),
        31 => f32::NAN,
        e => (1.0 + mant as f32 / (1 << mant_bits) as f32) * 2f32.powi(e as i32 - 15),
    }
}

fn pack_r11g11b10(c: [f32; 3]) -> u32 {
    f32_to_small_float(c[0], 6)
        | (f32_to_small_float(c[1], 6) << 11)
        | (f32_to_small_float(c[2], 5) << 22)
}

fn unpack_r11g11b10(p: u32) -> [f32; 3] {
    [
        small_float_to_f32(p & 0x7FF, 6),
        small_float_to_f32((p >> 11) & 0x7FF, 6),
        small_float_to_f32(p >> 22, 5),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(inliers: [u64; 6], samples: u64) -> [u64; 7] {
        let mut c = [0; 7];
        c[..6].copy_from_slice(&inliers);
        c[6] = samples;
        c
    }

    #[test]
    fn sky_decision_waits_for_enough_motion() {
        let c = counts([0, 0, 0, 9000, 0, 0], 10_000);
        assert_eq!(sky_decision(&c, SKY_MIN_FRAMES - 1), None);
        assert_eq!(sky_decision(&counts([0; 6], 100), SKY_MIN_FRAMES), None);
        assert_eq!(sky_decision(&c, SKY_MIN_FRAMES), Some(4));
    }

    #[test]
    fn sky_decision_needs_a_clear_winner_over_the_other_order() {
        // Both multiplication orders explain the vectors equally (camera barely turned).
        let tie = counts([6000, 0, 0, 6000, 0, 0], 10_000);
        assert_eq!(sky_decision(&tie, 100), None);
        assert_eq!(sky_decision(&tie, SKY_GIVE_UP_FRAMES), Some(u32::MAX));
        // Jitter variants of the same order may tie; the order must still win clearly.
        let jitter_tie = counts([1000, 0, 0, 7000, 7000, 0], 10_000);
        assert!(matches!(sky_decision(&jitter_tie, 100), Some(4 | 5)));
        // Most vectors unexplained (moving objects everywhere): no decision.
        assert_eq!(
            sky_decision(&counts([0, 0, 0, 4000, 0, 0], 10_000), 100),
            None
        );
    }
}
