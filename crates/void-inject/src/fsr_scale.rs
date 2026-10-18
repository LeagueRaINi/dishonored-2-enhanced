//! FSR around the TAA site (docs/fsr4-integration.md "Jitter", "Render scale and post
//! chain", "Mip bias", "Reactive and transparency inputs"): the FSR jitter sequence fed
//! through the engine's jitter table, engine resolution scaling for the upscale mode, the
//! switch of the post chain to display resolution, the mip bias of material samplers, and
//! the capture of the order-independent-transparency targets used as FSR masks.
//!
//! Hooks (MinHook, all installed together with FSR: at startup with `fsr = 1`, or when
//! `fsr` is turned on in the overlay; each one is inert unless its feature is active for
//! the current frame, so every FSR setting applies live; each target's first 16 bytes are
//! compared with the analysed build before patching):
//! - `idRenderView_SetupScaled` 0x140374B60 (main thread, per view): for the world view
//!   (`scale == &g_resolutionScale`), while FSR runs, writes the render scale into
//!   `g_resolutionScale` (upscale: snapped `fsr_scale`; native: 1, so engine dynamic
//!   resolution cannot scale an FSR frame) and the current FSR phase into all 8 entries of
//!   `g_taaJitterTable8` 0x141B35F10, and notes `m_justTeleported`.
//! - `PerViewCB_Upload` 0x1403590F0 (once per view per frame): fills the never-bound
//!   portal constant buffer (view+30720) as a PerViewCB twin with `resolutionScale = 1`.
//! - `PostFX_TemporalAA_and_Post` 0x140327270 / `POST_TONEMAP` 0x140323E70 (pass
//!   callbacks): run on a private copy of the view; after FSR took the frame
//!   ([`apply_switch`] from the TAA site) the copy carries unscaled rects and scale 1, so
//!   everything after TAA records at display size.
//! - `idTextureSamplerManagerLocal_CreateSampler` 0x140378280: registers every sampler
//!   cell for the live mip bias ([`samplers`]).
//! - `ARK_RENDERPASS_CLEAR` 0x1403110E0 / `ASYNC_CLEAR` 0x140310FB0 callbacks: note the
//!   `OpacityBuffer` / `hairOpacity` images from their write lists (masks).

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use hex_literal::hex;
use windows::Win32::Graphics::Direct3D11::ID3D11Resource;

use crate::{fsr, log, mem, settings};

// ------------------------------------------------------------------ engine sites --

const SETUP_SCALED_RVA: u64 = 0x37_4B60;
const SETUP_SCALED_HEAD: [u8; 16] = hex!("48 89 5C 24 08 48 89 74 24 10 48 89 7C 24 18 55");
const PERVIEW_UPLOAD_RVA: u64 = 0x35_90F0;
const PERVIEW_UPLOAD_HEAD: [u8; 16] = hex!("40 53 48 83 EC 20 4C 8B 05 3B A4 F9 02 48 8B DA");
/// Same prologue for both pass callbacks.
const POST_HEAD: [u8; 16] = hex!("48 8B C4 55 53 56 57 41 54 41 55 41 56 41 57 48");
const POSTPROC_RVA: u64 = 0x32_7270;
pub(crate) const POST_TONEMAP_RVA: u64 = 0x32_3E70;
const CREATE_SAMPLER_RVA: u64 = 0x37_8280;
const CREATE_SAMPLER_HEAD: [u8; 16] = hex!("48 8B C4 55 57 41 56 48 8D 68 A1 48 81 EC A0 00");
const CLEAR_RVA: u64 = 0x31_10E0;
const CLEAR_HEAD: [u8; 16] = hex!("48 89 5C 24 18 48 89 6C 24 20 56 57 41 54 41 56");
const ASYNC_CLEAR_RVA: u64 = 0x31_0FB0;
const ASYNC_CLEAR_HEAD: [u8; 16] = hex!("48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 48");
/// `sub_1402AF360(arkConstantBuffer*, engineCtx*, paramBlock*)`: Map/WRITE_DISCARD + copy
/// of every layout variable from the parameter block.
const FILL_CB_RVA: u64 = 0x2A_F360;
/// `qword_1432D4EF0`: immediate engine context (second argument of the fill).
const IMMEDIATE_ENGINE_CTX_RVA: u64 = 0x32D_4EF0;
/// `qword_1432F3538`: global parameter block (`SetPerViewShaderParms` target).
const PARAM_BLOCK_RVA: u64 = 0x32F_3538;
/// `g_taaJitterTable8`: 8 x (x, y) floats, display pixels / unscaled viewport size.
const JITTER_TABLE_RVA: u64 = 0x1B3_5F10;
/// `g_resolutionScale` {cur, prev, controller, ...}.
const RES_SCALE_RVA: u64 = 0x22F_D9F0;

// Parameter handles.
const H_PERVIEW_CB: u64 = 0x229_C308;
const H_RES_SCALE: u64 = 0x229_C508;
const H_PREV_RES_SCALE: u64 = 0x229_C528;

// View (`idRenderView`, 48,384 B by reflection) and pass layouts.
const VIEW_SIZE: usize = 48_384;
const VIEW_FLAGS_14900: u64 = 14_900;
/// `m_subrect_viewport` (origin used by `SetPerViewShaderParms` for non-portal views).
const VIEW_SUBRECT: u64 = 30_544;
const VIEW_FRAME: u64 = 30_644;
const VIEW_PERVIEW_CB: u64 = 30_648;
const VIEW_PORTAL_CB: u64 = 30_720;
const CB_OBJECT: usize = 72;
const VIEW_RENDER_WH: u64 = 30_792;
const VIEW_VIEWPORT: u64 = 30_816;
const VIEW_UNSCALED_VIEWPORT: u64 = 30_840;
const VIEW_SCISSOR: u64 = 30_864;
const VIEW_UNSCALED_SCISSOR: u64 = 30_888;
const VIEW_RES_SCALE: u64 = 30_912;
const VIEW_TELEPORTED: u64 = 48_380;
const PASS_VIEW: u64 = 0x70;
const PASS_CTX: u64 = 0x108;
const PASS_READS: u64 = 0x48;
const PASS_WRITES: u64 = 0x80;

// Resource-name hashes (FNV-1a over the name and its NUL).
const HASH_OPACITY: u32 = 0x0120_8B70;
const HASH_HAIR_OPACITY: u32 = 0x2029_3876;
const HASH_DISTO_VECTORS: u32 = 0x1BBE_2D1A;
const HASH_ZBUFFER: u32 = 0xBA72_9A59;

fn base() -> u64 {
    fsr::base()
}

// ---------------------------------------------------------------- pure helpers --

/// Render size for `display` at `scale`: rounded to a multiple of 8 pixels, within
/// [8, display]. Integer rects make `ScaleScreenRect` (floor(w*s+0.5)) and the shaders'
/// float `cb_resolutionscale` agree exactly (d2enhanced: blur at non-integer sizes);
/// multiples of 8 also keep the half/quarter/eighth-size targets' scaled rects whole
/// (the engine's own dynamic resolution steps in 8-pixel units: `sub_140376C80`).
pub fn snap_render(display: u32, scale: f32) -> u32 {
    let d = display.max(8);
    let r = ((d as f32 * scale.clamp(0.0, 1.0)) / 8.0).round() as u32 * 8;
    r.clamp(8, d)
}

/// `ScaleScreenRect` extent: floor(w * s + 0.5).
pub fn scaled_extent(w: u32, s: f32) -> u32 {
    (w as f32 * s + 0.5).floor().max(1.0) as u32
}

/// FFX 3.x jitter phase count: int(8 * (display / render)^2).
pub fn phase_count(render_w: u32, display_w: u32) -> u32 {
    let r = display_w as f32 / render_w.max(1) as f32;
    ((8.0 * r * r) as u32).max(1)
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// FFX 3.x jitter offset (render pixels, [-0.5, 0.5)) for sequence `index`.
pub fn halton_offset(index: u32, phases: u32) -> [f32; 2] {
    let i = index % phases.max(1) + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

/// Engine table entry that makes the engine render FSR jitter `j` (render pixels):
/// `entry = (-jx * W / rW, +jy * H / rH)` (docs "Sign and units"; inverse of the reading
/// `(-jit.x * rW, +jit.y * rH)` with `jit = entry / W`).
pub fn table_entry(j: [f32; 2], display: [u32; 2], render: [u32; 2]) -> [f32; 2] {
    [
        -j[0] * display[0] as f32 / render[0].max(1) as f32,
        j[1] * display[1] as f32 / render[1].max(1) as f32,
    ]
}

/// Auto mip bias for an upscale ratio (FSR 2/3 guidance): log2(render / display) - 1.
pub fn auto_mip_bias(scale: f32) -> f32 {
    scale.clamp(0.01, 1.0).log2() - 1.0
}

/// Sampler options bitfield (`sub_1402D9300` / `CreateSampler`): bits 0-1 min, 2-3 mag,
/// 4-5 mip filter (0 linear, 1 point, 2 anisotropic), 6-7 address U (0 wrap), 12-14
/// comparison (0 none), bit 16 `forceNoMipLodBias`. Biased: linear or anisotropic mip
/// filtering, wrap addressing, no comparison, no `forceNoMipLodBias`.
pub fn sampler_takes_bias(bits: u32) -> bool {
    let mip = (bits >> 4) & 3;
    let address_u = (bits >> 6) & 3;
    let compare = (bits >> 12) & 7;
    mip != 1 && address_u == 0 && compare == 0 && bits & 0x1_0000 == 0
}

// ---------------------------------------------------------------------- state --

static INSTALLED: AtomicBool = AtomicBool::new(false);
static ORIG_SETUP: AtomicUsize = AtomicUsize::new(0);
static ORIG_UPLOAD: AtomicUsize = AtomicUsize::new(0);
static ORIG_POSTPROC: AtomicUsize = AtomicUsize::new(0);
static ORIG_TONEMAP: AtomicUsize = AtomicUsize::new(0);
static ORIG_SAMPLER: AtomicUsize = AtomicUsize::new(0);
static ORIG_CLEAR: AtomicUsize = AtomicUsize::new(0);
static ORIG_ASYNC_CLEAR: AtomicUsize = AtomicUsize::new(0);
/// Runs of the POSTPROC pass callback: the world is being rendered (menus and loading
/// screens do not run it). `fsr::tick` warns about a missing TAA pass only then.
pub static POSTPROC_CALLS: AtomicU64 = AtomicU64::new(0);
/// The post-chain hooks (PerViewCB_Upload, POSTPROC, POST_TONEMAP) are installed: upscale
/// is possible.
static POST_HOOKS: AtomicBool = AtomicBool::new(false);
/// The jitter table currently holds FSR phases (restored when FSR stops).
static TABLE_OURS: AtomicBool = AtomicBool::new(false);
static TABLE_ORIGINAL: Mutex<Option<[u8; 64]>> = Mutex::new(None);
/// `m_justTeleported` seen for view frame `value - 1` (0 = never).
static TELEPORT_FRAME: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicU32 = AtomicU32::new(0);

fn log_once(bit: u32, msg: &str) {
    if LOGGED.fetch_or(1 << bit, Ordering::Relaxed) & (1 << bit) == 0 {
        log::line(msg);
    }
}

/// Jitter offsets reported by the FFX provider for (render width, display width).
struct JitterSeq {
    key: (u32, u32),
    offsets: Vec<[f32; 2]>,
}
static JITTER_SEQ: Mutex<Option<JitterSeq>> = Mutex::new(None);

/// Stores the provider's sequence (called by the FSR runtime, which owns the context).
pub(crate) fn set_jitter_sequence(render_w: u32, display_w: u32, offsets: Vec<[f32; 2]>) {
    if let Ok(mut s) = JITTER_SEQ.lock() {
        *s = Some(JitterSeq {
            key: (render_w, display_w),
            offsets,
        });
    }
}

/// True if the stored sequence is for these widths and has `phases` entries.
pub(crate) fn jitter_sequence_known(render_w: u32, display_w: u32, phases: usize) -> bool {
    JITTER_SEQ
        .lock()
        .ok()
        .and_then(|s| {
            s.as_ref()
                .map(|s| s.key == (render_w, display_w) && s.offsets.len() == phases)
        })
        .unwrap_or(false)
}

/// Phase `frame` of the FSR sequence for (render, display) widths.
fn jitter_for(frame: u32, render_w: u32, display_w: u32) -> [f32; 2] {
    if let Ok(s) = JITTER_SEQ.try_lock()
        && let Some(s) = s.as_ref()
        && s.key == (render_w, display_w)
        && !s.offsets.is_empty()
    {
        return s.offsets[frame as usize % s.offsets.len()];
    }
    halton_offset(frame, phase_count(render_w, display_w))
}

/// True if `m_justTeleported` was set when the view of frame `view_frame` was set up.
pub(crate) fn teleported(view_frame: u32) -> bool {
    TELEPORT_FRAME.load(Ordering::Relaxed) == u64::from(view_frame) + 1
}

// ------------------------------------------------------------------- view copies --

struct Slot {
    mem: Box<[u8]>,
    real: u64,
    frame: u32,
    switched: bool,
}

struct Ring {
    slots: Vec<Slot>,
    next: usize,
}

impl Ring {
    fn slot_at(&mut self, addr: u64) -> Option<&mut Slot> {
        self.slots
            .iter_mut()
            .find(|s| s.mem.as_ptr() as u64 == addr)
    }
}

static RING: Mutex<Option<Ring>> = Mutex::new(None);
/// Views whose portal CB was filled as twin this frame: (view, view frame).
static TWINS: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

fn ring() -> std::sync::MutexGuard<'static, Option<Ring>> {
    RING.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn view_scale(view: u64) -> Option<[f32; 2]> {
    mem::f32s::<2>(view + VIEW_RES_SCALE)
}

fn is_scaled(view: u64) -> bool {
    view_scale(view).is_some_and(|s| s[0] < 0.9999 || s[1] < 0.9999)
}

fn portal(view: u64) -> bool {
    let mut b = [0u8];
    !mem::read_into(view + VIEW_FLAGS_14900, &mut b) || b[0] & 1 != 0
}

/// Upscale mode configured, its post-chain hooks installed and the main menu not shown
/// (the menu renders at native resolution with FSR native AA, see [`MENU_VIEW`]).
fn upscale_mode() -> bool {
    let s = settings::get();
    s.fsr && s.fsr_upscale && POST_HOOKS.load(Ordering::Relaxed) && !in_menu()
}

/// Map folder of the main-menu scene (`maps/campaign/menu/menu_p`, view summary in the log).
const MENU_MAP_PREFIX: &str = "maps/campaign/menu/";
/// The render world of the last frame is the main-menu scene. Upscaling there showed the
/// two menu characters twice while the camera follows the mouse (user report 2026-10-18;
/// cause unknown, see docs/fsr4-integration.md "Known gaps"), so the menu keeps native AA.
static MENU_VIEW: AtomicBool = AtomicBool::new(false);
/// World pointer whose map name `MENU_VIEW` was derived from (re-read only when it changes).
static MENU_WORLD: AtomicU64 = AtomicU64::new(0);

fn in_menu() -> bool {
    MENU_VIEW.load(Ordering::Relaxed)
}

/// Main thread, per frame: re-derives [`MENU_VIEW`] when the render world changes (one
/// pointer read per frame; the map name is read only on a change).
fn update_menu() {
    let world = crate::rt::live::world(crate::hook::last_requests()).unwrap_or(0);
    if MENU_WORLD.swap(world, Ordering::Relaxed) == world {
        return;
    }
    let menu = world != 0 && crate::rt::live::map_name(world).starts_with(MENU_MAP_PREFIX);
    if MENU_VIEW.swap(menu, Ordering::Relaxed) != menu {
        log::line(if menu {
            "fsr: main-menu scene: native AA while it is shown (upscale resumes in game)"
        } else {
            "fsr: left the main-menu scene"
        });
    }
}

/// Render size the engine is asked for while FSR runs at `display`: upscale = `fsr_scale`
/// snapped to multiples of 8, native = display.
pub(crate) fn wanted_render(display: [u32; 2]) -> [u32; 2] {
    if upscale_mode() {
        let s = settings::get().fsr_scale;
        [snap_render(display[0], s), snap_render(display[1], s)]
    } else {
        display
    }
}

/// A post-chain switch prepared at the TAA site: the pass runs on ring copy `copy` of
/// view `real`.
pub(crate) struct Switch {
    copy: u64,
    real: u64,
}

/// Preconditions for taking a scaled frame (the caller falls back to the engine TAA at
/// render scale, which upscales itself, when this fails): the pass runs on our copy and
/// the PerViewCB twin of its view was filled this frame. `Ok(None)` = not scaled.
pub(crate) fn prepare_switch(pass: u64) -> Result<Option<Switch>, &'static str> {
    let view = mem::u64_at(pass + PASS_VIEW).ok_or("pass view")?;
    if !is_scaled(view) {
        return Ok(None);
    }
    let mut r = ring();
    let slot = r
        .as_mut()
        .and_then(|r| r.slot_at(view))
        .ok_or("scaled view without a private copy")?;
    let (real, frame) = (slot.real, slot.frame);
    let twin = TWINS
        .lock()
        .map(|t| t.contains(&(real, frame)))
        .unwrap_or(false);
    if !twin {
        return Err("no PerViewCB twin for this view/frame");
    }
    Ok(Some(Switch { copy: view, real }))
}

/// Resources of the POSTPROC pass needed for the switch (`distoVectors`).
pub(crate) fn disto_vectors(pass: u64) -> Option<ID3D11Resource> {
    pass_image(pass, HASH_DISTO_VECTORS)
}

fn write_scale_params(ctx: u64, real: u64) -> bool {
    let mut ok = true;
    if let Ok(s) = fsr::slot(ctx, H_PERVIEW_CB) {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&(real + VIEW_PORTAL_CB).to_le_bytes());
        ok &= mem::write_bytes(s, &b);
    } else {
        ok = false;
    }
    for h in [H_RES_SCALE, H_PREV_RES_SCALE] {
        match fsr::slot(ctx, h) {
            Ok(s) => ok &= mem::write_bytes(s, &vec2_param(1.0, 1.0)),
            Err(_) => ok = false,
        }
    }
    ok & write_viewport_params(ctx, real)
}

/// `SetPerViewShaderParms` also stores two viewport parameters per pass context from
/// `m_renderWidth/Height` (`sub_140310670`, slot indices at block +74704 / +74708):
/// `(x/w, y/h, 1/w, 1/h)` and `(x, y, w, h)` with x, y = `m_subrect_viewport` origin.
/// After the switch they must describe the display size.
fn write_viewport_params(ctx: u64, real: u64) -> bool {
    let block = ctx + 0x10;
    let (Some(i1), Some(i2), Some(display)) = (
        mem::u32_at(block + 74_704),
        mem::u32_at(block + 74_708),
        fsr::rect_size(real + VIEW_UNSCALED_VIEWPORT),
    ) else {
        return false;
    };
    if i1 >= 4096 || i2 >= 4096 {
        return false;
    }
    let (x, y) = (
        mem::u32_at(real + VIEW_SUBRECT).unwrap_or(0) as i32 as f32,
        mem::u32_at(real + VIEW_SUBRECT + 4).unwrap_or(0) as i32 as f32,
    );
    let (w, h) = (display[0] as f32, display[1] as f32);
    let pack = |v: [f32; 4]| {
        let mut b = [0u8; 16];
        for (k, f) in v.iter().enumerate() {
            b[4 * k..4 * k + 4].copy_from_slice(&f.to_le_bytes());
        }
        b
    };
    mem::write_bytes(
        block + 16 * (u64::from(i1) + 1),
        &pack([x / w, y / h, 1.0 / w, 1.0 / h]),
    ) && mem::write_bytes(block + 16 * (u64::from(i2) + 1), &pack([x, y, w, h]))
}

/// `sub_1403109E0` layout of a vec2 parameter: (x, y, 0, 1).
fn vec2_param(x: f32, y: f32) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&x.to_le_bytes());
    b[4..8].copy_from_slice(&y.to_le_bytes());
    b[12..16].copy_from_slice(&1.0f32.to_le_bytes());
    b
}

/// After FSR took the frame (record time, POSTPROC worker): switch the copy to display
/// size and point the pass's parameters at the twin CB.
pub(crate) fn apply_switch(ctx: u64, sw: &Switch) -> bool {
    let mut r = ring();
    let Some(slot) = r.as_mut().and_then(|r| r.slot_at(sw.copy)) else {
        return false;
    };
    let mut twin = [0u8; CB_OBJECT];
    if !mem::read_into(sw.real + VIEW_PORTAL_CB, &mut twin) {
        return false;
    }
    let m = &mut slot.mem;
    let copy24 = |m: &mut [u8], to: u64, from: u64| {
        let (t, f) = (to as usize, from as usize);
        m.copy_within(f..f + 24, t);
    };
    copy24(m, VIEW_VIEWPORT, VIEW_UNSCALED_VIEWPORT);
    copy24(m, VIEW_SCISSOR, VIEW_UNSCALED_SCISSOR);
    let one = 1.0f32.to_le_bytes();
    let rs = VIEW_RES_SCALE as usize;
    for k in 0..4 {
        m[rs + 4 * k..rs + 4 * k + 4].copy_from_slice(&one);
    }
    // m_renderWidth/Height = unscaled extents.
    let i32_at = |m: &[u8], o: usize| i32::from_le_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]]);
    let vp = VIEW_UNSCALED_VIEWPORT as usize;
    let (w, h) = (
        i32_at(m, vp + 8) - i32_at(m, vp) + 1,
        i32_at(m, vp + 12) - i32_at(m, vp + 4) + 1,
    );
    let rw = VIEW_RENDER_WH as usize;
    m[rw..rw + 4].copy_from_slice(&w.to_le_bytes());
    m[rw + 4..rw + 8].copy_from_slice(&h.to_le_bytes());
    let cb = VIEW_PERVIEW_CB as usize;
    m[cb..cb + CB_OBJECT].copy_from_slice(&twin);
    slot.switched = true;
    drop(r);
    write_scale_params(ctx, sw.real)
}

// -------------------------------------------------------------------- detours --

type SetupScaledFn = unsafe extern "system" fn(u64, u64, u64, u64) -> u64;
type PassFn = unsafe extern "system" fn(u64) -> u64;
type UploadFn = unsafe extern "system" fn(u64, u64) -> u64;
type SamplerFn = unsafe extern "system" fn(u64, u64, u64) -> u64;
type FillFn = unsafe extern "system" fn(u64, u64, u64);

fn orig<T: Copy>(slot: &AtomicUsize) -> T {
    let p = slot.load(Ordering::Relaxed);
    // SAFETY: only called after the trampoline was stored (hooks are enabled after the
    // store); `T` is the function-pointer type matching the hooked function.
    unsafe { std::mem::transmute_copy::<usize, T>(&p) }
}

unsafe extern "system" fn setup_scaled(view: u64, rect: u64, scale: u64, sub: u64) -> u64 {
    let _ = std::panic::catch_unwind(|| before_setup(view, rect, scale));
    let f: SetupScaledFn = orig(&ORIG_SETUP);
    // SAFETY: trampoline of `idRenderView_SetupScaled`, arguments unchanged.
    unsafe { f(view, rect, scale, sub) }
}

fn before_setup(view: u64, rect: u64, scale: u64) {
    if scale != base() + RES_SCALE_RVA {
        return;
    }
    let Some(frame) = mem::u32_at(base() + fsr::RS_RVA + fsr::RS_FRAME) else {
        return;
    };
    let mut t = [0u8];
    if mem::read_into(view + VIEW_TELEPORTED, &mut t) && t[0] != 0 {
        TELEPORT_FRAME.store(u64::from(frame) + 1, Ordering::Relaxed);
    }
    let Some(display) = fsr::rect_size(rect) else {
        return;
    };
    if !fsr::running_for(display) {
        restore_table();
        return;
    }
    // Per frame, so `fsr_mode` / `fsr_scale` changes apply at once (the engine's
    // `ResScale_ApplyCvars` recomputes the value every frame, so nothing needs undoing).
    let r = wanted_render(display);
    let f = [
        r[0] as f32 / display[0] as f32,
        r[1] as f32 / display[1] as f32,
    ];
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&f[0].to_le_bytes());
    b[4..].copy_from_slice(&f[1].to_le_bytes());
    if !mem::write_bytes(scale, &b) {
        log_once(0, "fsr: g_resolutionScale not writable; engine scale kept");
    }
    let used = mem::f32s::<2>(scale).unwrap_or([1.0, 1.0]);
    let render = [
        scaled_extent(display[0], used[0]).min(display[0]),
        scaled_extent(display[1], used[1]).min(display[1]),
    ];
    let j = jitter_for(frame, render[0], display[0]);
    let e = table_entry(j, display, render);
    let mut table = [0u8; 64];
    for k in 0..8 {
        table[8 * k..8 * k + 4].copy_from_slice(&e[0].to_le_bytes());
        table[8 * k + 4..8 * k + 8].copy_from_slice(&e[1].to_le_bytes());
    }
    if mem::write_bytes(base() + JITTER_TABLE_RVA, &table) {
        TABLE_OURS.store(true, Ordering::Relaxed);
    } else {
        log_once(1, "fsr: jitter table not writable; engine jitter kept");
    }
}

fn restore_table() {
    if TABLE_OURS.swap(false, Ordering::Relaxed)
        && let Ok(t) = TABLE_ORIGINAL.lock()
        && let Some(t) = t.as_ref()
    {
        let _ = mem::write_bytes(base() + JITTER_TABLE_RVA, t);
    }
}

unsafe extern "system" fn perview_upload(a1: u64, view: u64) -> u64 {
    let done = std::panic::catch_unwind(|| upload_twin(view)).unwrap_or(false);
    if done {
        return 0;
    }
    let f: UploadFn = orig(&ORIG_UPLOAD);
    // SAFETY: trampoline of `PerViewCB_Upload`, arguments unchanged.
    unsafe { f(a1, view) }
}

/// Re-implements `PerViewCB_Upload` for a scaled non-portal view (its portal branch is
/// skipped exactly as in the original): PerViewCB as is, then the portal CB with the
/// `resolutionScale` parameter set to (1, 1).
fn upload_twin(view: u64) -> bool {
    if !upscale_mode() || !is_scaled(view) || portal(view) {
        return false;
    }
    let b = base();
    let (Some(ctx), Some(block)) = (
        mem::u64_at(b + IMMEDIATE_ENGINE_CTX_RVA),
        mem::u64_at(b + PARAM_BLOCK_RVA),
    ) else {
        return false;
    };
    // The global block uses the per-context slot layout (`block + 16 * (index + 1)`).
    let Ok(rs_slot) = fsr::slot(block.wrapping_sub(0x10), H_RES_SCALE) else {
        return false;
    };
    let mut saved = [0u8; 16];
    if !mem::read_into(rs_slot, &mut saved) {
        return false;
    }
    let fill: FillFn =
        // SAFETY: address of `sub_1402AF360` in the verified executable.
        unsafe { std::mem::transmute::<usize, FillFn>((b + FILL_CB_RVA) as usize) };
    // SAFETY: the engine's own arguments for this function (as in `PerViewCB_Upload`).
    unsafe { fill(view + VIEW_PERVIEW_CB, ctx, block) };
    if mem::write_bytes(rs_slot, &vec2_param(1.0, 1.0)) {
        // SAFETY: as above, for the portal CB object.
        unsafe { fill(view + VIEW_PORTAL_CB, ctx, block) };
        let _ = mem::write_bytes(rs_slot, &saved);
        if let (Some(frame), Ok(mut t)) = (mem::u32_at(view + VIEW_FRAME), TWINS.lock()) {
            if t.len() > 16 {
                t.drain(..8);
            }
            t.push((view, frame));
        }
    } else {
        // SAFETY: as above (plain twin = original behaviour).
        unsafe { fill(view + VIEW_PORTAL_CB, ctx, block) };
    }
    true
}

unsafe extern "system" fn postproc(pass: u64) -> u64 {
    POSTPROC_CALLS.fetch_add(1, Ordering::Relaxed);
    let swapped = std::panic::catch_unwind(|| enter_postproc(pass)).unwrap_or(None);
    let f: PassFn = orig(&ORIG_POSTPROC);
    // SAFETY: trampoline of the POSTPROC callback.
    let r = unsafe { f(pass) };
    if let Some(real) = swapped {
        let _ = mem::write_bytes(pass + PASS_VIEW, &real.to_le_bytes());
    }
    r
}

/// Copies the view into the next ring slot and points the pass at it.
fn enter_postproc(pass: u64) -> Option<u64> {
    if !upscale_mode() || !fsr::running() {
        return None;
    }
    let real = mem::u64_at(pass + PASS_VIEW)?;
    if !is_scaled(real) || portal(real) {
        return None;
    }
    let frame = mem::u32_at(real + VIEW_FRAME)?;
    let mut g = ring();
    let r = g.get_or_insert_with(|| Ring {
        slots: (0..3)
            .map(|_| Slot {
                mem: vec![0u8; VIEW_SIZE].into_boxed_slice(),
                real: 0,
                frame: 0,
                switched: false,
            })
            .collect(),
        next: 0,
    });
    let i = r.next;
    r.next = (i + 1) % r.slots.len();
    let slot = &mut r.slots[i];
    if !mem::read_into(real, &mut slot.mem) {
        return None;
    }
    slot.real = real;
    slot.frame = frame;
    slot.switched = false;
    let copy = slot.mem.as_ptr() as u64;
    mem::write_bytes(pass + PASS_VIEW, &copy.to_le_bytes()).then_some(real)
}

unsafe extern "system" fn post_tonemap(pass: u64) -> u64 {
    let swapped = std::panic::catch_unwind(|| enter_tonemap(pass)).unwrap_or(None);
    let f: PassFn = orig(&ORIG_TONEMAP);
    // SAFETY: trampoline of the POST_TONEMAP callback.
    let r = unsafe { f(pass) };
    if let Some(real) = swapped {
        let _ = mem::write_bytes(pass + PASS_VIEW, &real.to_le_bytes());
    }
    r
}

fn enter_tonemap(pass: u64) -> Option<u64> {
    if !upscale_mode() {
        return None;
    }
    let real = mem::u64_at(pass + PASS_VIEW)?;
    let frame = mem::u32_at(real + VIEW_FRAME)?;
    let copy = {
        let g = ring();
        g.as_ref().and_then(|r| {
            r.slots
                .iter()
                .find(|s| s.switched && s.real == real && s.frame == frame)
                .map(|s| s.mem.as_ptr() as u64)
        })
    };
    let Some(copy) = copy else {
        if fsr::running() && is_scaled(real) {
            log_once(
                3,
                "fsr: POST_TONEMAP recorded without a switched POSTPROC for its frame (scaled frame not taken by FSR, or recorded first); it stays at render scale",
            );
        }
        return None;
    };
    let ctx = mem::u64_at(pass + PASS_CTX)?;
    if !mem::write_bytes(pass + PASS_VIEW, &copy.to_le_bytes()) {
        return None;
    }
    if !write_scale_params(ctx, real) {
        log_once(2, "fsr: POST_TONEMAP parameters not writable");
    }
    if settings::get().fsr_depth_resample
        && let Some(z) = pass_image(pass, HASH_ZBUFFER)
        && let (Some(render), Some(display)) = (
            fsr::rect_size(real + VIEW_VIEWPORT),
            fsr::rect_size(real + VIEW_UNSCALED_VIEWPORT),
        )
    {
        fsr::depth_resample(
            ctx,
            pass,
            fsr::DepthJob {
                target: z,
                render,
                display,
            },
        );
    }
    Some(real)
}

/// `CreateSampler` returns the sampler's cell (`rax` on both paths: lookup hit at
/// 0x1403782F4, new sampler at 0x1403785C7); the options are left untouched (the engine's
/// lookup compares them), the bias is applied to the cell afterwards.
unsafe extern "system" fn create_sampler(mgr: u64, opts: u64, name: u64) -> u64 {
    let f: SamplerFn = orig(&ORIG_SAMPLER);
    // SAFETY: trampoline of `CreateSampler`, arguments unchanged.
    let cell = unsafe { f(mgr, opts, name) };
    let _ = std::panic::catch_unwind(|| {
        let mut o = [0u8; 24];
        if cell != 0 && mem::read_into(opts, &mut o) {
            samplers::note(
                cell,
                u32::from_le_bytes([o[0], o[1], o[2], o[3]]),
                f32::from_le_bytes([o[20], o[21], o[22], o[23]]),
            );
        }
    });
    cell
}

/// Live mip bias of the engine's material samplers (docs/fsr4-integration.md "Mip
/// bias", "Implementation status").
///
/// O (IDB): `idTextureSamplerManagerLocal` is the static object 0x1422FDA20 (vtable
/// 0x141B37668, slot 6 `CreateSampler`, slot 3 find-by-name, slot 4 cell -> name). Its
/// sampler list is an `idList` at +8 (array pointer +8, count in the low 24 bits of +16)
/// of 56-byte records: +0 name, +8 `samplerOptions_t` (bits u32 at +8, `MipLODBias` f32
/// at +28), +48 *cell* = pointer to a separately allocated 8-byte slot holding the
/// `ID3D11SamplerState*` (`sub_1403778A0` allocates it, `CreateSamplerState` writes it).
/// Records are only appended (no erase in the vtable; the destructor `sub_140377B50` runs
/// at exit), so cells stay valid for the whole run while the array itself may move.
/// Material parameter blocks store the cell (`idParmBlock::Load` `sub_140343090` stores
/// the find-by-name result) and `Ctx_BindProgram` loads `*cell` at every bind
/// (0x1402C3A53 `mov rax,[r13+rcx*8+10h]`, 0x1402C3A58 `mov rax,[rax]`).
///
/// So a bias change re-creates each qualifying sampler with `MipLODBias = options bias +
/// bias` and swaps it into the cell (8-byte compare-exchange; the cell owns one reference,
/// as with the engine's object). Swapped-out objects are released [`GRACE_FRAMES`] frames
/// later: a recording thread may have read the old pointer just before the swap (command
/// lists reference what they bind, the grace covers the read-to-bind window).
pub(crate) mod samplers {
    use std::ffi::c_void;
    use std::sync::Mutex;

    use windows::Win32::Graphics::Direct3D11::{D3D11_SAMPLER_DESC, ID3D11SamplerState};
    use windows::core::Interface;

    use crate::{log, mem};

    /// Engine sampler manager (`idTextureSamplerManagerLocal`, static object).
    const MANAGER_RVA: u64 = 0x22F_DA20;
    const RECORD: u64 = 56;
    pub(crate) const GRACE_FRAMES: u64 = 8;
    /// A requested bias must stay the same this many frames before it is applied (slider
    /// drags would otherwise re-create every sampler per frame).
    const SETTLE_FRAMES: u32 = 8;

    struct Tracked {
        cell: u64,
        bits: u32,
        base_bias: f32,
    }

    struct State {
        tracked: Vec<Tracked>,
        /// Bias currently in the cells (0 = engine originals).
        applied: f32,
        /// (frame, object) swapped out, released after `GRACE_FRAMES`.
        graveyard: Vec<(u64, ID3D11SamplerState)>,
        pending: f32,
        pending_frames: u32,
        frame: u64,
        biased: usize,
    }

    // SAFETY: D3D11 sampler states are free-threaded; the state is used under `STATE`.
    unsafe impl Send for State {}

    static STATE: Mutex<State> = Mutex::new(State {
        tracked: Vec::new(),
        applied: 0.0,
        graveyard: Vec::new(),
        pending: 0.0,
        pending_frames: 0,
        frame: 0,
        biased: 0,
    });

    fn state() -> std::sync::MutexGuard<'static, State> {
        STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// (applied bias, biased samplers, tracked samplers) for the overlay.
    pub(crate) fn status() -> (f32, usize, usize) {
        STATE
            .try_lock()
            .map(|s| (s.applied, s.biased, s.tracked.len()))
            .unwrap_or((0.0, 0, 0))
    }

    /// True if `cell` holds a pointer to an object whose vtable lies in d3d11.dll (guards
    /// the install-time walk against a list array being moved concurrently).
    fn plausible_cell(cell: u64) -> bool {
        use windows::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleHandleExW, GetModuleHandleW,
        };
        let Some(vtbl) = mem::u64_at(cell).filter(|&o| o != 0).and_then(mem::u64_at) else {
            return false;
        };
        let mut m = windows::Win32::Foundation::HMODULE::default();
        // SAFETY: address lookups only; no reference is taken (UNCHANGED_REFCOUNT).
        unsafe {
            let Ok(d3d11) = GetModuleHandleW(windows::core::w!("d3d11.dll")) else {
                return false;
            };
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                windows::core::PCWSTR(vtbl as *const u16),
                &mut m,
            )
            .is_ok()
                && m == d3d11
        }
    }

    /// Registers the samplers created before the hook (fsr turned on at run time).
    pub(super) fn walk_existing(base: u64) -> usize {
        let mgr = base + MANAGER_RVA;
        let (Some(arr), Some(count)) = (mem::u64_at(mgr + 8), mem::u32_at(mgr + 16)) else {
            return 0;
        };
        let mut n = 0;
        for i in 0..u64::from((count & 0xFF_FFFF).min(4096)) {
            let r = arr + RECORD * i;
            let (Some(bits), Some(bias), Some(cell)) = (
                mem::u32_at(r + 8),
                mem::f32s::<1>(r + 28),
                mem::u64_at(r + 48),
            ) else {
                break;
            };
            if cell != 0 && plausible_cell(cell) {
                note(cell, bits, bias[0]);
                n += 1;
            }
        }
        n
    }

    /// A sampler from `CreateSampler` (or the walk): tracked once; biased at once when a
    /// bias is active.
    pub(crate) fn note(cell: u64, bits: u32, base_bias: f32) {
        let mut s = state();
        if s.tracked.iter().any(|t| t.cell == cell) {
            return;
        }
        let t = Tracked {
            cell,
            bits,
            base_bias: if base_bias.is_finite() {
                base_bias
            } else {
                0.0
            },
        };
        let bias = s.applied;
        if bias != 0.0 && super::sampler_takes_bias(bits) {
            let frame = s.frame;
            if let Ok(Some(old)) = set_cell(&t, bias) {
                s.graveyard.push((frame, old));
                s.biased += 1;
            }
        }
        s.tracked.push(t);
    }

    /// Stops tracking `cell` (offline check: its stand-in cell is freed afterwards; engine
    /// cells live for the whole run and are never forgotten).
    pub(crate) fn forget(cell: u64) {
        state().tracked.retain(|t| t.cell != cell);
    }

    /// Re-creates the cell's sampler with `base_bias + bias` and swaps it in; returns the
    /// swapped-out object (now owned by the caller).
    fn set_cell(t: &Tracked, bias: f32) -> Result<Option<ID3D11SamplerState>, String> {
        let cell = t.cell;
        let Some(cur) = mem::u64_at(cell).filter(|&c| c != 0) else {
            return Ok(None);
        };
        let raw = cur as *mut c_void;
        // SAFETY: `cur` is the live sampler object owned by the engine's cell (checked
        // readable above); borrowed without taking a reference.
        let Some(s) = (unsafe { ID3D11SamplerState::from_raw_borrowed(&raw) }) else {
            return Ok(None);
        };
        let mut desc = D3D11_SAMPLER_DESC::default();
        // SAFETY: plain getters / creation on live objects with valid out pointers.
        let new = unsafe {
            s.GetDesc(&mut desc);
            desc.MipLODBias = (t.base_bias + bias).clamp(-16.0, 15.99);
            let device = s.GetDevice().map_err(|e| format!("GetDevice: {e}"))?;
            let mut new = None;
            device
                .CreateSamplerState(&desc, Some(&mut new))
                .map_err(|e| format!("CreateSamplerState: {e}"))?;
            new.ok_or("no sampler")?
        };
        let new_raw = new.into_raw() as u64;
        // SAFETY: the cell is an engine heap slot that lives for the whole run (see the
        // module docs); 8-byte aligned, written atomically.
        let slot = unsafe { &*(cell as *const std::sync::atomic::AtomicU64) };
        match slot.compare_exchange(
            cur,
            new_raw,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        ) {
            // SAFETY: the cell's reference to `cur` is now ours.
            Ok(_) => Ok(Some(unsafe { ID3D11SamplerState::from_raw(raw) })),
            Err(_) => {
                // SAFETY: our own reference from `CreateSamplerState`.
                drop(unsafe { ID3D11SamplerState::from_raw(new_raw as *mut c_void) });
                Ok(None)
            }
        }
    }

    /// Main thread, once per frame: applies `target` once it is settled, releases expired
    /// objects. Returns `Some((bias, samplers))` when a new bias was applied.
    pub(crate) fn frame(target: f32) -> Option<(f32, usize)> {
        let mut s = state();
        s.frame += 1;
        let frame = s.frame;
        s.graveyard
            .retain(|(f, _)| frame.saturating_sub(*f) < GRACE_FRAMES);
        let target = if target.is_finite() { target } else { 0.0 };
        if target.to_bits() != s.pending.to_bits() {
            s.pending = target;
            s.pending_frames = 0;
        }
        s.pending_frames = s.pending_frames.saturating_add(1);
        if s.pending_frames < SETTLE_FRAMES || target.to_bits() == s.applied.to_bits() {
            return None;
        }
        let (mut done, mut failed) = (0usize, None);
        let mut old = Vec::new();
        for t in s
            .tracked
            .iter()
            .filter(|t| super::sampler_takes_bias(t.bits))
        {
            match set_cell(t, target) {
                Ok(Some(o)) => {
                    old.push((frame, o));
                    done += 1;
                }
                Ok(None) => {}
                Err(e) => failed = Some(e),
            }
        }
        s.graveyard.extend(old);
        s.applied = target;
        s.biased = if target == 0.0 { 0 } else { done };
        let total = s.tracked.len();
        drop(s);
        log::line(&format!(
            "fsr: mip bias {target:+.3} set on {done} of {total} tracked samplers{}",
            failed.map_or_else(String::new, |e| format!(" (last error: {e})"))
        ));
        Some((target, done))
    }
}

// ----------------------------------------------------------------- OIT targets --

struct Targets {
    opacity: Option<ID3D11Resource>,
    hair: Option<ID3D11Resource>,
}
// SAFETY: D3D11 resources are free-threaded.
unsafe impl Send for Targets {}
static OIT: Mutex<Targets> = Mutex::new(Targets {
    opacity: None,
    hair: None,
});

/// `OpacityBuffer` and `hairOpacity` images noted from the clear passes.
pub(crate) fn oit_targets() -> (Option<ID3D11Resource>, Option<ID3D11Resource>) {
    OIT.lock()
        .map(|t| (t.opacity.clone(), t.hair.clone()))
        .unwrap_or((None, None))
}

/// Image resource with name hash `hash` in a pass's read or write list
/// (`{u32 hash, pad, arkImageView*}` x count).
pub(crate) fn pass_image(pass: u64, hash: u32) -> Option<ID3D11Resource> {
    for list in [PASS_READS, PASS_WRITES] {
        let (Some(ptr), Some(count)) = (mem::u64_at(pass + list), mem::u32_at(pass + list + 8))
        else {
            continue;
        };
        for i in 0..u64::from((count & 0xFF_FFFF).min(256)) {
            if mem::u32_at(ptr + 16 * i) == Some(hash) {
                let view = mem::u64_at(ptr + 16 * i + 8)?;
                return fsr::view_resource(view);
            }
        }
    }
    None
}

fn note_oit(pass: u64) {
    if !settings::get().fsr_masks {
        return;
    }
    let (o, h) = (
        pass_image(pass, HASH_OPACITY),
        pass_image(pass, HASH_HAIR_OPACITY),
    );
    if o.is_none() && h.is_none() {
        return;
    }
    if let Ok(mut t) = OIT.lock() {
        if o.is_some() {
            t.opacity = o;
        }
        if h.is_some() {
            t.hair = h;
        }
    }
}

unsafe extern "system" fn clear_pass(pass: u64) -> u64 {
    let _ = std::panic::catch_unwind(|| note_oit(pass));
    let f: PassFn = orig(&ORIG_CLEAR);
    // SAFETY: trampoline of the CLEAR callback.
    unsafe { f(pass) }
}

unsafe extern "system" fn async_clear_pass(pass: u64) -> u64 {
    let _ = std::panic::catch_unwind(|| note_oit(pass));
    let f: PassFn = orig(&ORIG_ASYNC_CLEAR);
    // SAFETY: trampoline of the ASYNC_CLEAR callback.
    unsafe { f(pass) }
}

// ---------------------------------------------------------------- installation --

fn hook(rva: u64, head: &[u8; 16], detour: *mut c_void, slot: &AtomicUsize) -> Result<(), String> {
    let target = base() + rva;
    let mut cur = [0u8; 16];
    if !mem::read_into(target, &mut cur) {
        return Err(format!("{target:#x} not readable"));
    }
    if &cur != head {
        return Err(format!("unexpected prologue {cur:02x?} at {target:#x}"));
    }
    // SAFETY: `target` starts the named function of the verified executable (prologue
    // checked); `detour` has its calling convention and signature.
    let tramp = unsafe { minhook::MinHook::create_hook(target as *mut c_void, detour) }
        .map_err(|e| format!("create_hook {target:#x}: {e:?}"))?;
    slot.store(tramp as usize, Ordering::Relaxed);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook {target:#x}: {e:?}"))
}

/// Installs every FSR hook (from `fsr::install`, at startup or at run time); each one is
/// inert unless its feature is active for the frame. A failed hook disables only its
/// feature (missing post-chain hooks: upscale unavailable, native AA still works).
pub fn install() {
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    let b = base();
    // Jitter table: keep the original bytes, make the page writable.
    let mut t = [0u8; 64];
    let table_ok = mem::read_into(b + JITTER_TABLE_RVA, &mut t)
        && t[..8] == hex!("00 00 80 3D 00 00 40 BE")
        && mem::make_writable(b + JITTER_TABLE_RVA, 64).is_ok();
    if table_ok && let Ok(mut o) = TABLE_ORIGINAL.lock() {
        *o = Some(t);
    }
    let mut report = Vec::new();
    let mut try_hook = |name: &str, rva, head, detour, slot: &AtomicUsize| -> bool {
        match hook(rva, head, detour, slot) {
            Ok(()) => {
                report.push(format!("{name} {:#x}", b + rva));
                true
            }
            Err(e) => {
                log::line(&format!("fsr: {name} hook failed ({e}); feature off"));
                false
            }
        }
    };
    if table_ok {
        try_hook(
            "SetupScaled",
            SETUP_SCALED_RVA,
            &SETUP_SCALED_HEAD,
            setup_scaled as *mut c_void,
            &ORIG_SETUP,
        );
    } else {
        log::line(
            "fsr: jitter table not as expected / not writable; engine jitter and native scale kept",
        );
    }
    let post = try_hook(
        "PerViewCB_Upload",
        PERVIEW_UPLOAD_RVA,
        &PERVIEW_UPLOAD_HEAD,
        perview_upload as *mut c_void,
        &ORIG_UPLOAD,
    ) & try_hook(
        "POSTPROC",
        POSTPROC_RVA,
        &POST_HEAD,
        postproc as *mut c_void,
        &ORIG_POSTPROC,
    ) & try_hook(
        "POST_TONEMAP",
        POST_TONEMAP_RVA,
        &POST_HEAD,
        post_tonemap as *mut c_void,
        &ORIG_TONEMAP,
    );
    // Upscale needs the post-chain switch (and the scale writer).
    POST_HOOKS.store(post && table_ok, Ordering::Relaxed);
    let sampler_hook = try_hook(
        "CreateSampler",
        CREATE_SAMPLER_RVA,
        &CREATE_SAMPLER_HEAD,
        create_sampler as *mut c_void,
        &ORIG_SAMPLER,
    );
    // Samplers created before the hook (FSR installed at run time).
    let existing = if sampler_hook {
        samplers::walk_existing(b)
    } else {
        0
    };
    try_hook(
        "CLEAR",
        CLEAR_RVA,
        &CLEAR_HEAD,
        clear_pass as *mut c_void,
        &ORIG_CLEAR,
    );
    try_hook(
        "ASYNC_CLEAR",
        ASYNC_CLEAR_RVA,
        &ASYNC_CLEAR_HEAD,
        async_clear_pass as *mut c_void,
        &ORIG_ASYNC_CLEAR,
    );
    log::line(&format!(
        "fsr: hooks: {}; {existing} existing samplers tracked{}",
        report.join(", "),
        if POST_HOOKS.load(Ordering::Relaxed) {
            ""
        } else {
            "; post-chain hooks missing: upscale unavailable (native AA)"
        }
    ));
}

/// Main thread, once per frame (from `fsr::tick`, FSR installed): applies the live mip
/// bias and announces mode / render-size changes.
pub(crate) fn frame() {
    static LAST_MODE: Mutex<Option<(bool, [u32; 2])>> = Mutex::new(None);
    let s = settings::get();
    let display = fsr::ready_display().filter(|_| fsr::running());
    update_menu();
    let up = upscale_mode();
    let target = match display {
        Some(d) if up => s.fsr_mip_bias.unwrap_or_else(|| {
            let r = wanted_render(d);
            auto_mip_bias(r[0] as f32 / d[0] as f32)
        }),
        _ => 0.0,
    };
    if ORIG_SAMPLER.load(Ordering::Relaxed) != 0
        && let Some((bias, n)) = samplers::frame(target)
    {
        crate::notify::info(if bias == 0.0 {
            "FSR mip bias off (engine samplers restored)".to_owned()
        } else {
            format!("FSR mip bias {bias:+.2} on {n} samplers")
        });
    }
    let mode = display.map(|d| (up, wanted_render(d)));
    let mut last = LAST_MODE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mode != *last {
        if let (Some(_), Some((up, r))) = (*last, mode) {
            let msg = if up {
                format!("FSR upscale {} ({}x{})", s.fsr_scale, r[0], r[1])
            } else if s.fsr_upscale && in_menu() {
                format!("FSR native AA in the main menu ({}x{})", r[0], r[1])
            } else {
                format!("FSR native AA ({}x{})", r[0], r[1])
            };
            log::line(&format!("fsr: {msg}"));
            crate::notify::info(msg);
        }
        *last = mode;
    }
    if s.fsr && s.fsr_upscale && !POST_HOOKS.load(Ordering::Relaxed) {
        log_once(
            4,
            "fsr: upscale requested but the post-chain hooks are missing; native AA",
        );
    }
}

/// The overlay's mip-bias line: (bias in the cells, biased samplers, tracked samplers).
pub fn mip_bias_status() -> (f32, usize, usize) {
    samplers::status()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_sizes_are_multiples_of_8_and_scale_exactly() {
        for (d, s) in [
            (3440u32, 0.67f32),
            (1440, 0.67),
            (2560, 0.5),
            (1080, 0.77),
            (1440, 1.0),
        ] {
            let r = snap_render(d, s);
            assert_eq!(r % 8, 0, "{d} {s}");
            assert!(r <= d);
            let f = r as f32 / d as f32;
            assert_eq!(scaled_extent(d, f), r, "{d} {s}");
        }
        assert_eq!(snap_render(3440, 0.67), 2304);
        assert_eq!(snap_render(1440, 1.0), 1440);
    }

    #[test]
    fn halton_matches_ffx() {
        // FFX 3.x: index 0 -> halton(1,2)-0.5 = 0, halton(1,3)-0.5 = -1/6.
        let o = halton_offset(0, 8);
        assert!(o[0].abs() < 1e-6 && (o[1] + 1.0 / 6.0).abs() < 1e-6);
        let o = halton_offset(1, 8);
        assert!((o[0] + 0.25).abs() < 1e-6 && (o[1] - 1.0 / 6.0).abs() < 1e-6);
        assert_eq!(phase_count(3440, 3440), 8);
        assert_eq!(phase_count(1720, 3440), 32);
        assert_eq!(phase_count(2304, 3440), 17);
    }

    #[test]
    fn table_entry_round_trips_through_the_engine_reading() {
        let (display, render) = ([3440u32, 1440u32], [2304u32, 968u32]);
        let j = [0.3125f32, -0.2];
        let e = table_entry(j, display, render);
        // Engine: jit = entry / W (unscaled); FSR reads (-jit.x * rW, +jit.y * rH).
        let jit = [e[0] / display[0] as f32, e[1] / display[1] as f32];
        let back = [-jit[0] * render[0] as f32, jit[1] * render[1] as f32];
        assert!((back[0] - j[0]).abs() < 1e-5 && (back[1] - j[1]).abs() < 1e-5);
    }

    #[test]
    fn mip_bias_and_sampler_rules() {
        assert!((auto_mip_bias(0.5) + 2.0).abs() < 1e-6);
        assert!((auto_mip_bias(1.0) + 1.0).abs() < 1e-6);
        assert!(sampler_takes_bias(0x2A)); // anisotropic, wrap
        assert!(sampler_takes_bias(0)); // trilinear, wrap
        assert!(!sampler_takes_bias(0x10)); // point mip
        assert!(!sampler_takes_bias(0x40)); // clamp U
        assert!(!sampler_takes_bias(0x1000)); // comparison
        assert!(!sampler_takes_bias(0x1_0000)); // forceNoMipLodBias
    }
}
