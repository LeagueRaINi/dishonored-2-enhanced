//! Own ambient occlusion in place of NVIDIA HBAO+ (docs/lighting-options.md B1, STATUS E063).
//!
//! Splice (O, IDB of the analysed build): the `ARK_RENDERPASS_SSAO` callback
//! `sub_140305BE0` runs the HBAO+ path `sub_140305620` while `dword_1432EF798` is set
//! (`r_hbao 1`). That path sets the view's scaled viewport (view+30816) on the pass's
//! deferred context (`sub_1402BC610` -> `RSSetViewports`, vtable +352), creates the HBAO+
//! context once (`GFSDK_SSAO_CreateContext_D3D11`, context `qword_1432EF760`, status
//! `dword_1422AE2D0`, 22 = not created yet) and calls `RenderAO` (vtable +8) with the
//! pass's deferred context, the hardware-depth SRV (read hash 555227252), the projection
//! (view+29584, column-major), the HBAO+ parameters and the RTV of write hash 2006969548;
//! its next draw (program `qword_1422AE2C0`) composites that target. Struct layouts:
//! HBAO+ 3.0 public header (`GFSDK_SSAO.h` r3.0.0.20735892), every field checked against
//! the stack stores of `sub_140305620` (offsets asserted below).
//!
//! Hook: `RenderAO` inside `GFSDK_SSAO_D3D11.win64.dll` (3.0.0.0, file SHA-256 checked) is
//! detoured with MinHook. Its address comes from the context's vtable: from
//! `qword_1432EF760` when the context already exists (installed while the game runs), else
//! from the context returned through a detour of the `GFSDK_SSAO_CreateContext_D3D11`
//! export (installed at startup, before the first SSAO pass). With `ao_mode = hbao` the
//! detour only calls the original. With `ao_mode = void` it records our AO into the same
//! deferred context, writes the RTV as HBAO+ would (the Output struct's blend mode), puts
//! back every D3D11 state it touched and returns `GFSDK_SSAO_OK`; HBAO+ does not run. Inputs
//! outside what the game passes (MSAA or view depths, input normals, two-pass blending,
//! debug render masks, invalid projections) fall back to HBAO+ with a log line.
//!
//! Algorithm (all compute except the final write): hardware depth -> linear view depth
//! (5-level MIP chain, XeGTAO's depth filter) -> screen-space AO with a visibility bitmask
//! (Therrien, Levesque, Gilet 2023, "Screen Space Indirect Lighting with Visibility
//! Bitmask"; slice/step structure and sample MIP selection of Intel's XeGTAO, Jimenez et
//! al. 2016 GTAO): per slice, 32 cosine-weighted sectors are marked between each sample's
//! front and back angle (assumed thickness), so thin occluders do not halo the background;
//! depth-reconstructed normals; HBAO+'s radius (meters x MetersToViewSpaceUnits),
//! quadratic falloff (dithered), foreground/background radius clamps, bias (sectors next
//! to the tangent plane ignored), small/large-scale weights (near/far sample split) and
//! power exponent -> a spatially
//! fixed 4x4 interleaved noise pattern removed by a separable depth-aware blur whose
//! weights cancel any 4-periodic pattern (no temporal filter) -> full-screen draw into the
//! engine's RTV inside the viewport only. `ao_half_res` computes at half resolution and
//! upsamples depth-aware. Only the engine's viewport (the render-scaled sub-rectangle in
//! FSR upscale mode) is read and written.

pub mod external;
pub(crate) mod offline;

pub use external::{Channel, ExternalMode, ExternalSource};
pub use offline::{Image, OfflineReport, offline_check};

/// Registers (Some) or removes (None) an external `.x` term (S1 entry point, `external.rs`):
/// drawn after HBAO+ / our AO in the `RenderAO` detour. Needs the hook ([`install`]).
pub fn set_external_x(src: Option<ExternalSource>) {
    external::set(Channel::X, src);
}

/// `RenderAO` calls so far (one per rendered frame while HBAO+ runs; photo mode's counters).
pub fn passes() -> u64 {
    CALLS.load(Ordering::Acquire)
}

use std::ffi::c_void;
use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use windows::Win32::Foundation::{HMODULE, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D11_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW, GetProcAddress};
use windows::core::{Interface, s, w};

use crate::settings::{self, Settings};
use crate::{hlsl, log, mem, notify};

// ------------------------------------------------------------------ HBAO+ 3.0 ABI --

/// HBAO+ DLL in the game folder and its identity (version resource 3.0.0.0).
pub const HBAO_DLL: &str = "GFSDK_SSAO_D3D11.win64.dll";
const HBAO_SHA256: &str = "aed9793677fcbe4fb536ba2b0f2df0bf49bba831b9e63666f46bb199aa91b670";
/// `qword_1432EF760`: the engine's HBAO+ context (null until the first SSAO pass).
const CONTEXT_RVA: u64 = 0x32E_F760;
/// `dword_1422AE2D0`: status of the context creation (22 before the first attempt).
const STATUS_RVA: u64 = 0x22A_E2D0;
const STATUS_NOT_CREATED: u32 = 22;
const GFSDK_SSAO_OK: u32 = 0;
/// `GFSDK_SSAO_RENDER_AO` (DRAW_Z | DRAW_AO), the mask the game passes.
const RENDER_AO: u32 = 3;

/// `GFSDK_SSAO_Version`; the game passes {3, 0, 0, 0} (`v39 = 3` in `sub_140305620`).
#[repr(C)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub branch: u32,
    pub revision: u32,
}

/// `GFSDK_SSAO_Matrix`: 16 floats + `GFSDK_SSAO_MatrixLayout` (0 row-, 1 column-major).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    pub data: [f32; 16],
    pub layout: u32,
}

/// `GFSDK_SSAO_InputViewport`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputViewport {
    pub enable: u32,
    pub top_left_x: u32,
    pub top_left_y: u32,
    pub width: u32,
    pub height: u32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// `GFSDK_SSAO_InputDepthData_D3D11` (3.0: one SRV, no second layer).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputDepthData {
    pub depth_texture_type: u32,
    pub projection: Matrix,
    pub meters_to_view_space_units: f32,
    pub viewport: InputViewport,
    pub depth_srv: *mut c_void,
}

/// `GFSDK_SSAO_InputNormalData_D3D11`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputNormalData {
    pub enable: u32,
    pub world_to_view: Matrix,
    pub decode_scale: f32,
    pub decode_bias: f32,
    pub normal_srv: *mut c_void,
}

/// `GFSDK_SSAO_InputData_D3D11` (game: stack `v53`, rsp+0xF0).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputData {
    pub depth: InputDepthData,
    pub normal: InputNormalData,
}

/// `GFSDK_SSAO_ForegroundAO` / `GFSDK_SSAO_BackgroundAO`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DepthClamp {
    pub enable: u32,
    pub view_depth: f32,
}

/// `GFSDK_SSAO_DepthThreshold`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DepthThreshold {
    pub enable: u32,
    pub max_view_depth: f32,
    pub sharpness: f32,
}

/// `GFSDK_SSAO_BlurSharpnessProfile`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlurSharpnessProfile {
    pub enable: u32,
    pub foreground_sharpness_scale: f32,
    pub foreground_view_depth: f32,
    pub background_view_depth: f32,
}

/// `GFSDK_SSAO_BlurParameters`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlurParameters {
    pub enable: u32,
    pub radius: u32,
    pub sharpness: f32,
    pub profile: BlurSharpnessProfile,
}

/// `GFSDK_SSAO_Parameters` (3.0; game: stack `v41`, rsp+0x90).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub radius: f32,
    pub bias: f32,
    pub small_scale_ao: f32,
    pub large_scale_ao: f32,
    pub power_exponent: f32,
    pub foreground_ao: DepthClamp,
    pub background_ao: DepthClamp,
    pub depth_storage: u32,
    pub depth_clamp_mode: u32,
    pub depth_threshold: DepthThreshold,
    pub blur: BlurParameters,
}

impl Parameters {
    /// The values `sub_140305620` stores for render scale `s` (`min` of
    /// `g_resolutionScale.xy`): radius 0.7, bias 0.1, small scale 1/s, large scale 1.75 s,
    /// power 1.75, foreground AO on at 4.0, background AO off, FP16 depths, clamp to edge,
    /// no depth threshold (sharpness 100), blur **off** (radius 4, sharpness 8, profile off).
    pub fn game(s: f32) -> Self {
        Self {
            radius: 0.7,
            bias: 0.1,
            small_scale_ao: 1.0 / s,
            large_scale_ao: 1.75 * s,
            power_exponent: 1.75,
            foreground_ao: DepthClamp {
                enable: 1,
                view_depth: 4.0,
            },
            background_ao: DepthClamp {
                enable: 0,
                view_depth: 0.0,
            },
            depth_storage: 0,
            depth_clamp_mode: 0,
            depth_threshold: DepthThreshold {
                enable: 0,
                max_view_depth: 0.0,
                sharpness: 100.0,
            },
            blur: BlurParameters {
                enable: 0,
                radius: 1,
                sharpness: 8.0,
                profile: BlurSharpnessProfile {
                    enable: 0,
                    foreground_sharpness_scale: 4.0,
                    foreground_view_depth: 0.0,
                    background_view_depth: 1.0,
                },
            },
        }
    }
}

/// `GFSDK_SSAO_CustomBlendState_D3D11`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CustomBlend {
    pub blend_state: *mut c_void,
    pub blend_factor: *const f32,
}

/// `GFSDK_SSAO_BlendState_D3D11`: mode 0 OVERWRITE_RGB, 1 MULTIPLY_RGB, 2 CUSTOM_BLEND.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlendState {
    pub mode: u32,
    pub custom: CustomBlend,
}

/// `GFSDK_SSAO_DepthStencilState_D3D11` (mode + custom state, stencil ref).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DepthStencilState {
    pub mode: u32,
    pub depth_stencil_state: *mut c_void,
    pub stencil_ref: u32,
}

/// `GFSDK_SSAO_BlendPass_D3D11`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlendPass {
    pub blend: BlendState,
    pub depth_stencil: DepthStencilState,
}

/// `GFSDK_SSAO_TwoPassBlend_D3D11`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TwoPassBlend {
    pub enable: u32,
    pub dsv: *mut c_void,
    pub first: BlendPass,
    pub second: BlendPass,
}

/// `GFSDK_SSAO_Output_D3D11` (3.0 with `TwoPassBlend`; game: stack `v66`, rsp+0x1D0).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Output {
    pub rtv: *mut c_void,
    pub blend: BlendState,
    pub two_pass: TwoPassBlend,
}

// Offsets of the fields `sub_140305620` writes (frame offsets relative to each struct's
// base: InputData rsp+0xF0, Parameters rsp+0x90, Output rsp+0x1D0).
const _: () = {
    assert!(offset_of!(InputDepthData, projection) == 0x04); // v53[1..16]
    assert!(offset_of!(InputDepthData, projection) + offset_of!(Matrix, layout) == 0x44); // v54 = 1
    assert!(offset_of!(InputDepthData, meters_to_view_space_units) == 0x48); // v55 lo = 1.0
    assert!(offset_of!(InputDepthData, viewport) == 0x4C); // v55 hi = 0 (Enable)
    assert!(offset_of!(InputDepthData, depth_srv) == 0x68); // v60 = SRV
    assert!(offset_of!(InputData, normal) == 0x70); // v61 = 0 (Enable)
    assert!(offset_of!(InputData, normal) + offset_of!(InputNormalData, decode_scale) == 0xB8); // v63 lo = 1.0
    assert!(offset_of!(InputData, normal) + offset_of!(InputNormalData, normal_srv) == 0xC0); // v64 = 0
    assert!(size_of::<InputData>() == 0xC8);
    assert!(offset_of!(Parameters, power_exponent) == 0x10); // 1.75
    assert!(offset_of!(Parameters, foreground_ao) == 0x14); // 1, 4.0
    assert!(offset_of!(Parameters, depth_threshold) == 0x2C); // 0, 0, 100.0 at +0x34
    assert!(offset_of!(Parameters, blur) == 0x38); // r12d = 0 (Enable), 1, 8.0
    assert!(size_of::<Parameters>() == 0x54); // last store +0x50 = 1.0
    assert!(offset_of!(Output, blend) == 0x08); // v67 = 0 (OVERWRITE_RGB)
    assert!(offset_of!(Output, two_pass) == 0x20); // v69 = 0 (Enable)
    assert!(size_of::<BlendPass>() == 0x30);
    assert!(size_of::<Output>() == 0x90); // last store v82 at +0x88
};

type RenderAoFn = unsafe extern "system" fn(
    this: *mut c_void,
    ctx: *mut c_void,
    input: *const InputData,
    params: *const Parameters,
    output: *const Output,
    mask: u32,
) -> u32;
pub type CreateContextFn = unsafe extern "system" fn(
    device: *mut c_void,
    out: *mut *mut c_void,
    heap: *const c_void,
    version: *const Version,
) -> u32;

// ------------------------------------------------------------------------ status --

static RENDER_AO_ORIG: AtomicUsize = AtomicUsize::new(0);
static CREATE_ORIG: AtomicUsize = AtomicUsize::new(0);
static INSTALL_TRIED: AtomicBool = AtomicBool::new(false);
/// 0 nothing, 1 waiting for the context (CreateContext hooked), 2 RenderAO hooked.
static HOOK_STATE: AtomicU8 = AtomicU8::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static OURS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static FIRST_CALL_LOGGED: AtomicBool = AtomicBool::new(false);
static FALLBACK_NOTIFIED: AtomicBool = AtomicBool::new(false);
/// Last `ao_mode` seen by `tick` (0 = not yet, 1 = hbao, 2 = void).
static LAST_MODE: AtomicU8 = AtomicU8::new(0);
static ERROR: Mutex<Option<String>> = Mutex::new(None);
static INFO: Mutex<Option<String>> = Mutex::new(None);

/// State for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// `RenderAO` is detoured.
    pub hooked: bool,
    /// `GFSDK_SSAO_CreateContext_D3D11` is detoured; waiting for the engine's context.
    pub waiting: bool,
    pub calls: u64,
    /// Calls rendered by our AO.
    pub ours: u64,
    /// Calls handed to HBAO+ although `ao_mode = void` (see `error`).
    pub fallbacks: u64,
    /// First call: struct fields, formats and sizes.
    pub info: Option<String>,
    /// Installation error or the last fallback reason.
    pub error: Option<String>,
}

pub fn status() -> Status {
    let state = HOOK_STATE.load(Ordering::Relaxed);
    Status {
        hooked: state == 2,
        waiting: state == 1,
        calls: CALLS.load(Ordering::Relaxed),
        ours: OURS.load(Ordering::Relaxed),
        fallbacks: FALLBACKS.load(Ordering::Relaxed),
        info: lock(&INFO).clone(),
        error: lock(&ERROR).clone(),
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn set_error(e: &str) {
    *lock(&ERROR) = Some(e.to_owned());
}

// ------------------------------------------------------------------- settings --

/// `ao_quality` presets: name, slices (direction pairs), steps per direction.
pub const QUALITIES: [(&str, u32, u32); 4] = [
    ("low", 2, 2),
    ("medium", 2, 3),
    ("high", 3, 3),
    ("ultra", 4, 4),
];

/// Index of `ao_quality = high` (the default).
pub const DEFAULT_QUALITY: u8 = 2;

pub fn quality_name(q: u8) -> &'static str {
    QUALITIES[usize::from(q).min(QUALITIES.len() - 1)].0
}

pub fn parse_quality(v: &str) -> Option<u8> {
    QUALITIES
        .iter()
        .position(|(n, _, _)| *n == v)
        .map(|i| i as u8)
}

/// Per frame (main thread, `hook::observe`): a toast and a log line when `ao_mode`
/// changes; turning `void` on installs the hook if that was not tried yet.
pub fn tick() {
    let s = settings::get();
    let mode = if s.ao_void { 2 } else { 1 };
    let last = LAST_MODE.swap(mode, Ordering::Relaxed);
    if last == mode {
        return;
    }
    if s.ao_void && !INSTALL_TRIED.load(Ordering::Relaxed) {
        match install() {
            Ok(()) => log::line("ao: installed at run time (ao_mode = void)"),
            Err(e) => {
                log::line(&format!("ao: installation failed: {e}; HBAO+ stays"));
                notify::error(format!("Own AO unavailable: {e}"));
            }
        }
    }
    if last != 0 {
        let text = if s.ao_void {
            "Ambient occlusion: own (visibility bitmask; HBAO+ skipped)"
        } else {
            "Ambient occlusion: NVIDIA HBAO+"
        };
        log::line(&format!(
            "ao: ao_mode = {}",
            if s.ao_void { "void" } else { "hbao" }
        ));
        notify::info(text);
    }
}

// ----------------------------------------------------------------- installation --

fn hbao_module() -> Result<HMODULE, String> {
    // SAFETY: plain lookup of a loaded module by name.
    unsafe { GetModuleHandleW(w!("GFSDK_SSAO_D3D11.win64.dll")) }
        .map_err(|e| format!("{HBAO_DLL} is not loaded ({e})"))
}

/// Address range of a loaded PE image (`SizeOfImage` from its optional header).
fn module_range(m: HMODULE) -> Option<(u64, u64)> {
    let base = m.0 as u64;
    let nt = base + u64::from(mem::u32_at(base + 0x3C)?);
    let size = mem::u32_at(nt + 0x50)?;
    Some((base, base + u64::from(size)))
}

/// Checks the file behind `m` against the analysed HBAO+ build.
fn verify_module(m: HMODULE) -> Result<(), String> {
    let mut buf = [0u16; 1024];
    // SAFETY: valid module handle and output buffer.
    let n = unsafe { GetModuleFileNameW(Some(m), &mut buf) } as usize;
    if n == 0 || n >= buf.len() {
        return Err(format!("{HBAO_DLL}: module path unavailable"));
    }
    let path = std::path::PathBuf::from(String::from_utf16_lossy(&buf[..n]));
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hash = crate::sha256_hex(&bytes);
    if hash != HBAO_SHA256 {
        return Err(format!(
            "{} has SHA-256 {hash}, not the analysed HBAO+ 3.0 build ({}...); struct layouts unverified",
            path.display(),
            &HBAO_SHA256[..16]
        ));
    }
    Ok(())
}

/// Installs the HBAO+ hook (at startup with `ao_mode = void`, or when it is turned on).
/// Shader compilation starts on a background thread; HBAO+ runs until it is done.
pub fn install() -> Result<(), String> {
    if INSTALL_TRIED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let r = install_inner();
    if let Err(e) = &r {
        set_error(e);
    }
    r
}

fn install_inner() -> Result<(), String> {
    std::thread::spawn(|| {
        if let Err(e) = bytecode() {
            log::line(&format!("ao: shader compilation failed: {e}"));
        }
    });
    let module = hbao_module()?;
    verify_module(module)?;
    let base = mem::image_base().ok_or("no image base")?;
    let ctx = mem::u64_at(base + CONTEXT_RVA).unwrap_or(0);
    if ctx != 0 {
        return hook_render_ao(ctx);
    }
    let status = mem::u32_at(base + STATUS_RVA).unwrap_or(u32::MAX);
    if status != STATUS_NOT_CREATED {
        return Err(format!(
            "the engine's HBAO+ context creation returned status {status} (no context)"
        ));
    }
    // SAFETY: export lookup in the verified module.
    let create = unsafe { GetProcAddress(module, s!("GFSDK_SSAO_CreateContext_D3D11")) }
        .ok_or("GFSDK_SSAO_CreateContext_D3D11 not exported")?;
    let target = create as *mut c_void;
    // SAFETY: start of the exported function of the verified DLL; the detour has its
    // signature (four pointer-sized arguments, status result) and forwards them unchanged.
    let tramp =
        unsafe { minhook::MinHook::create_hook(target, create_context_detour as *mut c_void) }
            .map_err(|e| format!("create_hook CreateContext: {e:?}"))?;
    CREATE_ORIG.store(tramp as usize, Ordering::Relaxed);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target) }
        .map_err(|e| format!("enable_hook CreateContext: {e:?}"))?;
    HOOK_STATE.store(1, Ordering::Relaxed);
    log::line(&format!(
        "ao: GFSDK_SSAO_CreateContext_D3D11 hooked at {target:p}; RenderAO is hooked when the engine creates its context"
    ));
    Ok(())
}

/// Detours the `RenderAO` implementation reached through `context`'s vtable (slot 1).
pub fn hook_render_ao(context: u64) -> Result<(), String> {
    if RENDER_AO_ORIG.load(Ordering::Acquire) != 0 {
        return Ok(());
    }
    let module = hbao_module()?;
    let (lo, hi) = module_range(module).ok_or("HBAO+ module header unreadable")?;
    let vtable = mem::u64_at(context).ok_or("HBAO+ context unreadable")?;
    let target = mem::u64_at(vtable + 8).ok_or("HBAO+ vtable unreadable")?;
    if !(lo..hi).contains(&vtable) || !(lo..hi).contains(&target) {
        return Err(format!(
            "HBAO+ vtable {vtable:#x} / RenderAO {target:#x} outside the module {lo:#x}..{hi:#x}"
        ));
    }
    // SAFETY: `target` is `GFSDK_SSAO_Context_D3D11::RenderAO` of the verified DLL (vtable
    // slot 1 of a live context, inside the module); the detour has its signature.
    let tramp = unsafe {
        minhook::MinHook::create_hook(target as *mut c_void, render_ao_detour as *mut c_void)
    }
    .map_err(|e| format!("create_hook RenderAO: {e:?}"))?;
    RENDER_AO_ORIG.store(tramp as usize, Ordering::Release);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook RenderAO: {e:?}"))?;
    HOOK_STATE.store(2, Ordering::Relaxed);
    log::line(&format!(
        "ao: HBAO+ hook installed: RenderAO {target:#x} (context {context:#x}, vtable {vtable:#x}); ao_mode decides per call"
    ));
    Ok(())
}

unsafe extern "system" fn create_context_detour(
    device: *mut c_void,
    out: *mut *mut c_void,
    heap: *const c_void,
    version: *const Version,
) -> u32 {
    // SAFETY: trampoline of the hooked export, arguments unchanged.
    let f = unsafe {
        std::mem::transmute::<usize, CreateContextFn>(CREATE_ORIG.load(Ordering::Relaxed))
    };
    // SAFETY: as above.
    let status = unsafe { f(device, out, heap, version) };
    if status == GFSDK_SSAO_OK && !out.is_null() {
        // SAFETY: on success the DLL wrote the context pointer to `out`.
        let ctx = unsafe { *out } as u64;
        let _ = std::panic::catch_unwind(|| {
            if let Err(e) = hook_render_ao(ctx) {
                log::line(&format!("ao: {e}; HBAO+ stays"));
                set_error(&e);
            }
        });
    } else {
        log::line(&format!(
            "ao: GFSDK_SSAO_CreateContext_D3D11 returned status {status}"
        ));
    }
    status
}

unsafe extern "system" fn render_ao_detour(
    this: *mut c_void,
    ctx: *mut c_void,
    input: *const InputData,
    params: *const Parameters,
    output: *const Output,
    mask: u32,
) -> u32 {
    CALLS.fetch_add(1, Ordering::Relaxed);
    // The viewport the engine bound for this call (HBAO+ may change it while recording).
    // SAFETY: the caller's arguments, valid for the duration of the call.
    let bound = unsafe { bound_viewport(ctx) };
    let ours = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: as above.
        unsafe { replace(ctx, input, params, output, mask, bound.as_ref()) }
    }))
    .unwrap_or_else(|_| Err("panic in the AO path".to_owned()));
    let status = match ours {
        Ok(true) => {
            OURS.fetch_add(1, Ordering::Relaxed);
            GFSDK_SSAO_OK
        }
        r => {
            if let Err(e) = r {
                let n = FALLBACKS.fetch_add(1, Ordering::Relaxed);
                if n < 5 || n.is_power_of_two() {
                    log::line(&format!("ao: {e}; HBAO+ used (fallback {})", n + 1));
                }
                if !FALLBACK_NOTIFIED.swap(true, Ordering::Relaxed) {
                    notify::warn(format!("Own AO not applied ({e}); HBAO+ used"));
                }
                set_error(&e);
            }
            // SAFETY: MinHook trampoline of `RenderAO`, arguments unchanged.
            let f = unsafe {
                std::mem::transmute::<usize, RenderAoFn>(RENDER_AO_ORIG.load(Ordering::Acquire))
            };
            // SAFETY: as above.
            unsafe { f(this, ctx, input, params, output, mask) }
        }
    };
    if status == GFSDK_SSAO_OK && external::registered(Channel::X) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: as above.
            unsafe { apply_external_x(ctx, input, output, mask, bound.as_ref()) }
        }));
    }
    status
}

/// The first viewport bound on `ctx`.
///
/// # Safety
/// `ctx` is null or a live `ID3D11DeviceContext` recorded by this thread.
unsafe fn bound_viewport(ctx: *mut c_void) -> Option<D3D11_VIEWPORT> {
    // SAFETY: the caller's live context (or null).
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&ctx) }?;
    let mut count = 1u32;
    let mut bound = [D3D11_VIEWPORT::default()];
    // SAFETY: getter with a one-element output array.
    unsafe { ctx.RSGetViewports(&mut count, Some(bound.as_mut_ptr())) };
    (count > 0).then_some(bound[0])
}

/// The registered `.x` term after the AO was written (HBAO+ or ours): into the output RTV
/// inside the render region. Only for the game's call shape (AO render mask, single-pass
/// output); failures are counted in `external`.
///
/// # Safety
/// The arguments are those of a `RenderAO` call (valid for its duration).
unsafe fn apply_external_x(
    ctx: *mut c_void,
    input: *const InputData,
    output: *const Output,
    mask: u32,
    bound: Option<&D3D11_VIEWPORT>,
) -> Result<bool, String> {
    if input.is_null() || output.is_null() || mask != RENDER_AO {
        return Ok(false);
    }
    // SAFETY: non-null pointers to the caller's structs.
    let (input, output) = unsafe { (&*input, &*output) };
    if output.two_pass.enable != 0 {
        return Ok(false);
    }
    // SAFETY: the caller's live context and output view.
    let (ctx, rtv) = unsafe {
        (
            ID3D11DeviceContext::from_raw_borrowed(&ctx).ok_or("null device context")?,
            ID3D11RenderTargetView::from_raw_borrowed(&output.rtv).ok_or("null output RTV")?,
        )
    };
    let rt = texture_desc(&rtv.cast().map_err(create_err("output view"))?)?;
    let region =
        region(&input.depth.viewport, bound, [rt.Width, rt.Height]).ok_or("empty viewport")?;
    external::apply(Channel::X, ctx, rtv, region)
}

// ------------------------------------------------------------------ pure helpers --

/// Projection terms in the screen-consistent view frame (x right, y up, z = positive view
/// depth): `view.xy = z * (ndc - jitter) / (p00, p11)`, `z = depth_mul / (d - depth_add)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Proj {
    pub p00: f32,
    pub p11: f32,
    pub jitter: [f32; 2],
    pub depth_mul: f32,
    pub depth_add: f32,
}

/// Parses an HBAO+ projection matrix (layout 0 row-major, 1 column-major). HBAO+ requires
/// rows `{P00 0 0 0} {0 P11 0 0} {P20 P21 P22 P23} {0 0 P32 0}` with |P23| = 1; then the
/// view depth is `P32 / (d - P22 P23)` and `x = z (ndc.x - P20 P23) / P00`.
pub fn projection(data: &[f32; 16], layout: u32) -> Result<Proj, String> {
    if layout > 1 {
        return Err(format!("matrix layout {layout}"));
    }
    let at = |r: usize, c: usize| {
        if layout == 1 {
            data[4 * c + r]
        } else {
            data[4 * r + c]
        }
    };
    let scale = at(0, 0).abs().max(at(1, 1).abs()).max(1.0);
    let zeros = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 2),
        (1, 3),
        (3, 0),
        (3, 1),
        (3, 3),
    ];
    let p23 = at(2, 3);
    if !data.iter().all(|v| v.is_finite())
        || (p23.abs() - 1.0).abs() > 1e-3
        || zeros.iter().any(|&(r, c)| at(r, c).abs() > 1e-3 * scale)
        || at(0, 0).abs() < 1e-6
        || at(1, 1).abs() < 1e-6
        || at(3, 2).abs() < 1e-12
    {
        return Err(format!(
            "projection not of the HBAO+ form (layout {layout}): {data:?}"
        ));
    }
    let sign = p23.signum();
    Ok(Proj {
        p00: at(0, 0).abs(),
        p11: at(1, 1).abs(),
        jitter: [at(2, 0) * sign, at(2, 1) * sign],
        depth_mul: at(3, 2),
        depth_add: at(2, 2) * sign,
    })
}

/// The engine's projection as `sub_1401098E0` builds it (renderer-replacement.md "Far
/// plane": right-handed, reversed-Z `d = n f / ((f - n) z) - n / (f - n)`), stored
/// column-major as `sub_140305620` passes it (layout 1); `jitter` = (P20, P21).
pub fn engine_projection(
    aspect: f32,
    fov_y: f32,
    near: f32,
    far: f32,
    jitter: [f32; 2],
) -> [f32; 16] {
    let p11 = 1.0 / (fov_y * 0.5).tan();
    let p00 = p11 / aspect;
    let p22 = near / (far - near);
    let p32 = near * far / (far - near);
    [
        p00, 0.0, jitter[0], 0.0, //
        0.0, p11, jitter[1], 0.0, //
        0.0, 0.0, p22, p32, //
        0.0, 0.0, -1.0, 0.0,
    ]
}

/// Render region (x, y, w, h) inside a `tex`-sized texture: the HBAO+ input viewport when
/// enabled, else the viewport bound on the context (the engine binds the view's scaled
/// viewport right before `RenderAO`), else the whole texture.
fn region(
    input: &InputViewport,
    bound: Option<&D3D11_VIEWPORT>,
    tex: [u32; 2],
) -> Option<[u32; 4]> {
    let (x, y, w, h) = if input.enable != 0 {
        (
            input.top_left_x,
            input.top_left_y,
            input.width,
            input.height,
        )
    } else if let Some(v) = bound {
        let r = |f: f32| f.round().max(0.0) as u32;
        (r(v.TopLeftX), r(v.TopLeftY), r(v.Width), r(v.Height))
    } else {
        (0, 0, tex[0], tex[1])
    };
    if x >= tex[0] || y >= tex[1] {
        return None;
    }
    let (w, h) = (w.min(tex[0] - x), h.min(tex[1] - y));
    (w > 0 && h > 0).then_some([x, y, w, h])
}

fn ao_size(region: [u32; 2], half: bool) -> [u32; 2] {
    if half {
        [region[0].div_ceil(2), region[1].div_ceil(2)]
    } else {
        region
    }
}

/// Linear-depth texture size: the depth texture rounded up to multiples of 16 so the
/// MIP levels 1-4 are exact halvings (samples address all levels with one UV).
fn lin_dims(tex: [u32; 2]) -> [u32; 2] {
    [tex[0].next_multiple_of(16), tex[1].next_multiple_of(16)]
}

const MIPS: u32 = 5;
/// Far clamp of the linear depth (sky, reversed-Z 0): FP16 maximum as HBAO+'s view depths.
const MAX_VIEW_DEPTH: f32 = 65504.0;
/// Relative depth deviation (from the local plane) at which blur/upsample weights reach 0:
/// 1 / this.
const BLUR_SHARPNESS: f32 = 16.0;
/// Samples within this fraction of the screen radius form the small-scale (near) part.
const NEAR_FRACTION: f32 = 0.25;
/// Assumed thickness of every depth sample as a fraction of the radius (visibility
/// bitmask: sectors behind a thin foreground object stay visible, no GTAO-style halo).
const THICKNESS: f32 = 0.5;
/// Occlusion multiplier relating the bitmask's cosine-weighted occlusion to HBAO+'s
/// weighted sum at equal parameters. Calibrated offline against the real HBAO+ with the
/// game's parameters on `ao_offline`'s scene at 3440x1440: equal mean AO of the viewport
/// (0.967 vs HBAO+ 0.966); our corners are less black, contact shadows stronger.
const OCCLUSION_GAIN: f32 = 1.3;

/// Constant buffer `void_ao_params` (see the HLSL for each lane).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ShaderParams {
    view: [f32; 4],
    region: [f32; 4],
    tex: [f32; 4],
    ints: [u32; 4],
    depth: [f32; 4],
    screen: [f32; 4],
    amount: [f32; 4],
    lp: [f32; 4],
    mipf: [f32; 4],
}

fn shader_params(
    proj: &Proj,
    region: [u32; 4],
    tex: [u32; 2],
    hb: &Parameters,
    meters: f32,
    s: &Settings,
) -> ShaderParams {
    let [x, y, w, h] = region;
    let half = s.ao_half_res;
    let ao = ao_size([w, h], half);
    let dims = lin_dims(tex);
    let radius = hb.radius.max(1e-4) * meters.max(1e-6) * s.ao_radius_scale.max(0.01);
    // HBAO+: RadiusToScreen = R * 0.5 / tan(fovy / 2) * viewport height (full-res pixels).
    let to_screen = radius * 0.5 * proj.p11 * h as f32;
    let clamp_px = |c: &DepthClamp| {
        if c.enable != 0 {
            to_screen / c.view_depth.max(1e-4)
        } else {
            0.0
        }
    };
    let bias = hb.bias.clamp(0.0, 0.5);
    let gain = OCCLUSION_GAIN * s.ao_strength.max(0.0) / (1.0 - bias);
    let (_, slices, steps) = QUALITIES[usize::from(s.ao_quality).min(QUALITIES.len() - 1)];
    let (threshold_mul, threshold_sharp) = if hb.depth_threshold.enable != 0 {
        (
            -1.0 / hb.depth_threshold.max_view_depth.max(1e-4),
            hb.depth_threshold.sharpness.max(0.0),
        )
    } else {
        (0.0, 1.0)
    };
    // XeGTAO depth-MIP filter: falloff over the last 61.5 % of 0.75 x the radius.
    let effect = 0.75 * radius;
    let range = 0.615 * effect;
    ShaderParams {
        view: [
            2.0 / proj.p00,
            -2.0 / proj.p11,
            (-1.0 - proj.jitter[0]) / proj.p00,
            (1.0 - proj.jitter[1]) / proj.p11,
        ],
        region: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
        tex: [
            1.0 / dims[0] as f32,
            1.0 / dims[1] as f32,
            if half { 2.0 } else { 1.0 },
            if half { 1.0 } else { 0.0 },
        ],
        ints: [ao[0], ao[1], x, y],
        depth: [proj.depth_mul, proj.depth_add, MAX_VIEW_DEPTH, radius],
        screen: [
            to_screen,
            clamp_px(&hb.foreground_ao),
            clamp_px(&hb.background_ao),
            bias.asin(),
        ],
        amount: [
            gain * hb.small_scale_ao.clamp(0.0, 4.0),
            gain * hb.large_scale_ao.clamp(0.0, 4.0),
            hb.power_exponent.clamp(1.0, 8.0),
            BLUR_SHARPNESS,
        ],
        lp: [slices as f32, steps as f32, threshold_mul, threshold_sharp],
        mipf: [
            -1.0 / range,
            (effect - range) / range + 1.0,
            NEAR_FRACTION,
            THICKNESS,
        ],
    }
}

// ------------------------------------------------------------------------- HLSL --

const AO_HLSL: &str = r"
// void-inject own AO (ao.rs). Lanes of void_ao_params:
#ifndef AO_SLICES
#define AO_SLICES 2
#define AO_STEPS 2
#endif
cbuffer void_ao_params : register(b0)
{
    float4 g_view;   // view.xy = z * (uv * g_view.xy + g_view.zw), uv over the region
    float4 g_region; // region size w, h (full-res pixels), 1/w, 1/h
    float4 g_tex;    // 1 / linear-depth texture size (x, y), full-res px per AO px, AO MIP
    uint4  g_ints;   // AO-resolution size (x, y), region origin in the engine textures
    float4 g_depth;  // view depth = x / (d - y); far clamp; radius (view units)
    float4 g_screen; // radius px at depth 1, foreground max px, background min px, bias angle
    float4 g_amount; // small-scale weight, large-scale weight, power exponent, blur sharpness
    float4 g_loop;   // slices, steps (compiled in, informational), threshold -1/max depth, sharpness
    float4 g_mipf;   // depth-MIP filter mul, add; near fraction of the radius; thickness / radius
};
cbuffer void_ao_debug : register(b2) { float4 g_debug; }; // uv per pixel (x, y), power

Texture2D<float>   void_ao_hw_depth : register(t0);
Texture2D<float>   void_ao_lin      : register(t1);
Texture2D<float>   void_ao_input    : register(t2);
RWTexture2D<float> void_ao_out      : register(u0);  // also linear depth MIP 0
RWTexture2D<float> void_ao_mip1     : register(u1);
RWTexture2D<float> void_ao_mip2     : register(u2);
RWTexture2D<float> void_ao_mip3     : register(u3);
RWTexture2D<float> void_ao_mip4     : register(u4);
SamplerState       void_ao_point    : register(s0);
SamplerState       void_ao_linear   : register(s1);

static const float PI = 3.14159265;
static const float HALF_PI = 1.57079633;
// Spatially fixed 4x4 patterns (nothing changes per frame): slice rotation (Bayer) and
// step offset; the blur removes them.
static const uint NOISE_SLICE[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
static const uint NOISE_STEP[16] = { 5, 13, 9, 1, 15, 3, 11, 7, 2, 10, 6, 14, 8, 0, 12, 4 };

// fp: full-resolution pixel position inside the region (pixel centres at +0.5).
float2 ClampFp(float2 fp) { return clamp(fp, 0.5, g_region.xy - 0.5); }

float LinDepth(float2 fp, uint mip)
{
    return void_ao_lin.Load(int3(uint2(fp) >> mip, mip));
}

float3 ViewPos(float2 fp, float z)
{
    return float3(z * (fp * g_region.zw * g_view.xy + g_view.zw), z);
}

// Linear depth of AO pixel p (MIP 0 at full, MIP 1 at half resolution).
float AoDepth(int2 p) { return void_ao_lin.Load(int3(p, (uint)g_tex.w)); }

float LinearDepth(uint2 p)
{
    float d = void_ao_hw_depth.Load(int3(p + g_ints.zw, 0));
    float z = g_depth.x / (d - g_depth.y);
    return (z > 0.0 && z < g_depth.z) ? z : g_depth.z;
}

// XeGTAO's depth-MIP filter (weighted towards the farthest of four).
float MipFilter(float4 z)
{
    float zmax = max(max(z.x, z.y), max(z.z, z.w));
    float4 w = saturate((zmax - z) * g_mipf.x + g_mipf.y);
    return dot(w, z) / dot(w, 1.0);
}

bool InMip(uint2 p, uint k)
{
    return all(p < (((uint2)g_region.xy + (1u << k) - 1u) >> k));
}

groupshared float g_scratch[8][8];

// Linear depth MIP 0-4 of the region in one dispatch (XeGTAO's prefilter): each 8x8
// group covers 16x16 pixels; reads outside the region repeat its edge.
[numthreads(8, 8, 1)]
void cs_prefilter(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID)
{
    uint2 last = (uint2)g_region.xy - 1u;
    uint2 base = gid.xy * 16u + gt.xy * 2u;
    float4 z;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        uint2 p = base + uint2(i & 1u, i >> 1u);
        z[i] = LinearDepth(min(p, last));
        if (InMip(p, 0u)) void_ao_out[p] = z[i];
    }
    float m = MipFilter(z);
    uint2 p1 = gid.xy * 8u + gt.xy;
    if (InMip(p1, 1u)) void_ao_mip1[p1] = m;
    g_scratch[gt.x][gt.y] = m;
    GroupMemoryBarrierWithGroupSync();
    if (all((gt.xy & 1u) == 0u))
    {
        m = MipFilter(float4(g_scratch[gt.x][gt.y], g_scratch[gt.x + 1][gt.y],
                             g_scratch[gt.x][gt.y + 1], g_scratch[gt.x + 1][gt.y + 1]));
        uint2 p2 = gid.xy * 4u + gt.xy / 2u;
        if (InMip(p2, 2u)) void_ao_mip2[p2] = m;
        g_scratch[gt.x][gt.y] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all((gt.xy & 3u) == 0u))
    {
        m = MipFilter(float4(g_scratch[gt.x][gt.y], g_scratch[gt.x + 2][gt.y],
                             g_scratch[gt.x][gt.y + 2], g_scratch[gt.x + 2][gt.y + 2]));
        uint2 p3 = gid.xy * 2u + gt.xy / 4u;
        if (InMip(p3, 3u)) void_ao_mip3[p3] = m;
        g_scratch[gt.x][gt.y] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(gt.xy == 0u))
    {
        m = MipFilter(float4(g_scratch[0][0], g_scratch[4][0], g_scratch[0][4], g_scratch[4][4]));
        if (InMip(gid.xy, 4u)) void_ao_mip4[gid.xy] = m;
    }
}

// Sector t in [0, 1] of a direction at view-angle cosine c on side sign (+1 / -1) of a slice
// whose projected normal makes angle n (cosN, sinN) with the view vector: t = 0.5 + 0.5 *
// sin(angle from the normal), clamped to the hemisphere. Uniform sectors in t are cosine
// weighted. Pure ALU (no acos/sin per sample).
float SectorT(float c, float side, float cosN, float sinN)
{
    float sh = side * sqrt(saturate(1.0 - c * c));
    float sinRel = sh * cosN - c * sinN;
    float cosRel = c * cosN + sh * sinN;
    // Past the tangent plane (also when the angle wraps beyond pi): that side's edge.
    sinRel = cosRel < 0.0 ? side : sinRel;
    return saturate(0.5 + 0.5 * sinRel);
}

// Bits of the sectors between two t values (Therrien et al. 2023, visibility bitmask),
// minus the bias band next to the tangent plane.
uint SectorBits(float ta, float tb, float biasT)
{
    float lo = max(min(ta, tb), biasT), hi = min(max(ta, tb), 1.0 - biasT);
    if (hi <= lo) return 0u;
    uint start = min((uint)(lo * 32.0), 31u);
    uint count = (uint)ceil((hi - lo) * 32.0);
    return count >= 32u ? 0xFFFFFFFFu : ((1u << count) - 1u) << start;
}

[numthreads(8, 8, 1)]
void cs_ao(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_ints.xy)) return;
    float s = g_tex.z;
    uint mip0 = (uint)g_tex.w;
    float2 fp = ClampFp((id.xy + 0.5) * s);
    float z = LinDepth(fp, mip0);
    float radiusPx = g_screen.x / z;
    float r = radiusPx;
    if (g_screen.z > 0.0) r = max(r, g_screen.z);
    if (g_screen.y > 0.0) r = min(r, g_screen.y);
    if (r < 1.0 || z >= g_depth.z * 0.999)
    {
        void_ao_out[id.xy] = 1.0;
        return;
    }
    float R = g_depth.w * (r / radiusPx);
    float thickness = g_mipf.w * R;
    float3 P = ViewPos(fp, z);
    float3 V = -P * rsqrt(dot(P, P));

    // Normal from the neighbour pair with the smaller depth step on each axis.
    float2 fl = fp - float2(s, 0.0), fr = fp + float2(s, 0.0);
    float2 ft = fp - float2(0.0, s), fb = fp + float2(0.0, s);
    float zl = LinDepth(ClampFp(fl), mip0), zr = LinDepth(ClampFp(fr), mip0);
    float zt = LinDepth(ClampFp(ft), mip0), zb = LinDepth(ClampFp(fb), mip0);
    bool useL = fl.x > 0.0 && (fr.x >= g_region.x || abs(zl - z) <= abs(zr - z));
    bool useT = ft.y > 0.0 && (fb.y >= g_region.y || abs(zt - z) <= abs(zb - z));
    float3 dx = useL ? P - ViewPos(fl, zl) : ViewPos(ClampFp(fr), zr) - P;
    float3 dy = useT ? P - ViewPos(ft, zt) : ViewPos(ClampFp(fb), zb) - P;
    float3 c = cross(dx, dy);
    float cc = dot(c, c);
    float3 N = cc > 1e-30 ? c * rsqrt(cc) : V;
    if (dot(N, V) < 0.0) N = -N;

    uint ni = (id.x & 3u) + 4u * (id.y & 3u);
    float noiseSlice = (NOISE_SLICE[ni] + 0.5) / 16.0;
    float noiseStep = (NOISE_STEP[ni] + 0.5) / 16.0;
    // Per-quality permutations (AO_SLICES / AO_STEPS defines): fully unrolled loops let the
    // compiler batch the depth fetches.
    const float slices = AO_SLICES, steps = AO_STEPS;
    float rAo = r / s;
    float minT = 1.3 / rAo;
    float negInvR2 = -1.0 / (R * R);
    float biasT = 0.5 - 0.5 * cos(g_screen.w);
    float visAll = 0.0, visNear = 0.0, weight = 0.0;
    [unroll] for (float sl = 0.0; sl < slices; sl += 1.0)
    {
        float phi = (sl + noiseSlice) / slices * PI;
        float cphi = cos(phi), sphi = sin(phi);
        float2 omega = float2(cphi, -sphi) * rAo;
        float3 dir = float3(cphi, sphi, 0.0);
        float3 ortho = dir - dot(dir, V) * V;
        float3 axis = normalize(cross(ortho, V));
        float3 projN = N - axis * dot(N, axis);
        float lenN = length(projN);
        float cosN = saturate(dot(projN, V) / max(lenN, 1e-6));
        float sinN = (dot(ortho, projN) < 0.0 ? -1.0 : 1.0) * sqrt(1.0 - cosN * cosN);
        uint bitsAll = 0u, bitsNear = 0u;
        [unroll] for (float st = 0.0; st < steps; st += 1.0)
        {
            float jitter = frac(noiseStep + (sl + st * steps) * 0.6180339887);
            float t = (st + jitter) / steps;
            t = t * t + minT;
            float2 off = round(t * omega);
            uint mip = (uint)min(mip0 + clamp(log2(max(length(off), 1.0)) - 3.3, 0.0, 4.0), 4.0);
            [unroll] for (int side = 0; side < 2; side++)
            {
                float sd = side == 0 ? 1.0 : -1.0;
                float2 f = fp + sd * off * s;
                // Outside the region nothing is known: no occluder (a clamped sample would
                // lie off the slice and fake occlusion along the screen edges).
                if (any(f < 0.0) || any(f >= g_region.xy)) continue;
                float3 d = ViewPos(f, LinDepth(f, mip)) - P;
                float l = dot(d, d);
                // HBAO+'s quadratic falloff, dithered with the fixed pattern (the blur
                // turns it into a smooth fade).
                if (saturate(l * negInvR2 + 1.0) <= jitter) continue;
                float3 b = d - V * thickness;
                float cf = dot(d, V) * rsqrt(max(l, 1e-12));
                float cb = dot(b, V) * rsqrt(max(dot(b, b), 1e-12));
                uint bits = SectorBits(SectorT(cf, sd, cosN, sinN), SectorT(cb, sd, cosN, sinN), biasT);
                bitsAll |= bits;
                bitsNear |= t <= g_mipf.z ? bits : 0u;
            }
        }
        visAll += lenN * (1.0 - countbits(bitsAll) / 32.0);
        visNear += lenN * (1.0 - countbits(bitsNear) / 32.0);
        weight += lenN;
    }
    weight = max(weight, 1e-6);
    float occNear = saturate(1.0 - visNear / weight);
    float occFar = max(saturate(1.0 - visAll / weight) - occNear, 0.0);
    float occ = g_amount.x * occNear + g_amount.y * occFar;
    occ *= saturate((z * g_loop.z + 1.0) * g_loop.w);
    void_ao_out[id.xy] = saturate(1.0 - occ);
}

// Depth-aware blur, both axes in one pass through groupshared memory (12x12 tile with a
// 2-pixel apron per 8x8 group). Taps -2..2 with weights 0.5 1 1 1 0.5 give every residue
// mod 4 the same weight, so the 4x4 pattern cancels on smooth surfaces; the depth weight
// compares each tap with the centre's local plane.
groupshared float g_tileAo[12][12];
groupshared float g_tileZ[12][12];
groupshared float g_rowAo[12][8];

float BlurTaps(float a[5], float z[5])
{
    float zc = z[2];
    float g = abs(z[1] - zc) < abs(z[3] - zc) ? zc - z[1] : z[3] - zc;
    float acc = 0.0, wsum = 0.0;
    [unroll] for (int i = 0; i < 5; i++)
    {
        float k = (i == 0 || i == 4) ? 0.5 : 1.0;
        float w = k * saturate(1.0 - abs(z[i] - (zc + (i - 2) * g)) / zc * g_amount.w);
        acc += w * a[i];
        wsum += w;
    }
    return acc / max(wsum, 1e-6);
}

[numthreads(8, 8, 1)]
void cs_blur(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    int2 origin = int2(gid.xy) * 8 - 2;
    int2 last = int2(g_ints.xy) - 1;
    for (uint i = gi; i < 144u; i += 64u)
    {
        uint2 l = uint2(i % 12u, i / 12u);
        int2 q = clamp(origin + int2(l), 0, last);
        g_tileAo[l.y][l.x] = void_ao_input.Load(int3(q, 0));
        g_tileZ[l.y][l.x] = AoDepth(q);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = gi; j < 96u; j += 64u)
    {
        uint row = j / 8u, col = j % 8u + 2u;
        float a[5], z[5];
        [unroll] for (int k = 0; k < 5; k++)
        {
            a[k] = g_tileAo[row][col + k - 2];
            z[k] = g_tileZ[row][col + k - 2];
        }
        g_rowAo[row][col - 2u] = BlurTaps(a, z);
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 id = gid.xy * 8u + gt.xy;
    if (any(id >= g_ints.xy)) return;
    float a[5], z[5];
    [unroll] for (int k = 0; k < 5; k++)
    {
        a[k] = g_rowAo[gt.y + k][gt.x];
        z[k] = g_tileZ[gt.y + k][gt.x + 2u];
    }
    void_ao_out[id] = BlurTaps(a, z);
}

float4 vs_full(uint id : SV_VertexID) : SV_Position
{
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Final write inside the region (viewport): AO in RGB (alpha 1; the blend state's write
// mask keeps the target's alpha), power exponent applied here as HBAO+ does last.
float4 ps_composite(float4 pos : SV_Position) : SV_Target
{
    float2 fp = pos.xy - float2(g_ints.zw);
    float ao;
    if (g_tex.z < 1.5)
    {
        ao = void_ao_input.Load(int3(int2(fp), 0));
    }
    else
    {
        float z = LinDepth(ClampFp(fp), 0.0);
        float2 a = fp * 0.5 - 0.5;
        int2 b = int2(floor(a));
        float2 f = a - b;
        int2 last = int2(g_ints.xy) - 1;
        float acc = 0.0, wsum = 0.0;
        [unroll] for (int j = 0; j < 4; j++)
        {
            int2 o = int2(j & 1, j >> 1);
            int2 q = clamp(b + o, 0, last);
            float2 bw = lerp(1.0 - f, f, float2(o));
            float w = bw.x * bw.y * (saturate(1.0 - abs(AoDepth(q) - z) / z * g_amount.w) + 1e-3);
            acc += w * void_ao_input.Load(int3(q, 0));
            wsum += w;
        }
        ao = acc / max(wsum, 1e-6);
    }
    ao = pow(saturate(ao), g_amount.z);
    return float4(ao, ao, ao, 1.0);
}

float4 ps_debug(float4 pos : SV_Position) : SV_Target
{
    float ao = void_ao_input.SampleLevel(void_ao_linear, pos.xy * g_debug.xy, 0.0);
    ao = pow(saturate(ao), g_debug.z);
    return float4(ao, ao, ao, 1.0);
}
";

/// Compiled shaders (device independent).
pub struct Bytecode {
    prefilter: Vec<u8>,
    /// `cs_ao` per `QUALITIES` entry (slices/steps as compile-time constants).
    ao: Vec<Vec<u8>>,
    blur: Vec<u8>,
    vs: Vec<u8>,
    composite: Vec<u8>,
    debug: Vec<u8>,
    /// Compilation time (ms).
    pub compile_ms: f64,
}

/// Compiled on first use: by the background thread `install` starts (or the offline
/// check); render threads only look at it once `BYTECODE_READY` is set (no blocking).
static BYTECODE: LazyLock<Result<Bytecode, String>> = LazyLock::new(|| {
    let t = std::time::Instant::now();
    let c = |entry, target| hlsl::compile(AO_HLSL, entry, target);
    Ok(Bytecode {
        prefilter: c(s!("cs_prefilter"), s!("cs_5_0"))?,
        ao: QUALITIES
            .iter()
            .map(|(_, slices, steps)| {
                let src =
                    format!("#define AO_SLICES {slices}\n#define AO_STEPS {steps}\n{AO_HLSL}");
                hlsl::compile(&src, s!("cs_ao"), s!("cs_5_0"))
            })
            .collect::<Result<_, _>>()?,
        blur: c(s!("cs_blur"), s!("cs_5_0"))?,
        vs: c(s!("vs_full"), s!("vs_5_0"))?,
        composite: c(s!("ps_composite"), s!("ps_5_0"))?,
        debug: c(s!("ps_debug"), s!("ps_5_0"))?,
        compile_ms: t.elapsed().as_secs_f64() * 1000.0,
    })
});
static BYTECODE_READY: AtomicBool = AtomicBool::new(false);

/// The compiled shaders (blocks while compiling).
pub fn bytecode() -> Result<&'static Bytecode, String> {
    let r = BYTECODE.as_ref().map_err(Clone::clone);
    BYTECODE_READY.store(true, Ordering::Release);
    r
}

// --------------------------------------------------------------- state save/restore --

/// Slots saved per stage: shader resources, constant buffers, samplers, CS UAVs.
#[derive(Clone, Copy)]
pub(crate) struct Slots {
    pub(crate) srv: usize,
    pub(crate) cb: usize,
    pub(crate) sampler: usize,
    pub(crate) uav: usize,
}

/// Every slot the AO path and the debug view bind (t0-t2, b0-b2, s0-s1, u0-u4).
const TOUCHED: Slots = Slots {
    srv: 3,
    cb: 3,
    sampler: 2,
    uav: MIPS as usize,
};

/// D3D11 state of a context: IA layout/topology, all shader stages, the PS/CS slots in
/// `Slots`, CS UAVs, OM targets/UAVs/blend/depth-stencil, rasterizer state, viewports and
/// scissor rects. Restoring re-binds exactly these (references released afterwards).
pub(crate) struct Saved {
    layout: Option<ID3D11InputLayout>,
    topology: D3D_PRIMITIVE_TOPOLOGY,
    vs: Option<ID3D11VertexShader>,
    hs: Option<ID3D11HullShader>,
    ds: Option<ID3D11DomainShader>,
    gs: Option<ID3D11GeometryShader>,
    ps: Option<ID3D11PixelShader>,
    cs: Option<ID3D11ComputeShader>,
    ps_srv: Vec<Option<ID3D11ShaderResourceView>>,
    ps_cb: Vec<Option<ID3D11Buffer>>,
    ps_sampler: Vec<Option<ID3D11SamplerState>>,
    cs_srv: Vec<Option<ID3D11ShaderResourceView>>,
    cs_cb: Vec<Option<ID3D11Buffer>>,
    cs_sampler: Vec<Option<ID3D11SamplerState>>,
    cs_uav: Vec<Option<ID3D11UnorderedAccessView>>,
    rtv: [Option<ID3D11RenderTargetView>; 8],
    dsv: Option<ID3D11DepthStencilView>,
    om_uav: [Option<ID3D11UnorderedAccessView>; 8],
    blend: Option<ID3D11BlendState>,
    blend_factor: [f32; 4],
    sample_mask: u32,
    depth_stencil: Option<ID3D11DepthStencilState>,
    stencil_ref: u32,
    raster: Option<ID3D11RasterizerState>,
    viewports: Vec<D3D11_VIEWPORT>,
    scissors: Vec<RECT>,
}

impl Saved {
    /// # Safety
    /// `ctx` is a live context used by this thread only for the duration of the call.
    pub(crate) unsafe fn capture(ctx: &ID3D11DeviceContext, n: Slots) -> Self {
        // SAFETY (whole block): getters with output arrays of the declared lengths.
        unsafe {
            let mut s = Saved {
                layout: ctx.IAGetInputLayout().ok(),
                topology: ctx.IAGetPrimitiveTopology(),
                vs: None,
                hs: None,
                ds: None,
                gs: None,
                ps: None,
                cs: None,
                ps_srv: vec![None; n.srv],
                ps_cb: vec![None; n.cb],
                ps_sampler: vec![None; n.sampler],
                cs_srv: vec![None; n.srv],
                cs_cb: vec![None; n.cb],
                cs_sampler: vec![None; n.sampler],
                cs_uav: vec![None; n.uav],
                rtv: Default::default(),
                dsv: None,
                om_uav: Default::default(),
                blend: None,
                blend_factor: [0.0; 4],
                sample_mask: 0,
                depth_stencil: None,
                stencil_ref: 0,
                raster: ctx.RSGetState().ok(),
                viewports: Vec::new(),
                scissors: Vec::new(),
            };
            ctx.VSGetShader(&mut s.vs, None, None);
            ctx.HSGetShader(&mut s.hs, None, None);
            ctx.DSGetShader(&mut s.ds, None, None);
            ctx.GSGetShader(&mut s.gs, None, None);
            ctx.PSGetShader(&mut s.ps, None, None);
            ctx.CSGetShader(&mut s.cs, None, None);
            ctx.PSGetShaderResources(0, Some(&mut s.ps_srv));
            ctx.PSGetConstantBuffers(0, Some(&mut s.ps_cb));
            ctx.PSGetSamplers(0, Some(&mut s.ps_sampler));
            ctx.CSGetShaderResources(0, Some(&mut s.cs_srv));
            ctx.CSGetConstantBuffers(0, Some(&mut s.cs_cb));
            ctx.CSGetSamplers(0, Some(&mut s.cs_sampler));
            ctx.CSGetUnorderedAccessViews(0, Some(&mut s.cs_uav));
            ctx.OMGetRenderTargets(Some(&mut s.rtv), Some(&mut s.dsv));
            ctx.OMGetRenderTargetsAndUnorderedAccessViews(None, None, 0, Some(&mut s.om_uav));
            ctx.OMGetBlendState(
                Some(&mut s.blend),
                Some(&mut s.blend_factor),
                Some(&mut s.sample_mask),
            );
            ctx.OMGetDepthStencilState(Some(&mut s.depth_stencil), Some(&mut s.stencil_ref));
            let max = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
            let mut count = max;
            let mut vps = [D3D11_VIEWPORT::default(); 16];
            ctx.RSGetViewports(&mut count, Some(vps.as_mut_ptr()));
            s.viewports = vps[..count.min(max) as usize].to_vec();
            let mut count = max;
            let mut rects = [RECT::default(); 16];
            ctx.RSGetScissorRects(&mut count, Some(rects.as_mut_ptr()));
            s.scissors = rects[..count.min(max) as usize].to_vec();
            s
        }
    }

    /// # Safety
    /// As `capture`, on the same context.
    pub(crate) unsafe fn restore(&self, ctx: &ID3D11DeviceContext) {
        let keep = [u32::MAX; 8];
        // SAFETY (whole block): setters with arrays of their declared lengths; the views
        // and states are the context's own (kept alive by `self`). Targets are unbound
        // first so restoring shader resources cannot hit an output hazard with our RTV.
        unsafe {
            ctx.OMSetRenderTargets(None, None);
            ctx.CSSetUnorderedAccessViews(
                0,
                self.cs_uav.len() as u32,
                Some(self.cs_uav.as_ptr()),
                Some(keep.as_ptr()),
            );
            ctx.CSSetShader(self.cs.as_ref(), None);
            ctx.CSSetShaderResources(0, Some(&self.cs_srv));
            ctx.CSSetConstantBuffers(0, Some(&self.cs_cb));
            ctx.CSSetSamplers(0, Some(&self.cs_sampler));
            ctx.PSSetShader(self.ps.as_ref(), None);
            ctx.PSSetShaderResources(0, Some(&self.ps_srv));
            ctx.PSSetConstantBuffers(0, Some(&self.ps_cb));
            ctx.PSSetSamplers(0, Some(&self.ps_sampler));
            ctx.VSSetShader(self.vs.as_ref(), None);
            ctx.HSSetShader(self.hs.as_ref(), None);
            ctx.DSSetShader(self.ds.as_ref(), None);
            ctx.GSSetShader(self.gs.as_ref(), None);
            ctx.IASetInputLayout(self.layout.as_ref());
            ctx.IASetPrimitiveTopology(self.topology);
            ctx.RSSetState(self.raster.as_ref());
            ctx.RSSetViewports(Some(&self.viewports));
            ctx.RSSetScissorRects(Some(&self.scissors));
            ctx.OMSetBlendState(
                self.blend.as_ref(),
                Some(&self.blend_factor),
                self.sample_mask,
            );
            ctx.OMSetDepthStencilState(self.depth_stencil.as_ref(), self.stencil_ref);
            let n = self
                .rtv
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |i| i + 1);
            match self.om_uav.iter().position(Option::is_some) {
                Some(first) => {
                    let start = first.max(n);
                    ctx.OMSetRenderTargetsAndUnorderedAccessViews(
                        Some(&self.rtv[..n]),
                        self.dsv.as_ref(),
                        start as u32,
                        (8 - start) as u32,
                        Some(self.om_uav[start..].as_ptr()),
                        Some(keep.as_ptr()),
                    );
                }
                None => ctx.OMSetRenderTargets(Some(&self.rtv[..n]), self.dsv.as_ref()),
            }
        }
    }

    /// Differences to `o` (binding identities and values), for the offline check.
    pub(crate) fn diff(&self, o: &Saved) -> Vec<String> {
        fn raw<T: Interface>(v: &Option<T>) -> usize {
            v.as_ref().map_or(0, |x| x.as_raw() as usize)
        }
        fn list<T: Interface>(a: &[Option<T>], b: &[Option<T>]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| raw(x) == raw(y))
        }
        let mut d = Vec::new();
        let mut chk = |ok: bool, name: &str| {
            if !ok {
                d.push(name.to_owned());
            }
        };
        chk(raw(&self.layout) == raw(&o.layout), "input layout");
        chk(self.topology == o.topology, "topology");
        chk(raw(&self.vs) == raw(&o.vs), "VS");
        chk(raw(&self.hs) == raw(&o.hs), "HS");
        chk(raw(&self.ds) == raw(&o.ds), "DS");
        chk(raw(&self.gs) == raw(&o.gs), "GS");
        chk(raw(&self.ps) == raw(&o.ps), "PS");
        chk(raw(&self.cs) == raw(&o.cs), "CS");
        chk(list(&self.ps_srv, &o.ps_srv), "PS SRVs");
        chk(list(&self.ps_cb, &o.ps_cb), "PS CBs");
        chk(list(&self.ps_sampler, &o.ps_sampler), "PS samplers");
        chk(list(&self.cs_srv, &o.cs_srv), "CS SRVs");
        chk(list(&self.cs_cb, &o.cs_cb), "CS CBs");
        chk(list(&self.cs_sampler, &o.cs_sampler), "CS samplers");
        chk(list(&self.cs_uav, &o.cs_uav), "CS UAVs");
        chk(list(&self.rtv, &o.rtv), "RTVs");
        chk(raw(&self.dsv) == raw(&o.dsv), "DSV");
        chk(list(&self.om_uav, &o.om_uav), "OM UAVs");
        chk(
            raw(&self.blend) == raw(&o.blend)
                && self.blend_factor == o.blend_factor
                && self.sample_mask == o.sample_mask,
            "blend state",
        );
        chk(
            raw(&self.depth_stencil) == raw(&o.depth_stencil) && self.stencil_ref == o.stencil_ref,
            "depth-stencil state",
        );
        chk(raw(&self.raster) == raw(&o.raster), "rasterizer state");
        chk(self.viewports == o.viewports, "viewports");
        chk(self.scissors == o.scissors, "scissor rects");
        d
    }
}

// ----------------------------------------------------------------------- GPU side --

pub(crate) fn create_err(what: &str) -> impl Fn(windows::core::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

fn texture_desc(view: &ID3D11View) -> Result<D3D11_TEXTURE2D_DESC, String> {
    // SAFETY: plain getters on a live view and its resource.
    unsafe {
        let r = view.GetResource().map_err(create_err("view resource"))?;
        let t: ID3D11Texture2D = r
            .cast()
            .map_err(|_| "view of a non-2D resource".to_owned())?;
        let mut d = D3D11_TEXTURE2D_DESC::default();
        t.GetDesc(&mut d);
        Ok(d)
    }
}

struct AoTex {
    srv: ID3D11ShaderResourceView,
    uav: ID3D11UnorderedAccessView,
}

struct Targets {
    /// (depth texture size, half resolution).
    key: ([u32; 2], bool),
    lin_all: ID3D11ShaderResourceView,
    /// UAVs of MIP 0-4 (bound together at u0-u4 by the prefilter).
    lin_uav: Vec<Option<ID3D11UnorderedAccessView>>,
    /// [0]: AO / final, [1]: blur scratch.
    ao: [AoTex; 2],
    ao_dims: [u32; 2],
}

impl Targets {
    fn new(dev: &ID3D11Device, tex: [u32; 2], half: bool) -> Result<Self, String> {
        let dims = lin_dims(tex);
        let bind = (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32;
        let make = |size: [u32; 2], format, mips| -> Result<ID3D11Texture2D, String> {
            let d = D3D11_TEXTURE2D_DESC {
                Width: size[0],
                Height: size[1],
                MipLevels: mips,
                ArraySize: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: bind,
                ..Default::default()
            };
            let mut t = None;
            // SAFETY: valid descriptor and out pointer.
            unsafe { dev.CreateTexture2D(&d, None, Some(&mut t)) }
                .map_err(create_err("AO texture"))?;
            t.ok_or_else(|| "no AO texture".to_owned())
        };
        // FP32 view depths: FP16 (HBAO+'s default) doubled the mean occlusion of the open
        // test scene offline (normals from quantised depth) for ~6 % less time.
        let lin = make(dims, DXGI_FORMAT_R32_FLOAT, MIPS)?;
        let srv = |res: &ID3D11Texture2D| {
            let mut v = None;
            // SAFETY: valid resource and out pointer (view of all MIP levels).
            unsafe { dev.CreateShaderResourceView(res, None, Some(&mut v)) }
                .map_err(create_err("AO SRV"))?;
            v.ok_or_else(|| "no AO SRV".to_owned())
        };
        let uav = |res: &ID3D11Texture2D, mip: u32| {
            let d = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_UAV { MipSlice: mip },
                },
            };
            let mut v = None;
            // SAFETY: valid resource, descriptor and out pointer.
            unsafe { dev.CreateUnorderedAccessView(res, Some(&d), Some(&mut v)) }
                .map_err(create_err("AO UAV"))?;
            v.ok_or_else(|| "no AO UAV".to_owned())
        };
        let ao_dims = ao_size(dims, half);
        let ao_tex = || -> Result<AoTex, String> {
            let t = make(ao_dims, DXGI_FORMAT_R16_FLOAT, 1)?;
            Ok(AoTex {
                srv: srv(&t)?,
                uav: uav(&t, 0)?,
            })
        };
        Ok(Self {
            key: (tex, half),
            lin_all: srv(&lin)?,
            lin_uav: (0..MIPS)
                .map(|k| uav(&lin, k).map(Some))
                .collect::<Result<_, _>>()?,
            ao: [ao_tex()?, ao_tex()?],
            ao_dims,
        })
    }
}

/// Validated inputs of one `RenderAO` call.
struct Job {
    depth: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
    tex: [u32; 2],
    region: [u32; 4],
    params: ShaderParams,
    blend_mode: u32,
    custom_blend: Option<ID3D11BlendState>,
    blend_factor: Option<[f32; 4]>,
    /// Index into `QUALITIES` (selects the `cs_ao` permutation).
    quality: usize,
}

/// What the debug view shows: the final AO texture's valid size and the power exponent.
#[derive(Clone, Copy)]
struct Shown {
    ao_size: [u32; 2],
    power: f32,
}

struct Pipeline {
    device: ID3D11Device,
    prefilter: ID3D11ComputeShader,
    /// One per `QUALITIES` entry.
    ao: Vec<ID3D11ComputeShader>,
    blur: ID3D11ComputeShader,
    vs: ID3D11VertexShader,
    composite: ID3D11PixelShader,
    debug: ID3D11PixelShader,
    params: ID3D11Buffer,
    debug_params: ID3D11Buffer,
    point: ID3D11SamplerState,
    linear: ID3D11SamplerState,
    raster: ID3D11RasterizerState,
    no_depth: ID3D11DepthStencilState,
    overwrite: ID3D11BlendState,
    multiply: ID3D11BlendState,
    targets: Option<Targets>,
    shown: Option<Shown>,
}

// SAFETY: D3D11 devices and their child objects are free-threaded; the pipeline is used
// under `PIPE`'s mutex only.
unsafe impl Send for Pipeline {}

static PIPE: Mutex<Option<Pipeline>> = Mutex::new(None);

pub(crate) fn dynamic_cb(dev: &ID3D11Device, size: usize) -> Result<ID3D11Buffer, String> {
    let mut b = None;
    // SAFETY: valid descriptor and out pointer.
    unsafe {
        dev.CreateBuffer(
            &D3D11_BUFFER_DESC {
                ByteWidth: size.next_multiple_of(16) as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                ..Default::default()
            },
            None,
            Some(&mut b),
        )
    }
    .map_err(create_err("constant buffer"))?;
    b.ok_or_else(|| "no constant buffer".to_owned())
}

/// Writes `value` into a dynamic constant buffer (`WRITE_DISCARD`, valid on deferred
/// contexts).
pub(crate) fn upload<T: Copy>(
    ctx: &ID3D11DeviceContext,
    buf: &ID3D11Buffer,
    value: &T,
) -> Result<(), String> {
    let mut m = D3D11_MAPPED_SUBRESOURCE::default();
    // SAFETY: dynamic CPU-writable buffer of at least size_of::<T>() bytes; unmapped below.
    unsafe {
        ctx.Map(buf, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut m))
            .map_err(create_err("Map"))?;
        std::ptr::copy_nonoverlapping(
            (value as *const T).cast::<u8>(),
            m.pData.cast::<u8>(),
            size_of::<T>(),
        );
        ctx.Unmap(buf, 0);
    }
    Ok(())
}

impl Pipeline {
    fn new(dev: &ID3D11Device, code: &Bytecode) -> Result<Self, String> {
        // SAFETY (whole block): object creation with valid descriptors, bytecode compiled
        // for the declared stage, and out pointers.
        unsafe {
            let cs = |b: &[u8]| -> Result<ID3D11ComputeShader, String> {
                let mut s = None;
                dev.CreateComputeShader(b, None, Some(&mut s))
                    .map_err(create_err("compute shader"))?;
                s.ok_or_else(|| "no compute shader".to_owned())
            };
            let ps = |b: &[u8]| -> Result<ID3D11PixelShader, String> {
                let mut s = None;
                dev.CreatePixelShader(b, None, Some(&mut s))
                    .map_err(create_err("pixel shader"))?;
                s.ok_or_else(|| "no pixel shader".to_owned())
            };
            let mut vs = None;
            dev.CreateVertexShader(&code.vs, None, Some(&mut vs))
                .map_err(create_err("vertex shader"))?;
            let sampler = |filter| -> Result<ID3D11SamplerState, String> {
                let mut s = None;
                dev.CreateSamplerState(
                    &D3D11_SAMPLER_DESC {
                        Filter: filter,
                        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    },
                    Some(&mut s),
                )
                .map_err(create_err("sampler"))?;
                s.ok_or_else(|| "no sampler".to_owned())
            };
            let mut raster = None;
            dev.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut raster),
            )
            .map_err(create_err("rasterizer state"))?;
            let mut no_depth = None;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: false.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut no_depth),
            )
            .map_err(create_err("depth-stencil state"))?;
            let rgb = (D3D11_COLOR_WRITE_ENABLE_RED.0
                | D3D11_COLOR_WRITE_ENABLE_GREEN.0
                | D3D11_COLOR_WRITE_ENABLE_BLUE.0) as u8;
            let blend = |rt: D3D11_RENDER_TARGET_BLEND_DESC| -> Result<ID3D11BlendState, String> {
                let mut d = D3D11_BLEND_DESC::default();
                d.RenderTarget[0] = rt;
                let mut b = None;
                dev.CreateBlendState(&d, Some(&mut b))
                    .map_err(create_err("blend state"))?;
                b.ok_or_else(|| "no blend state".to_owned())
            };
            // GFSDK_SSAO_OVERWRITE_RGB / MULTIPLY_RGB: RGB written, alpha preserved.
            let overwrite = blend(D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                SrcBlend: D3D11_BLEND_ONE,
                DestBlend: D3D11_BLEND_ZERO,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_ZERO,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: rgb,
            })?;
            let multiply = blend(D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D11_BLEND_ZERO,
                DestBlend: D3D11_BLEND_SRC_COLOR,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ZERO,
                DestBlendAlpha: D3D11_BLEND_ONE,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: rgb,
            })?;
            Ok(Self {
                device: dev.clone(),
                prefilter: cs(&code.prefilter)?,
                ao: code.ao.iter().map(|b| cs(b)).collect::<Result<_, _>>()?,
                blur: cs(&code.blur)?,
                vs: vs.ok_or("no vertex shader")?,
                composite: ps(&code.composite)?,
                debug: ps(&code.debug)?,
                params: dynamic_cb(dev, size_of::<ShaderParams>())?,
                debug_params: dynamic_cb(dev, 16)?,
                point: sampler(D3D11_FILTER_MIN_MAG_MIP_POINT)?,
                linear: sampler(D3D11_FILTER_MIN_MAG_MIP_LINEAR)?,
                raster: raster.ok_or("no rasterizer state")?,
                no_depth: no_depth.ok_or("no depth-stencil state")?,
                overwrite,
                multiply,
                targets: None,
                shown: None,
            })
        }
    }

    fn record(&mut self, ctx: &ID3D11DeviceContext, job: &Job, half: bool) -> Result<(), String> {
        if self
            .targets
            .as_ref()
            .is_none_or(|t| t.key != (job.tex, half))
        {
            self.targets = None;
            self.targets = Some(Targets::new(&self.device, job.tex, half)?);
            log::line(&format!(
                "ao: targets for {}x{} ({} resolution)",
                job.tex[0],
                job.tex[1],
                if half { "half" } else { "full" }
            ));
        }
        let t = self.targets.as_ref().ok_or("no targets")?;
        let blend = match job.blend_mode {
            0 => Some(self.overwrite.clone()),
            1 => Some(self.multiply.clone()),
            _ => job.custom_blend.clone(),
        };
        // SAFETY: capture/restore on the caller's context, which this thread records.
        let saved = unsafe { Saved::capture(ctx, TOUCHED) };
        let r = upload(ctx, &self.params, &job.params).map(|()| {
            // SAFETY: recording on the caller's context with our own live objects.
            unsafe { self.passes(ctx, t, job, blend.as_ref()) }
        });
        // SAFETY: as above.
        unsafe { saved.restore(ctx) };
        r?;
        self.shown = Some(Shown {
            ao_size: [job.params.ints[0], job.params.ints[1]],
            power: job.params.amount[2],
        });
        Ok(())
    }

    /// # Safety
    /// `ctx` is recorded by this thread; `t` belongs to this pipeline's device.
    unsafe fn passes(
        &self,
        ctx: &ID3D11DeviceContext,
        t: &Targets,
        job: &Job,
        blend: Option<&ID3D11BlendState>,
    ) {
        let groups = |n: u32| n.div_ceil(8);
        let [x, y, w, h] = job.region;
        let ao = [job.params.ints[0], job.params.ints[1]];
        let none_uav: [Option<ID3D11UnorderedAccessView>; MIPS as usize] = Default::default();
        let keep = [u32::MAX; MIPS as usize];
        let set_uav = |u: Option<&ID3D11UnorderedAccessView>| {
            let a = [u.cloned()];
            // SAFETY: one-element arrays for one slot.
            unsafe { ctx.CSSetUnorderedAccessViews(0, 1, Some(a.as_ptr()), Some(keep.as_ptr())) };
        };
        // SAFETY (whole block): bindings of our own objects and the engine's views passed
        // to RenderAO; every slot used here is in `TOUCHED`.
        unsafe {
            ctx.OMSetRenderTargets(None, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetSamplers(
                0,
                Some(&[Some(self.point.clone()), Some(self.linear.clone())]),
            );
            // Linear depth MIP 0-4 of the region, one dispatch (16x16 pixels per group).
            ctx.CSSetShader(&self.prefilter, None);
            ctx.CSSetShaderResources(0, Some(&[Some(job.depth.clone()), None, None]));
            ctx.CSSetUnorderedAccessViews(0, MIPS, Some(t.lin_uav.as_ptr()), Some(keep.as_ptr()));
            ctx.Dispatch(w.div_ceil(16), h.div_ceil(16), 1);
            ctx.CSSetUnorderedAccessViews(0, MIPS, Some(none_uav.as_ptr()), Some(keep.as_ptr()));
            // Visibility-bitmask AO.
            ctx.CSSetShader(&self.ao[job.quality], None);
            ctx.CSSetShaderResources(0, Some(&[None, Some(t.lin_all.clone()), None]));
            set_uav(Some(&t.ao[1].uav));
            ctx.Dispatch(groups(ao[0]), groups(ao[1]), 1);
            // Blur (ao[1] -> ao[0], both axes in one dispatch).
            ctx.CSSetShader(&self.blur, None);
            set_uav(None);
            ctx.CSSetShaderResources(2, Some(&[Some(t.ao[1].srv.clone())]));
            set_uav(Some(&t.ao[0].uav));
            ctx.Dispatch(groups(ao[0]), groups(ao[1]), 1);
            set_uav(None);
            ctx.CSSetShaderResources(0, Some(&[None, None, None]));
            // Write the engine's target inside the region.
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.composite, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.PSSetSamplers(
                0,
                Some(&[Some(self.point.clone()), Some(self.linear.clone())]),
            );
            ctx.PSSetShaderResources(
                0,
                Some(&[None, Some(t.lin_all.clone()), Some(t.ao[0].srv.clone())]),
            );
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: x as f32,
                TopLeftY: y as f32,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(blend, job.blend_factor.as_ref(), D3D11_DEFAULT_SAMPLE_MASK);
            ctx.OMSetDepthStencilState(&self.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(job.rtv.clone())]), None);
            ctx.Draw(3, 0);
        }
    }

    /// Full-screen view of the last AO result (`ao_debug`).
    fn draw_debug(
        &self,
        ctx: &ID3D11DeviceContext,
        rtv: &ID3D11RenderTargetView,
        size: [u32; 2],
    ) -> Result<(), String> {
        let (Some(t), Some(shown)) = (self.targets.as_ref(), self.shown) else {
            return Ok(());
        };
        let p = [
            shown.ao_size[0] as f32 / t.ao_dims[0] as f32 / size[0] as f32,
            shown.ao_size[1] as f32 / t.ao_dims[1] as f32 / size[1] as f32,
            shown.power,
            0.0,
        ];
        upload(ctx, &self.debug_params, &p)?;
        // SAFETY (whole block): immediate context under the engine lock (caller); state
        // captured and restored around our draw.
        unsafe {
            let saved = Saved::capture(ctx, TOUCHED);
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.debug, None);
            ctx.PSSetConstantBuffers(2, Some(&[Some(self.debug_params.clone())]));
            ctx.PSSetSamplers(
                0,
                Some(&[Some(self.point.clone()), Some(self.linear.clone())]),
            );
            ctx.PSSetShaderResources(2, Some(&[Some(t.ao[0].srv.clone())]));
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size[0] as f32,
                Height: size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(None, None, D3D11_DEFAULT_SAMPLE_MASK);
            ctx.OMSetDepthStencilState(&self.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.Draw(3, 0);
            saved.restore(ctx);
        }
        Ok(())
    }
}

/// Main thread before the engine present, immediate context under the engine lock
/// (`overlay.rs`, `ao_debug = 1`): draws the last AO result over the back buffer.
pub fn draw_debug(
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    size: [u32; 2],
) -> Result<(), String> {
    let Ok(g) = PIPE.try_lock() else {
        return Ok(());
    };
    let Some(p) = g.as_ref() else {
        return Ok(());
    };
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    if dev.as_raw() != p.device.as_raw() {
        return Ok(());
    }
    p.draw_debug(ctx, rtv, size)
}

// ------------------------------------------------------------------- replacement --

fn describe_first_call(
    input: &InputData,
    params: &Parameters,
    output: &Output,
    mask: u32,
    bound: Option<&D3D11_VIEWPORT>,
) -> String {
    let view = |p: *mut c_void, srv: bool| -> String {
        if p.is_null() {
            return "null".to_owned();
        }
        let (fmt, dim) = if srv {
            // SAFETY: the engine's live SRV passed to RenderAO.
            match unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&p) } {
                Some(v) => {
                    let mut d = D3D11_SHADER_RESOURCE_VIEW_DESC::default();
                    // SAFETY: plain getter.
                    unsafe { v.GetDesc(&mut d) };
                    (d.Format, format!("dimension {}", d.ViewDimension.0))
                }
                None => return "null".to_owned(),
            }
        } else {
            // SAFETY: the engine's live RTV passed to RenderAO.
            match unsafe { ID3D11RenderTargetView::from_raw_borrowed(&p) } {
                Some(v) => {
                    let mut d = D3D11_RENDER_TARGET_VIEW_DESC::default();
                    // SAFETY: plain getter.
                    unsafe { v.GetDesc(&mut d) };
                    (d.Format, format!("dimension {}", d.ViewDimension.0))
                }
                None => return "null".to_owned(),
            }
        };
        // SAFETY: `p` is a live view (checked above).
        let res = unsafe { ID3D11View::from_raw_borrowed(&p) }.map(texture_desc);
        match res {
            Some(Ok(t)) => format!(
                "view format {} ({dim}), texture {}x{} format {} samples {} mips {} bind {:#x}",
                fmt.0, t.Width, t.Height, t.Format.0, t.SampleDesc.Count, t.MipLevels, t.BindFlags
            ),
            Some(Err(e)) => format!("view format {} ({dim}), {e}", fmt.0),
            None => "null".to_owned(),
        }
    };
    let d = &input.depth;
    let vp = bound.map_or("none".to_owned(), |v| {
        format!(
            "{}x{} at ({}, {})",
            v.Width, v.Height, v.TopLeftX, v.TopLeftY
        )
    });
    format!(
        "first RenderAO call: mask {mask}; depth type {} layout {} meters->view {} viewport enable {} ({}x{} at {},{}); projection {:?}; normals enable {}; depth SRV: {}; output RTV: {}; blend mode {} two-pass {}; params radius {} bias {} small {} large {} power {} fg {}/{} bg {}/{} storage {} clamp {} threshold {}/{}/{} blur {}/{}/{}; bound viewport {vp}",
        d.depth_texture_type,
        d.projection.layout,
        d.meters_to_view_space_units,
        d.viewport.enable,
        d.viewport.width,
        d.viewport.height,
        d.viewport.top_left_x,
        d.viewport.top_left_y,
        d.projection.data,
        input.normal.enable,
        view(d.depth_srv, true),
        view(output.rtv, false),
        output.blend.mode,
        output.two_pass.enable,
        params.radius,
        params.bias,
        params.small_scale_ao,
        params.large_scale_ao,
        params.power_exponent,
        params.foreground_ao.enable,
        params.foreground_ao.view_depth,
        params.background_ao.enable,
        params.background_ao.view_depth,
        params.depth_storage,
        params.depth_clamp_mode,
        params.depth_threshold.enable,
        params.depth_threshold.max_view_depth,
        params.depth_threshold.sharpness,
        params.blur.enable,
        params.blur.radius,
        params.blur.sharpness,
    )
}

/// One `RenderAO` call: Ok(true) = our AO was recorded, Ok(false) = HBAO+ should run
/// (`ao_mode = hbao`, shaders still compiling), Err = HBAO+ runs and the reason is logged.
///
/// # Safety
/// The arguments are those of a `RenderAO` call (valid for its duration).
unsafe fn replace(
    ctx: *mut c_void,
    input: *const InputData,
    params: *const Parameters,
    output: *const Output,
    mask: u32,
    bound: Option<&D3D11_VIEWPORT>,
) -> Result<bool, String> {
    if input.is_null() || params.is_null() || output.is_null() {
        return Err("null RenderAO argument".to_owned());
    }
    // SAFETY: non-null pointers to the caller's structs (layouts asserted above).
    let (input, params, output) = unsafe { (&*input, &*params, &*output) };
    // SAFETY: the caller's live device context.
    let ctx =
        unsafe { ID3D11DeviceContext::from_raw_borrowed(&ctx) }.ok_or("null device context")?;
    if !FIRST_CALL_LOGGED.swap(true, Ordering::Relaxed) {
        let info = describe_first_call(input, params, output, mask, bound);
        log::line(&format!("ao: {info}"));
        *lock(&INFO) = Some(info);
    }
    let s = settings::get();
    if !s.ao_void {
        return Ok(false);
    }
    if !BYTECODE_READY.load(Ordering::Acquire) {
        return Ok(false);
    }
    let code = bytecode().map_err(|e| format!("shader compilation failed: {e}"))?;
    let job = validate(input, params, output, mask, bound, &s)?;
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    let mut g = lock(&PIPE);
    if g.as_ref().is_none_or(|p| p.device.as_raw() != dev.as_raw()) {
        *g = None;
        *g = Some(Pipeline::new(&dev, code)?);
    }
    g.as_mut()
        .ok_or("no pipeline")?
        .record(ctx, &job, s.ao_half_res)?;
    Ok(true)
}

fn validate(
    input: &InputData,
    params: &Parameters,
    output: &Output,
    mask: u32,
    bound: Option<&D3D11_VIEWPORT>,
    s: &Settings,
) -> Result<Job, String> {
    if mask != RENDER_AO {
        return Err(format!("render mask {mask} (debug view) not handled"));
    }
    let d = &input.depth;
    if d.depth_texture_type != 0 {
        return Err(format!(
            "depth texture type {} (only hardware depths handled)",
            d.depth_texture_type
        ));
    }
    if input.normal.enable != 0 {
        return Err("input normals not handled".to_owned());
    }
    if output.two_pass.enable != 0 {
        return Err("two-pass blending not handled".to_owned());
    }
    if output.blend.mode > 2 {
        return Err(format!("blend mode {}", output.blend.mode));
    }
    let proj = projection(&d.projection.data, d.projection.layout)?;
    // SAFETY: the engine's live views passed to RenderAO.
    let depth = unsafe { ID3D11ShaderResourceView::from_raw_borrowed(&d.depth_srv) }
        .ok_or("null depth SRV")?
        .clone();
    // SAFETY: as above.
    let rtv = unsafe { ID3D11RenderTargetView::from_raw_borrowed(&output.rtv) }
        .ok_or("null output RTV")?
        .clone();
    let mut sd = D3D11_SHADER_RESOURCE_VIEW_DESC::default();
    // SAFETY: plain getter.
    unsafe { depth.GetDesc(&mut sd) };
    if sd.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D {
        return Err(format!(
            "depth SRV dimension {} (multisampled/array depth not handled)",
            sd.ViewDimension.0
        ));
    }
    // SAFETY: union member of a TEXTURE2D view.
    if unsafe { sd.Anonymous.Texture2D.MostDetailedMip } != 0 {
        return Err("depth SRV does not start at MIP 0".to_owned());
    }
    let dt = texture_desc(&depth.cast().map_err(create_err("depth view"))?)?;
    let rt = texture_desc(&rtv.cast().map_err(create_err("output view"))?)?;
    let tex = [dt.Width, dt.Height];
    if [rt.Width, rt.Height] != tex {
        return Err(format!(
            "output {}x{} differs from depth {}x{}",
            rt.Width, rt.Height, tex[0], tex[1]
        ));
    }
    let region = region(&d.viewport, bound, tex).ok_or("empty viewport")?;
    let custom_blend = if output.blend.mode == 2 {
        // SAFETY: the caller's blend state (may be null = default state).
        unsafe { ID3D11BlendState::from_raw_borrowed(&output.blend.custom.blend_state) }.cloned()
    } else {
        None
    };
    let blend_factor = (output.blend.mode == 2 && !output.blend.custom.blend_factor.is_null())
        // SAFETY: HBAO+ reads 4 floats from a non-null pBlendFactor.
        .then(|| unsafe { *output.blend.custom.blend_factor.cast::<[f32; 4]>() });
    Ok(Job {
        depth,
        rtv,
        tex,
        region,
        params: shader_params(&proj, region, tex, params, d.meters_to_view_space_units, s),
        blend_mode: output.blend.mode,
        custom_blend,
        blend_factor,
        quality: usize::from(s.ao_quality).min(QUALITIES.len() - 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clip-space projection with the parsed matrix (row-vector convention as HBAO+).
    fn project(data: &[f32; 16], layout: u32, v: [f32; 3]) -> ([f32; 2], f32) {
        let at = |r: usize, c: usize| {
            if layout == 1 {
                data[4 * c + r]
            } else {
                data[4 * r + c]
            }
        };
        let clip: Vec<f32> = (0..4)
            .map(|c| v[0] * at(0, c) + v[1] * at(1, c) + v[2] * at(2, c) + at(3, c))
            .collect();
        ([clip[0] / clip[3], clip[1] / clip[3]], clip[2] / clip[3])
    }

    fn transpose(m: &[f32; 16]) -> [f32; 16] {
        std::array::from_fn(|i| m[4 * (i % 4) + i / 4])
    }

    #[test]
    fn projection_round_trip() {
        let col = engine_projection(2.4, 1.0, 0.01, 7000.0, [0.0003, -0.0002]);
        // Column-major (as the game passes it) and the same matrix row-major.
        for (data, layout) in [(col, 1), (transpose(&col), 0)] {
            let p = projection(&data, layout).unwrap();
            // Right-handed engine view space: visible points have negative z.
            for v in [[0.3, -1.2, -4.0], [-2.0, 0.5, -0.5], [10.0, 3.0, -900.0]] {
                let (ndc, d) = project(&data, layout, v);
                assert!((0.0..=1.0).contains(&d), "{d}");
                let z = p.depth_mul / (d - p.depth_add);
                assert!((z / -v[2] - 1.0).abs() < 1e-4, "{z} {v:?}");
                let x = z * (ndc[0] - p.jitter[0]) / p.p00;
                let y = z * (ndc[1] - p.jitter[1]) / p.p11;
                assert!((x - v[0]).abs() < 1e-3 * z && (y - v[1]).abs() < 1e-3 * z);
            }
        }
        // Reversed-Z: near plane -> 1, far plane -> 0.
        let (_, near) = project(&col, 1, [0.0, 0.0, -0.01]);
        let (_, far) = project(&col, 1, [0.0, 0.0, -7000.0]);
        assert!((near - 1.0).abs() < 1e-5 && far.abs() < 1e-5);
    }

    #[test]
    fn projection_rejects_other_forms() {
        let mut m = engine_projection(2.4, 1.0, 0.01, 7000.0, [0.0; 2]);
        assert!(projection(&m, 2).is_err());
        // Read with the wrong layout, P23 lands in row 3.
        assert!(projection(&m, 0).is_err());
        m[14] = 0.5;
        assert!(projection(&m, 1).is_err());
        let mut n = engine_projection(2.4, 1.0, 0.01, 7000.0, [0.0; 2]);
        n[0] = f32::NAN;
        assert!(projection(&n, 1).is_err());
    }

    #[test]
    fn regions() {
        let off = InputViewport {
            enable: 0,
            top_left_x: 0,
            top_left_y: 0,
            width: 0,
            height: 0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let vp = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: 2304.0,
            Height: 968.0,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        assert_eq!(region(&off, None, [3440, 1440]), Some([0, 0, 3440, 1440]));
        assert_eq!(
            region(&off, Some(&vp), [3440, 1440]),
            Some([0, 0, 2304, 968])
        );
        let big = D3D11_VIEWPORT {
            Width: 5000.0,
            ..vp
        };
        assert_eq!(
            region(&off, Some(&big), [3440, 1440]),
            Some([0, 0, 3440, 968])
        );
        let on = InputViewport {
            enable: 1,
            top_left_x: 8,
            top_left_y: 4,
            width: 100,
            height: 50,
            ..off
        };
        assert_eq!(region(&on, Some(&vp), [3440, 1440]), Some([8, 4, 100, 50]));
        let outside = InputViewport {
            top_left_x: 4000,
            ..on
        };
        assert_eq!(region(&outside, None, [3440, 1440]), None);
        assert_eq!(ao_size([2305, 969], true), [1153, 485]);
        assert_eq!(lin_dims([3440, 1441]), [3440, 1456]);
    }

    #[test]
    fn game_parameters_map_to_shader_constants() {
        let s = Settings {
            ao_half_res: false,
            ..Settings::default()
        };
        let col = engine_projection(3440.0 / 1440.0, 1.2, 0.01, 7000.0, [0.0; 2]);
        let p = projection(&col, 1).unwrap();
        let k = shader_params(
            &p,
            [0, 0, 3440, 1440],
            [3440, 1440],
            &Parameters::game(1.0),
            1.0,
            &s,
        );
        // Radius 0.7 m at view depth 1: 0.7 * 0.5 * P11 * 1440 pixels.
        assert!((k.screen[0] - 0.7 * 0.5 * p.p11 * 1440.0).abs() < 1e-2);
        // Foreground AO at 4 m caps the screen radius at the radius seen at 4 m.
        assert!((k.screen[1] - k.screen[0] / 4.0).abs() < 1e-3 && k.screen[2] == 0.0);
        assert!((k.screen[3] - 0.1f32.asin()).abs() < 1e-6);
        assert_eq!(k.amount[2], 1.75);
        assert!((k.amount[1] / k.amount[0] - 1.75).abs() < 1e-5);
        assert_eq!(k.ints, [3440, 1440, 0, 0]);
        let (_, slices, steps) = QUALITIES[usize::from(DEFAULT_QUALITY)];
        assert_eq!((k.lp[0], k.lp[1]), (slices as f32, steps as f32));
        // Half resolution at render scale 0.67 in a 3440x1440 texture.
        let half = Settings {
            ao_half_res: true,
            ao_radius_scale: 2.0,
            ..s
        };
        let k = shader_params(
            &p,
            [0, 0, 2304, 968],
            [3440, 1440],
            &Parameters::game(0.67),
            1.0,
            &half,
        );
        assert_eq!(k.ints, [1152, 484, 0, 0]);
        assert_eq!((k.tex[2], k.tex[3]), (2.0, 1.0));
        assert!((k.depth[3] - 1.4).abs() < 1e-6);
    }

    #[test]
    fn quality_names() {
        for (i, (n, _, _)) in QUALITIES.iter().enumerate() {
            assert_eq!(parse_quality(n), Some(i as u8));
            assert_eq!(quality_name(i as u8), *n);
        }
        assert_eq!(parse_quality("max"), None);
        assert_eq!(quality_name(DEFAULT_QUALITY), "high");
    }
}
