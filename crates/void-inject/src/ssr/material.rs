//! Per-pixel material target for the reflections (docs/reflections.md "Material target",
//! STATUS E072): the lit forward pixel shaders are patched at creation (`patch.rs`) to write
//! `o2` = (octahedral world normal, perceptual roughness, sqrt(F0)), forward shaders of the
//! excluded classes (vegetation, hair, decals) alpha 1/255 instead, and `o3` = (sqrt(F0.rgb),
//! capture kind); this module binds our two R8G8B8A8 targets at slots 2 and 3 in the forward
//! passes and hands them to the reflections.
//!
//! - Shader creation: `trace.rs`'s `CreatePixelShader` hook calls [`create_pixel_shader`],
//!   which creates the patched bytecode in place of the original (the original on any error).
//!   Installed at startup only (`ssr = 1` and `ssr_material = auto`): shaders the game created
//!   before cannot be patched later. Unbound render-target writes are discarded (D3D11), so a
//!   patched shader draws exactly like the original while our target is not bound: the live
//!   switch `ssr_material` binds the target or not; no per-draw shader substitution.
//! - Forward passes (O, IDB): `ARK_RENDERPASS_FORWARD_WORLD` / `FORWARD_DYNAMIC` callbacks call
//!   `sub_140311AA0(pass, dynamic)`; every render-target change inside goes through
//!   `sub_1402BC370(engineCtx, count, imageViews, depthView, flags)` (OMSetRenderTargets,
//!   count 1 = `hdrBuffer`, 2 = + `sssBuffer` / `skyCloudMaskBuffer`). Both are detoured: the
//!   pass detour marks the thread (and clears our targets at the start of `FORWARD_WORLD`),
//!   the setter detour re-binds the engine's targets plus ours at slots 2 and 3 on that thread.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use hex_literal::hex;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{HRESULT, Interface};

use super::patch;
use crate::{log, mem, settings};

/// `sub_140311AA0(pass, dynamic)`: body of both forward-pass callbacks.
const FORWARD_RVA: u64 = 0x31_1AA0;
const FORWARD_HEAD: [u8; 16] = hex!("48 8B C4 88 50 10 48 89 48 08 55 53 56 57 41 54");
/// `sub_1402BC370(engineCtx, count, imageViews, depthView, flags)`: render-target setter.
const SET_TARGETS_RVA: u64 = 0x2B_C370;
const SET_TARGETS_HEAD: [u8; 16] = hex!("48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 20 4C");
const PASS_ENGINE_CTX: u64 = 0x108;

static PATCHING: AtomicBool = AtomicBool::new(false);
static PATCHED: AtomicU64 = AtomicU64::new(0);
static EXCLUDED: AtomicU64 = AtomicU64::new(0);
static UNPATCHED_LIT: AtomicU64 = AtomicU64::new(0);
static CREATE_FALLBACK: AtomicU64 = AtomicU64::new(0);
static BOUND: AtomicU64 = AtomicU64::new(0);
static HOOKED: AtomicBool = AtomicBool::new(false);
static FORWARD_ORIG: AtomicUsize = AtomicUsize::new(0);
static SET_TARGETS_ORIG: AtomicUsize = AtomicUsize::new(0);
static SKIP_REASONS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// Engine context of the forward pass this thread records (0 = none).
    static FORWARD_CTX: Cell<u64> = const { Cell::new(0) };
}

/// State for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// Shader patching is active (installed at startup).
    pub patching: bool,
    pub patched: u64,
    /// Of `patched`: excluded classes (vegetation, hair, decals) writing the exclusion marker.
    pub excluded: u64,
    /// Lit forward shaders (and excluded-class forward shaders) left unpatched, with reasons.
    pub unpatched: u64,
    pub reasons: Vec<(String, u64)>,
    /// Patched bytecode rejected by `CreatePixelShader` (original used).
    pub create_fallback: u64,
    /// Forward-pass render-target bindings with our target added.
    pub bound: u64,
    pub hooked: bool,
}

pub fn status() -> Status {
    Status {
        patching: PATCHING.load(Ordering::Relaxed),
        patched: PATCHED.load(Ordering::Relaxed),
        excluded: EXCLUDED.load(Ordering::Relaxed),
        unpatched: UNPATCHED_LIT.load(Ordering::Relaxed),
        reasons: SKIP_REASONS
            .lock()
            .map(|r| r.iter().map(|(s, n)| ((*s).to_owned(), *n)).collect())
            .unwrap_or_default(),
        create_fallback: CREATE_FALLBACK.load(Ordering::Relaxed),
        bound: BOUND.load(Ordering::Relaxed),
        hooked: HOOKED.load(Ordering::Relaxed),
    }
}

/// The material target is to be bound and read (live `ssr` and `ssr_material`).
fn wanted() -> bool {
    let s = settings::get();
    s.ssr && s.ssr_material
}

// ------------------------------------------------------------------- shader patch --

/// Startup (before the game creates its shaders): patching is wanted for this run.
pub fn enable_patching() {
    PATCHING.store(true, Ordering::Release);
    log::line("ssr: lit forward pixel shaders get the material output (o2) at creation");
}

pub fn patching() -> bool {
    PATCHING.load(Ordering::Acquire)
}

pub(crate) type CreateShaderFn = unsafe extern "system" fn(
    *mut c_void,
    *const c_void,
    usize,
    *mut c_void,
    *mut *mut c_void,
) -> HRESULT;

fn note_skip(reason: &'static str) {
    let mut r = SKIP_REASONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match r.iter_mut().find(|(s, _)| *s == reason) {
        Some(e) => e.1 += 1,
        None => r.push((reason, 1)),
    }
}

/// `CreatePixelShader` with the material output for lit forward shaders and the exclusion
/// marker for forward shaders of the excluded classes (`patch::Class`).
///
/// # Safety
/// The arguments are those of an `ID3D11Device::CreatePixelShader` call; `orig` is the
/// original implementation.
pub(crate) unsafe fn create_pixel_shader(
    orig: CreateShaderFn,
    this: *mut c_void,
    code: *const c_void,
    len: usize,
    linkage: *mut c_void,
    out: *mut *mut c_void,
) -> HRESULT {
    let patched = if code.is_null() || len == 0 || len > 16 << 20 {
        None
    } else {
        // SAFETY: the game passed `len` bytes of bytecode.
        let bytes = unsafe { std::slice::from_raw_parts(code.cast::<u8>(), len) };
        std::panic::catch_unwind(|| {
            let s = patch::shader(bytes);
            if !s.candidate() {
                return None;
            }
            match patch::patch(bytes, &s) {
                Ok(p) => Some(p),
                Err(e) => {
                    UNPATCHED_LIT.fetch_add(1, Ordering::Relaxed);
                    note_skip(e);
                    None
                }
            }
        })
        .ok()
        .flatten()
    };
    if let Some(p) = patched {
        // SAFETY: original implementation with our bytecode (valid for the call).
        let hr = unsafe { orig(this, p.code.as_ptr().cast(), p.code.len(), linkage, out) };
        if hr.is_ok() {
            PATCHED.fetch_add(1, Ordering::Relaxed);
            if p.class.excluded() {
                EXCLUDED.fetch_add(1, Ordering::Relaxed);
            }
            return hr;
        }
        if CREATE_FALLBACK.fetch_add(1, Ordering::Relaxed) < 3 {
            log::line(&format!(
                "ssr: patched pixel shader rejected ({hr:?}); original used"
            ));
        }
    }
    // SAFETY: arguments passed through unchanged.
    unsafe { orig(this, code, len, linkage, out) }
}

// ----------------------------------------------------------------------- target --

/// One RGBA8 target of the forward passes.
struct View {
    tex: ID3D11Texture2D,
    rtv: ID3D11RenderTargetView,
    srv: ID3D11ShaderResourceView,
}

struct Target {
    device: ID3D11Device,
    size: [u32; 2],
    /// `o2` (slot 2): normal, roughness, sqrt(F0) luminance / class marker.
    material: View,
    /// `o3` (slot 3): sqrt(F0.rgb), capture kind.
    f0: View,
    /// Frame counter at the last clear (start of `FORWARD_WORLD`).
    cleared: Option<u32>,
}

// SAFETY: D3D11 objects are free-threaded; used under `TARGET`'s mutex.
unsafe impl Send for Target {}

static TARGET: Mutex<Option<Target>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<Target>> {
    TARGET
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn frame() -> Option<u32> {
    mem::u32_at(crate::fsr::base() + crate::fsr::RS_RVA + crate::fsr::RS_FRAME)
}

fn create_view(dev: &ID3D11Device, size: [u32; 2], what: &str) -> Result<View, String> {
    let d = D3D11_TEXTURE2D_DESC {
        Width: size[0],
        Height: size[1],
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
        ..Default::default()
    };
    let (mut t, mut r, mut s) = (None, None, None);
    // SAFETY (whole block): creation with valid descriptors and out pointers.
    unsafe {
        dev.CreateTexture2D(&d, None, Some(&mut t))
            .map_err(|e| format!("{what} texture: {e}"))?;
        let t = t.as_ref().ok_or_else(|| format!("no {what} texture"))?;
        dev.CreateRenderTargetView(t, None, Some(&mut r))
            .map_err(|e| format!("{what} RTV: {e}"))?;
        dev.CreateShaderResourceView(t, None, Some(&mut s))
            .map_err(|e| format!("{what} SRV: {e}"))?;
    }
    Ok(View {
        tex: t.ok_or_else(|| format!("no {what} texture"))?,
        rtv: r.ok_or_else(|| format!("no {what} RTV"))?,
        srv: s.ok_or_else(|| format!("no {what} SRV"))?,
    })
}

fn create_target(dev: &ID3D11Device, size: [u32; 2]) -> Result<Target, String> {
    let material = create_view(dev, size, "material")?;
    let f0 = create_view(dev, size, "F0 colour")?;
    log::line(&format!(
        "ssr: material targets {}x{} (slots 2 and 3)",
        size[0], size[1]
    ));
    Ok(Target {
        device: dev.clone(),
        size,
        material,
        f0,
        cleared: None,
    })
}

/// The target for the TAA site (same device and texture size, cleared by `FORWARD_WORLD` in
/// this or the previous frame counter value), when `ssr_material` is on.
fn fresh<R>(dev: &ID3D11Device, size: [u32; 2], f: impl FnOnce(&Target) -> R) -> Option<R> {
    if !settings::get().ssr_material {
        return None;
    }
    let g = lock();
    let t = g.as_ref()?;
    let recent = match (t.cleared, frame()) {
        (Some(c), Some(now)) => now.wrapping_sub(c) <= 1,
        (Some(_), None) => true,
        _ => false,
    };
    (t.device.as_raw() == dev.as_raw() && t.size == size && recent).then(|| f(t))
}

/// The material SRV (`o2`) for the TAA site (see [`fresh`]).
pub(crate) fn for_frame(dev: &ID3D11Device, size: [u32; 2]) -> Option<ID3D11ShaderResourceView> {
    fresh(dev, size, |t| t.material.srv.clone())
}

/// The F0 colour SRV (`o3`) for the TAA site (see [`fresh`]).
pub(crate) fn f0_for_frame(dev: &ID3D11Device, size: [u32; 2]) -> Option<ID3D11ShaderResourceView> {
    fresh(dev, size, |t| t.f0.srv.clone())
}

/// D3D12-readable copies of `o2` and `o3` (photo mode: AO normals, class masks). Made at the
/// TAA site each frame once [`shared_d3d12`] was called; plain copies, so the game's own render
/// targets keep their (non-shared) layout.
#[derive(Clone)]
pub struct SharedMaterial {
    /// `o2`: octahedral world normal, perceptual roughness, sqrt(F0) / class (alpha byte 0 none,
    /// 1 excluded, >= 2 lit).
    pub material: ID3D12Resource,
    /// `o3`: sqrt(F0.rgb), alpha 1 captured / 0.5 luminance only / 0 none.
    pub f0: ID3D12Resource,
    pub size: [u32; 2],
    /// Frame counter of the last copy.
    pub frame: Option<u32>,
}

struct Copies {
    shared: SharedMaterial,
    /// The D3D11 side of the shared textures (copy destinations).
    material: ID3D11Resource,
    f0: ID3D11Resource,
    device: usize,
}

// SAFETY: D3D11/D3D12 objects are free-threaded; used under `COPIES`' mutex.
unsafe impl Send for Copies {}

static SHARE_WANTED: AtomicBool = AtomicBool::new(false);
static COPIES: Mutex<Option<Copies>> = Mutex::new(None);

/// The D3D12-readable copies (RGBA8_UNORM, COMMON; written on the D3D11 context that records
/// the TAA site, so a reader on another queue waits for that work with its own fence). The
/// first call starts the copies (None until a frame copied them).
pub fn shared_d3d12() -> Option<SharedMaterial> {
    SHARE_WANTED.store(true, Ordering::Relaxed);
    COPIES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(|c| c.shared.clone())
}

/// TAA site (`ssr.rs`, after the reflections): copies the frame's targets into the shared
/// textures once requested.
pub(crate) fn copy_shared(ctx: &ID3D11DeviceContext) {
    if !SHARE_WANTED.load(Ordering::Relaxed) {
        return;
    }
    let Some(rt) = crate::rt::device::get() else {
        return;
    };
    let g = lock();
    let Some(t) = g.as_ref() else {
        return;
    };
    let id = t.device.as_raw() as usize;
    let mut copies = COPIES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if copies
        .as_ref()
        .is_none_or(|c| c.shared.size != t.size || c.device != id)
    {
        // D3D11-created (NT handle) and opened on the ray-tracing device; the SRV gives the
        // D3D11 texture back.
        let mk = || -> Result<(ID3D12Resource, ID3D11Resource), String> {
            let s = crate::rt::device::shared_texture(
                &rt,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                t.size[0],
                t.size[1],
                false,
                true,
            )?;
            let srv = s.srv.ok_or("no SRV")?;
            // SAFETY: plain getter.
            let tex = unsafe { srv.GetResource() }.map_err(|e| e.to_string())?;
            Ok((s.res, tex))
        };
        *copies = match (mk(), mk()) {
            (Ok(m), Ok(f)) => Some(Copies {
                shared: SharedMaterial {
                    material: m.0,
                    f0: f.0,
                    size: t.size,
                    frame: None,
                },
                material: m.1,
                f0: f.1,
                device: id,
            }),
            (Err(e), _) | (_, Err(e)) => {
                log::line(&format!("ssr: shared material copies: {e}"));
                SHARE_WANTED.store(false, Ordering::Relaxed);
                None
            }
        };
    }
    let Some(c) = copies.as_mut() else {
        return;
    };
    // SAFETY: copies between textures of the same device, size and format; the shared ones
    // are only read by D3D12 after this context's work.
    unsafe {
        ctx.CopyResource(&c.material, &t.material.tex);
        ctx.CopyResource(&c.f0, &t.f0.tex);
    }
    c.shared.frame = frame();
}

/// Offline check: installs `srv` of a `size` target as if a forward pass had cleared and
/// filled it in the current frame.
pub(crate) fn set_for_test(
    dev: &ID3D11Device,
    size: [u32; 2],
) -> Result<ID3D11RenderTargetView, String> {
    let mut t = create_target(dev, size)?;
    t.cleared = Some(frame().unwrap_or(0));
    let rtv = t.material.rtv.clone();
    *lock() = Some(t);
    Ok(rtv)
}

/// Offline check: the F0 colour texture installed by [`set_for_test`].
pub(crate) fn f0_target_for_test() -> Option<ID3D11Resource> {
    lock().as_ref().and_then(|t| t.f0.tex.cast().ok())
}

/// Offline check: forgets the target.
pub(crate) fn reset_for_test() {
    *lock() = None;
}

// ------------------------------------------------------------------------ hooks --

type ForwardFn = unsafe extern "system" fn(u64, u32) -> u64;
type SetTargetsFn = unsafe extern "system" fn(u64, i32, u64, u64, u16) -> u64;

/// Detours the forward-pass body and the render-target setter (at startup, with patching).
pub fn install_hooks() -> Result<(), String> {
    if HOOKED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let base = mem::image_base().ok_or("no image base")?;
    for (rva, head, detour, slot, name) in [
        (
            FORWARD_RVA,
            &FORWARD_HEAD,
            forward_detour as *mut c_void,
            &FORWARD_ORIG,
            "forward pass",
        ),
        (
            SET_TARGETS_RVA,
            &SET_TARGETS_HEAD,
            set_targets_detour as *mut c_void,
            &SET_TARGETS_ORIG,
            "render-target setter",
        ),
    ] {
        let target = base + rva;
        let mut cur = [0u8; 16];
        if !mem::read_into(target, &mut cur) || cur != *head {
            return Err(format!(
                "{name} at {target:#x}: bytes {cur:02x?} differ from the analysed build"
            ));
        }
        // SAFETY: verified prologue; the detour has the function's signature and calls the
        // original with the same arguments.
        let tramp = unsafe { minhook::MinHook::create_hook(target as *mut c_void, detour) }
            .map_err(|e| format!("create_hook {name}: {e:?}"))?;
        slot.store(tramp as usize, Ordering::Release);
        // SAFETY: hook created above.
        unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
            .map_err(|e| format!("enable_hook {name}: {e:?}"))?;
    }
    HOOKED.store(true, Ordering::Relaxed);
    log::line("ssr: forward-pass hooks installed (material targets at slots 2 and 3)");
    Ok(())
}

unsafe extern "system" fn forward_detour(pass: u64, dynamic: u32) -> u64 {
    let engine = if wanted() {
        let _ = std::panic::catch_unwind(|| begin_forward(pass, dynamic & 0xFF != 0));
        mem::u64_at(pass + PASS_ENGINE_CTX).unwrap_or(0)
    } else {
        0
    };
    FORWARD_CTX.with(|c| c.set(engine));
    // SAFETY: MinHook trampoline of `sub_140311AA0`, arguments unchanged.
    let f =
        unsafe { std::mem::transmute::<usize, ForwardFn>(FORWARD_ORIG.load(Ordering::Acquire)) };
    // SAFETY: as above.
    let r = unsafe { f(pass, dynamic) };
    FORWARD_CTX.with(|c| c.set(0));
    r
}

fn d3d_context(engine: u64) -> Option<ID3D11DeviceContext> {
    let raw = mem::u64_at(engine).filter(|&d| d != 0)? as *mut c_void;
    // SAFETY: the engine context's live ID3D11DeviceContext.
    unsafe { ID3D11DeviceContext::from_raw_borrowed(&raw) }.cloned()
}

/// Start of a forward pass: (re)creates the target at the `hdrBuffer` size and, in
/// `FORWARD_WORLD`, clears it on the pass's context.
fn begin_forward(pass: u64, dynamic: bool) {
    let Some(color) = crate::fsr_scale::pass_image(pass, super::HASH_COLOR) else {
        return;
    };
    let Ok((_, d)) = super::tex2d(&color) else {
        return;
    };
    let Some(ctx) = mem::u64_at(pass + PASS_ENGINE_CTX).and_then(d3d_context) else {
        return;
    };
    // SAFETY: plain getter.
    let Ok(dev) = (unsafe { ctx.GetDevice() }) else {
        return;
    };
    let mut g = lock();
    if g.as_ref()
        .is_none_or(|t| t.device.as_raw() != dev.as_raw() || t.size != [d.Width, d.Height])
    {
        *g = None;
        match create_target(&dev, [d.Width, d.Height]) {
            Ok(t) => *g = Some(t),
            Err(e) => {
                log::line(&format!("ssr: {e}"));
                return;
            }
        }
    }
    if !dynamic && let Some(t) = g.as_mut() {
        // SAFETY: the pass's context, recorded by this thread.
        unsafe {
            ctx.ClearRenderTargetView(&t.material.rtv, &[0.0; 4]);
            ctx.ClearRenderTargetView(&t.f0.rtv, &[0.0; 4]);
        }
        t.cleared = frame();
    }
}

unsafe extern "system" fn set_targets_detour(
    engine: u64,
    count: i32,
    views: u64,
    depth: u64,
    flags: u16,
) -> u64 {
    // SAFETY: MinHook trampoline of `sub_1402BC370`, arguments unchanged.
    let f = unsafe {
        std::mem::transmute::<usize, SetTargetsFn>(SET_TARGETS_ORIG.load(Ordering::Acquire))
    };
    // SAFETY: as above.
    let r = unsafe { f(engine, count, views, depth, flags) };
    if engine != 0 && FORWARD_CTX.with(Cell::get) == engine {
        let _ = std::panic::catch_unwind(|| add_target(engine));
    }
    r
}

/// Re-binds the context's targets 0-1 and depth with our targets at slots 2 and 3.
pub(crate) fn add_target(engine: u64) {
    let Some(ctx) = d3d_context(engine) else {
        return;
    };
    let (rtv, f0) = match lock().as_ref() {
        Some(t) => (t.material.rtv.clone(), t.f0.rtv.clone()),
        None => return,
    };
    let mut rtvs: [Option<ID3D11RenderTargetView>; 2] = [None, None];
    let mut dsv = None;
    // SAFETY (whole block): getters/setters on the context this thread records, arrays of
    // the declared lengths.
    unsafe {
        ctx.OMGetRenderTargets(Some(&mut rtvs), Some(&mut dsv));
        if rtvs[0].is_none() {
            return;
        }
        let [a, b] = rtvs;
        ctx.OMSetRenderTargets(Some(&[a, b, Some(rtv), Some(f0)]), dsv.as_ref());
    }
    BOUND.fetch_add(1, Ordering::Relaxed);
}
