//! Detour installation and entry points (MinHook): `RenderAllViews` and the engine
//! present (profiler `present` phase, overlay).
//!
//! Invariants: every detour calls the original with unchanged arguments and returns its
//! result; our code runs inside `catch_unwind` and reads game memory only through `mem`.
//! `RenderAllViews` observation runs after the original (rate-limited logging); the
//! overlay draws before the original present, so its image is in the presented frame.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use hex_literal::hex;
use minhook::MinHook;

use crate::{ffi, log, mem, scene};

/// Shift+F3 (VK_F3): one-frame D3D11 trace; Ctrl+F3: benchmark run (both settings
/// `profile = 1`). (F5/F9 are the game's quicksave/load; d2enhanced uses F2, F6, F7, F8,
/// F10, F11.)
const PROFILE_KEY: i32 = 0x72;
static PROFILE_HELD: AtomicBool = AtomicBool::new(false);
/// F6 (VK_F6): FSR on/off; Shift+F6: FSR debug view (settings `fsr = 1`).
const FSR_KEY: i32 = 0x75;
static FSR_HELD: AtomicBool = AtomicBool::new(false);
const SHIFT_KEY: i32 = 0x10;
const CTRL_KEY: i32 = 0x11;

/// `RenderAllViews` (VA for the analysed build; image base 0x140000000).
const RENDER_ALL_VIEWS_VA: u64 = 0x1_4035_DC60;
const LINK_BASE: u64 = 0x1_4000_0000;

type RenderAllViews = unsafe extern "system" fn(a1: *mut c_void, a2: *mut c_void) -> i64;

static ORIGINAL: OnceLock<usize> = OnceLock::new();
/// `RenderAllViews` requests pointer of the last frame (benchmark start from the overlay).
static LAST_REQUESTS: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);
static LAST_LOG_MS: AtomicU64 = AtomicU64::new(0);
static START: OnceLock<Instant> = OnceLock::new();

pub fn install() -> Result<(), String> {
    let base = mem::image_base().ok_or("no image base")?;
    let target = (RENDER_ALL_VIEWS_VA - LINK_BASE + base) as *mut c_void;
    // Sanity check: the expected prologue bytes of the analysed build.
    let mut head = [0u8; 4];
    if !mem::read_into(target as u64, &mut head) {
        return Err("target not readable".into());
    }
    log::line(&format!("RenderAllViews at {target:p}, head {head:02x?}"));
    let _ = START.set(Instant::now());
    // SAFETY: `target` is the start of `RenderAllViews` in the verified executable and
    // `detour` has the same calling convention and signature.
    let original = unsafe { MinHook::create_hook(target, detour as *mut c_void) }
        .map_err(|e| format!("create_hook: {e:?}"))?;
    let _ = ORIGINAL.set(original as usize);
    // SAFETY: the hook was created above.
    unsafe { MinHook::enable_hook(target) }.map_err(|e| format!("enable_hook: {e:?}"))?;
    Ok(())
}

unsafe extern "system" fn detour(a1: *mut c_void, a2: *mut c_void) -> i64 {
    let Some(&orig) = ORIGINAL.get() else {
        return 0;
    };
    // SAFETY: `orig` is the trampoline returned by MinHook for this signature.
    let f = unsafe { std::mem::transmute::<usize, RenderAllViews>(orig) };
    let t = Instant::now();
    // SAFETY: arguments are passed through unchanged.
    let result = unsafe { f(a1, a2) };
    crate::profiler::record_render_all_views(t.elapsed().as_nanos() as u64);
    let requests = a1 as u64;
    LAST_REQUESTS.store(requests, Ordering::Relaxed);
    let _ = std::panic::catch_unwind(move || observe(requests));
    result
}

fn observe(requests: u64) {
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed);
    crate::cvars::enforce_overrides(frame);
    if crate::settings::get().profile {
        crate::bench::observe_view(requests);
    }
    // Every frame: `fsr` is a live setting (turning it on installs FSR at run time).
    crate::fsr::tick();
    // `ao_mode` is live (turning `void` on installs the HBAO+ hook at run time).
    crate::ao::tick();
    // `contact_shadows` is live (turning it on installs the SCREENSHADOWS hook).
    crate::contact::tick();
    // `dof_sharpen` is live (a value other than 1 installs the DOF settings hook).
    crate::dof_sharpen::tick();
    // `ssr` is live (turning it on installs the TAA call-site redirect).
    crate::ssr::tick();
    // `rt` / `photo` are live (scene service and photo mode; inert while both are off).
    crate::rt::scene::tick(requests, frame);
    crate::photo::tick(requests, frame);
    if crate::fsr::installed() && crate::settings::get().fsr {
        let fsr_down = ffi::key_down(FSR_KEY);
        if fsr_down && !FSR_HELD.swap(true, Ordering::Relaxed) {
            if ffi::key_down(SHIFT_KEY) {
                crate::fsr::toggle_debug_view();
            } else {
                crate::fsr::toggle();
            }
        } else if !fsr_down {
            FSR_HELD.store(false, Ordering::Relaxed);
        }
    }
    let down = ffi::key_down(PROFILE_KEY);
    let pressed = down && !PROFILE_HELD.swap(true, Ordering::Relaxed);
    if pressed && crate::settings::get().profile {
        if ffi::key_down(SHIFT_KEY) {
            let armed = crate::trace::arm();
            log::line(&format!(
                "trace: {} ({} shaders known)",
                if armed {
                    "armed for the next frame"
                } else {
                    "busy"
                },
                crate::trace::SHADER_COUNT.load(Ordering::Relaxed)
            ));
        } else if ffi::key_down(CTRL_KEY) {
            crate::bench::toggle(requests);
        }
    } else if !down {
        PROFILE_HELD.store(false, Ordering::Relaxed);
    }
    // Frame/view summary once a second: a diagnostic (`diagnostics = 1`).
    if !crate::settings::get().diagnostics {
        return;
    }
    let now = START.get().map_or(0, |s| s.elapsed().as_millis() as u64);
    let last = LAST_LOG_MS.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST_LOG_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let views = scene::summarize(requests);
    let map = scene::read_pose(requests).map(|p| p.0).unwrap_or_default();
    let mut msg = format!(
        "frame {frame}: {} view request(s), map '{map}'",
        views.len()
    );
    for (i, v) in views.iter().enumerate() {
        msg += &format!(
            "\n  view {i}: world {:#x} models {} lights {} vieworg ({:.2} {:.2} {:.2}) fov ({:.3} {:.3}) axis0 ({:.3} {:.3} {:.3})",
            v.world,
            v.models,
            v.lights,
            v.vieworg[0],
            v.vieworg[1],
            v.vieworg[2],
            v.fov[0],
            v.fov[1],
            v.viewaxis[0],
            v.viewaxis[1],
            v.viewaxis[2]
        );
    }
    log::line(&msg);
}

/// The `RenderAllViews` requests pointer of the last frame (0 before the first frame).
pub fn last_requests() -> u64 {
    LAST_REQUESTS.load(Ordering::Relaxed)
}

// ------------------------------------------------------------------ engine present --

/// `sub_14036B3A0(rs, a2, a3)`, the engine's present (only caller `sub_1403591B0`, from
/// `RenderFrame`, main thread). O (IDB): `obj = RS_CurrentFrameData(rs)` (vtable +32,
/// `0x14035EC00`: `rs+54728 + 976 * rs+54744` if `0 <= rs+54744 < (rs+54736 & 0xFFFFFF)`,
/// else 0); with `obj != 0 && a3 == 0` it calls `IDXGISwapChain::Present` (vtable +64) on
/// `*(obj + 56)`, otherwise only `Flush` (+888) on the immediate context.
const PRESENT_RVA: u64 = 0x36_B3A0;
const PRESENT_HEAD: [u8; 16] = hex!("48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57");
const RS_FRAME_DATA: u64 = 54728;
const RS_FRAME_COUNT: u64 = 54736;
const RS_FRAME_INDEX: u64 = 54744;
const FRAME_DATA_SIZE: u64 = 976;
const FRAME_SWAP_CHAIN: u64 = 56;

type Present = unsafe extern "system" fn(rs: u64, a2: u64, a3: u64) -> u64;
static PRESENT_ORIG: AtomicUsize = AtomicUsize::new(0);

/// The swap chain the engine is about to present to (replicates `RS_CurrentFrameData`
/// with checked reads); 0 when it will not present.
fn present_swap_chain(rs: u64, a3: u64) -> u64 {
    if a3 & 0xFF != 0 {
        return 0;
    }
    let (Some(index), Some(count), Some(data)) = (
        mem::u32_at(rs + RS_FRAME_INDEX),
        mem::u32_at(rs + RS_FRAME_COUNT),
        mem::u64_at(rs + RS_FRAME_DATA),
    ) else {
        return 0;
    };
    let index = index as i32;
    if index < 0 || index >= (count & 0xFF_FFFF) as i32 || data == 0 {
        return 0;
    }
    mem::u64_at(data + FRAME_DATA_SIZE * index as u64 + FRAME_SWAP_CHAIN).unwrap_or(0)
}

unsafe extern "system" fn present_detour(rs: u64, a2: u64, a3: u64) -> u64 {
    if crate::settings::get().overlay {
        let swap_chain = present_swap_chain(rs, a3);
        crate::overlay::before_present(rs, swap_chain);
    }
    // SAFETY: MinHook trampoline of the engine present with this signature.
    let f = unsafe { std::mem::transmute::<usize, Present>(PRESENT_ORIG.load(Ordering::Relaxed)) };
    let t = Instant::now();
    // SAFETY: arguments passed through unchanged.
    let r = unsafe { f(rs, a2, a3) };
    crate::profiler::record_present(t.elapsed().as_nanos() as u64);
    r
}

/// Hooks the engine present (settings `overlay = 1` or `profile = 1`).
pub fn install_present() -> Result<(), String> {
    let base = mem::image_base().ok_or("no image base")?;
    let target = base + PRESENT_RVA;
    let mut head = [0u8; 16];
    if !mem::read_into(target, &mut head) || head != PRESENT_HEAD {
        return Err(format!(
            "present at {target:#x}: bytes {head:02x?} differ from the analysed build"
        ));
    }
    // SAFETY: verified prologue of `sub_14036B3A0`; the detour has its signature (three
    // integer arguments, integer result) and forwards them unchanged.
    let tramp =
        unsafe { MinHook::create_hook(target as *mut c_void, present_detour as *mut c_void) }
            .map_err(|e| format!("create_hook present: {e:?}"))?;
    PRESENT_ORIG.store(tramp as usize, Ordering::Relaxed);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook present: {e:?}"))?;
    log::line(&format!("engine present hooked at {target:#x}"));
    Ok(())
}
