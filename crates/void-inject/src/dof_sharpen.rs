//! Vanilla edge halo fix (docs/lighting-options.md 5b "Edge halo", E082): the depth-of-field
//! combine (`postfx_dof_combine.decl`, PS of `ARK_RENDERPASS_DOF`, last draw before TAA)
//! runs every frame, also when no depth of field is visible, and sharpens the frame against
//! the half-resolution `PostFX/DOF/diffuseMap`:
//! `colour = blur + (1 + sharpen * edge) * (colour - blur)`, `edge` = 1 where the pixel is
//! brighter than the blur (and not fog-coloured). Next to every dark silhouette the blur is
//! darker than the background, so the background's first 1-2 pixels overshoot: the bright
//! outline around characters, fences and props.
//!
//! O (IDB, exe b671f824...6180): `sub_140313E70` (called per view per frame from
//! `sub_140313DE0`, frame-graph build `sub_140351780`) fills the DOF settings at
//! `rs + 0x5FA368` from `env/dof/settings/*` (or the `r_Dof*` cvars when `r_DofDebugMode`
//! is set); `+0x74` = `sharpen_coef`, `+0x97` = `use_sharpen`. The DOF combine setup
//! `sub_140313980` stores `+0x74` as `distortionParams.z` (0x140313B69). RenderDoc
//! (`local/rdc/A_original/d2_original_frame16159.rdc`, eid 29989): `distortionParams.z =
//! 0.75`, and the formula above with 0.75 reproduces the pass output.
//!
//! Fix: a post-hook on `sub_140313E70` scales `+0x74` by `dof_sharpen` (0 = the combine
//! returns its input unchanged; the permutation and everything else stay as they are).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

use hex_literal::hex;

use crate::{log, mem, settings};

const SETTINGS_RVA: u64 = 0x31_3E70;
/// `push rbx; sub rsp, 90h; cmp cs:r_DofDebugMode, 0` (the trampoline copies the first two).
const SETTINGS_HEAD: [u8; 16] = hex!("40 53 48 81 EC 90 00 00 00 83 3D F8 01 FE 02 00");
/// `sharpen_coef` in the DOF settings struct (first argument).
const SHARPEN: u64 = 0x74;

type SettingsFn = unsafe extern "system" fn(u64, u64) -> u64;

static ORIG: AtomicUsize = AtomicUsize::new(0);
/// 0 = not tried, 1 = installed, 2 = failed.
static STATE: AtomicU8 = AtomicU8::new(0);
/// The engine's coefficient of the last frame (f32 bits), before our scaling.
static ENGINE_COEF: AtomicU32 = AtomicU32::new(u32::MAX);

/// Overlay status: (hook installed, engine coefficient of the last frame).
pub fn status() -> (bool, Option<f32>) {
    let c = ENGINE_COEF.load(Ordering::Relaxed);
    (
        STATE.load(Ordering::Relaxed) == 1,
        (c != u32::MAX).then(|| f32::from_bits(c)),
    )
}

unsafe extern "system" fn settings_detour(dof: u64, env: u64) -> u64 {
    let p = ORIG.load(Ordering::Relaxed);
    // SAFETY: trampoline of `sub_140313E70` (stored before the hook was enabled), same
    // signature and arguments.
    let r = unsafe { std::mem::transmute::<usize, SettingsFn>(p)(dof, env) };
    let _ = std::panic::catch_unwind(|| after(dof));
    r
}

fn after(dof: u64) {
    let Some([c]) = mem::f32s::<1>(dof + SHARPEN) else {
        return;
    };
    ENGINE_COEF.store(c.to_bits(), Ordering::Relaxed);
    let k = settings::get().dof_sharpen;
    if k != 1.0 && c.is_finite() {
        let _ = mem::write_bytes(dof + SHARPEN, &(c * k).to_le_bytes());
    }
}

fn install() -> Result<u64, String> {
    let base = mem::image_base().ok_or("image base")?;
    let target = base + SETTINGS_RVA;
    let mut cur = [0u8; 16];
    if !mem::read_into(target, &mut cur) || cur != SETTINGS_HEAD {
        return Err(format!(
            "unexpected bytes at {target:#x}: {cur:02x?} (expected {SETTINGS_HEAD:02x?})"
        ));
    }
    // SAFETY: `target` is `sub_140313E70` of the verified executable (prologue checked);
    // the detour has its signature and calls the trampoline with the same arguments.
    let tramp = unsafe {
        minhook::MinHook::create_hook(target as *mut c_void, settings_detour as *mut c_void)
    }
    .map_err(|e| format!("create_hook: {e:?}"))?;
    ORIG.store(tramp as usize, Ordering::Release);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook: {e:?}"))?;
    Ok(target)
}

/// Main thread, every frame: installs the hook the first time `dof_sharpen` is not 1 (the
/// engine value needs no hook; once installed it stays and applies the live value).
pub fn tick() {
    if settings::get().dof_sharpen == 1.0 || STATE.load(Ordering::Relaxed) != 0 {
        return;
    }
    match install() {
        Ok(at) => {
            STATE.store(1, Ordering::Relaxed);
            log::line(&format!(
                "dof_sharpen: DOF settings hooked at {at:#x}; the engine's sharpening is scaled by dof_sharpen"
            ));
        }
        Err(e) => {
            STATE.store(2, Ordering::Relaxed);
            log::line(&format!(
                "dof_sharpen: hook not installed ({e}); engine sharpening kept"
            ));
            crate::notify::warn("DOF sharpening fix unavailable (see log)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hooked prologue matches the game executable next to the workspace (read only;
    /// skipped when the file is absent).
    #[test]
    fn prologue_matches_the_executable() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../Dishonored2.exe");
        let Ok(exe) = std::fs::read(&path) else {
            return;
        };
        let u16_at = |o: usize| u16::from_le_bytes([exe[o], exe[o + 1]]) as usize;
        let u32_at = |o: usize| u32::from_le_bytes(exe[o..o + 4].try_into().unwrap()) as usize;
        let pe = u32_at(0x3C);
        let sections = u16_at(pe + 6);
        let table = pe + 24 + u16_at(pe + 20);
        let rva = SETTINGS_RVA as usize;
        let file = (0..sections)
            .map(|i| table + 40 * i)
            .find_map(|s| {
                let (va, size, raw) = (u32_at(s + 12), u32_at(s + 8), u32_at(s + 20));
                (va..va + size).contains(&rva).then(|| rva - va + raw)
            })
            .expect("RVA in a section");
        assert_eq!(exe[file..file + 16], SETTINGS_HEAD);
    }
}
