//! In-process module for the original game.
//!
//! Built as `dinput8.dll` and placed in the game folder: the executable imports
//! `DINPUT8.dll!DirectInput8Create` statically, so this module loads at process
//! start. It forwards that export to the system DLL and, only if the executable
//! matches the analysed build (SHA-256), replaces or augments individual passes of the
//! game's D3D11 renderer: FSR 4 in place of the engine TAA (`fsr`, `fsr_scale`, `ffx`), our
//! own ambient occlusion in place of NVIDIA HBAO+ (`ao`), sun contact shadows in the
//! screen-shadow pass (`contact`), an optional profiler and benchmark
//! (`profiler`, `bench`, `passes`, `trace`), cvar overrides (`cvars`), texture mods
//! (`texmods`), code patches such as the boot-video skip (`patches`), an in-game
//! overlay (`overlay`), and the ray-tracing scene service (`rt`) with the path-traced photo
//! mode on top of it (`photo`).
//!
//! `unsafe` is confined to the FFI, hook, and memory-access code (export forwarding,
//! DllMain, detours, checked reads and patches of game memory, D3D11/FFX calls).

pub mod ao;
mod bench;
pub mod contact;
mod cvars;
pub mod dof_sharpen;
mod dxbc;
mod ffi;
pub mod ffx;
pub mod fsr;
mod fsr_check;
mod fsr_motion;
pub mod fsr_scale;
mod hlsl;
mod hook;
mod log;
mod mem;
mod mouse;
pub mod notify;
pub mod overlay;
mod passes;
mod patches;
pub mod photo;
mod profiler;
pub mod rt;
mod scene;
mod settings;
pub mod ssr;
mod texmods;
mod trace;

use std::path::PathBuf;
use std::sync::OnceLock;

/// SHA-256 of the analysed executable (E001); addresses in `hook`/`scene` are for it.
pub const EXPECTED_EXE_SHA256: &str =
    "b671f8249422862bf557d00aab2f4bb671ed4515cc10ad8a8a9decad84516180";

static GAME_DIR: OnceLock<PathBuf> = OnceLock::new();
static SETTINGS_NOTES: OnceLock<Vec<String>> = OnceLock::new();
/// Outcome of the executable check, for the overlay's status panel.
static EXE_STATUS: OnceLock<String> = OnceLock::new();

/// DllMain, before any game code runs: loads the settings
/// (`void-rs/local/inject/void-inject.ini`) and applies the patches that must precede the
/// game's initialisation (`patches.rs`; states are logged by `initialize`).
fn boot() {
    if let Some(dir) = ffi::exe_path().as_deref().and_then(std::path::Path::parent) {
        let notes = settings::load(&dir.join("void-rs").join("local").join("inject"));
        let _ = SETTINGS_NOTES.set(notes);
    }
    patches::apply_boot(&settings::get());
}

/// Executable check result ("analysed build (...)", "unknown build ...; hooks disabled").
pub(crate) fn exe_status() -> &'static str {
    EXE_STATUS.get().map_or("not checked", String::as_str)
}

/// Directory of the running executable (set during initialization).
pub fn game_dir() -> Option<&'static PathBuf> {
    GAME_DIR.get()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Runs on a separate thread after DLL attach (never under the loader lock).
fn initialize(exe: PathBuf) {
    let dir = exe.parent().map(PathBuf::from).unwrap_or_default();
    let _ = GAME_DIR.set(dir.clone());
    log::init(&dir.join("void-rs").join("local").join("inject"));
    log::line(&format!(
        "void-inject {} loaded into {}",
        env!("CARGO_PKG_VERSION"),
        exe.display()
    ));
    let hash = match std::fs::read(&exe) {
        Ok(b) => sha256_hex(&b),
        Err(e) => {
            log::line(&format!("cannot read executable: {e}; hooks disabled"));
            return;
        }
    };
    if hash != EXPECTED_EXE_SHA256 {
        log::line(&format!(
            "executable sha256 {hash} is not the analysed build; hooks disabled"
        ));
        let _ = EXE_STATUS.set(format!("unknown build {hash}; hooks disabled"));
        return;
    }
    let _ = EXE_STATUS.set(format!("analysed build ({}...)", &hash[..16]));
    for n in SETTINGS_NOTES.get().into_iter().flatten() {
        log::line(n);
    }
    patches::apply_init(&settings::get());
    patches::report();
    // Texture mods are hooked in DllMain (texmods.rs); log what happened so far.
    texmods::report();
    // Shader-creation hooks first (before the game creates its device and shaders); they
    // are D3D11-level and independent of the executable build. The reflections' material
    // output is patched into the lit forward pixel shaders there (only possible at startup).
    let s = settings::get();
    let material = s.ssr && s.ssr_material;
    if material {
        ssr::material::enable_patching();
    }
    if (s.profile || material)
        && let Err(e) = trace::install_shader_hooks()
    {
        log::line(&format!("trace: shader hook installation failed: {e}"));
    }
    log::line("executable matches the analysed build; installing hooks");
    cvars::set_overrides(&settings::get().cvars);
    match hook::install() {
        Ok(()) => {
            log::line("hooks installed");
            let s = settings::get();
            if s.fsr
                && let Err(e) = fsr::install()
            {
                log::line(&format!("fsr installation failed: {e}; engine TAA used"));
            }
            if s.ao_void
                && let Err(e) = ao::install()
            {
                log::line(&format!("ao: installation failed: {e}; HBAO+ stays"));
            }
            if s.contact_shadows
                && let Err(e) = contact::install()
            {
                log::line(&format!("contact: installation failed: {e}"));
            }
            if s.ssr
                && let Err(e) = ssr::install()
            {
                log::line(&format!("ssr: installation failed: {e}"));
            }
            if ssr::material::patching()
                && let Err(e) = ssr::material::install_hooks()
            {
                log::line(&format!("ssr: material target unavailable: {e}"));
            }
            if s.profile {
                match profiler::install() {
                    Ok(()) => log::line("profiler installed"),
                    Err(e) => log::line(&format!("profiler installation failed: {e}")),
                }
                if let Err(e) = trace::install_context_hooks() {
                    log::line(&format!("trace: context hook installation failed: {e}"));
                }
            }
            // DirectInput hooks: synthetic mouse look for the benchmark pan, input
            // blocking while the overlay is open.
            if (s.profile || s.overlay)
                && let Err(e) = mouse::install()
            {
                log::line(&format!("mouse hook installation failed: {e}"));
            }
            // Engine present: profiler `present` phase and the overlay.
            if s.profile || s.overlay {
                match hook::install_present() {
                    Ok(()) if s.overlay => match overlay::install() {
                        Ok(()) => log::line("overlay installed"),
                        Err(e) => log::line(&format!("overlay installation failed: {e}")),
                    },
                    Ok(()) => {}
                    Err(e) => log::line(&format!("present hook installation failed: {e}")),
                }
            }
        }
        Err(e) => log::line(&format!("hook installation failed: {e}; running unhooked")),
    }
}
