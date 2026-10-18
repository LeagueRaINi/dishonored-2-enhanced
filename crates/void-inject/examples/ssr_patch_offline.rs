//! Offline check of the material patch of the game's lit forward pixel shaders
//! (`ssr::patch_offline_check`): an end-to-end GPU test of a shader with the engine's IBL code,
//! then every `ps_*.dxbc` dump of a directory (default: the read-only d2enhanced capture
//! dumps next to the workspace) patched, created on the device and compared with the
//! original program. Usage: `ssr_patch_offline [shader_dir]`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let dir = std::env::args().nth(1).map_or_else(
        || {
            std::path::absolute(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../d2enhanced/captures/shaders"
            ))
            .unwrap_or_default()
        },
        std::path::PathBuf::from,
    );
    match dinput8::ssr::patch_offline_check(&dir) {
        Ok(r) => {
            println!("{}", r.text);
            if r.failures.is_empty() {
                ExitCode::SUCCESS
            } else {
                let n = r.failures.len();
                for f in r.failures.iter().take(20) {
                    eprintln!("  {f}");
                }
                eprintln!("ssr_patch_offline: {n} FAILED");
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("ssr_patch_offline: {e}");
            ExitCode::FAILURE
        }
    }
}
