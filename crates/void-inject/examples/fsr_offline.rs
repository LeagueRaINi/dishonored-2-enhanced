//! Offline check of the FSR TAA-replacement work (`fsr::offline_check`): stand-in engine
//! textures on a private D3D11 device, pack -> shared fence -> FSR on D3D12 -> fence ->
//! unpack, native (inline and deferred-list entry styles) then upscale (masks,
//! distoVectors stretch, depth resample). Usage: `fsr_offline [WxH] [frames] [scale]`
//! (defaults 3440x1440, 32, 0.67); the FidelityFX DLLs are loaded from the game
//! directory (the workspace's parent).

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let size = args
        .next()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some([w.parse().ok()?, h.parse().ok()?])
        })
        .unwrap_or([3440, 1440]);
    let frames = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let scale = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.67);
    let dir =
        std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../..")).unwrap_or_default();
    match dinput8::fsr::offline_check(&dir, size, frames, scale) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("fsr_offline: {e}");
            ExitCode::FAILURE
        }
    }
}
