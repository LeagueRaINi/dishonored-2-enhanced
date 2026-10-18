//! Offline check of the in-game overlay (`overlay::offline_preview`): hidden window,
//! D3D11 device and an `R8G8B8A8_UNORM_SRGB` swap chain like the game's, the real panels
//! with stand-in data and synthetic pointer input (tab clicks, hover and click on a
//! checkbox), drawn through the overlay's painter (state swap, UNORM view), read back and
//! written as PNGs to `void-rs/local/overlay/`. Fails if the game-side pipeline state is
//! not restored, a click is missed, or the UNORM view converts colours.
//! Usage: `overlay_preview [WxH]` (default 3440x1440).

use std::process::ExitCode;

fn main() -> ExitCode {
    let size = std::env::args()
        .nth(1)
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some([w.parse().ok()?, h.parse().ok()?])
        })
        .unwrap_or([3440, 1440]);
    let out = std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../local/overlay"))
        .unwrap_or_default();
    let preview = match dinput8::overlay::offline_preview(size) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("overlay_preview: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("overlay_preview: {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    for img in &preview.images {
        let path = out.join(format!("{}.png", img.name));
        let r = std::fs::File::create(&path)
            .map_err(|e| e.to_string())
            .and_then(|f| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.width, img.height);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
                let mut w = enc.write_header().map_err(|e| e.to_string())?;
                w.write_image_data(&img.rgba).map_err(|e| e.to_string())
            });
        match r {
            Ok(()) => println!("wrote {}", path.display()),
            Err(e) => {
                eprintln!("overlay_preview: {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!("{}", preview.report);
    ExitCode::SUCCESS
}
