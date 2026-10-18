//! Offline check of the AO replacement (`ao::offline_check`): the game's HBAO+ DLL with a
//! context created like the engine's, our RenderAO hook installed on it, both modes called
//! through the context's vtable on a deferred context with a synthetic scene; checks the
//! result, state restoration and region, prints GPU times and writes greyscale PNGs of the
//! AO to `void-rs/local/ao/`. Usage: `ao_offline [WxH] [scale]` (defaults 3440x1440, 0.67);
//! the DLL is loaded from the game directory (the workspace's parent).

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
    let scale = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.67);
    let dir =
        std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../..")).unwrap_or_default();
    let out = std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../local/ao"))
        .unwrap_or_default();
    let report = match dinput8::ao::offline_check(&dir, size, scale) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ao_offline: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("ao_offline: {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    for img in &report.images {
        let path = out.join(format!("{}.png", img.name));
        let r = std::fs::File::create(&path)
            .map_err(|e| e.to_string())
            .and_then(|f| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.width, img.height);
                enc.set_color(png::ColorType::Grayscale);
                enc.set_depth(png::BitDepth::Eight);
                let mut w = enc.write_header().map_err(|e| e.to_string())?;
                w.write_image_data(&img.gray).map_err(|e| e.to_string())
            });
        match r {
            Ok(()) => println!("wrote {}", path.display()),
            Err(e) => {
                eprintln!("ao_offline: {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!("{}", report.text);
    if !report.failures.is_empty() {
        eprintln!("ao_offline: FAILED:\n  {}", report.failures.join("\n  "));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
