//! Offline check of the sun contact shadows (`contact::offline_check`): stand-in engine
//! structures for one SCREENSHADOWS pass on a private D3D11 device, the hook's
//! after-callback entry point, a synthetic floor/wall/box scene at a grazing sun angle;
//! checks shadow placement, acne, min() with the CSM channel, region-only writes and state
//! restoration, prints GPU times and writes greyscale PNGs of the screen-shadow green
//! channel to `void-rs/local/contact/`. Usage: `contact_offline [WxH] [scale]` (defaults
//! 3440x1440, 0.67).

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
    let out = std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../local/contact"))
        .unwrap_or_default();
    let report = match dinput8::contact::offline_check(size, scale) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("contact_offline: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("contact_offline: {}: {e}", out.display());
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
                eprintln!("contact_offline: {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!("{}", report.text);
    if !report.failures.is_empty() {
        eprintln!(
            "contact_offline: FAILED:\n  {}",
            report.failures.join("\n  ")
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
