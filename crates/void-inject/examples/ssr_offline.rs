//! Offline check of the screen-space reflections (`ssr::offline_check`): stand-in engine
//! structures at the TAA site, a synthetic scene (reflective floor, red box, blue wall,
//! water patch), both execution paths (engine TAA deferred list, FSR immediate context);
//! checks hit placement, NaN, region-only writes, masks and state restoration, prints GPU
//! times and writes tonemapped PNGs to `void-rs/local/ssr/`. Usage: `ssr_offline [WxH]
//! [scale]` (defaults 3440x1440, 0.67).

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
    let out = std::path::absolute(concat!(env!("CARGO_MANIFEST_DIR"), "/../../local/ssr"))
        .unwrap_or_default();
    let report = match dinput8::ssr::offline_check(size, scale) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ssr_offline: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("ssr_offline: {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    for img in &report.images {
        let path = out.join(format!("{}.png", img.name));
        let r = std::fs::File::create(&path)
            .map_err(|e| e.to_string())
            .and_then(|f| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.width, img.height);
                enc.set_color(png::ColorType::Rgb);
                enc.set_depth(png::BitDepth::Eight);
                let mut w = enc.write_header().map_err(|e| e.to_string())?;
                w.write_image_data(&img.rgb).map_err(|e| e.to_string())
            });
        match r {
            Ok(()) => println!("wrote {}", path.display()),
            Err(e) => {
                eprintln!("ssr_offline: {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!("{}", report.text);
    if !report.failures.is_empty() {
        eprintln!("ssr_offline: FAILED:\n  {}", report.failures.join("\n  "));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
