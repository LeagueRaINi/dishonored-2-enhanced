//! Offline check of the hybrid ray-traced reflections (`ssr::rt_reflect::offline`): synthetic
//! BVH, CPU-made frame inputs, the in-game recording path on a D3D12 list. Usage:
//! `rt_reflect_offline [--warp] [WxH]` (default 1280x540 on the GPU, 256x144 on WARP).
//! Images to `void-rs/local/rt_reflect/`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let warp = args.iter().any(|a| a == "--warp");
    let size = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .find_map(|s| {
            let (w, h) = s.split_once('x')?;
            Some([w.parse().ok()?, h.parse().ok()?])
        })
        .unwrap_or(if warp { [256, 144] } else { [1280, 540] });
    let out = std::path::absolute(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../local/rt_reflect"
    ))
    .unwrap_or_default();
    let report = match dinput8::ssr::rt_reflect::offline::offline_check(warp, size) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rt_reflect_offline: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("rt_reflect_offline: {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    let tag = if warp { "warp" } else { "gpu" };
    for img in &report.images {
        let path = out.join(format!("{}_{tag}.png", img.name));
        let r = std::fs::File::create(&path)
            .map_err(|e| e.to_string())
            .and_then(|f| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.width, img.height);
                enc.set_color(png::ColorType::Rgb);
                enc.set_depth(png::BitDepth::Eight);
                let mut w = enc.write_header().map_err(|e| e.to_string())?;
                w.write_image_data(&img.rgb).map_err(|e| e.to_string())
            });
        if let Err(e) = r {
            eprintln!("rt_reflect_offline: {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    }
    println!("{}", report.text);
    if !report.failures.is_empty() {
        eprintln!(
            "rt_reflect_offline: FAILED:\n  {}",
            report.failures.join("\n  ")
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
