//! Offline check of the photo composite (`photo::offline_check`): a stand-in game D3D11
//! device with engine-like `hdrBuffer` / depth, our image written by D3D12 into a shared
//! slot, the FOG-pass recording on a deferred context, the overlay preview; checks the
//! composited pixels, untouched pixels/alpha and state restoration. Never touches the game.
//!
//! Usage: `photo_offline [--warp] [WxH]` (default hardware, 640x360). Run with `--warp`
//! first; every run appends one line to `void-rs/local/rt/runs.log`.

use std::io::Write as _;
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
        .unwrap_or([640, 360]);
    let t = std::time::Instant::now();
    let r = dinput8::photo::offline_check(warp, size);
    let line = format!(
        "{}\tphoto_offline\t{}\t{}x{}\t{}\t{:.1} s\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        if warp { "warp" } else { "gpu" },
        size[0],
        size[1],
        match &r {
            Ok(rep) => format!("ok: {}", rep.lines().next().unwrap_or_default()),
            Err(e) => format!("FAILED: {e}"),
        },
        t.elapsed().as_secs_f64()
    );
    let log = std::path::absolute(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../local/rt/runs.log"
    ))
    .unwrap_or_default();
    if let Some(dir) = log.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
    {
        Ok(mut f) => {
            let _ = f.write_all(line.as_bytes());
        }
        Err(e) => eprintln!("photo_offline: {}: {e}", log.display()),
    }
    match r {
        Ok(rep) => {
            println!("{rep}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("photo_offline: FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}
