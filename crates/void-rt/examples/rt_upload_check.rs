//! Video memory and submit lengths of the in-game scene upload (`void_rt::ingame::upload`,
//! docs/raytracing.md 11.3): decodes a map at a texture cap as the game's DLL does, prints
//! the memory estimate per texture cap (device allocation sizes and the sum of the mip
//! bytes), then (unless `--estimate-only`) uploads it with the budget guard and the paced
//! submits and reports every phase (submits, GPU ms, longest submit, DXGI usage) and a
//! cut-off TLAS build. Offline only; reads the archives and an F3 snapshot, never touches the
//! game. Logged in `local/rt/runs.log`.
//!
//! cargo run --release -p void-rt --example rt_upload_check -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 --tex-size 2048 [--warp] [--estimate-only]
//!
//! `--available-mib N` simulates a host with only N MiB free (the headroom is raised so
//! that budget - usage - headroom = N): exercises the texture-cap reduction and refusal.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use void_rt::gpu::mib;
use void_rt::ingame::{Gpu, SceneOptions, SceneSource, estimate_memory, load_scene, upload};
use void_rt::lights::Env;
use void_rt::run::{Report, run_log};
use void_rt::snapshot::{Snapshot, Sun};

#[derive(Parser)]
#[command(about = "Video memory and submit lengths of the in-game scene upload")]
struct Args {
    #[arg(long)]
    game: PathBuf,
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, default_value_t = 2048)]
    tex_size: u32,
    /// Without the FC6 primitives (they are built with `rt = 1`).
    #[arg(long)]
    no_spike: bool,
    /// Only decode and estimate (device size queries, no submits).
    #[arg(long)]
    estimate_only: bool,
    /// Simulated free video memory (MiB) for the budget guard.
    #[arg(long)]
    available_mib: Option<u64>,
    /// Sleep after every blocking submit (ms), as in game.
    #[arg(long, default_value_t = 1)]
    pause_ms: u64,
    /// Only the N instances nearest the snapshot camera in the full TLAS (WARP runs).
    #[arg(long)]
    tlas_limit: Option<usize>,
    #[arg(long)]
    warp: bool,
    #[arg(long, default_value = "local/rt/upload_check")]
    out: PathBuf,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

fn main() {
    let args = Args::parse();
    let mode = if args.warp {
        "upload-check warp"
    } else {
        "upload-check gpu"
    };
    let cmd: Vec<String> = std::env::args().skip(1).collect();
    run_log(
        &args.runs_log,
        &format!(
            "start {mode} run (out {}): {}",
            args.out.display(),
            cmd.join(" ")
        ),
    );
    let r = run(&args);
    run_log(
        &args.runs_log,
        &match &r {
            Ok(()) => format!("end {mode} run: ok"),
            Err(e) => format!("end {mode} run: error: {}", e.lines().next().unwrap_or("")),
        },
    );
    if let Err(e) = r {
        eprintln!("rt_upload_check: {e}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let mut rep = Report::default();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let mut gpu = Gpu::new(None, args.warp)?;
    rep.line(format!(
        "device: {} ({} MiB dedicated, driver {}); {}",
        gpu.info.name,
        gpu.info.vram_mib,
        gpu.info.driver,
        gpu.vram_text()
    ));
    let snap = Snapshot::load(&args.snapshot)?;
    let camera = snap.camera;
    let src =
        SceneSource {
            game_dir: args.game.clone(),
            map: args.map.clone(),
            entities: snap.entities.clone(),
            lights: snap.lights.clone(),
            sun: Some(snap.sun.unwrap_or_else(|| {
                Sun::from_angles(26.0, 100.0, [0.215_861, 0.171_441, 0.116_971])
            })),
            env: Some(snap.env.unwrap_or(Env {
                sky_zenith: [0.0; 3],
                sky_horizon: [37.7451, 52.411_766, 55.0],
                ambient: [1.0; 3],
            })),
        };
    // The in-game options (`rt::scene::options`): default batches, FC6 primitives with rt = 1.
    let opts = SceneOptions {
        texture_size: args.tex_size,
        spike_primitives: !args.no_spike,
        tlas_limit: args.tlas_limit.map(|n| (n, camera.org)),
        ..SceneOptions::default()
    };
    let t = Instant::now();
    let cpu = load_scene(&src, &opts)?;
    rep.line(format!(
        "load_scene at textures <= {}: {:.1} s, {} meshes, {} instances, {} triangles (records), {} textures",
        args.tex_size,
        t.elapsed().as_secs_f64(),
        cpu.scene.meshes.len(),
        cpu.scene.instances.len(),
        cpu.scene.records.len(),
        cpu.scene.textures.len()
    ));
    let est = estimate_memory(&gpu, &cpu);
    rep.line(format!(
        "estimate: geometry {:.1} MiB, BLAS uncompacted {:.1} MiB + scratch {:.1} MiB (peak {:.1} MiB), full TLAS {:.1} MiB, tables {:.1} MiB, primitives {:.1} MiB, staging {:.1} MiB",
        mib(est.geometry),
        mib(est.blas_uncompacted),
        mib(est.blas_scratch),
        mib(est.blas_peak()),
        mib(est.tlas),
        mib(est.tables),
        mib(est.primitives),
        mib(est.staging)
    ));
    for &(cap, dev, data) in &est.textures {
        rep.line(format!(
            "  textures <= {cap}: device {:.1} MiB (sum of mip bytes {:.1} MiB); resident {:.1} MiB, peak {:.1} MiB",
            mib(dev),
            mib(data),
            mib(est.resident(cap)),
            mib(est.peak(cap))
        ));
    }
    if args.estimate_only {
        rep.line(format!("total {:.1} s", t0.elapsed().as_secs_f64()));
        return std::fs::write(args.out.join("report.txt"), &rep.0).map_err(|e| e.to_string());
    }
    if let (Some(avail), Some(m)) = (args.available_mib, gpu.video_memory()) {
        gpu.vram_policy.headroom_bytes =
            m.budget.saturating_sub(m.usage).saturating_sub(avail << 20);
        gpu.vram_policy.headroom_fraction = 0.0;
        rep.line(format!(
            "simulated: {avail} MiB available ({})",
            gpu.vram_text()
        ));
    }
    gpu.set_pause(Duration::from_millis(args.pause_ms));
    gpu.set_log(|l| println!("{l}"));
    let t = Instant::now();
    let scene = upload(&mut gpu, cpu)?;
    let r = scene.report();
    rep.line(format!(
        "upload: {:.1} s wall, textures <= {} (requested {}), longest submit {:.2} ms GPU",
        t.elapsed().as_secs_f64(),
        r.texture_size,
        r.requested_texture_size,
        r.max_submit_ms
    ));
    for p in &r.phases {
        rep.line(format!(
            "  {}: {:.0} ms wall, {} submits, GPU {:.1} ms, longest {:.2} ms; usage {:.0} -> {:.0} MiB (+{:.0}, estimate {:.0}) of budget {:.0} MiB",
            p.name,
            p.wall_ms,
            p.submits,
            p.gpu_ms,
            p.max_submit_ms,
            mib(p.usage_before),
            mib(p.usage_after),
            mib(p.usage_after.saturating_sub(p.usage_before)),
            mib(p.estimate),
            mib(p.budget)
        ));
    }
    let b = scene.blas_stats();
    rep.line(format!(
        "BLAS: {} meshes, {} triangles, {} build batches ({:.1} ms GPU), {} compaction submits ({:.1} ms GPU), longest {:.2} ms; uncompacted {:.1} MiB -> compacted {:.1} MiB, scratch arena {:.1} MiB",
        b.count,
        b.triangles,
        b.batches,
        b.build_gpu_ms,
        b.compact_submits,
        b.compact_gpu_ms,
        b.max_submit_ms,
        mib(b.uncompacted_bytes),
        mib(b.compacted_bytes),
        mib(b.scratch_arena_bytes)
    ));
    let tx = scene.textures();
    rep.line(format!(
        "textures: {} slots, {:.1} MiB device, {} submits, {} rejected, {} chains completed",
        tx.count,
        mib(tx.bytes),
        tx.upload_submits,
        tx.rejected.len(),
        tx.completed_chains
    ));
    gpu.take_submit_stats();
    let mut cut = scene.build_cutoff_tlas(&mut gpu, camera.org, 40.0)?;
    let moved = [camera.org[0] + 30.0, camera.org[1], camera.org[2]];
    let n = scene.update_cutoff_tlas(&mut gpu, &mut cut, moved, 40.0)?;
    let st = gpu.take_submit_stats();
    rep.line(format!(
        "cut-off TLAS (40 m, capacity {}): build + rebuild ({n} instances) {} submits, longest {:.2} ms GPU",
        cut.capacity(),
        st.count,
        st.max_ms
    ));
    rep.line(format!("after: {}", gpu.vram_text()));
    rep.line(format!("total {:.1} s", t0.elapsed().as_secs_f64()));
    std::fs::write(args.out.join("report.txt"), &rep.0).map_err(|e| e.to_string())
}
