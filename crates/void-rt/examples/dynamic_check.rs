//! Dynamic objects, step A (docs/raytracing.md section 14): a map uploaded with the dynamic
//! reserve as the game's DLL does with `rt_dynamic = 1`, then
//!
//! - the cut-off instance list of `Dynamics` equals the static path's (nothing changed);
//! - moving a static instance (the one covering most of a primary-ray image from the
//!   snapshot camera) 50 m up and refitting the cut-off TLAS into a second TLAS removes it
//!   from the image and leaves every other pixel unchanged; moving it back and refitting
//!   again reproduces the first image; hiding it needs a rebuild (instance list changed);
//! - spawned props: entities of the snapshot decoded as rigid dynamic meshes
//!   (`DynamicAssets::prop`), placed in front of the camera, appear in the rebuilt TLAS,
//!   their reserve entries are freed again after removal;
//! - timings: N moved instances per frame (`--moved`), CPU (state update + refreshed
//!   instance list) and GPU ms of refit vs rebuild for the cut-off and the full TLAS.
//!
//! The engine-side reads (committed flag array, transforms) cannot run offline; their size
//! per frame is printed for the estimate. Offline only; logged in `local/rt/runs.log`.
//!
//! cargo run --release -p void-rt --example dynamic_check -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 [--warp --smoke]

#[path = "common/ray_probe.rs"]
mod ray_probe;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use void_rt::accel::median;
use void_rt::ingame::{
    DynamicAssets, DynamicReserve, Dynamics, Gpu, InstanceSet, NO_INSTANCE, SceneOptions,
    SceneSource, Tlas, load_scene, upload,
};
use void_rt::lights::Env;
use void_rt::run::{Report, parse_size, run_log};
use void_rt::snapshot::{Snapshot, Sun};

use ray_probe::{ImageExt, MISS, RayProbe};

#[derive(Parser)]
#[command(about = "Dynamic objects step A: moved / hidden / spawned props, refit vs rebuild")]
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
    #[arg(long, default_value_t = 256)]
    tex_size: u32,
    /// Moved instances per simulated frame (comma list).
    #[arg(long, default_value = "1,10,100,1000")]
    moved: String,
    #[arg(long, default_value_t = 20)]
    frames: usize,
    #[arg(long, default_value_t = 40.0)]
    radius: f32,
    /// Snapshot entities spawned again as dynamic props.
    #[arg(long, default_value_t = 4)]
    spawn: usize,
    /// Primary-ray image size.
    #[arg(long, default_value = "320x180")]
    size: String,
    /// WARP-sized run: TLAS limited to the 2000 instances nearest the camera, 96x54 rays,
    /// fewer frames.
    #[arg(long)]
    smoke: bool,
    #[arg(long)]
    warp: bool,
    #[arg(long, default_value = "local/rt/dynamic_check")]
    out: PathBuf,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

fn main() {
    let args = Args::parse();
    let mode = if args.warp {
        "dynamic-check warp"
    } else {
        "dynamic-check gpu"
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
        eprintln!("dynamic_check: {e}");
        std::process::exit(1);
    }
}

fn check(rep: &mut Report, failures: &mut Vec<String>, ok: bool, what: String) {
    rep.line(format!("{} {what}", if ok { "PASS" } else { "FAIL" }));
    if !ok {
        failures.push(what);
    }
}

/// `t` moved by `d` metres along z.
fn lifted(t: &[f32; 12], d: f32) -> [f32; 12] {
    let mut r = *t;
    r[11] += d;
    r
}

/// GPU ms of the last submit and CPU ms of `f`.
fn timed<T>(
    gpu: &mut Gpu,
    f: impl FnOnce(&mut Gpu) -> Result<T, String>,
) -> Result<(T, f64), String> {
    let r = f(gpu)?;
    Ok((r, gpu.last_submit_ms()))
}

#[allow(clippy::too_many_lines)]
fn run(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let mut rep = Report::default();
    let mut failures = Vec::new();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let (w, h) = if args.smoke {
        (96, 54)
    } else {
        parse_size(&args.size)?
    };
    let frames = if args.smoke {
        args.frames.min(5)
    } else {
        args.frames
    };
    let mut gpu = Gpu::new(None, args.warp)?;
    gpu.set_pause(std::time::Duration::from_millis(1));
    rep.line(format!(
        "device: {} ({} MiB, driver {}); {}",
        gpu.info.name,
        gpu.info.vram_mib,
        gpu.info.driver,
        gpu.vram_text()
    ));
    let snap = Snapshot::load(&args.snapshot)?;
    let camera = snap.camera;
    let src = SceneSource {
        game_dir: args.game.clone(),
        map: args.map.clone(),
        entities: snap.entities.clone(),
        lights: snap.lights.clone(),
        sun: Some(
            snap.sun
                .unwrap_or_else(|| Sun::from_angles(26.0, 100.0, [0.2, 0.17, 0.12])),
        ),
        env: Some(snap.env.unwrap_or(Env {
            sky_zenith: [0.0; 3],
            sky_horizon: [37.7, 52.4, 55.0],
            ambient: [1.0; 3],
        })),
    };
    // The in-game options with rt = 1 and rt_dynamic = 1 (`rt::scene::options`).
    let reserve = DynamicReserve::default();
    let opts = SceneOptions {
        texture_size: args.tex_size,
        spike_primitives: true,
        dynamic: Some(reserve),
        tlas_limit: args.smoke.then_some((2000, camera.org)),
        ..SceneOptions::default()
    };
    let t = Instant::now();
    let cpu = load_scene(&src, &opts)?;
    let base = cpu.dynamic_base;
    rep.line(format!(
        "load_scene: {:.1} s; static geometries {}, triangles {}, materials {}; reserve {:?} ({:.0} MiB of tables)",
        t.elapsed().as_secs_f64(),
        base.geometries,
        base.triangles,
        base.materials,
        reserve,
        reserve.table_bytes(true) as f64 / 1048576.0
    ));
    let mapped = cpu
        .entity_instances
        .iter()
        .filter(|&&i| i != NO_INSTANCE)
        .count();
    check(
        &mut rep,
        &mut failures,
        cpu.entity_instances.len() == snap.entities.len() && mapped > 0,
        format!(
            "entity -> instance map: {} of {} snapshot entities traced",
            mapped,
            snap.entities.len()
        ),
    );
    let gs = Arc::new(upload(&mut gpu, cpu)?);
    let tab = gs.tables();
    check(
        &mut rep,
        &mut failures,
        tab.geom_count == base.geometries + reserve.geometries
            && tab.prim_count == base.triangles + reserve.triangles
            && tab.textures.count >= tab.textures.reserve_start + reserve.textures
            && gs
                .primitives()
                .is_some_and(|p| p.size >= u64::from(tab.prim_count) * 24),
        format!(
            "kernel counts cover the reserve: geometries {}, triangles {}, materials {}, texture slots {} (reserve from {})",
            tab.geom_count,
            tab.prim_count,
            tab.mat_count,
            tab.textures.count,
            tab.textures.reserve_start
        ),
    );
    rep.line(format!(
        "upload: {:.1} s, {} instances; {}",
        gs.upload_seconds(),
        gs.instances().len(),
        gpu.vram_text()
    ));
    let mut dy = Dynamics::new(&gpu, gs.clone())?;
    let cam = camera.org;
    let cap = dy.tlas_capacity();
    // 1. Unchanged: the dynamic path lists exactly the static path's instances.
    let set0 = dy.cutoff(cam, args.radius);
    let stat = gs.cutoff_descs(cam, args.radius);
    check(
        &mut rep,
        &mut failures,
        set0.descs == stat,
        format!(
            "cut-off list without changes equals the static path's ({} instances)",
            set0.len()
        ),
    );
    let mut a = Tlas::build(&mut gpu, &set0.descs, cap)?;
    let mut b = Tlas::build(&mut gpu, &set0.descs, cap)?;
    let probe = RayProbe::new(&gpu, w, h)?;
    let img0 = probe.render(&mut gpu, &gs, &a.buffer, &camera)?;
    let hits = img0.id.iter().filter(|&&i| i != MISS).count();
    rep.line(format!("primary rays {w}x{h}: {hits} hits"));
    // 2. Move the static instance covering the most pixels 50 m up, refit into b.
    let target = (0..set0.len() as u32)
        .filter_map(|k| Some((img0.count(k), k, set0.static_instance(k as usize)?)))
        .max();
    let Some((covered, k, inst)) = target else {
        return Err("no static instance in view".into());
    };
    let orig = gs.instances()[inst as usize].transform;
    rep.line(format!(
        "target: TLAS instance {k} = scene instance {inst} ({} pixels)",
        covered
    ));
    dy.set_static(inst, Some(lifted(&orig, 50.0)), None);
    let set1 = dy
        .refresh(&set0)
        .ok_or("refresh after a move returned None")?;
    let (_, ms_refit) = timed(&mut gpu, |g| b.update_from(g, &a, &set1.descs))?;
    let img1 = probe.render(&mut gpu, &gs, &b.buffer, &camera)?;
    // Pixels not on the moved instance must keep their depth (ids of coplanar duplicates
    // may swap with the BVH layout; counted separately).
    let (mut others_changed, mut swaps) = (0, 0);
    for p in 0..img0.id.len() {
        if img0.id[p] == k {
            continue;
        }
        let (d0, d1) = (img0.depth[p], img1.depth[p]);
        if (d0 - d1).abs() > 1e-3 * d0.abs().max(d1.abs()).max(1e-9) {
            others_changed += 1;
        } else if img0.id[p] != img1.id[p] {
            swaps += 1;
        }
    }
    check(
        &mut rep,
        &mut failures,
        img1.count(k) == 0
            && others_changed == 0
            && set1.types[k as usize] == void_rt::dynamic::TYPE_MOVED
            && set1
                .types
                .iter()
                .filter(|&&t| t != void_rt::dynamic::TYPE_STATIC)
                .count()
                == 1,
        format!(
            "moved 50 m up + refit ({ms_refit:.3} ms GPU): instance {k} on {} pixels (was {covered}), other pixels with a changed depth {others_changed} (equal-depth id swaps {swaps}); instance type {} (1 = moved)",
            img1.count(k),
            set1.types[k as usize]
        ),
    );
    // 3. Back to the original transform: refit from b into a reproduces image 0.
    dy.set_static(inst, Some(orig), None);
    let set2 = dy
        .refresh(&set1)
        .ok_or("refresh after the move back returned None")?;
    a.update_from(&mut gpu, &b, &set2.descs)?;
    let img2 = probe.render(&mut gpu, &gs, &a.buffer, &camera)?;
    check(
        &mut rep,
        &mut failures,
        img2.depth_differing(&img0) == 0,
        format!(
            "moved back + refit: {} pixels with a different depth than the first image (equal-depth id swaps {})",
            img2.depth_differing(&img0),
            img2.differing(&img0)
                .saturating_sub(img2.depth_differing(&img0))
        ),
    );
    // 4. Hide: the instance list changes, so the set must be rebuilt.
    dy.set_static(inst, None, Some(true));
    let refreshed = dy.refresh(&set2).is_none();
    let set3 = dy.cutoff(cam, args.radius);
    b.rebuild(&mut gpu, &set3.descs)?;
    let img3 = probe.render(&mut gpu, &gs, &b.buffer, &camera)?;
    let gone = (0..set3.len()).all(|j| set3.static_instance(j) != Some(inst));
    check(
        &mut rep,
        &mut failures,
        refreshed && gone && set3.len() + 1 == set2.len(),
        format!(
            "hidden: refit refused, rebuilt list {} instances (was {}), {} pixels differ",
            set3.len(),
            set2.len(),
            img3.differing(&img0)
        ),
    );
    dy.set_static(inst, None, Some(false));
    // 5. Timing: N moved instances per frame, refit vs rebuild of the cut-off TLAS (the
    //    live path; the full TLAS is built once per photo, not per frame).
    let entity_slots: Vec<u32> = (0..set0.len())
        .filter_map(|j| set0.static_instance(j))
        .filter(|&i| gs.instances()[i as usize].origin == void_rt::ingame::Origin::Entity)
        .collect();
    rep.line(format!(
        "timing over {frames} frames; {} entity instances in the cut-off ({} instances)",
        entity_slots.len(),
        set0.len()
    ));
    rep.line(
        "| moved | engine reads / frame (est.) | CPU update + instance list ms | cut-off refit ms GPU | cut-off rebuild ms GPU |",
    );
    rep.line("| --- | --- | --- | --- | --- |");
    for n in args
        .moved
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
    {
        let n = n.min(entity_slots.len());
        let chosen: Vec<u32> = (0..n)
            .map(|j| entity_slots[(j * 7919) % entity_slots.len()])
            .collect();
        let mut cur = dy.cutoff(cam, args.radius);
        // Both TLASes hold this instance count before the refits (`Tlas::update_from`
        // requires it: WARP hung updating into a TLAS last built with one instance less).
        a.rebuild(&mut gpu, &cur.descs)?;
        b.rebuild(&mut gpu, &cur.descs)?;
        let (mut cpu, mut refit, mut rebuild) = (vec![], vec![], vec![]);
        for f in 0..frames {
            let off = 0.05 * ((f + 1) as f32).sin();
            let t = Instant::now();
            for &i in &chosen {
                let o = gs.instances()[i as usize].transform;
                dy.set_static(i, Some(lifted(&o, off)), None);
            }
            let next: InstanceSet = dy.refresh(&cur).ok_or("refresh (cut-off)")?;
            cpu.push(t.elapsed().as_secs_f64() * 1e3);
            let (src, dst) = if f % 2 == 0 {
                (&a, &mut b)
            } else {
                (&b, &mut a)
            };
            dst.update_from(&mut gpu, src, &next.descs)?;
            refit.push(gpu.last_submit_ms());
            cur = next;
        }
        // Rebuilds of the same list (into the TLAS not holding the latest refit).
        for _ in 0..frames.min(5) {
            let spare = if frames % 2 == 0 { &mut b } else { &mut a };
            spare.rebuild(&mut gpu, &cur.descs)?;
            rebuild.push(gpu.last_submit_ms());
        }
        for &i in &chosen {
            dy.set_static(i, Some(gs.instances()[i as usize].transform), None);
        }
        // Engine side per frame: one u32 flag per model slot (9,370 models in RX010) + one
        // 60-byte transform read and a 4-byte flag read per moved model.
        let bytes = 9370 * 4 + n * 64;
        rep.line(format!(
            "| {n} | {} reads, {:.1} KiB | {:.3} | {:.3} | {:.3} |",
            2 + 2 * n,
            bytes as f64 / 1024.0,
            median(&mut cpu),
            median(&mut refit),
            median(&mut rebuild)
        ));
    }
    let st = gpu.take_submit_stats();
    rep.line(format!(
        "submits so far: {}, longest {:.2} ms GPU",
        st.count, st.max_ms
    ));
    // 6. Spawned props: snapshot entities decoded as dynamic meshes, placed in front.
    let assets = DynamicAssets::open(&args.game, &args.map, gs.dynamic_tables().load)?;
    let fwd = camera.axis[0];
    let mut placed = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for e in snap.entities.iter() {
        if placed.len() >= args.spawn {
            break;
        }
        let key = format!("{}|{}", e.model_source, e.materials.join(","));
        if !seen.insert(key.clone()) {
            continue;
        }
        let t = Instant::now();
        let Ok(s) = assets.prop(&key, e) else {
            continue;
        };
        let tris = s.records.len();
        if tris > 20_000 {
            continue;
        }
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        dy.add_mesh(&mut gpu, &key, &s)?;
        let upload_ms = t.elapsed().as_secs_f64() * 1e3;
        let id = 0x1000 + placed.len() as u64;
        let dist = 2.0 + placed.len() as f32;
        let mut tr = e.transform;
        tr[3] = cam[0] + fwd[0] * dist;
        tr[7] = cam[1] + fwd[1] * dist;
        tr[11] = cam[2] + fwd[2] * dist;
        dy.place(id, &key, tr)?;
        rep.line(format!(
            "spawned {key}: {tris} triangles, decode {decode_ms:.0} ms, tables + BLAS {upload_ms:.0} ms"
        ));
        placed.push(id);
    }
    let set4 = dy.cutoff(cam, args.radius);
    b.rebuild(&mut gpu, &set4.descs)?;
    let img4 = probe.render(&mut gpu, &gs, &b.buffer, &camera)?;
    let dynamic_px = img4
        .id
        .iter()
        .filter(|&&i| i != MISS && set4.dynamic_id(i as usize).is_some())
        .count();
    let used = dy.stats().triangles_used;
    check(
        &mut rep,
        &mut failures,
        !placed.is_empty()
            && set4.dynamic_count() == placed.len()
            && dynamic_px > 0
            && (0..set4.len())
                .filter(|&j| set4.dynamic_id(j).is_some())
                .all(|j| set4.types[j] == void_rt::dynamic::TYPE_SPAWNED),
        format!(
            "{} props placed in front of the camera: {} in the TLAS (instance type 2 = spawned), {dynamic_px} pixels hit them; reserve {} triangles in use",
            placed.len(),
            set4.dynamic_count(),
            used
        ),
    );
    for id in &placed {
        dy.remove(*id);
    }
    let set5 = dy.cutoff(cam, args.radius);
    a.rebuild(&mut gpu, &set5.descs)?;
    dy.collect(&mut gpu)?;
    let held = dy.stats().triangles_used;
    drop(set4);
    dy.collect(&mut gpu)?;
    let freed = dy.stats().triangles_used;
    check(
        &mut rep,
        &mut failures,
        held == used && freed == 0,
        format!(
            "removed props: reserve kept while an instance set references them ({held} triangles), freed after ({freed})"
        ),
    );
    let img5 = probe.render(&mut gpu, &gs, &a.buffer, &camera)?;
    check(
        &mut rep,
        &mut failures,
        img5.depth_differing(&img0) == 0,
        format!(
            "after removal the depth equals the first image's ({} pixels differ, {} equal-depth id swaps)",
            img5.depth_differing(&img0),
            img5.differing(&img0)
                .saturating_sub(img5.depth_differing(&img0))
        ),
    );
    let st = gpu.take_submit_stats();
    rep.line(format!(
        "submits (props): {}, longest {:.2} ms GPU; total {:.1} s; {}",
        st.count,
        st.max_ms,
        t0.elapsed().as_secs_f64(),
        gpu.vram_text()
    ));
    rep.line(if failures.is_empty() {
        "RESULT: all checks passed".to_owned()
    } else {
        format!("RESULT: {} failed: {}", failures.len(), failures.join("; "))
    });
    let name = if args.warp {
        "report_warp.txt"
    } else {
        "report.txt"
    };
    std::fs::write(args.out.join(name), &rep.0).map_err(|e| e.to_string())?;
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("{} checks failed", failures.len()))
    }
}
