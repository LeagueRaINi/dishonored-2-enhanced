//! Dynamic objects, step B (docs/raytracing.md section 14): GPU skinning of an `.md6`
//! character (`Dynamics` + `dynamic.hlsl`) checked against the decoded bind pose and a CPU
//! reference, and the BLAS refit cost for several characters.
//!
//! - palette of the bind pose (joint world x inverse bind) = identity up to rounding; the
//!   skinned positions of that pose equal the decoded vertices;
//! - one joint rotated by 60 degrees (local x): the GPU result equals the CPU reference
//!   (`CharacterCpu::skin_vertex`), vertices without weight on that joint's subtree stay
//!   where they are, the others move; the skinned normals are written into the
//!   character's photo records only where the pose changed them;
//! - primary rays (`rt.hlsl` `primary`) through a TLAS with the character: the silhouette
//!   changes between the two poses (the refitted BLAS is traced);
//! - the in-game palette check (`palette_check`) recognises a skinning palette and a
//!   joint-world palette;
//! - `--count` characters posed per frame for `--frames` frames: GPU ms of the skinning +
//!   BLAS update submit, of the TLAS refit, memory per character.
//!
//! The scene is a floor quad uploaded with the dynamic reserve (no map geometry). Offline
//! only; logged in `local/rt/runs.log`.
//!
//! cargo run --release -p void-rt --example skin_check -- --game .. [--warp] \
//!   [--md6 models/characters/tall/overseer/overseer_head.md6] [--count 12]

#[path = "common/ray_probe.rs"]
#[allow(dead_code)] // shared with dynamic_check
mod ray_probe;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use void_formats::md6::{Mat34, mul34};
use void_rt::accel::median;
use void_rt::camera::Camera;
use void_rt::character::{IDENTITY, PaletteMode, Row34, palette_check, row34, unpack_influences};
use void_rt::ingame::{
    CpuScene, DynamicAssets, DynamicReserve, Dynamics, Gpu, LoadOptions, SceneOptions, SceneSource,
    SceneStats, Tlas, upload,
};
use void_rt::run::{Report, run_log};
use void_rt::scene::{
    Geometry, Instance, MATERIAL_WORDS, MaterialInfo, Mesh, Origin, RECORD_WORDS, Scene,
    pack_vertex,
};

use ray_probe::{ImageExt, MISS, RayProbe};

#[derive(Parser)]
#[command(about = "Dynamic objects step B: GPU skinning of md6 characters, BLAS refits")]
struct Args {
    #[arg(long)]
    game: PathBuf,
    /// Map whose archive group is preferred (as in game).
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    #[arg(
        long,
        default_value = "models/characters/tall/overseer/overseer_head.md6"
    )]
    md6: String,
    #[arg(long, default_value_t = 0)]
    lod: u32,
    /// More md6 decoded for the all-surfaces check (comma list; the user's guard, a rat,
    /// a female civilian).
    #[arg(
        long,
        default_value = "models/characters/small/guards/gs_guards/gs_guard_01/gs_guard_01_head1.md6,models/characters/animals/rat/rat.md6"
    )]
    also: String,
    /// Characters posed per frame in the timing part.
    #[arg(long, default_value_t = 12)]
    count: usize,
    #[arg(long, default_value_t = 20)]
    frames: usize,
    #[arg(long, default_value_t = 256)]
    tex_size: u32,
    #[arg(long)]
    warp: bool,
    #[arg(long, default_value = "local/rt/skin_check")]
    out: PathBuf,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

fn main() {
    let args = Args::parse();
    let mode = if args.warp {
        "skin-check warp"
    } else {
        "skin-check gpu"
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
        eprintln!("skin_check: {e}");
        std::process::exit(1);
    }
}

fn check(rep: &mut Report, failures: &mut Vec<String>, ok: bool, what: String) {
    rep.line(format!("{} {what}", if ok { "PASS" } else { "FAIL" }));
    if !ok {
        failures.push(what);
    }
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A static floor quad (20 x 20 m at z = 0) with photo records and FC6 primitives.
fn floor_scene() -> Scene {
    let m = MaterialInfo {
        name: "floor".into(),
        albedo: [0.5; 3],
        albedo_from_texture: false,
        smoothness: 0.3,
        metallic: 0.0,
        two_sided: true,
        alpha_test: false,
        decal: false,
        uv_set: 0,
    };
    let [w0, w1] = pack_vertex(&m, [0.0, 0.0, 1.0]);
    Scene {
        positions: vec![
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 10.0, 0.0],
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        primitives: vec![[w0, w1, w0, w1, w0, w1]; 2],
        records: vec![[0; RECORD_WORDS]; 2],
        meshes: vec![Mesh {
            name: "floor".into(),
            geometries: vec![Geometry {
                first_vertex: 0,
                vertex_count: 4,
                first_index: 0,
                index_count: 6,
                first_primitive: 0,
                opaque: true,
            }],
            first_geometry: 0,
            aabb_min: [-10.0, -10.0, 0.0],
            aabb_max: [10.0, 10.0, 0.0],
        }],
        primitive_offsets: vec![0],
        geometry_materials: vec![0],
        instances: vec![Instance {
            mesh: 0,
            transform: IDENTITY,
            origin: Origin::World,
            two_sided: true,
        }],
        materials: vec![m],
        gpu_materials: vec![[0; MATERIAL_WORDS]],
        textures: Vec::new(),
        stats: SceneStats::default(),
    }
}

/// Rotation about x by `deg` (row-major 3x4).
fn rot_x(deg: f32) -> Mat34 {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0]]
}

fn translate(t: [f32; 3]) -> Row34 {
    let mut r = IDENTITY;
    r[3] = t[0];
    r[7] = t[1];
    r[11] = t[2];
    r
}

#[allow(clippy::too_many_lines)]
fn run(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let mut rep = Report::default();
    let mut failures = Vec::new();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let mut gpu = Gpu::new(None, args.warp)?;
    rep.line(format!(
        "device: {} ({} MiB, driver {}); {}",
        gpu.info.name,
        gpu.info.vram_mib,
        gpu.info.driver,
        gpu.vram_text()
    ));
    let opts = SceneOptions {
        texture_size: args.tex_size,
        spike_primitives: true,
        dynamic: Some(DynamicReserve::default()),
        ..SceneOptions::default()
    };
    let src = SceneSource {
        game_dir: args.game.clone(),
        map: args.map.clone(),
        ..SceneSource::default()
    };
    let gs = Arc::new(upload(
        &mut gpu,
        CpuScene::from_scene(floor_scene(), &src, &opts),
    )?);
    let mut dy = Dynamics::new(&gpu, gs.clone())?;
    let load = LoadOptions {
        photo_texture_size: Some(args.tex_size),
        decals: true,
        spike_primitives: true,
    };
    let t = Instant::now();
    let mut assets = DynamicAssets::open(&args.game, &args.map, load)?;
    let ch = assets.character(&args.md6, args.lod, &[])?;
    rep.line(format!(
        "{} LOD {}: {} vertices, {} triangles, {} joints, {} materials, {} textures; decoded in {:.0} ms{}{}",
        ch.md6,
        ch.lod,
        ch.vertex_count(),
        ch.triangle_count(),
        ch.joint_count(),
        ch.mesh.materials.len(),
        ch.mesh.textures.len(),
        t.elapsed().as_secs_f64() * 1e3,
        if ch.notes.is_empty() { "" } else { "; " },
        ch.notes.join("; ")
    ));
    // 0. Every surface traced (fallback primitive data for unsupported programs), for this
    //    character and the extra ones of `--also`.
    let mut all_ok = ch.surfaces[0] == ch.surfaces[1] && ch.surfaces[1] > 0;
    let mut lines = vec![format!(
        "{}: {} of {} surfaces, parts {}, fallback materials {:?}",
        ch.md6,
        ch.surfaces[0],
        ch.surfaces[1],
        ch.meshes.join(", "),
        ch.fallbacks
    )];
    for md6 in args
        .also
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        match assets.character(md6, 0, &[]) {
            Ok(c) => {
                all_ok &= c.surfaces[0] == c.surfaces[1] && c.surfaces[1] > 0;
                lines.push(format!(
                    "{md6}: {} of {} surfaces, {} triangles, parts {}, fallback materials {:?}",
                    c.surfaces[0],
                    c.surfaces[1],
                    c.triangle_count(),
                    c.meshes.join(", "),
                    c.fallbacks
                ));
            }
            Err(e) => {
                all_ok = false;
                lines.push(format!("{md6}: {e}"));
            }
        }
    }
    check(
        &mut rep,
        &mut failures,
        all_ok,
        format!("every character surface traced:\n  {}", lines.join("\n  ")),
    );
    // 1. Bind palette = identity.
    let bind = ch.bind_palette();
    let id_err = bind
        .iter()
        .flat_map(|m| m.iter().zip(IDENTITY).map(|(a, b)| (a - b).abs()))
        .fold(0f32, f32::max);
    check(
        &mut rep,
        &mut failures,
        id_err < 1e-3,
        format!("bind palette = identity: largest element error {id_err:.2e}"),
    );
    // The character stands 3 m in front of a camera looking along +y.
    let place = [0.0, 3.0, 0.0];
    let camera = Camera::from_pose([0.0, 0.0, 1.0], 90.0, 0.0, 1.0);
    let id = 1u64;
    let t = Instant::now();
    dy.add_character(&mut gpu, id, &ch, translate(place))?;
    rep.line(format!(
        "add_character: {:.0} ms (tables, buffers, bind-pose skin + BLAS build), {:.1} MiB",
        t.elapsed().as_secs_f64() * 1e3,
        Dynamics::character_bytes(&gpu, &ch) as f64 / 1048576.0
    ));
    let max_err = |a: &[[f32; 3]], b: &[[f32; 3]]| {
        a.iter()
            .zip(b)
            .map(|(p, q)| dist(*p, *q))
            .fold(0f32, f32::max)
    };
    let skinned = dy.read_skinned(&mut gpu, id)?;
    let e_add = max_err(&skinned, &ch.mesh.positions);
    // 2. The same pose through the update path (pose + skin + BLAS update).
    dy.pose(id, bind.clone(), translate(place))?;
    let st = dy.update(&mut gpu, camera.org, 1, 100.0)?;
    let skinned = dy.read_skinned(&mut gpu, id)?;
    let e_upd = max_err(&skinned, &ch.mesh.positions);
    check(
        &mut rep,
        &mut failures,
        e_add < 1e-4 && e_upd < 1e-4 && st.skinned == 1,
        format!(
            "bind pose skinned = decoded vertices: largest error {e_add:.2e} m (add), {e_upd:.2e} m (update, {:.3} ms GPU)",
            st.gpu_ms
        ),
    );
    let records_bind = dy.read_records(&mut gpu, id)?;
    let set_bind = dy.full();
    let tlas_bind = Tlas::build(&mut gpu, &set_bind.descs, dy.tlas_capacity())?;
    let probe = RayProbe::new(&gpu, 160, 160)?;
    let img_bind = probe.render(&mut gpu, &gs, &tlas_bind.buffer, &camera)?;
    let char_k = (0..set_bind.len())
        .find(|&k| set_bind.dynamic_id(k) == Some(id))
        .ok_or("character not in the TLAS")? as u32;
    // 3. One joint rotated: the joint dominating the most vertices among non-roots.
    let joints = ch.joint_count() as usize;
    let mut dominant = vec![0usize; joints];
    for inf in &ch.influences {
        let u = unpack_influences(inf);
        if let Some(&(j, w)) = u.iter().max_by(|a, b| a.1.total_cmp(&b.1))
            && w > 0.5
        {
            dominant[j as usize] += 1;
        }
    }
    let j = (0..joints)
        .filter(|&k| ch.skeleton.joints[k].parent.is_some())
        .max_by_key(|&k| dominant[k])
        .ok_or("no joint")?;
    let mut subtree = vec![false; joints];
    for k in 0..joints {
        subtree[k] = k == j
            || ch.skeleton.joints[k]
                .parent
                .is_some_and(|p| subtree[usize::from(p)]);
    }
    let world = ch.skeleton.global_pose(|k| {
        let l = ch.skeleton.joints[k].local.matrix();
        if k == j { mul34(&l, &rot_x(60.0)) } else { l }
    });
    let posed = ch.palette_for(&world);
    dy.pose(id, posed.clone(), translate(place))?;
    let st = dy.update(&mut gpu, camera.org, 1, 100.0)?;
    let skinned = dy.read_skinned(&mut gpu, id)?;
    let reference: Vec<[f32; 3]> = (0..skinned.len())
        .map(|v| ch.skin_vertex(&posed, v))
        .collect();
    let e_ref = max_err(&skinned, &reference);
    let affected: Vec<bool> = ch
        .influences
        .iter()
        .map(|inf| {
            unpack_influences(inf)
                .iter()
                .any(|&(jj, w)| w > 0.0 && subtree[jj as usize])
        })
        .collect();
    let (mut still, mut moved_n, mut moved_max) = (0f32, 0usize, 0f32);
    for (v, p) in skinned.iter().enumerate() {
        let d = dist(*p, ch.mesh.positions[v]);
        if affected[v] {
            moved_n += usize::from(d > 0.01);
            moved_max = moved_max.max(d);
        } else {
            still = still.max(d);
        }
    }
    check(
        &mut rep,
        &mut failures,
        e_ref < 1e-3 && still < 1e-4 && moved_n > 0,
        format!(
            "joint {j} '{}' rotated 60 deg ({} subtree joints, {} weighted vertices): GPU vs CPU reference {e_ref:.2e} m; unweighted vertices moved {still:.2e} m; {moved_n} weighted vertices moved > 1 cm (max {moved_max:.2} m); {:.3} ms GPU",
            ch.skeleton.joints[j].name,
            subtree.iter().filter(|&&s| s).count(),
            affected.iter().filter(|&&a| a).count(),
            st.gpu_ms
        ),
    );
    // Skinned normals in the records: changed only on triangles touching moved vertices.
    let records = dy.read_records(&mut gpu, id)?;
    let (mut same_ok, mut changed) = (true, 0usize);
    for (t, tv) in ch.tri_verts.iter().enumerate() {
        let any = tv.iter().any(|&v| affected[v as usize]);
        let differs = records[t] != records_bind[t];
        if !any && differs {
            same_ok = false;
        }
        changed += usize::from(differs);
    }
    check(
        &mut rep,
        &mut failures,
        same_ok
            && changed > 0
            && records_bind
                .iter()
                .zip(&ch.mesh.records)
                .all(|(a, b)| a[0] == b[0]),
        format!(
            "records: skinned normals rewritten on {changed} triangles, untouched triangles unchanged: {same_ok}"
        ),
    );
    // 4. Primary rays through a TLAS refitted with the posed BLAS.
    let set_posed = dy
        .refresh(&set_bind)
        .ok_or("refresh after a pose returned None")?;
    let mut tlas_posed = Tlas::build(&mut gpu, &set_bind.descs, dy.tlas_capacity())?;
    tlas_posed.update_from(&mut gpu, &tlas_bind, &set_posed.descs)?;
    let img_posed = probe.render(&mut gpu, &gs, &tlas_posed.buffer, &camera)?;
    let (hb, hp) = (img_bind.count(char_k), img_posed.count(char_k));
    let diff = img_posed.differing(&img_bind);
    check(
        &mut rep,
        &mut failures,
        hb > 0 && hp > 0 && diff > 0,
        format!(
            "primary rays 160x160: character on {hb} pixels (bind) / {hp} (posed, refitted TLAS), {diff} pixels differ, misses {}",
            img_posed.id.iter().filter(|&&i| i == MISS).count()
        ),
    );
    // 5. The in-game palette check.
    let joint_world: Vec<Row34> = world.iter().map(row34).collect();
    let v_s = palette_check(&ch, &posed);
    let v_j = palette_check(&ch, &joint_world);
    check(
        &mut rep,
        &mut failures,
        v_s.mode == Some(PaletteMode::Skinning) && v_j.mode == Some(PaletteMode::JointWorld),
        format!(
            "palette check: skinning palette -> {:?} ({:?}), joint-world palette -> {:?} ({:?})",
            v_s.mode, v_s.skinning, v_j.mode, v_j.joint_world
        ),
    );
    drop((set_bind, set_posed, tlas_bind, tlas_posed));
    // 6. Timing: `count` characters posed every frame.
    let n = args.count.max(1);
    for k in 1..n {
        let p = [(k % 4) as f32 * 1.5 - 2.0, 3.0 + (k / 4) as f32 * 1.5, 0.0];
        dy.add_character(&mut gpu, id + k as u64, &ch, translate(p))?;
    }
    let ids: Vec<u64> = (0..n as u64).map(|k| id + k).collect();
    let pose_at = |f: usize| {
        let a = 45.0 * (f as f32 * 0.4).sin();
        let world = ch.skeleton.global_pose(|k| {
            let l = ch.skeleton.joints[k].local.matrix();
            if k == j { mul34(&l, &rot_x(a)) } else { l }
        });
        ch.palette_for(&world)
    };
    for &c in &ids {
        dy.pose(c, pose_at(0), dy_transform(&dy, c))?;
    }
    dy.update(&mut gpu, camera.org, n, 1000.0)?;
    let mut cur = dy.full();
    let mut ta = Tlas::build(&mut gpu, &cur.descs, dy.tlas_capacity())?;
    let mut tb = Tlas::build(&mut gpu, &cur.descs, dy.tlas_capacity())?;
    let (mut skin_ms, mut refit_ms, mut per_char) = (Vec::new(), Vec::new(), Vec::new());
    let mut skinned_total = 0;
    for f in 1..=args.frames {
        let pal = pose_at(f);
        for &c in &ids {
            dy.pose(c, pal.clone(), dy_transform(&dy, c))?;
        }
        let st = dy.update(&mut gpu, camera.org, n, 1000.0)?;
        skinned_total += st.skinned;
        if st.skinned > 0 {
            skin_ms.push(st.gpu_ms);
            per_char.push(st.gpu_ms / st.skinned as f64);
        }
        let next = dy
            .refresh(&cur)
            .ok_or("refresh in the timing loop returned None")?;
        let (src, dst) = if f % 2 == 1 {
            (&ta, &mut tb)
        } else {
            (&tb, &mut ta)
        };
        dst.update_from(&mut gpu, src, &next.descs)?;
        refit_ms.push(gpu.last_submit_ms());
        cur = next;
    }
    let stats = dy.stats();
    let sub = gpu.take_submit_stats();
    check(
        &mut rep,
        &mut failures,
        skinned_total == n * args.frames,
        format!(
            "{n} characters x {} frames: {skinned_total} skinned (expected {}); skin + BLAS update {:.3} ms GPU per frame (median), {:.3} ms per character; TLAS refit {:.3} ms; longest submit {:.2} ms",
            args.frames,
            n * args.frames,
            median(&mut skin_ms),
            median(&mut per_char),
            median(&mut refit_ms),
            sub.max_ms
        ),
    );
    rep.line(format!(
        "memory: {:.1} MiB for {} characters ({:.1} MiB each incl. {} BLAS copies max), reserve {} of {} triangles; {}",
        stats.bytes as f64 / 1048576.0,
        stats.characters,
        stats.bytes as f64 / 1048576.0 / stats.characters.max(1) as f64,
        void_rt::dynamic::BLAS_COPIES,
        stats.triangles_used,
        stats.triangles_reserved,
        gpu.vram_text()
    ));
    rep.line(format!("total {:.1} s", t0.elapsed().as_secs_f64()));
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

/// The character's current transform (unchanged while posing in place).
fn dy_transform(dy: &Dynamics, id: u64) -> Row34 {
    dy.character_transform(id).unwrap_or(IDENTITY)
}
