//! Offline photo mode (docs/raytracing.md section 10, STATUS E073): path-traces the decoded
//! static scene at a snapshot camera with many samples per pixel and compares the result with
//! the game's frame. Drives the same in-process API the game's DLL uses (`void_rt::ingame`:
//! `load_scene`, `upload`, `PhotoSession` with precompiled DXIL). Offline: reads the game's
//! archives (explicit install path) and F3/F8 captures under `local/`; never runs or touches
//! the game.
//!
//! Validation first (WARP, tiny image), then one GPU run:
//! cargo run --release -p void-rt --example photo -- --smoke --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288
//! cargo run --release -p void-rt --example photo -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 \
//!   --ref-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
//!   --ref-color local/ref/RX010/dump1_color_rgba16f_3440x1440.bin

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use void_rt::accel::median;
use void_rt::camera::Camera;
use void_rt::gpu::Gpu;
use void_rt::image::{auto_exposure, ratio_colour, tonemap, write_hdr, write_png};
use void_rt::ingame::{
    GpuScene, PhotoSession, PhotoSettings, SceneOptions, SceneSource, TF_NO_ALPHA, TF_NO_LIGHTS,
    TF_NO_SKY, TF_NO_SUN, TF_NO_TEXTURES, load_scene, upload,
};
use void_rt::lights::{Env, SUN_COLOR_SCALE};
use void_rt::run::{Report, parse_size, read_f32, read_rgba16f, run_log};
use void_rt::snapshot::{DEFAULT_ZNEAR, Snapshot, Sun};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Entities {
    /// Snapshot when `--snapshot` is given, else the map's `.entities` file.
    Auto,
    File,
    Snapshot,
    None,
}

#[derive(Parser)]
#[command(about = "Offline photo-mode path tracer over a decoded Dishonored 2 map")]
struct Args {
    /// Game installation root (the directory holding `base/`).
    #[arg(long)]
    game: PathBuf,
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    /// F3 snapshot directory (camera, sun, sky colours, live entities and lights).
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Entities::Auto)]
    entities: Entities,
    /// Camera pose `x,y,z,yaw,pitch` when no snapshot is given.
    #[arg(
        long,
        default_value = "-33.65,181.69,10.46,-49.25,13.72",
        allow_hyphen_values = true
    )]
    pose: String,
    /// Vertical field of view (radians) for `--pose` (RX010 `fov_y`).
    #[arg(long, default_value_t = 1.024_779)]
    fov_y: f32,
    /// Image size `WxH`.
    #[arg(long, default_value = "3440x1440")]
    size: String,
    /// Samples per pixel.
    #[arg(long, default_value_t = 512)]
    spp: u32,
    /// Sample counts at which intermediate images are written (plus `--spp`).
    #[arg(long, default_value = "1,4,16,64,256")]
    save_at: String,
    /// Bounces after the primary hit.
    #[arg(long, default_value_t = 4)]
    bounces: u32,
    /// Russian roulette from this bounce on.
    #[arg(long, default_value_t = 2)]
    rr_start: u32,
    /// Luminance clamp of every contribution after the primary hit (engine units).
    #[arg(long, default_value_t = 100.0)]
    clamp: f32,
    /// RIS candidates per local-light pick.
    #[arg(long, default_value_t = 8)]
    ris: u32,
    /// Largest texture edge uploaded (texels; mips above are skipped).
    #[arg(long, default_value_t = 512)]
    tex_size: u32,
    /// Exclude arkDecal mesh decals.
    #[arg(long)]
    no_decals: bool,
    /// Sun disc angular radius (degrees).
    #[arg(long, default_value_t = 0.27)]
    sun_radius: f32,
    /// Sky radiance multiplier on lerp(skyZenith, skyHorizon). Default fitted at RX010: with
    /// 1.0 the sky seen by both the game and us was 15x brighter than the game's (game/ours
    /// 0.067, `docs/raytracing.md` section 10); the env values are probe bounce factors,
    /// not sky radiance (H).
    #[arg(long, default_value_t = 0.067)]
    sky_scale: f32,
    /// Light grid cell (metres).
    #[arg(long, default_value_t = 4.0)]
    light_cell: f32,
    /// A-trous iterations of the denoiser.
    #[arg(long, default_value_t = 5)]
    denoise_iters: u32,
    /// Turn off terms (comma list of sun, sky, lights, textures, alpha = alpha-tested and
    /// decal geometry traced opaque).
    #[arg(long, default_value = "")]
    disable: String,
    /// Game depth dump (`r32f`, near / z) of the snapshot camera.
    #[arg(long)]
    ref_depth: Option<PathBuf>,
    /// Game colour dump (`rgba16f`, pre-TAA HDR) of the same frame.
    #[arg(long)]
    ref_color: Option<PathBuf>,
    #[arg(long, default_value = "3440x1440")]
    ref_size: String,
    /// Stored game colour per absolute unit (docs/renderer-replacement.md: 255 x LF /
    /// 139.26 with the daylight-clamped LF 1.956).
    #[arg(long, default_value_t = 3.58)]
    dump_scale: f32,
    /// GPU time budget per submit (ms, `PhotoSession::step`); at most 8 tiles per submit.
    #[arg(long, default_value_t = 400.0)]
    submit_ms: f64,
    /// Pixels per tile (dispatch); 0 = the whole image per dispatch.
    #[arg(long, default_value_t = 0)]
    tile_pixels: u32,
    #[arg(long, default_value = "local/rt/photo")]
    out: PathBuf,
    /// WARP (software D3D12 with DXR 1.1): driver-independent validation device.
    #[arg(long)]
    warp: bool,
    /// Validation run: WARP, 96x40, 4 spp, 64-texel textures, 2000 nearest instances.
    #[arg(long)]
    smoke: bool,
    /// Limit the TLAS to the instances nearest the camera (smoke default 2000).
    #[arg(long)]
    max_instances: Option<usize>,
    /// Debug output of the primary hit through every pixel centre (kernel `photo_aov`):
    /// `aov_albedo_id_rgba32f.bin` (linear albedo rgb, material index + 1 as f32 bits),
    /// `aov_uv_lod_depth_rgba32f.bin`, `aov_albedo.png` (sRGB) and `aov_materials.tsv`
    /// (index + 1, name, texture slots with image name and format).
    #[arg(long)]
    aov: bool,
    #[arg(long, default_value_t = 256)]
    scratch_mib: u64,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

const MIB: f64 = 1024.0 * 1024.0;

fn lum(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn rgb_image(n: usize, f: impl Fn(usize) -> [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 3);
    for i in 0..n {
        v.extend_from_slice(&f(i));
    }
    v
}

fn rel_rmse(a: &[f32], reference: &[f32]) -> f64 {
    let mean = reference.iter().map(|&v| f64::from(v)).sum::<f64>() / reference.len().max(1) as f64;
    let mse = a
        .iter()
        .zip(reference)
        .map(|(&x, &y)| f64::from(x - y).powi(2))
        .sum::<f64>()
        / a.len().max(1) as f64;
    mse.sqrt() / mean.max(1e-12)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("photo: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = Args::parse();
    if args.smoke {
        args.warp = true;
        args.size = "96x40".into();
        args.spp = args.spp.min(4);
        args.save_at = "1,2".into();
        args.tex_size = args.tex_size.min(64);
        if args.out == Path::new("local/rt/photo") {
            args.out = PathBuf::from("local/rt/photo/smoke");
        }
    }
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let mode = if args.smoke {
        "photo smoke"
    } else if args.warp {
        "photo warp"
    } else {
        "photo gpu"
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
    let r = photo_run(&args);
    run_log(
        &args.runs_log,
        &match &r {
            Ok(()) => format!("end {mode} run: ok"),
            Err(e) => format!("end {mode} run: error: {}", e.lines().next().unwrap_or("")),
        },
    );
    r
}

/// Per-region comparison: game stored colour vs ours x dump scale (depth-matched pixels).
struct Region {
    name: &'static str,
    pixels: usize,
    ratios: Vec<f32>,
    sum_game: f64,
    sum_ours: f64,
}

impl Region {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            pixels: 0,
            ratios: Vec::new(),
            sum_game: 0.0,
            sum_ours: 0.0,
        }
    }

    fn add(&mut self, game: f32, ours: f32) {
        self.pixels += 1;
        self.sum_game += f64::from(game);
        self.sum_ours += f64::from(ours);
        if ours > 0.0 && game > 0.0 {
            self.ratios.push(game / ours);
        }
    }

    fn line(&mut self, total: usize) -> String {
        let q = |v: &mut Vec<f32>, p: f64| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            let k = ((v.len() - 1) as f64 * p) as usize;
            let (_, x, _) = v.select_nth_unstable_by(k, f32::total_cmp);
            f64::from(*x)
        };
        let (p10, p50, p90) = (
            q(&mut self.ratios, 0.1),
            q(&mut self.ratios, 0.5),
            q(&mut self.ratios, 0.9),
        );
        format!(
            "  {:<34} {:>9} px ({:5.1} %): game/ours median {:.3} (p10 {:.3}, p90 {:.3}), ratio of sums {:.3}; mean luminance game {:.3} ours {:.3}",
            self.name,
            self.pixels,
            100.0 * self.pixels as f64 / total.max(1) as f64,
            p50,
            p10,
            p90,
            self.sum_game / self.sum_ours.max(1e-12),
            self.sum_game / self.pixels.max(1) as f64,
            self.sum_ours / self.pixels.max(1) as f64
        )
    }
}

/// Runs `photo_aov` over the image (bands of 64 rows, one blocking submit each) and writes
/// the debug outputs (see `Args::aov`).
#[allow(clippy::too_many_arguments)]
fn write_aov(
    gpu: &mut Gpu,
    gs: &GpuScene,
    camera: &Camera,
    w: u32,
    h: u32,
    settings: &PhotoSettings,
    dir: &Path,
    materials: &str,
) -> Result<(), String> {
    let n = (w * h) as usize;
    let params = gpu.upload(&gs.params(camera, w, h, settings), false)?;
    let a = gpu.buffer(n as u64 * 16, true)?;
    let b = gpu.buffer(n as u64 * 16, true)?;
    let t = gs.tables();
    let srv = [
        Some(&gs.full_tlas().buffer),
        Some(&t.geometry),
        Some(&t.records),
        Some(&t.materials),
        Some(&t.lights),
        Some(&t.cells),
        Some(&t.light_indices),
        Some(&params),
    ];
    const BAND: u32 = 64;
    for y0 in (0..h).step_by(BAND as usize) {
        gs.programs().dispatch_rows(
            gpu,
            "photo_aov",
            &[w, h, 0, 0, settings.flags, y0, 0, 0],
            &srv,
            &[None, None, None, Some(&a), Some(&b)],
            Some(&t.textures),
            (w, h),
            (y0, BAND.min(h - y0)),
        )?;
        gpu.submit()?;
    }
    let alb: Vec<[f32; 4]> = gpu.read(&b, n)?;
    let uv: Vec<[f32; 4]> = gpu.read(&a, n)?;
    let raw =
        |v: &[[f32; 4]]| -> Vec<u8> { v.iter().flatten().flat_map(|f| f.to_le_bytes()).collect() };
    let save = |name: &str, bytes: &[u8]| {
        std::fs::write(dir.join(name), bytes).map_err(|e| format!("{name}: {e}"))
    };
    save("aov_albedo_id_rgba32f.bin", &raw(&alb))?;
    save("aov_uv_lod_depth_rgba32f.bin", &raw(&uv))?;
    save("aov_materials.tsv", materials.as_bytes())?;
    let srgb = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    write_png(
        &dir.join("aov_albedo.png"),
        w,
        h,
        &rgb_image(n, |i| [srgb(alb[i][0]), srgb(alb[i][1]), srgb(alb[i][2])]),
    )
}

#[allow(clippy::too_many_lines)]
fn photo_run(args: &Args) -> Result<(), String> {
    let t_start = Instant::now();
    let mut rep = Report::default();
    let out = |n: &str| args.out.join(n);
    let (w, h) = parse_size(&args.size)?;
    let mut disable = 0;
    for f in args
        .disable
        .split(',')
        .map(str::trim)
        .filter(|f| !f.is_empty())
    {
        disable |= match f {
            "sun" => TF_NO_SUN,
            "sky" => TF_NO_SKY,
            "lights" => TF_NO_LIGHTS,
            "textures" => TF_NO_TEXTURES,
            "alpha" => TF_NO_ALPHA,
            _ => {
                return Err(format!(
                    "--disable {f}: expected sun, sky, lights, textures, alpha"
                ));
            }
        };
    }

    // ---------------------------------------------------------------- device
    let mut gpu = Gpu::new(None, args.warp)?;
    let vram0 = gpu.vram_usage();
    if args.warp {
        rep.line("device: WARP (software; timings are not GPU timings)");
    }
    let info = gpu.info.clone();
    rep.line(format!(
        "adapter: {} (vendor {:#06x}, device {:#06x}, {} MiB), driver {}; raytracing tier {}, shader model {}",
        info.name,
        info.vendor_id,
        info.device_id,
        info.vram_mib,
        info.driver,
        info.raytracing_tier_name(),
        info.shader_model_name()
    ));
    if !info.inline_raytracing() {
        return Err("inline ray tracing (tier 1.1, SM 6.5) not supported".into());
    }
    rep.line(format!(
        "shaders: precompiled DXIL (build.rs, cs_6_5): {}",
        void_rt::ingame::dxil_entries()
            .filter(|e| e.starts_with("photo_"))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    // ---------------------------------------------------------------- inputs
    let snapshot = args.snapshot.as_deref().map(Snapshot::load).transpose()?;
    let (camera, znear) = match &snapshot {
        Some(s) => (s.camera, s.znear),
        None => {
            let p: Vec<f32> = args
                .pose
                .split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect();
            let [x, y, z, yaw, pitch] = p[..] else {
                return Err(format!("--pose {}: expected x,y,z,yaw,pitch", args.pose));
            };
            (
                Camera::from_pose([x, y, z], yaw, pitch, args.fov_y),
                DEFAULT_ZNEAR,
            )
        }
    };
    // RX010 values (snap_1792183388_f1288 sun.tsv / env.tsv) when the snapshot lacks them.
    let sun = snapshot
        .as_ref()
        .and_then(|s| s.sun)
        .unwrap_or_else(|| Sun::from_angles(26.0, 100.0, [0.215_861, 0.171_441, 0.116_971]));
    let env = snapshot.as_ref().and_then(|s| s.env).unwrap_or(Env {
        sky_zenith: [0.0; 3],
        sky_horizon: [37.7451, 52.411_766, 55.0],
        ambient: [1.0; 3],
    });
    let (entities, map_entities) = match (args.entities, &snapshot) {
        (Entities::None, _) => (Vec::new(), false),
        (Entities::File, _) | (Entities::Auto, None) => (Vec::new(), true),
        (Entities::Snapshot | Entities::Auto, Some(s)) => (s.entities.clone(), false),
        (Entities::Snapshot, None) => return Err("--entities snapshot needs --snapshot".into()),
    };
    rep.line(format!(
        "camera: org {:?} forward {:?} fov_y {:.4} znear {} ({}); image {w}x{h}",
        camera.org,
        camera.axis[0],
        camera.fov_y,
        znear,
        args.snapshot
            .as_ref()
            .map_or("synthetic pose".to_owned(), |p| p.display().to_string())
    ));

    // ---------------------------------------------------------------- scene (ingame API)
    let snap_lights = snapshot
        .as_ref()
        .map(|s| s.lights.clone())
        .unwrap_or_default();
    let src = SceneSource {
        game_dir: args.game.clone(),
        map: args.map.clone(),
        entities,
        lights: snap_lights.clone(),
        sun: Some(sun),
        env: Some(env),
    };
    let limit = args.max_instances.or(args.smoke.then_some(2000));
    let opts = SceneOptions {
        texture_size: args.tex_size,
        decals: !args.no_decals,
        map_entities,
        light_cell: args.light_cell,
        blas_scratch_bytes: args.scratch_mib << 20,
        upload_batch_bytes: 256 << 20,
        tlas_limit: limit.map(|m| (m, camera.org)),
        ..SceneOptions::default()
    };
    let cpu = load_scene(&src, &opts)?;
    let scene = &cpu.scene;
    let st = &scene.stats;
    rep.line(format!(
        "scene: {} loaded in {:.1} s: {} meshes, {} instances ({} world kept of {}, {} entities kept of {}), {:.2} M unique / {:.2} M instanced triangles",
        st.bwm,
        st.load_seconds,
        scene.meshes.len(),
        scene.instances.len(),
        st.world_instances_kept,
        st.world_instances,
        st.entities_kept,
        st.entities_requested,
        scene.unique_triangles() as f64 / 1e6,
        scene.instanced_triangles() as f64 / 1e6
    ));
    rep.line(format!(
        "  materials: {} traced ({} layered, layer blending evaluated), {} textures (<= {} texels); alpha-tested (NON_OPAQUE) {:.2} M instanced triangles; decals {} instances, {:.3} M triangles; triangles without UVs {}",
        st.materials_traced,
        st.materials_layered,
        scene.textures.len(),
        args.tex_size,
        st.alpha_tested_triangles_instanced as f64 / 1e6,
        st.decal_instances,
        st.decal_triangles_instanced as f64 / 1e6,
        st.triangles_without_uv
    ));
    for (reason, e) in &st.excluded {
        rep.line(format!(
            "  excluded {reason}: {} ({:.2} M triangles)",
            e.count,
            e.triangles as f64 / 1e6
        ));
    }
    for (feature, n) in &st.ignored_features {
        rep.line(format!(
            "  ignored material feature ({n} materials): {feature}"
        ));
    }
    let mut kinds = [0usize; 5];
    for l in &snap_lights {
        kinds[(l.kind as usize).min(4)] += 1;
    }
    let grid = &cpu.grid;
    rep.line(format!(
        "lights: {} in light_parms.tsv (point {}, half-point {}, spot {}, area {} skipped, parallel {} = sun), {} local lights sampled; grid {}x{} cells of {:.1} m, {} list entries, max {} per cell; RIS M = {}",
        snap_lights.len(),
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        kinds[4],
        cpu.lights.len(),
        grid.dims[0],
        grid.dims[1],
        grid.cell,
        grid.indices.len(),
        grid.max_per_cell(),
        args.ris
    ));
    rep.line(format!(
        "sun: theta {} phi {} dir {:?}, irradiance {:?} (sunColor x {SUN_COLOR_SCALE:.2}), disc radius {} deg; sky: lerp(zenith {:?}, horizon {:?}) x {} over the upper hemisphere",
        sun.theta_deg,
        sun.phi_deg,
        sun.dir,
        sun.color.map(|c| c * SUN_COLOR_SCALE),
        args.sun_radius,
        env.sky_zenith,
        env.sky_horizon,
        args.sky_scale
    ));
    let mut formats: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for t in &scene.textures {
        let e = formats.entry(format!("{:?}", t.format)).or_default();
        e.0 += 1;
        e.1 += t.mips.iter().map(|m| m.data.len() as u64).sum::<u64>();
    }
    rep.line(format!(
        "  texture formats: {}",
        formats
            .iter()
            .map(|(f, (n, b))| format!("{f} {n} ({:.1} MiB)", *b as f64 / MIB))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    // Material table for the debug output (before `upload` consumes the CPU scene).
    let aov_materials = args.aov.then(|| {
        let mut s = String::from("id\tname\tslots (k=image:format, 0-2 diffuse, 3-5 bump, 6-8 gloss, 9-11 metallic, 12-14 mask, 15 occlusion)\n");
        for (i, (m, g)) in scene.materials.iter().zip(&scene.gpu_materials).enumerate() {
            s.push_str(&format!("{}\t{}\t", i + 1, m.name));
            for (k, &slot) in g[84..100].iter().enumerate() {
                if let Some(t) = slot.checked_sub(1).and_then(|j| scene.textures.get(j as usize)) {
                    s.push_str(&format!("{k}={}:{:?} ", t.name, t.format));
                }
            }
            s.push('\n');
        }
        s
    });

    // ---------------------------------------------------------------- upload (BVH, tables)
    let gs = Arc::new(upload(&mut gpu, cpu)?);
    let b = gs.blas_stats();
    let ts = gs.full_tlas().stats.clone();
    if let Some(m) = limit {
        rep.line(format!(
            "TLAS limited to the {m} instances nearest the camera"
        ));
    }
    rep.line(format!(
        "BVH: {} BLAS compacted {:.1} MiB (build {:.1} ms GPU), TLAS {} instances {:.1} MiB + scratch {:.1} MiB + descs {:.1} MiB",
        b.count,
        b.compacted_bytes as f64 / MIB,
        b.build_gpu_ms + b.compact_gpu_ms,
        ts.instances,
        ts.result_bytes as f64 / MIB,
        ts.scratch_bytes.max(ts.update_scratch_bytes) as f64 / MIB,
        ts.instance_bytes as f64 / MIB
    ));
    let sg = gs.tables();
    rep.line(format!(
        "upload: {} texture slots {:.1} MiB in {} submits ({} rejected), tables {:.1} MiB (records {:.1}, materials {:.2}), {:.1} s total (BVH + tables + kernels)",
        sg.textures.count,
        sg.textures.bytes as f64 / MIB,
        sg.textures.upload_submits,
        sg.textures.rejected.len(),
        sg.table_bytes() as f64 / MIB,
        sg.records.size as f64 / MIB,
        sg.materials.size as f64 / MIB,
        gs.upload_seconds()
    ));
    for (name, why) in sg.textures.rejected.iter().take(10) {
        rep.line(format!("  rejected texture {name}: {why}"));
    }

    let spp = args.spp.max(1);
    let settings = PhotoSettings {
        spp_target: 0,
        bounces: args.bounces,
        rr_start: args.rr_start,
        clamp: args.clamp,
        ris_m: args.ris,
        sky_scale: args.sky_scale,
        sun_radius_deg: args.sun_radius,
        flags: disable,
        znear,
        tile_pixels: if args.tile_pixels == 0 {
            w * h
        } else {
            args.tile_pixels
        },
        denoise_iters: args.denoise_iters,
    };
    let mut session = PhotoSession::new(&mut gpu, gs.clone(), camera, w, h, settings)?;
    let n = (w * h) as usize;
    let (tile_rows, tiles) = session.tiling();
    let vram_total = gpu.vram_usage().saturating_sub(vram0);
    rep.line(format!(
        "VRAM: textures {:.1} MiB, BLAS {:.1} MiB, TLAS (+scratch, descs) {:.1} MiB, tables {:.1} MiB, accumulation/denoise buffers {:.1} MiB; DXGI process usage {:.1} MiB",
        sg.textures.bytes as f64 / MIB,
        b.compacted_bytes as f64 / MIB,
        (ts.result_bytes + ts.scratch_bytes.max(ts.update_scratch_bytes) + ts.instance_bytes)
            as f64
            / MIB,
        sg.table_bytes() as f64 / MIB,
        session.accum().bytes() as f64 / MIB,
        vram_total as f64 / MIB
    ));
    if let Some(table) = &aov_materials {
        write_aov(&mut gpu, &gs, &camera, w, h, &settings, &args.out, table)?;
        rep.line(format!(
            "aov: photo_aov written to {} (aov_*.bin, aov_albedo.png, aov_materials.tsv)",
            args.out.display()
        ));
    }

    // ---------------------------------------------------------------- reference
    let basis = camera.basis(w, h);
    let ref_size = parse_size(&args.ref_size)?;
    let game_color = args
        .ref_color
        .as_deref()
        .map(|p| -> Result<Vec<[f32; 3]>, String> {
            let c = read_rgba16f(p)?;
            if c.len() != (ref_size.0 * ref_size.1) as usize {
                return Err(format!(
                    "{}: {} pixels, expected {:?}",
                    p.display(),
                    c.len(),
                    ref_size
                ));
            }
            Ok(if (w, h) == ref_size {
                c
            } else {
                basis.resample(&c, ref_size)
            })
        })
        .transpose()?;
    let game_depth = args
        .ref_depth
        .as_deref()
        .map(|p| -> Result<Vec<f32>, String> {
            let d = read_f32(p)?;
            if d.len() != (ref_size.0 * ref_size.1) as usize {
                return Err(format!(
                    "{}: {} values, expected {:?}",
                    p.display(),
                    d.len(),
                    ref_size
                ));
            }
            Ok(if (w, h) == ref_size {
                d
            } else {
                basis.resample(&d, ref_size)
            })
        })
        .transpose()?;
    let scale = args.dump_scale;
    let ours_stored = |c: [f32; 3]| c.map(|v| v * scale);

    // ---------------------------------------------------------------- sampling
    let mut save_at: Vec<u32> = args
        .save_at
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|&s| s > 0 && s < spp)
        .collect();
    save_at.push(spp);
    save_at.sort_unstable();
    save_at.dedup();
    // (spp, raw luminance, denoised luminance) for convergence.
    let mut stages: Vec<(u32, Vec<f32>, Vec<f32>, f64)> = Vec::new();
    let mut exposure: Option<f32> = game_color
        .as_ref()
        .map(|g| auto_exposure(g.iter().copied()));
    let t_sampling = Instant::now();
    let mut wall_sampling = 0.0;
    let mut final_den: Vec<[f32; 4]> = Vec::new();
    for &target in &save_at {
        let t_chunk = Instant::now();
        session.set_spp_target(target);
        // Offline: wait for every submit (the game's host polls instead).
        while !session.step(&mut gpu, args.submit_ms as f32)?.done {
            session.finish(&mut gpu)?;
        }
        wall_sampling += t_chunk.elapsed().as_secs_f64();
        let done = session.spp_done();
        let rad = session.read_hdr(&mut gpu)?;
        let (den, den_ms) = session.denoise(&mut gpu)?;
        let rgb: Vec<[f32; 3]> = rad.iter().map(|c| [c[0], c[1], c[2]]).collect();
        let e = *exposure.get_or_insert_with(|| auto_exposure(rgb.iter().map(|&c| ours_stored(c))));
        write_png(
            &out(&format!("photo_{done:04}spp.png")),
            w,
            h,
            &rgb_image(n, |i| tonemap(ours_stored(rgb[i]), e)),
        )?;
        write_png(
            &out(&format!("photo_{done:04}spp_denoised.png")),
            w,
            h,
            &rgb_image(n, |i| {
                tonemap(ours_stored([den[i][0], den[i][1], den[i][2]]), e)
            }),
        )?;
        stages.push((
            done,
            rgb.iter().map(|&c| lum(c)).collect(),
            den.iter().map(|c| lum([c[0], c[1], c[2]])).collect(),
            den_ms,
        ));
        if done == spp {
            final_den = den;
        }
    }
    let total_wall = t_sampling.elapsed().as_secs_f64();
    let per_sample = session.sample_ms().to_vec();
    let (submits, max_submit_ms) = session.submit_stats();
    let mut sorted = per_sample.clone();
    let med = median(&mut sorted);
    rep.line(format!(
        "--- sampling: {spp} spp at {w}x{h}, {} bounces (RR from bounce {}), NEE sun + sky (MIS with the BSDF) + 1 RIS local light, indirect clamp {}",
        args.bounces, args.rr_start, args.clamp
    ));
    rep.line(format!(
        "  GPU per sample: median {med:.2} ms, min {:.2}, max {:.2}; total GPU {:.2} s; {tiles} tiles of {tile_rows} rows per sample; {submits} submits (max {:.0} ms GPU per submit); depth pass {:.2} ms; wall {:.2} s sampling (+ readback/denoise/PNG {:.2} s)",
        sorted.first().copied().unwrap_or(0.0),
        sorted.last().copied().unwrap_or(0.0),
        per_sample.iter().sum::<f64>() / 1e3,
        max_submit_ms,
        session.depth_ms(),
        wall_sampling,
        total_wall - wall_sampling
    ));
    let (_, ref_lum, _, _) = stages.last().ok_or("no samples")?;
    let ref_lum = ref_lum.clone();
    for (s, raw, den, den_ms) in &stages {
        rep.line(format!(
            "  {s:5} spp: relative RMSE vs {spp} spp: raw {:.4}, denoised {:.4} (denoiser {} a-trous passes, {den_ms:.2} ms GPU)",
            rel_rmse(raw, &ref_lum),
            rel_rmse(den, &ref_lum),
            args.denoise_iters
        ));
    }
    session.write_hdr(
        &mut gpu,
        &out("photo_hdr_rgba32f.bin"),
        &format!("multiply by {scale} for the game's stored colour"),
    )?;
    let rgb: Vec<[f32; 3]> = session
        .read_hdr(&mut gpu)?
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let den_rgb: Vec<[f32; 3]> = final_den.iter().map(|c| [c[0], c[1], c[2]]).collect();
    write_hdr(
        &out("photo_denoised_hdr_rgba32f.bin"),
        w,
        h,
        &den_rgb,
        &format!(
            "{spp} spp denoised ({} a-trous passes), engine absolute units",
            args.denoise_iters
        ),
    )?;

    // ---------------------------------------------------------------- validation
    let (alb, nrm) = session.read_features(&mut gpu)?;
    let e = exposure.unwrap_or(1.0);
    rep.line(format!(
        "tonemap (all PNGs): ours x {scale} (game stored units), exposure {e:.4} (median luminance of {} -> 0.18), per-channel Reinhard x/(1+x), sRGB; not the engine's curve (env/tonemapping not recovered)",
        if game_color.is_some() { "the game frame" } else { "our image" }
    ));
    if let (Some(gc), Some(gd)) = (&game_color, &game_depth) {
        let mut side = Vec::with_capacity(n * 6);
        for y in 0..h as usize {
            for x in 0..w as usize {
                side.extend_from_slice(&tonemap(ours_stored(rgb[y * w as usize + x]), e));
            }
            for x in 0..w as usize {
                side.extend_from_slice(&tonemap(gc[y * w as usize + x], e));
            }
        }
        write_png(&out("photo_vs_game.png"), 2 * w, h, &side)?;
        let mut regions = [
            Region::new("all depth-matched"),
            Region::new("sunlit floor (N.z > 0.7)"),
            Region::new("shaded floor (N.z > 0.7)"),
            Region::new("sunlit wall (|N.z| < 0.3)"),
            Region::new("shaded wall (|N.z| < 0.3)"),
            Region::new("sky (both miss)"),
        ];
        let mut matched = 0usize;
        let mut ratio_px = vec![None; n];
        // Every game sky pixel (d = 0), whatever ours shows there.
        let (mut game_sky_n, mut game_sky_sum, mut game_sky_ours_hit) = (0usize, 0f64, 0usize);
        for i in 0..n {
            let ours = lum(ours_stored(rgb[i]));
            let game = lum(gc[i]);
            let z_ours = nrm[i][3];
            if gd[i] <= 0.0 {
                game_sky_n += 1;
                game_sky_sum += f64::from(game);
                game_sky_ours_hit += usize::from(z_ours > 0.0);
            }
            if gd[i] <= 0.0 && z_ours <= 0.0 {
                regions[5].add(game, ours);
                ratio_px[i] = Some(game / ours);
                continue;
            }
            if gd[i] <= 0.0 || z_ours <= 0.0 {
                continue;
            }
            let z_game = znear / gd[i];
            if (z_ours / z_game - 1.0).abs() > 0.01 {
                continue;
            }
            matched += 1;
            ratio_px[i] = Some(game / ours);
            regions[0].add(game, ours);
            let nl = (nrm[i][0].powi(2) + nrm[i][1].powi(2) + nrm[i][2].powi(2)).sqrt();
            let nz = if nl > 0.0 { nrm[i][2] / nl } else { 0.0 };
            let sun = alb[i][3];
            let lit = sun > 0.9;
            let shaded = sun < 0.1;
            if nz > 0.7 && lit {
                regions[1].add(game, ours);
            } else if nz > 0.7 && shaded {
                regions[2].add(game, ours);
            } else if nz.abs() < 0.3 && lit {
                regions[3].add(game, ours);
            } else if nz.abs() < 0.3 && shaded {
                regions[4].add(game, ours);
            }
        }
        write_png(
            &out("photo_ratio_game_over_ours.png"),
            w,
            h,
            &rgb_image(n, |i| ratio_colour(ratio_px[i])),
        )?;
        rep.line(format!(
            "--- validation vs the game frame ({}), game stored colour vs ours x {scale}; depth-matched (1 %) pixels {:.1} %; regions by our first-hit normal and sun visibility:",
            args.ref_color.as_deref().map_or(String::new(), |p| p.display().to_string()),
            100.0 * matched as f64 / n as f64
        ));
        for r in &mut regions {
            rep.line(r.line(n));
        }
        let sky_l = lum(env.sky_horizon) * args.sky_scale;
        rep.line(format!(
            "  game sky pixels (d = 0): {game_sky_n} ({:.2} %, ours hits geometry at {game_sky_ours_hit}), mean stored luminance {:.3} = {:.3} absolute; our sky model: horizon luminance {sky_l:.3} absolute (lerp(zenith, horizon) x {}), zenith {:.3}",
            100.0 * game_sky_n as f64 / n as f64,
            game_sky_sum / game_sky_n.max(1) as f64,
            game_sky_sum / game_sky_n.max(1) as f64 / f64::from(scale),
            args.sky_scale,
            lum(env.sky_zenith) * args.sky_scale
        ));
    } else {
        rep.line("validation: --ref-color and --ref-depth not both given -> skipped");
    }
    rep.line(format!(
        "outputs: {} (photo_<spp>spp[_denoised].png, photo_vs_game.png, photo_ratio_game_over_ours.png, photo_[denoised_]hdr_rgba32f.bin + .tsv, report.txt)",
        args.out.display()
    ));
    rep.line(format!(
        "total wall time {:.1} s",
        t_start.elapsed().as_secs_f64()
    ));
    std::fs::write(out("report.txt"), &rep.0).map_err(|e| e.to_string())?;
    Ok(())
}
