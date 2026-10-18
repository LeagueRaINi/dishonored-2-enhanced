//! Ray-tracing feasibility spike (docs/raytracing.md, STATUS E070). Offline: reads the
//! game's archives (explicit install path), never runs or touches the game.
//!
//! 1. D3D12 device: adapter, driver, raytracing tier, shader model, inline RT support.
//! 2. Scene: `.bwm` world + static entities -> one compacted PREFER_FAST_TRACE BLAS per
//!    object, TLAS over all instances and over a camera-radius cut-off (Far Cry 6 style);
//!    build/compaction/rebuild/refit timings and memory.
//! 3. Rays (inline RayQuery CS) at 2304x968 and 3440x1440: primary rays validated against a
//!    game depth dump; one reflection ray per pixel from the game depth (mirror, GGX,
//!    floors-only, sun-shadowed, cut-off, hybrid masks) with a screen-space reference.
//! 4. Photo-mode demo: progressive 1 spp/frame diffuse path tracing (sun + sky).
//! 5. Gates (BVH < 0.5 GB, reflections < 1 ms at 0.67 scale, SSR agreement > 90 %).
//!
//! Example (from `void-rs/`):
//! cargo run --release -p void-rt --example rt_spike -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 \
//!   --ref-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
//!   --ref-color local/ref/RX010/dump1_color_rgba16f_3440x1440.bin --ref-size 3440x1440

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use void_rt::accel::{self, BlasSet, Tlas, median};
use void_rt::camera::{Camera, RayBasis};
use void_rt::dxc::Dxc;
use void_rt::gpu::{Buffer, CONSTANTS, DebugLayer, Entry, Gpu, Layout, Programs, SRVS, UAVS};
use void_rt::image::{auto_exposure, distance_colour, id_colour, tonemap, write_png};
use void_rt::run::{Report, dxc_dir, parse_size, read_f32, read_rgba16f, run_log};
use void_rt::scene::{self, EntitySource, LoadOptions, Origin};
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
#[command(about = "Offline DXR 1.1 inline ray-tracing spike over a decoded Dishonored 2 map")]
struct Args {
    /// Game installation root (the directory holding `base/`).
    #[arg(long)]
    game: PathBuf,
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    /// F3 snapshot directory (camera, sun, live entities).
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Entities::Auto)]
    entities: Entities,
    /// Camera pose `x,y,z,yaw,pitch` when no snapshot is given (benchmark scene 0).
    #[arg(
        long,
        default_value = "-33.65,181.69,10.46,-49.25,13.72",
        allow_hyphen_values = true
    )]
    pose: String,
    /// Vertical field of view (radians) for `--pose` (RX010 `fov_y`).
    #[arg(long, default_value_t = 1.024_779)]
    fov_y: f32,
    /// Game depth dump (`r32f`, near / z) matching the snapshot camera.
    #[arg(long)]
    ref_depth: Option<PathBuf>,
    /// Game colour dump (`rgba16f`) of the same frame (image only).
    #[arg(long)]
    ref_color: Option<PathBuf>,
    /// Size of the reference dumps, `WxH`.
    #[arg(long, default_value = "3440x1440")]
    ref_size: String,
    #[arg(long, default_value = "local/rt")]
    out: PathBuf,
    /// Directory with dxcompiler.dll + dxil.dll (default: env VOID_DXC_DIR, then the newest
    /// Windows SDK bin/x64).
    #[arg(long)]
    dxc: Option<PathBuf>,
    /// BVH radius around the camera for the cut-off variant (metres).
    #[arg(long, default_value_t = 40.0)]
    cutoff: f32,
    /// Timed repetitions per measurement (median reported).
    #[arg(long, default_value_t = 20)]
    reps: u32,
    #[arg(long, default_value_t = 256)]
    gi_frames: u32,
    #[arg(long, default_value_t = 2)]
    gi_bounces: u32,
    #[arg(long, default_value = "3440x1440")]
    gi_size: String,
    /// BLAS scratch arena (MiB).
    #[arg(long, default_value_t = 256)]
    scratch_mib: u64,
    /// D3D12 debug layer + GPU-based validation (messages printed; errors abort). Needs an
    /// Agility SDK redistributable directory (`--agility`).
    #[arg(long)]
    debug_layer: bool,
    /// Directory with the Agility SDK's `D3D12Core.dll` + `d3d12SDKLayers.dll` (NuGet
    /// `Microsoft.Direct3D.D3D12`, `build/native/bin/x64`); env VOID_AGILITY_DIR.
    #[arg(long, env = "VOID_AGILITY_DIR")]
    agility: Option<PathBuf>,
    /// The Agility SDK version number of `--agility` (1.619.6 -> 619).
    #[arg(long, default_value_t = 619)]
    agility_version: u32,
    /// Use WARP (software D3D12, DXR 1.1 capable) instead of the GPU: driver-independent,
    /// so a faulty kernel cannot reset the GPU driver (the game may share the GPU).
    #[arg(long)]
    warp: bool,
    /// Validation run: WARP, tiny resolutions (64x27, 96x40), 4 GI frames, 2 repetitions,
    /// at most `--max-instances` TLAS instances (default 2000); add `--debug-layer` where
    /// the D3D12 SDK layers are available.
    /// Run this after every kernel change before a full-size run.
    #[arg(long)]
    smoke: bool,
    /// Limit the TLAS to the instances nearest the camera (smoke default 2000).
    #[arg(long)]
    max_instances: Option<usize>,
    /// Override the two ray resolutions, `WxH,WxH` (default 2304x968,3440x1440; smoke
    /// 64x27,96x40).
    #[arg(long)]
    sizes: Option<String>,
    /// Run log (start/end of every run, to match a driver reset to a run).
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

const F_SHADOW: u32 = 1;
const F_FLOORS: u32 = 2;
const F_GGX: u32 = 4;
const F_JITTER: u32 = 8;
const F_MASK_SSR: u32 = 16;
const F_MASK_DEPTH: u32 = 32;
const CULL_BACK: u32 = 0x10;
const CULL_FRONT: u32 = 0x20;
const MIB: f64 = 1024.0 * 1024.0;

/// Parameters shared by all kernels (layout of `Consts` in rt.hlsl).
#[derive(Clone, Copy)]
struct Frame {
    basis: RayBasis,
    znear: f32,
    sun_dir: [f32; 3],
    sun_rad: [f32; 3],
    sky_rad: [f32; 3],
    /// Sizes of the primitive offset table and primitive buffer (shader-side clamps).
    geom_count: u32,
    prim_count: u32,
}

#[derive(Clone, Copy, Default)]
struct Params {
    frame: u32,
    roughness: f32,
    flags: u32,
    ray_flags: u32,
    t_max: f32,
    bounces: u32,
}

fn constants(f: &Frame, p: &Params) -> [u32; CONSTANTS] {
    let b = &f.basis;
    let v = |a: [f32; 3]| a.map(f32::to_bits);
    let mut c = [0u32; CONSTANTS];
    c[0..3].copy_from_slice(&v(b.org));
    c[3] = b.width;
    c[4..7].copy_from_slice(&v(b.fwd));
    c[7] = b.height;
    c[8..11].copy_from_slice(&v(b.right));
    c[11] = b.tan_x.to_bits();
    c[12..15].copy_from_slice(&v(b.up));
    c[15] = b.tan_y.to_bits();
    c[16..19].copy_from_slice(&v(f.sun_dir));
    c[19] = f.znear.to_bits();
    c[20..23].copy_from_slice(&v(f.sun_rad));
    c[23] = p.frame;
    c[24..27].copy_from_slice(&v(f.sky_rad));
    c[27] = p.roughness.to_bits();
    c[28] = p.flags;
    c[29] = p.ray_flags;
    c[30] = p.t_max.to_bits();
    c[31] = p.bounces;
    c[32] = f.geom_count;
    c[33] = f.prim_count;
    c
}

/// Output buffers for one resolution.
struct Targets {
    a: Buffer,
    id: Buffer,
    color: Buffer,
    aux: Buffer,
    ssr: Buffer,
    pixels: usize,
}

impl Targets {
    fn new(gpu: &Gpu, w: u32, h: u32) -> Result<Self, String> {
        let n = u64::from(w) * u64::from(h);
        Ok(Self {
            a: gpu.buffer(n * 4, true)?,
            id: gpu.buffer(n * 4, true)?,
            color: gpu.buffer(n * 16, true)?,
            aux: gpu.buffer(n * 16, true)?,
            ssr: gpu.buffer(n * 4, true)?,
            pixels: n as usize,
        })
    }

    fn uavs(&self) -> [Option<&Buffer>; UAVS] {
        [
            Some(&self.a),
            Some(&self.id),
            Some(&self.color),
            Some(&self.aux),
        ]
    }
}

/// Kernel entry points with the per-pixel buffer sizes they need (checked per dispatch).
fn entries() -> [Entry<'static>; 4] {
    const SCENE: &[usize] = &[0, 1, 2];
    [
        Entry {
            name: "primary",
            required_srvs: SCENE,
            srv_bytes_per_pixel: &[0; SRVS],
            uav_bytes_per_pixel: &[4, 4, 16, 0],
        },
        Entry {
            name: "reflect_rays",
            required_srvs: SCENE,
            srv_bytes_per_pixel: &[0, 0, 0, 4, 4],
            uav_bytes_per_pixel: &[4, 4, 16, 0],
        },
        Entry {
            name: "ssr_ref",
            required_srvs: &[],
            srv_bytes_per_pixel: &[0, 0, 0, 4, 0],
            uav_bytes_per_pixel: &[4, 0, 0, 0],
        },
        Entry {
            name: "gi",
            required_srvs: SCENE,
            srv_bytes_per_pixel: &[0; SRVS],
            uav_bytes_per_pixel: &[0, 0, 0, 16],
        },
    ]
}

/// Dispatches per command list (submit + wait in between): keeps every submission far
/// below the 2 s TDR limit.
const DISPATCHES_PER_SUBMIT: u32 = 8;

struct Ctx {
    gpu: Gpu,
    programs: Programs,
    prim_offsets: Buffer,
    prims: Buffer,
    reps: u32,
}

impl Ctx {
    /// Runs `entry` 3x (warm-up) then `reps` times with timestamps, at most
    /// `DISPATCHES_PER_SUBMIT` dispatches per submit; returns the median ms.
    #[allow(clippy::too_many_arguments)]
    fn timed(
        &mut self,
        entry: &str,
        frame: &Frame,
        p: &Params,
        tlas: &Tlas,
        depth: Option<&Buffer>,
        ssr_in: Option<&Buffer>,
        uav: [Option<&Buffer>; UAVS],
    ) -> Result<f64, String> {
        let c = constants(frame, p);
        let srv = [
            Some(&tlas.buffer),
            Some(&self.prim_offsets),
            Some(&self.prims),
            depth,
            ssr_in,
        ];
        let size = (frame.basis.width, frame.basis.height);
        for _ in 0..3 {
            self.programs
                .dispatch(&self.gpu, entry, &c, &srv, &uav, None, size)?;
            self.gpu.uav_barrier();
        }
        self.gpu.submit()?;
        let mut ms = Vec::with_capacity(self.reps as usize);
        let mut done = 0;
        while done < self.reps {
            let chunk = (self.reps - done).min(DISPATCHES_PER_SUBMIT);
            for k in 0..chunk {
                self.gpu.timestamp(2 * k);
                self.programs
                    .dispatch(&self.gpu, entry, &c, &srv, &uav, None, size)?;
                self.gpu.uav_barrier();
                self.gpu.timestamp(2 * k + 1);
            }
            let ts = self.gpu.timestamps(2 * chunk)?;
            ms.extend(
                ts.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|t| self.gpu.ticks_to_ms(t[0], t[1])),
            );
            done += chunk;
        }
        Ok(median(&mut ms))
    }
}

/// Depth agreement of ours (game convention) vs the reference at the same pixels.
#[derive(Default, Clone, Copy)]
struct DepthStats {
    n: usize,
    within: usize,
    game_closer: usize,
    ours_closer: usize,
    game_sky: usize,
    ours_miss: usize,
}

impl DepthStats {
    fn of(ours: &[f32], game: &[f32]) -> Self {
        let mut s = Self::default();
        for (&o, &g) in ours.iter().zip(game) {
            s.n += 1;
            match (g > 0.0, o > 0.0) {
                (true, true) => {
                    // d = near / z: g / o = z_ours / z_game.
                    let r = (g / o - 1.0).abs();
                    if r <= 0.01 {
                        s.within += 1;
                    } else if g > o {
                        s.game_closer += 1;
                    } else {
                        s.ours_closer += 1;
                    }
                }
                (false, true) => s.game_sky += 1,
                (true, false) => s.ours_miss += 1,
                _ => s.within += 1,
            }
        }
        s
    }

    fn pct(&self, k: usize) -> f64 {
        100.0 * k as f64 / self.n.max(1) as f64
    }

    fn line(&self) -> String {
        format!(
            "within 1 % {:.1} %, game closer {:.1} %, ours closer {:.1} %, game sky/ours hit {:.2} %, game hit/ours miss {:.2} %",
            self.pct(self.within),
            self.pct(self.game_closer),
            self.pct(self.ours_closer),
            self.pct(self.game_sky),
            self.pct(self.ours_miss)
        )
    }
}

/// Reflection statistics against the screen-space reference.
#[derive(Default)]
struct ReflStats {
    surface: usize,
    traced: usize,
    rt_hit: usize,
    ssr_hit: usize,
    agree: [usize; 3],
    ssr_beyond_cutoff: usize,
    /// Where SS hit but RT disagrees (5 % tol): RT missed / hit nearer / hit farther.
    rt_miss: usize,
    rt_nearer: usize,
    rt_farther: usize,
    /// RT hits closer than 5 cm (origin under our geometry).
    rt_self: usize,
    /// Same, restricted to pixels whose primary ray matches the game depth within 1 %
    /// (our geometry agrees with the game at the ray origin).
    consistent_ssr_hit: usize,
    consistent_agree: usize,
}

const AGREE_TOL: [f32; 3] = [0.02, 0.05, 0.10];

fn refl_stats(rt: &[f32], ssr: &[f32], consistent: &[bool], cutoff: Option<f32>) -> ReflStats {
    let mut s = ReflStats::default();
    for ((&r, &q), &ok) in rt.iter().zip(ssr).zip(consistent) {
        if r == 0.0 {
            continue;
        }
        s.surface += 1;
        if r != -2.0 {
            s.traced += 1;
        }
        if r > 0.0 {
            s.rt_hit += 1;
            if r < 0.05 {
                s.rt_self += 1;
            }
        }
        if q > 0.0 {
            if cutoff.is_some_and(|c| q > c) {
                s.ssr_beyond_cutoff += 1;
                continue;
            }
            s.ssr_hit += 1;
            for (k, tol) in AGREE_TOL.iter().enumerate() {
                if r > 0.0 && (r - q).abs() <= (tol * q).max(0.1) {
                    s.agree[k] += 1;
                }
            }
            let agree = r > 0.0 && (r - q).abs() <= (0.05 * q).max(0.1);
            if !agree {
                if r <= 0.0 {
                    s.rt_miss += 1;
                } else if r < q {
                    s.rt_nearer += 1;
                } else {
                    s.rt_farther += 1;
                }
            }
            if ok {
                s.consistent_ssr_hit += 1;
                s.consistent_agree += usize::from(agree);
            }
        }
    }
    s
}

fn pct(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

fn rgb_image(n: usize, f: impl Fn(usize) -> [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 3);
    for i in 0..n {
        v.extend_from_slice(&f(i));
    }
    v
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rt_spike: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = Args::parse();
    if args.smoke {
        args.warp = true;
        args.reps = args.reps.min(2);
        args.gi_frames = args.gi_frames.min(4);
        args.gi_size = "64x27".into();
        if args.out == Path::new("local/rt") {
            args.out = PathBuf::from("local/rt/smoke");
        }
    }
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let mode = if args.smoke { "smoke" } else { "full" };
    let cmd: Vec<String> = std::env::args().skip(1).collect();
    run_log(
        &args.runs_log,
        &format!(
            "start {mode} run (out {}): {}",
            args.out.display(),
            cmd.join(" ")
        ),
    );
    let r = spike(&args);
    run_log(
        &args.runs_log,
        &match &r {
            Ok(()) => format!("end {mode} run: ok"),
            Err(e) => format!("end {mode} run: error: {}", e.lines().next().unwrap_or("")),
        },
    );
    r
}

#[allow(clippy::too_many_lines)]
fn spike(args: &Args) -> Result<(), String> {
    let t_start = Instant::now();
    let mut rep = Report::default();
    let out = |n: &str| args.out.join(n);

    // ---------------------------------------------------------------- 1. device
    let debug = if args.debug_layer {
        let dir = args
            .agility
            .as_deref()
            .ok_or("--debug-layer/--smoke need --agility <dir with D3D12Core.dll + d3d12SDKLayers.dll> (Windows Graphics Tools are not installed)")?;
        Some(DebugLayer {
            sdk_version: args.agility_version,
            sdk_dir: dir,
        })
    } else {
        None
    };
    let mut gpu = Gpu::new(debug, args.warp)?;
    if args.debug_layer {
        rep.line("D3D12 debug layer + GPU-based validation: on");
    }
    if args.warp {
        rep.line("device: WARP (software; timings are not GPU timings)");
    }
    let info = gpu.info.clone();
    rep.line(format!(
        "adapter: {} (vendor {:#06x}, device {:#06x}, {} MiB), driver {}",
        info.name, info.vendor_id, info.device_id, info.vram_mib, info.driver
    ));
    rep.line(format!(
        "D3D12: raytracing tier {}, highest shader model {}, inline ray tracing (DXR 1.1 RayQuery): {}",
        info.raytracing_tier_name(),
        info.shader_model_name(),
        if info.inline_raytracing() { "yes" } else { "no" }
    ));
    if !info.inline_raytracing() {
        return Err("inline ray tracing (tier 1.1, SM 6.5) not supported".into());
    }
    let dxc = Dxc::load(&dxc_dir(args.dxc.as_deref())?)?;
    let t = Instant::now();
    let programs = Programs::new(
        &gpu,
        &dxc,
        void_rt::SHADER_SOURCE,
        Layout::SPIKE,
        &entries(),
    )?;
    rep.line(format!(
        "shaders: cs_6_5 via {} in {:.0} ms",
        dxc.dir.display(),
        t.elapsed().as_secs_f64() * 1e3
    ));

    // ---------------------------------------------------------------- 2. scene
    let snapshot = args.snapshot.as_deref().map(Snapshot::load).transpose()?;
    let (camera, znear, sun) = match &snapshot {
        Some(s) => (s.camera, s.znear, s.sun),
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
                None,
            )
        }
    };
    // RX010 sun (sun.tsv of snap_1792183388_f1288) when the snapshot has none.
    let sun =
        sun.unwrap_or_else(|| Sun::from_angles(26.0, 100.0, [0.215_861, 0.171_441, 0.116_971]));
    let entities = match (args.entities, &snapshot) {
        (Entities::None, _) => EntitySource::None,
        (Entities::File, _) | (Entities::Auto, None) => EntitySource::File,
        (Entities::Snapshot | Entities::Auto, Some(s)) => {
            EntitySource::Snapshot(s.entities.clone())
        }
        (Entities::Snapshot, None) => return Err("--entities snapshot needs --snapshot".into()),
    };
    let entity_source = match &entities {
        EntitySource::None => "none".to_owned(),
        EntitySource::File => format!("{}.entities", args.map),
        EntitySource::Snapshot(v) => {
            format!("snapshot models.tsv ({} static drawable rows)", v.len())
        }
    };
    rep.line(format!(
        "camera: org {:?} forward {:?} fov_y {:.4} znear {} ({})",
        camera.org,
        camera.axis[0],
        camera.fov_y,
        znear,
        args.snapshot
            .as_ref()
            .map_or("synthetic pose".to_owned(), |p| p.display().to_string())
    ));
    if let Some(s) = &snapshot {
        let f = &s.filter;
        rep.line(format!(
            "snapshot models.tsv: {} rows, skinned {}, not mayBeDrawn {}, editor/collision {}, no model {}",
            f.rows, f.skinned, f.not_drawn_flag, f.editor_material, f.no_model
        ));
    }
    let scene = scene::load(&args.game, &args.map, entities, LoadOptions::default())?;
    let st = &scene.stats;
    rep.line(format!(
        "scene: {} + entities from {entity_source}, loaded in {:.1} s",
        st.bwm, st.load_seconds
    ));
    rep.line(format!(
        "  world: {} surfaces, {} batches, {} instances, kept {} ({} surfaces drawn with >1 material -> 1 BLAS each)",
        st.world_surfaces, st.world_batches, st.world_instances, st.world_instances_kept, st.world_surfaces_multi_material
    ));
    rep.line(format!(
        "  entities: {} requested, kept {}, skinned (no decoder) {}, model unavailable {}, runtime material count mismatch {}",
        st.entities_requested, st.entities_kept, st.entities_skinned, st.entities_without_model, st.entity_runtime_material_mismatch
    ));
    rep.line(format!(
        "  traced: {} meshes (BLAS), {} instances, {:.2} M unique triangles, {:.2} M instanced triangles, {:.2} M vertices; alpha-tested (treated opaque) {:.2} M instanced triangles",
        scene.meshes.len(),
        scene.instances.len(),
        scene.unique_triangles() as f64 / 1e6,
        scene.instanced_triangles() as f64 / 1e6,
        scene.positions.len() as f64 / 1e6,
        st.alpha_tested_triangles_instanced as f64 / 1e6
    ));
    rep.line(format!(
        "  materials traced: {} ({} with texture albedo)",
        st.materials_traced, st.materials_with_texture_albedo
    ));
    for (reason, e) in &st.excluded {
        rep.line(format!(
            "  excluded {reason}: {} ({:.2} M triangles)",
            e.count,
            e.triangles as f64 / 1e6
        ));
    }

    // ---------------------------------------------------------------- BLAS / TLAS
    let vram0 = gpu.vram_usage();
    let pos = gpu.upload(&scene.positions, false)?;
    let idx = gpu.upload(&scene.indices, false)?;
    let geometry_bytes = pos.size + idx.size;
    let blas: BlasSet = accel::build_blas(&mut gpu, &scene, &pos, &idx, args.scratch_mib << 20)?;
    let vram_peak_build = gpu.vram_usage();
    drop((pos, idx));
    let b = &blas.stats;
    rep.line(format!(
        "BLAS: {} ({} geometries, {:.2} M triangles), build {:.1} ms GPU in {} batches (scratch arena {:.0} MiB; all-at-once scratch would be {:.0} MiB), compaction {:.1} ms GPU, wall {:.0} ms incl. prebuild queries and readback",
        b.count, b.geometries, b.triangles as f64 / 1e6, b.build_gpu_ms, b.batches,
        b.scratch_arena_bytes as f64 / MIB, b.scratch_total_bytes as f64 / MIB, b.compact_gpu_ms, b.wall_ms
    ));
    rep.line(format!(
        "  memory: {:.1} MiB before compaction -> {:.1} MiB compacted ({:.1} %, {:.1} B/triangle); build inputs (positions + u16 indices, freed after build) {:.1} MiB",
        b.uncompacted_bytes as f64 / MIB,
        b.compacted_bytes as f64 / MIB,
        100.0 * b.compacted_bytes as f64 / b.uncompacted_bytes.max(1) as f64,
        b.compacted_bytes as f64 / b.triangles.max(1) as f64,
        geometry_bytes as f64 / MIB
    ));
    let prim_offsets = gpu.upload(&scene.primitive_offsets, false)?;
    let prims = gpu.upload(&scene.primitives, false)?;
    let shading_bytes = prims.size + prim_offsets.size;
    rep.line(format!(
        "  primitive buffer (FC6 layout, 24 B/triangle): {:.1} MiB + offsets {:.2} MiB",
        prims.size as f64 / MIB,
        prim_offsets.size as f64 / MIB
    ));

    let reps = args.reps.max(1);
    // Optional TLAS limit (smoke runs): the instances nearest to the camera.
    let dist2 = |i: usize| -> f32 {
        let inst = &scene.instances[i];
        let (lo, hi) = inst.world_aabb(&scene.meshes[inst.mesh as usize]);
        (0..3)
            .map(|k| {
                let c = camera.org[k].clamp(lo[k], hi[k]) - camera.org[k];
                c * c
            })
            .sum()
    };
    let limit = args.max_instances.or(args.smoke.then_some(2000));
    let allowed: Vec<bool> = match limit {
        Some(m) if m < scene.instances.len() => {
            let mut order: Vec<(f32, usize)> =
                (0..scene.instances.len()).map(|i| (dist2(i), i)).collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut v = vec![false; scene.instances.len()];
            for &(_, i) in &order[..m] {
                v[i] = true;
            }
            rep.line(format!(
                "TLAS limited to the {m} instances nearest the camera"
            ));
            v
        }
        _ => vec![true; scene.instances.len()],
    };
    let all = accel::instance_descs(&scene, &blas, true, |i| allowed[i]);
    let tlas = accel::build_tlas(&mut gpu, &all, reps)?;
    let r2 = args.cutoff * args.cutoff;
    let near_instances = |i: usize| allowed[i] && dist2(i) <= r2;
    let cut = accel::instance_descs(&scene, &blas, true, near_instances);
    let cut_tris: u64 = (0..scene.instances.len())
        .filter(|&i| near_instances(i))
        .map(|i| scene.meshes[scene.instances[i].mesh as usize].triangles())
        .sum();
    let tlas_cut = accel::build_tlas(&mut gpu, &cut, reps)?;
    for (name, t) in [("full", &tlas), ("cut-off", &tlas_cut)] {
        let s = &t.stats;
        rep.line(format!(
            "TLAS {name}: {} instances, {:.2} MiB (+ scratch {:.2} MiB, update scratch {:.2} MiB, instance descs {:.2} MiB); GPU median of {reps}: rebuild fast-trace {:.3} ms, rebuild fast-build {:.3} ms, refit {:.3} ms",
            s.instances,
            s.result_bytes as f64 / MIB,
            s.scratch_bytes as f64 / MIB,
            s.update_scratch_bytes as f64 / MIB,
            s.instance_bytes as f64 / MIB,
            s.build_fast_trace_ms,
            s.build_fast_build_ms,
            s.update_ms
        ));
    }
    rep.line(format!(
        "  cut-off: radius {} m around the camera (instance AABB test), {:.2} M instanced triangles",
        args.cutoff,
        cut_tris as f64 / 1e6
    ));
    let world_instances = scene
        .instances
        .iter()
        .filter(|i| i.origin == Origin::World)
        .count();
    let ts = &tlas.stats;
    let bvh_bytes = b.compacted_bytes
        + ts.result_bytes
        + ts.scratch_bytes.max(ts.update_scratch_bytes)
        + ts.instance_bytes;
    let rt_bytes =
        bvh_bytes + shading_bytes + tlas_cut.stats.result_bytes + tlas_cut.stats.instance_bytes;
    rep.line(format!(
        "memory: BVH resident (compacted BLAS + full TLAS + its scratch + instance descs) {:.1} MiB; + primitive data + cut-off TLAS = {:.1} MiB; peak during BLAS build (DXGI CurrentUsage delta) {:.1} MiB; after (DXGI delta) {:.1} MiB",
        bvh_bytes as f64 / MIB,
        rt_bytes as f64 / MIB,
        vram_peak_build.saturating_sub(vram0) as f64 / MIB,
        gpu.vram_usage().saturating_sub(vram0) as f64 / MIB
    ));
    rep.line(format!(
        "  instances: world {world_instances}, entity {}",
        scene.instances.len() - world_instances
    ));

    let mut ctx = Ctx {
        gpu,
        programs,
        prim_offsets,
        prims,
        reps,
    };

    // ---------------------------------------------------------------- 3. rays
    let ref_size = parse_size(&args.ref_size)?;
    let ref_depth = args.ref_depth.as_deref().map(read_f32).transpose()?;
    if let Some(d) = &ref_depth
        && d.len() != (ref_size.0 * ref_size.1) as usize
    {
        return Err(format!(
            "reference depth has {} values, expected {}",
            d.len(),
            ref_size.0 * ref_size.1
        ));
    }
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let sun_rad = sun.color.map(|c| 3.0 * c / lum(sun.color).max(1e-6));
    let sky_rad = [0.55f32, 0.70, 0.90].map(|c| 0.35 * c);
    rep.line(format!(
        "lighting (spike, relative units): sun dir {:?} (theta {} phi {}), irradiance {:?}; sky radiance {:?}",
        sun.dir, sun.theta_deg, sun.phi_deg, sun_rad, sky_rad
    ));
    if let Some(c) = &args.ref_color {
        let px = read_rgba16f(c)?;
        let e = auto_exposure(px.iter().copied());
        write_png(
            &out("game_color.png"),
            ref_size.0,
            ref_size.1,
            &rgb_image(px.len(), |i| tonemap(px[i], e)),
        )?;
    }

    let mut gate_refl: Vec<(String, f64)> = Vec::new();
    let mut gate_agree: Vec<(String, f64)> = Vec::new();
    let mut primary_cull = 0u32;
    let resolutions: [(u32, u32); 2] = match &args.sizes {
        Some(s) => {
            let v: Vec<(u32, u32)> = s.split(',').map(parse_size).collect::<Result<_, _>>()?;
            let [a, b] = v[..] else {
                return Err(format!("--sizes {s}: expected two sizes"));
            };
            [a, b]
        }
        None if args.smoke => [(64, 27), (96, 40)],
        None => [(2304, 968), (3440, 1440)],
    };
    let frame0 = Frame {
        basis: camera.basis(1, 1),
        znear,
        sun_dir: sun.dir,
        sun_rad,
        sky_rad,
        geom_count: scene.primitive_offsets.len() as u32,
        prim_count: scene.primitives.len() as u32,
    };
    for (w, h) in resolutions {
        let res = format!("{w}x{h}");
        let frame = Frame {
            basis: camera.basis(w, h),
            ..frame0
        };
        let tg = Targets::new(&ctx.gpu, w, h)?;
        let n = tg.pixels;
        rep.line(format!("--- {res} ({:.2} M pixels)", n as f64 / 1e6));
        let game = ref_depth.as_ref().map(|d| {
            if (w, h) == ref_size {
                d.clone()
            } else {
                frame.basis.resample(d, ref_size)
            }
        });
        // Primary rays, three cull conventions.
        let mut best: Option<(u32, f64)> = None;
        for (name, rf) in [
            ("no culling", 0u32),
            ("cull back", CULL_BACK),
            ("cull front", CULL_FRONT),
        ] {
            let p = Params {
                ray_flags: rf,
                t_max: 1e4,
                ..Default::default()
            };
            let ms = ctx.timed("primary", &frame, &p, &tlas, None, None, tg.uavs())?;
            let mut line = format!("primary ({name}): {ms:.3} ms");
            if let Some(g) = &game {
                let ours: Vec<f32> = ctx.gpu.read(&tg.a, n)?;
                let s = DepthStats::of(&ours, g);
                let _ = write!(line, "; vs game depth: {}", s.line());
                let score = s.pct(s.within);
                if best.is_none_or(|b| score > b.1) {
                    best = Some((rf, score));
                }
            }
            rep.line(line);
        }
        let rf = best.map_or(0, |b| b.0);
        if (w, h) == resolutions[1] {
            primary_cull = rf;
        }
        let p = Params {
            ray_flags: rf,
            t_max: 1e4,
            flags: F_SHADOW,
            ..Default::default()
        };
        let ms = ctx.timed("primary", &frame, &p, &tlas, None, None, tg.uavs())?;
        rep.line(format!(
            "primary + sun shadow ray (ray flags {rf:#x}): {ms:.3} ms"
        ));
        let depth_ours: Vec<f32> = ctx.gpu.read(&tg.a, n)?;
        let ids: Vec<u32> = ctx.gpu.read(&tg.id, n)?;
        let col: Vec<[f32; 4]> = ctx.gpu.read(&tg.color, n)?;
        let e = auto_exposure(col.iter().map(|c| [c[0], c[1], c[2]]));
        write_png(
            &out(&format!("primary_shaded_{res}.png")),
            w,
            h,
            &rgb_image(n, |i| tonemap([col[i][0], col[i][1], col[i][2]], e)),
        )?;
        write_png(
            &out(&format!("primary_distance_{res}.png")),
            w,
            h,
            &rgb_image(n, |i| distance_colour(col[i][3])),
        )?;
        write_png(
            &out(&format!("primary_instance_{res}.png")),
            w,
            h,
            &rgb_image(n, |i| id_colour(ids[i])),
        )?;
        if let Some(g) = &game {
            write_png(
                &out(&format!("primary_vs_game_{res}.png")),
                w,
                h,
                &rgb_image(n, |i| {
                    let (o, gd) = (depth_ours[i], g[i]);
                    match (gd > 0.0, o > 0.0) {
                        (true, true) if (gd / o - 1.0).abs() <= 0.01 => [40, 160, 40],
                        (true, true) if gd > o => [60, 90, 230],
                        (true, true) => [230, 50, 50],
                        (false, true) => [230, 210, 40],
                        (true, false) => [220, 40, 220],
                        _ => [0, 0, 0],
                    }
                }),
            )?;
        }
        // Reflection rays from the game depth (or our primary depth without a reference).
        let depth_src = if game.is_some() {
            "game depth"
        } else {
            "primary-ray depth"
        };
        let depth_buf = ctx
            .gpu
            .upload(game.as_deref().unwrap_or(&depth_ours), false)?;
        let consistent: Vec<bool> = match &game {
            Some(g) => g
                .iter()
                .zip(&depth_ours)
                .map(|(&gd, &o)| gd > 0.0 && o > 0.0 && (gd / o - 1.0).abs() <= 0.01)
                .collect(),
            None => vec![true; n],
        };
        rep.line(format!(
            "reflection rays from the {depth_src} (depth normals), 1 ray/pixel:"
        ));
        let ssr_runs: Vec<(&str, u32, f32)> = vec![
            ("mirror", 0, 0.0),
            ("ggx r0.3", F_GGX, 0.3),
            ("floors", F_FLOORS, 0.0),
        ];
        let mut ssr_out: Vec<(String, Vec<f32>)> = Vec::new();
        for &(name, flags, rough) in &ssr_runs {
            let p = Params {
                flags,
                roughness: rough,
                t_max: 200.0,
                ..Default::default()
            };
            let ms = ctx.timed(
                "ssr_ref",
                &frame,
                &p,
                &tlas,
                Some(&depth_buf),
                None,
                [Some(&tg.ssr), None, None, None],
            )?;
            let v: Vec<f32> = ctx.gpu.read(&tg.ssr, n)?;
            let surf = v.iter().filter(|&&t| t != 0.0).count();
            let hits = v.iter().filter(|&&t| t > 0.0).count();
            rep.line(format!(
                "  screen-space reference ({name}; linear 64 + 8 refine, 200 m): {ms:.3} ms, hits {:.1} % of {surf} surface pixels",
                pct(hits, surf)
            ));
            ssr_out.push((name.to_owned(), v));
        }
        let mirror_ssr = ssr_out[0].1.clone();
        let mirror_ssr_buf = ctx.gpu.upload(&mirror_ssr, false)?;
        struct Variant {
            name: &'static str,
            flags: u32,
            rough: f32,
            cut: bool,
            ssr: usize,
        }
        let variants = [
            Variant {
                name: "mirror, all pixels",
                flags: 0,
                rough: 0.0,
                cut: false,
                ssr: 0,
            },
            Variant {
                name: "mirror + sun shadow ray at hit",
                flags: F_SHADOW,
                rough: 0.0,
                cut: false,
                ssr: 0,
            },
            Variant {
                name: "GGX r=0.3",
                flags: F_GGX,
                rough: 0.3,
                cut: false,
                ssr: 1,
            },
            Variant {
                name: "mirror, floors only",
                flags: F_FLOORS,
                rough: 0.0,
                cut: false,
                ssr: 2,
            },
            Variant {
                name: "mirror, cut-off BVH",
                flags: 0,
                rough: 0.0,
                cut: true,
                ssr: 0,
            },
            Variant {
                name: "hybrid: RT where SS reference missed",
                flags: F_MASK_SSR,
                rough: 0.0,
                cut: false,
                ssr: 0,
            },
            Variant {
                name: "hybrid: RT where depth heuristic low",
                flags: F_MASK_DEPTH,
                rough: 0.0,
                cut: false,
                ssr: 0,
            },
            Variant {
                name: "hybrid SS-miss + cut-off BVH",
                flags: F_MASK_SSR,
                rough: 0.0,
                cut: true,
                ssr: 0,
            },
            Variant {
                name: "hybrid depth heuristic + cut-off BVH",
                flags: F_MASK_DEPTH,
                rough: 0.0,
                cut: true,
                ssr: 0,
            },
        ];
        for v in &variants {
            let (t, t_max) = if v.cut {
                (&tlas_cut, args.cutoff)
            } else {
                (&tlas, 1e4)
            };
            let p = Params {
                flags: v.flags,
                roughness: v.rough,
                t_max,
                ..Default::default()
            };
            let ms = ctx.timed(
                "reflect_rays",
                &frame,
                &p,
                t,
                Some(&depth_buf),
                Some(&mirror_ssr_buf),
                tg.uavs(),
            )?;
            let rt: Vec<f32> = ctx.gpu.read(&tg.a, n)?;
            let s = refl_stats(
                &rt,
                &ssr_out[v.ssr].1,
                &consistent,
                v.cut.then_some(args.cutoff),
            );
            let mut line = format!(
                "  {}: {ms:.3} ms; traced {:.1} % of {} surface pixels, RT hits {:.1} % of traced ({:.1} % closer than 5 cm)",
                v.name,
                pct(s.traced, s.surface),
                s.surface,
                pct(s.rt_hit, s.traced),
                pct(s.rt_self, s.rt_hit)
            );
            let masked = v.flags & (F_MASK_SSR | F_MASK_DEPTH) != 0;
            if !masked {
                let _ = write!(
                    line,
                    "; where SS hit ({:.1} %{}): RT agrees {:.1} / {:.1} / {:.1} % (tol 2/5/10 %, min 0.1 m); disagreeing: RT miss {:.1} %, RT nearer {:.1} %, RT farther {:.1} %; origin-consistent pixels (primary within 1 % of game depth): agree {:.1} % of {}",
                    pct(s.ssr_hit, s.surface),
                    if v.cut {
                        format!(
                            ", {} SS hits beyond {} m excluded",
                            s.ssr_beyond_cutoff, args.cutoff
                        )
                    } else {
                        String::new()
                    },
                    pct(s.agree[0], s.ssr_hit),
                    pct(s.agree[1], s.ssr_hit),
                    pct(s.agree[2], s.ssr_hit),
                    pct(s.rt_miss, s.ssr_hit),
                    pct(s.rt_nearer, s.ssr_hit),
                    pct(s.rt_farther, s.ssr_hit),
                    pct(s.consistent_agree, s.consistent_ssr_hit),
                    s.consistent_ssr_hit
                );
                if v.flags & F_SHADOW == 0 {
                    gate_agree.push((format!("{res} {}", v.name), pct(s.agree[1], s.ssr_hit)));
                    gate_agree.push((
                        format!("{res} {} (origin-consistent pixels)", v.name),
                        pct(s.consistent_agree, s.consistent_ssr_hit),
                    ));
                }
            }
            rep.line(line);
            if (w, h) == (2304, 968) && v.flags & F_SHADOW == 0 {
                gate_refl.push((v.name.to_owned(), ms));
            }
            let kind_tag = if v.flags & F_MASK_SSR != 0 {
                "hybrid_ssmask"
            } else if v.flags & F_MASK_DEPTH != 0 {
                "hybrid_depthmask"
            } else if v.flags & F_GGX != 0 {
                "ggx"
            } else if v.flags & F_FLOORS != 0 {
                "mirror_floors"
            } else if v.flags & F_SHADOW != 0 {
                "mirror_shadow"
            } else {
                "mirror"
            };
            let cut_tag = if v.cut { "_cutoff" } else { "" };
            let file = |kind: &str| out(&format!("reflect_{kind_tag}{cut_tag}_{kind}_{res}.png"));
            let col: Vec<[f32; 4]> = ctx.gpu.read(&tg.color, n)?;
            let e = auto_exposure(
                col.iter()
                    .filter(|c| c[3] > 0.0)
                    .map(|c| [c[0], c[1], c[2]]),
            );
            write_png(
                &file("shaded"),
                w,
                h,
                &rgb_image(n, |i| tonemap([col[i][0], col[i][1], col[i][2]], e)),
            )?;
            write_png(
                &file("distance"),
                w,
                h,
                &rgb_image(n, |i| distance_colour(rt[i])),
            )?;
            if !masked {
                let ssr = &ssr_out[v.ssr].1;
                write_png(
                    &file("vs_ss"),
                    w,
                    h,
                    &rgb_image(n, |i| {
                        let (r, q) = (rt[i], ssr[i]);
                        if r == 0.0 {
                            [0, 0, 0]
                        } else if q > 0.0 && r > 0.0 && (r - q).abs() <= (0.05 * q).max(0.1) {
                            [40, 160, 40]
                        } else if q > 0.0 {
                            [230, 50, 50]
                        } else if r > 0.0 {
                            [90, 90, 90]
                        } else {
                            [20, 30, 90]
                        }
                    }),
                )?;
            }
        }
        // Method-only agreement: both traces start from OUR primary-ray depth, so geometry
        // differences between the decoded scene and the game drop out.
        if game.is_some() {
            let own = ctx.gpu.upload(&depth_ours, false)?;
            let all_ok = vec![true; n];
            for (name, flags) in [
                ("mirror, all pixels", 0u32),
                ("mirror, floors only", F_FLOORS),
            ] {
                let p = Params {
                    flags,
                    t_max: 200.0,
                    ..Default::default()
                };
                ctx.timed(
                    "ssr_ref",
                    &frame,
                    &p,
                    &tlas,
                    Some(&own),
                    None,
                    [Some(&tg.ssr), None, None, None],
                )?;
                let ssr: Vec<f32> = ctx.gpu.read(&tg.ssr, n)?;
                let p = Params {
                    flags,
                    t_max: 1e4,
                    ..Default::default()
                };
                ctx.timed(
                    "reflect_rays",
                    &frame,
                    &p,
                    &tlas,
                    Some(&own),
                    Some(&own),
                    tg.uavs(),
                )?;
                let rt: Vec<f32> = ctx.gpu.read(&tg.a, n)?;
                let s = refl_stats(&rt, &ssr, &all_ok, None);
                rep.line(format!(
                    "  self-consistent ({name}; SS and RT both from our primary depth): SS hits {:.1} %, RT agrees {:.1} / {:.1} / {:.1} % (tol 2/5/10 %); disagreeing: RT miss {:.1} %, nearer {:.1} %, farther {:.1} %",
                    pct(s.ssr_hit, s.surface),
                    pct(s.agree[0], s.ssr_hit),
                    pct(s.agree[1], s.ssr_hit),
                    pct(s.agree[2], s.ssr_hit),
                    pct(s.rt_miss, s.ssr_hit),
                    pct(s.rt_nearer, s.ssr_hit),
                    pct(s.rt_farther, s.ssr_hit)
                ));
                gate_agree.push((
                    format!("{res} {name} (self-consistent depth)"),
                    pct(s.agree[1], s.ssr_hit),
                ));
            }
        }
    }

    // ---------------------------------------------------------------- 4. photo mode demo
    let (gw, gh) = parse_size(&args.gi_size)?;
    let frame = Frame {
        basis: camera.basis(gw, gh),
        ..frame0
    };
    let tg = Targets::new(&ctx.gpu, gw, gh)?;
    let n = tg.pixels;
    ctx.gpu.zero(&tg.aux)?;
    let frames = args.gi_frames.max(1);
    let save_at: Vec<u32> = [1u32, 4, 16, 64, 256, 1024, 4096]
        .into_iter()
        .filter(|&s| s < frames)
        .chain([frames])
        .collect();
    let srv = [
        Some(&tlas.buffer),
        Some(&ctx.prim_offsets),
        Some(&ctx.prims),
        None,
        None,
    ];
    let mut per_frame = Vec::new();
    let mut snapshots: Vec<(u32, Vec<[f32; 4]>)> = Vec::new();
    let mut f = 0u32;
    let t_gi = Instant::now();
    for &target in &save_at {
        // At most DISPATCHES_PER_SUBMIT samples per command list (TDR safety).
        while f < target {
            let chunk = (target - f).min(DISPATCHES_PER_SUBMIT);
            for k in 0..chunk {
                let p = Params {
                    frame: f + k,
                    ray_flags: primary_cull,
                    t_max: 1e4,
                    bounces: args.gi_bounces,
                    flags: F_JITTER,
                    ..Default::default()
                };
                ctx.gpu.timestamp(2 * k);
                ctx.programs.dispatch(
                    &ctx.gpu,
                    "gi",
                    &constants(&frame, &p),
                    &srv,
                    &tg.uavs(),
                    None,
                    (gw, gh),
                )?;
                ctx.gpu.uav_barrier();
                ctx.gpu.timestamp(2 * k + 1);
            }
            let ts = ctx.gpu.timestamps(2 * chunk)?;
            per_frame.extend(
                ts.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|t| ctx.gpu.ticks_to_ms(t[0], t[1])),
            );
            f += chunk;
        }
        snapshots.push((f, ctx.gpu.read(&tg.aux, n)?));
    }
    let gi_wall = t_gi.elapsed().as_secs_f64();
    let fin = &snapshots.last().ok_or("no GI frames")?.1;
    let mean = |a: &[[f32; 4]]| {
        a.iter()
            .map(|c| {
                [
                    c[0] / c[3].max(1.0),
                    c[1] / c[3].max(1.0),
                    c[2] / c[3].max(1.0),
                ]
            })
            .collect::<Vec<_>>()
    };
    let fin_rgb = mean(fin);
    let e = auto_exposure(fin_rgb.iter().copied());
    let fin_lum: f64 = fin_rgb.iter().map(|c| f64::from(lum(*c))).sum::<f64>() / n as f64;
    let mut ms_sorted = per_frame.clone();
    let med = median(&mut ms_sorted);
    rep.line(format!(
        "--- photo-mode demo: {gw}x{gh}, 1 path sample/pixel/frame (primary + {} diffuse bounces, sun NEE with 0.5 deg soft-shadow cone + sky), {frames} frames",
        args.gi_bounces
    ));
    rep.line(format!(
        "  GPU per sample: median {med:.2} ms, min {:.2}, max {:.2}; total GPU {:.2} s, wall {gi_wall:.2} s",
        ms_sorted.first().copied().unwrap_or(0.0),
        ms_sorted.last().copied().unwrap_or(0.0),
        per_frame.iter().sum::<f64>() / 1e3
    ));
    for (spp, acc) in &snapshots {
        let rgb = mean(acc);
        let rmse = (rgb
            .iter()
            .zip(&fin_rgb)
            .map(|(a, b)| f64::from(lum(*a) - lum(*b)).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        rep.line(format!(
            "  {spp:5} spp: relative RMSE vs {frames} spp {:.4}",
            rmse / fin_lum.max(1e-9)
        ));
        write_png(
            &out(&format!("gi_{spp:04}spp.png")),
            gw,
            gh,
            &rgb_image(n, |i| tonemap(rgb[i], e)),
        )?;
    }

    // ---------------------------------------------------------------- 5. gates
    rep.line("--- gates (docs/reflections.md section 3)");
    let gb = bvh_bytes as f64 / (1024.0 * MIB);
    rep.line(format!(
        "  BVH memory < 0.5 GiB: {:.3} GiB -> {} (with primitive data and cut-off TLAS {:.3} GiB)",
        gb,
        if gb < 0.5 { "PASS" } else { "FAIL" },
        rt_bytes as f64 / (1024.0 * MIB)
    ));
    let build_s = (b.build_gpu_ms + b.compact_gpu_ms) / 1e3;
    rep.line(format!(
        "  BLAS build < 2 s at map load: {:.3} s GPU (build + compaction), {:.2} s wall -> {}",
        build_s,
        b.wall_ms / 1e3,
        if b.wall_ms < 2000.0 { "PASS" } else { "FAIL" }
    ));
    for (name, ms) in &gate_refl {
        rep.line(format!(
            "  reflection rays < 1 ms at 2304x968 ({name}): {ms:.3} ms -> {}",
            if *ms < 1.0 { "PASS" } else { "FAIL" }
        ));
    }
    if ref_depth.is_some() {
        for (name, a) in &gate_agree {
            rep.line(format!(
                "  agreement with SS reference where SS hits > 90 % ({name}, 5 % tol): {a:.1} % -> {}",
                if *a > 90.0 { "PASS" } else { "FAIL" }
            ));
        }
    } else {
        rep.line("  agreement gate: no game depth reference given -> not evaluated");
    }
    rep.line(format!(
        "total wall time {:.1} s",
        t_start.elapsed().as_secs_f64()
    ));
    std::fs::write(out("report.txt"), &rep.0).map_err(|e| e.to_string())?;
    Ok(())
}
