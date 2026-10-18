//! Offline check of the hybrid photo terms (`void_rt::hybrid`, docs/raytracing.md section 13):
//! the kernels `photo_sun_vis` and `photo_ao` at the RX010 pose on a host-created device
//! (`Gpu::from_existing`, as the game's DLL), driven exactly as in game (non-blocking `step`
//! with a GPU budget per call, `copy_out` into R32F textures, `set_frame` from an R32F
//! texture). Offline only; reads the archives, the F3 snapshot and the F8 game-depth dump.
//!
//! Checks:
//! 1. Validation at `--size` from our own primary-hit depth (`photo_depth`): the hybrid sun
//!    visibility agrees with the path tracer's first-hit sun visibility (same disc, same
//!    scene) on pixels facing the sun (mean shading normal . sun > 0.2; the path tracer
//!    scores back faces 0, the hybrid term 1) - mean absolute difference and lit/shadowed
//!    classification agreement;
//! 2. `set_frame` (depth from an R32F texture) and `set_depth` (CPU upload) give
//!    bit-identical means (same seeds), `copy_out` textures equal the readback;
//! 3. the game's depth (`--game-depth`, `znear / z` R32F at `--game-depth-size`, resampled to
//!    each size): masks (PNG) of sun visibility and AO after `--target` rays, no NaN, values
//!    in [0, 1], progress complete;
//! 4. `--measure-size` (GPU only): GPU ms of one batch per term (`--rays` rays per pixel =
//!    one frame's work) and per ray, longest submit;
//! 5. false blacks at the game depth (`--dots-size`, default 3440x1440): pixels the path
//!    tracer (`--dots-spp` first-hit samples, no CSM) sees lit and sun-facing that the sun
//!    term after `--target` rays scores below 0.25, overall and on the steps of the pose
//!    (`steps_rect`, the in-game black dots), plus isolated black pixels; running mean and
//!    the filtered output, gates 0.5 % / 0.1 % (GPU runs), PNG `game_false_black_*.png`.
//!
//! WARP first (`--smoke`: 96x40, 64-texel textures, 2,000 nearest instances, 2 rays, target
//! 8), then the GPU; both runs are logged in `local/rt/runs.log`:
//! cargo run --release -p void-rt --example hybrid_check -- --smoke --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 \
//!   --game-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin
//! cargo run --release -p void-rt --example hybrid_check -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 \
//!   --game-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
//!   --measure-size 3440x1440,2304x968

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use void_rt::hybrid::{HybridFrame, HybridSession, HybridSettings, Term};
use void_rt::ingame::{
    Camera, Gpu, GpuScene, PhotoSession, PhotoSettings, SceneOptions, SceneSource, depth_pitch,
    load_scene, upload,
};
use void_rt::lights::Env;
use void_rt::run::{Report, parse_size, read_f32, run_log};
use void_rt::snapshot::{Snapshot, Sun};

#[derive(Parser)]
#[command(about = "Offline check of the hybrid photo terms (RT sun visibility and AO)")]
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
    /// Validation size (own depth vs the path tracer).
    #[arg(long, default_value = "860x360")]
    size: String,
    /// Path-tracer samples for the reference sun visibility.
    #[arg(long, default_value_t = 64)]
    ref_spp: u32,
    /// Rays per pixel and batch (one frame's work) for both terms.
    #[arg(long, default_value_t = 4)]
    rays: u32,
    /// Rays per pixel to accumulate per term.
    #[arg(long, default_value_t = 64)]
    target: u32,
    #[arg(long, default_value_t = 0.27)]
    sun_radius: f32,
    #[arg(long, default_value_t = 1.5)]
    ao_radius: f32,
    #[arg(long, default_value_t = 512)]
    tex_size: u32,
    /// GPU budget per `step` call (ms).
    #[arg(long, default_value_t = 8.0)]
    budget_ms: f32,
    /// F8 game depth (`znear / z`, R32F) of the snapshot's view.
    #[arg(long)]
    game_depth: Option<PathBuf>,
    #[arg(long, default_value = "3440x1440")]
    game_depth_size: String,
    /// Sizes to time and to write masks of (`WxH,WxH`; GPU only).
    #[arg(long)]
    measure_size: Option<String>,
    /// Size of the false-black count at the game depth (path-tracer reference; smoke: `--size`).
    #[arg(long, default_value = "3440x1440")]
    dots_size: String,
    /// Path-tracer samples of the false-black reference.
    #[arg(long, default_value_t = 16)]
    dots_spp: u32,
    #[arg(long)]
    warp: bool,
    /// WARP, 96x40, 64-texel textures, 2000 nearest instances, 2 rays, target 8.
    #[arg(long)]
    smoke: bool,
    #[arg(long, default_value = "local/rt/hybrid/check")]
    out: PathBuf,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

fn main() {
    let mut args = Args::parse();
    if args.smoke {
        args.warp = true;
        args.size = "96x40".into();
        args.ref_spp = args.ref_spp.min(8);
        args.rays = args.rays.min(2);
        args.target = args.target.min(8);
        args.tex_size = args.tex_size.min(64);
        args.measure_size = None;
        args.dots_size = args.size.clone();
        args.dots_spp = args.dots_spp.min(8);
        if args.out == Path::new("local/rt/hybrid/check") {
            args.out = PathBuf::from("local/rt/hybrid/check_smoke");
        }
    }
    let mode = if args.warp {
        "hybrid-check warp"
    } else {
        "hybrid-check gpu"
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
    let r = check(&args);
    run_log(
        &args.runs_log,
        &match &r {
            Ok(()) => format!("end {mode} run: ok"),
            Err(e) => format!("end {mode} run: error: {}", e.lines().next().unwrap_or("")),
        },
    );
    if let Err(e) = r {
        eprintln!("hybrid_check: {e}");
        std::process::exit(1);
    }
}

/// A device on the high-performance adapter (or WARP) and a direct queue, created outside
/// `void_rt` as the DLL does.
fn host_device(warp: bool) -> Result<(ID3D12Device5, ID3D12CommandQueue, String), String> {
    // SAFETY: DXGI / D3D12 creation calls with valid arguments.
    unsafe {
        let factory: IDXGIFactory6 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
        let adapter: IDXGIAdapter1 = if warp {
            factory.EnumWarpAdapter().map_err(|e| e.to_string())?
        } else {
            factory
                .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
                .map_err(|e| e.to_string())?
        };
        let d = adapter.GetDesc1().map_err(|e| e.to_string())?;
        let n = d.Description.iter().position(|&c| c == 0).unwrap_or(128);
        let name = String::from_utf16_lossy(&d.Description[..n]);
        let mut device: Option<ID3D12Device5> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
            .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
        let device = device.ok_or("no device")?;
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(|e| e.to_string())?;
        Ok((device, queue, name))
    }
}

/// An R32F texture in COMMON (the host's D3D11-shared textures arrive like this).
fn texture(gpu: &Gpu, w: u32, h: u32) -> Result<ID3D12Resource, String> {
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid descriptors and out pointer.
    unsafe {
        gpu.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("texture: {e}"))?;
    res.ok_or_else(|| "no texture".into())
}

fn pitched(v: &[f32], w: u32) -> Vec<f32> {
    let row = (depth_pitch(w) / 4) as usize;
    let w = w as usize;
    let mut out = vec![0f32; row * v.len() / w];
    for (dst, src) in out.chunks_mut(row).zip(v.chunks(w)) {
        dst[..w].copy_from_slice(src);
    }
    out
}

/// `values` (row-major) into `tex` (blocking), through a pitched buffer.
fn fill_texture(
    gpu: &mut Gpu,
    tex: &ID3D12Resource,
    values: &[f32],
    w: u32,
    h: u32,
) -> Result<(), String> {
    let b = gpu.upload(&pitched(values, w), false)?;
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    gpu.copy_buffer_to_texture(
        &b,
        depth_pitch(w),
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32_TYPELESS,
        w,
        h,
        tex,
    )?;
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON,
    );
    gpu.submit()
}

/// `tex` back to the CPU (blocking), through a pitched buffer.
fn read_texture(gpu: &mut Gpu, tex: &ID3D12Resource, w: u32, h: u32) -> Result<Vec<f32>, String> {
    let b = gpu.buffer(u64::from(h) * depth_pitch(w), false)?;
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    gpu.copy_texture_to_buffer(
        tex,
        DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32_TYPELESS,
        w,
        h,
        &b,
        depth_pitch(w),
    )?;
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    gpu.submit()?;
    let row = (depth_pitch(w) / 4) as usize;
    let raw: Vec<f32> = gpu.read(&b, row * h as usize)?;
    Ok(raw
        .chunks(row)
        .flat_map(|r| r[..w as usize].iter().copied())
        .collect())
}

fn gray_png(path: &Path, v: &[f32], w: u32, h: u32) -> Result<(), String> {
    let px: Vec<u8> = v
        .iter()
        .flat_map(|x| {
            let g = (x.clamp(0.0, 1.0) * 255.0).round() as u8;
            [g, g, g]
        })
        .collect();
    void_rt::image::write_png(path, w, h, &px)
}

/// Game depth `znear / z` -> view depth (0 = sky).
fn view_depth(d: &[f32], znear: f32) -> Vec<f32> {
    d.iter()
        .map(|&x| {
            if x > 0.0 && x.is_finite() {
                znear / x
            } else {
                0.0
            }
        })
        .collect()
}

fn frame(camera: &Camera, w: u32, h: u32, sun: [f32; 3], znear: f32) -> HybridFrame {
    HybridFrame {
        basis: camera.basis(w, h),
        jitter: [0.0; 2],
        sun_dir: sun,
        // Game depth znear / z: dz/dd = z^2 / znear; two R24 steps.
        depth_step: 2.0 / (znear * 16_777_215.0),
    }
}

/// Steps until done; returns (step calls, of which busy, wall s).
fn run_to_target(
    gpu: &mut Gpu,
    s: &mut HybridSession,
    budget: f32,
) -> Result<(usize, usize, f64), String> {
    let t = Instant::now();
    let (mut calls, mut busy) = (0usize, 0usize);
    loop {
        let p = s.step(gpu, budget)?;
        calls += 1;
        busy += usize::from(p.busy);
        if p.done {
            break;
        }
        if calls > 50_000_000 {
            return Err("step did not finish".into());
        }
        std::thread::yield_now();
    }
    Ok((calls, busy, t.elapsed().as_secs_f64()))
}

struct Stats {
    nan: usize,
    out_of_range: usize,
    mean: f64,
    /// Fraction of surface pixels below 0.5.
    dark: f64,
}

fn stats(v: &[f32], depth: &[f32]) -> Stats {
    let (mut nan, mut oor, mut sum, mut n, mut dark) = (0, 0, 0.0, 0usize, 0usize);
    for (x, z) in v.iter().zip(depth) {
        if !x.is_finite() {
            nan += 1;
            continue;
        }
        if !(0.0..=1.0).contains(x) {
            oor += 1;
        }
        if *z > 0.0 {
            sum += f64::from(*x);
            n += 1;
            dark += usize::from(*x < 0.5);
        }
    }
    Stats {
        nan,
        out_of_range: oor,
        mean: sum / n.max(1) as f64,
        dark: dark as f64 / n.max(1) as f64,
    }
}

/// Receivers whose depth-reconstructed geometric normal (the kernel's construction, CPU
/// port) faces the sun by more than 0.2; the hybrid term scores the others 1 by convention.
fn geo_facing(basis: &void_rt::camera::RayBasis, z: &[f32], sun: [f32; 3]) -> Vec<bool> {
    let (w, h) = (basis.width as i64, basis.height as i64);
    let at = |x: i64, y: i64| z[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
    let pos = |x: i64, y: i64, d: f32| {
        let r = basis.direction(x as f32 + 0.5, y as f32 + 0.5);
        std::array::from_fn::<f32, 3, _>(|k| basis.org[k] + d * r[k])
    };
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut out = vec![false; z.len()];
    for y in 0..h {
        for x in 0..w {
            let c = at(x, y);
            if c <= 0.0 {
                continue;
            }
            let p = pos(x, y, c);
            let (zpx, zmx, zpy, zmy) = (at(x + 1, y), at(x - 1, y), at(x, y + 1), at(x, y - 1));
            let mx = x > 0 && (x >= w - 1 || (zmx - c).abs() <= (zpx - c).abs());
            let my = y > 0 && (y >= h - 1 || (zmy - c).abs() <= (zpy - c).abs());
            let dx = if mx {
                sub(p, pos(x - 1, y, zmx))
            } else {
                sub(pos(x + 1, y, zpx), p)
            };
            let dy = if my {
                sub(p, pos(x, y - 1, zmy))
            } else {
                sub(pos(x, y + 1, zpy), p)
            };
            let mut n = [
                dx[1] * dy[2] - dx[2] * dy[1],
                dx[2] * dy[0] - dx[0] * dy[2],
                dx[0] * dy[1] - dx[1] * dy[0],
            ];
            let v = sub(p, basis.org);
            if n[0] * v[0] + n[1] * v[1] + n[2] * v[2] > 0.0 {
                n = n.map(|a| -a);
            }
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            out[(y * w + x) as usize] =
                l > 1e-12 && (n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / l > 0.2;
        }
    }
    out
}

/// Receivers on locally planar depth: both second differences below 0.2 % of the depth
/// (excludes foliage, silhouettes and the region border).
fn planar_mask(z: &[f32], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let c = z[i];
            let n = [z[i - 1], z[i + 1], z[i - w], z[i + w]];
            if c <= 0.0 || n.iter().any(|v| *v <= 0.0) {
                continue;
            }
            let t = 0.002 * c;
            out[i] = (n[0] + n[1] - 2.0 * c).abs() < t && (n[2] + n[3] - 2.0 * c).abs() < t;
        }
    }
    out
}

/// (mean absolute difference, lit/shadowed classification agreement, pixels) over `mask`.
fn compare_vis(a: &[f32], b: &[f32], mask: &[bool]) -> (f64, f64, usize) {
    let (mut diff, mut agree, mut n) = (0.0f64, 0usize, 0usize);
    for ((x, y), m) in a.iter().zip(b).zip(mask) {
        if *m {
            diff += f64::from((x - y).abs());
            agree += usize::from((*x >= 0.5) == (*y >= 0.5));
            n += 1;
        }
    }
    (diff / n.max(1) as f64, agree as f64 / n.max(1) as f64, n)
}

/// The path tracer's mean first-hit sun visibility and mean shading normal per pixel.
type Reference = (Vec<f32>, Vec<[f32; 4]>);

/// Primary-hit view depth of our own scene at `w x h` (`photo_depth`), and optionally the
/// path tracer's [`Reference`] after `spp` samples.
fn own_depth(
    gpu: &mut Gpu,
    gs: &Arc<GpuScene>,
    camera: Camera,
    (w, h): (u32, u32),
    znear: f32,
    spp: u32,
    budget: f32,
) -> Result<(Vec<f32>, Option<Reference>), String> {
    let settings = PhotoSettings {
        spp_target: spp,
        znear,
        tile_pixels: (w * h / 8).max(w * 4),
        ..PhotoSettings::default()
    };
    let mut s = PhotoSession::new(gpu, gs.clone(), camera, w, h, settings)?;
    while !s.step(gpu, budget)?.done {
        std::thread::yield_now();
    }
    let depth = view_depth(&s.read_depth(gpu)?, znear);
    let vis = if spp > 0 {
        let (alb, nrm) = s.read_features(gpu)?;
        Some((alb.iter().map(|a| a[3]).collect(), nrm))
    } else {
        None
    };
    Ok((depth, vis))
}

/// The steps of the RX010 pose (the in-game dots, `local/rt/dots_crop.png`): x 1300..2000,
/// y 700..1000 at 3440x1440, scaled to `w x h`.
fn steps_rect(w: u32, h: u32) -> [usize; 4] {
    let sx = |v: u32| (u64::from(v) * u64::from(w) / 3440) as usize;
    let sy = |v: u32| (u64::from(v) * u64::from(h) / 1440) as usize;
    [sx(1300), sy(700), sx(2000), sy(1000)]
}

/// False blacks of the sun term at the game depth: surface pixels the path tracer
/// (`--dots-spp` first-hit samples of our scene, no CSM) sees lit (mean >= 0.75, mean shading
/// normal . sun > 0.2) that the hybrid term after `--target` rays scores below 0.25 (black
/// under `min(CSM, RT)`). Whole image and the steps rectangle; diagnostic PNG (red).
#[allow(clippy::too_many_arguments)]
fn false_blacks(
    args: &Args,
    gpu: &mut Gpu,
    gs: &Arc<GpuScene>,
    snap: &Snapshot,
    sun: [f32; 3],
    settings: HybridSettings,
    game_depth: &[f32],
    game_size: (u32, u32),
    rep: &mut Report,
    fails: &mut Vec<String>,
) -> Result<(), String> {
    let (w, h) = parse_size(&args.dots_size)?;
    let camera = snap.camera;
    let depth = camera.basis(w, h).resample(game_depth, game_size);
    let (_, reference) = own_depth(
        gpu,
        gs,
        camera,
        (w, h),
        snap.znear,
        args.dots_spp.max(1),
        args.budget_ms,
    )?;
    let (pt, nrm) = reference.ok_or("no path-tracer features")?;
    let fr = frame(&camera, w, h, sun, snap.znear);
    let mut s = HybridSession::new(gpu, gs.clone(), fr, settings)?;
    s.set_depth(gpu, fr, settings, &depth)?;
    run_to_target(gpu, &mut s, args.budget_ms)?;
    let sv = s.read(gpu, Term::Sun)?;
    let mean = s.read_mean(gpu, Term::Sun)?;
    let receivers = s.read_depth(gpu)?;
    let rejected = receivers
        .iter()
        .zip(&depth)
        .filter(|(r, z)| **z > 0.0 && **r < 0.0)
        .count();
    let rect = steps_rect(w, h);
    let (wu, hu) = (w as usize, h as usize);
    let in_steps =
        |x: usize, y: usize| (rect[0]..rect[2]).contains(&x) && (rect[1]..rect[3]).contains(&y);
    // Path-tracer lit, sun-facing surface pixels.
    let candidate: Vec<bool> = (0..sv.len())
        .map(|i| {
            let n = nrm[i];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
            depth[i] > 0.0
                && (n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]) / l > 0.2
                && pt[i] >= 0.75
        })
        .collect();
    let lit = candidate.iter().filter(|c| **c).count();
    let lit_steps = (0..sv.len())
        .filter(|&i| candidate[i] && in_steps(i % wu, i / wu))
        .count();
    // (false blacks, of them in the steps rect, isolated black pixels, of them in the steps
    // rect). Isolated (dots and 1-px dashes): black (< 0.25) with at least 6 of the 8
    // neighbours lit (>= 0.75); a real shadow is wider than one pixel.
    let count = |v: &[f32]| {
        let mut c = [0usize; 4];
        for i in 0..v.len() {
            if candidate[i] && v[i] < 0.25 {
                c[0] += 1;
                c[1] += usize::from(in_steps(i % wu, i / wu));
            }
        }
        for y in 1..hu - 1 {
            for x in 1..wu - 1 {
                let i = y * wu + x;
                if depth[i] <= 0.0 || v[i] >= 0.25 {
                    continue;
                }
                let lit_n = [
                    i - wu - 1,
                    i - wu,
                    i - wu + 1,
                    i - 1,
                    i + 1,
                    i + wu - 1,
                    i + wu,
                    i + wu + 1,
                ]
                .iter()
                .filter(|&&j| v[j] >= 0.75)
                .count();
                if lit_n >= 6 {
                    c[2] += 1;
                    c[3] += usize::from(in_steps(x, y));
                }
            }
        }
        c
    };
    let px: Vec<u8> = (0..sv.len())
        .flat_map(|i| {
            let g = (sv[i].clamp(0.0, 1.0) * 200.0) as u8;
            if candidate[i] && sv[i] < 0.25 {
                [255, 0, 0]
            } else {
                [g, g, g]
            }
        })
        .collect();
    void_rt::image::write_png(
        &args.out.join(format!("game_false_black_{w}x{h}.png")),
        w,
        h,
        &px,
    )?;
    let surface = depth.iter().filter(|z| **z > 0.0).count().max(1);
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    rep.line(format!(
        "false blacks, game depth {w}x{h} ({} rays) vs path tracer ({} spp first-hit, no CSM): lit sun-facing px {lit} ({lit_steps} in the steps rect {:?}); receivers rejected (CSM only) {rejected} ({:.2} % of surface)",
        settings.target,
        args.dots_spp.max(1),
        rect,
        pct(rejected, surface)
    ));
    let mut out = [0usize; 4];
    for (name, v) in [("running mean", &mean), ("output (filtered)", &sv)] {
        let c = count(v);
        rep.line(format!(
            "  {name}: false black {} ({:.3} %), steps rect {} ({:.3} %); isolated black px {} ({} in the steps rect)",
            c[0],
            pct(c[0], lit),
            c[1],
            pct(c[1], lit_steps),
            c[2],
            c[3]
        ));
        out = c;
    }
    // Gate (GPU runs at full size; O before the fix at RX010 3440x1440: 1.584 % overall,
    // 0.523 % in the steps rect).
    if !args.smoke && (pct(out[0], lit) > 0.5 || pct(out[1], lit_steps) > 0.1) {
        fails.push(format!(
            "false blacks: {:.3} % overall, {:.3} % in the steps rect (gates 0.5 %, 0.1 %)",
            pct(out[0], lit),
            pct(out[1], lit_steps)
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn check(args: &Args) -> Result<(), String> {
    let mut rep = Report::default();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let (w, h) = parse_size(&args.size)?;
    let mut fails: Vec<String> = Vec::new();

    let (device, queue, adapter) = host_device(args.warp)?;
    let mut gpu = Gpu::from_existing(device, queue)?;
    rep.line(format!(
        "device: {adapter} (raytracing tier {}, SM {})",
        gpu.info.raytracing_tier_name(),
        gpu.info.shader_model_name()
    ));

    // ------------------------------------------------------------ scene
    let snap = Snapshot::load(&args.snapshot)?;
    let camera = snap.camera;
    let sun = snap
        .sun
        .unwrap_or_else(|| Sun::from_angles(26.0, 100.0, [0.215_861, 0.171_441, 0.116_971]));
    let src = SceneSource {
        game_dir: args.game.clone(),
        map: args.map.clone(),
        entities: snap.entities.clone(),
        lights: snap.lights.clone(),
        sun: Some(sun),
        env: Some(snap.env.unwrap_or(Env {
            sky_zenith: [0.0; 3],
            sky_horizon: [37.7451, 52.411_766, 55.0],
            ambient: [1.0; 3],
        })),
    };
    let opts = SceneOptions {
        texture_size: args.tex_size,
        upload_batch_bytes: 256 << 20,
        tlas_limit: args.smoke.then_some((2000, camera.org)),
        ..SceneOptions::default()
    };
    let t = Instant::now();
    let cpu = std::thread::spawn(move || load_scene(&src, &opts))
        .join()
        .map_err(|_| "load_scene thread panicked")??;
    rep.line(format!(
        "load_scene: {:.1} s, {} instances",
        t.elapsed().as_secs_f64(),
        cpu.scene.instances.len()
    ));
    let gs = Arc::new(upload(&mut gpu, cpu)?);
    rep.line(format!(
        "upload: {:.1} s, full TLAS {} instances; sun dir {:?}, disc radius {} deg, AO radius {} m, {} rays per batch, target {}",
        gs.upload_seconds(),
        gs.full_tlas().stats.instances,
        sun.dir,
        args.sun_radius,
        args.ao_radius,
        args.rays,
        args.target
    ));
    let settings = HybridSettings {
        sun_rays: args.rays,
        ao_rays: args.rays,
        target: args.target,
        sun_radius_deg: args.sun_radius,
        ao_radius: args.ao_radius,
        ..HybridSettings::default()
    };

    // ------------------------------------------------------------ 1. own depth vs path tracer
    let (depth, pt_vis) = own_depth(
        &mut gpu,
        &gs,
        camera,
        (w, h),
        snap.znear,
        args.ref_spp,
        args.budget_ms,
    )?;
    let (pt_vis, pt_nrm) = pt_vis.ok_or("no path-tracer features")?;
    // Pixels whose mean shading normal faces the sun (see the module doc).
    let facing: Vec<bool> = pt_nrm
        .iter()
        .zip(&depth)
        .map(|(n, z)| {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
            *z > 0.0 && (n[0] * sun.dir[0] + n[1] * sun.dir[1] + n[2] * sun.dir[2]) / l > 0.2
        })
        .collect();
    let fr = frame(&camera, w, h, sun.dir, snap.znear);
    let mut s = HybridSession::new(&mut gpu, gs.clone(), fr, settings)?;
    s.set_depth(&mut gpu, fr, settings, &depth)?;
    let (calls, busy, wall) = run_to_target(&mut gpu, &mut s, args.budget_ms)?;
    let p = s.finish(&mut gpu)?;
    let sun_v = s.read(&mut gpu, Term::Sun)?;
    let ao_v = s.read(&mut gpu, Term::Ao)?;
    let (submits, max_submit) = s.submit_stats();
    // Foliage (alpha cards, noisy depth) is ignored by the engine's `.y` (vegetation reads
    // its own CSM): the gate uses planar receivers (second depth differences < 0.2 % of z)
    // that face the sun by both conventions (path tracer: shading normal; hybrid: the
    // depth-reconstructed normal, back faces scored 1).
    let planar = planar_mask(&depth, w as usize, h as usize);
    let geo = geo_facing(&fr.basis, &depth, sun.dir);
    let both: Vec<bool> = facing
        .iter()
        .zip(&planar)
        .zip(&geo)
        .map(|((a, b), c)| *a && *b && *c)
        .collect();
    {
        // Diagnostic: red = hybrid lit / path tracer shadowed, blue = the reverse, grey =
        // agreeing min(); dark = outside the gate mask.
        let px: Vec<u8> = (0..sun_v.len())
            .flat_map(|i| {
                let (a, b) = (sun_v[i] >= 0.5, pt_vis[i] >= 0.5);
                let g = (sun_v[i].min(pt_vis[i]).clamp(0.0, 1.0) * 160.0) as u8;
                match (both[i], a, b) {
                    (false, ..) => [g / 4, g / 4, g / 4],
                    (true, true, false) => [255, 0, 0],
                    (true, false, true) => [0, 0, 255],
                    _ => [g, g, g],
                }
            })
            .collect();
        void_rt::image::write_png(&args.out.join(format!("own_diff_{w}x{h}.png")), w, h, &px)?;
    }
    let (mad_all, agree_all, n_all) = compare_vis(&sun_v, &pt_vis, &facing);
    let (mad, agreement, n) = compare_vis(&sun_v, &pt_vis, &both);
    let lifted = s
        .read_depth(&mut gpu)?
        .iter()
        .zip(&depth)
        .filter(|(a, b)| **b > 0.0 && **a > 0.0 && a != b)
        .count();
    let ss = stats(&sun_v, &depth);
    let sa = stats(&ao_v, &depth);
    rep.line(format!(
        "own depth {w}x{h}: {calls} step calls ({busy} busy), wall {wall:.2} s, {submits} submits, longest {max_submit:.2} ms GPU; sun {}/{} rays, {:.3} ms per batch; AO {}/{} rays, {:.3} ms per batch",
        p.sun.rays, p.sun.target, p.sun.ms_per_batch, p.ao.rays, p.ao.target, p.ao.ms_per_batch
    ));
    rep.line(format!(
        "  sun visibility vs path tracer ({} spp first-hit): planar sun-facing receivers ({n} px): mean abs diff {mad:.4}, lit/shadow agreement {:.2} %; all sun-facing ({n_all} px): {mad_all:.4}, {:.2} %; receivers moved by the prepass {lifted}; sun mean {:.3} (path tracer {:.3}), shadowed {:.1} %; AO mean {:.3}, below 0.5 {:.1} %; NaN {} / {}, out of [0,1] {} / {}",
        args.ref_spp,
        100.0 * agreement,
        100.0 * agree_all,
        ss.mean,
        pt_vis
            .iter()
            .zip(&both)
            .filter(|(_, f)| **f)
            .map(|(v, _)| f64::from(*v))
            .sum::<f64>()
            / n.max(1) as f64,
        100.0 * ss.dark,
        sa.mean,
        100.0 * sa.dark,
        ss.nan,
        sa.nan,
        ss.out_of_range,
        sa.out_of_range
    ));
    for (name, v) in [
        ("own_sun_vis", &sun_v),
        ("own_ao", &ao_v),
        ("own_pathtracer_sun_vis", &pt_vis),
    ] {
        gray_png(&args.out.join(format!("{name}_{w}x{h}.png")), v, w, h)?;
    }
    if !p.done || p.sun.rays < args.target || p.ao.rays < args.target {
        fails.push(format!("progress {p:?}"));
    }
    if ss.nan + sa.nan + ss.out_of_range + sa.out_of_range > 0 {
        fails.push("NaN or out-of-range values".into());
    }
    // Agreement gate (GPU runs: the smoke run's 8 samples are too noisy): a sample-count-
    // limited reference (ref_spp first-hit samples, jittered pixel positions) against our
    // depth-based origins.
    if !args.smoke && (agreement < 0.95 || mad > 0.08) {
        fails.push(format!(
            "sun visibility vs path tracer: agreement {:.2} %, mean abs diff {mad:.4}",
            100.0 * agreement
        ));
    }
    if max_submit > 16.0 && !args.warp {
        fails.push(format!("a submit took {max_submit:.2} ms GPU (> 16)"));
    }

    // ------------------------------------------------------------ 2. texture paths
    let dtex = texture(&gpu, w, h)?;
    fill_texture(&mut gpu, &dtex, &depth, w, h)?;
    let one = HybridSettings {
        target: settings.rays(Term::Sun),
        ..settings
    };
    s.set_frame(&mut gpu, fr, one, &dtex)?;
    run_to_target(&mut gpu, &mut s, args.budget_ms)?;
    let from_tex = [s.read(&mut gpu, Term::Sun)?, s.read(&mut gpu, Term::Ao)?];
    s.set_depth(&mut gpu, fr, one, &depth)?;
    run_to_target(&mut gpu, &mut s, args.budget_ms)?;
    let from_cpu = [s.read(&mut gpu, Term::Sun)?, s.read(&mut gpu, Term::Ao)?];
    let (sun_tex, ao_tex) = (texture(&gpu, w, h)?, texture(&gpu, w, h)?);
    let fence = s.copy_out(&mut gpu, Some(&sun_tex), Some(&ao_tex))?;
    while !gpu.completed(fence)? {
        std::thread::yield_now();
    }
    gpu.poll()?;
    let copied = [
        read_texture(&mut gpu, &sun_tex, w, h)?,
        read_texture(&mut gpu, &ao_tex, w, h)?,
    ];
    let bits = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    let same_input = bits(&from_tex[0], &from_cpu[0]) && bits(&from_tex[1], &from_cpu[1]);
    let same_copy = bits(&copied[0], &from_cpu[0]) && bits(&copied[1], &from_cpu[1]);
    rep.line(format!(
        "  depth from an R32F texture (set_frame) vs CPU upload (set_depth), one batch: bit-identical {same_input}; copy_out textures equal the readback {same_copy}"
    ));
    if !same_input || !same_copy {
        fails.push("texture paths differ".into());
    }
    drop(s);

    // ------------------------------------------------------------ 3./4. game depth, timing
    let game = match &args.game_depth {
        Some(p) => {
            let (gw, gh) = parse_size(&args.game_depth_size)?;
            let d = read_f32(p)?;
            if d.len() != (gw * gh) as usize {
                return Err(format!(
                    "{}: {} values, {gw}x{gh} expected",
                    p.display(),
                    d.len()
                ));
            }
            Some((view_depth(&d, snap.znear), (gw, gh)))
        }
        None => None,
    };
    let mut sizes = vec![(w, h)];
    if let Some(m) = &args.measure_size {
        for t in m.split(',') {
            sizes.push(parse_size(t.trim())?);
        }
    }
    for (k, &(mw, mh)) in sizes.iter().enumerate() {
        let timed = k > 0;
        let (depth, source) = match &game {
            Some((d, src)) => (camera.basis(mw, mh).resample(d, *src), "game depth"),
            None => (
                own_depth(
                    &mut gpu,
                    &gs,
                    camera,
                    (mw, mh),
                    snap.znear,
                    0,
                    args.budget_ms,
                )?
                .0,
                "own depth",
            ),
        };
        let fr = frame(&camera, mw, mh, sun.dir, snap.znear);
        let mut s = HybridSession::new(&mut gpu, gs.clone(), fr, settings)?;
        s.set_depth(&mut gpu, fr, settings, &depth)?;
        let (calls, _, wall) = run_to_target(&mut gpu, &mut s, args.budget_ms)?;
        let p = s.finish(&mut gpu)?;
        let (submits, max_submit) = s.submit_stats();
        let (rows, tiles) = s.tiling();
        let slowest = |t| s.tile_ms(t).into_iter().fold(0.0f64, f64::max);
        rep.line(format!(
            "  {mw}x{mh}: slowest tile (GPU ms): prepass {:.2}, sun {:.2}, AO {:.2}",
            slowest(None),
            slowest(Some(Term::Sun)),
            slowest(Some(Term::Ao))
        ));
        let sv = s.read(&mut gpu, Term::Sun)?;
        let av = s.read(&mut gpu, Term::Ao)?;
        let moved = s
            .read_depth(&mut gpu)?
            .iter()
            .zip(&depth)
            .filter(|(a, b)| **b > 0.0 && **a > 0.0 && a != b)
            .count();
        let surface = depth.iter().filter(|z| **z > 0.0).count().max(1);
        let (ss, sa) = (stats(&sv, &depth), stats(&av, &depth));
        rep.line(format!(
            "{source} {mw}x{mh}{}: {tiles} tiles of {rows} rows, {calls} step calls, wall {wall:.2} s, {submits} submits (longest {max_submit:.2} ms GPU); prepass {:.3} ms, receivers moved onto our surface {:.1} %; sun {:.3} ms per batch of {} rays ({:.3} ms per ray), mean {:.3}, shadowed {:.1} %; AO {:.3} ms per batch ({:.3} per ray), mean {:.3}, below 0.5 {:.1} %; NaN {}",
            if timed { " (timed)" } else { "" },
            p.prep_ms,
            100.0 * moved as f64 / surface as f64,
            p.sun.ms_per_batch,
            p.sun.rays_per_batch,
            p.sun.ms_per_batch / p.sun.rays_per_batch.max(1) as f32,
            ss.mean,
            100.0 * ss.dark,
            p.ao.ms_per_batch,
            p.ao.ms_per_batch / p.ao.rays_per_batch.max(1) as f32,
            sa.mean,
            100.0 * sa.dark,
            ss.nan + sa.nan
        ));
        let tag = if game.is_some() { "game" } else { "own" };
        gray_png(
            &args.out.join(format!("{tag}_sun_vis_{mw}x{mh}.png")),
            &sv,
            mw,
            mh,
        )?;
        gray_png(
            &args.out.join(format!("{tag}_ao_{mw}x{mh}.png")),
            &av,
            mw,
            mh,
        )?;
        if !p.done || ss.nan + sa.nan + ss.out_of_range + sa.out_of_range > 0 {
            fails.push(format!("{mw}x{mh}: progress {p:?} or invalid values"));
        }
        if timed && max_submit > 16.0 {
            fails.push(format!(
                "{mw}x{mh}: a submit took {max_submit:.2} ms GPU (> 16)"
            ));
        }
    }

    // ------------------------------------------------------------ 5. false blacks
    if let Some((gd, src)) = &game {
        false_blacks(
            args, &mut gpu, &gs, &snap, sun.dir, settings, gd, *src, &mut rep, &mut fails,
        )?;
    }
    std::fs::write(args.out.join("report.txt"), &rep.0).map_err(|e| format!("report.txt: {e}"))?;
    if fails.is_empty() {
        rep.line("PASS");
        Ok(())
    } else {
        Err(format!("FAILED:\n  {}", fails.join("\n  ")))
    }
}
