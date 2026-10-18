//! Denoiser of the ray-traced reflections over frame sequences (part of `offline_check`).
//! Camera pitched 20 degrees down, so the 12 m wall rises above the top of the frame and the
//! floor reflects its off-screen part (shaded hits, calibrated against the on-screen ones);
//! weights 0.8 on the whole floor (nothing left to SSR: every reflection is ray-traced).
//! "Before" = E077 (`F_LEGACY`: global calibration, hard on/off-screen switch, rotated samples
//! every frame, rays composited as traced), "after" = this build (denoiser as the SSR one).
//! Metrics on the reflection term (output - input colour):
//!
//! - stability: static camera with the engine's sub-pixel jitter (Halton 2,3 over 8 frames),
//!   frame-to-frame RMSE after the history settled, on a mirror floor (all reflecting pixels
//!   and those whose mirror ray hits geometry outside the frame), a glossy floor (roughness
//!   0.3) and the glossy floor with a 25 % ray budget;
//! - hand-off: a floor band that SSR covers on every other frame (no weight, no requests
//!   there), RMSE between consecutive frames that need the rays;
//! - convergence: glossy floor, no jitter, RMSE against a 256-sample photo accumulation;
//! - ghosting: 25 frames strafing 4 cm per frame with motion vectors, RMSE at the final pose
//!   against the settled result there, and the residual on pixels whose settled reflection
//!   changed between the two poses;
//! - GPU ms (D3D12 timestamps around our work) before / after.

use super::*;

const PITCH: f32 = 20.0 * std::f32::consts::PI / 180.0;
/// GGX alpha of the glossy floor (perceptual roughness 0.3).
const GLOSSY: f32 = 0.09;
const STRAFE_STEP: f32 = 0.04;
const STRAFE_FRAMES: u32 = 25;

fn halton(mut i: u32, b: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

/// The engine-like TAA jitter of frame k in NDC (8 phases, +-0.5 px).
fn jitter(size: [u32; 2], k: u32) -> [f32; 2] {
    let i = k % 8 + 1;
    [
        (halton(i, 2) - 0.5) * 2.0 / size[0] as f32,
        (halton(i, 3) - 0.5) * 2.0 / size[1] as f32,
    ]
}

fn pitched(eye: [f32; 3], jitter: [f32; 2]) -> Cam {
    Cam {
        eye,
        pitch: PITCH,
        jitter,
    }
}

/// How a sequence runs: E077 (`legacy`), the E085 denoiser (no firefly handling), or this
/// build with or without the denoiser.
#[derive(Clone, Copy, PartialEq)]
enum Path {
    Before,
    E085,
    NoDenoise,
    After,
}

struct Seq {
    /// Frame-to-frame RMSE of the reflection per mask (mean over the measured frames).
    stability: Vec<f64>,
    /// Largest frame-to-frame luminance change of one pixel's reflection per mask.
    max_change: Vec<f32>,
    /// Mean reflection luminance per pixel over the measured frames.
    mean_px: Vec<f32>,
    /// Mean output luminance per pixel over the measured frames.
    mean_out: Vec<f32>,
    /// Reflection term (output - input) of the requested snapshot frames.
    snaps: Vec<Vec<[f32; 3]>>,
    last: Vec<[f32; 4]>,
    ms: f64,
    nan: usize,
    stats: Stats,
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn inputs(t: &FrameTextures) -> Inputs<'_> {
    Inputs {
        color: &t.color,
        depth: &t.depth,
        motion: &t.motion,
        req0: &t.req0,
        req1: &t.req1,
        rtw: &t.rtw,
    }
}

/// RMSE of `a - b` over the pixels of `mask`.
fn rmse(a: &[[f32; 3]], b: &[[f32; 3]], mask: &[bool]) -> f64 {
    let (mut s, mut n) = (0f64, 0usize);
    for i in 0..mask.len() {
        if mask[i] {
            for c in 0..3 {
                s += f64::from(a[i][c] - b[i][c]).powi(2);
            }
            n += 3;
        }
    }
    (s / n.max(1) as f64).sqrt()
}

/// Runs `n` frames (`make(k)` = camera and inputs of frame k) on a fresh pipeline; frames
/// k >= `from` with k % `step` == 0 compare their reflection with frame k - `step` on every
/// mask; `snap` lists frames whose reflection term is kept.
#[allow(clippy::too_many_arguments)]
fn sequence(
    gpu: &mut Gpu,
    sc: &SceneRef,
    size: [u32; 2],
    path: Path,
    budget: u32,
    n: u32,
    from: u32,
    step: u32,
    masks: &[&[bool]],
    snap: &[u32],
    make: &mut dyn FnMut(u32) -> (Cam, Frame),
) -> Result<Seq, String> {
    let device = gpu.device.clone();
    let mut g12 = Gpu12::new(&device)?;
    let list: ID3D12GraphicsCommandList = gpu.list.cast().map_err(|e| e.to_string())?;
    let ts_freq = gpu.ts_freq as u64;
    let (p00, p11) = proj(size);
    let knobs = Knobs {
        debug: 0,
        flags: F_SHADOW
            | match path {
                Path::Before => F_LEGACY,
                Path::E085 => F_LINEAR,
                Path::NoDenoise | Path::After => 0,
            },
    };
    let mut ring: std::collections::VecDeque<Vec<[f32; 3]>> = std::collections::VecDeque::new();
    let mut sums = vec![0f64; masks.len()];
    let (mut compared, mut ms, mut timed, mut nan) = (0usize, 0f64, 0usize, 0usize);
    let mut max_change = vec![0f32; masks.len()];
    let mut mean_px = vec![0f32; (size[0] * size[1]) as usize];
    let mut mean_out = vec![0f32; (size[0] * size[1]) as usize];
    let mut measured = 0u32;
    let mut snaps = Vec::new();
    let mut last = Vec::new();
    GPU_LAST_MS.store(0, Ordering::Relaxed);
    for k in 0..n {
        let (cam, f) = make(k);
        let tex = upload_frame(gpu, &f, size)?;
        let view = View {
            cam: cam.eye,
            rot: cam.rot(),
            proj: [p00, p11, cam.jitter[0], cam.jitter[1]],
            depth: [NEAR, 0.0],
            region: [0, 0, size[0], size[1]],
            frame: k + 1,
            temporal: matches!(path, Path::After | Path::E085),
        };
        let recorded = g12.record(
            &list,
            u64::from(k),
            u64::from(k) + 1,
            ts_freq,
            &view,
            &inputs(&tex),
            sc,
            &knobs,
            &Mode::Live { budget },
            Box::new(()),
        )?;
        if !recorded {
            return Err(format!("frame {k} not recorded"));
        }
        gpu.submit()?;
        let out = read_rgba16f(gpu, &tex.color, size)?;
        nan += out
            .iter()
            .filter(|c| c.iter().any(|v| !v.is_finite()))
            .count();
        let refl: Vec<[f32; 3]> = out
            .iter()
            .zip(&tex.stored)
            .map(|(o, s)| [o[0] - s[0], o[1] - s[1], o[2] - s[2]])
            .collect();
        if k >= from && k % step == 0 && ring.len() == step as usize {
            for (m, mask) in masks.iter().enumerate() {
                sums[m] += rmse(&refl, &ring[0], mask);
                for (i, &on) in mask.iter().enumerate() {
                    if on {
                        let d = (luma(refl[i]) - luma(ring[0][i])).abs();
                        max_change[m] = max_change[m].max(d);
                    }
                }
            }
            compared += 1;
        }
        if k >= from {
            for (a, r) in mean_px.iter_mut().zip(&refl) {
                *a += luma(*r);
            }
            for (a, o) in mean_out.iter_mut().zip(&out) {
                *a += luma([o[0], o[1], o[2]]);
            }
            measured += 1;
        }
        // The timestamps of a frame are read when its ring slot is reused (SETS later).
        let t = f32::from_bits(GPU_LAST_MS.load(Ordering::Relaxed));
        if k >= from + SETS as u32 && t > 0.0 {
            ms += f64::from(t);
            timed += 1;
        }
        if snap.contains(&k) {
            snaps.push(refl.clone());
        }
        ring.push_back(refl);
        if ring.len() > step as usize {
            ring.pop_front();
        }
        last = out;
    }
    Ok(Seq {
        stability: sums.iter().map(|s| s / compared.max(1) as f64).collect(),
        max_change,
        mean_px: mean_px.iter().map(|v| v / measured.max(1) as f32).collect(),
        mean_out: mean_out
            .iter()
            .map(|v| v / measured.max(1) as f32)
            .collect(),
        snaps,
        last,
        ms: ms / timed.max(1) as f64,
        nan,
        stats: status().last,
    })
}

/// Pixels with a reflection weight in `f`, and those whose (mirror) ray hits geometry outside
/// the frame.
fn masks(f: &Frame, size: [u32; 2]) -> (Vec<bool>, Vec<bool>) {
    let half_w = size[0].div_ceil(2);
    let all: Vec<bool> = f.weight.iter().map(|w| w[3] > 0.0).collect();
    let off: Vec<bool> = (0..all.len())
        .map(|i| {
            let (x, y) = (i as u32 % size[0], i as u32 / size[0]);
            let q = ((y / 2) * half_w + x / 2) as usize;
            all[i]
                && !f.on_screen[q]
                && matches!(f.outcome[q], Some(Hit::Wall | Hit::Box | Hit::Floor))
        })
        .collect();
    (all, off)
}

/// Unbiased reference of the reflection term (output - input) on the floor with GGX `alpha`
/// seen by the pitched camera without jitter: a photo accumulation (linear, no clamps) after
/// a few live frames that settle the hit-shading calibration (on a copy of the inputs).
/// Returns it and its samples per texel.
fn photo_reference(
    gpu: &mut Gpu,
    sc: &SceneRef,
    size: [u32; 2],
    warp: bool,
    alpha: f32,
) -> Result<(Vec<[f32; 3]>, u32), String> {
    let cam = pitched(EYE, [0.0; 2]);
    let f = frame_cam(size, &cam, None, alpha);
    let warm = upload_frame(gpu, &f, size)?;
    let tex = upload_frame(gpu, &f, size)?;
    let half = [size[0].div_ceil(2), size[1].div_ceil(2)];
    let accum = texture(
        &gpu.device,
        half,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let device = gpu.device.clone();
    let mut g12 = Gpu12::new(&device)?;
    let list: ID3D12GraphicsCommandList = gpu.list.cast().map_err(|e| e.to_string())?;
    let (p00, p11) = proj(size);
    let view = View {
        cam: cam.eye,
        rot: cam.rot(),
        proj: [p00, p11, 0.0, 0.0],
        depth: [NEAR, 0.0],
        region: [0, 0, size[0], size[1]],
        frame: 1,
        temporal: false,
    };
    let knobs = Knobs {
        debug: 0,
        flags: F_SHADOW,
    };
    let mut fence = 0u64;
    for k in 0..SETS as u32 {
        g12.record(
            &list,
            fence,
            fence + 1,
            gpu.ts_freq as u64,
            &View {
                frame: k + 1,
                ..view
            },
            &inputs(&warm),
            sc,
            &knobs,
            &Mode::Live { budget: 100 },
            Box::new(()),
        )?;
        gpu.submit()?;
        fence += 1;
    }
    let calls = if warp { 4u32 } else { 8 };
    let spp = if warp { 16u32 } else { 32 };
    for c in 0..calls {
        // Composite only after the last call (the colour is not restored in between).
        let mode = Mode::Photo {
            accum: &accum,
            samples: spp,
            total: c * spp,
            composite: c + 1 == calls,
        };
        g12.record(
            &list,
            fence,
            fence + 1,
            gpu.ts_freq as u64,
            &view,
            &inputs(&tex),
            sc,
            &knobs,
            &mode,
            Box::new(()),
        )?;
        fence += 1;
        gpu.submit()?;
    }
    let out = read_rgba16f(gpu, &tex.color, size)?;
    Ok((
        out.iter()
            .zip(&tex.stored)
            .map(|(o, s)| [o[0] - s[0], o[1] - s[1], o[2] - s[2]])
            .collect(),
        calls * spp,
    ))
}

pub(super) fn check(
    gpu: &mut Gpu,
    sc: &SceneRef,
    size: [u32; 2],
    warp: bool,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
    images: &mut Vec<crate::ssr::RgbImage>,
) -> Result<(), String> {
    let (settle, measure) = if warp { (16u32, 8u32) } else { (48, 24) };
    let n = settle + measure;
    let still = |alpha: f32, jittered: bool| {
        move |k: u32| {
            let cam = pitched(EYE, if jittered { jitter(size, k) } else { [0.0; 2] });
            (cam, frame_cam(size, &cam, Some(&cam), alpha))
        }
    };
    let base = frame_cam(size, &pitched(EYE, [0.0; 2]), None, 0.0);
    let (all, off) = masks(&base, size);
    let n_off = off.iter().filter(|&&m| m).count();
    let mut nan = 0usize;
    // Below 2e-4 RMSE (an eighth of an 8-bit step at 1.0) a sequence counts as steady.
    let ratio_check =
        |failures: &mut Vec<String>, what: &str, before: f64, after: f64, limit: f64| {
            if after > 2e-4 && after > limit * before {
                failures.push(format!(
                    "RT denoiser {what}: after {after:.5} vs before {before:.5} (limit {limit} x)"
                ));
            }
        };
    report.push(format!(
        "  denoiser ({} frames settle + {measure} measured, pitched camera, weight {WEIGHT} on the floor: all ray-traced):",
        settle
    ));
    // Stability: mirror floor (off-screen geometry), glossy floor, glossy at a 25 % budget.
    for (name, alpha, budget, jittered) in [
        ("mirror floor", 0.0, 100, true),
        ("mirror floor", 0.0, 100, false),
        ("glossy floor (roughness 0.3)", GLOSSY, 100, true),
        ("glossy floor, ray budget 25 %", GLOSSY, 25, true),
    ] {
        let mut res = Vec::new();
        for path in [Path::Before, Path::After] {
            let s = sequence(
                gpu,
                sc,
                size,
                path,
                budget,
                n,
                settle,
                1,
                &[&all, &off],
                &[],
                &mut still(alpha, jittered),
            )?;
            nan += s.nan;
            res.push(s);
        }
        let (b, a) = (&res[0], &res[1]);
        let camera = if jittered {
            "static camera with jitter"
        } else {
            "static camera, no jitter"
        };
        report.push(format!(
            "    stability, {name}, {camera}: frame-to-frame RMSE of the reflection {:.5} -> {:.5} ({} px); off-screen hits {:.5} -> {:.5} ({n_off} px); shaded hits {} of {}, tile scales {:.2}..{:.2} (global {:.2}); GPU {:.3} -> {:.3} ms",
            b.stability[0],
            a.stability[0],
            all.iter().filter(|&&m| m).count(),
            b.stability[1],
            a.stability[1],
            a.stats.hits - a.stats.screen_hits,
            a.stats.hits,
            a.stats.tile_scale[0],
            a.stats.tile_scale[1],
            a.stats.scale,
            b.ms,
            a.ms
        ));
        // A mirror ray is deterministic: with jitter its reflection moves with the jittered
        // frame (as the input colour does; FSR resolves that), so the jittered mirror only has
        // to improve, the unjittered one to stay steady.
        let limit = if alpha == 0.0 && jittered { 0.75 } else { 0.5 };
        ratio_check(failures, name, b.stability[0], a.stability[0], limit);
        if alpha == 0.0 && jittered {
            ratio_check(
                failures,
                "mirror floor, off-screen hits",
                b.stability[1],
                a.stability[1],
                limit,
            );
            images.push(image("rt_denoise_mirror_before", &b.last, size));
            images.push(image("rt_denoise_mirror_after", &a.last, size));
        } else if budget == 100 && alpha > 0.0 {
            images.push(image("rt_denoise_glossy_before", &b.last, size));
            images.push(image("rt_denoise_glossy_after", &a.last, size));
        }
        if !a.stats.denoised {
            failures.push(format!("RT denoiser {name}: the denoiser did not run"));
        }
    }
    // Fireflies: ten small lamps (luminance ~35) on the wall reflected by the glossy floor,
    // jitter; the E085 denoiser vs now, and the per-hit clamp factor K around its default.
    // Lamp highlights: floor pixels where the E085 reflection adds > 0.05 luminance. Lamp
    // energy: the floor's mean reflection with the lamps minus without, against the same
    // difference of a linear photo accumulation (unbiased, no clamps).
    let mean_of = |px: &[f32], m: &[bool]| {
        let (s, c) = px
            .iter()
            .zip(m)
            .filter(|(_, on)| **on)
            .fold((0f64, 0usize), |(s, c), (v, _)| (s + f64::from(*v), c + 1));
        s / c.max(1) as f64
    };
    let run = |gpu: &mut Gpu, path: Path, masks: &[&[bool]], k: Option<f32>, jittered: bool| {
        set_offline_firefly_k(k);
        let s = sequence(
            gpu,
            sc,
            size,
            path,
            100,
            n,
            settle,
            1,
            masks,
            &[],
            &mut still(GLOSSY, jittered),
        );
        set_offline_firefly_k(None);
        s
    };
    let photo_mean = |gpu: &mut Gpu| -> Result<(f64, u32), String> {
        let (r, spp) = photo_reference(gpu, sc, size, warp, GLOSSY)?;
        let px: Vec<f32> = r.iter().map(|v| luma(*v)).collect();
        Ok((mean_of(&px, &all), spp))
    };
    // Without lamps.
    let base_e085 = run(gpu, Path::E085, &[&all], None, true)?;
    let base_now = run(gpu, Path::After, &[&all], None, true)?;
    let (ref_base, _) = photo_mean(gpu)?;
    nan += base_e085.nan + base_now.nan;
    LAMPS.store(true, Ordering::Relaxed);
    let probe = run(gpu, Path::E085, &[&all], None, true)?;
    nan += probe.nan;
    let hi: Vec<bool> = (0..all.len())
        .map(|i| all[i] && probe.mean_px[i] > 0.05)
        .collect();
    let n_hi = hi.iter().filter(|&&m| m).count();
    // (name, path, K override, jitter): the jitter-free rows show how much of the remaining
    // change is the engine's sub-pixel jitter moving the lamps on screen (FSR resolves that).
    let rows: [(&str, Path, Option<f32>, bool); 6] = [
        ("E085", Path::E085, None, true),
        ("K 4", Path::After, Some(4.0), true),
        ("K 8", Path::After, Some(8.0), true),
        ("K 16 (default)", Path::After, None, true),
        ("E085, no jitter", Path::E085, None, false),
        ("K 16, no jitter", Path::After, None, false),
    ];
    let mut fire = Vec::new();
    for (_, path, k, jittered) in rows {
        let s = run(gpu, path, &[&all, &hi], k, jittered)?;
        nan += s.nan;
        fire.push(s);
    }
    let (ref_lamps, fspp) = photo_mean(gpu)?;
    LAMPS.store(false, Ordering::Relaxed);
    let ref_energy = ref_lamps - ref_base;
    let lamp_energy = |i: usize| {
        let base = if rows[i].1 == Path::E085 {
            &base_e085
        } else {
            &base_now
        };
        mean_of(&fire[i].mean_px, &all) - mean_of(&base.mean_px, &all)
    };
    report.push(format!(
        "    fireflies (10 lamps of luminance ~35 on the wall, glossy floor, jitter; lamp highlights {n_hi} px; lamp energy of the {fspp}-sample linear reference {ref_energy:.5}):"
    ));
    for (i, (name, _, _, _)) in rows.iter().enumerate() {
        let s = &fire[i];
        report.push(format!(
            "      {name}: frame-to-frame RMSE floor {:.5}, highlights {:.5}; largest pixel change {:.3}; lamp energy {:.5} ({:.0} % of the reference); mean output floor {:.4}; GPU {:.3} ms",
            s.stability[0],
            s.stability[1],
            s.max_change[0],
            lamp_energy(i),
            100.0 * lamp_energy(i) / ref_energy.max(1e-9),
            mean_of(&s.mean_out, &all),
            s.ms
        ));
    }
    // Acceptance on the default: shimmer of the highlights halved without jitter (with it, the
    // lamps themselves move on screen every frame), no larger single-pixel jumps, the lamps'
    // energy within 15 % of the unbiased reference, the floor's mean output within 5 %. Not on
    // WARP: at its 256x144 smoke size a lamp is 1-2 px and 8 measured frames say nothing.
    let (b, a) = (&fire[0], &fire[3]);
    let (bn, an) = (&fire[4], &fire[5]);
    let energy = lamp_energy(3) / ref_energy.max(1e-9);
    let floor_ratio = mean_of(&a.mean_out, &all) / mean_of(&b.mean_out, &all).max(1e-9);
    if !warp
        && (n_hi < 20
            || an.stability[1] > 0.5 * bn.stability[1]
            || a.max_change[0] > b.max_change[0]
            || !(0.85..=1.15).contains(&energy)
            || !(0.95..=1.05).contains(&floor_ratio))
    {
        failures.push(format!(
            "RT fireflies: highlight RMSE without jitter {:.5} vs E085 {:.5}, max change {:.3} vs {:.3}, lamp energy {energy:.3} of the reference, floor output ratio {floor_ratio:.3}",
            an.stability[1], bn.stability[1], a.max_change[0], b.max_change[0]
        ));
    }
    images.push(image("rt_fireflies_before", &b.last, size));
    images.push(image("rt_fireflies_after", &a.last, size));
    // Hand-off: a band SSR covers on odd frames (no weight, no request there).
    let band = |y: u32| y >= size[1] * 6 / 10 && y < size[1] * 7 / 10;
    let band_mask: Vec<bool> = (0..all.len())
        .map(|i| all[i] && band(i as u32 / size[0]))
        .collect();
    let mut handoff = |k: u32| {
        let cam = pitched(EYE, [0.0; 2]);
        let mut f = frame_cam(size, &cam, Some(&cam), GLOSSY);
        if k % 2 == 1 {
            let half_w = size[0].div_ceil(2);
            for y in 0..size[1] {
                if band(y) {
                    for x in 0..size[0] {
                        f.weight[(y * size[0] + x) as usize] = [0.0; 4];
                        if y % 2 == 0 && x % 2 == 0 {
                            let q = ((y / 2) * half_w + x / 2) as usize;
                            f.req0[q] = [0.0; 4];
                            f.req1[q] = [0.0; 4];
                        }
                    }
                }
            }
        }
        (cam, f)
    };
    let mut hand = Vec::new();
    for path in [Path::Before, Path::After] {
        let s = sequence(
            gpu,
            sc,
            size,
            path,
            100,
            n,
            settle,
            2,
            &[&band_mask],
            &[],
            &mut handoff,
        )?;
        nan += s.nan;
        hand.push(s.stability[0]);
    }
    report.push(format!(
        "    hand-off (a band SSR covers every other frame), RMSE between consecutive frames needing rays: {:.5} -> {:.5}",
        hand[0], hand[1]
    ));
    ratio_check(failures, "hand-off band", hand[0], hand[1], 0.5);
    // Convergence: glossy floor, no jitter, against a photo accumulation.
    let reference = photo_reference(gpu, sc, size, warp, GLOSSY)?;
    let conv_frames = if warp { 16u32 } else { 40 };
    let mut conv = Vec::new();
    for path in [Path::Before, Path::After] {
        let s = sequence(
            gpu,
            sc,
            size,
            path,
            100,
            conv_frames,
            u32::MAX,
            1,
            &[],
            &[0, conv_frames - 1],
            &mut still(GLOSSY, false),
        )?;
        nan += s.nan;
        conv.push(s.snaps);
    }
    let e = |r: &Vec<[f32; 3]>| rmse(r, &reference.0, &all);
    report.push(format!(
        "    convergence, glossy floor, RMSE vs {}-sample reference: before (single frame, calibrated) {:.5}, after frame 1 {:.5}, after {conv_frames} frames {:.5}",
        reference.1,
        e(&conv[0][1]),
        e(&conv[1][0]),
        e(&conv[1][1])
    ));
    ratio_check(failures, "convergence", e(&conv[0][1]), e(&conv[1][1]), 0.6);
    // Ghosting: settle at pose 0, strafe 25 frames (motion vectors), compare at the final pose.
    let eye_at = |s: u32| [EYE[0], EYE[1] + STRAFE_STEP * s as f32, EYE[2]];
    let strafe_n = settle + STRAFE_FRAMES;
    let mut strafe = |k: u32| {
        let s = k.saturating_sub(settle);
        let cam = pitched(eye_at(s), [0.0; 2]);
        let prev = pitched(eye_at(s.saturating_sub(1)), [0.0; 2]);
        (cam, frame_cam(size, &cam, Some(&prev), GLOSSY))
    };
    let mut moving = Vec::new();
    for path in [Path::NoDenoise, Path::After] {
        let s = sequence(
            gpu,
            sc,
            size,
            path,
            100,
            strafe_n,
            u32::MAX,
            1,
            &[],
            &[strafe_n - 1],
            &mut strafe,
        )?;
        nan += s.nan;
        moving.push(s.snaps[0].clone());
    }
    let mut settled = Vec::new();
    for s in [0, STRAFE_FRAMES - 1] {
        let mut at = |_k: u32| {
            let cam = pitched(eye_at(s), [0.0; 2]);
            (cam, frame_cam(size, &cam, Some(&cam), GLOSSY))
        };
        let r = sequence(
            gpu,
            sc,
            size,
            Path::After,
            100,
            n,
            u32::MAX,
            1,
            &[],
            &[n - 1],
            &mut at,
        )?;
        nan += r.nan;
        settled.push(r.snaps[0].clone());
    }
    let final_frame = frame_cam(
        size,
        &pitched(eye_at(STRAFE_FRAMES - 1), [0.0; 2]),
        None,
        0.0,
    );
    let (final_all, _) = masks(&final_frame, size);
    // Trail: the share of the old pose's reflection still in the moving result, as the
    // projection of (moving - settled final) onto (settled initial - settled final) over the
    // pixels whose settled reflection changed (uncorrelated noise averages out).
    let mut trail_px = 0usize;
    let (mut num, mut den) = ([0f64; 2], 0f64);
    for i in 0..final_all.len() {
        if !final_all[i] {
            continue;
        }
        let d: [f32; 3] = std::array::from_fn(|c| settled[0][i][c] - settled[1][i][c]);
        if d.iter().map(|v| v.abs()).sum::<f32>() > 0.06 {
            trail_px += 1;
            for c in 0..3 {
                den += f64::from(d[c] * d[c]);
                for (m, mv) in moving.iter().enumerate() {
                    num[m] += f64::from((mv[i][c] - settled[1][i][c]) * d[c]);
                }
            }
        }
    }
    let residual = num.map(|v| v / den.max(1e-9));
    let (m_nd, m_after) = (
        rmse(&moving[0], &settled[1], &final_all),
        rmse(&moving[1], &settled[1], &final_all),
    );
    report.push(format!(
        "    ghosting ({STRAFE_FRAMES} frames x {STRAFE_STEP} m strafe): RMSE at the final pose vs its settled result: denoised {m_after:.5}, no denoiser {m_nd:.5}; trail ({trail_px} px whose settled reflection changed): old reflection left {:.1} % (no denoiser {:.1} %)",
        residual[1] * 100.0,
        residual[0] * 100.0
    ));
    if m_after > 1.1 * m_nd || residual[1] > 0.1 {
        failures.push(format!(
            "RT denoiser ghosting: moving {m_after:.5} vs no denoiser {m_nd:.5}, trail residual {:.1} %",
            residual[1] * 100.0
        ));
    }
    report.push(format!("    non-finite pixels over all sequences: {nan}"));
    if nan > 0 {
        failures.push(format!("RT denoiser: {nan} non-finite pixels"));
    }
    Ok(())
}
