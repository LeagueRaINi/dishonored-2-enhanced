//! Offline checks of surface coverage and of the temporal stage (`ssr_offline`, scaled case):
//! the scene of `offline.rs` with a material target and engine-like motion vectors.
//!
//! - Coverage (single frames, `ssr_temporal = 0`): a vertical panel with a gold, a copper and
//!   a polished dielectric section reflects (with and without `ssr_depth_floors_only`), the
//!   metals tinted by their coloured F0 (o3 target), the dielectric neutral; its rough section,
//!   a floor patch carrying the exclusion marker (vegetation) and a no-material patch with
//!   noisy depth (foliage seen by the depth fallback) stay bit-identical even with
//!   `ssr_depth_floors_only = 0`.
//! - Noise: RMSE over the glossy floor (roughness 0.3) against a reference = the mean of 64
//!   frames of the per-frame estimate with the rotated sample pattern (no accumulation):
//!   before (no temporal stage, fixed pattern) and after 40 accumulated frames; GPU times.
//! - Ghosting: the camera moves 4 cm per frame for 25 frames (motion vectors as the engine
//!   writes them: current - previous viewport UV); RMSE of the last frame against the reference
//!   at the final pose, and the residual of the content that moved (trail pixels: references
//!   at the start and the end pose differ) relative to that difference.

use super::*;

const GLOSSY: (f32, f32) = (0.3, 0.3);
const MIRROR: (f32, f32) = (0.05, 0.25);
const REFERENCE_FRAMES: usize = 64;
const ACCUMULATE_FRAMES: usize = 40;
const MOVE_FRAMES: usize = 25;
const MOVE_STEP: f32 = 0.04;
/// Weapon stand-in material: glossy steel (perceptual roughness, F0).
const BLADE_MATERIAL: (f32, f32) = (0.15, 0.6);
/// Frames before the stability measurement and frames measured.
const SETTLE_FRAMES: usize = 72;
const MEASURE_FRAMES: usize = 24;

/// View-frame normal -> world (the transpose of `view_rotation(view_matrix())`).
fn world_normal(n: [f32; 3]) -> [f32; 3] {
    let (s, c) = YAW.sin_cos();
    let (right, up, fwd) = ([s, -c, 0.0], [0.0, 0.0, 1.0], [c, s, 0.0]);
    std::array::from_fn(|i| n[0] * right[i] + n[1] * up[i] + n[2] * fwd[i])
}

/// Material texel as the patched lit shaders write it (`patch.rs`): octahedral world normal,
/// perceptual roughness, max(sqrt(F0), 2/255); `None` = the exclusion marker (alpha byte 1).
fn texel(view_normal: [f32; 3], m: Option<(f32, f32)>) -> u32 {
    let n = world_normal(view_normal);
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y) = (n[0] / l1, n[1] / l1);
    let (x, y) = if n[2] >= 0.0 {
        (x, y)
    } else {
        ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum())
    };
    let b = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
    let (r, a) = match m {
        Some((rough, f0)) => (
            (rough * 255.0).round() as u8,
            (f0.sqrt() * 255.0).round().max(2.0) as u8,
        ),
        None => (255, 1),
    };
    u32::from_le_bytes([b(x), b(y), r, a])
}

fn in_rect(v: f32, r: [f32; 2]) -> bool {
    (r[0]..=r[1]).contains(&v)
}

/// Coloured F0 texel (`patch.rs` o3): sqrt(F0.rgb), alpha 1 = RGB captured.
fn f0_texel(f0: [f32; 3]) -> u32 {
    let b = |v: f32| (v.sqrt() * 255.0).round() as u8;
    u32::from_le_bytes([b(f0[0]), b(f0[1]), b(f0[2]), 255])
}

/// Panel sections along z: gold 6..7, copper 7..8, polished dielectric 8..9, rough 9..10.
const GOLD: [f32; 3] = [1.0, 0.78, 0.34];
const COPPER: [f32; 3] = [0.95, 0.64, 0.54];
const DIELECTRIC: [f32; 3] = [0.04, 0.04, 0.04];

/// What a test pixel is (region pixels of the scene seen from x = `cx`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Glossy,
    Leaves,
    Gold,
    Copper,
    Dielectric,
    RoughPanel,
    Bush,
    /// The first-person weapon stand-in (`offline.rs` `Surf::Blade`).
    Blade,
    /// The thin pole (`offline.rs` `Surf::Pole`).
    Pole,
    Other,
}

fn kind(sc: &Scene, i: usize, cx: f32) -> Kind {
    let p = sc.pos[i];
    match sc.surf[i] {
        Some(Surf::Floor) if in_rect(p[0] + cx, LEAVES_X) && in_rect(p[2], LEAVES_Z) => {
            Kind::Leaves
        }
        Some(Surf::Floor) => Kind::Glossy,
        Some(Surf::Panel) if p[2] < 7.0 => Kind::Gold,
        Some(Surf::Panel) if p[2] < PANEL_SPLIT => Kind::Copper,
        Some(Surf::Panel) if p[2] < 9.0 => Kind::Dielectric,
        Some(Surf::Panel) => Kind::RoughPanel,
        Some(Surf::Bush) => Kind::Bush,
        Some(Surf::Blade) => Kind::Blade,
        Some(Surf::Pole) => Kind::Pole,
        _ => Kind::Other,
    }
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Material target of the scene: the floor `floor` (roughness, F0), the leaves patch
/// excluded, the panel gold / copper (roughness 0.08) / polished dielectric (0.05, 0.04) /
/// rough (0.8, 0.04); box, wall, bush, water and sky without material. Grey F0 = luminance as
/// the patch writes it.
fn material_data(sc: &Scene, size: [u32; 2], cx: f32, floor: (f32, f32)) -> Vec<u32> {
    let side = [-1.0, 0.0, 0.0];
    (0..sc.surf.len())
        .map(|i| match kind(sc, i, cx) {
            Kind::Glossy => texel([0.0, 1.0, 0.0], Some(floor)),
            Kind::Leaves => texel([0.0, 1.0, 0.0], None),
            Kind::Gold => texel(side, Some((0.08, luma(GOLD)))),
            Kind::Copper => texel(side, Some((0.08, luma(COPPER)))),
            Kind::Dielectric => texel(side, Some((0.05, 0.04))),
            Kind::RoughPanel => texel(side, Some((0.8, 0.04))),
            Kind::Blade => texel([0.0, 1.0, 0.0], Some(BLADE_MATERIAL)),
            Kind::Pole => texel([0.0, 0.0, -1.0], Some((0.05, 0.9))),
            Kind::Bush | Kind::Other => 0,
        })
        .take((size[0] * size[1]) as usize)
        .collect()
}

/// Coloured F0 target (o3) of the scene: the panel's metals and dielectric, grey F0 for the
/// floor.
fn f0_data(sc: &Scene, cx: f32, floor: (f32, f32)) -> Vec<u32> {
    (0..sc.surf.len())
        .map(|i| match kind(sc, i, cx) {
            Kind::Glossy => f0_texel([floor.1; 3]),
            Kind::Gold => f0_texel(GOLD),
            Kind::Copper => f0_texel(COPPER),
            Kind::Dielectric | Kind::RoughPanel => f0_texel(DIELECTRIC),
            Kind::Blade => f0_texel([BLADE_MATERIAL.1; 3]),
            Kind::Pole => f0_texel([0.9; 3]),
            _ => 0,
        })
        .collect()
}

/// Engine motion vectors (current - previous viewport UV, R16G16_FLOAT) for a camera that
/// moved by `dx` along view x since the previous frame (static scene); the weapon stand-in moves
/// with the camera and by its own sway `blade_dv` (view frame) since the previous frame.
fn motion_data(p: &Proj, sc: &Scene, dx: f32, blade_dv: [f32; 2]) -> Vec<u32> {
    let uv = |v: [f32; 3]| {
        let ndc = [
            v[0] / v[2] * p.p00 + p.jitter[0],
            v[1] / v[2] * p.p11 + p.jitter[1],
        ];
        [(ndc[0] + 1.0) * 0.5, (1.0 - ndc[1]) * 0.5]
    };
    sc.pos
        .iter()
        .zip(&sc.surf)
        .map(|(v, s)| {
            if s.is_none() || !v.iter().all(|c| c.is_finite()) {
                return 0;
            }
            let prev = if *s == Some(Surf::Blade) {
                [v[0] - blade_dv[0], v[1] - blade_dv[1], v[2]]
            } else {
                [v[0] + dx, v[1], v[2]]
            };
            if prev == *v {
                return 0;
            }
            let (a, b) = (uv(*v), uv(prev));
            u32::from(f16_bits(a[0] - b[0])) | (u32::from(f16_bits(a[1] - b[1])) << 16)
        })
        .collect()
}

/// Root-mean-square error over `mask` pixels.
fn rmse(a: &[[f32; 3]], b: &[[f64; 3]], mask: &[usize]) -> f64 {
    let s: f64 = mask
        .iter()
        .map(|&i| {
            (0..3)
                .map(|k| (f64::from(a[i][k]) - b[i][k]).powi(2))
                .sum::<f64>()
                / 3.0
        })
        .sum();
    (s / mask.len().max(1) as f64).sqrt()
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// The textures one frame of the scene goes into.
struct Frame<'a> {
    g: &'a Gpu,
    probe: &'a Probe,
    e: &'a Engine,
    color: &'a ID3D11Texture2D,
    color_srv: &'a ID3D11ShaderResourceView,
    depth: &'a ID3D11Texture2D,
    water: &'a ID3D11Texture2D,
    material: ID3D11Resource,
    f0: ID3D11Resource,
    motion: &'a ID3D11Texture2D,
    proj: &'a Proj,
    fmt: &'a Format,
    size: [u32; 2],
    region: [u32; 2],
}

impl Frame<'_> {
    /// Uploads the scene from `cx` (camera moved by `dx` since the last frame) and its
    /// material; returns it with its encoded colour.
    fn upload(&self, cx: f32, dx: f32, floor: (f32, f32)) -> (Scene, Vec<u8>) {
        self.upload_with(cx, dx, floor, None)
    }

    /// As [`Frame::upload`] with the weapon stand-in at sway `blade.0`, previous sway
    /// `blade.1`.
    fn upload_with(
        &self,
        cx: f32,
        dx: f32,
        floor: (f32, f32),
        blade: Option<([f32; 2], [f32; 2])>,
    ) -> (Scene, Vec<u8>) {
        let sc = scene(self.proj, self.size, self.region, cx, blade.map(|b| b.0));
        let mat = material_data(&sc, self.size, cx, floor);
        let f0 = f0_data(&sc, cx, floor);
        let dv = blade.map_or([0.0; 2], |(s, p)| [s[0] - p[0], s[1] - p[1]]);
        let mv = motion_data(self.proj, &sc, dx, dv);
        let pitch = self.size[0] * 4;
        // SAFETY: full-texture updates of 4-byte-per-texel textures of `size` with matching
        // pitch.
        unsafe {
            let imm = &self.g.imm;
            imm.UpdateSubresource(self.depth, 0, None, sc.depth.as_ptr().cast(), pitch, 0);
            imm.UpdateSubresource(self.water, 0, None, sc.water[0].as_ptr().cast(), pitch, 0);
            imm.UpdateSubresource(&self.material, 0, None, mat.as_ptr().cast(), pitch, 0);
            imm.UpdateSubresource(&self.f0, 0, None, f0.as_ptr().cast(), pitch, 0);
            imm.UpdateSubresource(self.motion, 0, None, mv.as_ptr().cast(), pitch, 0);
        }
        let initial = sc.color.iter().flat_map(|c| self.fmt.encode(*c)).collect();
        (sc, initial)
    }

    fn run(&self, initial: &[u8], time: bool) -> Result<Outcome, String> {
        run_engine(
            self.g,
            self.probe,
            self.e,
            self.color,
            self.color_srv,
            initial,
            self.fmt,
            self.size,
            self.region,
            time,
        )
    }

    /// Mean of `REFERENCE_FRAMES` frames of the per-frame estimate with the rotated pattern.
    fn reference(&self, initial: &[u8]) -> Result<Vec<[f64; 3]>, String> {
        OFFLINE_ROTATE_ONLY.store(true, Ordering::Relaxed);
        let mut acc = vec![[0f64; 3]; (self.size[0] * self.size[1]) as usize];
        let r = (|| {
            for _ in 0..REFERENCE_FRAMES {
                let o = self.run(initial, false)?;
                for (a, p) in acc.iter_mut().zip(&o.pixels) {
                    for k in 0..3 {
                        a[k] += f64::from(p[k]) / REFERENCE_FRAMES as f64;
                    }
                }
            }
            Ok(())
        })();
        OFFLINE_ROTATE_ONLY.store(false, Ordering::Relaxed);
        r.map(|()| acc)
    }
}

fn set(temporal: bool, floors_only: bool) {
    settings::edit(|f| {
        *f = Settings::default();
        f.ssr = true;
        f.ssr_temporal = temporal;
        f.ssr_depth_floors_only = floors_only;
    });
}

/// History of the next frame starts over (the capture counter skips one).
fn break_history() {
    FRAME.fetch_add(1, Ordering::Relaxed);
}

fn image_of(name: &str, px: &[[f32; 3]], size: [u32; 2], region: [u32; 2]) -> RgbImage {
    image(name.to_owned(), px, size, region)
}

fn image_f64(name: &str, px: &[[f64; 3]], size: [u32; 2], region: [u32; 2]) -> RgbImage {
    let f: Vec<[f32; 3]> = px.iter().map(|c| c.map(|v| v as f32)).collect();
    image(name.to_owned(), &f, size, region)
}

/// Silhouette halo ("faint glowy outline", user): the thin polished pole faces the camera, so
/// its mirror reflection points back at the camera and finds nothing on screen; its 1-2 px
/// silhouette pixels have depth normals from the far background step (noise). E074 used that
/// depth normal with the material's mirror roughness and F0 0.9 and reflected the floor/wall
/// there (a bright outline); now non-planar disagreeing pixels do not reflect. Static camera,
/// temporal on, after 8 frames: pole pixels brightened by more than 0.01 (luminance) and the
/// mean brightening, E074 vs now.
fn halo_check(
    f: &Frame,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let mut rows = Vec::new();
    for (name, legacy) in [("E074", true), ("now", false)] {
        set(true, false);
        OFFLINE_LEGACY.store(legacy, Ordering::Relaxed);
        break_history();
        let mut last = None;
        for _ in 0..8 {
            let (sc, initial) = f.upload_with(0.0, 0.0, MIRROR, Some(([0.0; 2], [0.0; 2])));
            let o = f.run(&initial, false)?;
            last = Some((sc, initial, o.pixels));
        }
        OFFLINE_LEGACY.store(false, Ordering::Relaxed);
        let (sc, initial, out) = last.ok_or("no frames")?;
        let input: Vec<[f32; 3]> = initial
            .chunks_exact(f.fmt.bpp)
            .map(|b| f.fmt.decode(b))
            .collect();
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let pole: Vec<usize> = (0..sc.surf.len())
            .filter(|&i| sc.surf[i] == Some(Surf::Pole))
            .collect();
        let bright = pole
            .iter()
            .filter(|&&i| lum(out[i]) - lum(input[i]) > 0.01)
            .count();
        let mean = pole
            .iter()
            .map(|&i| f64::from(lum(out[i]) - lum(input[i])))
            .sum::<f64>()
            / pole.len().max(1) as f64;
        rows.push((name, bright, pole.len(), mean));
    }
    report.push(format!(
        "  silhouette halo (thin polished pole facing the camera, 1-2 px wide): {}",
        rows.iter()
            .map(|(n, b, t, m)| format!("{n}: {b} of {t} px brightened > 0.01, mean {m:+.4}"))
            .collect::<Vec<_>>()
            .join("; ")
    ));
    let now = rows[1];
    if now.2 < 100 || now.1 * 100 > now.2 {
        failures.push(format!(
            "halo: {} of {} pole pixels brightened",
            now.1, now.2
        ));
    }
    Ok(())
}

/// Weapon stand-in idle sway (view frame, metres) at frame `k`: a few pixels per frame at the
/// stand-in's distance, as the first-person hands' idle animation.
fn sway(k: usize) -> [f32; 2] {
    let t = k as f32;
    [0.0015 * (t * 0.35).sin(), 0.0008 * (t * 0.23).sin()]
}

/// Frame-to-frame RMSE of the reflection contribution (output - input) over `mask`, averaged
/// over consecutive frame pairs.
fn flicker(frames: &[Vec<[f32; 3]>], inputs: &[Vec<[f32; 3]>], mask: &[usize]) -> f64 {
    let mut sum = 0f64;
    for k in 1..frames.len() {
        let d: f64 = mask
            .iter()
            .map(|&i| {
                (0..3)
                    .map(|c| {
                        let a = f64::from(frames[k][i][c] - inputs[k][i][c]);
                        let b = f64::from(frames[k - 1][i][c] - inputs[k - 1][i][c]);
                        (a - b).powi(2)
                    })
                    .sum::<f64>()
                    / 3.0
            })
            .sum();
        sum += (d / mask.len().max(1) as f64).sqrt();
    }
    sum / (frames.len() - 1).max(1) as f64
}

/// Static camera, after `SETTLE_FRAMES`: frame-to-frame flicker of the reflection on a mirror
/// floor (roughness 0.05), a glossy floor (0.3) and the swaying first-person weapon stand-in
/// (glossy steel, own motion vectors, depth-hacked), for temporal off (fixed pattern), the
/// E074 temporal parameters and the current ones.
fn stability_check(
    f: &Frame,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    // (mode name, temporal, legacy)
    let modes = [
        ("off", false, false),
        ("E074", true, true),
        ("now", true, false),
    ];
    let mut rows = Vec::new();
    for (scenario, floor) in [
        ("mirror floor", MIRROR),
        ("glossy floor", GLOSSY),
        ("weapon", MIRROR),
    ] {
        let weapon = scenario == "weapon";
        let mut vals = Vec::new();
        for (name, temporal, legacy) in modes {
            set(temporal, true);
            OFFLINE_LEGACY.store(legacy, Ordering::Relaxed);
            break_history();
            let (mut frames, mut inputs) = (Vec::new(), Vec::new());
            let mut mask_kind = Kind::Glossy;
            let mut last_sc = None;
            for k in 0..SETTLE_FRAMES + MEASURE_FRAMES {
                let blade = weapon.then(|| (sway(k), sway(k.saturating_sub(1))));
                let (sc, initial) = f.upload_with(0.0, 0.0, floor, blade);
                let o = f.run(&initial, false)?;
                if k >= SETTLE_FRAMES {
                    inputs.push(
                        initial
                            .chunks_exact(f.fmt.bpp)
                            .map(|b| f.fmt.decode(b))
                            .collect::<Vec<_>>(),
                    );
                    frames.push(o.pixels);
                    last_sc = Some(sc);
                }
            }
            OFFLINE_LEGACY.store(false, Ordering::Relaxed);
            let sc = last_sc.ok_or("no frames")?;
            if weapon {
                mask_kind = Kind::Blade;
            }
            // Pixels of that kind in every measured frame (the weapon moves: its interior).
            let mask: Vec<usize> = (0..sc.surf.len())
                .filter(|&i| {
                    kind(&sc, i, 0.0) == mask_kind
                        && (!weapon || inputs.iter().all(|inp| (inp[i][0] - PANEL[0]).abs() < 1e-3))
                })
                .collect();
            // Mean history length over the mask (the overlay's `history` view).
            let mut mean_len = 0.0;
            if temporal {
                let g2 = lock(&PIPE);
                if let Some((p, (s, _))) = g2.as_ref().and_then(|p| p.history.map(|h| (p, h)))
                    && let Some(t) = p.targets.as_ref()
                {
                    // SAFETY: plain getter.
                    let res = unsafe { t.meta[s].srv.GetResource() }.map_err(|e| e.to_string())?;
                    let tex: ID3D11Texture2D = res.cast().map_err(|e| e.to_string())?;
                    let raw = read_raw(f.g, &tex, f.region, 4)?;
                    let sum: f64 = mask
                        .iter()
                        .map(|&i| {
                            let (x, y) = (i as u32 % f.size[0], i as u32 / f.size[0]);
                            f64::from(raw[((y * f.region[0] + x) * 4 + 2) as usize])
                        })
                        .sum();
                    mean_len = sum / mask.len().max(1) as f64;
                }
            }
            let fl = flicker(&frames, &inputs, &mask);
            vals.push((name, fl, mask.len(), mean_len));
        }
        rows.push((scenario, vals));
    }
    for (scenario, vals) in &rows {
        report.push(format!(
            "  stability ({scenario}, static camera, frames {SETTLE_FRAMES}..{}): frame-to-frame RMSE of the reflection: {}",
            SETTLE_FRAMES + MEASURE_FRAMES,
            vals.iter()
                .map(|(n, v, px, len)| {
                    if *len > 0.0 {
                        format!("{n} {v:.5} (mean history {len:.1} frames)")
                    } else {
                        format!("{n} {v:.5} ({px} px)")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let get = |n: &str| vals.iter().find(|v| v.0 == n).map_or(f64::NAN, |v| v.1);
        let (off, legacy, now) = (get("off"), get("E074"), get("now"));
        // At least 2x steadier than E074 on every case.
        let steadier = now <= legacy * 0.5;
        if vals.iter().any(|v| v.2 < 200) || !steadier {
            failures.push(format!(
                "stability {scenario}: now {now:.5} vs E074 {legacy:.5}, temporal off {off:.5}"
            ));
        }
    }
    Ok(())
}

/// Reflection term (output - input) per pixel.
fn reflection(out: &[[f32; 3]], input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    out.iter()
        .zip(input)
        .map(|(o, i)| [o[0] - i[0], o[1] - i[1], o[2] - i[2]])
        .collect()
}

/// One static sequence of `SETTLE_FRAMES` + `MEASURE_FRAMES` on `initial`: the reflection
/// terms of the measured frames. `no_fire`: the E085 path; `k`: firefly clamp override.
fn firefly_frames(
    f: &Frame,
    initial: &[u8],
    input: &[[f32; 3]],
    no_fire: bool,
    k: Option<f32>,
) -> Result<Vec<Vec<[f32; 3]>>, String> {
    set(true, true);
    OFFLINE_NO_FIREFLY.store(no_fire, Ordering::Relaxed);
    rt_reflect::set_offline_firefly_k(k);
    break_history();
    let r = (|| {
        let mut frames = Vec::new();
        for i in 0..SETTLE_FRAMES + MEASURE_FRAMES {
            let o = f.run(initial, false)?;
            if i >= SETTLE_FRAMES {
                frames.push(reflection(&o.pixels, input));
            }
        }
        Ok(frames)
    })();
    OFFLINE_NO_FIREFLY.store(false, Ordering::Relaxed);
    rt_reflect::set_offline_firefly_k(None);
    r
}

/// Mean luminance of `px` over `mask` (per-frame values averaged over `frames`).
fn mean_luma(frames: &[Vec<[f32; 3]>], mask: &[usize]) -> f64 {
    frames
        .iter()
        .map(|fr| mask.iter().map(|&i| f64::from(luma(fr[i]))).sum::<f64>())
        .sum::<f64>()
        / (frames.len() * mask.len()).max(1) as f64
}

/// Fireflies: ten 0.1 m lamps (luminance ~35) on the back wall, reflected by the glossy floor
/// (roughness 0.3), static camera, after `SETTLE_FRAMES`. E085 (linear resolve, history
/// clamp to the frame's 3x3 only) vs now (per-hit clamp to K x the neighbourhood mean, history
/// clamp widened by the luminance's temporal deviation) for K around its default:
/// frame-to-frame RMSE and largest per-pixel luminance change of the reflection on the floor
/// and on its lamp highlights, and the lamps' energy (the floor's mean reflection with minus
/// without the lamps) against the same difference of the unbiased reference (64-frame mean of
/// the linear per-frame estimate).
fn firefly_check(
    f: &Frame,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let r = (|| -> Result<(), String> {
        let upload = |lamps: bool| {
            LAMPS.store(lamps, Ordering::Relaxed);
            let (sc, initial) = f.upload(0.0, 0.0, GLOSSY);
            let input: Vec<[f32; 3]> = initial
                .chunks_exact(f.fmt.bpp)
                .map(|b| f.fmt.decode(b))
                .collect();
            (sc, initial, input)
        };
        let reference = |initial: &[u8], input: &[[f32; 3]]| -> Result<Vec<[f64; 3]>, String> {
            set(true, true);
            OFFLINE_NO_FIREFLY.store(true, Ordering::Relaxed);
            let r = f.reference(initial);
            OFFLINE_NO_FIREFLY.store(false, Ordering::Relaxed);
            Ok(r?
                .iter()
                .zip(input)
                .map(|(r, i)| std::array::from_fn(|c| r[c] - f64::from(i[c])))
                .collect())
        };
        // Without lamps: reference and the two paths' floor reflection.
        let (sc, initial0, input0) = upload(false);
        let floor: Vec<usize> = (0..sc.surf.len())
            .filter(|&i| kind(&sc, i, 0.0) == Kind::Glossy)
            .collect();
        let ref0 = reference(&initial0, &input0)?;
        let base_e085 = mean_luma(&firefly_frames(f, &initial0, &input0, true, None)?, &floor);
        let base_now = mean_luma(&firefly_frames(f, &initial0, &input0, false, None)?, &floor);
        // With lamps.
        let (_, initial, input) = upload(true);
        let ref1 = reference(&initial, &input)?;
        let lamp_ref: Vec<f64> = ref1
            .iter()
            .zip(&ref0)
            .map(|(a, b)| lum(*a) - lum(*b))
            .collect();
        let ref_energy =
            floor.iter().map(|&i| lamp_ref[i]).sum::<f64>() / floor.len().max(1) as f64;
        // Lamp highlights: floor pixels where the lamps add > 0.05 luminance (reference).
        let highlight: Vec<usize> = floor
            .iter()
            .copied()
            .filter(|&i| lamp_ref[i] > 0.05)
            .collect();
        let rows: [(&str, bool, Option<f32>); 4] = [
            ("E085", true, None),
            ("K 4", false, Some(4.0)),
            ("K 8", false, Some(8.0)),
            ("K 16 (default)", false, None),
        ];
        let zero = vec![vec![[0f32; 3]; input.len()]; MEASURE_FRAMES];
        let mut vals = Vec::new();
        for (name, no_fire, k) in rows {
            let frames = firefly_frames(f, &initial, &input, no_fire, k)?;
            let mut max_change = 0f32;
            for j in 1..frames.len() {
                for &i in &floor {
                    max_change =
                        max_change.max((luma(frames[j][i]) - luma(frames[j - 1][i])).abs());
                }
            }
            let base = if no_fire { base_e085 } else { base_now };
            let energy = mean_luma(&frames, &floor) - base;
            vals.push((
                name,
                flicker(&frames, &zero, &floor),
                flicker(&frames, &zero, &highlight),
                max_change,
                energy,
                mean_luma(&frames, &floor),
            ));
        }
        report.push(format!(
            "  fireflies (10 lamps of luminance ~35 on the wall, glossy floor 0.3, static camera, frames {SETTLE_FRAMES}..{}; floor {} px, lamp highlights {} px; lamp energy of the 64-frame linear reference {ref_energy:.5}):",
            SETTLE_FRAMES + MEASURE_FRAMES,
            floor.len(),
            highlight.len()
        ));
        for v in &vals {
            report.push(format!(
                "    {}: frame-to-frame RMSE floor {:.5}, highlights {:.5}; largest pixel change {:.3}; lamp energy {:.5} ({:.0} % of the reference); mean floor reflection {:.4}",
                v.0,
                v.1,
                v.2,
                v.3,
                v.4,
                100.0 * v.4 / ref_energy.max(1e-9),
                v.5
            ));
        }
        // Acceptance on the default: highlight shimmer halved, single-pixel jumps below 0.6 x,
        // the lamps' energy within 15 % of the unbiased reference.
        let (b, a) = (&vals[0], &vals[3]);
        let energy = a.4 / ref_energy.max(1e-9);
        if highlight.len() < 50
            || a.2 > 0.5 * b.2
            || a.3 > 0.6 * b.3
            || !(0.85..=1.15).contains(&energy)
        {
            failures.push(format!(
                "fireflies: highlights {} px, RMSE {:.5} vs E085 {:.5}, max change {:.3} vs {:.3}, lamp energy {energy:.3} of the reference",
                highlight.len(),
                a.2,
                b.2,
                a.3,
                b.3
            ));
        }
        Ok(())
    })();
    LAMPS.store(false, Ordering::Relaxed);
    OFFLINE_NO_FIREFLY.store(false, Ordering::Relaxed);
    r
}

/// The D3D11 side of the hybrid ray-traced reflections (`cs_rt_classify` on the FSR path, into
/// plain D3D11 targets via `rt_reflect::set_offline_targets`) on the coverage scene: weights only
/// on reflective material/floor pixels (none on the rough panel, the excluded leaves, the
/// noisy-depth bush), the gold section's weights tinted like the composite, requests from the
/// pixel position (pushed off the surface) with unit normals, the floor's normal straight up
/// and its alpha (roughness 0.05).
fn classify_check(
    f: &Frame,
    sc: &Scene,
    initial: &[u8],
    kinds: &[Kind],
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let g = f.g;
    let region = f.region;
    let half = [region[0].div_ceil(2), region[1].div_ceil(2)];
    let bind = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0;
    let f16 = DXGI_FORMAT_R16G16B16A16_FLOAT;
    let f32x4 = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let (rtw, req0, req1, depth) = (
        texture(&g.dev, region, f16, 1, bind)?,
        texture(&g.dev, half, f32x4, 1, bind)?,
        texture(&g.dev, half, f16, 1, bind)?,
        texture(&g.dev, region, DXGI_FORMAT_R32_FLOAT, 1, bind)?,
    );
    rt_reflect::set_offline_targets(Some(rt_reflect::Classify {
        rtw: uav(&g.dev, &rtw, f16, 0)?,
        req0: uav(&g.dev, &req0, f32x4, 0)?,
        req1: uav(&g.dev, &req1, f16, 0)?,
        depth: uav(&g.dev, &depth, DXGI_FORMAT_R32_FLOAT, 0)?,
        params: [1.0, 0.5, 40.0, 0.0],
    }));
    let r = (|| -> Result<(), String> {
        set(false, true);
        let pitch = f.size[0] * f.fmt.bpp as u32;
        // SAFETY: full-texture update with data of the texture's size and pitch.
        unsafe {
            g.imm
                .UpdateSubresource(f.color, 0, None, initial.as_ptr().cast(), pitch, 0)
        };
        let job = capture(f.e.pass(), f.e.motion.clone()).ok_or("nothing captured")?;
        execute(&g.imm, job);
        let half4 = |b: &[u8; 8]| -> [f32; 4] {
            std::array::from_fn(|c| f16_value(u16::from_le_bytes([b[2 * c], b[2 * c + 1]])))
        };
        let w: Vec<[f32; 4]> = read_raw(g, &rtw, region, 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(half4)
            .collect();
        let r0: Vec<[f32; 4]> = read_raw(g, &req0, half, 16)?
            .as_chunks::<16>()
            .0
            .iter()
            .map(|b| {
                std::array::from_fn(|c| {
                    f32::from_le_bytes([b[4 * c], b[4 * c + 1], b[4 * c + 2], b[4 * c + 3]])
                })
            })
            .collect();
        let r1: Vec<[f32; 4]> = read_raw(g, &req1, half, 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(half4)
            .collect();
        // Depth copy: the R24 hardware depth as R32F.
        let dcopy: Vec<f32> = read_raw(g, &depth, region, 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let depth_bad = (0..region[1])
            .flat_map(|y| (0..region[0]).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let d24 =
                    (sc.depth[(y * f.size[0] + x) as usize] & 0x00FF_FFFF) as f32 / 16_777_215.0;
                (dcopy[(y * region[0] + x) as usize] - d24).abs() > 1e-6
            })
            .count();
        // Weights per kind (region pixels).
        let mut weighted = [0usize; 10];
        let mut gold = [0f64; 3];
        for y in 0..region[1] {
            for x in 0..region[0] {
                let i = (y * f.size[0] + x) as usize;
                let v = w[(y * region[0] + x) as usize];
                if v[3] > 0.0 {
                    weighted[kinds[i] as usize] += 1;
                    if kinds[i] == Kind::Gold {
                        for c in 0..3 {
                            gold[c] += f64::from(v[c] / v[3]);
                        }
                    }
                }
            }
        }
        let gold_tint = [gold[0] / gold[2].max(1e-9), gold[1] / gold[2].max(1e-9)];
        // Requests: origin = pixel position + normal bias, unit directions, mirror on the floor.
        let (mut requests, mut bad_origin, mut bad_dir, mut floor_req, mut floor_mirror) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for qy in 0..half[1] {
            for qx in 0..half[0] {
                let q = (qy * half[0] + qx) as usize;
                if r0[q][3] <= 0.0 {
                    continue;
                }
                requests += 1;
                let d = r1[q];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                bad_dir += usize::from((len - 1.0).abs() > 0.01);
                // The requesting pixel is one of the texel's 2x2 pixels.
                let near = (0..4).any(|k| {
                    let (x, y) = (qx * 2 + (k & 1), qy * 2 + (k >> 1));
                    if x >= region[0] || y >= region[1] {
                        return false;
                    }
                    let p = sc.pos[(y * f.size[0] + x) as usize];
                    (r0[q][2] - p[2]).abs() <= 0.02 * p[2] + 0.05
                });
                bad_origin += usize::from(!near);
                let i = (qy * 2 * f.size[0] + qx * 2) as usize;
                if kinds[i] == Kind::Glossy {
                    floor_req += 1;
                    // Floor: view-frame up (the world up of the level camera) and alpha
                    // 0.05^2.
                    let up = d[1] / len.max(1e-6);
                    let alpha_ok = (d[3] - 0.0025).abs() < 0.001;
                    floor_mirror += usize::from(up > 0.99 && alpha_ok);
                }
            }
        }
        let at = |k: Kind| weighted[k as usize];
        report.push(format!(
            "  RT classify (FSR path): weighted px floor {} gold {} copper {} dielectric {} | rough panel {} leaves {} bush {}; gold weight r:b {:.2} g:b {:.2}; requests {requests} (origin off {bad_origin}, non-unit normal {bad_dir}), floor requests {floor_req} with the floor normal and alpha {floor_mirror}",
            at(Kind::Glossy),
            at(Kind::Gold),
            at(Kind::Copper),
            at(Kind::Dielectric),
            at(Kind::RoughPanel),
            at(Kind::Leaves),
            at(Kind::Bush),
            gold_tint[0],
            gold_tint[1]
        ));
        let mut check = |ok: bool, what: String| {
            if !ok {
                failures.push(format!("RT classify: {what}"));
            }
        };
        check(
            at(Kind::Glossy) > 1000 && requests > 100,
            format!(
                "no weights/requests on the floor ({} px, {requests} requests)",
                at(Kind::Glossy)
            ),
        );
        // The planarity test of the depth fallback lets a rare noisy-depth pixel through by
        // chance (its SSR rays then find nothing, coverage above); at most 0.1 % of the bush.
        let bush_total = kinds.iter().filter(|&&k| k == Kind::Bush).count();
        check(
            at(Kind::RoughPanel) == 0
                && at(Kind::Leaves) == 0
                && at(Kind::Bush) * 1000 <= bush_total,
            format!(
                "weights on rough {} / excluded {} / no-material {} of {bush_total} pixels",
                at(Kind::RoughPanel),
                at(Kind::Leaves),
                at(Kind::Bush)
            ),
        );
        check(
            at(Kind::Gold) == 0 || (gold_tint[0] > 1.8 && gold_tint[1] > 1.4),
            format!("gold weights not tinted: {gold_tint:?}"),
        );
        check(
            bad_origin == 0 && bad_dir == 0,
            format!("{bad_origin} origins off their pixels, {bad_dir} non-unit directions"),
        );
        check(
            floor_req > 100 && floor_mirror * 10 >= floor_req * 9,
            format!("floor normal/alpha in {floor_mirror} of {floor_req} requests"),
        );
        report.push(format!(
            "  RT classify depth copy (R32F, D3D12 consumers): {depth_bad} of {} px differ from the R24 depth",
            region[0] * region[1]
        ));
        check(
            depth_bad == 0,
            format!("{depth_bad} depth-copy pixels differ"),
        );
        Ok(())
    })();
    rt_reflect::set_offline_targets(None);
    r
}

/// See the module documentation.
#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    g: &Gpu,
    probe: &Probe,
    color: &ID3D11Texture2D,
    color_srv: &ID3D11ShaderResourceView,
    depth: &ID3D11Texture2D,
    water: &ID3D11Texture2D,
    proj_data: &[f32; 16],
    proj: &Proj,
    fmt: &Format,
    size: [u32; 2],
    region: [u32; 2],
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
    images: &mut Vec<RgbImage>,
) -> Result<(), String> {
    let motion = g.texture(
        size,
        DXGI_FORMAT_R16G16_FLOAT,
        D3D11_BIND_SHADER_RESOURCE.0,
        None,
    )?;
    let mut e = Engine::new(&g.def, color, depth, water, proj_data, region);
    e.motion = Some(motion.cast().map_err(|e| e.to_string())?);
    let rtv = material::set_for_test(&g.dev, size)?;
    // SAFETY: plain getter.
    let material = unsafe { rtv.GetResource() }.map_err(create_err("material resource"))?;
    let f0 = material::f0_target_for_test().ok_or("no F0 colour target")?;
    let f = Frame {
        g,
        probe,
        e: &e,
        color,
        color_srv,
        depth,
        water,
        material,
        f0,
        motion: &motion,
        proj,
        fmt,
        size,
        region,
    };
    let decode = |initial: &[u8]| -> Vec<[f32; 3]> {
        initial
            .chunks_exact(fmt.bpp)
            .map(|b| fmt.decode(b))
            .collect()
    };
    let r = (|| -> Result<(), String> {
        // ---- coverage
        let (sc, initial) = f.upload(0.0, 0.0, MIRROR);
        let stored = decode(&initial);
        let kinds: Vec<Kind> = (0..sc.surf.len()).map(|i| kind(&sc, i, 0.0)).collect();
        for floors_only in [false, true] {
            set(false, floors_only);
            let o = f.run(&initial, false)?;
            let mut changed = [0usize; 10];
            let mut total = [0usize; 10];
            // Per section: sum over wall-reflecting pixels of the colour change divided by
            // (wall - panel) per channel = the per-channel blend weight, and their count.
            let mut weight = [[0f64; 3]; 10];
            let mut wall_px = [0usize; 10];
            for i in 0..o.pixels.len() {
                let k = kinds[i] as usize;
                total[k] += 1;
                if o.pixels[i] != stored[i] {
                    changed[k] += 1;
                }
                let d = [0, 1, 2].map(|c| f64::from(o.pixels[i][c] - stored[i][c]));
                if sc.surf[i] == Some(Surf::Panel) && d[2] > 0.01 {
                    wall_px[k] += 1;
                    for c in 0..3 {
                        weight[k][c] += d[c] / f64::from(WALL[c] - PANEL[c]);
                    }
                }
            }
            let at = |k: Kind| (changed[k as usize], total[k as usize]);
            // Per-channel weights relative to blue.
            let tint = |k: Kind| {
                let w = weight[k as usize];
                [w[0] / w[2].max(1e-9), w[1] / w[2].max(1e-9), 1.0]
            };
            let (gold, copper, diel, rough, leaves, bush) = (
                at(Kind::Gold),
                at(Kind::Copper),
                at(Kind::Dielectric),
                at(Kind::RoughPanel),
                at(Kind::Leaves),
                at(Kind::Bush),
            );
            let (tg, tc, td) = (tint(Kind::Gold), tint(Kind::Copper), tint(Kind::Dielectric));
            report.push(format!(
                "  coverage (material target, ssr_depth_floors_only = {}): gold changed {}/{} (weight r:g:b {:.2}:{:.2}:1), copper {}/{} ({:.2}:{:.2}:1), polished dielectric {}/{} ({:.2}:{:.2}:1), rough panel {}/{}, excluded leaves {}/{}, no-material bush {}/{}",
                u8::from(floors_only),
                gold.0,
                gold.1,
                tg[0],
                tg[1],
                copper.0,
                copper.1,
                tc[0],
                tc[1],
                diel.0,
                diel.1,
                td[0],
                td[1],
                rough.0,
                rough.1,
                leaves.0,
                leaves.1,
                bush.0,
                bush.1
            ));
            let mut check = |ok: bool, what: String| {
                if !ok {
                    failures.push(format!(
                        "coverage (floors only {}): {what}",
                        u8::from(floors_only)
                    ));
                }
            };
            for (name, px, n) in [
                ("gold", gold, wall_px[Kind::Gold as usize]),
                ("copper", copper, wall_px[Kind::Copper as usize]),
                ("dielectric", diel, wall_px[Kind::Dielectric as usize]),
            ] {
                check(
                    px.1 > 0 && px.0 * 2 > px.1 && n * 4 > px.1,
                    format!("{name} panel does not reflect: {px:?}, {n} px show the wall"),
                );
            }
            // Gold F0 r:b = 2.9, g:b = 2.3; copper r:b = 1.8, g:b = 1.2 (grazing angles pull
            // toward 1); the dielectric must stay neutral.
            check(
                tg[0] > 1.8 && tg[1] > 1.5,
                format!("gold reflection not tinted: {tg:?}"),
            );
            check(
                tc[0] > 1.3 && tc[0] < tg[0] && tc[1] > 1.0 && tc[1] < tg[1],
                format!("copper reflection tint {tc:?} (gold {tg:?})"),
            );
            check(
                (td[0] - 1.0).abs() < 0.15 && (td[1] - 1.0).abs() < 0.15,
                format!("dielectric reflection not neutral: {td:?}"),
            );
            check(
                rough.1 > 0 && rough.0 == 0,
                format!("rough panel changed {rough:?}"),
            );
            check(
                leaves.1 > 0 && leaves.0 == 0,
                format!("excluded leaves changed {leaves:?}"),
            );
            check(
                bush.1 > 0 && bush.0 == 0,
                format!("noisy-depth bush changed {bush:?}"),
            );
            if !floors_only {
                images.push(image_of("ssr_coverage", &o.pixels, size, region));
            }
        }

        // ---- hybrid RT classification (FSR path, D3D11 side of rt_reflect.rs)
        classify_check(&f, &sc, &initial, &kinds, report, failures)?;

        // ---- frame-to-frame stability (static camera, swaying weapon stand-in)
        stability_check(&f, report, failures)?;

        // ---- silhouette halo (thin polished pole in front of the bright wall)
        halo_check(&f, report, failures)?;

        // ---- fireflies (small bright lamps reflected by the glossy floor)
        firefly_check(&f, report, failures)?;

        // ---- noise (static camera)
        let (sc, initial) = f.upload(0.0, 0.0, GLOSSY);
        let glossy: Vec<usize> = (0..sc.surf.len())
            .filter(|&i| kind(&sc, i, 0.0) == Kind::Glossy)
            .collect();
        set(true, true);
        let reference = f.reference(&initial)?;
        let stored = decode(&initial);
        // Pixels whose reference carries a reflection.
        let mask: Vec<usize> = glossy
            .iter()
            .copied()
            .filter(|&i| (0..3).any(|k| (reference[i][k] - f64::from(stored[i][k])).abs() > 1e-3))
            .collect();
        set(false, true);
        let before = f.run(&initial, true)?;
        set(true, true);
        break_history();
        let first = f.run(&initial, false)?;
        let mut after = first.pixels.clone();
        let mut temporal_ms = 0.0;
        for k in 1..ACCUMULATE_FRAMES {
            let o = f.run(&initial, k + 1 == ACCUMULATE_FRAMES)?;
            temporal_ms = o.gpu_ms.unwrap_or(f64::NAN);
            after = o.pixels;
        }
        let (rb, r1, ra) = (
            rmse(&before.pixels, &reference, &mask),
            rmse(&first.pixels, &reference, &mask),
            rmse(&after, &reference, &mask),
        );
        report.push(format!(
            "  noise (glossy floor, roughness {} F0 {}, {} px with reflection): RMSE vs {REFERENCE_FRAMES}-frame reference: before (single frame, fixed pattern) {rb:.4}, temporal frame 1 {r1:.4}, after {ACCUMULATE_FRAMES} frames {ra:.4} ({:.1}x lower); GPU half medium: without temporal {:.3} ms, with {:.3} ms",
            GLOSSY.0,
            GLOSSY.1,
            mask.len(),
            rb / ra.max(1e-9),
            before.gpu_ms.unwrap_or(f64::NAN),
            temporal_ms
        ));
        if mask.len() < 1000 || ra.is_nan() || ra >= rb * 0.7 {
            failures.push(format!(
                "noise: after {ra:.4} not below 0.7 x before {rb:.4} ({} px)",
                mask.len()
            ));
        }
        images.push(image_f64("ssr_noise_reference", &reference, size, region));
        images.push(image_of("ssr_noise_before", &before.pixels, size, region));
        images.push(image_of("ssr_noise_after", &after, size, region));

        // ---- ghosting (camera moving along x)
        let start = -0.5f32;
        let end = start + MOVE_STEP * (MOVE_FRAMES - 1) as f32;
        let (_, init_start) = f.upload(start, 0.0, GLOSSY);
        let ref_start = f.reference(&init_start)?;
        let (sc_end, init_end) = f.upload(end, 0.0, GLOSSY);
        let ref_end = f.reference(&init_end)?;
        let stored_end = decode(&init_end);
        let glossy_end: Vec<usize> = (0..sc_end.surf.len())
            .filter(|&i| kind(&sc_end, i, end) == Kind::Glossy)
            .collect();
        let mask_end: Vec<usize> = glossy_end
            .iter()
            .copied()
            .filter(|&i| (0..3).any(|k| (ref_end[i][k] - f64::from(stored_end[i][k])).abs() > 1e-3))
            .collect();
        set(false, true);
        let off_end = f.run(&init_end, false)?;
        set(true, true);
        break_history();
        let mut static_end = f.run(&init_end, false)?;
        for _ in 1..ACCUMULATE_FRAMES {
            static_end = f.run(&init_end, false)?;
        }
        break_history();
        let mut moving = Vec::new();
        for k in 0..MOVE_FRAMES {
            let cx = start + MOVE_STEP * k as f32;
            let (_, init) = f.upload(cx, if k == 0 { 0.0 } else { MOVE_STEP }, GLOSSY);
            moving = f.run(&init, false)?.pixels;
        }
        // Content that moved between the start and the end pose.
        let trail: Vec<usize> = glossy_end
            .iter()
            .copied()
            .filter(|&i| (lum(ref_start[i]) - lum(ref_end[i])).abs() > 0.02)
            .collect();
        let (mut resid, mut diff) = (0f64, 0f64);
        for &i in &trail {
            let m = moving[i].map(f64::from);
            resid += (lum(m) - lum(ref_end[i])).abs();
            diff += (lum(ref_start[i]) - lum(ref_end[i])).abs();
        }
        let ghost = resid / diff.max(1e-9);
        let (rm, rs, ro) = (
            rmse(&moving, &ref_end, &mask_end),
            rmse(&static_end.pixels, &ref_end, &mask_end),
            rmse(&off_end.pixels, &ref_end, &mask_end),
        );
        report.push(format!(
            "  ghosting (camera {MOVE_FRAMES} frames x {MOVE_STEP} m along x): RMSE at the final pose: moving {rm:.4}, static accumulated {rs:.4}, no temporal {ro:.4}; trail ({} px whose reference changed): residual {:.0} % of the change",
            trail.len(),
            ghost * 100.0
        ));
        if trail.len() < 100 || ghost.is_nan() || ghost > 0.35 || rm >= ro {
            failures.push(format!(
                "ghosting: trail residual {:.0} % ({} px), moving RMSE {rm:.4} vs no temporal {ro:.4}",
                ghost * 100.0,
                trail.len()
            ));
        }
        images.push(image_of("ssr_ghost_moving", &moving, size, region));
        images.push(image_f64("ssr_ghost_reference", &ref_end, size, region));
        Ok(())
    })();
    material::reset_for_test();
    settings::edit(|f| *f = Settings::default());
    r
}
