//! Offline check of the reflections (`examples/ssr_offline.rs`): a private D3D11 device and
//! stand-in engine structures (POSTPROC pass with read/write lists, engine context, view
//! with projection, yawed view matrix and viewport) laid out at the offsets the TAA site
//! reads, so `capture` + `record_engine` / `execute` run exactly as at 0x140327817.
//! Scene (view frame, eye 1.6 m above the floor): reflective floor, a red box on it, a blue
//! back wall, bright sky, and a water surface 0.1 m above the floor that is not in the
//! opaque depth. Engine-like inputs: R11G11B10 colour (full case) / RGBA16F colour (scaled
//! case), reversed-Z R24G8 depth and a `depthWater` target in three variants: as the engine
//! fills it (opaque copy + water on top), a plain opaque copy (the configuration that masked
//! the whole frame in game with the first build), and everything in front (coverage above
//! the limit). Checks: the box shows up in the floor in front of it and the wall further out
//! (hit placement), no NaN/Inf, nothing written outside the region (FSR upscale),
//! walls/box faces/sky and the water bit-identical (floors only, water mask), the mask
//! ignored when it covers too much, the read-back coverage and fallback counters, every
//! D3D11 state of the deferred context unchanged (including a TAA-like colour SRV that our
//! RTV binding unbinds), the FSR-path result (immediate context) equal to the engine-path
//! result, the applied counters of both paths, nothing captured with `ssr = 0`; GPU time per
//! quality at full and half resolution, at native size and at render scale `scale`.

mod temporal;

use super::*;
use crate::ao::offline::{Gpu, Probe, WIDE};

const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.01;
const FAR: f32 = 7000.0;
const JITTER: [f32; 2] = [0.0004, -0.0003];
const EYE: f32 = 1.6;
const YAW: f32 = 0.7;
/// Height of the water surface above the floor.
const WATER_HEIGHT: f32 = 0.1;

const FLOOR: [f32; 3] = [0.08, 0.08, 0.08];
const BOX: [f32; 3] = [3.0, 0.2, 0.2];
const WALL: [f32; 3] = [0.2, 0.4, 1.2];
const SKY: [f32; 3] = [2.0, 2.0, 2.0];
const WATER: [f32; 3] = [0.05, 0.12, 0.15];
const PANEL: [f32; 3] = [0.02, 0.02, 0.02];
const BUSH: [f32; 3] = [0.05, 0.2, 0.05];
/// Vertical panel facing -x (the scene centre) at x = `PANEL_X`, z 6..10, floor to 1 m:
/// metal sections for z < `PANEL_SPLIT`, dielectric and rough beyond (material target cases,
/// `offline/temporal.rs`).
const PANEL_X: f32 = 3.0;
const PANEL_SPLIT: f32 = 8.0;
/// Floor patch with noisy depth and no material (foliage seen by the depth fallback):
/// x -3.5..-2.5, z 3..4.5.
const BUSH_X: [f32; 2] = [-3.5, -2.5];
const BUSH_Z: [f32; 2] = [3.0, 4.5];
/// Floor patch carrying the material exclusion marker (vegetation shaders): x -1..1,
/// z 7..8.5.
const LEAVES_X: [f32; 2] = [-1.0, 1.0];
const LEAVES_Z: [f32; 2] = [7.0, 8.5];
/// Outside the region (FSR upscale): must stay untouched.
const OUTSIDE: [f32; 3] = [0.5, 0.25, 0.125];

/// An RGB image for inspection (written as PNG by the example).
pub struct RgbImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

pub struct SsrReport {
    pub text: String,
    pub images: Vec<RgbImage>,
    pub failures: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Surf {
    Sky,
    Floor,
    Water,
    BoxTop,
    BoxSide,
    Wall,
    Panel,
    Bush,
    /// First-person weapon stand-in (`blade`): near the camera, idle sway, depth-hacked.
    Blade,
    /// A thin polished pole (1-2 px wide) in front of the bright wall/sky: silhouette pixels
    /// whose depth normals are noise (present with the blade).
    Pole,
}

/// First-person weapon stand-in: a horizontal polished plate (normal up) at view-frame y
/// `BLADE_Y`, x `BLADE_X`, z `BLADE_Z` (plus the per-frame sway), drawn as the engine draws
/// player models: clip-space depth times `BLADE_DEPTH_HACK` (O, `vertex.decl`
/// `_projPosition.z *= $modelDepthHack.x`), so its stored depth reads `BLADE_DEPTH_HACK`
/// times closer; its motion vectors carry the sway (previous model transform).
const BLADE_Y: f32 = -0.3;
const BLADE_X: [f32; 2] = [0.12, 0.42];
const BLADE_Z: [f32; 2] = [0.45, 0.9];
const BLADE_DEPTH_HACK: f32 = 2.0;
/// The thin pole: world x `POLE_X` .. + `POLE_W`, at z `POLE_Z`, floor to 1 m, facing the
/// camera.
const POLE_X: f32 = -1.2;
const POLE_W: f32 = 0.004;
const POLE_Z: f32 = 2.5;
/// Firefly check (`offline/temporal.rs`): small bright lamps (0.1 m squares, ~8 px) on the
/// back wall at view-frame x `LAMP_X`, height `LAMP_Y` above the eye, reflected by the floor.
static LAMPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const LAMP: [f32; 3] = [40.0, 34.0, 26.0];
const LAMP_X: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const LAMP_Y: [f32; 2] = [-0.6, 0.8];
const LAMP_HALF: f32 = 0.05;

fn lamp_at(x: f32, y: f32) -> bool {
    LAMP_X.iter().any(|&lx| (x - lx).abs() <= LAMP_HALF)
        && LAMP_Y.iter().any(|&ly| (y - ly).abs() <= LAMP_HALF)
}

/// Scene in the view frame of a camera at x = `cx` (x right, y up, z forward, eye at
/// (cx, 0, 0)): floor y = -EYE, a water surface 0.1 m above it over x 1.5..3.5, z 3..7 (not
/// in the opaque depth), box x -0.5..0.5, y -EYE..-EYE+1, z 4..5, back wall z = 12 up to
/// y = 3, the panel at x = `PANEL_X`; sky elsewhere. Returns (t of the opaque hit, surface
/// seen, t of the water surface if it is the closest).
fn trace(dir: [f32; 3], cx: f32, blade: Option<[f32; 2]>) -> (f32, Surf, Option<f32>) {
    let mut best = (f32::INFINITY, Surf::Sky);
    let mut hit = |t: f32, ok: bool, s: Surf| {
        if ok && t > 0.0 && t < best.0 {
            best = (t, s);
        }
    };
    if dir[1] < 0.0 {
        hit(-EYE / dir[1], true, Surf::Floor);
    }
    let t = 12.0 / dir[2];
    hit(t, dir[1] * t <= 3.0, Surf::Wall);
    if dir[0] > 0.0 {
        let t = (PANEL_X - cx) / dir[0];
        let (y, z) = (dir[1] * t, dir[2] * t);
        hit(
            t,
            (-EYE..=1.0).contains(&y) && (6.0..=10.0).contains(&z),
            Surf::Panel,
        );
    }
    let (lo, hi) = ([-0.5 - cx, -EYE, 4.0], [0.5 - cx, -EYE + 1.0, 5.0]);
    let (mut t0, mut t1, mut axis) = (0.0f32, f32::INFINITY, 0);
    for a in 0..3 {
        let (u, v) = (lo[a] / dir[a], hi[a] / dir[a]);
        if u.min(v) > t0 {
            t0 = u.min(v);
            axis = a;
        }
        t1 = t1.min(u.max(v));
    }
    hit(
        t0,
        t0 <= t1,
        if axis == 1 {
            Surf::BoxTop
        } else {
            Surf::BoxSide
        },
    );
    if let Some(s) = blade
        && dir[1] < 0.0
    {
        let t = (BLADE_Y + s[1]) / dir[1];
        let (x, z) = (dir[0] * t - s[0], dir[2] * t);
        hit(
            t,
            (BLADE_X[0]..=BLADE_X[1]).contains(&x) && (BLADE_Z[0]..=BLADE_Z[1]).contains(&z),
            Surf::Blade,
        );
        let t = POLE_Z / dir[2];
        let (x, y) = (dir[0] * t + cx, dir[1] * t);
        hit(
            t,
            (POLE_X..=POLE_X + POLE_W).contains(&x) && (-EYE..=1.0).contains(&y),
            Surf::Pole,
        );
    }
    let water = (dir[1] < 0.0)
        .then(|| -(EYE - WATER_HEIGHT) / dir[1])
        .filter(|&t| {
            let (x, z) = (dir[0] * t + cx, dir[2] * t);
            t < best.0 && (1.5..=3.5).contains(&x) && (3.0..=7.0).contains(&z)
        });
    match water {
        Some(t) => (best.0, Surf::Water, Some(t)),
        None => (best.0, best.1, None),
    }
}

fn color_of(s: Surf) -> [f32; 3] {
    match s {
        Surf::Sky => SKY,
        Surf::Floor => FLOOR,
        Surf::Water => WATER,
        Surf::BoxTop | Surf::BoxSide => BOX,
        Surf::Wall => WALL,
        Surf::Panel => PANEL,
        Surf::Bush => BUSH,
        Surf::Blade | Surf::Pole => PANEL,
    }
}

fn pixel(p: &Proj, size: [u32; 2], v: [f32; 3]) -> [u32; 2] {
    let ndc = [
        v[0] / v[2] * p.p00 + p.jitter[0],
        v[1] / v[2] * p.p11 + p.jitter[1],
    ];
    [
        ((ndc[0] + 1.0) * 0.5 * size[0] as f32) as u32,
        ((1.0 - ndc[1]) * 0.5 * size[1] as f32) as u32,
    ]
}

/// Contents of the stand-in `depthWater` target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaterTarget {
    /// As the engine (`sub_14034F740`): a copy of the opaque depth, water drawn on top.
    Real,
    /// A plain copy of the opaque depth, no water: the in-game failure of the first build
    /// (its `>=` test masked every pixel).
    Copy,
    /// Everything in front of the opaque depth: coverage above the limit, mask ignored.
    AllCloser,
}

struct Scene {
    depth: Vec<u32>,
    /// `depthWater` per `WaterTarget` (in declaration order).
    water: [Vec<u32>; 3],
    color: Vec<[f32; 3]>,
    surf: Vec<Option<Surf>>,
    /// Opaque hit in the camera's view frame (infinite for the sky).
    pos: Vec<[f32; 3]>,
    water_pixels: usize,
}

/// Deterministic per-pixel noise in 0..1.
fn hash01(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(0x8DA6_B343) ^ y.wrapping_mul(0xD816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 16;
    (h & 0xFFFF) as f32 / 65_535.0
}

/// Scene as seen by a camera at x = `cx` (see [`trace`]); the bush patch gets +-2 % depth
/// noise; `blade` = the weapon stand-in's sway offset (x, y) or none.
fn scene(p: &Proj, tex: [u32; 2], region: [u32; 2], cx: f32, blade: Option<[f32; 2]>) -> Scene {
    let hw = |z: f32| {
        let d = (p.depth_add + p.depth_mul / z).clamp(0.0, 1.0);
        ((d * 16_777_215.0).round() as u32) | (5 << 24)
    };
    let n = (tex[0] * tex[1]) as usize;
    let mut s = Scene {
        depth: Vec::with_capacity(n),
        water: std::array::from_fn(|_| Vec::with_capacity(n)),
        color: Vec::with_capacity(n),
        surf: Vec::with_capacity(n),
        pos: Vec::with_capacity(n),
        water_pixels: 0,
    };
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            if x >= region[0] || y >= region[1] {
                let d = hw(0.05);
                s.depth.push(d);
                s.water[0].push(d);
                s.water[1].push(d);
                s.water[2].push(d + 64);
                s.color.push(OUTSIDE);
                s.surf.push(None);
                s.pos.push([0.0, 0.0, 0.05]);
                continue;
            }
            let ndc = [
                (x as f32 + 0.5) / region[0] as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / region[1] as f32 * 2.0,
            ];
            let dir = [
                (ndc[0] - p.jitter[0]) / p.p00,
                (ndc[1] - p.jitter[1]) / p.p11,
                1.0,
            ];
            let (mut t, mut surf, water) = trace(dir, cx, blade);
            if surf == Surf::Floor {
                let (wx, wz) = (dir[0] * t + cx, t);
                if (BUSH_X[0]..=BUSH_X[1]).contains(&wx) && (BUSH_Z[0]..=BUSH_Z[1]).contains(&wz) {
                    surf = Surf::Bush;
                    t *= 1.0 + 0.04 * (hash01(x, y) - 0.5);
                }
            }
            let d = if !t.is_finite() {
                5 << 24
            } else if surf == Surf::Blade {
                let d = ((p.depth_add + p.depth_mul / t) * BLADE_DEPTH_HACK).clamp(0.0, 1.0);
                ((d * 16_777_215.0).round() as u32) | (5 << 24)
            } else {
                hw(t)
            };
            s.depth.push(d);
            s.water[0].push(water.map_or(d, hw));
            s.water[1].push(d);
            s.water[2].push(d + 64);
            s.water_pixels += usize::from(water.is_some());
            let lamp = surf == Surf::Wall
                && LAMPS.load(std::sync::atomic::Ordering::Relaxed)
                && lamp_at(dir[0] * t + cx, dir[1] * t);
            s.color.push(if lamp { LAMP } else { color_of(surf) });
            s.surf.push(Some(surf));
            s.pos.push(dir.map(|v| v * t));
        }
    }
    s
}

/// Engine view matrix (row-major, column vectors) of an id-style camera yawed by `YAW` in a
/// Z-up world: rows right, up, -forward (translation irrelevant here).
fn view_matrix() -> [f32; 16] {
    let (s, c) = YAW.sin_cos();
    let right = [s, -c, 0.0];
    let up = [0.0, 0.0, 1.0];
    let fwd = [c, s, 0.0];
    let mut m = [0f32; 16];
    for (i, row) in [right, up, [-fwd[0], -fwd[1], -fwd[2]]].iter().enumerate() {
        m[4 * i..4 * i + 3].copy_from_slice(row);
    }
    m[3] = -2.0;
    m[7] = -EYE;
    m[11] = 5.0;
    m[15] = 1.0;
    m
}

// ------------------------------------------------------------------ pixel formats --

fn f32_to_small(v: f32, mant: u32) -> u32 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let max_exp = 30;
    let e = v.log2().floor() as i32;
    let (mut exp, mut m) = if e < -14 {
        (
            0i32,
            (v / 2f32.powi(-14) * (1u32 << mant) as f32).round() as u32,
        )
    } else {
        (
            e + 15,
            ((v / 2f32.powi(e) - 1.0) * (1u32 << mant) as f32).round() as u32,
        )
    };
    if m >= 1 << mant && exp > 0 {
        m = 0;
        exp += 1;
    } else if exp == 0 && m >= 1 << mant {
        exp = 1;
        m = 0;
    }
    if exp > max_exp {
        return ((max_exp as u32) << mant) | ((1 << mant) - 1);
    }
    ((exp as u32) << mant) | m
}

fn small_to_f32(v: u32, mant: u32) -> f32 {
    let exp = v >> mant;
    let m = v & ((1 << mant) - 1);
    match exp {
        0 => m as f32 / (1u32 << mant) as f32 * 2f32.powi(-14),
        31 => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => 2f32.powi(exp as i32 - 15) * (1.0 + m as f32 / (1u32 << mant) as f32),
    }
}

fn pack_r11g11b10(c: [f32; 3]) -> [u8; 4] {
    (f32_to_small(c[0], 6) | (f32_to_small(c[1], 6) << 11) | (f32_to_small(c[2], 5) << 22))
        .to_le_bytes()
}

fn unpack_r11g11b10(b: &[u8]) -> [f32; 3] {
    let p = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    [
        small_to_f32(p & 0x7FF, 6),
        small_to_f32((p >> 11) & 0x7FF, 6),
        small_to_f32(p >> 22, 5),
    ]
}

fn f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let s = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    (f32_to_small(v.abs(), 10).min(0x7BFF) as u16) | s
}

fn f16_value(h: u16) -> f32 {
    let v = small_to_f32(u32::from(h & 0x7FFF), 10);
    if h & 0x8000 != 0 { -v } else { v }
}

struct Format {
    dxgi: DXGI_FORMAT,
    bpp: usize,
}

impl Format {
    fn encode(&self, c: [f32; 3]) -> Vec<u8> {
        if self.bpp == 4 {
            pack_r11g11b10(c).to_vec()
        } else {
            [c[0], c[1], c[2], 1.0]
                .iter()
                .flat_map(|v| f16_bits(*v).to_le_bytes())
                .collect()
        }
    }

    fn decode(&self, b: &[u8]) -> [f32; 3] {
        if self.bpp == 4 {
            unpack_r11g11b10(b)
        } else {
            let h = |i: usize| f16_value(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
            [h(0), h(1), h(2)]
        }
    }
}

fn make_texture(
    g: &Gpu,
    size: [u32; 2],
    format: DXGI_FORMAT,
    bind: i32,
    data: &[u8],
    pitch: u32,
) -> Result<ID3D11Texture2D, String> {
    let d = D3D11_TEXTURE2D_DESC {
        Width: size[0],
        Height: size[1],
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind as u32,
        ..Default::default()
    };
    let init = D3D11_SUBRESOURCE_DATA {
        pSysMem: data.as_ptr().cast(),
        SysMemPitch: pitch,
        SysMemSlicePitch: 0,
    };
    let mut t = None;
    // SAFETY: valid descriptor; the initial data covers size[1] rows of `pitch` bytes.
    unsafe { g.dev.CreateTexture2D(&d, Some(&init), Some(&mut t)) }
        .map_err(create_err("texture"))?;
    t.ok_or_else(|| "no texture".to_owned())
}

/// Raw readback of a texture of `bpp` bytes per texel.
fn read_raw(g: &Gpu, tex: &ID3D11Texture2D, size: [u32; 2], bpp: usize) -> Result<Vec<u8>, String> {
    let mut d = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { tex.GetDesc(&mut d) };
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    d.MiscFlags = 0;
    let mut st = None;
    // SAFETY (whole block): staging copy of an identical texture, mapped for reading within
    // the mapped pitch, unmapped afterwards.
    unsafe {
        g.dev
            .CreateTexture2D(&d, None, Some(&mut st))
            .map_err(create_err("staging"))?;
        let st = st.ok_or("no staging")?;
        g.imm.CopyResource(&st, tex);
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        g.imm
            .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
            .map_err(create_err("Map staging"))?;
        let row = size[0] as usize * bpp;
        let mut out = Vec::with_capacity(row * size[1] as usize);
        for y in 0..size[1] as usize {
            let p = m.pData.cast::<u8>().add(y * m.RowPitch as usize);
            out.extend_from_slice(std::slice::from_raw_parts(p, row));
        }
        g.imm.Unmap(&st, 0);
        Ok(out)
    }
}

/// Stand-in engine memory for one POSTPROC pass at the TAA site.
struct Engine {
    pass: Vec<u64>,
    ctx: Vec<u64>,
    _view: Vec<u64>,
    _lists: Vec<Vec<u64>>,
    _images: Vec<Vec<u64>>,
    _image_views: Vec<Vec<u64>>,
    /// The frame's `motionVectors` (what `fsr::motion_vectors` reads in game).
    motion: Option<ID3D11Resource>,
}

impl Engine {
    fn new(
        ctx: &ID3D11DeviceContext,
        color: &ID3D11Texture2D,
        depth: &ID3D11Texture2D,
        water: &ID3D11Texture2D,
        proj: &[f32; 16],
        region: [u32; 2],
    ) -> Self {
        let images: Vec<Vec<u64>> = [color.as_raw(), depth.as_raw(), water.as_raw()]
            .iter()
            .map(|&r| {
                let mut img = vec![0u64; 0x140 / 8];
                img[0x130 / 8] = r as u64;
                img
            })
            .collect();
        let image_views: Vec<Vec<u64>> = images.iter().map(|i| vec![i.as_ptr() as u64]).collect();
        let lists = vec![
            // Read list: depth, water stencil view.
            vec![
                u64::from(HASH_DEPTH),
                image_views[1].as_ptr() as u64,
                u64::from(HASH_WATER),
                image_views[2].as_ptr() as u64,
            ],
            // Write list: hdrBuffer.
            vec![u64::from(HASH_COLOR), image_views[0].as_ptr() as u64],
        ];
        let engine_ctx = vec![ctx.as_raw() as u64, 0, 0, 0];
        let mut view = vec![0u64; (VIEW_VIEWPORT as usize + 64) / 8];
        {
            // SAFETY: `view` is a u64 buffer viewed as bytes for the writes below.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(view.as_mut_ptr().cast::<u8>(), view.len() * 8)
            };
            let put = |b: &mut [u8], off: u64, v: &[u8]| {
                b[off as usize..off as usize + v.len()].copy_from_slice(v);
            };
            let floats = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>();
            put(bytes, VIEW_PROJECTION, &floats(proj));
            put(bytes, VIEW_VIEW, &floats(&view_matrix()));
            let rect = [0i32, 0, region[0] as i32 - 1, region[1] as i32 - 1];
            put(
                bytes,
                VIEW_VIEWPORT,
                &rect
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        let mut pass = vec![0u64; 0x110 / 8];
        pass[(PASS_VIEW / 8) as usize] = view.as_ptr() as u64;
        pass[0x48 / 8] = lists[0].as_ptr() as u64;
        pass[0x48 / 8 + 1] = 2;
        pass[0x80 / 8] = lists[1].as_ptr() as u64;
        pass[0x80 / 8 + 1] = 1;
        Self {
            pass,
            ctx: engine_ctx,
            _view: view,
            _lists: lists,
            _images: images,
            _image_views: image_views,
            motion: None,
        }
    }

    fn pass(&self) -> u64 {
        self.pass.as_ptr() as u64
    }
}

/// TAA-like bindings on top of the probe state: the colour SRV at CS t2 (`viewColorMap`) and
/// t6, so our RTV binding of the same texture unbinds them and the restore must put them back.
fn bind_taa_like(ctx: &ID3D11DeviceContext, color_srv: &ID3D11ShaderResourceView) {
    // SAFETY: bindings of live objects of this device.
    unsafe {
        ctx.CSSetShaderResources(2, Some(&[Some(color_srv.clone())]));
        ctx.CSSetShaderResources(6, Some(&[Some(color_srv.clone())]));
    }
}

struct Outcome {
    state_diff: Vec<String>,
    pixels: Vec<[f32; 3]>,
    gpu_ms: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    g: &Gpu,
    probe: &Probe,
    e: &Engine,
    color: &ID3D11Texture2D,
    color_srv: &ID3D11ShaderResourceView,
    initial: &[u8],
    fmt: &Format,
    size: [u32; 2],
    region: [u32; 2],
    time: bool,
) -> Result<Outcome, String> {
    probe.bind(&g.def, region);
    bind_taa_like(&g.def, color_srv);
    let (before_ok, before_fail) = (
        ENGINE.load(Ordering::Relaxed),
        FAILURES.load(Ordering::Relaxed),
    );
    // SAFETY: capture on the deferred context recorded by this thread.
    let before = unsafe { Saved::capture(&g.def, WIDE) };
    let job = capture(e.pass(), e.motion.clone()).ok_or_else(|| {
        format!(
            "nothing captured: {}",
            lock(&ERROR).clone().unwrap_or_default()
        )
    })?;
    record_engine(e.ctx.as_ptr() as u64, job);
    // SAFETY: as above.
    let after = unsafe { Saved::capture(&g.def, WIDE) };
    if FAILURES.load(Ordering::Relaxed) != before_fail {
        return Err(lock(&ERROR).clone().unwrap_or_default());
    }
    if ENGINE.load(Ordering::Relaxed) == before_ok {
        return Err("not recorded".into());
    }
    let mut list = None;
    // SAFETY: finishing the deferred context's list.
    unsafe { g.def.FinishCommandList(false, Some(&mut list)) }
        .map_err(create_err("FinishCommandList"))?;
    let list: ID3D11CommandList = list.ok_or("no command list")?;
    let pitch = size[0] * fmt.bpp as u32;
    // SAFETY: full-texture update with data of the texture's size and pitch, then the list.
    unsafe {
        g.imm
            .UpdateSubresource(color, 0, None, initial.as_ptr().cast(), pitch, 0);
        g.imm.ExecuteCommandList(&list, false);
    }
    let raw = read_raw(g, color, size, fmt.bpp)?;
    let gpu_ms = if time { Some(g.time(&list)?) } else { None };
    Ok(Outcome {
        state_diff: before.diff(&after),
        pixels: raw.chunks_exact(fmt.bpp).map(|b| fmt.decode(b)).collect(),
        gpu_ms,
    })
}

fn tonemap(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| ((v / (1.0 + v)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8)
}

fn image(name: String, px: &[[f32; 3]], tex: [u32; 2], region: [u32; 2]) -> RgbImage {
    RgbImage {
        name,
        width: region[0],
        height: region[1],
        rgb: (0..region[1])
            .flat_map(|y| (0..region[0]).map(move |x| (x, y)))
            .flat_map(|(x, y)| tonemap(px[(y * tex[0] + x) as usize]))
            .collect(),
    }
}

/// Mean colour of the 5x5 block around `at`.
fn mean(px: &[[f32; 3]], tex: [u32; 2], at: [u32; 2]) -> [f32; 3] {
    let mut s = [0f32; 3];
    for dy in 0..5 {
        for dx in 0..5 {
            let x = (at[0] + dx).saturating_sub(2).min(tex[0] - 1);
            let y = (at[1] + dy).saturating_sub(2).min(tex[1] - 1);
            let p = px[(y * tex[0] + x) as usize];
            for k in 0..3 {
                s[k] += p[k] / 25.0;
            }
        }
    }
    s
}

/// View-frame x of the floor point seen at region pixel (x, y).
fn floor_x(p: &Proj, region: [u32; 2], x: u32, y: u32) -> f32 {
    let ndc = [
        (x as f32 + 0.5) / region[0] as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / region[1] as f32 * 2.0,
    ];
    let dir = [
        (ndc[0] - p.jitter[0]) / p.p00,
        (ndc[1] - p.jitter[1]) / p.p11,
    ];
    dir[0] * (-EYE / dir[1])
}

/// Material target (as the patched lit shaders write it): floor beyond x < -1 m rough
/// (0.9), the rest of the floor a smooth (0.05) reflective (F0 0.25) material, everything
/// else no material. Checks: rough floor unchanged (also the wall-reflection point), the
/// box reflection stronger than with the user model, `ssr_material = depth` identical to the
/// default result, and the forward-pass binding keeps the engine's targets 0-1.
#[allow(clippy::too_many_arguments)]
fn check_material(
    g: &Gpu,
    probe: &Probe,
    e: &Engine,
    color: &ID3D11Texture2D,
    color_srv: &ID3D11ShaderResourceView,
    initial: &[u8],
    fmt: &Format,
    sc: &Scene,
    proj: &Proj,
    size: [u32; 2],
    region: [u32; 2],
    at: &[[u32; 2]],
    default_px: Option<&[[f32; 3]]>,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let stored: Vec<[f32; 3]> = initial
        .chunks_exact(fmt.bpp)
        .map(|b| fmt.decode(b))
        .collect();
    let rtv = material::set_for_test(&g.dev, size)?;
    // SAFETY: plain getter.
    let res = unsafe { rtv.GetResource() }.map_err(create_err("material resource"))?;
    let mut data = vec![0u32; (size[0] * size[1]) as usize];
    let mut rough_px = Vec::new();
    for y in 0..region[1] {
        for x in 0..region[0] {
            let i = (y * size[0] + x) as usize;
            if sc.surf[i] != Some(Surf::Floor) {
                continue;
            }
            let rough = floor_x(proj, region, x, y) < -1.0;
            let (r, f0) = if rough {
                (0.9f32, 0.04f32)
            } else {
                (0.05, 0.25)
            };
            if rough {
                rough_px.push(i);
            }
            // World up encodes as (0.5, 0.5).
            data[i] = u32::from_le_bytes([
                128,
                128,
                (r * 255.0).round() as u8,
                (f0.sqrt() * 255.0).round() as u8,
            ]);
        }
    }
    // SAFETY: full-texture update of an R8G8B8A8 texture of `size` with matching pitch.
    unsafe {
        g.imm
            .UpdateSubresource(&res, 0, None, data.as_ptr().cast(), size[0] * 4, 0)
    };
    settings::edit(|f| {
        *f = Settings::default();
        f.ssr = true;
    });
    let o = run_engine(
        g, probe, e, color, color_srv, initial, fmt, size, region, true,
    )?;
    let rough_changed = rough_px
        .iter()
        .filter(|&&i| o.pixels[i] != stored[i])
        .count();
    let box_mat = mean(&o.pixels, size, at[0]);
    let box_user = default_px.map(|d| mean(d, size, at[0]));
    settings::edit(|f| {
        *f = Settings::default();
        f.ssr = true;
        f.ssr_material = false;
    });
    let od = run_engine(
        g, probe, e, color, color_srv, initial, fmt, size, region, true,
    )?;
    let depth_same = default_px == Some(od.pixels.as_slice());
    report.push(format!(
        "  material target: rough floor ({} px) changed {rough_changed}, box reflection {box_mat:.3?} (user model {box_user:.3?}), ssr_material = depth equals the user-model result {depth_same}, state {}; GPU half medium with material {:.3} ms, depth {:.3} ms",
        rough_px.len(),
        if o.state_diff.is_empty() { "restored" } else { "CHANGED" },
        o.gpu_ms.unwrap_or(f64::NAN),
        od.gpu_ms.unwrap_or(f64::NAN)
    ));
    let mut check = |ok: bool, what: String| {
        if !ok {
            failures.push(format!("material target: {what}"));
        }
    };
    check(
        rough_changed == 0,
        format!("{rough_changed} rough floor pixels reflect"),
    );
    check(
        box_user.is_some_and(|u| box_mat[0] > u[0] + 0.05),
        format!("box reflection not stronger with F0 0.25: {box_mat:?} vs {box_user:?}"),
    );
    check(
        depth_same,
        "ssr_material = depth differs from the user model".into(),
    );
    check(
        o.state_diff.is_empty(),
        format!("state changed: {:?}", o.state_diff),
    );

    // Forward-pass binding: targets 0-1 and depth stay, ours at slot 2.
    let rt = |format| {
        g.texture(size, format, D3D11_BIND_RENDER_TARGET.0, None)
            .and_then(|t| {
                let mut v = None;
                // SAFETY: valid texture and out pointer.
                unsafe { g.dev.CreateRenderTargetView(&t, None, Some(&mut v)) }
                    .map_err(create_err("RTV"))?;
                v.ok_or_else(|| "no RTV".to_owned())
            })
    };
    let (r0, r1) = (
        rt(DXGI_FORMAT_R8G8B8A8_UNORM)?,
        rt(DXGI_FORMAT_R8G8B8A8_UNORM)?,
    );
    let engine_ctx = [g.def.as_raw() as u64, 0];
    let mut got: [Option<ID3D11RenderTargetView>; 3] = Default::default();
    // SAFETY (whole block): bindings of this device's views on the deferred context; the
    // list is finished and dropped afterwards.
    unsafe {
        g.def
            .OMSetRenderTargets(Some(&[Some(r0.clone()), Some(r1.clone())]), None);
        material::add_target(engine_ctx.as_ptr() as u64);
        g.def.OMGetRenderTargets(Some(&mut got), None);
        let mut list = None;
        g.def
            .FinishCommandList(false, Some(&mut list))
            .map_err(create_err("FinishCommandList"))?;
    }
    let raw = |v: &Option<ID3D11RenderTargetView>| v.as_ref().map_or(0, |v| v.as_raw() as usize);
    let ok = raw(&got[0]) == r0.as_raw() as usize
        && raw(&got[1]) == r1.as_raw() as usize
        && raw(&got[2]) == rtv.as_raw() as usize;
    report.push(format!(
        "  forward-pass binding: engine targets 0-1 kept and material target at slot 2: {ok}"
    ));
    if !ok {
        failures.push("forward-pass binding wrong".into());
    }
    material::reset_for_test();
    Ok(())
}

/// See the module documentation.
pub fn offline_check(size: [u32; 2], scale: f32) -> Result<SsrReport, String> {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let mut images = Vec::new();
    let g = Gpu::new()?;
    let code = bytecode()?;
    report.push(format!("ssr shaders compiled in {:.0} ms", code.compile_ms));
    let probe = Probe::new(&g)?;
    let depth_bind = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_DEPTH_STENCIL.0;
    let color_bind = D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0;
    let scaled = [
        crate::fsr_scale::snap_render(size[0], scale),
        crate::fsr_scale::snap_render(size[1], scale),
    ];
    let defaults = Settings::default();
    let cases = [
        (
            "native",
            size,
            Format {
                dxgi: DXGI_FORMAT_R11G11B10_FLOAT,
                bpp: 4,
            },
        ),
        (
            "scaled",
            scaled,
            Format {
                dxgi: DXGI_FORMAT_R16G16B16A16_FLOAT,
                bpp: 8,
            },
        ),
    ];
    for (case, region, fmt) in cases {
        let data = ao::engine_projection(
            region[0] as f32 / region[1] as f32,
            FOV_Y,
            NEAR,
            FAR,
            JITTER,
        );
        let proj = ao::projection(&data, 1)?;
        let sc = scene(&proj, size, region, 0.0, None);
        let depth = g.texture(
            size,
            DXGI_FORMAT_R24G8_TYPELESS,
            depth_bind,
            Some(&sc.depth),
        )?;
        let waters: Vec<ID3D11Texture2D> = sc
            .water
            .iter()
            .map(|w| g.texture(size, DXGI_FORMAT_R24G8_TYPELESS, depth_bind, Some(w)))
            .collect::<Result<_, _>>()?;
        let initial: Vec<u8> = sc.color.iter().flat_map(|c| fmt.encode(*c)).collect();
        // What the colour format stores of the scene colours (comparison baseline).
        let stored: Vec<[f32; 3]> = initial
            .chunks_exact(fmt.bpp)
            .map(|b| fmt.decode(b))
            .collect();
        let color = make_texture(
            &g,
            size,
            fmt.dxgi,
            color_bind,
            &initial,
            size[0] * fmt.bpp as u32,
        )?;
        let mut color_srv = None;
        // SAFETY: default view of a typed texture; valid out pointer.
        unsafe {
            g.dev
                .CreateShaderResourceView(&color, None, Some(&mut color_srv))
        }
        .map_err(create_err("colour SRV"))?;
        let color_srv = color_srv.ok_or("no colour SRV")?;
        let e = Engine::new(&g.def, &color, &depth, &waters[0], &data, region);
        let points = [
            ("box reflection", [0.0, -EYE, 3.05]),
            ("wall reflection", [-2.0, -EYE, 6.0]),
            ("water patch", [2.5, -EYE, 5.0]),
        ];
        let at: Vec<[u32; 2]> = points.iter().map(|p| pixel(&proj, region, p.1)).collect();
        report.push(format!(
            "\n{case}: texture {}x{}, region {}x{}, colour format {}; test pixels {at:?}",
            size[0], size[1], region[0], region[1], fmt.dxgi.0
        ));
        images.push(image(format!("ssr_{case}_input"), &stored, size, region));
        let mut engine_result = None;
        for half in [true, false] {
            for q in 0..QUALITIES.len() as u8 {
                settings::edit(|f| {
                    *f = Settings::default();
                    f.ssr = true;
                    f.ssr_quality = q;
                    f.ssr_half_res = half;
                });
                let label = format!(
                    "{} {} ({} steps)",
                    if half { "half" } else { "full" },
                    quality_name(q),
                    QUALITIES[usize::from(q)].1
                );
                let o = run_engine(
                    &g, &probe, &e, &color, &color_srv, &initial, &fmt, size, region, true,
                )?;
                let px = &o.pixels;
                let (mut nan, mut outside, mut other, mut water_changed, mut floor_changed) =
                    (0usize, 0usize, 0usize, 0usize, 0usize);
                for (i, p) in px.iter().enumerate() {
                    nan += usize::from(p.iter().any(|v| !v.is_finite()));
                    let changed = *p != stored[i];
                    match sc.surf[i] {
                        None => outside += usize::from(changed),
                        Some(Surf::Floor) => floor_changed += usize::from(changed),
                        Some(Surf::Water) => water_changed += usize::from(changed),
                        Some(Surf::BoxTop) => {}
                        Some(_) => other += usize::from(changed),
                    }
                }
                let vals: Vec<[f32; 3]> = at.iter().map(|&p| mean(px, size, p)).collect();
                report.push(format!(
                    "  {label:<22} GPU {:.3} ms | box refl {:.3?} wall refl {:.3?} water {:.3?} | floor px changed {floor_changed} | NaN {nan} outside-written {outside} non-floor-changed {other} water-changed {water_changed} | state {}",
                    o.gpu_ms.unwrap_or(f64::NAN),
                    vals[0],
                    vals[1],
                    vals[2],
                    if o.state_diff.is_empty() {
                        "restored".to_owned()
                    } else {
                        format!("CHANGED: {}", o.state_diff.join(", "))
                    }
                ));
                let mut check = |ok: bool, what: String| {
                    if !ok {
                        failures.push(format!("{case} {label}: {what}"));
                    }
                };
                check(
                    o.state_diff.is_empty(),
                    format!("state changed: {:?}", o.state_diff),
                );
                check(nan == 0, format!("{nan} non-finite pixels"));
                check(
                    outside == 0,
                    format!("{outside} pixels outside the region written"),
                );
                check(
                    other == 0,
                    format!("{other} wall/box-side/sky pixels changed"),
                );
                check(
                    water_changed == 0,
                    format!("{water_changed} water pixels changed"),
                );
                // Box: red rises, green stays low (reflection of a red face).
                check(
                    vals[0][0] > 2.0 * FLOOR[0] && vals[0][0] > 2.0 * vals[0][1],
                    format!("no box reflection at {:?}: {:?}", at[0], vals[0]),
                );
                // Wall: blue rises above red.
                check(
                    vals[1][2] > 1.3 * FLOOR[2] && vals[1][2] > 1.5 * vals[1][0],
                    format!("no wall reflection at {:?}: {:?}", at[1], vals[1]),
                );
                if half == defaults.ssr_half_res && q == defaults.ssr_quality {
                    images.push(image(
                        format!(
                            "ssr_{case}_{}_{}",
                            if half { "half" } else { "full" },
                            quality_name(q)
                        ),
                        px,
                        size,
                        region,
                    ));
                    engine_result = Some(o.pixels);
                }
            }
        }
        // FSR path: the default settings on the immediate context (as before FSR's pack).
        settings::edit(|f| {
            *f = Settings::default();
            f.ssr = true;
        });
        let before_submit = SUBMIT.load(Ordering::Relaxed);
        let pitch = size[0] * fmt.bpp as u32;
        // SAFETY: full-texture update with data of the texture's size and pitch.
        unsafe {
            g.imm
                .UpdateSubresource(&color, 0, None, initial.as_ptr().cast(), pitch, 0)
        };
        let job = capture(e.pass(), None).ok_or("FSR path: nothing captured")?;
        execute(&g.imm, job);
        let fsr_px: Vec<[f32; 3]> = read_raw(&g, &color, size, fmt.bpp)?
            .chunks_exact(fmt.bpp)
            .map(|b| fmt.decode(b))
            .collect();
        let same = engine_result.as_ref() == Some(&fsr_px);
        report.push(format!(
            "  FSR path (immediate context): ran {}, result equal to the engine path {same}",
            SUBMIT.load(Ordering::Relaxed) > before_submit
        ));
        if !same || SUBMIT.load(Ordering::Relaxed) == before_submit {
            failures.push(format!("{case}: FSR-path result differs or did not run"));
        }
        // Applied counter of both paths (the overlay's "Frames applied").
        let applied = status().applied;
        // Read-back of the water coverage (immediate context, a frame or more later).
        for _ in 0..3 {
            let job = capture(e.pass(), None).ok_or("FSR path: nothing captured")?;
            execute(&g.imm, job);
            // Waits for the GPU (staging read of the colour).
            read_raw(&g, &color, [1, 1], fmt.bpp)?;
        }
        let applied_after = status().applied;
        let want = sc.water_pixels as f32 / (region[0] * region[1]) as f32;
        let cov = status().water_coverage;
        report.push(format!(
            "  depthWater {:?}: {} water pixels = {:.2} % of the region, read back {:?}; applied counter +{}",
            WaterTarget::Real,
            sc.water_pixels,
            want * 100.0,
            cov.map(|c| c * 100.0),
            applied_after - applied
        ));
        if cov.is_none_or(|c| (c - want).abs() > 0.002) {
            failures.push(format!(
                "{case}: water coverage read back {cov:?}, expected {want:.4}"
            ));
        }
        if applied_after != applied + 3 {
            failures.push(format!(
                "{case}: applied counter {applied} -> {applied_after} after 3 frames"
            ));
        }
        // Other water-target contents: reflections must stay on.
        for (target, tex) in [
            (WaterTarget::Copy, &waters[1]),
            (WaterTarget::AllCloser, &waters[2]),
        ] {
            let ew = Engine::new(&g.def, &color, &depth, tex, &data, region);
            settings::edit(|f| {
                *f = Settings::default();
                f.ssr = true;
            });
            let applied = status().applied;
            let o = run_engine(
                &g, &probe, &ew, &color, &color_srv, &initial, &fmt, size, region, false,
            )?;
            let px = &o.pixels;
            let vals: Vec<[f32; 3]> = at.iter().map(|&p| mean(px, size, p)).collect();
            let (mut nan, mut outside, mut water_changed) = (0usize, 0usize, 0usize);
            for (i, p) in px.iter().enumerate() {
                nan += usize::from(p.iter().any(|v| !v.is_finite()));
                let changed = *p != stored[i];
                match sc.surf[i] {
                    None => outside += usize::from(changed),
                    Some(Surf::Water) => water_changed += usize::from(changed),
                    _ => {}
                }
            }
            let fallback = WATER_FALLBACK.load(Ordering::Relaxed);
            for _ in 0..3 {
                let job = capture(ew.pass(), None).ok_or("FSR path: nothing captured")?;
                execute(&g.imm, job);
                read_raw(&g, &color, [1, 1], fmt.bpp)?;
            }
            let fallback = WATER_FALLBACK.load(Ordering::Relaxed) - fallback;
            report.push(format!(
                "  depthWater {target:?}: box refl {:.3?} wall refl {:.3?} | water pixels changed {water_changed} | NaN {nan} outside-written {outside} | coverage read back {:?} %, fallback frames {fallback} | state {}",
                vals[0],
                vals[1],
                status().water_coverage.map(|c| c * 100.0),
                if o.state_diff.is_empty() {
                    "restored"
                } else {
                    "CHANGED"
                }
            ));
            let mut check = |ok: bool, what: String| {
                if !ok {
                    failures.push(format!("{case} depthWater {target:?}: {what}"));
                }
            };
            check(
                status().applied > applied,
                "applied counter unchanged".into(),
            );
            check(
                o.state_diff.is_empty(),
                format!("state changed: {:?}", o.state_diff),
            );
            check(
                nan == 0 && outside == 0,
                format!("NaN {nan}, outside {outside}"),
            );
            check(
                vals[0][0] > 2.0 * FLOOR[0] && vals[0][0] > 2.0 * vals[0][1],
                format!("no box reflection: {:?}", vals[0]),
            );
            check(
                vals[1][2] > 1.3 * FLOOR[2] && vals[1][2] > 1.5 * vals[1][0],
                format!("no wall reflection: {:?}", vals[1]),
            );
            // Neither variant masks the water: no water in the copy, mask ignored above the
            // limit.
            check(water_changed > 0, "water unchanged (mask applied)".into());
            match target {
                WaterTarget::AllCloser => {
                    check(fallback >= 1, "no fallback frame read back".into())
                }
                _ => check(fallback == 0, format!("{fallback} fallback frames")),
            }
        }
        if case == "native" {
            check_material(
                &g,
                &probe,
                &e,
                &color,
                &color_srv,
                &initial,
                &fmt,
                &sc,
                &proj,
                size,
                region,
                &at,
                engine_result.as_deref(),
                &mut report,
                &mut failures,
            )?;
        }
        // Water mask off: the patch reflects (the mask, not the scene, kept it unchanged).
        settings::edit(|f| {
            *f = Settings::default();
            f.ssr = true;
            f.ssr_skip_water = false;
        });
        let o = run_engine(
            &g, &probe, &e, &color, &color_srv, &initial, &fmt, size, region, true,
        )?;
        let water_changed = (0..o.pixels.len())
            .filter(|&i| sc.surf[i] == Some(Surf::Water) && o.pixels[i] != stored[i])
            .count();
        report.push(format!(
            "  ssr_skip_water = 0 (half medium): GPU {:.3} ms, water pixels changed {water_changed}",
            o.gpu_ms.unwrap_or(f64::NAN)
        ));
        if water_changed == 0 {
            failures.push(format!("{case}: water patch unchanged without the mask"));
        }
        // Off: nothing captured.
        settings::edit(|f| *f = Settings::default());
        if capture(e.pass(), None).is_some() {
            failures.push(format!("{case}: captured with ssr = 0"));
        }
        // Coverage, noise and ghosting (rewrites the scene textures: last).
        if case == "scaled" {
            temporal::check(
                &g,
                &probe,
                &color,
                &color_srv,
                &depth,
                &waters[0],
                &data,
                &proj,
                &fmt,
                size,
                region,
                &mut report,
                &mut failures,
                &mut images,
            )?;
        }
    }
    settings::edit(|f| *f = Settings::default());
    if failures.is_empty() {
        report.push("\nall checks passed".into());
    }
    Ok(SsrReport {
        text: report.join("\n"),
        images,
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_float_round_trip() {
        for v in [0.0f32, 0.08, 0.2, 1.0, 2.0, 3.0, 1000.0] {
            let c = unpack_r11g11b10(&pack_r11g11b10([v, v, v]));
            assert!((c[0] - v).abs() <= v * 0.02 + 1e-6, "{v} {c:?}");
            assert!((c[2] - v).abs() <= v * 0.04 + 1e-6, "{v} {c:?}");
            assert!((f16_value(f16_bits(v)) - v).abs() <= v * 0.001 + 1e-6);
        }
        assert!(small_to_f32(31 << 6 | 1, 6).is_nan());
    }
}
