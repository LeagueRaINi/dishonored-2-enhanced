//! Offline check of the hybrid ray-traced reflections (`examples/rt_reflect_offline.rs
//! [--warp] [WxH]`): `void_rt`'s D3D12 context (hardware, or WARP as a smoke test of the
//! kernels), a synthetic world (Z up, metres: floor z = 0, a red box x 6..7, a blue back wall
//! x = 20, 12 m high, a gold floor patch) built into BLAS/TLAS with `void_rt::accel` and the
//! FC6 per-triangle data `rt::SceneView` carries; frame inputs made on the CPU exactly as the
//! D3D11 classify pass writes them (requests on the floor, per-channel weights);
//! [`Gpu12::record`] into the context's list as on FSR's list. Checks: request/trace/hit
//! counters against an analytic trace of the same rays, the box and the wall in the reflection
//! where the analytic rays hit them (on-screen hits take the frame colour), pixels without
//! weight bit-identical, no NaN, per-channel weights on the gold patch, the ray budget (25 %: a
//! quarter traced, the reflection still filled), shading without the screen path
//! (calibration), the debug views, GPU ms; then the denoiser over frame sequences
//! ([`denoise`]). Images to `void-rs/local/rt_reflect/`.

mod denoise;

use void_rt::accel;
use void_rt::gpu::Gpu;
use void_rt::scene::{Geometry, Instance, Mesh, Origin, Scene, SceneStats};

use super::*;

pub struct RtReport {
    pub text: String,
    pub images: Vec<crate::ssr::RgbImage>,
    pub failures: Vec<String>,
}

const EYE: [f32; 3] = [0.0, 0.0, 1.6];
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.01;
const FLOOR: [f32; 3] = [0.3, 0.3, 0.3];
const BOX: [f32; 3] = [0.8, 0.1, 0.1];
const WALL: [f32; 3] = [0.1, 0.2, 0.8];
const WALL_TOP: f32 = 12.0;
const SKY: [f32; 3] = [1.0, 1.0, 1.0];
/// Gold weights per channel (F0 1.0 / 0.78 / 0.34 at normal incidence).
const GOLD_TINT: [f32; 3] = [1.0, 0.78, 0.34];
const WEIGHT: f32 = 0.8;
/// BVH radius around the eye (the hits' confidence fades from 0.5 to 0.7 x it).
const RADIUS: f32 = 60.0;
/// Firefly check ([`denoise`]): small bright lamps (0.3 m squares, ~8 px) on the wall at world
/// y `LAMP_Y`, height `LAMP_Z`; on-screen hits take their colour.
static LAMPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const LAMP: [f32; 3] = [40.0, 34.0, 26.0];
const LAMP_Y: [f32; 5] = [-6.0, -3.0, 0.0, 3.0, 6.0];
const LAMP_Z: [f32; 2] = [2.0, 3.5];
const LAMP_HALF: f32 = 0.15;

fn lamp_at(p: [f32; 3]) -> bool {
    LAMP_Y.iter().any(|&y| (p[1] - y).abs() <= LAMP_HALF)
        && LAMP_Z.iter().any(|&z| (p[2] - z).abs() <= LAMP_HALF)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Hit {
    Floor,
    Box,
    Wall,
    Sky,
}

fn albedo(h: Hit) -> [f32; 3] {
    match h {
        Hit::Floor => FLOOR,
        Hit::Box => BOX,
        Hit::Wall => WALL,
        Hit::Sky => SKY,
    }
}

/// Closest hit of the analytic scene from `o` along `d` (world): (t, what).
fn trace(o: [f32; 3], d: [f32; 3]) -> (f32, Hit) {
    let mut best = (f32::INFINITY, Hit::Sky);
    let mut hit = |t: f32, ok: bool, h: Hit| {
        if ok && t > 1e-4 && t < best.0 {
            best = (t, h);
        }
    };
    if d[2] < 0.0 {
        let t = -o[2] / d[2];
        let (x, y) = (o[0] + d[0] * t, o[1] + d[1] * t);
        hit(
            t,
            (-5.0..=40.0).contains(&x) && (-20.0..=20.0).contains(&y),
            Hit::Floor,
        );
    }
    if d[0] > 0.0 {
        let t = (20.0 - o[0]) / d[0];
        let (y, z) = (o[1] + d[1] * t, o[2] + d[2] * t);
        hit(
            t,
            (-20.0..=20.0).contains(&y) && (0.0..=WALL_TOP).contains(&z),
            Hit::Wall,
        );
    }
    let (lo, hi) = ([6.0f32, -0.5, 0.0], [7.0f32, 0.5, 1.0]);
    let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        let (u, v) = ((lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    }
    hit(t0, t0 <= t1, Hit::Box);
    best
}

fn rot_to_world(rot: &[[f32; 4]; 3], v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| v[0] * rot[0][i] + v[1] * rot[1][i] + v[2] * rot[2][i])
}

fn rot_to_view(rot: &[[f32; 4]; 3], w: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| rot[r][0] * w[0] + rot[r][1] * w[1] + rot[r][2] * w[2])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|c| c / l)
}

/// A camera looking along +x, pitched down by `pitch` radians, with a sub-pixel jitter in NDC
/// (the engine's TAA jitter, `View::proj` zw).
#[derive(Clone, Copy)]
struct Cam {
    eye: [f32; 3],
    pitch: f32,
    jitter: [f32; 2],
}

impl Cam {
    const LEVEL: Cam = Cam {
        eye: EYE,
        pitch: 0.0,
        jitter: [0.0; 2],
    };

    /// World -> view rows (right = -y, up, forward).
    fn rot(&self) -> [[f32; 4]; 3] {
        let (s, c) = self.pitch.sin_cos();
        [[0.0, -1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [c, 0.0, -s, 0.0]]
    }
}

// ------------------------------------------------------------------------ scene --

fn oct(n: [f32; 3]) -> [u8; 2] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (x, y) = (n[0] / l1, n[1] / l1);
    let (x, y) = if n[2] >= 0.0 {
        (x, y)
    } else {
        ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum())
    };
    [x, y].map(|v| ((v * 0.5 + 0.5) * 255.0).round() as u8)
}

fn unorm(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Quads (4 corners counter-clockwise seen from `n`) of one object.
struct Object {
    quads: Vec<[[f32; 3]; 4]>,
    normals: Vec<[f32; 3]>,
    albedo: [f32; 3],
}

fn box_object() -> Object {
    let (a, b) = ([6.0f32, -0.5, 0.0], [7.0f32, 0.5, 1.0]);
    let c = |x: usize, y: usize, z: usize| [[a, b][x][0], [a, b][y][1], [a, b][z][2]];
    Object {
        quads: vec![
            [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)], // -x
            [c(1, 0, 0), c(1, 0, 1), c(1, 1, 1), c(1, 1, 0)], // +x
            [c(0, 0, 0), c(0, 0, 1), c(1, 0, 1), c(1, 0, 0)], // -y
            [c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)], // +y
            [c(0, 0, 1), c(0, 1, 1), c(1, 1, 1), c(1, 0, 1)], // +z
        ],
        normals: vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        albedo: BOX,
    }
}

/// The synthetic scene in `void_rt`'s layout: one mesh (one geometry) per object, identity
/// instances (two-sided), FC6 primitive words per triangle.
fn build_scene() -> Scene {
    let objects = [
        Object {
            quads: vec![[
                [-5.0, -20.0, 0.0],
                [40.0, -20.0, 0.0],
                [40.0, 20.0, 0.0],
                [-5.0, 20.0, 0.0],
            ]],
            normals: vec![[0.0, 0.0, 1.0]],
            albedo: FLOOR,
        },
        box_object(),
        Object {
            quads: vec![[
                [20.0, -20.0, 0.0],
                [20.0, -20.0, WALL_TOP],
                [20.0, 20.0, WALL_TOP],
                [20.0, 20.0, 0.0],
            ]],
            normals: vec![[-1.0, 0.0, 0.0]],
            albedo: WALL,
        },
    ];
    let mut s = Scene {
        positions: Vec::new(),
        indices: Vec::new(),
        primitives: Vec::new(),
        records: Vec::new(),
        meshes: Vec::new(),
        primitive_offsets: Vec::new(),
        geometry_materials: Vec::new(),
        instances: Vec::new(),
        materials: Vec::new(),
        gpu_materials: Vec::new(),
        textures: Vec::new(),
        stats: SceneStats::default(),
    };
    for (m, o) in objects.iter().enumerate() {
        let first_vertex = s.positions.len() as u32;
        let first_index = s.indices.len() as u32;
        let first_primitive = s.primitives.len() as u32;
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for (q, n) in o.quads.iter().zip(&o.normals) {
            let base = (s.positions.len() as u32 - first_vertex) as u16;
            for p in q {
                s.positions.push(*p);
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            s.indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
            let [nx, ny] = oct(*n);
            let w0 = u32::from_le_bytes([
                unorm(o.albedo[0]),
                unorm(o.albedo[1]),
                unorm(o.albedo[2]),
                128,
            ]);
            let w1 = u32::from_le_bytes([nx, ny, 0, 10]);
            for _ in 0..2 {
                s.primitives.push([w0, w1, w0, w1, w0, w1]);
            }
        }
        let geometry = Geometry {
            first_vertex,
            vertex_count: s.positions.len() as u32 - first_vertex,
            first_index,
            index_count: s.indices.len() as u32 - first_index,
            first_primitive,
            opaque: true,
        };
        s.meshes.push(Mesh {
            name: format!("object {m}"),
            geometries: vec![geometry],
            first_geometry: m as u32,
            aabb_min: lo,
            aabb_max: hi,
        });
        s.primitive_offsets.push(first_primitive);
        s.geometry_materials.push(0);
        s.instances.push(Instance {
            mesh: m as u32,
            transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            origin: Origin::World,
            two_sided: true,
        });
    }
    s
}

// --------------------------------------------------------------------- textures --

fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if v.is_nan() {
        return 0x7e00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let r = (exp as u32) << 10 | (man + 0x1000) >> 13;
    sign | r.min(0x7bff) as u16
}

fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A default-heap texture in COMMON with `data` (tightly packed rows of `bpp` bytes).
fn upload_texture(
    gpu: &mut Gpu,
    format: DXGI_FORMAT,
    size: [u32; 2],
    bpp: u32,
    data: &[u8],
) -> Result<ID3D12Resource, String> {
    let tex = texture(&gpu.device, size, format, D3D12_RESOURCE_STATE_COPY_DEST)?;
    let pitch = (size[0] * bpp).next_multiple_of(256);
    let mut staging = vec![0u8; (pitch * size[1]) as usize];
    for y in 0..size[1] as usize {
        let row = (size[0] * bpp) as usize;
        staging[y * pitch as usize..y * pitch as usize + row]
            .copy_from_slice(&data[y * row..(y + 1) * row]);
    }
    let up = gpu.upload_heap(&staging)?;
    // SAFETY (whole block): copy between live resources in the open list, then a transition
    // from the state the texture was created in.
    unsafe {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(up.res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: size[0],
                        Height: size[1],
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let list: ID3D12GraphicsCommandList = gpu.list.cast().map_err(|e| e.to_string())?;
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        drop(std::mem::ManuallyDrop::into_inner(src.pResource));
        drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
        barriers(
            &list,
            vec![transition(
                &tex,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            )],
        );
    }
    gpu.submit()?;
    Ok(tex)
}

/// Reads an RGBA16F texture in COMMON back as linear RGB.
fn read_rgba16f(
    gpu: &mut Gpu,
    tex: &ID3D12Resource,
    size: [u32; 2],
) -> Result<Vec<[f32; 4]>, String> {
    let pitch = (size[0] * 8).next_multiple_of(256);
    let rb = buffer(
        &gpu.device,
        u64::from(pitch * size[1]),
        D3D12_HEAP_TYPE_READBACK,
        false,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    // SAFETY (whole block): copy of a live texture into our readback buffer, transitions from
    // and back to COMMON, map after the blocking submit.
    unsafe {
        let list: ID3D12GraphicsCommandList = gpu.list.cast().map_err(|e| e.to_string())?;
        barriers(
            &list,
            vec![transition(
                tex,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )],
        );
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(rb.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: size[0],
                        Height: size[1],
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        drop(std::mem::ManuallyDrop::into_inner(src.pResource));
        drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
        barriers(
            &list,
            vec![transition(
                tex,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )],
        );
    }
    gpu.submit()?;
    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut out = Vec::with_capacity((size[0] * size[1]) as usize);
    // SAFETY: mapping the readback buffer written by the completed submit.
    unsafe {
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        for y in 0..size[1] as usize {
            let row = std::slice::from_raw_parts(
                p.cast::<u8>().add(y * pitch as usize).cast::<u16>(),
                size[0] as usize * 4,
            );
            for x in 0..size[0] as usize {
                out.push(std::array::from_fn(|c| f16_value(row[x * 4 + c])));
            }
        }
        rb.Unmap(0, None);
    }
    Ok(out)
}

fn bytes16(v: &[[f32; 4]]) -> Vec<u8> {
    v.iter()
        .flat_map(|c| c.iter().flat_map(|x| f16(*x).to_le_bytes()))
        .collect()
}

fn bytes32(v: &[[f32; 4]]) -> Vec<u8> {
    v.iter()
        .flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes()))
        .collect()
}

/// One frame's inputs on the GPU (COMMON) and what the RGBA16F colour stores of its colours.
struct FrameTextures {
    color: ID3D12Resource,
    depth: ID3D12Resource,
    motion: ID3D12Resource,
    req0: ID3D12Resource,
    req1: ID3D12Resource,
    rtw: ID3D12Resource,
    stored: Vec<[f32; 4]>,
}

fn upload_frame(gpu: &mut Gpu, f: &Frame, size: [u32; 2]) -> Result<FrameTextures, String> {
    let half = [size[0].div_ceil(2), size[1].div_ceil(2)];
    let rgba16 = DXGI_FORMAT_R16G16B16A16_FLOAT;
    let depth: Vec<u8> = f.depth.iter().flat_map(|d| d.to_le_bytes()).collect();
    let motion: Vec<u8> = f
        .motion
        .iter()
        .flat_map(|m| m.iter().flat_map(|v| f16(*v).to_le_bytes()))
        .collect();
    Ok(FrameTextures {
        color: upload_texture(gpu, rgba16, size, 8, &bytes16(&f.color))?,
        depth: upload_texture(gpu, DXGI_FORMAT_R32_FLOAT, size, 4, &depth)?,
        motion: upload_texture(gpu, DXGI_FORMAT_R16G16_FLOAT, size, 4, &motion)?,
        req0: upload_texture(
            gpu,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            half,
            16,
            &bytes32(&f.req0),
        )?,
        req1: upload_texture(gpu, rgba16, half, 8, &bytes16(&f.req1))?,
        rtw: upload_texture(gpu, rgba16, size, 8, &bytes16(&f.weight))?,
        stored: f
            .color
            .iter()
            .map(|c| c.map(|v| f16_value(f16(v))))
            .collect(),
    })
}

// ------------------------------------------------------------------------ frame --

/// One frame's CPU inputs: what FSR packs (colour, hardware depth, engine motion vectors) and
/// what the D3D11 classify pass writes (requests, weights), plus the analytic outcome of every
/// request.
struct Frame {
    color: Vec<[f32; 4]>,
    depth: Vec<f32>,
    /// Engine convention: current - previous position in viewport UV (y down).
    motion: Vec<[f32; 2]>,
    req0: Vec<[f32; 4]>,
    req1: Vec<[f32; 4]>,
    weight: Vec<[f32; 4]>,
    /// Per half texel: what its (mirror) ray hits (None = no request) and whether that hit is
    /// inside the frame.
    outcome: Vec<Option<Hit>>,
    on_screen: Vec<bool>,
    /// Per pixel: on the gold patch.
    gold: Vec<bool>,
}

fn proj(size: [u32; 2]) -> (f32, f32) {
    let p11 = 1.0 / (FOV_Y * 0.5).tan();
    (p11 * size[1] as f32 / size[0] as f32, p11)
}

/// Viewport UV (y down) of world point `w` seen by `cam` without jitter (None behind it).
fn uv_of(size: [u32; 2], cam: &Cam, w: [f32; 3]) -> Option<[f32; 2]> {
    let (p00, p11) = proj(size);
    let v = rot_to_view(&cam.rot(), [0, 1, 2].map(|k| w[k] - cam.eye[k]));
    (v[2] > 1e-4).then(|| [p00 * v[0] / v[2] * 0.5 + 0.5, 0.5 - 0.5 * p11 * v[1] / v[2]])
}

fn frame(size: [u32; 2]) -> Frame {
    frame_cam(size, &Cam::LEVEL, None, 0.0)
}

/// The inputs seen by `cam` (motion vectors against `prev`, 0 without), floor requests with GGX
/// `alpha`.
fn frame_cam(size: [u32; 2], cam: &Cam, prev: Option<&Cam>, alpha: f32) -> Frame {
    let (p00, p11) = proj(size);
    let rot = cam.rot();
    let n = (size[0] * size[1]) as usize;
    let half = [size[0].div_ceil(2), size[1].div_ceil(2)];
    let mut f = Frame {
        color: Vec::with_capacity(n),
        depth: Vec::with_capacity(n),
        motion: vec![[0.0; 2]; n],
        req0: vec![[0.0; 4]; (half[0] * half[1]) as usize],
        req1: vec![[0.0; 4]; (half[0] * half[1]) as usize],
        weight: vec![[0.0; 4]; n],
        outcome: vec![None; (half[0] * half[1]) as usize],
        on_screen: vec![false; (half[0] * half[1]) as usize],
        gold: vec![false; n],
    };
    // The floor normal (world up) in the view frame.
    let up = rot_to_view(&rot, [0.0, 0.0, 1.0]);
    for y in 0..size[1] {
        for x in 0..size[0] {
            let ndc = [
                (x as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / size[1] as f32 * 2.0,
            ];
            let v = [
                (ndc[0] - cam.jitter[0]) / p00,
                (ndc[1] - cam.jitter[1]) / p11,
                1.0,
            ];
            let d = rot_to_world(&rot, v);
            let (t, h) = trace(cam.eye, d);
            // `v` has view depth 1, so t is the view depth.
            let wp: [f32; 3] = std::array::from_fn(|k| cam.eye[k] + d[k] * t);
            let lit = if h == Hit::Wall
                && LAMPS.load(std::sync::atomic::Ordering::Relaxed)
                && lamp_at(wp)
            {
                LAMP
            } else {
                albedo(h).map(|c| c * 0.6)
            };
            f.color.push([lit[0], lit[1], lit[2], 1.0]);
            f.depth.push(if t.is_finite() { NEAR / t } else { 0.0 });
            let i = (y * size[0] + x) as usize;
            if let Some(pc) = prev
                && t.is_finite()
                && let (Some(a), Some(b)) = (uv_of(size, cam, wp), uv_of(size, pc, wp))
            {
                f.motion[i] = [a[0] - b[0], a[1] - b[1]];
            }
            if h != Hit::Floor {
                continue;
            }
            let gold = wp[1] > 1.5 && wp[1] < 4.0;
            f.gold[i] = gold;
            let tint = if gold { GOLD_TINT } else { [1.0; 3] };
            f.weight[i] = [WEIGHT * tint[0], WEIGHT * tint[1], WEIGHT * tint[2], WEIGHT];
            if x % 2 == 0 && y % 2 == 0 {
                let q = ((y / 2) * half[0] + x / 2) as usize;
                let p = v.map(|c| c * t);
                // Request: origin pushed off the floor along its normal, the normal and the
                // GGX alpha; the kernel reflects the view direction of the biased origin.
                let bias = (0.003 * t).max(0.02);
                let o_view: [f32; 3] = std::array::from_fn(|k| p[k] + up[k] * bias);
                f.req0[q] = [o_view[0], o_view[1], o_view[2], (0.005 * t).max(0.05)];
                f.req1[q] = [up[0], up[1], up[2], alpha];
                let vn = normalize(o_view);
                let dn = vn[0] * up[0] + vn[1] * up[1] + vn[2] * up[2];
                let r: [f32; 3] = std::array::from_fn(|k| vn[k] - 2.0 * dn * up[k]);
                let o = [wp[0], wp[1], bias];
                let rw = rot_to_world(&rot, r);
                let (th, hit) = trace(o, rw);
                f.outcome[q] = Some(hit);
                let hp: [f32; 3] = std::array::from_fn(|k| o[k] + rw[k] * th);
                f.on_screen[q] = th.is_finite()
                    && uv_of(size, cam, hp)
                        .is_some_and(|u| u.iter().all(|c| (0.0..1.0).contains(c)));
            }
        }
    }
    f
}

fn tonemap(c: [f32; 4]) -> [u8; 3] {
    [c[0], c[1], c[2]]
        .map(|v| ((v / (1.0 + v)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8)
}

fn image(name: &str, px: &[[f32; 4]], size: [u32; 2]) -> crate::ssr::RgbImage {
    crate::ssr::RgbImage {
        name: name.to_owned(),
        width: size[0],
        height: size[1],
        rgb: px.iter().flat_map(|c| tonemap(*c)).collect(),
    }
}

/// See the module documentation.
pub fn offline_check(warp: bool, size: [u32; 2]) -> Result<RtReport, String> {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let mut images = Vec::new();
    let mut gpu = Gpu::new(None, warp)?;
    report.push(format!(
        "device: {} ({}), ray tracing tier {}, shader model {}",
        gpu.info.name,
        if warp { "WARP" } else { "hardware" },
        gpu.info.raytracing_tier_name(),
        gpu.info.shader_model_name()
    ));
    if !gpu.info.inline_raytracing() {
        return Err("no DXR 1.1 inline ray tracing on this device".into());
    }
    // Scene.
    let scene = build_scene();
    let pos = gpu.upload(&scene.positions, false)?;
    let idx_words: Vec<u16> = scene.indices.clone();
    let idx = gpu.upload(&idx_words, false)?;
    let blas = accel::build_blas(&mut gpu, &scene, &pos, &idx, 64 << 20)?;
    let descs = accel::instance_descs(&scene, &blas, true, |_| true);
    let tlas = accel::Tlas::build(&mut gpu, &descs, descs.len())?;
    let prims = gpu.upload(&scene.primitives, false)?;
    let offsets = gpu.upload(&scene.primitive_offsets, false)?;
    // Instance types by TLAS instance index (floor, box, wall): static, moved, character.
    let types = gpu.upload(&[0u32, 1, 3], false)?;
    report.push(format!(
        "scene: {} triangles in {} BLAS, TLAS {} instances",
        scene.primitives.len(),
        scene.meshes.len(),
        tlas.count()
    ));
    // Frame inputs.
    let f = frame(size);
    let half = [size[0].div_ceil(2), size[1].div_ceil(2)];
    let tex = upload_frame(&mut gpu, &f, size)?;
    let initial = bytes16(&f.color);
    let stored = tex.stored.clone();
    let (color, depth, motion, req0, req1, rtw) = (
        tex.color, tex.depth, tex.motion, tex.req0, tex.req1, tex.rtw,
    );
    let (p00, p11) = proj(size);
    let view = |frame: u32| View {
        cam: EYE,
        rot: Cam::LEVEL.rot(),
        proj: [p00, p11, 0.0, 0.0],
        depth: [NEAR, 0.0],
        region: [0, 0, size[0], size[1]],
        frame,
        temporal: false,
    };
    let sc = SceneRef {
        tlas_va: tlas.va(),
        center: EYE,
        radius: RADIUS,
        prim_offsets: &offsets.res,
        geom_count: scene.primitive_offsets.len() as u32,
        prims: &prims.res,
        prim_count: scene.primitives.len() as u32,
        sun_dir: normalize([0.3, 0.2, 0.9]),
        sun_irradiance: [3.0; 3],
        ambient: [0.3; 3],
        instance_types: Some(&types.res),
        instance_type_count: 3,
    };
    let inputs = Inputs {
        color: &color,
        depth: &depth,
        motion: &motion,
        req0: &req0,
        req1: &req1,
        rtw: &rtw,
    };
    let device = gpu.device.clone();
    let mut g12 = Gpu12::new(&device)?;
    let list: ID3D12GraphicsCommandList = gpu.list.cast().map_err(|e| e.to_string())?;
    let ts_freq = gpu.ts_freq as u64;
    let mut fence = 0u64;
    let pristine = upload_texture(&mut gpu, DXGI_FORMAT_R16G16B16A16_FLOAT, size, 8, &initial)?;
    // One frame: restore the colour, record, submit (blocking); returns the colour.
    let mut run =
        |gpu: &mut Gpu, knobs: &Knobs, mode: &Mode, frame: u32| -> Result<Vec<[f32; 4]>, String> {
            let (common, src, dst) = (
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            // SAFETY: copy between two live textures of identical description, COMMON before and
            // after.
            unsafe {
                barriers(
                    &list,
                    vec![
                        transition(&color, common, dst),
                        transition(&pristine, common, src),
                    ],
                );
                list.CopyResource(&color, &pristine);
                barriers(
                    &list,
                    vec![
                        transition(&color, dst, common),
                        transition(&pristine, src, common),
                    ],
                );
            }
            let recorded = g12.record(
                &list,
                fence,
                fence + 1,
                ts_freq,
                &view(frame),
                &inputs,
                &sc,
                knobs,
                mode,
                Box::new(()),
            )?;
            if !recorded {
                return Err("frame not recorded".into());
            }
            gpu.submit()?;
            fence += 1;
            read_rgba16f(gpu, &color, size)
        };
    let knobs = |debug: u8, flags: u32| Knobs { debug, flags };
    let live = |budget: u32| Mode::Live { budget };
    // Main run (a few frames: the counters of a frame are read back when its ring slot is
    // reused).
    let mut out = Vec::new();
    for k in 0..(SETS as u32 + 2) {
        out = run(&mut gpu, &knobs(0, F_SHADOW), &live(100), k)?;
    }
    let stats = status().last;
    let requested = f.outcome.iter().filter(|o| o.is_some()).count() as u32;
    let analytic_hits = f
        .outcome
        .iter()
        .filter(|o| matches!(o, Some(Hit::Box | Hit::Wall | Hit::Floor)))
        .count() as u32;
    report.push(format!(
        "{}x{}: requests {} (CPU {requested}), traced {}, hits {} (analytic {analytic_hits}), on-screen hits {}, misses {}, calibration scale {:.3}, GPU {:.3} ms",
        size[0],
        size[1],
        stats.requested,
        stats.traced,
        stats.hits,
        stats.screen_hits,
        stats.misses,
        stats.scale,
        status().gpu_ms.unwrap_or(f32::NAN)
    ));
    let mut check = |ok: bool, what: String| {
        if !ok {
            failures.push(what);
        }
    };
    check(
        stats.requested == requested && stats.traced == requested,
        format!(
            "requested/traced {}/{} vs {requested}",
            stats.requested, stats.traced
        ),
    );
    check(
        stats.hits + stats.misses == stats.traced
            && stats.hits.abs_diff(analytic_hits) <= requested / 50 + 4,
        format!(
            "hits {} misses {} vs analytic {analytic_hits}",
            stats.hits, stats.misses
        ),
    );
    // Unweighted pixels bit-identical, no NaN.
    let (mut nan, mut unweighted_changed) = (0usize, 0usize);
    for i in 0..out.len() {
        nan += usize::from(out[i].iter().any(|v| !v.is_finite()));
        if f.weight[i][3] == 0.0 && out[i] != stored[i] {
            unweighted_changed += 1;
        }
    }
    check(nan == 0, format!("{nan} non-finite pixels"));
    check(
        unweighted_changed == 0,
        format!("{unweighted_changed} pixels without weight changed"),
    );
    // Reflections of the box and the wall where the analytic rays hit them (not gold).
    let mean_at = |px: &[[f32; 4]], what: Hit| -> ([f32; 3], usize) {
        let mut s = [0f32; 3];
        let mut n = 0usize;
        for (q, o) in f.outcome.iter().enumerate() {
            if *o != Some(what) {
                continue;
            }
            let (x, y) = ((q as u32 % half[0]) * 2, (q as u32 / half[0]) * 2);
            let i = (y * size[0] + x) as usize;
            if f.gold[i] {
                continue;
            }
            for c in 0..3 {
                s[c] += px[i][c];
            }
            n += 1;
        }
        (s.map(|v| v / n.max(1) as f32), n)
    };
    let (box_refl, nb) = mean_at(&out, Hit::Box);
    let (wall_refl, nw) = mean_at(&out, Hit::Wall);
    report.push(format!(
        "  box reflection {box_refl:.3?} ({nb} px), wall reflection {wall_refl:.3?} ({nw} px), floor {:.3?}",
        FLOOR.map(|c| c * 0.6)
    ));
    check(
        nb > 20 && box_refl[0] > 2.0 * box_refl[1],
        format!("no box reflection: {box_refl:?} ({nb} px)"),
    );
    check(
        nw > 20 && wall_refl[2] > 1.5 * wall_refl[0],
        format!("no wall reflection: {wall_refl:?} ({nw} px)"),
    );
    // Gold patch: per-channel weights from the change toward the wall colour.
    let (mut dw, mut n_gold) = ([0f64; 3], 0usize);
    for (q, o) in f.outcome.iter().enumerate() {
        let (x, y) = ((q as u32 % half[0]) * 2, (q as u32 / half[0]) * 2);
        let i = (y * size[0] + x) as usize;
        if *o != Some(Hit::Wall) || !f.gold[i] {
            continue;
        }
        let target = WALL.map(|c| c * 0.6);
        for c in 0..3 {
            dw[c] += f64::from((out[i][c] - stored[i][c]) / (target[c] - stored[i][c]));
        }
        n_gold += 1;
    }
    let tint = [dw[0] / dw[2].max(1e-9), dw[1] / dw[2].max(1e-9), 1.0];
    report.push(format!(
        "  gold patch ({n_gold} px reflecting the wall): weight r:g:b {:.2}:{:.2}:1 (expected {:.2}:{:.2}:1)",
        tint[0],
        tint[1],
        GOLD_TINT[0] / GOLD_TINT[2],
        GOLD_TINT[1] / GOLD_TINT[2]
    ));
    check(
        n_gold > 20 && tint[0] > 2.0 && tint[1] > 1.6,
        format!("gold patch not tinted: {tint:?} ({n_gold} px)"),
    );
    images.push(image("rt_input", &stored, size));
    images.push(image("rt_result", &out, size));
    // Budget 25 %.
    let mut b = Vec::new();
    for k in 0..(SETS as u32 + 1) {
        b = run(&mut gpu, &knobs(0, F_SHADOW), &live(25), 100 + k)?;
    }
    let st = status().last;
    let cap = half[0] * half[1];
    let max_rays = (cap * 25).div_ceil(100);
    let (box_b, _) = mean_at(&b, Hit::Box);
    report.push(format!(
        "  budget 25 %: traced {} of {} (max {max_rays}), box reflection {box_b:.3?}",
        st.traced, st.requested
    ));
    check(
        st.traced == st.requested.min(max_rays) && box_b[0] > 2.0 * box_b[1],
        format!(
            "budget: traced {} of {}, box {box_b:?}",
            st.traced, st.requested
        ),
    );
    // Shading only (no on-screen colour): the calibration takes over.
    let mut s = Vec::new();
    for k in 0..(SETS as u32 + 1) {
        s = run(
            &mut gpu,
            &knobs(0, F_SHADOW | F_NO_SCREEN),
            &live(100),
            200 + k,
        )?;
    }
    let (box_s, _) = mean_at(&s, Hit::Box);
    report.push(format!(
        "  shaded hits only: box reflection {box_s:.3?}, scale {:.3}",
        status().last.scale
    ));
    check(
        box_s[0] > 1.5 * box_s[1] && box_s.iter().all(|v| v.is_finite()),
        format!("shaded box reflection {box_s:?}"),
    );
    images.push(image("rt_shaded", &s, size));
    // Debug views.
    for (d, name) in DEBUG_VIEWS.iter().enumerate().skip(1) {
        let o = run(
            &mut gpu,
            &knobs(d as u8, F_SHADOW),
            &live(100),
            300 + d as u32,
        )?;
        let finite = o.iter().all(|c| c.iter().all(|v| v.is_finite()));
        let differs = o != out;
        report.push(format!(
            "  debug view {name}: finite {finite}, differs from the result {differs}"
        ));
        check(finite && differs, format!("debug view {name}"));
        images.push(image(&format!("rt_debug_{name}"), &o, size));
        if *name == "instances" {
            // Box reflections yellow (moved), wall reflections red (character); linear values
            // of void_rt::dynamic::type_colour.
            let (b, nb) = mean_at(&o, Hit::Box);
            let (wl, nw) = mean_at(&o, Hit::Wall);
            report.push(format!(
                "  debug view instances: box reflection {b:.3?} ({nb} px, moved = yellow [0.83, 0.61, 0.02]), wall reflection {wl:.3?} ({nw} px, character = red [0.75, 0.02, 0.02])"
            ));
            let near = |a: [f32; 3], e: [f32; 3]| (0..3).all(|c| (a[c] - e[c]).abs() < 0.08);
            check(
                nb > 20 && nw > 20 && near(b, [0.83, 0.61, 0.02]) && near(wl, [0.75, 0.02, 0.02]),
                format!("instances view: box {b:?}, wall {wl:?}"),
            );
        }
    }
    // Photo accumulation: 4 calls x 8 samples into a persistent RGBA32F sum; mirror requests
    // trace the same ray every sample, so the average equals the live result; repeatable.
    let accum = texture(
        &gpu.device,
        half,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let photo = |total: u32, composite: bool| Mode::Photo {
        accum: &accum,
        samples: 8,
        total,
        composite,
    };
    let mut p = Vec::new();
    for (k, total) in [0u32, 8, 16, 24].into_iter().enumerate() {
        p = run(
            &mut gpu,
            &knobs(0, F_SHADOW),
            &photo(total, true),
            400 + k as u32,
        )?;
    }
    let mut p2 = Vec::new();
    for (k, total) in [0u32, 8, 16, 24].into_iter().enumerate() {
        p2 = run(
            &mut gpu,
            &knobs(0, F_SHADOW),
            &photo(total, true),
            500 + k as u32,
        )?;
    }
    let max_diff = p
        .iter()
        .zip(&out)
        .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
        .fold(0f32, f32::max);
    report.push(format!(
        "  photo 4 x 8 spp: max |photo - live| {max_diff:.4} (mirror requests), repeatable {}",
        p == p2
    ));
    check(
        max_diff < 0.01 && p == p2,
        format!("photo accumulation differs from live by {max_diff} or is not repeatable"),
    );
    images.push(image("rt_photo", &p, size));
    denoise::check(
        &mut gpu,
        &sc,
        size,
        warp,
        &mut report,
        &mut failures,
        &mut images,
    )?;
    if failures.is_empty() {
        report.push("all checks passed".into());
    }
    Ok(RtReport {
        text: report.join("\n"),
        images,
        failures,
    })
}
