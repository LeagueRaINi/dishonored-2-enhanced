//! Skinned characters (`.md6`, docs/formats/md6.md) as dynamic meshes (docs/raytracing.md
//! section 14): md6Def chain -> the meshes of one LOD (cloth-stripped `nocloth_*` builds
//! where the mesh has cloth) + the skeleton, decoded into a one-mesh [`Scene`]
//! ([`scene::mesh_scene`]: photo records, FC6 primitives, materials, textures) plus the
//! per-vertex skinning data the GPU kernel needs (`dynamic.hlsl`).
//!
//! Joint indices: skeleton joint = stored + the surface's joint offset (O, md6.md). Parts
//! built for another skeleton (skeleton checksum differs, the aristo02 heads) are remapped
//! by joint name. Palette convention (H, verified per character in game by
//! [`palette_check`]): palette[j] = joint world x inverse bind (bone -> object), so the bind
//! pose maps bind vertices to themselves.

use std::collections::HashMap;

use void_assets::Archive;
use void_formats::bwm::f16_to_f32;
use void_formats::md6::{self, MAX_INFLUENCES, Mat34, Model, Skeleton, Surface};

use crate::scene::{self, LoadOptions, PartSurface, Scene};

/// Per decoded surface: influences and normal frames per vertex.
type SkinData = (Vec<[u32; 4]>, Vec<[u32; 2]>);

/// Row-major 3x4 matrix as 12 floats (the palette row layout of the engine and the kernel).
pub type Row34 = [f32; 12];

pub const IDENTITY: Row34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

pub fn row34(m: &Mat34) -> Row34 {
    let mut r = [0f32; 12];
    for i in 0..3 {
        r[i * 4..i * 4 + 4].copy_from_slice(&m[i]);
    }
    r
}

/// `a * b` (affine).
pub fn mul_row34(a: &Row34, b: &Row34) -> Row34 {
    let mut m = [0f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            m[i * 4 + j] = (0..3).map(|k| a[i * 4 + k] * b[k * 4 + j]).sum::<f32>()
                + if j == 3 { a[i * 4 + 3] } else { 0.0 };
        }
    }
    m
}

pub fn apply_row34(m: &Row34, p: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i * 4] * p[0] + m[i * 4 + 1] * p[1] + m[i * 4 + 2] * p[2] + m[i * 4 + 3])
}

/// A decoded character: one mesh plus per-vertex skinning data (vertex order of
/// `mesh.positions`).
pub struct CharacterCpu {
    pub md6: String,
    pub lod: u32,
    /// One mesh, positions in bind pose (model space, metres, z up).
    pub mesh: Scene,
    /// Per vertex: joints 0-3 / 4-7 (u8, skeleton joints), weights 0-3 / 4-7 (unorm8).
    pub influences: Vec<[u32; 4]>,
    /// Per vertex: snorm8x4 normal, snorm8x4 tangent (as in the attribute blocks).
    pub frames: Vec<[u32; 2]>,
    /// Per triangle (record order): the vertex of each corner.
    pub tri_verts: Vec<[u32; 3]>,
    pub skeleton: Skeleton,
    /// Inverse bind and bind pose per skeleton joint.
    pub inverse_bind: Vec<Row34>,
    pub bind_pose: Vec<Row34>,
    /// Remaps, clamps and dropped surfaces (for the log).
    pub notes: Vec<String>,
    /// The md6mesh parts of the LOD (body, head, gear, ...).
    pub meshes: Vec<String>,
    /// Surfaces traced / decoded.
    pub surfaces: [u32; 2],
    /// Materials traced through the fallback (name, reason such as `program arkSkin`).
    pub fallbacks: Vec<(String, String)>,
}

impl CharacterCpu {
    pub fn joint_count(&self) -> u32 {
        self.inverse_bind.len() as u32
    }

    pub fn vertex_count(&self) -> u32 {
        self.mesh.positions.len() as u32
    }

    pub fn triangle_count(&self) -> u32 {
        self.tri_verts.len() as u32
    }

    /// The palette of the bind pose: joint world x inverse bind (identity up to rounding).
    pub fn bind_palette(&self) -> Vec<Row34> {
        self.bind_pose
            .iter()
            .zip(&self.inverse_bind)
            .map(|(b, i)| mul_row34(b, i))
            .collect()
    }

    /// Skinning palette for global joint matrices `world` (joint -> model space).
    pub fn palette_for(&self, world: &[Mat34]) -> Vec<Row34> {
        world
            .iter()
            .zip(&self.inverse_bind)
            .map(|(w, i)| mul_row34(&row34(w), i))
            .collect()
    }

    /// CPU reference of `skin_vertices`: vertex `v` skinned with `palette`.
    pub fn skin_vertex(&self, palette: &[Row34], v: usize) -> [f32; 3] {
        let p = self.mesh.positions[v];
        let inf = self.influences[v];
        let mut m = [0f32; 12];
        let mut wsum = 0.0;
        for k in 0..8 {
            let shift = 8 * (k & 3);
            let j = ((if k < 4 { inf[0] } else { inf[1] }) >> shift) & 0xff;
            let w = (((if k < 4 { inf[2] } else { inf[3] }) >> shift) & 0xff) as f32 / 255.0;
            if w > 0.0 {
                let pm = palette[(j as usize).min(palette.len().saturating_sub(1))];
                for (a, b) in m.iter_mut().zip(pm) {
                    *a += w * b;
                }
                wsum += w;
            }
        }
        if wsum <= 0.0 || palette.is_empty() {
            return p;
        }
        apply_row34(&m.map(|x| x / wsum), p)
    }
}

/// How the engine's CPU palette (`m_jointBufferCpu`) relates to the skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteMode {
    /// palette = joint world x inverse bind (bone -> object; the expected convention).
    Skinning,
    /// palette = joint world only (the inverse bind is applied here).
    JointWorld,
}

/// The palette to upload for an engine palette in `mode` (`engine.len()` must cover the
/// skeleton; extra entries are dropped).
pub fn effective_palette(cpu: &CharacterCpu, engine: &[Row34], mode: PaletteMode) -> Vec<Row34> {
    let n = cpu.inverse_bind.len().min(engine.len());
    match mode {
        PaletteMode::Skinning => engine[..n].to_vec(),
        PaletteMode::JointWorld => engine[..n]
            .iter()
            .zip(&cpu.inverse_bind)
            .map(|(w, i)| mul_row34(w, i))
            .collect(),
    }
}

/// Result of [`palette_check`]: the plausible mode (None: neither) and, per mode, the
/// diagonal of the skinned sample's bounds relative to the bind pose's and its centre
/// offset in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaletteVerdict {
    pub mode: Option<PaletteMode>,
    pub skinning: [f32; 2],
    pub joint_world: [f32; 2],
}

fn bounds(points: impl Iterator<Item = [f32; 3]>) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn diag(b: &([f32; 3], [f32; 3])) -> f32 {
    (0..3)
        .map(|k| (b.1[k] - b.0[k]).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn centre(b: &([f32; 3], [f32; 3])) -> [f32; 3] {
    [0, 1, 2].map(|k| 0.5 * (b.0[k] + b.1[k]))
}

/// Which palette convention makes a character's first engine palette plausible: the
/// skinned bounds of up to 512 sample vertices must stay within 0.5..2 x the bind-pose
/// diagonal and their centre within one bind diagonal of the bind centre (a pose moves
/// limbs, it does not shrink, explode or displace the body). Skinning wins ties.
pub fn palette_check(cpu: &CharacterCpu, engine: &[Row34]) -> PaletteVerdict {
    let n = cpu.mesh.positions.len();
    let step = (n / 512).max(1);
    let sample: Vec<usize> = (0..n).step_by(step).collect();
    let bind = bounds(sample.iter().map(|&v| cpu.mesh.positions[v]));
    let bd = diag(&bind).max(1e-3);
    let bc = centre(&bind);
    let eval = |mode| {
        let pal = effective_palette(cpu, engine, mode);
        let b = bounds(sample.iter().map(|&v| cpu.skin_vertex(&pal, v)));
        let c = centre(&b);
        let off = (0..3).map(|k| (c[k] - bc[k]).powi(2)).sum::<f32>().sqrt();
        [diag(&b) / bd, off]
    };
    let ok = |r: [f32; 2]| r[0].is_finite() && (0.5..=2.0).contains(&r[0]) && r[1] <= bd;
    let s = eval(PaletteMode::Skinning);
    let j = eval(PaletteMode::JointWorld);
    let mode = if ok(s) {
        Some(PaletteMode::Skinning)
    } else if ok(j) {
        Some(PaletteMode::JointWorld)
    } else {
        None
    };
    PaletteVerdict {
        mode,
        skinning: s,
        joint_world: j,
    }
}

/// Lookups shared by every character of a map: archive names of the cloth-stripped
/// `nocloth_*` mesh builds by mesh file stem (the first in name order, as `void-tool md6`).
pub struct Md6Index {
    nocloth: HashMap<String, String>,
}

impl Md6Index {
    pub fn new(archive: &Archive) -> Self {
        let mut nocloth = HashMap::new();
        for name in archive.find("baseModel", "nocloth_", ".bmd6model") {
            if let Some(stem) = md6::nocloth_stem(&name) {
                nocloth.entry(stem.to_owned()).or_insert(name.clone());
            }
        }
        Self { nocloth }
    }

    /// The built resource of a mesh reference: the direct build, or the cloth-stripped one
    /// for meshes with cloth (cloth parts are not decoded) or without a direct build.
    fn mesh_resource(&self, archive: &Archive, mesh: &md6::MeshRef) -> Result<String, String> {
        let direct =
            md6::mesh_resource(&mesh.mesh).ok_or_else(|| format!("{}: not .md6mesh", mesh.mesh))?;
        let has_direct = !archive.locate("baseModel", &direct).is_empty();
        let lower = mesh.mesh.to_ascii_lowercase();
        let stem = lower
            .strip_suffix(".md6mesh")
            .and_then(|s| s.rsplit('/').next())
            .unwrap_or_default();
        match (mesh.cloth.is_empty(), has_direct, self.nocloth.get(stem)) {
            (true, true, _) | (false, true, None) => Ok(direct),
            (_, _, Some(n)) => Ok(n.clone()),
            (_, false, None) => Err(format!("{}: no {direct} and no nocloth build", mesh.mesh)),
        }
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if v.is_nan() {
        return sign | 0x7e00;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        // Subnormal half: the 24-bit significand shifted down, rounded (ties up).
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        return sign | ((m >> shift) + ((m >> (shift - 1)) & 1)) as u16;
    }
    let h = u32::from(sign) | ((exp as u32) << 10) | (man >> 13);
    // Round to nearest (ties up); a carry into the exponent is the correct result.
    (h + ((man >> 12) & 1)) as u16
}

fn snorm8(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8
}

/// The 20-byte attribute blocks (uv0, uv1 f16x2; normal, tangent snorm8x4; colour) of an
/// md6 surface.
fn attribute_blocks(s: &Surface) -> Vec<[u8; 20]> {
    let v = &s.vertices;
    (0..v.positions.len())
        .map(|i| {
            let mut b = [0u8; 20];
            for (o, uv) in [(0, v.uv0.get(i)), (4, v.uv1.get(i))] {
                let uv = uv.copied().unwrap_or([0.0; 2]);
                b[o..o + 2].copy_from_slice(&f32_to_f16(uv[0]).to_le_bytes());
                b[o + 2..o + 4].copy_from_slice(&f32_to_f16(uv[1]).to_le_bytes());
            }
            let n = v.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0, 0.0]);
            let t = v.tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]);
            for k in 0..4 {
                b[8 + k] = snorm8(n[k]);
                b[12 + k] = snorm8(t[k]);
            }
            b[16..20].copy_from_slice(&v.colors.get(i).copied().unwrap_or([255; 4]));
            b
        })
        .collect()
}

fn read_skeleton(archive: &Archive, skl: &str) -> Result<Skeleton, String> {
    let res = md6::skeleton_resource(skl).ok_or_else(|| format!("{skl}: not .md6skl"))?;
    let data = archive.read("skeleton", &res)?;
    Skeleton::parse(&data).map_err(|e| format!("{res}: {e}"))
}

/// Loads character `md6` (an md6Def name, the engine model's name) at `lod` (the lowest
/// LOD when it has no such level). `materials`: the live surface materials in the
/// engine's order; used when their count equals the decoded surfaces', else the files'.
pub fn load_character(
    archive: &Archive,
    index: &Md6Index,
    md6_name: &str,
    lod: u32,
    materials: &[String],
    opts: LoadOptions,
) -> Result<CharacterCpu, String> {
    let def = md6::resolve_def(&|n| archive.read("md6Def", n), md6_name)?;
    let (&lod, meshes) = def
        .lods
        .get_key_value(&lod)
        .or_else(|| def.lods.iter().next())
        .ok_or_else(|| format!("{md6_name}: no LOD"))?;
    let mut notes = Vec::new();
    let mut models: Vec<Model> = Vec::new();
    for m in meshes {
        if !m.cloth.is_empty() {
            notes.push(format!("{}: cloth parts not traced", m.mesh));
        }
        let res = index.mesh_resource(archive, m)?;
        let data = archive.read("baseModel", &res)?;
        models.push(Model::parse(&data).map_err(|e| format!("{res}: {e}"))?);
    }
    let skl = def
        .skeleton
        .clone()
        .or_else(|| {
            models
                .iter()
                .map(|m| m.skeleton.clone())
                .find(|s| !s.is_empty())
        })
        .ok_or_else(|| format!("{md6_name}: no skeleton"))?;
    let skeleton = read_skeleton(archive, &skl)?;
    let joints = skeleton.joints.len();
    if joints == 0 || joints > 256 {
        return Err(format!("{md6_name}: {joints} skeleton joints"));
    }
    let by_name: HashMap<&str, usize> = skeleton
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();
    // Per decoded surface: the part surface, influences and frames per vertex.
    let mut parts: Vec<PartSurface> = Vec::new();
    let mut skin: Vec<SkinData> = Vec::new();
    let mut clamped = 0usize;
    for model in &models {
        // Parts built for another skeleton (checksum differs): remap by joint name.
        let remap: Option<Vec<usize>> = match model.skeleton_checksum {
            Some(c) if c != skeleton.model_checksum && !model.skeleton.is_empty() => {
                let own = read_skeleton(archive, &model.skeleton)?;
                let missing = own
                    .joints
                    .iter()
                    .filter(|j| !by_name.contains_key(j.name.as_str()))
                    .count();
                notes.push(format!(
                    "{}: built for skeleton checksum {c:#06x}, remapped by joint name ({missing} names missing -> root)",
                    model.skeleton
                ));
                Some(
                    own.joints
                        .iter()
                        .map(|j| by_name.get(j.name.as_str()).copied().unwrap_or(0))
                        .collect(),
                )
            }
            _ => None,
        };
        for s in &model.surfaces {
            let v = &s.vertices;
            let n = v.positions.len();
            if s.indices.is_empty() || n == 0 {
                continue;
            }
            let mut inf = Vec::with_capacity(n);
            for i in 0..n {
                let (j, w) = match (v.joints.get(i), v.weights.get(i)) {
                    (Some(j), Some(w)) => (*j, *w),
                    _ => ([0; MAX_INFLUENCES], [0; MAX_INFLUENCES]),
                };
                let mut out = [0u32; 4];
                for k in 0..MAX_INFLUENCES {
                    let mut g = usize::from(j[k]) + s.joint_offset as usize;
                    if let Some(r) = &remap {
                        g = r.get(g).copied().unwrap_or(0);
                    }
                    if g >= joints {
                        clamped += usize::from(w[k] > 0);
                        g = joints - 1;
                    }
                    out[k / 4] |= (g as u32) << (8 * (k % 4));
                    out[2 + k / 4] |= u32::from(w[k]) << (8 * (k % 4));
                }
                inf.push(out);
            }
            let blocks = attribute_blocks(s);
            let frames = blocks
                .iter()
                .map(|b| {
                    [
                        u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
                        u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
                    ]
                })
                .collect();
            parts.push(PartSurface {
                positions: v.positions.clone(),
                attributes: Some(blocks),
                indices: s.indices.clone(),
                material: s.material.clone(),
            });
            skin.push((inf, frames));
        }
    }
    if clamped > 0 {
        notes.push(format!("{clamped} influences beyond the skeleton clamped"));
    }
    let mats: Vec<String> = if materials.len() == parts.len() {
        materials.to_vec()
    } else {
        if !materials.is_empty() {
            notes.push(format!(
                "{} live materials for {} surfaces: file materials used",
                materials.len(),
                parts.len()
            ));
        }
        parts.iter().map(|p| p.material.clone()).collect()
    };
    let built = scene::mesh_scene(archive, md6_name, &parts, &mats, opts, true)?;
    let (mesh, kept) = (built.scene, built.kept);
    if kept.len() < parts.len() {
        notes.push(format!(
            "{} of {} surfaces not traced (invalid surface data)",
            parts.len() - kept.len(),
            parts.len()
        ));
    }
    let fallbacks = built.fallbacks;
    let mut influences = Vec::with_capacity(mesh.positions.len());
    let mut frames = Vec::with_capacity(mesh.positions.len());
    for &k in &kept {
        influences.extend_from_slice(&skin[k].0);
        frames.extend_from_slice(&skin[k].1);
    }
    if influences.len() != mesh.positions.len() {
        return Err(format!(
            "{md6_name}: {} skinned vertices for {} positions",
            influences.len(),
            mesh.positions.len()
        ));
    }
    let m = mesh.meshes.first().ok_or("no mesh")?;
    let mut tri_verts = Vec::with_capacity(mesh.records.len().max(mesh.primitives.len()));
    for g in &m.geometries {
        let idx = &mesh.indices[g.first_index as usize..(g.first_index + g.index_count) as usize];
        for t in idx.as_chunks::<3>().0 {
            tri_verts.push(t.map(|i| g.first_vertex + u32::from(i)));
        }
    }
    let inverse_bind = skeleton
        .joints
        .iter()
        .map(|j| row34(&j.inverse_bind))
        .collect();
    let bind_pose = skeleton.bind_pose().iter().map(row34).collect();
    Ok(CharacterCpu {
        md6: md6_name.to_owned(),
        lod,
        mesh,
        influences,
        frames,
        tri_verts,
        skeleton,
        inverse_bind,
        bind_pose,
        notes,
        meshes: meshes.iter().map(|m| m.mesh.clone()).collect(),
        surfaces: [kept.len() as u32, parts.len() as u32],
        fallbacks,
    })
}

/// Unpacked 8-bit joints / weights of influence word `inf` (tests and tools).
pub fn unpack_influences(inf: &[u32; 4]) -> [(u32, f32); 8] {
    std::array::from_fn(|k| {
        let shift = 8 * (k % 4);
        (
            (inf[k / 4] >> shift) & 0xff,
            ((inf[2 + k / 4] >> shift) & 0xff) as f32 / 255.0,
        )
    })
}

/// uv of attribute block `b` at byte `o` (tests).
pub fn block_uv(b: &[u8; 20], o: usize) -> [f32; 2] {
    let h = |k: usize| f16_to_f32(u16::from_le_bytes([b[o + k], b[o + k + 1]]));
    [h(0), h(2)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for v in [0.0f32, 1.0, -2.5, 0.333, 1e-5, 65504.0, 0.000_06] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
    }

    #[test]
    fn row34_products() {
        let t = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 3.0];
        assert_eq!(mul_row34(&IDENTITY, &t), t);
        assert_eq!(mul_row34(&t, &IDENTITY), t);
        assert_eq!(apply_row34(&t, [1.0, 2.0, 3.0]), [3.0, -3.0, 5.0]);
    }

    #[test]
    fn influences_pack_and_unpack() {
        let inf = [0x0403_0201, 0x0807_0605, 0x0000_00ff, 0];
        let u = unpack_influences(&inf);
        assert_eq!(u[0], (1, 1.0));
        assert_eq!(u[5].0, 6);
        assert_eq!(u[1].1, 0.0);
    }
}
