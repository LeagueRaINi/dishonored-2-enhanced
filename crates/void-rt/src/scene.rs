//! Static scene for ray tracing, assembled from the surviving decoders: the map's `.bwm`
//! world and static entity models (`.bmodel`).
//!
//! Layout follows AMD's Far Cry 6 hybrid reflections (GDC 2022, see `docs/raytracing.md`):
//! one BLAS per object (a `.bwm` surface, or an entity model with its kept surfaces), static
//! BLASes built once; lighting inputs are not fetched from materials at hit time but stored
//! per vertex in an unwrapped *primitive buffer* (24 bytes per triangle: three 8-byte
//! vertices of albedo RGB8 + smoothness A8, octahedral normal RG8 + metallic B8 +
//! reflectance A8), located through a per-geometry *primitive offset* and DXR's
//! `PrimitiveIndex`. A mesh is therefore keyed by geometry **and** material: `.bwm` surfaces
//! drawn with several materials (paint variants) get one BLAS per material.
//!
//! Inclusion rule (as the removed R3 renderer, checkpoint 9282529 `assets.rs`/`r3.rs`): a
//! surface is traced when its material is drawn by arkDefault/arkVegetation (classified by
//! `void_assets::material::Preparer`), is not an editor/collision material and is not a mesh
//! decal (coplanar overlay; traced only with [`LoadOptions::decals`]). Everything else is
//! counted per reason in [`SceneStats`].
//!
//! Two shading-data layouts:
//! - spike (default): the 24-byte primitive buffer above; every geometry `OPAQUE`
//!   (alpha testing ignored);
//! - photo ([`LoadOptions::photo_texture_size`]): per triangle a 64-byte [`Scene::records`]
//!   entry (texture LOD bias + the three raw 20-byte vertex attribute blocks: uv0, uv1,
//!   snorm8 normal, snorm8 tangent, RGBA8 vertex colour), a per-geometry material index,
//!   a GPU material table in the layout of the removed R3 renderer's `Mat` (checkpoint
//!   9282529 `gpu.rs`: `pack_params` + texture slots + `pack_lighting`) and the decoded
//!   texture mip chains for bindless upload. Alpha-tested and decal geometries are
//!   `NON_OPAQUE` (the kernel's candidate loop evaluates their alpha).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Instant;

use void_assets::Archive;
use void_assets::material::{PARAM_WORDS, Preparer, Support, pack_lighting, pack_params};
use void_assets::texture::TextureData;
use void_formats::arkmat::{ArkDefault, Cull};
use void_formats::bim::Format;
use void_formats::bmodel::StaticModel;
use void_formats::bwm::{WorldModel, f16_to_f32};
use void_formats::entities::{
    AxisConvention, EntitiesFile, ModelResource, Node, entity_def_resource_name, model_resource,
    parse_decl, placed_models,
};

use crate::snapshot::{Entity, material_is_drawn};

/// Largest texture edge loaded for per-material averages.
pub const MATERIAL_TEXTURE_SIZE: u32 = 32;
/// Dielectric F0 stored as the reflectance of every primitive (no per-material source
/// decoded yet).
pub const DIELECTRIC_F0: f32 = 0.04;
/// 32-bit words per photo-mode GPU material (see [`gpu_material`]).
pub const MATERIAL_WORDS: usize = 112;
/// Photo record words per triangle: LOD bias + 3 x 5-word attribute blocks.
pub const RECORD_WORDS: usize = 16;
/// LOD bias of triangles without texture coordinates: selects the 1x1 mip (the map's mean).
pub const LOD_BIAS_NO_UV: f32 = 64.0;
/// Mesh decals are coplanar with the surface under them (the engine pulls them forward
/// with `$zbias` in clip space); for ray tracing their vertices move this far (metres)
/// along the vertex normal in object space (H: value chosen, not the engine's).
pub const DECAL_OFFSET: f32 = 0.004;

/// Material flag bits in word 108 of a photo GPU material (beyond `pack_params` flags).
pub const MAT_DECAL: u32 = 1;
pub const MAT_TWO_SIDED_LIGHTING: u32 = 2;

/// What [`load`] builds besides the geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// Photo-mode shading data with material textures no larger than this edge (texels).
    pub photo_texture_size: Option<u32>,
    /// Trace arkDecal `decalType` 0 mesh decals (photo mode only).
    pub decals: bool,
    /// Photo mode only: also build the spike's 24-byte [`Scene::primitives`] next to the
    /// photo records (same triangles and order), so one BLAS set serves the photo path
    /// tracer and real-time kernels that read the FC6 primitive buffer.
    pub spike_primitives: bool,
}

/// One BLAS geometry: a surface-local u16 triangle list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub first_vertex: u32,
    pub vertex_count: u32,
    /// First element in [`Scene::indices`].
    pub first_index: u32,
    pub index_count: u32,
    /// First triangle in [`Scene::primitives`] / [`Scene::records`].
    pub first_primitive: u32,
    /// `false`: alpha-tested or decal material in photo mode (`NON_OPAQUE` geometry flag).
    pub opaque: bool,
}

/// One BLAS; `first_geometry` indexes the global geometry table (TLAS `InstanceID`).
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub name: String,
    pub geometries: Vec<Geometry>,
    pub first_geometry: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

impl Mesh {
    pub fn triangles(&self) -> u64 {
        self.geometries
            .iter()
            .map(|g| u64::from(g.index_count / 3))
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    World,
    Entity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub mesh: u32,
    /// Row-major 3x4 local-to-world.
    pub transform: [f32; 12],
    pub origin: Origin,
    /// Any of its materials renders both faces (`cull none`).
    pub two_sided: bool,
}

impl Instance {
    /// World-space AABB of the transformed mesh bounds.
    pub fn world_aabb(&self, mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let t = &self.transform;
        let mut lo = [t[3], t[7], t[11]];
        let mut hi = lo;
        for r in 0..3 {
            for c in 0..3 {
                let a = t[r * 4 + c] * mesh.aabb_min[c];
                let b = t[r * 4 + c] * mesh.aabb_max[c];
                lo[r] += a.min(b);
                hi[r] += a.max(b);
            }
        }
        (lo, hi)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialInfo {
    pub name: String,
    /// Linear RGB average of layer 0's diffuse map at its smallest stored mips
    /// (alpha-tested texels below 0.5 skipped); grey 0.5 when not decodable.
    pub albedo: [f32; 3],
    pub albedo_from_texture: bool,
    /// `lerp(wardRoughness.x, .y, mean layer-0 gloss map)`, `.y` without a decodable map
    /// (docs/formats/arkdefault.md `HQ_FetchGloss`; roughness = 1 - gloss).
    pub smoothness: f32,
    /// `lerp(metallicMinMax.x, .y, mean layer-0 metallic map)`, `.y` without a map.
    pub metallic: f32,
    pub two_sided: bool,
    pub alpha_test: bool,
    /// arkDecal mesh decal (photo mode with decals only).
    pub decal: bool,
    /// UV set of layer 0 (texture LOD bias source).
    pub uv_set: usize,
}

/// Per-reason exclusion counters (instances or entity surfaces, and their triangles).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Excluded {
    pub count: u64,
    pub triangles: u64,
}

#[derive(Clone, Debug, Default)]
pub struct SceneStats {
    pub bwm: String,
    pub world_surfaces: usize,
    pub world_batches: usize,
    pub world_instances: usize,
    pub world_instances_kept: usize,
    /// `.bwm` surfaces traced with more than one material (one BLAS per material).
    pub world_surfaces_multi_material: usize,
    pub entities_requested: usize,
    pub entities_kept: usize,
    pub entities_skinned: usize,
    pub entities_without_model: usize,
    pub entity_runtime_material_mismatch: usize,
    pub excluded: BTreeMap<String, Excluded>,
    pub materials_traced: usize,
    pub materials_with_texture_albedo: usize,
    pub alpha_tested_triangles_instanced: u64,
    /// Photo mode: decal instances traced and their instanced triangles.
    pub decal_instances: usize,
    pub decal_triangles_instanced: u64,
    /// Photo mode: triangles without vertex attributes (no UVs: mean texture colour).
    pub triangles_without_uv: u64,
    /// Photo mode: materials using more than layer 0 (layer blending evaluated).
    pub materials_layered: usize,
    /// Photo mode: `Partial` support notes per feature (count of materials).
    pub ignored_features: BTreeMap<String, usize>,
    pub load_seconds: f64,
}

pub struct Scene {
    /// BLAS build input (object space).
    pub positions: Vec<[f32; 3]>,
    /// BLAS build input (surface-local u16).
    pub indices: Vec<u16>,
    /// Spike: unwrapped per-triangle shading data, 6 words: per vertex
    /// `albedo.rgb8 | smooth8`, `oct_normal.rg8 | metallic8 | reflectance8` (object-space
    /// normal). Empty in photo mode unless [`LoadOptions::spike_primitives`].
    pub primitives: Vec<[u32; 6]>,
    /// Photo: per triangle `[lod_bias f32, block0[5], block1[5], block2[5]]` (raw 20-byte
    /// vertex attribute blocks, see the module doc). Empty in spike mode.
    pub records: Vec<[u32; RECORD_WORDS]>,
    pub meshes: Vec<Mesh>,
    /// Global geometry table: first primitive of every geometry of every mesh.
    pub primitive_offsets: Vec<u32>,
    /// Material index of every geometry (parallel to `primitive_offsets`).
    pub geometry_materials: Vec<u32>,
    pub instances: Vec<Instance>,
    pub materials: Vec<MaterialInfo>,
    /// Photo: GPU material table (parallel to `materials`).
    pub gpu_materials: Vec<[u32; MATERIAL_WORDS]>,
    /// Photo: decoded texture mip chains; GPU slot = index + 1 (slot 0 = white).
    pub textures: Vec<TextureData>,
    pub stats: SceneStats,
}

impl Scene {
    pub fn unique_triangles(&self) -> u64 {
        self.meshes.iter().map(Mesh::triangles).sum()
    }

    pub fn instanced_triangles(&self) -> u64 {
        self.instances
            .iter()
            .map(|i| self.meshes[i.mesh as usize].triangles())
            .sum()
    }

    /// `(first primitive, material)` per geometry (photo kernel table).
    pub fn geometry_info(&self) -> Vec<[u32; 2]> {
        self.primitive_offsets
            .iter()
            .zip(&self.geometry_materials)
            .map(|(&p, &m)| [p, m])
            .collect()
    }
}

/// Where static entity instances come from.
pub enum EntitySource {
    None,
    /// The map's `.entities` file (decoders: `void_formats::entities`, E021/E023).
    File,
    /// Live instances from an F3 snapshot (`models.tsv`).
    Snapshot(Vec<Entity>),
}

/// Level geometry path for an engine map name (`"%s%s/%s.bwm"`, O: IDB `sub_14023A5D0`).
pub fn bwm_path(map_name: &str) -> String {
    let name = map_name.replace('\\', "/");
    let name = name.trim_end_matches(".map").trim_end_matches(".entities");
    let (dir, base) = match name.rsplit_once('/') {
        Some((d, b)) => (format!("{d}/"), b),
        None => (String::new(), name),
    };
    format!("{dir}{base}/{base}.bwm")
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Decodes every texel of the largest loaded mip no larger than
/// [`MATERIAL_TEXTURE_SIZE`] (else the smallest) as RGBA8 (BC7, BC4/BC5 red, R8).
fn texels(t: &TextureData) -> Option<Vec<[u8; 4]>> {
    let mip = t
        .mips
        .iter()
        .find(|m| m.width.max(m.height) <= MATERIAL_TEXTURE_SIZE)
        .or(t.mips.last())?;
    let (w, h) = (mip.width as usize, mip.height as usize);
    let mut out = Vec::with_capacity(w * h);
    if t.format == Format::R8Unorm {
        let d = mip.data.get(..w * h)?;
        out.extend(d.iter().map(|&r| [r, r, r, 255]));
        return Some(out);
    }
    let block_bytes = match t.format {
        Format::Bc7Unorm | Format::Bc7UnormSrgb | Format::Bc5Unorm => 16,
        Format::Bc4Unorm => 8,
        _ => return None,
    };
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    if mip.data.len() < bw * bh * block_bytes {
        return None;
    }
    let mut rgba = [0u8; 64];
    let mut red = [0u8; 16];
    for by in 0..bh {
        for bx in 0..bw {
            let b = &mip.data[(by * bw + bx) * block_bytes..][..block_bytes];
            let is_bc7 = matches!(t.format, Format::Bc7Unorm | Format::Bc7UnormSrgb);
            if is_bc7 {
                bcdec_rs::bc7(b, &mut rgba, 16);
            } else {
                // BC5: the first 8 bytes are the red channel's BC4 block.
                bcdec_rs::bc4(&b[..8], &mut red, 4, false);
            }
            for y in 0..4 {
                for x in 0..4 {
                    if bx * 4 + x >= w || by * 4 + y >= h {
                        continue;
                    }
                    let k = y * 4 + x;
                    out.push(if is_bc7 {
                        [
                            rgba[k * 4],
                            rgba[k * 4 + 1],
                            rgba[k * 4 + 2],
                            rgba[k * 4 + 3],
                        ]
                    } else {
                        [red[k], red[k], red[k], 255]
                    });
                }
            }
        }
    }
    Some(out)
}

/// Mean linear RGB of a colour map (texels with alpha < 0.5 skipped when `alpha_test`).
pub fn average_albedo(t: &TextureData, alpha_test: bool) -> Option<[f32; 3]> {
    let srgb = match t.format {
        Format::Bc7UnormSrgb => true,
        Format::Bc7Unorm => false,
        _ => return None,
    };
    let mut sum = [0f64; 3];
    let mut n = 0u64;
    for p in texels(t)? {
        if alpha_test && p[3] < 128 {
            continue;
        }
        for c in 0..3 {
            let v = f32::from(p[c]) / 255.0;
            sum[c] += f64::from(if srgb { srgb_to_linear(v) } else { v });
        }
        n += 1;
    }
    (n > 0).then(|| sum.map(|s| (s / n as f64) as f32))
}

/// Mean of the red channel of a raw map in [0, 1].
pub fn average_red(t: &TextureData) -> Option<f32> {
    let px = texels(t)?;
    (!px.is_empty())
        .then(|| px.iter().map(|p| f32::from(p[0])).sum::<f32>() / (255.0 * px.len() as f32))
}

/// Octahedral encoding of a unit vector to two unorm8.
pub fn oct_encode(n: [f32; 3]) -> [u8; 2] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    if l1 <= 0.0 {
        return [128, 128];
    }
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    let q = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
    [q(x), q(y)]
}

fn unorm8(v: f32) -> u32 {
    u32::from((v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Packs one primitive vertex (see [`Scene::primitives`]).
pub fn pack_vertex(m: &MaterialInfo, normal: [f32; 3]) -> [u32; 2] {
    let [ox, oy] = oct_encode(normal);
    let c = m.albedo.map(unorm8);
    [
        c[0] | c[1] << 8 | c[2] << 16 | unorm8(m.smoothness) << 24,
        u32::from(ox) | u32::from(oy) << 8 | unorm8(m.metallic) << 16 | unorm8(DIELECTRIC_F0) << 24,
    ]
}

fn snorm8_normal(b: &[u8; 20]) -> [f32; 3] {
    let s = |k: usize| (f32::from(b[8 + k] as i8) / 127.0).max(-1.0);
    let n = [s(0), s(1), s(2)];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 1e-3 { n.map(|v| v / l) } else { [0.0; 3] }
}

enum Class {
    Traced(u32),
    Excluded(String),
}

/// Groups a `Support` failure into a short reason for the exclusion table.
fn unsupported_reason(s: &str) -> String {
    if let Some(p) = s.strip_prefix("program ") {
        format!("program {}", p.to_ascii_lowercase())
    } else if s.starts_with("decl") {
        "material decl unavailable".into()
    } else {
        s.to_owned()
    }
}

/// One decoded source surface.
struct Surface<'s> {
    positions: &'s [[f32; 3]],
    attributes: Option<&'s [[u8; 20]]>,
    indices: &'s [u16],
}

struct Builder<'a> {
    preparer: Preparer<'a>,
    scene: Scene,
    classes: HashMap<String, Class>,
    opts: LoadOptions,
    /// Photo: texture name -> GPU slot (index + 1).
    texture_slots: HashMap<String, u32>,
    triangles: u32,
    /// Characters: never exclude a surface for its material (fallback primitive data).
    fallback: bool,
    /// Materials traced through the fallback: (name, reason).
    fallbacks: Vec<(String, String)>,
}

/// Photo GPU material, the `Mat` layout of the removed R3 renderer (checkpoint 9282529
/// `gpu.rs`): words 0..84 [`pack_params`], 84..100 texture slots (0-2 diffuse, 3-5 bump,
/// 6-8 gloss, 9-11 metallic, 12-14 mask, 15 occlusion; 0 = white), 100..108
/// [`pack_lighting`], 108 photo flags (`MAT_*`), 109..112 mean layer-0 albedo (linear RGB).
pub fn gpu_material(
    a: &ArkDefault,
    slots: [u32; 16],
    alpha_test: bool,
    flags: u32,
    albedo: [f32; 3],
) -> [u32; MATERIAL_WORDS] {
    let mut w = [0u32; MATERIAL_WORDS];
    // Map flags only for loaded layer-0 maps (as the R3 renderer).
    let params = pack_params(a, slots[6] != 0, slots[9] != 0, slots[3] != 0, alpha_test);
    w[..PARAM_WORDS].copy_from_slice(&params);
    w[84..100].copy_from_slice(&slots);
    w[100..108].copy_from_slice(&pack_lighting(a));
    w[108] = flags;
    w[109..112].copy_from_slice(&albedo.map(f32::to_bits));
    w
}

/// Texture-space LOD bias of a triangle: `0.5 log2(uv area / object-space area)` (ray-cone
/// texture LOD, Akenine-Moller et al. 2021, the triangle's `Delta`).
pub fn lod_bias(p: [[f32; 3]; 3], uv: [[f32; 2]; 3]) -> f32 {
    let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let c = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let area = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let ta = ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
        - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]))
        .abs();
    if area.is_nan() || area <= 1e-12 || !ta.is_finite() {
        return 0.0;
    }
    if ta <= 0.0 {
        // Constant UVs: a single texel (finest mip).
        return -LOD_BIAS_NO_UV;
    }
    (0.5 * (ta / area).log2()).clamp(-LOD_BIAS_NO_UV, LOD_BIAS_NO_UV)
}

fn f16_pair(b: &[u8; 20], o: usize) -> [f32; 2] {
    let h = |k: usize| f16_to_f32(u16::from_le_bytes([b[o + k], b[o + k + 1]]));
    [h(0), h(2)]
}

fn snorm8(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8
}

impl Builder<'_> {
    fn photo(&self) -> bool {
        self.opts.photo_texture_size.is_some()
    }

    /// GPU slot of a texture (uploaded once per name).
    fn slot(&mut self, t: Option<TextureData>) -> u32 {
        let Some(t) = t else { return 0 };
        if let Some(&s) = self.texture_slots.get(&t.name) {
            return s;
        }
        self.scene.textures.push(t);
        let s = self.scene.textures.len() as u32;
        self.texture_slots
            .insert(self.scene.textures[s as usize - 1].name.clone(), s);
        s
    }

    fn class(&mut self, name: &str) -> &Class {
        if !self.classes.contains_key(name) {
            let c = if name.is_empty() && !self.fallback {
                Class::Excluded("no material".into())
            } else if !material_is_drawn(name) && !self.fallback {
                Class::Excluded("editor/collision material".into())
            } else {
                let mut p = if self.fallback {
                    // Characters: every surface is traced (`Preparer::prepare_fallback`).
                    let p = self.preparer.prepare_fallback(name);
                    if let Support::Partial(n) = &p.support
                        && let Some(f) = n.iter().find(|x| x.starts_with("fallback: "))
                    {
                        self.fallbacks
                            .push((name.to_owned(), f["fallback: ".len()..].to_owned()));
                    }
                    p
                } else {
                    self.preparer.prepare(name)
                };
                let decal_traced = p.decal && (self.opts.decals || self.fallback) && self.photo();
                match (&p.support, p.model.take()) {
                    (Support::EditorOnly(_), _) => Class::Excluded("editor-only program".into()),
                    (Support::Unsupported(r), _) => Class::Excluded(unsupported_reason(r)),
                    _ if p.decal && !decal_traced => Class::Excluded("mesh decal (overlay)".into()),
                    (_, Some(model)) => {
                        let tex = p.textures.diffuse[0]
                            .as_ref()
                            .and_then(|t| average_albedo(t, p.alpha_test));
                        let gloss = p.textures.gloss[0].as_ref().and_then(average_red);
                        let metal = p.textures.metallic[0].as_ref().and_then(average_red);
                        let lerp = |r: [f32; 2], v: Option<f32>| {
                            v.map_or(r[1], |v| r[0] + (r[1] - r[0]) * v)
                        };
                        let w = model.ward_roughness;
                        let info = MaterialInfo {
                            name: name.to_owned(),
                            albedo: tex.unwrap_or([0.5; 3]),
                            albedo_from_texture: tex.is_some(),
                            smoothness: lerp([w[0], w[1]], gloss),
                            metallic: lerp(model.metallic_min_max, metal),
                            two_sided: model.cull == Cull::None,
                            alpha_test: p.alpha_test,
                            decal: decal_traced,
                            uv_set: model.layers[0].uv_set.min(1),
                        };
                        if self.photo() {
                            if let Support::Partial(notes) = &p.support {
                                for n in notes {
                                    // Group per feature (drop per-texture detail).
                                    let key = n.split(':').next().unwrap_or(n).to_owned();
                                    *self.scene.stats.ignored_features.entry(key).or_default() += 1;
                                }
                            }
                            self.scene.stats.materials_layered +=
                                usize::from(model.extra_layer > 0);
                            let t = std::mem::take(&mut p.textures);
                            let mut slots = [0u32; 16];
                            for (k, x) in t
                                .diffuse
                                .into_iter()
                                .chain(t.normal)
                                .chain(t.gloss)
                                .chain(t.metallic)
                                .chain(t.mask)
                                .chain([t.occlusion])
                                .enumerate()
                            {
                                slots[k] = self.slot(x);
                            }
                            let flags = if decal_traced { MAT_DECAL } else { 0 }
                                | if p.two_sided {
                                    MAT_TWO_SIDED_LIGHTING
                                } else {
                                    0
                                };
                            self.scene.gpu_materials.push(gpu_material(
                                &model,
                                slots,
                                p.alpha_test,
                                flags,
                                info.albedo,
                            ));
                        }
                        self.scene.stats.materials_traced += 1;
                        self.scene.stats.materials_with_texture_albedo +=
                            usize::from(tex.is_some());
                        self.scene.materials.push(info);
                        Class::Traced((self.scene.materials.len() - 1) as u32)
                    }
                    (_, None) => Class::Excluded("no arkDefault model".into()),
                }
            };
            self.classes.insert(name.to_owned(), c);
        }
        &self.classes[name]
    }

    fn exclude(&mut self, reason: String, count: u64, triangles: u64) {
        let e = self.scene.stats.excluded.entry(reason).or_default();
        e.count += count;
        e.triangles += triangles * count;
    }

    /// Appends one geometry with its primitives; `None` if the data is inconsistent.
    fn geometry(
        &mut self,
        s: &Surface,
        material: u32,
        bounds: &mut ([f32; 3], [f32; 3]),
    ) -> Option<Geometry> {
        let vertex_count = s.positions.len();
        let index_count = s.indices.len() / 3 * 3;
        let indices = &s.indices[..index_count];
        if index_count == 0 || indices.iter().any(|&i| usize::from(i) >= vertex_count) {
            return None;
        }
        let attrs = s.attributes.filter(|a| a.len() == vertex_count);
        let photo = self.photo();
        let sc = &mut self.scene;
        let m = &sc.materials[material as usize];
        let g = Geometry {
            first_vertex: sc.positions.len() as u32,
            vertex_count: vertex_count as u32,
            first_index: sc.indices.len() as u32,
            index_count: index_count as u32,
            first_primitive: self.triangles,
            opaque: !(photo && (m.alpha_test || m.decal)),
        };
        self.triangles += (index_count / 3) as u32;
        let start = sc.positions.len();
        sc.positions.extend_from_slice(s.positions);
        if m.decal
            && let Some(a) = attrs
        {
            // Lift decals off the surface they overlay (see DECAL_OFFSET).
            for (p, b) in sc.positions[start..].iter_mut().zip(a) {
                let n = snorm8_normal(b);
                for k in 0..3 {
                    p[k] += n[k] * DECAL_OFFSET;
                }
            }
        }
        sc.indices.extend_from_slice(indices);
        for p in s.positions {
            for k in 0..3 {
                bounds.0[k] = bounds.0[k].min(p[k]);
                bounds.1[k] = bounds.1[k].max(p[k]);
            }
        }
        for t in indices.as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|k| s.positions[usize::from(t[k])]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let f = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let fl = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            let face = if fl > 0.0 {
                f.map(|v| v / fl)
            } else {
                [0.0, 0.0, 1.0]
            };
            if photo {
                let mut rec = [0u32; RECORD_WORDS];
                let blocks: [[u8; 20]; 3] = match attrs {
                    Some(a) => [0, 1, 2].map(|k| a[usize::from(t[k])]),
                    None => {
                        // No attribute stream: face normal, no UV/tangent, white colour.
                        let mut b = [0u8; 20];
                        for k in 0..3 {
                            b[8 + k] = snorm8(face[k]);
                        }
                        b[16..20].fill(255);
                        [b; 3]
                    }
                };
                let bias = if attrs.is_some() {
                    let o = 4 * m.uv_set;
                    lod_bias([a, b, c], blocks.map(|b| f16_pair(&b, o)))
                } else {
                    sc.stats.triangles_without_uv += 1;
                    LOD_BIAS_NO_UV
                };
                rec[0] = bias.to_bits();
                for (k, blk) in blocks.iter().enumerate() {
                    for (j, w) in blk.as_chunks::<4>().0.iter().enumerate() {
                        rec[1 + k * 5 + j] = u32::from_le_bytes(*w);
                    }
                }
                sc.records.push(rec);
                if !self.opts.spike_primitives {
                    continue;
                }
            }
            let mut prim = [0u32; 6];
            for k in 0..3 {
                let n = attrs
                    .map(|a| snorm8_normal(&a[usize::from(t[k])]))
                    .filter(|n| n.iter().any(|&v| v != 0.0))
                    .unwrap_or(face);
                let [w0, w1] = pack_vertex(m, n);
                prim[k * 2] = w0;
                prim[k * 2 + 1] = w1;
            }
            sc.primitives.push(prim);
        }
        Some(g)
    }

    /// Builds a mesh (one BLAS) from surfaces with their materials.
    fn mesh(&mut self, name: String, parts: &[(Surface, u32)]) -> Option<u32> {
        let mut bounds = ([f32::MAX; 3], [f32::MIN; 3]);
        let mut geometries = Vec::new();
        for (s, m) in parts {
            geometries.push(self.geometry(s, *m, &mut bounds)?);
        }
        let first_geometry = self.scene.primitive_offsets.len() as u32;
        self.scene
            .primitive_offsets
            .extend(geometries.iter().map(|g| g.first_primitive));
        self.scene
            .geometry_materials
            .extend(parts.iter().map(|p| p.1));
        self.scene.meshes.push(Mesh {
            name,
            geometries,
            first_geometry,
            aabb_min: bounds.0,
            aabb_max: bounds.1,
        });
        Some((self.scene.meshes.len() - 1) as u32)
    }

    /// (two-sided, alpha-tested, all decals) over an instance's materials.
    fn flags(&self, mats: &[u32]) -> (bool, bool, bool) {
        let m = |i: &u32| &self.scene.materials[*i as usize];
        (
            mats.iter().any(|i| m(i).two_sided),
            mats.iter().any(|i| m(i).alpha_test),
            mats.iter().all(|i| m(i).decal),
        )
    }
}

fn file_entities(archive: &Archive, map: &str) -> Result<Vec<Entity>, String> {
    let name = format!("{}.entities", map.trim_end_matches(".map"));
    let text = archive.read_text("File", &name)?;
    let file = EntitiesFile::parse(&text).map_err(|e| format!("{name}: {e}"))?;
    let mut defs: HashMap<String, Option<Node>> = HashMap::new();
    let placed = placed_models(&file, |inh| {
        defs.entry(inh.to_owned())
            .or_insert_with(|| {
                archive
                    .read_text("entityDef", &entity_def_resource_name(inh))
                    .ok()
                    .and_then(|t| parse_decl(&t).ok())
            })
            .clone()
    });
    Ok(placed
        .into_iter()
        .map(|p| Entity {
            transform: p.transform.matrix(AxisConvention::RowsAreAxes),
            model_source: p.model_source,
            materials: p.custom_materials,
            runtime_materials: false,
        })
        .collect())
}

/// Entities without a traced instance in [`load_mapped`].
pub const NO_INSTANCE: u32 = u32::MAX;

/// One decoded surface of a model: object-space positions, the 20-byte vertex attribute
/// blocks (uv0, uv1, snorm8 normal, snorm8 tangent, colour; as `.bmodel` / `.bwm`),
/// surface-local indices and the material named by the file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartSurface {
    pub positions: Vec<[f32; 3]>,
    pub attributes: Option<Vec<[u8; 20]>>,
    pub indices: Vec<u16>,
    pub material: String,
}

impl PartSurface {
    fn surface(&self) -> Surface<'_> {
        Surface {
            positions: &self.positions,
            attributes: self.attributes.as_deref(),
            indices: &self.indices,
        }
    }
}

/// The surfaces of a static `.bmodel` resource (`None` for an undecodable surface).
pub fn static_model_surfaces(
    archive: &Archive,
    res: &str,
) -> Result<Vec<Option<PartSurface>>, String> {
    let m = StaticModel::parse(&archive.read("model", res)?).map_err(|e| format!("{res}: {e}"))?;
    Ok(m.surfaces
        .iter()
        .map(|s| {
            Some(PartSurface {
                positions: s.positions()?,
                attributes: s.attribute_blocks(),
                indices: s.indices.clone(),
                material: s.material.clone(),
            })
        })
        .collect())
}

/// Material of surface `k` of entity `e`: its live or `.entities` override, else the
/// file's. Live (runtime) materials apply only when their count matches the model's
/// surfaces (`runtime_ok`).
pub fn entity_material(e: &Entity, runtime_ok: bool, k: usize, file: &str) -> String {
    let over = e.materials.get(k).filter(|m| !m.is_empty());
    match (e.runtime_materials, over) {
        (true, Some(m)) if runtime_ok => m.clone(),
        (false, Some(m)) => m.clone(),
        _ => file.to_owned(),
    }
}

fn empty_scene(stats: SceneStats) -> Scene {
    Scene {
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
        stats,
    }
}

/// What [`mesh_scene`] built.
pub struct MeshScene {
    pub scene: Scene,
    /// Indices of the kept parts (their vertices follow each other in `positions`).
    pub kept: Vec<usize>,
    /// With `fallback`: materials traced through `Preparer::prepare_fallback` (name,
    /// reason, e.g. the unsupported program).
    pub fallbacks: Vec<(String, String)>,
}

/// A one-mesh [`Scene`] of `parts` with their materials (`materials[k]`): positions,
/// indices, photo records and FC6 primitives from triangle 0, its own material table and
/// textures (GPU slot = index + 1), no instances. This is the CPU half of a dynamic mesh
/// (`crate::dynamic`). Without `fallback` the rules of static entities apply (excluded
/// materials drop their surface); with it (characters) no surface is dropped for its
/// material: unsupported programs are read as arkDefault layer 0 (diffuse average or
/// grey, alpha test via the candidate loop). Err when nothing is traced.
pub fn mesh_scene(
    archive: &Archive,
    name: &str,
    parts: &[PartSurface],
    materials: &[String],
    opts: LoadOptions,
    fallback: bool,
) -> Result<MeshScene, String> {
    let mut b = Builder {
        preparer: Preparer::new(
            archive,
            opts.photo_texture_size.unwrap_or(MATERIAL_TEXTURE_SIZE),
        ),
        scene: empty_scene(SceneStats::default()),
        classes: HashMap::new(),
        opts,
        texture_slots: HashMap::new(),
        triangles: 0,
        fallback,
        fallbacks: Vec::new(),
    };
    let mut kept: Vec<(usize, u32)> = Vec::new();
    for (k, p) in parts.iter().enumerate() {
        let mat = materials.get(k).unwrap_or(&p.material);
        match b.class(mat) {
            Class::Traced(m) => kept.push((k, *m)),
            Class::Excluded(r) => {
                let r = format!("dynamic: {r}");
                b.exclude(r, 1, (p.indices.len() / 3) as u64);
            }
        }
    }
    if kept.is_empty() {
        return Err(format!("{name}: no traced surface"));
    }
    let list: Vec<(Surface, u32)> = kept.iter().map(|&(k, m)| (parts[k].surface(), m)).collect();
    b.mesh(name.to_owned(), &list)
        .ok_or_else(|| format!("{name}: invalid surface data"))?;
    Ok(MeshScene {
        scene: b.scene,
        kept: kept.iter().map(|k| k.0).collect(),
        fallbacks: b.fallbacks,
    })
}

/// Loads the static world of `map` plus entities into a [`Scene`].
pub fn load(
    game_dir: &Path,
    map: &str,
    entities: EntitySource,
    opts: LoadOptions,
) -> Result<Scene, String> {
    load_mapped(game_dir, map, entities, opts).map(|(s, _)| s)
}

/// As [`load`], also returning the instance of every given entity ([`NO_INSTANCE`] for
/// entities that were not traced): hosts map engine models to TLAS instances with it.
pub fn load_mapped(
    game_dir: &Path,
    map: &str,
    entities: EntitySource,
    opts: LoadOptions,
) -> Result<(Scene, Vec<u32>), String> {
    let t0 = Instant::now();
    let mut archive = Archive::open(game_dir)?;
    let bwm = bwm_path(map);
    let &[(group, _)] = archive.locate("File", &bwm) else {
        return Err(format!(
            "{bwm}: expected exactly one File entry across groups"
        ));
    };
    archive.preferred_group = group;
    let world =
        WorldModel::parse(&archive.read("File", &bwm)?).map_err(|e| format!("{bwm}: {e}"))?;
    let entities = match entities {
        EntitySource::None => Vec::new(),
        EntitySource::File => file_entities(&archive, map)?,
        EntitySource::Snapshot(v) => v,
    };
    let mut b = Builder {
        preparer: Preparer::new(
            &archive,
            opts.photo_texture_size.unwrap_or(MATERIAL_TEXTURE_SIZE),
        ),
        scene: empty_scene(SceneStats {
            bwm: bwm.clone(),
            world_surfaces: world.surfaces.len(),
            world_batches: world.batches.len(),
            world_instances: world.instances.len(),
            entities_requested: entities.len(),
            ..Default::default()
        }),
        classes: HashMap::new(),
        opts,
        texture_slots: HashMap::new(),
        triangles: 0,
        fallback: false,
        fallbacks: Vec::new(),
    };

    // World: one mesh per (surface, material).
    let mut world_meshes: HashMap<(usize, u32), Option<u32>> = HashMap::new();
    let mut world_cache: HashMap<usize, (Vec<[f32; 3]>, Vec<u16>)> = HashMap::new();
    for batch in &world.batches {
        let s = batch.surface as usize;
        let surface = world
            .surfaces
            .get(s)
            .ok_or_else(|| format!("{bwm}: batch surface {s} out of range"))?;
        let tris = u64::from(surface.index_count / 3);
        let count = u64::from(batch.end_instance.saturating_sub(batch.first_instance));
        let mat = match b.class(&batch.material) {
            Class::Traced(m) => *m,
            Class::Excluded(r) => {
                let r = format!("world: {r}");
                b.exclude(r, count, tris);
                continue;
            }
        };
        let mesh = match world_meshes.get(&(s, mat)) {
            Some(&m) => m,
            None => {
                let (pos, idx) = world_cache.entry(s).or_insert_with(|| {
                    (
                        world.surface_positions(s).collect(),
                        world.surface_indices(s).collect(),
                    )
                });
                let surf = Surface {
                    positions: pos.as_slice(),
                    attributes: world.surface_attribute_blocks(s),
                    indices: idx.as_slice(),
                };
                let m = b.mesh(format!("bwm surface {s}"), &[(surf, mat)]);
                world_meshes.insert((s, mat), m);
                m
            }
        };
        let Some(mesh) = mesh else {
            b.exclude("world: invalid surface data".into(), count, tris);
            continue;
        };
        let (two_sided, alpha, decal) = b.flags(&[mat]);
        for i in batch.first_instance..batch.end_instance {
            let inst = world
                .instances
                .get(i as usize)
                .ok_or_else(|| format!("{bwm}: batch instance {i} out of range"))?;
            b.scene.instances.push(Instance {
                mesh,
                transform: inst.transform,
                origin: Origin::World,
                two_sided,
            });
            if alpha {
                b.scene.stats.alpha_tested_triangles_instanced += tris;
            }
            if decal {
                b.scene.stats.decal_instances += 1;
                b.scene.stats.decal_triangles_instanced += tris;
            }
            b.scene.stats.world_instances_kept += 1;
        }
    }
    let mut per_surface: HashMap<usize, usize> = HashMap::new();
    for &(s, _) in world_meshes.keys() {
        *per_surface.entry(s).or_default() += 1;
    }
    b.scene.stats.world_surfaces_multi_material = per_surface.values().filter(|&&n| n > 1).count();
    drop(world_cache);

    // Entities: one mesh per (model, kept surfaces, materials).
    let mut models: HashMap<String, Option<Vec<Option<PartSurface>>>> = HashMap::new();
    // Key: (model resource, kept surfaces with their materials) -> mesh (None = invalid).
    type MeshKey = (String, Vec<(usize, u32)>);
    let mut entity_meshes: HashMap<MeshKey, Option<u32>> = HashMap::new();
    let mut entity_instances = vec![NO_INSTANCE; entities.len()];
    for (ei, e) in entities.iter().enumerate() {
        let res = match model_resource(&e.model_source) {
            ModelResource::Static(r) => r,
            ModelResource::Skinned(_) => {
                b.scene.stats.entities_skinned += 1;
                continue;
            }
        };
        let surfaces = models
            .entry(res.clone())
            .or_insert_with(|| static_model_surfaces(&archive, &res).ok());
        let Some(surfaces) = surfaces.as_ref() else {
            b.scene.stats.entities_without_model += 1;
            continue;
        };
        let runtime_ok = e.runtime_materials && e.materials.len() == surfaces.len();
        if e.runtime_materials && !e.materials.is_empty() && !runtime_ok {
            b.scene.stats.entity_runtime_material_mismatch += 1;
        }
        let mut kept: Vec<(usize, u32)> = Vec::new();
        for (k, s) in surfaces.iter().enumerate() {
            let Some(s) = s else {
                b.exclude("entity: undecodable surface".into(), 1, 0);
                continue;
            };
            let idx = &s.indices;
            let name = entity_material(e, runtime_ok, k, &s.material);
            match b.class(&name) {
                Class::Traced(m) => kept.push((k, *m)),
                Class::Excluded(r) => {
                    let r = format!("entity: {r}");
                    b.exclude(r, 1, (idx.len() / 3) as u64);
                }
            }
        }
        if kept.is_empty() {
            continue;
        }
        let key = (res.clone(), kept.clone());
        let mesh = match entity_meshes.get(&key) {
            Some(&m) => m,
            None => {
                let parts: Vec<(Surface, u32)> = kept
                    .iter()
                    .filter_map(|&(k, m)| Some((surfaces[k].as_ref()?.surface(), m)))
                    .collect();
                let m = b.mesh(res.clone(), &parts);
                entity_meshes.insert(key, m);
                m
            }
        };
        let Some(mesh) = mesh else {
            b.exclude("entity: invalid surface data".into(), 1, 0);
            continue;
        };
        let mats: Vec<u32> = kept.iter().map(|k| k.1).collect();
        let (two_sided, alpha, decal) = b.flags(&mats);
        entity_instances[ei] = b.scene.instances.len() as u32;
        b.scene.instances.push(Instance {
            mesh,
            transform: e.transform,
            origin: Origin::Entity,
            two_sided,
        });
        let tris = b.scene.meshes[mesh as usize].triangles();
        if alpha {
            b.scene.stats.alpha_tested_triangles_instanced += tris;
        }
        if decal {
            b.scene.stats.decal_instances += 1;
            b.scene.stats.decal_triangles_instanced += tris;
        }
        b.scene.stats.entities_kept += 1;
    }
    let mut scene = b.scene;
    scene.stats.load_seconds = t0.elapsed().as_secs_f64();
    Ok((scene, entity_instances))
}

#[cfg(test)]
mod tests {
    use super::*;
    use void_assets::texture::Mip;

    #[test]
    fn bwm_path_follows_init_from_map() {
        assert_eq!(
            bwm_path("maps/campaign/edge_of_the_world/eotw_p_highchaos"),
            "maps/campaign/edge_of_the_world/eotw_p_highchaos/eotw_p_highchaos.bwm"
        );
        assert_eq!(bwm_path("maps/x/y.map"), "maps/x/y/y.bwm");
    }

    #[test]
    fn bc7_average_of_solid_block() {
        // BC7 mode 6 block with every endpoint bit set -> white opaque texels.
        let mut block = [0xffu8; 16];
        block[0] = 0x40;
        let t = TextureData {
            name: "t".into(),
            format: Format::Bc7Unorm,
            mips: vec![Mip {
                width: 4,
                height: 4,
                data: block.to_vec(),
            }],
        };
        let a = average_albedo(&t, false).unwrap();
        assert!(a.iter().all(|&c| c > 0.99), "{a:?}");
        let srgb = TextureData {
            format: Format::Bc7UnormSrgb,
            ..t
        };
        assert!(
            average_albedo(&srgb, true)
                .unwrap()
                .iter()
                .all(|&c| c > 0.99)
        );
    }

    #[test]
    fn r8_and_bc4_red_average() {
        let r8 = TextureData {
            name: "g".into(),
            format: Format::R8Unorm,
            mips: vec![Mip {
                width: 2,
                height: 1,
                data: vec![0, 255],
            }],
        };
        assert!((average_red(&r8).unwrap() - 0.5).abs() < 1e-6);
        // BC4 block with both endpoints 255 -> every texel 255.
        let bc4 = TextureData {
            format: Format::Bc4Unorm,
            mips: vec![Mip {
                width: 4,
                height: 4,
                data: vec![255, 255, 0, 0, 0, 0, 0, 0],
            }],
            ..r8
        };
        assert!((average_red(&bc4).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn oct_round_trip() {
        let dec = |e: [u8; 2]| {
            let (mut x, mut y) = (f32::from(e[0]) / 127.5 - 1.0, f32::from(e[1]) / 127.5 - 1.0);
            let z = 1.0 - x.abs() - y.abs();
            if z < 0.0 {
                let (ox, oy) = (x, y);
                x = (1.0 - oy.abs()) * ox.signum();
                y = (1.0 - ox.abs()) * oy.signum();
            }
            let l = (x * x + y * y + z * z).sqrt();
            [x / l, y / l, z / l]
        };
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.6, -0.8, 0.0],
            [0.267, 0.535, -0.802],
        ] {
            let d = dec(oct_encode(n));
            let dot = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
            assert!(dot > 0.995, "{n:?} -> {d:?}");
        }
    }

    #[test]
    fn world_aabb_transforms_bounds() {
        let mesh = Mesh {
            name: String::new(),
            geometries: Vec::new(),
            first_geometry: 0,
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 2.0, 3.0],
        };
        // Rotation +90 deg about Z, translation (10, 0, 0).
        let inst = Instance {
            mesh: 0,
            transform: [0.0, -1.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            origin: Origin::World,
            two_sided: false,
        };
        assert_eq!(inst.world_aabb(&mesh), ([8.0, 0.0, 0.0], [10.0, 1.0, 3.0]));
    }
}
