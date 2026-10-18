//! Dynamic objects over an uploaded [`GpuScene`] (docs/raytracing.md section 14): moved or
//! hidden static instances, added rigid meshes (spawned props) and skinned characters.
//!
//! A scene uploaded with [`crate::ingame::SceneOptions::dynamic`] has a reserve at the end
//! of its geometry table, photo records, materials, FC6 primitives + offsets and bindless
//! texture slots (zero / white until used; the kernels' counts cover the reserve, so the
//! consumers' bindings and clamps stay valid). [`Dynamics`] (owned by the host's GPU
//! thread) allocates mesh data in that reserve, builds their BLASes and produces the
//! instance lists of TLAS builds ([`Dynamics::cutoff`], [`Dynamics::full`]) and of refits
//! ([`Dynamics::refresh`]: same instances, new transforms / BLASes). Lifetimes: an
//! [`InstanceSet`] keeps the dynamic BLASes it references alive; mesh data in the reserve
//! is reused only after no set references the mesh any more ([`Dynamics::collect`]).
//!
//! Characters ([`crate::character`]): bind-pose vertices skinned on the GPU
//! (`dynamic.hlsl`) with the engine's joint palette, then their BLAS (PREFER_FAST_BUILD |
//! ALLOW_UPDATE) is updated into one of [`BLAS_COPIES`] buffers that no published TLAS
//! references (another queue may be tracing the current one); the skinned normals are
//! written into the character's photo records and FC6 primitives.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, ID3D12Resource,
};

use void_assets::Archive;
use void_formats::entities::{ModelResource, model_resource};

use crate::accel::{self, BlasKind, Tlas};
use crate::character::{CharacterCpu, Md6Index, Row34, load_character};
use crate::gpu::{Buffer, Entry, Gpu, Layout, Programs};
use crate::ingame::GpuScene;
use crate::scene::{
    self, Geometry, Instance, LoadOptions, MATERIAL_WORDS, Origin, RECORD_WORDS, Scene,
};
use crate::snapshot::Entity;

/// Root layout of `dynamic.hlsl`.
pub const LAYOUT: Layout = Layout {
    constants: 8,
    srvs: 5,
    uavs: 4,
    table: false,
};
const ENTRIES: [&str; 2] = ["skin_vertices", "skin_triangles"];
/// Threads per dispatch row of the skinning kernels.
const ROW: u32 = 256;
/// BLAS buffers per character: a pose update writes one that no TLAS still in use holds.
pub const BLAS_COPIES: usize = 3;
/// A character's bind-pose bounds grow by this much (metres) for the cut-off test.
const POSE_MARGIN: f32 = 1.0;

/// Something an [`InstanceSet`] / [`SceneTlas`] keeps alive (dynamic BLAS memory).
pub type Keep = Arc<dyn Any + Send + Sync>;

/// Room reserved at upload for dynamic data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicReserve {
    pub triangles: u32,
    pub geometries: u32,
    pub materials: u32,
    pub textures: u32,
    /// Extra TLAS instances (props + characters) beyond the static ones.
    pub instances: u32,
}

impl Default for DynamicReserve {
    /// 1 Mi triangles (88 MiB of records + FC6 primitives): about 20 characters at LOD 0.
    fn default() -> Self {
        Self {
            triangles: 1 << 20,
            geometries: 1 << 14,
            materials: 4096,
            textures: 1024,
            instances: 4096,
        }
    }
}

impl DynamicReserve {
    /// Video memory of the reserve in the scene's tables (records, FC6 primitives,
    /// geometry table + offsets, materials).
    pub fn table_bytes(&self, spike_primitives: bool) -> u64 {
        u64::from(self.triangles)
            * (RECORD_WORDS as u64 * 4 + if spike_primitives { 24 } else { 0 })
            + u64::from(self.geometries) * 12
            + u64::from(self.materials) * MATERIAL_WORDS as u64 * 4
    }
}

/// Static counts of the scene's tables: the reserve starts there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReserveBase {
    pub geometries: u32,
    pub triangles: u32,
    pub materials: u32,
}

/// What a dynamic-capable [`GpuScene`] records about its reserve.
#[derive(Clone, Debug, Default)]
pub struct DynamicTables {
    pub reserve: Option<DynamicReserve>,
    pub base: ReserveBase,
    /// Static texture name -> bindless slot (shared with dynamic meshes).
    pub texture_slots: HashMap<String, u32>,
    /// Decode options of dynamic meshes: the scene's (texture cap as uploaded).
    pub load: LoadOptions,
}

/// First-fit range allocator over `[start, start + len)`.
#[derive(Clone, Debug, Default)]
struct Ranges {
    /// Free (start, len), sorted by start, never adjacent.
    free: Vec<(u32, u32)>,
    total: u32,
}

impl Ranges {
    fn new(start: u32, len: u32) -> Self {
        Self {
            free: if len > 0 {
                vec![(start, len)]
            } else {
                Vec::new()
            },
            total: len,
        }
    }

    fn alloc(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        let i = self.free.iter().position(|r| r.1 >= n)?;
        let (s, l) = self.free[i];
        if l == n {
            self.free.remove(i);
        } else {
            self.free[i] = (s + n, l - n);
        }
        Some(s)
    }

    fn release(&mut self, start: u32, n: u32) {
        if n == 0 {
            return;
        }
        let i = self.free.partition_point(|r| r.0 < start);
        self.free.insert(i, (start, n));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    fn used(&self) -> u32 {
        self.total - self.free.iter().map(|r| r.1).sum::<u32>()
    }
}

/// The reserve entries of one mesh.
#[derive(Clone, Debug, Default)]
struct Region {
    geoms: (u32, u32),
    tris: (u32, u32),
    mats: (u32, u32),
    /// Dynamic textures it references (by name).
    textures: Vec<String>,
}

/// A rigid dynamic mesh (spawned prop) with its BLAS.
pub struct PropMesh {
    name: String,
    region: Region,
    blas: Buffer,
    triangles: u32,
    aabb: ([f32; 3], [f32; 3]),
    two_sided: bool,
}

struct BlasCopy {
    buf: Buffer,
}

struct CharacterGpu {
    md6: String,
    region: Region,
    geoms: Vec<Geometry>,
    index: Buffer,
    bind: Buffer,
    frame: Buffer,
    influences: Buffer,
    tri_verts: Buffer,
    skinned: Buffer,
    skinned_frame: Buffer,
    palette: Buffer,
    scratch: Buffer,
    copies: Vec<Arc<BlasCopy>>,
    current: usize,
    result_bytes: u64,
    bytes: u64,
    vertices: u32,
    triangles: u32,
    joints: u32,
    aabb: ([f32; 3], [f32; 3]),
    two_sided: bool,
}

struct Character {
    gpu: CharacterGpu,
    transform: Row34,
    hidden: bool,
    /// A pose was applied (characters enter TLASes only then).
    posed: bool,
    pending: Option<Vec<Row34>>,
}

struct Prop {
    mesh: Arc<PropMesh>,
    transform: Row34,
    hidden: bool,
}

#[derive(Clone, Copy, Debug)]
struct Over {
    transform: Row34,
    hidden: bool,
}

enum Grave {
    Prop(Arc<PropMesh>),
    Character(Box<CharacterGpu>),
}

struct DynTexture {
    slot: u32,
    users: u32,
    bytes: u64,
    _res: ID3D12Resource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Static(u32),
    Prop(u64),
    Character(u64),
}

/// Instance types (`InstanceSet::types`, `SceneView::instance_types` in the game's DLL): a
/// u32 per TLAS instance index (`CommittedInstanceIndex()`).
pub const TYPE_STATIC: u32 = 0;
/// A static scene instance the engine moved (its transform differs from the scene's).
pub const TYPE_MOVED: u32 = 1;
/// A model registered after the scene build (decoded as a dynamic mesh).
pub const TYPE_SPAWNED: u32 = 2;
/// A skinned character.
pub const TYPE_CHARACTER: u32 = 3;

/// Debug colour of an instance type (static grey, moved yellow, spawned green, character
/// red); `None` (a miss) dark blue.
pub fn type_colour(t: Option<u32>) -> [u8; 3] {
    match t {
        None => [16, 20, 48],
        Some(TYPE_MOVED) => [235, 205, 40],
        Some(TYPE_SPAWNED) => [40, 200, 70],
        Some(TYPE_CHARACTER) => [225, 40, 40],
        Some(_) => [130, 130, 130],
    }
}

/// The instances of one TLAS build: descriptors, what they are, and the dynamic BLASes
/// they reference (kept alive with the set).
#[derive(Clone)]
pub struct InstanceSet {
    pub descs: Vec<[u32; 16]>,
    /// `TYPE_*` per instance (parallel to `descs`).
    pub types: Vec<u32>,
    slots: Vec<Slot>,
    keep: Vec<Keep>,
    topology: u64,
    /// [`Dynamics::version`] the descriptors reflect.
    pub version: u64,
}

impl InstanceSet {
    pub fn len(&self) -> usize {
        self.descs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descs.is_empty()
    }

    /// Dynamic instances (props and characters) in the set.
    pub fn dynamic_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s, Slot::Static(_)))
            .count()
    }

    /// The static scene instance at TLAS instance index `k` (None for props, characters
    /// and out of range).
    pub fn static_instance(&self, k: usize) -> Option<u32> {
        match self.slots.get(k)? {
            Slot::Static(i) => Some(*i),
            _ => None,
        }
    }

    /// The prop or character id at TLAS instance index `k`.
    pub fn dynamic_id(&self, k: usize) -> Option<u64> {
        match self.slots.get(k)? {
            Slot::Prop(id) | Slot::Character(id) => Some(*id),
            Slot::Static(_) => None,
        }
    }
}

/// A TLAS with the dynamic BLASes its instances reference (photo sessions trace it instead
/// of [`GpuScene::full_tlas`] when the scene has dynamic objects) and their types.
pub struct SceneTlas {
    pub tlas: Tlas,
    /// `TYPE_*` per instance index.
    pub types: Vec<u32>,
    _keep: Vec<Keep>,
}

impl SceneTlas {
    /// Builds a TLAS (blocking, one submit) over `set` with room for `capacity` instances.
    pub fn build(gpu: &mut Gpu, set: &InstanceSet, capacity: usize) -> Result<Self, String> {
        Ok(Self {
            tlas: Tlas::build(gpu, &set.descs, capacity)?,
            types: set.types.clone(),
            _keep: set.keep.clone(),
        })
    }

    pub fn va(&self) -> u64 {
        self.tlas.va()
    }
}

/// Counters for logs and the overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicStats {
    pub moved_static: usize,
    pub hidden_static: usize,
    pub props: usize,
    pub prop_meshes: usize,
    pub characters: usize,
    pub posed: usize,
    pub pending_poses: usize,
    pub triangles_used: u32,
    pub triangles_reserved: u32,
    pub textures_used: u32,
    /// Video memory of dynamic BLASes, character buffers and dynamic textures.
    pub bytes: u64,
}

/// What [`Dynamics::update`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UpdateStats {
    /// Characters skinned and refitted in this update's submit.
    pub skinned: usize,
    /// Pending poses left for later (budget, or every BLAS copy still referenced).
    pub deferred: usize,
    pub no_free_copy: usize,
    /// GPU ms of the submit (0 without one).
    pub gpu_ms: f64,
}

/// Row-major 3x4 transform of world-space bounds of `lo..hi` under `t`.
pub fn world_aabb(t: &Row34, lo: [f32; 3], hi: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let mut a = [t[3], t[7], t[11]];
    let mut b = a;
    for r in 0..3 {
        for c in 0..3 {
            let x = t[r * 4 + c] * lo[c];
            let y = t[r * 4 + c] * hi[c];
            a[r] += x.min(y);
            b[r] += x.max(y);
        }
    }
    (a, b)
}

fn aabb_dist2(lo: [f32; 3], hi: [f32; 3], p: [f32; 3]) -> f32 {
    (0..3)
        .map(|k| {
            let c = p[k].clamp(lo[k], hi[k]) - p[k];
            c * c
        })
        .sum()
}

/// A sub-range view of a buffer (root descriptors take a GPU address).
fn view(b: &Buffer, offset: u64, len: u64) -> Result<Buffer, String> {
    if offset.checked_add(len).is_none_or(|e| e > b.size) {
        return Err(format!(
            "range {offset}+{len} outside a {}-byte buffer",
            b.size
        ));
    }
    Ok(Buffer {
        res: b.res.clone(),
        size: len,
        va: b.va + offset,
    })
}

fn desc(
    transform: &Row34,
    origin: Origin,
    two_sided: bool,
    first_geometry: u32,
    blas: u64,
) -> [u32; 16] {
    accel::instance_desc(
        &Instance {
            mesh: 0,
            transform: *transform,
            origin,
            two_sided,
        },
        first_geometry,
        blas,
        true,
    )
}

/// Dynamic objects of one uploaded scene (see the module doc). Owned by the thread that
/// owns the [`Gpu`].
pub struct Dynamics {
    scene: Arc<GpuScene>,
    tables: DynamicTables,
    reserve: DynamicReserve,
    programs: Programs,
    geoms: Ranges,
    tris: Ranges,
    mats: Ranges,
    slots: Ranges,
    textures: HashMap<String, DynTexture>,
    over_index: Vec<u32>,
    overs: Vec<Over>,
    props: HashMap<u64, Prop>,
    meshes: HashMap<String, Arc<PropMesh>>,
    characters: HashMap<u64, Character>,
    graveyard: Vec<Grave>,
    version: u64,
    topology: u64,
    /// GPU ms per skinned character (running mean of recent updates).
    ms_per_character: f64,
}

impl Dynamics {
    /// Fails for a scene uploaded without a dynamic reserve.
    pub fn new(gpu: &Gpu, scene: Arc<GpuScene>) -> Result<Self, String> {
        let tables = scene.dynamic_tables().clone();
        let reserve = tables
            .reserve
            .ok_or("the scene was uploaded without a dynamic reserve")?;
        let entries: Vec<Entry> = ENTRIES
            .iter()
            .map(|name| Entry {
                name,
                required_srvs: &[],
                srv_bytes_per_pixel: &[0; 5],
                uav_bytes_per_pixel: &[0; 4],
            })
            .collect();
        let programs = Programs::from_dxil(gpu, LAYOUT, &entries, crate::ingame::dxil)?;
        let tex = scene.textures();
        let b = tables.base;
        Ok(Self {
            geoms: Ranges::new(b.geometries, reserve.geometries),
            tris: Ranges::new(b.triangles, reserve.triangles),
            mats: Ranges::new(b.materials, reserve.materials),
            slots: Ranges::new(tex.reserve_start, tex.count - tex.reserve_start),
            over_index: vec![u32::MAX; scene.instances().len()],
            scene,
            tables,
            reserve,
            programs,
            textures: HashMap::new(),
            overs: Vec::new(),
            props: HashMap::new(),
            meshes: HashMap::new(),
            characters: HashMap::new(),
            graveyard: Vec::new(),
            version: 1,
            topology: 1,
            ms_per_character: 0.05,
        })
    }

    pub fn scene(&self) -> &Arc<GpuScene> {
        &self.scene
    }

    /// Changes with every transform, pose, add, remove or visibility change.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Changes when an instance appears or disappears (a refit is not possible then).
    pub fn topology(&self) -> u64 {
        self.topology
    }

    /// Instances a TLAS of this scene must have room for (static + reserve).
    pub fn tlas_capacity(&self) -> usize {
        self.scene.instances().len() + self.reserve.instances as usize
    }

    fn changed(&mut self, topology: bool) {
        self.version += 1;
        if topology {
            self.topology += 1;
        }
    }

    // ------------------------------------------------------------------ static instances --

    /// Moves (`transform`) and/or hides a static instance. Returns whether anything changed.
    pub fn set_static(
        &mut self,
        instance: u32,
        transform: Option<Row34>,
        hidden: Option<bool>,
    ) -> bool {
        let Some(base) = self.scene.instances().get(instance as usize) else {
            return false;
        };
        let k = self.over_index[instance as usize];
        let cur = if k == u32::MAX {
            Over {
                transform: base.transform,
                hidden: false,
            }
        } else {
            self.overs[k as usize]
        };
        let next = Over {
            transform: transform.unwrap_or(cur.transform),
            hidden: hidden.unwrap_or(cur.hidden),
        };
        if next.transform == cur.transform && next.hidden == cur.hidden {
            return false;
        }
        if k == u32::MAX {
            self.over_index[instance as usize] = self.overs.len() as u32;
            self.overs.push(next);
        } else {
            self.overs[k as usize] = next;
        }
        self.changed(next.hidden != cur.hidden);
        true
    }

    fn static_state(&self, i: usize) -> (Row34, bool, bool) {
        match self.over_index[i] {
            u32::MAX => (self.scene.instances()[i].transform, false, false),
            k => {
                let o = self.overs[k as usize];
                (o.transform, o.hidden, true)
            }
        }
    }

    fn static_type(&self, i: usize, transform: &Row34) -> u32 {
        if self.scene.instances()[i].transform == *transform {
            TYPE_STATIC
        } else {
            TYPE_MOVED
        }
    }

    fn static_desc(&self, i: usize, transform: &Row34) -> [u32; 16] {
        let inst = &self.scene.instances()[i];
        let m = inst.mesh as usize;
        desc(
            transform,
            inst.origin,
            inst.two_sided,
            self.scene.meshes()[m].first_geometry,
            self.scene.blas_va(m).unwrap_or(0),
        )
    }

    // ------------------------------------------------------------------------ mesh data --

    fn free_region(&mut self, gpu: &mut Gpu, r: &Region) -> Result<(), String> {
        self.geoms.release(r.geoms.0, r.geoms.1);
        self.tris.release(r.tris.0, r.tris.1);
        self.mats.release(r.mats.0, r.mats.1);
        for name in &r.textures {
            let gone = match self.textures.get_mut(name) {
                Some(t) => {
                    t.users -= 1;
                    t.users == 0
                }
                None => false,
            };
            if gone && let Some(t) = self.textures.remove(name) {
                // No dispatch may see a descriptor change (the heap's descriptors are
                // static while a list executes): wait for our work in flight first.
                gpu.wait_idle()?;
                self.scene.textures().clear_slot(gpu, t.slot);
                self.slots.release(t.slot, 1);
            }
        }
        Ok(())
    }

    /// Copies the tables of a one-mesh scene into the reserve (textures into reserve slots
    /// unless the scene already has them). Blocking submits.
    fn upload_tables(&mut self, gpu: &mut Gpu, s: &Scene) -> Result<Region, String> {
        let geoms = s.primitive_offsets.len() as u32;
        let tris = s.records.len() as u32;
        let mats = s.gpu_materials.len() as u32;
        let prims = self.scene.primitives();
        if s.meshes.len() != 1 || geoms == 0 || tris == 0 || mats == 0 {
            return Err("dynamic mesh: expected one mesh with photo records".into());
        }
        if prims.is_some() && s.primitives.len() != s.records.len() {
            return Err(format!(
                "dynamic mesh: {} FC6 primitives for {} records",
                s.primitives.len(),
                s.records.len()
            ));
        }
        let mut r = Region::default();
        let full = |what: &str, n: u32, ranges: &Ranges| {
            format!(
                "dynamic reserve full: {n} {what} needed, {} of {} in use",
                ranges.used(),
                ranges.total
            )
        };
        r.geoms = (
            self.geoms
                .alloc(geoms)
                .ok_or_else(|| full("geometries", geoms, &self.geoms))?,
            geoms,
        );
        let Some(t0) = self.tris.alloc(tris) else {
            let e = full("triangles", tris, &self.tris);
            self.free_region(gpu, &r)?;
            return Err(e);
        };
        r.tris = (t0, tris);
        let Some(m0) = self.mats.alloc(mats) else {
            let e = full("materials", mats, &self.mats);
            self.free_region(gpu, &r)?;
            return Err(e);
        };
        r.mats = (m0, mats);
        let res = self.fill_tables(gpu, s, &mut r);
        if let Err(e) = res {
            let _ = gpu.discard();
            self.free_region(gpu, &r)?;
            return Err(e);
        }
        Ok(r)
    }

    fn fill_tables(&mut self, gpu: &mut Gpu, s: &Scene, r: &mut Region) -> Result<(), String> {
        // Textures: local slot k + 1 -> global slot.
        let mut slot_of = vec![0u32; s.textures.len() + 1];
        for (k, t) in s.textures.iter().enumerate() {
            let global = if let Some(&g) = self.tables.texture_slots.get(&t.name) {
                g
            } else if let Some(d) = self.textures.get_mut(&t.name) {
                d.users += 1;
                r.textures.push(t.name.clone());
                d.slot
            } else if let Some(slot) = self.slots.alloc(1) {
                match self.scene.textures().upload_slot(gpu, slot, t) {
                    Ok((res, bytes)) => {
                        self.textures.insert(
                            t.name.clone(),
                            DynTexture {
                                slot,
                                users: 1,
                                bytes,
                                _res: res,
                            },
                        );
                        r.textures.push(t.name.clone());
                        slot
                    }
                    Err(e) if e.contains("removed") => return Err(e),
                    Err(_) => {
                        self.slots.release(slot, 1);
                        0
                    }
                }
            } else {
                0
            };
            slot_of[k + 1] = global;
        }
        let materials: Vec<[u32; MATERIAL_WORDS]> = s
            .gpu_materials
            .iter()
            .map(|m| {
                let mut m = *m;
                for w in &mut m[84..100] {
                    *w = slot_of.get(*w as usize).copied().unwrap_or(0);
                }
                m
            })
            .collect();
        let (g0, t0, m0) = (r.geoms.0, r.tris.0, r.mats.0);
        let geometry: Vec<[u32; 2]> = s
            .primitive_offsets
            .iter()
            .zip(&s.geometry_materials)
            .map(|(&p, &m)| [p + t0, m + m0])
            .collect();
        let offsets: Vec<u32> = s.primitive_offsets.iter().map(|&p| p + t0).collect();
        let t = self.scene.tables();
        let mut staging = vec![
            gpu.record_upload(&t.geometry, u64::from(g0) * 8, &geometry)?,
            gpu.record_upload(
                &t.records,
                u64::from(t0) * RECORD_WORDS as u64 * 4,
                &s.records,
            )?,
            gpu.record_upload(
                &t.materials,
                u64::from(m0) * MATERIAL_WORDS as u64 * 4,
                &materials,
            )?,
        ];
        if let (Some(p), Some(o)) = (self.scene.primitives(), self.scene.primitive_offsets()) {
            staging.push(gpu.record_upload(p, u64::from(t0) * 24, &s.primitives)?);
            staging.push(gpu.record_upload(o, u64::from(g0) * 4, &offsets)?);
        }
        gpu.submit()?;
        drop(staging);
        Ok(())
    }

    // ---------------------------------------------------------------------------- props --

    pub fn has_mesh(&self, key: &str) -> bool {
        self.meshes.contains_key(key)
    }

    /// Uploads a rigid one-mesh scene ([`crate::scene::mesh_scene`]) under `key` (blocking
    /// submits: tables, then a PREFER_FAST_TRACE BLAS).
    pub fn add_mesh(&mut self, gpu: &mut Gpu, key: &str, s: &Scene) -> Result<(), String> {
        if self.meshes.contains_key(key) {
            return Ok(());
        }
        let m = s.meshes.first().ok_or("no mesh")?;
        let (result, scratch, _) = accel::dynamic_blas_sizes(gpu, &m.geometries, BlasKind::Rigid);
        let inputs = std::mem::size_of_val(s.positions.as_slice())
            + std::mem::size_of_val(s.indices.as_slice());
        gpu.check_vram(result + scratch + inputs as u64, "a dynamic mesh")?;
        let region = self.upload_tables(gpu, s)?;
        let built = (|| -> Result<Buffer, String> {
            let pos = gpu.upload(&s.positions, false)?;
            let idx = gpu.upload(&s.indices, false)?;
            let blas = gpu.accel_buffer(result)?;
            let scr = gpu.buffer(scratch, true)?;
            gpu.transition(
                &pos,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            gpu.transition(
                &idx,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            gpu.transition(
                &scr,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            accel::record_dynamic_blas(
                gpu,
                &m.geometries,
                pos.va,
                idx.va,
                BlasKind::Rigid,
                blas.va,
                scr.va,
                None,
            );
            gpu.submit()?;
            Ok(blas)
        })();
        let blas = match built {
            Ok(b) => b,
            Err(e) => {
                let _ = gpu.discard();
                self.free_region(gpu, &region)?;
                return Err(e);
            }
        };
        self.meshes.insert(
            key.to_owned(),
            Arc::new(PropMesh {
                name: key.to_owned(),
                triangles: region.tris.1,
                region,
                blas,
                aabb: (m.aabb_min, m.aabb_max),
                two_sided: s.materials.iter().any(|x| x.two_sided),
            }),
        );
        Ok(())
    }

    /// Places (or moves) prop `id` with mesh `key` (from [`Dynamics::add_mesh`]).
    pub fn place(&mut self, id: u64, key: &str, transform: Row34) -> Result<(), String> {
        let mesh = self
            .meshes
            .get(key)
            .cloned()
            .ok_or_else(|| format!("no dynamic mesh {key}"))?;
        let new = !self.props.contains_key(&id);
        if new && self.props.len() + self.characters.len() >= self.reserve.instances as usize {
            return Err(format!(
                "dynamic instance reserve ({}) full",
                self.reserve.instances
            ));
        }
        self.props.insert(
            id,
            Prop {
                mesh,
                transform,
                hidden: false,
            },
        );
        self.changed(new);
        Ok(())
    }

    /// Removes prop or character `id` (its mesh data is reused once no TLAS holds it).
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(p) = self.props.remove(&id) {
            let key = p.mesh.name.clone();
            drop(p);
            // No prop uses the mesh any more: retire it (its reserve entries are reused
            // once no instance set holds it, `collect`).
            if !self.props.values().any(|q| q.mesh.name == key)
                && let Some(m) = self.meshes.remove(&key)
            {
                self.graveyard.push(Grave::Prop(m));
            }
            self.changed(true);
            return true;
        }
        if let Some(c) = self.characters.remove(&id) {
            self.graveyard.push(Grave::Character(Box::new(c.gpu)));
            self.changed(c.posed && !c.hidden);
            return true;
        }
        false
    }

    pub fn set_transform(&mut self, id: u64, t: Row34) -> bool {
        let changed = if let Some(p) = self.props.get_mut(&id) {
            std::mem::replace(&mut p.transform, t) != t
        } else if let Some(c) = self.characters.get_mut(&id) {
            std::mem::replace(&mut c.transform, t) != t
        } else {
            false
        };
        if changed {
            self.changed(false);
        }
        changed
    }

    pub fn set_hidden(&mut self, id: u64, hidden: bool) -> bool {
        let changed = if let Some(p) = self.props.get_mut(&id) {
            std::mem::replace(&mut p.hidden, hidden) != hidden
        } else if let Some(c) = self.characters.get_mut(&id) {
            std::mem::replace(&mut c.hidden, hidden) != hidden
        } else {
            false
        };
        if changed {
            self.changed(true);
        }
        changed
    }

    pub fn contains(&self, id: u64) -> bool {
        self.props.contains_key(&id) || self.characters.contains_key(&id)
    }

    /// Back to the static scene for props: every static instance at its scene transform and
    /// visible, every prop removed (characters stay).
    pub fn reset_props(&mut self) {
        let ids: Vec<u64> = self.props.keys().copied().collect();
        for id in ids {
            self.remove(id);
        }
        if !self.overs.is_empty() {
            self.overs.clear();
            self.over_index.fill(u32::MAX);
            self.changed(true);
        }
    }

    // ----------------------------------------------------------------------- characters --

    pub fn has_character(&self, id: u64) -> bool {
        self.characters.contains_key(&id)
    }

    /// Current model transform of character `id`.
    pub fn character_transform(&self, id: u64) -> Option<Row34> {
        self.characters.get(&id).map(|c| c.transform)
    }

    /// Video memory a character needs (buffers, one BLAS copy, scratch; no GPU work).
    pub fn character_bytes(gpu: &Gpu, cpu: &CharacterCpu) -> u64 {
        let m = cpu
            .mesh
            .meshes
            .first()
            .map_or(&[][..], |m| m.geometries.as_slice());
        let (result, build, update) = accel::dynamic_blas_sizes(gpu, m, BlasKind::Deforming);
        let v = u64::from(cpu.vertex_count());
        v * (12 + 8 + 16 + 12 + 8)
            + std::mem::size_of_val(cpu.mesh.indices.as_slice()) as u64
            + u64::from(cpu.triangle_count()) * 12
            + u64::from(cpu.joint_count()) * 48
            + result
            + build.max(update)
    }

    /// Uploads character `id` (tables, skinning buffers, a BLAS of the bind pose; blocking
    /// submits). It enters TLASes with its first [`Dynamics::pose`].
    pub fn add_character(
        &mut self,
        gpu: &mut Gpu,
        id: u64,
        cpu: &CharacterCpu,
        transform: Row34,
    ) -> Result<(), String> {
        if self.characters.contains_key(&id) {
            return Ok(());
        }
        if self.props.len() + self.characters.len() >= self.reserve.instances as usize {
            return Err(format!(
                "dynamic instance reserve ({}) full",
                self.reserve.instances
            ));
        }
        let s = &cpu.mesh;
        let m = s.meshes.first().ok_or("no mesh")?;
        let (verts, tris, joints) = (cpu.vertex_count(), cpu.triangle_count(), cpu.joint_count());
        if verts == 0
            || tris == 0
            || joints == 0
            || cpu.influences.len() != verts as usize
            || cpu.frames.len() != verts as usize
            || tris as usize != s.records.len()
            || cpu.tri_verts.iter().flatten().any(|&v| v >= verts)
        {
            return Err(format!(
                "{}: inconsistent skinning data ({verts} vertices, {tris} triangles, {joints} joints)",
                cpu.md6
            ));
        }
        let bytes = Self::character_bytes(gpu, cpu);
        gpu.check_vram(bytes, "a character")?;
        let region = self.upload_tables(gpu, s)?;
        let (result, build, update) =
            accel::dynamic_blas_sizes(gpu, &m.geometries, BlasKind::Deforming);
        let made = (|| -> Result<CharacterGpu, String> {
            let c = CharacterGpu {
                md6: cpu.md6.clone(),
                geoms: m.geometries.clone(),
                index: gpu.upload(&s.indices, false)?,
                bind: gpu.upload(&s.positions, false)?,
                frame: gpu.upload(&cpu.frames, false)?,
                influences: gpu.upload(&cpu.influences, false)?,
                tri_verts: gpu.upload(&cpu.tri_verts, false)?,
                skinned: gpu.buffer(u64::from(verts) * 12, true)?,
                skinned_frame: gpu.buffer(u64::from(verts) * 8, true)?,
                palette: gpu.upload_heap(&cpu.bind_palette())?,
                scratch: gpu.buffer(build.max(update), true)?,
                copies: vec![Arc::new(BlasCopy {
                    buf: gpu.accel_buffer(result)?,
                })],
                current: 0,
                result_bytes: result,
                bytes,
                vertices: verts,
                triangles: tris,
                joints,
                aabb: (
                    m.aabb_min.map(|v| v - POSE_MARGIN),
                    m.aabb_max.map(|v| v + POSE_MARGIN),
                ),
                two_sided: s.materials.iter().any(|x| x.two_sided),
                region: region.clone(),
            };
            self.record_skin(gpu, &c)?;
            self.record_blas(gpu, &c, 0, None);
            gpu.submit()?;
            Ok(c)
        })();
        let c = match made {
            Ok(c) => c,
            Err(e) => {
                let _ = gpu.discard();
                self.free_region(gpu, &region)?;
                return Err(e);
            }
        };
        self.characters.insert(
            id,
            Character {
                gpu: c,
                transform,
                hidden: false,
                posed: false,
                pending: None,
            },
        );
        Ok(())
    }

    /// Queues a pose of character `id`: `palette` (one bone -> object matrix per skeleton
    /// joint, see [`crate::character::effective_palette`]) and its model transform.
    pub fn pose(&mut self, id: u64, palette: Vec<Row34>, transform: Row34) -> Result<(), String> {
        let c = self
            .characters
            .get_mut(&id)
            .ok_or_else(|| format!("no character {id:#x}"))?;
        if palette.len() != c.gpu.joints as usize {
            return Err(format!(
                "{}: palette of {} joints, skeleton has {}",
                c.gpu.md6,
                palette.len(),
                c.gpu.joints
            ));
        }
        if palette.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("{}: non-finite palette", c.gpu.md6));
        }
        c.transform = transform;
        c.pending = Some(palette);
        self.version += 1;
        Ok(())
    }

    fn record_skin(&self, gpu: &Gpu, c: &CharacterGpu) -> Result<(), String> {
        let t = self.scene.tables();
        let tris = u64::from(c.triangles);
        let records = view(
            &t.records,
            u64::from(c.region.tris.0) * RECORD_WORDS as u64 * 4,
            tris * RECORD_WORDS as u64 * 4,
        )?;
        let prims = self
            .scene
            .primitives()
            .map(|p| view(p, u64::from(c.region.tris.0) * 24, tris * 24))
            .transpose()?;
        let need = [
            (&c.bind, u64::from(c.vertices) * 12),
            (&c.frame, u64::from(c.vertices) * 8),
            (&c.influences, u64::from(c.vertices) * 16),
            (&c.palette, u64::from(c.joints) * 48),
            (&c.tri_verts, tris * 12),
            (&c.skinned, u64::from(c.vertices) * 12),
            (&c.skinned_frame, u64::from(c.vertices) * 8),
        ];
        if let Some((b, n)) = need.iter().find(|(b, n)| b.size < *n) {
            return Err(format!("{}: buffer of {} bytes < {n}", c.md6, b.size));
        }
        gpu.transition(
            &c.skinned,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        gpu.transition(
            &c.skinned_frame,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        self.programs.dispatch(
            gpu,
            "skin_vertices",
            &[c.vertices, c.joints, ROW, c.vertices, 0, 0, 0, 0],
            &[
                Some(&c.bind),
                Some(&c.frame),
                Some(&c.influences),
                Some(&c.palette),
                None,
            ],
            &[Some(&c.skinned), Some(&c.skinned_frame), None, None],
            None,
            (ROW, c.vertices.div_ceil(ROW)),
        )?;
        gpu.uav_barrier();
        self.programs.dispatch(
            gpu,
            "skin_triangles",
            &[
                c.triangles,
                c.joints,
                ROW,
                c.vertices,
                u32::from(prims.is_some()),
                0,
                0,
                0,
            ],
            &[None, None, None, None, Some(&c.tri_verts)],
            &[None, Some(&c.skinned_frame), Some(&records), prims.as_ref()],
            None,
            (ROW, c.triangles.div_ceil(ROW)),
        )?;
        gpu.uav_barrier();
        gpu.transition(
            &c.skinned,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        Ok(())
    }

    /// Records the BLAS build (`source` None) or update of `c` into copy `dest`.
    fn record_blas(&self, gpu: &Gpu, c: &CharacterGpu, dest: usize, source: Option<usize>) {
        gpu.transition(
            &c.index,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        gpu.transition(
            &c.scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        accel::record_dynamic_blas(
            gpu,
            &c.geoms,
            c.skinned.va,
            c.index.va,
            BlasKind::Deforming,
            c.copies[dest].buf.va,
            c.scratch.va,
            source.map(|s| c.copies[s].buf.va),
        );
    }

    /// Applies queued poses, nearest to `center` first: at most `max_characters` and about
    /// `budget_ms` of GPU time (from the measured cost per character), skinning + BLAS
    /// update into a free copy, all in one blocking submit. Characters without a free copy
    /// (every other copy still referenced by a TLAS in use) keep their pose for later.
    pub fn update(
        &mut self,
        gpu: &mut Gpu,
        center: [f32; 3],
        max_characters: usize,
        budget_ms: f64,
    ) -> Result<UpdateStats, String> {
        let mut order: Vec<(f32, u64)> = self
            .characters
            .iter()
            .filter(|(_, c)| c.pending.is_some())
            .map(|(&id, c)| {
                let t = &c.transform;
                let d = (t[3] - center[0]).powi(2)
                    + (t[7] - center[1]).powi(2)
                    + (t[11] - center[2]).powi(2);
                (d, id)
            })
            .collect();
        let mut st = UpdateStats::default();
        if order.is_empty() {
            return Ok(st);
        }
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let by_budget = (budget_ms / self.ms_per_character.max(1e-3)).floor() as usize;
        let limit = max_characters.min(by_budget.max(1));
        let mut done: Vec<(u64, usize)> = Vec::new();
        for &(_, id) in &order {
            if done.len() >= limit {
                st.deferred += 1;
                continue;
            }
            let free = {
                let c = &self.characters[&id].gpu;
                (0..c.copies.len())
                    .find(|&k| k != c.current && Arc::strong_count(&c.copies[k]) == 1)
            };
            let (k, fresh) = match free {
                Some(k) => (k, false),
                None if self.characters[&id].gpu.copies.len() < BLAS_COPIES => {
                    let bytes = self.characters[&id].gpu.result_bytes;
                    gpu.check_vram(bytes, "a character BLAS copy")?;
                    let buf = gpu.accel_buffer(bytes)?;
                    let c = &mut self.characters.get_mut(&id).ok_or("character")?.gpu;
                    c.copies.push(Arc::new(BlasCopy { buf }));
                    c.bytes += bytes;
                    (c.copies.len() - 1, true)
                }
                None => {
                    st.no_free_copy += 1;
                    st.deferred += 1;
                    continue;
                }
            };
            let c = &self.characters[&id];
            if let Some(p) = &c.pending {
                gpu.write_mapped(&c.gpu.palette, p)?;
            }
            self.record_skin(gpu, &c.gpu)?;
            // A new copy gets a full build (an update into never-built memory was traced as
            // empty on WARP, skin_check); later poses update it from the current copy.
            let source = (!fresh).then_some(c.gpu.current);
            self.record_blas(gpu, &c.gpu, k, source);
            done.push((id, k));
        }
        if done.is_empty() {
            return Ok(st);
        }
        gpu.submit()?;
        st.gpu_ms = gpu.last_submit_ms();
        st.skinned = done.len();
        if st.gpu_ms > 0.0 {
            let per = st.gpu_ms / done.len() as f64;
            self.ms_per_character = 0.8 * self.ms_per_character + 0.2 * per;
        }
        let mut appeared = false;
        for (id, k) in done {
            if let Some(c) = self.characters.get_mut(&id) {
                c.gpu.current = k;
                c.pending = None;
                appeared |= !c.posed && !c.hidden;
                c.posed = true;
            }
        }
        self.changed(appeared);
        Ok(st)
    }

    /// Frees reserve entries and textures of removed meshes no TLAS references any more.
    pub fn collect(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let graves = std::mem::take(&mut self.graveyard);
        for g in graves {
            match g {
                Grave::Prop(m) if Arc::strong_count(&m) == 1 => {
                    let r = m.region.clone();
                    drop(m);
                    self.free_region(gpu, &r)?;
                }
                Grave::Character(c) if c.copies.iter().all(|k| Arc::strong_count(k) == 1) => {
                    let r = c.region.clone();
                    drop(c);
                    self.free_region(gpu, &r)?;
                }
                g => self.graveyard.push(g),
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------------------- instance sets --

    fn prop_desc(&self, p: &Prop) -> [u32; 16] {
        desc(
            &p.transform,
            Origin::Entity,
            p.mesh.two_sided,
            p.mesh.region.geoms.0,
            p.mesh.blas.va,
        )
    }

    fn character_desc(&self, c: &Character) -> ([u32; 16], Keep) {
        let copy = c.gpu.copies[c.gpu.current].clone();
        (
            desc(
                &c.transform,
                Origin::Entity,
                c.gpu.two_sided,
                c.gpu.region.geoms.0,
                copy.buf.va,
            ),
            copy,
        )
    }

    fn set(
        &self,
        keep_static: impl Fn(usize, &Row34, bool) -> bool,
        keep_aabb: impl Fn(([f32; 3], [f32; 3])) -> bool,
    ) -> InstanceSet {
        let mut s = InstanceSet {
            descs: Vec::new(),
            types: Vec::new(),
            slots: Vec::new(),
            keep: Vec::new(),
            topology: self.topology,
            version: self.version,
        };
        for i in 0..self.scene.instances().len() {
            let (t, hidden, over) = self.static_state(i);
            if hidden || !keep_static(i, &t, over) {
                continue;
            }
            s.descs.push(self.static_desc(i, &t));
            s.types.push(self.static_type(i, &t));
            s.slots.push(Slot::Static(i as u32));
        }
        for (&id, p) in &self.props {
            if p.hidden || !keep_aabb(world_aabb(&p.transform, p.mesh.aabb.0, p.mesh.aabb.1)) {
                continue;
            }
            s.descs.push(self.prop_desc(p));
            s.types.push(TYPE_SPAWNED);
            s.slots.push(Slot::Prop(id));
            s.keep.push(p.mesh.clone());
        }
        for (&id, c) in &self.characters {
            if c.hidden
                || !c.posed
                || !keep_aabb(world_aabb(&c.transform, c.gpu.aabb.0, c.gpu.aabb.1))
            {
                continue;
            }
            let (d, k) = self.character_desc(c);
            s.descs.push(d);
            s.types.push(TYPE_CHARACTER);
            s.slots.push(Slot::Character(id));
            s.keep.push(k);
        }
        s
    }

    /// Instances whose world bounds lie within `radius` of `center` (static ones with their
    /// current transform and visibility), props and posed characters included.
    pub fn cutoff(&self, center: [f32; 3], radius: f32) -> InstanceSet {
        let r2 = radius * radius;
        let meshes = self.scene.meshes();
        self.set(
            |i, t, over| {
                let (lo, hi) = if over {
                    let m = &meshes[self.scene.instances()[i].mesh as usize];
                    world_aabb(t, m.aabb_min, m.aabb_max)
                } else {
                    self.scene.instance_aabb(i).unwrap_or(([0.0; 3], [0.0; 3]))
                };
                aabb_dist2(lo, hi, center) <= r2
            },
            |(lo, hi)| aabb_dist2(lo, hi, center) <= r2,
        )
    }

    /// Every visible instance (the photo mode's frozen TLAS).
    pub fn full(&self) -> InstanceSet {
        self.set(|_, _, _| true, |_| true)
    }

    /// `set`'s instances with their current transforms and BLASes (for a TLAS refit from
    /// the TLAS built over `set`); None when instances appeared or disappeared since.
    pub fn refresh(&self, set: &InstanceSet) -> Option<InstanceSet> {
        if set.topology != self.topology {
            return None;
        }
        let mut s = InstanceSet {
            descs: set.descs.clone(),
            types: set.types.clone(),
            slots: set.slots.clone(),
            keep: Vec::with_capacity(set.keep.len()),
            topology: self.topology,
            version: self.version,
        };
        for (k, slot) in set.slots.iter().enumerate() {
            match *slot {
                Slot::Static(i) => {
                    if self.over_index[i as usize] != u32::MAX {
                        let (t, hidden, _) = self.static_state(i as usize);
                        if hidden {
                            return None;
                        }
                        s.descs[k] = self.static_desc(i as usize, &t);
                        s.types[k] = self.static_type(i as usize, &t);
                    }
                }
                Slot::Prop(id) => {
                    let p = self.props.get(&id).filter(|p| !p.hidden)?;
                    s.descs[k] = self.prop_desc(p);
                    s.keep.push(p.mesh.clone());
                }
                Slot::Character(id) => {
                    let c = self.characters.get(&id).filter(|c| !c.hidden && c.posed)?;
                    let (d, keep) = self.character_desc(c);
                    s.descs[k] = d;
                    s.keep.push(keep);
                }
            }
        }
        Some(s)
    }

    pub fn stats(&self) -> DynamicStats {
        DynamicStats {
            moved_static: self.overs.iter().filter(|o| !o.hidden).count(),
            hidden_static: self.overs.iter().filter(|o| o.hidden).count(),
            props: self.props.len(),
            prop_meshes: self.meshes.len(),
            characters: self.characters.len(),
            posed: self.characters.values().filter(|c| c.posed).count(),
            pending_poses: self
                .characters
                .values()
                .filter(|c| c.pending.is_some())
                .count(),
            triangles_used: self.tris.used(),
            triangles_reserved: self.tris.total,
            textures_used: self.slots.used(),
            bytes: self.meshes.values().map(|m| m.blas.size).sum::<u64>()
                + self.characters.values().map(|c| c.gpu.bytes).sum::<u64>()
                + self.textures.values().map(|t| t.bytes).sum::<u64>(),
        }
    }

    /// Triangles of the prop mesh `key` (None when not uploaded).
    pub fn mesh_triangles(&self, key: &str) -> Option<u32> {
        self.meshes.get(key).map(|m| m.triangles)
    }

    /// Reads back character `id`'s skinned positions (blocking; tests and tools).
    pub fn read_skinned(&self, gpu: &mut Gpu, id: u64) -> Result<Vec<[f32; 3]>, String> {
        let c = self.characters.get(&id).ok_or("no such character")?;
        gpu.read(&c.gpu.skinned, c.gpu.vertices as usize)
    }

    /// Reads back character `id`'s photo records (blocking; tests and tools).
    pub fn read_records(&self, gpu: &mut Gpu, id: u64) -> Result<Vec<[u32; RECORD_WORDS]>, String> {
        let c = self.characters.get(&id).ok_or("no such character")?;
        let t = self.scene.tables();
        let first = u64::from(c.gpu.region.tris.0);
        let all: Vec<[u32; RECORD_WORDS]> =
            gpu.read(&t.records, (first + u64::from(c.gpu.triangles)) as usize)?;
        Ok(all[first as usize..].to_vec())
    }
}

/// CPU decoding of dynamic meshes from the game's archives (run it off the GPU thread; one
/// per map): spawned static props and characters, with the scene's options.
pub struct DynamicAssets {
    archive: Archive,
    md6: Option<Md6Index>,
    opts: LoadOptions,
}

impl DynamicAssets {
    /// Opens the archives with `map`'s group preferred (as the scene load).
    pub fn open(game_dir: &std::path::Path, map: &str, opts: LoadOptions) -> Result<Self, String> {
        let mut archive = Archive::open(game_dir)?;
        if let Some(&(group, _)) = archive.locate("File", &scene::bwm_path(map)).first() {
            archive.preferred_group = group;
        }
        Ok(Self {
            archive,
            md6: None,
            opts,
        })
    }

    /// A spawned static model as a one-mesh scene (its live materials as for static
    /// entities, undecodable or excluded surfaces dropped).
    pub fn prop(&self, key: &str, e: &Entity) -> Result<Scene, String> {
        let ModelResource::Static(res) = model_resource(&e.model_source) else {
            return Err(format!("{}: skinned", e.model_source));
        };
        let surfaces = scene::static_model_surfaces(&self.archive, &res)?;
        let runtime_ok = e.runtime_materials && e.materials.len() == surfaces.len();
        let mut parts = Vec::new();
        let mut mats = Vec::new();
        for (k, s) in surfaces.iter().enumerate() {
            if let Some(s) = s {
                mats.push(scene::entity_material(e, runtime_ok, k, &s.material));
                parts.push(s.clone());
            }
        }
        scene::mesh_scene(&self.archive, key, &parts, &mats, self.opts, false).map(|m| m.scene)
    }

    /// Character `md6` at `lod` ([`load_character`]).
    pub fn character(
        &mut self,
        md6: &str,
        lod: u32,
        materials: &[String],
    ) -> Result<CharacterCpu, String> {
        let index = self.md6.get_or_insert_with(|| Md6Index::new(&self.archive));
        load_character(&self.archive, index, md6, lod, materials, self.opts)
    }
}

/// Primary rays (`rt.hlsl` `primary`, precompiled) into a TLAS of a scene with FC6
/// primitives: per pixel the hit's game-convention depth (`near / z`, 0 = miss) and TLAS
/// instance index. Used by the offline checks and the photo mode's instance-type image.
pub struct InstanceProbe {
    programs: Programs,
    depth: Buffer,
    id: Buffer,
    color: Buffer,
    pub width: u32,
    pub height: u32,
}

/// `ID_MISS` of `rt.hlsl`.
pub const PROBE_MISS: u32 = 0xFFFF_FFFF;

/// One [`InstanceProbe`] render.
#[derive(Clone, PartialEq)]
pub struct ProbeImage {
    pub width: u32,
    pub height: u32,
    pub depth: Vec<f32>,
    pub id: Vec<u32>,
}

impl ProbeImage {
    /// RGB8 by instance type (`types[id]`, [`type_colour`]).
    pub fn type_rgb(&self, types: &[u32]) -> Vec<u8> {
        self.id
            .iter()
            .flat_map(|&i| {
                let t = (i != PROBE_MISS)
                    .then(|| types.get(i as usize).copied().unwrap_or(TYPE_STATIC));
                type_colour(t)
            })
            .collect()
    }
}

impl InstanceProbe {
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Self, String> {
        use crate::gpu::SRVS;
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(format!("probe size {width}x{height}"));
        }
        let programs = Programs::from_dxil(
            gpu,
            Layout::SPIKE,
            &[Entry {
                name: "primary",
                required_srvs: &[0, 1, 2],
                srv_bytes_per_pixel: &[0; SRVS],
                uav_bytes_per_pixel: &[4, 4, 16, 0],
            }],
            crate::ingame::dxil,
        )?;
        let n = u64::from(width) * u64::from(height);
        gpu.check_vram(n * 24, "the instance probe")?;
        Ok(Self {
            programs,
            depth: gpu.buffer(n * 4, true)?,
            id: gpu.buffer(n * 4, true)?,
            color: gpu.buffer(n * 16, true)?,
            width,
            height,
        })
    }

    /// Traces `tlas` (a TLAS of `scene`) from `cam` (blocking, one submit) and reads back.
    pub fn render(
        &self,
        gpu: &mut Gpu,
        scene: &GpuScene,
        tlas: &Buffer,
        cam: &crate::camera::Camera,
    ) -> Result<ProbeImage, String> {
        use crate::gpu::CONSTANTS;
        let (Some(offsets), Some(prims)) = (scene.primitive_offsets(), scene.primitives()) else {
            return Err("the scene has no FC6 primitives".into());
        };
        let b = cam.basis(self.width, self.height);
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
        c[16..19].copy_from_slice(&v([0.3, 0.2, 0.93]));
        c[19] = 0.05f32.to_bits();
        c[20..23].copy_from_slice(&v([1.0; 3]));
        c[24..27].copy_from_slice(&v([0.2; 3]));
        c[30] = 1.0e4f32.to_bits();
        c[32] = (offsets.size / 4) as u32;
        c[33] = (prims.size / 24) as u32;
        self.programs.dispatch(
            gpu,
            "primary",
            &c,
            &[Some(tlas), Some(offsets), Some(prims), None, None],
            &[Some(&self.depth), Some(&self.id), Some(&self.color), None],
            None,
            (self.width, self.height),
        )?;
        gpu.submit()?;
        let n = (self.width * self.height) as usize;
        Ok(ProbeImage {
            width: self.width,
            height: self.height,
            depth: gpu.read(&self.depth, n)?,
            id: gpu.read(&self.id, n)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_allocate_free_and_merge() {
        let mut r = Ranges::new(100, 10);
        assert_eq!(r.alloc(4), Some(100));
        assert_eq!(r.alloc(4), Some(104));
        assert_eq!(r.alloc(4), None);
        assert_eq!(r.used(), 8);
        r.release(100, 4);
        assert_eq!(r.alloc(3), Some(100));
        r.release(104, 4);
        r.release(100, 3);
        assert_eq!(r.free, vec![(100, 10)]);
        assert_eq!(r.alloc(10), Some(100));
        assert_eq!(r.used(), 10);
    }

    #[test]
    fn world_bounds_of_a_rotated_box() {
        // +90 deg about z, then +10 in x.
        let t = [0.0, -1.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (lo, hi) = world_aabb(&t, [0.0, 0.0, 0.0], [2.0, 1.0, 3.0]);
        assert_eq!(lo, [9.0, 0.0, 0.0]);
        assert_eq!(hi, [10.0, 2.0, 3.0]);
    }
}
