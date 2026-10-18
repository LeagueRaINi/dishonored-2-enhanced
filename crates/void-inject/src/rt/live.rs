//! Copies of engine state for ray tracing, read from game memory with checked reads
//! (`mem`): the first view of a `RenderAllViews` request list (camera, map, render world),
//! the world's static model instances, its lights, and the view's sun and sky parameters.
//!
//! Layouts for exe `b671f824...6180` (O: checkpoint 9282529 `scene.rs` / `snapshot.rs`,
//! docs/renderer-replacement.md, E035; filters as `void_rt::snapshot::parse_models`).

use void_rt::lights::{Env, LightParms, SUN_COLOR_SCALE};
use void_rt::snapshot::{Entity, Sun, flags_may_be_drawn, material_is_drawn};

use crate::mem;

/// `RenderAllViews` argument 1: view request list (`idList`).
const REQUESTS_PTR: u64 = 0;
const REQUESTS_COUNT: u64 = 8;
const REQUEST_WORLD: u64 = 32;
const REQUEST_RENDER_VIEW: u64 = 48;
/// `renderView_t`: fov x / y (radians), origin, axis rows (forward, left, up).
const RV_FOV: u64 = 156;
const RV_VIEWORG: u64 = 168;
const RV_VIEWAXIS: u64 = 204;
/// `renderView_t::cramZNear` (near x 0.25 when set).
const RV_CRAM_ZNEAR: u64 = 164;
/// `r_znear` float value slot (cvar object `0x1439373E0` + 44).
const R_ZNEAR_VALUE_RVA: u64 = 0x393_740C;
/// `idRenderWorldLocal`.
const WORLD_MAP_NAME: u64 = 132_328;
const WORLD_MODELS_PTR: u64 = 132_408;
const WORLD_MODELS_COUNT: u64 = 132_416;
const WORLD_LIGHTS_PTR: u64 = 516_464;
const WORLD_LIGHTS_COUNT: u64 = 516_472;
const COUNT_MASK: u32 = 0x00FF_FFFF;

/// `idRenderModel`.
const MODEL_SURFACES_PTR: u64 = 8;
const MODEL_SURFACES_COUNT: u64 = 16;
const MODEL_PARMS: u64 = 6240;
const MODEL_NAME: u64 = 7088;
const MODEL_SURFACES_SHARED: u64 = 7168;
const MODEL_FLAGS: u64 = 7216;
const SURFACE_STRIDE: u64 = 128;
const MAX_MODELS: u32 = 200_000;
const MAX_SURFACES: u32 = 4096;
const MAX_LIGHTS: u32 = 100_000;

/// Dynamic objects (local://dynamic-entities.md, docs/raytracing.md section 14; O from the
/// IDB unless noted).
pub mod dynamic {
    /// `idRenderWorldCommitted*` of the world (`RenderAllViews` commit, `sub_140239400`).
    pub const WORLD_COMMITTED: u64 = 528_816;
    /// This frame's per-model dirty flags (u32 per model slot, cleared at the backend
    /// frame start `sub_140376980`, so complete after `RenderAllViews`).
    pub const COMMITTED_MODEL_FLAGS: u64 = 8328;
    /// Model slots of the flag arrays (state arrays, `memset 0xF000`).
    pub const MAX_SLOTS: u32 = 15_360;
    pub const FLAG_TRANSFORM: u32 = 0x1;
    pub const FLAG_PARMS: u32 = 0x2;
    pub const FLAG_UNLINK: u32 = 0x4;
    pub const FLAG_FREE: u32 = 0x80;
    /// `r_skipUpdateInView` integer value (`dword_143938308`): non-zero skips the commit.
    pub const SKIP_UPDATE_IN_VIEW_RVA: u64 = 0x393_8308;
    /// Vtables (RVA) of `arkSkinnedRenderModel`, `idRenderModelStatic`, `idRenderModel`.
    pub const VTABLE_SKINNED: u64 = 0x1B0_D628;
    pub const VTABLE_STATIC: u64 = 0x1B0_DD88;
    pub const VTABLE_BASE: u64 = 0x1B0_8128;
    /// `arkSkinnedRenderModel`: `m_needJointUploadToGpu` (2 when this frame computed new
    /// joints), `m_currentLodIndex`, `m_jointBufferCpu` idList (ptr, count low 24 bits;
    /// 48-byte row-major 3x4 per joint).
    pub const NEED_JOINT_UPLOAD: u64 = 8113;
    pub const CURRENT_LOD: u64 = 8120;
    pub const JOINTS_PTR: u64 = 8184;
    pub const JOINTS_COUNT: u64 = 8192;
    pub const MAX_JOINTS: u32 = 512;
    /// `modelDepthHack` (renderModel +6328, f32): > 0 for the first-person view model (the
    /// hands and weapon); `ARK_RENDERPASS_SORT` `sub_14033D320` skips such models (O,
    /// docs/raytracing.md 11.1).
    pub const DEPTH_HACK: u64 = 6328;
}

/// The first-person view model (player hands, weapons): `modelDepthHack` > 0, or as a
/// backstop a player weapon / hands path. Never traced (they sit at the camera).
pub fn first_person(m: u64, name: &str) -> bool {
    mem::u32_at(m + dynamic::DEPTH_HACK).is_some_and(|v| f32::from_bits(v) > 0.0)
        || first_person_name(name)
}

/// Path backstop of [`first_person`]: `models/weapons/.../*player*` (e.g.
/// `models/weapons/gun_player/gun_emily.md6`) and first-person hands / arms.
pub fn first_person_name(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    (l.starts_with("models/weapons/") && l.contains("player"))
        || l.contains("firstperson")
        || l.contains("first_person")
        || (l.contains("player") && (l.contains("hand") || l.contains("arms")))
}

/// `idRenderLight` fields read by the GPU light fill (`sub_1402F8670`, O).
mod light {
    pub const FLAGS0: u64 = 0;
    pub const AXIS: u64 = 4;
    pub const ORIGIN: u64 = 40;
    pub const COLOR: u64 = 52;
    pub const TYPE: u64 = 92;
    pub const VIEW_MASK: u64 = 96;
    pub const FALLOFF: u64 = 124;
    pub const CUTOFF: u64 = 128;
    pub const DECAY: u64 = 136;
    pub const COS_INNER: u64 = 140;
    pub const COS_OUTER: u64 = 144;
    pub const SPEC_ATTN: u64 = 152;
}

/// Environment parameters in `renderView_t::m_environmentParms` (+3008), an `idParmBlock`:
/// vec4 values at +0, u16 parm ids at +16, count (low 24 bits) at +24 (O: `sub_140342640`
/// / `sub_140342E70`). Handles are registration objects whose parm id is at +72.
mod env {
    pub const RV_ENV_PARMS: u64 = 3008;
    pub const SUN_THETA: u64 = 0x227_2990;
    pub const SUN_PHI: u64 = 0x227_2970;
    pub const SUN_COLOR: u64 = 0x227_29B0;
    pub const SKY_ZENITH: u64 = 0x229_AFD8;
    pub const SKY_HORIZON: u64 = 0x229_AFF8;
    pub const IL_AMBIENT: u64 = 0x229_B018;
    pub const HANDLE_ID: u64 = 72;
    pub const MAX_PARMS: u32 = 4096;
}

/// The first view of a request list.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub map: String,
    /// `idRenderWorldLocal*` (changes with the map).
    pub world: u64,
    /// `renderView_t*` (inside the request list; valid during this frame only).
    pub rv: u64,
    pub org: [f32; 3],
    /// Rows: forward, left, up.
    pub axis: [[f32; 3]; 3],
    /// Horizontal / vertical field of view (radians).
    pub fov: [f32; 2],
    /// Near plane as `idRenderView_Setup` uses it.
    pub znear: f32,
}

impl View {
    pub fn camera(&self) -> void_rt::camera::Camera {
        void_rt::camera::Camera {
            org: self.org,
            axis: self.axis,
            fov_y: self.fov[1],
        }
    }
}

pub fn c_string(ptr: u64) -> String {
    if ptr == 0 {
        return String::new();
    }
    for n in [256usize, 128, 64, 32] {
        let mut buf = vec![0u8; n];
        if mem::read_into(ptr, &mut buf) {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(n);
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
    }
    String::new()
}

/// The render world of the first view (one pointer read per call).
pub fn world(requests: u64) -> Option<u64> {
    let list = mem::u64_at(requests + REQUESTS_PTR).filter(|&l| l != 0)?;
    if mem::u32_at(requests + REQUESTS_COUNT)? & COUNT_MASK == 0 {
        return None;
    }
    mem::u64_at(list + REQUEST_WORLD).filter(|&w| w != 0)
}

pub fn map_name(world: u64) -> String {
    c_string(mem::u64_at(world + WORLD_MAP_NAME).unwrap_or(0))
}

/// The first view of the request list.
pub fn view(requests: u64) -> Option<View> {
    let list = mem::u64_at(requests + REQUESTS_PTR).filter(|&l| l != 0)?;
    if mem::u32_at(requests + REQUESTS_COUNT)? & COUNT_MASK == 0 {
        return None;
    }
    let world = mem::u64_at(list + REQUEST_WORLD).filter(|&w| w != 0)?;
    let rv = list + REQUEST_RENDER_VIEW;
    let a = mem::f32s::<9>(rv + RV_VIEWAXIS)?;
    let mut cram = [0u8; 1];
    let cram = mem::read_into(rv + RV_CRAM_ZNEAR, &mut cram) && cram[0] != 0;
    let znear = mem::image_base()
        .and_then(|b| mem::f32s::<1>(b + R_ZNEAR_VALUE_RVA))
        .map_or(void_rt::snapshot::DEFAULT_ZNEAR, |v| v[0])
        * if cram { 0.25 } else { 1.0 };
    Some(View {
        map: map_name(world),
        world,
        rv,
        org: mem::f32s::<3>(rv + RV_VIEWORG)?,
        axis: [[a[0], a[1], a[2]], [a[3], a[4], a[5]], [a[6], a[7], a[8]]],
        fov: mem::f32s::<2>(rv + RV_FOV)?,
        znear,
    })
}

/// Counters of the entity walk's filters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntityCounts {
    pub models: u32,
    pub kept: u32,
    pub skinned: u32,
    pub not_drawn: u32,
    pub editor: u32,
    pub unreadable: u32,
    /// First-person view models (hands, weapons) left out.
    pub first_person: u32,
}

/// A render model by its slot in the world's model list (= `model+7192`, O:
/// `RegisterRenderModel` fills the first null slot and stores its index there) and its
/// address. Freed models keep their address until the next commit, then the slot is
/// reused: the pair identifies one model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ModelKey {
    pub index: u32,
    pub ptr: u64,
}

impl ModelKey {
    /// Unique id of the model for the dynamic layer (slot in the top 16 bits).
    pub fn id(self) -> u64 {
        u64::from(self.index) << 48 | (self.ptr & 0xFFFF_FFFF_FFFF)
    }
}

/// What the entity walk copied.
#[derive(Default)]
pub struct Walk {
    pub entities: Vec<Entity>,
    /// The model of every entry of `entities`.
    pub keys: Vec<ModelKey>,
    /// Every model pointer of the list at the walk (index = slot; 0 = empty).
    pub present: Vec<u64>,
    pub counts: EntityCounts,
}

/// Why [`read_entity`] skipped a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    Unnamed,
    Skinned,
    NotDrawn,
    Editor,
    Unreadable,
    FirstPerson,
}

/// The world's model pointer list (one read; at most `MAX_MODELS`).
pub fn model_pointers(world: u64) -> Option<Vec<u64>> {
    let ptr = mem::u64_at(world + WORLD_MODELS_PTR)?;
    let count = (mem::u32_at(world + WORLD_MODELS_COUNT)? & COUNT_MASK).min(MAX_MODELS);
    let mut raw = vec![0u8; count as usize * 8];
    if count > 0 && !mem::read_into(ptr, &mut raw) {
        return None;
    }
    Some(
        raw.as_chunks::<8>()
            .0
            .iter()
            .map(|p| u64::from_le_bytes(*p))
            .collect(),
    )
}

/// The resource name of model `m` (`.md6` for skinned models).
pub fn model_name(m: u64) -> String {
    c_string(mem::u64_at(m + MODEL_NAME).unwrap_or(0))
}

/// The live surface materials of model `m` (own surfaces, else the shared list).
pub fn model_materials(m: u64) -> Vec<String> {
    let material_name = |mat: u64| mem::u64_at(mat + 8).map(c_string).unwrap_or_default();
    let own = (
        mem::u64_at(m + MODEL_SURFACES_PTR).unwrap_or(0),
        mem::u32_at(m + MODEL_SURFACES_COUNT).unwrap_or(0) & COUNT_MASK,
    );
    let (surfaces, surface_count) = if own.1 > 0 {
        own
    } else {
        let shared = mem::u64_at(m + MODEL_SURFACES_SHARED).unwrap_or(0);
        (
            mem::u64_at(shared).unwrap_or(0),
            mem::u32_at(shared + 8).unwrap_or(0) & COUNT_MASK,
        )
    };
    (0..u64::from(surface_count.min(MAX_SURFACES)))
        .map(|k| {
            mem::u64_at(surfaces + SURFACE_STRIDE * k)
                .filter(|&mat| mat != 0)
                .map(material_name)
                .unwrap_or_default()
        })
        .collect()
}

/// Row-major 3x4 local-to-world transform from the parms origin / axis / scale (E023:
/// rows of `axis` are the local axes in world space).
pub fn parms_transform(o: [f32; 3], a: [f32; 9], s: [f32; 3]) -> [f32; 12] {
    let mut t = [0f32; 12];
    for r in 0..3 {
        for c in 0..3 {
            t[r * 4 + c] = a[c * 3 + r] * s[c];
        }
        t[r * 4 + 3] = o[r];
    }
    t
}

/// Model `m`'s transform (one read of the parms origin, axis and scale).
pub fn model_transform(m: u64) -> Option<[f32; 12]> {
    let v = mem::f32s::<15>(m + MODEL_PARMS)?;
    let t = parms_transform(
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]],
        [v[12], v[13], v[14]],
    );
    t.iter().all(|x| x.is_finite()).then_some(t)
}

/// A static, drawable model as a scene entity (the filters of the walk below).
pub fn read_entity(m: u64) -> Result<Entity, Skip> {
    let name = model_name(m);
    if name.is_empty() {
        return Err(Skip::Unnamed);
    }
    if first_person(m, &name) {
        return Err(Skip::FirstPerson);
    }
    if name.to_ascii_lowercase().ends_with(".md6") {
        return Err(Skip::Skinned);
    }
    let flags = mem::u32_at(m + MODEL_FLAGS).unwrap_or(0) & 0xFF;
    if !flags_may_be_drawn(flags) {
        return Err(Skip::NotDrawn);
    }
    let materials = model_materials(m);
    if !materials.first().is_none_or(|m| material_is_drawn(m)) {
        return Err(Skip::Editor);
    }
    let transform = model_transform(m).ok_or(Skip::Unreadable)?;
    Ok(Entity {
        model_source: name,
        transform,
        materials,
        runtime_materials: true,
    })
}

/// Static, drawable model instances of `world` with their runtime surface materials
/// (skinned `.md6`, unnamed, `m_mayBeDrawn`-clear and editor-material models skipped;
/// E044: about 47 ms for ~99k models on the game thread) and their model keys.
pub fn entities(world: u64) -> Walk {
    let mut w = Walk::default();
    let Some(present) = model_pointers(world) else {
        return w;
    };
    w.counts.models = present.len() as u32;
    for (index, &m) in present.iter().enumerate() {
        if m == 0 {
            continue;
        }
        match read_entity(m) {
            Ok(e) => {
                w.counts.kept += 1;
                w.entities.push(e);
                w.keys.push(ModelKey {
                    index: index as u32,
                    ptr: m,
                });
            }
            Err(Skip::Unnamed) => {}
            Err(Skip::Skinned) => w.counts.skinned += 1,
            Err(Skip::NotDrawn) => w.counts.not_drawn += 1,
            Err(Skip::Editor) => w.counts.editor += 1,
            Err(Skip::Unreadable) => w.counts.unreadable += 1,
            Err(Skip::FirstPerson) => w.counts.first_person += 1,
        }
    }
    w.present = present;
    w
}

/// `m_mayBeDrawn` of model `m` (false when unreadable).
pub fn model_drawable(m: u64) -> bool {
    mem::u32_at(m + MODEL_FLAGS).is_some_and(|f| flags_may_be_drawn(f & 0xFF))
}

/// This frame's dirty flags of the first `slots` model slots (one read); None when the
/// committed world is unreadable.
pub fn committed_flags(world: u64, slots: u32) -> Option<Vec<u32>> {
    let committed = mem::u64_at(world + dynamic::WORLD_COMMITTED).filter(|&c| c != 0)?;
    let n = slots.min(dynamic::MAX_SLOTS) as usize;
    let mut raw = vec![0u8; n * 4];
    if n > 0 && !mem::read_into(committed + dynamic::COMMITTED_MODEL_FLAGS, &mut raw) {
        return None;
    }
    Some(
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect(),
    )
}

/// `r_skipUpdateInView` (None when unreadable).
pub fn skip_update_in_view() -> Option<u32> {
    mem::image_base().and_then(|b| mem::u32_at(b + dynamic::SKIP_UPDATE_IN_VIEW_RVA))
}

/// Class of a model by its vtable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelClass {
    Skinned,
    /// `idRenderModelStatic` or `idRenderModel`.
    Static,
    Other,
}

pub fn model_class(m: u64) -> Option<ModelClass> {
    let base = mem::image_base()?;
    let vt = mem::u64_at(m)?.wrapping_sub(base);
    Some(match vt {
        dynamic::VTABLE_SKINNED => ModelClass::Skinned,
        dynamic::VTABLE_STATIC | dynamic::VTABLE_BASE => ModelClass::Static,
        _ => ModelClass::Other,
    })
}

/// Per-frame state of a skinned model (one read of `+6240..+8200`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinnedState {
    pub transform: [f32; 12],
    /// `m_needJointUploadToGpu`: > 0 when this frame computed new joints.
    pub new_joints: u8,
    /// `modelDepthHack` > 0: the first-person view model.
    pub first_person: bool,
    pub lod: u32,
    pub joints_ptr: u64,
    pub joints: u32,
}

pub fn skinned_state(m: u64) -> Option<SkinnedState> {
    const START: u64 = MODEL_PARMS;
    const END: u64 = dynamic::JOINTS_COUNT + 4;
    let mut b = vec![0u8; (END - START) as usize];
    if !mem::read_into(m + START, &mut b) {
        return None;
    }
    let at = |o: u64| (o - START) as usize;
    let f = |k: usize| f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap_or([0; 4]));
    let v: [f32; 15] = std::array::from_fn(f);
    let t = parms_transform(
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]],
        [v[12], v[13], v[14]],
    );
    let u32_at = |o: u64| u32::from_le_bytes(b[at(o)..at(o) + 4].try_into().unwrap_or([0; 4]));
    let u64_at = |o: u64| u64::from_le_bytes(b[at(o)..at(o) + 8].try_into().unwrap_or([0; 8]));
    t.iter().all(|x| x.is_finite()).then(|| SkinnedState {
        transform: t,
        new_joints: b[at(dynamic::NEED_JOINT_UPLOAD)],
        first_person: f32::from_le_bytes(
            b[at(dynamic::DEPTH_HACK)..at(dynamic::DEPTH_HACK) + 4]
                .try_into()
                .unwrap_or([0; 4]),
        ) > 0.0,
        lod: u32_at(dynamic::CURRENT_LOD),
        joints_ptr: u64_at(dynamic::JOINTS_PTR),
        joints: u32_at(dynamic::JOINTS_COUNT) & COUNT_MASK,
    })
}

/// The CPU joint palette (one read of `count` x 48 bytes; None above `MAX_JOINTS` or with
/// non-finite values).
pub fn joint_palette(ptr: u64, count: u32) -> Option<Vec<[f32; 12]>> {
    if ptr == 0 || count == 0 || count > dynamic::MAX_JOINTS {
        return None;
    }
    let mut raw = vec![0u8; count as usize * 48];
    if !mem::read_into(ptr, &mut raw) {
        return None;
    }
    let rows: Vec<[f32; 12]> = raw
        .as_chunks::<48>()
        .0
        .iter()
        .map(|r| {
            std::array::from_fn(|k| {
                f32::from_le_bytes(r[k * 4..k * 4 + 4].try_into().unwrap_or([0; 4]))
            })
        })
        .collect();
    rows.iter().flatten().all(|v| v.is_finite()).then_some(rows)
}

fn read_light(l: u64) -> Option<LightParms> {
    let mut b = [0u8; 2];
    if !mem::read_into(l + light::FLAGS0, &mut b) {
        return None;
    }
    let mut vm = [0u8; 1];
    if !mem::read_into(l + light::VIEW_MASK, &mut vm) {
        return None;
    }
    let f = |o: u64| mem::f32s::<1>(l + o).map(|v| v[0]);
    Some(LightParms {
        flags0: b[0],
        flags1: b[1],
        view_mask: vm[0],
        axis: mem::f32s::<9>(l + light::AXIS)?,
        origin: mem::f32s::<3>(l + light::ORIGIN)?,
        color: mem::f32s::<3>(l + light::COLOR)?,
        kind: mem::u32_at(l + light::TYPE)?,
        falloff: f(light::FALLOFF)?,
        cutoff: f(light::CUTOFF)?,
        decay: f(light::DECAY)?,
        cos_inner: f(light::COS_INNER)?,
        cos_outer: f(light::COS_OUTER)?,
        spec_attn: f(light::SPEC_ATTN)?,
    })
}

/// All lights of a render world (E035 `idRenderLight` fields).
pub fn lights(world: u64) -> Vec<LightParms> {
    let (Some(ptr), Some(count)) = (
        mem::u64_at(world + WORLD_LIGHTS_PTR),
        mem::u32_at(world + WORLD_LIGHTS_COUNT),
    ) else {
        return Vec::new();
    };
    (0..u64::from((count & COUNT_MASK).min(MAX_LIGHTS)))
        .filter_map(|i| mem::u64_at(ptr + 8 * i).filter(|&l| l != 0))
        .filter_map(read_light)
        .collect()
}

/// Looks up environment parameters of a view: one read for the id array, then one per
/// found value. `handles` are RVAs of the parameter handles.
fn env_values<const N: usize>(rv: u64, handles: [u64; N]) -> [Option<[f32; 4]>; N] {
    let mut out = [None; N];
    let Some(base) = mem::image_base() else {
        return out;
    };
    let block = rv + env::RV_ENV_PARMS;
    let (Some(values), Some(ids), Some(count)) = (
        mem::u64_at(block),
        mem::u64_at(block + 16),
        mem::u32_at(block + 24),
    ) else {
        return out;
    };
    let count = (count & COUNT_MASK).min(env::MAX_PARMS) as usize;
    let mut raw = vec![0u8; count * 2];
    if count == 0 || !mem::read_into(ids, &mut raw) {
        return out;
    }
    for (k, h) in handles.into_iter().enumerate() {
        let Some(id) = mem::u64_at(base + h)
            .filter(|&p| p != 0)
            .and_then(|p| mem::u32_at(p + env::HANDLE_ID))
        else {
            continue;
        };
        // The engine keeps the last match (it overwrites while scanning).
        let found = raw
            .as_chunks::<2>()
            .0
            .iter()
            .rposition(|b| u32::from(u16::from_le_bytes(*b)) == id);
        out[k] = found.and_then(|i| mem::f32s::<4>(values + 16 * i as u64));
    }
    out
}

/// Sun angles and colour of the view (None when a parameter is absent from its block).
pub fn sun(rv: u64) -> Option<Sun> {
    let [t, p, c] = env_values(rv, [env::SUN_THETA, env::SUN_PHI, env::SUN_COLOR]);
    let (t, p, c) = (t?, p?, c?);
    Some(Sun::from_angles(t[0], p[0], [c[0], c[1], c[2]]))
}

/// Indirect-lighting sky colours and ambient of the view.
pub fn env(rv: u64) -> Option<Env> {
    let [z, h, a] = env_values(rv, [env::SKY_ZENITH, env::SKY_HORIZON, env::IL_AMBIENT]);
    let (z, h, a) = (z?, h?, a?);
    Some(Env {
        sky_zenith: [z[0], z[1], z[2]],
        sky_horizon: [h[0], h[1], h[2]],
        ambient: [a[0], a[1], a[2]],
    })
}

/// Sun and sky of a view in engine absolute units (see [`crate::rt::Lighting`]).
pub fn lighting(rv: u64) -> crate::rt::Lighting {
    let sun = sun(rv);
    let env = env(rv);
    crate::rt::Lighting {
        sun_dir: sun.map_or([0.0, 0.0, 1.0], |s| s.dir),
        sun_irradiance: sun.map_or([0.0; 3], |s| s.color.map(|c| c * SUN_COLOR_SCALE)),
        sky_zenith: env.map_or([0.0; 3], |e| e.sky_zenith),
        sky_horizon: env.map_or([0.0; 3], |e| e.sky_horizon),
        ambient: env.map_or([0.0; 3], |e| e.ambient),
        valid: sun.is_some() && env.is_some(),
    }
}
