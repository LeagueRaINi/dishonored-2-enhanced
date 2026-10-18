//! In-process API for the game's DLL (docs/raytracing.md sections 10.5 and 11): the photo
//! path tracer and the shared scene for real-time ray tracing, driven from a host thread
//! on a host-created D3D12 device. The offline `photo` example uses exactly this code path.
//!
//! Flow (each step returns `Err(String)`; the host then leaves photo mode cleanly):
//!
//! 1. [`Gpu::from_existing`] on the host's device and its own queue (`Gpu` is `Send`; the
//!    DXGI adapter is found by the device's LUID for the video memory budget).
//! 2. [`load_scene`] (CPU only, no GPU objects; [`CpuScene`] is `Send`) on a worker thread
//!    from a [`SceneSource`] the host fills from game memory.
//! 3. [`upload`] -> [`GpuScene`] (BLAS, full TLAS, tables, textures, photo kernels from the
//!    build-time DXIL). Video memory: [`estimate_memory`] first; the texture cap is lowered
//!    (halving down to `min_texture_size`) until the peak fits budget - usage - headroom
//!    ([`Gpu::vram_available`]), else the upload is refused. Blocking submits, each short:
//!    BLAS batches, compaction copies and texture batches follow the measured GPU time of
//!    the previous submit towards `accel::SUBMIT_TARGET_MS` (16 ms); [`Gpu::set_pause`]
//!    sleeps between them. Phases are logged through [`Gpu::set_log`] and kept in
//!    [`GpuScene::report`]. Share the scene as `Arc<GpuScene>`.
//! 4. [`PhotoSession::new`], then [`PhotoSession::step`] once per host frame (non-blocking:
//!    at most one submit per call, sized to the budget), [`PhotoSession::copy_out`] into the
//!    host's shared textures, [`PhotoSession::read_hdr`] / [`PhotoSession::write_png`] /
//!    [`PhotoSession::write_hdr`] to save.
//!
//! # Inputs the host reads from game memory ([`SceneSource`])
//!
//! - `entities` ([`Entity`]): static render models of the world model list (world +132408 /
//!   +132416): `model_source` (resource name of the `.bmodel`), `transform` (row-major 3x4
//!   local-to-world from the parms origin/axis/scale), `materials` (the live surface
//!   material names, `runtime_materials = true`); skip models with `m_mayBeDrawn` clear
//!   ([`crate::snapshot::flags_may_be_drawn`]) and skinned models (checkpoint 9282529
//!   `snapshot.rs`, F3 `models.tsv` / `surfaces.tsv` columns).
//! - `lights` ([`LightParms`]): the world's render lights (world +516464 / +516472, the
//!   232-byte struct of E035 that `sub_1402F8670` copies into `arkTiledForwardLight_t`;
//!   F3 `light_parms.tsv` columns): `flags0` (bit 0 = enabled), `flags1`, `view_mask`
//!   (bits 0-1), `axis` (three rows at +4/+16/+28 of the light), `origin`, `color` (as
//!   stored, absolute units), `kind` (0 point, 1 half-point, 2 spot, 3 area = skipped,
//!   4 parallel = skipped, the sun comes from `sun`), `falloff`, `cutoff`, `decay`,
//!   `cos_inner`, `cos_outer`, `spec_attn`. [`LightParms::to_gpu`] applies the engine's
//!   filter.
//! - `sun` ([`Sun::from_angles`]): env `sunTheta` / `sunPhi` (degrees) and `sunColor`
//!   (relative; irradiance = `sunColor` x [`SUN_COLOR_SCALE`]). `None` disables the sun.
//! - `env` ([`Env`]): `env/indirectLighting/bounceFactor/skyZenith`, `.../skyHorizon`,
//!   `ambient`. `None` disables the sky term.
//! - the camera ([`Camera`]): view origin, axis rows forward/left/up, vertical fov (radians),
//!   and `r_znear` ([`PhotoSettings::znear`]).
//!
//! # Units
//!
//! [`PhotoSession::hdr_buffer`] holds the mean radiance in the engine's absolute units (the
//! units the offline renderer validated against the game, section 10.3). The engine's
//! stored `hdrBuffer` holds absolute / `EngineLuminanceFactor` (docs/renderer-replacement.md;
//! `EngineLuminanceFactor` = field 0 of the `autoExposure_t` element selected by
//! `exposureIndex`, docs/fsr4-integration.md): multiply by [`stored_hdr_scale`]`(ELF)`. In
//! bright daylight ELF clamps at 0.2792 -> x [`DAYLIGHT_STORED_SCALE`] (3.58, RX010 dumps).
//! With `useCompressedHdrBuffers` the shaders also divide by `view_white_level` (H; the
//! RX010 cvar state measured 3.58 without it).
//!
//! # Root layout of the photo kernels (`photo.hlsl` `RS`, [`LAYOUT`])
//!
//! Other kernels may bind the same scene buffers with this layout or their own:
//!
//! | slot | content | element |
//! | --- | --- | --- |
//! | b0 | 8 root constants: width, height, frame, step, flags (`TF_*`), y0, 0, 0 | |
//! | t0 | TLAS ([`GpuScene::full_tlas`] or a cut-off [`Tlas`]) | `InstanceID` = first global geometry of the mesh, mask 1 world / 2 entity |
//! | t1 | geometry table ([`SceneGpu::geometry`], `geom_count`) | uint2 (first primitive, material) per global geometry; primitive = `GeomInfo[InstanceID + GeometryIndex].x + PrimitiveIndex` |
//! | t2 | triangle records ([`SceneGpu::records`], `prim_count`) | 64 B: f32 LOD bias + 3 x 20-byte vertex blocks (uv0 f16x2, uv1 f16x2, normal snorm8x4, tangent snorm8x4, colour unorm8x4) |
//! | t3 | materials ([`SceneGpu::materials`], `mat_count`) | 448 B `Mat` (`scene::gpu_material`) |
//! | t4 | lights ([`SceneGpu::lights`], `light_count`, at least one element) | 80 B (`LightParms::to_gpu`) |
//! | t5 / t6 | light grid cells / indices ([`SceneGpu::cells`], [`SceneGpu::light_indices`]) | uint2 (first, count) per XY cell / uint |
//! | t7 | `Params` ([`crate::photo::params`]) | 48 words |
//! | u0..u2 | accumulators (radiance + lum^2, albedo + sun visibility, normal + view depth) | float4 per pixel |
//! | u3 / u4 | denoiser ping-pong; resolve output (hdr) / depth output | float4, pitched |
//! | space1 t0.. | bindless textures ([`GpuScene::textures`]: heap + `gpu_start`, `count` slots, slot 0 white) | Texture2D |
//! | s0 | static linear wrap sampler | |
//!
//! With [`SceneOptions::spike_primitives`] the scene also holds the spike's FC6 primitive
//! buffer ([`GpuScene::primitives`], 24 B per triangle, same primitive index) and the u32
//! offset table ([`GpuScene::primitive_offsets`]) used by `rt.hlsl`.
//!
//! # Precompiled shaders
//!
//! `build.rs` compiles every kernel of `photo.hlsl` and `rt.hlsl` with `dxc.exe` (env
//! `VOID_DXC_DIR`, else the newest Windows SDK) into DXIL embedded in the library:
//! [`dxil`]. Other crates can copy `build.rs` (change `SOURCES`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_TYPELESS,
};

use crate::accel::{self, BlasSet};
pub use crate::accel::{BlasStats, Tlas, TlasStats};
pub use crate::camera::Camera;
pub use crate::character::{CharacterCpu, Md6Index, PaletteMode, PaletteVerdict, Row34};
pub use crate::dynamic::{
    DynamicAssets, DynamicReserve, DynamicStats, DynamicTables, Dynamics, InstanceSet, ReserveBase,
    SceneTlas, UpdateStats,
};
use crate::gpu::mib;
pub use crate::gpu::{Buffer, Gpu, Programs, SubmitGuard, SubmitStats, VideoMemory, VramPolicy};
use crate::image;
pub use crate::lights::{Env, LIGHT_WORDS, LightGrid, LightParms, SUN_COLOR_SCALE};
use crate::photo::{self, mean};
pub use crate::photo::{
    Accum, LAYOUT, Lighting, MAX_DISPATCHES_PER_SUBMIT, PARAM_WORDS, PathSettings, SceneGpu,
    TF_NO_ALPHA, TF_NO_LIGHTS, TF_NO_SKY, TF_NO_SUN, TF_NO_TEXTURES, depth_pitch, hdr_pitch,
};
use crate::scene::{self, EntitySource};
pub use crate::scene::{Instance, LoadOptions, NO_INSTANCE, Origin, Scene, SceneStats};
pub use crate::snapshot::{DEFAULT_ZNEAR, Entity, Sun};
pub use crate::textures::TextureSet;

include!(concat!(env!("OUT_DIR"), "/dxil.rs"));

/// Precompiled DXIL (cs_6_5, signed) of a kernel of `photo.hlsl` / `rt.hlsl`.
pub fn dxil(entry: &str) -> Option<&'static [u8]> {
    DXIL.iter().find(|(n, _)| *n == entry).map(|(_, b)| *b)
}

/// Names of the precompiled kernels.
pub fn dxil_entries() -> impl Iterator<Item = &'static str> {
    DXIL.iter().map(|(n, _)| *n)
}

/// Stored game HDR per absolute unit in bright daylight (1 / 0.2792, RX010).
pub const DAYLIGHT_STORED_SCALE: f32 = 3.58;

/// Factor from [`PhotoSession::hdr_buffer`] units to the engine's stored `hdrBuffer`
/// units: 1 / `EngineLuminanceFactor` (see the module doc).
pub fn stored_hdr_scale(engine_luminance_factor: f32) -> f32 {
    1.0 / engine_luminance_factor
}

/// Default tile: 1/8 of a 3440x1440 frame (rows of the image width, a multiple of 4).
pub const DEFAULT_TILE_PIXELS: u32 = 3440 * 1440 / 8;

/// Inputs from the game (see the module doc for the fields the host reads).
#[derive(Clone, Debug, Default)]
pub struct SceneSource {
    /// Game installation root (holds `base/`); only the archives are read.
    pub game_dir: PathBuf,
    /// Engine map name, e.g. `maps/campaign/edge_of_the_world/eotw_p_highchaos`.
    pub map: String,
    pub entities: Vec<Entity>,
    pub lights: Vec<LightParms>,
    pub sun: Option<Sun>,
    pub env: Option<Env>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneOptions {
    /// Largest texture edge uploaded (texels; larger mips skipped). VRAM: 890 MiB at 512.
    /// [`upload`] lowers it (halving, down to `min_texture_size`) when the scene would not
    /// fit the video memory budget.
    pub texture_size: u32,
    /// Lowest texture cap [`upload`] falls back to; below it the upload is refused.
    pub min_texture_size: u32,
    /// Video memory the host needs after the upload (photo buffers, live TLASes): kept free
    /// on top of the [`Gpu::vram_policy`] headroom when choosing the texture cap.
    pub vram_reserve_bytes: u64,
    /// Trace arkDecal mesh decals.
    pub decals: bool,
    /// Also build the FC6 24-byte primitive buffer for real-time kernels (`rt.hlsl`).
    pub spike_primitives: bool,
    /// Use the map's `.entities` placements instead of `SceneSource::entities` (offline
    /// runs without a snapshot).
    pub map_entities: bool,
    /// Light grid cell edge (metres) and cells per axis at most.
    pub light_cell: f32,
    pub max_light_cells: u32,
    /// Staging data per upload submit at most (tables and textures; texture batches also
    /// follow the measured GPU time).
    pub upload_batch_bytes: u64,
    /// BLAS build scratch arena (batches also end at a triangle count that follows the
    /// measured GPU time, `accel::build_blas`).
    pub blas_scratch_bytes: u64,
    /// Full TLAS limited to the N instances nearest a point (WARP smoke runs).
    pub tlas_limit: Option<(usize, [f32; 3])>,
    /// Room for dynamic objects ([`crate::dynamic`]): reserved table entries and texture
    /// slots after the static ones; records and FC6 primitives become UAV-capable. None:
    /// a static scene (no reserve).
    pub dynamic: Option<DynamicReserve>,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self {
            texture_size: 512,
            min_texture_size: 256,
            vram_reserve_bytes: 0,
            decals: true,
            spike_primitives: false,
            map_entities: false,
            light_cell: 4.0,
            max_light_cells: 512,
            upload_batch_bytes: 64 << 20,
            blas_scratch_bytes: 256 << 20,
            tlas_limit: None,
            dynamic: None,
        }
    }
}

/// Decoded scene (CPU only, `Send`): geometry, records, materials, textures, lights.
pub struct CpuScene {
    pub scene: Scene,
    /// GPU light records of the lights the engine would draw ([`LightParms::to_gpu`]).
    pub lights: Vec<[u32; LIGHT_WORDS]>,
    pub grid: LightGrid,
    /// Light rows given (before filtering).
    pub light_rows: usize,
    pub sun: Option<Sun>,
    pub env: Option<Env>,
    pub map: String,
    pub opts: SceneOptions,
    /// Instance of every `SceneSource::entities` entry (`scene::NO_INSTANCE`: not traced).
    pub entity_instances: Vec<u32>,
    /// Static table counts (the reserve starts there; the tables are padded with it).
    pub dynamic_base: ReserveBase,
}

/// Decodes the static scene of `src.map` from the archives plus the given entities and
/// lights (CPU only; runs on any thread; seconds for a full map).
pub fn load_scene(src: &SceneSource, opts: &SceneOptions) -> Result<CpuScene, String> {
    let entities = if opts.map_entities {
        EntitySource::File
    } else if src.entities.is_empty() {
        EntitySource::None
    } else {
        EntitySource::Snapshot(src.entities.clone())
    };
    let (scene, entity_instances) = scene::load_mapped(
        &src.game_dir,
        &src.map,
        entities,
        LoadOptions {
            photo_texture_size: Some(opts.texture_size),
            decals: opts.decals,
            spike_primitives: opts.spike_primitives,
        },
    )?;
    let mut cpu = CpuScene::from_scene(scene, src, opts);
    cpu.entity_instances = entity_instances;
    Ok(cpu)
}

impl CpuScene {
    /// A decoded `scene` (photo records, optional FC6 primitives) with `src`'s lights, sun
    /// and sky; with [`SceneOptions::dynamic`] the tables are padded with the reserve.
    /// `entity_instances` is left empty.
    pub fn from_scene(mut scene: Scene, src: &SceneSource, opts: &SceneOptions) -> Self {
        let dynamic_base = ReserveBase {
            geometries: scene.primitive_offsets.len() as u32,
            triangles: scene.records.len() as u32,
            materials: scene.gpu_materials.len() as u32,
        };
        if let Some(r) = opts.dynamic {
            // Zero entries: referenced by no instance until a dynamic mesh is written there.
            let g = scene.primitive_offsets.len() + r.geometries as usize;
            scene.primitive_offsets.resize(g, 0);
            scene.geometry_materials.resize(g, 0);
            let t = scene.records.len() + r.triangles as usize;
            scene.records.resize(t, [0; scene::RECORD_WORDS]);
            if opts.spike_primitives {
                scene.primitives.resize(t, [0; 6]);
            }
            let m = scene.gpu_materials.len() + r.materials as usize;
            scene.gpu_materials.resize(m, [0; scene::MATERIAL_WORDS]);
        }
        let lights: Vec<[u32; LIGHT_WORDS]> =
            src.lights.iter().filter_map(LightParms::to_gpu).collect();
        let grid = LightGrid::build(&lights, opts.light_cell, opts.max_light_cells);
        Self {
            scene,
            lights,
            grid,
            light_rows: src.lights.len(),
            sun: src.sun,
            env: src.env,
            map: src.map.clone(),
            opts: *opts,
            entity_instances: Vec::new(),
            dynamic_base,
        }
    }
}

/// Per-mesh data kept after the upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshInfo {
    /// First global geometry (TLAS `InstanceID`, base into the geometry table).
    pub first_geometry: u32,
    pub geometries: u32,
    pub triangles: u64,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The scene on the GPU: one compacted BLAS per mesh, the full TLAS, the photo tables and
/// textures, optional spike primitives, and the photo kernels. Immutable after
/// [`upload`] (the dynamic reserve's entries are written by [`Dynamics`] in place);
/// shared as `Arc<GpuScene>` (`Send + Sync`). All buffers belong to the [`Gpu`]'s device
/// it was uploaded with.
pub struct GpuScene {
    generation: u64,
    map: String,
    stats: SceneStats,
    upload_seconds: f64,
    meshes: Vec<MeshInfo>,
    instances: Vec<Instance>,
    instance_aabbs: Vec<([f32; 3], [f32; 3])>,
    entity_instances: Vec<u32>,
    blas: BlasSet,
    full_tlas: Tlas,
    tables: SceneGpu,
    primitives: Option<Buffer>,
    primitive_offsets: Option<Buffer>,
    programs: Programs,
    sun: Option<Sun>,
    env: Option<Env>,
    light_rows: usize,
    report: UploadReport,
    dynamic: DynamicTables,
}

const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send::<CpuScene>();
    send_sync::<GpuScene>();
    send::<PhotoSession>();
};

fn aabb_dist2(lo: [f32; 3], hi: [f32; 3], p: [f32; 3]) -> f32 {
    (0..3)
        .map(|k| {
            let c = p[k].clamp(lo[k], hi[k]) - p[k];
            c * c
        })
        .sum()
}

/// Video memory a scene upload needs, from prebuild queries and the decoded data
/// ([`estimate_memory`]; no GPU work).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoryEstimate {
    /// Positions + indices (BLAS build inputs, freed after the build).
    pub geometry: u64,
    /// BLAS results before compaction (also the bound of the compacted set) and the
    /// scratch arena.
    pub blas_uncompacted: u64,
    pub blas_scratch: u64,
    /// Full TLAS: result, scratch, instance descriptors.
    pub tlas: u64,
    /// Geometry table, records, materials, lights, light grid.
    pub tables: u64,
    /// FC6 primitives and their offset table (live ray tracing).
    pub primitives: u64,
    /// Staging and allocation overhead of the table / texture phase: 4 upload batches (O:
    /// the local usage of that phase exceeded the device sizes by about 250 MiB at caps
    /// 512 and 2048 with 64 MiB batches on an RX 7900 XTX; presumably upload heaps placed
    /// in local memory with resizable BAR).
    pub staging: u64,
    /// Per texture cap, largest first: (cap, device bytes, staging data bytes).
    pub textures: Vec<(u32, u64, u64)>,
}

impl MemoryEstimate {
    /// Peak of the BLAS phase: inputs + uncompacted results + the larger of the scratch
    /// arena (build) and the compacted copy (compaction, after the scratch is freed).
    pub fn blas_peak(&self) -> u64 {
        self.geometry + self.blas_uncompacted + self.blas_scratch.max(self.blas_uncompacted)
    }

    /// Device bytes of the textures at `cap` (0 for a cap not estimated).
    pub fn texture_bytes(&self, cap: u32) -> u64 {
        self.textures.iter().find(|t| t.0 == cap).map_or(0, |t| t.1)
    }

    /// Resident after the upload with textures at `cap` (BLAS at its uncompacted bound).
    pub fn resident(&self, cap: u32) -> u64 {
        self.blas_uncompacted + self.tlas + self.tables + self.primitives + self.texture_bytes(cap)
    }

    /// Most allocated at once during the upload.
    pub fn peak(&self, cap: u32) -> u64 {
        self.blas_peak().max(self.resident(cap) + self.staging)
    }
}

/// Texture caps tried for `requested`: it, then halvings while at least `min`.
pub fn texture_caps(requested: u32, min: u32) -> Vec<u32> {
    let mut v = vec![requested];
    let mut c = requested;
    while c / 2 >= min.max(1) {
        c /= 2;
        v.push(c);
    }
    v
}

/// Video memory [`upload`] will need for `cpu` (prebuild and allocation-size queries on
/// `gpu`'s device; no GPU work), with the textures at every cap of [`texture_caps`].
pub fn estimate_memory(gpu: &Gpu, cpu: &CpuScene) -> MemoryEstimate {
    let s = &cpu.scene;
    let o = &cpu.opts;
    let size = |b: usize| b as u64;
    let (blas_uncompacted, scratch_max, scratch_sum) = accel::blas_prebuild(gpu, s);
    let textures = texture_caps(o.texture_size, o.min_texture_size)
        .into_iter()
        .map(|cap| {
            let (dev, data) = s
                .textures
                .iter()
                .map(|t| crate::textures::texture_bytes(gpu, t, cap))
                .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            (cap, dev, data)
        })
        .collect();
    MemoryEstimate {
        geometry: size(size_of_val(s.positions.as_slice()) + size_of_val(s.indices.as_slice())),
        blas_uncompacted,
        blas_scratch: o.blas_scratch_bytes.min(scratch_sum).max(scratch_max),
        tlas: Tlas::bytes(gpu, s.instances.len()),
        tables: size(
            s.primitive_offsets.len() * 8
                + size_of_val(s.records.as_slice())
                + size_of_val(s.gpu_materials.as_slice())
                + cpu.lights.len().max(1) * LIGHT_WORDS * 4
                + size_of_val(cpu.grid.cells.as_slice())
                + size_of_val(cpu.grid.indices.as_slice()).max(4),
        ),
        primitives: if o.spike_primitives {
            size(size_of_val(s.primitives.as_slice()) + size_of_val(s.primitive_offsets.as_slice()))
        } else {
            0
        },
        staging: 4 * o.upload_batch_bytes,
        textures,
    }
}

/// One phase of [`upload`]: GPU time of its submits and the process's video memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UploadPhase {
    pub name: &'static str,
    /// Estimated bytes the phase allocates.
    pub estimate: u64,
    pub wall_ms: f64,
    pub submits: usize,
    pub gpu_ms: f64,
    pub max_submit_ms: f64,
    /// DXGI local usage before / after and the budget after (0 when unknown).
    pub usage_before: u64,
    pub usage_after: u64,
    pub budget: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UploadReport {
    pub estimate: MemoryEstimate,
    pub requested_texture_size: u32,
    /// Texture cap uploaded (lower than requested when the budget required it).
    pub texture_size: u32,
    pub phases: Vec<UploadPhase>,
    /// Longest single submit of the upload (GPU ms).
    pub max_submit_ms: f64,
}

/// Runs one upload phase after checking its estimate against the free budget; logs it
/// through [`Gpu::log`] and records it in `report`.
fn phase<T>(
    gpu: &mut Gpu,
    report: &mut UploadReport,
    name: &'static str,
    need: u64,
    f: impl FnOnce(&mut Gpu) -> Result<T, String>,
) -> Result<T, String> {
    gpu.check_vram(need, name)?;
    let before = gpu.video_memory().unwrap_or_default();
    gpu.take_submit_stats();
    let t = Instant::now();
    let r = f(gpu);
    let st = gpu.take_submit_stats();
    let after = gpu.video_memory().unwrap_or_default();
    let p = UploadPhase {
        name,
        estimate: need,
        wall_ms: t.elapsed().as_secs_f64() * 1e3,
        submits: st.count,
        gpu_ms: st.total_ms,
        max_submit_ms: st.max_ms,
        usage_before: before.usage,
        usage_after: after.usage,
        budget: after.budget,
    };
    gpu.log(&format!(
        "rt upload: {name} {} in {:.0} ms: {} submits, GPU {:.1} ms (longest {:.1} ms); VRAM in use {:.0} -> {:.0} MiB of budget {:.0} MiB (estimate +{:.0} MiB)",
        if r.is_ok() { "done" } else { "FAILED" },
        p.wall_ms,
        p.submits,
        p.gpu_ms,
        p.max_submit_ms,
        mib(p.usage_before),
        mib(p.usage_after),
        mib(p.budget),
        mib(need)
    ));
    report.max_submit_ms = report.max_submit_ms.max(p.max_submit_ms);
    report.phases.push(p);
    r
}

/// Uploads a decoded scene (consumes the CPU data). Blocking, short submits (each paced by
/// its measured GPU time, see `accel::build_blas` and [`TextureSet::upload`]).
///
/// Video memory: [`estimate_memory`] before anything is allocated; the largest texture cap
/// of [`texture_caps`] whose peak plus `vram_reserve_bytes` fits [`Gpu::vram_available`]
/// (budget - usage - headroom) is used, and re-checked before the textures (the game's
/// usage moves); if not even `min_texture_size` fits, the upload is refused. Every phase
/// checks its estimate against the free budget first and logs the usage before / after
/// through [`Gpu::log`]. The cap used is in [`GpuScene::report`]. On an error the open
/// command list is discarded.
pub fn upload(gpu: &mut Gpu, cpu: CpuScene) -> Result<GpuScene, String> {
    let r = upload_phases(gpu, cpu);
    if r.is_err() {
        let _ = gpu.discard();
    }
    r
}

/// The texture cap (from `caps`, largest first) whose `need` fits `avail`.
fn fitting_cap(caps: &[(u32, u64, u64)], avail: u64, need: impl Fn(u32) -> u64) -> Option<u32> {
    caps.iter().map(|t| t.0).find(|&c| need(c) <= avail)
}

#[allow(clippy::too_many_lines)]
fn upload_phases(gpu: &mut Gpu, cpu: CpuScene) -> Result<GpuScene, String> {
    let t0 = Instant::now();
    if cpu.scene.instances.is_empty() {
        return Err(format!("{}: no traced instances", cpu.map));
    }
    if cpu.opts.spike_primitives && cpu.scene.primitives.len() != cpu.scene.records.len() {
        return Err(format!(
            "spike primitives {} != photo records {}",
            cpu.scene.primitives.len(),
            cpu.scene.records.len()
        ));
    }
    let est = estimate_memory(gpu, &cpu);
    let requested = cpu.opts.texture_size;
    let reserve = cpu.opts.vram_reserve_bytes;
    let caps_text = est
        .textures
        .iter()
        .map(|t| format!("<= {}: {:.0} MiB", t.0, mib(t.1)))
        .collect::<Vec<_>>()
        .join(", ");
    gpu.log(&format!(
        "rt upload: '{}' estimate: geometry {:.0} MiB, BLAS {:.0} MiB (uncompacted) + scratch {:.0} MiB, TLAS {:.0} MiB, tables {:.0} MiB, primitives {:.0} MiB, staging {:.0} MiB, textures {caps_text}; peak {:.0} MiB at <= {requested}, host reserve {:.0} MiB; {}",
        cpu.map,
        mib(est.geometry),
        mib(est.blas_uncompacted),
        mib(est.blas_scratch),
        mib(est.tlas),
        mib(est.tables),
        mib(est.primitives),
        mib(est.staging),
        mib(est.peak(requested)),
        mib(reserve),
        gpu.vram_text()
    ));
    let min_cap = est.textures.last().map_or(requested, |t| t.0);
    let mut cap = match gpu.vram_available() {
        None => requested,
        Some(avail) => fitting_cap(&est.textures, avail, |c| est.peak(c) + reserve).ok_or_else(
            || {
                format!(
                    "not enough video memory for the scene: {:.0} MiB needed with textures <= {min_cap} (+{:.0} MiB reserve), {:.0} MiB available ({})",
                    mib(est.peak(min_cap)),
                    mib(reserve),
                    mib(avail),
                    gpu.vram_text()
                )
            },
        )?,
    };
    let CpuScene {
        mut scene,
        lights,
        grid,
        light_rows,
        sun,
        env,
        map,
        opts,
        entity_instances,
        dynamic_base,
    } = cpu;
    let reduce = |gpu: &mut Gpu, scene: &mut Scene, cap: u32, why: &str| {
        for t in &mut scene.textures {
            crate::textures::cap_mips(t, cap);
        }
        gpu.log(&format!(
            "rt upload: textures reduced to {cap} (requested {requested}) {why}"
        ));
    };
    if cap < requested {
        reduce(gpu, &mut scene, cap, "to fit the video memory budget");
    }
    let mut report = UploadReport {
        estimate: est.clone(),
        requested_texture_size: requested,
        texture_size: cap,
        ..UploadReport::default()
    };
    let batch = opts.upload_batch_bytes;
    let (pos, idx) = phase(gpu, &mut report, "geometry", est.geometry, |g| {
        Ok((
            g.upload_chunked(&scene.positions, false, batch)?,
            g.upload_chunked(&scene.indices, false, batch)?,
        ))
    })?;
    let blas = phase(
        gpu,
        &mut report,
        "BLAS",
        est.blas_peak() - est.geometry,
        |g| accel::build_blas(g, &scene, &pos, &idx, opts.blas_scratch_bytes),
    )?;
    drop((pos, idx));
    let meshes: Vec<MeshInfo> = scene
        .meshes
        .iter()
        .map(|m| MeshInfo {
            first_geometry: m.first_geometry,
            geometries: m.geometries.len() as u32,
            triangles: m.triangles(),
            aabb_min: m.aabb_min,
            aabb_max: m.aabb_max,
        })
        .collect();
    let instance_aabbs: Vec<([f32; 3], [f32; 3])> = scene
        .instances
        .iter()
        .map(|i| i.world_aabb(&scene.meshes[i.mesh as usize]))
        .collect();
    let keep: Vec<bool> = match opts.tlas_limit {
        Some((m, p)) if m < scene.instances.len() => {
            let mut order: Vec<(f32, usize)> = instance_aabbs
                .iter()
                .enumerate()
                .map(|(i, &(lo, hi))| (aabb_dist2(lo, hi, p), i))
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut v = vec![false; scene.instances.len()];
            for &(_, i) in &order[..m] {
                v[i] = true;
            }
            v
        }
        _ => vec![true; scene.instances.len()],
    };
    let descs = accel::instance_descs(&scene, &blas, true, |i| keep[i]);
    let full_tlas = phase(gpu, &mut report, "TLAS", est.tlas, |g| {
        Tlas::build(g, &descs, descs.len())
    })?;
    let (primitives, primitive_offsets) = if opts.spike_primitives {
        phase(gpu, &mut report, "primitives", est.primitives, |g| {
            Ok((
                // Dynamic characters rewrite their primitives' normals in place.
                Some(g.upload_chunked(&scene.primitives, opts.dynamic.is_some(), batch)?),
                Some(g.upload_chunked(&scene.primitive_offsets, false, batch)?),
            ))
        })?
    } else {
        (None, None)
    };
    // Textures last, against the budget of now (the game's usage moves while we upload).
    let rest = |c: u32| est.tables + est.texture_bytes(c) + est.staging + reserve;
    if let Some(avail) = gpu.vram_available() {
        let fits = fitting_cap(&est.textures, avail, rest).filter(|&c| c <= cap);
        match fits {
            Some(c) if c < cap => {
                cap = c;
                reduce(
                    gpu,
                    &mut scene,
                    cap,
                    "(budget re-checked before the textures)",
                );
                report.texture_size = cap;
            }
            Some(_) => {}
            None => {
                return Err(format!(
                    "not enough video memory for the textures: {:.0} MiB needed at <= {min_cap}, {:.0} MiB available ({})",
                    mib(rest(min_cap)),
                    mib(avail),
                    gpu.vram_text()
                ));
            }
        }
    }
    let tables = phase(
        gpu,
        &mut report,
        "tables + textures",
        est.tables + est.texture_bytes(cap) + est.staging,
        |g| {
            let t = SceneGpu::upload(
                g,
                &scene,
                &lights,
                grid,
                batch,
                opts.dynamic.map(|r| r.textures),
            )?;
            t.validate()?;
            Ok(t)
        },
    )?;
    let programs = Programs::from_dxil(gpu, photo::LAYOUT, &photo::entries(), dxil)?;
    let dynamic = DynamicTables {
        reserve: opts.dynamic,
        base: dynamic_base,
        texture_slots: if opts.dynamic.is_some() {
            scene
                .textures
                .iter()
                .enumerate()
                .map(|(i, t)| (t.name.clone(), i as u32 + 1))
                .collect()
        } else {
            Default::default()
        },
        load: LoadOptions {
            photo_texture_size: Some(cap),
            decals: opts.decals,
            spike_primitives: opts.spike_primitives,
        },
    };
    Ok(GpuScene {
        generation: GENERATION.fetch_add(1, Ordering::Relaxed) + 1,
        map,
        stats: scene.stats,
        upload_seconds: t0.elapsed().as_secs_f64(),
        meshes,
        instances: scene.instances,
        instance_aabbs,
        entity_instances,
        blas,
        full_tlas,
        tables,
        primitives,
        primitive_offsets,
        programs,
        sun,
        env,
        light_rows,
        report,
        dynamic,
    })
}

impl GpuScene {
    /// Unique per [`upload`] in this process (starts at 1): consumers compare it to know
    /// whether cached TLASes / bindings refer to the current scene.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn map(&self) -> &str {
        &self.map
    }

    pub fn stats(&self) -> &SceneStats {
        &self.stats
    }

    /// Wall time of [`upload`] (s).
    pub fn upload_seconds(&self) -> f64 {
        self.upload_seconds
    }

    /// Memory estimate, texture cap used, per-phase GPU time and video memory of [`upload`].
    pub fn report(&self) -> &UploadReport {
        &self.report
    }

    pub fn meshes(&self) -> &[MeshInfo] {
        &self.meshes
    }

    /// BLAS GPU address of `mesh`.
    pub fn blas_va(&self, mesh: usize) -> Option<u64> {
        self.blas.addresses.get(mesh).copied()
    }

    pub fn blas_stats(&self) -> &BlasStats {
        &self.blas.stats
    }

    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// World-space AABB of instance `i`.
    pub fn instance_aabb(&self, i: usize) -> Option<([f32; 3], [f32; 3])> {
        self.instance_aabbs.get(i).copied()
    }

    /// Instance of every entity given to [`load_scene`] ([`NO_INSTANCE`]: not traced).
    pub fn entity_instances(&self) -> &[u32] {
        &self.entity_instances
    }

    /// The dynamic reserve (`reserve` None for a static scene).
    pub fn dynamic_tables(&self) -> &DynamicTables {
        &self.dynamic
    }

    /// TLAS instance descriptors of the instances selected by `keep` (encoding of
    /// [`accel::instance_desc`]).
    pub fn instance_descs(&self, keep: impl Fn(usize) -> bool) -> Vec<[u32; 16]> {
        self.instances
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(_, inst)| {
                let m = inst.mesh as usize;
                accel::instance_desc(
                    inst,
                    self.meshes[m].first_geometry,
                    self.blas.addresses[m],
                    true,
                )
            })
            .collect()
    }

    /// Descriptors of the instances whose world AABB lies within `radius` of `center`.
    pub fn cutoff_descs(&self, center: [f32; 3], radius: f32) -> Vec<[u32; 16]> {
        let r2 = radius * radius;
        self.instance_descs(|i| {
            let (lo, hi) = self.instance_aabbs[i];
            aabb_dist2(lo, hi, center) <= r2
        })
    }

    /// A cut-off TLAS around `center` with room for every instance (so later
    /// [`GpuScene::update_cutoff_tlas`] calls always fit). Blocking, one submit.
    pub fn build_cutoff_tlas(
        &self,
        gpu: &mut Gpu,
        center: [f32; 3],
        radius: f32,
    ) -> Result<Tlas, String> {
        Tlas::build(
            gpu,
            &self.cutoff_descs(center, radius),
            self.instances.len(),
        )
    }

    /// Rebuilds `tlas` (from [`GpuScene::build_cutoff_tlas`]) around a new centre; returns
    /// the instance count. Blocking (waits for GPU work in flight), one submit.
    pub fn update_cutoff_tlas(
        &self,
        gpu: &mut Gpu,
        tlas: &mut Tlas,
        center: [f32; 3],
        radius: f32,
    ) -> Result<usize, String> {
        let d = self.cutoff_descs(center, radius);
        tlas.rebuild(gpu, &d)?;
        Ok(d.len())
    }

    /// TLAS over every instance (or the `tlas_limit` subset).
    pub fn full_tlas(&self) -> &Tlas {
        &self.full_tlas
    }

    pub fn full_tlas_va(&self) -> u64 {
        self.full_tlas.va()
    }

    /// Geometry table, records, materials, lights, light grid and textures with counts.
    pub fn tables(&self) -> &SceneGpu {
        &self.tables
    }

    pub fn textures(&self) -> &TextureSet {
        &self.tables.textures
    }

    /// FC6 primitive buffer (6 words per triangle) with [`SceneOptions::spike_primitives`].
    pub fn primitives(&self) -> Option<&Buffer> {
        self.primitives.as_ref()
    }

    /// First primitive per global geometry (u32) with [`SceneOptions::spike_primitives`].
    pub fn primitive_offsets(&self) -> Option<&Buffer> {
        self.primitive_offsets.as_ref()
    }

    /// The photo kernels (root signature [`LAYOUT`]).
    pub fn programs(&self) -> &Programs {
        &self.programs
    }

    pub fn sun(&self) -> Option<Sun> {
        self.sun
    }

    pub fn env(&self) -> Option<Env> {
        self.env
    }

    /// Light rows given to [`load_scene`] (`tables().light_count` were kept).
    pub fn light_rows(&self) -> usize {
        self.light_rows
    }

    /// Sun and sky inputs of the kernels for `s` (zero when the source had none).
    pub fn lighting(&self, s: &PhotoSettings) -> Lighting {
        let sun = self.sun.unwrap_or(Sun {
            theta_deg: 0.0,
            phi_deg: 0.0,
            color: [0.0; 3],
            dir: [0.0, 0.0, 1.0],
        });
        let env = self.env.unwrap_or(Env {
            sky_zenith: [0.0; 3],
            sky_horizon: [0.0; 3],
            ambient: [0.0; 3],
        });
        Lighting {
            sun_dir: sun.dir,
            sun_irradiance: sun.color.map(|c| c * SUN_COLOR_SCALE),
            sun_radius_deg: s.sun_radius_deg,
            sky_zenith: env.sky_zenith,
            sky_horizon: env.sky_horizon,
            sky_scale: s.sky_scale,
        }
    }

    /// The `Params` block (t7) for a camera and image size.
    pub fn params(
        &self,
        camera: &Camera,
        width: u32,
        height: u32,
        s: &PhotoSettings,
    ) -> [u32; PARAM_WORDS] {
        photo::params(
            &camera.basis(width, height),
            s.znear,
            &self.lighting(s),
            &PathSettings {
                max_bounces: s.bounces,
                rr_start: s.rr_start,
                clamp_indirect: s.clamp,
                ris_m: s.ris_m,
            },
            &self.tables,
        )
    }
}

/// Path-tracer settings (defaults = the validated offline settings, section 10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotoSettings {
    /// Samples per pixel to accumulate; [`PhotoSession::step`] stops there.
    pub spp_target: u32,
    /// Bounces after the primary hit.
    pub bounces: u32,
    /// Russian roulette from this bounce on.
    pub rr_start: u32,
    /// Luminance clamp of every contribution after the primary hit (absolute units).
    pub clamp: f32,
    /// RIS candidates per local-light pick.
    pub ris_m: u32,
    /// Sky radiance multiplier on lerp(skyZenith, skyHorizon) (fitted 0.067 at RX010).
    pub sky_scale: f32,
    /// Sun disc angular radius (degrees).
    pub sun_radius_deg: f32,
    /// `TF_*` switches.
    pub flags: u32,
    /// `r_znear` (game depth = znear / view z).
    pub znear: f32,
    /// Pixels per dispatch at most (rounded down to whole rows, a multiple of 4 rows).
    pub tile_pixels: u32,
    /// A-trous passes of [`PhotoSession::denoise`] (at most 6).
    pub denoise_iters: u32,
}

impl Default for PhotoSettings {
    fn default() -> Self {
        Self {
            spp_target: 256,
            bounces: 4,
            rr_start: 2,
            clamp: 100.0,
            ris_m: 8,
            sky_scale: 0.067,
            sun_radius_deg: 0.27,
            flags: 0,
            znear: DEFAULT_ZNEAR,
            tile_pixels: DEFAULT_TILE_PIXELS,
            denoise_iters: 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Progress {
    /// Samples complete in every pixel (GPU finished).
    pub spp_done: u32,
    pub spp_target: u32,
    /// GPU ms of the last complete sample (sum of its tiles; estimate before the first).
    pub ms_per_spp: f32,
    /// Mean GPU ms of a path-tracing tile (latest measurement of each tile).
    pub ms_per_tile: f32,
    pub tiles_per_spp: u32,
    /// The depth buffer is complete for the current camera.
    pub depth_ready: bool,
    /// A submission of this session is still executing.
    pub busy: bool,
    /// `spp_done == spp_target` and nothing in flight.
    pub done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Item {
    Depth(u32),
    Trace { sample: u32, tile: u32 },
}

/// [`PhotoSession::read_features`]: (albedo rgb + sun visibility, normal xyz + view depth)
/// means per pixel.
pub type Features = (Vec<[f32; 4]>, Vec<[f32; 4]>);

struct Pending {
    fence: u64,
    items: Vec<Item>,
}

/// Progressive path tracing of one camera into an accumulation buffer, in row tiles.
pub struct PhotoSession {
    // First field: dropped first, so the buffers below outlive the GPU work using them.
    guard: Option<SubmitGuard>,
    scene: Arc<GpuScene>,
    camera: Camera,
    width: u32,
    height: u32,
    settings: PhotoSettings,
    flags: u32,
    params: Buffer,
    acc: Accum,
    hdr: Buffer,
    depth: Buffer,
    tile_rows: u32,
    tiles: u32,
    // Next item to record: depth tiles, then (sample, tile).
    next_depth: u32,
    next_sample: u32,
    next_tile: u32,
    spp_done: u32,
    depth_ready: bool,
    pending: Option<Pending>,
    /// Latest GPU ms of each path-tracing tile (rows differ in cost; same rows per sample).
    tile_est: Vec<Option<f64>>,
    est_depth_ms: Option<f64>,
    sample_ms: Vec<f64>,
    cur_sample_ms: f64,
    cur_tile_ms: Vec<f64>,
    last_tile_ms: Vec<f64>,
    depth_ms: f64,
    submits: usize,
    max_submit_ms: f64,
    /// Traced instead of the scene's full TLAS (dynamic objects frozen at entry).
    tlas: Option<Arc<SceneTlas>>,
}

/// Timestamp slots used by sessions: the top of the query heap (blocking users take 0..).
fn ts_first(gpu: &Gpu) -> u32 {
    gpu.max_queries - 2 * MAX_DISPATCHES_PER_SUBMIT
}

impl PhotoSession {
    /// Allocates the accumulation (5 x 16 B per pixel), HDR and depth buffers and uploads
    /// the camera parameters (blocking, short submits). `gpu` must be the device `scene`
    /// was uploaded with.
    pub fn new(
        gpu: &mut Gpu,
        scene: Arc<GpuScene>,
        camera: Camera,
        width: u32,
        height: u32,
        settings: PhotoSettings,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(format!("image size {width}x{height}"));
        }
        if settings.denoise_iters + 2 > MAX_DISPATCHES_PER_SUBMIT {
            return Err(format!(
                "{} a-trous iterations: at most {}",
                settings.denoise_iters,
                MAX_DISPATCHES_PER_SUBMIT - 2
            ));
        }
        let flags = settings.flags
            | if scene.sun.is_none() { TF_NO_SUN } else { 0 }
            | if scene.env.is_none() { TF_NO_SKY } else { 0 };
        gpu.check_vram(Self::buffer_bytes(width, height), "the photo buffers")?;
        let params = gpu.upload(&scene.params(&camera, width, height, &settings), false)?;
        let acc = Accum::new(gpu, width, height)?;
        let hdr = gpu.buffer(u64::from(height) * hdr_pitch(width), true)?;
        let depth = gpu.buffer(u64::from(height) * depth_pitch(width), true)?;
        gpu.clear(&hdr)?;
        gpu.clear(&depth)?;
        gpu.submit()?;
        let rows = ((settings.tile_pixels / width).max(4) / 4 * 4).min(height);
        Ok(Self {
            guard: None,
            scene,
            camera,
            width,
            height,
            settings,
            flags,
            params,
            acc,
            hdr,
            depth,
            tile_rows: rows,
            tiles: height.div_ceil(rows),
            next_depth: 0,
            next_sample: 0,
            next_tile: 0,
            spp_done: 0,
            depth_ready: false,
            pending: None,
            tile_est: vec![None; height.div_ceil(rows) as usize],
            est_depth_ms: None,
            sample_ms: Vec::new(),
            cur_sample_ms: 0.0,
            cur_tile_ms: Vec::new(),
            last_tile_ms: Vec::new(),
            depth_ms: 0.0,
            submits: 0,
            max_submit_ms: 0.0,
            tlas: None,
        })
    }

    /// Video memory a `width x height` session allocates (accumulators, HDR, depth).
    pub fn buffer_bytes(width: u32, height: u32) -> u64 {
        let h = u64::from(height);
        u64::from(width) * h * photo::ACCUM_BYTES_PER_PIXEL * 5
            + h * hdr_pitch(width)
            + h * depth_pitch(width)
            + 256
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn settings(&self) -> &PhotoSettings {
        &self.settings
    }

    pub fn scene(&self) -> &Arc<GpuScene> {
        &self.scene
    }

    /// Traces `tlas` (a [`SceneTlas`] of this scene, e.g. [`Dynamics::full`] frozen at
    /// photo entry) instead of the full TLAS, or the full TLAS again with None; restarts
    /// the accumulation.
    pub fn set_tlas(&mut self, gpu: &mut Gpu, tlas: Option<Arc<SceneTlas>>) -> Result<(), String> {
        self.tlas = tlas;
        let cam = self.camera;
        self.reset_camera(gpu, cam)
    }

    pub fn spp_done(&self) -> u32 {
        self.spp_done
    }

    /// Changes the sample target (e.g. to stop at intermediate counts).
    pub fn set_spp_target(&mut self, spp: u32) {
        self.settings.spp_target = spp;
    }

    /// Rows per tile and tiles per sample.
    pub fn tiling(&self) -> (u32, u32) {
        (self.tile_rows, self.tiles)
    }

    /// GPU ms of every complete sample so far.
    pub fn sample_ms(&self) -> &[f64] {
        &self.sample_ms
    }

    /// GPU ms of each tile of the last complete sample (top to bottom).
    pub fn last_sample_tile_ms(&self) -> &[f64] {
        &self.last_tile_ms
    }

    /// GPU ms of the depth pass (all tiles).
    pub fn depth_ms(&self) -> f64 {
        self.depth_ms
    }

    /// (submits, largest GPU ms of one submit) of `step`.
    pub fn submit_stats(&self) -> (usize, f64) {
        (self.submits, self.max_submit_ms)
    }

    pub fn accum(&self) -> &Accum {
        &self.acc
    }

    /// Resolved mean radiance (absolute units, see the module doc), RGBA32F with rows of
    /// [`hdr_pitch`]`(width)` bytes; valid after [`PhotoSession::copy_out`] /
    /// [`PhotoSession::read_hdr`] (else zero).
    pub fn hdr_buffer(&self) -> &Buffer {
        &self.hdr
    }

    /// Game depth of the primary hit through each pixel centre (`znear / view z`, 0 = sky),
    /// R32F with rows of [`depth_pitch`]`(width)` bytes; complete once
    /// [`Progress::depth_ready`].
    pub fn depth_buffer(&self) -> &Buffer {
        &self.depth
    }

    /// Expected GPU ms of path-tracing tile `t`: its last measurement, else the slowest
    /// measured tile, else unknown.
    fn trace_est(&self, t: u32) -> Option<f64> {
        self.tile_est
            .get(t as usize)
            .copied()
            .flatten()
            .or_else(|| self.tile_est.iter().flatten().copied().reduce(f64::max))
    }

    fn progress(&self) -> Progress {
        let known: Vec<f64> = self.tile_est.iter().flatten().copied().collect();
        let tile_mean = if known.is_empty() {
            0.0
        } else {
            known.iter().sum::<f64>() / known.len() as f64
        };
        let est = self
            .sample_ms
            .last()
            .copied()
            .unwrap_or(tile_mean * f64::from(self.tiles));
        let busy = self.pending.is_some();
        Progress {
            spp_done: self.spp_done,
            spp_target: self.settings.spp_target,
            ms_per_spp: est as f32,
            ms_per_tile: tile_mean as f32,
            tiles_per_spp: self.tiles,
            depth_ready: self.depth_ready,
            busy,
            done: !busy && self.spp_done >= self.settings.spp_target,
        }
    }

    fn next_item(&self) -> Option<Item> {
        if self.next_depth < self.tiles {
            Some(Item::Depth(self.next_depth))
        } else if self.next_sample < self.settings.spp_target {
            Some(Item::Trace {
                sample: self.next_sample,
                tile: self.next_tile,
            })
        } else {
            None
        }
    }

    fn advance(&mut self, item: Item) {
        match item {
            Item::Depth(_) => self.next_depth += 1,
            Item::Trace { .. } => {
                self.next_tile += 1;
                if self.next_tile == self.tiles {
                    self.next_tile = 0;
                    self.next_sample += 1;
                }
            }
        }
    }

    fn band(&self, tile: u32) -> (u32, u32) {
        let y0 = tile * self.tile_rows;
        (y0, self.tile_rows.min(self.height - y0))
    }

    fn srvs(&self) -> [Option<&Buffer>; 8] {
        let t = &self.scene.tables;
        [
            Some(
                self.tlas
                    .as_ref()
                    .map_or(&self.scene.full_tlas.buffer, |t| &t.tlas.buffer),
            ),
            Some(&t.geometry),
            Some(&t.records),
            Some(&t.materials),
            Some(&t.lights),
            Some(&t.cells),
            Some(&t.light_indices),
            Some(&self.params),
        ]
    }

    fn constants(&self, frame: u32, step: u32, y0: u32) -> [u32; 8] {
        [self.width, self.height, frame, step, self.flags, y0, 0, 0]
    }

    fn record(&self, gpu: &Gpu, item: Item) -> Result<(), String> {
        let a = &self.acc;
        let (entry, frame, tile, uav) = match item {
            Item::Depth(t) => (
                "photo_depth",
                0,
                t,
                [None, None, None, None, Some(&self.depth)],
            ),
            Item::Trace { sample, tile } => (
                "photo_trace",
                sample,
                tile,
                [Some(&a.rad), Some(&a.alb), Some(&a.nrm), None, None],
            ),
        };
        let (y0, rows) = self.band(tile);
        self.scene.programs.dispatch_rows(
            gpu,
            entry,
            &self.constants(frame, 0, y0),
            &self.srvs(),
            &uav,
            Some(&self.scene.tables.textures),
            (self.width, self.height),
            (y0, rows),
        )
    }

    /// Accounts the finished submission (if any). `Ok(false)` while it still executes.
    fn collect(&mut self, gpu: &mut Gpu) -> Result<bool, String> {
        let Some(p) = &self.pending else {
            return Ok(true);
        };
        if !gpu.completed(p.fence)? {
            return Ok(false);
        }
        gpu.poll()?;
        let Some(p) = self.pending.take() else {
            return Ok(true);
        };
        let n = 2 * p.items.len() as u32;
        let ts = gpu.read_timestamps(ts_first(gpu), n)?;
        let mut total = 0.0;
        let mut worst_depth = None::<f64>;
        for (k, item) in p.items.iter().enumerate() {
            let ms = gpu.ticks_to_ms(ts[2 * k], ts[2 * k + 1]);
            total += ms;
            match *item {
                Item::Depth(t) => {
                    worst_depth = Some(worst_depth.map_or(ms, |w| w.max(ms)));
                    self.depth_ms += ms;
                    if t + 1 == self.tiles {
                        self.depth_ready = true;
                    }
                }
                Item::Trace { sample, tile } => {
                    self.tile_est[tile as usize] = Some(ms);
                    self.cur_sample_ms += ms;
                    self.cur_tile_ms.push(ms);
                    if tile + 1 == self.tiles {
                        self.sample_ms.push(self.cur_sample_ms);
                        self.cur_sample_ms = 0.0;
                        self.last_tile_ms = std::mem::take(&mut self.cur_tile_ms);
                        self.spp_done = sample + 1;
                    }
                }
            }
        }
        if worst_depth.is_some() {
            self.est_depth_ms = worst_depth;
        }
        self.submits += 1;
        self.max_submit_ms = self.max_submit_ms.max(total);
        Ok(true)
    }

    /// Advances the accumulation without blocking: accounts the previous submission once
    /// the GPU finished it (else returns at once with `busy`), then records as many tiles
    /// (depth first, then path samples) as fit `budget_ms` of GPU time by the last measured
    /// cost of each tile (path tiles per row band, depth tiles by the slowest; one tile while
    /// unmeasured; at most
    /// [`MAX_DISPATCHES_PER_SUBMIT`]) and submits them without waiting. A host calling it
    /// once per frame keeps at most one submission of `budget_ms` in flight.
    pub fn step(&mut self, gpu: &mut Gpu, budget_ms: f32) -> Result<Progress, String> {
        if !self.collect(gpu)? {
            return Ok(self.progress());
        }
        let budget = f64::from(budget_ms.max(0.1));
        let first = ts_first(gpu);
        let mut items = Vec::new();
        let mut cost = 0.0;
        while (items.len() as u32) < MAX_DISPATCHES_PER_SUBMIT {
            let Some(item) = self.next_item() else {
                break;
            };
            let est = match item {
                Item::Depth(_) => self.est_depth_ms,
                Item::Trace { tile, .. } => self.trace_est(tile),
            };
            match est {
                // Unmeasured kind: only as the first item of a submit.
                None if !items.is_empty() => break,
                Some(e) if !items.is_empty() && cost + e > budget => break,
                _ => {}
            }
            let k = items.len() as u32;
            gpu.timestamp(first + 2 * k);
            if let Err(e) = self.record(gpu, item) {
                let _ = gpu.discard();
                return Err(e);
            }
            gpu.uav_barrier();
            gpu.timestamp(first + 2 * k + 1);
            items.push(item);
            self.advance(item);
            cost += est.unwrap_or(budget);
            if est.is_none() {
                break;
            }
        }
        if items.is_empty() {
            return Ok(self.progress());
        }
        gpu.resolve_timestamps(first, 2 * items.len() as u32)?;
        let fence = gpu.submit_async()?;
        self.guard = Some(gpu.guard());
        self.pending = Some(Pending { fence, items });
        Ok(self.progress())
    }

    /// Blocks until this session's submission (if any) finished and accounts it.
    pub fn finish(&mut self, gpu: &mut Gpu) -> Result<Progress, String> {
        if let Some(p) = &self.pending {
            let fence = p.fence;
            if !gpu.completed(fence)? {
                gpu.wait_idle()?;
                if !gpu.completed(fence)? {
                    return Err(format!("photo: fence {fence} not reached after wait"));
                }
            }
        }
        self.collect(gpu)?;
        Ok(self.progress())
    }

    /// Restarts the accumulation for a new camera: waits for the submission in flight
    /// (at most one step's budget), uploads the parameters and zeroes the accumulators
    /// (blocking, short); depth is recomputed by the next steps.
    pub fn reset_camera(&mut self, gpu: &mut Gpu, camera: Camera) -> Result<(), String> {
        self.finish(gpu)?;
        self.camera = camera;
        self.params = gpu.upload(
            &self
                .scene
                .params(&camera, self.width, self.height, &self.settings),
            false,
        )?;
        self.acc.clear(gpu)?;
        gpu.submit()?;
        self.next_depth = 0;
        self.next_sample = 0;
        self.next_tile = 0;
        self.spp_done = 0;
        self.depth_ready = false;
        self.sample_ms.clear();
        self.cur_sample_ms = 0.0;
        self.cur_tile_ms.clear();
        self.last_tile_ms.clear();
        self.depth_ms = 0.0;
        Ok(())
    }

    /// Records the resolve of the accumulated mean into the HDR buffer. Rows already
    /// covered by the sample in progress hold one sample more than the rest; both bands
    /// divide by their own count.
    fn record_resolve(&self, gpu: &Gpu) -> Result<(), String> {
        let split = (self.next_tile * self.tile_rows).min(self.height);
        let uav = [Some(&self.acc.rad), None, None, Some(&self.hdr), None];
        for (y0, rows, n) in [
            (0, split, self.next_sample + 1),
            (split, self.height - split, self.next_sample),
        ] {
            if rows == 0 {
                continue;
            }
            self.scene.programs.dispatch_rows(
                gpu,
                "photo_resolve",
                &self.constants(n, 0, y0),
                &self.srvs(),
                &uav,
                Some(&self.scene.tables.textures),
                (self.width, self.height),
                (y0, rows),
            )?;
        }
        Ok(())
    }

    /// Resolves the current mean (absolute units) and copies it and the depth into the
    /// host's textures: `hdr_tex` R32G32B32A32_FLOAT (or TYPELESS), `depth_tex` R32_FLOAT
    /// (or TYPELESS), both 2D, 1 mip, exactly `width x height`, in COMMON on entry and
    /// returned to COMMON. Submits without waiting (waits only for a submission still in
    /// flight, at most one step's budget) and returns the fence value: poll
    /// [`Gpu::completed`] before the host reads the textures.
    pub fn copy_out(
        &mut self,
        gpu: &mut Gpu,
        hdr_tex: &ID3D12Resource,
        depth_tex: &ID3D12Resource,
    ) -> Result<u64, String> {
        let r = self.record_copy_out(gpu, hdr_tex, depth_tex);
        if let Err(e) = r {
            let _ = gpu.discard();
            return Err(e);
        }
        let v = gpu.submit_async()?;
        self.guard = Some(gpu.guard());
        Ok(v)
    }

    fn record_copy_out(
        &self,
        gpu: &Gpu,
        hdr_tex: &ID3D12Resource,
        depth_tex: &ID3D12Resource,
    ) -> Result<(), String> {
        self.record_resolve(gpu)?;
        // The resolve promoted the HDR buffer to UAV in this list; the depth buffer is in
        // COMMON (written in earlier submissions) and promotes to COPY_SOURCE implicitly.
        gpu.transition(
            &self.hdr,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        for (tex, buf, pitch, format, typeless) in [
            (
                hdr_tex,
                &self.hdr,
                hdr_pitch(self.width),
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                DXGI_FORMAT_R32G32B32A32_TYPELESS,
            ),
            (
                depth_tex,
                &self.depth,
                depth_pitch(self.width),
                DXGI_FORMAT_R32_FLOAT,
                DXGI_FORMAT_R32_TYPELESS,
            ),
        ] {
            gpu.transition_resource(
                tex,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            gpu.copy_buffer_to_texture(buf, pitch, format, typeless, self.width, self.height, tex)?;
            gpu.transition_resource(
                tex,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
        Ok(())
    }

    /// Strips the row padding of a pitched buffer read back as `T`.
    fn unpitch<T: Copy>(&self, raw: &[T], per_row: usize) -> Vec<T> {
        let w = self.width as usize;
        raw.chunks(per_row)
            .flat_map(|r| r[..w].iter().copied())
            .collect()
    }

    /// Mean radiance per pixel (absolute units, alpha 1), `width x height` row-major.
    /// Blocking: waits for the session, resolves, reads back.
    pub fn read_hdr(&mut self, gpu: &mut Gpu) -> Result<Vec<[f32; 4]>, String> {
        self.finish(gpu)?;
        if let Err(e) = self.record_resolve(gpu) {
            let _ = gpu.discard();
            return Err(e);
        }
        gpu.submit()?;
        let per_row = (hdr_pitch(self.width) / 16) as usize;
        let raw: Vec<[f32; 4]> = gpu.read(&self.hdr, per_row * self.height as usize)?;
        Ok(self.unpitch(&raw, per_row))
    }

    /// Game depth per pixel (`znear / view z`, 0 sky), `width x height`. Blocking.
    pub fn read_depth(&mut self, gpu: &mut Gpu) -> Result<Vec<f32>, String> {
        self.finish(gpu)?;
        let per_row = (depth_pitch(self.width) / 4) as usize;
        let raw: Vec<f32> = gpu.read(&self.depth, per_row * self.height as usize)?;
        Ok(self.unpitch(&raw, per_row))
    }

    /// Means of the first-hit feature accumulators: (albedo rgb + sun visibility, normal
    /// xyz + view depth (0 sky)). Blocking; at a sample boundary.
    pub fn read_features(&mut self, gpu: &mut Gpu) -> Result<Features, String> {
        self.finish(gpu)?;
        let n = self.acc.pixels();
        let alb = mean(&gpu.read(&self.acc.alb, n)?, self.spp_done);
        let nrm = mean(&gpu.read(&self.acc.nrm, n)?, self.spp_done);
        Ok((alb, nrm))
    }

    /// Denoised mean (albedo-demodulated a-trous, `denoise_iters` passes) and its GPU ms.
    /// Blocking, one submit of full-frame dispatches (8-19 ms at 3440x1440, section 10.2);
    /// at a sample boundary. Helps below ~16 spp only.
    pub fn denoise(&mut self, gpu: &mut Gpu) -> Result<(Vec<[f32; 4]>, f64), String> {
        self.finish(gpu)?;
        if self.next_tile != 0 || self.next_sample != self.spp_done {
            return Err("denoise: a sample is still in progress".into());
        }
        let r = self.record_denoise(gpu);
        let other = match r {
            Ok(o) => o,
            Err(e) => {
                let _ = gpu.discard();
                return Err(e);
            }
        };
        let ts = gpu.timestamps(2)?;
        let ms = gpu.ticks_to_ms(ts[0], ts[1]);
        Ok((gpu.read(other, self.acc.pixels())?, ms))
    }

    fn record_denoise(&self, gpu: &Gpu) -> Result<&Buffer, String> {
        let a = &self.acc;
        let srv = self.srvs();
        let tex = Some(&self.scene.tables.textures);
        let size = (self.width, self.height);
        let spp = self.spp_done;
        let p = &self.scene.programs;
        // `cur` holds the current filtered image, `other` receives the next.
        let (mut cur, mut other) = (&a.a, &a.b);
        gpu.timestamp(0);
        p.dispatch(
            gpu,
            "photo_demod",
            &self.constants(spp, 0, 0),
            &srv,
            &[Some(&a.rad), Some(&a.alb), Some(&a.nrm), Some(cur), None],
            tex,
            size,
        )?;
        gpu.uav_barrier();
        for k in 0..self.settings.denoise_iters {
            p.dispatch(
                gpu,
                "photo_atrous",
                &self.constants(spp, 1 << k, 0),
                &srv,
                &[None, None, Some(&a.nrm), Some(cur), Some(other)],
                tex,
                size,
            )?;
            gpu.uav_barrier();
            std::mem::swap(&mut cur, &mut other);
        }
        p.dispatch(
            gpu,
            "photo_remod",
            &self.constants(spp, 0, 0),
            &srv,
            &[
                Some(&a.rad),
                Some(&a.alb),
                Some(&a.nrm),
                Some(cur),
                Some(other),
            ],
            tex,
            size,
        )?;
        gpu.uav_barrier();
        gpu.timestamp(1);
        Ok(other)
    }

    /// Saves the mean as an 8-bit sRGB PNG with our own tone mapping (x
    /// [`DAYLIGHT_STORED_SCALE`], `exposure` or median luminance -> 0.18, per-channel
    /// Reinhard; not the engine's curve). Returns the exposure used. Blocking.
    pub fn write_png(
        &mut self,
        gpu: &mut Gpu,
        path: &Path,
        exposure: Option<f32>,
    ) -> Result<f32, String> {
        let hdr = self.read_hdr(gpu)?;
        let rgb: Vec<[f32; 3]> = hdr
            .iter()
            .map(|c| [c[0], c[1], c[2]].map(|v| v * DAYLIGHT_STORED_SCALE))
            .collect();
        let e = exposure.unwrap_or_else(|| image::auto_exposure(rgb.iter().copied()));
        let mut px = Vec::with_capacity(rgb.len() * 3);
        for c in &rgb {
            px.extend_from_slice(&image::tonemap(*c, e));
        }
        image::write_png(path, self.width, self.height, &px)?;
        Ok(e)
    }

    /// Saves the mean as raw little-endian RGBA32F (`.bin`, absolute units) plus a `.tsv`
    /// header (section 10 format). Blocking.
    pub fn write_hdr(&mut self, gpu: &mut Gpu, path: &Path, notes: &str) -> Result<(), String> {
        let hdr = self.read_hdr(gpu)?;
        let rgb: Vec<[f32; 3]> = hdr.iter().map(|c| [c[0], c[1], c[2]]).collect();
        let notes = format!(
            "{} spp, engine absolute units (x {DAYLIGHT_STORED_SCALE} = daylight stored game colour); {notes}",
            self.spp_done
        );
        image::write_hdr(path, self.width, self.height, &rgb, &notes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_photo_entry_is_precompiled() {
        for e in photo::entries() {
            let d = dxil(e.name).unwrap_or_else(|| panic!("{} not precompiled", e.name));
            // DXIL container magic.
            assert_eq!(&d[..4], b"DXBC", "{}", e.name);
        }
        for e in ["primary", "reflect_rays", "ssr_ref", "gi"] {
            assert!(dxil(e).is_some(), "{e}");
        }
    }

    #[test]
    fn pitches_match_the_hlsl() {
        // hdr_row(w) = ceil(w * 16 / 256) * 16 float4; depth_row(w) = ceil(w * 4 / 256) * 64.
        for w in [1u32, 15, 16, 17, 860, 1366, 3440] {
            assert_eq!(hdr_pitch(w), u64::from((w * 16).div_ceil(256) * 16) * 16);
            assert_eq!(depth_pitch(w), u64::from((w * 4).div_ceil(256) * 64) * 4);
            assert!(hdr_pitch(w) >= u64::from(w) * 16 && hdr_pitch(w).is_multiple_of(256));
            assert!(depth_pitch(w) >= u64::from(w) * 4 && depth_pitch(w).is_multiple_of(256));
        }
    }

    #[test]
    fn hdr_scale() {
        assert!((stored_hdr_scale(139.26 / (255.0 * 1.956)) - DAYLIGHT_STORED_SCALE).abs() < 0.01);
    }

    #[test]
    fn texture_caps_halve_to_the_minimum() {
        assert_eq!(texture_caps(2048, 256), vec![2048, 1024, 512, 256]);
        assert_eq!(texture_caps(512, 256), vec![512, 256]);
        // Below the minimum (smoke runs): the request alone.
        assert_eq!(texture_caps(64, 256), vec![64]);
    }

    #[test]
    fn the_largest_fitting_cap_is_chosen() {
        let est = MemoryEstimate {
            geometry: 100,
            blas_uncompacted: 800,
            blas_scratch: 256,
            tlas: 50,
            tables: 300,
            primitives: 100,
            staging: 64,
            textures: vec![
                (2048, 6000, 0),
                (1024, 2400, 0),
                (512, 900, 0),
                (256, 340, 0),
            ],
        };
        assert_eq!(est.blas_peak(), 100 + 800 + 800);
        assert_eq!(est.peak(2048), 800 + 50 + 300 + 100 + 6000 + 64);
        let pick = |avail| fitting_cap(&est.textures, avail, |c| est.peak(c));
        assert_eq!(pick(10_000), Some(2048));
        assert_eq!(pick(4_000), Some(1024));
        assert_eq!(pick(2_300), Some(512));
        // The BLAS phase bounds the minimum.
        assert_eq!(pick(1_699), None);
    }
}
