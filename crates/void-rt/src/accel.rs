//! Acceleration structures: one compacted, PREFER_FAST_TRACE BLAS per mesh (static, built
//! once) and TLASes over instance subsets, with GPU timings and memory accounting.

use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32G32B32_FLOAT};

use crate::gpu::{Buffer, Gpu};
use crate::scene::{Instance, Origin, Scene};

const ALIGN: u64 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT as u64;

fn align(v: u64) -> u64 {
    v.div_ceil(ALIGN) * ALIGN
}

pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[derive(Clone, Debug, Default)]
pub struct BlasStats {
    pub count: usize,
    pub geometries: usize,
    pub triangles: u64,
    /// Sum of 256-aligned `ResultDataMaxSizeInBytes` (before compaction).
    pub uncompacted_bytes: u64,
    /// Sum of 256-aligned compacted sizes.
    pub compacted_bytes: u64,
    /// Sum of all scratch sizes (if every BLAS were built at once) and the arena used.
    pub scratch_total_bytes: u64,
    pub scratch_arena_bytes: u64,
    /// Build submits (batches) and compaction submits.
    pub batches: usize,
    pub compact_submits: usize,
    pub build_gpu_ms: f64,
    pub compact_gpu_ms: f64,
    /// Longest single build / compaction submit (GPU ms).
    pub max_submit_ms: f64,
    /// CPU wall time of prebuild queries, recording, both submits and the readback.
    pub wall_ms: f64,
}

pub struct BlasSet {
    pub arena: Buffer,
    /// BLAS address per mesh.
    pub addresses: Vec<u64>,
    pub stats: BlasStats,
}

/// GPU time aimed at per build / compaction submit: the game keeps rendering on the same
/// GPU, so no submit of ours may hold it for long (TDR safety, frame pacing).
pub const SUBMIT_TARGET_MS: f64 = 16.0;
/// Triangles of the first BLAS build batch (later batches scale with the measured time).
const FIRST_BATCH_TRIANGLES: u64 = 1 << 20;
/// Compacted bytes of the first compaction submit (scaled likewise).
const FIRST_COMPACT_BYTES: u64 = 32 << 20;

/// The next batch size after a submit of `done` units took `ms` of GPU time: the measured
/// rate times [`SUBMIT_TARGET_MS`], at most twice the previous `limit`, within `lo..=hi`.
pub fn rescale(done: u64, limit: u64, ms: f64, lo: u64, hi: u64) -> u64 {
    let grow = limit.saturating_mul(2);
    let next = if ms > 0.0 {
        ((done as f64 / ms * SUBMIT_TARGET_MS) as u64).min(grow)
    } else {
        grow
    };
    next.clamp(lo, hi)
}

/// Prebuild sizes of the BLASes of `scene` (no GPU work): (sum of 256-aligned result
/// sizes, largest scratch, sum of scratch sizes).
pub fn blas_prebuild(gpu: &Gpu, scene: &Scene) -> (u64, u64, u64) {
    let flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    let (mut result, mut scratch_max, mut scratch_sum) = (0u64, 0u64, 0u64);
    for m in &scene.meshes {
        let d: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> = m
            .geometries
            .iter()
            .map(|g| geometry_desc(g, 0, 0))
            .collect();
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: `d` outlives the call; prebuild queries read no GPU memory.
        unsafe {
            gpu.device
                .GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs(&d, flags), &mut info)
        };
        result += align(info.ResultDataMaxSizeInBytes);
        let s = align(info.ScratchDataSizeInBytes);
        scratch_max = scratch_max.max(s);
        scratch_sum += s;
    }
    (result, scratch_max, scratch_sum)
}

fn geometry_desc(
    g: &crate::scene::Geometry,
    pos_va: u64,
    idx_va: u64,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        // Photo mode marks alpha-tested / decal geometry NON_OPAQUE (candidate loop in the
        // kernel); spike kernels force opaque per ray anyway.
        Flags: if g.opaque {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
        } else {
            D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
        },
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R16_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: g.index_count,
                VertexCount: g.vertex_count,
                IndexBuffer: idx_va + u64::from(g.first_index) * 2,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: pos_va + u64::from(g.first_vertex) * 12,
                    StrideInBytes: 12,
                },
            },
        },
    }
}

fn blas_inputs(
    d: &[D3D12_RAYTRACING_GEOMETRY_DESC],
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: flags,
        NumDescs: d.len() as u32,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: d.as_ptr(),
        },
    }
}

/// Kind of a dynamic BLAS ([`record_dynamic_blas`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasKind {
    /// Built once (spawned static props): PREFER_FAST_TRACE.
    Rigid,
    /// Refitted per pose (skinned characters): PREFER_FAST_BUILD | ALLOW_UPDATE.
    Deforming,
}

impl BlasKind {
    fn flags(self) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
        match self {
            Self::Rigid => D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            Self::Deforming => {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
                    | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
            }
        }
    }
}

/// Sizes of a dynamic BLAS over `geoms` (no GPU work): (result, build scratch, update
/// scratch), 256-aligned.
pub fn dynamic_blas_sizes(
    gpu: &Gpu,
    geoms: &[crate::scene::Geometry],
    kind: BlasKind,
) -> (u64, u64, u64) {
    let d: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> =
        geoms.iter().map(|g| geometry_desc(g, 0, 0)).collect();
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: `d` outlives the call; prebuild queries read no GPU memory.
    unsafe {
        gpu.device.GetRaytracingAccelerationStructurePrebuildInfo(
            &blas_inputs(&d, kind.flags()),
            &mut info,
        )
    };
    (
        align(info.ResultDataMaxSizeInBytes),
        align(info.ScratchDataSizeInBytes),
        align(info.UpdateScratchDataSizeInBytes),
    )
}

/// Records a build of the BLAS over `geoms` (positions at `pos_va`, 12-byte stride; u16
/// indices at `idx_va`) into `dest_va`, or with `source` an update of that BLAS into
/// `dest_va` (`Deforming` only: same geometry, new positions; source and destination may
/// differ). `scratch_va` holds the build / update scratch size of
/// [`dynamic_blas_sizes`] in UNORDERED_ACCESS; vertex and index buffers are in
/// NON_PIXEL_SHADER_RESOURCE. A UAV barrier follows.
#[allow(clippy::too_many_arguments)]
pub fn record_dynamic_blas(
    gpu: &Gpu,
    geoms: &[crate::scene::Geometry],
    pos_va: u64,
    idx_va: u64,
    kind: BlasKind,
    dest_va: u64,
    scratch_va: u64,
    source: Option<u64>,
) {
    let d: Vec<D3D12_RAYTRACING_GEOMETRY_DESC> = geoms
        .iter()
        .map(|g| geometry_desc(g, pos_va, idx_va))
        .collect();
    let mut flags = kind.flags();
    if source.is_some() && kind == BlasKind::Deforming {
        flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
    let build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: dest_va,
        Inputs: blas_inputs(&d, flags),
        SourceAccelerationStructureData: if kind == BlasKind::Deforming {
            source.unwrap_or(0)
        } else {
            0
        },
        ScratchAccelerationStructureData: scratch_va,
    };
    // SAFETY: the caller's contract above (live buffers in the required states, sizes from
    // `dynamic_blas_sizes`); `d` outlives the call.
    unsafe { gpu.list.BuildRaytracingAccelerationStructure(&build, None) };
    gpu.uav_barrier();
}

/// Builds and compacts one BLAS per mesh. `pos`/`idx` hold `scene.positions`/`indices`.
///
/// Blocking, short submits: a build batch ends when its scratch would exceed
/// `scratch_budget` or its triangles the current batch size; compaction copies go in
/// submits of a byte budget. Both sizes start small and follow the measured GPU time of the
/// previous submit towards [`SUBMIT_TARGET_MS`] (a single mesh larger than the batch size
/// is one batch on its own).
pub fn build_blas(
    gpu: &mut Gpu,
    scene: &Scene,
    pos: &Buffer,
    idx: &Buffer,
    scratch_budget: u64,
) -> Result<BlasSet, String> {
    let t0 = Instant::now();
    let flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    let descs: Vec<Vec<D3D12_RAYTRACING_GEOMETRY_DESC>> = scene
        .meshes
        .iter()
        .map(|m| {
            m.geometries
                .iter()
                .map(|g| geometry_desc(g, pos.va, idx.va))
                .collect()
        })
        .collect();
    let tris: Vec<u64> = scene.meshes.iter().map(|m| m.triangles()).collect();
    let mut stats = BlasStats {
        count: scene.meshes.len(),
        geometries: descs.iter().map(Vec::len).sum(),
        triangles: scene.unique_triangles(),
        ..Default::default()
    };
    let mut result_off = Vec::with_capacity(descs.len());
    let mut result_size = Vec::with_capacity(descs.len());
    let mut scratch_size = Vec::with_capacity(descs.len());
    for d in &descs {
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: `d` (geometry descs) outlives the call.
        unsafe {
            gpu.device
                .GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs(d, flags), &mut info)
        };
        result_off.push(stats.uncompacted_bytes);
        result_size.push(align(info.ResultDataMaxSizeInBytes));
        stats.uncompacted_bytes += align(info.ResultDataMaxSizeInBytes);
        let s = align(info.ScratchDataSizeInBytes);
        stats.scratch_total_bytes += s;
        scratch_size.push(s);
    }
    let scratch_arena = scratch_budget
        .min(stats.scratch_total_bytes)
        .max(scratch_size.iter().copied().max().unwrap_or(0));
    stats.scratch_arena_bytes = scratch_arena;
    let arena = gpu.accel_buffer(stats.uncompacted_bytes)?;
    let scratch = gpu.buffer(scratch_arena, true)?;
    let post = gpu.buffer(descs.len() as u64 * 8, true)?;

    // One command list (submit + wait) per batch: every submission stays short and the
    // scratch arena is reused.
    let mut batch_tris = FIRST_BATCH_TRIANGLES;
    let mut i = 0;
    while i < descs.len() {
        gpu.transition(
            pos,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        gpu.transition(
            idx,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        gpu.transition(
            &scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        gpu.transition(
            &post,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let (mut s_off, mut t_sum, first) = (0u64, 0u64, i);
        while i < descs.len()
            && (i == first
                || (s_off + scratch_size[i] <= scratch_arena && t_sum + tris[i] <= batch_tris))
        {
            let build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: arena.va + result_off[i],
                Inputs: blas_inputs(&descs[i], flags),
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: scratch.va + s_off,
            };
            let pb = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: post.va + i as u64 * 8,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            };
            // SAFETY: all addresses lie in live buffers in the required states; the scratch
            // range [s_off, s_off + size) is inside the arena (the first BLAS of a batch
            // fits: the arena is at least the largest scratch size).
            unsafe {
                gpu.list
                    .BuildRaytracingAccelerationStructure(&build, Some(&[pb]))
            };
            s_off += scratch_size[i];
            t_sum += tris[i];
            i += 1;
        }
        gpu.uav_barrier();
        gpu.submit()?;
        let ms = gpu.last_submit_ms();
        stats.build_gpu_ms += ms;
        stats.max_submit_ms = stats.max_submit_ms.max(ms);
        stats.batches += 1;
        batch_tris = rescale(t_sum, batch_tris, ms, 1 << 14, 1 << 24);
    }
    let compacted: Vec<u64> = gpu.read(&post, descs.len())?;
    drop((scratch, post));
    let mut offsets = Vec::with_capacity(compacted.len());
    let mut total = 0u64;
    for (k, &c) in compacted.iter().enumerate() {
        if c == 0 || align(c) > result_size[k] {
            return Err(format!(
                "BLAS {k}: compacted size {c} (result {}; postbuild info not written?)",
                result_size[k]
            ));
        }
        offsets.push(total);
        total += align(c);
    }
    stats.compacted_bytes = total;
    let compact = gpu.accel_buffer(total)?;
    let mut batch_bytes = FIRST_COMPACT_BYTES;
    let mut i = 0;
    while i < descs.len() {
        let (mut bytes, first) = (0u64, i);
        while i < descs.len() && (i == first || bytes + align(compacted[i]) <= batch_bytes) {
            // SAFETY: source BLASes are complete (previous submits); destinations sized by
            // the postbuild info (checked against the result size above).
            unsafe {
                gpu.list.CopyRaytracingAccelerationStructure(
                    compact.va + offsets[i],
                    arena.va + result_off[i],
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
                )
            };
            bytes += align(compacted[i]);
            i += 1;
        }
        gpu.uav_barrier();
        gpu.submit()?;
        let ms = gpu.last_submit_ms();
        stats.compact_gpu_ms += ms;
        stats.max_submit_ms = stats.max_submit_ms.max(ms);
        stats.compact_submits += 1;
        batch_bytes = rescale(bytes, batch_bytes, ms, 1 << 20, 1 << 30);
    }
    drop(arena);
    stats.wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(BlasSet {
        addresses: offsets.iter().map(|o| compact.va + o).collect(),
        arena: compact,
        stats,
    })
}

/// TLAS instance descriptors (64 bytes each) for the instances selected by `keep`.
/// `InstanceID` = the mesh's first global geometry (primitive offset table base); mask 1 =
/// world, 2 = entity; two-sided materials disable culling.
pub fn instance_descs(
    scene: &Scene,
    blas: &BlasSet,
    front_ccw: bool,
    keep: impl Fn(usize) -> bool,
) -> Vec<[u32; 16]> {
    scene
        .instances
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(_, inst)| {
            let m = inst.mesh as usize;
            instance_desc(
                inst,
                scene.meshes[m].first_geometry,
                blas.addresses[m],
                front_ccw,
            )
        })
        .collect()
}

/// One D3D12_RAYTRACING_INSTANCE_DESC (see [`instance_descs`]): rows 0..12 the row-major
/// 3x4 transform, word 12 `InstanceID` (24 bits) | mask << 24, word 13 flags << 24, words
/// 14..16 the BLAS address.
pub fn instance_desc(
    inst: &Instance,
    first_geometry: u32,
    blas_va: u64,
    front_ccw: bool,
) -> [u32; 16] {
    let mut d = [0u32; 16];
    for (k, v) in inst.transform.iter().enumerate() {
        d[k] = v.to_bits();
    }
    let mask: u32 = match inst.origin {
        Origin::World => 1,
        Origin::Entity => 2,
    };
    d[12] = (first_geometry & 0x00ff_ffff) | mask << 24;
    // The engine's front faces are counter-clockwise in DXR terms (measured: culling DXR
    // front faces keeps the game's visible surfaces, E070); a mirroring transform flips the
    // winding as in raster (`front_ccw` xor det < 0).
    let t = &inst.transform;
    let det = t[0] * (t[5] * t[10] - t[6] * t[9]) - t[1] * (t[4] * t[10] - t[6] * t[8])
        + t[2] * (t[4] * t[9] - t[5] * t[8]);
    let mut flags = 0;
    if inst.two_sided {
        flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE.0 as u32;
    }
    if (det >= 0.0) == front_ccw {
        flags |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE.0 as u32;
    }
    d[13] = flags << 24;
    d[14] = blas_va as u32;
    d[15] = (blas_va >> 32) as u32;
    d
}

#[derive(Clone, Debug, Default)]
pub struct TlasStats {
    pub instances: usize,
    pub result_bytes: u64,
    pub scratch_bytes: u64,
    pub update_scratch_bytes: u64,
    pub instance_bytes: u64,
    /// Medians over the repetitions (GPU ms).
    pub build_fast_trace_ms: f64,
    pub update_ms: f64,
    pub build_fast_build_ms: f64,
}

/// A TLAS with room for `capacity` instances: instance descriptors in an upload-heap buffer
/// (rewritten by [`Tlas::rebuild`] / [`Tlas::refit`]), result and scratch sized for
/// `capacity` (PREFER_FAST_TRACE or PREFER_FAST_BUILD, both with ALLOW_UPDATE).
pub struct Tlas {
    pub buffer: Buffer,
    pub stats: TlasStats,
    instances: Buffer,
    scratch: Buffer,
    capacity: u32,
    count: u32,
}

const TLAS_TRACE: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0,
    );
const TLAS_FAST: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0,
    );
const TLAS_UPDATE: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
        TLAS_TRACE.0 | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0,
    );

fn tlas_inputs(
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    count: u32,
    instances: u64,
) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: flags,
        NumDescs: count,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: instances,
        },
    }
}

impl Tlas {
    /// Prebuild sizes for `capacity` instances: [fast trace, fast build] (no GPU work).
    fn prebuild(
        gpu: &Gpu,
        capacity: u32,
    ) -> [D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO; 2] {
        let mut info = [D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default(); 2];
        for (k, f) in [TLAS_TRACE, TLAS_FAST].into_iter().enumerate() {
            // SAFETY: plain query with valid inputs (no descriptor memory is read).
            unsafe {
                gpu.device.GetRaytracingAccelerationStructurePrebuildInfo(
                    &tlas_inputs(f, capacity, 0),
                    &mut info[k],
                )
            };
        }
        info
    }

    /// (result, scratch) bytes [`Tlas::build`] allocates for `info`.
    fn sizes(info: &[D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO; 2]) -> (u64, u64) {
        let result = align(
            info[0]
                .ResultDataMaxSizeInBytes
                .max(info[1].ResultDataMaxSizeInBytes),
        );
        let scratch = align(
            info.iter()
                .map(|i| i.ScratchDataSizeInBytes.max(i.UpdateScratchDataSizeInBytes))
                .max()
                .unwrap_or(0),
        );
        (result, scratch)
    }

    /// Bytes of a TLAS with room for `capacity` instances: result + scratch (video memory)
    /// + instance descriptors (upload heap). No GPU work.
    pub fn bytes(gpu: &Gpu, capacity: usize) -> u64 {
        let c = capacity.clamp(1, u32::MAX as usize) as u32;
        let (r, s) = Self::sizes(&Self::prebuild(gpu, c));
        r + s + u64::from(c) * 64
    }

    /// Builds a PREFER_FAST_TRACE | ALLOW_UPDATE TLAS over `descs` (from
    /// [`instance_descs`]) with room for `capacity` instances (at least `descs.len()`). One
    /// blocking submit.
    pub fn build(gpu: &mut Gpu, descs: &[[u32; 16]], capacity: usize) -> Result<Self, String> {
        let capacity = u32::try_from(capacity.max(descs.len()).max(1))
            .map_err(|_| format!("TLAS capacity {capacity}"))?;
        let info = Self::prebuild(gpu, capacity);
        let (result, scratch_size) = Self::sizes(&info);
        let instance_bytes = u64::from(capacity) * 64;
        let mut t = Self {
            buffer: gpu.accel_buffer(result)?,
            stats: TlasStats {
                instances: descs.len(),
                result_bytes: align(info[0].ResultDataMaxSizeInBytes),
                scratch_bytes: align(info[0].ScratchDataSizeInBytes),
                update_scratch_bytes: align(info[0].UpdateScratchDataSizeInBytes),
                instance_bytes,
                ..Default::default()
            },
            instances: gpu.upload_heap(&vec![[0u32; 16]; capacity as usize])?,
            scratch: gpu.buffer(scratch_size, true)?,
            capacity,
            count: 0,
        };
        t.rebuild(gpu, descs)?;
        Ok(t)
    }

    /// GPU virtual address (bind as the `RaytracingAccelerationStructure` SRV).
    pub fn va(&self) -> u64 {
        self.buffer.va
    }

    /// Instances in the current build.
    pub fn count(&self) -> usize {
        self.count as usize
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Copies `descs` into the instance buffer after the GPU finished using it.
    fn write(&mut self, gpu: &mut Gpu, descs: &[[u32; 16]]) -> Result<(), String> {
        if descs.len() > self.capacity as usize {
            return Err(format!(
                "{} TLAS instances > capacity {}",
                descs.len(),
                self.capacity
            ));
        }
        // A previous build or trace may still read the descriptors / the structure.
        gpu.wait_idle()?;
        let bytes = crate::gpu::bytes_of(descs);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an upload buffer of capacity x 64 >= bytes.len() bytes, mapped for writing;
        // no GPU work referencing it is in flight (waited above).
        unsafe {
            self.instances
                .res
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))
                .map_err(|e| format!("TLAS instances Map: {e}{}", gpu.removed_reason()))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.instances.res.Unmap(0, None);
        }
        self.count = descs.len() as u32;
        self.stats.instances = descs.len();
        Ok(())
    }

    /// Records one build into the open list (scratch must be in UNORDERED_ACCESS).
    fn record(&self, gpu: &Gpu, flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS) {
        let update =
            flags.0 & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 != 0;
        let d = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: self.buffer.va,
            Inputs: tlas_inputs(flags, self.count, self.instances.va),
            SourceAccelerationStructureData: if update { self.buffer.va } else { 0 },
            ScratchAccelerationStructureData: self.scratch.va,
        };
        // SAFETY: result, scratch and instance buffers are live, sized for `capacity` >=
        // `count`, and in the required states.
        unsafe { gpu.list.BuildRaytracingAccelerationStructure(&d, None) };
        gpu.uav_barrier();
    }

    fn submit_build(
        &mut self,
        gpu: &mut Gpu,
        flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    ) -> Result<(), String> {
        gpu.transition(
            &self.scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        self.record(gpu, flags);
        gpu.submit()
    }

    /// Full PREFER_FAST_TRACE rebuild over a new instance set (`descs.len() <= capacity`),
    /// reusing the buffers. Blocking: waits for GPU work in flight, one submit.
    pub fn rebuild(&mut self, gpu: &mut Gpu, descs: &[[u32; 16]]) -> Result<(), String> {
        self.write(gpu, descs)?;
        self.submit_build(gpu, TLAS_TRACE)
    }

    /// Refit (PERFORM_UPDATE) with changed transforms of the same instances: `descs` must
    /// list the same BLASes in the same order as the last build (only transforms, masks,
    /// IDs and flags may change). Blocking, one submit.
    pub fn refit(&mut self, gpu: &mut Gpu, descs: &[[u32; 16]]) -> Result<(), String> {
        if descs.len() != self.count as usize {
            return Err(format!(
                "TLAS refit with {} instances, built with {}",
                descs.len(),
                self.count
            ));
        }
        self.write(gpu, descs)?;
        self.submit_build(gpu, TLAS_UPDATE)
    }

    /// Refit (PERFORM_UPDATE) of `src` into this TLAS with `descs`, which must hold as many
    /// instances as `src`'s last build; transforms, IDs, masks, flags and even the BLAS of
    /// each instance may change (DXR spec, "Top-level acceleration structure updates").
    /// `src` is only read, so it may still be traced elsewhere; both TLASes must have the
    /// same capacity (same buffer sizes) and this TLAS's last build the same instance count
    /// (O: WARP never finished an update into a TLAS last built with one instance less,
    /// `dynamic_check`; rebuild it first). Blocking, one submit.
    pub fn update_from(
        &mut self,
        gpu: &mut Gpu,
        src: &Tlas,
        descs: &[[u32; 16]],
    ) -> Result<(), String> {
        if descs.len() != src.count as usize
            || self.count != src.count
            || self.capacity != src.capacity
        {
            return Err(format!(
                "TLAS update with {} instances (capacity {}, last build {}) from a build of {} (capacity {})",
                descs.len(),
                self.capacity,
                self.count,
                src.count,
                src.capacity
            ));
        }
        if std::ptr::eq(self, src) || self.buffer.va == src.buffer.va {
            return Err("TLAS update_from itself (use refit)".into());
        }
        self.write(gpu, descs)?;
        gpu.transition(
            &self.scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let d = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: self.buffer.va,
            Inputs: tlas_inputs(TLAS_UPDATE, self.count, self.instances.va),
            SourceAccelerationStructureData: src.buffer.va,
            ScratchAccelerationStructureData: self.scratch.va,
        };
        // SAFETY: both TLASes are live with equal capacity (equal result / scratch sizes);
        // `src` was built with TLAS_TRACE (ALLOW_UPDATE) over `count` instances; scratch in
        // UNORDERED_ACCESS.
        unsafe { gpu.list.BuildRaytracingAccelerationStructure(&d, None) };
        gpu.uav_barrier();
        gpu.submit()
    }
}

/// Builds a TLAS (PREFER_FAST_TRACE | ALLOW_UPDATE) and measures full rebuilds, refits
/// (PERFORM_UPDATE, unchanged transforms) and PREFER_FAST_BUILD rebuilds, `reps` each
/// (`reps = 0`: no measurement, as [`Tlas::build`]).
pub fn build_tlas(gpu: &mut Gpu, descs: &[[u32; 16]], reps: u32) -> Result<Tlas, String> {
    let mut t = Tlas::build(gpu, descs, descs.len())?;
    if reps == 0 {
        return Ok(t);
    }
    gpu.transition(
        &t.scratch,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    // Order: fast-build rebuilds, fast-trace rebuilds, refits (of the fast-trace TLAS), then
    // a final fast-trace build that the ray dispatches use.
    let mut q = 0;
    for flags in [TLAS_FAST, TLAS_TRACE, TLAS_UPDATE] {
        for _ in 0..reps {
            gpu.timestamp(q);
            t.record(gpu, flags);
            gpu.timestamp(q + 1);
            q += 2;
        }
    }
    t.record(gpu, TLAS_TRACE);
    let ts = gpu.timestamps(q)?;
    let mut ms: Vec<f64> = ts
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| gpu.ticks_to_ms(p[0], p[1]))
        .collect();
    let r = reps as usize;
    t.stats.build_fast_build_ms = median(&mut ms[..r]);
    t.stats.build_fast_trace_ms = median(&mut ms[r..2 * r]);
    t.stats.update_ms = median(&mut ms[2 * r..]);
    Ok(t)
}
