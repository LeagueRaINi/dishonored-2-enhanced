//! Minimal D3D12 context: one direct queue, blocking submits plus one optional non-blocking
//! submit in flight ([`Gpu::submit_async`] / [`Gpu::poll`]), buffers bound as root
//! descriptors, an optional bindless texture table (`textures`), GPU timestamps.
//!
//! [`Gpu::new`] creates its own device (offline tools); [`Gpu::from_existing`] wraps a device
//! and queue created by a host (the in-game DLL) without DXGI enumeration or debug layer.

use std::collections::HashMap;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use crate::dxc::Dxc;
use crate::textures::TextureSet;

/// Plain data that may be viewed as bytes.
///
/// # Safety
/// Implementors have no padding and no invalid bit patterns.
pub unsafe trait Pod: Copy + 'static {}
// SAFETY: primitive numbers and arrays of them have no padding or invalid patterns.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u16 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl Pod for f32 {}
// SAFETY: arrays of `Pod` are contiguous without padding.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

pub fn bytes_of<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: `T: Pod` (no padding); the byte length equals the slice's size.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// Root layout of the spike kernels (`rt.hlsl` `RS`).
pub const SRVS: usize = 5;
pub const UAVS: usize = 4;
pub const CONSTANTS: usize = 36;

/// Root-parameter layout shared by a set of kernels, in this order: `constants` root
/// constants (b0), `srvs` SRV root descriptors (t0..), `uavs` UAV root descriptors (u0..),
/// then, with `table`, one SRV descriptor table (bindless textures, `space1`). The HLSL
/// `[RootSignature]` string of the kernels must declare exactly this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub constants: usize,
    pub srvs: usize,
    pub uavs: usize,
    pub table: bool,
}

impl Layout {
    pub const SPIKE: Self = Self {
        constants: CONSTANTS,
        srvs: SRVS,
        uavs: UAVS,
        table: false,
    };
}

pub struct Buffer {
    pub res: ID3D12Resource,
    pub size: u64,
    pub va: u64,
}

#[derive(Clone, Debug)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub vram_mib: u64,
    pub driver: String,
    pub raytracing_tier: i32,
    pub highest_shader_model: i32,
}

impl AdapterInfo {
    pub fn raytracing_tier_name(&self) -> String {
        match self.raytracing_tier {
            0 => "not supported".into(),
            10 => "1.0".into(),
            11 => "1.1 (inline RayQuery)".into(),
            t => format!("{}.{} ", t / 10, t % 10),
        }
    }

    pub fn shader_model_name(&self) -> String {
        format!(
            "{}.{}",
            self.highest_shader_model >> 4,
            self.highest_shader_model & 0xf
        )
    }

    pub fn inline_raytracing(&self) -> bool {
        self.raytracing_tier >= D3D12_RAYTRACING_TIER_1_1.0 && self.highest_shader_model >= 0x65
    }
}

pub struct Gpu {
    pub device: ID3D12Device5,
    pub queue: ID3D12CommandQueue,
    /// Two allocators: the list records into `allocators[cur]` while the other may still be
    /// executing the one submission allowed in flight.
    allocators: [ID3D12CommandAllocator; 2],
    cur: usize,
    pub list: ID3D12GraphicsCommandList4,
    fence: ID3D12Fence,
    fence_value: u64,
    /// Fence value of the non-blocking submission still executing.
    in_flight: Option<u64>,
    /// The adapter of the device (DXGI budget queries); [`Gpu::from_existing`] finds it by
    /// the device's LUID (`None` only when that lookup failed).
    adapter: Option<IDXGIAdapter3>,
    pub info: AdapterInfo,
    /// Timestamp ticks per second.
    pub ts_freq: f64,
    query_heap: ID3D12QueryHeap,
    query_readback: Buffer,
    /// Timestamp slots available to callers (`0..max_queries`); the heap holds
    /// [`SUBMIT_TIMING_QUERIES`] more above them for the per-submit GPU timing.
    pub max_queries: u32,
    /// Fence value of the submission whose begin/end timestamps are in timing pair `k`
    /// (pair = allocator index), not read yet.
    timed: [Option<u64>; 2],
    submit_stats: SubmitStats,
    /// Sleep after every blocking submit (lets other queues of the GPU run between ours).
    pause: std::time::Duration,
    /// Host log for progress lines of long operations (scene upload phases).
    log: Option<LogFn>,
    /// Free VRAM kept beyond our allocations ([`Gpu::vram_available`]).
    pub vram_policy: VramPolicy,
    dummy: Buffer,
    /// Source of [`Gpu::clear`] (created on first use).
    zeros: Option<Buffer>,
    /// Debug-layer message queue (only with `debug_layer`).
    info_queue: Option<ID3D12InfoQueue>,
}

/// Sink of [`Gpu::log`] lines.
type LogFn = Box<dyn FnMut(&str) + Send>;

/// Timestamp slots above [`Gpu::max_queries`] used to time every submitted list.
pub const SUBMIT_TIMING_QUERIES: u32 = 4;

/// GPU time of submitted command lists (begin/end timestamps in every list) since the last
/// [`Gpu::take_submit_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubmitStats {
    pub count: usize,
    pub total_ms: f64,
    pub max_ms: f64,
    /// The most recently completed submission.
    pub last_ms: f64,
}

impl SubmitStats {
    fn add(&mut self, ms: f64) {
        self.count += 1;
        self.total_ms += ms;
        self.max_ms = self.max_ms.max(ms);
        self.last_ms = ms;
    }
}

/// Local video memory of this process (DXGI `QueryVideoMemoryInfo`, segment group LOCAL).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VideoMemory {
    /// What the OS currently lets this process use without paging.
    pub budget: u64,
    /// What the process (every device on the adapter: the game's too) has allocated.
    pub usage: u64,
}

/// Free local video memory kept beyond our allocations: the larger of `headroom_bytes`
/// and `headroom_fraction` of the budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VramPolicy {
    pub headroom_bytes: u64,
    pub headroom_fraction: f64,
}

impl Default for VramPolicy {
    fn default() -> Self {
        Self {
            headroom_bytes: 1 << 30,
            headroom_fraction: 0.15,
        }
    }
}

impl VramPolicy {
    pub fn headroom(&self, budget: u64) -> u64 {
        self.headroom_bytes
            .max((budget as f64 * self.headroom_fraction) as u64)
    }
}

pub const MIB: f64 = 1024.0 * 1024.0;

/// `bytes` in MiB for log lines.
pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

/// `"; device removed: <reason>"` when `device` has been removed, else empty.
fn removed(device: &ID3D12Device5) -> String {
    // SAFETY: plain status query on a live device.
    match unsafe { device.GetDeviceRemovedReason() } {
        Ok(()) => String::new(),
        Err(e) => format!("; device removed: {e}"),
    }
}

/// Name, ids, dedicated VRAM and UMD version of `adapter`.
fn adapter_info(
    adapter: &IDXGIAdapter3,
    raytracing_tier: i32,
    highest_shader_model: i32,
) -> Result<AdapterInfo, String> {
    // SAFETY: plain queries on a live adapter.
    unsafe {
        let desc = adapter.GetDesc1().map_err(|e| e.to_string())?;
        let driver = match adapter.CheckInterfaceSupport(&IDXGIDevice::IID) {
            Ok(v) => {
                let umd = v as u64;
                format!(
                    "{}.{}.{}.{}",
                    umd >> 48,
                    (umd >> 32) & 0xffff,
                    (umd >> 16) & 0xffff,
                    umd & 0xffff
                )
            }
            Err(e) => format!("unknown ({e})"),
        };
        Ok(AdapterInfo {
            name: wide(&desc.Description),
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            vram_mib: (desc.DedicatedVideoMemory >> 20) as u64,
            driver,
            raytracing_tier,
            highest_shader_model,
        })
    }
}

// The in-game host moves a `Gpu` (and scenes / sessions holding its buffers) into a worker
// thread: checked at compile time. D3D12 devices, queues, resources and fences are
// free-threaded; the command list and allocators are only touched through `&mut self`
// or `&self` recording calls of the owning thread (`Gpu` is not shared, only moved).
const _: fn() = || {
    fn send<T: Send>() {}
    send::<Gpu>();
    send::<Buffer>();
    send::<Programs>();
};

/// See [`Gpu::guard`].
pub struct SubmitGuard {
    fence: ID3D12Fence,
    value: u64,
}

impl Drop for SubmitGuard {
    fn drop(&mut self) {
        // SAFETY: a null event blocks until the fence reaches the value; on device removal
        // the fence reports u64::MAX and the call returns.
        unsafe {
            if self.fence.GetCompletedValue() < self.value {
                let _ = self
                    .fence
                    .SetEventOnCompletion(self.value, HANDLE::default());
            }
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // The allocators and the list must outlive a submission in flight.
        let _ = self.wait_idle();
    }
}

fn heap_props(t: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: t,
        ..Default::default()
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn wide(s: &[u16]) -> String {
    let n = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf16_lossy(&s[..n])
}

/// Debug-layer source: the Windows "Graphics Tools" feature is not installed on the target
/// machine, so the debug layer comes from an Agility SDK redistributable
/// (`d3d12SDKLayers.dll` + `D3D12Core.dll`) loaded through an independent device factory
/// (`ID3D12SDKConfiguration1::CreateDeviceFactory`): it affects only devices created by
/// that factory, the exe exports nothing.
pub struct DebugLayer<'a> {
    pub sdk_version: u32,
    /// Directory holding `D3D12Core.dll` and `d3d12SDKLayers.dll`.
    pub sdk_dir: &'a std::path::Path,
}

impl Gpu {
    /// Creates the device on the first high-performance hardware adapter, or on WARP (the
    /// software rasterizer: GPU-driver-independent, for validation runs); with `debug`, the
    /// D3D12 debug layer and GPU-based validation are enabled.
    pub fn new(debug: Option<DebugLayer>, warp: bool) -> Result<Self, String> {
        let debug_layer = debug.is_some();
        // SAFETY (whole function): D3D12/DXGI object creation and feature queries with valid
        // descriptors and correctly sized out structures.
        unsafe {
            let device_factory = match &debug {
                None => None,
                Some(d) => {
                    let mut cfg: Option<ID3D12SDKConfiguration1> = None;
                    D3D12GetInterface(&CLSID_D3D12SDKConfiguration, &mut cfg)
                        .map_err(|e| format!("SDK configuration: {e}"))?;
                    let mut dir = d.sdk_dir.to_string_lossy().into_owned();
                    if !dir.ends_with(['\\', '/']) {
                        dir.push('\\');
                    }
                    let dir = std::ffi::CString::new(dir).map_err(|e| e.to_string())?;
                    let f: ID3D12DeviceFactory = cfg
                        .ok_or("no SDK configuration")?
                        .CreateDeviceFactory(
                            d.sdk_version,
                            windows::core::PCSTR(dir.as_ptr().cast()),
                        )
                        .map_err(|e| {
                            format!(
                                "Agility SDK {} at {}: {e}",
                                d.sdk_version,
                                d.sdk_dir.display()
                            )
                        })?;
                    // Debug layer + GPU-based validation (catches out-of-bounds root-descriptor
                    // accesses and state errors on the GPU before they can hang the device).
                    let dbg: ID3D12Debug1 = f
                        .GetConfigurationInterface(&CLSID_D3D12Debug)
                        .map_err(|e| format!("debug layer: {e}"))?;
                    dbg.EnableDebugLayer();
                    dbg.SetEnableGPUBasedValidation(true);
                    Some(f)
                }
            };
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| format!("CreateDXGIFactory2: {e}"))?;
            let mut chosen = None;
            if warp {
                let a: IDXGIAdapter3 = factory
                    .EnumWarpAdapter()
                    .map_err(|e| format!("WARP: {e}"))?;
                let d = a.GetDesc1().map_err(|e| e.to_string())?;
                chosen = Some((a, d));
            }
            for i in 0.. {
                if chosen.is_some() {
                    break;
                }
                let Ok(a) = factory.EnumAdapterByGpuPreference::<IDXGIAdapter3>(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                let d = a.GetDesc1().map_err(|e| e.to_string())?;
                if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    chosen = Some((a, d));
                }
            }
            let (adapter, _) = chosen.ok_or("no hardware adapter")?;
            let mut device: Option<ID3D12Device5> = None;
            match &device_factory {
                Some(f) => f
                    .CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                    .map_err(|e| format!("debug CreateDevice: {e}"))?,
                None => D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                    .map_err(|e| format!("D3D12CreateDevice: {e}"))?,
            }
            let device = device.ok_or("no device")?;
            let (raytracing_tier, highest_shader_model) = Self::features(&device)?;
            let info = adapter_info(&adapter, raytracing_tier, highest_shader_model)?;
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(|e| format!("queue: {e}"))?;
            let info_queue = if debug_layer {
                Some(
                    device
                        .cast::<ID3D12InfoQueue>()
                        .map_err(|e| format!("info queue: {e}"))?,
                )
            } else {
                None
            };
            Self::wrap(device, queue, Some(adapter), info, info_queue)
        }
    }

    /// Wraps a device and a direct (or compute) queue created by the host, e.g. the in-game
    /// DLL's D3D12 device on the game's adapter. No debug layer. The DXGI adapter (name,
    /// VRAM, driver, [`Gpu::video_memory`] budget queries) is found by the device's LUID;
    /// when that lookup fails `info` names the LUID only and the budget is unknown. Creates
    /// its own allocators, list, fence and timestamp queries on `device`. Fails when the
    /// device lacks inline ray tracing (tier 1.1, SM 6.5).
    pub fn from_existing(device: ID3D12Device5, queue: ID3D12CommandQueue) -> Result<Self, String> {
        let (raytracing_tier, highest_shader_model) = Self::features(&device)?;
        // SAFETY: plain queries on live objects.
        let (luid, kind) = unsafe { (device.GetAdapterLuid(), queue.GetDesc().Type) };
        if kind != D3D12_COMMAND_LIST_TYPE_DIRECT && kind != D3D12_COMMAND_LIST_TYPE_COMPUTE {
            return Err(format!(
                "queue type {} (direct or compute required)",
                kind.0
            ));
        }
        // SAFETY: factory creation and an adapter lookup by LUID.
        let adapter: Option<IDXGIAdapter3> = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid(luid))
                .ok()
        };
        let info = adapter
            .as_ref()
            .and_then(|a| adapter_info(a, raytracing_tier, highest_shader_model).ok())
            .unwrap_or_else(|| AdapterInfo {
                name: format!(
                    "host device (adapter LUID {:08x}:{:08x} not found)",
                    luid.HighPart, luid.LowPart
                ),
                vendor_id: 0,
                device_id: 0,
                vram_mib: 0,
                driver: "unknown (host device)".into(),
                raytracing_tier,
                highest_shader_model,
            });
        if !info.inline_raytracing() {
            return Err(format!(
                "inline ray tracing (tier 1.1, SM 6.5) not supported: tier {}, SM {}",
                info.raytracing_tier_name(),
                info.shader_model_name()
            ));
        }
        Self::wrap_typed(device, queue, kind, adapter, info, None)
    }

    /// (raytracing tier, highest shader model) of `device`.
    fn features(device: &ID3D12Device5) -> Result<(i32, i32), String> {
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: correctly sized out structures.
        unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut o5).cast(),
                    size_of_val(&o5) as u32,
                )
                .map_err(|e| format!("OPTIONS5: {e}"))?;
            // Highest shader model: ask from the newest known downwards.
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL::default();
            for v in (D3D_SHADER_MODEL_6_0.0..=0x69).rev() {
                sm.HighestShaderModel = D3D_SHADER_MODEL(v);
                if device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_SHADER_MODEL,
                        (&raw mut sm).cast(),
                        size_of_val(&sm) as u32,
                    )
                    .is_ok()
                {
                    break;
                }
            }
            Ok((o5.RaytracingTier.0, sm.HighestShaderModel.0))
        }
    }

    fn wrap(
        device: ID3D12Device5,
        queue: ID3D12CommandQueue,
        adapter: Option<IDXGIAdapter3>,
        info: AdapterInfo,
        info_queue: Option<ID3D12InfoQueue>,
    ) -> Result<Self, String> {
        Self::wrap_typed(
            device,
            queue,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            adapter,
            info,
            info_queue,
        )
    }

    fn wrap_typed(
        device: ID3D12Device5,
        queue: ID3D12CommandQueue,
        kind: D3D12_COMMAND_LIST_TYPE,
        adapter: Option<IDXGIAdapter3>,
        info: AdapterInfo,
        info_queue: Option<ID3D12InfoQueue>,
    ) -> Result<Self, String> {
        // SAFETY (whole function): object creation with valid descriptors on a live device.
        unsafe {
            let allocator = || -> Result<ID3D12CommandAllocator, String> {
                device
                    .CreateCommandAllocator(kind)
                    .map_err(|e| format!("allocator: {e}"))
            };
            let allocators = [allocator()?, allocator()?];
            let list: ID3D12GraphicsCommandList4 = device
                .CreateCommandList(0, kind, &allocators[0], None)
                .map_err(|e| format!("command list: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("fence: {e}"))?;
            let ts_freq = queue.GetTimestampFrequency().map_err(|e| e.to_string())? as f64;
            let max_queries = 4096 - SUBMIT_TIMING_QUERIES;
            let heap_queries = max_queries + SUBMIT_TIMING_QUERIES;
            let mut query_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: heap_queries,
                        NodeMask: 0,
                    },
                    &mut query_heap,
                )
                .map_err(|e| format!("query heap: {e}"))?;
            let mut gpu = Self {
                query_readback: Self::create(
                    &device,
                    D3D12_HEAP_TYPE_READBACK,
                    u64::from(heap_queries) * 8,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )?,
                dummy: Self::create(
                    &device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    4096,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )?,
                device,
                queue,
                allocators,
                cur: 0,
                list,
                fence,
                fence_value: 0,
                in_flight: None,
                adapter,
                info,
                ts_freq,
                query_heap: query_heap.ok_or("no query heap")?,
                max_queries,
                timed: [None; 2],
                submit_stats: SubmitStats::default(),
                pause: std::time::Duration::ZERO,
                log: None,
                vram_policy: VramPolicy::default(),
                zeros: None,
                info_queue,
            };
            gpu.begin_list();
            // Source of `clear`: explicitly zeroed (one 4 MiB copy).
            gpu.zeros = Some(gpu.upload(&vec![0u8; 4 << 20], false)?);
            gpu.take_submit_stats();
            Ok(gpu)
        }
    }

    fn create(
        device: &ID3D12Device5,
        heap: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Buffer, String> {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: valid descriptors and out pointer.
        unsafe {
            device.CreateCommittedResource(
                &heap_props(heap),
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size, flags),
                state,
                None,
                &mut res,
            )
        }
        .map_err(|e| {
            format!(
                "CreateCommittedResource ({size} bytes): {e}{}",
                removed(device)
            )
        })?;
        let res = res.ok_or("no resource")?;
        // SAFETY: buffers have a GPU virtual address.
        let va = unsafe { res.GetGPUVirtualAddress() };
        Ok(Buffer {
            res,
            size: size.max(256),
            va,
        })
    }

    /// Default-heap buffer in COMMON (implicit promotion), optionally UAV-capable.
    pub fn buffer(&self, size: u64, uav: bool) -> Result<Buffer, String> {
        let flags = if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        };
        Self::create(
            &self.device,
            D3D12_HEAP_TYPE_DEFAULT,
            size,
            flags,
            D3D12_RESOURCE_STATE_COMMON,
        )
    }

    /// Acceleration-structure storage (fixed state RAYTRACING_ACCELERATION_STRUCTURE).
    pub fn accel_buffer(&self, size: u64) -> Result<Buffer, String> {
        Self::create(
            &self.device,
            D3D12_HEAP_TYPE_DEFAULT,
            size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        )
    }

    /// Upload-heap buffer filled with `data` (CPU-written, GPU-readable).
    pub fn upload_heap<T: Pod>(&self, data: &[T]) -> Result<Buffer, String> {
        let bytes = bytes_of(data);
        let b = Self::create(
            &self.device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an upload buffer of at least `bytes.len()` bytes, mapped for writing.
        unsafe {
            b.res
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))
                .map_err(|e| format!("Map: {e}{}", self.removed_reason()))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            b.res.Unmap(0, None);
        }
        Ok(b)
    }

    /// Rewrites the start of the upload-heap buffer `b` with `data`. The caller makes sure
    /// no GPU work reading `b` is in flight.
    pub fn write_mapped<T: Pod>(&self, b: &Buffer, data: &[T]) -> Result<(), String> {
        let bytes = bytes_of(data);
        if bytes.len() as u64 > b.size {
            return Err(format!(
                "{} bytes into a {}-byte buffer",
                bytes.len(),
                b.size
            ));
        }
        let mut ptr = std::ptr::null_mut();
        // SAFETY: an upload buffer of at least `bytes.len()` bytes (checked), mapped for
        // writing; idle per the caller's contract.
        unsafe {
            b.res
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))
                .map_err(|e| format!("Map: {e}{}", self.removed_reason()))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            b.res.Unmap(0, None);
        }
        Ok(())
    }

    /// Records a copy of `data` into the default-heap buffer `dst` (in COMMON) at byte
    /// `offset` through a new upload buffer; the caller keeps the returned staging buffer
    /// until the submit of the open list has completed. None for empty `data`.
    pub fn record_upload<T: Pod>(
        &self,
        dst: &Buffer,
        offset: u64,
        data: &[T],
    ) -> Result<Option<Buffer>, String> {
        let n = bytes_of(data).len() as u64;
        if n == 0 {
            return Ok(None);
        }
        if offset.checked_add(n).is_none_or(|end| end > dst.size) {
            return Err(format!(
                "copy of {n} bytes at {offset} outside a {}-byte buffer",
                dst.size
            ));
        }
        let staging = self.upload_heap(data)?;
        // SAFETY: both buffers live; the range lies inside `dst` (checked) and `staging`;
        // COMMON promotes to COPY_DEST / the upload heap is GENERIC_READ.
        unsafe {
            self.list
                .CopyBufferRegion(&dst.res, offset, &staging.res, 0, n);
        }
        Ok(Some(staging))
    }

    /// Default-heap buffer initialised with `data` (blocking copy).
    pub fn upload<T: Pod>(&mut self, data: &[T], uav: bool) -> Result<Buffer, String> {
        let staging = self.upload_heap(data)?;
        let b = self.buffer(staging.size, uav)?;
        // SAFETY: both buffers are live; COMMON promotes to COPY_DEST/COPY_SOURCE.
        unsafe {
            self.list
                .CopyBufferRegion(&b.res, 0, &staging.res, 0, bytes_of(data).len() as u64);
        }
        self.submit()?;
        Ok(b)
    }

    /// Default-heap buffer initialised with `data` through a reused upload buffer of at most
    /// `chunk` bytes: one blocking submit per chunk (short submits, bounded staging memory).
    pub fn upload_chunked<T: Pod>(
        &mut self,
        data: &[T],
        uav: bool,
        chunk: u64,
    ) -> Result<Buffer, String> {
        let bytes = bytes_of(data);
        let len = bytes.len() as u64;
        let chunk = chunk.clamp(1 << 16, len.max(1 << 16));
        if len <= chunk {
            return self.upload(data, uav);
        }
        let b = self.buffer(len, uav)?;
        let staging = Self::create(
            &self.device,
            D3D12_HEAP_TYPE_UPLOAD,
            chunk,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut off = 0u64;
        while off < len {
            let n = chunk.min(len - off);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: the staging buffer holds `chunk >= n` bytes and is idle (every previous
            // copy from it finished in a blocking submit); `off + n <= len`.
            unsafe {
                staging
                    .res
                    .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr))
                    .map_err(|e| format!("Map: {e}{}", self.removed_reason()))?;
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(off as usize),
                    ptr.cast::<u8>(),
                    n as usize,
                );
                staging.res.Unmap(0, None);
                self.list.CopyBufferRegion(&b.res, off, &staging.res, 0, n);
            }
            self.submit()?;
            off += n;
        }
        Ok(b)
    }

    /// Records copies that zero `b` (from the zero buffer uploaded at creation; committed
    /// resources are not reliably zeroed: WARP returned recycled memory). `b` must be in
    /// COMMON at this point of the list (buffers decay to COMMON after every submit).
    pub fn clear(&self, b: &Buffer) -> Result<(), String> {
        let z = self.zeros.as_ref().ok_or("no zero buffer")?;
        let mut off = 0;
        while off < b.size {
            let n = z.size.min(b.size - off);
            // SAFETY: both buffers live, ranges inside them; COMMON promotes for copies.
            unsafe { self.list.CopyBufferRegion(&b.res, off, &z.res, 0, n) };
            off += n;
        }
        Ok(())
    }

    /// Clears `b` to zero (blocking).
    pub fn zero(&mut self, b: &Buffer) -> Result<(), String> {
        self.clear(b)?;
        self.submit()
    }

    /// Copies the first `len` elements of `b` back to the CPU (blocking).
    pub fn read<T: Pod + Default>(&mut self, b: &Buffer, len: usize) -> Result<Vec<T>, String> {
        let bytes = (len * size_of::<T>()) as u64;
        let rb = Self::create(
            &self.device,
            D3D12_HEAP_TYPE_READBACK,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: `b` is in COMMON at list start (buffers decay after every submit).
        unsafe { self.list.CopyBufferRegion(&rb.res, 0, &b.res, 0, bytes) };
        self.submit()?;
        let mut out = vec![T::default(); len];
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the readback buffer holds `bytes` bytes written by the finished copy.
        unsafe {
            rb.res
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: bytes as usize,
                    }),
                    Some(&mut ptr),
                )
                .map_err(|e| format!("Map: {e}{}", self.removed_reason()))?;
            std::ptr::copy_nonoverlapping(
                ptr.cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                bytes as usize,
            );
            rb.res.Unmap(0, None);
        }
        Ok(out)
    }

    /// First slot of timing pair `k` (two slots above `max_queries` per allocator).
    fn timing_slot(&self, k: usize) -> u32 {
        self.max_queries + 2 * k as u32
    }

    /// Records the begin timestamp of the freshly opened list (timing pair `cur`).
    fn begin_list(&self) {
        // SAFETY: the slot lies in the heap (max_queries + SUBMIT_TIMING_QUERIES slots).
        unsafe {
            self.list.EndQuery(
                &self.query_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                self.timing_slot(self.cur),
            );
        }
    }

    /// Closes and executes the open list (with its end timestamp and the resolve of its
    /// timing pair) and signals the fence; returns the fence value.
    fn execute(&mut self) -> Result<u64, String> {
        self.fence_value += 1;
        let v = self.fence_value;
        let slot = self.timing_slot(self.cur);
        // SAFETY: the list only references live resources; the timing slots lie in the heap
        // and the readback buffer (COPY_DEST) holds them.
        unsafe {
            self.list
                .EndQuery(&self.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, slot + 1);
            self.list.ResolveQueryData(
                &self.query_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                slot,
                2,
                &self.query_readback.res,
                u64::from(slot) * 8,
            );
            self.list
                .Close()
                .map_err(|e| format!("Close: {e}{}", self.debug_messages().1))?;
            let list: ID3D12CommandList = self.list.cast().map_err(|e| e.to_string())?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.queue
                .Signal(&self.fence, v)
                .map_err(|e| format!("Signal: {e}{}", self.removed_reason()))?;
        }
        self.timed[self.cur] = Some(v);
        Ok(v)
    }

    /// Blocks until the fence reaches `v` and checks for device removal.
    fn wait_value(&mut self, v: u64) -> Result<(), String> {
        // SAFETY: a null event makes the call block until the fence reaches `v`.
        unsafe {
            self.fence
                .SetEventOnCompletion(v, HANDLE::default())
                .map_err(|e| format!("wait: {e}{}", self.removed_reason()))?;
            self.device
                .GetDeviceRemovedReason()
                .map_err(|e| format!("device removed: {e}{}", self.debug_messages().1))?;
        }
        if self.in_flight.is_some_and(|f| f <= v) {
            self.in_flight = None;
        }
        for k in 0..2 {
            if self.timed[k].is_some_and(|f| f <= v) {
                self.timed[k] = None;
                let slot = self.timing_slot(k);
                let ts = self.read_query_slots(slot, 2)?;
                let ms = self.ticks_to_ms(ts[0], ts[1]);
                self.submit_stats.add(ms);
            }
        }
        let (errors, text) = self.debug_messages();
        if !text.is_empty() {
            eprint!("{text}");
        }
        if errors > 0 {
            return Err(format!("{errors} debug-layer error(s), see above"));
        }
        Ok(())
    }

    /// Reopens the list on allocator `cur` (whose previous work must be complete).
    fn reopen(&mut self) -> Result<(), String> {
        let a = &self.allocators[self.cur];
        // SAFETY: every list recorded with `a` has finished executing (caller's contract).
        unsafe {
            a.Reset().map_err(|e| format!("allocator reset: {e}"))?;
            self.list
                .Reset(a, None)
                .map_err(|e| format!("list reset: {e}"))?;
        }
        self.begin_list();
        Ok(())
    }

    /// Closes and submits the list, waits for the GPU (including any submission in flight),
    /// reopens the list, then sleeps for the [`Gpu::set_pause`] time (other queues of the
    /// GPU, e.g. the game's, get it between our blocking submits).
    pub fn submit(&mut self) -> Result<(), String> {
        let v = self.execute()?;
        self.wait_value(v)?;
        self.reopen()?;
        if !self.pause.is_zero() {
            std::thread::sleep(self.pause);
        }
        Ok(())
    }

    /// Sleep after every blocking [`Gpu::submit`] (0 = none, the default).
    pub fn set_pause(&mut self, pause: std::time::Duration) {
        self.pause = pause;
    }

    /// GPU time of the submissions completed since the last call (and resets it).
    pub fn take_submit_stats(&mut self) -> SubmitStats {
        std::mem::take(&mut self.submit_stats)
    }

    /// GPU ms of the most recently completed submission.
    pub fn last_submit_ms(&self) -> f64 {
        self.submit_stats.last_ms
    }

    /// Sends progress lines of long operations (scene upload phases) to `f`.
    pub fn set_log(&mut self, f: impl FnMut(&str) + Send + 'static) {
        self.log = Some(Box::new(f));
    }

    /// A progress line to the [`Gpu::set_log`] sink (dropped without one).
    pub fn log(&mut self, line: &str) {
        if let Some(f) = self.log.as_mut() {
            f(line);
        }
    }

    /// Submits the list without waiting and reopens it on the other allocator; returns the
    /// fence value to poll ([`Gpu::poll`], [`Gpu::completed`]). At most one submission is in
    /// flight: a still-running previous one is waited for first (the only blocking case).
    pub fn submit_async(&mut self) -> Result<u64, String> {
        if let Some(prev) = self.in_flight {
            self.wait_value(prev)?;
        }
        let v = self.execute()?;
        self.in_flight = Some(v);
        // The other allocator's lists finished: they belong to submissions <= `prev`.
        self.cur ^= 1;
        self.reopen()?;
        Ok(v)
    }

    /// Non-blocking: `true` when no submission is in flight any more. Errors on device
    /// removal (the fence then reports `u64::MAX`).
    pub fn poll(&mut self) -> Result<bool, String> {
        let Some(v) = self.in_flight else {
            return Ok(true);
        };
        if self.completed(v)? {
            self.wait_value(v)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Non-blocking: whether the fence passed `v` (a value returned by a submit).
    pub fn completed(&self, v: u64) -> Result<bool, String> {
        // SAFETY: plain queries on live objects.
        let (done, removed) = unsafe {
            (
                self.fence.GetCompletedValue(),
                self.device.GetDeviceRemovedReason(),
            )
        };
        removed.map_err(|e| format!("device removed: {e}"))?;
        Ok(done >= v)
    }

    /// Blocks until the submission in flight (if any) has finished.
    pub fn wait_idle(&mut self) -> Result<(), String> {
        match self.in_flight {
            Some(v) => self.wait_value(v),
            None => Ok(()),
        }
    }

    /// Whether a non-blocking submission is still executing (as of the last poll).
    pub fn busy(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Drains the debug layer's stored messages: (error/corruption count, text).
    fn debug_messages(&self) -> (usize, String) {
        let Some(q) = &self.info_queue else {
            return (0, String::new());
        };
        let mut errors = 0;
        let mut text = String::new();
        // SAFETY: two-call message retrieval into a buffer of the reported size (8-aligned).
        unsafe {
            for i in 0..q.GetNumStoredMessages() {
                let mut len = 0usize;
                if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
                    continue;
                }
                let mut buf = vec![0u64; len.div_ceil(8)];
                let m = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if q.GetMessage(i, Some(m), &mut len).is_err() {
                    continue;
                }
                let m = &*m;
                let sev = m.Severity;
                if sev == D3D12_MESSAGE_SEVERITY_INFO || sev == D3D12_MESSAGE_SEVERITY_MESSAGE {
                    continue;
                }
                if sev == D3D12_MESSAGE_SEVERITY_ERROR || sev == D3D12_MESSAGE_SEVERITY_CORRUPTION {
                    errors += 1;
                }
                let d = std::slice::from_raw_parts(
                    m.pDescription,
                    m.DescriptionByteLength.saturating_sub(1),
                );
                text.push_str(&format!(
                    "d3d12 [{}] {}\n",
                    sev.0,
                    String::from_utf8_lossy(d)
                ));
            }
            q.ClearStoredMessages();
        }
        (errors, text)
    }

    pub fn timestamp(&self, index: u32) {
        // SAFETY: index < max_queries is the caller's contract (checked in `timestamps`).
        unsafe {
            self.list
                .EndQuery(&self.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, index)
        };
    }

    /// Resolves timestamps `0..n` recorded in the open list, submits and returns ticks.
    pub fn timestamps(&mut self, n: u32) -> Result<Vec<u64>, String> {
        self.resolve_timestamps(0, n)?;
        self.submit()?;
        self.read_timestamps(0, n)
    }

    fn check_queries(&self, first: u32, n: u32) -> Result<(), String> {
        if u64::from(first) + u64::from(n) > u64::from(self.max_queries) {
            return Err(format!("timestamps {first}+{n} > {}", self.max_queries));
        }
        Ok(())
    }

    /// Records the resolve of timestamps `first..first + n` into the open list (to the same
    /// slots of the readback buffer; read them with [`Gpu::read_timestamps`] once that
    /// submission has completed). Non-blocking users take slots from the top of the heap so
    /// blocking [`Gpu::timestamps`] calls (slots from 0) never overwrite them.
    pub fn resolve_timestamps(&mut self, first: u32, n: u32) -> Result<(), String> {
        self.check_queries(first, n)?;
        // SAFETY: queries first..first+n were written in this list; readback in COPY_DEST.
        unsafe {
            self.list.ResolveQueryData(
                &self.query_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                first,
                n,
                &self.query_readback.res,
                u64::from(first) * 8,
            );
        }
        Ok(())
    }

    /// Ticks of the last completed [`Gpu::resolve_timestamps`] of `first..first + n`.
    pub fn read_timestamps(&self, first: u32, n: u32) -> Result<Vec<u64>, String> {
        self.check_queries(first, n)?;
        self.read_query_slots(first, n)
    }

    /// Resolved ticks of heap slots `first..first + n` (callers' slots or timing pairs).
    fn read_query_slots(&self, first: u32, n: u32) -> Result<Vec<u64>, String> {
        let mut out = vec![0u64; n as usize];
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the completed resolve wrote `n` u64 values from slot `first`, inside the
        // readback buffer (heap size x 8 bytes).
        unsafe {
            self.query_readback
                .res
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: first as usize * 8,
                        End: (first + n) as usize * 8,
                    }),
                    Some(&mut ptr),
                )
                .map_err(|e| format!("Map: {e}{}", self.removed_reason()))?;
            std::ptr::copy_nonoverlapping(
                ptr.cast::<u64>().add(first as usize),
                out.as_mut_ptr(),
                n as usize,
            );
            self.query_readback.res.Unmap(0, None);
        }
        Ok(out)
    }

    /// A guard that, when dropped, blocks until every submission made so far has finished.
    /// Owners of buffers used by a non-blocking submission keep one, so dropping them never
    /// frees memory the GPU is still accessing.
    pub fn guard(&self) -> SubmitGuard {
        SubmitGuard {
            fence: self.fence.clone(),
            value: self.fence_value,
        }
    }

    pub fn ticks_to_ms(&self, a: u64, b: u64) -> f64 {
        b.saturating_sub(a) as f64 * 1e3 / self.ts_freq
    }

    /// Global UAV barrier (orders acceleration-structure builds and dispatches).
    pub fn uav_barrier(&self) {
        let b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        // SAFETY: a global UAV barrier references no resource.
        unsafe { self.list.ResourceBarrier(&[b]) };
    }

    pub fn transition(
        &self,
        b: &Buffer,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        self.transition_resource(&b.res, before, after);
    }

    /// Records a copy of `buf` (rows of `row_pitch` bytes, a multiple of 256, from offset 0)
    /// into subresource 0 of the 2D texture `tex` (`width x height` texels of `format`)
    /// after checking `tex`'s description and the buffer size. The caller handles the
    /// texture's states (COPY_DEST during the copy).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer_to_texture(
        &self,
        buf: &Buffer,
        row_pitch: u64,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        typeless: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        width: u32,
        height: u32,
        tex: &ID3D12Resource,
    ) -> Result<(), String> {
        self.texture_copy(buf, row_pitch, format, typeless, width, height, tex, true)
    }

    /// The reverse of [`Gpu::copy_buffer_to_texture`]: subresource 0 of `tex` into `buf`
    /// (same checks). The caller handles the texture's states (COPY_SOURCE during the copy).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_to_buffer(
        &self,
        tex: &ID3D12Resource,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        typeless: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        width: u32,
        height: u32,
        buf: &Buffer,
        row_pitch: u64,
    ) -> Result<(), String> {
        self.texture_copy(buf, row_pitch, format, typeless, width, height, tex, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn texture_copy(
        &self,
        buf: &Buffer,
        row_pitch: u64,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        typeless: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        width: u32,
        height: u32,
        tex: &ID3D12Resource,
        to_texture: bool,
    ) -> Result<(), String> {
        // SAFETY: plain query on a live resource.
        let d = unsafe { tex.GetDesc() };
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || d.Width != u64::from(width)
            || d.Height != height
            || d.DepthOrArraySize != 1
            || (d.Format != format && d.Format != typeless)
        {
            return Err(format!(
                "texture {}x{} format {} (dimension {}), expected {width}x{height} format {}",
                d.Width, d.Height, d.Format.0, d.Dimension.0, format.0
            ));
        }
        if !row_pitch.is_multiple_of(256) || buf.size < row_pitch * u64::from(height) {
            return Err(format!(
                "buffer {} bytes / pitch {row_pitch} for {height} rows",
                buf.size
            ));
        }
        let texture = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let buffer = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(buf.res.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: row_pitch as u32,
                    },
                },
            },
        };
        let (dst, src) = if to_texture {
            (&texture, &buffer)
        } else {
            (&buffer, &texture)
        };
        // SAFETY: both resources live; footprint inside the buffer (checked above) and equal
        // to the texture's extent.
        unsafe { self.list.CopyTextureRegion(dst, 0, 0, 0, src, None) };
        drop(std::mem::ManuallyDrop::into_inner(texture.pResource));
        drop(std::mem::ManuallyDrop::into_inner(buffer.pResource));
        Ok(())
    }

    /// Drops whatever was recorded into the open list since the last submit (after a
    /// recording error), leaving an empty open list.
    pub fn discard(&mut self) -> Result<(), String> {
        // SAFETY: the open list was never submitted; resetting it onto the current allocator
        // (whose earlier lists have completed or were reset) is valid.
        unsafe {
            let _ = self.list.Close();
            self.list
                .Reset(&self.allocators[self.cur], None)
                .map_err(|e| format!("list reset: {e}"))?;
        }
        self.begin_list();
        Ok(())
    }

    pub fn transition_resource(
        &self,
        res: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        // SAFETY: `res` is live; the caller states its current state.
        unsafe { self.list.ResourceBarrier(std::slice::from_ref(&barrier)) };
        // Release the reference held by the barrier.
        let D3D12_RESOURCE_BARRIER { Anonymous, .. } = barrier;
        // SAFETY: the union holds the Transition variant written above.
        let t = std::mem::ManuallyDrop::into_inner(unsafe { Anonymous.Transition });
        drop(std::mem::ManuallyDrop::into_inner(t.pResource));
    }

    /// `"; device removed: <reason>"` when the device has been removed, else empty.
    pub fn removed_reason(&self) -> String {
        removed(&self.device)
    }

    /// Budget and usage of local video memory for this process (`None` without the DXGI
    /// adapter or when the query fails).
    pub fn video_memory(&self) -> Option<VideoMemory> {
        let adapter = self.adapter.as_ref()?;
        let mut m = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: valid out structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut m) }.ok()?;
        Some(VideoMemory {
            budget: m.Budget,
            usage: m.CurrentUsage,
        })
    }

    /// Bytes of local video memory in use by this process (0 when unknown).
    pub fn vram_usage(&self) -> u64 {
        self.video_memory().map_or(0, |m| m.usage)
    }

    /// Bytes we may still allocate: budget - usage - headroom ([`Gpu::vram_policy`]);
    /// `None` when the budget is unknown.
    pub fn vram_available(&self) -> Option<u64> {
        let m = self.video_memory()?;
        Some(
            m.budget
                .saturating_sub(m.usage)
                .saturating_sub(self.vram_policy.headroom(m.budget)),
        )
    }

    /// `"budget B MiB, in use U MiB, headroom H MiB"` (or "unknown") for log lines.
    pub fn vram_text(&self) -> String {
        match self.video_memory() {
            Some(m) => format!(
                "VRAM budget {:.0} MiB, in use {:.0} MiB, headroom {:.0} MiB",
                mib(m.budget),
                mib(m.usage),
                mib(self.vram_policy.headroom(m.budget))
            ),
            None => "VRAM budget unknown".into(),
        }
    }

    /// Errors when `need` bytes do not fit [`Gpu::vram_available`] (no check when the
    /// budget is unknown).
    pub fn check_vram(&self, need: u64, what: &str) -> Result<(), String> {
        match self.vram_available() {
            Some(a) if need > a => Err(format!(
                "not enough video memory for {what}: {:.0} MiB needed, {:.0} MiB available ({})",
                mib(need),
                mib(a),
                self.vram_text()
            )),
            _ => Ok(()),
        }
    }

    pub fn dummy_va(&self) -> u64 {
        self.dummy.va
    }
}

/// A kernel entry point and the bytes per dispatched pixel each root descriptor must hold
/// (0 = not read per pixel; such a slot may be unbound unless listed in `required_srvs`).
/// The slices have the layout's SRV / UAV counts.
pub struct Entry<'a> {
    pub name: &'a str,
    /// SRV slots that must be bound regardless of size (e.g. TLAS, primitive data).
    pub required_srvs: &'a [usize],
    pub srv_bytes_per_pixel: &'a [u64],
    pub uav_bytes_per_pixel: &'a [u64],
}

struct Kernel {
    pso: ID3D12PipelineState,
    required_srvs: Vec<usize>,
    srv_bytes: Vec<u64>,
    uav_bytes: Vec<u64>,
}

/// Compute programs sharing one root signature (declared in the HLSL source).
pub struct Programs {
    root: ID3D12RootSignature,
    layout: Layout,
    kernels: HashMap<String, Kernel>,
}

impl Programs {
    /// Compiles every entry of `source` with run-time DXC.
    pub fn new(
        gpu: &Gpu,
        dxc: &Dxc,
        source: &str,
        layout: Layout,
        entries: &[Entry],
    ) -> Result<Self, String> {
        Self::build(gpu, layout, entries, |name| {
            dxc.compile(source, name, "cs_6_5")
        })
    }

    /// Builds the entries from precompiled DXIL (`dxil(name)`, e.g. [`crate::ingame::dxil`]).
    pub fn from_dxil(
        gpu: &Gpu,
        layout: Layout,
        entries: &[Entry],
        dxil: impl Fn(&str) -> Option<&'static [u8]>,
    ) -> Result<Self, String> {
        Self::build(gpu, layout, entries, |name| {
            dxil(name)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| format!("{name}: no precompiled DXIL"))
        })
    }

    fn build(
        gpu: &Gpu,
        layout: Layout,
        entries: &[Entry],
        code: impl Fn(&str) -> Result<Vec<u8>, String>,
    ) -> Result<Self, String> {
        let mut root: Option<ID3D12RootSignature> = None;
        let mut kernels = HashMap::new();
        for e in entries {
            if e.srv_bytes_per_pixel.len() != layout.srvs
                || e.uav_bytes_per_pixel.len() != layout.uavs
                || e.required_srvs.iter().any(|&i| i >= layout.srvs)
            {
                return Err(format!("{}: entry does not match {layout:?}", e.name));
            }
            let dxil = code(e.name)?;
            let r = match &root {
                Some(r) => r.clone(),
                None => {
                    // SAFETY: the DXIL container embeds the `[RootSignature]` part.
                    let r: ID3D12RootSignature =
                        unsafe { gpu.device.CreateRootSignature(0, &dxil) }
                            .map_err(|err| format!("{}: root signature: {err}", e.name))?;
                    root = Some(r.clone());
                    r
                }
            };
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(r)),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: dxil.as_ptr().cast(),
                    BytecodeLength: dxil.len(),
                },
                ..Default::default()
            };
            // SAFETY: the bytecode outlives the call.
            let pso: Result<ID3D12PipelineState, _> =
                unsafe { gpu.device.CreateComputePipelineState(&desc) };
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            kernels.insert(
                e.name.to_owned(),
                Kernel {
                    pso: pso.map_err(|err| format!("{}: PSO: {err}", e.name))?,
                    required_srvs: e.required_srvs.to_vec(),
                    srv_bytes: e.srv_bytes_per_pixel.to_vec(),
                    uav_bytes: e.uav_bytes_per_pixel.to_vec(),
                },
            );
        }
        Ok(Self {
            root: root.ok_or("no entries")?,
            layout,
            kernels,
        })
    }

    /// Records a dispatch of `entry` over `w x h` threads (8x4 groups) after checking the
    /// bindings against the layout and that every per-pixel buffer covers `w * h` pixels
    /// (root descriptors are not bounds-checked by the GPU).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        gpu: &Gpu,
        entry: &str,
        constants: &[u32],
        srv: &[Option<&Buffer>],
        uav: &[Option<&Buffer>],
        table: Option<&TextureSet>,
        (w, h): (u32, u32),
    ) -> Result<(), String> {
        self.dispatch_rows(gpu, entry, constants, srv, uav, table, (w, h), (0, h))
    }

    /// As [`Programs::dispatch`] for the row band `y0..y0 + rows` of a `w x h` image (a
    /// tile): per-pixel buffers are checked against the whole image, the band must lie
    /// inside it, and `rows / 4` groups are dispatched. The kernel adds `y0` (passed in its
    /// constants by the caller) and bounds-checks against the full `w x h`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_rows(
        &self,
        gpu: &Gpu,
        entry: &str,
        constants: &[u32],
        srv: &[Option<&Buffer>],
        uav: &[Option<&Buffer>],
        table: Option<&TextureSet>,
        (w, h): (u32, u32),
        (y0, rows): (u32, u32),
    ) -> Result<(), String> {
        if rows == 0 || u64::from(y0) + u64::from(rows) > u64::from(h) {
            return Err(format!("{entry}: rows {y0}+{rows} outside {w}x{h}"));
        }
        // Groups are 4 rows high: a band ending inside the image must be a multiple of 4
        // rows, else its last group would also cover the first rows of the next band.
        if !rows.is_multiple_of(4) && y0 + rows != h {
            return Err(format!(
                "{entry}: band {y0}+{rows} not a multiple of 4 rows"
            ));
        }
        let k = self
            .kernels
            .get(entry)
            .ok_or_else(|| format!("unknown kernel {entry}"))?;
        let l = &self.layout;
        if constants.len() != l.constants
            || srv.len() != l.srvs
            || uav.len() != l.uavs
            || table.is_some() != l.table
        {
            return Err(format!(
                "{entry}: {} constants, {} SRVs, {} UAVs, table {} do not match {l:?}",
                constants.len(),
                srv.len(),
                uav.len(),
                table.is_some()
            ));
        }
        let px = u64::from(w) * u64::from(h);
        let check = |kind: &str, i: usize, b: Option<&Buffer>, per_px: u64, required: bool| match b
        {
            None if per_px > 0 || required => Err(format!("{entry}: {kind}{i} unbound")),
            Some(b) if b.size < per_px * px => Err(format!(
                "{entry}: {kind}{i} holds {} bytes, {w}x{h} needs {}",
                b.size,
                per_px * px
            )),
            _ => Ok(()),
        };
        for (i, b) in srv.iter().enumerate() {
            check("t", i, *b, k.srv_bytes[i], k.required_srvs.contains(&i))?;
        }
        for (i, b) in uav.iter().enumerate() {
            check("u", i, *b, k.uav_bytes[i], false)?;
        }
        let va = |b: Option<&Buffer>| b.map_or(gpu.dummy_va(), |b| b.va);
        // SAFETY: recording into the open list with live objects; sizes checked above.
        unsafe {
            let list = &gpu.list;
            if let Some(t) = table {
                list.SetDescriptorHeaps(&[Some(t.heap.clone())]);
            }
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&k.pso);
            list.SetComputeRoot32BitConstants(0, l.constants as u32, constants.as_ptr().cast(), 0);
            for (i, b) in srv.iter().enumerate() {
                list.SetComputeRootShaderResourceView(1 + i as u32, va(*b));
            }
            for (i, b) in uav.iter().enumerate() {
                list.SetComputeRootUnorderedAccessView((1 + l.srvs + i) as u32, va(*b));
            }
            if let Some(t) = table {
                list.SetComputeRootDescriptorTable((1 + l.srvs + l.uavs) as u32, t.gpu_start);
            }
            // 8x4 thread groups (`numthreads(8, 4, 1)` in every kernel): AMD RDNA performance
            // guide, "Use an 8x4 threadgroup size to optimally use the RDNA WGP" for ray
            // tracing.
            list.Dispatch(w.div_ceil(8), rows.div_ceil(4), 1);
        }
        Ok(())
    }
}
