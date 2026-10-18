//! The D3D12 device for ray tracing inside the game (docs/raytracing.md section 11.3).
//!
//! `D3D12CreateDevice` returns the existing device when one already exists for the adapter
//! in the process (D3D12 devices are per-adapter singletons), so the device created here on
//! the game's adapter (found by the LUID of the engine's D3D11 device) is the same object
//! FSR's runtime gets in `fsr.rs`. Work of ours runs on our own command queues; nothing here
//! submits to FSR's queue.
//!
//! Hand-off helpers: [`shared_texture`] (a D3D11-created shared texture opened by D3D12, the
//! `fsr.rs` pattern), [`CrossFence`] (a D3D12 fence opened by D3D11) and [`Queue`] (a queue
//! with its own fence: submit, completed value, bounded CPU wait).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, LUID, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11Device5, ID3D11DeviceContext, ID3D11Fence,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIDevice, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use crate::{log, mem};

pub(crate) use crate::fsr::Shared;

/// Immediate `ID3D11DeviceContext*` of the engine (`qword_1432D4EF8`, as `fsr.rs`).
const IMMEDIATE_CTX_RVA: u64 = 0x32D_4EF8;

/// The D3D12 device on the game's adapter and the game's D3D11 device.
pub struct Device {
    pub d12: ID3D12Device5,
    pub d11: ID3D11Device5,
    pub luid: LUID,
    /// DXR tier 1.1 (inline `RayQuery`) and shader model 6.5 are available.
    pub inline_raytracing: bool,
}

// SAFETY: D3D11 and D3D12 devices are free-threaded.
unsafe impl Send for Device {}
// SAFETY: as above.
unsafe impl Sync for Device {}

static DEVICE: Mutex<Option<Arc<Device>>> = Mutex::new(None);
static FAILED: Mutex<Option<String>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The device, if [`ensure`] created it (non-blocking).
pub fn get() -> Option<Arc<Device>> {
    lock(&DEVICE).clone()
}

/// Why the device could not be created (None: not tried or created).
pub fn failure() -> Option<String> {
    lock(&FAILED).clone()
}

/// The engine's D3D11 device (from its immediate context), if it exists yet.
pub(crate) fn game_d3d11() -> Option<ID3D11Device> {
    let base = mem::image_base()?;
    let raw = mem::u64_at(base + IMMEDIATE_CTX_RVA).filter(|&p| p != 0)? as *mut std::ffi::c_void;
    // SAFETY: the engine's live immediate context (created at startup, never released while
    // the game renders); `GetDevice` adds a reference.
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&raw) }?;
    // SAFETY: plain getter.
    unsafe { ctx.GetDevice() }.ok()
}

/// The device, created on first call from the engine's D3D11 device (any thread). A failure
/// is remembered and returned again without retrying.
pub fn ensure() -> Result<Arc<Device>, String> {
    if let Some(d) = get() {
        return Ok(d);
    }
    if let Some(e) = failure() {
        return Err(e);
    }
    let d11 = game_d3d11().ok_or("the game has no D3D11 device yet")?;
    let mut slot = lock(&DEVICE);
    if let Some(d) = slot.as_ref() {
        return Ok(d.clone());
    }
    match create(&d11) {
        Ok(d) => {
            log::line(&format!(
                "rt: D3D12 device on adapter LUID {:#x}:{:#x}, inline ray tracing {}",
                d.luid.HighPart,
                d.luid.LowPart,
                if d.inline_raytracing { "yes" } else { "no" }
            ));
            let d = Arc::new(d);
            *slot = Some(d.clone());
            Ok(d)
        }
        Err(e) => {
            log::line(&format!("rt: no D3D12 device: {e}"));
            *lock(&FAILED) = Some(e.clone());
            Err(e)
        }
    }
}

/// Creates (or obtains the existing) D3D12 device on `d11`'s adapter.
pub fn create(d11: &ID3D11Device) -> Result<Device, String> {
    let d11_5: ID3D11Device5 = d11.cast().map_err(|e| format!("ID3D11Device5: {e}"))?;
    // SAFETY (whole block): DXGI / D3D12 creation and feature queries with valid arguments
    // and out pointers on live objects.
    unsafe {
        let dxgi: IDXGIDevice = d11.cast().map_err(|e| e.to_string())?;
        let luid = dxgi
            .GetAdapter()
            .and_then(|a| a.GetDesc())
            .map_err(|e| format!("adapter: {e}"))?
            .AdapterLuid;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().map_err(|e| e.to_string())?;
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(luid)
            .map_err(|e| format!("EnumAdapterByLuid: {e}"))?;
        let mut d12: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut d12)
            .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
        let d12 = d12.ok_or("no D3D12 device")?;
        let d12: ID3D12Device5 = d12
            .cast()
            .map_err(|e| format!("ID3D12Device5 (DXR runtime): {e}"))?;
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        let rt_ok = d12
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut o5).cast(),
                size_of_val(&o5) as u32,
            )
            .is_ok()
            && o5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_5,
        };
        let sm_ok = d12
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .is_ok()
            && sm.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_5.0;
        Ok(Device {
            d12,
            d11: d11_5,
            luid,
            inline_raytracing: rt_ok && sm_ok,
        })
    }
}

/// A D3D11-created shared texture (NT handle) opened by D3D12 (`fsr.rs` pattern; native
/// D3D11 cannot open D3D12-created textures). Bind flags SRV + UAV on the D3D11 side.
pub(crate) fn shared_texture(
    dev: &Device,
    format: DXGI_FORMAT,
    w: u32,
    h: u32,
    uav: bool,
    srv: bool,
) -> Result<Shared, String> {
    let d12: ID3D12Device = dev.d12.cast().map_err(|e| e.to_string())?;
    crate::fsr::shared(&dev.d11, &d12, format, w, h, uav, srv)
}

/// A D3D12 fence shared with D3D11 (`ID3D11Fence` opened from its NT handle), with a
/// monotonic value counter for signals from either side.
pub struct CrossFence {
    pub f12: ID3D12Fence,
    pub f11: ID3D11Fence,
    value: AtomicU64,
}

// SAFETY: fences are free-threaded.
unsafe impl Send for CrossFence {}
// SAFETY: as above.
unsafe impl Sync for CrossFence {}

impl CrossFence {
    pub fn new(dev: &Device) -> Result<Self, String> {
        // SAFETY: fence creation / sharing with valid arguments; the handle is closed after
        // D3D11 opened its reference.
        unsafe {
            let f12: ID3D12Fence = dev
                .d12
                .CreateFence(0, D3D12_FENCE_FLAG_SHARED)
                .map_err(|e| format!("fence: {e}"))?;
            let h = dev
                .d12
                .CreateSharedHandle(&f12, None, GENERIC_ALL.0, None)
                .map_err(|e| format!("fence handle: {e}"))?;
            let mut f11: Option<ID3D11Fence> = None;
            let opened = dev.d11.OpenSharedFence(h, &mut f11);
            let _ = CloseHandle(h);
            opened.map_err(|e| format!("OpenSharedFence: {e}"))?;
            Ok(Self {
                f12,
                f11: f11.ok_or("no D3D11 fence")?,
                value: AtomicU64::new(0),
            })
        }
    }

    /// The next value to signal (strictly increasing).
    pub fn next(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Value the fence has reached on the GPU.
    pub fn completed(&self) -> u64 {
        // SAFETY: plain getter.
        unsafe { self.f12.GetCompletedValue() }
    }
}

/// A command queue with its own fence.
pub struct Queue {
    pub queue: ID3D12CommandQueue,
    pub fence: ID3D12Fence,
    /// Last signalled value (also serialises submit + signal).
    last: Mutex<u64>,
    event: HANDLE,
}

// SAFETY: queues and fences are free-threaded; the event handle is only waited on.
unsafe impl Send for Queue {}
// SAFETY: as above; `last` serialises submissions.
unsafe impl Sync for Queue {}

impl Queue {
    /// A direct queue of normal priority named `name` (PIX / debug tools).
    pub fn new(dev: &Device, name: &str) -> Result<Self, String> {
        // SAFETY: object creation with valid descriptors.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .d12
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(|e| format!("queue: {e}"))?;
            let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
            let _ = queue.SetName(windows::core::PCWSTR(wide.as_ptr()));
            let fence: ID3D12Fence = dev
                .d12
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("fence: {e}"))?;
            let event = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
            Ok(Self {
                queue,
                fence,
                last: Mutex::new(0),
                event,
            })
        }
    }

    /// Executes `lists` and signals the fence; returns the signalled value.
    pub fn submit(&self, lists: &[Option<ID3D12CommandList>]) -> Result<u64, String> {
        let mut last = lock(&self.last);
        let v = *last + 1;
        // SAFETY: closed command lists of this device; the fence is ours.
        unsafe {
            self.queue.ExecuteCommandLists(lists);
            self.queue
                .Signal(&self.fence, v)
                .map_err(|e| format!("signal: {e}"))?;
        }
        *last = v;
        Ok(v)
    }

    /// Value the queue's fence has reached.
    pub fn completed(&self) -> u64 {
        // SAFETY: plain getter.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// Blocks until the fence reaches `v` (at most `timeout_ms`).
    pub fn wait(&self, v: u64, timeout_ms: u32) -> Result<(), String> {
        if self.completed() >= v {
            return Ok(());
        }
        // SAFETY: our own event; the fence notifies it once.
        unsafe {
            self.fence
                .SetEventOnCompletion(v, self.event)
                .map_err(|e| format!("fence event: {e}"))?;
            if WaitForSingleObject(self.event, timeout_ms) != WAIT_OBJECT_0 {
                return Err(format!("GPU work not finished after {timeout_ms} ms"));
            }
        }
        Ok(())
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        let last = *lock(&self.last);
        let _ = self.wait(last, 5000);
        // SAFETY: our own event handle.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}
