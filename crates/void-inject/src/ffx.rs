//! AMD FidelityFX SDK 2.3 C API (`ffx_api`) bindings for the D3D12 upscaler (FSR 4.1 /
//! 3.1 / 2.3 providers), loaded at runtime from `amd_fidelityfx_loader_dx12.dll`.
//!
//! Layouts are transcribed from the MIT headers vendored with d2enhanced
//! (`third_party/FidelityFX-SDK/Kits/FidelityFX/api/include/{ffx_api.h, ffx_api_types.h,
//! dx12/ffx_api_dx12.h}`, `upscalers/include/ffx_upscale.h`, FFX_UPSCALER_VERSION 4.1.1).
//! Usage choices (flags, version query and override, exposure handling) are ported from
//! d2enhanced `src/fsr.cpp`, which runs FSR 4.1.1 in the game.
//!
//! `unsafe` in this module: the FFI calls into the loader, `LoadLibraryW`/`GetProcAddress`,
//! reading provider-owned C strings, and `GetDesc` on caller-provided resources.

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::Arc;

use windows::Win32::Foundation::{FreeLibrary, HMODULE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{HSTRING, Interface, PCSTR, PCWSTR};

// ---------------------------------------------------------------------------------------
// Header transcription (x64 MSVC: pointers 8 bytes / align 8, uint32_t and float 4,
// C `bool` 1 byte, struct size rounded up to the largest member alignment).
// ---------------------------------------------------------------------------------------

type FfxContext = *mut c_void;
type ReturnCode = u32;
type MessageFn = unsafe extern "C" fn(kind: u32, message: *const u16);

type PfnCreateContext = unsafe extern "C" fn(
    context: *mut FfxContext,
    desc: *mut Header,
    mem_cb: *const c_void,
) -> ReturnCode;
type PfnDestroyContext =
    unsafe extern "C" fn(context: *mut FfxContext, mem_cb: *const c_void) -> ReturnCode;
type PfnConfigure =
    unsafe extern "C" fn(context: *mut FfxContext, desc: *const Header) -> ReturnCode;
type PfnQuery = unsafe extern "C" fn(context: *mut FfxContext, desc: *mut Header) -> ReturnCode;
type PfnDispatch =
    unsafe extern "C" fn(context: *mut FfxContext, desc: *const Header) -> ReturnCode;
/// Non-null `GetProcAddress` result (`FARPROC`).
type Farproc = unsafe extern "system" fn() -> isize;

// enum FfxApiReturnCodes (ffx_api.h).
const RETURN_OK: ReturnCode = 0;

// Structure type ids. FFX_API_MAKE_EFFECT_SUB_ID(effect, sub) = (effect & 0x00ff0000) |
// (sub & ~0x00ff0000), FFX_API_EFFECT_ID_UPSCALE = 0x00010000; FFX_API_MAKE_BACKEND_SUB_ID
// with FFX_API_BACKEND_ID_DX12 = 0 is just the sub id.
const CONFIGURE_DESC_TYPE_GLOBALDEBUG1: u64 = 0x0000_0001;
const QUERY_DESC_TYPE_GET_VERSIONS: u64 = 4;
const DESC_TYPE_OVERRIDE_VERSION: u64 = 5;
const QUERY_DESC_TYPE_GET_PROVIDER_VERSION: u64 = 6;
const CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: u64 = 0x0000_0002;
const CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x0001_0001;
const QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 = 0x0001_0004;
const QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: u64 = 0x0001_0005;
const CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION: u64 = 0x0001_000b;

/// FFX_UPSCALER_MAKE_VERSION(4, 1, 1) = (4 << 22) | (1 << 12) | 1: the API version the
/// headers (and therefore these layouts) correspond to.
const UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 1;

const CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE: u32 = 0x0fff_ffff;
const MESSAGE_TYPE_ERROR: u32 = 0;

// enum FfxApiCreateContextUpscaleFlags.
const ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
const ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
const ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
const ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;
const ENABLE_DYNAMIC_RESOLUTION: u32 = 1 << 6;
const ENABLE_DEBUG_CHECKING: u32 = 1 << 7;
const ENABLE_DEBUG_VISUALIZATION: u32 = 1 << 9;
// enum FfxApiDispatchFsrUpscaleFlags.
const FLAG_DRAW_DEBUG_VIEW: u32 = 1 << 0;

// enum FfxApiResourceState.
const STATE_COMMON: u32 = 1 << 0;
const STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const STATE_COMPUTE_READ: u32 = 1 << 2;
const STATE_PIXEL_READ: u32 = 1 << 3;
const STATE_COPY_SRC: u32 = 1 << 4;
const STATE_COPY_DEST: u32 = 1 << 5;
const STATE_INDIRECT_ARGUMENT: u32 = 1 << 6;
// FFX_API_RESOURCE_STATE_PRESENT (1 << 7) is not used: D3D12_RESOURCE_STATE_PRESENT is 0
// (= COMMON) and maps to STATE_COMMON.
const STATE_RENDER_TARGET: u32 = 1 << 8;
const STATE_DEPTH_ATTACHMENT: u32 = 1 << 9;

// enum FfxApiResourceUsage.
const USAGE_READ_ONLY: u32 = 0;
const USAGE_UAV: u32 = 1 << 1;
const USAGE_DEPTHTARGET: u32 = 1 << 2;
const USAGE_STENCILTARGET: u32 = 1 << 5;

// enum FfxApiResourceType.
const RESOURCE_TYPE_BUFFER: u32 = 0;
const RESOURCE_TYPE_TEXTURE1D: u32 = 1;
const RESOURCE_TYPE_TEXTURE2D: u32 = 2;
const RESOURCE_TYPE_TEXTURE_CUBE: u32 = 3;
const RESOURCE_TYPE_TEXTURE3D: u32 = 4;

/// `ffxApiHeader`: u64 type + pointer = 16 bytes.
#[repr(C)]
struct Header {
    kind: u64,
    next: *mut Header,
}

impl Header {
    fn new(kind: u64) -> Self {
        Self {
            kind,
            next: null_mut(),
        }
    }
}

/// `FfxApiDimensions2D` / `FfxApiFloatCoords2D`: two 4-byte fields = 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Dim2 {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float2 {
    x: f32,
    y: f32,
}

/// `FfxApiResourceDescription`: type, format, width|size, height|stride, depth|alignment,
/// mipCount, flags, usage: 8 x u32 = 32 bytes (the unions are single u32s).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ResourceDescription {
    kind: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

/// `FfxApiResource`: pointer (0..8) + description (8..40) + state u32 (40..44), padded to
/// the 8-byte pointer alignment = 48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct ApiResource {
    resource: *mut c_void,
    description: ResourceDescription,
    state: u32,
}

impl ApiResource {
    const NULL: Self = Self {
        resource: null_mut(),
        description: ResourceDescription {
            kind: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            mip_count: 0,
            flags: 0,
            usage: 0,
        },
        state: 0,
    };
}

/// `ffxConfigureDescGlobalDebug1`: header 16 + fpMessage 8 (16..24) + debugLevel 4
/// (24..28), padded to 32.
#[repr(C)]
struct ConfigureGlobalDebug1 {
    header: Header,
    message: Option<MessageFn>,
    level: u32,
}

/// `ffxQueryDescGetVersions`: header 16 + createDescType 8 + device 8 + outputCount 8 +
/// versionIds 8 + versionNames 8 = 56.
#[repr(C)]
struct QueryGetVersions {
    header: Header,
    create_desc_type: u64,
    device: *mut c_void,
    output_count: *mut u64,
    version_ids: *mut u64,
    version_names: *mut *const c_char,
}

/// `ffxOverrideVersion`: header 16 + versionId 8 = 24.
#[repr(C)]
struct OverrideVersion {
    header: Header,
    version_id: u64,
}

/// `ffxQueryGetProviderVersion`: header 16 + versionId 8 + versionName 8 = 32.
#[repr(C)]
struct QueryGetProviderVersion {
    header: Header,
    version_id: u64,
    version_name: *const c_char,
}

/// `ffxCreateBackendDX12Desc`: header 16 + device pointer 8 = 24.
#[repr(C)]
struct CreateBackendDx12 {
    header: Header,
    device: *mut c_void,
}

/// `ffxCreateContextDescUpscale`: header 16, flags 4 (16..20), maxRenderSize 8 (20..28),
/// maxUpscaleSize 8 (28..36), 4 bytes padding, fpMessage 8 (40..48) = 48.
#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render_size: Dim2,
    max_upscale_size: Dim2,
    message: Option<MessageFn>,
}

/// `ffxCreateContextDescUpscaleVersion`: header 16 + version u32 4, padded to 24.
#[repr(C)]
struct CreateUpscaleVersion {
    header: Header,
    version: u32,
}

/// `ffxDispatchDescUpscale`: header 16, commandList 8 (16..24), 7 x FfxApiResource 48
/// (24..360), jitterOffset 8 (360..368), motionVectorScale 8 (..376), renderSize 8
/// (..384), upscaleSize 8 (..392), enableSharpening bool 1 at 392 + 3 padding,
/// sharpness (396..400), frameTimeDelta (..404), preExposure (..408), reset bool at 408,
/// 3 padding, cameraNear (412..416), cameraFar (..420), cameraFovAngleVertical (..424),
/// viewSpaceToMetersFactor (..428), flags (..432); 432 is a multiple of 8 = 432.
#[repr(C)]
struct DispatchUpscale {
    header: Header,
    command_list: *mut c_void,
    color: ApiResource,
    depth: ApiResource,
    motion_vectors: ApiResource,
    exposure: ApiResource,
    reactive: ApiResource,
    transparency_and_composition: ApiResource,
    output: ApiResource,
    jitter_offset: Float2,
    motion_vector_scale: Float2,
    render_size: Dim2,
    upscale_size: Dim2,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

/// `ffxQueryDescUpscaleGetJitterPhaseCount`: header 16 + renderWidth 4 + displayWidth 4
/// + pOutPhaseCount 8 (24..32) = 32.
#[repr(C)]
struct QueryJitterPhaseCount {
    header: Header,
    render_width: u32,
    display_width: u32,
    out_phase_count: *mut i32,
}

/// `ffxQueryDescUpscaleGetJitterOffset`: header 16 + index 4 + phaseCount 4 + pOutX 8 +
/// pOutY 8 = 40.
#[repr(C)]
struct QueryJitterOffset {
    header: Header,
    index: i32,
    phase_count: i32,
    out_x: *mut f32,
    out_y: *mut f32,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<Header>() == 16);
    assert!(size_of::<Dim2>() == 8);
    assert!(size_of::<Float2>() == 8);
    assert!(size_of::<ResourceDescription>() == 32);
    assert!(size_of::<ApiResource>() == 48);
    assert!(offset_of!(ApiResource, state) == 40);
    assert!(size_of::<ConfigureGlobalDebug1>() == 32);
    assert!(size_of::<QueryGetVersions>() == 56);
    assert!(size_of::<OverrideVersion>() == 24);
    assert!(size_of::<QueryGetProviderVersion>() == 32);
    assert!(size_of::<CreateBackendDx12>() == 24);
    assert!(size_of::<CreateUpscale>() == 48);
    assert!(offset_of!(CreateUpscale, message) == 40);
    assert!(size_of::<CreateUpscaleVersion>() == 24);
    assert!(size_of::<DispatchUpscale>() == 432);
    assert!(offset_of!(DispatchUpscale, jitter_offset) == 360);
    assert!(offset_of!(DispatchUpscale, enable_sharpening) == 392);
    assert!(offset_of!(DispatchUpscale, sharpness) == 396);
    assert!(offset_of!(DispatchUpscale, reset) == 408);
    assert!(offset_of!(DispatchUpscale, camera_near) == 412);
    assert!(offset_of!(DispatchUpscale, flags) == 428);
    assert!(size_of::<QueryJitterPhaseCount>() == 32);
    assert!(size_of::<QueryJitterOffset>() == 40);
};

/// Readable name for an `FfxApiReturnCodes` value.
fn rc_text(rc: ReturnCode) -> String {
    let name = match rc {
        1 => "FFX_API_RETURN_ERROR (unspecified)",
        2 => "FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE (structure type not recognised)",
        3 => "FFX_API_RETURN_ERROR_RUNTIME_ERROR (D3D12 or effect error)",
        4 => "FFX_API_RETURN_NO_PROVIDER (no provider for the structure type)",
        5 => "FFX_API_RETURN_ERROR_MEMORY (allocation failed)",
        6 => "FFX_API_RETURN_ERROR_PARAMETER (invalid parameter)",
        7 => {
            "FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE (provider too old for the structure type)"
        }
        _ => return format!("unknown ffx return code {rc}"),
    };
    format!("{name} ({rc})")
}

fn check(what: &str, rc: ReturnCode) -> Result<(), String> {
    if rc == RETURN_OK {
        Ok(())
    } else {
        Err(format!("{what}: {}", rc_text(rc)))
    }
}

/// Runtime messages (errors / warnings / debug checker) go to the module log (stderr in
/// offline tools).
unsafe extern "C" fn on_message(kind: u32, message: *const u16) {
    if message.is_null() {
        return;
    }
    // SAFETY: the runtime passes a NUL-terminated wide string valid for this call.
    let text = unsafe { PCWSTR(message).to_string() }.unwrap_or_else(|_| "<invalid UTF-16>".into());
    let level = if kind == MESSAGE_TYPE_ERROR {
        "error"
    } else {
        "warning"
    };
    crate::log::line(&format!("ffx {level}: {}", text.trim_end()));
}

/// Copies a provider-owned C string.
///
/// # Safety
/// `p` is null or a NUL-terminated string that is live for the call.
unsafe fn c_string(p: *const c_char) -> String {
    if p.is_null() {
        "?".into()
    } else {
        // SAFETY: per the function contract.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

// ---------------------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------------------

/// Effect DLLs the loader opens with `LoadLibraryA(<bare name>)` (names from its string
/// table, 2.3.0.2740). A bare name resolves against already-loaded modules first, so
/// preloading them from `dir` by full path makes the loader use these copies instead of
/// searching the process's DLL search order (exe dir, system, PATH).
const EFFECT_DLLS: [&str; 4] = [
    "amd_fidelityfx_upscaler_dx12.dll",
    "amd_fidelityfx_framegeneration_dx12.dll",
    "amd_fidelityfx_denoiser_dx12.dll",
    "amd_fidelityfx_radiancecache_dx12.dll",
];

struct LoaderInner {
    module: HMODULE,
    /// Effect DLLs preloaded from the loader's directory.
    effects: Vec<HMODULE>,
    create_context: PfnCreateContext,
    destroy_context: PfnDestroyContext,
    configure: PfnConfigure,
    query: PfnQuery,
    dispatch: PfnDispatch,
}

fn free_all(modules: impl IntoIterator<Item = HMODULE>) {
    for m in modules {
        // SAFETY: each handle came from a successful LoadLibraryW in `Loader::load`, is
        // freed once, and nothing created through it is alive (callers' contract).
        let _ = unsafe { FreeLibrary(m) };
    }
}

impl Drop for LoaderInner {
    fn drop(&mut self) {
        // Every `Upscaler` holds an `Arc` to this value, so no context or function pointer
        // outlives the modules.
        free_all(std::iter::once(self.module).chain(self.effects.drain(..)));
    }
}

/// `amd_fidelityfx_loader_dx12.dll` and its five exports. Cheap to clone; contexts keep
/// the module loaded.
#[derive(Clone)]
pub struct Loader(Arc<LoaderInner>);

// SAFETY: the module handles and the export addresses are process-global and immutable;
// the ffx functions are callable from any thread (contexts are synchronised by `&mut`).
unsafe impl Send for LoaderInner {}
// SAFETY: as above; `LoaderInner` has no interior mutability.
unsafe impl Sync for LoaderInner {}

fn load_library(path: &Path) -> Result<HMODULE, String> {
    // SAFETY: the HSTRING is a NUL-terminated copy of the path that outlives the call.
    unsafe { LoadLibraryW(&HSTRING::from(path)) }
        .map_err(|e| format!("LoadLibraryW({}): {e}", path.display()))
}

impl Loader {
    /// Loads `dir/amd_fidelityfx_loader_dx12.dll` by explicit path, after preloading the
    /// effect DLLs present in `dir` (see `EFFECT_DLLS`). Driver-side providers (FSR 4 on
    /// RDNA) are found by the upscaler DLL itself.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let mut effects = Vec::new();
        for name in EFFECT_DLLS {
            let p = dir.join(name);
            if p.is_file() {
                match load_library(&p) {
                    Ok(m) => effects.push(m),
                    Err(e) => {
                        free_all(effects);
                        return Err(e);
                    }
                }
            }
        }
        let path = dir.join("amd_fidelityfx_loader_dx12.dll");
        let module = match load_library(&path) {
            Ok(m) => m,
            Err(e) => {
                free_all(effects);
                return Err(e);
            }
        };
        let proc = |name: &'static [u8]| {
            // SAFETY: `module` is loaded and `name` is a NUL-terminated literal.
            unsafe { GetProcAddress(module, PCSTR(name.as_ptr())) }.ok_or_else(|| {
                format!(
                    "{} lacks export {}",
                    path.display(),
                    String::from_utf8_lossy(&name[..name.len() - 1])
                )
            })
        };
        let resolved = (|| {
            Ok::<_, String>((
                proc(b"ffxCreateContext\0")?,
                proc(b"ffxDestroyContext\0")?,
                proc(b"ffxConfigure\0")?,
                proc(b"ffxQuery\0")?,
                proc(b"ffxDispatch\0")?,
            ))
        })();
        let (create, destroy, configure, query, dispatch) = match resolved {
            Ok(f) => f,
            Err(e) => {
                free_all(std::iter::once(module).chain(effects));
                return Err(e);
            }
        };
        // SAFETY: the exports have the signatures declared in ffx_api.h
        // (PfnFfxCreateContext etc.); a function pointer transmute between
        // `extern "system" fn() -> isize` and those types is size-identical.
        unsafe {
            Ok(Self(Arc::new(LoaderInner {
                module,
                effects,
                create_context: std::mem::transmute::<Farproc, PfnCreateContext>(create),
                destroy_context: std::mem::transmute::<Farproc, PfnDestroyContext>(destroy),
                configure: std::mem::transmute::<Farproc, PfnConfigure>(configure),
                query: std::mem::transmute::<Farproc, PfnQuery>(query),
                dispatch: std::mem::transmute::<Farproc, PfnDispatch>(dispatch),
            })))
        }
    }
}

/// An upscaler provider (version) the loader offers for a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provider {
    /// Version id for `UpscalerDesc::provider` (ffxOverrideVersion).
    pub id: u64,
    /// Display name as reported by the provider, e.g. "4.1.1".
    pub version: String,
}

/// Lists the upscaler providers available on `device` in the loader's order
/// (d2enhanced observed newest first; not guaranteed by the header).
pub fn upscaler_providers(l: &Loader, device: &ID3D12Device) -> Result<Vec<Provider>, String> {
    let mut count = 0u64;
    let mut q = QueryGetVersions {
        header: Header::new(QUERY_DESC_TYPE_GET_VERSIONS),
        create_desc_type: CREATE_CONTEXT_DESC_TYPE_UPSCALE,
        device: device.as_raw(),
        output_count: &mut count,
        version_ids: null_mut(),
        version_names: null_mut(),
    };
    // SAFETY: `q` is a valid GET_VERSIONS query whose out pointer targets `count`; a null
    // context selects the global state.
    check("ffxQuery(GET_VERSIONS count)", unsafe {
        (l.0.query)(null_mut(), &mut q.header)
    })?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let capacity = count;
    let mut ids = vec![0u64; capacity as usize];
    let mut names: Vec<*const c_char> = vec![null(); capacity as usize];
    q.version_ids = ids.as_mut_ptr();
    q.version_names = names.as_mut_ptr();
    // SAFETY: both arrays hold `count` (the input capacity) entries.
    check("ffxQuery(GET_VERSIONS)", unsafe {
        (l.0.query)(null_mut(), &mut q.header)
    })?;
    let n = count.min(capacity) as usize;
    Ok(ids[..n]
        .iter()
        .zip(&names[..n])
        .map(|(&id, &name)| Provider {
            id,
            // SAFETY: provider-owned static strings (or null).
            version: unsafe { c_string(name) },
        })
        .collect())
}

// ---------------------------------------------------------------------------------------
// Upscaler
// ---------------------------------------------------------------------------------------

/// Context creation parameters (`ffxCreateContextDescUpscale` + flags).
#[derive(Clone, Debug)]
pub struct UpscalerDesc {
    pub max_render: [u32; 2],
    pub max_upscale: [u32; 2],
    /// Input colour is linear HDR (FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE).
    pub hdr: bool,
    /// Reversed-Z depth [1..0].
    pub depth_inverted: bool,
    /// Infinite far plane.
    pub depth_infinite: bool,
    /// FSR computes exposure itself; otherwise `UpscaleDispatch::exposure` is required in
    /// upscale mode (d2enhanced NOTES: missing -> FFX_API_RETURN_ERROR_PARAMETER).
    pub auto_exposure: bool,
    pub dynamic_resolution: bool,
    /// Version id from `upscaler_providers`; None = the loader's default (newest).
    pub provider: Option<u64>,
    /// FFX_UPSCALE_ENABLE_DEBUG_CHECKING + verbose global debug messages to the log.
    pub debug_checker: bool,
}

/// One FFX upscale context. Dropping it destroys the context: the GPU must be done with
/// every command list it recorded into.
pub struct Upscaler {
    loader: Loader,
    context: FfxContext,
    provider: Provider,
    /// Keeps the device referenced by the backend description alive.
    _device: ID3D12Device,
}

// SAFETY: an FFX context may be used from any thread as long as calls are serialised;
// `dispatch` takes `&mut self`, the `&self` queries are stateless jitter computations.
unsafe impl Send for Upscaler {}

/// A resource passed to FFX: `state` is the D3D12 state it is in when the recorded
/// dispatch executes (FFX transitions internally and restores it). The description
/// (type, format, size, usage) is read from the resource (`ffxApiGetResourceDX12`).
#[derive(Clone, Copy)]
pub struct Res<'a> {
    pub resource: &'a ID3D12Resource,
    pub state: D3D12_RESOURCE_STATES,
}

/// Per-frame parameters (`ffxDispatchDescUpscale`).
pub struct UpscaleDispatch<'a> {
    pub color: Res<'a>,
    pub depth: Res<'a>,
    pub motion_vectors: Res<'a>,
    pub exposure: Option<Res<'a>>,
    pub reactive: Option<Res<'a>>,
    pub transparency_composition: Option<Res<'a>>,
    pub output: Res<'a>,
    /// Sub-pixel jitter in render pixels (as returned by `jitter_offset`).
    pub jitter: [f32; 2],
    /// Multiplies the stored motion vectors to get render pixels (current -> previous).
    pub motion_vector_scale: [f32; 2],
    pub render_size: [u32; 2],
    pub upscale_size: [u32; 2],
    /// Some(0..=1) enables the RCAS sharpening pass.
    pub sharpening: Option<f32>,
    pub frame_time_delta_ms: f32,
    pub pre_exposure: f32,
    pub reset: bool,
    pub camera_near: f32,
    pub camera_far: f32,
    /// Vertical field of view in radians.
    pub camera_fov_y: f32,
    pub view_space_to_meters: f32,
    /// FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW (contexts are always created with
    /// FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION, as d2enhanced does).
    pub debug_view: bool,
}

impl Upscaler {
    pub fn new(l: &Loader, device: &ID3D12Device, d: &UpscalerDesc) -> Result<Self, String> {
        // Debug visualization always allowed so `debug_view` works without a new context
        // (d2enhanced's choice; costs some memory).
        let mut flags = ENABLE_DEBUG_VISUALIZATION;
        for (on, bit) in [
            (d.hdr, ENABLE_HIGH_DYNAMIC_RANGE),
            (d.depth_inverted, ENABLE_DEPTH_INVERTED),
            (d.depth_infinite, ENABLE_DEPTH_INFINITE),
            (d.auto_exposure, ENABLE_AUTO_EXPOSURE),
            (d.dynamic_resolution, ENABLE_DYNAMIC_RESOLUTION),
            (d.debug_checker, ENABLE_DEBUG_CHECKING),
        ] {
            if on {
                flags |= bit;
            }
        }
        let mut create = CreateUpscale {
            header: Header::new(CREATE_CONTEXT_DESC_TYPE_UPSCALE),
            flags,
            max_render_size: Dim2 {
                width: d.max_render[0],
                height: d.max_render[1],
            },
            max_upscale_size: Dim2 {
                width: d.max_upscale[0],
                height: d.max_upscale[1],
            },
            message: Some(on_message),
        };
        let mut backend = CreateBackendDx12 {
            header: Header::new(CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12),
            device: device.as_raw(),
        };
        let mut version = CreateUpscaleVersion {
            header: Header::new(CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION),
            version: UPSCALER_VERSION,
        };
        let mut pinned = OverrideVersion {
            header: Header::new(DESC_TYPE_OVERRIDE_VERSION),
            version_id: d.provider.unwrap_or(0),
        };
        // Chain: upscale -> DX12 backend -> API version -> (version override), as fsr.cpp.
        if d.provider.is_some() {
            version.header.next = &mut pinned.header;
        }
        backend.header.next = &mut version.header;
        create.header.next = &mut backend.header;

        let mut context: FfxContext = null_mut();
        // SAFETY: `create` heads a valid chain of live descriptions; `context` is a valid
        // out pointer; null allocation callbacks select malloc/free.
        check("ffxCreateContext(upscale)", unsafe {
            (l.0.create_context)(&mut context, &mut create.header, null())
        })?;

        let mut q = QueryGetProviderVersion {
            header: Header::new(QUERY_DESC_TYPE_GET_PROVIDER_VERSION),
            version_id: 0,
            version_name: null(),
        };
        // SAFETY: `context` was just created; `q` is a valid query.
        let rc = unsafe { (l.0.query)(&mut context, &mut q.header) };
        let provider = Provider {
            id: q.version_id,
            // SAFETY: provider-owned string (or null), copied immediately.
            version: unsafe { c_string(q.version_name) },
        };
        let up = Self {
            loader: l.clone(),
            context,
            provider,
            _device: device.clone(),
        };
        check("ffxQuery(GET_PROVIDER_VERSION)", rc)?;

        if d.debug_checker {
            let debug = ConfigureGlobalDebug1 {
                header: Header::new(CONFIGURE_DESC_TYPE_GLOBALDEBUG1),
                message: Some(on_message),
                level: CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE,
            };
            let mut ctx = up.context;
            // SAFETY: live context, valid description.
            check("ffxConfigure(GLOBALDEBUG1)", unsafe {
                (up.loader.0.configure)(&mut ctx, &debug.header)
            })?;
        }
        Ok(up)
    }

    /// Provider actually used by this context (`ffxQueryGetProviderVersion`).
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    pub fn jitter_phase_count(&self, render_w: u32, display_w: u32) -> Result<i32, String> {
        let mut count = 0i32;
        let mut q = QueryJitterPhaseCount {
            header: Header::new(QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT),
            render_width: render_w,
            display_width: display_w,
            out_phase_count: &mut count,
        };
        let mut ctx = self.context;
        // SAFETY: live context; the out pointer targets `count`.
        check("ffxQuery(GETJITTERPHASECOUNT)", unsafe {
            (self.loader.0.query)(&mut ctx, &mut q.header)
        })?;
        Ok(count)
    }

    /// Jitter for sequence `index` in render pixels, range [-0.5, 0.5].
    pub fn jitter_offset(&self, index: i32, phase_count: i32) -> Result<[f32; 2], String> {
        let (mut x, mut y) = (0f32, 0f32);
        let mut q = QueryJitterOffset {
            header: Header::new(QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET),
            index,
            phase_count,
            out_x: &mut x,
            out_y: &mut y,
        };
        let mut ctx = self.context;
        // SAFETY: live context; the out pointers target `x` and `y`.
        check("ffxQuery(GETJITTEROFFSET)", unsafe {
            (self.loader.0.query)(&mut ctx, &mut q.header)
        })?;
        Ok([x, y])
    }

    /// Records the upscale into `cmd` (open, direct or compute list on the context's
    /// device). The resources must stay alive until the list has executed.
    pub fn dispatch(
        &mut self,
        cmd: &ID3D12GraphicsCommandList,
        d: &UpscaleDispatch,
    ) -> Result<(), String> {
        let opt = |r: Option<Res>| r.map(api_resource).transpose();
        let desc = DispatchUpscale {
            header: Header::new(DISPATCH_DESC_TYPE_UPSCALE),
            command_list: cmd.as_raw(),
            color: api_resource(d.color)?,
            depth: api_resource(d.depth)?,
            motion_vectors: api_resource(d.motion_vectors)?,
            exposure: opt(d.exposure)?.unwrap_or(ApiResource::NULL),
            reactive: opt(d.reactive)?.unwrap_or(ApiResource::NULL),
            transparency_and_composition: opt(d.transparency_composition)?
                .unwrap_or(ApiResource::NULL),
            output: api_resource(d.output)?,
            jitter_offset: Float2 {
                x: d.jitter[0],
                y: d.jitter[1],
            },
            motion_vector_scale: Float2 {
                x: d.motion_vector_scale[0],
                y: d.motion_vector_scale[1],
            },
            render_size: Dim2 {
                width: d.render_size[0],
                height: d.render_size[1],
            },
            upscale_size: Dim2 {
                width: d.upscale_size[0],
                height: d.upscale_size[1],
            },
            enable_sharpening: d.sharpening.is_some(),
            sharpness: d.sharpening.unwrap_or(0.0),
            frame_time_delta: d.frame_time_delta_ms,
            pre_exposure: d.pre_exposure,
            reset: d.reset,
            camera_near: d.camera_near,
            camera_far: d.camera_far,
            camera_fov_angle_vertical: d.camera_fov_y,
            view_space_to_meters_factor: d.view_space_to_meters,
            flags: if d.debug_view {
                FLAG_DRAW_DEBUG_VIEW
            } else {
                0
            },
        };
        // SAFETY: live context; `desc` is a complete dispatch description whose resource
        // and command-list pointers are borrowed for the call.
        let rc = unsafe { (self.loader.0.dispatch)(&mut self.context, &desc.header) };
        check("ffxDispatch(upscale)", rc).map_err(|e| {
            if d.exposure.is_none() && d.render_size != d.upscale_size {
                format!(
                    "{e} (no exposure resource in upscale mode: required unless the context \
                     was created with auto_exposure)"
                )
            } else {
                e
            }
        })
    }
}

impl Drop for Upscaler {
    fn drop(&mut self) {
        // SAFETY: the context was created by this loader (kept alive by `self.loader`)
        // with null allocation callbacks; it is destroyed exactly once.
        let rc = unsafe { (self.loader.0.destroy_context)(&mut self.context, null()) };
        if rc != RETURN_OK {
            crate::log::line(&format!("ffxDestroyContext: {}", rc_text(rc)));
        }
    }
}

/// D3D12 resource state -> `FfxApiResourceState`.
fn ffx_state(s: D3D12_RESOURCE_STATES) -> Result<u32, String> {
    let read =
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    Ok(match s {
        D3D12_RESOURCE_STATE_COMMON => STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS => STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE => STATE_COMPUTE_READ,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE => STATE_PIXEL_READ,
        s if s == read => STATE_PIXEL_READ | STATE_COMPUTE_READ,
        D3D12_RESOURCE_STATE_COPY_SOURCE => STATE_COPY_SRC,
        D3D12_RESOURCE_STATE_COPY_DEST => STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_GENERIC_READ => STATE_COPY_SRC | STATE_COMPUTE_READ,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT => STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET => STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ => {
            STATE_DEPTH_ATTACHMENT
        }
        _ => return Err(format!("resource state {:#x} has no ffx equivalent", s.0)),
    })
}

/// `ffxApiGetSurfaceFormatDX12` (ffx_api_dx12.h), same cases.
fn ffx_format(f: DXGI_FORMAT) -> u32 {
    match f {
        DXGI_FORMAT_R32G32B32A32_TYPELESS => 1,
        DXGI_FORMAT_R32G32B32A32_UINT => 2,
        DXGI_FORMAT_R32G32B32A32_FLOAT => 3,
        DXGI_FORMAT_R16G16B16A16_FLOAT => 4,
        DXGI_FORMAT_R32G32_FLOAT => 6,
        DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
        | DXGI_FORMAT_X24_TYPELESS_G8_UINT
        | DXGI_FORMAT_R8_UINT => 7,
        DXGI_FORMAT_R24G8_TYPELESS
        | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        | DXGI_FORMAT_R32_UINT => 8,
        DXGI_FORMAT_R8G8B8A8_TYPELESS => 9,
        DXGI_FORMAT_R8G8B8A8_UNORM => 10,
        DXGI_FORMAT_R8G8B8A8_SNORM => 11,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => 12,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => 13,
        DXGI_FORMAT_B8G8R8A8_UNORM => 14,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => 15,
        DXGI_FORMAT_R11G11B10_FLOAT => 16,
        DXGI_FORMAT_R10G10B10A2_UNORM => 17,
        DXGI_FORMAT_R16G16_FLOAT => 18,
        DXGI_FORMAT_R16G16_UINT => 19,
        DXGI_FORMAT_R16_FLOAT => 21,
        DXGI_FORMAT_R16_UINT => 22,
        DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_R16_UNORM => 23,
        DXGI_FORMAT_R16_SNORM => 24,
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_A8_UNORM => 25,
        DXGI_FORMAT_R8G8_UNORM => 26,
        DXGI_FORMAT_R8G8_UINT => 27,
        DXGI_FORMAT_R32G8X24_TYPELESS
        | DXGI_FORMAT_D32_FLOAT_S8X24_UINT
        | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
        | DXGI_FORMAT_D32_FLOAT
        | DXGI_FORMAT_R32_FLOAT => 28,
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP => 29,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => 30,
        DXGI_FORMAT_R32G32_TYPELESS => 31,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => 32,
        DXGI_FORMAT_R16G16_TYPELESS => 33,
        DXGI_FORMAT_R16_TYPELESS => 34,
        DXGI_FORMAT_R8_TYPELESS => 35,
        DXGI_FORMAT_R8G8_TYPELESS => 36,
        DXGI_FORMAT_R32_TYPELESS => 37,
        DXGI_FORMAT_R8_SNORM => 39,
        _ => 0,
    }
}

/// `ffxApiGetResourceDX12` (ffx_api_dx12.h) with a D3D12 state.
fn api_resource(r: Res) -> Result<ApiResource, String> {
    // SAFETY: `r.resource` is a live resource borrowed by the caller.
    let desc = unsafe { r.resource.GetDesc() };
    let mut d = ResourceDescription::default();
    match desc.Dimension {
        D3D12_RESOURCE_DIMENSION_BUFFER => {
            d.usage = USAGE_UAV;
            d.width = desc.Width as u32;
            d.height = desc.Height;
            d.kind = RESOURCE_TYPE_BUFFER;
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE1D
        | D3D12_RESOURCE_DIMENSION_TEXTURE2D
        | D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            d.usage = match desc.Format {
                DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_D32_FLOAT => USAGE_DEPTHTARGET,
                DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => {
                    USAGE_DEPTHTARGET | USAGE_STENCILTARGET
                }
                _ => USAGE_READ_ONLY,
            };
            if desc
                .Flags
                .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
            {
                d.usage |= USAGE_UAV;
            }
            d.width = desc.Width as u32;
            d.height = desc.Height;
            d.depth = u32::from(desc.DepthOrArraySize);
            d.mip_count = u32::from(desc.MipLevels);
            d.kind = match desc.Dimension {
                D3D12_RESOURCE_DIMENSION_TEXTURE1D => RESOURCE_TYPE_TEXTURE1D,
                D3D12_RESOURCE_DIMENSION_TEXTURE3D => RESOURCE_TYPE_TEXTURE3D,
                _ if desc.DepthOrArraySize == 6 => RESOURCE_TYPE_TEXTURE_CUBE,
                _ => RESOURCE_TYPE_TEXTURE2D,
            };
        }
        _ => return Err("resource has an unknown dimension".into()),
    }
    d.format = ffx_format(desc.Format);
    if d.format == 0 && d.kind != RESOURCE_TYPE_BUFFER {
        return Err(format!("format {:?} has no ffx equivalent", desc.Format));
    }
    Ok(ApiResource {
        resource: r.resource.as_raw(),
        description: d,
        state: ffx_state(r.state)?,
    })
}
