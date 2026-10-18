//! Showing the photo (docs/raytracing.md 11.4) and the hybrid terms' depth capture (13).
//!
//! **Depth capture (`photo_display = hybrid`):** when `hybrid.rs` opened a request, the FOG
//! pass records `cs_export_depth` before the original: the game's view depth of the render
//! region (`zbuffer`, `z = depth_mul / (d - depth_add)`, 0 = sky) into the request's shared
//! R32F texture, and stores the frame's view matrix, projection jitter and sun direction.
//!
//! **Composite (default):** the `ARK_RENDERPASS_FOG` pass callback `sub_140309430(pass)` is
//! detoured; before the original records, we record into the pass's deferred context:
//! 1. `cs_stats`: over every 4th pixel in x and y whose depth matches the game's, the sum of
//!    `log2(game / ours)` luminance into a small buffer (the unit fit, `photo_units = auto`);
//! 2. a full-screen draw into `hdrBuffer` (RGB write mask, alpha kept) of our radiance x the
//!    fitted (or fixed) units where our primary-hit depth matches the game's depth within
//!    2 % + 2 cm; elsewhere (characters, objects we lack, sky, our misses) the game's pixel
//!    stays.
//!
//! The engine then applies fog, volumetrics, water, particles, DOF, TAA/FSR, bloom, tone
//! mapping and grading on top. The pass's write list holds `hdrBuffer` (hash 0x8F93AB75)
//! and the hardware depth `zbuffer` (0xBA729A59) (O, IDB: the callback looks both up first).
//! Depth: `z = depth_mul / (d - depth_add)` from the view's projection (`ao::projection`,
//! reversed Z with `r_zfar`); ours is `znear / z` (`void_rt::ingame`).
//!
//! Hand-off: two slots of D3D11-created shared textures (RGBA32F radiance, R32F depth)
//! opened by D3D12 (`rt::device::shared_texture`). The RT worker copies into a slot that is
//! not published and was not read for [`REUSE_FRAMES`] FOG passes, waits on the CPU for its
//! D3D12 fence, then publishes it; the FOG pass reads only the published slot. No GPU waits
//! across the APIs (E047/E048); the frame margin covers the D3D11 frames in flight (H:
//! DXGI's default maximum frame latency is 3).
//!
//! **Overlay (`photo_display = overlay`, needs `overlay = 1`):** our image tone-mapped
//! (log-average exposure to 0.18, Reinhard, sRGB) drawn over the back buffer at present.
//!
//! **Back-buffer capture (save):** at present, before the overlay draws, the back buffer is
//! copied into a staging texture and written as PNG a few frames later.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use arc_swap::ArcSwapOption;
use hex_literal::hex;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{
    D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D11_SRV_DIMENSION_BUFFEREX,
    D3D11_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, s};

use crate::ao::{Saved, Slots as StateSlots};
use crate::rt::device::{self, Shared};
use crate::{hlsl, log, mem, settings};

/// `ARK_RENDERPASS_FOG` callback.
const FOG_RVA: u64 = 0x30_9430;
const FOG_HEAD: [u8; 16] = hex!("40 53 55 56 57 41 54 41 55 41 56 41 57 48 81 EC");
/// Pass layout (`contact.rs`): view, engine context.
const PASS_VIEW: u64 = 0x70;
const PASS_ENGINE_CTX: u64 = 0x108;
/// `zbuffer` (FOG write list; `fsr_scale.rs` HASH_ZBUFFER).
const HASH_ZBUFFER: u32 = 0xBA72_9A59;
/// Engine context shader caches (`fsr.rs` ENGINE_CTX_SHADER_CACHE).
const ENGINE_CTX_SHADER_CACHE: u64 = 0x12B88;
/// FOG passes a slot stays unread before the worker may write it again.
pub const REUSE_FRAMES: u64 = 4;

type PassFn = unsafe extern "system" fn(u64) -> u64;

static ORIG: AtomicUsize = AtomicUsize::new(0);
static HOOKED: AtomicBool = AtomicBool::new(false);
static INSTALL_TRIED: AtomicBool = AtomicBool::new(false);
static FOG_FRAMES: AtomicU64 = AtomicU64::new(0);
static COMPOSITED: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
static ERROR: Mutex<Option<String>> = Mutex::new(None);
/// Region (x, y, w, h) of the last FOG pass in its targets, the targets' size, the
/// projection depth terms and the pass counter at that time.
static LAST: Mutex<Option<FogFrame>> = Mutex::new(None);
static CURRENT: ArcSwapOption<SlotSet> = ArcSwapOption::const_empty();

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogFrame {
    pub region: [u32; 4],
    pub tex: [u32; 2],
    pub depth_mul: f32,
    pub depth_add: f32,
    pub frame: u64,
}

/// The last FOG pass seen (None before the first one with `photo = 1`).
pub fn last_frame() -> Option<FogFrame> {
    *lock(&LAST)
}

/// FOG passes seen so far.
pub fn fog_frames() -> u64 {
    FOG_FRAMES.load(Ordering::Acquire)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub hooked: bool,
    pub fog_frames: u64,
    pub composited: u64,
    pub failures: u64,
    pub error: Option<String>,
}

pub fn status() -> Status {
    Status {
        hooked: HOOKED.load(Ordering::Relaxed),
        fog_frames: FOG_FRAMES.load(Ordering::Relaxed),
        composited: COMPOSITED.load(Ordering::Relaxed),
        failures: FAILURES.load(Ordering::Relaxed),
        error: lock(&ERROR).clone(),
    }
}

// ------------------------------------------------------------------------------ slots --

/// One hand-off slot.
pub(crate) struct Slot {
    pub hdr: Shared,
    pub depth: Shared,
}

/// The two slots of a photo session and their publication state.
pub(crate) struct SlotSet {
    pub slots: [Slot; 2],
    pub size: [u32; 2],
    /// `znear` of our depth (`znear / z`).
    pub znear: f32,
    /// -1 = nothing published, else the slot index.
    published: AtomicI32,
    /// FOG pass counter when each slot was last published or replaced.
    released: [AtomicU64; 2],
}

// SAFETY: D3D11 views and D3D12 resources are free-threaded; the slot protocol above keeps
// the GPU accesses apart.
unsafe impl Send for SlotSet {}
// SAFETY: as above.
unsafe impl Sync for SlotSet {}

impl SlotSet {
    pub fn new(dev: &device::Device, size: [u32; 2], znear: f32) -> Result<Self, String> {
        let slot = || -> Result<Slot, String> {
            Ok(Slot {
                hdr: device::shared_texture(
                    dev,
                    DXGI_FORMAT_R32G32B32A32_FLOAT,
                    size[0],
                    size[1],
                    false,
                    true,
                )?,
                depth: device::shared_texture(
                    dev,
                    DXGI_FORMAT_R32_FLOAT,
                    size[0],
                    size[1],
                    false,
                    true,
                )?,
            })
        };
        Ok(Self {
            slots: [slot()?, slot()?],
            size,
            znear,
            published: AtomicI32::new(-1),
            released: [AtomicU64::new(0), AtomicU64::new(0)],
        })
    }

    /// A slot the worker may write now (not published, unread for `REUSE_FRAMES` passes).
    pub fn writable(&self) -> Option<usize> {
        let p = self.published.load(Ordering::Acquire);
        let now = fog_frames();
        (0..2).find(|&k| {
            k as i32 != p && now >= self.released[k].load(Ordering::Acquire) + REUSE_FRAMES
        })
    }

    /// Makes slot `k` (written, its D3D12 fence passed on the CPU) the one the FOG pass reads.
    pub fn publish(&self, k: usize) {
        let now = fog_frames();
        let old = self.published.swap(k as i32, Ordering::AcqRel);
        if old >= 0 {
            self.released[old as usize].store(now, Ordering::Release);
        }
        self.released[k].store(now, Ordering::Release);
    }

    pub fn published(&self) -> Option<usize> {
        usize::try_from(self.published.load(Ordering::Acquire)).ok()
    }

    /// Stops showing any slot (camera moved: the old image no longer fits the frame).
    pub fn unpublish(&self) {
        let now = fog_frames();
        let old = self.published.swap(-1, Ordering::AcqRel);
        if old >= 0 {
            self.released[old as usize].store(now, Ordering::Release);
        }
    }

    /// Whether `set` is the one the FOG pass reads.
    pub fn is_current(self: &Arc<Self>) -> bool {
        current().is_some_and(|c| Arc::ptr_eq(&c, self))
    }
}

/// The FOG pass / present path read `slots` from now on.
pub(crate) fn activate(slots: Arc<SlotSet>) {
    CURRENT.store(Some(slots));
}

pub(crate) fn deactivate() {
    CURRENT.store(None);
}

fn current() -> Option<Arc<SlotSet>> {
    CURRENT.load_full()
}

// --------------------------------------------------------------------------- shaders --

const HLSL: &str = r"
// void-inject photo composite / preview (photo/composite.rs).
cbuffer P : register(b0) {
    int2 g_origin;       // region offset in the engine targets / preview destination offset
    uint2 g_size;        // our image size
    float g_depth_mul;   // game view z = depth_mul / (d - depth_add)
    float g_depth_add;
    float g_znear;       // our depth = znear / z
    float g_units;       // > 0: fixed engine units per photo unit; 0: fitted
    float g_fallback;    // units when the fit has too few samples
    float g_tolerance;   // relative depth tolerance
    float2 g_dst_size;   // preview: destination rectangle size
};
Texture2D<float4> ours : register(t0);
Texture2D<float> ours_depth : register(t1);
Texture2D<float> game_depth : register(t2);
Texture2D<float4> game_hdr : register(t3);
ByteAddressBuffer stats_in : register(t4);
RWByteAddressBuffer stats : register(u0);
RWTexture2D<float> depth_out : register(u1);

static const float LOG_SCALE = 64.0;
static const uint MIN_SAMPLES = 64;

float lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

bool finite3(float3 c) { return all(c == c) && all(abs(c) < 1e30); }

// Our radiance at p (image pixel) when our first hit matches the game's depth there.
bool matched(int2 p, out float3 c) {
    c = ours.Load(int3(p, 0)).rgb;
    float dg = game_depth.Load(int3(p + g_origin, 0));
    float dz = dg - g_depth_add;
    float dour = ours_depth.Load(int3(p, 0));
    // Game sky / far plane (cleared to 0 in reversed Z) and our misses keep the game pixel.
    if (dg <= 0.0 || dour <= 0.0 || dz <= 0.0 || !finite3(c)) return false;
    float zg = g_depth_mul / dz;
    float zo = g_znear / dour;
    return zg > 0.0 && abs(zo - zg) <= g_tolerance * zg + 0.02;
}

[numthreads(16, 16, 1)]
void cs_stats(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= g_size) || (id.x & 3) != 0 || (id.y & 3) != 0) return;
    float3 c;
    if (!matched(int2(id.xy), c)) return;
    float lo = lum(c);
    float lg = lum(game_hdr.Load(int3(int2(id.xy) + g_origin, 0)).rgb);
    if (!(lo > 1e-6) || !(lg > 1e-6)) return;
    int v = (int)round(clamp(log2(lg / lo), -16.0, 16.0) * LOG_SCALE);
    stats.InterlockedAdd(0, asuint(v));
    stats.InterlockedAdd(4, 1u);
}

// Log-average luminance of our image (preview exposure).
[numthreads(16, 16, 1)]
void cs_logavg(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= g_size) || (id.x & 3) != 0 || (id.y & 3) != 0) return;
    float3 c = ours.Load(int3(int2(id.xy), 0)).rgb;
    float l = lum(c);
    if (!finite3(c) || !(l > 1e-6)) return;
    int v = (int)round(clamp(log2(l), -32.0, 32.0) * LOG_SCALE);
    stats.InterlockedAdd(8, asuint(v));
    stats.InterlockedAdd(12, 1u);
}

// Hybrid terms: the game's view depth of the region (0 = sky / far plane / invalid).
[numthreads(16, 16, 1)]
void cs_export_depth(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= g_size)) return;
    float d = game_depth.Load(int3(int2(id.xy) + g_origin, 0));
    float dz = d - g_depth_add;
    float z = dz > 0.0 ? g_depth_mul / dz : 0.0;
    depth_out[id.xy] = (d > 0.0 && z > 0.0 && z < 1e5) ? z : 0.0;
}

float units() {
    if (g_units > 0.0) return g_units;
    uint n = stats_in.Load(4);
    if (n < MIN_SAMPLES) return g_fallback;
    return exp2((float)asint(stats_in.Load(0)) / (LOG_SCALE * (float)n));
}

float4 vs_full(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 ps_composite(float4 pos : SV_Position) : SV_Target {
    int2 p = int2(pos.xy) - g_origin;
    if (any(p < 0) || any(p >= int2(g_size))) discard;
    float3 c;
    if (!matched(p, c)) discard;
    return float4(c * units(), 1.0);
}

float3 to_srgb(float3 x) {
    x = saturate(x);
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float4 ps_preview(float4 pos : SV_Position) : SV_Target {
    float2 uv = (pos.xy - float2(g_origin)) / g_dst_size;
    if (any(uv < 0.0) || any(uv >= 1.0)) return float4(0.0, 0.0, 0.0, 1.0);
    int2 p = min(int2(uv * float2(g_size)), int2(g_size) - 1);
    float3 c = ours.Load(int3(p, 0)).rgb;
    if (!finite3(c)) c = 0.0;
    uint n = stats_in.Load(12);
    float avg = n > 0 ? exp2((float)asint(stats_in.Load(8)) / (LOG_SCALE * (float)n)) : 1.0;
    float3 x = max(c, 0.0) * (0.18 / max(avg, 1e-6));
    return float4(to_srgb(x / (1.0 + x)), 1.0);
}
";

struct Bytecode {
    stats: Vec<u8>,
    logavg: Vec<u8>,
    export: Vec<u8>,
    vs: Vec<u8>,
    composite: Vec<u8>,
    preview: Vec<u8>,
}

static BYTECODE: LazyLock<Result<Bytecode, String>> = LazyLock::new(|| {
    let c = |e, t| hlsl::compile(HLSL, e, t);
    Ok(Bytecode {
        stats: c(s!("cs_stats"), s!("cs_5_0"))?,
        logavg: c(s!("cs_logavg"), s!("cs_5_0"))?,
        export: c(s!("cs_export_depth"), s!("cs_5_0"))?,
        vs: c(s!("vs_full"), s!("vs_5_0"))?,
        composite: c(s!("ps_composite"), s!("ps_5_0"))?,
        preview: c(s!("ps_preview"), s!("ps_5_0"))?,
    })
});

/// `cbuffer P` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Params {
    pub origin: [i32; 2],
    pub size: [u32; 2],
    pub depth_mul: f32,
    pub depth_add: f32,
    pub znear: f32,
    pub units: f32,
    pub fallback: f32,
    pub tolerance: f32,
    pub dst_size: [f32; 2],
}

/// Relative depth tolerance of the match (plus 2 cm in the shader).
pub const TOLERANCE: f32 = 0.02;

/// Units when the fit has too few samples: stored game colour per absolute unit in bright
/// daylight (`void_rt::ingame::DAYLIGHT_STORED_SCALE`).
pub const FALLBACK_UNITS: f32 = void_rt::ingame::DAYLIGHT_STORED_SCALE;

fn create_err(what: &'static str) -> impl Fn(windows::core::Error) -> String {
    move |e| format!("{what}: {e}")
}

/// D3D11 objects for one device.
pub(crate) struct Pipe {
    pub device: ID3D11Device,
    cs_stats: ID3D11ComputeShader,
    cs_logavg: ID3D11ComputeShader,
    cs_export: ID3D11ComputeShader,
    vs: ID3D11VertexShader,
    ps_composite: ID3D11PixelShader,
    ps_preview: ID3D11PixelShader,
    cb: ID3D11Buffer,
    stats_uav: ID3D11UnorderedAccessView,
    stats_srv: ID3D11ShaderResourceView,
    blend_rgb: ID3D11BlendState,
    raster: ID3D11RasterizerState,
    depth_off: ID3D11DepthStencilState,
    /// Views of engine textures keyed by resource pointer (resource kept alive).
    srvs: HashMap<usize, (ID3D11Resource, ID3D11ShaderResourceView)>,
    rtvs: HashMap<usize, (ID3D11Resource, ID3D11RenderTargetView)>,
}

// SAFETY: D3D11 device objects are free-threaded; the pipe is used under its mutex.
unsafe impl Send for Pipe {}

const MAX_CACHED_VIEWS: usize = 8;

/// Slots the composite / preview / depth export bind (t0-t4, b0, u0-u1).
const TOUCHED: StateSlots = StateSlots {
    srv: 5,
    cb: 1,
    sampler: 0,
    uav: 2,
};

impl Pipe {
    pub fn new(dev: &ID3D11Device) -> Result<Self, String> {
        let code = BYTECODE.as_ref().map_err(Clone::clone)?;
        // SAFETY (whole block): object creation with valid descriptors, bytecode compiled
        // for the declared stage, and out pointers.
        unsafe {
            let (mut cs_stats, mut cs_logavg, mut cs_export, mut vs, mut ps_c, mut ps_p) =
                (None, None, None, None, None, None);
            dev.CreateComputeShader(&code.stats, None, Some(&mut cs_stats))
                .map_err(create_err("cs_stats"))?;
            dev.CreateComputeShader(&code.logavg, None, Some(&mut cs_logavg))
                .map_err(create_err("cs_logavg"))?;
            dev.CreateComputeShader(&code.export, None, Some(&mut cs_export))
                .map_err(create_err("cs_export_depth"))?;
            dev.CreateVertexShader(&code.vs, None, Some(&mut vs))
                .map_err(create_err("vs_full"))?;
            dev.CreatePixelShader(&code.composite, None, Some(&mut ps_c))
                .map_err(create_err("ps_composite"))?;
            dev.CreatePixelShader(&code.preview, None, Some(&mut ps_p))
                .map_err(create_err("ps_preview"))?;
            let mut cb = None;
            dev.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: size_of::<Params>() as u32,
                    Usage: D3D11_USAGE_DYNAMIC,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut cb),
            )
            .map_err(create_err("constant buffer"))?;
            let mut stats = None;
            dev.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: 16,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0)
                        as u32,
                    MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut stats),
            )
            .map_err(create_err("stats buffer"))?;
            let stats: ID3D11Buffer = stats.ok_or("no stats buffer")?;
            let mut uav = None;
            dev.CreateUnorderedAccessView(
                &stats,
                Some(&D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D11_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: 4,
                            Flags: D3D11_BUFFER_UAV_FLAG_RAW.0 as u32,
                        },
                    },
                }),
                Some(&mut uav),
            )
            .map_err(create_err("stats UAV"))?;
            let mut srv = None;
            dev.CreateShaderResourceView(
                &stats,
                Some(&D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D11_SRV_DIMENSION_BUFFEREX,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        BufferEx: D3D11_BUFFEREX_SRV {
                            FirstElement: 0,
                            NumElements: 4,
                            Flags: D3D11_BUFFEREX_SRV_FLAG_RAW.0 as u32,
                        },
                    },
                }),
                Some(&mut srv),
            )
            .map_err(create_err("stats SRV"))?;
            let mut blend_desc = D3D11_BLEND_DESC::default();
            blend_desc.RenderTarget[0].RenderTargetWriteMask = (D3D11_COLOR_WRITE_ENABLE_RED.0
                | D3D11_COLOR_WRITE_ENABLE_GREEN.0
                | D3D11_COLOR_WRITE_ENABLE_BLUE.0)
                as u8;
            let mut blend = None;
            dev.CreateBlendState(&blend_desc, Some(&mut blend))
                .map_err(create_err("blend state"))?;
            let mut raster = None;
            dev.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut raster),
            )
            .map_err(create_err("rasterizer state"))?;
            let mut depth_off = None;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: false.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut depth_off),
            )
            .map_err(create_err("depth state"))?;
            Ok(Self {
                device: dev.clone(),
                cs_stats: cs_stats.ok_or("no cs_stats")?,
                cs_logavg: cs_logavg.ok_or("no cs_logavg")?,
                cs_export: cs_export.ok_or("no cs_export_depth")?,
                vs: vs.ok_or("no vs_full")?,
                ps_composite: ps_c.ok_or("no ps_composite")?,
                ps_preview: ps_p.ok_or("no ps_preview")?,
                cb: cb.ok_or("no constant buffer")?,
                stats_uav: uav.ok_or("no stats UAV")?,
                stats_srv: srv.ok_or("no stats SRV")?,
                blend_rgb: blend.ok_or("no blend state")?,
                raster: raster.ok_or("no rasterizer state")?,
                depth_off: depth_off.ok_or("no depth state")?,
                srvs: HashMap::new(),
                rtvs: HashMap::new(),
            })
        }
    }

    /// SRV of an engine texture with a typed format for `format` (depth: first component).
    pub fn srv(
        &mut self,
        r: &ID3D11Resource,
        depth: bool,
    ) -> Result<ID3D11ShaderResourceView, String> {
        let key = r.as_raw() as usize;
        if let Some((_, v)) = self.srvs.get(&key) {
            return Ok(v.clone());
        }
        let (t, d) = tex2d(r)?;
        let format = if depth {
            depth_view_format(d.Format)
        } else {
            color_view_format(d.Format)
        }
        .ok_or_else(|| format!("unsupported format {} for an SRV", d.Format.0))?;
        if d.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
            return Err(format!("texture format {} has no SRV binding", d.Format.0));
        }
        let mut v = None;
        // SAFETY: valid descriptor and out pointer.
        unsafe {
            self.device.CreateShaderResourceView(
                &t,
                Some(&D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_SRV {
                            MostDetailedMip: 0,
                            MipLevels: 1,
                        },
                    },
                }),
                Some(&mut v),
            )
        }
        .map_err(create_err("engine SRV"))?;
        let v = v.ok_or("no SRV")?;
        if self.srvs.len() >= MAX_CACHED_VIEWS {
            self.srvs.clear();
        }
        self.srvs.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    /// RTV of the engine's HDR colour.
    pub fn rtv(&mut self, r: &ID3D11Resource) -> Result<ID3D11RenderTargetView, String> {
        let key = r.as_raw() as usize;
        if let Some((_, v)) = self.rtvs.get(&key) {
            return Ok(v.clone());
        }
        let (t, d) = tex2d(r)?;
        let format = color_view_format(d.Format)
            .ok_or_else(|| format!("unsupported colour format {}", d.Format.0))?;
        let mut v = None;
        // SAFETY: valid descriptor and out pointer.
        unsafe {
            self.device.CreateRenderTargetView(
                &t,
                Some(&D3D11_RENDER_TARGET_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
                    },
                }),
                Some(&mut v),
            )
        }
        .map_err(create_err("engine RTV"))?;
        let v = v.ok_or("no RTV")?;
        if self.rtvs.len() >= MAX_CACHED_VIEWS {
            self.rtvs.clear();
        }
        self.rtvs.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    fn write_params(&self, ctx: &ID3D11DeviceContext, p: &Params) -> Result<(), String> {
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: dynamic buffer of `size_of::<Params>()` bytes mapped for writing.
        unsafe {
            ctx.Map(&self.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut m))
                .map_err(create_err("map constants"))?;
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(p).cast::<u8>(),
                m.pData.cast::<u8>(),
                size_of::<Params>(),
            );
            ctx.Unmap(&self.cb, 0);
        }
        Ok(())
    }

    /// Records the unit fit and the composite into `ctx` (any context), restoring every
    /// state it touched.
    #[allow(clippy::too_many_arguments)]
    pub fn record_composite(
        &self,
        ctx: &ID3D11DeviceContext,
        p: &Params,
        ours: &ID3D11ShaderResourceView,
        ours_depth: &ID3D11ShaderResourceView,
        game_depth: &ID3D11ShaderResourceView,
        game_hdr: &ID3D11ShaderResourceView,
        target: &ID3D11RenderTargetView,
        region: [u32; 4],
    ) -> Result<(), String> {
        self.write_params(ctx, p)?;
        // SAFETY (whole block): state capture/restore and binds of live objects with arrays
        // of their declared lengths on a context this thread records.
        unsafe {
            let saved = Saved::capture(ctx, TOUCHED);
            ctx.OMSetRenderTargets(None, None);
            ctx.ClearUnorderedAccessViewUint(&self.stats_uav, &[0; 4]);
            ctx.CSSetShader(&self.cs_stats, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.CSSetShaderResources(
                0,
                Some(&[
                    Some(ours.clone()),
                    Some(ours_depth.clone()),
                    Some(game_depth.clone()),
                    Some(game_hdr.clone()),
                ]),
            );
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(self.stats_uav.clone())), None);
            ctx.Dispatch(p.size[0].div_ceil(16), p.size[1].div_ceil(16), 1);
            ctx.CSSetShaderResources(0, Some(&[None, None, None, None]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.ps_composite, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.PSSetShaderResources(
                0,
                Some(&[
                    Some(ours.clone()),
                    Some(ours_depth.clone()),
                    Some(game_depth.clone()),
                    None,
                    Some(self.stats_srv.clone()),
                ]),
            );
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: region[0] as f32,
                TopLeftY: region[1] as f32,
                Width: region[2] as f32,
                Height: region[3] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.RSSetScissorRects(Some(&[RECT {
                left: region[0] as i32,
                top: region[1] as i32,
                right: (region[0] + region[2]) as i32,
                bottom: (region[1] + region[3]) as i32,
            }]));
            ctx.OMSetBlendState(&self.blend_rgb, None, u32::MAX);
            ctx.OMSetDepthStencilState(&self.depth_off, 0);
            ctx.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            ctx.Draw(3, 0);
            ctx.OMSetRenderTargets(None, None);
            ctx.PSSetShaderResources(0, Some(&[None, None, None, None, None]));
            saved.restore(ctx);
        }
        Ok(())
    }

    /// Records the export of the game's view depth of `p`'s region (`p.origin`, `p.size`,
    /// depth terms) from `game_depth` into `out` (R32F, region-sized) on `ctx`, restoring
    /// every state it touched.
    pub fn record_depth_export(
        &self,
        ctx: &ID3D11DeviceContext,
        p: &Params,
        game_depth: &ID3D11ShaderResourceView,
        out: &ID3D11UnorderedAccessView,
    ) -> Result<(), String> {
        self.write_params(ctx, p)?;
        // SAFETY (whole block): as `record_composite`; the depth is unbound from the output
        // merger before it is read.
        unsafe {
            let saved = Saved::capture(ctx, TOUCHED);
            ctx.OMSetRenderTargets(None, None);
            ctx.CSSetShader(&self.cs_export, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.CSSetShaderResources(2, Some(&[Some(game_depth.clone())]));
            ctx.CSSetUnorderedAccessViews(1, 1, Some(&Some(out.clone())), None);
            ctx.Dispatch(p.size[0].div_ceil(16), p.size[1].div_ceil(16), 1);
            ctx.CSSetShaderResources(2, Some(&[None]));
            ctx.CSSetUnorderedAccessViews(1, 1, Some(&None), None);
            saved.restore(ctx);
        }
        Ok(())
    }

    /// Draws our image tone-mapped into `target` (`size`), letterboxed to its aspect.
    pub fn record_preview(
        &self,
        ctx: &ID3D11DeviceContext,
        ours: &ID3D11ShaderResourceView,
        image: [u32; 2],
        target: &ID3D11RenderTargetView,
        size: [u32; 2],
    ) -> Result<(), String> {
        let (sw, sh) = (size[0] as f32, size[1] as f32);
        let scale = (sw / image[0] as f32).min(sh / image[1] as f32);
        let (dw, dh) = (image[0] as f32 * scale, image[1] as f32 * scale);
        let p = Params {
            origin: [((sw - dw) * 0.5) as i32, ((sh - dh) * 0.5) as i32],
            size: image,
            dst_size: [dw, dh],
            ..Params::default()
        };
        self.write_params(ctx, &p)?;
        // SAFETY (whole block): as `record_composite`.
        unsafe {
            let saved = Saved::capture(ctx, TOUCHED);
            ctx.OMSetRenderTargets(None, None);
            ctx.ClearUnorderedAccessViewUint(&self.stats_uav, &[0; 4]);
            ctx.CSSetShader(&self.cs_logavg, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.CSSetShaderResources(0, Some(&[Some(ours.clone())]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(self.stats_uav.clone())), None);
            ctx.Dispatch(image[0].div_ceil(16), image[1].div_ceil(16), 1);
            ctx.CSSetShaderResources(0, Some(&[None]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.ps_preview, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
            ctx.PSSetShaderResources(
                0,
                Some(&[
                    Some(ours.clone()),
                    None,
                    None,
                    None,
                    Some(self.stats_srv.clone()),
                ]),
            );
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: sw,
                Height: sh,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.RSSetScissorRects(Some(&[RECT {
                left: 0,
                top: 0,
                right: size[0] as i32,
                bottom: size[1] as i32,
            }]));
            ctx.OMSetBlendState(None, None, u32::MAX);
            ctx.OMSetDepthStencilState(&self.depth_off, 0);
            ctx.OMSetRenderTargets(Some(&[Some(target.clone())]), None);
            ctx.Draw(3, 0);
            ctx.OMSetRenderTargets(None, None);
            ctx.PSSetShaderResources(0, Some(&[None, None, None, None, None]));
            saved.restore(ctx);
        }
        Ok(())
    }
}

fn tex2d(r: &ID3D11Resource) -> Result<(ID3D11Texture2D, D3D11_TEXTURE2D_DESC), String> {
    let t: ID3D11Texture2D = r.cast().map_err(|_| "not a 2D texture".to_owned())?;
    let mut d = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { t.GetDesc(&mut d) };
    Ok((t, d))
}

/// SRV format for a depth texture format (stencil ignored), as `ssr.rs`.
pub(crate) fn depth_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT => {
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        }
        DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => {
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
        }
        DXGI_FORMAT_R32_TYPELESS | DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_FLOAT => {
            DXGI_FORMAT_R32_FLOAT
        }
        DXGI_FORMAT_R16_TYPELESS | DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_R16_UNORM => {
            DXGI_FORMAT_R16_UNORM
        }
        _ => return None,
    })
}

/// Typed view format of the HDR colour (R11G11B10 in game, RGBA16F / RGBA32F variants).
pub(crate) fn color_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R11G11B10_FLOAT => f,
        DXGI_FORMAT_R16G16B16A16_TYPELESS | DXGI_FORMAT_R16G16B16A16_FLOAT => {
            DXGI_FORMAT_R16G16B16A16_FLOAT
        }
        DXGI_FORMAT_R32G32B32A32_TYPELESS | DXGI_FORMAT_R32G32B32A32_FLOAT => {
            DXGI_FORMAT_R32G32B32A32_FLOAT
        }
        _ => return None,
    })
}

static PIPE: Mutex<Option<Pipe>> = Mutex::new(None);

/// The pipe for `dev` (re-created on a device change).
fn pipe_for<'a>(g: &'a mut Option<Pipe>, dev: &ID3D11Device) -> Result<&'a mut Pipe, String> {
    if g.as_ref().is_none_or(|p| p.device.as_raw() != dev.as_raw()) {
        *g = None;
        *g = Some(Pipe::new(dev)?);
    }
    g.as_mut().ok_or_else(|| "no pipeline".to_owned())
}

// ------------------------------------------------------------------------- FOG hook --

/// Hooks the FOG pass callback (`photo = 1`; installed once, inert while no photo runs).
/// The shaders compile on a background thread.
pub fn install() -> Result<(), String> {
    if INSTALL_TRIED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let _ = std::thread::Builder::new()
        .name("void-photo-shaders".into())
        .spawn(|| {
            if let Err(e) = BYTECODE.as_ref() {
                log::line(&format!("photo: shader compilation failed: {e}"));
            }
        });
    let target = mem::image_base().ok_or("no image base")? + FOG_RVA;
    let mut head = [0u8; 16];
    if !mem::read_into(target, &mut head) || head != FOG_HEAD {
        return Err(format!(
            "FOG callback at {target:#x}: bytes {head:02x?} differ from the analysed build"
        ));
    }
    // SAFETY: verified prologue of `sub_140309430`; the detour has its signature (one
    // integer argument, integer result) and forwards it unchanged.
    let tramp =
        unsafe { minhook::MinHook::create_hook(target as *mut c_void, fog_detour as *mut c_void) }
            .map_err(|e| format!("create_hook FOG: {e:?}"))?;
    ORIG.store(tramp as usize, Ordering::Release);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook FOG: {e:?}"))?;
    HOOKED.store(true, Ordering::Relaxed);
    log::line(&format!("photo: FOG pass callback hooked at {target:#x}"));
    Ok(())
}

unsafe extern "system" fn fog_detour(pass: u64) -> u64 {
    let _ = std::panic::catch_unwind(|| {
        if let Err(e) = before_fog(pass) {
            let n = FAILURES.fetch_add(1, Ordering::Relaxed);
            if n < 5 || n.is_power_of_two() {
                log::line(&format!(
                    "photo: composite skipped: {e} (failure {})",
                    n + 1
                ));
            }
            *lock(&ERROR) = Some(e);
        }
    });
    // SAFETY: MinHook trampoline of the callback, argument unchanged.
    let f = unsafe { std::mem::transmute::<usize, PassFn>(ORIG.load(Ordering::Acquire)) };
    // SAFETY: as above.
    unsafe { f(pass) }
}

/// Region, projection, the hybrid terms' depth capture (when requested) and (when a photo
/// slot is published and `photo_display = pathtrace`) the composite, recorded before the FOG
/// pass.
fn before_fog(pass: u64) -> Result<(), String> {
    let frame = FOG_FRAMES.fetch_add(1, Ordering::AcqRel) + 1;
    let s = settings::get();
    if !s.photo {
        return Ok(());
    }
    let view = mem::u64_at(pass + PASS_VIEW)
        .filter(|&v| v != 0)
        .ok_or("pass view unreadable")?;
    let mut rect = [0i32; 4];
    for (i, r) in rect.iter_mut().enumerate() {
        *r = mem::u32_at(view + crate::ssr::VIEW_VIEWPORT + 4 * i as u64)
            .ok_or("viewport unreadable")? as i32;
    }
    let proj_mem =
        mem::f32s::<16>(view + crate::ssr::VIEW_PROJECTION).ok_or("projection unreadable")?;
    let proj = crate::ao::projection(&proj_mem, 1)?;
    let hdr = crate::fsr_scale::pass_image(pass, crate::ssr::HASH_COLOR).ok_or("no hdrBuffer")?;
    let (_, hd) = tex2d(&hdr)?;
    let tex = [hd.Width, hd.Height];
    let region = crate::ssr::region(rect, tex)
        .ok_or_else(|| format!("viewport {rect:?} outside {tex:?}"))?;
    *lock(&LAST) = Some(FogFrame {
        region,
        tex,
        depth_mul: proj.depth_mul,
        depth_add: proj.depth_add,
        frame,
    });
    if let Some(req) = super::hybrid::capture_request()
        && req.size == [region[2], region[3]]
    {
        record_capture(pass, view, &proj, region, frame, &req)?;
    }
    let Some(set) = current() else {
        return Ok(());
    };
    let Some(k) = set.published() else {
        return Ok(());
    };
    if s.photo_display != super::Display::PathTrace {
        return Ok(());
    }
    if set.size != [region[2], region[3]] {
        return Err(format!(
            "photo is {}x{}, the frame region {}x{} (resolution or render scale changed)",
            set.size[0], set.size[1], region[2], region[3]
        ));
    }
    let Ok(code) = BYTECODE.as_ref() else {
        return Err("shaders not compiled".into());
    };
    let _ = code;
    let zbuffer = crate::fsr_scale::pass_image(pass, HASH_ZBUFFER).ok_or("no zbuffer")?;
    let engine = mem::u64_at(pass + PASS_ENGINE_CTX).ok_or("engine context unreadable")?;
    let d3d = mem::u64_at(engine)
        .filter(|&d| d != 0)
        .ok_or("no D3D11 context")? as *mut c_void;
    // SAFETY: the engine context's live ID3D11DeviceContext (deferred while recording).
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&d3d) }
        .ok_or("null D3D11 context")?
        .clone();
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    let slot = &set.slots[k];
    let (ours, ours_depth) = (
        slot.hdr.srv.clone().ok_or("slot without SRV")?,
        slot.depth.srv.clone().ok_or("slot without SRV")?,
    );
    let mut g = lock(&PIPE);
    let p = pipe_for(&mut g, &dev)?;
    let game_depth = p.srv(&zbuffer, true)?;
    let game_hdr = p.srv(&hdr, false)?;
    let target = p.rtv(&hdr)?;
    let params = Params {
        origin: [region[0] as i32, region[1] as i32],
        size: set.size,
        depth_mul: proj.depth_mul,
        depth_add: proj.depth_add,
        znear: set.znear,
        units: s.photo_units.unwrap_or(0.0),
        fallback: FALLBACK_UNITS,
        tolerance: TOLERANCE,
        dst_size: [0.0; 2],
    };
    p.record_composite(
        &ctx,
        &params,
        &ours,
        &ours_depth,
        &game_depth,
        &game_hdr,
        &target,
        region,
    )?;
    drop(g);
    // The engine skips shader binds whose cached pointer matches (`fsr.rs`).
    let _ = mem::write_bytes(engine + ENGINE_CTX_SHADER_CACHE, &[0u8; 32]);
    set.released[k].store(frame, Ordering::Release);
    if COMPOSITED.fetch_add(1, Ordering::Relaxed) == 0 {
        log::line(&format!(
            "photo: first composite: hdrBuffer {}x{} format {}, zbuffer format {}, region {region:?}, depth z = {} / (d - {}), units {}",
            hd.Width,
            hd.Height,
            hd.Format.0,
            tex2d(&zbuffer).map_or(0, |(_, d)| d.Format.0),
            proj.depth_mul,
            proj.depth_add,
            s.photo_units.map_or("auto".to_owned(), |u| u.to_string())
        ));
    }
    Ok(())
}

static CAPTURES: AtomicU64 = AtomicU64::new(0);

/// The hybrid terms' depth capture of this FOG pass (`hybrid.rs`, module doc): the export
/// into `req`'s texture and the frame's camera, then the request is closed.
fn record_capture(
    pass: u64,
    view: u64,
    proj: &crate::ao::Proj,
    region: [u32; 4],
    frame: u64,
    req: &Arc<super::hybrid::DepthRequest>,
) -> Result<(), String> {
    let view_mem =
        mem::f32s::<16>(view + crate::contact::VIEW_VIEW).ok_or("view matrix unreadable")?;
    let sun = mem::f32s::<3>(view + crate::contact::VIEW_SUN).ok_or("sun direction unreadable")?;
    let size = [region[2], region[3]];
    let hf = super::hybrid::frame_from_view(&view_mem, proj, sun, size).ok_or_else(|| {
        format!("depth capture: degenerate view (matrix {view_mem:?}, sun {sun:?})")
    })?;
    if BYTECODE.as_ref().is_err() {
        return Err("shaders not compiled".into());
    }
    let zbuffer = crate::fsr_scale::pass_image(pass, HASH_ZBUFFER).ok_or("no zbuffer")?;
    let engine = mem::u64_at(pass + PASS_ENGINE_CTX).ok_or("engine context unreadable")?;
    let d3d = mem::u64_at(engine)
        .filter(|&d| d != 0)
        .ok_or("no D3D11 context")? as *mut c_void;
    // SAFETY: the engine context's live ID3D11DeviceContext (deferred while recording).
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&d3d) }
        .ok_or("null D3D11 context")?
        .clone();
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    let uav = req
        .tex
        .uav
        .clone()
        .ok_or("depth capture texture without UAV")?;
    let mut g = lock(&PIPE);
    let p = pipe_for(&mut g, &dev)?;
    let game_depth = p.srv(&zbuffer, true)?;
    let params = Params {
        origin: [region[0] as i32, region[1] as i32],
        size,
        depth_mul: proj.depth_mul,
        depth_add: proj.depth_add,
        ..Params::default()
    };
    p.record_depth_export(&ctx, &params, &game_depth, &uav)?;
    drop(g);
    // The engine skips shader binds whose cached pointer matches (`fsr.rs`).
    let _ = mem::write_bytes(engine + ENGINE_CTX_SHADER_CACHE, &[0u8; 32]);
    req.complete(super::hybrid::Captured {
        fog_frame: frame,
        frame: hf,
        region,
    });
    if CAPTURES.fetch_add(1, Ordering::Relaxed) == 0 {
        log::line(&format!(
            "photo: first depth capture: region {region:?}, zbuffer format {}, depth z = {} / (d - {}), jitter {:?}, camera {:?} forward {:?}, sun {:?}",
            tex2d(&zbuffer).map_or(0, |(_, d)| d.Format.0),
            proj.depth_mul,
            proj.depth_add,
            proj.jitter,
            hf.basis.org,
            hf.basis.fwd,
            hf.sun_dir
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------------- present --

/// A back-buffer capture: requested path, then the staging copy waiting to be read.
enum Capture {
    Requested(PathBuf),
    Copied {
        path: PathBuf,
        staging: ID3D11Texture2D,
        desc: D3D11_TEXTURE2D_DESC,
        tries: u32,
    },
}

// SAFETY: the staging texture is only used on the present path (main thread).
unsafe impl Send for Capture {}

static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
static CAPTURE_DONE: Mutex<Option<Result<PathBuf, String>>> = Mutex::new(None);

/// Asks the present path to save the next presented frame to `path` (PNG).
pub(crate) fn request_capture(path: PathBuf) {
    *lock(&CAPTURE) = Some(Capture::Requested(path));
}

/// The result of the last finished capture (taken).
pub(crate) fn take_capture_result() -> Option<Result<PathBuf, String>> {
    lock(&CAPTURE_DONE).take()
}

/// The present path has work: a capture, or the overlay-mode preview.
pub fn present_wanted() -> bool {
    lock(&CAPTURE).is_some()
        || (settings::get().photo_display == super::Display::Overlay
            && current().is_some_and(|c| c.published().is_some()))
}

/// Present path (main thread, under the engine's immediate-context lock, `overlay.rs`):
/// back-buffer capture for a save, then the overlay-mode preview.
pub fn present(
    ctx: &ID3D11DeviceContext,
    target: &ID3D11Texture2D,
    rtv: &ID3D11RenderTargetView,
    size: [u32; 2],
) -> Result<(), String> {
    capture_step(ctx, target)?;
    if settings::get().photo_display != super::Display::Overlay {
        return Ok(());
    }
    let Some(set) = current() else {
        return Ok(());
    };
    let Some(k) = set.published() else {
        return Ok(());
    };
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    let Ok(mut g) = PIPE.try_lock() else {
        return Ok(());
    };
    let p = pipe_for(&mut g, &dev)?;
    let ours = set.slots[k].hdr.srv.clone().ok_or("slot without SRV")?;
    p.record_preview(ctx, &ours, set.size, rtv, size)?;
    set.released[k].store(fog_frames(), Ordering::Release);
    Ok(())
}

fn capture_step(ctx: &ID3D11DeviceContext, target: &ID3D11Texture2D) -> Result<(), String> {
    let mut g = lock(&CAPTURE);
    match g.take() {
        None => Ok(()),
        Some(Capture::Requested(path)) => {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            // SAFETY: plain getter.
            unsafe { target.GetDesc(&mut desc) };
            let staging_desc = D3D11_TEXTURE2D_DESC {
                MipLevels: 1,
                ArraySize: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
                ..desc
            };
            // SAFETY: plain getter; staging texture with a valid descriptor; copy between
            // textures of identical size and format.
            let staging = unsafe {
                let dev = ctx.GetDevice().map_err(create_err("device"))?;
                let mut t = None;
                dev.CreateTexture2D(&staging_desc, None, Some(&mut t))
                    .map_err(create_err("staging texture"))?;
                let t: ID3D11Texture2D = t.ok_or("no staging texture")?;
                ctx.CopyResource(&t, target);
                t
            };
            *g = Some(Capture::Copied {
                path,
                staging,
                desc,
                tries: 0,
            });
            Ok(())
        }
        Some(Capture::Copied {
            path,
            staging,
            desc,
            tries,
        }) => {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            // SAFETY: staging texture created above, mapped for reading without waiting.
            let r = unsafe {
                ctx.Map(
                    &staging,
                    0,
                    D3D11_MAP_READ,
                    D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                    Some(&mut m),
                )
            };
            if r.is_err() {
                if tries > 120 {
                    *lock(&CAPTURE_DONE) = Some(Err("back-buffer copy never finished".into()));
                    return Ok(());
                }
                *g = Some(Capture::Copied {
                    path,
                    staging,
                    desc,
                    tries: tries + 1,
                });
                return Ok(());
            }
            let (w, h) = (desc.Width as usize, desc.Height as usize);
            let pitch = m.RowPitch as usize;
            let mut raw = vec![0u8; w * 4 * h];
            // SAFETY: the mapping holds `h` rows of `pitch >= 4 w` bytes.
            unsafe {
                for y in 0..h {
                    std::ptr::copy_nonoverlapping(
                        m.pData.cast::<u8>().add(y * pitch),
                        raw.as_mut_ptr().add(y * w * 4),
                        w * 4,
                    );
                }
                ctx.Unmap(&staging, 0);
            }
            let format = desc.Format;
            let _ = std::thread::Builder::new()
                .name("void-photo-png".into())
                .spawn(move || {
                    let r = to_rgb8(&raw, format)
                        .and_then(|rgb| void_rt::image::write_png(&path, w as u32, h as u32, &rgb))
                        .map(|()| path);
                    *lock(&CAPTURE_DONE) = Some(r);
                });
            Ok(())
        }
    }
}

/// 4-byte back-buffer pixels -> RGB8 (8-bit UNORM/sRGB as stored, 10-bit scaled).
pub(crate) fn to_rgb8(raw: &[u8], format: DXGI_FORMAT) -> Result<Vec<u8>, String> {
    let px = raw.as_chunks::<4>().0;
    Ok(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        | DXGI_FORMAT_R8G8B8A8_TYPELESS => px.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
        DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_B8G8R8A8_TYPELESS
        | DXGI_FORMAT_B8G8R8X8_UNORM => px.iter().flat_map(|p| [p[2], p[1], p[0]]).collect(),
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => px
            .iter()
            .flat_map(|p| {
                let v = u32::from_le_bytes(*p);
                [0, 10, 20].map(|s| (((v >> s) & 0x3FF) >> 2) as u8)
            })
            .collect(),
        f => return Err(format!("back-buffer format {} not supported for PNG", f.0)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_matches_cbuffer() {
        assert_eq!(size_of::<Params>(), 48);
    }

    #[test]
    fn back_buffer_conversion() {
        let raw = [10u8, 20, 30, 255, 1, 2, 3, 4];
        assert_eq!(
            to_rgb8(&raw, DXGI_FORMAT_B8G8R8A8_UNORM).unwrap(),
            vec![30, 20, 10, 3, 2, 1]
        );
        let v: u32 = 1023 | (512 << 10) | (4 << 20);
        assert_eq!(
            to_rgb8(&v.to_le_bytes(), DXGI_FORMAT_R10G10B10A2_UNORM).unwrap(),
            vec![255, 128, 1]
        );
        assert!(to_rgb8(&raw, DXGI_FORMAT_R16G16B16A16_FLOAT).is_err());
    }
}
