//! Screen-space reflections at the engine TAA site (docs/reflections.md, lighting-options.md
//! B3, fidelityfx-effects.md Q5, STATUS E067, E071, E072).
//!
//! Material data (docs/reflections.md "Material data", O): the engine keeps no per-pixel
//! normal, roughness or specular target before TAA. The forward pass `sub_140311AA0`
//! writes only `hdrBuffer` (+ `sssBuffer` / `skyCloudMaskBuffer` as a second target), the
//! depth prepass only depth + `motionVectors`; no per-view target of `sub_14036D0C0` holds
//! material data. We patch the lit forward pixel shaders to write one (`patch.rs`,
//! `material.rs`: octahedral normal, roughness, F0 per pixel, `ssr_material = auto`); where
//! it is absent (unpatched shaders, `ssr_material = depth`) reflections use depth normals and
//! a user-set surface model (one roughness and reflectance, optionally floors only).
//!
//! Splice (S2, O): the TAA dispatch `call Ctx_Dispatch` at 0x140327817 in
//! `PostFX_TemporalAA_and_Post` (pass `ARK_RENDERPASS_POSTPROC`), redirected by `fsr.rs`
//! (`fsr::install_taa_site`, also without FSR). The site calls [`capture`] for the pass:
//! - engine TAA (FSR off, or FSR not taking the frame): [`record_engine`] records the work
//!   into the pass's deferred context right before the TAA dispatch (the first GPU work of
//!   that command list), puts back every D3D11 state it touched (`ao::Saved`, CS slots
//!   widened to the highest slot the TAA program binds) and the engine TAA accumulates it;
//! - FSR: the job travels with the FSR job and [`execute`] runs on the immediate context in
//!   the submit hook right before FSR packs its inputs (no extra cross-API hand-off; the
//!   POSTPROC list itself runs after FSR, so recording there would be too late).
//!
//! Inputs (pass lists, O): `hdrBuffer` (write list 0x8F93AB75, = `viewColorMap`, the
//! pre-TAA HDR colour in stored units, rendered in the top-left `m_viewport` region in
//! FSR upscale mode), `decompressedDepth` (read list 0x21181874, = `fx/depthFull`, reversed-Z
//! R24G8), `depthWaterStencilView` (read list 0x7A833082: stencil view of the water depth
//! target `depthWater`, which the water depth pass fills with a copy of the opaque depth and
//! then the water surfaces, `sub_1402FE850` -> `sub_14034F740`, O; through our own depth view:
//! water = in front of the opaque depth, the mask is ignored for a frame in which it covers
//! more than `WATER_MAX_FRACTION`), the view's projection (+29584), view matrix (+29776,
//! world Z-up) and viewport (+30816).
//!
//! Algorithm (compute except the final blend): copy the colour region (radiance source and
//! fallback; same units as the target, so no exposure conversion) -> max-depth pyramid of
//! the hardware depth (7 levels, region padded to multiples of 64) -> per trace texel (full
//! or half resolution; half texel q = full pixel 2q everywhere) a depth-reconstructed
//! normal, a GGX VNDF reflection direction from a fixed 4x4 Hammersley pattern and a
//! hierarchical depth march (min-Z traversal as AMD FidelityFX SSSR, in pixel units) ->
//! hit validation (thickness, back face, own block, screen-edge fade) written as
//! confidence-premultiplied radiance `(c * L, c)` -> one-pass spatial resolve (5x5 tent
//! weights 0.5-1-1-1-0.5 cancel the 4x4 pattern; tangent-plane depth weights) averaging
//! both channels with the same weights -> composite into `hdrBuffer` inside the region:
//! `colour = lerp(colour, L / c, strength * F_schlick * surface * c)`; misses keep the
//! engine colour (its own IBL/specular) exactly. No temporal filter (nothing changes per
//! frame for a still camera; FSR / TAA accumulate).

pub mod material;
pub(crate) mod offline;
pub mod patch;
pub(crate) mod patch_offline;
pub mod rt_reflect;

pub use offline::{RgbImage, SsrReport, offline_check};
pub use patch_offline::{PatchReport, patch_offline_check};

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D11_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, s};

use crate::ao::{self, Proj, Saved, Slots, create_err, dynamic_cb, upload};
use crate::settings::{self, Settings};
use crate::{hlsl, log, mem, notify};

// ---------------------------------------------------------------------- engine sites --

pub(crate) const PASS_VIEW: u64 = 0x70;
pub(crate) const VIEW_PROJECTION: u64 = 29_584;
pub(crate) const VIEW_VIEW: u64 = 29_776;
pub(crate) const VIEW_VIEWPORT: u64 = 30_816;
/// `hdrBuffer` (POSTPROC write list; the TAA input `viewColorMap`).
pub(crate) const HASH_COLOR: u32 = 0x8F93_AB75;
/// `decompressedDepth` (POSTPROC read list; `fx/depthFull`).
pub(crate) const HASH_DEPTH: u32 = 0x2118_1874;
/// `depthWaterStencilView` (POSTPROC read list; the engine looks it up without using it).
pub(crate) const HASH_WATER: u32 = 0x7A83_3082;

// ---------------------------------------------------------------------------- status --

static INSTALL_TRIED: AtomicBool = AtomicBool::new(false);
static HOOKED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
/// Recorded into the POSTPROC deferred list (engine TAA).
static ENGINE: AtomicU64 = AtomicU64::new(0);
/// Run on the immediate context before FSR.
static SUBMIT: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
/// TAA frames with `ssr = 1` skipped because the shaders were still compiling.
static NOT_READY: AtomicU64 = AtomicU64::new(0);
/// Frames read back (a few frames late) whose water mask exceeded `WATER_MAX_FRACTION`
/// and was ignored, and the last read-back coverage (f32 bits, fraction of the region).
static WATER_FALLBACK: AtomicU64 = AtomicU64::new(0);
static WATER_COVERAGE: AtomicU32 = AtomicU32::new(u32::MAX);
static WATER_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);
static FIRST_LOGGED: AtomicBool = AtomicBool::new(false);
static FAIL_NOTIFIED: AtomicBool = AtomicBool::new(false);
static WATER_VIEW_LOGGED: AtomicBool = AtomicBool::new(false);
static MOTION_VIEW_LOGGED: AtomicBool = AtomicBool::new(false);
/// Captures (consecutive values continue the temporal history).
static FRAME: AtomicU32 = AtomicU32::new(0);
/// Frames whose temporal stage started without history (first frame, gap, resize).
static HISTORY_RESETS: AtomicU64 = AtomicU64::new(0);
/// Offline check only (`offline.rs` noise reference): rotate the sample pattern per frame
/// without accumulating, so the mean of many frames is the converged per-frame estimate.
static OFFLINE_ROTATE_ONLY: AtomicBool = AtomicBool::new(false);
/// Offline check only (`offline/temporal.rs` stability before/after): the E074 temporal
/// parameters (mirror history 4 frames, clamp k from 1, rotation everywhere).
static OFFLINE_LEGACY: AtomicBool = AtomicBool::new(false);
/// Offline check only (`offline/temporal.rs` firefly before/after): the E085 linear path (no
/// firefly clamp, plain resolve weights, linear history); E074 (`OFFLINE_LEGACY`) implies it.
static OFFLINE_NO_FIREFLY: AtomicBool = AtomicBool::new(false);
/// Last `ssr` seen by `tick` (0 = not yet, 1 = off, 2 = on).
static LAST_MODE: AtomicU8 = AtomicU8::new(0);
/// CS shader-resource slots saved around the engine-TAA recording: at least 16, widened to
/// the highest slot the TAA program had bound on the first call.
static CS_SLOTS: AtomicUsize = AtomicUsize::new(16);
static ERROR: Mutex<Option<String>> = Mutex::new(None);
static INFO: Mutex<Option<String>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// State for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// The TAA call site is redirected (FSR or SSR installed it).
    pub hooked: bool,
    /// TAA frames seen with `ssr = 1`.
    pub calls: u64,
    /// Frames whose reflections were written (both paths).
    pub applied: u64,
    /// Of those: recorded into the engine's POSTPROC list (engine TAA).
    pub engine: u64,
    /// Of those: run on the immediate context before FSR.
    pub submit: u64,
    /// Skipped: shaders still compiling.
    pub not_ready: u64,
    /// Skipped: inputs missing or a D3D11 error (see `error`).
    pub failures: u64,
    /// Frames whose water mask covered too much and was ignored (read back on the
    /// immediate context: FSR path or while a debug view is shown).
    pub water_fallback: u64,
    /// Last read-back water-mask coverage of the region (0..1).
    pub water_coverage: Option<f32>,
    /// Frames whose temporal accumulation started without history (first frame, frame gap,
    /// resize, debug or setting change).
    pub history_resets: u64,
    /// Hybrid ray-traced reflections (`rt_reflect.rs`).
    pub rt: rt_reflect::Status,
    /// Shader patch and material target (`material.rs`).
    pub material: material::Status,
    /// First call: inputs, formats, sizes, water mask.
    pub info: Option<String>,
    pub error: Option<String>,
}

pub fn status() -> Status {
    let (engine, submit) = (
        ENGINE.load(Ordering::Relaxed),
        SUBMIT.load(Ordering::Relaxed),
    );
    let cov = WATER_COVERAGE.load(Ordering::Relaxed);
    Status {
        hooked: HOOKED.load(Ordering::Relaxed),
        calls: CALLS.load(Ordering::Relaxed),
        applied: engine + submit,
        engine,
        submit,
        not_ready: NOT_READY.load(Ordering::Relaxed),
        failures: FAILURES.load(Ordering::Relaxed),
        water_fallback: WATER_FALLBACK.load(Ordering::Relaxed),
        water_coverage: (cov != u32::MAX).then(|| f32::from_bits(cov)),
        history_resets: HISTORY_RESETS.load(Ordering::Relaxed),
        rt: rt_reflect::status(),
        material: material::status(),
        info: lock(&INFO).clone(),
        error: lock(&ERROR).clone(),
    }
}

fn fail(e: String) {
    let n = FAILURES.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n.is_power_of_two() {
        log::line(&format!("ssr: {e} (failure {})", n + 1));
    }
    if !FAIL_NOTIFIED.swap(true, Ordering::Relaxed) {
        notify::warn(format!("Reflections not applied: {e}"));
    }
    *lock(&ERROR) = Some(e);
}

// --------------------------------------------------------------------------- settings --

/// `ssr_quality` presets: name, maximum hierarchical-march iterations per ray.
pub const QUALITIES: [(&str, u32); 4] = [("low", 24), ("medium", 40), ("high", 64), ("ultra", 96)];
/// Index of `ssr_quality = medium`.
pub const DEFAULT_QUALITY: u8 = 1;

pub fn quality_name(q: u8) -> &'static str {
    QUALITIES[usize::from(q).min(QUALITIES.len() - 1)].0
}

pub fn parse_quality(v: &str) -> Option<u8> {
    QUALITIES.iter().position(|(n, _)| *n == v).map(|i| i as u8)
}

/// `ssr_debug` views: off, raw trace confidence, final blend weight, blended reflection,
/// water mask.
pub const DEBUG_VIEWS: [&str; 9] = [
    "off",
    "hits",
    "confidence",
    "reflection",
    "water",
    "material",
    "history",
    "class",
    "f0",
];

pub fn debug_name(d: u8) -> &'static str {
    DEBUG_VIEWS[usize::from(d).min(DEBUG_VIEWS.len() - 1)]
}

pub fn parse_debug(v: &str) -> Option<u8> {
    DEBUG_VIEWS.iter().position(|n| *n == v).map(|i| i as u8)
}

/// Per frame (main thread, `hook::observe`): toast when `ssr` changes; turning it on
/// installs the TAA-site redirect if that was not tried yet.
pub fn tick() {
    rt_reflect::tick();
    let on = settings::get().ssr;
    let mode = if on { 2 } else { 1 };
    let last = LAST_MODE.swap(mode, Ordering::Relaxed);
    if last == mode {
        return;
    }
    if on && !INSTALL_TRIED.load(Ordering::Relaxed) {
        match install() {
            Ok(()) => log::line("ssr: installed at run time (ssr = 1)"),
            Err(e) => {
                log::line(&format!("ssr: installation failed: {e}"));
                notify::error(format!("Reflections unavailable: {e}"));
            }
        }
    }
    if last != 0 {
        log::line(&format!("ssr: ssr = {}", u8::from(on)));
        notify::info(if on {
            "Screen-space reflections on"
        } else {
            "Screen-space reflections off"
        });
    }
}

/// Redirects the TAA call site (shared with FSR; inert while `ssr` and FSR are off) and
/// compiles the shaders on a background thread; frames before that stay untouched.
pub fn install() -> Result<(), String> {
    if INSTALL_TRIED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    std::thread::spawn(|| {
        if let Err(e) = bytecode() {
            log::line(&format!("ssr: shader compilation failed: {e}"));
        }
    });
    let r = crate::fsr::install_taa_site();
    match &r {
        Ok(()) => {
            HOOKED.store(true, Ordering::Relaxed);
            log::line("ssr: TAA call site redirected; ssr decides per frame");
        }
        Err(e) => *lock(&ERROR) = Some(e.clone()),
    }
    r
}

// ----------------------------------------------------------------------- pure helpers --

/// World -> view-frame rotation as three rows (the transform of [`dir_in_view`]).
pub fn view_rotation(view: &[f32; 16]) -> [[f32; 4]; 3] {
    let row = |i: usize, s: f32| {
        [
            s * view[4 * i],
            s * view[4 * i + 1],
            s * view[4 * i + 2],
            0.0,
        ]
    };
    [row(0, 1.0), row(1, 1.0), row(2, -1.0)]
}

/// World direction `d` (Z-up world) into the screen-consistent view frame (x right, y up,
/// z = positive view depth); `view` is the engine's row-major view matrix applied to
/// column vectors (as `contact.rs` reads it).
pub fn dir_in_view(view: &[f32; 16], d: [f32; 3]) -> Option<[f32; 3]> {
    let r = |i: usize| view[4 * i] * d[0] + view[4 * i + 1] * d[1] + view[4 * i + 2] * d[2];
    let v = [r(0), r(1), -r(2)];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len.is_finite() && len > 1e-4).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

/// `idScreenRect` (x1, y1, x2, y2 inclusive) -> (x, y, w, h) clamped to `tex`.
pub fn region(rect: [i32; 4], tex: [u32; 2]) -> Option<[u32; 4]> {
    let [x1, y1, x2, y2] = rect;
    if x1 < 0 || y1 < 0 || x2 < x1 || y2 < y1 {
        return None;
    }
    let (x, y) = (x1 as u32, y1 as u32);
    if x >= tex[0] || y >= tex[1] {
        return None;
    }
    let w = ((x2 - x1 + 1) as u32).min(tex[0] - x);
    let h = ((y2 - y1 + 1) as u32).min(tex[1] - y);
    Some([x, y, w, h])
}

/// Levels of the max-depth pyramid (level 0 = region pixels).
const HIZ_LEVELS: u32 = 7;

/// Level-0 size of the pyramid: the region rounded up so every level halves exactly.
fn hiz_dims(region: [u32; 2]) -> [u32; 2] {
    let m = 1 << (HIZ_LEVELS - 1);
    [region[0].next_multiple_of(m), region[1].next_multiple_of(m)]
}

/// Trace grid for the region (half resolution: ceil(w / 2) x ceil(h / 2)).
fn trace_dims(region: [u32; 2], half: bool) -> [u32; 2] {
    if half {
        [region[0].div_ceil(2), region[1].div_ceil(2)]
    } else {
        region
    }
}

/// Surfaces count as floors up to this angle from world up (cosine; full weight), fading
/// out over the next 10 degrees.
const FLOOR_COS: f32 = 0.866;
/// Relative tangent-plane distance at which resolve weights reach 0: 1 / this.
const PLANE_SHARPNESS: f32 = 32.0;
/// The water mask is ignored for a frame in which it covers more than this fraction of the
/// region (a water target with unexpected contents must not switch the reflections off).
const WATER_MAX_FRACTION: f64 = 0.6;

fn water_limit(region: [u32; 2]) -> u32 {
    (f64::from(region[0]) * f64::from(region[1]) * WATER_MAX_FRACTION) as u32
}

/// Constant buffer `void_ssr_params` (lanes documented in the HLSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ShaderParams {
    view: [f32; 4],
    proj: [f32; 4],
    region: [f32; 4],
    ints: [u32; 4],
    depth: [f32; 4],
    up: [f32; 4],
    surface: [f32; 4],
    trace: [u32; 4],
    hiz: [u32; 4],
    water: [u32; 4],
    /// World -> view-frame rotation rows (material normals).
    rot: [[f32; 4]; 3],
    /// Frame index, flags (`TEMPORAL_*`), set per frame by `Pipeline::record`.
    temporal: [u32; 4],
    /// Hybrid ray-traced reflections (`rt_reflect.rs`): on, max perceptual roughness, BVH
    /// radius (m), 0; set per frame by `Pipeline::record`.
    rt: [f32; 4],
}

/// `g_temporal.y`: the history of the previous frame is valid / rotate the sample pattern per
/// frame / engine motion vectors bound / rotate only where the previous history holds >= 8
/// frames (previous meta bound at t8 during the trace).
const TEMPORAL_HISTORY: u32 = 1;
const TEMPORAL_ROTATE: u32 = 2;
const TEMPORAL_MOTION: u32 = 4;
const TEMPORAL_GATE: u32 = 8;
/// Offline comparison only (`OFFLINE_LEGACY`): the E074 history and clamp parameters.
const TEMPORAL_LEGACY: u32 = 16;
/// Offline comparison only (`OFFLINE_NO_FIREFLY`): no firefly handling (E085 and earlier).
const TEMPORAL_LINEAR: u32 = 32;

fn shader_params(
    proj: &Proj,
    region: [u32; 4],
    up: [f32; 3],
    rot: [[f32; 4]; 3],
    water: bool,
    s: &Settings,
) -> ShaderParams {
    let [x, y, w, h] = region;
    let t = trace_dims([w, h], s.ssr_half_res);
    let p = hiz_dims([w, h]);
    let roughness = s.ssr_roughness.clamp(0.0, 1.0);
    ShaderParams {
        view: [
            2.0 / proj.p00,
            -2.0 / proj.p11,
            (-1.0 - proj.jitter[0]) / proj.p00,
            (1.0 - proj.jitter[1]) / proj.p11,
        ],
        proj: [proj.p00, proj.p11, proj.jitter[0], proj.jitter[1]],
        region: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
        ints: [w, h, x, y],
        depth: [proj.depth_mul, proj.depth_add, 65_504.0, 1.0 / 16_777_215.0],
        up: [
            up[0],
            up[1],
            up[2],
            if s.ssr_depth_floors_only {
                FLOOR_COS
            } else {
                -2.0
            },
        ],
        surface: [
            roughness * roughness,
            s.ssr_reflectance.clamp(0.0, 1.0),
            s.ssr_strength.max(0.0),
            s.ssr_thickness.max(0.01),
        ],
        trace: [
            t[0],
            t[1],
            u32::from(s.ssr_half_res),
            u32::from(water) | if s.ssr_debug == 4 { 2 } else { 0 },
        ],
        hiz: [p[0], p[1], HIZ_LEVELS, PLANE_SHARPNESS.to_bits()],
        water: [water_limit([w, h]), 0, 0, 0],
        rot,
        temporal: [0; 4],
        rt: [0.0; 4],
    }
}

// ------------------------------------------------------------------------------- HLSL --

const SSR_HLSL: &str = r"
// void-inject screen-space reflections (ssr.rs).
#ifndef SSR_ITERS
#define SSR_ITERS 40
#endif
cbuffer void_ssr_params : register(b0)
{
    float4 g_view;    // view.xy = z * (fp * g_region.zw * g_view.xy + g_view.zw), fp in region pixels
    float4 g_proj;    // P00, P11, jitter x, y (ndc = P * view.xy / z + jitter)
    float4 g_region;  // region size w, h, 1/w, 1/h
    uint4  g_ints;    // w, h, region origin in the engine textures
    float4 g_depth;   // view depth = x / (d - y); far clamp; one R24 step
    float4 g_up;      // world up in the view frame; floor cosine (< -1: every surface)
    float4 g_surface; // depth-fallback GGX alpha, F0; strength, thickness (view units)
    uint4  g_trace;   // trace grid w, h; shift (1 = half resolution); flags (1 = water mask, 2 = write the mask texture, 4 = material target, 8 = F0 colour target)
    uint4  g_hiz;     // pyramid level-0 w, h; levels; plane sharpness (float bits)
    uint4  g_water;   // x: water pixels above which the mask is ignored this frame
    float4 g_rot[3];  // world -> view frame (rows), for material normals
    uint4  g_temporal; // x frame index; y flags (1 = history valid, 2 = per-frame sample rotation, 4 = motion vectors bound)
    float4 g_rt;      // hybrid RT (rt_reflect.rs): x > 0 on; y max perceptual roughness; z BVH radius (m)
};
cbuffer void_ssr_debug : register(b2) { float4 g_debug; }; // uv per pixel (x, y), view, 0

Texture2D<float>  void_ssr_depth : register(t0); // engine hardware depth (reversed-Z)
Texture2D<float4> void_ssr_color : register(t1); // our copy of the colour region
Texture2D<float>  void_ssr_hiz   : register(t2); // max-depth pyramid (or one level)
Texture2D<float4> void_ssr_input : register(t3); // trace / resolve / accumulated result
Texture2D<float>  void_ssr_water : register(t4); // water depth (when g_trace.w & 1)
Texture2D<uint>   void_ssr_count : register(t5); // water pixels of this frame (1x1)
Texture2D<float>  void_ssr_mask  : register(t6); // water mask / surface class (debug views); previous luminance moment (temporal)
Texture2D<float4> void_ssr_material : register(t7); // patched lit shaders' o2 (when g_trace.w & 4)
Texture2D<float4> void_ssr_aux   : register(t8); // trace rays (direction, pdf) / history meta
Texture2D<float4> void_ssr_history : register(t9); // previous accumulation (L c, c)
Texture2D<float>  void_ssr_hdepth : register(t10); // linear depth of the history meta
Texture2D<float2> void_ssr_motion : register(t11); // engine motion vectors (current - previous, viewport UV)
Texture2D<float4> void_ssr_f0     : register(t12); // patched lit shaders' o3 = sqrt(F0.rgb) (when g_trace.w & 8)
RWTexture2D<float>  void_ssr_r_out     : register(u0); // max-depth level / history depth
RWTexture2D<float4> void_ssr_out       : register(u1);
RWTexture2D<uint>   void_ssr_count_out : register(u2);
RWTexture2D<float>  void_ssr_mask_out  : register(u3); // water mask / surface class; luminance moment (temporal)
RWTexture2D<float4> void_ssr_aux_out   : register(u4); // trace rays / history meta
RWTexture2D<float4> void_ssr_rtw_out   : register(u5); // hybrid RT weight (rgb per channel, a scalar)
SamplerState void_ssr_linear : register(s1);

static const float BIG = 3.0e38;

// Non-finite lanes -> 0, others clamped to [0, 65000] (every value written to the colour,
// the result textures and the temporal history: one NaN would spread through the resolve,
// the history and FSR's accumulation).
float4 Safe(float4 c)
{
    bool4 bad = (asuint(c) & 0x7f800000u) == 0x7f800000u;
    return bad ? 0.0 : clamp(c, 0.0, 65000.0);
}

// Fireflies (reflections of small bright lights, E089): rays that hit a light are rare and
// very bright. The resolve clamps each glossy hit to K x the mean luminance of the hits
// around (a lone hit far above its neighbours; K = asfloat(g_temporal.z), rt_reflect.rs
// `FIREFLY_K`). The temporal stage keeps the second moment of the luminance per pixel (t6 /
// u3 during that pass) and widens its history clamp by the reflection's own frame-to-frame
// deviation: before, a frame whose rays missed the light shrank the box and pulled the
// accumulated highlight down, and the next hit spiked it back up (the shimmer), losing part
// of its energy on the way. g_temporal.y & 32: the E085 path (offline comparison only).
float FireflyK() { return asfloat(g_temporal.z); }
bool Fireflies() { return (g_temporal.y & 32u) == 0u; }
float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
static const float PI = 3.14159265;
static const uint BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
// Surface classes (resolve -> class texture, debug view `class`; stored as code / 8).
static const uint CLASS_NONE = 0u;      // sky, far, outside
static const uint CLASS_EXCLUDED = 1u;  // patch marker: vegetation, hair, particles, decals
static const uint CLASS_FALLBACK_OFF = 2u; // no material, depth model rejects (not a floor / not planar)
static const uint CLASS_FALLBACK = 3u;  // no material, depth model reflects
static const uint CLASS_ROUGH = 4u;     // lit material too rough (engine IBL only)
static const uint CLASS_LIT = 5u;       // lit material reflects
static const uint CLASS_WATER = 6u;

float HwDepth(int2 p) { return void_ssr_depth.Load(int3(p + int2(g_ints.zw), 0)); }

float LinZ(float d)
{
    float z = g_depth.x / (d - g_depth.y);
    return (z > 0.0 && z < g_depth.z) ? z : g_depth.z;
}

float LinAt(int2 p) { return LinZ(HwDepth(clamp(p, 0, int2(g_ints.xy) - 1))); }

float3 ViewPos(float2 fp, float z) { return float3(z * (fp * g_region.zw * g_view.xy + g_view.zw), z); }

float2 ToPixel(float3 v)
{
    float2 ndc = g_proj.xy * v.xy / v.z + g_proj.zw;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * g_region.xy;
}

float HwFromZ(float z) { return g_depth.y + g_depth.x / z; }

// Normal from the smaller one-sided depth step on each axis (no step across a silhouette),
// facing the camera.
float3 DepthNormal(int2 p, float3 P)
{
    int2 last = int2(g_ints.xy) - 1;
    float2 c = float2(p) + 0.5;
    float3 px = ViewPos(c + float2(1, 0), LinAt(p + int2(1, 0)));
    float3 nx = ViewPos(c - float2(1, 0), LinAt(p - int2(1, 0)));
    float3 py = ViewPos(c + float2(0, 1), LinAt(p + int2(0, 1)));
    float3 ny = ViewPos(c - float2(0, 1), LinAt(p - int2(0, 1)));
    bool usePx = p.x < last.x && (p.x == 0 || abs(px.z - P.z) < abs(nx.z - P.z));
    bool usePy = p.y < last.y && (p.y == 0 || abs(py.z - P.z) < abs(ny.z - P.z));
    float3 dx = usePx ? px - P : P - nx;
    float3 dy = usePy ? py - P : P - ny;
    float3 n = normalize(cross(dx, dy));
    return dot(n, P) > 0.0 ? -n : n;
}

float SurfaceWeight(float3 N)
{
    return g_up.w < -1.0 ? 1.0 : smoothstep(g_up.w - 0.1, g_up.w, dot(N, g_up.xyz));
}

float3 OctDecode(float2 m)
{
    float2 e = m * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

float2 OctEncode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return e * 0.5 + 0.5;
}

float3 WorldToView(float3 n) { return float3(dot(g_rot[0].xyz, n), dot(g_rot[1].xyz, n), dot(g_rot[2].xyz, n)); }
float3 ViewToWorld(float3 v) { return v.x * g_rot[0].xyz + v.y * g_rot[1].xyz + v.z * g_rot[2].xyz; }

// Material target of the patched lit forward shaders: octahedral world normal, perceptual
// roughness, sqrt(F0) in alpha; alpha byte 0 = no material (unpatched shader, sky,
// cleared), byte 1 = excluded class (patch marker: vegetation, hair, particles, decals).
// N out = the shading normal (view frame); `Surface` checks it against the depth normal.
uint Material(int2 p, float3 Nd, out float3 N, out float rough, out float f0)
{
    N = Nd;
    rough = 0.0;
    f0 = 0.0;
    if ((g_trace.w & 4u) == 0u) return CLASS_NONE;
    float4 m = void_ssr_material.Load(int3(p + int2(g_ints.zw), 0));
    uint a = uint(m.a * 255.0 + 0.5);
    if (a == 0u) return CLASS_NONE;
    if (a == 1u) return CLASS_EXCLUDED;
    N = WorldToView(OctDecode(m.xy));
    rough = m.z;
    f0 = m.a * m.a;
    return CLASS_LIT;
}

// Reversed-Z hardware depth is affine in screen space over a plane: a second difference
// beyond quantisation on either axis, at 1 and 2 pixels, means curved or broken depth
// (foliage, clutter, silhouettes), where depth normals are noise.
bool DepthPlanar(int2 p, float d)
{
    int2 last = int2(g_ints.xy) - 1;
    float tol = 0.001 * d + 4.0 * g_depth.w;
    [unroll] for (int s = 1; s <= 2; s++)
    {
        float sx = HwDepth(min(p + int2(s, 0), last)) + HwDepth(max(p - int2(s, 0), 0)) - 2.0 * d;
        float sy = HwDepth(min(p + int2(0, s), last)) + HwDepth(max(p - int2(0, s), 0)) - 2.0 * d;
        if (max(abs(sx), abs(sy)) > tol * float(s * s)) return false;
    }
    return true;
}

// Surface model at p: the material target where present, reflecting on any orientation by its
// own roughness and F0 (rough surfaces fade out: the engine's IBL already shows them);
// excluded classes never; without material the depth normal + user model, only where the
// depth is planar and (`ssr_depth_floors_only`) the surface faces up. Returns the reflection
// weight (before Fresnel and strength); N, GGX alpha, F0 and the class out.
float Surface(int2 p, float d, float3 Nd, out float3 N, out float alpha, out float f0, out uint cls)
{
    float rough;
    cls = Material(p, Nd, N, rough, f0);
    if (cls == CLASS_LIT)
    {
        alpha = rough * rough;
        float w = 1.0 - smoothstep(0.35, 0.55, rough);
        if (w <= 0.0) cls = CLASS_ROUGH;
        // Shading normal far from the depth normal: on planar depth a normal map (keep the
        // shading normal); elsewhere a silhouette or thin geometry, where the depth normal is
        // noise and the pixel's colour is an edge between surfaces: no reflection (else a
        // bright outline from reflections along garbage normals).
        if (dot(N, Nd) < 0.5 && (g_temporal.y & 16u) != 0u)
        {
            N = Nd; // E074: depth normal with the material's roughness / F0 (comparison)
        }
        else if (dot(N, Nd) < 0.5 && !DepthPlanar(p, d))
        {
            N = Nd;
            w = 0.0;
            cls = CLASS_FALLBACK_OFF;
        }
        return w;
    }
    N = Nd;
    alpha = g_surface.x;
    f0 = g_surface.y;
    if (cls == CLASS_EXCLUDED) return 0.0;
    float w = DepthPlanar(p, d) ? SurfaceWeight(Nd) : 0.0;
    cls = w > 0.0 ? CLASS_FALLBACK : CLASS_FALLBACK_OFF;
    return w;
}

// Split-sum specular reflectance of a GGX surface for the pre-integrated environment
// (Karis 2014 analytic fit): Fresnel averaged over the lobe, lower at grazing angles for rough
// surfaces than plain Schlick.
float3 EnvSpecular(float3 f0, float rough, float NoV)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return saturate(f0 * ab.x + ab.y);
}

// Coloured F0 of a lit material pixel (patched shaders' o3 = sqrt(F0.rgb), alpha 1 where the
// shader's F0 vector was captured); grey `f0` otherwise.
float3 F0Colour(int2 p, uint cls, float f0)
{
    if (cls != CLASS_LIT || (g_trace.w & 8u) == 0u) return f0;
    float4 m = void_ssr_f0.Load(int3(p + int2(g_ints.zw), 0));
    return m.a > 0.75 ? m.rgb * m.rgb : f0;
}

// Reflection weight before strength and confidence: the largest channel of the split-sum
// reflectance (the composite tints per channel by `tint` = F.rgb / that maximum).
float SpecWeight(int2 p, uint cls, float f0, float rough, float NoV, out float3 tint)
{
    float3 F = EnvSpecular(F0Colour(p, cls, f0), rough, NoV);
    float m = max(max(F.r, F.g), F.b);
    tint = m > 1e-4 ? F / m : 1.0;
    return m;
}

float GgxD(float NoH, float a)
{
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float SmithG1(float NoX, float a)
{
    float a2 = a * a;
    return 2.0 * NoX / (NoX + sqrt(a2 + (1.0 - a2) * NoX * NoX));
}

// Height-correlated Smith masking-shadowing.
float SmithG2(float NoV, float NoL, float a)
{
    float a2 = a * a;
    return 2.0 * NoL * NoV / (NoL * sqrt(a2 + (1.0 - a2) * NoV * NoV) + NoV * sqrt(a2 + (1.0 - a2) * NoL * NoL));
}

// `depthWater` is a copy of the opaque depth (`sub_14034F740` draws `decompressedDepth`
// into it at the start of the water depth pass) with the water surfaces drawn on top: water
// is where it lies in front of the opaque depth (reversed-Z: larger = closer).
bool WaterAt(int2 p, float d)
{
    if ((g_trace.w & 1u) == 0u) return false;
    float wd = void_ssr_water.Load(int3(p + int2(g_ints.zw), 0));
    return wd > d + 2.0 * g_depth.w;
}

// The mask is ignored for a frame in which it covers more than g_water.x pixels (a water
// target with other contents must not switch the reflections off).
bool IsWater(int2 p, float d)
{
    return WaterAt(p, d) && void_ssr_count.Load(int3(0, 0, 0)) <= g_water.x;
}

// 4x4 Hammersley set in Bayer order; with temporal accumulation rotated per frame by the R2
// sequence (Cranley-Patterson), so every frame stays stratified per 4x4 block and the history
// sees a different set each frame. With g_temporal.y & 8 the rotation only applies where the
// previous accumulation at pixel p (meta at t8, history length in .z) already holds at least
// 8 frames: where history is rejected or short (first frames, disocclusion, the swaying
// first-person hands and weapon) the fixed pattern stays, which is stable frame to frame.
float2 Xi(uint2 q, int2 p)
{
    uint k = BAYER[(q.x & 3u) + 4u * (q.y & 3u)];
    float2 u = float2((k + 0.5) / 16.0, (reversebits(k) >> 28) / 16.0 + 1.0 / 32.0);
    bool rotate = (g_temporal.y & 2u) != 0u;
    if (rotate && (g_temporal.y & 8u) != 0u)
    {
        float len = (g_temporal.y & 1u) != 0u ? void_ssr_aux.Load(int3(p, 0)).z * 255.0 : 0.0;
        rotate = len >= 3.5;
    }
    if (rotate)
        u = frac(u + frac(float(g_temporal.x & 4095u) * float2(0.7548777, 0.5698403)));
    return u;
}

// Heitz 2018, sampling the GGX distribution of visible normals (local frame, n = z).
float3 SampleGgxVndf(float3 Ve, float a, float2 u)
{
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = 6.2831853 * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
}

float3 Reflected(float3 N, float3 V, float a, float2 u)
{
    if (a < 1e-4) return reflect(-V, N);
    float3 T = normalize(abs(N.y) < 0.999 ? cross(N, float3(0.0, 1.0, 0.0)) : cross(N, float3(1.0, 0.0, 0.0)));
    float3 B = cross(N, T);
    float3 Hl = SampleGgxVndf(float3(dot(V, T), dot(V, B), dot(V, N)), a, u);
    return reflect(-V, Hl.x * T + Hl.y * B + Hl.z * N);
}

// Max-depth pyramid: level 0 = region depth (0 = no surface outside the region); water mask
// and its pixel count for the frame (one atomic per group).
groupshared uint g_waterPixels;

[numthreads(8, 8, 1)]
void cs_hiz0(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0u) g_waterPixels = 0u;
    GroupMemoryBarrierWithGroupSync();
    int2 p = int2(id.xy);
    bool pad = all(id.xy < g_hiz.xy);
    bool inside = all(p < int2(g_ints.xy));
    float d = inside ? HwDepth(p) : 0.0;
    if (pad) void_ssr_r_out[id.xy] = d;
    bool water = inside && WaterAt(p, d);
    if (inside && (g_trace.w & 2u) != 0u) void_ssr_mask_out[id.xy] = water ? 1.0 : 0.0;
    if (water) InterlockedAdd(g_waterPixels, 1u);
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0u && g_waterPixels > 0u) InterlockedAdd(void_ssr_count_out[uint2(0, 0)], g_waterPixels);
}

// One level from the previous one (bound alone at t2); sizes halve exactly.
[numthreads(8, 8, 1)]
void cs_hiz(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    void_ssr_r_out.GetDimensions(w, h);
    if (id.x >= w || id.y >= h) return;
    int2 s = int2(id.xy) * 2;
    float a = void_ssr_hiz.Load(int3(s, 0));
    float b = void_ssr_hiz.Load(int3(s + int2(1, 0), 0));
    float c = void_ssr_hiz.Load(int3(s + int2(0, 1), 0));
    float d = void_ssr_hiz.Load(int3(s + int2(1, 1), 0));
    void_ssr_r_out[id.xy] = max(max(a, b), max(c, d));
}

// Hierarchical march (FidelityFX SSSR's min-Z traversal for reversed depth) in level-0
// pixel units: o + t v, v.xy normalised to one pixel along the major axis, z = hardware
// depth (linear along the projected ray). True = the ray crossed a surface (hit = there).
bool HiZMarch(float3 o, float3 v, out float3 hit)
{
    float3 inv = float3(v.x != 0.0 ? 1.0 / v.x : BIG, v.y != 0.0 ? 1.0 / v.y : BIG, v.z != 0.0 ? 1.0 / v.z : BIG);
    float2 floorOff = float2(v.x < 0.0 ? 0.0 : 1.0, v.y < 0.0 ? 0.0 : 1.0);
    float2 eps = float2(v.x < 0.0 ? -0.005 : 0.005, v.y < 0.0 ? -0.005 : 0.005);
    float2 t2 = (floor(o.xy) + floorOff + eps - o.xy) * inv.xy;
    float t = min(t2.x, t2.y);
    float3 pos = o + t * v;
    int level = 0;
    uint i = 0;
    hit = pos;
    [loop] while (i < SSR_ITERS && level >= 0)
    {
        if (any(pos.xy < 0.0) || any(pos.xy >= g_region.xy) || pos.z >= 1.0 || pos.z <= 0.0)
            return false;
        float size = float(1u << uint(level));
        float2 cell = floor(pos.xy / size);
        float surface = void_ssr_hiz.Load(int3(int2(cell), level));
        float2 planes = (cell + floorOff) * size + eps;
        float3 tt = float3((planes - o.xy) * inv.xy, v.z < 0.0 ? (surface - o.z) * inv.z : BIG);
        float tmin = min(min(tt.x, tt.y), tt.z);
        // Reversed-Z: larger is closer; the ray is in front of the cell's nearest surface.
        bool above = surface < pos.z;
        bool skipped = asuint(tmin) != asuint(tt.z) && above;
        t = above ? tmin : t;
        pos = o + t * v;
        level = min(level + (skipped ? 1 : -1), int(g_hiz.z) - 1);
        i++;
    }
    hit = pos;
    return level < 0;
}

float4 Validate(int2 p, float3 R, float3 hit)
{
    int2 h = int2(hit.xy);
    if (any(h < 0) || any(h >= int2(g_ints.xy))) return 0.0;
    int block = 1 << g_trace.z;
    if (all(abs(h - p) <= block)) return 0.0;
    float ds = HwDepth(h);
    if (ds <= 0.0) return 0.0;
    float zs = LinZ(ds);
    float zr = LinZ(max(hit.z, 1e-9));
    // Thickness: the ray depth at the crossing within thickness (+1 % of the depth).
    float gap = abs(zr - zs) - 0.01 * zs;
    float conf = 1.0 - smoothstep(0.5 * g_surface.w, g_surface.w, gap);
    // Back faces of the hit surface do not reflect.
    float3 Nh = DepthNormal(h, ViewPos(float2(h) + 0.5, zs));
    conf *= saturate(-4.0 * dot(Nh, R));
    // Fade toward the region border (5 % of the smaller side).
    float2 border = min(hit.xy, g_region.xy - hit.xy);
    conf *= saturate(min(border.x, border.y) / (0.05 * min(g_region.x, g_region.y)));
    if (conf <= 0.0) return 0.0;
    return float4(void_ssr_color.Load(int3(h, 0)).rgb * conf, conf);
}

// One ray at p. aux out: (view-frame direction, pdf of that direction; 65000 = mirror;
// -1 = no ray at all: no surface, excluded, water, below the horizon).
float4 Trace(int2 p, uint2 q, out float4 aux)
{
    aux = float4(0.0, 0.0, 0.0, -1.0);
    float d = HwDepth(p);
    if (d <= 0.0) return 0.0;
    float z = LinZ(d);
    if (z >= g_depth.z * 0.999 || IsWater(p, d)) return 0.0;
    float3 P = ViewPos(float2(p) + 0.5, z);
    float3 Nd = DepthNormal(p, P);
    float3 N;
    float alpha, f0;
    uint cls;
    if (Surface(p, d, Nd, N, alpha, f0, cls) <= 0.0) return 0.0;
    float3 V = normalize(-P);
    float3 R = Reflected(N, V, alpha, Xi(q, p));
    if (dot(R, N) <= 0.0) return 0.0;
    float NoV = saturate(dot(N, V)) + 1e-4;
    float pdf = alpha < 1e-4 ? 65000.0
        : min(SmithG1(NoV, alpha) * GgxD(saturate(dot(N, normalize(V + R))), alpha) / (4.0 * NoV), 65000.0);
    aux = float4(R, pdf);
    // Screen-space ray (pixel, hardware depth); a ray toward the camera ends at half depth.
    float len = R.z < 0.0 ? min(z, -0.5 * z / R.z) : z;
    float3 E = P + R * len;
    float3 o = float3(float2(p) + 0.5, d);
    float3 v = float3(ToPixel(E), HwFromZ(E.z)) - o;
    float m = max(abs(v.x), abs(v.y));
    if (m < 1e-3) return 0.0;
    float3 hit;
    if (!HiZMarch(o, v / m, hit)) return 0.0;
    return Validate(p, R, hit);
}

// Trace grid: confidence-premultiplied radiance (c * L, c) and the ray (direction, pdf).
[numthreads(8, 8, 1)]
void cs_trace(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_trace.xy)) return;
    int2 p = int2(id.xy) << g_trace.z;
    float4 r = 0.0;
    float4 aux = float4(0.0, 0.0, 0.0, -1.0);
    [branch] if (all(p < int2(g_ints.xy))) r = Trace(p, id.xy, aux);
    void_ssr_out[id.xy] = Safe(r);
    void_ssr_aux_out[id.xy] = aux;
}

// Resolve: 8x8 region pixels per group; trace texels of the group plus a 2-texel apron in
// groupshared memory (at most 12x12). Every neighbour ray is reused for this pixel's lobe
// with the ratio estimator (Stachowiak 2015, FidelityFX SSSR): weight = this pixel's GGX
// BRDF x cosine / the pdf the ray was drawn with, times the 5x5 tent and the tangent-plane
// depth weight (near-mirror pixels: tent and plane only). Glossy pixels (stochastic rays)
// clamp each hit to FireflyK() x the mean luminance of the hits around (fireflies: a rare ray
// hitting a small bright light). Writes (L, blend weight) and the surface class.
groupshared float4 g_tileL[144];
groupshared float4 g_tileR[144];
groupshared float3 g_tileP[144];

[numthreads(8, 8, 1)]
void cs_resolve(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    int s = int(g_trace.z);
    int2 origin = (int2(gid.xy) * 8 >> s) - 2;
    int side = (8 >> s) + 4;
    int2 lastT = int2(g_trace.xy) - 1;
    int2 lastP = int2(g_ints.xy) - 1;
    for (uint i = gi; i < uint(side * side); i += 64u)
    {
        int2 tq = clamp(origin + int2(i % uint(side), i / uint(side)), 0, lastT);
        int2 tp = min(tq << s, lastP);
        g_tileL[i] = void_ssr_input.Load(int3(tq, 0));
        g_tileR[i] = void_ssr_aux.Load(int3(tq, 0));
        g_tileP[i] = ViewPos(float2(tp) + 0.5, LinAt(tp));
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 id = gid.xy * 8u + gt.xy;
    if (any(id >= g_ints.xy)) return;
    int2 p = int2(id);
    float4 o = 0.0;
    uint cls = CLASS_NONE;
    float3 tint = 1.0;
    float d = HwDepth(p);
    float z = LinZ(d);
    [branch] if (d > 0.0 && z < g_depth.z * 0.999)
    {
        float3 P = ViewPos(float2(p) + 0.5, z);
        float3 Nd = DepthNormal(p, P);
        float3 N;
        float alpha, f0;
        float surface = Surface(p, d, Nd, N, alpha, f0, cls);
        if (IsWater(p, d))
        {
            cls = CLASS_WATER;
            surface = 0.0;
        }
        if (surface > 0.0)
        {
            float3 V = normalize(-P);
            float NoV = saturate(dot(N, V)) + 1e-4;
            float F = SpecWeight(p, cls, f0, sqrt(alpha), NoV, tint);
            bool ratio = alpha >= 0.0025;
            int2 c = (p >> s) - origin;
            float sharp = asfloat(g_hiz.w);
            // E085 and earlier (offline comparison): no firefly handling.
            bool fire = ratio && Fireflies();
            float cap = 65000.0;
            if (fire)
            {
                float msum = 0.0, mn = 0.0;
                [unroll] for (int dy = -2; dy <= 2; dy++)
                {
                    [unroll] for (int dx = -2; dx <= 2; dx++)
                    {
                        int k = (c.y + dy) * side + c.x + dx;
                        float4 h = g_tileL[k];
                        float plane = abs(dot(N, g_tileP[k] - P)) / z;
                        if (g_tileR[k].w >= 0.0 && h.a > 1e-3 && plane * sharp < 1.0)
                        {
                            msum += Luma(h.rgb / h.a);
                            mn += 1.0;
                        }
                    }
                }
                cap = FireflyK() * msum / max(mn, 1.0) + 1e-3;
            }
            float3 accL = 0.0;
            float accK = 0.0;
            float wsum = 0.0;
            [unroll] for (int dy = -2; dy <= 2; dy++)
            {
                [unroll] for (int dx = -2; dx <= 2; dx++)
                {
                    int k = (c.y + dy) * side + c.x + dx;
                    float4 ray = g_tileR[k];
                    float tent = (abs(dx) == 2 ? 0.5 : 1.0) * (abs(dy) == 2 ? 0.5 : 1.0);
                    float plane = abs(dot(N, g_tileP[k] - P)) / z;
                    float w = ray.w < 0.0 ? 0.0 : tent * saturate(1.0 - plane * sharp);
                    if (ratio && w > 0.0)
                    {
                        float NoL = dot(N, ray.xyz);
                        float NoH = saturate(dot(N, normalize(V + ray.xyz)));
                        w = NoL <= 0.0 ? 0.0
                            : w * GgxD(NoH, alpha) * SmithG2(NoV, NoL, alpha) / (4.0 * NoV * max(ray.w, 1e-3));
                    }
                    float4 h = g_tileL[k];
                    if (h.a > 1e-6)
                    {
                        float3 l = h.rgb / h.a;
                        float y = Luma(l);
                        if (fire && y > cap) l *= cap / y;
                        accL += w * h.a * l;
                        accK += w * h.a;
                    }
                    wsum += w;
                }
            }
            float conf = saturate(accK / max(wsum, 1e-20));
            float3 L = conf > 1e-4 ? accL / accK : 0.0;
            o = float4(L, saturate(g_surface.z * F * surface) * conf);
        }
    }
    void_ssr_out[id] = Safe(o);
    void_ssr_aux_out[id] = float4(tint, 1.0);
    void_ssr_mask_out[id] = float(cls) / 8.0;
}

float3 YCoCg(float3 c) { return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25))); }
float3 FromYCoCg(float3 y) { return float3(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z); }

// History length a pixel may reach: long while the surface stands still (the reflection does
// not change; every frame only adds samples), shortened by the surface's own motion in pixels
// (reflections move with the reflected scene, not with the surface; the first-person weapon's
// idle sway moves every frame).
float MaxHistory(float rough, float moved)
{
    if ((g_temporal.y & 16u) != 0u) // E074 parameters (offline comparison only)
        return max(lerp(4.0, 32.0, saturate((rough - 0.05) / 0.3)) / (1.0 + 0.5 * moved), 2.0);
    return max(lerp(24.0, 64.0, saturate(rough / 0.35)) / (1.0 + moved), 2.0);
}

// Temporal accumulation (region pixels): reprojects the previous accumulation with the
// engine motion vectors (bilinear taps that pass a depth and world-normal test), clamps it to
// the YCoCg mean +- k sigma of the current 3x3 (k grows with roughness) and its confidence
// to the current maximum, then blends 1 / history length. Writes the accumulation
// (L c, c), meta (octahedral world normal, history length / 255, roughness) and linear depth.
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_ints.xy)) return;
    int2 p = int2(id.xy);
    float d = HwDepth(p);
    float z = LinZ(d);
    float4 acc = 0.0;
    float4 meta = 0.0;
    float moment = 0.0;
    [branch] if (d > 0.0 && z < g_depth.z * 0.999 && !IsWater(p, d))
    {
        float3 P = ViewPos(float2(p) + 0.5, z);
        float3 Nd = DepthNormal(p, P);
        float3 N;
        float alpha, f0;
        uint cls;
        if (Surface(p, d, Nd, N, alpha, f0, cls) > 0.0)
        {
            float rough = sqrt(alpha);
            float3 Nw = ViewToWorld(N);
            float4 cur = void_ssr_input.Load(int3(p, 0));
            bool fire = Fireflies();
            float4 prev = 0.0;
            float len = 0.0;
            // Second moment of the luminance (YCoCg Y) over the history (t6 during this pass).
            float mom = 0.0;
            // Surface motion in pixels: reflections move with the reflected scene, not with
            // the surface, so history is shortened and clamped tighter while it moves.
            float moved = 0.0;
            if ((g_temporal.y & 1u) != 0u)
            {
                float2 mv = (g_temporal.y & 4u) != 0u ? void_ssr_motion.Load(int3(p + int2(g_ints.zw), 0)) : 0.0;
                moved = length(mv * g_region.xy);
                float2 b = float2(p) - mv * g_region.xy;
                int2 i0 = int2(floor(b));
                float2 f = b - float2(i0);
                float wsum = 0.0;
                [unroll] for (int t = 0; t < 4; t++)
                {
                    int2 o = int2(t & 1, t >> 1);
                    int2 q = i0 + o;
                    if (any(q < 0) || any(q >= int2(g_ints.xy))) continue;
                    float4 m = void_ssr_aux.Load(int3(q, 0));
                    if (m.z <= 0.0) continue;
                    float zq = void_ssr_hdepth.Load(int3(q, 0));
                    float wb = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y);
                    float wz = abs(zq - z) <= 0.05 * z + 0.02 ? 1.0 : 0.0;
                    float wn = saturate((dot(OctDecode(m.xy), Nw) - 0.8) / 0.15);
                    float w = wb * wz * wn;
                    prev += w * void_ssr_history.Load(int3(q, 0));
                    mom += w * void_ssr_mask.Load(int3(q, 0));
                    len += w * m.z * 255.0;
                    wsum += w;
                }
                if (wsum > 0.05)
                {
                    prev /= wsum;
                    len /= wsum;
                    mom /= wsum;
                }
                else
                {
                    prev = 0.0;
                    len = 0.0;
                    mom = 0.0;
                }
            }
            if (len > 0.0)
            {
                float3 m1 = 0.0, m2 = 0.0;
                float n = 0.0, amax = 0.0;
                [unroll] for (int dy = -1; dy <= 1; dy++)
                {
                    [unroll] for (int dx = -1; dx <= 1; dx++)
                    {
                        float4 c = void_ssr_input.Load(int3(clamp(p + int2(dx, dy), 0, int2(g_ints.xy) - 1), 0));
                        amax = max(amax, c.a);
                        if (c.a > 1e-3)
                        {
                            float3 y = YCoCg(c.rgb);
                            m1 += y;
                            m2 += y * y;
                            n += 1.0;
                        }
                    }
                }
                // The resolve output is spatially smooth but noisy over time: its 3x3 spread
                // underestimates the frame-to-frame variance, so a tight box clamps the history
                // onto this frame's noise (flicker on a static glossy floor). Standing still
                // (no surface motion) the box widens with roughness (glossy reflections are
                // blurred, a slow reaction to moving reflected objects hides in the blur;
                // mirrors stay responsive); moving surfaces clamp tightly.
                bool legacy = (g_temporal.y & 16u) != 0u;
                float still = (1.0 - saturate(moved / 2.0)) * (legacy ? 0.0 : 1.0);
                float glossy = saturate(rough / 0.35);
                float a = min(prev.a, lerp(amax * 1.25 + 0.02, 1.0, still * glossy));
                float3 Lp = prev.a > 1e-4 ? prev.rgb / prev.a : 0.0;
                if (n > 0.0)
                {
                    float3 mu = m1 / n;
                    float3 sigma = sqrt(max(m2 / n - mu * mu, 0.0));
                    // The 3x3 of the resolve output is nearly constant where the 5x5 resolve
                    // smoothed it, so its spread collapses although the value still changes
                    // from frame to frame: a box of zero size replaces the history with this
                    // frame (full flicker). Floor: 5 % of the luminance.
                    if (!legacy) sigma = max(sigma, 0.05 * abs(mu.x) + 1e-4);
                    // Fireflies: the box also spans the luminance's own frame-to-frame
                    // deviation over the history, so a frame whose rays missed a small light
                    // does not pull the accumulated highlight down (and the next hit up).
                    if (fire)
                    {
                        float h = YCoCg(Lp).x;
                        sigma.x = max(sigma.x, sqrt(max(mom - h * h, 0.0)));
                    }
                    float k = legacy
                        ? lerp(lerp(1.0, 2.5, glossy), 1.0, saturate(moved / 4.0))
                        : lerp(lerp(1.25, 2.0, glossy), lerp(3.0, 12.0, glossy), still);
                    Lp = FromYCoCg(clamp(YCoCg(Lp), mu - k * sigma, mu + k * sigma));
                }
                prev = float4(max(Lp, 0.0) * a, a);
            }
            float yc = YCoCg(cur.rgb).x;
            moment = len > 0.0 ? mom : yc * yc;
            len = min(len + 1.0, MaxHistory(rough, moved));
            acc = lerp(prev, float4(cur.rgb * cur.a, cur.a), 1.0 / len);
            moment = lerp(moment, yc * yc, 1.0 / len);
            meta = float4(OctEncode(Nw), len / 255.0, rough);
        }
    }
    void_ssr_out[id.xy] = Safe(acc);
    void_ssr_aux_out[id.xy] = meta;
    void_ssr_r_out[id.xy] = z;
    void_ssr_mask_out[id.xy] = min(moment, 65000.0);
}

// Roughness-scaled bilateral blur of the accumulation (3x3 taps at a stride growing with
// roughness, doubled while the history is short; plane and normal weights): (L, blend weight).
[numthreads(8, 8, 1)]
void cs_blur(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_ints.xy)) return;
    int2 p = int2(id.xy);
    float4 m = void_ssr_aux.Load(int3(p, 0));
    float4 acc = void_ssr_input.Load(int3(p, 0));
    [branch] if (m.z > 0.0 && m.w > 0.08)
    {
        float z = void_ssr_hdepth.Load(int3(p, 0));
        float3 nw = OctDecode(m.xy);
        float3 nv = WorldToView(nw);
        float3 P = ViewPos(float2(p) + 0.5, z);
        int stride = clamp(int(m.w * 8.0) * (m.z * 255.0 < 4.0 ? 2 : 1), 1, 6);
        float wsum = 1.0;
        [unroll] for (int dy = -1; dy <= 1; dy++)
        {
            [unroll] for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0) continue;
                int2 q = clamp(p + int2(dx, dy) * stride, 0, int2(g_ints.xy) - 1);
                float4 mq = void_ssr_aux.Load(int3(q, 0));
                if (mq.z <= 0.0) continue;
                float3 Pq = ViewPos(float2(q) + 0.5, void_ssr_hdepth.Load(int3(q, 0)));
                float plane = abs(dot(nv, Pq - P)) / z;
                float w = (dx == 0 || dy == 0 ? 0.6 : 0.36) * saturate(1.0 - plane * 50.0)
                    * pow(saturate(dot(OctDecode(mq.xy), nw)), 16.0);
                acc += w * void_ssr_input.Load(int3(q, 0));
                wsum += w;
            }
        }
        acc /= wsum;
    }
    void_ssr_out[id.xy] = Safe(acc.a > 1e-4 ? float4(acc.rgb / acc.a, acc.a) : 0.0);
}

// Hybrid ray-traced reflections (rt_reflect.rs, FSR path only), after the final SSR result
// (t3, (L, weight)): per half-resolution texel, for its 2x2 pixels the weight the ray-traced
// reflection may still take (the SSR blend weight it could not fill, faded by roughness up to
// g_rt.y and by distance from 0.5 to 0.7 x the BVH radius g_rt.z, where the rays' confidence
// ends; rgb tinted by the material's reflectance as the composite) into u5, and one ray request
// when any of them wants one: u1 = (view-frame origin pushed off the surface, TMin), u4 =
// (view-frame shading normal, GGX alpha); w = 0: no ray. The request's surface is the first of
// the pixels that reflects at all (not the first that still wants rays), so it does not hop
// within the 2x2 while the SSR coverage changes. The D3D12 side draws the GGX directions
// (live: one per frame, photo: many).
[numthreads(8, 8, 1)]
void cs_rt_classify(uint3 id : SV_DispatchThreadID)
{
    uint2 hs = (g_ints.xy + 1u) >> 1;
    if (any(id.xy >= hs)) return;
    float4 r0 = 0.0, r1 = 0.0;
    bool any_want = false;
    [unroll] for (uint k = 0; k < 4; k++)
    {
        int2 p = int2(id.xy * 2u + uint2(k & 1u, k >> 1));
        if (any(p >= int2(g_ints.xy))) continue;
        float want = 0.0;
        float3 tint = 1.0;
        float d = HwDepth(p);
        float z = LinZ(d);
        [branch] if (d > 0.0 && z < g_depth.z * 0.999 && !IsWater(p, d))
        {
            float3 P = ViewPos(float2(p) + 0.5, z);
            float3 Nd = DepthNormal(p, P);
            float3 N;
            float alpha, f0;
            uint cls;
            float s = Surface(p, d, Nd, N, alpha, f0, cls);
            if (s > 0.0)
            {
                float3 V = normalize(-P);
                float rough = sqrt(alpha);
                float W = saturate(g_surface.z * SpecWeight(p, cls, f0, rough, saturate(dot(N, V)) + 1e-4, tint) * s);
                float filled = void_ssr_input.Load(int3(p, 0)).a;
                float fade = (1.0 - smoothstep(0.8 * g_rt.y, g_rt.y, rough))
                    * (1.0 - smoothstep(0.5 * g_rt.z, 0.7 * g_rt.z, length(P)));
                want = max(W - filled, 0.0) * fade;
                any_want = any_want || want > 1.0 / 256.0;
                if (W * fade > 1.0 / 256.0 && r0.w <= 0.0)
                {
                    // Our geometry and the game's differ by centimetres (E070): origin off the
                    // surface and near hits skipped, both growing with depth.
                    r0 = float4(P + N * max(0.02, 0.003 * z), max(0.05, 0.005 * z));
                    r1 = float4(N, alpha);
                }
            }
        }
        void_ssr_rtw_out[p] = want > 1.0 / 256.0 ? float4(want * tint, want) : 0.0;
        // Hardware-depth copy for D3D12 consumers (photo mode AO/shadows, `shared_inputs`).
        void_ssr_r_out[p] = d;
    }
    void_ssr_out[id.xy] = any_want ? r0 : 0.0;
    void_ssr_aux_out[id.xy] = any_want ? r1 : 0.0;
}

float4 vs_full(uint id : SV_VertexID) : SV_Position
{
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Into hdrBuffer inside the region with dual-source blending (ONE / INV_SRC1_COLOR, RGB
// only): colour = colour x (1 - w) + L x w per channel, w = blend weight x the material's
// reflectance tint (t8).
void ps_composite(float4 pos : SV_Position, out float4 c0 : SV_Target0, out float4 c1 : SV_Target1)
{
    int2 p = int2(pos.xy) - int2(g_ints.zw);
    float4 r = void_ssr_input.Load(int3(p, 0));
    r = Safe(r);
    if (r.a < 1.0 / 1024.0) discard;
    float3 w = Safe(float4(saturate(r.a * void_ssr_aux.Load(int3(p, 0)).rgb), 0.0)).rgb;
    c0 = float4(r.rgb * w, 0.0);
    c1 = float4(w, 0.0);
}

// g_debug.z: 1 = raw trace confidence, 2 = blend weight, 3 = reflection x weight,
// 4 = water mask (white; red when the frame's coverage disabled it), g_debug.w = limit,
// 5 = material target (octahedral normal, roughness; black = no material, magenta =
// excluded), 6 = history length (black 0, green -> white up to 32 frames),
// 7 = surface class (CLASS_* colours below), 8 = F0 colour target (sqrt(F0.rgb); black =
// none, blue-tinted = luminance only).
float4 ps_debug(float4 pos : SV_Position) : SV_Target
{
    float2 uv = pos.xy * g_debug.xy;
    if (g_debug.z > 7.5)
    {
        float4 m = void_ssr_input.SampleLevel(void_ssr_linear, uv, 0.0);
        if (m.a > 0.75) return float4(m.rgb, 1.0);
        return float4(m.a > 0.25 ? m.rgb * 0.5 + float3(0.0, 0.0, 0.3) : 0.0, 1.0);
    }
    if (g_debug.z > 6.5)
    {
        uint c = uint(void_ssr_mask.SampleLevel(void_ssr_linear, uv, 0.0) * 8.0 + 0.5);
        static const float3 COLOURS[7] = {
            float3(0.0, 0.0, 0.0),    // none
            float3(1.0, 0.0, 1.0),    // excluded (vegetation, hair, particles, decals)
            float3(0.35, 0.35, 0.35), // no material, depth model rejects
            float3(0.1, 0.4, 1.0),    // no material, depth model reflects
            float3(0.6, 0.4, 0.1),    // lit, too rough
            float3(0.1, 1.0, 0.2),    // lit, reflects
            float3(0.0, 0.8, 0.8)     // water
        };
        return float4(COLOURS[min(c, 6u)], 1.0);
    }
    if (g_debug.z > 5.5)
    {
        float4 m = void_ssr_input.SampleLevel(void_ssr_linear, uv, 0.0);
        float h = saturate(m.z * 255.0 / 32.0);
        return float4(h * h, h, h * h, 1.0);
    }
    if (g_debug.z > 4.5)
    {
        float4 m = void_ssr_input.SampleLevel(void_ssr_linear, uv, 0.0);
        uint a = uint(m.a * 255.0 + 0.5);
        return float4(a == 1u ? float3(1.0, 0.0, 1.0) : (a > 1u ? m.rgb : 0.0), 1.0);
    }
    if (g_debug.z > 3.5)
    {
        float m = void_ssr_mask.SampleLevel(void_ssr_linear, uv, 0.0);
        bool off = float(void_ssr_count.Load(int3(0, 0, 0))) > g_debug.w;
        return float4(m, off ? 0.0 : m, off ? 0.0 : m, 1.0);
    }
    float4 r = void_ssr_input.SampleLevel(void_ssr_linear, uv, 0.0);
    if (g_debug.z < 2.5) return float4(r.aaa, 1.0);
    float3 c = r.rgb * r.a;
    return float4(c / (1.0 + c), 1.0);
}
";

/// Compiled shaders (device independent).
pub struct Bytecode {
    hiz0: Vec<u8>,
    hiz: Vec<u8>,
    /// `cs_trace` per `QUALITIES` entry (iterations as a compile-time constant).
    trace: Vec<Vec<u8>>,
    resolve: Vec<u8>,
    temporal: Vec<u8>,
    blur: Vec<u8>,
    rt_classify: Vec<u8>,
    vs: Vec<u8>,
    composite: Vec<u8>,
    debug: Vec<u8>,
    pub compile_ms: f64,
}

/// Compiled on first use (background thread from `install`, or the offline check); render
/// threads only use it once `BYTECODE_READY` is set.
static BYTECODE: LazyLock<Result<Bytecode, String>> = LazyLock::new(|| {
    let t = std::time::Instant::now();
    let c = |entry, target| hlsl::compile(SSR_HLSL, entry, target);
    Ok(Bytecode {
        hiz0: c(s!("cs_hiz0"), s!("cs_5_0"))?,
        hiz: c(s!("cs_hiz"), s!("cs_5_0"))?,
        trace: QUALITIES
            .iter()
            .map(|(_, iters)| {
                let src = format!("#define SSR_ITERS {iters}\n{SSR_HLSL}");
                hlsl::compile(&src, s!("cs_trace"), s!("cs_5_0"))
            })
            .collect::<Result<_, _>>()?,
        resolve: c(s!("cs_resolve"), s!("cs_5_0"))?,
        temporal: c(s!("cs_temporal"), s!("cs_5_0"))?,
        blur: c(s!("cs_blur"), s!("cs_5_0"))?,
        rt_classify: c(s!("cs_rt_classify"), s!("cs_5_0"))?,
        vs: c(s!("vs_full"), s!("vs_5_0"))?,
        composite: c(s!("ps_composite"), s!("ps_5_0"))?,
        debug: c(s!("ps_debug"), s!("ps_5_0"))?,
        compile_ms: t.elapsed().as_secs_f64() * 1000.0,
    })
});
static BYTECODE_READY: AtomicBool = AtomicBool::new(false);

/// The compiled shaders (blocks while compiling).
pub fn bytecode() -> Result<&'static Bytecode, String> {
    let r = BYTECODE.as_ref().map_err(Clone::clone);
    BYTECODE_READY.store(true, Ordering::Release);
    r
}

// ---------------------------------------------------------------------------- GPU side --

/// CS shader-resource slots the HLSL declares (t0-t12).
const CS_SRVS: usize = 13;

/// Slots the passes and the debug view bind: CS t0-t12 (and up to `cs_srv`), b0-b2, s0-s1,
/// u0-u4.
fn touched(cs_srv: usize) -> Slots {
    Slots {
        srv: cs_srv.max(CS_SRVS),
        cb: 3,
        sampler: 2,
        uav: 8,
    }
}

/// SRV format for a depth texture format (stencil ignored).
fn depth_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
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

/// Typed view format of the HDR colour (R11G11B10 in game, RGBA16F for the TAA material
/// variant `g_mtrTAA_R16G16B16`).
fn color_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
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

fn tex2d(r: &ID3D11Resource) -> Result<(ID3D11Texture2D, D3D11_TEXTURE2D_DESC), String> {
    let t: ID3D11Texture2D = r.cast().map_err(|_| "not a 2D texture".to_owned())?;
    let mut d = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { t.GetDesc(&mut d) };
    Ok((t, d))
}

/// Views we created for engine textures, keyed by the resource pointer.
#[derive(Default)]
struct Views {
    depth: HashMap<usize, (ID3D11Resource, ID3D11ShaderResourceView)>,
    color: HashMap<usize, (ID3D11Resource, ID3D11RenderTargetView)>,
    motion: HashMap<usize, (ID3D11Resource, ID3D11ShaderResourceView)>,
}

const MAX_CACHED_VIEWS: usize = 4;

fn texture(
    dev: &ID3D11Device,
    size: [u32; 2],
    format: DXGI_FORMAT,
    mips: u32,
    bind: i32,
) -> Result<ID3D11Texture2D, String> {
    let d = D3D11_TEXTURE2D_DESC {
        Width: size[0],
        Height: size[1],
        MipLevels: mips,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind as u32,
        ..Default::default()
    };
    let mut t = None;
    // SAFETY: valid descriptor and out pointer.
    unsafe { dev.CreateTexture2D(&d, None, Some(&mut t)) }.map_err(create_err("texture"))?;
    t.ok_or_else(|| "no texture".to_owned())
}

fn srv(
    dev: &ID3D11Device,
    t: &ID3D11Texture2D,
    format: DXGI_FORMAT,
    mip: u32,
    mips: u32,
) -> Result<ID3D11ShaderResourceView, String> {
    let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_SRV {
                MostDetailedMip: mip,
                MipLevels: mips,
            },
        },
    };
    let mut v = None;
    // SAFETY: valid resource, descriptor and out pointer.
    unsafe { dev.CreateShaderResourceView(t, Some(&desc), Some(&mut v)) }
        .map_err(create_err("SRV"))?;
    v.ok_or_else(|| "no SRV".to_owned())
}

fn uav(
    dev: &ID3D11Device,
    t: &ID3D11Texture2D,
    format: DXGI_FORMAT,
    mip: u32,
) -> Result<ID3D11UnorderedAccessView, String> {
    let desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
        Format: format,
        ViewDimension: D3D11_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_UAV { MipSlice: mip },
        },
    };
    let mut v = None;
    // SAFETY: valid resource, descriptor and out pointer.
    unsafe { dev.CreateUnorderedAccessView(t, Some(&desc), Some(&mut v)) }
        .map_err(create_err("UAV"))?;
    v.ok_or_else(|| "no UAV".to_owned())
}

/// A texture read by one pass and written by another.
struct Rw {
    srv: ID3D11ShaderResourceView,
    uav: ID3D11UnorderedAccessView,
}

impl Rw {
    fn new(dev: &ID3D11Device, size: [u32; 2], format: DXGI_FORMAT) -> Result<Self, String> {
        let bind = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0;
        let t = texture(dev, size, format, 1, bind)?;
        Ok(Self {
            srv: srv(dev, &t, format, 0, 1)?,
            uav: uav(dev, &t, format, 0)?,
        })
    }
}

/// Our textures for one (region size, trace resolution, colour format).
struct Targets {
    key: ([u32; 2], bool, i32),
    hiz_all: ID3D11ShaderResourceView,
    hiz_level_srv: Vec<ID3D11ShaderResourceView>,
    hiz_level_uav: Vec<ID3D11UnorderedAccessView>,
    color: ID3D11Texture2D,
    color_srv: ID3D11ShaderResourceView,
    /// Trace grid: (c L, c) and the rays (direction, pdf).
    trace: Rw,
    aux: Rw,
    /// Region: resolve output, then (temporal on) the blurred accumulation; (L, weight).
    resolved: Rw,
    /// 1x1 R32_UINT water-pixel count of the frame.
    count: ID3D11Texture2D,
    count_srv: ID3D11ShaderResourceView,
    count_uav: ID3D11UnorderedAccessView,
    /// Region-sized R8 water mask (debug view) and surface class (resolve).
    mask: Rw,
    class: Rw,
    /// Region RGBA8: the reflectance tint F.rgb / max(F) of the frame (resolve -> composite).
    tint: Rw,
    /// Temporal ping-pong: accumulation (L c, c), meta (normal, history length, roughness),
    /// linear depth, second moment of the luminance (firefly-aware history clamp).
    hist: [Rw; 2],
    meta: [Rw; 2],
    hdepth: [Rw; 2],
    mom: [Rw; 2],
    trace_size: [u32; 2],
}

impl Targets {
    fn new(
        dev: &ID3D11Device,
        region: [u32; 2],
        half: bool,
        color_format: DXGI_FORMAT,
    ) -> Result<Self, String> {
        let typed = color_view_format(color_format)
            .ok_or_else(|| format!("colour format {} not handled", color_format.0))?;
        let rw = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0;
        let hiz = texture(dev, hiz_dims(region), DXGI_FORMAT_R32_FLOAT, HIZ_LEVELS, rw)?;
        let color = texture(dev, region, color_format, 1, D3D11_BIND_SHADER_RESOURCE.0)?;
        let trace = trace_dims(region, half);
        let count = texture(dev, [1, 1], DXGI_FORMAT_R32_UINT, 1, rw)?;
        let f16 = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let pair = |format| -> Result<[Rw; 2], String> {
            Ok([Rw::new(dev, region, format)?, Rw::new(dev, region, format)?])
        };
        log::line(&format!(
            "ssr: targets for {}x{} (trace {}x{}, colour format {})",
            region[0], region[1], trace[0], trace[1], color_format.0
        ));
        Ok(Self {
            key: (region, half, color_format.0),
            hiz_all: srv(dev, &hiz, DXGI_FORMAT_R32_FLOAT, 0, HIZ_LEVELS)?,
            hiz_level_srv: (0..HIZ_LEVELS)
                .map(|m| srv(dev, &hiz, DXGI_FORMAT_R32_FLOAT, m, 1))
                .collect::<Result<_, _>>()?,
            hiz_level_uav: (0..HIZ_LEVELS)
                .map(|m| uav(dev, &hiz, DXGI_FORMAT_R32_FLOAT, m))
                .collect::<Result<_, _>>()?,
            color_srv: srv(dev, &color, typed, 0, 1)?,
            color,
            trace: Rw::new(dev, trace, f16)?,
            aux: Rw::new(dev, trace, f16)?,
            resolved: Rw::new(dev, region, f16)?,
            count_srv: srv(dev, &count, DXGI_FORMAT_R32_UINT, 0, 1)?,
            count_uav: uav(dev, &count, DXGI_FORMAT_R32_UINT, 0)?,
            count,
            mask: Rw::new(dev, region, DXGI_FORMAT_R8_UNORM)?,
            class: Rw::new(dev, region, DXGI_FORMAT_R8_UNORM)?,
            tint: Rw::new(dev, region, DXGI_FORMAT_R8G8B8A8_UNORM)?,
            hist: pair(f16)?,
            meta: pair(DXGI_FORMAT_R8G8B8A8_UNORM)?,
            hdepth: pair(DXGI_FORMAT_R16_FLOAT)?,
            mom: pair(DXGI_FORMAT_R32_FLOAT)?,
            trace_size: trace,
        })
    }
}

/// Inputs of one frame, captured at the TAA site.
pub(crate) struct Job {
    color: ID3D11Resource,
    depth: ID3D11Resource,
    water: Option<ID3D11Resource>,
    /// The patched lit shaders' material target of this frame (`material.rs`).
    material: Option<ID3D11ShaderResourceView>,
    /// Their F0 colour target (`material.rs`, o3).
    f0: Option<ID3D11ShaderResourceView>,
    /// Engine motion vectors of this frame (`motionVectors`, current - previous in viewport
    /// UV); without them the temporal stage keeps no history.
    motion: Option<ID3D11Resource>,
    /// Region (x, y, w, h) in the engine textures.
    region: [u32; 4],
    params: ShaderParams,
    quality: usize,
    half: bool,
    /// `ssr_temporal`.
    temporal: bool,
    /// Capture counter (consecutive frames continue the history).
    frame: u32,
    /// Camera position from the view matrix (world metres; ray-traced reflections).
    cam: [f32; 3],
}

// SAFETY: D3D11 resources are free-threaded; a job moves from the recording worker to the
// submitting worker (FSR path) and is used there only.
unsafe impl Send for Job {}

/// What the debug view shows: the result texture sizes of the last frame.
#[derive(Clone, Copy)]
struct Shown {
    region: [u32; 2],
    trace: [u32; 2],
    /// Temporal slot written (history-length view).
    meta: Option<usize>,
}

/// Views and slots of one recording.
struct Inputs<'a> {
    depth: &'a ID3D11ShaderResourceView,
    water: Option<&'a ID3D11ShaderResourceView>,
    motion: Option<&'a ID3D11ShaderResourceView>,
    /// Temporal slot to write (`None`: no temporal stage).
    slot: Option<usize>,
    /// Hybrid RT classification targets (FSR path, `rt_reflect.rs`).
    rt: Option<&'a rt_reflect::Classify>,
    rtv: &'a ID3D11RenderTargetView,
    stage: &'a ID3D11Texture2D,
}

struct Pipeline {
    device: ID3D11Device,
    hiz0: ID3D11ComputeShader,
    hiz: ID3D11ComputeShader,
    trace: Vec<ID3D11ComputeShader>,
    resolve: ID3D11ComputeShader,
    temporal: ID3D11ComputeShader,
    blur: ID3D11ComputeShader,
    rt_classify: ID3D11ComputeShader,
    vs: ID3D11VertexShader,
    composite: ID3D11PixelShader,
    debug: ID3D11PixelShader,
    params: ID3D11Buffer,
    debug_params: ID3D11Buffer,
    linear: ID3D11SamplerState,
    raster: ID3D11RasterizerState,
    no_depth: ID3D11DepthStencilState,
    over: ID3D11BlendState,
    targets: Option<Targets>,
    views: Views,
    water: HashMap<usize, (ID3D11Resource, ID3D11ShaderResourceView)>,
    shown: Option<Shown>,
    /// Read-back ring of the water-pixel count: staging texture and, while a copy is
    /// pending, the region size it belongs to.
    stage: Vec<(ID3D11Texture2D, Option<[u32; 2]>)>,
    stage_next: usize,
    /// Material target of the last frame and its size (debug view).
    shown_material: Option<(ID3D11ShaderResourceView, [u32; 2])>,
    shown_f0: Option<(ID3D11ShaderResourceView, [u32; 2])>,
    /// Temporal ping-pong slot written last and the capture frame it belongs to (`None`: no
    /// usable history).
    history: Option<(usize, u32)>,
}

// SAFETY: D3D11 devices and their child objects are free-threaded; the pipeline is used
// under `PIPE`'s mutex only.
unsafe impl Send for Pipeline {}

static PIPE: Mutex<Option<Pipeline>> = Mutex::new(None);

impl Pipeline {
    fn new(dev: &ID3D11Device, code: &Bytecode) -> Result<Self, String> {
        // SAFETY (whole block): object creation with valid descriptors, bytecode compiled
        // for the declared stage, and out pointers.
        unsafe {
            let cs = |b: &[u8]| -> Result<ID3D11ComputeShader, String> {
                let mut s = None;
                dev.CreateComputeShader(b, None, Some(&mut s))
                    .map_err(create_err("compute shader"))?;
                s.ok_or_else(|| "no compute shader".to_owned())
            };
            let ps = |b: &[u8]| -> Result<ID3D11PixelShader, String> {
                let mut s = None;
                dev.CreatePixelShader(b, None, Some(&mut s))
                    .map_err(create_err("pixel shader"))?;
                s.ok_or_else(|| "no pixel shader".to_owned())
            };
            let mut vs = None;
            dev.CreateVertexShader(&code.vs, None, Some(&mut vs))
                .map_err(create_err("vertex shader"))?;
            let mut linear = None;
            dev.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                Some(&mut linear),
            )
            .map_err(create_err("sampler"))?;
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
            let mut no_depth = None;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: false.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut no_depth),
            )
            .map_err(create_err("depth-stencil state"))?;
            let mut bd = D3D11_BLEND_DESC::default();
            bd.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D11_BLEND_ONE,
                DestBlend: D3D11_BLEND_INV_SRC1_COLOR,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ZERO,
                DestBlendAlpha: D3D11_BLEND_ONE,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: (D3D11_COLOR_WRITE_ENABLE_RED.0
                    | D3D11_COLOR_WRITE_ENABLE_GREEN.0
                    | D3D11_COLOR_WRITE_ENABLE_BLUE.0) as u8,
            };
            let mut over = None;
            dev.CreateBlendState(&bd, Some(&mut over))
                .map_err(create_err("blend state"))?;
            Ok(Self {
                device: dev.clone(),
                hiz0: cs(&code.hiz0)?,
                hiz: cs(&code.hiz)?,
                trace: code.trace.iter().map(|b| cs(b)).collect::<Result<_, _>>()?,
                resolve: cs(&code.resolve)?,
                temporal: cs(&code.temporal)?,
                blur: cs(&code.blur)?,
                rt_classify: cs(&code.rt_classify)?,
                vs: vs.ok_or("no vertex shader")?,
                composite: ps(&code.composite)?,
                debug: ps(&code.debug)?,
                params: dynamic_cb(dev, size_of::<ShaderParams>())?,
                debug_params: dynamic_cb(dev, 16)?,
                linear: linear.ok_or("no sampler")?,
                raster: raster.ok_or("no rasterizer state")?,
                no_depth: no_depth.ok_or("no depth-stencil state")?,
                over: over.ok_or("no blend state")?,
                targets: None,
                views: Views::default(),
                water: HashMap::new(),
                shown: None,
                stage: (0..3)
                    .map(|_| staging_u32(dev).map(|t| (t, None)))
                    .collect::<Result<_, _>>()?,
                stage_next: 0,
                shown_material: None,
                shown_f0: None,
                history: None,
            })
        }
    }

    fn depth_srv(
        &mut self,
        r: &ID3D11Resource,
        water: bool,
    ) -> Result<ID3D11ShaderResourceView, String> {
        let key = r.as_raw() as usize;
        let cache = if water {
            &mut self.water
        } else {
            &mut self.views.depth
        };
        if let Some((_, v)) = cache.get(&key) {
            return Ok(v.clone());
        }
        let (t, d) = tex2d(r)?;
        let format = depth_view_format(d.Format)
            .ok_or_else(|| format!("depth format {} not handled", d.Format.0))?;
        if d.SampleDesc.Count != 1 {
            return Err(format!(
                "multisampled depth ({} samples)",
                d.SampleDesc.Count
            ));
        }
        let v = srv(&self.device, &t, format, 0, 1)?;
        if cache.len() >= MAX_CACHED_VIEWS {
            cache.clear();
        }
        cache.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    fn color_rtv(&mut self, r: &ID3D11Resource) -> Result<ID3D11RenderTargetView, String> {
        let key = r.as_raw() as usize;
        if let Some((_, v)) = self.views.color.get(&key) {
            return Ok(v.clone());
        }
        let (_, d) = tex2d(r)?;
        let format = color_view_format(d.Format)
            .ok_or_else(|| format!("colour format {} not handled", d.Format.0))?;
        let desc = D3D11_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
            },
        };
        let mut v = None;
        // SAFETY: valid resource, descriptor and out pointer.
        unsafe {
            self.device
                .CreateRenderTargetView(r, Some(&desc), Some(&mut v))
        }
        .map_err(create_err("colour RTV"))?;
        let v = v.ok_or("no colour RTV")?;
        if self.views.color.len() >= MAX_CACHED_VIEWS {
            self.views.color.clear();
        }
        self.views.color.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    /// SRV of the engine motion vectors (same size as the colour texture), cached.
    fn motion_srv(
        &mut self,
        r: &ID3D11Resource,
        size: [u32; 2],
    ) -> Result<ID3D11ShaderResourceView, String> {
        let key = r.as_raw() as usize;
        if let Some((_, v)) = self.views.motion.get(&key) {
            return Ok(v.clone());
        }
        let (t, d) = tex2d(r)?;
        if [d.Width, d.Height] != size {
            return Err(format!(
                "motion vectors {}x{} differ from the colour {}x{}",
                d.Width, d.Height, size[0], size[1]
            ));
        }
        let format = match d.Format {
            DXGI_FORMAT_R16G16_TYPELESS | DXGI_FORMAT_R16G16_FLOAT => DXGI_FORMAT_R16G16_FLOAT,
            DXGI_FORMAT_R32G32_TYPELESS | DXGI_FORMAT_R32G32_FLOAT => DXGI_FORMAT_R32G32_FLOAT,
            f => return Err(format!("motion-vector format {} not handled", f.0)),
        };
        let v = srv(&self.device, &t, format, 0, 1)?;
        if self.views.motion.len() >= MAX_CACHED_VIEWS {
            self.views.motion.clear();
        }
        self.views.motion.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    fn targets(
        &mut self,
        region: [u32; 2],
        half: bool,
        format: DXGI_FORMAT,
    ) -> Result<&Targets, String> {
        let key = (region, half, format.0);
        if self.targets.as_ref().is_none_or(|t| t.key != key) {
            self.targets = None;
            self.targets = Some(Targets::new(&self.device, region, half, format)?);
        }
        self.targets.as_ref().ok_or_else(|| "no targets".to_owned())
    }

    /// `rt`: the FSR path (immediate context before FSR's pack), where the ray-traced
    /// reflections can follow on FSR's D3D12 list.
    fn record(
        &mut self,
        ctx: &ID3D11DeviceContext,
        job: &Job,
        cs_srv: usize,
        rt: bool,
    ) -> Result<(), String> {
        let depth = self.depth_srv(&job.depth, false)?;
        let water = match &job.water {
            // A water target without a usable depth view only disables the mask.
            Some(w) => match self.depth_srv(w, true) {
                Ok(v) => Some(v),
                Err(e) => {
                    if !WATER_VIEW_LOGGED.swap(true, Ordering::Relaxed) {
                        log::line(&format!("ssr: water mask off: {e}"));
                    }
                    None
                }
            },
            None => None,
        };
        let rtv = self.color_rtv(&job.color)?;
        let (_, cd) = tex2d(&job.color)?;
        // Temporal stage only with motion vectors (the rotated pattern without accumulation
        // would flicker).
        let motion = match (&job.motion, job.temporal) {
            (Some(m), true) => match self.motion_srv(m, [cd.Width, cd.Height]) {
                Ok(v) => Some(v),
                Err(e) => {
                    if !MOTION_VIEW_LOGGED.swap(true, Ordering::Relaxed) {
                        log::line(&format!("ssr: temporal accumulation off: {e}"));
                    }
                    None
                }
            },
            _ => None,
        };
        let region = [job.region[2], job.region[3]];
        let key = (region, job.half, cd.Format.0);
        let continued = self.targets.as_ref().is_some_and(|t| t.key == key);
        self.targets(region, job.half, cd.Format)?;
        let mut params = job.params;
        let classify = if rt {
            rt_reflect::classify(&self.device, region)
        } else {
            None
        };
        if let Some(c) = &classify {
            params.rt = c.params;
        }
        if water.is_none() {
            params.trace[3] &= !1;
        }
        if job.material.is_none() {
            params.trace[3] &= !4;
        }
        if job.material.is_none() || job.f0.is_none() {
            params.trace[3] &= !8;
        }
        // Ping-pong slot written this frame; the other one holds the previous frame's history
        // if that frame was the capture before this one with the same targets.
        let slot = motion.as_ref().map(|_| match self.history {
            Some((last, frame)) if continued && frame.wrapping_add(1) == job.frame => {
                (last ^ 1, true)
            }
            Some((last, _)) => (last ^ 1, false),
            None => (0, false),
        });
        let slot = slot.filter(|_| !OFFLINE_ROTATE_ONLY.load(Ordering::Relaxed));
        let linear = if OFFLINE_NO_FIREFLY.load(Ordering::Relaxed)
            || OFFLINE_LEGACY.load(Ordering::Relaxed)
        {
            TEMPORAL_LINEAR
        } else {
            0
        };
        if let Some((_, valid)) = slot {
            params.temporal = [
                job.frame,
                TEMPORAL_ROTATE
                    | TEMPORAL_MOTION
                    | linear
                    | if OFFLINE_LEGACY.load(Ordering::Relaxed) {
                        TEMPORAL_LEGACY
                    } else {
                        TEMPORAL_GATE
                    }
                    | if valid { TEMPORAL_HISTORY } else { 0 },
                0,
                0,
            ];
        } else if OFFLINE_ROTATE_ONLY.load(Ordering::Relaxed) {
            params.temporal = [job.frame, TEMPORAL_ROTATE | linear, 0, 0];
        } else {
            params.temporal[1] |= linear;
        }
        params.temporal[2] = rt_reflect::firefly_k().to_bits();
        if slot.is_some_and(|(_, valid)| !valid) {
            HISTORY_RESETS.fetch_add(1, Ordering::Relaxed);
        }
        let stage = self.stage_next;
        // SAFETY: capture/restore on the caller's context, which this thread records.
        let saved = unsafe { Saved::capture(ctx, touched(cs_srv)) };
        let r = upload(ctx, &self.params, &params).map(|()| {
            let inputs = Inputs {
                depth: &depth,
                water: water.as_ref(),
                motion: motion.as_ref(),
                slot: slot.map(|s| s.0),
                rt: classify.as_ref(),
                rtv: &rtv,
                stage: &self.stage[stage].0,
            };
            // SAFETY: recording on the caller's context with our own live objects.
            unsafe { self.passes(ctx, job, &inputs) }
        });
        // SAFETY: as above.
        unsafe { saved.restore(ctx) };
        r?;
        self.history = slot.map(|(s, _)| (s, job.frame));
        if classify.is_some() {
            rt_reflect::queue(rt_reflect::View {
                cam: job.cam,
                rot: params.rot,
                proj: params.proj,
                depth: [params.depth[0], params.depth[1]],
                region: job.region,
                frame: job.frame,
                temporal: slot.is_some(),
            });
        }
        self.stage[stage].1 = Some(region);
        self.stage_next = (stage + 1) % self.stage.len();
        self.shown_material = job.material.clone().map(|m| (m, [cd.Width, cd.Height]));
        self.shown_f0 = job.f0.clone().map(|m| (m, [cd.Width, cd.Height]));
        self.shown = self.targets.as_ref().map(|t| Shown {
            region,
            trace: t.trace_size,
            meta: slot.map(|s| s.0),
        });
        Ok(())
    }

    /// Immediate context only: reads finished water-count copies (never waits) into the
    /// status; logs once when the mask covered too much and was ignored.
    fn poll_water(&mut self, ctx: &ID3D11DeviceContext) {
        for (tex, pending) in &mut self.stage {
            let Some(region) = *pending else { continue };
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            // SAFETY: our 1x1 staging texture, mapped without waiting and unmapped below.
            let mapped = unsafe {
                ctx.Map(
                    &*tex,
                    0,
                    D3D11_MAP_READ,
                    D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                    Some(&mut m),
                )
            };
            if mapped.is_err() {
                continue;
            }
            // SAFETY: a mapped 1x1 R32_UINT texture holds at least 4 bytes.
            let n = unsafe { m.pData.cast::<u32>().read_unaligned() };
            // SAFETY: mapped above.
            unsafe { ctx.Unmap(&*tex, 0) };
            *pending = None;
            let px = f64::from(region[0]) * f64::from(region[1]);
            let coverage = (f64::from(n) / px.max(1.0)) as f32;
            WATER_COVERAGE.store(coverage.to_bits(), Ordering::Relaxed);
            if n > water_limit(region) {
                WATER_FALLBACK.fetch_add(1, Ordering::Relaxed);
                if !WATER_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                    log::line(&format!(
                        "ssr: water mask covered {:.0} % of the region (> {:.0} %): ignored for such frames",
                        coverage * 100.0,
                        WATER_MAX_FRACTION * 100.0
                    ));
                }
            }
        }
    }

    /// # Safety
    /// `ctx` is recorded by this thread; the views belong to this pipeline's device.
    unsafe fn passes(&self, ctx: &ID3D11DeviceContext, job: &Job, inp: &Inputs) {
        let Some(t) = self.targets.as_ref() else {
            return;
        };
        let [x, y, w, h] = job.region;
        let groups = |n: u32| n.div_ceil(8);
        let keep = [u32::MAX; 5];
        let set_uavs = |v: [Option<&ID3D11UnorderedAccessView>; 5]| {
            let v = v.map(|u| u.cloned());
            // SAFETY: five-element arrays for five slots.
            unsafe { ctx.CSSetUnorderedAccessViews(0, 5, Some(v.as_ptr()), Some(keep.as_ptr())) };
        };
        // t3 and t8-t11 per pass.
        let set_srvs = |input: Option<&ID3D11ShaderResourceView>,
                        rest: [Option<&ID3D11ShaderResourceView>; 4]| {
            // SAFETY: bindings of live views of this device.
            unsafe {
                ctx.CSSetShaderResources(3, Some(&[input.cloned()]));
                ctx.CSSetShaderResources(8, Some(&rest.map(|v| v.cloned())));
            }
        };
        let hiz = hiz_dims([w, h]);
        // SAFETY (whole block): bindings of our own objects and the engine's textures; every
        // slot used here is saved by the caller.
        unsafe {
            ctx.OMSetRenderTargets(None, None);
            // Radiance / fallback: the colour region before we change it.
            ctx.CopySubresourceRegion(
                &t.color,
                0,
                0,
                0,
                0,
                &job.color,
                0,
                Some(&D3D11_BOX {
                    left: x,
                    top: y,
                    front: 0,
                    right: x + w,
                    bottom: y + h,
                    back: 1,
                }),
            );
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetShaderResources(
                0,
                Some(&[
                    Some(inp.depth.clone()),
                    Some(t.color_srv.clone()),
                    None,
                    None,
                    inp.water.cloned(),
                    None,
                    None,
                    job.material.clone(),
                    None,
                    None,
                    None,
                    None,
                    job.f0.clone(),
                ]),
            );
            // Max-depth pyramid, water mask and its pixel count.
            ctx.ClearUnorderedAccessViewUint(&t.count_uav, &[0; 4]);
            ctx.CSSetShader(&self.hiz0, None);
            set_uavs([
                Some(&t.hiz_level_uav[0]),
                None,
                Some(&t.count_uav),
                Some(&t.mask.uav),
                None,
            ]);
            ctx.Dispatch(groups(hiz[0]), groups(hiz[1]), 1);
            set_uavs([None; 5]);
            ctx.CSSetShaderResources(5, Some(&[Some(t.count_srv.clone())]));
            ctx.CSSetShader(&self.hiz, None);
            for m in 1..HIZ_LEVELS as usize {
                set_uavs([None; 5]);
                ctx.CSSetShaderResources(2, Some(&[Some(t.hiz_level_srv[m - 1].clone())]));
                set_uavs([Some(&t.hiz_level_uav[m]), None, None, None, None]);
                ctx.Dispatch(groups(hiz[0] >> m), groups(hiz[1] >> m), 1);
            }
            set_uavs([None; 5]);
            // Trace.
            ctx.CSSetShaderResources(2, Some(&[Some(t.hiz_all.clone())]));
            // Previous history meta (rotation gate): the slot not written this frame.
            let prev_meta = inp.slot.map(|cur| t.meta[cur ^ 1].srv.clone());
            ctx.CSSetShaderResources(8, Some(&[prev_meta]));
            ctx.CSSetShader(&self.trace[job.quality], None);
            set_uavs([None, Some(&t.trace.uav), None, None, Some(&t.aux.uav)]);
            ctx.Dispatch(groups(t.trace_size[0]), groups(t.trace_size[1]), 1);
            ctx.CSSetShaderResources(8, Some(&[None]));
            set_uavs([None; 5]);
            // Resolve (ray reuse), surface classes.
            set_srvs(Some(&t.trace.srv), [Some(&t.aux.srv), None, None, None]);
            ctx.CSSetShader(&self.resolve, None);
            set_uavs([
                None,
                Some(&t.resolved.uav),
                None,
                Some(&t.class.uav),
                Some(&t.tint.uav),
            ]);
            ctx.Dispatch(groups(w), groups(h), 1);
            set_uavs([None; 5]);
            if let Some(cur) = inp.slot {
                let prev = cur ^ 1;
                // Temporal accumulation into the current slot.
                set_srvs(
                    Some(&t.resolved.srv),
                    [
                        Some(&t.meta[prev].srv),
                        Some(&t.hist[prev].srv),
                        Some(&t.hdepth[prev].srv),
                        inp.motion,
                    ],
                );
                // Previous luminance moment at t6 (free during this pass).
                ctx.CSSetShaderResources(6, Some(&[Some(t.mom[prev].srv.clone())]));
                ctx.CSSetShader(&self.temporal, None);
                set_uavs([
                    Some(&t.hdepth[cur].uav),
                    Some(&t.hist[cur].uav),
                    None,
                    Some(&t.mom[cur].uav),
                    Some(&t.meta[cur].uav),
                ]);
                ctx.Dispatch(groups(w), groups(h), 1);
                set_uavs([None; 5]);
                ctx.CSSetShaderResources(6, Some(&[None]));
                // Blur into the composite input.
                set_srvs(
                    Some(&t.hist[cur].srv),
                    [Some(&t.meta[cur].srv), None, Some(&t.hdepth[cur].srv), None],
                );
                ctx.CSSetShader(&self.blur, None);
                set_uavs([None, Some(&t.resolved.uav), None, None, None]);
                ctx.Dispatch(groups(w), groups(h), 1);
                set_uavs([None; 5]);
            }
            if let Some(c) = inp.rt {
                // Hybrid RT classification from the final SSR result (rt_reflect.rs).
                set_srvs(Some(&t.resolved.srv), [None, None, None, None]);
                ctx.CSSetShader(&self.rt_classify, None);
                set_uavs([Some(&c.depth), Some(&c.req0), None, None, Some(&c.req1)]);
                let u5 = [Some(c.rtw.clone())];
                ctx.CSSetUnorderedAccessViews(5, 1, Some(u5.as_ptr()), Some([u32::MAX].as_ptr()));
                ctx.Dispatch(groups(w.div_ceil(2)), groups(h.div_ceil(2)), 1);
                set_uavs([None; 5]);
                let none5: [Option<ID3D11UnorderedAccessView>; 1] = [None];
                ctx.CSSetUnorderedAccessViews(
                    5,
                    1,
                    Some(none5.as_ptr()),
                    Some([u32::MAX].as_ptr()),
                );
            }
            let none: [Option<ID3D11ShaderResourceView>; CS_SRVS] = Default::default();
            ctx.CSSetShaderResources(0, Some(&none));
            // Blend into the colour inside the region.
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.composite, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.PSSetShaderResources(3, Some(&[Some(t.resolved.srv.clone())]));
            ctx.PSSetShaderResources(8, Some(&[Some(t.tint.srv.clone())]));
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: x as f32,
                TopLeftY: y as f32,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(&self.over, None, D3D11_DEFAULT_SAMPLE_MASK);
            ctx.OMSetDepthStencilState(&self.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(inp.rtv.clone())]), None);
            ctx.Draw(3, 0);
            // Water-pixel count for the read-back ring (last, so nothing waits on it).
            ctx.CopyResource(inp.stage, &t.count);
        }
    }

    fn draw_debug(
        &self,
        ctx: &ID3D11DeviceContext,
        rtv: &ID3D11RenderTargetView,
        size: [u32; 2],
        view: u8,
    ) -> Result<(), String> {
        let (Some(t), Some(shown)) = (self.targets.as_ref(), self.shown) else {
            return Ok(());
        };
        let shown_material = self.shown_material.clone();
        let (srv, valid, tex) = match (view, shown_material.as_ref()) {
            (1, _) => (&t.trace.srv, shown.trace, t.trace_size),
            (4, _) => (&t.mask.srv, shown.region, t.key.0),
            (5, Some((m, size))) => (m, shown.region, *size),
            (5, None) => return Ok(()),
            (6, _) => match shown.meta {
                Some(s) => (&t.meta[s].srv, shown.region, t.key.0),
                None => return Ok(()),
            },
            (7, _) => (&t.class.srv, shown.region, t.key.0),
            (8, _) => match self.shown_f0.as_ref() {
                Some((m, size)) => (m, shown.region, *size),
                None => return Ok(()),
            },
            _ => (&t.resolved.srv, shown.region, t.key.0),
        };
        let p = [
            valid[0] as f32 / tex[0] as f32 / size[0] as f32,
            valid[1] as f32 / tex[1] as f32 / size[1] as f32,
            f32::from(view),
            water_limit(shown.region) as f32,
        ];
        upload(ctx, &self.debug_params, &p)?;
        // SAFETY (whole block): immediate context under the engine lock (caller); state
        // captured and restored around our draw.
        unsafe {
            let saved = Saved::capture(ctx, touched(8));
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.debug, None);
            ctx.PSSetConstantBuffers(2, Some(&[Some(self.debug_params.clone())]));
            ctx.PSSetSamplers(1, Some(&[Some(self.linear.clone())]));
            let (input, mask) = if view == 4 || view == 7 {
                (None, Some(srv.clone()))
            } else {
                (Some(srv.clone()), None)
            };
            ctx.PSSetShaderResources(3, Some(&[input, None, Some(t.count_srv.clone()), mask]));
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size[0] as f32,
                Height: size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(None, None, D3D11_DEFAULT_SAMPLE_MASK);
            ctx.OMSetDepthStencilState(&self.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.Draw(3, 0);
            saved.restore(ctx);
        }
        Ok(())
    }
}

/// Main thread before the engine present, immediate context under the engine lock
/// (`overlay.rs`, `ssr_debug` != off): draws the last trace or resolve result full-screen.
pub fn draw_debug(
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    size: [u32; 2],
) -> Result<(), String> {
    let view = settings::get().ssr_debug;
    if view == 0 {
        return Ok(());
    }
    let Ok(mut g) = PIPE.try_lock() else {
        return Ok(());
    };
    let Some(p) = g.as_mut() else {
        return Ok(());
    };
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    if dev.as_raw() != p.device.as_raw() {
        return Ok(());
    }
    p.poll_water(ctx);
    p.draw_debug(ctx, rtv, size, view)
}

fn staging_u32(dev: &ID3D11Device) -> Result<ID3D11Texture2D, String> {
    let d = D3D11_TEXTURE2D_DESC {
        Width: 1,
        Height: 1,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R32_UINT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        ..Default::default()
    };
    let mut t = None;
    // SAFETY: valid descriptor and out pointer.
    unsafe { dev.CreateTexture2D(&d, None, Some(&mut t)) }.map_err(create_err("staging"))?;
    t.ok_or_else(|| "no staging texture".to_owned())
}

// ------------------------------------------------------------------------ engine read --

/// Highest bound CS shader-resource slot + 1 on `ctx` (0..128).
fn bound_cs_srvs(ctx: &ID3D11DeviceContext) -> usize {
    let mut all: [Option<ID3D11ShaderResourceView>;
        D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT as usize] = std::array::from_fn(|_| None);
    // SAFETY: getter with an output array of the declared length.
    unsafe { ctx.CSGetShaderResources(0, Some(&mut all)) };
    all.iter().rposition(Option::is_some).map_or(0, |i| i + 1)
}

/// Inputs of the TAA site for `pass` (`ssr` on and shaders ready), else `None`; `motion` =
/// the engine's motion vectors of the frame (`fsr::motion_vectors`). Errors are logged and
/// counted.
pub(crate) fn capture(pass: u64, motion: Option<ID3D11Resource>) -> Option<Job> {
    let s = settings::get();
    if !s.ssr {
        return None;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    if !BYTECODE_READY.load(Ordering::Acquire) {
        NOT_READY.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    match capture_inner(pass, motion, &s) {
        Ok(j) => Some(j),
        Err(e) => {
            fail(e);
            None
        }
    }
}

fn capture_inner(pass: u64, motion: Option<ID3D11Resource>, s: &Settings) -> Result<Job, String> {
    bytecode().map_err(|e| format!("shader compilation failed: {e}"))?;
    let view = mem::u64_at(pass + PASS_VIEW)
        .filter(|&v| v != 0)
        .ok_or("pass view unreadable")?;
    let color = crate::fsr_scale::pass_image(pass, HASH_COLOR).ok_or("no hdrBuffer in the pass")?;
    let depth =
        crate::fsr_scale::pass_image(pass, HASH_DEPTH).ok_or("no decompressedDepth in the pass")?;
    let proj_mem = mem::f32s::<16>(view + VIEW_PROJECTION).ok_or("projection unreadable")?;
    let view_mem = mem::f32s::<16>(view + VIEW_VIEW).ok_or("view matrix unreadable")?;
    let mut rect = [0i32; 4];
    for (i, r) in rect.iter_mut().enumerate() {
        *r = mem::u32_at(view + VIEW_VIEWPORT + 4 * i as u64).ok_or("viewport unreadable")? as i32;
    }
    let proj = ao::projection(&proj_mem, 1)?;
    let up = dir_in_view(&view_mem, [0.0, 0.0, 1.0]).ok_or("view matrix degenerate")?;
    let rot = view_rotation(&view_mem);
    // Camera position: -R^T t of the row-major view matrix (column vectors).
    let cam: [f32; 3] = std::array::from_fn(|j| {
        -(view_mem[j] * view_mem[3]
            + view_mem[4 + j] * view_mem[7]
            + view_mem[8 + j] * view_mem[11])
    });
    let (_, dd) = tex2d(&depth)?;
    let (_, cd) = tex2d(&color)?;
    let tex = [dd.Width, dd.Height];
    if [cd.Width, cd.Height] != tex {
        return Err(format!(
            "colour {}x{} differs from depth {}x{}",
            cd.Width, cd.Height, tex[0], tex[1]
        ));
    }
    let region = region(rect, tex).ok_or_else(|| format!("viewport {rect:?} outside {tex:?}"))?;
    // Material target of this frame (patched lit shaders), same texture size as the colour.
    // SAFETY: plain getter.
    let dev = unsafe { color.GetDevice() }.ok();
    let material = dev.as_ref().and_then(|d| material::for_frame(d, tex));
    let f0 = dev.as_ref().and_then(|d| material::f0_for_frame(d, tex));
    if region[2] < 16 || region[3] < 16 {
        return Err(format!("region {region:?} too small"));
    }
    // Water mask: the water depth target behind the stencil view, if it matches.
    let water = if s.ssr_skip_water {
        crate::fsr_scale::pass_image(pass, HASH_WATER).filter(|w| {
            tex2d(w).is_ok_and(|(_, wd)| {
                [wd.Width, wd.Height] == tex
                    && depth_view_format(wd.Format).is_some()
                    && wd.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 != 0
            })
        })
    } else {
        None
    };
    if !FIRST_LOGGED.swap(true, Ordering::Relaxed) {
        let wd = crate::fsr_scale::pass_image(pass, HASH_WATER)
            .and_then(|w| tex2d(&w).ok())
            .map(|(_, d)| format!("{}x{} format {}", d.Width, d.Height, d.Format.0));
        let info = format!(
            "first TAA site: colour {}x{} format {}, depth format {}, viewport {rect:?} -> region {region:?}, world up in view {up:?}, projection P00 {} P11 {} jitter {:?}, water depth {}",
            cd.Width,
            cd.Height,
            cd.Format.0,
            dd.Format.0,
            proj.p00,
            proj.p11,
            proj.jitter,
            wd.unwrap_or_else(|| "absent".into()),
        );
        log::line(&format!("ssr: {info}"));
        *lock(&INFO) = Some(info);
    }
    Ok(Job {
        params: {
            let mut k = shader_params(&proj, region, up, rot, water.is_some(), s);
            if material.is_some() {
                k.trace[3] |= 4;
                if f0.is_some() {
                    k.trace[3] |= 8;
                }
            }
            k
        },
        color,
        depth,
        water,
        material,
        f0,
        motion,
        region,
        quality: usize::from(s.ssr_quality).min(QUALITIES.len() - 1),
        half: s.ssr_half_res,
        temporal: s.ssr_temporal,
        frame: FRAME.fetch_add(1, Ordering::Relaxed).wrapping_add(1),
        cam,
    })
}

/// Runs `job` on `ctx` (deferred engine context or the immediate context; on the immediate
/// context finished water-count read-backs are also collected).
fn run(ctx: &ID3D11DeviceContext, job: &Job, cs_srv: usize, immediate: bool) -> Result<(), String> {
    let code = bytecode()?;
    let mut g = lock(&PIPE);
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    if g.as_ref().is_none_or(|p| p.device.as_raw() != dev.as_raw()) {
        *g = None;
        *g = Some(Pipeline::new(&dev, code)?);
    }
    let p = g.as_mut().ok_or("no pipeline")?;
    p.record(ctx, job, cs_srv, immediate)?;
    if job.material.is_some() {
        material::copy_shared(ctx);
    }
    if immediate {
        p.poll_water(ctx);
    }
    Ok(())
}

/// Engine TAA: records `job` into the TAA pass's deferred context (`engine_ctx` +0) before
/// the TAA dispatch, restoring every state the TAA program had bound.
pub(crate) fn record_engine(engine_ctx: u64, job: Job) {
    let r = (|| {
        let d3d = mem::u64_at(engine_ctx)
            .filter(|&d| d != 0)
            .ok_or("no D3D11 context")? as *mut c_void;
        // SAFETY: the engine context's live ID3D11DeviceContext (deferred while recording).
        let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&d3d) }
            .ok_or("null D3D11 context")?
            .clone();
        let bound = bound_cs_srvs(&ctx);
        let slots = CS_SLOTS.fetch_max(bound, Ordering::Relaxed).max(bound);
        run(&ctx, &job, slots, false)
    })();
    match r {
        Ok(()) => {
            ENGINE.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => fail(e),
    }
}

/// FSR: runs `job` on the immediate context before FSR packs its inputs (submit hook, or
/// inline at the TAA site without deferred contexts).
pub(crate) fn execute(ctx: &ID3D11DeviceContext, job: Job) {
    match run(ctx, &job, 8, true) {
        Ok(()) => {
            SUBMIT.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => fail(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn up_vector_and_regions() {
        // Camera looking along world +x (id convention): rows right, up, -forward.
        let view = [
            0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let up = dir_in_view(&view, [0.0, 0.0, 1.0]).unwrap();
        assert!((up[1] - 1.0).abs() < 1e-6 && up[0].abs() < 1e-6 && up[2].abs() < 1e-6);
        let fwd = dir_in_view(&view, [1.0, 0.0, 0.0]).unwrap();
        assert!((fwd[2] - 1.0).abs() < 1e-6, "{fwd:?}");
        assert_eq!(
            region([0, 0, 2303, 967], [3440, 1440]),
            Some([0, 0, 2304, 968])
        );
        assert_eq!(region([-1, 0, 10, 10], [3440, 1440]), None);
    }

    #[test]
    fn sizes_and_params() {
        assert_eq!(hiz_dims([3440, 1440]), [3456, 1472]);
        assert_eq!(hiz_dims([2304, 968]), [2304, 1024]);
        assert_eq!(trace_dims([2305, 967], true), [1153, 484]);
        assert_eq!(trace_dims([2305, 967], false), [2305, 967]);
        let col = ao::engine_projection(2.4, 1.0, 0.01, 7000.0, [0.0; 2]);
        let p = ao::projection(&col, 1).unwrap();
        let mut s = Settings {
            ssr_roughness: 0.5,
            ..Settings::default()
        };
        let rot = view_rotation(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(rot[2], [-0.0, -0.0, -1.0, 0.0]);
        let k = shader_params(&p, [0, 0, 2304, 968], [0.0, 1.0, 0.0], rot, true, &s);
        assert_eq!(k.ints, [2304, 968, 0, 0]);
        assert_eq!(k.trace, [1152, 484, 1, 1]);
        assert_eq!(k.hiz[..3], [2304, 1024, HIZ_LEVELS]);
        assert!((k.surface[0] - 0.25).abs() < 1e-6);
        assert_eq!(k.up[3], FLOOR_COS);
        s.ssr_depth_floors_only = false;
        s.ssr_half_res = false;
        let k = shader_params(&p, [0, 0, 2304, 968], [0.0, 1.0, 0.0], rot, false, &s);
        assert!(k.up[3] < -1.0);
        assert_eq!(k.trace, [2304, 968, 0, 0]);
        assert_eq!(size_of::<ShaderParams>() % 16, 0);
        for (i, (n, _)) in QUALITIES.iter().enumerate() {
            assert_eq!(parse_quality(n), Some(i as u8));
            assert_eq!(quality_name(i as u8), *n);
        }
        for (i, n) in DEBUG_VIEWS.iter().enumerate() {
            assert_eq!(parse_debug(n), Some(i as u8));
            assert_eq!(debug_name(i as u8), *n);
        }
        assert_eq!(
            color_view_format(DXGI_FORMAT_R16G16B16A16_TYPELESS),
            Some(DXGI_FORMAT_R16G16B16A16_FLOAT)
        );
        assert_eq!(color_view_format(DXGI_FORMAT_R8G8B8A8_UNORM), None);
    }
}
