//! Sun contact shadows (docs/lighting-options.md B2, fidelityfx-effects.md Q2, STATUS E066).
//!
//! Splice (O, IDB of the analysed build): `ARK_RENDERPASS_SCREENSHADOWS` callback
//! `sub_14032C7F0(pass)` resolves the sun's cascaded shadow map into screen space: it binds
//! the view's scaled viewport (`sub_1402BC610`, view+30816), renders program
//! `qword_1422B37D0+160` (`screenshadows.decl`, `ComputeParallelShadow` per pixel) into the
//! image of write hash 2006969548 (the same `_SSAO_ScreenShadows` target HBAO+/our AO wrote
//! `.x` into in the `SSAO` pass; R8G8B8A8_UNORM, E064) with the depth of read hash
//! 555227252 (`screenShadows/DepthBuffer`), then blurs `.y` with
//! `screenShadows/bilateralBlur/blurX|blurY` (`qword_1422AE200`, through a pooled
//! temporary). The forward pass (`sub_140311AA0`) binds the same image as
//! `texScreenshadows` and `lighting.decl` uses `.y` as `light.m_shadow` of the parallel
//! light only (vegetation, `noScreenShadows`, evaluates its own CSM), `.x` as AO for the
//! indirect term. With `view+14880 != 0` the callback skips the CSM and fills a constant
//! (mode 1: 1.0, i.e. no sun shadows).
//!
//! Hook: the callback is detoured (MinHook, 16 prologue bytes checked). After the original
//! has recorded (CSM + blur), and only in CSM mode, we record into the pass's deferred
//! context (engine context `pass+0x108`, `ID3D11DeviceContext*` at its +0): a screen-space
//! ray march from every pixel toward the sun against the hardware depth, a depth-aware blur
//! and a draw into the target's green channel with blend op MIN, i.e.
//! `.y = min(engine CSM visibility, contact visibility)`. Every D3D11 state we touch is put
//! back (`ao::Saved`), so the engine's state cache stays valid.
//!
//! Inputs from the view (`pass+0x70`): projection view+29584 (`idRenderMatrix`, row-major,
//! column vectors; `ao::projection` layout 1), view matrix view+29776 (same convention, the
//! in-game FSR sky-vector decision `mul(M, x)` with `row_major`, E057), sun direction
//! view+48344 (`m_sunDirection`, world space, toward the sun: `SetPerViewShaderParms` writes
//! it as `atm/sky/SunDirection`, the forward pass as `parallelLightDir` = `m_wsToLight`),
//! viewport view+30816 (`idScreenRect` x1, y1, x2, y2 inclusive; the render-scaled region in
//! FSR upscale mode). Only that region is read and written.
//!
//! Algorithm: per pixel, view position from depth; the ray `P -> P + L * length` is
//! projected to the screen and stepped with perspective-correct depth (1/z interpolated);
//! each sample snaps to its pixel centre and compares the ray depth with the scene depth: a
//! hit is `bias < rayZ - sceneZ < thickness` (bias = receiver slope per pixel + relative
//! term + 2 ulp of R24 depth), weighted down toward the ray end. The start offset uses a
//! fixed 4x4 pattern (nothing changes per frame), removed by the same one-pass depth-aware
//! 5x5 blur as the AO (weights cancel any 4-periodic pattern), so no temporal filter is
//! needed.

pub(crate) mod offline;

pub use offline::{ContactReport, offline_check};

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use hex_literal::hex;
use windows::Win32::Graphics::Direct3D::{
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D11_SRV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, s};

use crate::ao::{
    self, Channel, ExternalSource, Proj, Saved, Slots, create_err, dynamic_cb, upload,
};
use crate::settings::{self, Settings};
use crate::{hlsl, log, mem, notify};

// ---------------------------------------------------------------------- engine sites --

/// `sub_14032C7F0`, the `ARK_RENDERPASS_SCREENSHADOWS` callback.
const SCREENSHADOWS_RVA: u64 = 0x32_C7F0;
const SCREENSHADOWS_HEAD: [u8; 16] = hex!("48 8B C4 55 41 54 41 55 41 56 41 57 48 8D A8 48");
pub(crate) const PASS_VIEW: u64 = 0x70;
/// Engine context (`unknown_libname_22`: `pass+264`); `ID3D11DeviceContext*` at +0.
pub(crate) const PASS_ENGINE_CTX: u64 = 0x108;
pub(crate) const PASS_READS: u64 = 0x48;
pub(crate) const PASS_WRITES: u64 = 0x80;
/// Screen-shadow mode: 0 = CSM resolve, otherwise a constant fill (1: lit).
pub(crate) const VIEW_SHADOW_MODE: u64 = 14_880;
pub(crate) const VIEW_PROJECTION: u64 = 29_584;
pub(crate) const VIEW_VIEW: u64 = 29_776;
pub(crate) const VIEW_VIEWPORT: u64 = 30_816;
pub(crate) const VIEW_SUN: u64 = 48_344;
/// Read hash of the hardware depth (`screenShadows/DepthBuffer`, HBAO+'s depth input).
pub(crate) const HASH_DEPTH: u32 = 555_227_252;
/// Write hash of `_SSAO_ScreenShadows` (`texScreenshadows`).
pub(crate) const HASH_SCREEN_SHADOWS: u32 = 2_006_969_548;

type PassFn = unsafe extern "system" fn(u64) -> u64;

static ORIG: AtomicUsize = AtomicUsize::new(0);
static INSTALL_TRIED: AtomicBool = AtomicBool::new(false);
static HOOKED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
/// Every SCREENSHADOWS callback (photo mode's paused-frame counters).
static PASSES: AtomicU64 = AtomicU64::new(0);
static OURS: AtomicU64 = AtomicU64::new(0);
/// Passes without a CSM resolve (constant fill mode) or without a sun.
static SKIPPED: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
static FIRST_LOGGED: AtomicBool = AtomicBool::new(false);
static FAIL_NOTIFIED: AtomicBool = AtomicBool::new(false);
/// Last `contact_shadows` seen by `tick` (0 = not yet, 1 = off, 2 = on).
static LAST_MODE: AtomicU8 = AtomicU8::new(0);
static ERROR: Mutex<Option<String>> = Mutex::new(None);
static INFO: Mutex<Option<String>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// State for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub hooked: bool,
    pub calls: u64,
    /// Passes with our contact shadows recorded.
    pub ours: u64,
    /// Passes without the CSM resolve (no sun shadows) or without a sun direction.
    pub skipped: u64,
    pub failures: u64,
    /// First pass: inputs, formats, sizes, sun direction.
    pub info: Option<String>,
    pub error: Option<String>,
}

pub fn status() -> Status {
    Status {
        hooked: HOOKED.load(Ordering::Relaxed),
        calls: CALLS.load(Ordering::Relaxed),
        ours: OURS.load(Ordering::Relaxed),
        skipped: SKIPPED.load(Ordering::Relaxed),
        failures: FAILURES.load(Ordering::Relaxed),
        info: lock(&INFO).clone(),
        error: lock(&ERROR).clone(),
    }
}

/// SCREENSHADOWS callbacks so far (one per rendered frame while the hook is installed).
pub fn passes() -> u64 {
    PASSES.load(Ordering::Acquire)
}

/// Whether the SCREENSHADOWS hook is installed.
pub fn hooked() -> bool {
    HOOKED.load(Ordering::Relaxed)
}

/// Registers (Some) or removes (None) an external `.y` term (S1 entry point,
/// `ao/external.rs`): drawn after the engine's CSM resolve + blur and our contact shadows,
/// only in CSM mode (`view+14880 == 0`). Needs the hook ([`install`]).
pub fn set_external_y(src: Option<ExternalSource>) {
    ao::external::set(Channel::Y, src);
}

// --------------------------------------------------------------------------- settings --

/// `contact_quality` presets: name, ray steps.
pub const QUALITIES: [(&str, u32); 4] = [("low", 8), ("medium", 12), ("high", 16), ("ultra", 24)];
/// Index of `contact_quality = high`.
pub const DEFAULT_QUALITY: u8 = 2;

pub fn quality_name(q: u8) -> &'static str {
    QUALITIES[usize::from(q).min(QUALITIES.len() - 1)].0
}

pub fn parse_quality(v: &str) -> Option<u8> {
    QUALITIES.iter().position(|(n, _)| *n == v).map(|i| i as u8)
}

/// Per frame (main thread, `hook::observe`): toast when `contact_shadows` changes; turning
/// it on installs the hook if that was not tried yet.
pub fn tick() {
    let on = settings::get().contact_shadows;
    let mode = if on { 2 } else { 1 };
    let last = LAST_MODE.swap(mode, Ordering::Relaxed);
    if last == mode {
        return;
    }
    if on && !INSTALL_TRIED.load(Ordering::Relaxed) {
        match install() {
            Ok(()) => log::line("contact: installed at run time (contact_shadows = 1)"),
            Err(e) => {
                log::line(&format!("contact: installation failed: {e}"));
                notify::error(format!("Contact shadows unavailable: {e}"));
            }
        }
    }
    if last != 0 {
        log::line(&format!("contact: contact_shadows = {}", u8::from(on)));
        notify::info(if on {
            "Sun contact shadows on"
        } else {
            "Sun contact shadows off"
        });
    }
}

// ----------------------------------------------------------------------- installation --

/// Detours the SCREENSHADOWS callback (at startup with `contact_shadows = 1`, or when it is
/// turned on). Shaders compile on a background thread; passes before that stay untouched.
pub fn install() -> Result<(), String> {
    if INSTALL_TRIED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    std::thread::spawn(|| {
        if let Err(e) = bytecode() {
            log::line(&format!("contact: shader compilation failed: {e}"));
        }
    });
    let r = install_hook();
    if let Err(e) = &r {
        *lock(&ERROR) = Some(e.clone());
    }
    r
}

fn install_hook() -> Result<(), String> {
    let target = mem::image_base().ok_or("no image base")? + SCREENSHADOWS_RVA;
    let mut head = [0u8; 16];
    if !mem::read_into(target, &mut head) || head != SCREENSHADOWS_HEAD {
        return Err(format!(
            "SCREENSHADOWS callback at {target:#x}: bytes {head:02x?} differ from the analysed build"
        ));
    }
    // SAFETY: verified prologue of `sub_14032C7F0(pass)`; the detour has its signature
    // (one pointer argument, integer result) and calls the original unchanged first.
    let tramp = unsafe {
        minhook::MinHook::create_hook(target as *mut c_void, screen_shadows_detour as *mut c_void)
    }
    .map_err(|e| format!("create_hook SCREENSHADOWS: {e:?}"))?;
    ORIG.store(tramp as usize, Ordering::Release);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook SCREENSHADOWS: {e:?}"))?;
    HOOKED.store(true, Ordering::Relaxed);
    log::line(&format!(
        "contact: SCREENSHADOWS callback hooked at {target:#x}; contact_shadows decides per pass"
    ));
    Ok(())
}

unsafe extern "system" fn screen_shadows_detour(pass: u64) -> u64 {
    PASSES.fetch_add(1, Ordering::Relaxed);
    // SAFETY: MinHook trampoline of the callback, argument unchanged.
    let f = unsafe { std::mem::transmute::<usize, PassFn>(ORIG.load(Ordering::Acquire)) };
    // SAFETY: as above.
    let r = unsafe { f(pass) };
    if settings::get().contact_shadows || ao::external::registered(Channel::Y) {
        let _ = std::panic::catch_unwind(|| after_pass(pass));
    }
    r
}

/// After the engine recorded the SCREENSHADOWS pass `pass`: records our contact shadows
/// (`contact_shadows`) and the registered external `.y` term (also the offline check's entry
/// point, with stand-in engine structures).
pub(crate) fn after_pass(pass: u64) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    match record_pass(pass) {
        Ok(true) => {
            OURS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(false) => {}
        Err(e) => {
            let n = FAILURES.fetch_add(1, Ordering::Relaxed);
            if n < 5 || n.is_power_of_two() {
                log::line(&format!("contact: {e} (failure {})", n + 1));
            }
            if !FAIL_NOTIFIED.swap(true, Ordering::Relaxed) {
                notify::warn(format!("Contact shadows not applied: {e}"));
            }
            *lock(&ERROR) = Some(e);
        }
    }
}

// ----------------------------------------------------------------------- pure helpers --

/// World direction `d` into the screen-consistent view frame (x right, y up, z = positive
/// view depth): `view` is the engine's row-major view matrix applied to column vectors
/// (`v_view = V * v_world`, right-handed, camera looking down -z).
pub fn sun_in_view(view: &[f32; 16], d: [f32; 3]) -> Option<[f32; 3]> {
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

/// Relative depth deviation at which blur weights reach 0: 1 / this (as the AO blur).
const BLUR_SHARPNESS: f32 = 16.0;
/// Hits weigh less over the last part of the ray (fraction where the fade starts).
const FADE_START: f32 = 0.6;
/// Relative depth bias (fraction of the view depth).
const RELATIVE_BIAS: f32 = 0.002;

/// Constant buffer `void_cs_params` (lanes documented in the HLSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ShaderParams {
    view: [f32; 4],
    proj: [f32; 4],
    region: [f32; 4],
    ints: [u32; 4],
    depth: [f32; 4],
    sun: [f32; 4],
    march: [f32; 4],
}

fn shader_params(proj: &Proj, region: [u32; 4], sun: [f32; 3], s: &Settings) -> ShaderParams {
    let [x, y, w, h] = region;
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
        // dz/dd = z^2 / mul: two R24 steps of depth at view depth z = z^2 * this.
        depth: [
            proj.depth_mul,
            proj.depth_add,
            65_504.0,
            2.0 / (proj.depth_mul.abs().max(1e-12) * 16_777_215.0),
        ],
        sun: [sun[0], sun[1], sun[2], s.contact_length.max(0.01)],
        march: [
            s.contact_thickness.max(0.001),
            RELATIVE_BIAS,
            FADE_START,
            BLUR_SHARPNESS,
        ],
    }
}

// ------------------------------------------------------------------------------- HLSL --

const CONTACT_HLSL: &str = r"
// void-inject sun contact shadows (contact.rs).
#ifndef CS_STEPS
#define CS_STEPS 16
#endif
cbuffer void_cs_params : register(b0)
{
    float4 g_view;   // view.xy = z * (fp * g_region.zw * g_view.xy + g_view.zw), fp in region pixels
    float4 g_proj;   // P00, P11, jitter x, y (ndc = P * view.xy / z + jitter)
    float4 g_region; // region size w, h, 1/w, 1/h
    uint4  g_ints;   // w, h, region origin in the engine textures
    float4 g_depth;  // view depth = x / (d - y); far clamp; R24 quantisation factor (* z^2)
    float4 g_sun;    // view-frame direction to the sun, ray length (view units)
    float4 g_march;  // thickness, relative bias, fade start, blur sharpness
};
cbuffer void_cs_debug : register(b2) { float4 g_debug; }; // uv per pixel (x, y)

Texture2D<float>   void_cs_depth  : register(t0);
Texture2D<float>   void_cs_input  : register(t2);
RWTexture2D<float> void_cs_out    : register(u0);
SamplerState       void_cs_linear : register(s1);

// Fixed 4x4 pattern (Bayer) for the ray start offset; the blur removes it.
static const uint NOISE[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

float LinZ(int2 p)
{
    float d = void_cs_depth.Load(int3(p + int2(g_ints.zw), 0));
    float z = g_depth.x / (d - g_depth.y);
    return (z > 0.0 && z < g_depth.z) ? z : g_depth.z;
}

float3 ViewPos(float2 fp, float z)
{
    return float3(z * (fp * g_region.zw * g_view.xy + g_view.zw), z);
}

float2 ToPixel(float3 v)
{
    float2 ndc = g_proj.xy * v.xy / v.z + g_proj.zw;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * g_region.xy;
}

[numthreads(8, 8, 1)]
void cs_contact(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= g_ints.xy)) return;
    int2 p = int2(id.xy);
    int2 last = int2(g_ints.xy) - 1;
    float z = LinZ(p);
    if (z >= g_depth.z * 0.999)
    {
        void_cs_out[id.xy] = 1.0;
        return;
    }
    float2 s0 = p + 0.5;
    float3 P = ViewPos(s0, z);
    float3 E = P + g_sun.xyz * g_sun.w;
    // Rays toward the camera stop at half the start depth (stays in front of the near plane).
    float nearZ = 0.5 * P.z;
    if (E.z < nearZ) E = lerp(P, E, (P.z - nearZ) / (P.z - E.z));
    float2 d = ToPixel(E) - s0;
    float len2 = dot(d, d);
    if (len2 < 1.0)
    {
        void_cs_out[id.xy] = 1.0;
        return;
    }
    float w0 = 1.0 / P.z, w1 = 1.0 / E.z;
    // Receiver slope: the smaller one-sided depth step per pixel on each axis (no step
    // across a silhouette), the depth error of snapping a sample to a pixel centre. The same
    // neighbours give the depth-reconstructed normal.
    int2 px = min(p + int2(1, 0), last), mx = max(p - int2(1, 0), 0);
    int2 py = min(p + int2(0, 1), last), my = max(p - int2(0, 1), 0);
    float zpx = LinZ(px), zmx = LinZ(mx), zpy = LinZ(py), zmy = LinZ(my);
    float slope = max(min(abs(zpx - z), abs(zmx - z)), min(abs(zpy - z), abs(zmy - z)));
    float3 dx = abs(zmx - z) <= abs(zpx - z) ? P - ViewPos(mx + 0.5, zmx) : ViewPos(px + 0.5, zpx) - P;
    float3 dy = abs(zmy - z) <= abs(zpy - z) ? P - ViewPos(my + 0.5, zmy) : ViewPos(py + 0.5, zpy) - P;
    float3 N = cross(dx, dy);
    if (dot(N, P) > 0.0) N = -N;
    // Faces turned away from the sun are dark through N.L and the shadow map already; rays
    // from them would only enter their own surface.
    if (dot(N, g_sun.xyz) <= 0.02 * length(N))
    {
        void_cs_out[id.xy] = 1.0;
        return;
    }
    float jitter = (NOISE[(id.x & 3u) + 4u * (id.y & 3u)] + 0.5) / 16.0;
    float occ = 0.0;
    [unroll] for (float i = 0.0; i < CS_STEPS; i += 1.0)
    {
        float2 s = s0 + d * ((i + jitter) / CS_STEPS);
        if (any(s < 0.0) || any(s >= g_region.xy)) break;
        int2 q = int2(s);
        float2 c = q + 0.5;
        float2 rel = c - s0;
        if (dot(rel, rel) < 1.0) continue;
        // Ray depth at the snapped pixel centre (perspective-correct).
        float f = saturate(dot(rel, d) / len2);
        float rayZ = 1.0 / lerp(w0, w1, f);
        float dz = rayZ - LinZ(q);
        float bias = slope + g_march.y * rayZ + g_depth.w * rayZ * rayZ;
        if (dz > bias && dz < g_march.x)
            occ = max(occ, 1.0 - smoothstep(g_march.z, 1.0, f));
    }
    void_cs_out[id.xy] = 1.0 - occ;
}

// Depth-aware blur, both axes in one pass through groupshared memory (12x12 tile with a
// 2-pixel apron per 8x8 group); taps -2..2 weighted 0.5 1 1 1 0.5 cancel the 4x4 pattern.
groupshared float g_tileV[12][12];
groupshared float g_tileZ[12][12];
groupshared float g_rowV[12][8];

float BlurTaps(float a[5], float z[5])
{
    float zc = z[2];
    float g = abs(z[1] - zc) < abs(z[3] - zc) ? zc - z[1] : z[3] - zc;
    float acc = 0.0, wsum = 0.0;
    [unroll] for (int i = 0; i < 5; i++)
    {
        float k = (i == 0 || i == 4) ? 0.5 : 1.0;
        float w = k * saturate(1.0 - abs(z[i] - (zc + (i - 2) * g)) / zc * g_march.w);
        acc += w * a[i];
        wsum += w;
    }
    return acc / max(wsum, 1e-6);
}

[numthreads(8, 8, 1)]
void cs_blur(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    int2 origin = int2(gid.xy) * 8 - 2;
    int2 last = int2(g_ints.xy) - 1;
    for (uint i = gi; i < 144u; i += 64u)
    {
        uint2 l = uint2(i % 12u, i / 12u);
        int2 q = clamp(origin + int2(l), 0, last);
        g_tileV[l.y][l.x] = void_cs_input.Load(int3(q, 0));
        g_tileZ[l.y][l.x] = LinZ(q);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = gi; j < 96u; j += 64u)
    {
        uint row = j / 8u, col = j % 8u + 2u;
        float a[5], z[5];
        [unroll] for (int k = 0; k < 5; k++)
        {
            a[k] = g_tileV[row][col + k - 2];
            z[k] = g_tileZ[row][col + k - 2];
        }
        g_rowV[row][col - 2u] = BlurTaps(a, z);
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 id = gid.xy * 8u + gt.xy;
    if (any(id >= g_ints.xy)) return;
    float a[5], z[5];
    [unroll] for (int k = 0; k < 5; k++)
    {
        a[k] = g_rowV[gt.y + k][gt.x];
        z[k] = g_tileZ[gt.y + k][gt.x + 2u];
    }
    void_cs_out[id] = BlurTaps(a, z);
}

float4 vs_full(uint id : SV_VertexID) : SV_Position
{
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// Into the screen-shadow target inside the region; the blend state writes green only with
// op MIN: .y = min(engine CSM visibility, contact visibility).
float4 ps_composite(float4 pos : SV_Position) : SV_Target
{
    float v = void_cs_input.Load(int3(int2(pos.xy) - int2(g_ints.zw), 0));
    return float4(1.0, saturate(v), 1.0, 1.0);
}

float4 ps_debug(float4 pos : SV_Position) : SV_Target
{
    float v = void_cs_input.SampleLevel(void_cs_linear, pos.xy * g_debug.xy, 0.0);
    return float4(v, v, v, 1.0);
}
";

/// Compiled shaders (device independent).
pub struct Bytecode {
    /// `cs_contact` per `QUALITIES` entry (steps as a compile-time constant).
    contact: Vec<Vec<u8>>,
    blur: Vec<u8>,
    vs: Vec<u8>,
    composite: Vec<u8>,
    debug: Vec<u8>,
    pub compile_ms: f64,
}

/// Compiled on first use (background thread from `install`, or the offline check); render
/// threads only use it once `BYTECODE_READY` is set.
static BYTECODE: LazyLock<Result<Bytecode, String>> = LazyLock::new(|| {
    let t = std::time::Instant::now();
    let c = |entry, target| hlsl::compile(CONTACT_HLSL, entry, target);
    Ok(Bytecode {
        contact: QUALITIES
            .iter()
            .map(|(_, steps)| {
                let src = format!("#define CS_STEPS {steps}\n{CONTACT_HLSL}");
                hlsl::compile(&src, s!("cs_contact"), s!("cs_5_0"))
            })
            .collect::<Result<_, _>>()?,
        blur: c(s!("cs_blur"), s!("cs_5_0"))?,
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

/// Every slot the pass and the debug view bind (t0-t2, b0-b2, s0-s1, u0).
const TOUCHED: Slots = Slots {
    srv: 3,
    cb: 3,
    sampler: 2,
    uav: 1,
};

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

/// RTV format for the screen-shadow texture format (typeless formats get a UNORM/FLOAT view).
fn target_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match f {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R16G16_TYPELESS => DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R8G8_TYPELESS => DXGI_FORMAT_R8G8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM
        | DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_R10G10B10A2_UNORM
        | DXGI_FORMAT_R16G16B16A16_FLOAT
        | DXGI_FORMAT_R11G11B10_FLOAT
        | DXGI_FORMAT_R16G16_FLOAT
        | DXGI_FORMAT_R8G8_UNORM => f,
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

/// Views we created for engine textures, keyed by the resource pointer (at most a few
/// entries: the depth textures of the current resolution).
#[derive(Default)]
struct Views {
    depth: HashMap<usize, (ID3D11Resource, ID3D11ShaderResourceView)>,
}

const MAX_CACHED_VIEWS: usize = 4;

type TargetViews = HashMap<usize, (ID3D11Resource, ID3D11RenderTargetView)>;

/// RTVs of screen-shadow targets (contact shadows and the external `.y` term), keyed by the
/// resource pointer; created on the resource's own device.
static TARGET_VIEWS: LazyLock<Mutex<TargetViews>> = LazyLock::new(Mutex::default);

fn target_rtv(r: &ID3D11Resource) -> Result<ID3D11RenderTargetView, String> {
    let key = r.as_raw() as usize;
    let mut views = lock(&TARGET_VIEWS);
    if let Some((_, v)) = views.get(&key) {
        return Ok(v.clone());
    }
    let (_, d) = tex2d(r)?;
    let format = target_view_format(d.Format)
        .ok_or_else(|| format!("screen-shadow format {} not handled", d.Format.0))?;
    let desc = D3D11_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
        },
    };
    let mut v = None;
    // SAFETY: valid resource, descriptor and out pointer; the view is created on the
    // resource's device.
    unsafe {
        let dev = r.GetDevice().map_err(create_err("target device"))?;
        dev.CreateRenderTargetView(r, Some(&desc), Some(&mut v))
    }
    .map_err(create_err("screen-shadow RTV"))?;
    let v = v.ok_or("no screen-shadow RTV")?;
    if views.len() >= MAX_CACHED_VIEWS {
        views.clear();
    }
    views.insert(key, (r.clone(), v.clone()));
    Ok(v)
}

struct Vis {
    size: [u32; 2],
    srv: [ID3D11ShaderResourceView; 2],
    uav: [ID3D11UnorderedAccessView; 2],
}

/// Validated inputs of one pass.
struct Job {
    depth: ID3D11ShaderResourceView,
    rtv: ID3D11RenderTargetView,
    tex: [u32; 2],
    params: ShaderParams,
    quality: usize,
}

struct Pipeline {
    device: ID3D11Device,
    contact: Vec<ID3D11ComputeShader>,
    blur: ID3D11ComputeShader,
    vs: ID3D11VertexShader,
    composite: ID3D11PixelShader,
    debug: ID3D11PixelShader,
    params: ID3D11Buffer,
    debug_params: ID3D11Buffer,
    linear: ID3D11SamplerState,
    raster: ID3D11RasterizerState,
    no_depth: ID3D11DepthStencilState,
    min_green: ID3D11BlendState,
    vis: Option<Vis>,
    views: Views,
    /// Region size of the last result (debug view).
    shown: Option<[u32; 2]>,
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
                DestBlend: D3D11_BLEND_ONE,
                BlendOp: D3D11_BLEND_OP_MIN,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_ONE,
                BlendOpAlpha: D3D11_BLEND_OP_MIN,
                RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_GREEN.0 as u8,
            };
            let mut min_green = None;
            dev.CreateBlendState(&bd, Some(&mut min_green))
                .map_err(create_err("blend state"))?;
            Ok(Self {
                device: dev.clone(),
                contact: code
                    .contact
                    .iter()
                    .map(|b| cs(b))
                    .collect::<Result<_, _>>()?,
                blur: cs(&code.blur)?,
                vs: vs.ok_or("no vertex shader")?,
                composite: ps(&code.composite)?,
                debug: ps(&code.debug)?,
                params: dynamic_cb(dev, size_of::<ShaderParams>())?,
                debug_params: dynamic_cb(dev, 16)?,
                linear: linear.ok_or("no sampler")?,
                raster: raster.ok_or("no rasterizer state")?,
                no_depth: no_depth.ok_or("no depth-stencil state")?,
                min_green: min_green.ok_or("no blend state")?,
                vis: None,
                views: Views::default(),
                shown: None,
            })
        }
    }

    fn depth_srv(&mut self, r: &ID3D11Resource) -> Result<ID3D11ShaderResourceView, String> {
        let key = r.as_raw() as usize;
        if let Some((_, v)) = self.views.depth.get(&key) {
            return Ok(v.clone());
        }
        let (_, d) = tex2d(r)?;
        let format = depth_view_format(d.Format)
            .ok_or_else(|| format!("depth format {} not handled", d.Format.0))?;
        if d.SampleDesc.Count != 1 {
            return Err(format!(
                "multisampled depth ({} samples)",
                d.SampleDesc.Count
            ));
        }
        let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                },
            },
        };
        let mut v = None;
        // SAFETY: valid resource, descriptor and out pointer.
        unsafe {
            self.device
                .CreateShaderResourceView(r, Some(&desc), Some(&mut v))
        }
        .map_err(create_err("depth SRV"))?;
        let v = v.ok_or("no depth SRV")?;
        if self.views.depth.len() >= MAX_CACHED_VIEWS {
            self.views.depth.clear();
        }
        self.views.depth.insert(key, (r.clone(), v.clone()));
        Ok(v)
    }

    fn vis(&mut self, size: [u32; 2]) -> Result<&Vis, String> {
        if self.vis.as_ref().is_none_or(|v| v.size != size) {
            self.vis = None;
            let dev = &self.device;
            let make =
                || -> Result<(ID3D11ShaderResourceView, ID3D11UnorderedAccessView), String> {
                    let d = D3D11_TEXTURE2D_DESC {
                        Width: size[0],
                        Height: size[1],
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: DXGI_FORMAT_R16_FLOAT,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Usage: D3D11_USAGE_DEFAULT,
                        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0)
                            as u32,
                        ..Default::default()
                    };
                    let (mut t, mut s, mut u) = (None, None, None);
                    // SAFETY (whole block): creation with valid descriptors and out pointers.
                    unsafe {
                        dev.CreateTexture2D(&d, None, Some(&mut t))
                            .map_err(create_err("visibility texture"))?;
                        let t: ID3D11Texture2D = t.ok_or("no visibility texture")?;
                        dev.CreateShaderResourceView(&t, None, Some(&mut s))
                            .map_err(create_err("visibility SRV"))?;
                        dev.CreateUnorderedAccessView(&t, None, Some(&mut u))
                            .map_err(create_err("visibility UAV"))?;
                    }
                    Ok((s.ok_or("no SRV")?, u.ok_or("no UAV")?))
                };
            let (a, b) = (make()?, make()?);
            self.vis = Some(Vis {
                size,
                srv: [a.0, b.0],
                uav: [a.1, b.1],
            });
            log::line(&format!("contact: targets for {}x{}", size[0], size[1]));
        }
        self.vis
            .as_ref()
            .ok_or_else(|| "no visibility targets".to_owned())
    }

    fn record(&mut self, ctx: &ID3D11DeviceContext, job: &Job) -> Result<(), String> {
        self.vis(job.tex)?;
        // SAFETY: capture/restore on the caller's context, which this thread records.
        let saved = unsafe { Saved::capture(ctx, TOUCHED) };
        let r = upload(ctx, &self.params, &job.params).map(|()| {
            // SAFETY: recording on the caller's context with our own live objects.
            unsafe { self.passes(ctx, job) }
        });
        // SAFETY: as above.
        unsafe { saved.restore(ctx) };
        r?;
        self.shown = Some([job.params.ints[0], job.params.ints[1]]);
        Ok(())
    }

    /// # Safety
    /// `ctx` is recorded by this thread; the views belong to this pipeline's device.
    unsafe fn passes(&self, ctx: &ID3D11DeviceContext, job: &Job) {
        let Some(v) = self.vis.as_ref() else { return };
        let [w, h, x, y] = job.params.ints;
        let groups = |n: u32| n.div_ceil(8);
        let keep = [u32::MAX];
        let set_uav = |u: Option<&ID3D11UnorderedAccessView>| {
            let a = [u.cloned()];
            // SAFETY: one-element arrays for one slot.
            unsafe { ctx.CSSetUnorderedAccessViews(0, 1, Some(a.as_ptr()), Some(keep.as_ptr())) };
        };
        // SAFETY (whole block): bindings of our own objects and the engine's textures; every
        // slot used here is in `TOUCHED`.
        unsafe {
            ctx.OMSetRenderTargets(None, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            // Ray march into vis[1].
            ctx.CSSetShader(&self.contact[job.quality], None);
            ctx.CSSetShaderResources(0, Some(&[Some(job.depth.clone()), None, None]));
            set_uav(Some(&v.uav[1]));
            ctx.Dispatch(groups(w), groups(h), 1);
            // Blur vis[1] -> vis[0].
            set_uav(None);
            ctx.CSSetShader(&self.blur, None);
            ctx.CSSetShaderResources(2, Some(&[Some(v.srv[1].clone())]));
            set_uav(Some(&v.uav[0]));
            ctx.Dispatch(groups(w), groups(h), 1);
            set_uav(None);
            ctx.CSSetShaderResources(0, Some(&[None, None, None]));
            // min() into the target's green channel, inside the region.
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.composite, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.PSSetShaderResources(0, Some(&[None, None, Some(v.srv[0].clone())]));
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: x as f32,
                TopLeftY: y as f32,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(&self.min_green, None, D3D11_DEFAULT_SAMPLE_MASK);
            ctx.OMSetDepthStencilState(&self.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(job.rtv.clone())]), None);
            ctx.Draw(3, 0);
        }
    }

    fn draw_debug(
        &self,
        ctx: &ID3D11DeviceContext,
        rtv: &ID3D11RenderTargetView,
        size: [u32; 2],
    ) -> Result<(), String> {
        let (Some(v), Some(shown)) = (self.vis.as_ref(), self.shown) else {
            return Ok(());
        };
        let p = [
            shown[0] as f32 / v.size[0] as f32 / size[0] as f32,
            shown[1] as f32 / v.size[1] as f32 / size[1] as f32,
            0.0f32,
            0.0,
        ];
        upload(ctx, &self.debug_params, &p)?;
        // SAFETY (whole block): immediate context under the engine lock (caller); state
        // captured and restored around our draw.
        unsafe {
            let saved = Saved::capture(ctx, TOUCHED);
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&self.debug, None);
            ctx.PSSetConstantBuffers(2, Some(&[Some(self.debug_params.clone())]));
            ctx.PSSetSamplers(1, Some(&[Some(self.linear.clone())]));
            ctx.PSSetShaderResources(2, Some(&[Some(v.srv[0].clone())]));
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
/// (`overlay.rs`, `contact_debug = 1`): draws the last contact visibility full-screen.
pub fn draw_debug(
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    size: [u32; 2],
) -> Result<(), String> {
    let Ok(g) = PIPE.try_lock() else {
        return Ok(());
    };
    let Some(p) = g.as_ref() else {
        return Ok(());
    };
    // SAFETY: plain getter.
    let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
    if dev.as_raw() != p.device.as_raw() {
        return Ok(());
    }
    p.draw_debug(ctx, rtv, size)
}

// ------------------------------------------------------------------------ engine read --

/// Ok(true) = contact shadows recorded; Ok(false) = no contact shadows this pass (off,
/// shaders compiling, no CSM pass, no sun). The registered external `.y` term is drawn after
/// them (CSM mode only); its result is counted in `ao::external`.
fn record_pass(pass: u64) -> Result<bool, String> {
    let s = settings::get();
    let contact = s.contact_shadows && BYTECODE_READY.load(Ordering::Acquire);
    let external = ao::external::registered(Channel::Y);
    if !contact && !external {
        return Ok(false);
    }
    let view = mem::u64_at(pass + PASS_VIEW)
        .filter(|&v| v != 0)
        .ok_or("pass view unreadable")?;
    let mode = mem::u32_at(view + VIEW_SHADOW_MODE).ok_or("view unreadable")?;
    if mode != 0 {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }
    let engine = mem::u64_at(pass + PASS_ENGINE_CTX).ok_or("engine context unreadable")?;
    let d3d = mem::u64_at(engine)
        .filter(|&d| d != 0)
        .ok_or("no D3D11 context")? as *mut c_void;
    // SAFETY: the engine context's live ID3D11DeviceContext (deferred while recording).
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&d3d) }
        .ok_or("null D3D11 context")?
        .clone();
    let depth = crate::fsr_scale::pass_image(pass, HASH_DEPTH).ok_or("no depth image")?;
    let target =
        crate::fsr_scale::pass_image(pass, HASH_SCREEN_SHADOWS).ok_or("no screen-shadow image")?;
    let proj_mem = mem::f32s::<16>(view + VIEW_PROJECTION).ok_or("projection unreadable")?;
    let view_mem = mem::f32s::<16>(view + VIEW_VIEW).ok_or("view matrix unreadable")?;
    let sun_world = mem::f32s::<3>(view + VIEW_SUN).ok_or("sun direction unreadable")?;
    let mut rect = [0i32; 4];
    for (i, r) in rect.iter_mut().enumerate() {
        *r = mem::u32_at(view + VIEW_VIEWPORT + 4 * i as u64).ok_or("viewport unreadable")? as i32;
    }
    let proj = ao::projection(&proj_mem, 1)?;
    let sun = sun_in_view(&view_mem, sun_world);
    if contact && sun.is_none() {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
    }
    let (_, dd) = tex2d(&depth)?;
    let (_, td) = tex2d(&target)?;
    let tex = [dd.Width, dd.Height];
    if [td.Width, td.Height] != tex {
        return Err(format!(
            "screen-shadow target {}x{} differs from depth {}x{}",
            td.Width, td.Height, tex[0], tex[1]
        ));
    }
    let region = region(rect, tex).ok_or_else(|| format!("viewport {rect:?} outside {tex:?}"))?;
    if !FIRST_LOGGED.swap(true, Ordering::Relaxed) {
        let info = format!(
            "first SCREENSHADOWS pass: depth {}x{} format {}, target format {}, viewport {rect:?} -> region {region:?}, sun world {sun_world:?} -> view frame {sun:?}, view matrix row 3 {:?} / column 3 {:?} (translation slot), projection P00 {} P11 {}",
            dd.Width,
            dd.Height,
            dd.Format.0,
            td.Format.0,
            [view_mem[12], view_mem[13], view_mem[14], view_mem[15]],
            [view_mem[3], view_mem[7], view_mem[11]],
            proj.p00,
            proj.p11,
        );
        log::line(&format!("contact: {info}"));
        *lock(&INFO) = Some(info);
    }
    let mut recorded = false;
    if contact && let Some(sun) = sun {
        let code = bytecode().map_err(|e| format!("shader compilation failed: {e}"))?;
        let mut g = lock(&PIPE);
        // SAFETY: plain getter.
        let dev = unsafe { ctx.GetDevice() }.map_err(create_err("device"))?;
        if g.as_ref().is_none_or(|p| p.device.as_raw() != dev.as_raw()) {
            *g = None;
            *g = Some(Pipeline::new(&dev, code)?);
        }
        let p = g.as_mut().ok_or("no pipeline")?;
        let job = Job {
            depth: p.depth_srv(&depth)?,
            rtv: target_rtv(&target)?,
            tex,
            params: shader_params(&proj, region, sun, &s),
            quality: usize::from(s.contact_quality).min(QUALITIES.len() - 1),
        };
        p.record(&ctx, &job)?;
        recorded = true;
    }
    if external {
        // Failures are logged and counted by `ao::external` (not contact failures).
        let _ = ao::external::apply(Channel::Y, &ctx, &target_rtv(&target)?, region);
    }
    Ok(recorded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_direction_to_view_frame() {
        // Identity view: world = engine view space (camera looks down -z).
        let id = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let s = sun_in_view(&id, [0.0, 0.0, -2.0]).unwrap();
        assert_eq!(s, [0.0, 0.0, 1.0]);
        // Camera yawed 90 degrees left (looks down world -x): world -x is forward.
        let yaw = [
            0.0, 0.0, -1.0, 5.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let s = sun_in_view(&yaw, [-1.0, 0.0, 0.0]).unwrap();
        assert!((s[2] - 1.0).abs() < 1e-6 && s[0].abs() < 1e-6, "{s:?}");
        assert!(sun_in_view(&id, [0.0; 3]).is_none());
    }

    #[test]
    fn regions() {
        assert_eq!(
            region([0, 0, 2303, 967], [3440, 1440]),
            Some([0, 0, 2304, 968])
        );
        assert_eq!(
            region([0, 0, 3439, 1439], [3440, 1440]),
            Some([0, 0, 3440, 1440])
        );
        assert_eq!(
            region([0, 0, 5000, 1439], [3440, 1440]),
            Some([0, 0, 3440, 1440])
        );
        assert_eq!(region([-1, 0, 10, 10], [3440, 1440]), None);
        assert_eq!(region([4000, 0, 4010, 10], [3440, 1440]), None);
    }

    #[test]
    fn params_and_formats() {
        let col = ao::engine_projection(2.4, 1.0, 0.01, 7000.0, [0.0; 2]);
        let p = ao::projection(&col, 1).unwrap();
        let s = Settings::default();
        let k = shader_params(&p, [0, 0, 2304, 968], [0.0, 0.6, 0.8], &s);
        assert_eq!(k.ints, [2304, 968, 0, 0]);
        assert_eq!(k.sun[3], s.contact_length);
        assert_eq!(k.march[0], s.contact_thickness);
        // Two R24 steps at 10 m are a few millimetres.
        let q = k.depth[3] * 100.0;
        assert!(q > 1e-4 && q < 0.05, "{q}");
        assert_eq!(
            depth_view_format(DXGI_FORMAT_R24G8_TYPELESS),
            Some(DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
        );
        assert_eq!(
            target_view_format(DXGI_FORMAT_R8G8B8A8_TYPELESS),
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        );
        assert_eq!(target_view_format(DXGI_FORMAT_BC1_UNORM), None);
        for (i, (n, _)) in QUALITIES.iter().enumerate() {
            assert_eq!(parse_quality(n), Some(i as u8));
            assert_eq!(quality_name(i as u8), *n);
        }
    }
}
