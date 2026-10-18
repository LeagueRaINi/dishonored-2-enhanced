//! Hybrid ray-traced reflections (docs/reflections.md sections 4 and 4b, STATUS E077, E084):
//! screen-space reflections first, inline DXR 1.1 rays where they left something unresolved
//! (Far Cry 6 / AMD hybrid reflections).
//!
//! - D3D11, FSR path only (`ssr.rs` `cs_rt_classify`, after the SSR result): per region pixel
//!   the weight SSR could not fill (faded by roughness up to `rt_max_roughness` and toward the
//!   BVH radius), per half-resolution texel one ray request (shading normal + GGX alpha),
//!   written into D3D11-created textures shared with D3D12 ([`classify`]); the frame's camera
//!   goes to [`queue`].
//! - D3D12, FSR's command list after its input fence wait and before the FSR dispatch
//!   (`fsr.rs` `dispatch_fsr` -> [`record`]): `rt_reflect.hlsl` compacts the requests, applies
//!   the ray budget, traces them against `rt`'s camera-radius TLAS (inline `RayQuery`), shades
//!   hits (frame colour where the hit is on screen, else the packed per-primitive data with
//!   sun + shadow ray + ambient, calibrated per screen tile into stored units against the
//!   on-screen hits, blended smoothly between the two), denoises them like the SSR (ratio-
//!   estimator resolve, temporal accumulation with the engine motion vectors, bilateral blur;
//!   with `ssr_temporal`) and blends the result into FSR's colour input with premultiplied
//!   confidence; FSR accumulates it. Misses keep the engine's IBL.
//! - Never stalls: a frame is skipped when the scene is not ready, the ring slot is still in
//!   flight or anything does not match; all fallible work happens before the first command.
//! - Without FSR there is no D3D12 point before the TAA: no ray-traced reflections (section 4).
//!
//! Offline check with a synthetic BVH (hardware and WARP): [`offline`].

pub mod offline;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::{log, settings};

include!(concat!(env!("OUT_DIR"), "/dxil.rs"));

/// `rt_reflect_debug` views (FSR's colour input is replaced: FSR still upscales it).
pub const DEBUG_VIEWS: [&str; 6] = ["off", "rays", "hits", "rt", "history", "instances"];
/// `instances`: traced rays coloured by the type of the instance they hit (`rt::SceneView::
/// instance_types`); the denoiser is skipped.
const DEBUG_INSTANCES: u8 = 5;
/// Initial stored-units / absolute-units scale of shaded hits before the on-screen calibration
/// converged (H: engine stored hdrBuffer ~ absolute x 3.58 in daylight, E073).
const INITIAL_SCALE: f32 = 3.58;
/// Command-list ring (one set of descriptors, timestamps and keep-alives per FSR frame in
/// flight).
const SETS: usize = 4;
const SRVS: u32 = 5;
const UAVS: u32 = 14;
/// Calibration buffer: 64-byte header (counters, read back) + 16 bytes per screen tile (8 x 4).
const CALIB_HEADER: u64 = 64;
const CALIB_BYTES: u64 = CALIB_HEADER + 32 * 16;
const F_SHADOW: u32 = 1;
/// Offline check only: shade every hit (no on-screen colour).
const F_NO_SCREEN: u32 = 2;
/// Photo accumulation (`rt_photo` + averaging composite) / start the sum from 0.
const F_PHOTO: u32 = 4;
const F_CLEAR: u32 = 8;
/// The previous frame's accumulation is bound (consecutive denoised frames).
const F_HISTORY: u32 = 16;
/// Resolve + temporal + blur (with `ssr_temporal` and motion vectors).
const F_DENOISE: u32 = 32;
/// Offline before/after comparison only: the E077 hit shading (global calibration, hard
/// on/off-screen switch, BVH fade around the TLAS centre, rotation every frame), no denoiser.
const F_LEGACY: u32 = 64;
/// Offline before/after comparison only: the E085 denoiser without firefly handling.
const F_LINEAR: u32 = 128;
/// Fireflies (E089): a glossy hit brighter than this many times the mean luminance of the hits
/// around it is clamped to that (SSR resolve and ray resolve). 16 keeps the lamps' reflected
/// energy within ~10 % of an unclamped reference offline (SSR: 4 keeps 31 %, 8 67 %).
pub(crate) const FIREFLY_K: f32 = 16.0;
/// Offline sweeps only: `FIREFLY_K` override (f32 bits; 0 = the constant).
static OFFLINE_FIREFLY_K: AtomicU32 = AtomicU32::new(0);

/// The per-hit firefly clamp factor in use.
pub(crate) fn firefly_k() -> f32 {
    match OFFLINE_FIREFLY_K.load(Ordering::Relaxed) {
        0 => FIREFLY_K,
        b => f32::from_bits(b),
    }
}

/// Offline sweeps: override `FIREFLY_K` (None: back to the constant).
pub(crate) fn set_offline_firefly_k(k: Option<f32>) {
    OFFLINE_FIREFLY_K.store(k.map_or(0, f32::to_bits), Ordering::Relaxed);
}
const KERNELS: [&str; 9] = [
    "rt_prepare",
    "rt_compact",
    "rt_args",
    "rt_trace",
    "rt_composite",
    "rt_photo",
    "rt_resolve",
    "rt_temporal",
    "rt_blur",
];

pub fn debug_name(d: u8) -> &'static str {
    DEBUG_VIEWS[usize::from(d).min(DEBUG_VIEWS.len() - 1)]
}

pub fn parse_debug(v: &str) -> Option<u8> {
    DEBUG_VIEWS.iter().position(|n| *n == v).map(|i| i as u8)
}

fn dxil(entry: &str) -> Result<&'static [u8], String> {
    DXIL.iter()
        .find(|(n, _)| *n == entry)
        .map(|(_, b)| *b)
        .ok_or_else(|| format!("{entry}: no precompiled DXIL"))
}

// ------------------------------------------------------------------------------ status --

static FRAMES: AtomicU64 = AtomicU64::new(0);
static NO_SCENE: AtomicU64 = AtomicU64::new(0);
static BUSY: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
static GPU_MS: AtomicU32 = AtomicU32::new(0);
/// GPU ms of the last frame read back (offline per-frame measurements).
static GPU_LAST_MS: AtomicU32 = AtomicU32::new(0);
static LAST: Mutex<Stats> = Mutex::new(Stats {
    requested: 0,
    traced: 0,
    hits: 0,
    screen_hits: 0,
    misses: 0,
    scale: 0.0,
    tile_scale: [0.0; 2],
    calib_samples: 0,
    denoised: false,
});
static ERROR: Mutex<Option<String>> = Mutex::new(None);
static CAMERA_LOGGED: AtomicBool = AtomicBool::new(false);
/// Offline check of the D3D11 classify pass (`ssr/offline/temporal.rs`): plain D3D11 targets
/// returned by [`classify`] instead of the shared ones (no D3D12 device or scene needed).
static OFFLINE_TARGETS: Mutex<Option<Classify>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Counters of one traced frame (read back a few frames late).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub requested: u32,
    pub traced: u32,
    pub hits: u32,
    pub screen_hits: u32,
    pub misses: u32,
    /// Stored / absolute units of shaded hits (calibration).
    pub scale: f32,
    /// Smallest and largest per-tile scale.
    pub tile_scale: [f32; 2],
    /// Calibration samples (fully on-screen hits) of the frame.
    pub calib_samples: u32,
    /// The denoiser ran (`ssr_temporal` with motion vectors).
    pub denoised: bool,
}

/// State for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// Frames traced and composited into FSR's input.
    pub frames: u64,
    /// Skipped: no scene (rt off, building, map change), or the camera outside it.
    pub no_scene: u64,
    /// Skipped: the ring slot was still in flight.
    pub busy: u64,
    pub failures: u64,
    /// GPU time of our work on FSR's list (running average, ms).
    pub gpu_ms: Option<f32>,
    pub last: Stats,
    pub error: Option<String>,
    /// With `rt_reflections = 1`: the missing prerequisite or what it waits for (None: all
    /// present, or the feature off).
    pub blocker: Option<String>,
}

pub fn status() -> Status {
    let ms = f32::from_bits(GPU_MS.load(Ordering::Relaxed));
    Status {
        frames: FRAMES.load(Ordering::Relaxed),
        no_scene: NO_SCENE.load(Ordering::Relaxed),
        busy: BUSY.load(Ordering::Relaxed),
        failures: FAILURES.load(Ordering::Relaxed),
        gpu_ms: (ms > 0.0).then_some(ms),
        last: *lock(&LAST),
        error: lock(&ERROR).clone(),
        blocker: blocker(),
    }
}

/// How a missing prerequisite is reported besides the card and the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Severity {
    /// A setting that is off (`ssr`, `rt`, `fsr`): the user's own choice. Toasted only when
    /// the user turns ray-traced reflections on while it is off.
    Setting,
    /// A failure or missing capability: toasted at once.
    Definite,
    /// Clears by itself (FSR starting, device being created, scene building): toasted only
    /// after `TRANSIENT_TOAST` without change.
    Transient,
}

/// A missing prerequisite: the text, an identity (`key`; the scene's build phases share one)
/// and how it is reported.
#[derive(Clone, Debug, PartialEq)]
struct Blocker {
    text: String,
    key: &'static str,
    severity: Severity,
}

/// How long a transient blocker must persist before it is toasted.
const TRANSIENT_TOAST: std::time::Duration = std::time::Duration::from_secs(10);

/// What the prerequisite check looks at (gathered by [`prereq_now`]).
#[derive(Clone, Debug, Default)]
struct Prereq {
    rt_reflections: bool,
    ssr: bool,
    rt: bool,
    fsr: bool,
    fsr_running: bool,
    device_failure: Option<String>,
    /// None: no device yet; Some(inline ray tracing supported).
    device: Option<bool>,
    gpu_lost: bool,
    scene_error: Option<String>,
    scene_phase: String,
    camera_mismatch: bool,
}

fn prereq_now() -> Prereq {
    let s = settings::get();
    if !s.rt_reflections {
        return Prereq::default();
    }
    let sc = crate::rt::scene::status();
    Prereq {
        rt_reflections: s.rt_reflections,
        ssr: s.ssr,
        rt: s.rt,
        fsr: s.fsr,
        fsr_running: crate::fsr::running(),
        device_failure: crate::rt::device::failure(),
        device: crate::rt::device::get().map(|d| d.inline_raytracing),
        gpu_lost: crate::rt::scene::gpu_lost(),
        scene_error: sc.error,
        scene_phase: sc.phase,
        camera_mismatch: CAMERA_LOGGED.load(Ordering::Relaxed)
            && FRAMES.load(Ordering::Relaxed) == 0,
    }
}

/// The first missing prerequisite, in the order a user fixes them: settings first, then FSR,
/// device, scene and camera. None with `rt_reflections = 0` or everything present.
fn blocker_for(p: &Prereq) -> Option<Blocker> {
    let b = |key, text: &str, severity| {
        Some(Blocker {
            text: text.to_owned(),
            key,
            severity,
        })
    };
    use Severity::{Definite, Setting, Transient};
    if !p.rt_reflections {
        return None;
    }
    if !p.ssr {
        return b("ssr", "Reflections on (ssr = 1)", Setting);
    }
    if !p.rt {
        return b("rt", "the ray-tracing scene (rt = 1, Photo tab)", Setting);
    }
    if !p.fsr {
        return b(
            "fsr",
            "FSR on (the rays are recorded on FSR's D3D12 list; the engine TAA has none)",
            Setting,
        );
    }
    if !p.fsr_running {
        return b("fsr running", "FSR running (not started yet)", Transient);
    }
    if let Some(e) = &p.device_failure {
        return b("device failed", &format!("a D3D12 device ({e})"), Definite);
    }
    match p.device {
        None => return b("device", "the D3D12 device (not created yet)", Transient),
        Some(false) => return b("dxr", "a GPU with DXR 1.1 inline ray tracing", Definite),
        Some(true) => {}
    }
    if p.gpu_lost {
        return b(
            "device lost",
            "the D3D12 device (lost; restart the game)",
            Definite,
        );
    }
    if let Some(e) = &p.scene_error {
        return b(
            "scene failed",
            &format!("the scene (failed: {e})"),
            Definite,
        );
    }
    if p.scene_phase != "ready" {
        let phase = if p.scene_phase.is_empty() {
            "not built yet"
        } else {
            &p.scene_phase
        };
        return b("scene", &format!("the scene ({phase})"), Transient);
    }
    if p.camera_mismatch {
        return b(
            "camera",
            "a camera inside the scene (view-matrix units differ, see the log)",
            Definite,
        );
    }
    None
}

/// What ray-traced reflections still need (`rt_reflections = 1` only), in the order a user
/// fixes it: settings first, then device, scene and camera.
pub fn blocker() -> Option<String> {
    blocker_for(&prereq_now()).map(|b| b.text)
}

/// The blocker being reported: its key and text, since when, toasted yet.
struct Reported {
    key: &'static str,
    text: String,
    since: std::time::Instant,
    toasted: bool,
}

/// Toast policy over frames ([`tick`]; pure for the tests).
#[derive(Default)]
struct Reporter {
    current: Option<Reported>,
    /// `rt_reflections` at the previous step (None before the first).
    enabled: Option<bool>,
}

/// One step's messages.
#[derive(Debug, Default, PartialEq)]
struct Report {
    log: Option<String>,
    toast: Option<String>,
}

impl Reporter {
    /// A log line whenever the blocker changes. Toasts: a definite blocker at once; a
    /// transient one after it persisted `TRANSIENT_TOAST` (the scene's build phases count as
    /// one); a setting the user has off only when they turn ray-traced reflections on while it
    /// is off (turning a prerequisite off themselves shows only in the card). Clearing resets.
    fn step(&mut self, enabled: bool, b: Option<Blocker>, now: std::time::Instant) -> Report {
        let just_enabled = self.enabled == Some(false) && enabled;
        self.enabled = Some(enabled);
        let mut out = Report::default();
        let Some(b) = b else {
            self.current = None;
            return out;
        };
        match self.current.as_mut() {
            Some(r) if r.key == b.key => {
                if r.text != b.text {
                    out.log = Some(b.text.clone());
                    r.text = b.text;
                }
                let due = match b.severity {
                    Severity::Transient => now.duration_since(r.since) > TRANSIENT_TOAST,
                    Severity::Setting => just_enabled,
                    Severity::Definite => false,
                };
                if !r.toasted && due {
                    out.toast = Some(r.text.clone());
                    r.toasted = true;
                }
            }
            _ => {
                out.log = Some(b.text.clone());
                let now_toast = match b.severity {
                    Severity::Definite => true,
                    Severity::Setting => just_enabled,
                    Severity::Transient => false,
                };
                if now_toast {
                    out.toast = Some(b.text.clone());
                }
                self.current = Some(Reported {
                    key: b.key,
                    text: b.text,
                    since: now,
                    toasted: now_toast,
                });
            }
        }
        out
    }
}

static REPORTER: Mutex<Reporter> = Mutex::new(Reporter {
    current: None,
    enabled: None,
});

/// Per frame (main thread, from `ssr::tick`): see [`Reporter::step`].
pub fn tick() {
    let p = prereq_now();
    let r = lock(&REPORTER).step(p.rt_reflections, blocker_for(&p), std::time::Instant::now());
    if let Some(l) = r.log {
        log::line(&format!("rt_reflect: waiting for {l}"));
    }
    if let Some(t) = r.toast {
        crate::notify::warn(format!("Ray-traced reflections need: {t}"));
    }
}

fn fail(e: String) {
    let n = FAILURES.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n.is_power_of_two() {
        log::line(&format!("rt_reflect: {e} (failure {})", n + 1));
    }
    *lock(&ERROR) = Some(e);
}

// ------------------------------------------------------------------------- D3D11 side --

/// What `ssr.rs` binds for `cs_rt_classify` this frame.
#[derive(Clone)]
pub(crate) struct Classify {
    pub rtw: ID3D11UnorderedAccessView,
    pub req0: ID3D11UnorderedAccessView,
    pub req1: ID3D11UnorderedAccessView,
    /// Region-sized R32F copy of the hardware depth (D3D12 consumers).
    pub depth: ID3D11UnorderedAccessView,
    /// `g_rt`: on, max perceptual roughness, BVH radius (m), 0.
    pub params: [f32; 4],
}

/// The camera of the classified frame, for the D3D12 side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Camera position (world metres, Z up).
    pub cam: [f32; 3],
    /// World -> SSR view frame rows (x right, y up, z forward).
    pub rot: [[f32; 4]; 3],
    /// P00, P11, jitter x, y.
    pub proj: [f32; 4],
    /// Linear depth = mul / (d - add).
    pub depth: [f32; 2],
    /// Region (x, y, w, h) in the engine colour/depth (= FSR's inputs).
    pub region: [u32; 4],
    pub frame: u32,
    /// The SSR temporal stage ran this frame (`ssr_temporal` with motion vectors): the D3D12
    /// side denoises the rays the same way.
    pub temporal: bool,
}

/// The D3D11-created textures shared with D3D12 for one region size.
struct Shared {
    key: ([u32; 2], usize),
    rtw: crate::fsr::Shared,
    req0: crate::fsr::Shared,
    req1: crate::fsr::Shared,
    depth: crate::fsr::Shared,
}

struct State {
    shared: Option<Shared>,
    pending: Option<View>,
    gpu: Option<Gpu12>,
    /// Photo accumulation on the FSR path: rays per texel and frame (None: live mode), the
    /// persistent sum (resource, half size, device identity) and the samples in it.
    photo_spp: Option<u32>,
    accum: Option<(ID3D12Resource, [u32; 2], usize)>,
    photo_total: u32,
    /// The last classified frame (`shared_inputs`).
    last: Option<View>,
}

// SAFETY: D3D11/D3D12 objects are free-threaded; the state is used under its mutex only.
unsafe impl Send for State {}

static STATE: Mutex<State> = Mutex::new(State {
    shared: None,
    pending: None,
    gpu: None,
    photo_spp: None,
    accum: None,
    photo_total: 0,
    last: None,
});

fn half(region: [u32; 2]) -> [u32; 2] {
    [region[0].div_ceil(2), region[1].div_ceil(2)]
}

fn identity<T: Interface>(i: &T) -> usize {
    i.cast::<windows::core::IUnknown>()
        .map_or(0, |u| u.as_raw() as usize)
}

/// FSR path, before the SSR composite: the textures to classify into this frame, or `None`
/// (setting off, no RT device / scene, other device, creation failed).
pub(crate) fn classify(dev: &ID3D11Device, region: [u32; 2]) -> Option<Classify> {
    if let Some(c) = lock(&OFFLINE_TARGETS).clone() {
        return Some(c);
    }
    let s = settings::get();
    if !s.rt_reflections || !s.rt {
        return None;
    }
    let rt = crate::rt::device::get()?;
    if !rt.inline_raytracing || identity(&rt.d11) != identity(dev) {
        return None;
    }
    let d12: ID3D12Device = rt.d12.cast().ok()?;
    let Some(scene) = crate::rt::frame_scene(&d12) else {
        NO_SCENE.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let mut st = lock(&STATE);
    let key = (region, identity(&d12));
    if st.shared.as_ref().is_none_or(|t| t.key != key) {
        st.shared = None;
        let h = half(region);
        let mk = |f, size: [u32; 2]| {
            crate::rt::device::shared_texture(&rt, f, size[0], size[1], true, false)
        };
        let made = (|| {
            Ok::<_, String>(Shared {
                key,
                rtw: mk(DXGI_FORMAT_R16G16B16A16_FLOAT, region)?,
                req0: mk(DXGI_FORMAT_R32G32B32A32_FLOAT, h)?,
                req1: mk(DXGI_FORMAT_R16G16B16A16_FLOAT, h)?,
                depth: mk(DXGI_FORMAT_R32_FLOAT, region)?,
            })
        })();
        match made {
            Ok(t) => st.shared = Some(t),
            Err(e) => {
                fail(format!("shared textures: {e}"));
                return None;
            }
        }
    }
    let t = st.shared.as_ref()?;
    Some(Classify {
        rtw: t.rtw.uav.clone()?,
        req0: t.req0.uav.clone()?,
        req1: t.req1.uav.clone()?,
        depth: t.depth.uav.clone()?,
        params: [1.0, s.rt_max_roughness.clamp(0.05, 1.0), scene.radius, 0.0],
    })
}

/// Offline check: classify into `targets` (None: back to the shared textures); drops the
/// queued frame.
pub(crate) fn set_offline_targets(targets: Option<Classify>) {
    *lock(&OFFLINE_TARGETS) = targets;
    lock(&STATE).pending = None;
}

/// After a successful classification: the frame the D3D12 side traces next.
pub(crate) fn queue(view: View) {
    let mut st = lock(&STATE);
    st.pending = Some(view);
    st.last = Some(view);
}

/// The last classified frame's D3D12-readable inputs (region-local, COMMON): hardware depth
/// (R32F, reversed-Z), requests (origin + TMin RGBA32F, normal + alpha RGBA16F, half
/// resolution), weights (RGBA16F) and the camera. Written on the D3D11 immediate context
/// before FSR's pack; a reader on another queue must wait for that work (`rt::device::
/// CrossFence` signalled after it). Only while ray-traced reflections classify (None else).
#[derive(Clone)]
pub struct SharedInputs {
    pub depth: ID3D12Resource,
    pub req0: ID3D12Resource,
    pub req1: ID3D12Resource,
    pub rtw: ID3D12Resource,
    pub view: View,
}

pub fn shared_inputs() -> Option<SharedInputs> {
    let st = lock(&STATE);
    let t = st.shared.as_ref()?;
    let view = st.last?;
    (t.key.0 == [view.region[2], view.region[3]]).then(|| SharedInputs {
        depth: t.depth.res.clone(),
        req0: t.req0.res.clone(),
        req1: t.req1.res.clone(),
        rtw: t.rtw.res.clone(),
        view,
    })
}

/// Photo mode (`photo/`): `Some(spp)` makes the FSR-path frames accumulate `spp` rays per
/// requested half-resolution texel into a persistent sum (no budget) and composite its running
/// average; `None` returns to live mode and frees the sum. Starts a new accumulation.
pub fn set_photo(spp: Option<u32>) {
    let mut st = lock(&STATE);
    st.photo_spp = spp.map(|s| s.clamp(1, 1024));
    st.photo_total = 0;
    if spp.is_none() {
        st.accum = None;
    }
}

/// Photo mode: restart the accumulation (camera moved).
pub fn reset_photo() {
    lock(&STATE).photo_total = 0;
}

/// Photo mode: samples per texel accumulated so far (0 outside photo mode).
pub fn photo_samples() -> u32 {
    let st = lock(&STATE);
    if st.photo_spp.is_some() {
        st.photo_total
    } else {
        0
    }
}

// ------------------------------------------------------------------------- D3D12 side --

/// FSR's per-frame objects at its dispatch (`fsr.rs` `dispatch_fsr`).
pub(crate) struct FsrFrame<'a> {
    pub list: &'a ID3D12GraphicsCommandList,
    /// FSR's colour (RGBA16F), depth (R32F) and motion-vector (RG16F, engine convention)
    /// inputs, COMMON state, engine pixel coordinates.
    pub color: &'a ID3D12Resource,
    pub depth: &'a ID3D12Resource,
    pub motion: &'a ID3D12Resource,
    pub render: [u32; 2],
    /// FSR's fence and the value signalled after this list.
    pub fence: &'a ID3D12Fence,
    pub fence_next: u64,
    pub ts_freq: u64,
}

/// Records the ray-traced reflections of the frame classified last into `f.list` (nothing
/// when off, not classified or not ready).
pub(crate) fn record(f: &FsrFrame) {
    let mut guard = lock(&STATE);
    let st: &mut State = &mut guard;
    let Some(view) = st.pending.take() else {
        return;
    };
    let r = (|| -> Result<bool, String> {
        let mut dev: Option<ID3D12Device5> = None;
        // SAFETY: out pointer for the list's device.
        unsafe { f.list.GetDevice(&mut dev) }.map_err(|e| e.to_string())?;
        let d12 = dev.ok_or("no device")?;
        let base: ID3D12Device = d12.cast().map_err(|e| e.to_string())?;
        let Some(scene) = crate::rt::frame_scene(&base) else {
            NO_SCENE.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        };
        let [x, y, w, h] = view.region;
        if x + w > f.render[0] || y + h > f.render[1] {
            return Err(format!(
                "region {:?} outside the render size {:?}",
                view.region, f.render
            ));
        }
        let off = [0, 1, 2].map(|k| view.cam[k] - scene.center[k]);
        let dist = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
        if !dist.is_finite() || dist > scene.radius {
            if !CAMERA_LOGGED.swap(true, Ordering::Relaxed) {
                log::line(&format!(
                    "rt_reflect: camera {:?} (view matrix) is {dist:.1} m from the TLAS centre {:?} (radius {}): skipped",
                    view.cam, scene.center, scene.radius
                ));
            }
            NO_SCENE.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let Some(shared) = st.shared.as_ref() else {
            return Ok(false);
        };
        if shared.key.0 != [w, h] || shared.key.1 != identity(&base) {
            return Ok(false);
        }
        let inputs = Inputs {
            color: f.color,
            depth: f.depth,
            motion: f.motion,
            req0: &shared.req0.res,
            req1: &shared.req1.res,
            rtw: &shared.rtw.res,
        };
        let s = settings::get();
        let knobs = Knobs {
            debug: s.rt_reflect_debug,
            flags: F_SHADOW,
        };
        let sc = SceneRef {
            tlas_va: scene.tlas_va,
            center: scene.center,
            radius: scene.radius,
            prim_offsets: &scene.prim_offsets,
            geom_count: scene.geom_count,
            prims: &scene.prims,
            prim_count: scene.prim_count,
            sun_dir: scene.lighting.sun_dir,
            sun_irradiance: scene.lighting.sun_irradiance,
            ambient: scene.lighting.ambient,
            instance_types: scene.instance_types.as_ref(),
            instance_type_count: scene.instance_type_count,
        };
        if st
            .gpu
            .as_ref()
            .is_none_or(|g| identity(&g.device) != identity(&d12))
        {
            st.gpu = None;
            st.gpu = Some(Gpu12::new(&d12)?);
        }
        let g = st.gpu.as_mut().ok_or("no pipeline")?;
        // SAFETY: plain getter.
        let completed = unsafe { f.fence.GetCompletedValue() };
        // Photo accumulation (`set_photo`): a persistent sum on this device.
        let hs = half([w, h]);
        if st.photo_spp.is_some()
            && st
                .accum
                .as_ref()
                .is_none_or(|a| a.1 != hs || a.2 != identity(&d12))
        {
            st.accum = Some((
                texture(
                    &d12,
                    hs,
                    DXGI_FORMAT_R32G32B32A32_FLOAT,
                    D3D12_RESOURCE_STATE_COMMON,
                )?,
                hs,
                identity(&d12),
            ));
            st.photo_total = 0;
        }
        let mode = match (st.photo_spp, st.accum.as_ref()) {
            (Some(spp), Some(a)) => Mode::Photo {
                accum: &a.0,
                samples: spp,
                total: st.photo_total,
                composite: true,
            },
            _ => Mode::Live {
                budget: s.rt_ray_budget.clamp(1, 100),
            },
        };
        let recorded = g.record(
            f.list,
            completed,
            f.fence_next,
            f.ts_freq,
            &view,
            &inputs,
            &sc,
            &knobs,
            &mode,
            Box::new(scene.clone()),
        )?;
        if recorded && let Mode::Photo { samples, .. } = mode {
            st.photo_total = st.photo_total.saturating_add(samples.clamp(1, 1024));
        }
        Ok(recorded)
    })();
    match r {
        Ok(true) => {
            FRAMES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(false) => {}
        Err(e) => fail(e),
    }
}

/// The frame's input textures (all COMMON before and after our commands).
pub(crate) struct Inputs<'a> {
    pub color: &'a ID3D12Resource,
    pub depth: &'a ID3D12Resource,
    /// Engine motion vectors (RG16F, current - previous in viewport UV).
    pub motion: &'a ID3D12Resource,
    pub req0: &'a ID3D12Resource,
    pub req1: &'a ID3D12Resource,
    pub rtw: &'a ID3D12Resource,
}

/// What the kernels read of the scene (`rt::SceneView`, or the offline scene).
pub(crate) struct SceneRef<'a> {
    pub tlas_va: u64,
    pub center: [f32; 3],
    pub radius: f32,
    pub prim_offsets: &'a ID3D12Resource,
    pub geom_count: u32,
    pub prims: &'a ID3D12Resource,
    pub prim_count: u32,
    pub sun_dir: [f32; 3],
    pub sun_irradiance: [f32; 3],
    pub ambient: [f32; 3],
    /// u32 per TLAS instance index (`void_rt::dynamic::TYPE_*`), only for the `instances`
    /// debug view; None: every instance is static.
    pub instance_types: Option<&'a ID3D12Resource>,
    pub instance_type_count: u32,
}

pub(crate) struct Knobs {
    pub debug: u8,
    pub flags: u32,
}

/// What [`Gpu12::record`] traces.
pub(crate) enum Mode<'a> {
    /// One GGX ray per requested texel within `budget` percent of the half-resolution grid,
    /// composited into the colour.
    Live { budget: u32 },
    /// Photo accumulation: `samples` GGX rays per requested texel (no budget, no list) added
    /// as (sum L c, sum c) into `accum` (half-resolution RGBA32F, UAV-capable, COMMON before
    /// and after); `total` = samples accumulated before this call (0 clears it; also the
    /// sample-sequence index, i.e. the seed); `composite` blends the running average of
    /// `total + samples` samples into the colour.
    Photo {
        accum: &'a ID3D12Resource,
        samples: u32,
        total: u32,
        composite: bool,
    },
}

/// Dword of `Consts::max_rays` (photo: sample count for the composite).
const MAX_RAYS_DWORD: u32 = 26;

/// Root constants `Consts` of `rt_reflect.hlsl` (50 dwords).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Consts {
    cam: [f32; 3],
    radius: f32,
    rot: [[f32; 4]; 3],
    half_size: [u32; 2],
    region: [u32; 2],
    proj: [f32; 4],
    depth_mul: f32,
    depth_add: f32,
    max_rays: u32,
    flags: u32,
    sun_dir: [f32; 3],
    geom_count: u32,
    sun_irr: [f32; 3],
    prim_count: u32,
    ambient: [f32; 3],
    calib_init: f32,
    center: [f32; 3],
    frame: u32,
    origin: [u32; 2],
    debug: u32,
    jitter: f32,
    type_count: u32,
    firefly_k: f32,
}

const _: () = assert!(size_of::<Consts>() == 50 * 4);

/// One ring entry: what the list recorded with it keeps alive until its fence completed.
struct Set {
    fence: u64,
    timed: bool,
    keep: Vec<ID3D12Resource>,
    scene: Option<Box<dyn std::any::Any>>,
    /// The frame recorded with this set ran the denoiser.
    denoised: bool,
}

/// Size-dependent resources (half resolution): traced results and rays, the ray list, the
/// resolve output, the ping-pong accumulation and its metadata, the blurred output.
struct Sized {
    half: [u32; 2],
    result: ID3D12Resource,
    rays: ID3D12Resource,
    ray: ID3D12Resource,
    resolved: ID3D12Resource,
    hist: [ID3D12Resource; 2],
    meta: [ID3D12Resource; 2],
    blurred: ID3D12Resource,
    /// Second moment of the luminance (R32F ping-pong, firefly-aware history clamp).
    mom: [ID3D12Resource; 2],
}

/// Pipelines and persistent resources on one D3D12 device.
pub(crate) struct Gpu12 {
    device: ID3D12Device5,
    root: ID3D12RootSignature,
    pso: Vec<ID3D12PipelineState>,
    signature: ID3D12CommandSignature,
    heap: ID3D12DescriptorHeap,
    inc: u32,
    args: ID3D12Resource,
    calib: ID3D12Resource,
    queries: ID3D12QueryHeap,
    readback: ID3D12Resource,
    sized: Option<Sized>,
    /// The last denoised frame and the accumulation slot it wrote (history for the next).
    history: Option<(u32, usize)>,
    sets: Vec<Set>,
    next: usize,
}

fn heap(t: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: t,
        ..Default::default()
    }
}

fn buffer(
    dev: &ID3D12Device5,
    size: u64,
    t: D3D12_HEAP_TYPE,
    uav: bool,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut r = None;
    // SAFETY: valid descriptors and out pointer.
    unsafe {
        dev.CreateCommittedResource(&heap(t), D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
    }
    .map_err(|e| format!("buffer ({size} bytes): {e}"))?;
    r.ok_or_else(|| "no buffer".to_owned())
}

fn texture(
    dev: &ID3D12Device5,
    size: [u32; 2],
    format: DXGI_FORMAT,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size[0]),
        Height: size[1],
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut r = None;
    // SAFETY: valid descriptors and out pointer.
    unsafe {
        dev.CreateCommittedResource(
            &heap(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut r,
        )
    }
    .map_err(|e| format!("texture {}x{}: {e}", size[0], size[1]))?;
    r.ok_or_else(|| "no texture".to_owned())
}

/// A transition barrier holding its own reference (released by [`release`]).
fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(r.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// Records `barriers` and releases the references transition barriers hold.
///
/// # Safety
/// `list` is open; the states are the resources' current states.
unsafe fn barriers(list: &ID3D12GraphicsCommandList, barriers: Vec<D3D12_RESOURCE_BARRIER>) {
    // SAFETY: caller's contract.
    unsafe { list.ResourceBarrier(&barriers) };
    for b in barriers {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
            // SAFETY: the union holds the Transition variant for this type.
            let t = std::mem::ManuallyDrop::into_inner(unsafe { b.Anonymous.Transition });
            drop(std::mem::ManuallyDrop::into_inner(t.pResource));
        }
    }
}

impl Gpu12 {
    pub(crate) fn new(dev: &ID3D12Device5) -> Result<Self, String> {
        let code: Vec<&[u8]> = KERNELS.iter().map(|k| dxil(k)).collect::<Result<_, _>>()?;
        // SAFETY (whole block): object creation with valid descriptors, DXIL that embeds the
        // root signature, and out pointers.
        unsafe {
            let root: ID3D12RootSignature = dev
                .CreateRootSignature(0, code[0])
                .map_err(|e| format!("root signature: {e}"))?;
            let mut pso = Vec::new();
            for (k, c) in KERNELS.iter().zip(&code) {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: c.as_ptr().cast(),
                        BytecodeLength: c.len(),
                    },
                    ..Default::default()
                };
                let p: Result<ID3D12PipelineState, _> = dev.CreateComputePipelineState(&desc);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                pso.push(p.map_err(|e| format!("{k}: PSO: {e}"))?);
            }
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            };
            let mut signature: Option<ID3D12CommandSignature> = None;
            dev.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &arg,
                    NodeMask: 0,
                },
                None,
                &mut signature,
            )
            .map_err(|e| format!("command signature: {e}"))?;
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: SETS as u32 * (SRVS + UAVS),
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .map_err(|e| format!("descriptor heap: {e}"))?;
            let inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let mut queries: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2 * SETS as u32,
                    NodeMask: 0,
                },
                &mut queries,
            )
            .map_err(|e| format!("query heap: {e}"))?;
            let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            Ok(Self {
                device: dev.clone(),
                root,
                pso,
                signature: signature.ok_or("no command signature")?,
                heap,
                inc,
                args: buffer(dev, 16, D3D12_HEAP_TYPE_DEFAULT, true, uav)?,
                calib: buffer(dev, CALIB_BYTES, D3D12_HEAP_TYPE_DEFAULT, true, uav)?,
                queries: queries.ok_or("no query heap")?,
                readback: buffer(
                    dev,
                    SETS as u64 * (16 + CALIB_HEADER),
                    D3D12_HEAP_TYPE_READBACK,
                    false,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )?,
                sized: None,
                history: None,
                sets: (0..SETS)
                    .map(|_| Set {
                        fence: 0,
                        timed: false,
                        keep: Vec::new(),
                        scene: None,
                        denoised: false,
                    })
                    .collect(),
                next: 0,
            })
        }
    }

    /// Reads the timestamps and counters set `i` resolved (its list completed).
    fn read_back(&mut self, i: usize, ts_freq: u64) {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let total = SETS * (16 + CALIB_HEADER as usize);
        // SAFETY: mapping our readback buffer, written by a completed list; ranges in bounds.
        unsafe {
            if self
                .readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: total,
                    }),
                    Some(&mut p),
                )
                .is_err()
                || p.is_null()
            {
                return;
            }
            let b = p.cast::<u8>();
            let ts = std::slice::from_raw_parts(b.add(16 * i).cast::<u64>(), 2);
            let words = std::slice::from_raw_parts(
                b.add(SETS * 16 + i * CALIB_HEADER as usize).cast::<u32>(),
                12,
            );
            let (a, z) = (ts[0], ts[1]);
            let stats = Stats {
                scale: f32::from_bits(words[0]),
                requested: words[4],
                traced: words[5],
                hits: words[6],
                screen_hits: words[7],
                misses: words[8],
                tile_scale: [f32::from_bits(words[9]), f32::from_bits(words[10])],
                calib_samples: words[11],
                denoised: self.sets[i].denoised,
            };
            self.readback
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            *lock(&LAST) = stats;
            if z > a && ts_freq > 0 {
                let ms = ((z - a) as f64 * 1000.0 / ts_freq as f64) as f32;
                GPU_LAST_MS.store(ms.to_bits(), Ordering::Relaxed);
                let avg = f32::from_bits(GPU_MS.load(Ordering::Relaxed));
                let avg = if avg > 0.0 {
                    avg * 0.95 + ms * 0.05
                } else {
                    ms
                };
                GPU_MS.store(avg.to_bits(), Ordering::Relaxed);
            }
        }
    }

    /// Records one frame into `list` (open). `completed`: the fence value the GPU reached;
    /// `next`: the value signalled after `list`. Returns false (nothing recorded) when the ring
    /// slot is still in flight. Everything fallible happens before the first command.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        completed: u64,
        next: u64,
        ts_freq: u64,
        view: &View,
        inp: &Inputs,
        scene: &SceneRef,
        knobs: &Knobs,
        mode: &Mode,
        keep: Box<dyn std::any::Any>,
    ) -> Result<bool, String> {
        let i = self.next;
        if self.sets[i].fence > completed {
            BUSY.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        if self.sets[i].timed {
            self.read_back(i, ts_freq);
            self.sets[i].timed = false;
        }
        let [x, y, w, h] = view.region;
        let hs = half([w, h]);
        let mut retired = Vec::new();
        if self.sized.as_ref().is_none_or(|s| s.half != hs) {
            if let Some(old) = self.sized.take() {
                retired.extend([old.result, old.rays, old.ray, old.resolved, old.blurred]);
                retired.extend(old.hist);
                retired.extend(old.meta);
                retired.extend(old.mom);
            }
            self.history = None;
            let tex = || {
                texture(
                    &self.device,
                    hs,
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            };
            let mom = || {
                texture(
                    &self.device,
                    hs,
                    DXGI_FORMAT_R32_FLOAT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            };
            self.sized = Some(Sized {
                half: hs,
                result: tex()?,
                rays: buffer(
                    &self.device,
                    u64::from(hs[0]) * u64::from(hs[1]) * 4,
                    D3D12_HEAP_TYPE_DEFAULT,
                    true,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )?,
                ray: tex()?,
                resolved: tex()?,
                hist: [tex()?, tex()?],
                meta: [tex()?, tex()?],
                blurred: tex()?,
                mom: [mom()?, mom()?],
            });
        }
        let sized = self.sized.as_ref().ok_or("no sized resources")?;
        let cap = hs[0] * hs[1];
        // Denoiser (live frames with the SSR temporal stage): history from the previous frame
        // if that was the denoised frame before this one; the other ping-pong slot is written.
        let live = matches!(mode, Mode::Live { .. });
        let denoise =
            live && view.temporal && knobs.flags & F_LEGACY == 0 && knobs.debug != DEBUG_INSTANCES;
        let valid = denoise
            && self
                .history
                .is_some_and(|(f, _)| f.wrapping_add(1) == view.frame);
        let cur = self.history.map_or(0, |(_, s)| s ^ 1);
        // Descriptors of set i: t3-t7, u0-u11.
        // SAFETY (whole block): views of live resources into our heap's slots of set i,
        // which no list in flight uses (its fence completed above).
        let (srv_gpu, uav_gpu) = unsafe {
            let cpu0 = self.heap.GetCPUDescriptorHandleForHeapStart();
            let gpu0 = self.heap.GetGPUDescriptorHandleForHeapStart();
            let first = (i as u32 * (SRVS + UAVS)) as usize;
            let cpu = |k: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu0.ptr + (first + k) * self.inc as usize,
            };
            let tex_srv = |r: &ID3D12Resource, f: DXGI_FORMAT, k: usize| {
                let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: f,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                self.device.CreateShaderResourceView(r, Some(&d), cpu(k));
            };
            tex_srv(inp.req0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0);
            tex_srv(inp.req1, DXGI_FORMAT_R16G16B16A16_FLOAT, 1);
            tex_srv(inp.rtw, DXGI_FORMAT_R16G16B16A16_FLOAT, 2);
            tex_srv(inp.depth, DXGI_FORMAT_R32_FLOAT, 3);
            tex_srv(inp.motion, DXGI_FORMAT_R16G16_FLOAT, 4);
            let tex_uav = |r: &ID3D12Resource, f: DXGI_FORMAT, k: usize| {
                let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: f,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_UAV::default(),
                    },
                };
                self.device
                    .CreateUnorderedAccessView(r, None, Some(&d), cpu(k));
            };
            let buf_uav = |r: &ID3D12Resource, elements: u32, stride: u32, k: usize| {
                let raw = stride == 0;
                let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: if raw {
                        DXGI_FORMAT_R32_TYPELESS
                    } else {
                        DXGI_FORMAT_UNKNOWN
                    },
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: elements,
                            StructureByteStride: stride,
                            CounterOffsetInBytes: 0,
                            Flags: if raw {
                                D3D12_BUFFER_UAV_FLAG_RAW
                            } else {
                                D3D12_BUFFER_UAV_FLAG_NONE
                            },
                        },
                    },
                };
                self.device
                    .CreateUnorderedAccessView(r, None, Some(&d), cpu(k));
            };
            let rgba16 = DXGI_FORMAT_R16G16B16A16_FLOAT;
            match mode {
                Mode::Live { .. } => tex_uav(&sized.result, rgba16, 5),
                Mode::Photo { accum, .. } => tex_uav(accum, DXGI_FORMAT_R32G32B32A32_FLOAT, 5),
            }
            buf_uav(&sized.rays, cap, 4, 6);
            buf_uav(&self.args, 4, 0, 7);
            tex_uav(inp.color, rgba16, 8);
            buf_uav(&self.calib, (CALIB_BYTES / 4) as u32, 0, 9);
            tex_uav(&sized.ray, rgba16, 10);
            tex_uav(&sized.resolved, rgba16, 11);
            tex_uav(&sized.hist[cur], rgba16, 12);
            tex_uav(&sized.meta[cur], rgba16, 13);
            tex_uav(&sized.hist[cur ^ 1], rgba16, 14);
            tex_uav(&sized.meta[cur ^ 1], rgba16, 15);
            tex_uav(&sized.mom[cur], DXGI_FORMAT_R32_FLOAT, 17);
            tex_uav(&sized.mom[cur ^ 1], DXGI_FORMAT_R32_FLOAT, 18);
            // What the composite reads: the blurred accumulation, this frame's rays, or the
            // photo sum.
            match mode {
                Mode::Live { .. } if denoise => tex_uav(&sized.blurred, rgba16, 16),
                Mode::Live { .. } => tex_uav(&sized.result, rgba16, 16),
                Mode::Photo { accum, .. } => tex_uav(accum, DXGI_FORMAT_R32G32B32A32_FLOAT, 16),
            }
            let at = |k: usize| D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu0.ptr + ((first + k) * self.inc as usize) as u64,
            };
            (at(0), at(SRVS as usize))
        };
        let knob_flags =
            knobs.flags | if denoise { F_DENOISE } else { 0 } | if valid { F_HISTORY } else { 0 };
        let (max_rays, flags, frame, samples) = match mode {
            Mode::Live { budget } => (
                ((u64::from(cap) * u64::from((*budget).clamp(1, 100))).div_ceil(100)) as u32,
                knob_flags,
                view.frame,
                0,
            ),
            Mode::Photo { samples, total, .. } => {
                let spp = (*samples).clamp(1, 1024);
                let clear = if *total == 0 { F_CLEAR } else { 0 };
                (spp, knob_flags | F_PHOTO | clear, *total, spp)
            }
        };
        let c = Consts {
            cam: view.cam,
            radius: scene.radius,
            rot: view.rot,
            half_size: hs,
            region: [w, h],
            proj: view.proj,
            depth_mul: view.depth[0],
            depth_add: view.depth[1],
            max_rays,
            flags,
            sun_dir: scene.sun_dir,
            geom_count: scene.geom_count,
            sun_irr: scene.sun_irradiance,
            prim_count: scene.prim_count,
            ambient: scene.ambient,
            calib_init: INITIAL_SCALE,
            center: scene.center,
            frame,
            origin: [x, y],
            debug: u32::from(knobs.debug),
            jitter: (view.frame as f32 * 0.618_034).fract(),
            type_count: scene
                .instance_types
                .map_or(0, |_| scene.instance_type_count),
            firefly_k: firefly_k(),
        };
        let set = &mut self.sets[i];
        set.fence = next;
        set.timed = true;
        set.denoised = denoise;
        set.keep = vec![
            inp.color.clone(),
            inp.depth.clone(),
            inp.motion.clone(),
            inp.req0.clone(),
            inp.req1.clone(),
            inp.rtw.clone(),
        ];
        if let Mode::Photo { accum, .. } = mode {
            set.keep.push((*accum).clone());
        }
        set.keep.extend(retired);
        set.keep.extend(scene.instance_types.cloned());
        set.scene = Some(keep);
        self.next = (i + 1) % SETS;
        self.history = denoise.then_some((view.frame, cur));
        let common = D3D12_RESOURCE_STATE_COMMON;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let groups = |n: u32| n.div_ceil(8);
        // SAFETY (whole block): recording into the caller's open list with live objects; the
        // inputs are COMMON before and after (FSR's contract), ours keep their states.
        unsafe {
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root);
            list.SetComputeRoot32BitConstants(0, 50, (&raw const c).cast(), 0);
            list.SetComputeRootShaderResourceView(1, scene.tlas_va);
            list.SetComputeRootShaderResourceView(2, scene.prim_offsets.GetGPUVirtualAddress());
            list.SetComputeRootShaderResourceView(3, scene.prims.GetGPUVirtualAddress());
            // Without instance types any valid buffer (never read: type_count 0).
            list.SetComputeRootShaderResourceView(
                6,
                scene
                    .instance_types
                    .unwrap_or(scene.prim_offsets)
                    .GetGPUVirtualAddress(),
            );
            list.SetComputeRootDescriptorTable(4, srv_gpu);
            list.SetComputeRootDescriptorTable(5, uav_gpu);
            let mut first = vec![
                transition(inp.color, common, uav),
                transition(inp.depth, common, read),
                transition(inp.motion, common, read),
                transition(inp.req0, common, read),
                transition(inp.req1, common, read),
                transition(inp.rtw, common, read),
            ];
            if let Mode::Photo { accum, .. } = mode {
                first.push(transition(accum, common, uav));
            }
            barriers(list, first);
            list.EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i as u32);
            if let Mode::Photo { .. } = mode {
                list.SetPipelineState(&self.pso[5]);
                list.Dispatch(groups(hs[0]), groups(hs[1]), 1);
                barriers(list, vec![uav_barrier()]);
            } else {
                list.SetPipelineState(&self.pso[0]);
                list.Dispatch(1, 1, 1);
                barriers(list, vec![uav_barrier()]);
                list.SetPipelineState(&self.pso[1]);
                list.Dispatch(groups(hs[0]), groups(hs[1]), 1);
                barriers(list, vec![uav_barrier()]);
                list.SetPipelineState(&self.pso[2]);
                list.Dispatch(1, 1, 1);
                barriers(
                    list,
                    vec![
                        uav_barrier(),
                        transition(&self.args, uav, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
                    ],
                );
                list.SetPipelineState(&self.pso[3]);
                list.ExecuteIndirect(&self.signature, 1, &self.args, 0, None, 0);
                barriers(
                    list,
                    vec![
                        uav_barrier(),
                        transition(&self.args, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, uav),
                    ],
                );
                if denoise {
                    // Resolve, temporal, blur (half resolution).
                    for k in [6, 7, 8] {
                        list.SetPipelineState(&self.pso[k]);
                        list.Dispatch(groups(hs[0]), groups(hs[1]), 1);
                        barriers(list, vec![uav_barrier()]);
                    }
                }
            }
            if let Mode::Photo {
                total, composite, ..
            } = mode
            {
                // Running average: the composite divides the sums by all samples so far.
                list.SetComputeRoot32BitConstant(0, total + samples, MAX_RAYS_DWORD);
                if *composite {
                    list.SetPipelineState(&self.pso[4]);
                    list.Dispatch(groups(w), groups(h), 1);
                }
            } else {
                list.SetPipelineState(&self.pso[4]);
                list.Dispatch(groups(w), groups(h), 1);
            }
            list.EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i as u32 + 1);
            let mut last = vec![
                transition(inp.color, uav, common),
                transition(inp.depth, read, common),
                transition(inp.motion, read, common),
                transition(inp.req0, read, common),
                transition(inp.req1, read, common),
                transition(inp.rtw, read, common),
                transition(&self.calib, uav, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ];
            if let Mode::Photo { accum, .. } = mode {
                last.push(transition(accum, uav, common));
            }
            barriers(list, last);
            list.CopyBufferRegion(
                &self.readback,
                SETS as u64 * 16 + i as u64 * CALIB_HEADER,
                &self.calib,
                0,
                CALIB_HEADER,
            );
            barriers(
                list,
                vec![transition(
                    &self.calib,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    uav,
                )],
            );
            list.ResolveQueryData(
                &self.queries,
                D3D12_QUERY_TYPE_TIMESTAMP,
                2 * i as u32,
                2,
                &self.readback,
                16 * i as u64,
            );
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Everything present (ray-traced reflections active).
    fn ready() -> Prereq {
        Prereq {
            rt_reflections: true,
            ssr: true,
            rt: true,
            fsr: true,
            fsr_running: true,
            device: Some(true),
            scene_phase: "ready".into(),
            ..Prereq::default()
        }
    }

    fn text(p: &Prereq) -> Option<String> {
        blocker_for(p).map(|b| b.text)
    }

    #[test]
    fn reasons_in_every_state() {
        let t = |f: &dyn Fn(&mut Prereq)| {
            let mut p = ready();
            f(&mut p);
            text(&p)
        };
        assert_eq!(text(&ready()), None);
        assert_eq!(t(&|p| p.rt_reflections = false), None);
        assert_eq!(
            t(&|p| p.ssr = false).as_deref(),
            Some("Reflections on (ssr = 1)")
        );
        // The first missing setting wins, in the order the user fixes them.
        assert_eq!(
            t(&|p| {
                p.ssr = false;
                p.rt = false;
                p.fsr = false;
            })
            .as_deref(),
            Some("Reflections on (ssr = 1)")
        );
        assert_eq!(
            t(&|p| p.rt = false).as_deref(),
            Some("the ray-tracing scene (rt = 1, Photo tab)")
        );
        assert!(t(&|p| p.fsr = false).unwrap().starts_with("FSR on"));
        assert_eq!(
            t(&|p| p.fsr_running = false).as_deref(),
            Some("FSR running (not started yet)")
        );
        assert_eq!(
            t(&|p| p.device = None).as_deref(),
            Some("the D3D12 device (not created yet)")
        );
        assert_eq!(
            t(&|p| p.device = Some(false)).as_deref(),
            Some("a GPU with DXR 1.1 inline ray tracing")
        );
        assert_eq!(
            t(&|p| p.scene_phase = String::new()).as_deref(),
            Some("the scene (not built yet)")
        );
        assert_eq!(
            t(&|p| p.scene_error = Some("x".into())).as_deref(),
            Some("the scene (failed: x)")
        );
        assert!(
            t(&|p| p.camera_mismatch = true)
                .unwrap()
                .starts_with("a camera")
        );
    }

    /// Steps `r` with `p` at `t0 + secs`; returns the toast.
    fn step(r: &mut Reporter, p: &Prereq, t0: Instant, secs: u64) -> Option<String> {
        r.step(
            p.rt_reflections,
            blocker_for(p),
            t0 + Duration::from_secs(secs),
        )
        .toast
    }

    #[test]
    fn toasts_only_where_the_user_did_not_choose_it() {
        let t0 = Instant::now();
        let mut r = Reporter::default();
        let mut p = ready();
        assert_eq!(step(&mut r, &p, t0, 0), None);
        // The user turns Reflections off: card only, no toast (the reported bug).
        p.ssr = false;
        assert_eq!(step(&mut r, &p, t0, 1), None);
        assert_eq!(step(&mut r, &p, t0, 30), None);
        // Back on while the scene rebuilds: transient, no toast within the grace.
        p.ssr = true;
        p.scene_phase = "building".into();
        assert_eq!(step(&mut r, &p, t0, 31), None);
        p.scene_phase = "uploading".into();
        assert_eq!(step(&mut r, &p, t0, 38), None);
        p.scene_phase = "ready".into();
        assert_eq!(step(&mut r, &p, t0, 39), None);
        // The user turns the scene (rt) off: card only.
        p.rt = false;
        assert_eq!(step(&mut r, &p, t0, 40), None);
        // Turning the feature off and on again with rt off: now it is worth a toast.
        p.rt_reflections = false;
        assert_eq!(step(&mut r, &p, t0, 41), None);
        p.rt_reflections = true;
        assert_eq!(
            step(&mut r, &p, t0, 42).as_deref(),
            Some("the ray-tracing scene (rt = 1, Photo tab)")
        );
        assert_eq!(step(&mut r, &p, t0, 43), None);
    }

    #[test]
    fn startup_with_settings_off_does_not_toast() {
        let t0 = Instant::now();
        let mut r = Reporter::default();
        let p = Prereq {
            ssr: false,
            ..ready()
        };
        assert_eq!(step(&mut r, &p, t0, 0), None);
    }

    #[test]
    fn transient_toasts_after_grace_definite_at_once() {
        let t0 = Instant::now();
        let mut r = Reporter::default();
        let mut p = ready();
        p.device = None;
        assert_eq!(step(&mut r, &p, t0, 0), None);
        assert_eq!(step(&mut r, &p, t0, 9), None);
        assert_eq!(
            step(&mut r, &p, t0, 11).as_deref(),
            Some("the D3D12 device (not created yet)")
        );
        assert_eq!(step(&mut r, &p, t0, 20), None);
        p.device = Some(true);
        p.scene_error = Some("out of memory".into());
        assert_eq!(
            step(&mut r, &p, t0, 21).as_deref(),
            Some("the scene (failed: out of memory)")
        );
    }
}
