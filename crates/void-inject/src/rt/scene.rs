//! The map's ray-tracing scene and the per-frame view of it (docs/raytracing.md 11.3).
//!
//! One worker thread (`void-rt`) owns the [`Gpu`] (`void_rt::ingame`, on our own queue of
//! the shared device): it builds the map's [`GpuScene`] (CPU decode of the archives plus
//! the entities/lights/sun/sky the main thread copied from game memory, then BLAS/TLAS,
//! tables and textures in short blocking submits), keeps a TLAS of the instances within
//! `rt_radius` of the camera for live ray tracing, and runs the photo mode's [`Task`]
//! (the same `Gpu` must serve the upload and the photo session). The main thread only posts
//! requests ([`tick`], [`request_build`], [`set_task`]) and reads published state.
//!
//! Lifetime: a changed map drops the scene; `rt = 0` without a task stops the worker and
//! frees everything. Resources handed out in a [`SceneView`] stay alive until the last clone
//! is dropped. GPU errors (device removal included) stop the worker and are reported in
//! [`status`]; the game's D3D11 device is not affected.
//!
//! Dynamic objects (`rt_dynamic` / `rt_characters`, docs/raytracing.md section 14): the
//! scene is then built with a reserve; the main thread's tracker (`track.rs`) sends what
//! moved, appeared or posed, the worker applies it (`dynamic.rs`) and refits the cut-off
//! TLAS into a spare one (or rebuilds it when instances appeared / disappeared), so a
//! published view never changes under its holder.

use std::any::Any;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use void_rt::ingame::{
    Buffer, DynamicReserve, Gpu, GpuScene, InstanceSet, SceneOptions, SceneSource, SceneTlas, Tlas,
};
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Resource};
use windows::core::{IUnknown, Interface};

use super::device;
use super::dynamic::{DynSettings, DynState, DynStatus};
use super::live::ModelKey;
use super::track::{self, Event, TrackStatus, Tracker};
use crate::{log, notify, settings};

/// Sun and sky of the current view in engine ABSOLUTE units (engine stored hdrBuffer ≈
/// absolute x 3.58 in bright daylight; the exact factor is exposure-dependent, `fsr.rs`
/// `EngineLuminanceFactor`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lighting {
    /// World space (Z up), unit, towards the sun.
    pub sun_dir: [f32; 3],
    /// Irradiance normal to the sun: `sunColor` x `void_rt::lights::SUN_COLOR_SCALE`.
    pub sun_irradiance: [f32; 3],
    /// `env/indirectLighting` sky zenith / horizon colours and ambient.
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub ambient: [f32; 3],
    /// Sun and environment parameters were found in the view.
    pub valid: bool,
}

/// What live ray tracing needs for one frame. Hold it (clone into the frame slot) until
/// the GPU work that uses its resources has completed: the resources are released when the
/// last clone is dropped.
#[derive(Clone)]
pub struct SceneView {
    /// Changes with every scene (map) build.
    pub generation: u64,
    /// Changes with every cut-off TLAS rebuild.
    pub tlas_id: u64,
    /// `RaytracingAccelerationStructure` (root SRV).
    pub tlas_va: u64,
    /// Camera position the TLAS was built around (world metres, Z up) and its radius.
    pub center: [f32; 3],
    pub radius: f32,
    pub instance_count: u32,
    /// u32 first primitive per global geometry: index `InstanceID() + GeometryIndex()`
    /// (InstanceID 24 bits; InstanceMask 1 = world, 2 = entity).
    pub prim_offsets: ID3D12Resource,
    pub geom_count: u32,
    /// Far Cry 6 style 24 bytes (`uint[6]`) per triangle, 2 words per vertex: albedo.rgb
    /// unorm8 | smoothness8 (red in the low byte), oct-normal.xy unorm8 | metallic8 |
    /// reflectance8; object-space normal (use `ObjectToWorld3x4`).
    pub prims: ID3D12Resource,
    pub prim_count: u32,
    pub lighting: Lighting,
    /// `hook::observe` frame at which `lighting` was read.
    pub frame: u64,
    /// With dynamic objects (`rt_dynamic` / `rt_characters`): u32 per TLAS instance index
    /// (`CommittedInstanceIndex()`, clamp to `instance_type_count - 1`): 0 static, 1 moved
    /// static prop, 2 spawned prop, 3 character (`void_rt::dynamic::TYPE_*`). Upload heap,
    /// read-only, valid while the view is held. None: every instance is static.
    pub instance_types: Option<ID3D12Resource>,
    pub instance_type_count: u32,
    #[allow(dead_code)]
    keep: Arc<dyn Any + Send + Sync>,
}

// SAFETY: D3D12 resources are free-threaded; `keep` is Send + Sync.
unsafe impl Send for SceneView {}
// SAFETY: as above (read-only use).
unsafe impl Sync for SceneView {}

/// A built scene.
pub(crate) struct Built {
    pub map: String,
    pub scene: Arc<GpuScene>,
    /// Built with the FC6 primitive table (live ray tracing).
    pub spike: bool,
    pub texture_size: u32,
    pub load_s: f64,
    pub upload_s: f64,
    /// Built with the dynamic reserve (`rt_dynamic` / `rt_characters`).
    pub dynamic: bool,
    /// Engine model of every snapshot entity, and every model pointer at the snapshot.
    pub keys: Vec<ModelKey>,
    pub present: Vec<u64>,
}

/// What to build.
#[derive(Clone)]
pub(crate) struct Request {
    pub source: SceneSource,
    pub opts: SceneOptions,
    pub why: &'static str,
    pub keys: Vec<ModelKey>,
    pub present: Vec<u64>,
}

/// Work the RT worker runs every tick with its GPU (the photo mode).
pub(crate) trait Task: Send {
    /// One step; `scene` is the current scene (any map). Ok(false) = finished.
    fn step(&mut self, gpu: &mut Gpu, scene: Option<&Arc<Built>>) -> Result<bool, String>;
    /// The worker is about to replace the scene: drop everything holding the old one (its
    /// video memory is freed before the new scene is uploaded).
    fn release_scene(&mut self);
    /// The worker gives up on the task (error, device lost, worker stopping).
    fn abort(&mut self, why: &str);
    /// The live cut-off TLAS stays published (and is rebuilt when the camera moves) while
    /// the task runs; false by default: the photo path tracer has the GPU to itself.
    fn allows_live(&self) -> bool {
        false
    }
}

/// A cut-off TLAS with the scene it refers to, the camera position / radius it was built
/// for, and with dynamic objects the instance set it was built or refitted from (which
/// keeps the dynamic BLASes alive).
struct TlasHold {
    tlas: Tlas,
    built: Arc<Built>,
    center: [f32; 3],
    radius: f32,
    set: Option<InstanceSet>,
    /// Refits since the last full build.
    refits: u32,
    /// Instance types of `set` (upload heap, `capacity` u32), written with each rebuild /
    /// refit while nobody holds this TLAS.
    types: Option<Buffer>,
}

/// Live cut-off TLASes allocated (spare, current or held by frames) and the most that are
/// created; beyond it a rebuild waits for a frame to release one.
static TLAS_HOLDS: AtomicUsize = AtomicUsize::new(0);
const MAX_HOLDS: usize = 3;

impl TlasHold {
    fn new(
        tlas: Tlas,
        built: Arc<Built>,
        center: [f32; 3],
        radius: f32,
        set: Option<InstanceSet>,
    ) -> Self {
        TLAS_HOLDS.fetch_add(1, Ordering::Relaxed);
        Self {
            tlas,
            built,
            center,
            radius,
            set,
            refits: 0,
            types: None,
        }
    }
}

impl Drop for TlasHold {
    fn drop(&mut self) {
        TLAS_HOLDS.fetch_sub(1, Ordering::Relaxed);
    }
}

// SAFETY: the TLAS's D3D12 buffers are free-threaded; after its build it is only read.
unsafe impl Send for TlasHold {}
// SAFETY: as above.
unsafe impl Sync for TlasHold {}

/// The published view without the per-frame lighting.
struct Published {
    generation: u64,
    tlas_id: u64,
    tlas_va: u64,
    center: [f32; 3],
    radius: f32,
    instance_count: u32,
    instance_types: Option<ID3D12Resource>,
    instance_type_count: u32,
    prim_offsets: ID3D12Resource,
    geom_count: u32,
    prims: ID3D12Resource,
    prim_count: u32,
    keep: Arc<TlasHold>,
}

// SAFETY: as `SceneView`.
unsafe impl Send for Published {}
// SAFETY: as above.
unsafe impl Sync for Published {}

/// Main thread -> worker.
#[derive(Default)]
struct Ctrl {
    request: Option<Request>,
    task: Option<Box<dyn Task>>,
    /// The task was cancelled while the worker held it.
    cancel_task: bool,
    map: String,
    camera: Option<[f32; 3]>,
    live: bool,
    /// Keep the worker and the map's scene for further photos (`photo = 1`).
    keep: bool,
    radius: f32,
    running: bool,
    /// Dynamic objects: settings (None = off) and the tracker's events not applied yet.
    dynamic: Option<DynSettings>,
    events: Vec<Event>,
    events_dropped: u64,
}

static CTRL: Mutex<Ctrl> = Mutex::new(Ctrl {
    request: None,
    task: None,
    cancel_task: false,
    map: String::new(),
    camera: None,
    live: false,
    keep: false,
    radius: 40.0,
    running: false,
    dynamic: None,
    events: Vec::new(),
    events_dropped: 0,
});
/// Events queued for the worker at most (a paused worker, e.g. during a photo, must not
/// grow the queue without bound); older ones are dropped first.
const MAX_EVENTS: usize = 8192;
/// Refits of a cut-off TLAS before it is rebuilt (refit quality decays with motion).
const MAX_REFITS: u32 = 60;
/// The main thread's dynamic-object tracker (None until dynamics run).
static TRACKER: Mutex<Option<Tracker>> = Mutex::new(None);
/// The photo task's frozen TLAS with dynamic objects (scene generation, TLAS).
static PHOTO_TLAS: ArcSwapOption<(u64, Arc<SceneTlas>)> = ArcSwapOption::const_empty();
static WAKE: Condvar = Condvar::new();
static VIEW: ArcSwapOption<Published> = ArcSwapOption::const_empty();
static LIGHTING: Mutex<(Lighting, u64)> = Mutex::new((
    Lighting {
        sun_dir: [0.0, 0.0, 1.0],
        sun_irradiance: [0.0; 3],
        sky_zenith: [0.0; 3],
        sky_horizon: [0.0; 3],
        ambient: [0.0; 3],
        valid: false,
    },
    0,
));
/// `IUnknown` identity of our D3D12 device (0 = none).
static DEVICE_ID: AtomicUsize = AtomicUsize::new(0);
/// Live ray tracing asked for the scene (`rt_build = use`).
static WANTED: AtomicBool = AtomicBool::new(false);
/// The worker hit a GPU error; no new worker for this session.
static GPU_LOST: AtomicBool = AtomicBool::new(false);
static TLAS_ID: AtomicU64 = AtomicU64::new(0);
static STATUS: Mutex<Status> = Mutex::new(Status {
    worker: false,
    phase: String::new(),
    built_map: String::new(),
    load_s: 0.0,
    upload_s: 0.0,
    instances: 0,
    spike: false,
    tlas_instances: 0,
    tlas_rebuilds: 0,
    tlas_ms: 0.0,
    error: None,
    dynamic: None,
});

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// State for the overlay.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub worker: bool,
    /// idle / building <map> / ready / failed.
    pub phase: String,
    pub built_map: String,
    pub load_s: f64,
    pub upload_s: f64,
    pub instances: usize,
    /// Built with the FC6 primitive table.
    pub spike: bool,
    pub tlas_instances: usize,
    pub tlas_rebuilds: u64,
    pub tlas_ms: f64,
    pub error: Option<String>,
    /// Dynamic objects (None while off).
    pub dynamic: Option<DynamicStatus>,
}

/// Dynamic objects for the overlay: the main thread's tracker and the worker.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DynamicStatus {
    pub track: TrackStatus,
    pub worker: DynStatus,
    pub events_dropped: u64,
}

pub fn status() -> Status {
    lock(&STATUS).clone()
}

/// The worker hit a GPU error this session (no new worker is started).
pub(crate) fn gpu_lost() -> bool {
    GPU_LOST.load(Ordering::Relaxed)
}

fn set_phase(p: impl Into<String>) {
    lock(&STATUS).phase = p.into();
}

/// The scene for this frame on `device` (non-blocking, any thread); None while `rt = 0`,
/// while building, after a map change, or for another device.
pub fn frame_scene(device: &ID3D12Device) -> Option<SceneView> {
    if !settings::get().rt {
        return None;
    }
    let ours = DEVICE_ID.load(Ordering::Acquire);
    if ours == 0 || identity(device) != Some(ours) {
        return None;
    }
    let Some(p) = VIEW.load_full() else {
        WANTED.store(true, Ordering::Relaxed);
        return None;
    };
    let (lighting, frame) = *lock(&LIGHTING);
    Some(SceneView {
        generation: p.generation,
        tlas_id: p.tlas_id,
        tlas_va: p.tlas_va,
        center: p.center,
        radius: p.radius,
        instance_count: p.instance_count,
        instance_types: p.instance_types.clone(),
        instance_type_count: p.instance_type_count,
        prim_offsets: p.prim_offsets.clone(),
        geom_count: p.geom_count,
        prims: p.prims.clone(),
        prim_count: p.prim_count,
        lighting,
        frame,
        keep: p.keep.clone(),
    })
}

/// Sun and sky of the latest view (read every 30 frames while `rt = 1`).
pub fn lighting() -> Option<Lighting> {
    let (l, frame) = *lock(&LIGHTING);
    (frame != 0).then_some(l)
}

fn identity<T: Interface>(i: &T) -> Option<usize> {
    i.cast::<IUnknown>().ok().map(|u| u.as_raw() as usize)
}

/// Asks the worker to build the scene for `req` (replaces an older request).
pub(crate) fn request_build(req: Request) {
    let mut c = lock(&CTRL);
    c.request = Some(req);
    ensure_worker(&mut c);
    WAKE.notify_all();
}

/// Installs (or with None cancels) the worker's task.
pub(crate) fn set_task(task: Option<Box<dyn Task>>) {
    let mut c = lock(&CTRL);
    match task {
        Some(t) => {
            if let Some(mut old) = c.task.replace(t) {
                old.abort("replaced");
            }
            c.cancel_task = false;
            ensure_worker(&mut c);
        }
        None => {
            if let Some(mut old) = c.task.take() {
                old.abort("cancelled");
            }
            // The worker may be holding it right now.
            c.cancel_task = true;
        }
    }
    WAKE.notify_all();
}

/// The scene the worker holds now, if it was built for `map`.
pub(crate) fn built_for(map: &str) -> Option<Arc<Built>> {
    BUILT
        .load_full()
        .filter(|b| b.map.eq_ignore_ascii_case(map))
}

/// The scene for `map` is built, being built, or its build request is queued (a new
/// snapshot would only repeat it).
pub(crate) fn built_or_pending(map: &str) -> bool {
    if built_for(map).is_some() {
        return true;
    }
    let queued = lock(&CTRL)
        .request
        .as_ref()
        .is_some_and(|r| r.source.map.eq_ignore_ascii_case(map));
    let st = status();
    queued
        || (st.worker
            && st
                .phase
                .strip_prefix("building ")
                .is_some_and(|m| m.eq_ignore_ascii_case(map)))
}

/// The photo task's frozen TLAS with the dynamic objects as they were at photo entry, for
/// `built`'s scene (None: no dynamic objects; trace the scene's full TLAS).
pub(crate) fn photo_tlas(built: &Built) -> Option<Arc<SceneTlas>> {
    PHOTO_TLAS
        .load_full()
        .filter(|p| p.0 == built.scene.generation())
        .map(|p| p.1.clone())
}

static BUILT: ArcSwapOption<Built> = ArcSwapOption::const_empty();

/// Sleep after each blocking submit of the worker (scene upload, TLAS builds): the game's
/// queue gets the GPU between our submits.
const SUBMIT_PAUSE: Duration = Duration::from_millis(2);

/// Video memory kept free for what follows a scene upload: a photo session at up to
/// 3840x2160 and two live cut-off TLASes (about 60 MiB each for eotw); with dynamic
/// objects also their meshes and BLAS copies (a budget for ~24 characters).
fn vram_reserve(dynamic: bool) -> u64 {
    void_rt::ingame::PhotoSession::buffer_bytes(3840, 2160)
        + (128 << 20)
        + if dynamic { 384 << 20 } else { 0 }
}

/// Dynamic objects are requested by the settings (the scene then needs the reserve).
fn dynamic_wanted(s: &settings::Settings) -> bool {
    s.rt && (s.rt_dynamic || s.rt_characters)
}

/// Scene options from the settings.
pub(crate) fn options(s: &settings::Settings) -> SceneOptions {
    let dynamic = dynamic_wanted(s);
    SceneOptions {
        texture_size: s.photo_texture_size,
        spike_primitives: s.rt,
        vram_reserve_bytes: vram_reserve(dynamic),
        dynamic: dynamic.then(DynamicReserve::default),
        ..SceneOptions::default()
    }
}

/// Copies what a scene build needs from game memory (main thread; the entity walk takes
/// tens of milliseconds, E044) into a build request.
pub(crate) fn snapshot_request(
    view: &super::live::View,
    s: &settings::Settings,
    why: &'static str,
) -> Request {
    let walk = super::live::entities(view.world);
    let n = walk.counts;
    let lights = super::live::lights(view.world);
    let sun = super::live::sun(view.rv);
    let env = super::live::env(view.rv);
    log::line(&format!(
        "rt: snapshot of '{}': {} of {} models kept (skinned {}, not drawn {}, editor {}, unreadable {}, first-person view model {}), {} lights, sun {}, sky {}",
        view.map,
        n.kept,
        n.models,
        n.skinned,
        n.not_drawn,
        n.editor,
        n.unreadable,
        n.first_person,
        lights.len(),
        if sun.is_some() { "yes" } else { "no" },
        if env.is_some() { "yes" } else { "no" },
    ));
    Request {
        source: SceneSource {
            game_dir: crate::game_dir().cloned().unwrap_or_else(PathBuf::new),
            map: view.map.clone(),
            entities: walk.entities,
            lights,
            sun,
            env,
        },
        opts: options(s),
        why,
        keys: walk.keys,
        present: walk.present,
    }
}

fn ensure_worker(c: &mut Ctrl) {
    if c.running || GPU_LOST.load(Ordering::Relaxed) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("void-rt".into())
        .spawn(|| {
            let r = std::panic::catch_unwind(worker);
            if r.is_err() {
                log::line("rt: worker panicked");
                fail("internal error (panic)".into());
            }
            let mut c = lock(&CTRL);
            c.running = false;
            if let Some(mut t) = c.task.take() {
                t.abort("the ray-tracing worker stopped");
            }
            lock(&STATUS).worker = false;
            VIEW.store(None);
            BUILT.store(None);
        });
    match spawned {
        Ok(_) => {
            c.running = true;
            lock(&STATUS).worker = true;
        }
        Err(e) => log::line(&format!("rt: worker thread: {e}")),
    }
}

fn fail(e: String) {
    let mut s = lock(&STATUS);
    s.phase = "failed".into();
    s.error = Some(e);
}

/// GPU errors after which the device must not be used again (device removal).
fn device_lost(e: &str) -> bool {
    let l = e.to_ascii_lowercase();
    l.contains("removed")
        || l.contains("887a0005")
        || l.contains("887a0006")
        || l.contains("u64::max")
}

// --------------------------------------------------------------------------- main thread --

struct Tick {
    world: u64,
    map: String,
    world_frame: u64,
    snapshot_done_for: String,
}

static TICK: Mutex<Tick> = Mutex::new(Tick {
    world: 0,
    map: String::new(),
    world_frame: 0,
    snapshot_done_for: String::new(),
});

/// Frames after a world change before a load-time build snapshots the scene.
const SETTLE_FRAMES: u64 = 60;

/// Main thread, every frame after `RenderAllViews` (`hook::observe`): tracks the map and
/// camera, reads the lighting, starts load-time / first-use builds and stops the worker
/// when nothing needs it. With `rt = 0` and no photo task: one atomic and one mutex check.
pub fn tick(requests: u64, frame: u64) {
    let s = settings::get();
    let mut c = lock(&CTRL);
    let busy = c.running || c.task.is_some();
    if !s.rt && !busy {
        return;
    }
    c.live = s.rt;
    c.keep = s.photo;
    c.radius = s.rt_radius;
    let mut t = lock(&TICK);
    if let Some(world) = super::live::world(requests)
        && world != t.world
    {
        t.world = world;
        t.map = super::live::map_name(world);
        t.world_frame = frame;
        c.map.clone_from(&t.map);
        log::line(&format!("rt: map '{}'", t.map));
    }
    if s.rt {
        if let Some(v) = super::live::view(requests) {
            c.camera = Some(v.org);
            if frame.is_multiple_of(30) {
                *lock(&LIGHTING) = (super::live::lighting(v.rv), frame.max(1));
            }
            let dynamic = dynamic_wanted(&s);
            let wanted = s.rt_build_at_load || WANTED.load(Ordering::Relaxed);
            let built = BUILT
                .load_full()
                .filter(|b| b.map.eq_ignore_ascii_case(&t.map) && b.spike);
            let have = built.as_ref().is_some_and(|b| b.dynamic == dynamic);
            // Turning dynamic objects on or off needs a scene with / without the reserve.
            let snapshot_key = format!("{}|{dynamic}", t.map);
            if wanted
                && !have
                && c.request.is_none()
                && t.snapshot_done_for != snapshot_key
                && frame >= t.world_frame + SETTLE_FRAMES
                && frame.is_multiple_of(30)
                && in_game()
            {
                t.snapshot_done_for = snapshot_key;
                c.request = Some(snapshot_request(
                    &v,
                    &s,
                    if s.rt_build_at_load {
                        "map load"
                    } else {
                        "first use"
                    },
                ));
                ensure_worker(&mut c);
            }
            track_dynamic(
                &mut c,
                &s,
                built.as_deref().filter(|b| b.dynamic),
                &v,
                frame,
            );
        }
        if !c.running {
            ensure_worker(&mut c);
        }
    } else {
        track_dynamic_off(&mut c);
    }
    drop(t);
    drop(c);
    WAKE.notify_all();
}

/// Dynamic objects, main thread: runs the tracker over this frame's engine state and
/// queues its events for the worker (only with a scene built with the reserve).
fn track_dynamic(
    c: &mut Ctrl,
    s: &settings::Settings,
    built: Option<&Built>,
    v: &super::live::View,
    frame: u64,
) {
    let (Some(b), true) = (built, dynamic_wanted(s)) else {
        track_dynamic_off(c);
        return;
    };
    let set = DynSettings {
        props: s.rt_dynamic,
        characters: s.rt_characters,
        nearest: s.rt_characters_n as usize,
        budget_ms: s.rt_dynamic_budget_ms,
    };
    c.dynamic = Some(set);
    let mut tr = lock(&TRACKER);
    let tracker = tr.get_or_insert_with(Tracker::default);
    let events = tracker.tick(
        &track::Mapping {
            generation: b.scene.generation(),
            keys: &b.keys,
            entity_instances: b.scene.entity_instances(),
            present: &b.present,
        },
        &track::Frame {
            world: v.world,
            frame,
            camera: v.org,
            radius: s.rt_radius,
            props: set.props,
            characters: set.characters,
            nearest: set.nearest,
        },
    );
    let track = tracker.status;
    drop(tr);
    if c.events.len() + events.len() > MAX_EVENTS {
        let drop_n = (c.events.len() + events.len() - MAX_EVENTS).min(c.events.len());
        c.events.drain(..drop_n);
        c.events_dropped += drop_n as u64;
    }
    c.events.extend(events);
    let dropped = c.events_dropped;
    let mut st = lock(&STATUS);
    let d = st.dynamic.get_or_insert_with(DynamicStatus::default);
    d.track = track;
    d.events_dropped = dropped;
}

fn track_dynamic_off(c: &mut Ctrl) {
    c.dynamic = None;
    c.events.clear();
    if let Some(t) = lock(&TRACKER).as_mut()
        && t.generation() != 0
    {
        t.reset();
        lock(&STATUS).dynamic = None;
    }
}

/// The game is in the playing state with a running map (probe values).
fn in_game() -> bool {
    let p = crate::photo::probe::read();
    p.manager_ok && p.top == crate::photo::probe::STATE_IN_GAME && p.map_running()
}

// ------------------------------------------------------------------------------ worker --

fn worker() {
    let dev = match device::ensure() {
        Ok(d) => d,
        Err(e) => {
            fail(format!("no D3D12 device: {e}"));
            return;
        }
    };
    if !dev.inline_raytracing {
        fail("the GPU lacks inline ray tracing (DXR 1.1, SM 6.5)".into());
        return;
    }
    let queue = match device::Queue::new(&dev, "void-rt") {
        Ok(q) => q,
        Err(e) => {
            fail(e);
            return;
        }
    };
    let mut gpu = match Gpu::from_existing(dev.d12.clone(), queue.queue.clone()) {
        Ok(g) => g,
        Err(e) => {
            fail(format!("D3D12 context: {e}"));
            return;
        }
    };
    // Upload phases (VRAM before / after, submit GPU times) go to our log; every blocking
    // submit is followed by a short sleep so the game's queue gets the GPU in between.
    gpu.set_log(log::line);
    gpu.set_pause(SUBMIT_PAUSE);
    DEVICE_ID.store(identity(&dev.d12).unwrap_or(0), Ordering::Release);
    log::line(&format!(
        "rt: worker started on {} ({} MiB, driver {}); {}",
        gpu.info.name,
        gpu.info.vram_mib,
        gpu.info.driver,
        gpu.vram_text()
    ));
    let mut built: Option<Arc<Built>> = None;
    let mut current: Option<Arc<TlasHold>> = None;
    let mut spare: Vec<Arc<TlasHold>> = Vec::new();
    let mut last_tlas: Option<Instant> = None;
    let mut dyn_state: Option<DynState> = None;
    // Scene generation for which the dynamic layer could not start (not retried).
    let mut dyn_failed: u64 = 0;
    let r = (|| -> Result<(), String> {
        loop {
            let (request, live, radius, camera, map, task, cancel, dset, events) = {
                let mut c = lock(&CTRL);
                if c.task.is_none() && c.request.is_none() && !c.live && !c.keep {
                    return Ok(());
                }
                c = WAKE
                    .wait_timeout(c, Duration::from_millis(50))
                    .map(|(g, _)| g)
                    .unwrap_or_else(|p| p.into_inner().0);
                let cancel = std::mem::take(&mut c.cancel_task);
                (
                    c.request.take(),
                    c.live,
                    c.radius,
                    c.camera,
                    c.map.clone(),
                    c.task.take(),
                    cancel,
                    c.dynamic,
                    std::mem::take(&mut c.events),
                )
            };
            let mut task = if cancel { None } else { task };
            // Map change: drop the scene (a task for the old map ends itself).
            if let Some(b) = &built
                && !map.is_empty()
                && !b.map.eq_ignore_ascii_case(&map)
            {
                log::line(&format!(
                    "rt: map changed, dropping the scene of '{}'",
                    b.map
                ));
                if let Some(t) = task.as_mut() {
                    t.release_scene();
                }
                VIEW.store(None);
                BUILT.store(None);
                PHOTO_TLAS.store(None);
                current = None;
                spare.clear();
                dyn_state = None;
                built = None;
                set_phase("idle");
            }
            if let Some(req) = request {
                let have = built.as_ref().is_some_and(|b| {
                    b.map.eq_ignore_ascii_case(&req.source.map)
                        && (b.spike || !req.opts.spike_primitives)
                        && b.texture_size == req.opts.texture_size
                        && b.dynamic == req.opts.dynamic.is_some()
                });
                if !have {
                    // The old scene's video memory is freed before the new one is uploaded:
                    // the task drops its session on it, and goes back to CTRL for the
                    // build (seconds) so that leaving photo mode meanwhile aborts it.
                    if let Some(mut t) = task.take() {
                        t.release_scene();
                        let mut c = lock(&CTRL);
                        if c.task.is_none() && !c.cancel_task {
                            c.task = Some(t);
                        }
                    }
                    VIEW.store(None);
                    BUILT.store(None);
                    PHOTO_TLAS.store(None);
                    current = None;
                    spare.clear();
                    dyn_state = None;
                    built = None;
                    match build(&mut gpu, &req) {
                        Ok(b) => {
                            let b = Arc::new(b);
                            BUILT.store(Some(b.clone()));
                            built = Some(b);
                        }
                        Err(e) if device_lost(&e) => return Err(e),
                        Err(e) => {
                            log::line(&format!("rt: scene build failed: {e}"));
                            notify::error(format!("Ray-tracing scene not built: {e}"));
                            fail(e);
                        }
                    }
                    continue;
                }
            }
            // While a photo task runs, the live cut-off TLAS is not rebuilt unless the task
            // allows it (the hybrid photo terms: the reflections trace it): the camera is the
            // photo camera, the game is paused, and each rebuild waits for the task's submit
            // in flight (docs/raytracing.md 11.3, concurrency; 13).
            if live
                && task.as_ref().is_none_or(|t| t.allows_live())
                && let (Some(b), Some(cam)) = (&built, camera)
                && b.spike
            {
                // Dynamic objects: (re)start for this scene, apply this frame's events.
                match dset.filter(|_| b.dynamic) {
                    Some(ds) => {
                        if dyn_state
                            .as_ref()
                            .is_none_or(|d| !Arc::ptr_eq(d.scene(), &b.scene))
                            && dyn_failed != b.scene.generation()
                        {
                            dyn_state = None;
                            let dir = crate::game_dir().cloned().unwrap_or_default();
                            match DynState::new(&gpu, b.scene.clone(), dir, ds) {
                                Ok(d) => dyn_state = Some(d),
                                Err(e) => {
                                    dyn_failed = b.scene.generation();
                                    log::line(&format!("rt dynamic: not started: {e}"));
                                    notify::error(format!("Dynamic ray-tracing objects off: {e}"));
                                }
                            }
                        }
                        if let Some(d) = dyn_state.as_mut() {
                            d.tick(&mut gpu, events, ds, cam)?;
                        }
                    }
                    None => dyn_state = None,
                }
                let dynamics = dyn_state.as_ref().map(|d| &d.dynamics);
                let stale = current.as_ref().is_none_or(|h| {
                    !Arc::ptr_eq(&h.built, b)
                        || dist(h.center, cam) > 0.25 * radius
                        || (h.radius - radius).abs() > 1e-3
                        || h.set.is_some() != dynamics.is_some()
                });
                let changed = dynamics.and_then(|d| {
                    let set = current.as_ref()?.set.as_ref()?;
                    (set.version != d.version()).then_some(d)
                });
                let due = last_tlas.is_none_or(|t| t.elapsed() > Duration::from_millis(100));
                let mut refitted = None;
                let mut refit_ms = None;
                if !stale
                    && let Some(d) = changed
                    && current.as_ref().is_some_and(|h| h.refits < MAX_REFITS)
                {
                    let t = Instant::now();
                    refitted = refit(&mut gpu, b, &mut spare, &current, d)?;
                    if let Some(h) = &refitted {
                        refit_ms = Some(t.elapsed().as_secs_f64() * 1e3);
                        publish(h, b, h.center, h.radius);
                        lock(&STATUS).tlas_instances = h.tlas.count();
                    }
                }
                // A rebuild needs a TLAS nobody holds; new ones only up to MAX_HOLDS (each
                // has room for every instance: ~60 MiB for eotw), else wait for frames to
                // release theirs.
                let free = spare.iter().any(|h| Arc::strong_count(h) == 1);
                let can_build =
                    current.is_none() || free || TLAS_HOLDS.load(Ordering::Relaxed) < MAX_HOLDS;
                if let Some(h) = refitted {
                    if let Some(old) = current.replace(h) {
                        spare.push(old);
                    }
                } else if ((stale && due) || changed.is_some()) && can_build {
                    last_tlas = Some(Instant::now());
                    let t = Instant::now();
                    let hold =
                        cutoff(&mut gpu, b, &mut spare, &mut current, cam, radius, dynamics)?;
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    publish(&hold, b, cam, radius);
                    let mut st = lock(&STATUS);
                    st.tlas_instances = hold.tlas.count();
                    st.tlas_rebuilds += 1;
                    st.tlas_ms = ms;
                    drop(st);
                    current = Some(hold);
                }
                if let Some(d) = dyn_state.as_mut() {
                    if let Some(ms) = refit_ms {
                        d.status.tlas_refits += 1;
                        d.status.tlas_refit_ms = ms;
                    }
                    let mut st = lock(&STATUS);
                    st.dynamic.get_or_insert_with(DynamicStatus::default).worker = d.status.clone();
                }
            } else if VIEW.load().is_some() {
                // Unpublished (rt off, photo running): the next live frame rebuilds and
                // publishes again; the TLAS stays reusable once no frame holds it.
                VIEW.store(None);
                if let Some(h) = current.take() {
                    spare.push(h);
                }
            }
            // Spares of the current scene stay (a frame may still hold one: it becomes free
            // once released; at most MAX_HOLDS exist), older scenes' are freed.
            spare.retain(|h| built.as_ref().is_some_and(|b| Arc::ptr_eq(&h.built, b)));
            // The photo task traces the dynamic objects frozen at its start (the game is
            // paused; docs/raytracing.md 14.4): one full TLAS, kept until the task ends.
            match (&task, &dyn_state, &built) {
                (Some(_), Some(d), Some(b))
                    if PHOTO_TLAS
                        .load()
                        .as_ref()
                        .is_none_or(|p| p.0 != b.scene.generation()) =>
                {
                    let set = d.dynamics.full();
                    let t = SceneTlas::build(&mut gpu, &set, d.dynamics.tlas_capacity())?;
                    log::line(&format!(
                        "rt dynamic: photo TLAS frozen: {} instances ({} dynamic)",
                        set.len(),
                        set.dynamic_count()
                    ));
                    PHOTO_TLAS.store(Some(Arc::new((b.scene.generation(), Arc::new(t)))));
                }
                (None, _, _) if PHOTO_TLAS.load().is_some() => PHOTO_TLAS.store(None),
                _ => {}
            }
            if let Some(mut t) = task {
                match t.step(&mut gpu, built.as_ref()) {
                    Ok(true) => {
                        let mut c = lock(&CTRL);
                        if c.task.is_none() && !c.cancel_task {
                            c.task = Some(t);
                        }
                    }
                    Ok(false) => {}
                    Err(e) if device_lost(&e) => {
                        t.abort(&e);
                        return Err(e);
                    }
                    Err(e) => t.abort(&e),
                }
            }
        }
    })();
    VIEW.store(None);
    BUILT.store(None);
    PHOTO_TLAS.store(None);
    drop(current);
    drop(spare);
    drop(dyn_state);
    drop(built);
    if let Err(e) = r {
        log::line(&format!(
            "rt: GPU error, ray tracing disabled for this session: {e}{}",
            gpu.removed_reason()
        ));
        notify::error(format!("Ray tracing stopped: {e}"));
        GPU_LOST.store(true, Ordering::Relaxed);
        fail(e);
    } else {
        set_phase("stopped");
        log::line("rt: worker stopped (nothing to do), scene freed");
    }
    drop(gpu);
    DEVICE_ID.store(0, Ordering::Release);
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A cut-off TLAS around `cam` (with `dynamics`: its current instances): rebuilt in place
/// in a TLAS nobody else holds, else new.
fn cutoff(
    gpu: &mut Gpu,
    b: &Arc<Built>,
    spare: &mut Vec<Arc<TlasHold>>,
    current: &mut Option<Arc<TlasHold>>,
    cam: [f32; 3],
    radius: f32,
    dynamics: Option<&void_rt::ingame::Dynamics>,
) -> Result<Arc<TlasHold>, String> {
    if let Some(old) = current.take() {
        spare.push(old);
    }
    let set = dynamics.map(|d| d.cutoff(cam, radius));
    let capacity = dynamics.map_or(b.scene.instances().len(), |d| d.tlas_capacity());
    let reusable = spare.iter().position(|h| {
        Arc::strong_count(h) == 1 && Arc::ptr_eq(&h.built, b) && h.tlas.capacity() == capacity
    });
    let hold = if let Some(i) = reusable {
        let mut h = spare.swap_remove(i);
        let inner = Arc::get_mut(&mut h).ok_or("TLAS still shared")?;
        match &set {
            Some(s) => inner.tlas.rebuild(gpu, &s.descs)?,
            None => {
                b.scene
                    .update_cutoff_tlas(gpu, &mut inner.tlas, cam, radius)?;
            }
        }
        inner.center = cam;
        inner.radius = radius;
        inner.set = set;
        inner.refits = 0;
        write_types(gpu, inner)?;
        h
    } else {
        let tlas = match &set {
            Some(s) => Tlas::build(gpu, &s.descs, capacity)?,
            None => b.scene.build_cutoff_tlas(gpu, cam, radius)?,
        };
        let mut h = TlasHold::new(tlas, b.clone(), cam, radius, set);
        write_types(gpu, &mut h)?;
        Arc::new(h)
    };
    Ok(hold)
}

/// Writes the instance types of `h.set` into its type buffer (created on first use with
/// room for the TLAS capacity). `h` must not be published or held (its TLAS was just
/// built or refitted in place).
fn write_types(gpu: &mut Gpu, h: &mut TlasHold) -> Result<(), String> {
    let Some(set) = &h.set else {
        h.types = None;
        return Ok(());
    };
    if h.types.is_none() {
        h.types = Some(gpu.upload_heap(&vec![0u32; h.tlas.capacity().max(1)])?);
    }
    if let Some(b) = &h.types {
        gpu.write_mapped(b, &set.types)?;
    }
    Ok(())
}

/// Refit of the published cut-off TLAS with the dynamic objects' new transforms / BLASes
/// into a spare TLAS nobody holds (the published one may still be traced) whose last build
/// had the same instance count (`Tlas::update_from`). None when the instance list changed
/// or no such spare is free: the caller rebuilds (after one rebuild the previous TLAS is a
/// spare with the right count again).
fn refit(
    gpu: &mut Gpu,
    b: &Arc<Built>,
    spare: &mut Vec<Arc<TlasHold>>,
    current: &Option<Arc<TlasHold>>,
    dynamics: &void_rt::ingame::Dynamics,
) -> Result<Option<Arc<TlasHold>>, String> {
    let Some(cur) = current else { return Ok(None) };
    let Some(set) = cur.set.as_ref().and_then(|s| dynamics.refresh(s)) else {
        return Ok(None);
    };
    let Some(i) = spare.iter().position(|h| {
        Arc::strong_count(h) == 1
            && Arc::ptr_eq(&h.built, b)
            && h.tlas.capacity() == cur.tlas.capacity()
            && h.tlas.count() == cur.tlas.count()
    }) else {
        return Ok(None);
    };
    let mut h = spare.swap_remove(i);
    let inner = Arc::get_mut(&mut h).ok_or("TLAS still shared")?;
    inner.tlas.update_from(gpu, &cur.tlas, &set.descs)?;
    inner.center = cur.center;
    inner.radius = cur.radius;
    inner.set = Some(set);
    inner.refits = cur.refits + 1;
    write_types(gpu, inner)?;
    Ok(Some(h))
}

fn publish(hold: &Arc<TlasHold>, b: &Arc<Built>, cam: [f32; 3], radius: f32) {
    let (Some(offsets), Some(prims)) = (b.scene.primitive_offsets(), b.scene.primitives()) else {
        VIEW.store(None);
        return;
    };
    VIEW.store(Some(Arc::new(Published {
        generation: b.scene.generation(),
        tlas_id: TLAS_ID.fetch_add(1, Ordering::Relaxed) + 1,
        tlas_va: hold.tlas.va(),
        center: cam,
        radius,
        instance_count: hold.tlas.count() as u32,
        instance_types: hold.types.as_ref().map(|b| b.res.clone()),
        instance_type_count: hold.set.as_ref().map_or(0, |s| s.types.len() as u32),
        prim_offsets: offsets.res.clone(),
        geom_count: (offsets.size / 4) as u32,
        prims: prims.res.clone(),
        prim_count: (prims.size / 24) as u32,
        keep: hold.clone(),
    })));
}

fn build(gpu: &mut Gpu, req: &Request) -> Result<Built, String> {
    let map = req.source.map.clone();
    set_phase(format!("building {map}"));
    lock(&STATUS).error = None;
    log::line(&format!(
        "rt: building the scene of '{map}' ({}; {} entities, {} lights, textures <= {}, FC6 primitives {}, dynamic reserve {})",
        req.why,
        req.source.entities.len(),
        req.source.lights.len(),
        req.opts.texture_size,
        req.opts.spike_primitives,
        req.opts.dynamic.is_some()
    ));
    let t = Instant::now();
    log::line(&format!("rt: before the decode: {}", gpu.vram_text()));
    let cpu = void_rt::ingame::load_scene(&req.source, &req.opts)?;
    let load_s = t.elapsed().as_secs_f64();
    let scene = void_rt::ingame::upload(gpu, cpu)?;
    let upload_s = scene.upload_seconds();
    let instances = scene.instances().len();
    let rep = scene.report();
    log::line(&format!(
        "rt: scene of '{map}' ready: decode {load_s:.1} s, upload + BVH {upload_s:.1} s ({} submits, longest {:.1} ms GPU), {instances} instances, {} lights kept of {}, textures <= {}; {}",
        rep.phases.iter().map(|p| p.submits).sum::<usize>(),
        rep.max_submit_ms,
        scene.tables().light_count,
        scene.light_rows(),
        rep.texture_size,
        gpu.vram_text()
    ));
    if rep.texture_size < rep.requested_texture_size {
        notify::warn(format!(
            "Ray-tracing scene: textures reduced to {} (video memory)",
            rep.texture_size
        ));
    }
    let mut st = lock(&STATUS);
    st.phase = "ready".into();
    st.built_map.clone_from(&map);
    st.load_s = load_s;
    st.upload_s = upload_s;
    st.instances = instances;
    st.spike = req.opts.spike_primitives;
    Ok(Built {
        map,
        scene: Arc::new(scene),
        spike: req.opts.spike_primitives,
        texture_size: req.opts.texture_size,
        load_s,
        upload_s,
        dynamic: req.opts.dynamic.is_some(),
        keys: req.keys.clone(),
        present: req.present.clone(),
    })
}
