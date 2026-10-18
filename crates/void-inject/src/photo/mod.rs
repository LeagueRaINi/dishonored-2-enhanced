//! Photo mode in game (docs/raytracing.md sections 11 and 13; setting `photo = 1`).
//!
//! - [`probe`]: read-only values (Ansel session flag, game pointer, logic-state stack, map
//!   state, Ansel camera block); logged on change with `photo = 1` or `photo_probe = 1`
//!   (camera-only changes at most once per second).
//! - [`ansel`]: the engine's own Ansel session (pause, player/hands/HUD hidden) and the free
//!   camera through the `updateCamera` import slot.
//! - [`task`]: the photo's work on the RT worker (`rt::scene`): the hybrid terms or the
//!   path tracer, saving.
//! - [`hybrid`] (`photo_display = hybrid`, default): the game keeps rendering its paused
//!   frame; ray-traced sun visibility and AO replace its `_SSAO_ScreenShadows` inputs (S1
//!   entry point of `ao.rs` / `contact.rs`), ray-traced reflections accumulate on the FSR
//!   path (`ssr::rt_reflect`).
//! - [`composite`]: the FOG pass hook (render region, the hybrid terms' depth capture), the
//!   path-traced image into `hdrBuffer` (`pathtrace`) or drawn at present (`overlay`),
//!   back-buffer capture.
//!
//! Main thread ([`tick`], from `hook::observe` after `RenderAllViews`): hotkey, entering
//! (preconditions, scene snapshot, session start, camera redirect, input capture, pass hooks,
//! reflections, task) and leaving (task cancelled, external sources removed, reflections back
//! to live, camera slot restored, session stopped, input released). Every failure leaves
//! photo mode cleanly with a toast and a log line. `photo = 0`: nothing beyond one settings
//! check per frame (no hook, no patch, no thread).

pub mod ansel;
pub mod composite;
pub mod hybrid;
mod offline;
pub mod probe;
mod task;

pub use offline::offline_check;

/// `photo_display`: what photo mode shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Display {
    /// The game's frame with ray-traced sun visibility, AO and reflections (`hybrid`).
    #[default]
    Hybrid,
    /// Our path-traced image composited before the fog pass (`pathtrace`, debug).
    PathTrace,
    /// Our path-traced image tone-mapped over the frame at present (`overlay`).
    Overlay,
}

impl Display {
    pub const ALL: [Display; 3] = [Display::Hybrid, Display::PathTrace, Display::Overlay];

    pub fn name(self) -> &'static str {
        match self {
            Display::Hybrid => "hybrid",
            Display::PathTrace => "pathtrace",
            Display::Overlay => "overlay",
        }
    }

    pub fn parse(v: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == v)
    }

    /// The path tracer runs (pathtrace or overlay display).
    pub fn path_traced(self) -> bool {
        self != Display::Hybrid
    }
}

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{cvars, ffi, log, notify, overlay, settings};

pub use probe::Probe;

/// Overlay / hotkey actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Toggle,
    Save,
}

/// How the game is held still.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pause {
    /// The engine's Ansel session (player, hands and HUD hidden; free camera).
    Ansel,
    /// Fallback `g_stopTime 2` (the session was refused): frozen camera, HUD/hands visible.
    StopTime(i32),
}

struct Active {
    pause: Pause,
    free_camera: bool,
    world: u64,
    map: String,
    size: [u32; 2],
    since: Instant,
    /// Ticks (frames) since entering.
    ticks: u32,
    shared: Arc<task::Shared>,
    /// Ray-traced reflections accumulate for this photo (`rt_reflect::set_photo`).
    reflections: bool,
    /// Free-camera version last seen (camera moves restart the terms).
    cam_version: u64,
    /// Progress line last shown (`notify::status`) and when; the ready line was shown.
    progress: Option<(String, Instant)>,
    ready_shown: bool,
}

/// `notify::status` key of the accumulation progress line.
const PROGRESS_KEY: &str = "photo progress";
/// The progress line is refreshed at least this often (it fades out after `PROGRESS_TTL`
/// without a refresh, e.g. when the main thread stops ticking).
const PROGRESS_REFRESH: Duration = Duration::from_secs(1);
const PROGRESS_TTL: Duration = Duration::from_secs(3);
const READY_TTL: Duration = Duration::from_secs(5);

/// The accumulation progress line and whether everything reached its target. Hybrid
/// display: the enabled terms' rays per pixel (reflections: samples per texel against the
/// same `target`), or the worker's phase before the first rays; path tracer: samples per
/// pixel. None: nothing to report.
pub(crate) fn progress_text(
    display: Display,
    hybrid: Option<&hybrid::Status>,
    reflections: Option<u32>,
    target: u32,
    spp: (u32, u32),
) -> Option<(String, bool)> {
    if display != Display::Hybrid {
        let (done, of) = spp;
        return (of > 0).then(|| {
            (
                format!("Photo: path tracing {}/{of} spp", done.min(of)),
                done >= of,
            )
        });
    }
    let h = hybrid?;
    let mut parts = Vec::new();
    if h.sun.enabled {
        parts.push(format!(
            "sun {}/{}",
            h.sun.rays.min(h.sun.target),
            h.sun.target
        ));
    }
    if h.ao.enabled {
        parts.push(format!("AO {}/{}", h.ao.rays.min(h.ao.target), h.ao.target));
    }
    let target = target.max(1);
    if let Some(n) = reflections {
        parts.push(format!("reflections {}/{target}", n.min(target)));
    }
    let done = h.phase == "done" && reflections.is_none_or(|n| n >= target);
    let started = matches!(h.phase.as_str(), "depth prepass" | "accumulating" | "done");
    if !started {
        return Some((format!("Photo: {}", h.phase), false));
    }
    if parts.is_empty() {
        return None;
    }
    Some((format!("Photo: {}", parts.join(", ")), done))
}

/// Shows the progress line while accumulating, then once "Photo ready, Enter to save".
fn show_progress(a: &mut Active, s: &settings::Settings) {
    let spp = (
        a.shared.spp_done.load(Ordering::Relaxed),
        a.shared.spp_target.load(Ordering::Relaxed),
    );
    let hy = hybrid::status();
    let refl = a.reflections.then(crate::ssr::rt_reflect::photo_samples);
    let Some((text, done)) =
        progress_text(s.photo_display, hy.as_ref(), refl, s.photo_terms_spp, spp)
    else {
        return;
    };
    if done {
        if !a.ready_shown {
            a.ready_shown = true;
            a.progress = None;
            log::line(&format!("photo: ready ({text})"));
            notify::status(PROGRESS_KEY, "Photo ready, Enter to save", READY_TTL);
        }
        return;
    }
    a.ready_shown = false;
    let stale = a
        .progress
        .as_ref()
        .is_none_or(|(t, at)| *t != text || at.elapsed() >= PROGRESS_REFRESH);
    if stale {
        notify::status(PROGRESS_KEY, text.clone(), PROGRESS_TTL);
        a.progress = Some((text, Instant::now()));
    }
}

/// Ticks after entering during which the logic state and the session flag are not checked:
/// the engine applies the session from its next game frame on, and `enter` runs in the
/// middle of a tick (RtSpike in-game bug: the entry tick compared against the probe read
/// before `startSession` and left at once).
pub const GRACE_TICKS: u32 = 10;

/// Why photo mode must end now (None = stay). `p` is this tick's probe, `world_changed`
/// the render world of this frame differs from the one at entry, `task_error` an error of
/// the render task. Returns (reason, is an error).
pub(crate) fn exit_reason(
    p: &Probe,
    pause: Pause,
    ticks: u32,
    world_changed: bool,
    task_error: Option<&str>,
) -> Option<(String, bool)> {
    if let Some(e) = task_error {
        return Some((format!("render failed: {e}"), true));
    }
    if world_changed {
        return Some(("map changed".to_owned(), false));
    }
    if p.game == 0 {
        return Some(("the game ended (game pointer 0)".to_owned(), false));
    }
    if !p.manager_ok {
        return Some(("game logic manager no longer found".to_owned(), true));
    }
    if !p.map_running() {
        return Some((
            format!("the map stopped running (map state {:?})", p.map_state),
            false,
        ));
    }
    if ticks < GRACE_TICKS {
        return None;
    }
    if p.top != probe::STATE_IN_GAME {
        return Some((
            format!(
                "game state changed to {} ({}), depth {}",
                p.top,
                probe::state_name(p.top),
                p.depth
            ),
            false,
        ));
    }
    if pause == Pause::Ansel && p.session_flag != Some(1) {
        return Some((
            format!(
                "the Ansel session ended (session flag {:?} after {ticks} frames)",
                p.session_flag
            ),
            true,
        ));
    }
    None
}

/// Camera-only probe changes are logged at most this often.
const PROBE_CAMERA_LOG: std::time::Duration = std::time::Duration::from_secs(1);
/// Window of the paused-frame pass rates (logged once per window).
const RATE_WINDOW: std::time::Duration = std::time::Duration::from_secs(1);

/// Engine pass callbacks counted by our detours (cumulative) and our main-thread ticks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassCounts {
    pub ticks: u64,
    /// `ARK_RENDERPASS_FOG` (photo composite hook).
    pub fog: u64,
    /// `ARK_RENDERPASS_SCREENSHADOWS` (contact-shadow hook).
    pub shadows: u64,
    /// HBAO+ `RenderAO` (AO hook).
    pub ssao: u64,
}

impl PassCounts {
    fn now(ticks: u64) -> Self {
        Self {
            ticks,
            fog: composite::fog_frames(),
            shadows: crate::contact::passes(),
            ssao: crate::ao::passes(),
        }
    }
}

/// Passes per second while in photo mode (step 0 of the hybrid terms: the engine keeps
/// rendering the paused frame, so the `.x` / `.y` replacements apply every frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PassRates {
    pub ticks: f32,
    pub fog: f32,
    pub shadows: f32,
    pub ssao: f32,
    /// Which detours are installed (a rate of 0 without its hook means "not counted").
    pub fog_hooked: bool,
    pub shadows_hooked: bool,
    pub ssao_hooked: bool,
}

impl PassRates {
    pub(crate) fn between(a: PassCounts, b: PassCounts, seconds: f32) -> Self {
        let r = |x: u64, y: u64| y.saturating_sub(x) as f32 / seconds.max(1e-3);
        Self {
            ticks: r(a.ticks, b.ticks),
            fog: r(a.fog, b.fog),
            shadows: r(a.shadows, b.shadows),
            ssao: r(a.ssao, b.ssao),
            fog_hooked: composite::status().hooked,
            shadows_hooked: crate::contact::hooked(),
            ssao_hooked: crate::ao::status().hooked,
        }
    }

    /// The renderer stopped: ticks advance but no world pass was recorded in the window.
    pub fn renderer_stopped(&self) -> bool {
        self.ticks > 0.0 && self.fog_hooked && self.fog == 0.0
    }

    pub fn describe(&self) -> String {
        let v = |hooked: bool, x: f32| {
            if hooked {
                format!("{x:.1}/s")
            } else {
                "not hooked".to_owned()
            }
        };
        format!(
            "frames {:.1}/s, FOG {}, SCREENSHADOWS {}, SSAO {}",
            self.ticks,
            v(self.fog_hooked, self.fog),
            v(self.shadows_hooked, self.shadows),
            v(self.ssao_hooked, self.ssao)
        )
    }
}

struct State {
    active: Option<Active>,
    key_held: bool,
    save_held: bool,
    actions: Vec<Action>,
    /// Latest reading (overlay).
    last_probe: Option<Probe>,
    /// Last logged reading and when it was logged.
    logged_probe: Option<(Probe, Instant)>,
    hooks_tried: bool,
    last_error: Option<String>,
    last_save: Option<String>,
    /// Start of the current pass-rate window (photo mode active) and the last rates.
    rate_window: Option<(Instant, PassCounts)>,
    rates: Option<PassRates>,
    stopped_warned: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    active: None,
    key_held: false,
    save_held: false,
    actions: Vec::new(),
    last_probe: None,
    logged_probe: None,
    hooks_tried: false,
    last_error: None,
    last_save: None,
    rate_window: None,
    rates: None,
    stopped_warned: false,
});

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Queues an action for the next frame (overlay buttons).
pub fn action(a: Action) {
    lock(&STATE).actions.push(a);
}

/// State for the overlay.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub enabled: bool,
    pub active: bool,
    /// "Ansel session", "g_stopTime fallback" or empty.
    pub pause: String,
    pub free_camera: bool,
    pub phase: String,
    pub spp_done: u32,
    pub spp_target: u32,
    pub ms_per_spp: f32,
    pub size: [u32; 2],
    pub seconds: f32,
    pub probe: Option<Probe>,
    pub composite: composite::Status,
    pub scene: crate::rt::scene::Status,
    /// Hybrid terms (sun, AO) while the hybrid display runs.
    pub hybrid: Option<hybrid::Status>,
    /// Reflection samples per texel accumulated for this photo (None: not active).
    pub reflections: Option<u32>,
    pub last_error: Option<String>,
    pub last_save: Option<String>,
    /// Engine passes per second in photo mode (last 1-s window; kept after leaving).
    pub rates: Option<PassRates>,
}

pub fn status() -> Status {
    let st = lock(&STATE);
    let mut out = Status {
        enabled: settings::get().photo,
        probe: st.last_probe.clone(),
        composite: composite::status(),
        scene: crate::rt::scene::status(),
        last_error: st.last_error.clone(),
        last_save: st.last_save.clone(),
        rates: st.rates,
        ..Status::default()
    };
    if let Some(a) = &st.active {
        out.active = true;
        out.pause = match a.pause {
            Pause::Ansel => "Ansel session".into(),
            Pause::StopTime(_) => "g_stopTime fallback".into(),
        };
        out.free_camera = a.free_camera;
        out.phase = lock(&a.shared.phase).clone();
        out.spp_done = a.shared.spp_done.load(Ordering::Relaxed);
        out.spp_target = a.shared.spp_target.load(Ordering::Relaxed);
        out.ms_per_spp = f32::from_bits(a.shared.ms_per_spp.load(Ordering::Relaxed));
        out.size = a.size;
        out.seconds = a.since.elapsed().as_secs_f32();
        out.hybrid = hybrid::status();
        out.reflections = a.reflections.then(crate::ssr::rt_reflect::photo_samples);
    }
    out
}

/// Photo mode is running (input capture, hidden HUD).
pub fn active() -> bool {
    lock(&STATE).active.is_some()
}

/// The game window has the keyboard focus (foreground window of this process).
fn focused() -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};
    // SAFETY: plain queries; the out pointer is a local.
    unsafe {
        let w = GetForegroundWindow();
        let mut pid = 0u32;
        GetWindowThreadProcessId(w, Some(&mut pid));
        pid == std::process::id()
    }
}

/// Main thread, every frame after `RenderAllViews`.
pub fn tick(requests: u64, _frame: u64) {
    let s = settings::get();
    let mut st = lock(&STATE);
    let mut probe_now = None;
    if s.photo || s.photo_probe || st.active.is_some() {
        let p = probe::read();
        // State changes are logged at once; camera moves (free camera) at most once per
        // second.
        let log_it = st.logged_probe.as_ref().is_none_or(|(l, at)| {
            !l.same_except_camera(&p) || (!l.same_as(&p) && at.elapsed() >= PROBE_CAMERA_LOG)
        });
        if log_it {
            log::line(&format!("photo probe: {}", p.describe()));
            st.logged_probe = Some((p.clone(), Instant::now()));
        }
        st.last_probe = Some(p.clone());
        probe_now = Some(p);
    }
    if !s.photo {
        if st.active.is_some() {
            leave(&mut st, "photo mode turned off", false);
        }
        st.actions.clear();
        return;
    }
    if !st.hooks_tried {
        st.hooks_tried = true;
        if let Err(e) = composite::install() {
            log::line(&format!(
                "photo: {e}; only the overlay display is available"
            ));
        }
        if !crate::mouse::installed()
            && let Err(e) = crate::mouse::install()
        {
            log::line(&format!(
                "photo: mouse hooks: {e}; no free-camera mouse look"
            ));
        }
    }
    let focus = focused();
    let down = ffi::key_down(s.photo_key as i32);
    let mut actions = std::mem::take(&mut st.actions);
    if down && !st.key_held && focus {
        actions.push(Action::Toggle);
    }
    st.key_held = down;
    if st.active.is_some() {
        let enter = ffi::key_down(0x0D);
        if enter && !st.save_held && focus && !overlay::visible() {
            actions.push(Action::Save);
        }
        st.save_held = enter;
        if focus && !overlay::visible() && ffi::key_down(0x1B) {
            leave(&mut st, "left with Esc", false);
            return;
        }
    }
    let mut entered_now = false;
    for a in actions {
        match a {
            Action::Toggle if st.active.is_some() => leave(&mut st, "left", false),
            Action::Toggle => match enter(requests, &s) {
                Ok(a) => {
                    notify::info(format!(
                        "Photo mode: {}x{}, {}{}. WASD/QE + mouse move, Enter saves, {} or Esc leaves.",
                        a.size[0],
                        a.size[1],
                        match a.pause {
                            Pause::Ansel => "game paused (Ansel session)",
                            Pause::StopTime(_) => "game frozen (g_stopTime; HUD stays)",
                        },
                        if a.free_camera { "" } else { ", camera fixed" },
                        settings::key_name(s.photo_key)
                    ));
                    st.last_error = None;
                    st.active = Some(a);
                    entered_now = true;
                }
                Err(e) => {
                    log::line(&format!("photo: not started: {e}"));
                    notify::warn(format!("Photo mode not started: {e}"));
                    st.last_error = Some(e);
                }
            },
            Action::Save => match &st.active {
                Some(a) => a.shared.save.store(true, Ordering::Release),
                None => notify::warn("Photo mode is not running"),
            },
        }
    }
    // The probe of this tick was read before `enter` ran: no exit checks in the entry tick.
    if entered_now {
        return;
    }
    let Some(a) = st.active.as_mut() else {
        return;
    };
    a.ticks = a.ticks.saturating_add(1);
    // Leave on a map change, a state change (menu, cinematic, transition), a lost session
    // or a task error; each reason is logged separately by `leave`.
    let p = probe_now.unwrap_or_else(probe::read);
    let world_changed = crate::rt::live::world(requests).is_some_and(|w| w != a.world);
    let task_error = lock(&a.shared.error).clone();
    let reason = exit_reason(&p, a.pause, a.ticks, world_changed, task_error.as_deref());
    if a.ticks < GRACE_TICKS
        && a.pause == Pause::Ansel
        && (p.session_flag != Some(1) || p.top != probe::STATE_IN_GAME)
    {
        log::line(&format!(
            "photo: frame {} after entering (grace): {}",
            a.ticks,
            p.describe()
        ));
    }
    let Some(a) = st.active.as_ref() else {
        return;
    };
    let ticks = u64::from(a.ticks);
    if let Some((why, error)) = reason {
        log::line(&format!(
            "photo: exit condition: {why}; probe: {}",
            p.describe()
        ));
        leave(&mut st, &why, error);
        return;
    }
    pass_rates(&mut st, ticks);
    let Some(a) = st.active.as_mut() else {
        return;
    };
    // A camera move restarts the terms: the registered ones no longer fit the frame.
    if let Some((_, v)) = ansel::current()
        && v != a.cam_version
    {
        a.cam_version = v;
        hybrid::unregister();
        if a.reflections {
            crate::ssr::rt_reflect::reset_photo();
        }
    }
    let Some(a) = st.active.as_ref() else {
        return;
    };
    // Free-camera input while the overlay is closed; the cursor is hidden and held at the
    // window centre meanwhile (the mouse is read through DirectInput).
    let capture = !overlay::visible() && focus;
    crate::mouse::set_capture(capture);
    crate::mouse::lock_cursor(capture && a.free_camera);
    if a.free_camera {
        let input = if capture {
            ansel::poll_input()
        } else {
            let _ = crate::mouse::take_motion();
            ansel::Input::default()
        };
        ansel::set_input(input);
    }
    let messages = std::mem::take(&mut *lock(&a.shared.messages));
    for m in messages {
        if m.starts_with("Photo not saved") {
            notify::error(m.clone());
        } else {
            notify::info(m.clone());
        }
        st.last_save = Some(m);
    }
    match composite::take_capture_result() {
        Some(Ok(path)) => {
            log::line(&format!("photo: game frame saved to {}", path.display()));
            notify::info(format!("Game frame saved: {}", path.display()));
        }
        Some(Err(e)) => {
            log::line(&format!("photo: game frame not saved: {e}"));
            notify::error(format!("Game frame not saved: {e}"));
        }
        None => {}
    }
    if let Some(a) = st.active.as_mut() {
        show_progress(a, &s);
    }
}

/// Paused-frame check (main thread, photo mode active): per 1-s window the engine passes our
/// detours saw, logged and kept for the panel. No FOG pass in a window while our ticks run
/// means the renderer stopped: the `.x` / `.y` terms would be applied once only (logged and
/// toasted once per photo session; docs/raytracing.md 13.1).
fn pass_rates(st: &mut State, ticks: u64) {
    let now = Instant::now();
    let counts = PassCounts::now(ticks);
    let Some((t0, c0)) = st.rate_window else {
        st.rate_window = Some((now, counts));
        return;
    };
    let dt = now.duration_since(t0);
    if dt < RATE_WINDOW {
        return;
    }
    let r = PassRates::between(c0, counts, dt.as_secs_f32());
    log::line(&format!("photo: paused-frame passes: {}", r.describe()));
    if r.renderer_stopped() && !st.stopped_warned {
        st.stopped_warned = true;
        log::line(
            "photo: no FOG pass in the last second: the engine stopped rendering the paused frame; the ray-traced .x/.y terms would show only once (use photo_display = pathtrace)",
        );
        notify::warn("Photo mode: the game stopped rendering while paused (see the log)");
    }
    st.rates = Some(r);
    st.rate_window = Some((now, counts));
}

/// The SCREENSHADOWS and HBAO+ `RenderAO` detours on demand (photo entry): they count the
/// paused frames and carry the `.y` / `.x` terms (S1 entry point). Each is installed once
/// per process (also when `contact_shadows = 0` / `ao_mode = hbao`) and stays installed
/// after photo mode; with nothing registered and its own setting off it only forwards the
/// call (bit-identical output, `ao_offline` / `contact_offline`). Failures are logged; the
/// term needing the hook is then skipped.
fn install_pass_hooks() {
    if !crate::contact::hooked()
        && let Err(e) = crate::contact::install()
    {
        log::line(&format!(
            "photo: SCREENSHADOWS hook unavailable: {e}; no ray-traced sun shadows"
        ));
    }
    let ao = crate::ao::status();
    if !ao.hooked
        && !ao.waiting
        && let Err(e) = crate::ao::install()
    {
        log::line(&format!(
            "photo: HBAO+ RenderAO hook unavailable: {e}; no ray-traced AO"
        ));
    }
}

/// Enters photo mode (main thread). On error nothing stays changed.
fn enter(requests: u64, s: &settings::Settings) -> Result<Active, String> {
    let p = probe::read();
    if p.game == 0 {
        return Err("no game running".into());
    }
    if !p.manager_ok {
        return Err("game logic manager not found".into());
    }
    if p.top != probe::STATE_IN_GAME {
        return Err(format!(
            "only while playing (logic state {} = {})",
            p.top,
            probe::state_name(p.top)
        ));
    }
    if !p.map_running() {
        return Err("no map running".into());
    }
    if p.session_flag != Some(0) {
        return Err(format!("Ansel session flag is {:?}", p.session_flag));
    }
    if overlay::visible() {
        return Err("close the overlay first".into());
    }
    let view = crate::rt::live::view(requests).ok_or("no view this frame")?;
    if view.map.is_empty() {
        return Err("no map name".into());
    }
    let fog = composite::last_frame().ok_or("no world frame rendered yet (FOG pass not seen)")?;
    let size = [fog.region[2], fog.region[3]];
    let dev = crate::rt::device::ensure()?;
    if !dev.inline_raytracing {
        return Err("the GPU lacks inline ray tracing (DXR 1.1, SM 6.5)".into());
    }
    if crate::rt::scene::gpu_lost() {
        return Err("ray tracing is disabled after a GPU error this session (see the log)".into());
    }
    // Scene inputs from game memory only when the map's scene is neither built nor being
    // built. The build request goes to the RT worker directly (below, once the pause
    // started), not through the render task, so leaving photo mode does not cancel it: it
    // runs until it completes or the map changes.
    let request = if crate::rt::scene::built_or_pending(&view.map) {
        None
    } else {
        let t = Instant::now();
        let req = crate::rt::scene::snapshot_request(&view, s, "photo");
        log::line(&format!(
            "photo: scene snapshot took {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        ));
        Some(req)
    };
    // Pause: the engine's Ansel session, else g_stopTime.
    let pause = match ansel::start_session(&p) {
        Ok(true) => Pause::Ansel,
        r => {
            match r {
                Ok(_) => log::line("photo: the Ansel session was refused; g_stopTime fallback"),
                Err(e) => log::line(&format!(
                    "photo: Ansel session not started: {e}; g_stopTime fallback"
                )),
            }
            let old =
                cvars::set_int("g_stopTime", 2).ok_or("cannot pause: g_stopTime not found")?;
            Pause::StopTime(old)
        }
    };
    let free_camera = pause == Pause::Ansel
        && match ansel::install_camera() {
            Ok(()) => true,
            Err(e) => {
                log::line(&format!("photo: free camera unavailable: {e}"));
                false
            }
        };
    install_pass_hooks();
    let reflections = s.photo_display == Display::Hybrid && start_reflections(s);
    crate::mouse::set_capture(true);
    if let Some(req) = request {
        crate::rt::scene::request_build(req);
    }
    let shared = Arc::new(task::Shared::default());
    crate::rt::scene::set_task(Some(Box::new(task::PhotoTask::new(
        shared.clone(),
        view.map.clone(),
        view.camera(),
        size,
        view.znear,
        dev,
    ))));
    log::line(&format!(
        "photo: entered on '{}' at {}x{} ({:?}, free camera {free_camera}), camera org {:?} fov_y {:.4}, znear {}",
        view.map, size[0], size[1], pause, view.org, view.fov[1], view.znear
    ));
    Ok(Active {
        pause,
        free_camera,
        world: view.world,
        map: view.map,
        size,
        since: Instant::now(),
        ticks: 0,
        shared,
        reflections,
        cam_version: ansel::current().map_or(0, |(_, v)| v),
        progress: None,
        ready_shown: false,
    })
}

/// Ray-traced reflections for the photo (hybrid display, `photo_reflections`): the FSR-path
/// reflections accumulate `photo_reflect_rays` rays per texel and frame. Needs FSR and
/// `rt_reflections = 1` (with `ssr = 1`, `rt = 1`); otherwise a toast and the term is skipped.
fn start_reflections(s: &settings::Settings) -> bool {
    if !s.photo_reflections {
        return false;
    }
    let missing = if !s.fsr || !crate::fsr::running() {
        Some("reflections need FSR on".to_owned())
    } else if !s.rt_reflections || !s.ssr || !s.rt {
        Some("reflections need rt_reflections = 1 (with ssr = 1 and rt = 1)".to_owned())
    } else {
        None
    };
    if let Some(m) = missing {
        log::line(&format!("photo: {m}; reflection term skipped"));
        notify::warn(m);
        return false;
    }
    crate::ssr::rt_reflect::set_photo(Some(s.photo_reflect_rays));
    log::line(&format!(
        "photo: reflections accumulate {} rays per texel and frame",
        s.photo_reflect_rays
    ));
    true
}

/// Leaves photo mode: task cancelled, composite off, camera slot restored, session stopped
/// (or `g_stopTime` restored), input released. Logs and toasts every failure.
fn leave(st: &mut State, why: &str, error: bool) {
    let Some(a) = st.active.take() else {
        return;
    };
    st.rate_window = None;
    st.stopped_warned = false;
    crate::rt::scene::set_task(None);
    hybrid::unregister();
    if a.reflections {
        crate::ssr::rt_reflect::set_photo(None);
    }
    composite::deactivate();
    crate::mouse::set_capture(false);
    crate::mouse::lock_cursor(false);
    notify::clear_status(PROGRESS_KEY);
    let mut problems = Vec::new();
    if let Err(e) = ansel::uninstall_camera() {
        problems.push(format!("camera slot: {e}"));
    }
    match a.pause {
        Pause::Ansel => {
            if let Err(e) = ansel::stop_session() {
                problems.push(format!("stopSession: {e}"));
            }
        }
        Pause::StopTime(old) => {
            if cvars::set_int("g_stopTime", old).is_none() {
                problems.push("g_stopTime not restored".into());
            }
        }
    }
    log::line(&format!(
        "photo: left on '{}' after {:.1} s: {why}{}",
        a.map,
        a.since.elapsed().as_secs_f64(),
        if problems.is_empty() {
            String::new()
        } else {
            format!(" (problems: {})", problems.join("; "))
        }
    ));
    if error || !problems.is_empty() {
        let msg = if problems.is_empty() {
            why.to_owned()
        } else {
            format!("{why}; {}", problems.join("; "))
        };
        notify::error(format!("Photo mode ended: {msg}"));
        st.last_error = Some(msg);
    } else {
        notify::info(format!("Photo mode: {why}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The in-game reading of the bug report before `startSession` (flag 0, in game).
    fn playing(flag: u8) -> Probe {
        Probe {
            base: true,
            session_flag: Some(flag),
            game: 0x287_0BEA_D050,
            current_game: 0x287_0BEA_D050,
            manager_ok: true,
            top: probe::STATE_IN_GAME,
            depth: 1,
            map_state: Some(probe::MAP_RUNNING),
            camera: Some([0.0; 8]),
        }
    }

    #[test]
    fn no_exit_while_the_session_settles() {
        // In-game bug: the probe read before entering (flag 0) must not end the session.
        for t in 0..GRACE_TICKS {
            assert_eq!(exit_reason(&playing(0), Pause::Ansel, t, false, None), None);
        }
        assert_eq!(
            exit_reason(&playing(1), Pause::Ansel, GRACE_TICKS, false, None),
            None
        );
        // The g_stopTime fallback has no session flag.
        assert_eq!(
            exit_reason(&playing(0), Pause::StopTime(0), 100, false, None),
            None
        );
    }

    #[test]
    fn each_exit_has_its_own_reason() {
        let after = GRACE_TICKS + 5;
        let why =
            |p: &Probe, t, world, err| exit_reason(p, Pause::Ansel, t, world, err).map(|r| r.0);
        let ended = why(&playing(0), after, false, None).unwrap();
        assert!(ended.starts_with("the Ansel session ended"), "{ended}");
        let mut menu = playing(1);
        menu.top = probe::STATE_PAUSE;
        menu.depth = 2;
        assert!(
            why(&menu, after, false, None)
                .unwrap()
                .contains("2 (pause)")
        );
        // State changes wait for the grace period, map changes and errors do not.
        assert_eq!(why(&menu, 1, false, None), None);
        assert_eq!(why(&playing(1), 1, true, None).unwrap(), "map changed");
        assert!(
            why(&playing(1), 1, false, Some("x"))
                .unwrap()
                .starts_with("render failed")
        );
        let mut unloaded = playing(1);
        unloaded.map_state = Some(2);
        assert!(
            why(&unloaded, 1, false, None)
                .unwrap()
                .contains("map stopped running")
        );
        let mut gone = playing(1);
        gone.game = 0;
        assert!(why(&gone, 1, false, None).unwrap().contains("game ended"));
    }

    #[test]
    fn progress_line() {
        use void_rt::hybrid::TermProgress;
        let term = |rays| TermProgress {
            enabled: true,
            rays,
            target: 256,
            ms_per_batch: 0.0,
            rays_per_batch: 4,
        };
        let mut h = hybrid::Status {
            phase: "accumulating".into(),
            sun: term(128),
            ao: term(128),
            ..hybrid::Status::default()
        };
        let text = |h: &hybrid::Status, r| progress_text(Display::Hybrid, Some(h), r, 256, (0, 0));
        assert_eq!(
            text(&h, Some(64)),
            Some((
                "Photo: sun 128/256, AO 128/256, reflections 64/256".into(),
                false
            ))
        );
        h.phase = "done".into();
        h.sun = term(256);
        h.ao = term(256);
        assert!(!text(&h, Some(255)).unwrap().1);
        assert!(text(&h, Some(300)).unwrap().1);
        assert!(text(&h, None).unwrap().1);
        h.phase = "waiting for the scene".into();
        assert_eq!(
            text(&h, None),
            Some(("Photo: waiting for the scene".into(), false))
        );
        assert_eq!(
            progress_text(Display::PathTrace, None, None, 256, (100, 1024)),
            Some(("Photo: path tracing 100/1024 spp".into(), false))
        );
        assert_eq!(
            progress_text(Display::Hybrid, None, None, 256, (0, 0)),
            None
        );
    }
}
