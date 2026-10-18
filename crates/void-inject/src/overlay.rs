//! In-game overlay (settings `overlay = 1`, toggle `overlay_key`, default F2): egui panels
//! for status, FSR, engine cvars, patches, texture mods, profiler and log, and the
//! settings file (docs/overlay.md).
//!
//! Frame hook: the engine present `sub_14036B3A0` is hooked in `hook.rs`; before the
//! original runs, [`before_present`] checks the toggle key and, while the overlay is
//! visible and the engine is about to call `IDXGISwapChain::Present`, builds the UI and
//! draws it into the swap chain's current back buffer (`paint.rs`). Drawing holds the
//! engine's immediate-context lock (`RS_LockImmediate` 0x14036A6F0 /
//! `RS_UnlockImmediate` 0x14036B4E0 with the render system 0x1432FC580, a spin lock taken
//! only by pass submission and context binding, never on the present path), so worker
//! submissions cannot interleave. Hidden: one key poll per frame and the frame-interval
//! ring, no egui work.
//!
//! Input (`input.rs`, `mouse.rs`): while visible the game gets no input; window messages
//! feed egui, DirectInput devices are unacquired and blocked, the game's cursor
//! re-centring and clipping are suspended, and our own cursor is drawn.

mod hud;
mod input;
mod paint;
mod ui;

use std::collections::{HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use hex_literal::hex;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::IDXGISwapChain;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
use windows::core::Interface;

use crate::settings::{self, Corner, Settings};
use crate::{bench, cvars, ffi, fsr, hook, log, mem, notify, patches, profiler, texmods, trace};

use paint::Painter;

static VISIBLE: AtomicBool = AtomicBool::new(false);
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// `RS_LockImmediate(rs)` / `RS_UnlockImmediate(rs)` (O, IDB): spin lock at rs+6532172,
/// active when the byte at rs+6532170 is set (multi-threaded submission).
const LOCK_RVA: u64 = 0x36_A6F0;
const LOCK_HEAD: [u8; 16] = hex!("48 83 EC 28 80 B9 4A AC 63 00 00 74 3D 33 D2 48");
const UNLOCK_RVA: u64 = 0x36_B4E0;
const UNLOCK_HEAD: [u8; 16] = hex!("80 B9 4A AC 63 00 00 74 0E 48 81 C1 4C AC 63 00");
type LockFn = unsafe extern "system" fn(u64);

/// Frame-interval history shown by the status graph.
const HISTORY_S: f64 = 5.0;

/// The overlay is visible (input goes to it).
pub(crate) fn visible() -> bool {
    VISIBLE.load(Ordering::Relaxed)
}

/// The overlay's present path is installed (photo mode's back-buffer capture and preview
/// run in it).
pub(crate) fn installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// What one overlay frame draws: the menu (panels, input, cursor), the HUD, the toasts.
#[derive(Default)]
struct Layers<'a> {
    menu: Option<(&'a ui::Snapshot, &'a mut dyn ui::Backend)>,
    hud: Option<(&'a hud::HudData, Corner)>,
    toasts: Option<(&'a hud::Toasts, Instant)>,
}

/// egui context + panel state; shared by the game path and the offline preview.
struct Core {
    egui: egui::Context,
    ui: ui::State,
    start: Instant,
    /// Fixed time step per frame instead of the wall clock (offline preview: the window
    /// fade-in completes regardless of how fast frames are produced).
    step: Option<f64>,
    frames: u32,
}

impl Core {
    fn new() -> Self {
        let egui = egui::Context::default();
        let mut style = egui::Style::default();
        let color = egui::Color32::from_rgb;
        let accent = color(91, 207, 190);
        let text = color(225, 233, 239);
        let border = egui::Stroke::new(1.0, color(48, 62, 75));
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.interact_size.y = 28.0;
        style.spacing.window_margin = egui::Margin::same(16);
        style.spacing.slider_width = 180.0;
        for (kind, size) in [
            (egui::TextStyle::Heading, 24.0),
            (egui::TextStyle::Body, 14.0),
            (egui::TextStyle::Button, 14.0),
            (egui::TextStyle::Small, 11.0),
        ] {
            style
                .text_styles
                .insert(kind, egui::FontId::proportional(size));
        }
        style
            .text_styles
            .insert(egui::TextStyle::Monospace, egui::FontId::monospace(13.0));
        let v = &mut style.visuals;
        *v = egui::Visuals::dark();
        v.override_text_color = Some(text);
        v.weak_text_color = Some(color(139, 157, 173));
        v.window_fill = color(17, 22, 29);
        v.panel_fill = color(24, 31, 40);
        v.extreme_bg_color = color(13, 18, 24);
        v.faint_bg_color = color(24, 31, 40);
        v.window_stroke = border;
        v.window_corner_radius = egui::CornerRadius::same(12);
        v.selection.bg_fill = color(35, 75, 75);
        v.selection.stroke = egui::Stroke::new(1.0, accent);
        v.hyperlink_color = accent;
        for (widget, fill, foreground) in [
            (&mut v.widgets.noninteractive, color(24, 31, 40), text),
            (&mut v.widgets.inactive, color(30, 39, 50), text),
            (&mut v.widgets.hovered, color(42, 58, 69), text),
            (&mut v.widgets.active, color(35, 75, 75), accent),
            (&mut v.widgets.open, color(35, 75, 75), accent),
        ] {
            widget.bg_fill = fill;
            widget.weak_bg_fill = fill;
            widget.bg_stroke = border;
            widget.fg_stroke = egui::Stroke::new(1.0, foreground);
            widget.corner_radius = egui::CornerRadius::same(5);
        }
        egui.set_theme(egui::Theme::Dark);
        egui.set_style_of(egui::Theme::Dark, style);
        Self {
            egui,
            ui: ui::State::default(),
            start: Instant::now(),
            step: None,
            frames: 0,
        }
    }

    /// Runs one UI frame for a `size` (pixels) target; queued input positions are in
    /// `client` pixels and reach egui only with the menu open.
    fn frame(
        &mut self,
        size: [f32; 2],
        client: [f32; 2],
        scale: Option<f32>,
        mut layers: Layers<'_>,
    ) -> egui_directx11::RendererOutput {
        let ppp = scale.unwrap_or(size[1] / 1080.0).clamp(0.5, 4.0);
        let to_points = |p: egui::Pos2| {
            egui::pos2(
                p.x * size[0] / client[0].max(1.0) / ppp,
                p.y * size[1] / client[1].max(1.0) / ppp,
            )
        };
        let queued = input::take();
        let events = if layers.menu.is_some() {
            queued
        } else {
            Vec::new()
        };
        let events = events
            .into_iter()
            .map(|e| match e {
                egui::Event::PointerMoved(p) => egui::Event::PointerMoved(to_points(p)),
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers,
                } => egui::Event::PointerButton {
                    pos: to_points(pos),
                    button,
                    pressed,
                    modifiers,
                },
                e => e,
            })
            .collect();
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] / ppp, size[1] / ppp),
            )),
            max_texture_side: Some(8192),
            time: Some(self.step.map_or_else(
                || self.start.elapsed().as_secs_f64(),
                |s| s * f64::from(self.frames),
            )),
            modifiers: input::modifiers(),
            events,
            focused: input::focused(),
            ..Default::default()
        };
        raw.viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(ppp);
        self.frames += 1;
        let ui = &mut self.ui;
        let out = self.egui.run_ui(raw, |root| {
            let ctx = root.ctx().clone();
            if let Some((d, corner)) = layers.hud {
                hud::hud(&ctx, d, corner);
            }
            if let Some((snap, backend)) = layers.menu.as_mut() {
                ui.show(&ctx, snap, &mut **backend);
            }
            if let Some((t, now)) = layers.toasts {
                t.show(&ctx, now);
            }
        });
        egui_directx11::split_output(out).0
    }
}

// ------------------------------------------------------------------ game backend --

struct GameBackend {
    /// Names that are not static cvars (permanent: static cvars exist from start).
    missing: HashSet<String>,
    /// Benchmark started from the panel: close the overlay after this frame.
    close: bool,
}

impl ui::Backend for GameBackend {
    fn cvar(&mut self, name: &str) -> Option<cvars::Info> {
        if name.is_empty() || self.missing.contains(name) {
            return None;
        }
        let r = cvars::info(name);
        if r.is_none() {
            self.missing.insert(name.to_owned());
        }
        r
    }

    fn set_settings(&mut self, s: Settings) {
        let cvars_changed = s.cvars != settings::file().cvars;
        let before = settings::pending_restart();
        settings::edit(|f| *f = s);
        if cvars_changed {
            cvars::set_overrides(&settings::get().cvars);
        }
        let new: Vec<String> = settings::pending_restart()
            .into_iter()
            .filter(|k| !before.contains(k))
            .collect();
        if !new.is_empty() {
            notify::warn(format!(
                "applies at the next start (save settings first): {}",
                new.join(", ")
            ));
        }
    }

    fn set_patch(&mut self, id: &str, on: bool) -> Result<(), String> {
        patches::set_enabled(id, on)
    }

    fn fsr_toggle(&mut self) {
        fsr::toggle();
    }

    fn fsr_debug_set(&mut self, view: u8) {
        fsr::set_debug_view(view);
    }

    fn bench_toggle(&mut self) -> Result<(), String> {
        if !settings::get().profile {
            return Err("profiler not installed (profile = 0 at startup)".into());
        }
        let requests = hook::last_requests();
        if requests == 0 {
            return Err("no rendered frame yet".into());
        }
        bench::toggle(requests);
        self.close = !bench::running();
        Ok(())
    }

    fn trace_arm(&mut self) -> bool {
        settings::get().profile && trace::arm()
    }

    fn save(&mut self) -> Result<String, String> {
        let r = settings::save().map(|p| p.display().to_string());
        match &r {
            Ok(p) => notify::info(format!("settings saved to {p}")),
            Err(e) => notify::error(format!("settings not saved: {e}")),
        }
        r
    }

    fn photo_action(&mut self, a: crate::photo::Action) {
        crate::photo::action(a);
    }
}

/// One presented frame: when, interval since the previous one (ms), engine CPU and GPU
/// frame times (ms; read only while something is drawn).
struct FrameRec {
    at: Instant,
    interval_ms: f32,
    cpu_ms: Option<f32>,
    gpu_ms: Option<f32>,
}

fn game_snapshot(frames: &VecDeque<FrameRec>) -> ui::Snapshot {
    let now = Instant::now();
    let file = settings::file();
    ui::Snapshot {
        exe: crate::exe_status().to_owned(),
        frames: frames
            .iter()
            .map(|f| (-(now - f.at).as_secs_f64(), f.interval_ms, f.gpu_ms))
            .collect(),
        fsr: fsr::status(),
        ao: crate::ao::status(),
        contact: crate::contact::status(),
        ssr: crate::ssr::status(),
        photo: crate::photo::status(),
        texmods: texmods::status(),
        overrides: cvars::overrides(),
        patches: patches::status()
            .into_iter()
            .map(|(p, st)| ui::PatchRow {
                id: p.id,
                name: p.name,
                description: p.description,
                source: p.source,
                when: p.when,
                on: patches::enabled(&file, p),
                state: st.text(),
            })
            .collect(),
        bench_running: bench::running(),
        bench_last: bench::last_dir().map(|d| d.display().to_string()),
        log: log::recent(),
        pending: settings::pending_restart(),
        active: settings::get(),
        file,
    }
}

// --------------------------------------------------------------------- game path --

struct Overlay {
    core: Option<Core>,
    painter: Option<Painter>,
    backend: GameBackend,
    hwnd: isize,
    key_held: bool,
    hud_key_held: bool,
    last_present: Option<Instant>,
    frames: VecDeque<FrameRec>,
    toasts: hud::Toasts,
    lock: LockFn,
    unlock: LockFn,
    failed: bool,
    /// The AO / contact-shadow / reflection / photo present work failed once (logged; not
    /// retried).
    debug_failed: [bool; 5],
}

static OVERLAY: Mutex<Option<Overlay>> = Mutex::new(None);

fn engine_fn(rva: u64, head: &[u8; 16]) -> Result<LockFn, String> {
    let addr = mem::image_base().ok_or("no image base")? + rva;
    let mut b = [0u8; 16];
    if !mem::read_into(addr, &mut b) || b != *head {
        return Err(format!(
            "bytes {b:02x?} at {addr:#x} differ from the analysed build"
        ));
    }
    // SAFETY: verified start of the engine function `void f(idRenderSystemLocal*)`.
    Ok(unsafe { std::mem::transmute::<u64, LockFn>(addr) })
}

/// Installs the cursor hooks and prepares the per-frame state (after `hook::install_present`
/// and `mouse::install`).
pub fn install() -> Result<(), String> {
    let lock = engine_fn(LOCK_RVA, &LOCK_HEAD).map_err(|e| format!("RS_LockImmediate: {e}"))?;
    let unlock =
        engine_fn(UNLOCK_RVA, &UNLOCK_HEAD).map_err(|e| format!("RS_UnlockImmediate: {e}"))?;
    input::install_cursor_hooks()?;
    *OVERLAY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Overlay {
        core: None,
        painter: None,
        backend: GameBackend {
            missing: HashSet::new(),
            close: false,
        },
        hwnd: 0,
        key_held: false,
        hud_key_held: false,
        last_present: None,
        frames: VecDeque::new(),
        toasts: hud::Toasts::default(),
        lock,
        unlock,
        failed: false,
        debug_failed: [false; 5],
    });
    INSTALLED.store(true, Ordering::Relaxed);
    let s = settings::get();
    log::line(&format!(
        "overlay: {} toggles the overlay, {} the performance HUD",
        settings::key_name(s.overlay_key),
        settings::key_name(s.hud_key)
    ));
    Ok(())
}

fn set_visible(ov: &mut Overlay, on: bool) {
    if VISIBLE.swap(on, Ordering::Relaxed) == on {
        return;
    }
    if on {
        if ov.hwnd != 0
            && let Err(e) = input::subclass(HWND(ov.hwnd as *mut c_void))
        {
            log::line(&format!("overlay: {e}; no mouse/keyboard input"));
        }
        crate::mouse::set_blocked(true);
        input::on_open();
    } else {
        crate::mouse::set_blocked(false);
        input::on_close();
    }
    log::line(if on {
        "overlay opened"
    } else {
        "overlay closed"
    });
}

/// Main thread, before the engine present (`hook.rs`). `swap_chain` is the
/// `IDXGISwapChain*` the engine presents to this frame (0: no present). A panic hides
/// the overlay for good and gives the input back to the game.
pub(crate) fn before_present(rs: u64, swap_chain: u64) {
    if !INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    if std::panic::catch_unwind(|| frame(rs, swap_chain)).is_err() {
        INSTALLED.store(false, Ordering::Relaxed);
        if VISIBLE.swap(false, Ordering::Relaxed) {
            crate::mouse::set_blocked(false);
            input::on_close();
        }
        log::line("overlay: panic; overlay disabled");
    }
}

/// Releases the engine's immediate-context lock on every path (also on unwinding).
struct EngineLock {
    unlock: LockFn,
    rs: u64,
}

impl Drop for EngineLock {
    fn drop(&mut self) {
        // SAFETY: releases the lock taken in `draw` for the same render system.
        unsafe { (self.unlock)(self.rs) };
    }
}

fn frame(rs: u64, swap_chain: u64) {
    let mut g = match OVERLAY.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    let Some(ov) = g.as_mut() else { return };
    let now = Instant::now();
    let s = settings::get();
    // Engine timers only while something is drawn (checked reads cost a syscall each).
    let timers = visible() || s.hud;
    if let Some(prev) = ov.last_present.replace(now) {
        let us_to_ms = |us: u32| us as f32 / 1000.0;
        ov.frames.push_back(FrameRec {
            at: now,
            interval_ms: (now - prev).as_secs_f32() * 1000.0,
            cpu_ms: timers.then(profiler::engine_cpu_us).flatten().map(us_to_ms),
            gpu_ms: timers.then(profiler::engine_gpu_us).flatten().map(us_to_ms),
        });
        while ov
            .frames
            .front()
            .is_some_and(|f| (now - f.at).as_secs_f64() > HISTORY_S)
        {
            ov.frames.pop_front();
        }
    }
    let raw = swap_chain as *mut c_void;
    // SAFETY: `swap_chain` is the engine's live swap chain for this frame (or null).
    let sc = unsafe { IDXGISwapChain::from_raw_borrowed(&raw) };
    if ov.hwnd == 0
        && let Some(sc) = sc
    {
        // SAFETY: plain getter on a live swap chain.
        if let Ok(desc) = unsafe { sc.GetDesc() } {
            ov.hwnd = desc.OutputWindow.0 as isize;
        }
    }
    // Toggle keys: edges, only while the game window has the focus.
    // SAFETY: plain query.
    let focus = ov.hwnd != 0 && unsafe { GetForegroundWindow() }.0 as isize == ov.hwnd;
    let down = ffi::key_down(s.overlay_key as i32);
    if down && !ov.key_held && focus {
        let on = !visible();
        set_visible(ov, on);
    }
    ov.key_held = down;
    let down = ffi::key_down(s.hud_key as i32);
    if down && !ov.hud_key_held && focus {
        // A live setting: the active copy follows at once; saved with the settings.
        settings::edit(|f| f.hud = !f.hud);
    }
    ov.hud_key_held = down;
    let s = settings::get();
    if let Some((level, text)) = texmods::take_summary() {
        match level {
            notify::Level::Warn => notify::warn(text),
            _ => notify::info(text),
        }
    }
    let notes = notify::take();
    if s.notifications {
        ov.toasts.add(notes, now);
    } else {
        ov.toasts.clear();
    }
    let toasts = ov.toasts.retain(now);
    // Photo mode first (back-buffer capture of the game's frame for a save, the
    // `photo_display = overlay` preview), then `ao_debug` / `contact_debug` / `ssr_debug`:
    // our buffers full-screen, under the menu/HUD.
    for (on, view) in [
        (crate::photo::composite::present_wanted(), DebugView::Photo),
        (s.ao_debug, DebugView::Ao),
        (s.contact_debug, DebugView::Contact),
        (s.ssr_debug != 0, DebugView::Ssr),
        (fsr::debug_view() == fsr::DEBUG_MOTION, DebugView::FsrMotion),
    ] {
        if on
            && !ov.debug_failed[view as usize]
            && let Some(sc) = sc
            && let Err(e) = debug_view(ov, rs, sc, view)
        {
            log::line(&format!("{}: debug view failed: {e}", view.name()));
            ov.debug_failed[view as usize] = true;
        }
    }
    // While a benchmark segment records, the HUD and toasts are not drawn (toasts keep
    // their timers) unless `bench_show_overlay`; an explicitly opened menu still is.
    let quiet = profiler::is_recording() && !s.bench_show_overlay;
    let show_hud = s.hud && !quiet;
    let show_toasts = toasts && !quiet;
    // Nothing on screen: no egui work at all.
    if ov.failed || !(visible() || show_hud || show_toasts) {
        return;
    }
    let Some(sc) = sc else { return };
    if let Err(e) = draw(ov, rs, sc, &s, now, show_hud, show_toasts) {
        log::line(&format!("overlay: {e}; overlay disabled"));
        ov.failed = true;
        set_visible(ov, false);
    }
    if std::mem::take(&mut ov.backend.close) {
        set_visible(ov, false);
    }
}

/// HUD numbers from the frame ring.
fn hud_data(frames: &VecDeque<FrameRec>, now: Instant) -> hud::HudData {
    let fsr = fsr::status();
    let last = frames.back();
    hud::HudData {
        intervals: frames
            .iter()
            .map(|f| ((now - f.at).as_secs_f32(), f.interval_ms))
            .filter(|(age, _)| *age <= hud::GRAPH_S)
            .collect(),
        engine_cpu_ms: last.and_then(|f| f.cpu_ms),
        engine_gpu_ms: last.and_then(|f| f.gpu_ms),
        fsr_gpu_ms: (fsr.active && !fsr.failed && fsr.runtime.is_some())
            .then_some(fsr.gpu_ms)
            .flatten(),
    }
}

fn draw(
    ov: &mut Overlay,
    rs: u64,
    sc: &IDXGISwapChain,
    s: &Settings,
    now: Instant,
    show_hud: bool,
    show_toasts: bool,
) -> Result<(), String> {
    // SAFETY: COM getters on the engine's live swap chain.
    let (device, target) = unsafe {
        (
            sc.GetDevice::<ID3D11Device>()
                .map_err(|e| format!("swap chain device: {e}"))?,
            sc.GetBuffer::<ID3D11Texture2D>(0)
                .map_err(|e| format!("back buffer: {e}"))?,
        )
    };
    if ov
        .painter
        .as_ref()
        .is_none_or(|p| p.device().as_raw() != device.as_raw())
    {
        // New device: fresh egui context too (its textures live in the renderer).
        ov.painter = Some(Painter::new(&device)?);
        ov.core = Some(Core::new());
    }
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { target.GetDesc(&mut desc) };
    let size = [desc.Width as f32, desc.Height as f32];
    let client = input::client_size(HWND(ov.hwnd as *mut c_void)).unwrap_or(size);
    let snap = visible().then(|| game_snapshot(&ov.frames));
    let hud = show_hud.then(|| hud_data(&ov.frames, now));
    let (Some(core), Some(painter)) = (ov.core.as_mut(), ov.painter.as_mut()) else {
        return Ok(());
    };
    let layers = Layers {
        menu: snap
            .as_ref()
            .map(|sn| (sn, &mut ov.backend as &mut dyn ui::Backend)),
        hud: hud.as_ref().map(|d| (d, s.hud_position)),
        toasts: show_toasts.then_some((&ov.toasts, now)),
    };
    let output = core.frame(size, client, s.overlay_scale, layers);
    // SAFETY: plain getter.
    let ctx =
        unsafe { device.GetImmediateContext() }.map_err(|e| format!("immediate context: {e}"))?;
    // SAFETY: engine spin lock around our use of the immediate context (render system
    // object of this present call); released by the guard on every path.
    unsafe { (ov.lock)(rs) };
    let _lock = EngineLock {
        unlock: ov.unlock,
        rs,
    };
    painter.paint(&ctx, &target, &core.egui, output)
}

/// Full-screen debug views of our passes (index into `Overlay::debug_failed`).
#[derive(Clone, Copy)]
enum DebugView {
    Ao,
    Contact,
    Ssr,
    Photo,
    FsrMotion,
}

impl DebugView {
    fn name(self) -> &'static str {
        match self {
            Self::Ao => "ao",
            Self::Contact => "contact",
            Self::Ssr => "ssr",
            Self::Photo => "photo",
            Self::FsrMotion => "fsr motion",
        }
    }
}

/// Draws the last AO result (`ao_debug = 1`), contact-shadow visibility (`contact_debug =
/// 1`) or reflection view (`ssr_debug`) over the back buffer under the engine's
/// immediate-context lock, through a UNORM view (raw values); photo mode's capture and
/// preview (`photo::composite::present`).
fn debug_view(ov: &Overlay, rs: u64, sc: &IDXGISwapChain, view: DebugView) -> Result<(), String> {
    // SAFETY: COM getters on the engine's live swap chain.
    let (device, target) = unsafe {
        (
            sc.GetDevice::<ID3D11Device>()
                .map_err(|e| format!("swap chain device: {e}"))?,
            sc.GetBuffer::<ID3D11Texture2D>(0)
                .map_err(|e| format!("back buffer: {e}"))?,
        )
    };
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { target.GetDesc(&mut desc) };
    let vd = paint::unorm_view_format(desc.Format).map(|format| D3D11_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
        },
    });
    let mut rtv = None;
    // SAFETY: view of the back buffer with a valid (or default) descriptor; released at
    // the end of this call.
    unsafe {
        device.CreateRenderTargetView(&target, vd.as_ref().map(|d| d as *const _), Some(&mut rtv))
    }
    .map_err(|e| format!("back-buffer view: {e}"))?;
    let rtv = rtv.ok_or("no back-buffer view")?;
    // SAFETY: plain getter.
    let ctx =
        unsafe { device.GetImmediateContext() }.map_err(|e| format!("immediate context: {e}"))?;
    // SAFETY: engine spin lock around our use of the immediate context (released by the
    // guard on every path).
    unsafe { (ov.lock)(rs) };
    let _lock = EngineLock {
        unlock: ov.unlock,
        rs,
    };
    let size = [desc.Width, desc.Height];
    match view {
        DebugView::Ao => crate::ao::draw_debug(&ctx, &rtv, size),
        DebugView::Contact => crate::contact::draw_debug(&ctx, &rtv, size),
        DebugView::Ssr => crate::ssr::draw_debug(&ctx, &rtv, size),
        DebugView::Photo => crate::photo::composite::present(&ctx, &target, &rtv, size),
        DebugView::FsrMotion => crate::fsr::draw_debug(&ctx, &rtv, size),
    }
}

// --------------------------------------------------------------- offline preview --

/// One rendered preview frame (RGBA8, sRGB-encoded like the game's back buffer).
pub struct PreviewImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Result of [`offline_preview`].
pub struct Preview {
    pub images: Vec<PreviewImage>,
    pub report: String,
}

mod preview;
pub use preview::offline_preview;
