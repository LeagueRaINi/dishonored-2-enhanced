//! The overlay's panels. All game state comes in through a [`Snapshot`] and all actions go
//! out through [`Backend`], so the offline preview runs the same code with stand-in data.

use std::sync::Arc;

use egui::{Color32, RichText, Ui};
use egui_plot::{Line, Plot, PlotPoints};

use crate::cvars;
use crate::fsr;
use crate::patches::Apply;
use crate::settings::{self, Corner, Settings};
use crate::texmods;

mod photo;

/// One registered patch for the panel.
#[derive(Clone, Debug)]
pub struct PatchRow {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub source: &'static str,
    pub when: Apply,
    /// Switched on in the settings file.
    pub on: bool,
    pub state: String,
}

/// Everything the panels show, gathered once per visible frame.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// Executable check ("analysed build (b671f824...)").
    pub exe: String,
    /// Last ~5 s: (seconds before now, CPU frame interval ms, engine GPU frame ms).
    pub frames: Vec<(f64, f32, Option<f32>)>,
    pub fsr: fsr::Status,
    /// Own AO in place of HBAO+ (`ao.rs`).
    pub ao: crate::ao::Status,
    /// Sun contact shadows (`contact.rs`).
    pub contact: crate::contact::Status,
    /// Screen-space reflections (`ssr.rs`).
    pub ssr: crate::ssr::Status,
    /// Photo mode, its probe and the ray-tracing scene (`photo/`, `rt/`).
    pub photo: crate::photo::Status,
    pub texmods: texmods::Status,
    /// Registered cvar overrides: name, value, writes, not a static cvar.
    pub overrides: Vec<(String, f32, u32, bool)>,
    pub patches: Vec<PatchRow>,
    pub bench_running: bool,
    pub bench_last: Option<String>,
    pub log: Vec<String>,
    /// Settings as edited (saved by "Save settings").
    pub file: Settings,
    /// Settings in effect.
    pub active: Arc<Settings>,
    /// Restart-only keys whose edited value is not in effect yet.
    pub pending: Vec<String>,
}

/// Actions and lookups the panels need.
pub trait Backend {
    fn cvar(&mut self, name: &str) -> Option<cvars::Info>;
    /// Replaces the edited settings (live keys take effect at once).
    fn set_settings(&mut self, s: Settings);
    fn set_patch(&mut self, id: &str, on: bool) -> Result<(), String>;
    fn fsr_toggle(&mut self);
    fn fsr_debug_set(&mut self, view: u8);
    fn bench_toggle(&mut self) -> Result<(), String>;
    fn trace_arm(&mut self) -> bool;
    /// Writes the settings file; returns its path.
    fn save(&mut self) -> Result<String, String>;
    /// Photo mode: enter/leave, save.
    fn photo_action(&mut self, a: crate::photo::Action);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    Status,
    Fsr,
    Lighting,
    Reflections,
    Photo,
    Cvars,
    Patches,
    TextureMods,
    Profiler,
    Log,
    Settings,
}

const TABS: [(Tab, &str); 11] = [
    (Tab::Status, "Status"),
    (Tab::Fsr, "FSR"),
    (Tab::Lighting, "Lighting"),
    (Tab::Reflections, "Reflections"),
    (Tab::Photo, "Photo"),
    (Tab::Cvars, "Cvars"),
    (Tab::Patches, "Patches"),
    (Tab::TextureMods, "Texture mods"),
    (Tab::Profiler, "Profiler"),
    (Tab::Log, "Log"),
    (Tab::Settings, "Overlay"),
];

/// Engine cvars worth trying (docs/lighting-options.md table A): name, Ultra value,
/// suggested values.
const CANDIDATES: [(&str, f32, &[f32]); 4] = [
    ("image_anisotropy", 1.0, &[1.0, 4.0, 8.0, 16.0]),
    ("r_lodZoomMultiplicator", 0.6, &[0.6, 1.0, 2.0]),
    ("r_shadowCullDistance", 75.0, &[75.0, 100.0, 150.0]),
    ("r_maxShadowMapResolution", 1024.0, &[1024.0, 2048.0]),
];

const LIVE: Color32 = Color32::from_rgb(91, 207, 190);
const RESTART: Color32 = Color32::from_rgb(234, 185, 105);
const BAD: Color32 = Color32::from_rgb(236, 116, 116);
const BACKGROUND: Color32 = Color32::from_rgb(17, 22, 29);
const PANEL: Color32 = Color32::from_rgb(24, 31, 40);
const RAISED: Color32 = Color32::from_rgb(30, 39, 50);
const BORDER: Color32 = Color32::from_rgb(48, 62, 75);
const MUTED: Color32 = Color32::from_rgb(139, 157, 173);

/// Panel state kept between frames.
#[derive(Default)]
pub struct State {
    pub tab: Tab,
    lookup: String,
    lookup_result: Option<Result<cvars::Info, String>>,
    add_name: String,
    add_value: String,
    key_text: Option<String>,
    hud_key_text: Option<String>,
    photo_key_text: Option<String>,
    photo_dir_text: Option<String>,
    message: String,
    /// Last runtime numbers (the runtime is skipped while submission holds it).
    fsr_runtime: Option<fsr::RuntimeStatus>,
    /// Screen rectangles (points) of the tabs and a few widgets from the last frame, for
    /// synthetic input in the offline preview.
    pub probes: Vec<(&'static str, egui::Rect)>,
}

/// "live" / "restart" marker for a settings key, with the reason on hover.
fn tag(ui: &mut Ui, key: &str) {
    let Some(k) = settings::key_info(key) else {
        return;
    };
    let (text, color) = if k.live {
        ("live", LIVE)
    } else {
        ("restart", RESTART)
    };
    ui.label(RichText::new(text).small().color(color))
        .on_hover_text(k.why);
}

/// A small "restore default" button, shown only while the value differs from its default;
/// true when clicked.
fn restore_icon(ui: &mut Ui, changed: bool, default: &str) -> bool {
    if !changed {
        return false;
    }
    let response = ui
        .add(egui::Button::new(RichText::new("↺").color(MUTED)).frame(false))
        .on_hover_text(format!("Restore default ({default})"));
    response.clicked()
}

/// Restore button for a settings key edited by another widget (combo boxes, radios).
fn restore(ui: &mut Ui, s: &mut Settings, key: &str) {
    let default = settings::default_value(key).unwrap_or_default();
    if restore_icon(ui, settings::is_changed(s, key), &default) {
        settings::restore_default(s, key);
    }
}

fn check(ui: &mut Ui, v: &mut bool, text: &str, key: &str) -> egui::Response {
    ui.horizontal(|ui| {
        let response = ui.checkbox(v, text);
        tag(ui, key);
        let default = settings::default_value(key).is_some_and(|d| d == "1");
        if restore_icon(ui, *v != default, if default { "on" } else { "off" }) {
            *v = default;
        }
        response
    })
    .inner
}

fn slider(ui: &mut Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str, key: &str) {
    ui.horizontal(|ui| {
        ui.add_sized([126.0, 28.0], egui::Label::new(text));
        ui.add(egui::Slider::new(v, range));
        tag(ui, key);
        if let Some(default) = settings::default_value(key).and_then(|d| d.parse::<f32>().ok())
            && restore_icon(ui, (*v - default).abs() > 1e-6, &default.to_string())
        {
            *v = default;
        }
    });
}

fn heading(ui: &mut Ui, text: &str) {
    ui.add_space(12.0);
    ui.label(RichText::new(text).size(15.0).strong());
    ui.add_space(2.0);
}

fn secondary(ui: &mut Ui, text: impl Into<String>) {
    ui.label(RichText::new(text.into()).small().color(MUTED));
}

fn card(ui: &mut Ui, title: &str, content: impl FnOnce(&mut Ui)) {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(8)
        .inner_margin(14)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).size(15.0).strong());
            ui.add_space(6.0);
            content(ui);
        });
    ui.add_space(10.0);
}

fn page_title(tab: Tab) -> (&'static str, &'static str) {
    match tab {
        Tab::Status => (
            "Overview",
            "Frame pacing, runtime health and active features.",
        ),
        Tab::Fsr => (
            "FSR upscaling",
            "Image reconstruction and quality tuning. F6 toggles FSR live.",
        ),
        Tab::Lighting => (
            "Lighting",
            "Ambient occlusion and sun shadows: replacements and A/B switches, all live.",
        ),
        Tab::Reflections => (
            "Reflections",
            "Screen-space reflections traced before TAA/FSR, all live.",
        ),
        Tab::Photo => (
            "Photo mode",
            "Screenshots of the paused game with ray-traced sun shadows, AO and reflections (or the path tracer).",
        ),
        Tab::Cvars => (
            "Engine variables",
            "Inspect engine values and manage persistent overrides.",
        ),
        Tab::Patches => (
            "Game patches",
            "Targeted fixes with explicit application state and evidence.",
        ),
        Tab::TextureMods => (
            "Texture mods",
            "Replacement files and runtime loading activity.",
        ),
        Tab::Profiler => (
            "Profiler",
            "Repeatable benchmarks and one-frame GPU traces.",
        ),
        Tab::Log => (
            "Activity log",
            "Recent diagnostic messages from void-inject.",
        ),
        Tab::Settings => (
            "Overlay",
            "Make the menu, performance HUD and notifications your own.",
        ),
    }
}

impl State {
    /// Builds the overlay for one frame.
    pub fn show(&mut self, ctx: &egui::Context, snap: &Snapshot, backend: &mut dyn Backend) {
        if let Some(r) = &snap.fsr.runtime {
            self.fsr_runtime = Some(r.clone());
        }
        self.probes.clear();
        let screen = ctx.content_rect();
        let maximum = (screen.size() - egui::vec2(64.0, 64.0)).max(egui::vec2(1.0, 1.0));
        egui::Window::new("void-inject")
            .default_pos([16.0, 16.0])
            .default_size(egui::vec2(848.0, 708.0).min(maximum))
            .min_size(egui::vec2(240.0, 180.0).min(maximum))
            .max_size(maximum)
            .title_bar(false)
            .frame(
                egui::Frame::new()
                    .fill(BACKGROUND)
                    .stroke(egui::Stroke::new(1.0, BORDER))
                    .corner_radius(12)
                    .inner_margin(16),
            )
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(RAISED)
                    .corner_radius(7)
                    .inner_margin(12)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("VOID").size(21.0).strong().color(LIVE));
                            ui.label(RichText::new("INJECT").size(21.0).strong());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    secondary(
                                        ui,
                                        format!(
                                            "v{}  /  {}",
                                            env!("CARGO_PKG_VERSION"),
                                            settings::key_name(snap.active.overlay_key)
                                        ),
                                    );
                                },
                            );
                        });
                    });
                ui.add_space(6.0);
                let (area, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
                let footer_height = 76.0_f32.min(area.height() * 0.4);
                let footer_rect = egui::Rect::from_min_max(
                    egui::pos2(area.left(), area.bottom() - footer_height),
                    area.max,
                );
                let body = egui::Rect::from_min_max(
                    area.min,
                    egui::pos2(area.right(), footer_rect.top() - 10.0),
                );
                let sidebar_width = 148.0_f32.min(body.width() * 0.32);
                let sidebar =
                    egui::Rect::from_min_size(body.min, egui::vec2(sidebar_width, body.height()));
                let content = egui::Rect::from_min_max(
                    egui::pos2(sidebar.right() + 18.0, body.top()),
                    body.max,
                );
                let mut file = snap.file.clone();
                // Tab buttons shrink (32 -> 22 px) so the whole list fits the menu height; below
                // that the list still scrolls with the wheel, without a scroll bar.
                let headers = 2.0 + 2.0 * 16.0 + 3.0 * (14.0 + 4.0);
                let gap = ui.spacing().item_spacing.y;
                let button_height =
                    ((body.height() - headers) / TABS.len() as f32 - gap).clamp(22.0, 32.0);
                ui.scope_builder(egui::UiBuilder::new().max_rect(sidebar), |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("navigation")
                        .max_height(body.height())
                        .auto_shrink([false, false])
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                        .show(ui, |ui| {
                            for (index, (tab, name)) in TABS.into_iter().enumerate() {
                                if let Some(group) = match index {
                                    0 => Some("WORKSPACE"),
                                    1 => Some("RENDERING"),
                                    7 => Some("TOOLS"),
                                    _ => None,
                                } {
                                    ui.add_space(if index == 0 { 2.0 } else { 16.0 });
                                    ui.label(RichText::new(group).size(10.0).strong().color(MUTED));
                                    ui.add_space(4.0);
                                }
                                let label = if tab == Tab::Status { "Overview" } else { name };
                                let response = ui.add_sized(
                                    [ui.available_width(), button_height],
                                    egui::Button::new(label).selected(self.tab == tab),
                                );
                                self.probes.push((name, response.rect));
                                if response.clicked() {
                                    self.tab = tab;
                                }
                            }
                        });
                });
                ui.scope_builder(egui::UiBuilder::new().max_rect(content), |ui| {
                    ui.set_clip_rect(content);
                    egui::ScrollArea::both()
                        .id_salt(("page", self.tab as u8))
                        .max_height(body.height())
                        .auto_shrink([false, false])
                        .stick_to_bottom(self.tab == Tab::Log)
                        .show(ui, |ui| {
                            ui.set_width((content.width() - 12.0).max(420.0));
                            let (title, subtitle) = page_title(self.tab);
                            ui.label(RichText::new(title).size(25.0).strong());
                            secondary(ui, subtitle);
                            ui.add_space(14.0);
                            match self.tab {
                                Tab::Status => self.status(ui, snap),
                                Tab::Fsr => self.fsr(ui, snap, &mut file, backend),
                                Tab::Lighting => {
                                    // Visible page area (fixed while scrolling; preview).
                                    self.probes.push(("lighting_page", ui.clip_rect()));
                                    lighting(ui, snap, &mut file, backend);
                                }
                                Tab::Reflections => reflections(ui, snap, &mut file),
                                Tab::Photo => photo::panel(
                                    ui,
                                    snap,
                                    &mut file,
                                    backend,
                                    &mut self.photo_key_text,
                                    &mut self.photo_dir_text,
                                    &mut self.probes,
                                ),
                                Tab::Cvars => self.cvars(ui, snap, &mut file, backend),
                                Tab::Patches => self.patches(ui, snap, backend),
                                Tab::TextureMods => texture_mods(ui, snap, &mut file),
                                Tab::Profiler => self.profiler(ui, snap, &mut file, backend),
                                Tab::Log => log_panel(ui, snap),
                                Tab::Settings => self.general(ui, &mut file),
                            }
                        });
                });
                if file != snap.file {
                    backend.set_settings(file);
                }
                ui.scope_builder(egui::UiBuilder::new().max_rect(footer_rect), |ui| {
                    ui.set_clip_rect(footer_rect);
                    self.footer(ui, snap, backend);
                });
            });
        draw_cursor(ctx);
    }

    fn footer(&mut self, ui: &mut Ui, snap: &Snapshot, backend: &mut dyn Backend) {
        ui.separator();
        ui.horizontal(|ui| {
            let save = ui.add(
                egui::Button::new(RichText::new("Save settings").strong().color(BACKGROUND))
                    .fill(LIVE)
                    .min_size(egui::vec2(124.0, 30.0)),
            );
            self.probes.push(("save_settings", save.rect));
            if save.clicked() {
                self.message = match backend.save() {
                    Ok(p) => format!("saved {p}"),
                    Err(e) => format!("save failed: {e}"),
                };
            }
            let (status, color) = if snap.pending.is_empty() {
                ("No restart required".to_owned(), MUTED)
            } else {
                (
                    format!("{} changes need restart", snap.pending.len()),
                    RESTART,
                )
            };
            ui.label(RichText::new(status).small().color(color))
                .on_hover_text(format!("Applies at restart: {}", snap.pending.join(", ")));
        });
        if self.message.is_empty() {
            secondary(
                ui,
                "Live changes apply immediately. Save to keep them next launch.",
            );
        } else {
            ui.add(egui::Label::new(RichText::new(&self.message).small().color(MUTED)).truncate())
                .on_hover_text(&self.message);
        }
    }

    fn status(&mut self, ui: &mut Ui, snap: &Snapshot) {
        let (total, count) = snap
            .frames
            .iter()
            .filter(|f| f.0 > -1.0)
            .fold((0.0_f32, 0_u32), |(sum, count), frame| {
                (sum + frame.1, count + 1)
            });
        let cpu = (count != 0).then(|| total / count as f32);
        let gpu = snap.frames.last().and_then(|f| f.2);
        ui.columns(3, |columns| {
            for (ui, (label, value, detail)) in columns.iter_mut().zip([
                (
                    "FRAME RATE",
                    cpu.filter(|v| *v > 0.0)
                        .map_or("—".into(), |v| format!("{:.0}", 1000.0 / v)),
                    "frames / second",
                ),
                (
                    "CPU INTERVAL",
                    cpu.map_or("—".into(), |v| format!("{v:.2}")),
                    "ms · 1 second mean",
                ),
                (
                    "ENGINE GPU",
                    gpu.map_or("—".into(), |v| format!("{v:.2}")),
                    "ms · latest frame",
                ),
            ]) {
                egui::Frame::new()
                    .fill(RAISED)
                    .corner_radius(8)
                    .inner_margin(12)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(label).size(10.0).strong().color(MUTED));
                        ui.label(RichText::new(value).size(28.0).strong().color(LIVE));
                        secondary(ui, detail);
                    });
            }
        });
        ui.add_space(12.0);
        let cpu_pts: Vec<[f64; 2]> = snap.frames.iter().map(|f| [f.0, f64::from(f.1)]).collect();
        let gpu_pts: Vec<[f64; 2]> = snap
            .frames
            .iter()
            .filter_map(|f| f.2.map(|g| [f.0, f64::from(g)]))
            .collect();
        card(ui, "Frame pacing", |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("CPU interval").small().color(LIVE));
                ui.label(RichText::new("Engine GPU").small().color(RESTART));
                secondary(ui, "Milliseconds · last 5 seconds");
            });
            Plot::new("frame_times")
                .height(150.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .include_x(-5.0)
                .include_x(0.0)
                .include_y(0.0)
                .show(ui, |p| {
                    p.line(
                        Line::new("CPU interval ms", PlotPoints::from(cpu_pts))
                            .color(LIVE)
                            .width(1.8),
                    );
                    p.line(
                        Line::new("engine GPU ms", PlotPoints::from(gpu_pts))
                            .color(RESTART)
                            .width(1.8),
                    );
                });
        });
        let f = &snap.fsr;
        let fsr = if !f.installed {
            "not installed (turn on fsr to install it now)".to_owned()
        } else if f.failed {
            "failed (engine TAA)".to_owned()
        } else if f.active {
            format!(
                "on{}",
                if snap.active.fsr_upscale {
                    format!(", upscale {}", snap.active.fsr_scale)
                } else {
                    ", native AA".to_owned()
                }
            )
        } else {
            "off (F6 or fsr = 0; hooks installed, inert; engine TAA)".to_owned()
        };
        let applied = snap.patches.iter().filter(|p| p.state == "applied").count();
        card(ui, "Active features", |ui| {
            egui::Grid::new("features").striped(true).show(ui, |ui| {
                for (k, v) in [
                    ("FSR 4", fsr),
                    (
                        "Ambient occlusion",
                        if !snap.active.ao_void {
                            "NVIDIA HBAO+".to_owned()
                        } else if snap.ao.hooked {
                            format!(
                                "own ({}{})",
                                crate::ao::quality_name(snap.active.ao_quality),
                                if snap.active.ao_half_res {
                                    ", half res"
                                } else {
                                    ""
                                }
                            )
                        } else {
                            "own requested, hook not installed".to_owned()
                        },
                    ),
                    (
                        "Texture mods",
                        if snap.texmods.installed {
                            format!(
                                "{} files, {} served",
                                snap.texmods.files,
                                snap.texmods.served.len()
                            )
                        } else {
                            "off".to_owned()
                        },
                    ),
                    (
                        "Profiler",
                        if snap.active.profile {
                            "installed (Ctrl+F3 benchmark, Shift+F3 trace)".to_owned()
                        } else {
                            "off".to_owned()
                        },
                    ),
                    (
                        "Cvar overrides",
                        format!("{} registered", snap.overrides.len()),
                    ),
                    (
                        "Patches",
                        format!("{applied} of {} applied", snap.patches.len()),
                    ),
                    ("Overlay key", settings::key_name(snap.active.overlay_key)),
                ] {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
        secondary(ui, &snap.exe);
    }

    fn fsr(&mut self, ui: &mut Ui, snap: &Snapshot, s: &mut Settings, backend: &mut dyn Backend) {
        let f = &snap.fsr;
        card(ui, "Reconstruction & quality", |ui| {
            ui.add_enabled_ui(f.installed, |ui| {
                ui.horizontal(|ui| {
                    let mut on = f.active;
                    let response = ui.checkbox(&mut on, "FSR on");
                    self.probes.push(("fsr_on", response.rect));
                    if response.changed() {
                        backend.fsr_toggle();
                    }
                    ui.label(RichText::new("F6").small().color(MUTED));
                    tag(ui, "fsr");
                });
            });
            if !f.installed {
                ui.colored_label(RESTART, "Enable FSR below to install it now.");
            }
            check(ui, &mut s.fsr, "Enable FSR in settings", "fsr")
                .on_hover_text("Disabling leaves installed hooks inert and restores engine TAA.");
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Reconstruction"));
                egui::ComboBox::from_id_salt("fsr_mode")
                    .selected_text(if s.fsr_upscale {
                        "Upscale"
                    } else {
                        "Native AA"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut s.fsr_upscale, false, "Native AA");
                        ui.selectable_value(&mut s.fsr_upscale, true, "Upscale");
                    });
                tag(ui, "fsr_mode");
                restore(ui, s, "fsr_mode");
            });
            slider(
                ui,
                &mut s.fsr_scale,
                0.33..=1.0,
                "Render scale",
                "fsr_scale",
            );
            slider(
                ui,
                &mut s.fsr_sharpness,
                0.0..=1.0,
                "Sharpness",
                "fsr_sharpness",
            );
            secondary(
                ui,
                "Render scale applies to upscale mode; native AA keeps full resolution.",
            );
        });
        card(ui, "Motion & transparency", |ui| {
            check(
                ui,
                &mut s.fsr_masks,
                "Reactive / transparency masks",
                "fsr_masks",
            );
            slider(
                ui,
                &mut s.fsr_reactive,
                0.0..=1.0,
                "Reactive scale",
                "fsr_reactive",
            );
            check(
                ui,
                &mut s.fsr_sky_motion,
                "Sky motion vectors",
                "fsr_sky_motion",
            );
            check(
                ui,
                &mut s.fsr_depth_resample,
                "Depth resampling",
                "fsr_depth_resample",
            )
            .on_hover_text("Resample depth when upscaling.");
        });
        card(ui, "Expert controls", |ui| {
            let expert = ui.collapsing("Sampling, exposure & submission", |ui| {
                ui.horizontal(|ui| {
                    let mut auto = s.fsr_jitter_phases.is_none();
                    ui.checkbox(&mut auto, "AMD default jitter phases");
                    let mut n = s.fsr_jitter_phases.unwrap_or(16);
                    if !auto {
                        ui.add(egui::Slider::new(&mut n, 4..=128).text("phases"));
                    }
                    s.fsr_jitter_phases = (!auto).then_some(n);
                    tag(ui, "fsr_jitter_phases");
                    restore(ui, s, "fsr_jitter_phases");
                });
                ui.horizontal(|ui| {
                    let mut auto = s.fsr_mip_bias.is_none();
                    ui.checkbox(&mut auto, "Automatic mip bias");
                    let mut bias = s.fsr_mip_bias.unwrap_or(-1.5);
                    if !auto {
                        ui.add(egui::Slider::new(&mut bias, -4.0..=4.0).text("bias"));
                    }
                    s.fsr_mip_bias = (!auto).then_some(bias);
                    tag(ui, "fsr_mip_bias");
                    restore(ui, s, "fsr_mip_bias");
                });
                let (bias, biased, tracked) = f.mip_bias;
                secondary(
                    ui,
                    if bias == 0.0 {
                        format!("Engine samplers: {tracked} tracked. Bias applies in upscale mode.")
                    } else {
                        format!("Sampler bias {bias:+.2}: {biased} of {tracked} tracked")
                    },
                );
                check(
                    ui,
                    &mut s.fsr_auto_exposure,
                    "Automatic exposure",
                    "fsr_auto_exposure",
                );
                check(ui, &mut s.fsr_flush, "Flush before FSR", "fsr_flush")
                    .on_hover_text("Off uses less CPU time, but FSR starts later.");
                ui.add_enabled_ui(f.installed, |ui| {
                    ui.horizontal(|ui| {
                        ui.add_sized([126.0, 28.0], egui::Label::new("Debug view"));
                        let mut debug = f.debug_view;
                        let name = |d: u8| {
                            fsr::DEBUG_VIEWS
                                .get(usize::from(d))
                                .copied()
                                .unwrap_or("off")
                        };
                        let combo = egui::ComboBox::from_id_salt("fsr_debug")
                            .selected_text(name(debug))
                            .show_ui(ui, |ui| {
                                for (i, n) in fsr::DEBUG_VIEWS.iter().enumerate() {
                                    let r = ui.selectable_value(&mut debug, i as u8, *n);
                                    if i == usize::from(fsr::DEBUG_MOTION) {
                                        self.probes.push(("fsr_debug_motion", r.rect));
                                    }
                                }
                            });
                        self.probes.push(("fsr_debug", combo.response.rect));
                        ui.label(egui::RichText::new("Shift+F6").weak());
                        if debug != f.debug_view {
                            backend.fsr_debug_set(debug);
                        }
                    });
                    if f.debug_view == fsr::DEBUG_MOTION {
                        ui.label(
                            egui::RichText::new(
                                "Motion vectors FSR receives, per frame in render pixels. Brightness: log scale, black = 0, bottom bar from 0.02 px (left) to 32 px (right), white ticks at 0.1, 1 and 10 px. Colour: direction (hue wheel), stronger with more motion.",
                            )
                            .weak(),
                        );
                    }
                });
            });
            self.probes
                .push(("fsr_expert", expert.header_response.rect));
        });
        card(ui, "Runtime", |ui| {
            egui::Grid::new("fsr_stats")
                .striped(true)
                .spacing([24.0, 8.0])
                .show(ui, |ui| {
                    let rt = self.fsr_runtime.as_ref();
                    let size = |v: [u32; 2]| format!("{} × {}", v[0], v[1]);
                    for (key, value) in [
                        ("Provider", rt.map_or("—".into(), |r| r.provider.clone())),
                        ("Display", rt.map_or("—".into(), |r| size(r.display))),
                        ("Render", rt.map_or("—".into(), |r| size(r.render))),
                        (
                            "GPU ms · rolling",
                            f.gpu_ms.map_or("—".into(), |g| format!("{g:.3}")),
                        ),
                        ("Frames", rt.map_or("—".into(), |r| r.frames.to_string())),
                        ("Resets", rt.map_or("—".into(), |r| r.resets.to_string())),
                        ("Camera cuts", rt.map_or("—".into(), |r| r.cuts.to_string())),
                        ("Sky vectors", f.sky.clone()),
                        (
                            "State",
                            if f.failed {
                                "failed".into()
                            } else if rt.is_some() {
                                "ready".into()
                            } else {
                                "not initialised".into()
                            },
                        ),
                    ] {
                        ui.label(RichText::new(key).color(MUTED));
                        ui.label(value);
                        ui.end_row();
                    }
                });
        });
    }

    fn cvars(&mut self, ui: &mut Ui, snap: &Snapshot, s: &mut Settings, backend: &mut dyn Backend) {
        card(ui, "Persistent overrides", |ui| {
            tag(ui, "cvar.");
            secondary(
                ui,
                "Written every 30 frames. Removing an override leaves its last value until the engine or console changes it.",
            );
            let mut remove = None;
            egui::Grid::new("overrides").striped(true).show(ui, |ui| {
                ui.label("name");
                ui.label("value");
                ui.label("engine int / float");
                ui.label("");
                ui.end_row();
                for (i, (name, value)) in s.cvars.iter_mut().enumerate() {
                    ui.label(name.as_str());
                    ui.add(egui::DragValue::new(value).speed(0.05));
                    let st = snap.overrides.iter().find(|o| o.0 == *name);
                    match (st, backend.cvar(name)) {
                        (Some(o), _) if o.3 => {
                            ui.label(RichText::new("no such static cvar").color(BAD));
                        }
                        (_, Some(info)) => {
                            ui.label(format!("{} / {}", info.int, info.float));
                        }
                        _ => {
                            ui.label("-");
                        }
                    }
                    if ui.small_button("remove").clicked() {
                        remove = Some(i);
                    }
                    ui.end_row();
                }
            });
            if let Some(i) = remove {
                s.cvars.remove(i);
            }
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.add_name)
                        .hint_text("cvar name")
                        .desired_width(220.0),
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.add_value)
                        .hint_text("value")
                        .desired_width(60.0),
                );
                if ui.button("Add").clicked() {
                    let name = self.add_name.trim().to_owned();
                    match self.add_value.trim().parse::<f32>() {
                        Ok(v)
                            if v.is_finite()
                                && !name.is_empty()
                                && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') =>
                        {
                            set_override(s, &name, v);
                            self.add_name.clear();
                            self.add_value.clear();
                        }
                        _ => self.message = "add: need a cvar name and a number".into(),
                    }
                }
            });
        });
        card(ui, "Suggested candidates", |ui| {
            secondary(
                ui,
                "Reference: docs/lighting-options.md · table A. Hover a name for its Ultra preset.",
            );
            egui::Grid::new("candidates").striped(true).show(ui, |ui| {
                for (name, ultra, values) in CANDIDATES {
                    ui.label(name)
                        .on_hover_text(format!("Ultra preset: {ultra}"));
                    let cur = backend
                        .cvar(name)
                        .map_or("-".to_owned(), |i| i.float.to_string());
                    ui.label(format!("now {cur}"));
                    ui.horizontal(|ui| {
                        let set = s.cvars.iter().find(|c| c.0 == name).map(|c| c.1);
                        for &v in values {
                            if ui.selectable_label(set == Some(v), v.to_string()).clicked() {
                                set_override(s, name, v);
                            }
                        }
                        if set.is_some() && ui.small_button("no override").clicked() {
                            s.cvars.retain(|c| c.0 != name);
                        }
                    });
                    ui.end_row();
                }
            });
        });
        card(ui, "Inspect a variable", |ui| {
            ui.horizontal(|ui| {
                let r = ui.add(
                    egui::TextEdit::singleline(&mut self.lookup)
                        .hint_text("cvar name")
                        .desired_width(240.0),
                );
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Look up").clicked() || enter {
                    let name = self.lookup.trim();
                    self.lookup_result = Some(
                        backend
                            .cvar(name)
                            .ok_or_else(|| format!("{name}: no such static cvar")),
                    );
                }
            });
            match &self.lookup_result {
                Some(Ok(i)) => {
                    egui::Grid::new("lookup").striped(true).show(ui, |ui| {
                        for (k, v) in [
                            ("name", i.name.clone()),
                            ("string", i.value.clone()),
                            ("int / float", format!("{} / {}", i.int, i.float)),
                            ("default", i.default.clone()),
                            ("flags", format!("{:#x}", i.flags)),
                        ] {
                            ui.label(k);
                            ui.label(v);
                            ui.end_row();
                        }
                    });
                    ui.label(RichText::new(&i.description).italics());
                }
                Some(Err(e)) => {
                    ui.label(RichText::new(e).color(BAD));
                }
                None => {}
            }
        });
    }

    fn patches(&mut self, ui: &mut Ui, snap: &Snapshot, backend: &mut dyn Backend) {
        secondary(
            ui,
            "Live patches apply or revert immediately. Other patches take effect at the next launch.",
        );
        for p in &snap.patches {
            card(ui, p.name, |ui| {
                ui.horizontal(|ui| {
                    let mut on = p.on;
                    if ui.checkbox(&mut on, "Enable patch").changed()
                        && let Err(e) = backend.set_patch(p.id, on)
                    {
                        self.message = format!("patch {}: {e}", p.id);
                    }
                    let color = if p.when == Apply::Live { LIVE } else { RESTART };
                    ui.label(RichText::new(p.when.label()).small().color(color));
                });
                let state_color = match p.state.as_str() {
                    "applied" => LIVE,
                    "off" | "not applied yet" => Color32::GRAY,
                    _ => BAD,
                };
                ui.label(RichText::new(format!("patch.{}: {}", p.id, p.state)).color(state_color));
                ui.label(RichText::new(p.description).small());
                ui.label(
                    RichText::new(format!("evidence: {}", p.source))
                        .small()
                        .italics(),
                );
            });
        }
    }

    fn profiler(
        &mut self,
        ui: &mut Ui,
        snap: &Snapshot,
        s: &mut Settings,
        backend: &mut dyn Backend,
    ) {
        card(ui, "Capture session", |ui| {
            check(ui, &mut s.profile, "Enable profiler hooks", "profile");
            let installed = snap.active.profile;
            if !installed {
                ui.label(
                    RichText::new("Profiler not installed (profile = 0 at startup).")
                        .color(RESTART),
                );
            }
            ui.add_enabled_ui(installed, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let text = if snap.bench_running {
                        "Abort benchmark"
                    } else {
                        "Start benchmark"
                    };
                    if ui
                        .button(text)
                        .on_hover_text("Ctrl+F3 · starting a benchmark closes the overlay.")
                        .clicked()
                    {
                        self.message = match backend.bench_toggle() {
                            Ok(()) => "benchmark toggled".into(),
                            Err(e) => e,
                        };
                    }
                    if ui
                        .button("Arm one-frame trace")
                        .on_hover_text("Shift+F3 · captures the next frame.")
                        .clicked()
                    {
                        self.message = if backend.trace_arm() {
                            "trace armed for the next frame".into()
                        } else {
                            "trace busy".into()
                        };
                    }
                });
            });
            ui.label(format!(
                "benchmark: {}; last run: {}",
                if snap.bench_running {
                    "running"
                } else {
                    "idle"
                },
                snap.bench_last.as_deref().unwrap_or("-")
            ));
            secondary(ui, "Ctrl+F3 benchmark  ·  Shift+F3 one-frame trace");
        });
        card(ui, "Benchmark sequence", |ui| {
            slider(
                ui,
                &mut s.bench_warmup_s,
                3.0..=60.0,
                "Warm-up · seconds",
                "bench_warmup_s",
            );
            slider(
                ui,
                &mut s.bench_static_s,
                5.0..=120.0,
                "Static · seconds",
                "bench_static_s",
            );
            slider(
                ui,
                &mut s.bench_pan_s,
                5.0..=120.0,
                "Pan · seconds",
                "bench_pan_s",
            );
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Repeats"));
                ui.add(egui::Slider::new(&mut s.bench_repeats, 1..=10));
                tag(ui, "bench_repeats");
            });
            check(
                ui,
                &mut s.bench_uncap,
                "Frame limiter off during runs",
                "bench_uncap",
            );
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Run label"));
                ui.add(egui::TextEdit::singleline(&mut s.bench_label).desired_width(200.0));
                tag(ui, "bench_label");
            });
        });
        card(ui, "GPU instrumentation", |ui| {
            check(ui, &mut s.profile_gpu, "GPU timestamps", "profile_gpu");
            ui.horizontal(|ui| {
                ui.radio_value(&mut s.profile_gpu_passes, true, "per pass class");
                ui.radio_value(&mut s.profile_gpu_passes, false, "per command list");
                tag(ui, "profile_gpu_mode");
                restore(ui, s, "profile_gpu_mode");
            });
            check(
                ui,
                &mut s.profile_labels,
                "Command-list labels",
                "profile_labels",
            );
        });
    }

    fn general(&mut self, ui: &mut Ui, s: &mut Settings) {
        ui.spacing_mut().item_spacing.y = 6.0;
        card(ui, "Menu & scale", |ui| {
            ui.horizontal(|ui| {
                check(ui, &mut s.overlay, "Enable overlay", "overlay");
                ui.label("Toggle key");
                let text = self
                    .key_text
                    .get_or_insert_with(|| settings::key_name(s.overlay_key));
                let response = ui.add(egui::TextEdit::singleline(text).desired_width(70.0));
                if response.changed()
                    && let Some(vk) = settings::parse_key(text)
                {
                    s.overlay_key = vk;
                }
                if response.lost_focus() {
                    self.key_text = None;
                }
                tag(ui, "overlay_key");
            });
            ui.horizontal(|ui| {
                let mut auto = s.overlay_scale.is_none();
                ui.checkbox(&mut auto, "Automatic scale");
                let mut value = s.overlay_scale.unwrap_or(1.0);
                if !auto {
                    ui.add(egui::Slider::new(&mut value, 0.5..=4.0).text("scale"));
                }
                s.overlay_scale = (!auto).then_some(value);
                tag(ui, "overlay_scale");
                restore(ui, s, "overlay_scale");
            });
        });
        card(ui, "Performance HUD", |ui| {
            let response = check(ui, &mut s.hud, "Show performance HUD", "hud")
                .on_hover_text("The HUD remains visible when this menu is closed.");
            self.probes.push(("hud_on", response.rect));
            ui.horizontal(|ui| {
                ui.label("Key");
                let text = self
                    .hud_key_text
                    .get_or_insert_with(|| settings::key_name(s.hud_key));
                let response = ui.add(egui::TextEdit::singleline(text).desired_width(70.0));
                if response.changed()
                    && let Some(vk) = settings::parse_key(text)
                {
                    s.hud_key = vk;
                }
                if response.lost_focus() {
                    self.hud_key_text = None;
                }
                tag(ui, "hud_key");
                ui.add_space(6.0);
                ui.label("Position");
                egui::ComboBox::from_id_salt("hud_position")
                    .width(104.0)
                    .selected_text(s.hud_position.name())
                    .show_ui(ui, |ui| {
                        for corner in Corner::ALL {
                            ui.selectable_value(&mut s.hud_position, corner, corner.name());
                        }
                    });
                tag(ui, "hud_position");
                restore(ui, s, "hud_position");
            });
        });
        card(ui, "Notifications & benchmarks", |ui| {
            let response = check(
                ui,
                &mut s.notifications,
                "Show notifications",
                "notifications",
            )
            .on_hover_text("Status toasts also appear when the menu is closed.");
            self.probes.push(("notifications_on", response.rect));
            check(
                ui,
                &mut s.bench_show_overlay,
                "Show HUD and toasts during benchmarks",
                "bench_show_overlay",
            );
            check(
                ui,
                &mut s.diagnostics,
                "Diagnostics (for bug reports; costs frame time)",
                "diagnostics",
            )
            .on_hover_text(
                "FSR value check and motion-vector statistics, FSR timing report and a \
                 per-second frame summary in the log.",
            );
        });
    }
}

fn set_override(s: &mut Settings, name: &str, v: f32) {
    match s.cvars.iter_mut().find(|c| c.0 == name) {
        Some(c) => c.1 = v,
        None => s.cvars.push((name.to_owned(), v)),
    }
}

fn lighting(ui: &mut Ui, snap: &Snapshot, s: &mut Settings, backend: &mut dyn Backend) {
    card(ui, "Ambient occlusion", |ui| {
        ui.horizontal(|ui| {
            ui.add_sized([126.0, 28.0], egui::Label::new("Technique"));
            egui::ComboBox::from_id_salt("ao_mode")
                .selected_text(if s.ao_void {
                    "Own (visibility bitmask)"
                } else {
                    "NVIDIA HBAO+ (game)"
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.ao_void, false, "NVIDIA HBAO+ (game)");
                    ui.selectable_value(&mut s.ao_void, true, "Own (visibility bitmask)");
                });
            tag(ui, "ao_mode");
            restore(ui, s, "ao_mode");
        });
        ui.add_enabled_ui(s.ao_void, |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Quality"));
                egui::ComboBox::from_id_salt("ao_quality")
                    .selected_text(crate::ao::quality_name(s.ao_quality))
                    .show_ui(ui, |ui| {
                        for (i, (name, slices, steps)) in crate::ao::QUALITIES.iter().enumerate() {
                            ui.selectable_value(
                                &mut s.ao_quality,
                                i as u8,
                                format!("{name} ({slices} x {steps})"),
                            );
                        }
                    });
                tag(ui, "ao_quality");
                restore(ui, s, "ao_quality");
            });
            check(ui, &mut s.ao_half_res, "Half resolution", "ao_half_res")
                .on_hover_text("Depth-aware upsampling; cheaper than HBAO+ at native resolution.");
            slider(
                ui,
                &mut s.ao_radius_scale,
                0.25..=4.0,
                "Radius scale",
                "ao_radius_scale",
            );
            slider(ui, &mut s.ao_strength, 0.0..=4.0, "Strength", "ao_strength");
            check(ui, &mut s.ao_debug, "Show AO buffer", "ao_debug")
                .on_hover_text("Full-screen view of our AO (drawn before the present).");
        });
    });
    let a = &snap.ao;
    card(ui, "Runtime", |ui| {
        egui::Grid::new("ao_runtime").striped(true).show(ui, |ui| {
            let hook = if a.hooked {
                "RenderAO hooked".to_owned()
            } else if a.waiting {
                "waiting for the engine's HBAO+ context".to_owned()
            } else {
                "not installed (select the own technique to install)".to_owned()
            };
            for (k, v) in [
                ("Hook", hook),
                ("RenderAO calls", a.calls.to_string()),
                ("Rendered by us", a.ours.to_string()),
                ("Handed to HBAO+", a.fallbacks.to_string()),
            ] {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            }
        });
        if let Some(e) = &a.error {
            ui.add(egui::Label::new(RichText::new(e).small().color(BAD)).wrap());
        }
        if let Some(i) = &a.info {
            egui::CollapsingHeader::new("First call (struct fields)")
                .id_salt("ao_info")
                .show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(i).small().color(MUTED)).wrap());
                });
        }
    });
    contact_shadows(ui, snap, s);
    edge_halo(ui, s);
    shadow_ab(ui, snap, s, backend);
}

/// The game's DOF-combine sharpening (`dof_sharpen.rs`): bright outline around silhouettes.
fn edge_halo(ui: &mut Ui, s: &mut Settings) {
    card(ui, "Edge halo (game sharpening)", |ui| {
        slider(
            ui,
            &mut s.dof_sharpen,
            0.0..=1.0,
            "Sharpening",
            "dof_sharpen",
        );
        let (hooked, coef) = crate::dof_sharpen::status();
        ui.label(
            RichText::new(match (hooked, coef) {
                (true, Some(c)) => format!(
                    "Game coefficient {c:.2}, applied {:.2}. 0 removes the bright outline; 1 = game.",
                    c * s.dof_sharpen
                ),
                (true, None) => "Hook installed; waiting for the next frame.".to_owned(),
                _ => "1 = game (no hook). Lower it to remove the bright outline around characters and fences."
                    .to_owned(),
            })
            .weak(),
        );
    });
}

/// Engine cvar of the cascade time slicing (O: read per frame as `dword_1432F8B98` in
/// `sub_14032B280` / `sub_14032EE50` / `sub_140330630`).
const PARTIAL_SLICE_CVAR: &str = "r_shadowParallelPartialSliceUpdate";
const SHADOW_PATCH: &str = "shadows_submit_early";

fn contact_shadows(ui: &mut Ui, snap: &Snapshot, s: &mut Settings) {
    card(ui, "Sun contact shadows", |ui| {
        check(
            ui,
            &mut s.contact_shadows,
            "Contact shadows",
            "contact_shadows",
        )
        .on_hover_text(
            "Short screen-space rays toward the sun, combined with the engine's shadow map (min).",
        );
        ui.add_enabled_ui(s.contact_shadows, |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Quality"));
                egui::ComboBox::from_id_salt("contact_quality")
                    .selected_text(crate::contact::quality_name(s.contact_quality))
                    .show_ui(ui, |ui| {
                        for (i, (name, steps)) in crate::contact::QUALITIES.iter().enumerate() {
                            ui.selectable_value(
                                &mut s.contact_quality,
                                i as u8,
                                format!("{name} ({steps} steps)"),
                            );
                        }
                    });
                tag(ui, "contact_quality");
                restore(ui, s, "contact_quality");
            });
            slider(
                ui,
                &mut s.contact_length,
                0.05..=4.0,
                "Ray length (m)",
                "contact_length",
            );
            slider(
                ui,
                &mut s.contact_thickness,
                0.01..=2.0,
                "Thickness (m)",
                "contact_thickness",
            );
            check(
                ui,
                &mut s.contact_debug,
                "Show contact buffer",
                "contact_debug",
            )
            .on_hover_text("Full-screen view of the contact visibility (before the present).");
        });
        let c = &snap.contact;
        egui::Grid::new("contact_runtime")
            .striped(true)
            .show(ui, |ui| {
                for (k, v) in [
                    (
                        "Hook",
                        if c.hooked {
                            "SCREENSHADOWS callback hooked".to_owned()
                        } else {
                            "not installed (turn contact shadows on to install)".to_owned()
                        },
                    ),
                    ("Passes / applied", format!("{} / {}", c.calls, c.ours)),
                    ("Skipped (no sun shadow map)", c.skipped.to_string()),
                    ("Failures", c.failures.to_string()),
                ] {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                }
            });
        if let Some(e) = &c.error {
            ui.add(egui::Label::new(RichText::new(e).small().color(BAD)).wrap());
        }
        if let Some(i) = &c.info {
            egui::CollapsingHeader::new("First pass (inputs)")
                .id_salt("contact_info")
                .show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(i).small().color(MUTED)).wrap());
                });
        }
    });
}

/// Shadow-flicker A/B: cascade time slicing (cvar override) and the shadow submit patch.
fn shadow_ab(ui: &mut Ui, snap: &Snapshot, s: &mut Settings, backend: &mut dyn Backend) {
    card(ui, "Shadow map A/B", |ui| {
        let set = s
            .cvars
            .iter()
            .find(|c| c.0 == PARTIAL_SLICE_CVAR)
            .map(|c| c.1);
        let engine = backend.cvar(PARTIAL_SLICE_CVAR).map(|i| i.int);
        let mut on = set.map_or(engine.unwrap_or(1) != 0, |v| v != 0.0);
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut on, "Time-sliced cascade updates")
                .on_hover_text(
                    "r_shadowParallelPartialSliceUpdate (Ultra: on): cascades 1-3 render on alternate frames. Off renders every cascade every frame.",
                )
                .changed()
            {
                set_override(s, PARTIAL_SLICE_CVAR, if on { 1.0 } else { 0.0 });
            }
            tag(ui, "cvar.");
            if restore_icon(ui, !on, "on (engine Ultra)") {
                set_override(s, PARTIAL_SLICE_CVAR, 1.0);
            }
        });
        secondary(
            ui,
            format!(
                "engine value now: {} (override re-applied every 30 frames; saved as cvar.{PARTIAL_SLICE_CVAR})",
                engine.map_or("unknown".to_owned(), |v| v.to_string())
            ),
        );
        if let Some(p) = snap.patches.iter().find(|p| p.id == SHADOW_PATCH) {
            let mut on = p.on;
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut on, p.name)
                    .on_hover_text(p.description)
                    .changed()
                    && let Err(e) = backend.set_patch(p.id, on)
                {
                    ui.colored_label(BAD, e);
                }
                let (text, color) = if p.when == Apply::Live {
                    ("live", LIVE)
                } else {
                    ("restart", RESTART)
                };
                ui.label(RichText::new(text).small().color(color))
                    .on_hover_text(p.when.label());
            });
            secondary(ui, format!("patch state: {}", p.state));
        }
    });
}

/// Screen-space reflections (`ssr.rs`, docs/reflections.md).
fn reflections(ui: &mut Ui, snap: &Snapshot, s: &mut Settings) {
    card(ui, "Screen-space reflections", |ui| {
        check(ui, &mut s.ssr, "Reflections", "ssr").on_hover_text(
            "Traced in the frame before TAA/FSR and blended by Fresnel; misses keep the game's own reflections.",
        );
        secondary(
            ui,
            "Material source auto: normal, roughness and reflectance per pixel from the game's lit shaders; the surface values below apply to pixels without material data.",
        );
        ui.add_enabled_ui(s.ssr, |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Quality"));
                egui::ComboBox::from_id_salt("ssr_quality")
                    .selected_text(crate::ssr::quality_name(s.ssr_quality))
                    .show_ui(ui, |ui| {
                        for (i, (name, steps)) in crate::ssr::QUALITIES.iter().enumerate() {
                            ui.selectable_value(
                                &mut s.ssr_quality,
                                i as u8,
                                format!("{name} ({steps} steps)"),
                            );
                        }
                    });
                tag(ui, "ssr_quality");
                restore(ui, s, "ssr_quality");
            });
            check(ui, &mut s.ssr_half_res, "Half resolution", "ssr_half_res")
                .on_hover_text("Trace at half resolution; resolve and blend at full resolution.");
            slider(
                ui,
                &mut s.ssr_strength,
                0.0..=2.0,
                "Strength",
                "ssr_strength",
            );
            slider(
                ui,
                &mut s.ssr_roughness,
                0.0..=0.6,
                "Roughness",
                "ssr_roughness",
            );
            slider(
                ui,
                &mut s.ssr_reflectance,
                0.0..=1.0,
                "Reflectance (F0)",
                "ssr_reflectance",
            );
            slider(
                ui,
                &mut s.ssr_thickness,
                0.05..=4.0,
                "Thickness (m)",
                "ssr_thickness",
            );
            check(
                ui,
                &mut s.ssr_depth_floors_only,
                "Floors only (no material)",
                "ssr_depth_floors_only",
            )
            .on_hover_text(
                "Pixels without material data reflect only on upward-facing planar surfaces (floors, streets, tables); pixels with material data reflect on any orientation by their own roughness and reflectance.",
            );
            check(ui, &mut s.ssr_temporal, "Temporal accumulation", "ssr_temporal")
                .on_hover_text(
                    "Accumulates the reflections over frames (motion-vector reprojection, history clamp, per-frame rotated samples, roughness-scaled blur): much less noise on glossy surfaces. Also denoises the ray-traced reflections the same way.",
                );
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Material source"));
                egui::ComboBox::from_id_salt("ssr_material")
                    .selected_text(if s.ssr_material { "auto" } else { "depth" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut s.ssr_material, true, "auto (game shaders)");
                        ui.selectable_value(&mut s.ssr_material, false, "depth (values above)");
                    });
                tag(ui, "ssr_material");
                restore(ui, s, "ssr_material");
            })
            .response
            .on_hover_text(
                "auto: normal, roughness and reflectance per pixel from the game's lit shaders (patched when the game starts with reflections on); the values above where none.",
            );
            check(ui, &mut s.ssr_skip_water, "Skip water", "ssr_skip_water")
                .on_hover_text("Water has its own reflections (mask from the water depth target).");
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Debug view"));
                egui::ComboBox::from_id_salt("ssr_debug")
                    .selected_text(crate::ssr::debug_name(s.ssr_debug))
                    .show_ui(ui, |ui| {
                        for (i, name) in crate::ssr::DEBUG_VIEWS.iter().enumerate() {
                            ui.selectable_value(&mut s.ssr_debug, i as u8, *name);
                        }
                    });
                tag(ui, "ssr_debug");
                restore(ui, s, "ssr_debug");
            });
        });
        let r = &snap.ssr;
        egui::Grid::new("ssr_runtime").striped(true).show(ui, |ui| {
            for (k, v) in [
                (
                    "Hook",
                    if r.hooked || snap.fsr.installed {
                        "TAA call site redirected".to_owned()
                    } else {
                        "not installed (turn reflections on to install)".to_owned()
                    },
                ),
                ("TAA frames seen", r.calls.to_string()),
                (
                    "Frames applied",
                    format!(
                        "{} (engine TAA {}, before FSR {})",
                        r.applied, r.engine, r.submit
                    ),
                ),
                (
                    "Skipped",
                    format!(
                        "{} compiling, {} failed (inputs/D3D11)",
                        r.not_ready, r.failures
                    ),
                ),
                (
                    "Water mask",
                    match r.water_coverage {
                        Some(c) => format!(
                            "{:.1} % of the region; ignored in {} frames (> 60 %)",
                            c * 100.0,
                            r.water_fallback
                        ),
                        None => "no read-back yet (FSR path or a debug view)".to_owned(),
                    },
                ),
                (
                    "Temporal history",
                    if s.ssr_temporal {
                        format!("{} restarts (first frame, gaps, resizes)", r.history_resets)
                    } else {
                        "off".to_owned()
                    },
                ),
                ("Material target", {
                    let m = &r.material;
                    if m.patching {
                        format!(
                            "{} shaders patched ({} excluded: vegetation, hair, decals), {} unpatched, {} rejected; bound {} times{}",
                            m.patched,
                            m.excluded,
                            m.unpatched,
                            m.create_fallback,
                            m.bound,
                            if m.hooked {
                                ""
                            } else {
                                " (forward hooks missing)"
                            }
                        )
                    } else {
                        "off (start the game with reflections on and material auto)".to_owned()
                    }
                }),
            ] {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            }
        });
        if let Some(e) = &r.error {
            ui.add(egui::Label::new(RichText::new(e).small().color(BAD)).wrap());
        }
        if let Some(i) = &r.info {
            egui::CollapsingHeader::new("First frame (inputs)")
                .id_salt("ssr_info")
                .show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(i).small().color(MUTED)).wrap());
                });
        }
    });
    card(ui, "Ray-traced reflections", |ui| {
        check(
            ui,
            &mut s.rt_reflections,
            "Ray-traced reflections",
            "rt_reflections",
        )
        .on_hover_text(
            "Rays (DXR inline, the ray-tracing scene of the Photo tab) where the screen-space reflections found nothing; needs Reflections, the ray-tracing scene (rt) and FSR.",
        );
        if let Some(b) = &snap.ssr.rt.blocker {
            ui.add(
                egui::Label::new(RichText::new(format!("Not active: needs {b}")).color(BAD)).wrap(),
            );
            ui.horizontal(|ui| {
                if !s.ssr && ui.button("Turn on Reflections").clicked() {
                    s.ssr = true;
                }
                if !s.rt && ui.button("Turn on the ray-tracing scene (rt)").clicked() {
                    s.rt = true;
                }
                if !s.fsr && ui.button("Turn on FSR").clicked() {
                    s.fsr = true;
                }
            });
        }
        ui.add_enabled_ui(s.rt_reflections, |ui| {
            slider(
                ui,
                &mut s.rt_max_roughness,
                0.05..=1.0,
                "Max roughness",
                "rt_max_roughness",
            );
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Ray budget (%)"));
                ui.add(egui::Slider::new(&mut s.rt_ray_budget, 1..=100));
                tag(ui, "rt_ray_budget");
                restore(ui, s, "rt_ray_budget");
            })
            .response
            .on_hover_text("Rays per frame as a share of the half-resolution grid; beyond it an even subset is traced and neighbours fill in.");
            ui.horizontal(|ui| {
                ui.add_sized([126.0, 28.0], egui::Label::new("Debug view"));
                egui::ComboBox::from_id_salt("rt_reflect_debug")
                    .selected_text(crate::ssr::rt_reflect::debug_name(s.rt_reflect_debug))
                    .show_ui(ui, |ui| {
                        for (i, name) in crate::ssr::rt_reflect::DEBUG_VIEWS.iter().enumerate() {
                            ui.selectable_value(&mut s.rt_reflect_debug, i as u8, *name);
                        }
                    });
                tag(ui, "rt_reflect_debug");
                restore(ui, s, "rt_reflect_debug");
            });
        });
        let r = &snap.ssr.rt;
        let l = &r.last;
        egui::Grid::new("rt_reflect_runtime")
            .striped(true)
            .show(ui, |ui| {
                for (k, v) in [
                    (
                        "Frames traced",
                        format!(
                            "{} (skipped: {} without scene, {} ring busy, {} failed)",
                            r.frames, r.no_scene, r.busy, r.failures
                        ),
                    ),
                    (
                        "Rays",
                        format!(
                            "{} requested, {} traced, {} hits ({} on screen), {} misses",
                            l.requested, l.traced, l.hits, l.screen_hits, l.misses
                        ),
                    ),
                    (
                        "Hit shading scale",
                        if l.scale > 0.0 {
                            format!(
                                "{:.2} (stored / absolute; screen tiles {:.2}..{:.2}, {} samples)",
                                l.scale, l.tile_scale[0], l.tile_scale[1], l.calib_samples
                            )
                        } else {
                            "not calibrated yet (no on-screen hits)".to_owned()
                        },
                    ),
                    (
                        "Denoiser",
                        if l.denoised {
                            "on (with Temporal accumulation)".to_owned()
                        } else {
                            "off (Temporal accumulation off or no motion vectors)".to_owned()
                        },
                    ),
                    (
                        "GPU",
                        r.gpu_ms.map_or_else(
                            || "-".to_owned(),
                            |ms| format!("{ms:.3} ms on FSR's list"),
                        ),
                    ),
                ] {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                }
            });
        if let Some(e) = &r.error {
            ui.add(egui::Label::new(RichText::new(e).small().color(BAD)).wrap());
        }
    });
}

fn texture_mods(ui: &mut Ui, snap: &Snapshot, s: &mut Settings) {
    card(ui, "Mod loading", |ui| {
        check(
            ui,
            &mut s.texture_mods,
            "Enable texture mods",
            "texture_mods",
        );
        ui.horizontal(|ui| {
            ui.label("Directory");
            ui.add(egui::TextEdit::singleline(&mut s.mods_dir).desired_width(280.0));
            tag(ui, "mods_dir");
        });
    });
    let t = &snap.texmods;
    if !t.installed {
        ui.label("OpenFileRead hook not installed (off, or no mod files at startup).");
        return;
    }
    card(ui, "File activity", |ui| {
        ui.label(format!(
            "{} files in {}",
            t.files,
            t.dir
                .as_ref()
                .map_or("-".into(), |d| d.display().to_string())
        ));
        heading(ui, &format!("Served ({})", t.served.len()));
        for n in &t.served {
            ui.monospace(n);
        }
        heading(ui, &format!("Rejected ({})", t.rejected.len()));
        for (n, why) in &t.rejected {
            ui.label(RichText::new(format!("{n}: {why}")).color(BAD));
        }
    });
}

fn log_panel(ui: &mut Ui, snap: &Snapshot) {
    egui::Frame::new()
        .fill(PANEL)
        .corner_radius(8)
        .inner_margin(14)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            if snap.log.is_empty() {
                secondary(ui, "No diagnostic messages yet.");
            }
            for line in &snap.log {
                ui.label(RichText::new(line).monospace().small());
            }
        });
}

/// Arrow cursor at the pointer (the game hides the OS cursor).
fn draw_cursor(ctx: &egui::Context) {
    let Some(p) = ctx.input(|i| i.pointer.latest_pos()) else {
        return;
    };
    let s = 18.0;
    let pts = vec![p, p + egui::vec2(0.0, s), p + egui::vec2(s * 0.7, s * 0.7)];
    ctx.layer_painter(egui::LayerId::new(
        egui::Order::Debug,
        egui::Id::new("void_cursor"),
    ))
    .add(egui::Shape::convex_polygon(
        pts,
        Color32::WHITE,
        egui::Stroke::new(1.5, Color32::BLACK),
    ));
}
