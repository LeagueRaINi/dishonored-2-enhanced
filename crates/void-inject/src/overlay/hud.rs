//! Always-on layers drawn also while the overlay menu is closed: the performance HUD and
//! the notification toasts. Both are non-interactive egui areas (no input is captured).

use std::time::{Duration, Instant};

use egui::{Align2, Color32, CornerRadius, Frame, Margin, RichText, Stroke, pos2, vec2};

use crate::notify::{Level, Note};
use crate::settings::Corner;

/// Frame-time history shown by the HUD graph and used for its averages.
pub const GRAPH_S: f32 = 3.0;
/// Window for FPS, average and 1% low.
pub const STATS_S: f32 = 1.0;
/// Graph y range (ms): fixed so the 8.3 / 16.7 ms lines stay put.
const GRAPH_MAX_MS: f32 = 25.0;
const FADE_S: f32 = 0.6;

const PANEL: Color32 = Color32::from_rgb(24, 31, 40);
const BORDER: Color32 = Color32::from_rgb(48, 62, 75);
const TEXT: Color32 = Color32::from_rgb(225, 233, 239);
const MUTED: Color32 = Color32::from_rgb(139, 157, 173);
const ACCENT: Color32 = Color32::from_rgb(91, 207, 190);
const HUD_WIDTH: f32 = 264.0;

/// HUD numbers for one frame.
#[derive(Clone, Debug, Default)]
pub struct HudData {
    /// Frame intervals (ms, measured at the engine present) of the last `GRAPH_S`
    /// seconds, oldest first; the last `STATS_S` seconds give FPS, mean and 1% low.
    pub intervals: Vec<(f32, f32)>,
    /// Engine timers (ms): frame minus swap (CPU) and the GPU query ring.
    pub engine_cpu_ms: Option<f32>,
    pub engine_gpu_ms: Option<f32>,
    /// FSR dispatch GPU time (rolling) while FSR runs.
    pub fsr_gpu_ms: Option<f32>,
}

/// FPS, mean frame time (ms) and the 1% low (99th percentile frame time, ms) over the
/// intervals of the last `STATS_S` seconds (`(age s, ms)`).
pub fn stats(intervals: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    let mut v: Vec<f32> = intervals
        .iter()
        .filter(|(age, _)| *age <= STATS_S)
        .map(|(_, ms)| *ms)
        .collect();
    if v.is_empty() {
        return None;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    v.sort_by(f32::total_cmp);
    let p99 = v[((v.len() as f32 - 1.0) * 0.99).round() as usize];
    Some((1000.0 / mean.max(0.001), mean, p99))
}

fn timer(p: &egui::Painter, pos: egui::Pos2, value: Option<f32>) {
    let font = egui::FontId::monospace(13.0);
    match value {
        Some(value) => {
            p.text(pos, Align2::RIGHT_TOP, format!("{value:.2} ms"), font, TEXT);
        }
        None => {
            p.text(pos, Align2::RIGHT_TOP, "— ms", font, MUTED);
        }
    }
}

/// The performance HUD in `corner`.
pub fn hud(ctx: &egui::Context, d: &HudData, corner: Corner) {
    let (align, off) = match corner {
        Corner::TopLeft => (Align2::LEFT_TOP, vec2(10.0, 10.0)),
        Corner::TopRight => (Align2::RIGHT_TOP, vec2(-10.0, 10.0)),
        Corner::BottomLeft => (Align2::LEFT_BOTTOM, vec2(10.0, -10.0)),
        Corner::BottomRight => (Align2::RIGHT_BOTTOM, vec2(-10.0, -10.0)),
    };
    egui::Area::new(egui::Id::new("void_hud"))
        .anchor(align, off)
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            Frame::NONE
                .fill(PANEL)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::same(12))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = vec2(0.0, 8.0);
                    let summary_height = if d.fsr_gpu_ms.is_some() { 126.0 } else { 106.0 };
                    let (rect, _) = ui
                        .allocate_exact_size(vec2(HUD_WIDTH, summary_height), egui::Sense::hover());
                    let p = ui.painter();
                    let at = |x, y| rect.min + vec2(x, y);
                    let label = |x, y, text| {
                        p.text(
                            at(x, y),
                            Align2::LEFT_TOP,
                            text,
                            egui::FontId::proportional(11.0),
                            MUTED,
                        );
                    };
                    label(0.0, 0.0, "PERFORMANCE");
                    p.text(
                        at(HUD_WIDTH, 0.0),
                        Align2::RIGHT_TOP,
                        "1s window",
                        egui::FontId::proportional(11.0),
                        MUTED,
                    );
                    let numbers = stats(&d.intervals);
                    let fps_font = egui::FontId::monospace(34.0);
                    match numbers {
                        Some((fps, _, _)) => {
                            p.text(
                                at(0.0, 22.0),
                                Align2::LEFT_TOP,
                                format!("{fps:.0}"),
                                fps_font,
                                TEXT,
                            );
                        }
                        None => {
                            p.text(at(0.0, 22.0), Align2::LEFT_TOP, "—", fps_font, MUTED);
                        }
                    }
                    label(1.0, 62.0, "FRAMES / SECOND");
                    label(144.0, 29.0, "1% low FPS");
                    match numbers {
                        Some((_, _, p99)) => {
                            p.text(
                                at(HUD_WIDTH, 27.0),
                                Align2::RIGHT_TOP,
                                format!("{:.0}", 1000.0 / p99.max(0.001)),
                                egui::FontId::monospace(14.0),
                                TEXT,
                            );
                        }
                        None => {
                            p.text(
                                at(HUD_WIDTH, 27.0),
                                Align2::RIGHT_TOP,
                                "—",
                                egui::FontId::monospace(14.0),
                                MUTED,
                            );
                        }
                    }
                    label(144.0, 54.0, "Mean");
                    timer(p, at(HUD_WIDTH, 52.0), numbers.map(|(_, mean, _)| mean));
                    p.hline(rect.x_range(), at(0.0, 82.0).y, Stroke::new(1.0, BORDER));
                    label(0.0, 92.0, "CPU");
                    timer(p, at(124.0, 90.0), d.engine_cpu_ms);
                    label(144.0, 92.0, "GPU");
                    timer(p, at(HUD_WIDTH, 90.0), d.engine_gpu_ms);
                    if let Some(fsr) = d.fsr_gpu_ms {
                        label(0.0, 112.0, "FSR GPU");
                        timer(p, at(HUD_WIDTH, 110.0), Some(fsr));
                    }
                    graph(ui, &d.intervals);
                });
        });
}

/// Frame-time graph of the last `GRAPH_S` seconds with 8.3 / 16.7 ms lines.
fn graph(ui: &mut egui::Ui, intervals: &[(f32, f32)]) {
    let (rect, _) = ui.allocate_exact_size(vec2(HUD_WIDTH, 100.0), egui::Sense::hover());
    // Reuse the area's painter and clip rectangle, allowing the renderer to batch the
    // card, labels and graph without a separate clipped graph mesh.
    let p = ui.painter();
    p.text(
        rect.min,
        Align2::LEFT_TOP,
        "FRAME TIME",
        egui::FontId::proportional(11.0),
        MUTED,
    );
    p.text(
        rect.right_top(),
        Align2::RIGHT_TOP,
        "0–25 ms · 3s",
        egui::FontId::proportional(11.0),
        MUTED,
    );
    let frame = egui::Rect::from_min_max(rect.min + vec2(0.0, 20.0), rect.max);
    p.rect_filled(frame, 6.0, Color32::from_rgb(17, 22, 29));
    p.rect_stroke(
        frame,
        6.0,
        Stroke::new(1.0, BORDER),
        egui::StrokeKind::Inside,
    );
    let plot = egui::Rect::from_min_max(frame.min + vec2(8.0, 7.0), frame.max - vec2(52.0, 18.0));
    let y = |ms: f32| plot.bottom() - (ms / GRAPH_MAX_MS).clamp(0.0, 1.0) * plot.height();
    for (ms, label) in [(8.33, "8.3 ms"), (16.67, "16.7 ms")] {
        p.hline(plot.x_range(), y(ms), Stroke::new(1.0, BORDER));
        p.text(
            pos2(frame.right() - 7.0, y(ms)),
            Align2::RIGHT_CENTER,
            label,
            egui::FontId::monospace(10.0),
            MUTED,
        );
    }
    p.text(
        pos2(plot.left(), frame.bottom() - 4.0),
        Align2::LEFT_BOTTOM,
        "−3s",
        egui::FontId::proportional(10.0),
        MUTED,
    );
    p.text(
        pos2(plot.right(), frame.bottom() - 4.0),
        Align2::RIGHT_BOTTOM,
        "now",
        egui::FontId::proportional(10.0),
        MUTED,
    );
    let pts: Vec<egui::Pos2> = intervals
        .iter()
        .filter(|(age, _)| *age <= GRAPH_S)
        .map(|&(age, ms)| pos2(plot.right() - age / GRAPH_S * plot.width(), y(ms)))
        .collect();
    match pts.len() {
        0 => {
            p.text(
                plot.center(),
                Align2::CENTER_CENTER,
                "Waiting for samples",
                egui::FontId::proportional(11.0),
                MUTED,
            );
        }
        1 => {
            p.circle_filled(pts[0], 1.5, ACCENT);
        }
        _ => {
            p.add(egui::Shape::line(pts, Stroke::new(1.5, ACCENT)));
        }
    }
}

/// A toast on screen: when it appeared and how long it stays.
pub struct Toast {
    pub level: Level,
    pub text: String,
    pub shown: Instant,
    pub ttl: Duration,
    /// Live status line (`notify::status`): updated in place.
    pub key: Option<&'static str>,
}

/// Toasts currently on screen (newest last).
#[derive(Default)]
pub struct Toasts {
    list: Vec<Toast>,
}

const MAX_TOASTS: usize = 6;

impl Toasts {
    /// Adds drained notes; their lifetime starts now (notes queued before the first
    /// frame, e.g. at startup, are still shown in full). A note with a key replaces the
    /// toast of that key in place (an empty text removes it).
    pub fn add(&mut self, notes: Vec<Note>, now: Instant) {
        for n in notes {
            if let Some(k) = n.key {
                let at = self.list.iter().position(|t| t.key == Some(k));
                if n.text.is_empty() {
                    if let Some(i) = at {
                        self.list.remove(i);
                    }
                    continue;
                }
                if let Some(i) = at {
                    let t = &mut self.list[i];
                    t.level = n.level;
                    t.text = n.text;
                    t.shown = now;
                    t.ttl = n.ttl;
                    continue;
                }
            }
            self.list.push(Toast {
                level: n.level,
                text: n.text,
                shown: now,
                ttl: n.ttl,
                key: n.key,
            });
        }
        let excess = self.list.len().saturating_sub(MAX_TOASTS);
        self.list.drain(..excess);
    }

    /// Drops expired toasts; true while any remains.
    pub fn retain(&mut self, now: Instant) -> bool {
        self.list.retain(|t| now.duration_since(t.shown) < t.ttl);
        !self.list.is_empty()
    }

    pub fn clear(&mut self) {
        self.list.clear();
    }

    /// Draws the toasts stacked in the top-right corner, fading out at the end.
    pub fn show(&self, ctx: &egui::Context, now: Instant) {
        if self.list.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::new("void_toasts"))
            .anchor(Align2::RIGHT_TOP, vec2(-10.0, 10.0))
            .order(egui::Order::Tooltip)
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_max_width(420.0);
                for t in self.list.iter().rev() {
                    let left = t.ttl.as_secs_f32() - now.duration_since(t.shown).as_secs_f32();
                    let alpha = (left / FADE_S).clamp(0.0, 1.0);
                    let (color, tag) = match t.level {
                        Level::Info => (ACCENT, "INFO"),
                        Level::Warn => (Color32::from_rgb(234, 185, 105), "WARNING"),
                        Level::Error => (Color32::from_rgb(236, 116, 116), "ERROR"),
                    };
                    ui.scope(|ui| {
                        ui.multiply_opacity(alpha);
                        Frame::NONE
                            .fill(PANEL)
                            .stroke(Stroke::new(1.0, BORDER))
                            .corner_radius(CornerRadius::same(10))
                            .inner_margin(Margin::symmetric(12, 10))
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing.y = 5.0;
                                ui.horizontal(|ui| {
                                    let (mark, _) = ui
                                        .allocate_exact_size(vec2(3.0, 11.0), egui::Sense::hover());
                                    ui.painter().rect_filled(mark, 1.5, color);
                                    ui.label(RichText::new(tag).size(11.0).strong().color(color));
                                });
                                ui.label(RichText::new(&t.text).color(TEXT));
                            });
                    });
                    ui.add_space(4.0);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_over_the_last_second() {
        // 118 frames at 8 ms and two at 30 ms within 1 s (the 1% low is the 99th
        // percentile frame); older frames ignored.
        let mut v: Vec<(f32, f32)> = (0..118).map(|i| (i as f32 * 0.008, 8.0)).collect();
        v.extend([(0.5, 30.0), (0.6, 30.0), (2.0, 100.0)]);
        let (fps, mean, p99) = stats(&v).unwrap();
        assert!((mean - (118.0 * 8.0 + 60.0) / 120.0).abs() < 1e-3);
        assert!((fps - 1000.0 / mean).abs() < 1e-3);
        assert_eq!(p99, 30.0);
        assert!(stats(&[(2.0, 8.0)]).is_none());
    }

    #[test]
    fn toasts_expire_and_are_bounded() {
        let t0 = Instant::now();
        let note = |level, ttl| Note {
            level,
            text: "x".into(),
            at: t0,
            ttl: Duration::from_secs(ttl),
            key: None,
        };
        let mut t = Toasts::default();
        t.add(vec![note(Level::Info, 4), note(Level::Error, 12)], t0);
        assert!(t.retain(t0 + Duration::from_secs(5)));
        assert_eq!(t.list.len(), 1);
        assert!(!t.retain(t0 + Duration::from_secs(13)));
        t.add((0..10).map(|_| note(Level::Warn, 8)).collect(), t0);
        assert_eq!(t.list.len(), MAX_TOASTS);
    }

    #[test]
    fn status_lines_update_in_place() {
        let t0 = Instant::now();
        let line = |text: &str, ttl| Note {
            level: Level::Info,
            text: text.into(),
            at: t0,
            ttl: Duration::from_secs(ttl),
            key: Some("photo"),
        };
        let mut t = Toasts::default();
        t.add(vec![line("sun 4/256", 3)], t0);
        t.add(vec![line("sun 8/256", 3)], t0 + Duration::from_secs(2));
        assert_eq!(t.list.len(), 1);
        assert_eq!(t.list[0].text, "sun 8/256");
        // Refreshed: still visible after the first line's ttl.
        assert!(t.retain(t0 + Duration::from_secs(4)));
        t.add(vec![line("", 0)], t0 + Duration::from_secs(4));
        assert!(!t.retain(t0 + Duration::from_secs(4)));
    }
}
