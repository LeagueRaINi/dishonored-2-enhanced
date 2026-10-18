//! Photo panel: photo mode (`photo/`), its read-only probe, and the ray-tracing scene
//! service (`rt/`).

use egui::{RichText, Ui};

use super::{
    BAD, Backend, LIVE, MUTED, Snapshot, card, check, restore, restore_icon, secondary, slider, tag,
};
use crate::photo::{Action, Display, Status, hybrid, probe};
use crate::settings::{self, Settings};

/// Texture edge choices for the photo scene.
const TEXTURE_SIZES: [u32; 5] = [128, 256, 512, 1024, 2048];

fn row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.add_sized([126.0, 28.0], egui::Label::new(label));
        add(ui);
    });
}

/// A u32 slider with the live tag and the restore icon.
fn slider_u32(
    ui: &mut Ui,
    v: &mut u32,
    range: std::ops::RangeInclusive<u32>,
    text: &str,
    key: &str,
    log: bool,
) {
    row(ui, text, |ui| {
        ui.add(egui::Slider::new(v, range).logarithmic(log));
        tag(ui, key);
        if let Some(default) = settings::default_value(key).and_then(|d| d.parse::<u32>().ok())
            && restore_icon(ui, *v != default, &default.to_string())
        {
            *v = default;
        }
    });
}

fn grid(ui: &mut Ui, id: &str, rows: Vec<(&str, String)>) {
    egui::Grid::new(id)
        .striped(true)
        .num_columns(2)
        .show(ui, |ui| {
            for (k, v) in rows {
                ui.label(RichText::new(k).color(MUTED));
                ui.label(v);
                ui.end_row();
            }
        });
}

fn bar(ui: &mut Ui, done: u32, target: u32, text: String) {
    let frac = if target > 0 {
        done as f32 / target as f32
    } else {
        0.0
    };
    ui.add(egui::ProgressBar::new(frac.clamp(0.0, 1.0)).text(text));
}

/// Hybrid display: per-term progress (rays per pixel), GPU ms, reflection samples.
fn hybrid_progress(ui: &mut Ui, h: &hybrid::Status, reflections: Option<u32>) {
    for (name, t, shown) in [
        ("Sun shadows", &h.sun, h.shown[0]),
        ("Ambient occlusion", &h.ao, h.shown[1]),
    ] {
        if t.enabled {
            bar(
                ui,
                t.rays,
                t.target,
                format!(
                    "{name}: {} / {} rays per pixel ({shown} shown), {:.1} ms per {} rays",
                    t.rays, t.target, t.ms_per_batch, t.rays_per_batch
                ),
            );
        } else {
            secondary(ui, format!("{name}: off"));
        }
    }
    secondary(
        ui,
        match reflections {
            Some(n) => format!("Reflections: {n} samples per texel"),
            None => "Reflections: off (see the toast / log)".to_owned(),
        },
    );
    secondary(
        ui,
        format!(
            "{} — {}x{}, depth prepass {:.1} ms, {} submits (longest {:.1} ms GPU)",
            h.phase, h.size[0], h.size[1], h.prep_ms, h.submits.0, h.submits.1
        ),
    );
}

/// Path-traced displays: samples per pixel and speed.
fn path_progress(ui: &mut Ui, st: &Status) {
    bar(
        ui,
        st.spp_done,
        st.spp_target,
        format!(
            "{} / {} samples per pixel — {}",
            st.spp_done, st.spp_target, st.phase
        ),
    );
    let pixels = f64::from(st.size[0]) * f64::from(st.size[1]);
    secondary(
        ui,
        if st.ms_per_spp > 0.0 {
            format!(
                "{:.1} ms per sample, {:.2} spp/s, {:.0} M samples/s",
                st.ms_per_spp,
                1000.0 / st.ms_per_spp,
                pixels * 1000.0 / f64::from(st.ms_per_spp) / 1e6
            )
        } else {
            "measuring".to_owned()
        },
    );
}

pub(super) fn panel(
    ui: &mut Ui,
    snap: &Snapshot,
    s: &mut Settings,
    backend: &mut dyn Backend,
    key_text: &mut Option<String>,
    dir_text: &mut Option<String>,
    probes: &mut Vec<(&'static str, egui::Rect)>,
) {
    let st = &snap.photo;
    card(ui, "Photo mode", |ui| {
        ui.horizontal(|ui| {
            check(ui, &mut s.photo, "Photo mode", "photo").on_hover_text(
                "Off: no hook, no patch, no thread. On: the hotkey pauses the game through the engine's own Ansel session and path-traces the static scene.",
            );
            ui.label("Hotkey");
            let text = key_text.get_or_insert_with(|| settings::key_name(s.photo_key));
            let response = ui.add(egui::TextEdit::singleline(text).desired_width(70.0));
            if response.changed()
                && let Some(vk) = settings::parse_key(text)
            {
                s.photo_key = vk;
            }
            if response.lost_focus() {
                *key_text = None;
            }
            tag(ui, "photo_key");
            restore(ui, s, "photo_key");
        });
        secondary(
            ui,
            "In photo mode: W/A/S/D move, Q/E down/up, mouse looks, Shift fast, Ctrl slow, R back to the start, Enter saves, Esc or the hotkey leaves.",
        );
        ui.add_enabled_ui(snap.active.photo, |ui| {
            ui.horizontal(|ui| {
                let label = if st.active { "Leave photo mode" } else { "Enter photo mode" };
                let toggle = ui.button(label);
                probes.push(("photo_toggle", toggle.rect));
                if toggle.clicked() {
                    backend.photo_action(Action::Toggle);
                }
                if ui
                    .add_enabled(st.active, egui::Button::new("Save photo"))
                    .on_hover_text("PNG of the game frame (engine post applied) plus our own PNG and the raw HDR")
                    .clicked()
                {
                    backend.photo_action(Action::Save);
                }
            });
        });
        if st.active {
            match &st.hybrid {
                Some(h) => hybrid_progress(ui, h, st.reflections),
                None => path_progress(ui, st),
            }
            grid(
                ui,
                "photo_run",
                vec![
                    ("Pause", st.pause.clone()),
                    (
                        "Free camera",
                        if st.free_camera {
                            "yes"
                        } else {
                            "no (fixed view)"
                        }
                        .into(),
                    ),
                    ("Image", format!("{}x{}", st.size[0], st.size[1])),
                    ("Running for", format!("{:.0} s", st.seconds)),
                    (
                        "Paused frames",
                        st.rates.map_or("measuring".into(), |r| r.describe()),
                    ),
                ],
            );
        }
        if let Some(e) = &st.last_error {
            ui.label(RichText::new(format!("Last error: {e}")).color(BAD));
        }
        if let Some(m) = &st.last_save {
            secondary(ui, m.clone());
        }
    });
    card(ui, "Photo settings", |ui| {
        row(ui, "Display", |ui| {
            egui::ComboBox::from_id_salt("photo_display")
                .selected_text(s.photo_display.name())
                .show_ui(ui, |ui| {
                    for (d, text) in [
                        (
                            Display::Hybrid,
                            "hybrid (game frame + ray-traced shadows, AO, reflections)",
                        ),
                        (
                            Display::PathTrace,
                            "pathtrace (debug: our image before the fog pass)",
                        ),
                        (Display::Overlay, "overlay (our image, own tone mapping)"),
                    ] {
                        ui.selectable_value(&mut s.photo_display, d, text);
                    }
                });
            tag(ui, "photo_display");
            restore(ui, s, "photo_display");
        });
        slider(
            ui,
            &mut s.photo_budget_ms,
            1.0..=30.0,
            "GPU ms / frame",
            "photo_budget_ms",
        );
        row(ui, "Texture size", |ui| {
            egui::ComboBox::from_id_salt("photo_texture_size")
                .selected_text(format!("{} texels", s.photo_texture_size))
                .show_ui(ui, |ui| {
                    for t in TEXTURE_SIZES {
                        ui.selectable_value(&mut s.photo_texture_size, t, format!("{t} texels"));
                    }
                });
            tag(ui, "photo_texture_size");
            restore(ui, s, "photo_texture_size");
        });
        row(ui, "Folder", |ui| {
            let text = dir_text.get_or_insert_with(|| s.photo_dir.clone());
            let response = ui.add(egui::TextEdit::singleline(text).desired_width(240.0));
            if response.changed() && !text.trim().is_empty() {
                s.photo_dir = text.trim().to_owned();
            }
            if response.lost_focus() {
                *dir_text = None;
            }
            tag(ui, "photo_dir");
            restore(ui, s, "photo_dir");
        });
    });
    card(ui, "Hybrid terms", |ui| {
        secondary(
            ui,
            "The game keeps rendering its paused frame; only these lighting inputs become ray traced (fog, water, particles, characters, post and tone mapping stay the engine's). Characters still cast through the engine's shadow maps.",
        );
        ui.horizontal(|ui| {
            check(ui, &mut s.photo_sun, "Sun shadows", "photo_sun")
                .on_hover_text("Shadow rays toward the sun disc; .y = min(engine shadow maps, ray traced). Vegetation reads its own shadow maps.");
            check(ui, &mut s.photo_ao, "Ambient occlusion", "photo_ao")
                .on_hover_text("Cosine rays up to the AO radius into .x (ambient and probe light only).");
            check(ui, &mut s.photo_reflections, "Reflections", "photo_reflections")
                .on_hover_text("Ray-traced reflections accumulate on the FSR path; need FSR and rt_reflections = 1. Read when photo mode starts.");
        });
        slider_u32(
            ui,
            &mut s.photo_sun_rays,
            1..=4,
            "Sun rays / pass",
            "photo_sun_rays",
            false,
        );
        slider_u32(
            ui,
            &mut s.photo_ao_rays,
            1..=4,
            "AO rays / pass",
            "photo_ao_rays",
            false,
        );
        slider_u32(
            ui,
            &mut s.photo_reflect_rays,
            1..=64,
            "Reflection rays / frame",
            "photo_reflect_rays",
            true,
        );
        slider_u32(
            ui,
            &mut s.photo_terms_spp,
            1..=4096,
            "Target rays / pixel",
            "photo_terms_spp",
            true,
        );
        slider(
            ui,
            &mut s.photo_sun_radius,
            0.0..=2.0,
            "Sun radius (deg)",
            "photo_sun_radius",
        );
        slider(
            ui,
            &mut s.photo_ao_radius,
            0.1..=10.0,
            "AO radius (m)",
            "photo_ao_radius",
        );
        row(ui, "AO mode", |ui| {
            egui::ComboBox::from_id_salt("photo_ao_mode")
                .selected_text(if s.photo_ao_replace { "replace" } else { "min" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut s.photo_ao_replace,
                        false,
                        "min (with the engine's AO)",
                    );
                    ui.selectable_value(&mut s.photo_ao_replace, true, "replace");
                });
            tag(ui, "photo_ao_mode");
            restore(ui, s, "photo_ao_mode");
        });
        check(
            ui,
            &mut s.photo_save_masks,
            "Save the term masks (debug)",
            "photo_save_masks",
        );
        check(
            ui,
            &mut s.photo_save_instances,
            "Save the instance-type image (debug)",
            "photo_save_instances",
        )
        .on_hover_text(
            "Our primary rays from the photo camera: static grey, moved prop yellow, spawned green, character red, miss dark blue.",
        );
    });
    card(ui, "Path tracing (pathtrace / overlay)", |ui| {
        slider_u32(
            ui,
            &mut s.photo_spp,
            1..=4096,
            "Samples / pixel",
            "photo_spp",
            true,
        );
        slider_u32(
            ui,
            &mut s.photo_bounces,
            0..=8,
            "Bounces",
            "photo_bounces",
            false,
        );
        slider(
            ui,
            &mut s.photo_sky_scale,
            0.0..=4.0,
            "Sky scale",
            "photo_sky_scale",
        );
        row(ui, "Engine units", |ui| {
            let mut auto = s.photo_units.is_none();
            ui.checkbox(&mut auto, "auto (fitted)");
            let mut value = s
                .photo_units
                .unwrap_or(crate::photo::composite::FALLBACK_UNITS);
            if !auto {
                ui.add(egui::Slider::new(&mut value, 0.1..=20.0).logarithmic(true));
            }
            s.photo_units = (!auto).then_some(value);
            tag(ui, "photo_units");
            restore(ui, s, "photo_units");
        });
    });
    card(ui, "Ray-tracing scene", |ui| {
        check(ui, &mut s.rt, "Live scene service", "rt").on_hover_text(
            "D3D12 device, the map's BVH and a TLAS around the camera every frame for live ray tracing. Photo mode builds the scene on demand without it.",
        );
        row(ui, "Build", |ui| {
            egui::ComboBox::from_id_salt("rt_build")
                .selected_text(if s.rt_build_at_load {
                    "at map load"
                } else {
                    "on first use"
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.rt_build_at_load, true, "at map load");
                    ui.selectable_value(&mut s.rt_build_at_load, false, "on first use");
                });
            tag(ui, "rt_build");
            restore(ui, s, "rt_build");
        });
        slider(
            ui,
            &mut s.rt_radius,
            5.0..=200.0,
            "TLAS radius (m)",
            "rt_radius",
        );
        check(ui, &mut s.rt_dynamic, "Moving props", "rt_dynamic").on_hover_text(
            "Moved, hidden and spawned static models follow the game (its per-frame dirty flags). Turning it on or off rebuilds the scene.",
        );
        check(ui, &mut s.rt_characters, "Characters", "rt_characters").on_hover_text(
            "Skinned characters within the TLAS radius, posed from the game's joint palette on the GPU. Turning it on or off rebuilds the scene.",
        );
        slider_u32(
            ui,
            &mut s.rt_characters_n,
            1..=32,
            "Posed per frame",
            "rt_characters_n",
            false,
        );
        slider(
            ui,
            &mut s.rt_dynamic_budget_ms,
            0.25..=8.0,
            "Skinning GPU ms",
            "rt_dynamic_budget_ms",
        );
        let sc = &st.scene;
        grid(
            ui,
            "rt_scene",
            vec![
                (
                    "Worker",
                    if sc.worker { "running" } else { "stopped" }.into(),
                ),
                (
                    "State",
                    if sc.phase.is_empty() {
                        "idle".into()
                    } else {
                        sc.phase.clone()
                    },
                ),
                (
                    "Scene",
                    if sc.built_map.is_empty() {
                        "—".into()
                    } else {
                        sc.built_map.clone()
                    },
                ),
                (
                    "Build time",
                    if sc.built_map.is_empty() {
                        "—".into()
                    } else {
                        format!(
                            "decode {:.1} s, upload + BVH {:.1} s",
                            sc.load_s, sc.upload_s
                        )
                    },
                ),
                ("Instances", sc.instances.to_string()),
                (
                    "Live TLAS",
                    format!(
                        "{} instances, {} rebuilds, last {:.1} ms",
                        sc.tlas_instances, sc.tlas_rebuilds, sc.tlas_ms
                    ),
                ),
            ]
            .into_iter()
            .chain(sc.dynamic.as_ref().map(dynamic_rows).into_iter().flatten())
            .collect(),
        );
        if let Some(e) = &sc.error {
            ui.label(RichText::new(format!("Scene error: {e}")).color(BAD));
        }
    });
    card(ui, "Probe", |ui| {
        check(
            ui,
            &mut s.photo_probe,
            "Log changes with photo mode off",
            "photo_probe",
        )
        .on_hover_text("Read-only: the values below are logged whenever they change.");
        match &st.probe {
            None => secondary(ui, "No reading yet (photo mode or the probe must be on)."),
            Some(p) => {
                let cam = p.camera.map_or("unreadable".into(), |c| {
                    format!(
                        "({:.2}, {:.2}, {:.2}) q ({:.3}, {:.3}, {:.3}, {:.3}) fov {:.1}°",
                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
                    )
                });
                grid(
                    ui,
                    "photo_probe",
                    vec![
                        (
                            "Ansel session flag",
                            p.session_flag
                                .map_or("unreadable".into(), |f| f.to_string()),
                        ),
                        ("Game", format!("{:#x}", p.game)),
                        (
                            "Current game",
                            format!(
                                "{:#x}{}",
                                p.current_game,
                                if p.current_game == p.game {
                                    " (same)"
                                } else {
                                    " (differs)"
                                }
                            ),
                        ),
                        (
                            "Logic manager",
                            if p.manager_ok { "ok" } else { "not found" }.into(),
                        ),
                        (
                            "Logic state",
                            format!(
                                "{} ({}), depth {}",
                                p.top,
                                probe::state_name(p.top),
                                p.depth
                            ),
                        ),
                        (
                            "Map state",
                            p.map_state.map_or("unreadable".into(), |m| {
                                format!(
                                    "{m}{}",
                                    if m == probe::MAP_RUNNING {
                                        " (running)"
                                    } else {
                                        ""
                                    }
                                )
                            }),
                        ),
                        ("Ansel camera", cam),
                    ],
                );
            }
        }
        let c = &st.composite;
        ui.label(
            RichText::new(format!(
                "FOG hook {}, {} passes seen, {} composited, {} skipped",
                if c.hooked {
                    "installed"
                } else {
                    "not installed"
                },
                c.fog_frames,
                c.composited,
                c.failures
            ))
            .small()
            .color(if c.hooked { LIVE } else { MUTED }),
        );
        if let Some(e) = &c.error {
            secondary(ui, format!("Last composite skip: {e}"));
        }
    });
}

/// Rows of the dynamic objects (tracker on the main thread, worker).
fn dynamic_rows(d: &crate::rt::scene::DynamicStatus) -> Vec<(&'static str, String)> {
    let t = &d.track;
    let w = &d.worker;
    let st = &w.stats;
    vec![
        (
            "Dynamic reads",
            format!(
                "{} slots, {} flagged, {} reads / {:.1} KiB, {:.0} µs{}",
                t.slots,
                t.flagged,
                t.reads,
                f64::from(t.bytes) / 1024.0,
                t.us,
                if t.skip_update {
                    " — r_skipUpdateInView set: props static"
                } else {
                    ""
                }
            ),
        ),
        (
            "Props",
            format!(
                "{} moved, {} hidden, {} spawned ({} meshes, {} not traced); this second: {} moves, {} spawns",
                st.moved_static,
                st.hidden_static,
                st.props,
                st.prop_meshes,
                w.props_failed,
                t.moved_per_s,
                t.spawned_per_s
            ),
        ),
        (
            "Characters traced",
            format!(
                "{} (surfaces {} of {}); {} near ({} models), {} posed; {} skinned last ({:.2} ms GPU), {} deferred",
                w.characters_traced,
                w.surfaces_traced,
                w.surfaces_total,
                t.groups_near,
                t.wanted,
                st.posed,
                w.last_update.skinned,
                w.last_update.gpu_ms,
                w.last_update.deferred
            ),
        ),
        (
            "First person",
            format!("{} view models excluded (hands, weapons)", t.first_person),
        ),
        (
            "Palette",
            format!(
                "{} skinning, {} joint-world, {} rejected; {} types decoded, {} failed",
                w.mode_skinning,
                w.mode_joint_world,
                w.palettes_rejected,
                w.characters_decoded,
                w.characters_failed
            ),
        ),
        (
            "Reserve",
            format!(
                "{} / {} k triangles, {} textures, {:.0} MiB; {} refits (last {:.2} ms){}",
                st.triangles_used / 1000,
                st.triangles_reserved / 1000,
                st.textures_used,
                st.bytes as f64 / (1024.0 * 1024.0),
                w.tlas_refits,
                w.tlas_refit_ms,
                if d.events_dropped > 0 {
                    format!(", {} events dropped", d.events_dropped)
                } else {
                    String::new()
                }
            ),
        ),
    ]
}
