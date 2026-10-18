//! Main-thread side of dynamic objects (docs/raytracing.md section 14): reads what moved,
//! appeared or disappeared in the render world after `RenderAllViews` (`hook::observe`)
//! and turns it into [`Event`]s for the RT worker (`dynamic.rs`). Every read goes through
//! `mem` (checked); cached model pointers are never dereferenced except in the frame their
//! slot reports them (freed models stay valid until the next commit).
//!
//! Per frame with `rt_dynamic = 1`: one read of the committed dirty flags (`u32` per model
//! slot) and, per flagged static or spawned model, one read of its transform (+ its drawn
//! flag); every [`POLL_FRAMES`] one read of the model pointer list, and a vtable / name /
//! material read per changed slot (spawned models). With `rt_characters = 1`: per frame one
//! 1,960-byte read per character among the `rt_characters_n` nearest and, when the engine
//! computed new joints this frame, one palette read (48 B per joint).

use std::collections::HashSet;
use std::time::Instant;

use void_rt::ingame::{Entity, NO_INSTANCE, Row34};

use super::live::{self, ModelClass, ModelKey, dynamic as d};
use crate::log;

/// Frames between reads of the model pointer list (about 4 Hz at 60 fps).
pub const POLL_FRAMES: u64 = 15;
/// A wanted character is dropped beyond this factor of `rt_radius` (hysteresis).
const KEEP_FACTOR: f32 = 1.2;

/// What the worker applies to `void_rt::dynamic::Dynamics`.
#[derive(Clone, Debug)]
pub(crate) enum Event {
    /// A static scene instance moved and / or was hidden or shown.
    Static {
        instance: u32,
        transform: Option<Row34>,
        hidden: Option<bool>,
    },
    /// A model registered after the scene snapshot (static `.bmodel`): decode and place.
    Prop {
        id: u64,
        entity: Entity,
    },
    Moved {
        id: u64,
        transform: Row34,
    },
    Hidden {
        id: u64,
        hidden: bool,
    },
    Removed {
        id: u64,
    },
    /// A skinned model within the radius (decode the md6 at `lod`, add).
    Character {
        id: u64,
        md6: String,
        lod: u32,
        materials: Vec<String>,
        transform: Row34,
    },
    /// The engine's joint palette of this frame and the model transform.
    Pose {
        id: u64,
        palette: Vec<Row34>,
        transform: Row34,
    },
}

/// Counters of the last frame (overlay, log).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackStatus {
    pub slots: u32,
    pub flagged: u32,
    pub static_mapped: u32,
    pub props: u32,
    pub skinned: u32,
    pub wanted: u32,
    pub palettes: u32,
    pub events: u32,
    pub bytes: u32,
    pub reads: u32,
    pub us: f32,
    /// `r_skipUpdateInView` is non-zero: the engine skips the commit, props stay static.
    pub skip_update: bool,
    /// First-person view models (hands, weapons) left out since the scene build (distinct
    /// models).
    pub first_person: u32,
    /// Prop moves / spawns sent to the worker in the last full second.
    pub moved_per_s: u32,
    pub spawned_per_s: u32,
    /// Character groups (models sharing an origin: body, head, gear) near / posed.
    pub groups_near: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Kind {
    #[default]
    Empty,
    Static(u32),
    Prop,
    Skinned,
    Ignored,
}

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    ptr: u64,
    kind: Kind,
    hidden: bool,
    last: Option<Row34>,
    origin: [f32; 3],
    /// Announced to the worker as a character.
    wanted: bool,
    posed: bool,
}

/// What the tracker needs from the built scene.
pub(crate) struct Mapping<'a> {
    pub generation: u64,
    pub keys: &'a [ModelKey],
    pub entity_instances: &'a [u32],
    pub present: &'a [u64],
}

/// Per-frame inputs.
pub(crate) struct Frame {
    pub world: u64,
    pub frame: u64,
    pub camera: [f32; 3],
    pub radius: f32,
    pub props: bool,
    pub characters: bool,
    pub nearest: usize,
}

#[derive(Default)]
pub(crate) struct Tracker {
    generation: u64,
    world: u64,
    slots: Vec<Slot>,
    /// The first poll after mapping: unmapped models present at the snapshot were skipped
    /// by the scene build on purpose and stay ignored.
    initial: bool,
    present: HashSet<u64>,
    last_poll: u64,
    skip_logged: bool,
    skip_checked: u64,
    skip: bool,
    /// First-person view models already logged (by name) and counted (by model).
    fp_logged: HashSet<String>,
    fp_models: HashSet<u64>,
    /// One-second window: start, prop moves, spawns; last full second's values.
    window: Option<(std::time::Instant, u32, u32)>,
    rates: (u32, u32),
    pub status: TrackStatus,
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn origin(t: &Row34) -> [f32; 3] {
    [t[3], t[7], t[11]]
}

impl Tracker {
    /// The scene the tracker maps (0: none).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn map(&mut self, m: &Mapping, world: u64) {
        self.generation = m.generation;
        self.world = world;
        self.slots = vec![Slot::default(); m.present.len()];
        for (k, key) in m.keys.iter().enumerate() {
            let Some(s) = self.slots.get_mut(key.index as usize) else {
                continue;
            };
            s.ptr = key.ptr;
            s.kind = match m.entity_instances.get(k) {
                Some(&i) if i != NO_INSTANCE => Kind::Static(i),
                _ => Kind::Ignored,
            };
        }
        self.present = m.present.iter().copied().filter(|&p| p != 0).collect();
        self.initial = true;
        self.last_poll = 0;
        log::line(&format!(
            "rt dynamic: tracking {} model slots, {} static instances mapped",
            self.slots.len(),
            self.slots
                .iter()
                .filter(|s| matches!(s.kind, Kind::Static(_)))
                .count()
        ));
    }

    /// Forgets everything (map change, dynamics off).
    pub fn reset(&mut self) {
        *self = Self {
            skip_logged: self.skip_logged,
            ..Self::default()
        };
    }

    /// One frame (main thread): returns the events for the worker.
    pub fn tick(&mut self, m: &Mapping, f: &Frame) -> Vec<Event> {
        let t0 = Instant::now();
        let mut ev = Vec::new();
        let mut st = TrackStatus {
            skip_update: self.skip,
            ..TrackStatus::default()
        };
        if self.generation != m.generation || self.world != f.world {
            self.map(m, f.world);
        }
        if self.initial || f.frame >= self.last_poll + POLL_FRAMES {
            self.last_poll = f.frame;
            st.reads += 2;
            if let Some(ptrs) = live::model_pointers(f.world) {
                st.bytes += (ptrs.len() * 8) as u32;
                self.poll(&ptrs, f, &mut ev, &mut st);
                self.initial = false;
            }
        }
        if f.frame >= self.skip_checked + 60 || self.skip_checked == 0 {
            self.skip_checked = f.frame.max(1);
            self.skip = live::skip_update_in_view().unwrap_or(0) != 0;
            if self.skip && !self.skip_logged {
                self.skip_logged = true;
                log::line(
                    "rt dynamic: r_skipUpdateInView is set, the engine skips its model commit: moved props stay where the scene snapshot saw them",
                );
            }
        }
        st.skip_update = self.skip;
        if !self.skip {
            st.reads += 2;
            if let Some(flags) = live::committed_flags(f.world, self.slots.len() as u32) {
                st.bytes += (flags.len() * 4) as u32;
                for (idx, &fl) in flags.iter().enumerate() {
                    if fl != 0 {
                        st.flagged += 1;
                        self.flagged(idx, fl, f, &mut ev, &mut st);
                    }
                }
            }
        }
        if f.characters {
            self.characters(f, &mut ev, &mut st);
        } else {
            for (idx, s) in self.slots.iter_mut().enumerate() {
                if s.wanted {
                    s.wanted = false;
                    s.posed = false;
                    ev.push(Event::Removed {
                        id: ModelKey {
                            index: idx as u32,
                            ptr: s.ptr,
                        }
                        .id(),
                    });
                }
            }
        }
        st.slots = self.slots.len() as u32;
        for s in &self.slots {
            match s.kind {
                Kind::Static(_) => st.static_mapped += 1,
                Kind::Prop => st.props += 1,
                Kind::Skinned => {
                    st.skinned += 1;
                    st.wanted += u32::from(s.wanted);
                }
                _ => {}
            }
        }
        // Prop moves / spawns per second (one-second windows).
        let moves = ev
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::Static {
                        transform: Some(_),
                        ..
                    }
                ) || matches!(e, Event::Moved { id, .. } if self.is_prop(*id))
            })
            .count() as u32;
        let spawns = ev
            .iter()
            .filter(|e| matches!(e, Event::Prop { .. }))
            .count() as u32;
        let now = Instant::now();
        let w = self.window.get_or_insert((now, 0, 0));
        w.1 += moves;
        w.2 += spawns;
        if now.duration_since(w.0).as_secs_f32() >= 1.0 {
            self.rates = (w.1, w.2);
            *w = (now, 0, 0);
        }
        st.moved_per_s = self.rates.0;
        st.spawned_per_s = self.rates.1;
        st.first_person = self.fp_models.len() as u32;
        st.events = ev.len() as u32;
        st.us = t0.elapsed().as_secs_f32() * 1e6;
        self.status = st;
        ev
    }

    /// `id` belongs to a spawned prop slot.
    fn is_prop(&self, id: u64) -> bool {
        let idx = (id >> 48) as usize;
        self.slots
            .get(idx)
            .is_some_and(|s| s.kind == Kind::Prop && self.id(idx) == id)
    }

    /// Leaves model `p` (named `name`) out as a first-person view model; logged once per
    /// name.
    fn exclude_first_person(&mut self, p: u64, name: &str, why: &str) {
        self.fp_models.insert(p);
        if self.fp_logged.insert(name.to_owned()) {
            log::line(&format!(
                "rt dynamic: first-person view model left out of the scene: {name} ({why})"
            ));
        }
    }

    fn id(&self, idx: usize) -> u64 {
        ModelKey {
            index: idx as u32,
            ptr: self.slots[idx].ptr,
        }
        .id()
    }

    /// The slot's model is gone (freed or replaced): hide its instance / remove it.
    fn retire(&mut self, idx: usize, f: &Frame, ev: &mut Vec<Event>) {
        let s = self.slots[idx];
        match s.kind {
            Kind::Static(instance) if f.props && !s.hidden => ev.push(Event::Static {
                instance,
                transform: None,
                hidden: Some(true),
            }),
            Kind::Prop => ev.push(Event::Removed { id: self.id(idx) }),
            Kind::Skinned if s.wanted => ev.push(Event::Removed { id: self.id(idx) }),
            _ => {}
        }
        self.slots[idx] = Slot::default();
    }

    fn poll(&mut self, ptrs: &[u64], f: &Frame, ev: &mut Vec<Event>, st: &mut TrackStatus) {
        let n = ptrs.len().max(self.slots.len());
        self.slots.resize(n, Slot::default());
        for idx in 0..n {
            let p = ptrs.get(idx).copied().unwrap_or(0);
            if p == self.slots[idx].ptr {
                continue;
            }
            self.retire(idx, f, ev);
            if p == 0 {
                continue;
            }
            st.reads += 1;
            let kind = match live::model_class(p) {
                Some(ModelClass::Skinned) => {
                    st.reads += 2;
                    let name = live::model_name(p);
                    if live::first_person(p, &name) {
                        self.exclude_first_person(p, &name, "modelDepthHack > 0 or player path");
                        Kind::Ignored
                    } else {
                        self.slots[idx].origin =
                            live::model_transform(p).map_or([f32::MAX; 3], |t| origin(&t));
                        Kind::Skinned
                    }
                }
                Some(ModelClass::Static) if self.initial && self.present.contains(&p) => {
                    Kind::Ignored
                }
                Some(ModelClass::Static) if f.props => {
                    st.reads += 8;
                    match live::read_entity(p) {
                        Ok(e) => {
                            self.slots[idx].last = Some(e.transform);
                            self.slots[idx].ptr = p;
                            ev.push(Event::Prop {
                                id: self.id(idx),
                                entity: e,
                            });
                            Kind::Prop
                        }
                        Err(live::Skip::FirstPerson) => {
                            let name = live::model_name(p);
                            self.exclude_first_person(
                                p,
                                &name,
                                "modelDepthHack > 0 or player path",
                            );
                            Kind::Ignored
                        }
                        Err(_) => Kind::Ignored,
                    }
                }
                _ => Kind::Ignored,
            };
            self.slots[idx].ptr = p;
            self.slots[idx].kind = kind;
        }
    }

    fn flagged(
        &mut self,
        idx: usize,
        fl: u32,
        f: &Frame,
        ev: &mut Vec<Event>,
        st: &mut TrackStatus,
    ) {
        let Some(s) = self.slots.get(idx).copied() else {
            return;
        };
        if s.ptr == 0 {
            return;
        }
        if fl & d::FLAG_FREE != 0 {
            // Freed this frame: the slot is reused after the next commit (the poll
            // classifies whatever comes).
            self.retire(idx, f, ev);
            return;
        }
        match s.kind {
            Kind::Static(_) | Kind::Prop if f.props => {
                let mut hidden = s.hidden;
                let mut last = s.last;
                let mut moved = None;
                if fl & d::FLAG_UNLINK != 0 {
                    hidden = true;
                }
                if fl & (d::FLAG_TRANSFORM | d::FLAG_PARMS) != 0 {
                    st.reads += 2;
                    let drawable = live::model_drawable(s.ptr);
                    if fl & d::FLAG_UNLINK == 0 && drawable {
                        hidden = false;
                    } else if !drawable {
                        hidden = true;
                    }
                    if fl & d::FLAG_TRANSFORM != 0
                        && let Some(t) = live::model_transform(s.ptr)
                        && last != Some(t)
                    {
                        last = Some(t);
                        moved = Some(t);
                    }
                }
                let show = (hidden != s.hidden).then_some(hidden);
                if moved.is_some() || show.is_some() {
                    match s.kind {
                        Kind::Static(instance) => ev.push(Event::Static {
                            instance,
                            transform: moved,
                            hidden: show,
                        }),
                        _ => {
                            let id = self.id(idx);
                            if let Some(t) = moved {
                                ev.push(Event::Moved { id, transform: t });
                            }
                            if let Some(h) = show {
                                ev.push(Event::Hidden { id, hidden: h });
                            }
                        }
                    }
                }
                let slot = &mut self.slots[idx];
                slot.hidden = hidden;
                slot.last = last;
            }
            Kind::Skinned if f.characters && fl & d::FLAG_TRANSFORM != 0 => {
                st.reads += 1;
                if let Some(t) = live::model_transform(s.ptr) {
                    self.slots[idx].origin = origin(&t);
                    if s.wanted && s.last != Some(t) {
                        self.slots[idx].last = Some(t);
                        ev.push(Event::Moved {
                            id: self.id(idx),
                            transform: t,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn characters(&mut self, f: &Frame, ev: &mut Vec<Event>, st: &mut TrackStatus) {
        let cap = 2 * f.nearest.max(1);
        let mut near: Vec<(f32, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == Kind::Skinned)
            .map(|(i, s)| (dist(s.origin, f.camera), i))
            .filter(|&(dd, i)| {
                let keep = if self.slots[i].wanted {
                    KEEP_FACTOR
                } else {
                    1.0
                };
                dd.is_finite() && dd <= f.radius * keep
            })
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        // A character is often several render models (body, head, gear) at one origin:
        // the caps count such groups (origins on a 0.5 m grid), so every part of the
        // nearest characters is traced and posed.
        let mut groups: Vec<[i32; 3]> = Vec::new();
        let ranked: Vec<(usize, usize)> = near
            .iter()
            .filter_map(|&(_, idx)| {
                let o = self.slots[idx].origin;
                let key = o.map(|v| (v * 2.0).round() as i32);
                let rank = groups.iter().position(|g| *g == key).unwrap_or_else(|| {
                    groups.push(key);
                    groups.len() - 1
                });
                (rank < cap).then_some((rank, idx))
            })
            .collect();
        st.groups_near = groups.len().min(cap) as u32;
        let chosen: HashSet<usize> = ranked.iter().map(|r| r.1).collect();
        for idx in 0..self.slots.len() {
            if self.slots[idx].wanted && !chosen.contains(&idx) {
                ev.push(Event::Removed { id: self.id(idx) });
                self.slots[idx].wanted = false;
                self.slots[idx].posed = false;
            }
        }
        for &(rank, idx) in &ranked {
            let s = self.slots[idx];
            let id = self.id(idx);
            let pose = rank < f.nearest;
            let state = if !s.wanted || pose {
                st.reads += 1;
                st.bytes += (d::JOINTS_COUNT + 4 - 6240) as u32;
                live::skinned_state(s.ptr)
            } else {
                None
            };
            if let Some(state) = state
                && state.first_person
            {
                // The view model can switch on after registration (weapon drawn).
                let name = live::model_name(s.ptr);
                self.exclude_first_person(s.ptr, &name, "modelDepthHack > 0");
                if s.wanted {
                    ev.push(Event::Removed { id });
                }
                self.slots[idx].wanted = false;
                self.slots[idx].kind = Kind::Ignored;
                continue;
            }
            if !s.wanted {
                let Some(state) = state else { continue };
                st.reads += 4;
                let md6 = live::model_name(s.ptr);
                if md6.is_empty() {
                    self.slots[idx].kind = Kind::Ignored;
                    continue;
                }
                ev.push(Event::Character {
                    id,
                    md6,
                    lod: state.lod,
                    materials: live::model_materials(s.ptr),
                    transform: state.transform,
                });
                self.slots[idx].wanted = true;
            }
            let Some(state) = state else { continue };
            self.slots[idx].origin = origin(&state.transform);
            if !pose {
                continue;
            }
            if state.new_joints > 0 || !self.slots[idx].posed {
                st.reads += 1;
                st.bytes += state.joints.min(d::MAX_JOINTS) * 48;
                if let Some(palette) = live::joint_palette(state.joints_ptr, state.joints) {
                    st.palettes += 1;
                    self.slots[idx].posed = true;
                    self.slots[idx].last = Some(state.transform);
                    ev.push(Event::Pose {
                        id,
                        palette,
                        transform: state.transform,
                    });
                    continue;
                }
            }
            if self.slots[idx].last != Some(state.transform) {
                self.slots[idx].last = Some(state.transform);
                ev.push(Event::Moved {
                    id,
                    transform: state.transform,
                });
            }
        }
    }
}
