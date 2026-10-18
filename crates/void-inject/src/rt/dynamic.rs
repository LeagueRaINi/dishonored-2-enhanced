//! RT worker side of dynamic objects (docs/raytracing.md section 14): applies the
//! main thread's [`Event`]s (`track.rs`) to `void_rt`'s [`Dynamics`], decodes spawned
//! props and characters on a helper thread (`void-rt-decode`, CPU only), checks each
//! character's palette convention on its first pose, and runs the per-tick skinning
//! update within the GPU budget. The cut-off TLAS refit / rebuild is in `scene.rs`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use void_rt::character::{effective_palette, palette_check};
use void_rt::ingame::{
    CharacterCpu, DynamicAssets, DynamicStats, Dynamics, Entity, Gpu, GpuScene, LoadOptions,
    PaletteMode, Row34, Scene, UpdateStats,
};

use super::track::Event;
use crate::log;

/// Live settings of the dynamic layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DynSettings {
    pub props: bool,
    pub characters: bool,
    pub nearest: usize,
    pub budget_ms: f32,
}

/// Character types kept decoded (CPU data for later spawns of the same type).
const CPU_CACHE: usize = 16;
/// Characters uploaded per worker tick at most (each is a few blocking submits).
const ADDS_PER_TICK: usize = 1;
/// Failed decodes logged per scene at most.
const MAX_FAILURE_LOGS: usize = 20;
/// Palette checks before a character is given up (the first poses may be torn reads).
const PALETTE_TRIES: u32 = 3;

/// Worker counters for the overlay.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DynStatus {
    pub stats: DynamicStats,
    pub last_update: UpdateStats,
    pub props_decoded: u32,
    pub props_failed: u32,
    pub characters_decoded: u32,
    pub characters_failed: u32,
    pub palettes_rejected: u32,
    /// Palette convention found per character (`skinning` / `joint world`).
    pub mode_skinning: u32,
    pub mode_joint_world: u32,
    /// Characters in the scene (added) and their traced / decoded surfaces.
    pub characters_traced: u32,
    pub surfaces_traced: u32,
    pub surfaces_total: u32,
    pub tlas_refits: u64,
    pub tlas_refit_ms: f64,
    pub error: Option<String>,
}

enum Job {
    Prop {
        key: String,
        entity: Box<Entity>,
    },
    Character {
        key: String,
        md6: String,
        lod: u32,
        materials: Vec<String>,
    },
}

enum Done {
    Prop {
        key: String,
        result: Result<Box<Scene>, String>,
    },
    Character {
        key: String,
        result: Result<Arc<CharacterCpu>, String>,
    },
}

/// The decode thread: opens the archives once and decodes jobs in order. Dropping it
/// closes the job channel; the thread ends after its current job.
struct Decoder {
    tx: Sender<Job>,
    rx: Receiver<Done>,
}

impl Decoder {
    fn new(game_dir: PathBuf, map: String, opts: LoadOptions) -> Result<Self, String> {
        let (tx, jobs) = channel::<Job>();
        let (done, rx) = channel::<Done>();
        std::thread::Builder::new()
            .name("void-rt-decode".into())
            .spawn(move || {
                let mut assets = match DynamicAssets::open(&game_dir, &map, opts) {
                    Ok(a) => a,
                    Err(e) => {
                        log::line(&format!("rt dynamic: archives not opened: {e}"));
                        return;
                    }
                };
                for job in jobs {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
                        Job::Prop { key, entity } => Done::Prop {
                            result: assets.prop(&key, &entity).map(Box::new),
                            key,
                        },
                        Job::Character {
                            key,
                            md6,
                            lod,
                            materials,
                        } => Done::Character {
                            result: assets.character(&md6, lod, &materials).map(Arc::new),
                            key,
                        },
                    }));
                    let Ok(d) = r else {
                        log::line("rt dynamic: decoder panicked");
                        return;
                    };
                    if done.send(d).is_err() {
                        return;
                    }
                }
            })
            .map_err(|e| format!("decode thread: {e}"))?;
        Ok(Self { tx, rx })
    }

    fn send(&self, job: Job) {
        let _ = self.tx.send(job);
    }
}

struct CharState {
    key: String,
    transform: Row34,
    pose: Option<Vec<Row34>>,
    mode: Option<PaletteMode>,
    checks: u32,
    added: bool,
    pose_error_logged: bool,
    /// Traced / decoded surfaces once added.
    surfaces: [u32; 2],
}

/// Dynamic objects of the current scene on the worker.
pub(crate) struct DynState {
    pub dynamics: Dynamics,
    decoder: Decoder,
    /// Mesh key -> props waiting for its decode (id, transform).
    waiting: HashMap<String, Vec<(u64, Row34)>>,
    failed: HashSet<String>,
    chars: HashMap<u64, CharState>,
    cpu: HashMap<String, Arc<CharacterCpu>>,
    jobs: HashSet<String>,
    char_failed: HashSet<String>,
    failure_logs: usize,
    settings: DynSettings,
    pub status: DynStatus,
}

fn prop_key(e: &Entity) -> String {
    format!("{}|{}", e.model_source, e.materials.join(","))
}

impl DynState {
    pub fn new(
        gpu: &Gpu,
        scene: Arc<GpuScene>,
        game_dir: PathBuf,
        settings: DynSettings,
    ) -> Result<Self, String> {
        let opts = scene.dynamic_tables().load;
        let map = scene.map().to_owned();
        let dynamics = Dynamics::new(gpu, scene)?;
        log::line(&format!(
            "rt dynamic: on for '{map}' (props {}, characters {}, nearest {}, budget {} ms; TLAS capacity {})",
            settings.props,
            settings.characters,
            settings.nearest,
            settings.budget_ms,
            dynamics.tlas_capacity()
        ));
        Ok(Self {
            dynamics,
            decoder: Decoder::new(game_dir, map, opts)?,
            waiting: HashMap::new(),
            failed: HashSet::new(),
            chars: HashMap::new(),
            cpu: HashMap::new(),
            jobs: HashSet::new(),
            char_failed: HashSet::new(),
            failure_logs: 0,
            settings,
            status: DynStatus::default(),
        })
    }

    pub fn scene(&self) -> &Arc<GpuScene> {
        self.dynamics.scene()
    }

    fn fail_log(&mut self, what: &str) {
        if self.failure_logs < MAX_FAILURE_LOGS {
            self.failure_logs += 1;
            log::line(&format!("rt dynamic: {what}"));
        }
    }

    /// One worker tick: settings, events, decode results, character uploads and poses,
    /// the skinning update near `camera`, and freeing retired mesh data.
    pub fn tick(
        &mut self,
        gpu: &mut Gpu,
        events: Vec<Event>,
        settings: DynSettings,
        camera: [f32; 3],
    ) -> Result<(), String> {
        if self.settings.props && !settings.props {
            self.dynamics.reset_props();
            self.waiting.clear();
        }
        if self.settings.characters && !settings.characters {
            let ids: Vec<u64> = self.chars.keys().copied().collect();
            for id in ids {
                self.dynamics.remove(id);
            }
            self.chars.clear();
        }
        self.settings = settings;
        for e in events {
            self.apply(e);
        }
        self.results(gpu)?;
        self.admit(gpu)?;
        self.poses();
        if self.settings.characters {
            let st = self.dynamics.update(
                gpu,
                camera,
                self.settings.nearest,
                f64::from(self.settings.budget_ms),
            )?;
            if st.skinned > 0 || st.deferred > 0 {
                self.status.last_update = st;
            }
        }
        self.dynamics.collect(gpu)?;
        self.status.stats = self.dynamics.stats();
        let (mut n, mut traced, mut total) = (0, 0, 0);
        for c in self.chars.values().filter(|c| c.added) {
            n += 1;
            traced += c.surfaces[0];
            total += c.surfaces[1];
        }
        self.status.characters_traced = n;
        self.status.surfaces_traced = traced;
        self.status.surfaces_total = total;
        Ok(())
    }

    fn apply(&mut self, e: Event) {
        match e {
            Event::Static {
                instance,
                transform,
                hidden,
            } if self.settings.props => {
                self.dynamics.set_static(instance, transform, hidden);
            }
            Event::Prop { id, entity } if self.settings.props => {
                let key = prop_key(&entity);
                if self.dynamics.has_mesh(&key) {
                    if let Err(err) = self.dynamics.place(id, &key, entity.transform) {
                        self.fail_log(&format!("prop {key}: {err}"));
                    }
                } else if !self.failed.contains(&key) {
                    let w = self.waiting.entry(key.clone()).or_default();
                    if w.is_empty() {
                        self.decoder.send(Job::Prop {
                            key,
                            entity: Box::new(entity.clone()),
                        });
                    }
                    w.push((id, entity.transform));
                }
            }
            Event::Moved { id, transform } => {
                if !self.dynamics.set_transform(id, transform) {
                    for w in self.waiting.values_mut() {
                        for p in w.iter_mut().filter(|p| p.0 == id) {
                            p.1 = transform;
                        }
                    }
                }
                if let Some(c) = self.chars.get_mut(&id) {
                    c.transform = transform;
                }
            }
            Event::Hidden { id, hidden } => {
                self.dynamics.set_hidden(id, hidden);
            }
            Event::Removed { id } => {
                self.dynamics.remove(id);
                for w in self.waiting.values_mut() {
                    w.retain(|p| p.0 != id);
                }
                self.chars.remove(&id);
            }
            Event::Character {
                id,
                md6,
                lod,
                materials,
                transform,
            } if self.settings.characters => {
                let key = format!("{md6}|{lod}|{}", materials.join(","));
                if self.char_failed.contains(&key) {
                    return;
                }
                if !self.cpu.contains_key(&key) && self.jobs.insert(key.clone()) {
                    self.decoder.send(Job::Character {
                        key: key.clone(),
                        md6,
                        lod,
                        materials,
                    });
                }
                self.chars.insert(
                    id,
                    CharState {
                        key,
                        transform,
                        pose: None,
                        mode: None,
                        checks: 0,
                        added: false,
                        pose_error_logged: false,
                        surfaces: [0; 2],
                    },
                );
            }
            Event::Pose {
                id,
                palette,
                transform,
            } => {
                if let Some(c) = self.chars.get_mut(&id) {
                    c.pose = Some(palette);
                    c.transform = transform;
                }
            }
            _ => {}
        }
    }

    /// Decode results: upload prop meshes and place their waiting instances; keep
    /// character CPU data.
    fn results(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        while let Ok(done) = self.decoder.rx.try_recv() {
            match done {
                Done::Prop { key, result } => {
                    let waiting = self.waiting.remove(&key).unwrap_or_default();
                    let r = result.and_then(|s| self.dynamics.add_mesh(gpu, &key, &s));
                    match r {
                        Ok(()) => {
                            self.status.props_decoded += 1;
                            for (id, t) in waiting {
                                if let Err(e) = self.dynamics.place(id, &key, t) {
                                    self.fail_log(&format!("prop {key}: {e}"));
                                }
                            }
                        }
                        Err(e) if e.contains("removed") => return Err(e),
                        Err(e) => {
                            self.status.props_failed += 1;
                            self.fail_log(&format!("prop {key} not traced: {e}"));
                            self.failed.insert(key);
                        }
                    }
                }
                Done::Character { key, result } => {
                    self.jobs.remove(&key);
                    match result {
                        Ok(c) => {
                            self.status.characters_decoded += 1;
                            log::line(&format!(
                                "rt dynamic: character {} LOD {}: {} vertices, {} triangles, {} joints, surfaces {} of {} traced; parts {}{}{}",
                                c.md6,
                                c.lod,
                                c.vertex_count(),
                                c.triangle_count(),
                                c.joint_count(),
                                c.surfaces[0],
                                c.surfaces[1],
                                c.meshes.join(", "),
                                if c.notes.is_empty() { "" } else { "; " },
                                c.notes.join("; ")
                            ));
                            if !c.fallbacks.is_empty() {
                                log::line(&format!(
                                    "rt dynamic: character {}: {} materials traced as primitive data (diffuse average or grey): {}",
                                    c.md6,
                                    c.fallbacks.len(),
                                    c.fallbacks
                                        .iter()
                                        .map(|(m, why)| format!("{m} [{why}]"))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ));
                            }
                            if self.cpu.len() >= CPU_CACHE {
                                // Drop a type no character waits for.
                                let unused = self
                                    .cpu
                                    .keys()
                                    .find(|k| {
                                        !self.chars.values().any(|c| !c.added && &c.key == *k)
                                    })
                                    .cloned();
                                if let Some(k) = unused {
                                    self.cpu.remove(&k);
                                }
                            }
                            self.cpu.insert(key, c);
                        }
                        Err(e) => {
                            self.status.characters_failed += 1;
                            self.fail_log(&format!("character {key} not traced: {e}"));
                            self.char_failed.insert(key.clone());
                            self.chars.retain(|_, c| c.key != key);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Uploads decoded characters (at most [`ADDS_PER_TICK`] per tick).
    fn admit(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let ready: Vec<u64> = self
            .chars
            .iter()
            .filter(|(_, c)| !c.added && self.cpu.contains_key(&c.key))
            .map(|(&id, _)| id)
            .take(ADDS_PER_TICK)
            .collect();
        for id in ready {
            let Some(c) = self.chars.get(&id) else {
                continue;
            };
            let Some(cpu) = self.cpu.get(&c.key).cloned() else {
                continue;
            };
            match self.dynamics.add_character(gpu, id, &cpu, c.transform) {
                Ok(()) => {
                    if let Some(c) = self.chars.get_mut(&id) {
                        c.added = true;
                        c.surfaces = cpu.surfaces;
                    }
                }
                Err(e) if e.contains("removed") => return Err(e),
                Err(e) => {
                    self.fail_log(&format!("character {} not added: {e}", cpu.md6));
                    self.chars.remove(&id);
                }
            }
        }
        Ok(())
    }

    /// Forwards the latest pose of every added character (palette convention checked on
    /// the first ones).
    fn poses(&mut self) {
        let ids: Vec<u64> = self
            .chars
            .iter()
            .filter(|(_, c)| c.added && c.pose.is_some())
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let Some(c) = self.chars.get_mut(&id) else {
                continue;
            };
            let Some(cpu) = self.cpu.get(&c.key).cloned() else {
                continue;
            };
            let Some(palette) = c.pose.take() else {
                continue;
            };
            if palette.len() < cpu.joint_count() as usize {
                if !c.pose_error_logged {
                    c.pose_error_logged = true;
                    let msg = format!(
                        "character {}: engine palette has {} joints, skeleton {}: pose skipped",
                        cpu.md6,
                        palette.len(),
                        cpu.joint_count()
                    );
                    self.status.palettes_rejected += 1;
                    self.fail_log(&msg);
                }
                continue;
            }
            let c = self.chars.get_mut(&id).expect("present");
            if c.mode.is_none() {
                if c.checks >= PALETTE_TRIES {
                    continue;
                }
                c.checks += 1;
                let v = palette_check(&cpu, &palette);
                let msg = format!(
                    "character {}: palette check: bind-relative size / centre offset {:.2} / {:.2} m as skinning matrices, {:.2} / {:.2} m as joint matrices -> {}",
                    cpu.md6,
                    v.skinning[0],
                    v.skinning[1],
                    v.joint_world[0],
                    v.joint_world[1],
                    match v.mode {
                        Some(PaletteMode::Skinning) => "skinning (bone -> object)",
                        Some(PaletteMode::JointWorld) => "joint world (x inverse bind here)",
                        None => "neither plausible, pose skipped",
                    }
                );
                log::line(&format!("rt dynamic: {msg}"));
                match v.mode {
                    Some(PaletteMode::Skinning) => self.status.mode_skinning += 1,
                    Some(PaletteMode::JointWorld) => self.status.mode_joint_world += 1,
                    None => {
                        self.status.palettes_rejected += 1;
                        continue;
                    }
                }
                let c = self.chars.get_mut(&id).expect("present");
                c.mode = v.mode;
            }
            let c = self.chars.get_mut(&id).expect("present");
            let Some(mode) = c.mode else { continue };
            let pal = effective_palette(&cpu, &palette, mode);
            let t = c.transform;
            if let Err(e) = self.dynamics.pose(id, pal, t) {
                self.status.palettes_rejected += 1;
                self.fail_log(&format!("pose: {e}"));
            }
        }
    }
}
