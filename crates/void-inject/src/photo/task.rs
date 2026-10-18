//! The photo mode's work on the RT worker thread (`rt::scene::Task`): waits for the map's
//! scene, then either drives the hybrid terms (`photo_display = hybrid`, `hybrid.rs`) or runs
//! the progressive `PhotoSession` with the per-frame GPU budget (restarted when the free
//! camera moves or a path setting changes, copied into the composite slots), and saves on
//! request. A display change switches between the two (the other one's data is dropped).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use void_rt::camera::Camera;
use void_rt::ingame::{Gpu, PhotoSession, PhotoSettings};

use super::ansel;
use super::composite::{self, SlotSet};
use super::hybrid::Hybrid;
use crate::rt::device::Device;
use crate::rt::scene::{self, Built, Task};
use crate::{log, settings};

/// Copies into a composite slot at most this often while accumulating.
const COPY_EVERY: Duration = Duration::from_millis(250);
/// Give up when the scene is not ready after this long.
const SCENE_TIMEOUT: Duration = Duration::from_secs(300);

/// State shared between the main thread (photo mode, overlay) and the task.
#[derive(Default)]
pub(crate) struct Shared {
    pub phase: Mutex<String>,
    pub spp_done: AtomicU32,
    pub spp_target: AtomicU32,
    /// f32 bits.
    pub ms_per_spp: AtomicU32,
    pub save: AtomicBool,
    /// Set when the task ended with an error (the main thread leaves photo mode).
    pub error: Mutex<Option<String>>,
    /// Messages for toasts (saved files).
    pub messages: Mutex<Vec<String>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Shared {
    fn set_phase(&self, p: impl Into<String>) {
        *lock(&self.phase) = p.into();
    }
}

/// Path settings from the settings file (multipliers on the validated defaults).
pub(crate) fn photo_settings(s: &settings::Settings, znear: f32) -> PhotoSettings {
    let d = PhotoSettings::default();
    PhotoSettings {
        spp_target: s.photo_spp,
        bounces: s.photo_bounces,
        sky_scale: d.sky_scale * s.photo_sky_scale,
        znear,
        // Half the default tile (1/16 of 3440x1440): about 3 ms per tile on an RX 7900 XTX
        // (RtLib measurement), so one tile never exceeds a small per-frame budget much.
        tile_pixels: void_rt::ingame::DEFAULT_TILE_PIXELS / 2,
        ..d
    }
}

pub(crate) struct PhotoTask {
    shared: Arc<Shared>,
    map: String,
    camera: Camera,
    cam_version: u64,
    size: [u32; 2],
    znear: f32,
    dev: Arc<Device>,
    session: Option<PhotoSession>,
    slots: Option<Arc<SlotSet>>,
    /// The hybrid terms (`photo_display = hybrid`).
    hybrid: Option<Hybrid>,
    /// Slot being written and the fence of its copy.
    pending: Option<(usize, u64, u32)>,
    last_copy: Option<Instant>,
    copied_spp: u32,
    waiting_since: Instant,
    started: Instant,
    logged_done: bool,
}

impl PhotoTask {
    pub fn new(
        shared: Arc<Shared>,
        map: String,
        camera: Camera,
        size: [u32; 2],
        znear: f32,
        dev: Arc<Device>,
    ) -> Self {
        shared.set_phase("preparing the scene");
        Self {
            shared,
            map,
            camera,
            cam_version: 0,
            size,
            znear,
            dev,
            session: None,
            slots: None,
            hybrid: None,
            pending: None,
            last_copy: None,
            copied_spp: 0,
            waiting_since: Instant::now(),
            started: Instant::now(),
            logged_done: false,
        }
    }

    fn release_slots(&mut self) {
        if let Some(s) = self.slots.take()
            && s.is_current()
        {
            composite::deactivate();
        }
    }

    /// Drops the session (and with it its hold on the scene); the composite shows nothing
    /// until the next copy.
    fn drop_session(&mut self) {
        self.session = None;
        self.pending = None;
        self.copied_spp = 0;
        if let Some(sl) = &self.slots {
            sl.unpublish();
        }
    }

    /// `<dir>/<name>_instances.png` (`photo_save_instances`): our primary rays from `cam`
    /// through the photo's TLAS (dynamic objects frozen at entry, else the static scene),
    /// coloured by instance type (`void_rt::dynamic::type_colour`). Blocking, one submit.
    fn save_instances(
        &self,
        gpu: &mut Gpu,
        built: &Built,
        dir: &std::path::Path,
        name: &str,
        cam: Camera,
    ) -> Result<String, String> {
        if built.scene.primitives().is_none() {
            return Err("the scene has no FC6 primitives (rt = 0)".into());
        }
        let w = self.size[0].clamp(16, 1280);
        let h = ((u64::from(self.size[1]) * u64::from(w) / u64::from(self.size[0].max(1))) as u32)
            .clamp(16, 1280);
        let probe = void_rt::dynamic::InstanceProbe::new(gpu, w, h)?;
        let frozen = scene::photo_tlas(built);
        let (tlas, types) = match &frozen {
            Some(t) => (&t.tlas.buffer, t.types.as_slice()),
            None => (&built.scene.full_tlas().buffer, &[][..]),
        };
        let img = probe.render(gpu, &built.scene, tlas, &cam)?;
        let path = dir.join(format!("{name}_instances.png"));
        void_rt::image::write_png(&path, w, h, &img.type_rgb(types))?;
        Ok(path.display().to_string())
    }

    /// The photo folder (created) and the file stem `<map>_<unix time>`.
    fn save_target(&self, s: &settings::Settings) -> Result<(PathBuf, String), String> {
        let dir = {
            let d = PathBuf::from(&s.photo_dir);
            if d.is_absolute() {
                d
            } else {
                crate::game_dir().cloned().unwrap_or_default().join(d)
            }
        };
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let base = self
            .map
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or("photo")
            .to_owned();
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Ok((dir, format!("{base}_{t}")))
    }

    /// Hybrid display: the back buffer (the game's frame with the terms, engine post applied)
    /// as `<name>.png`, notes in `<name>.txt`, and with `photo_save_masks` the term masks.
    fn save_hybrid(
        &mut self,
        gpu: &mut Gpu,
        s: &settings::Settings,
        built: &Built,
    ) -> Result<(), String> {
        if !crate::overlay::installed() {
            return Err("saving the game frame needs the overlay (overlay = 1)".into());
        }
        let (dir, stem) = self.save_target(s)?;
        let name = format!("{stem}_hybrid");
        let cam = ansel::current().map_or(self.camera, |(c, _)| c);
        let instances = self.instances_if(gpu, s, built, &dir, &name, cam);
        let Some(h) = self.hybrid.as_mut() else {
            return Err("nothing rendered yet".into());
        };
        let rays = h.rays().unwrap_or_default();
        let reflections = crate::ssr::rt_reflect::photo_samples();
        let masks = if s.photo_save_masks {
            h.save_masks(gpu, &dir.join(&name))?
        } else {
            Vec::new()
        };
        let text = format!(
            "map {}\ncamera org {:?} forward {:?} left {:?} up {:?} fov_y {:.4} rad\nregion {}x{}\ndisplay hybrid: the game's frame with ray-traced terms\nsun shadows {} ({} rays per pixel, {} per frame, disc radius {} deg; .y = min(engine CSM, RT))\nambient occlusion {} ({} rays per pixel, {} per frame, radius {} m, mode {})\nreflections {} ({reflections} samples per texel)\nmasks {:?}\n",
            self.map,
            cam.org,
            cam.axis[0],
            cam.axis[1],
            cam.axis[2],
            cam.fov_y,
            self.size[0],
            self.size[1],
            u8::from(s.photo_sun),
            rays[0],
            s.photo_sun_rays,
            s.photo_sun_radius,
            u8::from(s.photo_ao),
            rays[1],
            s.photo_ao_rays,
            s.photo_ao_radius,
            if s.photo_ao_replace { "replace" } else { "min" },
            u8::from(s.photo_reflections),
            masks
        );
        std::fs::write(dir.join(format!("{name}.txt")), text)
            .map_err(|e| format!("{name}.txt: {e}"))?;
        composite::request_capture(dir.join(format!("{name}.png")));
        let msg = format!(
            "Photo saved: {name} (sun {} / AO {} rays, reflections {reflections} samples){}{}, game frame follows",
            rays[0],
            rays[1],
            if masks.is_empty() { "" } else { ", masks" },
            if instances { ", instance types" } else { "" }
        );
        log::line(&format!(
            "photo: saved {} ({msg})",
            dir.join(&name).display()
        ));
        lock(&self.shared.messages).push(msg);
        Ok(())
    }

    /// With `photo_save_instances`: [`PhotoTask::save_instances`], failures logged and
    /// toasted (the photo itself is saved anyway). True when the image was written.
    fn instances_if(
        &self,
        gpu: &mut Gpu,
        s: &settings::Settings,
        built: &Built,
        dir: &std::path::Path,
        name: &str,
        cam: Camera,
    ) -> bool {
        if !s.photo_save_instances {
            return false;
        }
        match self.save_instances(gpu, built, dir, name, cam) {
            Ok(p) => {
                log::line(&format!("photo: instance-type image {p}"));
                true
            }
            Err(e) => {
                log::line(&format!("photo: instance-type image not saved: {e}"));
                lock(&self.shared.messages).push(format!("Instance image not saved: {e}"));
                false
            }
        }
    }

    fn save(&mut self, gpu: &mut Gpu, s: &settings::Settings, built: &Built) -> Result<(), String> {
        let (dir, stem) = self.save_target(s)?;
        let cam = self.session.as_ref().map_or(self.camera, |x| *x.camera());
        self.instances_if(gpu, s, built, &dir, &stem, cam);
        let Some(session) = self.session.as_mut() else {
            return Err("nothing rendered yet".into());
        };
        let spp = session.spp_done();
        let name = format!("{stem}_{spp}spp");
        let cam = *session.camera();
        let notes = format!(
            "map {}; camera org {:?} forward {:?} left {:?} up {:?} fov_y {:.4} rad; {}x{}; bounces {}, sky scale {}",
            self.map,
            cam.org,
            cam.axis[0],
            cam.axis[1],
            cam.axis[2],
            cam.fov_y,
            self.size[0],
            self.size[1],
            session.settings().bounces,
            session.settings().sky_scale
        );
        session.write_hdr(gpu, &dir.join(format!("{name}.bin")), &notes)?;
        let composited =
            s.photo_display == super::Display::PathTrace && crate::overlay::installed();
        let ours = dir.join(if composited {
            format!("{name}_rt.png")
        } else {
            format!("{name}.png")
        });
        let exposure = session.write_png(gpu, &ours, None)?;
        let sun = built.scene.sun();
        let text = format!(
            "{notes}\nspp {spp}\nsun {:?}\nsky {:?}\nunits {}\nour tone-mapped PNG exposure {exposure}\nscene: {} instances, decode {:.1} s, upload {:.1} s\n",
            sun,
            built.scene.env(),
            s.photo_units
                .map_or("auto (fitted per frame)".to_owned(), |u| u.to_string()),
            built.scene.instances().len(),
            built.load_s,
            built.upload_s
        );
        std::fs::write(dir.join(format!("{name}.txt")), text)
            .map_err(|e| format!("{name}.txt: {e}"))?;
        if composited {
            composite::request_capture(dir.join(format!("{name}.png")));
        }
        let msg = format!(
            "Photo saved: {name} ({spp} spp){}",
            if composited {
                ", game frame follows"
            } else {
                ""
            }
        );
        log::line(&format!(
            "photo: saved {} ({})",
            dir.join(&name).display(),
            msg
        ));
        lock(&self.shared.messages).push(msg);
        Ok(())
    }

    /// Hybrid display: drops the path tracer's data, drives the terms, saves on request.
    fn run_hybrid(
        &mut self,
        gpu: &mut Gpu,
        s: &settings::Settings,
        built: &Arc<Built>,
    ) -> Result<bool, String> {
        if self.session.is_some() || self.slots.is_some() {
            log::line("photo: display changed to hybrid, path-tracer session dropped");
            self.drop_session();
            self.release_slots();
        }
        let epoch = ansel::current().map_or(0, |(_, v)| v);
        let h = self
            .hybrid
            .get_or_insert_with(|| Hybrid::new(self.dev.clone()));
        h.step(gpu, built, s, epoch)?;
        if let Some(st) = super::hybrid::status() {
            self.shared.set_phase(st.phase);
        }
        if self.shared.save.swap(false, Ordering::AcqRel)
            && let Err(e) = self.save_hybrid(gpu, s, built)
        {
            log::line(&format!("photo: save failed: {e}"));
            lock(&self.shared.messages).push(format!("Photo not saved: {e}"));
        }
        Ok(true)
    }

    fn run(&mut self, gpu: &mut Gpu, scene_now: Option<&Arc<Built>>) -> Result<bool, String> {
        let s = settings::get();
        // The build itself was requested at entry (`photo::enter`) and is owned by the RT
        // worker; this task only waits for it.
        let Some(built) = scene_now.filter(|b| b.map.eq_ignore_ascii_case(&self.map)) else {
            let st = scene::status();
            if st.phase == "failed" && self.waiting_since.elapsed() > Duration::from_secs(1) {
                return Err(st.error.unwrap_or_else(|| "scene build failed".into()));
            }
            if self.waiting_since.elapsed() > SCENE_TIMEOUT {
                return Err("the scene was not ready after 5 minutes".into());
            }
            self.shared.set_phase(format!(
                "preparing the scene ({}, {:.0} s)",
                if st.phase.is_empty() {
                    "queued"
                } else {
                    &st.phase
                },
                self.waiting_since.elapsed().as_secs_f64()
            ));
            return Ok(true);
        };
        let built = built.clone();
        if s.photo_display == super::Display::Hybrid {
            return self.run_hybrid(gpu, &s, &built);
        }
        if self.hybrid.take().is_some() {
            log::line("photo: display changed to the path tracer, hybrid terms dropped");
        }
        // A session on an older scene (rebuilt meanwhile) restarts on the current one, so
        // the old scene's memory is not kept alive.
        if self
            .session
            .as_ref()
            .is_some_and(|sess| sess.scene().generation() != built.scene.generation())
        {
            log::line("photo: the scene was rebuilt, restarting the accumulation");
            self.drop_session();
        }
        let want = photo_settings(&s, self.znear);
        if let Some(sess) = &self.session
            && (sess.settings().bounces != want.bounces
                || (sess.settings().sky_scale - want.sky_scale).abs() > 1e-6)
        {
            log::line("photo: path settings changed, restarting the accumulation");
            self.drop_session();
        }
        if let Some((cam, v)) = ansel::current()
            && v != self.cam_version
        {
            self.cam_version = v;
            if cam != self.camera {
                self.camera = cam;
                if let Some(sess) = self.session.as_mut() {
                    sess.reset_camera(gpu, cam)?;
                }
                self.pending = None;
                self.copied_spp = 0;
                if let Some(sl) = &self.slots {
                    sl.unpublish();
                }
            }
        }
        if self.session.is_none() {
            let t = Instant::now();
            let mut session = PhotoSession::new(
                gpu,
                built.scene.clone(),
                self.camera,
                self.size[0],
                self.size[1],
                want,
            )?;
            // Dynamic objects (props, characters) frozen at photo entry (section 14.4).
            if let Some(tlas) = scene::photo_tlas(&built) {
                session.set_tlas(gpu, Some(tlas))?;
            }
            self.session = Some(session);
            self.copied_spp = 0;
            log::line(&format!(
                "photo: session {}x{} ({} bounces, {} spp target) created in {:.0} ms",
                self.size[0],
                self.size[1],
                want.bounces,
                want.spp_target,
                t.elapsed().as_secs_f64() * 1e3
            ));
        }
        if self.slots.is_none() {
            let set = Arc::new(SlotSet::new(&self.dev, self.size, self.znear)?);
            composite::activate(set.clone());
            self.slots = Some(set);
        }
        let (Some(session), Some(slots)) = (self.session.as_mut(), self.slots.clone()) else {
            return Ok(true);
        };
        session.set_spp_target(s.photo_spp);
        if let Some((k, fence, spp)) = self.pending
            && gpu.completed(fence)?
        {
            slots.publish(k);
            self.copied_spp = spp;
            self.pending = None;
        }
        let p = session.step(gpu, s.photo_budget_ms)?;
        if self.pending.is_none()
            && p.depth_ready
            && p.spp_done > self.copied_spp
            && (p.done || self.last_copy.is_none_or(|t| t.elapsed() >= COPY_EVERY))
            && let Some(k) = slots.writable()
        {
            let slot = &slots.slots[k];
            let fence = session.copy_out(gpu, &slot.hdr.res, &slot.depth.res)?;
            self.pending = Some((k, fence, p.spp_done));
            self.last_copy = Some(Instant::now());
        }
        self.shared.spp_done.store(p.spp_done, Ordering::Relaxed);
        self.shared
            .spp_target
            .store(p.spp_target, Ordering::Relaxed);
        self.shared
            .ms_per_spp
            .store(p.ms_per_spp.to_bits(), Ordering::Relaxed);
        self.shared.set_phase(if p.done {
            "done"
        } else if !p.depth_ready {
            "first hits"
        } else {
            "accumulating"
        });
        if p.done && !self.logged_done {
            self.logged_done = true;
            log::line(&format!(
                "photo: {} spp done after {:.1} s ({:.1} ms per sample)",
                p.spp_done,
                self.started.elapsed().as_secs_f64(),
                p.ms_per_spp
            ));
        } else if !p.done {
            self.logged_done = false;
        }
        if self.shared.save.swap(false, Ordering::AcqRel)
            && let Err(e) = self.save(gpu, &s, &built)
        {
            log::line(&format!("photo: save failed: {e}"));
            lock(&self.shared.messages).push(format!("Photo not saved: {e}"));
        }
        Ok(true)
    }
}

impl Task for PhotoTask {
    fn step(&mut self, gpu: &mut Gpu, scene: Option<&Arc<Built>>) -> Result<bool, String> {
        match self.run(gpu, scene) {
            Ok(keep) => Ok(keep),
            Err(e) => {
                *lock(&self.shared.error) = Some(e.clone());
                self.release_slots();
                Err(e)
            }
        }
    }

    fn release_scene(&mut self) {
        if self.session.is_some() || self.hybrid.is_some() {
            log::line("photo: session dropped (the scene is being replaced)");
        }
        self.drop_session();
        if let Some(h) = self.hybrid.as_mut() {
            h.release_scene();
        }
    }

    fn abort(&mut self, why: &str) {
        log::line(&format!("photo: render task ended: {why}"));
        lock(&self.shared.error).get_or_insert_with(|| why.to_owned());
        self.release_slots();
        self.hybrid = None;
    }

    fn allows_live(&self) -> bool {
        // The hybrid display keeps the live TLAS for the ray-traced reflections.
        settings::get().photo_display == super::Display::Hybrid
    }
}

impl Drop for PhotoTask {
    fn drop(&mut self) {
        self.release_slots();
    }
}
