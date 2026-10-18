//! Hybrid photo terms in game (`photo_display = hybrid`, docs/raytracing.md section 13): the
//! engine keeps rendering its paused frame (fog, water, particles, characters, post, tone
//! mapping); only its lighting inputs are replaced with progressively accumulated ray-traced
//! terms through the S1 entry point (`ao::external`): `.y` = min(engine CSM [+ contact
//! shadows], RT sun visibility), `.x` = RT AO (`photo_ao_mode`: min with HBAO+/our AO, or
//! replace). Reflections accumulate in `ssr::rt_reflect` (photo mode of the FSR path).
//!
//! Hand-offs (no cross-API GPU waits inside a frame, the composite's frame margin):
//! 1. **Depth in:** the worker opens a [`DepthRequest`] (a D3D11-created shared R32F texture
//!    of the render region). The next FOG pass (`composite.rs`) records a compute pass that
//!    writes the game's view depth of the region into it and stores the frame's camera,
//!    projection jitter and sun direction ([`frame_from_view`]). The worker uses it
//!    [`composite::REUSE_FRAMES`] FOG passes later (the D3D11 work has executed) and copies
//!    it into the session (`HybridSession::set_frame`). Our own shared copy is used rather than
//!    Reflections' `shared_inputs().depth`: that one exists only while ray-traced reflections
//!    classify on the FSR path, and is rewritten every frame.
//! 2. **Terms out:** two slots of shared R32F textures (sun visibility, AO). The worker copies
//!    the running means into a slot that is not registered and that no pass read for
//!    [`REUSE_PASSES`] FOG passes (the detours stamp the FOG pass counter as their clock:
//!    the SSAO and SCREENSHADOWS passes precede FOG in a frame), waits for the D3D12 fence on
//!    the CPU, then registers the slot's textures as the external `.y` / `.x` sources.
//! 3. **Camera moves** (free-camera version): the sources are unregistered at once (main
//!    thread and worker), a new depth is captured and the accumulation restarts.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;

use void_rt::camera::RayBasis;
use void_rt::hybrid::{HybridFrame, HybridSession, HybridSettings, Term, TermProgress};
use void_rt::ingame::Gpu;

use super::composite;
use crate::ao::external::REUSE_PASSES;
use crate::ao::{ExternalMode, ExternalSource, Proj};
use crate::rt::device::{self, Device, Shared};
use crate::rt::scene::Built;
use crate::{log, settings};

/// Copies the means into a slot at most this often while accumulating.
const COPY_EVERY: Duration = Duration::from_millis(250);
/// Camera-move restarts are logged at most this often (one summary line; the free camera
/// moves the view every frame, ~100 lines/s in in-game run 1).
const MOVE_LOG_EVERY: Duration = Duration::from_secs(1);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ------------------------------------------------------------------ engine frame -> rays --

/// The frame's camera for the kernels from the engine view (FOG pass): `view` = view+29776
/// (row-major, column vectors, rows right / up / back, translation in column 3; `ssr.rs`
/// convention), `proj` = the parsed projection (P00, P11, jitter, depth terms), `sun_world`
/// = view+48344 (toward the sun), `size` = the render region. None if degenerate.
pub(crate) fn frame_from_view(
    view: &[f32; 16],
    proj: &Proj,
    sun_world: [f32; 3],
    size: [u32; 2],
) -> Option<HybridFrame> {
    let row = |i: usize| [view[4 * i], view[4 * i + 1], view[4 * i + 2]];
    let (right, up, back) = (row(0), row(1), row(2));
    let org: [f32; 3] = std::array::from_fn(|j| {
        -(view[j] * view[3] + view[4 + j] * view[7] + view[8 + j] * view[11])
    });
    let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let sl = len(sun_world);
    let ok = [right, up, back]
        .iter()
        .all(|v| (len(*v) - 1.0).abs() < 1e-2)
        && org.iter().all(|v| v.is_finite())
        && sl.is_finite()
        && sl > 1e-4
        && proj.p00.is_finite()
        && proj.p11.is_finite()
        && proj.p00.abs() > 1e-6
        && proj.p11.abs() > 1e-6
        && size[0] > 0
        && size[1] > 0;
    ok.then(|| HybridFrame {
        basis: RayBasis {
            org,
            fwd: back.map(|v| -v),
            right,
            up,
            tan_x: 1.0 / proj.p00,
            tan_y: 1.0 / proj.p11,
            width: size[0],
            height: size[1],
        },
        jitter: proj.jitter,
        sun_dir: sun_world.map(|v| v / sl),
        // Two R24 steps of view depth per z^2 (dz/dd = z^2 / depth_mul), as contact.rs.
        depth_step: 2.0 / (proj.depth_mul.abs().max(1e-12) * 16_777_215.0),
    })
}

// --------------------------------------------------------------------- depth capture --

/// A captured frame: the FOG pass counter when it was recorded, the frame for the kernels
/// and the region in the engine targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Captured {
    pub fog_frame: u64,
    pub frame: HybridFrame,
    pub region: [u32; 4],
}

/// A one-shot request for the game's view depth of the next FOG pass.
pub(crate) struct DepthRequest {
    /// R32F, region-sized; UAV on the D3D11 side.
    pub tex: Shared,
    pub size: [u32; 2],
    result: Mutex<Option<Captured>>,
}

// SAFETY: D3D11 views and D3D12 resources are free-threaded; the texture is written by the
// FOG pass once (before `result` is set) and read by the worker only after the frame margin.
unsafe impl Send for DepthRequest {}
// SAFETY: as above.
unsafe impl Sync for DepthRequest {}

impl DepthRequest {
    fn new(dev: &Device, size: [u32; 2]) -> Result<Self, String> {
        Ok(Self {
            tex: device::shared_texture(dev, DXGI_FORMAT_R32_FLOAT, size[0], size[1], true, false)?,
            size,
            result: Mutex::new(None),
        })
    }

    fn result(&self) -> Option<Captured> {
        *lock(&self.result)
    }

    /// FOG pass: the export was recorded for `c`; the request is closed.
    pub(crate) fn complete(self: &Arc<Self>, c: Captured) {
        *lock(&self.result) = Some(c);
        let mut r = lock(&REQUEST);
        if r.as_ref().is_some_and(|x| Arc::ptr_eq(x, self)) {
            *r = None;
        }
    }
}

static REQUEST: Mutex<Option<Arc<DepthRequest>>> = Mutex::new(None);

/// FOG pass: the open depth request, if any.
pub(crate) fn capture_request() -> Option<Arc<DepthRequest>> {
    lock(&REQUEST).clone()
}

fn open_request(req: Option<Arc<DepthRequest>>) {
    *lock(&REQUEST) = req;
}

// ------------------------------------------------------------------------ term slots --

struct TermSlot {
    sun: Shared,
    ao: Shared,
    sun_reads: Arc<AtomicU64>,
    ao_reads: Arc<AtomicU64>,
}

/// Two slots of R32F term textures (region-sized) and which one is registered.
struct TermSlots {
    slots: [TermSlot; 2],
    size: [u32; 2],
    published: Option<usize>,
    /// What the registration was made with (sun on, AO on, AO mode).
    registered: (bool, bool, ExternalMode),
}

impl TermSlots {
    fn new(dev: &Device, size: [u32; 2]) -> Result<Self, String> {
        let tex =
            || device::shared_texture(dev, DXGI_FORMAT_R32_FLOAT, size[0], size[1], false, true);
        let slot = || -> Result<TermSlot, String> {
            Ok(TermSlot {
                sun: tex()?,
                ao: tex()?,
                sun_reads: Arc::default(),
                ao_reads: Arc::default(),
            })
        };
        Ok(Self {
            slots: [slot()?, slot()?],
            size,
            published: None,
            registered: (false, false, ExternalMode::Min),
        })
    }

    /// A slot the worker may write now: not registered, and not read within the last
    /// `REUSE_PASSES` FOG passes.
    fn writable(&self) -> Option<usize> {
        let now = composite::fog_frames();
        (0..2).find(|&k| {
            let s = &self.slots[k];
            Some(k) != self.published
                && now >= s.ao_reads.load(Ordering::Acquire) + REUSE_PASSES
                && now >= s.sun_reads.load(Ordering::Acquire) + REUSE_PASSES
        })
    }

    /// Stamps the slot's read clocks with the current FOG pass count (a pass that loaded the
    /// registration just before it was removed reads at most in this frame).
    fn release(&self, k: usize) {
        let s = &self.slots[k];
        let now = composite::fog_frames();
        s.ao_reads.fetch_max(now, Ordering::AcqRel);
        s.sun_reads.fetch_max(now, Ordering::AcqRel);
    }

    /// Registers slot `k` (sun -> `.y` min, AO -> `.x` with `mode`), releasing the previous.
    fn publish(&mut self, k: usize, sun: bool, ao: bool, mode: ExternalMode) {
        let s = &self.slots[k];
        let src = |t: &Shared, reads: &Arc<AtomicU64>, mode| {
            t.srv.clone().map(|srv| ExternalSource {
                srv,
                size: self.size,
                mode,
                reads: reads.clone(),
                clock: composite::fog_frames,
            })
        };
        crate::contact::set_external_y(if sun {
            src(&s.sun, &s.sun_reads, ExternalMode::Min)
        } else {
            None
        });
        crate::ao::set_external_x(if ao {
            src(&s.ao, &s.ao_reads, mode)
        } else {
            None
        });
        if let Some(old) = self.published
            && old != k
        {
            self.release(old);
        }
        self.published = Some(k);
        self.registered = (sun, ao, mode);
    }

    fn unpublish(&mut self) {
        unregister();
        if let Some(k) = self.published.take() {
            self.release(k);
        }
    }
}

/// Removes both external sources (any thread; photo exit and camera moves).
pub(crate) fn unregister() {
    crate::contact::set_external_y(None);
    crate::ao::set_external_x(None);
}

// --------------------------------------------------------------------------- status --

/// For the overlay.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    pub phase: String,
    pub size: [u32; 2],
    pub sun: TermProgress,
    pub ao: TermProgress,
    /// Rays per pixel in the registered textures (sun, AO).
    pub shown: [u32; 2],
    /// GPU ms of the depth prepass.
    pub prep_ms: f32,
    /// (submits, longest GPU ms of one submit).
    pub submits: (usize, f32),
}

static STATUS: Mutex<Option<Status>> = Mutex::new(None);

/// The hybrid terms' state (None outside hybrid photo mode).
pub fn status() -> Option<Status> {
    lock(&STATUS).clone()
}

fn set_phase(p: &str) {
    lock(&STATUS).get_or_insert_with(Status::default).phase = p.to_owned();
}

/// Kernel settings from the settings file.
pub(crate) fn settings_from(s: &settings::Settings) -> HybridSettings {
    HybridSettings {
        sun: s.photo_sun,
        ao: s.photo_ao,
        sun_rays: s.photo_sun_rays,
        ao_rays: s.photo_ao_rays,
        target: s.photo_terms_spp,
        sun_radius_deg: s.photo_sun_radius,
        ao_radius: s.photo_ao_radius,
        ..HybridSettings::default()
    }
}

// ---------------------------------------------------------------------------- driver --

/// The worker side of the hybrid terms (owned by the photo task).
pub(crate) struct Hybrid {
    dev: Arc<Device>,
    /// Free-camera version the current capture / accumulation belongs to.
    epoch: Option<u64>,
    request: Option<Arc<DepthRequest>>,
    captured: Option<(Arc<DepthRequest>, Captured)>,
    session: Option<HybridSession>,
    slots: Option<TermSlots>,
    /// Slot being written, its fence and the rays it will show.
    pending: Option<(usize, u64, [u32; 2])>,
    last_copy: Option<Instant>,
    shown: [u32; 2],
    started: Instant,
    logged_done: bool,
    /// Camera-move restarts not logged yet and when the last summary was written.
    moves_unlogged: u32,
    moves_logged: Option<Instant>,
}

impl Hybrid {
    pub fn new(dev: Arc<Device>) -> Self {
        set_phase("waiting for the scene");
        Self {
            dev,
            epoch: None,
            request: None,
            captured: None,
            session: None,
            slots: None,
            pending: None,
            last_copy: None,
            shown: [0; 2],
            started: Instant::now(),
            logged_done: false,
            moves_unlogged: 0,
            moves_logged: None,
        }
    }

    /// Logs the camera-move restarts since the last summary, at most once per
    /// [`MOVE_LOG_EVERY`].
    fn log_moves(&mut self) {
        if self.moves_unlogged == 0
            || self
                .moves_logged
                .is_some_and(|t| t.elapsed() < MOVE_LOG_EVERY)
        {
            return;
        }
        log::line(&format!(
            "photo: camera moved, restarting the hybrid terms ({} restart{} since the last line)",
            self.moves_unlogged,
            if self.moves_unlogged == 1 { "" } else { "s" }
        ));
        self.moves_unlogged = 0;
        self.moves_logged = Some(Instant::now());
    }

    fn unpublish(&mut self) {
        if let Some(sl) = self.slots.as_mut() {
            sl.unpublish();
        } else {
            unregister();
        }
        self.pending = None;
        self.shown = [0; 2];
    }

    /// Drops the session and its hold on the scene (the scene is being replaced).
    pub fn release_scene(&mut self) {
        self.unpublish();
        self.session = None;
    }

    /// One worker tick: capture, (re)start, accumulate, publish. `epoch` = the free-camera
    /// version (0 without a free camera).
    pub fn step(
        &mut self,
        gpu: &mut Gpu,
        built: &Arc<Built>,
        s: &settings::Settings,
        epoch: u64,
    ) -> Result<(), String> {
        if self.epoch != Some(epoch) {
            if self.epoch.is_some() {
                self.moves_unlogged += 1;
            }
            self.epoch = Some(epoch);
            self.unpublish();
            self.captured = None;
            self.request = None;
            self.logged_done = false;
            self.started = Instant::now();
        }
        self.log_moves();
        let want = settings_from(s);
        // The game's depth of the current camera; a changed render size (resolution, render
        // scale) captures again.
        let region_size = composite::last_frame().map(|f| [f.region[2], f.region[3]]);
        if let Some((req, _)) = &self.captured
            && region_size.is_some_and(|s| s != req.size)
        {
            log::line("photo: render size changed, capturing the depth again");
            self.unpublish();
            self.captured = None;
        }
        let mut new_frame = false;
        if self.captured.is_none() {
            let Some(fog) = composite::last_frame() else {
                set_phase("waiting for a world frame");
                return Ok(());
            };
            let size = [fog.region[2], fog.region[3]];
            match self.request.clone() {
                None => {
                    let req = Arc::new(DepthRequest::new(&self.dev, size)?);
                    open_request(Some(req.clone()));
                    self.request = Some(req);
                }
                Some(req) => {
                    if let Some(c) = req.result()
                        && composite::fog_frames() >= c.fog_frame + composite::REUSE_FRAMES
                    {
                        self.captured = Some((req, c));
                        self.request = None;
                        new_frame = true;
                    } else if req.result().is_none() && req.size != size {
                        // The render size changed before the capture: ask again.
                        let req = Arc::new(DepthRequest::new(&self.dev, size)?);
                        open_request(Some(req.clone()));
                        self.request = Some(req);
                    }
                }
            }
            if !new_frame {
                set_phase("capturing the game depth");
                return Ok(());
            }
        }
        let Some((req, cap)) = self.captured.clone() else {
            return Ok(());
        };
        let size = req.size;
        // Session on the current scene and size; restarted by a new frame or settings.
        if self.session.as_ref().is_some_and(|x| {
            x.scene().generation() != built.scene.generation() || [x.width(), x.height()] != size
        }) {
            self.unpublish();
            self.session = None;
        }
        let t = Instant::now();
        match self.session.as_mut() {
            None => {
                let mut x = HybridSession::new(gpu, built.scene.clone(), cap.frame, want)?;
                // Dynamic objects frozen at photo entry occlude too (section 14.4).
                if let Some(tlas) = crate::rt::scene::photo_tlas(built) {
                    x.set_tlas(gpu, Some(tlas))?;
                }
                x.set_frame(gpu, cap.frame, want, &req.tex.res)?;
                self.session = Some(x);
                log::line(&format!(
                    "photo: hybrid terms {}x{} started in {:.0} ms (sun {} x {} rays, AO {} x {} rays within {} m, target {})",
                    size[0],
                    size[1],
                    t.elapsed().as_secs_f64() * 1e3,
                    u8::from(want.sun),
                    want.sun_rays,
                    u8::from(want.ao),
                    want.ao_rays,
                    want.ao_radius,
                    want.target
                ));
            }
            Some(x) if new_frame || x.settings().restart_needed(&want) => {
                self.pending = None;
                x.set_frame(gpu, cap.frame, want, &req.tex.res)?;
                if !new_frame {
                    log::line("photo: hybrid term settings changed, restarting");
                }
                self.shown = [0; 2];
                self.started = Instant::now();
                self.logged_done = false;
            }
            Some(x) => x.set_target(want.target),
        }
        if self.slots.as_ref().is_none_or(|sl| sl.size != size) {
            if let Some(sl) = self.slots.as_mut() {
                sl.unpublish();
            }
            self.slots = Some(TermSlots::new(&self.dev, size)?);
        }
        let (Some(x), Some(slots)) = (self.session.as_mut(), self.slots.as_mut()) else {
            return Ok(());
        };
        let mode = if s.photo_ao_replace {
            ExternalMode::Replace
        } else {
            ExternalMode::Min
        };
        if let Some((k, fence, rays)) = self.pending
            && gpu.completed(fence)?
        {
            slots.publish(k, want.sun, want.ao, mode);
            self.shown = rays;
            self.pending = None;
        } else if let Some(k) = slots.published
            && slots.registered != (want.sun, want.ao, mode)
        {
            // A term switched off or the AO mode changed: re-register at once.
            slots.publish(k, want.sun, want.ao, mode);
        }
        let p = x.step(gpu, s.photo_budget_ms)?;
        let rays = [
            if want.sun { p.sun.rays } else { 0 },
            if want.ao { p.ao.rays } else { 0 },
        ];
        let ready = p.prep_ready
            && (!want.sun || p.sun.rays > 0)
            && (!want.ao || p.ao.rays > 0)
            && (want.sun || want.ao);
        if self.pending.is_none()
            && ready
            && rays != self.shown
            && (p.done || self.last_copy.is_none_or(|t| t.elapsed() >= COPY_EVERY))
            && let Some(k) = slots.writable()
        {
            let slot = &slots.slots[k];
            let fence = x.copy_out(
                gpu,
                want.sun.then_some(&slot.sun.res),
                want.ao.then_some(&slot.ao.res),
            )?;
            self.pending = Some((k, fence, rays));
            self.last_copy = Some(Instant::now());
        }
        if p.done && !self.logged_done {
            self.logged_done = true;
            log::line(&format!(
                "photo: hybrid terms done after {:.1} s: sun {} rays ({:.2} ms per batch of {}), AO {} rays ({:.2} ms per batch of {}), prepass {:.2} ms",
                self.started.elapsed().as_secs_f64(),
                p.sun.rays,
                p.sun.ms_per_batch,
                p.sun.rays_per_batch,
                p.ao.rays,
                p.ao.ms_per_batch,
                p.ao.rays_per_batch,
                p.prep_ms
            ));
        }
        let (submits, longest) = x.submit_stats();
        *lock(&STATUS) = Some(Status {
            phase: if p.done {
                "done".into()
            } else if !p.prep_ready {
                "depth prepass".into()
            } else {
                "accumulating".into()
            },
            size,
            sun: p.sun,
            ao: p.ao,
            shown: self.shown,
            prep_ms: p.prep_ms,
            submits: (submits, longest as f32),
        });
        Ok(())
    }

    /// Writes the current means as grey PNGs `<base>_sun.png` / `<base>_ao.png` (debug,
    /// `photo_save_masks`). Blocking.
    pub fn save_masks(&mut self, gpu: &mut Gpu, base: &Path) -> Result<Vec<PathBuf>, String> {
        let Some(x) = self.session.as_mut() else {
            return Err("no hybrid terms yet".into());
        };
        let (w, h) = (x.width(), x.height());
        let mut out = Vec::new();
        for (t, on, suffix) in [
            (Term::Sun, x.settings().sun, "sun"),
            (Term::Ao, x.settings().ao, "ao"),
        ] {
            if !on {
                continue;
            }
            let v = x.read(gpu, t)?;
            let px: Vec<u8> = v
                .iter()
                .flat_map(|a| {
                    let g = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
                    [g, g, g]
                })
                .collect();
            let path = PathBuf::from(format!("{}_{suffix}.png", base.display()));
            void_rt::image::write_png(&path, w, h, &px)?;
            out.push(path);
        }
        Ok(out)
    }

    /// Rays per pixel (sun, AO) of the current accumulation (for the save notes).
    pub fn rays(&self) -> Option<[u32; 2]> {
        lock(&STATUS).as_ref().map(|s| [s.sun.rays, s.ao.rays])
    }
}

impl Drop for Hybrid {
    fn drop(&mut self) {
        self.unpublish();
        let mut r = lock(&REQUEST);
        if r.as_ref()
            .is_some_and(|x| self.request.as_ref().is_some_and(|m| Arc::ptr_eq(x, m)))
        {
            *r = None;
        }
        drop(r);
        *lock(&STATUS) = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Engine view matrix (row-major, column vectors) of a camera at `c` yawed by `yaw`
    /// about +Z (world Z-up), looking along +X at yaw 0, as contact_offline builds it.
    fn view(c: [f32; 3], yaw: f32) -> [f32; 16] {
        let (s, co) = yaw.sin_cos();
        let fwd = [co, s, 0.0];
        let right = [s, -co, 0.0];
        let up = [0.0, 0.0, 1.0];
        let back = fwd.map(|v| -v);
        let dot = |a: [f32; 3]| a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
        [
            right[0],
            right[1],
            right[2],
            -dot(right), //
            up[0],
            up[1],
            up[2],
            -dot(up), //
            back[0],
            back[1],
            back[2],
            -dot(back), //
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }

    #[test]
    fn frame_reconstructs_engine_positions() {
        let (w, h) = (1920u32, 800u32);
        let data =
            crate::ao::engine_projection(w as f32 / h as f32, 1.0, 0.01, 7000.0, [0.0004, -0.0003]);
        let proj = crate::ao::projection(&data, 1).unwrap();
        let v = view([5.0, 1.6, 2.0], 0.5);
        let f = frame_from_view(&v, &proj, [0.0, 0.3, 0.9], [w, h]).unwrap();
        assert!((f.basis.org[0] - 5.0).abs() < 1e-4 && (f.basis.org[2] - 2.0).abs() < 1e-4);
        // A world point -> engine view space -> NDC with jitter -> pixel; the kernel's
        // reconstruction (pixel, view depth) must give the point back.
        let p = [9.0, 4.0, 1.0];
        let r =
            |i: usize| v[4 * i] * p[0] + v[4 * i + 1] * p[1] + v[4 * i + 2] * p[2] + v[4 * i + 3];
        let (x, y, z) = (r(0), r(1), -r(2));
        let ndc = [
            proj.p00 * x / z + proj.jitter[0],
            proj.p11 * y / z + proj.jitter[1],
        ];
        let pix = [
            (ndc[0] + 1.0) * 0.5 * w as f32,
            (1.0 - ndc[1]) * 0.5 * h as f32,
        ];
        let b = &f.basis;
        let n = [
            2.0 * pix[0] / w as f32 - 1.0 - f.jitter[0],
            1.0 - 2.0 * pix[1] / h as f32 - f.jitter[1],
        ];
        let q: [f32; 3] = std::array::from_fn(|k| {
            b.org[k] + z * (b.fwd[k] + n[0] * b.tan_x * b.right[k] + n[1] * b.tan_y * b.up[k])
        });
        for k in 0..3 {
            assert!((q[k] - p[k]).abs() < 1e-3, "{q:?} vs {p:?}");
        }
        let sl = f.sun_dir.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((sl - 1.0).abs() < 1e-5);
        assert!(frame_from_view(&[0.0; 16], &proj, [0.0, 0.0, 1.0], [w, h]).is_none());
    }
}
