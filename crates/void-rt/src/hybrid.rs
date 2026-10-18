//! Hybrid photo terms (docs/raytracing.md section 13): ray-traced sun visibility and ambient
//! occlusion for the game's own paused frame. The game keeps rendering (fog, water,
//! particles, characters, post, tone mapping); the host replaces only the lighting inputs
//! `_SSAO_ScreenShadows.y` (sun visibility, `min` with the engine's CSM so characters, which
//! are not in the BVH, still cast) and `.x` (AO of the indirect term) with these means.
//!
//! Input: the game's view depth of the render region (metres along the forward axis, 0 =
//! sky), copied from a host texture (R32F) or uploaded (offline), plus the camera basis,
//! projection jitter and sun direction of that frame ([`HybridFrame`]). A prepass
//! (`photo_hybrid_prep`, once per frame) traces our primary hit per pixel: where it agrees
//! with the game's depth (within max(5 cm, 3 %), either side: coarser LOD, plank gaps,
//! decals) the receiver becomes our hit with that triangle's normal, so the rays start on our
//! own surface along its own normal; elsewhere (characters and objects we lack) the pixel
//! is rejected: the sun term is 1 there (the engine's CSM decides through `min`), AO falls
//! back to the game depth with a depth-rebuilt normal.
//!
//! Kernels `photo_sun_vis` / `photo_ao` (`photo.hlsl`, root layout [`crate::photo::LAYOUT`]):
//! per dispatch `rays` rays per pixel over a row tile, accumulated as a running mean per
//! pixel in R32F-copy-ready buffers ([`crate::photo::depth_pitch`] rows). Scheduling as
//! [`crate::ingame::PhotoSession::step`]: non-blocking, at most one submission in flight,
//! tiles packed by their last measured GPU time into the per-call budget (one tile while
//! unmeasured), at most [`MAX_DISPATCHES_PER_SUBMIT`] dispatches. The prepass tiles come
//! first, then batches alternate between the terms until each reaches its target. The sun
//! term's output (`copy_out`, `read`) passes an edge-aware 3x3 filter first
//! (`photo_hybrid_filter`: neighbours on the receiver's own plane only).

use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS};

use crate::camera::RayBasis;
use crate::gpu::{Buffer, Gpu, SubmitGuard};
use crate::ingame::{DEFAULT_TILE_PIXELS, GpuScene, SceneTlas};
use crate::photo::{
    self, Lighting, MAX_DISPATCHES_PER_SUBMIT, PARAM_WORDS, PathSettings, depth_pitch,
};

/// One of the two terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// Sun visibility (`.y`).
    Sun,
    /// Ambient occlusion (`.x`).
    Ao,
}

impl Term {
    pub const ALL: [Term; 2] = [Term::Sun, Term::Ao];

    fn entry(self) -> &'static str {
        match self {
            Term::Sun => "photo_sun_vis",
            Term::Ao => "photo_ao",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Term::Sun => "sun shadows",
            Term::Ao => "ambient occlusion",
        }
    }
}

/// The game frame the terms are traced for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridFrame {
    /// Camera of the frame; `width x height` = the render region.
    pub basis: RayBasis,
    /// Projection jitter (NDC) of the frame (`ndc = P view / z + jitter`).
    pub jitter: [f32; 2],
    /// World direction toward the sun (unit).
    pub sun_dir: [f32; 3],
    /// Depth quantisation per z^2 (two R24 steps: `2 / (depth_mul * (2^24 - 1))`), added to
    /// the ray-origin offset.
    pub depth_step: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridSettings {
    pub sun: bool,
    pub ao: bool,
    /// Rays per pixel and pass (dispatch over a tile), 1..=[`MAX_RAYS_PER_DISPATCH`]; the
    /// passes per frame follow the GPU budget of `step`.
    pub sun_rays: u32,
    pub ao_rays: u32,
    /// Rays per pixel to accumulate per term (rounded up to whole batches).
    pub target: u32,
    /// Sun disc angular radius (degrees).
    pub sun_radius_deg: f32,
    /// AO ray length (metres).
    pub ao_radius: f32,
    /// Pixels per dispatch at most (whole rows, a multiple of 4).
    pub tile_pixels: u32,
}

impl Default for HybridSettings {
    fn default() -> Self {
        Self {
            sun: true,
            ao: true,
            sun_rays: 4,
            ao_rays: 4,
            target: 256,
            sun_radius_deg: 0.27,
            ao_radius: 1.5,
            tile_pixels: DEFAULT_TILE_PIXELS / 2,
        }
    }
}

impl HybridSettings {
    pub fn enabled(&self, t: Term) -> bool {
        match t {
            Term::Sun => self.sun,
            Term::Ao => self.ao,
        }
    }

    pub fn rays(&self, t: Term) -> u32 {
        match t {
            Term::Sun => self.sun_rays,
            Term::Ao => self.ao_rays,
        }
        .clamp(1, MAX_RAYS_PER_DISPATCH)
    }

    /// Batches per term to reach `target`.
    pub fn batches(&self, t: Term) -> u32 {
        self.target.max(1).div_ceil(self.rays(t))
    }

    /// Whether `other` needs a restart of the accumulation (anything but the target).
    pub fn restart_needed(&self, other: &Self) -> bool {
        Self {
            target: other.target,
            ..*self
        } != *other
    }

    /// (rows per tile, tiles) for a `width x height` image (`tile_pixels`, whole rows, a
    /// multiple of 4).
    pub fn tiling(&self, width: u32, height: u32) -> (u32, u32) {
        let rows = ((self.tile_pixels / width.max(1)).max(4) / 4 * 4).min(height);
        (rows, height.div_ceil(rows))
    }
}

/// Rays per pixel and dispatch at most (O, hybrid_check RX 7900 XTX, 3440x1440: 3.0 ms per
/// ray and pixel at 4 rays per dispatch in 88-row tiles; 64 rays per dispatch cost 5.1 ms per
/// ray in 36-row tiles (18.8 ms per tile) and 30 ms per ray in 4-row tiles: long per-thread
/// ray loops and small dispatches hide the ray latency badly).
pub const MAX_RAYS_PER_DISPATCH: u32 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TermProgress {
    pub enabled: bool,
    /// Rays per pixel in the mean (GPU finished).
    pub rays: u32,
    pub target: u32,
    /// GPU ms of the last complete batch (all tiles).
    pub ms_per_batch: f32,
    pub rays_per_batch: u32,
}

impl TermProgress {
    pub fn done(&self) -> bool {
        !self.enabled || self.rays >= self.target
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HybridProgress {
    pub sun: TermProgress,
    pub ao: TermProgress,
    /// The depth prepass finished (the terms run after it).
    pub prep_ready: bool,
    /// GPU ms of the depth prepass (all tiles).
    pub prep_ms: f32,
    /// A submission is executing.
    pub busy: bool,
    /// Both terms reached their target and nothing is in flight.
    pub done: bool,
}

impl HybridProgress {
    pub fn term(&self, t: Term) -> &TermProgress {
        match t {
            Term::Sun => &self.sun,
            Term::Ao => &self.ao,
        }
    }
}

/// One dispatch: the depth prepass (`term` None, batch 0) or a term batch, over one tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Item {
    term: Option<Term>,
    batch: u32,
    tile: u32,
}

#[derive(Default)]
struct TermState {
    /// Batches complete on the GPU.
    done: u32,
    next_batch: u32,
    next_tile: u32,
    /// Latest GPU ms of each tile.
    tile_est: Vec<Option<f64>>,
    cur_ms: f64,
    last_batch_ms: f64,
}

struct Pending {
    fence: u64,
    items: Vec<Item>,
}

/// Progressive sun visibility and AO of one game frame.
pub struct HybridSession {
    // First field: dropped first, so the buffers below outlive the GPU work using them.
    guard: Option<SubmitGuard>,
    scene: Arc<GpuScene>,
    frame: HybridFrame,
    settings: HybridSettings,
    params: Buffer,
    /// Receiver depth (u1; our surface after the prepass, negative = rejected).
    depth: Buffer,
    /// Packed receiver normal of the prepass (u2).
    normal: Buffer,
    sun: Buffer,
    ao: Buffer,
    /// Filtered sun visibility (u0), the sun term's output.
    sun_out: Buffer,
    tile_rows: u32,
    tiles: u32,
    terms: [TermState; 2],
    /// The depth prepass as a one-batch "term".
    prep: TermState,
    pending: Option<Pending>,
    submits: usize,
    max_submit_ms: f64,
    /// Traced instead of the scene's full TLAS (dynamic objects frozen at entry).
    tlas: Option<Arc<SceneTlas>>,
}

/// Timestamp slots at the top of the query heap (as `PhotoSession`; the two sessions never
/// share a `Gpu` at the same time).
fn ts_first(gpu: &Gpu) -> u32 {
    gpu.max_queries - 2 * MAX_DISPATCHES_PER_SUBMIT
}

fn idx(t: Term) -> usize {
    match t {
        Term::Sun => 0,
        Term::Ao => 1,
    }
}

/// The `Params` block of the hybrid kernels: the frame's camera basis, the sun direction and
/// disc, the AO radius (word 43) and the depth quantisation (word 47).
pub fn params(scene: &GpuScene, frame: &HybridFrame, s: &HybridSettings) -> [u32; PARAM_WORDS] {
    let lighting = Lighting {
        sun_dir: frame.sun_dir,
        sun_irradiance: [0.0; 3],
        sun_radius_deg: s.sun_radius_deg,
        sky_zenith: [0.0; 3],
        sky_horizon: [0.0; 3],
        sky_scale: 0.0,
    };
    let path = PathSettings {
        max_bounces: 0,
        rr_start: 0,
        clamp_indirect: 0.0,
        ris_m: 1,
    };
    let mut w = photo::params(&frame.basis, 0.0, &lighting, &path, scene.tables());
    w[43] = s.ao_radius.max(0.0).to_bits();
    w[47] = frame.depth_step.max(0.0).to_bits();
    w
}

impl HybridSession {
    /// Video memory of a `width x height` session (depth, normal, two means and the
    /// filtered sun term, packed rows).
    pub fn buffer_bytes(width: u32, height: u32) -> u64 {
        5 * u64::from(height) * depth_pitch(width) + 256
    }

    /// Allocates the buffers and uploads the frame's parameters (blocking, short). The depth
    /// must be supplied with [`HybridSession::set_frame`] or [`HybridSession::set_depth`]
    /// before stepping. `gpu` must be the device `scene` was uploaded with.
    pub fn new(
        gpu: &mut Gpu,
        scene: Arc<GpuScene>,
        frame: HybridFrame,
        settings: HybridSettings,
    ) -> Result<Self, String> {
        let (w, h) = (frame.basis.width, frame.basis.height);
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            return Err(format!("image size {w}x{h}"));
        }
        gpu.check_vram(Self::buffer_bytes(w, h), "the hybrid photo buffers")?;
        let bytes = u64::from(h) * depth_pitch(w);
        let params = gpu.upload(&params(&scene, &frame, &settings), false)?;
        let depth = gpu.buffer(bytes, true)?;
        let normal = gpu.buffer(bytes, true)?;
        let sun = gpu.buffer(bytes, true)?;
        let ao = gpu.buffer(bytes, true)?;
        let sun_out = gpu.buffer(bytes, true)?;
        for b in [&depth, &normal, &sun, &ao, &sun_out] {
            gpu.clear(b)?;
        }
        gpu.submit()?;
        let (rows, tiles) = settings.tiling(w, h);
        let term = || TermState {
            tile_est: vec![None; tiles as usize],
            ..TermState::default()
        };
        Ok(Self {
            guard: None,
            scene,
            frame,
            settings,
            params,
            depth,
            normal,
            sun,
            ao,
            sun_out,
            tile_rows: rows,
            tiles,
            terms: [term(), term()],
            prep: term(),
            pending: None,
            submits: 0,
            max_submit_ms: 0.0,
            tlas: None,
        })
    }

    pub fn width(&self) -> u32 {
        self.frame.basis.width
    }

    pub fn height(&self) -> u32 {
        self.frame.basis.height
    }

    pub fn frame(&self) -> &HybridFrame {
        &self.frame
    }

    pub fn settings(&self) -> &HybridSettings {
        &self.settings
    }

    pub fn scene(&self) -> &Arc<GpuScene> {
        &self.scene
    }

    /// Traces `tlas` (a [`SceneTlas`] of this scene, e.g. `Dynamics::full` frozen at photo
    /// entry) instead of the full TLAS, or the full TLAS again with None; waits for the
    /// submission in flight and restarts the prepass and both terms.
    pub fn set_tlas(&mut self, gpu: &mut Gpu, tlas: Option<Arc<SceneTlas>>) -> Result<(), String> {
        self.finish(gpu)?;
        self.tlas = tlas;
        self.reset_terms();
        Ok(())
    }

    /// Rows per tile and tiles per batch.
    pub fn tiling(&self) -> (u32, u32) {
        (self.tile_rows, self.tiles)
    }

    /// (submits, largest GPU ms of one submit) of `step`.
    pub fn submit_stats(&self) -> (usize, f64) {
        (self.submits, self.max_submit_ms)
    }

    /// Latest GPU ms of each measured tile of `t` (None: the depth prepass), top to bottom.
    pub fn tile_ms(&self, t: Option<Term>) -> Vec<f64> {
        self.state(t).tile_est.iter().flatten().copied().collect()
    }

    /// Changes the ray target (no restart).
    pub fn set_target(&mut self, target: u32) {
        self.settings.target = target;
    }

    /// Clears the progress of the prepass and both terms; with a new tiling also the tile
    /// cost estimates.
    fn reset_terms(&mut self) {
        let (rows, tiles) = self.settings.tiling(self.width(), self.height());
        let retile = (rows, tiles) != (self.tile_rows, self.tiles);
        self.tile_rows = rows;
        self.tiles = tiles;
        for t in self.terms.iter_mut().chain(std::iter::once(&mut self.prep)) {
            t.done = 0;
            t.next_batch = 0;
            t.next_tile = 0;
            t.cur_ms = 0.0;
            t.last_batch_ms = 0.0;
            if retile {
                t.tile_est = vec![None; tiles as usize];
            }
        }
    }

    fn state(&self, term: Option<Term>) -> &TermState {
        term.map_or(&self.prep, |t| &self.terms[idx(t)])
    }

    fn state_mut(&mut self, term: Option<Term>) -> &mut TermState {
        match term {
            None => &mut self.prep,
            Some(t) => &mut self.terms[idx(t)],
        }
    }

    /// New frame (camera moved) and/or settings: waits for the submission in flight, uploads
    /// the parameters, copies `depth_tex` (R32F view depth of the region, exactly
    /// `width x height`, COMMON before and after) into the session and restarts both terms.
    /// Blocking, one short submit.
    pub fn set_frame(
        &mut self,
        gpu: &mut Gpu,
        frame: HybridFrame,
        settings: HybridSettings,
        depth_tex: &ID3D12Resource,
    ) -> Result<(), String> {
        self.restart(gpu, frame, settings)?;
        let (w, h) = (self.width(), self.height());
        gpu.transition_resource(
            depth_tex,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let r = gpu.copy_texture_to_buffer(
            depth_tex,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_TYPELESS,
            w,
            h,
            &self.depth,
            depth_pitch(w),
        );
        gpu.transition_resource(
            depth_tex,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        if let Err(e) = r {
            let _ = gpu.discard();
            return Err(e);
        }
        gpu.submit()
    }

    /// As [`HybridSession::set_frame`] with the depth from the CPU (`width x height`
    /// row-major view depths, 0 = sky; offline checks).
    pub fn set_depth(
        &mut self,
        gpu: &mut Gpu,
        frame: HybridFrame,
        settings: HybridSettings,
        depth: &[f32],
    ) -> Result<(), String> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        if depth.len() != w * h {
            return Err(format!(
                "depth has {} values, {w}x{h} expected",
                depth.len()
            ));
        }
        self.restart(gpu, frame, settings)?;
        let row = (depth_pitch(w as u32) / 4) as usize;
        let mut packed = vec![0f32; row * h];
        for (dst, src) in packed.chunks_mut(row).zip(depth.chunks(w)) {
            dst[..w].copy_from_slice(src);
        }
        let b = gpu.upload(&packed, true)?;
        if b.size != self.depth.size {
            return Err(format!(
                "depth upload {} bytes, session buffer {}",
                b.size, self.depth.size
            ));
        }
        self.depth = b;
        Ok(())
    }

    fn restart(
        &mut self,
        gpu: &mut Gpu,
        frame: HybridFrame,
        settings: HybridSettings,
    ) -> Result<(), String> {
        if [frame.basis.width, frame.basis.height] != [self.width(), self.height()] {
            return Err(format!(
                "frame {}x{} differs from the session {}x{}",
                frame.basis.width,
                frame.basis.height,
                self.width(),
                self.height()
            ));
        }
        self.finish(gpu)?;
        self.frame = frame;
        self.settings = settings;
        self.params = gpu.upload(&params(&self.scene, &frame, &settings), false)?;
        self.reset_terms();
        Ok(())
    }

    fn progress(&self) -> HybridProgress {
        let term = |t: Term| {
            let st = &self.terms[idx(t)];
            let rays = self.settings.rays(t);
            TermProgress {
                enabled: self.settings.enabled(t),
                rays: st.done * rays,
                target: self.settings.batches(t) * rays,
                ms_per_batch: st.last_batch_ms as f32,
                rays_per_batch: rays,
            }
        };
        let (sun, ao) = (term(Term::Sun), term(Term::Ao));
        let busy = self.pending.is_some();
        HybridProgress {
            sun,
            ao,
            prep_ready: self.prep.done > 0,
            prep_ms: self.prep.last_batch_ms as f32,
            busy,
            done: !busy && sun.done() && ao.done(),
        }
    }

    /// The next dispatch: the depth prepass first; then an unfinished batch, else the enabled
    /// term with fewer batches recorded (sun first on ties) below its target.
    fn next_item(&self) -> Option<Item> {
        if self.prep.next_batch == 0 {
            return Some(Item {
                term: None,
                batch: 0,
                tile: self.prep.next_tile,
            });
        }
        let open = |t: Term| {
            let st = &self.terms[idx(t)];
            (self.settings.enabled(t) && st.next_batch < self.settings.batches(t)).then_some(st)
        };
        let pick = Term::ALL
            .into_iter()
            .filter_map(|t| open(t).map(|st| (t, st)))
            .min_by_key(|(t, st)| (st.next_tile == 0, st.next_batch, idx(*t)))?;
        Some(Item {
            term: Some(pick.0),
            batch: pick.1.next_batch,
            tile: pick.1.next_tile,
        })
    }

    fn advance(&mut self, item: Item) {
        let tiles = self.tiles;
        let st = self.state_mut(item.term);
        st.next_tile += 1;
        if st.next_tile == tiles {
            st.next_tile = 0;
            st.next_batch += 1;
        }
    }

    fn band(&self, tile: u32) -> (u32, u32) {
        let y0 = tile * self.tile_rows;
        (y0, self.tile_rows.min(self.height() - y0))
    }

    /// Expected GPU ms of `item`: its tile's last measurement; None (unknown) otherwise, so
    /// an unmeasured tile goes alone into a submit. Other tiles are no guide: the prepass
    /// runs each tile once, and rows differ several-fold (O, hybrid_check: sky rows at the
    /// top made eight prepass tiles estimated from the first one take 14.9 ms GPU).
    fn estimate(&self, item: Item) -> Option<f64> {
        self.state(item.term)
            .tile_est
            .get(item.tile as usize)
            .copied()
            .flatten()
    }

    fn srvs(&self) -> [Option<&Buffer>; 8] {
        let t = self.scene.tables();
        [
            Some(
                self.tlas
                    .as_ref()
                    .map_or(&self.scene.full_tlas().buffer, |t| &t.tlas.buffer),
            ),
            Some(&t.geometry),
            Some(&t.records),
            Some(&t.materials),
            Some(&t.lights),
            Some(&t.cells),
            Some(&t.light_indices),
            Some(&self.params),
        ]
    }

    fn constants(&self, batch: u32, rays: u32, y0: u32) -> [u32; 8] {
        [
            self.width(),
            self.height(),
            batch,
            rays,
            0,
            y0,
            self.frame.jitter[0].to_bits(),
            self.frame.jitter[1].to_bits(),
        ]
    }

    fn record(&self, gpu: &Gpu, item: Item) -> Result<(), String> {
        let (d, n) = (Some(&self.depth), Some(&self.normal));
        let uav = match item.term {
            None => [None, d, n, None, None],
            Some(Term::Sun) => [None, d, n, Some(&self.sun), None],
            Some(Term::Ao) => [None, d, n, None, Some(&self.ao)],
        };
        let (y0, rows) = self.band(item.tile);
        let rays = item.term.map_or(0, |t| self.settings.rays(t));
        self.scene.programs().dispatch_rows(
            gpu,
            item.term.map_or("photo_hybrid_prep", Term::entry),
            &self.constants(item.batch, rays, y0),
            &self.srvs(),
            &uav,
            Some(&self.scene.tables().textures),
            (self.width(), self.height()),
            (y0, rows),
        )
    }

    /// Records the edge-aware filter of the sun term (`sun` -> `sun_out`, whole image, one
    /// dispatch: a 3x3 gather, well under a millisecond at 3440x1440) and a UAV barrier.
    fn record_filter(&self, gpu: &Gpu) -> Result<(), String> {
        let uav = [
            Some(&self.sun_out),
            Some(&self.depth),
            Some(&self.normal),
            Some(&self.sun),
            None,
        ];
        self.scene.programs().dispatch_rows(
            gpu,
            "photo_hybrid_filter",
            &self.constants(0, 0, 0),
            &self.srvs(),
            &uav,
            Some(&self.scene.tables().textures),
            (self.width(), self.height()),
            (0, self.height()),
        )?;
        gpu.uav_barrier();
        Ok(())
    }

    fn collect(&mut self, gpu: &mut Gpu) -> Result<bool, String> {
        let Some(p) = &self.pending else {
            return Ok(true);
        };
        if !gpu.completed(p.fence)? {
            return Ok(false);
        }
        gpu.poll()?;
        let Some(p) = self.pending.take() else {
            return Ok(true);
        };
        let ts = gpu.read_timestamps(ts_first(gpu), 2 * p.items.len() as u32)?;
        let mut total = 0.0;
        for (k, item) in p.items.iter().enumerate() {
            let ms = gpu.ticks_to_ms(ts[2 * k], ts[2 * k + 1]);
            total += ms;
            let tiles = self.tiles;
            let st = self.state_mut(item.term);
            st.tile_est[item.tile as usize] = Some(ms);
            st.cur_ms += ms;
            if item.tile + 1 == tiles {
                st.done = item.batch + 1;
                st.last_batch_ms = st.cur_ms;
                st.cur_ms = 0.0;
            }
        }
        self.submits += 1;
        self.max_submit_ms = self.max_submit_ms.max(total);
        Ok(true)
    }

    /// Advances the accumulation without blocking (see the module doc). A host calling it
    /// once per frame keeps at most one submission of about `budget_ms` in flight.
    pub fn step(&mut self, gpu: &mut Gpu, budget_ms: f32) -> Result<HybridProgress, String> {
        if !self.collect(gpu)? {
            return Ok(self.progress());
        }
        let budget = f64::from(budget_ms.max(0.1));
        let first = ts_first(gpu);
        let mut items = Vec::new();
        let mut cost = 0.0;
        while (items.len() as u32) < MAX_DISPATCHES_PER_SUBMIT {
            let Some(item) = self.next_item() else {
                break;
            };
            let est = self.estimate(item);
            match est {
                None if !items.is_empty() => break,
                Some(e) if !items.is_empty() && cost + e > budget => break,
                _ => {}
            }
            let k = items.len() as u32;
            gpu.timestamp(first + 2 * k);
            if let Err(e) = self.record(gpu, item) {
                let _ = gpu.discard();
                return Err(e);
            }
            gpu.uav_barrier();
            gpu.timestamp(first + 2 * k + 1);
            items.push(item);
            self.advance(item);
            cost += est.unwrap_or(budget);
            if est.is_none() {
                break;
            }
        }
        if items.is_empty() {
            return Ok(self.progress());
        }
        gpu.resolve_timestamps(first, 2 * items.len() as u32)?;
        let fence = gpu.submit_async()?;
        self.guard = Some(gpu.guard());
        self.pending = Some(Pending { fence, items });
        Ok(self.progress())
    }

    /// Blocks until the submission in flight (if any) finished and accounts it.
    pub fn finish(&mut self, gpu: &mut Gpu) -> Result<HybridProgress, String> {
        if let Some(p) = &self.pending {
            let fence = p.fence;
            if !gpu.completed(fence)? {
                gpu.wait_idle()?;
                if !gpu.completed(fence)? {
                    return Err(format!("hybrid: fence {fence} not reached after wait"));
                }
            }
        }
        self.collect(gpu)?;
        Ok(self.progress())
    }

    /// The running mean of `t` (unfiltered).
    fn buffer(&self, t: Term) -> &Buffer {
        match t {
            Term::Sun => &self.sun,
            Term::Ao => &self.ao,
        }
    }

    /// The output of `t`: the filtered sun term (valid after `record_filter`), the AO mean.
    fn output(&self, t: Term) -> &Buffer {
        match t {
            Term::Sun => &self.sun_out,
            Term::Ao => &self.ao,
        }
    }

    /// Copies the current outputs (the sun term after the edge-aware filter, the AO mean)
    /// into host textures (R32_FLOAT or TYPELESS, exactly `width x height`, COMMON before
    /// and after; None = skip that term). Submits without waiting (the queue orders it after
    /// a submission still in flight) and returns the fence value: poll [`Gpu::completed`]
    /// before the host reads the textures.
    pub fn copy_out(
        &mut self,
        gpu: &mut Gpu,
        sun_tex: Option<&ID3D12Resource>,
        ao_tex: Option<&ID3D12Resource>,
    ) -> Result<u64, String> {
        let (w, h) = (self.width(), self.height());
        if sun_tex.is_some()
            && let Err(e) = self.record_filter(gpu)
        {
            let _ = gpu.discard();
            return Err(e);
        }
        for (t, tex) in [(Term::Sun, sun_tex), (Term::Ao, ao_tex)] {
            let Some(tex) = tex else { continue };
            gpu.transition_resource(
                tex,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let r = gpu.copy_buffer_to_texture(
                self.output(t),
                depth_pitch(w),
                DXGI_FORMAT_R32_FLOAT,
                DXGI_FORMAT_R32_TYPELESS,
                w,
                h,
                tex,
            );
            gpu.transition_resource(
                tex,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
            if let Err(e) = r {
                let _ = gpu.discard();
                return Err(e);
            }
        }
        let v = gpu.submit_async()?;
        self.guard = Some(gpu.guard());
        Ok(v)
    }

    /// `b` (packed rows) as `width x height` row-major values.
    fn read_buffer(&self, gpu: &mut Gpu, b: &Buffer) -> Result<Vec<f32>, String> {
        let (w, h) = (self.width() as usize, self.height() as usize);
        let row = (depth_pitch(w as u32) / 4) as usize;
        let raw: Vec<f32> = gpu.read(b, row * h)?;
        Ok(raw
            .chunks(row)
            .flat_map(|r| r[..w].iter().copied())
            .collect())
    }

    /// The current output of `t` per pixel (`width x height`, row-major; the sun term after
    /// the edge-aware filter, as `copy_out`). Blocking.
    pub fn read(&mut self, gpu: &mut Gpu, t: Term) -> Result<Vec<f32>, String> {
        self.finish(gpu)?;
        if t == Term::Sun {
            if let Err(e) = self.record_filter(gpu) {
                let _ = gpu.discard();
                return Err(e);
            }
            gpu.submit()?;
        }
        self.read_buffer(gpu, self.output(t))
    }

    /// The running mean of `t` before the filter (offline checks). Blocking.
    pub fn read_mean(&mut self, gpu: &mut Gpu, t: Term) -> Result<Vec<f32>, String> {
        self.finish(gpu)?;
        self.read_buffer(gpu, self.buffer(t))
    }

    /// The receiver depth after the prepass (`width x height`; our surface's depth, the
    /// negated game depth for rejected pixels, 0 = sky). Blocking.
    pub fn read_depth(&mut self, gpu: &mut Gpu) -> Result<Vec<f32>, String> {
        self.finish(gpu)?;
        self.read_buffer(gpu, &self.depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_and_ray_cap() {
        let s = HybridSettings::default();
        assert_eq!(s.tiling(3440, 1440), (88, 17));
        assert_eq!(s.tiling(2304, 968), (132, 8));
        assert_eq!(s.tiling(96, 40), (40, 1));
        let many = HybridSettings { ao_rays: 64, ..s };
        assert_eq!(many.rays(Term::Ao), MAX_RAYS_PER_DISPATCH);
    }

    #[test]
    fn batches_and_restart_rules() {
        let s = HybridSettings::default();
        assert_eq!(s.batches(Term::Sun), 64);
        let odd = HybridSettings {
            target: 10,
            sun_rays: 4,
            ..s
        };
        assert_eq!(odd.batches(Term::Sun), 3);
        assert!(!s.restart_needed(&HybridSettings { target: 512, ..s }));
        assert!(s.restart_needed(&HybridSettings {
            ao_radius: 3.0,
            ..s
        }));
        assert_eq!(HybridSettings { sun_rays: 0, ..s }.rays(Term::Sun), 1);
    }
}
