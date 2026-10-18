//! Runtime settings: `void-rs/local/inject/void-inject.ini` (`key = value`, `#`/`;`
//! comments). Read at startup; a missing file is created with the defaults so the
//! active configuration is always visible next to the log. Unknown keys are logged.
//!
//! Two copies exist: the *active* settings ([`get`], read by the hooks) and the *file*
//! settings ([`file`], edited by the overlay and written by [`save`]). [`edit`] changes
//! the file copy and copies only the keys marked live in [`KEYS`] into the active copy;
//! restart-only keys (hooks installed at startup, values fixed when a resource is
//! created) take effect after the next start.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use arc_swap::ArcSwap;

use crate::patches::Apply;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Game patches (`patch.<id> = 0|1`, `patches.rs`): on/off per registered patch.
    pub patches: BTreeMap<String, bool>,
    /// Profiler hooks (engine phases, D3D11 command lists, GPU timestamps). Off: none of
    /// the profiler's hooks are installed.
    pub profile: bool,
    /// GPU timestamp queries around each command list while recording.
    pub profile_gpu: bool,
    /// GPU timestamps at pass-class changes (`passes.rs`) instead of around each
    /// command list (`profile_gpu_mode = passes | lists`).
    pub profile_gpu_passes: bool,
    /// Command-list labels from FinishCommandList call stacks while recording.
    pub profile_labels: bool,
    /// Benchmark sequence (Ctrl+F3): warm-up, then per repeat a stationary and a pan
    /// segment (seconds).
    pub bench_warmup_s: f32,
    pub bench_static_s: f32,
    pub bench_pan_s: f32,
    pub bench_repeats: u32,
    /// Disable the engine frame limiter (`com_disableFrameSpinning 1`) during the
    /// benchmark; restored afterwards.
    pub bench_uncap: bool,
    /// Free-text label stored with every run (e.g. "native", "d2enhanced-fsr").
    pub bench_label: String,
    /// Engine cvar overrides (`cvar.<name> = <number>`), enforced from the first frame on
    /// (`cvars::enforce_overrides`).
    pub cvars: Vec<(String, f32)>,
    /// FSR 4 in place of the engine TAA. Off at startup: no call sites are patched; turned
    /// on later: installed at run time; turned off: deactivated (hooks stay, inert).
    pub fsr: bool,
    /// RCAS sharpening strength 0..1 (0 = off).
    pub fsr_sharpness: f32,
    /// Let FSR compute exposure instead of using the engine's auto-exposure buffer.
    pub fsr_auto_exposure: bool,
    /// `fsr_mode = upscale`: render below display size (`fsr_scale`) and let FSR upscale;
    /// the post chain runs at display size. `native` (default): FSR anti-aliases only.
    pub fsr_upscale: bool,
    /// Render scale per axis for the upscale mode (0.33..1; snapped to 8-pixel sizes).
    pub fsr_scale: f32,
    /// Mip bias of material samplers in upscale mode; None = auto, log2(scale) - 1.
    pub fsr_mip_bias: Option<f32>,
    /// Reactive + transparency/composition masks from the engine's OIT targets.
    pub fsr_masks: bool,
    /// Reactive mask = transparency coverage x this (0..1).
    pub fsr_reactive: f32,
    /// Upscale mode: resample render-scaled depth targets to display size after FSR
    /// (`fx/depthFull` for post effects, the depth buffer before the post-tonemap surface
    /// draws such as in-world UI), with stencil where the GPU supports `SV_StencilRef`.
    pub fsr_depth_resample: bool,
    /// Motion vectors for sky pixels (the engine leaves them 0) from the camera matrices.
    pub fsr_sky_motion: bool,
    /// Jitter sequence length; None = AMD's default (`ceil(8 * scale^-2)`: 8 at native).
    pub fsr_jitter_phases: Option<u32>,
    /// Flush the immediate context after signalling FSR's inputs (D3D12 starts at once).
    /// 0 = let the driver submit the batch on its own (saves the flush's CPU time on the
    /// frame's critical path; FSR then starts when the batch is submitted, at the latest
    /// at Present). Experiment, measured with the benchmark.
    pub fsr_flush: bool,
    /// `ao_mode = void` (default): our AO replaces NVIDIA HBAO+ (`ao.rs`; the RenderAO hook
    /// is installed when turned on); `hbao`: the game's HBAO+.
    pub ao_void: bool,
    /// `ao_quality` preset index into `ao::QUALITIES` (low, medium, high, ultra).
    pub ao_quality: u8,
    /// Compute the AO at half resolution (depth-aware upsampling; default: below HBAO+'s
    /// GPU cost at native resolution, docs/lighting-options.md B1).
    pub ao_half_res: bool,
    /// Multiplies the game's AO radius (0.7 m).
    pub ao_radius_scale: f32,
    /// Multiplies the occlusion (0 = no AO).
    pub ao_strength: f32,
    /// Show our AO buffer full-screen (drawn before the present; needs `overlay = 1`).
    pub ao_debug: bool,
    /// Sun contact shadows (`contact.rs`, default on since 2026-10-18): screen-space march
    /// toward the sun, combined as `min` with the engine's CSM visibility. Installs the hook.
    pub contact_shadows: bool,
    /// Ray length (meters).
    pub contact_length: f32,
    /// Assumed thickness of depth-buffer surfaces (meters).
    pub contact_thickness: f32,
    /// `contact_quality` preset index into `contact::QUALITIES` (ray steps).
    pub contact_quality: u8,
    /// Show the contact visibility full-screen (needs `overlay = 1`).
    pub contact_debug: bool,
    /// Multiplier of the engine's DOF-combine sharpening (`env/dof/settings/sharpen_coef`,
    /// 0.75 in the analysed eotw capture): an unsharp mask against a half-resolution copy
    /// that draws a bright 1-2 px outline around every silhouette (`dof_sharpen.rs`, E082).
    /// 1 = engine value (no hook), 0 = no sharpening.
    pub dof_sharpen: f32,
    /// Screen-space reflections at the TAA site (`ssr.rs`, docs/reflections.md). Turning it
    /// on installs the TAA call-site redirect (shared with FSR).
    pub ssr: bool,
    /// `ssr_quality` preset index into `ssr::QUALITIES` (march iterations).
    pub ssr_quality: u8,
    /// Trace at half resolution (resolve and blend at full resolution).
    pub ssr_half_res: bool,
    /// Multiplier of the Fresnel blend weight.
    pub ssr_strength: f32,
    /// Assumed perceptual roughness of every reflecting surface (no material data, E067).
    pub ssr_roughness: f32,
    /// Assumed reflectance at normal incidence (F0).
    pub ssr_reflectance: f32,
    /// Pixels without material data (depth-normal fallback) reflect only when they face up
    /// (floors, streets, tables); material pixels reflect on any orientation.
    pub ssr_depth_floors_only: bool,
    /// Assumed thickness of depth-buffer surfaces for hits (meters).
    pub ssr_thickness: f32,
    /// No reflections where the water depth target shows water (water has its own SSR).
    pub ssr_skip_water: bool,
    /// Debug view index into `ssr::DEBUG_VIEWS` (0 = off).
    pub ssr_debug: u8,
    /// Material source (`ssr_material = auto | depth`): auto patches the lit forward pixel
    /// shaders at startup (with `ssr = 1`) and uses their per-pixel normal/roughness/F0 where
    /// present; depth = depth normals + the user surface model only.
    pub ssr_material: bool,
    /// Temporal accumulation of the reflections (reprojected history, per-frame rotated
    /// samples) before the composite.
    pub ssr_temporal: bool,
    /// Ray-tracing scene service (`rt/`, docs/raytracing.md section 11): a D3D12 device on
    /// the game's adapter, the map's BVH and a per-frame camera cut-off TLAS for live
    /// ray tracing. Off: no device, thread or hook for it.
    pub rt: bool,
    /// `rt_build = load` (default): with `rt = 1` the map's BVH is built on a worker at
    /// map load; `use`: on the first use (live RT or a photo).
    pub rt_build_at_load: bool,
    /// Instances whose bounds lie within this distance of the camera (metres) go into the
    /// per-frame TLAS.
    pub rt_radius: f32,
    /// Dynamic objects in the ray-tracing scene (docs/raytracing.md section 14): moved,
    /// hidden and spawned static models follow the engine (per-frame dirty flags). A change
    /// rebuilds the scene (it then has a reserve for dynamic meshes).
    pub rt_dynamic: bool,
    /// Skinned characters within `rt_radius` in the ray-tracing scene (GPU skinning of the
    /// engine's joint palette, refitted BLAS). A change rebuilds the scene like `rt_dynamic`.
    pub rt_characters: bool,
    /// Characters (nearest first) whose pose is updated per frame; up to twice as many are
    /// kept in the scene with their last pose.
    pub rt_characters_n: u32,
    /// GPU milliseconds per frame for character skinning + BLAS updates.
    pub rt_dynamic_budget_ms: f32,
    /// Hybrid ray-traced reflections (`ssr/rt_reflect.rs`, docs/reflections.md section 4):
    /// rays where the screen-space reflections left something unresolved; needs `ssr = 1`,
    /// `rt = 1` and FSR.
    pub rt_reflections: bool,
    /// Perceptual roughness above which no rays are traced (faded from 80 % of it).
    pub rt_max_roughness: f32,
    /// Rays per frame, percent of the half-resolution grid (an even subset beyond it).
    pub rt_ray_budget: u32,
    /// Debug view index into `ssr::rt_reflect::DEBUG_VIEWS` (FSR's input replaced).
    pub rt_reflect_debug: u8,
    /// Path-traced photo mode (`photo/`, docs/raytracing.md section 11). Off: no hook, no
    /// patch, no thread.
    pub photo: bool,
    /// Virtual-key code that enters/leaves photo mode (`photo_key = F4`; F5/F9 are the
    /// game's quicksave/quickload).
    pub photo_key: u32,
    /// Samples per pixel after which accumulation stops.
    pub photo_spp: u32,
    /// Bounces after the primary hit.
    pub photo_bounces: u32,
    /// Multiplier of the photo renderer's validated sky radiance scale (0.067, section 10).
    pub photo_sky_scale: f32,
    /// GPU time per game frame given to the path tracer (ms).
    pub photo_budget_ms: f32,
    /// Largest material texture edge uploaded for the photo scene (texels).
    pub photo_texture_size: u32,
    /// `photo_display = hybrid` (default): the game keeps rendering its paused frame and only
    /// its sun visibility, AO and reflections are replaced by ray-traced terms (`photo/
    /// hybrid.rs`); `pathtrace` (debug): our path-traced image is written into the engine's
    /// HDR buffer before its fog pass; `overlay`: our path-traced image tone-mapped over the
    /// frame at present (needs `overlay = 1`).
    pub photo_display: crate::photo::Display,
    /// Hybrid terms on/off: ray-traced sun shadows (`.y`), AO (`.x`), reflections.
    pub photo_sun: bool,
    pub photo_ao: bool,
    pub photo_reflections: bool,
    /// Rays per pixel and pass of the sun / AO terms (1..=4; the passes per frame follow
    /// `photo_budget_ms`); reflection rays per half-resolution texel and frame
    /// (`rt_reflect::set_photo`).
    pub photo_sun_rays: u32,
    pub photo_ao_rays: u32,
    pub photo_reflect_rays: u32,
    /// Rays per pixel the sun and AO terms accumulate.
    pub photo_terms_spp: u32,
    /// Sun disc angular radius (degrees) of the shadow rays.
    pub photo_sun_radius: f32,
    /// AO ray length (metres).
    pub photo_ao_radius: f32,
    /// `photo_ao_mode = replace`: `.x` = RT AO; `min` (default): min(HBAO+/our AO, RT AO).
    pub photo_ao_replace: bool,
    /// A save also writes the hybrid term masks (debug).
    pub photo_save_masks: bool,
    /// A save also writes `<name>_instances.png`: our primary rays coloured by instance type
    /// (static grey, moved prop yellow, spawned green, character red; docs/raytracing.md 14).
    pub photo_save_instances: bool,
    /// Engine stored-HDR units per photo unit; None = auto (fitted per frame over pixels
    /// whose depth matches the game's).
    pub photo_units: Option<f32>,
    /// Folder for saved photos, relative to the game directory unless absolute.
    pub photo_dir: String,
    /// Read-only probe of the pause / logic-state / camera values (logged on change and
    /// shown in the Photo panel), also with `photo = 0`.
    pub photo_probe: bool,
    /// Texture/resource mods (docs/texture-replacement.md): hook `OpenFileRead` and
    /// serve validated files from `mods_dir`. Off: no hook is installed.
    pub texture_mods: bool,
    /// Mods directory, relative to the game directory unless absolute.
    pub mods_dir: String,
    /// In-game overlay (`overlay.rs`). Off: none of its hooks are installed.
    pub overlay: bool,
    /// Virtual-key code that shows/hides the overlay (`overlay_key = F2`).
    pub overlay_key: u32,
    /// UI scale; None = auto (window height / 1080).
    pub overlay_scale: Option<f32>,
    /// Toasts for events (`notify.rs`), also while the overlay menu is closed.
    pub notifications: bool,
    /// Diagnostics with a runtime cost: FSR value check and motion-vector statistics, the
    /// FSR timing report and the per-second frame/view log line. Off by default.
    pub diagnostics: bool,
    /// Performance HUD (FPS, frame times, graph), also while the menu is closed.
    pub hud: bool,
    /// Virtual-key code that shows/hides the HUD (`hud_key = F7`).
    pub hud_key: u32,
    pub hud_position: Corner,
    /// Draw the HUD and toasts while a benchmark segment records (default off, so the
    /// overlay does not perturb the measurement).
    pub bench_show_overlay: bool,
}

/// Screen corner of the performance HUD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    pub const ALL: [Corner; 4] = [
        Corner::TopLeft,
        Corner::TopRight,
        Corner::BottomLeft,
        Corner::BottomRight,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Corner::TopLeft => "top-left",
            Corner::TopRight => "top-right",
            Corner::BottomLeft => "bottom-left",
            Corner::BottomRight => "bottom-right",
        }
    }

    pub fn parse(v: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == v)
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            patches: crate::patches::PATCHES
                .iter()
                .map(|p| (p.id.to_owned(), p.default_on))
                .collect(),
            profile: false,
            profile_gpu: true,
            profile_gpu_passes: true,
            profile_labels: true,
            bench_warmup_s: 10.0,
            bench_static_s: 20.0,
            bench_pan_s: 20.0,
            bench_repeats: 3,
            bench_uncap: true,
            bench_label: String::new(),
            cvars: Vec::new(),
            fsr: false,
            fsr_sharpness: 0.0,
            fsr_auto_exposure: false,
            fsr_upscale: false,
            fsr_scale: 0.67,
            fsr_mip_bias: None,
            fsr_masks: true,
            fsr_reactive: 0.9,
            fsr_depth_resample: true,
            fsr_sky_motion: true,
            fsr_jitter_phases: None,
            fsr_flush: true,
            ao_void: true,
            ao_quality: crate::ao::DEFAULT_QUALITY,
            ao_half_res: true,
            ao_radius_scale: 1.0,
            ao_strength: 1.0,
            ao_debug: false,
            contact_shadows: true,
            contact_length: 0.5,
            contact_thickness: 0.25,
            contact_quality: crate::contact::DEFAULT_QUALITY,
            contact_debug: false,
            dof_sharpen: 1.0,
            ssr: false,
            ssr_quality: crate::ssr::DEFAULT_QUALITY,
            ssr_half_res: true,
            ssr_strength: 1.0,
            ssr_roughness: 0.15,
            ssr_reflectance: 0.04,
            ssr_depth_floors_only: true,
            ssr_thickness: 0.5,
            ssr_skip_water: true,
            ssr_debug: 0,
            ssr_material: true,
            ssr_temporal: true,
            rt: false,
            rt_build_at_load: true,
            rt_radius: 40.0,
            rt_dynamic: false,
            rt_characters: false,
            rt_characters_n: 12,
            rt_dynamic_budget_ms: 2.0,
            rt_reflections: false,
            rt_max_roughness: 0.5,
            rt_ray_budget: 100,
            rt_reflect_debug: 0,
            photo: false,
            photo_key: 0x73,
            photo_spp: 256,
            photo_bounces: 3,
            photo_sky_scale: 1.0,
            photo_budget_ms: 8.0,
            photo_texture_size: 512,
            photo_display: crate::photo::Display::Hybrid,
            photo_sun: true,
            photo_ao: true,
            photo_reflections: true,
            photo_sun_rays: 4,
            photo_ao_rays: 4,
            photo_reflect_rays: 4,
            photo_terms_spp: 256,
            photo_sun_radius: 0.27,
            photo_ao_radius: 1.5,
            photo_ao_replace: false,
            photo_save_masks: false,
            photo_save_instances: false,
            photo_units: None,
            photo_dir: "void-rs/local/photos".to_owned(),
            photo_probe: false,
            texture_mods: false,
            mods_dir: "void-rs/mods".to_owned(),
            overlay: true,
            overlay_key: 0x71,
            overlay_scale: None,
            notifications: true,
            diagnostics: false,
            hud: false,
            hud_key: 0x76,
            hud_position: Corner::TopLeft,
            bench_show_overlay: false,
        }
    }
}

/// Canonical file layout: comments, key order and default values; the `patch.*` line
/// expands to every registered patch (see `render`).
const TEMPLATE: &str = "\
# void-inject settings (read at game start; the overlay edits and saves them).
# Game patches (patches.rs): patch.<id> = 1 applies the patch, 0 leaves the code as is.
patch.*
# Profiler: engine phase timings, D3D11 command lists, GPU timestamps.
# 0 = no profiler hooks at all. Ctrl+F3 starts/aborts a benchmark run when 1.
profile = 0
profile_gpu = 1
# passes = timestamps at pass-class changes (shadows, ocean, opaque, post, ...);
# lists = timestamps around each command list.
profile_gpu_mode = passes
profile_labels = 1
# Benchmark sequence: warm-up, then per repeat a stationary and a 360 degree pan segment.
bench_warmup_s = 10
bench_static_s = 20
bench_pan_s = 20
bench_repeats = 3
# 1 = engine frame limiter off during the run (restored afterwards).
bench_uncap = 1
# Free-text label stored with each run.
bench_label =
# Engine cvar overrides: cvar.<name> = <number>, written from the main menu on and
# re-applied if the game resets them. Anisotropic filtering: the Ultra preset leaves
# image_anisotropy at 1 (docs/lighting-options.md A1); remove the line for the original.
cvar.image_anisotropy = 16
# FSR 4 replaces the engine TAA (native anti-aliasing; in-game Anti-aliasing must be TAA).
# F6 toggles FSR / engine TAA, Shift+F6 the FSR debug view. 0 at startup = call sites
# untouched; turning it on in the overlay installs FSR while the game runs, turning it
# off deactivates FSR (hooks stay installed, inert). Every fsr_* key applies live.
fsr = 0
# Sharpening 0..1 (0 = off).
fsr_sharpness = 0
# 1 = FSR computes exposure itself instead of reading the engine's auto exposure.
fsr_auto_exposure = 0
# native = FSR anti-aliases at full size; upscale = render at fsr_scale (snapped to
# multiples of 8 pixels) and upscale, post-processing at display size.
fsr_mode = native
fsr_scale = 0.67
# Mip bias of material samplers in upscale mode: auto = log2(scale) - 1, or a number.
fsr_mip_bias = auto
# Reactive / transparency masks from the engine's transparency (OIT) buffers.
fsr_masks = 1
fsr_reactive = 0.9
# Upscale: resample render-scaled depth (post effects, in-world UI) to display size.
fsr_depth_resample = 1
# Motion vectors for the sky from the camera matrices (the engine writes none).
fsr_sky_motion = 1
# Jitter phases: auto = AMD's ceil(8 * (display/render)^2), i.e. 8 at native; or 4..128.
fsr_jitter_phases = auto
# 1 = flush the D3D11 batch so FSR starts at once; 0 = no flush (less CPU on the submit
# path, FSR starts when the driver submits). Experiment: compare with the benchmark.
fsr_flush = 1
# Ambient occlusion (docs/lighting-options.md B1): hbao = the game's NVIDIA HBAO+; void =
# our visibility-bitmask AO written into HBAO+'s target instead (HBAO+ does not run).
# Every ao_* key applies live; void installs the hook while the game runs. Default void
# since 2026-10-18 (STATUS E064: 0.08 ms instead of HBAO+'s ~0.22 ms).
ao_mode = void
# low | medium | high | ultra (directions x steps per pixel).
ao_quality = high
# 1 = compute at half resolution with depth-aware upsampling (default; 0 = full
# resolution, sharper contact detail, about 0.4 ms more GPU at 3440x1440 native).
ao_half_res = 1
# Multipliers of the game's AO radius (0.7 m) and of the occlusion.
ao_radius_scale = 1
ao_strength = 1
# 1 = show our AO buffer full-screen (drawn before the present; needs overlay = 1).
ao_debug = 0
# Sun contact shadows (docs/lighting-options.md B2): a short screen-space ray toward the
# sun from every pixel, combined with the engine's sun shadow map as min(). All live.
# Default on since 2026-10-18 (user check in game).
contact_shadows = 1
# Ray length and the thickness assumed for depth-buffer surfaces, in meters.
contact_length = 0.5
contact_thickness = 0.25
# low | medium | high | ultra (8 / 12 / 16 / 24 ray steps).
contact_quality = high
# 1 = show the contact visibility full-screen (needs overlay = 1).
contact_debug = 0
# Multiplier of the engine's depth-of-field sharpening (docs/lighting-options.md, Edge
# halo): the game sharpens the whole frame against a half-resolution copy, which draws a
# bright 1-2 px outline around characters, fences and other silhouettes. 1 = game (no
# hook), 0 = off. Live.
dof_sharpen = 1
# Screen-space reflections (docs/reflections.md): traced in the frame before TAA/FSR and
# blended by Fresnel. The game has no per-pixel material data, so every reflecting pixel
# uses the roughness and reflectance below. All live.
ssr = 0
# low | medium | high | ultra (24 / 40 / 64 / 96 hierarchical march steps).
ssr_quality = medium
# 1 = trace at half resolution (default), 0 = full resolution.
ssr_half_res = 1
# Blend weight multiplier, assumed roughness (0 = mirror) and reflectance at normal
# incidence (0.04 = typical non-metal).
ssr_strength = 1
ssr_roughness = 0.15
ssr_reflectance = 0.04
# 1 = pixels without material data (depth-normal fallback) reflect only on upward-facing
# surfaces (floors, streets); material pixels reflect on any orientation.
ssr_depth_floors_only = 1
# Thickness assumed for depth-buffer surfaces when a ray passes behind them (meters).
ssr_thickness = 0.5
# 1 = no reflections on water (it has its own).
ssr_skip_water = 1
# off | hits | confidence | reflection | water | material | history | class | f0
# (full-screen view; needs overlay = 1).
ssr_debug = off
# auto = per-pixel normal, roughness and reflectance from the game's lit shaders (patched
# when the game starts with ssr = 1); depth = depth normals and the values above only.
ssr_material = auto
# 1 = accumulate the reflections over frames (motion-vector reprojection, history clamp,
# per-frame rotated samples, roughness-scaled blur; also denoises the ray-traced
# reflections); 0 = this frame only.
ssr_temporal = 1
# Ray-tracing scene (docs/raytracing.md section 11): a D3D12 device on the game's GPU,
# the map's static geometry as a BVH and a TLAS of the instances near the camera, for
# live ray tracing. 0 = no device, thread or hook. All rt* keys apply live.
rt = 0
# load = build the map's BVH on a worker thread at map load; use = on first use.
rt_build = load
# Instances within this distance of the camera (metres) go into the per-frame TLAS.
rt_radius = 40
# Dynamic objects (docs/raytracing.md section 14): rt_dynamic = 1 lets moved, hidden and
# spawned static models follow the engine; rt_characters = 1 adds the skinned characters
# within rt_radius (GPU-skinned from the engine's joint palette), posing the
# rt_characters_n nearest per frame within rt_dynamic_budget_ms of GPU time. Turning either
# on or off rebuilds the scene.
rt_dynamic = 0
rt_characters = 0
rt_characters_n = 12
rt_dynamic_budget_ms = 2
# Hybrid ray-traced reflections (needs ssr = 1, rt = 1 and FSR): rays where the screen-space
# reflections left something unresolved, up to rt_max_roughness (perceptual), at most
# rt_ray_budget percent of the half-resolution grid per frame. Debug view rt_reflect_debug
# = off | rays | hits | rt | history | instances (replaces FSR's input; instances = hits
# coloured by instance type: grey static, yellow moved, green spawned, red character).
rt_reflections = 0
rt_max_roughness = 0.5
rt_ray_budget = 100
rt_reflect_debug = off
# Photo mode: photo_key pauses the game (the engine's own Ansel photo session: player,
# hands and HUD hidden), gives a free camera (WASD + mouse, Q/E down/up, Shift fast, Ctrl
# slow, R back to the start, Enter saves, Esc or photo_key leaves). 0 = no hook, no patch,
# no thread. All photo* keys are live.
photo = 0
photo_key = F4
# hybrid = the game renders its paused frame and only its sun shadows, ambient occlusion
# and reflections become ray-traced terms (engine fog, water, particles, characters, post
# and tone mapping stay); pathtrace = our path-traced image replaces the static scene's
# lighting before the fog pass (debug); overlay = our path-traced image tone-mapped over
# the frame (needs overlay = 1).
photo_display = hybrid
# GPU milliseconds per game frame (both displays), largest texture edge (texels).
photo_budget_ms = 8
photo_texture_size = 512
# Hybrid terms: on/off per term; rays per pixel and pass for sun shadows / AO (1-4; the
# passes per frame follow photo_budget_ms) and per reflection texel and frame (reflections
# need FSR and rt_reflections = 1); rays per pixel accumulated for sun and AO; sun disc
# radius (degrees); AO ray length (metres); AO mode: min (with the engine's AO) or replace.
# Shadows are always min(engine CSM, ray traced).
photo_sun = 1
photo_ao = 1
photo_reflections = 1
photo_sun_rays = 4
photo_ao_rays = 4
photo_reflect_rays = 4
photo_terms_spp = 256
photo_sun_radius = 0.27
photo_ao_radius = 1.5
photo_ao_mode = min
# 1 = a save also writes the hybrid term masks (<name>_sun.png, <name>_ao.png; debug).
photo_save_masks = 0
# 1 = a save also writes <name>_instances.png: the ray-traced scene seen from the photo
# camera, coloured by instance type (static grey, moved prop yellow, spawned green,
# character red, miss dark blue; docs/raytracing.md 14).
photo_save_instances = 0
# Path tracer (pathtrace / overlay): samples per pixel, bounces after the first hit,
# multiplier of the validated sky radiance.
photo_spp = 256
photo_bounces = 3
photo_sky_scale = 1
# Engine HDR units per photo unit: auto (fitted per frame) or a number.
photo_units = auto
# Saved photos (PNG + raw HDR), relative to the game folder unless absolute.
photo_dir = void-rs/local/photos
# 1 = log the pause/state/camera values the photo mode relies on (on change) also with
# photo = 0 (read-only).
photo_probe = 0
# Texture/resource mods (docs/texture-replacement.md): textures/<image>.dds and
# override/<resource> under mods_dir (relative to the game folder). 0 = no hook at all.
texture_mods = 0
mods_dir = void-rs/mods
# In-game overlay: status, FSR, cvars, texture mods, profiler, log, settings (saved here).
# 0 = no overlay hooks at all. overlay_key: F1-F24, Insert, Home, End, PageUp, PageDown,
# Delete, Pause, ScrollLock, Tilde, or a virtual-key number.
overlay = 1
overlay_key = F2
# UI scale: auto (window height / 1080) or a factor 0.5..4.
overlay_scale = auto
# Toasts for events (benchmark, FSR, patches, texture mods, settings), also with the
# overlay closed.
notifications = 1
# Diagnostics that cost frame time (for bug reports): FSR NaN/Inf value check and
# motion-vector statistics (GPU passes every frame), the FSR timing report and a frame/view
# summary in the log every second. Errors and state changes are logged either way. Live.
diagnostics = 0
# Performance HUD (FPS, frame times, small graph), shown with the overlay closed too;
# hud_key toggles it (same key names as overlay_key); hud_position: top-left, top-right,
# bottom-left, bottom-right.
hud = 0
hud_key = F7
hud_position = top-left
# 1 = HUD and toasts stay visible while a benchmark segment records (0: hidden, so they
# do not perturb the measurement).
bench_show_overlay = 0
";

/// A settings key: whether a change applies while the game runs, and why (shown by the
/// overlay). Decided per key from where it is consumed.
pub struct Key {
    pub name: &'static str,
    pub live: bool,
    pub why: &'static str,
}

const fn key(name: &'static str, live: bool, why: &'static str) -> Key {
    Key { name, live, why }
}

/// Every key of the file, in file order (`cvar.<name>` lines stand for the override list).
pub const KEYS: &[Key] = &[
    key("patch.", false, "per patch: see the patch's apply time"),
    key("profile", false, "profiler hooks are installed at startup"),
    key("profile_gpu", true, "read per command list while recording"),
    key(
        "profile_gpu_mode",
        true,
        "read per command list while recording",
    ),
    key(
        "profile_labels",
        true,
        "read per command list while recording",
    ),
    key("bench_warmup_s", true, "read when a run starts"),
    key("bench_static_s", true, "read when a run starts"),
    key("bench_pan_s", true, "read when a run starts"),
    key("bench_repeats", true, "read when a run starts"),
    key("bench_uncap", true, "read when a run starts"),
    key("bench_label", true, "read when a run starts"),
    key("cvar.", true, "override list, enforced every 30 frames"),
    key(
        "fsr",
        true,
        "on: installs FSR now if needed (call sites patched with threads suspended); off: FSR deactivated like F6, hooks stay installed but inert",
    ),
    key("fsr_sharpness", true, "read per FSR dispatch"),
    key(
        "fsr_auto_exposure",
        true,
        "a change re-creates the FFX context (engine TAA for a few frames)",
    ),
    key(
        "fsr_mode",
        true,
        "render scale written per frame; post-chain hooks are always installed with FSR (history reset on change)",
    ),
    key(
        "fsr_scale",
        true,
        "render scale written per frame (history reset on change; auto mip bias follows)",
    ),
    key(
        "fsr_mip_bias",
        true,
        "the engine's samplers are re-created with the new bias and swapped in (upscale only)",
    ),
    key(
        "fsr_masks",
        true,
        "read per frame (OIT target hooks are always installed with FSR)",
    ),
    key("fsr_reactive", true, "read per FSR dispatch"),
    key("fsr_depth_resample", true, "read per frame (upscale mode)"),
    key("fsr_sky_motion", true, "read per FSR dispatch"),
    key(
        "fsr_jitter_phases",
        true,
        "sequence rebuilt when the value changes",
    ),
    key("fsr_flush", true, "read per FSR dispatch"),
    key(
        "ao_mode",
        true,
        "read per RenderAO call; void installs the HBAO+ hook now if needed, hbao leaves it installed but passing through",
    ),
    key("ao_quality", true, "read per RenderAO call"),
    key(
        "ao_half_res",
        true,
        "read per RenderAO call (AO targets re-created)",
    ),
    key("ao_radius_scale", true, "read per RenderAO call"),
    key("ao_strength", true, "read per RenderAO call"),
    key("ao_debug", true, "checked every frame (needs overlay = 1)"),
    key(
        "contact_shadows",
        true,
        "read per SCREENSHADOWS pass; on installs the hook now if needed",
    ),
    key("contact_length", true, "read per SCREENSHADOWS pass"),
    key("contact_thickness", true, "read per SCREENSHADOWS pass"),
    key("contact_quality", true, "read per SCREENSHADOWS pass"),
    key(
        "contact_debug",
        true,
        "checked every frame (needs overlay = 1)",
    ),
    key(
        "dof_sharpen",
        true,
        "applied every frame; a value other than 1 installs the DOF settings hook now",
    ),
    key(
        "ssr",
        true,
        "read per TAA frame; on installs the TAA call-site redirect now if needed",
    ),
    key("ssr_quality", true, "read per TAA frame"),
    key("ssr_half_res", true, "read per TAA frame"),
    key("ssr_strength", true, "read per TAA frame"),
    key("ssr_roughness", true, "read per TAA frame"),
    key("ssr_reflectance", true, "read per TAA frame"),
    key("ssr_depth_floors_only", true, "read per TAA frame"),
    key("ssr_thickness", true, "read per TAA frame"),
    key("ssr_skip_water", true, "read per TAA frame"),
    key("ssr_debug", true, "checked every frame (needs overlay = 1)"),
    key(
        "ssr_material",
        true,
        "used per TAA frame; the shaders are patched only when the game starts with ssr = 1",
    ),
    key("ssr_temporal", true, "read per TAA frame"),
    key(
        "rt",
        true,
        "on: creates the D3D12 device and the scene worker now; off: drops the scene",
    ),
    key("rt_build", true, "read at each map load"),
    key("rt_radius", true, "read per TLAS rebuild"),
    key(
        "rt_dynamic",
        true,
        "a change rebuilds the scene (reserve for dynamic meshes)",
    ),
    key(
        "rt_characters",
        true,
        "a change rebuilds the scene (reserve for dynamic meshes)",
    ),
    key("rt_characters_n", true, "read per frame"),
    key("rt_dynamic_budget_ms", true, "read per frame"),
    key("rt_reflections", true, "read per TAA frame (FSR path)"),
    key("rt_max_roughness", true, "read per TAA frame"),
    key("rt_ray_budget", true, "read per frame"),
    key("rt_reflect_debug", true, "read per frame"),
    key(
        "photo",
        true,
        "on: installs the fog-pass hook now if needed; off: leaves photo mode and frees its data",
    ),
    key("photo_key", true, "polled every frame"),
    key("photo_spp", true, "read per accumulation step"),
    key("photo_bounces", true, "a change restarts the accumulation"),
    key(
        "photo_sky_scale",
        true,
        "a change restarts the accumulation",
    ),
    key("photo_budget_ms", true, "read per accumulation step"),
    key("photo_texture_size", true, "used for the next scene build"),
    key(
        "photo_display",
        true,
        "checked every frame; a change restarts the photo's work",
    ),
    key("photo_sun", true, "read per frame"),
    key("photo_ao", true, "read per frame"),
    key("photo_reflections", true, "read when photo mode starts"),
    key("photo_sun_rays", true, "a change restarts the accumulation"),
    key("photo_ao_rays", true, "a change restarts the accumulation"),
    key("photo_reflect_rays", true, "read when photo mode starts"),
    key("photo_terms_spp", true, "read per accumulation step"),
    key(
        "photo_sun_radius",
        true,
        "a change restarts the accumulation",
    ),
    key(
        "photo_ao_radius",
        true,
        "a change restarts the accumulation",
    ),
    key("photo_ao_mode", true, "read per frame"),
    key("photo_save_masks", true, "read per save"),
    key("photo_save_instances", true, "read per save"),
    key("photo_units", true, "read per composited frame"),
    key("photo_dir", true, "read per save"),
    key("photo_probe", true, "checked every frame"),
    key(
        "texture_mods",
        false,
        "OpenFileRead hook installed in DllMain",
    ),
    key("mods_dir", false, "scanned once in DllMain"),
    key("overlay", false, "overlay hooks are installed at startup"),
    key("overlay_key", true, "polled every frame"),
    key("overlay_scale", true, "applied every overlay frame"),
    key("notifications", true, "checked every frame"),
    key("diagnostics", true, "checked every frame"),
    key("hud", true, "checked every frame (hud_key toggles it)"),
    key("hud_key", true, "polled every frame"),
    key("hud_position", true, "applied every HUD frame"),
    key("bench_show_overlay", true, "checked every frame"),
];

/// Looks up a key of [`KEYS`].
pub fn key_info(name: &str) -> Option<&'static Key> {
    let name = if name.starts_with("cvar.") {
        "cvar."
    } else if name.starts_with("patch.") {
        "patch."
    } else {
        name
    };
    KEYS.iter().find(|k| k.name == name)
}

/// Copies the live keys of `src` into `dst`.
fn apply_live(dst: &mut Settings, src: &Settings) {
    dst.profile_gpu = src.profile_gpu;
    dst.profile_gpu_passes = src.profile_gpu_passes;
    dst.profile_labels = src.profile_labels;
    dst.bench_warmup_s = src.bench_warmup_s;
    dst.bench_static_s = src.bench_static_s;
    dst.bench_pan_s = src.bench_pan_s;
    dst.bench_repeats = src.bench_repeats;
    dst.bench_uncap = src.bench_uncap;
    dst.bench_label.clone_from(&src.bench_label);
    dst.cvars.clone_from(&src.cvars);
    dst.fsr = src.fsr;
    dst.fsr_sharpness = src.fsr_sharpness;
    dst.fsr_auto_exposure = src.fsr_auto_exposure;
    dst.fsr_upscale = src.fsr_upscale;
    dst.fsr_scale = src.fsr_scale;
    dst.fsr_mip_bias = src.fsr_mip_bias;
    dst.fsr_masks = src.fsr_masks;
    dst.fsr_reactive = src.fsr_reactive;
    dst.fsr_depth_resample = src.fsr_depth_resample;
    dst.fsr_sky_motion = src.fsr_sky_motion;
    dst.fsr_jitter_phases = src.fsr_jitter_phases;
    dst.fsr_flush = src.fsr_flush;
    dst.ao_void = src.ao_void;
    dst.ao_quality = src.ao_quality;
    dst.ao_half_res = src.ao_half_res;
    dst.ao_radius_scale = src.ao_radius_scale;
    dst.ao_strength = src.ao_strength;
    dst.ao_debug = src.ao_debug;
    dst.contact_shadows = src.contact_shadows;
    dst.contact_length = src.contact_length;
    dst.contact_thickness = src.contact_thickness;
    dst.contact_quality = src.contact_quality;
    dst.contact_debug = src.contact_debug;
    dst.dof_sharpen = src.dof_sharpen;
    dst.ssr = src.ssr;
    dst.ssr_quality = src.ssr_quality;
    dst.ssr_half_res = src.ssr_half_res;
    dst.ssr_strength = src.ssr_strength;
    dst.ssr_roughness = src.ssr_roughness;
    dst.ssr_reflectance = src.ssr_reflectance;
    dst.ssr_depth_floors_only = src.ssr_depth_floors_only;
    dst.ssr_thickness = src.ssr_thickness;
    dst.ssr_skip_water = src.ssr_skip_water;
    dst.ssr_debug = src.ssr_debug;
    dst.ssr_material = src.ssr_material;
    dst.ssr_temporal = src.ssr_temporal;
    dst.rt = src.rt;
    dst.rt_build_at_load = src.rt_build_at_load;
    dst.rt_radius = src.rt_radius;
    dst.rt_dynamic = src.rt_dynamic;
    dst.rt_characters = src.rt_characters;
    dst.rt_characters_n = src.rt_characters_n;
    dst.rt_dynamic_budget_ms = src.rt_dynamic_budget_ms;
    dst.rt_reflections = src.rt_reflections;
    dst.rt_max_roughness = src.rt_max_roughness;
    dst.rt_ray_budget = src.rt_ray_budget;
    dst.rt_reflect_debug = src.rt_reflect_debug;
    dst.photo = src.photo;
    dst.photo_key = src.photo_key;
    dst.photo_spp = src.photo_spp;
    dst.photo_bounces = src.photo_bounces;
    dst.photo_sky_scale = src.photo_sky_scale;
    dst.photo_budget_ms = src.photo_budget_ms;
    dst.photo_texture_size = src.photo_texture_size;
    dst.photo_display = src.photo_display;
    dst.photo_sun = src.photo_sun;
    dst.photo_ao = src.photo_ao;
    dst.photo_reflections = src.photo_reflections;
    dst.photo_sun_rays = src.photo_sun_rays;
    dst.photo_ao_rays = src.photo_ao_rays;
    dst.photo_reflect_rays = src.photo_reflect_rays;
    dst.photo_terms_spp = src.photo_terms_spp;
    dst.photo_sun_radius = src.photo_sun_radius;
    dst.photo_ao_radius = src.photo_ao_radius;
    dst.photo_ao_replace = src.photo_ao_replace;
    dst.photo_save_masks = src.photo_save_masks;
    dst.photo_save_instances = src.photo_save_instances;
    dst.photo_units = src.photo_units;
    dst.photo_dir.clone_from(&src.photo_dir);
    dst.photo_probe = src.photo_probe;
    dst.overlay_key = src.overlay_key;
    dst.overlay_scale = src.overlay_scale;
    dst.notifications = src.notifications;
    dst.diagnostics = src.diagnostics;
    dst.hud = src.hud;
    dst.hud_key = src.hud_key;
    dst.hud_position = src.hud_position;
    dst.bench_show_overlay = src.bench_show_overlay;
    for p in crate::patches::PATCHES
        .iter()
        .filter(|p| p.when == Apply::Live)
    {
        if let Some(&on) = src.patches.get(p.id) {
            dst.patches.insert(p.id.to_owned(), on);
        }
    }
}

/// Active settings (what the hooks use).
static ACTIVE: LazyLock<ArcSwap<Settings>> =
    LazyLock::new(|| ArcSwap::from_pointee(Settings::default()));
/// Settings as edited (written by `save`).
static FILE: LazyLock<Mutex<Settings>> = LazyLock::new(|| Mutex::new(Settings::default()));
/// The settings file, once `load` ran.
static PATH: OnceLock<PathBuf> = OnceLock::new();

/// The active settings (cheap: one reference-count increment).
pub fn get() -> Arc<Settings> {
    ACTIVE.load_full()
}

/// The settings as edited (restart-only changes included).
pub fn file() -> Settings {
    FILE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Edits the file settings and applies the live keys to the active settings at once.
pub fn edit(f: impl FnOnce(&mut Settings)) {
    let mut file = FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut file);
    let mut active = Settings::clone(&ACTIVE.load());
    apply_live(&mut active, &file);
    ACTIVE.store(Arc::new(active));
}

/// Restart-only keys whose edited value differs from the running one.
pub fn pending_restart() -> Vec<String> {
    let (file, active) = (file(), get());
    let mut out: Vec<String> = KEYS
        .iter()
        .filter(|k| !k.live && value(&file, k.name) != value(&active, k.name))
        .map(|k| k.name.to_owned())
        .collect();
    out.extend(
        crate::patches::PATCHES
            .iter()
            .filter(|p| p.when != Apply::Live && file.patches.get(p.id) != active.patches.get(p.id))
            .map(|p| format!("patch.{}", p.id)),
    );
    out
}

/// Key names accepted by `overlay_key` (besides a number).
const KEY_NAMES: [(&str, u32); 10] = [
    ("Insert", 0x2D),
    ("Delete", 0x2E),
    ("Home", 0x24),
    ("End", 0x23),
    ("PageUp", 0x21),
    ("PageDown", 0x22),
    ("Pause", 0x13),
    ("ScrollLock", 0x91),
    ("Tilde", 0xC0),
    ("Apps", 0x5D),
];

/// Parses a key name (F1-F24, `KEY_NAMES`, case-insensitive) or a virtual-key number
/// (decimal or 0x hex).
pub fn parse_key(v: &str) -> Option<u32> {
    let v = v.trim();
    if let Some(n) = v
        .strip_prefix(['F', 'f'])
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|n| (1..=24).contains(n))
    {
        return Some(0x6F + n);
    }
    if let Some(&(_, vk)) = KEY_NAMES.iter().find(|(n, _)| n.eq_ignore_ascii_case(v)) {
        return Some(vk);
    }
    let n = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(h) => u32::from_str_radix(h, 16).ok(),
        None => v.parse().ok(),
    };
    n.filter(|n| (1..=0xFE).contains(n))
}

/// Name of a virtual key for the file (inverse of `parse_key`).
pub fn key_name(vk: u32) -> String {
    if (0x70..=0x87).contains(&vk) {
        return format!("F{}", vk - 0x6F);
    }
    KEY_NAMES
        .iter()
        .find(|(_, k)| *k == vk)
        .map_or_else(|| format!("{vk:#04x}"), |(n, _)| (*n).to_owned())
}

fn flag(b: bool) -> String {
    if b { "1" } else { "0" }.to_owned()
}

/// The file representation of one key (`None` for unknown keys and `cvar.`).
pub fn value(s: &Settings, key: &str) -> Option<String> {
    Some(match key {
        "profile" => flag(s.profile),
        "profile_gpu" => flag(s.profile_gpu),
        "profile_gpu_mode" => if s.profile_gpu_passes {
            "passes"
        } else {
            "lists"
        }
        .to_owned(),
        "profile_labels" => flag(s.profile_labels),
        "bench_warmup_s" => s.bench_warmup_s.to_string(),
        "bench_static_s" => s.bench_static_s.to_string(),
        "bench_pan_s" => s.bench_pan_s.to_string(),
        "bench_repeats" => s.bench_repeats.to_string(),
        "bench_uncap" => flag(s.bench_uncap),
        "bench_label" => s.bench_label.clone(),
        "fsr" => flag(s.fsr),
        "fsr_sharpness" => s.fsr_sharpness.to_string(),
        "fsr_auto_exposure" => flag(s.fsr_auto_exposure),
        "fsr_mode" => if s.fsr_upscale { "upscale" } else { "native" }.to_owned(),
        "fsr_scale" => s.fsr_scale.to_string(),
        "fsr_mip_bias" => s.fsr_mip_bias.map_or("auto".to_owned(), |b| b.to_string()),
        "fsr_masks" => flag(s.fsr_masks),
        "fsr_reactive" => s.fsr_reactive.to_string(),
        "fsr_depth_resample" => flag(s.fsr_depth_resample),
        "fsr_sky_motion" => flag(s.fsr_sky_motion),
        "fsr_jitter_phases" => s
            .fsr_jitter_phases
            .map_or("auto".to_owned(), |n| n.to_string()),
        "fsr_flush" => flag(s.fsr_flush),
        "ao_mode" => if s.ao_void { "void" } else { "hbao" }.to_owned(),
        "ao_quality" => crate::ao::quality_name(s.ao_quality).to_owned(),
        "ao_half_res" => flag(s.ao_half_res),
        "ao_radius_scale" => s.ao_radius_scale.to_string(),
        "ao_strength" => s.ao_strength.to_string(),
        "ao_debug" => flag(s.ao_debug),
        "contact_shadows" => flag(s.contact_shadows),
        "contact_length" => s.contact_length.to_string(),
        "contact_thickness" => s.contact_thickness.to_string(),
        "contact_quality" => crate::contact::quality_name(s.contact_quality).to_owned(),
        "contact_debug" => flag(s.contact_debug),
        "dof_sharpen" => s.dof_sharpen.to_string(),
        "ssr" => flag(s.ssr),
        "ssr_quality" => crate::ssr::quality_name(s.ssr_quality).to_owned(),
        "ssr_half_res" => flag(s.ssr_half_res),
        "ssr_strength" => s.ssr_strength.to_string(),
        "ssr_roughness" => s.ssr_roughness.to_string(),
        "ssr_reflectance" => s.ssr_reflectance.to_string(),
        "ssr_depth_floors_only" => flag(s.ssr_depth_floors_only),
        "ssr_thickness" => s.ssr_thickness.to_string(),
        "ssr_skip_water" => flag(s.ssr_skip_water),
        "ssr_debug" => crate::ssr::debug_name(s.ssr_debug).to_owned(),
        "ssr_material" => if s.ssr_material { "auto" } else { "depth" }.to_owned(),
        "ssr_temporal" => flag(s.ssr_temporal),
        "rt" => flag(s.rt),
        "rt_build" => if s.rt_build_at_load { "load" } else { "use" }.to_owned(),
        "rt_radius" => s.rt_radius.to_string(),
        "rt_dynamic" => flag(s.rt_dynamic),
        "rt_characters" => flag(s.rt_characters),
        "rt_characters_n" => s.rt_characters_n.to_string(),
        "rt_dynamic_budget_ms" => s.rt_dynamic_budget_ms.to_string(),
        "rt_reflections" => flag(s.rt_reflections),
        "rt_max_roughness" => s.rt_max_roughness.to_string(),
        "rt_ray_budget" => s.rt_ray_budget.to_string(),
        "rt_reflect_debug" => crate::ssr::rt_reflect::debug_name(s.rt_reflect_debug).to_owned(),
        "photo" => flag(s.photo),
        "photo_key" => key_name(s.photo_key),
        "photo_spp" => s.photo_spp.to_string(),
        "photo_bounces" => s.photo_bounces.to_string(),
        "photo_sky_scale" => s.photo_sky_scale.to_string(),
        "photo_budget_ms" => s.photo_budget_ms.to_string(),
        "photo_texture_size" => s.photo_texture_size.to_string(),
        "photo_display" => s.photo_display.name().to_owned(),
        "photo_sun" => flag(s.photo_sun),
        "photo_ao" => flag(s.photo_ao),
        "photo_reflections" => flag(s.photo_reflections),
        "photo_sun_rays" => s.photo_sun_rays.to_string(),
        "photo_ao_rays" => s.photo_ao_rays.to_string(),
        "photo_reflect_rays" => s.photo_reflect_rays.to_string(),
        "photo_terms_spp" => s.photo_terms_spp.to_string(),
        "photo_sun_radius" => s.photo_sun_radius.to_string(),
        "photo_ao_radius" => s.photo_ao_radius.to_string(),
        "photo_ao_mode" => if s.photo_ao_replace { "replace" } else { "min" }.to_owned(),
        "photo_save_masks" => flag(s.photo_save_masks),
        "photo_save_instances" => flag(s.photo_save_instances),
        "photo_units" => s.photo_units.map_or("auto".to_owned(), |v| v.to_string()),
        "photo_dir" => s.photo_dir.clone(),
        "photo_probe" => flag(s.photo_probe),
        "texture_mods" => flag(s.texture_mods),
        "mods_dir" => s.mods_dir.clone(),
        "overlay" => flag(s.overlay),
        "overlay_key" => key_name(s.overlay_key),
        "overlay_scale" => s.overlay_scale.map_or("auto".to_owned(), |v| v.to_string()),
        "notifications" => flag(s.notifications),
        "diagnostics" => flag(s.diagnostics),
        "hud" => flag(s.hud),
        "hud_key" => key_name(s.hud_key),
        "hud_position" => s.hud_position.name().to_owned(),
        "bench_show_overlay" => flag(s.bench_show_overlay),
        _ => return None,
    })
}

/// The settings file for `s`: the default file's comments and key order with `s`'s
/// values; the cvar overrides replace the default file's `cvar.` lines.
pub fn render(s: &Settings) -> String {
    let mut out = String::new();
    let mut cvars_done = false;
    for line in TEMPLATE.lines() {
        let t = line.trim();
        let key = t.split_once('=').map(|(k, _)| k.trim());
        match key {
            _ if t.is_empty() || t.starts_with('#') || t.starts_with(';') => {
                out += line;
                out.push('\n');
            }
            None if t == "patch.*" => {
                for p in crate::patches::PATCHES {
                    out += &format!(
                        "# {}: {} ({}; {})\npatch.{} = {}\n",
                        p.name,
                        p.description,
                        p.source,
                        p.when.label(),
                        p.id,
                        flag(crate::patches::enabled(s, p))
                    );
                }
            }
            Some(k) if k.starts_with("cvar.") => {
                if !cvars_done {
                    for (name, v) in &s.cvars {
                        out += &format!("cvar.{name} = {v}\n");
                    }
                    cvars_done = true;
                }
            }
            Some(k) => {
                let v = value(s, k).unwrap_or_default();
                out += format!("{k} = {v}").trim_end();
                out.push('\n');
            }
            None => {}
        }
    }
    out
}

/// Cvar overrides written into a newly created file (the file's comment explains them);
/// a file without `cvar.` lines has none.
const NEW_FILE_CVARS: [(&str, f32); 1] = [("image_anisotropy", 16.0)];

/// Contents of a newly created settings file.
pub fn default_file() -> String {
    render(&Settings {
        cvars: NEW_FILE_CVARS
            .iter()
            .map(|(n, v)| ((*n).to_owned(), *v))
            .collect(),
        ..Settings::default()
    })
}

/// Writes the file settings to the settings file (the previous file is kept as
/// `void-inject.ini.bak`); returns the path.
pub fn save() -> Result<PathBuf, String> {
    let path = PATH.get().ok_or("settings were not loaded from a file")?;
    let text = render(&file());
    if path.exists() {
        std::fs::copy(path, path.with_extension("ini.bak"))
            .map_err(|e| format!("backup of {}: {e}", path.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path.clone())
}

/// Parses `text`; returns the settings and the unknown/invalid lines.
pub fn parse(text: &str) -> (Settings, Vec<String>) {
    let mut s = Settings::default();
    let mut bad = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            bad.push(line.to_owned());
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        if !apply(&mut s, k, v) {
            bad.push(line.to_owned());
        }
    }
    (s, bad)
}

/// Sets key `k` of `s` from its file text `v`; false for unknown keys and invalid values.
fn apply(s: &mut Settings, k: &str, v: &str) -> bool {
    let flag = |v: &str| matches!(v, "1" | "true" | "on" | "yes");
    match k {
        "profile" => {
            s.profile = flag(v);
            true
        }
        "profile_gpu" => {
            s.profile_gpu = flag(v);
            true
        }
        "profile_gpu_mode" => match v {
            "passes" => {
                s.profile_gpu_passes = true;
                true
            }
            "lists" => {
                s.profile_gpu_passes = false;
                true
            }
            _ => false,
        },
        "profile_labels" => {
            s.profile_labels = flag(v);
            true
        }
        "bench_warmup_s" => v.parse().map(|x| s.bench_warmup_s = x).is_ok(),
        "bench_static_s" => v.parse().map(|x| s.bench_static_s = x).is_ok(),
        "bench_pan_s" => v.parse().map(|x| s.bench_pan_s = x).is_ok(),
        "bench_repeats" => v.parse().map(|x| s.bench_repeats = x).is_ok(),
        "bench_uncap" => {
            s.bench_uncap = flag(v);
            true
        }
        "bench_label" => {
            s.bench_label = v.to_owned();
            true
        }
        "fsr" => {
            s.fsr = flag(v);
            true
        }
        "fsr_sharpness" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .map(|x| s.fsr_sharpness = x)
            .is_some(),
        "fsr_auto_exposure" => {
            s.fsr_auto_exposure = flag(v);
            true
        }
        "fsr_mode" => match v {
            "native" => {
                s.fsr_upscale = false;
                true
            }
            "upscale" => {
                s.fsr_upscale = true;
                true
            }
            _ => false,
        },
        "fsr_scale" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.33..=1.0).contains(x))
            .map(|x| s.fsr_scale = x)
            .is_some(),
        "fsr_mip_bias" if v == "auto" => {
            s.fsr_mip_bias = None;
            true
        }
        "fsr_mip_bias" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (-4.0..=4.0).contains(x))
            .map(|x| s.fsr_mip_bias = Some(x))
            .is_some(),
        "fsr_masks" => {
            s.fsr_masks = flag(v);
            true
        }
        "fsr_reactive" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .map(|x| s.fsr_reactive = x)
            .is_some(),
        "fsr_depth_resample" => {
            s.fsr_depth_resample = flag(v);
            true
        }
        "fsr_sky_motion" => {
            s.fsr_sky_motion = flag(v);
            true
        }
        "fsr_jitter_phases" if v == "auto" => {
            s.fsr_jitter_phases = None;
            true
        }
        "fsr_jitter_phases" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (4..=128).contains(x))
            .map(|x| s.fsr_jitter_phases = Some(x))
            .is_some(),
        "fsr_flush" => {
            s.fsr_flush = flag(v);
            true
        }
        "ao_mode" => match v {
            "hbao" => {
                s.ao_void = false;
                true
            }
            "void" => {
                s.ao_void = true;
                true
            }
            _ => false,
        },
        "ao_quality" => crate::ao::parse_quality(v)
            .map(|q| s.ao_quality = q)
            .is_some(),
        "ao_half_res" => {
            s.ao_half_res = flag(v);
            true
        }
        "ao_radius_scale" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.25..=4.0).contains(x))
            .map(|x| s.ao_radius_scale = x)
            .is_some(),
        "ao_strength" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=4.0).contains(x))
            .map(|x| s.ao_strength = x)
            .is_some(),
        "ao_debug" => {
            s.ao_debug = flag(v);
            true
        }
        "contact_shadows" => {
            s.contact_shadows = flag(v);
            true
        }
        "contact_length" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.05..=4.0).contains(x))
            .map(|x| s.contact_length = x)
            .is_some(),
        "contact_thickness" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.01..=2.0).contains(x))
            .map(|x| s.contact_thickness = x)
            .is_some(),
        "contact_quality" => crate::contact::parse_quality(v)
            .map(|q| s.contact_quality = q)
            .is_some(),
        "contact_debug" => {
            s.contact_debug = flag(v);
            true
        }
        "dof_sharpen" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .map(|x| s.dof_sharpen = x)
            .is_some(),
        "ssr" => {
            s.ssr = flag(v);
            true
        }
        "ssr_quality" => crate::ssr::parse_quality(v)
            .map(|q| s.ssr_quality = q)
            .is_some(),
        "ssr_half_res" => {
            s.ssr_half_res = flag(v);
            true
        }
        "ssr_strength" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=2.0).contains(x))
            .map(|x| s.ssr_strength = x)
            .is_some(),
        "ssr_roughness" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=0.6).contains(x))
            .map(|x| s.ssr_roughness = x)
            .is_some(),
        "ssr_reflectance" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=1.0).contains(x))
            .map(|x| s.ssr_reflectance = x)
            .is_some(),
        "ssr_depth_floors_only" => {
            s.ssr_depth_floors_only = flag(v);
            true
        }
        "ssr_thickness" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.05..=4.0).contains(x))
            .map(|x| s.ssr_thickness = x)
            .is_some(),
        "ssr_skip_water" => {
            s.ssr_skip_water = flag(v);
            true
        }
        "ssr_debug" => crate::ssr::parse_debug(v)
            .map(|d| s.ssr_debug = d)
            .is_some(),
        "ssr_material" => match v {
            "auto" => {
                s.ssr_material = true;
                true
            }
            "depth" => {
                s.ssr_material = false;
                true
            }
            _ => false,
        },
        "ssr_temporal" => {
            s.ssr_temporal = flag(v);
            true
        }
        "rt" => {
            s.rt = flag(v);
            true
        }
        "rt_build" => match v {
            "load" => {
                s.rt_build_at_load = true;
                true
            }
            "use" => {
                s.rt_build_at_load = false;
                true
            }
            _ => false,
        },
        "rt_radius" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (5.0..=1000.0).contains(x))
            .map(|x| s.rt_radius = x)
            .is_some(),
        "rt_dynamic" => {
            s.rt_dynamic = flag(v);
            true
        }
        "rt_characters" => {
            s.rt_characters = flag(v);
            true
        }
        "rt_characters_n" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=32).contains(x))
            .map(|x| s.rt_characters_n = x)
            .is_some(),
        "rt_dynamic_budget_ms" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.25..=8.0).contains(x))
            .map(|x| s.rt_dynamic_budget_ms = x)
            .is_some(),
        "rt_reflections" => {
            s.rt_reflections = flag(v);
            true
        }
        "rt_max_roughness" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.05..=1.0).contains(x))
            .map(|x| s.rt_max_roughness = x)
            .is_some(),
        "rt_ray_budget" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=100).contains(x))
            .map(|x| s.rt_ray_budget = x)
            .is_some(),
        "rt_reflect_debug" => crate::ssr::rt_reflect::parse_debug(v)
            .map(|d| s.rt_reflect_debug = d)
            .is_some(),
        "photo" => {
            s.photo = flag(v);
            true
        }
        "photo_key" => parse_key(v).map(|k| s.photo_key = k).is_some(),
        "photo_spp" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=65_536).contains(x))
            .map(|x| s.photo_spp = x)
            .is_some(),
        "photo_bounces" => v
            .parse::<u32>()
            .ok()
            .filter(|x| *x <= 16)
            .map(|x| s.photo_bounces = x)
            .is_some(),
        "photo_sky_scale" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=16.0).contains(x))
            .map(|x| s.photo_sky_scale = x)
            .is_some(),
        "photo_budget_ms" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (1.0..=50.0).contains(x))
            .map(|x| s.photo_budget_ms = x)
            .is_some(),
        "photo_texture_size" => v
            .parse::<u32>()
            .ok()
            .filter(|x| x.is_power_of_two() && (16..=4096).contains(x))
            .map(|x| s.photo_texture_size = x)
            .is_some(),
        "photo_display" => crate::photo::Display::parse(v)
            .map(|d| s.photo_display = d)
            .is_some(),
        "photo_sun" => {
            s.photo_sun = flag(v);
            true
        }
        "photo_ao" => {
            s.photo_ao = flag(v);
            true
        }
        "photo_reflections" => {
            s.photo_reflections = flag(v);
            true
        }
        "photo_sun_rays" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=4).contains(x))
            .map(|x| s.photo_sun_rays = x)
            .is_some(),
        "photo_ao_rays" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=4).contains(x))
            .map(|x| s.photo_ao_rays = x)
            .is_some(),
        "photo_reflect_rays" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=64).contains(x))
            .map(|x| s.photo_reflect_rays = x)
            .is_some(),
        "photo_terms_spp" => v
            .parse::<u32>()
            .ok()
            .filter(|x| (1..=65_536).contains(x))
            .map(|x| s.photo_terms_spp = x)
            .is_some(),
        "photo_sun_radius" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.0..=5.0).contains(x))
            .map(|x| s.photo_sun_radius = x)
            .is_some(),
        "photo_ao_radius" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.05..=20.0).contains(x))
            .map(|x| s.photo_ao_radius = x)
            .is_some(),
        "photo_ao_mode" => match v {
            "min" => {
                s.photo_ao_replace = false;
                true
            }
            "replace" => {
                s.photo_ao_replace = true;
                true
            }
            _ => false,
        },
        "photo_save_masks" => {
            s.photo_save_masks = flag(v);
            true
        }
        "photo_save_instances" => {
            s.photo_save_instances = flag(v);
            true
        }
        "photo_units" if v == "auto" => {
            s.photo_units = None;
            true
        }
        "photo_units" => v
            .parse::<f32>()
            .ok()
            .filter(|x| *x > 0.0 && *x <= 1000.0)
            .map(|x| s.photo_units = Some(x))
            .is_some(),
        "photo_dir" if !v.is_empty() => {
            s.photo_dir = v.to_owned();
            true
        }
        "photo_probe" => {
            s.photo_probe = flag(v);
            true
        }
        "texture_mods" => {
            s.texture_mods = flag(v);
            true
        }
        "mods_dir" if !v.is_empty() => {
            s.mods_dir = v.to_owned();
            true
        }
        "overlay" => {
            s.overlay = flag(v);
            true
        }
        "overlay_key" => parse_key(v).map(|k| s.overlay_key = k).is_some(),
        "overlay_scale" if v == "auto" => {
            s.overlay_scale = None;
            true
        }
        "overlay_scale" => v
            .parse::<f32>()
            .ok()
            .filter(|x| (0.5..=4.0).contains(x))
            .map(|x| s.overlay_scale = Some(x))
            .is_some(),
        "notifications" => {
            s.notifications = flag(v);
            true
        }
        "diagnostics" => {
            s.diagnostics = flag(v);
            true
        }
        "hud" => {
            s.hud = flag(v);
            true
        }
        "hud_key" => parse_key(v).map(|k| s.hud_key = k).is_some(),
        "bench_show_overlay" => {
            s.bench_show_overlay = flag(v);
            true
        }
        "hud_position" => Corner::parse(v).map(|c| s.hud_position = c).is_some(),
        _ if k.starts_with("patch.") => {
            let id = &k["patch.".len()..];
            let known = crate::patches::PATCHES.iter().any(|p| p.id == id);
            if known {
                s.patches.insert(id.to_owned(), flag(v));
            }
            known
                && matches!(
                    v,
                    "0" | "1" | "true" | "false" | "on" | "off" | "yes" | "no"
                )
        }
        _ => match k.strip_prefix("cvar.") {
            Some(name)
                if !name.is_empty()
                    && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') =>
            {
                v.parse::<f32>()
                    .ok()
                    .filter(|x| x.is_finite())
                    .map(|x| s.cvars.push((name.to_owned(), x)))
                    .is_some()
            }
            _ => false,
        },
    }
}

/// The default settings (comparisons and restores in the overlay).
static DEFAULTS: LazyLock<Settings> = LazyLock::new(Settings::default);

/// The file representation of `key`'s default value (`None` for unknown keys).
pub fn default_value(key: &str) -> Option<String> {
    value(&DEFAULTS, key)
}

/// `key` of `s` differs from its default.
pub fn is_changed(s: &Settings, key: &str) -> bool {
    value(s, key) != default_value(key)
}

/// Puts `key` of `s` back to its default value.
pub fn restore_default(s: &mut Settings, key: &str) {
    if let Some(v) = default_value(key) {
        apply(s, key, &v);
    }
}

/// Loads (or creates) the settings file in `dir`; returns notes for the log.
pub fn load(dir: &Path) -> Vec<String> {
    let path = dir.join("void-inject.ini");
    let mut notes = Vec::new();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => {
            let _ = std::fs::create_dir_all(dir);
            let text = default_file();
            match std::fs::write(&path, &text) {
                Ok(()) => notes.push(format!("settings: created {}", path.display())),
                Err(e) => notes.push(format!("settings: cannot create {}: {e}", path.display())),
            }
            text
        }
    };
    let (s, bad) = parse(&text);
    for b in bad {
        notes.push(format!("settings: ignored line {b:?}"));
    }
    notes.push(format!("settings: {s:?}"));
    *FILE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = s.clone();
    ACTIVE.store(Arc::new(s));
    let _ = PATH.set(path);
    notes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_file_parses_to_defaults() {
        let (s, bad) = parse(&default_file());
        assert!(bad.is_empty(), "{bad:?}");
        assert!(!s.profile && !s.diagnostics && s.bench_uncap);
        assert!(!s.texture_mods && s.mods_dir == "void-rs/mods");
        assert_eq!(s.bench_repeats, 3);
        assert!(s.overlay && s.overlay_key == 0x71 && s.overlay_scale.is_none());
        // The template's defaults are the struct defaults, cvar override aside.
        let d = Settings {
            cvars: vec![("image_anisotropy".to_owned(), 16.0)],
            ..Settings::default()
        };
        assert_eq!(s, d);
    }

    #[test]
    fn render_round_trips() {
        let text = default_file();
        assert!(text.contains("patch.skip_intro = 1\n"), "{text}");
        assert_eq!(render(&parse(&text).0), text);
        let (s, bad) = parse(
            "fsr_mode = upscale\nfsr_mip_bias = -0.5\noverlay_key = Insert\noverlay_scale = 1.5\ncvar.a = 1\ncvar.b = 2.5\nbench_label = x y\npatch.skip_intro = 0\n",
        );
        assert!(bad.is_empty(), "{bad:?}");
        let text = render(&s);
        assert!(text.contains("cvar.a = 1\ncvar.b = 2.5\n"), "{text}");
        assert!(text.contains("overlay_key = Insert\n") && text.contains("patch.skip_intro = 0\n"));
        assert_eq!(parse(&text).0, s);
    }

    #[test]
    fn patch_keys() {
        let (s, bad) = parse("patch.skip_intro = 0\npatch.nope = 1\npatch.skip_intro = maybe\n");
        assert_eq!(bad.len(), 2, "{bad:?}");
        assert_eq!(s.patches.get("skip_intro"), Some(&false));
        assert!(!s.patches.contains_key("nope"));
        assert!(!key_info("patch.skip_intro").unwrap().live);
    }

    #[test]
    fn overlay_keys() {
        assert_eq!(parse_key("F2"), Some(0x71));
        assert_eq!(parse_key("f12"), Some(0x7B));
        assert_eq!(parse_key("insert"), Some(0x2D));
        assert_eq!(parse_key("0x2d"), Some(0x2D));
        assert_eq!(parse_key("113"), Some(0x71));
        assert_eq!(
            (parse_key("F25"), parse_key("0"), parse_key("x")),
            (None, None, None)
        );
        for vk in [0x71, 0x2D, 0xC0, 0x41] {
            assert_eq!(parse_key(&key_name(vk)), Some(vk));
        }
        let (s, bad) = parse("overlay = 0\noverlay_scale = 9\noverlay_scale = 2\n");
        assert!(!s.overlay && s.overlay_scale == Some(2.0));
        assert_eq!(bad.len(), 1);
        let d = Settings::default();
        assert!(
            d.notifications && !d.hud && d.hud_key == 0x76 && d.hud_position == Corner::TopLeft
        );
        assert!(!d.bench_show_overlay && parse("bench_show_overlay = 1\n").0.bench_show_overlay);
        let (s, bad) = parse(
            "hud = 1\nhud_key = F8\nhud_position = bottom-right\nnotifications = 0\nhud_position = middle\n",
        );
        assert!(s.hud && !s.notifications && s.hud_key == 0x77);
        assert_eq!(s.hud_position, Corner::BottomRight);
        assert_eq!(bad.len(), 1);
        for c in Corner::ALL {
            assert_eq!(Corner::parse(c.name()), Some(c));
        }
    }

    #[test]
    fn live_keys_only_reach_the_active_copy() {
        let mut active = Settings::default();
        let mut edited = Settings {
            fsr_sharpness: 0.5,
            texture_mods: true,
            cvars: vec![("x".to_owned(), 1.0)],
            ..Settings::default()
        };
        edited.patches.insert("skip_intro".to_owned(), false);
        apply_live(&mut active, &edited);
        assert_eq!(active.fsr_sharpness, 0.5);
        assert!(!active.texture_mods);
        assert_eq!(active.cvars, edited.cvars);
        assert_eq!(active.patches.get("skip_intro"), Some(&true));
        for k in KEYS {
            assert!(
                k.name.ends_with('.') || value(&Settings::default(), k.name).is_some(),
                "{}",
                k.name
            );
        }
    }

    /// Every FSR setting is live, and `apply_live` really copies each of them.
    #[test]
    fn every_fsr_key_is_live() {
        let (edited, bad) = parse(
            "fsr = 1\nfsr_sharpness = 0.3\nfsr_auto_exposure = 1\nfsr_mode = upscale\nfsr_scale = 0.5\nfsr_mip_bias = -1\nfsr_masks = 0\nfsr_reactive = 0.4\nfsr_depth_resample = 0\nfsr_sky_motion = 0\nfsr_jitter_phases = 24\nfsr_flush = 0\n",
        );
        assert!(bad.is_empty(), "{bad:?}");
        let mut active = Settings::default();
        apply_live(&mut active, &edited);
        for k in KEYS.iter().filter(|k| k.name.starts_with("fsr")) {
            assert!(k.live, "{} is not live", k.name);
            assert_ne!(
                value(&Settings::default(), k.name),
                value(&edited, k.name),
                "test does not change {}",
                k.name
            );
            assert_eq!(value(&active, k.name), value(&edited, k.name), "{}", k.name);
        }
    }

    #[test]
    fn values_and_unknown_keys() {
        let (s, bad) = parse(
            "profile = 1\nbench_pan_s=12.5\nnope = 2\nbench_label = fsr on\ntexture_mods = 1\nmods_dir = void-rs/local/mods-test\n",
        );
        assert!(s.texture_mods && s.mods_dir == "void-rs/local/mods-test");
        assert!(s.profile);
        assert_eq!(s.bench_pan_s, 12.5);
        assert_eq!(s.bench_label, "fsr on");
        assert_eq!(bad, vec!["nope = 2".to_owned()]);
    }

    #[test]
    fn fsr_keys() {
        let (d, _) = parse(&default_file());
        assert!(
            !d.fsr && !d.fsr_upscale && d.fsr_masks && d.fsr_depth_resample && d.fsr_sky_motion
        );
        assert_eq!(
            (d.fsr_scale, d.fsr_mip_bias, d.fsr_reactive),
            (0.67, None, 0.9)
        );
        let (s, bad) = parse(
            "fsr_mode = upscale\nfsr_scale = 0.5\nfsr_mip_bias = -1.5\nfsr_masks = 0\nfsr_reactive = 0.5\nfsr_depth_resample = 0\nfsr_sky_motion = 0\nfsr_mode = big\nfsr_scale = 0.1\nfsr_mip_bias = x\n",
        );
        assert!(s.fsr_upscale && !s.fsr_masks && !s.fsr_depth_resample && !s.fsr_sky_motion);
        assert_eq!(
            (s.fsr_scale, s.fsr_mip_bias, s.fsr_reactive),
            (0.5, Some(-1.5), 0.5)
        );
        assert_eq!(bad.len(), 3, "{bad:?}");
    }

    /// Every AO setting is live, round-trips and reaches the active copy.
    #[test]
    fn ao_keys() {
        let d = Settings::default();
        assert!(d.ao_void && d.ao_half_res && !d.ao_debug);
        assert_eq!(d.ao_quality, crate::ao::DEFAULT_QUALITY);
        let (edited, bad) = parse(
            "ao_mode = hbao\nao_quality = ultra\nao_half_res = 0\nao_radius_scale = 1.5\nao_strength = 0.5\nao_debug = 1\nao_mode = ssao\nao_quality = max\nao_radius_scale = 9\nao_strength = -1\n",
        );
        assert_eq!(bad.len(), 4, "{bad:?}");
        let mut active = Settings::default();
        apply_live(&mut active, &edited);
        for k in KEYS.iter().filter(|k| k.name.starts_with("ao_")) {
            assert!(k.live, "{} is not live", k.name);
            assert_ne!(value(&d, k.name), value(&edited, k.name), "{}", k.name);
            assert_eq!(value(&active, k.name), value(&edited, k.name), "{}", k.name);
        }
        assert_eq!(parse(&render(&edited)).0, edited);
    }

    /// Every contact-shadow setting is live, round-trips and reaches the active copy.
    #[test]
    fn contact_keys() {
        let d = Settings::default();
        assert!(d.contact_shadows && !d.contact_debug);
        let (edited, bad) = parse(
            "contact_shadows = 0\ncontact_length = 1.5\ncontact_thickness = 0.1\ncontact_quality = ultra\ncontact_debug = 1\ncontact_length = 9\ncontact_quality = max\n",
        );
        assert_eq!(bad.len(), 2, "{bad:?}");
        let mut active = Settings::default();
        apply_live(&mut active, &edited);
        for k in KEYS.iter().filter(|k| k.name.starts_with("contact_")) {
            assert!(k.live, "{} is not live", k.name);
            assert_ne!(value(&d, k.name), value(&edited, k.name), "{}", k.name);
            assert_eq!(value(&active, k.name), value(&edited, k.name), "{}", k.name);
        }
        assert_eq!(parse(&render(&edited)).0, edited);
    }

    /// Every hybrid photo-term setting (and the display) is live, round-trips and reaches
    /// the active copy; out-of-range values are rejected.
    #[test]
    fn photo_hybrid_keys() {
        let d = Settings::default();
        assert_eq!(d.photo_display, crate::photo::Display::Hybrid);
        assert!(d.photo_sun && d.photo_ao && d.photo_reflections && !d.photo_ao_replace);
        let (edited, bad) = parse(
            "photo_display = pathtrace\nphoto_sun = 0\nphoto_ao = 0\nphoto_reflections = 0\nphoto_sun_rays = 2\nphoto_ao_rays = 1\nphoto_reflect_rays = 16\nphoto_terms_spp = 64\nphoto_sun_radius = 0.5\nphoto_ao_radius = 3\nphoto_ao_mode = replace\nphoto_save_masks = 1\nphoto_display = raster\nphoto_sun_rays = 5\nphoto_ao_radius = 100\nphoto_ao_mode = max\n",
        );
        assert_eq!(bad.len(), 4, "{bad:?}");
        let mut active = Settings::default();
        apply_live(&mut active, &edited);
        let hybrid = [
            "photo_display",
            "photo_sun",
            "photo_ao",
            "photo_reflections",
            "photo_sun_rays",
            "photo_ao_rays",
            "photo_reflect_rays",
            "photo_terms_spp",
            "photo_sun_radius",
            "photo_ao_radius",
            "photo_ao_mode",
            "photo_save_masks",
        ];
        for k in hybrid {
            assert!(key_info(k).is_some_and(|i| i.live), "{k} is not live");
            assert_ne!(value(&d, k), value(&edited, k), "{k}");
            assert_eq!(value(&active, k), value(&edited, k), "{k}");
        }
        assert_eq!(parse(&render(&edited)).0, edited);
    }

    /// Every reflection setting is live, round-trips and reaches the active copy.
    #[test]
    fn ssr_keys() {
        let d = Settings::default();
        assert!(
            !d.ssr
                && d.ssr_half_res
                && d.ssr_depth_floors_only
                && d.ssr_skip_water
                && d.ssr_temporal
        );
        assert_eq!(d.ssr_debug, 0);
        let (edited, bad) = parse(
            "ssr = 1\nssr_quality = ultra\nssr_half_res = 0\nssr_strength = 0.5\nssr_roughness = 0.3\nssr_reflectance = 0.5\nssr_depth_floors_only = 0\nssr_thickness = 1.5\nssr_skip_water = 0\nssr_debug = hits\nssr_material = depth\nssr_temporal = 0\nssr_roughness = 0.9\nssr_debug = all\nssr_material = maybe\n",
        );
        assert_eq!(bad.len(), 3, "{bad:?}");
        let mut active = Settings::default();
        apply_live(&mut active, &edited);
        for k in KEYS
            .iter()
            .filter(|k| k.name == "ssr" || k.name.starts_with("ssr_"))
        {
            assert!(k.live, "{} is not live", k.name);
            assert_ne!(value(&d, k.name), value(&edited, k.name), "{}", k.name);
            assert_eq!(value(&active, k.name), value(&edited, k.name), "{}", k.name);
        }
        assert_eq!(parse(&render(&edited)).0, edited);
    }

    /// Every key with a file value can be put back to its default (overlay restore icons).
    #[test]
    fn restore_defaults() {
        let (mut s, bad) = parse(
            "ao_mode = void\nao_quality = low\nao_half_res = 0\nao_radius_scale = 2\nao_strength = 0.25\nfsr_mode = upscale\nfsr_mip_bias = -1\nfsr_jitter_phases = 16\nfsr_scale = 0.5\nprofile_gpu_mode = lists\noverlay_scale = 2\nhud_position = bottom-left\nhud = 1\n",
        );
        assert!(bad.is_empty(), "{bad:?}");
        assert!(is_changed(&s, "ao_strength") && is_changed(&s, "fsr_mip_bias"));
        for k in KEYS.iter().filter(|k| !k.name.ends_with('.')) {
            restore_default(&mut s, k.name);
            assert!(!is_changed(&s, k.name), "{}", k.name);
        }
        assert_eq!(s, Settings::default());
        assert_eq!(default_value("ao_radius_scale").as_deref(), Some("1"));
        assert_eq!(default_value("ao_strength").as_deref(), Some("1"));
    }

    #[test]
    fn cvar_overrides_reject_bad_names_and_values() {
        let (s, bad) = parse(
            "cvar.image_anisotropy = 16\ncvar.r_lodZoomMultiplicator=1.5\ncvar. = 1\ncvar.a b = 1\ncvar.x = nan\ncvar.y = high\n",
        );
        assert_eq!(
            s.cvars,
            vec![
                ("image_anisotropy".to_owned(), 16.0),
                ("r_lodZoomMultiplicator".to_owned(), 1.5)
            ]
        );
        assert_eq!(bad.len(), 4, "{bad:?}");
    }
}
