//! Ray-tracing scene service inside the game (docs/raytracing.md section 11), shared by
//! the photo mode (`photo/`) and live ray-traced effects:
//! - [`device`]: the D3D12 device on the game's adapter (the same per-adapter singleton FSR
//!   uses), queues, D3D11 <-> D3D12 shared textures and fences;
//! - [`scene`]: the map's static scene (BLAS, materials, textures, lights) built on a worker
//!   thread per map, a camera cut-off TLAS for live use and the per-frame [`scene::SceneView`];
//! - [`live`]: copies of engine state read from game memory (entities, lights, sun, sky);
//! - [`track`], [`dynamic`]: dynamic objects (section 14): what moved, appeared or posed in
//!   the engine this frame (main thread) and its application to the scene (worker).
//!
//! Setting `rt = 0` (default): no device, no thread, no hook. `rt_dynamic = 0` and
//! `rt_characters = 0` (defaults): no reserve, no tracker, no per-frame reads.

pub mod device;
pub mod dynamic;
pub mod live;
pub mod scene;
pub mod track;

pub use scene::{Lighting, SceneView, frame_scene};
