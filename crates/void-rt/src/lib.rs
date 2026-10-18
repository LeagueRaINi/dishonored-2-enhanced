//! Ray tracing over the static scene decoded from the game's archives
//! (docs/raytracing.md): the DXR 1.1 inline-`RayQuery` spike (STATUS E070), the photo-mode
//! path tracer (E073) and its in-process API for the game's DLL ([`ingame`]).
//!
//! - [`scene`]: world + static entity geometry, one BLAS per object; spike per-vertex
//!   shading data (Far Cry 6 primitive buffer) or photo-mode records, materials and textures;
//! - [`accel`]: compacted BLAS / TLAS builds (rebuild, refit) with timings and memory;
//! - [`gpu`], [`dxc`], [`textures`]: D3D12 context (own or host device, root-descriptor
//!   buffers, bindless texture table, non-blocking submits) and run-time DXC;
//! - [`camera`], [`snapshot`], [`lights`]: engine camera/depth conventions, F3 snapshots,
//!   engine lights / sun / sky inputs and the light grid;
//! - [`photo`], [`ingame`]: photo-mode kernel data and the session that drives them
//!   (tiled, budgeted, offline and in game) with build-time DXIL;
//! - [`hybrid`]: ray-traced sun visibility and AO for the game's own paused frame (the
//!   photo mode's hybrid terms);
//! - [`dynamic`], [`character`]: dynamic objects over an uploaded scene (moved / hidden
//!   instances, spawned rigid meshes, GPU-skinned `.md6` characters with refitted BLASes);
//! - [`image`], [`run`]: image output and comparison, shared tool helpers.
//!
//! The offline tools only read the archives; the in-game host owns every game access.

pub mod accel;
pub mod camera;
pub mod character;
pub mod dxc;
pub mod dynamic;
pub mod gpu;
pub mod hybrid;
pub mod image;
pub mod ingame;
pub mod lights;
pub mod photo;
pub mod run;
pub mod scene;
pub mod snapshot;
pub mod textures;

/// The spike kernels (`primary`, `reflect_rays`, `ssr_ref`, `gi`).
pub const SHADER_SOURCE: &str = include_str!("rt.hlsl");

/// The photo-mode kernels (`photo_trace`, `photo_demod`, `photo_atrous`, `photo_remod`).
pub const PHOTO_SHADER_SOURCE: &str = include_str!("photo.hlsl");
