//! Photo mode (docs/raytracing.md section 10, STATUS E073): host-side data of the progressive
//! path tracer in `photo.hlsl`: kernel entries and root layout, the photo-mode scene tables
//! (geometry table, triangle records, materials, bindless textures, engine lights + grid),
//! the `Params` block and the accumulation buffers. Sampling, tiling, resolve and the
//! denoiser are driven by [`crate::ingame::PhotoSession`] (the one renderer used offline
//! and in game).
//!
//! Units are the engine's absolute lighting units (sun = `sunColor` x [`SUN_COLOR_SCALE`],
//! light colours as captured, sky from `env.tsv`), so the result compares directly with
//! the game's HDR colour after the dump scale (docs/renderer-replacement.md: dump =
//! absolute x 3.58 in bright daylight).
//!
//! [`SUN_COLOR_SCALE`]: crate::lights::SUN_COLOR_SCALE

use crate::camera::RayBasis;
use crate::gpu::{Buffer, Entry, Gpu, Layout};
use crate::lights::{LIGHT_WORDS, LightGrid};
use crate::scene::{MATERIAL_WORDS, RECORD_WORDS, Scene};
use crate::textures::TextureSet;

/// Root layout of `photo.hlsl` (`RS`).
pub const LAYOUT: Layout = Layout {
    constants: 8,
    srvs: 8,
    uavs: 5,
    table: true,
};

/// Words of the `Params` structured buffer (t7).
pub const PARAM_WORDS: usize = 48;

/// `photo_trace` flags (`K.flags`).
pub const TF_NO_SUN: u32 = 1;
pub const TF_NO_SKY: u32 = 2;
pub const TF_NO_LIGHTS: u32 = 4;
pub const TF_NO_TEXTURES: u32 = 8;
/// Alpha-tested and decal geometry traced as opaque (`RAY_FLAG_FORCE_OPAQUE`).
pub const TF_NO_ALPHA: u32 = 16;

/// Dispatches per command list at most (TDR safety: every submit well under 2 s).
pub const MAX_DISPATCHES_PER_SUBMIT: u32 = 8;

/// Per-pixel accumulation and denoiser buffers (float4 each).
pub const ACCUM_BYTES_PER_PIXEL: u64 = 16;

/// Kernel entry points with their per-pixel buffer sizes (checked per dispatch). The
/// copy-ready outputs of `photo_depth` (u4) and `photo_resolve` (u3) have 256-byte row
/// pitches ([`depth_pitch`], [`hdr_pitch`]); their full padded size is checked by the
/// session on allocation, the per-pixel minimum here.
pub fn entries() -> [Entry<'static>; 11] {
    const SCENE: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7];
    const PARAMS: &[usize] = &[7];
    const NO_SRV_PX: &[u64] = &[0; 8];
    [
        Entry {
            name: "photo_trace",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[16, 16, 16, 0, 0],
        },
        Entry {
            name: "photo_demod",
            required_srvs: PARAMS,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[16, 16, 16, 16, 0],
        },
        Entry {
            name: "photo_atrous",
            required_srvs: PARAMS,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 0, 16, 16, 16],
        },
        Entry {
            name: "photo_remod",
            required_srvs: PARAMS,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[16, 16, 16, 16, 16],
        },
        Entry {
            name: "photo_depth",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 0, 0, 0, 4],
        },
        Entry {
            name: "photo_resolve",
            required_srvs: PARAMS,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[16, 0, 0, 16, 0],
        },
        // Offline debug output (albedo + material id, uv/LOD/depth of the primary hit).
        Entry {
            name: "photo_aov",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 0, 0, 16, 16],
        },
        // Hybrid photo terms (`hybrid.rs`): u1 game view depth (replaced by our surface's
        // depth or rejected by the prepass), u2 the prepass's packed receiver normal, u3 / u4
        // running means, u0 the filtered sun visibility, all packed floats in rows of
        // `depth_pitch` (full padded size checked by the session).
        Entry {
            name: "photo_hybrid_prep",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 4, 4, 0, 0],
        },
        Entry {
            name: "photo_sun_vis",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 4, 4, 4, 0],
        },
        Entry {
            name: "photo_ao",
            required_srvs: SCENE,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[0, 4, 4, 0, 4],
        },
        Entry {
            name: "photo_hybrid_filter",
            required_srvs: PARAMS,
            srv_bytes_per_pixel: NO_SRV_PX,
            uav_bytes_per_pixel: &[4, 4, 4, 4, 0],
        },
    ]
}

/// Row pitch in bytes of the resolved HDR buffer (RGBA32F, rows padded to 256 bytes;
/// `photo.hlsl` `hdr_row`).
pub fn hdr_pitch(width: u32) -> u64 {
    (u64::from(width) * 16).div_ceil(256) * 256
}

/// Row pitch in bytes of the depth buffer (R32F, rows padded to 256 bytes; `depth_row`).
pub fn depth_pitch(width: u32) -> u64 {
    (u64::from(width) * 4).div_ceil(256) * 256
}

/// Sun, sky and path settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Unit vector towards the sun.
    pub sun_dir: [f32; 3],
    /// Irradiance normal to the sun (engine units).
    pub sun_irradiance: [f32; 3],
    /// Angular radius of the sun disc (degrees).
    pub sun_radius_deg: f32,
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub sky_scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSettings {
    /// Bounces after the primary hit (0 = direct light only).
    pub max_bounces: u32,
    /// Russian roulette from this bounce on.
    pub rr_start: u32,
    /// Luminance clamp of every contribution after the primary hit (fireflies).
    pub clamp_indirect: f32,
    /// RIS candidates per local-light pick.
    pub ris_m: u32,
}

/// Scene data on the GPU (all read-only during sampling).
pub struct SceneGpu {
    pub geometry: Buffer,
    pub records: Buffer,
    pub materials: Buffer,
    pub lights: Buffer,
    pub cells: Buffer,
    pub light_indices: Buffer,
    pub textures: TextureSet,
    pub geom_count: u32,
    pub prim_count: u32,
    pub mat_count: u32,
    pub light_count: u32,
    pub grid: LightGrid,
}

impl SceneGpu {
    /// Uploads the photo-mode tables of `scene` and the local lights; large tables and the
    /// textures go up in submits of at most `batch_bytes` of staging data. With `dynamic`
    /// (dynamic objects, `crate::dynamic`): the records are UAV-capable (skinned normals are
    /// rewritten in place) and that many texture slots are reserved.
    pub fn upload(
        gpu: &mut Gpu,
        scene: &Scene,
        lights: &[[u32; LIGHT_WORDS]],
        grid: LightGrid,
        batch_bytes: u64,
        dynamic: Option<u32>,
    ) -> Result<Self, String> {
        let geom = scene.geometry_info();
        if geom.is_empty() || scene.records.is_empty() || scene.gpu_materials.is_empty() {
            return Err(
                "photo scene data missing (load with LoadOptions::photo_texture_size)".into(),
            );
        }
        let records_needed = geom
            .iter()
            .zip(scene.meshes.iter().flat_map(|m| m.geometries.iter()))
            .map(|(g, ge)| u64::from(g[0]) + u64::from(ge.index_count / 3))
            .max()
            .unwrap_or(0);
        if records_needed > scene.records.len() as u64 {
            return Err(format!(
                "geometry table addresses {records_needed} records, {} built",
                scene.records.len()
            ));
        }
        if let Some(&m) = scene.geometry_materials.iter().max()
            && m as usize >= scene.gpu_materials.len()
        {
            return Err(format!(
                "material index {m} >= {}",
                scene.gpu_materials.len()
            ));
        }
        let zero_light = [[0u32; LIGHT_WORDS]];
        let light_data: &[[u32; LIGHT_WORDS]] = if lights.is_empty() {
            &zero_light
        } else {
            lights
        };
        let zero_index = [0u32];
        let index_data: &[u32] = if grid.indices.is_empty() {
            &zero_index
        } else {
            &grid.indices
        };
        Ok(Self {
            geometry: gpu.upload_chunked(&geom, false, batch_bytes)?,
            records: gpu.upload_chunked(&scene.records, dynamic.is_some(), batch_bytes)?,
            materials: gpu.upload_chunked(&scene.gpu_materials, false, batch_bytes)?,
            lights: gpu.upload(light_data, false)?,
            cells: gpu.upload_chunked(&grid.cells, false, batch_bytes)?,
            light_indices: gpu.upload_chunked(index_data, false, batch_bytes)?,
            textures: TextureSet::upload(gpu, &scene.textures, batch_bytes, dynamic.unwrap_or(0))?,
            geom_count: geom.len() as u32,
            prim_count: scene.records.len() as u32,
            mat_count: scene.gpu_materials.len() as u32,
            light_count: lights.len() as u32,
            grid,
        })
    }

    /// Bytes of the uploaded tables (textures separately in `textures.bytes`).
    pub fn table_bytes(&self) -> u64 {
        [
            &self.geometry,
            &self.records,
            &self.materials,
            &self.lights,
            &self.cells,
            &self.light_indices,
        ]
        .iter()
        .map(|b| b.size)
        .sum()
    }

    /// Checks every count the kernels clamp against against the bound buffer sizes.
    pub fn validate(&self) -> Result<(), String> {
        let need = [
            ("geometry", &self.geometry, u64::from(self.geom_count) * 8),
            (
                "records",
                &self.records,
                u64::from(self.prim_count) * RECORD_WORDS as u64 * 4,
            ),
            (
                "materials",
                &self.materials,
                u64::from(self.mat_count) * MATERIAL_WORDS as u64 * 4,
            ),
            (
                "lights",
                &self.lights,
                u64::from(self.light_count.max(1)) * LIGHT_WORDS as u64 * 4,
            ),
            ("cells", &self.cells, self.grid.cells.len() as u64 * 8),
            (
                "light indices",
                &self.light_indices,
                self.grid.indices.len().max(1) as u64 * 4,
            ),
        ];
        for (name, b, bytes) in need {
            if b.size < bytes {
                return Err(format!("{name}: buffer {} bytes < {bytes}", b.size));
            }
        }
        if self.geom_count == 0
            || self.prim_count == 0
            || self.mat_count == 0
            || self.textures.count == 0
            || self.grid.cells.is_empty()
        {
            return Err("photo scene: empty table".into());
        }
        Ok(())
    }
}

/// The `Params` block (layout of `Params` in photo.hlsl).
pub fn params(
    basis: &RayBasis,
    znear: f32,
    l: &Lighting,
    s: &PathSettings,
    sg: &SceneGpu,
) -> [u32; PARAM_WORDS] {
    let mut w = [0u32; PARAM_WORDS];
    let v3 = |w: &mut [u32; PARAM_WORDS], at: usize, a: [f32; 3]| {
        for k in 0..3 {
            w[at + k] = a[k].to_bits();
        }
    };
    v3(&mut w, 0, basis.org);
    w[3] = s.max_bounces;
    v3(&mut w, 4, basis.fwd);
    w[7] = znear.to_bits();
    v3(&mut w, 8, basis.right);
    w[11] = basis.tan_x.to_bits();
    v3(&mut w, 12, basis.up);
    w[15] = basis.tan_y.to_bits();
    v3(&mut w, 16, l.sun_dir);
    w[19] = l.sun_radius_deg.to_radians().cos().to_bits();
    v3(&mut w, 20, l.sun_irradiance);
    w[23] = (2.0 * basis.tan_y / basis.height as f32).to_bits();
    v3(&mut w, 24, l.sky_zenith);
    w[27] = l.sky_scale.to_bits();
    v3(&mut w, 28, l.sky_horizon);
    w[31] = s.clamp_indirect.to_bits();
    w[32] = sg.geom_count;
    w[33] = sg.prim_count;
    w[34] = sg.mat_count;
    w[35] = sg.textures.count;
    w[36] = sg.light_count;
    w[37] = sg.grid.cells.len() as u32;
    w[38] = sg.grid.indices.len() as u32;
    w[39] = s.ris_m;
    w[40] = sg.grid.origin[0].to_bits();
    w[41] = sg.grid.origin[1].to_bits();
    w[42] = sg.grid.cell.to_bits();
    w[44] = sg.grid.dims[0];
    w[45] = sg.grid.dims[1];
    w[46] = s.rr_start;
    w
}

/// Accumulators (radiance + lum^2, albedo + sun visibility, normal + depth) and the two
/// denoiser buffers, `width x height` float4 each.
pub struct Accum {
    pub rad: Buffer,
    pub alb: Buffer,
    pub nrm: Buffer,
    pub a: Buffer,
    pub b: Buffer,
    pub width: u32,
    pub height: u32,
}

impl Accum {
    /// Allocates the buffers and zeroes the accumulators (one blocking submit).
    pub fn new(gpu: &mut Gpu, width: u32, height: u32) -> Result<Self, String> {
        let n = u64::from(width) * u64::from(height) * ACCUM_BYTES_PER_PIXEL;
        let acc = Self {
            rad: gpu.buffer(n, true)?,
            alb: gpu.buffer(n, true)?,
            nrm: gpu.buffer(n, true)?,
            a: gpu.buffer(n, true)?,
            b: gpu.buffer(n, true)?,
            width,
            height,
        };
        acc.clear(gpu)?;
        gpu.submit()?;
        Ok(acc)
    }

    /// Records the zeroing of the three accumulators (they must be in COMMON).
    pub fn clear(&self, gpu: &mut Gpu) -> Result<(), String> {
        for b in [&self.rad, &self.alb, &self.nrm] {
            gpu.clear(b)?;
        }
        Ok(())
    }

    pub fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    pub fn bytes(&self) -> u64 {
        [&self.rad, &self.alb, &self.nrm, &self.a, &self.b]
            .iter()
            .map(|b| b.size)
            .sum()
    }
}

/// Mean of accumulated float4 sums over `n` samples.
pub fn mean(sums: &[[f32; 4]], n: u32) -> Vec<[f32; 4]> {
    let k = 1.0 / n.max(1) as f32;
    sums.iter().map(|c| c.map(|v| v * k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_match_layout() {
        for e in entries() {
            assert_eq!(e.srv_bytes_per_pixel.len(), LAYOUT.srvs, "{}", e.name);
            assert_eq!(e.uav_bytes_per_pixel.len(), LAYOUT.uavs, "{}", e.name);
        }
        // Root signature budget: 64 DWORDs (constants + 2 per root descriptor + 1 table).
        let dwords = LAYOUT.constants + 2 * (LAYOUT.srvs + LAYOUT.uavs) + usize::from(LAYOUT.table);
        assert!(dwords <= 64, "{dwords}");
    }

    #[test]
    fn hlsl_declares_the_layout() {
        let src = crate::PHOTO_SHADER_SOURCE;
        assert!(src.contains(&format!("num32BitConstants={}", LAYOUT.constants)));
        for i in 0..LAYOUT.srvs {
            assert!(src.contains(&format!("SRV(t{i})")));
        }
        assert!(!src.contains(&format!("SRV(t{})", LAYOUT.srvs)));
        for i in 0..LAYOUT.uavs {
            assert!(src.contains(&format!("UAV(u{i})")));
        }
        assert!(!src.contains(&format!("UAV(u{})", LAYOUT.uavs)));
    }

    #[test]
    fn mean_divides() {
        assert_eq!(mean(&[[2.0, 4.0, 6.0, 8.0]], 2), vec![[1.0, 2.0, 3.0, 4.0]]);
    }
}
