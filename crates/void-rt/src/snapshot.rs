//! F3 scene snapshots written by the in-process module (`local/inject/snap_<time>_f<frame>/`,
//! format of checkpoint 9282529 `snapshot.rs`): camera (`view.tsv`), sun (`sun.tsv`), cvars
//! (`cvars.tsv`), live render models (`models.tsv`) and their runtime surfaces
//! (`surfaces.tsv`), environment sky colours (`env.tsv`) and live lights
//! (`light_parms.tsv`). Read only; nothing here touches the game.

use std::collections::HashMap;
use std::path::Path;

use crate::camera::Camera;
use crate::lights::{Env, LightParms};

/// `r_znear` default (O: cvar registration, `cvars.tsv` default column).
pub const DEFAULT_ZNEAR: f32 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    pub theta_deg: f32,
    pub phi_deg: f32,
    /// `sunColor` as captured (relative; the engine's scale is not applied).
    pub color: [f32; 3],
    /// Unit vector towards the sun: `(sin t cos p, sin t sin p, cos t)` (O: matches the
    /// `dir` the R3 renderer logged for RX010, `local/inject/r3a_1792183406_f1352`).
    pub dir: [f32; 3],
}

impl Sun {
    pub fn from_angles(theta_deg: f32, phi_deg: f32, color: [f32; 3]) -> Self {
        let (st, ct) = theta_deg.to_radians().sin_cos();
        let (sp, cp) = phi_deg.to_radians().sin_cos();
        Self {
            theta_deg,
            phi_deg,
            color,
            dir: [st * cp, st * sp, ct],
        }
    }
}

/// A static model instance (row-major 3x4 local-to-world).
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub model_source: String,
    pub transform: [f32; 12],
    /// Material overrides in surface order (may be empty or shorter than the model's list).
    pub materials: Vec<String>,
    /// `true`: the list is the live runtime surface list and replaces the model file's
    /// materials only as a whole when the counts agree (checkpoint 9282529 rule);
    /// `false`: `m_customMaterials` items, each non-empty item replaces surface `i`.
    pub runtime_materials: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EntityFilter {
    pub rows: usize,
    pub skinned: usize,
    pub not_drawn_flag: usize,
    pub editor_material: usize,
    pub no_model: usize,
}

pub struct Snapshot {
    pub camera: Camera,
    pub znear: f32,
    pub sun: Option<Sun>,
    pub entities: Vec<Entity>,
    pub filter: EntityFilter,
    /// `env.tsv` (absent in older snapshots).
    pub env: Option<Env>,
    /// `light_parms.tsv` rows (empty when absent).
    pub lights: Vec<LightParms>,
}

fn floats(s: &str) -> Vec<f32> {
    s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
}

/// `idRenderModel` flags byte bit 4 = `m_mayBeDrawn` (O, RX004; checkpoint 9282529 `r3.rs`).
pub fn flags_may_be_drawn(flags: u32) -> bool {
    flags & 0x10 != 0
}

/// Materials of never-drawn editor/collision geometry (O: checkpoint 9282529 `r3.rs`).
pub fn material_is_drawn(material: &str) -> bool {
    let m = material.to_ascii_lowercase();
    !(m.starts_with("editors/") || m.starts_with("models/templates/collisions/"))
}

impl Snapshot {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let read = |n: &str| {
            std::fs::read_to_string(dir.join(n))
                .map_err(|e| format!("{}: {e}", dir.join(n).display()))
        };
        let view = read("view.tsv")?;
        let row: Vec<&str> = view
            .lines()
            .nth(1)
            .ok_or("view.tsv: no view row")?
            .split('\t')
            .collect();
        let (org, axis, fov) = (
            floats(row.get(3).ok_or("view.tsv: vieworg")?),
            floats(row.get(4).ok_or("view.tsv: viewaxis")?),
            floats(row.get(5).ok_or("view.tsv: fov")?),
        );
        if org.len() != 3 || axis.len() != 9 || fov.len() != 2 {
            return Err(format!("view.tsv: malformed row {row:?}"));
        }
        let camera = Camera {
            org: [org[0], org[1], org[2]],
            axis: [
                [axis[0], axis[1], axis[2]],
                [axis[3], axis[4], axis[5]],
                [axis[6], axis[7], axis[8]],
            ],
            fov_y: fov[1],
        };
        let cvars = read("cvars.tsv").unwrap_or_default();
        let znear = cvars
            .lines()
            .map(|l| l.split('\t').collect::<Vec<_>>())
            .find(|f| f.first() == Some(&"r_znear"))
            .and_then(|f| f.get(3)?.parse().ok())
            .unwrap_or(DEFAULT_ZNEAR);
        let sun = read("sun.tsv").ok().and_then(|s| {
            let f: Vec<&str> = s.split_whitespace().collect();
            let c = floats(f.get(2)?);
            Some(Sun::from_angles(
                f.first()?.parse().ok()?,
                f.get(1)?.parse().ok()?,
                [*c.first()?, *c.get(1)?, *c.get(2)?],
            ))
        });
        let (entities, filter) = parse_models(&read("models.tsv")?, &read("surfaces.tsv")?)?;
        let env = read("env.tsv")
            .ok()
            .and_then(|s| Env::from_tsv(s.lines().next()?));
        let lights = match read("light_parms.tsv") {
            Ok(t) => LightParms::parse_file(&t)?,
            Err(_) => Vec::new(),
        };
        Ok(Self {
            camera,
            znear,
            sun,
            entities,
            filter,
            env,
            lights,
        })
    }
}

/// Static, drawable model instances of `models.tsv` with their runtime materials.
pub fn parse_models(models: &str, surfaces: &str) -> Result<(Vec<Entity>, EntityFilter), String> {
    let mut surface_materials: HashMap<&str, Vec<String>> = HashMap::new();
    for l in surfaces.lines().skip(1) {
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 3 {
            return Err(format!("surfaces.tsv: short row {l:?}"));
        }
        surface_materials
            .entry(f[0])
            .or_default()
            .push(f[2].to_owned());
    }
    let mut out = Vec::new();
    let mut filter = EntityFilter::default();
    for l in models.lines().skip(1) {
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 12 {
            return Err(format!("models.tsv: short row {l:?}"));
        }
        filter.rows += 1;
        let (ptr, name) = (f[2], f[3]);
        if name.is_empty() {
            filter.no_model += 1;
            continue;
        }
        if name.to_ascii_lowercase().ends_with(".md6") {
            filter.skinned += 1;
            continue;
        }
        let flags = u32::from_str_radix(f[11].trim_start_matches("0x"), 16).unwrap_or(0);
        if !flags_may_be_drawn(flags) {
            filter.not_drawn_flag += 1;
            continue;
        }
        let materials = surface_materials.get(ptr).cloned().unwrap_or_default();
        if !materials.first().is_none_or(|m| material_is_drawn(m)) {
            filter.editor_material += 1;
            continue;
        }
        let (o, ax, s) = (floats(f[4]), floats(f[5]), floats(f[6]));
        if o.len() != 3 || ax.len() != 9 || s.len() != 3 {
            return Err(format!("models.tsv: malformed transform in {l:?}"));
        }
        // Axis rows are the local axes in world space (E023 convention).
        let mut t = [0f32; 12];
        for r in 0..3 {
            for c in 0..3 {
                t[r * 4 + c] = ax[c * 3 + r] * s[c];
            }
            t[r * 4 + 3] = o[r];
        }
        out.push(Entity {
            model_source: name.to_owned(),
            transform: t,
            materials,
            runtime_materials: true,
        });
    }
    Ok((out, filter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_direction_matches_r3_log() {
        // RX010: sun theta 26 phi 100 -> dir [-0.07612236, 0.4317113, 0.89879405].
        let s = Sun::from_angles(26.0, 100.0, [1.0; 3]);
        let want = [-0.076_122_36, 0.431_711_3, 0.898_794_05];
        for k in 0..3 {
            assert!((s.dir[k] - want[k]).abs() < 1e-5, "{:?}", s.dir);
        }
    }

    #[test]
    fn models_filter_and_transform() {
        let models = "h\th\th\th\th\th\th\th\th\th\th\th\n\
            w\t0\t0xa\tmodels/x.lwo\t1,2,3\t0,1,0,-1,0,0,0,0,1\t2,1,1\t-\t1\t0\t0x8\t0x10\n\
            w\t1\t0xb\tmodels/p/body.md6\t0,0,0\t1,0,0,0,1,0,0,0,1\t1,1,1\t-\t1\t0\t0x8\t0x10\n\
            w\t2\t0xc\tmodels/y.lwo\t0,0,0\t1,0,0,0,1,0,0,0,1\t1,1,1\t-\t1\t0\t0x8\t0x0\n\
            w\t3\t0xd\tvol\t0,0,0\t1,0,0,0,1,0,0,0,1\t1,1,1\t-\t1\t0\t0x8\t0x10\n";
        let surfaces = "model\tsurface\tmaterial\n0xa\t0\tm/a\n0xa\t1\tm/b\n0xd\t0\teditors/v\n";
        let (e, f) = parse_models(models, surfaces).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].materials, ["m/a", "m/b"]);
        // local x (scaled 2) -> world +y; translation column.
        assert_eq!(
            e[0].transform,
            [0.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0]
        );
        assert_eq!(
            f,
            EntityFilter {
                rows: 4,
                skinned: 1,
                not_drawn_flag: 1,
                editor_material: 1,
                no_model: 0
            }
        );
    }
}
