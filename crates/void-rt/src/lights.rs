//! Engine lighting inputs for the photo renderer, from F3 snapshot files: the live render
//! lights (`light_parms.tsv`, the `idRenderLight` fields `sub_1402F8670` copies into
//! `arkTiledForwardLight_t`, E035), the sun (`sun.tsv`) and the environment's sky colours
//! (`env.tsv`). Interpretation follows the removed R3 renderer (checkpoint 9282529
//! `r3.rs` `LightParms::to_gpu` and the forward-lighting HLSL transcribed from
//! `lighting.decl`), whose direct sun matched the game's sunlit pixels within ~0.8-1.15
//! (docs/renderer-replacement.md, RX008/RX009); its local-light attenuation is H (not
//! validated per light). Area lights (type 3) are not modelled.
//!
//! A light reaches `cutoff` metres: attenuation `lerp(1, 1/d, decay)^2 *
//! smoothstep(cutoff, falloff, d)` times the spot / half-point cone. The kernel picks one
//! light per shading point by resampled importance sampling (RIS) over the light list of the
//! point's cell in a 2D (XY) grid; [`LightGrid`] lists every light whose range circle
//! touches a cell.

/// `parallelLightColor` = `sunColor` x this (O: checkpoint 9282529 `SunParms::parallel_light`,
/// 89,939.73 / (1750 / 4 pi) x 0.889; the type-4 sun light's colour is 645.84 per unit
/// `sunColor` and `parallelLightColor` is 0.889 of it).
pub const SUN_COLOR_SCALE: f32 = 89_939.73 / (1750.0 / (4.0 * std::f32::consts::PI)) * 0.889;

/// One `light_parms.tsv` row (columns as the header written by the F3 snapshot).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LightParms {
    pub flags0: u8,
    pub flags1: u8,
    pub view_mask: u8,
    /// Rows at +4, +16, +28 of the light.
    pub axis: [f32; 9],
    pub origin: [f32; 3],
    pub color: [f32; 3],
    /// 0 point, 1 half-point, 2 spot, 3 area, 4 parallel.
    pub kind: u32,
    pub falloff: f32,
    pub cutoff: f32,
    pub decay: f32,
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub spec_attn: f32,
}

fn csv<const N: usize>(s: &str) -> Option<[f32; N]> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse().ok())
        .collect::<Option<_>>()?;
    v.try_into().ok()
}

/// 32-bit words per GPU light (5 x float4, see [`LightParms::to_gpu`]).
pub const LIGHT_WORDS: usize = 20;

impl LightParms {
    pub fn from_tsv(line: &str) -> Option<Self> {
        let f: Vec<&str> = line.split('\t').collect();
        let num = |i: usize| -> Option<f32> { f.get(i)?.trim().parse().ok() };
        Some(Self {
            flags0: f.first()?.trim().parse().ok()?,
            flags1: f.get(1)?.trim().parse().ok()?,
            view_mask: f.get(2)?.trim().parse().ok()?,
            axis: csv(f.get(3)?)?,
            origin: csv(f.get(4)?)?,
            color: csv(f.get(5)?)?,
            kind: f.get(6)?.trim().parse().ok()?,
            falloff: num(7)?,
            cutoff: num(8)?,
            decay: num(9)?,
            cos_inner: num(10)?,
            cos_outer: num(11)?,
            spec_attn: num(12)?,
        })
    }

    /// Parses `light_parms.tsv` (header line first).
    pub fn parse_file(text: &str) -> Result<Vec<Self>, String> {
        text.lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| Self::from_tsv(l).ok_or_else(|| format!("light_parms.tsv: bad row {l:?}")))
            .collect()
    }

    /// Range in metres: `smoothstep(cutoff, falloff, d)` is 0 from `cutoff` on when
    /// `cutoff > falloff`; otherwise unbounded (None).
    pub fn range(&self) -> Option<f32> {
        (self.cutoff > self.falloff).then_some(self.cutoff)
    }

    /// GPU record (5 x float4): (pos, falloff), (colour, cutoff), (dir, decay),
    /// (spot params, kind as uint), (specAttn, range, 0, 0). `None` for lights the engine's
    /// fill loop skips (flag bit 0 clear, no view-mask bit, parallel) and for area lights
    /// (type 3), as checkpoint 9282529 `LightParms::to_gpu`.
    pub fn to_gpu(&self) -> Option<[u32; LIGHT_WORDS]> {
        if self.flags0 & 1 == 0 || self.view_mask & 3 == 0 || self.kind >= 3 {
            return None;
        }
        let range = self.range()?;
        let row0 = [self.axis[0], self.axis[1], self.axis[2]];
        let len = row0.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-19);
        let fwd = row0.map(|v| v / len);
        let (dir, spot) = match self.kind {
            // Point: no cone.
            0 => (fwd, [0.0; 3]),
            // Half-point: dir = row0, spotParms = (-1 / cos, 1, 0).
            1 => (fwd, [1.0 / -self.cos_inner, 1.0, 0.0]),
            // Spot: dir = -row0; c_out = min(c_out, c_in - 0.001).
            _ => {
                let c_in = self.cos_inner;
                let c_out = self.cos_outer.min(c_in - 0.001);
                (
                    fwd.map(|v| -v),
                    [
                        1.0 / (c_out - c_in),
                        -(c_in / (c_out - c_in)),
                        (1.0 - c_out * c_out).max(0.0).sqrt() / c_out,
                    ],
                )
            }
        };
        let f = |v: f32| v.to_bits();
        let mut w = [0u32; LIGHT_WORDS];
        w[0..3].copy_from_slice(&self.origin.map(f));
        w[3] = f(self.falloff);
        w[4..7].copy_from_slice(&self.color.map(f));
        w[7] = f(self.cutoff);
        w[8..11].copy_from_slice(&dir.map(f));
        w[11] = f(self.decay.clamp(0.0, 1.0));
        w[12..15].copy_from_slice(&spot.map(f));
        w[15] = self.kind;
        w[16] = f(self.spec_attn);
        w[17] = f(range);
        Some(w)
    }
}

/// `env.tsv`: `env/indirectLighting/bounceFactor/skyZenith`, `.../skyHorizon`, `ambient`
/// (checkpoint 9282529 `EnvParms`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Env {
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub ambient: [f32; 3],
}

impl Env {
    pub fn from_tsv(line: &str) -> Option<Self> {
        let f: Vec<&str> = line.trim_end().split('\t').collect();
        Some(Self {
            sky_zenith: csv(f.first()?)?,
            sky_horizon: csv(f.get(1)?)?,
            ambient: csv(f.get(2)?)?,
        })
    }
}

/// Lights bucketed per XY cell: `cells[y * dims[0] + x] = (first, count)` into `indices`.
#[derive(Clone, Debug, PartialEq)]
pub struct LightGrid {
    pub origin: [f32; 2],
    pub cell: f32,
    pub dims: [u32; 2],
    pub cells: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
}

impl LightGrid {
    /// Grid over the union of the lights' range circles with cells of at least `cell`
    /// metres and at most `max_dim` cells per axis.
    pub fn build(lights: &[[u32; LIGHT_WORDS]], cell: f32, max_dim: u32) -> Self {
        let pr = |l: &[u32; LIGHT_WORDS]| {
            (
                [f32::from_bits(l[0]), f32::from_bits(l[1])],
                f32::from_bits(l[17]),
            )
        };
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        for l in lights {
            let (p, r) = pr(l);
            for k in 0..2 {
                lo[k] = lo[k].min(p[k] - r);
                hi[k] = hi[k].max(p[k] + r);
            }
        }
        if lights.is_empty() {
            return Self {
                origin: [0.0; 2],
                cell: 1.0,
                dims: [1, 1],
                cells: vec![[0, 0]],
                indices: Vec::new(),
            };
        }
        let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let cell = cell.max(extent / max_dim as f32).max(1e-3);
        let dims = [0, 1].map(|k| (((hi[k] - lo[k]) / cell).ceil() as u32).clamp(1, max_dim));
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); (dims[0] * dims[1]) as usize];
        for (i, l) in lights.iter().enumerate() {
            let (p, r) = pr(l);
            let c0 = [0, 1]
                .map(|k| (((p[k] - r - lo[k]) / cell).floor().max(0.0) as u32).min(dims[k] - 1));
            let c1 = [0, 1]
                .map(|k| (((p[k] + r - lo[k]) / cell).floor().max(0.0) as u32).min(dims[k] - 1));
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    // Closest point of the cell rectangle to the light.
                    let q = [
                        p[0].clamp(lo[0] + x as f32 * cell, lo[0] + (x + 1) as f32 * cell),
                        p[1].clamp(lo[1] + y as f32 * cell, lo[1] + (y + 1) as f32 * cell),
                    ];
                    if (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) <= r * r {
                        lists[(y * dims[0] + x) as usize].push(i as u32);
                    }
                }
            }
        }
        let mut cells = Vec::with_capacity(lists.len());
        let mut indices = Vec::new();
        for l in lists {
            cells.push([indices.len() as u32, l.len() as u32]);
            indices.extend(l);
        }
        Self {
            origin: lo,
            cell,
            dims,
            cells,
            indices,
        }
    }

    pub fn max_per_cell(&self) -> u32 {
        self.cells.iter().map(|c| c[1]).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: &str = "1\t64\t3\t0.83146894,0.55557144,0,-0.5555714,0.8314688,0,0,0,1\t-47.39657,93.35668,23.11604\t0.31511378,0.14382128,0.010431767\t0\t0\t1\t1\t0.9961947\t0.81915206\t1";

    #[test]
    fn parses_snapshot_row_and_packs_point_light() {
        let l = LightParms::from_tsv(ROW).unwrap();
        assert_eq!(l.kind, 0);
        assert_eq!(l.range(), Some(1.0));
        let g = l.to_gpu().unwrap();
        assert_eq!(f32::from_bits(g[0]), -47.39657);
        assert_eq!(f32::from_bits(g[17]), 1.0);
        // Parallel and area lights are not local lights.
        assert!(LightParms { kind: 4, ..l }.to_gpu().is_none());
        assert!(LightParms { kind: 3, ..l }.to_gpu().is_none());
        assert!(LightParms { flags0: 0, ..l }.to_gpu().is_none());
    }

    #[test]
    fn spot_cone_maps_inner_to_one_and_outer_to_zero() {
        let l = LightParms {
            flags0: 1,
            view_mask: 3,
            axis: [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            kind: 2,
            falloff: 1.0,
            cutoff: 5.0,
            decay: 1.0,
            cos_inner: 0.9,
            cos_outer: 0.5,
            ..Default::default()
        };
        let g = l.to_gpu().unwrap();
        let spot = [f32::from_bits(g[12]), f32::from_bits(g[13])];
        let att = |c: f32| 1.0 - (c * spot[0] + spot[1]).clamp(0.0, 1.0);
        assert!((att(0.9) - 1.0).abs() < 1e-5);
        assert!(att(0.5).abs() < 1e-5);
        // Spot direction is -row0 (pointing up here).
        assert_eq!(f32::from_bits(g[10]), 1.0);
    }

    #[test]
    fn grid_lists_lights_by_range_circle() {
        let mk = |x: f32, y: f32, r: f32| {
            LightParms {
                flags0: 1,
                view_mask: 3,
                axis: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                origin: [x, y, 0.0],
                color: [1.0; 3],
                falloff: 0.0,
                cutoff: r,
                ..Default::default()
            }
            .to_gpu()
            .unwrap()
        };
        let lights = [mk(0.0, 0.0, 1.0), mk(10.0, 0.0, 1.0)];
        let g = LightGrid::build(&lights, 1.0, 64);
        let at = |x: f32, y: f32| {
            let c = [
                ((x - g.origin[0]) / g.cell) as u32,
                ((y - g.origin[1]) / g.cell) as u32,
            ];
            let [first, n] = g.cells[(c[1] * g.dims[0] + c[0]) as usize];
            g.indices[first as usize..(first + n) as usize].to_vec()
        };
        assert_eq!(at(0.1, 0.1), vec![0]);
        assert_eq!(at(10.1, 0.1), vec![1]);
        assert!(at(5.0, 0.1).is_empty());
        assert!(g.max_per_cell() >= 1);
    }

    #[test]
    fn env_and_sun_scale() {
        let e = Env::from_tsv("0,0,0\t37.7451,52.411766,55\t1,1,1\n").unwrap();
        assert_eq!(e.sky_horizon, [37.7451, 52.411766, 55.0]);
        // RX010 r3a params.txt: sunColor 0.21586096 -> colour 123.93651.
        assert!((0.215_860_96 * SUN_COLOR_SCALE - 123.936_51).abs() < 0.01);
    }
}
