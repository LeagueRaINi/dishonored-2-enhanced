//! Steady-state probe lighting (static + sun + sky + ambient terms of
//! `indirectlighting/update/accumulatesh.decl`), computed on the CPU from the baked
//! `.bprobe` results, plus tetrahedron point location for per-object probe lookup.
//!
//! O (exe `b671f824...6180`): per-probe upload `sub_140291430` (probe info: static
//! bounce x AO_global, dynamic bounce, sky bounce x AO_sky, sun bounce x AO_global,
//! ambient factor; SH buffers = `*_WithOccluders`), `sub_1402B75B0` called with
//! "filtered" = 1 by `sub_1402B3AB0` (base = static SH, sun factor *filtered*, sky
//! occlusion *filtered*, plus occluder SH x occluder state), sky ZH `sub_1402B0370`,
//! accumulation inputs `sub_1402B4380` (parmsFloat = (1, 1), ambient = env
//! `env/indirectLighting/ambient`, sun colour = `parallelLightColor`).
//! Multi-bounce: `sub_140291430` adds `r_ILProbeUpdate_ContributionFactorMultiBounces`
//! (3.0) x the baked `*_Initial` SH to the static, Sun and sky-occlusion SH at load.
//! Not modelled: the dynamic relight term (probe samples x probe lights, L3b),
//! occluder contributions (state weights not recovered; treated as 0), temporal
//! smoothing (steady state = accumulated value).

use void_formats::probes::{EnvProbe, ProbeNetwork};

pub type Sh9 = [[f32; 3]; 9];

/// Environment inputs of the accumulation (env decl / live view parameters).
#[derive(Clone, Copy, Debug)]
pub struct ProbeEnv {
    pub bounce_static: [f32; 3],
    pub bounce_dynamic: [f32; 3],
    pub bounce_sun: [f32; 3],
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub ambient: [f32; 3],
    /// `atm/sky/sunColorAtGroundLevel` (= `parallelLightColor`).
    pub sun_color: [f32; 3],
    /// `r_ILProbeUpdate_ContributionFactorMultiBounces` (default 3.0; flags 7 = static,
    /// Sun and sky all enabled). Negative = replace the baked SH (`sub_140291430`).
    pub multi_bounce: f32,
}

/// `sub_1402B0370`: zonal harmonics of the sky (l = 0, 1, 2) from zenith and horizon
/// colours: 40 equal-area rings over the upper hemisphere, colour lerped by polar angle.
/// Transcribed as is, including the ring weight 2 * 2pi / 40.
pub fn sky_zh(zenith: [f32; 3], horizon: [f32; 3]) -> [[f32; 3]; 3] {
    let tiny = |v: f32| if v.abs() <= 1e-18 { 0.0 } else { v };
    let (z, h) = (zenith.map(tiny), horizon.map(tiny));
    let dw = (std::f32::consts::TAU * 2.0) / 40.0;
    let k0 = (1.0 / std::f32::consts::PI).sqrt() * 0.5 * dw;
    let k2 = (5.0 / std::f32::consts::PI).sqrt() * 0.25 * dw;
    let mut out = [[0f32; 3]; 3];
    for i in 0..40 {
        let s = (((i as f32 + 0.5) * 0.5) / 40.0).sqrt();
        let theta = 2.0 * s.clamp(-1.0, 1.0).asin();
        let t = (theta * 2.0) / std::f32::consts::PI;
        let c = theta.cos();
        let p2 = c * 3.0 * c - 1.0;
        for k in 0..3 {
            let col = (1.0 - t) * z[k] + h[k] * t;
            out[0][k] += col * k0;
            out[2][k] += p2 * col * k2;
        }
    }
    out
}

/// `ZHSHProductSky` (accumulatesh.decl): product of a ZH (a) with SH coefficients (b).
pub fn zh_sh_product_sky(a: &[[f32; 3]; 3], b: &Sh9) -> Sh9 {
    const C0: f32 = 0.282_094_8;
    const C1: f32 = -0.126_156_63;
    const C3: f32 = 0.252_313_26;
    const C4: f32 = 0.180_223_75;
    const C6: f32 = 0.090_111_875;
    let mut r = [[0f32; 3]; 9];
    for k in 0..3 {
        let c0a0 = C0 * a[0][k];
        let c1a2 = C1 * a[2][k];
        r[0][k] = c0a0 * b[0][k];
        r[1][k] = b[1][k] * (c0a0 + c1a2);
        r[2][k] = (c0a0 + C3 * a[2][k]) * b[2][k];
        r[3][k] = b[3][k] * (c0a0 + c1a2);
        r[4][k] = b[4][k] * (c0a0 - C4 * a[2][k]);
        r[5][k] = b[5][k] * (c0a0 + C6 * a[2][k]);
        r[6][k] = C0 * (b[6][k] * a[0][k] + b[0][k] * a[2][k]);
        let t = a[2][k] * b[6][k];
        r[0][k] += C0 * t;
        r[6][k] += C4 * t;
        r[7][k] = b[7][k] * (c0a0 + C6 * a[2][k]);
        r[8][k] = b[8][k] * (c0a0 - C4 * a[2][k]);
    }
    r
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// Steady-state SH of one probe (dynamic term excluded).
pub fn probe_sh(p: &EnvProbe, env: &ProbeEnv, sky: &[[f32; 3]; 3]) -> Sh9 {
    let r = &p.state_a;
    let ao_g = p.ao_global;
    let f_static = mul3(env.bounce_static, p.bounce_static.map(|v| v * ao_g));
    let f_sun = mul3(
        mul3(env.bounce_sun, p.bounce_sun.map(|v| v * ao_g)),
        env.sun_color,
    );
    let f_sky = p.bounce_sky.map(|v| v * p.ao_sky);
    let zh = sky.map(|z| mul3(f_sky, z));
    // sub_140291430: base SH (+ occluders) plus factor x the `*_Initial` multi-bounce SH.
    let (keep, m) = match env.multi_bounce {
        m if m.abs() <= 0.001 => (1.0, 0.0),
        m if m < 0.0 => (0.0, -m),
        m => (1.0, m),
    };
    let add = |base: Sh9, init: &Sh9| -> Sh9 {
        std::array::from_fn(|i| std::array::from_fn(|k| keep * base[i][k] + m * init[i][k]))
    };
    let sh_static = add(r.sh_static, &r.sh_initial[0]);
    let sh_sun = add(r.sh_static_sun_factor_filtered, &r.sh_initial[1]);
    let sky_occ = add(
        r.sh_sky_occlusion_filtered.map(|v| [v; 3]),
        &r.sh_initial[2],
    );
    let occluded_sky = zh_sh_product_sky(&zh, &sky_occ);
    let mut out = [[0f32; 3]; 9];
    for i in 0..9 {
        for k in 0..3 {
            out[i][k] =
                f_static[k] * sh_static[i][k] + f_sun[k] * sh_sun[i][k] + occluded_sky[i][k];
        }
    }
    // "HORRIBLE PER-PROBE AMBIENT TERM" (accumulatesh.decl), parmsFloat.y = 1.
    for k in 0..3 {
        out[0][k] += 3.544_907_7 * p.ambient_factor[k] * env.ambient[k];
    }
    out
}

/// Tetrahedron lookup for points (per-object probe SH): uniform grid over the
/// tetrahedra's bounding boxes; barycentrics as `ComputeTetrahedronBarycentrics`.
pub struct TetraLocator {
    min: [f32; 3],
    cell: f32,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

/// Barycentrics of `p` in tetrahedron `t` (shader convention: weights for P0..P2, P3 = rest).
pub fn barycentrics(net: &ProbeNetwork, t: usize, p: [f32; 3]) -> [f32; 4] {
    let tet = &net.tetrahedra[t];
    let v = tet.vertices.map(|i| net.vertices[i as usize].position);
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let dc = |a, b, c| dot(a, cross(b, c));
    let d = sub(p, v[3]);
    let (a0, a1, a2) = (sub(v[0], v[3]), sub(v[1], v[3]), sub(v[2], v[3]));
    let det = dc(a0, a1, a2);
    let r = if det.abs() > 0.0 { 1.0 / det } else { 0.0 };
    let b0 = r * dc(d, a1, a2);
    let b1 = r * dc(d, a2, a0);
    let b2 = r * dc(d, a0, a1);
    [b0, b1, b2, 1.0 - b0 - b1 - b2]
}

impl TetraLocator {
    pub fn new(net: &ProbeNetwork, cell: f32) -> Self {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &net.vertices {
            for k in 0..3 {
                min[k] = min[k].min(v.position[k]);
                max[k] = max[k].max(v.position[k]);
            }
        }
        let dims: [usize; 3] =
            std::array::from_fn(|k| (((max[k] - min[k]) / cell).ceil() as usize).clamp(1, 512));
        let mut cells = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        let idx = |c: [usize; 3]| (c[2] * dims[1] + c[1]) * dims[0] + c[0];
        let cell_of = |p: f32, k: usize| (((p - min[k]) / cell) as usize).min(dims[k] - 1);
        for (t, tet) in net.tetrahedra.iter().enumerate() {
            let ps = tet.vertices.map(|i| net.vertices[i as usize].position);
            let lo: [usize; 3] = std::array::from_fn(|k| {
                cell_of(ps.iter().map(|p| p[k]).fold(f32::MAX, f32::min), k)
            });
            let hi: [usize; 3] = std::array::from_fn(|k| {
                cell_of(ps.iter().map(|p| p[k]).fold(f32::MIN, f32::max), k)
            });
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        cells[idx([x, y, z])].push(t as u32);
                    }
                }
            }
        }
        Self {
            min,
            cell,
            dims,
            cells,
        }
    }

    /// Tetrahedron containing `p` (all barycentrics >= -eps) or, failing that, the one
    /// with the largest minimum barycentric among the cell's candidates.
    pub fn locate(&self, net: &ProbeNetwork, p: [f32; 3]) -> Option<(u32, [f32; 4])> {
        let c: [usize; 3] = std::array::from_fn(|k| {
            (((p[k] - self.min[k]) / self.cell).max(0.0) as usize).min(self.dims[k] - 1)
        });
        let cand = &self.cells[(c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]];
        let mut best: Option<(u32, [f32; 4], f32)> = None;
        for &t in cand {
            let b = barycentrics(net, t as usize, p);
            let m = b.iter().copied().fold(f32::MAX, f32::min);
            if m >= -1e-4 {
                return Some((t, b));
            }
            if best.is_none_or(|(_, _, bm)| m > bm) {
                best = Some((t, b, m));
            }
        }
        best.map(|(t, b, _)| (t, b))
    }
}

/// Probe weights for barycentrics in tetrahedron `t` (`InterpolateSH`: per-vertex u8
/// weights blended by the barycentrics) and the 4 probe indices.
pub fn probe_weights(net: &ProbeNetwork, t: usize, bary: [f32; 4]) -> ([u16; 4], [f32; 4]) {
    let tet = &net.tetrahedra[t];
    let mut w = [0f32; 4];
    for (k, &vi) in tet.vertices.iter().enumerate() {
        let vw = net.vertices[vi as usize].weights;
        for j in 0..4 {
            w[j] += bary[k] * f32::from(vw[j]) / 255.0;
        }
    }
    (tet.probes, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sky_zh_has_no_l2_and_expected_l0() {
        // Uniform colour c: l2 integrates P2(cos) over equal-area rings of the upper
        // hemisphere -> ~0; l0 = c * 0.5/sqrt(pi) * (4 pi / 40) * 40.
        let z = sky_zh([1.0; 3], [1.0; 3]);
        let l0 = 0.5 / std::f32::consts::PI.sqrt() * 4.0 * std::f32::consts::PI;
        assert!((z[0][0] - l0).abs() < 1e-3, "{z:?}");
        assert!(z[2][0].abs() < 1e-3, "{z:?}");
        assert_eq!(z[1], [0.0; 3]);
    }

    #[test]
    fn zh_product_with_unit_dc_scales_only_band_0_and_6() {
        let a = [[2.0; 3], [0.0; 3], [0.0; 3]];
        let mut b = [[0f32; 3]; 9];
        b[0] = [1.0; 3];
        let r = zh_sh_product_sky(&a, &b);
        assert!((r[0][0] - 0.282_094_8 * 2.0).abs() < 1e-6);
        assert!(r[1..].iter().all(|c| c[0] == 0.0));
    }
}
