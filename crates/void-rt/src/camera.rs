//! Camera conventions shared with the engine captures.
//!
//! - `viewaxis` rows are forward, left, up (idTech `idMat3`), world Z-up (O: F3 `view.tsv`,
//!   inject log `axis0`).
//! - `getviewpos`-style poses: yaw/pitch in degrees, pitch positive = looking down; forward
//!   `(cp cy, cp sy, -sp)`, left `(-sy, cy, 0)`, up `(sp cy, sp sy, cp)` (O: checkpoint
//!   9282529 `r3a_offline.rs`, matches the F4 pose/axis pair of RX010).
//! - Projection is Hor+: the vertical field of view is fixed (`fov_y`), the horizontal
//!   one follows the render aspect (O: RX010 3440x1440 dump vs `proj 0.75,1.7778`).
//! - Game depth (`decompressedDepth` dumps, `r32f`): reversed infinite Z, `d = znear / z`
//!   with `z` the distance along the forward axis; `d = 0` is sky (O: E044 comparisons).

/// Camera position and orientation in world space (Z-up, metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub org: [f32; 3],
    /// Rows: forward, left, up.
    pub axis: [[f32; 3]; 3],
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

/// Per-resolution ray basis: pixel centre `(u, v)` maps to
/// `fwd + ndc_x * tan_x * right + ndc_y * tan_y * up` with `ndc` in [-1, 1], y up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayBasis {
    pub org: [f32; 3],
    pub fwd: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub tan_x: f32,
    pub tan_y: f32,
    pub width: u32,
    pub height: u32,
}

impl Camera {
    /// Camera from an idTech pose (degrees).
    pub fn from_pose(org: [f32; 3], yaw_deg: f32, pitch_deg: f32, fov_y: f32) -> Self {
        let (sy, cy) = yaw_deg.to_radians().sin_cos();
        let (sp, cp) = pitch_deg.to_radians().sin_cos();
        Self {
            org,
            axis: [
                [cp * cy, cp * sy, -sp],
                [-sy, cy, 0.0],
                [sp * cy, sp * sy, cp],
            ],
            fov_y,
        }
    }

    pub fn basis(&self, width: u32, height: u32) -> RayBasis {
        let tan_y = (self.fov_y * 0.5).tan();
        let [fwd, left, up] = self.axis;
        RayBasis {
            org: self.org,
            fwd,
            right: left.map(|v| -v),
            up,
            tan_x: tan_y * width as f32 / height as f32,
            tan_y,
            width,
            height,
        }
    }
}

impl RayBasis {
    /// World-space direction through the pixel position `(px, py)` (pixel centres at +0.5),
    /// scaled so that its forward component is 1 (view depth = parameter).
    pub fn direction(&self, px: f32, py: f32) -> [f32; 3] {
        let nx = 2.0 * px / self.width as f32 - 1.0;
        let ny = 1.0 - 2.0 * py / self.height as f32;
        std::array::from_fn(|k| {
            self.fwd[k] + nx * self.tan_x * self.right[k] + ny * self.tan_y * self.up[k]
        })
    }

    /// Projects a world position to pixel coordinates and view depth (None behind the camera).
    pub fn project(&self, p: [f32; 3]) -> Option<([f32; 2], f32)> {
        let d: [f32; 3] = std::array::from_fn(|k| p[k] - self.org[k]);
        let dot = |a: [f32; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
        let z = dot(self.fwd);
        if z <= 1e-4 {
            return None;
        }
        let nx = dot(self.right) / (z * self.tan_x);
        let ny = dot(self.up) / (z * self.tan_y);
        Some((
            [
                (nx + 1.0) * 0.5 * self.width as f32,
                (1.0 - ny) * 0.5 * self.height as f32,
            ],
            z,
        ))
    }

    /// Nearest resample of a same-camera reference image of `(sw, sh)` pixels (Hor+:
    /// equal vertical field of view) to this basis' pixels; `T::default()` outside.
    pub fn resample<T: Copy + Default>(&self, src: &[T], (sw, sh): (u32, u32)) -> Vec<T> {
        let src_tan_x = self.tan_y * sw as f32 / sh as f32;
        let mut out = Vec::with_capacity((self.width * self.height) as usize);
        for v in 0..self.height {
            let ny = 1.0 - 2.0 * (v as f32 + 0.5) / self.height as f32;
            let sy = ((1.0 - ny) * 0.5 * sh as f32) as i64;
            for u in 0..self.width {
                let nx =
                    (2.0 * (u as f32 + 0.5) / self.width as f32 - 1.0) * self.tan_x / src_tan_x;
                let sx = ((nx + 1.0) * 0.5 * sw as f32) as i64;
                let inside = (0..i64::from(sw)).contains(&sx) && (0..i64::from(sh)).contains(&sy);
                out.push(match src.get((sy * i64::from(sw) + sx) as usize) {
                    Some(&p) if inside => p,
                    _ => T::default(),
                });
            }
        }
        out
    }
}

/// View depth from a game depth value (`None` for sky).
pub fn game_depth_to_view(d: f32, znear: f32) -> Option<f32> {
    (d > 0.0).then(|| znear / d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_matches_rx010_axis() {
        // RX010 F4 request: pose -55.201263, 14.707513 -> axis0 (0.552, -0.794, -0.254).
        let c = Camera::from_pose([0.0; 3], -55.201_263, 14.707_513, 1.024_779);
        let f = c.axis[0];
        assert!((f[0] - 0.551_996).abs() < 1e-4, "{f:?}");
        assert!((f[1] + 0.794_256).abs() < 1e-4, "{f:?}");
        assert!((f[2] + 0.253_885).abs() < 1e-4, "{f:?}");
    }

    #[test]
    fn project_inverts_direction() {
        let c = Camera::from_pose([1.0, 2.0, 3.0], 30.0, -10.0, 1.0);
        let b = c.basis(3440, 1440);
        let d = b.direction(1000.5, 200.5);
        let p = std::array::from_fn(|k| b.org[k] + 7.0 * d[k]);
        let (px, z) = b.project(p).unwrap();
        assert!(
            (px[0] - 1000.5).abs() < 1e-2 && (px[1] - 200.5).abs() < 1e-2,
            "{px:?}"
        );
        assert!((z - 7.0).abs() < 1e-4);
    }

    #[test]
    fn hor_plus_projection() {
        // proj y = 1 / tan(fov_y / 2) = 1.7778 for fov_y 1.024779 (RX010 params).
        let b = Camera::from_pose([0.0; 3], 0.0, 0.0, 1.024_779).basis(3440, 1440);
        assert!((1.0 / b.tan_y - 1.777_8).abs() < 1e-3);
        assert!((b.tan_x / b.tan_y - 3440.0 / 1440.0).abs() < 1e-5);
        assert_eq!(game_depth_to_view(0.0, 0.01), None);
        assert_eq!(game_depth_to_view(0.25, 0.5), Some(2.0));
    }
}
