//! Comparisons over `void_rt::dynamic::InstanceProbe` images shared by the dynamic-object
//! checks (`dynamic_check`, `skin_check`).

pub use void_rt::dynamic::{InstanceProbe as RayProbe, PROBE_MISS as MISS, ProbeImage as Image};

pub trait ImageExt {
    /// Pixels whose instance or depth (relative 1e-4) differ.
    fn differing(&self, other: &Image) -> usize;
    /// Pixels whose hit depth differs (relative 1e-3, or hit vs miss). Coplanar duplicate
    /// surfaces (LOD copies, decals) tie in depth; which instance wins a tie depends on the
    /// BVH layout on hardware traversal, so a rebuilt or refitted TLAS may swap their ids.
    fn depth_differing(&self, other: &Image) -> usize;
    fn count(&self, inst: u32) -> usize;
}

impl ImageExt for Image {
    fn differing(&self, other: &Image) -> usize {
        self.id
            .iter()
            .zip(&other.id)
            .zip(self.depth.iter().zip(&other.depth))
            .filter(|((x, y), (dx, dy))| x != y || (*dx - *dy).abs() > 1e-4 * dx.abs().max(1e-6))
            .count()
    }

    fn depth_differing(&self, other: &Image) -> usize {
        self.depth
            .iter()
            .zip(&other.depth)
            .filter(|(x, y)| (**x - **y).abs() > 1e-3 * x.abs().max(y.abs()).max(1e-9))
            .count()
    }

    fn count(&self, inst: u32) -> usize {
        self.id.iter().filter(|&&i| i == inst).count()
    }
}
