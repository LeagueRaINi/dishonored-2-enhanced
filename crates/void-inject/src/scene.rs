//! Read-only view of the engine's per-frame view requests (the `RenderAllViews`
//! argument): camera pose and map name for the benchmark, per-view counts for the
//! frame log. All reads are checked; any unreadable pointer yields `None` for that item
//! instead of faulting.

use crate::mem;

/// Constant layouts for exe `b671f824...6180` (from `void-reflect` and the IDB).
mod layout {
    /// `RenderAllViews` argument 1: view request list (`idList`).
    pub const REQUESTS_PTR: u64 = 0;
    pub const REQUESTS_COUNT: u64 = 8;
    pub const REQUEST_STRIDE: u64 = 15_024;
    pub const REQUEST_WORLD: u64 = 32;
    pub const REQUEST_RENDER_VIEW: u64 = 48;
    /// `renderView_t` members.
    pub const RV_FOV_X: u64 = 156;
    pub const RV_VIEWORG: u64 = 168;
    pub const RV_VIEWAXIS: u64 = 204;
    /// `idRenderWorldLocal` members (from the index accessors).
    /// Map name `idStr` (data pointer first), assigned by `InitFromMap` (`sub_14023A5D0`).
    pub const WORLD_MAP_NAME: u64 = 132_328;
    pub const WORLD_MODELS_COUNT: u64 = 132_416;
    pub const WORLD_LIGHTS_COUNT: u64 = 516_472;
    /// `idList` counts occupy the low 24 bits.
    pub const COUNT_MASK: u32 = 0x00FF_FFFF;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewSummary {
    pub world: u64,
    pub vieworg: [f32; 3],
    pub viewaxis: [f32; 9],
    pub fov: [f32; 2],
    pub models: u32,
    pub lights: u32,
}

/// Summarizes the view requests passed to `RenderAllViews`.
pub fn summarize(requests: u64) -> Vec<ViewSummary> {
    use layout::*;
    let Some(list) = mem::u64_at(requests + REQUESTS_PTR) else {
        return Vec::new();
    };
    let count = mem::u32_at(requests + REQUESTS_COUNT).map_or(0, |c| c & COUNT_MASK);
    (0..count.min(16) as u64)
        .filter_map(|i| {
            let r = list + i * REQUEST_STRIDE;
            let world = mem::u64_at(r + REQUEST_WORLD).unwrap_or(0);
            let rv = r + REQUEST_RENDER_VIEW;
            let count_at = |a: u64| mem::u32_at(a).map_or(0, |c| c & COUNT_MASK);
            Some(ViewSummary {
                world,
                vieworg: mem::f32s::<3>(rv + RV_VIEWORG)?,
                viewaxis: mem::f32s::<9>(rv + RV_VIEWAXIS)?,
                fov: mem::f32s::<2>(rv + RV_FOV_X)?,
                models: if world != 0 {
                    count_at(world + WORLD_MODELS_COUNT)
                } else {
                    0
                },
                lights: if world != 0 {
                    count_at(world + WORLD_LIGHTS_COUNT)
                } else {
                    0
                },
            })
        })
        .collect()
}

/// Camera of the first view and the map name: (map, vieworg, forward axis).
pub fn read_pose(requests: u64) -> Option<(String, [f32; 3], [f32; 3])> {
    use layout::*;
    let list = mem::u64_at(requests + REQUESTS_PTR)?;
    if mem::u32_at(requests + REQUESTS_COUNT)? & COUNT_MASK == 0 {
        return None;
    }
    let world = mem::u64_at(list + REQUEST_WORLD)?;
    let rv = list + REQUEST_RENDER_VIEW;
    let org = mem::f32s::<3>(rv + RV_VIEWORG)?;
    let fwd = mem::f32s::<3>(rv + RV_VIEWAXIS)?;
    let map = read_c_string(mem::u64_at(world + WORLD_MAP_NAME).unwrap_or(0));
    Some((map, org, fwd))
}

fn read_c_string(ptr: u64) -> String {
    if ptr == 0 {
        return String::new();
    }
    for n in [256usize, 128, 64, 32] {
        let mut buf = vec![0u8; n];
        if mem::read_into(ptr, &mut buf) {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(n);
            return String::from_utf8_lossy(&buf[..end]).into_owned();
        }
    }
    String::new()
}
