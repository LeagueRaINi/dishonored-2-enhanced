//! Pass classes for targeted GPU timing (profiler `profile_gpu_mode = passes`).
//!
//! Classes come from the pass map of RenderDoc captures of the original renderer
//! (docs/performance-baseline.md, E050): shaders are classified once at creation from
//! their RDEF binding names; render-target state is classified at OMSetRenderTargets.
//! The profiler writes a GPU timestamp only when the class of the current work
//! changes (and at command-list boundaries), so a frame costs ~100-150 timestamps.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Timed pass classes. `ListSetup` is in-list work before the first classified
/// draw/dispatch (state, clears, copies); `BetweenLists` is everything between command
/// lists (immediate-context work and GPU idle while waiting for submission).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Class {
    Shadow,
    DepthPrepass,
    OceanVelocity,
    Opaque,
    Ocean,
    Water,
    Fog,
    Particles,
    Taa,
    Post,
    OtherDraw,
    OtherCompute,
    /// Our AO in place of HBAO+ (`ao.rs`, shaders binding `void_ao_*`).
    Ao,
    /// Our sun contact shadows (`contact.rs`, shaders binding `void_cs_*`).
    Contact,
    /// Our screen-space reflections (`ssr.rs`, shaders binding `void_ssr_*`).
    Ssr,
    ListSetup,
    BetweenLists,
}

pub const CLASS_NAMES: [&str; 17] = [
    "shadow",
    "depth_prepass",
    "ocean_velocity",
    "opaque_lit",
    "ocean",
    "water",
    "fog",
    "particles",
    "taa",
    "post",
    "other_draw",
    "other_compute",
    "ao",
    "contact_shadows",
    "ssr",
    "list_setup",
    "between_lists",
];

// Shader flags (from binding names).
pub const F_OCEAN_PREV: u32 = 1;
pub const F_OCEAN: u32 = 2;
pub const F_LIT: u32 = 4;
pub const F_FX: u32 = 8;
pub const F_WATER: u32 = 16;
pub const F_ATM: u32 = 32;
pub const F_TAA: u32 = 64;
pub const F_POST: u32 = 128;
pub const F_AO: u32 = 256;
pub const F_CONTACT: u32 = 512;
pub const F_SSR: u32 = 1024;
pub const F_KNOWN: u32 = 1 << 31;

/// Flags of a shader from its binding description ("t0:name u0:name ...").
pub fn flags(bindings: &str) -> u32 {
    let has = |s: &str| bindings.contains(s);
    let mut f = F_KNOWN;
    if has("fftwaves_previousfftdisplacement") {
        f |= F_OCEAN_PREV;
    }
    if has("fftwaves_fftdisplacement") || has("ro_ocean_") {
        f |= F_OCEAN;
    }
    if has("tlf_ll_heads") || has("texpreintegratedbrdf") {
        f |= F_LIT;
    }
    if has("ro_fx_") {
        f |= F_FX;
    }
    if has("ro_water_") {
        f |= F_WATER;
    }
    if has("ro_atm_") {
        f |= F_ATM;
    }
    if has("taahistory") || has("taaresult") {
        f |= F_TAA;
    }
    if has("postfx_")
        || has("tonemapping")
        || has("ro_identity_")
        || has("viewcolormap")
        || has("motionvectors")
    {
        f |= F_POST;
    }
    if has("void_ao_") {
        f |= F_AO;
    }
    if has("void_cs_") {
        f |= F_CONTACT;
    }
    if has("void_ssr_") {
        f |= F_SSR;
    }
    f
}

static FLAGS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
/// Bumped when a shader is created (invalidates the per-thread caches: a freed
/// shader's address can be reused).
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn note_shader(ptr: usize, bindings: &str) {
    FLAGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .insert(ptr, flags(bindings));
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

thread_local! {
    static CACHE: RefCell<(u64, HashMap<usize, u32>)> = RefCell::new((u64::MAX, HashMap::new()));
}

fn lookup(ptr: usize) -> u32 {
    if ptr == 0 {
        return 0;
    }
    let generation = GENERATION.load(Ordering::Relaxed);
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != generation {
            c.0 = generation;
            c.1.clear();
        }
        *c.1.entry(ptr).or_insert_with(|| {
            FLAGS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .and_then(|m| m.get(&ptr).copied())
                .unwrap_or(0)
        })
    })
}

/// Render-target state relevant to classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Targets {
    /// DXGI format of RT0 (0 = none).
    pub rt0_format: u32,
    /// Depth target is square (shadow atlas / cube faces; the screen is not square).
    pub ds_square: bool,
    pub has_ds: bool,
}

thread_local! {
    static VS: Cell<u32> = const { Cell::new(0) };
    static PS: Cell<u32> = const { Cell::new(0) };
    static PS_BOUND: Cell<bool> = const { Cell::new(false) };
    static CS: Cell<u32> = const { Cell::new(0) };
    static RT: Cell<Targets> = const { Cell::new(Targets { rt0_format: 0, ds_square: false, has_ds: false }) };
    static LAST: Cell<Option<Class>> = const { Cell::new(None) };
}

pub fn set_vs(ptr: usize) {
    VS.with(|c| c.set(lookup(ptr)));
}
pub fn set_ps(ptr: usize) {
    PS.with(|c| c.set(lookup(ptr)));
    PS_BOUND.with(|c| c.set(ptr != 0));
}
pub fn set_cs(ptr: usize) {
    CS.with(|c| c.set(lookup(ptr)));
}
pub fn set_targets(t: Targets) {
    RT.with(|c| c.set(t));
}

const R16G16_FLOAT: u32 = 34;

pub fn classify_draw() -> Class {
    let (vs, ps, ps_bound, rt) = (
        VS.with(Cell::get),
        PS.with(Cell::get),
        PS_BOUND.with(Cell::get),
        RT.with(Cell::get),
    );
    if ps & F_AO != 0 {
        return Class::Ao;
    }
    if ps & F_CONTACT != 0 {
        return Class::Contact;
    }
    if ps & F_SSR != 0 {
        return Class::Ssr;
    }
    if vs & F_OCEAN_PREV != 0 {
        return Class::OceanVelocity;
    }
    if vs & F_OCEAN != 0 || ps & F_OCEAN != 0 {
        return Class::Ocean;
    }
    if rt.rt0_format == 0 && rt.has_ds && rt.ds_square {
        return Class::Shadow;
    }
    if rt.rt0_format == R16G16_FLOAT {
        return Class::DepthPrepass;
    }
    if ps & F_FX != 0 {
        return Class::Particles;
    }
    if ps & F_WATER != 0 {
        return Class::Water;
    }
    if ps & F_ATM != 0 {
        return Class::Fog;
    }
    if ps & F_LIT != 0 {
        return Class::Opaque;
    }
    if ps & F_POST != 0 || (!ps_bound && rt.rt0_format != 0) {
        return Class::Post;
    }
    Class::OtherDraw
}

pub fn classify_dispatch() -> Class {
    let cs = CS.with(Cell::get);
    if cs & F_AO != 0 {
        Class::Ao
    } else if cs & F_CONTACT != 0 {
        Class::Contact
    } else if cs & F_SSR != 0 {
        Class::Ssr
    } else if cs & F_TAA != 0 {
        Class::Taa
    } else if cs & F_OCEAN != 0 {
        Class::Ocean
    } else if cs & F_ATM != 0 {
        Class::Fog
    } else if cs & F_POST != 0 {
        Class::Post
    } else {
        Class::OtherCompute
    }
}

/// Returns `Some(class)` when `class` differs from the last marked class on this thread.
pub fn changed(class: Class) -> Option<Class> {
    LAST.with(|l| {
        if l.get() == Some(class) {
            None
        } else {
            l.set(Some(class));
            Some(class)
        }
    })
}

/// Forces the next classified work to be marked (after a list boundary mark).
pub fn reset_last(to: Option<Class>) {
    LAST.with(|l| l.set(to));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_rules_match_capture_shaders() {
        // Binding sets observed in the RenderDoc capture of frame 16159 (E050).
        let ocean_vel = flags(
            "t0:ro_fftwaves_previousfftdisplacement t1:ro_fftwaves_fftdisplacement t2:ro_noise_perm2d",
        );
        assert!(ocean_vel & F_OCEAN_PREV != 0);
        let lit = flags("t0:ro_texpreintegratedbrdf_ward t3:ro_tlf_ll_heads t4:ro_bumpmap");
        assert!(lit & F_LIT != 0 && lit & F_FX == 0);
        let fx = flags(
            "t0:ro_texpoolarealights t1:ro_fx_depthfull t3:ro_tlf_ll_heads t6:ro_fx_flipbook",
        );
        assert!(fx & F_FX != 0);
        let taa = flags("t0:ro_taahistory_read t1:ro_motionvectors u1:rw_taaresult");
        assert!(taa & F_TAA != 0);
        let ao = flags("t1:void_ao_lin t2:void_ao_input u0:void_ao_out");
        assert!(ao & F_AO != 0 && ao & F_POST == 0);
        let cs = flags("t0:void_cs_depth t2:void_cs_input u0:void_cs_out");
        assert!(cs & F_CONTACT != 0 && cs & F_AO == 0);
        let ssr = flags("t0:void_ssr_depth t3:void_ssr_input u1:void_ssr_out");
        assert!(ssr & F_SSR != 0 && ssr & (F_AO | F_CONTACT | F_POST) == 0);
    }

    #[test]
    fn draw_classes() {
        set_vs(0);
        set_ps(0);
        set_targets(Targets {
            rt0_format: 0,
            ds_square: true,
            has_ds: true,
        });
        assert_eq!(classify_draw(), Class::Shadow);
        set_targets(Targets {
            rt0_format: R16G16_FLOAT,
            ds_square: false,
            has_ds: true,
        });
        assert_eq!(classify_draw(), Class::DepthPrepass);
        reset_last(None);
        assert_eq!(changed(Class::Opaque), Some(Class::Opaque));
        assert_eq!(changed(Class::Opaque), None);
    }
}
