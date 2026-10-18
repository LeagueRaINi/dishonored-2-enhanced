//! One-frame D3D11 trace (Shift+F3 with `profile = 1`): the pass map RenderDoc would
//! give, recorded live. The game blocks capture tools (`sub_14036A530`: exits on
//! GPUPerfServer / Nsight.Monitor processes or a loaded renderdoc.dll, and calls
//! `D3DPERF_SetOptions(1)`); we do not circumvent that check.
//!
//! On AMD the D3D11 runtime replays deferred command lists through the immediate
//! context's methods (d2enhanced NOTES), so hooking the immediate implementations gives
//! the frame's work in GPU submission order. For each draw / dispatch / clear / copy /
//! render-target change during the traced frame we record arguments, bound shaders
//! (named by their RDEF bindings, captured when the game creates them), render-target
//! formats and sizes, and a GPU timestamp issued before the call. Timestamps are read
//! back asynchronously (`DONOTFLUSH`) in later frames and the trace is discarded if the
//! frame was disjoint. A per-event timestamp is approximate (GPU pipelining) and the
//! queries themselves slightly perturb the traced frame; use traces for attribution,
//! not as the measurement of record.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::{HRESULT, Interface};

use crate::{dxbc, log, profiler};

// ------------------------------------------------------------------- shader table --

#[derive(Clone, Debug)]
pub struct ShaderInfo {
    pub stage: char,
    pub id: String,
    pub bindings: String,
}

static SHADERS: Mutex<Option<HashMap<usize, ShaderInfo>>> = Mutex::new(None);
pub static SHADER_COUNT: AtomicU64 = AtomicU64::new(0);

fn note_shader(stage: char, code: *const c_void, len: usize, out: *mut *mut c_void) {
    if code.is_null() || out.is_null() || len == 0 || len > 16 << 20 {
        return;
    }
    // SAFETY: the game passed `len` bytes of bytecode; `out` was filled on success.
    let (bytes, ptr) = unsafe { (std::slice::from_raw_parts(code.cast::<u8>(), len), *out) };
    if ptr.is_null() {
        return;
    }
    let info = ShaderInfo {
        stage,
        id: dxbc::id(bytes).unwrap_or_default(),
        bindings: dxbc::describe(&dxbc::bindings(bytes)),
    };
    crate::passes::note_shader(ptr as usize, &info.bindings);
    SHADERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .insert(ptr as usize, info);
    SHADER_COUNT.fetch_add(1, Ordering::Relaxed);
}

type CreateShaderFn = unsafe extern "system" fn(
    *mut c_void,
    *const c_void,
    usize,
    *mut c_void,
    *mut *mut c_void,
) -> HRESULT;
static CREATE_ORIG: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
const STAGES: [char; 3] = ['v', 'p', 'c'];

unsafe extern "system" fn create_shader<const I: usize>(
    this: *mut c_void,
    code: *const c_void,
    len: usize,
    linkage: *mut c_void,
    out: *mut *mut c_void,
) -> HRESULT {
    // SAFETY: MinHook trampoline with this signature.
    let f = unsafe {
        std::mem::transmute::<usize, CreateShaderFn>(CREATE_ORIG[I].load(Ordering::Relaxed))
    };
    // SAFETY: arguments passed through unchanged (lit pixel shaders get the material output
    // when `ssr.rs` patching is on; the original bytecode on any problem).
    let hr = unsafe {
        if I == 1 && crate::ssr::material::patching() {
            crate::ssr::material::create_pixel_shader(f, this, code, len, linkage, out)
        } else {
            f(this, code, len, linkage, out)
        }
    };
    if hr.is_ok() && crate::settings::get().profile {
        let _ = std::panic::catch_unwind(|| note_shader(STAGES[I], code, len, out));
    }
    hr
}

type CreateSamplerFn =
    unsafe extern "system" fn(*mut c_void, *const D3D11_SAMPLER_DESC, *mut *mut c_void) -> HRESULT;
static SAMPLER_ORIG: AtomicUsize = AtomicUsize::new(0);
/// Distinct (Filter, MaxAnisotropy) pairs of created samplers and their call counts.
static SAMPLERS: Mutex<Option<HashMap<(i32, u32), u32>>> = Mutex::new(None);

/// Sampler census: logs each new (Filter, MaxAnisotropy) pair once, so the effect of
/// `image_anisotropy` (anisotropic filter 0x55, LO4) is visible in the log.
unsafe extern "system" fn create_sampler(
    this: *mut c_void,
    desc: *const D3D11_SAMPLER_DESC,
    out: *mut *mut c_void,
) -> HRESULT {
    // SAFETY: MinHook trampoline with this signature.
    let f = unsafe {
        std::mem::transmute::<usize, CreateSamplerFn>(SAMPLER_ORIG.load(Ordering::Relaxed))
    };
    // SAFETY: arguments passed through unchanged.
    let hr = unsafe { f(this, desc, out) };
    if desc.is_null() {
        return hr;
    }
    // SAFETY: the game passed a valid descriptor.
    let key = unsafe { ((*desc).Filter.0, (*desc).MaxAnisotropy) };
    let _ = std::panic::catch_unwind(|| {
        let mut g = SAMPLERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let map = g.get_or_insert_with(HashMap::new);
        let n = map.entry(key).or_insert(0);
        *n += 1;
        if *n == 1 {
            log::line(&format!(
                "sampler: new filter 0x{:x} max anisotropy {} ({} distinct)",
                key.0,
                key.1,
                map.len()
            ));
        }
    });
    hr
}

// ------------------------------------------------------------------------ events --

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    Draw,
    DrawIndexed,
    DrawInstanced,
    DrawIndexedInstanced,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    DispatchIndirect,
    SetTargets,
    ClearRtv,
    ClearDsv,
    ClearUav,
    Copy,
    CopyRegion,
    Update,
    Resolve,
    GenMips,
}

struct Ev {
    kind: Kind,
    args: [u32; 4],
    shaders: [usize; 3],
    /// Interned description of the bound targets (SetTargets) or of the resources
    /// involved (clears/copies).
    desc: u32,
    stamp: Option<ID3D11Query>,
}

struct Trace {
    frame: u64,
    disjoint: Option<ID3D11Query>,
    events: Vec<Ev>,
    strings: Vec<String>,
    string_ix: HashMap<String, u32>,
    end: Option<ID3D11Query>,
    started: bool,
    finished: bool,
    age: u32,
}

// SAFETY: D3D11 objects used under the mutex on the immediate-context thread.
unsafe impl Send for Trace {}

static TRACING: AtomicBool = AtomicBool::new(false);
static TRACE: Mutex<Option<Trace>> = Mutex::new(None);
const MAX_EVENTS: usize = 40_000;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Arms a trace of the next engine frame.
pub fn arm() -> bool {
    let mut t = TRACE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if t.is_some() {
        return false;
    }
    *t = Some(Trace {
        frame: profiler::current_frame() + 2,
        disjoint: None,
        events: Vec::new(),
        strings: Vec::new(),
        string_ix: HashMap::new(),
        end: None,
        started: false,
        finished: false,
        age: 0,
    });
    TRACING.store(true, Ordering::Relaxed);
    true
}

fn intern(t: &mut Trace, s: String) -> u32 {
    if let Some(i) = t.string_ix.get(&s) {
        return *i;
    }
    let i = t.strings.len() as u32;
    t.strings.push(s.clone());
    t.string_ix.insert(s, i);
    i
}

fn fmt_name(f: DXGI_FORMAT) -> String {
    format!("{}", f.0)
}

fn res_desc(r: &ID3D11Resource) -> String {
    if let Ok(t) = r.cast::<ID3D11Texture2D>() {
        let mut d = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: valid out pointer.
        unsafe { t.GetDesc(&mut d) };
        return format!(
            "tex2d {}x{} f{} m{} a{} s{}",
            d.Width,
            d.Height,
            fmt_name(d.Format),
            d.MipLevels,
            d.ArraySize,
            d.SampleDesc.Count
        );
    }
    if let Ok(b) = r.cast::<ID3D11Buffer>() {
        let mut d = D3D11_BUFFER_DESC::default();
        // SAFETY: valid out pointer.
        unsafe { b.GetDesc(&mut d) };
        return format!("buf {}B", d.ByteWidth);
    }
    if let Ok(t) = r.cast::<ID3D11Texture3D>() {
        let mut d = D3D11_TEXTURE3D_DESC::default();
        // SAFETY: valid out pointer.
        unsafe { t.GetDesc(&mut d) };
        return format!(
            "tex3d {}x{}x{} f{}",
            d.Width,
            d.Height,
            d.Depth,
            fmt_name(d.Format)
        );
    }
    "res".into()
}

fn view_desc(v: &ID3D11View) -> String {
    // SAFETY: plain getter on a live view.
    unsafe { v.GetResource() }.map_or_else(|_| "null".into(), |r| res_desc(&r))
}

/// Handles the trace frame window on the immediate-context thread; returns the open
/// trace if this call should be recorded.
fn begin_event(
    this: *mut c_void,
) -> Option<(
    std::sync::MutexGuard<'static, Option<Trace>>,
    ID3D11DeviceContext,
)> {
    // SAFETY: `this` is the context the game called.
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&this) }?.clone();
    // SAFETY: plain getter.
    if unsafe { ctx.GetType() } != D3D11_DEVICE_CONTEXT_IMMEDIATE {
        return None;
    }
    let mut g = TRACE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let t = g.as_mut()?;
    let frame = profiler::current_frame();
    if t.finished {
        try_resolve(&ctx, &mut g);
        return None;
    }
    if !t.started {
        if frame < t.frame {
            return None;
        }
        // SAFETY: plain getter.
        let device = unsafe { ctx.GetDevice() }.ok()?;
        t.disjoint = profiler::make_query(&device, D3D11_QUERY_TIMESTAMP_DISJOINT);
        if let Some(q) = &t.disjoint {
            // SAFETY: Begin on the immediate context.
            unsafe { ctx.Begin(q) };
        }
        t.frame = frame;
        t.started = true;
    } else if frame != t.frame || t.events.len() >= MAX_EVENTS {
        // End of the traced frame.
        // SAFETY: plain getter.
        if let Ok(device) = unsafe { ctx.GetDevice() } {
            t.end = profiler::make_query(&device, D3D11_QUERY_TIMESTAMP);
            if let Some(q) = &t.end {
                // SAFETY: End on the immediate context.
                unsafe { ctx.End(q) };
            }
        }
        if let Some(q) = &t.disjoint {
            // SAFETY: End on the immediate context.
            unsafe { ctx.End(q) };
        }
        t.finished = true;
        log::line(&format!(
            "trace: frame {} recorded ({} events)",
            t.frame,
            t.events.len()
        ));
        return None;
    }
    Some((g, ctx))
}

fn stamp(ctx: &ID3D11DeviceContext) -> Option<ID3D11Query> {
    // SAFETY: plain getter.
    let device = unsafe { ctx.GetDevice() }.ok()?;
    let q = profiler::make_query(&device, D3D11_QUERY_TIMESTAMP)?;
    // SAFETY: End on the immediate context.
    unsafe { ctx.End(&q) };
    Some(q)
}

fn bound_shaders(ctx: &ID3D11DeviceContext, compute: bool) -> [usize; 3] {
    // SAFETY: getters with valid out pointers; the returned references are released.
    unsafe {
        if compute {
            let mut cs = None;
            ctx.CSGetShader(&mut cs, None, None);
            [0, 0, cs.map_or(0, |s| s.as_raw() as usize)]
        } else {
            let (mut vs, mut ps) = (None, None);
            ctx.VSGetShader(&mut vs, None, None);
            ctx.PSGetShader(&mut ps, None, None);
            [
                vs.map_or(0, |s| s.as_raw() as usize),
                ps.map_or(0, |s| s.as_raw() as usize),
                0,
            ]
        }
    }
}

/// Records one event (called before the original method runs).
fn record(this: *mut c_void, kind: Kind, args: [u32; 4], desc: impl FnOnce() -> String) {
    if DEPTH.with(Cell::get) > 0 {
        return;
    }
    let Some((mut g, ctx)) = begin_event(this) else {
        return;
    };
    let Some(t) = g.as_mut() else { return };
    let compute = matches!(kind, Kind::Dispatch | Kind::DispatchIndirect);
    let draw = matches!(
        kind,
        Kind::Draw
            | Kind::DrawIndexed
            | Kind::DrawInstanced
            | Kind::DrawIndexedInstanced
            | Kind::DrawIndirect
            | Kind::DrawIndexedIndirect
    );
    let shaders = if compute || draw {
        bound_shaders(&ctx, compute)
    } else {
        [0; 3]
    };
    let d = desc();
    let desc = intern(t, d);
    let stamp = stamp(&ctx);
    t.events.push(Ev {
        kind,
        args,
        shaders,
        desc,
        stamp,
    });
}

/// Runs `f` (the original method) with the nesting guard set.
fn guarded<R>(f: impl FnOnce() -> R) -> R {
    DEPTH.with(|d| d.set(d.get() + 1));
    let r = f();
    DEPTH.with(|d| d.set(d.get() - 1));
    r
}

// ----------------------------------------------------------------------- detours --

macro_rules! orig {
    ($i:expr, $t:ty) => {{
        // SAFETY: MinHook trampoline of the method with signature `$t`.
        unsafe { std::mem::transmute::<usize, $t>(ORIG[$i].load(Ordering::Relaxed)) }
    }};
}

const M_DRAW_INDEXED: usize = 0;
const M_DRAW: usize = 1;
const M_DII: usize = 2;
const M_DI: usize = 3;
const M_OMSET: usize = 4;
const M_OMSET_UAV: usize = 5;
const M_DII_IND: usize = 6;
const M_DI_IND: usize = 7;
const M_DISPATCH: usize = 8;
const M_DISPATCH_IND: usize = 9;
const M_COPY_REGION: usize = 10;
const M_COPY: usize = 11;
const M_UPDATE: usize = 12;
const M_CLEAR_RTV: usize = 13;
const M_CLEAR_UAV_U: usize = 14;
const M_CLEAR_UAV_F: usize = 15;
const M_CLEAR_DSV: usize = 16;
const M_GENMIPS: usize = 17;
const M_RESOLVE: usize = 18;
const M_VSSET: usize = 19;
const M_PSSET: usize = 20;
const M_CSSET: usize = 21;
const NM: usize = 22;
/// vtable index per method (ID3D11DeviceContext).
const INDEX: [usize; NM] = [
    12, 13, 20, 21, 33, 34, 39, 40, 41, 42, 46, 47, 48, 50, 51, 52, 53, 54, 57, 11, 9, 69,
];
static ORIG: [AtomicUsize; NM] = [const { AtomicUsize::new(0) }; NM];

type P = *mut c_void;

fn tracing() -> bool {
    TRACING.load(Ordering::Relaxed)
}

/// Pass-class GPU timing (profiler): mark a class change at this draw/dispatch.
fn pass_draw(this: P) {
    if DEPTH.with(Cell::get) == 0
        && profiler::pass_timing()
        && let Some(c) = crate::passes::changed(crate::passes::classify_draw())
    {
        profiler::pass_mark(this, c);
    }
}

fn pass_dispatch(this: P) {
    if DEPTH.with(Cell::get) == 0
        && profiler::pass_timing()
        && let Some(c) = crate::passes::changed(crate::passes::classify_dispatch())
    {
        profiler::pass_mark(this, c);
    }
}

thread_local! {
    /// Depth view -> square (shadow atlas) cache for pass classification.
    static DSV_SQUARE: std::cell::RefCell<HashMap<usize, bool>> = std::cell::RefCell::new(HashMap::new());
}

fn pass_targets(n: u32, rtvs: *const P, dsv: P) {
    if DEPTH.with(Cell::get) > 0 || !profiler::pass_timing() {
        return;
    }
    let mut t = crate::passes::Targets::default();
    if n > 0 && !rtvs.is_null() {
        // SAFETY: the game passed `n` view pointers.
        let p = unsafe { *rtvs };
        // SAFETY: live view pointer or null.
        if let Some(v) = unsafe { ID3D11RenderTargetView::from_raw_borrowed(&p) } {
            let mut d = D3D11_RENDER_TARGET_VIEW_DESC::default();
            // SAFETY: valid out pointer.
            unsafe { v.GetDesc(&mut d) };
            t.rt0_format = d.Format.0 as u32;
        }
    }
    // SAFETY: live view pointer or null.
    if let Some(v) = unsafe { ID3D11View::from_raw_borrowed(&dsv) } {
        t.has_ds = true;
        t.ds_square = DSV_SQUARE.with(|m| {
            *m.borrow_mut().entry(dsv as usize).or_insert_with(|| {
                // SAFETY: plain getters on a live view / texture.
                unsafe { v.GetResource() }
                    .ok()
                    .and_then(|r| r.cast::<ID3D11Texture2D>().ok())
                    .is_some_and(|tex| {
                        let mut d = D3D11_TEXTURE2D_DESC::default();
                        // SAFETY: valid out pointer.
                        unsafe { tex.GetDesc(&mut d) };
                        d.Width == d.Height
                    })
            })
        });
    }
    crate::passes::set_targets(t);
}

unsafe extern "system" fn d_vsset(this: P, sh: P, ci: *const P, n: u32) {
    let f = orig!(M_VSSET, unsafe extern "system" fn(P, P, *const P, u32));
    if DEPTH.with(Cell::get) == 0 && profiler::pass_timing() {
        crate::passes::set_vs(sh as usize);
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, sh, ci, n) })
}

unsafe extern "system" fn d_psset(this: P, sh: P, ci: *const P, n: u32) {
    let f = orig!(M_PSSET, unsafe extern "system" fn(P, P, *const P, u32));
    if DEPTH.with(Cell::get) == 0 && profiler::pass_timing() {
        crate::passes::set_ps(sh as usize);
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, sh, ci, n) })
}

unsafe extern "system" fn d_csset(this: P, sh: P, ci: *const P, n: u32) {
    let f = orig!(M_CSSET, unsafe extern "system" fn(P, P, *const P, u32));
    if DEPTH.with(Cell::get) == 0 && profiler::pass_timing() {
        crate::passes::set_cs(sh as usize);
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, sh, ci, n) })
}

unsafe extern "system" fn d_draw_indexed(this: P, n: u32, start: u32, base: i32) {
    let f = orig!(M_DRAW_INDEXED, unsafe extern "system" fn(P, u32, u32, i32));
    if tracing() {
        record(
            this,
            Kind::DrawIndexed,
            [n, 1, start, base as u32],
            String::new,
        );
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, start, base) })
}

unsafe extern "system" fn d_draw(this: P, n: u32, start: u32) {
    let f = orig!(M_DRAW, unsafe extern "system" fn(P, u32, u32));
    if tracing() {
        record(this, Kind::Draw, [n, 1, start, 0], String::new);
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, start) })
}

unsafe extern "system" fn d_dii(this: P, n: u32, inst: u32, start: u32, base: i32, si: u32) {
    let f = orig!(M_DII, unsafe extern "system" fn(P, u32, u32, u32, i32, u32));
    if tracing() {
        record(
            this,
            Kind::DrawIndexedInstanced,
            [n, inst, start, si],
            String::new,
        );
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, inst, start, base, si) })
}

unsafe extern "system" fn d_di(this: P, n: u32, inst: u32, start: u32, si: u32) {
    let f = orig!(M_DI, unsafe extern "system" fn(P, u32, u32, u32, u32));
    if tracing() {
        record(this, Kind::DrawInstanced, [n, inst, start, si], String::new);
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, inst, start, si) })
}

fn targets_desc(n: u32, rtvs: *const P, dsv: P) -> String {
    let mut parts = Vec::new();
    for i in 0..n.min(8) as usize {
        // SAFETY: the game passed `n` view pointers.
        let p = unsafe { *rtvs.add(i) };
        // SAFETY: live view pointer or null.
        match unsafe { ID3D11View::from_raw_borrowed(&p) } {
            Some(v) => parts.push(format!("rt{i}:{}", view_desc(v))),
            None => parts.push(format!("rt{i}:-")),
        }
    }
    // SAFETY: live view pointer or null.
    if let Some(v) = unsafe { ID3D11View::from_raw_borrowed(&dsv) } {
        parts.push(format!("ds:{}", view_desc(v)));
    }
    parts.join(" | ")
}

unsafe extern "system" fn d_omset(this: P, n: u32, rtvs: *const P, dsv: P) {
    let f = orig!(M_OMSET, unsafe extern "system" fn(P, u32, *const P, P));
    pass_targets(n, rtvs, dsv);
    if tracing() {
        record(this, Kind::SetTargets, [n, 0, 0, 0], || {
            if rtvs.is_null() {
                targets_desc(0, rtvs, dsv)
            } else {
                targets_desc(n, rtvs, dsv)
            }
        });
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, rtvs, dsv) })
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn d_omset_uav(
    this: P,
    n: u32,
    rtvs: *const P,
    dsv: P,
    us: u32,
    nu: u32,
    uavs: *const P,
    counts: *const u32,
) {
    let f = orig!(
        M_OMSET_UAV,
        unsafe extern "system" fn(P, u32, *const P, P, u32, u32, *const P, *const u32)
    );
    if n != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL {
        pass_targets(n, rtvs, dsv);
    }
    if tracing() && n != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL {
        record(this, Kind::SetTargets, [n, nu, 1, 0], || {
            if rtvs.is_null() {
                targets_desc(0, rtvs, dsv)
            } else {
                targets_desc(n, rtvs, dsv)
            }
        });
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, n, rtvs, dsv, us, nu, uavs, counts) })
}

unsafe extern "system" fn d_dii_ind(this: P, buf: P, off: u32) {
    let f = orig!(M_DII_IND, unsafe extern "system" fn(P, P, u32));
    if tracing() {
        record(this, Kind::DrawIndexedIndirect, [off, 0, 0, 0], String::new);
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, buf, off) })
}

unsafe extern "system" fn d_di_ind(this: P, buf: P, off: u32) {
    let f = orig!(M_DI_IND, unsafe extern "system" fn(P, P, u32));
    if tracing() {
        record(this, Kind::DrawIndirect, [off, 0, 0, 0], String::new);
    }
    pass_draw(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, buf, off) })
}

unsafe extern "system" fn d_dispatch(this: P, x: u32, y: u32, z: u32) {
    let f = orig!(M_DISPATCH, unsafe extern "system" fn(P, u32, u32, u32));
    if tracing() {
        record(this, Kind::Dispatch, [x, y, z, 0], String::new);
    }
    pass_dispatch(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, x, y, z) })
}

unsafe extern "system" fn d_dispatch_ind(this: P, buf: P, off: u32) {
    let f = orig!(M_DISPATCH_IND, unsafe extern "system" fn(P, P, u32));
    if tracing() {
        record(this, Kind::DispatchIndirect, [off, 0, 0, 0], String::new);
    }
    pass_dispatch(this);
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, buf, off) })
}

fn res_of(p: P) -> String {
    // SAFETY: live resource pointer or null.
    unsafe { ID3D11Resource::from_raw_borrowed(&p) }.map_or_else(|| "null".into(), res_desc)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn d_copy_region(
    this: P,
    dst: P,
    ds: u32,
    x: u32,
    y: u32,
    z: u32,
    src: P,
    ss: u32,
    bx: *const c_void,
) {
    let f = orig!(
        M_COPY_REGION,
        unsafe extern "system" fn(P, P, u32, u32, u32, u32, P, u32, *const c_void)
    );
    if tracing() {
        record(this, Kind::CopyRegion, [ds, ss, x, y], || {
            format!("{} <- {}", res_of(dst), res_of(src))
        });
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, dst, ds, x, y, z, src, ss, bx) })
}

unsafe extern "system" fn d_copy(this: P, dst: P, src: P) {
    let f = orig!(M_COPY, unsafe extern "system" fn(P, P, P));
    if tracing() {
        record(this, Kind::Copy, [0; 4], || {
            format!("{} <- {}", res_of(dst), res_of(src))
        });
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, dst, src) })
}

unsafe extern "system" fn d_update(
    this: P,
    dst: P,
    sub: u32,
    bx: *const c_void,
    data: *const c_void,
    rp: u32,
    dp: u32,
) {
    let f = orig!(
        M_UPDATE,
        unsafe extern "system" fn(P, P, u32, *const c_void, *const c_void, u32, u32)
    );
    if tracing() {
        record(this, Kind::Update, [sub, rp, dp, 0], || res_of(dst));
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, dst, sub, bx, data, rp, dp) })
}

fn view_of(p: P) -> String {
    // SAFETY: live view pointer or null.
    unsafe { ID3D11View::from_raw_borrowed(&p) }.map_or_else(|| "null".into(), view_desc)
}

unsafe extern "system" fn d_clear_rtv(this: P, v: P, c: *const f32) {
    let f = orig!(M_CLEAR_RTV, unsafe extern "system" fn(P, P, *const f32));
    if tracing() {
        record(this, Kind::ClearRtv, [0; 4], || view_of(v));
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, v, c) })
}

unsafe extern "system" fn d_clear_uav_u(this: P, v: P, c: *const u32) {
    let f = orig!(M_CLEAR_UAV_U, unsafe extern "system" fn(P, P, *const u32));
    if tracing() {
        record(this, Kind::ClearUav, [0; 4], || view_of(v));
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, v, c) })
}

unsafe extern "system" fn d_clear_uav_f(this: P, v: P, c: *const f32) {
    let f = orig!(M_CLEAR_UAV_F, unsafe extern "system" fn(P, P, *const f32));
    if tracing() {
        record(this, Kind::ClearUav, [1, 0, 0, 0], || view_of(v));
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, v, c) })
}

unsafe extern "system" fn d_clear_dsv(this: P, v: P, flags: u32, depth: f32, stencil: u8) {
    let f = orig!(M_CLEAR_DSV, unsafe extern "system" fn(P, P, u32, f32, u8));
    if tracing() {
        record(
            this,
            Kind::ClearDsv,
            [flags, depth.to_bits(), u32::from(stencil), 0],
            || view_of(v),
        );
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, v, flags, depth, stencil) })
}

unsafe extern "system" fn d_genmips(this: P, v: P) {
    let f = orig!(M_GENMIPS, unsafe extern "system" fn(P, P));
    if tracing() {
        record(this, Kind::GenMips, [0; 4], || view_of(v));
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, v) })
}

unsafe extern "system" fn d_resolve(this: P, dst: P, ds: u32, src: P, ss: u32, fmt: DXGI_FORMAT) {
    let f = orig!(
        M_RESOLVE,
        unsafe extern "system" fn(P, P, u32, P, u32, DXGI_FORMAT)
    );
    if tracing() {
        record(this, Kind::Resolve, [ds, ss, fmt.0 as u32, 0], || {
            format!("{} <- {}", res_of(dst), res_of(src))
        });
    }
    // SAFETY: arguments passed through unchanged.
    guarded(|| unsafe { f(this, dst, ds, src, ss, fmt) })
}

fn detours() -> [*mut c_void; NM] {
    [
        d_draw_indexed as *mut c_void,
        d_draw as *mut c_void,
        d_dii as *mut c_void,
        d_di as *mut c_void,
        d_omset as *mut c_void,
        d_omset_uav as *mut c_void,
        d_dii_ind as *mut c_void,
        d_di_ind as *mut c_void,
        d_dispatch as *mut c_void,
        d_dispatch_ind as *mut c_void,
        d_copy_region as *mut c_void,
        d_copy as *mut c_void,
        d_update as *mut c_void,
        d_clear_rtv as *mut c_void,
        d_clear_uav_u as *mut c_void,
        d_clear_uav_f as *mut c_void,
        d_clear_dsv as *mut c_void,
        d_genmips as *mut c_void,
        d_resolve as *mut c_void,
        d_vsset as *mut c_void,
        d_psset as *mut c_void,
        d_csset as *mut c_void,
    ]
}

// ------------------------------------------------------------------- readback --

fn try_resolve(ctx: &ID3D11DeviceContext, g: &mut Option<Trace>) {
    let Some(t) = g.as_mut() else { return };
    t.age += 1;
    let mut dj = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
    let ready = t.disjoint.as_ref().is_some_and(|q| {
        profiler::get_data(
            ctx,
            q,
            (&raw mut dj).cast(),
            std::mem::size_of_val(&dj) as u32,
        )
        .0 == 0
    });
    if !ready {
        if t.age > 120 {
            log::line("trace: GPU timestamps never became available; trace dropped");
            *g = None;
            TRACING.store(false, Ordering::Relaxed);
        }
        return;
    }
    let Some(t) = g.take() else { return };
    TRACING.store(false, Ordering::Relaxed);
    let mut stamps = Vec::with_capacity(t.events.len() + 1);
    for e in &t.events {
        let mut v = 0u64;
        let ok = e
            .stamp
            .as_ref()
            .is_some_and(|q| profiler::get_data(ctx, q, (&raw mut v).cast(), 8).0 == 0);
        stamps.push(ok.then_some(v));
    }
    let mut end = 0u64;
    let end_ok = t
        .end
        .as_ref()
        .is_some_and(|q| profiler::get_data(ctx, q, (&raw mut end).cast(), 8).0 == 0);
    stamps.push(end_ok.then_some(end));
    let disjoint = dj.Disjoint.as_bool();
    let freq = dj.Frequency;
    let rows: Vec<(Kind, [u32; 4], [usize; 3], u32)> = t
        .events
        .iter()
        .map(|e| (e.kind, e.args, e.shaders, e.desc))
        .collect();
    let strings = t.strings.clone();
    let frame = t.frame;
    std::thread::spawn(move || {
        if let Err(e) = write(frame, &rows, &stamps, &strings, freq, disjoint) {
            log::line(&format!("trace: write failed: {e}"));
        }
    });
}

fn write(
    frame: u64,
    rows: &[(Kind, [u32; 4], [usize; 3], u32)],
    stamps: &[Option<u64>],
    strings: &[String],
    freq: u64,
    disjoint: bool,
) -> Result<(), String> {
    use std::fmt::Write as _;
    let dir = crate::game_dir()
        .ok_or("no game dir")?
        .join("void-rs")
        .join("local")
        .join("trace");
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let label = crate::settings::get()
        .bench_label
        .replace(['/', '\\', ' '], "_");
    let dir = dir.join(format!(
        "trace_{secs}_f{frame}{}{label}",
        if label.is_empty() { "" } else { "_" }
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let shaders = SHADERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_default();
    let ms = |t: u64| t as f64 * 1000.0 / freq.max(1) as f64;
    let t0 = stamps.iter().flatten().copied().min().unwrap_or(0);
    let mut csv =
        String::from("seq\tkind\tgpu_start_ms\tgpu_dur_ms\ta0\ta1\ta2\ta3\tvs\tps\tcs\tdesc\n");
    let name = |p: usize| -> String {
        if p == 0 {
            return String::new();
        }
        shaders
            .get(&p)
            .map_or_else(|| format!("?{p:x}"), |s| s.id.clone())
    };
    for (i, (kind, a, sh, d)) in rows.iter().enumerate() {
        let start = stamps[i];
        let next = stamps[i + 1..].iter().flatten().next().copied();
        let (s, dur) = match (start, next) {
            (Some(s), Some(n)) => (
                format!("{:.4}", ms(s - t0)),
                format!("{:.4}", ms(n.saturating_sub(s))),
            ),
            (Some(s), None) => (format!("{:.4}", ms(s - t0)), String::new()),
            _ => (String::new(), String::new()),
        };
        let _ = writeln!(
            csv,
            "{i}\t{kind:?}\t{s}\t{dur}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            a[0],
            a[1],
            a[2],
            a[3],
            name(sh[0]),
            name(sh[1]),
            name(sh[2]),
            strings.get(*d as usize).map_or("", String::as_str)
        );
    }
    std::fs::write(dir.join("events.tsv"), csv).map_err(|e| e.to_string())?;
    let mut sc = String::from("id\tstage\tbindings\n");
    let mut seen = std::collections::BTreeMap::new();
    for s in shaders.values() {
        seen.insert(s.id.clone(), s.clone());
    }
    for s in seen.values() {
        let _ = writeln!(sc, "{}\t{}\t{}", s.id, s.stage, s.bindings);
    }
    std::fs::write(dir.join("shaders.tsv"), sc).map_err(|e| e.to_string())?;
    let total = match (
        stamps.iter().flatten().min(),
        stamps.last().copied().flatten(),
    ) {
        (Some(a), Some(b)) => ms(b - a),
        _ => 0.0,
    };
    let meta = format!(
        "frame {frame}\nevents {}\ndisjoint {disjoint}\nfrequency {freq}\ngpu span (first event to frame end) {total:.3} ms\nshaders known {}\nsettings {:?}\n",
        rows.len(),
        shaders.len(),
        crate::settings::get()
    );
    std::fs::write(dir.join("meta.txt"), meta).map_err(|e| e.to_string())?;
    log::line(&format!(
        "trace: written to {} ({} events, gpu span {total:.2} ms{})",
        dir.display(),
        rows.len(),
        if disjoint {
            ", DISJOINT: timings invalid"
        } else {
            ""
        }
    ));
    Ok(())
}

// ---------------------------------------------------------------- installation --

fn dummy_device() -> Result<(ID3D11Device, ID3D11DeviceContext), String> {
    let mut device = None;
    let mut ctx = None;
    // SAFETY: standard device creation with valid out pointers.
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_1]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut ctx),
        )
    }
    .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
    Ok((device.ok_or("no device")?, ctx.ok_or("no context")?))
}

fn vfn(raw: *mut c_void, index: usize) -> usize {
    // SAFETY: live COM interface pointer; first field is its vtable.
    unsafe { *(*(raw as *const *const usize)).add(index) }
}

/// Shader-creation hooks (install as early as possible, before the game creates its
/// device): the device implementation found through a temporary device. Used by the
/// profiler (`profile = 1`) and the reflections' shader patch; installed once.
pub fn install_shader_hooks() -> Result<(), String> {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    let (device, _ctx) = dummy_device()?;
    let fns: [CreateShaderFn; 3] = [create_shader::<0>, create_shader::<1>, create_shader::<2>];
    for (i, index) in [12usize, 15, 18].into_iter().enumerate() {
        let t = vfn(device.as_raw(), index);
        // SAFETY: d3d11.dll Create*Shader implementation with this signature.
        let tr = unsafe { minhook::MinHook::create_hook(t as *mut c_void, fns[i] as *mut c_void) }
            .map_err(|e| format!("create_hook Create*Shader: {e:?}"))?;
        CREATE_ORIG[i].store(tr as usize, Ordering::Relaxed);
        // SAFETY: hook created above.
        unsafe { minhook::MinHook::enable_hook(t as *mut c_void) }
            .map_err(|e| format!("enable_hook Create*Shader: {e:?}"))?;
    }
    let t = vfn(device.as_raw(), 23);
    // SAFETY: d3d11.dll CreateSamplerState implementation with this signature.
    let tr =
        unsafe { minhook::MinHook::create_hook(t as *mut c_void, create_sampler as *mut c_void) }
            .map_err(|e| format!("create_hook CreateSamplerState: {e:?}"))?;
    SAMPLER_ORIG.store(tr as usize, Ordering::Relaxed);
    // SAFETY: hook created above.
    unsafe { minhook::MinHook::enable_hook(t as *mut c_void) }
        .map_err(|e| format!("enable_hook CreateSamplerState: {e:?}"))?;
    log::line("trace: shader and sampler creation hooks installed");
    Ok(())
}

/// Hooks the immediate context's methods.
pub fn install_context_hooks() -> Result<(), String> {
    let (_device, ctx) = dummy_device()?;
    // Newer interface vtables may point at other implementations (d2enhanced hooks all of
    // them); only the base vtable is hooked here, differences are logged as a coverage gap.
    if let Ok(c1) = ctx.cast::<ID3D11DeviceContext1>() {
        let differ: Vec<usize> = INDEX
            .iter()
            .copied()
            .filter(|&i| vfn(c1.as_raw(), i) != vfn(ctx.as_raw(), i))
            .collect();
        if !differ.is_empty() {
            log::line(&format!(
                "trace: ID3D11DeviceContext1 vtable differs for methods {differ:?} (not hooked)"
            ));
        }
    }
    let d = detours();
    let mut hooked = 0;
    for m in 0..NM {
        let t = vfn(ctx.as_raw(), INDEX[m]);
        // SAFETY: d3d11.dll immediate-context implementation of method `INDEX[m]`; the
        // detour has the identical signature.
        let tr = unsafe { minhook::MinHook::create_hook(t as *mut c_void, d[m]) }
            .map_err(|e| format!("create_hook method {}: {e:?}", INDEX[m]))?;
        ORIG[m].store(tr as usize, Ordering::Relaxed);
        // SAFETY: hook created above.
        unsafe { minhook::MinHook::enable_hook(t as *mut c_void) }
            .map_err(|e| format!("enable_hook method {}: {e:?}", INDEX[m]))?;
        hooked += 1;
    }
    log::line(&format!("trace: {hooked} immediate-context methods hooked"));
    Ok(())
}
