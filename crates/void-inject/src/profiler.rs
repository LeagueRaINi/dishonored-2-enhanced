//! Optional low-overhead profiler (settings `profile = 1`; docs/performance-baseline.md).
//!
//! Sources, all read without forcing CPU/GPU synchronisation:
//! - Engine timers (O, IDB of the analysed exe): `RenderFrame` (0x140361BC0) stores the
//!   frame time `qword_1425AE8E8`, swap phase `dword_1425AE8E0` (sync + frame limiter +
//!   present, measured by `sub_1403591B0`), GPU frame time `dword_1425AE8E4` (the
//!   engine's own query ring), `RenderFrame` duration `dword_1425AE8D8`; the renderer
//!   object (`RenderFrame` arg 1) holds front-end timestamps (+544/+552, around
//!   `RenderAllViews` and the subsystem updates in `sub_1403619D0`) and the `r_showStats`
//!   CPU numbers (+592 portal query, +600 light query, +608 shadow casters query, +616 cpu
//!   render, +624 cpu wait gpuSubmits; counts at +496...). All in microseconds. The
//!   engine's per-pass GPU timers are compiled out in retail (they push constant 0).
//! - Our CPU timings (QPC) around selected engine functions with integer-only signatures
//!   and around the immediate context's `ExecuteCommandList` (on AMD the D3D11 runtime
//!   replays deferred command lists there, so this is submission + replay CPU cost).
//! - Our GPU timestamps (D3D11 queries on the immediate context) around each
//!   `ExecuteCommandList`, bracketed by a per-frame disjoint query, read back several
//!   frames later with `D3D11_ASYNC_GETDATA_DONOTFLUSH`; unready or disjoint frames are
//!   dropped (counted), never waited for.
//! - Command-list labels: the first exe return addresses on the stack when a deferred
//!   context's `FinishCommandList` runs (identifies the engine job that recorded it).
//!
//! Nested intervals are reported separately and are not additive (e.g. `present` is
//! inside `swap`). A frame record covers one `RenderFrame` entry to the next; intervals
//! on other threads are attributed to the frame in which they end.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::{HRESULT, Interface};

use crate::passes::{self, Class};
use crate::{log, mem, settings};

// --------------------------------------------------------------- engine hooks --

/// Engine functions timed with the generic 4-integer-argument detour. All have at most
/// four integer/pointer parameters and no floating-point parameters (checked in the
/// IDB); the frame limiter `sub_1400462E0` takes a float and is therefore not hooked.
/// The engine present (`sub_14036B3A0`) is hooked once in `hook.rs` (shared with the
/// overlay), which reports it through `record_present`.
const PHASES: [(&str, u64); 10] = [
    ("render_frame", 0x36_1BC0),
    ("swap_sync_limit_present", 0x35_91B0),
    ("gpu_timer_wait", 0x36_AE40),
    ("render_views", 0x36_1D60),
    ("render_views_alt_a", 0x36_2E90),
    ("end_frame_flush", 0x2E_CD00),
    // Critical-path analysis (docs/performance-baseline.md "CPU submission without
    // driver command lists"; IDA names): the main thread starts the render workers
    // (`JobSys_KickWorkers` 0x1402C8310) and waits for every pass to be recorded and
    // submitted (`JobSys_WaitWorkers` 0x1402C8440); `RenderPass_Record` 0x1402C5D50 runs
    // one pass callback + FinishCommandList on a worker (summed over all workers);
    // `RenderPass_Submit` 0x1402C5F60 executes a recorded list under the immediate lock
    // (lock wait included; the FSR redirect sits inside it at +0x107, not at the entry).
    ("graph_kick", 0x2C_8310),
    ("graph_wait", 0x2C_8440),
    ("pass_record", 0x2C_5D50),
    ("pass_submit", 0x2C_5F60),
];
const PH_KICK: usize = 6;
const PH_WAIT: usize = 7;
const PH_RECORD: usize = 8;
const PH_SUBMIT: usize = 9;
/// Extra CPU phases measured outside the generic table.
const PH_RENDER_ALL_VIEWS: usize = PHASES.len();
const PH_ECL: usize = PHASES.len() + 1;
const PH_FINISH: usize = PHASES.len() + 2;
const PH_PRESENT: usize = PHASES.len() + 3;
const NPH: usize = PHASES.len() + 4;

pub fn phase_names() -> Vec<&'static str> {
    let mut v: Vec<&str> = PHASES.iter().map(|p| p.0).collect();
    v.extend([
        "render_all_views",
        "execute_command_list",
        "finish_command_list",
        "present",
    ]);
    v
}

const FRAME_TIME_RVA: u64 = 0x25A_E8E8;
const SWAP_RVA: u64 = 0x25A_E8E0;
const GPU_RVA: u64 = 0x25A_E8E4;
const RENDER_FRAME_US_RVA: u64 = 0x25A_E8D8;

struct Acc {
    ns: [AtomicU64; NPH],
    count: [AtomicU32; NPH],
    max_ns: [AtomicU64; NPH],
}

static ACC: Acc = Acc {
    ns: [const { AtomicU64::new(0) }; NPH],
    count: [const { AtomicU32::new(0) }; NPH],
    max_ns: [const { AtomicU64::new(0) }; NPH],
};
/// Thread id first seen per phase (for the report: which thread runs what).
static THREADS: [AtomicU32; NPH] = [const { AtomicU32::new(0) }; NPH];

static RECORDING: AtomicBool = AtomicBool::new(false);
static FRAME: AtomicU64 = AtomicU64::new(0);
static ORIG: [AtomicUsize; PHASES.len()] = [const { AtomicUsize::new(0) }; PHASES.len()];
static T0: OnceLock<Instant> = OnceLock::new();

fn now_ns() -> u64 {
    T0.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn thread_id() -> u32 {
    // SAFETY: no preconditions.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

fn add(phase: usize, ns: u64) {
    ACC.ns[phase].fetch_add(ns, Ordering::Relaxed);
    ACC.count[phase].fetch_add(1, Ordering::Relaxed);
    ACC.max_ns[phase].fetch_max(ns, Ordering::Relaxed);
    if THREADS[phase].load(Ordering::Relaxed) == 0 {
        THREADS[phase].store(thread_id(), Ordering::Relaxed);
    }
}

/// Called by `hook.rs` around `RenderAllViews` (already hooked there).
pub fn record_render_all_views(ns: u64) {
    if RECORDING.load(Ordering::Relaxed) {
        add(PH_RENDER_ALL_VIEWS, ns);
    }
}

/// Called by `hook.rs` around the engine present (`sub_14036B3A0`, overlay excluded).
pub fn record_present(ns: u64) {
    if RECORDING.load(Ordering::Relaxed) {
        add(PH_PRESENT, ns);
    }
}

/// The engine's CPU frame time (µs): frame time `qword_1425AE8E8` minus the swap phase
/// `dword_1425AE8E0` (as the benchmark's `eng_cpu_ms`).
pub fn engine_cpu_us() -> Option<u32> {
    let base = mem::image_base()?;
    let frame = mem::u64_at(base + FRAME_TIME_RVA)?;
    let swap = mem::u32_at(base + SWAP_RVA)?;
    u32::try_from(frame.saturating_sub(u64::from(swap))).ok()
}

/// The engine's own GPU frame time (`dword_1425AE8E4`, µs), as the profiler records it.
pub fn engine_gpu_us() -> Option<u32> {
    mem::u32_at(mem::image_base()? + GPU_RVA)
}

type Fn4 = unsafe extern "system" fn(u64, u64, u64, u64) -> u64;

unsafe extern "system" fn timed<const I: usize>(a: u64, b: u64, c: u64, d: u64) -> u64 {
    if I == 0 {
        let _ = std::panic::catch_unwind(|| frame_begin(a));
    }
    let orig = ORIG[I].load(Ordering::Relaxed);
    // SAFETY: MinHook trampoline of the hooked function (<= 4 integer arguments).
    let f = unsafe { std::mem::transmute::<usize, Fn4>(orig) };
    if !RECORDING.load(Ordering::Relaxed) {
        // SAFETY: arguments passed through unchanged.
        return unsafe { f(a, b, c, d) };
    }
    let t = now_ns();
    // SAFETY: arguments passed through unchanged.
    let r = unsafe { f(a, b, c, d) };
    let t1 = now_ns();
    add(I, t1.saturating_sub(t));
    if I == 0 {
        let _ = std::panic::catch_unwind(|| frame_end(a));
    }
    if let Some(kind) = PassKind::of_phase(I) {
        pass_event(kind, a, t, t1);
    }
    r
}

const DETOURS: [Fn4; PHASES.len()] = [
    timed::<0>, timed::<1>, timed::<2>, timed::<3>, timed::<4>, timed::<5>, timed::<6>, timed::<7>,
    timed::<8>, timed::<9>,
];

// ------------------------------------------------------------------- pass events --

/// Render-graph events for the critical-path analysis (bench `<segment>_passes.tsv`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// `JobSys_KickWorkers` on the main thread (frame reference point).
    Kick,
    /// `JobSys_WaitWorkers` on the main thread.
    Wait,
    /// `RenderPass_Record` on a worker: pass callback + FinishCommandList.
    Record,
    /// `RenderPass_Submit`: ExecuteCommandList under the immediate lock (lock wait
    /// included), or for inline passes (flag 0x20) the callback on the immediate context.
    Submit,
}

impl PassKind {
    const fn of_phase(i: usize) -> Option<Self> {
        match i {
            PH_KICK => Some(Self::Kick),
            PH_WAIT => Some(Self::Wait),
            PH_RECORD => Some(Self::Record),
            PH_SUBMIT => Some(Self::Submit),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Kick => "kick",
            Self::Wait => "wait",
            Self::Record => "record",
            Self::Submit => "submit",
        }
    }
}

/// One timed render-graph call. `callback_rva` / `flags` are the pass's callback
/// (pass+216, IDB `RenderPass_Record`) relative to the image base and its flag word
/// (pass+244); zero for `Kick` / `Wait`.
#[derive(Clone, Copy, Debug)]
pub struct PassEv {
    pub frame: u64,
    pub kind: PassKind,
    pub callback_rva: u32,
    pub flags: u32,
    pub thread: u32,
    pub t0_ns: u64,
    pub t1_ns: u64,
}

static PASS_EVENTS: Mutex<Vec<PassEv>> = Mutex::new(Vec::new());

fn pass_event(kind: PassKind, pass: u64, t0_ns: u64, t1_ns: u64) {
    let (callback_rva, flags) = if matches!(kind, PassKind::Record | PassKind::Submit) {
        // SAFETY: `pass` is the live pass object the engine function just used; it reads
        // these same fields (+216 callback, +244 flags) itself.
        let (cb, fl) = unsafe {
            (
                std::ptr::read_unaligned((pass + 216) as *const u64),
                std::ptr::read_unaligned((pass + 244) as *const u32),
            )
        };
        let base = mem::image_base().unwrap_or(0);
        (u32::try_from(cb.wrapping_sub(base)).unwrap_or(u32::MAX), fl)
    } else {
        (0, 0)
    };
    let ev = PassEv {
        frame: FRAME.load(Ordering::Relaxed),
        kind,
        callback_rva,
        flags,
        thread: thread_id(),
        t0_ns,
        t1_ns,
    };
    PASS_EVENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(ev);
}

/// Takes the pass events recorded since `start_recording`.
pub fn take_pass_events() -> Vec<PassEv> {
    std::mem::take(
        &mut *PASS_EVENTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

// ------------------------------------------------------------------ frame records --

/// One frame (RenderFrame entry to the next entry).
#[derive(Clone, Debug, Default)]
pub struct FrameRec {
    pub frame: u64,
    pub t_ms: f64,
    pub interval_ms: f64,
    /// Engine values (µs) read after `RenderFrame` of this frame.
    pub eng_frame_us: u64,
    pub eng_swap_us: u32,
    pub eng_gpu_us: u32,
    pub eng_render_frame_us: u32,
    pub eng_front_end_us: i64,
    /// Renderer stats: portal, light, shadow-caster queries, cpu render, wait gpuSubmits.
    pub eng_stats_us: [u64; 5],
    pub eng_counts: [u32; 7],
    pub phase_ms: [f64; NPH],
    pub phase_count: [u32; NPH],
    pub phase_max_ms: [f64; NPH],
}

struct FrameState {
    start_ns: u64,
    rec: FrameRec,
}

static CUR: Mutex<Option<FrameState>> = Mutex::new(None);
static FRAMES: Mutex<Vec<FrameRec>> = Mutex::new(Vec::new());

fn frame_begin(renderer: u64) {
    let now = now_ns();
    let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
    let mut cur = CUR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(prev) = cur.take()
        && RECORDING.load(Ordering::Relaxed)
    {
        let mut rec = prev.rec;
        rec.interval_ms = (now - prev.start_ns) as f64 / 1e6;
        for p in 0..NPH {
            rec.phase_ms[p] = ACC.ns[p].swap(0, Ordering::Relaxed) as f64 / 1e6;
            rec.phase_count[p] = ACC.count[p].swap(0, Ordering::Relaxed);
            rec.phase_max_ms[p] = ACC.max_ns[p].swap(0, Ordering::Relaxed) as f64 / 1e6;
        }
        FRAMES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(rec);
    } else {
        for p in 0..NPH {
            ACC.ns[p].store(0, Ordering::Relaxed);
            ACC.count[p].store(0, Ordering::Relaxed);
            ACC.max_ns[p].store(0, Ordering::Relaxed);
        }
    }
    *cur = Some(FrameState {
        start_ns: now,
        rec: FrameRec {
            frame,
            t_ms: now as f64 / 1e6,
            ..Default::default()
        },
    });
    drop(cur);
    let _ = renderer;
    crate::bench::step(frame);
}

fn frame_end(renderer: u64) {
    let Some(base) = mem::image_base() else {
        return;
    };
    let mut cur = CUR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(st) = cur.as_mut() else { return };
    let r = &mut st.rec;
    r.eng_frame_us = mem::u64_at(base + FRAME_TIME_RVA).unwrap_or(0);
    r.eng_swap_us = mem::u32_at(base + SWAP_RVA).unwrap_or(0);
    r.eng_gpu_us = mem::u32_at(base + GPU_RVA).unwrap_or(0);
    r.eng_render_frame_us = mem::u32_at(base + RENDER_FRAME_US_RVA).unwrap_or(0);
    // Renderer block +496..+632 in one read.
    let mut b = [0u8; 136];
    if mem::read_into(renderer + 496, &mut b) {
        let q = |o: usize| u64::from_le_bytes(b[o - 496..o - 488].try_into().unwrap_or([0; 8]));
        let d = |o: usize| u32::from_le_bytes(b[o - 496..o - 492].try_into().unwrap_or([0; 4]));
        r.eng_front_end_us = q(552) as i64 - q(544) as i64;
        r.eng_stats_us = [q(592), q(600), q(608), q(616), q(624)];
        r.eng_counts = [d(496), d(500), d(504), d(508), d(512), d(516), d(520)];
    }
}

/// Starts a recording segment (clears the frame buffer; GPU results are taken by
/// frame range, since they resolve a few frames late).
pub fn start_recording() {
    FRAMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    PASS_EVENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    RECORDING.store(true, Ordering::Relaxed);
}

pub fn is_recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// Stops recording; returns the frames recorded so far.
pub fn stop_recording() -> Vec<FrameRec> {
    RECORDING.store(false, Ordering::Relaxed);
    std::mem::take(
        &mut *FRAMES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// Takes the resolved GPU results of frames `from..to`.
pub fn take_gpu_results(from: u64, to: u64) -> Vec<GpuFrame> {
    let mut g = GPU_RESULTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (take, keep): (Vec<GpuFrame>, Vec<GpuFrame>) =
        g.drain(..).partition(|r| r.frame >= from && r.frame < to);
    // Results older than the current window are stale; keep only newer ones.
    *g = keep.into_iter().filter(|r| r.frame >= to).collect();
    take
}

pub fn thread_ids() -> Vec<(&'static str, u32)> {
    phase_names()
        .into_iter()
        .zip(THREADS.iter().map(|t| t.load(Ordering::Relaxed)))
        .collect()
}

// ----------------------------------------------------------- command list labels --

/// Label table: id -> exe RVAs of the first return addresses inside the executable.
static LABELS: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());
static LIST_LABEL: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

pub fn labels() -> Vec<Vec<u32>> {
    LABELS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn exe_range() -> (u64, u64) {
    static R: OnceLock<(u64, u64)> = OnceLock::new();
    *R.get_or_init(|| {
        let base = mem::image_base().unwrap_or(0);
        // PE: e_lfanew at +0x3C, SizeOfImage at optional header +56 (NT headers +80).
        let size = mem::u32_at(base + 0x3C)
            .and_then(|l| mem::u32_at(base + u64::from(l) + 80))
            .unwrap_or(0);
        (base, base + u64::from(size))
    })
}

fn caller_label() -> u32 {
    let mut frames = [std::ptr::null_mut::<c_void>(); 24];
    // SAFETY: valid output buffer.
    let n = unsafe {
        windows::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, &mut frames, None)
    } as usize;
    let (lo, hi) = exe_range();
    let rvas: Vec<u32> = frames[..n]
        .iter()
        .map(|f| *f as u64)
        .filter(|a| *a >= lo && *a < hi)
        .take(6)
        .map(|a| (a - lo) as u32)
        .collect();
    let mut labels = LABELS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(i) = labels.iter().position(|l| *l == rvas) {
        return i as u32;
    }
    labels.push(rvas);
    (labels.len() - 1) as u32
}

// --------------------------------------------------------------- D3D11 hooks --

type EclFn = unsafe extern "system" fn(*mut c_void, *mut c_void, i32);
type FinishFn = unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> HRESULT;
static ECL_ORIG: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static FINISH_ORIG: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

unsafe extern "system" fn ecl_detour<const I: usize>(
    this: *mut c_void,
    list: *mut c_void,
    restore: i32,
) {
    // SAFETY: MinHook trampoline with this signature.
    let f = unsafe { std::mem::transmute::<usize, EclFn>(ECL_ORIG[I].load(Ordering::Relaxed)) };
    if !RECORDING.load(Ordering::Relaxed) {
        // SAFETY: arguments passed through unchanged.
        return unsafe { f(this, list, restore) };
    }
    let gpu = settings::get().profile_gpu;
    let passes_mode = gpu && settings::get().profile_gpu_passes;
    let stamp = if gpu && !passes_mode {
        std::panic::catch_unwind(|| gpu_before(this, list)).unwrap_or(None)
    } else {
        None
    };
    if passes_mode {
        let _ = std::panic::catch_unwind(|| pass_mark(this, Class::ListSetup));
        passes::reset_last(Some(Class::ListSetup));
    }
    let t = now_ns();
    // SAFETY: arguments passed through unchanged.
    unsafe { f(this, list, restore) };
    add(PH_ECL, now_ns().saturating_sub(t));
    if let Some(s) = stamp {
        let _ = std::panic::catch_unwind(|| gpu_after(this, s));
    }
    if passes_mode {
        let _ = std::panic::catch_unwind(|| pass_mark(this, Class::BetweenLists));
        passes::reset_last(Some(Class::BetweenLists));
    }
}

/// True while pass-class GPU timing is active (recording, `profile_gpu_passes`).
pub fn pass_timing() -> bool {
    RECORDING.load(Ordering::Relaxed) && {
        let s = settings::get();
        s.profile_gpu && s.profile_gpu_passes
    }
}

/// Writes a GPU timestamp for the start of `class` on the immediate context.
pub fn pass_mark(this: *mut c_void, class: Class) {
    // SAFETY: `this` is the context the game called.
    let Some(ctx) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&this) }) else {
        return;
    };
    // SAFETY: plain getter.
    if unsafe { ctx.GetType() } != D3D11_DEVICE_CONTEXT_IMMEDIATE {
        return;
    }
    let mut g = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(st) = ensure_frame(ctx, &mut g) else {
        return;
    };
    let Some(cur) = st.cur.as_ref() else { return };
    if cur.marks.len() >= MAX_MARKS {
        return;
    }
    let Some(q) = st
        .free
        .pop()
        .or_else(|| make_query(&st.device, D3D11_QUERY_TIMESTAMP))
    else {
        return;
    };
    // SAFETY: timestamp query end on the immediate context.
    unsafe { ctx.End(&q) };
    if let Some(cur) = st.cur.as_mut() {
        cur.marks.push((class, q));
    }
}

unsafe extern "system" fn finish_detour<const I: usize>(
    this: *mut c_void,
    restore: i32,
    out: *mut *mut c_void,
) -> HRESULT {
    // SAFETY: MinHook trampoline with this signature.
    let f =
        unsafe { std::mem::transmute::<usize, FinishFn>(FINISH_ORIG[I].load(Ordering::Relaxed)) };
    if !RECORDING.load(Ordering::Relaxed) {
        // SAFETY: arguments passed through unchanged.
        return unsafe { f(this, restore, out) };
    }
    let t = now_ns();
    // SAFETY: arguments passed through unchanged.
    let hr = unsafe { f(this, restore, out) };
    add(PH_FINISH, now_ns().saturating_sub(t));
    if settings::get().profile_labels && hr.is_ok() && !out.is_null() {
        // SAFETY: `out` was filled by FinishCommandList on success.
        let list = unsafe { *out } as usize;
        let label = caller_label();
        LIST_LABEL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .insert(list, label);
    }
    hr
}

// ---------------------------------------------------------------- GPU queries --

/// Resolved GPU timings of one frame (ms).
#[derive(Clone, Debug, Default)]
pub struct GpuFrame {
    pub frame: u64,
    /// First command-list start to last command-list end.
    pub span_ms: f64,
    /// Sum of command-list durations (gaps excluded).
    pub lists_ms: f64,
    pub lists: Vec<(u32, f64)>,
    /// Pass-class durations (ms), indexed by `passes::Class` (pass mode).
    pub passes: [f64; passes::CLASS_NAMES.len()],
}

struct Pending {
    frame: u64,
    disjoint: ID3D11Query,
    stamps: Vec<(u32, ID3D11Query, ID3D11Query)>,
    marks: Vec<(Class, ID3D11Query)>,
    age: u32,
}

struct GpuState {
    device: ID3D11Device,
    free: Vec<ID3D11Query>,
    free_disjoint: Vec<ID3D11Query>,
    cur: Option<Pending>,
    pending: std::collections::VecDeque<Pending>,
    dropped_disjoint: u64,
    dropped_unready: u64,
}

// SAFETY: only used on the thread that calls the immediate context (inside its
// ExecuteCommandList), under the mutex.
unsafe impl Send for GpuState {}

static GPU: Mutex<Option<GpuState>> = Mutex::new(None);
static GPU_RESULTS: Mutex<Vec<GpuFrame>> = Mutex::new(Vec::new());
pub static GPU_DROPPED: AtomicU64 = AtomicU64::new(0);
const MAX_STAMPS: usize = 256;
const MAX_MARKS: usize = 1024;
/// Frames a query set may stay unresolved before it is dropped.
const MAX_AGE: u32 = 8;

/// Current engine frame number (RenderFrame entries).
pub fn current_frame() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

pub(crate) fn make_query(device: &ID3D11Device, kind: D3D11_QUERY) -> Option<ID3D11Query> {
    let mut q = None;
    // SAFETY: valid descriptor and out pointer.
    unsafe {
        device
            .CreateQuery(
                &D3D11_QUERY_DESC {
                    Query: kind,
                    MiscFlags: 0,
                },
                Some(&mut q),
            )
            .ok()?;
    }
    q
}

/// Raw `GetData` (S_FALSE = not ready must be distinguished from S_OK).
pub(crate) fn get_data(
    ctx: &ID3D11DeviceContext,
    q: &ID3D11Query,
    out: *mut c_void,
    size: u32,
) -> HRESULT {
    // SAFETY: vtable call with a live context and query; `out` holds `size` bytes.
    unsafe {
        (Interface::vtable(ctx).GetData)(
            Interface::as_raw(ctx),
            Interface::as_raw(q),
            out,
            size,
            D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
        )
    }
}

fn gpu_before(this: *mut c_void, list: *mut c_void) -> Option<(u32, ID3D11Query, ID3D11Query)> {
    // SAFETY: `this` is the context the game called.
    let ctx = unsafe { ID3D11DeviceContext::from_raw_borrowed(&this) }?;
    // SAFETY: plain getter.
    if unsafe { ctx.GetType() } != D3D11_DEVICE_CONTEXT_IMMEDIATE {
        return None;
    }
    let mut g = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let st = ensure_frame(ctx, &mut g)?;
    let cur = st.cur.as_ref()?;
    if cur.stamps.len() >= MAX_STAMPS {
        return None;
    }
    let mut take = || {
        st.free
            .pop()
            .or_else(|| make_query(&st.device, D3D11_QUERY_TIMESTAMP))
    };
    let (a, b) = (take()?, take()?);
    // SAFETY: timestamp query end on the immediate context.
    unsafe { ctx.End(&a) };
    let label = LIST_LABEL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(|m| m.get(&(list as usize)).copied())
        .unwrap_or(u32::MAX);
    Some((label, a, b))
}

/// The GPU state with an open query set for the current engine frame (closing and
/// queuing the previous frame's set, and resolving older ones, at a frame change).
fn ensure_frame<'a>(
    ctx: &ID3D11DeviceContext,
    g: &'a mut Option<GpuState>,
) -> Option<&'a mut GpuState> {
    if g.is_none() {
        // SAFETY: plain getter.
        let device = unsafe { ctx.GetDevice() }.ok()?;
        *g = Some(GpuState {
            device,
            free: Vec::new(),
            free_disjoint: Vec::new(),
            cur: None,
            pending: Default::default(),
            dropped_disjoint: 0,
            dropped_unready: 0,
        });
    }
    let st = g.as_mut()?;
    let frame = FRAME.load(Ordering::Relaxed);
    if st.cur.as_ref().is_none_or(|c| c.frame != frame) {
        if let Some(c) = st.cur.take() {
            // SAFETY: ending a begun disjoint query on the immediate context.
            unsafe { ctx.End(&c.disjoint) };
            st.pending.push_back(c);
        }
        resolve(ctx, st);
        let dq = st
            .free_disjoint
            .pop()
            .or_else(|| make_query(&st.device, D3D11_QUERY_TIMESTAMP_DISJOINT))?;
        // SAFETY: beginning a disjoint query on the immediate context.
        unsafe { ctx.Begin(&dq) };
        st.cur = Some(Pending {
            frame,
            disjoint: dq,
            stamps: Vec::new(),
            marks: Vec::new(),
            age: 0,
        });
    }
    Some(st)
}

fn gpu_after(this: *mut c_void, (label, a, b): (u32, ID3D11Query, ID3D11Query)) {
    // SAFETY: `this` is the context the game called.
    let Some(ctx) = (unsafe { ID3D11DeviceContext::from_raw_borrowed(&this) }) else {
        return;
    };
    // SAFETY: timestamp query end on the immediate context.
    unsafe { ctx.End(&b) };
    let mut g = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(cur) = g.as_mut().and_then(|s| s.cur.as_mut()) {
        cur.stamps.push((label, a, b));
    }
}

fn resolve(ctx: &ID3D11DeviceContext, st: &mut GpuState) {
    while let Some(p) = st.pending.front_mut() {
        p.age += 1;
        let mut dj = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
        let hr = get_data(
            ctx,
            &p.disjoint,
            (&raw mut dj).cast(),
            std::mem::size_of_val(&dj) as u32,
        );
        if hr.0 == 1 || hr.is_err() {
            // S_FALSE: not ready yet.
            if p.age <= MAX_AGE && hr.is_ok() {
                break;
            }
            st.dropped_unready += 1;
            GPU_DROPPED.fetch_add(1, Ordering::Relaxed);
            let p = st.pending.pop_front().unwrap_or_else(|| unreachable!());
            recycle(st, p);
            continue;
        }
        let p = st.pending.pop_front().unwrap_or_else(|| unreachable!());
        if dj.Disjoint.as_bool() || dj.Frequency == 0 {
            st.dropped_disjoint += 1;
            GPU_DROPPED.fetch_add(1, Ordering::Relaxed);
            recycle(st, p);
            continue;
        }
        let ms = |t: u64| t as f64 * 1000.0 / dj.Frequency as f64;
        let mut out = GpuFrame {
            frame: p.frame,
            ..Default::default()
        };
        let (mut first, mut last, mut ok) = (u64::MAX, 0u64, true);
        for (label, a, b) in &p.stamps {
            let (mut ta, mut tb) = (0u64, 0u64);
            let ha = get_data(ctx, a, (&raw mut ta).cast(), 8);
            let hb = get_data(ctx, b, (&raw mut tb).cast(), 8);
            if ha.0 != 0 || hb.0 != 0 {
                ok = false;
                break;
            }
            first = first.min(ta);
            last = last.max(tb);
            let d = ms(tb.saturating_sub(ta));
            out.lists_ms += d;
            out.lists.push((*label, d));
        }
        // Pass marks: each segment runs from its mark to the next one; the last
        // segment (after the frame's final command list) is not attributed.
        let mut times = Vec::with_capacity(p.marks.len());
        for (class, q) in &p.marks {
            let mut t = 0u64;
            if !ok || get_data(ctx, q, (&raw mut t).cast(), 8).0 != 0 {
                ok = false;
                break;
            }
            times.push((*class, t));
        }
        if ok && times.len() >= 2 {
            for w in times.windows(2) {
                out.passes[w[0].0 as usize] += ms(w[1].1.saturating_sub(w[0].1));
            }
            first = first.min(times[0].1);
            last = last.max(times[times.len() - 1].1);
            out.lists_ms = out
                .passes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != Class::BetweenLists as usize)
                .map(|(_, v)| v)
                .sum();
        }
        if ok && first < last {
            out.span_ms = ms(last - first);
            GPU_RESULTS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(out);
        } else {
            st.dropped_unready += 1;
            GPU_DROPPED.fetch_add(1, Ordering::Relaxed);
        }
        recycle(st, p);
    }
}

fn recycle(st: &mut GpuState, p: Pending) {
    st.free_disjoint.push(p.disjoint);
    for (_, q) in p.marks {
        st.free.push(q);
    }
    for (_, a, b) in p.stamps {
        st.free.push(a);
        st.free.push(b);
    }
}

// ----------------------------------------------------------------- installation --

/// Installs the profiler hooks (engine functions, D3D11 ExecuteCommandList of the
/// immediate context and FinishCommandList of deferred contexts).
pub fn install() -> Result<(), String> {
    let _ = T0.get_or_init(Instant::now);
    let base = mem::image_base().ok_or("no image base")?;
    for (i, (name, rva)) in PHASES.iter().enumerate() {
        let target = (base + rva) as *mut c_void;
        // SAFETY: start of the named function in the verified executable; the detour
        // forwards its (<= 4 integer) arguments unchanged.
        let tramp = unsafe { minhook::MinHook::create_hook(target, DETOURS[i] as *mut c_void) }
            .map_err(|e| format!("create_hook {name}: {e:?}"))?;
        ORIG[i].store(tramp as usize, Ordering::Relaxed);
    }
    for (_, rva) in PHASES {
        // SAFETY: hooks created above.
        unsafe { minhook::MinHook::enable_hook((base + rva) as *mut c_void) }
            .map_err(|e| format!("enable_hook: {e:?}"))?;
    }
    // D3D11 implementations from a temporary device (immediate and deferred contexts
    // use different implementations; newer interface vtables may differ too).
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
    let device: ID3D11Device = device.ok_or("no device")?;
    let ctx: ID3D11DeviceContext = ctx.ok_or("no context")?;
    let mut deferred = None;
    // SAFETY: valid out pointer.
    unsafe { device.CreateDeferredContext(0, Some(&mut deferred)) }
        .map_err(|e| format!("CreateDeferredContext: {e}"))?;
    let deferred: ID3D11DeviceContext = deferred.ok_or("no deferred context")?;
    let targets = |c: &ID3D11DeviceContext, index: usize| -> Vec<usize> {
        let mut out = Vec::new();
        let mut add = |raw: *mut c_void| {
            // SAFETY: live COM interface pointer; first field is its vtable.
            let f = unsafe { *(*(raw as *const *const usize)).add(index) };
            if !out.contains(&f) {
                out.push(f);
            }
        };
        add(c.as_raw());
        if let Ok(c1) = c.cast::<ID3D11DeviceContext1>() {
            add(c1.as_raw());
        }
        if let Ok(c4) = c.cast::<ID3D11DeviceContext4>() {
            add(c4.as_raw());
        }
        out
    };
    const ECL_INDEX: usize = 58;
    const FINISH_INDEX: usize = 114;
    let ecl: [EclFn; 4] = [
        ecl_detour::<0>,
        ecl_detour::<1>,
        ecl_detour::<2>,
        ecl_detour::<3>,
    ];
    let fin: [FinishFn; 4] = [
        finish_detour::<0>,
        finish_detour::<1>,
        finish_detour::<2>,
        finish_detour::<3>,
    ];
    let ecl_t = targets(&ctx, ECL_INDEX);
    let fin_t = targets(&deferred, FINISH_INDEX);
    for (i, &t) in ecl_t.iter().enumerate().take(4) {
        // SAFETY: d3d11.dll ExecuteCommandList implementation; same signature.
        let tr = unsafe { minhook::MinHook::create_hook(t as *mut c_void, ecl[i] as *mut c_void) }
            .map_err(|e| format!("create_hook ExecuteCommandList: {e:?}"))?;
        ECL_ORIG[i].store(tr as usize, Ordering::Relaxed);
        // SAFETY: hook created above.
        unsafe { minhook::MinHook::enable_hook(t as *mut c_void) }
            .map_err(|e| format!("enable_hook ExecuteCommandList: {e:?}"))?;
    }
    for (i, &t) in fin_t.iter().enumerate().take(4) {
        // SAFETY: d3d11.dll FinishCommandList implementation; same signature.
        let tr = unsafe { minhook::MinHook::create_hook(t as *mut c_void, fin[i] as *mut c_void) }
            .map_err(|e| format!("create_hook FinishCommandList: {e:?}"))?;
        FINISH_ORIG[i].store(tr as usize, Ordering::Relaxed);
        // SAFETY: hook created above.
        unsafe { minhook::MinHook::enable_hook(t as *mut c_void) }
            .map_err(|e| format!("enable_hook FinishCommandList: {e:?}"))?;
    }
    log::line(&format!(
        "profiler: {} engine functions hooked; ExecuteCommandList at {ecl_t:x?}, FinishCommandList at {fin_t:x?}",
        PHASES.len()
    ));
    Ok(())
}
