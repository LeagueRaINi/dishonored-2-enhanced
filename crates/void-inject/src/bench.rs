//! Benchmark driver (Ctrl+F3 with `profile = 1`): a repeatable sequence of a warm-up,
//! then per repeat a settle, a stationary segment and a 360 degree pan, all at a fixed
//! pose per map (`local/profile/pose_<map>.txt`, written on first use). The player is
//! placed with the engine's `setviewpos <x> <y> <z> [yaw] [pitch]` console command (O:
//! handler `sub_140886D10` queues a view teleport; z is the eye height), executed on the
//! main thread through `idCmdSystemLocal` (static object 0x14228BB00, vtable slot 8
//! "execute text now", from d2enhanced `src/console.cpp`), once per settle.
//!
//! The pan turns the player with synthetic mouse-look input (`mouse`: X counts injected
//! into the game's DirectInput mouse reads), so the view, first-person hands/weapon and
//! HUD markers turn as in normal play. Two earlier methods were rejected: per-frame
//! `setviewpos` teleports turned the body and the eye bobbed up to 0.55 m and circled
//! ~0.1 m (eotw run 1792193254); overwriting the render view left hands, weapon and UI
//! markers behind (user, eotw runs 1792194132/1792194435). The counts-to-degrees gain is
//! measured once per run during the warm-up (a probe of `CAL_COUNTS`, then turned back);
//! the pan then feeds counts at a constant rate for 360 degrees and logs the measured
//! turn. Saves are not touched; the frame-limiter cvar is restored at the end.
//!
//! Output per run: `local/profile/run_<time>_<label>/` with `meta.txt`, `cvars.tsv`,
//! `modules.txt`, and per recorded segment `<seg>_frames.tsv`, `<seg>_gpu_lists.tsv`,
//! plus `summary.tsv` (one row per segment and metric).

use std::ffi::CString;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::profiler::{self, FrameRec, GpuFrame, PassEv, PassKind};
use crate::{cvars, log, mem, mouse, settings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SegKind {
    Warmup,
    Settle,
    Static,
    Pan,
}

#[derive(Clone, Copy, Debug)]
struct Seg {
    kind: SegKind,
    rep: u32,
    secs: f32,
}

/// Mouse gain calibration (warm-up).
#[derive(Clone, Copy, Debug)]
enum Cal {
    Idle,
    Probing { yaw0: f32, delivered0: i64 },
    Done,
}

/// Progress of the current pan.
#[derive(Clone, Copy, Debug, Default)]
struct Pan {
    requested: i64,
    delivered0: i64,
    prev_yaw: f32,
    turned: f32,
    /// Counts per second currently fed (signed).
    counts_per_s: f32,
    /// Fractional count carried to the next frame.
    carry: f32,
    /// Smoothed measured turn rate (deg/s).
    rate: f32,
    /// Segment time of the previous step (s).
    last_t: f32,
}

struct Run {
    dir: PathBuf,
    segs: Vec<Seg>,
    idx: usize,
    seg_start: Instant,
    seg_first_frame: u64,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    cap_restore: Option<i32>,
    cal: Cal,
    /// Degrees of yaw per mouse count (signed); `None` if calibration failed.
    gain: Option<f32>,
    pan: Pan,
}

static RUN: Mutex<Option<Run>> = Mutex::new(None);
static TOGGLE: AtomicBool = AtomicBool::new(false);
static REQUESTS: AtomicU64 = AtomicU64::new(0);
/// Output directory of the last started run (overlay).
static LAST_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// A run is in progress.
pub fn running() -> bool {
    RUN.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

/// Directory of the last run started in this session.
pub fn last_dir() -> Option<PathBuf> {
    LAST_DIR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Requests a start (or an abort) at the next frame; called from the input check.
pub fn toggle(requests: u64) {
    REQUESTS.store(requests, Ordering::Relaxed);
    TOGGLE.store(true, Ordering::Relaxed);
}

const CMD_SYSTEM_RVA: u64 = 0x228_BB00;
const CMD_EXECUTE_SLOT: usize = 8;

fn exec(cmd: &str) {
    let Some(base) = mem::image_base() else {
        return;
    };
    let Ok(text) = CString::new(cmd) else { return };
    let this = base + CMD_SYSTEM_RVA;
    let Some(vt) = mem::u64_at(this) else { return };
    let Some(f) = mem::u64_at(vt + 8 * CMD_EXECUTE_SLOT as u64) else {
        return;
    };
    type Exec = unsafe extern "system" fn(u64, *const i8);
    // SAFETY: `f` is the engine's ExecuteCommandText (vtable slot 8 of the static
    // idCmdSystemLocal, d2enhanced console.cpp); called on the main thread with a
    // NUL-terminated string that outlives the call.
    unsafe { std::mem::transmute::<u64, Exec>(f)(this, text.as_ptr()) };
}

fn setviewpos(p: [f32; 3], yaw: f32, pitch: f32) {
    exec(&format!(
        "setviewpos {:.3} {:.3} {:.3} {:.3} {:.3}",
        p[0], p[1], p[2], yaw, pitch
    ));
}

/// Calibration probe size (mouse counts) and its timing within the warm-up (s).
const CAL_COUNTS: i32 = 200;
const CAL_START_S: f32 = 1.0;
const CAL_READ_S: f32 = 2.5;

/// Yaw (degrees) of the first view request, recorded every frame by `observe_view`.
static VIEW_YAW: AtomicU32 = AtomicU32::new(0);
static VIEW_VALID: AtomicBool = AtomicBool::new(false);

/// Called after `RenderAllViews` (main thread): records the camera yaw.
pub fn observe_view(requests: u64) {
    match crate::scene::read_pose(requests) {
        Some((_, _, f)) => {
            VIEW_YAW.store(f[1].atan2(f[0]).to_degrees().to_bits(), Ordering::Relaxed);
            VIEW_VALID.store(true, Ordering::Relaxed);
        }
        None => VIEW_VALID.store(false, Ordering::Relaxed),
    }
}

fn view_yaw() -> Option<f32> {
    VIEW_VALID
        .load(Ordering::Relaxed)
        .then(|| f32::from_bits(VIEW_YAW.load(Ordering::Relaxed)))
}

/// Angle difference wrapped to [-180, 180).
fn wrap180(d: f32) -> f32 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

fn calibrate(run: &mut Run, t: f32) {
    match run.cal {
        Cal::Idle if t >= CAL_START_S => {
            if let Some(yaw0) = view_yaw() {
                run.cal = Cal::Probing {
                    yaw0,
                    delivered0: mouse::delivered(),
                };
                mouse::add_x(CAL_COUNTS);
            }
        }
        Cal::Probing { yaw0, delivered0 } if t >= CAL_READ_S => {
            mouse::clear_pending();
            let counts = mouse::delivered() - delivered0;
            let turn = view_yaw().map_or(0.0, |y| wrap180(y - yaw0));
            run.gain = if counts == 0 {
                log::line(
                    "bench: mouse calibration counts were not read by the game; pans will not turn",
                );
                None
            } else if turn.abs() < 0.5 {
                log::line(&format!(
                    "bench: mouse calibration: {counts} counts turned the view {turn:.2} deg; pans will not turn"
                ));
                None
            } else {
                let g = turn / counts as f32;
                log::line(&format!(
                    "bench: mouse calibration: {counts} counts -> {turn:.2} deg ({g:.5} deg/count)"
                ));
                Some(g)
            };
            // Turn back for the rest of the warm-up.
            mouse::add_x(-(counts as i32));
            run.cal = Cal::Done;
        }
        _ => {}
    }
}

/// Feeds mouse counts so the view turns at a constant rate, 360 degrees over `secs`.
///
/// The game's response to injected counts is not linear: degrees per count depend on the
/// batch size and the frame rate (run 1792261049: calibration 0.044 deg/count, open-loop
/// pan 600 degrees; runs 1792263543/1792263688: 0.052 deg/count at ~148 fps, 0.068 at
/// ~200 fps). Earlier controllers either oscillated (per-frame position feedback, view
/// lag of a few frames) or changed speed in steps (gain re-estimated every 0.5 s while
/// the frame rate varied). This one controls the turn rate: it measures the smoothed
/// rate (`PAN_RATE_TAU`) and adapts the count rate multiplicatively with a slower time
/// constant (`PAN_ADAPT_TAU`), so changes in the response are absorbed continuously; a
/// bounded position term keeps the total near 360 degrees.
fn pan_step(run: &mut Run, t: f32, secs: f32) {
    let mut step = 0.0;
    if let Some(y) = view_yaw() {
        step = wrap180(y - run.pan.prev_yaw);
        run.pan.turned += step;
        run.pan.prev_yaw = y;
    }
    let Some(cal) = run.gain else { return };
    let add = run.pan.counts(cal, t, secs, step);
    if add != 0 {
        mouse::add_x(add as i32);
    }
}

impl Pan {
    /// Counts to inject this frame (also added to `requested`); `step` = degrees the view
    /// turned since the previous call, `cal` = calibrated degrees per count (sign = turn
    /// direction, magnitude = starting estimate).
    fn counts(&mut self, cal: f32, t: f32, secs: f32, step: f32) -> i64 {
        let dt = (t - self.last_t).clamp(0.0, 0.1);
        self.last_t = t;
        let nominal = 360.0 / secs * cal.signum();
        let start = nominal / cal;
        if self.counts_per_s == 0.0 {
            self.counts_per_s = start;
        }
        if dt <= 0.0 || t >= secs {
            return 0;
        }
        self.rate += (1.0 - (-dt / PAN_RATE_TAU).exp()) * (step / dt - self.rate);
        // Gentle pull towards the ramp position (bounded to +-25 % of the rate).
        let lag = nominal * t - self.turned;
        let bound = 0.25 * nominal.abs();
        let want = nominal + (lag / PAN_POSITION_TAU).clamp(-bound, bound);
        if t > PAN_RATE_TAU && self.rate * want > 0.0 {
            let ratio = (want / self.rate).clamp(0.5, 2.0);
            let k = (self.counts_per_s.abs() * ratio.powf(dt / PAN_ADAPT_TAU))
                .clamp(start.abs() * 0.25, start.abs() * 4.0);
            self.counts_per_s = k * start.signum();
        }
        let c = self.counts_per_s * dt + self.carry;
        let n = c.round();
        self.carry = c - n;
        let add = (n as i64).clamp(-PAN_MAX_STEP, PAN_MAX_STEP);
        self.requested += add;
        add
    }
}

/// Largest count batch per frame for the pan.
const PAN_MAX_STEP: i64 = 60;
/// Smoothing of the measured turn rate (s).
const PAN_RATE_TAU: f32 = 0.1;
/// Adaptation time constant of the count rate (s); slower than the rate smoothing.
const PAN_ADAPT_TAU: f32 = 0.25;
/// Time constant of the position correction (s).
const PAN_POSITION_TAU: f32 = 2.0;

fn profile_dir() -> Option<PathBuf> {
    Some(
        crate::game_dir()?
            .join("void-rs")
            .join("local")
            .join("profile"),
    )
}

/// Pose for `map`: the saved one, or the current camera (then saved).
/// Radius (m) within which a saved pose is reused for a new run (same scene).
const POSE_REUSE_M: f32 = 15.0;

/// Pose for this run: the saved pose of this map nearest to the camera if within
/// `POSE_REUSE_M`, else the current camera saved as a new scene `pose_<map>_<k>.txt`.
/// Returns (position, yaw, pitch, map, scene index).
fn pose(requests: u64) -> Option<([f32; 3], f32, f32, String, u32)> {
    let (map, org, fwd) = crate::scene::read_pose(requests)?;
    let safe: String = map
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let dir = profile_dir()?;
    let mut best: Option<(f32, [f32; 5], u32)> = None;
    let mut next = 0u32;
    for k in 0..64u32 {
        let Ok(t) = std::fs::read_to_string(dir.join(format!("pose_{safe}_{k}.txt"))) else {
            continue;
        };
        next = next.max(k + 1);
        let v: Vec<f32> = t
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        if v.len() < 5 {
            continue;
        }
        let d =
            ((v[0] - org[0]).powi(2) + (v[1] - org[1]).powi(2) + (v[2] - org[2]).powi(2)).sqrt();
        if d <= POSE_REUSE_M && best.is_none_or(|b| d < b.0) {
            best = Some((d, [v[0], v[1], v[2], v[3], v[4]], k));
        }
    }
    if let Some((d, v, k)) = best {
        log::line(&format!(
            "bench: reusing scene {k} pose ({d:.1} m from the camera)"
        ));
        return Some(([v[0], v[1], v[2]], v[3], v[4], map, k));
    }
    let yaw = fwd[1].atan2(fwd[0]).to_degrees();
    let pitch = -fwd[2].clamp(-1.0, 1.0).asin().to_degrees();
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join(format!("pose_{safe}_{next}.txt"));
    let _ = std::fs::write(
        &file,
        format!("{} {} {} {yaw} {pitch}\n", org[0], org[1], org[2]),
    );
    log::line(&format!(
        "bench: new scene {next}, pose saved to {}",
        file.display()
    ));
    Some((org, yaw, pitch, map, next))
}

fn start(requests: u64) -> Option<Run> {
    let s = settings::get();
    let (pos, yaw, pitch, map, scene) = pose(requests)?;
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let label: String = s
        .bench_label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = profile_dir()?.join(if label.is_empty() {
        format!("run_{secs}_scene{scene}")
    } else {
        format!("run_{secs}_scene{scene}_{label}")
    });
    std::fs::create_dir_all(&dir).ok()?;
    let mut segs = vec![Seg {
        kind: SegKind::Warmup,
        rep: 0,
        // Room for the mouse calibration.
        secs: s.bench_warmup_s.max(CAL_READ_S + 0.5),
    }];
    for rep in 1..=s.bench_repeats {
        segs.push(Seg {
            kind: SegKind::Settle,
            rep,
            secs: 2.0,
        });
        segs.push(Seg {
            kind: SegKind::Static,
            rep,
            secs: s.bench_static_s,
        });
        segs.push(Seg {
            kind: SegKind::Pan,
            rep,
            secs: s.bench_pan_s,
        });
    }
    let cap_restore = if s.bench_uncap {
        cvars::set_int("com_disableFrameSpinning", 1)
    } else {
        None
    };
    let meta_dir = dir.clone();
    let meta_map = map.clone();
    std::thread::spawn(move || write_meta(&meta_dir, &meta_map, pos, yaw, pitch));
    setviewpos(pos, yaw, pitch);
    *LAST_DIR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(dir.clone());
    log::line(&format!(
        "bench: started in {} (map {map}, pose {pos:?} yaw {yaw:.1} pitch {pitch:.1}, {} segments)",
        dir.display(),
        segs.len()
    ));
    crate::notify::info(format!(
        "Benchmark '{}': warming up ({:.0} s), then {} x (static {:.0} s + 360° pan {:.0} s)",
        s.bench_label, s.bench_warmup_s, s.bench_repeats, s.bench_static_s, s.bench_pan_s
    ));
    Some(Run {
        dir,
        segs,
        idx: 0,
        seg_start: Instant::now(),
        seg_first_frame: profiler::current_frame(),
        pos,
        yaw,
        pitch,
        cap_restore,
        cal: Cal::Idle,
        gain: None,
        pan: Pan::default(),
    })
}

fn finish(run: Run, aborted: bool) {
    mouse::clear_pending();
    if profiler::is_recording() {
        let _ = profiler::stop_recording();
    }
    if let Some(v) = run.cap_restore {
        cvars::set_int("com_disableFrameSpinning", v);
    }
    log::line(&format!(
        "bench: {} ({})",
        if aborted { "aborted" } else { "finished" },
        run.dir.display()
    ));
    let name = run
        .dir
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    if aborted {
        crate::notify::warn(format!("Benchmark aborted ({name})"));
    } else {
        crate::notify::info(format!("Benchmark finished: {name}"));
    }
}

/// Called at every `RenderFrame` entry (main thread).
pub fn step(frame: u64) {
    let mut g = RUN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if TOGGLE.swap(false, Ordering::Relaxed) {
        match g.take() {
            Some(run) => {
                finish(run, true);
                return;
            }
            None => *g = start(REQUESTS.load(Ordering::Relaxed)),
        }
    }
    let Some(run) = g.as_mut() else { return };
    let seg = run.segs[run.idx];
    let t = run.seg_start.elapsed().as_secs_f32();
    if t >= seg.secs {
        // Segment done.
        if matches!(seg.kind, SegKind::Static | SegKind::Pan) {
            let frames = profiler::stop_recording();
            let pass_events = profiler::take_pass_events();
            if seg.kind == SegKind::Pan {
                mouse::clear_pending();
                log::line(&format!(
                    "bench: r{}_pan turned {:.1} deg (mouse counts requested {}, read by the game {})",
                    seg.rep,
                    run.pan.turned,
                    run.pan.requested,
                    mouse::delivered() - run.pan.delivered0
                ));
            }
            let name = format!(
                "r{}_{}",
                seg.rep,
                if seg.kind == SegKind::Static {
                    "static"
                } else {
                    "pan"
                }
            );
            let (dir, range) = (run.dir.clone(), (run.seg_first_frame, frame));
            std::thread::spawn(move || {
                // GPU results resolve a few frames after the segment.
                std::thread::sleep(std::time::Duration::from_millis(500));
                let gpu = profiler::take_gpu_results(range.0, range.1);
                if let Err(e) = write_segment(&dir, &name, &frames, &gpu) {
                    log::line(&format!("bench: write {name} failed: {e}"));
                }
                if let Err(e) = write_passes(&dir, &name, &pass_events) {
                    log::line(&format!("bench: write {name} passes failed: {e}"));
                }
            });
        }
        run.idx += 1;
        if run.idx >= run.segs.len() {
            let run = g.take().unwrap_or_else(|| unreachable!());
            finish(run, false);
            return;
        }
        run.seg_start = Instant::now();
        run.seg_first_frame = frame;
        let next = run.segs[run.idx];
        match next.kind {
            SegKind::Settle => {
                mouse::clear_pending();
                setviewpos(run.pos, run.yaw, run.pitch);
                // Shown during the settle (the overlay draws nothing while recording).
                let reps = run.segs.iter().map(|s| s.rep).max().unwrap_or(0);
                crate::notify::info(format!(
                    "Benchmark: repeat {} of {reps} (static, then 360° pan)",
                    next.rep
                ));
            }
            SegKind::Static => profiler::start_recording(),
            SegKind::Pan => {
                mouse::clear_pending();
                run.pan = Pan {
                    delivered0: mouse::delivered(),
                    prev_yaw: view_yaw().unwrap_or(run.yaw),
                    ..Pan::default()
                };
                if run.gain.is_none() {
                    log::line(&format!(
                        "bench: r{}_pan without turning (no mouse calibration)",
                        next.rep
                    ));
                    crate::notify::warn(format!(
                        "Benchmark: r{} pan without turning (mouse calibration failed)",
                        next.rep
                    ));
                }
                profiler::start_recording();
            }
            SegKind::Warmup => {}
        }
        return;
    }
    match seg.kind {
        SegKind::Warmup => calibrate(run, t),
        SegKind::Pan => pan_step(run, t, seg.secs),
        SegKind::Settle | SegKind::Static => {}
    }
}

// ---------------------------------------------------------------------- output --

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Summary row: n, mean, p50, p90, p99, p99.9, max.
pub fn stats(v: &[f64]) -> [f64; 7] {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    s.sort_by(f64::total_cmp);
    let n = s.len() as f64;
    let mean = if s.is_empty() {
        f64::NAN
    } else {
        s.iter().sum::<f64>() / n
    };
    [
        n,
        mean,
        pct(&s, 0.5),
        pct(&s, 0.9),
        pct(&s, 0.99),
        pct(&s, 0.999),
        s.last().copied().unwrap_or(f64::NAN),
    ]
}

/// Frames of each segment written raw to `<segment>_timeline.tsv`.
const TIMELINE_FRAMES: u64 = 60;

/// Executable address of an RVA for the tables (0 stays 0: no pass callback).
fn exe_addr(rva: u32) -> u64 {
    if rva == 0 {
        0
    } else {
        0x1_4000_0000 + u64::from(rva)
    }
}

/// Writes the render-graph tables of one segment: `<segment>_passes.tsv` (one row per
/// pass callback and event kind; times relative to the frame's first `JobSys_KickWorkers`)
/// and `<segment>_timeline.tsv` (raw events of the first `TIMELINE_FRAMES` frames).
fn write_passes(dir: &std::path::Path, name: &str, events: &[PassEv]) -> Result<(), String> {
    use std::collections::{BTreeMap, HashMap, HashSet};
    let mut kick: HashMap<u64, u64> = HashMap::new();
    for e in events.iter().filter(|e| e.kind == PassKind::Kick) {
        kick.entry(e.frame)
            .and_modify(|t| *t = (*t).min(e.t0_ns))
            .or_insert(e.t0_ns);
    }
    let frames = kick.len().max(1) as f64;
    #[derive(Default)]
    struct Row {
        dur: Vec<f64>,
        start: Vec<f64>,
        end: Vec<f64>,
        threads: HashSet<u32>,
        flags: u32,
    }
    let ms = |ns: u64| ns as f64 / 1e6;
    let rel = |t: u64, k: u64| (t as f64 - k as f64) / 1e6;
    let mut rows: BTreeMap<(&str, u32), Row> = BTreeMap::new();
    let first = events.iter().map(|e| e.frame).min().unwrap_or(0);
    let mut timeline = String::from("frame\tkind\tcallback\tflags\tthread\tstart_ms\tend_ms\n");
    for e in events {
        let Some(&k) = kick.get(&e.frame) else {
            continue;
        };
        let r = rows.entry((e.kind.name(), e.callback_rva)).or_default();
        r.dur.push(ms(e.t1_ns.saturating_sub(e.t0_ns)));
        r.start.push(rel(e.t0_ns, k));
        r.end.push(rel(e.t1_ns, k));
        r.threads.insert(e.thread);
        r.flags |= e.flags;
        if e.frame < first + TIMELINE_FRAMES {
            let _ = writeln!(
                timeline,
                "{}\t{}\t0x{:x}\t0x{:x}\t{}\t{:.4}\t{:.4}",
                e.frame,
                e.kind.name(),
                exe_addr(e.callback_rva),
                e.flags,
                e.thread,
                rel(e.t0_ns, k),
                rel(e.t1_ns, k)
            );
        }
    }
    let mut t = String::from(
        "kind\tcallback\tflags\tper_frame\tms_per_frame\tdur_p50_ms\tdur_p99_ms\tdur_max_ms\
         \tstart_p50_ms\tend_p50_ms\tend_p99_ms\tthreads\n",
    );
    // stats(): [n, mean, p50, p90, p99, p99.9, max]; rows ordered by median end time.
    let mut sorted: Vec<_> = rows
        .into_iter()
        .map(|(k, r)| (k, stats(&r.dur), stats(&r.start), stats(&r.end), r))
        .collect();
    sorted.sort_by(|a, b| a.3[2].total_cmp(&b.3[2]));
    for ((kind, cb), d, s, en, r) in &sorted {
        let _ = writeln!(
            t,
            "{kind}\t0x{:x}\t0x{:x}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
            exe_addr(*cb),
            r.flags,
            r.dur.len() as f64 / frames,
            r.dur.iter().sum::<f64>() / frames,
            d[2],
            d[4],
            d[6],
            s[2],
            en[2],
            en[4],
            r.threads.len()
        );
    }
    std::fs::write(dir.join(format!("{name}_passes.tsv")), t).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}_timeline.tsv")), timeline).map_err(|e| e.to_string())
}

fn write_segment(
    dir: &std::path::Path,
    name: &str,
    frames: &[FrameRec],
    gpu: &[GpuFrame],
) -> Result<(), String> {
    let names = profiler::phase_names();
    let gpu_by_frame: std::collections::HashMap<u64, &GpuFrame> =
        gpu.iter().map(|g| (g.frame, g)).collect();
    let mut csv = String::from(
        "frame\tt_ms\tinterval_ms\teng_frame_ms\teng_cpu_ms\teng_swap_ms\teng_gpu_ms\teng_render_frame_ms\teng_front_end_ms\
         \tportal_query_ms\tlight_query_ms\tshadow_casters_query_ms\tcpu_render_ms\twait_gpu_submits_ms\
         \tmodel_commits\tlight_commits\tc2\tc3\tapprox_light_models\ttrans_sort_quads\ttrans_add_quads",
    );
    for n in &names {
        let _ = write!(csv, "\t{n}_ms\t{n}_n");
    }
    csv += "\tgpu_span_ms\tgpu_lists_ms\tgpu_lists_n";
    for c in crate::passes::CLASS_NAMES {
        let _ = write!(csv, "\tgpu_{c}_ms");
    }
    csv += "\n";
    let mut pass_cols: Vec<Vec<f64>> = vec![Vec::new(); crate::passes::CLASS_NAMES.len()];
    let mut cols: Vec<(&str, Vec<f64>)> = vec![
        ("interval_ms", vec![]),
        ("eng_cpu_ms", vec![]),
        ("eng_swap_ms", vec![]),
        ("eng_gpu_ms", vec![]),
        ("eng_front_end_ms", vec![]),
        ("eng_render_frame_ms", vec![]),
        ("cpu_render_ms", vec![]),
        ("wait_gpu_submits_ms", vec![]),
        ("gpu_span_ms", vec![]),
        ("gpu_lists_ms", vec![]),
    ];
    let mut phase_cols: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut lists = String::from("frame\tlabel\tms\n");
    for f in frames {
        let ms = |us: f64| us / 1000.0;
        let eng_cpu = ms(f.eng_frame_us as f64 - f64::from(f.eng_swap_us));
        let _ = write!(
            csv,
            "{}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            f.frame,
            f.t_ms,
            f.interval_ms,
            ms(f.eng_frame_us as f64),
            eng_cpu,
            ms(f64::from(f.eng_swap_us)),
            ms(f64::from(f.eng_gpu_us)),
            ms(f64::from(f.eng_render_frame_us)),
            ms(f.eng_front_end_us as f64)
        );
        for v in f.eng_stats_us {
            let _ = write!(csv, "\t{:.4}", ms(v as f64));
        }
        for c in f.eng_counts {
            let _ = write!(csv, "\t{c}");
        }
        for p in 0..names.len() {
            let _ = write!(csv, "\t{:.4}\t{}", f.phase_ms[p], f.phase_count[p]);
            phase_cols[p].push(f.phase_ms[p]);
        }
        let g = gpu_by_frame.get(&f.frame);
        match g {
            Some(g) => {
                let _ = write!(
                    csv,
                    "\t{:.4}\t{:.4}\t{}",
                    g.span_ms,
                    g.lists_ms,
                    g.lists.len()
                );
                for (i, v) in g.passes.iter().enumerate() {
                    let _ = write!(csv, "\t{v:.4}");
                    pass_cols[i].push(*v);
                }
                csv += "\n";
                for (l, m) in &g.lists {
                    let _ = writeln!(
                        lists,
                        "{}\t{}\t{m:.4}",
                        f.frame,
                        if *l == u32::MAX { -1 } else { *l as i64 }
                    );
                }
            }
            None => {
                csv += "\t\t\t";
                for _ in crate::passes::CLASS_NAMES {
                    csv += "\t";
                }
                csv += "\n";
            }
        }
        let vals = [
            f.interval_ms,
            eng_cpu,
            ms(f64::from(f.eng_swap_us)),
            ms(f64::from(f.eng_gpu_us)),
            ms(f.eng_front_end_us as f64),
            ms(f64::from(f.eng_render_frame_us)),
            ms(f.eng_stats_us[3] as f64),
            ms(f.eng_stats_us[4] as f64),
            g.map_or(f64::NAN, |g| g.span_ms),
            g.map_or(f64::NAN, |g| g.lists_ms),
        ];
        for (c, v) in cols.iter_mut().zip(vals) {
            c.1.push(v);
        }
    }
    std::fs::write(dir.join(format!("{name}_frames.tsv")), csv).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}_gpu_lists.tsv")), lists).map_err(|e| e.to_string())?;
    let mut sum = String::new();
    let path = dir.join("summary.tsv");
    if !path.exists() {
        sum += "segment\tmetric\tn\tmean\tp50\tp90\tp99\tp99.9\tmax\n";
    }
    let mut row = |metric: &str, v: &[f64]| {
        let s = stats(v);
        let _ = writeln!(
            sum,
            "{name}\t{metric}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
            s[0], s[1], s[2], s[3], s[4], s[5], s[6]
        );
    };
    for (m, v) in &cols {
        row(m, v);
    }
    for (p, v) in phase_cols.iter().enumerate() {
        row(&format!("{}_ms", names[p]), v);
    }
    for (i, v) in pass_cols.iter().enumerate() {
        if !v.is_empty() {
            row(&format!("gpu_{}_ms", crate::passes::CLASS_NAMES[i]), v);
        }
    }
    let gpu_frames = gpu.len();
    let _ = writeln!(
        sum,
        "{name}\tgpu_frames_resolved\t{gpu_frames}\t\t\t\t\t\t\n{name}\tgpu_frames_dropped_total\t{}\t\t\t\t\t\t",
        profiler::GPU_DROPPED.load(Ordering::Relaxed)
    );
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    f.write_all(sum.as_bytes()).map_err(|e| e.to_string())?;
    // Label table (cumulative for the run).
    let mut lt = String::from("label\texe_rvas\n");
    for (i, l) in profiler::labels().iter().enumerate() {
        let rv: Vec<String> = l
            .iter()
            .map(|r| format!("0x{:x}", 0x1_4000_0000u64 + u64::from(*r)))
            .collect();
        let _ = writeln!(lt, "{i}\t{}", rv.join(" "));
    }
    std::fs::write(dir.join("labels.tsv"), lt).map_err(|e| e.to_string())?;
    let threads: Vec<String> = profiler::thread_ids()
        .into_iter()
        .map(|(n, t)| format!("{n}	{t}"))
        .collect();
    std::fs::write(
        dir.join("threads.tsv"),
        format!(
            "phase	thread_id
{}
",
            threads.join(
                "
"
            )
        ),
    )
    .map_err(|e| e.to_string())?;
    log::line(&format!(
        "bench: {name}: {} frames, {gpu_frames} with GPU data",
        frames.len()
    ));
    Ok(())
}

fn cpu_brand() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut b = Vec::new();
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            let r = std::arch::x86_64::__cpuid(leaf);
            for v in [r.eax, r.ebx, r.ecx, r.edx] {
                b.extend(v.to_le_bytes());
            }
        }
        String::from_utf8_lossy(&b)
            .trim_matches(char::from(0))
            .trim()
            .to_owned()
    }
    #[cfg(not(target_arch = "x86_64"))]
    String::new()
}

fn gpu_info() -> String {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIDevice, IDXGIFactory1};
    use windows::core::Interface;
    // SAFETY: plain DXGI factory / adapter queries.
    unsafe {
        let Ok(f) = CreateDXGIFactory1::<IDXGIFactory1>() else {
            return "unknown".into();
        };
        let Ok(a) = f.EnumAdapters1(0) else {
            return "unknown".into();
        };
        let Ok(d) = a.GetDesc1() else {
            return "unknown".into();
        };
        let name = String::from_utf16_lossy(&d.Description)
            .trim_matches(char::from(0))
            .to_owned();
        let ver = a
            .CheckInterfaceSupport(&IDXGIDevice::IID)
            .map(|v| {
                let v = v as u64;
                format!(
                    "{}.{}.{}.{}",
                    v >> 48,
                    (v >> 32) & 0xFFFF,
                    (v >> 16) & 0xFFFF,
                    v & 0xFFFF
                )
            })
            .unwrap_or_else(|_| "?".into());
        format!(
            "{name} (UMD {ver}, vendor {:04x} device {:04x}, {} MiB VRAM)",
            d.VendorId,
            d.DeviceId,
            d.DedicatedVideoMemory >> 20
        )
    }
}

fn modules() -> String {
    use windows::Win32::System::Diagnostics::ToolHelp::*;
    let mut out = String::new();
    // SAFETY: snapshot of the current process's modules; entries read with the correct
    // dwSize and the handle closed.
    unsafe {
        let Ok(h) = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0) else {
            return out;
        };
        let mut e = MODULEENTRY32W {
            dwSize: std::mem::size_of::<MODULEENTRY32W>() as u32,
            ..Default::default()
        };
        let mut ok = Module32FirstW(h, &mut e).is_ok();
        while ok {
            let path = String::from_utf16_lossy(&e.szExePath);
            let _ = writeln!(out, "{}", path.trim_matches(char::from(0)));
            ok = Module32NextW(h, &mut e).is_ok();
        }
        let _ = windows::Win32::Foundation::CloseHandle(h);
    }
    out
}

fn write_meta(dir: &std::path::Path, map: &str, pos: [f32; 3], yaw: f32, pitch: f32) {
    let s = settings::get();
    let base = mem::image_base().unwrap_or(0);
    let scale = mem::f32s::<2>(base + 0x22F_D9F0).unwrap_or([f32::NAN; 2]);
    let hwnd = mem::u64_at(base + 0x313_7F40).unwrap_or(0);
    let mut client = String::from("?");
    if hwnd != 0 {
        let mut r = windows::Win32::Foundation::RECT::default();
        // SAFETY: the game's window handle (global 0x143137F40, d2enhanced NOTES).
        let ok = unsafe {
            windows::Win32::UI::WindowsAndMessaging::GetClientRect(
                windows::Win32::Foundation::HWND(hwnd as *mut _),
                &mut r,
            )
        }
        .is_ok();
        if ok {
            client = format!("{}x{}", r.right - r.left, r.bottom - r.top);
        }
    }
    let game_dlls: Vec<String> = crate::game_dir()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| {
                    let l = n.to_ascii_lowercase();
                    l.ends_with(".dll")
                        || l.ends_with(".ini")
                        || l.ends_with(".asi")
                        || l.ends_with(".conf")
                })
                .collect()
        })
        .unwrap_or_default();
    let meta = format!(
        "label {}\nexe sha256 {}\ngpu {}\ncpu {}\nwindow client {client}\nresolution scale {scale:?}\nmap {map}\npose {pos:?} yaw {yaw} pitch {pitch}\n\
         game folder files {game_dlls:?}\nsettings {s:?}\ncapture: in-process (void-inject profiler); frame limiter off during run: {}\n\
         pan: synthetic mouse look (DirectInput X counts; gain measured in the warm-up)\n",
        s.bench_label,
        crate::EXPECTED_EXE_SHA256,
        gpu_info(),
        cpu_brand(),
        s.bench_uncap
    );
    let _ = std::fs::write(dir.join("meta.txt"), meta);
    if let Ok((tsv, _)) = cvars::dump_tsv() {
        let _ = std::fs::write(dir.join("cvars.tsv"), tsv);
    }
    let _ = std::fs::write(dir.join("modules.txt"), modules());
}

#[cfg(test)]
mod tests {
    use super::{Pan, wrap180};

    /// Simulated game: counts are read one frame after they are queued and the view
    /// turns `lag` frames later. The frame rate swings between `fps_lo` and `fps_hi`
    /// (period 8 s, about the time a 20 s pan spends passing heavy or light scenery) and
    /// degrees per count follow the frame rate as measured in game (0.052 at ~148 fps,
    /// 0.068 at ~200 fps: ~fps^0.85), scaled by `ratio` against the calibration. Returns
    /// (final turn, turn rates over 0.25 s windows after 2 s).
    fn simulate(fps_lo: f32, fps_hi: f32, ratio: f32, lag: usize) -> (f32, Vec<f32>) {
        let (cal, secs) = (-0.044f32, 20.0f32);
        let mut p = Pan::default();
        let mut queued = 0i64;
        let mut visible = std::collections::VecDeque::from(vec![0.0f32; lag]);
        let (mut t, mut window, mut window_t, mut rates) = (0.0f32, 0.0f32, 0.0f32, Vec::new());
        while t < secs + 0.2 {
            let fps =
                fps_lo + (fps_hi - fps_lo) * 0.5 * (1.0 + (t * std::f32::consts::TAU / 8.0).sin());
            let deg_per_count = cal * ratio * (fps / 148.0).powf(0.85);
            visible.push_back(queued as f32 * deg_per_count);
            let step = visible.pop_front().unwrap_or(0.0);
            p.turned += step;
            if t > 2.0 && t < secs - 0.5 {
                window += -step;
                window_t += 1.0 / fps;
                if window_t >= 0.25 {
                    rates.push(window / window_t);
                    (window, window_t) = (0.0, 0.0);
                }
            }
            queued = p.counts(cal, t.min(secs), secs, step);
            t += 1.0 / fps;
        }
        (-p.turned, rates)
    }

    #[test]
    fn pan_turns_at_a_steady_rate_while_the_response_changes() {
        let nominal = 360.0 / 20.0;
        for (lo, hi, ratio, lag) in [
            (120.0, 220.0, 1.0, 2),
            (120.0, 220.0, 1.6, 3),
            (140.0, 160.0, 0.6, 3),
            (55.0, 70.0, 1.6, 2),
        ] {
            let (turned, rates) = simulate(lo, hi, ratio, lag);
            assert!(
                (turned - 360.0).abs() < 8.0,
                "{lo}-{hi} fps x{ratio}: turned {turned}"
            );
            let (min, max) = rates
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &r| (a.min(r), b.max(r)));
            assert!(
                min > 0.88 * nominal && max < 1.12 * nominal,
                "{lo}-{hi} fps x{ratio}: rate {min:.2}..{max:.2} deg/s (nominal {nominal})"
            );
        }
    }

    #[test]
    fn yaw_steps_across_the_atan2_seam_accumulate_to_a_full_turn() {
        // atan2 yaw jumps from +179 to -179 when the view passes the -X axis.
        assert!((wrap180(-179.0 - 179.0) - 2.0).abs() < 1e-4);
        assert!((wrap180(179.0 - -179.0) + 2.0).abs() < 1e-4);
        let mut turned = 0.0f32;
        let mut prev = 170.0f32;
        for k in 1..=360 {
            let yaw = wrap180(170.0 + k as f32);
            turned += wrap180(yaw - prev);
            prev = yaw;
        }
        assert!((turned - 360.0).abs() < 1e-2, "{turned}");
    }
}
