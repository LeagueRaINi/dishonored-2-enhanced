# Performance baseline (P1)

Status 2026-10-17: native-renderer baseline recorded for two eotw scenes (stationary
segments valid, pan segments superseded, see Limitations). Comparisons against d2enhanced
/ DXVK, profiler on-vs-off, and PresentMon cross-checks are still open.

## Conditions

| Item | Value |
| --- | --- |
| Executable | `Dishonored2.exe` SHA-256 `b671f824...6180` (E001) |
| GPU / driver | AMD Radeon RX 7900 XTX, UMD 32.0.32015.2008, 24 GiB |
| CPU | AMD Ryzen 7 7700X |
| Output | 3440x1440 fullscreen (`r_mode 19`), render scale 1.0 (`rs_enable 0`) |
| Graphics | in-game Ultra; AA = TAA (`r_AntiAliasingPreset 3`); `r_oceanQualityPreset 5`; `image_anisotropy 1`; `r_lodZoomMultiplicator 0.6`; full cvar dump per run in `cvars.tsv` |
| Frame cap / VSync | limiter off during recording (`com_disableFrameSpinning 1`, restored after), `r_swapInterval 0` |
| API path | native D3D11 (AMD driver replays deferred command lists on the immediate context) |
| Mods | `dinput8.dll` = void-inject with `profile = 1`; d2enhanced disabled (`d3d11.bak.dll`); FidelityFX DLLs present but unused |
| Map | `maps/campaign/edge_of_the_world/eotw_p_highchaos` |

## Reproduction

1. `void-inject.ini`: `profile = 1`, `profile_gpu = 1` (runs before 2026-10-17 also had
   `experimental_renderer = 0`; that renderer was removed),
   `bench_warmup_s = 10`, `bench_static_s = 20`, `bench_pan_s = 20`, `bench_repeats = 3`,
   `bench_uncap = 1`.
2. Load the map, stand at the pose in `local/profile/pose_<map>_<n>.txt` (written on first
   use; scene 0 `-33.65 181.69 10.46 yaw -49.25 pitch 13.72`, scene 1
   `-25.68 102.71 12.17 yaw -52.37 pitch 0.49`).
3. Ctrl+F3: warm-up, then per repeat settle -> 20 s stationary -> 360 degree pan
   (`crates/void-inject/src/bench.rs`). Output: `local/profile/run_<time>_<scene>_<label>/`.

Measurement sources (`crates/void-inject/src/profiler.rs`): engine timers (frame, swap,
engine GPU query), QPC around engine phases and the immediate context's
`ExecuteCommandList` / deferred `FinishCommandList`, asynchronous D3D11 timestamp queries
per command list and per pass class (`passes.rs`), read several frames late with
`DONOTFLUSH`; unready/disjoint frames are dropped and counted (0 dropped in all runs below).

## Results: stationary segments (ms, p50 / p99, three repeats each)

| Metric | Scene 0 (run 1792194132) | Scene 1 (run 1792194435) |
| --- | --- | --- |
| Frame interval | 8.21-8.28 / 9.63-9.74 | 8.10-8.13 / 9.43-9.58 |
| Engine CPU frame | 7.90-7.99 / 9.32-9.50 | 7.66-7.69 / 9.05-9.20 |
| Engine GPU frame | 5.91-5.96 / 7.26-7.37 | 5.70-5.82 / 6.84-6.93 |
| `render_views` (CPU, main thread) | 5.23-5.28 | 5.17-5.19 |
| `ExecuteCommandList` (CPU, submit + driver replay) | 1.90-1.91 | 1.94-1.95 |
| `FinishCommandList` (CPU, summed over workers) | 0.64 | 0.62-0.63 |
| GPU opaque lit (forward) | 2.25-2.29 | 1.49-1.58 |
| GPU shadows | 0.67-0.68 | 0.96-0.97 |
| GPU depth/velocity prepass | 0.68-0.69 | 0.51-0.52 |
| GPU post (after TAA) | 0.65 | 0.64 |
| GPU ocean colour / velocity | 0.35 / 0.03 | 0.08 / 0.003 |
| GPU TAA | 0.13 | 0.13 |

Run 1792193254 (scene 0, same pose, earlier build) agrees: interval p50 8.12-8.14, CPU
7.82-7.84, GPU 5.87-5.96. Run 1792191643 is a different map/pose (tutorial), not comparable.

Reading: the frame is CPU-limited at ~122 fps (CPU frame ~= interval, GPU ~2 ms shorter).
The main-thread `render_views` phase dominates; the AMD replay of deferred command lists
costs ~1.9 ms on the submit thread. GPU-side savings do not raise the frame rate on this
machine until the CPU side changes; they do create headroom for added passes (FSR 4,
lighting) at little or no frame-rate cost. [INFERENCE until measured with each change.]

## FSR 4 native AA A/B (2026-10-17, one session, scene 0, mean of r1-r3 p50, ms)

Runs `run_1792263543_scene0_fsr` (FSR on), `run_1792263688_scene0_taa` (FSR installed, off
via F6; overlay and `image_anisotropy 16` active), `run_1792264026_scene0_native` ("clean":
`fsr = 0`, `overlay = 0`, no cvar overrides; profiler on in all three).

| Segment / metric | FSR on | FSR off | clean |
| --- | --- | --- | --- |
| static frame interval | 9.51 | 8.87 | 8.93 |
| static engine CPU | 9.28 | 8.58 | 8.65 |
| static engine GPU | 6.72 | 6.57 | 6.67 |
| static `render_views` | 6.68 | 6.00 | 6.03 |
| pan frame interval | 7.02 | 5.32 | 5.32 |
| pan engine GPU | 6.07 | 4.11 | 4.13 |
| pan `present` (GPU wait) | 0.99 | 0.12 | 0.11 |

- Our hooks, overlay, cvar override and the installed-but-off FSR cost nothing measurable
  (FSR off vs clean within run noise).
- FSR on, static (CPU-bound): +0.64 ms, matching the FSR submit-path CPU time logged in the
  same session (p50 0.3-0.65 ms: copy pass + D3D11 signal + flush 0.2-0.5, D3D12 submit
  0.1). It runs inside the engine's immediate-context lock, i.e. on the frame's critical path.
- FSR on, pan (lighter CPU load): +1.7 ms; the frame becomes GPU-bound (`present` waits)
  and FSR's own 1.8 ms GPU time adds up.
- All three runs are ~0.7 ms slower (CPU and GPU) than the 2026-10-16 baseline at the same
  pose, including the clean run: different game state; compare only within one session.
- Not yet measured: profiler on vs off (the D3D11 trace hooks wrap every replayed call).

## FSR upscale and `fsr_flush = 0` (2026-10-17, mean of r1-r3 p50, ms)

Runs `run_1792268990_scene0_fsr_flush_off` (upscale 0.67, `fsr_flush = 0`) and
`run_1792269294_scene0_fsr_flush_off` (native AA, `fsr_flush = 0`; one session; both
labelled `fsr_flush_off`, modes from `meta.txt`), against `run_1792266923_scene0_fsr` (native,
flush on, earlier session).

| Segment / metric | native, flush on | native, flush off | upscale, flush off |
| --- | --- | --- | --- |
| static frame interval | 9.49 | 9.43 | 9.61 |
| static engine CPU | 9.25 | 9.19 | 9.35 |
| static engine GPU (span) | 6.75 | 6.72 | 6.26 |
| pan frame interval | 6.84 | 6.73 | 5.80 |
| pan engine GPU (span) | 6.00 | 5.99 | 4.52 |
| pan `present` (GPU wait) | 1.05 | 1.17 | 0.12 |

- `fsr_flush = 0`: no measurable change (-0.06 ms static, across sessions); FSR's logged
  pack+flush CPU stays 0.18-0.53 ms p50 without the explicit `Flush`, so the cost sits in the
  D3D11 copy/`Signal` on the immediate context, not in the `Flush` call. Default unchanged.
- Upscale removes the GPU limit while panning (-0.93 ms, `present` wait gone); no gain in
  the CPU-bound static view.
- The engine GPU time in CPU-bound frames is a span, not GPU load: it tracks the serial
  submission below (~6.1 ms), so GPU-side savings do not show there.

## Render-graph critical path (2026-10-17, `r1_static_timeline.tsv`, 60 frames, medians)

Run `run_1792269294` (native FSR). Profiler events: `JobSys_KickWorkers`/`JobSys_WaitWorkers`
(main thread), `RenderPass_Record` (workers), `RenderPass_Submit` (in pass order, one at a
time under the immediate lock; on AMD this is the runtime's command-list replay). Pass names
from the registration table in `sub_14034FE00` (callback `lea` followed by the
`ARK_RENDERPASS_*` name; `local/profile/pass_names.json`). Times in ms after the kick.

- Main thread waits 6.23 ms. Workers record 18.1 ms in total, all done by 3.48 ms (8
  workers ~65 % busy). Submission is busy 3.54 ms and idle 2.51 ms waiting for records;
  it ends at 6.13-6.17 ms.
- Chain: `GATHER_UMBRA` 0-0.85 -> `SORT` 1.14-1.87 -> `DEPTHPASS` record 2.09-2.72 -> serial
  submission from 2.72 to 6.13: `DEPTHPASS` 0.71, `SHADOWS_WORLD` 0.90, `SHADOWS_DYNAMIC`
  1.03, `FORWARD_WORLD` 0.49 (recorded 2.19-3.48), `FORWARD_DYNAMIC` 0.27, `POSTPROC` 0.59
  (includes FSR), the rest < 0.1 each.
- Largest recording costs: chunk jobs of `sub_140328070` (20.7 per frame, 4.8 ms),
  `SHADOWS_DYNAMIC` + `SHADOWS_WORLD` (16 lists each, 4.6 ms), `FORWARD_WORLD` 1.29,
  `DEPTHPASS` 1.23 (3 lists).
- Reading: the frame is bound by the in-order serial submission after `DEPTHPASS`; shadows
  are recorded early (from 1.1 ms) but submitted only after the depth pre-pass. [H] Submitting
  the shadow lists before `DEPTHPASS` (no shared targets; GPU order shadows -> depth ->
  forward stays valid) would overlap 1.9 ms of replay with recording; bound from these
  numbers ~1 ms per frame. Needs the pass-order/dependency mechanism of the job system (RE).
  FSR's ~0.6 ms sits at the very end of this chain.

## `patch.shadows_submit_early` A/B (2026-10-17, one session, FSR upscale 0.67)

Runs `run_1792272113_scene0_native` (patch off) and `run_1792272267_scene0_native` (patch on,
switched live in between; label default, patch state from `meta.txt`). p50 per repeat, ms:

| Metric | off r1 / r2 / r3 | on r1 / r2 / r3 | change (mean) |
| --- | --- | --- | --- |
| static frame interval | 9.71 / 9.79 / 9.77 | 9.30 / 9.29 / 9.29 | -0.47 (-4.8 %) |
| static engine CPU | 9.46 / 9.54 / 9.50 | 9.06 / 9.03 / 9.02 | -0.46 |
| static `graph_wait` | 6.32 / 6.36 / 6.36 | 5.88 / 5.87 / 5.87 | -0.48 |
| pan frame interval | 5.82 / 5.77 / 5.80 | 5.86 / 5.87 / 5.84 | +0.06 |

- Static p99 interval 10.61 -> 10.48; pan p99 9.98 -> 9.55.
- Timeline (r1_static, medians): shadow submissions now run 1.17-4.22 ms interleaved with the
  rest (before: 3.54-4.71 after `DEPTHPASS`); `DEPTHPASS` submits slightly later (3.04-4.11
  instead of 2.77-3.51) because it shares the submission thread; `FORWARD_WORLD` starts at 4.45
  instead of 4.73; the last submission ends at 6.10 instead of 6.35.
- Pan (lighter CPU load, closer to GPU-bound): no change.
- Image: unchanged (user check, toggled live several times) -> on by default.
- Remaining gaps in the submission (patched run, r1_static): before `UPDATE_OCEAN_FFT`
  0.13 -> 1.03 ms (nothing recorded yet), before shadow lists still being recorded
  (0.56 + 0.28 ms per frame), before `INDIRECT_LIGHTING_UPDATE_IBL` (0.36). All are waits for
  recording, not for other submissions; after the last record (3.82 ms) 2.3 ms of serial
  replay remain. No further submit edge promises a measurable gain.

## Limitations

- Pan segments of 1792194132/1792194435 used a render-view override (hands, weapon and HUD
  markers did not turn; rejected by the user). The pan now injects mouse-look counts
  (`mouse.rs`); no run with it yet. Pan rows of these runs are not used.
- Profiler overhead not measured (on vs off A/B open). PresentMon cross-check folder is
  empty: not done.
- d2enhanced not measured with this driver. Its notes report ~90 fps native vs ~170 fps
  DXVK at another location (lead, not corroborated). DXVK is out of scope (user, 2026-10-17):
  the CPU work targets native D3D11 (PLAN P2).
- One GPU, one resolution, two scenes of one map. GPU pass classes are by shader binding
  names (E050); boundaries between adjacent classes inside one command list are
  approximate.

## Opportunities (ranked by expected benefit on this machine)

| # | Opportunity | Expected benefit | Effort | Risk | Image-quality effect |
| --- | --- | --- | --- | --- | --- |
| 1 | CPU submission on native D3D11: engine scheduler options (`r_schedulerUseDeferredCtx`, `r_schedulerThreadingMode`, section below); the driver reports no command lists, so the runtime emulates deferred contexts | frame rate (CPU-bound) | S to measure | non-default engine path; the FSR submit hook needs command lists (see below) | none |
| 2 | FSR 4 replacing TAA (native AA first, then upscale) | image quality; GPU headroom while CPU-bound | M | engine post chain at display resolution (d2enhanced notes) | large (AA, stability) |
| 3 | GPU headroom: ocean double pass (up to ~1.3 ms panning, E050), forward lighting | only once CPU-bound is solved | M-L | ocean shading parity | none if exact |

Next measurements: profiler on/off A/B; scheduler options (section below) at the same poses;
PresentMon cross-check; pan rerun with mouse injection.

## CPU submission without driver command lists

Static RE in `void-rs/local/ida/Dishonored2.exe.i64` (exe `b671f824...6180`), 2026-10-17. Tags:
**O** observed in code, **H** inferred, **U** unknown. Function names such as
`JobSys_PickPass` are this IDB's (renamed 2026-10-17); render-pass and engine-context
layouts are in `fsr4-integration.md` (Command-list flow). Measured fact (Main, this
machine): `DriverCommandLists = FALSE`, `DriverConcurrentCreates = TRUE`, so the D3D11
runtime emulates deferred contexts.

### Scheduler cvars (**O**)

| Cvar | Value address | Default / flags | Meaning (help text) | Read by |
| --- | --- | --- | --- | --- |
| `r_schedulerUseDeferredCtx` | `dword_1432D5378` (cvar `0x1432D5350`) | "1", 0x4002 (int, command line only) | -1 auto (driver caps), 0 off, 1 on | D3D init `sub_1403694A0` 0x140369C36 (only `== -1`), pool init `sub_1402C7580` 0x1402C7611 (`== 1`), Iggy init `sub_1402DDA30` 0x1402DDA4E (`!= 0`) |
| `r_schedulerThreadingMode` | `dword_1432D52F8` (cvar `0x1432D52D0`) | "2", 0x2 | 0 single threaded (one worker), 1 CPU-only tasks multithreaded, 2 CPU and GPU tasks (needs deferred contexts) | `JobSys_PickPass` 0x1402C7C0A, `sub_1402C5A60`, `sub_1402C6290`, `sub_1402C7580`, job-split counts in `sub_1403295C0`, `sub_140338A40`, `sub_14037D8D0`, `sub_14037EF90` |
| `r_schedulerNoBackground` | cvar `0x1432D53D0` | bool, 0x1 | background workers disabled, every task required for frame finish | not traced (**U**) |
| `jobs_numThreads` | set in game init `sub_14003FBF0` from a computed value | | | effect on render workers **U** |

Render worker count: `sub_140079C60` = clamp(CPU count from `sub_1400DE070` - 1, 2, 15)
("Running with %d + 1 worker threads"), cached in `dword_142BF8414`; contexts/ids =
workers + 1 (`sub_14007A910`). **O**. Which CPU count (physical or logical) **U**; d2enhanced's
8 deferred contexts on the 7700X fit 8 physical cores - 1 + 1 (**H**).

### Q1: `r_schedulerUseDeferredCtx` 0 and -1 on this driver

- **-1**: D3D init logs `D3D11_FEATURE_DATA_THREADING` and, only if the value is -1, sets it
  to `DriverCommandLists != 0` with the engine setter (0x140369C52, `sub_1401028E0(cvar, v,
  force)`). Here that gives **0**, so -1 behaves exactly like 0. **O** (code) + measured caps.
- **0** (pool init `sub_1402C7580`, called from renderer init `sub_14036F800` ->
  `sub_1403511A0`):
  - every pooled engine context wraps the **immediate** context (`sub_1402BBAB0`) instead of
    `CreateDeferredContext1`; job-system byte `+8000` = 0; Iggy creates no deferred
    contexts (`sub_1402DDA30`) and its passes get flag 0x20 (`sub_14034FE00` 0x140350FD1). **O**
  - if threading mode is 2 it is forced to 1 with a warning ("You can't switch to full
    multithreaded rendering because deferred context are not enabled", `sub_1402C6290`, also
    registered as the cvar's change callback). **O**
- Recording with mode 1 (`JobSys_PickPass`): a pass may run on worker `i` only if bit `i` of
  its mask is set; mask = pass mask for CPU-only passes (flag 0x10), **1 (worker index 0
  only) for every GPU pass**. So all GPU passes (depth, shadows, forward, water, post, UI...)
  are recorded one after another by a single worker, **directly on the immediate context**.
  `Ctx_FinishCommandList` returns no list when the context is the immediate one, flag 0x400
  is never set, `RenderPass_Submit` / `ExecuteCommandList` are not used; completion goes
  through `sub_1402C7250`. **O**
- Dependencies: `sub_1402C5A60` has two kinds, bit0 = execution order (successor list
  pass+176, counter +208), bit1 = submission order (+192, counter +212). In mode 2 some edges
  are submission-only, letting dependent passes record concurrently; in modes 0/1 every edge
  is forced to both. **O**
- Not under `RS_LockImmediate`: direct recording does not take the immediate lock (enabled
  on Windows >= 6.3 at 0x140369538); the lock wraps `ExecuteCommandList`, lazy buffer-SRV/UAV
  creation and the loading context. All GPU passes are on one thread, so the immediate
  context is not used concurrently by passes. **O**; that no other thread touches it
  meanwhile is **H**.
- Parallelism left: CPU-only passes (gather/cull/sort/shadow prepare/indirect-lighting
  updates/Iggy texture management, registered with flags 0x10) still run on all workers,
  and job splits use the full worker count (`sub_1403295C0`: count = workers unless mode 0).
  All engine per-draw CPU work of GPU passes (material/state binding, parameter blocks,
  `Ctx_BindProgram`) becomes serial on worker 0, plus the driver's own cost; the runtime's
  record (FinishCommandList 0.64 ms summed) and replay (ECL 1.9 ms) layers disappear. **O**
  (structure) / **H** (cost balance).
- Correctness: an engine-supported path (auto mode selects it on any driver without command
  lists), but the shipped default forces 1 (reason **U**: possibly slower, possibly less
  tested). **H**

### Q2: `r_schedulerThreadingMode`

- 2 (default): any worker records any pass on its own deferred context; submission in
  dependency order by whichever worker wins the try-lock (`JobSys_SubmitReady`). **O**
- 1: CPU-only passes on all workers, GPU passes on worker 0; with deferred contexts still on,
  GPU passes are still recorded deferred and replayed (worse than 2 in principle). **O** / **H**
- 0: every pass on worker 0 and job splits of 1: effectively single-threaded rendering. **O**
- Only takes effect where it is read (pool init forces it; `JobSys_PickPass` reads it per
  pick, so runtime changes affect masks immediately, but dependency kinds are fixed when the
  graph is built each frame). **O** / **H**

### Q3: what `render_views` (`sub_140361D60`) contains

Per view, `sub_140351780` (render function, 0x6CB4 bytes) builds the frame graph: hundreds
of `sub_1402C7FB0` (enqueue pass instances, release worker semaphore), `sub_1402C1480` /
`sub_1402C1530` / `sub_1402C5A60` / `sub_1402C5BA0` (dependencies) and parameter setup, then
at its end **kicks the render task workers** (`sub_1402C8310` at 0x140358202: job list
`arkRenderTaskWorker`, worker count + 1 instances of `JobSys_WorkerLoop`) and **immediately
waits** for them (`sub_1402C8440` at 0x14035820A -> `sub_14007AD60`, then
`sub_1402BBB30(immediate)`). Workers exit when every pass is done; in deferred mode a pass is
"done" only after it was submitted, so the wait covers recording **and** the
`ExecuteCommandList` replays. **O**

Around it: per-world `sub_140376980`, `sub_140304C40` / `sub_14030F3D0` on the immediate
context, vtable +456 (view setup), and `sub_140351630` (2D/GUI surfaces drawn directly on the
immediate context). **O**

So the 5.2 ms is mostly the critical path of the pass graph seen from the main thread, not
main-thread work. Whether the main thread executes worker jobs while waiting (`sub_14007AD60`
-> per-thread job context vtable +88) is **U**. The split needs measurement: hook
`sub_1402C8310` (kick) and `sub_1402C8440` (wait) to time graph build vs wait, and
`RenderPass_Record` (`sub_1402C5D50`, callback at pass+0xD8, thread id) to get per-pass
record time and the per-worker busy timeline. D3D11 call overhead is inside the pass
callbacks (recording) and inside ECL (replay); static analysis cannot apportion it.

### Q4: setting the cvars before they are read

- `DINPUT8.dll` is a static import, so void-inject's `DllMain` runs **before** the exe's
  static initializers, which register the cvar with its default (`sub_141366640`, "1"). A
  write from our init thread races with that registration. **O** (import table) / **H** (race)
- Order inside game init `sub_14003FBF0`: cvar system init -> `jobs_numThreads` -> command
  line `sub_140041240(args, allowCommandLineOnly = 1)`, which accepts `+set name value` and
  `+name value` and applies 0x4000 cvars with the force setter `sub_140102950`. D3D init
  (`sub_1403694A0`) and render-pool init (`sub_1402C7580`) come later. **O**
- Option A (no code): launch option `+set r_schedulerUseDeferredCtx 0` (and/or
  `+set r_schedulerThreadingMode N`). **O**
- Option B (void-inject): MinHook `sub_1403694A0` entry (main thread, after command line,
  before both reads) and call `sub_1401028E0(0x1432D5350, value, 1)`: `force = 1` bypasses
  the command-line-only, read-only and retail checks in `idCVar::Set` `sub_140102400` and
  updates string, int and float. Writing the int at `0x1432D5378` directly also works (all
  readers read the int) but leaves the string stale. Changing it after pool init has no effect
  (pool and Iggy contexts are built once). **O**
- Threading mode can be changed at run time (callback + per-pick read), but values that need
  deferred contexts are forced back to 1 when they are off. **O**

### Q5: A/B experiments (Ctrl+F3 bench, same poses, three repeats)

| Rank | Setting | Expected | Risk |
| --- | --- | --- | --- |
| 1 | `+set r_schedulerUseDeferredCtx 0` (mode forced to 1) | removes runtime record (0.64 ms summed) and replay (1.9 ms) layers, but serialises GPU-pass engine CPU on one worker: net sign **U**; the only lever without DXVK that removes the emulation | path rarely exercised on PC (default forces 1); stutter if worker 0 picks a long CPU-only pass while GPU passes wait; void-inject's ECL/FinishCommandList profiling and the FSR submit-time hook design see no lists (frame interval, engine CPU/GPU timers and `render_views` remain valid) |
| 2 | `+set r_schedulerUseDeferredCtx -1` | identical to rank 1 on this driver (resolves to 0); run once only to confirm the auto path logs "DriverCommandLists: 0" | as rank 1 |
| 3 | default deferred + `r_schedulerThreadingMode 1` | control: GPU recording on one worker with emulation kept; isolates how much parallel recording buys | expected slower; no correctness risk |
| 4 | `r_schedulerNoBackground 1` | frame waits on background tasks too: likely neutral or worse; checks whether background tasks compete with render workers | hitching (**H**) |
| 5 | `r_schedulerThreadingMode 0` | single-threaded reference (worst case), gives total serial CPU cost | slow, debug only |

Instrument before running: kick/wait split and per-pass record times (Q3); per-worker busy
time shows whether worker 0 becomes the critical path in rank 1. Not investigated:
`jobs_numThreads` (whether it reaches the render worker count is **U**).
