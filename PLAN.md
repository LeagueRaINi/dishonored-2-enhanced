# Plan

## Current direction (2026-10-17): replace individual rendering passes

Goal (user, 2026-10-17): better image quality and performance in the installed Dishonored 2
by replacing or adding *individual* passes of the original D3D11 renderer, plus texture
replacement for moddability. Full renderer replacement and removing D3D11 are not goals.
Prefer, in order: engine settings, targeted fixes, replacement of individual passes; use
the independently decoded data (void-formats/void-assets) where a specific improvement
needs it. The in-process module is `void-inject` (dinput8 proxy, Rust).

Removed 2026-10-17 (user-authorised cleanup; the code is kept in checkpoint 9282529 of the
private pre-release history): the standalone Bevy reconstruction (M-track) and the
in-process full-scene renderer/takeover (R-track). Their findings stay in docs/STATUS.md
(E001-E050), docs/renderer-replacement.md, docs/m2-materials.md, docs/m3-*.md and
docs/experiments/.

Relation to `d2enhanced` (the earlier C++ Dishonored 2 Enhanced, replaced by this project):
it shipped FSR 4.1, SSGI/SSR and HDR as a C++ D3D11 proxy that detects passes by shader
bindings and patches 3,705 forward pixel shaders. void-inject supersedes it for these
features (both cannot own the TAA pass; only one is installed at a time). Its findings are
leads, reused with corroboration.

Design rules for every pass replacement:
- Hook at engine level where the engine identifies the pass (call sites, view/job
  structures from the reflection tables), not by guessing from D3D11 state; read frame
  constants from the engine's view, not from shadowed constant buffers.
- One shared splice mechanism for D3D12 work inside the game's frame (the TAA point first:
  pre-TAA HDR colour, depth, motion vectors); it must work on native D3D11 regardless of
  driver command-list support, and must not depend on GPU waits inside a driver replay (E047).
- Every replacement has an off switch, an A/B comparison against the original pass, and a
  profiler class; the original path stays measurable.

### P1: low-overhead profiling and a repeatable baseline (gate passed for native)

Done: profiler, pass classes, benchmark driver, native baseline (docs/performance-baseline.md:
CPU-bound at ~122 fps on eotw, CPU 7.7-8.0 ms vs GPU 5.7-6.0 ms). Open: profiler on/off A/B,
PresentMon cross-check, pan rerun with mouse injection.
Consequence: on this machine GPU-side savings do not raise fps; FSR 4 and added lighting
passes fit inside the CPU-bound gap (to be measured per change).

### P2: CPU submission on native D3D11 (no DXVK)

User decision (2026-10-17): avoid DXVK; it was only used because AMD's D3D11 path made the
CPU limit worse. Observed 2026-10-17: the installed driver (32.0.32015.2008) still reports
`DriverCommandLists = FALSE` (D3D11 `CheckFeatureSupport(THREADING)`), so the runtime
emulates the engine's 8 deferred contexts and replays them on the immediate context.
Every pass replacement targets native D3D11 (the G1 design needs no DXVK and no driver
command lists). Next: the engine's own scheduler options (`r_schedulerUseDeferredCtx`,
`r_schedulerThreadingMode`; research in docs/performance-baseline.md "CPU submission
without driver command lists"), measured A/B with the benchmark.
Results 2026-10-17 (STATUS E059): `+set r_schedulerUseDeferredCtx 0` showed rendering
artifacts and lower fps at once (user; not benchmarked) -> rejected. D3D11-on-12: although it
reports driver command lists, a synthetic 8-worker / 4,000-draw / 43-list microbenchmark
(`local/tools/on12bench`) costs 2.25x the main-thread CPU and 3.6x the total CPU of native
D3D11 (recording 8.9x) -> rejected. Critical path in game (FSR run): the main thread waits
6.2 ms for the pass graph while workers record 18.2 ms in total (~2.9 busy on average) and
the serialised submission costs 2.8 ms. Next: per-pass record times on the critical path.

### G1: FSR 4 replacing TAA

Research: docs/fsr4-integration.md (engine TAA site, inputs, command-list flow, jitter,
render scale and post chain, mip bias, reactive masks).
1. Done: FFX API bindings (`void-inject/src/ffx.rs`), `examples/fsr_check` (providers 4.1.1 /
   3.1.5 / 2.3.4 on this driver; ~1.75 ms GPU at 3440x1440).
2. Done (offline): splice without a command-list split (`fsr.rs`): record-time redirect of
   the TAA `Ctx_Dispatch` call, submit-time redirect of the POSTPROC `ExecuteCommandList`
   call; cross-API fences between command lists only; inline on the immediate context when
   the engine records without deferred contexts (P2 experiments). `examples/fsr_offline`
   exercises both entry styles on stand-in engine textures.
3. Implemented, offline-verified, not yet run in game (checklist in docs/fsr4-integration.md
   "Implementation status"): native AA (`fsr = 1`, F6 / Shift+F6), FSR jitter sequence via
   `idRenderView_SetupScaled`, upscale mode (`fsr_mode = upscale`, `fsr_scale`; post chain at
   display resolution on a private view copy + portal-CB twin), mip bias in `CreateSampler`,
   reactive / transparency masks from the OIT opacity buffers, camera cuts.
4. Done offline 2026-10-17: sky motion vectors (matrix convention measured against the
   engine's own vectors in game), display-size `fx/depthFull` and depth buffer with stencil.
5. Next: in-game validation of native, then upscale; open: distortion-vector units.

Gate per step: in-game run with recorded settings and the profiler; no regression in the
original path when disabled.

### G2: lighting, shadows, reflections

Research: docs/lighting-options.md (engine settings beyond Ultra, screen-space vs ray-traced
passes on the decoded scene, material-data sources, splice points, denoising). Order and
first experiments are decided there. Step 1 (settings pass) started: cvar overrides from
`void-inject.ini` (`cvar.<name> = <value>`), `image_anisotropy 16` (Ultra leaves 1), sampler
census in the log; next the pre-forward screen-shadow splice (own AO / contact shadows).
FidelityFX SDK 2.3 only (user decision): measured on this GPU, upscaler 4.1.1, frame
generation 3.1.6 and radiance cache 0.9.0 are available, the Ray Regeneration denoiser is
not. Candidate effects (performance and image quality), how replaced engine passes are
switched off (pass flag `0x100`) and the order: docs/fidelityfx-effects.md.

Ray tracing: docs/raytracing.md (E070, crate `void-rt`, offline `rt_spike`). DXR 1.1
inline ray tracing works on this GPU. The compacted BVH of the whole eotw static scene is
0.36 GiB and builds in 13 ms GPU. Mirror reflection rays cost 1.5 ms at 0.67 scale
(0.94 ms hybrid + 40 m cut-off). Path-traced photo mode: 17 ms per sample at 3440x1440.
Next: photo-mode path offline, then a paused in-game prototype; real-time RT reflections
after the in-game SSR results.

### G3: texture replacement and moddability

Research: docs/texture-replacement.md: one hook on `idFileSystemLocal::OpenFileRead`
serving engine-built memory files (base images, streamed mips, decls, models); no loose-file
override exists in the engine and D3D11-level texture hashing cannot work here.
Implemented, not yet run in game: BIM writer + DDS reader (`void-formats`), shared
validation (`texmod.rs`), `void-tool texmod check|import|export|test-pattern|roundtrip`
(3,722 tutorial images re-written byte-identically), `texmods.rs` hook installed from
DllMain (`texture_mods = 1`, `mods_dir`); test mod `local/mods-test` (magenta player sword).
Next: in-game smoke test, then streamed (split) replacements and material decl overrides.

### Known leads (to be confirmed, not assumed)

- Native D3D11 on AMD replays deferred command lists serially on the immediate context
  (driver reports no command lists, see P2; ~1.9 ms `ExecuteCommandList` CPU per frame).
- Engine cvars: `r_schedulerThreadingMode` (deferred contexts on/off), `r_useGPUCulling`,
  frame limiter (`com_disableFrameSpinning`, `com_timeScaler`), shadow update slicing.
- Per-frame engine wait for its own GPU timer query (`sub_14036AE40`): measured ~0.001 ms
  (not a stall on this machine).

---

## History

- 2026-10-15/16, M-track: standalone Rust/Bevy reconstruction of `tutorial_p` (archives,
  BWM/BIM/material/entity/probe/Havok decoders, Bevy viewer validated against game depth,
  collision and movement prototype from RX003). Decoders remain in `void-formats` /
  `void-assets`; viewer and simulation removed 2026-10-17.
- 2026-10-16/17, R-track: in-process D3D12 full-scene renderer and takeover of the game's
  frame (docs/renderer-replacement.md, E031-E048). Removed 2026-10-17; the engine-structure
  findings (reflection tables, views, scene lists, pass graph) feed the current work.
