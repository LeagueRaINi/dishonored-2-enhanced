# G2 options study: lighting, shadows, reflections

Status 2026-10-17, research only (no code changed); 2026-10-18: B1 implemented (E063, see B1).
Scope: improve lighting, shadows and
reflections of the installed game by changing settings or replacing/adding *individual*
passes of the original D3D11 renderer (PLAN.md G2), not by replacing the renderer.

Labels: **O** observed (IDB of exe `b671f824...6180`, shipped decl source, cvar dumps,
measurements), **H** inferred, **U** unknown. "Lead" = claim from `d2enhanced/` not
corroborated here; "corroborated" = checked independently in this study. New findings are
numbered LO1-LO10 (candidates for the STATUS.md ledger).

Inputs: STATUS.md (E016, E034-E050), renderer-replacement.md, performance-baseline.md,
d2enhanced `re/NOTES.md`, `re/SSGI_SSR_REVIEW_2026-10-15.md` (+ `_RESPONSE.md`),
`src/{ssgi,material,shader_patch,hooks}.cpp`, `shaders/*.hlsl`, FSR SDK 2.3
(`third_party/FidelityFX-SDK/Kits/FidelityFX`), shipped renderprog decls extracted in
`local/extract/renderprog` (270 files) and `local/extract/renderparm`, cvar dump
`local/profile/run_1792194435_scene1_native/cvars.tsv` (in-game Ultra), IDB
`local/ida/Dishonored2.exe.i64` (function comments added for the pass callbacks below).

## 1. Constraints that shape every option

- **Frame budget (O, performance-baseline.md):** eotw Ultra 3440x1440 native TAA is
  CPU-bound: interval p50 8.1-8.3 ms, engine CPU 7.7-8.0 ms, GPU 5.7-6.0 ms. GPU headroom
  before becoming GPU-bound is ~2.2 ms. FSR 4 native AA alone costs 1.77-1.90 ms GPU (lead:
  d2enhanced NOTES "FSR native-AA results"; SDK table: RX 7900 XTX 3.07 ms at 3840x2160,
  `docs/techniques/super-resolution-ml.md`). **H:** every lighting addition beyond a few
  tenths of a ms only stays free if it is paired with FSR upscale mode (render scale < 1),
  which cuts the engine's own per-pixel passes. Any CPU work on the game's main thread
  (render_views 5.2 ms) costs frame rate directly.
- **Forward renderer, no G-buffer (O, corroborated):** NOTES "Renderer type" lists the
  render targets (no normal/albedo/roughness). Shipped `lighting.decl`
  `IntegrateLighting` shades every light and the indirect term in one forward pixel shader
  (brdf_ward/vegetation/skin/eye `ComputeIndirect`). The depth prepass writes only the
  R16G16_FLOAT velocity target (E050, `passes.rs` classification). Post-hoc changes to the
  final colour therefore cannot separate direct sun, local lights, ambient and IBL.
- **No opaque SSR in the engine (O, LO10):** among the 270 shipped renderprog decls only
  `arkscreenspaceraytracing.decl` and `arkwaterhelpers.decl` use `ScreenSpaceRayTrace`;
  opaque surfaces get specular from `COMPUTE_IBL` (`global.decl:469` -> `ComputeIBL`,
  `_IBLCubeMapArray`) times AO.
- **No runtime shader compile path found (O, LO9, bounded):** the D3D init
  `sub_1403694A0` does `LoadLibraryExA("\\d3dcompiler_47.dll")`, but the exe has no
  `D3DCompile*` string (string regex `(?i)D3DCompile`) and its only `compile` imports are
  AGS async-compile counters. Shaders come from `base/shaderCache/ShaderCache_PC.*`
  (precompiled). Editing renderprog source to add outputs is therefore not available;
  per-pixel material data from the engine needs DXBC patching or shader substitution.
- **Interop (O):** FSR and DXR need D3D12; the game is D3D11. Shared textures must be
  created by D3D11 and opened in D3D12 (NOTES "FSR native-AA results"); a GPU wait inside
  the AMD driver's command-list replay caused 6 fps and a TDR (E047); the agreed rule is
  cross-API fences only between command lists (PLAN.md "Design rules"). The working
  pattern is d2enhanced's command-list split at record time (NOTES "FSR injection v2").
  Screen-space work that needs no D3D12 can run as **D3D11 compute recorded into the
  engine's own deferred context** (as d2enhanced SSGI does), with no fences at all.

## 2. Engine pass graph and splice points (LO1-LO3)

**LO1 (O, IDB):** `sub_14034FE00` builds the frame as named render passes
(`sub_1402C2960(slot, "ARK_RENDERPASS_*", ..., callback, ...)`). Relevant order and
callbacks (comments added in the IDB):

| Pass | Callback | Notes |
| --- | --- | --- |
| `SHADOWS_PREPARE` / `COLLECT_DYNAMIC_CASTERS` / `GATHER_CASTERS` | `sub_140333550` / `sub_140330020` / `sub_140330C10` | CPU jobs (cascade fit, caster lists) |
| `DEPTHPASS`, `DEPTHMINMAX` | `sub_1402FD8D0`, `sub_1402FC130` | depth + velocity prepass, tile min/max |
| `SSAO` | `sub_140305BE0` | HBAO+ (`r_hbao 1`) or DWAO (E038) |
| `SHADOWS_WORLD`, `SHADOWS_DYNAMIC` | `sub_14032F390`, `sub_14032F380` | CSM + local-light atlas |
| `INDIRECT_LIGHTING_{ACCUMULATE_FINAL_SH,UPDATE_IBL,UPDATE_PROBES}` | `sub_140310190`, `sub_1403102A0`, `sub_140310260` | probe SH / IBL state |
| `SCREENSHADOWS` | `sub_14032C7F0` | sun shadow resolved to screen space |
| `TLFLIGHTCULL`, `FORWARD_WORLD`, `FORWARD_DYNAMIC` | `sub_1402F7260`, `sub_1403133D0`, `sub_1403133B0` | tiled light lists, forward lighting |
| `FOG`, `RENDER_WATER`, ..., `POST_TONEMAP` | `sub_140309430`, `sub_1403045A0`, ..., `sub_140323E70` | TAA lives in `PostFX_TemporalAA_and_Post` `0x140327270` (lead, NOTES) |

Each callback receives a job whose view is at `job+112` (seen in `sub_14032C7F0`; matches
d2enhanced's `job+0x70`). Hooking a callback's return is an engine-level, driver-agnostic
record-time splice (PLAN design rule 1), unlike immediate-context hooks that depend on AMD's
replay (E045 limits).

**LO2 (O, shipped `lighting.decl:79-143`):** the forward shader reads
`$texScreenshadows[pixel].xy`: **`.x` = SSAO**, used only by
`brdf.ComputeIndirect(..., ssaoScreenShadow.x, ...)` (ambient probe term and IBL specular,
`ComputeAmbientOcclusion` with material AO, E038); **`.y` = sun shadow** for the parallel
light. Programs with `noScreenShadows 1` (`arkvegetation.decl:7`) use `.x` only and compute
their own `COMPUTE_PARALLEL_SHADOW`. The target is created as `_SSAO_ScreenShadows_%p` in
`sub_14036D0C0` (`0x14036DC0D`); the link between that name and `texScreenshadows` is
H (names; marker `ssaoScreenShadows` in `sub_140351780`). Format: U.

**LO3 (O, `screenshadows.decl`, exe strings `0x141B2A2B0..0x141B2A3E8`):** `SCREENSHADOWS`
is a full-screen pass that reconstructs world position from `screenShadows/DepthBuffer`
and evaluates `ComputeParallelShadow(..., HIGH_QUALITY)` (optional interleaved jitter),
followed by materials `screenShadows/bilateralBlur/blurX|blurY`. Whether the blur runs on
every frame and in which order relative to the `.x` write: U.

Splice points used below:

| Id | Where | Data available | Mechanism | Engine consumes it as |
| --- | --- | --- | --- | --- |
| S0 | cvars / data | - | `cvars.rs`-style writes (`set_int`), at startup or per frame | native |
| S1 | end of `SSAO` and/or `SCREENSHADOWS` callbacks, before `FORWARD_WORLD` | current depth, MVs, view (sun dir `+48344`, matrices), `_SSAO_ScreenShadows` | D3D11 CS in the job's deferred context; or command-list split + D3D12 | AO in `.x` (indirect only), sun visibility in `.y` (direct sun only) |
| S2 | TAA dispatch = FSR splice (G1) | pre-TAA HDR colour, depth, MVs, exposure | command-list split (d2enhanced v2), D3D12 | nothing: our composite replaces pixels of the final lit colour |
| S3 | `INDIRECT_LIGHTING_UPDATE_IBL` / IBL cube array | `_IBLCubeMapArray` (lead: 256^2 BC6H, 7 mips) | resource update | IBL lookups in every forward shader |

S1 is the important finding of this study: AO and sun-shadow improvements can be written
into the one texture the forward shader already reads, so the engine's own BRDF, probes,
fog and TAA/FSR apply them correctly, with no colour composite and no material data.
S2 is the shared FSR splice (same point: pre-TAA HDR colour, depth, MVs) and is needed for
anything that adds or replaces radiance (SSR, GI bounce, RT reflections).

## 3. Options

Cost basis legend: **M** measured (source cited), **H** estimate, **unmeasured** where no
basis exists. Effort: S (days), M (1-3 weeks), L (more).

### A. Engine settings beyond Ultra (S0)

Ultra values below are from the in-game Ultra dump (O, `cvars.tsv` of run 1792194435);
registration flags from the `idCVar` constructor calls (O, IDB).

| # | Setting | Ultra (default) | Evidence of effect | Proposal |
| --- | --- | --- | --- | --- |
| A1 | `image_anisotropy` | **1** (1) | **LO4 (O):** `CreateSampler` `sub_140378280`: a sampler whose options select `ANISO` gets `Filter = 0x55` (anisotropic) and `MaxAnisotropy = image_anisotropy` (image manager +100, set by `sub_1402EB830`, clamped >= 1) unless the sampler carries its own `MaxAnisotropy`. Shaders sample layers/masks through `$anisotropicSampler` (`arkdefaulthelpers.decl`, `arkdecal.decl`, ...). The sampler cache key (`sub_1403778A0`, `"sampler_%d_..."`) does not include the global value. | 16. H: the anisotropicSampler decl itself (not extracted) leaves the global in charge; samplers already created keep 1 (set before map load). |
| A2 | `r_shadowParallelSliceSize` (sun CSM) | 2048 (2048), registration flags 0x4002 | read in `sub_140371210` / `sub_140373070` (target creation). 0x4000 = command-line only (lead, NOTES "Retail cvar lock"). | 4096 via command line (+256 MB at 32 bit, `r_shadows32Bits 1`). |
| A3 | `r_shadowCascadeFars` | `4 12 30 75` | **LO5 (O):** `sub_140330630` parses exactly 4 floats (`"%f %f %f %f"`, fallback 4/12/30/75), rescales all if `r_zfar` < last far, clamps to the visible range when `r_tightenParallelShadows`. Slice count is fixed by the format. Beyond the last far no CSM (H, renderer-replacement L2). | e.g. `5 16 45 150` together with A2 (keeps near density, doubles shadow range). |
| A4 | `r_shadowParallelPartialSliceUpdate` | 1 (1) | description: time-slices cascade updates except slice 0. | A/B 0: removes lagging far cascades; costs shadow GPU and caster CPU every frame. |
| A5 | `r_maxShadowMapResolution` (spot/omni) | 1024 (1024), flags 2 | **LO6 (O):** `sub_140330AC0`: size = nearest power of two of `max x cot(fov/2) x lightRadius / distance`, capped at `max` (or `r_maxShadowMapExtraResolution` 2048 for "extra" lights), halved when the low-res divider is on. Runtime-read per light. | 2048. Risk H: atlas is 8 x 2048^2 slices + 12 cache slices (`r_shadowAtlasSlice*`, `r_shadowCacheSlice*`); omni lights use 6 faces, so larger maps may evict/drop shadows. |
| A6 | `r_shadowCullDistance` | 75 (25) | **LO7 (O):** read squared in `COLLECT_DYNAMIC_CASTERS` `sub_140330020`. | 100-150 for distant dynamic casters; CPU cost (main-thread jobs). Note: d2enhanced Ultra+ forces 60 (`hooks.cpp:188`, comment assumes default 25) = a reduction from Ultra's 75. |
| A7 | `shadowFilteringQuality` renderParm | option 1, range 0-2 (`renderparm/shadowfilteringquality.decl`) | shader permutation option; whether quality-2 variants exist in `ShaderCache_PC`: U. | Check the shader cache before trying. |
| A8 | `r_lodZoomMultiplicator` | 0.6 (1.0) | **LO8 (O):** `0x1403522D1` in `sub_140351780`: zoom = min(fov term, 1) x multiplier, stored per view; the menu LOD option writes it (`sub_1405105A0`, data-driven). Direction (higher = finer LODs further away) is a lead (NOTES "Quality cvars"), not verified. | A/B 0.6 / 1.0 / 2.0, watch LOD pops and GPU vertex cost. |
| A9 | `lod_dist_level0..3` (entities) | 30/40/50/70 (3/6/12/20) | Ultra already raised them 10x. | Leave; small payoff. |
| A10 | HBAO+ parameters | hard-coded in `sub_140305620` (E038, E063: radius 0.7, bias 0.1, small/large scale 1/s and 1.75 s for render scale s, power 1.75, foreground AO at 4 m, blur **disabled** (radius 4 / sharpness 8 stored but `Blur.Enable = 0`), input normals off, viewport input off) | code immediates, not cvars. | Superseded by option B1 (implemented, `ao_mode = void`). |
| A11 | `r_SSSBlurQuality` | 0 = full 92 taps (0) | already maximum (description in dump). | none. |
| A12 | `r_textureStreamerMemoryBudgetMB` | 4096 (-1) | 24 GiB VRAM available (meta.txt). | Belongs to G3; raise if mip drops are observed. |
| A13 | `r_IBL_AreaMin/Max`, `r_IBL_DistancePower`, `r_IBL_GrazingAnglePower` | 100 / 2000 / 2 / 2 | footprint-based IBL roughening (descriptions). | Low priority tunables; effect U. |

Already maxed at Ultra (O dump): `r_shadowParallelResolutionDividerLevel 0`,
`r_shadowSpotAndOmniLowResolutionDividerEnabled 0`, `r_hbao 1`, `r_shadows32Bits 1`,
`r_water_ScreenSpaceMaxStepsCount 128`. So d2enhanced's Ultra+ set is, at Ultra, effectively
only `image_anisotropy 16` (plus the cull-distance regression) - corroborated by the dump.

- Mechanism: write the cvar int/float fields (`cvars.rs` layout, int +40, float +44) at a
  defined time (A1 before map load; A2 only on the command line), or through the console
  path with the retail lock patched (lead, NOTES).
- Cost basis: shadows class 0.67-0.97 ms p50 at Ultra (M, performance-baseline); A2/A3/A5
  scale fill and caster count: H 1.5-3x of that class; A1: H < 0.2 ms on RDNA3; A4/A6 add
  CPU (unmeasured, important because the frame is CPU-bound).
- Compatibility risk: low (engine paths); A5 atlas pressure; A2 needs a launch option.
- Payoff: A1 large on oblique ground/floors; A2+A3 crisper sun shadows and shadows past 75 m;
  A5 sharper lamp shadows; A8 fewer LOD pops.
- Effort: S.
- Validate first: A1 sampler descs actually created (hook `CreateSamplerState`, histogram
  of `MaxAnisotropy`), then each change A/B with the profiler at the two eotw poses.

### B. Screen-space passes

**B1. Own AO into `.x` at S1 (GTAO-class or visibility-bitmask AO).**
- Mechanism: after the `SSAO` callback (or at the end of `SCREENSHADOWS`), D3D11 CS in the
  same deferred context writes our AO into `_SSAO_ScreenShadows.x` (replace or `min` with
  HBAO+). Engine applies it only to ambient + IBL specular, with material AO (LO2, E038).
  Our horizon AO with the game's parameters already exists in D3D12 (renderer-replacement
  "Ambient occlusion"); d2enhanced has a visibility-bitmask trace (`ssgi_trace.hlsl`,
  Therrien et al.) whose review fixes are done (response items 2, 6-8).
- Data: depth (`DEPTHPASS`), projection from the view (`+29584`...), depth-reconstructed
  normals (bump normals gave far too much AO, E038). No material data needed.
- Splice: S1. No D3D12, no interop.
- Cost: HBAO+ is inside `other_draw` 0.59 ms + `other_compute` 0.07 ms p50 (M, class sums,
  not isolated). Own AO: unmeasured; H 0.4-1.0 ms full res at 3440x1440, half that at half res.
- Risk: format and bind flags of `_SSAO_ScreenShadows` (U; without a UAV flag the write must
  be a full-screen draw with a colour write mask on the channel) and whether `.x` is written
  before or after the screen-shadow blur (LO3 U); temporal AO needs MVs and resets (review
  item 2).
- Payoff: medium (contact darkening in corners, under props, characters; no darkening of
  direct light, unlike d2enhanced's post-hoc `colour * lerp(1, visibility, ao)`).
- Effort: S-M. Validate first: constant-write test (see section 5).
- **Implementation status (2026-10-18, E063, offline-verified, not run in game):**
  `void-inject/src/ao.rs`, settings `ao_mode = hbao | void` (default `hbao`), `ao_quality`
  (low 2x2 / medium 2x3 / high 3x3 / ultra 4x4 slices x steps, default high),
  `ao_half_res` (default 1), `ao_radius_scale`, `ao_strength`, `ao_debug`; all live
  (overlay tab Lighting). Splice is not S1 but the HBAO+ call itself: MinHook detour of
  `RenderAO` in `GFSDK_SSAO_D3D11.win64.dll` (3.0.0.0, SHA-256 checked), found through the
  context's vtable (`qword_1432EF760` if it exists, else a detour of the
  `GFSDK_SSAO_CreateContext_D3D11` export). With `void` our AO is recorded into the pass's
  deferred context, writes HBAO+'s output RTV (blend mode from the Output struct, RGB only)
  and HBAO+ does not run; the engine's following composite draw is unchanged. Every D3D11
  state we touch is restored (checked offline with a wider snapshot than we save).
  Algorithm: single-dispatch linear-depth MIP prefilter (XeGTAO filter, groupshared) ->
  visibility-bitmask AO (Therrien et al. 2023: 32 cosine-weighted sectors per slice, sample
  thickness 0.5 x radius, so no halos behind thin foreground objects) with XeGTAO's
  slice/step layout and MIP sampling, per-quality unrolled permutations, depth-reconstructed
  normals, HBAO+'s radius/foreground clamp/bias/small-large weights/power from the game's
  Parameters, fixed 4x4 noise -> one-pass groupshared 5x5 bilateral blur whose weights cancel
  the 4x4 pattern (no temporal filter) -> depth-aware upsample + write inside the engine's
  viewport only (render-scaled region in FSR upscale; samples outside it count as
  unoccluded). Occlusion gain calibrated against the real HBAO+ (equal mean AO of the test
  viewport).
- **Offline cost (M, `ao_offline`, RX 7900 XTX, 3440x1440 texture, median of 40 executions
  of the recorded command list; same harness also times the real HBAO+):**

  | | HBAO+ | low | medium | high | ultra |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | native, full res | 0.44 | 0.60 | 0.63 | 0.80 | 1.13 |
  | native, half res | | 0.30 | 0.31 | 0.37 | 0.47 |
  | 0.67 region, full res | 0.48 | 0.25 | 0.29 | 0.35 | 0.50 |
  | 0.67 region, half res | | 0.14 | 0.16 | 0.17 | 0.21 |

  ms; run-to-run variation about 5-10 %. The output target in the harness is RGBA32F
  (NaN detection), so the write cost is pessimistic for both. Default (high, half res) is
  below HBAO+ in both modes. HBAO+ always processes the whole texture (viewport input
  disabled by the game), so in upscale mode it also writes outside the render region (2.7 M
  pixels offline). Measured and rejected: FP16 view depths (6 % faster, doubled false
  occlusion of the test scene), phase-coherent thread mapping (25 % slower), lower MIP
  sampling offset (no gain).
- **In-game unknowns:** the RTV format and the engine's use of the target (composite draw
  `qword_1422AE2C0`), real scene look vs HBAO+ (calibration used a synthetic scene), GPU ms
  in the profiler's new `ao` class, depth precision far away (R24, reversed-Z).
- **In-game test (user run; stop after step 6 or at the first failure):**
  1. `void-inject.ini`: `overlay = 1`, `profile = 1` (for the `ao` GPU class), `ao_mode =
     hbao`. Start, load the eotw save, stand at a benchmark pose.
  2. F2 -> Lighting -> Technique "Own". Expect a toast "Ambient occlusion: own (visibility
     bitmask; HBAO+ skipped)" and in the log `ao: HBAO+ hook installed: RenderAO ...` then
     `ao: first RenderAO call: ...` (projection, depth SRV/RTV formats and sizes, bound
     viewport). Runtime card: "Rendered by us" counts up, "Handed to HBAO+" stays 0. Any
     fallback reason is logged and shown in red.
  3. Look: corners, props on floors, characters' feet darker than with HBAO+, no halo
     around thin foreground objects, no darkening of directly lit areas (AO only reaches
     ambient/IBL, LO2), no flicker when standing still or turning (fixed pattern + blur).
     Toggle Technique back and forth for an A/B; screenshots at the same pose.
  4. "Show AO buffer": the AO alone full-screen (white = open); compare quality presets and
     half resolution live; Radius scale / Strength react immediately.
  5. Profiler A/B (Ctrl+F3) `ao_mode = hbao` vs `void` at the same pose: `gpu_ao_ms` (ours)
     vs the HBAO+ share of `gpu_other_draw_ms`/`gpu_other_compute_ms`; frame interval.
  6. With FSR upscale (`fsr = 1`, `fsr_mode = upscale`): AO aligned with geometry, no band
     at the right/bottom edge of the render region (log shows the scaled bound viewport).

**B2. Sun contact shadows into `.y` at S1.**
- Mechanism: short screen-space ray march from each pixel toward the sun
  (`m_sunDirection` view `+48344`, O renderer-replacement L1 / NOTES) against the current
  depth; `.y = min(engine CSM, contact)`. Catches small-scale occlusion the 2048^2 cascades
  miss (window frames, props on tables, character feet).
- Data: depth, sun direction, matrices. Characters are in the depth buffer, so they cast.
- Splice: S1. D3D11 CS. Vegetation receivers ignore `.y` (LO2), so grass/foliage would not
  receive contact shadows (casters in depth still shade others).
- Cost: unmeasured; H 0.2-0.5 ms (16-32 steps, full res).
- Risk: off-screen casters absent (by design, short range only); thickness heuristics on thin
  geometry. The engine blurs `.y` inside the callback before we write (E066).
- Payoff: medium-high for indoor/close views; complements A2/A3.
- Effort: S-M. Validate first: constant-write test, then a step-count sweep with profiler.
- **Implementation status (2026-10-18, E066; in game: user check passed, on by default):**
  `void-inject/src/contact.rs`; settings `contact_shadows` (default 1), `contact_length`
  (0.5 m), `contact_thickness` (0.25 m), `contact_quality` (low 8 / medium 12 / high 16 /
  ultra 24 steps, default high), `contact_debug`; all live (Lighting tab, card "Sun contact
  shadows"). LO3 resolved: the SCREENSHADOWS callback `sub_14032C7F0` renders the CSM into
  `.y` of image 2006969548 and blurs `.y` itself (blurX/blurY) before returning, so our
  detour records after the engine's blur: `.y = min(blurred CSM, contact)` (green write
  mask, blend op MIN), only in CSM mode (`view+14880 == 0`), only inside the view viewport.
  Per pixel: ray `P -> P + L * length` projected to the screen, perspective-correct ray
  depth at each sample's pixel centre, hit if `bias < rayZ - sceneZ < thickness` (bias:
  receiver depth slope per pixel + 0.2 % of depth + 2 R24 steps), hits fade over the last 40 %
  of the ray; receivers facing away from the sun (depth-reconstructed normal) are skipped
  (they are dark through N.L and the CSM, and their rays would enter their own surface).
  Fixed 4x4 start offsets + the AO's one-pass depth-aware blur (no temporal filter).
- **Offline cost (M, `contact_offline`, RX 7900 XTX, 3440x1440 texture, median of 40):**

  | | low | medium | high | ultra |
  | --- | ---: | ---: | ---: | ---: |
  | native | 0.25 | 0.29 | 0.32 | 0.40 |
  | 0.67 region | 0.12 | 0.14 | 0.15 | 0.19 |

  ms (march + blur + composite; synthetic scene with large sky/early-out areas, so real
  scenes may cost more).
- **In-game test (user run):**
  1. Lighting tab -> "Sun contact shadows" on (installs the hook live). Log:
     `contact: SCREENSHADOWS callback hooked` and `contact: first SCREENSHADOWS pass: ...`
     (formats, region, sun world and view-frame vectors, view-matrix translation slot:
     column 3 should be non-zero, row 3 = 0 0 0 1). Runtime grid: "Passes / applied" count
     up together; "Skipped" rises indoors without sun shadows.
  2. "Show contact buffer": white except thin dark contact areas on the side of objects away
     from the sun (under props, at wall bases, around feet). Shadows pointing toward the sun,
     or large dark areas, mean the matrix/sun convention is wrong: report the first-pass log
     line.
  3. A/B by toggling at a sunlit pose (props on tables, railings, characters); try length
     0.25-1.5 m and thickness 0.1-0.5 m.
  4. Profiler (Ctrl+F3) on vs off: `gpu_contact_shadows_ms`, frame interval.
  5. With FSR upscale: contact shadows aligned with geometry, nothing past the render region.
- **Shadow flicker while moving (user report 2026-10-18; not AO, not
  `patch.shadows_submit_early` - toggled by the user):** ranked hypotheses.
  1. (a, most likely for distant objects) **Time-sliced cascade updates**,
     `r_shadowParallelPartialSliceUpdate 1` (Ultra; description "enable time slicing on
     cascade update except for slice 0"). O (IDB): the cvar's int (`dword_1432F8B98`) is read
     every frame in `sub_14032EE50` (directional shadow-map list), `sub_14032B280` and the
     cascade fit `sub_140330630`, combined with the frame parity `dword_1432FC64C`: slice 0
     renders every frame, the last slice on even frames, the middle slices on odd frames. So
     cascades 1-3 (beyond `r_shadowCascadeFars` 4 m) are refreshed every second frame while
     their matrices follow the camera: casters/receivers there alternate between a current
     and a one-frame-old map while moving. Live A/B: Lighting -> "Shadow map A/B" ->
     uncheck "Time-sliced cascade updates" (writes `cvar.r_shadowParallelPartialSliceUpdate =
     0`, applied within 30 frames; cost: every cascade every frame, watch the profiler
     `shadow` class).
  2. (c) **FSR 4 / TAA history with shadow-map aliasing:** the screen-shadow pass samples the
     2048^2 cascades at jittered positions (projection jitter) and adds an 8x8 interleaved
     ray offset (`screenshadows/useJitter`) that the engine's bilateral blur smooths;
     undersampled cascade texels shimmer when the camera moves and the history is rejected.
     A/B: F6 (FSR <-> engine TAA) while moving past the same object.
  3. (d) **Local-light partial updates:** `r_shadowSpotAndOmniPartialUpdateDynaThreshold`
     0.5 ("update of dynamic only if light shadow priority (screen cover) is above the
     threshold"): dynamic casters in lamp shadows update only for lights covering enough of
     the screen, so lamp shadows of moving objects can freeze and pop. Applies only to
     spot/omni lights; where it is read per frame is U (only the constructor and
     `sub_14050FF40` reference the cvar object). A/B via the Cvars tab (override 0) if the
     flicker is on lamp shadows.
  Steps: stand where it flickers, strafe slowly past the object; toggle (a) and repeat; if
  unchanged, F6 for (c); note whether the shadow comes from the sun or a lamp for (d).

**B3. SSR for non-water surfaces at S2.**
- Mechanism: half/full-res screen-space trace (d2enhanced `ssr_trace.hlsl` GGX VNDF +
  hierarchical depth as the engine's own `arkscreenspaceraytracing.decl`, E043), temporal +
  spatial filter, composite `colour += EnvBRDF * (SSR - IBL) * confidence` before FSR.
- Data: **requires** normal and roughness per pixel and the engine's IBL contribution to
  remove (otherwise reflections double): see section D. Radiance = pre-TAA colour of the
  previous/current frame (screen-visible only).
- Splice: S2 (needs the final lit colour). Can be D3D11 CS (d2enhanced) or D3D12 alongside FSR.
- Cost: unmeasured anywhere (d2enhanced has no per-pass timestamps; review "Diagnostics
  first"). H 0.5-1.5 ms half res.
- Risk: high for correctness of the IBL subtraction (review finding 5: the first of up to 3
  blended cube samples, LOD/6 roughness, mod BRDF; still open per the response item 5);
  screen-space disocclusion and edge fade; moving reflections need a reactive input for FSR
  (review roadmap 4).
- Payoff: high on wet streets, polished floors, metal; zero on rough materials.
- Effort: M (given a material source) to L (with a robust IBL replacement).
- **Implementation status (2026-10-18, E067, offline-verified, not run in game):** built on
  material source D3 (decision in docs/reflections.md section 1: no engine target holds
  material data; DXBC MRT = upgrade path, own G-buffer pass rejected for CPU cost).
  `void-inject/src/ssr.rs`, settings `ssr*` (live, default off), overlay tab Reflections,
  profiler class `ssr`. Splice: the TAA call 0x140327817 (deferred POSTPROC list with the
  engine TAA; immediate context right before FSR's pack with FSR). Hi-Z march, fixed 4x4
  GGX pattern + spatial resolve, premultiplied confidence, blend `lerp(colour, L, strength *
  F * c)` with a user surface model (roughness, F0, floors only, water mask) instead of an IBL
  subtraction. Offline GPU (default half res, medium): 0.31 ms at 3440x1440, 0.14 ms at the
  0.67 region.

**B4. SSGI bounce at S2.**
- Mechanism: d2enhanced visibility-bitmask GI: bounce = screen radiance x albedo estimate,
  added to the colour.
- Data: albedo (d2enhanced: patched `ro_diffusemap` sample or `exposed / (exposed + 0.6)`
  estimate), depth normals.
- Risk: double counting with the baked probe GI (static + sun + sky SH, E036/E037, already
  multi-bounce x3) and the engine's IBL; the additive term cannot be restricted to the
  ambient part of the colour. Its visibility part is better used as AO (B1).
- Payoff: low-medium (scenes are already probe-lit); Effort M. Not recommended before B1/B2.

### C. Ray tracing against the decoded static scene (DXR, D3D12)

Common facts:
- Hardware: RX 7900 XTX is RDNA3 with DXR 1.1 (H: vendor specification; not queried on this
  driver - check `D3D12_FEATURE_D3D12_OPTIONS5.RaytracingTier` first). No D3D11 path exists,
  so every RT option needs the S1/S2 command-list split and D3D12 interop.
- Geometry we have (O): static world `.bwm` (tutorial 1.16 M triangles / 17,279 instances,
  STATUS "Scene"), static entity `.bmodel` (E022/E023), runtime per-instance materials
  (E040), alpha-tested vegetation through `arkVegetation` + `$alphaTest` (E034). Depth
  agreement with the game: 97.7 % tutorial (E034), 86.3 % eotw world-only (E035).
- Missing (O): skinned `.md6` characters/hands (STATUS "Unresolved decisions"), wind
  animation (E034 residual), projected decals (E039, irrelevant for shadows), physics props
  move at runtime (E023), our live mode froze entity placements after the first frame and a
  per-frame walk of all models cost 47 ms on the game thread (E044).
- In-game cost reference (M, E044): our D3D12 raster of the same scene took 27-33 ms GPU
  without culling (shadows 7.4 ms); RT does not need culling but BLAS build/refit and memory
  are unmeasured (H: BLAS of 1-3 M triangles ~0.1-0.3 GB, built once per map; TLAS rebuild
  per frame for ~100k instances on eotw (E044 16.8k draws / 99k instances) H 0.3-1 ms).

**C1. RT sun shadows into `.y` (hybrid).**
- Mechanism: per pixel, reconstruct world position from the game depth, one shadow ray
  (RayQuery, inline) toward the sun with a small cone; `.y = min(engine CSM, RT)` so
  skinned characters and moved props still shadow through the engine CSM; static geometry
  gets unlimited range and contact-accurate shadows.
- Data: TLAS (static world + static entities, rebuilt when entities move), sun direction,
  depth, alpha-tested foliage via any-hit or opacity masks.
- Splice: S1 with a command-list split after `SCREENSHADOWS` (D3D11 part A -> fence ->
  D3D12 RT -> fence -> D3D11 part B). Interop payload small: depth copy (R32F, 19.8 MB at
  3440x1440) in, R8 mask out; H < 0.1 ms copies; fence bubble unmeasured. A second split per
  frame (besides FSR) doubles the flush/fence work (CPU cost unmeasured; d2enhanced's single
  split cost ~0.4 ms frame time including 1.8 ms FSR GPU while CPU-bound, lead).
- Cost: unmeasured; H 1-3 ms for 1 ray/pixel at 3440x1440 on a static TLAS; half with
  FSR upscale render scale 0.67.
- Risk: geometry mismatch (LOD, 2.3-13.7 % depth disagreement) -> self-shadow/light leaks
  -> needs normal-offset origins and the `min()` hybrid; wind-animated foliage shadows are
  static; vegetation receivers ignore `.y` (LO2); map changes need BLAS rebuild (E033);
  entity sync on the game thread must stay incremental.
- Payoff: high (penumbra-correct, long-range sun shadows; no cascade seams).
- Effort: L. Validate first: offline RT mask (new harness; the former `r3a_snapshot` was removed 2026-10-17, checkpoint 9282529) vs a game screen-shadow dump at
  the same camera, plus GPU ms at 3440x1440.

**C2. RT AO into `.x`.** Same TLAS, short rays (1-2 m), cosine-weighted; better than any
screen-space AO for off-screen occluders. Risk as C1 plus noise (needs a denoiser, section F).
Cost unmeasured (H 1-2 ms at 1 spp). Effort L (after C1). Payoff medium.

**C3. RT reflections at S2.** Needs everything B3 needs (normal, roughness, IBL removal)
plus hit shading: our arkDefault material port (R3b/E034), lights (L1), probe GI at hit
points (E036/E037), fog. Characters absent from reflections. Payoff high on mirror-like
surfaces; risk very high (two lighting models visible side by side, E037 residual ratios
1.2-1.5); effort L+. Only after B3 proves the composite contract.
Feasibility study 2026-10-18 (E069, docs/reflections.md section 3): no DXR code ever existed
(checkpoint 9282529 searched); static world/entity/material decoders survive, scene assembly
and the D3D12 renderer were removed; first spike = offline BLAS/TLAS of the eotw world +
1 reflection ray per pixel with timings and memory, before any hit shading.

**C4. RT GI.** Same as C3 for diffuse; competes with the engine's tuned baked probes.
Not recommended in this scope.

### D. Sources of per-pixel material data (for B3, B4, C3, F)

| Source | Normal | Roughness / metal | Albedo | Coverage | Cost | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| D1 DXBC patching of forward PS (d2enhanced `shader_patch.cpp`, `material.cpp`) | not captured (depth normals) | IBL LOD operand of the first cube sample (roughness = LOD/6, unverified), raw `ro_metallicmap` sample before remap | first `ro_diffusemap` sample (layers/decals/colour constants missing) | 3,705 forward PS patched, offline-verified (lead); opaque draws only; 141 PS without IBL sample | 3 extra MRTs during the forward pass: unmeasured; patch time at startup (lead) | shader-cache-specific but static for this build; semantics approximate (review 5); user rated it too volatile |
| D1' our DXBC patch (2026-10-18, E072, docs/reflections.md 2b; `void-inject/src/ssr/patch.rs`) | shading normal after normal mapping (captured at the IBL reflect `mad R, N, -t, -V`) | exact `m = roughness^2` (source of the `0.504055` mip-formula mad; d2enhanced's LOD/6 not corroborated) and F0 luminance (IBL luminance-factor lerp) | none | 3,037 / 3,037 lit forward PS with IBL patched, created and verified unchanged offline; one RGBA8 target at slot 2; no-IBL and particle shaders excluded (depth fallback) | one extra 4 B/px target in the forward pass (H 0.02-0.05 ms); ~1 s shader patching at load (H) | blended lit draws share RT0's blend state (H); in-game test pending |
| D2 own thin G-buffer raster (R3b, the former `void-inject/gpu.rs`, removed 2026-10-17; checkpoint 9282529) of decoded static world + entities at the game camera | full arkDefault layered normal map + vertex TBN | gloss/metal ranges per arkDefault (E020: gloss r 0.888, metal r 0.941 vs debug captures) | layered albedo | 97.7 % depth agreement tutorial (E034), 86.3 % eotw (E035); no characters, projected decals, vista/sky/translucent, wind | full scene raster: 17-24 ms unculled in game (E044); with frustum culling + LOD selection unmeasured (H 1-3 ms) | CPU traversal of ~100k instances off the main thread; per-pixel validity mask needed (compare with game depth, fall back where it disagrees) |
| D3 depth-reconstructed normals | geometric only | none | none | everything in depth | ~0.1 ms (H) | no normal-map detail; enough for AO/contact shadows/RT origins |
| D4 engine targets | none (no G-buffer) | none | none | - | - | depth, velocity, `_SSAO_ScreenShadows`, `ro_postfx_dof_refractionmap` RG16F (NOTES), `_IBLCubeMapArray` only. Corroborated 2026-10-18 (E067, docs/reflections.md section 1): full per-view target list (`sub_14036D0C0`), forward/depth/SSAO/screen-shadow/water write lists, 3 RenderDoc captures and 8,434 shipped PS output signatures: the only second targets are `sssBuffer`, `skyCloudMaskBuffer` and particle velocity |
| D5 shader substitution by hash (compile shipped renderprog source with an extra output) | as source | as source (final processed values, incl. `COMPUTE_IBL` result) | as source | per permutation | at load | needs the permutation macro set per cached shader (`ShaderCache_PC.str`/`.rpi` mapping not investigated); no runtime compiler in the exe (LO9) |

Reading: B1/B2/C1/C2 need only D3. B3/C3 need normal + roughness + the engine's IBL term;
only D1 (approximate) or D5 (exact, unexplored) provide the IBL term, D2 provides better
normals/roughness for static geometry but no IBL term (it would have to be re-evaluated from
`_IBLCubeMapArray` with the engine's parallax planes, U).

### E. Splice and interop summary

| Option | Splice | API | Cross-API sync per frame |
| --- | --- | --- | --- |
| A* | S0 | none | none |
| B1, B2 | S1 | D3D11 CS in the engine's deferred context | none |
| B3, B4 | S2 | D3D11 CS (or D3D12 with FSR) | none / shared with FSR |
| C1, C2 | S1 | D3D12 DXR | +1 split (before forward) |
| C3, C4 | S2 | D3D12 DXR | shared with FSR split |
| FSR 4 (G1) | S2 | D3D12 | 1 split (TAA) |

Common requirements (PLAN rules, E047/E048): engine-level callback hooks, D3D11-created
shared textures, fences only between command lists, no binding of an engine SRV as our RTV
(E048: unbinding hazard), off switch + A/B + profiler class per pass.

### F. Denoising and temporal story

- **FSR 4 itself:** the final temporal accumulator for everything injected before S2. It is
  an upscaler, not a denoiser: inputs should already be temporally stable; per-pixel 1 spp
  noise would read as shimmer. S1 inputs (AO/shadow) are consumed by the forward shader
  before FSR, so FSR also smooths them, and FSR 4 needs no reactive mask (SDK
  `super-resolution-ml.md`), but moving reflections (B3/C3) still benefit from one (review
  roadmap 4).
- **FSR Ray Regeneration 1.2.0** (AO, dominant-light visibility, direct/indirect
  diffuse/specular, specular occlusion signals; would fit C1/C2/C3 exactly): **not available
  on this GPU** - requirements "AMD Radeon RX 9000 Series GPU or later, DirectX 12 + Shader
  Model 6.6, Windows 11" (O, `docs/techniques/denoising.md` section 6); `ffxCreateContext`
  fails on unsupported hardware. Timings published only for RX 9070 XT (e.g. one indirect
  specular signal 5.34 ms at 2560x1440).
- **FSR Radiance Caching 0.9.0 (technical preview):** a neural cache trained online by a
  path tracer (`docs/techniques/radiance-cache.md`); requirements section lists only
  HLSL `CS_6_6`, optional WMMA backend flag; no GPU list found (bounded: that doc and
  `whats-new/index.md`). Not applicable: we have no path tracer.
- **Not in SDK 2.3 (O, kit listing):** the classic FidelityFX SSSR, Denoiser (shadow /
  reflection), CACAO, Brixelizer GI; `Kits/FidelityFX` contains only upscalers, frame
  generation, `denoisers` (Ray Regeneration) and `radiancecache`. The classic ones exist in
  the older MIT FidelityFX SDK 1.x (external, not vendored here).
- **Own filters:** d2enhanced's temporal/spatial filters after the review fixes
  (world-space reprojection with the engine matrices, normal check, variance clipping,
  resets on frame gaps/projection changes, premultiplied confidence; response items 1-8)
  are the reusable baseline; missing: camera-cut reset from the engine
  (`m_justTeleported` view `+48380`, NOTES) and per-pass timestamps. For C1 a small
  shadow-specific spatio-temporal filter (or the engine's own bilateral blur, LO3 U) is
  enough because sun rays are near-coherent.

## 4. Recommended order

1. **A: settings pass (S).** Cheapest and reversible, no splice. A1 anisotropy first (largest
   expected visual gain per ms), then A2+A3 (sun CSM 4096 + longer fars), A5, A4/A6 with
   CPU watched, A8.
2. **S1 splice + B1 own AO (S-M).** Establishes the record-time engine-callback splice that
   needs no interop, and fixes AO where the engine actually applies it (indirect only).
3. **B2 sun contact shadows (S-M).** Same splice and inputs as B1; covers the scale CSM
   cannot reach even after A2.
4. **C1 hybrid RT sun shadows (L)** once G1's split/fence mechanism is proven in game: reuses
   the decoded static scene and gives the largest shadow upgrade; `min()` with the CSM keeps
   characters correct.
5. **B3 SSR (M-L)**: implemented (E067) and fed by our own DXBC material target D1' (E072:
   per-pixel normal, roughness, F0 from the patched lit shaders, depth model as fallback);
   in-game tests pending (docs/reflections.md sections 2 and 2b).
6. Deferred: C2 RT AO (after C1, needs a denoiser), C3/C4 (L+), B4.

## 5. First concrete experiment per recommended item

1. **A1 anisotropy:** void-inject hook on `ID3D11Device::CreateSamplerState` logging
   `(Filter, MaxAnisotropy)` histograms; run 1 with Ultra (expect 0x55 samplers with
   MaxAnisotropy 1), run 2 with `image_anisotropy` float/int written to 16 at the first frame
   tick before loading the eotw save. Pass: anisotropic samplers report 16; screenshot A/B
   of the eotw street at a grazing angle; profiler static segment GPU delta. Then A2/A3:
   launch with `+r_shadowParallelSliceSize 4096` (and fars `5 16 45 150`), profiler `shadow`
   class and frame interval A/B at both eotw poses.
2. **S1 + B1:** hook the `SCREENSHADOWS` callback `sub_14032C7F0` (and `SSAO`
   `sub_140305BE0`) return; record a D3D11 CS into the job's context that writes constants:
   `.x = 0` (expect: ambient + IBL vanish, sun and local lights unchanged), then `.y = 0`
   (expect: sun light vanishes except on vegetation). This proves LO2's channel semantics,
   the target identity/format and the pass order in one run; then replace `.x` with the
   existing horizon AO port, A/B against HBAO+.
3. **B2:** with the S1 hook in place, a 16-step depth march toward `m_sunDirection` written as
   `min(.y, contact)`; debug view of contact-only visibility; profiler GPU ms; sweep steps
   8/16/32 at full and half res.
4. **C1:** offline first: a RayQuery shadow pass in a new offline harness (the former
   `examples/r3a_snapshot` is in checkpoint 9282529) on the RX010 /
   eotw snapshot (TLAS from the decoded static world + entities), output a mask at
   3440x1440 and its GPU time; compare with a screen-shadow `.y` dump captured through the
   S1 hook at the same camera (agreement %, leak/acne map by material). Gate: < 2 ms and
   > 95 % agreement on sunlit/shadowed classification outside character pixels before any
   in-game split.
5. **B3 (material source):** list `ShaderCache_PC.rpi/.str` entries to see whether each
   cached forward PS maps to a renderprog + macro set (D5 feasibility); in parallel measure
   D1's per-variant accuracy on the review's validation set (single/multi-probe, metal,
   tinted, wet decal) using the S1 constant-write trick as a reference (`.x = 0` removes the
   ambient probe term and the IBL term together: colour difference = AO x (ambient + IBL);
   separating IBL from ambient additionally needs one of them captured, e.g. D1's o3).

## 5b. Edge halo (vanilla bright outline, 2026-10-18, E082)

User: a faint bright 1-2 px rim around characters (whole silhouette), fences and props,
also with our DLL removed, on every preset, with anti-aliasing off.

**Cause: the depth-of-field combine sharpens the frame (O).** `ARK_RENDERPASS_DOF`'s
combine (`postfx_dof_combine.decl`, last draw before TAA; RenderDoc
`local/rdc/A_original/d2_original_frame16159.rdc` eid 29989, PS 77082) runs every frame,
also with no visible depth of field (far params ~(1, 1)), and applies
`colour = blur + (1 + sharpen * edge) * (colour - blur)` with `blur` = the
half-resolution `PostFX/DOF/diffuseMap` (1720x720, bilinear) and `edge` = 1 where the pixel
is brighter than the blur and not fog-coloured. Next to every dark silhouette the blur is
darker than the background, so the background's first 1-2 pixels overshoot. Evidence:
- `distortionParams.z = 0.75` in that capture (constant buffer read back); the formula with
  0.75 reproduces the pass output (mean error 0.0004 vs 0.0010 for "output = input").
- Paired edge statistic (background pixel 1 px from a silhouette vs the same surface 4 px
  further out, 64,538 pairs): before the combine median 1.000, after 1.067 (54 % of pairs
  > 5 % brighter, 25 % darker). Before/after/difference crop of a fence:
  `local/rdc/halo_f16159/dof_halo_crop.png` (white outline around every bar).
- The other reduced-resolution passes do not brighten edges in that frame: fog composite
  (nearest-depth upscale `atm/rendernearestupscale`, PS `ps_bb91920f716ba73e`: bilinear only
  when all four half-res depths are within `r_fogNearestThreshold` 0.0002 m, else the
  closest-depth texel) 1.016 -> 1.002; OIT resolve 1.000; AO `.x` and blurred sun shadow `.y`
  1.000 (contact darkening, no lighter fringe). The fog's bilateral blur weights depth with
  `(r_bilateralSharpness / r_znear)^2` = 100 (near) / 1 (far, > `r_bilateralDepthRange`
  10 m), i.e. separates surfaces 0.1 m (near) / 1 m apart.
- Source of the coefficient (IDB): `sub_140313E70` (per view per frame from
  `sub_140313DE0`, frame-graph build) fills the DOF settings at `rs + 0x5FA368` from
  `env/dof/settings/*` (map environment), or from the `r_Dof*` cvars when `r_DofDebugMode`
  is set; `+0x74` = `sharpen_coef`, `+0x97` = `use_sharpen`; `sub_140313980` stores `+0x74`
  as `distortionParams.z` (0x140313B69). Hence vanilla, every preset (presets do not touch
  it), independent of TAA. `r_DofSharpenCOef 0` / `r_DofUseSharpen 0` in the dump are only
  read in debug mode.

**Fix (`void-inject/src/dof_sharpen.rs`, setting `dof_sharpen`, live, default 1 = game until
the user confirms):** a post-hook on `sub_140313E70` (prologue checked, unit test against the
executable) multiplies `+0x74` by `dof_sharpen`; 0 makes the combine return its input
(depth of field itself unchanged). The hook is installed the first time the value is not 1.
Overlay: Lighting tab, card "Edge halo (game sharpening)" (game coefficient and applied
value).

**In-game A/B (all live; cvars through the overlay Cvars tab or `cvar.<name> = <value>` in
`void-inject.ini`; every listed cvar is read per frame, IDB readers in E082):**
1. `dof_sharpen = 0` (fix) and back to 1: the outline around characters/fences disappears
   and returns; nothing else changes. Expected to be the cause.
2. `cvar.r_skipDOF = 1`: removes the whole DOF combine (halo gone too; no depth of field).
3. If 1-2 do not remove it: `cvar.r_skipFog = 1`, `cvar.r_skipLightScattering = 1`,
   `cvar.r_skipFogBlur = 1`, `cvar.r_fogNearestUpscale = 0`, `cvar.r_skipGodRays = 1`
   (fog / scattering), `cvar.r_skipSSAO = 1` and `ao_mode = hbao/void` (AO),
   `cvar.r_skipBloom = 1`, `cvar.r_skipOutlines = 1` (interactable outlines), one at a time.
Not recommended as a fix: `r_DofDebugMode 1` (switches every DOF parameter to the cvar
defaults, focal distance 5 m).

## 6. New findings (for the ledger)

| Id | Observation | Address / source | Label |
| --- | --- | --- | --- |
| LO1 | Frame built as named render passes with callbacks; order SSAO -> SHADOWS -> IL -> SCREENSHADOWS -> TLFLIGHTCULL -> FORWARD | `sub_14034FE00`, table in section 2 | O |
| LO2 | Forward lighting reads `texScreenshadows.xy`: x = SSAO for indirect, y = sun shadow; vegetation ignores y | `lighting.decl:79-143`, `arkvegetation.decl:7`; target `_SSAO_ScreenShadows_%p` `0x14036DC0D` | O (link name->target H) |
| LO3 | Screen-shadow pass = depth reconstruct + `ComputeParallelShadow` + bilateral blur materials | `screenshadows.decl`, strings `0x141B2A2B0..` | O (blur order U) |
| LO4 | Anisotropic samplers take `MaxAnisotropy = image_anisotropy`; Ultra = 1 | `sub_140378280`, `sub_1402EB830`, dump | O (anisotropicSampler decl U) |
| LO5 | CSM fars: exactly 4 floats, rescaled to `r_zfar`; slice size flagged 0x4000 | `sub_140330630`; ctor `0x141377ABB` flags 0x4002 | O |
| LO6 | Local-light shadow size from projected light radius, pow2, capped by r_max(Extra)ShadowMapResolution | `sub_140330AC0` | O |
| LO7 | `r_shadowCullDistance` gates dynamic casters (squared); Ultra 75, d2enhanced Ultra+ 60 | `sub_140330020`, `hooks.cpp:188` | O |
| LO8 | `r_lodZoomMultiplicator` multiplies the per-view LOD zoom; Ultra 0.6 | `0x1403522D1` | O (direction H) |
| LO9 | No runtime HLSL compile entry; precompiled shader cache only | `sub_1403694A0`, string/import search | O (bounded) |
| LO10 | No opaque SSR in shipped renderprogs | 270 decls searched for `ScreenSpaceRayTrace` | O (bounded) |

d2enhanced claims corroborated here: no G-buffer / forward shading (via shipped source),
`job+0x70` view pointer, Quality cvar values (via dump), `r_shadowParallelSliceSize` flag
0x4002 (ctor). Not corroborated: IBL cube size/mips, FSR 4 timing on this machine, the
`0x4000` = command-line-only meaning, SSGI/SSR quality and cost.
