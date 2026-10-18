# R track: replace the original game's renderer in-process

Decision 2026-10-16 (user): pursue option 3 — keep the original executable running the
simulation (gameplay, AI, animation, physics, scripts, audio, UI logic) and replace its
D3D11 render back-end with a new D3D12 renderer driven by the engine's own frame/scene
description. Reasons given: a full reimplementation needs every system recovered (movement
"feel" alone depends on data-driven camera, bob, FOV, and animation systems); patching
individual passes (d2enhanced) is too volatile. The void-rs reimplementation (M-track) is
paused, not discarded: its format, material, and asset work is reused here.

Out of scope: modifying `d2enhanced/` (separate project). Experiments that need it disabled
must do so reversibly and record it.

## Approach

1. **Interface = engine scene/frame data, not D3D11 calls.** Read the render world
   (per-entity `idRenderModel` instances, lights, view `renderView_t`/`idRenderView`) and/or
   the per-view draw-surface lists, and render them with our own D3D12 renderer.
2. **Structure layouts from the engine's reflection metadata.** The executable registers
   class and member tables (name, type, offset, size) for thousands of classes at startup
   (see "Reflection metadata" below). Extract them offline into a type database instead of
   guessing offsets.
3. **Assets:** decode from the shipped archives with void-rs (static world, models, textures,
   materials, shader source) where possible; reuse the game's GPU resources (skinned
   vertex buffers) via D3D11/D3D12 interop where they only exist at runtime.
4. **Incremental takeover with validation at each step** (as RX001/RX002): read-only capture
   -> shadow rendering compared against the game's own depth/colour -> replace passes.

## Phases (each with evidence gates)

| Phase | Content | Gate |
| --- | --- | --- |
| R0 | Architecture map: reflection type DB; front-end/back-end boundary; render threads and deferred contexts; resource ownership; UI (Iggy) path; where to skip original 3D rendering | Documented structures and entry points with addresses and provenance |
| R1 | Injection scaffold: Rust `cdylib` loaded by a proxy DLL, logging, narrowly-scoped `unsafe` hooking layer, crash safety, version check by exe hash | Loads and unloads cleanly; no effect on the frame |
| R2 | Read-only frame capture: per frame, dump view matrices and the visible entity/surface list with original ids | Reprojected capture matches the game's depth (RX001-style) |
| R3 | Shadow renderer: our D3D12 renderer draws the captured frame offscreen (depth, then materials) | Depth/material comparisons within stated tolerances |
| R4 | Takeover: skip original opaque passes, composite ours, keep UI; then lighting, shadows, decals, particles, sky/water, post | Feature-by-feature comparison; explicit list of not-yet-replaced features |

`unsafe` policy: the injection crate necessarily uses FFI and raw memory reads. It gets a
crate-level exception to the workspace `unsafe_code = forbid`, confined to small modules
(process memory access, hooks, D3D interop) with documented invariants; all parsing of
game memory goes through checked readers.

## R0 findings (in progress)

Executable SHA-256 `b671f824...6180` (E001); IDB copy `local/ida/`.

### Reflection metadata (O)

- Class registry: `sub_141413800` (0xE0146 bytes) fills 0x40-byte class records:
  `{name, base name (strings via sub_1400010B0), u32 size, RTTI type descriptor, 0,
  member table pointer, 0, 0}`. Example: `idRenderModel`, base
  `IIndirectLightingDynamicObjectHost`, size 0x1C40, members at `0x143AD1110`.
- Member tables are filled by per-class initializers (e.g. `sub_141532F90` for
  `idRenderModel`, referenced from the CRT initializer table at `0x141A921C0`): 0x48-byte
  records `{type name, type modifier ("*", "[2]"), member name, u32 offset, u32 size,
  RTTI descriptor, flags (e.g. 2048), bitfield getter/setter (for 1-bit members)...}`.
- `idRenderModel` (per-entity render instance) members include `m_surfaceList` (8),
  `modelMatrix[2]` (32), `inverseModelMatrix[2]` (160), `mvMatrix[2]` (288), `mvpMatrix[2]`
  (416), `m_decalDataListAdded/Linked`, `m_worldSpaceAabb` (6208), `m_renderModelParms`
  (6240, 240 bytes), `gameParmBlock` (6480), `m_name` (7088), `world` (7160),
  `m_surfaceListPtr` (7168), `m_activeMaterialEffect` (7184), `index` (7192), visibility and
  shadow flags. `[2]` matrices suggest double buffering between front-end and back-end (H).
- Registered render types include `renderView_t`, `idRenderView`, `idRenderWorldLocal`,
  `idRenderWorldCommitted`, `idRenderLight`, `idRenderLightParms`, `idRenderLightCommitted`,
  `idRenderModelParms`, `idRenderModelSurface`, `drawSurf_t`, `idRenderSystemLocal`.

### Leads from d2enhanced/re/NOTES.md (not yet corroborated here)

`RenderFrame` 0x140361BC0 (called from `sub_1403619D0`), `RenderAllViews` 0x14035DC60,
large render function `sub_140351780`; 8 D3D11 deferred contexts; tiled forward+ shading
(no G-buffer); post chain functions; game simulation in fixed 120 Hz tics
(`sub_140059990`); swap chain creation `sub_140368B80`; `idRenderView` member offsets
(e.g. `m_viewMatrix` 29776, `m_renderWidth` 30792).

### Reflection type database (E031)

Language decision (user, 2026-10-16): Rust for the renderer as well; HLSL shaders; direct
D3D12 through the `windows` crate (not Bevy/wgpu, which hide queue/PSO/interop control and
own the frame loop); vendor SDKs (FidelityFX, Streamline) through their C ABIs only.

`crates/void-reflect` statically evaluates the static-initializer table (26,025 functions,
`0x141A7EF08..0x141AB1CB8`) plus the class registry `sub_141413800`, records writes to the
data section, and reads class records (0x40) and member records (0x48):
`void-reflect --exe <Dishonored2.exe> --out local/reflect` -> `classes.tsv`, `members.tsv`.
Result for exe `b671f824...6180`: 5,844 classes with member tables, 30,346 members; 1,817
stores outside the evaluated subset (reported, not interpreted).

Validation: every `idRenderView` offset recorded independently in d2enhanced's notes matches
(`m_viewMatrix` 29776, `m_renderWidth` 30792, `m_subPixelJitter` 30800, `m_resolutionScale`
30912, `m_sunDirection` 48344, `m_justTeleported` 48380, ...): those leads are now corroborated.

Selected layouts:
- `idRenderModelParms` (240 B): `origin` 0, `axis` 12, `scale` 48, `m_customMaterials` 64
  (list of `idMaterial*`), `customSkin` 80, `fovScale` 84, `modelDepthHack` 88,
  `maxVisibleRange` 96, `modelDepthSort` 200, `referenceModel` 208, `viewport` 216, bit flags
  (`isStatic`, `castShadow`, `viewOriented`, ...).
- `idRenderLightParms` (232 B): `m_axis` 4, `m_origin` 40, `m_linearColor` 52, type/state/
  class/culling 92-104, distance/angular falloff and cutoff 116-144, scatter and area-light
  parameters, bit flags (`m_visible`, `m_castShadows`, `m_isSun`, `m_scatterOnly`, ...).

Limits: bitfield members are reported with offset 0 / size -1 (bit position is in their
getter functions, not yet decoded); `drawSurf_t`, `idRenderWorldLocal`, and
`idRenderSystemLocal` are not found as class records with member tables (different record
shape or registration path; open).

### Scene access path (O, IDB; addresses for exe `b671f824...6180`)

- `RenderAllViews` `0x14035DC60` (`a1` = frame view requests): requests list at `a1+0`
  (pointer) / `a1+8` (count, low 24 bits), 15,024 bytes each; per request: view index `+24`,
  flag `+29`, world `+32` (`idRenderWorld*`; vtable `+192` = `RenderViewForIndex` -> the
  persistent `idRenderView`), the game's `renderView_t` copied from `+48`.
- `idRenderWorldLocal` (opaque type; layout from its accessors): render models `ptr +132408 /
  count +132416` (`RenderModelForIndex` `0x14023B020`), lights `+516464 / +516472`
  (`0x14023AFD0`), views `+528776 / +528784` (`0x140239740`). Counts use the low 24 bits
  (`idList`). Vtable slots at `0x141B08990`.
- `idGameLocal` (3,413,504 B, vtable `0x141C940A8`, constructor `0x1408A37C0`, allocated in
  `AllocGame` `0x1408B2110`) holds `m_renderWorld` at 2001216.
- `idRenderModel` (per entity): matrices, `m_renderModelParms` (origin/axis/scale, custom
  materials), `m_surfaceList` of `idRenderModelSurface` (128 B: `material`, `geometry`
  (`idTriangles*`), `binaryModelId`, `joints` / `previousFrameJoints` buffer views (GPU
  skinning), index range, `m_decalMap`, particle stage, instancing).
- `renderView_t` (14,784 B): `fov_x/fov_y_RADIANS` 156/160, `vieworg` 168, `viewaxis` 204,
  `explicitProjectionMatrix` 256, `m_environmentParms` (640 parameters) 3008, fog LUT parms
  14600, colour cubes 14704/14720, lens dirt: the game's per-view fog/environment/grading
  inputs are available to a replacement renderer.
- Bitfields: decoded from getters (`movzx eax, byte [rcx+off]; and/shr`), e.g.
  `idRenderModel` byte 7216 bits 0-7 (`m_unlinked` .. `m_useLargeObjectBBoxSHInterpolation`),
  `idRenderModelParms` byte 92 (`isStatic`, `m_semiStatic`, `isBrush`, `castShadow`, ...);
  924 one-bit getters and multi-bit fields use other patterns (reported, not decoded).
- `drawSurf_t` is registered opaque (size -1, empty member table): back-end internals are
  not reflected, which favours interception at the render-world level.

### Loader

The executable statically imports `DINPUT8.dll` for one function (`DirectInput8Create`):
a `dinput8.dll` proxy in the game folder loads at process start, forwards that export to
the system DLL, and does not collide with d2enhanced's `d3d11.dll`. Experiments that need a
clean renderer still disable d2enhanced reversibly (user agreement 2026-10-16).

### R3a: in-process D3D12 depth renderer (E032)

`crates/void-inject`: `gpu.rs` (own D3D12 device; depth-only instanced renderer, reversed-Z
D32, VS-only PSO compiled at runtime with FXC `vs_5_1`, fence-synchronised readback),
`assets.rs` (static world `.bwm` + static entity `.bmodel` via void-rs), `r3.rs` (render
thread; F10 in game copies the live camera and static entity transforms from
`idRenderModelParms` and renders offscreen), example `r3a_offline` (same code outside the game).

Validation (offline, RX001 P1 pose, 1920x1080, world only) against the **game's own depth
capture** `local/ref/RX001/P1/dump0_depth_r32f_1920x1080.bin`: both hit 85.4 % of pixels,
**89.2 % of those within 1 % depth** (= the RX001 ray-caster result at this pose), game closer
4.3 % (entities not drawn in world-only), ours closer 4.9 %, game sky / ours geometry 14.6 %
(sky dome drawn at infinity in game), game geometry / ours empty 0.0 %.
Reproduce: `cargo run --release -p void-inject --example r3a_offline -- <ROOT>
15.95,12.02,-3.14,166.0,0.2 local/r3a_offline local/ref/RX001/P1/dump0_depth_r32f_1920x1080.bin 1920 1080`.

### Map detection (E033)

`idRenderWorldLocal::InitFromMap` (`sub_14023A5D0`) assigns the map name to an `idStr` at
world + 132328 and loads `"%s%s/%s.bwm"` (directory, base, base), then the probes network and
umbra tome. The R3 renderer reads that name per request and reloads the static world when it
changes, searching all archive groups (preferring the map's group for its models). All 36
shipped `.bwm` files (groups 0-2) follow `<dir>/<base>/<base>.bwm`.

### R3b: material G-buffer with alpha test (E034)

New crate `void-assets` (no Bevy): `Archive` (name table over all groups, preferred group),
`texture::load` (stored BC/R8 mip chains incl. streamed `_mipN` parts, capped at 1024 px),
`material::Preparer` (support classification, arkDefault resolution, layer textures) and
`pack_params` (the 84-word `ArkParams` layout shared with the viewer shader).
`void-inject/gpu.rs` now draws with the full vertex stream (stream-1 20-byte block: half uv0/uv1,
snorm8 normal/tangent, unorm8 colour), a 16,384-slot bindless SRV table of native BC textures
(slot 0 white), a material structured buffer (84 params + 15 texture slots), and an HLSL port
of `ark_layers.wesl` (masks, layer blend, `$alphaTest` clip at 0.5, gloss/metallic ranges,
BC5 normal unpack and vertex TBN) writing albedo+metallic (sRGB RGBA8) and world normal+gloss
(RGBA8); depth unchanged. Materials of other programs are drawn opaque **magenta** and listed
in `materials.tsv`; editor-program materials are skipped.

- `arkVegetation` (O, `renderprog/arkvegetation.decl` standard pass): albedo is `$diffusemap`
  at `m_layerTexCoords[0]` with `clip(a - 0.5)` under `$alphatest`, then the shared HQ normal/
  gloss/metallic helpers; vertex shader adds wind animation, pixel shader translucency. We
  render it through the arkDefault path restricted to layer 0 (no colour constant, no
  vertex-colour multiply) and report wind/translucency as ignored (Partial).
- Validation, RX004 frame 7487 view (`r3a_snapshot` on `snap_..._f7741` vs
  `dump12_depth_r32f_3440x1440.bin`), world + filtered models:

  | Renderer | within 1 % | game closer | ours closer |
  | --- | --- | --- | --- |
  | R3a depth-only (E032) | 94.3 % | 0.2 % | 5.5 % |
  | R3b, arkDefault only (vegetation magenta/solid) | 94.3 % | 0.2 % | 5.5 % |
  | R3b + arkVegetation alpha test | **97.7 %** | 0.6 % | 1.7 % |

  Remaining residual (H): wind displacement of grass, TAA jitter / resampling at edges.
- Materials at this view: 118 basic, 184 partial (mostly ignored `occlusionmap`, emissive),
  27 unsupported (arkDecal 16, arkVista 7, arkSky 1, no program 2, 1 decl not UTF-8);
  1,131 textures, 0 upload failures, upload 1.2 s.
- Visual check (not a metric): `albedo.png` shows the same props, posters, grass and
  surfaces as the game colour dump; `normal.png` shows ground +Z, walls +-X/Y with
  normal-map detail. Lit colour is not comparable yet (no lighting).
- Caveat (user, 2026-10-16): d2enhanced was active for the RX004 dumps; whether its SSGI/SSR
  passes ran is unknown. The colour dumps are therefore **not a clean original baseline**;
  numeric colour comparison needs a new F8-equivalent capture with d2enhanced's effects
  disabled (or a capture method independent of d2enhanced). Depth is less exposed
  (H: those passes read depth but do not write it; upscaler jitter may still shift edges).
- Not covered: decals, vista, sky, translucent/refractive surfaces, custom entity materials
  (`idRenderModelParms` +64/72), skinned models.

Reproduce: `cargo run --release -p void-inject --example r3a_snapshot -- <ROOT>
local/inject/snap_1792173566_f7741 local/r3b local/ref/RX004/dump12_depth_r32f_3440x1440.bin 3440 1440`.

### Lighting plan (decision 2026-10-16: full reimplementation)

User decision: every lighting component is reimplemented (no reuse of the game's GPU
results such as screen shadows, probe SH buffers or shadow maps); game render outputs are
validation references only. Scene state read from game memory (camera, transforms, light
parameters) remains an input.

Model (O, shipped `lighting.decl`, `brdf_ward.decl`, `lightcontext.decl`, `ward.decl`,
`fresnel.decl`, `orennayar.decl`, `arkdefaulthelpers.decl`): forward per-material lighting;
parallel (sun) light with screen shadows; indirect = tetrahedral probe network SH
(`indirectlighting/*`, updated on the GPU at runtime) + IBL probes, scaled by
min(SSAO, material AO); clustered half-point / point / spot / area lights (<= 32 per type
per tile) with shadow maps; Ward specular, Oren-Nayar diffuse, Fresnel from IOR derived by
`NOTOUCHY_MapGlossAndMetalToLightingPipelineValue_NOTOUCHY`; attenuation
`(lerp(1, 1/r, decay))^2 * smoothstep(cutoff, falloff, r)`, cones `1 - sat(dot * spotParms.x + spotParms.y)`.

Stages, each validated before the next:

- **L1** direct lights without shadows; needs the CPU fill of `tlf/light_t` (`mDecay`,
  `mSpotParms`, `mSpecularAttn`, colour scale) from `idRenderLightParms` (IDA) and the sun's
  `parallelLightColor/Dir`; also per-material `specularTintDielectric`, `diffuseAlbedoMetallic`,
  `sheen`, `sheenImportance`, occlusion maps.
- **L2** shadows: own sun shadow (transcribe `screenshadows.decl`/`shadow*.decl`) and spot/omni maps.
- **L3** indirect: probe network file format + transcribed runtime SH update
  (`indirectlighting/update/*`), principal-direction packing, IBL probes.
- **L4** SSAO, fog (`fogHelpers`), exposure/tonemap/post.

L1 progress (O, IDB, exe `b671f824...6180`):

- GPU light struct `arkTiledForwardLight_t` (112 bytes, reflection table) is filled per
  visible light by `sub_1402F7260` (loop; position = idRenderLight +40; filter: +96 view-mask
  bits, type +92 < 4, flags bit 0) via `sub_1402F8670`. From idRenderLight: falloff +124,
  cutoff +128, decay = clamp01(+136), colour +52, specularAttn +152, scatter +156/+164/+172,
  scatterTurbulence clamp01(+176), axis rows at +4/+16. Spot: dir = -row0, c_in = +140,
  c_out = min(+144, c_in - 0.001), spotParms = (1/(c_out - c_in), -c_in/(c_out - c_in),
  sqrt(1 - c_out^2)/c_out), originOffset +160. Half-point: dir = row0, spotParms =
  (-1/(+140), 1, 0). Point: spotParms 0 (row0/row1 used only with flag byte1 bit 1).
  Area: dir row0, spotParms from +200..208 with `1000 - 999.99 * pow(+200, 0.01)`.
- `parallelLightDir/Color` are bound from the view at +48344 / +48368 (`sub_140311AA0`);
  candidate writer `idRenderView_Setup` (0x140374E40, writes near 0x1403762BE). Not yet traced.

L1 implementation (2026-10-16): forward direct lighting in the R3 pixel shader
(`gpu.rs` `DirectLighting`): Ward specular + Oren-Nayar diffuse with `FresnelAccurate`,
sheen colour, `NOTOUCHY_MapGlossAndMetalToLightingPipelineValue_NOTOUCHY` (no specular maps),
sun as parallel light, point/half-point/spot lights with the recovered attenuation and cone
parameters; no shadows, no indirect, no area lights (type 3 counted and skipped), no light
cap per tile. Inputs: material `specularTintDielectric` (default 0.04), `diffuseAlbedoMetallic`
(0), `sheen` (0), `sheenImportance` (1) from shipped renderParm defaults; lights read live
(`scene::read_lights`, idRenderLight +0/+1/+4/+40/+52/+92/+96/+124/+128/+136/+140/+144/+152);
sun read live from the view's environment block (`env/atm/sky/SunAngleTheta`, `SunAnglePhi`,
`sunColor`; `sub_140017D30`): dir = (sin t cos p, sin t sin p, cos t), colour =
sunColor x 89939.727 / (1750 / 4 pi) x 0.889. Corroboration: the formula gives 645.84 for
sunColor 1, equal to the runtime sun light colour observed in RX004/RX007 snapshots.
Outputs `lit.rgba16f` + `lit.png` (preview exposure only). F3 snapshots now also write
`sun.tsv` and `light_parms.tsv` for offline renders. Not yet validated against the game.

RX009 (F3 `snap_1792177070_f3773`, eotw, same camera as RX006-8): live sun read = theta 26,
phi 100, sunColor (0.215861, 0.171441, 0.116971), identical to the shipped env decl
`env/campaign/edge_of_the_world/eotw_p/global_ext01.decl` (O) -> parallelLightDir
(-0.076, 0.432, 0.899), parallelLightColor (123.9, 98.4, 67.2). The neutral 645.84 type-4
"sun" render light is a separate object (H: shadow/volumetric use). Lights: 621 (88 point,
44 half-point, 397 spot, 91 area skipped, 1 sun). Offline L1 render: sun-facing surfaces lit,
back faces black (no indirect), decals black (unsupported). Qualitatively plausible only: a
numeric comparison with the game's HDR colour needs shadows (L2) and indirect (L3) first.

L2 sun shadows (2026-10-16):

- Game model (O, `shadow.decl`, `shadowcommon.decl`, renderParm decls, cvar dump): cascaded
  parallel shadows, slice chosen by view depth against `parallelLightSliceFars`
  (`r_shadowCascadeFars` 4/12/30/75, `r_shadowParallelSliceCnt` 4, slices 2048^2 32-bit,
  `r_tightenParallelShadows` 1), `SampleShadowMapOptimizedPCF` (quality = `$shadowFilteringQuality`,
  decl default 1), comparison sampler linear/point-mip/clamp/Less, `ComputeShadowOffset` with
  decl defaults: normal offset off, slope scale off, `shadow/bias` 2.0 (live value not read),
  cloud shadows `GetFastCloudShadowing` (cloud texture/env parms; not implemented).
- Ours: own cascade construction (bounding sphere per view-depth slice, orthographic along
  the sun, 1 km caster range, texel-snapped), depth-only caster pass with alpha test,
  rasterizer slope bias + 2 cm receiver bias, the game's quality-1 PCF kernel, no shadow
  beyond 75 m (H; the engine's behaviour there is not recovered), no culling of casters.
- RX009 check (lighting ratio game colour / albedo vs ours, `light_ratio_rx009.png`): tree
  shadow patterns on the facade and street, shaded risers and the area under the tree agree
  visually; the game's shadows are lifted by indirect light (L3). Not a numeric validation.

L3a indirect lighting, steady state without dynamic relight (2026-10-16):

- Inputs (O): probe upload `sub_140291430` (probe info factors: static x AO_global,
  dynamic, sky x AO_sky, sun x AO_global, ambient; SH = `*_WithOccluders`), built by
  `sub_1402B75B0(filtered = 1)` from static SH + *filtered* sun-factor SH + *filtered*
  sky-occlusion SH + occluder SH x occluder state (states not recovered -> 0, H);
  accumulation inputs `sub_1402B4380`: parmsFloat (1, 1), ambient = env
  `indirectLighting/ambient`, sky ZH from env skyZenith/skyHorizon via `sub_1402B0370`
  (40 equal-area rings, transcribed incl. its 2*2pi/40 ring weight), sun colour =
  `parallelLightColor` (`atm/sky/sunColorAtGroundLevel` is bound from the same view field).
- Ours: probe SH on the CPU (`void_assets::probes`), per-vertex tetrahedron lookup for the
  static world (reference buffer + per-instance offsets), per-object lookup at the model
  origin for entities (H), principal-direction packing (VS) and unpacking (PS), ambient
  `1/pi * albedo * Fdr_Ward * irradiance`, AO = 1. Live env parms (skyZenith/Horizon,
  ambient) read per request; F3 writes `env.tsv`. Bounce static/dynamic/sun = 1 (H).
- RX009 (eotw, RX008 colour at the same camera, depth-agreeing pixels): game/ours luminance
  median **3.94** (p10 1.2, p90 12.9), log correlation 0.69; per material 1.0 (tree bark) to
  11.9 (stairs). Not matched: global scale (H: exposure / pre-exposure of the dumped colour
  buffer, `EngineLuminanceFactor`), dynamic relight term, IBL specular, SSAO/material AO,
  fog, and our shadow approximation.
- **Dump scale (O, shader source):** every forward/lit shader writes
  `RangeCompressColorToR11G11B10(finalizedColor)` (global.decl): with
  `$useCompressedHdrBuffers`, the stored colour is absolute / (EngineLuminanceFactor x
  view_white_level), EngineLuminanceFactor coming from the GPU auto-exposure
  (`postfx/luminance/autoexposure.decl`: histogram, temporal adaptation, 1 / white-level
  luminance in Arkane units). d2enhanced reads a `compressedHdr` flag for the same buffer.
  So F8 colour dumps carry an unknown per-frame scale; `r_postProcessDebugExposure` has no
  readers in retail (O: only its registration). Until our own auto-exposure exists (L4),
  colour comparisons are scale-invariant: RX009 quintiles of our lit/albedo give game/ours
  5.2, 4.1, 3.7, 4.0, 2.2 -> relative to the median factor our sunlit pixels are ~1.8x too
  bright and shaded ones ~1.3x too dark (contrast too high; H: missing indirect terms).
- **Dump scale resolved (O + D):** `SetPerViewShaderParms` / `sub_140325910` bind
  `view_white_level` and `PostFX/Luminance/white_level` = 1 unless
  `r_postProcessExposureUseToneMappingWhiteLevel` is set (cvar dump: 0), so stored colour =
  absolute / EngineLuminanceFactor = absolute x 255 x LF / 139.26. Auto-exposure
  (`autoexposure.decl`, shipped source): 128-bucket histogram over -40..100 dB of
  downsampled luminance x 139.26, best window of dB(255)/(140/128) = 44.03 buckets, tail
  clamped to [bucket(adapt_min), bucket(adapt_max) - 44.03]; eotw env: adapt 1..500 cd/m2,
  EV 0, clip 0/1 (decl defaults) -> LF in [1.0, 1.956]. In bright daylight LF clamps at
  1.956 -> **dump = absolute x 3.58** (measured median 3.94). With that fixed factor (no
  fit), RX009 game/ours quintiles: 1.45, 1.14, 1.04, 1.12, 0.61 -> overall level within
  ~10 %; our sunlit pixels ~1.6x too bright, darkest ~1.45x too dark.
- **Per-material split by our sun visibility** (lit.a; `snap_1792178666_f1778` render vs
  RX008 dump 8, game / (ours x 3.58), no fit): sunlit trims 0.99, mud rock 1.07, cliff 1.11,
  planks 1.14, blood puddle ground 1.08, harbour floor 0.78, white stone 0.83 -> direct sun,
  BRDF, cascaded shadows and the exposure chain agree within ~0.8-1.15 for most sunlit
  materials. Outliers: large harbour stone blocks 0.30 (under the harbour water, which we
  do not draw), stairs 3.19 (material/albedo H). Shadowed: open-ground materials 0.37-0.64
  (ours too bright; H: missing material AO and SSAO), white stone 14x and sawmill sign 170x
  (ours far too dark; H: per-object probe position, two-sided handling or probes inside
  geometry).
- **Bug fixed (2026-10-16):** per-vertex probe references were indexed with
  `SV_VertexID - BaseVertexLocation`; `SV_VertexID` is the index-buffer value (base vertex
  not included), so every static-world vertex fetched an unrelated tetrahedron (faceted,
  random ambient). After the fix (render `r3a_1792179137_f0`), game / (ours x 3.58):
  sunlit floor 1.01, trims 1.00, mud 1.03, planks 1.02, sawmill sign 1.02, algae blocks
  1.04, bushes 1.11, cliff 0.79, white stone 0.81; shadowed floor 1.37, trims 1.27, mud
  1.54, planks 0.92, sign 1.27, algae blocks 0.94, bushes 1.00. Remaining: stairs 3.2x
  (sun and shade; material H), white stone in shade 13x under the sawmill arches.
- Debug modes (`VOID_DEBUG`, lit target): 1 = unpacked ambient irradiance, 2 = probe path
  (red per-vertex, green per-object, blue none) x weight sum; lit alpha = sun visibility.
  Mode 2 shows every static-world pixel on the per-vertex path with weights, including the
  dark arches. eotw probes have no occluders (0 records), so occluder states are not the
  cause. Probe term magnitudes (`void-assets/examples/probe_terms.rs`, DC luminance, eotw):
  median static 0.012, sun 0.70, sky 3.46; several probes near the sawmill have near-zero
  baked terms -> H: the missing runtime dynamic relight (samples x probe lights, dynamic
  sun with shadows; `updateprobe.decl`) supplies their light in the game (next: L3b).
- Live env (F3 `snap_1792178666_f1778`): skyZenith 0, skyHorizon (37.7451, 52.4118, 55),
  ambient 1 -- equal to the shipped eotw env decl (O).
- **Dynamic Sun relight is off in retail (O):** only `indirectLighting/update/updateProbe.material`
  is referenced by the exe (string search; `updateprobe_dynamicsunupdate.material` exists in
  the archives but has no reference), so the probe Sun term is the baked filtered Sun SH x
  `sunColorAtGroundLevel`. The runtime relight covers probe-list local lights and emissives
  only (`updateprobe.decl`: 128 samples, `1/max(1, r)` distance clamp, `(1 - Fschlick(F0))`
  x albedo/pi x N.L, Lanczos 3).
- **Multi-bounce (O, implemented):** `sub_1402B4380` calls `sub_140291430` per probe on load or
  cvar change with `r_ILProbeUpdate_ContributionFactorMultiBounces` (3.0) and
  `r_ILProbeUpdate_MultiBouncesDebugFlags` (7): static += 3 x `*_Initial`[0], Sun += 3 x [1],
  sky occlusion (broadcast to RGB) += 3 x [2] (results +1536/+1644/+1752; negative factor
  replaces the base). Base SH from `sub_1402B75B0(filtered = 1)`: static, filtered Sun,
  filtered sky, + occluder SH x occluder state. `VOID_MULTIBOUNCE` overrides (default 3).
  Effect at RX008 dump 8: shaded floor 1.37 -> 1.31, mud 1.54 -> 1.44, white stone 13.0 -> 9.2.
- **Bug fixed: mirrored instances.** Debug views 3/4 (provoking-vertex probe indices) showed
  the dark white stone used ordinary probes (Sun/sky terms not small); view 1 showed negative
  unpacked irradiance and view 5 (pixel normal in the vertex frame) tsN down to -0.82, i.e.
  `SV_IsFrontFace` was inverted. Instances with a negative-determinant transform reverse
  the winding; the VS now passes the determinant sign, the PS flips `front`, and the tangent
  handedness `t.w` is multiplied by it. Result (render `r3a_1792179927_f0`): white stone
  shaded 9.2x -> **1.49x**, sunlit 0.73; shaded samples 3,018 -> 8,629; other materials
  unchanged. Remaining shaded ratios 1.2-1.5 (floor, trims, mud, sign) are ours too dark.
- Debug modes added: 3/4 = probe indices 0-1 / 2-3 (lo/hi byte per channel), 5 = pixel normal
  in the vertex tangent frame (alpha = two-sided flag). Scripts: scratch `probe_ids.py`.
- **Stairs (`ext_stairs_03`, 3.3x too dark, O/U):** its decl sets `use_layer0_colorconstant 1`,
  `layer0_colorconstant (0.279, 0.279, 0.296)`, mask mode 0 on vertex channel 3 (default);
  the vertex alpha is 255 on every stairs vertex and the exe confirms RGBA (arkdefault.md),
  so the shipped shader source would darken by 0.279. The game frame shows no darkening:
  brightness 3.3x ours (1/0.279 = 3.58) and blue/red 0.895 of ours (controls 1.00-1.02;
  the constant is 6 % bluish). Mechanism unknown; our shader keeps the source behaviour.
  Debug view 6 = interpolated vertex colour.
- **Ambient occlusion (implemented 2026-10-16).** Game (O, `sub_140305BE0` /
  `sub_140305620`): `r_hbao 1` (user setting, default 0) runs NVIDIA HBAO+
  (`GFSDK_SSAO_D3D11.win64.dll`) instead of the built-in DWAO (`r_SsaoType 1`). Parameters
  as passed: radius 0.7, bias 0.1, small scale 1/resScale, large scale 1.75 x resScale,
  power 1.75, foreground AO on (view depth 4), background off, FP16 depth, depth threshold
  off, sharpness 100, blur struct (enable field 0 under the HBAO+ 3.x layout we assume,
  radius 4, sharpness 8). Lighting uses `AO = min(SSAO, occlusionMap.r)` on the ambient term
  (`ComputeAmbientOcclusion`, `HQ_FetchAO` at layer-0 UVs) and IBL. Ours: main pass writes
  direct light and the ambient term (+ material AO in alpha) to separate targets; compute
  passes `cs_ao` (8 directions x 4 steps + 4-neighbour detail term, HBAO+-style
  `saturate(N.V - bias) * (1 - d^2/R^2)`, depth-reconstructed normals, the game's
  parameters), separable depth-aware blur, `cs_composite`. Not bit-exact (HBAO+ is closed);
  `VOID_SSAO=0` disables, debug 7 shows the AO. Effect at RX008 dump 8 (game / ours, shade):
  white stone 1.49 -> 2.10, floor 1.31 -> 1.37, trims 1.21 -> 1.49, planks 0.88 -> 1.07.
  Since the shaded baseline was already slightly dark, a positive term is missing (H: IBL
  diffuse/specular ambient, L3c). Using bump-mapped normals instead of depth normals gave
  far too much AO (white stone 6.7x); input normals appear disabled in the game's
  `GFSDK_SSAO_InputData` (H).
- **Black patches (images; user: "most ... seem to be decals"), resolved 2026-10-16.**
  Lit = 0/negative pixels: 144,235 -> 6,829 of 2.76 M after:
  - **arkDecal mesh decals** (`decalType` 0; O, `arkdecal.decl`): layer-0 albedo with colour
    constant / vertex multiply / alpha test, HQ normal/gloss/metal, full lighting, blend
    SRC_ALPHA / ONE_MINUS_SRC_ALPHA, depth GEQUAL, `sv_position.z -= $zbias` (decls use
    -0.0001). Ours: separate PSO after the opaque draws, blending albedo, direct and ambient
    colour (alphas keep the surface's metallic / sun visibility / material AO), normals and
    material ids kept, no depth write, no shadow casting. Projected types 1-3 still
    unsupported (need per-instance world-to-decal transforms). stageSort order not modelled.
  - **Face culling** from the material `state { cull none | front }` (default back; 1,332
    shipped materials say `cull none`, 10 `cull front`), and arkVegetation's program-level
    `cull none` (O, arkvegetation.decl) unless the material sets one. Done in the PS after
    the mirrored-instance face correction. Depth agreement unchanged (88.3 %).
  - **Back-face geometric frame**: the one-sided unpack now also flips the vertex frame for
    back faces (common.decl comment: "I first swapped the geometric normal's direction in
    the pixel.inc"), fixing negative ambient on two-sided vegetation.
  - **Colour textures in raw slots**: `white_stone_01` layers 1/2 use
    `plaster_building_dirt_03_d` (BC7 sRGB) as bump and gloss map; we refused it and fell back
    to white (normal in the tangent plane -> negative unpack). The game binds the image's
    own view, so we now sample it as is (noted per material). White stone: sunlit 1.00,
    shaded 0.90 (with AO).
  - Materials without `mainProgram` (`models/templates/colors/*`) are drawn as arkDefault (H).
- **Material assignment bugs (fixed 2026-10-16; user report: houses red instead of blue,
  fence blue).** (1) World: `.bwm` surfaces are shared by batches with different materials
  (paint variants); we took every surface's material from its last batch. Each draw now
  uses its batch's material. This also explains the "stairs colour constant" discrepancy
  (withdrawn): the stairs batches use `ext_stairs_01`, not `ext_stairs_03` (sunlit 0.98).
  (2) Entities: the model file's surface materials are placeholders for many props
  (`models/templates/colors/*`); the runtime surface list (F3 `surfaces.tsv`; live: model
  +8/+16 or the shared list at +7168, 128-byte surfaces, material at +0) carries the
  instance's materials, including `renderModelInfo.m_customMaterials` from the
  `.entities` file. Used when its count equals the model file's (5,366 entities at RX008;
  33 count mismatches keep the file materials).
- **RX010** (user save, eotw, 2026-10-16, new DLL): F3 `snap_1792183388_f1288`, F4 frame 1352,
  F8 dumps 0-3 (frames 1432-1435; `local/ref/RX010/`, dump 1 used; older captures folder
  contents were copied too). Depth within 1 %: 89.4 %. Over-drawn pixels (ours nearer)
  by material: vegetation edges (H: wind sway), `roots_lod2_green` / `bushes_01_lod2` world
  batches (LOD duplicates; the game's LOD/visibility selection from the unknown `.bwm`
  instance lists is not recovered), and unsupported particle meshes (screen gates,
  `templates/occlusion/*` = program `particle/add_default`). Unsupported programs are no
  longer drawn (listed in materials.tsv; `VOID_DRAW_UNSUPPORTED=1` restores magenta). The
  user-flagged house entrance matches in depth (243,843 of ~252k px); its pilasters differ
  in shading (ours lighter/bluer on bevels; H: missing IBL/specular ambient).
- **Visible gaps (images, gallery):** no sky, water, characters, hands, fog; drain/blood
  particle effects.

RX005 (user save, Edge of the World `eotw_p_highchaos`, F8 dumps frames 5505-5508,
2026-10-16): d2enhanced SSGI/SSR off, FSR 4.1 on at native 3440x1440 (jitter <= 0.5 px,
recorded per dump), no F3 snapshot. Files `local/ref/RX005/`. Camera from the inject log:
(-34.35, 192.31, 5.63), forward (0.560, -0.794, -0.236), fov (1.862, 1.025). Offline world-only
render at this camera: 1,049 materials (544/404/101), 2,888 textures, 10 s total; the
in-game F4 request (5,522 entities) had not finished when the game closed ~7 s later.
Visual: same structures, signage, stairs and grates; water (arkWater/arkOcean) not drawn,
so the harbour floor shows; magenta = vista/decal/other programs.

RX006 (same save, F3 snap f4705 + F4 f4890 + F8 dumps 5111-5114, camera equal within 5 mm;
`local/ref/RX006/`): in-game F4 first render 13.3 s (5,519 entities, 120 without model,
3,405 textures). Depth vs game (world + filtered entities): **76.2 % within 1 %**, game closer
7.3 %, ours closer 16.5 %. New per-pixel material ids (`material_id.u32` + `material_names.txt`)
attribute the mismatch:

- ours closer: `roots_lod2_green` 4.3 % of samples (77 % of its pixels wrong; also
  "layer 0 diffusemap unavailable"), `rock/cliff_middle_01` 3.9 % (99 % wrong),
  `eotw_vista_mid_left_*` 2.2 %, vista buildings/trees, `templates/colors/black` 0.4 %,
  `screen_gate_eotw_01_fade` 0.15 %. H: the game does not draw these world instances in
  this view (LOD / distance or visibility-set selection); we draw every `.bwm` instance.
- game closer: NPCs and first-person hands/weapon (skinned, not drawn yet; stairs and
  floors behind them), the rotating searchlight (dynamic), foliage edges.
- `.bwm` fields checked for these instances (`examples/bwm_fields.rs`): instance u16 is
  0xFFFF everywhere, instance u32 looks like an id/hash, batch flag/unknown split a surface's
  instances but do not separate drawn from undrawn; none is a LOD distance (O). Next: recover
  the static-world instance selection in the exe (`arkStaticWorldModel` draw/cull, Umbra tome
  queries, `lists` table with 70,929 entries) before more lighting validation on this map.

Static-world visibility (O unless marked):

- RX007 (live F4 with the new DLL, frame 1435; F8 1516-1519): 76.1 % within 1 %, same
  material attribution as RX006 (live and offline paths agree).
- `.bwm` load (`sub_140295FA0`; version < 0x10015 recomputes in `sub_140295510`): each
  instance either has -1 at record +24 and goes to an always-visible list, or is mapped to
  an Umbra object through the tome; the 70,929 `lists` are per-Umbra-object instance lists.
  Static-world visibility is therefore decided by Umbra tome queries
  (`maps/.../<map>/<map>.tome`, 77 MB for eotw_p_highchaos; proprietary format).
- Gates: the map has 129 Umbra-gate references (`arkUmbraGate` entities, model
  `templates/occlusion/occlusion_model.lwo` scaled, `cpntUmbraGate` with e.g.
  `m_minDistance = 70`, `m_minScreenPctToOcclude = 3000`). H (geometric, RX006/7): the hidden
  cliff instances (~200 m south of the camera) lie behind gates near (-26.0, 104.5, 17.6) /
  (-48.0, 103.2, 18.0), ~78 m from the camera (> 70 m), so a distance-closed gate would hide
  them and leave sky, as observed.
- **Withdrawn (RX008):** the gate hypothesis above. RX008 (user console experiments at the
  same spot, `local/ref/RX008/`): `umbra_skipOcclusion 1` (dumps 4-7) leaves only part of the
  static world drawn (tree, ground, building parts missing; water shows) -> tome-mapped
  instances are drawn only when the query returns them (O). `r_umbraGateVisibilityRatio 0`
  (dumps 8-11) changes nothing measurable. Per-material depth ratios then showed the cliff
  and vista ARE drawn by the game at our positions, only ~1.3 % farther: the mismatch was
  our projection, not visibility.
- **Far plane (O, IDB):** `idRenderView_Setup` calls `idRenderView_BuildJitteredProjection`
  with near = `r_znear` (x 0.25 if `renderView_t::cramZNear`, +164) and far = `r_zfar`
  (cvar `0x143937460`, "far Z clip plane distance, 0 = infinity", default 7000); the matrix
  (`sub_1401098E0`) yields reversed-Z `d = n f / ((f - n) z) - n / (f - n)`. Fits of game depth
  against ours: eotw f ~ 6516, consistent with 7000 within fit error; tutorial range too short
  for a stable fit. With f = 7000 (renderer now reads `r_znear`/`r_zfar` live per request):
  RX004 tutorial 97.7 -> **97.8 %**, RX007 eotw 76.4 -> **86.3 %** (ours closer 16.5 -> 6.5 %).
- Remaining eotw residual: NPCs/first-person/moving searchlight (game closer 7.2 %),
  `roots_lod2_green` (diffuse unavailable -> solid magenta instead of alpha-tested),
  water/decal programs; Umbra occlusion itself only removes hidden objects, so it is not
  needed for correct depth/colour of a single view (performance only).
- F3 snapshots now include `cvars.tsv` (all static cvars: value, int, float, default, flags,
  description; list head `0x1431C04B0`, next +120; layout corroborated by d2enhanced
  `console.cpp`).

Validation needs clean captures (d2enhanced effects off) and, where possible, the game's
own debug views to isolate terms (cvar inventory pending).

## Live mode and performance (E044)

Shift+F4 toggles live rendering: every game frame the hook copies a `FrameRequest`
(game thread cost logged once a second) and the worker renders the newest one with
`RenderOptions { readback: false, scene_key }`. Nothing is shown yet: the game still
renders its own frame, so in-game numbers are game + ours on one GPU. Logged per
2 s: our fps, CPU prepare, submit+wait, GPU ms per pass (timestamp queries).

Offline reproduction (no game): `VOID_LIVE_FRAMES=200 cargo run --release -p void-inject
--example r3a_snapshot -- <ROOT> <snapshot> <out> <depth dump> <w> <h>` (logs to stderr).

RX010 snapshot, 2560x1080, RX 7900 XTX (GPU ms): shadows 2.0, opaque 7.0, AO +
composite 0.2, FFT 0.5-0.75, depth pyramid 0.05, ocean 0.7; total 10.7. Fixed on the
way: the foam warm-up (61 FFT steps) ran every frame; the pixel shader looped over
every light (~620) - now 32 px screen-tile lists (`light_tiles`, conservative box
projection, unbounded lights in every tile; output bit-identical).

In game (eotw_p_highchaos, 16829 draws / 99421 instances, batch built in 8.9 s): the
game fell from ~120 to ~18 fps because `frame_request` walked all models on the game
thread every frame (47 ms); live frames now copy entities only once. Our GPU time there
was 27-33 ms while sharing the GPU (shadows 7.4, opaque 17-24): no frustum culling,
every draw goes into each shadow cascade, LOD duplicates drawn.

Open: per-frame upload-heap allocations and the synchronous fence wait (~6 ms over the
GPU time), no depth pre-pass, entity placements frozen at the first live frame.

## Takeover (E045)

Ctrl+F4 cycles: **replace** (game skips its static world and ocean via `r_skipStatic`,
`r_skipOcean`; ours is depth-merged with what the game still draws: characters, hands,
particles, sky) -> **split screen** (left half ours, right half the game's own world) ->
off. Both turn live rendering on.

- Injection point: the game's TAA `Dispatch`, replayed through the immediate context on
  AMD (d2enhanced NOTES, "TAA shader internals"). Our pass writes `viewcolormap` just
  before it, so TAA, motion blur, bloom, tonemap, colour grading and the Iggy HUD run
  on our image. Recognition by bindings (t1 R16G16_FLOAT motion vectors, t2 scene colour,
  two UAVs), not shader hashes; the game depth is the t0 view (R24_UNORM_X8) of the
  depth-downsampling CS.
- State: `ID3D11DeviceContext1::SwapDeviceContextState` around our draw, so the game's
  bound state is untouched.
- Interop: shared textures created on the game's device (NT handles), opened by D3D12;
  `ready` / `consumed` shared fences created by D3D12. The composite only waits for
  values already signalled-queued (no GPU deadlock); the worker waits up to 50 ms for a
  slot to be consumed.
- Depth merge: distance of ours `a / (z - b)` vs the game's `0.01 / z` (reversed, infinite).
- Known limits: 1+ frame lag (async worker, synchronous fence wait), no TAA jitter in
  our projection, constant exposure scale x3.58, `r_skipStatic` coverage of the game's
  world not yet verified, no NVIDIA support (native command lists).

## Next

1. R1 scaffold: `dinput8.dll` proxy crate, logging, exe-hash gate, MinHook-based hook of
   `RenderAllViews` that only counts views/models/lights (no effect on the frame).
2. Decode remaining bitfield patterns; find the `drawSurf_t` / `idRenderWorldLocal` registration path.
3. Map the back-end path (draw-surface generation, deferred-context recording, submission)
   and the UI pass, to find where original 3D rendering can be skipped.

### Water (ocean), first pass (2026-10-16)

- **Recovered (O):** the harbour water is the ocean system (`arkOcean`), not mesh water. The map's
  world env (`m_worldEnvironment` in `.entities`: eotw `env/campaign/edge_of_the_world/eotw_p/
  global_ext01`, inheriting `default`) sets `ocean/enable 1`, Perlin/FFT/shore wave parameters,
  `ocean/waveamplitudemask` (BC4 512^2), `ocean/shoredistancefieldtex` (R16 512^2, bim format
  pair (2, 5) -> DXGI 56 via `sub_1402C2F50`), `water/skycubemap` (BC6H cube 256^2, 7 mips),
  water colour / extinction / inscattering. `sub_1403014B0` reads the ocean settings (handles
  `qword_1422ADxx0`, names 16 B before each); the ocean level is `ocean/WaterLevel` (default 0;
  measured game surface at z 0..6 with waves). Variants `ocean/{default,skycube,ssr,...}.material`
  select sky cube / SSR. Noise tables: `sub_1402F6450` builds `_perlinNoisePerm2D` (256^2 RGBA8)
  and `_perlinNoiseGradients` (256x1 RGBA8 SNORM) from Ken Perlin's permutation (exe 0x141AE8930
  / 0x141B1EB50) and the 16 gradients at 0x141B1EF50: GPU Gems 2 improved noise.
- **Ours:** `crate::ocean` (env chain + renderParm defaults + textures) and the `OCEAN_HLSL` pass
  after the AO composite: camera-centred polar grid (256 x 384 quads, cubic ring spacing to
  zfar), shipped Perlin + shore waves, `ComputeOceanSurfaceNormal`, `WaterShading` (refraction of
  a copy of the composited image with the opaque depth, extinction, in-scattering, sky cube
  reflection with parallax sphere, sun specular, soft edge). Not modelled: FFT waves (near field
  inside `PerlinBlendStart` 150 m is flat), SSR, caustics, projected shadow, foam depth,
  underwater, the game's `localTime` (time 0; `VOID_OCEAN_TIME`). `VOID_OCEAN=0` disables;
  `VOID_OCEAN_DEBUG` 2-6 shows normal / Fresnel / refraction / reflection / foam.
- **Bug found while bringing it up:** the grid's counter-clockwise winding made the top side a
  back face, which the shipped PS treats as underwater (Fresnel 1 by total internal
  reflection); fixed by clockwise winding.
- **RX010 dump 1 (game vs ours x 3.58, medians):** drain basin (1.33, 0.96, 0.75) vs (1.15, 1.39,
  1.20) (game tinted by blood FX); left basin (1.30, 1.44, 1.25) vs (1.23, 1.74, 1.71); harbour
  (4.9, 4.8, 3.6) vs (2.7, 6.8, 8.2) (game reflects buildings via SSR; ours sky).

### Water: FFT near waves (2026-10-16)

- **Recovered (O):** `arkGpuFFTWaves` follows NVIDIA's DX11 ocean sample. Setup `sub_1402FA110`:
  N = 512, three-slice FFT plan (`sub_1402F9590`), buffers 512^2 x 3 complex. Spectrum
  `sub_1402FA690` ("re-computing H0 and Omega"): `srand(0)`; per (y, x) in 0..=N two Gaussians
  (`sub_1402FA020`: Box-Muller cosine over MSVC `rand()`), Phillips spectrum with amplitude
  0.2 x 0.001, wind direction (normalized; (0.6, 0.8) if near zero), wind speed, dependency
  (applied when K.w < 0), small-wave length L / `ocean/FFTSmallWaves`, patch 50 m
  (K = (i - N/2) 2pi / 50); H0 = gauss x sqrt(P) / sqrt(2) as half2, Omega = sqrt(g |K|) as half,
  row stride N + 4. Inputs from `sub_1403014B0` (handles `ocean/FFTWindDirectionX/Y`,
  `FFTWindSpeed`, `FFTWindDependency`, `FFTSmallWaves`, `FFTChoppiness`, `FFTWaves/FoamType`).
  Per frame `sub_1402FB380` dispatches the shipped `fftwaves/updatespectrum` (InWidth N + 4,
  ActualDim N, OutSize N x N, DtAddressOffset (N^2, 2N^2)), the FFT, `updatedisplacement`,
  `gengradientsandjacobian`, `updatefoam/*` (shipped source).
- **Ours (`crate::fftwaves`, `FFT_*_HLSL`):** exact CPU H0/Omega (tests: MSVC rand sequence,
  engine half rounding), shipped kernels transcribed, our Stockham radix-2 inverse FFT (twiddle
  sign +1; H: the engine's FFT sign convention is unverified, a mirror at most), foam warm-up of
  2 s at 30 Hz before the frame (the game integrates over frame history). Wave heights at the
  drain: +-0.15 m (std 0.11 m), consistent with a numpy check of the same spectrum (0.22 m rms
  over the whole patch).
- **Bug while bringing it up:** root constants shifted by one slot from index 7 (InWidth read
  as 1); found with debug view 8 (raw FFT height).

### Water: SSR, caustics, game clock (2026-10-16)

- **SSR (O, shipped source):** `arkscreenspaceraytracing.decl` `ScreenSpaceRayTrace` (hierarchical
  march in 1/Z through a min/max depth pyramid, SSR_MIP_MAX 4, depth-scaled thickness, fade on
  rays towards the camera, vertical border fade) and `ComputeReflection` (normal z x
  `ScreenSpaceReflectionSoften`, `lerp(sky(Rs), hit, fade)`, hits beyond z 6599 dropped). eotw
  env: steps 128, vertical fade 0.5, soften 6.9546, thickness distance near/far 1.05/1.2 (as
  authored), mul far 1.0097; defaults mul near 1.02, pow 1, border fade 0.25. Ours: pyramid
  pass `cs_zmip` (avg/min/max linear Z, 4 levels), the trace transcribed with world-space
  inputs (view direction = clip row 3). Not modelled: fog on the hit (`atm/distanceFog`); the
  game's choice between the `ssr` and `ssr_ssrcull` ocean materials (quality setting; we use
  plain SSR, H). RX010: harbour (4.9, 4.7, 3.6) vs ours (7.8, 6.7, 4.5) - brighter by the
  missing hit fog (H); basins within ~10-20 %.
- **Caustics (O):** shipped `ComputeRefraction` caustics with `rendering/water/caustics3` (bim type
  0 with 32 faces = 2D array; loader `texture::load_array`), env speed 25 / tiling 0.4 /
  aberration 0.003 / fade 0.02; `water/wsRefractedLightDir` = refract(-L, (0,0,1), 1/1.333)
  (`sub_1403045A0`, constant normal at 0x14228A920 = (0, 0, 1)). The sun shadow at the
  underwater point is not wired (1).
- **Game clock (O):** `SetPerViewShaderParms` sets `localTime` from its view struct (= our
  renderView - 14800; FOV +14956 = +156, env block +17808 = +3008): flag byte +4 selects the
  double at +32 and delta at +60, else +16 / +52. Read live (`scene::read_local_time`) and
  saved by F3 as `time.tsv`; drives the ocean (Perlin, shore waves, FFT, caustics).
  `VOID_OCEAN_TIME` overrides.
