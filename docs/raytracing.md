# Ray tracing spike: DXR 1.1 inline ray tracing on the decoded eotw scene

Status 2026-10-18 (STATUS E070). Follows reflections.md section 3 (E069, the recommended
spike) and lighting-options.md option C. Offline only: the tool reads the game's archives
and two RX010 captures; it never starts the game or touches its files. Labels: **O**
observed (measured in the run below), **H** inferred, **U** unknown.

Code: crate `crates/void-rt` (library + examples `rt_spike` and `photo`). Results:
`local/rt/report.txt` (full text of the spike run), `local/rt/*.png`, `local/rt/runs.log`;
photo mode: `local/rt/photo/` (section 10).

## 1. Result

Measured on AMD Radeon RX 7900 XTX (device 0x744C, 24,516 MiB), driver 32.0.32015.2008,
Windows build 26200: **raytracing tier 1.1 (inline `RayQuery`), shader model 6.8** (O,
`D3D12_FEATURE_D3D12_OPTIONS5` / `D3D12_FEATURE_SHADER_MODEL`). Kernels are compiled as
`cs_6_5` with DXC from the Windows SDK (10.0.26100, loaded at run time).

| Gate (reflections.md section 3) | Measured | Result |
| --- | --- | --- |
| BVH memory < 0.5 GiB | 0.360 GiB (compacted BLAS 312.2 MiB + full TLAS 34.7 + its scratch 15.2 + instance descriptors 6.0 MiB); 0.480 GiB with the 121.4 MiB primitive buffer and the cut-off TLAS | **pass** |
| BLAS build < 2 s at map load | 12.2 ms GPU build + 1.2 ms compaction, 83 ms wall (prebuild queries, 4 submits, readback) | **pass** |
| Reflection rays < 1 ms at 2304x968, 1 ray/pixel, all pixels | 1.53 ms (mirror), 2.86 ms (GGX r = 0.3) | **fail** |
| same, floors only (N.z > 0.7, 27 % of pixels) | 0.96 ms | pass |
| same, cut-off BVH (40 m) | 1.53 ms | fail |
| same, hybrid: RT only where the screen-space reference missed | 1.66 ms (66 % of pixels traced) | fail |
| same, hybrid: RT only where the depth heuristic gives low SS confidence | 1.30 ms (54 % traced) | fail |
| same, hybrid SS-miss + cut-off BVH | 1.09 ms | fail |
| same, hybrid depth heuristic + cut-off BVH | 0.94 ms | pass |
| Agreement with SS where SS hits > 90 % (5 % distance tolerance, min 0.1 m) | from the game depth: 46.4 % (2304x968) / 52.3 % (3440x1440); floors only 57.8 / 63.9 %; only pixels whose primary ray matches the game depth: 51.6 / 57.7 %; method only (SS and RT both from our primary-ray depth): 74.2 % (floors 82.8 %) at both sizes | **fail** (section 5) |

All numbers are O from one run (`local/rt/runs.log` 1792281806.832..1792281834.372), GPU
time = median of 20 timestamped dispatches after 3 warm-up dispatches.

Conclusion: the BVH of the whole static map is small and cheap to build, so memory and
load time do not block RT. A full-screen reflection ray costs about 1.5 ms at the 0.67
scale. That is over the 1 ms budget, and divergent masks save little until the rays are
compacted (section 4). The agreement gate fails, mostly because our scene does not match
the game's geometry and because of limits in the screen-space reference, not because of
RT errors (section 5). The photo-mode idea is cheap enough to be practical: 17.1 ms per
path sample at 3440x1440, so 256 samples take 4.4 s of GPU time (section 6).

## 2. Scene and acceleration structures

**Scene (O):** `eotw_p_highchaos.bwm` (5,794 surfaces, 10,999 batches, 94,111 instances;
93,437 kept) plus the 5,452 static drawable model instances of the RX010 F3 snapshot
(`local/inject/snap_1792183388_f1288/models.tsv`, filtered as the old R3 renderer: no `.md6`,
`m_mayBeDrawn`, no editor/collision materials; 5,307 kept, 55 without a decodable model).
The `.entities` file is the alternative source (`--entities file`, decoders of E021/E023).
Kept: materials that arkDefault/arkVegetation can draw. Excluded (counted per reason in the
report): 82 `arkVista` instances (0.73 M triangles), 585 mesh decals, sky, clouds, water,
particle programs, 66 entity surfaces without material decl/diffuse map. Result: **6,706
BLASes (6,830 geometries), 98,744 instances, 5.30 M unique triangles, 51.7 M instanced
triangles**, 1,241 materials (1,240 with a texture albedo). 607 `.bwm` surfaces are drawn
with more than one material (paint variants, E040), and each variant gets its own BLAS
because shading data are stored per vertex (below).

**BLAS (O):** one BLAS per object (a `.bwm` surface, or an entity model with its kept
surfaces as geometries), `OPAQUE` geometry, u16 indices + float3 positions straight from
the decoders, `PREFER_FAST_TRACE | ALLOW_COMPACTION`, built in 4 batches through a 256 MiB
scratch arena (one submit each), then compacted: **826.4 MiB -> 312.2 MiB (37.8 %, 61.7
bytes/triangle)**. The build inputs (93.4 MiB) are freed after the build. DXGI process
usage delta: 409.5 MiB peak during build, 497.3 MiB after (all RT data + output buffers).

**TLAS (O, median of 20):**

| TLAS | Instances | Size | Rebuild (fast trace) | Rebuild (fast build) | Refit |
| --- | ---: | ---: | ---: | ---: | ---: |
| full map | 98,744 | 34.7 MiB (+15.2 MiB scratch) | 0.519 ms | 0.520 ms | 0.090 ms |
| cut-off: instance AABB within 40 m of the camera | 5,858 (3.13 M instanced triangles) | 2.1 MiB | 0.251 ms | 0.254 ms | 0.022 ms |

The driver reports 0 bytes of update scratch. Refitting with unchanged transforms is the
proxy for per-frame updates of moving entities: 0.09 ms for the whole map (H: patching only
the few moving instances with a refit is cheaper still).

**Shading data (O, Far Cry 6 layout):** a primitive buffer with 24 bytes per triangle
(3 vertices x [albedo RGB8 + smoothness A8, octahedral normal RG8 + metallic B8 +
reflectance A8]), 121.4 MiB, found through `PrimOffsets[InstanceID + GeometryIndex] +
PrimitiveIndex`. Albedo is the linear mean of layer 0's BC7 diffuse map (32 px mip, decoded
with `bcdec_rs`), smoothness = `lerp(wardRoughness.x, .y, mean gloss)`, metallic =
`lerp(metallicMinMax, mean metallic map)` (arkdefault.md), reflectance 0.04. Hit shading
needs no material table, no textures and no index or position fetch.

## 3. Rays

Kernels: `crates/void-rt/src/rt.hlsl`. Every kernel is a DXR 1.1 compute shader with inline
`RayQuery`, 8x4 thread groups, buffers bound as root descriptors (no descriptor heaps), and
geometry/primitive indices clamped in the shader.

**Camera (O):** the RX010 F3 snapshot view (`view.tsv`: org (-33.6576, 181.6801, 10.4555),
forward (0.552, -0.794, -0.254), fov_y 1.0248, `r_znear` 0.01). The log shows the same view
at frames 1190-1449, which contain F8 dump 1 (frame 1433). Projection is Hor+, and game
depth is reversed `znear / z`. Reference: `local/ref/RX010/dump1_depth_r32f_3440x1440.bin`
+ `dump1_color_rgba16f_3440x1440.bin`.
Without a snapshot the tool uses a synthetic camera (`--pose`, default benchmark scene 0
`-33.65,181.69,10.46,-49.25,13.72`).

| GPU ms (median of 20) | 2304x968 | 3440x1440 |
| --- | ---: | ---: |
| primary rays, no culling (closest hit + primitive-data shading) | 2.04 | 3.05 |
| primary rays + sun shadow ray | 1.95 | 4.09 |
| reflection, mirror, all pixels | 1.53 | 3.48 |
| reflection, mirror + sun shadow ray at the hit | 2.79 | 4.46 |
| reflection, GGX VNDF r = 0.3 | 2.86 | 4.89 |
| reflection, mirror, floors only | 0.96 | 2.07 |
| reflection, mirror, cut-off BVH (40 m, TMax 40 m) | 1.53 | 2.86 |
| hybrid: RT where the SS reference missed | 1.66 (66 % traced) | 2.72 (54 %) |
| hybrid: RT where the depth-heuristic SS confidence is low | 1.30 (54 %) | 2.30 (43 %) |
| hybrid SS-miss + cut-off | 1.09 | 1.96 |
| hybrid depth heuristic + cut-off | 0.94 | 1.62 |
| screen-space reference (dense linear trace, not a real-time SSR) | 2.53 | 6.10 |

Reflection rays start from the game depth: position from depth, normal from the depth
neighbours (the SSR's surface model, E067), origin offset along the normal by
`max(2 cm, 0.003 z)` and `TMin = max(5 cm, 0.005 z)`. Without that bias, 37 % of hits lay
within 5 cm (O, WARP run at 688x288), because our geometry and the game's surfaces differ
by centimetres. With the bias it is 0.0 %.

**Primary validation (O):** against the game depth, **73.6 % of pixels within 1 %**
(3440x1440; 73.5 % at 2304x968), game closer 5.9 % (first-person hands and sword, which are
not in our scene), ours closer 20.5 % (the cliff/vista backdrop at top left, opaque
alpha-tested foliage, `.bwm` LOD duplicates; same causes as E044/RX010's raster 89.4 %;
alpha test and LOD selection are the difference here), game sky / ours hit 0.01 %.
`primary_vs_game_*.png`: green match, red ours closer, blue game closer.
Culling: no culling 73.6 % > cull back 69.8 % > cull front 34.8 %. The engine's front faces
are counter-clockwise in DXR terms, so instances carry `FRONT_COUNTERCLOCKWISE` (flipped
for mirroring transforms) and two-sided materials `CULL_DISABLE`. The spike traces without
face culling.

**Masks are not free (O):** "RT where SS missed" traces 66 % of pixels and still costs
1.66 ms, more than all pixels (1.53 ms). Masked threads idle inside their wave, and the
remaining rays are the long, divergent ones. H: the saving needs ray compaction (FC6's ray
buffer / FidelityFX classifier tiles: a list of rays to trace, dispatched indirectly) and
half-resolution tracing (FC6: one ray per texel at half resolution with neighbour reuse).
Neither is measured here; quarter-rate tracing at 0.67 scale would put mirror rays near
0.4 ms (H).

## 4. AMD-optimised practice: what the spike applies

Sources: Brenham/Szlachtycz, "Performant Reflective Beauty: Hybrid Ray Traced Reflections
in Far Cry 6", GDC 2022 ([PDF](https://gpuopen.com/download/GDC_Performant_Reflective_Beauty_Hybrid_Ray_Traced_Reflections_In_Far_Cry_6.pdf));
[AMD RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/), ray tracing
section; [Improving raytracing performance with RRA](https://gpuopen.com/learn/improving-rt-perf-with-rra/);
FidelityFX Hybrid Stochastic Reflections sample ([GPUOpen](https://gpuopen.com/learn/hybrid-reflections/),
[source](https://github.com/GPUOpen-Effects/Hybrid-Stochastic-Reflections)), used as a technique
reference only (linked libraries stay on SDK 2.3).

| Practice | Source | Spike |
| --- | --- | --- |
| DXR 1.1 inline ray tracing in compute ("best performance ... in compute shaders, on a compute queue"); one active `RayQuery` at a time | RDNA guide | applied: all kernels are CS with `RayQuery`; one query object live at a time. Direct queue only (no async compute yet) |
| 8x4 thread groups for ray tracing | RDNA guide | applied |
| No per-material hit shaders: one lighting path with unified per-vertex parameters (albedo, smoothness, oct normal, metallic, reflectance) in a primitive buffer indexed by `PrimitiveIndex` | FC6 p. 23-27, 62 | applied: 24 B/triangle primitive buffer, primitive offset table per geometry |
| One BLAS per object (preferred over per material); static BLASes built once; only skinned ones rebuilt (FC6: max 12 skinned, 16 BLAS rebuilds per frame) | FC6 p. 22 | applied for static geometry (one per `.bwm` surface / entity model; material variants split, see section 2). No skinned geometry (no `.md6` decoder) |
| BLAS compaction; high-quality build flags; "smaller BVHs are always better" | RDNA guide, RRA article | applied: `PREFER_FAST_TRACE` + compaction (-62 %) |
| TLAS rebuilt per frame (FC6 / guide: async compute) or refit | RDNA guide, FC6 p. 66-69 | measured: rebuild 0.52 ms, refit 0.09 ms (full map); async queue not used yet |
| Opaque geometry, explicit ray flags; shadows stop at first hit | RDNA guide | applied: `OPAQUE` geometry + `FORCE_OPAQUE`; shadow rays `ACCEPT_FIRST_HIT_AND_END_SEARCH`. Alpha-tested foliage is opaque (5.75 M instanced triangles): wrong silhouettes, see section 5 |
| Limit the TLAS to a radius around the player; far reflections from SS/env map; confidence fades to 0 at the BVH edge | FC6 p. 16, 18, 71-72 (FC6: radius 100) | measured with 40 m: TLAS 5,858 instances. Cost at 2304x968: 1.53 -> 1.53 ms for all-pixel mirror rays, 1.30 -> 0.94 ms for the hybrid. The fade is not implemented |
| Lower-LOD BLASes for reflections | FC6 p. 71-72 | not applied: the `.bwm` LOD selection is not recovered (E044); all LODs are in the BVH |
| Avoid instance-AABB overlap and huge sparse instances (terrain) | RRA article | not addressed: 98,744 instances, overlap not analysed (RRA not run). Merging small static instances is a follow-up only if RRA shows overlap |
| Hybrid: SS trace first, HW rays only where SS confidence is low (large depth discontinuities, off-screen directions); per-tile hit/miss counters and random reconversion | FC6 p. 9-19, 74-77 | measured with two masks (SS reference miss; depth-discontinuity / off-screen heuristic). No ray compaction, so masked waves still cost (section 3) |
| SS trace: linear 64 steps + 8 refine, random start offset | FC6 p. 10-12 | the reference trace is dense (1 step per 2 px, up to 1,024) + 8 refine, with a random start offset |
| One ray per texel at half resolution, neighbour hit reuse with Hammersley GGX, temporal accumulation with reflection-motion validation | FC6 p. 6-8, 36-41 | not applied (full resolution, 1 ray/pixel, fixed seed); next step for rough reflections |
| Real-time GI on AMD | [Capsaicin framework (GI-1.0)](https://gpuopen.com/capsaicin/), GPUOpen research | not used; reference if real-time GI is pursued (section 6) |

## 5. Why the agreement gate fails

Where the screen-space reference reports a hit, an RT hit within 5 % of the same distance
was found for 46-52 % of pixels from the game depth and for 74 % (floors 83 %) when both
traces start from our own primary-ray depth (O). Breakdown at 3440x1440 from the game depth:
RT nearer 26.3 %, RT farther 20.2 %, RT miss 1.3 %.

- **Geometry mismatch** (game depth vs decoded scene): 26 % of pixels differ by more than 1 %
  at the primary ray. Restricting to pixels whose primary ray agrees within 1 % raises
  agreement only to 57.7 %, because the reflected ray also crosses mismatching geometry:
  opaque foliage cards, LOD duplicates, missing characters/hands/water (H).
- **Limits of the reference** (method-only 74 %): "RT farther" (16 %) are SS hits behind
  thin objects that the thickness test accepts (0.15 m + 2 % of depth). "RT nearer" (8.5 %)
  are occluders that SS misses: not visible on screen, or stepped over (H, from the
  `reflect_*_vs_ss_*.png` maps: red concentrates on foliage, railings and stair risers;
  floors are mostly green).
- The S2 SSR in void-inject (hierarchical, validated, E067/E071) was not the reference: it
  runs in D3D11 inside the game. Its real-time hits are probably closer to RT on large
  planes (H).

What the gate needs before it can be meaningful: alpha-tested foliage in the BVH
(any-hit-style candidate loop or opacity micromaps, plus alpha masks in the primitive
data), the `.bwm` LOD/visibility selection, and a comparison against the in-game SSR output
(or the game's own reflections) instead of this reference.

## 6. Photo mode (user idea: much more tracing for screenshots)

The `gi` demo below is the spike measurement; it is superseded by the photo-mode renderer
(section 10: textures, alpha test, engine lights, MIS, denoiser, validation against the
game). Rows of the gap table below that section 10 closes: sun/sky intensity (engine units,
sky scale fitted), local lights (RIS over a light grid), materials (arkDefault layers,
normal maps), alpha-tested foliage (candidate loop).

**Offline demo (O):** `gi` kernel, 3440x1440, one path sample per pixel per frame:
primary ray (sub-pixel jitter), 2 diffuse bounces (cosine sampling, albedo and metallic
from the primitive data), sun next-event estimation with a 0.5 degree cone (soft shadows),
and a gradient sky. RX010 sun direction (theta 26, phi 100). Relative units; exposure is
set automatically.

| Samples | Relative RMSE vs 256 spp | Image |
| ---: | ---: | --- |
| 1 | 0.228 | `local/rt/gi_0001spp.png` |
| 4 | 0.114 | `gi_0004spp.png` |
| 16 | 0.055 | `gi_0016spp.png` |
| 64 | 0.025 | `gi_0064spp.png` |
| 256 | (reference) | `gi_0256spp.png` |

GPU per sample: median 17.10 ms (16.37-25.30); 256 samples = 4.41 s GPU, 4.49 s wall,
8 samples per submit. Error halves per 4x samples, as expected for Monte Carlo
(1/sqrt(n)). 64 samples (about 1.1 s) are visually converged for diffuse GI.

**Design for in-game use (H unless noted).**

1. Trigger: a photo hotkey in the overlay. Hotkey handling exists in void-inject (overlay,
   F-keys). Pause detection: the engine pause/menu state (U: flag not identified), or a
   user-held "freeze" plus a camera-unchanged check. The view matrix is already read per
   frame (SSR: view projection +29584, view +29776). Accumulation restarts whenever the
   view changes.
2. Scene: the static BVH (this crate) is built once per map load (E033 map detection) on a
   D3D12 device in our DLL. Moving entities: refit the TLAS from the live render-model list
   once per photo, not per frame (the full walk costs 47 ms, E044; acceptable once per
   photo).
3. Per frame while paused: N samples per pixel (budget-driven, e.g. 4 x 17 ms on this GPU)
   into an RGBA32F accumulator, at most 8 dispatches per submit (TDR safety). Output: either
   (a) replace: full path-traced image, tonemapped with the engine's exposure, or
   (b) composite: path-traced indirect light / reflections added to the engine's frame
   (needs the material target of E072 for per-pixel albedo/roughness, and splitting the
   engine's direct light from its indirect light, U).
4. Hand-off to D3D11: a shared texture created by D3D11 (E047/E048 rules) and the FSR
   hand-off path (copy + signal, E058/E060), drawn over the back buffer at present time
   while the photo mode is active.
5. Optional: FidelityFX radiance cache 0.9.0 (available here, fidelityfx-effects.md Q7)
   trained by this path tracer to end paths early. Only worth it for real-time GI, not for
   a paused accumulation.

**What the game must provide, and the gaps:**

| Input | Source today | Gap |
| --- | --- | --- |
| camera (view, projection, jitter) | per-frame view data (SSR/FSR, O) | none |
| pause / frozen camera | U | pause flag not identified; fallback: camera-unchanged + hotkey |
| sun direction and colour | F3 snapshot `sun.tsv` (theta/phi/colour, O); engine scale of `sunColor` -> radiance U | absolute sun/sky intensity relative to the engine's exposure (E037) |
| sky / environment | env decl + probe SH (E036/E037) | sky radiance model (the spike uses a constant gradient) |
| local lights | live light list (E035 struct; `lights.tsv` 621 lights at RX010) | no light sampling in the kernel yet (621 lights need a light BVH or reservoir sampling) |
| materials | decoded arkDefault (layer 0 averages in the primitive buffer) | per-texel albedo/normal maps (bindless BC textures), layer blending, emissive, translucency |
| alpha-tested foliage | material flag known | opacity in the BVH (candidate loop or OMM) |
| dynamic objects | live model transforms (snapshot path, O) | skinned characters (`.md6` undecoded), physics props sync, particles, water surface |
| first-person hands/weapon | game frame only | composite them from the game frame (depth test) |
| fog / volumetrics | engine pass | apply the engine's fog after compositing (U) |

## 7. Missing for an in-game real-time integration

- A D3D12 device inside the DLL, created once per map, with the BVH; the D3D11 frame
  supplies depth (+ the E072 material target) via the existing FSR hand-off (0.2-0.5 ms CPU
  on the submit path, E058/E060), and the result goes back through a shared texture.
- The SS-first hybrid as in FC6/FidelityFX: the existing S2 SSR provides hits and
  confidence; RT only for low-confidence pixels via a compacted ray list (indirect
  dispatch), at half resolution with neighbour reuse, plus a temporal accumulator with
  reflection-motion validation (no FFX denoiser on RDNA3, fidelityfx-effects.md).
- Hit lighting comparable to the engine (sun + shadow, probes/IBL, local lights, fog),
  or rough reflections only, where differences are less visible.
- BVH completeness: alpha-tested foliage, LOD selection, water, characters (`.md6`),
  moving entities (TLAS refit from the engine's render-entity updates, function U).
- Async compute for the TLAS refit and the ray dispatches.

## 8. Safety and validation of the spike itself

The first full-size GPU run (before any logging) removed the D3D12 device in the
screen-space reference kernel: depth reads at unclamped pixel coordinates went past a
root-descriptor buffer, and the user saw an AMD driver reset. Since then (O):

- every per-pixel buffer is checked on the CPU against `width x height` before each
  dispatch (`gpu::Programs::dispatch`); shader-side clamps cover pixel, geometry and
  primitive indices; root descriptors have no GPU bounds checks;
- at most 8 dispatches per submit; the BLAS build submits once per scratch batch; GI
  submits every 8 samples;
- `--smoke` runs every kernel on **WARP** (Microsoft's software D3D12 device; it supports
  DXR 1.1 inline ray tracing, O) at 64x27/96x40 with the 2,000 nearest instances. WARP runs
  on the CPU and cannot reset the GPU driver;
- `local/rt/runs.log` records the start and end of every run (the first, unlogged crash is
  noted at the top);
- `--debug-layer` (debug layer + GPU-based validation) is implemented through an Agility SDK
  device factory (`--agility <dir>`), but it does not work on this machine (O): the Windows
  "Graphics Tools" feature is not installed (no `d3d12SDKLayers.dll`). Retail Agility
  1.619.6 through `ID3D12SDKConfiguration1::CreateDeviceFactory` fails at
  `GetConfigurationInterface(CLSID_D3D12Debug)` (0x887A002D); preview 1.721 needs Developer
  Mode (0x887E0003). Installing Graphics Tools enables it.

## 9. Reproduction

From `void-rs/` (game root one level up). Run the validation first, then the GPU run:

```
cargo run --release -p void-rt --example rt_spike -- --smoke --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --ref-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin
cargo run --release -p void-rt --example rt_spike -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --ref-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
  --ref-color local/ref/RX010/dump1_color_rgba16f_3440x1440.bin
```

Options: `--entities file|snapshot|none`, `--pose x,y,z,yaw,pitch` (no snapshot),
`--cutoff 40`, `--reps 20`, `--gi-frames 256`, `--gi-size 3440x1440`, `--sizes WxH,WxH`,
`--warp`, `--max-instances N`, `--dxc <dir>` (default: env `VOID_DXC_DIR`, then the newest
Windows SDK `bin/<ver>/x64`). Outputs: `report.txt`, `primary_{shaded,distance,instance,
vs_game}_<size>.png`, `reflect_<variant>_{shaded,distance,vs_ss}_<size>.png`,
`gi_<spp>spp.png`, `game_color.png`. Inputs are read-only (archives under `base/`, RX010
snapshot and dumps under `local/`).

## 10. Photo mode (offline renderer)

Status 2026-10-18 (STATUS E073). Example `photo` (`crates/void-rt/examples/photo.rs`),
library modules `photo.rs` (host), `photo.hlsl` (kernels), `textures.rs` (bindless
upload), `lights.rs` (engine lights, sun, sky, light grid), photo layout in `scene.rs`.
Outputs: `local/rt/photo/` (`report.txt`, `photo_<spp>spp[_denoised].png`,
`photo_vs_game.png`, `photo_ratio_game_over_ours.png`, HDR `photo_[denoised_]hdr_rgba32f.bin`
with `.tsv` headers); first full run with sky scale 1 in `run1_sky1/`, sky-scale sweep at
860x360 in `sky_{0,0.067,0.25}/`, WARP validation in `smoke/`.

### 10.1 What is implemented

- **Scene data (O):** same BLAS/TLAS as the spike plus 585 decal instances: 7,263 BLAS,
  99,329 instances, 5.38 M unique / 51.8 M instanced triangles. Per triangle a 64-byte
  record (texture LOD bias + the three raw 20-byte vertex attribute blocks: uv0/uv1 f16,
  snorm8 normal and tangent, RGBA8 vertex colour), per geometry (first primitive,
  material), 328 MiB. 1,319 materials in the removed R3 renderer's `Mat` layout
  (`pack_params` + texture slots + `pack_lighting`).
- **Textures (O):** 4,002 maps capped at 512 texels (BC4 1,870, BC5 SNORM 909, BC7 sRGB
  1,223; 0 rejected), one shader-visible SRV table, sRGB decoded by the SRV format. Stored
  chains end at one block row/column (4x4, 4x8); they are completed to 1x1 with the mean
  colour. Single-level `$nomips` images stay as stored (10.6). Index clamped to the table
  size; divergent indices go through an explicit waterfall loop (see 10.4).
- **Materials:** arkDefault port of checkpoint 9282529 `ps`: 3 layers (144 materials use
  more than layer 0), masks from vertex colour / mask maps, colour constants, reuse,
  vertex-colour multiply, gloss/metallic ranges and maps, BC5 normal maps with the vertex
  tangent frame; engine mapping tint = lerp(specularTintDielectric, albedo, metal^2),
  diffuse = lerp(albedo, diffuseAlbedoMetallic, metal^2). BSDF: Lambert x (1 - F) + GGX
  (Smith, Schlick) with alpha = (1 - gloss)^2 (H: the engine uses Ward + Oren-Nayar).
  Ignored (counted per material in the report): emissive maps (63 materials), translucency
  (10 + 32 maps), sheen, detail normal maps, wind, refraction, material AO maps (loaded, not
  applied: the path tracer occludes).
- **Texture LOD:** ray cones (per-triangle Delta from UV vs object area, instance scale,
  cone width / |cos|); after a bounce the cone widens (0.5 rad diffuse, 2 alpha specular,
  simplified, H).
- **Alpha:** alpha-tested geometry (5.75 M instanced triangles) and decals are
  `NON_OPAQUE`; the inline query's candidate loop commits alpha-tested triangles at layer-0
  alpha >= 0.5 (engine `clip`), decals stochastically with probability alpha (= alpha
  blending in expectation, lifted 4 mm along the normal, H). Shadow rays: alpha test yes,
  decals never occlude. Depth agreement with the game rose from 73.6 % (spike, opaque
  foliage) to 77.6 %.
- **Lights:** `light_parms.tsv` of the F3 snapshot: 524 of 621 (88 point, 44 half-point,
  397 spot; 91 area lights skipped, 1 parallel = sun), attenuation and cones as the R3
  renderer (`lerp(1, 1/d, decay)^2 smoothstep(cutoff, falloff, d)`, H for local lights).
  XY grid 45x54 cells of 4 m (8,241 entries, max 36 per cell); one light per shading point
  by RIS (8 candidates, target = unshadowed luminance x cos) + one shadow ray.
- **Sun and sky:** sun irradiance = `sunColor` x 574.15 = (123.9, 98.4, 67.2) (O, R3
  scale), 0.27 degree disc, light-sampled with MIS against the BSDF (sun reflections in
  glossy surfaces). Sky = lerp(skyZenith, skyHorizon) over the upper hemisphere (`env.tsv`:
  0 / (37.7, 52.4, 55)) x `--sky-scale`, uniform-hemisphere light sampling with MIS. Default
  scale 0.067 = fitted so the sky seen by both matches the game (see 10.3); emissive surfaces
  are missing.
- **Paths:** 4 bounces, Russian roulette from bounce 2, luminance clamp 100 (engine units)
  on every contribution after the primary hit.
- **Denoiser (option):** albedo-demodulated edge-aware a-trous (5 passes, normal / depth /
  luminance stopping, variance of the mean, 3x3 spatial variance below 16 spp). No FFX.
- **Output:** engine absolute units; HDR raw RGBA32F + `.tsv`; PNGs show ours x 3.58 (the
  daylight dump scale, docs/renderer-replacement.md) with one exposure (median luminance of
  the game frame -> 0.18), per-channel Reinhard, sRGB. The engine's tone curve
  (`env/tonemapping/*`) is not recovered, so the PNGs are not the game's look.

### 10.2 Measured (O, RX 7900 XTX, 3440x1440, RX010 pose)

| Item | Value |
| --- | --- |
| GPU per sample (1,024 samples) | median 34.68 ms (31.65-59.22) |
| 1,024 spp | 35.84 s GPU, 35.89 s wall sampling, 133 submits, max 337 ms GPU per submit (adaptive chunk <= 400 ms, <= 8 samples) |
| whole run | 62.5 s wall (load 3.1 s, upload 4.7 s, readback/denoise/PNG 14.9 s) |
| VRAM | textures 890.4 MiB, BLAS 316.9, TLAS + scratch + descs 56.3, tables 329.1, accumulation/denoise 377.9; DXGI process usage 2,297 MiB |
| denoiser | 8-19 ms GPU (5 passes) |

Convergence (relative luminance RMSE against 1,024 spp):

| spp | raw | denoised |
| ---: | ---: | ---: |
| 1 | 0.279 | 0.188 |
| 4 | 0.138 | 0.117 |
| 16 | 0.069 | 0.062 |
| 64 | 0.034 | 0.037 |
| 256 | 0.015 | 0.021 |
| 512 | 0.009 | 0.014 |

The denoiser helps up to 16 spp; from 64 spp its bias (albedo demodulation at edges,
blur of smooth shading) exceeds the noise, so converged photos use the raw image. 64-256
spp (2-9 s) are visually converged; the sky-scale-1 run (more sky light, more noise) had
raw 0.59 / denoised 0.23 at 1 spp.

### 10.3 Validation against the game (RX010, dump1 pre-TAA HDR colour)

Game stored colour vs ours x 3.58, pixels whose depth matches within 1 % (77.6 %), regions
by our first-hit normal and sun visibility; game/ours luminance, median (ratio of sums):

| Region (share of pixels) | sky x 1 (512 spp) | sky x 0.067 (1,024 spp, default) |
| --- | ---: | ---: |
| all depth-matched (77.6 %) | 0.541 (0.558) | 0.943 (0.963) |
| sunlit floor, N.z > 0.7 (22.7 %) | 0.595 (0.598) | 0.869 (0.873) |
| sunlit wall, abs(N.z) < 0.3 (25.1 %) | 0.466 (0.471) | 0.887 (0.907) |
| shaded wall (15.8 %) | 0.427 (0.449) | 2.393 (2.809) |
| shaded floor (1.9 %) | 1.216 (2.029) | 9.442 (14.11) |
| sky seen by both (647 px) | 0.066 (0.067) | 0.988 (1.003) |

Sweep at 860x360, 64 spp (game resampled): sky x 0 gives sunlit floor 0.907, sunlit wall
0.942, shaded wall 3.36; sky x 0.25 gives sunlit floor 0.80, sunlit wall 0.73, shaded wall
1.11, sky 0.26. Game sky pixels: mean 3.06 absolute vs our horizon 3.32 at x 0.067.

Differences and causes:

- **Overall level and sunlit surfaces** match within 4 % / 9-13 % with no exposure fit
  (only the documented 3.58 dump scale). Ours is 7-10 % brighter in sun even without sky:
  path-traced bounce light from the bright ground, which the game's baked probes give less
  (H).
- **Sky:** `skyZenith`/`skyHorizon` are probe bounce factors, not sky radiance: used as
  radiance the sky is 15x the game's visible sky and doubles the light on sunlit floors.
  Scaled to match the visible sky (0.067), shaded walls are 2.4x darker than in the game:
  the game's ambient (probe SH x sky ZH with these factors, `ambient` 1) is brighter than a
  sky consistent with its dome (H). No single scale fits both; `--sky-scale 0.25` matches
  shaded walls at the cost of sunlit surfaces (0.73-0.80).
- **Shaded floor (9.4)** and the red areas of `photo_ratio_game_over_ours.png`: harbour
  water (the game draws water and its reflections; we trace the stones under it), foliage
  (the game lights leaves two-sided with translucency; ours are opaque alpha-tested cards),
  sides in shadow (ambient, above). Blue areas (ours brighter): metal shutters and some
  floors (GGX + path-traced reflections vs the engine's Ward + IBL).
- **Missing in ours:** water, sky dome/clouds, 82 `arkVista` backdrop instances, skinned
  characters and the first-person hands (excluded from the statistics by the depth test),
  particles, emissive surfaces, area lights, fog/volumetrics, the engine's tone curve.

### 10.4 Safety and validation

Every kernel ran on WARP first (`--smoke`: 96x40, 4 spp, 64-texel textures, 2,000 nearest
instances). WARP removed its device (0x887A0005, no GPU involvement) in two situations,
all fixed before any GPU run: (1) `NonUniformResourceIndex` texture sampling at LOD > 0
with the real per-material slots (slot 0 only, one uniform slot or LOD 0 worked; not
isolated further) -> explicit waterfall over `WaveReadLaneFirst`; (2)
`CommitNonOpaqueTriangleHit` in a query with `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`
(shadow rays) -> the candidate is not committed, the query is aborted. Both were bisected
with the `--disable sun,sky,lights,textures,alpha` switches. GPU runs (all
logged in `local/rt/runs.log`, no device removal): one 860x360 sizing run, two full-size
runs (512 and 1,024 spp), three 860x360 sky-scale runs. Per dispatch the CPU checks the
layout and per-pixel buffer sizes (`gpu::Programs::dispatch`); every table count the
kernel clamps against is checked against its buffer (`SceneGpu::validate`).

Reproduction (from `void-rs/`):

```
cargo run --release -p void-rt --example photo -- --smoke --game .. \
  --snapshot local/inject/snap_1792183388_f1288
cargo run --release -p void-rt --example photo -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --ref-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
  --ref-color local/ref/RX010/dump1_color_rgba16f_3440x1440.bin --spp 1024 \
  --save-at 1,4,16,64,256,512
```

Options: `--size`, `--spp`, `--bounces 4`, `--rr-start 2`, `--clamp 100`, `--ris 8`,
`--tex-size 512`, `--no-decals`, `--sun-radius 0.27`, `--sky-scale 0.067`,
`--light-cell 4`, `--denoise-iters 5`, `--disable ...`, `--dump-scale 3.58`,
`--submit-ms 400`, `--tile-pixels 0` (whole image per dispatch), `--warp`,
`--max-instances`, `--aov` (primary-hit albedo + material id, uv/LOD/depth; 10.6). The
example drives the in-process API of 10.5 (no duplicate renderer) and needs no DXC at run
time.

### 10.5 In-process API and precompiled shaders (`void_rt::ingame`)

Status 2026-10-18 (O, this machine). The module doc of `crates/void-rt/src/ingame.rs`
lists every signature, the root layout table for other kernels, and the game fields the
host fills. Summary:

- **Device:** `Gpu::from_existing(ID3D12Device5, ID3D12CommandQueue)` wraps the host's
  device and queue (no debug layer, no DXGI enumeration, `vram_usage` 0). `Gpu` is `Send`.
  Besides blocking submits it has one optional non-blocking submit in flight
  (`submit_async`, `poll`, `completed(fence)`) on two allocators. Owners of buffers keep
  a `SubmitGuard`, so dropping a session or a `Gpu` waits for its GPU work. `clear`
  copies from a zero buffer uploaded at creation. WARP returned non-zero memory for a
  fresh committed buffer, so implicit zero-initialisation is not relied on.
- **Scene:** `load_scene(&SceneSource, &SceneOptions) -> CpuScene` decodes on any thread
  without GPU objects (`Send`). `upload(&mut Gpu, CpuScene) -> GpuScene` builds BLASes
  (one submit per scratch batch), the full TLAS, the tables and textures (one submit per
  `upload_batch_bytes` of staging data, through a reused upload buffer) and the photo
  kernels. `GpuScene` is `Send + Sync` and is shared as `Arc`. For real-time ray tracing
  it exposes per-mesh BLAS addresses, instances with world AABBs, `instance_descs`,
  `build_cutoff_tlas` / `update_cutoff_tlas` (`Tlas::build(capacity)` / `rebuild` /
  `refit`, one blocking submit each), the geometry table, records, materials, lights, the
  light grid, the bindless texture heap, optionally the spike's FC6 primitive buffer
  (`SceneOptions::spike_primitives`), and a `generation` counter.
- **Session:** `PhotoSession::new(&mut Gpu, Arc<GpuScene>, Camera, w, h, PhotoSettings)`.
  `step(&mut Gpu, budget_ms)` never blocks: it returns `busy` while its previous submit
  runs, then records up to 8 row tiles (default 1/8 of 3440x1440 = 180 rows) that fit
  the budget by the last measured cost of each tile and submits them without waiting.
  `reset_camera` zeroes the accumulation. `copy_out` resolves the mean and copies HDR
  (RGBA32F) and depth (R32F) into host textures (256-byte pitched footprints, COMMON in
  and out) in one non-blocking submit. `read_hdr`, `read_depth`, `denoise`, `write_png`
  and `write_hdr` are blocking helpers. New kernels: `photo_depth` traces the pixel-centre
  primary hit (game depth `znear / z`, 0 = sky, alpha test), and `photo_resolve` writes
  the mean (rows with one more sample divided separately). `photo_trace` takes the
  tile's first row in a root constant; `Programs::dispatch_rows` checks the band and the
  whole-image buffer sizes.
- **Units:** the HDR buffer is engine absolute radiance (section 10.3). The engine's
  stored `hdrBuffer` = absolute / `EngineLuminanceFactor` (`stored_hdr_scale(ELF)`;
  x 3.58 = `DAYLIGHT_STORED_SCALE` in bright daylight, docs/renderer-replacement.md).
- **Precompiled DXIL:** `build.rs` runs `dxc.exe` (env `VOID_DXC_DIR`, else the newest
  Windows SDK `bin/<ver>/x64` that has `dxc.exe` and `dxil.dll`; otherwise the build
  stops with a message) on every `[numthreads]` kernel of `photo.hlsl` and `rt.hlsl`
  (`-T cs_6_5 -HV 2021 -O3`, compiled in parallel). The library embeds the signed DXIL
  (`ingame::dxil(entry)`), so the game machine needs no `dxcompiler.dll`. Run-time DXC
  stays in use for `rt_spike`.

Check (`photo_ingame_check`, logged in `local/rt/runs.log`). It creates the device the
way the host does, loads the scene on a worker thread, and exercises cut-off TLAS
build / rebuild / refit, reset, non-blocking stepping, `copy_out`, the readbacks and
saving:

| Run | Result |
| --- | --- |
| WARP 96x40, 4 spp, 10 tiles of 4 rows | PASS; vs the previous `photo` output 3,730 of 3,840 pixels bit-identical, rel. luminance RMSE 3.1e-4 (two runs of the new code differ in 7 pixels: WARP is not deterministic) |
| RX 7900 XTX 860x360, 64 spp, 9 tiles of 44 rows, 8 ms budget | PASS; **bit-identical** to the previous `photo` output (309,600 of 309,600 pixels); 146 submits, max 8.8 ms GPU per submit; 14.7 ms per sample; `copy_out` textures equal the readbacks; centre depth vs accumulated first-hit depth: median ratio 1.00000 |
| RX 7900 XTX 3440x1440, 8 spp, 8 tiles of 180 rows | 2.9-6.1 ms per tile (top tiles cost most); 39-48 ms per sample (offline full-frame dispatches: 34.7 ms, 10.2; per-tile barriers and idle GPU between short submits, H); depth pass 14 ms; max 11.5 ms GPU per submit |

```
cargo run --release -p void-rt --example photo_ingame_check -- --smoke --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --reference local/rt/ingame/baseline_smoke/photo_hdr_rgba32f.bin
cargo run --release -p void-rt --example photo_ingame_check -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288 --size 860x360 --spp 64 \
  --reference local/rt/ingame/baseline_gpu/photo_hdr_rgba32f.bin --measure-size 3440x1440
```

The references are `photo` outputs (same camera, size and spp), regenerated after the
`$nomips` texture fix of 10.6; both checks then PASS bit-identical (3,840 of 3,840 and
309,600 of 309,600 pixels).

### 10.6 Material and texture quality (2026-10-18, RX010)

User report after the first in-game run: "missing a lot of effects and has some broken
textures". Design since then (user): photo mode keeps the game's paused frame as the base
and only adds ray-traced reflections, AO and sun shadows, so our scene data matters as the
material of reflected hits and as occluding geometry/alpha. Outputs in `local/rt/quality/`.

**Method (O).** `photo --aov` (kernel `photo_aov`: pixel-centre primary hit with alpha
test, the first-hit material evaluation of `photo_trace`) at 2560x1080 on
`snap_1792183388_f1288`, the pose of the R3 G-buffer `r3a_1792183406_f1352` (6 mm apart;
its albedo was validated against the game, E034/E039/E040). Compared per pixel where the
depths agree within 1 % and both name the same material (`compare_r3a.py`, per-material
table `albedo_vs_r3a_per_material.tsv`). Runs in `local/rt/runs.log` (WARP smoke first).

| RX010, 2560x1080 | before | after the fix |
| --- | ---: | ---: |
| depth-matched pixels | 87.5 % | 87.6 % |
| same material among them | 97.74 % | 97.94 % |
| mean abs. sRGB albedo error | 0.0108 | 0.0104 |
| pixels with error > 0.1 | 0.72 % | 0.51 % |
| `roots_lod2_green` error (20,687 px) | 0.057 (mean 0.114/0.120/0.054, r3a 0.176/0.187/0.086) | 0.012 (0.178/0.188/0.086) |
| `roots_lod2_green` coverage (r3a 39,323 px) | 23,925 px | 38,972 px |

Findings:

- **Fixed: `$nomips` images (O).** Four maps here (the `roots_lod2` set, one 1024x1024
  level each) had levels 1.. filled with the mean colour by the chain completion, so any
  LOD > 0 showed a flat colour and alpha-tested cards got the mean alpha: distant ivy and
  roots vanished (`crop_roots_nomips_before_after_r3a.png`). The engine samples their only
  level; `textures::complete_chain` now completes only chains that end at one block.
- **Decoders verified (O).** Example `mip_check` compares every BC4/BC5/BC7 level with the
  2x2 box average of the level above (4x4 block means) for all 4,002 maps at <= 2048
  texels, streamed `_mipN` parts included: mean difference 0.0013-0.0022 at every level,
  worst pair 0.047 (alpha of small decal mips), none above 0.08. No wrong part, block
  order, format, sRGB or channel. Per material, the arkDefault evaluation (layers, masks,
  colour constants, vertex colour, UV sets) agrees with r3a: 152 materials with >= 200
  pixels, all below 0.085 mean error; the top four (`wrench_docker` 0.082, `seaweed_01`
  0.071, `poster_01` 0.056, `shop_light_wall_01_lod2` 0.053) are small distant props whose
  difference is filtering (crops `crop_<material>_ours_vs_r3a.png`).
- **Textures 512 vs 2048 (O):** 99.99 % of the pixels are identical. The ray-cone LOD at
  this view (log2 UV footprint percentiles 1/50/99 %: -8.4/-6.7/-3.1) never selects the top
  two levels of a 2048 map, while the 2048 cap needs 6,164 MiB of textures (890 MiB at
  512; process 7.4 GiB) and 15 s of decoding.
- **Blur, not fixed (O measured, H cause):** our albedo has 0.62x the high-frequency
  (Laplacian) energy of r3a at every distance (`crop_floor_lod_ours_vs_r3a.png`): the cone
  footprint is isotropic and stretched by 1/cos with trilinear `SampleLevel`, r3a (and the
  game, H) sample anisotropically (x8). Feasible fix: camera-ray LOD without the 1/cos
  stretch (jittered samples integrate the anisotropic footprint). Not applied: primary
  albedo is no longer composited, reflected hits use widened cones.
- **Material mismatches (2.1 % of matched pixels):** decals (ours commits a decal with
  probability alpha, r3a blends; means agree, e.g. `dcl_puddle_01` 0.41/0.38/0.32 vs
  0.41/0.37/0.32), `screen_gate_eotw_03` behind its additive `_fade` card, vegetation
  edges. `models/templates/occlusion/black_noshadow_unlit_grey` (9,167 r3a pixels) is a
  `particle/add_default` card, not traced by either (r3a drew it magenta).

Missing compared with the game. With the game frame as the base none of these is needed
in the photo itself; the last column is what they mean for the added terms:

| Effect | In our scene | Effect on reflections / AO / shadows |
| --- | --- | --- |
| water (`arkWater`/`arkOcean`) | not traced | water is neither reflector nor occluder; rays reach the bed (H) |
| sky dome, clouds (`arksky`/`arkclouds`) | analytic sky (lerp zenith/horizon x 0.067) | reflection misses show the sky model, not the engine sky (BC6H cube, E041, could be sampled) |
| vistas (`arkVista`, 82 instances) | not traced | distant backdrops missing in reflections (sky instead) |
| emissive maps (63 materials) | ignored | reflected lamps/windows black; feasible: emissive map x intensity at the hit |
| area lights (91) | skipped | no light from them on reflected hits |
| characters, hands (`.md6`) | traced with `rt_characters = 1` (section 14: GPU-skinned, refitted BLAS; cloth and skin/hair materials not traced) | without it no reflections or shadows of NPCs |
| foliage translucency | opaque alpha-tested cards | reflected leaves darker on the back side |
| fog, particles, decal types 1-3 | not traced (fog is the engine's pass after injection) | none for occlusion; reflections without them |

## 11. Photo mode: in-game integration

Design 11.1-11.5 (research, 2026-10-18); implementation status, offline checks and the
in-game test steps in 11.6-11.8 (STATUS E075). Exe `b671f824...6180`, image base
0x140000000. **O** = IDB `local/ida/Dishonored2.exe.i64`, repo files or the RX010 cvar
dump; **H** inferred; **U** unknown. Nothing has been run in game.

### 11.1 Pause and photo-mode detection

**The engine contains a complete NVIDIA Ansel photo mode (O).** It never activates on AMD,
but its game-side hooks remain in the binary and we can drive them ourselves.

- `AnselSDK64.dll` ships in the game root. Imports: `setConfiguration`, `updateCamera` (IAT
  slot 0x141A7E058), `markHdrBufferBind`, `markHdrBufferFinished`. `sub_14036F510`
  (render-system init, call at 0x14036F6B9) registers four callbacks: startSession
  `sub_1408B0160`, stopSession `sub_1408B02C0`, startCapture `sub_1408B0110`, stopCapture
  `sub_1408B0280`.
- **Session flag `byte_1439A529C`** (RVA 0x39A529C) is written only by those callbacks.
  - startSession refuses when there is no game, the game is already paused (`idGameLocal`
    vtable +232), vtable +248 returns true (meaning U), no map is running (vtable +64
    0x1408A9670: `game+3412056 == 3`), `off_1422FCD70` vtable +272 returns true (meaning
    U), or the top logic state is 5 (cinematic).
  - Otherwise it saves `rs_enable` and sets it to 0, sets `r_skipIggyFullscreen 1`, saves
    `r_skipIggy3D` and sets it to 1, buffers `Ansel_PlayerVisible 0` (handler
    `sub_1408A6BB0` hides the local player entity), and sets `byte_1439A529C = 1` and
    `byte_1439A529E = 1` (initialise the camera). stopSession restores all of this.
  - startCapture also sets `r_skipCameraMotionBlur 1` and
    `r_skipAutoExposureTemporalAdapatation 1`; the cvar description says this freezes
    exposure "for HQScreenShots".
- **The game pauses during a session.** `idGameLocal::IsPaused` (vtable +232, 0x1408A89B0)
  returns `byte_1439A529C | topStateDecl->byte[+120]`, so the game frame `sub_1408ACDF0`
  takes the paused path `sub_1408AD580` (logic manager and UI only) instead of
  `sub_1408AD260` (full simulation).
- **The renderer reacts to the session flag:**
  - `idRenderView_Setup` (0x140375711) calls `updateCamera(&cam)` with the `ansel::Camera`
    block at 0x14397BC60: position[3], quaternion[4] (0x14397BC6C), fov in degrees
    (0x14397BC7C), projection offset[2] (0x14397BC80). The first frame fills the block
    from view +14968 (origin) and +14960 (fov y). After the call the block is written back
    into the view (origin, axis, fov x/y), and at 0x140375AE9 the projection offset is
    added to projection +29592/+29608.
  - `ARK_RENDERPASS_SORT` `sub_14033D320` skips models with `modelDepthHack`
    (renderModel +6328) > 0, so **the first-person hands and weapon are not drawn**.
  - `PostFX_TemporalAA_and_Post` (0x140327E92) copies into `exrBuffer`; the frame graph
    `sub_140351780` (0x14035820F) brackets the frame with `markHdrBufferBind/Finished`.
- **Game logic state** (d2enhanced `re/NOTES.md` lead, confirmed in the IDB):
  - `arkGameLogicManager` = `game + 0x1F6EC8` (e.g. `mov rcx, cs:qword_1439A5290; add rcx,
    1F6EC8h` at 0x1404EA077; `idGameLocal` vtable +240 0x1408AD8B0 calls
    `SwitchState(game+2059976, 2)` on focus loss, `win_pauseOnFocusLoss`).
  - `manager+8` points back to the game (null check in CommitSwitch 0x140787510).
  - State stack: `int*` at +1320, count in the low 24 bits of +1328, top-of-stack getter
    `sub_140787900`. IDs: 1 in game, 2 pause, 5 cinematic, 6 main menu, 15 map transition
    (d2enhanced: verified in game).
  - **Stable game pointer:** `idCommonLocal` is the static object 0x1425AF110; its vtable
    (0x141AC4D88) slot +400 is `mov rax,[rcx+19A8h]; ret`, so **game =
    `*(u64*)0x1425B0AB8`** (RVA 0x25B0AB8). `qword_1439A5290` is only a scoped "current
    game" (set/restored in `sub_1408A2DC0`/`DE0`) and must not be used from our hooks.
- Cvars searched (dump plus IDB xrefs): `g_freezeTime`, `g_freeCam`, `g_showHud` have no
  readers besides registration (dead, H). `g_stopTime` (int 0x1439A0E38) is live: the
  game frame takes `sub_1408AD4E0` and the renderer freezes particle/time inputs;
  `g_runFrames` steps it. `com_timeScaler` is live but its meaning is U. There is no
  `com_pause`.

**Design.**

- The photo hotkey is polled in `hook::observe` (main thread, after `RenderAllViews`;
  `void-inject/src/hook.rs`). It checks that game = `*(0x1425B0AB8)` is non-zero,
  `*(game+0x1F6EC8+8) == game`, the top state is 1, `game+3412056 == 3`, and
  `byte_1439A529C == 0`. If all pass, it calls `sub_1408B0160(&zeroed, 0)` on the main
  thread, after checking its prologue bytes (`48 83 EC 28 48 8B 0D ...`) as the other hooks
  do. Requirement (H): `qword_1439A5290 == game` at that moment; otherwise refuse.
- Exit calls `sub_1408B02C0`; we also exit when the state stack or the map changes, or on
  any read failure.
- Free camera: replace the `updateCamera` IAT slot in memory (no file change). Our function
  moves `ansel::Camera` from mouse/keys while the overlay holds input (`mouse.rs` and the
  overlay input blocking exist). Accumulation restarts when the block changes.
- During accumulation, set `r_skipAutoExposureTemporalAdapatation 1` (as the engine's
  capture callback does; `cvars::set_int`).
- **Fallback** if the session refuses or crashes: our own freeze with `g_stopTime 2`
  through the cvar override mechanism (`cvars.rs`), or the user's pause menu (state 2)
  with `r_skipIggyFullscreen 1` to hide the menu and `r_skipDOF 1` (pause OnEnter
  `sub_1404F2FC0` turns DOF on, O). The camera is then frozen from the view request, with a
  camera-unchanged check.

### 11.2 Camera, sun, lights and objects at the photo moment

| Input | Source | Status |
| --- | --- | --- |
| pause / freeze | `byte_1439A529C`; logic stack `game+0x1F6EC8` +1320/+1328 (11.1) | O in the IDB; untested in game |
| camera pose | Ansel block 0x14397BC60 (free camera); view request `renderView` vieworg +168 / axis +204 / fov +156 (`void-inject/src/scene.rs`) | O |
| projection / jitter | `r_znear` / `r_zfar` (E035; 0.01 / 7000); view +29584 projection, +29776 view, +30800 jitter (`fsr.rs`) | O |
| map / reload | world +132328 map name (E033); top state 15; `game+3412056 == 3` | O (live transition untested) |
| static world | `.bwm` decoded offline (`void-rt`: 1.9 s load, 83 ms BVH build, E070) | O |
| live entities | world model list +132408/+132416, parms origin/axis/scale, flags (checkpoint 9282529 `snapshot.rs`); one walk ≈ 47 ms on the main thread (E044), once per photo while paused | O |
| sun | view +48344 `m_sunDirection` (`contact.rs`); env handles for `sunColor`/theta/phi (9282529 `scene.rs::read_sun`) | O direction; radiance scale: 574.15 x `sunColor` (section 10) |
| sky | env sky zenith/horizon handles (9282529 `scene.rs::read_env`); engine sky cube (BC6H, E041) on miss | H (section 10 fits a sky scale) |
| local lights | world lights +516464/+516472, 232-byte struct (E035, `read_lights`), 621 at RX010 | O; the offline renderer samples 524 (section 10) |
| exposure / units | auto-exposure buffer + `exposureIndex`, EV bias, white level (`fsr.rs`, `EngineLuminanceFactor`) | O |
| skinned NPCs / hands | `.md6` not decodable; taken from the game frame (11.4) | gap |
| game HDR colour + depth | TAA site (`fsr.rs` record/submit hooks) or before FOG (`sub_140309430`) | O sites, H use |

The photo snapshot is taken in one frame, the first one after the session starts: camera,
sun, env, lights and the model walk are copied on the main thread into owned Rust structs
and handed to the worker. Game memory is not touched after that.

### 11.3 Where the D3D12 render runs

- **Device:** one D3D12 device on the game's adapter inside the DLL. FSR already creates
  one (`fsr.rs`, `D3D12CreateDevice`); factor it into a shared module and give photo mode
  **its own command queue**, so that long path-tracing lists never sit in front of FSR jobs
  (FSR waits on its queue on the submit path, E058/E060).
- **Thread:** a photo worker thread owns scene loading, the BVH build and accumulation; the
  main thread only posts commands (start, camera changed, cancel) and reads progress
  atomics.
- **TDR and responsiveness:** at most 8 dispatches per submit and one submit in flight;
  dispatches tiled so that each takes ≤ ~10 ms, because the game keeps rendering paused
  frames on the same GPU (preemption of long compute is U). The offline renderer's 35 ms
  per full 3440x1440 sample (section 10) means tiles of about a quarter frame; 1-2 tiles
  per game frame (H).
- **Memory:** check the DXGI budget (`IDXGIAdapter3::QueryVideoMemoryInfo`) before building.
  The offline renderer uses 2.3 GiB for the process (textures 890 MiB at 512 texels, RT
  structures 373 MiB, scene tables 329 MiB, accumulators 378 MiB, section 10); the game's
  own usage is U, so the texture cap must follow the budget. Build lazily on the first
  photo of a map, keep it until the map changes, free it on `photo = 0`. Implemented with
  the texture cap following the budget: section 11.9.
- **Map change:** a changed map name or state 15 cancels the work and drops the BVH; it is
  rebuilt on the next request.
- **GPU safety:** the same rules as `void-rt` (dispatch size checks, shader clamps, WARP
  smoke test of every kernel, run log). On `DEVICE_REMOVED`, disable photo mode for the
  session and log the HRESULT; the game's D3D11 device is separate.

### 11.4 Display, saving, hands, fog and post

- **Preview (recommended, H):** inject the accumulated HDR into `hdrBuffer` **at the start
  of `ARK_RENDERPASS_FOG`** (`sub_140309430`, prologue `40 53 55 56 57 41 54 ...`), so
  fog, sun disk, volumetrics, water, OIT particles, DOF, TAA/FSR, bloom, tone mapping and
  grading apply on top (pass order forward -> FOG -> water/fog composite -> OIT -> DOF,
  fsr4-integration.md, O).
  - Write only where our primary-hit depth matches the game depth (74-78 % of pixels,
    E070/E073); keep game pixels where the game is closer (NPCs, missing objects) and in
    the sky.
  - Units: engine "stored HDR" (radiance / `EngineLuminanceFactor`,
    renderer-replacement.md). Fit one scalar per photo from the median game/ours ratio over
    matched sunlit pixels.
  - Hand-off: a D3D11-created shared texture opened by D3D12 (as in `fsr.rs`), two slots,
    each published after the CPU sees the D3D12 fence complete; no cross-API GPU waits
    inside the frame (E047/E048).
- **Simple fallback:** draw our tonemapped image at present time in the overlay paint path
  (`overlay.rs`).
- **Hands and weapon:** the Ansel session hides them. With the option "with hands",
  capture the game's HDR + depth of the last frame before the session. Hand pixels are
  those whose depth differs between that frame and the first session frame (same camera,
  since the block is initialised from the view); composite them over the photo. This is
  valid only while the free camera has not moved (H).
- **Saving:** a worker writes PNG (crate `png`), either from the back buffer (engine post
  applied) or from our own tone mapping. HDR output of the linear accumulator: the raw
  RGBA32F + `.tsv` format of section 10, or EXR with a new crate (U). Folder: setting
  `photo_dir`, default `void-rs/local/photos/`. File name `<map>_<unix time>_<spp>spp`,
  plus a sidecar `.txt` with camera, sun and spp.
- **UI:** an egui "Photo" panel with the states preparing (scene load / BVH), accumulating
  n/N spp with ms/sample, and done; buttons Save and Cancel; a HUD toast (`notify.rs`) on
  save. The hotkey toggles the session; Cancel resets the accumulator and ends the
  session.

### 11.5 Implementation steps and risks

1. **Read-only probe in the DLL:** log the game pointer, top logic state, `byte_1439A529C`
   and `game+3412056` once per second while the user pauses, opens menus and changes the
   map. Gate: the states match the d2enhanced IDs.
2. **Session toggle:** the hotkey calls startSession/stopSession with prologue checks.
   Gate (user): the world is frozen, the player, hands and HUD are hidden, the game keeps
   rendering, and exit is clean. Risks: the two U refusal conditions; the
   `Ansel_PlayerVisible` command buffered from a non-main thread; `AnselSDK64.dll`
   behaviour on AMD (U).
3. **`updateCamera` IAT replacement** with a free camera. Gate: the camera moves, TAA
   settles, and exiting restores the player view.
4. **Photo worker in the DLL:** `void-rt` as a library (scene from the archives + the live
   entity walk, BVH on its own queue, kernels WARP-tested offline first), a tiled
   accumulator with chunked submits, and a present-time preview. Gate: one short full-size
   run with the game staying responsive.
5. **Pre-FOG injection** with the depth-match mask and the unit fit, then the hands
   composite. Risks: the unit/exposure mismatch (E037 ratios 1.2-1.5; section 10 shade
   2.4x), seams at mismatching geometry (LOD), TAA jitter vs our unjittered image.
6. **Save PNG (+ HDR), progress UI and cancel;** map-change and device-removed handling.

Other risks: VRAM headroom is unknown; d2enhanced also hooks CommitSwitch (we only read);
other features hook the same `void-inject` sites, so `hook.rs` changes need coordination.

### 11.6 Implementation status (2026-10-18, STATUS E075)

Code: `crates/void-inject/src/photo/` (photo mode), `src/rt/` (scene service shared with
live ray tracing), the Photo panel `src/overlay/ui/photo.rs`. Settings (all live):
`photo = 0` (default: no hook, no patch, no thread), `photo_key = F4`, `photo_spp`,
`photo_bounces`, `photo_sky_scale` (x the validated 0.067), `photo_budget_ms = 8`,
`photo_texture_size = 512`, `photo_display = composite | overlay` (since E081: `hybrid`
(default, section 13) `| pathtrace | overlay`; `pathtrace` is the former `composite`), `photo_units = auto |
<number>`, `photo_dir = void-rs/local/photos`, `photo_probe = 0`, and for the scene service
`rt = 0`, `rt_build = load | use`, `rt_radius = 40`. The default key is F4, not F9: F5/F9
are the game's quicksave/quickload (the user's `dishonored2Config.cfg`), d2enhanced uses
F2, F6-F8, F10, F11, ours F2/F3/F6/F7. There is no separate ambient scale: the photo
renderer has no ambient term besides the sky (section 10), so only the sky scale exists.

| Item | Implemented | Evidence |
| --- | --- | --- |
| Probe (`photo/probe.rs`) | Ansel flag `byte_1439A529C`, game `*(u64*)0x1425B0AB8`, scoped game `qword_1439A5290`, manager back-pointer, state stack top + depth (`sub_140787900` logic), `game+3412056`, camera block 0x14397BC60 (pos, quat, fov); checked reads; logged on change (`photo probe: ...`) with `photo = 1` or `photo_probe = 1`; Probe card in the panel | O (IDB decompilation of the getters); in-game values U |
| Entering | Preconditions of 11.1 (game, back-pointer, state 1, map running, flag 0), overlay closed, a FOG pass seen (gives the render-region size), D3D12 device with DXR 1.1; then `startSession` `sub_1408B0160` on the main thread after its 16 prologue bytes (`48 83 EC 28 48 8B 0D 25 51 0F 03 48 85 C9 0F 84`) and only if `qword_1439A5290 == game`; accepted = returns 1 and the flag reads 1 | O: the callback ignores its arguments and refuses itself (returns 0) under the conditions of 11.1 |
| Fallback | Refused session or failed check: `g_stopTime 2` (`cvars::set_int`, old value restored on exit); camera fixed, HUD/hands stay | H |
| Free camera (`photo/ansel.rs`) | `updateCamera` IAT slot 0x141A7E058 swapped (atomic compare-exchange, `mem::swap_pointer`) only if it holds `GetProcAddress(AnselSDK64.dll, "updateCamera")`, restored on exit. W/A/S/D, Q/E down/up, mouse look, Shift x5, Ctrl x0.2, R start pose. Quaternion = rotation whose columns are forward/left/up (`sub_140110080` / `sub_140111410`, unit-tested round trip); fov untouched; projection offset 0 | O (IDB: block written back into the view after the call) |
| Input | `mouse::set_capture`: game mouse/keyboard records withheld while the devices stay acquired, mouse X/Y summed for the camera (the overlay's blocking still takes precedence when open); keys polled with `GetAsyncKeyState` while the game window has the focus; Enter saves, Esc/hotkey leaves | offline-built only |
| Leaving | Hotkey, Esc, the panel button, `photo = 0`, a logic-state change (menu, cinematic, transition), map change, a session flag that drops, or a render error. `photo::exit_reason` evaluates them on this tick's probe, never in the entry tick, and checks the logic state and session flag only after a grace of 10 frames. In-game run 1 bug: the entry tick compared against the probe read before `startSession` (flag 0) and left after 0.0 s with "the Ansel session ended". O: `byte_1439A529C` is written only by startSession/stopSession, and both are referenced only from the Ansel callback registration `sub_14036F510`, so the flag cannot drop by itself while AnselSDK stays inactive. Each exit logs `photo: exit condition: <reason>; probe: ...`. Order: task cancelled, composite off, input released, IAT restored, `stopSession` `sub_1408B02C0` (prologue `48 83 EC 28 8B 15 DE 4F 0F 03 48 8D 0D EF BB 0C`, same current-game check) or `g_stopTime` restored. Each failure: toast + log line | exit logic unit-tested; in game U |
| Scene (`rt/scene.rs`, `rt/live.rs`) | One `void-rt` worker thread owns the `Gpu` on our own queue of the shared device (RtLib: the same `Gpu` must upload the scene and run its sessions). Photo entry copies entities (static drawable models, checkpoint 9282529 filters), lights (E035 struct), sun and sky (env parm block) from game memory on the main thread when the map's scene is neither built nor queued/being built, and posts the build to the worker directly (not through the render task), so leaving photo mode does not cancel it; `void_rt::ingame::load_scene` + `upload` then run on the worker, and decode/upload times are logged and shown (in-game run 1: snapshot 43 ms, 5,538 entities, 622 lights). The scene is kept for further photos while `photo = 1` and the map is unchanged, dropped on a map change or with `photo = 0` and `rt = 0`. Decision: photo-only use builds at entry (no VRAM cost while not used: 2.3 GiB at 3440x1440 in section 10); with `rt = 1` and `rt_build = load` it is built at map load (60 frames after a world change, in game) | offline-built; load time in game U (offline: decode 1.9-2.3 s + upload, section 10) |
| Device | `rt/device.rs`: `D3D12CreateDevice` on the adapter of the engine's D3D11 device (LUID) returns the per-adapter singleton, i.e. the same device FSR's runtime creates; our work runs on our own direct queue, never on FSR's | H (singleton rule of D3D12) |
| Render (`photo/task.rs`) | `PhotoSession` at the render region size, camera from the view (then the free camera), `znear` = `r_znear` (x 0.25 with cramZNear); one `step` per worker tick with `photo_budget_ms`, tiles of 1/16 of 3440x1440 (~3 ms on the RX 7900 XTX, RtLib); a camera move restarts the accumulation (`reset_camera`); changed bounces/sky scale re-create the session | void-rt validated (section 10.5); in game U |
| Composite (`photo/composite.rs`) | FOG pass callback `sub_140309430` hooked when `photo = 1` (16 prologue bytes `40 53 55 56 57 41 54 41 55 41 56 41 57 48 81 EC`). Before the original records: `cs_stats` (log2 game/ours luminance over every 4th matched pixel) and a full-screen draw into `hdrBuffer` (RGB mask, alpha kept) of ours x units where our depth matches the game's (`zbuffer` of the pass write list, `z = depth_mul / (d - depth_add)` from the view projection; tolerance 2 % + 2 cm; game sky and our misses keep the game pixel). Units: `photo_units = auto` = exp2(mean log2 ratio) of this frame (fallback 3.58 below 64 samples) | O: the FOG callback's first lookups are `hdrBuffer` 0x8F93AB75 and `zbuffer` 0xBA729A59 in the write list; offline check below |
| Hand-off | Two slots of D3D11-created shared textures (RGBA32F, R32F) opened by D3D12; `copy_out` into a slot that is neither published nor read for 4 FOG passes, CPU poll of its fence, then publish; at most every 250 ms while accumulating | H: 4 frames > DXGI's default frame latency 3 |
| Paused frames | The engine keeps rendering during the session: `idRenderView_Setup` calls `updateCamera` and writes the block back into the view each frame while the flag is set (that is what makes Ansel's free camera move), and `IsPaused` only switches the game frame to its paused logic path | O for the per-view call; continuous rendering H until the first in-game run |
| Overlay display | `photo_display = overlay` or composite unavailable: our image tone-mapped (log-average to 0.18, Reinhard, sRGB) and letterboxed over the back buffer in the overlay's present path (needs `overlay = 1`) | offline check |
| Saving | Enter or the panel button: `<map>_<unix>_<spp>spp.bin/.tsv` (raw RGBA32F, absolute units), `.txt` (camera, sun, sky, units, scene times), our tone-mapped PNG (`_rt.png` when composited), and with the composite + overlay installed the back buffer (engine fog/post/tone mapping, Ansel-hidden HUD) as `<name>.png`, copied before the overlay draws and written a few frames later; toasts on save | offline-built only |
| Panel | Overlay "Photo" tab: enable, hotkey, enter/leave and save buttons, progress bar, ms/sample, spp/s, M samples/s, path settings with restore icons, scene service state, probe values, FOG/composite counters (`overlay_preview` images 25-27) | offline preview |

**Offline checks (O, this machine):** `cargo run -p void-inject --release --example
photo_offline -- --warp 256x144` (WARP), then `-- 640x360` (RX 7900 XTX), both logged in
`local/rt/runs.log`: a stand-in D3D11 device with an `hdrBuffer` (RGBA16F with alpha, and
R11G11B10) and R32F hardware-style depth, our image written by D3D12 into a slot
(`CopyTextureRegion`, COMMON -> COPY_DEST -> COMMON, CPU fence), the FOG recording on a
deferred context: 162,000 matched pixels = ours x units (fitted 2.500 for data built for
2.5, worst error 0.5 %; fixed 1.7; R11G11B10 1.2 %), 82,032 other pixels bit-identical,
alpha unchanged, deferred-context state restored; preview letterboxed. Unit tests: the
quaternion round trip against the engine's matrix, free-camera movement, the constant
buffer layout, back-buffer format conversion. `overlay_preview` clicks the Photo tab and
its enter/leave button.

**Needs the game (untested):** every engine interaction: probe values and state IDs,
the Ansel session (accepted or refused, what it hides), the IAT swap and camera
convention, input capture, the FOG hook and `hdrBuffer`/`zbuffer` formats and depth
agreement, the unit fit, frame pacing with the path tracer on the same GPU, VRAM, the
back-buffer PNG, leaving cleanly.

### 11.7 In-game test steps

Stop at the first failure and send `void-rs/local/inject/void-inject.log` (plus a
screenshot when something looks wrong). Settings in `void-rs/local/inject/void-inject.ini`
or the overlay (F2, Photo tab). These steps test the path tracer: set
`photo_display = pathtrace` (the default `hybrid` is tested by section 13.10).

1. **Probe only:** `photo_probe = 1`, `photo = 0`. Start a save; in the log lines
   `photo probe: ...` appear on every change. Check: in game `state 1 (in game)`, map
   state 3, `manager ok`, `current ... same`, session flag 0; Esc menu -> state 2; main
   menu -> state 6; loading another map -> state 15 then 1. No other behaviour change.
2. **Enable:** `photo = 1` (live). Log: `photo: FOG pass callback hooked`, mouse hooks
   present. Playing normally shows no change; the Photo tab shows FOG passes counting.
3. **Enter:** in game, overlay closed, press F4. Expected: toast "Photo mode: ... game
   paused (Ansel session)"; log `startSession returned 1, session flag Some(1)`,
   `updateCamera slot ... redirected`, `photo: entered ...`; the world freezes, player
   hands and HUD disappear, the game keeps rendering. If the toast says "g_stopTime" the
   session was refused: report the log line and continue (fixed camera).
4. **Free camera:** W/A/S/D, Q/E, mouse, Shift/Ctrl; R returns to the start. The view
   moves smoothly; nothing reaches the game (no menu opens, no weapon change).
5. **Scene:** first entry on a map logs `rt: snapshot of ...` and `rt: scene of ... ready:
   decode X s, upload + BVH Y s`; the panel shows "preparing the scene" then
   "accumulating". Note the times, the game's responsiveness meanwhile and the VRAM line.
6. **Composite:** once accumulating, static geometry turns into the path-traced image
   with the engine's fog, bloom and grading on top; characters, sky and unsupported objects
   stay as the game draws them. Log `photo: first composite: hdrBuffer ... format ...,
   zbuffer format ...`. Moving the camera restarts it (game image while it refills). Watch
   the frame rate (budget `photo_budget_ms = 8`).
7. **Units:** brightness close to the game's. If clearly off, try `photo_units = 3.58` and
   report both impressions.
8. **Save:** Enter. Toasts "Photo saved: ..." and "Game frame saved: ..."; files in
   `void-rs/local/photos/` (`.png` game frame, `_rt.png`, `.bin`/`.tsv`, `.txt`).
9. **Overlay display:** `photo_display = overlay`: our tone-mapped image over the frame;
   back to `pathtrace`.
10. **Leave:** F4 (and in another run Esc). Toast "Photo mode: left"; log `stopSession`,
    `updateCamera slot restored`; HUD, hands and control return at the old position.
11. **Exits:** enter again, then open the overlay and use "Leave photo mode"; enter and
    let a checkpoint/transition happen or quit to the menu: photo mode must end with a
    toast and the game must behave normally afterwards.
12. **Off:** `photo = 0` while in photo mode leaves it; after a restart with `photo = 0`
    the log has no `photo:` hook line.

### 11.8 Ray-tracing scene service for live effects (`rt`)

`rt = 1` starts the same worker without photo mode: the map's scene (with the FC6
24-byte primitive table) at map load (`rt_build = load`) or on first use, and a TLAS of the
instances whose bounds lie within `rt_radius` of the camera, rebuilt in a TLAS no consumer
holds when the camera moved a quarter radius (at most every 100 ms; logged counts and times
in the panel). Consumers call `rt::frame_scene(&ID3D12Device) -> Option<SceneView>` (TLAS
VA, centre/radius, primitive offsets + FC6 primitives with counts, sun/sky in absolute
units read every 30 frames) and keep the view until their GPU work finished. A GPU error
(device removal) stops the worker for the session and is shown in the panel; the game's
D3D11 device is separate. Not run in game; no live consumer is in the tree yet.

### 11.9 Video memory, submit lengths and concurrency (2026-10-18, STATUS E078)

**The removal (user run of the E075 build).** `rt = 1`, `rt_build = load`, photo mode,
`photo_texture_size = 2048`, `photo_spp 1000`, `photo_bounces 5`. The log
(`local/inject/void-inject.log` 33386-33932): the 512-texel photo scene finished 1000 spp,
then a 2048-texel scene with FC6 primitives was built ("decode 22.2 s, upload + BVH 8.1 s"
at 1792290636.568); 0.38 s later the overlay's back-buffer view failed with 0x887A0005,
then the photo task's `CreateCommittedResource (192 bytes)` (its `Params` upload). Evidence:

- **Kind of removal (O):** Windows Error Reporting has a `LiveKernelEvent 141`
  (VIDEO_ENGINE_TIMEOUT_DETECTED, `WATCHDOG-20261018-0430.dmp`) at 04:30, the minute of the
  log's removal (1792290636 = 04:30:36 local): a GPU engine timeout (TDR) that reset the
  adapter, so the game's D3D11 device was removed too. The dump needs a kernel debugger and
  admin rights (not available here).
- **Our submits were short (O, offline reproduction):** `rt_upload_check` (below) at 2048
  with FC6 primitives on the RX 7900 XTX: longest submit 8.3 ms GPU (a 64 MiB texture
  batch); BLAS 4 batches (longest 4.5-5.2 ms), compaction 1.2-1.3 ms in total, full TLAS
  1.5 ms, cut-off TLAS build/rebuild 0.3 ms. The E075 code used the same batches (64 MiB
  textures, 256 MiB scratch arena = the same 4 BLAS batches) plus one compaction submit
  for all 7,263 BLASes (1.2 ms here). The 8.1 s are CPU: in the reproduction the texture
  phase took 7.1 s wall for 265 ms of GPU time (per-texture allocation, staging copies).
- **Memory (O offline, H in game):** textures at 2048 are 6,162 MiB on the device
  (5,982 MiB of mip data) against 889 MiB at 512; the whole scene took 7,312 MiB of local
  memory after the upload. In game the previous 512-texel scene stayed resident during the
  build: the worker had taken the photo task (whose session holds the `Arc<GpuScene>`) for
  the whole 30 s build, so two scenes, the game, FSR and our SSR/AO targets shared the
  card. The game's own usage was never logged ("VRAM in use 0 MiB": `Gpu::from_existing`
  had no adapter).
- **Race (O, code review):** none found that matches. All our submits come from the one
  worker thread and one `Gpu`; the cut-off TLAS of the live service is rebuilt only when no
  consumer holds it and has its own scratch; photo traces the scene's own full TLAS;
  `rt_reflections` was 0, so no other queue read our structures.

**Most probable cause (H):** video memory over-subscription (a second, 7.3 GiB scene while
the old one and the game were resident) made the memory manager page around a GPU job
until an engine exceeded the 2 s TDR limit. A long submit of ours is ruled out by the
measurements; a race is not supported by the code. The next run logs the budget and the
usage per upload phase and the device-removed reason, which confirms or refutes this.

**Fixes (O offline):**

| Area | Change |
| --- | --- |
| Adapter | `Gpu::from_existing` finds the DXGI adapter by `GetAdapterLuid` + `IDXGIFactory4::EnumAdapterByLuid`; `video_memory()` (`QueryVideoMemoryInfo` LOCAL: budget, usage), `vram_available()` = budget - usage - headroom (max(1 GiB, 15 % of the budget), `VramPolicy`); the worker logs adapter, driver and budget at start |
| Budget guard | `ingame::estimate_memory` before any allocation: geometry, BLAS prebuild (uncompacted + scratch arena), full TLAS prebuild, tables, FC6 primitives, staging (4 batches: measured overhead), textures per cap by `GetResourceAllocationInfo` of the capped and completed chain. `upload` takes the largest cap of 2048 -> 1024 -> 512 -> 256 whose peak + host reserve (a 3840x2160 photo session + 128 MiB of cut-off TLASes) fits, drops the mips above it (same chain as decoding at that cap), re-checks before the textures, refuses below the minimum; every phase checks its estimate first. Toast "Ray-tracing scene: textures reduced to N (video memory)"; a refusal is the existing "Ray-tracing scene not built" toast. `PhotoSession::new` checks its buffers too |
| Phase log | `rt upload: <phase> done in N ms: K submits, GPU X ms (longest Y ms); VRAM in use A -> B MiB of budget C MiB (estimate +D MiB)` for geometry, BLAS, TLAS, primitives, tables + textures; summary in `GpuScene::report()` |
| Submit length | Every list carries begin/end timestamps (4 query slots above `max_queries`), `Gpu::take_submit_stats` / `last_submit_ms`. BLAS batches end at the scratch arena or a triangle count, compaction copies at a byte count, texture batches at a byte count (at most `upload_batch_bytes`); each next size = measured rate x 16 ms (`accel::SUBMIT_TARGET_MS`, at most doubling). `Gpu::set_pause`: the worker sleeps 2 ms after every blocking submit so the game's queue gets the GPU in between |
| Errors | `CreateCommittedResource` errors carry `GetDeviceRemovedReason`; the worker's final GPU error line too. A failed upload or texture batch discards the open list (it referenced staging buffers freed on return; a later submit would have executed it) |
| Scene lifetime | Before a rebuild the worker calls `Task::release_scene` (the photo task drops its session and unpublishes its slot) and hands the task back to `CTRL` for the build, so leaving photo mode during a build aborts it and the old scene is freed before the new upload. A session on an older scene generation restarts on the current one |
| Probe log | State changes logged at once, camera-only changes at most once per second |

**Measured (O, RX 7900 XTX, eotw high chaos, snapshot `snap_1792183388_f1288`,
`local/rt/upload_check/*/report.txt`):**

| Texture cap | Device bytes | Sum of mip bytes | Scene peak (estimate) |
| --- | --- | --- | --- |
| 2048 | 6,162 MiB | 5,982 MiB | 7,573 MiB (measured usage after upload 7,312 MiB) |
| 1024 | 2,477 MiB | 2,302 MiB | 3,887 MiB |
| 512 | 889 MiB | 725 MiB | 2,300 MiB (measured 2,048 MiB) |
| 256 | 339 MiB | 203 MiB | 1,772 MiB (bounded by the BLAS build) |

Rest of the scene: BLAS 838 MiB uncompacted -> 317 MiB compacted (scratch arena 256 MiB),
full TLAS 56 MiB, tables 329 MiB, FC6 primitives 123 MiB, positions + indices 95 MiB (freed
after the BLAS build). Upload at 2048: 119 submits, longest 8.3 ms GPU, 10.2 s wall (decode
14.7 s on its own). Simulated 3,000 MiB free: textures reduced to 512, longest submit
5.2 ms; simulated 1,000 MiB free: refused ("1772 MiB needed with textures <= 256").
`rt_upload_check` (`cargo run --release -p void-rt --example rt_upload_check -- --game ..
--snapshot <snap> --tex-size 2048 [--estimate-only] [--available-mib N] [--warp]`) prints
the estimate per cap and the phases; logged in `local/rt/runs.log`.

**Concurrency (O, code):**

- *Queues:* the worker's `Gpu` owns one direct queue (`void-rt`) on the per-adapter D3D12
  device; FSR (and the live RT reflections recorded into FSR's list) use FSR's queue; the
  game renders with D3D11. Only the worker thread records and submits on our queue (scene
  upload, cut-off TLAS, photo tiles, photo copies): one command list, two allocators, one
  fence, never shared between threads (`Gpu` is moved into the worker, not shared).
- *TLASes:* photo traces `GpuScene::full_tlas` (built once, immutable). The live service
  has its own cut-off TLASes, each with its own scratch and instance buffer; one is
  rebuilt in place only when its `Arc` count shows no published view or frame holds it
  (`SceneView` clones stay with FSR's frame until its fence passes). While a photo task
  runs the live updates pause (the camera is the photo camera, the game is paused, and a
  rebuild would wait for the photo tile in flight); the view is unpublished and rebuilt
  after photo mode.
- *States:* buffers live in COMMON with implicit promotion and decay after every
  submit; the TLAS / BLAS buffers stay in RAYTRACING_ACCELERATION_STRUCTURE. Resources
  read on two queues (FC6 primitives, offsets, the TLAS) are only read there; nothing of
  ours gets a barrier on FSR's queue. The photo slots (D3D11-created shared textures) are
  written by our queue only into a slot that is not published, published after a CPU
  fence check.
- *Lifetimes:* `PhotoSession` keeps a `SubmitGuard` (dropped first, waits for its submit);
  scenes are `Arc`s released by the worker, the views and the task; on a rebuild the task
  releases its session first (above).

**Unknown:** the game's own video memory use and how far the budget was exceeded at the
removal (U until the next run with the phase log); whether paging alone can exceed the
TDR limit on this driver (H).

## 12. Recommended next step

The offline photo mode exists (section 10); the in-game photo mode is implemented and
offline-checked (section 11.6), and since E081 shows the game's own frame with ray-traced
terms by default (section 13). Order:

1. The in-game test of 13.10 (hybrid terms; the paused-frame check first), then 11.7 for
   the path tracer (`photo_display = pathtrace`).
2. Photo quality of the path tracer: water surface, emissive maps, foliage translucency and
   an ambient term that reproduces the engine's probe lighting in shade.

Real-time RT reflections should come after the in-game SSR (S2) results and the E072
material target. Then build the FC6-style hybrid: half-resolution rays, compacted rays
only for SS failures, a BVH cut-off with fade, and temporal accumulation. Start from the
measured 0.94 ms (hybrid + cut-off, full resolution at 0.67 scale).

## 13. Photo mode: hybrid terms (STATUS E081)

User decision: photo mode is "game frame + RT on top". The Ansel session keeps the game's
logic paused and its renderer running (HUD, player and hands hidden); the engine renders its
own paused frame every frame, so fog, water, particles, characters, post and tone mapping
stay the engine's. Only lighting inputs inside that frame are replaced with progressively
accumulated ray-traced terms: sun visibility (`_SSAO_ScreenShadows.y`), ambient occlusion
(`.x`) and reflections (the FSR-path RT reflections in photo mode). `photo_display = hybrid`
is the default; the path tracer of sections 10-11 stays as `pathtrace` (debug, the former
`composite`) and `overlay`. Code: `void-inject/src/photo/hybrid.rs` (driver),
`photo/composite.rs` (depth capture), `ao/external.rs` + `ao.rs` + `contact.rs` (S1 entry
point), `void-rt/src/hybrid.rs` + `photo.hlsl` (kernels), example `hybrid_check`. Labels: O
observed (offline here), H inferred, U unknown. First in-game run 2026-10-18 (build
46faa27): the paused frame keeps rendering (68-103 passes/s) and the terms converge to 256
rays in 4.2-5.2 s; the four issues it found are fixed in 13.12.

### 13.1 Paused-frame check (step 0)

The terms are drawn by our detours inside the engine's passes, so they show only while the
engine keeps recording those passes. While photo mode runs, `photo::tick` takes the FOG,
SCREENSHADOWS and SSAO (`RenderAO`) callback counts of our detours once per second and logs
`photo: paused-frame passes: frames N/s, FOG N/s, SCREENSHADOWS N/s, SSAO N/s` (not hooked
= "not hooked"); the Photo tab shows the same line ("Paused frames"). The SCREENSHADOWS and
`RenderAO` detours are installed on photo entry if they are not installed yet (13.2), so
all three are counted. **If the renderer stops** (FOG 0/s while our main-thread ticks run),
a log line and a toast say so once per session: the `.x`/`.y` replacements would then show
in one frame only. Fallback: `photo_display = pathtrace` (our image composited by the FOG
hook into the last frame, section 11.4), which needs no per-frame engine passes beyond the
FOG pass itself (H: if FOG also stops, only the `overlay` display remains, drawn at present).
O (IDB, 11.1): `idRenderView_Setup` calls `updateCamera` per view while the session flag is
set; continuous rendering is H until this counter is read in game.

### 13.2 S1 external-source entry point

`ao::set_external_x(Option<ExternalSource>)` and `contact::set_external_y(..)` (types in
`ao::external`): `ExternalSource { srv, size, mode: Replace | Min, reads, clock }`. The
source texture is region-local (texel (i, j) = pixel (x + i, y + j) of the pass region),
its SRV returns the term in `.r` (R32F here), on the engine's D3D11 device. Consumed in the
existing detours, no second hook:

- `.x`: in the `RenderAO` detour after HBAO+ (`ao_mode = hbao`) or our AO (`void`) wrote the
  output RTV, for the game's call shape (AO render mask, single-pass output); region = the
  HBAO+ input viewport or the viewport the engine bound before the call.
- `.y`: in the SCREENSHADOWS detour after the engine's CSM resolve + blur and our contact
  shadows, only in CSM mode (`view+14880 == 0`; constant-fill mode keeps the engine's
  decision), inside the view viewport.

One full-screen triangle over the region, colour write mask on the channel (red for `.x`,
green for `.y`), blend off (`Replace`) or op MIN (`Min`); `saturate()` maps NaN to 0. Every
touched D3D11 state is restored (`ao::Saved`). A size different from the region (other
views, a changed render scale) skips the draw (counted). With nothing registered the
detours do one atomic load more and behave as before. Hand-off: each draw stores the
owner's `clock` (the photo composite's FOG pass counter: SSAO and SCREENSHADOWS precede FOG
in a frame) into `reads`; the owner rewrites a texture only after unregistering it and 4
more FOG passes (`REUSE_PASSES`, the composite's margin; H: DXGI's frame latency 3).

**Hook lifetime:** photo entry installs the SCREENSHADOWS detour (`contact::install`) and
the HBAO+ `RenderAO` detour (`ao::install`, SHA-256-checked DLL) if they are not installed,
also with `contact_shadows = 0` / `ao_mode = hbao`. They stay installed after photo mode
(no restart needed); with nothing registered and their own setting off they only forward
the call: bit-identical output (O, 13.5). Not removed on exit: unhooking a callback another
render thread may be inside is avoidable risk for no gain.

### 13.3 Inputs: depth, camera, sun

- **Depth: our own shared copy (decision).** Reflections' `shared_inputs().depth` exists
  only while ray-traced reflections classify on the FSR path and is rewritten every frame;
  the terms need a stable depth of one camera, also without FSR. The worker opens a
  one-shot `DepthRequest` (D3D11-created shared R32F texture of the render region, D3D11
  UAV); the next FOG pass records `cs_export_depth` (`zbuffer` -> view depth `depth_mul /
  (d - depth_add)`, 0 for sky/far/invalid) into the pass's deferred context before the
  original and stores the frame's view matrix (+29776), projection (P00, P11, jitter, depth
  terms) and sun direction (+48344, as `contact.rs`). The worker copies it 4 FOG passes
  later (`copy_texture_to_buffer`). A camera move or a changed render size captures again.
- **Normals: from our surface** (13.12): the prepass stores the interpolated vertex normal
  of our primary hit, facing the camera (octahedral, 12 bits per axis, one float per pixel).
  Only rejected receivers (below) rebuild a normal from the depth (the contact-shadow
  construction: per axis the neighbour with the smaller depth step), for AO. Not the o2
  material normal: o2 exists only on the FSR path.
- **Camera** (`photo::hybrid::frame_from_view`): rows of the 3x3 view matrix = right, up,
  back; position = -R^T t with the translation in column 3 (the `ssr.rs` convention);
  `tan_x = 1/P00`, `tan_y = 1/P11`, NDC minus the frame's jitter (unit-tested against the
  engine projection). The terms are traced for the captured frame's jitter and applied to
  later frames with other jitters: sub-pixel misalignment at depth edges (H, not visible
  through TAA/FSR).
- **Depth prepass** (`photo_hybrid_prep`, once per capture): our primary hit through each
  pixel. Where it agrees with the game's depth within max(5 cm, 3 %) on either side (a
  coarser LOD of ours, plank gaps, decals) the receiver becomes our hit with its normal.
  Elsewhere (characters, the first-person hands of the dump, foliage cards, objects we lack)
  the pixel is **rejected** (depth stored negated): the sun term is 1 there, so `min()`
  leaves the engine's CSM; AO falls back to the game depth. O: without any prepass the game
  depth at RX010 shadowed 48-50 % of the pixels against 18 % from our own depth (rays
  starting inside our geometry); now 17.0 %. 90.8 % of the receivers move onto our surface,
  9.2 % are rejected (map: hands, characters, foliage).

### 13.4 Kernels (`photo.hlsl`, root layout of the photo kernels)

| Kernel | Per pixel | Output |
| --- | --- | --- |
| `photo_hybrid_prep` | primary ray with alpha test (as `photo_depth`) | u1 depth (ours, or negated = rejected), u2 packed normal |
| `photo_sun_vis` | receiver of the prepass; rejected receivers and those facing away from the sun = 1 (N.L and the CSM darken them); else K.step shadow rays to points of the sun disc (`photo_sun_radius`, default 0.27 deg), `visible()` with the alpha-test candidate loop (foliage), decals never block | u3 running mean |
| `photo_ao` | K.step cosine rays up to `photo_ao_radius` (default 1.5 m), visibility = fraction that escapes; rejected receivers from the game depth | u4 running mean |
| `photo_hybrid_filter` | before every copy / read of the sun term: 3x3 mean over neighbours on the receiver's own plane (normal dot > 0.9, plane distance < the ray offset), centre weight 2; sky / rejected pass through | u0 (the sun output) |

Ray origin (`hybrid_origin`): eps = 1 cm + 0.2 % of the depth (+ two R24 depth steps,
`z^2 * depth_step`, for depth-rebuilt receivers); origin = P + (0.25 N - 0.5 v) eps with v
the unit camera ray: half of it back along the camera ray (free space the primary ray
crossed), a quarter along the normal. Running mean per pixel: `m = (m n + v) / (n + 1)` with
n = passes already in (K.frame), so the buffers copy straight into the R32F textures. `.y` =
min(engine CSM [+ contact shadows], RT): characters are in the BVH only with `rt_characters = 1` (section 14; otherwise they cast
through the CSM only); vegetation receivers read their own CSM (LO2), not `.y`. `.x` mode
`photo_ao_mode = min` (default, with HBAO+ / our AO) or `replace`.

**GPU safety (O):** WARP first (`hybrid_check --smoke`, logged in `local/rt/runs.log`);
every index clamped (depth reads to the region, scene indices as the photo kernels, rays
per dispatch 1..4 in the kernel); the CPU checks every per-pixel buffer per dispatch and the
session allocates the full padded rows. Scheduling as `PhotoSession`: non-blocking `step`
per worker tick with `photo_budget_ms`, row tiles (`DEFAULT_TILE_PIXELS / 2`: 88 rows at
3440), tiles packed by their own last measured GPU time, an unmeasured tile alone in its
submit, at most 8 dispatches per submit; prepass first, then sun/AO passes alternate. Two
fixes from the measurements: (1) estimating unmeasured prepass tiles from the first one
(sky rows) packed 8 tiles into a 14.9 ms submit; now an unmeasured tile goes alone (longest
submit 8.7 ms); (2) many rays per dispatch hide the ray latency badly (64 rays in 4-row
tiles: 30 ms per ray; in 36-row tiles 5.1 ms per ray and 18.8 ms per tile), so rays per
dispatch are capped at 4 and `photo_sun_rays` / `photo_ao_rays` are rays per pass (1..4);
the passes per frame follow `photo_budget_ms`.

### 13.5 Offline results (O, RX 7900 XTX, eotw, RX010 pose, `hybrid_check`)

Game depth = F8 dump `dump1_depth_r32f_3440x1440.bin` (resampled to 2304x968), 4 rays per
pass, target 64, sun disc 0.27 deg, AO radius 1.5 m, budget 8 ms. Masks:
`local/rt/hybrid/check/game_{sun_vis,ao}_{3440x1440,2304x968}.png`.

Before the 13.12 fixes (pure normal offset, depth-rebuilt normals):

| GPU ms | 3440x1440 | 2304x968 |
| --- | ---: | ---: |
| depth prepass (once per camera) | 19.2 | 12.3 |
| sun pass, 4 rays per pixel (whole image) | 12.6 (3.16 per ray) | 6.5 (1.62) |
| AO pass, 4 rays per pixel | 12.9 (3.22 per ray) | 6.3 (1.58) |
| slowest tile: prepass / sun / AO | 3.8 / 1.36 / 1.22 (88 rows) | 3.4 / 1.41 / 1.11 (132 rows) |
| longest submit | 8.7 | 7.9 |
| shadowed (sun < 0.5) / AO < 0.5, surface pixels | 18.1 % / 10.7 % | 18.1 % / 10.7 % |

After them: prepass 19.0 ms, sun pass 11.8 ms, AO pass 12.9 ms at 3440x1440, longest submit
8.2 ms (an earlier run, before the last change, the ray origin of 13.12 item 1, which does
not change the ray count); in the final run, with other offline jobs on the same
GPU (H: its upload took 59 s instead of 4 s), 19.1 / 13.2 / 14.3 ms, longest submit 13.6 ms
(still under 16). Shadowed 17.0 %, AO
< 0.5 11.2 %. The filter is one 3x3 gather per copy (every 250 ms while accumulating; not
timed separately, H: well under 1 ms).

Per frame the terms take at most `photo_budget_ms` (8 ms default). 256 rays per term =
64 passes each = 1.63 s GPU at 3440x1440 (H: about 200 frames at 8 ms) and 0.82 s at
2304x968; the first copy is shown after the prepass and one pass per term.

- **Validation** (860x360, our own primary-hit depth, against the path tracer's first-hit
  sun visibility, 64 spp, same disc): planar receivers facing the sun by both conventions
  (163,619 px): mean abs. difference 0.044 (before 13.12: 0.050), lit/shadowed agreement
  95.95 % (95.3 %); all sun-facing pixels (257,113): 0.087 / 92.2 % (0.103 / 90.6 %), the
  rest on foliage (alpha cards; the engine's vegetation ignores `.y`). Means: 0.810 vs path
  tracer 0.812 (before 0.817). Diff image `own_diff_860x360.png`.
- **False blacks** at the game depth (13.12 item 1): 46,479 -> 6,966 pixels at 3440x1440,
  923 -> 56 on the steps.
- **Texture paths:** depth from an R32F texture (`set_frame`) and from the CPU give
  bit-identical means; `copy_out` textures equal the readback.
- **WARP smoke** (96x40, 2,000 instances, 2 rays, target 8): PASS, no device removal.
- **S1 entry point** (`ao_offline`, `contact_offline`, hardware D3D11): `.x` replace and min
  after HBAO+ and after our AO: 0 pixels differ from the expected red channel, G/B/A and
  outside unchanged, state as without the source; `.y` replace and min with contact shadows
  on and off: 0 wrong (UNORM rounding +-1), constant-fill mode untouched; with nothing
  registered before and after a registration: bit-identical (also with contact shadows off:
  target untouched). Cost of the `.y` draw alone: 0.034 ms at 3440x1440, 0.017 ms at the
  0.67 region; `.x` after HBAO+ +0.05-0.10 ms.
- **Depth capture** (`photo_offline`, WARP then GPU): 201,600 view depths within 1.3e-7
  relative, sky 0, state restored.
- **Path tracer unchanged:** `photo_ingame_check` WARP and GPU bit-identical to the
  references after the shader changes.

### 13.6 Reflections

On entry (hybrid display, `photo_reflections = 1`): without FSR running a toast "reflections
need FSR on" and the term is skipped; without `rt_reflections = 1` (with `ssr = 1`,
`rt = 1`) a toast and skip; else `ssr::rt_reflect::set_photo(Some(photo_reflect_rays))`. A
free-camera move calls `reset_photo()`, leaving calls `set_photo(None)`. The tab shows
`photo_samples()`. The live cut-off TLAS the reflections trace stays published during a
hybrid photo task (`Task::allows_live`; the path tracer still pauses it).

### 13.7 Settings (all live) and Photo tab

`photo_display = hybrid | pathtrace | overlay`, `photo_sun = 1`, `photo_ao = 1`,
`photo_reflections = 1` (read at entry), `photo_sun_rays = 4`, `photo_ao_rays = 4` (rays per
pass, 1..4), `photo_reflect_rays = 4`, `photo_terms_spp = 256`, `photo_sun_radius = 0.27`,
`photo_ao_radius = 1.5`, `photo_ao_mode = min | replace`, `photo_save_masks = 0`; the
budget `photo_budget_ms` and `photo_texture_size` are shared with the path tracer. Ray
counts, radii and enables restart the accumulation; the target does not; the AO mode and
the enables re-register at once. Photo tab: progress bar per term (rays done / target, rays
shown, ms per pass), reflection samples, phase, prepass ms, submits, paused-frame rates;
cards "Photo settings", "Hybrid terms", "Path tracing". Restored on exit: both external
sources unregistered (also on every camera move), reflections back to live, the worker's
terms and capture dropped.

### 13.8 Saving

Enter / "Save photo": the back buffer (the game's frame with the terms, engine post and tone
mapping, Ansel-hidden HUD) as `<map>_<unix>_hybrid.png` (needs `overlay = 1`), notes
`.txt` (camera, region, per-term rays and settings, reflection samples), and with
`photo_save_masks = 1` the term means as `<name>_sun.png` / `<name>_ao.png`. The progress
line (13.12 item 2) is never in the PNG: the overlay copies the back buffer in its present
hook (`photo::composite::present`, `CopyResource` into a staging texture) before it draws
the HUD and toasts of that frame.

### 13.9 Untested in game (U) and risks

Every engine interaction: whether the paused frame keeps rendering (13.1), the detours'
calls during the session, `zbuffer` at FOG being the depth the SCREENSHADOWS/SSAO passes
use, the view-matrix translation slot and jitter sign in game (offline consistent with
`ssr.rs` and the engine projection), R24 precision far away, the `_SSAO_ScreenShadows`
format (R8G8B8A8_UNORM per E064), the frame margin of the hand-offs, GPU time with the game
rendering on the same GPU, video memory (term slots 16 B per pixel, capture 4 B, session
20 B: depth, normal, two means, filtered sun), the reflections' photo mode (Reflections'
API). Known limits: without `rt_characters = 1` (section 14) characters and other
skinned models receive but do not cast RT shadows (the CSM keeps theirs), water is not an
occluder, alpha-tested foliage casts through the candidate loop but vegetation receivers
ignore `.y`; terms from a stale epoch can show for one worker tick after a camera move
(the main thread unregisters at once; the worker checks the epoch before publishing).

### 13.10 In-game test steps

Stop at the first failure and send `void-rs/local/inject/void-inject.log` (plus screenshots
of the step). Settings in `void-rs/local/inject/void-inject.ini` or the overlay (F2, Photo
tab).

1. **Settings:** `photo = 1`, `overlay = 1`, `photo_display = hybrid` (default). For the
   reflection term also `fsr = 1`, `ssr = 1`, `rt = 1`, `rt_reflections = 1` (without them
   expect step 2's toast). Load the eotw save, stand outdoors in sunlight.
2. **Enter:** F4. Toasts as in 11.7 step 3; without FSR "reflections need FSR on". Log:
   `photo: entered ...`, and if not installed before `contact: SCREENSHADOWS callback
   hooked` and `ao: HBAO+ hook installed: RenderAO ...` (or `ao: ... CreateContext hooked`).
3. **Paused frames:** within 2 s the log has `photo: paused-frame passes: frames N/s, FOG
   N/s, SCREENSHADOWS N/s, SSAO N/s` with all four about the frame rate; the Photo tab shows
   it. If FOG is 0/s (toast "the game stopped rendering while paused"): stop and report.
4. **Capture:** the tab goes "preparing the scene" (first entry on a map) -> "capturing the
   game depth" -> "depth prepass" -> "accumulating". Log `photo: first depth capture:
   region ..., camera [x, y, z] forward [...], sun [...]`: the camera position must equal
   the probe's camera block position (log `photo probe:` / tab "Ansel camera") within a few
   cm, forward plausible; then `photo: hybrid terms WxH started`.
5. **Look:** within about a second sun shadows sharpen and long-range shadows of static
   geometry appear; characters keep their shadows; corners and contacts darken (AO).
   Shadows must line up with their casters (a constant offset = camera convention wrong:
   report with a screenshot). Toggle "Sun shadows" and "Ambient occlusion" in the tab for an
   A/B (live, engine shading returns at once). **Black dots** (13.12 item 1): stone steps,
   kerbs and wall bases in sunlight show no black dots or dashes along the edge where a
   tread meets a riser; with the tab's "Sun shadows" off and on the edges look the same
   except real shadows. Report a screenshot crop if dots remain (with `photo_save_masks = 1`
   and a save, the `_sun.png` mask of the same view).
6. **Progress:** with the overlay closed a toast in the top-right corner reads `Photo:
   capturing the game depth` / `Photo: sun N/256, AO N/256[, reflections N/256]` and counts
   up (refreshed at least once per second), then is replaced by `Photo ready, Enter to save`
   for 5 s and disappears. Both bars in the tab reach the target (256); note ms per pass,
   the submits line and the frame rate while accumulating; log `photo: hybrid terms done
   after ...` and `photo: ready (...)`. Save while the progress line shows: the PNG has no
   toast (13.8).
7. **Camera move:** WASD/QE and the **mouse**: mouse left/right turns, up/down pitches (the
   view must follow the mouse; in run 1 it did not). With the overlay closed the cursor is
   hidden and does not move; opening the overlay (F2) shows it again and the mouse works in
   the menu; leaving photo mode restores it. The terms vanish at once (engine shading), the
   progress line restarts, then the terms refill within about half a second after the
   camera stops; no stale shadows. Log: at most one `photo: camera moved, restarting the
   hybrid terms (N restarts since the last line)` per second while moving.
8. **Reflections** (prerequisites on): "Reflections: N samples per texel" grows while still,
   drops after a move.
9. **Settings live:** AO mode replace vs min, sun radius 0.27 -> 1.5 (softer penumbrae,
   restarts), AO radius 1.5 -> 4 (broader darkening, restarts).
10. **Save:** Enter. Toasts "Photo saved: <map>_<t>_hybrid (...)" and "Game frame saved:
    ...png"; the PNG is the game frame with the ray-traced terms, no HUD. With
    `photo_save_masks = 1` also `_sun.png` / `_ao.png`.
11. **Display switch:** `photo_display = pathtrace` (live): log `display changed to the path
    tracer`, the path tracer composites (11.7 step 6); back to `hybrid`.
12. **Leave:** F4 or Esc: engine shading at once, reflections live again; `contact_shadows`
    and `ao_mode` toggles behave as before photo mode.
13. **Off:** after a restart with `photo = 0` the log has no `photo:` hook line and no
    SCREENSHADOWS / RenderAO hook unless `contact_shadows` / `ao_mode = void` ask for it.
14. **Load toast** (13.12 item 4): load a save with `rt_reflections = 1`: no toast
    "Ray-traced reflections need: the D3D12 device (not created yet)" (the log still has
    `rt_reflect: waiting for the D3D12 device (not created yet)` just before `rt: D3D12
    device ...`); a toast appears only if a blocker persists > 10 s or is definite (e.g.
    `ssr = 0`).

### 13.11 Reproduction

From `void-rs/` (WARP first; GPU runs are logged in `local/rt/runs.log`):

```
cargo run --release -p void-rt --example hybrid_check -- --smoke --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --game-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin
cargo run --release -p void-rt --example hybrid_check -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288 \
  --game-depth local/ref/RX010/dump1_depth_r32f_3440x1440.bin \
  --measure-size 3440x1440,2304x968
cargo run --release -p void-inject --example ao_offline
cargo run --release -p void-inject --example contact_offline
cargo run --release -p void-inject --example photo_offline -- --warp 256x144
cargo run --release -p void-inject --example photo_offline -- 640x360
```

### 13.12 Fixes after in-game run 1 (build 46faa27)

1. **Black dots in sun shadows** along stone step edges (tread meets riser;
   `local/rt/dots_crop.png`). Reproduced offline: `hybrid_check` section 5 counts, at the
   RX010 game depth (3440x1440, 64 rays), pixels the path tracer (16 spp first-hit, no CSM)
   sees lit and sun-facing but the term scores < 0.25 (black under `min(CSM, RT)`), overall
   and in the steps rectangle x 1300-2000, y 700-1000, plus isolated black pixels (>= 6 of
   8 neighbours lit). Causes (O): (a) the normal was rebuilt from the depth, mixing pixels
   the prepass had moved onto our surface with unmoved ones and neighbours across depth
   edges: at the isolated black pixels its median cosine to the path tracer's normal was
   0.36 (0.998 at lit pixels), so the origin went along a wrong normal and the sun test
   used it; (b) after (a) was fixed, the remaining steps dots were mostly riser slivers one
   pixel tall at the riser foot: a pure normal offset runs parallel to the tread there and
   the shadow ray starts on or below the tread (fully blocked, 0 of 64 rays, path tracer
   1.0). Fix: the prepass stores our hit's normal and snaps on either side within max(5 cm,
   3 %), rejects disagreeing pixels (sun term 1: the CSM decides), the origin backs off
   along the camera ray (13.4), and an edge-aware 3x3 filter runs on the sun output.

   | RX010, 3440x1440 | false black, all (of 2,933,974 lit) | steps (of 176,596) | isolated, all / steps |
   | --- | ---: | ---: | ---: |
   | before | 46,479 (1.58 %) | 923 (0.52 %) | 39,345 / 1,028 |
   | prepass normal + snap/reject (mean) | 17,607 (0.60 %) | 612 (0.35 %) | 27,679 / 882 |
   | + camera-ray origin (mean) | 10,895 (0.37 %) | 219 (0.12 %) | 24,995 / 580 |
   | + filter (output) | **6,966 (0.24 %)** | **56 (0.03 %)** | 12,379 / 152 |

   The remaining isolated pixels on the steps are mostly ones the path tracer also sees
   dark or half lit (median 0.25: real thin shadows and sub-pixel edges). Gates in
   `hybrid_check`: <= 0.5 % overall, <= 0.1 % on the steps. 9.2 % of the surface is
   rejected (hands of the dump frame, characters, foliage cards): CSM only there.
2. **Progress toast:** `notify::status(key, text, ttl)` is a live line: the overlay's toast
   list replaces the toast of the same key in place (`hud::Toasts::add`), it is not logged,
   and it fades when not refreshed. `photo::tick` shows `Photo: sun a/b, AO c/d,
   reflections e/f` (reflections against `photo_terms_spp`; before the first rays the
   worker's phase; path tracer: `Photo: path tracing a/b spp`) on change and at least once
   per second (ttl 3 s), then once `Photo ready, Enter to save` (5 s, logged as `photo:
   ready (...)`); leaving removes it. Toasts draw with the overlay closed (`notifications =
   1`) and after the save's back-buffer copy (13.8).
3. **Mouse look:** cause (O, code): `photo::tick` called `mouse::set_capture(capture)` every
   frame, and `set_capture` zeroed the collected motion on every call, so the counts the
   engine's mouse poll had delivered earlier in the frame were dropped right before
   `take_motion` (WASD reads `GetAsyncKeyState` and was unaffected). The poll itself runs
   while paused (O, IDB: `idCommonLocal` frame `sub_1400450D0` -> `sub_14003A900` ->
   `sub_14003A470` -> `idSysLocal` slot 23 `GetMouse` -> poll slot 8 `sub_1400E2930`, gated
   only by the device's active bit, which is set by `Activate` and never cleared). Fix:
   `set_capture` resets only on a state change (unit test). Cursor: `mouse::lock_cursor`
   hides it (`ShowCursor(FALSE)` while `GetCursorInfo` reports it shown, counted and undone
   on release) and clips it to the window centre every frame while the free camera
   captures with the overlay closed; released when the overlay opens, focus is lost or
   photo mode ends (previous clip rectangle restored; H: `photo::tick` runs on the thread
   that owns the game window, `ShowCursor` counts per thread). Log: `photo: camera moved,
   restarting the hybrid terms (N restarts since the last line)` at most once per second.
4. **False toast at load** ("Ray-traced reflections need: the D3D12 device (not created
   yet)", 3 ms before `rt: D3D12 device ...`): produced by `ssr::rt_reflect::tick`
   (Reflections' file, changed by Reflections on request). Rule: definite blockers
   (settings `ssr`/`rt`/`fsr` = 0, device failure, no DXR 1.1, device lost, scene failed,
   camera units) toast at once; transient ones (device not created yet, scene not built or
   building, FSR on but not running yet) toast only after the same blocker persisted > 10 s
   (`TRANSIENT_TOAST`); the log line stays immediate.


## 14. Dynamic objects: moving props and characters (STATUS E086, E087)

Settings `rt_dynamic` (moving / hidden / spawned static models) and `rt_characters`
(skinned characters), both default 0. With both 0 nothing of this section runs: no scene
reserve, no per-frame reads, no decode thread, no skinning. Turning either on or off
rebuilds the map's scene (it then carries the reserve). Engine evidence:
`local://dynamic-entities.md` (DynRE), docs/formats/md6.md. O = observed (offline here), H =
inferred, U = unknown. Nothing of this section has run in game.

### 14.1 Data flow

| Where | What | Code |
| --- | --- | --- |
| Scene build | Every snapshot entity keeps its engine model (`ModelKey` = list slot + pointer) and its TLAS instance (`load_mapped`, `GpuScene::entity_instances`; 5,307 of 5,452 RX010 entities traced). With `SceneOptions::dynamic` the tables get a reserve after the static entries: 1 Mi triangles (photo records + FC6 primitives, 90 MiB), 16 Ki geometries, 4 Ki materials, 1 Ki bindless texture slots (white), records / primitives UAV-capable. The kernels' counts cover the reserve, so `frame_scene()` and every consumer binding stay as they were | `void-rt` `scene.rs`, `ingame.rs`, `photo.rs`, `textures.rs` |
| Main thread, `hook::observe` | `rt/track.rs`: one read of the committed dirty flags (`*(world+528816)+8328`, u32 per model slot, the slot count of the model list); per flagged mapped / spawned model one 60-byte transform read (+6240 origin, axis, scale) and its drawn flag; 0x4 unlink or not drawable = hidden, 0x80 = removed. Every 15 frames one read of the model pointer list: a changed slot is classified by vtable; a static `idRenderModel(Static)` that appears after the snapshot becomes a spawned prop (name + live materials read once), a skinned one (vtable 0x141B0D628) a character candidate. `r_skipUpdateInView` != 0 (the engine skips its commit): logged once, props stay static. Characters: the `2 x rt_characters_n` nearest within `rt_radius` (1.2 x hysteresis) are announced (md6 name, LOD +8120, live materials); per frame one 1,960-byte read (+6240..+8196: transform, `m_needJointUploadToGpu` +8113, LOD, palette pointer / count) for the `rt_characters_n` nearest and, when +8113 > 0 (or the first time), one palette read (48 B per joint). Events go to the worker (queue capped at 8,192, oldest dropped) | `void-inject` `rt/track.rs`, `rt/live.rs` |
| RT worker | `rt/dynamic.rs` applies the events to `void_rt::dynamic::Dynamics`; spawned props and characters are decoded on a `void-rt-decode` thread (CPU) and uploaded by the worker (tables into the reserve, rigid BLAS PREFER_FAST_TRACE; character buffers and a PREFER_FAST_BUILD + ALLOW_UPDATE BLAS), at most one character upload per tick. Each character's first palettes go through `palette_check` (14.3); then per tick `Dynamics::update` skins the queued poses nearest the camera first, at most `rt_characters_n` and about `rt_dynamic_budget_ms` of GPU time (measured cost per character), in one submit | `void-inject` `rt/dynamic.rs`, `void-rt` `dynamic.rs`, `character.rs`, `dynamic.hlsl` |
| Live TLAS | When only transforms / poses changed, the published cut-off TLAS is refitted (`Tlas::update_from`, PERFORM_UPDATE) into a spare TLAS no frame holds and whose last build had the same instance count; otherwise (instances appeared / disappeared, the camera moved a quarter radius, 60 refits) it is rebuilt. A published view is never changed under its holder; every refit / rebuild publishes a new `tlas_id`. Cut-off TLASes have room for static + 4,096 dynamic instances; at most 3 are allocated | `void-inject` `rt/scene.rs` |
| Photo mode | **Frozen** (decision): at photo entry the worker builds one full TLAS of the dynamic state (`Dynamics::full`, `SceneTlas`) and both the path tracer (`PhotoSession::set_tlas`) and the hybrid terms (`HybridSession::set_tlas`) trace it until the photo task ends. The game is paused during a photo, so nothing moves; freezing keeps the accumulation valid without resets. Characters therefore occlude the ray-traced AO and sun shadows in photos when `rt_characters = 1` | `rt/scene.rs` `photo_tlas`, `photo/task.rs`, `photo/hybrid.rs` |

Lifetimes: an `InstanceSet` (the list a TLAS was built from) keeps the dynamic BLASes it
references; a removed mesh's reserve entries and textures are reused only once no set holds
it (`Dynamics::collect`). Each character has up to 3 BLAS copies: a pose is written into a
copy no TLAS in use references (another queue may be tracing the current one); a new copy
gets a full build, later poses update it from the current copy.

### 14.2 Characters (md6)

`character.rs`: md6Def chain (`void_formats::md6::resolve_def`) -> the meshes of the
engine's current LOD (+8120; the lowest LOD the decl has otherwise) -> the direct
`.bmd6model` or, for meshes with cloth, the cloth-stripped `nocloth_*` build (cloth parts
are not traced) + the skeleton. Skeleton joint = stored index + surface joint offset (O,
md6.md); parts built for another skeleton (checksum differs: the aristo02 heads) are
remapped by joint name. Materials: the live surface materials when their count equals
the decoded surfaces', else the files'. **No character surface is dropped for its
material** (after in-game run 1 of build 574fbc5: a guard kept 2 of 11 surfaces):
materials the static path excludes (O: skin `arksssrender`, hair `arkHair`, gore parts)
are traced as primitive data (`Preparer::prepare_fallback`: the decl read as arkDefault
layer 0, the diffuse map's average as albedo when it names one, else grey 0.5; alpha test
as declared, so hair / cloth cards go through the candidate loop) and listed once per
character in the log (`rt dynamic: character ...: N materials traced as primitive data
...: <material> [program X]`). O offline: overseer 8 of 8, gs_guard_01_head1 12 of 12, rat
2 of 2 surfaces. Parts: every md6mesh of the decl chain's LOD (body + head, ...) is one
character mesh; render models that are separate parts (gear) are separate skinned models
in the engine's list and tracked each; the per-frame caps count **groups** of skinned
models sharing an origin (0.5 m grid), so all parts of the `rt_characters_n` nearest
characters are posed. Gore parts (dismemberment caps, normally hidden by
`m_visibleBodyParts` +8176) are traced too (inside the body; U whether they show).
Kernel `skin_vertices`: per vertex the weighted sum of up to 8 palette matrices x the bind
position (and normal / tangent), NaN or > 64 m displacement keeps the bind vertex;
`skin_triangles` writes the skinned normal frame into the character's photo records and the
octahedral normal into its FC6 primitives. Indices are clamped in the kernel to the counts
the CPU checked against the buffer sizes.

### 14.3 Palette convention

H (DynRE): the engine's CPU palette (+8184/+8192, 48 B row-major 3x4) is bone -> object,
i.e. joint world x inverse bind, so the bind pose maps bind vertices to themselves. O
offline: the decoded bind palette of the overseer is the identity to 8.3e-7. In game the
first palettes of every character are checked (`palette_check`): the skinned bounds of 512
sample vertices must stay within 0.5..2 x the bind diagonal and their centre within one
diagonal, tried as skinning matrices and as joint-world matrices (x inverse bind applied
here); the plausible one is used and logged (`rt dynamic: character ...: palette check:
...`), neither = pose skipped (3 tries). Palettes with fewer joints than the skeleton are
skipped and logged.

### 14.4 Measured (O, offline)

`dynamic_check` (eotw high chaos, RX010 snapshot, cut-off radius 40 m: 5,890 instances,
227 of them entities), `skin_check` (overseer_head.md6 LOD 0: 22,377 vertices, 37,674
triangles, 219 joints, 8 of 8 surfaces traced; also decodes gs_guard_01_head1 and rat).
WARP first, then RX 7900 XTX, every run in `local/rt/runs.log`, reports in
`local/rt/dynamic_check/` and `local/rt/skin_check/`. Numbers below are from the run before
the material fallback; with all surfaces traced (RX 7900 XTX, 12 overseers): 0.69 ms GPU
per frame (0.058 ms each), 21.4 MiB each, longest submit 4.5 ms; cut-off refit
0.060-0.064 ms, rebuild 0.39-0.41 ms (same scene, run-to-run variation).

| Check | WARP | RX 7900 XTX |
| --- | --- | --- |
| dynamic list without changes = static cut-off list | pass | pass |
| static instance moved 50 m + refit: gone, other depths unchanged | pass (2.0 ms) | pass (0.05 ms; 301 equal-depth id swaps of coplanar duplicates) |
| moved back + refit = first image (depth) | pass | pass |
| hidden: refit refused, rebuild without it | pass | pass |
| spawned props (decoded as dynamic meshes) traced, reserve freed after removal | pass (2 props) | pass (4 props, 17,030 pixels) |
| instance types: moved = 1 (only that instance), spawned props = 2 | pass | pass |
| bind palette = identity; bind pose skinned = decoded vertices | 8.3e-7; 3.8e-7 m | 8.3e-7; 3.8e-7 m |
| one joint rotated 60 deg: GPU = CPU reference; unweighted vertices fixed | 3.6e-7 m; 3.8e-7 m | 3.9e-7 m; 3.8e-7 m |
| skinned normals in the records only where the pose changed them | pass | pass |
| primary rays see the character in the bind and posed TLAS | 1,251 / 1,244 px | 1,251 / 1,244 px |
| palette check recognises skinning and joint-world palettes | pass | pass |

Cut-off TLAS per frame (RX 7900 XTX, medians of 20 frames; engine reads estimated from
the sizes, the read cost itself cannot be measured offline: one u32 per model slot, 9,370
in RX010, plus 64 B per moved model):

| Moved per frame | Engine reads (est.) | CPU (state + instance list) | Refit GPU | Rebuild GPU |
| --- | --- | --- | --- | --- |
| 1 | 4 reads, 36.7 KiB | 0.018 ms | 0.047 ms | 0.288 ms |
| 10 | 22 reads, 37.2 KiB | 0.018 ms | 0.048 ms | 0.293 ms |
| 100 | 202 reads, 42.9 KiB | 0.021 ms | 0.049 ms | 0.290 ms |
| 227 | 456 reads, 50.8 KiB | 0.021 ms | 0.047 ms | 0.297 ms |

Characters (RX 7900 XTX): 12 overseers posed every frame for 20 frames: skinning + BLAS
update 0.82 ms GPU per frame (0.069 ms per character), TLAS refit 0.016 ms, longest
submit 4.6 ms (the first BLAS builds); 19.6 MiB per character with 3 BLAS copies; adding one
character 264 ms on the worker (tables, buffers, bind-pose skin + BLAS build). Every submit
<= 16 ms (longest of the whole step A run 0.34 ms). WARP (functional only): 53 ms per
character.

O, WARP: an update (refit) into a TLAS last built with a different instance count never
finished; `Tlas::update_from` now requires equal counts and the worker only refits into
such a spare (after one rebuild the previous TLAS qualifies). A BLAS update into a newly
allocated copy was traced as empty on WARP; new copies get a full build.

### 14.5 Untested in game (U) and risks

Every engine read: that the committed flags hold this frame's changes after
`RenderAllViews` and how many models flag per frame, the slot count (+132416) as the flag
range, the parms layout of spawned models, the vtables, the skinned model fields (+8113,
+8120, +8184/+8192), the palette convention and joint count against the decoded skeleton,
LOD changes (a LOD change is a new character key: removed and re-added), runtime material
swaps (live materials used only when their count matches), cloth (not traced), first-person
view models (14.8), the read cost per frame, VRAM with many characters (`rt_characters_n`
x 2 groups kept, ~21 MiB per mesh), and GPU time next to the game. Reflections see dynamic
objects through `frame_scene()` unchanged (same table layout); their shading of characters
uses the skinned FC6 normals; fallback materials have albedo / smoothness from layer 0
only (the path tracer shows white where a fallback material has no diffuse map: its
texture slot is the white texture).

### 14.6 In-game test steps

Stop at the first failure and send `void-rs/local/inject/void-inject.log`. Settings in
`void-rs/local/inject/void-inject.ini` or the overlay (F2, Photo tab, "Ray-tracing scene").

**Quick verification (3 steps, after build 574fbc5 feedback):**

1. `rt = 1`, `rt_dynamic = 1`, `rt_characters = 1`, `rt_reflections = 1`, `ssr = 1`, and
   `rt_reflect_debug = instances` (raw rays, no denoiser; docs/reflections.md 4). Stand near NPCs and a
   pushable prop: reflected statics are grey, NPCs red, a pushed prop yellow, a spawned or
   dropped item green. The overlay's Photo tab ("Ray-tracing scene") shows "Characters
   traced: N (surfaces M of K)" with M = K, "Props: ... this second: N moves", and "First
   person: N view models excluded" with N >= 1 once a weapon is drawn; the log has
   `rt dynamic: first-person view model left out of the scene: models/weapons/...` once per
   model and no `gun_emily` character line.
2. `photo = 1`, `photo_save_instances = 1`: F4 near NPCs, Enter. Besides the photo,
   `void-rs/local/photos/<map>_<time>_hybrid_instances.png` shows our primary rays coloured
   the same way (red NPC silhouettes with their heads, bodies and gear; no hands / weapon
   in front of the camera); the log has `rt dynamic: photo TLAS frozen: N instances (K
   dynamic)`.
3. In that photo, NPCs cast ray-traced shadows and darken AO around their feet
   (`photo_save_masks = 1` writes the sun / AO masks to compare); push a prop, re-enter
   photo mode: the yellow prop is at its new place in the instance image.

Full steps:

1. **Off:** with the defaults (`rt_dynamic = 0`, `rt_characters = 0`) nothing changes; the
   log has no `rt dynamic:` line.
2. **Props on:** `rt = 1`, `rt_dynamic = 1` in game. Log: `rt: building the scene of ...
   dynamic reserve true`, then `rt dynamic: on for '...'` and `rt dynamic: tracking N model
   slots, M static instances mapped`. Photo tab rows "Dynamic reads" (slots, flagged per
   frame, reads, µs) and "Props". Note the µs value while walking around.
3. **Move props:** push / throw objects (bottles, chairs, crates). "Props: N moved"
   grows; with `rt_reflections = 1` the reflection of the object follows it (no ghost at
   the old place). Pick up and drop objects; destroy one: "hidden" grows.
4. **Spawn:** drop an item from the inventory or let a guard drop one: "spawned" grows;
   lines `rt dynamic: prop ... not traced: ...` name what could not be decoded.
5. **r_skipUpdateInView:** in the console `r_skipUpdateInView 1`: log `r_skipUpdateInView is
   set ...` once, the row shows it; set it back to 0.
6. **Characters on:** `rt_characters = 1` (scene rebuilds). Near NPCs the log shows
   `rt dynamic: character <md6> LOD n: ... vertices` and `palette check: ... -> skinning
   (bone -> object)` (report it if it says joint world or neither). Row "Characters: N near,
   M in the scene, K posed; ... ms GPU". With `rt_reflections = 1` NPCs appear in
   reflections and move with their animation.
7. **Budget:** watch the frame rate with 10+ NPCs near; lower `rt_characters_n` or
   `rt_dynamic_budget_ms` and report the "ms GPU" value.
8. **Photo:** F4 near NPCs (`photo_display = hybrid`): log `rt dynamic: photo TLAS frozen:
   N instances (K dynamic)`; NPCs now cast ray-traced shadows onto the ground and darken AO
   around them; moved props sit where they are in the frame.
9. **Map change / quickload:** the scene and dynamic state rebuild for the new map; no
   errors; `rt_dynamic = 0` / `rt_characters = 0` again rebuild the scene without the
   reserve.

### 14.7 Reproduction

```
cargo run --release -p void-rt --example dynamic_check -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288 --warp --smoke --moved 1,10,100 --frames 3 --spawn 2
cargo run --release -p void-rt --example dynamic_check -- --game .. \
  --snapshot local/inject/snap_1792183388_f1288
cargo run --release -p void-rt --example skin_check -- --game .. --warp --count 2 --frames 2
cargo run --release -p void-rt --example skin_check -- --game .. --count 12 --frames 20
```

### 14.8 First-person view models and instance types

**Excluded (after in-game run 1 of build 574fbc5, where `gun_emily.md6` and
`gun_emily_accuracy_lvl0.md6` were skinned and traced):** the player's hands and weapons sit
at the camera and would block or shadow wrongly. A model is left out when its
`modelDepthHack` (renderModel +6328, f32) is > 0 (O: the engine's own first-person test,
`ARK_RENDERPASS_SORT` `sub_14033D320`, section 11.1) or, as a backstop, its name is a player
weapon / hands path (`models/weapons/...player...`, `firstperson`, player hands / arms).
Checked in the scene snapshot (static models; counted as "first-person view model" in the
`rt: snapshot` line), when a slot is classified (spawned static and skinned models) and in
every per-frame character read (the flag can switch on when a weapon is drawn: the model
is removed). Each name is logged once (`rt dynamic: first-person view model left out of the
scene: <name> (...)`); the Photo tab counts them. The player entity itself is not
identified (no verified entity offset); depth hack + path cover the view model. U: whether
the third-person player body (if any is registered) has the flag.

**Instance types:** `void_rt::dynamic::TYPE_*` per TLAS instance index: 0 static, 1 moved
static prop (transform differs from the scene's), 2 spawned prop, 3 character
(`InstanceSet::types`, `SceneTlas::types`). The live cut-off TLAS publishes them as
`SceneView::instance_types` (upload heap, u32 per `CommittedInstanceIndex()`, count
`instance_type_count`; None without dynamic objects) - written only while no view holds
that TLAS, like the TLAS. Debug colours (`type_colour`): static grey, moved yellow, spawned
green, character red, miss dark blue: Reflections' `rt_reflect_debug = instances`
(docs/reflections.md 4: types at t8, count in root constant 48, raw rays without the
denoiser; offline-checked by `rt_reflect_offline`) and the photo save's `photo_save_instances` image
(`void_rt::dynamic::InstanceProbe`: `rt.hlsl` `primary` rays from the photo camera through
the frozen photo TLAS, at most 1280 px wide, one blocking submit).
