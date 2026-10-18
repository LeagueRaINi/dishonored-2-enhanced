# Reflections: material data, screen-space reflections, ray-tracing feasibility

Status 2026-10-18 (STATUS E067, E069). Executable `b671f824...6180` (image base 0x140000000),
IDB `local/ida/Dishonored2.exe.i64`. Labels: **O** observed, **H** inferred, **U** unknown.
Builds on lighting-options.md (B3, C3, §D, S2) and fsr4-integration.md (TAA site).

## 1. Material data before TAA (the SSR/SSGI blocker)

Question: does any pass before TAA write per-pixel normal, roughness or specular data that
a reflection pass could read?

**Answer: no (O).** Evidence, three independent sources:

| Source | Observation | Provenance |
| --- | --- | --- |
| Per-view render targets | The complete list of `_<name>_%p` targets created per view (strings 0x141B10B08, 0x141B34218..0x141B35E48; creator `sub_14036D0C0`): back buffer, depth/decompressed depth, `HDRBuffer1/2`, TAA histories, `maskBuffer`, `skyCloudMask_Buffer`, sRGB buffer, `MotionVectors`, `SSAO_ScreenShadows`, `DepthAlphaBuffer`, `DepthWaterBuffer`, downsampled depths, DOF/fog/LUT/luminance buffers, OIT accumulation/opacity, hair buffers, light lists, area lights, distortion vectors, dynamic canvas. No normal, G-buffer, roughness or specular target. A string regex for `gbuffer|normalbuffer|viewnormal|worldnormal|roughness.*buffer|specular.*buffer` finds nothing render-related. | IDB strings + xrefs |
| Pass write lists (FNV-1a hashes resolved against all exe strings) | `FORWARD_WORLD`/`FORWARD_DYNAMIC` (`sub_1403133D0`/`sub_1403133B0` -> `sub_140311AA0`): `hdrBuffer` (cmp 0x140311B40), `zBuffer` (0x140311B72), `sssBuffer` (0x140311BA2), `skyCloudMaskBuffer` (0x140311BD2); render targets set through `sub_1402BC370(ctx, count, rtvs, dsv, flags)` (= `OMSetRenderTargets`, vtable +264) with count 1 or 2 only (call sites 0x140311D21, 0x1403120A6, 0x1403125D0, 0x14031275A, 0x1403127EA, 0x140312A69, 0x140312D4B, 0x140312FF2, 0x140313203). `DEPTHPASS` (`sub_1402FD8D0` -> `sub_1402FDB60` / `sub_1402FCFF0` / `sub_1402FE850`): `zBuffer`, `motionVectors`, `depthAlpha`, `decompressedDepth`, `depthWater`, `depthAlphaDiv2`. `SSAO` (`sub_140305BE0`) / `SCREENSHADOWS` (`sub_14032C7F0`): `ssaoScreenShadows`, `zBuffer`, `decompressedDepth`. `RENDER_WATER` (`sub_1403045A0`): `hdrBuffer`, `depthWater`, `downsampledWaterDepth`, `hdrBufferIn`. `maskBuffer` is used only by `sub_1403134E0` and the distortion/masking post code `sub_1403222F0`. | IDB, cmp immediates listed |
| RenderDoc captures of the original renderer (rooftop f16159, waterfront f6210, alley f7805; `local/rdc/A_original/*/events.tsv`) | Opaque forward draws: one target, 3440x1440 R11G11B10_FLOAT (709 / 728 / 203 draws). Depth prepass: one target R16G16_FLOAT (velocity); 681 of its 730 draws in f16159 bind no texture at all (no material input). Every multi-target draw is: particles (RGBA16F colour + RG16F velocity), skin/translucency (R11G11B10 x2 = `hdrBuffer` + `sssBuffer`), sky (R11G11B10 + R8 `skyCloudMask`), DOF, or an 8 x R16F downsample. | capture dumps, script in this study (`eval`, not kept) |
| Shipped pixel shaders (8,434 DXBC dumps in `d2enhanced/captures/shaders`, read only) | Output signatures: 5,881 one `SV_Target`, 1,550 two, 994 none, 8 depth-only/depth+1, 1 eight-target. The two-target set is 1,075 particle shaders (`ro_fx_*`), 421 lit SSS/translucency shaders (`ro_sss_skinfresnellookup` / `ro_translucency_mask`), 3 sky shaders, 53 others (no lighting inputs). Forward lit (non-particle) shaders: 3,178, of which 141 have no IBL cube sample (corroborates d2enhanced's "141 PS without IBL sample"). | OSGN/RDEF parse of the dumps |

Water: `depthWater` (`_DepthWaterBuffer_%p`, 0x14036DDB8) is written by `DEPTHPASS` and
`RENDER_WATER`; POSTPROC's read list holds `depthWaterStencilView` (0x7A833082, created at
0x1403577AA in `sub_140351780` with format code 22/stencil remap = X24_G8 stencil view of
that texture). **Correction after the first in-game run (user, 2026-10-18; O, IDB):** the
water depth pass `sub_1402FE850` first calls `sub_1402BAFC0(ctx, depthWater, 0)`
(`ClearDepthStencilView`, vtable +424) and then `sub_14034F740(ctx, decompressedDepth,
depthWater)`, which binds `depthWater` as depth target (`sub_1402BC340(ctx, 0, depthWater)`,
viewport = its size) and draws program `qword_143308D28` reading `decompressedDepth`
(parameter `qword_143308D38`): a full-screen **copy of the opaque depth** into `depthWater`;
the water surfaces are then drawn on top (`sub_1402FD8F0` etc.). So `depthWater` equals the
opaque depth everywhere except where water lies in front of it. The first build tested
`water depth >= opaque depth`, which is true on every pixel of a copy: the whole frame was
masked (user: nothing in the debug views unless "Skip water" was off). The water surface in
the RX010 FSR dump depth is therefore **H** (d2enhanced's input, not necessarily
`decompressedDepth`); if the main depth also contained the water, the corrected mask finds no
water and reflections appear on it (visible in the `water` debug view).

### Options evaluated

| | (a) Depth normals + surface model | (b) Extra MRT in the forward shaders (DXBC patching) | (c) Own thin G-buffer pass from the engine's draw lists |
| --- | --- | --- | --- |
| Normal | geometric, from depth (no normal maps) | shading normal only if found per shader by data flow; the IBL lookup direction is parallax-corrected (`ibl_virtualplanes_tCB`), so `normalize(R + V)` is not exact (H) | full (layered normal maps) |
| Roughness / reflectivity | none: one user roughness + F0, optional "upward-facing only" mask | proxy: LOD operand of the first IBL sample (d2enhanced: roughness = LOD/6, unverified, review item 5) for the 3,037 IBL lit shaders; none for 141 | exact if the material port is exact (E020 gloss r 0.888) |
| GPU | 0 extra (normals in the trace) | one more RGBA8 target written by the forward pass, H 0.05-0.15 ms at 3440x1440 | a second opaque raster: E044 measured 17-24 ms unculled for our own raster; with the engine's culled lists H 0.5-1.5 ms |
| CPU (frame is CPU-bound, interval ~9.2 ms) | ~0 (a few dozen D3D11 calls) | none per frame: patch once in `CreatePixelShader` (3,178 shaders) + one extra RTV at the forward pass's ~9 `sub_1402BC370` call sites (per pass, not per draw) | re-issuing ~700-1,450 opaque draws per frame on a deferred context; on AMD the lists are replayed serially at submit (E052): the comparable `DEPTHPASS` submit costs 0.71 ms (E060) -> H +0.7-1.5 ms frame time; plus RE of the draw lists |
| Risk | stylised look; normals at silhouettes | volatile per shader-cache build; semantics approximate; user rated d2enhanced's D1 too volatile | highest CPU cost on the critical path; RE effort |

**Decision (E067): (a) now.** It is the only source with zero CPU cost and no shader
patching, and screen-space reflections pay off mostly on large planar receivers (streets,
floors, tables) where depth normals are exact. The missing per-pixel reflectivity is
replaced by an explicit, user-visible surface model (roughness, F0, floors-only, water
skip) instead of a hidden heuristic. **(b) is the upgrade path** when material-aware
reflections are wanted: a narrow "D1-lite" patch that exports only a per-pixel
reflectivity/roughness proxy (one R8G8 target, slot 2), validated per shader family against
captures; normals stay from depth. **(c) is rejected** on CPU grounds.

## 2. Screen-space reflections at S2 (implemented, offline-verified, not yet run in game)

Code: `void-inject/src/ssr.rs` (+ `ssr/offline.rs`, `ssr/offline/temporal.rs`,
`examples/ssr_offline.rs`); settings `ssr*` (all live, default off); overlay tab
**Reflections** (restore icons on every control); profiler GPU class `ssr`; debug views
`ssr_debug = hits | confidence | reflection | water | material | history | class | f0`.
Coverage classes, ray reuse and temporal accumulation: section 2c.

**Splice (O, fsr4-integration.md):** the TAA dispatch call 0x140327817 in
`PostFX_TemporalAA_and_Post` (POSTPROC), already redirected by `fsr.rs`. `fsr.rs` gained
`install_taa_site()` (the redirect alone, used by `ssr` without FSR) and an `ssr` field in
its job; `taa_dispatch` captures our job first:

- engine TAA (FSR off, or FSR declines the frame): recorded into the POSTPROC deferred
  context right before the TAA dispatch (no hand-off, no fence). Our RTV binding of
  `hdrBuffer` unbinds the TAA program's `viewColorMap` SRV (D3D11 hazard rule); every state
  is put back with `ao::Saved`, CS SRV slots widened to the highest slot the TAA program
  binds (checked per frame);
- FSR on: FSR runs on the immediate context at submit time **before** the POSTPROC list
  executes, so a recording in that list would come after FSR. The job therefore travels with
  the FSR job and runs on the immediate context right before FSR's pack (still D3D11, no
  extra cross-API hand-off). In the scheduler mode without deferred contexts it runs inline.

The TAA site only exists with TAA anti-aliasing (`r_AntiAliasingPreset 3`); with other AA
modes there are no reflections.

**Inputs (O):** POSTPROC write list `hdrBuffer` (0x8F93AB75, = `viewColorMap`), read list
`decompressedDepth` (0x21181874, = `fx/depthFull`) and `depthWaterStencilView` (0x7A833082;
we create our own R24 depth view of that texture: water = water depth **in front of** the
opaque depth by more than 2 R24 steps; the mask is ignored for a frame in which it covers more
than 60 % of the region, counted on the GPU in the pyramid pass and read back a few frames
late on the immediate context for the overlay/log), view projection (+29584),
view matrix (+29776, world Z-up -> up vector in view space), viewport (+30816; the
render-scaled top-left region in FSR upscale mode, the only area read or written).

**Algorithm:** copy of the colour region (radiance and fallback, the same stored units as the
target: no exposure conversion) -> max-depth pyramid of the reversed-Z hardware depth (7
levels, region padded to multiples of 64 so every level halves exactly) -> per trace texel
(half resolution by default; half texel q = full pixel 2q in trace, resolve and validation)
surface model (section 2c), GGX VNDF reflection direction from a 4x4 Hammersley pattern
(rotated per frame with temporal accumulation, fixed without), hierarchical march
(FidelityFX SSSR's min-Z traversal for inverted depth, in pixel units) -> validation
(thickness + 1 % of depth, back faces of the hit, own pixel block, 5 % border fade) written
as `(c * L, c)` plus the ray (direction, pdf) -> resolve: 5x5 tent x tangent-plane depth
weights, each neighbour ray reweighted for this pixel's lobe by the ratio estimator (2c) ->
(`ssr_temporal = 1`) temporal accumulation and roughness-scaled blur (2c) -> blend
`colour = lerp(colour, L / c, strength * EnvSpecular(F0, roughness, N.V) * surface * c)`
(SRC_ALPHA / INV_SRC_ALPHA, RGB only; pixels with weight < 1/1024 are discarded and stay
bit-exact).

d2enhanced review findings and how this pass handles them: (1) one premultiplied-confidence
convention end to end, divided once at the blend; (2) no history, so no camera-space depth
comparison or reset problem (FSR/TAA are the accumulators); (3) units: radiance and target
are the same buffer; (4) hierarchical march with crossing in the pyramid, own-block and
back-face rejection, region clipping; (5) no IBL subtraction at all: misses keep the engine's
full colour (including its IBL), hits replace a Fresnel share of it; (6) one representative
pixel convention; (7) resolve weights only from matching surfaces, no forced epsilon; misses
contribute zero confidence.

**Offline result (M, `ssr_offline`, RX 7900 XTX, median of repeated executions of the
recorded list; synthetic scene: floor, red box, blue wall, water patch):**

| Region | half low | half medium (default) | half high | half ultra | full medium |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3440x1440 native (R11G11B10) | 0.30 | 0.31 | 0.32 | 0.32 | 0.88-0.89 |
| 2304x968 = 0.67 (RGBA16F) | 0.14 | 0.14 | 0.15 | 0.16 | 0.21-0.22 |

ms. Full resolution at 3440x1440 scaled super-linearly and varied between runs (2880x1200
full: 0.30-0.69 ms), so half resolution is the default. All checks passed: box and wall
reflections at the analytic floor points, 0 NaN, 0 pixels written outside the region, walls,
box sides, sky and the water bit-identical, every D3D11 state restored (including the
TAA-like SRVs at CS t2/t6), FSR-path result bit-identical to the engine-path result, nothing
captured with `ssr = 0`; with `ssr_skip_water = 0` the water reflects. Since the water fix the
harness models `depthWater` as the engine fills it (opaque copy + water surface 0.1 m above the
floor), as a plain copy (the in-game failure: reflections must stay on) and as "everything in
front" (coverage 100 %: mask ignored, fallback read back 3/3 frames), checks the read-back
coverage (4.27 % = exact water pixel count) and that "Frames applied" grows on both paths.
Timing note: later runs of the same harness (also of the committed build in a separate
worktree) measured 0.54-0.57 ms native / 0.24-0.37 ms at 0.67 for half medium, against 0.31
/ 0.14 earlier the same day: the machine state changed, not the pass (mask + count add
< 0.05 ms, measured as the difference to `ssr_skip_water = 0`).

**What needs the game (U):** whether `decompressedDepth` excludes the water surface (then the
corrected mask covers exactly the water), the real hdrBuffer format (R11G11B10 per d2enhanced; RGBA16F variant handled), the
TAA program's CS slot use (logged via the widened save), the look on real scenes (depth
normals on stairs/cobbles, characters' feet, first-person hands reflected), GPU ms in the
profiler class `ssr`, any CPU cost on the submit path.

**In-game test of the SSR pass (user run; stop at the first failure):**
1. `void-inject.ini`: `overlay = 1`, `profile = 1`, `ssr = 0`; Anti-aliasing = TAA. Load the
   eotw save, stand at a benchmark pose with street/floor in view.
2. F2 -> Reflections -> tick "Reflections". Expect the toast "Screen-space reflections on",
   log `ssr: TAA call site redirected` (if FSR was off) and `ssr: first TAA site: ...`
   (formats, region, world up in view ~(0, 1, 0) when looking level, water depth format).
   Runtime card: "Frames applied" counts up (engine TAA or before FSR, matching the mode);
   "Skipped" stays at a few "compiling" frames and 0 failed.
3. Debug view "hits": white where rays found screen content (floors only); "confidence":
   final blend weight (water must be black, walls black); "reflection": what is added.
4. Look: buildings/lamps mirrored in the street at grazing angles, no reflections on walls
   or water, no flicker standing still; toggle on/off for A/B screenshots; try roughness 0
   and 0.3, reflectance 0.04 vs 0.2, floors only off.
5. With FSR (`fsr = 1`, native and `fsr_mode = upscale`): same look, "before FSR" counts up,
   nothing outside the render region (no band at right/bottom edge).
6. Profiler (Ctrl+F3) A/B `ssr = 0/1` at the same pose: `gpu_ssr_ms` and frame interval.
   Debug view "water": white = masked water (should be exactly the harbour/canals); red =
   the frame's mask covered > 60 % and was ignored. The "Water mask" row shows the read-back
   coverage. If water still shows reflections (white missing over water), report the
   first-frame info line: the main depth then contains the water surface.

## 2b. Material target from the game's lit shaders (implemented, offline-verified)

User decision 2026-10-18: option (b) of section 1, "the shader patching approach is a good
approach, we did that in d2enhanced as well". Code: `void-inject/src/ssr/patch.rs` (DXBC
patch), `ssr/material.rs` (shader-creation integration, forward-pass binding),
`ssr/patch_offline.rs` + `examples/ssr_patch_offline.rs`; STATUS E072.

**d2enhanced findings reused and checked for this build** (`src/shader_patch.cpp`,
`material.cpp`, `re/NOTES.md` "Material capture v2", read only):

| d2enhanced claim | Here |
| --- | --- |
| Lit forward shaders recognised by `ro_tlf_ll_heads` (3,707; 3,705 patched) | 3,178 of the 8,434 dumps bind `ro_tlf_ll_heads` without particle inputs; 3,037 of them sample the IBL cube array (the set we patch). The other ~530 of d2enhanced's count are particle shaders (`ro_fx_*`) |
| 141 lit shaders without an IBL sample; 2,378 sample it once, 1,188 three times (probe blending) | 141 corroborated; the three samples share one direction and one LOD (`ibl.decl` `HQ_ComputeIBL`), so capturing before the first is exact for N and roughness |
| Roughness = IBL LOD / 6 | **Not corroborated.** `ibl.decl`: `MipLevel = 6 * sqrt((0.000347 + m (1.333129 - 0.504055 m)) / 0.834)` with `m = roughness^2`; LOD / 6 is 0.60 for roughness 0.5. We capture `m` itself (the source operand of the `0.504055` mad) |
| Patch technique: extra `dcl_output`, signature element, `mov`s after samples, MD5-variant checksum | Reused (own Rust implementation; checksum verified against `D3DCompile` output in a unit test). We capture registers before arithmetic instructions instead of after samples |
| Decal forward shaders blend: material targets bound only for opaque draws | Not handled per draw here (see risks) |
| No shading normal captured (depth normals) | We capture the shading normal (after normal mapping) at the reflect instruction |

**Capture rules** (token-level analysis, `patch.rs`; O for every shipped lit shader):
- roughness: from the IBL `sample_l`'s LOD register back through `mul 6`, `sqrt`, `mul
  1.199041`, `mad .., 0.000347` to `mad X, -m, 0.504055, 1.333129` (also vectorised forms);
  `m` copied before that instruction;
- normal: the lerp `mad L, m, A, R` before the sample whose `R` is defined by the reflect
  `mad R, N, -t, -V` (any three-component destination; `-V` a register or folded constant);
  `N` copied before the reflect, swizzle remapped to x, y, z;
- F0 (luminance): after the sample, the IBL luminance-factor lerp `mad X, F0, 1 - |v.c|, |v.c|`
  (`lerp(IBLLuminanceFactor, 1, F0)`); the F0 operand is a register component or a folded
  constant (constant-IOR materials, e.g. 0.046521, 0.027998).
- Output (24 instructions before every `ret`/`retc`, four new temporaries): `o2 = (oct(N) *
  0.5 + 0.5, sqrt(m), m captured ? max(sqrt(F0), 2/255) : 0)` and `o3 = (sqrt(F0.rgb), kind)`
  (below).

**Classes (2026-10-18; in game, `ssr_floors_only = 0` made leaves mirror-like).** Alpha
encoding: 0 = no material (depth-normal fallback), byte 1 = excluded class (SSR writes
nothing, no fallback), >= 2 = material. Signal (`patch::shader`, bytecode only): RDEF binding
names plus cbuffer variable names; the game's cbuffers list every parameter of the render
program, vertex stage included (O: the vegetation pixel shaders carry the `vegetAnim/*` wind
parameters of `arkvegetation.decl`'s vertex program). SV_IsFrontFace and `discard` are no
class signal (O: 486 non-vegetation lit shaders read the face, 866 discard).
- Vegetation (`cb_vegetanim_*`, `cb_vegetation_*`: `arkVegetation` / `BRDF_vegetation`): 61
  dumps, 37 forward lit. **Today's path was (b), patched (O):** all 37 were among the 3,037
  patched shaders; `BRDF_vegetation` calls the same `COMPUTE_IBL`, roughness `1 - lerp(
  cb_wardroughness.xy, glossmap)` (disassembly of `ps_04014dd7685b48b1`), the normal flipped
  by SV_IsFrontFace, F0 from IOR. Mirror-like leaves therefore come from low authored leaf
  roughness (H; the gloss maps and `wardRoughness` values were not read), not from the
  fallback. The other 24 are Z-prepass / shadow / masking programs (diffuse map only).
- Hair (`cb_hair*`, `ro_hair*`: `arkHair` / `BRDF_hair`, IBL roughness = the artist's
  `hairIBL`): 546 dumps, 298 forward: 159 lit (were patched, path b) and 139 hair-BRDF forward
  shaders without the IBL sample (were unpatched: alpha 0, depth fallback, path a); 248 blended
  (`ro_fx_depthfull`) or other passes.
- Decals (`cb_decalboxprojection`, `cb_decalbufferoffset`, `cb_world2decalmatrix*`: `arkDecal`,
  `blend SRC_ALPHA ONE_MINUS_SRC_ALPHA`): 69 dumps, 9 forward lit (were patched). With
  `IndependentBlendEnable` off the decal's blend also applies to our target, so alpha 1/255 as
  the blend factor leaves the underlying surface's material (H) instead of mixing the decal's
  normals in by sqrt(F0).
- Particles (`ro_fx_*`, 3,225 dumps): never patched (as before): additive or premultiplied
  blending would add the marker to the surface underneath.

Patch: forward excluded-class shaders with the IBL captures get the same epilogue with `mov
o2.w, l(1/255)` and `mov o3, l(0)` (17 instructions); without them (139 hair) only `dcl_output
o2/o3`, `mov o2.xyzw, l(0.5, 0.5, 1.0, 1/255)` and `mov o3, l(0)` before every `ret`/`retc`.
`ssr_patch_offline` (M): 3,176
candidates, 3,176 patched, created and unchanged: surface 2,832 material; vegetation 37,
hair 159 + 139, decal 9 excluded; 2 surface forward shaders without IBL left alone. End to
end: vegetation-class test shaders with and without the IBL code read alpha byte 1 at 4,096 /
4,096 pixels, `o0` bit-identical. Overlay status: `material::Status::excluded` (subset of
`patched`). Classifier + patcher: 0.82 s for all 8,434 dumps (0.26 ms per candidate).

**Format: one R8G8B8A8_UNORM target at slot 2** (4 B/px, 19.8 MB at 3440x1440): octahedral
normal in 2 x 8 bit (end-to-end error 0.0024 in the encoding, about 1 degree), perceptual
roughness 8 bit, `sqrt(F0)` 8 bit (dielectric 0.04 -> 51, enough steps where reflections
matter) and "no material" = alpha 0 (the clear value; unpatched shaders, sky). R10G10B10A2
has no room for F0. Slot 2 because the forward pass uses slot 1 for `sssBuffer` /
`skyCloudMaskBuffer` (`sub_140311AA0`).

**Coloured F0: second R8G8B8A8_UNORM target `o3` at slot 3 (2026-10-18, user decision: coloured
F0 for metals).** `o3 = (sqrt(F0.rgb), kind)` where the normal/roughness capture ran, else 0;
kind 1.0 = the shader's F0 vector (or its exact grey constant), 128/255 = luminance only
(rgb = sqrt of the `o2` luminance, grey), 0 = no material / excluded class (all excluded
shaders write `o3 = 0`; under `arkDecal`'s `SRC_ALPHA` blend that leaves the surface's value,
H). Evidence (O, `ibl.decl` + 3,037 lit dumps): `HQ_ComputeIBL` computes `float3 F0 =
Fresnel_F0FromIOR(IOR)` per channel and feeds `dot(lum, F0)` to the luminance-factor lerp;
in **1,947** shaders that is `dp3 X, l(0.2126, 0.7152, 0.0722, 0), F0.xyz` (all after the IBL
sample; F0 operand `rN.xyz` or `rN.yzw`), whose vector operand the patch copies before the
dp3 (`patch::find_f0_rgb`); in the other **1,090** the compiler folded a constant IOR to a
scalar (0.027998 x 900, 0.046521 x 159, 0.025493 x 31) that the shader also uses as the scalar
F0 of the pre-integrated BRDF term (`mad X, brdf.y, l(c), ...`), so grey is exact. **0**
luminance-only: `AQ/LQ_ComputeIBL` (`Fresnel_F0FromIOR(dot(lum, IOR))`) does not occur in the
dumps; the encoding is kept for it. Patch: a fourth temporary (initialised to 0.04 grey), one
`mov` before the dp3 (or at the lerp for constants), 4 more epilogue instructions (`max`,
`sqrt`, two `and`s with the `m captured` mask), `dcl_output o3` and a second OSGN element;
29.3 inserted instructions per shader (was 23.7). Runtime: `material.rs` creates both targets
together (same size and lifetime), clears both at `FORWARD_WORLD`, binds them at slots 2 and 3
(`add_target`) and hands `f0_for_frame` to `ssr.rs` (same freshness rules as `for_frame`).
Offline (M, `ssr_patch_offline`, RX 7900 XTX): 3,176 / 3,176 candidates still patched,
created and unchanged; F0 source counts as above; end to end, an engine-like coloured IOR from
dielectric 0.04 to gold-like (1.0, 0.78, 0.34): `o3` within 0.0022 of sqrt(F0.rgb), alpha 1 at
4,096 / 4,096 pixels, gold decoded as (0.992, 0.772, 0.337), `o0` bit-identical; excluded test
shaders `o3` all 0. Cost at 3440x1440 per full-screen layer of the patched test shader (fastest
of 3 interleaved medians): `o0` only 0.024 ms, `+ o2` 0.031 ms, `+ o2 + o3` 0.035 ms: **o3
costs ~0.004 ms per covered layer** (19.8 MB written per layer; the clear to 0 is a fast clear,
below timer resolution). The test output is smooth and compresses well (DCC); real forward
passes have overdraw and less compressible content, so H: 0.01-0.03 ms per frame for `o3`.

**Offline results (M, `ssr_patch_offline`):** all 8,434 dumped pixel shaders through the
patcher: 3,037 lit forward shaders, **3,037 patched** (all with the F0 capture), 3,037 created
by `CreatePixelShader` on the RX 7900 XTX, and for 3,037 the disassembly minus the inserted
instructions equals the original (the original program is untouched). End-to-end: a pixel
shader compiled from `HQ_ComputeIBL`-like HLSL, patched and drawn next to the original:
`o0` bit-identical, `o2` within 0.0024 (normal), 0.0019 (roughness), 0.0022 (sqrt F0) of the
values the shader used. Patcher time: 0.69 s for all 8,434 dumps in a standalone run (~0.23
ms per lit shader, the rest rejected by the binding check), 0.87-1.4 s inside the harness: about
1 s added to the game's shader creation at load (H).

**Runtime (not run in game):**
- `trace.rs`'s `CreatePixelShader` hook (installed at startup when `ssr = 1` and `ssr_material
  = auto`, or `profile = 1`) creates the patched bytecode for lit shaders, the original on any
  analysis or creation failure (counted in the overlay). Patching is startup-only: shaders the
  game already created cannot be redone. No per-draw shader swap: while our target is not
  bound the extra output is discarded (D3D11), so `ssr_material` (live) only decides whether we
  bind and read the target. Turning the patch off needs a restart with `ssr = 0` or
  `ssr_material = depth`.
- Forward passes (O): MinHook on `sub_140311AA0(pass, dynamic)` (body of `FORWARD_WORLD` /
  `FORWARD_DYNAMIC`, prologue `48 8B C4 88 50 10 ...`) marks the recording thread and, for
  `FORWARD_WORLD`, clears our target on the pass's context; MinHook on `sub_1402BC370`
  (render-target setter, 10 callers, prologue `48 89 5C 24 08 ...`) re-binds the engine's
  targets 0-1 and depth with ours at slot 2 on that thread (`OMGet/SetRenderTargets`). The
  target has the `hdrBuffer` size; the engine's viewport restricts it to the render region.
- Reflections (`ssr.rs`): the TAA site uses the target when it was cleared in this or the
  previous frame-counter value. Per pixel: material normal (world -> view frame) if it agrees
  with the depth normal (dot >= 0.5; decals, silhouettes and blended draws otherwise fall back),
  GGX alpha = roughness^2, F0 from the target, weight fading out between roughness 0.35 and
  0.55 (rough surfaces keep the engine's IBL); pixels without material use depth normals and
  the user model (floors only etc.). `ssr_offline`: rough floor untouched, the smooth F0 0.25
  floor reflects the box at 0.91 red instead of 0.32 with the user model, `ssr_material = depth`
  bit-identical to the user model, the forward binding keeps targets 0-1 and puts ours at slot 2.

**Risks / needs the game (U):** blended lit draws (decals): with `IndependentBlendEnable`
off their blend state also applies to our target (d2enhanced bound its targets only for
opaque draws; `arkDecal` now writes alpha 1/255, see Classes; other blended lit draws remain,
the normal-agreement check limits the damage); the order `FORWARD_WORLD`
before `FORWARD_DYNAMIC` on the GPU (H, LO1; otherwise dynamic objects lose their material
data); load-time cost; the forward pass's extra target bandwidth (H: 19.8 MB, ~0.02-0.05 ms;
profiler class `opaque_lit`).

**In-game test (after the SSR steps of section 2):**
1. `void-inject.ini`: `ssr = 1`, `ssr_material = auto` **before starting the game**. Log:
   `ssr: lit forward pixel shaders get the material output (o2) at creation`, `ssr:
   forward-pass hooks installed (material targets at slots 2 and 3)`, then `ssr: material
   targets WxH (slots 2 and 3)`.
2. Reflections tab: "Material target" shows thousands of shaders patched, few unpatched, 0
   rejected, "bound" counting up.
3. Debug view `material`: normals as red/green (octahedral), roughness as blue; black = no
   material (sky, characters with other shaders). Floors and walls must look like smooth
   normal maps, not noise; decals should not stand out as garbage.
4. A/B `ssr_material = auto` vs `depth` (live): polished floors, metal and wet stone reflect
   more, carpets/rough plaster stop reflecting; normal-map detail visible in reflections.
5. Profiler: `gpu_opaque_lit_ms` with and without `ssr = 1` at startup (patched shaders + extra
   target), loading time of the save.

## 2c. Coverage classes, ray reuse, temporal accumulation (implemented, offline-verified)

In-game feedback on run 2 (E074; material patch active, FSR upscale 0.67, `ssr_quality =
ultra`, `ssr_half_res = 0`, `ssr_strength = 1.42`): "works nicely, but noisy even on the
highest settings; why only floors; can we detect metal"; with floors-only off, foliage
reflected like a mirror.

**Coverage (O = code, M = offline).** Floors-only applied to every pixel through the depth
model's up-vector test only for pixels *without* material; material pixels already ignored it
but weighed by plain Schlick, so dielectric walls (F0 0.04 head-on) show ~4 % and the
screen-space ray of a wall mostly leaves the screen: walls look unreflective by design, metal
(F0 = luminance of `lerp(0.04, albedo, metal)` from the patch) reflects strongly. Changes:

- `ssr_floors_only` is now `ssr_depth_floors_only` (default 1): it applies only to the
  depth-normal fallback; material pixels reflect on any orientation by their own roughness and
  F0. A shading normal that disagrees with the geometry (> 60°) now takes the depth normal
  but keeps the material's roughness/F0 (before: the whole user model, i.e. a mirror).
- Depth fallback only where the hardware depth is planar (second differences at 1 and 2 px
  within 0.1 % + 4 R24 steps; reversed-Z depth is affine over a plane): noisy depth (foliage,
  clutter) never takes the user model.
- Material classes (patch side, section 2b "Classes"): vegetation, hair and decal shaders
  write the exclusion marker (alpha byte 1); SSR writes nothing there. Vegetation was case (b)
  (patched lit shaders reporting the artists' low leaf roughness), not the fallback.
- Fresnel: the split-sum fit `EnvSpecular(F0, roughness, N.V)` (Karis 2014) replaces Schlick:
  same at roughness 0, lower at grazing angles for rough surfaces, i.e. the share of the
  engine's own pre-integrated IBL that the reflection replaces.
- Debug view `class`: black none/sky, magenta excluded, grey no material + rejected, blue no
  material + reflecting, brown lit but too rough, green lit + reflecting, cyan water.

**Coloured F0 (E076, user decision "properly resolving reflected colours").** Option (a),
capturing the shader's own F0 vector, chosen over (b) metalness + engine chroma because it
is exact where it exists (O, section 2b "Coloured F0": 1,947 lit shaders feed
`dot(lum, F0.rgb)`, 1,090 use a folded grey IOR constant). The patch writes a second RGBA8
target o3 = sqrt(F0.rgb) (slot 3, alpha 1 captured / 0.5 luminance only / 0 none). SSR reads
it at t12 (`g_trace.w & 8`), evaluates the split-sum reflectance per channel and weighs with
the largest channel; the per-pixel tint `F.rgb / max(F)` (region RGBA8, written by the
resolve for every pixel, not accumulated: deterministic per pixel) goes to the composite,
which now uses dual-source blending (`ONE / INV_SRC1_COLOR`, RGB only):
`colour = colour x (1 - w.rgb) + L x w.rgb`, `w = weight x tint`; grey F0 gives the old
result. Debug view `f0`: sqrt(F0.rgb) (black none, blue-tinted luminance only). Offline (M,
`ssr_offline`, polished panels at roughness 0.08 reflecting the blue wall): per-channel weight
r:g:b = 2.62:2.07:1 gold (F0 1.0/0.78/0.34; split-sum at that angle 2.55:2.03), 1.70:1.17:1
copper (0.95/0.64/0.54), 1.00:1.00:1 polished dielectric (0.04). Cost: the second target adds
~0.004 ms per covered full-screen layer at 3440x1440 (19.8 MB written + one fast clear per
frame; patch side, section 2b); the tint texture and per-channel math in the resolve were within
the run-to-run spread of this machine (half medium 0.33-0.40 ms without temporal).

Strength (H): with the split-sum weight `ssr_strength = 1` is the physical value. The
composite replaces `w x` the engine colour (diffuse + specular), not only its specular IBL, so
on bright diffuse surfaces the result is darker than physical, and the 5x5 resolve spreads
misses into hits (lower confidence near reflection edges); both read as "too weak". The
default stays 1.0 until in-game A/B with the temporal stage (more confidence where the
single frame had misses).

**Ray reuse (SSSR resolve).** The trace writes each ray's view-frame direction and VNDF pdf
(`G1(V) D(H) / 4 N.V`); the resolve weighs neighbour rays by this pixel's `D G2 / (4 N.V) /
pdf_k` (ratio estimator, Stachowiak 2015) times the tent and plane weights; near-mirror pixels
(alpha < 0.0025) keep tent x plane only.

**Temporal (`ssr_temporal`, default 1, needs the engine's `motionVectors`, read at the TAA
site via `fsr::motion_vectors`).** Sample pattern rotated per frame (R2 Cranley-Patterson on
the 4x4 Hammersley set) -> resolve -> reprojection with the motion vectors (current - previous
viewport UV; bilinear taps that pass a 5 % + 2 cm depth test and a world-normal test) -> clamp
of the history to the current 3x3 YCoCg mean +- k sigma (k 1 at mirror .. 2.5 at roughness
0.35, back to 1 while the surface moves 4 px/frame) and of its confidence to the current
maximum -> blend 1/N, N up to 4 (mirror) .. 32 (roughness >= 0.35), divided by
(1 + 0.5 x motion in px): reflections move with the reflected scene, not the surface ->
3x3 bilateral blur at a stride growing with roughness (doubled while N < 4). History
ping-pong in region-sized RGBA16F (accumulation `(L c, c)`), RGBA8 (world normal, N,
roughness) and R16F (linear depth); restarts on a frame gap, resize or format change (counter
"Temporal history" in the overlay). Before FSR: FSR still accumulates on top.
Debug view `history`: N per pixel (black 0 -> white 32).

**Offline (M, `ssr_offline`, scaled case 2304x968 in 3440x1440, RGBA16F, half medium):**

| Check | Result |
| --- | --- |
| Vertical metal panel (roughness 0.08, F0 0.9) | 30,959 / 32,448 px reflect (blue wall: +0.66), with and without `ssr_depth_floors_only` |
| Rough half of the panel (0.8) / excluded "leaves" / no-material noisy-depth "bush" | 0 / 15,443, 0 / 357, 0 / 34,698 changed (also with floors-only off) |
| Noise, glossy floor (roughness 0.3, F0 0.3), RMSE vs 64-frame mean of the rotated per-frame estimate | before (single frame, fixed pattern) 0.0163; temporal frame 1 0.0114; after 40 frames 0.0040 (4.1x lower) |
| Ghosting, camera 25 frames x 4 cm along x | RMSE moving 0.0124 (static accumulated 0.0052, no temporal 0.0162); residual of the content that moved: 2 % of the change |
| GPU half medium | 0.33 ms without temporal, 0.47 ms with (+0.13 ms: temporal + blur at 2304x968) |

All previous checks unchanged (FSR path = engine path without motion vectors, state restored,
region-only writes, water mask, material target).

**What needs the game (U):** motion-vector format/size at the TAA site (logged once if the
temporal stage cannot use them), smearing of mirror-like reflections while strafing (history
capped at 4 frames there), the class view on foliage/hair/decals, strength A/B.

**In-game steps:** Reflections tab -> Temporal accumulation on (default); compare
on/off standing still and strafing at a glossy floor; debug view `history` (mostly white when
still, darker on moving surfaces), `class` (leaves/hair magenta, walls with material green or
brown, no blue on foliage); `Floors only (no material)` off: walls with material reflect, no
mirror foliage; debug view `f0`: gold/brass/copper fixtures yellow-orange, stone/wood dark grey
(F0 0.03-0.05), nothing blue-tinted (would mean luminance-only shaders); polished metal now
reflects tinted; report `ssr_strength` that looks right.

## 2d. Temporal stability, first-person weapon, silhouette halo (E083, offline-verified)

User (build d640e97, FSR upscale 0.67, `ssr_temporal = 1`): "quite noisy / unstable even
standing still, especially on the player's sword"; "a faint glowy outline on some things".
The FSR motion-vector view, standing still, shows motion only on the first-person hands/sword.

**View model (O):** player models are drawn with the scene's view-projection and clip-space
depth times `$modelDepthHack.x` (`vertex.decl`: `_projPosition.z *= $modelDepthHack.x`, also
for the previous-frame position; per-entity value via `sub_140227C70` -> parm
`qword_1422B1A18`, cvar `r_playerDepthHack`; `unhackdepthbuffer.decl` / `screenshadows.decl`
undo it with the reciprocal). So the weapon's x/y projection equals the scene's, its stored
depth reads closer by a constant factor (positions scaled toward the camera: depth normals
keep their direction), and its motion vectors are true object motion (idle sway). Our
reprojection already uses the engine's motion vectors, so the weapon's history is valid but
short (its surface moves every frame).

**Causes of the flicker (M, offline):** (1) the clamp box: the 3x3 spread of the resolve output
collapses (the 5x5 resolve makes it spatially smooth) while the value still changes from frame
to frame, so the history was clamped onto every frame's noise (glossy floor: history 58 frames
deep and still flickering 0.0032; with a noise floor 0.0006); (2) history capped at 4 frames on
near-mirrors and ~12 on the swaying weapon; (3) the per-frame rotated sample pattern also on
pixels with no usable history, where the fixed pattern is stable.

**Changes (`ssr.rs`):** clamp sigma floored at 5 % of the luminance; box k standing still
3 (mirror) .. 12 (roughness >= 0.35), moving 1.25 .. 2 (blend by surface motion over 2 px);
confidence clamp relaxed for still glossy pixels; history length 24 (mirror) .. 64 (roughness
>= 0.35) frames divided by (1 + surface motion in px); pattern rotation only where the previous
history at the pixel holds >= 4 frames (previous meta bound at t8 in the trace).

**Halo (O, offline repro):** since E074 a material pixel whose shading normal disagrees with the
depth normal (> 60 deg) used the depth normal with the material's roughness/F0. On 1-2 px
silhouettes (thin poles, wires, edges against the background) the depth normal is the step to
the background, so the pixel reflected along a garbage direction with metal F0: a bright
outline. Now such pixels reflect only where the depth is planar (normal-mapped detail; shading
normal kept), else not at all. Independent of the vanilla DOF-sharpening halo (E082).

**Offline (M, `ssr_offline`, 2304x968 RGBA16F, half medium, static camera, frames 72..96):**

| Frame-to-frame RMSE of the reflection | temporal off (fixed pattern) | E074 | now |
| --- | ---: | ---: | ---: |
| mirror floor (roughness 0.05) | 0 | 0.00495 (history 4) | 0.00081 (history 30) |
| glossy floor (0.3) | 0 | 0.00463 (history 28) | 0.00059 (history 59) |
| swaying weapon stand-in (0.15, F0 0.6, depth hack 2, own MVs) | 0.00022 | 0.00176 (history 12) | 0.00062 (history 29) |

Silhouette halo (1-2 px polished pole facing the camera in front of the bright wall): E074 63
of 484 pole pixels brightened > 0.01 (mean +0.022), now 0. Convergence (glossy RMSE vs the
64-frame reference after 40 frames) 0.0040 -> 0.0035; strafing 25 x 4 cm: RMSE 0.0146 (no
temporal 0.0161), trail residual 2 %; GPU half medium 0.26 ms without / 0.37 ms with temporal.

**In-game steps:** standing still at a glossy floor and looking at the sword: A/B
`ssr_temporal` 0/1; debug view `history` (static floor white, the swaying weapon mid-grey);
thin bright-backed objects (poles, railings, wires, window frames) without a light rim with
reflections on vs off.

## 3. Ray tracing feasibility (doc only)

**Geometry available today (O, void-formats / void-assets):** `.bwm` world
(`bwm::WorldModel`: float3 positions, u16 surface-local indices, 3x4 instance transforms,
per-batch materials; all 36 shipped worlds parse, E014), static entity models
(`bmodel::StaticModel`, one 32-byte vertex layout, 417 models, E022), entity placement
(`entities::*`, 3x4 spawn transforms, E021/E023), layered arkDefault materials on the CPU
(`arkmat`, `void-assets::material`: three layers, gloss/metal ranges, BC5 normals; E020,
E034), images (`bim`, BC payloads kept native), probes (PNW/PRB, SH and relight samples,
E036/E037), Havok collision (E026-E028). eotw high chaos: 4.90 M unique triangles, 5.15 M
vertices, 94,111 world instances, 1,131 materials (`local/m2/bwm_all.tsv`); E044's assembled
scene 16,829 draws / 99,421 instances. Packed vertex streams ~165 MB + ~29 MB u16 indices
before instancing (H). **Missing:** skinned `.md6` characters and hands (no decoder),
vegetation wind, projected decals 1-3, LOD/visibility selection, live physics props.
**Removed in cleanup 190c284 (from checkpoint 9282529, 64 files, -22,752 lines):** scene
assembly (`void-inject/src/assets.rs`, viewer `entity_scene.rs`), the D3D12 raster renderer
with bindless BC textures and the HLSL arkDefault port (`gpu.rs`), live request worker
(`r3.rs`), `r3a_snapshot`. No DXR code ever existed: `git grep -i
"RayQuery|BuildRaytracingAccelerationStructure|DispatchRays|RAYTRACING_TIER" 9282529` finds
nothing (O).

**Acceleration structures (H unless noted):** one BLAS per unique `.bwm` surface / `.bmodel`
surface (static, built once per map load, compacted; memory H 150-350 MB for eotw at
~30-70 B/triangle), TLAS of ~100k instances (64 B each, 6.4 MB) rebuilt or refitted per frame
(H 0.2-0.5 ms GPU) or only when instances change. Alpha-tested vegetation needs any-hit or
opacity micro-maps (DXR 1.1 inline: any-hit via candidate loop). Map changes rebuild (E033
map detection exists).

**Keeping in sync with the engine:** static world: never changes. Entities: E044 measured a
per-frame walk of all models at 47 ms on the game thread (O), so sync must be incremental:
hook the engine's render-entity update (U: function not identified) and patch TLAS instance
transforms for the few that moved (doors, physics props). Skinned characters: absent ->
hybrid: screen-space reflections (this pass) for on-screen characters, RT for static world
only, or decode md6 + GPU skinning + BLAS refit per frame (L on its own). Wind animation:
static foliage in reflections (acceptable).

**Hit shading:** RT reflections need the full lighting at the hit point: arkDefault layered
material (CPU resolver exists, HLSL port in 9282529), sun + shadow ray, local lights
(E035 light struct), probe SH (parsers exist), fog. Two lighting models side by side; E037
residual ratios 1.2-1.5 against the engine (O) are visible in mirrors. Probe-only shading of
hits (no direct lights) is a cheaper first step for rough reflections.

**Interop with the D3D11 frame:** DXR is D3D12 only. Composited at S2, RT reflections can
share FSR's existing hand-off (copy + Signal 0.2-0.5 ms CPU p50 on the submit path, E058 /
E060, O); without FSR they add that cost again. Inputs: depth + colour copies (as FSR), output
one RGBA16F texture blended like this SSR (D3D11 side). A D3D11-created shared texture must be
used (E047/E048 rules).

**GPU budget (H):** at 0.67 scale (2.23 M pixels) with floors-only gating (~30-50 % of
pixels), 1 ray per pixel against a 5 M-triangle scene on RDNA3: H 1-2 ms traversal + 1-3 ms
hit shading; the frame has ~2-3 ms GPU headroom while CPU-bound (performance-baseline.md).
Native 3440x1440: about 2.2x that. Only with FSR upscale.

**Denoising:** FSR Ray Regeneration is not available on RDNA3 (O, NO_PROVIDER; SDK: "requires
an AMD Radeon RX 9000 Series GPU or later", `denoising.md:1121`). Own: for low roughness 1
ray/pixel + this pass's fixed-pattern spatial resolve is enough (mirror-like); rough
reflections need temporal accumulation with hit-point (virtual) reprojection and history
validation (d2enhanced review items 2/3 apply) -> M.

**Radiance cache 0.9.0:** available here (default backend OK, WMMA forced fails, O). It is
trained online from a Monte Carlo path tracer's world-space samples (position, normal, view
direction, albedo, roughness -> radiance; inference over up to all pixels, training 1-5 %;
CS 6.6) (O, `radiance-cache.md`). Role: terminate secondary rays of rough RT reflections / GI
with a cache query instead of full shading. It learns our tracer's lighting, not the engine's,
so it does not reduce the two-lighting-model risk; useful only once a path-traced GI (C4) is a
goal. Not useful for this reflection work.

**Effort (S/M/L):** BLAS/TLAS of the static scene + RayQuery reflections with probe-only hit
shading, static only: **L**. Full hit shading matching the engine (lights, shadows, fog): **L+**.
Characters (md6 + skinning): **L** extra. Own temporal denoiser for rough reflections: **M**.

**Recommended first spike (S-M, offline, no game):** a standalone D3D12 example that checks
`D3D12_FEATURE_D3D12_OPTIONS5.RaytracingTier` on this driver, builds compacted BLASes for
the eotw high-chaos `.bwm` surfaces + one TLAS from its instances (world only), and traces 1
reflection ray per pixel from the RX010 snapshot camera (`local/r3a_snapshot`) using
depth-reconstructed normals; output hit distance / instance id and timings. Gates: BLAS
memory < 0.5 GB, build < 2 s at map load, rays < 1 ms at 2304x968, hit agreement with this
SSR where SSR hits > 90 %. Only then decide on hit shading and entity sync.

## 4. Hybrid ray-traced reflections (implemented, offline-verified, not run in game)

Code: `void-inject/src/ssr/rt_reflect.rs` + `rt_reflect.hlsl` (cs_6_5, DXIL precompiled by
`crates/void-inject/build.rs` with the Windows SDK `dxc.exe`/`dxil.dll` or `VOID_DXC_DIR`),
`ssr.rs` `cs_rt_classify`, one call in `fsr.rs` `dispatch_fsr`; offline
`ssr/rt_reflect/offline.rs` + `examples/rt_reflect_offline.rs [--warp] [WxH]` and the
classify check in `ssr/offline/temporal.rs`; settings `rt_reflections` (default 0),
`rt_max_roughness` (0.5), `rt_ray_budget` (100 %), `rt_reflect_debug = off | rays | hits |
rt`; overlay Reflections tab card "Ray-traced reflections"; STATUS E077. Approach: Far Cry 6 /
AMD hybrid reflections: screen space first, rays only where it found nothing.

**Where it runs (O, code):** needs `ssr = 1`, `rt = 1` (the scene service of E075: per-map
BLAS, camera-radius TLAS on the per-adapter D3D12 device that FSR uses too) and FSR. On the
FSR path the SSR job runs on the immediate context before FSR's pack; after the SSR result,
`cs_rt_classify` writes, per region pixel, the weight SSR could not fill (`max(W - filled, 0)`,
W = strength x split-sum reflectance x surface; faded by roughness from 80 % of
`rt_max_roughness` and by camera distance toward the BVH radius; rgb tinted by the coloured
F0 like the SSR composite) and per half-resolution texel one ray request (biased view-frame
origin + TMin `max(5 cm, 0.005 z)`, shading normal + GGX alpha; the direction is drawn on the D3D12 side, 4b)
into three D3D11-created textures shared with D3D12. The pack's signal and FSR's D3D12 wait
order them; FSR's own list records our work after its wait and before the FSR dispatch, so
the result is in FSR's colour input when FSR reads it, and FSR accumulates it.
Without FSR there is no D3D12 point before the engine TAA: no ray-traced reflections (a
separate D3D12 submission around the TAA would need its own cross-API fence pair on the
critical path; not done).

**D3D12 work per frame (5 dispatches, one ExecuteIndirect; + 3 denoiser dispatches, 4b):** counters reset ->
compaction of the requests into a ray list (one atomic per wave), results "not traced" ->
ray budget and indirect arguments (an even subset of the list beyond the budget, offset per
frame) and the hit-shading calibration -> trace (`RayQuery`, FORCE_OPAQUE + back-face
culling, TMax = the TLAS sphere exit; closest hit: on screen at matching depth = the frame's
own colour (exact, stored units) and a calibration sample, else FC6 per-primitive albedo /
normal / metallic with sun (`rt::Lighting`, shadow ray) + ambient in absolute units times the
stored/absolute scale (per screen tile since E085, 4b; start 3.58, E073); confidence fades
toward the radius (around the camera since E085, 4b); misses keep the engine IBL) -> composite (3x3 half texels, tent x depth similarity,
premultiplied confidence, per-channel lerp). Inputs are COMMON before and after (explicit
transitions); a ring of 4 descriptor/timestamp/keep-alive sets bound by FSR's fence: a frame
whose slot is still in flight is skipped, never waited for; the `SceneView` (TLAS, buffers)
is held until that fence passed. GPU time per frame from D3D12 timestamps in the overlay.

**Offline (M, `rt_reflect_offline`, synthetic scene: floor, red box, blue wall, gold floor
patch; 1 mirror ray per half texel on the floor):**

| Device / size | requests = CPU | hits = analytic | box / wall in reflection | gold r:g:b | GPU |
| --- | --- | --- | --- | --- | --- |
| RX 7900 XTX 1280x540 | 71,986 | 31,562 = 31,562 | red 0.415 / blue 0.419 | 2.95:2.31:1 (exp. 2.94:2.29) | 0.035 ms |
| RX 7900 XTX 3440x1440 | 516,970 | 224,603 (analytic 224,724) | passed | passed | 0.214 ms |
| WARP 256x144 (smoke) | 3,858 | 2,295 = 2,295 | red 0.405 / blue 0.418 | 2.95:2.31:1 | 20 ms |

Also checked: pixels without weight bit-identical, no NaN, budget 25 % traces exactly the
cap and still fills the reflection, shaded hits only (no screen colour) give a red box
reflection after the calibration (scale 3.58 -> 3.05), the three debug views. The D3D11
classify pass (ssr_offline, 2304x968 coverage scene): weights on floor/gold/copper/dielectric,
none on the rough panel and excluded leaves, gold weights tinted 2.57:2.04:1, 44,554
requests all from their pixels with unit directions, 41,048 of 41,093 floor requests mirror
(roughness 0.05). The synthetic scene's cost says nothing about the game's BVH (spike E070:
mirror rays at all pixels 1.53 ms at 2304x968 against the 5,858-instance cut-off TLAS).

**Needs the game (U):** units of the view-matrix translation vs the TLAS (the camera is
checked against the TLAS centre each frame; a mismatch is logged once and the frame
skipped), winding of the engine's front faces under back-face culling (E070 measured
FRONT_COUNTERCLOCKWISE), lighting scale of shaded hits (calibration against on-screen hits),
cost with the real TLAS, look with FSR accumulation.

**Prerequisite feedback:** with `rt_reflections = 1` the card shows "Not active: needs ..." with
the first missing item (Reflections, the ray-tracing scene `rt`, FSR running, the D3D12 device
or DXR 1.1, the scene failed / not built / building, the camera outside the scene) and buttons
that turn `ssr`, `rt` or `fsr` on; a toast "Ray-traced reflections need: ..." once per new
reason (scene progress only in the log).

**Photo accumulation (for photo mode):** `rt_reflect::set_photo(Some(spp))` switches the
FSR-path frames to `Mode::Photo`: every requested texel traces `spp` GGX rays per frame (no list,
no budget) into a persistent RGBA32F sum, and the composite blends its running average;
`reset_photo()` restarts it (camera moved), `set_photo(None)` returns to live mode,
`photo_samples()` reports the samples per texel. Requests carry the shading normal and GGX
alpha; the directions are drawn on the D3D12 side from the sample index (4x4 Hammersley in
Bayer order rotated by R2), so live (index = frame) and photo (index = running count) share
the sampler. Offline: 4 x 8 spp on mirror requests equals the live result exactly and repeats
bit-identically.

**In-game steps:** `ssr = 1`, `rt = 1`, FSR on (native or upscale); Reflections tab ->
Ray-traced reflections on. The card shows frames traced, rays requested/traced/hits (on
screen), the hit-shading scale and GPU ms; the log line `rt_reflect: camera ... from the TLAS
centre` means a unit mismatch. Debug view `rays` (red = ray requested, green = weight),
`hits` (traced reflection, dark blue = miss), `rt` (only the ray-traced contribution),
`history` (4b), `instances` (each traced ray's hit coloured by the instance type of
`rt::SceneView::instance_types`, docs/raytracing.md 14: grey static, yellow moved static prop,
green spawned prop, red character, dark blue miss; the denoiser is skipped; offline: box set to
moved reflects yellow, wall set to character red). Look at a glossy floor or metal facing away
from on-screen content (things behind the camera should now reflect), A/B with the toggle, try
budget 25 %.

## 4b. Ray-traced reflections: denoiser and stable hit shading (E085, offline-verified)

User (after E083): "the normal reflections look superb now (stabilisation worked); what's
still very noisy and unstable are the ray-traced ones". Cause (O, code): the live rays were 1
GGX sample per half texel, rotated every frame, composited through a 3x3 tent only; FSR alone
had to average them. Hit shading switched hard between the frame colour (hit on screen at
matching depth) and our shading, used one global scale updated by 10 % of a noisy ratio per
frame, and the BVH confidence faded around the TLAS centre, which jumps by 0.25 x radius at
every TLAS rebuild.

**Denoiser (D3D12, after the trace, half resolution; same parameters as the SSR, ssr.rs
`cs_resolve` / `cs_temporal` / `cs_blur`; runs with `ssr_temporal = 1` and motion vectors):**

- `rt_trace` also stores each ray (view-frame direction, pdf). The sample pattern rotates per
  frame only where the previous accumulation holds >= 4 frames (E083 rule), so first frames,
  disocclusions and the swaying weapon keep the fixed, stable pattern.
- `rt_resolve`: 5x5 ratio-estimator reuse (weight = this texel's GGX BRDF x cosine / the pdf
  the neighbour's ray was drawn with, x tent x tangent-plane weight; mirrors 3x3 tent + plane).
  Texels left untraced by the ray budget take their neighbours' rays.
- `rt_temporal`: reprojection with FSR's packed engine motion vectors (the same `motionVectors`
  the SSR uses, so the first-person weapon's sway is handled as in E083), bilinear taps with
  depth and world-normal tests, YCoCg clamp of the history to this frame's resolved 3x3 (sigma
  floored at 5 % of the luminance; k 3..12 standing still, 1.25..2 moving), confidence clamp,
  history length 24 (mirror) .. 64 (roughness >= 0.35) / (1 + motion px). A requested texel
  without a ray this frame keeps its history; a texel without a request carries its
  reprojected history forward unchanged.
- `rt_blur`: 3x3 bilateral blur at a stride growing with roughness (doubled while the history
  is short), plane and normal weights; the composite upsamples it as before.

**Joint SSR + ray convention:** both sides keep premultiplied (L c, c) histories. The SSR
history's c is its blend weight; the classify pass hands the rays `W - filled` from the
temporally accumulated SSR result, and the ray history's c is the hit confidence alone,
independent of that weight. Where SSR covers a pixel only on some frames, only the split
between two individually stable estimates moves: the ray history survives the frames without
a request (carried forward), and on-screen hits return the same frame colour SSR uses. The
classify pass takes the request's surface from the first pixel of the 2x2 that reflects at all
(no longer the first that still wants rays), so it does not hop within the 2x2 as the SSR
coverage changes. A literal single history across both is not possible: the SSR result is
composited on D3D11 before FSR's pack, the rays run after it on D3D12.

**Hit shading:**

- On-screen weight `ws = (1 - smoothstep(0.5, 1, depth error / tolerance)) x saturate(px to
  the region border / 16)` blends the frame colour over our shading instead of switching.
- Calibration per screen tile (8 x 4, by the reflecting pixel): sums of on-screen colour and
  our absolute shading of fully on-screen hits (every 16th texel). Each tile's scale is
  smoothed in the log domain (5 % per frame), clamped to 1/4..4 x the global scale, and
  relaxes toward the global one without samples (2 % per frame). The global scale is the ratio
  over all tiles, smoothed the same way. Both jump to their first estimate and stay unset
  before it, so there is no start-up drift. Shaded hits use the scale interpolated
  bilinearly between tile centres (no tile edges).
- BVH confidence fades from 0.5 to 0.7 x radius around the camera. The TLAS holds every
  instance within the radius of its centre and is rebuilt once the camera moved 0.25 x radius
  (rt/scene.rs), so this sphere is always covered and moves continuously. The classify
  pass's distance fade matches it.

**Offline (M, `rt_reflect_offline`, RX 7900 XTX 1280x540; camera pitched 20 deg down so the
12 m wall rises above the frame and the floor reflects its off-screen part; weight 0.8 on the
whole floor = all ray-traced; 48 frames settle + 24 measured; engine-like jitter, Halton 2,3
over 8 frames). Before = E077 path (offline flag `F_LEGACY`), after = this build:**

| Metric (reflection term, output - input) | before | after |
| --- | ---: | ---: |
| Mirror floor, jitter: frame-to-frame RMSE (502,056 px) | 0.00573 | 0.00271 |
| Mirror floor, off-screen hits only (108,236 px) | 0.00942 | 0.00278 |
| Mirror floor, no jitter | 0.00000 | 0.00000 |
| Glossy floor (roughness 0.3), jitter | 0.03478 | 0.00246 |
| Glossy, ray budget 25 % | 0.04921 | 0.00247 |
| Hand-off band (SSR covers it every other frame), RMSE between frames needing rays | 0.02668 | 0.00021 |
| Glossy, RMSE vs 256-sample photo reference | 0.02156 (1 frame) | 0.00211 (40 frames) |
| GPU per frame (D3D12 timestamps, glossy) | 0.054 ms | 0.123 ms |

- The remaining mirror change with jitter is the jitter itself: a mirror ray is deterministic,
  and without jitter both paths are exactly steady.
- Strafing 25 frames x 4 cm (motion vectors from the camera motion): RMSE at the final pose
  against its settled result is 0.0066 with the denoiser and 0.0193 without. The old pose's
  reflection left in the moving result is -1.5 % of the change (no denoiser -0.2 %), so there
  is no measurable trail.
- No NaN in any sequence.
- The WARP smoke test (256x144, 16 + 8 frames) passes the same checks: glossy 0.0497 ->
  0.0051, hand-off 0.0443 -> 0.0016.
- The main RT check is unchanged: box/wall/gold, budget, photo = live within 0.0002.

**Needs the game (U):** cost on the real TLAS (the three extra passes are half resolution,
memory-bound: about +0.07 ms here), and whether the tile calibration helps where interiors
and exteriors mix. The motion vectors are FSR's packed copy of the engine's (O, code), which
are the same ones the SSR temporal stage reads.

**In-game steps:**

1. Set `ssr = 1`, `ssr_temporal = 1`, `rt = 1`, `rt_reflections = 1`, FSR on.
2. Check the card. "Denoiser: on" and "Hit shading scale ..., screen tiles a..b" should
   show.
3. Stand still at a glossy floor or metal reflecting things off screen (behind the camera,
   above the frame). The ray-traced part should be as steady as the SSR. A/B
   `ssr_temporal` 0/1 (0 = the rays as traced).
4. Switch to debug view `history` (grey = ray history length / 64, red = no history yet). A
   still floor should turn bright; the swaying weapon and disocclusions should stay dark.
5. Strafe and turn: no trails behind reflected objects.
6. Walk across a TLAS rebuild (more than 10 m at `rt_radius = 40`). Far reflections should
   not pop.
7. Try `rt_ray_budget = 25`. It should still be steady.

## 4c. Light shimmer (fireflies) and the prerequisite toast (E089, offline-verified)

User (after E085): "light reflections still shimmer a bit". A second report: a warning
"ray-traced reflections need ..." popped up whenever Reflections was switched off, although
ray tracing was on.

**Cause of the shimmer (H, from the offline reproduction):** a glossy pixel reflecting a
small bright light gets a ray that hits the light only on some frames. That hit is far
brighter than its neighbours. The temporal clamp box is built from this frame's resolved
neighbourhood, so a frame whose rays missed the light shrinks the box and pulls the
accumulated highlight down. The next hit spikes it back up. The highlight flickers, and it
loses part of its energy along the way.

**Change (SSR `ssr.rs` and rays `rt_reflect.hlsl`, same rules on both):**

- Resolve: on glossy pixels (the ratio-estimator path) each hit is clamped to `FIREFLY_K`
  times the mean luminance of the hits around it (5x5, same plane). Mirror pixels are
  unchanged.
- Temporal: a second moment of the luminance (R32F ping-pong: SSR `mom`, rays `Moment` /
  `MomentPrev`) is accumulated with the history. The luminance axis of the YCoCg history
  clamp is widened to the pixel's own frame-to-frame deviation, `sigma.x = max(sigma.x,
  sqrt(m2 - Y^2))`. A frame without a hit then no longer pulls the highlight down.
- `FIREFLY_K = 16` (`rt_reflect.rs`, passed as a root constant / `g_temporal.z`). A sweep
  chose it as the lowest K that keeps the highlights' energy near an unclamped linear
  reference (table). The offline flags `F_LINEAR` / `TEMPORAL_LINEAR` keep the E085 path
  for the before/after comparison.

**Offline (M, RX 7900 XTX; ten lamps of luminance ~35 on the wall, reflected by a floor
with roughness 0.3; reflection term = output - input):**

| Path | SSR highlight RMSE | SSR lamp energy vs reference | Rays highlight RMSE (jitter / none) | Rays lamp energy (no jitter) |
| --- | ---: | ---: | ---: | ---: |
| E085 (before) | 0.03156 | 96 % | 0.01822 / 0.01561 | 67 % |
| K 4 | 0.00122 | 31 % | 0.00974 / - | - |
| K 8 | 0.00237 | 68 % | 0.01155 / - | - |
| K 16 (default) | 0.00382 | 110 % | 0.01395 / 0.00533 | 92 % |

- SSR: static camera, frames 72..96, 15,410 highlight pixels. The 64-frame linear reference
  is 0.00414. The largest single-pixel change drops from 0.564 to 0.241, and the mean floor
  reflection moves from 0.0818 to 0.0824.
- Rays: 44,189 highlight pixels, 256-sample photo reference. Without jitter the
  highlights' frame-to-frame RMSE drops to 34 % of E085's, and the largest pixel change from
  0.438 to 0.279. With the engine-like jitter the lamps themselves move on screen every
  frame, and the reduction is 23 %. Mean floor output is 4.2-4.4 % above E085's.
- GPU: the moment adds one R32F read and one write per half texel. Ray timings vary
  between runs (E085 0.13 ms, K 16 0.14-0.22 ms). The SSR cost was not timed separately.
- Acceptance on hardware: the highlight RMSE is at most half of E085's (rays: without
  jitter), lamp energy stays within 15 % of the reference, and single-pixel jumps shrink
  (SSR: below 0.6x of E085's; rays: no larger than E085's, floor mean within 5 %). WARP runs
  skip the ray firefly acceptance.

**Toast (`rt_reflect.rs` `Reporter`, unit tests `toasts_only_where_the_user_did_not_choose_it`,
`startup_with_settings_off_does_not_toast`, `transient_toasts_after_grace_definite_at_once`):**
the text "Ray-traced reflections need: Reflections on (ssr = 1)" was correct, but it was
toasted because the user had just switched `ssr` off themselves. Blockers now have a
severity. **Setting** (`ssr`, `rt`, `fsr` off) shows in the card and the log. It is toasted
only when the user turns `rt_reflections` on while that setting is off. **Definite**
(device failure, no DXR, device lost, scene failure, camera mismatch) is toasted at once.
**Transient** (FSR starting, device being created, scene building) is toasted after 10 s
without change.

**In-game steps:** a glossy floor reflecting lamps or windows with `ssr = 1`, then with
`rt_reflections = 1`. Standing still, the highlights should be steady and not dimmer than
before. Switching Reflections off with ray tracing on should only change the card.
