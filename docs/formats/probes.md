# Environment probes: `.probesnetwork` (PNW) and `.bprobe` (PRB)

Status: both formats parse completely (every byte consumed) for tutorial_p and
eotw_p_highchaos; structural cross-checks pass (below). Semantics of the per-instance
4-int table and of the Voronoi cell/record fields we skip are unknown (U).
Parser: `crates/void-formats/src/probes.rs`; check: `void-assets/examples/probes_check.rs`.

Provenance (exe `b671f824...6180`, O): PNW reader `sub_1402B15E0`, Voronoi block
`sub_1403843F0` (+ `sub_140384870` cell, `sub_140384BB0` face, `sub_140384D70`
record), PRB serializer `sub_1402BAA80` -> `sub_1402BA2B0` (results) / `sub_1402BA900`
(sample) / `sub_1402BA9D0` (Voronoi info) / `sub_1402BA1E0` (occluder); field names from
the reflection tables (`arkEnvProbe`, `arkEnvProbe::results_t`, `...::sample_t`, ...).
GPU-side meaning from the shipped shader source (`indirectlighting/*.decl`, `vertex.decl`).

## Where it fits (O, shaders)

Static-world vertices fetch `indexTetrahedron` from a per-vertex reference buffer at
`bufferPerVertexReferenceOffsets[instance] + vertexID`; the tetrahedron's 4 probes are
blended with barycentric x per-vertex u8 weights; the probe SH (9 RGB coefficients on PC,
`bufferProbesSH_RGBE`) is packed into 4 irradiances along tangent-frame principal
directions per vertex and unpacked per pixel (`UnPackSHPrincipalDirections`), then
`ambient = 1/pi * albedo * Fdr * irradiance` (`lightcontext.decl`). Probe SH is rebuilt
at runtime: `updateprobe.decl` relights the 128 samples per probe (probe light list, sun
with shadows) into SH (Lanczos-filtered), `accumulatesh.decl` sums
`static x bounceStatic + sun x bounceSun x sunColorAtGroundLevel + skyZH (x) skyOcclusionSH
+ dynamic x bounceDynamic + ambient term`.

## PNW (version 7)

```
u32 magic 'PNW' | version<<24 (7)
u32 n, n x string            probe resource names ("<map dir>/probes_probe_N"; + ".bprobe")
u32 n, n x string            IBL probe names
Voronoi block:
  u32 vversion (9), u8 flag
  u32 cells (= probes), each: string probe, f32, f32, u8, f32, f32, i32 cell, u32,
       u32 faces, each: i32, u32, 3x vec3, u8, u8, u32 nv, nv x vec3
       u32 n, n x u32; 24 bytes (AABB)
  u32 records (= tetrahedra), each: u32, u32, 48 bytes            (vversion 9)
  u32 bytes, tetrahedra x 24: u32 vertex[4], u32 probes01 (u16 lo/hi), u32 probes23
  u32 bytes, vertices x 16:   vec3 position, u32 weights (u8 x4, /255)
u32 n (= .bwm instances), u32 stride (4), n x stride i32 (-1 padded; U)
u32 bytes, u32 per-vertex tetrahedron references   (bufferPerVertexReferences)
u8  0
u32 bytes, u32 per-instance offsets into the references
```

The reader's call sequence suggested the zero byte before the references; the files have
it after them (O: byte inspection); the layout above consumes both files exactly.

## PRB (14.1)

```
'PRB\x0b', u32 major 14, u32 minor 1
vec3 shootingPosition, u32 zoneGroup, u8 isTransition,
vec3 bounceFactorSun, Sky, DynamicLighting, StaticLighting,
f32 AO_Sky, f32 AO_Global, vec3 ambientLightingFactor, u8 usedForIBL, u8 useVirtualPlanes,
f32 IBLIntensity
results StateA, u8 hasStateB, [results StateB]
results: f32 distance mean/harmonicMean/min/max, vec3 lsBBoxMin/Max, vec3 staticLightsIlluminance,
  f32 skyOcclusionFactor, f32 sunIrradianceFactor, vec3 illuminanceEstimate, u32 pixelCounts[9],
  SH3 static, SH3 static_SunFactor, SH3 static_SunFactor_Filtered (9 x vec3 each),
  f32[9] skyOcclusion, f32[9] skyOcclusion_Filtered, f32[256] environmentDistance,
  3 x SH3 *_Initial (multi-bounce static, Sun, sky; x3.0 at load), f32 distanceSample min/max/mean,
  u32 128, 128 x sample (vec3 pos, vec3 normal, f32 radiusMinor/Major, u32 albedo_F0,
     f32 sunIlluminanceFactor, f32 SHFactor, u16 occluderIndex, u16 attachedProbeIndex),
  u32 n emissive (string material, f32[9] SH, f32 importance),
  f32 nearest/farthest neighbour distance,
  u32 n neighbours (string probe, u8 visible, i32 voronoiIndex, f32 distance, f32 solidAngle,
     vec3 direction, u8 isTransition),
  u32 n voronoi infos (string, u32, vec3, vec3, f32, f32[9], u8, u8),
  u32 n occluders (string, 3 x SH3, 2 x f32[9], u32, u64, u32),
  string cubeMapName, vec4[6] virtualPlanes
```

## Checks (O)

| Map | Probes parsed | Tetrahedra / vertices | Instance offsets vs `.bwm` vertex counts | Max reference |
| --- | --- | --- | --- | --- |
| tutorial_p | 531 / 531 | 38,080 / 30,665 | 17,279 / 17,279 | 38,079 |
| eotw_p_highchaos (`eotw_p.probesnetwork`) | 3,312 / 3,312 | 261,888 / 205,502 | 94,111 / 94,111 | 261,887 |

Map note: eotw high- and low-chaos each ship their own `eotw_p.probesnetwork`, named after
the base map, not `<map>.probesnetwork` (O, index).
