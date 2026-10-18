# Static world model (`maps/**/<map>.bwm`, `BWM1`)

Implementation: `crates/void-formats/src/bwm.rs`. Labels: **O** observed, **D** derived, **H** hypothesis.

## Provenance

- Executable `Dishonored2.exe` SHA-256 `b671f824…6180` (E001), analysed through a
  copy of the existing IDB in `local/ida/` (the `d2enhanced/re` database is not opened).
- Reader: `sub_140295FA0` checks magic `0x314D5742` (`BWM1`) and reads the version.
  Versions `<= 0x10013` go to `sub_140296240` (not recovered); later versions go to
  `sub_1402974C0` (recovered below). Class `arkStaticWorldModel`, source string
  `render\arkStaticWorldModel.cpp`. Versions `< 0x10015` additionally compute an Umbra
  object mapping at load time.
- Sample: `maps/campaign/tutorial/tutorial_p/tutorial_p.bwm` (group 0, 50,658,877 bytes,
  version `0x10015`). The recovered layout consumes the file exactly (O).

## Layout (version > 0x10013; little-endian)

```
[4]  "BWM1"
u32  version                                 0x10015 observed
u32  stream_count                            1..=2 (exe rejects others)
     stream[i]: u32 bytes, bytes             vertex data
u32  index_bytes, bytes                      u16 indices (D)
u32  instance_xform_bytes, bytes             48-byte records: f32[12] row-major 3x4 [R|t] (D)
u32  buf24_bytes, bytes                      24-byte stride per exe; empty in sample
u32  surface_count
     surface: u32 n (<= 2)
              (u32 vertex_byte_offset, u32 stride) x n     per stream
              u32 sentinel                   0x12345678 in every sample surface
              u32 index_byte_offset          (exe field +56)
              u32 index_count                (exe field +64)
              u32 vertex_count               (exe field +60)
              u32 unknown                    (exe field +68) = 3 in all samples
              u8  flag                       = 1 in all samples
u32  batch_count                             (exe allocates 32-byte records)
     batch:   u32 first_instance, u32 end_instance, u32 surface
              u8  flag
              u32 len, [len] material name   resolved via the material decl manager
              u32 unknown                    0 or 1
u32  instance_count
     instance: f32[6] AABB (min xyz, max xyz) | u32 unknown | u16 unknown
u32  count[4], then 4 arrays of count[k] u32     sample: [8,0,0,0]
u32  n, n x { u32, u32 m, u32[m] }                 sample: 0
if version > 0x10014:
u32  n, n x { u32 m, u32[m] }                      sample: 12,189 singletons
u32  m, u32[m]                                     sample: 0
```

## Verified relationships (tutorial_p)

| Check | Result |
| --- | --- |
| Stream strides from surfaces | (12, 20) for every surface; both streams hold 1,313,108 vertices |
| Surface vertex ranges | contiguous, end exactly at stream 0 length |
| Surface index ranges | contiguous with u16 size, end exactly at index buffer length |
| Index bound | every index < surface `vertex_count` (first 200 surfaces): indices are surface-local |
| Batches | tile instances `[0, 17279)` exactly; `surface` in 0..=1060 |
| Transforms | instance matrix x surface positions reproduces the stored AABB, max error 1e-4 over 76 instances |

## Vertex format

- Stream 0 (12 bytes): `f32 x, y, z` object-space position (O: matched against game depth, RX001).
- Stream 1 (20 bytes):
  `f16x2 uv0 | f16x2 uv1 | snorm8x4 normal | snorm8x4 tangent | unorm8x4 vertex colour`.
  - UV sets: the shader reads `m_texCoords.xy` / `.zw` per layer `UVset` (shader source, O);
    uv0 confirmed visually in RX001.
  - **Normal (bytes 8..11): snorm8 xyz + w = barycentric selector** (-1/0/+1;
    `ExtractBarycentricsFromNormalW` in `vertex.decl`). E019: 40,098 sampled vertices are
    unit length (median 1.000), agree with outward (clockwise-front) face normals
    (median 15 deg smoothing spread, 0.15 % beyond 90 deg); w takes -1/0/+1 exactly
    13,366 times each.
  - **Tangent (bytes 12..15): snorm8 xyz + w = bitangent sign** (`vertex.decl`:
    bitangent = `cross(normal, tangent) * w`); |dot(normal, tangent)| median 0.000.
  - The earlier "rejected" verdict (E015) was a viewer artifact: before the RX001 winding
    fix the viewer drew both sides with front faces treated as back faces, so Bevy flipped
    the correct file normals.
  - Vertex colour channel order: see docs/formats/arkdefault.md (unresolved).

## Conventions verified against the game (RX001, docs/experiments/RX001-…)

- Units are game world units; Z up; id-style angles; not mirrored.
- **Front faces are clockwise** in index order as seen by the camera.
- **Mirrored instances** (1,284 of 17,279 have negative determinant) keep object-space
  facing: the game compensates culling for mirroring (62.8 % vs 41.2 % depth match at P2).
- The `arkSky` dome is in the file but drawn at infinite depth; vista meshes get farther
  depth in game than their geometry implies (treatment unknown).

## Unknowns

- Units (meters vs other); handedness beyond "Z-up" (D from normals/rotations about Z).
- Meaning of surface +68 / flag, batch flag / trailing u32, instance trailing u32/u16,
  and the four trailing tables (probably visibility/Umbra and dark-vision lists).
- Pre-0x10014 layout (`sub_140296240`); not needed while all shipped maps use 0x10015
  (unverified for other maps).
