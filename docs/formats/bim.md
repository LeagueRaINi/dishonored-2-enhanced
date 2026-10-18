# Images (`generated/image/**.bimage7`, `.bimage7_mipN`)

Implementation: `crates/void-formats/src/bim.rs`. Labels: **O** observed, **D** derived, **H** hypothesis.

## Provenance

- Executable `b671f824…6180`: format-name table at `0x14229CC80` (14 pointers to
  `_R8_UNORM` … `_BC7_UNORM_SRGB`) followed by 14 four-byte descriptors
  `(0, channel?, numeric type, format)`. Mip part names use `"%s.bimage%d_mip%d"`
  (`0x141B174E8`). Loader: `idBinaryImage_LoadFromGeneratedFile` 0x1402D2740 with
  `idImageOpts_Read` 0x1402EA840 (header field mapping in docs/texture-replacement.md §1.2).
- Samples: all 1,491 base images of `tutorial_p.mapresources` (header survey), and the
  1,275 images referenced by the tutorial world materials including every streamed part.

## Layout (little-endian)

```
[4]  "BIM" 0x18
u32  @0x04  header version            19 in all samples
u32  @0x08  texture type              0 = 2D, 1 = 3D (colour cubes), 2 = cube (probes)
u32  @0x0C  format                    see table
u32  @0x10  numeric type              5 UNORM, 6 UNORM_SRGB, 4 SNORM, 3 UF16, 2 SF16
u32  @0x14  ?                         1 in all samples
u32  @0x18  width of first mip in this file
u32  @0x1C  height of first mip in this file
u32  @0x20  full width
u32  @0x24  full height
u32  @0x28  depth                     1, 32 for colour cubes
u32  @0x2C  mips stored in this file
u32  @0x30  total mips                chain ends at 4x4 for BC formats
u32  @0x34  faces                     1, 6 for cubes
@0x38..0x3E  unknown                  0x80000000 then flag bits (0 / 0x80 cube / 0x4002)
mip record, repeated (records = mips_in_file x faces; cubes stored with faces = 1 still
carry 6; 3D images one record per depth slice of every mip, depth halving per mip):
  u32 level  u32 face  u32 width  u32 height  u32 size  [size] data
```

Streamed part `_mipN` (index entry flag 0x20, F007 of archives.md) is exactly one mip
record for level N, where N = 0 is the full size. Base-file records number their own
levels from 0 (O).

## Formats (O: exe table; counts from the tutorial survey)

| format | numeric | meaning | tutorial base images |
| --- | --- | --- | --- |
| 0x01 | 5 | R8 / A8 UNORM | 0 |
| 0x04 | 5 / 4 | R8G8 UNORM / SNORM | 2 |
| 0x0A | 5 / 6 | R8G8B8A8 UNORM / SRGB | 1 |
| 0x11 | 5 / 4 | BC4 UNORM / SNORM | 712 |
| 0x12 | 5 / 4 | BC5 UNORM / SNORM | 326 (SNORM, normal maps) |
| 0x13 | 3 / 2 | BC6H UF16 / SF16 | 20 (probe cubes) |
| 0x14 | 5 / 6 | BC7 UNORM / SRGB | 430 (SRGB, colour maps) |

Unknown pairs are reported (`Format::from_fields` returns `None`), never guessed.

## Verification

`void-tool materials --bwm maps/campaign/tutorial/tutorial_p/tutorial_p.bwm --mips`:
323 materials, 1,275 images, every base file and every streamed part parses with exact
sizes (`mip_size` per format), 3.25 G texels, 0 failures (`local/m2/materials_tutorial.txt`).
Synthetic tests cover truncation, trailing data, size mismatch, streamed level/size
checks, unknown formats, and BC4/BC5 decoding.

## Unknowns

- Field @0x14 and bytes 0x38..0x3E (writer and mod validation preserve them; see
  docs/texture-replacement.md §1.2 for the engine's use of 0x3C flags 0x80/0x4000).

## Writer

`Image::to_bytes` / `Mip::to_bytes` (streamed part) rewrite every image of
`tutorial_p.mapresources` byte-identically (`void-tool texmod roundtrip`: 1,491 base files,
2,231 parts). The record-count rule above comes from `idBinaryImage_LoadFromGeneratedFile`
(0x1402D2740, O); before it, three 3D images of the survey failed to parse.
