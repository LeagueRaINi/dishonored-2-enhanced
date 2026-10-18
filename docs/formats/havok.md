# Havok 2016 tagfiles, `.bphysworld`, and collision shapes

Implementation (engine-independent): `crates/void-formats/src/havok.rs` (generic tagfile and
type-compendium reader), `physworld.rs` (`.bphysworld` container, `physhape` prefix),
`collision.rs` (shape trees -> world-space hulls/triangles). Tools:
`void-tool havok --kind physhape|physworld --name <res> [--schema] [--dump i,..] [--dump-type T]`,
`void-tool collision [--region x0,y0,z0,x1,y1,z1] [--obj out.obj]`,
`void-probe --collision <physworld>` (render vs collision heights), `tools/hktag.py` (sections).

Provenance: shipped resources of the Steam build (exe SHA-256 `b671f824...6180`, E001);
`tutorial_p.bphysworld` (9,560,286 bytes) and the door physhape
`door_350cm_ext_poor_metal_01.lwo.hkt._100_100_100.10.hkx` (2,132 bytes), group 0.
No executable code was traced for this format; everything below is from file structure,
cross-checked as noted. Havok's own runtime is not used.

## Container: binary tagfile (`TAG0`, SDK `20160100`)

Sections: u32 **big-endian** header (bits 30-31: 0 = container, 1 = leaf; bits 0-29: size
including the 8-byte header) + 4-char tag. Payload integers are little-endian except the
TNAM/TBOD varints.

| Section | Content |
| --- | --- |
| `SDKV` | `20160100` |
| `DATA` | object data in native 64-bit layout (record sizes/offsets from TBOD) |
| `TYPE` | `TSTR` type/template strings (NUL-separated), `TNAM` names + templates, `FSTR` field names, `TBOD` bodies, `TPAD` |
| `TCRF` | instead of `TYPE`: 8-byte id of a `TCM0` type compendium (+16 zero bytes) |
| `INDX` | `ITEM` (12-byte LE `{type | flags<<24, data offset, count}`; item 0 null; flags 0x10 object, 0x20 array), `PTCH` (`{type, n, n x data offset}` groups) |

`TCM0` = `TCID` (8-byte id) + `TYPE { TPTR TSTR TNAM FSTR TBOD THSH TPAD }`.

Varints (TNAM/TBOD): first byte `0xxxxxxx` = 7 bits; `10` prefix = 14 bits (2 bytes);
`110` = 21 bits; `11100` = 27 bits; `11101` = 35 bits; `11110` = 59 bits; `11111` = 64 bits,
big-endian continuation bytes.

TNAM: count n (including null type 0), then for types 1..n-1: name (TSTR index), template
count, templates `{name (TSTR index; "t..." = type parameter, "v..." = value), value}`.

TBOD: repeated `type, parent, options`, then by option bit: 0x01 format, 0x02 subtype,
0x04 version, 0x08 size + alignment, 0x10 flags, 0x20 members (`count`, each
`{name (FSTR), flags, offset, type}`), 0x40 interfaces (`count`, each `{type, value}`),
0x80 attribute. Absent format/subtype/size are inherited from the parent chain.

Format word (D): low nibble = kind (0 void, 1 opaque, 2 bool, 3 string, 4 int, 5 float,
6 pointer, 7 record, 8 array); arrays with bit 0x20 are inline (`count = format >> 8`);
ints: bit 0x200 signed, size bits 0x2000/0x4000/0x8000/0x10000 = 1/2/4/8 bytes; float
`0x1746..` = f32. Pointer, string, and variable array fields hold a u32 **item index** at
their native offset (the item carries the element count).

**Validation (E026):** the offsets of every non-null reference field found by the
schema-driven decoder over all 58,551 items of the tutorial physworld equal the `PTCH`
table exactly (73,476 offsets) — this checks every record size, member offset,
inheritance chain, and inline-array size the decoder uses. The door file has no `PTCH`.

## `.bphysworld` (little-endian, every byte accounted for)

```
u32 n | n x { u32 len, bytes }   contact-system material names (21 in tutorial_p)
u32 m | m x u32                  index into those names (101 entries; H: per Havok material id)
u32 size | TCM0                  type compendium (209 types)
u32 size | TAG0                  tagfile with TCRF matching the TCID
```

Content: one `hknpPhysicsSystemData` with 12 static bodies (`motionType` 0, identity
position/orientation, different `collisionFilterInfo`: H, collision layers), each with a
shape tree.

`physhape` `.hkx`: 5-byte prefix (all zero in the sample; U) + self-contained `TAG0`.

## Shapes (`collision.rs`)

| Type | Decoding | Status |
| --- | --- | --- |
| `hknpCompoundShape` | `instances.elements` (`hkFreeListArray` of `hknpShapeInstance {transform, scale, shape}`); null shape = free slot | decoded |
| `hknpConvexPolytopeShape` | `vertices` (hkVector4f; w carries an index), `planes`, `faces {firstIndex, numIndices, minHalfAngle}`, `indices` (u8), `convexRadius` | decoded |
| `hknpConvexShape` | vertices only | decoded |
| `hknpCompressedMeshShape` (`numConvexShapes` 0) | quantized `hkcdStaticMeshTree`, see below | decoded for section flags 0 |

`hkTransform` = 3 rotation columns + translation (column-major; the transposed reading
fits render geometry worse). Instance scale is applied in the child frame (H; all scales
in tutorial_p are uniform).

Compressed mesh tree (D, consistent with all section domains: max vertex excess 0.0 over
56 sections):

- packed vertex (u32): x bits 0-10, y 11-21, z 22-31; `codecParms[0..3] + q * codecParms[3..6]`;
- shared vertex (u64): x bits 0-20, y 21-41, z 42-63 over the tree `domain`:
  `min + q * (max - min) / (2^bits - 1)`;
- section: `primitives.data` and `dataRuns.data` = `offset << 8 | count`;
  `sharedVertices.data >> 8` = offset into `sharedVerticesIndex` (count `numSharedIndices`);
  `firstPackedVertex`/`numPackedVertices` select packed vertices;
- primitive `indices[4]`: `< numPackedVertices` = packed, else shared via the index list;
  quad = (0,1,2)+(0,2,3), triangle when indices 2 and 3 are equal.
- **Unsupported:** sections with `flags` 1 (invalid codec values, primitives such as
  `(146,150,150,150)`): 5 meshes / 264 primitives in tutorial_p, all at x >= 647
  (outside areas D1 and spawn); reported by `void-tool collision`.

## Coverage (tutorial_p, E027)

15,797 placed shapes: 15,736 convex hulls (all convex radius 0) and 61 compressed-mesh
instances (51,122 triangles); `void-tool collision` exits non-zero while unsupported
parts remain.

## Unknowns

`collisionFilterInfo` layer semantics; the 101-entry material map and per-primitive
`dataRuns` values (shape tags -> materials); `triangleIsInterior`, welding map,
connectivity; edge/face `Connectivity` of polytopes (not needed for queries); section
flags 1 encoding; the physhape prefix; whether any shape is player-only or AI-only.
