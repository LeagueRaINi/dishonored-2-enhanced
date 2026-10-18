# Skinned models (`.bmd6model`, magic `MM6B`), Edge skeletons (`.edgeskl`, `60SE`), `md6Def`

Implementation: `crates/void-formats/src/md6.rs`. Tool: `void-tool md6` (see Validation).
Evidence labels: O = observed in the exe (IDB `local/ida/Dishonored2.exe.i64`, exe SHA-256
`b671f824...6180`) or in the shipped data; H = hypothesis consistent with the data; U = unknown.

## Resource chain

An entity's `renderModelInfo.model "x/y.md6"` names an **md6Def decl**, not a file (O):

```
x/y.md6 --> generated/decls/md6def/x/y.md6.decl           (kind md6Def, text)
   init { inherit "<other>.md6"                            parent decl (chains of 4 decls seen)
          skeleton "<s>.md6skl"  --> generated/skeleton/<s>.edgeskl   (kind skeleton)
          lod N { [clear] mesh "<m>.md6mesh" [{ cloth "<hkt>" "<setup>" ... }] ... } }
   <m>.md6mesh --> generated/basemodel/<m>.bmd6model        (kind baseModel)
                   or, for meshes with cloth, nocloth_<file stem of m>_<32 hex>.bmd6model
```

- Inheritance (O, eotw snapshot): a child's `lod N` adds its meshes to the parent's LOD
  N unless it says `clear`; the child's `skeleton` overrides. Example
  `models/characters/tall/overseer/overseer_head.md6` -> `overseer_body.md6` ->
  `base_overseer.md6` -> `base/base_body.md6`; LOD 0 = body (`nocloth_t_grd_overseer01_body_*`)
  + head mesh. The snapshot's surface list for that model has the body and head materials
  together (one render model per character).
- `nocloth_%s_%s.bmd6model` (format string 0x141AD2C70, built by `sub_140088150`): the mesh
  without the cloth-simulated parts; the hash is a digest over name hashes of the cloth
  setups (not recomputed; 2 eotw parts have two candidate builds, the tool takes the first).
  Cloth parts are separate `voidresourcecache/md6model/cloth28/<x>/<mesh>_<setup>_<hash>_<hash>_42.bmd6model`
  models (single surface, no skeleton name, model flag 1); their link to a decl and to
  joints is U, so the tool leaves them out (e.g. a guard's trousers below the knee).
- Entity `.md6` names equal to the mesh name (props, tutorial_p's 13) make the direct
  `generated/basemodel/<stem>.bmd6model` guess work; for characters it does not
  (56 of 129 eotw snapshot md6 need the decl).

## Model (`sub_14025AA00`, little-endian; `str` = i32 length + bytes, <= 0 empty)

```
"MM6B"                                   O: compared to 0x424D364D at start and end
str  skeleton (.md6skl)                  O: looked up in the skeleton manager; empty in nocloth/cloth builds
u16  skeleton checksum                   only if skeleton non-empty; = skeleton custom +338 (below)
f32[6] bounds min xyz, max xyz           -> model +52
u32  surface count, surfaces             -> list at model +88
u32  material count, {str, u32, u32, u32} 24-byte runtime entries at model +104
                                         H: (material, surface index, first vertex, last vertex)
joint helpers (sub_1402538F0/sub_1402537D0): u32 n; n * { str name (arkhelper*_jnt),
     u32 count, count * { u16 (low byte kept), u32 (low 3 bits kept), f32[3] } }
u8   flags                               bit 0 -> model +145 (H: free CPU vertex copies)
"MM6B"
```

Surface (`sub_140259960`):

```
str  name ("body", "head", "body_lod2", ...)
str  material                            unknown names fall back to an implicit material
u32  kind      -> surface +48            loader tests 1 and 3..6 (U meaning; 0..10 seen)
u8             -> +52  (1 in all samples)
u32            -> +56  (2 in all samples)
BTRI block                               shared with .bmodel (docs/formats/bmodel.md):
     "BTRI" u32 version (> 0x10001; 0x10002 in all) i32 cpu_decl i32 gpu_decl
     f32[10] header (1,1,1, 0,0,0, 1,1, 0,0 in all; H: position/uv scale+bias)
     u32 vertex count (<= 0xFFFF for u16 indices) u64 gpu bytes (0 in all 1,090 models)
     gpu bytes = 0: vertex_count * stride(cpu_decl) bytes; else u8 streams { u64 off, stride, size, data }
     u32 index count, u16 indices (triangle list, surface-local), f32[6] bounds, u32 tail (0)
u32  range count, {u32 part, u32 first index, u32 count}   list at +168 (H: dismemberment parts)
u32  joint offset   -> +92
u32  joint count    -> +96               H: palette size the stored indices address
u32[8] used-joint bitset, u32 8, u32 256 -> +100 (40 bytes)
```

### Vertex declarations (O: `sub_1402D0DC0` builds 13 element lists; `sub_1402D1AF0` maps id -> list)

Element record: `u32 format | u8 semantic | u8 index | u8 offset | u8 slot`; semantic names
`off_14229AB40` (`VERTEX_POSITION` 0x141B11798 ... `VERTEX_SKINWEIGHTS` 0x141B11800,
`VERTEX_SKININDICES` 0x141B11818); vertex size = sum of element sizes (`sub_1402D1DA0`).
The decoder's table (`md6::declaration`) reproduces these sums (unit test). Skinned models use:

| decl | influences | layout (slot: elements) | surfaces (all 1,090 baseModel) |
| --- | --- | --- | --- |
| 10 | 8 | 0: f32x3 pos; 1: f16x2 uv0, f16x2 uv1; 2: snorm8x4 normal; 3: snorm8x4 tangent; 4: unorm8x4 w0, unorm8x4 w1, u8x4 j0, u8x4 j1 (44 B) | 3,043 |
| 9 | 4 | same without w1/j1 (36 B) | 221 |

Both appear in the eotw snapshot's md6 (survey: declarations {9, 10}). Decls 3/4 are the
same elements in one interleaved slot; 7 adds vertex colour and two extra texcoords.

Multi-slot data is stored **slot after slot** (all positions, then all uv pairs, ...): H
from the reader (one blob of `count * stride` bytes) and O from the data: normals and
tangents decode to unit length, every weight set sums to 255, and dominant joints lie next
to their vertices (below). Normal/tangent encoding as in `.bwm` (tangent w = bitangent sign).

### Joint indices: mesh-local + `joint offset` (answer to the palette question)

Stored indices are **relative**; the skeleton joint is `stored + joint offset` (surface +92):

- O: the 32-byte used-joint bitset equals `{stored + offset : weight > 0}` exactly for every
  influence of all 1,090 models (`--all`: 0 influences outside the set).
- O: median distance from a vertex to its dominant joint's bind position, surfaces with a
  non-zero offset: overseer 0.086 m with the offset vs 0.699 m without; guard 0.044 vs
  0.809; female civilian 0.012 vs 1.650.
- 278 of 3,264 surfaces have an offset (e.g. 16 and 48 on the overseer's gore and skin
  surfaces); a decoder must never assume 0.
- U: whether the engine adds the offset on the CPU (palette sub-range per draw) or in the
  shader. Either way, skeleton-global joint = stored + offset indexes the skeleton's joint
  list, which is the order of the inverse-bind matrices.

## Skeleton (`.edgeskl`; Edge animation skeleton + arkSkelBase custom data)

Header (`sub_14024F2A0`, 64 bytes; O: tag 0x45533036 and size == file size checked):

```
+0x00 "60SE"  +0x04 u32 file size  +0x08 u32 custom data size  +0x0C u32 (U)
+0x10 u16 joint count  +0x12 u16 user channel count  +0x14 u32 (2 in all)
+0x18..+0x30 7 self-relative u32 offsets (field address + value):
      [0] base pose, [6] custom data, others U (name hashes, channel data)
+0x34 u32[3] (U)
+0x40 hierarchy: (u16 joint, u16 parent) pairs in 16-byte quads, parent 0x7fff = root,
      padding repeats entries (H: Edge SIMD hierarchy; O: parents precede children in all)
base pose: per joint f32x4 rotation quaternion (x y z w), f32x4 translation, f32x4 scale
```

Custom data (`sub_1402564E0`):

```
u32 version (14)  str parent skeleton  [u16 parent checksum if parent]
u16 checksum (+336)   u16 model checksum (+338; models store this one)
u16 joint count (= header)   u16 n_lod (<= 4)
n_lod > 0: u16[n_lod] (H: per-LOD joint counts), u16[joint count] (H: joint order)
str[joint count] joint names          10 * joint count bytes (U flags)
u16 channel count, str[channels], 10 * channels bytes
f32[12] * round_up(joint count, 8)    inverse bind matrices, row-major 3x4 (-> +288)
then joint groups, helpers, locators, aim constraints (sub_1402529B0, sub_140255450,
sub_140253710, sub_140251DF0; not decoded)
```

Bind pose: global = parent global * `[R(q) * S | t]` (column vectors). O: the stored
inverse binds invert these globals to <= 3.9e-6 (max element) in all 288 skeleton names,
so `inverse_bind[j] * bind[j] = I` and the mesh vertices are already in bind pose (model
space, metres, z up, characters face +x).

Checksums (O, eotw survey): model checksum = skeleton +338 for every part except the 12
LODs of the three `t_civ_aristo02_head*.md6`, which use `t_civ_aristo01_head*` meshes
(checksum 0x0618) on the `t_civ_aristo02` skeleton (0x2a64). The two skeletons share the
names of joints 0..211 (aristo01: 218 joints, aristo02: 219; they differ from 212 on);
all indices stay in range. H: the engine remaps such meshes by joint name; safest for a
renderer is to remap by name when the checksums differ.

## Runtime skinning

H (corroborated): GPU vertex-shader skinning from a per-model 3x4 joint palette, 4 or 8
influences: render-prog options `useSkinning` (0x141B17F30), `useSkinning8` (0x141B17F40),
`useskinning4` (0x141B19078), `PreviousFrameSkinningMatrices` (0x141B354C8). Peer finding
(local://dynamic-entities.md, DynRE): palette of 48-byte row-major 3x4 matrices at render
model +8184/+8192, surface geometry = the shared bind-pose `idMD6Mesh`. Skinned position
= `sum_k w_k/255 * (palette[j_k] * p)` with palette[j] = joint world * inverse_bind[j] (H,
standard; the decoded data satisfies its preconditions).

## Validation (`void-tool md6`, results in `local/md6/`)

- `--snapshot local/inject/snap_1792183388_f1288/models.tsv` (eotw, `eotw_survey.tsv`):
  **129 of 129** distinct md6 names (741 rows) resolve and parse in every LOD (293 of 293);
  0 weight sets != 255, 0 joint indices out of range, 0 influences outside the used-joint
  set; 12 checksum notes (aristo02, above). The snapshot's drawn materials are all present
  in the decoded meshes for 56; the rest are NPC variation skins (`body01`/`body02`
  variants, `g_nohair`, arm/hand materials; `m_npcVariation.m_skin` 0x141D33978) and
  `*_usable` prop materials (H: runtime material remap).
- `--all` (`all_survey.txt`): 1,090 of 1,090 baseModel names parse with full consumption
  and 288 of 288 skeleton names parse (custom data after the inverse binds not decoded);
  influences per vertex 1..8 all occur.
- Bind-pose OBJs (`--name ... --obj`): `emily_body.obj` (first-person body, 1.689 m),
  `overseer_head.obj` (1.935 m, T-pose span 1.33 m), `guard.obj` (1.820 m),
  `civ_female.obj` (1.793 m), `rat.obj` (0.206 m); `bind_pose_front.png` renders them:
  intact figures, no stretched triangles (longest edge 0.18 m; 99.9th percentile <= 0.14 m).
  The 51 eotw human NPC md6 (small/tall/female) measure 1.777-1.954 m (survey column `height`).

## Unknowns

Surface kind values, the +52/+56 fields, the BTRI header and tail; joint-count field
semantics beyond "covers the stored indices"; skeleton header +0x0C/+0x34 and offsets
1..5; 10-byte joint/channel records; custom data after the inverse binds; cloth-cache
naming and joint binding; `nocloth` hash; GPU-stream files (none shipped); how the engine
applies `joint offset` and resolves skeleton-checksum mismatches; LOD selection distances.
