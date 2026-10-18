# Static entity models (`.bmodel`, magic `1LMB`)

Implementation: `crates/void-formats/src/bmodel.rs`. Survey: `void-tool bmodels --group 0`.

## Provenance

- Writer `idStaticModel::WriteStaticBModel` (`sub_1402722B0`) and the `BTRI` surface
  reader (`sub_140259D50` -> `sub_14025A040`, versions > 0x10001) in `Dishonored2.exe`
  SHA-256 `b671f824...6180` (E001), via the IDB copy in `local/ida/`.
- E022: `void-tool bmodels --group 0 --kind model`: all 417 `model`-kind resources in
  group 0 (including the 16 tutorial map-local volume models) parse with whole-file
  consumption; all 499 surfaces use one layout: version 0x10002, CPU declaration 1,
  GPU declaration 1, one 32-byte interleaved stream. Other groups not surveyed.

## Layout (little-endian; `str` = i32 length + bytes, length <= 0 = empty)

```
"1LMB" | u32 stamp | str source | u32 surface_count
surface:
  str material | i32 unknown | "BTRI" | u32 version | i32 cpu_decl | i32 gpu_decl
  [40] header | u32 vertex_count | u64 gpu_bytes
  gpu_bytes > 0: u8 streams; per stream { u64 offset, u64 stride, u64 size, [size] }
  gpu_bytes = 0: vertex_count * stride(cpu_decl) bytes        (all observed files)
  u32 index_count | u16[index_count] | f32[6] bounds | u32 tail | "BML1"
u32 n | n * { str, u32, u32, u32 }
```

CPU declaration 1 (32 bytes): `f32x3 position` + the same 20-byte attribute block as BWM
stream 1 (docs/formats/bwm.md: uv0, uv1, normal + barycentric w, tangent + bitangent
sign, vertex colour). D: the stride and field match are inferred from file structure and
from rendering (entity props shade and texture like world geometry, E023); the declaration
table itself is runtime data not yet recovered.

Indices are surface-local and checked against `vertex_count`. Winding follows the world
model (clockwise front faces).

## Unknowns

`unknown` i32 (often -1), the 40-byte header, `tail`, the trailer records (name + 3 u32;
H: LOD/attachment tags), GPU-stream files (none observed).
