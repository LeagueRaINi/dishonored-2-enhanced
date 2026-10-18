# M2: tutorial_p rendering status and remaining differences

Updated 2026-10-16. Scope: static world model (`tutorial_p.bwm`) plus static entity models
from `tutorial_p.entities`, Bevy 0.20 on D3D12. References: RX001 (geometry, depth) and
RX002 (clean material/lighting captures), docs/experiments/.

## Reproduce

```
cargo run --release -p void-viewer -- --game <ROOT> --window 1920x1080 \
    --game-pose=15.95,12.02,-3.14,166.0,0.0 --material-report local/m2/material_support_v3.tsv
cargo run --release -p void-compare --bin void-compare -- --game <ROOT> --pose=<getviewpos> \
    --proj 1.0,1.77778 --depth <dumpN_depth_r32f_1920x1080.bin> --width 1920 --height 1080 \
    --cull-back --entities maps/campaign/tutorial/tutorial_p.entities --out <dir>
```

Viewer options: `--normals file|computed` (default `file`, validated E019),
`--unsupported checker|hide`, `--shading debug`, `--no-entities`, `--entity-lights`
(approximate light rig, H), `--max-texture` (memory cap, not a compatibility setting).

## Geometry: validated (RX001)

| Property | Result | Evidence |
| --- | --- | --- |
| Scale/units | 1:1 game units | median depth error 0.18-0.64 % |
| Orientation | Z-up, id angles, pitch+ down, not mirrored | variant comparison, 3 poses |
| Transforms | instance 3x4 rows correct | depth match 89-96 % of non-sky pixels (P1, P2) |
| Winding | front = clockwise in file order | 99-100 % of matched samples |
| Mirrored instances | object-space facing kept | P2 62.8 % vs 41.2 % |
| Geometric (face) normals | agree with depth-derived normals | 0.06-0.13 deg median |
| File vertex normals/tangents | decoded and validated (snorm8 + w selector/sign) | E019 |
| Normal-map orientation | not separately validated (green channel, texel-level basis) | open |
| Entity placement | rows are local axes; game-closer 4.22 % -> 1.11 % at P1 | E023 |

Bevy adapter: indices reversed (front faces CCW), mirrored instances use an
inverted-index mesh variant with the same outward normals, back faces culled.

## Materials

Classification (`local/m2/material_support_v3.tsv`; 357 materials = world + entities):

| Class | Count | Rendering |
| --- | --- | --- |
| basic | 134 | arkDefault layered model (docs/formats/arkdefault.md): layer UVs, masks, albedo/normal/gloss/metallic blending |
| partial | 183 | same; ignored features listed per material |
| unsupported | 38 | magenta checker (unlit) or hidden with `--unsupported hide` |
| editor-only | 2 | `editorDrawTrans` volume materials; never drawn (H: invisible in game) |

Unsupported by cause: arkDecal 16, arkVegetation 10, arkVista 7, arkWater 2, arkSky 1,
no program 1, decl not UTF-8 1 (`models/weapons/crossbow/bolt_01_pickup.material`; encoding
not investigated). Ignored features in partial materials (materials): occlusionmap 167,
emissive 19, detail bump 6, sheen 3, translucency 2, refraction 1; texture load failures
for a layer's gloss (4), metallic (2), mask (1) are listed per material.

Validated (E020, RX002 debug views): gloss r 0.888 and metallic r 0.941 at P2.

## Matched comparisons

- RX001/RX002 P1-P2: placement, silhouettes, and texturing agree with the game frames;
  entity props (barrels, boat, crates) now present (`local/m2/entities/p1_entities.png`).
- The exterior door (`ld_door_350cm_ext_poor_metal_01`) fills the world's doorway
  (`local/m2/entities/door_c.png` vs `door_c_noent.png`).

## Remaining differences (classified)

| Difference | Cause | Class |
| --- | --- | --- |
| Lighting, shadows, fog, exposure, tone mapping, sky colour | placeholder light rig; light intensity units, sun direction, probes, env/fog decls not decoded | not implemented |
| P1 mud ground gloss (r 0.48) | unexplained extra contribution (decals or wetness candidates) | open |
| Vertex colour byte order | RGBA vs BGRA evidence conflicts | open |
| Water, sky dome, vista city, decals, vegetation | programs unsupported (deferred by decision unless area D1 needs them) | unsupported program |
| Occlusion, emissive, detail bump, sheen, translucency | not in the transcribed model | partial material |
| Skinned entity models (13: chain, latch, shield door, valve, ...) | `.bmd6model` not decoded | unsupported format |
| Physics props at spawn pose (e.g. floating barrels) | no physics simulation | missing system |
| Entity lights | `--entity-lights` is approximate (lumens from preset wattage, spot axis +X: H); area, parallel, scatter-only lights skipped | approximate |
| 23 world instances slightly off | non-TRS matrices approximated by Bevy `Transform` (max element error 1.6e-2) | adapter limitation |
| P3 (in water) depth mismatch | player state at capture; unresolved | open |

## Next steps (M2)

1. P1 ground gloss and vertex-colour order (captures of a vertex-colour-masked material).
2. Lights/environment decode (intensity units, sun, probes) before lit-colour comparison.
3. Special programs only as area D1 or later slices require them.
