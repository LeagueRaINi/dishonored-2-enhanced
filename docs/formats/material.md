# Material declarations and decl text syntax

Implementation: `crates/void-formats/src/decl.rs` (tokenizer), `material.rs` (material view).

## Location (O)

A world-model batch names a material such as `models/x/y.material`; its decl is the
index entry of kind `material` named `generated/decls/material/models/x/y.material.decl`
(group 0 for tutorial_p; all 323 resolve).

## Syntax (O, 323 tutorial materials)

- One top-level `{ ... }` block. `//` and `/* */` comments, quoted strings.
- Newline-terminated `key value` lines; `key { n, n, n, n }` numeric vectors; nested blocks
  (`options { ... }`, `state { }`); `version 6` in newer decls.
- Texture references appear **with and without** file extensions (`.../x_d.tga` and
  `.../x_d`). They map to `generated/image/<lowercased path without extension>.bimage7`.
- Image programs such as `ipr_constantcolor( 0.5, 0.5, 0, 0)` can stand in for a texture.
- Entity/component decls use `key = value;` instead; the tokenizer handles both styles.

## Tutorial survey (O, `local/m2/materials_tutorial.txt`)

- `mainProgram` (case varies): arkDefault 286, arkDecal 16, arkVegetation 10, arkVista 7,
  arkWater 2, arkSky 1, none 1.
- Slots: diffusemap 323, bumpmap 296, glossmap 287, metallicmap 167, occlusionmap 152,
  emissivemap 19, layer1_* 41-48, layer2_* 20-28, lightmap 7, translucencymap 7, ...
- Options include `alphatest`, `twoSidedLighting`, `extralayer` (0-2), per-layer
  `vtxcolormaskchannel`, `maskmode`, `uvset`, `*reuselayer`, `vtxcolormultiply`, `needswind`.

## Semantics status

Only the slot/option *names* are observed. The meaning of gloss, metallic, occlusion,
layer masks, `wardRoughness`, and `metallicminmax` is not recovered (no shader analysis
yet); see docs/m2-materials.md for what the viewer does with them.
