# Map `.entities` and `entityDef` decls

Implementation: `crates/void-formats/src/entities.rs` (engine-independent; uses the decl
tokenizer in `decl.rs`). Viewer adapter: `crates/void-viewer/src/entity_scene.rs`.
Inventory: `void-tool entities --group 0 --name maps/campaign/tutorial/tutorial_p.entities --tsv <out>`.

Sample: `maps/campaign/tutorial/tutorial_p.entities` (group 0, `File` kind, 4,460,385 bytes,
path-loaded, not listed in `.mapresources`; E013).

## Grammar (O: all 1,104 items of tutorial_p and every referenced entityDef parse)

```
Version 6
entity    { entityDef <name> { <body> } }
component { <cpntType> <name> { <body> } }
```

Body entries:

| Form | Example |
| --- | --- |
| scalar | `class = "arkDoor";`, `x = 520.7284545898;`, `m_DistanceCutoff = 15m;` |
| block | `edit = { ... }`, `editorVars { ... }` (with or without `=`) |
| tuple | `color = ( 1, 1, 0, 1 );` |
| type-annotated key | `"idLight::lightType_t" m_Type = "LIGHT_SPOT";` (the last token before `=` is the key) |
| bracketed key | `item[0] = ...`, `item_add["MAl2j2"] = { ... }` (key = tokens from the first bracket token on) |

Sparse arrays: `num = N;` plus `item[i]` entries. Key lookup is case-insensitive, last
occurrence wins. Values with unit suffixes (`15m`) keep the suffix in the tree; typed
accessors strip it (`m` -> metres, H: game units are metres, consistent with E016).

## Inheritance (O)

`inherit = "<path>"` names an entityDef decl stored as
`generated/decls/entitydef/<path without .def>.def.decl` (kind `entityDef`; both
`x.def` and `x` spellings occur). The child's blocks are deep-merged over the parent's;
scalars replace. Chains are followed to the root (e.g. `ld_door_350cm_ext_poor_metal_01 ->
door_350cm_ext_poor_metal_01.def -> entitydef_ref/door_single_parent`). An empty
`inherit = ""` means no parent. tutorial_p: 0 unresolved inherit targets.

`expandInheritance = false` appears on every map entity; its effect is unknown (U).

## Identity

Stable original identity used by the viewer and future gameplay: entity name (unique in
the file) plus item ordinal in file order. `m_componentDecls.item_add["<id>"]` ids are
6-character component keys; `mapComponentDecl = "map_cpnt..._decl_N"` refers to a
`component` item in the same file, `inheritedDecl = "<path>.<cpnt type>"` to a decl.

## Transforms (O + validation)

- `spawnPosition {x y z}`: file/world space (Z up), absent axes = 0.
- `spawnOrientation.mat.mat[r].{x,y,z}`: sparse 3x3; missing elements take identity values.
- `renderModelInfo.scale {x y z}`: local scale, default 1.
- **Rows are local axes** (`AxisConvention::RowsAreAxes`): world = position +
  sum_i(local_i * scale_i * mat[i]). E023: at RX001 P1 the "game closer than viewer"
  depth fraction fell from 4.22 % (no entities) to 1.11 % with this convention; the
  transposed convention misplaced rotated props. `ld_start`/player start corroborate.
- Physics-driven entities (`idDynamicEntity`, e.g. floating barrels) settle at runtime;
  their spawn pose is not their in-game pose (observed at RX001 P1).

## Model references (O: tutorial_p index)

`renderModelInfo.model` source path -> built resource (`entities::model_resource`):

| Source form | Resource | Kind | Status |
| --- | --- | --- | --- |
| `<stem>.lwo` (or other static source) | `generated/model/<lowercase stem>.bmodel` | `model` | decoded (docs/formats/bmodel.md) |
| `<stem>.md6` | `md6Def` decl `generated/decls/md6def/<stem>.md6.decl` -> LOD meshes `generated/basemodel/<mesh stem>.bmd6model` (or `nocloth_*` builds) + skeleton | `md6Def`, `baseModel`, `skeleton` | decoded (docs/formats/md6.md); `entities::model_resource`'s direct `<stem>.bmd6model` guess holds only when the mesh is named like the decl (all 13 in tutorial_p, 73 of 129 eotw snapshot md6) |
| no extension (map-local brush) | `<source>.bmodel` | `model` | decoded |

tutorial_p: 297 entities reference models; all 297 resources exist: 284 static
(80 distinct), 13 skinned (10 distinct: chain, door latch, shield door, valve, fountain,
grindstone, vise, metronome, cables, crossbow).

Map-local brush models of `arkAreaVolume` entities (16: audio/water volumes) use
`editors/arkane/volumes/*.material`, whose program is `editorDrawTrans` (O). The viewer
classifies `editorDraw*` programs as editor-only and does not draw them (H: invisible in
game). `editorVars.showInRenderMode = false` is **not** a usable visibility flag: it is
false on almost every entity (falsified 2026-10-16).

`renderModelInfo.m_customMaterials` (`num` + `item[i]`): 52 surface overrides in
tutorial_p; H: item i replaces surface i's material (viewer applies it; not validated).

## Lights (`class = "idLight"`)

Fields read: `m_Type` (`LIGHT_POINT`, `LIGHT_SPOT`, `LIGHT_PARALLEL`, `LIGHT_AREA`),
`m_Color {r g b}`, `m_DistanceCutoff`/`m_DistanceFalloff` (m), `m_AngularCutoff`/
`m_AngularFalloff` (deg), `m_preset` (`LIGHT_PRESET_40W`, `LIGHT_PRESET_SUN`, ...),
`m_scatterOnly`. tutorial_p: 160 lights. Unknown (U): intensity units/preset mapping,
spot axis (viewer assumes local +X, H), area-light shape, the sun's direction (`p_sun`
has no orientation; likely from environment decls).

## Components

Component types (tutorial_p): `cpntRigidBody` 49 map decls / 251 references,
`cpntOpenable` 3 (the three doors), `cpntTrigger`, `cpntWater`, `cpntConstraint`,
`cpntPatrol*`, `cpntSpawner`, AI/audio components, and others (full counts in
`void-tool entities` output). **No component behaviour is implemented**; the viewer
places static spawn poses only and says so in its log.
