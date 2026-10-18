# M3 prerequisites: collision resources, representation, integration strategy

Updated 2026-10-16. Status: **collision data decoded and compared; original behaviour measured
(RX003 run 1). No movement, query, or door code is implemented yet.**
Render meshes are not collision; a generic character controller is not the original
movement. Neither is treated as validated original behaviour.

## Selected bounded area and door

Area **D1, "exterior door courtyard"**: file-space box x 505..535, y -20..10, z 8..20
(tutorial_p, group 0). Chosen because it contains, within ~15 m:

| Feature | Original identity | Location (file space) | Source |
| --- | --- | --- | --- |
| Door | `ld_door_350cm_ext_poor_metal_01` (`arkDoor`, ordinal 610) | origin (520.73, -2.76, 11.73); local X -> world -Y, local Y -> +X; leaf bounds local x 0..1.5, y +-0.134, z 0..3.5 -> world y -4.26..-2.76, x 520.60..520.87, z 11.73..15.23 (origin edge = hinge side, H) | `.entities`, bmodel bounds (O) |
| Stairs (step candidate) | world instance 5375, material `street_blocker_planks_01_dirty_stairs` | x 522.0..524.25, rising in +Y over y -5.75..-4.25: 10.71 -> 10.91 -> 11.10 -> 11.29 -> 11.49 -> 11.70 (risers 0.19-0.21, treads 0.25); landing 11.70-11.75 in front of the door | `void-probe` on render geometry (D) |
| Slope candidate | world instance 17277, `concrete_dirt_blend_01` | around (523.25, -10.25), ~12.6 deg mean | `void-probe` (D) |
| Blocking wall | static metal gate in the brick wall at the landing's +Y end | render face y ~ -1.0, x ~522.5..524 | `void-probe`, viewer (D) |
| Interior behind door | room with wooden floor and pipes | x ~516..520 | viewer capture `door_a.png` (D) |

The door is entity-only geometry: without entities the world model has an open doorway
of the same size (`local/m2/entities/door_c.png` vs `door_c_noent.png`, E023).
Riser heights (0.19-0.21) are below the observed `pm_stepsize 0.25` (RX002), so the
stairs should be walkable without jumping; that is a prediction to test, not a fact.

Door behaviour inputs (O, `.entities` + decls): component `cpntOpenable`
`map_cpntopenable_decl_2` (hinge attachment tag `hinge`, animated part
`models/interactive/doors/doors_structure/metal_door_structure_animated_single.def`,
player animations `player_open_cw/ccw`, locked/blocked usages), `rigidbody_door.cpntrigidbody`,
`navmeshtoggle_door`, AI components. No lock state is set in the map (U: default lock state).

Reproduce: `void-probe --game <ROOT> --region=505,-20,535,10 --ztop 16 --zmin 0 --step 0.25`.

## Collision resources (O: tutorial_p group-0 index)

| Resource | Kind | Size | Content |
| --- | --- | --- | --- |
| `generated/physworld/maps/campaign/tutorial/tutorial_p/tutorial_p.bphysworld` | `physworld` | 9,560,286 | static world collision (all world instances) |
| `generated/models/.../door_350cm_ext_poor_metal_01.lwo.hkt._100_100_100.10.hkx` | `physhape` | 2,132 | door collision shape |
| same stem `.massProps` | `massprops` | 568 | `hknpShapeMassProperties` (compressed) |
| `generated/maps/campaign/tutorial/tutorial_p/<volume>.cvd._100_100_100.0.hkx` (46) | `physhape` | 4-15 KB | trigger/audio/water volume shapes (`ld_trigger_*`, `audio_*`) |
| `generated/navmesh/maps/campaign/tutorial/tutorial_p.bnavmesh` | `Navmesh` | 2,165,975 | AI navigation (M4), not player collision |

Physhape naming (O): `<model source>.hkt._<sx>_<sy>_<sz>.<n>.hkx`, where the triple is
the scale in percent (`_100_100_-100` occurs for mirrored placements); `<n>` is 10 for
the door and 0 for volumes (U: meaning, perhaps a shape/LOD or collision-layer variant).

### Havok tagfile container (O)

Both files embed Havok **2016.1** binary tagfiles (`TAG0` with `SDKV` = `20160100`), the
`hknp` ("Havok Physics" new-generation) API. Section header: u32 big-endian, top 2 bits
flags (0 = container, 1 = leaf), low 30 bits size including the 8-byte header, then a
4-character tag. Inspect with `python -I tools/hktag.py <file>`.

- **Door `.hkx`**: 5-byte game prefix, then `TAG0 { SDKV, DATA, TYPE { TSTR TNAM FSTR TBOD TPAD }, INDX { ITEM } }`.
  Objects: one `hknpConvexPolytopeShape` with 8 vertices and 6 faces (H: a box around the
  door leaf). 44 types.
- **`.bphysworld`** (layout D, every byte accounted for):

  ```
  u32 n (21) | n x { u32 len, bytes }          contact-system material names ("contactsystem/env.stone", ...)
  u32 m (101) | m x u32                        index into the names (H: per Havok material id)
  u32 size (11272) | TCM0 { TCID, TYPE { TPTR TSTR TNAM FSTR TBOD THSH TPAD } }   type compendium
  u32 size (9547964) | TAG0 { SDKV, DATA, TCRF, INDX { ITEM PTCH } }               objects; types via TCRF
  ```

  Objects (item counts): 1 `hknpPhysicsSystemData` with 12 `bodyCinfoWithAttachment`
  bodies; 1,592 `hknpCompoundShape` (365 `hknpDynamicCompoundShapeData`); 8,661
  `hknpConvexPolytopeShape`; 61 `hknpCompressedMeshShape` (17 with their own
  `hknpCompressedMeshShapeData`); 2 `hknpConvexShape`. Static collision is therefore
  mostly **convex hulls in compound shapes plus quantized triangle meshes**, organized
  as 12 bodies; it is not the render mesh.

Contact materials link to footstep/impact behaviour (`contactsystem/*` decls; not inventoried).

### Decoded collision (2026-10-16; format: docs/formats/havok.md)

- Generic tagfile reader validated against the file's own pointer-patch table (E026).
- Whole map flattened: 15,736 convex hulls + 61 compressed-mesh instances (51,122
  triangles); 5 far-away mesh sections (flags 1) unsupported and reported (E027).
- **Door:** collision = one box, door space x 0.01..1.49, y +-0.03, z 0.01..3.49, convex
  radius 0.01 -> surface x 0..1.5, y +-0.04, z 0..3.5. Every render vertex of the door
  inside |y| <= 0.04 lies within it; the only render geometry outside is the handle
  (x 1.33-1.47, z 1.05-1.57), which has no collision (E028).
- **Area D1, render ground vs nearest upward collision surface** (`void-probe --collision`,
  0.25 grid, 3,531 render ground samples, front faces): median |dz| 0.015 m; 81 % within
  0.10 m. Outliers: junk piles (no collision within 8 m vertically: walkable render
  clutter without collision, or collision through a layer not decoded; U), wall tops.
  **Stairs have two collision representations** (column at x 523): body 9 has one box
  per step (items 11267-11271, tops 10.924 / 11.125 / 11.308 / 11.511, within 2 cm of
  the render treads), and body 1 has a single inclined slab over the stairs (item 465,
  top 11.062 at y -5.25 rising ~0.67 m per m, i.e. ~34 deg, to 11.566 at y -4.5). H:
  bodies are collision layers (e.g. a smoothed ramp for characters vs exact steps for
  objects/traces). RX003 segment D (z profile on the stairs) distinguishes step-up from
  ramp motion; the stairs' slope (~34 deg) also bounds any slope limit if the ramp is used.
  **RX003 result:** the player rests on the ramp (linear 0.67 m/m rise), so the character
  collides with body 1; whether it also collides with body 9 is not distinguishable here.
- **Spawn area** (x 0..30, y 0..30): median |dz| 0.081 m, 56 % within 0.10 m; the mud
  ground `blend_mud_dry_rock_01` collision deviates -0.27..+0.38 m (p10..p90) from the
  render surface; at RX001 P1 collision is 0.11 m above render ground.
- Body layers (`collisionFilterInfo`) are not interpreted; all bodies are included.

### Remaining unknowns

- Which bodies/layers block the player (vs camera, AI, projectiles); the 101-entry
  material map and per-primitive shape tags. Bodies (index: `collisionFilterInfo`, hulls):
  0: 0x0000e000 (1), 1: 0x0000e800 (668, includes the stair ramp), 2: 0x01880702 (37),
  3: 0x0188ec01 (164), 4: 0x01c8ef00 (298), 5: 0x02010002 (54, compressed meshes),
  6: 0x03812401 (5), 7: 0x03890702 (185), 8: 0x03c12701 (16), 9: 0x03c90702 (2,651,
  includes the per-step boxes), 10: 0x03c9e701 (216), 11: 0x03c9ef01 (11,502). Bit
  meaning not recovered (needs the game's filter setup in the executable).
- How the player is represented (capsule/cylinder size), step-up, slope limits, ground
  snapping, and the movement timestep: native code in the executable (`idPlayer`/
  `ark*` physics; not traced yet). Only cvar values are known (RX002).

## Integration strategy (proposal)

1. **Decode, do not translate**: a Bevy-independent crate (`void-havok`, created only when
   implemented) that reads Havok 2016 tagfiles generically from `TBOD`/`ITEM`/`PTCH`, with
   checked offsets and bounded allocation, and exposes the needed objects as plain data:
   convex polytopes (vertices + planes), compressed meshes (decoded to triangles with
   their quantization error recorded), compound children with transforms, bodies, materials.
   Validate on the door (8 vertices must bound the door render mesh) and on area D1
   (collision surfaces must lie within a tolerance of render surfaces where both exist).
2. **Query layer**: load decoded shapes into a Rust collision library (candidate: `parry3d`,
   as data structures and queries only) to get ray/shape casts and contacts. This replaces
   Havok's query engine; differences (convex radius, welding, contact tolerances) are
   recorded and measured, not assumed away.
3. **Movement rules from the original**: recover the player movement routine (ground
   check, step-up, slide, slope limit, speeds, acceleration, fixed or variable step) from
   the executable and cvars, and implement it on top of the query layer as an explicitly
   ordered Bevy system. A generic kinematic controller may be used as a temporary
   scaffold only if labelled as such and excluded from validation claims.
4. **Door**: entity mechanism keyed by original identity; state machine and hinge motion
   from `cpntOpenable` and the animated part def; door collision from its physhape moved
   with the hinge transform.
5. **Validate** with RX003 (docs/experiments/RX003-gameplay-reference.md): same start pose,
   input timeline, and measured positions; tolerances defined there.

Havok runtime itself is not used (licensing and availability); only data is read.
