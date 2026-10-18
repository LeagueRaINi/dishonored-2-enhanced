# RX004: R1 injection scaffold, read-only scene probe

Status: **run 1 done 2026-10-16** (user installed the DLL and ran the game with their own
config; d2enhanced active). R2 snapshot build prepared.

## Run 1 result (user session, `void-rs/local/inject/void-inject.log`)

- Loaded; exe hash gate passed; `RenderAllViews` hooked at image base + 0x35DC60 (prologue
  `48 8B C4 48`); ~9,700 frames at ~120 fps without errors or visible effect.
- Tutorial view: 1 view request; world with **537 render models and 160 lights** (160 =
  the `idLight` entities parsed from `tutorial_p.entities`, E021).
- Camera: hook `vieworg` (15.94 12.04 -3.14), view axis (-0.970 0.244) = yaw 165.9;
  user's `getviewpos` at the same time: (15.94 12.05 -3.14) yaw 165.9 pitch 0.5. Position
  and yaw agree to the 0.01 display precision; pitch differs slightly (-0.001 axis z vs 0.5 deg,
  sampled at different moments: the log is 1 Hz).
- `fov_y` 1.025 rad = 58.7 deg (= RX001 projection), `fov_x` 1.862 rad (3440x1440 window).
- A loading/probe view earlier had 245 models, 1,536 lights, fov 90x90 deg (U: what it is).

R1 gate: loads, hooks, reads the live scene correctly, no visible effect. Passed for this run.

## Keys (current build)

F3 = R2 snapshot, F4 = R3a render (moved from F9/F10: F9 is quickload, F10 is d2enhanced's FSR debug view).

## R2 snapshot (prepared)

F9 writes `void-rs/local/inject/snap_<time>_f<frame>/{view,models,surfaces,lights}.tsv`:
per model name, origin/axis/scale, world AABB, model matrix, flags, surfaces (material name,
geometry, binary model id, joints, index range); per light flags, origin, axis, linear
colour, type, distance falloff/cutoff, cone cosines.

## R2 snapshot 1 (user session, F9 at spawn after a quick reload)

`local/inject/snap_1792172792_f2393`: taken inside `RenderAllViews` after the original call
(a rendered tutorial frame, camera at spawn, fov as before): 529 models, 160 lights.

- **401 / 529 render models match a parsed `.entities` entity** with the same model source
  and position within 2 cm (idEntity 154, idEntityFx 106, idDynamicEntity 45, arkPickup 21,
  arkAreaVolume 16, arkMovable 14, idWorldspawn 13, arkBreakable 12, arkUsable 9, ...).
  Unmatched: runtime pieces of breakable planks (55), moved physics props (barrels, buoys,
  wagon doors), animated rats, 23 unnamed models (U).
- **160 / 160 lights match `idLight` entities by position.** Types: 2 (spot) 122, 0 (point)
  27, 3 (8), 1 (2), 4 (sun, 1). Runtime linear colours are absolute engine values, e.g. sun
  645.8, street lamp spot (272, 115, 25) with cutoff 14: this resolves the M2 light-intensity
  unknown (values are now readable per frame).
- Surfaces: only skinned `.md6` models have entries in `m_surfaceList` (material, geometry,
  GPU joint buffer, index range; e.g. Emily's body with `player_body_hidden.material`).
  Static `.lwo` models report 0: H, they use the shared list behind `m_surfaceListPtr`.
- The static world (`.bwm`) is not among the render models (H: separate world-model path).

## R3a live test (user session, frame 7487)

F10 at (-16.79, -8.44, 4.31), F8 (d2enhanced dump 12, native 3440x1440, same frame number
7487) and F9 (snapshot f7741, same spot). Our render: 2560x1080, same vertical fov; game depth
resampled through both projections. Log: D3D12 init 21 ms, static world load 0.29 s, first
render 2.3 s (incl. entity model decoding), later renders ~50-70 ms.

| Render | within 1 % of game depth | game closer | ours closer |
| --- | --- | --- | --- |
| static world only | **94.0 %** | 0.2 % | 5.7 % |
| world + all named static models (unfiltered) | 59.7 % | 0.1 % | 40.2 % |
| world + filtered models (`m_mayBeDrawn` and non-editor material; `r3a_snapshot`) | **94.3 %** | 0.2 % | 5.5 % |

- Unfiltered failure cause (O, snapshot): models the game holds but does not draw:
  `m_mayBeDrawn` = 0 (102 `collision_1meter` helpers, 98 unnamed, hidden player body) and
  16 area/water volumes with `editors/arkane/volumes/*` materials. Filter now in `r3.rs`.
- Remaining "ours closer" (mismatch map `local/ref/RX004/r3a_mismatch.png`): alpha-tested
  vegetation drawn as solid cards (no alpha test in the depth-only pass yet) and resampling/
  TAA-jitter edges. No prop is in front of the camera in this view: props remain to be
  validated at a prop-rich view.
- Static models without a decodable `.bmodel`: 138 of 502 named models (U: resource naming
  for those; to inventory).

## R3a live test 2: spawn with props (F4 frame 3309, F8 dump frame 3416)

Same spot (spawn, camera unchanged between the two frames within ~1 s; floating barrels
move with physics). Filtered models: 278 drawn, 28 without a decodable `.bmodel`.

| Render | within 1 % | game closer | ours closer | game sky / ours geometry |
| --- | --- | --- | --- | --- |
| static world only | 81.0 % | 6.7 % | 4.6 % | 7.6 % |
| world + live static models | **86.0 %** | **0.9 %** | 5.5 % | 7.6 % |

Props validated: adding the live entity transforms removes almost all "game closer" error
(barrels, boat, posts, the physics-moved barrel in place). Remaining classes
(`local/ref/RX004/spawn/mismatch.png`): animated rats (skinned `.md6`, not rendered), sky dome
(game: infinite depth) and vista city (game: different depth treatment), sampling/jitter edges.

## Target and purpose

Load `void-inject` (`crates/void-inject`, built as `dinput8.dll`) into the original game,
verify the executable hash gate, install the read-only `RenderAllViews` hook, and check that
the logged camera (`renderView_t.vieworg`) equals `getviewpos` and that model/light counts
are plausible. Gate R1: loads and unloads cleanly, no visible effect on the frame.

## Procedure

1. Back up and hash saves, `dishonored2Config.cfg`, `profile.bin`, `d2enhanced.ini` (as RX003).
2. Install the RX002 windowed/clean configuration; d2enhanced stays loaded (its console is
   needed for `getviewpos`; the hook is read-only and does not touch d2enhanced's hooks).
3. Copy `void-rs/target/release/dinput8.dll` to the game folder (there is no existing
   `dinput8.dll` there; the proxy forwards `DirectInput8Create` to `System32\dinput8.dll`).
4. Launch, enter the Tutorial, read `getviewpos` at two poses; compare with
   `void-rs/local/inject/void-inject.log` (1 Hz camera/count lines).
5. Quit; **delete `dinput8.dll` from the game folder**; restore config/ini; verify all hashes.

Stopping condition: both poses compared, or any crash, input malfunction, or unexpected file
change.

## Manual equivalent (for the user)

```
cp void-rs/target/release/dinput8.dll ./dinput8.dll      # install (game folder)
rm ./dinput8.dll                                          # uninstall
```
