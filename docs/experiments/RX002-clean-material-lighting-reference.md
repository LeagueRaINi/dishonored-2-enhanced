# RX002: clean material/lighting reference for tutorial_p

Authorized: the user's standing go-ahead to run the original game (2026-10-16) and the
instruction to obtain a clean baseline through reversible configuration changes.

## What d2enhanced changes, and what this experiment does about it

Source: `../d2enhanced/src/config.cpp` (settings), `fsr.cpp` (capture), `README.md`.
"Capture stage" = the texture d2enhanced's F8 dumps as `color`: FSR's input, i.e. the
engine's lit HDR scene colour before TAA and post-processing.

| Mod feature | Effect on the capture stage | RX002 setting |
| --- | --- | --- |
| FSR (replaces TAA) | none before the stage; F8 dumps exist only while FSR runs | `Enabled=1`, `Mode=native` (capture hook only) |
| Halton jitter (32-phase) | sub-pixel projection jitter | `HaltonJitter=0` (engine 8-phase) |
| MipBias | texture LOD bias while upscaling | `MipBias=0` |
| Render scale | render resolution | `ForceScale=0`, `RenderScale=1.0` |
| SSGI | **added to FSR's input colour** | `Enabled=0` |
| SSR | replaces cubemap reflections in the lit colour | `Enabled=0` |
| Material capture / decals | patches 3,700+ forward pixel shaders (extra outputs) | `Capture=0`, `Decals=0` (unpatched shaders) |
| HDR output, grading | after the stage (tonemap/output) | `Enabled=0` |
| UltraPlus | engine cvars beyond the menu's Ultra (shadows, LOD, filtering) | `UltraPlus=0` (mod restores originals) |
| Cvar unlock, console, F8, frame limiter, mouse fixes, intro skip | no rendering change | unchanged |

Remaining mod influence at the capture stage: none known. The device hooks stay
installed (needed for F8); this is recorded as a residual risk, not assumed harmless.

Game display configuration is set to windowed 1920x1080 (as in RX001) so the window can
be captured and the projection matches RX001. Graphics quality otherwise unchanged.

## Procedure

1. Back up and hash: saves, `dishonored2Config.cfg`, `profile.bin`, `d2enhanced.ini`,
   `d2enhanced/captures/*_info.txt`.
2. Write the RX002 `d2enhanced.ini` and windowed display settings.
3. Launch, Tutorial from the main menu, mod console: `g_showHud 0`, `pm_noBob 1`.
4. Poses: RX001 P1 and P2 positions via `setviewpos`; actual pose read with `getviewpos`.
5. Per pose: F8 with default settings; then debug-view trials (`r_skipFog 1`,
   `r_showGlossOnly 1`, `r_showMetalOnly 1`), each followed by F8 and reset.
6. Quit with `quit`; restore every backed-up file; verify hashes.

Recorded per capture: pose, projection, exposure constants (mod capture log), cvars set,
render size, dump indices, executable/mod/config hashes.

Stopping condition: both poses captured, or any unexpected file change, crash, or
blocked step.

## Execution log (2026-10-16)

- Pre-run hashes equal RX001's baseline (manifest `local/backup/rx002_*/manifest_before.txt`).
- Mod log confirmed the clean state: "0 shaders patched", HDR off, no Ultra+ overrides,
  no Halton jitter; FSR 4.1.1 native 1920x1080 as capture hook only.
- Main menu -> Tutorial -> in game. HUD off, `pm_noBob 1`. Poses via `setviewpos`.
- After `quit`: every save, `dishonored2Config.cfg`, `profile.bin`, `d2enhanced.ini`, exe and
  mod DLL matched the pre-run manifest; `profile.bin` was not modified by this run.
  16 mod capture info files created beyond the existing numbering were moved to
  `local/ref/RX002/info/`; the mod's captures folder is back to its 143 files.

## Captures (`local/ref/RX002/P{1,2}/`)

Projection as RX001 (P0 1.0, P5 1.77778, near 0.01, reversed-Z). Four frames per F8.

| Pose | getviewpos | base | r_skipFog | r_showGlossOnly | r_showMetalOnly |
| --- | --- | --- | --- | --- | --- |
| P1 | (15.95, 12.02, -3.14) yaw 166.0 pitch 0.0 | dump0-3 | 4-7 | 8-11 | 12-15 |
| P2 | (48.30, -40.01, 22.13) yaw 90.0 pitch 0.0 | dump16-19 | 20-23 | 24-27 | 28-31 |

Frame-to-frame difference of consecutive base frames (jitter/animation): mean 0.0025-0.0046
(P1). Debug views change the frame far beyond that (P1: gloss 0.08, metal 0.015-0.023,
skipFog 0.006-0.013 mean abs), and previews confirm the intended content.

## Observed movement parameters (cvar values in this session)

`pm_walkspeed 2.5`, `pm_runspeed 4.375`, `pm_crouchspeed 2.5`, `pm_backSpeedRatio 0.75`,
`pm_stepsize 0.25` ("maximum height the player can step up without jumping"),
`pm_jumpheight 0.8125`, `pm_normalviewheight 1.359375`, `pm_crouchviewheight 0.71875`,
`pm_maxviewpitch 89` ("look down"), `pm_minviewpitch -89`, `pm_noclipspeed 400`,
`pm_sprintspeed` about 7.03 (partially obscured in the capture; unconfirmed).
Non-default render cvar noticed: `r_water_ScreenSpaceMaxStepsCount` (highlighted as changed).

## Use and limits

- Material stage: gloss/metal debug captures validate the transcribed material model
  (docs/formats/arkdefault.md, E020).
- Lit colour (base dumps) is the engine's lit HDR scene colour before TAA/post with mod
  shading effects disabled; not yet compared, because the viewer's lighting is not the
  game's. Exposure: per-frame auto exposure (ELF) is not recorded in the dumps.
