# RX003: gameplay reference scenario (movement, blocking, stairs/slope, door)

Status: **executed 2026-10-16 (run 1)**, user go-ahead given for M3. Results at the end
("Run 1 results"). Area D1 and the door are described in docs/m3-collision.md. The plan
below is kept as written; deviations are listed with the results.

## Purpose and comparison target

Produce original-game trajectories and timings in area D1 that a reimplementation must
reproduce: free movement on flat ground, movement blocked by a wall and by the closed
door, climbing the plank stairs (step-up), walking the dirt slope, and opening and passing
the door. RX003 measures the original; it does not validate any implementation by itself.

## Conditions to hold fixed and record

| Item | Value |
| --- | --- |
| Build | `Dishonored2.exe` SHA-256 `b671f824...6180` (E001); record d2enhanced DLL hash |
| Mod state | RX002 clean settings (no SSGI/SSR/patched shaders); FSR capture hook optional. Frame limiter state recorded |
| Display | windowed 1920x1080 (RX001/RX002 config), V-sync and FPS cap recorded |
| Scene | main menu -> Tutorial (fresh map load; no save loaded; saves untouched) |
| Difficulty / settings | as in the user's profile; record `dishonored2Config.cfg` hash |
| Cvars | `g_showHud 0`, `pm_noBob 1`; record all `pm_*` values (`listCvars pm_` or console reads) |
| Input | default binds (O, `base/default_keybinds.cfg`): W forward, S back, A/D strafe, Shift sprint, C crouch, Space jump, F interact. Synthetic key down/up via `tools/refcap.py key` with timestamps logged |
| Timestep | game is variable-rate: log wall-clock time of each key edge and each `getviewpos`; record average FPS (`com_showFPS` if available) |
| Measurement | `getviewpos` (clipboard) at each checkpoint; window screenshot at each checkpoint; optional F8 depth dump at S0 for camera validation |

## Prerequisite checks (stop if any fails)

1. Synthetic keyboard input moves the player (not yet observed; mouse look did not work
   in RX001). If keys do not work, stop and record; do not substitute console movement.
2. Relationship between `getviewpos` z and the floor: at S0 after settling, read
   `getviewpos` standing and crouched. Render ground is known from `void-probe`; RX001 P1
   showed view z 1.81 above render ground (P2: ~1.9) vs `pm_normalviewheight 1.359375`, so
   the offset is unexplained (U).
3. Facing: `setviewpos` applies position but not pitch (RX001); confirm yaw is applied.

## Scenario (each segment starts from a `setviewpos` reset)

Poses are candidates from render geometry; confirm each in the viewer (`void-viewer --game-pose`) before the run. They use the `setviewpos x y z yaw pitch` console form; z = render ground + offset
measured in check 2 (initially ground + 1.81, then corrected). Ground heights from
`void-probe` (render geometry, D).

| Seg | Start (x, y, ground z) yaw | Input | Expected observation (to be measured) |
| --- | --- | --- | --- |
| A Flat walk | (528.0, -8.0, ~10.7) yaw 90 | W for 1.00 s | distance, speed; accel/decel from intermediate `getviewpos` at 0.25 s steps (repeat run) |
| B Sprint | same | Shift+W 1.00 s | sprint speed (cvar ~7.03 unconfirmed) |
| C Blocked by wall | landing (523.0, -3.0, 11.75) yaw 90 (+Y) | W 2.0 s | stop position vs the static metal gate in the brick wall (render face at y ~ -1.0; world geometry, not an entity); any sliding |
| D Stairs up | (523.0, -7.0, 10.71) yaw 90 (+Y, up the stairs) | W until y >= -3.5 or 3.0 s | z profile over 5 risers (10.71 -> 10.91 -> 11.10 -> 11.29 -> 11.49 -> 11.70 at y -5.75..-4.25, treads 0.25); step-up smoothing |
| E Blocked by closed door | landing (522.5, -3.5, 11.72) yaw 180 (door centre) | W 1.5 s | stop distance from the door's outer face (x ~520.87) |
| F Door interaction | same as E | F tap, wait 2.0 s, screenshot; then W 2.0 s | time to open (screenshots at 0.25 s), swing direction (hinge expected at y -2.76, H), passage into interior (x < 520.6) |
| G Slope | (523.25, -12.0, ground) yaw 90 (up-slope direction to be confirmed from `local/m2/entities/probe_door.tsv`) | W 1.5 s | z profile; speed on slope vs flat |

Stair and landing heights are render-geometry samples (`void-probe`, 0.25 grid; table
reproduced from `probe_door.tsv`); collision may differ and is part of what RX003 measures.
Viewer checks of the poses: `local/m2/entities/rx003_d.png` (stairs ahead, door top left),
`rx003_e.png` (door centred, handle on the -Y edge). Rejected C candidates:
`rx003_cw.png` (camera inside a post), `rx003_c.png`/`rx003_c2.png` (table of loose props
in the path; props may be physics objects). The gate face for C comes from
`local/m2/entities/probe_gate.tsv`.

Each segment is run 3 times; report mean and spread. Door state must be reset between
runs of F (reload the map from the menu; do not save).

## Tolerances (initial; revise after the first run's spread)

- Position after a timed input: within max(0.05 m, 3x observed run-to-run spread).
- Blocked stop distance: within 0.03 m.
- Stair z profile: per-checkpoint within 0.05 m; final z within 0.02 m.
- Door: open/close completion time within 2 frames at the recorded FPS; final angle within 2 deg
  (screenshot-based estimate, noted as approximate).

## Protections and stopping condition

Back up and hash saves, `dishonored2Config.cfg`, `profile.bin`, `d2enhanced.ini` before
the run; restore and verify afterwards (procedure of RX002). Never save the game.
Stop after all segments have 3 runs, or on the first crash, unexpected file change,
failed prerequisite, or ambiguous measurement. Outputs: `local/ref/RX003/` (logs,
clipboard reads, screenshots, timing CSV).

## Run 1 results (2026-10-16, 18:36-18:55)

Raw log: `local/ref/RX003/log.txt` (every key edge, measured hold, and `getviewpos`);
screenshots and door frame sequences in `local/ref/RX003/`. Scripts: `tools/rx003/rx3.sh`
(setviewpos / timed hold / getviewpos), `tools/rx003/door_seq.py` (F + timestamped frames),
`tools/rx003/motion.py` (frame-difference timing), `tools/refcap.py key SHIFT+W` (chords).

Conditions (O): exe `b671f824...6180`, d3d11.dll `0d403bd2...7b5d`; RX002 clean mod ini
(`c5659fc2...`) and windowed 1920x1080 config (`279cd61b...`) installed from the RX002
snapshot; `FpsLimit=-1`, `r_swapInterval 0`; overlay showed **120 fps (8.33 ms)** throughout.
The game started directly in a fresh Tutorial (objective "Mobility Training") without the
main menu (U: why). Protections: saves, profile, config, and ini backed up and hashed
(`local/backup/rx003_*`); after `quit` every file matched the pre-run manifest after
restoring the two installed files; the game wrote no save/profile changes; mod captures
folder unchanged. Positions below are `getviewpos` (2 decimals), file space.

### Prerequisites

1. Synthetic keys move the player (W 1.0 s from spawn: 4.07 m along yaw 165.9). Passed.
2. `getviewpos` z vs collision ground: natural spawn (15.94, 12.04, -3.14) over collision
   z -4.834 -> **1.69 m**; settled at (528, -7.95) 12.40-12.41 over collision 10.692 ->
   **1.71-1.72 m** (render ground is 0.11 m lower at spawn). Not `pm_normalviewheight` 1.359.
3. `setviewpos` applies yaw (all runs report the commanded yaw); pitch stays 358.5.
   After `setviewpos` the player settles (falls) onto the ground; starts are logged.

### A/B: flat run and sprint (start 530.95, -8, yaw 180; 3 runs each; spread <= 0.03 m)

| Hold (ms) | Run distance (W) | Sprint distance (Shift+W) |
| --- | --- | --- |
| 250 | 0.97 | 0.97 |
| 500 | 2.00-2.03 | 2.74 |
| 1000 | 4.07-4.10 | 6.31 |
| 1500 | 6.14 | - |

Run: steady **4.14 m/s** (linear in hold time, effective lag ~0.015 s; cvar
`pm_runspeed` 4.375). Sprint: first 0.25 s equals running, then **~7.1 m/s** (cvar
`pm_sprintspeed` ~7.03). Hold times are nominal Python sleeps (process overhead excluded
from the hold; the logged "measured" values include ~95 ms process start).

### C/E: blocked movement (3 runs each, identical)

- C (landing 523, -2.95 -> +Y, 2 s): stops at **y -1.85**; collision gate face at y -1.493
  (ray test on decoded collision) -> clearance **0.357 m**.
- E (522.45, -3.5 -> -X into the closed door, 1.5 s): stops at **x 521.13**; door collision
  surface (physhape box + entity transform) at x 520.77 -> clearance **0.36 m**; no static
  collision behind it within 4 m.
- H: horizontal player radius ~0.36 m around the `getviewpos` axis.

### D: stairs (start 523, -7 -> +Y; 2 runs per hold, spread <= 0.03 m)

| Hold (ms) | rest y | rest z | z - 1.71 |
| --- | --- | --- | --- |
| 250 | -5.95..-5.98 | 12.41 | 10.70 (ground) |
| 300 | -5.78 | 12.40-12.41 | 10.70 |
| 350 | -5.62 | 12.48-12.49 | 10.77 |
| 400 | -5.45..-5.48 | 12.58-12.59 | 10.87 |
| 450 | -5.33 | 12.68 | 10.97 |
| 500 | -5.19 | 12.77 | 11.06 |
| 550 | -5.05 | 12.86-12.87 | 11.15 |
| 600 | -4.91 | 12.96 | 11.25 |
| 700 | -4.62 | 13.15-13.16 | 11.44 |
| 900 | -3.96 | 13.44 | 11.73 (landing) |

The rest height rises **linearly at 0.67 m per m** (no tread plateaus): the player stands
on the body-1 ramp slab (~34 deg), not on the per-step boxes (docs/m3-collision.md).
Relative to the ramp surface the eye is ~1.67-1.69 m (vs 1.71 on flat ground; U).

### G: slope

The planned slope (render ~12.6 deg at 523.25, -10.25) is **flat in collision**
(10.69-10.77); the incline exists only in the render mesh. Slope behaviour is covered by
segment D (34 deg ramp). A steeper-slope / slope-limit test remains open.

### F: door (`ld_door_350cm_ext_poor_metal_01`)

- From (522.45, -3.5), yaw 180 (1.7 m from the door), the HUD shows "Door - F Open"; F
  opens it without a player-side delay visible in the first frames (first grabbed frame
  0.09 s after key down already differs).
- The leaf swings **away from the player (inward, -X), hinge on the +Y edge (y -2.76)**:
  confirms the hinge hypothesis from the model origin.
- Opening settles (frame-difference below the noise level) after **0.73-1.31 s** (3 valid
  runs; noisy metric). Visual progress in run 1: ~30 % at 0.28 s, ~60 % at 0.46 s, ~80 % at
  0.64 s, fully open by ~0.9 s.
- After opening, W 2 s passes through to **x 517.46** in all 3 runs (interior).
- Closing works from outside only when aiming at the open leaf (yaw ~166 from the same
  spot); a close attempt from inside at (518.6, -3.5) facing +X did nothing. Close-timing
  frame differences never settled (U: flicker); not quantified.
- Deviation: runs F2/F3 of the first batch started with the door still open and are
  marked invalid in the log; F_r2v/F_r3v are the valid repeats.

### H: crouch (flat, 528, -7.95)

Stand 12.40 -> crouch **11.73** (Delta 0.67 m; cvars give 0.64) -> stand 12.40.
H: origin ~0.35 m above the ground (1.71 - 1.359 standing; 1.04 - 0.719 crouched = 0.32),
matching the 0.36 m radius: e.g. the centre of a capsule's bottom sphere. Not verified in code.

### Not measured in run 1

Jump height/airtime, acceleration curve inside 0.25 s, strafing/backward speeds,
walk (non-run) speed, door close timing, slope limit, collision against dynamic props.
