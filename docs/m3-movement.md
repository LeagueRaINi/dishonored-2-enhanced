# M3: movement prototype and RX003 replay

Updated 2026-10-16. Crate `crates/void-sim` (engine-independent): `world.rs` (static
collision from decoded original shapes; `parry3d` 0.31.1 for queries only), `load.rs`
(physworld + entity physhapes, e.g. the door, re-posable about the hinge), `player.rs`
(kinematic player prototype), `replay.rs` + `void-sim` binary (RX003 replay).

**Status: prototype, not recovered original behaviour.** The movement routine is a
stand-in until the executable's routine is traced. Parameters: RX003 measurements (M),
cvars (C), hypotheses (H) — see `Params` in `player.rs`.

## Reproduce

```
cargo run --release -p void-sim -- --game <ROOT> --ground capsule|axis [--verbose]
```

Input data: `crates/void-sim/data/rx003_run1.tsv` (51 runs; our own measurements, extracted
from the local RX003 log by `tools/rx003/extract_runs.py`). Each run: teleport to the
logged settled start, settle, hold the input for the nominal time at a fixed 120 Hz tick,
settle 1 s, compare the eye position with the logged `getviewpos`.

## Model

- Player: vertical capsule, radius 0.36 (M), origin = bottom sphere centre, eye = origin +
  1.359 (C; total 1.72 above ground, M), top sphere 1.0 above origin (H).
- Speeds: run 4.14, sprint 7.1 after 0.25 s, start lag 0.02 s (M, fitted to segments A/B).
- Grounded move (H, id Tech style): velocity clipped to the ground plane, collide-and-slide,
  step-up (`pm_stepsize` 0.25, C) only when blocked by a steep surface, ground trace.
  Walkable limit 50 deg (H), gravity 9.81 (H).
- Ground support: `capsule` (bottom sphere rests on the surface) or `axis` (ray under the
  axis; capsule only blocks above the step height).
- All 12 static bodies collide (layer selection unknown); the door collision box follows
  the entity transform, opened = -90 deg about its local Z (inward, observed in RX003).

## Results (RX003 run 1, max |error| of eye position, metres)

| Segment | Role | capsule xy / z | axis xy / z |
| --- | --- | --- | --- |
| A run (12) | fit (speeds, lag) | 0.027 / 0.030 | 0.044 / 0.034 |
| B sprint (9) | fit (speeds) | 0.029 / 0.031 | 0.068 / 0.034 |
| C gate block (3) | fit (radius) | 0.004 / 0.015 | 0.004 / 0.017 |
| E door block (3) | fit (radius) | 0.010 / 0.029 | 0.010 / 0.031 |
| F open door, walk through (3) | prediction | 0.001 / 0.009 | 0.001 / 0.012 |
| D stairs profile (20) | prediction | 0.091 / 0.150 | 0.109 / 0.098 |
| P1 spawn mud downhill (1) | prediction | 0.039 / 0.024 | 0.039 / 0.023 |

Findings:

- **Velocity clipping explains the stair speed (D):** the game's horizontal speed on the
  34 deg ramp is 2.85-2.95 m/s = 4.14 x cos^2(34 deg). A floating (lift-and-drop) mover
  overshot by up to 0.74 m; clipping brings stairs to <= 0.09-0.11 m. Supports an id Tech
  `PM_ClipVelocity`-like rule (H until traced).
- **Vertical support on slopes is not reproduced:** the game rests ~1.67-1.69 m eye above
  the ramp (vs 1.71-1.72 on flat); a resting sphere gives +0.15, an axis probe +0.05..0.10.
- Door passage, door and wall blocking, and the downhill spawn run agree within 1-4 cm.
- RX003 tolerances (docs/experiments/RX003): flat/blocking pass; stairs fail vertically for
  both models.

## Walk mode (viewer)

`void-viewer --game <ROOT> --walk [--ground capsule|axis] [--game-pose x,y,z,yaw,pitch]`.
`walk.rs`: input in Update -> player tick in FixedUpdate at 120 Hz -> camera at the eye in
Update (one frame input latency). Mouse look (cursor locked; Esc frees, click relocks),
WASD (back 0.75x: `pm_backSpeedRatio`, C; strafe = run speed, H), Shift sprint, Space jump
(`pm_jumpheight` 0.8125 with gravity 9.81: H), F opens/closes the door within 2.5 m when
facing it (collision and render surfaces swing -90 deg at 100 deg/s, approximating RX003),
V toggles walk/fly. Smoke checks: spawn eye -3.118 vs game -3.14; door view matches the
RX003 frame. Not implemented: crouch, lean, mantle, climbing, powers, AI, other doors.

## Next

Trace the original movement (ground trace and slope handling, step-up, character shape,
collision layers) in the executable; drive the player from the viewer through an explicitly
ordered fixed-timestep Bevy system (walk mode) for interactive checks.
