# RX001: tutorial_p geometry reference captures

Authorized by the user on 2026-10-16 ("go ahead to run the original game").

| Item | Value |
| --- | --- |
| Target | `Dishonored2.exe` SHA-256 `b671f824…6180` (E001), map `campaign/tutorial/tutorial.mapinfo` (`maps/campaign/tutorial/tutorial_p`), Corvo |
| Reference variant | **d2enhanced active** (`d3d11.dll` proxy, native D3D11; SHA-256 in `local/backup/mod_identity.txt`). Its FSR/SSGI/SSR/HDR/grading change shading, not geometry. The mod is not modified; its ini is hashed before and after. |
| Purpose | Matched-camera depth and color references for validating BWM1 scale, orientation, handedness, transforms, winding, normals, and missing surfaces (M2). |
| Launch | `tools/refcap.py launch --game <ROOT> -- +set saveGame_enable 0 +campaign campaign/tutorial/tutorial.mapinfo corvo` with `SteamAppId=403640` in the child environment only. |
| Camera control | Mod console (F2 overlay): `noclip`, `g_showHud 0`, `setviewpos x y z yaw pitch`; pose read back with `getviewpos` (engine copies it to the clipboard). |
| Capture | Mod F8 frame capture: scene color (RGBA16F), depth (R32F), projection constants in its log/info file; window screen grab for orientation. Copies go to `void-rs/local/ref/RX001/`. |
| Save protection | `saveGame_enable 0` (the engine skips physical saves and profile writes). Saves, `dishonored2Config.cfg`, and `d2enhanced.ini` backed up with a SHA-256 manifest in `local/backup/` and compared after exit. |
| Stopping condition | After at least 3 matched poses are captured and read back, or on the first unexpected save/config change, crash, or blocked step; then quit with the `quit` command. |

## Execution log (2026-10-16)

1. `+campaign` on the launch command line exits the game during startup (r_reachedFrame50
   left at 0). From inside a loaded save it also crashed. From the main menu it works.
   The main menu has a **Tutorial** entry (answer: tutorial_p is reachable in normal play);
   it was used for the captures.
2. Command channels: the hidden engine console (`Dishonored 2 WinConsole`, edit id 101)
   accepts commands at the main menu; in game, commands were sent through the mod's F2
   console via synthetic input. `noclip` is not a command in this build. `setviewpos`
   sets x, y, and yaw; z lands on the ground and pitch is not applied. Synthetic mouse
   motion does not turn the camera. `getviewpos` writes the view pose to the clipboard
   (2 decimals, 0.1 deg).
3. Exclusive fullscreen defeats GDI capture; the session ran windowed 1920x1080 (changed
   by the user), restored afterwards.
4. Save protection: `+set saveGame_enable 0` did **not** stop writes to `profile.bin` and
   `dishonored2Config.cfg`. Both were restored from backup after the session; every file
   matched the pre-experiment manifest afterwards. Save slots were never modified.

## Captures

Mod F8 writes four consecutive frames (color RGBA16F, depth R32F, motion, output) per press.
Projection in all captures: P0 = 1.0, P5 = 1.77778, reversed-Z infinite, near 0.01.

| Pose | getviewpos | Dumps | Notes |
| --- | --- | --- | --- |
| P1 | (15.95, 12.02, -3.14) yaw 166.0 pitch 0.2 | dump0-3 | spawn, waterfront |
| P2 | (48.30, -39.97, 22.13) yaw 90.0 pitch 7.4 | dump4-7 | on a structure, looking down |
| P3 | (-19.01, 23.25, -4.66) yaw 300.0 pitch 346.0 | dump8-11 | in water in a canal, looking up |

Files: `local/ref/RX001/P{1,2,3}/` (raw dumps, log copy, `color_preview.png`, compare output).

## Results (`void-compare`, stride 4, |rel depth| <= 1% counts as a match)

| Check | P1 | P2 | P3 |
| --- | --- | --- | --- |
| id basis vs pitch-flipped vs mirrored (two-sided) | 76.1 / 44.2 / 3.3 % | 41.1 / 5.2 / 0.4 % | 42.2 / 0.0 / 0.7 % (culled) |
| match with back-face culling | 76.1 % | 62.8 % | 42.2 % |
| match excluding game sky pixels | 89.2 % | 96.4 % | 42.2 % (no sky) |
| median abs rel depth error (two-sided) | 0.18 % | 0.64 % | 1.4 % |
| front faces = (b-a)x(c-a) pointing away | 99.98 % | 99.0 % | 100 % |
| depth-derived vs geometric normal, median | 0.06 deg | 0.13 deg | 0.70 deg |

Conclusions (O = observed on these captures, D = derived):

- **Scale (O):** file units are game world units; depths agree to 0.2-0.6 % median.
- **Orientation/handedness (O):** id convention, Z-up, `forward = (cp cy, cp sy, -sp)`,
  `right = (sy, -cy, 0)`, positive pitch looks down. Mirrored variants fail everywhere.
- **Transforms (O):** instance 3x4 matrices place geometry correctly in all three views.
- **Winding (O):** front faces are clockwise in index order as seen by the camera, i.e. the
  geometric normal of front faces is `(c-a)x(b-a)`. With culling, P2 rises from 41 % to 63 %.
- **Normals (O):** geometric normals agree with depth-derived normals. The stream-1 normal
  hypothesis remains rejected (viewer comparison, E015).
- **Missing surfaces (O/D):** not in the static world, by material/category:
  entity props (barrels, P1, "game closer"), probably a water surface (P1 mud at 6 m).
- **Present but rendered differently (O/D):** `vista/skies/tutorial_sky_01` dome at
  ~5,200 units is drawn at infinite depth (game sky pixels where ours hits: 14.7 % P1,
  34.9 % P2); `dunwall_vista_01` buildings/floors at 330-830 units have farther depth in
  game than geometry (vista depth treatment, unknown).
- **P3 (unresolved):** pose search confirms pitch -14 and zero offset, yet ground/walls are
  closer in ours over 55 % of pixels. The player was in water and may have moved between
  the F8 frame and the getviewpos read; underwater rendering is another candidate. Not used
  for conclusions beyond the pitch-sign check.
