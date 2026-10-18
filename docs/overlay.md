# In-game overlay and code patches (void-inject)

Code: `crates/void-inject/src/overlay.rs` (+ `overlay/{paint,input,ui,hud,preview}.rs`),
`patches.rs`, `settings.rs`, `mouse.rs`, `hook.rs`. Offline check:
`cargo run -p void-inject --release --example overlay_preview` (PNGs in `local/overlay/`).
Pass `-- 1280x720` for the smaller-resolution check. The preview exercises navigation,
FSR/HUD/notification toggles, Save settings, scrolling and expanded FSR expert controls,
all four HUD anchors, missing samples and absent FSR timing. It checks D3D11 state
restoration and colour conversion and prints the HUD cost. Images `0`–`8` cover the
original panels/fallback, `9`–`11` the always-on layers, `12`–`15` texture mods and overlay
settings, `16`–`18` HUD corners/missing data, `19`–`21` scrolled FSR controls, `22` the
Lighting panel (ambient occlusion), `23` its scrolled part (sun contact shadows, shadow
map A/B), `24` the Reflections panel, `25`–`27` the Photo panel (photo mode running, after
clicking "Leave photo mode", scrolled to the scene service and probe).
Status: redesigned UI rendered and visually inspected offline at 3440x1440 and
1280x720; not yet run in game (checklist below).

## Use

`overlay = 1` (default) installs it; `overlay_key = F2` (F1-F24, Insert, Home, End, PageUp,
PageDown, Delete, Pause, ScrollLock, Tilde, Apps or a virtual-key number; F2 is unbound in
the user's `dishonored2Config.cfg`, whose 67 `arkBind` lines use F5/F9 only among F-keys)
toggles it while the game window has the focus; `overlay_scale = auto` (window height /
1080) or a factor 0.5..4. Panels: Overview (build, frame time, FPS, 5 s graph of the CPU
frame interval and the engine GPU time `dword_1425AE8E4`, active features), FSR, Lighting
(AO technique HBAO+ / own, quality, half resolution, radius, strength, AO debug view,
hook state and first-call struct fields; sun contact shadows with quality, ray length,
thickness, debug view and pass counters; edge halo: multiplier of the game's DOF-combine
sharpening `dof_sharpen` with the game coefficient (docs/lighting-options.md 5b); shadow-map
A/B: cascade time slicing through the `cvar.r_shadowParallelPartialSliceUpdate` override and
the `shadows_submit_early` patch),
Reflections (screen-space reflections: on/off, quality, half resolution, strength,
roughness, reflectance, thickness, floors only for pixels without material, temporal
accumulation, material source auto/depth, skip water, debug view incl. water mask, material
target, history length, surface class and F0 colour, hook state, frames applied/skipped, water-mask
coverage, temporal restarts, shader-patch counts (excluded classes) and first-frame inputs;
card "Ray-traced reflections": on/off, max roughness, ray budget, debug view, frames
traced/skipped, rays requested/traced/hits, hit-shading scale, GPU ms;
docs/reflections.md), Photo (photo mode on/off and hotkey `photo_key = F4`, enter/leave and
Save buttons; while running with `photo_display = hybrid` one progress bar per term (sun
shadows, ambient occlusion: rays per pixel done / target, rays shown, ms per pass), the
reflection samples per texel, phase, depth-prepass ms and submits; with the path tracer the
samples per pixel, ms per sample, spp/s and M samples/s; pause kind, free camera, image
size, running time and the paused-frame pass rates (frames, FOG, SCREENSHADOWS, SSAO per
second); card "Photo settings": display hybrid/pathtrace/overlay, GPU ms per frame, texture
size, folder; card "Hybrid terms": sun shadows / AO / reflections on/off, sun and AO rays
per pass, reflection rays per frame, target rays per pixel, sun radius, AO radius, AO mode
min/replace, save masks; card "Path tracing": samples, bounces, sky scale, engine units
auto/fixed; the ray-tracing scene service `rt`, build at load/use, TLAS radius, moving
props `rt_dynamic`, characters `rt_characters`, posed per frame `rt_characters_n`, skinning
GPU ms `rt_dynamic_budget_ms`, and its worker/scene/TLAS state with, while dynamic objects
run, rows Dynamic reads / Props / Characters / Palette / Reserve; the read-only probe values
and FOG/composite counters; docs/raytracing.md 11.6, 13 and 14),
Cvars,
Patches, Texture mods, Profiler, Log (last 300 lines), Overlay. "Save settings" writes
`local/inject/void-inject.ini` (previous file kept as `void-inject.ini.bak`) in the
canonical layout with comments.

The slate/teal menu uses grouped sidebar navigation (Workspace, Rendering, Tools),
page headings and section cards. Overview separates headline metrics, frame pacing and
active features. FSR separates reconstruction quality, motion/transparency, collapsible
expert controls and runtime diagnostics. Overlay separates menu/scale, performance HUD
and notifications/benchmark visibility. The window starts around 880x740 points,
clamped to the viewport; page content scrolls independently of navigation and the
persistent Save settings/restart footer. Each page retains its own scroll position.

Always-on layers (drawn with the menu closed too, non-interactive, no input capture):
- Performance HUD: `hud = 0|1` (live, default 0), `hud_key = F7` toggles it (live; F7 is
  unbound in the user's config, see above), `hud_position = top-left | top-right |
  bottom-left | bottom-right`. Shows FPS, mean frame time and 1% low over 1 s (frame
  interval at the engine present), engine CPU frame (`qword_1425AE8E8` minus the swap
  phase `dword_1425AE8E0`, as the benchmark's `eng_cpu_ms`), engine GPU frame
  (`dword_1425AE8E4`), FSR GPU ms while FSR runs (rolling), and a 3 s frame-time graph
  (fixed 0..25 ms, labeled lines at 8.3 / 16.7 ms). A rounded instrumentation card gives
  FPS visual priority, with aligned secondary metrics and a teal graph; missing samples
  show dashes and a waiting message rather than fabricated readings.
- Notification toasts: `notifications = 1` (live). Producers call `notify::info/warn/error`
  from any thread; the overlay drains the queue once per frame and stacks the toasts in
  the top-right corner (slate cards with a severity marker/heading, fade-out at the end
  of their lifetime, at most 6). A toast's lifetime starts when the overlay first shows it, so startup notes are not
  lost. Producers here: patches (applied / mismatch / failure at startup, failed live
  toggle), texture mods (served / rejected summary after each loading burst, 3 s quiet),
  overlay (settings saved / not saved, restart-only change pending).
- Benchmark: while a segment records (`profiler::is_recording()`), HUD and toasts are not
  drawn (toasts keep their timers) unless `bench_show_overlay = 1` (live, default 0); an
  explicitly opened menu is still drawn.
- AO / contact-shadow debug views: `ao_debug = 1` / `contact_debug = 1` (live) draw our last
  AO result / contact visibility full-screen before the present (under menu/HUD/toasts),
  through a UNORM view of the back buffer and under the engine's immediate-context lock;
  need `overlay = 1` (they run in the overlay's present path) and the effect enabled.
- FSR debug view (Shift+F6 cycles, or the combo in FSR > Expert controls; not saved):
  `off`, `FFX` (FFX's own view, drawn by FSR into its output) and `motion vectors` (the
  vectors FSR received, drawn in the same present path as the AO view: log brightness over
  0.02-32 render px per frame, hue = direction, legend bar with ticks at 0.1 / 1 / 10 px;
  the legend text is shown under the combo). docs/fsr4-integration.md "Motion-vector
  coverage and debug views".
- Photo mode (`photo = 1`, docs/raytracing.md 11.6 and 13): first in the same present path,
  under the same lock, before the debug views and our UI: a requested back-buffer copy for a
  save (staging texture, mapped without waiting a few frames later, PNG written on a
  thread; with `photo_display = hybrid` this PNG is the photo: the game's frame with the
  ray-traced terms and the engine's post) and, with `photo_display = overlay`, our
  tone-mapped photo letterboxed over the frame. Toasts: photo mode entered/left/ended with
  the reason, photo and game frame saved or not, "reflections need FSR on" (reflection term
  skipped), the game stopped rendering while paused.

Cost: with menu, HUD and toasts all hidden the present detour does no egui work (key
polls, frame-interval ring, `notify::take` on an empty queue, two atomic loads for the
texture-mod summary). Redesigned HUD measured offline (`overlay_preview`, 3440x1440,
native D3D11, 300 frames; CPU time of egui build + tessellation + D3D11 calls):

| Layers | Mean | p50 | p99 |
| --- | ---: | ---: | ---: |
| HUD | 108.9 us | 69.0 us | 541.8 us |
| HUD + 3 toasts | 157.1 us | 92.0 us | 503.5 us |

These are one offline run, not in-game GPU/frame-time measurements. The graph uses the
HUD's existing painter clip to avoid extra per-graph meshes; `egui-directx11` creates
vertex/index buffers per mesh each frame (no renderer fork).

Every setting carries a `live` or `restart` marker (reason on hover, `settings::KEYS`).
Restart-only edits are saved but not applied; the footer lists them. A ↺ icon next to a
setting appears while it differs from its default; clicking it restores the default (hover
shows the value). Covered: every checkbox and slider plus FSR mode/jitter phases/mip bias,
GPU timing mode, overlay scale, HUD position, AO technique and quality (not the key-name
text fields).

| live | restart |
| --- | --- |
| `profile_gpu`, `profile_gpu_mode`, `profile_labels`, `bench_*` (incl. `bench_show_overlay`), `cvar.*`, `notifications`, `hud`, `hud_key`, `hud_position`, every `fsr*` key (`fsr` on installs FSR at run time if it was off at startup, off deactivates it with the hooks left installed and inert; `fsr_mode` / `fsr_scale` / `fsr_masks` per frame; `fsr_mip_bias` re-creates the engine's samplers; `fsr_auto_exposure` re-creates the FFX context; see docs/fsr4-integration.md "Implementation status"), every `ao_*` key (`ao_mode = void` installs the HBAO+ hook at run time; docs/lighting-options.md B1), every `contact_*` key (`contact_shadows = 1` installs the SCREENSHADOWS hook at run time; docs/lighting-options.md B2), every `rt*` and `photo*` key (`photo = 1` installs the FOG hook at run time, `photo = 0` leaves photo mode; docs/raytracing.md 11.6), `overlay_key`, `overlay_scale`, FSR on/off and debug view (F6) | `profile`, `texture_mods`, `mods_dir`, `overlay`, `patch.*` of non-live patches |

## Hooks (analysed build, image base 0x140000000)

| Hook | Address | Check |
| --- | --- | --- |
| Engine present `sub_14036B3A0(rs, a2, a3)` (only caller `sub_1403591B0`; also the profiler's `present` phase, moved out of the profiler's own table) | 0x14036B3A0 | 16 prologue bytes `48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57` |
| `RS_LockImmediate` / `RS_UnlockImmediate` (called, not hooked) | 0x14036A6F0 / 0x14036B4E0 | 16 bytes each (`48 83 EC 28 80 B9 4A AC ...` / `80 B9 4A AC 63 00 00 74 ...`) |
| idMouseWin32 poll `sub_1400E2930` (buffer flush while open, as d2enhanced) | 0x1400E2930 | `48 89 4C 24 08 55 41 56 48 8D AC 24 78 E8 FF FF` |
| DirectInput `GetDeviceData` / `GetDeviceState` / `Acquire` (system dinput8.dll, found through temporary mouse + keyboard devices) | runtime | - |
| user32 `SetCursorPos` / `GetCursorPos` / `ClipCursor` | runtime | - |
| Game window procedure (subclassed on first opening; window = swap chain `OutputWindow`) | runtime | - |
| Photo mode (`photo = 1` only): FOG pass callback `sub_140309430` (composite, hybrid depth capture), `updateCamera` IAT slot (swapped while photo mode runs), input capture in the DirectInput hooks above (installed by photo mode if `profile`/`overlay` did not); on entering photo mode also the SCREENSHADOWS (`contact.rs`) and HBAO+ `RenderAO` (`ao.rs`) detours if not installed yet (they stay installed and only forward the call while nothing is registered and their own setting is off) | 0x140309430 / 0x141A7E058 | 16 prologue bytes `40 53 55 56 57 41 54 41 55 41 56 41 57 48 81 EC` / slot must hold `AnselSDK64.dll!updateCamera` |

Present path (IDB): `obj = RS_CurrentFrameData(rs)` (vtable +32, `0x14035EC00`, a pure getter:
`rs+54728 + 976 * rs+54744` when the index is in range) and, when `obj != 0 && a3 == 0`,
`IDXGISwapChain::Present` on `*(obj+56)`; otherwise only `Flush`. The detour replicates the
getter with checked reads and draws only for frames that present. The lock is a spin lock
(byte rs+6532170 enables it); its callers are `RenderPass_Submit`, `sub_1402BD5A0`,
`sub_1402D37C0` (binding/submission, reached from `Ctx_BindProgram` and
`JobSys_SubmitReady`) and `sub_14035EA10` (no direct callers), none on the present path, so taking it in the detour cannot re-enter.

Drawing: `egui-directx11` under `SwapDeviceContextState` with our own state object
(cleared before swapping back, so no back-buffer reference survives); per frame
`GetBuffer(0)` and a UNORM view (R8G8B8A8_UNORM_SRGB/TYPELESS -> R8G8B8A8_UNORM, B8G8R8A8
likewise; fallback to the default view, logged once), both released in the same frame.
Hidden: one key poll and the frame-interval ring per frame, no egui work.

Input while open: window mouse/keyboard messages go to egui only; the game's DirectInput
mouse/keyboard devices are unacquired, `Acquire` returns `DIERR_OTHERAPPHASPRIO`,
`GetDeviceData` returns no records, `GetDeviceState` zeros; the mouse poll flushes its
buffer; the game's own `SetCursorPos` is skipped and its `GetCursorPos` sees the last
position it set (callers told apart by return address in the executable); `ClipCursor` is
lifted. Closing restores the cursor to the game's last position and flushes the mouse
buffers; the game re-acquires by itself.

## Code patches (`patches.rs`)

One registry entry per patch: `id` (settings key `patch.<id> = 0|1`), name, description,
evidence, apply time (`DllMain` before the game's init, `Init` after the executable check,
`Live` = toggled at run time), default, and byte edits `{ rva, original, patched }`
(`hex-literal`). A patch is written only if all original bytes match (all or nothing);
live patches revert only if all patched bytes match. States (applied / off / original bytes
mismatch / failed) are logged at initialisation and shown in the Patches panel. The new
settings file lists every registered patch with its description. Adding a patch = adding
one `Patch` entry (unit tests check unique ids, equal lengths, non-empty, no overlaps).

Registered: `skip_intro` (boot videos, 0x140064F08 `0F 85 BC 00 00 00` -> `E9 BD 00 00 00
90`, E046), which replaces the former `skip_intro` key.

## In-game checklist

1. Log: `engine present hooked`, `mouse: ... hooked`, `mouse: engine mouse poll hooked`,
   `overlay installed`, `patch skip_intro: applied`; boot videos skipped.
2. F2 in game and main menu: window appears with the arrow cursor; F2 again closes it.
3. While open: mouse look, clicks and keys do not reach the game; the cursor is free; text
   fields accept typing (depends on the game's message loop translating WM_CHAR).
4. After closing: no view jump, mouse look and keys work at once.
5. Alt+Tab / resolution change in the menu with the overlay open and closed (ResizeBuffers).
6. FSR panel: F6 checkbox toggles like F6; stats fill in; sharpness changes visibly.
7. Cvars: add/edit/remove an override, lookup of `image_anisotropy`, candidate buttons.
8. Profiler (`profile = 1`): start a benchmark from the panel (overlay closes), log shows
   the run; arm a trace; `present` column present in the run's frames file.
9. Save settings: `.ini` rewritten in canonical layout, `.ini.bak` holds the previous one.
10. Frame time with the overlay hidden vs before the change (profiler A/B).
11. F7 shows/hides the HUD with the menu closed; numbers plausible against `com_showFPS`;
    position setting moves it; the HUD state is saved with Save settings.
12. Toasts at startup (patches applied), after a map load with texture mods, on Save
    settings and on a restart-only edit; none while a benchmark segment records.
13. Photo panel: the in-game steps of docs/raytracing.md 13.10 (hybrid display, default) and
    11.7 (probe values first; path tracer with `photo_display = pathtrace`).
