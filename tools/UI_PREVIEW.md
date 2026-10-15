# Working on the actual ImGui UI without the game

The preview compiles **src/ui.cpp and src/counter.cpp unchanged**, with the same fonts, style, ImGui version and precompiled DX11 renderer used by the mod. Only FSR/HDR status and console services are simulated. It is a native Win32/D3D11 window, not an HTML mockup. Hardware rendering falls back to WARP if needed.

## Build and run

From the d2enhanced directory:

```powershell
.\build_ui_preview.bat
.\build\ui_preview\ui_preview.exe
```

The build uses the same Visual Studio Build Tools location as build.bat. Objects and the executable stay under build/ui_preview. It does not deploy the game DLL, attach to the game, or link any game hooks. The preview starts from repeatable defaults; Save settings writes **build/ui_preview/d2enhanced.ini**, never the game's INI. Console commands only echo. Displayed performance/resolution/HDR status is sample data, not a benchmark of this application.

Use the preview controls to change frame-time scenarios, counter mode, HDR status and UI scale. F2 toggles the settings window; F11 starts/stops the benchmark. All normal menu controls remain interactive. Resize the window and check keyboard navigation as well as mouse input. Export preview.png captures the client render next to the executable.

## Agent iteration loop

1. Edit the shared UI or counter source.
2. Rebuild the preview (close a running interactive preview first so its EXE can be replaced).
3. Render PNGs, inspect them with an image viewer/tool, and correct the actual layout.
4. Exercise controls by mouse/keyboard or repeatable CLI clicks.
5. Run build.bat to check integration with the complete mod. Deploy only as part of the intended game-testing workflow.

```powershell
.\build\ui_preview\ui_preview.exe --hidden --capture build/ui_preview/lighting.png --page 1 --scenario 1
.\build\ui_preview\ui_preview.exe --hidden --capture build/ui_preview/small.png --page 4 --width 1280 --height 720 --mode 0
.\build\ui_preview\ui_preview.exe --hidden --capture build/ui_preview/hidpi.png --scale 1.5 --width 1920 --height 1080
.\tools\capture_ui_gallery.ps1 -Build
```

Capture reads back the actual D3D11 backbuffer and writes PNG through WIC, so the window can be hidden or covered. It waits eight rendered frames by default for ImGui sizing/font uploads, then exits. `--frames N` changes this. A nonzero exit code means build/render/capture failure; missing screenshots are not successful validation.

`--page`: 0 Upscaling, 1 Lighting, 2 Display & HDR, 3 Colour, 4 Performance, 5 Console, 6 Log, 7 Diagnostics.

`--mode`: 0 Off, 1 Minimal, 2 Frame time, 3 FSR, 4 Full, 5 Benchmark.

`--scenario`: 0 Steady, 1 Periodic hitches, 2 Slow, 3 FSR failed, 4 FSR starting. `--sdr` switches the simulated HDR display off. Captures seed the real counter statistics with deterministic intervals; interactive mode continues feeding synthetic intervals.

### Repeatable interaction

`--click X,Y` injects a real ImGui mouse press/release at client-render coordinates after layout settles. Repeat the option for a sequence; the capture automatically waits for the sequence to finish. Coordinates depend on size and scale, so inspect the matching baseline screenshot first. This can select lighting subtabs, expand advanced controls, toggle settings, or activate Save. For example, at the default 1600x1000 / 1.25 scale:

```powershell
# Select Reflections within the Lighting page.
.\build\ui_preview\ui_preview.exe --hidden --page 1 --click 425,275 --capture build/ui_preview/reflections.png
# Select Materials & debug.
.\build\ui_preview\ui_preview.exe --hidden --page 1 --click 550,275 --capture build/ui_preview/materials.png
```

The preview uses the real config draft/publication/save code. Its source includes only fixture implementations at the service boundaries, so future UI dependency additions should extend these fixtures, not introduce separate copies of the menu.

## What screenshots do not validate

The preview validates layout, fonts, input and settings wiring. It does not validate game hooks, HDR10 output, actual FSR timings or whether a rendering option looks correct in-game. The background and service data are explicitly simulated. The FPS panel may sit behind the open settings window at small viewport sizes, just as other overlay windows can; the settings header also shows the frame rate.
