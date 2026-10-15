# Dishonored 2 Enhanced

A rendering and quality-of-life mod for the PC version of Dishonored 2. It loads as a `d3d11.dll` proxy and adds
modern upscaling, HDR output, screen-space global illumination and reflections, colour grading and an in-game
settings overlay. Nothing in the game's files is modified on disk.

> **Work in progress.** A personal project, developed and tested on one machine (AMD RX 7900 XTX, 3440x1440, DXVK) with
> the **Steam version** of the game. Features change, some are experimental, and there will be bugs: read the log
> (`d2enhanced\logs\d2enhanced.log`) when something misbehaves, and attach it to issues.

## Features

**Upscaling**
- AMD FSR (4.1 when the driver offers it, otherwise FSR 3.1 / 2.3) replaces the game's TAA. FSR runs on a D3D12
  device next to the game's D3D11 device and is spliced into the engine's command stream exactly where TAA ran.
- Upscale mode: the game renders at a reduced resolution and FSR upscales *before* post-processing, which then runs at
  display resolution (motion blur, depth of field, bloom and UI stay sharp). Native mode: FSR anti-aliasing only.
- 32-phase Halton jitter instead of the engine's 8 phases, texture mip bias while upscaling, optional sharpening.

**Lighting**
- Screen-space global illumination (visibility-bitmask GI) with temporal accumulation and denoising.
- Screen-space reflections replacing the game's cubemap reflections where a ray hits, including wet decals.
- Real material data: the game's forward-lighting pixel shaders are patched at load time to also output base colour,
  roughness, metallic and the cubemap reflection they computed.

**Display and colour**
- HDR10 (PQ / Rec. 2020) output when Windows HDR is already on for the display: paper white, peak brightness and
  highlight boost settings, with the UI composited at paper white.
- Colour grading in the game's tonemap pass: exposure, contrast, saturation, vibrance, white balance and the strength
  of the game's own grading LUT. Works in SDR and HDR.
- Optional "Ultra+" engine settings beyond the menu's Ultra (shadow resolution and distance, filtering, skin shading).

**Quality of life**
- In-game overlay (F2) for every setting, with live changes; a console for engine cvars and commands (the retail
  build's cvar lock can be lifted); a log viewer.
- Frame limiter or unlocked frame rate. Above 120 fps the game's fixed 120 Hz simulation tick is raised to match the
  cap (experimental), which removes the stutter the game otherwise shows above 120 fps.
- Mouse input fixes (no dropped movement on frame hitches), intro video skip.
- Frame time counter with several layouts and a benchmark recorder (F11).
- HUD at the screen edges on ultrawide displays (the game keeps it in a centred 16:9 area).
- DXVK backend (a patched DXVK 3.1.1 that can share textures with D3D12), see below.

### DXVK or native D3D11

The mod runs on the game's own D3D11 and on DXVK (D3D11 on Vulkan). **DXVK is recommended**: the game is CPU-bound in
the D3D11 driver, and on the development machine DXVK runs it at about 170 fps against 90 natively. The mod is developed
on DXVK; native D3D11 has been tested with the same features except HDR output, which has only been tested with DXVK
(on native D3D11 the mod falls back to SDR if Windows refuses the HDR swap chain).

DXVK has to be the patched build from this repository (`patches\dxvk-3.1.1-d3d12-interop.patch`): stock DXVK can't
share textures with the D3D12 device FSR runs on.

## Requirements

- **Dishonored 2, Steam version**, current build (the executable from February 2025). The mod was only developed and
  tested against this executable. Other versions (GOG, Microsoft Store / Game Pass, Epic, older Steam builds) have
  different executables: the mod detects this, disables its engine patches and says so in the log, so upscaling and
  most features won't work there.
- Windows 10 or 11 and a GPU with D3D12 support. FSR 4 needs a GPU and driver that provide it; the FSR 3.1 fallback
  runs on any D3D12 GPU.
- HDR output: an HDR display with "Use HDR" enabled in Windows.

## Installation

**Release build:** the [latest release](https://github.com/LeagueRaINi/dishonored-2-enhanced/releases/latest) has two
zips. Extract one into the game folder (`...\steamapps\common\Dishonored2`, next to `Dishonored2.exe`), start the game
and press **F2** for the settings overlay:
- `...-dxvk.zip` (recommended): the mod, AMD's signed FidelityFX DLLs and the patched DXVK (`dxgi.dll` and
  `dxvk.conf` next to the exe, its `d3d11.dll` in `d2enhanced\dxvk\`).
- `...-native.zip`: the mod and the FidelityFX DLLs, on the game's own D3D11.

The mod creates `d2enhanced\logs\` there for its log. Switching later: delete `dxgi.dll` and `dxvk.conf` for native,
or extract the DXVK zip over the native install.

**Own build:** build the mod (below), then copy into the game folder:
- `d3d11.dll` (from `build\`) and `d2enhanced.ini`
- `amd_fidelityfx_loader_dx12.dll` and `amd_fidelityfx_upscaler_dx12.dll` from
  `third_party\FidelityFX-SDK\Kits\FidelityFX\signedbin\`

Keep the repository itself as `d2enhanced\` inside the game folder if you want the DXVK backend; logs and captures
go there too.

In the game's graphics menu, anti-aliasing must stay on TAA (the mod enforces this while FSR is on).

To uninstall, delete `d3d11.dll`, `d2enhanced.ini`, the two `amd_fidelityfx_*.dll` and the `d2enhanced` folder from
the game folder, plus `dxgi.dll` and `dxvk.conf` with DXVK.

### Hotkeys

| Key | Action |
| --- | --- |
| F2  | Settings overlay |
| F6  | FSR on / off (game TAA) |
| F7  | Reduced render resolution on / off |
| F8  | Capture one frame (pass log and upscaler inputs, for debugging) |
| F10 | FSR debug view |
| F11 | Benchmark recording start / stop |

Keys can be changed in `d2enhanced.ini` (`[Keys]`, virtual key codes).

## Building

Requirements: Visual Studio 2026 Build Tools (MSVC with the latest C++ standard), the Windows SDK, Python 3 (shader
compilation through `d3dcompiler_47.dll`) and git.

```bat
git clone --recursive https://github.com/LeagueRaINi/dishonored-2-enhanced.git d2enhanced
cd d2enhanced
build.bat
```

`build.bat` compiles the helper shaders (`tools\compile_shaders.py` -> `src\shaders_generated.h`), MinHook, Dear ImGui
and the mod into `build\d3d11.dll`. The Visual Studio path is set at the top of the script.

Optional:
- `build_dxvk.bat` builds DXVK 3.1.1 with `patches\dxvk-3.1.1-d3d12-interop.patch` into `dxvk\` (needs meson and
  ninja in `.venv\`, and glslang in `third_party\glslang_dl\x\bin\`). `use_dxvk.bat` / `use_native.bat` switch the
  game between DXVK and native D3D11.
- `build_ui_preview.bat` builds a standalone window with the real overlay UI and simulated data, for working on the
  UI without the game (`tools\UI_PREVIEW.md`).

## How it works

The engine addresses and structure layouts in the sources come from reverse engineering the current Steam build.
In short:

- `src\proxy.cpp` forwards device creation to the real `d3d11.dll` (system or DXVK) and installs the hooks.
- `src\hooks.cpp` hooks device and context methods; the engine's TAA dispatch is replaced at record time.
- `src\fsr.cpp` splits the command list at the TAA dispatch and runs FSR on D3D12 between the two halves, synchronised
  with shared fences.
- `src\game.cpp` holds the engine patches (render scale, jitter, frame limiter, game tick, menu locks, mouse).
- `src\ssgi.cpp`, `src\material.cpp` and `src\shader_patch.cpp` implement the screen-space lighting and the material
  capture; `src\hdr.cpp` the HDR output and grading; `src\ui.cpp` and `src\overlay.cpp` the overlay.

## License

The mod's code is MIT licensed (`LICENSE`). Third-party components keep their own licenses (FidelityFX SDK and Dear
ImGui: MIT, MinHook: BSD-2, DXVK and the DXVK patch: zlib).

Dishonored 2 is a trademark of ZeniMax Media Inc. This project is not affiliated with or endorsed by Arkane Studios,
Bethesda Softworks or ZeniMax, and contains no game files.
