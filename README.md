# Dishonored 2 Enhanced

Rendering improvements for the PC version of Dishonored 2, written in Rust. The mod is a
`dinput8.dll` proxy. It hooks the Void engine at the call sites of individual render passes
and replaces or extends those passes inside the game's native D3D11 renderer. It does not
replace the whole renderer and does not need DXVK.

Not affiliated with or endorsed by Arkane Studios, Bethesda or AMD. This repository has no
game assets, extracted shaders or executables.

> **Status: experimental.** Every feature has an off switch. Most are validated with
> offline harnesses (`examples/`), and only some have been checked in the game. The
> per-feature state is tracked in [docs/STATUS.md](docs/STATUS.md) and [PLAN.md](PLAN.md).
> The only GPU tested so far is an AMD RX 7900 XTX.

## Features

| Feature | Setting (default) | Notes |
| --- | --- | --- |
| FSR 4 in place of the engine TAA (native AA or upscaling) | `fsr` (0), `fsr_mode`, `fsr_scale` | Needs the FidelityFX SDK 2.3 DLLs in the game folder (below). F6 switches between FSR and the engine TAA. |
| Own ambient occlusion in place of NVIDIA HBAO+ | `ao_mode` (`void`) | `hbao` = the game's original AO. |
| Sun contact shadows in the screen-shadow pass | `contact_shadows` (1) | |
| Screen-space reflections with per-pixel material data from patched lit shaders | `ssr` (0), `ssr_*` | Temporal accumulation and firefly suppression. |
| Hybrid ray-traced reflections | `rt` + `rt_reflections` (0) | DXR 1.1 inline ray tracing on a D3D12 device. Needs `ssr` and FSR. |
| Photo mode: free camera with ray-traced shadows, AO and reflections | `photo` (0), key F4 | Uses the engine's own pause/photo session. |
| Texture and resource replacement | `texture_mods` (0), `mods_dir` | DDS textures and overrides served through the engine's file system. [docs/texture-replacement.md](docs/texture-replacement.md) |
| Engine cvar overrides | `cvar.<name> = <value>` | Default: `image_anisotropy 16` (Ultra leaves it at 1). |
| Game patches | `patch.<id>` | Skip the boot videos; submit shadow command lists earlier (CPU-bound frames −0.5 ms). |
| Overlay, performance HUD, profiler and benchmark | `overlay` (1, F2), `hud` (F7), `profile` (Ctrl+F3) | [docs/overlay.md](docs/overlay.md), [docs/performance-baseline.md](docs/performance-baseline.md) |
| Diagnostics for bug reports | `diagnostics` (0) | FSR value check and motion-vector statistics (GPU passes every frame), the FSR timing report and a per-second frame summary in the log. Leave off for play. |

## Requirements

- Windows 10/11 x64 and the Steam build of Dishonored 2 whose `Dishonored2.exe` has SHA-256
  `b671f8249422862bf557d00aab2f4bb671ed4515cc10ad8a8a9decad84516180`. All engine
  addresses are specific to this build. With any other executable the runtime hooks stay
  off. Patches applied at DLL load and every other code patch first verify the exact
  original bytes, so they leave a different build untouched.
- FSR: `amd_fidelityfx_loader_dx12.dll` and `amd_fidelityfx_upscaler_dx12.dll` from the AMD
  FidelityFX SDK 2.3, placed next to `Dishonored2.exe`. The FSR 4 provider itself ships with
  the AMD driver.
- Ray-traced features: a GPU with DXR 1.1 (inline ray tracing, Shader Model 6.5).

## Install a release

1. If the older C++ Dishonored 2 Enhanced (v0.1.0) is installed, remove it first. Delete
   `d3d11.dll`, `d2enhanced.ini` and the `d2enhanced` folder, plus `dxgi.dll` and
   `dxvk.conf` if you used its DXVK zip. Both mods replace the TAA pass, so only one can
   be installed at a time.
2. Extract the release zip into the game folder (`...\steamapps\common\Dishonored2`, next
   to `Dishonored2.exe`). It contains `dinput8.dll`, AMD's signed FidelityFX DLLs and the
   default settings file `void-rs\local\inject\void-inject.ini`.
3. Start the game and press **F2** for the overlay. Keep the in-game anti-aliasing setting
   on TAA when you use FSR.

The log is `void-rs\local\inject\void-inject.log`; attach it to bug reports, ideally after
a run with `diagnostics = 1`. To uninstall, delete `dinput8.dll`, the two
`amd_fidelityfx_*_dx12.dll` files and the `void-rs` folder.

## Build from source

```sh
cargo build --release -p void-inject
```

Building needs a recent Rust toolchain (edition 2024; development uses nightly MSVC). It
also needs `dxc.exe` + `dxil.dll` from the Windows SDK (or set `VOID_DXC_DIR`) to precompile
the ray-tracing shaders.

Copy `target/release/dinput8.dll` next to `Dishonored2.exe`. If the settings file is
missing, the DLL creates `void-rs/local/inject/void-inject.ini` under the game folder on
the first start, with every key commented. Most settings apply live from the overlay (F2),
which can also save them.

## Repository layout

| Path | Contents |
| --- | --- |
| `crates/void-inject` | The injected DLL (`dinput8.dll`): hooks, passes, overlay, profiler, offline harnesses in `examples/` |
| `crates/void-rt` | D3D12 DXR scene building and ray-tracing kernels (live reflections, photo mode) |
| `crates/void-res` | Read-only access to the game's `.index`/`.resources` archives |
| `crates/void-formats` | Decoders for engine formats (world `.bwm`, models, `.bimage`, materials, entities, probes, Havok, `.md6`) |
| `crates/void-assets` | Material and texture preparation on top of the decoders |
| `crates/void-tool` | Inspection CLI (`void-tool info/list/extract/verify/texmod/...`) |
| `crates/void-reflect` | Offline extraction of the engine's reflection tables from the executable |
| `docs/` | Research and design notes, format documentation, the evidence ledger |
| `tools/` | Python helpers for captures and image comparisons |

The engine is Arkane's Void engine, so the crates use the `void-` prefix.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Checks:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --workspace --release
cargo run --release -p void-inject --example ssr_offline   # and the other offline harnesses
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
