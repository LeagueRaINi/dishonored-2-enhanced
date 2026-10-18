//! Offline check of the overlay (no game): a hidden window with a D3D11 device and an
//! `R8G8B8A8_UNORM_SRGB` swap chain like the game's, the real panels with stand-in data,
//! synthetic pointer input (tab clicks, hovering and clicking a checkbox), drawing through
//! the same painter (state swap, UNORM view of the sRGB back buffer), and read-back.
//! Checks: the game-side pipeline state set before each draw is unchanged after it, the
//! clicks reached the UI, and the UNORM view stores gamma values unconverted (compared
//! with the default sRGB view of a typed texture, the painter's fallback path).

use std::sync::Arc;

use windows::Win32::Foundation::{HMODULE, HWND, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
    D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,
};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    DXGI_SWAP_CHAIN_DESC, DXGI_SWAP_EFFECT_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGISwapChain,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, WINDOW_EX_STYLE, WNDCLASSW,
    WS_OVERLAPPEDWINDOW,
};
use windows::core::{Interface, w};

use super::{Core, Layers, Painter, Preview, PreviewImage, hud, input, ui};
use crate::fsr;
use crate::patches::{PATCHES, State as PatchState};
use crate::settings::Corner;
use crate::settings::Settings;
use crate::{cvars, notify, texmods};

struct Backend {
    file: Settings,
    fsr_active: bool,
    debug: u8,
    patches: Vec<bool>,
    saves: usize,
    /// Photo actions clicked (the stand-in photo mode toggles with each `Toggle`).
    photo_actions: usize,
    photo_active: bool,
}

impl ui::Backend for Backend {
    fn cvar(&mut self, name: &str) -> Option<cvars::Info> {
        let (v, d) = match name {
            "image_anisotropy" => (16.0, "1"),
            "r_lodZoomMultiplicator" => (0.6, "1.0"),
            "r_shadowCullDistance" => (75.0, "25"),
            "r_maxShadowMapResolution" => (1024.0, "1024"),
            _ => return None,
        };
        Some(cvars::Info {
            name: name.to_owned(),
            value: v.to_string(),
            int: v as i32,
            float: v,
            default: d.to_owned(),
            flags: 0x2,
            description: "stand-in value (offline preview)".to_owned(),
        })
    }

    fn set_settings(&mut self, s: Settings) {
        self.file = s;
    }

    fn set_patch(&mut self, id: &str, on: bool) -> Result<(), String> {
        let i = PATCHES
            .iter()
            .position(|p| p.id == id)
            .ok_or("no such patch")?;
        self.patches[i] = on;
        self.file.patches.insert(id.to_owned(), on);
        Ok(())
    }

    fn fsr_toggle(&mut self) {
        self.fsr_active = !self.fsr_active;
    }

    fn fsr_debug_set(&mut self, view: u8) {
        self.debug = view;
    }

    fn bench_toggle(&mut self) -> Result<(), String> {
        Err("offline preview: no game".into())
    }

    fn trace_arm(&mut self) -> bool {
        false
    }

    fn save(&mut self) -> Result<String, String> {
        self.saves += 1;
        Ok("(offline preview: not written)".into())
    }

    fn photo_action(&mut self, a: crate::photo::Action) {
        self.photo_actions += 1;
        if a == crate::photo::Action::Toggle {
            self.photo_active = !self.photo_active;
        }
    }
}

impl Backend {
    fn snapshot(&self, frame: u32) -> ui::Snapshot {
        let mut active = self.file.clone();
        active.fsr = true;
        active.profile = true;
        active.photo = true;
        // Stand-in frame times: 120 fps with a little noise and one hitch.
        let frames = (0..600)
            .map(|i| {
                let t = -5.0 + f64::from(i) / 120.0;
                let n = ((i * 37 + frame as i32 * 11) % 17) as f32 * 0.05;
                let cpu = if i == 400 { 21.0 } else { 8.2 + n };
                (t, cpu, Some(5.8 + n * 0.5))
            })
            .collect();
        ui::Snapshot {
            exe: "analysed build (b671f8249422862b...) [offline preview]".into(),
            frames,
            fsr: fsr::Status {
                installed: true,
                active: self.fsr_active,
                debug_view: self.debug,
                failed: false,
                runtime: Some(fsr::RuntimeStatus {
                    provider: "4.1.1".into(),
                    display: [3440, 1440],
                    render: [3440, 1440],
                    frames: 12_345,
                    resets: 3,
                    cuts: 2,
                }),
                gpu_ms: Some(1.74),
                sky: "mul(M, x), - jitterRel".into(),
                mip_bias: if self.file.fsr_upscale {
                    (-1.58, 37, 112)
                } else {
                    (0.0, 0, 112)
                },
            },
            ao: crate::ao::Status {
                hooked: true,
                waiting: false,
                calls: 12_345,
                ours: 12_000,
                fallbacks: 0,
                info: Some(
                    "first RenderAO call: mask 3; depth type 0 layout 1 ... [offline preview]"
                        .into(),
                ),
                error: None,
            },
            contact: crate::contact::Status {
                hooked: true,
                calls: 9_000,
                ours: 8_990,
                skipped: 10,
                failures: 0,
                info: Some("first SCREENSHADOWS pass: ... [offline preview]".into()),
                error: None,
            },
            ssr: crate::ssr::Status {
                hooked: true,
                calls: 9_000,
                applied: 8_990,
                engine: 4_000,
                submit: 4_990,
                not_ready: 10,
                failures: 0,
                water_fallback: 0,
                water_coverage: Some(0.18),
                history_resets: 3,
                rt: crate::ssr::rt_reflect::Status {
                    frames: 4_200,
                    no_scene: 12,
                    busy: 0,
                    failures: 0,
                    gpu_ms: Some(0.94),
                    last: crate::ssr::rt_reflect::Stats {
                        requested: 61_000,
                        traced: 61_000,
                        hits: 38_500,
                        screen_hits: 21_300,
                        misses: 22_500,
                        scale: 3.41,
                        tile_scale: [2.9, 4.1],
                        calib_samples: 1_840,
                        denoised: true,
                    },
                    error: None,
                    blocker: None,
                },
                material: crate::ssr::material::Status {
                    patching: true,
                    patched: 2_943,
                    excluded: 344,
                    unpatched: 94,
                    reasons: vec![("shading normal (reflect/lerp) not found".into(), 94)],
                    create_fallback: 0,
                    bound: 120_000,
                    hooked: true,
                },
                info: Some("first TAA site: ... [offline preview]".into()),
                error: None,
            },
            photo: self.photo_status(),
            texmods: texmods::Status {
                installed: true,
                dir: Some("void-rs/local/mods-test".into()),
                files: 2,
                served: vec!["models/characters/corvo/textures/sword_albedo".into()],
                rejected: vec![("textures/bad".into(), "size mismatch".into())],
            },
            overrides: self
                .file
                .cvars
                .iter()
                .map(|(n, v)| (n.clone(), *v, 1, n == "r_nope"))
                .collect(),
            patches: PATCHES
                .iter()
                .zip(&self.patches)
                .map(|(p, &on)| ui::PatchRow {
                    id: p.id,
                    name: p.name,
                    description: p.description,
                    source: p.source,
                    when: p.when,
                    on,
                    state: if on {
                        PatchState::Applied.text()
                    } else {
                        PatchState::Off.text()
                    },
                })
                .collect(),
            bench_running: false,
            bench_last: Some("void-rs/local/profile/run_1760000000_scene0_native".into()),
            log: (0..40)
                .map(|i| format!("1760000000.{i:03} offline preview log line {i}"))
                .collect(),
            pending: if self.file.texture_mods {
                vec!["texture_mods".into()]
            } else {
                Vec::new()
            },
            active: Arc::new(active),
            file: self.file.clone(),
        }
    }

    /// Stand-in photo mode: accumulating at 3440x1440 while active.
    fn photo_status(&self) -> crate::photo::Status {
        use crate::photo::probe::{MAP_RUNNING, STATE_IN_GAME};
        let probe = crate::photo::Probe {
            base: true,
            session_flag: Some(u8::from(self.photo_active)),
            game: 0x1_7A2B_3C40,
            current_game: 0x1_7A2B_3C40,
            manager_ok: true,
            top: STATE_IN_GAME,
            depth: 1,
            map_state: Some(MAP_RUNNING),
            camera: Some([-33.65, 181.69, 10.46, 0.02, -0.11, 0.84, 0.53, 58.7]),
        };
        crate::photo::Status {
            enabled: true,
            active: self.photo_active,
            pause: if self.photo_active {
                "Ansel session".into()
            } else {
                String::new()
            },
            free_camera: self.photo_active,
            phase: "accumulating".into(),
            spp_done: 96,
            spp_target: 256,
            ms_per_spp: 35.2,
            size: [3440, 1440],
            seconds: 41.0,
            probe: Some(probe),
            composite: crate::photo::composite::Status {
                hooked: true,
                fog_frames: 5_120,
                composited: 1_480,
                failures: 0,
                error: None,
            },
            scene: crate::rt::scene::Status {
                worker: true,
                phase: "ready".into(),
                built_map: "maps/campaign/edge_of_the_world/eotw_p_highchaos".into(),
                load_s: 2.3,
                upload_s: 4.1,
                instances: 99_329,
                spike: false,
                tlas_instances: 0,
                tlas_rebuilds: 0,
                tlas_ms: 0.0,
                error: None,
                dynamic: Some(crate::rt::scene::DynamicStatus {
                    track: crate::rt::track::TrackStatus {
                        slots: 9_370,
                        flagged: 41,
                        static_mapped: 5_538,
                        props: 3,
                        skinned: 129,
                        wanted: 9,
                        palettes: 6,
                        events: 47,
                        bytes: 92_000,
                        reads: 58,
                        us: 182.0,
                        skip_update: false,
                        first_person: 3,
                        moved_per_s: 4,
                        spawned_per_s: 0,
                        groups_near: 5,
                    },
                    worker: crate::rt::dynamic::DynStatus {
                        stats: void_rt::ingame::DynamicStats {
                            moved_static: 2,
                            props: 3,
                            prop_meshes: 2,
                            characters: 9,
                            posed: 9,
                            triangles_used: 312_000,
                            triangles_reserved: 1 << 20,
                            textures_used: 41,
                            bytes: 96 << 20,
                            ..Default::default()
                        },
                        last_update: void_rt::ingame::UpdateStats {
                            skinned: 6,
                            gpu_ms: 0.41,
                            ..Default::default()
                        },
                        props_decoded: 2,
                        characters_decoded: 5,
                        mode_skinning: 9,
                        characters_traced: 9,
                        surfaces_traced: 61,
                        surfaces_total: 61,
                        tlas_refits: 1_204,
                        tlas_refit_ms: 0.6,
                        ..Default::default()
                    },
                    events_dropped: 0,
                }),
            },
            hybrid: self.photo_active.then(|| {
                let term = |rays, ms| void_rt::hybrid::TermProgress {
                    enabled: true,
                    rays,
                    target: 256,
                    ms_per_batch: ms,
                    rays_per_batch: 4,
                };
                crate::photo::hybrid::Status {
                    phase: "accumulating".into(),
                    size: [3440, 1440],
                    sun: term(96, 13.7),
                    ao: term(92, 14.2),
                    shown: [88, 84],
                    prep_ms: 13.2,
                    submits: (61, 11.4),
                }
            }),
            reflections: self.photo_active.then_some(48),
            last_error: None,
            last_save: Some("Photo saved: eotw_p_highchaos_1760000000_hybrid (sun 256 / AO 256 rays, reflections 128 samples), game frame follows".into()),
            rates: Some(crate::photo::PassRates {
                ticks: 60.0,
                fog: 60.0,
                shadows: 60.0,
                ssao: 60.0,
                fog_hooked: true,
                shadows_hooked: true,
                ssao_hooked: true,
            }),
        }
    }
}

/// Game-side pipeline state set before each draw and compared afterwards.
struct Probe {
    rtv: ID3D11RenderTargetView,
    srv: ID3D11ShaderResourceView,
    sampler: ID3D11SamplerState,
    blend: ID3D11BlendState,
    raster: ID3D11RasterizerState,
    depth: ID3D11DepthStencilState,
    cb: ID3D11Buffer,
    viewport: D3D11_VIEWPORT,
    scissor: RECT,
}

impl Probe {
    fn new(dev: &ID3D11Device) -> Result<Self, String> {
        fn e(what: &'static str) -> impl Fn(windows::core::Error) -> String {
            move |err| format!("probe {what}: {err}")
        }
        let tex = |bind: D3D11_BIND_FLAG| -> Result<ID3D11Texture2D, String> {
            let d = D3D11_TEXTURE2D_DESC {
                Width: 64,
                Height: 64,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: bind.0 as u32,
                ..Default::default()
            };
            let mut t = None;
            // SAFETY: valid descriptor / out pointer.
            unsafe { dev.CreateTexture2D(&d, None, Some(&mut t)) }.map_err(e("texture"))?;
            t.ok_or_else(|| "no texture".to_owned())
        };
        let (rt, sr) = (
            tex(D3D11_BIND_RENDER_TARGET)?,
            tex(D3D11_BIND_SHADER_RESOURCE)?,
        );
        // SAFETY: object creation with valid descriptors and out pointers.
        unsafe {
            let (mut rtv, mut srv, mut sampler, mut blend, mut raster, mut depth, mut cb) =
                (None, None, None, None, None, None, None);
            dev.CreateRenderTargetView(&rt, None, Some(&mut rtv))
                .map_err(e("rtv"))?;
            dev.CreateShaderResourceView(&sr, None, Some(&mut srv))
                .map_err(e("srv"))?;
            dev.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_POINT,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MaxLOD: 3.0,
                    ..Default::default()
                },
                Some(&mut sampler),
            )
            .map_err(e("sampler"))?;
            let mut bd = D3D11_BLEND_DESC::default();
            bd.RenderTarget[0].BlendEnable = true.into();
            bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
            bd.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
            bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
            bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
            bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
            bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
            bd.RenderTarget[0].RenderTargetWriteMask = 0x7;
            dev.CreateBlendState(&bd, Some(&mut blend))
                .map_err(e("blend"))?;
            dev.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_WIREFRAME,
                    CullMode: D3D11_CULL_FRONT,
                    ScissorEnable: true.into(),
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut raster),
            )
            .map_err(e("raster"))?;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D11_COMPARISON_GREATER_EQUAL,
                    ..Default::default()
                },
                Some(&mut depth),
            )
            .map_err(e("depth"))?;
            dev.CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: 256,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut cb),
            )
            .map_err(e("cb"))?;
            Ok(Self {
                rtv: rtv.ok_or("rtv")?,
                srv: srv.ok_or("srv")?,
                sampler: sampler.ok_or("sampler")?,
                blend: blend.ok_or("blend")?,
                raster: raster.ok_or("raster")?,
                depth: depth.ok_or("depth")?,
                cb: cb.ok_or("cb")?,
                viewport: D3D11_VIEWPORT {
                    TopLeftX: 1.0,
                    TopLeftY: 2.0,
                    Width: 30.0,
                    Height: 40.0,
                    MinDepth: 0.25,
                    MaxDepth: 0.75,
                },
                scissor: RECT {
                    left: 3,
                    top: 4,
                    right: 50,
                    bottom: 60,
                },
            })
        }
    }

    fn set(&self, ctx: &ID3D11DeviceContext) {
        // SAFETY: binding our own live objects on the immediate context.
        unsafe {
            ctx.OMSetRenderTargets(Some(&[Some(self.rtv.clone())]), None);
            ctx.OMSetBlendState(&self.blend, Some(&[0.1, 0.2, 0.3, 0.4]), 0x00FF_00FF);
            ctx.OMSetDepthStencilState(&self.depth, 7);
            ctx.RSSetState(&self.raster);
            ctx.RSSetViewports(Some(&[self.viewport]));
            ctx.RSSetScissorRects(Some(&[self.scissor]));
            ctx.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx.PSSetShaderResources(0, Some(&[Some(self.srv.clone())]));
            ctx.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            ctx.VSSetConstantBuffers(0, Some(&[Some(self.cb.clone())]));
        }
    }

    /// Differences between the bound state and what `set` bound.
    fn verify(&self, ctx: &ID3D11DeviceContext) -> Vec<String> {
        let mut bad = Vec::new();
        fn raw<T: Interface>(o: &Option<T>) -> Option<*mut std::ffi::c_void> {
            o.as_ref().map(Interface::as_raw)
        }
        // SAFETY: plain getters with correctly sized out buffers.
        unsafe {
            let mut rtvs = [None];
            ctx.OMGetRenderTargets(Some(&mut rtvs), None);
            if raw(&rtvs[0]) != Some(self.rtv.as_raw()) {
                bad.push("render target".to_owned());
            }
            let (mut blend, mut factor, mut mask) = (None, [0f32; 4], 0u32);
            ctx.OMGetBlendState(Some(&mut blend), Some(&mut factor), Some(&mut mask));
            if raw(&blend) != Some(self.blend.as_raw())
                || factor != [0.1, 0.2, 0.3, 0.4]
                || mask != 0x00FF_00FF
            {
                bad.push("blend state".to_owned());
            }
            let (mut depth, mut sref) = (None, 0u32);
            ctx.OMGetDepthStencilState(Some(&mut depth), Some(&mut sref));
            if raw(&depth) != Some(self.depth.as_raw()) || sref != 7 {
                bad.push("depth-stencil state".to_owned());
            }
            if ctx.RSGetState().ok().map(|r| r.as_raw()) != Some(self.raster.as_raw()) {
                bad.push("rasterizer state".to_owned());
            }
            let (mut n, mut vp) = (1u32, [D3D11_VIEWPORT::default()]);
            ctx.RSGetViewports(&mut n, Some(vp.as_mut_ptr()));
            if n != 1 || vp[0] != self.viewport {
                bad.push("viewport".to_owned());
            }
            let (mut n, mut sc) = (1u32, [RECT::default()]);
            ctx.RSGetScissorRects(&mut n, Some(sc.as_mut_ptr()));
            if n != 1 || sc[0] != self.scissor {
                bad.push("scissor".to_owned());
            }
            if ctx.IAGetPrimitiveTopology() != D3D11_PRIMITIVE_TOPOLOGY_POINTLIST {
                bad.push("topology".to_owned());
            }
            let mut srv = [None];
            ctx.PSGetShaderResources(0, Some(&mut srv));
            if raw(&srv[0]) != Some(self.srv.as_raw()) {
                bad.push("PS SRV 0".to_owned());
            }
            let mut smp = [None];
            ctx.PSGetSamplers(0, Some(&mut smp));
            if raw(&smp[0]) != Some(self.sampler.as_raw()) {
                bad.push("PS sampler 0".to_owned());
            }
            let mut cb = [None];
            ctx.VSGetConstantBuffers(0, Some(&mut cb));
            if raw(&cb[0]) != Some(self.cb.as_raw()) {
                bad.push("VS constant buffer 0".to_owned());
            }
            if ctx.IAGetInputLayout().is_ok() {
                bad.push("input layout (expected none)".to_owned());
            }
        }
        bad
    }
}

/// Read-back image: width, height, RGBA8 rows.
type Image = (u32, u32, Vec<u8>);

fn readback(
    dev: &ID3D11Device,
    ctx: &ID3D11DeviceContext,
    tex: &ID3D11Texture2D,
) -> Result<Image, String> {
    let mut d = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { tex.GetDesc(&mut d) };
    let sd = D3D11_TEXTURE2D_DESC {
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
        ..d
    };
    let mut staging = None;
    // SAFETY: valid descriptor / out pointer; copy between same-format textures; the
    // mapped rows are read within the mapped size.
    unsafe {
        dev.CreateTexture2D(&sd, None, Some(&mut staging))
            .map_err(|e| format!("staging: {e}"))?;
        let staging = staging.ok_or("no staging texture")?;
        ctx.CopyResource(&staging, tex);
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m))
            .map_err(|e| format!("map: {e}"))?;
        let row = d.Width as usize * 4;
        let mut out = Vec::with_capacity(row * d.Height as usize);
        for y in 0..d.Height as usize {
            out.extend_from_slice(std::slice::from_raw_parts(
                m.pData.cast::<u8>().add(y * m.RowPitch as usize),
                row,
            ));
        }
        ctx.Unmap(&staging, 0);
        Ok((d.Width, d.Height, out))
    }
}

fn clear(
    dev: &ID3D11Device,
    ctx: &ID3D11DeviceContext,
    tex: &ID3D11Texture2D,
) -> Result<(), String> {
    let mut rtv = None;
    // SAFETY: default view of our render target; cleared and released here.
    unsafe {
        dev.CreateRenderTargetView(tex, None, Some(&mut rtv))
            .map_err(|e| format!("clear view: {e}"))?;
        // A dark blue-grey "scene" (linear values through the sRGB view).
        ctx.ClearRenderTargetView(&rtv.ok_or("no view")?, &[0.02, 0.035, 0.06, 1.0]);
    }
    Ok(())
}

fn pixel(img: &Image, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * img.0 + x) * 4) as usize;
    [img.2[i], img.2[i + 1], img.2[i + 2], img.2[i + 3]]
}

unsafe extern "system" fn def_proc(
    hwnd: HWND,
    msg: u32,
    wp: windows::Win32::Foundation::WPARAM,
    lp: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    // SAFETY: default handling of our hidden window's messages.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

/// Renders the scripted preview at `size` (pixels); fails if a check fails.
pub fn offline_preview(size: [u32; 2]) -> Result<Preview, String> {
    let [w, h] = size;
    // SAFETY: window-class registration and creation with valid arguments; the window
    // is never shown and destroyed at the end.
    let hwnd = unsafe {
        let hinst = GetModuleHandleW(None).map_err(|e| format!("module: {e}"))?;
        let class = w!("void_overlay_preview");
        RegisterClassW(&WNDCLASSW {
            lpfnWndProc: Some(def_proc),
            hInstance: hinst.into(),
            lpszClassName: class,
            ..Default::default()
        });
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("void-inject overlay preview"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            w as i32,
            h as i32,
            None,
            None,
            Some(hinst.into()),
            None,
        )
        .map_err(|e| format!("window: {e}"))?
    };
    let r = run(hwnd, w, h);
    // SAFETY: our own window.
    unsafe {
        let _ = DestroyWindow(hwnd);
    }
    r
}

fn run(hwnd: HWND, w: u32, h: u32) -> Result<Preview, String> {
    let desc = DXGI_SWAP_CHAIN_DESC {
        BufferDesc: DXGI_MODE_DESC {
            Width: w,
            Height: h,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
            ..Default::default()
        },
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 1,
        OutputWindow: hwnd,
        Windowed: true.into(),
        SwapEffect: DXGI_SWAP_EFFECT_DISCARD,
        ..Default::default()
    };
    let (mut sc, mut dev, mut ctx) = (None, None, None);
    // SAFETY: valid arguments and out pointers.
    unsafe {
        D3D11CreateDeviceAndSwapChain(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&desc),
            Some(&mut sc),
            Some(&mut dev),
            None,
            Some(&mut ctx),
        )
    }
    .map_err(|e| format!("D3D11CreateDeviceAndSwapChain: {e}"))?;
    let (sc, dev, ctx): (IDXGISwapChain, ID3D11Device, ID3D11DeviceContext) = (
        sc.ok_or("no swap chain")?,
        dev.ok_or("no device")?,
        ctx.ok_or("no context")?,
    );
    // SAFETY: buffer 0 of our swap chain.
    let bb: ID3D11Texture2D = unsafe { sc.GetBuffer(0) }.map_err(|e| format!("GetBuffer: {e}"))?;
    let size = [w as f32, h as f32];
    let ppp = (size[1] / 1080.0).clamp(0.5, 4.0);
    let probe = Probe::new(&dev)?;
    let mut painter = Painter::new(&dev)?;
    let mut core = Core::new();
    core.step = Some(0.25);
    let mut backend = Backend {
        file: Settings {
            cvars: vec![
                ("image_anisotropy".into(), 16.0),
                ("r_shadowCullDistance".into(), 100.0),
                ("r_nope".into(), 1.0),
            ],
            // Lighting panel: own AO with one changed value (restore icon visible).
            ao_void: true,
            ao_strength: 1.5,
            ..Settings::default()
        },
        fsr_active: true,
        debug: 0,
        patches: PATCHES.iter().map(|p| p.default_on).collect(),
        saves: 0,
        photo_actions: 0,
        photo_active: true,
    };
    let mut report = Vec::new();
    let mut images = Vec::new();
    let mut failures = Vec::new();
    let center = |core: &Core, name: &str| -> Option<egui::Pos2> {
        core.ui
            .probes
            .iter()
            .find(|p| p.0 == name)
            .map(|p| egui::pos2(p.1.center().x * ppp, p.1.center().y * ppp))
    };
    let modifiers = egui::Modifiers::default();
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers,
    };
    // (target probe, action, image name)
    let script: [(&str, &str, Option<&str>); 48] = [
        // First frames of a window are a sizing pass; let it settle.
        ("", "", None),
        ("", "", Some("0_status")),
        ("FSR", "press", None),
        ("FSR", "release", Some("1_fsr_tab")),
        ("fsr_on", "move", Some("2_fsr_hover_toggle")),
        ("fsr_on", "press", None),
        ("fsr_on", "release", Some("3_fsr_toggled_off")),
        ("Cvars", "press", None),
        ("Cvars", "release", Some("4_cvars")),
        ("Patches", "press", None),
        ("Patches", "release", Some("5_patches")),
        ("Profiler", "press", None),
        ("Profiler", "release", Some("6_profiler")),
        ("Log", "press", None),
        ("Log", "release", Some("7_log")),
        ("Texture mods", "press", None),
        ("Texture mods", "release", Some("12_texture_mods")),
        ("Overlay", "press", None),
        ("Overlay", "release", Some("13_overlay")),
        ("hud_on", "press", None),
        ("hud_on", "release", None),
        ("notifications_on", "press", None),
        ("notifications_on", "release", Some("14_overlay_toggled")),
        ("save_settings", "press", None),
        ("save_settings", "release", Some("15_settings_saved")),
        ("Log", "press", None),
        ("Log", "release", None),
        ("FSR", "press", None),
        ("FSR", "release", None),
        ("fsr_on", "scroll", Some("19_fsr_scrolled")),
        ("fsr_expert", "press", None),
        ("fsr_expert", "release", Some("20_fsr_expert")),
        ("fsr_debug", "press", None),
        ("fsr_debug", "release", Some("21_fsr_debug_menu")),
        ("fsr_debug_motion", "press", None),
        ("fsr_debug_motion", "release", Some("21_fsr_debug_motion")),
        ("Lighting", "press", None),
        ("Lighting", "release", Some("22_lighting")),
        ("lighting_page", "scroll", None),
        ("lighting_page", "scroll", Some("23_lighting_shadows")),
        ("Reflections", "press", None),
        ("Reflections", "release", Some("24_reflections")),
        ("Photo", "press", None),
        ("Photo", "release", Some("25_photo_active")),
        ("photo_toggle", "press", None),
        ("photo_toggle", "release", Some("26_photo_left")),
        ("photo_toggle", "scroll", None),
        ("photo_toggle", "scroll", Some("27_photo_settings_probe")),
    ];
    let mut frame0_pixel = None;
    let initial_hud = backend.file.hud;
    let initial_notifications = backend.file.notifications;
    for (n, (target, action, save)) in script.into_iter().enumerate() {
        if target.is_empty() {
            input::push(egui::Event::PointerMoved(egui::pos2(
                size[0] * 0.6,
                size[1] * 0.5,
            )));
        } else {
            let p = center(&core, target).ok_or_else(|| {
                format!(
                    "no widget {target:?} (tab {:?}, widgets {:?})",
                    core.ui.tab, core.ui.probes
                )
            })?;
            if p.x < 0.0 || p.y < 0.0 || p.x >= size[0] || p.y >= size[1] {
                return Err(format!("widget {target:?} is outside the viewport: {p:?}"));
            }
            input::push(egui::Event::PointerMoved(p));
            match action {
                "press" => input::push(button(p, true)),
                "release" => input::push(button(p, false)),
                "scroll" => input::push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, -430.0),
                    phase: egui::TouchPhase::Move,
                    modifiers,
                }),
                _ => {}
            }
        }
        // Let scroll smoothing and collapsing animations settle before capturing
        // geometry or pressing the next widget.
        for _ in 0..if save.is_some() { 30 } else { 1 } {
            let snap = backend.snapshot(n as u32);
            clear(&dev, &ctx, &bb)?;
            probe.set(&ctx);
            let out = core.frame(
                size,
                size,
                None,
                Layers {
                    menu: Some((&snap, &mut backend)),
                    ..Layers::default()
                },
            );
            painter.paint(&ctx, &bb, &core.egui, out)?;
            let bad = probe.verify(&ctx);
            if !bad.is_empty() {
                failures.push(format!("step {n}: state not restored: {}", bad.join(", ")));
            }
            // SAFETY: unbind our probe state before the copies below.
            unsafe { ctx.ClearState() };
        }
        if let Some(name) = save {
            let img = readback(&dev, &ctx, &bb)?;
            if n == 1 {
                frame0_pixel = Some(pixel(&img, (500.0 * ppp) as u32, (40.0 * ppp) as u32));
            }
            images.push(PreviewImage {
                name: name.to_owned(),
                width: img.0,
                height: img.1,
                rgba: img.2,
            });
        }
        match (n, action) {
            (3, _) if core.ui.tab != ui::Tab::Fsr => failures.push("FSR tab click missed".into()),
            (6, _) if backend.fsr_active => failures.push("FSR checkbox click missed".into()),
            (14, _) if core.ui.tab != ui::Tab::Log => failures.push("Log tab click missed".into()),
            (16, _) if core.ui.tab != ui::Tab::TextureMods => {
                failures.push("Texture mods navigation click missed".into());
            }
            (18, _) if core.ui.tab != ui::Tab::Settings => {
                failures.push("Overlay navigation click missed".into());
            }
            (20, _) if backend.file.hud == initial_hud => {
                failures.push("HUD setting click missed".into());
            }
            (22, _) if backend.file.notifications == initial_notifications => {
                failures.push("Notifications setting click missed".into());
            }
            (24, _) if backend.saves != 1 => {
                failures.push("Save settings click missed".into());
            }
            (35, _) if backend.debug != fsr::DEBUG_MOTION => {
                failures.push("Scrolled FSR expert/debug-view selection missed".into());
            }
            (37, _) if core.ui.tab != ui::Tab::Lighting => {
                failures.push("Lighting tab click missed".into());
            }
            (41, _) if core.ui.tab != ui::Tab::Reflections => {
                failures.push("Reflections tab click missed".into());
            }
            (43, _) if core.ui.tab != ui::Tab::Photo => {
                failures.push("Photo tab click missed".into());
            }
            (45, _) if backend.photo_actions != 1 || backend.photo_active => {
                failures.push("Photo enter/leave click missed".into());
            }
            _ => {}
        }
    }
    report.push(format!(
        "{} scripted steps at {w}x{h} (pixels per point {ppp:.3}); pipeline state restored after every draw: {}",
        script.len(),
        failures.iter().all(|f| !f.contains("state"))
    ));
    report.push(format!(
        "clicks: FSR tab -> {:?}, FSR checkbox -> active {}, final tab {:?}",
        ui::Tab::Fsr,
        backend.fsr_active,
        core.ui.tab
    ));
    report.push(format!(
        "settings clicks: HUD {} -> {}, notifications {} -> {}, saves {}",
        initial_hud,
        backend.file.hud,
        initial_notifications,
        backend.file.notifications,
        backend.saves
    ));
    report.push(format!(
        "scrolled FSR expert controls: debug view {}",
        fsr::DEBUG_VIEWS[usize::from(backend.debug)]
    ));

    // Fallback path: a typed sRGB texture has no UNORM view; the painter logs and uses
    // the default (sRGB) view. The same first frame drawn there must come out brighter.
    let tex_desc = D3D11_TEXTURE2D_DESC {
        Width: w,
        Height: h,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
        ..Default::default()
    };
    let mut typed = None;
    // SAFETY: valid descriptor / out pointer.
    unsafe { dev.CreateTexture2D(&tex_desc, None, Some(&mut typed)) }
        .map_err(|e| format!("typed texture: {e}"))?;
    let typed = typed.ok_or("no typed texture")?;
    let mut painter2 = Painter::new(&dev)?;
    let mut core2 = Core::new();
    core2.step = Some(0.25);
    let mut backend2 = Backend {
        file: backend.file.clone(),
        fsr_active: true,
        debug: 0,
        patches: backend.patches.clone(),
        saves: 0,
        photo_actions: 0,
        photo_active: false,
    };
    backend2.file.fsr_upscale = false;
    for _ in 0..2 {
        input::push(egui::Event::PointerMoved(egui::pos2(
            size[0] * 0.6,
            size[1] * 0.5,
        )));
        let snap = backend2.snapshot(1);
        clear(&dev, &ctx, &typed)?;
        probe.set(&ctx);
        let out = core2.frame(
            size,
            size,
            None,
            Layers {
                menu: Some((&snap, &mut backend2)),
                ..Layers::default()
            },
        );
        painter2.paint(&ctx, &typed, &core2.egui, out)?;
    }
    let bad = probe.verify(&ctx);
    if !bad.is_empty() {
        failures.push(format!("fallback: state not restored: {}", bad.join(", ")));
    }
    // SAFETY: unbind the probe state before the copy.
    unsafe { ctx.ClearState() };
    let img = readback(&dev, &ctx, &typed)?;
    let fallback_pixel = pixel(&img, (500.0 * ppp) as u32, (40.0 * ppp) as u32);
    images.push(PreviewImage {
        name: "8_fallback_srgb_view".into(),
        width: img.0,
        height: img.1,
        rgba: img.2,
    });
    let unorm = frame0_pixel.ok_or("no first frame")?;
    report.push(format!(
        "window fill at ({}, {}) px: UNORM view {:?}, sRGB fallback view {:?}",
        (500.0 * ppp) as u32,
        (40.0 * ppp) as u32,
        unorm,
        fallback_pixel
    ));
    if u32::from(unorm[0]) + 20 > u32::from(fallback_pixel[0]) {
        failures.push("UNORM view did not store gamma values unconverted".into());
    }
    let section = always_on_layers(&dev, &ctx, &bb, &probe, size)?;
    images.extend(section.images);
    report.extend(section.report);
    failures.extend(section.failures);
    if failures.is_empty() {
        report.push("all checks passed".into());
        Ok(Preview {
            images,
            report: report.join("\n"),
        })
    } else {
        Err(format!(
            "{}\nFAILED: {}",
            report.join("\n"),
            failures.join("; ")
        ))
    }
}

/// Stand-in HUD numbers: 3 s of ~120 fps with noise and a few hitches; `shift` moves the
/// pattern so every frame's graph differs (as in game).
fn hud_stand_in(shift: usize) -> hud::HudData {
    let n = 360;
    hud::HudData {
        intervals: (0..n)
            .map(|i| {
                let k = i + shift;
                let ms = if k.is_multiple_of(97) {
                    19.5
                } else {
                    8.1 + ((k * 37) % 13) as f32 * 0.06
                };
                ((n - 1 - i) as f32 * hud::GRAPH_S / n as f32, ms)
            })
            .collect(),
        engine_cpu_ms: Some(7.84),
        engine_gpu_ms: Some(5.91),
        fsr_gpu_ms: Some(1.74),
    }
}

/// Pixels in `img` differing from the background colour within the given fraction of the
/// image (x0, y0, x1, y1).
fn changed(img: &Image, bg: [u8; 4], r: [f32; 4]) -> usize {
    let (w, h) = (img.0 as f32, img.1 as f32);
    let mut n = 0;
    for y in (r[1] * h) as u32..(r[3] * h) as u32 {
        for x in (r[0] * w) as u32..(r[2] * w) as u32 {
            n += usize::from(pixel(img, x, y) != bg);
        }
    }
    n
}

/// HUD alone, toasts alone (one per level, through `notify`), both with the menu closed;
/// pipeline state checked after each draw; per-frame CPU cost of build + paint.
/// Images, report lines and failed checks of a preview section.
#[derive(Default)]
struct Section {
    images: Vec<PreviewImage>,
    report: Vec<String>,
    failures: Vec<String>,
}

fn always_on_layers(
    dev: &ID3D11Device,
    ctx: &ID3D11DeviceContext,
    bb: &ID3D11Texture2D,
    probe: &Probe,
    size: [f32; 2],
) -> Result<Section, String> {
    let mut out = Section::default();
    let (images, report, failures) = (&mut out.images, &mut out.report, &mut out.failures);
    let mut painter = Painter::new(dev)?;
    let mut core = Core::new();
    core.step = Some(1.0 / 120.0);
    notify::info("FSR upscale 0.67 (2304x968)");
    notify::warn("applies at the next start (save settings first): texture_mods");
    notify::error("patch skip_intro: original bytes mismatch: [0f, 84, ...] at 0x140064f08");
    let mut toasts = hud::Toasts::default();
    let now = std::time::Instant::now();
    toasts.add(notify::take(), now);
    let hud = hud_stand_in(0);
    let mut draw = |hud: Option<(&hud::HudData, Corner)>,
                    toasts: Option<&hud::Toasts>,
                    name: Option<&str>|
     -> Result<(Option<Image>, f64), String> {
        clear(dev, ctx, bb)?;
        probe.set(ctx);
        let t = std::time::Instant::now();
        let out = core.frame(
            size,
            size,
            None,
            Layers {
                menu: None,
                hud,
                toasts: toasts.map(|t| (t, now)),
            },
        );
        painter.paint(ctx, bb, &core.egui, out)?;
        let us = t.elapsed().as_secs_f64() * 1e6;
        let bad = probe.verify(ctx);
        if !bad.is_empty() {
            return Err(format!(
                "{}: state not restored: {}",
                name.unwrap_or("timed frame"),
                bad.join(", ")
            ));
        }
        // SAFETY: unbind the probe state before the copy.
        unsafe { ctx.ClearState() };
        let img = name.map(|_| readback(dev, ctx, bb)).transpose()?;
        Ok((img, us))
    };
    // Background colour: an empty frame.
    let (bg, _) = draw(None, None, Some("empty"))?;
    let bg = bg.ok_or("no readback")?;
    let bg = pixel(&bg, bg.0 / 2, bg.1 / 2);
    let left = [0.0, 0.0, 0.25, 0.25];
    let right = [0.75, 0.0, 1.0, 0.25];
    for (name, h, t) in [
        ("9_hud", Some(&hud), None),
        ("10_toasts", None, Some(&toasts)),
        ("11_hud_and_toasts", Some(&hud), Some(&toasts)),
    ] {
        // The first frame lays out the areas, the next ones complete their fade-in.
        for _ in 0..30 {
            draw(h.map(|d| (d, Corner::TopLeft)), t, None)?;
        }
        let (img, _) = draw(h.map(|d| (d, Corner::TopLeft)), t, Some(name))?;
        let img = img.ok_or("no readback")?;
        let (l, r) = (changed(&img, bg, left), changed(&img, bg, right));
        report.push(format!(
            "{name}: changed pixels top-left quarter {l}, top-right quarter {r}"
        ));
        if (h.is_some() && l == 0) || (h.is_none() && l != 0) {
            failures.push(format!("{name}: HUD area wrong ({l} changed pixels)"));
        }
        if (t.is_some() && r == 0) || (t.is_none() && r != 0) {
            failures.push(format!("{name}: toast area wrong ({r} changed pixels)"));
        }
        images.push(PreviewImage {
            name: name.to_owned(),
            width: img.0,
            height: img.1,
            rgba: img.2,
        });
    }
    // Cost: egui build + tessellation + D3D11 draw calls (CPU side) per frame.
    for (label, with_toasts) in [("HUD", false), ("HUD + 3 toasts", true)] {
        let mut us = Vec::new();
        for i in 0..340 {
            let d = hud_stand_in(i);
            let (_, t) = draw(
                Some((&d, Corner::TopLeft)),
                with_toasts.then_some(&toasts),
                None,
            )?;
            if i >= 40 {
                us.push(t);
            }
        }
        us.sort_by(f64::total_cmp);
        let q = |f: f64| us[((us.len() - 1) as f64 * f).round() as usize];
        report.push(format!(
            "{label} CPU per frame at {}x{} (build + paint, {} frames): mean {:.1} us, p50 {:.1} us, p99 {:.1} us",
            size[0],
            size[1],
            us.len(),
            us.iter().sum::<f64>() / us.len() as f64,
            q(0.5),
            q(0.99)
        ));
    }
    let empty = hud::HudData::default();
    let no_fsr = hud::HudData {
        fsr_gpu_ms: None,
        ..hud.clone()
    };
    for (name, data, corner, region, opposite) in [
        ("16_hud_top_right", &hud, Corner::TopRight, right, left),
        (
            "17_hud_bottom_left",
            &no_fsr,
            Corner::BottomLeft,
            [0.0, 0.75, 0.25, 1.0],
            right,
        ),
        (
            "18_hud_no_data",
            &empty,
            Corner::BottomRight,
            [0.75, 0.75, 1.0, 1.0],
            left,
        ),
    ] {
        for _ in 0..30 {
            draw(Some((data, corner)), None, None)?;
        }
        let (img, _) = draw(Some((data, corner)), None, Some(name))?;
        let img = img.ok_or("no corner readback")?;
        if changed(&img, bg, region) == 0 || changed(&img, bg, opposite) != 0 {
            failures.push(format!("{name}: incorrect HUD anchor"));
        }
        report.push(format!("{name}: rendered at {}", corner.name()));
        images.push(PreviewImage {
            name: name.into(),
            width: img.0,
            height: img.1,
            rgba: img.2,
        });
    }
    Ok(out)
}
