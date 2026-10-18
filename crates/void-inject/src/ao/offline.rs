//! Offline check of the AO replacement (`examples/ao_offline.rs`): a private D3D11 device,
//! the game's HBAO+ DLL loaded from the game folder with a context created exactly as the
//! engine does (version {3, 0, 0, 0}), our `RenderAO` hook installed on it through the same
//! code path as in the game (`hook_render_ao`), and both modes called through the context's
//! vtable with structs laid out and filled like `sub_140305620`'s. Scene: synthetic
//! reversed-Z R24G8 depth of a floor, two walls (corner), a box and sky. Checks per
//! configuration: status `GFSDK_SSAO_OK`, our path taken, every D3D11 state of the
//! deferred context unchanged, RGB written only inside the viewport and alpha preserved, no
//! NaN, open floor ~1, corner/contact clearly darker; GPU time from timestamp queries.
//! External `.x` source (S1 entry point, `external.rs`) after HBAO+ and after ours: replace
//! and min give exactly the expected red channel inside the viewport, G/B/A and everything
//! outside equal the run without it, state restored; with nothing registered (before and
//! after a registration) the output is bit-identical.

use std::ffi::c_void;
use std::path::Path;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
    D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
};
use windows::Win32::System::LibraryLoader::LoadLibraryW;
use windows::core::HSTRING;

use super::*;
use crate::profiler::{get_data, make_query};

/// A grey image for inspection (written as PNG by the example).
pub struct Image {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub gray: Vec<u8>,
}

pub struct OfflineReport {
    pub text: String,
    pub images: Vec<Image>,
    /// Failed checks (empty = all passed).
    pub failures: Vec<String>,
}

const FOV_Y: f32 = std::f32::consts::FRAC_PI_3; // 60 degrees
const NEAR: f32 = 0.01; // r_znear (in-game Ultra dump)
const FAR: f32 = 7000.0; // r_zfar
const JITTER: [f32; 2] = [0.0004, -0.0003];
const SENTINEL: [f32; 4] = [0.25, 0.5, 0.75, 0.625];
const TIMING_ITERS: usize = 40;

/// Synthetic scene in the screen-consistent view frame (x right, y up, z forward, camera
/// at the origin, eye height 1.6 m): floor y = -1.6, back wall z = 6 and right wall x = 2.5
/// (both up to y = 1, sky above), box x -1.6..-0.6, y -1.6..-0.9, z 3.2..4.4.
fn trace(dir: [f32; 3]) -> Option<f32> {
    let mut best = f32::INFINITY;
    let mut hit = |t: f32, ok: bool| {
        if ok && t > 0.0 && t < best {
            best = t;
        }
    };
    if dir[1] < 0.0 {
        hit(-1.6 / dir[1], true);
    }
    let t = 6.0 / dir[2];
    hit(t, dir[1] * t <= 1.0);
    if dir[0] > 0.0 {
        let t = 2.5 / dir[0];
        hit(t, dir[1] * t <= 1.0 && dir[2] * t <= 6.0);
    }
    let (lo, hi) = ([-1.6f32, -1.6, 3.2], [-0.6f32, -0.9, 4.4]);
    let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        let (u, v) = ((lo[a]) / dir[a], (hi[a]) / dir[a]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    }
    hit(t0, t0 <= t1);
    best.is_finite().then_some(best)
}

/// Pixel of a view-frame point inside a region of `size`.
fn pixel(p: &Proj, size: [u32; 2], v: [f32; 3]) -> [u32; 2] {
    let ndc = [
        v[0] / v[2] * p.p00 + p.jitter[0],
        v[1] / v[2] * p.p11 + p.jitter[1],
    ];
    [
        ((ndc[0] + 1.0) * 0.5 * size[0] as f32) as u32,
        ((1.0 - ndc[1]) * 0.5 * size[1] as f32) as u32,
    ]
}

/// R24G8 texels of the full texture: the scene inside `region` (origin 0, 0), a wall
/// 5 cm in front of the camera outside it (must not influence the result).
fn depth_texels(p: &Proj, tex: [u32; 2], region: [u32; 2]) -> Vec<u32> {
    let hw = |z: f32| {
        let d = (p.depth_add + p.depth_mul / z).clamp(0.0, 1.0);
        ((d * 16_777_215.0).round() as u32) | (5 << 24)
    };
    let mut out = Vec::with_capacity((tex[0] * tex[1]) as usize);
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            out.push(if x < region[0] && y < region[1] {
                let ndc = [
                    (x as f32 + 0.5) / region[0] as f32 * 2.0 - 1.0,
                    1.0 - (y as f32 + 0.5) / region[1] as f32 * 2.0,
                ];
                let dir = [
                    (ndc[0] - p.jitter[0]) / p.p00,
                    (ndc[1] - p.jitter[1]) / p.p11,
                    1.0,
                ];
                trace(dir).map_or(0, hw)
            } else {
                hw(0.05)
            });
        }
    }
    out
}

pub(crate) struct Gpu {
    pub(crate) dev: ID3D11Device,
    pub(crate) imm: ID3D11DeviceContext,
    pub(crate) def: ID3D11DeviceContext,
}

impl Gpu {
    /// Hardware device (feature level 11.1 or 11.0), its immediate and one deferred context.
    pub(crate) fn new() -> Result<Self, String> {
        let (mut dev, mut imm) = (None, None);
        // SAFETY: device creation with valid arguments and out pointers.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut dev),
                None,
                Some(&mut imm),
            )
        }
        .map_err(create_err("D3D11CreateDevice"))?;
        let dev: ID3D11Device = dev.ok_or("no device")?;
        let mut def = None;
        // SAFETY: valid out pointer.
        unsafe { dev.CreateDeferredContext(0, Some(&mut def)) }
            .map_err(create_err("deferred context"))?;
        Ok(Gpu {
            dev,
            imm: imm.ok_or("no context")?,
            def: def.ok_or("no deferred context")?,
        })
    }

    pub(crate) fn texture(
        &self,
        size: [u32; 2],
        format: DXGI_FORMAT,
        bind: i32,
        data: Option<&[u32]>,
    ) -> Result<ID3D11Texture2D, String> {
        let d = D3D11_TEXTURE2D_DESC {
            Width: size[0],
            Height: size[1],
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind as u32,
            ..Default::default()
        };
        let init = data.map(|b| D3D11_SUBRESOURCE_DATA {
            pSysMem: b.as_ptr().cast(),
            SysMemPitch: size[0] * 4,
            SysMemSlicePitch: 0,
        });
        let mut t = None;
        // SAFETY: valid descriptor; the initial data covers the declared size (4 B/texel).
        unsafe {
            self.dev
                .CreateTexture2D(&d, init.as_ref().map(|i| i as *const _), Some(&mut t))
        }
        .map_err(create_err("texture"))?;
        t.ok_or_else(|| "no texture".to_owned())
    }

    /// RGBA32F readback.
    pub(crate) fn read(
        &self,
        tex: &ID3D11Texture2D,
        size: [u32; 2],
    ) -> Result<Vec<[f32; 4]>, String> {
        let d = D3D11_TEXTURE2D_DESC {
            Width: size[0],
            Height: size[1],
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..Default::default()
        };
        let mut st = None;
        // SAFETY (whole block): staging copy of an identical texture, mapped for reading
        // within the mapped pitch, unmapped afterwards.
        unsafe {
            self.dev
                .CreateTexture2D(&d, None, Some(&mut st))
                .map_err(create_err("staging"))?;
            let st = st.ok_or("no staging")?;
            self.imm.CopyResource(&st, tex);
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            self.imm
                .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
                .map_err(create_err("Map staging"))?;
            let mut out = Vec::with_capacity((size[0] * size[1]) as usize);
            for y in 0..size[1] as usize {
                let row = m
                    .pData
                    .cast::<u8>()
                    .add(y * m.RowPitch as usize)
                    .cast::<[f32; 4]>();
                out.extend_from_slice(std::slice::from_raw_parts(row, size[0] as usize));
            }
            self.imm.Unmap(&st, 0);
            Ok(out)
        }
    }

    /// Median GPU time of executing `list` (timestamps around `ExecuteCommandList`).
    pub(crate) fn time(&self, list: &ID3D11CommandList) -> Result<f64, String> {
        let mut ms = Vec::with_capacity(TIMING_ITERS);
        let mut bad = (0, 0);
        for _ in 0..TIMING_ITERS {
            let q = |k| make_query(&self.dev, k).ok_or_else(|| "query".to_owned());
            let (dj, t0, t1) = (
                q(D3D11_QUERY_TIMESTAMP_DISJOINT)?,
                q(D3D11_QUERY_TIMESTAMP)?,
                q(D3D11_QUERY_TIMESTAMP)?,
            );
            // SAFETY: queries and list of this device on its immediate context.
            unsafe {
                self.imm.Begin(&dj);
                self.imm.End(&t0);
                self.imm.ExecuteCommandList(list, false);
                self.imm.End(&t1);
                self.imm.End(&dj);
                self.imm.Flush();
            }
            let mut d = D3D11_QUERY_DATA_TIMESTAMP_DISJOINT::default();
            let start = std::time::Instant::now();
            while get_data(&self.imm, &dj, (&raw mut d).cast(), size_of_val(&d) as u32).0 != 0 {
                if start.elapsed().as_secs() > 5 {
                    return Err("timestamp query timed out".into());
                }
                std::thread::yield_now();
            }
            let (mut a, mut b) = (0u64, 0u64);
            while get_data(&self.imm, &t0, (&raw mut a).cast(), 8).0 != 0
                || get_data(&self.imm, &t1, (&raw mut b).cast(), 8).0 != 0
            {
                if start.elapsed().as_secs() > 5 {
                    return Err("timestamp query timed out".into());
                }
                std::thread::yield_now();
            }
            if d.Disjoint.as_bool() {
                bad.0 += 1;
            } else if b <= a {
                bad.1 += 1;
            } else {
                ms.push((b - a) as f64 * 1000.0 / d.Frequency as f64);
            }
        }
        if ms.len() < TIMING_ITERS / 2 {
            return Err(format!(
                "timestamps unusable: {} disjoint, {} not increasing of {TIMING_ITERS}",
                bad.0, bad.1
            ));
        }
        ms.sort_by(f64::total_cmp);
        Ok(ms[ms.len() / 2])
    }
}

/// Deterministic test term in [0.05, 0.95] for a `region`-sized external source.
pub(crate) fn external_pattern(region: [u32; 2]) -> Vec<f32> {
    (0..region[1])
        .flat_map(|y| (0..region[0]).map(move |x| (x, y)))
        .map(|(x, y)| 0.05 + 0.9 * ((x * 7 + y * 13) % 61) as f32 / 60.0)
        .collect()
}

/// A registered-source stand-in: R32F texture of `values` (`size`, row-major) and its SRV.
pub(crate) fn external_source(
    g: &Gpu,
    size: [u32; 2],
    values: &[f32],
    mode: ExternalMode,
) -> Result<ExternalSource, String> {
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    let t = g.texture(
        size,
        DXGI_FORMAT_R32_FLOAT,
        D3D11_BIND_SHADER_RESOURCE.0,
        Some(&bits),
    )?;
    let mut srv = None;
    // SAFETY: default view of the texture above.
    unsafe { g.dev.CreateShaderResourceView(&t, None, Some(&mut srv)) }
        .map_err(create_err("source SRV"))?;
    Ok(ExternalSource {
        srv: srv.ok_or("no source SRV")?,
        size,
        mode,
        reads: std::sync::Arc::default(),
        // The `RenderAO` call counter: one per call here (the read-stamp check).
        clock: passes,
    })
}

/// Known, non-default state bound on the deferred context before each call (distinct
/// objects in every slot the AO path touches, plus an OM UAV and a scissor rect).
pub(crate) struct Probe {
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    cs: ID3D11ComputeShader,
    srv: ID3D11ShaderResourceView,
    uav: ID3D11UnorderedAccessView,
    om_uav: ID3D11UnorderedAccessView,
    rtv: ID3D11RenderTargetView,
    dsv: ID3D11DepthStencilView,
    cb: ID3D11Buffer,
    sampler: ID3D11SamplerState,
    blend: ID3D11BlendState,
    dss: ID3D11DepthStencilState,
    raster: ID3D11RasterizerState,
}

const PROBE_HLSL: &str = r"
float4 vs_probe(uint i : SV_VertexID) : SV_Position { return float4(i, 0, 0, 1); }
float4 ps_probe() : SV_Target { return 0.25; }
RWTexture2D<float> probe_out : register(u0);
[numthreads(1, 1, 1)] void cs_probe(uint3 id : SV_DispatchThreadID) { probe_out[id.xy] = 1; }
";

impl Probe {
    pub(crate) fn new(g: &Gpu) -> Result<Self, String> {
        let dev = &g.dev;
        let c = |e, t| hlsl::compile(PROBE_HLSL, e, t);
        let srv_bind = D3D11_BIND_SHADER_RESOURCE.0;
        let uav_bind = D3D11_BIND_UNORDERED_ACCESS.0;
        let small = [64, 64];
        let srv_tex = g.texture(small, DXGI_FORMAT_R8G8B8A8_UNORM, srv_bind, None)?;
        let uav_tex = g.texture(small, DXGI_FORMAT_R32_FLOAT, uav_bind, None)?;
        let om_tex = g.texture(small, DXGI_FORMAT_R32_FLOAT, uav_bind, None)?;
        let rt_tex = g.texture(
            small,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D11_BIND_RENDER_TARGET.0,
            None,
        )?;
        let ds_tex = g.texture(
            small,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            D3D11_BIND_DEPTH_STENCIL.0,
            None,
        )?;
        // SAFETY (whole block): object creation with valid descriptors and out pointers.
        unsafe {
            let (mut vs, mut ps, mut cs) = (None, None, None);
            dev.CreateVertexShader(&c(s!("vs_probe"), s!("vs_5_0"))?, None, Some(&mut vs))
                .map_err(create_err("probe VS"))?;
            dev.CreatePixelShader(&c(s!("ps_probe"), s!("ps_5_0"))?, None, Some(&mut ps))
                .map_err(create_err("probe PS"))?;
            dev.CreateComputeShader(&c(s!("cs_probe"), s!("cs_5_0"))?, None, Some(&mut cs))
                .map_err(create_err("probe CS"))?;
            let (mut srv, mut uav, mut om_uav, mut rtv, mut dsv) = (None, None, None, None, None);
            dev.CreateShaderResourceView(&srv_tex, None, Some(&mut srv))
                .map_err(create_err("probe SRV"))?;
            dev.CreateUnorderedAccessView(&uav_tex, None, Some(&mut uav))
                .map_err(create_err("probe UAV"))?;
            dev.CreateUnorderedAccessView(&om_tex, None, Some(&mut om_uav))
                .map_err(create_err("probe OM UAV"))?;
            dev.CreateRenderTargetView(&rt_tex, None, Some(&mut rtv))
                .map_err(create_err("probe RTV"))?;
            dev.CreateDepthStencilView(&ds_tex, None, Some(&mut dsv))
                .map_err(create_err("probe DSV"))?;
            let cb = dynamic_cb(dev, 64)?;
            let mut sampler = None;
            dev.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_ANISOTROPIC,
                    AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                    AddressV: D3D11_TEXTURE_ADDRESS_MIRROR,
                    AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                    MaxAnisotropy: 3,
                    MaxLOD: 7.0,
                    ..Default::default()
                },
                Some(&mut sampler),
            )
            .map_err(create_err("probe sampler"))?;
            let mut bd = D3D11_BLEND_DESC::default();
            bd.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D11_BLEND_SRC_ALPHA,
                DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D11_BLEND_OP_ADD,
                SrcBlendAlpha: D3D11_BLEND_ONE,
                DestBlendAlpha: D3D11_BLEND_ZERO,
                BlendOpAlpha: D3D11_BLEND_OP_ADD,
                RenderTargetWriteMask: 0x0B,
            };
            let mut blend = None;
            dev.CreateBlendState(&bd, Some(&mut blend))
                .map_err(create_err("probe blend"))?;
            let mut dss = None;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D11_COMPARISON_GREATER_EQUAL,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut dss),
            )
            .map_err(create_err("probe depth state"))?;
            let mut raster = None;
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
            .map_err(create_err("probe raster"))?;
            Ok(Self {
                vs: vs.ok_or("probe VS")?,
                ps: ps.ok_or("probe PS")?,
                cs: cs.ok_or("probe CS")?,
                srv: srv.ok_or("probe SRV")?,
                uav: uav.ok_or("probe UAV")?,
                om_uav: om_uav.ok_or("probe OM UAV")?,
                rtv: rtv.ok_or("probe RTV")?,
                dsv: dsv.ok_or("probe DSV")?,
                cb,
                sampler: sampler.ok_or("probe sampler")?,
                blend: blend.ok_or("probe blend")?,
                dss: dss.ok_or("probe dss")?,
                raster: raster.ok_or("probe raster")?,
            })
        }
    }

    /// Binds the probe state and the engine's viewport for `region` on `ctx`.
    pub(crate) fn bind(&self, ctx: &ID3D11DeviceContext, region: [u32; 2]) {
        let srv = Some(self.srv.clone());
        let cb = Some(self.cb.clone());
        let smp = Some(self.sampler.clone());
        let keep = [u32::MAX; 1];
        // SAFETY (whole block): bindings of live objects of this device.
        unsafe {
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(&self.ps, None);
            ctx.CSSetShader(&self.cs, None);
            ctx.PSSetShaderResources(0, Some(&[srv.clone(), srv.clone(), srv.clone()]));
            ctx.CSSetShaderResources(0, Some(&[srv.clone(), srv.clone(), srv]));
            ctx.PSSetConstantBuffers(0, Some(&[cb.clone(), cb.clone(), cb.clone()]));
            ctx.CSSetConstantBuffers(0, Some(&[cb.clone(), cb.clone(), cb]));
            ctx.PSSetSamplers(0, Some(&[smp.clone(), smp.clone()]));
            ctx.CSSetSamplers(0, Some(&[smp.clone(), smp]));
            let uav = [Some(self.uav.clone())];
            ctx.CSSetUnorderedAccessViews(0, 1, Some(uav.as_ptr()), Some(keep.as_ptr()));
            let om = [Some(self.om_uav.clone())];
            ctx.OMSetRenderTargetsAndUnorderedAccessViews(
                Some(&[Some(self.rtv.clone())]),
                &self.dsv,
                1,
                1,
                Some(om.as_ptr()),
                Some(keep.as_ptr()),
            );
            ctx.OMSetBlendState(&self.blend, Some(&[0.1, 0.2, 0.3, 0.4]), 0x0F0F_0F0F);
            ctx.OMSetDepthStencilState(&self.dss, 7);
            ctx.RSSetState(&self.raster);
            // As `sub_1402BC610` before RenderAO: the view's (scaled) viewport.
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: region[0] as f32,
                Height: region[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.RSSetScissorRects(Some(&[windows::Win32::Foundation::RECT {
                left: 1,
                top: 2,
                right: 300,
                bottom: 200,
            }]));
        }
    }
}

/// Slots compared by the state check (wider than what the AO path saves).
pub(crate) const WIDE: Slots = Slots {
    srv: 16,
    cb: 8,
    sampler: 4,
    uav: 8,
};

struct Run {
    status: u32,
    ours: bool,
    state_diff: Vec<String>,
    pixels: Vec<[f32; 4]>,
    gpu_ms: f64,
}

#[derive(Default)]
struct Stats {
    nan: usize,
    outside_written: usize,
    alpha_changed: usize,
    not_grey: usize,
    min: f32,
    mean: f64,
}

fn stats(px: &[[f32; 4]], tex: [u32; 2], region: [u32; 2]) -> Stats {
    let mut s = Stats {
        min: f32::INFINITY,
        ..Default::default()
    };
    let mut n = 0usize;
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            let p = px[(y * tex[0] + x) as usize];
            if p.iter().any(|v| !v.is_finite()) {
                s.nan += 1;
                continue;
            }
            if x < region[0] && y < region[1] {
                if p[3] != SENTINEL[3] {
                    s.alpha_changed += 1;
                }
                if p[0] != p[1] || p[1] != p[2] || !(0.0..=1.0).contains(&p[0]) {
                    s.not_grey += 1;
                }
                s.min = s.min.min(p[0]);
                s.mean += f64::from(p[0]);
                n += 1;
            } else if p != SENTINEL {
                s.outside_written += 1;
            }
        }
    }
    s.mean /= n.max(1) as f64;
    s
}

/// Mean AO in the 5x5 block around `at`.
fn sample(px: &[[f32; 4]], tex: [u32; 2], at: [u32; 2]) -> f32 {
    let mut sum = 0.0;
    for dy in 0..5 {
        for dx in 0..5 {
            let x = (at[0] + dx).saturating_sub(2).min(tex[0] - 1);
            let y = (at[1] + dy).saturating_sub(2).min(tex[1] - 1);
            sum += px[(y * tex[0] + x) as usize][0];
        }
    }
    sum / 25.0
}

fn gray(px: &[[f32; 4]], tex: [u32; 2], region: [u32; 2], name: String) -> Image {
    let mut g = Vec::with_capacity((region[0] * region[1]) as usize);
    for y in 0..region[1] {
        for x in 0..region[0] {
            g.push((px[(y * tex[0] + x) as usize][0].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    Image {
        name,
        width: region[0],
        height: region[1],
        gray: g,
    }
}

/// See the module documentation. `game_dir` holds `GFSDK_SSAO_D3D11.win64.dll`.
pub fn offline_check(game_dir: &Path, size: [u32; 2], scale: f32) -> Result<OfflineReport, String> {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let mut images = Vec::new();
    let g = Gpu::new()?;

    // HBAO+ context as the engine creates it, then our hook through the game's path.
    let dll = game_dir.join(HBAO_DLL);
    // SAFETY: loading the game's HBAO+ DLL by explicit path (read-only use).
    let module = unsafe { LoadLibraryW(&HSTRING::from(dll.as_path())) }
        .map_err(|e| format!("LoadLibraryW({}): {e}", dll.display()))?;
    verify_module(module)?;
    // SAFETY: export lookup in the verified module.
    let create = unsafe { GetProcAddress(module, s!("GFSDK_SSAO_CreateContext_D3D11")) }
        .ok_or("GFSDK_SSAO_CreateContext_D3D11 not exported")?;
    // SAFETY: the export has the `CreateContextFn` signature (header + `sub_140305620` call).
    let create = unsafe {
        std::mem::transmute::<unsafe extern "system" fn() -> isize, CreateContextFn>(create)
    };
    let mut hctx: *mut c_void = std::ptr::null_mut();
    let version = Version {
        major: 3,
        minor: 0,
        branch: 0,
        revision: 0,
    };
    // SAFETY: live device, out pointer, no custom heap, version struct as the game passes.
    let st = unsafe { create(g.dev.as_raw(), &mut hctx, std::ptr::null(), &version) };
    if st != GFSDK_SSAO_OK || hctx.is_null() {
        return Err(format!("GFSDK_SSAO_CreateContext_D3D11 status {st}"));
    }
    hook_render_ao(hctx as u64)?;
    let vtable = mem::u64_at(hctx as u64).ok_or("context unreadable")?;
    let render = mem::u64_at(vtable + 8).ok_or("vtable unreadable")?;
    // SAFETY: vtable slot 1 = RenderAO (hooked: MinHook jumps to `render_ao_detour`).
    let render_ao = unsafe { std::mem::transmute::<u64, RenderAoFn>(render) };
    report.push(format!(
        "HBAO+ {} loaded (SHA-256 verified), context created with version 3.0.0.0; RenderAO {render:#x} hooked",
        dll.display()
    ));
    let code = bytecode()?;
    report.push(format!("our shaders compiled in {:.0} ms", code.compile_ms));
    external::compile_now()?;
    let probe = Probe::new(&g)?;

    let depth_bind = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_DEPTH_STENCIL.0;
    let out_bind = D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0;
    let region_scaled = [
        crate::fsr_scale::snap_render(size[0], scale),
        crate::fsr_scale::snap_render(size[1], scale),
    ];
    for (case, region) in [("full", size), ("scaled", region_scaled)] {
        let data = engine_projection(
            region[0] as f32 / region[1] as f32,
            FOV_Y,
            NEAR,
            FAR,
            JITTER,
        );
        let proj = projection(&data, 1)?;
        let texels = depth_texels(&proj, size, region);
        let depth = g.texture(size, DXGI_FORMAT_R24G8_TYPELESS, depth_bind, Some(&texels))?;
        let out = g.texture(size, DXGI_FORMAT_R32G32B32A32_FLOAT, out_bind, None)?;
        let (mut srv, mut rtv) = (None, None);
        // SAFETY: views of the textures above with valid descriptors.
        unsafe {
            g.dev
                .CreateShaderResourceView(
                    &depth,
                    Some(&D3D11_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                        ViewDimension: D3D11_SRV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D11_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: 1,
                            },
                        },
                    }),
                    Some(&mut srv),
                )
                .map_err(create_err("depth SRV"))?;
            g.dev
                .CreateRenderTargetView(&out, None, Some(&mut rtv))
                .map_err(create_err("output RTV"))?;
        }
        let (srv, rtv): (ID3D11ShaderResourceView, ID3D11RenderTargetView) =
            (srv.ok_or("no SRV")?, rtv.ok_or("no RTV")?);
        let render_scale =
            (region[0] as f32 / size[0] as f32).min(region[1] as f32 / size[1] as f32);
        let input = InputData {
            depth: InputDepthData {
                depth_texture_type: 0,
                projection: Matrix { data, layout: 1 },
                meters_to_view_space_units: 1.0,
                viewport: InputViewport {
                    enable: 0,
                    top_left_x: 0,
                    top_left_y: 0,
                    width: 0,
                    height: 0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                depth_srv: srv.as_raw(),
            },
            normal: InputNormalData {
                enable: 0,
                world_to_view: Matrix {
                    data: [0.0; 16],
                    layout: 0,
                },
                decode_scale: 1.0,
                decode_bias: 0.0,
                normal_srv: std::ptr::null_mut(),
            },
        };
        let params = Parameters::game(render_scale);
        let none_blend = BlendState {
            mode: 0,
            custom: CustomBlend {
                blend_state: std::ptr::null_mut(),
                blend_factor: std::ptr::null(),
            },
        };
        let pass = BlendPass {
            blend: none_blend,
            depth_stencil: DepthStencilState {
                mode: 0,
                depth_stencil_state: std::ptr::null_mut(),
                stencil_ref: 0,
            },
        };
        let output = Output {
            rtv: rtv.as_raw(),
            blend: none_blend,
            two_pass: TwoPassBlend {
                enable: 0,
                dsv: std::ptr::null_mut(),
                first: pass,
                second: pass,
            },
        };
        let points = [
            ("open floor", [0.6, -1.6, 4.2]),
            ("floor/back-wall corner", [0.0, -1.6, 5.97]),
            ("box contact", [-0.55, -1.6, 3.8]),
            ("three-plane corner", [2.45, -1.6, 5.95]),
        ];
        let at: Vec<[u32; 2]> = points
            .iter()
            .map(|(_, v)| pixel(&proj, region, *v))
            .collect();
        report.push(format!(
            "\n{case}: texture {}x{}, viewport {}x{} (render scale {render_scale:.3}); test pixels {:?}",
            size[0], size[1], region[0], region[1], at
        ));
        let mut configs: Vec<(String, Option<(u8, bool)>)> =
            vec![("HBAO+ (original)".into(), None)];
        for half in [false, true] {
            for q in 0..QUALITIES.len() as u8 {
                configs.push((
                    format!(
                        "ours {} {}",
                        quality_name(q),
                        if half { "half" } else { "full" }
                    ),
                    Some((q, half)),
                ));
            }
        }
        for (label, cfg) in configs {
            settings::edit(|f| {
                f.ao_void = cfg.is_some();
                if let Some((q, h)) = cfg {
                    f.ao_quality = q;
                    f.ao_half_res = h;
                }
            });
            let run = run_once(
                &g, &probe, render_ao, hctx, &input, &params, &output, region, &out, &rtv,
            )?;
            let st = stats(&run.pixels, size, region);
            let vals: Vec<f32> = at.iter().map(|&p| sample(&run.pixels, size, p)).collect();
            report.push(format!(
                "  {label:<22} GPU {:.3} ms | AO open {:.3} corner {:.3} box {:.3} 3-corner {:.3} | region mean {:.3} min {:.3} | NaN {} outside-written {} alpha-changed {} non-grey {} | state {}",
                run.gpu_ms,
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                st.mean,
                st.min,
                st.nan,
                st.outside_written,
                st.alpha_changed,
                st.not_grey,
                if run.state_diff.is_empty() {
                    "restored".to_owned()
                } else {
                    format!("CHANGED: {}", run.state_diff.join(", "))
                }
            ));
            let name = label.replace(['(', ')', '+'], "").replace(' ', "_");
            if cfg.is_none()
                || cfg == Some((DEFAULT_QUALITY, false))
                || cfg == Some((DEFAULT_QUALITY, true))
            {
                images.push(gray(&run.pixels, size, region, format!("{case}_{name}")));
            }
            let Some(_) = cfg else { continue };
            let mut fail = |ok: bool, what: String| {
                if !ok {
                    failures.push(format!("{case} {label}: {what}"));
                }
            };
            fail(
                run.status == GFSDK_SSAO_OK,
                format!("status {}", run.status),
            );
            fail(run.ours, "our path not taken (fell back to HBAO+)".into());
            fail(
                run.state_diff.is_empty(),
                format!("state changed: {:?}", run.state_diff),
            );
            fail(st.nan == 0, format!("{} non-finite pixels", st.nan));
            fail(
                st.outside_written == 0,
                format!("{} pixels outside the viewport written", st.outside_written),
            );
            fail(
                st.alpha_changed == 0,
                format!("alpha changed on {} pixels", st.alpha_changed),
            );
            fail(
                st.not_grey == 0,
                format!("{} pixels not grey AO in [0, 1]", st.not_grey),
            );
            fail(
                (vals[0] - 1.0).abs() <= 0.05,
                format!("open floor AO {:.3}", vals[0]),
            );
            for (i, v) in vals.iter().enumerate().skip(1) {
                fail(*v < 0.8, format!("{} AO {v:.3} not < 0.8", points[i].0));
            }
        }
        // External `.x` source after HBAO+ and after ours (default quality, half res).
        let ext_vals = external_pattern(region);
        let ext = external_source(&g, region, &ext_vals, ExternalMode::Replace)?;
        for (label, cfg) in [("HBAO+", None), ("ours", Some((DEFAULT_QUALITY, true)))] {
            settings::edit(|f| {
                f.ao_void = cfg.is_some();
                if let Some((q, h)) = cfg {
                    f.ao_quality = q;
                    f.ao_half_res = h;
                }
            });
            set_external_x(None);
            // HBAO+ itself leaves OM UAVs changed; the external draw must add nothing.
            let Run {
                pixels: base,
                state_diff: base_diff,
                ..
            } = run_once(
                &g, &probe, render_ao, hctx, &input, &params, &output, region, &out, &rtv,
            )?;
            for mode in [ExternalMode::Replace, ExternalMode::Min] {
                set_external_x(Some(ExternalSource {
                    mode,
                    ..ext.clone()
                }));
                let applied = external::counters(Channel::X).0;
                let run = run_once(
                    &g, &probe, render_ao, hctx, &input, &params, &output, region, &out, &rtv,
                )?;
                set_external_x(None);
                let applied = external::counters(Channel::X).0 > applied;
                let read_stamp = ext.reads.load(Ordering::Acquire);
                let mut wrong = 0usize;
                for y in 0..size[1] {
                    for x in 0..size[0] {
                        let i = (y * size[0] + x) as usize;
                        let b = base[i];
                        let want = if x < region[0] && y < region[1] {
                            let v = ext_vals[(y * region[0] + x) as usize];
                            let r = match mode {
                                ExternalMode::Replace => v,
                                ExternalMode::Min => b[0].min(v),
                            };
                            [r, b[1], b[2], b[3]]
                        } else {
                            b
                        };
                        wrong += usize::from(run.pixels[i] != want);
                    }
                }
                let name = format!("external .x {} after {label}", mode.name());
                report.push(format!(
                    "  {name:<30} GPU {:.3} ms | applied {applied} | pixels differing from the expected {wrong} | read stamp {read_stamp} (passes {}) | state {}",
                    run.gpu_ms,
                    passes(),
                    if run.state_diff.is_empty() {
                        "restored".to_owned()
                    } else {
                        format!("CHANGED: {}", run.state_diff.join(", "))
                    }
                ));
                if mode == ExternalMode::Min {
                    images.push(gray(
                        &run.pixels,
                        size,
                        region,
                        format!("{case}_external_min_after_{label}").replace('+', ""),
                    ));
                }
                let mut fail = |ok: bool, what: String| {
                    if !ok {
                        failures.push(format!("{case} {name}: {what}"));
                    }
                };
                fail(
                    run.status == GFSDK_SSAO_OK,
                    format!("status {}", run.status),
                );
                fail(applied, "not applied".into());
                fail(
                    wrong == 0,
                    format!("{wrong} pixels differ from the expected"),
                );
                fail(
                    run.state_diff == base_diff,
                    format!(
                        "state changed: {:?} (without the source: {base_diff:?})",
                        run.state_diff
                    ),
                );
                fail(read_stamp == passes(), format!("read stamp {read_stamp}"));
            }
            let after = run_once(
                &g, &probe, render_ao, hctx, &input, &params, &output, region, &out, &rtv,
            )?;
            let identical = after.pixels.len() == base.len()
                && after
                    .pixels
                    .iter()
                    .zip(&base)
                    .all(|(a, b)| a.map(f32::to_bits) == b.map(f32::to_bits));
            report.push(format!(
                "  nothing registered after {label}: bit-identical to the run before registering {identical}"
            ));
            if !identical {
                failures.push(format!(
                    "{case} {label}: output changed after unregistering the external source"
                ));
            }
        }
    }
    settings::edit(|f| *f = Settings::default());
    if failures.is_empty() {
        report.push("\nall checks passed".into());
    }
    Ok(OfflineReport {
        text: report.join("\n"),
        images,
        failures,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    g: &Gpu,
    probe: &Probe,
    render_ao: RenderAoFn,
    hctx: *mut c_void,
    input: &InputData,
    params: &Parameters,
    output: &Output,
    region: [u32; 2],
    out: &ID3D11Texture2D,
    rtv: &ID3D11RenderTargetView,
) -> Result<Run, String> {
    probe.bind(&g.def, region);
    let ours_before = OURS.load(Ordering::Relaxed);
    // SAFETY: capture on the deferred context recorded by this thread.
    let before = unsafe { Saved::capture(&g.def, WIDE) };
    // SAFETY: RenderAO through the context's vtable with the engine's argument layout.
    let status = unsafe { render_ao(hctx, g.def.as_raw(), input, params, output, RENDER_AO) };
    // SAFETY: as above.
    let after = unsafe { Saved::capture(&g.def, WIDE) };
    let ours = OURS.load(Ordering::Relaxed) > ours_before;
    let state_diff = before.diff(&after);
    let mut list = None;
    // SAFETY (whole block): finishing the deferred context's list and running it on the
    // immediate context of the same device.
    unsafe {
        g.def
            .FinishCommandList(false, Some(&mut list))
            .map_err(create_err("FinishCommandList"))?;
    }
    let list: ID3D11CommandList = list.ok_or("no command list")?;
    // SAFETY: as above.
    unsafe {
        g.imm.ClearRenderTargetView(rtv, &SENTINEL);
        g.imm.ExecuteCommandList(&list, false);
    }
    let pixels = g.read(out, {
        let mut d = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter.
        unsafe { out.GetDesc(&mut d) };
        [d.Width, d.Height]
    })?;
    let gpu_ms = g.time(&list)?;
    Ok(Run {
        status,
        ours,
        state_diff,
        pixels,
        gpu_ms,
    })
}
