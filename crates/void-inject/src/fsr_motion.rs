//! Motion-vector diagnostics of the FSR path (docs/fsr4-integration.md "Motion-vector
//! coverage"):
//! - [`MotionStats`]: on static-camera frames (one in [`SAMPLE_EVERY`]) a compute shader
//!   counts, per depth band, the pixels whose *engine* motion vector exceeds
//!   [`THRESHOLD_PX`] render pixels; read back with `D3D11_MAP_FLAG_DO_NOT_WAIT` (never
//!   stalls) and logged every [`LOG_EVERY_S`] seconds.
//! - [`MotionView`]: the debug view of the vectors FSR receives (`fsr::DEBUG_VIEWS`
//!   "motion vectors"): brightness = log2 magnitude from 0.02 to 32 render pixels per frame,
//!   hue = direction, black = zero; a legend bar along the bottom edge with ticks at 0.1, 1
//!   and 10 px. FFX's own debug view scales linearly, so sub-pixel object motion (a walking
//!   NPC at 30-100 m moves 0.07-0.7 px per frame, E080) looks black there.

use std::collections::VecDeque;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
use windows::core::{Interface, s};

use crate::log;

/// Depth bands (engine units, metres **H**): [0, 5), [5, 20), [20, 80), >= 80, sky (depth 0).
pub(crate) const BANDS: [&str; 5] = ["0-5 m", "5-20 m", "20-80 m", ">80 m", "sky"];
const NB: usize = BANDS.len();
/// Words: per band the moving pixels, then per band all pixels.
const WORDS: usize = 2 * NB;
/// Engine motion above this many render pixels per frame counts as moving.
pub(crate) const THRESHOLD_PX: f32 = 0.25;
pub(crate) const SAMPLE_EVERY: u32 = 8;
const LOG_EVERY_S: f32 = 10.0;
const RING: usize = 4;
/// Debug-view magnitude range (render pixels per frame) and legend ticks.
pub(crate) const VIEW_MIN_PX: f32 = 0.02;
pub(crate) const VIEW_MAX_PX: f32 = 32.0;
pub(crate) const VIEW_TICKS_PX: [f32; 3] = [0.1, 1.0, 10.0];

const HLSL: &str = r"
Texture2D<float2> g_mv : register(t0);
Texture2D<float> g_z : register(t1);
RWByteAddressBuffer o_bands : register(u0);
cbuffer BandParams : register(b0)
{
    uint2 size;
    float nearZ;
    float thresholdPx;
};
groupshared uint gs[10];
// Each thread 4x4 pixels, group-shared counters, one global add per counter and group.
[numthreads(8, 8, 1)]
void bands(uint2 gid : SV_GroupID, uint2 tid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    if (gi < 10)
        gs[gi] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint2 o = gid * 32 + tid * 4;
    [unroll] for (uint y = 0; y < 4; y++)
    {
        [unroll] for (uint x = 0; x < 4; x++)
        {
            uint2 p = o + uint2(x, y);
            if (p.x < size.x && p.y < size.y)
            {
                float d = g_z[p];
                uint band = 4;
                if (d > 0.0)
                {
                    float z = nearZ / d;
                    band = z < 5.0 ? 0 : (z < 20.0 ? 1 : (z < 80.0 ? 2 : 3));
                }
                InterlockedAdd(gs[5 + band], 1);
                float2 m = g_mv[p] * float2(size);
                if (dot(m, m) > thresholdPx * thresholdPx)
                    InterlockedAdd(gs[band], 1);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi < 10 && gs[gi] != 0)
        o_bands.InterlockedAdd(gi * 4, gs[gi]);
}

Texture2D<float2> g_pmv : register(t0);
cbuffer ViewParams : register(b0)
{
    uint2 srcSize;
    uint2 dstSize;
    float2 scale;
    float minPx;
    float maxPx;
    float4 ticks;
};
float Brightness(float px)
{
    return saturate(log2(1.0 + px / minPx) / log2(1.0 + maxPx / minPx));
}
float3 Hue(float h)
{
    return saturate(float3(abs(h * 6.0 - 3.0) - 1.0, 2.0 - abs(h * 6.0 - 2.0), 2.0 - abs(h * 6.0 - 4.0)));
}
float4 vs_view(uint i : SV_VertexID) : SV_Position
{
    float2 uv = float2((i << 1) & 2, i & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
float4 ps_view(float4 pos : SV_Position) : SV_Target
{
    // Legend: the bottom 24 px show brightness over the log magnitude axis, white ticks.
    if (pos.y >= float(dstSize.y) - 24.0)
    {
        float t = pos.x / float(dstSize.x);
        [unroll] for (uint k = 0; k < 3; k++)
        {
            if (abs(t - Brightness(ticks[k])) * float(dstSize.x) < 1.5 && pos.y < float(dstSize.y) - 12.0)
                return float4(1.0, 1.0, 1.0, 1.0);
        }
        return float4(t.xxx, 1.0);
    }
    uint2 p = min(uint2(pos.xy * scale), srcSize - 1);
    float2 m = g_pmv.Load(int3(p, 0)) * float2(srcSize);
    float len = length(m);
    if (!(len > 0.0))
        return float4(0.0, 0.0, 0.0, 1.0);
    float h = atan2(-m.y, m.x) * 0.15915494 + 0.5;
    float3 c = lerp(1.0.xxx, Hue(h), 0.7) * Brightness(len);
    return float4(c, 1.0);
}
";

/// The debug view's brightness for a magnitude in render pixels (as in `ps_view`).
pub(crate) fn view_brightness(px: f32) -> f32 {
    ((1.0 + px / VIEW_MIN_PX).log2() / (1.0 + VIEW_MAX_PX / VIEW_MIN_PX).log2()).clamp(0.0, 1.0)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BandParams {
    size: [u32; 2],
    near: f32,
    threshold: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ViewParams {
    src: [u32; 2],
    dst: [u32; 2],
    scale: [f32; 2],
    min_px: f32,
    max_px: f32,
    ticks: [f32; 4],
}

fn buffer(d11: &ID3D11Device5, desc: D3D11_BUFFER_DESC) -> Result<ID3D11Buffer, String> {
    let mut b = None;
    // SAFETY: valid descriptor / out pointer.
    unsafe { d11.CreateBuffer(&desc, None, Some(&mut b)) }
        .map_err(|e| format!("motion buffer: {e}"))?;
    b.ok_or_else(|| "no motion buffer".to_owned())
}

fn constants(d11: &ID3D11Device5, bytes: usize) -> Result<ID3D11Buffer, String> {
    buffer(
        d11,
        D3D11_BUFFER_DESC {
            ByteWidth: bytes as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            ..Default::default()
        },
    )
}

/// One read-back result: per band (moving pixels, all pixels).
pub(crate) type Bands = [(u64, u64); NB];

pub(crate) struct MotionStats {
    cs: ID3D11ComputeShader,
    params: ID3D11Buffer,
    stats: ID3D11Buffer,
    uav: ID3D11UnorderedAccessView,
    staging: Vec<ID3D11Buffer>,
    pending: VecDeque<usize>,
    next: usize,
    calls: u32,
    /// Accumulated since the last log line.
    acc: Bands,
    frames: u32,
    since: Instant,
    /// The latest single-frame result (offline check).
    pub last: Option<Bands>,
}

impl MotionStats {
    pub(crate) fn new(d11: &ID3D11Device5) -> Result<Self, String> {
        let code = crate::hlsl::compile(HLSL, s!("bands"), s!("cs_5_0"))?;
        let bytes = (WORDS * 4) as u32;
        let params = constants(d11, std::mem::size_of::<BandParams>())?;
        let stats = buffer(
            d11,
            D3D11_BUFFER_DESC {
                ByteWidth: bytes,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
                MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
                ..Default::default()
            },
        )?;
        let staging = (0..RING)
            .map(|_| {
                buffer(
                    d11,
                    D3D11_BUFFER_DESC {
                        ByteWidth: bytes,
                        Usage: D3D11_USAGE_STAGING,
                        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                        ..Default::default()
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mut cs, mut uav) = (None, None);
        // SAFETY: valid bytecode, raw view of the whole buffer, out pointers.
        unsafe {
            d11.CreateComputeShader(&code, None, Some(&mut cs))
                .map_err(|e| format!("motion-band shader: {e}"))?;
            d11.CreateUnorderedAccessView(
                &stats,
                Some(&D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D11_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: WORDS as u32,
                            Flags: D3D11_BUFFER_UAV_FLAG_RAW.0 as u32,
                        },
                    },
                }),
                Some(&mut uav),
            )
            .map_err(|e| format!("motion-band UAV: {e}"))?;
        }
        Ok(Self {
            cs: cs.ok_or("no motion-band shader")?,
            params,
            stats,
            uav: uav.ok_or("no motion-band UAV")?,
            staging,
            pending: VecDeque::new(),
            next: 0,
            calls: 0,
            acc: [(0, 0); NB],
            frames: 0,
            since: Instant::now(),
            last: None,
        })
    }

    /// Per FSR frame: reads finished results; on one static-camera frame in
    /// [`SAMPLE_EVERY`] counts the engine vectors (`motion`, viewport UV) per depth band of
    /// `depth` (reversed-Z infinite: `z = near / d`).
    pub(crate) fn frame(
        &mut self,
        ctx: &ID3D11DeviceContext,
        motion: &ID3D11ShaderResourceView,
        depth: &ID3D11ShaderResourceView,
        render: [u32; 2],
        near: f32,
        static_camera: bool,
    ) {
        self.poll(ctx);
        if !static_camera {
            return;
        }
        self.calls = self.calls.wrapping_add(1);
        if !self.calls.is_multiple_of(SAMPLE_EVERY) || self.pending.len() >= RING {
            return;
        }
        self.sample(ctx, motion, depth, render, near);
    }

    /// One measurement (no sampling cadence; offline check).
    pub(crate) fn sample(
        &mut self,
        ctx: &ID3D11DeviceContext,
        motion: &ID3D11ShaderResourceView,
        depth: &ID3D11ShaderResourceView,
        render: [u32; 2],
        near: f32,
    ) {
        let p = BandParams {
            size: render,
            near,
            threshold: THRESHOLD_PX,
        };
        let i = self.next;
        self.next = (i + 1) % RING;
        // SAFETY: compute work on our own resources and live SRVs; every binding is cleared
        // afterwards; the staging buffer is not mapped (only inside `poll`).
        unsafe {
            ctx.ClearUnorderedAccessViewUint(&self.uav, &[0; 4]);
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.CSSetShader(&self.cs, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetShaderResources(0, Some(&[Some(motion.clone()), Some(depth.clone())]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(self.uav.clone())), None);
            ctx.Dispatch(render[0].div_ceil(32), render[1].div_ceil(32), 1);
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.CSSetShaderResources(0, Some(&[None, None]));
            ctx.CSSetConstantBuffers(0, Some(&[None]));
            ctx.CSSetShader(None, None);
            ctx.CopyResource(&self.staging[i], &self.stats);
        }
        self.pending.push_back(i);
    }

    fn poll(&mut self, ctx: &ID3D11DeviceContext) {
        while let Some(&i) = self.pending.front() {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            let mut w = [0u32; WORDS];
            // SAFETY: our staging buffer; DO_NOT_WAIT fails while the copy is in flight; the
            // mapped range holds `WORDS` words.
            unsafe {
                if ctx
                    .Map(
                        &self.staging[i],
                        0,
                        D3D11_MAP_READ,
                        D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32,
                        Some(&mut m),
                    )
                    .is_err()
                {
                    break;
                }
                w.copy_from_slice(std::slice::from_raw_parts(m.pData.cast::<u32>(), WORDS));
                ctx.Unmap(&self.staging[i], 0);
            }
            self.pending.pop_front();
            let bands: Bands = std::array::from_fn(|b| (u64::from(w[b]), u64::from(w[NB + b])));
            for (a, b) in self.acc.iter_mut().zip(&bands) {
                a.0 += b.0;
                a.1 += b.1;
            }
            self.last = Some(bands);
            self.frames += 1;
        }
        if self.frames > 0 && self.since.elapsed().as_secs_f32() >= LOG_EVERY_S {
            log::line(&format!(
                "fsr: engine motion vectors, static camera ({} frames sampled): {}",
                self.frames,
                describe(&self.acc)
            ));
            self.acc = [(0, 0); NB];
            self.frames = 0;
            self.since = Instant::now();
        } else if self.frames == 0 {
            self.since = Instant::now();
        }
    }

    /// Offline only: flushes and polls until every queued result was read.
    pub(crate) fn drain(&mut self, ctx: &ID3D11DeviceContext, timeout_ms: u64) {
        let t = Instant::now();
        while !self.pending.is_empty() && t.elapsed().as_millis() < u128::from(timeout_ms) {
            // SAFETY: plain flush of the (offline) immediate context.
            unsafe { ctx.Flush() };
            self.poll(ctx);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// `0-5 m 12.3 % of 456789 px > 0.25 px, ...`.
pub(crate) fn describe(b: &Bands) -> String {
    b.iter()
        .zip(BANDS)
        .map(|(&(moving, all), name)| {
            format!(
                "{name} {:.2} % of {all} px > {THRESHOLD_PX} px",
                moving as f64 * 100.0 / all.max(1) as f64
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The "motion vectors" debug view (drawn over the back buffer at present).
pub(crate) struct MotionView {
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    params: ID3D11Buffer,
    /// SRV of FSR's packed motion input (shared texture, render-size top-left region).
    srv: ID3D11ShaderResourceView,
}

impl MotionView {
    /// `packed_uav`: the D3D11 UAV of FSR's shared motion-vector input.
    pub(crate) fn new(
        d11: &ID3D11Device5,
        packed_uav: &ID3D11UnorderedAccessView,
    ) -> Result<Self, String> {
        let vs_code = crate::hlsl::compile(HLSL, s!("vs_view"), s!("vs_5_0"))?;
        let ps_code = crate::hlsl::compile(HLSL, s!("ps_view"), s!("ps_5_0"))?;
        let (mut vs, mut ps, mut srv) = (None, None, None);
        // SAFETY: valid bytecode; SRV of our own shared texture; out pointers.
        unsafe {
            d11.CreateVertexShader(&vs_code, None, Some(&mut vs))
                .map_err(|e| format!("motion view VS: {e}"))?;
            d11.CreatePixelShader(&ps_code, None, Some(&mut ps))
                .map_err(|e| format!("motion view PS: {e}"))?;
            let res = packed_uav
                .GetResource()
                .map_err(|e| format!("motion texture: {e}"))?;
            d11.CreateShaderResourceView(&res, None, Some(&mut srv))
                .map_err(|e| format!("motion SRV: {e}"))?;
        }
        Ok(Self {
            vs: vs.ok_or("no motion view VS")?,
            ps: ps.ok_or("no motion view PS")?,
            params: constants(d11, std::mem::size_of::<ViewParams>())?,
            srv: srv.ok_or("no motion SRV")?,
        })
    }

    /// Draws the view of the last packed vectors (`render` region) over `rtv` (`size`),
    /// under `state` (swapped in and back: the caller's bindings are untouched).
    pub(crate) fn draw(
        &self,
        ctx: &ID3D11DeviceContext,
        state: &ID3DDeviceContextState,
        raster: &ID3D11RasterizerState,
        rtv: &ID3D11RenderTargetView,
        render: [u32; 2],
        size: [u32; 2],
    ) -> Result<(), String> {
        if render[0] < 2 || render[1] < 2 || size[0] == 0 || size[1] == 0 {
            return Ok(());
        }
        let ctx1: ID3D11DeviceContext1 = ctx.cast().map_err(|e| format!("context1: {e}"))?;
        let p = ViewParams {
            src: render,
            dst: size,
            scale: [
                render[0] as f32 / size[0] as f32,
                render[1] as f32 / size[1] as f32,
            ],
            min_px: VIEW_MIN_PX,
            max_px: VIEW_MAX_PX,
            ticks: [VIEW_TICKS_PX[0], VIEW_TICKS_PX[1], VIEW_TICKS_PX[2], 0.0],
        };
        // SAFETY (whole block): our own device context state is swapped in and the
        // previous one restored (every binding of the caller's state comes back); our
        // bindings are cleared before the swap back.
        unsafe {
            let mut prev = None;
            ctx1.SwapDeviceContextState(state, Some(&mut prev));
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.IASetInputLayout(None);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(&self.ps, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.PSSetShaderResources(0, Some(&[Some(self.srv.clone())]));
            ctx.RSSetState(raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size[0] as f32,
                Height: size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(None, None, u32::MAX);
            ctx.OMSetDepthStencilState(None, 0);
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.Draw(3, 0);
            ctx.OMSetRenderTargets(None, None);
            ctx.PSSetShaderResources(0, Some(&[None]));
            ctx.PSSetConstantBuffers(0, Some(&[None]));
            if let Some(p) = prev {
                ctx1.SwapDeviceContextState(&p, None);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_scale_and_description() {
        assert_eq!(view_brightness(0.0), 0.0);
        assert!((view_brightness(VIEW_MAX_PX) - 1.0).abs() < 1e-6);
        // Sub-pixel NPC motion is clearly visible: 0.1 px -> ~0.24, 1 px -> ~0.53.
        assert!(view_brightness(0.1) > 0.2 && view_brightness(1.0) > 0.5);
        let b: Bands = [(1, 4), (0, 10), (5, 10), (0, 0), (0, 3)];
        let d = describe(&b);
        assert!(d.starts_with("0-5 m 25.00 % of 4 px > 0.25 px"), "{d}");
        assert!(d.contains("20-80 m 50.00 % of 10 px"), "{d}");
    }
}
