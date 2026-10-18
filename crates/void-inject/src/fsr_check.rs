//! Non-blocking GPU check of the values flowing through the FSR path (white-flash
//! diagnostics, docs/fsr4-integration.md "Value check"): per frame a compute shader counts
//! NaN, Inf and very large values (> [`HUGE`]) and keeps the maximum of each stage's texture
//! region; the 64-byte result is copied to a staging ring and read a few frames later with
//! `D3D11_MAP_FLAG_DO_NOT_WAIT` (never stalls). Frames with bad values are logged with the
//! frame number and the stage.
//!
//! Stages (in GPU order): 0 engine colour before our reflections (SSR), 1 engine colour
//! after SSR (what the FSR pack reads), 2 FSR colour input after the RT-reflection
//! composite (shared texture, written on our D3D12 queue), 3 FSR output.

use std::collections::VecDeque;

use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;
use windows::core::s;

use crate::log;

pub(crate) const STAGES: usize = 4;
pub(crate) const STAGE_NAMES: [&str; STAGES] = [
    "engine colour before reflections",
    "colour after SSR (FSR pack input)",
    "FSR colour input after RT reflections",
    "FSR output",
];
/// Values above this count as "huge": the clamp of the FSR pack / unpack guard (`Sanitize`
/// in `fsr.rs`; FP16 max 65504, R11G11B10 max 65024), so a guarded stage never reports them.
pub(crate) const HUGE: f32 = 65_000.0;
const WORDS: usize = 4 * STAGES;
const RING: usize = 4;

const HLSL: &str = r"
Texture2D<float4> g_tex : register(t0);
RWByteAddressBuffer o_stats : register(u0);
cbuffer Params : register(b0)
{
    uint2 size;
    uint stage;
    float huge;
};
// Per stage 4 words: NaN count, Inf count, huge count, max (asuint of the largest finite
// |value|). Each thread scans 4x4 pixels, each 8x8 group reduces in group-shared memory and
// one thread per group adds to the buffer (few global atomics). NaN / Inf are tested on the
// bits (fxc may fold isnan without IEEE strictness).
groupshared uint gs_nan;
groupshared uint gs_inf;
groupshared uint gs_huge;
groupshared uint gs_max;
[numthreads(8, 8, 1)]
void scan(uint2 gid : SV_GroupID, uint2 tid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    if (gi == 0)
    {
        gs_nan = 0;
        gs_inf = 0;
        gs_huge = 0;
        gs_max = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint n_nan = 0, n_inf = 0, big = 0;
    float m = 0.0;
    uint2 o = gid * 32 + tid * 4;
    [unroll] for (uint y = 0; y < 4; y++)
    {
        [unroll] for (uint x = 0; x < 4; x++)
        {
            uint2 p = o + uint2(x, y);
            if (p.x < size.x && p.y < size.y)
            {
                uint3 b = asuint(g_tex[p].rgb) & 0x7fffffff;
                bool3 special = (b & 0x7f800000) == 0x7f800000;
                bool3 isn = special && (b & 0x007fffff) != 0;
                if (any(isn))
                    n_nan++;
                else if (any(special))
                    n_inf++;
                else
                {
                    float v = max(max(asfloat(b.x), asfloat(b.y)), asfloat(b.z));
                    big += v > huge ? 1 : 0;
                    m = max(m, v);
                }
            }
        }
    }
    if (n_nan != 0)
        InterlockedAdd(gs_nan, n_nan);
    if (n_inf != 0)
        InterlockedAdd(gs_inf, n_inf);
    if (big != 0)
        InterlockedAdd(gs_huge, big);
    InterlockedMax(gs_max, asuint(m));
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0)
    {
        uint s = stage * 16;
        if (gs_nan != 0)
            o_stats.InterlockedAdd(s, gs_nan);
        if (gs_inf != 0)
            o_stats.InterlockedAdd(s + 4, gs_inf);
        if (gs_huge != 0)
            o_stats.InterlockedAdd(s + 8, gs_huge);
        o_stats.InterlockedMax(s + 12, gs_max);
    }
}
";

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    size: [u32; 2],
    stage: u32,
    huge: f32,
}

/// One frame's result: per stage (NaN, Inf, huge, max).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct StageStats {
    pub nan: u32,
    pub inf: u32,
    pub huge: u32,
    pub max: f32,
}

impl StageStats {
    pub(crate) fn bad(&self) -> bool {
        self.nan + self.inf + self.huge > 0
    }
}

pub(crate) fn decode(words: &[u32; WORDS]) -> [StageStats; STAGES] {
    std::array::from_fn(|s| StageStats {
        nan: words[4 * s],
        inf: words[4 * s + 1],
        huge: words[4 * s + 2],
        max: f32::from_bits(words[4 * s + 3]),
    })
}

pub(crate) struct ValueCheck {
    cs: ID3D11ComputeShader,
    params: ID3D11Buffer,
    stats: ID3D11Buffer,
    uav: ID3D11UnorderedAccessView,
    staging: Vec<ID3D11Buffer>,
    /// (staging index, frame) with a copy in flight, oldest first.
    pending: VecDeque<(usize, u32)>,
    next: usize,
    /// The current frame's stats were cleared (`begin`) and not yet copied (`end`).
    open: bool,
    /// Frames checked / frames with bad values (since the last summary) and in total.
    pub checked: u64,
    pub bad_frames: u64,
    pub bad_total: u64,
    /// Per-stage frames with bad values since the last summary.
    pub bad_by_stage: [u64; STAGES],
    /// Largest finite value per stage since the last summary.
    pub max_by_stage: [f32; STAGES],
    logged: u32,
    /// SRV of FSR's shared colour input (written by the pack, then by the RT-reflection
    /// composite on the D3D12 queue); stage 2 reads it.
    pub fsr_input: Option<ID3D11ShaderResourceView>,
    /// The latest frame with bad values and its stats (offline check).
    pub last_bad: Option<(u32, [StageStats; STAGES])>,
}

impl ValueCheck {
    /// `fsr_input_uav`: the D3D11 UAV of FSR's shared colour input (its texture gets an SRV).
    pub(crate) fn new(
        d11: &ID3D11Device5,
        fsr_input_uav: Option<&ID3D11UnorderedAccessView>,
    ) -> Result<Self, String> {
        let code = crate::hlsl::compile(HLSL, s!("scan"), s!("cs_5_0"))?;
        let buffer = |desc: D3D11_BUFFER_DESC| -> Result<ID3D11Buffer, String> {
            let mut b = None;
            // SAFETY: valid descriptor / out pointer.
            unsafe { d11.CreateBuffer(&desc, None, Some(&mut b)) }
                .map_err(|e| format!("value-check buffer: {e}"))?;
            b.ok_or_else(|| "no value-check buffer".to_owned())
        };
        let bytes = (WORDS * 4) as u32;
        let params = buffer(D3D11_BUFFER_DESC {
            ByteWidth: std::mem::size_of::<Params>() as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            ..Default::default()
        })?;
        let stats = buffer(D3D11_BUFFER_DESC {
            ByteWidth: bytes,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
            MiscFlags: D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS.0 as u32,
            ..Default::default()
        })?;
        let staging = (0..RING)
            .map(|_| {
                buffer(D3D11_BUFFER_DESC {
                    ByteWidth: bytes,
                    Usage: D3D11_USAGE_STAGING,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mut cs, mut uav, mut fsr_input) = (None, None, None);
        // SAFETY: valid bytecode, raw view of the whole buffer, out pointers.
        unsafe {
            d11.CreateComputeShader(&code, None, Some(&mut cs))
                .map_err(|e| format!("value-check shader: {e}"))?;
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
            .map_err(|e| format!("value-check UAV: {e}"))?;
            if let Some(u) = fsr_input_uav {
                let res = u.GetResource().map_err(|e| format!("FSR input: {e}"))?;
                d11.CreateShaderResourceView(&res, None, Some(&mut fsr_input))
                    .map_err(|e| format!("FSR input SRV: {e}"))?;
            }
        }
        Ok(Self {
            cs: cs.ok_or("no value-check shader")?,
            params,
            stats,
            uav: uav.ok_or("no value-check UAV")?,
            staging,
            pending: VecDeque::new(),
            next: 0,
            open: false,
            checked: 0,
            bad_frames: 0,
            bad_total: 0,
            bad_by_stage: [0; STAGES],
            max_by_stage: [0.0; STAGES],
            logged: 0,
            fsr_input,
            last_bad: None,
        })
    }

    /// Starts a frame: reads finished results, clears the counters. Skipped (false) while
    /// the read-back ring is full (the GPU is far behind; never waits).
    pub(crate) fn begin(&mut self, ctx: &ID3D11DeviceContext) -> bool {
        self.poll(ctx);
        self.open = self.pending.len() < RING;
        if self.open {
            // SAFETY: our own UAV.
            unsafe { ctx.ClearUnorderedAccessViewUint(&self.uav, &[0; 4]) };
        }
        self.open
    }

    /// Scans the top-left `size` region of `srv` into `stage` (no-op outside a frame).
    pub(crate) fn scan(
        &self,
        ctx: &ID3D11DeviceContext,
        srv: &ID3D11ShaderResourceView,
        size: [u32; 2],
        stage: usize,
    ) {
        if !self.open || stage >= STAGES || size[0] == 0 || size[1] == 0 {
            return;
        }
        let p = Params {
            size,
            stage: stage as u32,
            huge: HUGE,
        };
        // SAFETY: compute work on our own resources and a live SRV; every binding is
        // cleared afterwards.
        unsafe {
            ctx.UpdateSubresource(&self.params, 0, None, (&raw const p).cast(), 0, 0);
            ctx.CSSetShader(&self.cs, None);
            ctx.CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
            ctx.CSSetShaderResources(0, Some(&[Some(srv.clone())]));
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&Some(self.uav.clone())), None);
            ctx.Dispatch(size[0].div_ceil(32), size[1].div_ceil(32), 1);
            ctx.CSSetUnorderedAccessViews(0, 1, Some(&None), None);
            ctx.CSSetShaderResources(0, Some(&[None]));
            ctx.CSSetConstantBuffers(0, Some(&[None]));
            ctx.CSSetShader(None, None);
        }
    }

    /// Ends the frame: queues the copy for a later, non-blocking read.
    pub(crate) fn end(&mut self, ctx: &ID3D11DeviceContext, frame: u32) {
        if !std::mem::take(&mut self.open) {
            return;
        }
        let i = self.next;
        self.next = (i + 1) % RING;
        // SAFETY: same-size buffers; the staging buffer is not mapped (only mapped while
        // in `poll`, which unmaps it).
        unsafe { ctx.CopyResource(&self.staging[i], &self.stats) };
        self.pending.push_back((i, frame));
    }

    /// Reads finished copies (never waits) and logs frames with bad values.
    pub(crate) fn poll(&mut self, ctx: &ID3D11DeviceContext) {
        while let Some(&(i, frame)) = self.pending.front() {
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            let mut words = [0u32; WORDS];
            // SAFETY: our staging buffer; DO_NOT_WAIT fails while the copy is in flight;
            // the mapped range holds `WORDS` words.
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
                words.copy_from_slice(std::slice::from_raw_parts(m.pData.cast::<u32>(), WORDS));
                ctx.Unmap(&self.staging[i], 0);
            }
            self.pending.pop_front();
            self.record(frame, &decode(&words));
        }
    }

    /// Offline only (`fsr::offline_check`): flushes and polls until every queued result was
    /// read or `timeout_ms` passed.
    pub(crate) fn drain(&mut self, ctx: &ID3D11DeviceContext, timeout_ms: u64) {
        let t = std::time::Instant::now();
        while !self.pending.is_empty() && t.elapsed().as_millis() < u128::from(timeout_ms) {
            // SAFETY: plain flush of the (offline) immediate context.
            unsafe { ctx.Flush() };
            self.poll(ctx);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn record(&mut self, frame: u32, stats: &[StageStats; STAGES]) {
        self.checked += 1;
        let mut bad = false;
        for (s, st) in stats.iter().enumerate() {
            if st.max.is_finite() {
                self.max_by_stage[s] = self.max_by_stage[s].max(st.max);
            }
            if st.bad() {
                self.bad_by_stage[s] += 1;
                bad = true;
            }
        }
        if !bad {
            return;
        }
        self.bad_frames += 1;
        self.last_bad = Some((frame, *stats));
        self.bad_total += 1;
        // Every bad frame for the first 50, then every 100th (the summary in the FSR
        // report counts all of them).
        if self.logged < 50 || self.bad_total.is_multiple_of(100) {
            self.logged += 1;
            log::line(&format!(
                "fsr: value check frame {frame}: {}",
                describe(stats)
            ));
        }
    }

    /// The summary for the periodic FSR report; resets the interval counters.
    pub(crate) fn summary(&mut self) -> String {
        let s = format!(
            "value check: {} frames checked, {} with NaN/Inf/>{HUGE} (by stage {:?}), max per stage {:?}",
            self.checked, self.bad_frames, self.bad_by_stage, self.max_by_stage
        );
        self.checked = 0;
        self.bad_frames = 0;
        self.bad_by_stage = [0; STAGES];
        self.max_by_stage = [0.0; STAGES];
        s
    }
}

/// Human-readable stages with bad values (first stage in GPU order marked as the origin).
pub(crate) fn describe(stats: &[StageStats; STAGES]) -> String {
    let first = stats.iter().position(StageStats::bad);
    stats
        .iter()
        .enumerate()
        .map(|(s, st)| {
            format!(
                "[{}{}: NaN {} Inf {} huge {} max {:.1}]",
                STAGE_NAMES[s],
                if Some(s) == first { " (first)" } else { "" },
                st.nan,
                st.inf,
                st.huge,
                st.max
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_and_describe() {
        let mut w = [0u32; WORDS];
        w[4 * 2] = 3; // NaN in stage 2
        w[4 * 2 + 3] = 1234.5f32.to_bits();
        w[4 * 3 + 1] = 7; // Inf in stage 3
        let s = decode(&w);
        assert!(!s[0].bad() && !s[1].bad() && s[2].bad() && s[3].bad());
        assert_eq!(s[2].max, 1234.5);
        let d = describe(&s);
        assert!(d.contains("RT reflections (first): NaN 3"), "{d}");
        assert!(d.contains("FSR output: NaN 0 Inf 7"), "{d}");
    }
}
