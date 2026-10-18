//! S1 external-source entry point (docs/lighting-options.md S1, docs/raytracing.md section
//! 13): another module (photo mode's ray-traced terms) registers a per-pixel term for one
//! channel of `_SSAO_ScreenShadows` (`.x` = AO for the indirect term, `.y` = sun visibility),
//! and the existing detours draw it into the engine's target right after the channel was
//! written: `.x` in the `RenderAO` detour after HBAO+ or our AO (`ao.rs`), `.y` in the
//! SCREENSHADOWS detour after the CSM resolve, its blur and our contact shadows
//! (`contact.rs`). No second hook. With nothing registered the detours behave exactly as
//! before (one atomic load per pass).
//!
//! Source texture: region-local (texel `(i, j)` = target pixel `(x + i, y + j)` of the pass
//! region), any format whose SRV returns the term in `.r` (R32F / R16F), on the engine's
//! D3D11 device; applied only when its size equals the pass region (other views and a changed
//! render scale are skipped and counted). Mode `Replace` writes the term, `Min` writes
//! `min(channel, term)` (blend op MIN); the colour write mask limits the draw to the channel.
//!
//! Hand-off: the owner must not write a registered texture. Each pass that reads it stores
//! the owner's frame clock ([`ExternalSource::clock`], e.g. the photo composite's FOG pass
//! counter) into [`ExternalSource::reads`]; the owner writes a texture again only after it is
//! unregistered and its clock advanced [`REUSE_PASSES`] beyond the last read (the frames in
//! flight, the same margin as the photo composite's slots). A clock that advances once per
//! frame regardless of the consuming pass keeps the protocol live when that pass does not
//! run (HBAO+ off, no CSM).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use arc_swap::ArcSwapOption;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::{Interface, s};

use super::{Saved, Slots, create_err, dynamic_cb, upload};
use crate::{hlsl, log};

/// Passes of the consuming callback after the last read before the owner may write the
/// texture again (D3D11 frames in flight; DXGI's default maximum frame latency is 3).
pub const REUSE_PASSES: u64 = 4;

/// How an external term combines with the value already in its channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalMode {
    /// The channel becomes the term.
    Replace,
    /// `min(channel, term)`.
    Min,
}

impl ExternalMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Min => "min",
        }
    }

    pub fn parse(v: &str) -> Option<Self> {
        match v {
            "replace" => Some(Self::Replace),
            "min" => Some(Self::Min),
            _ => None,
        }
    }
}

/// The `_SSAO_ScreenShadows` channel a source is drawn into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// `.x`: ambient occlusion of the indirect term.
    X,
    /// `.y`: sun visibility (direct sun only; vegetation ignores it).
    Y,
}

/// A registered term (see the module doc).
#[derive(Clone)]
pub struct ExternalSource {
    pub srv: ID3D11ShaderResourceView,
    /// Texture size; must equal the pass region (w, h).
    pub size: [u32; 2],
    pub mode: ExternalMode,
    /// The owner's clock at the last pass that recorded a read.
    pub reads: Arc<AtomicU64>,
    /// The owner's frame clock, read by the consuming pass when it records a read.
    pub clock: fn() -> u64,
}

// SAFETY: D3D11 views are free-threaded; the struct is read-only after registration.
unsafe impl Send for ExternalSource {}
// SAFETY: as above.
unsafe impl Sync for ExternalSource {}

static SOURCE_X: ArcSwapOption<ExternalSource> = ArcSwapOption::const_empty();
static SOURCE_Y: ArcSwapOption<ExternalSource> = ArcSwapOption::const_empty();
/// Draws recorded / passes skipped (size mismatch, shaders not ready) per channel.
static APPLIED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static SKIPPED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static FAILURES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static ERROR: Mutex<Option<String>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn slot(c: Channel) -> &'static ArcSwapOption<ExternalSource> {
    match c {
        Channel::X => &SOURCE_X,
        Channel::Y => &SOURCE_Y,
    }
}

/// Registers (Some) or removes (None) the term of `channel`. Registering starts the shader
/// compilation on a background thread the first time; passes before it finishes skip the
/// term (counted).
pub fn set(channel: Channel, src: Option<ExternalSource>) {
    if src.is_some() && !COMPILE_STARTED.swap(true, Ordering::AcqRel) {
        let _ = std::thread::Builder::new()
            .name("void-ext-shaders".into())
            .spawn(|| {
                if let Err(e) = bytecode() {
                    log::line(&format!("external sources: shader compilation failed: {e}"));
                }
            });
    }
    slot(channel).store(src.map(Arc::new));
}

/// The registered term of `channel`.
pub fn get(channel: Channel) -> Option<Arc<ExternalSource>> {
    slot(channel).load_full()
}

/// Whether a term is registered for `channel` (one atomic load).
pub fn registered(channel: Channel) -> bool {
    slot(channel).load().is_some()
}

/// Counters for the overlay: (applied, skipped, failures) of `channel`.
pub fn counters(channel: Channel) -> (u64, u64, u64) {
    let i = channel as usize;
    (
        APPLIED[i].load(Ordering::Relaxed),
        SKIPPED[i].load(Ordering::Relaxed),
        FAILURES[i].load(Ordering::Relaxed),
    )
}

pub fn last_error() -> Option<String> {
    lock(&ERROR).clone()
}

const HLSL: &str = r"
// void-inject external S1 sources (ao/external.rs).
cbuffer void_ext : register(b0) { int4 g_ext; }; // region origin in the target
Texture2D<float> void_ext_src : register(t0);

float4 vs_ext(uint id : SV_VertexID) : SV_Position
{
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

// The write mask selects the channel; saturate() also maps NaN to 0.
float4 ps_ext(float4 pos : SV_Position) : SV_Target
{
    float v = saturate(void_ext_src.Load(int3(int2(pos.xy) - g_ext.xy, 0)));
    return float4(v, v, v, 1.0);
}
";

struct Bytecode {
    vs: Vec<u8>,
    ps: Vec<u8>,
}

static COMPILE_STARTED: AtomicBool = AtomicBool::new(false);
static BYTECODE_READY: AtomicBool = AtomicBool::new(false);
static BYTECODE: LazyLock<Result<Bytecode, String>> = LazyLock::new(|| {
    Ok(Bytecode {
        vs: hlsl::compile(HLSL, s!("vs_ext"), s!("vs_5_0"))?,
        ps: hlsl::compile(HLSL, s!("ps_ext"), s!("ps_5_0"))?,
    })
});

/// The compiled shaders (blocks while compiling; the offline checks call it first).
fn bytecode() -> Result<&'static Bytecode, String> {
    let r = BYTECODE.as_ref().map_err(Clone::clone);
    BYTECODE_READY.store(true, Ordering::Release);
    r
}

/// Compiles the shaders now (offline checks; render threads never block on it).
pub fn compile_now() -> Result<(), String> {
    COMPILE_STARTED.store(true, Ordering::Release);
    bytecode().map(|_| ())
}

/// Slots the draw binds (t0, b0).
const TOUCHED: Slots = Slots {
    srv: 1,
    cb: 1,
    sampler: 0,
    uav: 0,
};

struct Pipe {
    device: ID3D11Device,
    vs: ID3D11VertexShader,
    ps: ID3D11PixelShader,
    params: ID3D11Buffer,
    raster: ID3D11RasterizerState,
    no_depth: ID3D11DepthStencilState,
    /// [channel][mode]: write mask of the channel, blend off (replace) or op MIN.
    blend: [[ID3D11BlendState; 2]; 2],
}

// SAFETY: D3D11 device objects are free-threaded; the pipe is used under its mutex only.
unsafe impl Send for Pipe {}

static PIPE: Mutex<Option<Pipe>> = Mutex::new(None);

impl Pipe {
    fn new(dev: &ID3D11Device, code: &Bytecode) -> Result<Self, String> {
        // SAFETY (whole block): object creation with valid descriptors, bytecode compiled
        // for the declared stage, and out pointers.
        unsafe {
            let (mut vs, mut ps, mut raster, mut no_depth) = (None, None, None, None);
            dev.CreateVertexShader(&code.vs, None, Some(&mut vs))
                .map_err(create_err("vertex shader"))?;
            dev.CreatePixelShader(&code.ps, None, Some(&mut ps))
                .map_err(create_err("pixel shader"))?;
            dev.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut raster),
            )
            .map_err(create_err("rasterizer state"))?;
            dev.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: false.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D11_COMPARISON_ALWAYS,
                    StencilEnable: false.into(),
                    ..Default::default()
                },
                Some(&mut no_depth),
            )
            .map_err(create_err("depth-stencil state"))?;
            let blend = |mask: D3D11_COLOR_WRITE_ENABLE, min: bool| {
                let mut d = D3D11_BLEND_DESC::default();
                d.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: min.into(),
                    SrcBlend: D3D11_BLEND_ONE,
                    DestBlend: if min {
                        D3D11_BLEND_ONE
                    } else {
                        D3D11_BLEND_ZERO
                    },
                    BlendOp: if min {
                        D3D11_BLEND_OP_MIN
                    } else {
                        D3D11_BLEND_OP_ADD
                    },
                    SrcBlendAlpha: D3D11_BLEND_ONE,
                    DestBlendAlpha: D3D11_BLEND_ZERO,
                    BlendOpAlpha: D3D11_BLEND_OP_ADD,
                    RenderTargetWriteMask: mask.0 as u8,
                };
                let mut b = None;
                dev.CreateBlendState(&d, Some(&mut b))
                    .map_err(create_err("blend state"))?;
                b.ok_or_else(|| "no blend state".to_owned())
            };
            let r = D3D11_COLOR_WRITE_ENABLE_RED;
            let g = D3D11_COLOR_WRITE_ENABLE_GREEN;
            Ok(Self {
                device: dev.clone(),
                vs: vs.ok_or("no vertex shader")?,
                ps: ps.ok_or("no pixel shader")?,
                params: dynamic_cb(dev, 16)?,
                raster: raster.ok_or("no rasterizer state")?,
                no_depth: no_depth.ok_or("no depth-stencil state")?,
                blend: [
                    [blend(r, false)?, blend(r, true)?],
                    [blend(g, false)?, blend(g, true)?],
                ],
            })
        }
    }
}

/// Draws the registered term of `channel` into `rtv` inside `region` (x, y, w, h) on `ctx`
/// (the pass's deferred context), restoring every state it touches; stores the source's
/// clock into its `reads`. Ok(false): nothing registered, or skipped (size mismatch, shaders
/// compiling). Called from the detours only.
pub(crate) fn apply(
    channel: Channel,
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    region: [u32; 4],
) -> Result<bool, String> {
    let Some(src) = get(channel) else {
        return Ok(false);
    };
    let i = channel as usize;
    let r = apply_inner(channel, &src, ctx, rtv, region);
    match &r {
        Ok(true) => {
            APPLIED[i].fetch_add(1, Ordering::Relaxed);
        }
        Ok(false) => {
            SKIPPED[i].fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            let n = FAILURES[i].fetch_add(1, Ordering::Relaxed);
            if n < 5 || n.is_power_of_two() {
                log::line(&format!(
                    "external source {channel:?}: {e} (failure {})",
                    n + 1
                ));
            }
            *lock(&ERROR) = Some(e.clone());
        }
    }
    r
}

fn apply_inner(
    channel: Channel,
    src: &ExternalSource,
    ctx: &ID3D11DeviceContext,
    rtv: &ID3D11RenderTargetView,
    region: [u32; 4],
) -> Result<bool, String> {
    let [x, y, w, h] = region;
    if src.size != [w, h] || !BYTECODE_READY.load(Ordering::Acquire) {
        return Ok(false);
    }
    let code = bytecode()?;
    // SAFETY: plain getters.
    let (dev, src_dev) = unsafe {
        (
            ctx.GetDevice().map_err(create_err("device"))?,
            src.srv.GetDevice().map_err(create_err("source device"))?,
        )
    };
    if dev.as_raw() != src_dev.as_raw() {
        return Err("the source texture belongs to another D3D11 device".into());
    }
    let mut g = lock(&PIPE);
    if g.as_ref().is_none_or(|p| p.device.as_raw() != dev.as_raw()) {
        *g = None;
        *g = Some(Pipe::new(&dev, code)?);
    }
    let p = g.as_ref().ok_or("no pipe")?;
    // SAFETY: capture/restore on the caller's context, which this thread records.
    let saved = unsafe { Saved::capture(ctx, TOUCHED) };
    let r = upload(ctx, &p.params, &[x as i32, y as i32, 0, 0]).map(|()| {
        let mode = match src.mode {
            ExternalMode::Replace => 0,
            ExternalMode::Min => 1,
        };
        // SAFETY (whole block): bindings of our own objects, the registered source and the
        // engine's target; every slot used is in `TOUCHED`.
        unsafe {
            ctx.IASetInputLayout(None);
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&p.vs, None);
            ctx.HSSetShader(None, None);
            ctx.DSSetShader(None, None);
            ctx.GSSetShader(None, None);
            ctx.PSSetShader(&p.ps, None);
            ctx.PSSetConstantBuffers(0, Some(&[Some(p.params.clone())]));
            ctx.PSSetShaderResources(0, Some(&[Some(src.srv.clone())]));
            ctx.RSSetState(&p.raster);
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: x as f32,
                TopLeftY: y as f32,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
            ctx.OMSetBlendState(
                &p.blend[channel as usize][mode],
                None,
                D3D11_DEFAULT_SAMPLE_MASK,
            );
            ctx.OMSetDepthStencilState(&p.no_depth, 0);
            ctx.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);
            ctx.Draw(3, 0);
        }
    });
    // SAFETY: as above.
    unsafe { saved.restore(ctx) };
    r?;
    src.reads.fetch_max((src.clock)(), Ordering::AcqRel);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_round_trip() {
        for m in [ExternalMode::Replace, ExternalMode::Min] {
            assert_eq!(ExternalMode::parse(m.name()), Some(m));
        }
        assert_eq!(ExternalMode::parse("max"), None);
    }
}
