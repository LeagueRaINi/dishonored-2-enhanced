//! Draws egui output into a back buffer with `egui-directx11`, leaving the caller's D3D11
//! pipeline state exactly as it was.
//!
//! - State: the draw runs under `ID3D11DeviceContext1::SwapDeviceContextState` with our
//!   own context-state object (as `fsr.rs` depth resample); the game's state object comes
//!   back untouched. Before swapping back, our state is cleared, so the state object holds
//!   no reference to the back buffer between frames.
//! - Target: egui blends in gamma space and needs a non-sRGB view. sRGB and typeless
//!   back-buffer formats get a UNORM view (R8G8B8A8_UNORM_SRGB -> R8G8B8A8_UNORM,
//!   B8G8R8A8 likewise; the game's swap chain is R8G8B8A8_UNORM_SRGB and its HUD draws
//!   through format 28 views, d2enhanced NOTES "HDR"). If that view cannot be created the
//!   default view is used (logged once; colours then look washed out).
//! - The view is created and released within `paint`; no back-buffer reference survives
//!   the call (the game's `ResizeBuffers` must keep working).

use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::log;

pub struct Painter {
    device: ID3D11Device,
    renderer: egui_directx11::Renderer,
    state: ID3DDeviceContextState,
    fallback_logged: bool,
}

/// The non-sRGB view format for a back-buffer format (None: the default view is fine).
pub fn unorm_view_format(f: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    match f {
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => {
            Some(DXGI_FORMAT_R8G8B8A8_UNORM)
        }
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => {
            Some(DXGI_FORMAT_B8G8R8A8_UNORM)
        }
        DXGI_FORMAT_R10G10B10A2_TYPELESS => Some(DXGI_FORMAT_R10G10B10A2_UNORM),
        _ => None,
    }
}

impl Painter {
    pub fn new(device: &ID3D11Device) -> Result<Self, String> {
        let renderer =
            egui_directx11::Renderer::new(device).map_err(|e| format!("egui renderer: {e}"))?;
        let device1: ID3D11Device1 = device.cast().map_err(|e| format!("ID3D11Device1: {e}"))?;
        let mut state = None;
        // SAFETY: valid arguments and out pointer.
        unsafe {
            device1.CreateDeviceContextState(
                0,
                &[device.GetFeatureLevel()],
                D3D11_SDK_VERSION,
                &ID3D11Device::IID,
                None,
                Some(&mut state),
            )
        }
        .map_err(|e| format!("context state: {e}"))?;
        Ok(Self {
            device: device.clone(),
            renderer,
            state: state.ok_or("no context state")?,
            fallback_logged: false,
        })
    }

    pub fn device(&self) -> &ID3D11Device {
        &self.device
    }

    fn target_view(&mut self, target: &ID3D11Texture2D) -> Result<ID3D11RenderTargetView, String> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter.
        unsafe { target.GetDesc(&mut desc) };
        let mut rtv = None;
        if let Some(format) = unorm_view_format(desc.Format) {
            let vd = D3D11_RENDER_TARGET_VIEW_DESC {
                Format: format,
                ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 },
                },
            };
            // SAFETY: valid descriptor and out pointer.
            let r = unsafe {
                self.device
                    .CreateRenderTargetView(target, Some(&vd), Some(&mut rtv))
            };
            if let Err(e) = r
                && !std::mem::replace(&mut self.fallback_logged, true)
            {
                log::line(&format!(
                    "overlay: no {format:?} view of the {:?} back buffer ({e}); default view used",
                    desc.Format
                ));
            }
        }
        if rtv.is_none() {
            // SAFETY: default view of a render-target texture.
            unsafe {
                self.device
                    .CreateRenderTargetView(target, None, Some(&mut rtv))
            }
            .map_err(|e| format!("back-buffer view: {e}"))?;
        }
        rtv.ok_or_else(|| "no back-buffer view".to_owned())
    }

    /// Draws `output` onto `target` on the immediate context `ctx`.
    pub fn paint(
        &mut self,
        ctx: &ID3D11DeviceContext,
        target: &ID3D11Texture2D,
        egui_ctx: &egui::Context,
        output: egui_directx11::RendererOutput,
    ) -> Result<(), String> {
        let rtv = self.target_view(target)?;
        let ctx1: ID3D11DeviceContext1 = ctx.cast().map_err(|e| format!("context1: {e}"))?;
        let mut prev = None;
        // SAFETY: immediate context; our state object was created on the same device. The
        // game's state is restored below on every path.
        unsafe { ctx1.SwapDeviceContextState(&self.state, Some(&mut prev)) };
        let r = self
            .renderer
            .render(ctx, &rtv, egui_ctx, output)
            .map_err(|e| format!("egui render: {e}"));
        // SAFETY: clears only our state object (drops its view/buffer references), then
        // swaps the game's state back.
        unsafe {
            ctx.ClearState();
            if let Some(p) = prev {
                ctx1.SwapDeviceContextState(&p, None);
            }
        }
        r
    }
}
