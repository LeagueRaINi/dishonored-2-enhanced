// HDR10 output, only while Windows HDR is on for the game's display (decided when the swap chain is created).
//
// The game renders exactly as before into a stand-in 8-bit back buffer; the real swap chain is HDR10
// (R10G10B10A2, PQ, Rec.2020). Three engine passes are changed while HDR is active:
//   - tonemap (PS 6b486ef38e1bbe42): replaced by shaders/hdr_tonemap_ps.hlsl, which also writes an HDR copy of the
//     graded scene to a 16-bit float target;
//   - scene copy into the back buffer (PS 5b793501b92112cc, the RENDER_UPSCALE filter): runs on the HDR scene
//     instead, and the stand-in back buffer is cleared to transparent, so the UI draws onto an empty layer;
//   - UI draws into the back buffer (UNORM views): the alpha blend accumulates coverage.
// At Present, shaders/hdr_composite_ps.hlsl combines scene and UI into the real back buffer.
#pragma once

#include "core.h"

#include <dxgi1_6.h>

struct ShaderInfo;

namespace hdr
{
    struct Status
    {
        bool        windowsHdr = false;  // advanced colour on for the game's display (at swap chain creation)
        bool        active = false;      // HDR10 swap chain in use
        float       displayPeakNits = 0; // reported by the display, 0 = unknown
        std::string note;                // why HDR is off, or the swap chain format
    };
    Status GetStatus();
    bool   IsActive();

    // overlay.cpp, with the throwaway swap chain it creates to find DXGI's implementation.
    void InstallDxgiHooks(IDXGIFactory* factory, IDXGISwapChain* swapChain);

    // Swap chain: Present (after the overlay drew into the stand-in) and ResizeBuffers.
    void        Composite(IDXGISwapChain* swapChain);
    DXGI_FORMAT ResizeFormat(IDXGISwapChain* swapChain, DXGI_FORMAT requested);
    void        AfterResize(IDXGISwapChain* swapChain, HRESULT result);

    // Context hooks (record time). OnDraw returns true if the draw must be skipped; AfterDraw undoes OnDraw's
    // state changes and must be called after every draw for which NeedsAfterDraw() was true. fullscreenCandidate:
    // a draw of at most 6 vertices/indices (the tonemap and scene copy are indexed quads).
    void OnSetRenderTargets(ID3D11DeviceContext* ctx, UINT count, ID3D11RenderTargetView* const* rtvs);
    bool OnDraw(ID3D11DeviceContext* ctx, bool fullscreenCandidate);
    bool NeedsAfterDraw();
    void OnFinishCommandList(ID3D11DeviceContext* ctx);  // forgets the context's UI target
    void AfterDraw(ID3D11DeviceContext* ctx);
}
