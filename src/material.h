// Material data for SSGI and SSR: the patched forward-lighting shaders (shader_patch.h) write base colour to render
// target slot 2, the game's specular reflection to slot 3 and roughness + metallic to slot 4; this binds our three
// targets there whenever one of them draws.
//
// Opaque draws write straight through. Alpha-blended draws (decals, glass) lerp their material data in by their
// opacity, through a blend state of our own for our targets, so a wet decal's roughness lands only where the decal
// is visible. The game's own render targets are never displaced: ours are only added when slots 2..4 are free and
// no UAVs are bound (OMSetRenderTargets would unbind them). Consumed and cleared in FSR's input pack each frame.
#pragma once

#include "core.h"

namespace material
{
    // Context hooks (record time). AfterDraw puts the game's blend state back after a blended draw we captured; it
    // must be called after every draw for which NeedsAfterDraw() was true.
    void OnSetPixelShader(ID3D11DeviceContext* ctx, ID3D11PixelShader* shader);
    void OnSetRenderTargets(ID3D11DeviceContext* ctx);
    void OnFinishCommandList(ID3D11DeviceContext* ctx);
    void OnDraw(ID3D11DeviceContext* ctx);
    bool NeedsAfterDraw();
    void AfterDraw(ID3D11DeviceContext* ctx);

    // This frame's material data (render region at the top left), or nulls if none was captured recently.
    struct Views
    {
        ID3D11ShaderResourceView* albedo = nullptr;    // rgb base colour (0 = unknown)
        ID3D11ShaderResourceView* specular = nullptr;  // rgb the game's IBL reflection
        ID3D11ShaderResourceView* surface = nullptr;   // r roughness LOD (-1 = unknown), g metallic
        explicit operator bool() const { return albedo && specular && surface; }
    };
    Views ShaderResources();
    void  Clear(ID3D11DeviceContext* ctx);  // after use: pixels no patched shader covers next frame read "unknown"
    void  OnFrame();                        // frame boundary (main thread)
}
