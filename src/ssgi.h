// Screen-space lighting for the FSR path: global illumination (indirect bounce light + visibility) and
// reflections (SSR, using the material data from material.h).
//
// Runs at half render resolution at the engine's TAA point, before FSR. Shared: prepare (normals from depth,
// radiance) and a radiance mip chain. GI: visibility-bitmask trace (shaders/ssgi_trace.hlsl), temporal accumulation,
// two edge-aware denoise passes. SSR: GGX-sampled ray march (shaders/ssr_trace.hlsl), variance-clipped temporal
// accumulation and a roughness x ray-length spatial filter. Both are combined with the scene colour in FSR's input
// pack (shaders/pack.hlsl), so FSR's own accumulation smooths what noise is left.
#pragma once

#include "core.h"

namespace ssgi
{
    struct Inputs
    {
        ID3D11ShaderResourceView* color = nullptr;     // engine HDR colour (render region), stored units
        ID3D11ShaderResourceView* depth = nullptr;     // engine depth, reversed infinite Z
        ID3D11ShaderResourceView* motion = nullptr;    // engine motion vectors
        ID3D11ShaderResourceView* exposure = nullptr;  // engine auto-exposure buffer (storage scale of the colour)
        ID3D11ShaderResourceView* surface = nullptr;   // material.h: roughness LOD, metallic (null: no SSR)
        uint32_t width = 0, height = 0;                // render region
        float    cameraNear = 0.01f;
        float    projX = 1.0f, projY = 1.0f;           // projection[0], projection[5]
        uint32_t frame = 0;                            // engine frame number (a gap resets the histories)
        uint32_t exposureIndex = 0, compressedHdr = 0;
        float    viewWhite = 1.0f;
        Float4x4 inverseViewProjection = {};           // engine matrices (PerViewCB), for history validation
        Float4x4 previousViewProjection = {};
    };

    struct Result
    {
        ID3D11ShaderResourceView* light = nullptr;       // half res: indirect light, visibility (null: GI off)
        ID3D11ShaderResourceView* reflection = nullptr;  // half res: reflected radiance, confidence (null: SSR off)
        ID3D11ShaderResourceView* geometry = nullptr;    // half res: normal, view z (for the upsample)
        ID3D11ShaderResourceView* debug = nullptr;       // half res: intermediate buffer of the selected debug view
        uint32_t halfWidth = 0, halfHeight = 0;
        explicit operator bool() const { return geometry != nullptr; }
    };

    // At the TAA dispatch (deferred context, record time). Empty if GI and SSR are off or resources failed.
    Result Record(ID3D11DeviceContext* ctx, const Inputs& in);
}
