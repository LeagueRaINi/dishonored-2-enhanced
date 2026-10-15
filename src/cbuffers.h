// Engine constant buffers: CPU shadows of PerViewCB / PerInstanceCB, and display-resolution PerViewCB twins.
//
// Shadows: the engine fills its dynamic constant buffers with Map/Unmap (or UpdateSubresource). Their contents
// are copied at Unmap so frame constants (jitter, matrices, exposure) can be read without a GPU readback.
//
// Twins: after FSR the post passes read a display-size image, but their shaders scale UVs by PerViewCB's
// resolutionscale. While a thread records post passes at display resolution, every draw/dispatch gets a twin of
// its PerViewCB with resolutionscale = 1. Dynamic buffers must be mapped in each command list that uses them, so a
// twin is rewritten the first time a list needs it, and again whenever the real buffer changed.
//
// Both are attached to the engine buffer as private data, so they live exactly as long as the buffer does.
#pragma once

#include "core.h"

#include <array>
#include <initializer_list>

namespace cb
{
    constexpr UINT kPerViewSize     = 800;
    constexpr UINT kPerInstanceSize = 160;

    // Byte offsets inside PerViewCB (b1).
    namespace perview
    {
        constexpr size_t kSubpixelOffset         = 160;
        constexpr size_t kProjection             = 176;  // float4x4, reversed infinite Z: [5] = 1/tan(fovY/2), [14] = near
        constexpr size_t kPreviousViewProjection = 240;  // float4x4, row-vector convention: clip = (world, 1) * M
        constexpr size_t kJitterVectors          = 464;  // (cur.xy, prev.xy) as fractions of the render size
        constexpr size_t kInverseViewProjection  = 480;  // float4x4: world = (ndc.xy, depth, 1) * M, then / w
        constexpr size_t kCompressedHdr          = 716;  // uint
        constexpr size_t kJitterRelative         = 752;
        constexpr size_t kResolutionScale        = 776;  // float2
        constexpr size_t kFrameNumber            = 792;  // float
    }

    // Byte offsets inside the TAA pass's PerInstanceCB (b2).
    namespace perinstance
    {
        constexpr size_t kTaaTexSize     = 16;
        constexpr size_t kDitherViewport = 32;   // (dither.xy, render width, render height)
        constexpr size_t kExposureIndex  = 128;  // uint2, .y indexes the auto-exposure buffer
        constexpr size_t kPrevResScale   = 136;
        constexpr size_t kEnvWhite       = 144;
        constexpr size_t kViewWhite      = 148;
        constexpr size_t kTaaAmount      = 152;
        constexpr size_t kEvBias         = 156;
    }

    struct Snapshot
    {
        std::array<std::byte, kPerViewSize> bytes{};
        UINT     size = 0;
        uint64_t revision = 0;  // increments with every engine update

        template <typename T>
        T Get(size_t offset) const
        {
            T value;
            memcpy(&value, bytes.data() + offset, sizeof(T));
            return value;
        }
    };

    // Latest CPU contents of an engine constant buffer, false if it isn't tracked (yet).
    bool Read(ID3D11Buffer* buffer, Snapshot& out);

    // Context hooks.
    void OnMap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, D3D11_MAP type, const D3D11_MAPPED_SUBRESOURCE* mapped);
    void OnUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* resource);
    void OnUpdateSubresource(ID3D11Resource* resource, const D3D11_BOX* box, const void* data);
    void OnFinishCommandList(ID3D11DeviceContext* ctx);

    // Display-resolution post passes on the calling thread (game.cpp).
    enum class Stage { VS, PS, CS };
    void BeginDisplayResolution(ID3D11DeviceContext* ctx);
    void EndDisplayResolution();  // puts the real PerViewCBs back where twins are bound
    bool IsDisplayResolution(ID3D11DeviceContext* ctx);
    void BindTwins(ID3D11DeviceContext* ctx, std::initializer_list<Stage> stages);  // before each draw/dispatch
}
