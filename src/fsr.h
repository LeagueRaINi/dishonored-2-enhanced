// FSR upscaler on a D3D12 device next to the game's D3D11 device (native D3D11 or DXVK).
#pragma once

#include "core.h"

#include <string>
#include <vector>

namespace fsr
{
    // Per-frame values decoded from the engine's PerViewCB / PerInstanceCB at the TAA dispatch.
    struct FrameConstants
    {
        uint32_t renderWidth = 0, renderHeight = 0;  // PerInstanceCB taaditherandviewportsize.zw
        float    jitterX = 0, jitterY = 0;           // PerViewCB jittervectors.xy, fractions of the render size
        float    cameraNear = 0.01f;
        float    fovY = 1.0f;                        // radians
        float    projX = 1.0f, projY = 1.0f;         // projection[0], projection[5] (view -> clip scale)
        Float4x4 inverseViewProjection = {};         // PerViewCB, for screen-space lighting history validation
        Float4x4 previousViewProjection = {};
        float    frameNumber = 0;
        uint32_t exposureIndex = 0;                  // index into the auto-exposure buffer
        uint32_t compressedHdr = 0;
        float    viewWhite = 1, evBias = 0;
    };

    enum class TaaMode
    {
        Prepare,   // upscale mode outside a display-resolution frame: only create devices/textures, engine TAA runs
        NativeAA,  // FSR anti-aliases at render resolution
        Upscale,   // FSR writes a display-size image; falls back to a plain upsample if FSR can't run
    };

    // At the engine's TAA dispatch (record time, deferred context). Returns true if the dispatch was replaced.
    bool RecordTaa(ID3D11DeviceContext* deferred, const FrameConstants& fc, TaaMode mode);

    bool IsUpscaleReady();  // upscale mode can run (devices, textures and the motion vector path are usable)

    // Command list plumbing.
    void OnFinishCommandList(ID3D11DeviceContext* deferred, ID3D11CommandList* list);
    void OnExecuteCommandList(ID3D11DeviceContext* immediate, ID3D11CommandList* list, BOOL restoreState);
    void OnFrame();  // frame boundary: frees retired resources, drops jobs whose command list was never executed

    void SetDepthSRV(ID3D11ShaderResourceView* srv);  // engine depth, from the depth downsample pass
    void RequestInputDump(int frames);                // debug: write the next frames' upscaler inputs to disk

    // Overlay / perf log.
    struct Status
    {
        bool        active = false;     // a context exists and FSR is replacing engine passes
        bool        failed = false;     // initialization or dispatch failed; engine passes are used
        std::string provider;           // e.g. "4.1.1"
        uint32_t    renderWidth = 0, renderHeight = 0, outputWidth = 0, outputHeight = 0;
        double      gpuMs = 0;          // smoothed GPU time of the upscale dispatch
        double      gpuMsPeak = 0;      // highest GPU time of the last ~256 dispatches
        bool        upscaling = false;  // last dispatch upscaled (output larger than input)
        int         resetsTotal = 0;    // history resets since launch
        std::vector<std::string> versions;  // providers offered by the driver, newest first
    };
    Status GetStatus();

    struct Stats
    {
        int    frames = 0;    // upscaled frames since the last call
        int    resets = 0;    // history resets in that window
        double gpuMsAvg = 0;
        double gpuMsMax = 0;
    };
    Stats TakeStats();
}
