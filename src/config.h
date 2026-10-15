// Settings (kIniFileName next to the exe).
//
// The overlay and hotkeys edit a draft; the frame tick publishes it as an immutable snapshot before the engine
// records the frame. Render workers and the FSR dispatch only ever read snapshots, and every recorded job keeps
// the snapshot it was recorded with, so one frame never mixes old and new values.
#pragma once

#include "core.h"

#include <memory>

struct Config
{
    // [General]
    bool  skipIntro       = true;    // patch out the boot logo/legal videos
    int   fpsLimit        = -1;      // -1 = game setting, 0 = unlocked, N = cap at N fps
    int   fpsCounter      = 1;       // 0 off, 1 minimal, 2 frame time, 3 FSR, 4 full, 5 benchmark
    int   maxQueuedFrames = 0;       // DXGI maximum frame latency, 0 = driver default (3)
    bool  forceScale      = false;   // render at renderScale (applied once the game has started up)
    float renderScale     = 0.667f;  // per-axis render scale = 1 / upscale ratio
    bool  unlockCvars     = true;    // allow setting cvars the retail build locks ("CANNOT be set in retail")
    bool  ultraPlus       = false;   // engine settings beyond the menu's Ultra (shadows, LOD, filtering)
    bool  ultrawideHud    = true;    // HUD elements at the screen edges on screens wider than 16:9 (hud.h)

    // [Debug] (ini only)
    bool  dumpShaders     = false;   // write every created shader's bytecode to <mod folder>\captures\shaders

    // [Keys] virtual key codes
    int keyCapture     = VK_F8;   // capture one frame (dispatch bindings + upscaler inputs)
    int keyToggleScale = VK_F7;   // reduced render resolution on/off
    int keyToggleFsr   = VK_F6;   // FSR on/off (engine TAA)
    int keyDebugView   = VK_F10;  // FSR debug visualization
    int keyOverlay     = VK_F2;   // settings overlay
    int keyBenchmark   = VK_F11;  // benchmark recording start/stop

    // [FSR]
    bool        fsrEnabled         = true;
    bool        fsrUpscale         = true;   // upscale: replaces TAA, post-processing at display resolution
                                             // native:  anti-aliasing only, at render resolution
    float       fsrSharpness       = 0.0f;   // RCAS strength, 0 = off
    bool        jitterCancellation = false;  // engine motion vectors contain the jitter delta
    bool        haltonJitter       = true;   // 32-phase Halton jitter instead of the engine's 8 phases
    bool        fsrDebugChecking   = false;  // FFX runtime validation messages in the log
    std::string fsrVersion;                  // provider pinned by name (e.g. "3.1.5"), empty = newest
    bool        fsrMipBias         = true;   // texture mip bias log2(render scale) - 1 while upscaling

    // [HDR]
    bool  hdrEnabled        = true;    // HDR10 output while Windows HDR is on (read at startup)
    float hdrPaperWhite     = 200.0f;  // nits of SDR white: UI and the scene's mid tones
    float hdrPeakNits       = 0.0f;    // highlight ceiling, 0 = what the display reports
    float hdrHighlightBoost = 1.3f;    // contrast above mid grey (1 = the scene's own)

    // [SSGI] screen-space global illumination (FSR path only)
    bool  ssgiEnabled   = true;
    float ssgiStrength  = 1.0f;   // indirect light multiplier
    float ssgiRadius    = 2.0f;   // view-space units (world units)
    float ssgiThickness = 0.5f;   // assumed thickness of occluders
    int   ssgiSlices    = 3;
    int   ssgiSteps     = 8;      // per side of a slice
    float ssgiAo        = 0.0f;   // 0..1: GI visibility applied on top of the game's ambient occlusion
    float ssgiAlbedo    = 0.45f;  // assumed surface reflectance where the material colour is unknown

    // [Materials] material data for SSGI and SSR
    bool materialCapture = true;  // patch the forward shaders to output material data (read at startup)
    bool decalCapture    = true;  // alpha-blended draws (decals) blend their material data in (wet decals)

    // [SSR] screen-space reflections (FSR path; needs material capture)
    bool  ssrEnabled        = true;
    float ssrStrength       = 1.0f;
    float ssrMaxRoughness   = 0.6f;   // rougher surfaces keep the game's cubemap reflections only
    float ssrThickness      = 0.05f;  // a hit may lie this fraction of its depth behind the surface
    int   ssrSteps          = 32;
    float ssrMaxDistance    = 40.0f;  // view units
    float specularOcclusion = 1.0f;   // GI visibility darkens the game's cubemap reflection (needs GI)

    // [Grading] colour grading in the game's tonemap pass (SDR and HDR); neutral values = the game's look
    bool  gradeEnabled     = false;
    float gradeExposure    = 0.0f;   // EV
    float gradeContrast    = 1.0f;
    float gradeSaturation  = 1.0f;
    float gradeVibrance    = 0.0f;
    float gradeTemperature = 0.0f;   // -1 cool .. +1 warm
    float gradeTint        = 0.0f;   // -1 green .. +1 magenta
    float gradeGameLook    = 1.0f;   // strength of the game's colour grading LUT (0 = neutral)

    // Runtime only (not saved): diagnostics
    int   ssgiDebug       = 0;      // pack.hlsl debugView: 1 bounce .. 10 GI history length
    bool  fsrDebugView    = false;
    bool  fsrAutoExposure = false;  // FSR computes exposure itself instead of using the engine's
    float highlightClamp  = 0.0f;   // max exposed luminance of a pixel fed to FSR, 0 = off
    bool  hdrHeatmap      = false;  // HDR output shows scene brightness in false colour

    bool operator==(const Config&) const = default;
};

namespace config
{
    void Load();  // reads the ini and publishes the first snapshot
    void Save();  // writes the draft

    std::shared_ptr<const Config> Current();  // snapshot of the current frame
    Config Draft();                            // pending edits (copy)
    void   SetDraft(const Config& draft);
    bool   Publish();                          // frame boundary; true if the snapshot changed

    template <typename Fn>
    void Edit(Fn&& edit)
    {
        Config draft = Draft();
        edit(draft);
        SetDraft(draft);
    }
}
