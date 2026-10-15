// Corner frame counter: frame statistics, frame-time graph, FSR details and a benchmark recorder.
//
// Modes (Config::fpsCounter): 0 off, 1 minimal, 2 frame time, 3 FSR, 4 full, 5 benchmark.
// Numbers refresh at 4 Hz (only the graph is per frame), every row is fixed-width on a monospace grid and every
// mode has a fixed size, so the panel never jumps.
#pragma once

#include "core.h"

namespace counter
{
    constexpr int kModes = 6;
    inline constexpr const char* kModeNames = "Off\0Minimal\0Frame time\0FSR\0Full\0Benchmark\0";

    void LoadFont();               // before the first frame: Consolas Bold if available
    void SetRefreshRate(int hz);   // display refresh, the default frame-time target
    void OnPresent(float sampleMs = 0); // once per present; 0 measures the clock, preview supplies a synthetic interval
    void RecordFrame(float ms);    // measured frame interval; preview can supply deterministic samples
    void Draw();                   // inside an ImGui frame
    void ToggleBenchmark();        // any thread: starts/stops a benchmark recording
    float AverageFps();            // last 0.25 s, for other UI
}
