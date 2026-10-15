// ImGui content of the overlay: style and the settings window (the corner counter lives in counter.cpp).
#pragma once

#include "core.h"

namespace ui
{
    enum class Page { Upscaling, Lighting, Display, Colour, Performance, Console, Log, Diagnostics, Count };
    void SelectPage(Page page); // also used by the standalone preview's screenshot automation
    void LoadFonts();          // after creating the ImGui context, before the first frame
    void ApplyStyle(float scale);
    void DrawSettings();   // main window
}
