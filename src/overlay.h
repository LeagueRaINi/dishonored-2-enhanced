// ImGui settings overlay drawn at swap chain Present, plus input blocking while it is open.
#pragma once

#include "core.h"

namespace overlay
{
    void Install(ID3D11Device* device);  // hooks Present/ResizeBuffers and the game's input paths
    bool IsVisible();
    void PollToggleKey();                // frame tick
}
