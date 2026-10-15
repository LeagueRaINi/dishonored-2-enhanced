// Ultrawide HUD. The game's HUD is a 1920x1080 Iggy (Flash) movie shown centred at 16:9; on wider screens
// the health/mana ring and anything else in the movie's side columns end up far from the screen edges. Those
// elements are moved out to the edges; world markers, the crosshair, the hotbar and other centred elements stay.
#pragma once

#include "core.h"

namespace hud
{
    // Iggy UI vertex shaders (HUD, menus): element transform in cb0 rows 0-1, stage (screen pixels) -> clip in
    // rows 5-7.
    bool IsUiVertexShader(uint64_t hash);

    // Draw hooks (record time). AfterDraw puts the game's constants back; it must be called after every draw for
    // which NeedsAfterDraw() was true.
    void OnDraw(ID3D11DeviceContext* ctx);
    void OnFrame();  // frame boundary (main thread)
    bool NeedsAfterDraw();
    void AfterDraw(ID3D11DeviceContext* ctx);
}
