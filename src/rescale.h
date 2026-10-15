// Upscale mode: render-scaled inputs of the display-resolution post chain.
//
// Some textures the post passes sample are rendered before FSR at render resolution: the motion blur velocity
// buffer, the depth of field CoC/data maps, the refraction map. Their content covers only the render-scaled
// top-left region, but after the switch the post shaders sample them with unscaled UVs, which stretches the
// valid corner over the screen and cuts the effect off at the right and bottom edges.
//
// The first time a display-resolution draw samples such a texture in a frame, it is stretched in place over its
// whole extent. Candidates are render-target textures sized at display / 1, 2 or 4 that the post chain has not
// written itself this frame.
#pragma once

#include "core.h"

#include <initializer_list>

namespace rescale
{
    // At the TAA dispatch of an upscale frame, before the post chain is recorded. displaySized: textures that
    // already hold display-resolution content (the FSR output, the expanded motion vectors).
    void BeginFrame(UINT renderWidth, UINT renderHeight, UINT displayWidth, UINT displayHeight,
                    std::initializer_list<ID3D11Resource*> displaySized);

    // Display-resolution draws and dispatches, before the original call.
    void BeforeDraw(ID3D11DeviceContext* ctx);
    void BeforeDispatch(ID3D11DeviceContext* ctx);

    // At the TAA dispatch (FSR or the game's own), with a reduced render scale: the engine's scene colour has a
    // bright last column and row in its render-scaled region (some scene pass reads past the region; the vanilla
    // game shows it too). They are replaced with their inner neighbours before the colour is upscaled.
    void RepairEdge(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* color, UINT width, UINT height);
}
