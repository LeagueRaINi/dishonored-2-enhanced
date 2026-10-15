// F8 diagnostics: logs one frame's compute dispatches with their bindings (plus the decoded frame constants at the
// TAA pass) and its draw passes (runs of draws with the same pixel shader and render targets). Used to find engine
// passes and their resources.
#pragma once

#include "core.h"

namespace capture
{
    void SetBackbuffer(const void* resource);  // marked in the capture header (overlay, on (re)init)
    void Begin(uint64_t frame);
    void End(uint64_t frame);  // at the next frame boundary
    bool Active();
    void LogDispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z);
    void LogDraw(ID3D11DeviceContext* ctx, UINT count);  // count: vertices or indices of the draw
}
