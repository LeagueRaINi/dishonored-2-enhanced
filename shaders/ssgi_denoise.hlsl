// SSGI 4/4: edge-aware a-trous filter (5x5 B3 spline taps at `stepWidth` pixels), run twice with growing steps.
// Weights fall off with view-depth and normal differences, so light doesn't bleed across edges.

#include "ssgi_common.hlsli"

Texture2D<float4> g_input    : register(t0);  // light, visibility
Texture2D<float4> g_geometry : register(t1);  // normal, view z

RWTexture2D<float4> o_output : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    float4 center = g_geometry[id];
    if (IsSky(center.w))
    {
        o_output[id] = g_input[id];
        return;
    }
    static const float kernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };
    float4 sum = 0.0;
    float weightSum = 0.0;
    [unroll] for (int y = -2; y <= 2; ++y)
    {
        [unroll] for (int x = -2; x <= 2; ++x)
        {
            int2 q = clamp(int2(id) + int2(x, y) * int(stepWidth), int2(0, 0), int2(halfSize) - 1);
            float4 g = g_geometry[q];
            if (IsSky(g.w))
                continue;
            float w = kernel[abs(x)] * kernel[abs(y)];
            w *= exp(-abs(g.w - center.w) / (0.02 * center.w));
            w *= pow(saturate(dot(g.xyz, center.xyz)), 16.0);
            sum += g_input[q] * w;
            weightSum += w;
        }
    }
    o_output[id] = weightSum > 1e-6 ? sum / weightSum : g_input[id];
}
