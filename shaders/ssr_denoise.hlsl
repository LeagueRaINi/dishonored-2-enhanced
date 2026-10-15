// SSR 3/3: spatial filter of the accumulated reflections (5x5 a-trous taps, confidence-premultiplied, so radiance
// and confidence are averaged with the same weights). The radius follows what a glossy reflection blurs over:
// roughness times the accumulated ray length, projected to pixels (a mirror isn't filtered at all). Weights stop at
// depth and normal edges and between surfaces of different roughness, so reflections stay on their surface.

#include "ssgi_common.hlsli"

Texture2D<float4> g_input    : register(t0);  // temporal result: premultiplied radiance, confidence
Texture2D<float4> g_geometry : register(t1);  // normal, view z
Texture2D<float2> g_depth    : register(t2);  // view z, accumulated ray length
Texture2D<float2> g_surface  : register(t3);  // full resolution: roughness LOD, metallic

RWTexture2D<float4> o_output : register(u0);

float Roughness(int2 halfPixel)
{
    return saturate(g_surface[RepresentativePixel(halfPixel)].r / maxIblLod);
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    float4 center = g_geometry[id];
    float roughness = Roughness(int2(id));
    // Footprint of the glossy lobe: about roughness * ray length, in half-resolution pixels at this depth.
    float footprint = roughness * g_depth[id].y / max(center.w * projScale.y, 1e-4) * 0.5 * float(halfSize.y);
    int step = int(clamp(footprint * 0.25, 0.0, 4.0));
    if (IsSky(center.w) || step == 0)
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
            int2 q = clamp(int2(id) + int2(x, y) * step, int2(0, 0), int2(halfSize) - 1);
            float4 g = g_geometry[q];
            if (IsSky(g.w))
                continue;
            float w = kernel[abs(x)] * kernel[abs(y)];
            w *= exp(-abs(g.w - center.w) / (0.02 * center.w));
            w *= pow(saturate(dot(g.xyz, center.xyz)), 32.0);
            w *= exp(-abs(Roughness(q) - roughness) * 8.0);
            sum += g_input[q] * w;
            weightSum += w;
        }
    }
    o_output[id] = weightSum > 1e-6 ? sum / weightSum : g_input[id];
}
