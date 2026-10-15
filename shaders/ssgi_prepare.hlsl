// SSGI 1/4: half-resolution geometry and radiance from the engine's depth and HDR colour (render region).
// Conventions in ssgi_common.hlsli: one representative full-resolution pixel per half texel, scene-linear radiance.

#include "ssgi_common.hlsli"

struct AutoExposure
{
    float EngineLuminanceFactor;
    float LuminanceFactor;
    float MinLuminanceLDR;
    float MaxLuminanceLDR;
    float MiddleGreyLuminanceLDR;
    float EV;
    float Fstop;
    uint  PeakHistogramValue;
};

Texture2D<float>               g_depth    : register(t0);  // engine depth (render region at the top left)
Texture2D<float3>              g_color    : register(t1);  // engine HDR colour (TAA input), stored units
StructuredBuffer<AutoExposure> g_exposure : register(t2);

RWTexture2D<float4> o_geometry : register(u0);  // view-space normal, view z (0 = sky)
RWTexture2D<float4> o_radiance : register(u1);  // scene-linear radiance (mip 0 of the chain)

float ViewZ(int2 p)
{
    p = clamp(p, int2(0, 0), int2(renderSize) - 1);
    float d = g_depth[p];
    return d > 0.0 ? cameraNear / d : 0.0;
}

float3 Position(int2 p, float z) { return ViewPositionFull(float2(p) + 0.5, z); }

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    int2 p = RepresentativePixel(int2(id));
    float z = ViewZ(p);

    // Radiance of the representative pixel's surface: the 2x2 block's pixels at a similar depth (no light from
    // a different surface across a silhouette), in scene-linear units.
    float scale = compressedHdr != 0 ? g_exposure[exposureIndex].EngineLuminanceFactor * viewWhite : 1.0;
    float3 radiance = 0.0;
    float count = 0.0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 q = min(p + int2(i & 1, i >> 1), int2(renderSize) - 1);
        float zq = ViewZ(q);
        if (i == 0 || (IsSky(z) ? IsSky(zq) : abs(zq - z) <= 0.02 * z))
        {
            radiance += g_color[q];
            count += 1.0;
        }
    }
    o_radiance[id] = float4(radiance / count * scale, 1.0);

    if (IsSky(z))
    {
        o_geometry[id] = 0.0;
        return;
    }
    // Normal from the depth gradient, each axis from the neighbour on the same surface (smaller depth step), so
    // silhouettes don't bend the normal.
    float3 c = Position(p, z);
    float3 r = Position(p + int2(1, 0), ViewZ(p + int2(1, 0)));
    float3 l = Position(p - int2(1, 0), ViewZ(p - int2(1, 0)));
    float3 d = Position(p + int2(0, 1), ViewZ(p + int2(0, 1)));
    float3 u = Position(p - int2(0, 1), ViewZ(p - int2(0, 1)));
    float3 dx = abs(r.z - c.z) < abs(c.z - l.z) ? r - c : c - l;
    float3 dy = abs(d.z - c.z) < abs(c.z - u.z) ? d - c : c - u;
    float3 n = normalize(cross(dx, dy));
    if (dot(n, -c) < 0.0)
        n = -n;  // facing the camera
    o_geometry[id] = float4(n, z);
}
