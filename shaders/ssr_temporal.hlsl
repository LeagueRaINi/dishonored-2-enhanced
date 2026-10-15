// SSR 2/3: temporal accumulation of the reflections at half resolution (after the FidelityFX reflection denoiser's
// approach), in the trace's confidence-premultiplied form.
//
// The history is reprojected with the surface's motion vectors and validated like the GI history (expected
// previous depth through world space, normal). Reflections move differently from the surface, so it is clipped to
// this frame's 3x3 neighbourhood statistics (mean +- 1.25 standard deviations, in YCoCg). Smooth surfaces respond
// quickly (sharp reflections must not lag), rough ones accumulate longer. The ray length is accumulated too, so the
// spatial filter keeps its radius on frames where the ray missed.

#include "ssgi_common.hlsli"

Texture2D<float4> g_current          : register(t0);  // this frame's trace: premultiplied radiance, confidence
Texture2D<float4> g_geometry         : register(t1);  // normal, view z
Texture2D<float4> g_history          : register(t2);  // previous accumulation
Texture2D<float2> g_historyDepth     : register(t3);  // previous: view z, ray length
Texture2D<float2> g_motion           : register(t4);  // engine motion vectors (full resolution)
Texture2D<float2> g_surface          : register(t5);  // full resolution: roughness LOD, metallic
Texture2D<float>  g_hitDistance      : register(t6);  // this frame's ray length (0 = miss)
Texture2D<float4> g_previousGeometry : register(t7);  // previous frame's normal, view z

RWTexture2D<float4> o_history      : register(u0);
RWTexture2D<float2> o_historyDepth : register(u1);  // view z, accumulated ray length

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    float4 current = g_current[id];
    float hitDistance = g_hitDistance[id];
    float4 geometry = g_geometry[id];
    float z = geometry.w;

    float3 m1 = 0.0, m2 = 0.0;
    float lowA = current.a, highA = current.a;
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            float4 c = g_current[clamp(int2(id) + int2(x, y), int2(0, 0), int2(halfSize) - 1)];
            float3 v = ToYCoCg(c.rgb);
            m1 += v;
            m2 += v * v;
            lowA = min(lowA, c.a);
            highA = max(highA, c.a);
        }
    }
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));

    float4 result = current;
    float rayLength = hitDistance;
    if (historyValid > 0.0 && !IsSky(z))
    {
        float2 full = FullFromHalf(float2(id));
        float2 previousHalf = HalfFromFull(full - g_motion[RepresentativePixel(int2(id))] * float2(renderSize));
        int2 previous = int2(round(previousHalf));
        if (all(previous >= 0) && all(previous < int2(halfSize)))
        {
            float expectedZ = PreviousViewZ(full, z);
            float2 depth = g_historyDepth[previous];
            if (abs(depth.x - expectedZ) <= 0.02 * expectedZ && dot(g_previousGeometry[previous].xyz, geometry.xyz) >= 0.9)
            {
                float roughness = saturate(g_surface[RepresentativePixel(int2(id))].r / maxIblLod);
                float alpha = lerp(0.25, 0.06, sqrt(roughness));
                float4 history = g_history[previous];
                float3 clipped = FromYCoCg(clamp(ToYCoCg(history.rgb), mean - 1.25 * sigma, mean + 1.25 * sigma));
                result = lerp(float4(clipped, clamp(history.a, lowA, highA)), current, alpha);
                rayLength = hitDistance > 0.0 ? lerp(depth.y, hitDistance, depth.y > 0.0 ? alpha : 1.0) : depth.y;
            }
        }
    }
    o_history[id] = max(result, 0.0);
    o_historyDepth[id] = float2(z, rayLength);
}
