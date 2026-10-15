// SSGI 3/4: temporal accumulation at half resolution (scene-linear units, up to 32 frames).
//
// The history is fetched where the engine's motion vectors say the pixel was ((current - previous) in viewport UV),
// from up to four bilinear taps, each kept only if it saw the same surface: its stored view z matches the depth
// this point had in the previous frame (reprojected through world space with the engine's matrices, so camera
// motion doesn't reject or falsely accept history) and its normal agrees. Once enough frames are accumulated, the
// history is clipped to this frame's neighbourhood statistics, so changed lighting doesn't linger.

#include "ssgi_common.hlsli"

Texture2D<float4> g_current          : register(t0);  // this frame's trace: light, visibility
Texture2D<float4> g_geometry         : register(t1);  // normal, view z
Texture2D<float4> g_history          : register(t2);  // previous accumulation
Texture2D<float2> g_historyMeta      : register(t3);  // previous: frame count, view z
Texture2D<float2> g_motion           : register(t4);  // engine motion vectors (render region, full resolution)
Texture2D<float4> g_previousGeometry : register(t5);  // previous frame's normal, view z

RWTexture2D<float4> o_history     : register(u0);
RWTexture2D<float2> o_historyMeta : register(u1);

static const float kMaxFrames = 32.0;

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    float4 current = g_current[id];
    float4 geometry = g_geometry[id];
    float z = geometry.w;
    if (IsSky(z) || historyValid <= 0.0)
    {
        o_history[id] = current;
        o_historyMeta[id] = float2(1.0, z);
        return;
    }

    float2 full = FullFromHalf(float2(id));
    float2 previousFull = full - g_motion[RepresentativePixel(int2(id))] * float2(renderSize);
    float2 previousHalf = HalfFromFull(previousFull);
    float expectedZ = PreviousViewZ(full, z);

    int2 base = int2(floor(previousHalf));
    float2 f = previousHalf - float2(base);
    float4 history = 0.0;
    float frames = 0.0, weightSum = 0.0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 offset = int2(i & 1, i >> 1);
        int2 q = base + offset;
        if (any(q < 0) || any(q >= int2(halfSize)))
            continue;
        float2 meta = g_historyMeta[q];
        float4 previousGeometry = g_previousGeometry[q];
        if (abs(meta.y - expectedZ) > 0.02 * expectedZ || dot(previousGeometry.xyz, geometry.xyz) < 0.9)
            continue;
        float w = (offset.x ? f.x : 1.0 - f.x) * (offset.y ? f.y : 1.0 - f.y);
        history += g_history[q] * w;
        frames += meta.x * w;
        weightSum += w;
    }

    float4 result = current;
    float count = 1.0;
    if (weightSum > 0.05)
    {
        history /= weightSum;
        count = min(frames / weightSum + 1.0, kMaxFrames);
        if (count > 4.0)
        {
            float3 m1 = 0.0, m2 = 0.0;
            [unroll] for (int y = -1; y <= 1; ++y)
            {
                [unroll] for (int x = -1; x <= 1; ++x)
                {
                    float3 c = ToYCoCg(g_current[clamp(int2(id) + int2(x, y), int2(0, 0), int2(halfSize) - 1)].rgb);
                    m1 += c;
                    m2 += c * c;
                }
            }
            float3 mean = m1 / 9.0;
            float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
            history.rgb = FromYCoCg(clamp(ToYCoCg(history.rgb), mean - 3.0 * sigma, mean + 3.0 * sigma));
        }
        result = lerp(history, current, 1.0 / count);
    }
    o_history[id] = max(result, 0.0);
    o_historyMeta[id] = float2(count, z);
}
