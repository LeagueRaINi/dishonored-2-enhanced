// SSGI 2/4: screen-space indirect lighting with a visibility bitmask (Therrien, Levesque, Gilet 2023), at half
// resolution.
//
// Per pixel, a few slices through the hemisphere around the view direction. Along each slice the samples' front
// and back faces (the back assumed `thickness` behind) cover a range of 32 angular sectors of the hemisphere above
// the pixel's normal. Sectors a sample covers first carry its light: the bounce is the sum of newly covered
// sectors times the sample's radiance and the two cosine terms. The fraction of sectors left uncovered is the
// visibility (ambient occlusion). Only samples within the support distance (2 x radius) take part, so far-away
// geometry neither occludes nor lights. Slice rotation and step offsets change every frame; the temporal pass
// accumulates them.

#include "ssgi_common.hlsli"

Texture2D<float4> g_geometry : register(t0);  // normal, view z
Texture2D<float4> g_radiance : register(t1);  // mip chain (ssgi_downsample.hlsl)
SamplerState      s_linear   : register(s0);

RWTexture2D<float4> o_light : register(u0);  // indirect light, visibility

// Sectors touched by the angular range [minHorizon, maxHorizon] of [0, 1] (both ends included).
uint Sectors(float minHorizon, float maxHorizon)
{
    uint first = min(uint(minHorizon * 32.0), 31u);
    uint last = min(uint(ceil(maxHorizon * 32.0)), 32u);
    uint count = last > first ? last - first : 0u;
    return count > 0 ? (0xFFFFFFFFu >> (32u - count)) << first : 0u;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    float4 geometry = g_geometry[id];
    if (IsSky(geometry.w))
    {
        o_light[id] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float3 n = geometry.xyz;
    float3 p = ViewPositionHalf(float2(id), geometry.w);
    float3 v = normalize(-p);
    float support = radius * 2.0;

    // Sample radius in half-resolution pixels for `radius` view units at this depth.
    float radiusPixels = clamp(radius / (p.z * projScale.y) * 0.5 * float(halfSize.y), 4.0, 0.25 * float(halfSize.y));
    float2 xi = Random2(id, frameIndex);

    float3 light = 0.0;
    float visibility = 0.0;
    for (uint s = 0; s < slices; ++s)
    {
        float phi = (float(s) + xi.x) * kPi / float(slices);
        float2 omega = float2(cos(phi), sin(phi));               // screen direction (y down)
        float3 direction = float3(omega.x, -omega.y, 0.0);        // the same direction in view space
        float3 ortho = direction - dot(direction, v) * v;
        float3 axis = normalize(cross(direction, v));
        float3 projectedN = n - axis * dot(n, axis);
        float projectedLength = max(length(projectedN), 1e-4);
        float angleN = sign(dot(ortho, projectedN)) * acos(saturate(dot(projectedN, v) / projectedLength));

        uint occlusion = 0;
        [unroll] for (int side = -1; side <= 1; side += 2)
        {
            for (uint t = 0; t < steps; ++t)
            {
                float f = (float(t) + xi.y) / float(steps);
                float2 offset = side * omega * max(f * f * radiusPixels, float(t) + 1.0);
                int2 q = int2(round(float2(id) + offset));
                if (any(q < 0) || any(q >= int2(halfSize)))
                    break;
                float4 sampleGeometry = g_geometry[q];
                if (IsSky(sampleGeometry.w))
                    continue;
                float3 delta = ViewPositionHalf(float2(q), sampleGeometry.w) - p;
                float distance = length(delta);
                if (distance < 1e-4 || distance > support)
                    continue;  // beyond the support: neither occludes nor lights
                float3 h = delta / distance;
                float front = acos(clamp(dot(h, v), -1.0, 1.0));
                float back = acos(clamp(dot(normalize(delta - v * thickness), v), -1.0, 1.0));
                float2 horizons = saturate((side * float2(front, back) - angleN + kHalfPi) / kPi);
                uint covered = Sectors(min(horizons.x, horizons.y), max(horizons.x, horizons.y));
                uint fresh = covered & ~occlusion;
                if (fresh != 0)
                {
                    float3 sampleN = sampleGeometry.xyz;
                    float emit = saturate(dot(sampleN, -h));
                    float falloff = saturate(1.0 - distance / support);
                    // Farther samples stand for a larger area: read the prefiltered radiance of that footprint.
                    float lod = clamp(log2(max(length(offset) / float(steps), 1.0)), 0.0, 4.0);
                    float2 uv = (float2(q) + 0.5) / float2(halfSize);
                    float3 radiance = g_radiance.SampleLevel(s_linear, uv, lod).rgb;
                    light += float(countbits(fresh)) / 32.0 * radiance * saturate(dot(n, h)) * emit * falloff;
                }
                occlusion |= covered;
            }
        }
        visibility += 1.0 - float(countbits(occlusion)) / 32.0;
    }
    // Uniform angular sectors weighted by cos(theta) integrate to 2/pi over a lit hemisphere; normalized to 1.
    o_light[id] = float4(light * kHalfPi / float(slices), visibility / float(slices));
}
