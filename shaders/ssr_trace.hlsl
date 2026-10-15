// SSR 1/3: screen-space reflections at half resolution.
//
// Per pixel with material data (shader_patch.h: roughness from the game's own IBL sample LOD), one reflection ray
// per frame: the half vector is importance-sampled from the GGX distribution (visible normals), so rough surfaces
// get blurry reflections over time. The ray starts slightly off the surface, is clipped to the screen and marched
// in screen space with perspective-correct depth. A hit needs a crossing: the previous sample in front of the depth
// buffer, this one behind it by at most the thickness; it is refined with a binary search and checked again, and
// rejected on back faces or its own pixel. The hit's radiance comes from the prefiltered mip chain, wider for
// rougher surfaces and longer rays.
//
// Output is confidence-premultiplied (rgb = radiance * confidence, a = confidence) so every later filter treats the
// two channels consistently; misses are 0.

#include "ssgi_common.hlsli"

Texture2D<float4> g_geometry : register(t0);  // half resolution: normal, view z
Texture2D<float4> g_radiance : register(t1);  // half resolution radiance mip chain (scene-linear)
Texture2D<float2> g_surface  : register(t2);  // full resolution: roughness LOD (-1 = unknown), metallic
SamplerState      s_linear   : register(s0);

RWTexture2D<float4> o_reflection  : register(u0);  // premultiplied reflected radiance, confidence
RWTexture2D<float>  o_hitDistance : register(u1);  // view-space length of the ray (0 = no hit), for the denoiser

// GGX visible-normal sample (Heitz 2018) around n for view direction v.
float3 SampleHalfVector(float3 v, float3 n, float alpha, float2 xi)
{
    float3 up = abs(n.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    float3 vl = float3(dot(v, t), dot(v, b), dot(v, n));
    float3 vh = normalize(float3(alpha * vl.x, alpha * vl.y, vl.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0 ? float3(-vh.y, vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(xi.x);
    float phi = 2.0 * kPi * xi.y;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
    float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    float3 hl = normalize(float3(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
    return hl.x * t + hl.y * b + hl.z * n;
}

// Parameter where the segment start + t * (end - start) leaves [lo, hi] (t >= 0), 1 if it doesn't.
float ClipToRect(float2 start, float2 end, float2 lo, float2 hi)
{
    float2 d = end - start;
    float t = 1.0;
    [unroll] for (int a = 0; a < 2; ++a)
    {
        if (d[a] > 1e-6)
            t = min(t, (hi[a] - start[a]) / d[a]);
        else if (d[a] < -1e-6)
            t = min(t, (lo[a] - start[a]) / d[a]);
    }
    return max(t, 0.0);
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= halfSize.x || id.y >= halfSize.y)
        return;
    o_reflection[id] = 0.0;
    o_hitDistance[id] = 0.0;
    float4 geometry = g_geometry[id];
    float lod = g_surface[RepresentativePixel(int2(id))].r;
    if (IsSky(geometry.w) || lod < 0.0)
        return;
    float roughness = saturate(lod / maxIblLod);
    if (roughness > ssrMaxRoughness)
        return;

    float3 n = geometry.xyz;
    float3 surface = ViewPositionHalf(float2(id), geometry.w);
    float3 v = normalize(-surface);
    float alpha = roughness * roughness;
    float3 h = alpha < 0.002 ? n : SampleHalfVector(v, n, alpha, Random2(id, frameIndex));
    float3 r = reflect(-v, h);
    if (dot(r, n) <= 0.0)
        return;

    // Start just off the surface (scale-aware), end at the maximum distance or just in front of the near plane.
    float3 p = surface + n * (0.003 * surface.z);
    float length_ = ssrMaxDistance;
    if (r.z < 0.0)
        length_ = min(length_, (cameraNear * 2.0 - p.z) / r.z);
    float3 e = p + r * length_;
    float2 start = HalfCoordOf(p);
    float2 end = HalfCoordOf(e);
    float tEnd = ClipToRect(start, end, float2(0.0, 0.0), float2(halfSize) - 1.0);
    float segmentPixels = length(end - start) * tEnd;
    if (segmentPixels < 1.5)
        return;
    float tStart = 1.5 / length(end - start);  // at least 1.5 texels away from the origin
    float invStart = 1.0 / p.z, invEnd = 1.0 / e.z;

    float jitter = Noise(float2(id) + 13.0, frameIndex);
    float previousT = tStart;
    bool previousInFront = true;
    float hitT = -1.0;
    for (uint i = 0; i < ssrSteps; ++i)
    {
        float u = (float(i) + jitter) / float(ssrSteps);
        float t = lerp(tStart, tEnd, u * u);  // denser near the surface
        float2 coord = lerp(start, end, t);
        float rayZ = 1.0 / lerp(invStart, invEnd, t);
        float sceneZ = g_geometry[int2(round(coord))].w;
        bool inFront = IsSky(sceneZ) || rayZ <= sceneZ;
        if (!inFront && previousInFront && rayZ - sceneZ <= ssrThickness * rayZ)
        {
            float lo = previousT, hi = t;
            [unroll] for (int k = 0; k < 5; ++k)
            {
                float mid = 0.5 * (lo + hi);
                float midSceneZ = g_geometry[int2(round(lerp(start, end, mid)))].w;
                if (!IsSky(midSceneZ) && 1.0 / lerp(invStart, invEnd, mid) > midSceneZ)
                    hi = mid;
                else
                    lo = mid;
            }
            float hiZ = 1.0 / lerp(invStart, invEnd, hi);
            float hiSceneZ = g_geometry[int2(round(lerp(start, end, hi)))].w;
            if (!IsSky(hiSceneZ) && hiZ - hiSceneZ >= 0.0 && hiZ - hiSceneZ <= ssrThickness * hiZ)
            {
                hitT = hi;
                break;
            }
        }
        previousInFront = inFront;
        previousT = t;
    }
    if (hitT < 0.0)
        return;

    float2 hitCoord = lerp(start, end, hitT);
    int2 hitTexel = int2(round(hitCoord));
    float4 hitGeometry = g_geometry[hitTexel];
    if (all(hitTexel == int2(id)) || dot(hitGeometry.xyz, r) >= 0.0)
        return;  // its own texel, or the back of a surface (what's really seen there is unknown)

    float2 uv = (hitCoord + 0.5) / float2(halfSize);
    float2 edge = saturate(min(uv, 1.0 - uv) * 10.0);  // fade at the screen border
    float rough = 1.0 - smoothstep(ssrMaxRoughness * 0.7, ssrMaxRoughness, roughness);
    float confidence = edge.x * edge.y * rough;
    float distancePixels = length(hitCoord - start);
    float lodRadiance = clamp(log2(max(distancePixels * roughness * 0.5, 1.0)), 0.0, 4.0);
    float3 radiance = g_radiance.SampleLevel(s_linear, uv, lodRadiance).rgb;
    o_reflection[id] = float4(radiance * confidence, confidence);
    o_hitDistance[id] = length(ViewPositionHalf(hitCoord, hitGeometry.w) - surface);
}
