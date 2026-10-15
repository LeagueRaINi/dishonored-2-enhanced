// Copies the engine's TAA inputs into the textures shared with the D3D12 upscaler, adding the screen-space lighting
// (GI bounce light, SSR) to the colour.
// Exposure mirrors the engine TAA shader:
//   tonemap input = stored * (compressedHdr ? ELF * viewWhite : 1) * exp(-evBias) * ELF

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

Texture2D<float3>               g_color      : register(t0);
Texture2D<float2>               g_motion     : register(t1);
Texture2D<float>                g_depth      : register(t2);
StructuredBuffer<AutoExposure>  g_exposure   : register(t3);
Texture2D<float4>               g_gi         : register(t4);  // half res: indirect light, visibility
Texture2D<float4>               g_geometry   : register(t5);  // half res: normal, view z
Texture2D<float4>               g_material   : register(t6);  // base colour (0: unknown)
Texture2D<float4>               g_reflection : register(t7);  // half res: reflected radiance, confidence
Texture2D<float4>               g_specular   : register(t8);  // the game's IBL reflection
Texture2D<float4>               g_debug      : register(t9);  // half res: the selected debug view's buffer
Texture2D<float2>               g_surface    : register(t10); // roughness LOD (-1: unknown), metallic

RWTexture2D<float4> o_color    : register(u0);
RWTexture2D<float2> o_motion   : register(u1);
RWTexture2D<float>  o_depth    : register(u2);
RWTexture2D<float>  o_exposure : register(u3);

cbuffer PackParams : register(b0)
{
    uint2  renderSize;
    uint   exposureIndex;
    uint   compressedHdr;
    float  viewWhite;
    float  evBias;
    float  maxLuminance;   // diagnostics: max exposed luminance per pixel, 0 = off
    float  cameraNear;
    float  giStrength;     // indirect light multiplier
    float  aoStrength;     // 0 = keep the game's ambient occlusion only, 1 = full GI visibility on top
    uint   debugView;      // 1 bounce added, 2 visibility, 3 albedo, 4 reflections, 5 roughness, 6 normals,
                           // 7 SSR ray length, 8 SSR raw hits, 9 GI raw trace, 10 GI history length
    uint   giEnabled;
    uint2  halfSize;
    float  giAlbedo;       // assumed surface reflectance where the material colour is unknown
    uint   materialValid;  // g_material / g_specular / g_surface hold this frame's data
    float2 projScale;      // view.xy = ndc.xy * z * projScale
    float  ssrStrength;
    uint   ssrEnabled;
    float  specularOcclusion;  // how much GI visibility darkens the game's cubemap reflection (0 = off)
    float3 padding;
};

static const float3 kLuma709 = float3(0.2126, 0.7152, 0.0722);
static const float  kMaxIblLod = 6.0;  // the game's IBL cubemaps: 256x256, 7 mips

float Exposure()
{
    float elf = g_exposure[exposureIndex].EngineLuminanceFactor;
    float exposure = exp(-evBias) * elf;
    if (compressedHdr != 0)
        exposure *= elf * viewWhite;
    return exposure;
}

// A half-resolution buffer at a full-resolution pixel: bilinear taps weighted by view-depth similarity. Half
// texel q stands for full-resolution pixel 2q (ssgi_common.hlsli). If no tap is on this pixel's surface, the
// neutral value is returned instead of light from unrelated surfaces.
float4 UpsampleHalf(Texture2D<float4> source, uint2 id, float z, float4 neutral)
{
    float2 position = float2(id) * 0.5;
    int2 base = int2(floor(position));
    float2 f = position - float2(base);
    float4 sum = 0.0;
    float weightSum = 0.0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 offset = int2(i & 1, i >> 1);
        int2 q = clamp(base + offset, int2(0, 0), int2(halfSize) - 1);
        float bilinear = (offset.x ? f.x : 1.0 - f.x) * (offset.y ? f.y : 1.0 - f.y);
        float w = bilinear * exp(-abs(g_geometry[q].w - z) / (0.02 * z));
        sum += source[q] * w;
        weightSum += w;
    }
    return weightSum > 1e-3 ? sum / weightSum : neutral;
}

// Blue (0) .. green .. yellow .. red (1).
float3 Heat(float t)
{
    return t < 0.33 ? lerp(float3(0, 0, 1), float3(0, 1, 0), t / 0.33)
         : t < 0.66 ? lerp(float3(0, 1, 0), float3(1, 1, 0), (t - 0.33) / 0.33)
                    : lerp(float3(1, 1, 0), float3(1, 0, 0), (t - 0.66) / 0.34);
}

// Split-sum environment BRDF, analytic fit (Karis 2014): the share of the incoming reflection that is reflected.
float3 EnvironmentBrdf(float3 f0, float roughness, float nv)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    float exposure = Exposure();
    if (id.x == 0 && id.y == 0)
        o_exposure[uint2(0, 0)] = exposure;
    if (id.x >= renderSize.x || id.y >= renderSize.y)
        return;

    float3 color = g_color[id];
    float depth = g_depth[id];
    if ((giEnabled != 0 || ssrEnabled != 0) && depth > 0.0)
    {
        float z = cameraNear / depth;
        // Lighting is accumulated in scene-linear units; the colour here is stored, which with compressed HDR
        // buffers is divided by ELF * view white.
        float storage = compressedHdr != 0 ? max(g_exposure[exposureIndex].EngineLuminanceFactor * viewWhite, 1e-6) : 1.0;
        float4 material = materialValid != 0 ? g_material[id] : 0.0;
        float2 surface = materialValid != 0 ? g_surface[id] : float2(-1.0, 0.0);
        float metallic = saturate(surface.g);
        bool knownMaterial = any(material.rgb > 0.0);

        // Albedo: the material's base colour where a patched forward shader wrote it, else estimated. A pixel's
        // brightness mostly says how lit it is, not how reflective, so the estimate takes the hue from the pixel
        // and assumes the reflectance (giAlbedo); brightly lit pixels may use their own, higher value. Near-black
        // pixels have no reliable hue: grey.
        float3 exposed = color * exposure;
        float peak = max(max(exposed.r, exposed.g), exposed.b);
        float3 hue = lerp(1.0, exposed / max(peak, 1e-6), saturate(peak * 50.0));
        float3 albedo = knownMaterial ? min(material.rgb, 0.95)
                                      : max(hue * giAlbedo, min(exposed / (exposed + 0.6), 0.9));
        float3 lit = color;
        float visibility = 1.0;

        if (giEnabled != 0)
        {
            float4 gi = UpsampleHalf(g_gi, id, z, float4(0.0, 0.0, 0.0, 1.0));
            visibility = gi.a;
            float3 bounce = gi.rgb / storage * albedo * (1.0 - metallic) * giStrength;  // metals: no diffuse bounce
            lit = lit * lerp(1.0, gi.a, aoStrength) + bounce;
            if (debugView == 1)
                color = bounce;
            else if (debugView == 2)
                color = gi.a.xxx / max(exposure, 1e-6) * 0.18;
        }

        float3 specularIbl = materialValid != 0 ? g_specular[id].rgb : 0.0;
        bool occlude = giEnabled != 0 && specularOcclusion > 0.0;
        if ((ssrEnabled != 0 || occlude) && surface.r >= 0.0)
        {
            // SSR replaces the game's cubemap reflection where a ray hit (weighted by its confidence); the
            // cubemap stays elsewhere, darkened where the GI trace found the surroundings occluded (the cubemap
            // knows nothing of what's next to the surface). Both, like the game's IBL sample, in scene units.
            // The reflection is confidence-premultiplied (rgb = radiance * confidence).
            float roughness = saturate(surface.r / kMaxIblLod);
            float4 reflection = ssrEnabled != 0 ? UpsampleHalf(g_reflection, id, z, 0.0) : 0.0;
            float2 uv = (float2(id) + 0.5) / float2(renderSize);
            float3 view = normalize(-float3(float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * projScale * z, z));
            float3 n = g_geometry[min(int2(id / 2), int2(halfSize) - 1)].xyz;
            float3 f0 = lerp(0.04, knownMaterial ? albedo : 0.04, metallic);
            float nv = saturate(dot(n, view));
            float3 weight = EnvironmentBrdf(f0, roughness, nv);
            float3 ibl = specularIbl;
            // Specular occlusion from ambient visibility (Lagarde & de Rousiers 2014).
            float occlusion = saturate(pow(abs(nv + visibility), exp2(-16.0 * roughness - 1.0)) - 1.0 + visibility);
            float3 cubemapLeft = ibl * (1.0 - reflection.a);
            float3 delta = (reflection.rgb - ibl * reflection.a) * ssrStrength;
            if (occlude)
                delta -= cubemapLeft * (1.0 - occlusion) * specularOcclusion;
            lit = max(lit + delta * weight / storage, 0.0);
            if (debugView == 4)
                color = reflection.rgb / storage;
            else if (debugView == 5)
                color = roughness.xxx / max(exposure, 1e-6) * 0.5;
        }

        // Debug views of intermediate buffers, shown at about mid-grey brightness whatever the exposure.
        float display = 0.5 / max(exposure, 1e-6);
        int2 halfPixel = min(int2(id / 2), int2(halfSize) - 1);
        float4 debug = g_debug[halfPixel];
        if (debugView == 3)
            color = albedo * display;
        else if (debugView == 6)
            color = (g_geometry[halfPixel].xyz * 0.5 + 0.5) * display;
        else if (debugView == 7)
            color = (debug.r > 0.0 ? Heat(saturate(log2(1.0 + debug.r) / log2(101.0))) : 0.0) * display;
        else if (debugView == 8)
            color = debug.rgb / storage;
        else if (debugView == 9)
            color = debug.rgb / storage * giStrength;
        else if (debugView == 10)
            color = saturate(debug.r / 32.0).xxx * display;
        else if (debugView == 0 || (debugView <= 2 && giEnabled == 0) || ((debugView == 4 || debugView == 5) && ssrEnabled == 0))
            color = lit;
    }
    if (maxLuminance > 0.0)
    {
        // Scales the pixel (keeping its hue) so that its exposed luminance stays below the limit.
        float luminance = dot(color, kLuma709) * exposure;
        if (luminance > maxLuminance)
            color *= maxLuminance / luminance;
    }
    o_color[id]  = float4(color, 1.0);
    o_motion[id] = g_motion[id];
    o_depth[id]  = depth;
}
