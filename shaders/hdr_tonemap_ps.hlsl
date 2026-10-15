// HDR output: the engine's tonemap pass (PS 6b486ef38e1bbe42) transcribed from its disassembly, plus a second
// output with the highlights extended past SDR white.
//
// SV_Target0 is the game's SDR result, bit for bit the original math, so everything downstream of the pass still
// sees what it expects. SV_Target1 is that same graded result scaled by a luminance ratio: up to mid grey the two
// are identical; highlights follow the scene without the game's shoulder, rolled off towards the display peak
// (HdrLuminance). The game's grading LUT, brightness and gamma stay in effect. Units: 1 = paper white.
//
// Bindings, constant buffers and the vertex shader are the game's own.

Texture2D<float4>   ro_postfx_bloom_lensdirt_to        : register(t0);
Texture2D<float4>   ro_postfx_bloom_lensdirt_from      : register(t1);
Texture3D<float4>   ro_tonemapping_finalcolorcube      : register(t2);
Texture2D<float4>   ro_postfx_lensflares_texlensflares : register(t3);
Texture2D<float4>   ro_postfx_bloom_texbloom           : register(t4);
Texture2D<float4>   ro_viewcolormap                    : register(t5);

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
StructuredBuffer<AutoExposure> ro_postfx_luminance_buffautoexposure : register(t6);

SamplerState smp_linearclamp : register(s0);

cbuffer PerViewCB : register(b1)
{
    float4 pv[49];  // [7].z / [8].x: view matrix terms (streak rotation), [10].xy: subpixel offset,
                    // [44].w: compressed HDR buffers, [48].zw: resolution scale
};

cbuffer PerInstanceCB : register(b2)
{
    float4 cb_positiontoviewtexture;
    float4 cb_tonemappingparms;    // .x: split between the two curve segments
    float4 cb_tonemappingcoeffs1;  // upper segment: (A, C, B, D) of (A*x + B) / (C*x + D)
    float4 cb_tonemappingcoeffs0;  // lower segment
    float4 cb_lensdirt_usedefault;
    float2 cb_gamma_brightness;
    uint2  cb_exposureindex;
    float  cb_bloom_veil_strength;
    float  cb_view_white_level;
    float  cb_customevbias;
    float  cb_streakwidth;
    float  cb_streakradius;
    float  cb_streakopacity;
    float  cb_streakoffset;
    float  cb_lensdirt_strength;
    float  cb_lensdirt_blendweight;
    uint   cb_bloom_enabled;
};

// The mod's parameters (bound at b3 for this draw only): HDR extension and colour grading (neutral = the game's look).
cbuffer HdrParams : register(b3)
{
    float  hdr_peak;            // display peak / paper white
    float  hdr_knee;            // tonemapped luminance where the extension is fully in effect (0..1)
    float  hdr_boost;           // contrast above mid grey: 1 = the scene's own, higher spreads glows and skies upwards
    float  hdr_pad;
    float3 grade_whiteBalance;  // per-channel multipliers (temperature / tint), luminance kept
    float  grade_exposure;      // multiplier
    float  grade_contrast;      // around mid grey, in log2 space (1 = unchanged)
    float  grade_saturation;    // 1 = unchanged
    float  grade_vibrance;      // saturation boost for muted colours (0 = none)
    float  grade_lutStrength;   // 1 = the game's colour grading LUT, 0 = neutral
};

static const float3 kLuma709 = float3(0.2126, 0.7152, 0.0722);

float Curve(float x, float4 c) { return (c.x * x + c.z) / (c.y * x + c.w); }

float3 GameCurve(float3 x)
{
    float3 y;
    [unroll] for (int i = 0; i < 3; ++i)
        y[i] = Curve(x[i], x[i] < cb_tonemappingparms.x ? cb_tonemappingcoeffs0 : cb_tonemappingcoeffs1);
    return y;
}

// Curve input x with Curve(x) == t (the rational inverts in closed form).
float InverseCurve(float t)
{
    float4 c = cb_tonemappingcoeffs1;
    float x = (c.z - t * c.w) / (t * c.y - c.x);
    if (x < cb_tonemappingparms.x)
    {
        c = cb_tonemappingcoeffs0;
        x = (c.z - t * c.w) / (t * c.y - c.x);
    }
    return x;
}

// HDR luminance for linear luminance y, whose game-tonemapped luminance is ySdr.
//
// The scene without the game's shoulder: y scaled so that mid grey lands where the game puts it. That is rolled
// off towards the display peak above the knee, and blended in between mid grey and the knee, so shadows and mid
// tones keep the game's curve exactly while highlights use the HDR range.
float HdrLuminance(float y, float ySdr)
{
    const float midGrey = 0.18;
    float k = hdr_knee;
    float scale = midGrey / max(InverseCurve(midGrey), 1e-6);
    float u = scale * y;
    if (u > midGrey)
        u = midGrey * pow(abs(u / midGrey), hdr_boost);  // u > midGrey here
    float range = max(hdr_peak - k, 1e-3);
    float h = u <= k ? u : k + range * (1.0 - exp(-(u - k) / range));
    return lerp(ySdr, max(h, ySdr), smoothstep(midGrey, k, ySdr));
}

// Colour grading on the linear scene, before the curve, so highlights still roll off naturally.
float3 Grade(float3 color)
{
    color *= grade_whiteBalance * grade_exposure;
    float y = dot(color, kLuma709);
    if (y <= 1e-6)
        return color;
    float midGrey = max(InverseCurve(0.18), 1e-4);  // scene value the game's curve shows as mid grey
    float graded = midGrey * exp2(log2(y / midGrey) * grade_contrast);
    color *= graded / y;
    float3 grey = graded.xxx;
    float peak = max(max(color.r, color.g), color.b);
    float chroma = peak > 1e-6 ? (peak - min(min(color.r, color.g), color.b)) / peak : 0.0;
    float saturation = grade_saturation * (1.0 + grade_vibrance * (1.0 - chroma));
    return max(lerp(grey, color, saturation), 0.0);
}

struct Output
{
    float4 sdr : SV_Target0;
    float4 hdr : SV_Target1;
};

Output main(float4 interp : INTERP0, float4 pos : SV_Position)
{
    float3 color = ro_viewcolormap.Load(int3(uint2(pos.xy), 0)).rgb;
    float exposure = ro_postfx_luminance_buffautoexposure[cb_exposureindex.y].EngineLuminanceFactor * cb_view_white_level;
    bool compressed = asuint(pv[44].w) != 0;
    if (compressed)
        color *= exposure;

    if (cb_bloom_enabled)
    {
        float2 uv = interp.xy * pv[48].zw;
        float2 dirtUV = interp.xy + pv[10].xy;
        float3 dirt;
        if (0.0 < cb_lensdirt_usedefault.x)
        {
            dirt = ro_postfx_bloom_lensdirt_from.SampleLevel(smp_linearclamp, dirtUV, 0).rgb;
        }
        else
        {
            float3 from = ro_postfx_bloom_lensdirt_from.SampleLevel(smp_linearclamp, dirtUV, 0).rgb;
            float3 to = ro_postfx_bloom_lensdirt_to.SampleLevel(smp_linearclamp, dirtUV, 0).rgb;
            dirt = from + cb_lensdirt_blendweight * (to - from);
        }
        if (0.0 < cb_bloom_veil_strength)
        {
            float3 bloom = ro_postfx_bloom_texbloom.SampleLevel(smp_linearclamp, uv, 0).rgb;
            if (compressed)
                bloom *= exposure;
            bloom *= cb_bloom_veil_strength;
            color += (dirt * cb_lensdirt_strength) * bloom + bloom;
        }

        // Lens streaks: a noisy ring around the screen centre, rotated with the camera.
        float2 d = interp.xy - 0.5;
        float s, co;
        sincos(pv[7].z + pv[8].x, s, co);
        float2 r = float2(co * d.x - 0.5 * s * d.y, 0.5 * s * d.x + co * d.y);
        float ringDistance2 = dot(r, r);
        float angle = atan2(r.x, r.y) + 3.141593;
        float t = angle * 651.898621;
        float cell = floor(t);
        float n0 = frac(sin(cell) * 43758.546875);
        float n1 = frac(sin(cell + 1.0) * 43758.546875);
        float radius = lerp(n0, n1, t - cell) * cb_streakoffset + cb_streakradius;
        float ring = saturate((abs(ringDistance2 - radius * radius) - cb_streakwidth) / -cb_streakwidth);
        dirt += ring * ring * (3.0 - 2.0 * ring) * cb_streakopacity;

        float2 flareUV = saturate(uv - cb_positiontoviewtexture.zw / pv[48].zw);
        float3 flares = ro_postfx_lensflares_texlensflares.SampleLevel(smp_linearclamp, flareUV, 0).rgb;
        if (compressed)
            flares *= exposure;
        color += flares * (dirt * 0.8 + 0.2);
    }

    color = max(color, 0.0) * interp.z;  // vignette
    color = Grade(color);
    float3 tonemapped = GameCurve(color);
    float3 lut = ro_tonemapping_finalcolorcube.SampleLevel(smp_linearclamp, (tonemapped * 31.0 + 0.5) * 0.03125, 0).rgb;
    // The LUT's neutral equivalent: an identity cube stored as sRGB returns the linear value of its coordinate.
    float3 neutral = tonemapped <= 0.04045 ? tonemapped / 12.92 : pow(max((tonemapped + 0.055) / 1.055, 0.0), 2.4);
    lut = lerp(neutral, lut, grade_lutStrength);
    float3 sdr = exp(log(saturate(lut * cb_gamma_brightness.y)) * cb_gamma_brightness.x);

    float y = dot(color, kLuma709);
    float ySdr = Curve(y, y < cb_tonemappingparms.x ? cb_tonemappingcoeffs0 : cb_tonemappingcoeffs1);
    float ratio = HdrLuminance(y, ySdr) / max(ySdr, 1e-4);

    Output o;
    o.sdr = float4(sdr, 1.0);
    o.hdr = float4(sdr * max(ratio, 1.0), 1.0);
    return o;
}
