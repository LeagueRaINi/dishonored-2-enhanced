// HDR output, at Present: the HDR scene plus the UI layer -> HDR10 (Rec.2020 primaries, PQ) back buffer.
//
// The UI layer is the game's (stand-in) back buffer: the game's scene copy into it is replaced by a clear to
// transparent, so the HUD and menus draw onto transparent black (premultiplied colour, coverage in alpha). It is
// read through a UNORM view: the game writes it in display gamma (sRGB). Frames without a 3D scene (videos,
// loading screens) leave the game's own full-screen image in it, shown opaque.

Texture2D<float4> g_scene : register(t0);  // HDR scene, linear Rec.709, 1 = paper white
Texture2D<float4> g_ui    : register(t1);  // UI layer, sRGB-encoded, premultiplied

cbuffer CompositeParams : register(b0)
{
    float paperWhiteNits;
    float uiNits;
    float peakNits;
    uint  hasScene;  // 0: the UI layer is the whole (opaque) frame
    uint  heatmap;   // 1: false colour of the scene brightness (debug)
    float3 pad;
};

// Grey up to paper white, then blue, green, yellow, red as highlights approach the peak.
float3 Heatmap(float nits)
{
    float x = nits / paperWhiteNits;
    if (x <= 1.0)
        return x.xxx * 0.5;
    float t = saturate(log2(x) / max(log2(peakNits / paperWhiteNits), 1e-3));
    float3 ramp = t < 0.33 ? lerp(float3(0, 0, 1), float3(0, 1, 0), t / 0.33)
                : t < 0.66 ? lerp(float3(0, 1, 0), float3(1, 1, 0), (t - 0.33) / 0.33)
                           : lerp(float3(1, 1, 0), float3(1, 0, 0), (t - 0.66) / 0.34);
    return ramp;
}

float3 SrgbToLinear(float3 c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 Rec709ToRec2020(float3 c)
{
    static const float3x3 m = {
        0.627404, 0.329283, 0.043313,
        0.069097, 0.919540, 0.011362,
        0.016391, 0.088013, 0.895595,
    };
    return mul(m, c);
}

float3 PqEncode(float3 nits)
{
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(saturate(nits / 10000.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

float4 main(float4 pos : SV_Position) : SV_Target
{
    int3 p = int3(pos.xy, 0);
    float4 ui = g_ui.Load(p);
    float3 nits;
    if (hasScene)
    {
        float3 scene = max(g_scene.Load(p).rgb, 0.0) * paperWhiteNits;
        if (heatmap)
            scene = Heatmap(dot(scene, float3(0.2126, 0.7152, 0.0722))) * paperWhiteNits;
        float a = saturate(ui.a);
        float3 uiColor = a > 1e-4 ? SrgbToLinear(saturate(ui.rgb / a)) * uiNits : 0.0;
        nits = scene * (1.0 - a) + uiColor * a;
    }
    else
    {
        nits = SrgbToLinear(saturate(ui.rgb)) * uiNits;
    }
    nits = min(Rec709ToRec2020(max(nits, 0.0)), peakNits);
    return float4(PqEncode(nits), 1.0);
}
