// Upscale mode: stretches a texture the engine rendered before FSR (content in the render-scaled top-left
// region) over its whole extent, so the display-resolution post passes can sample it with unscaled UVs.
// Bilinear from four Loads, so no sampler state has to be saved and restored around the draw.

Texture2D<float4> g_source : register(t0);  // copy of the texture

cbuffer StretchParams : register(b0)
{
    float2 validSize;  // render-scaled region, in texels
    float2 scale;      // validSize / texture size
};

float4 main(float4 pos : SV_Position) : SV_Target
{
    float2 p = clamp(pos.xy * scale - 0.5, 0.0, validSize - 1.0);
    int2   i = int2(p);
    int2   j = min(i + 1, int2(validSize) - 1);
    float2 f = p - i;
    float4 top    = lerp(g_source.Load(int3(i.x, i.y, 0)), g_source.Load(int3(j.x, i.y, 0)), f.x);
    float4 bottom = lerp(g_source.Load(int3(i.x, j.y, 0)), g_source.Load(int3(j.x, j.y, 0)), f.x);
    return lerp(top, bottom, f.y);
}
