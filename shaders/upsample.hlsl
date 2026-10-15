// Upscale mode fallback: when FSR can't run in a frame whose post chain is already set to display resolution,
// the render-resolution image is stretched to the whole texture with a bilinear filter (no anti-aliasing).

Texture2D<float3>   g_color  : register(t0);  // render-resolution image in the top-left of a display-size texture
SamplerState        g_linear : register(s0);
RWTexture2D<float3> o_result : register(u0);  // engine taaresult, written over its whole extent

cbuffer UpsampleParams : register(b0)
{
    uint2  outputSize;
    float2 uvScale;  // render size / texture size
};

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= outputSize.x || id.y >= outputSize.y)
        return;
    float2 uv = (float2(id) + 0.5) / float2(outputSize) * uvScale;
    o_result[id] = g_color.SampleLevel(g_linear, uv, 0);
}
