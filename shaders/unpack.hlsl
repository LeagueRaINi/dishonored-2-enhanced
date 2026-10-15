// Writes the upscaler output back into the engine's TAA result texture.

Texture2D<float4>   g_upscaled : register(t0);
RWTexture2D<float3> o_result   : register(u0);

cbuffer UnpackParams : register(b0)
{
    uint2 outputSize;
    uint2 padding;
};

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= outputSize.x || id.y >= outputSize.y)
        return;
    o_result[id] = g_upscaled[id].rgb;
}
