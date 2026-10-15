// SSGI: radiance mip chain. Each level averages 2x2 texels of the previous one, so trace samples far from the
// pixel read the light of the whole area they stand for (a small bright lamp is seen consistently, not by luck).

Texture2D<float4>   g_source : register(t0);  // previous level
RWTexture2D<float4> o_target : register(u0);

cbuffer DownsampleParams : register(b0)
{
    uint2 sourceSize;
    uint2 targetSize;
};

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= targetSize.x || id.y >= targetSize.y)
        return;
    int2 p = int2(id * 2);
    int2 last = int2(sourceSize) - 1;
    float4 sum = g_source[min(p, last)] + g_source[min(p + int2(1, 0), last)] + g_source[min(p + int2(0, 1), last)] +
                 g_source[min(p + int2(1, 1), last)];
    o_target[id] = sum * 0.25;
}
