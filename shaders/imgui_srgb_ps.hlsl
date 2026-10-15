// Dear ImGui pixel shader for sRGB render targets: ImGui colors are authored in sRGB, so they are
// linearized here and the target's sRGB encoding brings them back (otherwise the UI looks washed out).

struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float4 col : COLOR0;
    float2 uv  : TEXCOORD0;
};

sampler   sampler0 : register(s0);
Texture2D texture0 : register(t0);

float4 main(PS_INPUT input) : SV_Target
{
    float4 c = input.col * texture0.Sample(sampler0, input.uv);
    return float4(pow(saturate(c.rgb), 2.2), c.a);
}
