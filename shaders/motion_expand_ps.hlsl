// Upscale mode: after FSR, motion blur runs at display resolution and samples the engine's motion vector texture
// over its whole extent. Motion vectors are viewport-UV deltas (resolution independent), so the render-resolution
// field is expanded to display size with a nearest-texel lookup.

Texture2D<float2> g_motion : register(t0);  // render-resolution copy shared with D3D12

cbuffer MotionUpscaleParams : register(b0)
{
    uint2 renderSize;
    uint2 outputSize;
};

float2 main(float4 pos : SV_Position) : SV_Target
{
    uint2 src = min(uint2(pos.xy * (float2)renderSize / (float2)outputSize), renderSize - 1);
    return g_motion.Load(int3(src, 0));
}
