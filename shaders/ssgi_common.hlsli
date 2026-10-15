// Shared by the SSGI / SSR shaders: parameters (src/ssgi.cpp Params) and conventions.
//
// View space is the mod's own: x right, y up, z = distance in front of the camera. The engine depth is reversed
// infinite Z: device depth d = near / z, so z = near / d (d = 0: sky).
//
// Half-resolution texel id represents ONE full-resolution pixel, 2 * id (its centre is full-resolution coordinate
// 2 * id + 0.5): depth, normal and position all come from that pixel. Continuous half-resolution coordinates use
// the same convention: half coordinate h <-> full coordinate 2 * h + 0.5.
//
// Lighting is accumulated in scene-linear units: the engine stores colour divided by ELF * view white while its
// HDR buffers are compressed, and that factor changes with exposure (pack.hlsl converts back).

cbuffer SsgiParams : register(b0)
{
    uint2  renderSize;
    uint2  halfSize;
    float2 projScale;      // view.xy = ndc.xy * z * projScale (1 / projection[0], 1 / projection[5])
    float  cameraNear;
    float  radius;         // GI support, view units
    float  thickness;      // GI: assumed thickness of occluders
    uint   frameIndex;
    uint   slices;
    uint   steps;
    float  stepWidth;      // a-trous step in pixels
    float  historyValid;   // 0 = start over (first frame, camera cut, resize)
    uint   exposureIndex;  // into the engine's auto-exposure buffer
    uint   compressedHdr;
    float  ssrMaxRoughness;
    float  ssrThickness;   // fraction of the ray's view depth a hit may be behind the surface
    uint   ssrSteps;
    float  ssrMaxDistance; // view units
    float  maxIblLod;      // LOD of roughness 1 in the game's cubemaps
    float  viewWhite;
    float2 paramsPad;
    row_major float4x4 inverseViewProjection;   // engine: world = (ndc.x, ndc.y, depth, 1) * M, then / w
    row_major float4x4 previousViewProjection;  // engine: previous clip = (world, 1) * M
};

static const float kPi = 3.14159265;
static const float kHalfPi = 1.57079633;

bool IsSky(float z) { return z <= 0.0; }  // geometry stores z = 0 for the sky

int2 RepresentativePixel(int2 halfId) { return min(halfId * 2, int2(renderSize) - 1); }

float2 FullFromHalf(float2 halfCoord) { return halfCoord * 2.0 + 0.5; }
float2 HalfFromFull(float2 fullCoord) { return (fullCoord - 0.5) * 0.5; }

float2 NdcFromFull(float2 fullCoord)
{
    float2 uv = fullCoord / float2(renderSize);
    return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

float3 ViewPositionFull(float2 fullCoord, float z) { return float3(NdcFromFull(fullCoord) * projScale * z, z); }
float3 ViewPositionHalf(float2 halfCoord, float z) { return ViewPositionFull(FullFromHalf(halfCoord), z); }

// Continuous half-resolution coordinate of a view-space point.
float2 HalfCoordOf(float3 p)
{
    float2 ndc = p.xy / (p.z * projScale);
    float2 full = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * float2(renderSize);
    return HalfFromFull(full);
}

// The view z this frame's surface point had in the previous frame (camera motion only), for history validation.
float PreviousViewZ(float2 fullCoord, float z)
{
    float4 world = mul(float4(NdcFromFull(fullCoord), cameraNear / z, 1.0), inverseViewProjection);
    world /= world.w;
    float4 clip = mul(float4(world.xyz, 1.0), previousViewProjection);
    float depth = clip.z / clip.w;
    return depth > 0.0 ? cameraNear / depth : z;
}

// Interleaved gradient noise (Jimenez 2014), shifted per frame.
float Noise(float2 p, uint frame)
{
    p += 5.588238 * float(frame & 63);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// Two decorrelated uniform numbers per pixel and frame (PCG hash).
uint Pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float2 Random2(uint2 p, uint frame)
{
    uint h = Pcg(p.x + Pcg(p.y + Pcg(frame)));
    return float2(h & 0xFFFF, Pcg(h) & 0xFFFF) / 65536.0;
}

float Luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
float3 ToYCoCg(float3 c) { return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25))); }
float3 FromYCoCg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
