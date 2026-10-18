// void-inject hybrid ray-traced reflections (rt_reflect.rs, docs/reflections.md section 4).
// DXR 1.1 inline RayQuery in compute shaders (cs_6_5, precompiled by build.rs), recorded on
// FSR's D3D12 command list after its input fence wait and before the FSR dispatch, so the
// result lands in FSR's colour input before FSR reads it.
//
// Inputs written by the D3D11 SSR classify pass (ssr.rs `cs_rt_classify`, region-local): per
// pixel the weight the ray-traced reflection may still take (SSR blend weight it could not
// fill x roughness and distance fades), per half-resolution texel a ray request (biased
// view-frame origin + TMin, view-frame shading normal + GGX alpha; w = 0: none). The GGX
// directions are drawn here (sample_u: live one per frame, photo many). Live pipeline (Far Cry 6
// / AMD hybrid reflections): rt_prepare (counters) -> rt_compact (wave-aggregated ray list,
// results "not traced") -> rt_args (ray budget, indirect arguments, hit-shading calibration per
// screen tile) -> ExecuteIndirect rt_trace (closest hit in the camera-radius TLAS; a hit visible
// on screen takes the frame's own colour, others are shaded from the packed per-primitive data:
// sun with a shadow ray + ambient, scaled into stored units by the calibration, blended smoothly
// between the two; misses keep the engine's IBL) -> denoiser as the SSR one (ssr.rs): rt_resolve
// (5x5 ratio-estimator reuse), rt_temporal (engine motion vectors, YCoCg clamp, history by
// roughness and motion), rt_blur (roughness-scaled bilateral) -> rt_composite (3x3 half-texel
// upsampling, premultiplied confidence, lerp into the colour). Without the denoiser
// (ssr_temporal = 0) the composite reads the traced results directly.
// Photo: rt_photo (spp rays per request into a persistent sum) -> rt_composite (average).
// Root descriptors and typed loads are not bounds-checked: every index is clamped.

#define RS "RootConstants(num32BitConstants=50, b0), SRV(t0), SRV(t1), SRV(t2), " \
           "DescriptorTable(SRV(t3, numDescriptors=5)), DescriptorTable(UAV(u0, numDescriptors=14)), " \
           "SRV(t8)"

struct Consts
{
    float3 cam;       float radius;      // camera world position; TLAS radius (m)
    float4 rot0;                         // world -> view frame rows (x right, y up,
    float4 rot1;                         // z forward); view -> world = transpose
    float4 rot2;
    uint2 half_size;  uint2 region;      // request grid; SSR region size
    float4 proj;                         // P00, P11, jitter x, y
    float depth_mul;  float depth_add;  uint max_rays; uint flags;
    float3 sun_dir;   uint geom_count;   // world, toward the sun
    float3 sun_irr;   uint prim_count;   // absolute engine units
    float3 ambient;   float calib_init;  // absolute; initial stored / absolute scale
    float3 center;    uint frame;        // TLAS centre (world)
    uint2 origin;     uint debug;        float jitter; // region origin in the colour/depth
    uint type_count;                     // entries of InstanceTypes (0 = none bound)
    float firefly_k;                     // per-hit clamp: x the neighbourhood mean (E089)
};
ConstantBuffer<Consts> C : register(b0);
// u32 per TLAS instance index (rt::SceneView::instance_types, void_rt::dynamic::TYPE_*):
// 0 static, 1 moved static prop, 2 spawned prop, 3 character. Read only for the
// `instances` debug view and only with C.type_count > 0.
StructuredBuffer<uint> InstanceTypes : register(t8);

RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<uint> PrimOffsets : register(t1);
ByteAddressBuffer Prims : register(t2);
Texture2D<float4> Req0 : register(t3);
Texture2D<float4> Req1 : register(t4);
Texture2D<float4> RtWeight : register(t5); // rgb per-channel weight (material tint), a scalar
Texture2D<float> Depth : register(t6);      // FSR's copy of the hardware depth (reversed-Z)
Texture2D<float2> Motion : register(t7);    // FSR's motion input: engine (current - previous), viewport UV

RWTexture2D<float4> Result : register(u0);  // half res: (L c, c) stored units; a < 0 = not traced
RWStructuredBuffer<uint> RayList : register(u1);
RWByteAddressBuffer Args : register(u2);    // dispatch arguments (12 bytes)
RWTexture2D<float4> Color : register(u3);   // FSR's colour input (stored units)
// Header: 0 global scale (f32 bits), 4 / 8 / 12 E077 calibration sums (screen, shaded x 64,
// samples), 16 requested rays, 20 traced rays, 24 hits, 28 screen hits, 32 misses, 36 / 40
// smallest / largest tile scale, 44 calibration samples of the last frame. From 64: per screen
// tile (TX x TY) its scale (f32), sums of screen and shaded absolute luminance (x 1024) and
// the samples.
RWByteAddressBuffer Calib : register(u4);
RWTexture2D<float4> Ray : register(u5);      // half: traced view-frame direction, its pdf
RWTexture2D<float4> Resolved : register(u6); // half: (L c, c) after the reuse; a < 0 = no sample
RWTexture2D<float4> Hist : register(u7);     // half: accumulation (L c, c) written this frame
RWTexture2D<float4> Meta : register(u8);     // half: oct world normal, history length, linear z
RWTexture2D<float4> HistPrev : register(u9); // the previous frame's (F_HISTORY)
RWTexture2D<float4> MetaPrev : register(u10);
RWTexture2D<float4> Final : register(u11);   // what the composite reads: (L c, c); a < 0 = none
RWTexture2D<float> Moment : register(u12);   // half: second moment of the luminance (history)
RWTexture2D<float> MomentPrev : register(u13);

#define F_SHADOW 1u
#define F_NO_SCREEN 2u
#define F_PHOTO 4u    // composite: Final holds sums over C.max_rays samples per texel
#define F_CLEAR 8u    // rt_photo: start the accumulation from 0
#define F_HISTORY 16u // HistPrev / MetaPrev hold the previous frame
#define F_DENOISE 32u // resolve + temporal + blur ran (Meta valid)
#define F_LEGACY 64u  // offline comparison only: E077 hit shading (global calibration, hard
                      // on/off-screen switch, BVH fade around the TLAS centre, rotation always)
#define F_LINEAR 128u // offline comparison only: the E085 denoiser without firefly handling

#define DEBUG_RAYS 1u
#define DEBUG_HITS 2u
#define DEBUG_RT 3u
#define DEBUG_HISTORY 4u
#define DEBUG_INSTANCES 5u
#define NO_INSTANCE 0xFFFFFFFFu

#define TX 8u
#define TY 4u
#define TILES 32u
#define TILE0 64u
// Calibration: log-domain rate per frame with enough samples; relaxation of an unsampled tile
// toward the global scale.
#define CAL_RATE 0.05
#define CAL_RELAX 0.02
// On-screen weight of a hit fades out within this many pixels of the region border.
#define EDGE_PX 16.0

static const float PI = 3.14159265;

float3 to_world_dir(float3 v) { return v.x * C.rot0.xyz + v.y * C.rot1.xyz + v.z * C.rot2.xyz; }
float3 to_view_dir(float3 w) { return float3(dot(C.rot0.xyz, w), dot(C.rot1.xyz, w), dot(C.rot2.xyz, w)); }
float3 to_view(float3 w) { return to_view_dir(w - C.cam); }
float lin_z(float d)
{
    float z = C.depth_mul / (d - C.depth_add);
    return (z > 0.0 && z < 65504.0) ? z : 65504.0;
}
// View-frame position of region pixel coordinate px (pixel edges at integers) at depth z.
float3 view_pos(float2 px, float z)
{
    float2 ndc = float2(px.x / float(C.region.x) * 2.0 - 1.0, 1.0 - px.y / float(C.region.y) * 2.0);
    return float3((ndc - C.proj.zw) / C.proj.xy * z, z);
}
float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
uint cap() { return C.half_size.x * C.half_size.y; }
bool legacy() { return (C.flags & F_LEGACY) != 0u; }

// Fireflies (ssr.rs, E089): per-hit clamp to FIREFLY_K x the neighbourhood mean in the
// resolve; the temporal clamp widened by the luminance's own deviation over the history
// (second moment in Moment). C.firefly_k from rt_reflect.rs `FIREFLY_K`.
bool fireflies() { return (C.flags & F_LINEAR) == 0u; }

// Non-finite lanes -> 0, others clamped to [0, 65000]: on every colour and result write (a
// single NaN in FSR's input would spread through its accumulation).
float4 safe(float4 c)
{
    bool4 bad = (asuint(c) & 0x7f800000u) == 0x7f800000u;
    return select(bad, 0.0, clamp(c, 0.0, 65000.0));
}

float2 oct_enc(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
}

float3 oct_dec(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

float ggx_d(float NoH, float a)
{
    float a2 = a * a;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float smith_g1(float NoX, float a)
{
    float a2 = a * a;
    return 2.0 * NoX / (NoX + sqrt(a2 + (1.0 - a2) * NoX * NoX));
}

// Height-correlated Smith masking-shadowing.
float smith_g2(float NoV, float NoL, float a)
{
    float a2 = a * a;
    return 2.0 * NoL * NoV / (NoL * sqrt(a2 + (1.0 - a2) * NoV * NoV) + NoV * sqrt(a2 + (1.0 - a2) * NoL * NoL));
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void rt_prepare(uint3 id : SV_DispatchThreadID)
{
    Calib.Store(16, 0u);
    Calib.Store4(20, uint4(0, 0, 0, 0));
}

// One thread per request texel: marks the result "not traced" and appends the requested
// rays (one atomic per wave).
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_compact(uint3 id : SV_DispatchThreadID)
{
    bool inside = all(id.xy < C.half_size);
    bool want = inside && Req0.Load(int3(id.xy, 0)).w > 0.0;
    if (inside) Result[id.xy] = float4(0.0, 0.0, 0.0, -1.0);
    uint n = WaveActiveCountBits(want);
    uint base = 0;
    if (n > 0 && WaveIsFirstLane()) Calib.InterlockedAdd(16, n, base);
    base = WaveReadLaneFirst(base);
    if (want)
    {
        uint slot = base + WavePrefixCountBits(want);
        if (slot < cap()) RayList[slot] = (id.y << 16) | id.x;
    }
}

float valid_or(float s, float fallback) { return (s > 0.0 && s < 1e6) ? s : fallback; }

groupshared float3 g_cal[TILES];
groupshared float g_tile[TILES];

// Ray budget, indirect arguments (64 threads per group) and the hit-shading calibration from
// the previous frame's samples (one thread per tile). The scale relates the frame's stored
// colour of fully on-screen hits to our absolute shading of the same hits; it is smoothed in
// the log domain (the first estimate is taken as is), per tile clamped to 1/4..4 x the global
// scale, and a tile without samples relaxes toward the global scale.
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void rt_args(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    float g0 = asfloat(Calib.Load(0));
    bool have = g0 > 0.0 && g0 < 1e6;
    float g = have ? g0 : C.calib_init;
    uint at = TILE0 + 16u * t;
    float s0 = asfloat(Calib.Load(at));
    uint3 sums = Calib.Load3(at + 4u);
    g_cal[t] = float3(sums);
    GroupMemoryBarrierWithGroupSync();
    if (legacy())
    {
        if (t == 0u)
        {
            if (Calib.Load(12) >= 64u)
            {
                float screen = float(Calib.Load(4));
                float shaded = float(Calib.Load(8));
                if (shaded > 0.0)
                    g = clamp(lerp(g, g * screen / shaded, 0.1), 0.01, 1000.0);
            }
            Calib.Store(0, asuint(g));
            Calib.Store3(4, uint3(0, 0, 0));
        }
    }
    else
    {
        // No estimate yet (0): the global scale stays unset (readers use calib_init) and a
        // tile unset (readers use the global one) until their first samples, then jumps to
        // that estimate and follows later ones smoothly.
        float3 tot = 0.0;
        for (uint i = 0; i < TILES; i++) tot += g_cal[i];
        float gs = have ? g0 : 0.0;
        if (tot.z >= 64.0 && tot.y > 0.0)
        {
            float target = clamp(tot.x / tot.y, 0.01, 1000.0);
            gs = have ? exp(lerp(log(g0), log(target), CAL_RATE)) : target;
        }
        g = gs > 0.0 ? gs : C.calib_init;
        bool tile_have = s0 > 0.0 && s0 < 1e6;
        float s = tile_have ? s0 : 0.0;
        if (sums.z >= 16u && sums.y > 0u)
        {
            float target = clamp(float(sums.x) / float(sums.y), 0.25 * g, 4.0 * g);
            s = tile_have ? exp(lerp(log(s0), log(target), CAL_RATE)) : target;
        }
        else if (tile_have)
        {
            s = exp(lerp(log(s0), log(g), CAL_RELAX));
        }
        Calib.Store(at, asuint(s));
        Calib.Store3(at + 4u, uint3(0, 0, 0));
        g_tile[t] = s > 0.0 ? s : g;
        g = gs;
        if (t == 0u)
        {
            Calib.Store(0, asuint(g));
            Calib.Store(44, uint(tot.z));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0u)
    {
        if (!legacy())
        {
            float lo = 1e9, hi = 0.0;
            for (uint i = 0; i < TILES; i++)
            {
                lo = min(lo, g_tile[i]);
                hi = max(hi, g_tile[i]);
            }
            Calib.Store2(36, uint2(asuint(lo), asuint(hi)));
        }
        uint n = min(Calib.Load(16), min(C.max_rays, cap()));
        Args.Store3(0, uint3((n + 63) / 64, 1, 1));
        Calib.Store(20, n);
    }
}

// Calibration tile of half texel q (by its first pixel).
uint tile_of(uint2 q)
{
    uint2 t = uint2((float2(q) * 2.0 + 1.0) / float2(max(C.region, 1u)) * float2(TX, TY));
    t = min(t, uint2(TX - 1u, TY - 1u));
    return t.y * TX + t.x;
}

float global_scale() { return valid_or(asfloat(Calib.Load(0)), C.calib_init); }

float tile_scale(uint x, uint y, float g) { return valid_or(asfloat(Calib.Load(TILE0 + 16u * (y * TX + x))), g); }

// Stored / absolute scale for the request at q: bilinear between the tile centres (no steps
// at tile borders); E077: the global scale.
float scale_at(uint2 q)
{
    float g = global_scale();
    if (legacy()) return g;
    float2 t = (float2(q) * 2.0 + 1.0) / float2(max(C.region, 1u)) * float2(TX, TY) - 0.5;
    t = clamp(t, 0.0, float2(TX - 1u, TY - 1u));
    uint2 i0 = min(uint2(t), uint2(TX - 2u, TY - 2u));
    float2 f = t - float2(i0);
    float a = lerp(tile_scale(i0.x, i0.y, g), tile_scale(i0.x + 1u, i0.y, g), f.x);
    float b = lerp(tile_scale(i0.x, i0.y + 1u, g), tile_scale(i0.x + 1u, i0.y + 1u, g), f.x);
    return lerp(a, b, f.y);
}

struct Hit
{
    float3 pos;
    float3 n;
    float3 albedo;
    float metallic;
};

float3 oct_decode(uint w)
{
    float2 f = float2(w & 255u, (w >> 8) & 255u) / 255.0 * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

float4 unorm4(uint w) { return float4(w & 255u, (w >> 8) & 255u, (w >> 16) & 255u, w >> 24) / 255.0; }

// FC6 per-primitive data (rt::SceneView::prims): 6 words per triangle, per vertex
// (albedo.rgb + smoothness, oct normal xy + metallic + reflectance), object-space normal.
Hit surface_at(uint inst_id, uint geom, uint prim, float2 bary, float3x4 o2w, float3 org, float3 dir, float t)
{
    uint g = min(inst_id + geom, max(C.geom_count, 1u) - 1u);
    uint p = min(PrimOffsets[g] + prim, max(C.prim_count, 1u) - 1u);
    uint2 v0 = Prims.Load2(p * 24u);
    uint2 v1 = Prims.Load2(p * 24u + 8u);
    uint2 v2 = Prims.Load2(p * 24u + 16u);
    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float4 a = unorm4(v0.x) * w.x + unorm4(v1.x) * w.y + unorm4(v2.x) * w.z;
    float4 m = unorm4(v0.y) * w.x + unorm4(v1.y) * w.y + unorm4(v2.y) * w.z;
    float3 n_obj = oct_decode(v0.y) * w.x + oct_decode(v1.y) * w.y + oct_decode(v2.y) * w.z;
    // Inverse transpose of the object-to-world 3x3 (cofactors; any scale is normalised away).
    float3 c0 = o2w._m00_m10_m20, c1 = o2w._m01_m11_m21, c2 = o2w._m02_m12_m22;
    float3x3 cof = float3x3(cross(c1, c2), cross(c2, c0), cross(c0, c1));
    float3 n = normalize(mul(n_obj, cof));
    if (dot(n, dir) > 0.0) n = -n;
    Hit h;
    h.pos = org + dir * t;
    h.n = n;
    h.albedo = a.rgb;
    h.metallic = m.z;
    return h;
}

bool sun_visible(float3 p)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
    RayDesc r;
    r.Origin = p;
    r.Direction = C.sun_dir;
    r.TMin = 0.0;
    r.TMax = 2.0 * C.radius;
    q.TraceRayInline(Scene, 0, 0xFF, r);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

// Absolute radiance leaving the hit toward the ray origin: Lambert sun (shadow ray) +
// ambient; metals keep the ambient tinted by their albedo (no specular lobe at hits).
float3 shade(Hit h)
{
    float nl = saturate(dot(h.n, C.sun_dir));
    float vis = 1.0;
    if ((C.flags & F_SHADOW) != 0u && nl > 0.0)
        vis = sun_visible(h.pos + h.n * 0.02) ? 1.0 : 0.0;
    float3 amb = C.ambient * lerp(0.4, 1.0, h.n.z * 0.5 + 0.5);
    float3 diffuse = h.albedo * (1.0 - h.metallic);
    return diffuse * (C.sun_irr * (nl * vis / PI) + amb) + h.albedo * h.metallic * amb;
}

// Distance to the TLAS sphere boundary along the ray (0 = origin outside it).
float sphere_exit(float3 o, float3 d)
{
    float3 oc = o - C.center;
    float b = dot(oc, d);
    float c = dot(oc, oc) - C.radius * C.radius;
    if (c >= 0.0) return 0.0;
    return -b + sqrt(max(b * b - c, 0.0));
}

// Heitz 2018, GGX distribution of visible normals (local frame, n = z).
float3 sample_vndf(float3 ve, float a, float2 u)
{
    float3 vh = normalize(float3(a * ve.x, a * ve.y, ve.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(u.x);
    float phi = 6.2831853 * u.y;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(float3(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

static const uint BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

// Sample `index` of texel q: the 4x4 Hammersley set in Bayer order (stratified per 4x4 block,
// as the SSR trace) rotated by the R2 sequence of the index (Cranley-Patterson). Live: index =
// frame (0 = the fixed pattern); photo: running sample count.
float2 sample_u(uint2 q, uint index)
{
    uint k = BAYER[(q.x & 3u) + 4u * (q.y & 3u)];
    float2 u = float2((k + 0.5) / 16.0, (reversebits(k) >> 28) / 16.0 + 1.0 / 32.0);
    return frac(u + frac(float(index & 0xFFFFu) * float2(0.7548777, 0.5698403)));
}

// Live sample index: as the SSR trace (`Xi`), the pattern rotates per frame only where the
// previous accumulation at q holds >= 4 frames; where history is rejected or short (first
// frames, disocclusion, the swaying first-person weapon) the fixed pattern stays, which is
// stable frame to frame. E077: always rotated.
uint live_index(uint2 q)
{
    if (legacy()) return C.frame;
    bool rotate = (C.flags & F_HISTORY) != 0u && MetaPrev[q].z >= 3.5;
    return rotate ? max(C.frame, 1u) : 0u;
}

// GGX reflection direction (view frame) for the request (origin r0, normal + alpha r1) and the
// pdf it was drawn with (65000 = mirror).
float3 reflected(float4 r0, float4 r1, float2 u, out float pdf)
{
    float3 n = normalize(r1.xyz);
    float3 v = normalize(-r0.xyz);
    pdf = 65000.0;
    if (r1.w < 1e-4) return reflect(-v, n);
    float3 t = normalize(abs(n.y) < 0.999 ? cross(n, float3(0.0, 1.0, 0.0)) : cross(n, float3(1.0, 0.0, 0.0)));
    float3 b = cross(n, t);
    float3 h = sample_vndf(float3(dot(v, t), dot(v, b), dot(v, n)), r1.w, u);
    float3 r = reflect(-v, h.x * t + h.y * b + h.z * n);
    if (dot(r, n) <= 0.0) return reflect(-v, n);
    float NoV = saturate(dot(n, v)) + 1e-4;
    pdf = min(smith_g1(NoV, r1.w) * ggx_d(saturate(dot(n, normalize(v + r))), r1.w) / (4.0 * NoV), 65000.0);
    return r;
}

// One ray of request texel q along view-frame direction `rv`: (L c, c) in stored units, 0 =
// miss or outside the BVH. `calib`: contribute calibration samples (live only).
float4 trace_ray(uint2 q, float4 r0, float3 rv, bool calib, out bool hit, out bool screen, out uint inst)
{
    hit = false;
    screen = false;
    inst = NO_INSTANCE;
    float3 org = C.cam + to_world_dir(r0.xyz);
    float3 dir = normalize(to_world_dir(rv));
    float tmax = sphere_exit(org, dir);
    if (!(tmax > r0.w)) return 0.0;
    RayDesc ray;
    ray.Origin = org;
    ray.Direction = dir;
    ray.TMin = r0.w;
    ray.TMax = tmax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> rq;
    rq.TraceRayInline(Scene, 0, 0xFF, ray);
    while (rq.Proceed()) {}
    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 0.0; // miss: engine IBL stays
    inst = rq.CommittedInstanceIndex();
    Hit h = surface_at(rq.CommittedInstanceID(), rq.CommittedGeometryIndex(), rq.CommittedPrimitiveIndex(),
                       rq.CommittedTriangleBarycentrics(), rq.CommittedObjectToWorld3x4(), org, dir,
                       rq.CommittedRayT());
    // Confidence fades out where geometry may be missing. The TLAS holds the instances within
    // C.radius of its centre and is rebuilt once the camera moved 0.25 x radius from it
    // (rt/scene.rs), so everything within ~0.7 x radius of the camera is always present: a
    // fade around the camera moves continuously with it, where one around the centre would
    // step at every rebuild. E077: around the centre.
    float c = legacy()
        ? 1.0 - smoothstep(0.8 * C.radius, C.radius, length(h.pos - C.center))
        : 1.0 - smoothstep(0.5 * C.radius, 0.7 * C.radius, length(h.pos - C.cam));
    float3 shaded = shade(h);
    float3 L = shaded * scale_at(q);
    // A hit visible on screen at matching depth: the frame's own colour (exact, stored units)
    // and a calibration sample relating it to our shading. The on-screen weight falls off
    // smoothly with the depth mismatch and toward the region border, so a hit moving across
    // the test (jitter, camera motion) blends instead of switching.
    float3 v = to_view(h.pos);
    if ((C.flags & F_NO_SCREEN) == 0u && v.z > 0.05)
    {
        float2 ndc = C.proj.xy * v.xy / v.z + C.proj.zw;
        float2 px = float2(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * float2(C.region);
        if (all(px >= 0.0) && all(px < float2(C.region)))
        {
            int2 ip = min(int2(px), int2(C.region) - 1) + int2(C.origin);
            float zs = lin_z(Depth.Load(int3(ip, 0)));
            float e = abs(zs - v.z) / (0.02 * v.z + 0.05);
            float2 edge = min(px, float2(C.region) - px);
            float ws = legacy() ? (e < 1.0 ? 1.0 : 0.0)
                : (1.0 - smoothstep(0.5, 1.0, e)) * saturate(min(edge.x, edge.y) / EDGE_PX);
            if (ws > 0.0)
            {
                float3 s = Color[ip].rgb;
                if (calib && ((q.x ^ (q.y * 7u)) & 15u) == 0u)
                {
                    uint ignored;
                    if (legacy())
                    {
                        uint ls = uint(min(luminance(s) * 64.0, 65535.0));
                        uint lr = uint(min(luminance(L) * 64.0, 65535.0));
                        if (lr > 0u)
                        {
                            Calib.InterlockedAdd(4, ls, ignored);
                            Calib.InterlockedAdd(8, lr, ignored);
                            Calib.InterlockedAdd(12, 1u, ignored);
                        }
                    }
                    else if (ws >= 1.0)
                    {
                        uint ls = uint(min(luminance(s) * 1024.0 + 0.5, 65535.0));
                        uint lr = uint(min(luminance(shaded) * 1024.0 + 0.5, 65535.0));
                        if (lr > 0u)
                        {
                            uint at = TILE0 + 16u * tile_of(q);
                            Calib.InterlockedAdd(at + 4u, ls, ignored);
                            Calib.InterlockedAdd(at + 8u, lr, ignored);
                            Calib.InterlockedAdd(at + 12u, 1u, ignored);
                        }
                    }
                }
                L = lerp(L, s, ws);
                screen = ws >= 0.5;
            }
        }
    }
    if (!(c > 0.0) || any(!(L >= 0.0)) || any(L >= 65000.0)) return 0.0;
    hit = true;
    return float4(L * c, c);
}

// Debug colour of the instance a ray hit (void_rt::dynamic::type_colour, sRGB -> linear):
// static grey, moved yellow, spawned green, character red; a miss dark blue.
float3 instance_colour(uint inst)
{
    float3 s = float3(16.0, 20.0, 48.0);
    if (inst != NO_INSTANCE)
    {
        uint t = C.type_count > 0u ? InstanceTypes[min(inst, C.type_count - 1u)] : 0u;
        s = t == 1u ? float3(235.0, 205.0, 40.0)
            : t == 2u ? float3(40.0, 200.0, 70.0)
            : t == 3u ? float3(225.0, 40.0, 40.0)
            : float3(130.0, 130.0, 130.0);
    }
    return pow(s / 255.0, 2.2);
}

// Live: one GGX ray per listed request (the list subset within the budget); the result and
// the ray (direction, pdf) for the resolve.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void rt_trace(uint3 id : SV_DispatchThreadID)
{
    uint raw = min(Calib.Load(16), cap());
    uint n = min(Calib.Load(20), raw);
    if (id.x >= n) return;
    // Over budget: an even stride through the list, offset per frame.
    uint k = n < raw ? min(uint((float(id.x) + C.jitter) * float(raw) / float(n)), raw - 1u) : id.x;
    uint e = RayList[k];
    uint2 q = min(uint2(e & 0xFFFFu, e >> 16), C.half_size - 1u);
    float4 r0 = Req0.Load(int3(q, 0));
    float4 r1 = Req1.Load(int3(q, 0));
    bool hit, screen;
    uint inst;
    float pdf;
    float3 rv = reflected(r0, r1, sample_u(q, live_index(q)), pdf);
    float4 res = trace_ray(q, r0, rv, true, hit, screen, inst);
    Result[q] = C.debug == DEBUG_INSTANCES ? float4(instance_colour(inst), 1.0) : safe(res);
    Ray[q] = float4(rv, pdf);
    uint hits = WaveActiveCountBits(hit);
    uint scr = WaveActiveCountBits(screen && hit);
    uint miss = WaveActiveCountBits(!hit);
    if (WaveIsFirstLane())
    {
        uint ignored;
        Calib.InterlockedAdd(24, hits, ignored);
        Calib.InterlockedAdd(28, scr, ignored);
        Calib.InterlockedAdd(32, miss, ignored);
    }
}

// Resolve (as the SSR one, ssr.rs `cs_resolve`): 8x8 request texels per group plus a 2-texel
// apron in groupshared memory. Every traced neighbour ray is reused for this texel's lobe with
// the ratio estimator (Stachowiak 2015, FidelityFX SSSR): weight = this texel's GGX BRDF x
// cosine / the pdf the ray was drawn with, times the 5x5 tent and the tangent-plane weight;
// near-mirror texels (alpha < 0.0025): 3x3 tent and plane only. Untraced texels (over the
// budget) take their neighbours' rays. Glossy texels also suppress fireflies (as the SSR,
// E089): each hit clamped to FIREFLY_K x the mean luminance of the hits around. Writes
// (L c, c), a < 0 = no ray around.
groupshared float4 g_L[144];
groupshared float4 g_R[144];
groupshared float4 g_P[144];

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_resolve(uint3 gid : SV_GroupID, uint3 gt : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    int2 origin = int2(gid.xy) * 8 - 2;
    int2 last = int2(C.half_size) - 1;
    for (uint i = gi; i < 144u; i += 64u)
    {
        int2 k = clamp(origin + int2(i % 12u, i / 12u), int2(0, 0), last);
        g_L[i] = Result[k];
        g_R[i] = Ray[k];
        g_P[i] = Req0.Load(int3(k, 0));
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 q = gid.xy * 8u + gt.xy;
    if (any(q >= C.half_size)) return;
    int c = int(gt.y + 2u) * 12 + int(gt.x + 2u);
    float4 r0 = g_P[c];
    if (r0.w <= 0.0)
    {
        Resolved[q] = float4(0.0, 0.0, 0.0, -1.0);
        return;
    }
    float4 r1 = Req1.Load(int3(q, 0));
    float3 N = normalize(r1.xyz);
    float3 V = normalize(-r0.xyz);
    float alpha = r1.w;
    float NoV = saturate(dot(N, V)) + 1e-4;
    bool ratio = alpha >= 0.0025;
    float z = max(r0.z, 1e-3);
    bool fire = ratio && fireflies();
    float cap = 65000.0;
    if (fire)
    {
        float msum = 0.0, mn = 0.0;
        [unroll] for (int dy = -2; dy <= 2; dy++)
        {
            [unroll] for (int dx = -2; dx <= 2; dx++)
            {
                int k = c + dy * 12 + dx;
                float4 L = g_L[k];
                float4 p = g_P[k];
                float plane = abs(dot(N, p.xyz - r0.xyz)) / z;
                if (p.w > 0.0 && L.a > 1e-3 && plane * 32.0 < 1.0)
                {
                    msum += luminance(L.rgb / L.a);
                    mn += 1.0;
                }
            }
        }
        cap = C.firefly_k * msum / max(mn, 1.0) + 1e-3;
    }
    float4 acc = 0.0;
    float wsum = 0.0;
    [unroll] for (int dy = -2; dy <= 2; dy++)
    {
        [unroll] for (int dx = -2; dx <= 2; dx++)
        {
            bool outer = abs(dx) == 2 || abs(dy) == 2;
            int k = c + dy * 12 + dx;
            float4 L = g_L[k];
            float4 p = g_P[k];
            if ((outer && !ratio) || p.w <= 0.0 || L.a < 0.0) continue;
            float tent = (abs(dx) == 2 ? 0.5 : 1.0) * (abs(dy) == 2 ? 0.5 : 1.0);
            float plane = abs(dot(N, p.xyz - r0.xyz)) / z;
            float w = tent * saturate(1.0 - plane * 32.0);
            if (ratio && w > 0.0)
            {
                float4 ray = g_R[k];
                float NoL = dot(N, ray.xyz);
                float NoH = saturate(dot(N, normalize(V + ray.xyz)));
                w = NoL <= 0.0 ? 0.0
                    : w * ggx_d(NoH, alpha) * smith_g2(NoV, NoL, alpha) / (4.0 * NoV * max(ray.w, 1e-3));
            }
            if (fire && L.a > 1e-6)
            {
                float y = luminance(L.rgb / L.a);
                if (y > cap) L.rgb *= cap / y;
            }
            acc += w * L;
            wsum += w;
        }
    }
    Resolved[q] = wsum > 1e-20 ? safe(acc / wsum) : float4(0.0, 0.0, 0.0, -1.0);
}

float3 ycocg(float3 c) { return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25))); }
float3 from_ycocg(float3 y) { return float3(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z); }

// History length a texel may reach (ssr.rs `MaxHistory`): long while the surface stands still,
// shortened by its own motion in pixels (the first-person weapon's idle sway).
float max_history(float rough, float moved)
{
    return max(lerp(24.0, 64.0, saturate(rough / 0.35)) / (1.0 + moved), 2.0);
}

// Temporal accumulation (as ssr.rs `cs_temporal`, per half texel): reprojects the previous
// accumulation with the engine motion vectors (bilinear taps passing a depth and world-normal
// test), clamps it to the YCoCg mean +- k sigma of this frame's resolved 3x3 (sigma floored at
// 5 % of the luminance; k wide for still glossy surfaces, tight while moving) and its confidence
// to the current maximum, then blends 1 / history length. A requested texel without a ray this
// frame keeps its history; a texel without a request carries its reprojected history forward
// (unchanged length), so a reflection handed from SSR to the rays and back finds its history.
// Writes Hist (L c, c) and Meta (oct world normal, length, linear z of its first pixel).
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_temporal(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= C.half_size)) return;
    int2 q = int2(id.xy);
    int2 p = min(q * 2, int2(C.region) - 1);
    float z = lin_z(Depth.Load(int3(p + int2(C.origin), 0)));
    float4 r0 = Req0.Load(int3(q, 0));
    float4 r1 = Req1.Load(int3(q, 0));
    bool want = r0.w > 0.0;
    float3 Nw = want ? normalize(to_world_dir(r1.xyz)) : float3(0.0, 0.0, 1.0);
    float rough = sqrt(max(r1.w, 0.0));
    float2 mv = Motion.Load(int3(p + int2(C.origin), 0));
    float moved = length(mv * float2(C.region));
    float4 prev = 0.0;
    float len = 0.0;
    float3 nprev = 0.0;
    // Second moment of the luminance (YCoCg Y) over the history.
    float mom = 0.0;
    if ((C.flags & F_HISTORY) != 0u)
    {
        float2 b = float2(q) + 0.5 - mv * float2(C.region) * 0.5 - 0.5;
        int2 i0 = int2(floor(b));
        float2 f = b - float2(i0);
        float wsum = 0.0;
        [unroll] for (int t = 0; t < 4; t++)
        {
            int2 o = int2(t & 1, t >> 1);
            int2 k = i0 + o;
            if (any(k < 0) || any(k >= int2(C.half_size))) continue;
            float4 m = MetaPrev[k];
            if (!(m.z > 0.0)) continue;
            float3 nk = oct_dec(m.xy);
            float wb = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y);
            float wz = abs(m.w - z) <= 0.05 * z + 0.02 ? 1.0 : 0.0;
            float wn = want ? saturate((dot(nk, Nw) - 0.8) / 0.15) : 1.0;
            float w = wb * wz * wn;
            prev += w * HistPrev[k];
            mom += w * MomentPrev[k];
            len += w * m.z;
            nprev += w * nk;
            wsum += w;
        }
        if (wsum > 0.05)
        {
            prev /= wsum;
            len /= wsum;
            mom /= wsum;
        }
        else
        {
            prev = 0.0;
            len = 0.0;
            mom = 0.0;
        }
    }
    float4 acc = 0.0;
    float4 meta = 0.0;
    float moment = 0.0;
    if (!want)
    {
        if (len > 0.0)
        {
            acc = prev;
            meta = float4(oct_enc(normalize(nprev)), len, z);
            moment = mom;
        }
    }
    else
    {
        bool fire = fireflies();
        float4 cur = Resolved[q];
        if (len > 0.0 && cur.a >= 0.0)
        {
            float3 m1 = 0.0, m2 = 0.0;
            float n = 0.0, amax = 0.0;
            [unroll] for (int dy = -1; dy <= 1; dy++)
            {
                [unroll] for (int dx = -1; dx <= 1; dx++)
                {
                    float4 c = Resolved[clamp(q + int2(dx, dy), int2(0, 0), int2(C.half_size) - 1)];
                    amax = max(amax, c.a);
                    if (c.a > 1e-3)
                    {
                        float3 y = ycocg(c.rgb / c.a);
                        m1 += y;
                        m2 += y * y;
                        n += 1.0;
                    }
                }
            }
            float still = 1.0 - saturate(moved / 2.0);
            float glossy = saturate(rough / 0.35);
            float a = min(prev.a, lerp(amax * 1.25 + 0.02, 1.0, still * glossy));
            float3 Lp = prev.a > 1e-4 ? prev.rgb / prev.a : 0.0;
            if (n > 0.0)
            {
                float3 mu = m1 / n;
                float3 sigma = max(sqrt(max(m2 / n - mu * mu, 0.0)), 0.05 * abs(mu.x) + 1e-4);
                // Fireflies: the box also spans the luminance's own frame-to-frame deviation
                // over the history, so a frame whose rays missed a small light does not pull
                // the accumulated highlight down (and the next hit up).
                if (fire)
                {
                    float h = ycocg(Lp).x;
                    sigma.x = max(sigma.x, sqrt(max(mom - h * h, 0.0)));
                }
                float k = lerp(lerp(1.25, 2.0, glossy), lerp(3.0, 12.0, glossy), still);
                Lp = from_ycocg(clamp(ycocg(Lp), mu - k * sigma, mu + k * sigma));
            }
            prev = float4(max(Lp, 0.0) * a, a);
        }
        if (cur.a >= 0.0)
        {
            float yc = cur.a > 1e-6 ? ycocg(cur.rgb / cur.a).x : 0.0;
            moment = len > 0.0 ? mom : yc * yc;
            len = min(len + 1.0, max_history(rough, moved));
            acc = lerp(prev, cur, 1.0 / len);
            moment = lerp(moment, yc * yc, 1.0 / len);
        }
        else
        {
            acc = prev;
            moment = mom;
        }
        if (len > 0.0) meta = float4(oct_enc(Nw), len, z);
    }
    Hist[q] = safe(acc);
    Meta[q] = meta;
    Moment[q] = min(moment, 65000.0);
}

// Roughness-scaled bilateral blur of the accumulation (as ssr.rs `cs_blur`, in half texels):
// 3x3 taps at a stride growing with roughness, doubled while the history is short; plane and
// normal weights. Writes Final (L c, c); a < 0 where nothing is requested or accumulated.
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_blur(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= C.half_size)) return;
    int2 q = int2(id.xy);
    float4 m = Meta[q];
    float4 r1 = Req1.Load(int3(q, 0));
    if (Req0.Load(int3(q, 0)).w <= 0.0 || !(m.z > 0.0))
    {
        Final[q] = float4(0.0, 0.0, 0.0, -1.0);
        return;
    }
    float4 acc = Hist[q];
    float rough = sqrt(max(r1.w, 0.0));
    [branch] if (rough > 0.08)
    {
        float z = m.w;
        float3 nw = oct_dec(m.xy);
        float3 nv = to_view_dir(nw);
        float3 P = view_pos(float2(q) * 2.0 + 1.0, z);
        int stride = clamp(int(rough * 8.0) * (m.z < 4.0 ? 2 : 1), 1, 6);
        float wsum = 1.0;
        [unroll] for (int dy = -1; dy <= 1; dy++)
        {
            [unroll] for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0) continue;
                int2 k = clamp(q + int2(dx, dy) * stride, int2(0, 0), int2(C.half_size) - 1);
                float4 mk = Meta[k];
                if (!(mk.z > 0.0) || Req0.Load(int3(k, 0)).w <= 0.0) continue;
                float3 Pk = view_pos(float2(k) * 2.0 + 1.0, mk.w);
                float plane = abs(dot(nv, Pk - P)) / max(z, 1e-3);
                float w = (dx == 0 || dy == 0 ? 0.6 : 0.36) * saturate(1.0 - plane * 50.0)
                    * pow(saturate(dot(oct_dec(mk.xy), nw)), 16.0);
                acc += w * Hist[k];
                wsum += w;
            }
        }
        acc /= wsum;
    }
    Final[q] = safe(acc);
}

// Photo accumulation (no list, no budget): every request texel traces C.max_rays GGX rays,
// samples C.frame .. C.frame + max_rays - 1, and adds (sum L c, sum c) into the accumulation
// target at u0 (RGBA32F; F_CLEAR starts it from 0).
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_photo(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= C.half_size)) return;
    uint2 q = id.xy;
    float4 sum = (C.flags & F_CLEAR) != 0u ? 0.0 : Result[q];
    float4 r0 = Req0.Load(int3(q, 0));
    if (r0.w > 0.0)
    {
        float4 r1 = Req1.Load(int3(q, 0));
        uint spp = min(C.max_rays, 1024u);
        for (uint s = 0; s < spp; s++)
        {
            bool hit, screen;
            uint inst;
            float pdf;
            sum += trace_ray(q, r0, reflected(r0, r1, sample_u(q, C.frame + s), pdf), false, hit, screen, inst);
        }
    }
    Result[q] = safe(sum);
}

// Full-resolution blend into FSR's colour (region pixels): 3x3 half texels of Final around the
// pixel, tent x depth similarity, premultiplied confidence (misses count with c = 0, texels
// without anything are skipped). Debug views replace the colour.
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void rt_composite(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= C.region)) return;
    int2 cp = int2(id.xy + C.origin);
    float4 wt = RtWeight.Load(int3(id.xy, 0));
    float w = wt.a;
    float4 col = Color[cp];
    int2 c0 = int2(id.xy >> 1);
    int2 cc = min(c0, int2(C.half_size) - 1);
    if (C.debug == DEBUG_RAYS)
    {
        float req = Req0.Load(int3(cc, 0)).w > 0.0 ? 1.0 : 0.0;
        float l = luminance(col.rgb);
        Color[cp] = safe(float4(l * 0.2 + req * 0.6, l * 0.2 + saturate(w) * 0.8, l * 0.2, col.a));
        return;
    }
    if (C.debug == DEBUG_INSTANCES)
    {
        // What each traced ray hit, by instance type (the record skips the denoiser: Final =
        // this frame's rays); the frame dimmed where nothing was traced.
        float l = luminance(col.rgb) * 0.15;
        float4 r = Final[cc];
        bool traced = Req0.Load(int3(cc, 0)).w > 0.0 && r.a >= 0.0;
        Color[cp] = safe(float4(traced ? r.rgb : l.xxx, col.a));
        return;
    }
    if (C.debug == DEBUG_HISTORY)
    {
        // Grey = history length / 64 where rays are requested (red: none yet); the frame dimmed
        // elsewhere.
        float l = luminance(col.rgb) * 0.15;
        float4 o = float4(l, l, l, col.a);
        if (Req0.Load(int3(cc, 0)).w > 0.0)
        {
            float h = (C.flags & F_DENOISE) != 0u ? Meta[cc].z : 0.0;
            o.rgb = h > 0.0 ? saturate(h / 64.0).xxx : float3(0.6, 0.0, 0.0);
        }
        Color[cp] = safe(o);
        return;
    }
    if (!(w > 1.0 / 1024.0))
    {
        if (C.debug != 0u) Color[cp] = safe(float4(col.rgb * 0.15, col.a));
        return;
    }
    float zp = lin_z(Depth.Load(int3(cp, 0)));
    float4 acc = 0.0;
    float wsum = 0.0;
    [unroll] for (int dy = -1; dy <= 1; dy++)
    {
        [unroll] for (int dx = -1; dx <= 1; dx++)
        {
            int2 k = clamp(c0 + int2(dx, dy), int2(0, 0), int2(C.half_size) - 1);
            float4 r = Final[k];
            float4 r0 = Req0.Load(int3(k, 0));
            if (r0.w <= 0.0 || r.a < 0.0) continue;
            if ((C.flags & F_PHOTO) != 0u) r /= float(max(C.max_rays, 1u));
            float tent = (dx == 0 ? 1.0 : 0.5) * (dy == 0 ? 1.0 : 0.5);
            float sim = saturate(1.0 - abs(r0.z - zp) / max(zp, 1e-3) * 32.0);
            float ww = tent * sim;
            acc += ww * r;
            wsum += ww;
        }
    }
    float conf = wsum > 0.0 ? saturate(acc.a / wsum) : 0.0;
    float3 L = acc.a > 1e-4 ? acc.rgb / acc.a : 0.0;
    if (C.debug == DEBUG_HITS)
    {
        Color[cp] = safe(float4(conf > 0.0 ? L * conf : float3(0.0, 0.0, 0.25), col.a));
        return;
    }
    float3 a = saturate(wt.rgb * conf);
    if (C.debug == DEBUG_RT)
    {
        Color[cp] = safe(float4(L * a, col.a));
        return;
    }
    if (any(a > 0.0)) Color[cp] = safe(float4(lerp(col.rgb, L, a), col.a));
}
