// void-rt spike kernels: DXR 1.1 inline RayQuery in compute shaders (no RT pipeline, no hit
// shaders). Shading data come from the unwrapped primitive buffer (Far Cry 6 layout, see
// scene.rs): 24 bytes per triangle, located by PrimOffsets[InstanceID + GeometryIndex] +
// PrimitiveIndex. Every resource is a root descriptor (buffers only). Root descriptors are
// not bounds-checked: geometry/primitive indices are clamped to the table sizes passed in
// the constants, pixel indices stay below width * height (buffer sizes checked on the CPU).

#define RS "RootConstants(num32BitConstants=36, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"

struct Consts {
    float3 org;     uint width;
    float3 fwd;     uint height;
    float3 right;   float tan_x;
    float3 up;      float tan_y;
    float3 sun_dir; float near_z;
    float3 sun_rad; uint frame;
    float3 sky_rad; float roughness;
    uint flags;     uint ray_flags; float t_max; uint bounces;
    uint geom_count; uint prim_count; uint pad0; uint pad1;
};
ConstantBuffer<Consts> C : register(b0);

RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<uint> PrimOffsets : register(t1);
ByteAddressBuffer Prims : register(t2);
StructuredBuffer<float> DepthIn : register(t3);   // game convention: near / view z, 0 = sky
StructuredBuffer<float> SsrIn : register(t4);     // ssr_ref output: hit t, < 0 miss

RWStructuredBuffer<float> OutA : register(u0);
RWStructuredBuffer<uint> OutId : register(u1);
RWStructuredBuffer<float4> OutColor : register(u2);
RWStructuredBuffer<float4> OutAux : register(u3);

// flags
#define F_SHADOW      1u   // sun shadow ray at shaded hits
#define F_FLOORS      2u   // reflect: only upward-facing pixels (N.z > 0.7)
#define F_GGX         4u   // reflect: GGX VNDF sample instead of mirror
#define F_JITTER      8u   // primary/gi: sub-pixel jitter
#define F_MASK_SSR   16u   // reflect: skip pixels where SsrIn hit (hybrid, SSR first)
#define F_MASK_DEPTH 32u   // reflect: skip pixels with high screen-space confidence (depth heuristic)

#define ID_MISS    0xffffffffu
#define ID_SKIPPED 0xfffffffeu

static const float PI = 3.14159265;

uint pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { s = pcg(s); return float(s) * (1.0 / 4294967296.0); }
uint seed_of(uint2 p, uint frame) { return pcg(p.x + pcg(p.y + pcg(frame))); }
float rnd_hash(uint2 p, uint frame) { uint s = seed_of(p, frame ^ 0x9e3779b9u); return rnd(s); }

float3 oct_decode(uint2 e) {
    float2 f = float2(e) / 255.0 * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0) {
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0 ? 1.0 : -1.0, n.y >= 0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

struct Surf {
    float3 pos;
    float3 n;          // shading normal, facing the ray origin
    float3 albedo;
    float smoothness;
    float metallic;
};

// Interpolates the primitive's packed vertex data at the committed hit.
Surf surface_at(uint geom_base, uint geom, uint prim, float2 bary, float3x4 w2o, RayDesc r, float t) {
    uint g = min(geom_base + geom, C.geom_count - 1);
    uint p = min(PrimOffsets[g] + prim, C.prim_count - 1);
    uint2 v0 = Prims.Load2(p * 24);
    uint2 v1 = Prims.Load2(p * 24 + 8);
    uint2 v2 = Prims.Load2(p * 24 + 16);
    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float4 a = (float4(v0.x & 255, (v0.x >> 8) & 255, (v0.x >> 16) & 255, v0.x >> 24) * w.x
              + float4(v1.x & 255, (v1.x >> 8) & 255, (v1.x >> 16) & 255, v1.x >> 24) * w.y
              + float4(v2.x & 255, (v2.x >> 8) & 255, (v2.x >> 16) & 255, v2.x >> 24) * w.z) / 255.0;
    float3 n_obj = oct_decode(uint2(v0.y & 255, (v0.y >> 8) & 255)) * w.x
                 + oct_decode(uint2(v1.y & 255, (v1.y >> 8) & 255)) * w.y
                 + oct_decode(uint2(v2.y & 255, (v2.y >> 8) & 255)) * w.z;
    float metal = (float((v0.y >> 16) & 255) * w.x + float((v1.y >> 16) & 255) * w.y + float((v2.y >> 16) & 255) * w.z) / 255.0;
    // Normals transform with the inverse transpose: n_world = transpose(L_w2o) * n_obj.
    float3 n = normalize(mul(n_obj, (float3x3)w2o));
    if (dot(n, r.Direction) > 0) n = -n;
    Surf s;
    s.pos = r.Origin + r.Direction * t;
    s.n = n;
    s.albedo = a.rgb;
    s.smoothness = a.a;
    s.metallic = metal;
    return s;
}

bool trace_closest(RayDesc r, uint ray_flags, out Surf s, out uint inst, out float t) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(Scene, ray_flags, 0xFF, r);
    while (q.Proceed()) {}
    s = (Surf)0;
    inst = ID_MISS;
    t = -1;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return false;
    t = q.CommittedRayT();
    inst = q.CommittedInstanceIndex();
    s = surface_at(q.CommittedInstanceID(), q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex(),
                   q.CommittedTriangleBarycentrics(), q.CommittedWorldToObject3x4(), r, t);
    return true;
}

bool visible(float3 p, float3 dir, float t_max) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
    RayDesc r;
    r.Origin = p;
    r.Direction = dir;
    r.TMin = 0;
    r.TMax = t_max;
    q.TraceRayInline(Scene, 0, 0xFF, r);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

float3 offset_origin(Surf s, float dist) {
    return s.pos + s.n * max(0.003, 1e-4 * dist);
}

float3 sky(float3 d) {
    return C.sky_rad * lerp(0.35, 1.0, saturate(d.z * 2.0 + 0.5));
}

// Hit shading from the primitive data only (no material textures): sun (optionally
// shadowed) + hemispheric sky ambient; dielectric/metal split by the stored metallic.
float3 shade(Surf s, float dist) {
    float nl = saturate(dot(s.n, C.sun_dir));
    float vis = 1.0;
    if ((C.flags & F_SHADOW) != 0 && nl > 0) vis = visible(offset_origin(s, dist), C.sun_dir, 1e4) ? 1.0 : 0.0;
    float3 diffuse = s.albedo * (1.0 - s.metallic);
    float3 amb = C.sky_rad * lerp(0.4, 1.0, s.n.z * 0.5 + 0.5);
    return diffuse * (C.sun_rad * nl * vis / PI + amb);
}

float3 cam_dir(float2 pix) {
    float2 ndc = float2(2.0 * pix.x / C.width - 1.0, 1.0 - 2.0 * pix.y / C.height);
    return C.fwd + ndc.x * C.tan_x * C.right + ndc.y * C.tan_y * C.up;
}

// ---------------------------------------------------------------------------------------
// primary: camera rays. OutA = game-convention depth (near / z, 0 = miss), OutId = TLAS
// instance index, OutColor = (shaded rgb, hit distance or -1).
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void primary(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C.width || id.y >= C.height) return;
    uint i = id.y * C.width + id.x;
    uint seed = seed_of(id.xy, C.frame);
    float2 jitter = (C.flags & F_JITTER) != 0 ? float2(rnd(seed), rnd(seed)) : 0.5;
    float3 d = cam_dir(id.xy + jitter);
    RayDesc r;
    r.Origin = C.org;
    r.Direction = normalize(d);
    r.TMin = C.near_z;
    r.TMax = C.t_max;
    Surf s;
    uint inst;
    float t;
    if (trace_closest(r, C.ray_flags, s, inst, t)) {
        float z = t / length(d);
        OutA[i] = C.near_z / z;
        OutId[i] = inst;
        OutColor[i] = float4(shade(s, t), t);
    } else {
        OutA[i] = 0;
        OutId[i] = ID_MISS;
        OutColor[i] = float4(sky(r.Direction), -1);
    }
}

// ---------------------------------------------------------------------------------------
// Depth-buffer surface reconstruction shared by reflect and ssr_ref. Root descriptors have
// no bounds checking: every fetch is clamped to the image.
int2 clamp_px(int2 p) { return clamp(p, int2(0, 0), int2(C.width - 1, C.height - 1)); }
float3 depth_pos(int2 p) {
    p = clamp_px(p);
    float d = DepthIn[p.y * C.width + p.x];
    return C.org + cam_dir(p + 0.5) * (C.near_z / max(d, 1e-9));
}
float view_z(int2 p) {
    p = clamp_px(p);
    float d = DepthIn[p.y * C.width + p.x];
    return d > 0 ? C.near_z / d : 1e9;
}

// Position, depth normal (closest neighbour per axis) and reflection direction of a pixel.
bool reflect_setup(uint2 id, out float3 P, out float3 N, out float3 R, out float z, out float disc) {
    P = 0; N = 0; R = 0; z = 0; disc = 0;
    float d = DepthIn[id.y * C.width + id.x];
    if (d <= 0) return false;
    z = C.near_z / d;
    int2 p = int2(id.xy);
    P = depth_pos(p);
    int2 l = int2(max(p.x - 1, 0), p.y), rr = int2(min(p.x + 1, (int)C.width - 1), p.y);
    int2 u = int2(p.x, max(p.y - 1, 0)), dd = int2(p.x, min(p.y + 1, (int)C.height - 1));
    float zl = view_z(l), zr = view_z(rr), zu = view_z(u), zd = view_z(dd);
    float3 dx = abs(zr - z) < abs(zl - z) ? depth_pos(rr) - P : P - depth_pos(l);
    float3 dy = abs(zd - z) < abs(zu - z) ? depth_pos(dd) - P : P - depth_pos(u);
    N = normalize(cross(dx, dy));
    float3 V = normalize(C.org - P);
    if (dot(N, V) < 0) N = -N;
    disc = max(max(abs(zl - z), abs(zr - z)), max(abs(zu - z), abs(zd - z))) / z;
    R = reflect(-V, N);
    if ((C.flags & F_GGX) != 0 && C.roughness > 0) {
        // GGX VNDF sampling (Heitz 2018) around N; the seed depends on pixel and frame only,
        // so reflect and ssr_ref draw the same direction.
        uint seed = seed_of(id.xy, C.frame);
        float a = C.roughness * C.roughness;
        float3 t1 = abs(N.z) < 0.999 ? normalize(cross(float3(0, 0, 1), N)) : float3(1, 0, 0);
        float3 t2 = cross(N, t1);
        float3 Ve = float3(dot(V, t1), dot(V, t2), dot(V, N));
        float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
        float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
        float3 T2 = cross(Vh, T1);
        float u1 = rnd(seed), u2 = rnd(seed);
        float rr2 = sqrt(u1);
        float phi = 2.0 * PI * u2;
        float s1 = rr2 * cos(phi);
        float s2 = rr2 * sin(phi);
        float s = 0.5 * (1.0 + Vh.z);
        s2 = (1.0 - s) * sqrt(1.0 - s1 * s1) + s * s2;
        float3 Nh = s1 * T1 + s2 * T2 + sqrt(max(0.0, 1.0 - s1 * s1 - s2 * s2)) * Vh;
        float3 He = normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
        float3 H = He.x * t1 + He.y * t2 + He.z * N;
        float3 Rg = reflect(-V, H);
        if (dot(Rg, N) > 0) R = Rg;
    }
    return true;
}

// Ray start for a depth-buffer surface. The game's surface and our decoded geometry differ
// by up to centimetres (LOD duplicates in the .bwm, displacement-free meshes, E070), so the
// origin is pushed off the depth-normal plane and nearby hits are skipped, both growing
// with view depth (depth-buffer precision and geometry mismatch scale with distance).
float3 refl_origin(float3 P, float3 N, float z) { return P + N * max(0.02, 0.003 * z); }
float refl_tmin(float z) { return max(0.05, 0.005 * z); }

// Projects a world point to pixel coordinates (xy) and view z.
float3 project(float3 X) {
    float3 v = X - C.org;
    float z = dot(v, C.fwd);
    float2 ndc = float2(dot(v, C.right) / (z * C.tan_x), dot(v, C.up) / (z * C.tan_y));
    return float3((ndc.x + 1) * 0.5 * C.width, (1 - ndc.y) * 0.5 * C.height, z);
}

// High screen-space confidence (FC6 "reflecting with confidence: SSLR"): small local depth
// discontinuity and the reflected ray stays on screen and in front of the camera for
// `probe` metres.
bool ss_confident(float3 P, float3 R, float z, float disc) {
    if (disc > 0.1) return false;
    float probe = clamp(0.5 * z, 2.0, 20.0);
    float3 e = project(P + R * probe);
    return e.z > C.near_z * 10 && e.x >= 0 && e.y >= 0 && e.x < C.width && e.y < C.height;
}

// ---------------------------------------------------------------------------------------
// reflect: one reflection ray per pixel from the depth buffer. OutA = hit distance (-1 miss,
// 0 no surface/gated, -2 skipped by a hybrid mask), OutId = instance, OutColor = shaded hit.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void reflect_rays(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C.width || id.y >= C.height) return;
    uint i = id.y * C.width + id.x;
    float3 P, N, R;
    float z, disc;
    if (!reflect_setup(id.xy, P, N, R, z, disc) || ((C.flags & F_FLOORS) != 0 && N.z < 0.7)) {
        OutA[i] = 0;
        OutId[i] = ID_SKIPPED;
        OutColor[i] = 0;
        return;
    }
    if (((C.flags & F_MASK_SSR) != 0 && SsrIn[i] > 0) ||
        ((C.flags & F_MASK_DEPTH) != 0 && ss_confident(P, R, z, disc))) {
        OutA[i] = -2;
        OutId[i] = ID_SKIPPED;
        OutColor[i] = 0;
        return;
    }
    RayDesc r;
    r.Origin = refl_origin(P, N, z);
    r.Direction = R;
    r.TMin = refl_tmin(z);
    r.TMax = C.t_max;
    Surf s;
    uint inst;
    float t;
    if (trace_closest(r, 0, s, inst, t)) {
        OutA[i] = t;
        OutId[i] = inst;
        OutColor[i] = float4(shade(s, t), 1);
    } else {
        OutA[i] = -1;
        OutId[i] = ID_MISS;
        OutColor[i] = float4(sky(R), 0);
    }
}

// ---------------------------------------------------------------------------------------
// ssr_ref: screen-space reference for the same rays: dense linear trace (one step per 2
// pixels of the on-screen segment, at most 1024) + 8 refine steps, perspective-correct depth,
// thickness test. A reference for agreement, not a real-time SSR (FC6 uses 64 + 8 steps).
// OutA = hit distance along R (< 0 miss).
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void ssr_ref(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C.width || id.y >= C.height) return;
    uint i = id.y * C.width + id.x;
    float3 P, N, R;
    float z, disc;
    if (!reflect_setup(id.xy, P, N, R, z, disc) || ((C.flags & F_FLOORS) != 0 && N.z < 0.7)) {
        OutA[i] = 0;
        return;
    }
    float3 O = refl_origin(P, N, z);
    // Clip the segment to stay in front of the near plane and within C.t_max.
    float rz = dot(R, C.fwd);
    float oz = dot(O - C.org, C.fwd);
    float len = C.t_max;
    if (rz < 0) len = min(len, (oz - 0.05) / -rz);
    float3 s0 = project(O);
    float3 s1 = project(O + R * len);
    // Clip to the screen rectangle in screen space (linear in the projected segment).
    float2 dir = s1.xy - s0.xy;
    float a = 1.0;
    if (s1.x < 0) a = min(a, (0 - s0.x) / dir.x);
    if (s1.x > C.width) a = min(a, (C.width - s0.x) / dir.x);
    if (s1.y < 0) a = min(a, (0 - s0.y) / dir.y);
    if (s1.y > C.height) a = min(a, (C.height - s0.y) / dir.y);
    a = saturate(a);
    float k0 = 1.0 / s0.z, k1 = lerp(k0, 1.0 / s1.z, a);
    float2 e = lerp(s0.xy, s1.xy, a);
    uint STEPS = (uint)clamp(ceil(length(e - s0.xy) * 0.5), 16.0, 1024.0);
    float prev = 0;
    float hit = -1;
    float jitter = rnd_hash(id.xy, C.frame);
    [loop] for (uint j = 1; j <= STEPS; j++) {
        float f = (j - 1 + jitter) / STEPS;
        float2 sp = lerp(s0.xy, e, f);
        if (length(sp - s0.xy) < 2.0) { prev = f; continue; }
        float zr = 1.0 / lerp(k0, k1, f);
        int2 px = int2(sp);
        float zb = view_z(px);
        if (zr > zb && zr < zb + max(0.15, 0.02 * zb)) {
            // Refine between prev and f (8 bisection steps).
            float lo = prev, hi = f;
            [unroll] for (uint b = 0; b < 8; b++) {
                float m = 0.5 * (lo + hi);
                float2 mp = lerp(s0.xy, e, m);
                float zm = 1.0 / lerp(k0, k1, m);
                if (zm > view_z(int2(mp))) hi = m; else lo = m;
            }
            int2 hp = int2(lerp(s0.xy, e, hi));
            float3 X = depth_pos(hp);
            hit = dot(X - O, R);
            break;
        }
        prev = f;
    }
    OutA[i] = hit;
}

// ---------------------------------------------------------------------------------------
// gi: one path sample per pixel per frame (diffuse, sun + sky), accumulated in OutAux.
float3 cosine_dir(float3 n, inout uint seed) {
    float u1 = rnd(seed), u2 = rnd(seed);
    float r = sqrt(u1), phi = 2 * PI * u2;
    float3 t1 = abs(n.z) < 0.999 ? normalize(cross(float3(0, 0, 1), n)) : float3(1, 0, 0);
    float3 t2 = cross(n, t1);
    return normalize(t1 * r * cos(phi) + t2 * r * sin(phi) + n * sqrt(max(0.0, 1 - u1)));
}

float3 sun_sample(inout uint seed) {
    // Soft shadows: uniform direction in a cone of half-angle 0.5 degrees around the sun.
    const float cos_max = 0.99996192;
    float u1 = rnd(seed), u2 = rnd(seed);
    float c = lerp(1.0, cos_max, u1), sn = sqrt(max(0.0, 1 - c * c)), phi = 2 * PI * u2;
    float3 n = C.sun_dir;
    float3 t1 = abs(n.z) < 0.999 ? normalize(cross(float3(0, 0, 1), n)) : float3(1, 0, 0);
    float3 t2 = cross(n, t1);
    return normalize(t1 * sn * cos(phi) + t2 * sn * sin(phi) + n * c);
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void gi(uint3 id : SV_DispatchThreadID) {
    if (id.x >= C.width || id.y >= C.height) return;
    uint i = id.y * C.width + id.x;
    uint seed = seed_of(id.xy, C.frame);
    float2 jitter = float2(rnd(seed), rnd(seed));
    RayDesc r;
    r.Origin = C.org;
    r.Direction = normalize(cam_dir(id.xy + jitter));
    r.TMin = C.near_z;
    r.TMax = C.t_max;
    float3 L = 0, thr = 1;
    for (uint b = 0; b <= C.bounces; b++) {
        Surf s;
        uint inst;
        float t;
        if (!trace_closest(r, b == 0 ? C.ray_flags : 0, s, inst, t)) {
            L += thr * sky(r.Direction);
            break;
        }
        float3 alb = s.albedo * (1.0 - s.metallic);
        float3 o = offset_origin(s, t);
        float3 l = sun_sample(seed);
        float nl = dot(s.n, l);
        if (nl > 0 && visible(o, l, 1e4)) L += thr * alb / PI * C.sun_rad * nl;
        thr *= alb;
        r.Origin = o;
        r.Direction = cosine_dir(s.n, seed);
        r.TMin = 0;
        r.TMax = 1e4;
    }
    OutAux[i] += float4(L, 1);
}
