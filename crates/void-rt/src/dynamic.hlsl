// void-rt dynamic objects (docs/raytracing.md section 14): GPU skinning of characters for
// their BLAS refit. Every resource is a root descriptor (raw buffers). Root descriptors are
// not bounds-checked: vertex, triangle and joint indices are clamped to the counts passed in
// the constants, whose buffer sizes the host checks on the CPU (`dynamic.rs`).
//
// skin_vertices:  bind-pose position / normal / tangent x the weighted joint palette ->
//                 object-space positions (BLAS vertex buffer) and the skinned normal frame.
// skin_triangles: writes the skinned normal / tangent of every triangle corner into the
//                 character's photo records (words 3 + 5k, 4 + 5k) and the octahedral normal
//                 into its FC6 primitives (low 16 bits of word 2k + 1), so shading at hits
//                 follows the pose.

#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"

struct Consts {
    uint count;        // vertices (skin_vertices) or triangles (skin_triangles)
    uint joint_count;  // palette entries
    uint width;        // threads per dispatch row
    uint vertex_count; // vertex index clamp (skin_triangles)
    uint prims_on;     // FC6 primitives bound at u3 (skin_triangles)
    uint pad0; uint pad1; uint pad2;
};
ConstantBuffer<Consts> C : register(b0);

ByteAddressBuffer Bind : register(t0);       // float3 bind position per vertex
ByteAddressBuffer BindFrame : register(t1);  // uint2 per vertex: snorm8x4 normal, tangent
ByteAddressBuffer Influences : register(t2); // uint4 per vertex: joints 0-3, 4-7 (u8),
                                             // weights 0-3, 4-7 (unorm8)
ByteAddressBuffer Palette : register(t3);    // row-major 3x4 per joint (48 B): bone -> object
ByteAddressBuffer TriVerts : register(t4);   // uint3 per triangle: vertex of each corner

RWByteAddressBuffer OutPos : register(u0);   // float3 per vertex
RWByteAddressBuffer OutFrame : register(u1); // uint2 per vertex (written by skin_vertices)
RWByteAddressBuffer Records : register(u2);  // 64 B per triangle, from the character's first
RWByteAddressBuffer Prims : register(u3);    // 24 B per triangle, from the character's first

float3 snorm8x3(uint v)
{
    int3 i = int3(int(v << 24) >> 24, int(v << 16) >> 24, int(v << 8) >> 24);
    return max(float3(i) / 127.0, -1.0);
}

uint pack_snorm8x3(float3 n, uint w)
{
    int3 i = int3(round(clamp(n, -1.0, 1.0) * 127.0));
    return (uint(i.x) & 0xff) | (uint(i.y) & 0xff) << 8 | (uint(i.z) & 0xff) << 16 | (w & 0xff000000u);
}

float3 normalize_or(float3 v, float3 fallback)
{
    float l = dot(v, v);
    return l > 1e-12 ? v * rsqrt(l) : fallback;
}

// Same encoding as `scene::oct_encode` (two unorm8).
uint oct16(float3 n)
{
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    if (l1 <= 0.0)
        return 128u | 128u << 8;
    float2 p = n.xy / l1;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * select(p >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    uint2 q = uint2(round(saturate(p * 0.5 + 0.5) * 255.0));
    return q.x | q.y << 8;
}

float3x4 joint_matrix(uint j)
{
    uint o = min(j, C.joint_count - 1u) * 48u;
    return float3x4(asfloat(Palette.Load4(o)), asfloat(Palette.Load4(o + 16u)), asfloat(Palette.Load4(o + 32u)));
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void skin_vertices(uint3 id : SV_DispatchThreadID)
{
    uint v = id.y * C.width + id.x;
    if (id.x >= C.width || v >= C.count || C.joint_count == 0u)
        return;
    uint4 inf = Influences.Load4(v * 16u);
    float3x4 m = (float3x4)0;
    float wsum = 0.0;
    [unroll]
    for (uint k = 0u; k < 8u; k++) {
        uint shift = 8u * (k & 3u);
        uint j = ((k < 4u ? inf.x : inf.y) >> shift) & 0xffu;
        float w = float(((k < 4u ? inf.z : inf.w) >> shift) & 0xffu) / 255.0;
        if (w > 0.0) {
            m += w * joint_matrix(j);
            wsum += w;
        }
    }
    float3 p = asfloat(Bind.Load3(v * 12u));
    uint2 f = BindFrame.Load2(v * 8u);
    float3 q = p;
    uint2 g = f;
    if (wsum > 0.0) {
        m /= wsum;
        q = mul(m, float4(p, 1.0));
        float3x3 r = (float3x3)m;
        float3 n = snorm8x3(f.x);
        float3 t = snorm8x3(f.y);
        g = uint2(pack_snorm8x3(normalize_or(mul(r, n), n), f.x),
                  pack_snorm8x3(normalize_or(mul(r, t), t), f.y));
    }
    // A broken palette (torn read, garbage) must not put NaN or far-away vertices into the
    // BLAS: keep the bind pose for such vertices.
    if (any(isnan(q)) || any(abs(q - p) > 64.0)) {
        q = p;
        g = f;
    }
    OutPos.Store3(v * 12u, asuint(q));
    OutFrame.Store2(v * 8u, g);
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void skin_triangles(uint3 id : SV_DispatchThreadID)
{
    uint t = id.y * C.width + id.x;
    if (id.x >= C.width || t >= C.count || C.vertex_count == 0u)
        return;
    uint3 vs = min(TriVerts.Load3(t * 12u), C.vertex_count - 1u);
    [unroll]
    for (uint k = 0u; k < 3u; k++) {
        uint2 f = OutFrame.Load2(vs[k] * 8u);
        Records.Store2(t * 64u + (3u + 5u * k) * 4u, f);
        if (C.prims_on != 0u) {
            uint po = t * 24u + (2u * k + 1u) * 4u;
            uint old = Prims.Load(po);
            Prims.Store(po, (old & 0xffff0000u) | oct16(normalize_or(snorm8x3(f.x), float3(0.0, 0.0, 1.0))));
        }
    }
}
