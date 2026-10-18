// void-rt photo mode: progressive path tracer over the decoded static scene (DXR 1.1 inline
// RayQuery in compute shaders) plus an edge-aware a-trous denoiser.
//
// Data (scene.rs photo layout): per geometry (first primitive, material); per triangle a
// 64-byte record (LOD bias + three raw 20-byte vertex attribute blocks: uv0 f16x2, uv1 f16x2,
// normal snorm8x4, tangent snorm8x4 (w = bitangent sign), colour unorm8x4); a material table
// in the removed R3 renderer's `Mat` layout (arkDefault port: 3 layers, masks, colour
// constants, gloss/metallic ranges, BC5 normal maps); bindless textures in space1.
//
// Root descriptors and the texture table are not bounds-checked: every geometry, primitive,
// material, texture, light, grid-cell and list index is clamped to the counts in the
// parameter block (sizes of the bound buffers, checked on the CPU), and every pixel index is
// below width x height of the dispatch (per-pixel buffer sizes checked on the CPU).

#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), SRV(t5), SRV(t6), SRV(t7), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4), DescriptorTable(SRV(t0, space=1, numDescriptors=unbounded, flags=DESCRIPTORS_VOLATILE)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_WRAP, addressV=TEXTURE_ADDRESS_WRAP, addressW=TEXTURE_ADDRESS_WRAP)"

// Per-dispatch constants. `width x height` is always the whole image (bounds of every
// per-pixel index); a tiled dispatch covers rows `y0..` (group rows added to `y0`).
struct Pc {
    uint width;
    uint height;
    uint frame;    // sample index (trace) / accumulated samples (denoiser, resolve)
    uint step;     // a-trous step in pixels
    uint flags;
    uint y0;       // first row of the tile (photo_trace, photo_depth)
    uint pad1, pad2; // hybrid terms (photo_sun_vis, photo_ao): asfloat projection jitter x, y
};
ConstantBuffer<Pc> K : register(b0);

// Scene parameters (one element; layout = photo.rs `Params::words`).
struct Params {
    float3 org;         uint max_bounces;
    float3 fwd;         float near_z;
    float3 right;       float tan_x;
    float3 up;          float tan_y;
    float3 sun_dir;     float sun_cos;       // towards the sun; cos of the cone half-angle
    float3 sun_irr;     float pixel_spread;  // irradiance normal to the sun; cone angle / pixel
    float3 sky_zenith;  float sky_scale;
    float3 sky_horizon; float clamp_indirect;
    uint geom_count; uint prim_count; uint mat_count; uint tex_count;
    uint light_count; uint cell_count; uint index_count; uint ris_m;
    float2 grid_origin; float grid_cell; float ao_radius;   // ao_radius: photo_ao only
    uint2 grid_dims; uint rr_start; float depth_step;      // depth_step: hybrid terms only
};

struct Mat {
    float4 scale_bias[3]; float4 mask_scale_bias[3]; float4 color_constant[3]; float4 rescale[3];
    uint4 modes[3]; uint4 reuse[3]; uint4 mask_source; float4 ranges; uint4 flags; uint4 tex[4];
    float4 lit0;   // specularTintDielectric.xyz, sheen
    float4 lit1;   // diffuseAlbedoMetallic.xyz, sheenImportance
    uint4 extra;   // x: photo flags, yzw: mean layer-0 albedo (asfloat)
};

struct Light {
    float4 a;   // pos, falloff
    float4 b;   // colour, cutoff
    float4 c;   // dir, decay
    float4 d;   // spot params xyz, w = asfloat(kind)
    float4 e;   // specAttn, range
};

RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<uint2> GeomInfo : register(t1);
ByteAddressBuffer Records : register(t2);
StructuredBuffer<Mat> Mats : register(t3);
StructuredBuffer<Light> Lights : register(t4);
StructuredBuffer<uint2> Cells : register(t5);
StructuredBuffer<uint> LightIdx : register(t6);
StructuredBuffer<Params> Prm : register(t7);

RWStructuredBuffer<float4> AccRad : register(u0);   // sum rgb, sum luminance^2
RWStructuredBuffer<float4> AccAlb : register(u1);   // sum first-hit albedo, sum sun visibility
RWStructuredBuffer<float4> AccNrm : register(u2);   // sum first-hit normal, sum view depth (0 sky)
RWStructuredBuffer<float4> BufA : register(u3);     // denoiser ping
RWStructuredBuffer<float4> BufB : register(u4);     // denoiser pong / output

Texture2D<float4> Tex[] : register(t0, space1);
SamplerState Samp : register(s0);

// Material flags (pack_params flags.y) and photo flags (extra.x).
#define MF_GLOSS 1u
#define MF_METAL 2u
#define MF_BUMP 4u
#define MF_VTX_MUL 8u
#define MF_ALPHA 16u
#define PF_DECAL 1u
#define PF_TWO_SIDED_LIGHTING 2u

// K.flags (trace)
#define TF_NO_SUN 1u
#define TF_NO_SKY 2u
#define TF_NO_LIGHTS 4u
#define TF_NO_TEXTURES 8u
#define TF_NO_ALPHA 16u   // alpha-tested / decal geometry treated as opaque (FORCE_OPAQUE)

static const float PI = 3.14159265;
static const float INV_PI = 0.31830988618;

uint pcg(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { s = pcg(s); return float(s >> 8) * (1.0 / 16777216.0); }
uint seed_of(uint2 p, uint frame) { return pcg(p.x + pcg(p.y + pcg(frame ^ 0x5bd1e995u))); }

float lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 basis_dir(float3 t1, float3 t2, float3 n, float3 v) { return t1 * v.x + t2 * v.y + n * v.z; }
void frame_of(float3 n, out float3 t1, out float3 t2) {
    t1 = abs(n.z) < 0.999 ? normalize(cross(float3(0, 0, 1), n)) : float3(1, 0, 0);
    t2 = cross(n, t1);
}

// ---------------------------------------------------------------------------------------
// Vertex data

struct Vtx { float4 uv; float3 n; float4 t; float4 col; };

float4 snorm8x4(uint u) {
    int4 s = int4(u << 24, u << 16, u << 8, u) >> 24;
    return max(float4(s) / 127.0, -1.0);
}
float4 unorm8x4(uint u) {
    return float4(u & 255, (u >> 8) & 255, (u >> 16) & 255, u >> 24) / 255.0;
}
Vtx load_vtx(uint addr) {
    uint4 a = Records.Load4(addr);
    uint b = Records.Load(addr + 16);
    Vtx v;
    v.uv = float4(f16tof32(a.x & 0xffff), f16tof32(a.x >> 16), f16tof32(a.y & 0xffff), f16tof32(a.y >> 16));
    v.n = snorm8x4(a.z).xyz;
    v.t = snorm8x4(a.w);
    v.col = unorm8x4(b);
    return v;
}

struct Tri {
    uint mat;
    float lod_bias;
    float4 uv;      // uv0.xy, uv1.zw at the hit
    float3 n_obj;
    float4 t_obj;   // tangent (object), w sign (0 = none)
    float4 col;
};

// Geometry -> (primitive record address, material), clamped to the table sizes.
uint2 locate(uint geom_base, uint geom, uint prim) {
    Params P = Prm[0];
    uint g = min(geom_base + geom, P.geom_count - 1);
    uint2 gi = GeomInfo[g];
    uint p = min(gi.x + prim, P.prim_count - 1);
    return uint2(p * 64, min(gi.y, P.mat_count - 1));
}

Tri tri_at(uint2 loc, float2 bary, bool full) {
    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    Tri r;
    r.mat = loc.y;
    r.lod_bias = asfloat(Records.Load(loc.x));
    Vtx v0 = load_vtx(loc.x + 4), v1 = load_vtx(loc.x + 24), v2 = load_vtx(loc.x + 44);
    r.uv = v0.uv * w.x + v1.uv * w.y + v2.uv * w.z;
    r.n_obj = 0; r.t_obj = 0; r.col = 1;
    if (full) {
        r.n_obj = v0.n * w.x + v1.n * w.y + v2.n * w.z;
        r.t_obj = float4(v0.t.xyz * w.x + v1.t.xyz * w.y + v2.t.xyz * w.z, v0.t.w < 0 ? -1.0 : 1.0);
        r.col = v0.col * w.x + v1.col * w.y + v2.col * w.z;
    }
    return r;
}

// ---------------------------------------------------------------------------------------
// Textures (bindless, clamped index). `lod_uv` = log2 of the footprint in UV units; the
// texture's own size is added here. Non-finite coordinates or LODs (f16 UV overflow,
// degenerate cones) are replaced before sampling.
//
// Divergent indices are resolved with an explicit waterfall (each iteration serves the
// lanes holding the first active lane's index, which is then wave-uniform) instead of
// NonUniformResourceIndex: with NonUniformResourceIndex the WARP validation device was
// removed (0x887A0005) whenever the real per-material texture slots were sampled at LODs
// > 0 (O, E073 smoke bisection: slot 0 only, one uniform slot, or LOD 0 worked; the
// failing condition was not isolated further). With the waterfall every variant runs. On
// AMD hardware NonUniformResourceIndex compiles to the same kind of loop (H).

float4 sample_uniform(uint idx, float2 c, float lod_uv) {
    Texture2D<float4> t = Tex[idx];
    uint w, h, levels;
    t.GetDimensions(0, w, h, levels);
    float lod = lod_uv + 0.5 * log2(float(max(w * h, 1u)));
    lod = isfinite(lod) ? clamp(lod, 0.0, float(max(levels, 1u) - 1)) : 0.0;
    return t.SampleLevel(Samp, c, lod);
}

float4 S(uint slot, float2 uv, float lod_uv) {
    uint idx = min(slot, Prm[0].tex_count - 1);
    float2 c = all(isfinite(uv)) ? frac(uv) : float2(0, 0);
    float4 r = 0;
    [loop] for (;;) {
        uint first = WaveReadLaneFirst(idx);
        if (first == idx) {
            r = sample_uniform(first, c, lod_uv);
            break;
        }
    }
    return r;
}
uint slot_of(Mat m, uint k) { return m.tex[k >> 2][k & 3]; }
float2 pick_uv(uint bit, float4 uv) { return bit != 0 ? uv.zw : uv.xy; }
float scale_lod(float4 sb) { return 0.5 * log2(max(abs(sb.x * sb.y), 1e-8)); }

// Layer i's coordinates (arkdefaulthelpers: layers above extra_layer reuse layer 0).
float2 layer_uv(Mat m, uint i, float4 uv) {
    uint l = i > m.flags.x ? 0 : i;
    float4 sb = m.scale_bias[l];
    return pick_uv(m.modes[l].w & 1, uv) * sb.xy + sb.zw;
}
float layer_lod(Mat m, uint i, float lod_uv) {
    uint l = i > m.flags.x ? 0 : i;
    return lod_uv + scale_lod(m.scale_bias[l]);
}

float linear_step(float lo, float hi, float t) { return saturate((t - lo) / (hi - lo)); }

float layer_mask(Mat m, uint i, float4 vtx, float4 uv, float lod_uv) {
    uint4 mode = m.modes[i];
    float k = 0.0;
    if (i == 0 && (mode.w & 4) == 0) {
        k = 0.0;
    } else if (mode.x == 0) {
        k = vtx[min(mode.y, 3u)];
    } else {
        uint src = min(m.mask_source[i], 2u);
        float4 sb = m.mask_scale_bias[src];
        float t = S(slot_of(m, 12 + src), pick_uv(m.modes[src].w & 2, uv) * sb.xy + sb.zw, lod_uv + scale_lod(sb)).r;
        k = mode.x == 1 ? t : linear_step(m.rescale[i].x, m.rescale[i].y, t * vtx[min(mode.y, 3u)]);
    }
    if (mode.z != 0) { k = 1.0 - k; }
    return k;
}

float4 pick3(uint reuse, float4 own, float4 l0, float4 l1) { return reuse == 2 ? l1 : (reuse == 1 ? l0 : own); }

// Layer-0 diffuse alpha (alpha test / decal coverage), as the engine's clip / blend input.
float coverage(Mat m, Tri r, float lod_uv) {
    float4 d = S(slot_of(m, 0), layer_uv(m, 0, r.uv), layer_lod(m, 0, lod_uv));
    if ((m.extra.x & PF_DECAL) != 0 && (m.modes[0].w & 4) != 0) { d.a *= m.color_constant[0].a; }
    return d.a;
}

// Object -> world normal: the inverse transpose of the 3x3 part is cofactor(R) / det; its
// rows are the cross products of R's rows (scale and sign removed by normalize / the
// facing flip done by the callers).
float3 world_normal(float3x3 R, float3 n) {
    float3 w = mul(float3x3(cross(R[1], R[2]), cross(R[2], R[0]), cross(R[0], R[1])), n);
    return dot(w, w) > 1e-20 ? normalize(w) : float3(0, 0, 1);
}

struct Surf {
    float3 pos;
    float3 ng;        // interpolated vertex normal, facing the incoming ray
    float3 n;         // shading normal (normal map), facing the incoming ray
    float3 albedo;    // base colour (feature buffer)
    float3 diffuse;   // Lambert colour
    float3 f0;        // specular reflectance at normal incidence
    float alpha;      // GGX roughness^2
    float spec_ok;    // 0 when the material has no specular (tint 0)
};

// arkDefault material evaluation (port of the R3 renderer's `ps`, checkpoint 9282529) at the
// hit, then the engine's gloss/metal -> lighting parameter mapping
// (NOTOUCHY_MapGlossAndMetalToLightingPipelineValue: tint = lerp(specularTintDielectric,
// albedo, metal^2), diffuse = lerp(albedo, diffuseAlbedoMetallic, metal^2)).
Surf shade_surface(Tri r, float3x4 o2w, float3 dir, float t, float3 origin, float lod_uv, uint tflags) {
    Mat m = Mats[r.mat];
    Surf s;
    s.pos = origin + dir * t;
    float3x3 R = (float3x3)o2w;
    float det = determinant(R);
    float3 nw = world_normal(R, r.n_obj);
    bool back = dot(nw, dir) > 0;
    float3 ng = back ? -nw : nw;
    float4 uv = r.uv;
    float4 vtx = r.col;
    uint extra = m.flags.x;
    float3 masks = 0;
    float4 albedo = float4(asfloat(m.extra.yzw), 1);
    float gloss = m.ranges.y;
    float metallic = m.ranges.w;
    float3 n = ng;
    if ((tflags & TF_NO_TEXTURES) == 0) {
        masks.x = layer_mask(m, 0, vtx, uv, lod_uv);
        if (extra > 0) { masks.y = layer_mask(m, 1, vtx, uv, lod_uv); }
        if (extra > 1) { masks.z = layer_mask(m, 2, vtx, uv, lod_uv); }
        float2 t0 = layer_uv(m, 0, uv), t1 = layer_uv(m, 1, uv), t2 = layer_uv(m, 2, uv);
        float l0 = layer_lod(m, 0, lod_uv), l1 = layer_lod(m, 1, lod_uv), l2 = layer_lod(m, 2, lod_uv);
        float4 D0 = S(slot_of(m, 0), t0, l0);
        albedo = D0;
        if ((m.modes[0].w & 4) != 0) { albedo *= lerp(float4(1, 1, 1, 1), m.color_constant[0], masks.x); }
        float4 D1 = D0;
        if (extra > 0) {
            if (m.reuse[1].x == 0) { D1 = S(slot_of(m, 1), t1, l1); }
            float4 L1 = D1;
            if ((m.modes[1].w & 4) != 0) { L1 *= m.color_constant[1]; }
            albedo = lerp(albedo, L1, masks.y);
            if (extra > 1) {
                float4 L2 = pick3(m.reuse[2].x, S(slot_of(m, 2), t2, l2), D0, D1);
                if ((m.modes[2].w & 4) != 0) { L2 *= m.color_constant[2]; }
                albedo = lerp(albedo, L2, masks.z);
            }
        }
        if ((m.flags.y & MF_VTX_MUL) != 0) { albedo *= vtx; }
        if ((m.flags.y & MF_GLOSS) != 0) {
            float G0 = S(slot_of(m, 6), t0, l0).r, G1 = G0, g = G0;
            if (extra > 0 && m.reuse[1].z == 0) { G1 = S(slot_of(m, 7), t1, l1).r; g = lerp(g, G1, masks.y); }
            if (extra > 1) { g = lerp(g, pick3(m.reuse[2].z, S(slot_of(m, 8), t2, l2), G0, G1).r, masks.z); }
            gloss = lerp(m.ranges.x, m.ranges.y, g);
        }
        if ((m.flags.y & MF_METAL) != 0) {
            float M0 = S(slot_of(m, 9), t0, l0).r, M1 = M0, k = M0;
            if (extra > 0 && m.reuse[1].w == 0) { M1 = S(slot_of(m, 10), t1, l1).r; k = lerp(k, M1, masks.y); }
            if (extra > 1) { k = lerp(k, pick3(m.reuse[2].w, S(slot_of(m, 11), t2, l2), M0, M1).r, masks.z); }
            metallic = lerp(m.ranges.z, m.ranges.w, k);
        }
        float3 tw = mul(R, r.t_obj.xyz);
        if ((m.flags.y & MF_BUMP) != 0 && dot(tw, tw) > 1e-8) {
            float2 R0 = S(slot_of(m, 3), t0, l0).xy, R1 = R0, raw = R0;
            if (extra > 0 && m.reuse[1].y == 0) { R1 = S(slot_of(m, 4), t1, l1).xy; raw = lerp(raw, R1, masks.y); }
            if (extra > 1) { raw = lerp(raw, pick3(m.reuse[2].y, S(slot_of(m, 5), t2, l2), float4(R0, 0, 0), float4(R1, 0, 0)).xy, masks.z); }
            // UnpackBC5NormalMap; TBN from the vertex frame (bitangent = cross(n, t) * w,
            // mirrored instances flip the handedness).
            float3 ts = normalize(float3(raw, 1.001 - saturate(dot(raw, raw))));
            float3 T = normalize(tw - nw * dot(nw, tw));
            float3 B = cross(nw, T) * r.t_obj.w * (det < 0 ? -1.0 : 1.0);
            float3 nm = normalize(ts.x * T + ts.y * B + ts.z * nw);
            nm = back ? -nm : nm;
            // Keep the shading normal on the visible side.
            if (dot(nm, -dir) > 0.02) { n = nm; }
        }
    }
    albedo = max(albedo, 0.00001);
    s.ng = ng;
    s.n = n;
    s.albedo = saturate(albedo.rgb);
    float met = metallic * metallic;
    float3 tint = lerp(m.lit0.xyz, s.albedo, met);
    float g2 = lerp(smoothstep(0.0, 1.07, gloss), smoothstep(0.05, 1.0, gloss), gloss);
    tint *= saturate((g2 + met) / 0.05);
    s.diffuse = saturate(lerp(s.albedo, m.lit1.xyz, met));
    s.f0 = saturate(tint);
    float rough = saturate(1.0 - gloss);
    s.alpha = max(0.01, rough * rough);
    s.spec_ok = any(s.f0 > 0) ? 1.0 : 0.0;
    return s;
}

// ---------------------------------------------------------------------------------------
// BSDF: Lambert (scaled by 1 - F) + GGX (Smith height-correlated, Schlick Fresnel).

float3 fresnel(float3 f0, float c) { float m = 1.0 - saturate(c); float m2 = m * m; return f0 + (1.0 - f0) * (m2 * m2 * m); }
float ggx_d(float a2, float nh) { float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
float smith_g1(float a2, float nv) { return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv)); }
float smith_vis(float a2, float nv, float nl) {
    return 0.5 / max(nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2), 1e-7);
}
float spec_prob(Surf s, float nv) {
    if (s.spec_ok == 0) return 0;
    float fs = lum(fresnel(s.f0, nv));
    float fd = lum(s.diffuse) * (1.0 - fs);
    return clamp(fs / max(fs + fd, 1e-6), 0.1, 0.9);
}

// f * cos for direction l (spec scaled by `spec_attn`), and the sampling pdf of l.
float3 bsdf_eval(Surf s, float3 v, float3 l, float spec_attn, out float pdf) {
    pdf = 0;
    float nl = dot(s.n, l), nv = dot(s.n, v);
    if (nl <= 0 || nv <= 0 || dot(s.ng, l) <= 0) return 0;
    float3 h = normalize(v + l);
    float nh = saturate(dot(s.n, h)), vh = saturate(dot(v, h));
    float a2 = s.alpha * s.alpha;
    float3 F = fresnel(s.f0, vh);
    float D = ggx_d(a2, nh);
    float3 spec = s.spec_ok * D * smith_vis(a2, nv, nl) * F * spec_attn;
    float3 diff = s.diffuse * INV_PI * (1.0 - F * s.spec_ok);
    float ps = spec_prob(s, nv);
    pdf = ps * smith_g1(a2, nv) * D / (4.0 * nv) + (1.0 - ps) * nl * INV_PI;
    return (diff + spec) * nl;
}

float3 cosine_dir(float3 n, inout uint seed) {
    float u1 = rnd(seed), u2 = rnd(seed);
    float r = sqrt(u1), phi = 2 * PI * u2;
    float3 t1, t2; frame_of(n, t1, t2);
    return normalize(basis_dir(t1, t2, n, float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1 - u1)))));
}

// GGX VNDF sample (Heitz 2018): reflected direction.
float3 ggx_dir(Surf s, float3 v, inout uint seed) {
    float a = s.alpha;
    float3 t1, t2; frame_of(s.n, t1, t2);
    float3 Ve = float3(dot(v, t1), dot(v, t2), dot(v, s.n));
    float3 Vh = normalize(float3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);
    float u1 = rnd(seed), u2 = rnd(seed);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    float p1 = r * cos(phi), p2 = r * sin(phi);
    float k = 0.5 * (1.0 + Vh.z);
    p2 = (1.0 - k) * sqrt(max(0.0, 1.0 - p1 * p1)) + k * p2;
    float3 Nh = p1 * T1 + p2 * T2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * Vh;
    float3 He = normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));
    float3 H = basis_dir(t1, t2, s.n, He);
    return reflect(-v, H);
}

// ---------------------------------------------------------------------------------------
// Tracing with alpha: alpha-tested geometry commits when layer-0 alpha >= 0.5 (engine
// `clip(a - 0.5)`); decals commit stochastically with probability alpha (expected value =
// alpha blending) and never block shadow rays.

struct Hit { bool hit; float t; uint2 loc; float2 bary; float3x4 o2w; };

// Candidate test of the current non-opaque triangle.
bool accept_candidate(RayQuery<RAY_FLAG_NONE> q, bool shadow, float cone_w, float cone_g, float u) {
    uint2 loc = locate(q.CandidateInstanceID(), q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
    bool decal = (Mats[loc.y].extra.x & PF_DECAL) != 0;
    if (decal && shadow) return false;
    if ((K.flags & TF_NO_TEXTURES) != 0) return true;
    Tri t = tri_at(loc, q.CandidateTriangleBarycentrics(), false);
    float w = cone_w + cone_g * q.CandidateTriangleRayT();
    float a = coverage(Mats[loc.y], t, t.lod_bias + log2(max(w, 1e-6)));
    return decal ? a > u : a >= 0.5;
}

Hit trace(RayDesc r, float cone_w, float cone_g, inout uint seed) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, (K.flags & TF_NO_ALPHA) != 0 ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_NONE, 0xFF, r);
    float u = rnd(seed);
    [loop] while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && accept_candidate(q, false, cone_w, cone_g, u)) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    Hit h;
    h.hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    h.t = h.hit ? q.CommittedRayT() : -1;
    h.loc = 0; h.bary = 0; h.o2w = (float3x4)0;
    if (h.hit) {
        h.loc = locate(q.CommittedInstanceID(), q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        h.bary = q.CommittedTriangleBarycentrics();
        h.o2w = q.CommittedObjectToWorld3x4();
    }
    return h;
}

// Shadow ray: any opaque hit or accepted alpha-tested candidate occludes. Opaque hits end
// the search through RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH; an accepted candidate is
// not committed but ends the query with Abort(): CommitNonOpaqueTriangleHit under that
// flag removed the WARP validation device (0x887A0005, O, E073 smoke runs).
bool visible(float3 p, float3 dir, float t_max) {
    RayDesc r;
    r.Origin = p;
    r.Direction = dir;
    r.TMin = 0;
    r.TMax = t_max;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | ((K.flags & TF_NO_ALPHA) != 0 ? RAY_FLAG_FORCE_OPAQUE : RAY_FLAG_NONE), 0xFF, r);
    bool blocked = false;
    [loop] while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && accept_candidate(q, true, 0, 0, 0)) {
            blocked = true;
            q.Abort();
        }
    }
    return !blocked && q.CommittedStatus() == COMMITTED_NOTHING;
}

float3 offset_origin(Surf s, float dist, float3 dir) {
    float3 n = dot(dir, s.ng) >= 0 ? s.ng : -s.ng;
    return s.pos + n * max(0.003, 1e-4 * dist);
}

// ---------------------------------------------------------------------------------------
// Sun and sky. Sky radiance: lerp(skyZenith, skyHorizon, polar angle / 90 deg) over the upper
// hemisphere (the engine's sky ZH source, `sub_1402B0370`), zero below; x sky_scale.

float3 sky(float3 d) {
    Params P = Prm[0];
    if (d.z <= 0) return 0;
    float t = acos(saturate(d.z)) / (0.5 * PI);
    return P.sky_scale * lerp(P.sky_zenith, P.sky_horizon, t);
}
float sky_pdf(float3 d) { return d.z > 0 ? 1.0 / (2.0 * PI) : 0.0; }
float3 sky_sample(inout uint seed) {
    float z = rnd(seed), phi = 2 * PI * rnd(seed);
    float r = sqrt(max(0.0, 1 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}
float sun_pdf() { return 1.0 / (2.0 * PI * (1.0 - Prm[0].sun_cos)); }
bool in_sun(float3 d) { return dot(d, Prm[0].sun_dir) >= Prm[0].sun_cos; }
float3 sun_sample(inout uint seed) {
    float c = lerp(1.0, Prm[0].sun_cos, rnd(seed));
    float sn = sqrt(max(0.0, 1 - c * c)), phi = 2 * PI * rnd(seed);
    float3 n = Prm[0].sun_dir;
    float3 t1, t2; frame_of(n, t1, t2);
    return normalize(basis_dir(t1, t2, n, float3(sn * cos(phi), sn * sin(phi), c)));
}
float power_heuristic(float a, float b) { float a2 = a * a; return a2 / max(a2 + b * b, 1e-30); }

// ---------------------------------------------------------------------------------------
// Local lights (engine attenuation, see lights.rs).

float3 light_at(Light L, float3 p, out float3 l, out float d) {
    float3 to = L.a.xyz - p;
    d = length(to);
    l = to / max(d, 1e-6);
    float att = lerp(1.0, 1.0 / max(d, 1e-3), L.c.w);
    att *= att;
    float s = saturate((d - L.b.w) / (L.a.w - L.b.w));   // smoothstep(cutoff, falloff, d)
    att *= s * s * (3.0 - 2.0 * s);
    uint kind = asuint(L.d.w);
    if (kind == 2) {
        att *= 1.0 - saturate(dot(l, L.c.xyz) * L.d.x + L.d.y);
    } else if (kind == 1) {
        float a = dot(l, -L.c.xyz);
        att = a <= 0.0 ? 0.0 : att * (1.0 - saturate(a * L.d.x + L.d.y));
    }
    return L.b.xyz * att;
}

// RIS over the light list of p's grid cell: one light, its unbiased contribution weight W.
bool pick_light(float3 p, float3 n, inout uint seed, out uint li, out float W) {
    Params P = Prm[0];
    li = 0; W = 0;
    if (P.light_count == 0) return false;
    float2 c = (p.xy - P.grid_origin) / P.grid_cell;
    if (any(c < 0) || c.x >= P.grid_dims.x || c.y >= P.grid_dims.y) return false;
    uint ci = min(uint(c.y) * P.grid_dims.x + uint(c.x), P.cell_count - 1);
    uint2 cell = Cells[ci];
    if (P.index_count == 0) return false;
    uint first = min(cell.x, P.index_count - 1);
    uint count = min(cell.y, P.index_count - first);
    if (count == 0) return false;
    uint M = min(count, max(P.ris_m, 1u));
    float wsum = 0, chosen_p = 0;
    for (uint j = 0; j < M; j++) {
        uint k = first + min(uint(rnd(seed) * count), count - 1);
        uint cand = min(LightIdx[min(k, P.index_count - 1)], P.light_count - 1);
        float3 l; float d;
        float3 e = light_at(Lights[cand], p, l, d);
        float ph = lum(e) * saturate(dot(n, l));
        float w = ph * count;   // target / source pdf (uniform over the cell list)
        wsum += w;
        if (w > 0 && rnd(seed) * wsum <= w) { li = cand; chosen_p = ph; }
    }
    if (wsum <= 0 || chosen_p <= 0) return false;
    W = wsum / (M * chosen_p);
    return true;
}

// ---------------------------------------------------------------------------------------

float3 cam_dir(float2 pix) {
    Params P = Prm[0];
    float2 ndc = float2(2.0 * pix.x / K.width - 1.0, 1.0 - 2.0 * pix.y / K.height);
    return P.fwd + ndc.x * P.tan_x * P.right + ndc.y * P.tan_y * P.up;
}

float3 clamp_lum(float3 c, float m) { float l = lum(c); return l > m ? c * (m / l) : c; }

// photo_trace: one path sample per pixel; accumulates radiance, luminance^2 and the
// first-hit feature buffers.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_trace(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    Params P = Prm[0];
    uint i = px.y * K.width + px.x;
    uint seed = seed_of(px, K.frame);
    float2 jitter = float2(rnd(seed), rnd(seed));
    float3 d = cam_dir(px + jitter);
    RayDesc r;
    r.Origin = P.org;
    r.Direction = normalize(d);
    r.TMin = P.near_z;
    r.TMax = 1e5;
    float3 L = 0, thr = 1;
    float cone_w = 0, cone_g = P.pixel_spread;
    float bsdf_pdf = 0;   // pdf of the last BSDF-sampled direction (0 = camera ray)
    float4 f_alb = 0, f_nrm = 0;
    uint tf = K.flags;
    [loop] for (uint b = 0; b <= P.max_bounces; b++) {
        Hit h = trace(r, cone_w, cone_g, seed);
        float limit = b == 0 ? 3.0e38 : P.clamp_indirect;
        if (!h.hit) {
            float3 e = 0;
            if ((tf & TF_NO_SKY) == 0) {
                float w = b == 0 ? 1.0 : power_heuristic(bsdf_pdf, sky_pdf(r.Direction));
                e += sky(r.Direction) * w;
            }
            if ((tf & TF_NO_SUN) == 0 && in_sun(r.Direction)) {
                float w = b == 0 ? 1.0 : power_heuristic(bsdf_pdf, sun_pdf());
                e += P.sun_irr * sun_pdf() * w;
            }
            L += clamp_lum(thr * e, limit);
            break;
        }
        float w_hit = cone_w + cone_g * h.t;
        Tri tri = tri_at(h.loc, h.bary, true);
        float det = abs(determinant((float3x3)h.o2w));
        // Ray-cone LOD (Akenine-Moller et al. 2021): Delta (object space) + world scale +
        // log2(cone width / |cos|); the texture size is added per map.
        float cosv = max(abs(dot(world_normal((float3x3)h.o2w, tri.n_obj), r.Direction)), 0.1);
        float lod_uv = tri.lod_bias - log2(max(det, 1e-12)) / 3.0 + log2(max(w_hit, 1e-6) / cosv);
        Surf s = shade_surface(tri, h.o2w, r.Direction, h.t, r.Origin, lod_uv, tf);
        float3 v = -r.Direction;
        float3 o = offset_origin(s, h.t, s.ng);
        if (b == 0) {
            f_alb.rgb = s.albedo;
            f_nrm = float4(s.n, h.t * dot(r.Direction, P.fwd));
        }
        // Next-event estimation: sun (MIS), sky (MIS), one local light (RIS).
        float3 direct = 0;
        if ((tf & TF_NO_SUN) == 0) {
            float3 l = sun_sample(seed);
            float pdf;
            float3 f = bsdf_eval(s, v, l, 1.0, pdf);
            bool vis = false;
            if (any(f > 0)) {
                vis = visible(offset_origin(s, h.t, l), l, 1e5);
                if (vis) direct += f * P.sun_irr * power_heuristic(sun_pdf(), pdf);
            }
            if (b == 0) f_alb.a = vis ? 1.0 : 0.0;
        }
        if ((tf & TF_NO_SKY) == 0) {
            float3 l = sky_sample(seed);
            float pdf;
            float3 f = bsdf_eval(s, v, l, 1.0, pdf);
            if (any(f > 0) && visible(offset_origin(s, h.t, l), l, 1e5)) {
                direct += f * sky(l) / sky_pdf(l) * power_heuristic(sky_pdf(l), pdf);
            }
        }
        uint li;
        float W;
        if ((tf & TF_NO_LIGHTS) == 0 && pick_light(s.pos, s.n, seed, li, W)) {
            Light Lt = Lights[li];
            float3 l; float dist;
            float3 e = light_at(Lt, s.pos, l, dist);
            float pdf;
            float3 f = bsdf_eval(s, v, l, Lt.e.x, pdf);
            if (any(f > 0) && any(e > 0)) {
                float3 so = offset_origin(s, h.t, l);
                if (visible(so, l, max(0.0, length(Lt.a.xyz - so) - 0.01))) direct += f * e * W;
            }
        }
        L += clamp_lum(thr * direct, limit);
        if (b == P.max_bounces) break;
        // BSDF sample for the next bounce.
        float ps = spec_prob(s, dot(s.n, v));
        bool spec = rnd(seed) < ps;
        float3 l = spec ? ggx_dir(s, v, seed) : cosine_dir(s.n, seed);
        float pdf;
        float3 f = bsdf_eval(s, v, l, 1.0, pdf);
        if (pdf <= 0 || !any(f > 0)) break;
        thr *= f / pdf;
        // Russian roulette from bounce rr_start on.
        if (b + 1 >= P.rr_start) {
            float q = clamp(max(thr.r, max(thr.g, thr.b)), 0.05, 0.95);
            if (rnd(seed) > q) break;
            thr /= q;
        }
        if (!all(isfinite(thr))) break;
        bsdf_pdf = pdf;
        // Cone after the bounce (simplified: diffuse lobes widen it strongly).
        cone_w = w_hit;
        cone_g = cone_g + (spec ? 2.0 * s.alpha : 0.5);
        r.Origin = offset_origin(s, h.t, l);
        r.Direction = l;
        r.TMin = 0;
        r.TMax = 1e5;
    }
    if (!all(isfinite(L))) L = 0;
    AccRad[i] += float4(L, lum(L) * lum(L));
    AccAlb[i] += f_alb;
    AccNrm[i] += f_nrm;
}

// ---------------------------------------------------------------------------------------
// Copy-ready outputs (photo.rs `hdr_pitch` / `depth_pitch`): rows padded to 256 bytes so the
// buffers copy straight into textures (CopyTextureRegion placed footprints).

uint hdr_row(uint w) { return (w * 16u + 255u) / 256u * 16u; }   // float4 elements per row
uint depth_row(uint w) { return (w * 4u + 255u) / 256u * 64u; }  // floats per row

// photo_depth: game depth of the primary hit through the pixel centre (znear / view z, 0 =
// sky: the engine's reversed infinite-Z convention), alpha test as in photo_trace. Writes
// u4 as packed floats (R32_FLOAT rows of depth_row(width)).
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_depth(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    Params P = Prm[0];
    uint seed = seed_of(px, 0x7fffffffu);
    RayDesc r;
    r.Origin = P.org;
    r.Direction = normalize(cam_dir(float2(px) + 0.5));
    r.TMin = P.near_z;
    r.TMax = 1e5;
    Hit h = trace(r, 0.0, P.pixel_spread, seed);
    float d = 0.0;
    if (h.hit) {
        float z = h.t * dot(r.Direction, P.fwd);
        d = z > 0.0 ? P.near_z / z : 0.0;
    }
    uint j = px.y * depth_row(K.width) + px.x;
    uint e = j >> 2;
    switch (j & 3u) {
        case 0u: BufB[e].x = d; break;
        case 1u: BufB[e].y = d; break;
        case 2u: BufB[e].z = d; break;
        default: BufB[e].w = d; break;
    }
}

// photo_aov (offline debug): primary hit through the pixel centre (alpha test and decals as
// photo_depth), material evaluation as photo_trace's first hit. u4 = (albedo rgb,
// asfloat(material + 1), 0 = miss), u3 = (uv0, texture LOD = log2 of the footprint in UV
// units before the map size, view depth); rows of `width` float4.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_aov(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    Params P = Prm[0];
    uint i = px.y * K.width + px.x;
    uint seed = seed_of(px, 0x7fffffffu);
    RayDesc r;
    r.Origin = P.org;
    r.Direction = normalize(cam_dir(float2(px) + 0.5));
    r.TMin = P.near_z;
    r.TMax = 1e5;
    Hit h = trace(r, 0.0, P.pixel_spread, seed);
    if (!h.hit) {
        BufB[i] = 0;
        BufA[i] = 0;
        return;
    }
    Tri tri = tri_at(h.loc, h.bary, true);
    float det = abs(determinant((float3x3)h.o2w));
    float cosv = max(abs(dot(world_normal((float3x3)h.o2w, tri.n_obj), r.Direction)), 0.1);
    float lod_uv = tri.lod_bias - log2(max(det, 1e-12)) / 3.0 + log2(max(P.pixel_spread * h.t, 1e-6) / cosv);
    Surf s = shade_surface(tri, h.o2w, r.Direction, h.t, r.Origin, lod_uv, K.flags);
    BufB[i] = float4(s.albedo, asfloat(tri.mat + 1));
    BufA[i] = float4(tri.uv.xy, lod_uv, h.t * dot(r.Direction, P.fwd));
}

// photo_resolve: mean radiance AccRad.rgb / K.frame (engine absolute units) into u3, rows of
// hdr_row(width) float4 (alpha 1).
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_resolve(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    float n = max(float(K.frame), 1.0);
    BufA[px.y * hdr_row(K.width) + px.x] = float4(AccRad[px.y * K.width + px.x].rgb / n, 1.0);
}

// ---------------------------------------------------------------------------------------
// Denoiser (edge-aware a-trous on albedo-demodulated radiance, Dammertz et al. 2010 with
// SVGF-style luminance stopping from the variance of the per-pixel mean). The variance is
// the per-pixel sample variance / n; below 16 samples it is at least the 3x3 spatial
// variance of the demodulated mean (SVGF's fallback for short histories: one sample has no
// sample variance). K.frame = samples.

static const float SKY_DEPTH = 0.0;

float demod_lum(uint j, float n) {
    return lum(AccRad[j].rgb / n) / max(lum(AccAlb[j].rgb / n), 0.02);
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_demod(uint3 id : SV_DispatchThreadID) {
    if (id.x >= K.width || id.y >= K.height) return;
    uint i = id.y * K.width + id.x;
    float n = max(float(K.frame), 1.0);
    float4 a = AccRad[i] / n;
    float3 alb = max(AccAlb[i].rgb / n, 0.02);
    float depth = AccNrm[i].w / n;
    if (depth <= SKY_DEPTH) { BufA[i] = float4(a.rgb, -1); return; }
    float var = max(0.0, a.w - lum(a.rgb) * lum(a.rgb)) / n / (lum(alb) * lum(alb));
    if (n < 16) {
        float s1 = 0, s2 = 0, c = 0;
        [unroll] for (int dy = -1; dy <= 1; dy++) {
            [unroll] for (int dx = -1; dx <= 1; dx++) {
                int2 q = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), int2(K.width - 1, K.height - 1));
                uint j = uint(q.y) * K.width + uint(q.x);
                if (AccNrm[j].w <= SKY_DEPTH) continue;
                float l = demod_lum(j, n);
                s1 += l; s2 += l * l; c += 1;
            }
        }
        var = max(var, max(0.0, s2 / c - (s1 / c) * (s1 / c)));
    }
    BufA[i] = float4(a.rgb / alb, var);
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_atrous(uint3 id : SV_DispatchThreadID) {
    if (id.x >= K.width || id.y >= K.height) return;
    uint i = id.y * K.width + id.x;
    float n = max(float(K.frame), 1.0);
    float4 c = BufA[i];
    if (c.w < 0) { BufB[i] = c; return; }
    float4 nd = AccNrm[i] / n;
    float3 np = normalize(nd.xyz + 1e-9);
    float zp = nd.w;
    float lp = lum(c.rgb);
    float sigma_l = 4.0 * sqrt(max(c.w, 0.0)) + 1e-4 * lp + 1e-6;
    const float k[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };
    float3 sum = 0;
    float var = 0, wsum = 0;
    int step = int(max(K.step, 1u));
    [loop] for (int dy = -2; dy <= 2; dy++) {
        [loop] for (int dx = -2; dx <= 2; dx++) {
            int2 q = int2(id.xy) + int2(dx, dy) * step;
            if (q.x < 0 || q.y < 0 || q.x >= int(K.width) || q.y >= int(K.height)) continue;
            uint j = uint(q.y) * K.width + uint(q.x);
            float4 cq = BufA[j];
            if (cq.w < 0) continue;
            float4 nq = AccNrm[j] / n;
            float wn = pow(saturate(dot(np, normalize(nq.xyz + 1e-9))), 64.0);
            float wz = exp(-abs(zp - nq.w) / (0.02 * zp * step + 1e-4));
            float wl = exp(-abs(lp - lum(cq.rgb)) / sigma_l);
            float w = k[abs(dx)] * k[abs(dy)] * wn * wz * wl;
            sum += cq.rgb * w;
            var += cq.w * w * w;
            wsum += w;
        }
    }
    BufB[i] = wsum > 0 ? float4(sum / wsum, var / (wsum * wsum)) : c;
}

[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_remod(uint3 id : SV_DispatchThreadID) {
    if (id.x >= K.width || id.y >= K.height) return;
    uint i = id.y * K.width + id.x;
    float n = max(float(K.frame), 1.0);
    float4 c = BufA[i];
    if (c.w < 0) { BufB[i] = float4(AccRad[i].rgb / n, 1); return; }
    float3 alb = max(AccAlb[i].rgb / n, 0.02);
    BufB[i] = float4(c.rgb * alb, 1);
}

// ---------------------------------------------------------------------------------------
// Hybrid photo terms (docs/raytracing.md section 13; host `hybrid.rs`): the game keeps
// rendering its paused frame and only its sun-visibility (`_SSAO_ScreenShadows.y`) and AO
// (`.x`) inputs are replaced. Per pixel of the game's depth: K.step rays per dispatch, the
// running mean over batches kept per pixel.
//
// Buffers (all packed floats in rows of depth_row(width), R32F-copy-ready like photo_depth):
// u1 = the game's view depth of the region (metres along the forward axis, 0 = sky); after
// the prepass our surface's depth where it agrees with the game's, or -(game depth) where it
// does not (rejected), u2 = the receiver normal from the prepass (hybrid_pack_normal, 0 =
// none), u3 = mean sun visibility (photo_sun_vis), u4 = mean ambient visibility (photo_ao),
// u0 = the edge-aware filtered sun visibility (photo_hybrid_filter, the copied output).
// K.frame = batches already in the mean, K.step = rays per pixel in this batch (1..4: long
// per-thread ray loops ran slower per ray, hybrid.rs MAX_RAYS_PER_DISPATCH), K.pad1 / pad2 =
// asfloat projection jitter (NDC) of the depth's frame. Params: camera basis
// of that frame (fwd/right/up/tan as the engine projection), sun_dir / sun_cos (engine sun,
// disc radius), ao_radius, depth_step (two R24 depth steps per z^2: the quantisation error
// of the captured depth). Every index stays inside the padded buffers the host allocates
// (height rows of depth_row(width)); the CPU checks at least 4 bytes per pixel.

// Element j of a packed-float buffer (no selects between buffers: an unbound slot holds the
// dummy address, so only the buffer the kernel binds is ever touched).
float packed_get(RWStructuredBuffer<float4> b, uint j) {
    float4 e = b[j >> 2];
    uint c = j & 3u;
    return c == 0u ? e.x : (c == 1u ? e.y : (c == 2u ? e.z : e.w));
}

void packed_set(RWStructuredBuffer<float4> b, uint j, float v) {
    uint e = j >> 2;
    switch (j & 3u) {
        case 0u: b[e].x = v; break;
        case 1u: b[e].y = v; break;
        case 2u: b[e].z = v; break;
        default: b[e].w = v; break;
    }
}

int2 hybrid_clamp(int2 p) { return clamp(p, int2(0, 0), int2(K.width - 1, K.height - 1)); }
uint hybrid_index(int2 p) { p = hybrid_clamp(p); return uint(p.y) * depth_row(K.width) + uint(p.x); }

// Raw receiver depth at region pixel p (clamped into the region): > 0 our surface (or the
// game's before the prepass), < 0 rejected (-game depth), 0 = sky / invalid.
float hybrid_raw(int2 p) {
    float z = packed_get(AccAlb, hybrid_index(p));
    return isfinite(z) ? z : 0.0;
}

// Unsigned depth (rejected pixels keep the game's depth for the AO fallback).
float hybrid_depth(int2 p) { return abs(hybrid_raw(p)); }

// World direction through pixel position `pix` with a unit forward component (the engine
// projection: ndc = P * view / z + jitter).
float3 hybrid_ray(float2 pix) {
    Params P = Prm[0];
    float2 ndc = float2(2.0 * pix.x / K.width - 1.0, 1.0 - 2.0 * pix.y / K.height)
        - float2(asfloat(K.pad1), asfloat(K.pad2));
    return P.fwd + ndc.x * P.tan_x * P.right + ndc.y * P.tan_y * P.up;
}

float3 hybrid_pos(int2 p, float z) { return Prm[0].org + z * hybrid_ray(float2(p) + 0.5); }

// Unit normal <-> one float: octahedral, 12 bits per axis, 1 + (u << 12 | v) (an integer
// below 2^24, exact in a float; 0 = no normal).
float hybrid_pack_normal(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    uint2 q = uint2(round(saturate(e * 0.5 + 0.5) * 4095.0));
    return float(1u + (q.x << 12) + q.y);
}

bool hybrid_normal(int2 p, out float3 n) {
    float f = packed_get(AccNrm, hybrid_index(p));
    n = float3(0, 0, 1);
    if (!(isfinite(f) && f >= 1.0)) return false;
    uint u = min(uint(f) - 1u, 0xFFFFFFu);
    float2 e = float2(u >> 12, u & 4095u) / 4095.0 * 2.0 - 1.0;
    float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-v.z);
    v.xy += float2(v.x >= 0.0 ? -t : t, v.y >= 0.0 ? -t : t);
    n = normalize(v);
    return true;
}

// Ray-origin offset along the normal: 1 cm + 0.2 % of the depth (float precision of the
// hit and the coarser LODs at distance).
float hybrid_eps(float z) { return 0.01 + 0.002 * z; }

// Receiver of the prepass: our surface's position and its geometric (interpolated vertex)
// normal facing the camera. False for sky and rejected pixels.
bool hybrid_receiver(int2 p, out float3 P, out float3 N, out float eps) {
    float z = hybrid_raw(p);
    P = 0; N = float3(0, 0, 1); eps = 0;
    if (!(z > 0.0) || !hybrid_normal(p, N)) return false;
    P = hybrid_pos(p, z);
    eps = hybrid_eps(z);
    return true;
}

// Ray origin: half of `eps` back along the camera ray (the free space the primary ray
// crossed) plus a quarter of it along the normal. A pure normal offset at a concave corner
// (riser foot, wall base) lands on or below the adjacent face and the shadow ray starts
// inside it (O, hybrid_check RX010: the isolated black pixels left on the steps after the
// prepass fix were mostly riser slivers one pixel tall whose normal offset ran parallel to
// the tread below).
float3 hybrid_origin(float3 P, float3 N, float eps) {
    float3 v = normalize(P - Prm[0].org);
    return P + (0.25 * N - 0.5 * v) * eps;
}

// Fallback receiver from the depth alone (AO of rejected pixels: characters and objects we
// lack): world position, geometric normal facing the camera (per axis the neighbour with
// the smaller depth step, as contact.rs; the other side at the region edge), and the offset
// with the R24 quantisation error.
bool hybrid_surface(int2 p, out float3 P, out float3 N, out float eps) {
    float z = hybrid_depth(p);
    P = 0; N = float3(0, 0, 1); eps = 0;
    if (z <= 0.0) return false;
    P = hybrid_pos(p, z);
    int2 last = int2(K.width - 1, K.height - 1);
    float zpx = hybrid_depth(p + int2(1, 0)), zmx = hybrid_depth(p - int2(1, 0));
    float zpy = hybrid_depth(p + int2(0, 1)), zmy = hybrid_depth(p - int2(0, 1));
    bool mx = p.x > 0 && (p.x >= last.x || abs(zmx - z) <= abs(zpx - z));
    bool my = p.y > 0 && (p.y >= last.y || abs(zmy - z) <= abs(zpy - z));
    float3 dx = mx ? P - hybrid_pos(p - int2(1, 0), zmx) : hybrid_pos(p + int2(1, 0), zpx) - P;
    float3 dy = my ? P - hybrid_pos(p - int2(0, 1), zmy) : hybrid_pos(p + int2(0, 1), zpy) - P;
    float3 n = cross(dx, dy);
    float l = length(n);
    float3 view = P - Prm[0].org;
    N = (isfinite(l) && l > 1e-12) ? n / l : -normalize(view);
    if (dot(N, view) > 0.0) N = -N;
    eps = hybrid_eps(z) + Prm[0].depth_step * z * z;
    return true;
}

// Adds this batch's mean `v` into the running mean in `b` at pixel j.
void hybrid_accumulate(RWStructuredBuffer<float4> b, uint j, float v) {
    float n = float(K.frame);
    float r = v;
    if (K.frame > 0u) {
        r = (packed_get(b, j) * n + v) / (n + 1.0);
    }
    packed_set(b, j, isfinite(r) ? r : 1.0);
}

// Receivers whose game depth and our primary hit differ by more than this are rejected.
float hybrid_snap_tolerance(float z) { return max(0.05, 0.03 * z); }

// photo_hybrid_prep (once per frame, before the terms; in place on u1, normal into u2): our
// primary hit through the pixel centre (the frame's jitter, alpha test as photo_depth).
// Where it agrees with the game's depth (either side, within hybrid_snap_tolerance: a
// coarser LOD of ours, plank gaps, decals) the receiver becomes our hit, with that
// triangle's normal facing the camera: the shadow ray then starts from our own surface
// along its own normal, never inside our geometry or along a normal rebuilt across a depth
// edge (O, hybrid_check RX010: the depth-rebuilt normals of the isolated black pixels
// disagreed with the path tracer's, median cos 0.36). Otherwise (characters and objects we
// lack, geometry only we have, no hit) the pixel is rejected: -z, no normal; the sun term
// writes 1 there (min() keeps the engine's CSM) and AO falls back to the game depth.
// Non-finite or negative depths become sky.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_hybrid_prep(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    uint j = px.y * depth_row(K.width) + px.x;
    float z = packed_get(AccAlb, j);
    packed_set(AccNrm, j, 0.0);
    if (!(isfinite(z) && z > 0.0)) {
        packed_set(AccAlb, j, 0.0);
        return;
    }
    float3 d = hybrid_ray(float2(px) + 0.5);
    RayDesc r;
    r.Origin = Prm[0].org;
    r.Direction = normalize(d);
    r.TMin = 0.0;
    r.TMax = 1e5;
    uint seed = seed_of(px, 0x7fffffffu);
    Hit h = trace(r, 0.0, Prm[0].pixel_spread, seed);
    float zr = h.hit ? h.t / length(d) : 0.0;
    if (h.hit && isfinite(zr) && zr > 0.0 && abs(zr - z) <= hybrid_snap_tolerance(z)) {
        Tri t = tri_at(h.loc, h.bary, true);
        float3 n = world_normal((float3x3)h.o2w, t.n_obj);
        if (dot(n, r.Direction) > 0.0) n = -n;
        packed_set(AccAlb, j, zr);
        packed_set(AccNrm, j, hybrid_pack_normal(n));
    } else {
        packed_set(AccAlb, j, -z);
    }
}

// photo_sun_vis: K.step shadow rays per pixel toward points of the sun disc (alpha-tested
// foliage through the candidate loop of `visible`, decals never block) from the prepass
// receiver, offset along its normal. Receivers facing away from the sun, rejected receivers
// and sky count as visible: N.L and the engine's CSM (kept by min()) darken them.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_sun_vis(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    float3 P, N;
    float eps;
    float v = 1.0;
    if (hybrid_receiver(int2(px), P, N, eps) && dot(N, Prm[0].sun_dir) > 0.0) {
        uint seed = seed_of(px, K.frame * 2u + 0x51ED27u);
        uint n = clamp(K.step, 1u, 4u);
        float3 o = hybrid_origin(P, N, eps);
        float lit = 0.0;
        for (uint k = 0; k < n; k++) {
            float3 l = sun_sample(seed);
            lit += (dot(N, l) <= 0.0 || visible(o, l, 1e5)) ? 1.0 : 0.0;
        }
        v = lit / float(n);
    }
    hybrid_accumulate(BufA, px.y * depth_row(K.width) + px.x, v);
}

// photo_ao: K.step cosine-distributed rays per pixel up to ao_radius metres; visibility =
// the fraction that escapes (alpha test as above). Rejected receivers use the game depth.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_ao(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    float3 P, N;
    float eps;
    float v = 1.0;
    float radius = Prm[0].ao_radius;
    if (radius > 0.0
        && (hybrid_receiver(int2(px), P, N, eps) || hybrid_surface(int2(px), P, N, eps))) {
        uint seed = seed_of(px, K.frame * 2u + 0xA0761Du);
        uint n = clamp(K.step, 1u, 4u);
        float3 o = hybrid_origin(P, N, eps);
        float open = 0.0;
        for (uint k = 0; k < n; k++) {
            open += visible(o, cosine_dir(N, seed), radius) ? 1.0 : 0.0;
        }
        v = open / float(n);
    }
    hybrid_accumulate(BufB, px.y * depth_row(K.width) + px.x, v);
}

// photo_hybrid_filter (before each copy of the sun term; u3 -> u0): 3x3 mean over the
// neighbours on the receiver's own plane (normals within ~25 degrees, plane distance below
// the ray offset), centre weight 2. Removes isolated black pixels and one-pixel dashes along
// edges and the residual noise, keeps shadow edges to about a pixel; never mixes across a
// depth or normal edge. Pixels without a receiver (sky, rejected) pass through.
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void photo_hybrid_filter(uint3 id : SV_DispatchThreadID) {
    uint2 px = uint2(id.x, id.y + K.y0);
    if (px.x >= K.width || px.y >= K.height) return;
    int2 p = int2(px);
    uint j = px.y * depth_row(K.width) + px.x;
    float v = packed_get(BufA, j);
    float3 P, N;
    float eps;
    float r = v;
    if (hybrid_receiver(p, P, N, eps)) {
        float sum = 2.0 * v, w = 2.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 q = p + int2(dx, dy);
                if ((dx == 0 && dy == 0) || any(q != hybrid_clamp(q))) continue;
                float3 Pq, Nq;
                float eq;
                if (hybrid_receiver(q, Pq, Nq, eq) && dot(Nq, N) > 0.9
                    && abs(dot(Pq - P, N)) < eps) {
                    sum += packed_get(BufA, hybrid_index(q));
                    w += 1.0;
                }
            }
        }
        r = sum / w;
    }
    packed_set(AccRad, j, isfinite(r) ? saturate(r) : 1.0);
}
