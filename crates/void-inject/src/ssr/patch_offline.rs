//! Offline checks of the material patch (`examples/ssr_patch_offline.rs`):
//! 1. every shipped pixel shader of a DXBC dump directory (`d2enhanced/captures/shaders`,
//!    read only) classified (`patch::shader`) and the candidates through `patch::patch`:
//!    counts per class (material patched / excluded-lit / excluded-unlit / unpatched / not in
//!    the forward passes), per F0 source of `o3` and skip reasons; each patched shader created
//!    on a real device (`CreatePixelShader`) and its disassembly compared with the original's
//!    after removing the inserted instructions (the new temporaries, `o2` and `o3`): the
//!    original program must be unchanged;
//! 2. an end-to-end GPU test: a pixel shader compiled from HLSL with the engine's IBL code
//!    (`ibl.decl` `HQ_ComputeIBL`: reflect/lerp direction, mip formula, coloured F0 from a
//!    per-channel IOR (dielectric to gold-like), luminance-factor lerp) is patched and drawn
//!    next to the original: `o0` must be bit-identical, `o2` must decode to the normal,
//!    roughness and F0 luminance the shader used, `o3` to its sqrt(F0.rgb) with alpha 1; the
//!    same with a vegetation cbuffer variable (with and without the IBL code): `o2.a` must be
//!    byte 1 (exclusion marker), `o3` 0;
//! 3. the GPU cost of the targets at 3440x1440: full-screen layers of the patched test shader
//!    with `o0` only, `o0 + o2` and `o0 + o2 + o3` bound.

use std::path::Path;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D::Fxc::D3DDisassemble;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use super::patch;
use crate::ao::create_err;
use crate::ao::offline::Gpu;
use crate::hlsl;

pub struct PatchReport {
    pub text: String,
    pub failures: Vec<String>,
}

fn disasm(code: &[u8]) -> Result<Vec<String>, String> {
    // SAFETY: `code` is a complete DXBC container of the given length.
    let blob = unsafe { D3DDisassemble(code.as_ptr().cast(), code.len(), 0, PCSTR::null()) }
        .map_err(create_err("D3DDisassemble"))?;
    // SAFETY: the blob owns `GetBufferSize()` bytes at `GetBufferPointer()`.
    let text = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    Ok(String::from_utf8_lossy(text)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//") && *l != "\0")
        .map(str::to_owned)
        .collect())
}

/// Disassembly lines of the original program in the patched shader: without `dcl_temps`,
/// the added `dcl_output o2/o3` and every instruction naming the new temporaries, `o2` or
/// `o3`.
fn strip_inserted(lines: &[String], first_temp: u32) -> Vec<String> {
    let ours: Vec<String> = (first_temp..first_temp + patch::NEW_TEMPS)
        .map(|r| format!("r{r}."))
        .collect();
    lines
        .iter()
        .filter(|l| {
            !l.starts_with("dcl_temps") && !contains_reg(l, "o2.") && !contains_reg(l, "o3.")
        })
        .filter(|l| !ours.iter().any(|o| contains_reg(l, o)))
        .cloned()
        .collect()
}

/// `l` names register `o` ("r12.") as a whole token (not "r112.").
fn contains_reg(l: &str, o: &str) -> bool {
    l.match_indices(o)
        .any(|(i, _)| i == 0 || !l.as_bytes()[i - 1].is_ascii_alphanumeric())
}

fn create_ps(dev: &ID3D11Device, code: &[u8]) -> Result<ID3D11PixelShader, String> {
    let mut ps = None;
    // SAFETY: bytecode slice and valid out pointer.
    unsafe { dev.CreatePixelShader(code, None, Some(&mut ps)) }
        .map_err(create_err("CreatePixelShader"))?;
    ps.ok_or_else(|| "no pixel shader".to_owned())
}

/// Part 1 (see the module documentation).
fn survey(
    g: &Gpu,
    dir: &Path,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("ps_") && n.ends_with(".dxbc"))
        })
        .collect();
    files.sort();
    let (mut total, mut lit, mut cand, mut patched, mut created, mut identical) =
        (0, 0, 0, 0, 0, 0);
    let mut f0: std::collections::BTreeMap<patch::F0Source, usize> = Default::default();
    let mut skipped: std::collections::BTreeMap<&'static str, usize> = Default::default();
    // Per class: dumps, forward, patched with the captures, marker only, unpatched candidates.
    let mut classes: std::collections::BTreeMap<patch::Class, [usize; 5]> = Default::default();
    let mut inserted = 0usize;
    let codes: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|p| {
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned();
            std::fs::read(p)
                .map(|c| (name, c))
                .map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect::<Result<_, _>>()?;
    // What the CreatePixelShader hook costs: every shader classified, the candidates patched,
    // timed without the validation below.
    let t = Instant::now();
    let results: Vec<_> = codes
        .iter()
        .map(|(_, c)| {
            let s = patch::shader(c);
            (s, s.candidate().then(|| patch::patch(c, &s)))
        })
        .collect();
    let patch_ms = t.elapsed().as_secs_f64() * 1000.0;
    for ((name, code), (s, r)) in codes.iter().zip(results) {
        total += 1;
        let row = classes.entry(s.class).or_default();
        row[0] += 1;
        row[1] += usize::from(s.forward);
        lit += usize::from(s.forward && s.ibl.is_some());
        let Some(r) = r else { continue };
        cand += 1;
        let code = code.as_slice();
        let p = match r {
            Ok(p) => p,
            Err(e) => {
                row[4] += 1;
                *skipped.entry(e).or_default() += 1;
                continue;
            }
        };
        row[if p.captured { 2 } else { 3 }] += 1;
        patched += 1;
        if let Some(k) = p.f0 {
            *f0.entry(k).or_default() += 1;
        }
        inserted += p.inserted;
        let name = name.as_str();
        match create_ps(&g.dev, &p.code) {
            Ok(_) => created += 1,
            Err(e) => {
                failures.push(format!("{name}: patched shader rejected: {e}"));
                continue;
            }
        }
        let (a, b) = (disasm(code)?, disasm(&p.code)?);
        let a: Vec<String> = a
            .into_iter()
            .filter(|l| !l.starts_with("dcl_temps"))
            .collect();
        if strip_inserted(&b, p.first_temp) == a {
            identical += 1;
        } else {
            failures.push(format!("{name}: original program changed by the patch"));
        }
    }
    report.push(format!(
        "shader dumps {total}, candidates {cand} (lit forward = tlf + IBL, no fx: {lit}; plus forward shaders of excluded classes without IBL): patched {patched}, created on the device {created}, original program unchanged {identical}; {:.1} inserted instructions per shader; all {total} dumps through the classifier and patcher in {patch_ms:.0} ms ({:.3} ms per candidate)",
        inserted as f64 / patched.max(1) as f64,
        patch_ms / cand.max(1) as f64
    ));
    report.push(format!(
        "  F0 (o3) of the {} shaders with the captures: {}",
        f0.values().sum::<usize>(),
        f0.iter()
            .map(|(k, n)| format!(
                "{} {n}",
                match k {
                    patch::F0Source::Rgb => "RGB vector (dp3 operand)",
                    patch::F0Source::Constant => "folded constant (grey, exact)",
                    patch::F0Source::Luminance => "luminance only",
                    patch::F0Source::Default => "none (0.04)",
                }
            ))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for (class, [n, fwd, cap, marker, failed]) in &classes {
        let (cap_kind, marker_kind) = if class.excluded() {
            (
                "excluded-lit (captures, alpha 1/255)",
                "excluded-unlit (marker only)",
            )
        } else {
            ("patched (material)", "marker only")
        };
        report.push(format!(
            "  {:<10} {n:>5} dumps, {fwd:>4} forward: {cap_kind} {cap}, {marker_kind} {marker}, unpatched {failed}, left alone (forward without IBL) {}; not forward (other passes, blended fx) {}",
            class.name(),
            fwd - cap - marker - failed,
            n - fwd
        ));
    }
    for (reason, n) in &skipped {
        report.push(format!("  skipped {n}: {reason}"));
    }
    if patched == 0 {
        failures.push("no shader patched".into());
    }
    if !classes.iter().any(|(c, r)| c.excluded() && r[2] + r[3] > 0) {
        failures.push("no excluded-class shader patched".into());
    }
    Ok(())
}

const MIMIC_HLSL: &str = r"
TextureCubeArray<float4> ro_ibl_texcubemaparray : register(t6);
Texture2D<uint> ro_tlf_ll_heads : register(t2);
SamplerState smp_iblsampler : register(s4);
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; float2 ibl : TEXCOORD1; };

V vs(uint id : SV_VertexID)
{
    V o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.pos = float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = t;
    o.ibl = float2(0.0, 0.35);
    return o;
}

// The normal, roughness and IOR the shader uses at uv (the CPU check repeats this).
float3 TestNormal(float2 uv) { float2 s = uv * 2.0 - 1.0; return normalize(float3(s.x, 0.6, s.y * 0.8 - 0.1)); }

float4 ps(V i) : SV_Target0
{
    float3 N = TestNormal(i.uv);
    float3 Vw = normalize(float3(0.3, 0.8, 0.5));
    float rough = i.uv.x;
    float m = rough * rough;
    float3 R = reflect(-Vw, N);
    float3 L = normalize(lerp(R, N, m));
    float lobe = 0.0003474660443456835 + m * (1.3331290497744692 - m * 0.5040552688878546);
    float mip = 6.0 * sqrt(lobe / 0.834);
    float3 rad = ro_ibl_texcubemaparray.SampleLevel(smp_iblsampler, float4(L, 0.0), mip).xyz;
    // Coloured F0 the engine's way (`HQ_FetchMetallicSpecularFresnelAndIOR`: IOR from
    // sqrt(F0)): dielectric 0.04 at uv.y = 0 to gold-like (1.0, 0.78, 0.34) at uv.y = 1.
    float3 spec = sqrt(lerp(0.04, float3(1.0, 0.78, 0.34), i.uv.y));
    float3 ior = (1.0 + spec) / (1.00001 - spec);
    float3 r0 = (ior - 1.0) / (ior + 1.0);
    float3 F0 = r0 * r0;
    float f = lerp(abs(i.ibl.y), 1.0, dot(float3(0.2126, 0.7152, 0.0722), F0));
    uint heads = ro_tlf_ll_heads.Load(int3(0, 0, 0));
    return float4(rad * f + heads * 1e-9, 1.0);
}

// Excluded class by its cbuffer variable (as `arkVegetation`'s wind parameters; unbound: 0).
cbuffer PerInstanceCB : register(b2) { float4 cb_vegetanim_leaffreq; };

float4 ps_veg(V i) : SV_Target0 { return ps(i) + cb_vegetanim_leaffreq * 1e-9; }

float4 ps_veg_unlit(V i) : SV_Target0
{
    uint heads = ro_tlf_ll_heads.Load(int3(0, 0, 0));
    return float4(i.uv, heads * 1e-9, 1.0) + cb_vegetanim_leaffreq * 1e-9;
}
";

fn test_normal(uv: [f32; 2]) -> [f32; 3] {
    let s = [uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0];
    let n = [s[0], 0.6, s[1] * 0.8 - 0.1];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / l)
}

/// The F0 the test shader computes at `uv` (per channel, as `Fresnel_F0FromIOR`).
fn test_f0(uv: [f32; 2]) -> [f32; 3] {
    let gold = [1.0f32, 0.78, 0.34];
    gold.map(|g| {
        let s = (0.04 + (g - 0.04) * uv[1]).sqrt();
        let ior = (1.0 + s) / (1.00001 - s);
        ((ior - 1.0) / (ior + 1.0)).powi(2)
    })
}

/// Octahedral encoding as the patch writes it (0..1).
pub fn oct_encode(n: [f32; 3]) -> [f32; 2] {
    let l1 = (n[0].abs() + n[1].abs() + n[2].abs()).max(1e-6);
    let p = [n[0] / l1, n[1] / l1, n[2] / l1];
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let xy = if p[2] < 0.0 {
        [
            (1.0 - p[1].abs()) * sign(p[0]),
            (1.0 - p[0].abs()) * sign(p[1]),
        ]
    } else {
        [p[0], p[1]]
    };
    [xy[0] * 0.5 + 0.5, xy[1] * 0.5 + 0.5]
}

fn texture(
    g: &Gpu,
    desc: &D3D11_TEXTURE2D_DESC,
    data: Option<&[D3D11_SUBRESOURCE_DATA]>,
) -> Result<ID3D11Texture2D, String> {
    let mut t = None;
    // SAFETY: valid descriptor; initial data (if any) covers every subresource.
    unsafe {
        g.dev
            .CreateTexture2D(desc, data.map(|d| d.as_ptr()), Some(&mut t))
    }
    .map_err(create_err("texture"))?;
    t.ok_or_else(|| "no texture".to_owned())
}

fn desc2d(size: [u32; 2], format: DXGI_FORMAT, bind: D3D11_BIND_FLAG) -> D3D11_TEXTURE2D_DESC {
    D3D11_TEXTURE2D_DESC {
        Width: size[0],
        Height: size[1],
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind.0 as u32,
        ..Default::default()
    }
}

/// A render target and its view.
fn target(
    g: &Gpu,
    size: [u32; 2],
    format: DXGI_FORMAT,
) -> Result<(ID3D11Texture2D, ID3D11RenderTargetView), String> {
    let t = texture(g, &desc2d(size, format, D3D11_BIND_RENDER_TARGET), None)?;
    let mut v = None;
    // SAFETY: view of this device's texture, valid out pointer.
    unsafe { g.dev.CreateRenderTargetView(&t, None, Some(&mut v)) }.map_err(create_err("RTV"))?;
    Ok((t, v.ok_or("no RTV")?))
}

/// Inputs of the test shaders: full-screen VS, IBL cube array, `ro_tlf_ll_heads`, sampler.
struct Inputs {
    vs: ID3D11VertexShader,
    cube: ID3D11ShaderResourceView,
    heads: ID3D11ShaderResourceView,
    smp: ID3D11SamplerState,
}

impl Inputs {
    fn new(g: &Gpu) -> Result<Self, String> {
        let vs_code = hlsl::compile(MIMIC_HLSL, s!("vs"), s!("vs_5_0"))?;
        let cube_texels = vec![0x3C00_3800u64; 16 * 16]; // RGBA16F (0.5, 1.0, 0, 0)
        let cube_data: Vec<D3D11_SUBRESOURCE_DATA> = (0..6)
            .map(|_| D3D11_SUBRESOURCE_DATA {
                pSysMem: cube_texels.as_ptr().cast(),
                SysMemPitch: 16 * 8,
                SysMemSlicePitch: 0,
            })
            .collect();
        let cube = texture(
            g,
            &D3D11_TEXTURE2D_DESC {
                ArraySize: 6,
                MiscFlags: D3D11_RESOURCE_MISC_TEXTURECUBE.0 as u32,
                ..desc2d(
                    [16, 16],
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    D3D11_BIND_SHADER_RESOURCE,
                )
            },
            Some(&cube_data),
        )?;
        let zero = [0u32; 1];
        let heads = texture(
            g,
            &desc2d([1, 1], DXGI_FORMAT_R32_UINT, D3D11_BIND_SHADER_RESOURCE),
            Some(&[D3D11_SUBRESOURCE_DATA {
                pSysMem: zero.as_ptr().cast(),
                SysMemPitch: 4,
                SysMemSlicePitch: 0,
            }]),
        )?;
        let (mut vs, mut cs, mut hs, mut smp) = (None, None, None, None);
        // SAFETY (whole block): creation of this device's objects with valid descriptors and
        // out pointers.
        unsafe {
            g.dev
                .CreateVertexShader(&vs_code, None, Some(&mut vs))
                .map_err(create_err("VS"))?;
            g.dev
                .CreateShaderResourceView(
                    &cube,
                    Some(&D3D11_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        ViewDimension: D3D11_SRV_DIMENSION_TEXTURECUBEARRAY,
                        Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                            TextureCubeArray: D3D11_TEXCUBE_ARRAY_SRV {
                                MostDetailedMip: 0,
                                MipLevels: 1,
                                First2DArrayFace: 0,
                                NumCubes: 1,
                            },
                        },
                    }),
                    Some(&mut cs),
                )
                .map_err(create_err("cube SRV"))?;
            g.dev
                .CreateShaderResourceView(&heads, None, Some(&mut hs))
                .map_err(create_err("heads SRV"))?;
            g.dev
                .CreateSamplerState(
                    &D3D11_SAMPLER_DESC {
                        Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                        AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    },
                    Some(&mut smp),
                )
                .map_err(create_err("sampler"))?;
        }
        Ok(Inputs {
            vs: vs.ok_or("no VS")?,
            cube: cs.ok_or("no cube SRV")?,
            heads: hs.ok_or("no heads SRV")?,
            smp: smp.ok_or("no sampler")?,
        })
    }

    /// Full-screen draw state of `ps` for a `size` viewport on `ctx` (targets not set).
    fn bind(&self, ctx: &ID3D11DeviceContext, ps: &ID3D11PixelShader, size: [u32; 2]) {
        // SAFETY (whole block): state of this device's objects on `ctx`.
        unsafe {
            ctx.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.VSSetShader(&self.vs, None);
            ctx.PSSetShader(ps, None);
            ctx.PSSetShaderResources(2, Some(&[Some(self.heads.clone())]));
            ctx.PSSetShaderResources(6, Some(&[Some(self.cube.clone())]));
            ctx.PSSetSamplers(4, Some(&[Some(self.smp.clone())]));
            ctx.RSSetViewports(Some(&[D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: size[0] as f32,
                Height: size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]));
        }
    }
}

/// Bytes of `t` (`bpp` bytes per texel, `size` x `size` texels) through a staging copy.
fn read_back(g: &Gpu, t: &ID3D11Texture2D, size: u32, bpp: usize) -> Result<Vec<u8>, String> {
    let mut d = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: plain getter.
    unsafe { t.GetDesc(&mut d) };
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    let st = texture(g, &d, None)?;
    // SAFETY (whole block): staging copy, mapped for reading within its pitch.
    unsafe {
        g.imm.CopyResource(&st, t);
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        g.imm
            .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
            .map_err(create_err("Map"))?;
        let mut out = Vec::with_capacity((size * size) as usize * bpp);
        for y in 0..size as usize {
            let row = m.pData.cast::<u8>().add(y * m.RowPitch as usize);
            out.extend_from_slice(std::slice::from_raw_parts(row, size as usize * bpp));
        }
        g.imm.Unmap(&st, 0);
        Ok(out)
    }
}

/// Texel `(x, y)` of an `s`-wide RGBA8 read-back, 0..1.
fn rgba8(px: &[u8], s: u32, x: u32, y: u32) -> [f32; 4] {
    let p = &px[((y * s + x) * 4) as usize..][..4];
    std::array::from_fn(|k| f32::from(p[k]) / 255.0)
}

/// Part 2 (see the module documentation). Returns the patched test shader.
fn end_to_end(
    g: &Gpu,
    inp: &Inputs,
    report: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> Result<Vec<u8>, String> {
    const S: u32 = 64;
    let ps_code = hlsl::compile(MIMIC_HLSL, s!("ps"), s!("ps_5_0"))?;
    let sh = patch::shader(&ps_code);
    if sh.class != patch::Class::Surface {
        return Err(format!("test shader classified {:?}", sh.class));
    }
    let p = patch::patch(&ps_code, &sh).map_err(|e| format!("test shader not patched: {e}"))?;
    report.push(format!(
        "end-to-end test shader: patched, F0 capture {:?}, {} instructions inserted",
        p.f0, p.inserted
    ));
    // o0, o2, o3 of a full-screen draw of `code` (o3 cleared to 0.5 so "not written" shows).
    let run = |code: &[u8]| -> Result<[Vec<u8>; 3], String> {
        let (rt0, v0) = target(g, [S, S], DXGI_FORMAT_R16G16B16A16_FLOAT)?;
        let (rt2, v2) = target(g, [S, S], DXGI_FORMAT_R8G8B8A8_UNORM)?;
        let (rt3, v3) = target(g, [S, S], DXGI_FORMAT_R8G8B8A8_UNORM)?;
        let ps = create_ps(&g.dev, code)?;
        let ctx = &g.imm;
        // SAFETY (whole block): this device's objects on the immediate context; read back
        // below.
        unsafe {
            ctx.ClearState();
            ctx.ClearRenderTargetView(&v0, &[0.0; 4]);
            ctx.ClearRenderTargetView(&v2, &[0.0; 4]);
            ctx.ClearRenderTargetView(&v3, &[0.5; 4]);
            inp.bind(ctx, &ps, [S, S]);
            ctx.OMSetRenderTargets(Some(&[Some(v0), None, Some(v2), Some(v3)]), None);
            ctx.Draw(3, 0);
            ctx.ClearState();
        }
        Ok([
            read_back(g, &rt0, S, 8)?,
            read_back(g, &rt2, S, 4)?,
            read_back(g, &rt3, S, 4)?,
        ])
    };
    let [orig0, ..] = run(&ps_code)?;
    let [pat0, pat2, pat3] = run(&p.code)?;
    let same = orig0 == pat0;
    let (mut worst_n, mut worst_r, mut worst_f, mut worst_c) = (0f32, 0f32, 0f32, 0f32);
    let mut kind_ok = 0;
    for y in 0..S {
        for x in 0..S {
            let uv = [(x as f32 + 0.5) / S as f32, (y as f32 + 0.5) / S as f32];
            let got = rgba8(&pat2, S, x, y);
            let f0 = test_f0(uv);
            let want_n = oct_encode(test_normal(uv));
            let f0l = 0.2126 * f0[0] + 0.7152 * f0[1] + 0.0722 * f0[2];
            worst_n = worst_n
                .max((got[0] - want_n[0]).abs())
                .max((got[1] - want_n[1]).abs());
            worst_r = worst_r.max((got[2] - uv[0]).abs());
            worst_f = worst_f.max((got[3] - f0l.sqrt().max(patch::MATERIAL_ALPHA_MIN)).abs());
            let c = rgba8(&pat3, S, x, y);
            for k in 0..3 {
                worst_c = worst_c.max((c[k] - f0[k].sqrt()).abs());
            }
            kind_ok += usize::from(c[3] == patch::F0_RGB_ALPHA);
        }
    }
    let gold = rgba8(&pat3, S, S / 2, S - 1).map(|v| v * v);
    report.push(format!(
        "  o0 original vs patched bit-identical {same}; o2 max error: octahedral normal {worst_n:.4}, roughness {worst_r:.4}, sqrt(F0) {worst_f:.4}; o3 max error sqrt(F0.rgb) {worst_c:.4}, alpha 1 (RGB) at {kind_ok} / {} pixels, F0 decoded at the gold end ({:.3}, {:.3}, {:.3}) (8-bit target: 0.0039 per step)",
        S * S,
        gold[0],
        gold[1],
        gold[2]
    ));
    let tol = 2.5 / 255.0;
    if !same {
        failures.push("end-to-end: o0 changed by the patch".into());
    }
    if p.f0 != Some(patch::F0Source::Rgb) {
        failures.push(format!("end-to-end: F0 vector not captured ({:?})", p.f0));
    }
    if worst_n > tol || worst_r > tol || worst_f > tol {
        failures.push(format!(
            "end-to-end: o2 off (normal {worst_n}, roughness {worst_r}, F0 {worst_f})"
        ));
    }
    if worst_c > tol || kind_ok != (S * S) as usize {
        failures.push(format!(
            "end-to-end: o3 off (sqrt F0.rgb {worst_c}, RGB alpha at {kind_ok} pixels)"
        ));
    }
    // Excluded class (vegetation by its cbuffer variable): with the IBL captures (normal and
    // roughness as above) and marker only (no IBL); o2 alpha must be byte 1 everywhere, o3 0.
    for (entry, captured) in [(s!("ps_veg"), true), (s!("ps_veg_unlit"), false)] {
        let code = hlsl::compile(MIMIC_HLSL, entry, s!("ps_5_0"))?;
        let sh = patch::shader(&code);
        let p = patch::patch(&code, &sh).map_err(|e| format!("excluded test shader: {e}"))?;
        let [orig0, ..] = run(&code)?;
        let [pat0, pat2, pat3] = run(&p.code)?;
        let (mut worst_n, mut worst_r, mut alpha_ok) = (0f32, 0f32, 0usize);
        for y in 0..S {
            for x in 0..S {
                let uv = [(x as f32 + 0.5) / S as f32, (y as f32 + 0.5) / S as f32];
                let got = rgba8(&pat2, S, x, y);
                let (want_n, rough) = if captured {
                    (oct_encode(test_normal(uv)), uv[0])
                } else {
                    ([0.5, 0.5], 1.0)
                };
                worst_n = worst_n
                    .max((got[0] - want_n[0]).abs())
                    .max((got[1] - want_n[1]).abs());
                worst_r = worst_r.max((got[2] - rough).abs());
                alpha_ok += usize::from(got[3] == patch::EXCLUDED_ALPHA);
            }
        }
        let o3_zero = pat3.iter().all(|&b| b == 0);
        let same = orig0 == pat0;
        report.push(format!(
            "  excluded-class test shader ({}): class {}, captures {}, {} instructions inserted; o0 bit-identical {same}; o2 alpha byte 1 at {alpha_ok} / {} pixels, max error normal {worst_n:.4}, roughness {worst_r:.4}; o3 all 0 {o3_zero}",
            if captured { "IBL" } else { "no IBL" },
            sh.class.name(),
            p.captured,
            p.inserted,
            S * S
        ));
        if sh.class != patch::Class::Vegetation || p.captured != captured {
            failures.push(format!(
                "end-to-end: excluded test shader classified {:?}, captures {}",
                sh.class, p.captured
            ));
        }
        if !same {
            failures.push("end-to-end: o0 changed by the exclusion patch".into());
        }
        if alpha_ok != (S * S) as usize || worst_n > tol || worst_r > tol || !o3_zero {
            failures.push(format!(
                "end-to-end: excluded o2/o3 off (alpha byte 1 at {alpha_ok} pixels, normal {worst_n}, roughness {worst_r}, o3 zero {o3_zero})"
            ));
        }
    }
    Ok(p.code)
}

/// Part 3: GPU cost of the material targets at 3440x1440 (one full-screen layer of the patched
/// test shader per draw, `DRAWS` draws per timed list): `o0` only, `o0 + o2`, `o0 + o2 + o3`.
fn target_cost(g: &Gpu, inp: &Inputs, code: &[u8], report: &mut Vec<String>) -> Result<(), String> {
    const SIZE: [u32; 2] = [3440, 1440];
    const DRAWS: u32 = 32;
    let ps = create_ps(&g.dev, code)?;
    let (_t0, v0) = target(g, SIZE, DXGI_FORMAT_R11G11B10_FLOAT)?;
    let (_t2, v2) = target(g, SIZE, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    let (_t3, v3) = target(g, SIZE, DXGI_FORMAT_R8G8B8A8_UNORM)?;
    let sets = [
        vec![Some(v0.clone())],
        vec![Some(v0.clone()), None, Some(v2.clone())],
        vec![Some(v0), None, Some(v2), Some(v3.clone())],
    ];
    let ctx = &g.def;
    let finish = || -> Result<ID3D11CommandList, String> {
        let mut list = None;
        // SAFETY: recording of this device's deferred context, valid out pointer.
        unsafe { ctx.FinishCommandList(false, Some(&mut list)) }
            .map_err(create_err("FinishCommandList"))?;
        list.ok_or_else(|| "no command list".to_owned())
    };
    // Three interleaved rounds, the fastest median per variant (thermal and clock drift).
    let mut ms = [f64::INFINITY; 3];
    let mut clear_ms = f64::INFINITY;
    for _ in 0..3 {
        for (k, rtvs) in sets.iter().enumerate() {
            inp.bind(ctx, &ps, SIZE);
            // SAFETY: this device's views on the deferred context.
            unsafe {
                ctx.OMSetRenderTargets(Some(rtvs), None);
                for _ in 0..DRAWS {
                    ctx.Draw(3, 0);
                }
            }
            ms[k] = ms[k].min(g.time(&finish()?)? / f64::from(DRAWS));
        }
        // SAFETY: this device's view on the deferred context.
        unsafe {
            for _ in 0..DRAWS {
                ctx.ClearRenderTargetView(&v3, &[0.0; 4]);
            }
        }
        clear_ms = clear_ms.min(g.time(&finish()?)? / f64::from(DRAWS));
    }
    let mb = f64::from(SIZE[0] * SIZE[1] * 4) / 1e6;
    report.push(format!(
        "target cost at {}x{} (patched test shader, per full-screen layer, fastest of 3 interleaved medians): o0 only {:.4} ms, + o2 {:.4} ms ({:+.4}), + o2 + o3 {:.4} ms ({:+.4} for o3); clear of one RGBA8 target {clear_ms:.4} ms; each RGBA8 target: {mb:.1} MB written per covered layer plus {mb:.1} MB cleared per frame",
        SIZE[0],
        SIZE[1],
        ms[0],
        ms[1],
        ms[1] - ms[0],
        ms[2],
        ms[2] - ms[1]
    ));
    Ok(())
}

/// See the module documentation. `shader_dir` holds `ps_*.dxbc` dumps.
pub fn patch_offline_check(shader_dir: &Path) -> Result<PatchReport, String> {
    let g = Gpu::new()?;
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let inp = Inputs::new(&g)?;
    let patched = end_to_end(&g, &inp, &mut report, &mut failures)?;
    target_cost(&g, &inp, &patched, &mut report)?;
    survey(&g, shader_dir, &mut report, &mut failures)?;
    if failures.is_empty() {
        report.push("all checks passed".into());
    }
    Ok(PatchReport {
        text: report.join("\n"),
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octahedral_encoding() {
        assert_eq!(oct_encode([0.0, 0.0, 1.0]), [0.5, 0.5]);
        assert_eq!(oct_encode([1.0, 0.0, 0.0]), [1.0, 0.5]);
        let e = oct_encode([0.0, 0.0, -1.0]);
        assert!(
            (e[0] - 1.0).abs() < 1e-6 && (e[1] - 1.0).abs() < 1e-6,
            "{e:?}"
        );
        assert!(contains_reg("mov r12.x, r1.y", "r12."));
        assert!(!contains_reg("mov r112.x, r1.y", "r12."));
    }
}
