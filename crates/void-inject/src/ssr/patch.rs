//! DXBC patch of the game's lit forward pixel shaders: two more render targets (`o2`, `o3`,
//! R8G8B8A8_UNORM, slots 2 and 3) with the per-pixel material data the reflections need
//! (docs/reflections.md "Material target").
//!
//! Capture points (shipped `ibl.decl` `HQ/AQ/LQ_ComputeIBL`, corroborated in the shader
//! cache dumps): the IBL cube-array `sample_l` (resource `ro_ibl_texcubemaparray`) takes its
//! LOD from `6 * sqrt((0.000347 + m * (1.333129 - m * 0.504055)) * 1.199041)` with `m =
//! roughness^2` (perceptual roughness squared, `ComputeIBL`); the sampling direction is
//! `normalize(lerp(reflect(-V, N), N, m))`, compiled to `mad R, N, -t, -V` (reflect) and
//! `mad L, m, A, R` (lerp), which yields the shading normal N after normal mapping; after the
//! sample `lerp(IBLLuminanceFactor, 1, F0)` is `mad X, F0, 1 - |v.c|, |v.c|`, which yields the
//! luminance of F0 (a register or a folded constant). The luminance comes from `dp3 X,
//! l(0.2126, 0.7152, 0.0722, 0), F0.xyz` (`Fresnel_F0FromIOR(IOR)` per channel) where the
//! IOR is not constant; a folded constant is grey (the shader uses it as a scalar). The patch
//! copies m (before the `0.504055` mad), N (before the reflect), F0 (before that mad) and
//! the F0 vector (before the dp3) into new temporaries and writes, before every `ret`/`retc`:
//! `o2 = (octahedral(N) * 0.5 + 0.5, sqrt(m), m captured ? max(sqrt(F0), 2/255) : 0)`,
//! `o3 = m captured ? (sqrt(F0.rgb), 1 if the vector or the constant was captured else
//! 128/255 with sqrt(F0 luminance) grey) : 0`.
//!
//! Classes ([`Class`], by RDEF names only: binding names and cbuffer variable names, which
//! list every parameter of the render program, vertex stage included): vegetation
//! (`arkVegetation`: `cb_vegetanim_*`, `cb_vegetation_*`), hair (`arkHair` / `BRDF_hair`:
//! `cb_hair*`, `ro_hair*`) and box-projected decals (`arkDecal`: `cb_decalboxprojection`, ...)
//! are not reflection receivers: their forward shaders write the exclusion marker alpha 1/255
//! instead of sqrt(F0) (with the IBL capture), or only `o2 = (0.5, 0.5, 1, 1/255)` (without
//! it), and `o3 = 0` (under `arkDecal`'s `SRC_ALPHA` blend: the surface's value stays).
//! Particles (`ro_fx_*` inputs) are blended through the render target's blend state and are
//! never patched.
//!
//! Nothing the original program computes changes: the inserted instructions only write the new
//! temporaries, `o2` and `o3` (checked offline against the disassembly of every patched
//! shader).

/// Render-target slot of the material output (slot 1 holds `sssBuffer` / `skyCloudMask`).
pub const SLOT: u32 = 2;
/// `o2.a` of the excluded classes (byte 1): no reflection, no depth-normal fallback.
pub const EXCLUDED_ALPHA: f32 = 1.0 / 255.0;
/// Smallest `o2.a` of a material (byte 2): `max(sqrt(F0), MATERIAL_ALPHA_MIN)`.
pub const MATERIAL_ALPHA_MIN: f32 = 2.0 / 255.0;
/// Render-target slot of the specular colour output (`o3`: sqrt(F0.rgb), kind in alpha).
pub const SLOT_F0: u32 = 3;
/// `o3.a` where the shader's F0 vector (or its grey folded constant) was captured.
pub const F0_RGB_ALPHA: f32 = 1.0;
/// `o3.a` (byte 128) where only the F0 luminance exists (`o3.rgb` = its square root, grey).
pub const F0_LUMINANCE_ALPHA: f32 = 128.0 / 255.0;
/// New temporaries of a patch with the captures (N/m, F0 luminance, scratch, F0 colour).
pub const NEW_TEMPS: u32 = 4;

// d3d11TokenizedProgramFormat.hpp opcodes.
const OP_ADD: u32 = 0;
const OP_AND: u32 = 1;
const OP_DIV: u32 = 14;
const OP_DP3: u32 = 16;
const OP_GE: u32 = 29;
const OP_LT: u32 = 49;
const OP_MAD: u32 = 50;
const OP_MAX: u32 = 52;
const OP_CUSTOMDATA: u32 = 53;
const OP_MOV: u32 = 54;
const OP_MOVC: u32 = 55;
const OP_MUL: u32 = 56;
const OP_RET: u32 = 62;
const OP_RETC: u32 = 63;
const OP_SAMPLE_L: u32 = 72;
const OP_SQRT: u32 = 75;
const OP_DCL_OUTPUT: u32 = 101;
const OP_DCL_TEMPS: u32 = 104;

const T_TEMP: u32 = 0;
const T_INPUT: u32 = 1;
const T_OUTPUT: u32 = 2;
const T_IMM32: u32 = 4;
const T_RESOURCE: u32 = 7;

const MOD_NEG: u32 = 1;
const MOD_ABS: u32 = 2;
const MOD_ABSNEG: u32 = 3;

/// Component selection of a source or destination operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sel {
    /// Destination write mask (or "as the destination" for a source).
    Mask(u8),
    Swizzle([u8; 4]),
    Select(u8),
    /// Zero/one-component operands (immediates, samplers).
    None,
}

impl Sel {
    /// The source component read for destination component `c`.
    fn component(self, c: u8) -> u8 {
        match self {
            Sel::Mask(_) | Sel::None => c,
            Sel::Swizzle(s) => s[usize::from(c)],
            Sel::Select(s) => s,
        }
    }
}

#[derive(Clone, Debug)]
struct Operand {
    ty: u32,
    sel: Sel,
    /// 1D immediate index (registers), if that is the operand's only index.
    reg: Option<u32>,
    modifier: u32,
    /// Immediate values (1 or 4).
    imm: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Instr {
    start: usize,
    opcode: u32,
    ops: Vec<Operand>,
}

impl Instr {
    /// Temporary register and write mask of the destination.
    fn dest_temp(&self) -> Option<(u32, u8)> {
        let d = self.ops.first()?;
        match (d.ty, d.reg, d.sel) {
            (T_TEMP, Some(r), Sel::Mask(m)) => Some((r, m)),
            _ => None,
        }
    }
}

fn is_declaration(op: u32) -> bool {
    op == OP_CUSTOMDATA || (88..=106).contains(&op) || (143..=163).contains(&op)
}

/// Decodes the operand at `t[i..]`; returns it and its length in dwords.
fn operand(t: &[u32], i: usize) -> Option<(Operand, usize)> {
    let tok = *t.get(i)?;
    let mut len = 1usize;
    let mut modifier = 0;
    let mut ext = tok >> 31 != 0;
    while ext {
        let e = *t.get(i + len)?;
        if e & 0x3F == 1 {
            modifier = (e >> 6) & 0xFF;
        }
        ext = e >> 31 != 0;
        len += 1;
    }
    let ty = (tok >> 12) & 0xFF;
    let ncomp = tok & 3;
    let sel = if ncomp == 2 {
        let bits = ((tok >> 4) & 0xFF) as u8;
        match (tok >> 2) & 3 {
            0 => Sel::Mask(bits & 0xF),
            1 => Sel::Swizzle([bits & 3, (bits >> 2) & 3, (bits >> 4) & 3, (bits >> 6) & 3]),
            2 => Sel::Select(bits & 3),
            _ => return None,
        }
    } else {
        Sel::None
    };
    if ty == T_IMM32 || ty == 5 {
        let n = match ncomp {
            1 => 1,
            2 => 4,
            _ => 0,
        };
        let words = n * if ty == 5 { 2 } else { 1 };
        let imm = t.get(i + len..i + len + words)?.to_vec();
        return Some((
            Operand {
                ty,
                sel,
                reg: None,
                modifier,
                imm,
            },
            len + words,
        ));
    }
    let dims = (tok >> 20) & 3;
    let mut reg = None;
    for d in 0..dims {
        match (tok >> (22 + 3 * d)) & 7 {
            0 => {
                let v = *t.get(i + len)?;
                if dims == 1 {
                    reg = Some(v);
                }
                len += 1;
            }
            1 => len += 2,
            2 => len += operand(t, i + len)?.1,
            3 => len += 1 + operand(t, i + len + 1)?.1,
            4 => len += 2 + operand(t, i + len + 2)?.1,
            _ => return None,
        }
    }
    Some((
        Operand {
            ty,
            sel,
            reg,
            modifier,
            imm: Vec::new(),
        },
        len,
    ))
}

struct Program {
    instrs: Vec<Instr>,
    /// Token index after the last `dcl_output`.
    dcl_output_end: usize,
    /// Token index of the first non-declaration.
    dcl_end: usize,
    /// Token index of `dcl_temps`' count (none without temporaries).
    temps_at: Option<usize>,
    returns: Vec<usize>,
}

fn decode(t: &[u32]) -> Result<Program, &'static str> {
    let mut p = Program {
        instrs: Vec::new(),
        dcl_output_end: 0,
        dcl_end: 0,
        temps_at: None,
        returns: Vec::new(),
    };
    let mut i = 2;
    while i < t.len() {
        let opcode = t[i] & 0x7FF;
        let len = if opcode == OP_CUSTOMDATA {
            *t.get(i + 1).ok_or("truncated custom data")? as usize
        } else {
            ((t[i] >> 24) & 0x7F) as usize
        };
        if len == 0 || i + len > t.len() {
            return Err("bad instruction length");
        }
        if opcode == OP_DCL_OUTPUT {
            p.dcl_output_end = i + len;
        }
        if opcode == OP_DCL_TEMPS {
            p.temps_at = Some(i + 1);
        }
        if !is_declaration(opcode) {
            if p.dcl_end == 0 {
                p.dcl_end = i;
            }
            if opcode == OP_RET || opcode == OP_RETC {
                p.returns.push(i);
            }
            let mut at = i + 1;
            let mut ext = t[i] >> 31 != 0;
            while ext && at < i + len {
                ext = t[at] >> 31 != 0;
                at += 1;
            }
            let mut ops = Vec::new();
            while at < i + len {
                let (o, l) = operand(t, at).ok_or("bad operand")?;
                ops.push(o);
                at += l;
            }
            p.instrs.push(Instr {
                start: i,
                opcode,
                ops,
            });
        }
        i += len;
    }
    if p.dcl_output_end == 0 || p.dcl_end == 0 || p.dcl_end < p.dcl_output_end {
        return Err("no output declarations before the code");
    }
    if p.returns.is_empty() {
        return Err("no ret");
    }
    Ok(p)
}

/// Index of the last instruction before `before` writing component `c` of temp `reg`.
fn last_def(p: &Program, before: usize, reg: u32, c: u8) -> Option<usize> {
    (0..before).rev().find(|&k| {
        p.instrs[k]
            .dest_temp()
            .is_some_and(|(r, m)| r == reg && m & (1 << c) != 0)
    })
}

fn imm_has(o: &Operand, value: f32) -> bool {
    o.ty == T_IMM32
        && o.imm
            .iter()
            .any(|&v| (f32::from_bits(v) - value).abs() < 2e-6)
}

/// A temp source as (reg, component) for destination component `c`.
fn temp_comp(o: &Operand, c: u8) -> Option<(u32, u8)> {
    (o.ty == T_TEMP).then_some(())?;
    Some((o.reg?, o.sel.component(c)))
}

/// Follows the LOD of the IBL sample back to the `-m * 0.504055 + 1.333129` mad.
fn find_m(p: &Program, sample: usize, mut cur: (u32, u8)) -> Option<(usize, (u32, u8))> {
    let mut k = sample;
    for _ in 0..10 {
        let d = last_def(p, k, cur.0, cur.1)?;
        let ins = &p.instrs[d];
        if ins.opcode == OP_MAD
            && ins.ops.len() == 4
            && imm_has(&ins.ops[2], 0.504_055)
            && imm_has(&ins.ops[3], 1.333_129)
        {
            return Some((d, temp_comp(&ins.ops[1], cur.1)?));
        }
        let srcs: Vec<(u32, u8)> = ins.ops[1..]
            .iter()
            .filter_map(|o| temp_comp(o, cur.1))
            .collect();
        cur = srcs
            .iter()
            .copied()
            .find(|&s| s == cur)
            .or_else(|| srcs.first().copied())?;
        k = d;
    }
    None
}

/// The reflect `mad R, N, -t, -V` feeding the lerp `mad L, m, A, R` before the sample.
fn find_n(p: &Program, sample: usize, m: (u32, u8)) -> Option<(usize, Operand)> {
    for j in (0..sample).rev() {
        let ins = &p.instrs[j];
        if ins.opcode != OP_MAD || ins.ops.len() != 4 {
            continue;
        }
        let Some((_, mask)) = ins.dest_temp() else {
            continue;
        };
        let c0 = mask.trailing_zeros() as u8;
        if temp_comp(&ins.ops[1], c0) != Some(m) || ins.ops[3].ty != T_TEMP {
            continue;
        }
        let Some(r) = temp_comp(&ins.ops[3], c0) else {
            continue;
        };
        let Some(d) = last_def(p, j, r.0, r.1) else {
            continue;
        };
        let refl = &p.instrs[d];
        let Some((_, mask)) = refl.dest_temp() else {
            continue;
        };
        if refl.opcode == OP_MAD
            && refl.ops.len() == 4
            && mask.count_ones() == 3
            && refl.ops[2].modifier == MOD_NEG
            && (refl.ops[3].modifier == MOD_NEG || refl.ops[3].ty == T_IMM32)
            && refl.ops[1].ty == T_TEMP
            && refl.ops[1].modifier == 0
        {
            // N's components in the order of the written components (x, y, z of the normal).
            let comps: Vec<u8> = (0..4u8).filter(|c| mask & (1 << c) != 0).collect();
            let n = &refl.ops[1];
            let s = |k: usize| n.sel.component(comps[k]);
            return Some((
                d,
                Operand {
                    sel: Sel::Swizzle([s(0), s(1), s(2), s(2)]),
                    ..n.clone()
                },
            ));
        }
    }
    None
}

/// The IBL luminance-factor lerp `mad X, F0, 1 - |v.c|, |v.c|` after the sample: the F0
/// operand (temp component or immediate) as a scalar source.
fn find_f0(p: &Program, sample: usize) -> Option<(usize, Operand)> {
    for j in sample + 1..p.instrs.len() {
        let ins = &p.instrs[j];
        if ins.opcode != OP_MAD || ins.ops.len() != 4 {
            continue;
        }
        let c = &ins.ops[3];
        if c.ty != T_INPUT || c.modifier != MOD_ABS {
            continue;
        }
        let Some((_, mask)) = ins.dest_temp() else {
            continue;
        };
        let c0 = mask.trailing_zeros() as u8;
        for (a, b) in [(&ins.ops[1], &ins.ops[2]), (&ins.ops[2], &ins.ops[1])] {
            let Some(ra) = temp_comp(a, c0) else { continue };
            let Some(d) = last_def(p, j, ra.0, ra.1) else {
                continue;
            };
            let add = &p.instrs[d];
            let from_input = add.opcode == OP_ADD
                && add.ops[1..]
                    .iter()
                    .any(|o| o.ty == T_INPUT && o.reg == c.reg && o.modifier == MOD_ABSNEG);
            if !from_input {
                continue;
            }
            let scalar = match b.ty {
                T_TEMP => src_sel(T_TEMP, b.reg?, b.sel.component(c0), 0),
                T_IMM32 => {
                    let v = if b.imm.len() == 4 {
                        b.imm[usize::from(c0)]
                    } else {
                        *b.imm.first()?
                    };
                    imm1(f32::from_bits(v))
                }
                _ => return None,
            };
            return Some((j, decode_one(&scalar)?));
        }
    }
    None
}

/// The F0 vector behind the luminance capture `f` (a temp component read by the lerp at
/// `at`): `dp3 f, l(0.2126, 0.7152, 0.0722, 0), F0.xyz` (`HQ_ComputeIBL`: `dot(lum,
/// Fresnel_F0FromIOR(IOR))`). Returns the dp3 and the vector as an x, y, z source.
fn find_f0_rgb(p: &Program, at: usize, f: &Operand) -> Option<(usize, Operand)> {
    let (T_TEMP, Sel::Select(c)) = (f.ty, f.sel) else {
        return None;
    };
    let d = last_def(p, at, f.reg?, c)?;
    let dp = &p.instrs[d];
    if dp.opcode != OP_DP3 || dp.ops.len() != 3 {
        return None;
    }
    let (lum, v) = if dp.ops[1].ty == T_IMM32 {
        (&dp.ops[1], &dp.ops[2])
    } else {
        (&dp.ops[2], &dp.ops[1])
    };
    if !(imm_has(lum, 0.2126) && imm_has(lum, 0.7152) && imm_has(lum, 0.0722))
        || v.ty != T_TEMP
        || v.modifier != 0
    {
        return None;
    }
    let s = |k: u8| v.sel.component(k);
    Some((
        d,
        Operand {
            sel: Sel::Swizzle([s(0), s(1), s(2), s(2)]),
            ..v.clone()
        },
    ))
}

// ---------------------------------------------------------------------- encoding --

fn decode_one(t: &[u32]) -> Option<Operand> {
    operand(t, 0).map(|o| o.0)
}

fn dst(ty: u32, reg: u32, mask: u8) -> Vec<u32> {
    vec![2 | (u32::from(mask) << 4) | (ty << 12) | (1 << 20), reg]
}

fn src_swz(ty: u32, reg: u32, s: [u8; 4], modifier: u32) -> Vec<u32> {
    let bits =
        u32::from(s[0]) | (u32::from(s[1]) << 2) | (u32::from(s[2]) << 4) | (u32::from(s[3]) << 6);
    let mut v = vec![2 | (1 << 2) | (bits << 4) | (ty << 12) | (1 << 20)];
    if modifier != 0 {
        v[0] |= 1 << 31;
        v.push(1 | (modifier << 6));
    }
    v.push(reg);
    v
}

fn src_sel(ty: u32, reg: u32, c: u8, modifier: u32) -> Vec<u32> {
    let mut v = vec![2 | (2 << 2) | (u32::from(c) << 4) | (ty << 12) | (1 << 20)];
    if modifier != 0 {
        v[0] |= 1 << 31;
        v.push(1 | (modifier << 6));
    }
    v.push(reg);
    v
}

fn imm4(v: [f32; 4]) -> Vec<u32> {
    let mut t = vec![2 | (T_IMM32 << 12)];
    t.extend(v.map(f32::to_bits));
    t
}

fn imm1(v: f32) -> Vec<u32> {
    vec![1 | (T_IMM32 << 12), v.to_bits()]
}

fn instr(opcode: u32, ops: &[Vec<u32>]) -> Vec<u32> {
    let mut t = vec![0];
    for o in ops {
        t.extend_from_slice(o);
    }
    t[0] = opcode | ((t.len() as u32) << 24);
    t
}

/// Re-encodes a decoded temp operand as a source (no modifier).
fn reencode_temp(o: &Operand) -> Option<Vec<u32>> {
    let reg = o.reg?;
    Some(match o.sel {
        Sel::Swizzle(s) => src_swz(T_TEMP, reg, s, 0),
        Sel::Select(c) => src_sel(T_TEMP, reg, c, 0),
        Sel::Mask(_) => src_swz(T_TEMP, reg, [0, 1, 2, 3], 0),
        Sel::None => return None,
    })
}

fn reencode_scalar(o: &Operand) -> Option<Vec<u32>> {
    match o.ty {
        T_IMM32 => Some(imm1(f32::from_bits(*o.imm.first()?))),
        T_TEMP => match o.sel {
            Sel::Select(c) => Some(src_sel(T_TEMP, o.reg?, c, 0)),
            _ => None,
        },
        _ => None,
    }
}

const X: u8 = 0;
const Y: u8 = 1;
const Z: u8 = 2;
const W: u8 = 3;

/// `o2` and `o3` from the captures: tA = (N.xyz, m), tB.x = F0 luminance, tD.xyz = F0 colour;
/// tC and tB.yzw scratch. `exclude`: `o2.a` is the exclusion marker instead of the
/// material's sqrt(F0) and `o3` is 0. `f0_alpha`: `o3.a` of the material.
fn epilogue(a: u32, b: u32, c: u32, d: u32, exclude: bool, f0_alpha: f32) -> Vec<u32> {
    let mut t = Vec::new();
    let mut push = |v: Vec<u32>| t.extend(v);
    // L1 norm and octahedral projection p = N / (|x| + |y| + |z|).
    push(instr(
        OP_DP3,
        &[
            dst(T_TEMP, c, 1 << W),
            src_swz(T_TEMP, a, [X, Y, Z, X], MOD_ABS),
            imm4([1.0, 1.0, 1.0, 0.0]),
        ],
    ));
    push(instr(
        OP_MAX,
        &[dst(T_TEMP, c, 1 << W), src_sel(T_TEMP, c, W, 0), imm1(1e-6)],
    ));
    push(instr(
        OP_DIV,
        &[
            dst(T_TEMP, c, 0b0111),
            src_swz(T_TEMP, a, [X, Y, Z, X], 0),
            src_sel(T_TEMP, c, W, 0),
        ],
    ));
    // Lower hemisphere fold: (1 - |p.yx|) * sign(p.xy).
    push(instr(
        OP_ADD,
        &[
            dst(T_TEMP, b, 0b0110),
            src_swz(T_TEMP, c, [Y, Y, X, X], MOD_ABSNEG),
            imm4([1.0; 4]),
        ],
    ));
    for (comp, out) in [(X, Y), (Y, Z)] {
        push(instr(
            OP_GE,
            &[
                dst(T_TEMP, c, 1 << W),
                src_sel(T_TEMP, c, comp, 0),
                imm1(0.0),
            ],
        ));
        push(instr(
            OP_MOVC,
            &[
                dst(T_TEMP, c, 1 << W),
                src_sel(T_TEMP, c, W, 0),
                imm1(1.0),
                imm1(-1.0),
            ],
        ));
        push(instr(
            OP_MUL,
            &[
                dst(T_TEMP, b, 1 << out),
                src_sel(T_TEMP, b, out, 0),
                src_sel(T_TEMP, c, W, 0),
            ],
        ));
    }
    push(instr(
        OP_LT,
        &[dst(T_TEMP, c, 1 << W), src_sel(T_TEMP, c, Z, 0), imm1(0.0)],
    ));
    push(instr(
        OP_MOVC,
        &[
            dst(T_TEMP, c, 0b0011),
            src_sel(T_TEMP, c, W, 0),
            src_swz(T_TEMP, b, [Y, Z, Y, Y], 0),
            src_swz(T_TEMP, c, [X, Y, X, X], 0),
        ],
    ));
    push(instr(
        OP_MAD,
        &[
            dst(T_OUTPUT, SLOT, 0b0011),
            src_swz(T_TEMP, c, [X, Y, X, X], 0),
            imm4([0.5, 0.5, 0.0, 0.0]),
            imm4([0.5, 0.5, 0.0, 0.0]),
        ],
    ));
    // Perceptual roughness sqrt(m).
    push(instr(
        OP_MAX,
        &[dst(T_TEMP, c, 1 << W), src_sel(T_TEMP, a, W, 0), imm1(0.0)],
    ));
    push(instr(
        OP_SQRT,
        &[dst(T_OUTPUT, SLOT, 1 << Z), src_sel(T_TEMP, c, W, 0)],
    ));
    if exclude {
        push(instr(
            OP_MOV,
            &[dst(T_OUTPUT, SLOT, 1 << W), imm1(EXCLUDED_ALPHA)],
        ));
        push(instr(
            OP_MOV,
            &[dst(T_OUTPUT, SLOT_F0, 0xF), imm4([0.0; 4])],
        ));
        return t;
    }
    // Alpha: max(sqrt(F0), 2/255) where m was captured, else 0 (no material).
    push(instr(
        OP_MAX,
        &[dst(T_TEMP, c, 1 << W), src_sel(T_TEMP, b, X, 0), imm1(0.0)],
    ));
    push(instr(
        OP_SQRT,
        &[dst(T_TEMP, c, 1 << W), src_sel(T_TEMP, c, W, 0)],
    ));
    push(instr(
        OP_MAX,
        &[
            dst(T_TEMP, c, 1 << W),
            src_sel(T_TEMP, c, W, 0),
            imm1(MATERIAL_ALPHA_MIN),
        ],
    ));
    push(instr(
        OP_GE,
        &[dst(T_TEMP, c, 1 << Z), src_sel(T_TEMP, a, W, 0), imm1(0.0)],
    ));
    push(instr(
        OP_AND,
        &[
            dst(T_OUTPUT, SLOT, 1 << W),
            src_sel(T_TEMP, c, W, 0),
            src_sel(T_TEMP, c, Z, 0),
        ],
    ));
    // o3 = (sqrt(F0.rgb), f0_alpha) under the same mask (tD is left as captured: the
    // epilogue may run more than once, before a `retc` that is not taken).
    push(instr(
        OP_MAX,
        &[
            dst(T_TEMP, b, 0b1110),
            src_swz(T_TEMP, d, [X, X, Y, Z], 0),
            imm1(0.0),
        ],
    ));
    push(instr(
        OP_SQRT,
        &[dst(T_TEMP, b, 0b1110), src_swz(T_TEMP, b, [X, Y, Z, W], 0)],
    ));
    push(instr(
        OP_AND,
        &[
            dst(T_OUTPUT, SLOT_F0, 0b0111),
            src_swz(T_TEMP, b, [Y, Z, W, Y], 0),
            src_sel(T_TEMP, c, Z, 0),
        ],
    ));
    push(instr(
        OP_AND,
        &[
            dst(T_OUTPUT, SLOT_F0, 1 << W),
            imm1(f0_alpha),
            src_sel(T_TEMP, c, Z, 0),
        ],
    ));
    t
}

// ---------------------------------------------------------------------- container --

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(o..o.checked_add(4)?)?.try_into().ok()?,
    ))
}

struct Chunk {
    fourcc: [u8; 4],
    start: usize,
    len: usize,
}

fn chunks(code: &[u8]) -> Option<Vec<Chunk>> {
    if code.get(0..4)? != b"DXBC" || u32_at(code, 24)? as usize != code.len() {
        return None;
    }
    let n = u32_at(code, 28)? as usize;
    if n > 64 {
        return None;
    }
    (0..n)
        .map(|i| -> Option<Chunk> {
            let off = u32_at(code, 32 + 4 * i)? as usize;
            let len = u32_at(code, off + 4)? as usize;
            let start = off + 8;
            if start + len > code.len() {
                return None;
            }
            Some(Chunk {
                fourcc: code.get(off..off + 4)?.try_into().ok()?,
                start,
                len,
            })
        })
        .collect()
}

/// Output signature with an `SV_Target` element at `reg` (a copy of target 0).
fn add_target(osgn: &[u8], reg: u32) -> Result<Vec<u8>, &'static str> {
    let count = u32_at(osgn, 0).ok_or("OSGN truncated")? as usize;
    const H: usize = 8;
    const E: usize = 24;
    if H + count * E > osgn.len() {
        return Err("OSGN truncated");
    }
    let mut target0 = None;
    for i in 0..count {
        let e = H + i * E;
        let name_off = u32_at(osgn, e).ok_or("OSGN")? as usize;
        let name = osgn.get(name_off..name_off + 9).ok_or("OSGN name")?;
        let r = u32_at(osgn, e + 16).ok_or("OSGN")?;
        if r == reg {
            return Err("output register 2 already used");
        }
        if name.eq_ignore_ascii_case(b"SV_Target") && u32_at(osgn, e + 4) == Some(0) {
            target0 = Some(e);
        }
    }
    let t0 = target0.ok_or("no SV_Target0")?;
    let mut out = Vec::with_capacity(osgn.len() + E);
    out.extend_from_slice(&osgn[..H + count * E]);
    out.extend_from_slice(&osgn[t0..t0 + E]);
    out.extend_from_slice(&osgn[H + count * E..]);
    out[0..4].copy_from_slice(&(count as u32 + 1).to_le_bytes());
    for i in 0..=count {
        let e = H + i * E;
        let off = u32_at(&out, e).ok_or("OSGN")? + E as u32;
        out[e..e + 4].copy_from_slice(&off.to_le_bytes());
    }
    let added = H + count * E;
    out[added + 4..added + 8].copy_from_slice(&reg.to_le_bytes());
    out[added + 16..added + 20].copy_from_slice(&reg.to_le_bytes());
    out[added + 20] = 0xF;
    out[added + 21] = 0;
    Ok(out)
}

/// DXBC container checksum: MD5 over everything after the hash field with the container's
/// own final-block layout (as d2enhanced `shader_patch.cpp`, matching the D3D compiler).
pub fn checksum(container: &[u8]) -> [u8; 16] {
    let data = &container[20..];
    let size = data.len() as u32;
    let bits = size.wrapping_mul(8);
    let bits2 = (bits >> 2) | 1;
    let full = data.len() / 64 * 64;
    let mut st = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for blk in data[..full].as_chunks::<64>().0 {
        md5_block(&mut st, blk);
    }
    let left = &data[full..];
    let mut b = [0u8; 64];
    if left.len() >= 56 {
        b[..left.len()].copy_from_slice(left);
        b[left.len()] = 0x80;
        md5_block(&mut st, &b);
        b = [0; 64];
        b[0..4].copy_from_slice(&bits.to_le_bytes());
        b[60..64].copy_from_slice(&bits2.to_le_bytes());
    } else {
        b[0..4].copy_from_slice(&bits.to_le_bytes());
        b[4..4 + left.len()].copy_from_slice(left);
        b[4 + left.len()] = 0x80;
        b[60..64].copy_from_slice(&bits2.to_le_bytes());
    }
    md5_block(&mut st, &b);
    let mut out = [0u8; 16];
    for (i, s) in st.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
    }
    out
}

fn md5_block(st: &mut [u32; 4], blk: &[u8; 64]) {
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    const R: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let m: [u32; 16] = std::array::from_fn(|i| {
        u32::from_le_bytes([blk[4 * i], blk[4 * i + 1], blk[4 * i + 2], blk[4 * i + 3]])
    });
    let [mut a, mut b, mut c, mut d] = *st;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let t = d;
        d = c;
        c = b;
        let x = a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]);
        b = b.wrapping_add(x.rotate_left(R[i]));
        a = t;
    }
    st[0] = st[0].wrapping_add(a);
    st[1] = st[1].wrapping_add(b);
    st[2] = st[2].wrapping_add(c);
    st[3] = st[3].wrapping_add(d);
}

// -------------------------------------------------------------------------- patch --

/// Material class of a pixel shader by its RDEF names ([`shader`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Everything else: reflects by its own roughness and F0.
    Surface,
    /// `arkVegetation` (`cb_vegetanim_*` wind animation, `cb_vegetation_*` transmittance).
    Vegetation,
    /// `arkHair` / `BRDF_hair` (`cb_hair*`, `ro_hair*`).
    Hair,
    /// `arkDecal` projected decals (`cb_decalboxprojection`, `cb_decalbufferoffset`,
    /// `cb_world2decalmatrix*`).
    Decal,
    /// Particle inputs (`ro_fx_*`) and none of the above.
    Particles,
}

impl Class {
    pub const ALL: [Class; 5] = [
        Class::Surface,
        Class::Vegetation,
        Class::Hair,
        Class::Decal,
        Class::Particles,
    ];

    /// Not a reflection receiver: forward shaders of the class write the exclusion marker.
    pub fn excluded(self) -> bool {
        self != Class::Surface
    }

    pub fn name(self) -> &'static str {
        match self {
            Class::Surface => "surface",
            Class::Vegetation => "vegetation",
            Class::Hair => "hair",
            Class::Decal => "decal",
            Class::Particles => "particles",
        }
    }
}

/// What the RDEF names say about a pixel shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shader {
    pub class: Class,
    /// Forward lighting (`ro_tlf_ll_heads`) without particle inputs (`ro_fx_*`, blended
    /// through the target's blend state): drawn in the forward passes that bind our target.
    pub forward: bool,
    /// Texture slot of the IBL cube array (`ro_ibl_texcubemaparray`).
    pub ibl: Option<u32>,
}

impl Shader {
    /// Gets the material output: lit forward shaders (IBL) and forward shaders of the
    /// excluded classes.
    pub fn candidate(&self) -> bool {
        self.forward && (self.ibl.is_some() || self.class.excluded())
    }
}

/// Chunk payload by four-character code.
fn find_chunk<'a>(code: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    if code.get(0..4)? != b"DXBC" {
        return None;
    }
    for i in 0..(u32_at(code, 28)? as usize).min(64) {
        let off = u32_at(code, 32 + 4 * i)? as usize;
        if code.get(off..off.checked_add(4)?)? == fourcc {
            let start = off + 8;
            return code.get(start..start.checked_add(u32_at(code, off + 4)? as usize)?);
        }
    }
    None
}

/// NUL-terminated name at `o`.
fn name_at(b: &[u8], o: usize) -> Option<&[u8]> {
    let s = b.get(o..)?;
    Some(&s[..s.iter().take(256).position(|&c| c == 0)?])
}

/// Visits the RDEF binding names (with `(input type, slot)`) and cbuffer variable names.
fn rdef_names(code: &[u8], mut f: impl FnMut(&[u8], Option<(u32, u32)>)) -> Option<()> {
    let r = find_chunk(code, b"RDEF")?;
    let (cbs, cb_at) = (u32_at(r, 0)? as usize, u32_at(r, 4)? as usize);
    let (binds, bind_at) = (u32_at(r, 8)? as usize, u32_at(r, 12)? as usize);
    let (minor, major) = (*r.get(16)?, *r.get(17)?);
    // SM 5.1 binding descriptors add space and id; SM 5 variable descriptors add the
    // texture/sampler ranges.
    let bind_stride = if major > 5 || (major == 5 && minor >= 1) {
        40
    } else {
        32
    };
    let var_stride = if major >= 5 { 40 } else { 24 };
    for i in 0..binds.min(256) {
        let e = bind_at + i * bind_stride;
        let name = name_at(r, u32_at(r, e)? as usize)?;
        f(name, Some((u32_at(r, e + 4)?, u32_at(r, e + 20)?)));
    }
    for i in 0..cbs.min(64) {
        let c = cb_at + i * 24;
        let (vars, var_at) = (u32_at(r, c + 4)? as usize, u32_at(r, c + 8)? as usize);
        for j in 0..vars.min(4096) {
            f(
                name_at(r, u32_at(r, var_at + j * var_stride)? as usize)?,
                None,
            );
        }
    }
    Some(())
}

/// Class and pass of a pixel shader from its RDEF chunk (the bytecode is all there is at
/// `CreatePixelShader`). The game's cbuffers list every parameter of the render program,
/// vertex stage included, so `arkVegetation`'s wind parameters name its pixel shaders too.
pub fn shader(code: &[u8]) -> Shader {
    let (mut tlf, mut fx, mut ibl) = (false, false, None);
    let (mut veg, mut hair, mut decal) = (false, false, false);
    rdef_names(code, |n, bind| {
        if let Some((kind, slot)) = bind {
            tlf |= n == b"ro_tlf_ll_heads";
            fx |= n.starts_with(b"ro_fx_");
            // D3D_SIT_TEXTURE
            if kind == 2 && n == b"ro_ibl_texcubemaparray" {
                ibl = Some(slot);
            }
        }
        veg |= n.starts_with(b"cb_vegetanim_") || n.starts_with(b"cb_vegetation_");
        hair |= n.starts_with(b"cb_hair") || n.starts_with(b"ro_hair");
        decal |= n == b"cb_decalboxprojection"
            || n == b"cb_decalbufferoffset"
            || n.starts_with(b"cb_world2decalmatrix");
    });
    let class = if veg {
        Class::Vegetation
    } else if hair {
        Class::Hair
    } else if decal {
        Class::Decal
    } else if fx {
        Class::Particles
    } else {
        Class::Surface
    };
    Shader {
        class,
        forward: tlf && !fx,
        ibl,
    }
}

/// Where `o3`'s F0 colour comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum F0Source {
    /// `HQ_ComputeIBL`'s F0 vector (the `dp3` with the luminance weights) captured.
    Rgb,
    /// Folded constant: a constant IOR, grey (the shader uses it as a scalar throughout).
    Constant,
    /// Luminance register without the `dp3` behind it: grey, `o3.a` byte 128.
    Luminance,
    /// No F0 capture: 0.04 grey, `o3.a` byte 128.
    Default,
}

impl F0Source {
    /// `o3.a` of the material.
    pub fn alpha(self) -> f32 {
        match self {
            F0Source::Rgb | F0Source::Constant => F0_RGB_ALPHA,
            F0Source::Luminance | F0Source::Default => F0_LUMINANCE_ALPHA,
        }
    }
}

/// What `patch` found and did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patched {
    pub code: Vec<u8>,
    pub class: Class,
    /// Normal and roughness captured at the IBL sample (else only the exclusion marker is
    /// written: excluded classes without the capture).
    pub captured: bool,
    /// First new temporary ([`NEW_TEMPS`] are added with the captures, none without).
    pub first_temp: u32,
    /// F0 capture (`None`: marker only).
    pub f0: Option<F0Source>,
    /// Inserted instructions.
    pub inserted: usize,
}

/// Capture points of the IBL code.
struct Captures {
    m_at: usize,
    m: (u32, u8),
    n_at: usize,
    n: Operand,
    /// The lerp reading the F0 luminance and its F0 operand.
    f0: Option<(usize, Operand)>,
    /// The luminance `dp3` and its F0 vector operand.
    f0_rgb: Option<(usize, Operand)>,
}

fn captures(p: &Program, ibl: u32) -> Result<Captures, &'static str> {
    let sample = p
        .instrs
        .iter()
        .position(|i| {
            i.opcode == OP_SAMPLE_L
                && i.ops.len() >= 5
                && i.ops[2].ty == T_RESOURCE
                && i.ops[2].reg == Some(ibl)
        })
        .ok_or("no IBL sample_l")?;
    let lod = &p.instrs[sample].ops[4];
    let lod = temp_comp(lod, 0).ok_or("IBL LOD is not a register")?;
    let (m_at, m) = find_m(p, sample, lod).ok_or("roughness (mip formula) not found")?;
    let (n_at, n) = find_n(p, sample, m).ok_or("shading normal (reflect/lerp) not found")?;
    let f0 = find_f0(p, sample);
    let f0_rgb = f0.as_ref().and_then(|(at, f)| find_f0_rgb(p, *at, f));
    Ok(Captures {
        m_at,
        m,
        n_at,
        n,
        f0,
        f0_rgb,
    })
}

/// Adds the material output to a [`Shader::candidate`] (`s` = [`shader`]`(code)`): the
/// captures where the IBL code is found, else (excluded classes only) the exclusion marker.
/// `Err` = reason it was skipped.
pub fn patch(code: &[u8], s: &Shader) -> Result<Patched, &'static str> {
    if !s.candidate() {
        return Err("not a lit forward shader or a forward shader of an excluded class");
    }
    let cs = chunks(code).ok_or("malformed container")?;
    let shex = cs
        .iter()
        .position(|c| &c.fourcc == b"SHEX" || &c.fourcc == b"SHDR")
        .ok_or("no shader code chunk")?;
    let osgn = cs
        .iter()
        .position(|c| &c.fourcc == b"OSGN")
        .ok_or("no OSGN chunk (OSG5/OSG1 not handled)")?;
    let body = &code[cs[shex].start..cs[shex].start + cs[shex].len];
    if !body.len().is_multiple_of(4) {
        return Err("code chunk size");
    }
    let mut t: Vec<u32> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if (t.first().copied().unwrap_or(0) >> 16) & 0xFFFF != 0 {
        return Err("not a pixel shader");
    }
    let p = decode(&t)?;
    let exclude = s.class.excluded();
    let caps = match s.ibl {
        Some(ibl) => captures(&p, ibl),
        None => Err("no IBL cube array"),
    };
    let caps = match caps {
        Ok(c) => Some(c),
        Err(_) if exclude => None,
        Err(e) => return Err(e),
    };
    let osgn_new = add_target(&code[cs[osgn].start..cs[osgn].start + cs[osgn].len], SLOT)?;
    let osgn_new = add_target(&osgn_new, SLOT_F0)?;

    // (token index, order at that index, tokens): inserted back to front.
    let mut decls = instr(OP_DCL_OUTPUT, &[dst(T_OUTPUT, SLOT, 0xF)]);
    decls.extend(instr(OP_DCL_OUTPUT, &[dst(T_OUTPUT, SLOT_F0, 0xF)]));
    let mut ins: Vec<(usize, u8, Vec<u32>)> = vec![(p.dcl_output_end, 0, decls)];
    let first_temp;
    let mut f0 = None;
    if let Some(cap) = &caps {
        let temps_at = p.temps_at.ok_or("no dcl_temps")?;
        first_temp = t[temps_at];
        let (a, b, c, d) = (first_temp, first_temp + 1, first_temp + 2, first_temp + 3);
        ins.extend([
            (
                p.dcl_end,
                1,
                instr(OP_MOV, &[dst(T_TEMP, a, 0xF), imm4([0.0, 0.0, 0.0, -1.0])]),
            ),
            (
                p.dcl_end,
                1,
                instr(OP_MOV, &[dst(T_TEMP, b, 0xF), imm4([0.04, 0.0, 0.0, 0.0])]),
            ),
            (
                p.dcl_end,
                1,
                instr(
                    OP_MOV,
                    &[dst(T_TEMP, d, 0xF), imm4([0.04, 0.04, 0.04, 0.0])],
                ),
            ),
            (
                p.instrs[cap.m_at].start,
                2,
                instr(
                    OP_MOV,
                    &[dst(T_TEMP, a, 1 << W), src_sel(T_TEMP, cap.m.0, cap.m.1, 0)],
                ),
            ),
            (
                p.instrs[cap.n_at].start,
                2,
                instr(
                    OP_MOV,
                    &[
                        dst(T_TEMP, a, 0b0111),
                        reencode_temp(&cap.n).ok_or("normal operand")?,
                    ],
                ),
            ),
        ]);
        let source = match (&cap.f0, &cap.f0_rgb) {
            (None, _) => F0Source::Default,
            (Some((at, f)), rgb) => {
                let src = reencode_scalar(f).ok_or("F0 operand")?;
                let at = p.instrs[*at].start;
                ins.push((at, 3, instr(OP_MOV, &[dst(T_TEMP, b, 1 << X), src.clone()])));
                if let Some((rgb_at, v)) = rgb {
                    ins.push((
                        p.instrs[*rgb_at].start,
                        3,
                        instr(
                            OP_MOV,
                            &[
                                dst(T_TEMP, d, 0b0111),
                                reencode_temp(v).ok_or("F0 vector operand")?,
                            ],
                        ),
                    ));
                    F0Source::Rgb
                } else if f.ty == T_IMM32 {
                    let v = f32::from_bits(*f.imm.first().ok_or("F0 constant")?);
                    ins.push((
                        at,
                        3,
                        instr(OP_MOV, &[dst(T_TEMP, d, 0b0111), imm4([v; 4])]),
                    ));
                    F0Source::Constant
                } else {
                    ins.push((at, 3, instr(OP_MOV, &[dst(T_TEMP, d, 0b0111), src])));
                    F0Source::Luminance
                }
            }
        };
        f0 = Some(source);
        let epi = epilogue(a, b, c, d, exclude, source.alpha());
        for &r in &p.returns {
            ins.push((r, 4, epi.clone()));
        }
        t[temps_at] = first_temp + NEW_TEMPS;
    } else {
        // No new temporaries: their registers would start at the current count.
        first_temp = p.temps_at.map_or(0, |i| t[i]);
        let mut marker = instr(
            OP_MOV,
            &[
                dst(T_OUTPUT, SLOT, 0xF),
                imm4([0.5, 0.5, 1.0, EXCLUDED_ALPHA]),
            ],
        );
        marker.extend(instr(
            OP_MOV,
            &[dst(T_OUTPUT, SLOT_F0, 0xF), imm4([0.0; 4])],
        ));
        for &r in &p.returns {
            ins.push((r, 4, marker.clone()));
        }
    }
    // Everything but the two `dcl_output`s.
    let inserted = ins.iter().map(|x| instr_count(&x.2)).sum::<usize>() - 2;
    ins.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)));
    for (at, _, tokens) in ins {
        t.splice(at..at, tokens);
    }
    t[1] = t.len() as u32;

    // Reassemble the container.
    let bodies: Vec<Vec<u8>> = cs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i == shex {
                t.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else if i == osgn {
                osgn_new.clone()
            } else {
                code[c.start..c.start + c.len].to_vec()
            }
        })
        .collect();
    let header = 32 + 4 * cs.len();
    let total = header + bodies.iter().map(|b| 8 + b.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&code[..32]);
    out[24..28].copy_from_slice(&(total as u32).to_le_bytes());
    let mut at = header;
    for b in &bodies {
        out.extend_from_slice(&(at as u32).to_le_bytes());
        at += 8 + b.len();
    }
    for (c, b) in cs.iter().zip(&bodies) {
        out.extend_from_slice(&c.fourcc);
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    let sum = checksum(&out);
    out[4..20].copy_from_slice(&sum);
    Ok(Patched {
        code: out,
        class: s.class,
        captured: caps.is_some(),
        first_temp,
        f0,
        inserted,
    })
}

/// Instructions in a token sequence without custom data.
fn instr_count(t: &[u32]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < t.len() {
        n += 1;
        i += ((t[i] >> 24) & 0x7F).max(1) as usize;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_round_trip() {
        let s = src_swz(T_TEMP, 14, [X, Z, W, X], MOD_ABS);
        let (o, len) = operand(&s, 0).unwrap();
        assert_eq!(len, s.len());
        assert_eq!((o.ty, o.reg, o.modifier), (T_TEMP, Some(14), MOD_ABS));
        assert_eq!(o.sel, Sel::Swizzle([X, Z, W, X]));
        assert_eq!(o.sel.component(1), Z);
        let (o, _) = operand(&src_sel(T_TEMP, 5, W, MOD_NEG), 0).unwrap();
        assert_eq!((o.sel, o.modifier), (Sel::Select(W), MOD_NEG));
        let (o, len) = operand(&imm4([0.0, 0.504_055, 0.504_055, 0.0]), 0).unwrap();
        assert_eq!(len, 5);
        assert!(imm_has(&o, 0.504_055));
        let (o, _) = operand(&dst(T_OUTPUT, 2, 0b0011), 0).unwrap();
        assert_eq!((o.ty, o.sel), (T_OUTPUT, Sel::Mask(3)));
        assert_eq!(instr_count(&epilogue(0, 1, 2, 3, false, F0_RGB_ALPHA)), 24);
        assert_eq!(instr_count(&epilogue(0, 1, 2, 3, true, F0_RGB_ALPHA)), 17);
    }

    #[test]
    fn class_from_cbuffer_variables() {
        let code = crate::hlsl::compile(
            "cbuffer PerInstanceCB : register(b2) { float4 cb_vegetanim_leaffreq; float4 cb_x; };
             Texture2D<uint> ro_tlf_ll_heads : register(t2);
             float4 main(float4 p : SV_Position) : SV_Target {
                 return p * cb_x + ro_tlf_ll_heads.Load(int3(0, 0, 0));
             }",
            windows::core::s!("main"),
            windows::core::s!("ps_5_0"),
        )
        .unwrap();
        let s = shader(&code);
        assert_eq!(
            s,
            Shader {
                class: Class::Vegetation,
                forward: true,
                ibl: None
            }
        );
        let p = patch(&code, &s).unwrap();
        assert!(!p.captured);
        // The marker (o2 and o3) before the one `ret`.
        assert_eq!(p.inserted, 2);
        assert_eq!(
            checksum(&p.code),
            <[u8; 16]>::try_from(&p.code[4..20]).unwrap()
        );
        let s = Shader {
            class: Class::Surface,
            ..s
        };
        assert!(!s.candidate() && patch(&code, &s).is_err());
    }

    #[test]
    fn checksum_of_a_compiled_shader() {
        let code = crate::hlsl::compile(
            "float4 main(float4 p : SV_Position) : SV_Target { return p * 0.5; }",
            windows::core::s!("main"),
            windows::core::s!("ps_5_0"),
        )
        .unwrap();
        assert_eq!(checksum(&code), <[u8; 16]>::try_from(&code[4..20]).unwrap());
    }

    #[test]
    fn output_signature_gains_target_2() {
        let code = crate::hlsl::compile(
            "float4 main(float4 p : SV_Position) : SV_Target { return p * 0.5; }",
            windows::core::s!("main"),
            windows::core::s!("ps_5_0"),
        )
        .unwrap();
        let cs = chunks(&code).unwrap();
        let o = cs.iter().find(|c| &c.fourcc == b"OSGN").unwrap();
        let sig = add_target(&code[o.start..o.start + o.len], SLOT).unwrap();
        assert_eq!(u32_at(&sig, 0), Some(2));
        assert_eq!(u32_at(&sig, 8 + 24 + 16), Some(SLOT));
        assert!(add_target(&sig, SLOT).is_err());
    }
}
