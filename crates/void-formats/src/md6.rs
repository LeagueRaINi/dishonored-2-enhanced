//! Skinned models (`generated/basemodel/**.bmd6model`, magic `MM6B`), their
//! Edge skeletons (`generated/skeleton/**.edgeskl`, tag `60SE`) and the subset
//! of `md6Def` decls that links them (docs/formats/md6.md).
//!
//! Model reader: `sub_14025AA00` (model), `sub_140259960` (surface), the
//! `BTRI` reader shared with `.bmodel` (`sub_14025A040`), joint helpers
//! `sub_1402538F0`/`sub_1402537D0`. Vertex declarations: `sub_1402D0DC0`
//! (element lists) and `sub_1402D1AF0` (declaration id -> list). Skeleton:
//! `sub_14024F2A0` (Edge header) and `sub_1402564E0` (custom data). All
//! little-endian; `str` = i32 length + bytes, length <= 0 = empty.

use crate::Error;
use crate::bwm::f16_to_f32;
use crate::decl::{Tok, tokenize};
use crate::reader::Reader;

pub const MAGIC: [u8; 4] = *b"MM6B";
pub const SKELETON_TAG: [u8; 4] = *b"60SE";
/// Influences per vertex in the skinned declarations (two 4-byte groups).
pub const MAX_INFLUENCES: usize = 8;

fn invalid(field: &'static str, detail: String) -> Error {
    Error::Invalid { field, detail }
}

fn string(r: &mut Reader, field: &'static str) -> Result<String, Error> {
    let n = r.u32(field)? as i32;
    if n <= 0 {
        return Ok(String::new());
    }
    let b = r.bytes(n as usize, field)?;
    // Paths are ASCII in all samples; keep any other byte losslessly visible.
    Ok(b.iter().map(|&c| c as char).collect())
}

fn f32s<const N: usize>(r: &mut Reader, field: &'static str) -> Result<[f32; N], Error> {
    let mut out = [0f32; N];
    for v in &mut out {
        *v = r.f32(field)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Vertex declarations (runtime table built by sub_1402D0DC0)
// ---------------------------------------------------------------------------

/// Element semantic; order of the engine's name table `off_14229AB40`
/// (`VERTEX_POSITION` .. `VERTEX_SKININDICES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantic {
    Position,
    Normal,
    Tangent,
    Bitangent,
    TexCoord,
    Color,
    SkinWeights,
    SkinIndices,
}

/// Element formats used by the 13 declarations (DXGI mapping `sub_1402C2F50`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// R32G32B32_FLOAT
    F32x3,
    /// R32G32_FLOAT
    F32x2,
    /// R16G16_FLOAT
    F16x2,
    /// R16G16B16A16_FLOAT
    F16x4,
    /// R8G8B8A8_SNORM
    Snorm8x4,
    /// R8G8B8A8_UNORM
    Unorm8x4,
    /// R8G8B8A8_UINT
    Uint8x4,
}

impl Format {
    pub const fn size(self) -> usize {
        match self {
            Format::F32x3 => 12,
            Format::F32x2 | Format::F16x4 => 8,
            Format::F16x2 | Format::Snorm8x4 | Format::Unorm8x4 | Format::Uint8x4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Element {
    pub semantic: Semantic,
    pub index: u8,
    pub format: Format,
    /// Byte offset inside the element's slot.
    pub offset: u8,
    /// Input slot (vertex stream).
    pub slot: u8,
}

const fn el(semantic: Semantic, index: u8, format: Format, offset: u8, slot: u8) -> Element {
    Element {
        semantic,
        index,
        format,
        offset,
        slot,
    }
}

use Format::{F16x2, F16x4, F32x2, F32x3, Snorm8x4, Uint8x4, Unorm8x4};
use Semantic::{Color, Normal, Position, SkinIndices, SkinWeights, Tangent, TexCoord};

const POS: Element = el(Position, 0, F32x3, 0, 0);

/// Element lists per declaration id. Each 8-byte record of `sub_1402D0DC0` is
/// `u32 format | u8 semantic | u8 index | u8 offset | u8 slot`.
static DECLS: [&[Element]; 13] = [
    // 0
    &[POS],
    // 1: static models (.bmodel, E022)
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 16, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Tangent, 0, Snorm8x4, 24, 0),
        el(Color, 0, Unorm8x4, 28, 0),
    ],
    // 2 (texcoords 0, 1 and 4 share offset 12 in the engine table)
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 12, 0),
        el(TexCoord, 4, F16x4, 12, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Tangent, 0, Snorm8x4, 24, 0),
        el(Color, 0, Unorm8x4, 28, 0),
        el(TexCoord, 5, F16x4, 32, 0),
    ],
    // 3: 4 influences
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 16, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Tangent, 0, Snorm8x4, 24, 0),
        el(SkinWeights, 0, Unorm8x4, 28, 0),
        el(SkinIndices, 0, Uint8x4, 32, 0),
    ],
    // 4: 8 influences
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 16, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Tangent, 0, Snorm8x4, 24, 0),
        el(SkinWeights, 0, Unorm8x4, 28, 0),
        el(SkinWeights, 1, Unorm8x4, 32, 0),
        el(SkinIndices, 0, Uint8x4, 36, 0),
        el(SkinIndices, 1, Uint8x4, 40, 0),
    ],
    // 5
    &[POS, el(Normal, 0, F32x3, 12, 0)],
    // 6
    &[
        POS,
        el(TexCoord, 0, F32x2, 12, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Color, 0, Unorm8x4, 24, 0),
    ],
    // 7
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 16, 0),
        el(Color, 0, Snorm8x4, 20, 0),
        el(Normal, 0, Snorm8x4, 24, 0),
        el(Tangent, 0, Snorm8x4, 28, 0),
        el(SkinWeights, 0, Unorm8x4, 32, 0),
        el(SkinIndices, 0, Uint8x4, 36, 0),
        el(SkinWeights, 1, Unorm8x4, 40, 0),
        el(SkinIndices, 1, Uint8x4, 44, 0),
        el(TexCoord, 4, F16x4, 48, 0),
        el(TexCoord, 5, F16x4, 56, 0),
    ],
    // 8: declaration 1 split into two slots
    &[
        POS,
        el(TexCoord, 0, F16x2, 0, 1),
        el(TexCoord, 1, F16x2, 4, 1),
        el(Normal, 0, Snorm8x4, 8, 1),
        el(Tangent, 0, Snorm8x4, 12, 1),
        el(Color, 0, Unorm8x4, 16, 1),
    ],
    // 9: declaration 3 split into five slots
    &[
        POS,
        el(TexCoord, 0, F16x2, 0, 1),
        el(TexCoord, 1, F16x2, 4, 1),
        el(Normal, 0, Snorm8x4, 0, 2),
        el(Tangent, 0, Snorm8x4, 0, 3),
        el(SkinWeights, 0, Unorm8x4, 0, 4),
        el(SkinIndices, 0, Uint8x4, 4, 4),
    ],
    // 10: declaration 4 split into five slots (every observed .bmd6model)
    &[
        POS,
        el(TexCoord, 0, F16x2, 0, 1),
        el(TexCoord, 1, F16x2, 4, 1),
        el(Normal, 0, Snorm8x4, 0, 2),
        el(Tangent, 0, Snorm8x4, 0, 3),
        el(SkinWeights, 0, Unorm8x4, 0, 4),
        el(SkinWeights, 1, Unorm8x4, 4, 4),
        el(SkinIndices, 0, Uint8x4, 8, 4),
        el(SkinIndices, 1, Uint8x4, 12, 4),
    ],
    // 11: two positions (H: morph source)
    &[
        POS,
        el(Position, 1, F32x3, 0, 1),
        el(TexCoord, 0, F16x2, 0, 2),
        el(TexCoord, 1, F16x2, 4, 2),
        el(Normal, 0, Snorm8x4, 0, 3),
        el(Tangent, 0, Snorm8x4, 0, 4),
    ],
    // 12: declaration 1 + unorm8x4 texcoord 3
    &[
        POS,
        el(TexCoord, 0, F16x2, 12, 0),
        el(TexCoord, 1, F16x2, 16, 0),
        el(Normal, 0, Snorm8x4, 20, 0),
        el(Tangent, 0, Snorm8x4, 24, 0),
        el(Color, 0, Unorm8x4, 28, 0),
        el(TexCoord, 3, Unorm8x4, 32, 0),
    ],
];

/// Elements of declaration `id` (0..13), as built at startup by the engine.
pub fn declaration(id: i32) -> Option<&'static [Element]> {
    usize::try_from(id).ok().and_then(|i| DECLS.get(i)).copied()
}

/// Vertex size of a declaration: the engine sums the element sizes
/// (`sub_1402D1DA0`), independent of offsets.
pub fn stride(elements: &[Element]) -> usize {
    elements.iter().map(|e| e.format.size()).sum()
}

const MAX_SLOTS: usize = 8;

fn slot_strides(elements: &[Element]) -> [usize; MAX_SLOTS] {
    let mut s = [0usize; MAX_SLOTS];
    for e in elements {
        s[usize::from(e.slot)] += e.format.size();
    }
    s
}

/// Decoded vertex attributes. Each list is empty when the declaration lacks
/// the attribute, otherwise it has one entry per vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vertices {
    pub positions: Vec<[f32; 3]>,
    pub uv0: Vec<[f32; 2]>,
    pub uv1: Vec<[f32; 2]>,
    /// snorm8 (w as stored) or f32x3 (w = 0).
    pub normals: Vec<[f32; 4]>,
    /// snorm8; w is the bitangent sign as in the static formats.
    pub tangents: Vec<[f32; 4]>,
    pub colors: Vec<[u8; 4]>,
    /// Joint indices relative to [`Surface::joint_offset`]; entries 4..8 are
    /// zero for 4-influence declarations.
    pub joints: Vec<[u8; MAX_INFLUENCES]>,
    /// unorm8 weights paired with `joints` (sum 255 in every observed vertex).
    pub weights: Vec<[u8; MAX_INFLUENCES]>,
}

fn snorm(b: u8) -> f32 {
    (f32::from(b as i8) / 127.0).max(-1.0)
}

fn snorm4(b: &[u8]) -> [f32; 4] {
    [snorm(b[0]), snorm(b[1]), snorm(b[2]), snorm(b[3])]
}

fn half(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn float(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Decodes `count` vertices laid out slot after slot: slot `s` holds
/// `count * slot_stride(s)` bytes with its elements interleaved. Single-slot
/// declarations are therefore plain interleaved buffers.
pub fn decode_vertices(elements: &[Element], count: usize, data: &[u8]) -> Result<Vertices, Error> {
    let total = stride(elements)
        .checked_mul(count)
        .ok_or_else(|| invalid("vertex data", format!("{count} vertices overflow")))?;
    if data.len() != total {
        return Err(invalid(
            "vertex data",
            format!("{} bytes, declaration needs {total}", data.len()),
        ));
    }
    let strides = slot_strides(elements);
    let mut base = [0usize; MAX_SLOTS];
    for s in 1..MAX_SLOTS {
        base[s] = base[s - 1] + strides[s - 1] * count;
    }
    let mut out = Vertices::default();
    for e in elements {
        let slot = usize::from(e.slot);
        let size = e.format.size();
        if usize::from(e.offset) + size > strides[slot] {
            return Err(invalid(
                "vertex declaration",
                format!("{e:?} exceeds slot stride {}", strides[slot]),
            ));
        }
        let at = |v: usize| {
            let o = base[slot] + v * strides[slot] + usize::from(e.offset);
            &data[o..o + size]
        };
        let n = count;
        match (e.semantic, e.index, e.format) {
            (Position, 0, F32x3) => {
                out.positions = (0..n)
                    .map(|v| {
                        let b = at(v);
                        [float(b, 0), float(b, 4), float(b, 8)]
                    })
                    .collect();
            }
            (TexCoord, i @ (0 | 1), F16x2 | F32x2) => {
                let uv = (0..n)
                    .map(|v| {
                        let b = at(v);
                        match e.format {
                            F16x2 => [half(b, 0), half(b, 2)],
                            _ => [float(b, 0), float(b, 4)],
                        }
                    })
                    .collect();
                if i == 0 {
                    out.uv0 = uv;
                } else {
                    out.uv1 = uv;
                }
            }
            (Normal, 0, Snorm8x4) => out.normals = (0..n).map(|v| snorm4(at(v))).collect(),
            (Normal, 0, F32x3) => {
                out.normals = (0..n)
                    .map(|v| {
                        let b = at(v);
                        [float(b, 0), float(b, 4), float(b, 8), 0.0]
                    })
                    .collect();
            }
            (Tangent, 0, Snorm8x4) => {
                out.tangents = (0..n).map(|v| snorm4(at(v))).collect();
            }
            (Color, 0, Unorm8x4 | Snorm8x4) => {
                out.colors = (0..n).map(|v| at(v).try_into().expect("4 bytes")).collect();
            }
            (SkinWeights | SkinIndices, i @ (0 | 1), Unorm8x4 | Uint8x4) => {
                let dst = if e.semantic == SkinWeights {
                    &mut out.weights
                } else {
                    &mut out.joints
                };
                if dst.is_empty() {
                    *dst = vec![[0u8; MAX_INFLUENCES]; n];
                }
                let k = 4 * usize::from(i);
                for (v, d) in dst.iter_mut().enumerate() {
                    d[k..k + 4].copy_from_slice(at(v));
                }
            }
            // Extra texcoords and the second position are not decoded.
            _ => {}
        }
    }
    if out.positions.len() != count {
        return Err(invalid("vertex declaration", "no f32x3 POSITION0".into()));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Model (`MM6B`)
// ---------------------------------------------------------------------------

/// GPU stream header of a `BTRI` block with GPU data (none observed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub offset: u64,
    pub stride: u64,
    pub size: u64,
}

/// Index sub-range of a surface (dismemberment part).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRange {
    /// H: dismemberment body-part id (0..43 in the overseer body; the loader
    /// rewrites a lone range's 0 to 1 for surface kinds 1 and 3..6).
    pub part: u32,
    pub first: u32,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    pub name: String,
    pub material: String,
    /// Stored at surface +48; the loader tests for 1 and 3..6 (U meaning).
    pub kind: u32,
    /// Byte at surface +52 (1 in every sample).
    pub flag: u8,
    /// u32 at surface +56 (2 in every sample).
    pub unknown: u32,
    pub version: u32,
    pub cpu_decl: i32,
    pub gpu_decl: i32,
    /// 40-byte `BTRI` header; H: position scale/offset and uv scale/offset
    /// (always 1,1,1, 0,0,0, 1,1, 0,0 observed).
    pub header: [f32; 10],
    pub vertex_count: u32,
    pub streams: Vec<StreamInfo>,
    pub vertices: Vertices,
    pub indices: Vec<u16>,
    pub bounds: [f32; 6],
    pub tail: u32,
    pub ranges: Vec<IndexRange>,
    /// Added to every stored joint index (O: matches `used_joints`).
    pub joint_offset: u32,
    /// H: stored indices are below this (max stored index + 1 in samples).
    pub joint_count: u32,
    /// Bitset of skeleton joints with a non-zero weight (offset applied).
    pub used_joints: [u32; 8],
    /// The two u32 after the bitset (8 and 256 in every sample; H: word and
    /// bit count of the set).
    pub used_joints_size: [u32; 2],
}

impl Surface {
    /// Skeleton joint indices and normalized weights of vertex `v`.
    pub fn influences(&self, v: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let (j, w) = match (self.vertices.joints.get(v), self.vertices.weights.get(v)) {
            (Some(j), Some(w)) => (*j, *w),
            _ => ([0; MAX_INFLUENCES], [0; MAX_INFLUENCES]),
        };
        (0..MAX_INFLUENCES)
            .filter(move |&k| w[k] != 0)
            .map(move |k| (u32::from(j[k]) + self.joint_offset, f32::from(w[k]) / 255.0))
    }

    pub fn uses_joint(&self, joint: u32) -> bool {
        joint < 256 && self.used_joints[(joint / 32) as usize] >> (joint % 32) & 1 != 0
    }

    fn parse(r: &mut Reader) -> Result<Self, Error> {
        let name = string(r, "surface name")?;
        let material = string(r, "surface material")?;
        let kind = r.u32("surface kind")?;
        let flag = r.u8("surface flag")?;
        let unknown = r.u32("surface field")?;
        let tag: [u8; 4] = r.array("BTRI tag")?;
        if &tag != b"BTRI" {
            return Err(Error::BadMagic {
                expected: *b"BTRI",
                found: tag,
            });
        }
        let version = r.u32("BTRI version")?;
        if version <= 0x10001 {
            return Err(Error::UnsupportedVersion(version));
        }
        let cpu_decl = r.u32("cpu decl")? as i32;
        let gpu_decl = r.u32("gpu decl")? as i32;
        for d in [cpu_decl, gpu_decl] {
            if d >= 0 && declaration(d).is_none() {
                return Err(invalid("vertex declaration", d.to_string()));
            }
        }
        let header = f32s::<10>(r, "BTRI header")?;
        let vertex_count = r.u32("vertex count")?;
        if vertex_count > 0xFFFF {
            return Err(invalid("vertex count", vertex_count.to_string()));
        }
        let gpu_bytes =
            u64::from(r.u32("gpu bytes lo")?) | (u64::from(r.u32("gpu bytes hi")?) << 32);
        // The CPU copy uses the CPU declaration, falling back to the GPU one.
        let decl_id = if cpu_decl >= 0 { cpu_decl } else { gpu_decl };
        let elements = declaration(decl_id)
            .ok_or_else(|| invalid("vertex declaration", decl_id.to_string()))?;
        let size = stride(elements) * vertex_count as usize;
        let mut streams = Vec::new();
        let vertices = if gpu_bytes > 0 {
            if cpu_decl != gpu_decl {
                return Err(invalid(
                    "vertex declaration",
                    format!(
                        "GPU streams with cpu {cpu_decl} != gpu {gpu_decl} (conversion not decoded)"
                    ),
                ));
            }
            let n = r.u8("stream count")?;
            if n > 15 {
                return Err(invalid("stream count", n.to_string()));
            }
            let mut data = Vec::new();
            for _ in 0..n {
                let mut q = [0u64; 3];
                for v in &mut q {
                    *v = u64::from(r.u32("stream field lo")?)
                        | (u64::from(r.u32("stream field hi")?) << 32);
                }
                let [offset, stride, size] = q;
                if size > r.remaining() as u64 || offset >= size.max(1) || stride > size {
                    return Err(invalid(
                        "vertex stream",
                        format!("offset {offset} stride {stride} size {size}"),
                    ));
                }
                // With equal declarations the engine copies the streams back to
                // back into the CPU buffer (memcpy_s in sub_14025A040).
                data.extend_from_slice(r.bytes(size as usize, "stream data")?);
                streams.push(StreamInfo {
                    offset,
                    stride,
                    size,
                });
            }
            decode_vertices(elements, vertex_count as usize, &data)?
        } else {
            decode_vertices(
                elements,
                vertex_count as usize,
                r.bytes(size, "cpu vertices")?,
            )?
        };
        let ni = r.count(2, "index count")?;
        let indices: Vec<u16> = r
            .bytes(2 * ni, "indices")?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        if let Some(&bad) = indices.iter().find(|&&i| u32::from(i) >= vertex_count) {
            return Err(invalid(
                "index",
                format!("{bad} >= vertex count {vertex_count}"),
            ));
        }
        let bounds = f32s::<6>(r, "BTRI bounds")?;
        let tail = r.u32("BTRI tail")?;
        let nr = r.count(12, "index range count")?;
        let mut ranges = Vec::with_capacity(nr);
        for _ in 0..nr {
            let range = IndexRange {
                part: r.u32("range part")?,
                first: r.u32("range first")?,
                count: r.u32("range count")?,
            };
            if u64::from(range.first) + u64::from(range.count) > ni as u64 {
                return Err(invalid(
                    "index range",
                    format!("{range:?} beyond {ni} indices"),
                ));
            }
            ranges.push(range);
        }
        let joint_offset = r.u32("joint offset")?;
        let joint_count = r.u32("joint count")?;
        let mut used_joints = [0u32; 8];
        for w in &mut used_joints {
            *w = r.u32("used joints")?;
        }
        let used_joints_size = [r.u32("used joints size")?, r.u32("used joints size")?];
        Ok(Self {
            name,
            material,
            kind,
            flag,
            unknown,
            version,
            cpu_decl,
            gpu_decl,
            header,
            vertex_count,
            streams,
            vertices,
            indices,
            bounds,
            tail,
            ranges,
            joint_offset,
            joint_count,
            used_joints,
            used_joints_size,
        })
    }
}

/// Model-level material record (24-byte runtime entry).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialEntry {
    pub material: String,
    /// H: surface index.
    pub surface: u32,
    /// H: lowest and highest vertex of that surface (0 and count - 1 in samples).
    pub first_vertex: u32,
    pub last_vertex: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HelperAttribute {
    /// u16 as stored; the loader keeps the low byte.
    pub channel: u16,
    /// u32 as stored; the loader keeps the low 3 bits.
    pub kind: u32,
    pub values: [f32; 3],
}

/// Joint helper (`arkJointHelperCollection`, named `arkhelper*` joints).
#[derive(Clone, Debug, PartialEq)]
pub struct JointHelper {
    pub name: String,
    pub attributes: Vec<HelperAttribute>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// `.md6skl` path; empty in cloth-split (`nocloth_*`) and cloth-cache models.
    pub skeleton: String,
    /// Present with a skeleton; equals the skeleton's [`Skeleton::model_checksum`]
    /// (O: all models of the eotw snapshot; the loader reads it, `sub_14025AA00`).
    pub skeleton_checksum: Option<u16>,
    pub bounds: [f32; 6],
    pub surfaces: Vec<Surface>,
    pub materials: Vec<MaterialEntry>,
    pub joint_helpers: Vec<JointHelper>,
    /// Bit 0 (H: release CPU vertex copies after upload); 1 in cloth caches.
    pub flags: u8,
}

impl Model {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("md6 magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let skeleton = string(&mut r, "skeleton name")?;
        let skeleton_checksum = if skeleton.is_empty() {
            None
        } else {
            Some(r.u16("skeleton checksum")?)
        };
        let bounds = f32s::<6>(&mut r, "model bounds")?;
        // Smallest surface: 2 empty strings + fixed fields + BTRI + trailer.
        let ns = r.count(4 * 2 + 9 + 12 + 40 + 12 + 4 + 28 + 4 + 48, "surface count")?;
        let mut surfaces = Vec::with_capacity(ns);
        for _ in 0..ns {
            surfaces.push(Surface::parse(&mut r)?);
        }
        let nm = r.count(16, "material count")?;
        let mut materials = Vec::with_capacity(nm);
        for _ in 0..nm {
            materials.push(MaterialEntry {
                material: string(&mut r, "material name")?,
                surface: r.u32("material surface")?,
                first_vertex: r.u32("material first vertex")?,
                last_vertex: r.u32("material last vertex")?,
            });
        }
        let nh = r.count(8, "joint helper count")?;
        let mut joint_helpers = Vec::with_capacity(nh);
        for _ in 0..nh {
            let name = string(&mut r, "joint helper name")?;
            let na = r.count(18, "helper attribute count")?;
            let mut attributes = Vec::with_capacity(na);
            for _ in 0..na {
                attributes.push(HelperAttribute {
                    channel: r.u16("helper channel")?,
                    kind: r.u32("helper kind")?,
                    values: f32s::<3>(&mut r, "helper values")?,
                });
            }
            joint_helpers.push(JointHelper { name, attributes });
        }
        let flags = r.u8("model flags")?;
        let end: [u8; 4] = r.array("md6 end magic")?;
        if end != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: end,
            });
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        Ok(Self {
            skeleton,
            skeleton_checksum,
            bounds,
            surfaces,
            materials,
            joint_helpers,
            flags,
        })
    }
}

// ---------------------------------------------------------------------------
// Skeleton (`60SE`, Sony Edge animation skeleton + arkSkelBase custom data)
// ---------------------------------------------------------------------------

/// Local joint transform from the Edge base pose (48 bytes per joint).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Quaternion x, y, z, w.
    pub rotation: [f32; 4],
    pub translation: [f32; 3],
    pub scale: [f32; 3],
}

impl Transform {
    /// Row-major 3x4 matrix `[R*S | t]` (column vectors).
    pub fn matrix(&self) -> Mat34 {
        let [x, y, z, w] = self.rotation;
        let r = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        let mut m = [[0f32; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j] * self.scale[j];
            }
            m[i][3] = self.translation[i];
        }
        m
    }
}

/// Row-major affine 3x4 matrix (implicit last row 0 0 0 1).
pub type Mat34 = [[f32; 4]; 3];

pub fn mul34(a: &Mat34, b: &Mat34) -> Mat34 {
    let mut m = [[0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            m[i][j] =
                (0..3).map(|k| a[i][k] * b[k][j]).sum::<f32>() + if j == 3 { a[i][3] } else { 0.0 };
        }
    }
    m
}

pub fn transform_point(m: &Mat34, p: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3])
}

#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub name: String,
    pub parent: Option<u16>,
    pub local: Transform,
    /// Inverse bind matrix from the custom data (O: inverse of the global
    /// base pose to <= 3.9e-6 in all 288 shipped skeleton names).
    pub inverse_bind: Mat34,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    pub num_user_channels: u16,
    /// u32 at header +0x14 (2 in every sample).
    pub flags: u32,
    /// The three u32 at header +0x34..+0x40 (U).
    pub header_tail: [u32; 3],
    /// First u32 of the custom data (14 in every sample).
    pub custom_version: u32,
    pub parent_skeleton: String,
    pub parent_checksum: Option<u16>,
    /// u16 at skeleton +336 (U: differs from `model_checksum` in skeletons
    /// with a parent skeleton).
    pub checksum: u16,
    /// u16 at skeleton +338; models store it ([`Model::skeleton_checksum`]).
    pub model_checksum: u16,
    /// H: per-LOD joint counts (up to 4 entries).
    pub lod_joint_counts: Vec<u16>,
    /// H: joint order list read together with `lod_joint_counts`.
    pub joint_order: Vec<u16>,
    pub joints: Vec<Joint>,
    pub channel_names: Vec<String>,
}

const SKELETON_HEADER: usize = 64;

impl Skeleton {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let tag: [u8; 4] = r.array("skeleton tag")?;
        if tag != SKELETON_TAG {
            return Err(Error::BadMagic {
                expected: SKELETON_TAG,
                found: tag,
            });
        }
        let size_total = r.u32("skeleton size")? as usize;
        if size_total != data.len() {
            return Err(invalid(
                "skeleton size",
                format!("header {size_total}, file {}", data.len()),
            ));
        }
        let size_custom = r.u32("custom data size")? as usize;
        let _size_name_hashes = r.u32("name hash size")?;
        let num_joints = usize::from(r.u16("joint count")?);
        let num_user_channels = r.u16("user channel count")?;
        let flags = r.u32("skeleton flags")?;
        // Self-relative offsets (EDGE_OFFSET style): field position + value.
        let mut offsets = [0usize; 7];
        for (i, o) in offsets.iter_mut().enumerate() {
            let field = 0x18 + 4 * i;
            *o = field + r.u32("skeleton offset")? as usize;
            if *o > data.len() {
                return Err(invalid(
                    "skeleton offset",
                    format!("field {field:#x} -> {o:#x}"),
                ));
            }
        }
        let header_tail = [
            r.u32("skeleton header")?,
            r.u32("skeleton header")?,
            r.u32("skeleton header")?,
        ];
        let base_pose = offsets[0];
        let custom = offsets[6];
        if base_pose < SKELETON_HEADER
            || base_pose + 48 * num_joints > data.len()
            || custom
                .checked_add(size_custom)
                .is_none_or(|e| e > data.len())
        {
            return Err(invalid(
                "skeleton layout",
                format!("base pose {base_pose:#x}, custom {custom:#x}+{size_custom:#x}"),
            ));
        }
        // H: Edge SIMD hierarchy, (joint, parent) u16 pairs in quads between the
        // header and the base pose; 0x7fff = root, padding repeats entries.
        let hier = &data[SKELETON_HEADER..base_pose];
        if !hier.len().is_multiple_of(4) {
            return Err(invalid("joint hierarchy", format!("{} bytes", hier.len())));
        }
        let mut parents: Vec<Option<Option<u16>>> = vec![None; num_joints];
        for p in hier.as_chunks::<4>().0 {
            let j = usize::from(u16::from_le_bytes([p[0], p[1]]));
            let raw = u16::from_le_bytes([p[2], p[3]]);
            let parent = (raw != 0x7fff).then_some(raw);
            let slot = parents
                .get_mut(j)
                .ok_or_else(|| invalid("joint hierarchy", format!("joint {j} >= {num_joints}")))?;
            if parent.is_some_and(|p| usize::from(p) >= j) || slot.is_some_and(|q| q != parent) {
                return Err(invalid(
                    "joint hierarchy",
                    format!("joint {j} parent {raw:#x}"),
                ));
            }
            *slot = Some(parent);
        }
        let mut pr = Reader::new(&data[base_pose..base_pose + 48 * num_joints]);
        let mut locals = Vec::with_capacity(num_joints);
        for _ in 0..num_joints {
            let rotation = f32s::<4>(&mut pr, "base pose rotation")?;
            let t = f32s::<4>(&mut pr, "base pose translation")?;
            let s = f32s::<4>(&mut pr, "base pose scale")?;
            locals.push(Transform {
                rotation,
                translation: [t[0], t[1], t[2]],
                scale: [s[0], s[1], s[2]],
            });
        }

        // arkSkelBase custom data (sub_1402564E0).
        let mut c = Reader::new(&data[custom..custom + size_custom]);
        let custom_version = c.u32("custom version")?;
        let parent_skeleton = string(&mut c, "parent skeleton")?;
        let parent_checksum = if parent_skeleton.is_empty() {
            None
        } else {
            Some(c.u16("parent checksum")?)
        };
        let checksum = c.u16("skeleton checksum")?;
        let model_checksum = c.u16("skeleton model checksum")?;
        let n = usize::from(c.u16("custom joint count")?);
        if n != num_joints {
            return Err(invalid(
                "custom joint count",
                format!("{n} != header {num_joints}"),
            ));
        }
        let nl = usize::from(c.u16("lod joint count count")?);
        if nl > 4 {
            return Err(invalid("lod joint counts", nl.to_string()));
        }
        let mut lod_joint_counts = Vec::new();
        let mut joint_order = Vec::new();
        if nl != 0 {
            for _ in 0..nl {
                lod_joint_counts.push(c.u16("lod joint count")?);
            }
            for _ in 0..n {
                joint_order.push(c.u16("joint order")?);
            }
        }
        let mut names = Vec::with_capacity(n);
        for _ in 0..n {
            names.push(string(&mut c, "joint name")?);
        }
        c.bytes(10 * n, "joint flags")?;
        let nc = usize::from(c.u16("channel count")?);
        let mut channel_names = Vec::with_capacity(nc.min(c.remaining() / 4));
        for _ in 0..nc {
            channel_names.push(string(&mut c, "channel name")?);
        }
        c.bytes(10 * nc, "channel flags")?;
        // Inverse bind matrices, padded to a multiple of 8 joints.
        let padded = n.div_ceil(8) * 8;
        let mut inverse = Vec::with_capacity(n);
        for k in 0..padded {
            let m = f32s::<12>(&mut c, "inverse bind")?;
            if k < n {
                inverse.push([
                    [m[0], m[1], m[2], m[3]],
                    [m[4], m[5], m[6], m[7]],
                    [m[8], m[9], m[10], m[11]],
                ]);
            }
        }
        let joints = names
            .into_iter()
            .zip(locals)
            .zip(inverse)
            .zip(parents)
            .enumerate()
            .map(|(j, (((name, local), inverse_bind), parent))| {
                parent
                    .map(|parent| Joint {
                        name,
                        parent,
                        local,
                        inverse_bind,
                    })
                    .ok_or_else(|| invalid("joint hierarchy", format!("joint {j} has no entry")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            num_user_channels,
            flags,
            header_tail,
            custom_version,
            parent_skeleton,
            parent_checksum,
            checksum,
            model_checksum,
            lod_joint_counts,
            joint_order,
            joints,
            channel_names,
        })
    }

    /// Global bind pose per joint (parents precede children, checked on parse).
    pub fn bind_pose(&self) -> Vec<Mat34> {
        self.global_pose(|j| self.joints[j].local.matrix())
    }

    /// Global (model-space) joint matrices for the local matrices `local(j)`.
    pub fn global_pose(&self, local: impl Fn(usize) -> Mat34) -> Vec<Mat34> {
        let mut out: Vec<Mat34> = Vec::with_capacity(self.joints.len());
        for (k, j) in self.joints.iter().enumerate() {
            let m = match j.parent {
                Some(p) => mul34(&out[usize::from(p)], &local(k)),
                None => local(k),
            };
            out.push(m);
        }
        out
    }

    /// Largest element of `|inverse_bind * bind - I|` over all joints.
    pub fn inverse_bind_error(&self) -> f32 {
        let id: Mat34 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        self.bind_pose()
            .iter()
            .zip(&self.joints)
            .flat_map(|(g, j)| {
                let p = mul34(&j.inverse_bind, g);
                (0..3).flat_map(move |i| (0..4).map(move |k| (p[i][k] - id[i][k]).abs()))
            })
            .fold(0.0, f32::max)
    }
}

// ---------------------------------------------------------------------------
// md6Def decl subset
// ---------------------------------------------------------------------------

/// `cloth "<hkt>" "<setup>" ...` inside a mesh block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClothRef {
    pub args: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshRef {
    /// `.md6mesh` path.
    pub mesh: String,
    pub cloth: Vec<ClothRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lod {
    pub level: u32,
    /// `clear`: drop meshes inherited for this level.
    pub clear: bool,
    pub meshes: Vec<MeshRef>,
}

/// The `init` block of an `md6Def` decl (inherit, skeleton, lod meshes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Md6Def {
    pub inherit: Option<String>,
    pub skeleton: Option<String>,
    pub lods: Vec<Lod>,
}

fn decl_err(line: u32, detail: &str) -> Error {
    invalid("md6Def", format!("line {line}: {detail}"))
}

/// Index after the block whose `Open` token is at `open`.
fn skip_block(t: &[crate::decl::Token], open: usize) -> Result<usize, Error> {
    let mut depth = 0usize;
    for (i, tok) in t.iter().enumerate().skip(open) {
        match tok.tok {
            Tok::Open => depth += 1,
            Tok::Close => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i + 1);
                }
            }
            _ => {}
        }
    }
    Err(decl_err(t[open].line, "unterminated block"))
}

fn quoted(t: &[crate::decl::Token], i: usize) -> Result<String, Error> {
    match t.get(i).map(|x| &x.tok) {
        Some(Tok::Str(s)) => Ok(s.clone()),
        _ => Err(decl_err(t.get(i).map_or(0, |x| x.line), "expected string")),
    }
}

impl Md6Def {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let t = tokenize(text)?;
        if t.first().map(|x| &x.tok) != Some(&Tok::Open) {
            return Err(decl_err(1, "expected '{'"));
        }
        let end = skip_block(&t, 0)?;
        let mut def = Md6Def::default();
        let mut i = 1;
        while i < end - 1 {
            match &t[i].tok {
                Tok::Word(w) if w == "init" && t.get(i + 1).map(|x| &x.tok) == Some(&Tok::Open) => {
                    let close = skip_block(&t, i + 1)? - 1;
                    def.parse_init(&t[..close], i + 2)?;
                    i = close + 1;
                }
                Tok::Open => i = skip_block(&t, i)?,
                _ => i += 1,
            }
        }
        Ok(def)
    }

    /// Parses tokens `t[start..]` (the init block body, `t` ends at its `}`).
    fn parse_init(&mut self, t: &[crate::decl::Token], start: usize) -> Result<(), Error> {
        let mut i = start;
        while i < t.len() {
            match &t[i].tok {
                Tok::Word(w) if w == "inherit" => {
                    self.inherit = Some(quoted(t, i + 1)?);
                    i += 2;
                }
                Tok::Word(w) if w == "skeleton" => {
                    self.skeleton = Some(quoted(t, i + 1)?);
                    i += 2;
                }
                Tok::Word(w) if w == "lod" => {
                    let level = match t.get(i + 1).map(|x| &x.tok) {
                        Some(Tok::Word(n)) => n.parse::<u32>().ok(),
                        _ => None,
                    }
                    .ok_or_else(|| decl_err(t[i].line, "lod without level"))?;
                    if t.get(i + 2).map(|x| &x.tok) != Some(&Tok::Open) {
                        return Err(decl_err(t[i].line, "lod without block"));
                    }
                    let close = skip_block(t, i + 2)? - 1;
                    self.lods.push(Self::parse_lod(level, &t[i + 3..close])?);
                    i = close + 1;
                }
                Tok::Open => i = skip_block(t, i)?,
                _ => i += 1,
            }
        }
        Ok(())
    }

    fn parse_lod(level: u32, t: &[crate::decl::Token]) -> Result<Lod, Error> {
        let mut lod = Lod {
            level,
            clear: false,
            meshes: Vec::new(),
        };
        let mut i = 0;
        while i < t.len() {
            match &t[i].tok {
                Tok::Word(w) if w == "clear" => {
                    lod.clear = true;
                    i += 1;
                }
                Tok::Word(w) if w == "mesh" => {
                    let mut mesh = MeshRef {
                        mesh: quoted(t, i + 1)?,
                        cloth: Vec::new(),
                    };
                    i += 2;
                    if t.get(i).map(|x| &x.tok) == Some(&Tok::Open) {
                        let close = skip_block(t, i)? - 1;
                        let mut k = i + 1;
                        while k < close {
                            if t[k].tok == Tok::Word("cloth".into()) {
                                let mut args = Vec::new();
                                k += 1;
                                while let Some(Tok::Str(s)) =
                                    t.get(k).filter(|_| k < close).map(|x| &x.tok)
                                {
                                    args.push(s.clone());
                                    k += 1;
                                }
                                mesh.cloth.push(ClothRef { args });
                            } else {
                                k += 1;
                            }
                        }
                        i = close + 1;
                    }
                    lod.meshes.push(mesh);
                }
                Tok::Open => i = skip_block(t, i)?,
                _ => i += 1,
            }
        }
        Ok(lod)
    }
}

/// Resource name of an md6Def decl (`models/x/y.md6` ->
/// `generated/decls/md6def/models/x/y.md6.decl`).
pub fn def_resource(md6: &str) -> String {
    format!("generated/decls/md6def/{}.decl", md6.to_ascii_lowercase())
}

/// Built mesh resource (`x.md6mesh` -> `generated/basemodel/x.bmd6model`).
pub fn mesh_resource(mesh: &str) -> Option<String> {
    let stem = mesh.to_ascii_lowercase();
    let stem = stem.strip_suffix(".md6mesh")?;
    Some(format!("generated/basemodel/{stem}.bmd6model"))
}

/// Built skeleton resource (`x.md6skl` -> `generated/skeleton/x.edgeskl`).
pub fn skeleton_resource(skl: &str) -> Option<String> {
    let stem = skl.to_ascii_lowercase();
    let stem = stem.strip_suffix(".md6skl")?;
    Some(format!("generated/skeleton/{stem}.edgeskl"))
}

/// True if `name` is the cloth-stripped build of `mesh`:
/// `nocloth_<mesh file stem>_<32 hex>.bmd6model` (format string at
/// 0x141AD2C70, built by `sub_140088150`; the hash is an MD5 over name hashes
/// and is not recomputed here).
pub fn is_nocloth_build(name: &str, mesh: &str) -> bool {
    let mesh = mesh.to_ascii_lowercase();
    let Some(stem) = mesh.strip_suffix(".md6mesh") else {
        return false;
    };
    let base = stem.rsplit('/').next().unwrap_or(stem);
    name.strip_prefix("nocloth_")
        .and_then(|r| r.strip_prefix(base))
        .and_then(|r| r.strip_prefix('_'))
        .and_then(|r| r.strip_suffix(".bmd6model"))
        .is_some_and(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Merged `init` data of an md6Def and its `inherit` chain (child first in `chain`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedDef {
    pub chain: Vec<String>,
    /// The last `skeleton` along the chain from the root (children override).
    pub skeleton: Option<String>,
    /// LOD level -> meshes: a child's `lod N` adds to the parent's unless it says `clear`.
    pub lods: std::collections::BTreeMap<u32, Vec<MeshRef>>,
}

/// Resolves the md6Def decl `md6` and its `inherit` chain (at most 8 deep); `read` returns
/// the bytes of an archive resource name ([`def_resource`]).
pub fn resolve_def(
    read: &dyn Fn(&str) -> Result<Vec<u8>, String>,
    md6: &str,
) -> Result<ResolvedDef, String> {
    resolve_def_at(read, md6, 0)
}

fn resolve_def_at(
    read: &dyn Fn(&str) -> Result<Vec<u8>, String>,
    md6: &str,
    depth: usize,
) -> Result<ResolvedDef, String> {
    if depth > 8 {
        return Err(format!("{md6}: inherit chain deeper than 8"));
    }
    let res = def_resource(md6);
    let data = read(&res).map_err(|e| format!("{res}: {e}"))?;
    let text = std::str::from_utf8(&data).map_err(|e| format!("{res}: {e}"))?;
    let def = Md6Def::parse(text).map_err(|e| format!("{res}: {e}"))?;
    let mut out = match &def.inherit {
        Some(parent) => resolve_def_at(read, parent, depth + 1)?,
        None => ResolvedDef::default(),
    };
    out.chain.insert(0, md6.to_owned());
    if def.skeleton.is_some() {
        out.skeleton = def.skeleton;
    }
    for lod in def.lods {
        let meshes = out.lods.entry(lod.level).or_default();
        if lod.clear {
            meshes.clear();
        }
        meshes.extend(lod.meshes);
    }
    Ok(out)
}

/// The mesh file stem a `nocloth_<stem>_<32 hex>.bmd6model` name was built from.
pub fn nocloth_stem(name: &str) -> Option<&str> {
    let stem = name
        .strip_prefix("nocloth_")?
        .strip_suffix(".bmd6model")
        .and_then(|r| r.len().checked_sub(33).map(|n| &r[..n]))?;
    is_nocloth_build(name, &format!("{stem}.md6mesh")).then_some(stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(o: &mut Vec<u8>, v: u32) {
        o.extend_from_slice(&v.to_le_bytes());
    }
    fn put16(o: &mut Vec<u8>, v: u16) {
        o.extend_from_slice(&v.to_le_bytes());
    }
    fn putf(o: &mut Vec<u8>, v: &[f32]) {
        for f in v {
            o.extend_from_slice(&f.to_le_bytes());
        }
    }
    fn s(o: &mut Vec<u8>, t: &str) {
        put(o, t.len() as u32);
        o.extend_from_slice(t.as_bytes());
    }

    /// One triangle, declaration 10 (five planar slots), joint offset 2.
    fn model() -> Vec<u8> {
        let mut o = MAGIC.to_vec();
        s(&mut o, "models/x/x.md6skl");
        put16(&mut o, 0x1542);
        putf(&mut o, &[0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        put(&mut o, 1);
        s(&mut o, "body");
        s(&mut o, "models/x/x.material");
        put(&mut o, 6);
        o.push(1);
        put(&mut o, 2);
        o.extend_from_slice(b"BTRI");
        put(&mut o, 0x10002);
        put(&mut o, 10);
        put(&mut o, 10);
        putf(&mut o, &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        put(&mut o, 3);
        put(&mut o, 0);
        put(&mut o, 0);
        // slot 0: positions
        putf(&mut o, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
        // slot 1: uv0 + uv1 (f16 1.0 = 0x3c00, 0.5 = 0x3800)
        for _ in 0..3 {
            for h in [0x3c00u16, 0x3800, 0, 0] {
                put16(&mut o, h);
            }
        }
        // slot 2: normals, slot 3: tangents
        for _ in 0..3 {
            o.extend_from_slice(&[0, 0, 127, 0]);
        }
        for _ in 0..3 {
            o.extend_from_slice(&[127, 0, 0, 0x81]);
        }
        // slot 4: weights0, weights1, indices0, indices1
        for v in 0..3u8 {
            o.extend_from_slice(&[200, 55, 0, 0, 0, 0, 0, 0, v, 3, 0, 0, 0, 0, 0, 0]);
        }
        put(&mut o, 3);
        for i in [0u16, 1, 2] {
            put16(&mut o, i);
        }
        putf(&mut o, &[0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        put(&mut o, 0);
        put(&mut o, 1);
        put(&mut o, 0);
        put(&mut o, 0);
        put(&mut o, 3);
        put(&mut o, 2); // joint offset
        put(&mut o, 4); // joint count
        put(&mut o, 0b11_1100); // joints 2, 3, 4, 5
        for _ in 0..7 {
            put(&mut o, 0);
        }
        put(&mut o, 8);
        put(&mut o, 256);
        put(&mut o, 1);
        s(&mut o, "models/x/x.material");
        put(&mut o, 0);
        put(&mut o, 0);
        put(&mut o, 2);
        put(&mut o, 1);
        s(&mut o, "arkhelper4_x_jnt");
        put(&mut o, 1);
        put16(&mut o, 3);
        put(&mut o, 1);
        putf(&mut o, &[-75.0, 0.0, 40.0]);
        o.push(0);
        o.extend_from_slice(&MAGIC);
        o
    }

    #[test]
    fn parses_planar_skinned_surface() {
        let m = Model::parse(&model()).unwrap();
        assert_eq!(m.skeleton, "models/x/x.md6skl");
        assert_eq!(m.skeleton_checksum, Some(0x1542));
        let sf = &m.surfaces[0];
        assert_eq!((sf.kind, sf.cpu_decl, sf.vertex_count), (6, 10, 3));
        let v = &sf.vertices;
        assert_eq!(v.positions[2], [0.0, 1.0, 2.0]);
        assert_eq!(v.uv0[1], [1.0, 0.5]);
        assert_eq!(v.uv1[1], [0.0, 0.0]);
        assert_eq!(v.normals[0], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(v.tangents[0], [1.0, 0.0, 0.0, -1.0]);
        assert_eq!(v.weights[0], [200, 55, 0, 0, 0, 0, 0, 0]);
        assert_eq!(v.joints[2], [2, 3, 0, 0, 0, 0, 0, 0]);
        let inf: Vec<_> = sf.influences(1).collect();
        assert_eq!(inf, [(3, 200.0 / 255.0), (5, 55.0 / 255.0)]);
        assert!(sf.uses_joint(5) && !sf.uses_joint(6));
        assert_eq!(
            sf.ranges,
            [IndexRange {
                part: 0,
                first: 0,
                count: 3
            }]
        );
        assert_eq!(m.materials[0].last_vertex, 2);
        assert_eq!(m.joint_helpers[0].attributes[0].values, [-75.0, 0.0, 40.0]);
    }

    #[test]
    fn rejects_truncation_trailing_and_bad_indices() {
        let full = model();
        for cut in 0..full.len() {
            assert!(Model::parse(&full[..cut]).is_err(), "cut {cut}");
        }
        let mut t = full.clone();
        t.push(0);
        assert!(matches!(Model::parse(&t), Err(Error::TrailingData { .. })));
        // Index 2 -> 9 (>= vertex count).
        let mut b = full.clone();
        let tail = b.windows(4).rposition(|w| w == [2, 0, 0, 0]).unwrap();
        let idx = b
            .windows(10)
            .position(|w| w == [3, 0, 0, 0, 0, 0, 1, 0, 2, 0])
            .unwrap();
        assert!(idx < tail);
        b[idx + 8] = 9;
        assert!(Model::parse(&b).is_err());
        // Unknown vertex declaration.
        let mut d = full;
        let btri = d.windows(4).position(|w| w == b"BTRI").unwrap();
        d[btri + 8] = 13;
        assert!(Model::parse(&d).is_err());
    }

    #[test]
    fn declaration_strides_match_engine_sums() {
        let strides: Vec<_> = (0..13).map(|d| stride(declaration(d).unwrap())).collect();
        assert_eq!(
            strides,
            [12, 32, 48, 36, 44, 24, 28, 64, 32, 36, 44, 40, 36]
        );
        assert!(declaration(13).is_none() && declaration(-1).is_none());
    }

    /// Two joints: root at origin, child translated by (0, 0, 1) and rotated
    /// 90 degrees about z.
    fn skeleton() -> Vec<u8> {
        let names = ["origin", "child_jnt"];
        let mut custom = Vec::new();
        put(&mut custom, 14);
        put(&mut custom, 0);
        put16(&mut custom, 0x75cd);
        put16(&mut custom, 0x1542);
        put16(&mut custom, 2);
        put16(&mut custom, 0);
        for n in names {
            s(&mut custom, n);
        }
        custom.extend_from_slice(&[0; 20]);
        put16(&mut custom, 0);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        // Inverse binds: identity; inverse of [Rz90 | (0,0,1)] = [Rz-90 | (0,0,-1)].
        putf(
            &mut custom,
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        putf(
            &mut custom,
            &[0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        );
        custom.extend_from_slice(&[0; 6 * 48]);
        custom.extend_from_slice(&[0xAA; 8]); // unparsed helper data

        let hier: [u16; 8] = [0, 0x7fff, 1, 0, 1, 0, 1, 0];
        let base = 64 + hier.len() * 2;
        let custom_at = base + 2 * 48;
        let total = custom_at + custom.len() + 4;
        let mut o = SKELETON_TAG.to_vec();
        put(&mut o, total as u32);
        put(&mut o, custom.len() as u32);
        put(&mut o, 4);
        put16(&mut o, 2);
        put16(&mut o, 0);
        put(&mut o, 2);
        let rel = |field: usize, at: usize| (at - field) as u32;
        put(&mut o, rel(0x18, base));
        for f in [0x1c, 0x20, 0x24, 0x28, 0x2c] {
            put(&mut o, rel(f, custom_at));
        }
        put(&mut o, rel(0x30, custom_at));
        put(&mut o, 0);
        put(&mut o, 0);
        put(&mut o, 0);
        assert_eq!(o.len(), 64);
        for v in hier {
            put16(&mut o, v);
        }
        putf(
            &mut o,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        putf(
            &mut o,
            &[0.0, 0.0, h, h, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        o.extend_from_slice(&custom);
        put(&mut o, 0);
        assert_eq!(o.len(), total);
        o
    }

    #[test]
    fn parses_edge_skeleton_and_checks_bind_pose() {
        let sk = Skeleton::parse(&skeleton()).unwrap();
        assert_eq!((sk.checksum, sk.model_checksum), (0x75cd, 0x1542));
        assert_eq!(sk.joints.len(), 2);
        assert_eq!(sk.joints[1].name, "child_jnt");
        assert_eq!(sk.joints[0].parent, None);
        assert_eq!(sk.joints[1].parent, Some(0));
        let g = sk.bind_pose();
        let p = transform_point(&g[1], [1.0, 0.0, 0.0]);
        assert!((p[0]).abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6 && (p[2] - 1.0).abs() < 1e-6);
        assert!(sk.inverse_bind_error() < 1e-6);
    }

    #[test]
    fn rejects_bad_skeletons() {
        let full = skeleton();
        for cut in 0..full.len() {
            assert!(Skeleton::parse(&full[..cut]).is_err(), "cut {cut}");
        }
        // Parent after child.
        let mut b = full.clone();
        b[64 + 6] = 1;
        assert!(Skeleton::parse(&b).is_err());
        // Joint index out of range.
        let mut c = full;
        c[64 + 4] = 7;
        assert!(Skeleton::parse(&c).is_err());
    }

    #[test]
    fn parses_md6def_lods_with_inherit_and_cloth() {
        let text = r#"{
	init {
		inherit "models/characters/tall/base/base_body.md6"
		skeleton "models/a/a.md6skl"
		offset ( 0 0 0 )
		bounds ( -0.5 -0.5 -0.1 ) ( 0.5 0.5 1.9 )
		lod 0 {
			clear
			mesh "models/a/a_body.md6mesh" {
				cloth "models/a/a_cloth.hkt" "coatD_setup" "LOD0" "LOD0P" "LOD0C"
			}
			mesh "models/a/a_hair.md6mesh"
		}
		lod 1 {
			mesh "models/a/a_body_lod01.md6mesh"
		}
	}
	animsets {
		"models/a/a.animset"
	}
	props {
		prop "_info" {
			tag "t" { trans ( 0 0 0 ) rot ( 0 0 0 1 ) parent "origin" }
		}
	}
}"#;
        let d = Md6Def::parse(text).unwrap();
        assert_eq!(
            d.inherit.as_deref(),
            Some("models/characters/tall/base/base_body.md6")
        );
        assert_eq!(d.skeleton.as_deref(), Some("models/a/a.md6skl"));
        assert_eq!(d.lods.len(), 2);
        assert!(d.lods[0].clear && !d.lods[1].clear);
        assert_eq!(d.lods[0].meshes.len(), 2);
        assert_eq!(d.lods[0].meshes[0].cloth[0].args[1], "coatD_setup");
        assert_eq!(d.lods[1].meshes[0].mesh, "models/a/a_body_lod01.md6mesh");
    }

    #[test]
    fn resource_names() {
        assert_eq!(
            mesh_resource("models/A/b.md6mesh").as_deref(),
            Some("generated/basemodel/models/a/b.bmd6model")
        );
        assert_eq!(
            skeleton_resource("models/a/b.md6skl").as_deref(),
            Some("generated/skeleton/models/a/b.edgeskl")
        );
        assert_eq!(
            def_resource("models/a/b.md6"),
            "generated/decls/md6def/models/a/b.md6.decl"
        );
        let m = "models/x/t_grd_overseer02_body.md6mesh";
        assert!(is_nocloth_build(
            "nocloth_t_grd_overseer02_body_104a404f7912c0369bfdfddb8c5f067a.bmd6model",
            m
        ));
        assert!(!is_nocloth_build(
            "nocloth_t_grd_overseer02_body_lod01_697fb5780d74303a8f4c5be6b04230a1.bmd6model",
            m
        ));
    }
}
