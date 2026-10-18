//! Engine-independent collision geometry flattened from Havok `hknp` shape
//! trees (docs/formats/havok.md).
//!
//! Supported (O: tutorial_p physworld and door physhape):
//! - `hknpConvexPolytopeShape` / `hknpConvexShape`: vertices (`hknpConvexShape::vertices`),
//!   polytope faces (`faces` -> `indices`), plane equations, `convexRadius`;
//! - `hknpCompoundShape`: `instances` (`hkFreeListArray` of `hknpShapeInstance`
//!   `{transform, scale, shape}`), recursively;
//! - `hknpCompressedMeshShape` with `numConvexShapes == 0`: the quantized
//!   `hkcdStaticMeshTree` triangles (see [`decode_compressed_mesh`]).
//!
//! Anything else is reported per type in [`Collision::unsupported`], never
//! silently dropped.
//!
//! Transform convention (D): `hkTransform` = rotation as three column vectors
//! (12 floats incl. w) then translation (4 floats); world = R * (scale * v) + t.

use std::collections::{BTreeMap, HashMap};

use crate::Error;
use crate::havok::{TagFile, Value};

/// Row-major 3x4 affine transform.
pub type Affine = [[f32; 4]; 3];

pub const IDENTITY: Affine = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

pub fn apply(m: &Affine, p: [f32; 3]) -> [f32; 3] {
    let r = |i: usize| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3];
    [r(0), r(1), r(2)]
}

pub fn compose(a: &Affine, b: &Affine) -> Affine {
    let mut out = [[0.0; 4]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v =
                (0..3).map(|k| a[i][k] * b[k][j]).sum::<f32>() + if j == 3 { a[i][3] } else { 0.0 };
        }
    }
    out
}

/// Collision geometry in world space: a convex hull, or a triangle mesh
/// (`is_mesh`, from a compressed mesh; not convex, faces are triangles in
/// stored winding).
#[derive(Clone, Debug, PartialEq)]
pub struct Convex {
    pub vertices: Vec<[f32; 3]>,
    /// Polygon faces as vertex index loops (empty for plain `hknpConvexShape`).
    pub faces: Vec<Vec<u32>>,
    /// Havok convex radius (surface lies this far outside the hull), scaled.
    pub radius: f32,
    pub is_mesh: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// Index into `hknpPhysicsSystemData.bodyCinfos` (0 for standalone shapes).
    pub body: usize,
    /// Item index of the leaf shape (stable identity within the file).
    pub shape_item: u32,
    /// Compound instance indices from the body's root shape down to the leaf.
    pub path: Vec<u32>,
    pub convex: Convex,
}

#[derive(Clone, Debug, Default)]
pub struct Collision {
    pub shapes: Vec<Placed>,
    /// Unsupported leaf shapes by type name: (count, example item).
    pub unsupported: BTreeMap<String, (usize, u32)>,
    /// Every unsupported leaf: (type name, item, body, world transform).
    pub unsupported_leaves: Vec<(String, u32, usize, Affine)>,
    /// Non-uniform or negative instance scales seen (count).
    pub nonuniform_scales: usize,
    /// Decoded compressed-mesh data by item (shared between instances).
    mesh_cache: HashMap<u32, MeshData>,
}

/// Object-space triangles of an `hknpCompressedMeshShapeData`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// Largest distance of a decoded vertex outside its section's domain.
    pub max_domain_excess: f32,
    /// Sections skipped because their `flags` select an encoding not decoded
    /// (O: flags 1 has no valid codec and 2-index primitives); (flags, primitives).
    pub skipped_sections: Vec<(u64, usize)>,
}

fn field_u64(v: &Value, name: &'static str) -> Result<u64, Error> {
    v.field(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| bad(name, format!("missing in {v:?}")))
}

fn aabb(v: Option<&Value>, what: &'static str) -> Result<([f32; 3], [f32; 3]), Error> {
    let v = v.ok_or(bad(what, "missing"))?;
    let lo = floats(v.field("min"), what)?;
    let hi = floats(v.field("max"), what)?;
    Ok(([lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]))
}

/// Packed section vertex: 11/11/10-bit x/y/z with `codec` = offset[3], scale[3].
pub fn unpack_packed(v: u64, codec: &[f32]) -> [f32; 3] {
    let q = [v & 0x7FF, (v >> 11) & 0x7FF, (v >> 22) & 0x3FF];
    std::array::from_fn(|k| codec[k] + q[k] as f32 * codec[3 + k])
}

/// Shared vertex: 21/21/22-bit x/y/z quantized over the tree domain.
pub fn unpack_shared(v: u64, lo: [f32; 3], hi: [f32; 3]) -> [f32; 3] {
    let q = [v & 0x1F_FFFF, (v >> 21) & 0x1F_FFFF, v >> 42];
    let bits = [21u32, 21, 22];
    std::array::from_fn(|k| lo[k] + q[k] as f32 * (hi[k] - lo[k]) / ((1u64 << bits[k]) - 1) as f32)
}

/// Decodes the `meshTree` of an `hknpCompressedMeshShapeData` item.
///
/// Encoding (O: tutorial_p physworld; D: consistent with every section domain):
/// - packed vertex (u32): x = bits 0-10, y = 11-21, z = 22-31; position =
///   `codecParms[0..3] + q * codecParms[3..6]` of its section;
/// - shared vertex (u64): x = bits 0-20, y = 21-41, z = 42-63, quantized over the
///   tree `domain`: `min + q * (max - min) / (2^bits - 1)`;
/// - section `primitives.data` / `dataRuns.data` = `offset << 8 | count`;
///   `sharedVertices.data >> 8` = offset into `sharedVerticesIndex`, with
///   `numSharedIndices` entries;
/// - primitive indices `< numPackedVertices` select the section's packed vertices
///   (from `firstPackedVertex`), others the shared list; a primitive is a quad
///   (0,1,2)+(0,2,3), or a triangle when indices 2 and 3 are equal.
pub fn decode_compressed_mesh(tag: &TagFile, data_item: u32) -> Result<MeshData, Error> {
    let data = tag.element(data_item, 0)?;
    let tree = data.field("meshTree").ok_or(bad("meshTree", "missing"))?;
    let (dlo, dhi) = aabb(tree.field("domain"), "tree domain")?;
    let list = |name: &'static str| -> Result<Vec<Value>, Error> {
        tag.deref(tree.field(name).ok_or(bad(name, "missing"))?)
    };
    let packed: Vec<u64> = list("packedVertices")?
        .iter()
        .map(|v| v.as_u64().ok_or(bad("packed vertex", format!("{v:?}"))))
        .collect::<Result<_, _>>()?;
    let shared_raw: Vec<u64> = list("sharedVertices")?
        .iter()
        .map(|v| v.as_u64().ok_or(bad("shared vertex", format!("{v:?}"))))
        .collect::<Result<_, _>>()?;
    let shared_index: Vec<u64> = list("sharedVerticesIndex")?
        .iter()
        .map(|v| v.as_u64().ok_or(bad("shared index", format!("{v:?}"))))
        .collect::<Result<_, _>>()?;
    let prims = list("primitives")?;
    let sections = list("sections")?;

    let mut vertices = Vec::with_capacity(packed.len() + shared_raw.len());
    let shared_base = 0usize;
    for &v in &shared_raw {
        vertices.push(unpack_shared(v, dlo, dhi));
    }
    let mut triangles = Vec::new();
    let mut max_excess = 0f32;
    let mut skipped_sections = Vec::new();
    for sec in &sections {
        let flags = field_u64(sec, "flags")?;
        if flags != 0 {
            let w = sec
                .field("primitives")
                .and_then(|r| r.field("data"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            skipped_sections.push((flags, (w & 0xFF) as usize));
            continue;
        }
        let codec = floats(sec.field("codecParms"), "codecParms")?;
        if codec.len() != 6 {
            return Err(bad("codecParms", format!("{} values", codec.len())));
        }
        let (slo, shi) = aabb(sec.field("domain"), "section domain")?;
        let first = field_u64(sec, "firstPackedVertex")? as usize;
        let npacked = field_u64(sec, "numPackedVertices")? as usize;
        let nshared = field_u64(sec, "numSharedIndices")? as usize;
        let word = |name: &'static str| -> Result<u64, Error> {
            sec.field(name)
                .and_then(|r| r.field("data"))
                .and_then(Value::as_u64)
                .ok_or(bad(name, "missing data word"))
        };
        let shared_off = (word("sharedVertices")? >> 8) as usize;
        let prim_word = word("primitives")?;
        let (prim_off, prim_n) = ((prim_word >> 8) as usize, (prim_word & 0xFF) as usize);
        let packed_sec = packed.get(first..first + npacked).ok_or(bad(
            "packed range",
            format!("{first}+{npacked} > {}", packed.len()),
        ))?;
        let shared_sec = shared_index
            .get(shared_off..shared_off + nshared)
            .ok_or(bad("shared range", format!("{shared_off}+{nshared}")))?;
        let base = vertices.len() as u32;
        for &v in packed_sec {
            let p = unpack_packed(v, &codec);
            for k in 0..3 {
                max_excess = max_excess.max(slo[k] - p[k]).max(p[k] - shi[k]);
            }
            vertices.push(p);
        }
        let local = |i: u64| -> Result<u32, Error> {
            let i = i as usize;
            if i < npacked {
                Ok(base + i as u32)
            } else {
                let s = *shared_sec.get(i - npacked).ok_or(bad(
                    "primitive index",
                    format!("{i} beyond {npacked}+{nshared}"),
                ))? as usize;
                if s >= shared_raw.len() {
                    return Err(bad(
                        "shared vertex index",
                        format!(
                            "{s} >= {} shared vertices (data item {data_item}, {} packed, {} shared indices)",
                            shared_raw.len(),
                            packed.len(),
                            shared_index.len()
                        ),
                    ));
                }
                Ok((shared_base + s) as u32)
            }
        };
        for p in prims.get(prim_off..prim_off + prim_n).ok_or(bad(
            "primitive range",
            format!("{prim_off}+{prim_n} > {}", prims.len()),
        ))? {
            let idx: Vec<u64> = match p.field("indices") {
                Some(Value::Fixed(v)) => v.iter().filter_map(Value::as_u64).collect(),
                other => return Err(bad("primitive", format!("{other:?}"))),
            };
            let [a, b, c, d] = idx[..] else {
                return Err(bad("primitive", format!("{} indices", idx.len())));
            };
            let (a, b, c2) = (local(a)?, local(b)?, local(c)?);
            triangles.push([a, b, c2]);
            if c != d {
                triangles.push([a, c2, local(d)?]);
            }
        }
    }
    Ok(MeshData {
        vertices,
        triangles,
        max_domain_excess: max_excess,
        skipped_sections,
    })
}

fn floats(v: Option<&Value>, what: &'static str) -> Result<Vec<f32>, Error> {
    v.and_then(Value::floats).ok_or_else(|| Error::Invalid {
        field: what,
        detail: format!("{v:?}"),
    })
}

fn bad(field: &'static str, detail: impl Into<String>) -> Error {
    Error::Invalid {
        field,
        detail: detail.into(),
    }
}

/// `hkTransform` (16 floats: 3 rotation columns + translation) as row-major 3x4.
pub fn affine_from_hk(f: &[f32]) -> Result<Affine, Error> {
    if f.len() != 16 {
        return Err(bad("hkTransform", format!("{} floats", f.len())));
    }
    let mut m = IDENTITY;
    for (c, col) in [&f[0..4], &f[4..8], &f[8..12]].iter().enumerate() {
        for r in 0..3 {
            m[r][c] = col[r];
        }
    }
    for r in 0..3 {
        m[r][3] = f[12 + r];
    }
    Ok(m)
}

impl Collision {
    /// Flattens every body of an `hknpPhysicsSystemData`.
    pub fn from_physics_system(tag: &TagFile) -> Result<Self, Error> {
        let [sys] = tag.items_of("hknpPhysicsSystemData").collect::<Vec<_>>()[..] else {
            return Err(bad("hknpPhysicsSystemData", "expected exactly one"));
        };
        let sys = tag.element(sys, 0)?;
        let bodies = tag.deref(
            sys.field("bodyCinfos")
                .ok_or(bad("bodyCinfos", "missing"))?,
        )?;
        let mut out = Self::default();
        for (b, body) in bodies.iter().enumerate() {
            let shape = body
                .field("shape")
                .and_then(Value::as_ref)
                .ok_or(bad("body shape", format!("body {b}")))?;
            let pos = floats(body.field("position"), "body position")?;
            let q = floats(body.field("orientation"), "body orientation")?;
            let mut m = quat_matrix([q[0], q[1], q[2], q[3]]);
            for r in 0..3 {
                m[r][3] = pos[r];
            }
            out.walk(tag, shape, &m, 1.0, b, &mut Vec::new(), 0)?;
        }
        Ok(out)
    }

    /// Flattens a standalone shape file (e.g. a `physhape`), placed by `m`.
    /// Skipped compressed-mesh sections: (data item, flags, primitives).
    pub fn skipped_mesh_sections(&self) -> Vec<(u32, u64, usize)> {
        let mut v: Vec<_> = self
            .mesh_cache
            .iter()
            .flat_map(|(&i, m)| m.skipped_sections.iter().map(move |&(f, n)| (i, f, n)))
            .collect();
        v.sort_unstable();
        v
    }

    /// Largest section-domain excess over all decoded compressed meshes.
    pub fn max_mesh_domain_excess(&self) -> f32 {
        self.mesh_cache
            .values()
            .map(|m| m.max_domain_excess)
            .fold(0.0, f32::max)
    }

    pub fn from_shape(tag: &TagFile, root: u32, m: &Affine) -> Result<Self, Error> {
        let mut out = Self::default();
        out.walk(tag, root, m, 1.0, 0, &mut Vec::new(), 0)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        tag: &TagFile,
        item: u32,
        m: &Affine,
        radius_scale: f32,
        body: usize,
        path: &mut Vec<u32>,
        depth: u32,
    ) -> Result<(), Error> {
        if depth > 16 {
            return Err(bad("shape tree", "nesting deeper than 16"));
        }
        if item == 0 {
            return Ok(());
        }
        let ty = tag.type_name(tag.item(item)?.ty).to_owned();
        let shape = tag.element(item, 0)?;
        match ty.as_str() {
            "hknpCompoundShape" => {
                let elems = shape
                    .path(&["instances", "elements"])
                    .ok_or(bad("compound instances", format!("item {item}")))?;
                for (k, inst) in tag.deref(elems)?.iter().enumerate() {
                    let child = inst.field("shape").and_then(Value::as_ref).unwrap_or(0);
                    if child == 0 {
                        continue; // free-list slot
                    }
                    let t =
                        affine_from_hk(&floats(inst.field("transform"), "instance transform")?)?;
                    let s = floats(inst.field("scale"), "instance scale")?;
                    let uniform =
                        (s[0] - s[1]).abs() < 1e-5 && (s[1] - s[2]).abs() < 1e-5 && s[0] > 0.0;
                    if !uniform {
                        self.nonuniform_scales += 1;
                    }
                    // H: scale applies in the child's local frame before the transform.
                    let mut ts = t;
                    for row in ts.iter_mut() {
                        for c in 0..3 {
                            row[c] *= s[c];
                        }
                    }
                    let child_m = compose(m, &ts);
                    let rs = radius_scale * s[0].abs().min(s[1].abs()).min(s[2].abs());
                    path.push(k as u32);
                    self.walk(tag, child, &child_m, rs, body, path, depth + 1)?;
                    path.pop();
                }
            }
            "hknpConvexPolytopeShape" | "hknpConvexShape" => {
                let verts =
                    tag.deref(shape.field("vertices").ok_or(bad("vertices", "missing"))?)?;
                let vertices = verts
                    .iter()
                    .map(|v| {
                        let f = v.floats().ok_or(bad("vertex", format!("{v:?}")))?;
                        Ok(apply(m, [f[0], f[1], f[2]]))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let mut faces = Vec::new();
                if ty == "hknpConvexPolytopeShape" {
                    let idx: Vec<u32> = tag
                        .deref(shape.field("indices").ok_or(bad("indices", "missing"))?)?
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(u64::MAX) as u32)
                        .collect();
                    for f in tag.deref(shape.field("faces").ok_or(bad("faces", "missing"))?)? {
                        let first =
                            f.field("firstIndex").and_then(Value::as_u64).unwrap_or(0) as usize;
                        let n = f.field("numIndices").and_then(Value::as_u64).unwrap_or(0) as usize;
                        let loop_ = idx
                            .get(first..first + n)
                            .ok_or(bad("face", format!("{first}+{n} > {}", idx.len())))?;
                        if loop_.iter().any(|&i| i as usize >= vertices.len()) {
                            return Err(bad("face index", format!("item {item}")));
                        }
                        faces.push(loop_.to_vec());
                    }
                }
                let radius = shape
                    .field("convexRadius")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                self.shapes.push(Placed {
                    body,
                    shape_item: item,
                    path: path.clone(),
                    convex: Convex {
                        vertices,
                        faces,
                        radius: radius * radius_scale,
                        is_mesh: false,
                    },
                });
            }
            "hknpCompressedMeshShape"
                if shape
                    .field("numConvexShapes")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    == 0 =>
            {
                let data = shape
                    .field("data")
                    .and_then(Value::as_ref)
                    .ok_or(bad("compressed mesh data", format!("item {item}")))?;
                let mesh = match self.mesh_cache.entry(data) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(decode_compressed_mesh(tag, data)?)
                    }
                };
                self.shapes.push(Placed {
                    body,
                    shape_item: item,
                    path: path.clone(),
                    convex: Convex {
                        vertices: mesh.vertices.iter().map(|&v| apply(m, v)).collect(),
                        faces: mesh.triangles.iter().map(|t| t.to_vec()).collect(),
                        radius: 0.0,
                        is_mesh: true,
                    },
                });
            }
            other => {
                let e = self
                    .unsupported
                    .entry(other.to_owned())
                    .or_insert((0, item));
                e.0 += 1;
                self.unsupported_leaves
                    .push((other.to_owned(), item, body, *m));
            }
        }
        Ok(())
    }
}

/// Rotation matrix of a unit quaternion (x, y, z, w).
pub fn quat_matrix(q: [f32; 4]) -> Affine {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            0.0,
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            0.0,
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_vertices_cover_their_domains() {
        let codec = [-3.4, -0.3, 0.0, 3.4 / 2047.0, 0.6 / 2047.0, 4.0 / 1023.0];
        assert_eq!(unpack_packed(0, &codec), [-3.4, -0.3, 0.0]);
        let top = unpack_packed(0x7FF | (0x7FF << 11) | (0x3FF << 22), &codec);
        assert!(
            (top[0] - 0.0).abs() < 1e-5
                && (top[1] - 0.3).abs() < 1e-5
                && (top[2] - 4.0).abs() < 1e-5
        );
        // O: a tutorial_p shared vertex on a section seam (z = 4.266 of 0..6.84).
        let v = unpack_shared(
            11_503_346_149_536_739_942,
            [-3.4, -0.295_312_5, 0.0],
            [0.0, 0.295_312_5, 6.841_413_5],
        );
        assert!((v[2] - 4.2663).abs() < 1e-3, "{v:?}");
        let all = (0x1F_FFFFu64) | (0x1F_FFFF << 21) | (0x3F_FFFF << 42);
        assert_eq!(
            unpack_shared(all, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            [4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn hk_transform_is_column_major() {
        // 90 degrees about Z: columns (0,1,0), (-1,0,0), (0,0,1); translation (5,6,7).
        let f = [
            0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 6.0, 7.0, 1.0,
        ];
        let m = affine_from_hk(&f).unwrap();
        assert_eq!(apply(&m, [1.0, 0.0, 0.0]), [5.0, 7.0, 7.0]);
        assert!(affine_from_hk(&f[..12]).is_err());
    }

    #[test]
    fn compose_and_quaternion_agree() {
        let r = quat_matrix([
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ]);
        let mut t = IDENTITY;
        t[0][3] = 2.0;
        let p = apply(&compose(&r, &t), [1.0, 0.0, 0.0]);
        assert!(
            (p[0] - 0.0).abs() < 1e-6 && (p[1] - 3.0).abs() < 1e-6,
            "{p:?}"
        );
    }
}
