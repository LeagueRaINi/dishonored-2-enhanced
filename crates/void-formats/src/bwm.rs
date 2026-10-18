//! `BWM1` static world model (`maps/**/<map>.bwm`).
//!
//! Layout recovered from `Dishonored2.exe` `sub_1402974C0` (versions above
//! 0x10013); see `docs/formats/bwm.md`. Coordinates are kept in the file's
//! own space (Z-up); conversion to an engine's convention belongs to adapters.

use crate::Error;
use crate::reader::Reader;

pub const MAGIC: [u8; 4] = *b"BWM1";
/// Oldest version handled by the recovered reader (`> 0x10013` in the exe).
pub const MIN_VERSION: u32 = 0x10014;
/// Newest version observed in shipped content.
pub const MAX_KNOWN_VERSION: u32 = 0x10015;
/// The exe rejects stream counts outside 1..=2.
const MAX_STREAMS: usize = 2;
pub const POSITION_STRIDE: u32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRef {
    pub byte_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    pub streams: Vec<StreamRef>,
    /// 0x12345678 in every observed surface; read and discarded by the exe.
    pub sentinel: u32,
    pub index_byte_offset: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    /// Exe field +68; 3 in every observed surface (H: primitive topology).
    pub unknown68: u32,
    pub flag: u8,
}

/// A run of instances drawn with one surface and material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub first_instance: u32,
    pub end_instance: u32,
    pub surface: u32,
    pub flag: u8,
    pub material: String,
    pub unknown: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Row-major 3x4 `[R | t]`; maps surface positions into world space.
    pub transform: [f32; 12],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub unknown_u32: u32,
    pub unknown_u16: u16,
}

/// Tables after the instances whose meaning is not yet recovered.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrailingTables {
    pub lists4: [Vec<u32>; 4],
    pub keyed_lists: Vec<(u32, Vec<u32>)>,
    /// Present only for version > 0x10014.
    pub lists: Vec<Vec<u32>>,
    pub final_list: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct WorldModel {
    pub version: u32,
    pub streams: Vec<Vec<u8>>,
    pub indices: Vec<u8>,
    pub buffer24: Vec<u8>,
    pub surfaces: Vec<Surface>,
    pub batches: Vec<Batch>,
    pub instances: Vec<Instance>,
    pub tables: TrailingTables,
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::Invalid { field, detail }
}

impl WorldModel {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let version = r.u32("version")?;
        if !(MIN_VERSION..=MAX_KNOWN_VERSION).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }

        let stream_count = r.u32("stream count")? as usize;
        if !(1..=MAX_STREAMS).contains(&stream_count) {
            return Err(invalid("stream count", stream_count.to_string()));
        }
        let streams = (0..stream_count)
            .map(|_| r.sized_blob("vertex stream").map(<[u8]>::to_vec))
            .collect::<Result<Vec<_>, _>>()?;
        let indices = r.sized_blob("index buffer")?.to_vec();

        let xforms = r.sized_blob("instance transforms")?;
        if xforms.len() % 48 != 0 {
            return Err(invalid(
                "instance transforms",
                format!("{} bytes", xforms.len()),
            ));
        }
        let buffer24 = r.sized_blob("buffer24")?.to_vec();
        if buffer24.len() % 24 != 0 {
            return Err(invalid("buffer24", format!("{} bytes", buffer24.len())));
        }

        let surface_count = r.count(4 + 5 * 4 + 1, "surface count")?;
        let mut surfaces = Vec::with_capacity(surface_count);
        for _ in 0..surface_count {
            let n = r.u32("surface stream count")? as usize;
            if n > MAX_STREAMS {
                return Err(invalid("surface stream count", n.to_string()));
            }
            let streams = (0..n)
                .map(|_| {
                    Ok(StreamRef {
                        byte_offset: r.u32("surface stream offset")?,
                        stride: r.u32("surface stream stride")?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            surfaces.push(Surface {
                streams,
                sentinel: r.u32("surface sentinel")?,
                index_byte_offset: r.u32("surface index offset")?,
                index_count: r.u32("surface index count")?,
                vertex_count: r.u32("surface vertex count")?,
                unknown68: r.u32("surface unknown68")?,
                flag: r.u8("surface flag")?,
            });
        }

        let batch_count = r.count(4 * 3 + 1 + 4 + 4, "batch count")?;
        let mut batches = Vec::with_capacity(batch_count);
        for _ in 0..batch_count {
            batches.push(Batch {
                first_instance: r.u32("batch first instance")?,
                end_instance: r.u32("batch end instance")?,
                surface: r.u32("batch surface")?,
                flag: r.u8("batch flag")?,
                material: r.string("batch material")?,
                unknown: r.u32("batch unknown")?,
            });
        }

        let instance_count = r.count(30, "instance count")?;
        if instance_count * 48 != xforms.len() {
            return Err(invalid(
                "instance count",
                format!(
                    "{instance_count} records but {} transform bytes",
                    xforms.len()
                ),
            ));
        }
        let mut instances = Vec::with_capacity(instance_count);
        for chunk in xforms.as_chunks::<48>().0 {
            let mut transform = [0f32; 12];
            for (dst, src) in transform.iter_mut().zip(chunk.as_chunks::<4>().0) {
                *dst = f32::from_le_bytes(*src);
            }
            let mut b = [0f32; 6];
            for v in &mut b {
                *v = r.f32("instance aabb")?;
            }
            instances.push(Instance {
                transform,
                aabb_min: [b[0], b[1], b[2]],
                aabb_max: [b[3], b[4], b[5]],
                unknown_u32: r.u32("instance unknown u32")?,
                unknown_u16: r.u16("instance unknown u16")?,
            });
        }

        let mut tables = TrailingTables::default();
        let mut counts = [0usize; 4];
        for c in &mut counts {
            *c = r.count(4, "list4 count")?;
        }
        for (list, &n) in tables.lists4.iter_mut().zip(&counts) {
            if n * 4 > r.remaining() {
                return Err(invalid("list4", format!("{n} entries exceed input")));
            }
            *list = (0..n)
                .map(|_| r.u32("list4 entry"))
                .collect::<Result<_, _>>()?;
        }
        let n = r.count(8, "keyed list count")?;
        for _ in 0..n {
            let key = r.u32("keyed list key")?;
            tables.keyed_lists.push((key, r.u32_vec("keyed list")?));
        }
        if version > 0x10014 {
            let n = r.count(4, "list count")?;
            for _ in 0..n {
                tables.lists.push(r.u32_vec("list")?);
            }
            tables.final_list = r.u32_vec("final list")?;
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }

        let model = Self {
            version,
            streams,
            indices,
            buffer24,
            surfaces,
            batches,
            instances,
            tables,
        };
        model.check_ranges()?;
        Ok(model)
    }

    /// Enforces the invariants observed in shipped data so accessors cannot
    /// read out of bounds. Any violation means the layout model is wrong.
    fn check_ranges(&self) -> Result<(), Error> {
        for (i, s) in self.surfaces.iter().enumerate() {
            if s.streams.len() != self.streams.len() {
                return Err(invalid(
                    "surface",
                    format!(
                        "{i}: {} stream refs, {} streams",
                        s.streams.len(),
                        self.streams.len()
                    ),
                ));
            }
            for (k, sr) in s.streams.iter().enumerate() {
                let end =
                    u64::from(sr.byte_offset) + u64::from(sr.stride) * u64::from(s.vertex_count);
                if sr.stride == 0 || end > self.streams[k].len() as u64 {
                    return Err(invalid(
                        "surface",
                        format!("{i}: stream {k} range {sr:?} out of bounds"),
                    ));
                }
            }
            if s.streams[0].stride != POSITION_STRIDE {
                return Err(invalid(
                    "surface",
                    format!("{i}: position stride {}", s.streams[0].stride),
                ));
            }
            let end = u64::from(s.index_byte_offset) + 2 * u64::from(s.index_count);
            if s.index_byte_offset % 2 != 0 || end > self.indices.len() as u64 {
                return Err(invalid(
                    "surface",
                    format!("{i}: index range out of bounds"),
                ));
            }
            if let Some(bad) = self
                .surface_indices(i)
                .find(|&ix| u32::from(ix) >= s.vertex_count)
            {
                return Err(invalid(
                    "surface",
                    format!("{i}: index {bad} >= vertex count {}", s.vertex_count),
                ));
            }
        }
        for (i, b) in self.batches.iter().enumerate() {
            if b.first_instance > b.end_instance
                || b.end_instance as usize > self.instances.len()
                || b.surface as usize >= self.surfaces.len()
            {
                return Err(invalid("batch", format!("{i}: {b:?} out of range")));
            }
        }
        Ok(())
    }

    /// Surface-local u16 indices.
    pub fn surface_indices(&self, surface: usize) -> impl Iterator<Item = u16> + '_ {
        let s = &self.surfaces[surface];
        let start = s.index_byte_offset as usize;
        self.indices[start..start + 2 * s.index_count as usize]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
    }

    /// Object-space positions (stream 0, three f32).
    pub fn surface_positions(&self, surface: usize) -> impl Iterator<Item = [f32; 3]> + '_ {
        let s = &self.surfaces[surface];
        let r = s.streams[0];
        let start = r.byte_offset as usize;
        self.streams[0][start..start + 12 * s.vertex_count as usize]
            .as_chunks::<12>()
            .0
            .iter()
            .map(|c| {
                let (xyz, _) = c.as_chunks::<4>();
                [xyz[0], xyz[1], xyz[2]].map(f32::from_le_bytes)
            })
    }

    /// Stream-1 attributes (20-byte layout, docs/formats/bwm.md). Returns
    /// `None` when the surface has no second stream or a different stride.
    /// Raw 20-byte stream-1 blocks of a surface (layout: `VertexAttributes`), for
    /// direct GPU upload (R16G16_FLOAT x2, R8G8B8A8_SNORM x2, R8G8B8A8_UNORM).
    pub fn surface_attribute_blocks(&self, surface: usize) -> Option<&[[u8; 20]]> {
        let s = &self.surfaces[surface];
        let r = *s.streams.get(1)?;
        if r.stride != 20 {
            return None;
        }
        let start = r.byte_offset as usize;
        let data = self.streams[1].get(start..start + 20 * s.vertex_count as usize)?;
        Some(data.as_chunks::<20>().0)
    }

    pub fn surface_attributes(&self, surface: usize) -> Option<Vec<VertexAttributes>> {
        let s = &self.surfaces[surface];
        let r = *s.streams.get(1)?;
        if r.stride != 20 {
            return None;
        }
        let start = r.byte_offset as usize;
        let data = &self.streams[1][start..start + 20 * s.vertex_count as usize];
        Some(
            data.as_chunks::<20>()
                .0
                .iter()
                .map(VertexAttributes::from_block)
                .collect(),
        )
    }
}

/// Decoded stream-1 attributes (docs/formats/bwm.md). Evidence: shipped
/// shader source (`vertex.decl`, `pixel.decl`) and the file-normal test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexAttributes {
    /// `m_texCoords.xy` (UV set 0).
    pub uv0: [f32; 2],
    /// `m_texCoords.zw` (UV set 1).
    pub uv1: [f32; 2],
    /// snorm8 object-space normal; `w` is the barycentric selector -1/0/+1
    /// (`ExtractBarycentricsFromNormalW`).
    pub normal: [f32; 4],
    /// snorm8 object-space tangent; `w` is the bitangent sign
    /// (`bitangent = cross(normal, tangent) * w`).
    pub tangent: [f32; 4],
    /// Bytes 16..19 as stored (unorm8). Mapping to shader channels is
    /// determined by validation, not assumed here.
    pub color: [u8; 4],
}

impl VertexAttributes {
    /// Decodes one 20-byte attribute block (shared by BWM stream 1 and bmodel CPU decl 1).
    pub fn from_block(c: &[u8; 20]) -> Self {
        let h = |o: usize| f16_to_f32(u16::from_le_bytes([c[o], c[o + 1]]));
        let sn = |o: usize| [0, 1, 2, 3].map(|k| (f32::from(c[o + k] as i8) / 127.0).max(-1.0));
        Self {
            uv0: [h(0), h(2)],
            uv1: [h(4), h(6)],
            normal: sn(8),
            tangent: sn(12),
            color: [c[16], c[17], c[18], c[19]],
        }
    }
}

/// IEEE 754 binary16 to binary32, including subnormals, infinities and NaN.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize the mantissa.
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn blob(out: &mut Vec<u8>, b: &[u8]) {
        put(out, b.len() as u32);
        out.extend_from_slice(b);
    }

    /// One triangle surface drawn by two instances.
    fn sample() -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        put(&mut out, 0x10015);
        put(&mut out, 2);
        let pos: Vec<u8> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        blob(&mut out, &pos);
        blob(&mut out, &[0u8; 60]);
        blob(&mut out, &[0, 0, 1, 0, 2, 0]);
        let mut xf = Vec::new();
        for t in [0f32, 10.0] {
            for v in [1., 0., 0., t, 0., 1., 0., 0., 0., 0., 1., 0.] {
                xf.extend_from_slice(&f32::to_le_bytes(v));
            }
        }
        blob(&mut out, &xf);
        blob(&mut out, &[]);
        put(&mut out, 1); // surfaces
        put(&mut out, 2);
        for (o, s) in [(0, 12), (0, 20)] {
            put(&mut out, o);
            put(&mut out, s);
        }
        for v in [0x1234_5678, 0, 3, 3, 3] {
            put(&mut out, v);
        }
        out.push(1);
        put(&mut out, 1); // batches
        for v in [0, 2, 0] {
            put(&mut out, v);
        }
        out.push(0);
        blob(&mut out, b"models/test.material");
        put(&mut out, 0);
        put(&mut out, 2); // instances
        for _ in 0..2 {
            for v in [0f32, 0., 0., 1., 1., 0.] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            put(&mut out, 7);
            out.extend_from_slice(&0xffffu16.to_le_bytes());
        }
        for _ in 0..4 {
            put(&mut out, 0);
        }
        put(&mut out, 0); // keyed lists
        put(&mut out, 1); // lists
        put(&mut out, 1);
        put(&mut out, 1);
        put(&mut out, 0); // final list
        out
    }

    #[test]
    fn parses_synthetic_model() {
        let m = WorldModel::parse(&sample()).unwrap();
        assert_eq!(m.surfaces.len(), 1);
        assert_eq!(m.instances[1].transform[3], 10.0);
        assert_eq!(m.batches[0].material, "models/test.material");
        assert_eq!(m.surface_indices(0).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(m.surface_positions(0).nth(1), Some([1., 0., 0.]));
        assert_eq!(m.tables.lists, [vec![1]]);
        assert_eq!(m.surface_attributes(0).unwrap().len(), 3);
    }

    #[test]
    fn rejects_every_truncation_and_trailing_byte() {
        let full = sample();
        for cut in 0..full.len() {
            assert!(WorldModel::parse(&full[..cut]).is_err(), "cut {cut} parsed");
        }
        let mut extra = full;
        extra.push(0);
        assert!(matches!(
            WorldModel::parse(&extra),
            Err(Error::TrailingData { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_index_and_batch() {
        let mut data = sample();
        // Index buffer begins after header(12) + pos blob(4+36) + stream1 blob(4+60) + len(4).
        let idx = 12 + 40 + 64 + 4;
        data[idx + 4] = 9;
        assert!(matches!(
            WorldModel::parse(&data),
            Err(Error::Invalid { .. })
        ));

        let mut data = sample();
        let pos = data
            .windows(4)
            .rposition(|w| w == 2u32.to_le_bytes())
            .unwrap();
        // Last `2` before the material string is the batch end instance.
        let mat = data.windows(4).position(|w| w == b"mode").unwrap();
        let end = data[..mat]
            .windows(4)
            .rposition(|w| w == 2u32.to_le_bytes())
            .unwrap();
        assert!(end < pos);
        data[end] = 3;
        assert!(matches!(
            WorldModel::parse(&data),
            Err(Error::Invalid { .. })
        ));
    }

    #[test]
    fn rejects_bad_header_values() {
        let mut data = sample();
        data[0] = b'X';
        assert!(matches!(
            WorldModel::parse(&data),
            Err(Error::BadMagic { .. })
        ));
        let mut data = sample();
        data[4..8].copy_from_slice(&0x10013u32.to_le_bytes());
        assert!(matches!(
            WorldModel::parse(&data),
            Err(Error::UnsupportedVersion(_))
        ));
        let mut data = sample();
        data[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            WorldModel::parse(&data),
            Err(Error::Invalid { .. })
        ));
    }

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }
}
