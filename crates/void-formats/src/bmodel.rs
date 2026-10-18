//! Static entity models (`generated/model/**.bmodel`, magic `1LMB`).
//!
//! Layout (docs/formats/bmodel.md) from the exe's writer
//! `idStaticModel::WriteStaticBModel` (sub_1402722B0) and the `BTRI` surface
//! reader (sub_140259D50 -> sub_14025A040, versions > 0x10001), all
//! little-endian:
//!
//! ```text
//! "1LMB" | u32 stamp | str source | u32 surfaces
//! surface: str material | i32 | "BTRI" | u32 version | i32 cpu decl | i32 gpu decl
//!          [40] header | u32 vertices | u64 gpu bytes
//!          gpu bytes > 0: u8 streams, per stream { u64 offset, u64 stride, u64 size, [size] }
//!          gpu bytes = 0: vertices * stride(cpu decl) raw bytes
//!          u32 indices | [2 * indices] u16 | f32[6] bounds | u32 | "BML1"
//! u32 n | n * { str, u32, u32, u32 }
//! ```
//! `str` = i32 length + bytes (length <= 0 means empty).

use crate::Error;
use crate::bwm::VertexAttributes;
use crate::reader::Reader;

pub const MAGIC: [u8; 4] = *b"1LMB";

/// Stride of CPU vertex declaration 1: `f32x3` position followed by the
/// 20-byte attribute block of BWM stream 1 (inferred from file structure,
/// E022; the declaration table itself is runtime data not yet recovered).
pub const CPU_DECL1_STRIDE: usize = 32;

#[derive(Clone, Debug)]
pub struct Stream {
    pub offset: u64,
    pub stride: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Surface {
    pub material: String,
    pub unknown: i32,
    pub version: u32,
    pub cpu_decl: i32,
    pub gpu_decl: i32,
    pub header: [u8; 40],
    pub vertex_count: u32,
    pub streams: Vec<Stream>,
    pub indices: Vec<u16>,
    pub bounds: [f32; 6],
    pub tail: u32,
}

#[derive(Clone, Debug)]
pub struct StaticModel {
    pub stamp: u32,
    pub source: String,
    pub surfaces: Vec<Surface>,
    pub extra: Vec<(String, u32, u32, u32)>,
}

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

impl StaticModel {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("bmodel magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let stamp = r.u32("bmodel stamp")?;
        let source = string(&mut r, "bmodel source")?;
        let count = r.count(
            4 + 4 + 4 + 4 + 4 + 4 + 40 + 4 + 8 + 4 + 24 + 4 + 4,
            "surface count",
        )?;
        let mut surfaces = Vec::with_capacity(count);
        for _ in 0..count {
            surfaces.push(Self::surface(&mut r)?);
        }
        let n = r.count(16, "extra count")?;
        let mut extra = Vec::with_capacity(n);
        for _ in 0..n {
            extra.push((
                string(&mut r, "extra name")?,
                r.u32("extra a")?,
                r.u32("extra b")?,
                r.u32("extra c")?,
            ));
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        Ok(Self {
            stamp,
            source,
            surfaces,
            extra,
        })
    }

    fn surface(r: &mut Reader) -> Result<Surface, Error> {
        let material = string(r, "surface material")?;
        let unknown = r.u32("surface field")? as i32;
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
        let header: [u8; 40] = r.array("BTRI header")?;
        let vertex_count = r.u32("vertex count")?;
        let gpu_bytes =
            u64::from(r.u32("gpu bytes lo")?) | (u64::from(r.u32("gpu bytes hi")?) << 32);
        let mut streams = Vec::new();
        if gpu_bytes > 0 {
            let n = r.u8("stream count")?;
            if n > 15 {
                return Err(invalid("stream count", n.to_string()));
            }
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
                streams.push(Stream {
                    offset,
                    stride,
                    data: r.bytes(size as usize, "stream data")?.to_vec(),
                });
            }
        } else {
            if cpu_decl != 1 {
                return Err(invalid(
                    "cpu vertex declaration",
                    format!("{cpu_decl} (only 1 is known)"),
                ));
            }
            let size = vertex_count as usize * CPU_DECL1_STRIDE;
            streams.push(Stream {
                offset: 0,
                stride: CPU_DECL1_STRIDE as u64,
                data: r.bytes(size, "cpu vertices")?.to_vec(),
            });
        }
        let ni = r.count(2, "index count")?;
        let indices: Vec<u16> = r
            .bytes(2 * ni, "indices")?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let mut bounds = [0f32; 6];
        for b in &mut bounds {
            *b = r.f32("bounds")?;
        }
        let tail = r.u32("surface tail")?;
        let end: [u8; 4] = r.array("BML1 terminator")?;
        if &end != b"BML1" {
            return Err(Error::BadMagic {
                expected: *b"BML1",
                found: end,
            });
        }
        if let Some(&bad) = indices.iter().find(|&&i| u32::from(i) >= vertex_count) {
            return Err(invalid(
                "index",
                format!("{bad} >= vertex count {vertex_count}"),
            ));
        }
        Ok(Surface {
            material,
            unknown,
            version,
            cpu_decl,
            gpu_decl,
            header,
            vertex_count,
            streams,
            indices,
            bounds,
            tail,
        })
    }
}

impl Surface {
    /// Interleaved CPU declaration 1 (one 32-byte stream) is the only layout
    /// decoded so far; others return `None` so callers can report them.
    fn interleaved(&self) -> Option<&[u8]> {
        match self.streams.as_slice() {
            [s] if s.stride == CPU_DECL1_STRIDE as u64 && s.offset == 0 => Some(&s.data),
            _ => None,
        }
    }

    pub fn positions(&self) -> Option<Vec<[f32; 3]>> {
        let d = self.interleaved()?;
        Some(
            d.as_chunks::<32>()
                .0
                .iter()
                .map(|v| {
                    let (xyz, _) = v.as_chunks::<4>();
                    [xyz[0], xyz[1], xyz[2]].map(f32::from_le_bytes)
                })
                .collect(),
        )
    }

    /// Raw 20-byte attribute blocks (bytes 12..32 of each CPU decl 1 vertex).
    pub fn attribute_blocks(&self) -> Option<Vec<[u8; 20]>> {
        let d = self.interleaved()?;
        Some(
            d.as_chunks::<32>()
                .0
                .iter()
                .map(|v| v[12..32].try_into().expect("20 bytes"))
                .collect(),
        )
    }

    pub fn attributes(&self) -> Option<Vec<VertexAttributes>> {
        let d = self.interleaved()?;
        Some(
            d.as_chunks::<32>()
                .0
                .iter()
                .map(|v| VertexAttributes::from_block(v[12..32].try_into().expect("20 bytes")))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(o: &mut Vec<u8>, v: u32) {
        o.extend_from_slice(&v.to_le_bytes());
    }
    fn s(o: &mut Vec<u8>, t: &str) {
        put(o, t.len() as u32);
        o.extend_from_slice(t.as_bytes());
    }

    fn sample() -> Vec<u8> {
        let mut o = MAGIC.to_vec();
        put(&mut o, 0x54bf_a606);
        s(&mut o, "V:/src/door.mb");
        put(&mut o, 1);
        s(&mut o, "models/door.material");
        put(&mut o, u32::MAX);
        o.extend_from_slice(b"BTRI");
        put(&mut o, 0x10002);
        put(&mut o, 1);
        put(&mut o, 1);
        o.extend_from_slice(&[0; 40]);
        put(&mut o, 3);
        put(&mut o, 0);
        put(&mut o, 0);
        for k in 0..3u32 {
            for f in [k as f32, 0.0, 1.0] {
                o.extend_from_slice(&f.to_le_bytes());
            }
            o.extend_from_slice(&[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 0x81, 127, 0, 0, 127, 1, 2, 3, 4,
            ]);
        }
        put(&mut o, 3);
        for i in [0u16, 1, 2] {
            o.extend_from_slice(&i.to_le_bytes());
        }
        o.extend_from_slice(&[0; 24]);
        put(&mut o, 0);
        o.extend_from_slice(b"BML1");
        put(&mut o, 1);
        s(&mut o, "tag");
        put(&mut o, 1);
        put(&mut o, 2);
        put(&mut o, 3);
        o
    }

    #[test]
    fn parses_cpu_decl1_model() {
        let m = StaticModel::parse(&sample()).unwrap();
        assert_eq!(m.source, "V:/src/door.mb");
        let sfc = &m.surfaces[0];
        assert_eq!(sfc.material, "models/door.material");
        assert_eq!(sfc.positions().unwrap()[2], [2.0, 0.0, 1.0]);
        let a = sfc.attributes().unwrap();
        assert_eq!(a[0].normal, [0.0, 0.0, 1.0, -1.0]);
        assert_eq!(a[0].color, [1, 2, 3, 4]);
        assert_eq!(m.extra, [("tag".to_owned(), 1, 2, 3)]);
    }

    #[test]
    fn rejects_truncation_trailing_and_bad_indices() {
        let full = sample();
        for cut in 0..full.len() {
            assert!(StaticModel::parse(&full[..cut]).is_err(), "cut {cut}");
        }
        let mut t = full.clone();
        t.push(0);
        assert!(matches!(
            StaticModel::parse(&t),
            Err(Error::TrailingData { .. })
        ));
        let mut b = full;
        let pos = b.windows(4).position(|w| w == b"BML1").unwrap() - 4 - 24 - 2;
        b[pos] = 9;
        assert!(StaticModel::parse(&b).is_err());
    }
}
