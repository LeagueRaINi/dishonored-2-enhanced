//! `.mapresources`: the list of logical resource ids a map depends on.
//!
//! Layout (docs/formats/archives.md F010), big-endian:
//!
//! ```text
//! u64 zero
//! u32 count
//! record[count] { u32 id, [u8; 20] zero, u32 tag }   // 28 bytes; tag observed 0x80
//! ```
//!
//! Ids refer to entries in the same group's index (F009). Unexpected non-zero
//! padding is kept so callers can report it instead of losing it.

use crate::Error;
use crate::reader::Reader;

const RECORD_LEN: usize = 28;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapResource {
    pub id: u32,
    pub padding: [u8; 20],
    pub tag: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapResources {
    pub header: u64,
    pub records: Vec<MapResource>,
}

impl MapResources {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let header = r.u64_be("mapresources header")?;
        let count = r.u32_be("mapresources count")? as usize;
        let expected = count as u64 * RECORD_LEN as u64;
        if expected != r.remaining() as u64 {
            return Err(Error::SizeMismatch {
                field: "mapresources records",
                declared: expected,
                actual: r.remaining() as u64,
            });
        }
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            records.push(MapResource {
                id: r.u32_be("record id")?,
                padding: r.array("record padding")?,
                tag: r.u32_be("record tag")?,
            });
        }
        Ok(Self { header, records })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(ids: &[u32]) -> Vec<u8> {
        let mut out = vec![0; 8];
        out.extend_from_slice(&(ids.len() as u32).to_be_bytes());
        for id in ids {
            out.extend_from_slice(&id.to_be_bytes());
            out.extend_from_slice(&[0; 20]);
            out.extend_from_slice(&0x80u32.to_be_bytes());
        }
        out
    }

    #[test]
    fn parses_records() {
        let m = MapResources::parse(&encode(&[0x5a92, 0x5a93])).unwrap();
        assert_eq!(m.records.len(), 2);
        assert_eq!(m.records[1].id, 0x5a93);
        assert_eq!(m.records[0].tag, 0x80);
    }

    #[test]
    fn rejects_count_mismatch_and_huge_count() {
        let mut data = encode(&[1, 2]);
        data.pop();
        assert!(MapResources::parse(&data).is_err());
        let mut data = encode(&[]);
        data[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            MapResources::parse(&data),
            Err(Error::SizeMismatch { .. })
        ));
    }
}
