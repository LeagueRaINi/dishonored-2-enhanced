//! `.index` files: the table of contents for a resource group.
//!
//! Layout (evidence: docs/formats/archives.md, F002/F003). Numbers are
//! big-endian except string lengths, which are little-endian u32.
//!
//! ```text
//! header (32 bytes)
//!   u8   type           always 0x05
//!   [3]  "SER"
//!   u32  body_size      file length - 32
//!   [24] reserved       zero in all observed files
//! body
//!   u32  entry_count
//!   entry[entry_count]
//!     u32  id
//!     str  kind         e.g. "image", "model", "File"
//!     str  source       pre-build name
//!     str  name         built resource name (unique key used by consumers)
//!     u64  offset       into the resource file selected by `location`
//!     u32  size         uncompressed size
//!     u32  packed_size  stored size; zlib-compressed when != size
//!     u32  reserved     zero in all observed entries
//!     u32  flags        observed values: 0, 0x20 (meaning unknown)
//!     u16  location     see [`Location`]
//! ```

use std::collections::HashMap;

use crate::Error;
use crate::reader::Reader;

pub const INDEX_MAGIC: [u8; 4] = [0x05, b'S', b'E', b'R'];
pub const HEADER_LEN: usize = 32;
/// Smallest possible encoded entry: id + 3 empty strings + fixed fields.
const MIN_ENTRY_LEN: usize = 4 + 3 * 4 + 8 + 4 + 4 + 4 + 4 + 2;

/// Selects the file containing an entry's bytes.
///
/// Observed encoding: bits 0-1 are the zero-based group (game1..game3),
/// bits 2-4 the slot within that group (0 = base `.resources`, 1-5 =
/// `_001`..`_005`, 6 = `_patch`), and bit 15 marks the shared resource file.
/// Bits 5-14 have never been observed set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Location(pub u16);

impl Location {
    pub const SHARED_BIT: u16 = 0x8000;
    pub const KNOWN_BITS: u16 = Self::SHARED_BIT | 0x1f;

    pub fn group(self) -> usize {
        (self.0 & 0x3) as usize
    }

    pub fn slot(self) -> usize {
        ((self.0 >> 2) & 0x7) as usize
    }

    pub fn is_shared(self) -> bool {
        self.0 & Self::SHARED_BIT != 0
    }

    pub fn unknown_bits(self) -> u16 {
        self.0 & !Self::KNOWN_BITS
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: u32,
    pub kind: String,
    pub source: String,
    pub name: String,
    pub offset: u64,
    pub size: u32,
    pub packed_size: u32,
    pub reserved: u32,
    pub flags: u32,
    pub location: Location,
}

impl Entry {
    pub fn is_compressed(&self) -> bool {
        self.size != self.packed_size
    }
}

#[derive(Clone, Debug)]
pub struct Index {
    pub entries: Vec<Entry>,
    /// Non-zero bytes in the reserved header area (kept for diagnostics).
    pub header_reserved: [u8; 24],
}

/// Lookup tables over one index. Ids are scoped to their group's index and
/// name a logical resource that may span several entries (F009); names are
/// unique only together with the kind (F007).
pub struct Lookup<'a> {
    index: &'a Index,
    by_id: HashMap<u32, Vec<usize>>,
    by_kind_name: HashMap<(&'a str, &'a str), Vec<usize>>,
}

impl<'a> Lookup<'a> {
    pub fn new(index: &'a Index) -> Self {
        let mut by_id: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut by_kind_name: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
        for (i, e) in index.entries.iter().enumerate() {
            by_id.entry(e.id).or_default().push(i);
            by_kind_name
                .entry((e.kind.as_str(), e.name.as_str()))
                .or_default()
                .push(i);
        }
        Self {
            index,
            by_id,
            by_kind_name,
        }
    }

    /// All entries forming logical resource `id`, in index order.
    pub fn by_id(&self, id: u32) -> Vec<&'a Entry> {
        self.collect(self.by_id.get(&id))
    }

    /// All entries with this kind and name, in index order.
    pub fn by_kind_name(&self, kind: &str, name: &str) -> Vec<&'a Entry> {
        self.collect(self.by_kind_name.get(&(kind, name)))
    }

    fn collect(&self, idx: Option<&Vec<usize>>) -> Vec<&'a Entry> {
        let entries = &self.index.entries;
        idx.map_or(Vec::new(), |v| v.iter().map(|&i| &entries[i]).collect())
    }

    /// Entries matching `name` regardless of kind.
    pub fn by_name<'s>(&'s self, name: &'s str) -> impl Iterator<Item = &'a Entry> + 's {
        self.index.entries.iter().filter(move |e| e.name == name)
    }
}

impl Index {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("index magic")?;
        if magic != INDEX_MAGIC {
            return Err(Error::BadMagic {
                expected: INDEX_MAGIC,
                found: magic,
            });
        }
        let body_size = r.u32_be("index body size")? as u64;
        let header_reserved: [u8; 24] = r.array("index header reserved")?;
        let actual = (data.len() - HEADER_LEN) as u64;
        if body_size != actual {
            return Err(Error::SizeMismatch {
                field: "index body size",
                declared: body_size,
                actual,
            });
        }

        let count = r.u32_be("entry count")? as usize;
        // Bound the allocation by what the remaining bytes could possibly hold.
        if count > r.remaining() / MIN_ENTRY_LEN {
            return Err(Error::Truncated {
                field: "entry table",
                offset: r.pos() as u64,
                needed: count as u64 * MIN_ENTRY_LEN as u64,
                available: r.remaining() as u64,
            });
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(Entry {
                id: r.u32_be("entry id")?,
                kind: r.string_le32("entry kind")?,
                source: r.string_le32("entry source")?,
                name: r.string_le32("entry name")?,
                offset: r.u64_be("entry offset")?,
                size: r.u32_be("entry size")?,
                packed_size: r.u32_be("entry packed size")?,
                reserved: r.u32_be("entry reserved")?,
                flags: r.u32_be("entry flags")?,
                location: Location(r.u16_be("entry location")?),
            });
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos() as u64,
                len: r.remaining() as u64,
            });
        }
        Ok(Self {
            entries,
            header_reserved,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn encode_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    pub fn encode(entries: &[Entry]) -> Vec<u8> {
        let mut body = (entries.len() as u32).to_be_bytes().to_vec();
        for e in entries {
            body.extend_from_slice(&e.id.to_be_bytes());
            encode_str(&mut body, &e.kind);
            encode_str(&mut body, &e.source);
            encode_str(&mut body, &e.name);
            body.extend_from_slice(&e.offset.to_be_bytes());
            body.extend_from_slice(&e.size.to_be_bytes());
            body.extend_from_slice(&e.packed_size.to_be_bytes());
            body.extend_from_slice(&e.reserved.to_be_bytes());
            body.extend_from_slice(&e.flags.to_be_bytes());
            body.extend_from_slice(&e.location.0.to_be_bytes());
        }
        let mut out = INDEX_MAGIC.to_vec();
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(&[0; 24]);
        out.extend_from_slice(&body);
        out
    }

    pub fn entry(name: &str, offset: u64, size: u32, packed: u32, loc: u16) -> Entry {
        Entry {
            id: 7,
            kind: "File".into(),
            source: format!("src/{name}"),
            name: name.into(),
            offset,
            size,
            packed_size: packed,
            reserved: 0,
            flags: 0x20,
            location: Location(loc),
        }
    }

    #[test]
    fn roundtrip() {
        let entries = vec![
            entry("a/b.decl", 4, 10, 10, 0x0000),
            entry("c.bin", 14, 100, 30, 0x8000),
        ];
        let idx = Index::parse(&encode(&entries)).unwrap();
        assert_eq!(idx.entries, entries);
    }

    #[test]
    fn empty_index_matches_game4_shape() {
        // game4*.index files are exactly this 36-byte form.
        let data = encode(&[]);
        assert_eq!(data.len(), 36);
        assert!(Index::parse(&data).unwrap().entries.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = encode(&[]);
        data[0] = 0x04;
        assert!(matches!(Index::parse(&data), Err(Error::BadMagic { .. })));
    }

    #[test]
    fn rejects_body_size_mismatch() {
        let mut data = encode(&[entry("x", 4, 1, 1, 0)]);
        data.push(0);
        assert!(matches!(
            Index::parse(&data),
            Err(Error::SizeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_every_truncation() {
        let full = encode(&[entry("abc", 4, 1, 1, 0), entry("def", 5, 1, 1, 0)]);
        for cut in HEADER_LEN..full.len() {
            let mut data = full[..cut].to_vec();
            let body = (cut - HEADER_LEN) as u32;
            data[4..8].copy_from_slice(&body.to_be_bytes());
            assert!(Index::parse(&data).is_err(), "cut at {cut} parsed");
        }
    }

    #[test]
    fn huge_count_does_not_allocate() {
        let mut data = encode(&[]);
        data[32..36].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(Index::parse(&data), Err(Error::Truncated { .. })));
    }

    #[test]
    fn huge_string_length_is_rejected() {
        let mut data = encode(&[entry("abc", 4, 1, 1, 0)]);
        // First string length follows count (4) and id (4) after the header.
        data[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(Index::parse(&data), Err(Error::Truncated { .. })));
    }

    #[test]
    fn rejects_invalid_utf8() {
        let mut data = encode(&[entry("abc", 4, 1, 1, 0)]);
        let pos = data.windows(3).position(|w| w == b"abc").unwrap();
        data[pos] = 0xff;
        assert!(matches!(
            Index::parse(&data),
            Err(Error::InvalidUtf8 { .. })
        ));
    }

    #[test]
    fn location_fields() {
        let l = Location(0x0019); // game2_patch observed value
        assert_eq!((l.group(), l.slot(), l.is_shared()), (1, 6, false));
        assert!(Location(0x8000).is_shared());
        assert_eq!(Location(0x0040).unknown_bits(), 0x0040);
        assert_eq!(Location(0x801f).unknown_bits(), 0);
    }
}
