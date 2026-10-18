//! `master.index`: names each group's index and the resource files it uses.
//!
//! Derived from the single installed sample (docs/formats/archives.md, F004);
//! the parser requires every byte to be consumed so a misread layout fails
//! loudly instead of yielding plausible garbage.
//!
//! ```text
//! u8 0x04, "SER"
//! u16  unknown        observed 2
//! u16  group_count
//! group[group_count]  { str index_file, str base_resources_file }
//! u32  extra_count
//! extra[extra_count]  { str resources_file, u16 group }   // in slot order
//! u32  shared_count
//! shared[shared_count] { u32 n, u16 groups[n], str file }
//! ```
//! Strings are little-endian u32 length + UTF-8; numbers are big-endian.

use crate::Error;
use crate::reader::Reader;

pub const MASTER_MAGIC: [u8; 4] = [0x04, b'S', b'E', b'R'];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MasterGroup {
    pub index_file: String,
    /// Slot 0 is the base resources file; slots 1.. are the extra files
    /// assigned to this group, in master.index order.
    pub resource_files: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedFile {
    pub file: String,
    pub groups: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MasterIndex {
    pub unknown: u16,
    pub groups: Vec<MasterGroup>,
    pub shared: Vec<SharedFile>,
}

impl MasterIndex {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("master magic")?;
        if magic != MASTER_MAGIC {
            return Err(Error::BadMagic {
                expected: MASTER_MAGIC,
                found: magic,
            });
        }
        let unknown = r.u16_be("master unknown")?;
        let group_count = r.u16_be("group count")?;
        let mut groups = Vec::new();
        for _ in 0..group_count {
            let index_file = r.string_le32("group index file")?;
            let base = r.string_le32("group resources file")?;
            groups.push(MasterGroup {
                index_file,
                resource_files: vec![base],
            });
        }
        let extra_count = r.u32_be("extra resource count")?;
        for _ in 0..extra_count {
            let file = r.string_le32("extra resource file")?;
            let group = r.u16_be("extra resource group")? as usize;
            groups
                .get_mut(group)
                .ok_or(Error::UnknownGroup(group))?
                .resource_files
                .push(file);
        }
        let shared_count = r.u32_be("shared count")?;
        let mut shared = Vec::new();
        for _ in 0..shared_count {
            let n = r.u32_be("shared group count")? as usize;
            if n > r.remaining() / 2 {
                return Err(Error::Truncated {
                    field: "shared groups",
                    offset: r.pos() as u64,
                    needed: n as u64 * 2,
                    available: r.remaining() as u64,
                });
            }
            let groups = (0..n)
                .map(|_| r.u16_be("shared group"))
                .collect::<Result<Vec<_>, _>>()?;
            let file = r.string_le32("shared file")?;
            shared.push(SharedFile { file, groups });
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos() as u64,
                len: r.remaining() as u64,
            });
        }
        Ok(Self {
            unknown,
            groups,
            shared,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::index::tests::encode_str;

    /// Encodes a master index; `extras` are (file, group) in slot order.
    pub fn encode(
        groups: &[(&str, &str)],
        extras: &[(&str, u16)],
        shared: &[(&str, &[u16])],
    ) -> Vec<u8> {
        let mut out = MASTER_MAGIC.to_vec();
        out.extend_from_slice(&2u16.to_be_bytes());
        out.extend_from_slice(&(groups.len() as u16).to_be_bytes());
        for (i, r) in groups {
            encode_str(&mut out, i);
            encode_str(&mut out, r);
        }
        out.extend_from_slice(&(extras.len() as u32).to_be_bytes());
        for (f, g) in extras {
            encode_str(&mut out, f);
            out.extend_from_slice(&g.to_be_bytes());
        }
        out.extend_from_slice(&(shared.len() as u32).to_be_bytes());
        for (f, gs) in shared {
            out.extend_from_slice(&(gs.len() as u32).to_be_bytes());
            for g in *gs {
                out.extend_from_slice(&g.to_be_bytes());
            }
            encode_str(&mut out, f);
        }
        out
    }

    #[test]
    fn parses_installed_shape() {
        let data = encode(
            &[
                ("g1_patch.index", "g1.resources"),
                ("g2_patch.index", "g2.resources"),
            ],
            &[
                ("g1_001.resources", 0),
                ("g2_001.resources", 1),
                ("g1_patch.resources", 0),
            ],
            &[("shared_2.sharedrsc", &[1])],
        );
        let m = MasterIndex::parse(&data).unwrap();
        assert_eq!(
            m.groups[0].resource_files,
            ["g1.resources", "g1_001.resources", "g1_patch.resources"]
        );
        assert_eq!(
            m.groups[1].resource_files,
            ["g2.resources", "g2_001.resources"]
        );
        assert_eq!(m.shared[0].groups, [1]);
    }

    #[test]
    fn rejects_unknown_group_and_trailing_bytes() {
        let bad_group = encode(&[("a", "b")], &[("c", 3)], &[]);
        assert!(matches!(
            MasterIndex::parse(&bad_group),
            Err(Error::UnknownGroup(3))
        ));
        let mut trailing = encode(&[("a", "b")], &[], &[]);
        trailing.push(0);
        assert!(matches!(
            MasterIndex::parse(&trailing),
            Err(Error::TrailingData { .. })
        ));
    }

    #[test]
    fn rejects_huge_shared_group_count() {
        let mut data = encode(&[("a", "b")], &[], &[]);
        let n = data.len();
        data[n - 4..].copy_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            MasterIndex::parse(&data),
            Err(Error::Truncated { .. })
        ));
    }
}
