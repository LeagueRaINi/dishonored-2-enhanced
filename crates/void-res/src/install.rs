//! Resolves an installation's `base/` directory into readable resources.
//!
//! Precedence model (F005): master.index names one index per group (the
//! `_patch` index), and each later index is a full superset of earlier ones.
//! The named index is therefore authoritative for its group; older numbered
//! indexes are not consulted. Entries of different groups are not merged:
//! the same name may legitimately exist in several groups.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;

use crate::Error;
use crate::error::io;
use crate::index::{Entry, Index};
use crate::master::MasterIndex;

pub struct Group {
    pub index_path: PathBuf,
    pub resource_paths: Vec<PathBuf>,
    pub index: Index,
}

pub struct Installation {
    pub base: PathBuf,
    pub master: MasterIndex,
    pub groups: Vec<Group>,
    pub shared_paths: Vec<PathBuf>,
}

impl Installation {
    /// Opens `<game_root>/base`. The path must be supplied explicitly.
    pub fn open(game_root: &Path) -> Result<Self, Error> {
        let base = game_root.join("base");
        let master_path = base.join("master.index");
        let master = MasterIndex::parse(&std::fs::read(&master_path).map_err(io(&master_path))?)?;
        let mut groups = Vec::with_capacity(master.groups.len());
        for g in &master.groups {
            let index_path = base.join(&g.index_file);
            let index = Index::parse(&std::fs::read(&index_path).map_err(io(&index_path))?)?;
            groups.push(Group {
                index_path,
                resource_paths: g.resource_files.iter().map(|f| base.join(f)).collect(),
                index,
            });
        }
        let shared_paths = master.shared.iter().map(|s| base.join(&s.file)).collect();
        Ok(Self {
            base,
            master,
            groups,
            shared_paths,
        })
    }

    pub fn group(&self, group: usize) -> Result<&Group, Error> {
        self.groups.get(group).ok_or(Error::UnknownGroup(group))
    }

    /// Resolves the file holding `entry`, which must belong to group `group`.
    pub fn resource_path(&self, group: usize, entry: &Entry) -> Result<&Path, Error> {
        let loc = entry.location;
        let fail = |reason| Error::UnresolvedLocation {
            entry: entry.name.clone(),
            location: loc.0,
            reason,
        };
        if loc.unknown_bits() != 0 {
            return Err(fail("unknown location bits set"));
        }
        if loc.is_shared() {
            if loc.0 & 0x1f != 0 {
                return Err(fail("shared location with group/slot bits set"));
            }
            // Pick the shared file that serves this group.
            return self
                .master
                .shared
                .iter()
                .zip(&self.shared_paths)
                .find(|(s, _)| s.groups.contains(&(group as u16)))
                .map(|(_, p)| p.as_path())
                .ok_or_else(|| fail("no shared file serves this group"));
        }
        if loc.group() != group {
            return Err(fail("location group differs from owning index"));
        }
        self.group(group)?
            .resource_paths
            .get(loc.slot())
            .map(PathBuf::as_path)
            .ok_or_else(|| fail("slot not defined by master.index"))
    }

    /// Reads and, if needed, decompresses one entry, verifying sizes exactly.
    pub fn read(&self, group: usize, entry: &Entry) -> Result<Vec<u8>, Error> {
        let path = self.resource_path(group, entry)?;
        let mut file = File::open(path).map_err(io(path))?;
        let file_len = file.metadata().map_err(io(path))?.len();
        let packed = u64::from(entry.packed_size);
        if entry
            .offset
            .checked_add(packed)
            .is_none_or(|end| end > file_len)
        {
            return Err(Error::EntryOutOfRange {
                entry: entry.name.clone(),
                offset: entry.offset,
                packed_size: packed,
                file_len,
            });
        }
        file.seek(SeekFrom::Start(entry.offset)).map_err(io(path))?;
        let mut stored = vec![0u8; entry.packed_size as usize];
        file.read_exact(&mut stored).map_err(io(path))?;
        decode(entry, stored)
    }
}

/// Converts stored bytes to the entry's contents.
pub fn decode(entry: &Entry, stored: Vec<u8>) -> Result<Vec<u8>, Error> {
    if !entry.is_compressed() {
        return Ok(stored);
    }
    let fail = |detail: String| Error::Decompress {
        entry: entry.name.clone(),
        detail,
    };
    let size = entry.size as usize;
    let mut decoder = ZlibDecoder::new(stored.as_slice());
    let mut out = Vec::with_capacity(size);
    // Read one byte past the declared size to detect oversized streams
    // without unbounded allocation.
    (&mut decoder)
        .take(size as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| fail(e.to_string()))?;
    if out.len() != size {
        return Err(fail(format!(
            "produced {} bytes, declared {size}",
            out.len()
        )));
    }
    if decoder.total_in() != stored.len() as u64 {
        return Err(fail(format!(
            "consumed {} of {} stored bytes",
            decoder.total_in(),
            stored.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tests::{encode as encode_index, entry};
    use crate::master::tests::encode as encode_master;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use std::io::Write;

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("void-res-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("base")).unwrap();
        dir
    }

    #[test]
    fn decode_validates_sizes() {
        let packed = zlib(b"hello world");
        let mut e = entry("x", 0, 11, packed.len() as u32, 0);
        assert_eq!(decode(&e, packed.clone()).unwrap(), b"hello world");
        e.size = 5;
        assert!(
            decode(&e, packed.clone()).is_err(),
            "oversized stream accepted"
        );
        e.size = 20;
        assert!(
            decode(&e, packed.clone()).is_err(),
            "undersized stream accepted"
        );
        e.size = 11;
        let mut trailing = packed;
        trailing.push(0);
        e.packed_size = trailing.len() as u32;
        assert!(
            decode(&e, trailing).is_err(),
            "trailing stored bytes accepted"
        );
        e.packed_size = 3;
        assert!(decode(&e, vec![1, 2, 3]).is_err(), "garbage accepted");
    }

    #[test]
    fn reads_through_master_slots_and_shared() {
        let root = temp_dir("install");
        let base = root.join("base");
        let compressed = zlib(b"compressed payload");
        let mut res0 = b"\x04SERplain".to_vec();
        res0.extend_from_slice(&compressed);
        std::fs::write(base.join("g1.resources"), &res0).unwrap();
        std::fs::write(base.join("g1_001.resources"), b"\x04SERpatched").unwrap();
        std::fs::write(base.join("g2.resources"), b"\x04SER").unwrap();
        std::fs::write(base.join("s.sharedrsc"), b"\x04SERshared").unwrap();
        std::fs::write(
            base.join("master.index"),
            encode_master(
                &[("g1.index", "g1.resources"), ("g2.index", "g2.resources")],
                &[("g1_001.resources", 0)],
                &[("s.sharedrsc", &[1])],
            ),
        )
        .unwrap();
        let g1 = vec![
            entry("plain", 4, 5, 5, 0x0000),
            entry("zipped", 9, 18, compressed.len() as u32, 0x0000),
            entry("patched", 4, 7, 7, 0x0004),
            entry("past_end", 4, 100, 100, 0x0000),
            entry("bad_slot", 4, 1, 1, 0x0008),
        ];
        std::fs::write(base.join("g1.index"), encode_index(&g1)).unwrap();
        std::fs::write(
            base.join("g2.index"),
            encode_index(&[entry("shared", 4, 6, 6, 0x8000)]),
        )
        .unwrap();

        let inst = Installation::open(&root).unwrap();
        let read = |g: usize, i: usize| inst.read(g, &inst.groups[g].index.entries[i]);
        assert_eq!(read(0, 0).unwrap(), b"plain");
        assert_eq!(read(0, 1).unwrap(), b"compressed payload");
        assert_eq!(read(0, 2).unwrap(), b"patched");
        assert!(matches!(read(0, 3), Err(Error::EntryOutOfRange { .. })));
        assert!(matches!(read(0, 4), Err(Error::UnresolvedLocation { .. })));
        assert_eq!(read(1, 0).unwrap(), b"shared");
        // Group 0 has no shared file; a shared entry there must not resolve.
        let stray = entry("stray", 4, 1, 1, 0x8000);
        assert!(inst.read(0, &stray).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
