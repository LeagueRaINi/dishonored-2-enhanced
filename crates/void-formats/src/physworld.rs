//! Static map collision container (`generated/physworld/**.bphysworld`) and
//! the `physhape` `.hkx` prefix (docs/formats/havok.md).
//!
//! `.bphysworld` (little-endian; O: tutorial_p, every byte accounted for):
//!
//! ```text
//! u32 n | n x { u32 len, bytes }   contact-system material names
//! u32 m | m x u32                  material table (index into the names; meaning H)
//! u32 size | TCM0 type compendium (size bytes)
//! u32 size | TAG0 tagfile (size bytes, types via TCRF)
//! ```

use crate::Error;
use crate::havok::{TagFile, TypeSet};
use crate::reader::Reader;

#[derive(Clone, Debug)]
pub struct PhysWorld {
    pub contact_materials: Vec<String>,
    pub material_map: Vec<u32>,
    pub compendium: TypeSet,
    pub tagfile: TagFile,
}

impl PhysWorld {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let n = r.count(4, "contact material count")?;
        let contact_materials = (0..n)
            .map(|_| r.string("contact material"))
            .collect::<Result<Vec<_>, _>>()?;
        let material_map = r.u32_vec("material map")?;
        if let Some(&bad) = material_map.iter().find(|&&m| m as usize >= n) {
            return Err(Error::Invalid {
                field: "material map",
                detail: format!("{bad} >= {n} names"),
            });
        }
        let compendium = TypeSet::parse_compendium(r.sized_blob("type compendium")?)?;
        let tagfile = TagFile::parse(r.sized_blob("tagfile")?, Some(&compendium))?;
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        Ok(Self {
            contact_materials,
            material_map,
            compendium,
            tagfile,
        })
    }
}

/// `physhape` `.hkx`: a game prefix (5 zero bytes in all samples; meaning U)
/// followed by a self-contained `TAG0` tagfile.
pub fn parse_physhape(data: &[u8]) -> Result<(Vec<u8>, TagFile), Error> {
    let at = data
        .windows(4)
        .position(|w| w == b"TAG0")
        .filter(|&p| (4..=16).contains(&p))
        .ok_or(Error::Invalid {
            field: "physhape",
            detail: "no TAG0 section header within the first 20 bytes".into(),
        })?;
    let prefix = data[..at - 4].to_vec();
    Ok((prefix, TagFile::parse(&data[at - 4..], None)?))
}
