//! Renderer-independent preparation of original assets from the game's archives:
//! a cross-group name table ([`Archive`]), textures as stored mip chains
//! ([`texture`]), and arkDefault materials with their parameter packing
//! ([`material`]). Used by `void-rt` (material classification and albedo/gloss/metal
//! averages for the ray-tracing scene); the in-process renderer was removed (2026-10-17).

pub mod material;
pub mod probes;
pub mod texture;

use std::collections::HashMap;
use std::path::Path;

use void_res::Installation;

/// The installation plus a (kind, name) table over every archive group.
/// Lookups prefer `preferred_group` when several groups contain a name.
pub struct Archive {
    pub inst: Installation,
    names: HashMap<(String, String), Vec<(usize, usize)>>,
    pub preferred_group: usize,
}

impl Archive {
    pub fn open(game_dir: &Path) -> Result<Self, String> {
        let inst = Installation::open(game_dir).map_err(|e| e.to_string())?;
        let mut names: HashMap<(String, String), Vec<(usize, usize)>> = HashMap::new();
        for (g, group) in inst.groups.iter().enumerate() {
            for (i, e) in group.index.entries.iter().enumerate() {
                names
                    .entry((e.kind.clone(), e.name.clone()))
                    .or_default()
                    .push((g, i));
            }
        }
        Ok(Self {
            inst,
            names,
            preferred_group: 0,
        })
    }

    /// Names of `kind` starting with `prefix` and ending with `suffix` (sorted).
    pub fn find(&self, kind: &str, prefix: &str, suffix: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .names
            .keys()
            .filter(|(k, n)| k == kind && n.starts_with(prefix) && n.ends_with(suffix))
            .map(|(_, n)| n.clone())
            .collect();
        v.sort();
        v
    }

    /// All groups holding (kind, name).
    pub fn locate(&self, kind: &str, name: &str) -> &[(usize, usize)] {
        self.names
            .get(&(kind.to_owned(), name.to_owned()))
            .map_or(&[], Vec::as_slice)
    }

    pub fn read(&self, kind: &str, name: &str) -> Result<Vec<u8>, String> {
        let c = self.locate(kind, name);
        let &(g, i) = c
            .iter()
            .find(|(g, _)| *g == self.preferred_group)
            .or_else(|| c.first())
            .ok_or_else(|| format!("{kind} {name}: not in any index"))?;
        self.inst
            .read(g, &self.inst.groups[g].index.entries[i])
            .map_err(|e| e.to_string())
    }

    pub fn read_text(&self, kind: &str, name: &str) -> Result<String, String> {
        String::from_utf8(self.read(kind, name)?).map_err(|_| format!("{name} is not UTF-8"))
    }
}
