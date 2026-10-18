//! Resource overrides from a mods directory (docs/texture-replacement.md §3): layout,
//! name normalisation, validation against the original resources, and the bytes to
//! serve. Shared by `void-tool texmod` (offline check) and `void-inject` (in game), so
//! both accept exactly the same mods.
//!
//! Layout of a mods directory:
//! - `textures/<image name>.dds`: replaces `generated/image/<image name>.bimage7`
//!   (image name as in the archive, e.g. `models/props/foo_d.tga`). Converted to a fully
//!   resident BIM (every mip in the base file), so the engine never streams it.
//! - `override/<exact resource name>`: served byte for byte, any resource kind. Image
//!   bases and `_mipN` parts are validated like DDS replacements; other kinds must
//!   exist in the archive index.
//!
//! Archive access is abstracted by [`Originals`] so this module stays independent of
//! the archive reader.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::bim::{Format, Header, Image, check_replacement, mip_dim};
use crate::dds::Dds;

pub const IMAGE_PREFIX: &str = "generated/image/";
pub const IMAGE_EXT: &str = ".bimage7";
pub const TEXTURES_DIR: &str = "textures";
pub const OVERRIDE_DIR: &str = "override";

/// Engine-style resource name normalisation (O, `idFileSystemLocal_OpenFileRead` and
/// `idBackgroundLoader_GetFile`): lowercase, `\` -> `/`, absolute paths cut after
/// `/base/`, no leading `/`.
pub fn normalize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c == '\\' {
                '/'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect();
    let absolute = s.starts_with('/') || s.as_bytes().get(1) == Some(&b':');
    if absolute && let Some(i) = s.find("/base/") {
        s = s[i + "/base/".len()..].to_owned();
    }
    s.trim_start_matches('/').to_owned()
}

/// `generated/image/<name>.bimage7` for an image name (`models/x_d.tga`).
pub fn image_resource(image_name: &str) -> String {
    format!("{IMAGE_PREFIX}{}{IMAGE_EXT}", normalize(image_name))
}

/// Accepts an image resource name or a bare image name; returns the resource name.
pub fn image_resource_from_arg(arg: &str) -> String {
    let n = normalize(arg);
    if n.starts_with(IMAGE_PREFIX) && n.ends_with(IMAGE_EXT) {
        n
    } else {
        image_resource(&n)
    }
}

/// Image name of an image base resource (inverse of [`image_resource`]).
pub fn image_name(resource: &str) -> Option<&str> {
    resource.strip_prefix(IMAGE_PREFIX)?.strip_suffix(IMAGE_EXT)
}

/// Path of the DDS replacement of image base `resource` inside a mods directory.
pub fn texture_path(mods: &Path, resource: &str) -> Option<PathBuf> {
    let name = image_name(resource)?;
    let mut p = mods.join(TEXTURES_DIR);
    for c in name.split('/') {
        p.push(c);
    }
    let file = format!("{}.dds", p.file_name()?.to_str()?);
    p.set_file_name(file);
    Some(p)
}

/// `(base, n)` for a streamed part name `<base>_mip<n>` of an image base.
pub fn split_part(resource: &str) -> Option<(&str, u32)> {
    let (base, n) = resource.rsplit_once("_mip")?;
    if !base.ends_with(IMAGE_EXT) || n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((base, n.parse().ok()?))
}

fn is_image_base(resource: &str) -> bool {
    image_name(resource).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Dds,
    Raw,
}

#[derive(Clone, Debug)]
pub struct ModFile {
    /// Normalised resource name served.
    pub resource: String,
    pub path: PathBuf,
    pub source: Source,
}

/// The mods found in a directory, keyed by normalised resource name.
#[derive(Clone, Debug, Default)]
pub struct ModSet {
    pub dir: PathBuf,
    pub files: BTreeMap<String, ModFile>,
    /// Ignored files and conflicts, for the log.
    pub notes: Vec<String>,
}

fn walk(
    dir: &Path,
    rel: &mut Vec<String>,
    out: &mut Vec<(String, PathBuf)>,
    notes: &mut Vec<String>,
) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        let Some(name) = e.file_name().to_str().map(str::to_owned) else {
            notes.push(format!("{}: not a UTF-8 name, ignored", path.display()));
            continue;
        };
        if path.is_dir() {
            rel.push(name);
            walk(&path, rel, out, notes);
            rel.pop();
        } else {
            let mut r = rel.join("/");
            if !r.is_empty() {
                r.push('/');
            }
            r.push_str(&name);
            out.push((r, path));
        }
    }
}

impl ModSet {
    /// Lists `dir/textures` and `dir/override` (names only, no file contents). Cheap
    /// enough for DLL attach.
    pub fn scan(dir: &Path) -> Self {
        let mut set = ModSet {
            dir: dir.to_owned(),
            ..Default::default()
        };
        if !dir.is_dir() {
            set.notes
                .push(format!("mods directory {} does not exist", dir.display()));
            return set;
        }
        let mut found = Vec::new();
        walk(
            &dir.join(OVERRIDE_DIR),
            &mut Vec::new(),
            &mut found,
            &mut set.notes,
        );
        for (rel, path) in found {
            let resource = normalize(&rel);
            set.files.insert(
                resource.clone(),
                ModFile {
                    resource,
                    path,
                    source: Source::Raw,
                },
            );
        }
        let mut found = Vec::new();
        walk(
            &dir.join(TEXTURES_DIR),
            &mut Vec::new(),
            &mut found,
            &mut set.notes,
        );
        for (rel, path) in found {
            let lower = rel.to_ascii_lowercase();
            let Some(name) = lower.strip_suffix(".dds") else {
                set.notes
                    .push(format!("{}: not a .dds file, ignored", path.display()));
                continue;
            };
            let resource = image_resource(name);
            if set.files.contains_key(&resource) {
                set.notes.push(format!(
                    "{}: ignored, {OVERRIDE_DIR}/{resource} replaces the same resource",
                    path.display()
                ));
                continue;
            }
            set.files.insert(
                resource.clone(),
                ModFile {
                    resource,
                    path,
                    source: Source::Dds,
                },
            );
        }
        set
    }
}

/// Access to the game's original resources.
pub trait Originals {
    /// Every copy of image resource `name` across archive groups (identical copies may
    /// be returned once); empty if no group has it.
    fn images(&self, name: &str) -> Result<Vec<Vec<u8>>, String>;
    /// True if any archive entry (any kind) has this name.
    fn exists(&self, name: &str) -> bool;
}

/// Outcome of a successful [`check`].
#[derive(Clone, Debug)]
pub struct Checked {
    /// Header of the original image (template for DDS conversion); `None` for
    /// non-image resources.
    pub template: Option<Header>,
    /// The resource name exists in the archive index (the engine's loaders may have
    /// queued it; see the call-through rule in docs/texture-replacement.md §3.1).
    pub in_index: bool,
    pub summary: String,
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn originals(orig: &dyn Originals, base: &str) -> Result<Vec<Image>, String> {
    let copies = orig.images(base)?;
    if copies.is_empty() {
        return Err(format!(
            "{base} is not an image in the archive index (only existing resources can be replaced)"
        ));
    }
    copies
        .iter()
        .map(|b| Image::parse(b).map_err(|e| format!("original {base}: {e}")))
        .collect()
}

fn describe(h: &Header) -> String {
    format!(
        "{}x{} mips {}/{}",
        h.width, h.height, h.mips_in_file, h.total_mips
    )
}

/// Validates the mod for `resource` against the originals. Image mods are checked with
/// [`check_replacement`] against every archive copy of the original.
pub fn check(set: &ModSet, resource: &str, orig: &dyn Originals) -> Result<Checked, String> {
    let m = set
        .files
        .get(resource)
        .ok_or_else(|| format!("{resource}: no mod"))?;
    if let Some((base, n)) = split_part(resource) {
        if let Some(bm) = set.files.get(base) {
            if bm.source == Source::Dds {
                return Err(format!(
                    "{resource}: conflicts with the DDS replacement of {base}, which keeps every mip in the base file"
                ));
            }
            let c = check(set, base, orig)?;
            return Ok(Checked {
                template: c.template,
                in_index: orig.exists(resource),
                summary: format!("part {n} of replaced {base}"),
            });
        }
        let data = read(&m.path)?;
        let olds = originals(orig, base)?;
        for o in &olds {
            if n >= o.streamed_mips() {
                return Err(format!(
                    "{resource}: the original has {} streamed parts",
                    o.streamed_mips()
                ));
            }
            o.parse_streamed(n, &data)
                .map_err(|e| format!("{resource}: {e}"))?;
        }
        return Ok(Checked {
            template: Some(olds[0].header.clone()),
            in_index: orig.exists(resource),
            summary: format!(
                "part {n} of original {} ({} bytes)",
                describe(&olds[0].header),
                data.len()
            ),
        });
    }
    if is_image_base(resource) {
        let olds = originals(orig, resource)?;
        let first = &olds[0].header;
        let data = read(&m.path)?;
        let (img, parts) = match m.source {
            Source::Dds => {
                let dds = Dds::parse(&data).map_err(|e| format!("{}: {e}", m.path.display()))?;
                let img = Image::from_dds(&dds, first).map_err(|e| format!("{resource}: {e}"))?;
                (img, Vec::new())
            }
            Source::Raw => {
                let img = Image::parse(&data).map_err(|e| format!("{resource}: {e}"))?;
                let parts = (0..img.streamed_mips())
                    .map(|n| {
                        let name = Image::streamed_mip_name(resource, n);
                        match set.files.get(&name) {
                            Some(p) if p.source == Source::Raw => read(&p.path),
                            _ => Err(format!(
                                "{resource}: streamed part {name} missing from {OVERRIDE_DIR}/"
                            )),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (img, parts)
            }
        };
        for o in &olds {
            check_replacement(&o.header, &img, &parts).map_err(|e| format!("{resource}: {e}"))?;
        }
        return Ok(Checked {
            template: Some(first.clone()),
            in_index: true,
            summary: format!(
                "{} {} (original {})",
                img.format
                    .map_or_else(|| "?".to_owned(), |f| format!("{f:?}")),
                describe(&img.header),
                describe(first)
            ),
        });
    }
    if m.source == Source::Dds {
        return Err(format!("{resource}: DDS mods must target images"));
    }
    if !orig.exists(resource) {
        return Err(format!(
            "{resource}: not in the archive index (only existing resources can be replaced)"
        ));
    }
    let len = std::fs::metadata(&m.path)
        .map_err(|e| format!("{}: {e}", m.path.display()))?
        .len();
    Ok(Checked {
        template: None,
        in_index: true,
        summary: format!("{len} bytes"),
    })
}

/// Bytes to serve for a mod that passed [`check`]: the converted BIM for DDS sources,
/// the file itself otherwise.
pub fn materialize(m: &ModFile, checked: &Checked) -> Result<Vec<u8>, String> {
    let data = read(&m.path)?;
    match m.source {
        Source::Raw => Ok(data),
        Source::Dds => {
            let template = checked
                .template
                .as_ref()
                .ok_or_else(|| format!("{}: no template", m.resource))?;
            let dds = Dds::parse(&data).map_err(|e| format!("{}: {e}", m.path.display()))?;
            let img =
                Image::from_dds(&dds, template).map_err(|e| format!("{}: {e}", m.resource))?;
            Ok(img.to_bytes())
        }
    }
}

/// Converts a colour byte (0..255 meaning -1..1) to a signed-normalised byte.
fn snorm(v: u8) -> u8 {
    ((v as f32 / 255.0 * 2.0 - 1.0) * 127.0).round() as i8 as u8
}

/// BC7 mode 6 block (RGBA 7.7.7.7 + one p-bit per endpoint, 4-bit indices) whose 16
/// texels all decode to `c` (within 1 when the channels' low bits differ: the p-bit
/// is shared by the four channels of an endpoint).
pub fn bc7_constant_block(c: [u8; 4]) -> [u8; 16] {
    let p = u128::from(c.iter().filter(|v| *v & 1 == 1).count() >= 2);
    let mut bits: u128 = 1 << 6; // mode 6: six 0 bits, then 1
    let mut pos = 7;
    for v in c {
        let v = u128::from(v >> 1);
        bits |= v << pos | v << (pos + 7); // endpoint 0 and 1 of this channel
        pos += 14;
    }
    bits |= p << 63 | p << 64; // indices (bits 65..127) stay 0: texel = endpoint 0
    bits.to_le_bytes()
}

/// BC4 block of one constant value (equal endpoints, all indices 0).
fn bc4_constant_block(v: u8) -> [u8; 8] {
    [v, v, 0, 0, 0, 0, 0, 0]
}

/// One constant-colour texel/block of `format`; `None` for formats without an encoder
/// here (BC6H).
fn constant_unit(format: Format, c: [u8; 4]) -> Option<Vec<u8>> {
    use Format::*;
    Some(match format {
        R8Unorm => vec![c[0]],
        A8Unorm => vec![c[3]],
        R16Unorm => vec![c[0], c[0]],
        Rg8Unorm => vec![c[0], c[1]],
        Rg8Snorm => vec![snorm(c[0]), snorm(c[1])],
        Rgba8Unorm | Rgba8UnormSrgb => c.to_vec(),
        Bc4Unorm => bc4_constant_block(c[0]).to_vec(),
        Bc4Snorm => bc4_constant_block(snorm(c[0])).to_vec(),
        Bc5Unorm => [bc4_constant_block(c[0]), bc4_constant_block(c[1])].concat(),
        Bc5Snorm => [
            bc4_constant_block(snorm(c[0])),
            bc4_constant_block(snorm(c[1])),
        ]
        .concat(),
        Bc7Unorm | Bc7UnormSrgb => bc7_constant_block(c).to_vec(),
        Bc6hUf16 | Bc6hSf16 => return None,
    })
}

/// A test texture of `format` and size whose mip `i` is the constant colour
/// `colour(i)` (RGBA bytes; signed formats map 0..255 to -1..1).
pub fn constant_dds(
    format: Format,
    width: u32,
    height: u32,
    mips: u32,
    colour: &dyn Fn(u32) -> [u8; 4],
) -> Result<Dds, String> {
    let mut out = Vec::with_capacity(mips as usize);
    for i in 0..mips {
        let unit = constant_unit(format, colour(i))
            .ok_or_else(|| format!("no constant-colour encoder for {format:?}"))?;
        let size = format.mip_size(mip_dim(width, i), mip_dim(height, i));
        out.push(unit.iter().copied().cycle().take(size).collect());
    }
    Ok(Dds {
        format,
        width,
        height,
        mips: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn names() {
        assert_eq!(
            normalize("Generated\\Image\\A.TGA"),
            "generated/image/a.tga"
        );
        assert_eq!(
            normalize("D:/Games/Dishonored2/base/generated/x.decl"),
            "generated/x.decl"
        );
        assert_eq!(normalize("/generated/x"), "generated/x");
        assert_eq!(
            image_resource_from_arg("models/A_d.tga"),
            "generated/image/models/a_d.tga.bimage7"
        );
        assert_eq!(
            image_resource_from_arg("generated/image/m/a.tga.bimage7"),
            "generated/image/m/a.tga.bimage7"
        );
        assert_eq!(
            split_part("generated/image/a.tga.bimage7_mip12"),
            Some(("generated/image/a.tga.bimage7", 12))
        );
        assert_eq!(split_part("generated/image/a_mip1.tga.bimage7"), None);
        assert_eq!(
            texture_path(Path::new("m"), "generated/image/x/y.tga.bimage7").unwrap(),
            Path::new("m").join("textures").join("x").join("y.tga.dds")
        );
    }

    /// Decodes texel 0 of a BC7 mode 6 block (reference for the encoder).
    fn bc7_mode6_texel0(b: [u8; 16]) -> [u8; 4] {
        let bits = u128::from_le_bytes(b);
        assert_eq!(bits & 0x7F, 1 << 6, "mode 6");
        let field = |pos: u32| ((bits >> pos) & 0x7F) as u8;
        let p0 = ((bits >> 63) & 1) as u8;
        let idx0 = (bits >> 65) & 0x7;
        assert_eq!(idx0, 0);
        std::array::from_fn(|ch| field(7 + 14 * ch as u32) << 1 | p0)
    }

    #[test]
    fn constant_blocks_decode_to_their_colour() {
        for c in [[255, 0, 255, 255], [10, 20, 30, 40], [0, 0, 0, 0]] {
            let d = bc7_mode6_texel0(bc7_constant_block(c));
            for ch in 0..4 {
                assert!(d[ch].abs_diff(c[ch]) <= 1, "{c:?} -> {d:?}");
            }
        }
        let dds = constant_dds(Format::Bc5Snorm, 8, 4, 2, &|_| [255, 128, 0, 0]).unwrap();
        let v = crate::bim::decode_bc45(&dds.mips[0], 8, 4, 2, true).unwrap();
        assert!((v[0] - 1.0).abs() < 0.01 && v[1].abs() < 0.01);
        assert!(constant_dds(Format::Bc6hUf16, 4, 4, 1, &|_| [0; 4]).is_err());
    }

    struct Fake(HashMap<String, Vec<u8>>);
    impl Originals for Fake {
        fn images(&self, name: &str) -> Result<Vec<Vec<u8>>, String> {
            Ok(self.0.get(name).cloned().into_iter().collect())
        }
        fn exists(&self, name: &str) -> bool {
            self.0.contains_key(name)
        }
    }

    fn original() -> Vec<u8> {
        // 16x16 BC7 sRGB, 3 mips, 16 streamed, 8 and 4 in the base file.
        let dds = constant_dds(Format::Bc7UnormSrgb, 8, 8, 2, &|_| [1, 2, 3, 4]).unwrap();
        let mut h = [0u8; 0x3E];
        h[..4].copy_from_slice(&crate::bim::MAGIC);
        let mut img = Image::parse(&{
            let mut b = h.to_vec();
            for (o, v) in [
                (0x04, 19u32),
                (0x0C, 0x14),
                (0x10, 6),
                (0x14, 1),
                (0x18, 8),
                (0x1C, 8),
                (0x20, 8),
                (0x24, 8),
                (0x28, 1),
                (0x2C, 2),
                (0x30, 2),
                (0x34, 1),
            ] {
                b[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
            for (i, m) in dds.mips.iter().enumerate() {
                b.extend(
                    crate::bim::Mip {
                        level: i as u32,
                        face: 0,
                        width: 8 >> i,
                        height: 8 >> i,
                        data: m.clone(),
                    }
                    .to_bytes(),
                );
            }
            b
        })
        .unwrap();
        img.header.width = 16;
        img.header.height = 16;
        img.header.total_mips = 3;
        img.to_bytes()
    }

    #[test]
    fn scan_check_and_materialize() {
        let dir = std::env::temp_dir().join(format!("void-texmod-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = "generated/image/m/wall_d.tga.bimage7";
        let orig = Fake(HashMap::from([
            (base.to_owned(), original()),
            ("generated/decls/material/m.decl".to_owned(), Vec::new()),
        ]));
        let tex = texture_path(&dir, base).unwrap();
        std::fs::create_dir_all(tex.parent().unwrap()).unwrap();
        let big = constant_dds(Format::Bc7UnormSrgb, 32, 32, 6, &|_| [255, 0, 255, 255]).unwrap();
        std::fs::write(&tex, big.to_bytes().unwrap()).unwrap();
        let ov = dir.join(OVERRIDE_DIR).join("generated/decls/material");
        std::fs::create_dir_all(&ov).unwrap();
        std::fs::write(ov.join("m.decl"), b"x").unwrap();
        std::fs::write(ov.join("new.decl"), b"x").unwrap();
        let set = ModSet::scan(&dir);
        assert_eq!(set.files.len(), 3, "{:?}", set.files.keys());
        let c = check(&set, base, &orig).unwrap();
        assert!(c.in_index);
        let bytes = materialize(&set.files[base], &c).unwrap();
        let img = Image::parse(&bytes).unwrap();
        assert_eq!(
            (img.header.width, img.header.total_mips, img.streamed_mips()),
            (32, 4, 0)
        );
        assert!(check(&set, "generated/decls/material/m.decl", &orig).is_ok());
        assert!(check(&set, "generated/decls/material/new.decl", &orig).is_err());
        // Wrong format class is refused.
        let bad = constant_dds(Format::Bc7Unorm, 32, 32, 6, &|_| [0; 4]).unwrap();
        std::fs::write(&tex, bad.to_bytes().unwrap()).unwrap();
        assert!(check(&set, base, &orig).is_err());
        // A raw part next to a DDS base conflicts.
        let part_dir = dir.join(OVERRIDE_DIR).join("generated/image/m");
        std::fs::create_dir_all(&part_dir).unwrap();
        std::fs::write(part_dir.join("wall_d.tga.bimage7_mip0"), b"x").unwrap();
        let set = ModSet::scan(&dir);
        assert!(check(&set, &format!("{base}_mip0"), &orig).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
