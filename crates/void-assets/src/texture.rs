//! Material textures as stored mip chains (BC formats are kept compressed; the
//! renderer uploads them natively and the shader applies the game's decoding).

use void_formats::bim::{Format, Image as Bim};

use crate::Archive;

#[derive(Clone, Debug)]
pub struct Mip {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct TextureData {
    pub name: String,
    pub format: Format,
    /// Largest first.
    pub mips: Vec<Mip>,
}

/// How a map is sampled: colour maps must be BC7 (sRGB or linear); raw maps are
/// BC4/BC5/R8 and decoded by the shader (e.g. `UnpackBC5NormalMap`). A raw slot may also
/// hold a BC7 image when a material assigns a colour texture to it (e.g. `white_stone_01`
/// layer 1/2 bump and gloss = `plaster_building_dirt_03_d`): the game binds the image's
/// own view and samples it as is, so we do the same and the caller notes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Usage {
    Color,
    Raw,
}

pub fn format_allowed(usage: Usage, f: Format) -> bool {
    match usage {
        Usage::Color => matches!(f, Format::Bc7UnormSrgb | Format::Bc7Unorm),
        Usage::Raw => matches!(
            f,
            Format::Bc4Unorm
                | Format::Bc4Snorm
                | Format::Bc5Unorm
                | Format::Bc5Snorm
                | Format::R8Unorm
                | Format::R16Unorm
                | Format::Bc7UnormSrgb
                | Format::Bc7Unorm
        ),
    }
}

/// Archive image name for a material texture source. Sources with `$` load options keep
/// them in the archive name (O, archive index): `x_d$nomips` -> `x_d$nomips.tga.bimage7`,
/// `x_n.tga$nomips` -> `x_n.tga$nomips.bimage7`. Other sources use
/// `image_resource_name` (extension replaced). The first candidate present in the index wins.
pub fn resolve_image(ar: &Archive, source: &str) -> Result<String, String> {
    let primary = void_formats::material::image_resource_name(source)
        .ok_or_else(|| format!("{source}: image programs are not supported"))?;
    if source.contains('$') {
        let base = format!("generated/image/{}", source.to_ascii_lowercase());
        for c in [format!("{base}.bimage7"), format!("{base}.tga.bimage7")] {
            if !ar.locate("image", &c).is_empty() {
                return Ok(c);
            }
        }
    }
    Ok(primary)
}

/// A cube map: six faces (+x, -x, +y, -y, +z, -z as stored), each a mip chain.
#[derive(Clone, Debug)]
pub struct CubeData {
    pub name: String,
    pub format: Format,
    pub faces: [Vec<Mip>; 6],
}

/// A 2D texture array (`texture_type` 0 with several "faces", e.g. the 32 caustics
/// frames of `rendering/water/caustics3`): one mip chain per slice.
#[derive(Clone, Debug)]
pub struct ArrayData {
    pub name: String,
    pub format: Format,
    pub slices: Vec<Vec<Mip>>,
}

/// Loads a 2D texture array from the base file's mips (as `load_cube`).
pub fn load_array(ar: &Archive, source: &str) -> Result<ArrayData, String> {
    let name = resolve_image(ar, source)?;
    let img = Bim::parse(&ar.read("image", &name)?).map_err(|e| format!("{name}: {e}"))?;
    if img.header.texture_type != 0 || img.header.faces < 2 {
        return Err(format!(
            "{name}: texture type {} with {} faces is not an array",
            img.header.texture_type, img.header.faces
        ));
    }
    let (format, slices) = layers(&img, &name)?;
    Ok(ArrayData {
        name,
        format,
        slices,
    })
}

/// Mip chains per face/slice of an image whose mips are all in the base file.
fn layers(img: &Bim, name: &str) -> Result<(Format, Vec<Vec<Mip>>), String> {
    if img.streamed_mips() != 0 {
        return Err(format!(
            "{name}: streamed mips of layered images are not supported"
        ));
    }
    let format = img.format.ok_or_else(|| {
        format!(
            "{name}: unknown format pair ({}, {})",
            img.header.format_field, img.header.numeric_field
        )
    })?;
    let mut out: Vec<Vec<(u32, Mip)>> = vec![Vec::new(); img.header.faces as usize];
    for m in &img.mips {
        let f = out
            .get_mut(m.face as usize)
            .ok_or_else(|| format!("{name}: face {} out of range", m.face))?;
        f.push((
            m.level,
            Mip {
                width: m.width,
                height: m.height,
                data: m.data.clone(),
            },
        ));
    }
    let out: Vec<Vec<Mip>> = out
        .into_iter()
        .map(|mut f| {
            f.sort_by_key(|(l, _)| *l);
            f.into_iter().map(|(_, m)| m).collect()
        })
        .collect();
    if out.iter().any(|f| f.is_empty() || f.len() != out[0].len()) {
        return Err(format!("{name}: layers have different mip counts"));
    }
    Ok((format, out))
}

/// Loads a cube map (`texture_type` 2, 6 faces) from the base file's mips only (cube
/// images observed so far carry all their mips in the base file).
pub fn load_cube(ar: &Archive, source: &str) -> Result<CubeData, String> {
    let name = resolve_image(ar, source)?;
    let img = Bim::parse(&ar.read("image", &name)?).map_err(|e| format!("{name}: {e}"))?;
    if img.header.texture_type != 2 || img.header.faces != 6 {
        return Err(format!(
            "{name}: texture type {} with {} faces is not a cube",
            img.header.texture_type, img.header.faces
        ));
    }
    if img.streamed_mips() != 0 {
        return Err(format!("{name}: streamed cube mips are not supported"));
    }
    let format = img.format.ok_or_else(|| {
        format!(
            "{name}: unknown format pair ({}, {})",
            img.header.format_field, img.header.numeric_field
        )
    })?;
    let mut faces: [Vec<(u32, Mip)>; 6] = Default::default();
    for m in &img.mips {
        let f = faces
            .get_mut(m.face as usize)
            .ok_or_else(|| format!("{name}: face {} out of range", m.face))?;
        f.push((
            m.level,
            Mip {
                width: m.width,
                height: m.height,
                data: m.data.clone(),
            },
        ));
    }
    let faces = faces.map(|mut f| {
        f.sort_by_key(|(l, _)| *l);
        f.into_iter().map(|(_, m)| m).collect::<Vec<_>>()
    });
    if faces
        .iter()
        .any(|f| f.is_empty() || f.len() != faces[0].len())
    {
        return Err(format!("{name}: faces have different mip counts"));
    }
    Ok(CubeData {
        name,
        format,
        faces,
    })
}

/// Loads `source` (a material texture path) with mips no larger than `max_size`.
pub fn load(
    ar: &Archive,
    source: &str,
    usage: Usage,
    max_size: u32,
) -> Result<TextureData, String> {
    let name = resolve_image(ar, source)?;
    let img = Bim::parse(&ar.read("image", &name)?).map_err(|e| format!("{name}: {e}"))?;
    if img.header.texture_type != 0 || img.header.faces != 1 {
        return Err(format!(
            "{name}: texture type {} with {} faces is not 2D",
            img.header.texture_type, img.header.faces
        ));
    }
    let format = img.format.ok_or_else(|| {
        format!(
            "{name}: unknown format pair ({}, {})",
            img.header.format_field, img.header.numeric_field
        )
    })?;
    if !format_allowed(usage, format) {
        return Err(format!(
            "{name}: {format:?} not supported for {usage:?} use"
        ));
    }
    let total = img.header.total_mips;
    let first = (0..total)
        .find(|&l| (img.header.width >> l).max(img.header.height >> l) <= max_size)
        .unwrap_or(total.saturating_sub(1));
    let mut mips = Vec::new();
    for level in first..total {
        if level < img.streamed_mips() {
            let data = ar.read("image", &Bim::streamed_mip_name(&name, level))?;
            let mip = img
                .parse_streamed(level, &data)
                .map_err(|e| format!("{name}: {e}"))?;
            mips.push(Mip {
                width: mip.width,
                height: mip.height,
                data: mip.data,
            });
        } else {
            let mip = &img.mips[(level - img.streamed_mips()) as usize];
            mips.push(Mip {
                width: mip.width,
                height: mip.height,
                data: mip.data.clone(),
            });
        }
    }
    if mips.is_empty() {
        return Err(format!("{name}: no mips"));
    }
    Ok(TextureData { name, format, mips })
}
