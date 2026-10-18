//! `BIM` images (`generated/image/**.bimage7` plus streamed `.bimage7_mipN`).
//!
//! Layout and format enum: docs/formats/bim.md. Little-endian. The base file
//! holds a 62-byte header and the small mips; the largest mips live in
//! separate `_mipN` resources (N = 0 is the largest), each one 20-byte mip
//! record header plus data.

use crate::Error;
use crate::reader::Reader;

pub const MAGIC: [u8; 4] = *b"BIM\x18";
const HEADER_LEN: usize = 0x3E;
const MIP_HEADER_LEN: usize = 20;

/// Pixel formats from the exe's format-name table (bim.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    R8Unorm,
    /// (2, 5): engine format kind 2 / numeric 5 -> DXGI 56 `R16_UNORM`
    /// (`sub_1402C2F50`, O). E.g. `eotw_shore_df` (ocean shore distance field).
    R16Unorm,
    A8Unorm,
    Rg8Unorm,
    Rg8Snorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc6hUf16,
    Bc6hSf16,
    Bc7Unorm,
    Bc7UnormSrgb,
}

impl Format {
    /// (format field, numeric-type field) -> format. `None` for pairs absent
    /// from the exe table, which callers must report rather than guess.
    pub fn from_fields(format: u32, numeric: u32) -> Option<Self> {
        use Format::*;
        Some(match (format, numeric) {
            (0x01, 5) => R8Unorm,
            (0x02, 5) => R16Unorm,
            (0x04, 5) => Rg8Unorm,
            (0x04, 4) => Rg8Snorm,
            (0x0A, 5) => Rgba8Unorm,
            (0x0A, 6) => Rgba8UnormSrgb,
            (0x11, 5) => Bc4Unorm,
            (0x11, 4) => Bc4Snorm,
            (0x12, 5) => Bc5Unorm,
            (0x12, 4) => Bc5Snorm,
            (0x13, 3) => Bc6hUf16,
            (0x13, 2) => Bc6hSf16,
            (0x14, 5) => Bc7Unorm,
            (0x14, 6) => Bc7UnormSrgb,
            _ => return None,
        })
    }

    pub fn is_block_compressed(self) -> bool {
        use Format::*;
        matches!(
            self,
            Bc4Unorm
                | Bc4Snorm
                | Bc5Unorm
                | Bc5Snorm
                | Bc6hUf16
                | Bc6hSf16
                | Bc7Unorm
                | Bc7UnormSrgb
        )
    }

    /// Bytes per 4x4 block, or per pixel for uncompressed formats.
    pub fn unit_size(self) -> usize {
        use Format::*;
        match self {
            R8Unorm | A8Unorm => 1,
            Rg8Unorm | Rg8Snorm | R16Unorm => 2,
            Rgba8Unorm | Rgba8UnormSrgb => 4,
            Bc4Unorm | Bc4Snorm => 8,
            _ => 16,
        }
    }

    pub fn mip_size(self, w: u32, h: u32) -> usize {
        if self.is_block_compressed() {
            w.div_ceil(4) as usize * h.div_ceil(4) as usize * self.unit_size()
        } else {
            w as usize * h as usize * self.unit_size()
        }
    }

    /// (format field, numeric-type field) as stored in the BIM header; inverse of
    /// `from_fields` (`A8Unorm` shares (0x01, 5) with `R8Unorm` and differs only in
    /// header field 0x14).
    pub fn fields(self) -> (u32, u32) {
        use Format::*;
        match self {
            R8Unorm | A8Unorm => (0x01, 5),
            R16Unorm => (0x02, 5),
            Rg8Unorm => (0x04, 5),
            Rg8Snorm => (0x04, 4),
            Rgba8Unorm => (0x0A, 5),
            Rgba8UnormSrgb => (0x0A, 6),
            Bc4Unorm => (0x11, 5),
            Bc4Snorm => (0x11, 4),
            Bc5Unorm => (0x12, 5),
            Bc5Snorm => (0x12, 4),
            Bc6hUf16 => (0x13, 3),
            Bc6hSf16 => (0x13, 2),
            Bc7Unorm => (0x14, 5),
            Bc7UnormSrgb => (0x14, 6),
        }
    }
}

/// Dimension of mip `level` of an axis of size `v`.
pub fn mip_dim(v: u32, level: u32) -> u32 {
    (v >> level.min(31)).max(1)
}

/// Largest mip count the engine creates for `w` x `h` (`sub_1402E9860`, O): halve while
/// both axes exceed 1; block-compressed chains stop before an axis drops below 4.
pub fn max_mips(mut w: u32, mut h: u32, block_compressed: bool) -> u32 {
    let mut n = 1;
    while w > 1 && h > 1 {
        w >>= 1;
        h >>= 1;
        if block_compressed && (w < 4 || h < 4) {
            break;
        }
        n += 1;
    }
    n
}

/// Header flag (u16 at 0x3C): keep every mip in the base file (the engine's writer
/// never splits such images; O, `idBinaryImage_WriteGeneratedFile`).
pub const FLAG_NO_STREAMING: u16 = 0x80;
/// Header flag: the loader keeps the data as a CPU-side image instead of creating a
/// GPU texture (O for the code path, meaning H).
pub const FLAG_CPU_IMAGE: u16 = 0x4000;
/// `r_textureStreamingMaxTextureSizeKB` default: larger streamed mips are never loaded.
pub const MAX_STREAMED_MIP_KB: u64 = 131_072;

impl Header {
    pub fn format(&self) -> Option<Format> {
        Format::from_fields(self.format_field, self.numeric_field)
    }

    /// Flags word at 0x3C (`idImageOpts` +72).
    pub fn flags(&self) -> u16 {
        u16::from_le_bytes([self.raw[0x3C], self.raw[0x3D]])
    }

    /// Faces per mip as the loader counts them: cubes stored with `faces = 1` still
    /// carry 6 (O, `idBinaryImage_LoadFromGeneratedFile`).
    pub fn record_faces(&self) -> u32 {
        if self.texture_type == 2 && self.faces == 1 {
            6
        } else {
            self.faces.max(1)
        }
    }

    /// Records in the base file (O, same function): `mips_in_file x faces`, except 3D
    /// images, which store one record per depth slice of every mip (depth halving per
    /// mip, at least 1).
    pub fn record_count(&self) -> usize {
        if self.texture_type == 1 {
            (0..self.mips_in_file)
                .map(|i| mip_dim(self.depth, i) as usize)
                .sum()
        } else {
            self.mips_in_file as usize * self.record_faces() as usize
        }
    }

    /// Header bytes: `raw` with the modelled fields written over it, so fields whose
    /// meaning is unknown (0x14, 0x38..0x3E) are preserved.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = self.raw;
        b[..4].copy_from_slice(&MAGIC);
        let mut put = |o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(0x04, self.version);
        put(0x08, self.texture_type);
        put(0x0C, self.format_field);
        put(0x10, self.numeric_field);
        put(0x18, self.stored_width);
        put(0x1C, self.stored_height);
        put(0x20, self.width);
        put(0x24, self.height);
        put(0x28, self.depth);
        put(0x2C, self.mips_in_file);
        put(0x30, self.total_mips);
        put(0x34, self.faces);
        b
    }
}

impl Mip {
    /// Record bytes: 20-byte record header plus data. A streamed `_mipN` part is exactly
    /// one such record.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MIP_HEADER_LEN + self.data.len());
        for v in [
            self.level,
            self.face,
            self.width,
            self.height,
            self.data.len() as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.data);
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    /// 0 = 2D, 1 = 3D, 2 = cube.
    pub texture_type: u32,
    pub format_field: u32,
    pub numeric_field: u32,
    pub stored_width: u32,
    pub stored_height: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mips_in_file: u32,
    pub total_mips: u32,
    pub faces: u32,
    /// The whole header as read (magic included). `to_bytes` starts from it, so fields
    /// not modelled here (0x14, 0x38..0x3E) survive a rewrite.
    pub raw: [u8; HEADER_LEN],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mip {
    pub level: u32,
    pub face: u32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub header: Header,
    pub format: Option<Format>,
    /// Mips stored in the base file, largest first.
    pub mips: Vec<Mip>,
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::Invalid { field, detail }
}

fn read_mip(r: &mut Reader) -> Result<Mip, Error> {
    let level = r.u32("mip level")?;
    let face = r.u32("mip face")?;
    let width = r.u32("mip width")?;
    let height = r.u32("mip height")?;
    let size = r.u32("mip size")? as usize;
    Ok(Mip {
        level,
        face,
        width,
        height,
        data: r.bytes(size, "mip data")?.to_vec(),
    })
}

impl Image {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("bim magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let mut raw = [0u8; HEADER_LEN];
        raw.copy_from_slice(data.get(..HEADER_LEN).ok_or(Error::Truncated {
            field: "bim header",
            offset: 0,
            needed: HEADER_LEN as u64,
            available: data.len(),
        })?);
        let f = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().expect("4 bytes"));
        let header = Header {
            version: f(0x04),
            texture_type: f(0x08),
            format_field: f(0x0C),
            numeric_field: f(0x10),
            stored_width: f(0x18),
            stored_height: f(0x1C),
            width: f(0x20),
            height: f(0x24),
            depth: f(0x28),
            mips_in_file: f(0x2C),
            total_mips: f(0x30),
            faces: f(0x34),
            raw,
        };
        r.bytes(HEADER_LEN - 4, "bim header")?;
        let format = Format::from_fields(header.format_field, header.numeric_field);
        if header.mips_in_file > 16 {
            return Err(invalid(
                "bim mips",
                format!("{} mips in file", header.mips_in_file),
            ));
        }
        let records = header.record_count();
        if records * MIP_HEADER_LEN > r.remaining() {
            return Err(invalid(
                "bim mip count",
                format!("{records} records exceed input"),
            ));
        }
        let mut mips = Vec::with_capacity(records);
        for _ in 0..records {
            mips.push(read_mip(&mut r)?);
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        let image = Self {
            header,
            format,
            mips,
        };
        image.check()?;
        Ok(image)
    }

    fn check(&self) -> Result<(), Error> {
        let h = &self.header;
        if h.mips_in_file > h.total_mips || h.total_mips > 16 {
            return Err(invalid(
                "bim mips",
                format!("{} of {}", h.mips_in_file, h.total_mips),
            ));
        }
        if let Some(fmt) = self.format {
            for m in &self.mips {
                let expected = fmt.mip_size(m.width, m.height);
                if m.data.len() != expected {
                    return Err(invalid(
                        "bim mip size",
                        format!(
                            "{}x{} {:?}: {} bytes, expected {expected}",
                            m.width,
                            m.height,
                            fmt,
                            m.data.len()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Number of streamed `_mipN` parts that precede the base file's mips.
    pub fn streamed_mips(&self) -> u32 {
        self.header.total_mips - self.header.mips_in_file
    }

    /// Resource name of streamed part `n` for base resource `name`.
    pub fn streamed_mip_name(name: &str, n: u32) -> String {
        format!("{name}_mip{n}")
    }

    /// Parses one streamed part and checks it against this image.
    pub fn parse_streamed(&self, n: u32, data: &[u8]) -> Result<Mip, Error> {
        let mut r = Reader::new(data);
        let mip = read_mip(&mut r)?;
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        let (w, h) = (self.header.width >> n, self.header.height >> n);
        if mip.level != n || mip.width != w.max(1) || mip.height != h.max(1) {
            return Err(invalid(
                "bim streamed mip",
                format!(
                    "part {n}: level {} {}x{}, expected {}x{}",
                    mip.level, mip.width, mip.height, w, h
                ),
            ));
        }
        if let Some(fmt) = self.format
            && mip.data.len() != fmt.mip_size(mip.width, mip.height)
        {
            return Err(invalid(
                "bim streamed mip size",
                format!("part {n}: {} bytes", mip.data.len()),
            ));
        }
        Ok(mip)
    }

    /// Serializes the base file (header plus records in stored order). Inverse of `parse`
    /// for any parsed image.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.header.to_bytes().to_vec();
        for m in &self.mips {
            out.extend_from_slice(&m.to_bytes());
        }
        out
    }

    /// Builds a fully resident image (every mip in the base file, no streamed parts)
    /// from a DDS, keeping the template's type, format class and unmodelled header
    /// fields (0x14, 0x38..0x3E). The chain is cut where the engine stops creating mips
    /// (`max_mips`); BC chains therefore end at 4 texels.
    pub fn from_dds(dds: &crate::dds::Dds, like: &Header) -> Result<Self, Error> {
        if like.texture_type != 0 || like.faces != 1 || like.depth > 1 {
            return Err(invalid(
                "replacement layout",
                format!(
                    "the original is type {} with {} faces; only single 2D images can be replaced from DDS",
                    like.texture_type, like.faces
                ),
            ));
        }
        if dds.format.fields() != (like.format_field, like.numeric_field) {
            return Err(invalid(
                "replacement format",
                format!(
                    "DDS is {:?}, the original is {}",
                    dds.format,
                    like.format()
                        .map_or_else(|| "an unknown format".to_owned(), |f| format!("{f:?}"))
                ),
            ));
        }
        let bc = dds.format.is_block_compressed();
        if bc && (dds.width < 4 || dds.height < 4) {
            return Err(invalid(
                "replacement size",
                format!("{}x{} block-compressed", dds.width, dds.height),
            ));
        }
        let n = (dds.mips.len() as u32).min(max_mips(dds.width, dds.height, bc));
        let mut header = like.clone();
        header.stored_width = dds.width;
        header.stored_height = dds.height;
        header.width = dds.width;
        header.height = dds.height;
        header.depth = 1;
        header.mips_in_file = n;
        header.total_mips = n;
        let mips = (0..n)
            .map(|i| Mip {
                level: i,
                face: 0,
                width: mip_dim(dds.width, i),
                height: mip_dim(dds.height, i),
                data: dds.mips[i as usize].clone(),
            })
            .collect();
        let format = header.format();
        Ok(Self {
            header,
            format,
            mips,
        })
    }
}

/// Checks that `img` (base file) plus its streamed `parts` (`_mip0`, `_mip1`, ... in
/// order) can replace an image whose header is `original`, under the engine rules of
/// docs/texture-replacement.md §1.3/§3.2: same type, faces, depth, format class and
/// unmodelled header fields; mip chain the engine accepts (records match their levels,
/// count within `max_mips`); streamed levels halve exactly (2D, single face, BC levels
/// multiples of 4, no flag 0x80, each part within the streamer's size limit); CPU-side
/// images keep their size.
pub fn check_replacement(original: &Header, img: &Image, parts: &[Vec<u8>]) -> Result<(), Error> {
    let h = &img.header;
    let fail = |field: &'static str, detail: String| Err(invalid(field, detail));
    if h.version != 19 {
        return Err(Error::UnsupportedVersion(h.version));
    }
    if (h.texture_type, h.faces, h.depth) != (original.texture_type, original.faces, original.depth)
    {
        return fail(
            "replacement layout",
            format!(
                "type/faces/depth {}/{}/{}, original {}/{}/{}",
                h.texture_type,
                h.faces,
                h.depth,
                original.texture_type,
                original.faces,
                original.depth
            ),
        );
    }
    if (h.format_field, h.numeric_field) != (original.format_field, original.numeric_field) {
        return fail(
            "replacement format",
            format!(
                "{:?}, original {:?}",
                (h.format_field, h.numeric_field),
                (original.format_field, original.numeric_field)
            ),
        );
    }
    if h.raw[0x14..0x18] != original.raw[0x14..0x18]
        || h.raw[0x38..0x3E] != original.raw[0x38..0x3E]
    {
        return fail(
            "replacement header fields",
            format!(
                "0x14 {:02x?} / 0x38 {:02x?}, original {:02x?} / {:02x?}",
                &h.raw[0x14..0x18],
                &h.raw[0x38..0x3E],
                &original.raw[0x14..0x18],
                &original.raw[0x38..0x3E]
            ),
        );
    }
    let Some(format) = img.format else {
        return fail("replacement format", "unknown format pair".into());
    };
    let bc = format.is_block_compressed();
    if h.flags() & FLAG_CPU_IMAGE != 0
        && (h.width, h.height, h.total_mips)
            != (original.width, original.height, original.total_mips)
    {
        return fail(
            "replacement size",
            "CPU-side image (flag 0x4000): size and mip count must stay as in the original".into(),
        );
    }
    if h.mips_in_file == 0 || h.width == 0 || h.height == 0 {
        return fail("replacement mips", "empty image".into());
    }
    let limit = max_mips(h.width, h.height, bc);
    if h.total_mips > limit {
        return fail(
            "replacement mips",
            format!(
                "{} mips for {}x{}, the engine creates at most {limit}",
                h.total_mips, h.width, h.height
            ),
        );
    }
    let streamed = h.total_mips - h.mips_in_file;
    if (h.stored_width, h.stored_height)
        != (mip_dim(h.width, streamed), mip_dim(h.height, streamed))
    {
        return fail(
            "replacement size",
            format!(
                "base file starts at {}x{}, expected level {streamed} of {}x{}",
                h.stored_width, h.stored_height, h.width, h.height
            ),
        );
    }
    let faces = h.record_faces();
    if h.texture_type == 1 {
        // 3D: one record per slice (record layout beyond sizes not modelled), so the
        // size must stay as in the original; `parse` already checked the record sizes.
        if (h.width, h.height, h.depth, h.total_mips, h.mips_in_file)
            != (
                original.width,
                original.height,
                original.depth,
                original.total_mips,
                original.mips_in_file,
            )
        {
            return fail(
                "replacement size",
                "3D images must keep the original size and mip count".into(),
            );
        }
    } else {
        let mut seen = vec![false; (h.mips_in_file * faces) as usize];
        for m in &img.mips {
            let ok = m.level < h.mips_in_file
                && m.face < faces
                && m.width == mip_dim(h.stored_width, m.level)
                && m.height == mip_dim(h.stored_height, m.level);
            let slot = (m.level * faces + m.face) as usize;
            if !ok || std::mem::replace(&mut seen[slot], true) {
                return fail(
                    "replacement mip record",
                    format!("level {} face {} {}x{}", m.level, m.face, m.width, m.height),
                );
            }
        }
    }
    if parts.len() != streamed as usize {
        return fail(
            "replacement streamed parts",
            format!("{} parts, header needs {streamed}", parts.len()),
        );
    }
    if streamed > 0 {
        let div = 1u32 << streamed;
        if h.texture_type != 0
            || faces != 1
            || h.flags() & FLAG_NO_STREAMING != 0
            || !h.width.is_multiple_of(div)
            || !h.height.is_multiple_of(div)
            || (bc && !(h.stored_width.is_multiple_of(4) && h.stored_height.is_multiple_of(4)))
        {
            return fail(
                "replacement streaming",
                format!(
                    "{streamed} streamed levels need a 2D single-face streamable image whose size ({}x{}) halves exactly down to a {} base",
                    h.width,
                    h.height,
                    if bc { "multiple-of-4" } else { "whole" }
                ),
            );
        }
        let kb = format.mip_size(h.width, h.height) as u64 / 1024;
        if kb > MAX_STREAMED_MIP_KB {
            return fail(
                "replacement streaming",
                format!(
                    "top mip {kb} KB exceeds r_textureStreamingMaxTextureSizeKB {MAX_STREAMED_MIP_KB}"
                ),
            );
        }
        for (n, p) in parts.iter().enumerate() {
            img.parse_streamed(n as u32, p)?;
        }
    }
    Ok(())
}

/// Decodes one BC4 block channel (8 bytes) into 16 values.
fn bc4_block(b: &[u8], signed: bool) -> [f32; 16] {
    let (e0, e1) = if signed {
        (
            (b[0] as i8 as f32 / 127.0).max(-1.0),
            (b[1] as i8 as f32 / 127.0).max(-1.0),
        )
    } else {
        (b[0] as f32 / 255.0, b[1] as f32 / 255.0)
    };
    let gt = if signed {
        (b[0] as i8) > (b[1] as i8)
    } else {
        b[0] > b[1]
    };
    let mut pal = [0f32; 8];
    pal[0] = e0;
    pal[1] = e1;
    if gt {
        for i in 1..7 {
            pal[i + 1] = ((7 - i) as f32 * e0 + i as f32 * e1) / 7.0;
        }
    } else {
        for i in 1..5 {
            pal[i + 1] = ((5 - i) as f32 * e0 + i as f32 * e1) / 5.0;
        }
        pal[6] = if signed { -1.0 } else { 0.0 };
        pal[7] = 1.0;
    }
    let bits = u64::from_le_bytes([b[2], b[3], b[4], b[5], b[6], b[7], 0, 0]);
    std::array::from_fn(|i| pal[((bits >> (3 * i)) & 7) as usize])
}

/// Decodes BC4/BC5 data to `channels` f32 per pixel (row-major).
pub fn decode_bc45(
    data: &[u8],
    w: u32,
    h: u32,
    channels: usize,
    signed: bool,
) -> Result<Vec<f32>, Error> {
    let bw = w.div_ceil(4) as usize;
    let bh = h.div_ceil(4) as usize;
    let block = 8 * channels;
    if data.len() != bw * bh * block {
        return Err(invalid(
            "bc block data",
            format!("{} bytes for {w}x{h}", data.len()),
        ));
    }
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0f32; w * h * channels];
    for by in 0..bh {
        for bx in 0..bw {
            let base = (by * bw + bx) * block;
            for c in 0..channels {
                let vals = bc4_block(&data[base + 8 * c..base + 8 * c + 8], signed);
                for (i, v) in vals.iter().enumerate() {
                    let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                    if x < w && y < h {
                        out[(y * w + x) * channels + c] = *v;
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(fmt: u32, num: u32, w: u32, total: u32, in_file: u32) -> Vec<u8> {
        let mut h = vec![0u8; HEADER_LEN];
        h[..4].copy_from_slice(&MAGIC);
        let mut put = |o: usize, v: u32| h[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(0x04, 19);
        put(0x0C, fmt);
        put(0x10, num);
        put(0x14, 1);
        put(0x18, w >> (total - in_file));
        put(0x1C, w >> (total - in_file));
        put(0x20, w);
        put(0x24, w);
        put(0x28, 1);
        put(0x2C, in_file);
        put(0x30, total);
        put(0x34, 1);
        h
    }

    fn mip(level: u32, w: u32, size: usize) -> Vec<u8> {
        let mut m = Vec::new();
        for v in [level, 0, w, w, size as u32] {
            m.extend_from_slice(&v.to_le_bytes());
        }
        m.extend(std::iter::repeat_n(0xAB, size));
        m
    }

    #[test]
    fn parses_base_and_streamed() {
        // 16x16 BC7, 3 mips (16, 8, 4); 16 streamed, 8 and 4 in the base file.
        let mut data = header(0x14, 6, 16, 3, 2);
        data.extend(mip(0, 8, 64));
        data.extend(mip(1, 4, 16));
        let img = Image::parse(&data).unwrap();
        assert_eq!(img.format, Some(Format::Bc7UnormSrgb));
        assert_eq!(img.streamed_mips(), 1);
        assert_eq!(img.mips[0].width, 8);
        let part = img.parse_streamed(0, &mip(0, 16, 256)).unwrap();
        assert_eq!(part.data.len(), 256);
        assert!(img.parse_streamed(0, &mip(0, 16, 128)).is_err());
        assert!(img.parse_streamed(1, &mip(0, 16, 256)).is_err());
    }

    #[test]
    fn rejects_bad_sizes_and_trailing() {
        let mut data = header(0x14, 6, 16, 3, 2);
        data.extend(mip(0, 8, 63));
        data.extend(mip(1, 4, 16));
        assert!(Image::parse(&data).is_err());
        let mut data = header(0x14, 6, 16, 3, 2);
        data.extend(mip(0, 8, 64));
        data.extend(mip(1, 4, 16));
        data.push(0);
        assert!(matches!(
            Image::parse(&data),
            Err(Error::TrailingData { .. })
        ));
        let mut data = header(0x14, 6, 16, 3, 2);
        data[0x2C..0x30].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Image::parse(&data).is_err());
    }

    #[test]
    fn unknown_format_pair_is_reported_not_guessed() {
        let mut data = header(0x14, 9, 16, 1, 1);
        data.extend(mip(0, 16, 256));
        assert_eq!(Image::parse(&data).unwrap().format, None);
    }

    #[test]
    fn bc4_endpoints_and_modes() {
        // Unsigned 8-value mode: index 0 = e0, index 1 = e1.
        let mut b = [255u8, 0, 0, 0, 0, 0, 0, 0];
        b[2] = 0b001_000; // pixel0 -> 0, pixel1 -> 1
        let v = bc4_block(&b, false);
        assert_eq!((v[0], v[1]), (1.0, 0.0));
        // Signed: endpoints -127..127 map to -1..1.
        let s = bc4_block(&[127, 0x81, 0b001_000, 0, 0, 0, 0, 0], true);
        assert!((s[0] - 1.0).abs() < 1e-6 && (s[1] + 1.0).abs() < 1e-6);
        // 6-value mode extremes.
        let m = bc4_block(&[0, 255, 0b111_110, 0, 0, 0, 0, 0], false);
        assert_eq!((m[0], m[1]), (0.0, 1.0));
    }

    #[test]
    fn bc5_decodes_two_channels_with_edge_blocks() {
        let block = [255u8, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0];
        let out = decode_bc45(&block, 2, 2, 2, false).unwrap();
        assert_eq!(out, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        assert!(decode_bc45(&block[..8], 2, 2, 2, false).is_err());
    }

    #[test]
    fn writer_round_trips_base_and_parts() {
        let mut data = header(0x14, 6, 16, 3, 2);
        data[0x38..0x3E].copy_from_slice(&[0, 0, 0, 0x80, 0x02, 0x40]);
        data.extend(mip(0, 8, 64));
        data.extend(mip(1, 4, 16));
        let img = Image::parse(&data).unwrap();
        assert_eq!(img.to_bytes(), data);
        assert_eq!(img.header.flags(), 0x4002);
        let part = mip(0, 16, 256);
        assert_eq!(img.parse_streamed(0, &part).unwrap().to_bytes(), part);
        assert!(check_replacement(&img.header, &img, std::slice::from_ref(&part)).is_ok());
        assert!(check_replacement(&img.header, &img, &[]).is_err());
    }

    #[test]
    fn engine_mip_limits() {
        assert_eq!(max_mips(16, 16, true), 3);
        assert_eq!(max_mips(16, 16, false), 5);
        assert_eq!(max_mips(2048, 512, true), 8);
        assert_eq!(max_mips(1, 64, false), 1);
        assert_eq!(mip_dim(5, 1), 2);
        assert_eq!(mip_dim(5, 9), 1);
    }

    fn original_bc7() -> Header {
        let mut data = header(0x14, 6, 16, 3, 2);
        data[0x38..0x3E].copy_from_slice(&[0, 0, 0, 0x80, 0, 0]);
        data.extend(mip(0, 8, 64));
        data.extend(mip(1, 4, 16));
        Image::parse(&data).unwrap().header
    }

    fn dds(format: Format, w: u32, h: u32, n: u32) -> crate::dds::Dds {
        crate::dds::Dds {
            format,
            width: w,
            height: h,
            mips: (0..n)
                .map(|i| vec![i as u8 + 1; format.mip_size(mip_dim(w, i), mip_dim(h, i))])
                .collect(),
        }
    }

    #[test]
    fn dds_to_bim_round_trip() {
        let orig = original_bc7();
        // 64x32 with a full chain down to 1x1: the BIM keeps 64..8x4 (engine limit).
        let d = dds(Format::Bc7UnormSrgb, 64, 32, 7);
        let img = Image::from_dds(&d, &orig).unwrap();
        assert_eq!((img.header.total_mips, img.header.mips_in_file), (4, 4));
        let bytes = img.to_bytes();
        let back = Image::parse(&bytes).unwrap();
        assert_eq!(back.to_bytes(), bytes);
        assert_eq!(back.header.raw[0x38..0x3E], orig.raw[0x38..0x3E]);
        assert_eq!((back.header.width, back.header.stored_height), (64, 32));
        for (i, m) in back.mips.iter().enumerate() {
            assert_eq!(m.data, d.mips[i]);
        }
        check_replacement(&orig, &back, &[]).unwrap();
    }

    #[test]
    fn rejects_incompatible_replacements() {
        let orig = original_bc7();
        // Other format class (BC7 UNORM instead of sRGB).
        assert!(Image::from_dds(&dds(Format::Bc7Unorm, 16, 16, 3), &orig).is_err());
        // Too small for blocks.
        assert!(Image::from_dds(&dds(Format::Bc7UnormSrgb, 2, 2, 1), &orig).is_err());
        // Streamed chain that does not halve exactly (24 -> 12 -> 6).
        let mut img = Image::from_dds(&dds(Format::Bc7UnormSrgb, 24, 24, 1), &orig).unwrap();
        img.header.width = 48;
        img.header.height = 48;
        img.header.total_mips = 2;
        let part = Mip {
            level: 0,
            face: 0,
            width: 48,
            height: 48,
            data: vec![0; Format::Bc7UnormSrgb.mip_size(48, 48)],
        }
        .to_bytes();
        assert!(check_replacement(&orig, &img, std::slice::from_ref(&part)).is_ok());
        img.header.width = 40;
        assert!(check_replacement(&orig, &img, &[part]).is_err());
        // CPU-side images keep their size.
        let mut cpu = orig.clone();
        cpu.raw[0x3C..0x3E].copy_from_slice(&FLAG_CPU_IMAGE.to_le_bytes());
        let big = Image::from_dds(&dds(Format::Bc7UnormSrgb, 32, 32, 4), &cpu).unwrap();
        assert!(check_replacement(&cpu, &big, &[]).is_err());
        // Changed unmodelled header field.
        let mut other = Image::from_dds(&dds(Format::Bc7UnormSrgb, 16, 16, 3), &orig).unwrap();
        other.header.raw[0x14] = 2;
        assert!(check_replacement(&orig, &other, &[]).is_err());
    }
}
