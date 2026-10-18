//! DirectDraw Surface files: input and output of texture modding
//! (docs/texture-replacement.md).
//!
//! Only single 2D textures with a mip chain are accepted, in the formats the engine's
//! BIM loader knows (`bim::Format`). The reader takes both the DX10 extension header and
//! the legacy FourCC/bit-mask descriptions that texconv and Compressonator still write
//! for BC4/BC5/RGBA8/L8/A8. BC1-BC3 are rejected because they are absent from the engine's
//! image format table (bim.md); BC6H/BC7 need the DX10 header. The writer always writes
//! the DX10 header.

use crate::Error;
use crate::bim::{Format, mip_dim};
use crate::reader::Reader;

pub const MAGIC: [u8; 4] = *b"DDS ";
const HEADER_SIZE: u32 = 124;
const PIXEL_FORMAT_SIZE: u32 = 32;

// DDS_HEADER.dwFlags
const DDSD_CAPS: u32 = 0x1;
const DDSD_HEIGHT: u32 = 0x2;
const DDSD_WIDTH: u32 = 0x4;
const DDSD_PIXELFORMAT: u32 = 0x1000;
const DDSD_MIPMAPCOUNT: u32 = 0x2_0000;
const DDSD_LINEARSIZE: u32 = 0x8_0000;
const DDSD_DEPTH: u32 = 0x80_0000;
// DDS_PIXELFORMAT.dwFlags
const DDPF_ALPHA: u32 = 0x2;
const DDPF_FOURCC: u32 = 0x4;
const DDPF_RGB: u32 = 0x40;
const DDPF_LUMINANCE: u32 = 0x2_0000;
// dwCaps / dwCaps2
const DDSCAPS_COMPLEX: u32 = 0x8;
const DDSCAPS_TEXTURE: u32 = 0x1000;
const DDSCAPS_MIPMAP: u32 = 0x40_0000;
const DDSCAPS2_CUBEMAP: u32 = 0x200;
const DDSCAPS2_VOLUME: u32 = 0x20_0000;
// DDS_HEADER_DXT10
const DIMENSION_TEXTURE2D: u32 = 3;
const MISC_TEXTURECUBE: u32 = 0x4;

/// A single 2D texture: mips largest first, each `format.mip_size(w >> i, h >> i)` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dds {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub mips: Vec<Vec<u8>>,
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::Invalid {
        field,
        detail: detail.into(),
    }
}

/// DXGI_FORMAT value of `format`.
pub fn dxgi_format(format: Format) -> u32 {
    use Format::*;
    match format {
        R8Unorm => 61,
        R16Unorm => 56,
        A8Unorm => 65,
        Rg8Unorm => 49,
        Rg8Snorm => 51,
        Rgba8Unorm => 28,
        Rgba8UnormSrgb => 29,
        Bc4Unorm => 80,
        Bc4Snorm => 81,
        Bc5Unorm => 83,
        Bc5Snorm => 84,
        Bc6hUf16 => 95,
        Bc6hSf16 => 96,
        Bc7Unorm => 98,
        Bc7UnormSrgb => 99,
    }
}

fn from_dxgi(v: u32) -> Result<Format, Error> {
    use Format::*;
    Ok(match v {
        61 => R8Unorm,
        56 => R16Unorm,
        65 => A8Unorm,
        49 => Rg8Unorm,
        51 => Rg8Snorm,
        28 => Rgba8Unorm,
        29 => Rgba8UnormSrgb,
        80 => Bc4Unorm,
        81 => Bc4Snorm,
        83 => Bc5Unorm,
        84 => Bc5Snorm,
        95 => Bc6hUf16,
        96 => Bc6hSf16,
        98 => Bc7Unorm,
        99 => Bc7UnormSrgb,
        70..=78 => {
            return Err(invalid(
                "dds format",
                format!("DXGI {v} (BC1-BC3) is not in the engine's image format table"),
            ));
        }
        _ => {
            return Err(invalid(
                "dds format",
                format!("DXGI {v} is not supported (typeless or not a BIM format)"),
            ));
        }
    })
}

fn from_fourcc(cc: [u8; 4]) -> Result<Format, Error> {
    Ok(match &cc {
        b"ATI1" | b"BC4U" => Format::Bc4Unorm,
        b"BC4S" => Format::Bc4Snorm,
        b"ATI2" | b"BC5U" => Format::Bc5Unorm,
        b"BC5S" => Format::Bc5Snorm,
        b"DXT1" | b"DXT2" | b"DXT3" | b"DXT4" | b"DXT5" => {
            return Err(invalid(
                "dds format",
                format!(
                    "{} (BC1-BC3) is not in the engine's image format table",
                    String::from_utf8_lossy(&cc)
                ),
            ));
        }
        _ => {
            return Err(invalid(
                "dds format",
                format!("FourCC {cc:02x?} is not supported (use the DX10 header)"),
            ));
        }
    })
}

/// Legacy bit-mask formats: RGBA8 in R,G,B,A byte order, 8-bit luminance, 8-bit alpha.
fn from_masks(flags: u32, bits: u32, masks: [u32; 4]) -> Result<Format, Error> {
    let rgba = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
    if flags & DDPF_RGB != 0 && bits == 32 && masks == rgba {
        return Ok(Format::Rgba8Unorm);
    }
    if flags & DDPF_LUMINANCE != 0 && bits == 8 && masks[0] == 0xFF {
        return Ok(Format::R8Unorm);
    }
    if flags & DDPF_ALPHA != 0 && bits == 8 && masks[3] == 0xFF {
        return Ok(Format::A8Unorm);
    }
    Err(invalid(
        "dds format",
        format!(
            "legacy pixel format flags {flags:#x} bits {bits} masks {masks:08x?} (only RGBA8 in R,G,B,A order, L8 and A8; use the DX10 header otherwise)"
        ),
    ))
}

impl Dds {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader::new(data);
        let magic: [u8; 4] = r.array("dds magic")?;
        if magic != MAGIC {
            return Err(Error::BadMagic {
                expected: MAGIC,
                found: magic,
            });
        }
        let size = r.u32("dds header size")?;
        if size != HEADER_SIZE {
            return Err(invalid("dds header size", size.to_string()));
        }
        let flags = r.u32("dds flags")?;
        let height = r.u32("dds height")?;
        let width = r.u32("dds width")?;
        let _pitch = r.u32("dds pitch")?;
        let depth = r.u32("dds depth")?;
        let mip_count = r.u32("dds mip count")?;
        r.bytes(44, "dds reserved")?;
        let pf_size = r.u32("dds pixel format size")?;
        if pf_size != PIXEL_FORMAT_SIZE {
            return Err(invalid("dds pixel format size", pf_size.to_string()));
        }
        let pf_flags = r.u32("dds pixel format flags")?;
        let fourcc: [u8; 4] = r.array("dds fourcc")?;
        let bits = r.u32("dds bit count")?;
        let masks = [
            r.u32("dds red mask")?,
            r.u32("dds green mask")?,
            r.u32("dds blue mask")?,
            r.u32("dds alpha mask")?,
        ];
        let _caps = r.u32("dds caps")?;
        let caps2 = r.u32("dds caps2")?;
        r.bytes(12, "dds caps3/caps4/reserved2")?;
        if caps2 & (DDSCAPS2_CUBEMAP | DDSCAPS2_VOLUME) != 0
            || (flags & DDSD_DEPTH != 0 && depth > 1)
        {
            return Err(invalid(
                "dds layout",
                "cube maps and volume textures are not supported",
            ));
        }
        let format = if pf_flags & DDPF_FOURCC != 0 && &fourcc == b"DX10" {
            let dxgi = r.u32("dx10 format")?;
            let dimension = r.u32("dx10 dimension")?;
            let misc = r.u32("dx10 misc flags")?;
            let array = r.u32("dx10 array size")?;
            let _misc2 = r.u32("dx10 misc flags 2")?;
            if dimension != DIMENSION_TEXTURE2D || misc & MISC_TEXTURECUBE != 0 || array != 1 {
                return Err(invalid(
                    "dds layout",
                    format!(
                        "dimension {dimension}, misc {misc:#x}, array size {array}: only a single 2D texture is supported"
                    ),
                ));
            }
            from_dxgi(dxgi)?
        } else if pf_flags & DDPF_FOURCC != 0 {
            from_fourcc(fourcc)?
        } else {
            from_masks(pf_flags, bits, masks)?
        };
        if width == 0 || height == 0 {
            return Err(invalid("dds size", format!("{width}x{height}")));
        }
        let count = if flags & DDSD_MIPMAPCOUNT != 0 {
            mip_count.max(1)
        } else {
            1
        };
        if count > 32 || (width.max(height) >> (count - 1)) == 0 {
            return Err(invalid(
                "dds mip count",
                format!("{count} mips for {width}x{height}"),
            ));
        }
        let mut mips = Vec::with_capacity(count as usize);
        for i in 0..count {
            let len = format.mip_size(mip_dim(width, i), mip_dim(height, i));
            mips.push(r.bytes(len, "dds mip data")?.to_vec());
        }
        if r.remaining() != 0 {
            return Err(Error::TrailingData {
                offset: r.pos(),
                len: r.remaining(),
            });
        }
        Ok(Self {
            format,
            width,
            height,
            mips,
        })
    }

    /// Serializes with the DX10 header. Mip sizes must match the format and dims.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        for (i, m) in self.mips.iter().enumerate() {
            let i = i as u32;
            let want = self
                .format
                .mip_size(mip_dim(self.width, i), mip_dim(self.height, i));
            if m.len() != want {
                return Err(invalid(
                    "dds mip data",
                    format!("mip {i}: {} bytes, expected {want}", m.len()),
                ));
            }
        }
        if self.mips.is_empty() {
            return Err(invalid("dds mip count", "no mips"));
        }
        let mut out = Vec::with_capacity(148 + self.mips.iter().map(Vec::len).sum::<usize>());
        let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
        let mipmapped = self.mips.len() > 1;
        put(u32::from_le_bytes(MAGIC));
        put(HEADER_SIZE);
        put(DDSD_CAPS
            | DDSD_HEIGHT
            | DDSD_WIDTH
            | DDSD_PIXELFORMAT
            | DDSD_MIPMAPCOUNT
            | DDSD_LINEARSIZE);
        put(self.height);
        put(self.width);
        put(self.mips[0].len() as u32);
        put(0); // depth
        put(self.mips.len() as u32);
        for _ in 0..11 {
            put(0);
        }
        put(PIXEL_FORMAT_SIZE);
        put(DDPF_FOURCC);
        put(u32::from_le_bytes(*b"DX10"));
        for _ in 0..5 {
            put(0);
        }
        put(DDSCAPS_TEXTURE
            | if mipmapped {
                DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
            } else {
                0
            });
        for _ in 0..4 {
            put(0); // caps2, caps3, caps4, reserved2
        }
        put(dxgi_format(self.format));
        put(DIMENSION_TEXTURE2D);
        put(0); // misc
        put(1); // array size
        put(0); // misc flags 2 (alpha mode unknown)
        for m in &self.mips {
            out.extend_from_slice(m);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const DDPF_ALPHAPIXELS: u32 = 0x1;

    fn bc5_chain(w: u32, h: u32, n: u32) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| vec![i as u8; Format::Bc5Snorm.mip_size(mip_dim(w, i), mip_dim(h, i))])
            .collect()
    }

    /// Legacy header with a FourCC (no DX10 extension).
    fn legacy(cc: &[u8; 4], w: u32, h: u32, mips: &[Vec<u8>]) -> Vec<u8> {
        // Any format with these mip sizes; only the header's FourCC matters.
        let format = if mips[0].len() == Format::Bc4Unorm.mip_size(w, h) {
            Format::Bc4Unorm
        } else {
            Format::Bc5Unorm
        };
        let dds = Dds {
            format,
            width: w,
            height: h,
            mips: mips.to_vec(),
        };
        let mut b = dds.to_bytes().unwrap();
        // Replace "DX10" with the FourCC and drop the 20-byte DX10 header.
        b[84..88].copy_from_slice(cc);
        b.drain(128..148);
        b
    }

    #[test]
    fn dx10_round_trip() {
        let dds = Dds {
            format: Format::Bc7UnormSrgb,
            width: 16,
            height: 8,
            mips: vec![
                vec![1; 128],
                vec![2; 32],
                vec![3; 16],
                vec![4; 16],
                vec![5; 16],
            ],
        };
        let bytes = dds.to_bytes().unwrap();
        assert_eq!(Dds::parse(&bytes).unwrap(), dds);
    }

    #[test]
    fn legacy_fourcc_bc4_bc5() {
        let mips = bc5_chain(8, 8, 2);
        let d = Dds::parse(&legacy(b"ATI2", 8, 8, &mips)).unwrap();
        assert_eq!((d.format, d.mips.len()), (Format::Bc5Unorm, 2));
        let d = Dds::parse(&legacy(b"BC5S", 8, 8, &mips)).unwrap();
        assert_eq!(d.format, Format::Bc5Snorm);
        let half: Vec<Vec<u8>> = mips.iter().map(|m| m[..m.len() / 2].to_vec()).collect();
        assert_eq!(
            Dds::parse(&legacy(b"ATI1", 8, 8, &half)).unwrap().format,
            Format::Bc4Unorm
        );
        assert_eq!(
            Dds::parse(&legacy(b"BC4S", 8, 8, &half)).unwrap().format,
            Format::Bc4Snorm
        );
    }

    #[test]
    fn rejects_bc1_to_bc3_and_unknown() {
        let mips = bc5_chain(8, 8, 1);
        for cc in [b"DXT1", b"DXT5"] {
            let e = Dds::parse(&legacy(cc, 8, 8, &mips)).unwrap_err();
            assert!(e.to_string().contains("BC1-BC3"), "{e}");
        }
        let mut b = Dds {
            format: Format::Bc7Unorm,
            width: 4,
            height: 4,
            mips: vec![vec![0; 16]],
        }
        .to_bytes()
        .unwrap();
        b[128..132].copy_from_slice(&71u32.to_le_bytes()); // BC1_UNORM
        assert!(Dds::parse(&b).unwrap_err().to_string().contains("BC1-BC3"));
        b[128..132].copy_from_slice(&97u32.to_le_bytes()); // BC7_TYPELESS
        assert!(Dds::parse(&b).is_err());
    }

    #[test]
    fn rejects_truncated_trailing_and_layouts() {
        let dds = Dds {
            format: Format::Bc4Unorm,
            width: 8,
            height: 8,
            mips: vec![vec![0; 32], vec![0; 8]],
        };
        let b = dds.to_bytes().unwrap();
        for cut in [0, 3, 60, 127, 140, b.len() - 1] {
            assert!(
                matches!(
                    Dds::parse(&b[..cut]),
                    Err(Error::Truncated { .. } | Error::BadMagic { .. })
                ),
                "cut {cut}"
            );
        }
        let mut t = b.clone();
        t.push(0);
        assert!(matches!(Dds::parse(&t), Err(Error::TrailingData { .. })));
        let mut cube = b.clone();
        cube[112..116].copy_from_slice(&DDSCAPS2_CUBEMAP.to_le_bytes());
        assert!(Dds::parse(&cube).is_err());
        let mut array = b;
        array[140..144].copy_from_slice(&6u32.to_le_bytes());
        assert!(Dds::parse(&array).is_err());
    }

    #[test]
    fn legacy_rgba8_masks() {
        let dds = Dds {
            format: Format::Rgba8Unorm,
            width: 2,
            height: 1,
            mips: vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
        };
        let mut b = dds.to_bytes().unwrap();
        b.drain(128..148);
        b[80..84].copy_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes());
        b[84..88].copy_from_slice(&[0; 4]);
        b[88..92].copy_from_slice(&32u32.to_le_bytes());
        for (i, m) in [0xFFu32, 0xFF00, 0xFF_0000, 0xFF00_0000].iter().enumerate() {
            b[92 + 4 * i..96 + 4 * i].copy_from_slice(&m.to_le_bytes());
        }
        assert_eq!(Dds::parse(&b).unwrap(), dds);
        // BGRA order is refused rather than swizzled silently.
        b[92..96].copy_from_slice(&0xFF_0000u32.to_le_bytes());
        b[100..104].copy_from_slice(&0xFFu32.to_le_bytes());
        assert!(Dds::parse(&b).is_err());
    }
}
