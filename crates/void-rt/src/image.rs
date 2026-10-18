//! PNG / raw HDR output and the false-colour maps used by the tools.

use std::path::Path;

pub fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<(), String> {
    if rgb.len() != (width * height * 3) as usize {
        return Err(format!(
            "{}: {} bytes for {width}x{height}",
            path.display(),
            rgb.len()
        ));
    }
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(rgb))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Exposure that maps the median luminance of finite, positive pixels to 0.18.
pub fn auto_exposure(rgb: impl Iterator<Item = [f32; 3]>) -> f32 {
    let mut l: Vec<f32> = rgb
        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if l.is_empty() {
        return 1.0;
    }
    let k = l.len() / 2;
    let (_, m, _) = l.select_nth_unstable_by(k, f32::total_cmp);
    0.18 / *m
}

/// Reinhard on luminance-scaled colour, then sRGB.
pub fn tonemap(c: [f32; 3], exposure: f32) -> [u8; 3] {
    c.map(|v| {
        let x = (v * exposure).max(0.0);
        linear_to_srgb8(x / (1.0 + x))
    })
}

/// Distance -> grey (log scale over 0.1..1000 m); misses dark blue, skipped black.
pub fn distance_colour(t: f32) -> [u8; 3] {
    if t == -1.0 {
        return [20, 30, 90];
    }
    if t <= 0.0 {
        return [0, 0, 0];
    }
    let v = ((t.max(0.1).log10() + 1.0) / 4.0).clamp(0.0, 1.0);
    let g = (255.0 * (1.0 - v)) as u8;
    [g, g, g]
}

/// Stable pseudo-random colour per id (misses/skipped black).
pub fn id_colour(id: u32) -> [u8; 3] {
    if id >= 0xffff_fffe {
        return [0, 0, 0];
    }
    let mut h = id.wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    [
        (h & 0xff) as u8 | 0x30,
        ((h >> 8) & 0xff) as u8 | 0x30,
        ((h >> 16) & 0xff) as u8 | 0x30,
    ]
}

/// Raw little-endian RGBA32F (`.bin`, alpha 1) plus a `.tsv` header describing it.
pub fn write_hdr(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[[f32; 3]],
    notes: &str,
) -> Result<(), String> {
    if rgb.len() != (width * height) as usize {
        return Err(format!(
            "{}: {} pixels for {width}x{height}",
            path.display(),
            rgb.len()
        ));
    }
    let mut b = Vec::with_capacity(rgb.len() * 16);
    for c in rgb {
        for v in [c[0], c[1], c[2], 1.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, b).map_err(|e| format!("{}: {e}", path.display()))?;
    let tsv = path.with_extension("tsv");
    std::fs::write(
        &tsv,
        format!(
            "file\twidth\theight\tformat\trow_order\tnotes\n{}\t{width}\t{height}\tRGBA32F little-endian\ttop row first\t{notes}\n",
            path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())
        ),
    )
    .map_err(|e| format!("{}: {e}", tsv.display()))
}

/// Ratio map: grey 1.0, red above, blue below (log2, saturating at 4x); NaN/None black.
pub fn ratio_colour(r: Option<f32>) -> [u8; 3] {
    let Some(r) = r.filter(|r| r.is_finite() && *r > 0.0) else {
        return [0, 0, 0];
    };
    let t = (r.log2() / 2.0).clamp(-1.0, 1.0);
    let g = (160.0 * (1.0 - t.abs())) as u8 + 40;
    if t >= 0.0 {
        [(160.0 + 95.0 * t) as u8, g, g]
    } else {
        [g, g, (160.0 - 95.0 * t) as u8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_maps() {
        assert_eq!(distance_colour(0.0), [0, 0, 0]);
        assert_eq!(distance_colour(-1.0), [20, 30, 90]);
        assert!(distance_colour(1.0)[0] > distance_colour(100.0)[0]);
        assert_eq!(id_colour(0xffff_ffff), [0, 0, 0]);
        assert_ne!(id_colour(1), id_colour(2));
        assert_eq!(tonemap([0.0; 3], 1.0), [0, 0, 0]);
        assert_eq!(ratio_colour(None), [0, 0, 0]);
        assert_eq!(ratio_colour(Some(1.0)), [160, 200, 200]);
        assert!(ratio_colour(Some(4.0))[0] > ratio_colour(Some(0.25))[0]);
        let e = auto_exposure([[0.36f32; 3]; 5].into_iter());
        assert!((e - 0.5).abs() < 1e-4);
    }
}
