//! Decoder check of the photo-mode material textures: for every BC4/BC5/BC7 map of a scene,
//! each stored mip is compared with the 2x2 box average of the level above it (per 4x4
//! block means). A mis-decoded level (wrong streamed part, block order, format or channel)
//! breaks the chain; a correct chain differs only by the encoder's filtering. CPU only.
//!
//! cargo run --release -p void-rt --example mip_check -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 --tex-size 2048

use std::path::PathBuf;

use clap::Parser;
use void_assets::texture::{Mip, TextureData};
use void_formats::bim::Format;
use void_rt::ingame::{SceneOptions, SceneSource, load_scene};
use void_rt::snapshot::Snapshot;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    game: PathBuf,
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, default_value_t = 2048)]
    tex_size: u32,
    /// Mean absolute difference (format range, per channel) reported as broken.
    #[arg(long, default_value_t = 0.08)]
    threshold: f32,
}

/// Mean of every 4x4 block of a BC mip (row-major blocks), RGBA in the format's range.
fn block_means(format: Format, m: &Mip) -> Option<(usize, usize, Vec<[f32; 4]>)> {
    let (bw, bh) = (m.width.div_ceil(4) as usize, m.height.div_ceil(4) as usize);
    let bytes = match format {
        Format::Bc4Unorm | Format::Bc4Snorm => 8,
        Format::Bc5Unorm | Format::Bc5Snorm | Format::Bc7Unorm | Format::Bc7UnormSrgb => 16,
        _ => return None,
    };
    let mut out = Vec::with_capacity(bw * bh);
    for b in m.data.chunks_exact(bytes) {
        let mut s = [0f32; 4];
        match format {
            Format::Bc4Unorm | Format::Bc4Snorm => {
                let mut r = [0f32; 16];
                bcdec_rs::bc4_float(b, &mut r, 4, format == Format::Bc4Snorm);
                s[0] = r.iter().sum::<f32>() / 16.0;
            }
            Format::Bc5Unorm | Format::Bc5Snorm => {
                let mut rg = [0f32; 32];
                bcdec_rs::bc5_float(b, &mut rg, 8, format == Format::Bc5Snorm);
                for k in 0..16 {
                    s[0] += rg[2 * k] / 16.0;
                    s[1] += rg[2 * k + 1] / 16.0;
                }
            }
            _ => {
                let mut px = [0u8; 64];
                bcdec_rs::bc7(b, &mut px, 16);
                for k in 0..16 {
                    for c in 0..4 {
                        s[c] += f32::from(px[4 * k + c]) / 255.0 / 16.0;
                    }
                }
            }
        }
        out.push(s);
    }
    (out.len() == bw * bh).then_some((bw, bh, out))
}

/// Mean absolute difference between level `k + 1` and the 2x2 box average of level `k`
/// (block means), over the channels the format has.
fn chain_error(t: &TextureData, k: usize) -> Option<f32> {
    let (aw, ah, a) = block_means(t.format, &t.mips[k])?;
    let (bw, bh, b) = block_means(t.format, &t.mips[k + 1])?;
    if aw < 2 || ah < 2 || bw * 2 != aw || bh * 2 != ah {
        return None;
    }
    let ch = match t.format {
        Format::Bc4Unorm | Format::Bc4Snorm => 1,
        Format::Bc5Unorm | Format::Bc5Snorm => 2,
        _ => 4,
    };
    let mut sum = 0f64;
    for y in 0..bh {
        for x in 0..bw {
            let q = |dx: usize, dy: usize| a[(2 * y + dy) * aw + 2 * x + dx];
            for (c, &v) in b[y * bw + x].iter().enumerate().take(ch) {
                let avg = (q(0, 0)[c] + q(1, 0)[c] + q(0, 1)[c] + q(1, 1)[c]) / 4.0;
                sum += f64::from((avg - v).abs());
            }
        }
    }
    Some((sum / (bw * bh * ch) as f64) as f32)
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let snap = Snapshot::load(&args.snapshot)?;
    let src = SceneSource {
        game_dir: args.game.clone(),
        map: args.map.clone(),
        entities: snap.entities.clone(),
        lights: Vec::new(),
        sun: None,
        env: None,
    };
    let opts = SceneOptions {
        texture_size: args.tex_size,
        decals: true,
        ..SceneOptions::default()
    };
    let cpu = load_scene(&src, &opts)?;
    let textures = &cpu.scene.textures;
    let mut worst: Vec<(f32, usize, &TextureData)> = Vec::new();
    let mut per_level = vec![(0f64, 0usize); 16];
    for t in textures {
        for k in 0..t.mips.len().saturating_sub(1) {
            if let Some(e) = chain_error(t, k) {
                let from_top = (t.mips[0].width.max(t.mips[0].height) >> k).max(1);
                let lvl = from_top.ilog2() as usize;
                per_level[lvl].0 += f64::from(e);
                per_level[lvl].1 += 1;
                worst.push((e, k, t));
            }
        }
    }
    println!(
        "{} textures (<= {} texels); mean |level k+1 - box(level k)| by level-k edge:",
        textures.len(),
        args.tex_size
    );
    for (l, (s, n)) in per_level.iter().enumerate().rev() {
        if *n > 0 {
            println!(
                "  {:5} texels: {:.4} over {n} pairs",
                1u32 << l,
                s / *n as f64
            );
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    let broken = worst.iter().filter(|w| w.0 > args.threshold).count();
    println!("pairs above {}: {broken}; worst 25:", args.threshold);
    for (e, k, t) in worst.iter().take(25) {
        println!(
            "  {e:.3} level {k} ({}x{}) -> {} of {} mips, {:?}: {}",
            t.mips[*k].width,
            t.mips[*k].height,
            k + 1,
            t.mips.len(),
            t.format,
            t.name
        );
    }
    let short: Vec<&TextureData> = textures
        .iter()
        .filter(|t| t.mips.last().is_some_and(|m| m.width.min(m.height) > 4))
        .collect();
    println!(
        "chains ending above one block (kept as stored, not completed): {} ({} single-mip, {} named $nomips):",
        short.len(),
        short.iter().filter(|t| t.mips.len() == 1).count(),
        short.iter().filter(|t| t.name.contains("$nomips")).count()
    );
    for t in &short {
        let m = t.mips.last().map_or((0, 0), |m| (m.width, m.height));
        println!(
            "  {} mips, last {}x{}, {:?}: {}",
            t.mips.len(),
            m.0,
            m.1,
            t.format,
            t.name
        );
    }
    Ok(())
}
