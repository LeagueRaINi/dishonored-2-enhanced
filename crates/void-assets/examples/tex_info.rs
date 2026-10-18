//! Prints format, top-mip size and mip count of textures.
//! usage: tex_info <game root> <image source>...
use std::path::Path;
use void_assets::texture::{self, Usage};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ar = void_assets::Archive::open(Path::new(&a[1]))?;
    for src in &a[2..] {
        match texture::load(&ar, src, Usage::Color, 4096) {
            Ok(t) => println!(
                "{src}: {:?} {}x{} mips {}",
                t.format,
                t.mips[0].width,
                t.mips[0].height,
                t.mips.len()
            ),
            Err(e) => println!("{src}: {e}"),
        }
    }
    Ok(())
}
