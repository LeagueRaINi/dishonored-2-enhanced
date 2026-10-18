//! `void-reflect`: extract class/member reflection tables from Dishonored2.exe.
//!
//! Usage: `void-reflect --exe <Dishonored2.exe> --out <dir> [--init-table 0x141A921C0]`
//! Writes `classes.tsv` and `members.tsv` (offline; the executable is only read).

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::Parser;
use goblin::pe::PE;
use sha2::{Digest, Sha256};
use void_reflect::{Evaluator, Image, Section, read_types};

#[derive(Parser)]
#[command(about = "Extract the engine's reflection metadata from Dishonored2.exe")]
struct Cli {
    #[arg(long)]
    exe: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// A pointer inside the static-initializer function table (O: 0x141A921C0 holds
    /// the `idRenderModel` member initializer); the contiguous run around it is used.
    #[arg(long, default_value = "0x141A921C0")]
    init_table: String,
    /// Additional initializer functions to evaluate (e.g. the class registry 0x141413800).
    #[arg(long, value_delimiter = ',', default_value = "0x141413800")]
    extra: Vec<String>,
}

fn hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s.trim_start_matches("0x").trim_start_matches("0X"), 16)
        .map_err(|e| format!("{s}: {e}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let bytes = std::fs::read(&cli.exe)?;
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let pe = PE::parse(&bytes)?;
    let base = pe.image_base;
    let mut img = Image {
        base,
        sections: Vec::new(),
    };
    for s in &pe.sections {
        let start = s.pointer_to_raw_data as usize;
        let len = s.size_of_raw_data as usize;
        let data = bytes
            .get(start..start + len)
            .ok_or("section outside file")?
            .to_vec();
        img.sections.push(Section {
            name: String::from_utf8_lossy(&s.name)
                .trim_end_matches('\0')
                .to_owned(),
            va: base + u64::from(s.virtual_address),
            vsize: u64::from(s.virtual_size),
            data,
            executable: s.characteristics & 0x2000_0000 != 0,
        });
    }
    // Initializer table: contiguous run of code pointers (zeros allowed) around the seed.
    let seed = hex(&cli.init_table)?;
    let ok = |va: u64| img.u64_at(va).is_some_and(|p| p == 0 || img.is_code(p));
    let (mut lo, mut hi) = (seed, seed);
    while ok(lo - 8) {
        lo -= 8;
    }
    while ok(hi + 8) {
        hi += 8;
    }
    let mut funcs: Vec<u64> = (lo..=hi)
        .step_by(8)
        .filter_map(|va| img.u64_at(va))
        .filter(|&p| p != 0)
        .collect();
    for e in &cli.extra {
        funcs.push(hex(e)?);
    }
    let mut ev = Evaluator::default();
    for &f in &funcs {
        ev.run(&img, f);
    }
    let types = read_types(&ev, &img);
    std::fs::create_dir_all(&cli.out)?;
    let mut classes = format!(
        "# exe sha256 {}; initializer table {lo:#x}..={hi:#x} ({} functions + {} extra); \
         {} writes, {} unhandled stores\nclass\tbase\tsize\trecord\tmembers\n",
        sha,
        funcs.len() - cli.extra.len(),
        cli.extra.len(),
        ev.writes.len(),
        ev.unhandled
    );
    let mut members = String::from("class\toffset\tsize\ttype\tmodifier\tname\tbit\n");
    for c in &types {
        writeln!(
            classes,
            "{}\t{}\t{}\t{:#x}\t{}",
            c.name,
            c.base,
            c.size,
            c.record,
            c.members.len()
        )?;
        for m in &c.members {
            writeln!(
                members,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                c.name,
                m.bit.map_or(m.offset, |b| b.0),
                m.size,
                m.type_name,
                m.modifier,
                m.name,
                m.bit.map_or(String::new(), |b| b.1.to_string())
            )?;
        }
    }
    std::fs::write(cli.out.join("classes.tsv"), classes)?;
    std::fs::write(cli.out.join("members.tsv"), members)?;
    let nmem: usize = types.iter().map(|c| c.members.len()).sum();
    println!(
        "exe sha256 {sha}\ninitializers {} (table {lo:#x}..={hi:#x}) + {} extra; writes {}; unhandled stores {}\nclasses with members {}; members {nmem}",
        funcs.len() - cli.extra.len(),
        cli.extra.len(),
        ev.writes.len(),
        ev.unhandled,
        types.len()
    );
    Ok(())
}
