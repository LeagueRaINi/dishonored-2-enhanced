//! Inspection of an installed game's resources.
//!
//! The game directory is always passed explicitly and never written. Output goes only
//! to the `--out` file/directory of `extract`/`texmod export`, the `--mods`
//! directory of `texmod import`/`texmod test-pattern`, and the `--obj` file of
//! `collision`/`md6`.

mod md6;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use sha2::{Digest, Sha256};
use void_res::path::safe_relative_path;
use void_res::{Entry, Installation, Lookup, MapResources};

#[derive(Parser)]
#[command(about = "Inspect Dishonored 2 resource archives (read-only)")]
struct Cli {
    /// Game installation root (the directory containing `base/`).
    #[arg(long, global = true, env = "VOID_GAME_DIR")]
    game: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarize master.index, index identities, and entry statistics.
    Info,
    /// List entries as tab-separated values.
    List {
        #[command(flatten)]
        filter: Filter,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Extract matching entries into an explicit output directory.
    Extract {
        #[command(flatten)]
        filter: Filter,
        #[arg(long)]
        out: PathBuf,
        /// Refuse to run if more than this many entries match.
        #[arg(long, default_value_t = 50)]
        max: usize,
    },
    /// Read and decode matching entries, reporting any failure.
    /// Resolve and read every resource listed by a `.mapresources` file.
    Deps {
        /// Group whose index the ids refer to (0 = game1).
        #[arg(long)]
        group: usize,
        /// Name of the `.mapresources` entry, e.g.
        /// `generated/buildgame/campaign/tutorial/tutorial_p.mapresources`.
        #[arg(long)]
        name: String,
        /// Print every resolved entry as TSV instead of a summary only.
        #[arg(long)]
        list: bool,
    },
    /// Resolve every material and texture referenced by one world model.
    Materials {
        #[arg(long, default_value_t = 0)]
        group: usize,
        #[arg(long)]
        bwm: String,
        /// Also parse every streamed mip part (reads all texture data).
        #[arg(long)]
        mips: bool,
    },
    /// Parse static entity models (`.bmodel`) matching the filter and summarize them.
    Bmodels {
        #[command(flatten)]
        filter: Filter,
    },
    /// Parse a Havok resource (`physhape` .hkx or `physworld` .bphysworld): item
    /// inventory, schema of used types, PTCH cross-check, optional item dumps.
    Havok {
        #[arg(long, default_value_t = 0)]
        group: usize,
        /// Resource kind: `physhape` or `physworld`.
        #[arg(long)]
        kind: String,
        #[arg(long)]
        name: String,
        /// Print the member layout of every type used by an item.
        #[arg(long)]
        schema: bool,
        /// Item indices to dump (first elements only, see --max-elements).
        #[arg(long, value_delimiter = ',')]
        dump: Vec<u32>,
        /// Dump the first items of this type name (up to 3 items).
        #[arg(long)]
        dump_type: Option<String>,
        #[arg(long, default_value_t = 4)]
        max_elements: u32,
    },
    /// Flatten a map's `.bphysworld` into world-space convex hulls; report
    /// coverage, unsupported shape types, and optionally export a region as OBJ.
    Collision {
        #[arg(long, default_value_t = 0)]
        group: usize,
        #[arg(
            long,
            default_value = "generated/physworld/maps/campaign/tutorial/tutorial_p/tutorial_p.bphysworld"
        )]
        name: String,
        /// Region filter x0,y0,z0,x1,y1,z1 (hull bounds must overlap).
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        region: Vec<f32>,
        /// Write the selected hulls as Wavefront OBJ (file space).
        #[arg(long)]
        obj: Option<PathBuf>,
    },
    /// Parse a map's `.entities`, resolve inheritance, and inventory entities.
    Entities {
        #[arg(long, default_value_t = 0)]
        group: usize,
        #[arg(long)]
        name: String,
        /// Write one TSV row per entity here.
        #[arg(long)]
        tsv: Option<PathBuf>,
    },
    /// Parse static world models (`.bwm`) matching the filter and summarize them.
    Bwm {
        #[command(flatten)]
        filter: Filter,
    },
    /// Skinned models (docs/formats/md6.md): resolve an `md6Def` (inherit chain,
    /// LOD meshes, skeleton), decode and check meshes and skeleton, write the bind
    /// pose as OBJ; or survey a snapshot's md6 models / every baseModel and skeleton.
    Md6 {
        /// Preferred group when a name exists in several groups (default: lowest).
        #[arg(long)]
        group: Option<usize>,
        /// md6 decl name, e.g. `models/characters/player/emily/body.md6`.
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = 0)]
        lod: u32,
        /// Write the bind pose of `--name` as Wavefront OBJ (model space, z up).
        #[arg(long)]
        obj: Option<PathBuf>,
        /// Print the joint table of `--name`.
        #[arg(long)]
        joints: bool,
        /// Parse every md6 of an F3 snapshot `models.tsv` in every LOD and compare
        /// with the drawn materials of the sibling `surfaces.tsv`.
        #[arg(long)]
        snapshot: Option<PathBuf>,
        /// Parse every `baseModel` and `skeleton` resource.
        #[arg(long)]
        all: bool,
    },
    /// Texture/resource mods (docs/texture-replacement.md): validation shared with the
    /// in-game hook, DDS import/export, test patterns, BIM writer round trip.
    Texmod {
        #[command(subcommand)]
        cmd: TexmodCmd,
    },
    Verify {
        #[command(flatten)]
        filter: Filter,
        /// Only check every Nth matching entry.
        #[arg(long, default_value_t = 1)]
        every: usize,
    },
}

#[derive(Subcommand)]
enum TexmodCmd {
    /// Validate every mod in a mods directory against the archives (the same checks
    /// void-inject runs before serving a mod).
    Check {
        #[arg(long)]
        mods: PathBuf,
    },
    /// Validate a DDS as replacement for an image and copy it to
    /// `<mods>/textures/<image name>.dds`.
    Import {
        dds: PathBuf,
        /// Image name (`models/x_d.tga`) or resource (`generated/image/....bimage7`).
        #[arg(long)]
        image: String,
        #[arg(long)]
        mods: PathBuf,
    },
    /// Write an original image (base file plus streamed parts) as one DDS.
    Export {
        #[arg(long)]
        image: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write a constant-colour replacement with the original's format, size and mip
    /// count into `<mods>/textures/` (in-game smoke test without external tools).
    TestPattern {
        #[arg(long)]
        image: String,
        #[arg(long)]
        mods: PathBuf,
        /// RGBA bytes (signed formats map 0..255 to -1..1).
        #[arg(long, value_delimiter = ',', default_value = "255,0,255,255")]
        color: Vec<u8>,
        /// One colour per mip level instead (magenta, green, cyan, yellow, red, blue,
        /// white, black, repeating) to show which mip the engine samples.
        #[arg(long)]
        per_mip: bool,
    },
    /// Rewrite every image (base files and streamed parts) listed by a `.mapresources`
    /// from its parsed form and compare the bytes with the archive.
    Roundtrip {
        #[arg(long, default_value_t = 0)]
        group: usize,
        #[arg(
            long,
            default_value = "generated/buildgame/campaign/tutorial/tutorial_p.mapresources"
        )]
        name: String,
    },
}

#[derive(Args)]
struct Filter {
    /// Zero-based group (0 = game1).
    #[arg(long)]
    group: Option<usize>,
    /// Exact resource kind, e.g. `model` or `File`.
    #[arg(long)]
    kind: Option<String>,
    /// Exact resource name.
    #[arg(long)]
    name: Option<String>,
    /// Substring of the resource name.
    #[arg(long)]
    contains: Option<String>,
}

impl Filter {
    fn matches<'a>(
        &'a self,
        inst: &'a Installation,
    ) -> impl Iterator<Item = (usize, &'a Entry)> + 'a {
        inst.groups
            .iter()
            .enumerate()
            .filter(|(g, _)| self.group.is_none_or(|want| want == *g))
            .flat_map(|(g, group)| group.index.entries.iter().map(move |e| (g, e)))
            .filter(|(_, e)| {
                self.kind.as_deref().is_none_or(|k| e.kind == k)
                    && self.name.as_deref().is_none_or(|n| e.name == n)
                    && self.contains.as_deref().is_none_or(|s| e.name.contains(s))
            })
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let Some(game) = cli.game else {
        eprintln!("error: pass --game <DIR> or set VOID_GAME_DIR");
        return ExitCode::from(2);
    };
    let inst = match Installation::open(&game) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = match cli.command {
        Command::Info => info(&inst),
        Command::List { filter, limit } => list(&inst, &filter, limit),
        Command::Extract { filter, out, max } => extract(&inst, &filter, &out, max),
        Command::Deps { group, name, list } => deps(&inst, group, &name, list),
        Command::Bwm { filter } => bwm(&inst, &filter),
        Command::Materials { group, bwm, mips } => materials(&inst, group, &bwm, mips),
        Command::Bmodels { filter } => bmodels(&inst, &filter),
        Command::Havok {
            group,
            kind,
            name,
            schema,
            dump,
            dump_type,
            max_elements,
        } => {
            let opts = HavokDump {
                schema,
                items: dump,
                type_name: dump_type,
                max_elements,
            };
            havok(&inst, group, &kind, &name, &opts)
        }
        Command::Collision {
            group,
            name,
            region,
            obj,
        } => collision(&inst, group, &name, &region, obj.as_deref()),
        Command::Entities { group, name, tsv } => entities(&inst, group, &name, tsv.as_deref()),
        Command::Verify { filter, every } => verify(&inst, &filter, every.max(1)),
        Command::Md6 {
            group,
            name,
            lod,
            obj,
            joints,
            snapshot,
            all,
        } => md6::run(
            &inst,
            &md6::Opts {
                group,
                name: name.as_deref(),
                lod,
                obj: obj.as_deref(),
                joints,
                snapshot: snapshot.as_deref(),
                all,
            },
        ),
        Command::Texmod { cmd } => texmod(&inst, cmd),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Res = Result<bool, Box<dyn std::error::Error>>;

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn info(inst: &Installation) -> Res {
    let master = inst.base.join("master.index");
    println!("base: {}", inst.base.display());
    println!("master.index sha256 {}", sha256_file(&master)?);
    println!("master unknown field: {}", inst.master.unknown);
    for s in &inst.master.shared {
        println!("shared file {} serves groups {:?}", s.file, s.groups);
    }
    for (g, group) in inst.groups.iter().enumerate() {
        let entries = &group.index.entries;
        println!("\ngroup {g}: {}", group.index_path.display());
        println!("  sha256 {}", sha256_file(&group.index_path)?);
        for (slot, p) in group.resource_paths.iter().enumerate() {
            let len = fs::metadata(p)
                .map(|m| m.len().to_string())
                .unwrap_or_else(|e| e.to_string());
            println!("  slot {slot}: {} ({len} bytes)", p.display());
        }
        let compressed = entries.iter().filter(|e| e.is_compressed()).count();
        println!("  entries {} (compressed {compressed})", entries.len());

        let mut locations = BTreeMap::new();
        let mut flags = BTreeMap::new();
        let mut kinds = BTreeMap::new();
        let mut names = HashSet::new();
        let mut duplicates = 0usize;
        let mut nonzero_reserved = 0usize;
        let mut unresolved = 0usize;
        for e in entries {
            *locations.entry(e.location.0).or_insert(0usize) += 1;
            *flags.entry(e.flags).or_insert(0usize) += 1;
            *kinds.entry(e.kind.as_str()).or_insert(0usize) += 1;
            duplicates += usize::from(!names.insert(e.name.as_str()));
            nonzero_reserved += usize::from(e.reserved != 0);
            unresolved += usize::from(inst.resource_path(g, e).is_err());
        }
        println!("  kinds {}", kinds.len());
        println!("  duplicate names {duplicates}");
        println!("  non-zero reserved fields {nonzero_reserved}");
        println!("  unresolvable locations {unresolved}");
        println!(
            "  header reserved non-zero: {}",
            group.index.header_reserved.iter().any(|b| *b != 0)
        );
        println!("  flags {flags:x?}");
        println!(
            "  locations {}",
            locations
                .iter()
                .map(|(l, n)| format!("{l:#06x}:{n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok(true)
}

fn list(inst: &Installation, filter: &Filter, limit: Option<usize>) -> Res {
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(
        out,
        "group\tkind\tname\tsize\tpacked\tlocation\tflags\tsource"
    )?;
    for (g, e) in filter.matches(inst).take(limit.unwrap_or(usize::MAX)) {
        writeln!(
            out,
            "{g}\t{}\t{}\t{}\t{}\t{:#06x}\t{:#x}\t{}",
            e.kind, e.name, e.size, e.packed_size, e.location.0, e.flags, e.source
        )?;
    }
    Ok(true)
}

fn extract(inst: &Installation, filter: &Filter, out: &Path, max: usize) -> Res {
    let selected: Vec<_> = filter.matches(inst).collect();
    if selected.len() > max {
        eprintln!(
            "{} entries match; refusing to extract more than --max {max}",
            selected.len()
        );
        return Ok(false);
    }
    let mut ok = true;
    for (g, e) in selected {
        let dest = match safe_relative_path(&e.name) {
            Ok(rel) => out.join(format!("group{g}")).join(rel),
            Err(err) => {
                eprintln!("skip: {err}");
                ok = false;
                continue;
            }
        };
        let data = match inst.read(g, e) {
            Ok(d) => d,
            Err(err) => {
                eprintln!("fail: {err}");
                ok = false;
                continue;
            }
        };
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        match File::create_new(&dest) {
            Ok(mut f) => f.write_all(&data)?,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                eprintln!("skip: {} already exists", dest.display());
                ok = false;
                continue;
            }
            Err(err) => return Err(err.into()),
        }
        println!("{}\t{}", data.len(), dest.display());
    }
    Ok(ok)
}

fn deps(inst: &Installation, group: usize, name: &str, list: bool) -> Res {
    let index = &inst.group(group)?.index;
    let lookup = Lookup::new(index);
    let [source] = lookup.by_kind_name("File", name)[..] else {
        eprintln!("expected exactly one File entry named {name} in group {group}");
        return Ok(false);
    };
    let deps = MapResources::parse(&inst.read(group, source)?)?;
    let odd = deps
        .records
        .iter()
        .filter(|r| r.tag != 0x80 || r.padding != [0; 20])
        .count();
    let (mut missing, mut failed, mut entries, mut bytes) = (Vec::new(), 0usize, 0usize, 0u64);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = BufWriter::new(io::stdout().lock());
    if list {
        writeln!(out, "id	kind	name	size")?;
    }
    for rec in &deps.records {
        let mut any = false;
        for e in lookup.by_id(rec.id) {
            any = true;
            entries += 1;
            *kinds.entry(e.kind.as_str()).or_default() += 1;
            match inst.read(group, e) {
                Ok(d) => bytes += d.len() as u64,
                Err(err) => {
                    failed += 1;
                    eprintln!("fail: {err}");
                }
            }
            if list {
                writeln!(out, "{:#x}	{}	{}	{}", rec.id, e.kind, e.name, e.size)?;
            }
        }
        if !any {
            missing.push(rec.id);
        }
    }
    writeln!(
        out,
        "records {} (non-standard tag/padding {odd}); entries {entries}; read failures {failed}; bytes {bytes}",
        deps.records.len()
    )?;
    writeln!(out, "unresolved ids {:x?}", missing)?;
    writeln!(out, "kinds {kinds:?}")?;
    Ok(failed == 0)
}

fn bmodels(inst: &Installation, filter: &Filter) -> Res {
    use void_formats::bmodel::StaticModel;
    let (mut ok, mut failed, mut decodable) = (0usize, BTreeMap::<String, usize>::new(), 0usize);
    let mut layouts: BTreeMap<String, usize> = BTreeMap::new();
    for (g, e) in filter
        .matches(inst)
        .filter(|(_, e)| e.name.ends_with(".bmodel"))
    {
        match inst
            .read(g, e)
            .map_err(|e| e.to_string())
            .and_then(|d| StaticModel::parse(&d).map_err(|e| e.to_string()))
        {
            Ok(m) => {
                ok += 1;
                let mut all = true;
                for s in &m.surfaces {
                    let key = format!(
                        "v{:#x} cpu{} gpu{} streams {:?}",
                        s.version,
                        s.cpu_decl,
                        s.gpu_decl,
                        s.streams
                            .iter()
                            .map(|x| (x.offset, x.stride))
                            .collect::<Vec<_>>()
                    );
                    *layouts.entry(key).or_default() += 1;
                    all &= s.positions().is_some();
                }
                decodable += usize::from(all);
            }
            Err(err) => {
                let k = err.split(':').next().unwrap_or(&err).to_owned();
                *failed.entry(k).or_default() += 1;
            }
        }
    }
    println!(
        "parsed {ok}, fully decodable vertex layout {decodable}, failed {}",
        failed.values().sum::<usize>()
    );
    println!("surface layouts {layouts:#?}");
    println!("failures by kind {failed:#?}");
    Ok(failed.is_empty())
}

fn collision(
    inst: &Installation,
    group: usize,
    name: &str,
    region: &[f32],
    obj: Option<&Path>,
) -> Res {
    use void_formats::collision::Collision;
    let lookup = Lookup::new(&inst.group(group)?.index);
    let [entry] = lookup.by_kind_name("physworld", name)[..] else {
        return Err(format!("expected one physworld entry named {name}").into());
    };
    let w = void_formats::physworld::PhysWorld::parse(&inst.read(group, entry)?)?;
    let c = Collision::from_physics_system(&w.tagfile)?;
    let verts: usize = c.shapes.iter().map(|s| s.convex.vertices.len()).sum();
    println!(
        "convex hulls {} ({verts} vertices), non-uniform scales {}",
        c.shapes.len(),
        c.nonuniform_scales
    );
    let meshes: Vec<_> = c.shapes.iter().filter(|s| s.convex.is_mesh).collect();
    println!(
        "  of which compressed-mesh instances {} ({} triangles); max vertex excess over section domain {:.2e}",
        meshes.len(),
        meshes.iter().map(|s| s.convex.faces.len()).sum::<usize>(),
        c.max_mesh_domain_excess()
    );
    for (item, flags, n) in c.skipped_mesh_sections() {
        println!(
            "UNSUPPORTED compressed-mesh section: data item {item} flags {flags} ({n} primitives skipped)"
        );
    }
    let mut per_body: BTreeMap<usize, usize> = BTreeMap::new();
    for s in &c.shapes {
        *per_body.entry(s.body).or_default() += 1;
    }
    println!("hulls per body {per_body:?}");
    for (t, (n, example)) in &c.unsupported {
        println!("UNSUPPORTED leaf shape {t}: {n} instances (e.g. item {example})");
    }
    // Bounds of unsupported compressed meshes (tree domain, transformed).
    for (t, item, body, m) in &c.unsupported_leaves {
        let shape = w.tagfile.element(*item, 0)?;
        let data = shape.field("data").and_then(|v| v.as_ref()).unwrap_or(0);
        let dom = if data != 0 {
            w.tagfile
                .element(data, 0)?
                .path(&["meshTree", "domain"])
                .cloned()
        } else {
            None
        };
        let world = dom.as_ref().and_then(|d| {
            let lo = d.field("min")?.floats()?;
            let hi = d.field("max")?.floats()?;
            let (mut a, mut b) = ([f32::MAX; 3], [f32::MIN; 3]);
            for k in 0..8 {
                let p = [
                    if k & 1 == 0 { lo[0] } else { hi[0] },
                    if k & 2 == 0 { lo[1] } else { hi[1] },
                    if k & 4 == 0 { lo[2] } else { hi[2] },
                ];
                let q = void_formats::collision::apply(m, p);
                for i in 0..3 {
                    a[i] = a[i].min(q[i]);
                    b[i] = b[i].max(q[i]);
                }
            }
            Some((a, b))
        });
        println!("  {t} item {item} body {body} data {data} world bounds {world:.2?}");
    }
    let mut radii: Vec<f32> = c.shapes.iter().map(|s| s.convex.radius).collect();
    radii.sort_by(f32::total_cmp);
    if !radii.is_empty() {
        println!(
            "convex radius min {} median {} max {}",
            radii[0],
            radii[radii.len() / 2],
            radii[radii.len() - 1]
        );
    }
    let selected: Vec<_> = c
        .shapes
        .iter()
        .filter(|s| {
            let [x0, y0, z0, x1, y1, z1] = match region {
                [a, b, c, d, e, f] => [*a, *b, *c, *d, *e, *f],
                _ => return true,
            };
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &s.convex.vertices {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            hi[0] >= x0 && lo[0] <= x1 && hi[1] >= y0 && lo[1] <= y1 && hi[2] >= z0 && lo[2] <= z1
        })
        .collect();
    println!("selected hulls {}", selected.len());
    if let Some(path) = obj {
        let mut out = String::from("# void-tool collision export (file space, Z up)\n");
        let mut base = 1usize;
        for s in &selected {
            out += &format!("o body{}_item{}\n", s.body, s.shape_item);
            for v in &s.convex.vertices {
                out += &format!("v {} {} {}\n", v[0], v[1], v[2]);
            }
            for f in &s.convex.faces {
                out += "f";
                for &i in f {
                    out += &format!(" {}", base + i as usize);
                }
                out += "\n";
            }
            base += s.convex.vertices.len();
        }
        std::fs::write(path, out)?;
    }
    Ok(c.unsupported.is_empty() && c.skipped_mesh_sections().is_empty())
}

struct HavokDump {
    schema: bool,
    items: Vec<u32>,
    type_name: Option<String>,
    max_elements: u32,
}

fn havok(inst: &Installation, group: usize, kind: &str, name: &str, opts: &HavokDump) -> Res {
    let (schema, max_elements) = (opts.schema, opts.max_elements);
    use std::collections::BTreeSet;
    use void_formats::havok::TagFile;
    let lookup = Lookup::new(&inst.group(group)?.index);
    let [entry] = lookup.by_kind_name(kind, name)[..] else {
        return Err(format!("expected one {kind} entry named {name}").into());
    };
    let data = inst.read(group, entry)?;
    let tag: TagFile = match kind {
        "physworld" => {
            let w = void_formats::physworld::PhysWorld::parse(&data)?;
            println!(
                "physworld: {} contact materials, material map {} entries, compendium {} types",
                w.contact_materials.len(),
                w.material_map.len(),
                w.compendium.types.len() - 1
            );
            for (i, m) in w.contact_materials.iter().enumerate() {
                println!("  contact[{i}] {m}");
            }
            println!("  material map {:?}", w.material_map);
            w.tagfile
        }
        "physhape" => {
            let (prefix, t) = void_formats::physworld::parse_physhape(&data)?;
            println!("physhape prefix {prefix:02x?}");
            t
        }
        _ => return Err(format!("unsupported kind {kind}").into()),
    };
    println!(
        "tagfile SDK {} types {} items {} data {} bytes",
        tag.sdk,
        tag.types.types.len() - 1,
        tag.items.len(),
        tag.data.len()
    );
    let mut per_type: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for it in tag.items.iter().filter(|it| it.ty != 0) {
        let e = per_type.entry(tag.type_name(it.ty)).or_default();
        e.0 += 1;
        e.1 += u64::from(it.count);
    }
    for (n, (items, elems)) in &per_type {
        println!("  {items:7} items {elems:9} elements  {n}");
    }
    // Cross-check: offsets where the schema-driven decoder reads item
    // references must equal the PTCH pointer-fixup offsets.
    use void_formats::havok::Kind;
    for (label, kinds) in [
        ("pointer", &[Kind::Pointer][..]),
        ("array", &[Kind::Array]),
        ("string", &[Kind::String]),
    ] {
        println!(
            "  decoder {label} fields: {}",
            tag.reference_offsets(kinds)?.len()
        );
    }
    // O (tutorial_p physworld): PTCH lists exactly the non-null references.
    let ours: Vec<u32> = tag
        .reference_offsets(&[Kind::Pointer, Kind::Array, Kind::String])?
        .into_iter()
        .filter(|&o| {
            tag.data
                .get(o as usize..o as usize + 4)
                .is_some_and(|b| b != [0, 0, 0, 0])
        })
        .collect();
    let mut theirs: Vec<u32> = tag
        .patches
        .iter()
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
    theirs.sort_unstable();
    let same = ours == theirs || theirs.is_empty();
    println!(
        "non-null reference offsets: decoder {} PTCH {} -> {}",
        ours.len(),
        theirs.len(),
        if theirs.is_empty() {
            "no PTCH to compare"
        } else if same {
            "identical"
        } else {
            "DIFFERENT"
        }
    );
    if !same {
        let a: BTreeSet<_> = ours.iter().collect();
        let b: BTreeSet<_> = theirs.iter().collect();
        println!(
            "  only decoder: {:?}",
            a.difference(&b).take(10).collect::<Vec<_>>()
        );
        println!(
            "  only PTCH: {:?}",
            b.difference(&a).take(10).collect::<Vec<_>>()
        );
    }
    if schema {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<u32> = tag
            .items
            .iter()
            .filter(|it| it.ty != 0)
            .map(|it| it.ty)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        while let Some(t) = stack.pop() {
            if t == 0 || !seen.insert(t) {
                continue;
            }
            let ty = tag.types.get(t)?;
            let l = tag.types.layout(t);
            println!(
                "type {t} {}{} parent {} layout {:?} version {:?}",
                ty.name,
                if ty.templates.is_empty() {
                    String::new()
                } else {
                    format!("<{:?}>", ty.templates)
                },
                tag.type_name(ty.parent),
                l.as_ref().map(|l| (l.kind, l.size)).ok(),
                ty.version
            );
            for m in &ty.members {
                println!(
                    "    +{:<4} {:<24} {} (flags {:#x})",
                    m.offset,
                    m.name,
                    tag.type_name(m.ty),
                    m.flags
                );
                stack.push(m.ty);
            }
            stack.push(ty.parent);
            if let Some(s) = ty.subtype {
                stack.push(s);
            }
        }
    }
    let mut dump = opts.items.clone();
    if let Some(t) = &opts.type_name {
        dump.extend(tag.items_of(t).take(3));
    }
    for &i in &dump {
        let it = tag.item(i)?;
        println!(
            "item {i}: {} flags {:#x} offset {} count {}",
            tag.type_name(it.ty),
            it.flags,
            it.offset,
            it.count
        );
        for k in 0..it.count.min(max_elements) {
            println!("  [{k}] {:?}", tag.element(i, k)?);
        }
    }
    Ok(same)
}

fn entities(inst: &Installation, group: usize, name: &str, tsv: Option<&Path>) -> Res {
    use void_formats::entities::{
        EntitiesFile, Node, SpawnTransform, component_refs, entity_def_resource_name, parse_decl,
        resolve_inheritance,
    };
    let lookup = Lookup::new(&inst.group(group)?.index);
    let read = |kind: &str, n: &str| -> Option<Vec<u8>> {
        match lookup.by_kind_name(kind, n)[..] {
            [e] => inst.read(group, e).ok(),
            _ => None,
        }
    };
    let text = String::from_utf8(read("File", name).ok_or("entities file not found")?)?;
    let file = EntitiesFile::parse(&text)?;
    let mut defs: HashMap<String, Option<Node>> = HashMap::new();
    let (mut sections, mut classes, mut decl_types, mut comp_targets, mut missing_defs) = (
        BTreeMap::<String, usize>::new(),
        BTreeMap::<String, usize>::new(),
        BTreeMap::<String, usize>::new(),
        BTreeMap::<String, usize>::new(),
        BTreeMap::<String, usize>::new(),
    );
    let (mut with_model, mut model_found, mut non_identity_scale) = (0usize, 0usize, 0usize);
    let mut rows = String::from(
        "ordinal	name	class	inherit	model	model_resource	x	y	z	scale	components
",
    );
    let mut notable = Vec::new();
    for item in &file.items {
        *sections.entry(item.section.clone()).or_default() += 1;
        *decl_types.entry(item.decl_type.clone()).or_default() += 1;
        if item.section != "entity" {
            continue;
        }
        let (merged, chain, missing) = resolve_inheritance(&item.body, |inh| {
            defs.entry(inh.to_owned())
                .or_insert_with(|| {
                    read("entityDef", &entity_def_resource_name(inh))
                        .and_then(|b| String::from_utf8(b).ok())
                        .and_then(|t| parse_decl(&t).ok())
                })
                .clone()
        });
        for m in &missing {
            *missing_defs.entry(m.clone()).or_default() += 1;
        }
        let class = merged
            .get("class")
            .and_then(Node::scalar)
            .unwrap_or("?")
            .to_owned();
        *classes.entry(class.clone()).or_default() += 1;
        let empty = Node::Block(Vec::new());
        let edit = merged.get("edit").unwrap_or(&empty);
        let t = SpawnTransform::from_edit(edit);
        if t.scale != [1.0; 3] {
            non_identity_scale += 1;
        }
        let model = edit
            .path(&["renderModelInfo", "model"])
            .and_then(Node::scalar)
            .unwrap_or("");
        let mut model_res = String::new();
        if !model.is_empty() {
            with_model += 1;
            let (kind, cand) = match void_formats::entities::model_resource(model) {
                void_formats::entities::ModelResource::Static(n) => ("model", n),
                void_formats::entities::ModelResource::Skinned(n) => ("baseModel", n),
            };
            if !lookup.by_kind_name(kind, &cand).is_empty() {
                model_res = cand;
                model_found += 1;
            }
        }
        let comps = component_refs(edit);
        for (_, _, target) in &comps {
            let kind = target.split_once(':').map_or(target.as_str(), |(_, t)| t);
            let key = if let Some(d) = kind.rsplit_once('.') {
                format!(".{}", d.1)
            } else if kind.starts_with("map_") {
                kind.trim_start_matches("map_")
                    .rsplit_once("_decl")
                    .map_or(kind, |(t, _)| t)
                    .to_owned()
            } else {
                kind.to_owned()
            };
            *comp_targets.entry(key).or_default() += 1;
        }
        let lname = format!("{} {} {}", item.name, class, chain.join(" ")).to_ascii_lowercase();
        if lname.contains("door") || class.to_ascii_lowercase().contains("light") {
            notable.push(format!(
                "{}	{}	{}	{:.2},{:.2},{:.2}",
                item.name,
                class,
                chain.first().map_or("", String::as_str),
                t.position[0],
                t.position[1],
                t.position[2]
            ));
        }
        rows += &format!(
            "{}	{}	{}	{}	{}	{}	{}	{}	{}	{:?}	{}
",
            item.ordinal,
            item.name,
            class,
            chain.join(">"),
            model,
            model_res,
            t.position[0],
            t.position[1],
            t.position[2],
            t.scale,
            comps.len()
        );
    }
    println!(
        "version {} items {} sections {sections:?}",
        file.version,
        file.items.len()
    );
    println!("decl types {decl_types:?}");
    println!("entity classes ({}) {classes:?}", classes.len());
    println!(
        "entities with model {with_model}, model resource found {model_found}; non-identity scale {non_identity_scale}"
    );
    println!(
        "unresolved inherit targets {} {missing_defs:?}",
        missing_defs.len()
    );
    println!("component reference kinds {comp_targets:?}");
    println!("door/light candidates ({}):", notable.len());
    for n in notable.iter().take(60) {
        println!("  {n}");
    }
    if let Some(p) = tsv {
        fs::write(p, rows)?;
    }
    Ok(missing_defs.is_empty())
}

fn materials(inst: &Installation, group: usize, bwm_name: &str, mips: bool) -> Res {
    use void_formats::bim::Image;
    use void_formats::bwm::WorldModel;
    use void_formats::material::{Material, image_resource_name, material_resource_name};
    let lookup = Lookup::new(&inst.group(group)?.index);
    let one = |kind: &str, name: &str| -> Result<Vec<u8>, String> {
        match lookup.by_kind_name(kind, name)[..] {
            [e] => inst.read(group, e).map_err(|e| e.to_string()),
            [] => Err("not in index".into()),
            _ => Err("ambiguous".into()),
        }
    };
    let [bwm_entry] = lookup.by_kind_name("File", bwm_name)[..] else {
        eprintln!("world model {bwm_name} not found");
        return Ok(false);
    };
    let model = WorldModel::parse(&inst.read(group, bwm_entry)?)?;
    let names: std::collections::BTreeSet<&str> =
        model.batches.iter().map(|b| b.material.as_str()).collect();
    let mut programs: BTreeMap<String, usize> = BTreeMap::new();
    let mut slots: BTreeMap<String, usize> = BTreeMap::new();
    let mut options: BTreeMap<String, usize> = BTreeMap::new();
    let mut formats: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut images = std::collections::BTreeSet::new();
    for name in &names {
        let text = match one("material", &material_resource_name(name)) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("material {name}: {e}"));
                continue;
            }
        };
        let mat = match std::str::from_utf8(&text)
            .map_err(|e| e.to_string())
            .and_then(|t| Material::parse(t).map_err(|e| e.to_string()))
        {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("material {name}: {e}"));
                continue;
            }
        };
        *programs
            .entry(mat.program().unwrap_or("<none>").to_owned())
            .or_default() += 1;
        for (k, v) in mat.options() {
            *options.entry(format!("{k}={v}")).or_default() += 1;
        }
        for (slot, src) in mat.image_slots() {
            *slots.entry(slot.to_ascii_lowercase()).or_default() += 1;
            match image_resource_name(src) {
                Some(img) => {
                    images.insert(img);
                }
                None => {
                    *slots
                        .entry(format!("{}<image program>", slot.to_ascii_lowercase()))
                        .or_default() += 1
                }
            }
        }
    }
    let mut texels = 0u64;
    for img in &images {
        let parsed = one("image", img).and_then(|d| Image::parse(&d).map_err(|e| e.to_string()));
        match parsed {
            Ok(i) => {
                *formats
                    .entry(format!("{:?} type{}", i.format, i.header.texture_type))
                    .or_default() += 1;
                if mips {
                    for n in 0..i.streamed_mips() {
                        let part = Image::streamed_mip_name(img, n);
                        match one("image", &part)
                            .and_then(|d| i.parse_streamed(n, &d).map_err(|e| e.to_string()))
                        {
                            Ok(m) => texels += u64::from(m.width) * u64::from(m.height),
                            Err(e) => failures.push(format!("image {part}: {e}")),
                        }
                    }
                }
            }
            Err(e) => failures.push(format!("image {img}: {e}")),
        }
    }
    println!(
        "materials {} images {} streamed texels {texels}",
        names.len(),
        images.len()
    );
    println!("programs {programs:?}");
    println!("slots {slots:?}");
    println!("formats {formats:?}");
    println!("options {options:?}");
    println!("failures {}", failures.len());
    for f in &failures {
        println!("  {f}");
    }
    Ok(failures.is_empty())
}

fn bwm(inst: &Installation, filter: &Filter) -> Res {
    use void_formats::bwm::WorldModel;
    let mut ok = true;
    let selected: Vec<_> = filter
        .matches(inst)
        .filter(|(_, e)| e.name.ends_with(".bwm"))
        .collect();
    for (g, e) in selected {
        let data = inst.read(g, e)?;
        match WorldModel::parse(&data) {
            Ok(m) => {
                let verts: u64 = m.surfaces.iter().map(|s| u64::from(s.vertex_count)).sum();
                let tris: u64 = m
                    .surfaces
                    .iter()
                    .map(|s| u64::from(s.index_count / 3))
                    .sum();
                let materials: HashSet<&str> =
                    m.batches.iter().map(|b| b.material.as_str()).collect();
                let odd_surfaces = m
                    .surfaces
                    .iter()
                    .filter(|s| s.sentinel != 0x1234_5678 || s.unknown68 != 3 || s.flag != 1)
                    .count();
                println!(
                    "ok	{g}	{}	v{:#x}	streams {:?}	surfaces {}	verts {verts}	tris {tris}	instances {}	batches {}	materials {}	atypical surfaces {odd_surfaces}	lists4 {:?}",
                    e.name,
                    m.version,
                    m.streams.iter().map(Vec::len).collect::<Vec<_>>(),
                    m.surfaces.len(),
                    m.instances.len(),
                    m.batches.len(),
                    materials.len(),
                    m.tables.lists4.iter().map(Vec::len).collect::<Vec<_>>(),
                );
            }
            Err(err) => {
                ok = false;
                println!("fail	{g}	{}	{err}", e.name);
            }
        }
    }
    Ok(ok)
}

fn verify(inst: &Installation, filter: &Filter, every: usize) -> Res {
    let (mut checked, mut failed, mut bytes) = (0usize, 0usize, 0u64);
    for (g, e) in filter.matches(inst).step_by(every) {
        checked += 1;
        match inst.read(g, e) {
            Ok(d) => bytes += d.len() as u64,
            Err(err) => {
                failed += 1;
                eprintln!("fail: group {g}: {err}");
            }
        }
    }
    println!("checked {checked} entries, {failed} failed, {bytes} bytes decoded");
    Ok(failed == 0)
}

/// Original resources for `void_formats::texmod` from every archive group.
struct ArchiveOriginals<'a> {
    inst: &'a Installation,
    images: HashMap<&'a str, Vec<(usize, &'a Entry)>>,
    names: HashSet<&'a str>,
}

impl<'a> ArchiveOriginals<'a> {
    fn new(inst: &'a Installation) -> Self {
        let mut images: HashMap<&str, Vec<(usize, &Entry)>> = HashMap::new();
        let mut names = HashSet::new();
        for (g, group) in inst.groups.iter().enumerate() {
            for e in &group.index.entries {
                names.insert(e.name.as_str());
                if e.kind == "image" {
                    images.entry(e.name.as_str()).or_default().push((g, e));
                }
            }
        }
        Self {
            inst,
            images,
            names,
        }
    }
}

impl void_formats::texmod::Originals for ArchiveOriginals<'_> {
    fn images(&self, name: &str) -> Result<Vec<Vec<u8>>, String> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for (g, e) in self.images.get(name).into_iter().flatten() {
            let d = self.inst.read(*g, e).map_err(|e| e.to_string())?;
            if !out.contains(&d) {
                out.push(d);
            }
        }
        Ok(out)
    }

    fn exists(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

fn texmod(inst: &Installation, cmd: TexmodCmd) -> Res {
    use void_formats::texmod::{
        ModFile, ModSet, Originals, Source, check, constant_dds, image_resource_from_arg,
        texture_path,
    };
    let orig = ArchiveOriginals::new(inst);
    let first_original = |resource: &str| -> Result<void_formats::bim::Image, String> {
        let copies = orig.images(resource)?;
        let bytes = copies
            .first()
            .ok_or_else(|| format!("{resource}: not an image in the archive index"))?;
        void_formats::bim::Image::parse(bytes).map_err(|e| format!("{resource}: {e}"))
    };
    match cmd {
        TexmodCmd::Check { mods } => {
            let set = ModSet::scan(&mods);
            for n in &set.notes {
                println!("note\t{n}");
            }
            let mut failed = 0;
            for name in set.files.keys() {
                match check(&set, name, &orig) {
                    Ok(c) => println!(
                        "ok\t{name}\t{}{}",
                        c.summary,
                        if c.in_index { "" } else { " (not in index)" }
                    ),
                    Err(e) => {
                        failed += 1;
                        println!("fail\t{name}\t{e}");
                    }
                }
            }
            println!(
                "mods {}, ok {}, failed {failed}",
                set.files.len(),
                set.files.len() - failed
            );
            Ok(failed == 0 && mods.is_dir())
        }
        TexmodCmd::Import { dds, image, mods } => {
            let resource = image_resource_from_arg(&image);
            let mut set = ModSet {
                dir: mods.clone(),
                ..Default::default()
            };
            set.files.insert(
                resource.clone(),
                ModFile {
                    resource: resource.clone(),
                    path: dds.clone(),
                    source: Source::Dds,
                },
            );
            let c = check(&set, &resource, &orig)?;
            let dest = texture_path(&mods, &resource).ok_or("not an image resource")?;
            if let Some(dir) = dest.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::copy(&dds, &dest)?;
            println!("ok\t{resource}\t{}\t-> {}", c.summary, dest.display());
            Ok(true)
        }
        TexmodCmd::Export { image, out } => {
            let resource = image_resource_from_arg(&image);
            let img = first_original(&resource)?;
            let h = &img.header;
            let Some(format) = img.format else {
                return Err(format!("{resource}: unknown format").into());
            };
            if h.texture_type != 0 || h.faces != 1 {
                return Err(format!("{resource}: only single 2D images can be exported").into());
            }
            let mut mips = Vec::new();
            for n in 0..img.streamed_mips() {
                let part = void_formats::bim::Image::streamed_mip_name(&resource, n);
                let bytes = orig
                    .images(&part)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("{part}: missing"))?;
                mips.push(img.parse_streamed(n, &bytes)?.data);
            }
            mips.extend(img.mips.iter().map(|m| m.data.clone()));
            let dds = void_formats::dds::Dds {
                format,
                width: h.width,
                height: h.height,
                mips,
            };
            fs::write(&out, dds.to_bytes()?)?;
            println!(
                "{resource}: {format:?} {}x{} mips {} -> {}",
                h.width,
                h.height,
                dds.mips.len(),
                out.display()
            );
            Ok(true)
        }
        TexmodCmd::TestPattern {
            image,
            mods,
            color,
            per_mip,
        } => {
            const PALETTE: [[u8; 4]; 8] = [
                [255, 0, 255, 255],
                [0, 255, 0, 255],
                [0, 255, 255, 255],
                [255, 255, 0, 255],
                [255, 0, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 255],
            ];
            let [r, g, b, a] = color[..] else {
                return Err("--color needs four values R,G,B,A".into());
            };
            let resource = image_resource_from_arg(&image);
            let img = first_original(&resource)?;
            let h = &img.header;
            let format = img.format.ok_or("unknown format")?;
            let pick = move |i: u32| {
                if per_mip {
                    PALETTE[i as usize % PALETTE.len()]
                } else {
                    [r, g, b, a]
                }
            };
            let dds = constant_dds(format, h.width, h.height, h.total_mips, &pick)?;
            let dest = texture_path(&mods, &resource).ok_or("not an image resource")?;
            if let Some(dir) = dest.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::write(&dest, dds.to_bytes()?)?;
            // Validate the written file with the shared code path.
            let set = ModSet::scan(&mods);
            let c = check(&set, &resource, &orig)?;
            println!("ok\t{resource}\t{}\t-> {}", c.summary, dest.display());
            Ok(true)
        }
        TexmodCmd::Roundtrip { group, name } => texmod_roundtrip(inst, group, &name),
    }
}

/// Writer check: every image entry of a map's resource list, re-serialized from its
/// parsed form, must equal the archive bytes. Every original must also pass the mod
/// validation as a replacement for itself (the rules accept what the game ships).
fn texmod_roundtrip(inst: &Installation, group: usize, name: &str) -> Res {
    use void_formats::bim::Image;
    use void_formats::texmod::split_part;
    let lookup = Lookup::new(&inst.group(group)?.index);
    let [source] = lookup.by_kind_name("File", name)[..] else {
        eprintln!("expected exactly one File entry named {name} in group {group}");
        return Ok(false);
    };
    let deps = MapResources::parse(&inst.read(group, source)?)?;
    let mut images: Vec<&Entry> = deps
        .records
        .iter()
        .flat_map(|r| lookup.by_id(r.id))
        .filter(|e| e.kind == "image")
        .collect();
    images.sort_by(|a, b| a.name.cmp(&b.name));
    images.dedup_by(|a, b| a.name == b.name);
    let (mut bases, mut parts, mut bytes, mut failures) = (0usize, 0usize, 0u64, Vec::new());
    let mut parsed: HashMap<&str, Image> = HashMap::new();
    for e in &images {
        let data = inst.read(group, e)?;
        let result = if let Some((base, n)) = split_part(&e.name) {
            if !parsed.contains_key(base) {
                match lookup.by_kind_name("image", base)[..] {
                    [b] => {
                        let img = Image::parse(&inst.read(group, b)?)?;
                        parsed.insert(base, img);
                    }
                    _ => {
                        failures.push(format!("{}: base {base} not unique/present", e.name));
                        continue;
                    }
                }
            }
            parts += 1;
            parsed[base].parse_streamed(n, &data).map(|m| m.to_bytes())
        } else {
            bases += 1;
            Image::parse(&data).map(|i| {
                let out = i.to_bytes();
                parsed.insert(e.name.as_str(), i);
                out
            })
        };
        match result {
            Ok(out) if out == data => bytes += data.len() as u64,
            Ok(_) => failures.push(format!("{}: rewritten bytes differ", e.name)),
            Err(err) => failures.push(format!("{}: {err}", e.name)),
        }
    }
    let mut self_checked = 0usize;
    for (base, img) in &parsed {
        let parts: Result<Vec<Vec<u8>>, String> = (0..img.streamed_mips())
            .map(|n| {
                let part = Image::streamed_mip_name(base, n);
                match lookup.by_kind_name("image", &part)[..] {
                    [p] => inst.read(group, p).map_err(|e| e.to_string()),
                    _ => Err(format!("{part} not unique/present")),
                }
            })
            .collect();
        match parts.and_then(|p| {
            void_formats::bim::check_replacement(&img.header, img, &p).map_err(|e| e.to_string())
        }) {
            Ok(()) => self_checked += 1,
            Err(err) => failures.push(format!("{base}: self-check: {err}")),
        }
    }
    println!("originals accepted by the mod validation: {self_checked}");
    println!(
        "images {} (base files {bases}, streamed parts {parts}), identical bytes {bytes}, failures {}",
        images.len(),
        failures.len()
    );
    for f in &failures {
        println!("  {f}");
    }
    Ok(failures.is_empty())
}
