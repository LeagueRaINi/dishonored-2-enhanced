//! `void-tool md6`: skinned models (docs/formats/md6.md). Resolves an `md6Def`
//! (inheritance, LOD meshes, skeleton), decodes the meshes and the Edge
//! skeleton, checks skinning data, and optionally writes the bind pose as OBJ.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use void_formats::md6::{self, MeshRef, Model, Skeleton};
use void_res::{Entry, Installation};

use crate::Res;

pub struct Opts<'a> {
    pub group: Option<usize>,
    pub name: Option<&'a str>,
    pub lod: u32,
    pub obj: Option<&'a Path>,
    pub joints: bool,
    pub snapshot: Option<&'a Path>,
    pub all: bool,
}

/// Name lookup over every group (a name in several groups is read from the
/// preferred group if it has it, else from the lowest group).
struct Names<'a> {
    inst: &'a Installation,
    prefer: Option<usize>,
    by_name: HashMap<&'a str, Vec<(usize, &'a Entry)>>,
    /// Mesh file stem -> `nocloth_<stem>_<hash>.bmd6model` names.
    nocloth: HashMap<&'a str, BTreeSet<&'a str>>,
}

impl<'a> Names<'a> {
    fn new(inst: &'a Installation, prefer: Option<usize>) -> Self {
        let mut by_name: HashMap<&str, Vec<(usize, &Entry)>> = HashMap::new();
        let mut nocloth: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for (g, group) in inst.groups.iter().enumerate() {
            for e in &group.index.entries {
                by_name.entry(e.name.as_str()).or_default().push((g, e));
                if let Some(stem) = md6::nocloth_stem(&e.name) {
                    nocloth.entry(stem).or_default().insert(e.name.as_str());
                }
            }
        }
        Self {
            inst,
            prefer,
            by_name,
            nocloth,
        }
    }

    fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let list = self.by_name.get(name).ok_or("not in any index")?;
        let &(g, e) = list
            .iter()
            .find(|(g, _)| Some(*g) == self.prefer)
            .unwrap_or(&list[0]);
        self.inst.read(g, e).map_err(|e| e.to_string())
    }
}

fn resolve_def(names: &Names, md6: &str) -> Result<md6::ResolvedDef, String> {
    md6::resolve_def(&|n| names.read(n), md6)
}

/// Built resource for a mesh reference: the direct build, or for meshes with
/// cloth the cloth-stripped `nocloth_*` build (cloth parts live in hashed
/// `voidresourcecache/md6model/cloth28/` entries whose naming is not decoded).
fn mesh_resource(names: &Names, mesh: &MeshRef) -> Result<(String, usize), String> {
    let direct =
        md6::mesh_resource(&mesh.mesh).ok_or_else(|| format!("{}: not .md6mesh", mesh.mesh))?;
    let lower = mesh.mesh.to_ascii_lowercase();
    let stem = lower
        .strip_suffix(".md6mesh")
        .and_then(|s| s.rsplit('/').next())
        .unwrap_or_default();
    let nocloth = names.nocloth.get(stem);
    let first_nocloth = nocloth.and_then(|s| s.iter().next());
    match (mesh.cloth.is_empty(), names.has(&direct), first_nocloth) {
        (true, true, _) | (false, true, None) => Ok((direct, 1)),
        (_, _, Some(n)) => Ok(((*n).to_owned(), nocloth.map_or(0, BTreeSet::len))),
        (_, false, None) => Err(format!("{}: no {direct} and no nocloth build", mesh.mesh)),
    }
}

struct Part {
    mesh: String,
    resource: String,
    candidates: usize,
    model: Model,
}

struct Character {
    chain: Vec<String>,
    lods: Vec<u32>,
    skeleton: Option<(String, Skeleton)>,
    parts: Vec<Part>,
    cloth_meshes: usize,
}

fn load(names: &Names, md6: &str, lod: u32) -> Result<Character, String> {
    let def = resolve_def(names, md6)?;
    let meshes = def.lods.get(&lod).ok_or_else(|| {
        format!(
            "{md6}: no lod {lod} (has {:?})",
            def.lods.keys().collect::<Vec<_>>()
        )
    })?;
    let mut parts = Vec::new();
    let mut cloth_meshes = 0;
    for m in meshes {
        cloth_meshes += usize::from(!m.cloth.is_empty());
        let (resource, candidates) = mesh_resource(names, m)?;
        let data = names
            .read(&resource)
            .map_err(|e| format!("{resource}: {e}"))?;
        let model = Model::parse(&data).map_err(|e| format!("{resource}: {e}"))?;
        parts.push(Part {
            mesh: m.mesh.clone(),
            resource,
            candidates,
            model,
        });
    }
    let skl = def.skeleton.clone().or_else(|| {
        parts
            .iter()
            .map(|p| p.model.skeleton.clone())
            .find(|s| !s.is_empty())
    });
    let skeleton = match skl {
        Some(s) => {
            let res = md6::skeleton_resource(&s).ok_or_else(|| format!("{s}: not .md6skl"))?;
            let data = names.read(&res).map_err(|e| format!("{res}: {e}"))?;
            let sk = Skeleton::parse(&data).map_err(|e| format!("{res}: {e}"))?;
            Some((res, sk))
        }
        None => None,
    };
    Ok(Character {
        chain: def.chain,
        lods: def.lods.keys().copied().collect(),
        skeleton,
        parts,
        cloth_meshes,
    })
}

/// Skinning and geometry checks over all surfaces of a character.
#[derive(Default)]
struct Check {
    surfaces: usize,
    vertices: usize,
    triangles: usize,
    min: [f32; 3],
    max: [f32; 3],
    /// Vertices whose weights do not sum to 255.
    weight_sum_bad: usize,
    /// Influences (non-zero weight) whose joint is outside the skeleton.
    joint_out_of_range: usize,
    /// Influences whose joint is missing from the surface's used-joint set.
    joint_not_in_set: usize,
    /// Vertices by number of non-zero weights (index 0..=8).
    influence_hist: [usize; 9],
    /// Parts whose skeleton checksum differs from the character's skeleton.
    checksum_mismatch: Vec<String>,
    decls: BTreeSet<i32>,
    /// Median distance from a vertex to its dominant joint's bind position.
    median_joint_distance: f32,
    /// Over surfaces with a non-zero `joint_offset` only: the median with the
    /// offset applied, and with the stored indices used directly.
    offset_surfaces_distance: [f32; 2],
    surfaces_with_offset: usize,
    materials: BTreeSet<String>,
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

fn check(ch: &Character) -> Check {
    let mut c = Check {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
        ..Check::default()
    };
    let (joints, bind) = match &ch.skeleton {
        Some((_, sk)) => (sk.joints.len(), sk.bind_pose()),
        None => (0, Vec::new()),
    };
    let (mut dist, mut dist_off, mut dist_raw) = (Vec::new(), Vec::new(), Vec::new());
    for p in &ch.parts {
        let m = &p.model;
        if let (Some(a), Some((_, sk))) = (m.skeleton_checksum, &ch.skeleton)
            && a != sk.model_checksum
        {
            c.checksum_mismatch.push(format!(
                "{} built for {} ({a:#06x}), character skeleton {} ({:#06x})",
                p.resource,
                m.skeleton,
                ch.skeleton.as_ref().map_or("-", |(r, _)| r.as_str()),
                sk.model_checksum
            ));
        }
        for s in &m.surfaces {
            c.surfaces += 1;
            c.decls.insert(s.cpu_decl);
            c.materials.insert(s.material.clone());
            c.surfaces_with_offset += usize::from(s.joint_offset != 0);
            c.vertices += s.vertex_count as usize;
            c.triangles += s.indices.len() / 3;
            let v = &s.vertices;
            for pos in &v.positions {
                for k in 0..3 {
                    c.min[k] = c.min[k].min(pos[k]);
                    c.max[k] = c.max[k].max(pos[k]);
                }
            }
            for (i, (w, j)) in v.weights.iter().zip(&v.joints).enumerate() {
                let sum: u32 = w.iter().map(|&x| u32::from(x)).sum();
                c.weight_sum_bad += usize::from(sum != 255);
                c.influence_hist[w.iter().filter(|&&x| x != 0).count()] += 1;
                for (joint, _) in s.influences(i) {
                    c.joint_out_of_range += usize::from(joint as usize >= joints);
                    c.joint_not_in_set += usize::from(!s.uses_joint(joint));
                }
                let k = (0..w.len()).max_by_key(|&k| w[k]).unwrap_or(0);
                let d = |j: usize| {
                    bind.get(j).map(|g| {
                        let p = v.positions[i];
                        ((p[0] - g[0][3]).powi(2)
                            + (p[1] - g[1][3]).powi(2)
                            + (p[2] - g[2][3]).powi(2))
                        .sqrt()
                    })
                };
                if let Some(x) = d(usize::from(j[k]) + s.joint_offset as usize) {
                    dist.push(x);
                    if s.joint_offset != 0 {
                        dist_off.push(x);
                    }
                }
                if s.joint_offset != 0
                    && let Some(x) = d(usize::from(j[k]))
                {
                    dist_raw.push(x);
                }
            }
        }
    }
    c.median_joint_distance = median(dist);
    c.offset_surfaces_distance = [median(dist_off), median(dist_raw)];
    c
}

fn print_character(md6: &str, lod: u32, ch: &Character, c: &Check) {
    println!(
        "md6 {md6} lod {lod} (lods {:?}, chain {:?})",
        ch.lods, ch.chain
    );
    match &ch.skeleton {
        Some((res, sk)) => println!(
            "skeleton {res}: {} joints, checksums {:#06x}/{:#06x}, inverse-bind error {:.2e}",
            sk.joints.len(),
            sk.checksum,
            sk.model_checksum,
            sk.inverse_bind_error()
        ),
        None => println!("skeleton: none"),
    }
    for p in &ch.parts {
        let alt = if p.candidates > 1 {
            format!(" (1 of {} nocloth builds)", p.candidates)
        } else {
            String::new()
        };
        println!("mesh {} -> {}{alt}", p.mesh, p.resource);
        for s in &p.model.surfaces {
            println!(
                "  surface {:<12} kind {:>2} decl {:>2} verts {:>6} tris {:>6} ranges {:>2} joints {:>3}+{:<3} {}",
                s.name,
                s.kind,
                s.cpu_decl,
                s.vertex_count,
                s.indices.len() / 3,
                s.ranges.len(),
                s.joint_offset,
                s.joint_count,
                s.material
            );
        }
    }
    if ch.cloth_meshes > 0 {
        println!(
            "{} mesh(es) with cloth: cloth-simulated parts not included",
            ch.cloth_meshes
        );
    }
    println!(
        "total: {} surfaces, {} vertices, {} triangles, bounds min {:.3?} max {:.3?} (height {:.3} m)",
        c.surfaces,
        c.vertices,
        c.triangles,
        c.min,
        c.max,
        c.max[2] - c.min[2]
    );
    println!(
        "skinning: decls {:?}, weight sum != 255: {}, joint out of range: {}, joint not in used set: {}, skeleton checksum mismatches: {}",
        c.decls,
        c.weight_sum_bad,
        c.joint_out_of_range,
        c.joint_not_in_set,
        c.checksum_mismatch.len()
    );
    println!(
        "influences per vertex (0..8): {:?}; median vertex-to-dominant-joint distance {:.3} m; {} surfaces with joint_offset: {:.3} m with the offset, {:.3} m without",
        c.influence_hist,
        c.median_joint_distance,
        c.surfaces_with_offset,
        c.offset_surfaces_distance[0],
        c.offset_surfaces_distance[1]
    );
}

fn print_joints(sk: &Skeleton) {
    println!("joint\tparent\tname\tbind_x\tbind_y\tbind_z");
    for (i, (j, g)) in sk.joints.iter().zip(sk.bind_pose()).enumerate() {
        let parent = j.parent.map_or("-".to_owned(), |p| p.to_string());
        println!(
            "{i}\t{parent}\t{}\t{:.4}\t{:.4}\t{:.4}",
            j.name, g[0][3], g[1][3], g[2][3]
        );
    }
}

/// Bind-pose OBJ in model space (z up, metres). Faces are written
/// counter-clockwise (the engine's front faces are clockwise, as for `.bwm`).
fn write_obj(path: &Path, ch: &Character) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# void-tool md6 bind pose: {}", ch.chain.join(" <- "))?;
    let mut base = 1usize;
    for p in &ch.parts {
        for (si, s) in p.model.surfaces.iter().enumerate() {
            let v = &s.vertices;
            writeln!(w, "o {}#{si}_{}", p.resource, s.name)?;
            writeln!(w, "usemtl {}", s.material)?;
            for q in &v.positions {
                writeln!(w, "v {} {} {}", q[0], q[1], q[2])?;
            }
            for t in &v.uv0 {
                writeln!(w, "vt {} {}", t[0], 1.0 - t[1])?;
            }
            for n in &v.normals {
                writeln!(w, "vn {} {} {}", n[0], n[1], n[2])?;
            }
            let (has_t, has_n) = (!v.uv0.is_empty(), !v.normals.is_empty());
            for t in s.indices.as_chunks::<3>().0 {
                write!(w, "f")?;
                for &i in [t[0], t[2], t[1]].iter() {
                    let i = base + usize::from(i);
                    match (has_t, has_n) {
                        (true, true) => write!(w, " {i}/{i}/{i}")?,
                        (true, false) => write!(w, " {i}/{i}")?,
                        (false, true) => write!(w, " {i}//{i}")?,
                        (false, false) => write!(w, " {i}")?,
                    }
                }
                writeln!(w)?;
            }
            base += v.positions.len();
        }
    }
    w.flush()
}

/// md6 names and their drawn materials from an F3 snapshot
/// (`models.tsv` + `surfaces.tsv` in the same directory).
fn snapshot_md6(models: &Path) -> std::io::Result<BTreeMap<String, BTreeSet<String>>> {
    let mut by_ptr: HashMap<String, String> = HashMap::new();
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in BufReader::new(File::open(models)?).lines().skip(1) {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        if let (Some(ptr), Some(name)) = (f.get(2), f.get(3))
            && name.ends_with(".md6")
        {
            by_ptr.insert((*ptr).to_owned(), (*name).to_owned());
            out.entry((*name).to_owned()).or_default();
        }
    }
    let surfaces = models.with_file_name("surfaces.tsv");
    if let Ok(file) = File::open(&surfaces) {
        for line in BufReader::new(file).lines().skip(1) {
            let line = line?;
            let f: Vec<&str> = line.split('\t').collect();
            if let (Some(ptr), Some(mat)) = (f.first(), f.get(2))
                && let Some(name) = by_ptr.get(*ptr)
            {
                out.entry(name.clone())
                    .or_default()
                    .insert((*mat).to_owned());
            }
        }
    }
    Ok(out)
}

fn survey_snapshot(names: &Names, path: &Path) -> Res {
    let wanted = snapshot_md6(path)?;
    let (mut ok, mut lods_ok, mut lods_total, mut mat_match) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    let mut notes = BTreeSet::new();
    let mut decls = BTreeSet::new();
    let mut hist = [0usize; 9];
    let (mut bad_w, mut bad_j, mut not_set, mut cks) = (0, 0, 0, 0);
    let mut ambiguous = 0;
    println!(
        "md6\tlods\tparts\tvertices\ttriangles\theight\tjoints\tmedian_joint_dist\tmaterials_match"
    );
    for (md6, drawn) in &wanted {
        let def = match resolve_def(names, md6) {
            Ok(d) => d,
            Err(e) => {
                failures.push(e);
                continue;
            }
        };
        let mut all = true;
        let mut ours = BTreeSet::new();
        let mut first: Option<(Character, Check)> = None;
        for &lod in def.lods.keys() {
            lods_total += 1;
            match load(names, md6, lod) {
                Ok(ch) => {
                    lods_ok += 1;
                    let c = check(&ch);
                    decls.extend(c.decls.iter().copied());
                    for (h, x) in hist.iter_mut().zip(c.influence_hist) {
                        *h += x;
                    }
                    bad_w += c.weight_sum_bad;
                    bad_j += c.joint_out_of_range;
                    not_set += c.joint_not_in_set;
                    cks += c.checksum_mismatch.len();
                    for x in &c.checksum_mismatch {
                        notes.insert(format!("{md6} lod {lod}: checksum {x}"));
                    }
                    ambiguous += ch.parts.iter().filter(|p| p.candidates > 1).count();
                    ours.extend(c.materials.iter().cloned());
                    if first.is_none() {
                        first = Some((ch, c));
                    }
                }
                Err(e) => {
                    all = false;
                    failures.push(format!("lod {lod}: {e}"));
                }
            }
        }
        let matched = drawn.is_subset(&ours);
        mat_match += usize::from(matched);
        if all && let Some((ch, c)) = &first {
            ok += 1;
            println!(
                "{md6}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{:.3}\t{}",
                def.lods.len(),
                ch.parts.len(),
                c.vertices,
                c.triangles,
                c.max[2] - c.min[2],
                ch.skeleton.as_ref().map_or(0, |(_, s)| s.joints.len()),
                c.median_joint_distance,
                if matched {
                    "yes".to_owned()
                } else {
                    format!("no: {:?}", drawn.difference(&ours).collect::<Vec<_>>())
                }
            );
        }
    }
    println!(
        "snapshot md6: {ok} of {} parse in every LOD ({lods_ok} of {lods_total} LODs); drawn materials found in the decoded meshes for {mat_match}",
        wanted.len()
    );
    println!(
        "vertex declarations {decls:?}; influences per vertex (0..8) {hist:?}; weight sum != 255: {bad_w}; joint out of range: {bad_j}; joint not in used set: {not_set}; checksum mismatches: {cks}; parts with several nocloth builds: {ambiguous}"
    );
    for f in &failures {
        println!("FAIL {f}");
    }
    for f in &notes {
        println!("NOTE {f}");
    }
    Ok(failures.is_empty())
}

fn survey_all(names: &Names) -> Res {
    let mut models: BTreeSet<&str> = BTreeSet::new();
    let mut skeletons: BTreeSet<&str> = BTreeSet::new();
    for list in names.by_name.values() {
        for (_, e) in list {
            match e.kind.as_str() {
                "baseModel" => models.insert(e.name.as_str()),
                "skeleton" => skeletons.insert(e.name.as_str()),
                _ => false,
            };
        }
    }
    let mut failures: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    let mut decls: BTreeMap<i32, usize> = BTreeMap::new();
    let mut hist = [0usize; 9];
    let (mut ok, mut gpu, mut bad_w, mut not_set, mut offset, mut flags1) = (0, 0, 0, 0, 0, 0);
    for name in &models {
        match names
            .read(name)
            .and_then(|d| Model::parse(&d).map_err(|e| e.to_string()))
        {
            Ok(m) => {
                ok += 1;
                flags1 += usize::from(m.flags & 1 != 0);
                for s in &m.surfaces {
                    *decls.entry(s.cpu_decl).or_default() += 1;
                    gpu += usize::from(!s.streams.is_empty());
                    offset += usize::from(s.joint_offset != 0);
                    let v = &s.vertices;
                    for (i, w) in v.weights.iter().enumerate() {
                        bad_w += usize::from(w.iter().map(|&x| u32::from(x)).sum::<u32>() != 255);
                        hist[w.iter().filter(|&&x| x != 0).count()] += 1;
                        not_set += s.influences(i).filter(|(j, _)| !s.uses_joint(*j)).count();
                    }
                }
            }
            Err(e) => failures.entry(e).or_default().push(name),
        }
    }
    println!(
        "baseModel: {ok} of {} parse; surfaces by vertex declaration {decls:?}; with GPU streams {gpu}; with joint offset {offset}; models with flag bit 0 {flags1}",
        models.len()
    );
    println!(
        "influences per vertex (0..8) {hist:?}; weight sum != 255: {bad_w}; influences outside the used-joint set: {not_set}"
    );
    let (mut sk_ok, mut worst) = (0, 0f32);
    for name in &skeletons {
        match names
            .read(name)
            .and_then(|d| Skeleton::parse(&d).map_err(|e| e.to_string()))
        {
            Ok(s) => {
                sk_ok += 1;
                worst = worst.max(s.inverse_bind_error());
            }
            Err(e) => failures.entry(e).or_default().push(name),
        }
    }
    println!(
        "skeleton: {sk_ok} of {} parse; max |inverse_bind * bind - I| {worst:.2e}",
        skeletons.len()
    );
    for (e, list) in &failures {
        println!("FAIL {} x {e} (e.g. {})", list.len(), list[0]);
    }
    Ok(failures.is_empty())
}

pub fn run(inst: &Installation, o: &Opts) -> Res {
    let names = Names::new(inst, o.group);
    let mut ok = true;
    if o.all {
        ok &= survey_all(&names)?;
    }
    if let Some(path) = o.snapshot {
        ok &= survey_snapshot(&names, path)?;
    }
    if let Some(md6) = o.name {
        let ch = match load(&names, md6, o.lod) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {e}");
                return Ok(false);
            }
        };
        let c = check(&ch);
        print_character(md6, o.lod, &ch, &c);
        if o.joints
            && let Some((_, sk)) = &ch.skeleton
        {
            print_joints(sk);
        }
        if let Some(path) = o.obj {
            write_obj(path, &ch)?;
            println!("wrote {}", path.display());
        }
        ok &= c.weight_sum_bad == 0 && c.joint_out_of_range == 0 && c.checksum_mismatch.is_empty();
    }
    if !o.all && o.snapshot.is_none() && o.name.is_none() {
        eprintln!("error: pass --name, --snapshot or --all");
        return Ok(false);
    }
    Ok(ok)
}
