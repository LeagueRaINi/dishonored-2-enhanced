//! Parses a map's probe network and every probe it names; prints a summary.
//! usage: probes_check <game root> <network resource name>

use std::path::Path;

use void_formats::probes::{EnvProbe, ProbeNetwork};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ar = void_assets::Archive::open(Path::new(&a[1]))?;
    let net = ProbeNetwork::parse(&ar.read("File", &a[2])?).map_err(|e| e.to_string())?;
    println!(
        "PNW v{}: {} probes, {} IBL probes, voronoi v{} {} cells, {} tetrahedra, {} vertices, {} instance lists (stride {}), {} vertex references, {} instance offsets",
        net.version,
        net.probes.len(),
        net.ibl_probes.len(),
        net.voronoi_version,
        net.cells.len(),
        net.tetrahedra.len(),
        net.vertices.len(),
        net.instance_lists.len(),
        net.instance_lists.first().map_or(0, Vec::len),
        net.vertex_references.len(),
        net.instance_offsets.len()
    );
    // Optional 3rd arg: the map's .bwm, to check per-instance offsets against vertex counts.
    if let Some(bwm) = a.get(3) {
        let w = void_formats::bwm::WorldModel::parse(&ar.read("File", bwm)?)
            .map_err(|e| e.to_string())?;
        let mut inst_vertices = vec![0u32; w.instances.len()];
        for b in &w.batches {
            for i in b.first_instance..b.end_instance {
                inst_vertices[i as usize] = w.surfaces[b.surface as usize].vertex_count;
            }
        }
        let mut mismatches = 0;
        for (i, o) in net.instance_offsets.iter().enumerate() {
            let end = net
                .instance_offsets
                .get(i + 1)
                .copied()
                .unwrap_or(net.vertex_references.len() as u32);
            if end - o != inst_vertices.get(i).copied().unwrap_or(u32::MAX) {
                mismatches += 1;
            }
        }
        let max_ref = net.vertex_references.iter().max().copied().unwrap_or(0);
        println!(
            "bwm: {} instances; offset steps matching instance vertex counts: {}/{}; max reference {max_ref} (< {} tetrahedra)",
            w.instances.len(),
            net.instance_offsets.len() - mismatches,
            net.instance_offsets.len(),
            net.tetrahedra.len()
        );
    }
    let (mut ok, mut b, mut fail) = (0, 0, Vec::new());
    let (mut occ_probes, mut occ_total) = (0usize, 0usize);
    for (i, name) in net.probes.iter().enumerate() {
        let res = format!("{name}.bprobe");
        match ar
            .read("EnvProbe", &res)
            .and_then(|d| EnvProbe::parse(&d).map_err(|e| e.to_string()))
        {
            Ok(p) => {
                ok += 1;
                occ_probes += usize::from(p.state_a.occluders > 0);
                occ_total += p.state_a.occluders;
                b += usize::from(p.state_b.is_some());
                if i < 2 {
                    let s = &p.state_a;
                    println!(
                        "  {name}: pos {:?} bounce sun {:?} sky {:?} skyOcc {:?} shStatic0 {:?} samples {} emissive {} neighbours {} cube '{}'",
                        p.shooting_position,
                        p.bounce_sun,
                        p.bounce_sky,
                        s.sky_occlusion_factor,
                        s.sh_static[0],
                        s.samples.len(),
                        s.emissive.len(),
                        s.neighbors.len(),
                        s.cube_map
                    );
                }
            }
            Err(e) => fail.push(format!("{name}: {e}")),
        }
    }
    println!(
        "probes parsed {ok}/{} (state B in {b}); failures {}",
        net.probes.len(),
        fail.len()
    );
    println!("probes with occluders: {occ_probes}, occluder records: {occ_total}");
    for f in fail.iter().take(5) {
        println!("  {f}");
    }
    Ok(())
}
