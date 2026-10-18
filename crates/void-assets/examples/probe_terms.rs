//! Magnitude of each probe SH term (DC luminance) across a map's probes, using the live
//! sun colour factor of RX009 (parallelLightColor) and the eotw env sky.
//! usage: probe_terms <game root> <network resource>
use std::path::Path;
use void_formats::probes::{EnvProbe, ProbeNetwork};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ar = void_assets::Archive::open(Path::new(&a[1]))?;
    let net = ProbeNetwork::parse(&ar.read("File", &a[2])?).map_err(|e| e.to_string())?;
    let sun = [123.9f32, 98.4, 67.2];
    let sky = void_assets::probes::sky_zh([0.0; 3], [37.7451, 52.4118, 55.0]);
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let mut rows = Vec::new();
    let pick: Vec<usize> = a[3..].iter().filter_map(|v| v.parse().ok()).collect();
    for (pi, n) in net.probes.iter().enumerate() {
        let p = EnvProbe::parse(&ar.read("EnvProbe", &format!("{n}.bprobe"))?)
            .map_err(|e| e.to_string())?;
        let r = &p.state_a;
        let st = lum(r.sh_static[0]) * p.ao_global;
        let su = lum(std::array::from_fn(|k| {
            r.sh_static_sun_factor_filtered[0][k] * sun[k]
        })) * p.ao_global;
        let zh =
            sky.map(|z| std::array::from_fn::<f32, 3, _>(|k| z[k] * p.bounce_sky[k] * p.ao_sky));
        let occ: [[f32; 3]; 9] = r.sh_sky_occlusion_filtered.map(|v| [v; 3]);
        let sk = lum(void_assets::probes::zh_sh_product_sky(&zh, &occ)[0]);
        let amb = lum(p.ambient_factor) * 3.5449;
        let dyn_samples = r.samples.iter().filter(|s| s.albedo_f0 >> 8 != 0).count();
        if pick.contains(&pi) {
            let init: Vec<f32> = r.sh_initial.iter().map(|sh| lum(sh[0])).collect();
            println!(
                "probe {pi} {n}: pos {:?} ao sky/global {}/{} bounce static {:?} sun {:?} sky {:?} ambient {:?} | DC static {:.4} sunSH {:.5} skyOcc {:.4} initial {:?} skyOccFactor {} sunIrrFactor {} dist mean {}",
                p.shooting_position,
                p.ao_sky,
                p.ao_global,
                p.bounce_static,
                p.bounce_sun,
                p.bounce_sky,
                p.ambient_factor,
                lum(r.sh_static[0]),
                lum(r.sh_static_sun_factor_filtered[0]),
                r.sh_sky_occlusion_filtered[0],
                init,
                r.sky_occlusion_factor,
                r.sun_irradiance_factor,
                r.distance_mean
            );
        }
        let b = p.state_b.as_ref().map(|b| {
            let su = lum(std::array::from_fn(|k| {
                b.sh_static_sun_factor_filtered[0][k] * sun[k]
            })) * p.ao_global;
            let occ: [[f32; 3]; 9] = b.sh_sky_occlusion_filtered.map(|v| [v; 3]);
            (
                su,
                lum(void_assets::probes::zh_sh_product_sky(&zh, &occ)[0]),
            )
        });
        rows.push((
            st,
            su,
            sk,
            amb,
            p.shooting_position,
            dyn_samples,
            p.is_transition,
            b,
        ));
    }
    let med = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "{} probes: median DC luminance static {:.4} sun {:.4} sky {:.4} ambient {:.4}",
        rows.len(),
        med(rows.iter().map(|r| r.0).collect()),
        med(rows.iter().map(|r| r.1).collect()),
        med(rows.iter().map(|r| r.2).collect()),
        med(rows.iter().map(|r| r.3).collect())
    );
    // Probes near the sawmill arches (RX009 camera ~ (-33.7, 181.7, 10.5), looking +x/-y).
    let c = [-20.0f32, 170.0, 8.0];
    let mut near: Vec<_> = rows
        .iter()
        .filter(|r| {
            let d: f32 = (0..3).map(|k| (r.4[k] - c[k]).powi(2)).sum::<f32>().sqrt();
            d < 25.0
        })
        .collect();
    near.sort_by(|a, b| (a.1 + a.0 + a.2).total_cmp(&(b.1 + b.0 + b.2)));
    for r in near.iter().take(8) {
        println!(
            "  pos {:?}: static {:.4} sun {:.4} sky {:.4} ambient {:.4} samples-with-albedo {} transition {} state B (sun, sky) {:?}",
            r.4, r.0, r.1, r.2, r.3, r.5, r.6, r.7
        );
    }
    println!(
        "transition probes {}, with state B {}",
        rows.iter().filter(|r| r.6).count(),
        rows.iter().filter(|r| r.7.is_some()).count()
    );
    Ok(())
}
