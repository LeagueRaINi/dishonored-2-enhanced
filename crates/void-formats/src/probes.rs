//! Environment probes: the probe network (`.probesnetwork`, magic `PNW`) and per-probe
//! results (`.bprobe`, magic `PRB`).
//!
//! Layouts recovered from the executable (exe `b671f824...6180`, O):
//! - `PNW` reader `sub_1402B15E0` ("arkEnvironmentProbesNetwork"), Voronoi block reader
//!   `sub_1403843F0` ("arkVoronoi: %d cells, %d tetrahedrons, %d vertices") with
//!   `sub_140384870` (cell), `sub_140384BB0` (cell face), `sub_140384D70` (144-byte
//!   record), and `sub_140290C10` (per-vertex tetrahedron references).
//! - `PRB` serializer `sub_1402BAA80` / `sub_1402BA2B0` (results) / `sub_1402BA900`
//!   (sample) / `sub_1402BA9D0` (Voronoi info) / `sub_1402BA1E0` (occluder); field
//!   names from the engine reflection tables (`arkEnvProbe`, `arkEnvProbe::results_t`).
//!
//! Only the versions present in the shipped files are accepted (PNW 7, Voronoi block as
//! read; PRB 14.1); anything else is an error rather than a guess.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn err<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.d.len())
            .ok_or_else(|| Error(format!("truncated at {} (+{n})", self.p)))?;
        let s = &self.d[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().expect("2")))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().expect("4")))
    }
    fn i32(&mut self) -> Result<i32, Error> {
        Ok(self.u32()? as i32)
    }
    fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn vec3(&mut self) -> Result<[f32; 3], Error> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }
    fn floats<const N: usize>(&mut self) -> Result<[f32; N], Error> {
        let mut out = [0f32; N];
        for v in &mut out {
            *v = self.f32()?;
        }
        Ok(out)
    }
    /// Count with a bound on the bytes it can describe (no huge allocations).
    fn count(&mut self, min_item_bytes: usize) -> Result<usize, Error> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min_item_bytes.max(1)) > self.d.len() - self.p {
            return err(format!(
                "count {n} at {} exceeds the remaining data",
                self.p - 4
            ));
        }
        Ok(n)
    }
    /// `u32 len` + bytes (engine `idStr` serialization; len <= 0 = empty).
    fn string(&mut self) -> Result<String, Error> {
        let n = self.i32()?;
        if n <= 0 {
            return Ok(String::new());
        }
        let b = self.bytes(n as usize)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn done(&self) -> Result<(), Error> {
        if self.p == self.d.len() {
            Ok(())
        } else {
            err(format!(
                "{} trailing bytes at {}",
                self.d.len() - self.p,
                self.p
            ))
        }
    }
}

/// `indirectLighting/tetrahedron_t`: 4 vertex indices and 4 probe indices (u16 pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tetrahedron {
    pub vertices: [u32; 4],
    pub probes: [u16; 4],
}

/// `indirectLighting/tetrahedronVertex_t`: position and 4 u8 probe weights (/255).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedronVertex {
    pub position: [f32; 3],
    pub weights: [u8; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoronoiCell {
    pub probe: String,
    pub faces: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeNetwork {
    pub version: u8,
    /// Probe resource names (EnvProbe), index = probe index used by tetrahedra.
    pub probes: Vec<String>,
    /// IBL probe resource names.
    pub ibl_probes: Vec<String>,
    pub voronoi_version: u32,
    pub cells: Vec<VoronoiCell>,
    pub tetrahedra: Vec<Tetrahedron>,
    pub vertices: Vec<TetrahedronVertex>,
    /// One entry per static-world instance (count = `.bwm` instances in tutorial_p),
    /// `stride` (<= 4) i32 each, -1 padded; semantics U.
    pub instance_lists: Vec<Vec<i32>>,
    /// Per-vertex tetrahedron references for the static world (`indexTetrahedron`).
    pub vertex_references: Vec<u32>,
    /// Per static-world instance: offset of its first vertex in `vertex_references`.
    pub instance_offsets: Vec<u32>,
}

impl ProbeNetwork {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader { d: data, p: 0 };
        let magic = r.u32()?;
        if magic & 0x00FF_FFFF != 0x0057_4E50 {
            return err("not a PNW probe network");
        }
        let version = (magic >> 24) as u8;
        if version != 7 {
            return err(format!("PNW version {version} not supported (7 observed)"));
        }
        let n = r.count(4)?;
        let probes = (0..n).map(|_| r.string()).collect::<Result<Vec<_>, _>>()?;
        let n = r.count(4)?;
        let ibl_probes = (0..n).map(|_| r.string()).collect::<Result<Vec<_>, _>>()?;
        // Voronoi block (sub_1403843F0).
        let vv = r.u32()?;
        if !(7..=9).contains(&vv) {
            return err(format!("Voronoi version {vv} not supported"));
        }
        let _flag = r.u8()?;
        let ncells = r.count(4)?;
        let mut cells = Vec::with_capacity(ncells);
        for _ in 0..ncells {
            let probe = r.string()?;
            r.bytes(4 + 4 + 1 + 4 + 4)?;
            let _neighbour_cell = r.i32()?;
            r.bytes(4)?;
            let nfaces = r.count(4)?;
            for _ in 0..nfaces {
                r.bytes(4 + 4 + 12 + 12 + 12 + 1 + 1)?;
                let nv = r.count(12)?;
                r.bytes(nv * 12)?;
            }
            let n = r.count(4)?;
            r.bytes(n * 4)?;
            r.bytes(24)?;
            cells.push(VoronoiCell {
                probe,
                faces: nfaces,
            });
        }
        let n144 = r.count(56)?;
        for _ in 0..n144 {
            r.bytes(4 + 4)?;
            if vv < 8 {
                r.bytes(2)?;
            }
            if vv < 9 {
                r.bytes(16)?;
            }
            r.bytes(48)?;
            if vv < 9 {
                r.bytes(48 + 48 + 12 + 12 + 12 + 4)?;
            }
        }
        let tb = r.u32()? as usize;
        if !tb.is_multiple_of(24) {
            return err(format!("tetrahedron bytes {tb} not a multiple of 24"));
        }
        let mut tetrahedra = Vec::with_capacity(tb / 24);
        for _ in 0..tb / 24 {
            let v = [r.u32()?, r.u32()?, r.u32()?, r.u32()?];
            let p0 = r.u32()?;
            let p1 = r.u32()?;
            tetrahedra.push(Tetrahedron {
                vertices: v,
                probes: [p0 as u16, (p0 >> 16) as u16, p1 as u16, (p1 >> 16) as u16],
            });
        }
        let vb = r.u32()? as usize;
        if !vb.is_multiple_of(16) {
            return err(format!("vertex bytes {vb} not a multiple of 16"));
        }
        let mut vertices = Vec::with_capacity(vb / 16);
        for _ in 0..vb / 16 {
            let position = r.vec3()?;
            let w = r.u32()?;
            vertices.push(TetrahedronVertex {
                position,
                weights: w.to_le_bytes(),
            });
        }
        // Per-probe lists (sub_1402B15E0 after the Voronoi block).
        let nlists = r.count(0)?;
        let stride = r.u32()? as usize;
        if stride > 4 {
            return err(format!("list stride {stride} > 4"));
        }
        let mut instance_lists = Vec::with_capacity(nlists);
        for _ in 0..nlists {
            instance_lists.push((0..stride).map(|_| r.i32()).collect::<Result<_, _>>()?);
        }
        // Observed order (tutorial_p, eotw_p): per-vertex references, a zero byte, then
        // per-instance offsets into the references (`bufferPerVertexReferenceOffsets`).
        let rb = r.u32()? as usize;
        if !rb.is_multiple_of(4) {
            return err(format!("reference bytes {rb} not a multiple of 4"));
        }
        let vertex_references = (0..rb / 4)
            .map(|_| r.u32())
            .collect::<Result<Vec<_>, _>>()?;
        if r.u8()? != 0 {
            return err("PNW flag set (engine returns error 9)");
        }
        let ob = r.u32()? as usize;
        if !ob.is_multiple_of(4) {
            return err(format!("offset bytes {ob} not a multiple of 4"));
        }
        let instance_offsets = (0..ob / 4)
            .map(|_| r.u32())
            .collect::<Result<Vec<_>, _>>()?;
        r.done()?;
        Ok(Self {
            version,
            probes,
            ibl_probes,
            voronoi_version: vv,
            cells,
            tetrahedra,
            vertices,
            instance_lists,
            vertex_references,
            instance_offsets,
        })
    }
}

/// `arkEnvProbe::results_t::sample_t` as serialized (axes are not stored).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSample {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub radius_minor: f32,
    pub radius_major: f32,
    /// Packed sqrt(albedo) RGB in the top 3 bytes, sqrt(F0) in the low byte
    /// (`updateprobe.decl`).
    pub albedo_f0: u32,
    pub sun_illuminance_factor: f32,
    pub sh_factor: f32,
    pub occluder_index: u16,
    pub attached_probe_index: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmissiveSurface {
    pub material: String,
    pub sh: [f32; 9],
    pub importance: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeighborProbe {
    pub probe: String,
    pub directly_visible: bool,
    pub voronoi_index: i32,
    pub distance: f32,
    pub solid_angle: f32,
    pub direction: [f32; 3],
    pub is_transition: bool,
}

/// `arkEnvProbe::results_t` (one state).
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeResults {
    pub distance_mean: f32,
    pub distance_harmonic_mean: f32,
    pub distance_min: f32,
    pub distance_max: f32,
    pub ls_bbox: [[f32; 3]; 2],
    pub static_lights_illuminance: [f32; 3],
    pub sky_occlusion_factor: f32,
    pub sun_irradiance_factor: f32,
    pub illuminance_estimate: [f32; 3],
    pub pixel_counts: [u32; 9],
    pub sh_static: [[f32; 3]; 9],
    pub sh_static_sun_factor: [[f32; 3]; 9],
    pub sh_static_sun_factor_filtered: [[f32; 3]; 9],
    pub sh_sky_occlusion: [f32; 9],
    pub sh_sky_occlusion_filtered: [f32; 9],
    /// Multi-bounce SH (`*_Initial`, results +1536/+1644/+1752): static, Sun and sky (RGB).
    /// Added x `r_ILProbeUpdate_ContributionFactorMultiBounces` at load (`sub_140291430`).
    pub sh_initial: [[[f32; 3]; 9]; 3],
    pub samples: Vec<ProbeSample>,
    pub emissive: Vec<EmissiveSurface>,
    pub neighbors: Vec<NeighborProbe>,
    pub voronoi_infos: usize,
    pub occluders: usize,
    pub cube_map: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvProbe {
    pub version: (u32, u32),
    pub shooting_position: [f32; 3],
    pub zone_group: u32,
    pub is_transition: bool,
    pub bounce_sun: [f32; 3],
    pub bounce_sky: [f32; 3],
    pub bounce_dynamic: [f32; 3],
    pub bounce_static: [f32; 3],
    pub ao_sky: f32,
    pub ao_global: f32,
    pub ambient_factor: [f32; 3],
    pub used_for_ibl: bool,
    pub use_virtual_planes: bool,
    pub ibl_intensity: f32,
    pub state_a: ProbeResults,
    pub state_b: Option<ProbeResults>,
}

fn sh3(r: &mut Reader) -> Result<[[f32; 3]; 9], Error> {
    let mut out = [[0f32; 3]; 9];
    for c in &mut out {
        *c = r.vec3()?;
    }
    Ok(out)
}

fn results(r: &mut Reader) -> Result<ProbeResults, Error> {
    let distance_mean = r.f32()?;
    let distance_harmonic_mean = r.f32()?;
    let distance_min = r.f32()?;
    let distance_max = r.f32()?;
    let ls_bbox = [r.vec3()?, r.vec3()?];
    let static_lights_illuminance = r.vec3()?;
    let sky_occlusion_factor = r.f32()?;
    let sun_irradiance_factor = r.f32()?;
    let illuminance_estimate = r.vec3()?;
    let mut pixel_counts = [0u32; 9];
    for c in &mut pixel_counts {
        *c = r.u32()?;
    }
    let sh_static = sh3(r)?;
    let sh_static_sun_factor = sh3(r)?;
    let sh_static_sun_factor_filtered = sh3(r)?;
    let sh_sky_occlusion = r.floats::<9>()?;
    let sh_sky_occlusion_filtered = r.floats::<9>()?;
    r.bytes(1024)?; // m_SHEnvironmentDistance
    let sh_initial = [sh3(r)?, sh3(r)?, sh3(r)?];
    r.bytes(12)?; // distanceSample min/max/mean
    let ns = r.u32()?;
    if ns != 128 {
        return err(format!("sample count {ns} (128 expected)"));
    }
    let mut samples = Vec::with_capacity(128);
    for _ in 0..128 {
        samples.push(ProbeSample {
            position: r.vec3()?,
            normal: r.vec3()?,
            radius_minor: r.f32()?,
            radius_major: r.f32()?,
            albedo_f0: r.u32()?,
            sun_illuminance_factor: r.f32()?,
            sh_factor: r.f32()?,
            occluder_index: r.u16()?,
            attached_probe_index: r.u16()?,
        });
    }
    let n = r.count(40)?;
    let mut emissive = Vec::with_capacity(n);
    for _ in 0..n {
        emissive.push(EmissiveSurface {
            material: r.string()?,
            sh: r.floats::<9>()?,
            importance: r.f32()?,
        });
    }
    r.bytes(8)?; // nearest / farthest neighbour distance
    let n = r.count(30)?;
    let mut neighbors = Vec::with_capacity(n);
    for _ in 0..n {
        neighbors.push(NeighborProbe {
            probe: r.string()?,
            directly_visible: r.u8()? != 0,
            voronoi_index: r.i32()?,
            distance: r.f32()?,
            solid_angle: r.f32()?,
            direction: r.vec3()?,
            is_transition: r.u8()? != 0,
        });
    }
    let voronoi_infos = r.count(66)?;
    for _ in 0..voronoi_infos {
        r.string()?;
        r.bytes(4 + 12 + 12 + 4 + 36 + 1 + 1)?;
    }
    let occluders = r.count(424)?;
    for _ in 0..occluders {
        r.string()?;
        r.bytes(108 * 3 + 36 * 2 + 4 + 8 + 4)?;
    }
    let cube_map = r.string()?;
    r.bytes(96)?; // virtual planes
    Ok(ProbeResults {
        distance_mean,
        distance_harmonic_mean,
        distance_min,
        distance_max,
        ls_bbox,
        static_lights_illuminance,
        sky_occlusion_factor,
        sun_irradiance_factor,
        illuminance_estimate,
        pixel_counts,
        sh_static,
        sh_static_sun_factor,
        sh_static_sun_factor_filtered,
        sh_sky_occlusion,
        sh_sky_occlusion_filtered,
        sh_initial,
        samples,
        emissive,
        neighbors,
        voronoi_infos,
        occluders,
        cube_map,
    })
}

impl EnvProbe {
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut r = Reader { d: data, p: 0 };
        if r.bytes(4)? != b"PRB\x0b" {
            return err("not a PRB probe");
        }
        let version = (r.u32()?, r.u32()?);
        if version != (14, 1) {
            return err(format!(
                "PRB version {version:?} not supported (14.1 observed)"
            ));
        }
        let shooting_position = r.vec3()?;
        let zone_group = r.u32()?;
        let is_transition = r.u8()? != 0;
        let bounce_sun = r.vec3()?;
        let bounce_sky = r.vec3()?;
        let bounce_dynamic = r.vec3()?;
        let bounce_static = r.vec3()?;
        let ao_sky = r.f32()?;
        let ao_global = r.f32()?;
        let ambient_factor = r.vec3()?;
        let used_for_ibl = r.u8()? != 0;
        let use_virtual_planes = r.u8()? != 0;
        let ibl_intensity = r.f32()?;
        let state_a = results(&mut r)?;
        let state_b = if r.u8()? != 0 {
            Some(results(&mut r)?)
        } else {
            None
        };
        r.done()?;
        Ok(Self {
            version,
            shooting_position,
            zone_group,
            is_transition,
            bounce_sun,
            bounce_sky,
            bounce_dynamic,
            bounce_static,
            ao_sky,
            ao_global,
            ambient_factor,
            used_for_ibl,
            use_virtual_planes,
            ibl_intensity,
            state_a,
            state_b,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_magic_and_unsupported_versions() {
        assert!(ProbeNetwork::parse(b"XYZ\x07").is_err());
        let mut v6 = b"PNW\x06".to_vec();
        v6.extend_from_slice(&[0; 16]);
        assert!(
            ProbeNetwork::parse(&v6)
                .unwrap_err()
                .0
                .contains("version 6")
        );
        let mut prb = b"PRB\x0b".to_vec();
        prb.extend_from_slice(&13u32.to_le_bytes());
        prb.extend_from_slice(&1u32.to_le_bytes());
        assert!(EnvProbe::parse(&prb).unwrap_err().0.contains("(13, 1)"));
    }

    #[test]
    fn minimal_network_round_trip_and_truncation() {
        // PNW 7: 1 probe "p", 0 IBL, voronoi v9 (flag 0, 0 cells, 0 records),
        // 1 tetrahedron, 1 vertex, 0 instance lists (stride 4), 2 refs, flag 0, 1 offset.
        let mut d = b"PNW\x07".to_vec();
        let u = |d: &mut Vec<u8>, v: u32| d.extend_from_slice(&v.to_le_bytes());
        u(&mut d, 1);
        u(&mut d, 1);
        d.push(b'p');
        u(&mut d, 0);
        u(&mut d, 9);
        d.push(0);
        u(&mut d, 0);
        u(&mut d, 0);
        u(&mut d, 24);
        for v in [0, 0, 0, 0, 0x0001_0000, 0x0003_0002] {
            u(&mut d, v);
        }
        u(&mut d, 16);
        for v in [1.0f32, 2.0, 3.0] {
            u(&mut d, v.to_bits());
        }
        u(&mut d, 0x04_03_02_01);
        u(&mut d, 0);
        u(&mut d, 4);
        u(&mut d, 8);
        u(&mut d, 0);
        u(&mut d, 0);
        d.push(0);
        u(&mut d, 4);
        u(&mut d, 0);
        let n = ProbeNetwork::parse(&d).unwrap();
        assert_eq!(n.probes, vec!["p".to_owned()]);
        assert_eq!(n.tetrahedra[0].probes, [0, 1, 2, 3]);
        assert_eq!(n.vertices[0].weights, [1, 2, 3, 4]);
        assert_eq!(n.vertex_references, vec![0, 0]);
        assert_eq!(n.instance_offsets, vec![0]);
        // Any truncation or extra byte is an error.
        assert!(ProbeNetwork::parse(&d[..d.len() - 1]).is_err());
        let mut extra = d.clone();
        extra.push(0);
        assert!(ProbeNetwork::parse(&extra).is_err());
    }
}
