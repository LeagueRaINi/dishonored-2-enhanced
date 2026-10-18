//! Minimal checked reader for DXBC shader containers: the container checksum (used as the
//! shader id, as d2enhanced names its shader dumps) and the RDEF resource bindings. The
//! game's shaders keep RDEF with names like `ro_<shader>_<param>` (SRV) and
//! `rw_<shader>_<param>` (UAV) (d2enhanced NOTES), which identify passes.

/// Shader input type (`D3D_SHADER_INPUT_TYPE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindKind {
    CBuffer,
    Texture,
    Sampler,
    Uav,
    Buffer,
    Other(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub kind: BindKind,
    pub slot: u32,
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(o..o.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn cstr(b: &[u8], o: usize) -> Option<String> {
    let s = b.get(o..)?;
    let end = s.iter().take(256).position(|&c| c == 0)?;
    Some(String::from_utf8_lossy(&s[..end]).into_owned())
}

/// First 8 bytes of the container checksum as hex (stable shader id).
pub fn id(code: &[u8]) -> Option<String> {
    if code.get(0..4)? != b"DXBC" {
        return None;
    }
    Some(
        code.get(4..12)?
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

/// Chunk payloads by four-character code.
fn chunk<'a>(code: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    if code.get(0..4)? != b"DXBC" {
        return None;
    }
    let total = (u32_at(code, 24)? as usize).min(code.len());
    let count = u32_at(code, 28)? as usize;
    if count > 64 {
        return None;
    }
    for i in 0..count {
        let off = u32_at(code, 32 + 4 * i)? as usize;
        if code.get(off..off.checked_add(4)?)? == fourcc {
            let size = u32_at(code, off + 4)? as usize;
            let start = off + 8;
            let end = start.checked_add(size)?;
            if end <= total {
                return code.get(start..end);
            }
        }
    }
    None
}

/// Resource bindings from the RDEF chunk (empty if absent or malformed).
pub fn bindings(code: &[u8]) -> Vec<Binding> {
    let Some(r) = chunk(code, b"RDEF") else {
        return Vec::new();
    };
    let (Some(count), Some(off)) = (u32_at(r, 8), u32_at(r, 12)) else {
        return Vec::new();
    };
    // Version word: minor (u8), major (u8). SM 5.1 descriptors add space and id (40 B).
    let (minor, major) = (
        r.get(16).copied().unwrap_or(0),
        r.get(17).copied().unwrap_or(0),
    );
    let stride = if major > 5 || (major == 5 && minor >= 1) {
        40
    } else {
        32
    };
    let mut out = Vec::new();
    for i in 0..(count as usize).min(256) {
        let b = off as usize + i * stride;
        let (Some(name_off), Some(kind), Some(slot)) =
            (u32_at(r, b), u32_at(r, b + 4), u32_at(r, b + 20))
        else {
            break;
        };
        let Some(name) = cstr(r, name_off as usize) else {
            break;
        };
        let kind = match kind {
            0 => BindKind::CBuffer,
            2 => BindKind::Texture,
            3 => BindKind::Sampler,
            4 | 6 | 8 | 9 | 10 | 11 => BindKind::Uav,
            1 | 5 | 7 => BindKind::Buffer,
            k => BindKind::Other(k),
        };
        out.push(Binding { name, kind, slot });
    }
    out
}

/// Compact description: "t0:name u0:name b2:name" (samplers omitted).
pub fn describe(b: &[Binding]) -> String {
    let mut v: Vec<String> = b
        .iter()
        .filter(|x| x.kind != BindKind::Sampler)
        .map(|x| {
            let l = match x.kind {
                BindKind::CBuffer => 'b',
                BindKind::Uav => 'u',
                _ => 't',
            };
            format!("{l}{}:{}", x.slot, x.name)
        })
        .collect();
    v.sort();
    v.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a DXBC container with one RDEF chunk holding the given bindings.
    fn container(binds: &[(&str, u32, u32)], minor: u8, major: u8) -> Vec<u8> {
        let stride = if major == 5 && minor >= 1 { 40 } else { 32 };
        let header = 28usize;
        let table = header;
        let names_off = table + stride * binds.len();
        let mut names = Vec::new();
        let mut name_offs = Vec::new();
        for (n, _, _) in binds {
            name_offs.push((names_off + names.len()) as u32);
            names.extend_from_slice(n.as_bytes());
            names.push(0);
        }
        let mut rdef = vec![0u8; names_off];
        rdef[8..12].copy_from_slice(&(binds.len() as u32).to_le_bytes());
        rdef[12..16].copy_from_slice(&(table as u32).to_le_bytes());
        rdef[16] = minor;
        rdef[17] = major;
        for (i, (_, kind, slot)) in binds.iter().enumerate() {
            let b = table + i * stride;
            rdef[b..b + 4].copy_from_slice(&name_offs[i].to_le_bytes());
            rdef[b + 4..b + 8].copy_from_slice(&kind.to_le_bytes());
            rdef[b + 20..b + 24].copy_from_slice(&slot.to_le_bytes());
        }
        rdef.extend(names);
        let mut c = b"DXBC".to_vec();
        c.extend(hex_literal::hex!(
            "12 34 56 78 9a bc de f0 00 00 00 00 00 00 00 00"
        ));
        c.extend(1u32.to_le_bytes());
        let total = 32 + 4 + 8 + rdef.len();
        c.extend((total as u32).to_le_bytes());
        c.extend(1u32.to_le_bytes());
        c.extend(36u32.to_le_bytes());
        c.extend(b"RDEF");
        c.extend((rdef.len() as u32).to_le_bytes());
        c.extend(rdef);
        c
    }

    #[test]
    fn reads_id_and_bindings() {
        let c = container(
            &[
                ("ro_taa_history", 2, 0),
                ("samp", 3, 0),
                ("rw_taa_result", 4, 1),
                ("PerViewCB", 0, 1),
            ],
            0,
            5,
        );
        assert_eq!(id(&c).as_deref(), Some("123456789abcdef0"));
        let b = bindings(&c);
        assert_eq!(b.len(), 4);
        assert_eq!(b[2].kind, BindKind::Uav);
        assert_eq!(
            describe(&b),
            "b1:PerViewCB t0:ro_taa_history u1:rw_taa_result"
        );
    }

    #[test]
    fn sm51_stride_and_malformed_input() {
        let c = container(&[("a", 2, 3), ("b", 2, 4)], 1, 5);
        assert_eq!(describe(&bindings(&c)), "t3:a t4:b");
        assert!(bindings(&c[..40]).is_empty());
        assert!(bindings(b"not a shader").is_empty());
        assert!(id(b"DXB").is_none());
    }
}
