//! Helpers shared by the offline tools (`rt_spike`, `photo`): report text, the run log that
//! matches GPU driver resets to runs, input parsing and the DXC location.

use std::path::{Path, PathBuf};

use crate::dxc;

/// Text report printed while it is built and written at the end.
#[derive(Default)]
pub struct Report(pub String);

impl Report {
    pub fn line(&mut self, s: impl AsRef<str>) {
        println!("{}", s.as_ref());
        self.0.push_str(s.as_ref());
        self.0.push('\n');
    }
}

/// Appends a line to the run log (run start/end, to match driver resets to runs).
pub fn run_log(path: &Path, line: &str) {
    use std::io::Write as _;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{t:.3} pid {} {line}", std::process::id());
    }
}

/// `WxH`.
pub fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| format!("size {s}: expected WxH"))?;
    let (w, h): (u32, u32) = (
        w.parse().map_err(|_| format!("size {s}"))?,
        h.parse().map_err(|_| format!("size {s}"))?,
    );
    if w == 0 || h == 0 {
        return Err(format!("size {s}: zero extent"));
    }
    Ok((w, h))
}

/// Little-endian f32 values of a raw dump.
pub fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// RGB of a raw `rgba16f` dump.
pub fn read_rgba16f(path: &Path) -> Result<Vec<[f32; 3]>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b.as_chunks::<8>()
        .0
        .iter()
        .map(|p| {
            let h = |k: usize| void_formats::bwm::f16_to_f32(u16::from_le_bytes([p[k], p[k + 1]]));
            [h(0), h(2), h(4)]
        })
        .collect())
}

/// DXC directory: the argument, else env `VOID_DXC_DIR`, else the newest Windows SDK.
pub fn dxc_dir(arg: Option<&Path>) -> Result<PathBuf, String> {
    arg.map(Path::to_path_buf)
        .or_else(|| std::env::var_os("VOID_DXC_DIR").map(PathBuf::from))
        .or_else(dxc::find_windows_sdk)
        .ok_or_else(|| "no DXC: pass --dxc <dir with dxcompiler.dll and dxil.dll>".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("3440x1440"), Ok((3440, 1440)));
        assert!(parse_size("0x5").is_err());
        assert!(parse_size("12").is_err());
    }
}
