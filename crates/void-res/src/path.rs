//! Maps resource names onto relative output paths without escaping the
//! output directory. Names that cannot be represented safely are rejected
//! rather than rewritten, so extraction never silently renames content.

use std::path::PathBuf;

use crate::Error;

const RESERVED: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

pub fn safe_relative_path(name: &str) -> Result<PathBuf, Error> {
    let fail = |reason| Error::UnsafePath {
        name: name.to_owned(),
        reason,
    };
    if name.is_empty() {
        return Err(fail("empty name"));
    }
    let mut out = PathBuf::new();
    for part in name.split('/') {
        if part.is_empty() {
            return Err(fail("empty component or absolute path"));
        }
        if part == "." || part == ".." {
            return Err(fail("relative component"));
        }
        if part
            .chars()
            .any(|c| c.is_control() || matches!(c, '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*'))
        {
            return Err(fail("character not allowed in Windows paths"));
        }
        if part.ends_with('.') || part.ends_with(' ') {
            return Err(fail("component ends with dot or space"));
        }
        let stem = part.split('.').next().unwrap_or(part);
        if RESERVED.iter().any(|r| r.eq_ignore_ascii_case(stem)) {
            return Err(fail("reserved device name"));
        }
        out.push(part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_names() {
        let p = safe_relative_path("generated/decls/renderparm/time.decl").unwrap();
        assert_eq!(
            p,
            PathBuf::from("generated").join("decls/renderparm/time.decl")
        );
        // Observed real names contain these characters.
        assert!(
            safe_relative_path("a/audio_area_volume_crossbow_5.cvd._100_100_100.0.hkx").is_ok()
        );
        assert!(safe_relative_path("a/b$c#d.decl").is_ok());
    }

    #[test]
    fn rejects_escapes_and_windows_hazards() {
        for bad in [
            "",
            "/abs",
            "a//b",
            "../x",
            "a/./b",
            "a/../../b",
            "C:/x",
            "a\\b",
            "a/b:stream",
            "a/con.txt",
            "a/LPT1",
            "a/b.",
            "a/b ",
            "a/\u{1}",
        ] {
            assert!(safe_relative_path(bad).is_err(), "{bad:?} accepted");
        }
    }
}
