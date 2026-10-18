//! Environment decls (`generated/decls/env/<name>.decl`): a version, an `inherit { .. }`
//! list of parent env names, and a `renderParms { name value }` block whose values are a
//! scalar, a `{ a, b, c, d }` vector or a resource path (O: eotw `global_ext01`, 168
//! lines; the map selects its world env with `m_worldEnvironment` in `.entities`).
//! Values are kept as text; callers interpret them with the renderParm's declared type.

use crate::Error;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvDecl {
    pub inherit: Vec<String>,
    /// `renderParms` entries in file order (lower-case names as stored).
    pub render_parms: Vec<(String, String)>,
}

fn invalid(detail: String) -> Error {
    Error::Invalid {
        field: "env decl",
        detail,
    }
}

/// Body of the `name { ... }` block (outer braces excluded), or None if absent.
fn block<'a>(text: &'a str, name: &str) -> Result<Option<&'a str>, Error> {
    let mut search = 0;
    while let Some(i) = text[search..].find(name) {
        let start = search + i;
        search = start + name.len();
        let before_ok = start == 0
            || text[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace);
        let rest = &text[search..];
        let trimmed = rest.trim_start();
        if !before_ok || !trimmed.starts_with('{') {
            continue;
        }
        let open = search + (rest.len() - trimmed.len());
        let mut depth = 0usize;
        for (k, c) in text[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(Some(&text[open + 1..open + k]));
                    }
                }
                _ => {}
            }
        }
        return Err(invalid(format!("unterminated `{name}` block")));
    }
    Ok(None)
}

impl EnvDecl {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let mut out = Self::default();
        if let Some(b) = block(text, "inherit")? {
            out.inherit = b.split_whitespace().map(str::to_owned).collect();
        }
        if let Some(b) = block(text, "renderParms")? {
            for line in b.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with("//") {
                    continue;
                }
                let (k, v) = line
                    .split_once(char::is_whitespace)
                    .ok_or_else(|| invalid(format!("renderParms line without value: {line}")))?;
                out.render_parms
                    .push((k.to_ascii_lowercase(), v.trim().to_owned()));
            }
        }
        Ok(out)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        let n = name.to_ascii_lowercase();
        self.render_parms
            .iter()
            .rev()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| v.as_str())
    }
}

/// Parses a renderParm value: a scalar or `{ a, b, .. }` into floats.
pub fn parse_floats(v: &str) -> Option<Vec<f32>> {
    let v = v.trim().trim_start_matches('{').trim_end_matches('}');
    let out: Option<Vec<f32>> = v
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().ok())
        .collect();
    out.filter(|o| !o.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "{\n\tversion\t18\n\tinherit {\n\t\tdefault\n\t}\n\tblendDuration\t1.0000\n\trenderParms {\nocean/enable\t1\nwater/watercolor\t{ 0.49, 0.56, 0.62, 0.0 }\nwater/skycubemap\tmodels/campaign/eotw/eotw_oceancube.dds\n\t}\n\taudioParms {\nsoundEvent NULL\n\t}\n}";

    #[test]
    fn parses_inherit_and_render_parms() {
        let e = EnvDecl::parse(SAMPLE).unwrap();
        assert_eq!(e.inherit, vec!["default".to_owned()]);
        assert_eq!(e.get("Ocean/Enable"), Some("1"));
        assert_eq!(
            parse_floats(e.get("water/watercolor").unwrap()).unwrap(),
            vec![0.49, 0.56, 0.62, 0.0]
        );
        assert_eq!(
            e.get("water/skycubemap"),
            Some("models/campaign/eotw/eotw_oceancube.dds")
        );
        assert_eq!(e.get("audio"), None);
    }
}
