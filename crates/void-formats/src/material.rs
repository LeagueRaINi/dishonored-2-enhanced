//! Material declarations (`generated/decls/material/<name>.material.decl`).
//!
//! Structure observed in shipped decls: one top-level block of
//! newline-terminated `key value` lines, `key { n, n, ... }` vectors, and
//! nested blocks such as `options { ... }`. Values are kept verbatim; the
//! accessors below only classify, they do not assign rendering semantics.

use crate::Error;
use crate::decl::{Tok, Token, tokenize};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Text(String),
    Vector(Vec<f32>),
    Block(Vec<(String, Value)>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub entries: Vec<(String, Value)>,
}

fn invalid(line: u32, detail: String) -> Error {
    Error::Invalid {
        field: "material decl",
        detail: format!("line {line}: {detail}"),
    }
}

struct Parser<'a> {
    t: &'a [Token],
    i: usize,
}

impl Parser<'_> {
    fn block(&mut self, open_line: u32) -> Result<Vec<(String, Value)>, Error> {
        let mut entries = Vec::new();
        loop {
            let Some(tok) = self.t.get(self.i) else {
                return Err(invalid(open_line, "unterminated block".into()));
            };
            self.i += 1;
            let key = match &tok.tok {
                Tok::Close => return Ok(entries),
                Tok::Word(w) | Tok::Str(w) => w.clone(),
                other => return Err(invalid(tok.line, format!("expected key, found {other:?}"))),
            };
            let line = tok.line;
            let value = match self.t.get(self.i).map(|t| &t.tok) {
                Some(Tok::Open) => {
                    self.i += 1;
                    self.braced(line)?
                }
                _ => Value::Text(self.rest_of_line(line)?),
            };
            entries.push((key, value));
        }
    }

    /// A brace group: a numeric vector, or a nested block.
    fn braced(&mut self, line: u32) -> Result<Value, Error> {
        let start = self.i;
        let mut nums = Vec::new();
        while let Some(t) = self.t.get(self.i) {
            match &t.tok {
                Tok::Comma => {}
                Tok::Word(w) => match w.parse::<f32>() {
                    Ok(v) => nums.push(v),
                    Err(_) => break,
                },
                Tok::Close => {
                    self.i += 1;
                    return Ok(Value::Vector(nums));
                }
                _ => break,
            }
            self.i += 1;
        }
        self.i = start;
        Ok(Value::Block(self.block(line)?))
    }

    /// Tokens on the key's line, joined; parentheses keep image programs intact.
    fn rest_of_line(&mut self, line: u32) -> Result<String, Error> {
        let mut out = String::new();
        let mut depth = 0i32;
        while let Some(t) = self.t.get(self.i) {
            if depth == 0 && (t.line != line || t.tok == Tok::Close || t.tok == Tok::Open) {
                break;
            }
            let piece = match &t.tok {
                Tok::Word(w) | Tok::Str(w) => w.as_str(),
                Tok::LParen => {
                    depth += 1;
                    "("
                }
                Tok::RParen => {
                    depth -= 1;
                    ")"
                }
                Tok::Comma => ",",
                Tok::Eq => "=",
                Tok::Semi => ";",
                Tok::Open | Tok::Close => {
                    return Err(invalid(t.line, "brace inside image program".into()));
                }
            };
            if !out.is_empty() && !matches!(piece, "(" | ")" | ",") && !out.ends_with('(') {
                out.push(' ');
            }
            out.push_str(piece);
            self.i += 1;
        }
        if depth != 0 {
            return Err(invalid(line, "unbalanced parentheses".into()));
        }
        Ok(out)
    }
}

impl Material {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let tokens = tokenize(text)?;
        let first = tokens
            .first()
            .ok_or_else(|| invalid(1, "empty material".into()))?;
        if first.tok != Tok::Open {
            return Err(invalid(first.line, "expected '{'".into()));
        }
        let mut p = Parser { t: &tokens, i: 1 };
        let entries = p.block(first.line)?;
        if p.i != tokens.len() {
            return Err(invalid(
                tokens[p.i].line,
                "content after material block".into(),
            ));
        }
        Ok(Self { entries })
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v)
    }

    pub fn text(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn vector(&self, key: &str) -> Option<&[f32]> {
        match self.get(key)? {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    pub fn program(&self) -> Option<&str> {
        self.text("mainProgram")
    }

    /// `options { key value }` entries.
    pub fn options(&self) -> impl Iterator<Item = (&str, &str)> {
        let block = match self.get("options") {
            Some(Value::Block(b)) => b.as_slice(),
            _ => &[],
        };
        block.iter().filter_map(|(k, v)| match v {
            Value::Text(t) => Some((k.as_str(), t.as_str())),
            _ => None,
        })
    }

    pub fn option(&self, key: &str) -> Option<&str> {
        self.options()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v)
    }

    /// Top-level texture slots: keys ending in `map` (case-insensitive) plus
    /// `skydepth`, or any value that names an image file or image program.
    /// Shipped decls reference textures both with and without extensions.
    pub fn image_slots(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().filter_map(|(k, v)| match v {
            Value::Text(t) if is_image_slot(k) || is_image_ref(t) => Some((k.as_str(), t.as_str())),
            _ => None,
        })
    }
}

pub fn is_image_slot(key: &str) -> bool {
    let l = key.to_ascii_lowercase();
    l.ends_with("map") || l == "skydepth"
}

pub fn is_image_ref(t: &str) -> bool {
    let l = t.to_ascii_lowercase();
    l.starts_with("ipr_")
        || [".tga", ".png", ".dds", ".bmp", ".jpg", ".bimage"]
            .iter()
            .any(|e| l.ends_with(e))
}

/// Archive name of the built image for a texture source path, e.g.
/// `models/a/b_d.tga` -> `generated/image/models/a/b_d.bimage7`.
/// Image programs (`ipr_...`) have no direct file and return `None`.
pub fn image_resource_name(source: &str) -> Option<String> {
    if source.to_ascii_lowercase().starts_with("ipr_") {
        return None;
    }
    let file = source.rsplit('/').next().unwrap_or(source);
    let stem = match file.rfind('.') {
        Some(dot) => &source[..source.len() - (file.len() - dot)],
        None => source,
    };
    Some(format!(
        "generated/image/{}.bimage7",
        stem.to_ascii_lowercase()
    ))
}

/// Archive name of a material decl referenced by `name` (e.g. from a BWM batch).
pub fn material_resource_name(name: &str) -> String {
    format!("generated/decls/material/{name}.decl")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
	m_PhysicsMaterial		"contactsystem/env.mud"
	options {
		hasBumpMap				1
		extralayer		2
	}
	mainProgram				arkDefault
	diffusemap				models/env/dirt_d.tga
	bumpmap					models/env/dirt_n.tga
	wardRoughness			{ 0.30967, 1, 0, 0 }
	layer2_scalebias		{ 2.5, 2.5, 0, 0 }
	occlusionmap ipr_constantcolor( 0.5, 0.5, 0, 0)
	glossmap	models/env/dirt_g
}"#;

    #[test]
    fn parses_shipped_shape() {
        let m = Material::parse(SAMPLE).unwrap();
        assert_eq!(m.program(), Some("arkDefault"));
        assert_eq!(m.option("extralayer"), Some("2"));
        assert_eq!(
            m.vector("wardRoughness"),
            Some(&[0.30967, 1.0, 0.0, 0.0][..])
        );
        assert_eq!(m.text("m_PhysicsMaterial"), Some("contactsystem/env.mud"));
        let slots: Vec<_> = m.image_slots().collect();
        assert_eq!(slots[0], ("diffusemap", "models/env/dirt_d.tga"));
        assert_eq!(
            slots[2],
            ("occlusionmap", "ipr_constantcolor(0.5, 0.5, 0, 0)")
        );
    }

    #[test]
    fn resource_names() {
        assert_eq!(
            image_resource_name("models/A/b_d.tga").as_deref(),
            Some("generated/image/models/a/b_d.bimage7")
        );
        assert_eq!(image_resource_name("ipr_constantcolor(1,1,1,1)"), None);
        assert_eq!(
            image_resource_name("models/a.b/c_d").as_deref(),
            Some("generated/image/models/a.b/c_d.bimage7")
        );
        assert_eq!(
            material_resource_name("models/x.material"),
            "generated/decls/material/models/x.material.decl"
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(Material::parse("diffusemap a.tga").is_err());
        assert!(Material::parse("{ a b").is_err());
        assert!(Material::parse("{ a f( }").is_err());
        assert!(Material::parse("{ a b } extra").is_err());
    }
}
