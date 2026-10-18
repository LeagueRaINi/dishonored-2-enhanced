//! Map `.entities` files and `entityDef` decls (docs/formats/entities.md).
//!
//! Grammar observed in shipped files: `Version 6`, then `entity { <type>
//! <name> { body } }` and `component { <type> <name> { body } }` blocks.
//! Bodies hold `key = value;`, `key = { .. }`, `key = ( a, b );`, and
//! `key { .. }` (no `=`) entries. Keys may contain brackets and quoted ids
//! (`item[0]`, `item_add["3H5CN2"]`). Arrays are sparse: `num = N;` plus
//! `item[i]` entries. Inheritance (`inherit = "x.def"`) deep-merges the
//! child's blocks over the parent's.

use crate::Error;
use crate::decl::{Tok, Token, tokenize};

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Scalar(String),
    Tuple(Vec<String>),
    Block(Vec<(String, Node)>),
}

impl Node {
    pub fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Block(b) => b
                .iter()
                .rev()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn path(&self, keys: &[&str]) -> Option<&Node> {
        keys.iter().try_fold(self, |n, k| n.get(k))
    }

    pub fn scalar(&self) -> Option<&str> {
        match self {
            Node::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn f32_at(&self, keys: &[&str]) -> Option<f32> {
        self.path(keys)?.scalar()?.parse().ok()
    }

    pub fn entries(&self) -> &[(String, Node)] {
        match self {
            Node::Block(b) => b,
            _ => &[],
        }
    }
}

fn invalid(line: u32, detail: String) -> Error {
    Error::Invalid {
        field: "decl tree",
        detail: format!("line {line}: {detail}"),
    }
}

struct P<'a> {
    t: &'a [Token],
    i: usize,
}

impl P<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i).map(|t| &t.tok)
    }

    fn line(&self) -> u32 {
        self.t.get(self.i).or(self.t.last()).map_or(0, |t| t.line)
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), Error> {
        if self.peek() == Some(want) {
            self.i += 1;
            Ok(())
        } else {
            Err(invalid(
                self.line(),
                format!("expected {what}, found {:?}", self.peek()),
            ))
        }
    }

    fn word(&mut self, what: &str) -> Result<String, Error> {
        match self.peek() {
            Some(Tok::Word(w)) | Some(Tok::Str(w)) => {
                let w = w.clone();
                self.i += 1;
                Ok(w)
            }
            other => Err(invalid(
                self.line(),
                format!("expected {what}, found {other:?}"),
            )),
        }
    }

    /// Entries until a closing brace (consumed).
    fn body(&mut self) -> Result<Vec<(String, Node)>, Error> {
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(invalid(self.line(), "unterminated block".into())),
                Some(Tok::Close) => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(Tok::Semi) => self.i += 1,
                _ => out.push(self.entry()?),
            }
        }
    }

    fn entry(&mut self) -> Result<(String, Node), Error> {
        let line = self.line();
        // Observed key forms: `name`, `item[0]`, `item_add["id"]`, and annotated
        // `"Type" name`, `"Type" "" "name"`. Bracketed keys are concatenated;
        // otherwise the last token is the name and earlier ones are annotation.
        let mut parts: Vec<(bool, String)> = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Word(w)) => parts.push((false, w.clone())),
                Some(Tok::Str(s)) => parts.push((true, s.clone())),
                Some(Tok::Eq) | Some(Tok::Open) => break,
                other => return Err(invalid(line, format!("bad key token {other:?}"))),
            }
            self.i += 1;
        }
        let bracketed = parts.iter().any(|(quoted, t)| !quoted && t.contains('['));
        let key = if bracketed {
            let start = parts
                .iter()
                .position(|(q, t)| !q && t.contains('['))
                .unwrap_or(0);
            parts[start..]
                .iter()
                .map(|(q, t)| if *q { format!("\"{t}\"") } else { t.clone() })
                .collect::<String>()
        } else {
            parts.last().map(|(_, t)| t.clone()).unwrap_or_default()
        };
        if key.is_empty() {
            return Err(invalid(line, "empty key".into()));
        }
        if self.peek() == Some(&Tok::Open) {
            self.i += 1;
            return Ok((key, Node::Block(self.body()?)));
        }
        self.expect(&Tok::Eq, "'='")?;
        let value = match self.peek() {
            Some(Tok::Open) => {
                self.i += 1;
                Node::Block(self.body()?)
            }
            Some(Tok::LParen) => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    match self.peek() {
                        Some(Tok::RParen) => {
                            self.i += 1;
                            break;
                        }
                        Some(Tok::Comma) => self.i += 1,
                        Some(Tok::Word(w)) | Some(Tok::Str(w)) => {
                            items.push(w.clone());
                            self.i += 1;
                        }
                        other => {
                            return Err(invalid(self.line(), format!("bad tuple token {other:?}")));
                        }
                    }
                }
                Node::Tuple(items)
            }
            _ => {
                let mut s = String::new();
                while let Some(tok) = self.peek() {
                    match tok {
                        Tok::Semi => break,
                        Tok::Word(w) | Tok::Str(w) => {
                            if !s.is_empty() {
                                s.push(' ');
                            }
                            s.push_str(w);
                        }
                        other => {
                            return Err(invalid(self.line(), format!("bad value token {other:?}")));
                        }
                    }
                    self.i += 1;
                }
                Node::Scalar(s)
            }
        };
        if self.peek() == Some(&Tok::Semi) {
            self.i += 1;
        }
        Ok((key, value))
    }
}

/// Parses a stand-alone decl body `{ ... }` (entityDef and similar decls).
pub fn parse_decl(text: &str) -> Result<Node, Error> {
    let t = tokenize(text)?;
    let mut p = P { t: &t, i: 0 };
    p.expect(&Tok::Open, "'{'")?;
    let body = p.body()?;
    if p.i != t.len() {
        return Err(invalid(p.line(), "content after decl".into()));
    }
    Ok(Node::Block(body))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// `entity` or `component`.
    pub section: String,
    /// Decl type, e.g. `entityDef` or `cpntRigidBody`.
    pub decl_type: String,
    pub name: String,
    pub body: Node,
    /// 0-based position in the file (stable within one file version).
    pub ordinal: usize,
}

pub struct EntitiesFile {
    pub version: String,
    pub items: Vec<Item>,
}

impl EntitiesFile {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let t = tokenize(text)?;
        let mut p = P { t: &t, i: 0 };
        let tag = p.word("'Version'")?;
        if !tag.eq_ignore_ascii_case("version") {
            return Err(invalid(1, format!("expected Version, found {tag}")));
        }
        let version = p.word("version number")?;
        let mut items = Vec::new();
        while p.peek().is_some() {
            let section = p.word("section")?;
            p.expect(&Tok::Open, "'{'")?;
            let decl_type = p.word("decl type")?;
            let name = p.word("item name")?;
            p.expect(&Tok::Open, "'{'")?;
            let body = Node::Block(p.body()?);
            p.expect(&Tok::Close, "'}' closing section")?;
            items.push(Item {
                section,
                decl_type,
                name,
                body,
                ordinal: items.len(),
            });
        }
        Ok(Self { version, items })
    }
}

/// Deep merge: blocks merge key by key (child wins), anything else is replaced.
pub fn merge(parent: &Node, child: &Node) -> Node {
    match (parent, child) {
        (Node::Block(pb), Node::Block(cb)) => {
            let mut out = pb.clone();
            for (k, v) in cb {
                match out.iter_mut().find(|(pk, _)| pk.eq_ignore_ascii_case(k)) {
                    Some((_, existing)) => *existing = merge(existing, v),
                    None => out.push((k.clone(), v.clone())),
                }
            }
            Node::Block(out)
        }
        _ => child.clone(),
    }
}

/// Resolves `inherit` chains. `load` returns the decl body for an inherit name.
/// Returns the merged body and the chain (nearest first), or the unresolved names.
pub fn resolve_inheritance(
    body: &Node,
    mut load: impl FnMut(&str) -> Option<Node>,
) -> (Node, Vec<String>, Vec<String>) {
    let mut chain = Vec::new();
    let mut missing = Vec::new();
    let mut stack = vec![body.clone()];
    // An empty `inherit = "";` (observed on 50 tutorial entities) means no parent.
    let parent_of = |n: &Node| {
        n.get("inherit")
            .and_then(Node::scalar)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let mut next = parent_of(body);
    while let Some(name) = next.take() {
        if chain.contains(&name) || chain.len() > 32 {
            missing.push(format!("{name} (cycle or depth limit)"));
            break;
        }
        chain.push(name.clone());
        match load(&name) {
            Some(parent) => {
                next = parent_of(&parent);
                stack.push(parent);
            }
            None => missing.push(name),
        }
    }
    let merged = stack
        .iter()
        .rev()
        .fold(Node::Block(Vec::new()), |acc, n| merge(&acc, n));
    (merged, chain, missing)
}

/// Archive name of an `entityDef` decl for an `inherit` value.
pub fn entity_def_resource_name(inherit: &str) -> String {
    format!(
        "generated/decls/entitydef/{}.decl",
        inherit.to_ascii_lowercase()
    )
}

/// Spawn transform from a merged `edit` block, in file space.
/// Orientation rows default to identity per element; position to zero;
/// scale components to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnTransform {
    pub position: [f32; 3],
    /// `spawnOrientation.mat[row]` as written (row-major).
    pub rows: [[f32; 3]; 3],
    pub scale: [f32; 3],
}

impl SpawnTransform {
    pub fn from_edit(edit: &Node) -> Self {
        let axis = ["x", "y", "z"];
        let position = axis.map(|a| edit.f32_at(&["spawnPosition", a]).unwrap_or(0.0));
        let mut rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (r, row) in rows.iter_mut().enumerate() {
            let key = format!("mat[{r}]");
            for (c, v) in row.iter_mut().enumerate() {
                if let Some(x) = edit.f32_at(&["spawnOrientation", "mat", &key, axis[c]]) {
                    *v = x;
                }
            }
        }
        let scale = axis.map(|a| edit.f32_at(&["renderModelInfo", "scale", a]).unwrap_or(1.0));
        Self {
            position,
            rows,
            scale,
        }
    }
}

/// Component references in a merged `m_componentDecls` block: (id, op, target).
pub fn component_refs(edit: &Node) -> Vec<(String, String, String)> {
    let Some(list) = edit.get("m_componentDecls") else {
        return Vec::new();
    };
    let mut out: Vec<(String, String, String)> = Vec::new();
    for (k, v) in list.entries() {
        let Some((op, rest)) = k.split_once('[') else {
            continue;
        };
        let id = rest.trim_end_matches(']').trim_matches('"').to_owned();
        let target = ["mapComponentDecl", "inheritedDecl"]
            .iter()
            .find_map(|f| v.get(f).and_then(Node::scalar).map(|s| format!("{f}:{s}")))
            .unwrap_or_else(|| {
                if v.get("inlineDecl").is_some() {
                    "inlineDecl".into()
                } else {
                    "?".into()
                }
            });
        match op {
            "item_remove" => out.retain(|(i, _, _)| *i != id),
            _ => match out.iter_mut().find(|(i, _, _)| *i == id) {
                Some(e) if target != "?" && target != "inlineDecl" => e.2 = target,
                Some(_) => {}
                None => out.push((id, op.to_owned(), target)),
            },
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_resources_by_source_kind() {
        assert_eq!(
            model_resource("models/Props/Barrel_01.lwo"),
            ModelResource::Static("generated/model/models/props/barrel_01.bmodel".into())
        );
        assert_eq!(
            model_resource("models/a.b/door_latch.md6"),
            ModelResource::Skinned("generated/basemodel/models/a.b/door_latch.bmd6model".into())
        );
        assert_eq!(
            model_resource("maps/x/x_p/audio_area_volume_combat"),
            ModelResource::Static("maps/x/x_p/audio_area_volume_combat.bmodel".into())
        );
    }

    const FILE: &str = r#"Version 6
component {
	cpntAudio map_cpntaudio_decl_1 {
		edit = {
			m_enablePropagation = false;
		}
	}
}
entity {
	entityDef rail_23 {
		inherit = "models/x/rail.def";
		class = "idEntity";
		edit = {
			renderModelInfo = {
				scale = {
					z = -1;
				}
			}
			m_componentDecls = {
				item_begin = true;
				item_add["3H5CN2"] = {
					mapComponentDecl = "map_cpntrigidbody_decl_12";
				}
			}
			spawnPosition = {
				x = 24.5;
				z = 3.75;
			}
			spawnOrientation = {
				mat = {
					mat[0] = {
						x = -0.5;
						y = -0.75;
					}
				}
			}
		}
	}
}
"#;

    #[test]
    fn parses_entities_file() {
        let f = EntitiesFile::parse(FILE).unwrap();
        assert_eq!(f.version, "6");
        assert_eq!(f.items.len(), 2);
        let e = &f.items[1];
        assert_eq!(
            (e.section.as_str(), e.decl_type.as_str(), e.name.as_str()),
            ("entity", "entityDef", "rail_23")
        );
        let edit = e.body.get("edit").unwrap();
        let t = SpawnTransform::from_edit(edit);
        assert_eq!(t.position, [24.5, 0.0, 3.75]);
        assert_eq!(t.rows[0], [-0.5, -0.75, 0.0]);
        assert_eq!(t.rows[2], [0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, -1.0]);
        let c = component_refs(edit);
        assert_eq!(c[0].0, "3H5CN2");
        assert_eq!(c[0].2, "mapComponentDecl:map_cpntrigidbody_decl_12");
    }

    #[test]
    fn parses_entity_def_with_bare_blocks_and_tuples() {
        let d = parse_decl(
            r#"{
	inherit = "entitydef_ref/static_parent.def";
	editorVars {
		color = ( 0, 0.5, 1, 1 );
	}
	edit = {
		renderModelInfo = {
			model = "models/x/rail.lwo";
		}
	}
}"#,
        )
        .unwrap();
        assert_eq!(
            d.path(&["editorVars", "color"]),
            Some(&Node::Tuple(vec![
                "0".into(),
                "0.5".into(),
                "1".into(),
                "1".into()
            ]))
        );
        assert_eq!(
            d.path(&["edit", "renderModelInfo", "model"])
                .and_then(Node::scalar),
            Some("models/x/rail.lwo")
        );
    }

    #[test]
    fn inheritance_merges_child_over_parent() {
        let parent = parse_decl("{ edit = { a = 1; b = { x = 1; y = 2; } } }").unwrap();
        let child =
            parse_decl(r#"{ inherit = "p.def"; edit = { b = { y = 5; } c = 3; } }"#).unwrap();
        let (m, chain, missing) =
            resolve_inheritance(&child, |n| (n == "p.def").then(|| parent.clone()));
        assert_eq!(chain, ["p.def"]);
        assert!(missing.is_empty());
        let edit = m.get("edit").unwrap();
        assert_eq!(edit.f32_at(&["a"]), Some(1.0));
        assert_eq!(edit.f32_at(&["b", "x"]), Some(1.0));
        assert_eq!(edit.f32_at(&["b", "y"]), Some(5.0));
        assert_eq!(edit.f32_at(&["c"]), Some(3.0));
        // Missing parents are reported, not invented.
        let (_, _, missing) = resolve_inheritance(&child, |_| None);
        assert_eq!(missing, ["p.def"]);
    }

    #[test]
    fn typed_keys_and_unit_values() {
        let d = parse_decl(
            r#"{ edit = { "idLight::lightType_t" m_Type = "LIGHT_SPOT"; m_DistanceCutoff = 15m; } }"#,
        )
        .unwrap();
        let edit = d.get("edit").unwrap();
        assert_eq!(
            edit.get("m_Type").and_then(Node::scalar),
            Some("LIGHT_SPOT")
        );
        assert_eq!(
            edit.get("m_DistanceCutoff").and_then(Node::scalar),
            Some("15m")
        );
        // Fully annotated form used by light/spot.def.
        let d = parse_decl(
            r#"{ "idLight" "" edit = { "idLight::lightType_t" "" "m_Type" = "LIGHT_POINT"; } }"#,
        )
        .unwrap();
        assert_eq!(
            d.path(&["edit", "m_Type"]).and_then(Node::scalar),
            Some("LIGHT_POINT")
        );
        // Bracketed keys keep their quoted ids.
        let d = parse_decl(r#"{ l = { item_add["ab"] = { x = 1; } } }"#).unwrap();
        assert!(d.path(&["l", "item_add[\"ab\"]"]).is_some());
    }

    #[test]
    fn rejects_malformed() {
        assert!(EntitiesFile::parse("Version 6 entity { entityDef x { a = 1; }").is_err());
        assert!(parse_decl("{ a = ( 1, 2 ").is_err());
        assert!(parse_decl("{ = 1; }").is_err());
    }
}

/// How `spawnOrientation.mat` rows map to the local-to-world rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisConvention {
    /// Row i is local axis i expressed in world space (idTech `idMat3 axis`):
    /// world = position + sum_i local_i * scale_i * row_i.
    RowsAreAxes,
    /// Rows are matrix rows: world = position + mat * (scale * local).
    RowsAreRows,
}

impl SpawnTransform {
    /// Row-major 3x4 local-to-world matrix in file space.
    pub fn matrix(&self, convention: AxisConvention) -> [f32; 12] {
        let mut m = [0f32; 12];
        for r in 0..3 {
            for c in 0..3 {
                let v = match convention {
                    AxisConvention::RowsAreAxes => self.rows[c][r],
                    AxisConvention::RowsAreRows => self.rows[r][c],
                };
                m[r * 4 + c] = v * self.scale[c];
            }
            m[r * 4 + 3] = self.position[r];
        }
        m
    }
}

/// An entity whose merged definition references a model.
#[derive(Clone, Debug)]
pub struct PlacedModel {
    pub entity: String,
    pub ordinal: usize,
    pub class: String,
    /// `renderModelInfo.model` as written (source path, e.g. `.lwo`).
    pub model_source: String,
    pub transform: SpawnTransform,
    /// `renderModelInfo.m_customMaterials` items in index order.
    pub custom_materials: Vec<String>,
}

/// Archive resource holding the built model for a `renderModelInfo.model`
/// source path (O: tutorial_p index entries, E021).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelResource {
    /// `model` kind, `1LMB` static model (decoded by `bmodel`).
    Static(String),
    /// `baseModel` kind, skinned `.md6mesh` build (`.bmd6model`; not decoded).
    Skinned(String),
}

impl ModelResource {
    pub fn name(&self) -> &str {
        match self {
            Self::Static(n) | Self::Skinned(n) => n,
        }
    }
}

/// Maps a model source path to its built resource:
/// - `<stem>.md6` -> `generated/basemodel/<stem>.bmd6model` (skinned);
/// - no extension (map-local brush models such as area volumes) ->
///   `<source>.bmodel` stored beside the map;
/// - otherwise (`.lwo`, ...) -> `generated/model/<stem>.bmodel`.
pub fn model_resource(source: &str) -> ModelResource {
    let file = source.rsplit('/').next().unwrap_or(source);
    match file.rsplit_once('.') {
        None => ModelResource::Static(format!("{source}.bmodel")),
        Some((_, ext)) => {
            let stem = &source[..source.len() - ext.len() - 1];
            if ext.eq_ignore_ascii_case("md6") {
                ModelResource::Skinned(format!(
                    "generated/basemodel/{}.bmd6model",
                    stem.to_ascii_lowercase()
                ))
            } else {
                ModelResource::Static(format!(
                    "generated/model/{}.bmodel",
                    stem.to_ascii_lowercase()
                ))
            }
        }
    }
}

/// Archive name of the built static model for a `.lwo`-style source path.
pub fn model_resource_name(source: &str) -> String {
    let stem = source.rsplit_once('.').map_or(source, |(s, _)| s);
    format!("generated/model/{}.bmodel", stem.to_ascii_lowercase())
}

/// Entities with models, after inheritance. `load_def` resolves `inherit` names.
pub fn placed_models(
    file: &EntitiesFile,
    mut load_def: impl FnMut(&str) -> Option<Node>,
) -> Vec<PlacedModel> {
    let mut out = Vec::new();
    for item in file.items.iter().filter(|i| i.section == "entity") {
        let (merged, _, _) = resolve_inheritance(&item.body, &mut load_def);
        let Some(edit) = merged.get("edit") else {
            continue;
        };
        let Some(model) = edit
            .path(&["renderModelInfo", "model"])
            .and_then(Node::scalar)
        else {
            continue;
        };
        if model.is_empty() {
            continue;
        }
        let custom = edit.path(&["renderModelInfo", "m_customMaterials"]);
        let num = custom.and_then(|c| c.f32_at(&["num"])).unwrap_or(0.0) as usize;
        let custom_materials = (0..num)
            .map(|i| {
                custom
                    .and_then(|c| c.get(&format!("item[{i}]")))
                    .and_then(Node::scalar)
                    .unwrap_or("")
                    .to_owned()
            })
            .collect();
        out.push(PlacedModel {
            entity: item.name.clone(),
            ordinal: item.ordinal,
            class: merged
                .get("class")
                .and_then(Node::scalar)
                .unwrap_or("")
                .to_owned(),
            model_source: model.to_owned(),
            transform: SpawnTransform::from_edit(edit),
            custom_materials,
        });
    }
    out
}

/// Light parameters read from a merged `idLight` edit block. Units follow the
/// decl (`15m` -> 15.0 metres); fields absent from the block are `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct LightInfo {
    pub entity: String,
    pub ordinal: usize,
    /// `m_Type`, e.g. `LIGHT_SPOT`, `LIGHT_POINT`.
    pub kind: Option<String>,
    pub color: Option<[f32; 3]>,
    pub distance_cutoff: Option<f32>,
    pub distance_falloff: Option<f32>,
    /// Degrees.
    pub angular_cutoff: Option<f32>,
    pub angular_falloff: Option<f32>,
    pub preset: Option<String>,
    pub scatter_only: bool,
    pub transform: SpawnTransform,
}

fn metres(v: &str) -> Option<f32> {
    v.trim().trim_end_matches('m').trim().parse().ok()
}

pub fn lights(
    file: &EntitiesFile,
    mut load_def: impl FnMut(&str) -> Option<Node>,
) -> Vec<LightInfo> {
    let mut out = Vec::new();
    for item in file.items.iter().filter(|i| i.section == "entity") {
        let (merged, _, _) = resolve_inheritance(&item.body, &mut load_def);
        if !merged
            .get("class")
            .and_then(Node::scalar)
            .is_some_and(|c| c == "idLight")
        {
            continue;
        }
        let empty = Node::Block(Vec::new());
        let edit = merged.get("edit").unwrap_or(&empty);
        let s = |k: &str| edit.get(k).and_then(Node::scalar);
        let color = ["r", "g", "b"].map(|c| edit.f32_at(&["m_Color", c]));
        out.push(LightInfo {
            entity: item.name.clone(),
            ordinal: item.ordinal,
            kind: s("m_Type").map(str::to_owned),
            color: (color.iter().all(Option::is_some)).then(|| color.map(|c| c.unwrap_or(0.0))),
            distance_cutoff: s("m_DistanceCutoff").and_then(metres),
            distance_falloff: s("m_DistanceFalloff").and_then(metres),
            angular_cutoff: s("m_AngularCutoff").and_then(|v| v.parse().ok()),
            angular_falloff: s("m_AngularFalloff").and_then(|v| v.parse().ok()),
            preset: s("m_preset").map(str::to_owned),
            scatter_only: s("m_scatterOnly").is_some_and(|v| v == "true"),
            transform: SpawnTransform::from_edit(edit),
        });
    }
    out
}
