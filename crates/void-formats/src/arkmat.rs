//! `arkDefault` material model, transcribed from the game's shipped shader
//! source (`generated/decls/renderprog/arkdefaulthelpers.decl`, `pixel.decl`,
//! `global.decl`, `helper.decl`; see docs/formats/arkdefault.md).
//!
//! Only the HQ path's material evaluation is modelled: layer UVs, masks,
//! diffuse/gloss/metallic/normal blending, and the gloss/metallic remaps.
//! Lighting (Ward BRDF, IBL, SH) is out of scope. Texture sampling is left to
//! the caller so this module stays independent of any renderer.

use crate::Error;
use crate::decl::{Tok, tokenize};
use crate::material::{Material, Value};

/// Face culling from the material's `state { cull none | front | back }` (default back).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cull {
    #[default]
    Back,
    Front,
    None,
}

impl Cull {
    /// The material's own mode, or back culling when its state does not set one.
    pub fn of(material: &Material) -> Self {
        Self::explicit(material).unwrap_or_default()
    }

    /// `cull` from the material's `state` block, if present (the program's own state
    /// block supplies the default otherwise, e.g. arkVegetation `cull none`).
    pub fn explicit(material: &Material) -> Option<Self> {
        let state = material.entries.iter().find_map(|(k, v)| match v {
            Value::Block(b) if k.eq_ignore_ascii_case("state") => Some(b),
            _ => None,
        });
        let mode = state.and_then(|b| {
            b.iter().find_map(|(k, v)| match v {
                Value::Text(t) if k.eq_ignore_ascii_case("cull") => {
                    Some(t.trim().to_ascii_lowercase())
                }
                _ => None,
            })
        });
        match mode.as_deref()? {
            "none" | "twosided" => Some(Cull::None),
            "front" => Some(Cull::Front),
            _ => Some(Cull::Back),
        }
    }
}

/// A shipped `renderParm` default: `{ option N range [..] }` or
/// `{ Uniform type { v, .. } range .. }` / `{ Uniform float v range .. }`.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderParm {
    Option(i64),
    Uniform(Vec<f32>),
}

impl RenderParm {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let bad = |d: &str| Error::Invalid {
            field: "renderParm decl",
            detail: d.to_owned(),
        };
        let t = tokenize(text)?;
        let words: Vec<&Tok> = t.iter().map(|t| &t.tok).collect();
        match words.as_slice() {
            [Tok::Open, Tok::Word(kind), rest @ ..] if kind.eq_ignore_ascii_case("option") => {
                match rest.first() {
                    Some(Tok::Word(v)) => v
                        .parse()
                        .map(RenderParm::Option)
                        .map_err(|_| bad("option value")),
                    _ => Err(bad("missing option value")),
                }
            }
            [Tok::Open, Tok::Word(kind), Tok::Word(_ty), rest @ ..]
                if kind.eq_ignore_ascii_case("uniform") =>
            {
                let mut vals = Vec::new();
                match rest.first() {
                    Some(Tok::Open) => {
                        for tok in &rest[1..] {
                            match tok {
                                Tok::Word(w) => {
                                    vals.push(w.parse().map_err(|_| bad("uniform component"))?)
                                }
                                Tok::Comma => {}
                                Tok::Close => break,
                                _ => return Err(bad("uniform vector")),
                            }
                        }
                    }
                    Some(Tok::Word(w)) => vals.push(w.parse().map_err(|_| bad("uniform scalar"))?),
                    _ => return Err(bad("missing uniform value")),
                }
                Ok(RenderParm::Uniform(vals))
            }
            _ => Err(bad("unrecognized renderParm shape")),
        }
    }
}

/// Archive name of the default for parameter `name`.
pub fn render_parm_resource_name(name: &str) -> String {
    format!(
        "generated/decls/renderparm/{}.decl",
        name.to_ascii_lowercase()
    )
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layer {
    pub uv_set: usize,
    pub scale_bias: [f32; 4],
    pub mask_mode: i64,
    pub vtx_channel: usize,
    pub invert_mask: bool,
    pub rescale: [f32; 2],
    pub mask_uv_set: usize,
    pub mask_scale_bias: [f32; 4],
    pub use_color_constant: bool,
    pub color_constant: [f32; 4],
    /// Reuse options for this layer (0 = own map; 1 = layer 0; 2 = layer 1).
    pub diffuse_reuse: i64,
    pub bump_reuse: i64,
    pub gloss_reuse: i64,
    pub metallic_reuse: i64,
    pub mask_reuse: i64,
    pub diffuse_map: Option<String>,
    pub bump_map: Option<String>,
    pub gloss_map: Option<String>,
    pub metallic_map: Option<String>,
    pub mask_map: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArkDefault {
    pub extra_layer: usize,
    pub layers: [Layer; 3],
    pub ward_roughness: [f32; 4],
    pub metallic_min_max: [f32; 2],
    pub has_gloss_map: bool,
    pub has_metallic_map: bool,
    pub has_bump_map: bool,
    /// `$hasOcclusionMap` / `$occlusionMap` (`HQ_FetchAO`: `.r` at layer-0 UVs).
    pub has_occlusion_map: bool,
    pub occlusion_map: Option<String>,
    /// `$zbias` (arkDecal: subtracted from clip-space z; renderParm default 0).
    pub zbias: f32,
    /// `$decalType` (arkDecal: 0 none = mesh decal, 1 clip model, 2 cull model, 3 depth box).
    pub decal_type: i64,
    /// `state { cull .. }`.
    pub cull: Cull,
    pub alpha_test: bool,
    pub vtx_color_multiply: bool,
    pub two_sided_lighting: bool,
    /// Lighting inputs read by `HQ_ComputeLightingParams` / `HQ_FetchMetallicSpecularFresnelAndIOR`
    /// (`$specularTintDielectric`, `$diffuseAlbedoMetallic`, `$sheen`, `$sheenImportance`).
    pub specular_tint_dielectric: [f32; 3],
    pub diffuse_albedo_metallic: [f32; 3],
    pub sheen: f32,
    pub sheen_importance: f32,
    /// Parameters resolved from neither the material nor a shipped default.
    pub missing: Vec<String>,
}

/// Resolves parameters: the material's own value (top level or `options`,
/// case-insensitive), then the shipped renderParm default via `default`.
pub struct Resolver<'a, F: Fn(&str) -> Option<RenderParm>> {
    pub material: &'a Material,
    pub default: F,
    pub missing: Vec<String>,
}

impl<F: Fn(&str) -> Option<RenderParm>> Resolver<'_, F> {
    fn own(&self, name: &str) -> Option<Value> {
        let find = |entries: &[(String, Value)]| {
            entries
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.clone())
        };
        find(&self.material.entries).or_else(|| {
            self.material.entries.iter().find_map(|(k, v)| match v {
                Value::Block(b) if k.eq_ignore_ascii_case("options") => find(b),
                _ => None,
            })
        })
    }

    pub fn vec(&mut self, name: &str, n: usize) -> Vec<f32> {
        let v = match self.own(name) {
            Some(Value::Vector(v)) => Some(v),
            Some(Value::Text(t)) => t.trim().parse::<f32>().ok().map(|x| vec![x]),
            _ => match (self.default)(name) {
                Some(RenderParm::Uniform(v)) => Some(v),
                Some(RenderParm::Option(o)) => Some(vec![o as f32]),
                None => None,
            },
        };
        let mut v = v.unwrap_or_else(|| {
            self.missing.push(name.to_owned());
            Vec::new()
        });
        v.resize(n, 0.0);
        v
    }

    pub fn int(&mut self, name: &str) -> i64 {
        match self.own(name) {
            Some(Value::Text(t)) => t.trim().parse().unwrap_or_else(|_| {
                self.missing.push(format!("{name} (unparsed '{t}')"));
                0
            }),
            _ => match (self.default)(name) {
                Some(RenderParm::Option(o)) => o,
                Some(RenderParm::Uniform(v)) => v.first().copied().unwrap_or(0.0) as i64,
                None => {
                    self.missing.push(name.to_owned());
                    0
                }
            },
        }
    }

    pub fn map(&self, name: &str) -> Option<String> {
        match self.own(name) {
            Some(Value::Text(t)) if !t.trim().is_empty() => Some(t),
            _ => None,
        }
    }
}

impl ArkDefault {
    /// Resolves an arkDefault material. Parameter names follow the shader's
    /// `$name` spelling; the decl matches them case-insensitively.
    pub fn resolve(material: &Material, default: impl Fn(&str) -> Option<RenderParm>) -> Self {
        let mut r = Resolver {
            material,
            default,
            missing: Vec::new(),
        };
        let mut layers: [Layer; 3] = Default::default();
        for (i, l) in layers.iter_mut().enumerate() {
            let p = |s: &str| format!("Layer{i}_{s}");
            l.uv_set = r.int(&p("UVset")) as usize;
            l.scale_bias = r.vec(&p("ScaleBias"), 4).try_into().expect("4");
            l.mask_mode = r.int(&p("MaskMode"));
            l.vtx_channel = r.int(&p("VtxColorMaskChannel")).clamp(0, 3) as usize;
            l.invert_mask = r.int(&p("InvertMask")) != 0;
            l.rescale = r.vec(&p("RescaleValues"), 2).try_into().expect("2");
            l.mask_uv_set = r.int(&p("Mask_UVset")) as usize;
            l.mask_scale_bias = r.vec(&p("MaskScaleBias"), 4).try_into().expect("4");
            l.color_constant = r.vec(&p("ColorConstant"), 4).try_into().expect("4");
            // `use_LayerN_ColorConstant` has no shipped default decl; absent means off.
            l.use_color_constant = r
                .own(&format!("use_Layer{i}_ColorConstant"))
                .is_some_and(|v| matches!(v, Value::Text(t) if t.trim() != "0"));
            if i > 0 {
                l.diffuse_reuse = r.int(&p("diffuseReuseLayer"));
                l.bump_reuse = r.int(&p("bumpReuseLayer"));
                l.gloss_reuse = r.int(&p("glossReuseLayer"));
                l.metallic_reuse = r.int(&p("metallicReuseLayer"));
                l.mask_reuse = r.int(&p("maskReuseLayer"));
                l.diffuse_map = r.map(&p("diffuseMap"));
                l.bump_map = r.map(&p("bumpMap"));
                l.gloss_map = r.map(&p("glossMap"));
                l.metallic_map = r.map(&p("metallicMap"));
            } else {
                l.diffuse_map = r.map("diffusemap");
                l.bump_map = r.map("bumpmap");
                l.gloss_map = r.map("glossMap");
                l.metallic_map = r.map("metallicMap");
            }
            l.mask_map = r.map(&p("Maskmap"));
        }
        let extra_layer = r.int("extraLayer").clamp(0, 2) as usize;
        let ward_roughness = r.vec("wardRoughness", 4).try_into().expect("4");
        let metallic_min_max = r.vec("metallicMinMax", 2).try_into().expect("2");
        let flag = |r: &mut Resolver<_>, n: &str| r.int(n) != 0;
        let has_gloss_map = flag(&mut r, "hasGlossMap");
        let has_metallic_map = flag(&mut r, "hasMetallicMap");
        let has_bump_map = flag(&mut r, "hasBumpMap");
        let has_occlusion_map = flag(&mut r, "hasOcclusionMap");
        let occlusion_map = r.map("occlusionMap");
        let zbias = r.vec("zbias", 1)[0];
        let decal_type = r.int("decalType");
        let alpha_test = flag(&mut r, "alphaTest");
        let vtx_color_multiply = flag(&mut r, "VtxColorMultiply");
        let two_sided_lighting = flag(&mut r, "twoSidedLighting");
        let specular_tint_dielectric = r.vec("specularTintDielectric", 3).try_into().expect("3");
        let diffuse_albedo_metallic = r.vec("diffuseAlbedoMetallic", 3).try_into().expect("3");
        let sheen = r.vec("sheen", 1)[0];
        let sheen_importance = r.vec("sheenImportance", 1)[0];
        Self {
            specular_tint_dielectric,
            diffuse_albedo_metallic,
            sheen,
            sheen_importance,
            extra_layer,
            layers,
            ward_roughness,
            metallic_min_max,
            has_gloss_map,
            has_metallic_map,
            has_bump_map,
            has_occlusion_map,
            occlusion_map,
            zbias,
            decal_type,
            cull: Cull::of(material),
            alpha_test,
            vtx_color_multiply,
            two_sided_lighting,
            missing: r.missing,
        }
    }

    /// `m_layerTexCoords[i]` (pixel.decl `initLayeredTexcoord`). Layers beyond
    /// `extra_layer` reuse layer 0's coordinates, as the shader does.
    pub fn layer_uv(&self, i: usize, uv: [[f32; 2]; 2]) -> [f32; 2] {
        let i = if i <= self.extra_layer { i } else { 0 };
        let l = &self.layers[i];
        let s = uv[l.uv_set.min(1)];
        [
            s[0] * l.scale_bias[0] + l.scale_bias[2],
            s[1] * l.scale_bias[1] + l.scale_bias[3],
        ]
    }

    pub fn mask_uv(&self, i: usize, uv: [[f32; 2]; 2]) -> [f32; 2] {
        let l = &self.layers[i];
        let s = uv[l.mask_uv_set.min(1)];
        [
            s[0] * l.mask_scale_bias[0] + l.mask_scale_bias[2],
            s[1] * l.mask_scale_bias[1] + l.mask_scale_bias[3],
        ]
    }

    /// Which layer's own mask texture layer `i` samples, honouring reuse.
    pub fn mask_source(&self, i: usize) -> usize {
        match (i, self.layers[i].mask_reuse) {
            (0, _) | (_, 0) => i,
            (_, 1) => 0,
            _ => 1,
        }
    }

    /// `HQ_FetchMasks`: `mask_tex[i]` is the `.r` of layer i's mask sample
    /// (`None` when the mode does not sample), `vtx` the vertex colour.
    pub fn masks(&self, vtx: [f32; 4], mask_tex: [Option<f32>; 3]) -> [f32; 3] {
        let mut m = [0.0f32; 3];
        for (i, out) in m.iter_mut().enumerate() {
            if i > self.extra_layer {
                break;
            }
            let l = &self.layers[i];
            if i == 0 && !l.use_color_constant {
                *out = 0.0;
            } else {
                let tex = mask_tex[i].unwrap_or(0.0);
                *out = match l.mask_mode {
                    0 => vtx[l.vtx_channel],
                    1 => tex,
                    _ => linear_step(l.rescale[0], l.rescale[1], tex * vtx[l.vtx_channel]),
                };
            }
            if l.invert_mask {
                *out = 1.0 - *out;
            }
        }
        m
    }

    /// Blend of a scalar channel (gloss or metallic) across layers. `own[i]`
    /// is layer i's own sample (`None` when that layer has no map), `reuse`
    /// the per-layer reuse option. Mirrors `HQ_FetchGloss`/`HQ_FetchMetallic`.
    fn blend_scalar(
        &self,
        masks: [f32; 3],
        own: [Option<f32>; 3],
        reuse: [i64; 3],
        gloss_style: bool,
    ) -> f32 {
        let l0 = own[0].unwrap_or(0.0);
        let mut v = l0;
        if self.extra_layer > 0 {
            // Gloss with reuse=1 keeps layer 0 unchanged (no lerp); metallic skips the lerp the same way.
            let l1 = if reuse[1] != 0 {
                l0
            } else {
                own[1].unwrap_or(0.0)
            };
            if reuse[1] == 0 {
                v = lerp(v, l1, masks[1]);
            }
            if self.extra_layer > 1 {
                let l2 = match reuse[2] {
                    2 => l1,
                    1 => l0,
                    _ => own[2].unwrap_or(0.0),
                };
                v = lerp(v, l2, masks[2]);
            }
        }
        let _ = gloss_style;
        v
    }

    /// Final gloss (`HQ_FetchGloss`, no decals/emissive); roughness = 1 - gloss.
    pub fn gloss(&self, masks: [f32; 3], own: [Option<f32>; 3]) -> f32 {
        if !self.has_gloss_map {
            return self.ward_roughness[1];
        }
        let reuse = [0, self.layers[1].gloss_reuse, self.layers[2].gloss_reuse];
        let g = self.blend_scalar(masks, own, reuse, true);
        lerp(self.ward_roughness[0], self.ward_roughness[1], g)
    }

    /// Final metallic (`HQ_FetchMetallic`).
    pub fn metallic(&self, masks: [f32; 3], own: [Option<f32>; 3]) -> f32 {
        if !self.has_metallic_map {
            return self.metallic_min_max[1];
        }
        let reuse = [
            0,
            self.layers[1].metallic_reuse,
            self.layers[2].metallic_reuse,
        ];
        let m = self.blend_scalar(masks, own, reuse, false);
        lerp(self.metallic_min_max[0], self.metallic_min_max[1], m)
    }

    /// Diffuse albedo (`HQ_FetchDiffuseAlbedo`, no decals/debug). `own[i]` is
    /// layer i's own linear RGBA sample.
    pub fn diffuse(&self, masks: [f32; 3], own: [[f32; 4]; 3], vtx: [f32; 4]) -> [f32; 4] {
        let mul = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|k| a[k] * b[k]);
        let mix = |a: [f32; 4], b: [f32; 4], t: f32| std::array::from_fn(|k| lerp(a[k], b[k], t));
        let mut d = own[0];
        if self.layers[0].use_color_constant {
            d = mul(d, mix([1.0; 4], self.layers[0].color_constant, masks[0]));
        }
        if self.extra_layer > 0 {
            let t1 = if self.layers[1].diffuse_reuse != 0 {
                own[0]
            } else {
                own[1]
            };
            let mut l1 = t1;
            if self.layers[1].use_color_constant {
                l1 = mul(l1, self.layers[1].color_constant);
            }
            d = mix(d, l1, masks[1]);
            if self.extra_layer > 1 {
                let mut l2 = match self.layers[2].diffuse_reuse {
                    2 => t1,
                    1 => own[0],
                    _ => own[2],
                };
                if self.layers[2].use_color_constant {
                    l2 = mul(l2, self.layers[2].color_constant);
                }
                d = mix(d, l2, masks[2]);
            }
        }
        if self.vtx_color_multiply {
            d = mul(d, vtx);
        }
        d.map(|x| x.max(0.00001))
    }

    /// Blended raw BC5 xy (`HQ_FetchAndComputeNormalAndTBN`, before unpack).
    pub fn raw_normal(&self, masks: [f32; 3], own: [[f32; 2]; 3]) -> [f32; 2] {
        let mix = |a: [f32; 2], b: [f32; 2], t: f32| [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
        let mut n = own[0];
        if self.extra_layer > 0 {
            let l1 = own[1];
            if self.layers[1].bump_reuse == 0 {
                n = mix(n, l1, masks[1]);
            }
            if self.extra_layer > 1 {
                let l2 = match self.layers[2].bump_reuse {
                    2 => l1,
                    1 => own[0],
                    _ => own[2],
                };
                n = mix(n, l2, masks[2]);
            }
        }
        n
    }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// `helper.decl` LinearStep.
pub fn linear_step(min: f32, max: f32, t: f32) -> f32 {
    ((t - min) / (max - min)).clamp(0.0, 1.0)
}

/// `global.decl` UnpackBC5NormalMap: z = 1.001 - saturate(dot(xy, xy)), normalized.
pub fn unpack_bc5_normal(xy: [f32; 2]) -> [f32; 3] {
    let z = 1.001 - (xy[0] * xy[0] + xy[1] * xy[1]).clamp(0.0, 1.0);
    let l = (xy[0] * xy[0] + xy[1] * xy[1] + z * z).sqrt();
    [xy[0] / l, xy[1] / l, z / l]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults(name: &str) -> Option<RenderParm> {
        let l = name.to_ascii_lowercase();
        Some(match l.as_str() {
            n if n.ends_with("rescalevalues") => RenderParm::Uniform(vec![0.4, 0.6]),
            n if n.ends_with("scalebias") => RenderParm::Uniform(vec![1.0, 1.0, 0.0, 0.0]),
            n if n.ends_with("colorconstant") => RenderParm::Uniform(vec![1.0; 4]),
            "wardroughness" => RenderParm::Uniform(vec![0.0, 0.5, 0.0, 0.0]),
            "metallicminmax" => RenderParm::Uniform(vec![0.0, 0.5]),
            n if n.ends_with("vtxcolormaskchannel") => RenderParm::Option(3),
            _ => RenderParm::Option(0),
        })
    }

    #[test]
    fn cull_mode_from_state_block() {
        let m = |t: &str| Material::parse(t).unwrap();
        assert_eq!(Cull::explicit(&m("{ state { } diffusemap x }")), None);
        assert_eq!(Cull::of(&m("{ state { } diffusemap x }")), Cull::Back);
        assert_eq!(
            Cull::explicit(&m("{ state {
 blend SRC_ALPHA ONE_MINUS_SRC_ALPHA
 cull none
 } }")),
            Some(Cull::None)
        );
        assert_eq!(
            Cull::of(&m("{ state {
 cull front
 } }")),
            Cull::Front
        );
    }

    #[test]
    fn parses_render_parms() {
        assert_eq!(
            RenderParm::parse("{ option 3 range [0, 3] }").unwrap(),
            RenderParm::Option(3)
        );
        assert_eq!(
            RenderParm::parse("{ Uniform float2 { 0.4, 0.6 } range [{0,0}, {1,1}] }").unwrap(),
            RenderParm::Uniform(vec![0.4, 0.6])
        );
        assert_eq!(
            RenderParm::parse("{ Uniform float { 0.5 } range[ 0.0, 1.0 ] }").unwrap(),
            RenderParm::Uniform(vec![0.5])
        );
        assert_eq!(
            RenderParm::parse("{ Uniform float 0.5 range[ 0.0, 1.0 ] }").unwrap(),
            RenderParm::Uniform(vec![0.5])
        );
        assert!(RenderParm::parse("{ texture foo }").is_err());
    }

    #[test]
    fn resolves_layered_ground_like_material() {
        let m = Material::parse(
            r#"{
            options {
                hasGlossMap 1
                extralayer 2
                layer1_maskmode 2
                layer1_vtxcolormaskchannel 0
                layer2_maskmode 2
                layer2_vtxcolormaskchannel 1
            }
            diffusemap a_d
            glossMap a_g
            layer1_glossmap b_g
            layer2_glossmap c_g
            layer1_maskmap m1
            layer2_maskmap m2
            wardRoughness { 0.3, 1, 0, 0 }
            layer1_rescalevalues { 0.188, 0.282 }
            layer2_scalebias { 2.5, 2.5, 0, 0 }
        }"#,
        )
        .unwrap();
        let a = ArkDefault::resolve(&m, defaults);
        assert!(a.missing.is_empty(), "{:?}", a.missing);
        assert_eq!(a.extra_layer, 2);
        assert_eq!(a.layers[1].rescale, [0.188, 0.282]);
        assert_eq!(a.layers[2].vtx_channel, 1);
        assert_eq!(a.layers[2].scale_bias, [2.5, 2.5, 0.0, 0.0]);
        assert_eq!(a.layer_uv(2, [[1.0, 2.0], [0.0, 0.0]]), [2.5, 5.0]);
        // Mode 2: LinearStep(rescale, mask * vertex channel).
        let masks = a.masks([1.0, 0.0, 0.0, 0.0], [None, Some(0.235), Some(1.0)]);
        assert!((masks[1] - 0.5).abs() < 1e-3);
        assert_eq!(masks[2], 0.0);
        let g = a.gloss(masks, [Some(0.0), Some(1.0), Some(0.5)]);
        assert!((g - lerp(0.3, 1.0, 0.5)).abs() < 1e-3);
    }

    #[test]
    fn defaults_without_maps() {
        let m =
            Material::parse("{ options { hasGlossMap 0 hasMetallicMap 0 } diffusemap x }").unwrap();
        let a = ArkDefault::resolve(&m, defaults);
        assert_eq!(a.gloss([0.0; 3], [None; 3]), 0.5);
        assert_eq!(a.metallic([0.0; 3], [None; 3]), 0.5);
    }

    #[test]
    fn bc5_unpack_matches_shader_formula() {
        let n = unpack_bc5_normal([0.0, 0.0]);
        assert!((n[2] - 1.0).abs() < 1e-6);
        let n = unpack_bc5_normal([0.6, 0.0]);
        // z = 1.001 - 0.36 = 0.641 before normalisation.
        let l = (0.36f32 + 0.641 * 0.641).sqrt();
        assert!((n[0] - 0.6 / l).abs() < 1e-6 && (n[2] - 0.641 / l).abs() < 1e-6);
    }
}
