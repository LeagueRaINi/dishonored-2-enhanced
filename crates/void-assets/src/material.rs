//! arkDefault material preparation (classification, resolved model, layer
//! textures) and the shader parameter packing shared by every renderer.
//! Semantics: docs/formats/arkdefault.md (validated against RX002, E020).

use std::cell::RefCell;
use std::collections::HashMap;

use void_formats::arkmat::{ArkDefault, RenderParm, render_parm_resource_name};
use void_formats::bim::Format;
use void_formats::material::{Material, material_resource_name};

use crate::Archive;
use crate::texture::{self, TextureData, Usage};

#[derive(Clone, Debug, PartialEq)]
pub enum Support {
    Basic,
    /// Rendered with the arkDefault model; the listed features are ignored.
    Partial(Vec<String>),
    Unsupported(String),
    /// Editor visualisation program (`editorDraw*`); never drawn.
    EditorOnly(String),
}

/// Per-layer textures (index = layer).
#[derive(Default)]
pub struct LayerTextures {
    pub diffuse: [Option<TextureData>; 3],
    pub normal: [Option<TextureData>; 3],
    pub gloss: [Option<TextureData>; 3],
    pub metallic: [Option<TextureData>; 3],
    pub mask: [Option<TextureData>; 3],
    /// `$occlusionMap` (only when `$hasOcclusionMap`).
    pub occlusion: Option<TextureData>,
}

pub struct Prepared {
    pub name: String,
    pub support: Support,
    pub model: Option<ArkDefault>,
    pub textures: LayerTextures,
    pub alpha_test: bool,
    pub two_sided: bool,
    /// `arkDecal` mesh decal: alpha-blended over opaque surfaces after them.
    pub decal: bool,
}

impl Prepared {
    pub fn drawable(&self) -> bool {
        matches!(self.support, Support::Basic | Support::Partial(_))
    }
}

fn ignored_features(m: &Material, a: &ArkDefault) -> Vec<String> {
    let mut out = Vec::new();
    for (slot, _) in m.image_slots() {
        let s = slot.to_ascii_lowercase();
        let modelled = matches!(
            s.as_str(),
            "diffusemap" | "bumpmap" | "glossmap" | "metallicmap" | "occlusionmap"
        ) || (s.starts_with("layer")
            && [
                "diffusemap",
                "bumpmap",
                "glossmap",
                "metallicmap",
                "maskmap",
            ]
            .iter()
            .any(|k| s.ends_with(k)));
        if !modelled {
            out.push(s);
        }
    }
    for key in [
        "needswind",
        "translucency/enable",
        "alpha/enablerefraction",
        "hasdetailbumpmap",
        "hasemissivemap",
    ] {
        if m.options()
            .any(|(k, v)| k.eq_ignore_ascii_case(key) && v.trim() != "0")
        {
            out.push(key.to_owned());
        }
    }
    if !a.missing.is_empty() {
        out.push(format!(
            "parameters without default: {}",
            a.missing.join(",")
        ));
    }
    out.sort();
    out.dedup();
    out
}

pub struct Preparer<'a> {
    pub archive: &'a Archive,
    pub max_size: u32,
    parms: RefCell<HashMap<String, Option<RenderParm>>>,
}

impl<'a> Preparer<'a> {
    pub fn new(archive: &'a Archive, max_size: u32) -> Self {
        Self {
            archive,
            max_size,
            parms: RefCell::new(HashMap::new()),
        }
    }

    fn default_parm(&self, name: &str) -> Option<RenderParm> {
        self.parms
            .borrow_mut()
            .entry(name.to_ascii_lowercase())
            .or_insert_with(|| {
                self.archive
                    .read_text("renderParm", &render_parm_resource_name(name))
                    .ok()
                    .and_then(|t| RenderParm::parse(&t).ok())
            })
            .clone()
    }

    pub fn prepare(&self, name: &str) -> Prepared {
        let mut p = Prepared {
            name: name.to_owned(),
            support: Support::Basic,
            model: None,
            textures: LayerTextures::default(),
            alpha_test: false,
            two_sided: false,
            decal: false,
        };
        let m = match self
            .archive
            .read_text("material", &material_resource_name(name))
            .and_then(|t| Material::parse(&t).map_err(|e| format!("decl parse: {e}")))
        {
            Ok(m) => m,
            Err(e) => {
                p.support = Support::Unsupported(format!("decl: {e}"));
                return p;
            }
        };
        // No `mainProgram`: treated as arkDefault (H: the engine's default program; e.g.
        // `models/templates/colors/*` flat-colour blockers).
        let no_program = m.program().is_none();
        let program = m.program().unwrap_or("arkDefault");
        if program.to_ascii_lowercase().starts_with("editor") {
            p.support = Support::EditorOnly(program.to_owned());
            return p;
        }
        let vegetation = program.eq_ignore_ascii_case("arkvegetation");
        let decal = program.eq_ignore_ascii_case("arkdecal");
        if !program.eq_ignore_ascii_case("arkdefault") && !vegetation && !decal {
            p.support = Support::Unsupported(format!("program {program}"));
            return p;
        }
        let mut a = ArkDefault::resolve(&m, |n| self.default_parm(n));
        let mut ignored = ignored_features(&m, &a);
        if no_program {
            ignored.push("no mainProgram: arkDefault assumed".into());
        }
        if vegetation {
            // arkvegetation.decl standard pass (O): albedo = `$diffusemap` at
            // m_layerTexCoords[0], `clip(a - 0.5)` under `$alphatest`, then the shared
            // HQ normal/gloss/metallic helpers. No layer blending, colour constant or
            // vertex-colour multiply (vertex colours drive wind animation instead).
            a.extra_layer = 0;
            a.layers[0].use_color_constant = false;
            a.vtx_color_multiply = false;
            // Program state (arkvegetation.decl, O): `cull none` unless the material sets it.
            if void_formats::arkmat::Cull::explicit(&m).is_none() {
                a.cull = void_formats::arkmat::Cull::None;
            }
            ignored.push("arkvegetation: layer 0 only; wind animation, translucency".into());
        }
        if decal {
            // arkdecal.decl (O): `Decal_FetchDiffuseAlbedo` samples layer 0 only (colour
            // constant, vertex-colour multiply, alpha test), normal/gloss/metal from the HQ
            // helpers, full lighting, then blend SRC_ALPHA / ONE_MINUS_SRC_ALPHA with
            // depth GEQUAL and `sv_position.z -= $zbias`. Only `decalType` 0 (mesh decal
            // with its own UVs) is implemented; the projected types need per-instance
            // world-to-decal transforms.
            if a.decal_type != 0 {
                p.support = Support::Unsupported(format!("arkDecal decalType {}", a.decal_type));
                return p;
            }
            a.extra_layer = 0;
            p.decal = true;
        }
        p.alpha_test = a.alpha_test;
        p.two_sided = a.two_sided_lighting;
        let mut load = |src: &Option<String>, usage: Usage, what: &str, layer: usize| {
            let s = src.as_ref()?;
            match texture::load(self.archive, s, usage, self.max_size) {
                Ok(t) => {
                    if usage == Usage::Raw
                        && matches!(t.format, Format::Bc7Unorm | Format::Bc7UnormSrgb)
                    {
                        ignored.push(format!(
                            "layer{layer} {what} is a colour texture ({:?}), sampled as is",
                            t.format
                        ));
                    }
                    Some(t)
                }
                Err(e) => {
                    ignored.push(format!("layer{layer} {what} failed: {e}"));
                    None
                }
            }
        };
        for i in 0..=a.extra_layer {
            let l = &a.layers[i];
            p.textures.diffuse[i] = load(&l.diffuse_map, Usage::Color, "diffuse", i);
            p.textures.normal[i] = load(&l.bump_map, Usage::Raw, "bump", i);
            p.textures.gloss[i] = load(&l.gloss_map, Usage::Raw, "gloss", i);
            p.textures.metallic[i] = load(&l.metallic_map, Usage::Raw, "metallic", i);
            if l.mask_mode != 0 && a.mask_source(i) == i {
                p.textures.mask[i] = load(&l.mask_map, Usage::Raw, "mask", i);
            }
        }
        if a.has_occlusion_map {
            p.textures.occlusion = load(&a.occlusion_map, Usage::Raw, "occlusion", 0);
        }
        if p.textures.diffuse[0].is_none() {
            p.support = Support::Unsupported("layer 0 diffusemap unavailable".into());
            return p;
        }
        p.model = Some(a);
        if !ignored.is_empty() {
            p.support = Support::Partial(ignored);
        }
        p
    }

    /// A material the normal path does not support (another program such as skin or hair,
    /// an unreadable decl, an editor program), read as arkDefault anyway for traced
    /// primitive data: layer 0 only, its diffuse map if the decl names one (`$diffusemap`
    /// spellings, any program), alpha test as declared. `support` is `Partial` with the
    /// reason; `model` is always set (renderParm defaults where the decl has nothing), so
    /// the caller can always build a material (grey without a diffuse map).
    pub fn prepare_fallback(&self, name: &str) -> Prepared {
        let normal = self.prepare(name);
        if normal.drawable() && normal.model.is_some() {
            return normal;
        }
        let why = match &normal.support {
            Support::Unsupported(r) => r.clone(),
            Support::EditorOnly(p) => format!("editor program {p}"),
            _ => "no arkDefault model".into(),
        };
        let m = self
            .archive
            .read_text("material", &material_resource_name(name))
            .ok()
            .and_then(|t| Material::parse(&t).ok())
            .unwrap_or(Material {
                entries: Vec::new(),
            });
        let mut a = ArkDefault::resolve(&m, |n| self.default_parm(n));
        a.extra_layer = 0;
        let mut p = Prepared {
            name: name.to_owned(),
            support: Support::Partial(vec![format!("fallback: {why}")]),
            model: None,
            textures: LayerTextures::default(),
            alpha_test: a.alpha_test,
            two_sided: a.two_sided_lighting,
            decal: false,
        };
        if let Some(s) = a.layers[0].diffuse_map.as_ref() {
            p.textures.diffuse[0] =
                texture::load(self.archive, s, Usage::Color, self.max_size).ok();
        }
        if p.textures.diffuse[0].is_none() {
            // Without a map the alpha test has nothing to test.
            p.alpha_test = false;
            a.alpha_test = false;
        }
        p.model = Some(a);
        p
    }
}

/// Lighting inputs as two vec4: (specularTintDielectric.xyz, sheen),
/// (diffuseAlbedoMetallic.xyz, sheenImportance).
pub fn pack_lighting(a: &ArkDefault) -> [u32; 8] {
    let t = a.specular_tint_dielectric;
    let d = a.diffuse_albedo_metallic;
    [
        t[0],
        t[1],
        t[2],
        a.sheen,
        d[0],
        d[1],
        d[2],
        a.sheen_importance,
    ]
    .map(f32::to_bits)
}

/// Number of 32-bit words in [`pack_params`] (21 x vec4).
pub const PARAM_WORDS: usize = 84;

/// Packs the resolved model into 21 vec4s (layout of `ArkParams` in the shaders):
/// scale_bias[3], mask_scale_bias[3], color_constant[3], rescale[3] (f32);
/// modes[3] (u32: mask mode, vtx channel, invert, bits 1 uv_set / 2 mask_uv_set /
/// 4 use_color_constant), reuse[3] (u32: diffuse, bump, gloss, metallic),
/// mask_source (u32), ranges (f32: ward.x, ward.y, metal min, metal max),
/// flags (u32: extra_layer, bits 1 gloss / 2 metallic / 4 bump / 8 vtx colour multiply /
/// 16 alpha test). Map flags are set only when layer 0's map was loaded.
pub fn pack_params(
    a: &ArkDefault,
    has_gloss: bool,
    has_metal: bool,
    has_bump: bool,
    alpha_test: bool,
) -> [u32; PARAM_WORDS] {
    let mut w = [0u32; PARAM_WORDS];
    let put_f = |slot: usize, v: [f32; 4], w: &mut [u32; PARAM_WORDS]| {
        for k in 0..4 {
            w[slot * 4 + k] = v[k].to_bits();
        }
    };
    for (i, l) in a.layers.iter().enumerate() {
        put_f(i, l.scale_bias, &mut w);
        put_f(3 + i, l.mask_scale_bias, &mut w);
        put_f(6 + i, l.color_constant, &mut w);
        put_f(9 + i, [l.rescale[0], l.rescale[1], 0.0, 0.0], &mut w);
        let bits = u32::from(l.uv_set == 1)
            | (u32::from(l.mask_uv_set == 1) << 1)
            | (u32::from(l.use_color_constant) << 2);
        let modes = [
            l.mask_mode.clamp(0, 2) as u32,
            l.vtx_channel as u32,
            u32::from(l.invert_mask),
            bits,
        ];
        let reuse = [
            l.diffuse_reuse.clamp(0, 2) as u32,
            l.bump_reuse.clamp(0, 2) as u32,
            l.gloss_reuse.clamp(0, 2) as u32,
            l.metallic_reuse.clamp(0, 2) as u32,
        ];
        w[(12 + i) * 4..(12 + i) * 4 + 4].copy_from_slice(&modes);
        w[(15 + i) * 4..(15 + i) * 4 + 4].copy_from_slice(&reuse);
    }
    w[18 * 4..18 * 4 + 4].copy_from_slice(&[
        a.mask_source(0) as u32,
        a.mask_source(1) as u32,
        a.mask_source(2) as u32,
        0,
    ]);
    put_f(
        19,
        [
            a.ward_roughness[0],
            a.ward_roughness[1],
            a.metallic_min_max[0],
            a.metallic_min_max[1],
        ],
        &mut w,
    );
    let flags = u32::from(a.has_gloss_map && has_gloss)
        | (u32::from(a.has_metallic_map && has_metal) << 1)
        | (u32::from(a.has_bump_map && has_bump) << 2)
        | (u32::from(a.vtx_color_multiply) << 3)
        | (u32::from(alpha_test) << 4);
    w[20 * 4..20 * 4 + 4].copy_from_slice(&[a.extra_layer as u32, flags, 0, 0]);
    w
}
