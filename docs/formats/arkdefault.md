# arkDefault material model

Implementation: `crates/void-formats/src/arkmat.rs` (engine-independent),
`crates/void-viewer/src/ark_layers.wesl` + `layered.rs` (Bevy adapter),
`crates/void-compare/src/matval.rs` (validation).

## Primary evidence (O)

The game ships its shader **source** as `renderProg` decls (270 in group 0), e.g.
`generated/decls/renderprog/arkdefault.decl`, `arkdefaulthelpers.decl` (1,508 lines),
`pixel.decl`, `vertex.decl`, `global.decl`, `helper.decl`, `brdf_ward.decl`. Parameters and
permutation options have shipped defaults as `renderParm` decls
(`generated/decls/renderparm/<name>.decl`: `{ option N range [..] }` or
`{ Uniform floatN { .. } range .. }`). Extract with
`void-tool extract --group 0 --kind renderProg --contains renderprog/ --out <dir> --max 300`.

## Transcribed semantics (HQ path; function names from arkdefaulthelpers.decl)

- Layer UVs (`pixel.decl initLayeredTexcoord`): `uv = (UVset ? uv1 : uv0) * ScaleBias.xy + ScaleBias.zw`;
  masks use `Mask_UVset` and `MaskScaleBias`. Layers beyond `extraLayer` reuse layer 0's UV.
- Masks (`HQ_FetchMasks`), per layer i: layer 0 only when `use_Layer0_ColorConstant`;
  mode 0 = vertex colour `[VtxColorMaskChannel]`; mode 1 = mask map `.r`; mode 2 =
  `LinearStep(RescaleValues.x, .y, mask * vertexColour[channel])`; `InvertMask` -> `1 - m`;
  mask maps can be reused from earlier layers (`maskReuseLayer`).
  `LinearStep(a,b,t) = saturate((t-a)/(b-a))` (`helper.decl`).
- Albedo (`HQ_FetchDiffuseAlbedo`): layer 0 diffuse, times `lerp(1, Layer0_ColorConstant, mask0)`
  if enabled; `lerp` to layer 1/2 diffuse (times their colour constants) by mask1/mask2;
  `VtxColorMultiply` multiplies by vertex colour; alpha test clips layer-0 alpha at 0.5.
- Normal (`HQ_FetchAndComputeNormalAndTBN`): blend raw BC5 `.xy` by masks, then
  `UnpackBC5NormalMap`: `z = 1.001 - saturate(dot(xy,xy))`, normalize; TBN from vertex tangent,
  bitangent `cross(n, t) * t.w`, vertex normal.
- Gloss (`HQ_FetchGloss`): blend layer gloss maps by masks, then
  `gloss = lerp(wardRoughness.x, wardRoughness.y, g)`; without a map `wardRoughness.y`.
  **roughness = 1 - gloss** (`_surf.roughness = 1.0 - HQ_FetchGloss`).
- Metallic (`HQ_FetchMetallic`): same blend, `lerp(metallicMinMax.x, .y, m)`; default `.y`.
- Defaults observed: `wardRoughness (0, 0.5)`, `metallicMinMax (0, 0.5)`, `RescaleValues
  (0.4, 0.6)`, `ScaleBias/MaskScaleBias (1,1,0,0)`, `ColorConstant (1,1,1,1)`,
  `LayerN_VtxColorMaskChannel` default 3, other options 0. `use_LayerN_ColorConstant` has
  no default decl (treated as 0 when absent; H).
- Not modelled: static/dynamic decals, material effects (layer change), detail bump,
  occlusion, emissive, sheen, translucency, parallax, wind; lighting (Ward BRDF, IBL, SH).

## Debug views (O, RX002)

`r_showGlossOnly 1` / `r_showMetalOnly 1` work in the retail build despite "idStudio only"
descriptions. Source (`DebugDrawOverride`): output `1 - roughness` (= gloss) or `metallic`,
divided by `EngineLuminanceFactor` ("makeLDR"); fog appears to apply afterwards.
`r_skipFog 1` removes fog.

## Validation (E020, `void-compare --gloss --metal`, RX002 captures)

One scale factor is fitted per capture (the 1/ELF of that frame; it differs between
captures because auto-exposure adapts to each debug image). Samples: depth-matched,
nearer than 25 m (P1) / 40 m (P2), stride 4.

| Capture | Samples | Gloss r | Metallic r | Fitted scale per material (gloss) |
| --- | --- | --- | --- | --- |
| P2 (all) | 60,480 | **0.888** | **0.941** | 1.31-1.48 across 5 materials |
| P2 layered materials only | 51,309 | 0.875 | 0.885 | brick blend r 0.70, roof blend r 0.83, dock stone blend r 0.63 |
| P1 non-layered materials | ~40,000 | 0.65-0.81 per material | — | 0.55-0.61 across 6 materials |
| P1 `blend_mud_dry_rock_01` (ground) | 38,704 | **0.48 best (unexplained)** | — | 0.31 vs ~0.57 elsewhere |

Interpretation: the transcription matches the game for ordinary and several layered
materials. The P1 ground has an additional gloss contribution not in the model (candidates:
static decals, wetness; not yet identified). Residual noise comes from approximate texture
LOD selection (nearest mip from neighbouring-sample UV steps, bilinear).

## Vertex colour byte order (resolved 2026-10-16)

O: the world/model vertex layouts built by `sub_1402D0DC0` declare COLOR (semantic 5) with
engine format `0x0A050100`, which `sub_1402C2EF0`/`sub_1402C2F50` map to DXGI 28
`R8G8B8A8_UNORM` (kind 10, sub 5, order 1; order 3 would be BGRA 87, never used). In the
stream-1 layout COLOR sits at offset 16 after TEXCOORD0/1 (0, 4), NORMAL (8) and TANGENT (12),
both `R8G8B8A8_SNORM`. So shader channels x, y, z, w = bytes 16, 17, 18, 19 (RGBA), as the
viewer already assumed. Input-layout creation: `sub_1402ABA10` (semantic table `VERTEX_*`).
