// DXBC patching of the engine's forward-lighting pixel shaders: three extra render target outputs with material
// data for SSGI and SSR, taken from texture samples the shaders already make.
//
//   o2 (kAlbedoSlot):   rgb = base colour (first ro_diffusemap sample)
//   o3 (kSpecularSlot): rgb = the game's specular reflection (first ro_ibl_texcubemaparray sample_l)
//   o4 (kSurfaceSlot):  r = that sample's LOD (the material's roughness in cubemap mips, -1 = no IBL sample),
//                       g = metallic (first ro_metallicmap sample)
//   w of all three = the alpha of the colour output (o0): the opacity of decals and other blended draws, so
//                    material.cpp can blend the material data by it (wet decals get their own roughness)
//
// o0 is computed into a new temp instead and written, with the alphas, before every return. The forward shaders
// write o0, some also o1 (refraction map), so 2..4 are free in all of them. There is no SM5 assembler: the token
// stream is edited directly (declarations, movs, the output signature) and the container checksum is recomputed
// (the runtime validates it).
#pragma once

#include "core.h"

#include <vector>

namespace shader_patch
{
    constexpr int kAlbedoSlot   = 2;
    constexpr int kSpecularSlot = 3;
    constexpr int kSurfaceSlot  = 4;

    struct MaterialPatch
    {
        std::vector<std::byte> bytecode;  // empty: not a forward-lighting shader, or no safe patch
        bool                   hasSurface = false;  // the shader samples the IBL cubemaps (roughness is known)
    };

    MaterialPatch AddMaterialOutputs(std::span<const std::byte> bytecode);
}
