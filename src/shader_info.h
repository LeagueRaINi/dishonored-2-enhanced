// Engine shader identification. The game's compiled shaders keep their RDEF chunk, whose binding names
// ("ro_<shader>_<param>" / "rw_<shader>_<param>") identify the passes the mod cares about.
#pragma once

#include "core.h"

struct ShaderInfo
{
    uint64_t hash = 0;
    uint32_t size = 0;
    bool     isTaa = false;              // binds the TAA history/result
    bool     isDepthDownsample = false;  // t0 is the full-resolution engine depth buffer
    uint16_t textureSlots = 0;           // bit n: tn is a texture the shader declares
    bool     materialOutputs = false;    // patched to write material data to slots 2..4 (shader_patch.h)
    bool     materialSurface = false;    // ... including the roughness (the shader samples the IBL cubemaps)
    char     bindings[512] = {};         // e.g. "t0:ro_taa_history u1:rw_taa_result b1:PerViewCB "
};

// Parses the bytecode and attaches the result to the shader as private data.
// bytecode: the original (hash, bindings), also for patched shaders.
void TagShader(ID3D11DeviceChild* shader, std::span<const std::byte> bytecode, bool materialOutputs = false,
               bool materialSurface = false);
bool ReadShaderInfo(ID3D11DeviceChild* shader, ShaderInfo& out);
