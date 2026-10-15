#include "shader_info.h"

#include "dxbc.h"

namespace
{
    // {7D3B6E2A-5C1F-4E0B-9A47-2F6D1C8E4B13}
    constexpr GUID kShaderInfoGuid = { 0x7d3b6e2a, 0x5c1f, 0x4e0b, { 0x9a, 0x47, 0x2f, 0x6d, 0x1c, 0x8e, 0x4b, 0x13 } };

    // D3D_SHADER_INPUT_TYPE -> register letter.
    char RegisterLetter(uint32_t type)
    {
        switch (type)
        {
        case dxbc::kInputCbuffer: return 'b';
        case dxbc::kInputSampler: return 's';
        case 4:               // RWTexture / RWBuffer
        case 6:               // RWStructuredBuffer
        case 8:               // RWByteAddressBuffer
        case 9:               // AppendStructuredBuffer
        case 10:              // ConsumeStructuredBuffer
        case 11: return 'u';  // RWStructuredBuffer with counter
        default: return 't';  // tbuffer, texture, StructuredBuffer, ByteAddressBuffer
        }
    }

    // Builds "t0:name u0:name b0:name ..." from the RDEF chunk (samplers left out, as many as fit), and the mask of
    // t slots that are textures.
    void ReadBindingNames(std::span<const std::byte> code, std::span<char> out, uint16_t& textureSlots)
    {
        size_t used = 0;
        for (const dxbc::Binding& binding : dxbc::Bindings(code, dxbc::Chunks(code)))
        {
            char letter = RegisterLetter(binding.type);
            if (letter == 's')
                continue;
            if (binding.type == dxbc::kInputTexture && binding.slot < 16)
                textureSlots |= static_cast<uint16_t>(1u << binding.slot);
            size_t room = out.size() - 1 - used;  // keeps the terminating zero
            auto result = std::format_to_n(out.data() + used, room, "{}{}:{} ", letter, binding.slot, binding.name.substr(0, 80));
            if (static_cast<size_t>(result.size) > room)
                break;  // full: the last entry would be cut off
            used += static_cast<size_t>(result.size);
        }
        out[used] = '\0';
    }
}

void TagShader(ID3D11DeviceChild* shader, std::span<const std::byte> bytecode, bool materialOutputs,
               bool materialSurface)
{
    ShaderInfo info;
    info.materialOutputs = materialOutputs;
    info.materialSurface = materialSurface;
    info.hash = Fnv1a64(bytecode);
    info.size = static_cast<uint32_t>(bytecode.size());
    ReadBindingNames(bytecode, info.bindings, info.textureSlots);
    std::string_view bindings = info.bindings;
    info.isTaa = bindings.contains("taahistory") || bindings.contains("taaresult");
    info.isDepthDownsample = bindings.contains("t0:ro_downsampling_depthbuffer ");
    shader->SetPrivateData(kShaderInfoGuid, sizeof(info), &info);
}

bool ReadShaderInfo(ID3D11DeviceChild* shader, ShaderInfo& out)
{
    UINT size = sizeof(out);
    return shader && SUCCEEDED(shader->GetPrivateData(kShaderInfoGuid, &size, &out)) && size == sizeof(out);
}
