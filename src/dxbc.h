// Reading DXBC shader containers: chunks and the RDEF resource bindings (shader identification and patching).
#pragma once

#include "core.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace dxbc
{
    // D3D_SHADER_INPUT_TYPE values used here.
    constexpr uint32_t kInputCbuffer = 0;
    constexpr uint32_t kInputTexture = 2;
    constexpr uint32_t kInputSampler = 3;

    // Little-endian dword at `offset`, 0 past the end.
    inline uint32_t Read32(std::span<const std::byte> data, size_t offset)
    {
        uint32_t value = 0;
        if (offset + 4 <= data.size())
            memcpy(&value, data.data() + offset, 4);
        return value;
    }

    constexpr uint32_t FourCC(const char (&s)[5])
    {
        return uint32_t(uint8_t(s[0])) | uint32_t(uint8_t(s[1])) << 8 | uint32_t(uint8_t(s[2])) << 16 |
               uint32_t(uint8_t(s[3])) << 24;
    }

    struct Chunk
    {
        uint32_t fourcc;
        size_t   offset;  // of the chunk data (after the 8-byte header)
        uint32_t size;
    };

    // The container's chunks in order; empty if it isn't a well-formed DXBC container.
    inline std::vector<Chunk> Chunks(std::span<const std::byte> code)
    {
        std::vector<Chunk> chunks;
        if (code.size() < 32 || memcmp(code.data(), "DXBC", 4) != 0)
            return chunks;
        uint32_t count = Read32(code, 28);
        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t offset = Read32(code, 32 + size_t{ i } * 4);
            uint32_t size = Read32(code, offset + 4);
            if (size_t{ offset } + 8 + size > code.size())
                return {};
            chunks.push_back({ Read32(code, offset), size_t{ offset } + 8, size });
        }
        return chunks;
    }

    inline const Chunk* FindChunk(const std::vector<Chunk>& chunks, uint32_t fourcc)
    {
        auto found = std::ranges::find(chunks, fourcc, &Chunk::fourcc);
        return found != chunks.end() ? &*found : nullptr;
    }

    struct Binding
    {
        std::string_view name;  // points into the bytecode
        uint32_t         type;  // D3D_SHADER_INPUT_TYPE
        uint32_t         slot;  // bind point
    };

    // Resource bindings of an RDEF chunk's data, up to the first malformed record.
    inline std::vector<Binding> Bindings(std::span<const std::byte> rdef)
    {
        std::vector<Binding> bindings;
        uint32_t count = Read32(rdef, 8), offset = Read32(rdef, 12);
        for (uint32_t i = 0; i < count; ++i)
        {
            size_t record = offset + size_t{ i } * 32;
            uint32_t nameOffset = Read32(rdef, record);
            if (record + 24 > rdef.size() || nameOffset >= rdef.size())
                break;
            const char* name = reinterpret_cast<const char*>(rdef.data() + nameOffset);
            bindings.push_back({ std::string_view(name, strnlen(name, rdef.size() - nameOffset)), Read32(rdef, record + 4),
                                 Read32(rdef, record + 20) });
        }
        return bindings;
    }

    // The bindings of a whole container (empty without an RDEF chunk).
    inline std::vector<Binding> Bindings(std::span<const std::byte> code, const std::vector<Chunk>& chunks)
    {
        const Chunk* rdef = FindChunk(chunks, FourCC("RDEF"));
        return rdef ? Bindings(code.subspan(rdef->offset, rdef->size)) : std::vector<Binding>{};
    }
}
