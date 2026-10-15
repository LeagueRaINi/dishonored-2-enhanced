#include "shader_patch.h"

#include "dxbc.h"

#include <algorithm>
#include <bit>
#include <optional>
#include <string_view>

namespace
{
    using dxbc::Read32;

    // d3d11TokenizedProgramFormat.hpp
    constexpr uint32_t kOpCustomData = 53;
    constexpr uint32_t kOpMov        = 54;
    constexpr uint32_t kOpRet        = 62;
    constexpr uint32_t kOpRetc       = 63;
    constexpr uint32_t kOpSample     = 69;
    constexpr uint32_t kOpSampleL    = 72;
    constexpr uint32_t kOpSampleD    = 73;
    constexpr uint32_t kOpSampleB    = 74;
    constexpr uint32_t kOpDclOutput  = 101;
    constexpr uint32_t kOpDclTemps   = 104;

    constexpr uint32_t kOperandTemp     = 0;
    constexpr uint32_t kOperandOutput   = 2;
    constexpr uint32_t kOperandResource = 7;

    // Texture slot bound to `name`.
    std::optional<uint32_t> TextureSlot(const std::vector<dxbc::Binding>& bindings, std::string_view name)
    {
        for (const dxbc::Binding& b : bindings)
            if (b.type == dxbc::kInputTexture && b.name == name)
                return b.slot;
        return std::nullopt;
    }

    // Length in dwords of the operand starting at tokens[i] (0 if malformed).
    size_t OperandLength(std::span<const uint32_t> tokens, size_t i)
    {
        if (i >= tokens.size())
            return 0;
        uint32_t token = tokens[i];
        size_t length = 1;
        bool extended = token >> 31;
        while (extended)
        {
            if (i + length >= tokens.size())
                return 0;
            extended = tokens[i + length] >> 31;
            ++length;
        }
        uint32_t type = (token >> 12) & 0xFF;
        if (type == 4 || type == 5)  // immediate32 / immediate64 values
        {
            uint32_t components = (token & 3) == 1 ? 1 : (token & 3) == 2 ? 4 : 0;
            return length + components * (type == 5 ? 2 : 1);
        }
        uint32_t dimensions = (token >> 20) & 3;
        for (uint32_t d = 0; d < dimensions; ++d)
        {
            uint32_t representation = (token >> (22 + 3 * d)) & 7;
            switch (representation)
            {
            case 0: length += 1; break;  // immediate32
            case 1: length += 2; break;  // immediate64
            case 2: { size_t r = OperandLength(tokens, i + length); if (!r) return 0; length += r; break; }
            case 3: { length += 1; size_t r = OperandLength(tokens, i + length); if (!r) return 0; length += r; break; }
            case 4: { length += 2; size_t r = OperandLength(tokens, i + length); if (!r) return 0; length += r; break; }
            default: return 0;
            }
        }
        return length;
    }

    // Index of a simple operand (1D immediate index).
    std::optional<uint32_t> OperandIndex(std::span<const uint32_t> tokens, size_t i)
    {
        uint32_t token = tokens[i];
        size_t at = i + 1;
        bool extended = token >> 31;
        while (extended && at < tokens.size())
            extended = tokens[at++] >> 31;
        if (((token >> 20) & 3) != 1 || ((token >> 22) & 7) != 0 || at >= tokens.size())
            return std::nullopt;
        return tokens[at];
    }

    // Declarations (dcl_*, the immediate constant buffer) must all precede the first instruction.
    bool IsDeclaration(uint32_t opcode)
    {
        return opcode == kOpCustomData || (opcode >= 88 && opcode <= 106) || (opcode >= 143 && opcode <= 163);
    }

    struct Sample
    {
        uint32_t opcode = 0;
        uint32_t resource = 0;     // texture slot
        uint32_t destRegister = 0;
        uint32_t destMask = 0;     // write mask of the destination
        size_t   end = 0;          // token index after the instruction
        size_t   lodStart = 0, lodLength = 0;  // sample_l: the LOD operand
    };

    struct Program
    {
        size_t              dclOutputEnd = 0;  // after the last dcl_output
        size_t              dclEnd = 0;        // the first instruction
        size_t              tempCount = 0;     // token index of dcl_temps' register count (0: none)
        std::vector<Sample> samples;           // texture samples into temps, in program order
        std::vector<size_t> colorWrites;       // operand tokens naming o0 (outputs are only ever destinations)
        std::vector<size_t> returns;           // ret / retc instructions
    };

    // Walks the program tokens (after the version and length dwords).
    std::optional<Program> Analyze(std::span<const uint32_t> tokens)
    {
        Program program;
        for (size_t i = 2; i < tokens.size();)
        {
            uint32_t opcode = tokens[i] & 0x7FF;
            size_t length = opcode == kOpCustomData ? (i + 1 < tokens.size() ? tokens[i + 1] : 0) : (tokens[i] >> 24) & 0x7F;
            if (length == 0 || i + length > tokens.size())
                return std::nullopt;
            if (opcode == kOpDclOutput)
                program.dclOutputEnd = i + length;
            if (opcode == kOpDclTemps)
                program.tempCount = i + 1;
            if (!program.dclEnd && !IsDeclaration(opcode))
                program.dclEnd = i;
            if (!IsDeclaration(opcode))
            {
                if (opcode == kOpRet || opcode == kOpRetc)
                    program.returns.push_back(i);
                size_t at = i + 1;
                bool extendedOpcode = tokens[i] >> 31;
                while (extendedOpcode && at < i + length)
                    extendedOpcode = tokens[at++] >> 31;
                while (at < i + length)
                {
                    size_t operandLength = OperandLength(tokens, at);
                    if (!operandLength)
                        return std::nullopt;
                    if (((tokens[at] >> 12) & 0xFF) == kOperandOutput && OperandIndex(tokens, at) == 0u)
                        program.colorWrites.push_back(at);
                    at += operandLength;
                }
            }
            if (opcode == kOpSample || opcode == kOpSampleL || opcode == kOpSampleD || opcode == kOpSampleB)
            {
                size_t at = i + 1;
                bool extended = tokens[i] >> 31;
                while (extended && at < i + length)
                    extended = tokens[at++] >> 31;
                size_t dest = at;
                size_t address = dest + OperandLength(tokens, dest);
                size_t resource = address + OperandLength(tokens, address);
                size_t sampler = resource + OperandLength(tokens, resource);
                size_t lod = sampler + OperandLength(tokens, sampler);
                std::optional<uint32_t> reg = OperandIndex(tokens, dest);
                std::optional<uint32_t> slot = resource < i + length ? OperandIndex(tokens, resource) : std::nullopt;
                if (reg && slot && ((tokens[dest] >> 12) & 0xFF) == kOperandTemp && ((tokens[dest] >> 2) & 3) == 0 &&
                    ((tokens[resource] >> 12) & 0xFF) == kOperandResource)
                {
                    Sample sample{ opcode, *slot, *reg, (tokens[dest] >> 4) & 0xF, i + length };
                    if (opcode == kOpSampleL && lod < i + length)
                    {
                        sample.lodStart = lod;
                        sample.lodLength = OperandLength(tokens, lod);
                    }
                    program.samples.push_back(sample);
                }
            }
            i += length;
        }
        if (!program.dclOutputEnd || !program.dclEnd || program.dclEnd < program.dclOutputEnd || !program.tempCount ||
            program.returns.empty() || program.colorWrites.empty())
            return std::nullopt;
        return program;
    }

    const Sample* FirstSample(const Program& program, std::optional<uint32_t> slot, bool needLod = false)
    {
        if (!slot)
            return nullptr;
        for (const Sample& s : program.samples)
            if (s.resource == *slot && (!needLod || s.lodLength))
                return &s;
        return nullptr;
    }

    // Operand tokens.
    constexpr uint32_t OutputMask(uint32_t mask) { return 0x2u | (mask << 4) | (kOperandOutput << 12) | (1u << 20); }
    constexpr uint32_t TempSelect(uint32_t component) { return 0x2u | (2u << 2) | (component << 4) | (kOperandTemp << 12) | (1u << 20); }
    constexpr uint32_t kTempSwizzleXyzx = 0x2u | (1u << 2) | (0x24u << 4) | (kOperandTemp << 12) | (1u << 20);
    constexpr uint32_t kTempSwizzleXyzw = 0x2u | (1u << 2) | (0xE4u << 4) | (kOperandTemp << 12) | (1u << 20);
    constexpr uint32_t TempMask(uint32_t mask) { return 0x2u | (mask << 4) | (kOperandTemp << 12) | (1u << 20); }
    constexpr uint32_t kImmediate4 = 0x2u | (4u << 12);

    struct Insertion
    {
        size_t                position;
        int                   order;   // same position: higher order ends up later
        std::vector<uint32_t> tokens;
    };

    std::vector<uint32_t> Mov(uint32_t dest, uint32_t outputIndex, std::span<const uint32_t> source)
    {
        std::vector<uint32_t> t = { 0, dest, outputIndex };
        t.insert(t.end(), source.begin(), source.end());
        t[0] = kOpMov | (static_cast<uint32_t>(t.size()) << 24);
        return t;
    }

    // Output signature with one more SV_Target element (copy of target 0) at `reg`.
    std::optional<std::vector<std::byte>> AddTarget(std::span<const std::byte> osgn, uint32_t reg)
    {
        uint32_t count = Read32(osgn, 0);
        constexpr size_t kHeader = 8, kElement = 24;
        if (kHeader + count * kElement > osgn.size())
            return std::nullopt;
        std::optional<size_t> target0;
        for (uint32_t i = 0; i < count; ++i)
        {
            size_t e = kHeader + i * kElement;
            uint32_t nameOffset = Read32(osgn, e);
            if (nameOffset >= osgn.size())
                return std::nullopt;
            const char* name = reinterpret_cast<const char*>(osgn.data() + nameOffset);
            if (_strnicmp(name, "SV_Target", 9) == 0 && Read32(osgn, e + 4) == 0)
                target0 = e;
        }
        if (!target0)
            return std::nullopt;

        std::vector<std::byte> out(osgn.size() + kElement);
        memcpy(out.data(), osgn.data(), kHeader + count * kElement);
        memcpy(out.data() + kHeader + count * kElement, osgn.data() + *target0, kElement);
        memcpy(out.data() + kHeader + (count + 1) * kElement, osgn.data() + kHeader + count * kElement,
               osgn.size() - kHeader - count * kElement);
        uint32_t newCount = count + 1;
        memcpy(out.data(), &newCount, 4);
        for (uint32_t i = 0; i <= count; ++i)  // strings moved back by one element
        {
            size_t e = kHeader + i * kElement;
            uint32_t nameOffset = Read32(out, e) + kElement;
            memcpy(out.data() + e, &nameOffset, 4);
        }
        size_t added = kHeader + count * kElement;
        memcpy(out.data() + added + 4, &reg, 4);   // semantic index
        memcpy(out.data() + added + 16, &reg, 4);  // register
        out[added + 20] = std::byte{ 0xF };        // mask xyzw
        out[added + 21] = std::byte{ 0x0 };        // all components written
        return out;
    }

    // ---- DXBC checksum: MD5 over the container after the hash field, with its own final block layout ----------

    struct Md5
    {
        uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };

        void Transform(const uint8_t* block)
        {
            static constexpr uint32_t k[64] = {
                0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
                0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
                0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
                0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
                0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
                0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
                0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
                0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
            };
            static constexpr uint32_t r[64] = {
                7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
            };
            uint32_t m[16];
            memcpy(m, block, 64);
            uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
            for (uint32_t i = 0; i < 64; ++i)
            {
                uint32_t f, g;
                if (i < 16)      { f = (b & c) | (~b & d); g = i; }
                else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
                else if (i < 48) { f = b ^ c ^ d;          g = (3 * i + 5) % 16; }
                else             { f = c ^ (b | ~d);       g = (7 * i) % 16; }
                uint32_t t = d;
                d = c;
                c = b;
                uint32_t x = a + f + k[i] + m[g];
                b += (x << r[i]) | (x >> (32 - r[i]));
                a = t;
            }
            state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        }
    };

    void UpdateChecksum(std::vector<std::byte>& container)
    {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(container.data()) + 20;
        uint32_t size = static_cast<uint32_t>(container.size() - 20);
        uint32_t bits = size * 8, bitsPart2 = (bits >> 2) | 1;
        uint32_t full = size / 64 * 64, left = size - full;
        Md5 md5;
        for (uint32_t i = 0; i < full; i += 64)
            md5.Transform(data + i);
        uint8_t block[64] = {};
        if (left >= 56)
        {
            memcpy(block, data + full, left);
            block[left] = 0x80;
            md5.Transform(block);
            memset(block, 0, 64);
            memcpy(block, &bits, 4);
            memcpy(block + 60, &bitsPart2, 4);
        }
        else
        {
            memcpy(block, &bits, 4);
            memcpy(block + 4, data + full, left);
            block[4 + left] = 0x80;
            memcpy(block + 60, &bitsPart2, 4);
        }
        md5.Transform(block);
        memcpy(container.data() + 4, md5.state, 16);
    }

    uint32_t NextFreeTarget(std::span<const std::byte> osgn)
    {
        uint32_t count = Read32(osgn, 0), next = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t reg = Read32(osgn, 8 + i * 24 + 16);
            if (reg < 8)
                next = std::max(next, reg + 1);
        }
        return next;
    }
}

shader_patch::MaterialPatch shader_patch::AddMaterialOutputs(std::span<const std::byte> code)
{
    using dxbc::FourCC;
    std::vector<dxbc::Chunk> chunks = dxbc::Chunks(code);
    const dxbc::Chunk* osgn = dxbc::FindChunk(chunks, FourCC("OSGN"));
    const dxbc::Chunk* shex = dxbc::FindChunk(chunks, FourCC("SHEX"));
    if (!shex)
        shex = dxbc::FindChunk(chunks, FourCC("SHDR"));
    if (!osgn || !shex || shex->size % 4)
        return {};
    std::vector<dxbc::Binding> bindings = dxbc::Bindings(code, chunks);
    if (!std::ranges::contains(bindings, "ro_tlf_ll_heads", &dxbc::Binding::name))
        return {};  // not a forward-lighting shader
    std::span<const std::byte> osgnData = code.subspan(osgn->offset, osgn->size);
    if (NextFreeTarget(osgnData) > kAlbedoSlot)
        return {};

    std::vector<uint32_t> tokens(shex->size / 4);
    memcpy(tokens.data(), code.data() + shex->offset, shex->size);
    std::optional<Program> program = Analyze(tokens);
    if (!program)
        return {};
    const Sample* diffuse = FirstSample(*program, TextureSlot(bindings, "ro_diffusemap"));
    const Sample* metallic = FirstSample(*program, TextureSlot(bindings, "ro_metallicmap"));
    const Sample* ibl = FirstSample(*program, TextureSlot(bindings, "ro_ibl_texcubemaparray"), true);
    if (!diffuse && !ibl)
        return {};

    std::optional<std::vector<std::byte>> newOsgn = AddTarget(osgnData, kAlbedoSlot);
    if (newOsgn)
        newOsgn = AddTarget(*newOsgn, kSpecularSlot);
    if (newOsgn)
        newOsgn = AddTarget(*newOsgn, kSurfaceSlot);
    if (!newOsgn)
        return {};

    // The colour output is computed into a new temp instead (outputs are write-only), so its final alpha, the
    // opacity of decals and other blended draws, can also go to the material outputs' w at every return.
    const uint32_t colorTemp = tokens[program->tempCount];
    tokens[program->tempCount] = colorTemp + 1;
    for (size_t operand : program->colorWrites)
    {
        size_t index = operand + 1;  // after the operand token and any extended operand tokens
        for (bool extended = tokens[operand] >> 31; extended; ++index)
            extended = tokens[index] >> 31;
        tokens[operand] = (tokens[operand] & ~(0xFFu << 12)) | (kOperandTemp << 12);
        tokens[index] = colorTemp;
    }

    // Declarations, defaults (unknown albedo, no IBL sample: LOD -1), a copy after each sample, and the epilogue.
    std::vector<Insertion> insertions;
    for (uint32_t slot : { uint32_t(kAlbedoSlot), uint32_t(kSpecularSlot), uint32_t(kSurfaceSlot) })
        insertions.push_back({ program->dclOutputEnd, 0, { kOpDclOutput | (3u << 24), OutputMask(0xF), slot } });
    const uint32_t zero[] = { kImmediate4, 0, 0, 0, 0 };
    const uint32_t opaque[] = { kImmediate4, 0, 0, 0, std::bit_cast<uint32_t>(1.0f) };
    const uint32_t noSurface[] = { kImmediate4, std::bit_cast<uint32_t>(-1.0f), 0, 0, 0 };
    insertions.push_back({ program->dclEnd, 1, Mov(TempMask(0xF), colorTemp, opaque) });
    insertions.push_back({ program->dclEnd, 1, Mov(OutputMask(0xF), kAlbedoSlot, zero) });
    insertions.push_back({ program->dclEnd, 1, Mov(OutputMask(0xF), kSpecularSlot, zero) });
    insertions.push_back({ program->dclEnd, 1, Mov(OutputMask(0xF), kSurfaceSlot, noSurface) });
    if (diffuse)
    {
        const uint32_t source[] = { kTempSwizzleXyzx, diffuse->destRegister };
        insertions.push_back({ diffuse->end, 2, Mov(OutputMask(0x7), kAlbedoSlot, source) });
    }
    if (metallic && metallic->destMask)
    {
        uint32_t component = static_cast<uint32_t>(std::countr_zero(metallic->destMask));
        const uint32_t source[] = { TempSelect(component), metallic->destRegister };
        insertions.push_back({ metallic->end, 2, Mov(OutputMask(0x2), kSurfaceSlot, source) });
    }
    if (ibl)
    {
        const uint32_t radiance[] = { kTempSwizzleXyzx, ibl->destRegister };
        insertions.push_back({ ibl->end, 2, Mov(OutputMask(0x7), kSpecularSlot, radiance) });
        insertions.push_back({ ibl->end, 3, Mov(OutputMask(0x1), kSurfaceSlot,
                                                std::span(tokens).subspan(ibl->lodStart, ibl->lodLength)) });
    }
    const uint32_t color[] = { kTempSwizzleXyzw, colorTemp };
    const uint32_t alpha[] = { TempSelect(3), colorTemp };
    for (size_t ret : program->returns)
    {
        insertions.push_back({ ret, 4, Mov(OutputMask(0xF), 0, color) });
        for (uint32_t slot : { uint32_t(kAlbedoSlot), uint32_t(kSpecularSlot), uint32_t(kSurfaceSlot) })
            insertions.push_back({ ret, 5, Mov(OutputMask(0x8), slot, alpha) });
    }
    // Back to front, so earlier positions stay valid; at one position the higher order is inserted first.
    std::ranges::sort(insertions, [](const Insertion& a, const Insertion& b) {
        return a.position != b.position ? a.position > b.position : a.order > b.order;
    });
    for (const Insertion& insertion : insertions)
        tokens.insert(tokens.begin() + static_cast<ptrdiff_t>(insertion.position), insertion.tokens.begin(),
                      insertion.tokens.end());
    tokens[1] = static_cast<uint32_t>(tokens.size());  // program length in dwords

    // Reassemble the container with the replaced chunks.
    std::vector<std::vector<std::byte>> bodies;
    for (const dxbc::Chunk& c : chunks)
    {
        if (&c == shex)
        {
            std::vector<std::byte> body(tokens.size() * 4);
            memcpy(body.data(), tokens.data(), body.size());
            bodies.push_back(std::move(body));
        }
        else if (&c == osgn)
        {
            bodies.push_back(*newOsgn);
        }
        else
        {
            auto data = code.subspan(c.offset, c.size);
            bodies.emplace_back(data.begin(), data.end());
        }
    }
    size_t headerSize = 32 + chunks.size() * 4;
    size_t total = headerSize;
    for (const auto& b : bodies)
        total += 8 + b.size();

    MaterialPatch result;
    result.hasSurface = ibl != nullptr;
    result.bytecode.resize(total);
    memcpy(result.bytecode.data(), code.data(), 32);  // magic, checksum (recomputed below), version
    uint32_t total32 = static_cast<uint32_t>(total);
    memcpy(result.bytecode.data() + 24, &total32, 4);
    size_t at = headerSize;
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        uint32_t offset = static_cast<uint32_t>(at), size = static_cast<uint32_t>(bodies[i].size());
        memcpy(result.bytecode.data() + 32 + i * 4, &offset, 4);
        memcpy(result.bytecode.data() + at, &chunks[i].fourcc, 4);
        memcpy(result.bytecode.data() + at + 4, &size, 4);
        memcpy(result.bytecode.data() + at + 8, bodies[i].data(), size);
        at += 8 + size;
    }
    UpdateChecksum(result.bytecode);
    return result;
}
