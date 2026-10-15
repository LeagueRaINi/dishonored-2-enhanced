#include "capture.h"

#include "cbuffers.h"
#include "shader_info.h"

#include <atomic>
#include <iterator>
#include <mutex>
#include <unordered_map>

namespace
{
    std::atomic<bool> g_active{ false };
    std::atomic<int>  g_dispatches{ 0 };

    // Draws are logged per pass: consecutive draws on a context with the same pixel shader and targets.
    struct PassKey
    {
        void* pixelShader = nullptr;
        void* target0 = nullptr;
        void* depth = nullptr;

        bool operator==(const PassKey&) const = default;
    };
    struct PendingPass
    {
        PassKey     key;
        int         draws = 0;
        std::string description;
    };
    std::mutex                                            g_passMutex;
    std::unordered_map<ID3D11DeviceContext*, PendingPass> g_passes;
    int                                                   g_passCount = 0;

    void FlushPass(ID3D11DeviceContext* ctx, const PendingPass& pass)
    {
        if (pass.draws)
            LogFileOnly("  pass #{} ctx={} draws={} {}", g_passCount++, Ptr(ctx), pass.draws, pass.description);
    }

    const char* DimensionName(D3D11_SRV_DIMENSION dimension)
    {
        switch (dimension)
        {
        case D3D11_SRV_DIMENSION_BUFFER:         return "buf";
        case D3D11_SRV_DIMENSION_TEXTURE2D:      return "tex2d";
        case D3D11_SRV_DIMENSION_TEXTURE2DARRAY: return "tex2darr";
        case D3D11_SRV_DIMENSION_TEXTURE3D:      return "tex3d";
        case D3D11_SRV_DIMENSION_TEXTURECUBE:    return "cube";
        default:                                 return "other";
        }
    }

    std::string DescribeResource(ID3D11Resource* resource)
    {
        D3D11_RESOURCE_DIMENSION dimension;
        resource->GetType(&dimension);
        switch (dimension)
        {
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
        {
            D3D11_TEXTURE2D_DESC d;
            static_cast<ID3D11Texture2D*>(resource)->GetDesc(&d);
            return std::format("{}x{} fmt={} mips={} bind={:#x}", d.Width, d.Height, static_cast<int>(d.Format), d.MipLevels,
                               d.BindFlags);
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
        {
            D3D11_TEXTURE3D_DESC d;
            static_cast<ID3D11Texture3D*>(resource)->GetDesc(&d);
            return std::format("{}x{}x{} fmt={}", d.Width, d.Height, d.Depth, static_cast<int>(d.Format));
        }
        case D3D11_RESOURCE_DIMENSION_BUFFER:
        {
            D3D11_BUFFER_DESC d;
            static_cast<ID3D11Buffer*>(resource)->GetDesc(&d);
            return std::format("buffer {} bytes", d.ByteWidth);
        }
        default:
            return std::format("dimension {}", static_cast<int>(dimension));
        }
    }

    template <typename View>
    std::string DescribeView(View* view)
    {
        ComPtr<ID3D11Resource> resource;
        view->GetResource(&resource);
        return std::format("res={} {}", Ptr(resource.Get()), DescribeResource(resource.Get()));
    }

    template <typename Shader>
    std::string DescribeShader(Shader* shader, const char* label)
    {
        if (!shader)
            return std::format("{}=none", label);
        ShaderInfo info;
        ReadShaderInfo(shader, info);
        return std::format("{}={:016x} [{}]", label, info.hash, info.bindings);
    }

    std::string DescribePass(ID3D11DeviceContext* ctx, ID3D11VertexShader* vs, ID3D11PixelShader* ps,
                             ID3D11RenderTargetView* const* rtvs, UINT rtvCount, ID3D11DepthStencilView* dsv)
    {
        std::string out = std::format("\n     {}\n     {}", DescribeShader(vs, "VS"), DescribeShader(ps, "PS"));
        for (UINT i = 0; i < rtvCount; ++i)
        {
            if (!rtvs[i])
                continue;
            D3D11_RENDER_TARGET_VIEW_DESC desc;
            rtvs[i]->GetDesc(&desc);
            std::format_to(std::back_inserter(out), "\n     rt{} viewfmt={} {}", i, static_cast<int>(desc.Format),
                           DescribeView(rtvs[i]));
        }
        if (dsv)
        {
            D3D11_DEPTH_STENCIL_VIEW_DESC desc;
            dsv->GetDesc(&desc);
            std::format_to(std::back_inserter(out), "\n     depth viewfmt={} {}", static_cast<int>(desc.Format), DescribeView(dsv));
        }
        ComArray<ID3D11ShaderResourceView, 16> srvs;
        ctx->PSGetShaderResources(0, srvs.size(), srvs.data());
        for (UINT i = 0; i < srvs.size(); ++i)
        {
            if (!srvs[i])
                continue;
            D3D11_SHADER_RESOURCE_VIEW_DESC desc;
            srvs[i]->GetDesc(&desc);
            std::format_to(std::back_inserter(out), "\n     t{} {} viewfmt={} {}", i, DimensionName(desc.ViewDimension),
                           static_cast<int>(desc.Format), DescribeView(srvs[i]));
        }
        UINT viewports = 1;
        D3D11_VIEWPORT viewport = {};
        ctx->RSGetViewports(&viewports, &viewport);
        std::format_to(std::back_inserter(out), "\n     viewport {}x{}", viewport.Width, viewport.Height);
        return out;
    }

    void AppendFrameConstants(std::string& out, ID3D11DeviceContext* ctx)
    {
        ComArray<ID3D11Buffer, 3> buffers;
        ctx->CSGetConstantBuffers(0, buffers.size(), buffers.data());
        cb::Snapshot view, instance;
        bool haveView = cb::Read(buffers[1], view) && view.size == cb::kPerViewSize;
        bool haveInstance = cb::Read(buffers[2], instance) && instance.size == cb::kPerInstanceSize;
        std::format_to(std::back_inserter(out), "\n     PerViewCB shadow: {} (rev {})  PerInstanceCB shadow: {} (rev {})",
                       haveView ? "yes" : "NO", view.revision, haveInstance ? "yes" : "NO", instance.revision);

        auto floats = [](const cb::Snapshot& s, size_t offset, int count) {
            std::string text;
            for (int i = 0; i < count; ++i)
                text += std::format("{}{:.5f}", i ? " " : "", s.Get<float>(offset + i * 4));
            return text;
        };
        if (haveView)
        {
            using namespace cb::perview;
            std::format_to(std::back_inserter(out),
                           "\n     jittervectors=({}) subpixeloffset=({}) jitterrel=({}) resolutionscale=({})"
                           "\n     framenumber={:.0f} compressedhdr={} proj r0=({}) r1=({}) r2=({}) r3=({})",
                           floats(view, kJitterVectors, 4), floats(view, kSubpixelOffset, 4), floats(view, kJitterRelative, 2),
                           floats(view, kResolutionScale, 2), view.Get<float>(kFrameNumber), view.Get<uint32_t>(kCompressedHdr),
                           floats(view, kProjection, 4), floats(view, kProjection + 16, 4), floats(view, kProjection + 32, 4),
                           floats(view, kProjection + 48, 4));
        }
        if (haveInstance)
        {
            using namespace cb::perinstance;
            std::format_to(std::back_inserter(out),
                           "\n     taatexsize=({}) dither+viewport=({}) exposureindex=({} {}) prevresscale=({})"
                           "\n     envwhite={:.3f} viewwhite={:.3f} taaamount={:.3f} evbias={:.3f}",
                           floats(instance, kTaaTexSize, 4), floats(instance, kDitherViewport, 4),
                           instance.Get<uint32_t>(kExposureIndex), instance.Get<uint32_t>(kExposureIndex + 4),
                           floats(instance, kPrevResScale, 2), instance.Get<float>(kEnvWhite), instance.Get<float>(kViewWhite),
                           instance.Get<float>(kTaaAmount), instance.Get<float>(kEvBias));
        }
    }
}

namespace
{
    std::atomic<const void*> g_backbuffer{ nullptr };
}

void capture::SetBackbuffer(const void* resource)
{
    g_backbuffer = resource;
}

void capture::Begin(uint64_t frame)
{
    g_dispatches = 0;
    g_passCount = 0;
    g_active = true;
    Log("=== capture begin (frame {}) === backbuffer res={}", frame, g_backbuffer.load());
}

void capture::End(uint64_t frame)
{
    if (!g_active.exchange(false))
        return;
    {
        std::lock_guard lock(g_passMutex);
        for (const auto& [ctx, pass] : g_passes)
            FlushPass(ctx, pass);
        g_passes.clear();
    }
    Log("=== capture end (frame {}, {} dispatches, {} draw passes) ===", frame, g_dispatches.load(), g_passCount);
}

bool capture::Active()
{
    return g_active;
}

void capture::LogDispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z)
{
    ComPtr<ID3D11ComputeShader> shader;
    ctx->CSGetShader(&shader, nullptr, nullptr);
    ShaderInfo info;
    ReadShaderInfo(shader.Get(), info);

    std::string out = std::format("  #{} ctx={} type={} thread={} Dispatch({},{},{}) CS={:016x}{}\n     bindings: {}",
                                  g_dispatches++, Ptr(ctx), static_cast<int>(ctx->GetType()), GetCurrentThreadId(), x, y, z,
                                  info.hash, info.isTaa ? " [TAA]" : "", info.bindings);

    ComArray<ID3D11ShaderResourceView, 16> srvs;
    ctx->CSGetShaderResources(0, srvs.size(), srvs.data());
    for (UINT i = 0; i < srvs.size(); ++i)
    {
        if (!srvs[i])
            continue;
        D3D11_SHADER_RESOURCE_VIEW_DESC desc;
        srvs[i]->GetDesc(&desc);
        std::format_to(std::back_inserter(out), "\n     t{:<2} {:<8} viewfmt={:<3} {}", i, DimensionName(desc.ViewDimension),
                       static_cast<int>(desc.Format), DescribeView(srvs[i]));
    }

    ComArray<ID3D11UnorderedAccessView, 8> uavs;
    ctx->CSGetUnorderedAccessViews(0, uavs.size(), uavs.data());
    for (UINT i = 0; i < uavs.size(); ++i)
    {
        if (!uavs[i])
            continue;
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
        uavs[i]->GetDesc(&desc);
        std::format_to(std::back_inserter(out), "\n     u{:<2} viewfmt={:<3} {}", i, static_cast<int>(desc.Format),
                       DescribeView(uavs[i]));
    }

    ComArray<ID3D11Buffer, 8> buffers;
    ctx->CSGetConstantBuffers(0, buffers.size(), buffers.data());
    for (UINT i = 0; i < buffers.size(); ++i)
    {
        if (!buffers[i])
            continue;
        D3D11_BUFFER_DESC desc;
        buffers[i]->GetDesc(&desc);
        std::format_to(std::back_inserter(out), "\n     b{:<2} {} bytes usage={} cpu={:#x} res={}", i, desc.ByteWidth,
                       static_cast<int>(desc.Usage), desc.CPUAccessFlags, Ptr(buffers[i]));
    }

    if (info.isTaa)
        AppendFrameConstants(out, ctx);
    LogFileOnly("{}", out);
}

void capture::LogDraw(ID3D11DeviceContext* ctx)
{
    ComPtr<ID3D11PixelShader> ps;
    ctx->PSGetShader(&ps, nullptr, nullptr);
    ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> rtvs;
    ComPtr<ID3D11DepthStencilView> dsv;
    ctx->OMGetRenderTargets(rtvs.size(), rtvs.data(), &dsv);
    PassKey key{ ps.Get(), rtvs[0], dsv.Get() };

    std::lock_guard lock(g_passMutex);
    PendingPass& pass = g_passes[ctx];
    if (pass.draws && pass.key == key)
    {
        ++pass.draws;
        return;
    }
    FlushPass(ctx, pass);
    ComPtr<ID3D11VertexShader> vs;
    ctx->VSGetShader(&vs, nullptr, nullptr);
    pass = { key, 1, DescribePass(ctx, vs.Get(), ps.Get(), rtvs.data(), rtvs.size(), dsv.Get()) };
}
