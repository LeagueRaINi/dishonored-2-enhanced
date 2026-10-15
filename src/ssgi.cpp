#include "ssgi.h"

#include "config.h"
#include "d3d11_state.h"
#include "shaders_generated.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace
{
    // shaders/ssgi_common.hlsli SsgiParams (b0).
    struct Params
    {
        uint32_t renderWidth, renderHeight, halfWidth, halfHeight;
        float    projScaleX, projScaleY, cameraNear, radius;
        float    thickness;
        uint32_t frameIndex, slices, steps;
        float    stepWidth, historyValid;
        uint32_t exposureIndex, compressedHdr;
        float    ssrMaxRoughness, ssrThickness;
        uint32_t ssrSteps;
        float    ssrMaxDistance, maxIblLod, viewWhite, pad[2];
        Float4x4 inverseViewProjection;
        Float4x4 previousViewProjection;
    };
    static_assert(sizeof(Params) == 224);

    constexpr float kMaxIblLod = 6.0f;  // the game's IBL cubemaps: 256x256, 7 mips

    struct Texture
    {
        ComPtr<ID3D11Texture2D>           texture;
        ComPtr<ID3D11ShaderResourceView>  srv;
        ComPtr<ID3D11UnorderedAccessView> uav;
    };

    // Radiance with a mip chain: the full chain for the trace, single levels for building it.
    constexpr UINT kRadianceMips = 5;
    struct MipTexture
    {
        ComPtr<ID3D11Texture2D>           texture;
        ComPtr<ID3D11ShaderResourceView>  srv;  // all levels
        ComPtr<ID3D11ShaderResourceView>  levelSrv[kRadianceMips];
        ComPtr<ID3D11UnorderedAccessView> levelUav[kRadianceMips];
        uint32_t                          width[kRadianceMips] = {}, height[kRadianceMips] = {};
    };

    // GI buffers (made when GI is first used); [parity] of a pair is written this frame.
    struct GiBuffers
    {
        Texture trace, history[2], meta[2], denoised[2];
    };

    // SSR buffers (made when SSR is first used).
    struct SsrBuffers
    {
        Texture trace, hitDistance, denoised, history[2], depth[2];
    };

    struct State
    {
        ComPtr<ID3D11Device>        device;
        ComPtr<ID3D11ComputeShader> prepareCS, downsampleCS, traceCS, temporalCS, denoiseCS;
        ComPtr<ID3D11ComputeShader> ssrTraceCS, ssrTemporalCS, ssrDenoiseCS;
        ComPtr<ID3D11Buffer>        params;
        ComPtr<ID3D11SamplerState>  linearClamp;
        bool                        failed = false;

        uint32_t   width = 0, height = 0;  // render region the textures were made for
        Texture    geometry[2];            // this frame's and the previous frame's (history validation)
        MipTexture radiance;
        GiBuffers  gi;
        SsrBuffers ssr;
        bool       giReady = false, ssrReady = false;
        int        parity = 0;
        bool       giHistoryValid = false, ssrHistoryValid = false;

        uint32_t lastFrame = 0;            // continuity: a gap or a projection change starts the histories over
        float    lastProjX = 0.0f, lastProjY = 0.0f;
    };
    State g;

    bool CreateTexture(Texture& t, uint32_t w, uint32_t h, DXGI_FORMAT format)
    {
        t = {};
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        return SUCCEEDED(g.device->CreateTexture2D(&desc, nullptr, &t.texture)) &&
               SUCCEEDED(g.device->CreateShaderResourceView(t.texture.Get(), nullptr, &t.srv)) &&
               SUCCEEDED(g.device->CreateUnorderedAccessView(t.texture.Get(), nullptr, &t.uav));
    }

    bool CreateMipTexture(MipTexture& t, uint32_t w, uint32_t h, DXGI_FORMAT format)
    {
        t = {};
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = kRadianceMips;
        desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(g.device->CreateTexture2D(&desc, nullptr, &t.texture)) ||
            FAILED(g.device->CreateShaderResourceView(t.texture.Get(), nullptr, &t.srv)))
            return false;
        for (UINT mip = 0; mip < kRadianceMips; ++mip)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Format = format;
            sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D = { mip, 1 };
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format = format;
            ud.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            ud.Texture2D.MipSlice = mip;
            if (FAILED(g.device->CreateShaderResourceView(t.texture.Get(), &sd, &t.levelSrv[mip])) ||
                FAILED(g.device->CreateUnorderedAccessView(t.texture.Get(), &ud, &t.levelUav[mip])))
                return false;
            t.width[mip] = std::max(1u, w >> mip);
            t.height[mip] = std::max(1u, h >> mip);
        }
        return true;
    }

    bool CreateHelpers(ID3D11DeviceContext* ctx)
    {
        if (g.device || g.failed)
            return !g.failed;
        ctx->GetDevice(&g.device);
        struct Shader
        {
            const uint8_t*               code;
            size_t                       size;
            ComPtr<ID3D11ComputeShader>* out;
        };
        const Shader shaders[] = {
            { g_cs_ssgi_prepare, sizeof(g_cs_ssgi_prepare), &g.prepareCS },
            { g_cs_ssgi_downsample, sizeof(g_cs_ssgi_downsample), &g.downsampleCS },
            { g_cs_ssgi_trace, sizeof(g_cs_ssgi_trace), &g.traceCS },
            { g_cs_ssgi_temporal, sizeof(g_cs_ssgi_temporal), &g.temporalCS },
            { g_cs_ssgi_denoise, sizeof(g_cs_ssgi_denoise), &g.denoiseCS },
            { g_cs_ssr_trace, sizeof(g_cs_ssr_trace), &g.ssrTraceCS },
            { g_cs_ssr_temporal, sizeof(g_cs_ssr_temporal), &g.ssrTemporalCS },
            { g_cs_ssr_denoise, sizeof(g_cs_ssr_denoise), &g.ssrDenoiseCS },
        };
        HRESULT hr = S_OK;
        for (const Shader& s : shaders)
            if (SUCCEEDED(hr))
                hr = g.device->CreateComputeShader(s.code, s.size, nullptr, s.out->ReleaseAndGetAddressOf());
        if (SUCCEEDED(hr))
        {
            D3D11_SAMPLER_DESC sd = {};
            sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.MaxLOD = D3D11_FLOAT32_MAX;
            hr = g.device->CreateSamplerState(&sd, &g.linearClamp);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = sizeof(Params);
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            hr = g.device->CreateBuffer(&bd, nullptr, &g.params);
        }
        if (FAILED(hr))
        {
            Log("ssgi: shader creation failed {}", HrText(hr));
            g.failed = true;
        }
        return !g.failed;
    }

    constexpr DXGI_FORMAT kRgba16 = DXGI_FORMAT_R16G16B16A16_FLOAT;

    // Shared buffers at this render size; the effect buffers are dropped and remade on demand after a resize.
    bool EnsureShared(uint32_t w, uint32_t h)
    {
        if (w == g.width && h == g.height && g.geometry[0].texture)
            return true;
        uint32_t hw = (w + 1) / 2, hh = (h + 1) / 2;
        g.gi = {};
        g.ssr = {};
        g.giReady = g.ssrReady = false;
        if (!CreateTexture(g.geometry[0], hw, hh, kRgba16) || !CreateTexture(g.geometry[1], hw, hh, kRgba16) ||
            !CreateMipTexture(g.radiance, hw, hh, kRgba16))
        {
            Log("ssgi: {}x{} textures failed", hw, hh);
            g.width = g.height = 0;
            return false;
        }
        g.width = w;
        g.height = h;
        g.giHistoryValid = g.ssrHistoryValid = false;
        Log("ssgi: half-resolution buffers {}x{}", hw, hh);
        return true;
    }

    bool EnsureGi(uint32_t hw, uint32_t hh)
    {
        if (g.giReady)
            return true;
        GiBuffers& b = g.gi;
        bool ok = CreateTexture(b.trace, hw, hh, kRgba16);
        for (int i = 0; i < 2 && ok; ++i)
            ok = CreateTexture(b.history[i], hw, hh, kRgba16) && CreateTexture(b.meta[i], hw, hh, DXGI_FORMAT_R32G32_FLOAT) &&
                 CreateTexture(b.denoised[i], hw, hh, kRgba16);
        g.giReady = ok;
        g.giHistoryValid = false;
        return ok;
    }

    bool EnsureSsr(uint32_t hw, uint32_t hh)
    {
        if (g.ssrReady)
            return true;
        SsrBuffers& b = g.ssr;
        bool ok = CreateTexture(b.trace, hw, hh, kRgba16) && CreateTexture(b.denoised, hw, hh, kRgba16) &&
                  CreateTexture(b.hitDistance, hw, hh, DXGI_FORMAT_R16_FLOAT);
        for (int i = 0; i < 2 && ok; ++i)
            ok = CreateTexture(b.history[i], hw, hh, kRgba16) && CreateTexture(b.depth[i], hw, hh, DXGI_FORMAT_R32G32_FLOAT);
        g.ssrReady = ok;
        g.ssrHistoryValid = false;
        return ok;
    }

    // Runs a shader over w x h threads (8x8 groups); `params` is written to the shared constant buffer.
    template <typename P>
    bool Dispatch(ID3D11DeviceContext* ctx, ID3D11ComputeShader* shader, const P& params, uint32_t w, uint32_t h,
                  std::initializer_list<ID3D11ShaderResourceView*> srvs, std::initializer_list<ID3D11UnorderedAccessView*> uavs)
    {
        static_assert(sizeof(P) <= sizeof(Params));
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (FAILED(ctx->Map(g.params.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            return false;
        memcpy(mapped.pData, &params, sizeof(params));
        ctx->Unmap(g.params.Get(), 0);
        // Unbind first: a texture written by the previous pass is read by this one.
        ID3D11ShaderResourceView* noSrvs[8] = {};
        ID3D11UnorderedAccessView* noUavs[8] = {};
        ctx->CSSetShaderResources(0, 8, noSrvs);
        ctx->CSSetUnorderedAccessViews(0, 8, noUavs, nullptr);
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetShaderResources(0, static_cast<UINT>(srvs.size()), srvs.begin());
        ctx->CSSetUnorderedAccessViews(0, static_cast<UINT>(uavs.size()), uavs.begin(), nullptr);
        ctx->CSSetConstantBuffers(0, 1, g.params.GetAddressOf());
        ctx->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        return true;
    }

    // A gap in the frame sequence (loading, menus, Record not called) or a projection change (FOV, resolution) means
    // the histories describe another view.
    bool ContinuesPreviousFrame(const ssgi::Inputs& in)
    {
        bool continuous = in.frame == g.lastFrame + 1 && std::abs(in.projX - g.lastProjX) <= 0.001f * std::abs(in.projX) &&
                          std::abs(in.projY - g.lastProjY) <= 0.001f * std::abs(in.projY);
        g.lastFrame = in.frame;
        g.lastProjX = in.projX;
        g.lastProjY = in.projY;
        return continuous;
    }
}

ssgi::Result ssgi::Record(ID3D11DeviceContext* ctx, const Inputs& in)
{
    std::shared_ptr<const Config> cfg = config::Current();
    bool continuous = ContinuesPreviousFrame(in);
    bool gi = cfg->ssgiEnabled;
    bool ssr = cfg->ssrEnabled && in.surface;
    if ((!gi && !ssr) || !in.color || !in.depth || !in.motion || !in.exposure || !in.width || !in.height ||
        !CreateHelpers(ctx) || !EnsureShared(in.width, in.height))
    {
        g.giHistoryValid = g.ssrHistoryValid = false;
        return {};
    }
    const uint32_t hw = (in.width + 1) / 2, hh = (in.height + 1) / 2;
    gi = gi && EnsureGi(hw, hh);
    ssr = ssr && EnsureSsr(hw, hh);
    if (!continuous)
        g.giHistoryValid = g.ssrHistoryValid = false;

    Params p = {};
    p.renderWidth = in.width;
    p.renderHeight = in.height;
    p.halfWidth = hw;
    p.halfHeight = hh;
    p.projScaleX = 1.0f / in.projX;
    p.projScaleY = 1.0f / in.projY;
    p.cameraNear = in.cameraNear;
    p.radius = cfg->ssgiRadius;
    p.thickness = cfg->ssgiThickness;
    p.frameIndex = in.frame;
    p.slices = static_cast<uint32_t>(std::clamp(cfg->ssgiSlices, 1, 8));
    p.steps = static_cast<uint32_t>(std::clamp(cfg->ssgiSteps, 2, 32));
    p.exposureIndex = in.exposureIndex;
    p.compressedHdr = in.compressedHdr;
    p.ssrMaxRoughness = cfg->ssrMaxRoughness;
    p.ssrThickness = cfg->ssrThickness;
    p.ssrSteps = static_cast<uint32_t>(std::clamp(cfg->ssrSteps, 8, 128));
    p.ssrMaxDistance = cfg->ssrMaxDistance;
    p.maxIblLod = kMaxIblLod;
    p.viewWhite = in.viewWhite;
    p.inverseViewProjection = in.inverseViewProjection;
    p.previousViewProjection = in.previousViewProjection;

    ComputeStateGuard guard(ctx);
    ComPtr<ID3D11SamplerState> savedSampler;
    ctx->CSGetSamplers(0, 1, &savedSampler);
    ctx->CSSetSamplers(0, 1, g.linearClamp.GetAddressOf());

    const int cur = g.parity, prev = 1 - g.parity;
    const Texture& geometry = g.geometry[cur];
    const Texture& previousGeometry = g.geometry[prev];
    MipTexture& r = g.radiance;
    bool ok = Dispatch(ctx, g.prepareCS.Get(), p, hw, hh, { in.depth, in.color, in.exposure },
                       { geometry.uav.Get(), r.levelUav[0].Get() });
    for (UINT mip = 1; mip < kRadianceMips && ok; ++mip)
    {
        struct { uint32_t sourceW, sourceH, targetW, targetH; } level = {
            r.width[mip - 1], r.height[mip - 1], r.width[mip], r.height[mip] };
        ok = Dispatch(ctx, g.downsampleCS.Get(), level, r.width[mip], r.height[mip], { r.levelSrv[mip - 1].Get() },
                      { r.levelUav[mip].Get() });
    }
    if (gi && ok)
    {
        GiBuffers& b = g.gi;
        Params giParams = p;
        giParams.historyValid = g.giHistoryValid ? 1.0f : 0.0f;
        ok = Dispatch(ctx, g.traceCS.Get(), giParams, hw, hh, { geometry.srv.Get(), r.srv.Get() }, { b.trace.uav.Get() }) &&
             Dispatch(ctx, g.temporalCS.Get(), giParams, hw, hh,
                      { b.trace.srv.Get(), geometry.srv.Get(), b.history[prev].srv.Get(), b.meta[prev].srv.Get(), in.motion,
                        previousGeometry.srv.Get() },
                      { b.history[cur].uav.Get(), b.meta[cur].uav.Get() });
        giParams.stepWidth = 1.0f;
        ok = ok && Dispatch(ctx, g.denoiseCS.Get(), giParams, hw, hh, { b.history[cur].srv.Get(), geometry.srv.Get() },
                            { b.denoised[0].uav.Get() });
        giParams.stepWidth = 2.0f;
        ok = ok && Dispatch(ctx, g.denoiseCS.Get(), giParams, hw, hh, { b.denoised[0].srv.Get(), geometry.srv.Get() },
                            { b.denoised[1].uav.Get() });
    }
    if (ssr && ok)
    {
        SsrBuffers& b = g.ssr;
        Params ssrParams = p;
        ssrParams.historyValid = g.ssrHistoryValid ? 1.0f : 0.0f;
        ok = Dispatch(ctx, g.ssrTraceCS.Get(), ssrParams, hw, hh, { geometry.srv.Get(), r.srv.Get(), in.surface },
                      { b.trace.uav.Get(), b.hitDistance.uav.Get() }) &&
             Dispatch(ctx, g.ssrTemporalCS.Get(), ssrParams, hw, hh,
                      { b.trace.srv.Get(), geometry.srv.Get(), b.history[prev].srv.Get(), b.depth[prev].srv.Get(), in.motion,
                        in.surface, b.hitDistance.srv.Get(), previousGeometry.srv.Get() },
                      { b.history[cur].uav.Get(), b.depth[cur].uav.Get() }) &&
             Dispatch(ctx, g.ssrDenoiseCS.Get(), ssrParams, hw, hh,
                      { b.history[cur].srv.Get(), geometry.srv.Get(), b.depth[cur].srv.Get(), in.surface },
                      { b.denoised.uav.Get() });
    }
    ctx->CSSetSamplers(0, 1, savedSampler.GetAddressOf());

    g.parity = prev;
    g.giHistoryValid = gi && ok;
    g.ssrHistoryValid = ssr && ok;
    if (!ok)
    {
        LOG_LIMITED(3, "ssgi: a pass couldn't be recorded (constant buffer map failed)");
        return {};
    }
    LOG_LIMITED(1, "ssgi: active (GI {}, SSR {})", gi ? "on" : "off", ssr ? "on" : "off");
    Result result{
        .light = gi ? g.gi.denoised[1].srv.Get() : nullptr,
        .reflection = ssr ? g.ssr.denoised.srv.Get() : nullptr,
        .geometry = geometry.srv.Get(),
        .halfWidth = hw,
        .halfHeight = hh,
    };
    switch (cfg->ssgiDebug)  // intermediate buffers for the debug views (pack.hlsl)
    {
    case 7: result.debug = ssr ? g.ssr.hitDistance.srv.Get() : nullptr; break;
    case 8: result.debug = ssr ? g.ssr.trace.srv.Get() : nullptr; break;
    case 9: result.debug = gi ? g.gi.trace.srv.Get() : nullptr; break;
    case 10: result.debug = gi ? g.gi.meta[cur].srv.Get() : nullptr; break;
    default: break;
    }
    return result;
}
