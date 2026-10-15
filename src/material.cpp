#include "material.h"

#include "config.h"
#include "shader_info.h"
#include "shader_patch.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

namespace
{
    constexpr int kAlbedo   = shader_patch::kAlbedoSlot;
    constexpr int kSpecular = shader_patch::kSpecularSlot;
    constexpr int kSurface  = shader_patch::kSurfaceSlot;

    struct Target
    {
        ComPtr<ID3D11Texture2D>          texture;
        ComPtr<ID3D11RenderTargetView>   rtv;
        ComPtr<ID3D11ShaderResourceView> srv;
    };

    struct State
    {
        std::mutex mutex;  // creation, blend state cache
        Target     albedo, specular, surface;
        UINT       width = 0, height = 0;
        std::vector<std::pair<ID3D11BlendState*, ComPtr<ID3D11BlendState>>> decalBlends;  // game state -> derived
    };
    State g;

    std::atomic<uint64_t> g_frame{ 0 };
    std::atomic<uint64_t> g_lastWrite{ 0 };  // frame of the last forward draw that got our targets

    enum class Capture { None, Opaque, Blended };

    thread_local ID3D11DeviceContext*     t_patchedShaderCtx = nullptr;  // context whose current PS is patched
    thread_local bool                     t_shaderHasSurface = false;    // ... and knows the roughness
    thread_local ID3D11DeviceContext*     t_boundCtx = nullptr;          // context with our targets bound
    thread_local bool                     t_internal = false;
    thread_local ComPtr<ID3D11BlendState> t_savedBlend;                  // game blend state to restore after a draw
    thread_local float                    t_savedFactor[4] = {};
    thread_local UINT                     t_savedMask = 0;
    thread_local bool                     t_restoreBlend = false;

    bool CreateTarget(ID3D11Device* device, Target& t, UINT w, UINT h, DXGI_FORMAT format)
    {
        t = {};
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        return SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &t.texture)) &&
               SUCCEEDED(device->CreateRenderTargetView(t.texture.Get(), nullptr, &t.rtv)) &&
               SUCCEEDED(device->CreateShaderResourceView(t.texture.Get(), nullptr, &t.srv));
    }

    struct Rtvs
    {
        ComPtr<ID3D11RenderTargetView> albedo, specular, surface;
    };

    // Our targets at the size of the game's colour target; false if they can't be made.
    bool EnsureTargets(ID3D11RenderTargetView* colorTarget, Rtvs& out)
    {
        ComPtr<ID3D11Resource> resource;
        colorTarget->GetResource(&resource);
        ComPtr<ID3D11Texture2D> texture;
        if (FAILED(resource.As(&texture)))
            return false;
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);

        std::lock_guard lock(g.mutex);
        if (g.width != desc.Width || g.height != desc.Height || !g.albedo.rtv || !g.specular.rtv || !g.surface.rtv)
        {
            ComPtr<ID3D11Device> device;
            texture->GetDevice(&device);
            constexpr DXGI_FORMAT kColor = DXGI_FORMAT_R16G16B16A16_FLOAT;
            if (!CreateTarget(device.Get(), g.albedo, desc.Width, desc.Height, kColor) ||
                !CreateTarget(device.Get(), g.specular, desc.Width, desc.Height, kColor) ||
                !CreateTarget(device.Get(), g.surface, desc.Width, desc.Height, DXGI_FORMAT_R16G16_FLOAT))
            {
                LOG_LIMITED(3, "material: {}x{} targets failed", desc.Width, desc.Height);
                g.albedo = g.specular = g.surface = {};
                g.width = g.height = 0;
                return false;
            }
            g.width = desc.Width;
            g.height = desc.Height;
            Log("material: targets {}x{}", desc.Width, desc.Height);
        }
        out = { g.albedo.rtv, g.specular.rtv, g.surface.rtv };
        return true;
    }

    // What a draw contributes to the material data. Blending applies to every render target, ours included: an
    // alpha-blended draw (decals, glass) lerps its material data in by its opacity (the patched shaders put it in
    // .w of our outputs), through a blend state of our own for our targets. Other blend modes don't contribute.
    Capture CaptureFor(ID3D11DeviceContext* ctx, ComPtr<ID3D11BlendState>& blend)
    {
        if (t_patchedShaderCtx != ctx)
            return Capture::None;
        float factor[4];
        UINT mask;
        ctx->OMGetBlendState(&blend, factor, &mask);
        if (!blend)
            return Capture::Opaque;
        D3D11_BLEND_DESC desc;
        blend->GetDesc(&desc);
        const D3D11_RENDER_TARGET_BLEND_DESC& rt = desc.RenderTarget[0];
        if (!rt.BlendEnable)
            return Capture::Opaque;
        bool alphaBlend = (rt.SrcBlend == D3D11_BLEND_SRC_ALPHA || rt.SrcBlend == D3D11_BLEND_ONE) &&
                          rt.DestBlend == D3D11_BLEND_INV_SRC_ALPHA && rt.BlendOp == D3D11_BLEND_OP_ADD;
        return alphaBlend && t_shaderHasSurface && config::Current()->decalCapture ? Capture::Blended : Capture::None;
    }

    // The game's blend for its own targets, opacity blending for ours.
    ComPtr<ID3D11BlendState> DecalBlend(ID3D11DeviceContext* ctx, ID3D11BlendState* game)
    {
        std::lock_guard lock(g.mutex);
        for (const auto& [from, to] : g.decalBlends)
            if (from == game)
                return to;
        D3D11_BLEND_DESC desc;
        game->GetDesc(&desc);
        if (!desc.IndependentBlendEnable)
            for (int i = 1; i < 8; ++i)
                desc.RenderTarget[i] = desc.RenderTarget[0];
        desc.IndependentBlendEnable = TRUE;
        for (int slot : { kAlbedo, kSpecular, kSurface })
        {
            D3D11_RENDER_TARGET_BLEND_DESC& rt = desc.RenderTarget[slot];
            rt.BlendEnable = TRUE;
            rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
            rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
            rt.DestBlendAlpha = D3D11_BLEND_ONE;
            rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
            rt.RenderTargetWriteMask = slot == kSurface ? (D3D11_COLOR_WRITE_ENABLE_RED | D3D11_COLOR_WRITE_ENABLE_GREEN)
                                                        : (D3D11_COLOR_WRITE_ENABLE_ALL & ~D3D11_COLOR_WRITE_ENABLE_ALPHA);
        }
        ComPtr<ID3D11Device> device;
        ctx->GetDevice(&device);
        ComPtr<ID3D11BlendState> derived;
        if (FAILED(device->CreateBlendState(&desc, &derived)))
            return nullptr;
        g.decalBlends.emplace_back(game, derived);  // the game keeps its states alive
        return derived;
    }
}

void material::OnSetPixelShader(ID3D11DeviceContext* ctx, ID3D11PixelShader* shader)
{
    ShaderInfo info;
    if (shader && ReadShaderInfo(shader, info) && info.materialOutputs)
    {
        t_patchedShaderCtx = ctx;
        t_shaderHasSurface = info.materialSurface;
    }
    else if (t_patchedShaderCtx == ctx)
    {
        t_patchedShaderCtx = nullptr;
    }
}

void material::OnSetRenderTargets(ID3D11DeviceContext* ctx)
{
    if (!t_internal && t_boundCtx == ctx)
        t_boundCtx = nullptr;  // the game replaced the targets
}

void material::OnFinishCommandList(ID3D11DeviceContext* ctx)
{
    if (t_boundCtx == ctx)
        t_boundCtx = nullptr;
    if (t_patchedShaderCtx == ctx)
        t_patchedShaderCtx = nullptr;
}

// Our targets are bound for draws that contribute (see CaptureFor) and taken off again for anything else.
void material::OnDraw(ID3D11DeviceContext* ctx)
{
    ComPtr<ID3D11BlendState> blend;
    Capture capture = CaptureFor(ctx, blend);
    bool wanted = capture != Capture::None;
    if (wanted != (t_boundCtx == ctx))
    {
        ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> rtvs;
        ComPtr<ID3D11DepthStencilView> dsv;
        ComArray<ID3D11UnorderedAccessView, 8> uavs;
        ctx->OMGetRenderTargetsAndUnorderedAccessViews(rtvs.size(), rtvs.data(), &dsv, 0, uavs.size(), uavs.data());
        Rtvs ours;
        if (wanted)
        {
            bool usable = rtvs[0] && !rtvs[kAlbedo] && !rtvs[kSpecular] && !rtvs[kSurface] &&
                          std::ranges::none_of(uavs.items, std::identity{});
            if (!usable || !EnsureTargets(rtvs[0], ours))
                return;
        }
        ID3D11RenderTargetView* targets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
        UINT count = 0;
        for (UINT i = 0; i < rtvs.size(); ++i)
        {
            targets[i] = i == kAlbedo ? ours.albedo.Get() : i == kSpecular ? ours.specular.Get()
                       : i == kSurface ? ours.surface.Get() : rtvs[i];  // ours are null when unbinding
            if (targets[i])
                count = i + 1;
        }
        t_internal = true;
        ctx->OMSetRenderTargets(count, targets, dsv.Get());
        t_internal = false;
        t_boundCtx = wanted ? ctx : nullptr;
    }
    if (capture == Capture::Blended)
    {
        if (ComPtr<ID3D11BlendState> derived = DecalBlend(ctx, blend.Get()))
        {
            ctx->OMGetBlendState(&t_savedBlend, t_savedFactor, &t_savedMask);
            ctx->OMSetBlendState(derived.Get(), t_savedFactor, t_savedMask);
            t_restoreBlend = true;
            LOG_LIMITED(1, "material: capturing blended draws (decals) by opacity");
        }
    }
    if (wanted)
    {
        g_lastWrite = g_frame.load();
        LOG_LIMITED(1, "material: capturing material data from the forward pass");
    }
}

bool material::NeedsAfterDraw()
{
    return t_restoreBlend;
}

void material::AfterDraw(ID3D11DeviceContext* ctx)
{
    ctx->OMSetBlendState(t_savedBlend.Get(), t_savedFactor, t_savedMask);
    t_savedBlend.Reset();
    t_restoreBlend = false;
}

material::Views material::ShaderResources()
{
    std::lock_guard lock(g.mutex);
    if (!g.albedo.srv || g_frame - g_lastWrite > 2)
        return {};
    return { g.albedo.srv.Get(), g.specular.srv.Get(), g.surface.srv.Get() };
}

void material::Clear(ID3D11DeviceContext* ctx)
{
    Rtvs rtvs;
    {
        std::lock_guard lock(g.mutex);
        rtvs = { g.albedo.rtv, g.specular.rtv, g.surface.rtv };
    }
    constexpr float kZero[4] = {};
    constexpr float kNoSurface[4] = { -1.0f, 0.0f, 0.0f, 0.0f };
    if (rtvs.albedo)
        ctx->ClearRenderTargetView(rtvs.albedo.Get(), kZero);
    if (rtvs.specular)
        ctx->ClearRenderTargetView(rtvs.specular.Get(), kZero);
    if (rtvs.surface)
        ctx->ClearRenderTargetView(rtvs.surface.Get(), kNoSurface);
}

void material::OnFrame()
{
    ++g_frame;
}
