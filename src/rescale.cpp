#include "rescale.h"

#include "d3d11_state.h"
#include "shader_info.h"
#include "shaders_generated.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <tuple>
#include <vector>

namespace
{
    // Formats the stretch shader can read and write as float4 (no integer formats).
    bool IsFilterable(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R16G16_FLOAT:
        case DXGI_FORMAT_R16G16_UNORM:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R8G8_UNORM:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R8_UNORM:
            return true;
        default:
            return false;
        }
    }

    // Copy of a stretched texture's valid region, one per format and size.
    struct Scratch
    {
        DXGI_FORMAT                      format;
        UINT                             width, height;
        ComPtr<ID3D11Texture2D>          texture;
        ComPtr<ID3D11ShaderResourceView> srv;
    };

    // 1-texel-wide column / row of a texture, for copying within it (a copy can't overlap its source).
    struct EdgeScratch
    {
        DXGI_FORMAT             format;
        UINT                    width, height;
        ComPtr<ID3D11Texture2D> column, row;
    };

    struct Helpers
    {
        ComPtr<ID3D11Device>          device;
        ComPtr<ID3D11VertexShader>    fullscreenVS;
        ComPtr<ID3D11PixelShader>     stretchPS;
        ComPtr<ID3D11RasterizerState> noCull;
        ComPtr<ID3D11Buffer>          params;
        std::vector<Scratch>          scratch;
        std::vector<EdgeScratch>      edges;
        bool                          failed = false;
    };

    struct Frame
    {
        UINT renderWidth = 0, renderHeight = 0, displayWidth = 0, displayHeight = 0;
        std::vector<ID3D11Resource*> settled;  // display-resolution content this frame (identity only, no refs)
    };

    std::mutex g_mutex;  // guards everything below
    Helpers    g;
    Frame      g_frame;

    thread_local bool t_stretching = false;  // our own draw goes through the draw hook too

    bool CreateHelpers(ID3D11DeviceContext* ctx)
    {
        if (g.device || g.failed)
            return !g.failed;
        ctx->GetDevice(&g.device);
        HRESULT hr = g.device->CreateVertexShader(g_vs_fullscreen, sizeof(g_vs_fullscreen), nullptr, &g.fullscreenVS);
        if (SUCCEEDED(hr))
            hr = g.device->CreatePixelShader(g_ps_stretch, sizeof(g_ps_stretch), nullptr, &g.stretchPS);
        if (SUCCEEDED(hr))
        {
            D3D11_RASTERIZER_DESC rd = {};
            rd.FillMode = D3D11_FILL_SOLID;
            rd.CullMode = D3D11_CULL_NONE;
            rd.DepthClipEnable = TRUE;
            hr = g.device->CreateRasterizerState(&rd, &g.noCull);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = 16;
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            hr = g.device->CreateBuffer(&bd, nullptr, &g.params);
        }
        if (FAILED(hr))
        {
            Log("rescale: helper creation failed {}", HrText(hr));
            g.failed = true;
        }
        return !g.failed;
    }

    const Scratch* FindScratch(const D3D11_TEXTURE2D_DESC& desc, DXGI_FORMAT viewFormat)
    {
        for (const Scratch& s : g.scratch)
            if (s.format == desc.Format && s.width == desc.Width && s.height == desc.Height)
                return &s;

        Scratch s = { desc.Format, desc.Width, desc.Height };
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = desc.Width;
        td.Height = desc.Height;
        td.MipLevels = td.ArraySize = 1;
        td.Format = desc.Format;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = viewFormat;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        HRESULT hr = g.device->CreateTexture2D(&td, nullptr, &s.texture);
        if (SUCCEEDED(hr))
            hr = g.device->CreateShaderResourceView(s.texture.Get(), &sd, &s.srv);
        if (FAILED(hr))
        {
            LOG_LIMITED(4, "rescale: scratch {}x{} fmt {} failed {}", desc.Width, desc.Height, int(desc.Format), HrText(hr));
            return nullptr;
        }
        return &g.scratch.emplace_back(std::move(s));
    }

    const EdgeScratch* FindEdgeScratch(ID3D11Device* device, const D3D11_TEXTURE2D_DESC& desc)
    {
        for (const EdgeScratch& s : g.edges)
            if (s.format == desc.Format && s.width == desc.Width && s.height == desc.Height)
                return &s;
        EdgeScratch s = { desc.Format, desc.Width, desc.Height };
        D3D11_TEXTURE2D_DESC td = {};
        td.MipLevels = td.ArraySize = 1;
        td.Format = desc.Format;
        td.SampleDesc.Count = 1;
        td.Width = 1;
        td.Height = desc.Height;
        bool ok = SUCCEEDED(device->CreateTexture2D(&td, nullptr, &s.column));
        td.Width = desc.Width;
        td.Height = 1;
        ok = ok && SUCCEEDED(device->CreateTexture2D(&td, nullptr, &s.row));
        if (!ok)
            return nullptr;
        return &g.edges.emplace_back(std::move(s));
    }

    bool IsSettled(ID3D11Resource* resource)
    {
        return std::ranges::contains(g_frame.settled, resource);
    }

    // Texture size as a fraction of the display: 1, 2 or 4 on both axes (rounding either way), else 0.
    UINT DisplayDivisor(const D3D11_TEXTURE2D_DESC& desc)
    {
        for (UINT d : { 1u, 2u, 4u })
        {
            auto matches = [d](UINT size, UINT display) { return size == display / d || size == (display + d - 1) / d; };
            if (matches(desc.Width, g_frame.displayWidth) && matches(desc.Height, g_frame.displayHeight))
                return d;
        }
        return 0;
    }

    // "t3:ro_postfx_dof_dofdatamap" -> "ro_postfx_dof_dofdatamap", for the log.
    std::string_view BindingName(const ShaderInfo& info, UINT slot)
    {
        std::string_view bindings = info.bindings;
        std::string tag = std::format("t{}:", slot);
        size_t at = bindings.find(tag);
        if (at == std::string_view::npos || (at > 0 && bindings[at - 1] != ' '))
            return "?";
        std::string_view name = bindings.substr(at + tag.size());
        return name.substr(0, name.find(' '));
    }

    // Records the stretch of `texture`'s render-scaled region over its whole extent. Called with g_mutex held.
    void Stretch(ID3D11DeviceContext* ctx, ID3D11Texture2D* texture, const D3D11_TEXTURE2D_DESC& desc,
                 DXGI_FORMAT viewFormat, UINT validWidth, UINT validHeight)
    {
        const Scratch* scratch = FindScratch(desc, viewFormat);
        ComPtr<ID3D11RenderTargetView> rtv;
        D3D11_RENDER_TARGET_VIEW_DESC rd = {};
        rd.Format = viewFormat;
        rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (!scratch || FAILED(g.device->CreateRenderTargetView(texture, &rd, &rtv)))
            return;

        D3D11_BOX region = { 0, 0, 0, validWidth, validHeight, 1 };
        ctx->CopySubresourceRegion(scratch->texture.Get(), 0, 0, 0, 0, texture, 0, &region);

        DrawStateGuard guard(ctx);
        struct { float validW, validH, scaleX, scaleY; } params = {
            float(validWidth), float(validHeight), float(validWidth) / desc.Width, float(validHeight) / desc.Height };
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (FAILED(ctx->Map(g.params.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            return;
        memcpy(mapped.pData, &params, sizeof(params));
        ctx->Unmap(g.params.Get(), 0);

        D3D11_VIEWPORT viewport = { 0.0f, 0.0f, float(desc.Width), float(desc.Height), 0.0f, 1.0f };
        ctx->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
        ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        ctx->OMSetDepthStencilState(nullptr, 0);
        ctx->RSSetState(g.noCull.Get());
        ctx->RSSetViewports(1, &viewport);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(g.fullscreenVS.Get(), nullptr, 0);
        ctx->PSSetShader(g.stretchPS.Get(), nullptr, 0);
        ctx->GSSetShader(nullptr, nullptr, 0);
        ctx->HSSetShader(nullptr, nullptr, 0);
        ctx->DSSetShader(nullptr, nullptr, 0);
        ctx->PSSetShaderResources(0, 1, scratch->srv.GetAddressOf());
        ctx->PSSetConstantBuffers(0, 1, g.params.GetAddressOf());
        t_stretching = true;
        ctx->Draw(3, 0);
        t_stretching = false;
    }

    // Stretches `srv`'s texture if it is a render-scaled input that this frame's post chain hasn't settled yet.
    void StretchIfRenderScaled(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* srv, const ShaderInfo& reader, UINT slot)
    {
        ComPtr<ID3D11Resource> resource;
        srv->GetResource(&resource);
        if (IsSettled(resource.Get()))
            return;
        g_frame.settled.push_back(resource.Get());  // decided once per frame, whatever the outcome

        D3D11_SHADER_RESOURCE_VIEW_DESC view;
        srv->GetDesc(&view);
        ComPtr<ID3D11Texture2D> texture;
        if (view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || FAILED(resource.As(&texture)))
            return;
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        UINT divisor = DisplayDivisor(desc);
        if (!divisor || !(desc.BindFlags & D3D11_BIND_RENDER_TARGET) || desc.MipLevels != 1 || desc.ArraySize != 1 ||
            desc.SampleDesc.Count != 1 || !IsFilterable(view.Format))
            return;

        // Same rounding as the engine's render-scaled viewports (2296 -> 1148 -> 574 at a 3440 display).
        UINT validWidth = std::max(1u, (desc.Width * g_frame.renderWidth + g_frame.displayWidth / 2) / g_frame.displayWidth);
        UINT validHeight = std::max(1u, (desc.Height * g_frame.renderHeight + g_frame.displayHeight / 2) / g_frame.displayHeight);
        if (validWidth >= desc.Width && validHeight >= desc.Height)
            return;
        // Logged once per reader, slot and region (the same stretch repeats every frame).
        static std::vector<std::tuple<uint64_t, UINT, UINT, UINT>> logged;
        auto key = std::tuple(reader.hash, slot, validWidth, validHeight);
        if (!std::ranges::contains(logged, key) && logged.size() < 64)
        {
            logged.push_back(key);
            Log("rescale: stretching t{} {} ({}x{} fmt {}, render region {}x{}) for PS {:016x}", slot,
                BindingName(reader, slot), desc.Width, desc.Height, int(view.Format), validWidth, validHeight, reader.hash);
        }
        Stretch(ctx, texture.Get(), desc, view.Format, validWidth, validHeight);
    }

    template <typename View, size_t N>
    void SettleViews(const ComArray<View, N>& views)
    {
        for (UINT i = 0; i < views.size(); ++i)
        {
            View* view = views[i];
            if (!view)
                continue;
            ComPtr<ID3D11Resource> resource;
            view->GetResource(&resource);
            if (!IsSettled(resource.Get()))
                g_frame.settled.push_back(resource.Get());
        }
    }
}

void rescale::BeginFrame(UINT renderWidth, UINT renderHeight, UINT displayWidth, UINT displayHeight,
                         std::initializer_list<ID3D11Resource*> displaySized)
{
    std::lock_guard lock(g_mutex);
    g_frame.renderWidth = renderWidth;
    g_frame.renderHeight = renderHeight;
    g_frame.displayWidth = displayWidth;
    g_frame.displayHeight = displayHeight;
    g_frame.settled.assign(displaySized.begin(), displaySized.end());
}

void rescale::BeforeDraw(ID3D11DeviceContext* ctx)
{
    if (t_stretching)
        return;
    ComPtr<ID3D11PixelShader> shader;
    ctx->PSGetShader(&shader, nullptr, nullptr);
    ShaderInfo info;
    if (!ReadShaderInfo(shader.Get(), info))
        return;

    std::lock_guard lock(g_mutex);
    if (!g_frame.displayWidth || !CreateHelpers(ctx))
        return;
    if (info.textureSlots)
    {
        ComArray<ID3D11ShaderResourceView, 16> srvs;
        ctx->PSGetShaderResources(0, srvs.size(), srvs.data());
        for (UINT slot = 0; slot < srvs.size(); ++slot)
            if (srvs[slot] && (info.textureSlots & (1u << slot)))
                StretchIfRenderScaled(ctx, srvs[slot], info, slot);
    }

    // What this draw renders is display-resolution content from here on.
    ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> rtvs;
    ctx->OMGetRenderTargets(rtvs.size(), rtvs.data(), nullptr);
    SettleViews(rtvs);
}

void rescale::BeforeDispatch(ID3D11DeviceContext* ctx)
{
    ComArray<ID3D11UnorderedAccessView, 8> uavs;
    ctx->CSGetUnorderedAccessViews(0, uavs.size(), uavs.data());
    std::lock_guard lock(g_mutex);
    SettleViews(uavs);
}

void rescale::RepairEdge(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* color, UINT width, UINT height)
{
    ComPtr<ID3D11Resource> resource;
    color->GetResource(&resource);
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(resource.As(&texture)))
        return;
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    if (width < 2 || height < 2 || width > desc.Width || height > desc.Height ||
        (width == desc.Width && height == desc.Height))
        return;

    ComPtr<ID3D11Device> device;
    ctx->GetDevice(&device);
    std::lock_guard lock(g_mutex);
    const EdgeScratch* scratch = FindEdgeScratch(device.Get(), desc);
    if (!scratch)
        return;
    if (width < desc.Width)
    {
        D3D11_BOX inner = { width - 2, 0, 0, width - 1, height, 1 };
        ctx->CopySubresourceRegion(scratch->column.Get(), 0, 0, 0, 0, texture.Get(), 0, &inner);
        D3D11_BOX back = { 0, 0, 0, 1, height, 1 };
        ctx->CopySubresourceRegion(texture.Get(), 0, width - 1, 0, 0, scratch->column.Get(), 0, &back);
    }
    if (height < desc.Height)
    {
        D3D11_BOX inner = { 0, height - 2, 0, width, height - 1, 1 };
        ctx->CopySubresourceRegion(scratch->row.Get(), 0, 0, 0, 0, texture.Get(), 0, &inner);
        D3D11_BOX back = { 0, 0, 0, width, 1, 1 };
        ctx->CopySubresourceRegion(texture.Get(), 0, 0, height - 1, 0, scratch->row.Get(), 0, &back);
    }
    LOG_LIMITED(1, "rescale: repairing the last column and row of the {}x{} scene colour ({}x{} texture)", width,
                height, desc.Width, desc.Height);
}
