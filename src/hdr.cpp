#include "hdr.h"

#include "config.h"
#include "d3d11_state.h"
#include "hooks.h"
#include "shader_info.h"
#include "shaders_generated.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <vector>

namespace
{
    constexpr uint64_t kTonemapPS   = 0x6b486ef38e1bbe42;  // PostFX tonemap: scene -> _sRGB_offscreenBackBuffer
    constexpr uint64_t kSceneCopyPS = 0x5b793501b92112cc;  // RENDER_UPSCALE: offscreen buffer -> back buffer
    constexpr float    kKnee        = 0.6f;                // tonemapped luminance where the HDR extension starts

    constexpr DXGI_FORMAT kHdrFormat   = DXGI_FORMAT_R10G10B10A2_UNORM;
    constexpr DXGI_FORMAT kSceneFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

    // DXGI vtable slots.
    constexpr int kFactoryCreateSwapChain = 10;
    constexpr int kSwapChainGetBuffer     = 9;
    constexpr int kSwapChainGetDesc       = 12;
    constexpr int kSwapChain1GetDesc1     = 18;

    using PFN_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
    using PFN_GetBuffer       = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, REFIID, void**);
    using PFN_GetDesc         = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, DXGI_SWAP_CHAIN_DESC*);
    using PFN_GetDesc1        = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, DXGI_SWAP_CHAIN_DESC1*);
    PFN_CreateSwapChain o_CreateSwapChain;
    PFN_GetBuffer       o_GetBuffer;
    PFN_GetDesc         o_GetDesc;
    PFN_GetDesc1        o_GetDesc1;

    struct Target
    {
        ComPtr<ID3D11Texture2D>          texture;
        ComPtr<ID3D11RenderTargetView>   rtv;
        ComPtr<ID3D11ShaderResourceView> srv;
        UINT                             width = 0, height = 0;
    };

    struct State
    {
        std::mutex                     mutex;  // creation and replacement of the objects below
        ComPtr<ID3D11Device>           device;
        IDXGISwapChain*                swapChain = nullptr;  // the game's (identity only)
        DXGI_FORMAT                    gameFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        Target                         proxy;       // the game's back buffer: R8G8B8A8_TYPELESS, UNORM views
        Target                         scene;       // HDR tonemap output (offscreen buffer size)
        Target                         display;     // HDR scene after the game's copy filter (back buffer size)
        ComPtr<ID3D11RenderTargetView> backbufferRTV;
        ID3D11Resource*                backbuffer = nullptr;
        ComPtr<ID3D11VertexShader>     fullscreenVS;
        ComPtr<ID3D11PixelShader>      tonemapPS;
        ComPtr<ID3D11PixelShader>      compositePS;
        ComPtr<ID3D11RasterizerState>  noCull;
        ComPtr<ID3D11Buffer>           tonemapParams;    // DEFAULT usage: UpdateSubresource works in any context
        ComPtr<ID3D11Buffer>           compositeParams;
        std::vector<std::pair<ID3D11BlendState*, ComPtr<ID3D11BlendState>>> uiBlends;  // game state -> coverage alpha
    };
    State g;

    // shaders/hdr_tonemap_ps.hlsl, HdrParams (b3).
    struct TonemapParams
    {
        float peak, knee, boost, pad;
        float whiteBalance[3], exposure;
        float contrast, saturation, vibrance, lutStrength;
    };

    // White balance multipliers for a temperature (blue -1 .. +1 warm) and tint (green -1 .. +1 magenta),
    // normalized to keep luminance.
    std::array<float, 3> WhiteBalance(float temperature, float tint)
    {
        std::array<float, 3> m = { 1.0f + 0.12f * temperature, 1.0f - 0.08f * tint, 1.0f - 0.12f * temperature };
        m[0] *= 1.0f + 0.04f * tint;
        m[2] *= 1.0f + 0.04f * tint;
        float luma = 0.2126f * m[0] + 0.7152f * m[1] + 0.0722f * m[2];
        for (float& v : m)
            v /= luma;
        return m;
    }

    std::atomic<bool> g_active{ false };
    std::atomic<bool> g_sceneRecorded{ false };  // our tonemap ran since the last Present
    int               g_framesWithoutScene = 1000;

    // Tonemaps and scene copies recorded so far. They are recorded on different threads, and a frame's copy can come
    // after the next frame's tonemap, so a copy is redirected while there are more tonemaps than redirected copies.
    std::atomic<uint64_t> g_tonemaps{ 0 };
    std::atomic<uint64_t> g_redirectedCopies{ 0 };

    std::mutex  g_statusMutex;
    hdr::Status g_status;

    void SetStatus(bool windowsHdr, bool active, float peak, std::string note)
    {
        std::lock_guard lock(g_statusMutex);
        g_status = { windowsHdr, active, peak, std::move(note) };
        Log("hdr: {} ({})", active ? "on" : "off", g_status.note);
    }

    // ---- Display queries ----------------------------------------------------------------------------------------

    // Windows "Use HDR" for the monitor showing hwnd.
    bool IsWindowsHdrOn(HWND hwnd)
    {
        MONITORINFOEXW monitor = {};
        monitor.cbSize = sizeof(monitor);
        if (!GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor))
            return false;
        UINT32 pathCount = 0, modeCount = 0;
        if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &pathCount, &modeCount) != ERROR_SUCCESS)
            return false;
        std::vector<DISPLAYCONFIG_PATH_INFO> paths(pathCount);
        std::vector<DISPLAYCONFIG_MODE_INFO> modes(modeCount);
        if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &pathCount, paths.data(), &modeCount, modes.data(), nullptr) != ERROR_SUCCESS)
            return false;
        for (const DISPLAYCONFIG_PATH_INFO& path : paths)
        {
            DISPLAYCONFIG_SOURCE_DEVICE_NAME source = {};
            source.header = { DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId, path.sourceInfo.id };
            if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS || wcscmp(source.viewGdiDeviceName, monitor.szDevice) != 0)
                continue;
            DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color = {};
            color.header = { DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color), path.targetInfo.adapterId, path.targetInfo.id };
            return DisplayConfigGetDeviceInfo(&color.header) == ERROR_SUCCESS && color.advancedColorEnabled;
        }
        return false;
    }

    // Peak luminance the display reports through DXGI (EDID), 0 if unknown.
    float DisplayPeakNits(IDXGISwapChain* swapChain)
    {
        ComPtr<IDXGIOutput> output;
        ComPtr<IDXGIOutput6> output6;
        DXGI_OUTPUT_DESC1 desc = {};
        if (FAILED(swapChain->GetContainingOutput(&output)) || FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)))
            return 0.0f;
        return desc.MaxLuminance;
    }

    float PeakNits()
    {
        float configured = config::Current()->hdrPeakNits;
        if (configured > 0.0f)
            return configured;
        std::lock_guard lock(g_statusMutex);
        return g_status.displayPeakNits > 0.0f ? g_status.displayPeakNits : 1000.0f;
    }

    // ---- Resources ----------------------------------------------------------------------------------------------

    bool CreateTarget(Target& t, UINT width, UINT height, DXGI_FORMAT format, DXGI_FORMAT viewFormat)
    {
        t = {};
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        D3D11_RENDER_TARGET_VIEW_DESC rtv = {};
        rtv.Format = viewFormat;
        rtv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = viewFormat;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;
        HRESULT hr = g.device->CreateTexture2D(&desc, nullptr, &t.texture);
        if (SUCCEEDED(hr))
            hr = g.device->CreateRenderTargetView(t.texture.Get(), &rtv, &t.rtv);
        if (SUCCEEDED(hr))
            hr = g.device->CreateShaderResourceView(t.texture.Get(), &srv, &t.srv);
        if (FAILED(hr))
        {
            Log("hdr: {}x{} target (format {}) failed {}", width, height, static_cast<int>(format), HrText(hr));
            t = {};
            return false;
        }
        t.width = width;
        t.height = height;
        return true;
    }

    bool CreateHelpers()
    {
        auto buffer = [](UINT size, ComPtr<ID3D11Buffer>& out) {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = size;
            bd.Usage = D3D11_USAGE_DEFAULT;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            return g.device->CreateBuffer(&bd, nullptr, &out);
        };
        HRESULT hr = g.device->CreateVertexShader(g_vs_fullscreen, sizeof(g_vs_fullscreen), nullptr, &g.fullscreenVS);
        if (SUCCEEDED(hr))
            hr = g.device->CreatePixelShader(g_ps_hdr_composite, sizeof(g_ps_hdr_composite), nullptr, &g.compositePS);
        if (SUCCEEDED(hr))
        {
            D3D11_RASTERIZER_DESC rd = {};
            rd.FillMode = D3D11_FILL_SOLID;
            rd.CullMode = D3D11_CULL_NONE;
            rd.DepthClipEnable = TRUE;
            hr = g.device->CreateRasterizerState(&rd, &g.noCull);
        }
        if (SUCCEEDED(hr))
            hr = buffer(32, g.compositeParams);
        if (FAILED(hr))
            Log("hdr: helper creation failed {}", HrText(hr));
        return SUCCEEDED(hr);
    }

    // The tonemap replacement's shader and parameters, on first use (HDR output or SDR colour grading).
    bool EnsureTonemap(ID3D11DeviceContext* ctx)
    {
        std::lock_guard lock(g.mutex);
        if (g.tonemapPS && g.tonemapParams)
            return true;
        ComPtr<ID3D11Device> device;
        ctx->GetDevice(&device);
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = sizeof(TonemapParams);
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        HRESULT hr = device->CreatePixelShader(g_ps_hdr_tonemap, sizeof(g_ps_hdr_tonemap), nullptr, &g.tonemapPS);
        if (SUCCEEDED(hr))
            hr = device->CreateBuffer(&bd, nullptr, &g.tonemapParams);
        if (FAILED(hr))
        {
            LOG_LIMITED(3, "hdr: tonemap replacement unavailable {}", HrText(hr));
            g.tonemapPS.Reset();
            g.tonemapParams.Reset();
            return false;
        }
        return true;
    }

    // Stand-in back buffer and the display-size HDR target, at the swap chain's current size.
    bool CreateSizedTargets(IDXGISwapChain* swapChain)
    {
        DXGI_SWAP_CHAIN_DESC desc = {};
        if (FAILED(o_GetDesc(swapChain, &desc)))
            return false;
        UINT w = desc.BufferDesc.Width, h = desc.BufferDesc.Height;
        return CreateTarget(g.proxy, w, h, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM) &&
               CreateTarget(g.display, w, h, kSceneFormat, kSceneFormat);
    }

    // ---- DXGI hooks ---------------------------------------------------------------------------------------------

    // Switches a freshly created HDR10 swap chain to PQ / Rec.2020; false if DXGI (DXVK) can't present it.
    bool EnableHdr10(IDXGISwapChain* swapChain)
    {
        ComPtr<IDXGISwapChain3> sc3;
        UINT support = 0;
        constexpr DXGI_COLOR_SPACE_TYPE kSpace = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        return SUCCEEDED(swapChain->QueryInterface(IID_PPV_ARGS(&sc3))) &&
               SUCCEEDED(sc3->CheckColorSpaceSupport(kSpace, &support)) &&
               (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) && SUCCEEDED(sc3->SetColorSpace1(kSpace));
    }

    HRESULT STDMETHODCALLTYPE Hook_CreateSwapChain(IDXGIFactory* self, IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc,
                                                  IDXGISwapChain** out)
    {
        if (!desc || !out || g.swapChain || !config::Current()->hdrEnabled)
            return o_CreateSwapChain(self, device, desc, out);
        if (!IsWindowsHdrOn(desc->OutputWindow))
        {
            SetStatus(false, false, 0.0f, "Windows HDR is off for this display");
            return o_CreateSwapChain(self, device, desc, out);
        }

        DXGI_SWAP_CHAIN_DESC hdrDesc = *desc;
        hdrDesc.BufferDesc.Format = kHdrFormat;
        HRESULT hr = o_CreateSwapChain(self, device, &hdrDesc, out);
        if (SUCCEEDED(hr) && !EnableHdr10(*out))
        {
            (*out)->Release();
            *out = nullptr;
            hr = E_FAIL;
        }
        if (FAILED(hr))
        {
            SetStatus(true, false, 0.0f, "the swap chain doesn't support HDR10 (is DXVK's dxgi.dll loaded?)");
            return o_CreateSwapChain(self, device, desc, out);
        }

        std::lock_guard lock(g.mutex);
        (*out)->GetDevice(IID_PPV_ARGS(&g.device));
        g.swapChain = *out;
        g.gameFormat = desc->BufferDesc.Format;
        if (!g.device || !CreateHelpers() || !CreateSizedTargets(*out))
        {
            SetStatus(true, false, 0.0f, "resource creation failed (see log)");
            g.swapChain = nullptr;  // the 10-bit swap chain stays, the game renders as before
            return hr;
        }
        float peak = DisplayPeakNits(*out);
        g_active = true;
        SetStatus(true, true, peak, std::format("HDR10 PQ, display peak {:.0f} nits", peak));
        return hr;
    }

    bool IsOurs(IDXGISwapChain* swapChain)
    {
        return g_active && swapChain == g.swapChain;
    }

    HRESULT STDMETHODCALLTYPE Hook_GetBuffer(IDXGISwapChain* self, UINT buffer, REFIID riid, void** out)
    {
        if (!IsOurs(self) || buffer != 0)
            return o_GetBuffer(self, buffer, riid, out);
        std::lock_guard lock(g.mutex);
        return g.proxy.texture ? g.proxy.texture->QueryInterface(riid, out) : E_FAIL;
    }

    HRESULT STDMETHODCALLTYPE Hook_GetDesc(IDXGISwapChain* self, DXGI_SWAP_CHAIN_DESC* desc)
    {
        HRESULT hr = o_GetDesc(self, desc);
        if (SUCCEEDED(hr) && desc && IsOurs(self))
            desc->BufferDesc.Format = g.gameFormat;  // the game sees the format it asked for
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hook_GetDesc1(IDXGISwapChain1* self, DXGI_SWAP_CHAIN_DESC1* desc)
    {
        HRESULT hr = o_GetDesc1(self, desc);
        if (SUCCEEDED(hr) && desc && IsOurs(self))
            desc->Format = g.gameFormat;
        return hr;
    }

    // ---- Draw-time state ----------------------------------------------------------------------------------------

    enum class Pending { None, Tonemap, SceneCopy, UiBlend };

    struct Saved
    {
        Pending                                                                  pending = Pending::None;
        ComPtr<ID3D11PixelShader>                                                ps;
        ComPtr<ID3D11Buffer>                                                     cb3;
        ComPtr<ID3D11ShaderResourceView>                                         srv0;
        ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> rtvs;
        ComPtr<ID3D11DepthStencilView>                                           dsv;
        ComPtr<ID3D11BlendState>                                                 blend;
        float                                                                    blendFactor[4] = {};
        UINT                                                                     sampleMask = 0;
        bool                                                                     targetsChanged = false;
    };
    thread_local Saved                t_saved;
    thread_local ID3D11DeviceContext* t_uiContext = nullptr;  // context whose render target is the UI layer
    thread_local bool                 t_internal = false;     // our own OMSetRenderTargets calls

    void SetTargets(ID3D11DeviceContext* ctx, UINT count, ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv)
    {
        t_internal = true;
        ctx->OMSetRenderTargets(count, rtvs, dsv);
        t_internal = false;
    }

    void ClearSaved()
    {
        t_saved.ps.Reset();
        t_saved.cb3.Reset();
        t_saved.srv0.Reset();
        t_saved.rtvs.Reset();
        t_saved.dsv.Reset();
        t_saved.blend.Reset();
        t_saved.targetsChanged = false;
    }

    void SaveRenderTargets(ID3D11DeviceContext* ctx)
    {
        ClearSaved();
        ctx->OMGetRenderTargets(t_saved.rtvs.size(), t_saved.rtvs.data(), &t_saved.dsv);
    }

    ID3D11Resource* ResourceOf(ID3D11View* view)
    {
        ComPtr<ID3D11Resource> resource;
        view->GetResource(&resource);
        return resource.Get();  // identity only; the view keeps it alive
    }

    // Tonemap: our transcription, with the colour grading, and in HDR mode a second, HDR render target.
    void BeginTonemap(ID3D11DeviceContext* ctx)
    {
        if (!EnsureTonemap(ctx))
            return;
        SaveRenderTargets(ctx);
        ID3D11RenderTargetView* sdr = t_saved.rtvs[0];
        if (!sdr)
            return;

        ComPtr<ID3D11RenderTargetView> hdrRTV;
        if (g_active)
        {
            ComPtr<ID3D11Texture2D> texture;
            D3D11_TEXTURE2D_DESC desc;
            ComPtr<ID3D11Resource> resource;
            sdr->GetResource(&resource);
            if (FAILED(resource.As(&texture)))
                return;
            texture->GetDesc(&desc);
            std::lock_guard lock(g.mutex);
            if (g.scene.width != desc.Width || g.scene.height != desc.Height)
                CreateTarget(g.scene, desc.Width, desc.Height, kSceneFormat, kSceneFormat);
            hdrRTV = g.scene.rtv;
            if (!hdrRTV)
                return;
        }

        std::shared_ptr<const Config> cfg = config::Current();
        bool grade = cfg->gradeEnabled;
        std::array<float, 3> wb = grade ? WhiteBalance(cfg->gradeTemperature, cfg->gradeTint) : std::array{ 1.0f, 1.0f, 1.0f };
        TonemapParams params = {
            PeakNits() / cfg->hdrPaperWhite, kKnee, cfg->hdrHighlightBoost, 0.0f,
            { wb[0], wb[1], wb[2] }, grade ? std::exp2(cfg->gradeExposure) : 1.0f,
            grade ? cfg->gradeContrast : 1.0f, grade ? cfg->gradeSaturation : 1.0f, grade ? cfg->gradeVibrance : 0.0f,
            grade ? cfg->gradeGameLook : 1.0f,
        };
        ctx->UpdateSubresource(g.tonemapParams.Get(), 0, nullptr, &params, 0, 0);
        ctx->PSGetShader(&t_saved.ps, nullptr, nullptr);
        ctx->PSGetConstantBuffers(3, 1, &t_saved.cb3);
        if (hdrRTV)
        {
            ID3D11RenderTargetView* targets[2] = { sdr, hdrRTV.Get() };
            SetTargets(ctx, 2, targets, t_saved.dsv.Get());
            t_saved.targetsChanged = true;
            g_sceneRecorded = true;
            ++g_tonemaps;
        }
        ctx->PSSetShader(g.tonemapPS.Get(), nullptr, 0);
        ctx->PSSetConstantBuffers(3, 1, g.tonemapParams.GetAddressOf());
        t_saved.pending = Pending::Tonemap;
        LOG_LIMITED(1, "hdr: tonemap replaced ({})", g_active ? "HDR output" : "SDR colour grading");
    }

    // Scene copy into the back buffer: the game's filter runs on the HDR scene into our display-size target, and the
    // stand-in back buffer becomes an empty UI layer (cleared in AfterDraw).
    void BeginSceneCopy(ID3D11DeviceContext* ctx)
    {
        // Other tonemap permutations (the brightness calibration screen, ...) aren't replaced: their frame stays SDR.
        uint64_t tonemaps = g_tonemaps, copies = g_redirectedCopies;
        if (copies >= tonemaps)
            return;
        g_redirectedCopies = std::max(copies + 1, tonemaps - 1);  // never more than one frame behind
        SaveRenderTargets(ctx);
        ComPtr<ID3D11RenderTargetView> finalRTV;
        ComPtr<ID3D11ShaderResourceView> sceneSRV;
        {
            std::lock_guard lock(g.mutex);
            if (!t_saved.rtvs[0] || ResourceOf(t_saved.rtvs[0]) != g.proxy.texture.Get())
                return;
            finalRTV = g.display.rtv;
            sceneSRV = g.scene.srv;
        }
        if (!finalRTV || !sceneSRV)
            return;
        ctx->PSGetShaderResources(0, 1, &t_saved.srv0);
        SetTargets(ctx, 1, finalRTV.GetAddressOf(), nullptr);
        ctx->PSSetShaderResources(0, 1, sceneSRV.GetAddressOf());
        t_saved.pending = Pending::SceneCopy;
        LOG_LIMITED(1, "hdr: scene copy redirected, UI layer cleared");
    }

    // UI draw into the UI layer: alpha accumulates coverage (one, 1 - src alpha) so the layer can be composited.
    void BeginUiBlend(ID3D11DeviceContext* ctx)
    {
        ctx->OMGetBlendState(&t_saved.blend, t_saved.blendFactor, &t_saved.sampleMask);
        if (!t_saved.blend)
            return;  // blending off: the shader's alpha is written as is
        ComPtr<ID3D11BlendState> coverage;
        {
            std::lock_guard lock(g.mutex);
            auto it = std::ranges::find(g.uiBlends, t_saved.blend.Get(), &decltype(g.uiBlends)::value_type::first);
            if (it != g.uiBlends.end())
            {
                coverage = it->second;
            }
            else
            {
                D3D11_BLEND_DESC desc;
                t_saved.blend->GetDesc(&desc);
                for (D3D11_RENDER_TARGET_BLEND_DESC& rt : desc.RenderTarget)
                {
                    if (rt.BlendEnable)
                    {
                        rt.SrcBlendAlpha = D3D11_BLEND_ONE;
                        rt.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
                        rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
                    }
                    rt.RenderTargetWriteMask |= D3D11_COLOR_WRITE_ENABLE_ALPHA;
                }
                g.device->CreateBlendState(&desc, &coverage);
                g.uiBlends.emplace_back(t_saved.blend.Get(), coverage);  // the game keeps its states alive
            }
        }
        if (!coverage)
            return;
        ctx->OMSetBlendState(coverage.Get(), t_saved.blendFactor, t_saved.sampleMask);
        t_saved.pending = Pending::UiBlend;
        LOG_LIMITED(1, "hdr: UI draws accumulate coverage");
    }
}

hdr::Status hdr::GetStatus()
{
    std::lock_guard lock(g_statusMutex);
    return g_status;
}

bool hdr::IsActive()
{
    return g_active;
}

void hdr::InstallDxgiHooks(IDXGIFactory* factory, IDXGISwapChain* swapChain)
{
    if (!config::Current()->hdrEnabled)
    {
        SetStatus(false, false, 0.0f, "disabled in the settings");
        return;
    }
    void** factoryVtable = *reinterpret_cast<void***>(factory);
    void** swapChainVtable = *reinterpret_cast<void***>(swapChain);
    bool ok = hooks::InstallInline(swapChainVtable[kSwapChainGetBuffer], &Hook_GetBuffer, &o_GetBuffer, "hdr: GetBuffer") &&
              hooks::InstallInline(swapChainVtable[kSwapChainGetDesc], &Hook_GetDesc, &o_GetDesc, "hdr: GetDesc");
    ComPtr<IDXGISwapChain1> swapChain1;
    if (ok && SUCCEEDED(swapChain->QueryInterface(IID_PPV_ARGS(&swapChain1))))
        ok = hooks::InstallInline((*reinterpret_cast<void***>(swapChain1.Get()))[kSwapChain1GetDesc1], &Hook_GetDesc1,
                                  &o_GetDesc1, "hdr: GetDesc1");
    // Last: once CreateSwapChain is hooked, the swap chain hooks must already be in place.
    if (ok)
        hooks::InstallInline(factoryVtable[kFactoryCreateSwapChain], &Hook_CreateSwapChain, &o_CreateSwapChain,
                             "hdr: CreateSwapChain");
}

void hdr::Composite(IDXGISwapChain* swapChain)
{
    if (!IsOurs(swapChain))
        return;
    bool hadScene = g_framesWithoutScene < 3;
    g_framesWithoutScene = g_sceneRecorded.exchange(false) ? 0 : std::min(g_framesWithoutScene + 1, 1000);
    if (hadScene != (g_framesWithoutScene < 3))
        LOG_LIMITED(20, "hdr: {}", hadScene ? "no 3D scene, showing the game's frame as SDR" : "3D scene in HDR");

    std::lock_guard lock(g.mutex);
    ComPtr<ID3D11Resource> backbuffer;
    if (FAILED(o_GetBuffer(swapChain, 0, IID_PPV_ARGS(&backbuffer))))
        return;
    if (backbuffer.Get() != g.backbuffer || !g.backbufferRTV)
    {
        g.backbufferRTV.Reset();
        D3D11_RENDER_TARGET_VIEW_DESC rd = {};
        rd.Format = kHdrFormat;
        rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (FAILED(g.device->CreateRenderTargetView(backbuffer.Get(), &rd, &g.backbufferRTV)))
            return;
        g.backbuffer = backbuffer.Get();
    }

    ComPtr<ID3D11DeviceContext> ctx;
    g.device->GetImmediateContext(&ctx);
    std::shared_ptr<const Config> cfg = config::Current();
    struct { float paperWhite, ui, peak; uint32_t hasScene, heatmap; float pad[3]; } params = {
        cfg->hdrPaperWhite, cfg->hdrPaperWhite, PeakNits(), g_framesWithoutScene < 3 ? 1u : 0u, cfg->hdrHeatmap ? 1u : 0u, {} };
    ctx->UpdateSubresource(g.compositeParams.Get(), 0, nullptr, &params, 0, 0);

    DrawStateGuard guard(ctx.Get());
    D3D11_VIEWPORT viewport = { 0.0f, 0.0f, float(g.proxy.width), float(g.proxy.height), 0.0f, 1.0f };
    ID3D11ShaderResourceView* srvs[2] = { g.display.srv.Get(), g.proxy.srv.Get() };
    ctx->OMSetRenderTargets(1, g.backbufferRTV.GetAddressOf(), nullptr);
    ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(nullptr, 0);
    ctx->RSSetState(g.noCull.Get());
    ctx->RSSetViewports(1, &viewport);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(g.fullscreenVS.Get(), nullptr, 0);
    ctx->PSSetShader(g.compositePS.Get(), nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
    ctx->PSSetShaderResources(0, 2, srvs);
    ctx->PSSetConstantBuffers(0, 1, g.compositeParams.GetAddressOf());
    ctx->Draw(3, 0);
}

DXGI_FORMAT hdr::ResizeFormat(IDXGISwapChain* swapChain, DXGI_FORMAT requested)
{
    if (!IsOurs(swapChain))
        return requested;
    std::lock_guard lock(g.mutex);
    g.backbufferRTV.Reset();  // DXGI can't resize while a view of the real buffer exists
    g.backbuffer = nullptr;
    if (requested != DXGI_FORMAT_UNKNOWN)
        g.gameFormat = requested;
    return requested == DXGI_FORMAT_UNKNOWN ? requested : kHdrFormat;
}

void hdr::AfterResize(IDXGISwapChain* swapChain, HRESULT result)
{
    if (!IsOurs(swapChain) || FAILED(result))
        return;
    std::lock_guard lock(g.mutex);
    if (!CreateSizedTargets(swapChain))
        Log("hdr: stand-in back buffer recreation failed");
}

void hdr::OnSetRenderTargets(ID3D11DeviceContext* ctx, UINT count, ID3D11RenderTargetView* const* rtvs)
{
    if (!g_active || t_internal)
        return;  // our own target changes don't count
    bool ui = false;
    if (count && rtvs && rtvs[0])
    {
        D3D11_RENDER_TARGET_VIEW_DESC desc;
        rtvs[0]->GetDesc(&desc);
        ui = desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM && ResourceOf(rtvs[0]) == g.proxy.texture.Get();
    }
    if (ui)
        t_uiContext = ctx;
    else if (t_uiContext == ctx)
        t_uiContext = nullptr;
}

bool hdr::OnDraw(ID3D11DeviceContext* ctx, bool fullscreenCandidate)
{
    if (!g_active && !config::Current()->gradeEnabled)
        return false;
    if (g_active && t_uiContext == ctx)
    {
        BeginUiBlend(ctx);
        if (t_saved.pending == Pending::None)
            ClearSaved();
        return false;
    }
    if (!fullscreenCandidate)
        return false;
    ComPtr<ID3D11PixelShader> shader;
    ctx->PSGetShader(&shader, nullptr, nullptr);
    ShaderInfo info;
    if (!ReadShaderInfo(shader.Get(), info))
        return false;
    if (info.hash == kTonemapPS)
        BeginTonemap(ctx);
    else if (g_active && info.hash == kSceneCopyPS)
        BeginSceneCopy(ctx);
    if (t_saved.pending == Pending::None)
        ClearSaved();  // not applicable after all: hold no references
    return false;
}

void hdr::OnFinishCommandList(ID3D11DeviceContext* ctx)
{
    if (t_uiContext == ctx)
        t_uiContext = nullptr;  // the context's state is reset; its next job starts without the UI target
}

bool hdr::NeedsAfterDraw()
{
    return t_saved.pending != Pending::None;
}

void hdr::AfterDraw(ID3D11DeviceContext* ctx)
{
    Pending pending = std::exchange(t_saved.pending, Pending::None);
    switch (pending)
    {
    case Pending::Tonemap:
        ctx->PSSetShader(t_saved.ps.Get(), nullptr, 0);
        ctx->PSSetConstantBuffers(3, 1, t_saved.cb3.GetAddressOf());
        if (t_saved.targetsChanged)
            SetTargets(ctx, t_saved.rtvs.size(), t_saved.rtvs.data(), t_saved.dsv.Get());
        break;
    case Pending::SceneCopy:
    {
        constexpr float kTransparent[4] = {};
        SetTargets(ctx, t_saved.rtvs.size(), t_saved.rtvs.data(), t_saved.dsv.Get());
        ctx->PSSetShaderResources(0, 1, t_saved.srv0.GetAddressOf());
        ComPtr<ID3D11RenderTargetView> layer;
        {
            std::lock_guard lock(g.mutex);
            layer = g.proxy.rtv;
        }
        if (layer)
            ctx->ClearRenderTargetView(layer.Get(), kTransparent);
        break;
    }
    case Pending::UiBlend:
        ctx->OMSetBlendState(t_saved.blend.Get(), t_saved.blendFactor, t_saved.sampleMask);
        break;
    case Pending::None:
        break;
    }
    ClearSaved();
}
