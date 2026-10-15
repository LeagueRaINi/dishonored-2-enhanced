// FSR upscaler running on a D3D12 device next to the game's D3D11 device (native D3D11 or DXVK).
//
// The engine records its post-processing on deferred contexts. At the TAA dispatch (record time, worker thread)
// the upscaler is spliced into the command stream:
//
//     pack inputs -> FinishCommandList(TRUE) = part A | part B: write FSR output, expand motion vectors, ...
//
// When the engine executes part B on the immediate context, part A is executed first, then
//
//     D3D11 Signal(inputs) -> D3D12 Wait, ffxDispatch, Signal(done) -> D3D11 Wait(done)
//
// so FSR runs in GPU order exactly where the engine TAA would have. All waits are GPU-side.
//
// Fence values are allocated under g_mutex and every signal is ordered after the wait for the previous value in its
// chain, so the shared fence only ever increases.
#include "fsr.h"

#include "config.h"
#include "d3d11_state.h"
#include "material.h"
#include "shaders_generated.h"
#include "ssgi.h"

#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../third_party/FidelityFX-SDK/Kits/FidelityFX/api/include/ffx_api.h"
#include "../third_party/FidelityFX-SDK/Kits/FidelityFX/api/include/ffx_api_loader.h"
#include "../third_party/FidelityFX-SDK/Kits/FidelityFX/api/include/dx12/ffx_api_dx12.h"
#include "../third_party/FidelityFX-SDK/Kits/FidelityFX/upscalers/include/ffx_upscale.h"

namespace
{
    // Engine -> FFX conventions, verified with the debug view and measured image shifts.
    constexpr float kJitterSignX = -1.0f;
    constexpr float kJitterSignY = 1.0f;
    constexpr float kMotionSign  = -1.0f;  // engine motion vectors are (current - previous) in viewport UV

    constexpr int      kAllocators     = 3;     // D3D12 command allocators in flight
    constexpr DWORD    kFenceTimeoutMs = 2000;
    constexpr int      kMaxFailures    = 3;     // consecutive dispatch failures before falling back for good
    constexpr uint64_t kMaxJobAge      = 16;    // frames a recorded job may wait for its command list to execute
    constexpr UINT     kParamsSize     = 96;    // helper shader constant buffer

    // ---- Resources --------------------------------------------------------------------------------------------

    // Texture shared between D3D12 (FSR) and D3D11 (pack/unpack passes).
    struct SharedTexture
    {
        ComPtr<ID3D12Resource>            res12;
        ComPtr<ID3D11Texture2D>           tex11;
        ComPtr<ID3D11UnorderedAccessView> uav11;
        ComPtr<ID3D11ShaderResourceView>  srv11;
    };

    // Everything that determines the shared textures and the FFX context.
    struct GenerationKey
    {
        uint32_t width = 0, height = 0;  // engine render target (= display) size
        bool     jitterCancellation = false;
        bool     debugChecking = false;
        bool     autoExposure = false;
        uint64_t versionId = 0;          // 0 = newest provider

        bool operator==(const GenerationKey&) const = default;
    };

    // One set of shared textures + FFX context. Recorded jobs keep the generation they were recorded against;
    // a replaced generation is retired and freed once the GPU is past all work submitted until then.
    struct Generation
    {
        GenerationKey key;
        SharedTexture color, motion, depth, exposure, output;
        ffxContext    context = nullptr;
        std::string   provider;
    };

    // A TAA dispatch replaced at record time, waiting for its command list to be executed.
    struct Job
    {
        ComPtr<ID3D11CommandList>     partA;  // engine work up to the TAA dispatch + our pack pass
        std::shared_ptr<Generation>   generation;
        std::shared_ptr<const Config> settings;
        fsr::FrameConstants           fc;
        uint32_t                      outputWidth = 0, outputHeight = 0;
        uint64_t                      recordedFrame = 0;
    };

    // Engine bindings at the TAA dispatch.
    struct TaaBindings
    {
        ComPtr<ID3D11ShaderResourceView>  motion;    // t1 ro_motionvectors (R16G16_FLOAT)
        ComPtr<ID3D11ShaderResourceView>  color;     // t2 ro_viewcolormap (HDR)
        ComPtr<ID3D11ShaderResourceView>  exposure;  // t3 auto-exposure buffer
        ComPtr<ID3D11UnorderedAccessView> result;    // u1 rw_taaresult (display-size texture)
    };

    // Render target view of the engine's motion vector texture plus a scratch copy to read from.
    struct MotionExpansion
    {
        ComPtr<ID3D11Resource>           engineTexture;
        ComPtr<ID3D11RenderTargetView>   engineRTV;
        ComPtr<ID3D11Texture2D>          scratch;
        ComPtr<ID3D11ShaderResourceView> scratchSRV;
    };

    struct State
    {
        bool initialized = false;
        bool failed = false;               // FSR unusable for the session; upscale frames use the fallback
        bool helpersReady = false;         // D3D11 helper shaders (needed by the fallback too)
        bool motionPathBroken = false;     // motion vector expansion impossible: upscale mode unavailable
        bool d3d12CreatesTextures = true;  // DXVK fork imports D3D12 textures; native D3D11 needs the reverse
        int  consecutiveFailures = 0;
        uint64_t frame = 0;

        // D3D11 helpers
        ComPtr<ID3D11Device1>            device11;
        ComPtr<ID3D11DeviceContext4>     immediate4;
        ComPtr<ID3D11ComputeShader>      packCS, unpackCS, upsampleCS;
        ComPtr<ID3D11VertexShader>       fullscreenVS;
        ComPtr<ID3D11PixelShader>        motionExpandPS;
        ComPtr<ID3D11RasterizerState>    noCull;
        ComPtr<ID3D11SamplerState>       linearClamp;
        ComPtr<ID3D11Buffer>             params;
        ComPtr<ID3D11ShaderResourceView> depthSRV;  // latest engine depth
        MotionExpansion                  motionExpansion;

        // D3D12
        ComPtr<ID3D12Device>              device12;
        ComPtr<ID3D12CommandQueue>        queue;
        ComPtr<ID3D12CommandAllocator>    allocators[kAllocators];
        uint64_t                          allocatorFence[kAllocators] = {};
        ComPtr<ID3D12GraphicsCommandList> cmdList;
        ComPtr<ID3D12Fence>               fence12;
        ComPtr<ID3D11Fence>               fence11;
        HANDLE                            fenceEvent = nullptr;
        std::atomic<uint64_t>             fenceValue{ 0 };  // allocated under g_mutex, read anywhere
        int                               slotCounter = 0;

        // GPU timing: one timestamp pair per allocator slot
        ComPtr<ID3D12QueryHeap> timestamps;
        ComPtr<ID3D12Resource>  timestampReadback;
        uint64_t                timestampFreq = 0;
        bool                    timestampPending[kAllocators] = {};

        // FFX
        HMODULE                  ffxModule = nullptr;
        ffxFunctions             ffx = {};
        std::vector<std::string> versionNames;  // newest first
        std::vector<uint64_t>    versionIds;

        std::shared_ptr<Generation> generation;  // current

        // Jobs between record and execution
        std::unordered_map<ID3D11DeviceContext*, Job> recording;  // recorded, command list not finished yet
        std::unordered_map<ID3D11CommandList*, Job>   finished;   // finished, waiting for ExecuteCommandList

        // Dispatch continuity
        float         lastFrameNumber = -1.0f;
        Generation*   lastGeneration = nullptr;
        LARGE_INTEGER lastDispatchTime = {};

        // Stats / status
        int      frames = 0, resets = 0, gpuSamples = 0;
        double   gpuMsSum = 0, gpuMsMax = 0, gpuMsSmoothed = 0;
        std::array<float, 256> gpuMsRecent{};  // ring of recent samples for the overlay's peak
        size_t   gpuMsRecentNext = 0;
        int      resetsTotal = 0;
        uint32_t lastRenderW = 0, lastRenderH = 0, lastOutputW = 0, lastOutputH = 0;
        int      dumpFrames = 0, dumpIndex = 0;
    };

    State      g;
    std::mutex g_mutex;
    thread_local bool t_ownFinish = false;  // FinishCommandList issued by the mod (the split), not the engine

    // Retired generations wait here until the fence passes the value at retirement.
    struct Retired
    {
        std::unique_ptr<Generation> generation;
        uint64_t                    fence;
    };
    std::mutex           g_retiredMutex;
    std::vector<Retired> g_retired;

    std::shared_ptr<Generation> MakeGeneration()
    {
        return std::shared_ptr<Generation>(new Generation(), [](Generation* retired) {
            std::lock_guard lock(g_retiredMutex);
            g_retired.push_back({ std::unique_ptr<Generation>(retired), g.fenceValue.load() });
        });
    }

    void DestroyGeneration(Generation& gen)
    {
        if (gen.context)
            g.ffx.DestroyContext(&gen.context, nullptr);
    }

    void FfxMessage(uint32_t type, const wchar_t* message)
    {
        Log("FFX {}: {}", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", Narrow(message));
    }

    void Fail(const char* reason)
    {
        if (!g.failed)
            Log("fsr: {}, falling back to engine passes", reason);
        g.failed = true;
    }

    // ---- Fences -----------------------------------------------------------------------------------------------

    // Waits until the D3D12 fence reaches value. False on timeout or device removal (then nothing that the GPU
    // might still use may be reset or freed).
    bool WaitForFence(uint64_t value, DWORD timeoutMs = kFenceTimeoutMs)
    {
        auto completed = [] {
            uint64_t v = g.fence12->GetCompletedValue();
            if (v == UINT64_MAX)
                Fail("D3D12 device removed");
            return v;
        };
        if (completed() >= value)
            return true;
        if (FAILED(g.fence12->SetEventOnCompletion(value, g.fenceEvent)))
            return false;
        WaitForSingleObject(g.fenceEvent, timeoutMs);
        bool done = completed() >= value;
        if (!done)
            LOG_LIMITED(5, "fsr: fence wait for {} timed out", value);
        return done;
    }

    bool WaitForQueueIdle()
    {
        uint64_t value = ++g.fenceValue;
        return SUCCEEDED(g.queue->Signal(g.fence12.Get(), value)) && WaitForFence(value);
    }

    // ---- Device setup -----------------------------------------------------------------------------------------

    // The game's adapter may be DXVK's wrapper, so D3D12 gets the real adapter from the system DXGI by LUID.
    ComPtr<IDXGIAdapter> FindNativeAdapter(ID3D11Device* device)
    {
        DXGI_ADAPTER_DESC desc = {};
        ComPtr<IDXGIDevice> dxgiDevice;
        ComPtr<IDXGIAdapter> gameAdapter;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice))) && SUCCEEDED(dxgiDevice->GetAdapter(&gameAdapter)))
            gameAdapter->GetDesc(&desc);
        Log("fsr: game adapter '{}'", Narrow(desc.Description));

        wchar_t system[MAX_PATH] = {};
        GetSystemDirectoryW(system, MAX_PATH);
        HMODULE dxgi = LoadLibraryW((std::filesystem::path(system) / L"dxgi.dll").c_str());
        using PFN_CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
        auto createFactory = dxgi ? reinterpret_cast<PFN_CreateFactory2>(GetProcAddress(dxgi, "CreateDXGIFactory2")) : nullptr;
        ComPtr<IDXGIFactory4> factory;
        if (!createFactory || FAILED(createFactory(0, IID_PPV_ARGS(&factory))))
            return nullptr;
        ComPtr<IDXGIAdapter> native;
        if (FAILED(factory->EnumAdapterByLuid(desc.AdapterLuid, IID_PPV_ARGS(&native))))
            factory->EnumAdapters(0, &native);
        return native;
    }

    bool CreateD3D11Helpers()
    {
        auto* device = g.device11.Get();
        HRESULT hr = device->CreateComputeShader(g_cs_pack, sizeof(g_cs_pack), nullptr, &g.packCS);
        if (SUCCEEDED(hr))
            hr = device->CreateComputeShader(g_cs_unpack, sizeof(g_cs_unpack), nullptr, &g.unpackCS);
        if (SUCCEEDED(hr))
            hr = device->CreateComputeShader(g_cs_upsample, sizeof(g_cs_upsample), nullptr, &g.upsampleCS);
        if (SUCCEEDED(hr))
            hr = device->CreateVertexShader(g_vs_fullscreen, sizeof(g_vs_fullscreen), nullptr, &g.fullscreenVS);
        if (SUCCEEDED(hr))
            hr = device->CreatePixelShader(g_ps_motion_expand, sizeof(g_ps_motion_expand), nullptr, &g.motionExpandPS);
        if (SUCCEEDED(hr))
        {
            D3D11_RASTERIZER_DESC rd = {};
            rd.FillMode = D3D11_FILL_SOLID;
            rd.CullMode = D3D11_CULL_NONE;
            rd.DepthClipEnable = TRUE;
            hr = device->CreateRasterizerState(&rd, &g.noCull);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_SAMPLER_DESC sd = {};
            sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.MaxLOD = D3D11_FLOAT32_MAX;
            hr = device->CreateSamplerState(&sd, &g.linearClamp);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = kParamsSize;
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            hr = device->CreateBuffer(&bd, nullptr, &g.params);
        }
        if (FAILED(hr))
            Log("fsr: helper shaders failed {}", HrText(hr));
        return SUCCEEDED(hr);
    }

    bool CreateD3D12Queue(IDXGIAdapter* adapter, ID3D11Device5* device5)
    {
        HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g.device12));
        Log("fsr: D3D12 device -> {}", HrText(hr));
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (SUCCEEDED(hr))
            hr = g.device12->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
        for (auto& allocator : g.allocators)
            if (SUCCEEDED(hr))
                hr = g.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        if (SUCCEEDED(hr))
            hr = g.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.allocators[0].Get(), nullptr,
                                               IID_PPV_ARGS(&g.cmdList));
        if (SUCCEEDED(hr))
            hr = g.cmdList->Close();

        // One shared fence, created by D3D12 (the direction both native D3D11 and DXVK can open).
        HANDLE handle = nullptr;
        if (SUCCEEDED(hr))
            hr = g.device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g.fence12));
        if (SUCCEEDED(hr))
            hr = g.device12->CreateSharedHandle(g.fence12.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
        if (SUCCEEDED(hr))
            hr = device5->OpenSharedFence(handle, IID_PPV_ARGS(&g.fence11));
        if (handle)
            CloseHandle(handle);
        g.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (FAILED(hr) || !g.fenceEvent)
        {
            Log("fsr: D3D12 queue/fence setup failed {}", HrText(hr));
            return false;
        }

        D3D12_QUERY_HEAP_DESC qh = {};
        qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qh.Count = 2 * kAllocators;
        D3D12_HEAP_PROPERTIES readback = { D3D12_HEAP_TYPE_READBACK };
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 16 * kAllocators;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(g.device12->CreateQueryHeap(&qh, IID_PPV_ARGS(&g.timestamps))) ||
            FAILED(g.device12->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                       nullptr, IID_PPV_ARGS(&g.timestampReadback))) ||
            FAILED(g.queue->GetTimestampFrequency(&g.timestampFreq)))
        {
            g.timestamps.Reset();
            Log("fsr: GPU timestamps unavailable");
        }
        return true;
    }

    bool LoadFfx()
    {
        g.ffxModule = LoadLibraryW(ExeFile(L"amd_fidelityfx_loader_dx12.dll").c_str());
        if (!g.ffxModule)
        {
            Log("fsr: amd_fidelityfx_loader_dx12.dll not found next to the exe ({})", GetLastError());
            return false;
        }
        ffxLoadFunctions(&g.ffx, g.ffxModule);
        if (!g.ffx.CreateContext || !g.ffx.Dispatch || !g.ffx.Query || !g.ffx.DestroyContext)
        {
            Log("fsr: loader exports missing");
            return false;
        }

        ffxQueryDescGetVersions versions = {};
        versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        versions.device = g.device12.Get();
        uint64_t count = 0;
        versions.outputCount = &count;
        g.ffx.Query(nullptr, &versions.header);

        constexpr uint64_t kMaxVersions = 32;
        if (count > 0 && count <= kMaxVersions)
        {
            uint64_t ids[kMaxVersions];
            const char* names[kMaxVersions];
            versions.versionIds = ids;
            versions.versionNames = names;
            g.ffx.Query(nullptr, &versions.header);
            for (uint64_t i = 0; i < count; ++i)
            {
                Log("fsr: available upscaler {}: {}", i, names[i]);
                g.versionNames.emplace_back(names[i]);
                g.versionIds.push_back(ids[i]);
            }
        }
        return true;
    }

    // Lazily brings up both devices on the first TAA dispatch. Caller holds g_mutex.
    bool EnsureInitialized(ID3D11DeviceContext* ctx)
    {
        if (g.initialized)
            return g.helpersReady;
        g.initialized = true;

        ComPtr<ID3D11Device> device;
        ctx->GetDevice(&device);
        ComPtr<ID3D11DeviceContext> immediate;
        device->GetImmediateContext(&immediate);
        ComPtr<ID3D11Device5> device5;
        HRESULT hr = device.As(&g.device11);
        if (SUCCEEDED(hr))
            hr = device.As(&device5);
        if (SUCCEEDED(hr))
            hr = immediate.As(&g.immediate4);
        if (FAILED(hr))
        {
            Log("fsr: D3D11.4 interfaces unavailable {}", HrText(hr));
            Fail("initialization failed");
            return false;
        }

        g.helpersReady = CreateD3D11Helpers();
        ComPtr<IDXGIAdapter> adapter = FindNativeAdapter(device.Get());
        if (!g.helpersReady || !CreateD3D12Queue(adapter.Get(), device5.Get()) || !LoadFfx())
            Fail("initialization failed");
        return g.helpersReady;
    }

    // ---- Shared textures and generations ------------------------------------------------------------------------

    bool CreateViews(SharedTexture& t)
    {
        HRESULT hr = g.device11->CreateUnorderedAccessView(t.tex11.Get(), nullptr, &t.uav11);
        if (SUCCEEDED(hr))
            hr = g.device11->CreateShaderResourceView(t.tex11.Get(), nullptr, &t.srv11);
        if (FAILED(hr))
            Log("fsr: D3D11 view on shared texture failed {}", HrText(hr));
        return SUCCEEDED(hr);
    }

    // D3D12-created texture opened by D3D11 (works on the patched DXVK).
    bool ShareFromD3D12(SharedTexture& t, DXGI_FORMAT format, uint32_t w, uint32_t h)
    {
        D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = w;
        desc.Height = h;
        desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        HRESULT hr = g.device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                         nullptr, IID_PPV_ARGS(&t.res12));
        HANDLE handle = nullptr;
        if (SUCCEEDED(hr))
            hr = g.device12->CreateSharedHandle(t.res12.Get(), nullptr, GENERIC_ALL, nullptr, &handle);
        if (SUCCEEDED(hr))
            hr = g.device11->OpenSharedResource1(handle, IID_PPV_ARGS(&t.tex11));
        if (handle)
            CloseHandle(handle);
        return SUCCEEDED(hr);
    }

    // D3D11-created texture opened by D3D12 (native D3D11 runtime).
    bool ShareFromD3D11(SharedTexture& t, DXGI_FORMAT format, uint32_t w, uint32_t h)
    {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = w;
        desc.Height = h;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_RENDER_TARGET;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        HRESULT hr = g.device11->CreateTexture2D(&desc, nullptr, &t.tex11);
        ComPtr<IDXGIResource1> dxgiResource;
        HANDLE handle = nullptr;
        if (SUCCEEDED(hr))
            hr = t.tex11.As(&dxgiResource);
        if (SUCCEEDED(hr))
            hr = dxgiResource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                                  &handle);
        if (SUCCEEDED(hr))
            hr = g.device12->OpenSharedHandle(handle, IID_PPV_ARGS(&t.res12));
        if (handle)
            CloseHandle(handle);
        return SUCCEEDED(hr);
    }

    bool CreateShared(SharedTexture& t, DXGI_FORMAT format, uint32_t w, uint32_t h)
    {
        bool ok = g.d3d12CreatesTextures ? ShareFromD3D12(t, format, w, h) : ShareFromD3D11(t, format, w, h);
        if (!ok && g.d3d12CreatesTextures)
        {
            t = {};
            g.d3d12CreatesTextures = false;  // not the DXVK fork: D3D11 can't import D3D12 textures
            ok = ShareFromD3D11(t, format, w, h);
        }
        if (!ok)
        {
            Log("fsr: sharing a {}x{} texture (format {}) failed in both directions", w, h, static_cast<int>(format));
            return false;
        }
        return CreateViews(t);
    }

    bool CreateContext(Generation& gen)
    {
        const GenerationKey& key = gen.key;
        ffxCreateContextDescUpscale create = {};
        create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        // Both modes feed the engine's linear HDR scene color before tonemapping.
        create.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE |
                       FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION | FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION |
                       FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
        if (key.jitterCancellation)
            create.flags |= FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
        if (key.debugChecking)
            create.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
        if (key.autoExposure)
            create.flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        create.maxRenderSize = { key.width, key.height };
        create.maxUpscaleSize = { key.width, key.height };
        create.fpMessage = FfxMessage;

        ffxCreateBackendDX12Desc backend = {};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = g.device12.Get();
        ffxCreateContextDescUpscaleVersion version = {};
        version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        version.version = FFX_UPSCALER_VERSION;
        ffxOverrideVersion pinned = {};
        pinned.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        pinned.versionId = key.versionId;
        create.header.pNext = &backend.header;
        backend.header.pNext = &version.header;
        if (key.versionId)
            version.header.pNext = &pinned.header;

        if (ffxReturnCode_t rc = g.ffx.CreateContext(&gen.context, &create.header, nullptr); rc != FFX_API_RETURN_OK)
        {
            Log("fsr: ffxCreateContext failed ({})", rc);
            gen.context = nullptr;
            return false;
        }

        ffxQueryGetProviderVersion provider = {};
        provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        g.ffx.Query(&gen.context, &provider.header);
        gen.provider = provider.versionName ? provider.versionName : "?";

        if (key.debugChecking)
        {
            ffxConfigureDescGlobalDebug1 debug = {};
            debug.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
            debug.fpMessage = FfxMessage;
            debug.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
            g.ffx.Configure(&gen.context, &debug.header);
        }
        return true;
    }

    uint64_t VersionId(const std::string& name)
    {
        auto found = std::ranges::find(g.versionNames, name);
        return found != g.versionNames.end() ? g.versionIds[found - g.versionNames.begin()] : 0;
    }

    // The generation matching the current settings and texture size, created if needed. Caller holds g_mutex.
    std::shared_ptr<Generation> EnsureGeneration(const Config& cfg, uint32_t width, uint32_t height)
    {
        if (g.failed)
            return nullptr;
        GenerationKey key{ width, height, cfg.jitterCancellation, cfg.fsrDebugChecking, cfg.fsrAutoExposure,
                           VersionId(cfg.fsrVersion) };
        if (g.generation && g.generation->key == key)
            return g.generation;

        LARGE_INTEGER t0, t1, freq;
        QueryPerformanceCounter(&t0);
        std::shared_ptr<Generation> gen = MakeGeneration();
        gen->key = key;
        bool ok = CreateShared(gen->color, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height) &&
                  CreateShared(gen->motion, DXGI_FORMAT_R16G16_FLOAT, width, height) &&
                  CreateShared(gen->depth, DXGI_FORMAT_R32_FLOAT, width, height) &&
                  CreateShared(gen->exposure, DXGI_FORMAT_R32_FLOAT, 1, 1) &&
                  CreateShared(gen->output, DXGI_FORMAT_R16G16B16A16_FLOAT, width, height) && CreateContext(*gen);
        if (!ok)
        {
            Fail("shared texture / context creation failed");
            return nullptr;
        }
        QueryPerformanceCounter(&t1);
        QueryPerformanceFrequency(&freq);
        Log("fsr: generation {}x{} ready in {:.1f} ms (provider {}, textures {})", width, height,
            (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart, gen->provider,
            g.d3d12CreatesTextures ? "D3D12 -> D3D11" : "D3D11 -> D3D12");
        if (!cfg.fsrVersion.empty() && !key.versionId)
            Log("fsr: requested version {} not available, using the newest", cfg.fsrVersion);
        g.generation = gen;  // the previous one retires once its last job has executed
        return gen;
    }

    void FreeRetiredGenerations()
    {
        if (!g.fence12)
            return;
        uint64_t completed = g.fence12->GetCompletedValue();
        std::lock_guard lock(g_retiredMutex);
        std::erase_if(g_retired, [completed](Retired& r) {
            if (completed < r.fence)
                return false;
            DestroyGeneration(*r.generation);
            return true;
        });
    }

    // ---- Recording helpers --------------------------------------------------------------------------------------

    template <typename T>
    void WriteParams(ID3D11DeviceContext* ctx, const T& params)
    {
        static_assert(sizeof(T) <= kParamsSize);
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(ctx->Map(g.params.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        {
            memcpy(mapped.pData, &params, sizeof(T));
            ctx->Unmap(g.params.Get(), 0);
        }
    }

    // Runs a helper compute shader over w x h pixels (8x8 groups) with the given bindings and parameters.
    template <typename Params>
    void RunCompute(ID3D11DeviceContext* ctx, ID3D11ComputeShader* shader, std::initializer_list<ID3D11ShaderResourceView*> srvs,
                    std::initializer_list<ID3D11UnorderedAccessView*> uavs, const Params& params, uint32_t w, uint32_t h,
                    ID3D11SamplerState* sampler = nullptr)
    {
        ComputeStateGuard guard(ctx);
        ComPtr<ID3D11SamplerState> savedSampler;
        if (sampler)
            ctx->CSGetSamplers(0, 1, &savedSampler);
        WriteParams(ctx, params);
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->CSSetShaderResources(0, static_cast<UINT>(srvs.size()), srvs.begin());
        ctx->CSSetUnorderedAccessViews(0, static_cast<UINT>(uavs.size()), uavs.begin(), nullptr);
        ctx->CSSetConstantBuffers(0, 1, g.params.GetAddressOf());
        if (sampler)
            ctx->CSSetSamplers(0, 1, &sampler);
        ctx->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        if (sampler)
            ctx->CSSetSamplers(0, 1, savedSampler.GetAddressOf());
    }

    D3D11_TEXTURE2D_DESC TextureDesc(ID3D11View* view)
    {
        ComPtr<ID3D11Resource> resource;
        view->GetResource(&resource);
        D3D11_TEXTURE2D_DESC desc = {};
        ComPtr<ID3D11Texture2D> texture;
        if (SUCCEEDED(resource.As(&texture)))
            texture->GetDesc(&desc);
        return desc;
    }

    std::optional<TaaBindings> ReadTaaBindings(ID3D11DeviceContext* ctx)
    {
        ComArray<ID3D11ShaderResourceView, 4> srvs;
        ComArray<ID3D11UnorderedAccessView, 2> uavs;
        ctx->CSGetShaderResources(0, srvs.size(), srvs.data());
        ctx->CSGetUnorderedAccessViews(0, uavs.size(), uavs.data());
        if (!srvs[1] || !srvs[2] || !srvs[3] || !uavs[1])
            return std::nullopt;
        return TaaBindings{ srvs[1], srvs[2], srvs[3], uavs[1] };
    }

    // Engine inputs -> shared textures (render region) with the SSGI bounce light added to the colour, plus exposure
    // as the engine TAA computes it.
    void PackInputs(ID3D11DeviceContext* ctx, const Generation& gen, const fsr::FrameConstants& fc, const TaaBindings& in)
    {
        std::shared_ptr<const Config> cfg = config::Current();
        material::Views materials = material::ShaderResources();
        ssgi::Result lighting = ssgi::Record(ctx, {
            .color                  = in.color.Get(),
            .depth                  = g.depthSRV.Get(),
            .motion                 = in.motion.Get(),
            .exposure               = in.exposure.Get(),
            .surface                = materials.surface,
            .width                  = fc.renderWidth,
            .height                 = fc.renderHeight,
            .cameraNear             = fc.cameraNear,
            .projX                  = fc.projX,
            .projY                  = fc.projY,
            .frame                  = static_cast<uint32_t>(fc.frameNumber),
            .exposureIndex          = fc.exposureIndex,
            .compressedHdr          = fc.compressedHdr,
            .viewWhite              = fc.viewWhite,
            .inverseViewProjection  = fc.inverseViewProjection,
            .previousViewProjection = fc.previousViewProjection,
        });
        struct
        {
            uint32_t width, height, exposureIndex, compressedHdr;
            float    viewWhite, evBias, maxLuminance, cameraNear;
            float    giStrength, aoStrength;
            uint32_t debugView, giEnabled, halfWidth, halfHeight;
            float    giAlbedo;
            uint32_t materialValid;
            float    projScaleX, projScaleY, ssrStrength;
            uint32_t ssrEnabled;
            float    specularOcclusion, pad[3];
        } params = { fc.renderWidth, fc.renderHeight, fc.exposureIndex, fc.compressedHdr, fc.viewWhite, fc.evBias,
                     cfg->highlightClamp, fc.cameraNear, cfg->ssgiStrength, cfg->ssgiAo,
                     static_cast<uint32_t>(cfg->ssgiDebug), lighting.light ? 1u : 0u, lighting.halfWidth, lighting.halfHeight,
                     cfg->ssgiAlbedo, materials ? 1u : 0u, 1.0f / fc.projX, 1.0f / fc.projY, cfg->ssrStrength,
                     lighting.reflection ? 1u : 0u, cfg->specularOcclusion, {} };
        RunCompute(ctx, g.packCS.Get(),
                   { in.color.Get(), in.motion.Get(), g.depthSRV.Get(), in.exposure.Get(), lighting.light, lighting.geometry,
                     materials.albedo, lighting.reflection, materials.specular, lighting.debug, materials.surface },
                   { gen.color.uav11.Get(), gen.motion.uav11.Get(), gen.depth.uav11.Get(), gen.exposure.uav11.Get() }, params,
                   fc.renderWidth, fc.renderHeight);
        if (materials)
            material::Clear(ctx);  // unknown until the next frame's forward pass writes it
    }

    // FSR output -> engine taaresult (top-left w x h).
    void UnpackOutput(ID3D11DeviceContext* ctx, const Generation& gen, ID3D11UnorderedAccessView* result, uint32_t w, uint32_t h)
    {
        struct { uint32_t width, height, pad[2]; } params = { w, h, {} };
        RunCompute(ctx, g.unpackCS.Get(), { gen.output.srv11.Get() }, { result }, params, w, h);
    }

    // Fallback: render-resolution color stretched over the whole taaresult (bilinear, no anti-aliasing).
    void UpsampleColor(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc, const TaaBindings& in, uint32_t ow, uint32_t oh)
    {
        D3D11_TEXTURE2D_DESC source = TextureDesc(in.color.Get());
        if (!source.Width || !source.Height)
            return;
        struct { uint32_t width, height; float uvScaleX, uvScaleY; } params = {
            ow, oh, static_cast<float>(fc.renderWidth) / source.Width, static_cast<float>(fc.renderHeight) / source.Height };
        RunCompute(ctx, g.upsampleCS.Get(), { in.color.Get() }, { in.result.Get() }, params, ow, oh, g.linearClamp.Get());
    }

    // Prepares writing display-size motion vectors into the engine's motion vector texture (for motion blur at
    // display resolution). Must succeed before an upscale frame is committed.
    bool PrepareMotionExpansion(ID3D11ShaderResourceView* engineMotion)
    {
        ComPtr<ID3D11Resource> resource;
        engineMotion->GetResource(&resource);
        MotionExpansion& mv = g.motionExpansion;
        if (mv.engineTexture == resource && mv.engineRTV && mv.scratchSRV)
            return true;

        mv = {};
        D3D11_TEXTURE2D_DESC desc = TextureDesc(engineMotion);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = 0;
        desc.CPUAccessFlags = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        D3D11_RENDER_TARGET_VIEW_DESC rtv = {};
        rtv.Format = DXGI_FORMAT_R16G16_FLOAT;
        rtv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        D3D11_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Format = DXGI_FORMAT_R16G16_FLOAT;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;

        HRESULT hr = g.device11->CreateRenderTargetView(resource.Get(), &rtv, &mv.engineRTV);
        if (SUCCEEDED(hr))
            hr = g.device11->CreateTexture2D(&desc, nullptr, &mv.scratch);
        if (SUCCEEDED(hr))
            hr = g.device11->CreateShaderResourceView(mv.scratch.Get(), &srv, &mv.scratchSRV);
        if (FAILED(hr))
        {
            mv = {};
            g.motionPathBroken = true;
            Log("fsr: motion vector expansion unavailable ({}), upscale mode off", HrText(hr));
            return false;
        }
        mv.engineTexture = resource;
        Log("fsr: motion vector expansion ready ({}x{})", desc.Width, desc.Height);
        return true;
    }

    // Engine motion vectors (render region) -> nearest-texel expansion over the whole texture. MVs are viewport-UV
    // deltas, so the values themselves are resolution independent.
    void ExpandMotionVectors(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc, uint32_t ow, uint32_t oh)
    {
        const MotionExpansion& mv = g.motionExpansion;
        D3D11_BOX region = { 0, 0, 0, fc.renderWidth, fc.renderHeight, 1 };
        ctx->CopySubresourceRegion(mv.scratch.Get(), 0, 0, 0, 0, mv.engineTexture.Get(), 0, &region);

        DrawStateGuard guard(ctx);
        struct { uint32_t renderW, renderH, outputW, outputH; } params = { fc.renderWidth, fc.renderHeight, ow, oh };
        WriteParams(ctx, params);
        D3D11_VIEWPORT viewport = { 0.0f, 0.0f, static_cast<float>(ow), static_cast<float>(oh), 0.0f, 1.0f };
        ctx->OMSetRenderTargets(1, mv.engineRTV.GetAddressOf(), nullptr);
        ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        ctx->OMSetDepthStencilState(nullptr, 0);
        ctx->RSSetState(g.noCull.Get());
        ctx->RSSetViewports(1, &viewport);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(g.fullscreenVS.Get(), nullptr, 0);
        ctx->PSSetShader(g.motionExpandPS.Get(), nullptr, 0);
        ctx->GSSetShader(nullptr, nullptr, 0);
        ctx->HSSetShader(nullptr, nullptr, 0);
        ctx->DSSetShader(nullptr, nullptr, 0);
        ctx->PSSetShaderResources(0, 1, mv.scratchSRV.GetAddressOf());
        ctx->PSSetConstantBuffers(0, 1, g.params.GetAddressOf());
        ctx->Draw(3, 0);
    }

    // Ends the engine's command list here (part A) and keeps recording into part B with the same state.
    ComPtr<ID3D11CommandList> SplitCommandList(ID3D11DeviceContext* ctx)
    {
        ComPtr<ID3D11CommandList> partA;
        t_ownFinish = true;
        HRESULT hr = ctx->FinishCommandList(TRUE, &partA);
        t_ownFinish = false;
        if (FAILED(hr))
        {
            LOG_LIMITED(5, "fsr: FinishCommandList split failed {}", HrText(hr));
            partA.Reset();
        }
        return partA;
    }

    void StoreJob(ID3D11DeviceContext* ctx, ComPtr<ID3D11CommandList> partA, std::shared_ptr<Generation> gen,
                  const fsr::FrameConstants& fc, uint32_t ow, uint32_t oh)
    {
        g.recording[ctx] = Job{ std::move(partA), std::move(gen), config::Current(), fc, ow, oh, g.frame };
    }

    // ---- Record paths -------------------------------------------------------------------------------------------

    // Native mode: FSR anti-aliases at render resolution; the engine upsamples later.
    bool RecordNativeAA(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc, const TaaBindings& in,
                        std::shared_ptr<Generation> gen)
    {
        PackInputs(ctx, *gen, fc, in);
        ComPtr<ID3D11CommandList> partA = SplitCommandList(ctx);
        if (!partA)
            return false;
        UnpackOutput(ctx, *gen, in.result.Get(), fc.renderWidth, fc.renderHeight);
        StoreJob(ctx, std::move(partA), std::move(gen), fc, fc.renderWidth, fc.renderHeight);
        return true;
    }

    // Upscale mode: a display-size image in taaresult and display-size motion vectors, from FSR when possible,
    // else from the bilinear fallback. The caller has committed this frame's post chain to display resolution.
    bool RecordUpscale(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc, const TaaBindings& in,
                       std::shared_ptr<Generation> gen)
    {
        D3D11_TEXTURE2D_DESC target = TextureDesc(in.result.Get());
        uint32_t ow = target.Width, oh = target.Height;
        bool expandMotion = PrepareMotionExpansion(in.motion.Get());
        LOG_LIMITED(1, "fsr: upscale at TAA, render {}x{} -> {}x{}", fc.renderWidth, fc.renderHeight, ow, oh);

        if (gen)
        {
            PackInputs(ctx, *gen, fc, in);
            if (ComPtr<ID3D11CommandList> partA = SplitCommandList(ctx))
            {
                UnpackOutput(ctx, *gen, in.result.Get(), ow, oh);
                if (expandMotion)
                    ExpandMotionVectors(ctx, fc, ow, oh);
                StoreJob(ctx, std::move(partA), std::move(gen), fc, ow, oh);
                return true;
            }
        }

        LOG_LIMITED(5, "fsr: FSR unavailable in an upscale frame, bilinear fallback");
        UpsampleColor(ctx, fc, in, ow, oh);
        if (expandMotion)
            ExpandMotionVectors(ctx, fc, ow, oh);
        return true;
    }

    // ---- Execution --------------------------------------------------------------------------------------------

    FfxApiResource Resource(const SharedTexture& t)
    {
        return ffxApiGetResourceDX12(t.res12.Get(), FFX_API_RESOURCE_STATE_COMMON);
    }

    void ReadTimestamps(int slot)
    {
        if (!g.timestamps || !g.timestampPending[slot])
            return;
        g.timestampPending[slot] = false;
        D3D12_RANGE range = { slot * 16u, slot * 16u + 16u };
        uint64_t* ticks = nullptr;
        if (FAILED(g.timestampReadback->Map(0, &range, reinterpret_cast<void**>(&ticks))))
            return;
        double ms = static_cast<double>(ticks[slot * 2 + 1] - ticks[slot * 2]) * 1000.0 / static_cast<double>(g.timestampFreq);
        D3D12_RANGE none = { 0, 0 };
        g.timestampReadback->Unmap(0, &none);
        if (ms <= 0.0 || ms >= 1000.0)
            return;
        g.gpuMsSum += ms;
        g.gpuMsMax = std::max(g.gpuMsMax, ms);
        g.gpuMsSmoothed = g.gpuMsSmoothed == 0.0 ? ms : g.gpuMsSmoothed * 0.9 + ms * 0.1;
        g.gpuMsRecent[g.gpuMsRecentNext++ % g.gpuMsRecent.size()] = static_cast<float>(ms);
        ++g.gpuSamples;
    }

    void CountFailure(const char* what, int code)
    {
        LOG_LIMITED(5, "fsr: {} ({})", what, code);
        if (++g.consecutiveFailures >= kMaxFailures)
            Fail("repeated dispatch failures");
    }

    void DumpInputs(const Job& job, FfxApiFloatCoords2D jitter, uint64_t afterFence);

    // D3D12 side of one job: wait for the packed inputs, upscale, signal. Returns the fence value D3D11 waits on
    // (inputsReady itself, already signaled, if the dispatch had to be skipped).
    uint64_t RunUpscale(const Job& job, uint64_t inputsReady)
    {
        Generation& gen = *job.generation;
        const Config& cfg = *job.settings;
        const fsr::FrameConstants& fc = job.fc;

        int slot = g.slotCounter++ % kAllocators;
        if (!WaitForFence(g.allocatorFence[slot]))
        {
            CountFailure("command allocator still in use, frame skipped", 0);
            return inputsReady;
        }
        ReadTimestamps(slot);
        ID3D12CommandAllocator* allocator = g.allocators[slot].Get();
        if (FAILED(allocator->Reset()) || FAILED(g.cmdList->Reset(allocator, nullptr)))
        {
            CountFailure("command list reset failed", 0);
            return inputsReady;
        }
        if (g.timestamps)
            g.cmdList->EndQuery(g.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2);

        LARGE_INTEGER now, freq;
        QueryPerformanceCounter(&now);
        QueryPerformanceFrequency(&freq);
        float frameMs = g.lastDispatchTime.QuadPart
                            ? static_cast<float>((now.QuadPart - g.lastDispatchTime.QuadPart) * 1000.0 / freq.QuadPart)
                            : 16.7f;
        g.lastDispatchTime = now;
        // History is only valid for consecutive frames of the same context (FSR toggles break the sequence too).
        bool reset = fc.frameNumber != g.lastFrameNumber + 1.0f || g.lastGeneration != job.generation.get();
        g.lastFrameNumber = fc.frameNumber;
        g.lastGeneration = job.generation.get();
        g.resets += reset ? 1 : 0;
        g.resetsTotal += reset ? 1 : 0;

        float rw = static_cast<float>(fc.renderWidth), rh = static_cast<float>(fc.renderHeight);
        ffxDispatchDescUpscale d = {};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = g.cmdList.Get();
        d.color = Resource(gen.color);
        d.depth = Resource(gen.depth);
        d.motionVectors = Resource(gen.motion);
        d.exposure = gen.key.autoExposure ? FfxApiResource{} : Resource(gen.exposure);
        d.output = Resource(gen.output);
        d.jitterOffset = { kJitterSignX * fc.jitterX * rw, kJitterSignY * fc.jitterY * rh };
        d.motionVectorScale = { kMotionSign * rw, kMotionSign * rh };
        d.renderSize = { fc.renderWidth, fc.renderHeight };
        d.upscaleSize = { job.outputWidth, job.outputHeight };
        d.enableSharpening = cfg.fsrSharpness > 0.0f;
        d.sharpness = cfg.fsrSharpness;
        d.frameTimeDelta = frameMs;
        d.preExposure = 1.0f;
        d.reset = reset;
        d.cameraNear = fc.cameraNear;
        d.cameraFar = 1.0e6f;  // infinite projection
        d.cameraFovAngleVertical = fc.fovY;
        d.viewSpaceToMetersFactor = 1.0f;
        d.flags = cfg.fsrDebugView ? FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW : 0;

        if (FAILED(g.queue->Wait(g.fence12.Get(), inputsReady)))
        {
            g.cmdList->Close();
            CountFailure("queue wait failed", 0);
            return inputsReady;
        }
        ffxReturnCode_t rc = g.ffx.Dispatch(&gen.context, &d.header);
        if (g.timestamps)
        {
            g.cmdList->EndQuery(g.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
            g.cmdList->ResolveQueryData(g.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2,
                                        g.timestampReadback.Get(), slot * 16);
        }
        bool closed = SUCCEEDED(g.cmdList->Close());
        if (rc == FFX_API_RETURN_OK && closed)
        {
            ID3D12CommandList* lists[] = { g.cmdList.Get() };
            g.queue->ExecuteCommandLists(1, lists);
            g.timestampPending[slot] = g.timestamps != nullptr;
            g.consecutiveFailures = 0;
            ++g.frames;
        }
        else
        {
            CountFailure("ffxDispatch failed", static_cast<int>(rc));
        }

        uint64_t done = ++g.fenceValue;
        if (FAILED(g.queue->Signal(g.fence12.Get(), done)))
        {
            Fail("queue signal failed");
            return inputsReady;
        }
        g.allocatorFence[slot] = done;

        g.lastRenderW = fc.renderWidth;
        g.lastRenderH = fc.renderHeight;
        g.lastOutputW = job.outputWidth;
        g.lastOutputH = job.outputHeight;
        if (g.dumpFrames > 0)
        {
            --g.dumpFrames;
            DumpInputs(job, d.jitterOffset, done);
        }
        return done;
    }

    // Debug (F8): copies the shared upscaler inputs to disk after the GPU is done with them. Stalls the CPU.
    void DumpInputs(const Job& job, FfxApiFloatCoords2D jitter, uint64_t afterFence)
    {
        const Generation& gen = *job.generation;
        uint32_t rw = job.fc.renderWidth, rh = job.fc.renderHeight;
        struct Item
        {
            const SharedTexture& texture;
            const wchar_t*       name;
            uint32_t             bytesPerPixel, width, height;
        };
        const Item items[] = {
            { gen.color, L"color_rgba16f", 8, rw, rh },
            { gen.motion, L"motion_rg16f", 4, rw, rh },
            { gen.depth, L"depth_r32f", 4, rw, rh },
            { gen.output, L"output_rgba16f", 8, job.outputWidth, job.outputHeight },
        };
        constexpr size_t kItems = std::size(items);

        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Resource> readback[kItems];
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint[kItems] = {};
        if (FAILED(g.device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(g.device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))))
            return;
        for (size_t i = 0; i < kItems; ++i)
        {
            D3D12_RESOURCE_DESC td = items[i].texture.res12->GetDesc();
            UINT64 total = 0;
            g.device12->GetCopyableFootprints(&td, 0, 1, 0, &footprint[i], nullptr, nullptr, &total);
            D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_READBACK };
            D3D12_RESOURCE_DESC bd = {};
            bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bd.Width = total;
            bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
            bd.SampleDesc.Count = 1;
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(g.device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                           nullptr, IID_PPV_ARGS(&readback[i]))))
                return;
            D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
            src.pResource = items[i].texture.res12.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = readback[i].Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = footprint[i];
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);  // COMMON -> COPY_SOURCE promotes implicitly
        }
        if (FAILED(list->Close()) || FAILED(g.queue->Wait(g.fence12.Get(), afterFence)))
            return;
        ID3D12CommandList* lists[] = { list.Get() };
        g.queue->ExecuteCommandLists(1, lists);
        if (!WaitForQueueIdle())
            return;

        int index = g.dumpIndex++;
        for (size_t i = 0; i < kItems; ++i)
        {
            const Item& item = items[i];
            auto path = ModFile(L"captures", std::format(L"dump{}_{}_{}x{}.bin", index, item.name, item.width, item.height));
            FILE* file = nullptr;
            uint8_t* data = nullptr;
            if (_wfopen_s(&file, path.c_str(), L"wb") != 0 || !file)
                continue;
            if (SUCCEEDED(readback[i]->Map(0, nullptr, reinterpret_cast<void**>(&data))))
            {
                for (uint32_t y = 0; y < item.height; ++y)
                    fwrite(data + footprint[i].Offset + size_t{ y } * footprint[i].Footprint.RowPitch, item.bytesPerPixel,
                           item.width, file);
                readback[i]->Unmap(0, nullptr);
            }
            fclose(file);
        }

        std::string info = std::format("fsr dump {}: frame {:.0f} render {}x{} output {}x{} engine jitter ({:.6f}, {:.6f}) -> "
                                       "FSR jitter px ({:.4f}, {:.4f}) provider {}",
                                       index, job.fc.frameNumber, rw, rh, job.outputWidth, job.outputHeight, job.fc.jitterX,
                                       job.fc.jitterY, jitter.x, jitter.y, gen.provider);
        Log("{}", info);
        // Sidecar, so the values survive the log being recreated at the next launch.
        FILE* file = nullptr;
        if (_wfopen_s(&file, ModFile(L"captures", std::format(L"dump{}_info.txt", index)).c_str(), L"w") == 0 && file)
        {
            fprintf(file, "%s\n", info.c_str());
            fclose(file);
        }
    }
}

// ---- Public interface -------------------------------------------------------------------------------------------

bool fsr::RecordTaa(ID3D11DeviceContext* ctx, const FrameConstants& fc, TaaMode mode)
{
    std::shared_ptr<const Config> cfg = config::Current();
    if (!cfg->fsrEnabled || ctx->GetType() != D3D11_DEVICE_CONTEXT_DEFERRED)
        return false;

    std::lock_guard lock(g_mutex);
    if (!EnsureInitialized(ctx) || !g.depthSRV || g.recording.contains(ctx))
        return false;
    std::optional<TaaBindings> in = ReadTaaBindings(ctx);
    if (!in)
        return false;

    D3D11_TEXTURE2D_DESC target = TextureDesc(in->result.Get());
    bool sizesOk = fc.renderWidth > 0 && fc.renderHeight > 0 && fc.renderWidth <= target.Width &&
                   fc.renderHeight <= target.Height;
    std::shared_ptr<Generation> gen = sizesOk ? EnsureGeneration(*cfg, target.Width, target.Height) : nullptr;

    switch (mode)
    {
    case TaaMode::Prepare:
        return false;
    case TaaMode::NativeAA:
        return gen && RecordNativeAA(ctx, fc, *in, std::move(gen));
    case TaaMode::Upscale:
        return sizesOk && RecordUpscale(ctx, fc, *in, std::move(gen));
    }
    return false;
}

bool fsr::IsUpscaleReady()
{
    std::lock_guard lock(g_mutex);
    return g.initialized && g.helpersReady && !g.failed && !g.motionPathBroken && g.depthSRV && g.generation;
}

void fsr::OnFinishCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* list)
{
    if (t_ownFinish || !list)
        return;
    std::lock_guard lock(g_mutex);
    g.finished.erase(list);  // a stale entry for a reused address
    if (auto node = g.recording.extract(ctx))
        g.finished.emplace(list, std::move(node.mapped()));
}

void fsr::OnExecuteCommandList(ID3D11DeviceContext* immediate, ID3D11CommandList* list, BOOL restoreState)
{
    std::unique_lock lock(g_mutex);
    auto node = g.finished.extract(list);
    if (!node)
        return;
    Job job = std::move(node.mapped());
    lock.unlock();

    // Part A (engine work + pack) first, then the upscaler, then the engine executes part B.
    immediate->ExecuteCommandList(job.partA.Get(), restoreState);

    lock.lock();
    if (g.failed)
        return;
    uint64_t inputsReady = ++g.fenceValue;
    if (FAILED(g.immediate4->Signal(g.fence11.Get(), inputsReady)))
    {
        Fail("D3D11 fence signal failed");
        return;
    }
    immediate->Flush();
    uint64_t upscaled = RunUpscale(job, inputsReady);
    if (FAILED(g.immediate4->Wait(g.fence11.Get(), upscaled)))
        Fail("D3D11 fence wait failed");
}

void fsr::OnFrame()
{
    std::lock_guard lock(g_mutex);
    ++g.frame;
    std::erase_if(g.finished, [](const auto& entry) { return entry.second.recordedFrame + kMaxJobAge < g.frame; });
    std::erase_if(g.recording, [](const auto& entry) { return entry.second.recordedFrame + kMaxJobAge < g.frame; });
    FreeRetiredGenerations();
}

void fsr::SetDepthSRV(ID3D11ShaderResourceView* srv)
{
    std::lock_guard lock(g_mutex);
    g.depthSRV = srv;
}

void fsr::RequestInputDump(int frames)
{
    std::lock_guard lock(g_mutex);
    g.dumpFrames = frames;
}

fsr::Status fsr::GetStatus()
{
    std::lock_guard lock(g_mutex);
    Status s;
    s.active = g.generation && !g.failed;
    s.failed = g.failed;
    s.provider = g.generation ? g.generation->provider : std::string();
    s.renderWidth = g.lastRenderW;
    s.renderHeight = g.lastRenderH;
    s.outputWidth = g.lastOutputW;
    s.outputHeight = g.lastOutputH;
    s.gpuMs = g.gpuMsSmoothed;
    s.gpuMsPeak = std::ranges::max(g.gpuMsRecent);
    s.upscaling = g.lastOutputW > g.lastRenderW;
    s.resetsTotal = g.resetsTotal;
    s.versions = g.versionNames;
    return s;
}

fsr::Stats fsr::TakeStats()
{
    std::lock_guard lock(g_mutex);
    Stats s{ g.frames, g.resets, g.gpuSamples ? g.gpuMsSum / g.gpuSamples : 0.0, g.gpuMsMax };
    g.frames = g.resets = g.gpuSamples = 0;
    g.gpuMsSum = g.gpuMsMax = 0.0;
    return s;
}
