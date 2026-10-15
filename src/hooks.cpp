#include "hooks.h"

#include "capture.h"
#include "cbuffers.h"
#include "config.h"
#include "console.h"
#include "counter.h"
#include "fsr.h"
#include "game.h"
#include "hdr.h"
#include "hud.h"
#include "material.h"
#include "overlay.h"
#include "rescale.h"
#include "shader_info.h"
#include "shader_patch.h"

#include <d3d11_3.h>
#include <MinHook.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    // ---- Device vtable hooks ------------------------------------------------------------------------------------
    // Devices share static vtables; a few slots per method cover the ID3D11Device1..3 interfaces.

    namespace device_index
    {
        constexpr int kCreateVertexShader     = 12;
        constexpr int kCreatePixelShader      = 15;
        constexpr int kCreateComputeShader    = 18;
        constexpr int kCreateDeferredContext  = 27;
        constexpr int kCreateDeferredContext1 = 44;  // ID3D11Device1
        constexpr int kCreateDeferredContext2 = 51;  // ID3D11Device2
        constexpr int kCreateDeferredContext3 = 62;  // ID3D11Device3
    }

    class VtableHook
    {
    public:
        explicit constexpr VtableHook(int index) : m_index(index) {}

        void Install(void** vtable, void* hook)
        {
            if (vtable[m_index] == hook)
                return;  // already patched through another object sharing the vtable
            auto slot = std::ranges::find(m_slots, nullptr, &Slot::vtable);
            if (slot == m_slots.end())
            {
                Log("hooks: out of vtable slots for index {}", m_index);
                return;
            }
            *slot = { vtable, vtable[m_index] };
            DWORD old;
            VirtualProtect(&vtable[m_index], sizeof(void*), PAGE_EXECUTE_READWRITE, &old);
            vtable[m_index] = hook;
            VirtualProtect(&vtable[m_index], sizeof(void*), old, &old);
            Log("hooks: vtable [{}] {} -> original {}", m_index, DescribeAddress(vtable), DescribeAddress(slot->original));
        }

        template <typename Fn>
        Fn Original(void* self) const
        {
            void** vtable = *static_cast<void***>(self);
            for (const Slot& s : m_slots)
                if (s.vtable == vtable)
                    return reinterpret_cast<Fn>(s.original);
            return reinterpret_cast<Fn>(m_slots[0].original);
        }

    private:
        struct Slot
        {
            void** vtable = nullptr;
            void*  original = nullptr;
        };
        int                  m_index;
        std::array<Slot, 8> m_slots{};
    };

    VtableHook g_createVertexShader{ device_index::kCreateVertexShader };
    VtableHook g_createPixelShader{ device_index::kCreatePixelShader };
    VtableHook g_createComputeShader{ device_index::kCreateComputeShader };
    VtableHook g_createDeferred[] = {
        VtableHook{ device_index::kCreateDeferredContext },
        VtableHook{ device_index::kCreateDeferredContext1 },
        VtableHook{ device_index::kCreateDeferredContext2 },
        VtableHook{ device_index::kCreateDeferredContext3 },
    };

    // ---- Context method hooks -----------------------------------------------------------------------------------
    // Deferred contexts carry per-object vtable copies that the runtime rewrites, so context methods are hooked
    // inline at their implementation. Immediate and deferred contexts use different implementations, hence a few
    // thunks per method.

    namespace context_index
    {
        constexpr int kPSSetShader          = 9;
        constexpr int kDrawIndexed          = 12;
        constexpr int kDraw                 = 13;
        constexpr int kMap                  = 14;
        constexpr int kUnmap                = 15;
        constexpr int kDrawIndexedInstanced = 20;
        constexpr int kDrawInstanced        = 21;
        constexpr int kOMSetRenderTargets   = 33;
        constexpr int kOMSetTargetsAndUAVs  = 34;
        constexpr int kDispatch             = 41;
        constexpr int kUpdateSubresource    = 48;
        constexpr int kExecuteCommandList   = 58;
        constexpr int kFinishCommandList    = 114;
    }

    constexpr int kInlineSlots = 4;

    class InlineHookSet
    {
    public:
        InlineHookSet(const char* name, std::array<void*, kInlineSlots> thunks, std::array<void**, kInlineSlots> originals)
            : m_name(name), m_thunks(thunks), m_originals(originals) {}

        void Install(void* target)
        {
            if (std::ranges::contains(m_targets, target) || std::ranges::contains(m_thunks, target))
                return;  // hooked already, or the vtable points at one of our thunks
            auto slot = std::ranges::find(m_targets, nullptr);
            if (slot == m_targets.end())
            {
                Log("hooks: no inline slot left for {}", m_name);
                return;
            }
            size_t i = slot - m_targets.begin();
            MH_STATUS status = MH_CreateHook(target, m_thunks[i], m_originals[i]);
            if (status == MH_OK)
                status = MH_EnableHook(target);
            Log("hooks: {}[{}] at {}: {}", m_name, i, DescribeAddress(target), MH_StatusToString(status));
            if (status == MH_OK)
                *slot = target;
        }

    private:
        const char*                       m_name;
        std::array<void*, kInlineSlots>   m_thunks;
        std::array<void**, kInlineSlots>  m_originals;
        std::array<void*, kInlineSlots>   m_targets{};
    };

    // Builds a hook set from a thunk template Thunk<0> ... Thunk<kInlineSlots - 1>.
    template <template <int> class Thunk, int... I>
    InlineHookSet MakeHookSet(const char* name, std::integer_sequence<int, I...>)
    {
        return InlineHookSet(name, { reinterpret_cast<void*>(&Thunk<I>::Fn)... },
                             { reinterpret_cast<void**>(&Thunk<I>::original)... });
    }

    template <template <int> class Thunk>
    InlineHookSet MakeHookSet(const char* name)
    {
        return MakeHookSet<Thunk>(name, std::make_integer_sequence<int, kInlineSlots>{});
    }

    // ---- Frame state --------------------------------------------------------------------------------------------

    uint64_t             g_frame = 0;
    std::atomic<bool>    g_upscaleFrame{ false };
    std::atomic<bool>    g_scaleActive{ false };
    std::atomic<int64_t> g_patchTicks{ 0 };  // time spent patching shaders (startup report)
    std::atomic<int>     g_patchedShaders{ 0 };

    // Cvars the mod enforces: the Ultra+ preset (beyond the menu's Ultra) and the texture mip bias for upscaling.
    // Ultra+ only holds changes that can't cost quality elsewhere: longer sun cascades spread the fixed-size shadow
    // map over more ground (blurrier near shadows), bigger light shadows crowd the shared shadow atlas.
    console::Overrides QualityOverrides(const Config& cfg)
    {
        console::Overrides overrides;
        if (cfg.ultraPlus)
        {
            overrides = {
                { "r_shadowParallelResolutionDividerLevel", "0" },
                { "r_shadowSpotAndOmniLowResolutionDividerEnabled", "0" },
                { "r_shadowCullDistance", "60" },  // model shadows drawn further (25)
                { "image_anisotropy", "16" },
                { "r_SSSBlurQuality", "0" },       // skin subsurface scattering: full 92 taps
            };
        }
        // AMD's recommendation for upscaling: textures sampled as sharp as at display resolution.
        if (cfg.fsrMipBias && cfg.fsrEnabled && cfg.fsrUpscale && cfg.forceScale && cfg.renderScale < 0.999f)
            overrides.emplace_back("image_lodbias", std::format("{:.2f}", std::log2(cfg.renderScale) - 1.0f));
        return overrides;
    }

    // Once, at the first frame: how long startup took and how much of it the shader patching cost.
    void LogStartup()
    {
        FILETIME creation, exit, kernel, user, now;
        GetSystemTimeAsFileTime(&now);
        if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
            return;
        auto ticks = [](FILETIME t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        Log("startup: first frame {:.1f} s after launch; {} shaders patched in {:.0f} ms",
            double(ticks(now) - ticks(creation)) / 1e7, g_patchedShaders.load(),
            1000.0 * double(g_patchTicks.load()) / double(frequency.QuadPart));
    }

    // ---- TAA routing --------------------------------------------------------------------------------------------

    bool DecodeFrameConstants(ID3D11DeviceContext* ctx, fsr::FrameConstants& fc)
    {
        ComArray<ID3D11Buffer, 3> buffers;
        ctx->CSGetConstantBuffers(0, buffers.size(), buffers.data());
        cb::Snapshot view, instance;
        if (!cb::Read(buffers[1], view) || view.size != cb::kPerViewSize || !cb::Read(buffers[2], instance) ||
            instance.size != cb::kPerInstanceSize)
            return false;

        using namespace cb;
        float projection[16];
        memcpy(projection, view.bytes.data() + perview::kProjection, sizeof(projection));
        fc.renderWidth   = static_cast<uint32_t>(instance.Get<float>(perinstance::kDitherViewport + 8));
        fc.renderHeight  = static_cast<uint32_t>(instance.Get<float>(perinstance::kDitherViewport + 12));
        fc.jitterX       = view.Get<float>(perview::kJitterVectors);
        fc.jitterY       = view.Get<float>(perview::kJitterVectors + 4);
        fc.cameraNear    = projection[14] > 0.0f ? projection[14] : 0.01f;
        fc.fovY          = 2.0f * std::atan(1.0f / projection[5]);
        fc.projX         = projection[0];
        fc.projY         = projection[5];
        fc.inverseViewProjection  = view.Get<Float4x4>(perview::kInverseViewProjection);
        fc.previousViewProjection = view.Get<Float4x4>(perview::kPreviousViewProjection);
        fc.frameNumber   = view.Get<float>(perview::kFrameNumber);
        fc.exposureIndex = instance.Get<uint32_t>(perinstance::kExposureIndex + 4);
        fc.compressedHdr = view.Get<uint32_t>(perview::kCompressedHdr);
        fc.viewWhite     = instance.Get<float>(perinstance::kViewWhite);
        fc.evBias        = instance.Get<float>(perinstance::kEvBias);
        return true;
    }

    // The FSR output (u1) and the expanded motion vectors (t1) already hold display-resolution content.
    void BeginRescaleFrame(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc)
    {
        ComArray<ID3D11UnorderedAccessView, 2> uavs;
        ComArray<ID3D11ShaderResourceView, 2> srvs;
        ctx->CSGetUnorderedAccessViews(0, uavs.size(), uavs.data());
        ctx->CSGetShaderResources(0, srvs.size(), srvs.data());
        ComPtr<ID3D11Resource> result, motion;
        ComPtr<ID3D11Texture2D> output;
        if (!uavs[1] || !srvs[1])
            return;
        uavs[1]->GetResource(&result);
        srvs[1]->GetResource(&motion);
        if (FAILED(result.As(&output)))
            return;
        D3D11_TEXTURE2D_DESC desc;
        output->GetDesc(&desc);
        rescale::BeginFrame(fc.renderWidth, fc.renderHeight, desc.Width, desc.Height, { result.Get(), motion.Get() });
    }

    // The engine TAA dispatch, at record time. Returns true if it was replaced.
    bool RouteTaa(ID3D11DeviceContext* ctx, const fsr::FrameConstants& fc)
    {
        std::shared_ptr<const Config> cfg = config::Current();
        if (!cfg->fsrEnabled)
            return false;
        if (!cfg->fsrUpscale)
            return fsr::RecordTaa(ctx, fc, fsr::TaaMode::NativeAA);
        if (!game::IsDisplayResolutionPost())
            return fsr::RecordTaa(ctx, fc, fsr::TaaMode::Prepare);  // not an upscale frame (yet): get ready
        if (!fsr::RecordTaa(ctx, fc, fsr::TaaMode::Upscale))
            return false;  // game.cpp sees the missing switch and turns upscale mode off
        BeginRescaleFrame(ctx, fc);
        game::SwitchPostToDisplayResolution(ctx);
        return true;
    }

    // Returns true if the dispatch was replaced.
    bool OnDispatch(ID3D11DeviceContext* ctx)
    {
        if (cb::IsDisplayResolution(ctx))
        {
            cb::BindTwins(ctx, { cb::Stage::CS });
            rescale::BeforeDispatch(ctx);
        }

        ComPtr<ID3D11ComputeShader> shader;
        ctx->CSGetShader(&shader, nullptr, nullptr);
        ShaderInfo info;
        if (!ReadShaderInfo(shader.Get(), info))
            return false;
        if (info.isDepthDownsample)
        {
            ComPtr<ID3D11ShaderResourceView> depth;
            ctx->CSGetShaderResources(0, 1, &depth);
            fsr::SetDepthSRV(depth.Get());
            return false;
        }
        if (!info.isTaa || ctx->GetType() != D3D11_DEVICE_CONTEXT_DEFERRED)
            return false;
        fsr::FrameConstants fc;
        if (!DecodeFrameConstants(ctx, fc))
            return false;
        ComPtr<ID3D11ShaderResourceView> color;
        ctx->CSGetShaderResources(2, 1, &color);  // ro_viewcolormap
        if (color)
            rescale::RepairEdge(ctx, color.Get(), fc.renderWidth, fc.renderHeight);
        return RouteTaa(ctx, fc);
    }

    // Returns true if the draw must be skipped. count: vertices or indices; fullscreenCandidate: at most 6 of them,
    // one instance.
    bool OnDraw(ID3D11DeviceContext* ctx, UINT count, bool fullscreenCandidate)
    {
        if (cb::IsDisplayResolution(ctx))
        {
            cb::BindTwins(ctx, { cb::Stage::VS, cb::Stage::PS });
            rescale::BeforeDraw(ctx);
        }
        if (capture::Active())
            capture::LogDraw(ctx, count);
        material::OnDraw(ctx);
        hud::OnDraw(ctx);
        return hdr::OnDraw(ctx, fullscreenCandidate);
    }

    // ---- Context thunks -----------------------------------------------------------------------------------------

    // Some runtimes implement one draw method with another (DrawIndexed -> DrawIndexedInstanced): only the outermost
    // call of a draw is the game's, nested ones go straight to the original.
    thread_local int t_drawDepth = 0;

    template <int N, typename... Args>
    struct DrawThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, Args...);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, Args... args)
        {
            if (t_drawDepth > 0)
            {
                original(self, args...);
                return;
            }
            struct Depth
            {
                Depth() { ++t_drawDepth; }
                ~Depth() { --t_drawDepth; }
            } depth;

            // A full-screen pass: one triangle or quad (Draw, DrawIndexed) and, if instanced, a single instance.
            auto params = std::tuple(args...);
            UINT count = std::get<0>(params);
            bool fullscreen = count <= 6;
            if constexpr (sizeof...(Args) >= 4)  // DrawInstanced / DrawIndexedInstanced: (count, instances, ...)
                fullscreen = fullscreen && std::get<1>(params) == 1;
            if (OnDraw(self, count, fullscreen))
                return;
            original(self, args...);
            if (hdr::NeedsAfterDraw())
                hdr::AfterDraw(self);
            if (material::NeedsAfterDraw())
                material::AfterDraw(self);
            if (hud::NeedsAfterDraw())
                hud::AfterDraw(self);
        }
    };
    template <int N> using DrawIndexedThunk          = DrawThunk<N, UINT, UINT, INT>;
    template <int N> using DrawPlainThunk            = DrawThunk<N, UINT, UINT>;
    template <int N> using DrawIndexedInstancedThunk = DrawThunk<N, UINT, UINT, UINT, INT, UINT>;
    template <int N> using DrawInstancedThunk        = DrawThunk<N, UINT, UINT, UINT, UINT>;

    template <int N>
    struct DispatchThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, UINT, UINT, UINT);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, UINT x, UINT y, UINT z)
        {
            bool replaced = OnDispatch(self);
            if (capture::Active())
                capture::LogDispatch(self, x, y, z);
            if (!replaced)
                original(self, x, y, z);
        }
    };

    template <int N>
    struct PSSetShaderThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, ID3D11PixelShader*, ID3D11ClassInstance* const*, UINT);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, ID3D11PixelShader* shader,
                                         ID3D11ClassInstance* const* instances, UINT count)
        {
            original(self, shader, instances, count);
            material::OnSetPixelShader(self, shader);
        }
    };

    template <int N>
    struct OMSetRenderTargetsThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                        ID3D11DepthStencilView*);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, UINT count, ID3D11RenderTargetView* const* rtvs,
                                         ID3D11DepthStencilView* dsv)
        {
            original(self, count, rtvs, dsv);
            hdr::OnSetRenderTargets(self, count, rtvs);
            material::OnSetRenderTargets(self);
        }
    };

    template <int N>
    struct OMSetRenderTargetsAndUAVsThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*,
                                                        ID3D11DepthStencilView*, UINT, UINT,
                                                        ID3D11UnorderedAccessView* const*, const UINT*);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, UINT count, ID3D11RenderTargetView* const* rtvs,
                                         ID3D11DepthStencilView* dsv, UINT uavStart, UINT uavCount,
                                         ID3D11UnorderedAccessView* const* uavs, const UINT* counts)
        {
            original(self, count, rtvs, dsv, uavStart, uavCount, uavs, counts);
            if (count != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
            {
                hdr::OnSetRenderTargets(self, count, rtvs);
                material::OnSetRenderTargets(self);
            }
        }
    };

    template <int N>
    struct MapThunk
    {
        static inline HRESULT(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                                           D3D11_MAPPED_SUBRESOURCE*);
        static HRESULT STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, ID3D11Resource* resource, UINT subresource,
                                            D3D11_MAP type, UINT flags, D3D11_MAPPED_SUBRESOURCE* mapped)
        {
            HRESULT hr = original(self, resource, subresource, type, flags, mapped);
            if (SUCCEEDED(hr))
                cb::OnMap(self, resource, type, mapped);
            return hr;
        }
    };

    template <int N>
    struct UnmapThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, ID3D11Resource* resource, UINT subresource)
        {
            cb::OnUnmap(self, resource);  // before the original: the mapped memory is still valid
            original(self, resource, subresource);
        }
    };

    template <int N>
    struct UpdateSubresourceThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*,
                                                        const void*, UINT, UINT);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, ID3D11Resource* resource, UINT subresource,
                                         const D3D11_BOX* box, const void* data, UINT rowPitch, UINT depthPitch)
        {
            cb::OnUpdateSubresource(resource, box, data);
            original(self, resource, subresource, box, data, rowPitch, depthPitch);
        }
    };

    template <int N>
    struct ExecuteCommandListThunk
    {
        static inline void(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, ID3D11CommandList*, BOOL);
        static void STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, ID3D11CommandList* list, BOOL restoreState)
        {
            if (self->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE)
                fsr::OnExecuteCommandList(self, list, restoreState);  // part A + upscaler, if this list carries a job
            original(self, list, restoreState);
        }
    };

    template <int N>
    struct FinishCommandListThunk
    {
        static inline HRESULT(STDMETHODCALLTYPE* original)(ID3D11DeviceContext*, BOOL, ID3D11CommandList**);
        static HRESULT STDMETHODCALLTYPE Fn(ID3D11DeviceContext* self, BOOL restoreState, ID3D11CommandList** list)
        {
            HRESULT hr = original(self, restoreState, list);
            cb::OnFinishCommandList(self);
            hdr::OnFinishCommandList(self);
            material::OnFinishCommandList(self);
            if (SUCCEEDED(hr) && list)
                fsr::OnFinishCommandList(self, *list);
            return hr;
        }
    };

    struct ContextHook
    {
        int           index;
        InlineHookSet set;
    };
    ContextHook g_contextHooks[] = {
        { context_index::kDrawIndexed, MakeHookSet<DrawIndexedThunk>("DrawIndexed") },
        { context_index::kDraw, MakeHookSet<DrawPlainThunk>("Draw") },
        { context_index::kDrawIndexedInstanced, MakeHookSet<DrawIndexedInstancedThunk>("DrawIndexedInstanced") },
        { context_index::kDrawInstanced, MakeHookSet<DrawInstancedThunk>("DrawInstanced") },
        { context_index::kDispatch, MakeHookSet<DispatchThunk>("Dispatch") },
        { context_index::kPSSetShader, MakeHookSet<PSSetShaderThunk>("PSSetShader") },
        { context_index::kOMSetRenderTargets, MakeHookSet<OMSetRenderTargetsThunk>("OMSetRenderTargets") },
        { context_index::kOMSetTargetsAndUAVs, MakeHookSet<OMSetRenderTargetsAndUAVsThunk>("OMSetRenderTargetsAndUAVs") },
        { context_index::kMap, MakeHookSet<MapThunk>("Map") },
        { context_index::kUnmap, MakeHookSet<UnmapThunk>("Unmap") },
        { context_index::kUpdateSubresource, MakeHookSet<UpdateSubresourceThunk>("UpdateSubresource") },
        { context_index::kExecuteCommandList, MakeHookSet<ExecuteCommandListThunk>("ExecuteCommandList") },
        { context_index::kFinishCommandList, MakeHookSet<FinishCommandListThunk>("FinishCommandList") },
    };

    void HookContext(ID3D11DeviceContext* ctx)
    {
        auto hookVtable = [](void** vtable) {
            for (ContextHook& hook : g_contextHooks)
                hook.set.Install(vtable[hook.index]);
        };
        hookVtable(*reinterpret_cast<void***>(ctx));
        // The game may call through a newer interface pointer with its own vtable.
        ComPtr<ID3D11DeviceContext1> ctx1;
        if (SUCCEEDED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1))))
            hookVtable(*reinterpret_cast<void***>(ctx1.Get()));
    }

    // ---- Device thunks ------------------------------------------------------------------------------------------

    using PFN_CreateDeferredContext = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, UINT, ID3D11DeviceContext**);

    // [Debug] DumpShaders: <mod folder>\captures\shaders\<stage>_<hash>.dxbc, for offline disassembly.
    template <typename Shader>
    void DumpShader(std::span<const std::byte> bytecode)
    {
        constexpr std::wstring_view stage = std::is_same_v<Shader, ID3D11VertexShader> ? L"vs"
                                          : std::is_same_v<Shader, ID3D11PixelShader>  ? L"ps"
                                                                                       : L"cs";
        std::filesystem::path path =
            ModFile(L"captures/shaders", std::format(L"{}_{:016x}.dxbc", stage, Fnv1a64(bytecode)));
        std::error_code error;
        if (std::filesystem::exists(path, error))
            return;
        FILE* file = nullptr;
        if (_wfopen_s(&file, path.c_str(), L"wb") == 0 && file)
        {
            fwrite(bytecode.data(), 1, bytecode.size(), file);
            fclose(file);
        }
    }

    // CreateVertexShader / CreatePixelShader / CreateComputeShader share their shape: every shader is tagged with
    // its RDEF binding names (pass identification, F8 capture).
    template <typename Shader, VtableHook& Hook>
    HRESULT STDMETHODCALLTYPE Hook_CreateShader(ID3D11Device* self, const void* bytecode, SIZE_T size,
                                                ID3D11ClassLinkage* linkage, Shader** shader)
    {
        using PFN = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*, Shader**);
        auto original = Hook.template Original<PFN>(self);
        std::span code(static_cast<const std::byte*>(bytecode), size);

        // Forward-lighting pixel shaders also write material data (SSGI albedo, SSR roughness), if the patch applies.
        if constexpr (std::is_same_v<Shader, ID3D11PixelShader>)
        {
            if (bytecode && shader && config::Current()->materialCapture)
            {
                LARGE_INTEGER start, end;
                QueryPerformanceCounter(&start);
                shader_patch::MaterialPatch patch = shader_patch::AddMaterialOutputs(code);
                QueryPerformanceCounter(&end);
                g_patchTicks += end.QuadPart - start.QuadPart;
                if (!patch.bytecode.empty() &&
                    SUCCEEDED(original(self, patch.bytecode.data(), patch.bytecode.size(), linkage, shader)) && *shader)
                {
                    ++g_patchedShaders;
                    TagShader(*shader, code, true, patch.hasSurface);
                    if (config::Current()->dumpShaders)
                        DumpShader<Shader>(code);
                    LOG_LIMITED(1, "material: forward shaders patched for material data");
                    return S_OK;
                }
            }
        }

        HRESULT hr = original(self, bytecode, size, linkage, shader);
        if (SUCCEEDED(hr) && shader && *shader)
        {
            TagShader(*shader, code);
            if (config::Current()->dumpShaders)
                DumpShader<Shader>(code);
        }
        return hr;
    }

    // One thunk per CreateDeferredContext variant (they share a signature but live at different vtable slots).
    template <int Variant>
    HRESULT STDMETHODCALLTYPE Hook_CreateDeferredContext(ID3D11Device* self, UINT flags, ID3D11DeviceContext** ctx)
    {
        HRESULT hr = g_createDeferred[Variant].Original<PFN_CreateDeferredContext>(self)(self, flags, ctx);
        if (SUCCEEDED(hr) && ctx && *ctx)
            HookContext(*ctx);
        return hr;
    }

    template <typename Interface>
    void HookDeviceInterface(ID3D11Device* device, VtableHook& hook, void* thunk)
    {
        ComPtr<Interface> iface;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&iface))))
            hook.Install(*reinterpret_cast<void***>(iface.Get()), thunk);
    }

    // ---- Frame pacing statistics --------------------------------------------------------------------------------

    class FramePacing
    {
    public:
        void Tick()
        {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            if (!m_freq.QuadPart)
            {
                QueryPerformanceFrequency(&m_freq);
                m_last = m_windowStart = now;
                return;
            }
            m_frameMs.push_back(static_cast<float>((now.QuadPart - m_last.QuadPart) * 1000.0 / m_freq.QuadPart));
            m_last = now;
            if (now.QuadPart - m_windowStart.QuadPart >= kWindowSeconds * m_freq.QuadPart)
            {
                Report();
                m_windowStart = now;
                m_frameMs.clear();
            }
        }

    private:
        static constexpr int kWindowSeconds = 10;

        void Report() const
        {
            if (m_frameMs.empty())
                return;
            std::vector<float> sorted = m_frameMs;
            std::ranges::sort(sorted);
            double sum = 0.0, swing = 0.0;
            for (size_t i = 0; i < m_frameMs.size(); ++i)
            {
                sum += m_frameMs[i];
                if (i)
                    swing += std::abs(m_frameMs[i] - m_frameMs[i - 1]);
            }
            double average = sum / m_frameMs.size();
            float low = sorted[static_cast<size_t>(sorted.size() * 0.99f)];
            fsr::Stats fsr = fsr::TakeStats();
            Log("perf: {:.1f} fps ({:.2f} ms, 1% low {:.2f} ms, worst {:.2f} ms, frame-to-frame {:.2f} ms) | FSR {}/{} frames, "
                "{} resets, GPU {:.2f} ms (max {:.2f}) | scale {}",
                1000.0 / average, average, low, sorted.back(), m_frameMs.size() > 1 ? swing / (m_frameMs.size() - 1) : 0.0,
                fsr.frames, m_frameMs.size(), fsr.resets, fsr.gpuMsAvg, fsr.gpuMsMax, g_scaleActive ? "reduced" : "native");
        }

        LARGE_INTEGER      m_freq = {}, m_last = {}, m_windowStart = {};
        std::vector<float> m_frameMs;
    };
    FramePacing g_pacing;

    // ---- Hotkeys ------------------------------------------------------------------------------------------------

    void HandleHotkeys(const Config& cfg)
    {
        static KeyEdge captureKey, scaleKey, fsrKey, debugViewKey, benchmarkKey;
        if (captureKey.Pressed(cfg.keyCapture))
        {
            capture::Begin(g_frame);
            fsr::RequestInputDump(4);  // also save four consecutive frames of upscaler inputs
        }
        if (scaleKey.Pressed(cfg.keyToggleScale))
            config::Edit([](Config& c) { c.forceScale = !c.forceScale; });
        if (fsrKey.Pressed(cfg.keyToggleFsr))
            config::Edit([](Config& c) { c.fsrEnabled = !c.fsrEnabled; });
        if (debugViewKey.Pressed(cfg.keyDebugView))
            config::Edit([](Config& c) { c.fsrDebugView = !c.fsrDebugView; });
        if (benchmarkKey.Pressed(cfg.keyBenchmark))
            counter::ToggleBenchmark();
    }
}

bool hooks::InstallInline(void* target, void* detour, void** original, std::string_view what)
{
    MH_STATUS status = MH_CreateHook(target, detour, original);
    if (status == MH_OK)
        status = MH_EnableHook(target);
    Log("{} hook: {}", what, MH_StatusToString(status));
    return status == MH_OK;
}

void hooks::InstallDeviceHooks(ID3D11Device* device, ID3D11DeviceContext* immediate)
{
    static const bool minHookReady = MH_Initialize() == MH_OK;
    if (!minHookReady)
    {
        Log("hooks: MinHook initialization failed");
        return;
    }
    game::InstallRenderHooks();
    console::Install();

    void** vtable = *reinterpret_cast<void***>(device);
    g_createVertexShader.Install(vtable, reinterpret_cast<void*>(&Hook_CreateShader<ID3D11VertexShader, g_createVertexShader>));
    g_createPixelShader.Install(vtable, reinterpret_cast<void*>(&Hook_CreateShader<ID3D11PixelShader, g_createPixelShader>));
    g_createComputeShader.Install(vtable, reinterpret_cast<void*>(&Hook_CreateShader<ID3D11ComputeShader, g_createComputeShader>));
    g_createDeferred[0].Install(vtable, reinterpret_cast<void*>(&Hook_CreateDeferredContext<0>));
    HookDeviceInterface<ID3D11Device1>(device, g_createDeferred[1], reinterpret_cast<void*>(&Hook_CreateDeferredContext<1>));
    HookDeviceInterface<ID3D11Device2>(device, g_createDeferred[2], reinterpret_cast<void*>(&Hook_CreateDeferredContext<2>));
    HookDeviceInterface<ID3D11Device3>(device, g_createDeferred[3], reinterpret_cast<void*>(&Hook_CreateDeferredContext<3>));
    HookContext(immediate);
    overlay::Install(device);
    Log("hooks: device {} hooked", Ptr(device));
}

void hooks::OnFrame()
{
    // Engine frames can be several apart from our startup; the reduced resolution is applied once the game runs.
    constexpr uint64_t kScaleStartFrame = 600;

    ++g_frame;
    if (g_frame == 1)
        LogStartup();
    capture::End(g_frame);
    HandleHotkeys(*config::Current());
    overlay::PollToggleKey();
    config::Publish();
    std::shared_ptr<const Config> cfg = config::Current();

    fsr::OnFrame();
    material::OnFrame();
    hud::OnFrame();
    console::SetOverrides(QualityOverrides(*cfg));
    console::OnFrame();
    bool scaleActive = cfg->forceScale && g_frame >= kScaleStartFrame;
    if (scaleActive || g_scaleActive)
        game::ApplyRenderScale(scaleActive, cfg->renderScale);  // reasserted every frame: menus rewrite the cvars
    g_scaleActive = scaleActive;
    game::ApplyJitterPattern(cfg->haltonJitter);
    game::ApplyFrameLimit(cfg->fpsLimit);
    game::ApplyCvarUnlock(cfg->unlockCvars);
    game::ApplyAntiAliasing(cfg->fsrEnabled);

    // Decided once per frame so every job agrees (only settings or failures change it).
    g_upscaleFrame = cfg->fsrEnabled && cfg->fsrUpscale && game::PostUpscaleAvailable() && fsr::IsUpscaleReady();
    g_pacing.Tick();
}

bool hooks::IsUpscaleFrame()
{
    return g_upscaleFrame;
}
