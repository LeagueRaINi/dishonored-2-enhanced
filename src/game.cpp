#include "game.h"

#include "cbuffers.h"
#include "config.h"
#include "hooks.h"
#include "overlay.h"

#include <MinHook.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>

namespace
{
    constexpr DWORD kExeTimestamp = 0x67A50CDE;  // Dishonored2.exe, Feb 6 2025 build

    namespace addr
    {
        // Console variables: int value at +0x28, float value at +0x2C.
        constexpr uintptr_t kCvarRsEnable                = 0x14397BEC0;
        constexpr uintptr_t kCvarRsForceFractionX        = 0x14397BFC0;
        constexpr uintptr_t kCvarRsForceFractionY        = 0x14397C040;
        // Frame limiter: sub_1403591B0 spin-waits to 30 * com_timeScaler fps unless com_disableFrameSpinning is set.
        constexpr uintptr_t kCvarComTimeScaler           = 0x1425B1E00;
        constexpr uintptr_t kCvarComDisableFrameSpinning = 0x1425B1E80;
        // Anti-aliasing: FSR replaces the TAA pass, so TAA must stay on (r_skipTemporalAA 0, r_AAMode -1 = no FXAA).
        constexpr uintptr_t kCvarSkipTemporalAA          = 0x14397BBD0;
        constexpr uintptr_t kCvarAAMode                  = 0x14397BB50;
        // What the graphics menu shows for anti-aliasing: 0 off, 1/2 FXAA low/high, 3 TAA.
        constexpr uintptr_t kCvarAAPreset                = 0x143982EF0;

        // Graphics menu "apply" functions for the settings the mod controls (blocked while it controls them):
        // anti-aliasing (writes r_skipTemporalAA + r_AAMode) and adaptive resolution (item 25; third argument
        // false = write rs_enable / rs_VSyncBased / rs_FPSTargeted, true = read for display).
        constexpr uintptr_t kMenuApplyAntiAliasing  = 0x14050FA10;
        constexpr uintptr_t kMenuAdaptiveResolution = 0x140512AC0;
        // Graphics menu item display: sub_140512DD0(menu, item) refreshes one item, sub_14050F670(menu) enables or
        // greys out items 26/27 depending on item 25. Items are found by id in a {name, id} table.
        constexpr uintptr_t kMenuRefreshItem        = 0x140512DD0;
        constexpr uintptr_t kMenuRefreshAdaptive    = 0x14050F670;
        constexpr uintptr_t kMenuUpdate             = 0x140511410;  // vtable slot 5, every frame while open
        constexpr uintptr_t kMenuItemTable          = 0x1432CDCE8;  // -> +16: {const char* name, int64 id}[]

        constexpr uintptr_t kGameWindow = 0x143137F40;  // HWND

        // sub_140064E00 queues the boot videos unless its skip flag is set ("test sil, sil; jnz"); making the jnz
        // unconditional skips them. (Deleting the .bk2 files instead hangs the game on a black screen.)
        constexpr uintptr_t kIntroVideoBranch = 0x140064F08;

        // RenderFrame (sub_140361BC0) is called once per frame from sub_1403619D0, before ResScale_ApplyCvars runs,
        // so cvar writes in the frame tick apply to the same frame.
        constexpr uintptr_t kCallRenderFrame = 0x140361B92;
        constexpr uintptr_t kRenderFrame     = 0x140361BC0;

        // idMouseWin32 vtable slot 8 = per-frame poll sub_1400E2930: reads the buffered DirectInput mouse at +0x178.
        constexpr uintptr_t kMouseVtablePoll = 0x141AE0B38;
        constexpr uintptr_t kMousePoll       = 0x1400E2930;
        // The poll processes GetDeviceData's events only for S_OK ("test eax, eax; jz"); any other result,
        // including the success code DI_BUFFEROVERFLOW, takes the reacquire path and drops the frame's movement.
        // jz -> jns keeps recovery for real failures (negative HRESULTs) only.
        constexpr uintptr_t kMousePollResultCheck = 0x1400E2996;
        // Init (sub_1400E2800) sets DIPROP_BUFFERSIZE to 256 events (~128 ms of a 1000 Hz mouse); a frame hitch
        // overflows it. 1024 keeps every event; the poll still reads up to 256 per frame (fixed stack buffer).
        constexpr uintptr_t kMouseBufferSize = 0x1400E28BF;  // mov dword ptr [rsp+30h], 100h

        // Render jobs (frame graph). Both read their view through job+0x70.
        constexpr uintptr_t kPostFxTaa = 0x140327270;  // PostFX_TemporalAA_and_Post: TAA, motion blur ... tonemap
        constexpr uintptr_t kFinalPass = 0x140323E70;  // in-world UI + RENDER_UPSCALE

        // idRenderView_BuildJitteredProjection: "and eax, 7" selects one of 8 entries of the D3D 8x MSAA table,
        // "lea rax, [table]" loads it. The entries are divided by the UNSCALED viewport size (idRenderView_Setup
        // runs before the resolution scale), so in render pixels the offsets shrink by the render scale.
        constexpr uintptr_t kJitterIndexMask = 0x14037443B;  // 83 E0 07        and eax, 7
        constexpr uintptr_t kJitterTableLea  = 0x140374441;  // 48 8D 05 rel32  lea rax, [rip + rel32]
        constexpr uintptr_t kJitterTable8    = 0x141B35F10;

        // The three cvar setters (console "name value", idCVar::Set, config/command-line batches) allow a change
        // only if the developer flag dword_1432CD1C8 is set or the cvar has flag 0x10; otherwise they print
        // "%s CANNOT be set in retail!". Each check is "cmp [flag], 0; jnz <allowed>": jnz -> jmp unlocks it.
        // (The flag itself is read elsewhere too, so it is left alone.)
        constexpr uintptr_t kCvarRetailChecks[] = { 0x140101015, 0x140102465, 0x1401027B7 };

        // Game clock sub_140059990 simulates in fixed 120 Hz tics: tics = round(dt * 120), at most 12 per frame,
        // game dt = tics * 1/120. Above 120 fps some frames get no tic and the camera stops on them. The 120.0
        // is a shared constant (also used for angle wrapping), so its load is redirected to a mod-owned value.
        constexpr uintptr_t kTicRateLoad   = 0x140059E0D;  // F3 44 0F 10 15 rel32  movss xmm10, [flt 120.0]
        constexpr uintptr_t kTicRate120    = 0x141AC9C28;
        constexpr uintptr_t kMaxTicsImm    = 0x140059B60;  // BD 0C 00 00 00        mov ebp, 12
        constexpr uintptr_t kTicLength     = 0x142280A54;  // float 1/120, read only by the clock
    }

    // Engine structure offsets.
    namespace layout
    {
        // Graphics menu: items at *(menu + 0x78) + 160 + 288 * index; +248 = greyed out (not selectable).
        constexpr size_t kMenuItems        = 0x78;
        constexpr size_t kMenuItemsBase    = 160;
        constexpr size_t kMenuItemStride   = 288;
        constexpr size_t kMenuItemOption   = 152;  // selected option index
        constexpr size_t kMenuItemDisabled = 248;
        constexpr size_t kMenuPushItemSlot = 120 / 8;  // menu vtable: push item(id) to the UI
        constexpr size_t kMenuWidgets      = 0x90;     // UI widget per item id (null until the menu page is built)
        constexpr int    kItemAntiAliasing       = 15;
        constexpr int    kItemAdaptiveResolution = 25;
        constexpr int    kItemTargetFps          = 26;
        constexpr int    kItemMinResolution      = 27;

        constexpr size_t kJobView      = 0x70;   // render job: idRenderView*
        constexpr size_t kJobEngineCtx = 0x108;  // render job: engine context, ID3D11DeviceContext* at +0

        constexpr size_t kMouseDevice = 0x178;   // idMouseWin32: IDirectInputDevice8*

        // idRenderView. Scaled/unscaled idScreenRect pairs lie 24 bytes apart.
        constexpr size_t kViewFlagsSubview    = 14900;  // bit 0: portal/subview, the engine skips TAA
        constexpr size_t kViewFlagsTaa        = 14903;  // bit 2: TAA disabled (cvar)
        constexpr size_t kViewPortalViewport  = 30544;
        constexpr size_t kViewPortalScissor   = 30592;
        constexpr size_t kViewRenderSize      = 30792;  // m_renderWidth, m_renderHeight
        constexpr size_t kViewViewport        = 30816;  // m_viewport (scaled)
        constexpr size_t kViewUnscaled        = 30840;  // m_unscaledViewport
        constexpr size_t kViewScissor         = 30864;
        constexpr size_t kViewResolutionScale = 30912;  // m_resolutionScale, m_previousResolutionScale (idVec2 x2)
        constexpr size_t kRectSize            = 24;     // idScreenRect: x1, y1, x2, y2, zmin, zmax
        constexpr size_t kViewCopySize        = 0xC000; // covers every known field (m_justTeleported @ 0xBCFC)
    }

    constexpr int   kRenderSizeAlign = 8;   // forced render sizes are multiples of this (see ApplyRenderScale)
    constexpr int   kJitterPhases    = 32;
    constexpr DWORD kDInputGetDeviceData = 10;  // IDirectInputDevice8 vtable index
    constexpr DWORD kDInputObjectSize    = 24;  // sizeof(DIDEVICEOBJECTDATA) on x64

    uintptr_t g_base = 0;  // set only for the validated build

    template <typename T>
    T* At(uintptr_t idbVa)
    {
        return reinterpret_cast<T*>(g_base + (idbVa - kIdbImageBase));
    }

    // ---- Code patching ------------------------------------------------------------------------------------------

    void WriteProtected(void* dst, const void* src, size_t size)
    {
        DWORD old;
        VirtualProtect(dst, size, PAGE_EXECUTE_READWRITE, &old);
        memcpy(dst, src, size);
        VirtualProtect(dst, size, old, &old);
        FlushInstructionCache(GetCurrentProcess(), dst, size);
    }

    bool PatchBytes(uintptr_t va, std::span<const uint8_t> expected, std::span<const uint8_t> patched, const char* what)
    {
        auto* code = At<uint8_t>(va);
        if (memcmp(code, expected.data(), expected.size()) != 0)
        {
            Log("{}: unexpected code bytes, not patched", what);
            return false;
        }
        WriteProtected(code, patched.data(), patched.size());
        return true;
    }

    // One executable page within rel32 reach of the exe, for call stubs and the jitter table.
    class NearPage
    {
    public:
        void* Allocate(size_t size)
        {
            if (!m_page)
                m_page = Reserve(At<uint8_t>(kIdbImageBase));
            if (!m_page || m_used + size > kPageSize)
                return nullptr;
            void* block = m_page + m_used;
            m_used += (size + 15) & ~size_t{ 15 };
            return block;
        }

    private:
        static constexpr size_t kPageSize = 4096;

        static uint8_t* Reserve(uint8_t* nearTo)
        {
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            uintptr_t granularity = si.dwAllocationGranularity;
            uintptr_t target = reinterpret_cast<uintptr_t>(nearTo);
            for (uintptr_t a = (target - 0x70000000) & ~(granularity - 1); a < target; a += granularity)
                if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), kPageSize, MEM_RESERVE | MEM_COMMIT,
                                           PAGE_EXECUTE_READWRITE))
                    return static_cast<uint8_t*>(p);
            return nullptr;
        }

        uint8_t* m_page = nullptr;
        size_t   m_used = 0;
    };
    NearPage g_nearPage;

    int32_t Rel32(const uint8_t* nextInstruction, const void* target)
    {
        return static_cast<int32_t>(static_cast<const uint8_t*>(target) - nextInstruction);
    }

    // Redirects the E8 rel32 call at callVa through an absolute-jump stub to hook.
    bool HookCall(uintptr_t callVa, uintptr_t expectedTarget, void* hook, const char* what)
    {
        auto* call = At<uint8_t>(callVa);
        const uint8_t* target = call + 5 + *reinterpret_cast<const int32_t*>(call + 1);
        if (call[0] != 0xE8 || target != At<uint8_t>(expectedTarget))
        {
            Log("{}: unexpected call site bytes, not hooked", what);
            return false;
        }
        auto* stub = static_cast<uint8_t*>(g_nearPage.Allocate(14));
        if (!stub)
            return false;
        const uint8_t jmp[6] = { 0xFF, 0x25, 0, 0, 0, 0 };  // jmp [rip + 0]
        memcpy(stub, jmp, sizeof(jmp));
        memcpy(stub + 6, &hook, sizeof(hook));

        int32_t rel = Rel32(call + 5, stub);
        WriteProtected(call + 1, &rel, sizeof(rel));
        Log("{} hooked", what);
        return true;
    }

    // Replaces a vtable entry that must currently hold `expected`.
    bool HookVtableSlot(uintptr_t slotVa, uintptr_t expected, void* hook, void** original, const char* what)
    {
        void** slot = At<void*>(slotVa);
        if (*slot != At<void>(expected))
        {
            Log("{}: unexpected vtable entry, not hooked", what);
            return false;
        }
        *original = *slot;
        DWORD old;
        VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old);
        *slot = hook;
        VirtualProtect(slot, sizeof(void*), old, &old);
        Log("{} hooked", what);
        return true;
    }

    // ---- Console variables --------------------------------------------------------------------------------------

    struct Cvar
    {
        uintptr_t va;

        int   Int() const { return *At<int>(va + 0x28); }
        float Float() const { return *At<float>(va + 0x2C); }
        void  Set(int i, float f) const
        {
            *At<int>(va + 0x28)   = i;
            *At<float>(va + 0x2C) = f;
        }
        void Set(float f) const { Set(static_cast<int>(f), f); }
    };

    constexpr Cvar kRsEnable{ addr::kCvarRsEnable };
    constexpr Cvar kRsForceFractionX{ addr::kCvarRsForceFractionX };
    constexpr Cvar kRsForceFractionY{ addr::kCvarRsForceFractionY };
    constexpr Cvar kTimeScaler{ addr::kCvarComTimeScaler };
    constexpr Cvar kDisableFrameSpinning{ addr::kCvarComDisableFrameSpinning };
    constexpr Cvar kSkipTemporalAA{ addr::kCvarSkipTemporalAA };
    constexpr Cvar kAAMode{ addr::kCvarAAMode };
    constexpr Cvar kAAPreset{ addr::kCvarAAPreset };

    // A cvar the mod takes over: the game's value is remembered when the mod starts forcing its own (every frame,
    // menus and config loads rewrite cvars) and put back once the mod no longer needs it.
    class CvarOverride
    {
    public:
        explicit constexpr CvarOverride(const Cvar& cvar) : m_cvar(cvar) {}

        void Force(int i, float f)
        {
            if (!m_active)
            {
                m_savedInt = m_cvar.Int();
                m_savedFloat = m_cvar.Float();
                m_active = true;
            }
            m_cvar.Set(i, f);
        }
        void Force(float f) { Force(static_cast<int>(f), f); }

        void Release()
        {
            if (m_active)
                m_cvar.Set(m_savedInt, m_savedFloat);
            m_active = false;
        }

        bool Active() const { return m_active; }

    private:
        const Cvar& m_cvar;
        bool        m_active = false;
        int         m_savedInt = 0;
        float       m_savedFloat = 0.0f;
    };

    CvarOverride g_rsEnable{ kRsEnable };
    CvarOverride g_rsFractionX{ kRsForceFractionX };
    CvarOverride g_rsFractionY{ kRsForceFractionY };
    CvarOverride g_frameSpinning{ kDisableFrameSpinning };
    CvarOverride g_timeScaler{ kTimeScaler };
    CvarOverride g_skipTemporalAA{ kSkipTemporalAA };
    CvarOverride g_aaMode{ kAAMode };
    CvarOverride g_aaPreset{ kAAPreset };

    // The graphics menu would otherwise change these behind the mod's back; its own display still reads the
    // (forced) cvars, so it shows what is actually in effect.
    using PFN_MenuApply = int64_t(__fastcall*)(void*, int64_t, int64_t);
    PFN_MenuApply o_MenuApplyAntiAliasing;
    PFN_MenuApply o_MenuAdaptiveResolution;

    // Greys out the menu items whose cvars the mod currently controls. The flag only reaches the UI through the
    // menu's item refresh (vtable +120, as the game's own sub_14050F670 does after setting it), so each locked item
    // is pushed again. Adaptive resolution shows "Off": the mod renders at a fixed scale, nothing adapts.
    //
    // After a refresh every locked item is pushed (force). The per-frame update only pushes an item whose flag was
    // reset or whose UI widget is new: opening the menu builds the widgets without going through a refresh.
    thread_local bool t_lockingMenu = false;
    const void*       g_lockedWidgets[32] = {};  // widget each locked item was last pushed to, by item id

    void LockMenuItems(void* menu, bool force)
    {
        const void* tableObject = *At<const void*>(addr::kMenuItemTable);
        const auto* table = tableObject ? FieldAt<const uint8_t*>(tableObject, 16) : nullptr;
        auto* items = menu ? FieldAt<uint8_t*>(menu, layout::kMenuItems) : nullptr;
        if (!table || !items || t_lockingMenu)
            return;
        struct Entry
        {
            const char* name;
            int64_t     id;
        };
        auto findItem = [&](int id) -> uint8_t* {
            const auto* entries = reinterpret_cast<const Entry*>(table);
            for (int index = 0; entries[index].name && entries[index].name[0]; ++index)
                if (entries[index].id == id)
                    return items + layout::kMenuItemsBase + layout::kMenuItemStride * index;
            return nullptr;
        };
        using PFN_PushItem = void(__fastcall*)(void*, int64_t);
        auto push = (*reinterpret_cast<PFN_PushItem**>(menu))[layout::kMenuPushItemSlot];

        const void* const* widgets = FieldAt<const void* const*>(menu, layout::kMenuWidgets);

        t_lockingMenu = true;
        auto lock = [&](int id, int shownOption) {
            uint8_t* item = findItem(id);
            const void* widget = widgets ? widgets[id] : nullptr;
            if (!item || (!force && item[layout::kMenuItemDisabled] && widget == g_lockedWidgets[id]))
                return;
            if (!force)
                LOG_LIMITED(8, "menu: item {} locked on update (widget {})", id, widget);
            g_lockedWidgets[id] = widget;
            item[layout::kMenuItemDisabled] = 1;
            if (shownOption >= 0)
                FieldAt<int>(item, layout::kMenuItemOption) = shownOption;
            push(menu, id);
        };
        if (g_skipTemporalAA.Active())
            lock(layout::kItemAntiAliasing, -1);  // its value comes from the forced r_AntiAliasingPreset
        if (g_rsEnable.Active())
        {
            lock(layout::kItemAdaptiveResolution, 0);
            lock(layout::kItemTargetFps, -1);
            lock(layout::kItemMinResolution, -1);
        }
        t_lockingMenu = false;
    }

    using PFN_MenuRefreshItem = int64_t(__fastcall*)(void*, int);
    using PFN_MenuRefreshAdaptive = int64_t(__fastcall*)(void*);
    PFN_MenuRefreshItem     o_MenuRefreshItem;
    PFN_MenuRefreshAdaptive o_MenuRefreshAdaptive;

    using PFN_MenuUpdate = void(__fastcall*)(void*);
    PFN_MenuUpdate o_MenuUpdate;

    int64_t __fastcall Hook_MenuRefreshItem(void* menu, int item)
    {
        int64_t result = o_MenuRefreshItem(menu, item);
        LockMenuItems(menu, true);
        return result;
    }

    int64_t __fastcall Hook_MenuRefreshAdaptive(void* menu)
    {
        int64_t result = o_MenuRefreshAdaptive(menu);
        LockMenuItems(menu, true);
        return result;
    }

    void __fastcall Hook_MenuUpdate(void* menu)
    {
        o_MenuUpdate(menu);
        LockMenuItems(menu, false);
    }

    int64_t __fastcall Hook_MenuApplyAntiAliasing(void* menu, int64_t option, int64_t flags)
    {
        if (g_skipTemporalAA.Active())
        {
            LOG_LIMITED(3, "menu: anti-aliasing is locked to TAA while FSR is on");
            return 0;
        }
        return o_MenuApplyAntiAliasing(menu, option, flags);
    }

    int64_t __fastcall Hook_MenuAdaptiveResolution(void* menu, int64_t item, int64_t readOnly)
    {
        if (!(readOnly & 0xFF) && g_rsEnable.Active())
        {
            LOG_LIMITED(3, "menu: adaptive resolution is locked while the mod sets the render scale");
            return 0;
        }
        return o_MenuAdaptiveResolution(menu, item, readOnly);
    }

    // ---- Frame tick and mouse -----------------------------------------------------------------------------------

    using PFN_RenderFrame = void(__fastcall*)(void*, int, uint8_t, uint8_t);
    using PFN_MousePoll   = void(__fastcall*)(void*);
    PFN_RenderFrame o_RenderFrame;
    PFN_MousePoll   o_MousePoll;

    void __fastcall Hook_RenderFrame(void* self, int a2, uint8_t a3, uint8_t a4)
    {
        hooks::OnFrame();
        o_RenderFrame(self, a2, a3, a4);
    }

    // While the overlay is open the mouse's buffered events are flushed before the engine reads them, so the poll
    // writes a zero look delta and no clicks.
    void __fastcall Hook_MousePoll(void* self)
    {
        if (overlay::IsVisible())
        {
            if (void* device = FieldAt<void*>(self, layout::kMouseDevice))
            {
                using PFN_GetDeviceData = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, void*, DWORD*, DWORD);
                auto getDeviceData = reinterpret_cast<PFN_GetDeviceData>((*static_cast<void***>(device))[kDInputGetDeviceData]);
                DWORD count = INFINITE;
                getDeviceData(device, kDInputObjectSize, nullptr, &count, 0);  // null buffer = flush
            }
        }
        o_MousePoll(self);
    }

    // ---- Display-resolution post chain (upscale mode) -----------------------------------------------------------
    // FSR replaces the TAA dispatch and writes a display-size image. The rest of PostFX_TemporalAA_and_Post and the
    // final pass then run at display resolution: each job gets a private copy of its view whose viewports/scale are
    // switched to the unscaled values (render jobs recording in parallel keep the real view), and cbuffers.cpp
    // swaps the PerViewCB for one with resolutionscale = 1.
    //
    // Both jobs decide from the same view flags whether TAA runs, so they always agree on the resolution. When
    // FSR can't run in such a frame, fsr.cpp records a plain upsample instead, keeping the image display-size.

    using PFN_RenderJob = char(__fastcall*)(uint8_t* job);
    PFN_RenderJob o_PostFxTaa;
    PFN_RenderJob o_FinalPass;

    std::atomic<bool> g_postHooksActive{ false };
    std::atomic<bool> g_postDisabled{ false };     // set when an invariant failed; upscale mode turns off
    std::atomic<int>  g_displayWidth{ 0 }, g_displayHeight{ 0 };  // engine's unscaled viewport

    bool RunsTaa(const uint8_t* view)
    {
        return (FieldAt<uint8_t>(view, layout::kViewFlagsSubview) & 0x01) == 0 &&
               (FieldAt<uint8_t>(view, layout::kViewFlagsTaa) & 0x04) == 0;
    }

    void RememberDisplaySize(const uint8_t* view)
    {
        const auto* rect = &FieldAt<int>(view, layout::kViewUnscaled);
        int width = rect[2] - rect[0] + 1, height = rect[3] - rect[1] + 1;
        if (width > 0 && height > 0)
        {
            g_displayWidth = width;
            g_displayHeight = height;
        }
    }

    void DisablePostUpscale(const char* reason)
    {
        if (!g_postDisabled.exchange(true))
            Log("post chain: {}, upscale mode disabled", reason);
    }

    ID3D11DeviceContext* JobContext(uint8_t* job)
    {
        auto* engineCtx = FieldAt<uint8_t*>(job, layout::kJobEngineCtx);
        return engineCtx ? FieldAt<ID3D11DeviceContext*>(engineCtx, 0) : nullptr;
    }

    // Number of readable bytes at p, up to max (the view's allocation size is unknown).
    size_t ReadableBytes(const void* p, size_t max)
    {
        const auto* begin = static_cast<const uint8_t*>(p);
        size_t total = 0;
        while (total < max)
        {
            MEMORY_BASIC_INFORMATION mbi;
            if (!VirtualQuery(begin + total, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
                (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
                break;
            total = static_cast<const uint8_t*>(mbi.BaseAddress) + mbi.RegionSize - begin;
        }
        return std::min(total, max);
    }

    // Swaps a job's view for a per-thread shallow copy for the job's duration.
    class ViewCopy
    {
    public:
        explicit ViewCopy(uint8_t* job) : m_slot(&FieldAt<uint8_t*>(job, layout::kJobView)), m_original(*m_slot)
        {
            static thread_local std::unique_ptr<uint8_t, decltype(&_aligned_free)> buffer{
                static_cast<uint8_t*>(_aligned_malloc(layout::kViewCopySize, 64)), &_aligned_free };
            size_t size = m_original ? ReadableBytes(m_original, layout::kViewCopySize) : 0;
            if (!buffer || size <= layout::kViewResolutionScale + 16)
                return;
            memcpy(buffer.get(), m_original, size);
            m_copy = buffer.get();
            *m_slot = m_copy;
        }

        ~ViewCopy()
        {
            if (m_copy)
                *m_slot = m_original;
        }

        ViewCopy(const ViewCopy&) = delete;
        ViewCopy& operator=(const ViewCopy&) = delete;

        bool Valid() const { return m_copy != nullptr; }

        // Everything after FSR works on the display-size image: unscaled viewports/scissors, scale 1.
        void SwitchToDisplayResolution()
        {
            using namespace layout;
            for (size_t scaled : { kViewPortalViewport, kViewPortalScissor, kViewViewport, kViewScissor })
                memcpy(m_copy + scaled, m_copy + scaled + kRectSize, kRectSize);
            const auto* rect = &FieldAt<int>(m_copy, kViewViewport);
            auto* size = &FieldAt<int>(m_copy, kViewRenderSize);
            size[0] = rect[2] - rect[0] + 1;
            size[1] = rect[3] - rect[1] + 1;
            std::fill_n(&FieldAt<float>(m_copy, kViewResolutionScale), 4, 1.0f);
        }

    private:
        uint8_t** m_slot;
        uint8_t*  m_original;
        uint8_t*  m_copy = nullptr;
    };

    // PostFX job that will switch to display resolution at its TAA dispatch.
    struct PostJob
    {
        ViewCopy view;
        bool     switched = false;
    };
    thread_local PostJob* t_postJob = nullptr;

    bool WantsDisplayResolution(uint8_t* job)
    {
        const uint8_t* view = FieldAt<uint8_t*>(job, layout::kJobView);
        return view && hooks::IsUpscaleFrame() && !g_postDisabled && RunsTaa(view);
    }

    char __fastcall Hook_PostFxTaa(uint8_t* job)
    {
        if (const uint8_t* view = FieldAt<uint8_t*>(job, layout::kJobView))
            RememberDisplaySize(view);
        if (!WantsDisplayResolution(job))
            return o_PostFxTaa(job);

        PostJob post{ ViewCopy(job) };
        if (!post.view.Valid())
        {
            DisablePostUpscale("view copy failed");
            return o_PostFxTaa(job);
        }
        t_postJob = &post;
        char result = o_PostFxTaa(job);
        t_postJob = nullptr;
        cb::EndDisplayResolution();
        if (!post.switched)
            DisablePostUpscale("TAA pass was expected but not seen");  // the final pass is at display res this frame
        return result;
    }

    char __fastcall Hook_FinalPass(uint8_t* job)
    {
        if (!WantsDisplayResolution(job))
            return o_FinalPass(job);

        ViewCopy view(job);
        ID3D11DeviceContext* ctx = JobContext(job);
        if (!view.Valid() || !ctx)
        {
            DisablePostUpscale("final pass view copy failed");
            return o_FinalPass(job);
        }
        view.SwitchToDisplayResolution();
        cb::BeginDisplayResolution(ctx);
        char result = o_FinalPass(job);
        cb::EndDisplayResolution();
        return result;
    }

    // ---- Render scale and jitter --------------------------------------------------------------------------------

    float g_appliedScale[2] = { 1.0f, 1.0f };  // fractions written to rs_forceFraction*

    float Halton(int index, int base)
    {
        float fraction = 1.0f, result = 0.0f;
        for (int i = index; i > 0; i /= base)
        {
            fraction /= static_cast<float>(base);
            result += fraction * static_cast<float>(i % base);
        }
        return result;
    }

    // Replaces the engine's 8-phase table with a 32-phase Halton(2,3) table spanning one render pixel.
    class JitterPatch
    {
    public:
        void Apply(bool halton)
        {
            if (!Prepare())
                return;
            if (halton)
            {
                if (m_scale[0] != g_appliedScale[0] || m_scale[1] != g_appliedScale[1])
                    Fill(g_appliedScale[0], g_appliedScale[1]);
                if (!m_active)
                    Install(m_table, kJitterPhases);
            }
            else if (m_active)
            {
                Install(At<float>(addr::kJitterTable8), 8);
            }
        }

    private:
        bool Prepare()
        {
            if (m_table)
                return true;
            if (m_failed)
                return false;
            const uint8_t* mask = At<uint8_t>(addr::kJitterIndexMask);
            const uint8_t* lea = At<uint8_t>(addr::kJitterTableLea);
            static constexpr uint8_t kMask[] = { 0x83, 0xE0, 0x07 };
            static constexpr uint8_t kLea[]  = { 0x48, 0x8D, 0x05 };
            const uint8_t* table = lea + 7 + *reinterpret_cast<const int32_t*>(lea + 3);
            m_failed = memcmp(mask, kMask, sizeof(kMask)) != 0 || memcmp(lea, kLea, sizeof(kLea)) != 0 ||
                       table != At<uint8_t>(addr::kJitterTable8);
            if (!m_failed)
                m_table = static_cast<float*>(g_nearPage.Allocate(kJitterPhases * 2 * sizeof(float)));
            if (!m_table)
            {
                m_failed = true;
                Log("jitter: unexpected code bytes, keeping the engine pattern");
            }
            return m_table != nullptr;
        }

        // Entries are divided by the display size; dividing them by the render scale makes them span a render pixel.
        void Fill(float scaleX, float scaleY)
        {
            for (int i = 0; i < kJitterPhases; ++i)
            {
                m_table[i * 2 + 0] = (Halton(i + 1, 2) - 0.5f) / scaleX;
                m_table[i * 2 + 1] = (Halton(i + 1, 3) - 0.5f) / scaleY;
            }
            m_scale[0] = scaleX;
            m_scale[1] = scaleY;
            Log("jitter: {}-phase Halton table for scale ({:.4f}, {:.4f})", kJitterPhases, scaleX, scaleY);
        }

        void Install(const float* table, int phases)
        {
            auto* mask = At<uint8_t>(addr::kJitterIndexMask);
            auto* lea = At<uint8_t>(addr::kJitterTableLea);
            uint8_t maskBits = static_cast<uint8_t>(phases - 1);
            int32_t rel = Rel32(lea + 7, table);
            WriteProtected(mask + 2, &maskBits, 1);
            WriteProtected(lea + 3, &rel, sizeof(rel));
            m_active = table == m_table;
            Log("jitter: {} pattern", m_active ? "32-phase Halton" : "engine 8-phase");
        }

        float* m_table = nullptr;
        float  m_scale[2] = {};
        bool   m_active = false;
        bool   m_failed = false;
    };
    JitterPatch g_jitter;

    // Retail cvar lock: the jnz of each check becomes a jmp while unlocked.
    class CvarUnlockPatch
    {
    public:
        void Apply(bool unlock)
        {
            if (unlock == m_unlocked || m_failed)
                return;
            constexpr uint8_t kJnzShort = 0x75, kJmpShort = 0xEB;
            for (uintptr_t va : addr::kCvarRetailChecks)
            {
                uint8_t expected = unlock ? kJnzShort : kJmpShort;
                if (*At<uint8_t>(va) != expected)
                {
                    m_failed = true;
                    Log("cvar unlock: unexpected code bytes, not patched");
                    return;
                }
            }
            for (uintptr_t va : addr::kCvarRetailChecks)
            {
                uint8_t opcode = unlock ? kJmpShort : kJnzShort;
                WriteProtected(At<uint8_t>(va), &opcode, 1);
            }
            m_unlocked = unlock;
            Log("cvar unlock: {}", unlock ? "all cvars settable" : "retail lock restored");
        }

    private:
        bool m_unlocked = false;
        bool m_failed = false;
    };
    CvarUnlockPatch g_cvarUnlock;

    // Experimental: runs the game simulation at `rate` tics per second (120 = engine default), so a frame cap above
    // 120 still advances the game on every frame. Catch-up stays at 100 ms of tics per frame.
    class GameTickPatch
    {
    public:
        void Apply(int rate)
        {
            if (rate == m_appliedRate || !Prepare())
                return;
            int32_t maxTics = std::max(12, (rate + 9) / 10);
            float length = 1.0f / static_cast<float>(rate);
            *m_rate = static_cast<float>(rate);
            WriteProtected(At<float>(addr::kTicLength), &length, sizeof(length));
            WriteProtected(At<uint8_t>(addr::kMaxTicsImm) + 1, &maxTics, sizeof(maxTics));
            m_appliedRate = rate;
            Log("game tick: {} Hz (up to {} tics per frame){}", rate, maxTics, rate == 120 ? "" : " [experimental]");
        }

    private:
        bool Prepare()
        {
            if (m_rate)
                return true;
            if (m_failed)
                return false;
            auto* load = At<uint8_t>(addr::kTicRateLoad);
            const uint8_t* constant = load + 9 + *reinterpret_cast<const int32_t*>(load + 5);
            static constexpr uint8_t kLoad[]    = { 0xF3, 0x44, 0x0F, 0x10, 0x15 };
            static constexpr uint8_t kMaxTics[] = { 0xBD, 0x0C, 0x00, 0x00, 0x00 };
            m_failed = memcmp(load, kLoad, sizeof(kLoad)) != 0 || constant != At<uint8_t>(addr::kTicRate120) ||
                       memcmp(At<uint8_t>(addr::kMaxTicsImm), kMaxTics, sizeof(kMaxTics)) != 0 ||
                       *At<float>(addr::kTicLength) != 1.0f / 120.0f;
            if (!m_failed)
                m_rate = static_cast<float*>(g_nearPage.Allocate(sizeof(float)));
            if (!m_rate)
            {
                m_failed = true;
                Log("game tick: unexpected code bytes, keeping 120 Hz");
                return false;
            }
            *m_rate = 120.0f;
            int32_t rel = Rel32(load + 9, m_rate);
            WriteProtected(load + 5, &rel, sizeof(rel));
            return true;
        }

        float* m_rate = nullptr;
        int    m_appliedRate = 120;
        bool   m_failed = false;
    };
    GameTickPatch g_gameTick;
}

void game::Init()
{
    auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->FileHeader.TimeDateStamp != kExeTimestamp)
    {
        Log("game: unknown exe build (timestamp {:#010x}), engine patches disabled", nt->FileHeader.TimeDateStamp);
        return;
    }
    g_base = base;
    Log("game: exe build ok, base {:#x}", base);

    if (config::Current()->skipIntro)
    {
        static constexpr uint8_t kJnz[] = { 0x0F, 0x85, 0xBC, 0x00, 0x00, 0x00 };  // jnz +0xBC
        static constexpr uint8_t kJmp[] = { 0xE9, 0xBD, 0x00, 0x00, 0x00, 0x90 };  // jmp +0xBD; nop
        if (PatchBytes(addr::kIntroVideoBranch, kJnz, kJmp, "intro skip"))
            Log("intro videos skipped");
    }

    o_RenderFrame = reinterpret_cast<PFN_RenderFrame>(At<void>(addr::kRenderFrame));
    HookCall(addr::kCallRenderFrame, addr::kRenderFrame, reinterpret_cast<void*>(&Hook_RenderFrame), "RenderFrame");
    HookVtableSlot(addr::kMouseVtablePoll, addr::kMousePoll, reinterpret_cast<void*>(&Hook_MousePoll),
                   reinterpret_cast<void**>(&o_MousePoll), "mouse poll");

    static constexpr uint8_t kJzNormal[]  = { 0x74, 0x59 };  // jz  loc_1400E29F1
    static constexpr uint8_t kJnsNormal[] = { 0x79, 0x59 };  // jns loc_1400E29F1
    if (PatchBytes(addr::kMousePollResultCheck, kJzNormal, kJnsNormal, "mouse overflow handling"))
        Log("mouse: buffer overflows keep their events");
    static constexpr uint8_t kBuffer256[]  = { 0xC7, 0x44, 0x24, 0x30, 0x00, 0x01, 0x00, 0x00 };
    static constexpr uint8_t kBuffer1024[] = { 0xC7, 0x44, 0x24, 0x30, 0x00, 0x04, 0x00, 0x00 };
    if (PatchBytes(addr::kMouseBufferSize, kBuffer256, kBuffer1024, "mouse buffer size"))
        Log("mouse: DirectInput buffer 1024 events");
}

bool game::IsSupported()
{
    return g_base != 0;
}

uintptr_t game::Address(uintptr_t idbVa)
{
    return g_base ? g_base + (idbVa - kIdbImageBase) : 0;
}

void game::InstallRenderHooks()
{
    static bool attempted = false;
    if (!g_base || attempted)
        return;
    attempted = true;

    struct Target { uintptr_t va; void* hook; void** original; };
    const std::array targets = {
        Target{ addr::kPostFxTaa, reinterpret_cast<void*>(&Hook_PostFxTaa), reinterpret_cast<void**>(&o_PostFxTaa) },
        Target{ addr::kFinalPass, reinterpret_cast<void*>(&Hook_FinalPass), reinterpret_cast<void**>(&o_FinalPass) },
    };

    MH_STATUS status = MH_OK;
    size_t created = 0;
    for (; created < targets.size() && status == MH_OK; ++created)
        status = MH_CreateHook(At<void>(targets[created].va), targets[created].hook, targets[created].original);
    if (status == MH_OK)
    {
        for (const Target& t : targets)
            MH_QueueEnableHook(At<void>(t.va));
        status = MH_ApplyQueued();
    }
    if (status != MH_OK)
    {
        for (size_t i = 0; i < created; ++i)
            MH_RemoveHook(At<void>(targets[i].va));
        Log("post chain hooks: {} - upscale mode unavailable", MH_StatusToString(status));
        return;
    }
    g_postHooksActive = true;
    Log("post chain hooks installed");

    hooks::InstallInline(At<void>(addr::kMenuApplyAntiAliasing), &Hook_MenuApplyAntiAliasing, &o_MenuApplyAntiAliasing,
                         "menu anti-aliasing");
    hooks::InstallInline(At<void>(addr::kMenuAdaptiveResolution), &Hook_MenuAdaptiveResolution,
                         &o_MenuAdaptiveResolution, "menu adaptive resolution");
    hooks::InstallInline(At<void>(addr::kMenuRefreshItem), &Hook_MenuRefreshItem, &o_MenuRefreshItem, "menu item refresh");
    hooks::InstallInline(At<void>(addr::kMenuRefreshAdaptive), &Hook_MenuRefreshAdaptive, &o_MenuRefreshAdaptive,
                         "menu adaptive refresh");
    hooks::InstallInline(At<void>(addr::kMenuUpdate), &Hook_MenuUpdate, &o_MenuUpdate, "menu update");
}

bool game::PostUpscaleAvailable()
{
    return g_postHooksActive && !g_postDisabled;
}

void game::ApplyRenderScale(bool enable, float scale)
{
    if (!g_base)
        return;
    if (!enable)
    {
        g_rsEnable.Release();
        g_rsFractionX.Release();
        g_rsFractionY.Release();
        g_appliedScale[0] = g_appliedScale[1] = 1.0f;
        return;
    }

    // ScaleScreenRect rounds the viewport to whole pixels, but shaders scale UVs by the raw fraction
    // (cb_resolutionscale), so display * fraction must be a whole number. Multiples of kRenderSizeAlign also keep
    // the half/quarter/eighth resolution passes integral (odd sizes such as 2023x847 looked blurry in testing).
    LONG displayW = g_displayWidth, displayH = g_displayHeight;
    if (displayW <= 0 || displayH <= 0)
    {
        RECT client = {};  // until the first frame reaches the post chain
        if (HWND hwnd = *At<HWND>(addr::kGameWindow); hwnd && GetClientRect(hwnd, &client))
        {
            displayW = client.right;
            displayH = client.bottom;
        }
    }
    float fx = scale, fy = scale;
    if (displayW > 0 && displayH > 0)
    {
        auto snap = [scale](LONG display) {
            float pixels = std::round(display * scale / kRenderSizeAlign) * kRenderSizeAlign;
            return std::clamp(pixels, static_cast<float>(kRenderSizeAlign), static_cast<float>(display)) / display;
        };
        fx = snap(displayW);
        fy = snap(displayH);
    }
    if (fx != g_appliedScale[0] || fy != g_appliedScale[1])
        Log("render scale {:.3f} -> {}x{} of {}x{}", scale, std::lround(displayW * fx), std::lround(displayH * fy),
            displayW, displayH);

    g_rsEnable.Force(3, 3.0f);  // bit mask: scale X and Y
    g_rsFractionX.Force(0, fx);
    g_rsFractionY.Force(0, fy);
    g_appliedScale[0] = fx;
    g_appliedScale[1] = fy;
}

void game::ApplyFrameLimit(int fpsLimit)
{
    if (!g_base)
        return;
    if (fpsLimit < 0)
    {
        g_frameSpinning.Release();
        g_timeScaler.Release();
    }
    else if (fpsLimit == 0)
    {
        g_frameSpinning.Force(1, 1.0f);
        g_timeScaler.Release();
    }
    else
    {
        g_frameSpinning.Force(0, 0.0f);
        g_timeScaler.Force(std::clamp(fpsLimit / 30.0f, 0.1f, 10.0f));  // engine cap = 30 fps * com_timeScaler
    }
    // A cap above 120 raises the game tick to match; unlocked keeps 120 Hz (the UI warns about it).
    g_gameTick.Apply(std::max(fpsLimit, kEngineTickRate));
}

void game::ApplyAntiAliasing(bool forceTaa)
{
    if (!g_base)
        return;
    if (forceTaa)
    {
        g_skipTemporalAA.Force(0, 0.0f);
        g_aaMode.Force(-1, -1.0f);
        g_aaPreset.Force(3, 3.0f);  // the menu shows "Temporal AA"
    }
    else
    {
        g_skipTemporalAA.Release();
        g_aaMode.Release();
        g_aaPreset.Release();
    }
}

void game::ApplyJitterPattern(bool halton)
{
    if (g_base)
        g_jitter.Apply(halton);
}

void game::ApplyCvarUnlock(bool unlock)
{
    if (g_base)
        g_cvarUnlock.Apply(unlock);
}

bool game::IsDisplayResolutionPost()
{
    return t_postJob && !t_postJob->switched;
}

void game::SwitchPostToDisplayResolution(ID3D11DeviceContext* ctx)
{
    if (!IsDisplayResolutionPost())
        return;
    t_postJob->view.SwitchToDisplayResolution();
    t_postJob->switched = true;
    cb::BeginDisplayResolution(ctx);
}
