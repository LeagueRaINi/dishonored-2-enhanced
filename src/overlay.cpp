// Present/ResizeBuffers are inline-hooked (MinHook) at their implementation, found through a throwaway swap chain;
// under DXVK this is DXVK's dxgi.dll, which the Steam overlay doesn't hook (it uses Vulkan).
//
// While the overlay is open the game gets no input: DirectInput reads come back empty (the mouse poll itself is
// drained in game.cpp), cursor re-centring is blocked and window input messages go to ImGui only.
#include "overlay.h"

#include "capture.h"
#include "config.h"
#include "console.h"
#include "counter.h"
#include "hdr.h"
#include "hooks.h"
#include "ui.h"

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include <dxgi1_2.h>
#include <intrin.h>
#include <MinHook.h>

#include <atomic>
#include <mutex>
#include <unordered_set>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void ImGui_ImplDX11_SetSrgbTarget(bool srgb);  // imgui_impl_dx11_precompiled.cpp

namespace
{
    constexpr int kSwapChainPresent       = 8;
    constexpr int kSwapChainResizeBuffers = 13;
    constexpr int kSwapChain1Present1     = 22;
    constexpr int kDInputAcquire          = 7;
    constexpr int kDInputUnacquire        = 8;
    constexpr int kDInputGetDeviceState   = 9;
    constexpr int kDInputGetDeviceData    = 10;

    using PFN_Present       = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using PFN_Present1      = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
    using PFN_ResizeBuffers = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
    using PFN_Acquire       = HRESULT(STDMETHODCALLTYPE*)(void*);
    using PFN_GetDeviceState = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, void*);
    using PFN_GetDeviceData = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, void*, DWORD*, DWORD);
    using PFN_SetCursorPos  = BOOL(WINAPI*)(int, int);
    using PFN_GetCursorPos  = BOOL(WINAPI*)(LPPOINT);
    using PFN_ClipCursor    = BOOL(WINAPI*)(const RECT*);

    PFN_Present       o_Present;
    PFN_Present1      o_Present1;
    PFN_ResizeBuffers o_ResizeBuffers;
    PFN_SetCursorPos  o_SetCursorPos;
    PFN_GetCursorPos  o_GetCursorPos;
    PFN_ClipCursor    o_ClipCursor;
    PFN_Acquire       o_Unacquire;  // not hooked; used to release the game's devices when the overlay opens

    std::atomic<bool> g_visible{ false };
    bool              g_imguiReady = false;
    HWND              g_hwnd = nullptr;
    WNDPROC           g_gameWndProc = nullptr;

    ComPtr<ID3D11Device>           g_device;
    ComPtr<ID3D11DeviceContext>    g_context;
    ComPtr<ID3D11RenderTargetView> g_backbufferRTV;
    ID3D11Resource*                g_backbuffer = nullptr;  // identity of the buffer g_backbufferRTV was made for

    // Mouse-look re-centres the cursor (SetCursorPos) and reads the offset (GetCursorPos). While the overlay is
    // open the game's GetCursorPos calls see the last position it set: zero movement.
    POINT     g_gameCursor = {};
    bool      g_haveGameCursor = false;
    uintptr_t g_exeBegin = 0, g_exeEnd = 0;

    // DirectInput devices the game reads from (unacquired when the overlay opens).
    std::mutex                g_inputMutex;
    std::unordered_set<void*> g_inputDevices;

    // ---- Rendering --------------------------------------------------------------------------------------------

    LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    int MonitorRefreshRate(HWND hwnd)
    {
        MONITORINFOEXW info = {};
        info.cbSize = sizeof(info);
        DEVMODEW mode = {};
        mode.dmSize = sizeof(mode);
        if (!GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &info) ||
            !EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode))
            return 60;
        Log("overlay: display refresh {} Hz", mode.dmDisplayFrequency);
        return static_cast<int>(mode.dmDisplayFrequency);
    }

    bool InitImGui(IDXGISwapChain* swapChain)
    {
        DXGI_SWAP_CHAIN_DESC desc;
        if (FAILED(swapChain->GetDesc(&desc)) || FAILED(swapChain->GetDevice(IID_PPV_ARGS(&g_device))))
            return false;
        g_device->GetImmediateContext(&g_context);
        g_hwnd = desc.OutputWindow;

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;  // settings live in the mod's ini
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ui::ApplyStyle(1.25f);
        ui::LoadFonts();
        counter::LoadFont();
        counter::SetRefreshRate(MonitorRefreshRate(g_hwnd));

        DXGI_FORMAT format = desc.BufferDesc.Format;
        ImGui_ImplDX11_SetSrgbTarget(format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
        if (!ImGui_ImplWin32_Init(g_hwnd) || !ImGui_ImplDX11_Init(g_device.Get(), g_context.Get()))
        {
            Log("overlay: ImGui initialization failed");
            return false;
        }
        g_gameWndProc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(g_hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&OverlayWndProc)));
        Log("overlay: ready ({}x{}, format {})", desc.BufferDesc.Width, desc.BufferDesc.Height, static_cast<int>(format));
        return true;
    }

    bool UpdateBackbufferView(IDXGISwapChain* swapChain)
    {
        ComPtr<ID3D11Resource> buffer;
        if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&buffer))))
            return false;
        if (buffer.Get() == g_backbuffer && g_backbufferRTV)
            return true;
        g_backbufferRTV.Reset();
        g_backbuffer = nullptr;
        // In HDR mode the game's back buffer is a typeless stand-in: ImGui draws through an sRGB view of it.
        D3D11_RENDER_TARGET_VIEW_DESC srgb = {};
        srgb.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        srgb.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (FAILED(g_device->CreateRenderTargetView(buffer.Get(), hdr::IsActive() ? &srgb : nullptr, &g_backbufferRTV)))
            return false;
        g_backbuffer = buffer.Get();
        capture::SetBackbuffer(g_backbuffer);
        return true;
    }

    // DXGI maximum frame latency of the game's device (Config::maxQueuedFrames, 0 = driver default).
    void ApplyQueuedFrames(int frames)
    {
        static int applied = -1;
        if (frames == applied || !g_device)
            return;
        ComPtr<IDXGIDevice1> dxgi;
        if (SUCCEEDED(g_device.As(&dxgi)) && SUCCEEDED(dxgi->SetMaximumFrameLatency(frames > 0 ? frames : 3)))
            Log("overlay: max queued frames {}", frames > 0 ? std::to_string(frames) : std::string("default"));
        applied = frames;
    }

    void RenderOverlay(IDXGISwapChain* swapChain)
    {
        if (!g_imguiReady)
        {
            static bool attempted = false;
            if (std::exchange(attempted, true) || !(g_imguiReady = InitImGui(swapChain)))
                return;
        }
        std::shared_ptr<const Config> cfg = config::Current();
        counter::OnPresent();
        ApplyQueuedFrames(cfg->maxQueuedFrames);
        if ((!g_visible && cfg->fpsCounter == 0) || !UpdateBackbufferView(swapChain))
            return;

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::GetIO().MouseDrawCursor = g_visible;  // the game hides the OS cursor
        counter::Draw();
        if (g_visible)
            ui::DrawSettings();
        ImGui::Render();

        ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> savedRTVs;
        ComPtr<ID3D11DepthStencilView> savedDSV;
        g_context->OMGetRenderTargets(savedRTVs.size(), savedRTVs.data(), &savedDSV);
        g_context->OMSetRenderTargets(1, g_backbufferRTV.GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());  // saves/restores the rest of the pipeline state
        g_context->OMSetRenderTargets(savedRTVs.size(), savedRTVs.data(), savedDSV.Get());
    }

    HRESULT STDMETHODCALLTYPE Hook_Present(IDXGISwapChain* self, UINT sync, UINT flags)
    {
        if (!(flags & DXGI_PRESENT_TEST))
        {
            RenderOverlay(self);
            hdr::Composite(self);
        }
        return o_Present(self, sync, flags);
    }

    HRESULT STDMETHODCALLTYPE Hook_Present1(IDXGISwapChain1* self, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* params)
    {
        if (!(flags & DXGI_PRESENT_TEST))
        {
            RenderOverlay(self);
            hdr::Composite(self);
        }
        return o_Present1(self, sync, flags, params);
    }

    HRESULT STDMETHODCALLTYPE Hook_ResizeBuffers(IDXGISwapChain* self, UINT count, UINT width, UINT height, DXGI_FORMAT format,
                                                 UINT flags)
    {
        g_backbufferRTV.Reset();  // the swap chain buffers can't be resized while we hold a view
        g_backbuffer = nullptr;
        HRESULT hr = o_ResizeBuffers(self, count, width, height, hdr::ResizeFormat(self, format), flags);
        hdr::AfterResize(self, hr);
        return hr;
    }

    // Finds the swap chain implementation through a throwaway swap chain and hooks it.
    void HookSwapChain(ID3D11Device* device)
    {
        WNDCLASSW wc = {};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"d2enhanced_dummy";
        RegisterClassW(&wc);
        HWND hwnd = CreateWindowW(wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW, 0, 0, 8, 8, nullptr, nullptr, wc.hInstance, nullptr);

        ComPtr<IDXGIDevice> dxgiDevice;
        ComPtr<IDXGIAdapter> adapter;
        ComPtr<IDXGIFactory> factory;
        if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice))) && SUCCEEDED(dxgiDevice->GetAdapter(&adapter)))
            adapter->GetParent(IID_PPV_ARGS(&factory));

        DXGI_SWAP_CHAIN_DESC desc = {};
        desc.BufferCount = 2;
        desc.BufferDesc = { 8, 8, {}, DXGI_FORMAT_R8G8B8A8_UNORM };
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.OutputWindow = hwnd;
        desc.SampleDesc.Count = 1;
        desc.Windowed = TRUE;
        desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        ComPtr<IDXGISwapChain> swapChain;
        HRESULT hr = factory ? factory->CreateSwapChain(device, &desc, &swapChain) : E_FAIL;
        if (SUCCEEDED(hr))
        {
            void** vtable = *reinterpret_cast<void***>(swapChain.Get());
            hooks::InstallInline(vtable[kSwapChainPresent], &Hook_Present, &o_Present, "overlay: Present");
            hooks::InstallInline(vtable[kSwapChainResizeBuffers], &Hook_ResizeBuffers, &o_ResizeBuffers,
                                 "overlay: ResizeBuffers");
            ComPtr<IDXGISwapChain1> swapChain1;
            if (SUCCEEDED(swapChain.As(&swapChain1)))
                hooks::InstallInline((*reinterpret_cast<void***>(swapChain1.Get()))[kSwapChain1Present1], &Hook_Present1,
                                     &o_Present1, "overlay: Present1");
            hdr::InstallDxgiHooks(factory.Get(), swapChain.Get());
        }
        else
        {
            Log("overlay: dummy swap chain failed {}, overlay unavailable", HrText(hr));
        }
        swapChain.Reset();
        DestroyWindow(hwnd);
    }

    // ---- Input --------------------------------------------------------------------------------------------------

    void RememberInputDevice(void* device)
    {
        std::lock_guard lock(g_inputMutex);
        g_inputDevices.insert(device);
    }

    template <int N>
    struct GetDeviceStateThunk
    {
        static inline PFN_GetDeviceState original;
        static HRESULT STDMETHODCALLTYPE Fn(void* self, DWORD size, void* data)
        {
            RememberInputDevice(self);
            HRESULT hr = original(self, size, data);
            if (g_visible && data)
                memset(data, 0, size);  // no keys, buttons or motion while the overlay is open
            return hr;
        }
    };

    template <int N>
    struct GetDeviceDataThunk
    {
        static inline PFN_GetDeviceData original;
        static HRESULT STDMETHODCALLTYPE Fn(void* self, DWORD objectSize, void* objects, DWORD* count, DWORD flags)
        {
            RememberInputDevice(self);
            HRESULT hr = original(self, objectSize, objects, count, flags);
            if (g_visible && count)
                *count = 0;  // drop buffered input events
            return hr;
        }
    };

    template <int N>
    struct AcquireThunk
    {
        static inline PFN_Acquire original;
        static HRESULT STDMETHODCALLTYPE Fn(void* self)
        {
            return g_visible ? DIERR_OTHERAPPHASPRIO : original(self);  // keep mouse/keyboard free for the overlay
        }
    };

    // Hooks one DirectInput device method for the wide and ANSI interfaces (often the same implementation).
    template <template <int> class Thunk>
    void HookDInputMethod(void* const (&targets)[2], std::string_view name)
    {
        if (targets[0])
            hooks::InstallInline(targets[0], &Thunk<0>::Fn, &Thunk<0>::original, name);
        if (targets[1] && targets[1] != targets[0])
            hooks::InstallInline(targets[1], &Thunk<1>::Fn, &Thunk<1>::original, name);
    }

    void HookDirectInput()
    {
        HMODULE dinput = LoadLibraryW(L"dinput8.dll");
        auto create = dinput ? reinterpret_cast<decltype(&DirectInput8Create)>(GetProcAddress(dinput, "DirectInput8Create")) : nullptr;
        if (!create)
            return;

        void* acquire[2] = {}, *state[2] = {}, *data[2] = {};
        const IID* interfaces[2] = { &IID_IDirectInput8W, &IID_IDirectInput8A };
        for (int i = 0; i < 2; ++i)
        {
            ComPtr<IDirectInput8W> input;
            ComPtr<IDirectInputDevice8W> keyboard;
            if (FAILED(create(GetModuleHandleW(nullptr), DIRECTINPUT_VERSION, *interfaces[i], reinterpret_cast<void**>(input.GetAddressOf()), nullptr)) ||
                FAILED(input->CreateDevice(GUID_SysKeyboard, &keyboard, nullptr)))
                continue;
            void** vtable = *reinterpret_cast<void***>(keyboard.Get());
            acquire[i] = vtable[kDInputAcquire];
            state[i] = vtable[kDInputGetDeviceState];
            data[i] = vtable[kDInputGetDeviceData];
            if (!o_Unacquire)
                o_Unacquire = reinterpret_cast<PFN_Acquire>(vtable[kDInputUnacquire]);
        }
        HookDInputMethod<AcquireThunk>(acquire, "overlay: DirectInput Acquire");
        HookDInputMethod<GetDeviceStateThunk>(state, "overlay: DirectInput GetDeviceState");
        HookDInputMethod<GetDeviceDataThunk>(data, "overlay: DirectInput GetDeviceData");
    }

    bool CalledFromGame(void* returnAddress)
    {
        auto address = reinterpret_cast<uintptr_t>(returnAddress);
        return address >= g_exeBegin && address < g_exeEnd;
    }

    BOOL WINAPI Hook_SetCursorPos(int x, int y)
    {
        if (CalledFromGame(_ReturnAddress()))
        {
            g_gameCursor = { x, y };
            g_haveGameCursor = true;
            if (g_visible)
                return TRUE;  // the game re-centres the cursor every frame; leave it to the overlay
        }
        return o_SetCursorPos(x, y);
    }

    BOOL WINAPI Hook_GetCursorPos(LPPOINT point)
    {
        if (g_visible && g_haveGameCursor && point && CalledFromGame(_ReturnAddress()))
        {
            *point = g_gameCursor;  // no mouse-look while the overlay is open
            return TRUE;
        }
        return o_GetCursorPos(point);
    }

    BOOL WINAPI Hook_ClipCursor(const RECT* rect)
    {
        return o_ClipCursor(g_visible ? nullptr : rect);
    }

    void HookCursor()
    {
        auto exe = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(exe + reinterpret_cast<const IMAGE_DOS_HEADER*>(exe)->e_lfanew);
        g_exeBegin = exe;
        g_exeEnd = exe + nt->OptionalHeader.SizeOfImage;

        MH_STATUS status = MH_CreateHookApi(L"user32", "SetCursorPos", reinterpret_cast<void*>(&Hook_SetCursorPos),
                                            reinterpret_cast<void**>(&o_SetCursorPos));
        if (status == MH_OK)
            status = MH_CreateHookApi(L"user32", "GetCursorPos", reinterpret_cast<void*>(&Hook_GetCursorPos),
                                      reinterpret_cast<void**>(&o_GetCursorPos));
        if (status == MH_OK)
            status = MH_CreateHookApi(L"user32", "ClipCursor", reinterpret_cast<void*>(&Hook_ClipCursor),
                                      reinterpret_cast<void**>(&o_ClipCursor));
        if (status == MH_OK)
            status = MH_EnableHook(MH_ALL_HOOKS);
        Log("overlay: cursor hooks: {}", MH_StatusToString(status));
    }

    void SetVisible(bool visible)
    {
        g_visible = visible;
        console::SetCapture(visible);
        if (visible)
        {
            // The game re-acquires its devices by itself once the overlay is closed.
            std::lock_guard lock(g_inputMutex);
            if (o_Unacquire)
                for (void* device : g_inputDevices)
                    o_Unacquire(device);
            if (o_ClipCursor)
                o_ClipCursor(nullptr);
        }
        Log("overlay {}", visible ? "opened" : "closed");
    }

    bool IsInputMessage(UINT msg)
    {
        return (msg >= WM_MOUSEFIRST && msg <= WM_MOUSELAST) || (msg >= WM_KEYFIRST && msg <= WM_KEYLAST) || msg == WM_INPUT;
    }

    LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        if (g_visible)
        {
            ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam);
            if (IsInputMessage(msg))
                return 0;  // the game doesn't see input while the overlay is open
        }
        return CallWindowProcW(g_gameWndProc, hwnd, msg, wParam, lParam);
    }
}

void overlay::Install(ID3D11Device* device)
{
    static bool installed = false;
    if (std::exchange(installed, true))
        return;
    HookSwapChain(device);
    HookDirectInput();
    HookCursor();
}

bool overlay::IsVisible()
{
    return g_visible;
}

// Polled once per frame: the game reads its keyboard through DirectInput, so window messages may not arrive.
void overlay::PollToggleKey()
{
    static KeyEdge toggle;
    if (toggle.Pressed(config::Current()->keyOverlay) && g_imguiReady)
        SetVisible(!g_visible);
}
