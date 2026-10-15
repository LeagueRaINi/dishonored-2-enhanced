// d3d11.dll proxy: forwards device creation to the real d3d11.dll (system or DXVK) and installs the mod's hooks
// on the devices the game creates.
#include "config.h"
#include "game.h"
#include "hooks.h"

#include <atomic>

namespace
{
    HMODULE g_realD3D11 = nullptr;

    bool FileExists(const std::filesystem::path& path)
    {
        return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;
    }

    // DXVK mode = DXVK's dxgi.dll next to the exe (use_dxvk.bat). Its d3d11.dll lives in <mod folder>\dxvk\ and is
    // chain-loaded here; its own dxgi import then resolves to the DXVK dxgi.dll in the exe directory.
    bool LoadRealD3D11()
    {
        if (g_realD3D11)
            return true;

        std::filesystem::path dxvk = ExeDir() / kModFolder / L"dxvk" / L"d3d11.dll";
        if (FileExists(ExeFile(L"dxgi.dll")) && FileExists(dxvk))
        {
            g_realD3D11 = LoadLibraryW(dxvk.c_str());
            Log("backend: DXVK ({})", g_realD3D11 ? "loaded" : "FAILED to load");
            if (g_realD3D11)
                return true;
        }

        wchar_t system[MAX_PATH] = {};
        GetSystemDirectoryW(system, MAX_PATH);
        g_realD3D11 = LoadLibraryW((std::filesystem::path(system) / L"d3d11.dll").c_str());
        if (g_realD3D11)
            Log("backend: native D3D11");
        else
            Log("backend: failed to load the system d3d11.dll ({})", GetLastError());
        return g_realD3D11 != nullptr;
    }

    template <typename Fn>
    Fn RealExport(const char* name)
    {
        return LoadRealD3D11() ? reinterpret_cast<Fn>(GetProcAddress(g_realD3D11, name)) : nullptr;
    }

    void OnDeviceCreated(HRESULT hr, ID3D11Device** device, ID3D11DeviceContext** context)
    {
        if (SUCCEEDED(hr) && device && *device && context && *context)
            hooks::InstallDeviceHooks(*device, *context);
    }

    // ---- Crash diagnostics --------------------------------------------------------------------------------------

    bool IsFatal(DWORD code)
    {
        switch (code)
        {
        case EXCEPTION_ACCESS_VIOLATION:
        case EXCEPTION_ILLEGAL_INSTRUCTION:
        case EXCEPTION_STACK_OVERFLOW:
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_PRIV_INSTRUCTION:
        case STATUS_STACK_BUFFER_OVERRUN:
            return true;
        default:
            return false;
        }
    }

    bool IsExecutableImage(uintptr_t address)
    {
        MEMORY_BASIC_INFORMATION mbi;
        return VirtualQuery(reinterpret_cast<void*>(address), &mbi, sizeof(mbi)) && mbi.Type == MEM_IMAGE &&
               (mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE));
    }

    // First-chance handler: the game installs its own handlers that exit silently, so fatal exceptions are logged
    // here with a heuristic stack scan (values pointing into executable images are likely return addresses).
    LONG CALLBACK CrashHandler(EXCEPTION_POINTERS* info)
    {
        const EXCEPTION_RECORD& record = *info->ExceptionRecord;
        static std::atomic<int> reported{ 0 };
        if (t_faultsExpected || !IsFatal(record.ExceptionCode) || reported++ >= 4)
            return EXCEPTION_CONTINUE_SEARCH;

        Log("!!! exception {:#010x} at {} thread {}", record.ExceptionCode, DescribeAddress(record.ExceptionAddress),
            GetCurrentThreadId());
        if (record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2)
            Log("    {} address {:#x}", record.ExceptionInformation[0] ? "write to" : "read from",
                record.ExceptionInformation[1]);

        const CONTEXT& c = *info->ContextRecord;
        Log("    rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x}", c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi);

        const auto* stack = reinterpret_cast<const uintptr_t*>(c.Rsp);
        for (int slot = 0, found = 0; slot < 512 && found < 16; ++slot)
        {
            MEMORY_BASIC_INFORMATION mbi;
            if (!VirtualQuery(&stack[slot], &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
                break;
            if (!IsExecutableImage(stack[slot]))
                continue;
            Log("    [rsp+{:#05x}] {}", slot * 8, DescribeAddress(reinterpret_cast<const void*>(stack[slot])));
            ++found;
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // The CRT's secure functions would otherwise terminate the game on a bad argument inside the mod.
    void InvalidParameterHandler(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t)
    {
        Log("!!! CRT invalid parameter in {} (ignored)", kModName);
    }
}

extern "C" HRESULT WINAPI Proxy_D3D11CreateDevice(IDXGIAdapter* adapter, D3D_DRIVER_TYPE driverType, HMODULE software,
                                                  UINT flags, const D3D_FEATURE_LEVEL* featureLevels, UINT featureLevelCount,
                                                  UINT sdkVersion, ID3D11Device** device, D3D_FEATURE_LEVEL* featureLevel,
                                                  ID3D11DeviceContext** context)
{
    auto real = RealExport<PFN_D3D11_CREATE_DEVICE>("D3D11CreateDevice");
    if (!real)
        return E_FAIL;
    HRESULT hr = real(adapter, driverType, software, flags, featureLevels, featureLevelCount, sdkVersion, device,
                      featureLevel, context);
    Log("D3D11CreateDevice(driver {}, flags {:#x}) -> {}", static_cast<int>(driverType), flags, HrText(hr));
    OnDeviceCreated(hr, device, context);
    return hr;
}

extern "C" HRESULT WINAPI Proxy_D3D11CreateDeviceAndSwapChain(IDXGIAdapter* adapter, D3D_DRIVER_TYPE driverType,
                                                              HMODULE software, UINT flags,
                                                              const D3D_FEATURE_LEVEL* featureLevels,
                                                              UINT featureLevelCount, UINT sdkVersion,
                                                              const DXGI_SWAP_CHAIN_DESC* swapChainDesc,
                                                              IDXGISwapChain** swapChain, ID3D11Device** device,
                                                              D3D_FEATURE_LEVEL* featureLevel, ID3D11DeviceContext** context)
{
    auto real = RealExport<PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN>("D3D11CreateDeviceAndSwapChain");
    if (!real)
        return E_FAIL;
    HRESULT hr = real(adapter, driverType, software, flags, featureLevels, featureLevelCount, sdkVersion, swapChainDesc,
                      swapChain, device, featureLevel, context);
    Log("D3D11CreateDeviceAndSwapChain -> {}", HrText(hr));
    OnDeviceCreated(hr, device, context);
    return hr;
}

BOOL WINAPI DllMain(HMODULE module, DWORD reason, LPVOID)
{
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;
    DisableThreadLibraryCalls(module);
    // DXVK writes its logs next to the exe unless told otherwise (read when its DLLs initialize).
    SetEnvironmentVariableW(L"DXVK_LOG_PATH", ModFile(L"logs", L"").c_str());
    LogInit();
    Log("{} loaded", kModName);
    AddVectoredExceptionHandler(1, CrashHandler);
    _set_invalid_parameter_handler(InvalidParameterHandler);
    config::Load();
    if (config::Current()->hdrEnabled)
        SetEnvironmentVariableW(L"DXVK_HDR", L"1");  // DXVK exposes HDR colour spaces (read at factory creation)
    game::Init();
    return TRUE;
}
