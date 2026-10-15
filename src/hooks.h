// D3D11 interception (device creation hooks, context method hooks) and the per-frame tick.
#pragma once

#include "core.h"

#include <type_traits>

namespace hooks
{
    // MinHook inline hook at `target`, created and enabled; logs "<what> hook: <status>".
    bool InstallInline(void* target, void* detour, void** original, std::string_view what);

    // Typed: the detour and the original share one function pointer type.
    template <typename Fn>
        requires std::is_function_v<std::remove_pointer_t<Fn>>
    bool InstallInline(void* target, Fn detour, Fn* original, std::string_view what)
    {
        return InstallInline(target, reinterpret_cast<void*>(detour), reinterpret_cast<void**>(original), what);
    }

    void InstallDeviceHooks(ID3D11Device* device, ID3D11DeviceContext* immediate);

    // Engine frame tick (game.cpp): main thread, right before the engine renders the frame.
    void OnFrame();

    bool IsUpscaleFrame();  // this frame's post chain runs at display resolution (decided in OnFrame)
}
