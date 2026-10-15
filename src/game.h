// Engine patches and hooks inside Dishonored2.exe. Addresses are virtual addresses at the exe's preferred base, rebased at
// runtime; they are only valid for the validated build, so nothing here touches engine memory on another build.
#pragma once

#include "core.h"

namespace game
{
    // DllMain: validates the exe build and applies the startup patches (intro skip, frame tick, mouse poll).
    void Init();
    bool IsSupported();
    uintptr_t Address(uintptr_t idbVa);  // runtime address of an IDB VA, 0 on an unknown build

    // After MH_Initialize. The two post-chain hooks are installed together or not at all.
    void InstallRenderHooks();
    bool PostUpscaleAvailable();  // post-chain hooks active and no inconsistency detected so far

    // Main thread, once per frame before RenderFrame.
    void ApplyRenderScale(bool enable, float scale);
    // The game simulates at 120 tics/s; above that, frames without a tic repeat the camera. A cap above 120 raises
    // the tick rate to the cap (experimental), unlocked keeps 120.
    constexpr int kEngineTickRate = 120;
    void ApplyFrameLimit(int fpsLimit);    // -1 = game setting, 0 = unlocked, N = cap at N fps
    void ApplyAntiAliasing(bool forceTaa); // FSR replaces the TAA pass: TAA forced on, menu entry locked
    void ApplyJitterPattern(bool halton);  // the Halton table follows the applied render scale
    void ApplyCvarUnlock(bool unlock);     // lifts the retail lock on cvars without the "settable" flag

    // The game logic state machine (arkGameLogicManager): what the game shows, from its state stack.
    enum class LogicState
    {
        Unknown = -1,  // not seen yet, or an unsupported build
        None, InGame, Pause, Notification, ReadNote, Cinematic, MainMenu, Journal, Powerwheel, Conversation,
        BlackMarket, Results, Tutorial, GameOver, CharacterChoice, MapTransition, Demo,
    };
    LogicState CurrentLogicState();  // the active state (top of the stack) after the game logic's last update
    const char* LogicStateName(LogicState state);

    // Render workers, at the TAA dispatch inside PostFX_TemporalAA_and_Post (upscale mode).
    bool IsDisplayResolutionPost();        // this job will run the rest of its post chain at display resolution
    void SwitchPostToDisplayResolution(ID3D11DeviceContext* ctx);  // call once the display-size image is recorded
}
