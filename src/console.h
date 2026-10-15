// Bridge to the engine's idTech-style console: executes commands, lists cvars/commands and captures the
// engine's Printf output (compiled out in the retail build).
//
// Commands and list snapshots run in the frame tick on the game's main thread, where the engine's own console
// would run them; the overlay only queues requests and reads snapshots.
#pragma once

#include "core.h"

#include <memory>
#include <vector>

namespace console
{
    struct Cvar
    {
        std::string name, value, defaultValue, description;
        bool        changed = false;  // value differs from the default (numerically for numbers: "5" == "5.f")
    };

    struct Command
    {
        std::string name, description;
    };

    struct Lists
    {
        std::vector<Cvar>    cvars;     // sorted by name
        std::vector<Command> commands;  // sorted by name
    };

    struct Line
    {
        std::string text;
        bool        isInput = false;  // echoed command
    };

    void Install();  // after MH_Initialize
    void OnFrame();  // main thread: runs queued commands, refreshes requested lists

    void Execute(std::string command);  // runs at the next frame
    void RequestLists();
    std::shared_ptr<const Lists> GetLists();  // latest snapshot (may be empty)

    // Cvar values the mod keeps enforced (Ultra+ preset, FSR mip bias), as {name, value}. Applied in OnFrame and
    // re-applied about once a second (the game's menus write some of them too). A cvar's value from before its
    // first override is restored when it's no longer in the list.
    using Overrides = std::vector<std::pair<std::string, std::string>>;
    void SetOverrides(Overrides overrides);

    void SetCapture(bool capture);  // collect engine messages (while the overlay is open)
    std::vector<Line> Output();
    void ClearOutput();
}
