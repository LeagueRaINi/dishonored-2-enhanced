#include "console.h"

#include "game.h"
#include "hooks.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <mutex>
#include <utility>

namespace
{
    namespace addr
    {
        constexpr uintptr_t kCmdSystem       = 0x14228BB00;  // idCmdSystemLocal (static object)
        constexpr uintptr_t kCvarSystemPtr   = 0x14228BAE8;  // idCVarSystem*
        constexpr uintptr_t kPrintf          = 0x140166830;  // idCommon::Printf, an empty stub in retail
    }

    namespace layout
    {
        // idCmdSystemLocal: singly linked command list.
        constexpr size_t kCommandList        = 0x08;
        constexpr size_t kCommandNext        = 0x00;
        constexpr size_t kCommandName        = 0x08;
        constexpr size_t kCommandDescription = 0x20;
        constexpr int    kCmdExecuteText     = 8;   // vtable slot: tokenize + execute a command line now

        // idCVarSystem vtable slots and idCVar fields.
        constexpr int    kCvarCount          = 1;
        constexpr int    kCvarGet            = 2;
        constexpr size_t kCvarValue          = 0x00;  // idStr, char* data at +0
        constexpr size_t kCvarName           = 0x30;
        constexpr size_t kCvarReset          = 0x38;
        constexpr size_t kCvarDescription    = 0x40;
    }

    constexpr size_t kMaxLines    = 2000;
    constexpr size_t kMaxCommands = 4096;  // guards against a corrupt list

    std::mutex                                         g_mutex;
    std::vector<std::string>                           g_queue;
    console::Overrides                                 g_overrides;        // wanted (guarded by g_mutex)
    bool                                               g_overridesChanged = false;
    bool                                               g_listsRequested = false;
    std::atomic<std::shared_ptr<const console::Lists>> g_lists{ std::make_shared<const console::Lists>() };

    std::mutex                g_outputMutex;
    std::deque<console::Line> g_output;
    std::string               g_partialLine;
    std::atomic<bool>         g_capture{ false };

    // ---- Engine memory --------------------------------------------------------------------------------------

    // Reads a char* field of an engine object and copies the string; false on a bad pointer. Kept free of C++
    // objects so it can use SEH.
    bool CopyStringField(const void* object, size_t offset, char* out, size_t size)
    {
        out[0] = 0;
        __try
        {
            const char* source = *reinterpret_cast<const char* const*>(static_cast<const char*>(object) + offset);
            if (!source)
                return true;
            size_t i = 0;
            for (; i + 1 < size && source[i]; ++i)
                out[i] = source[i];
            out[i] = 0;
            return true;
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            out[0] = 0;
            return false;
        }
    }

    const void* ReadPointerField(const void* object, size_t offset)
    {
        __try
        {
            return *reinterpret_cast<const void* const*>(static_cast<const char*>(object) + offset);
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            return nullptr;
        }
    }

    std::string ReadString(const void* object, size_t offset)
    {
        char buffer[1024];
        return object && CopyStringField(object, offset, buffer, sizeof(buffer)) ? std::string(buffer) : std::string();
    }

    void* CvarSystem()
    {
        uintptr_t slot = game::Address(addr::kCvarSystemPtr);
        return slot ? *reinterpret_cast<void**>(slot) : nullptr;
    }

    template <typename Fn>
    Fn VirtualFunction(void* object, int slot)
    {
        return reinterpret_cast<Fn>((*static_cast<void***>(object))[slot]);
    }

    // "5", "5.0", "5.f" -> 5. False unless the whole string is a number (an optional C-style 'f' suffix allowed).
    bool ParseNumber(std::string_view text, double& out)
    {
        while (!text.empty() && text.front() == ' ')
            text.remove_prefix(1);
        while (!text.empty() && (text.back() == ' ' || text.back() == 'f' || text.back() == 'F'))
            text.remove_suffix(1);
        if (!text.empty() && text.front() == '+')
            text.remove_prefix(1);
        auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
        return !text.empty() && error == std::errc() && end == text.data() + text.size();
    }

    bool SameValue(const std::string& value, const std::string& defaultValue)
    {
        double a, b;
        if (ParseNumber(value, a) && ParseNumber(defaultValue, b))
            return std::abs(a - b) <= 1e-6 * std::max(1.0, std::abs(b));
        return value == defaultValue;
    }

    std::vector<console::Cvar> SnapshotCvars()
    {
        std::vector<console::Cvar> cvars;
        void* system = CvarSystem();
        if (!system)
            return cvars;
        using PFN_Count = int(__fastcall*)(void*);
        using PFN_Get = void*(__fastcall*)(void*, int);
        int count = VirtualFunction<PFN_Count>(system, layout::kCvarCount)(system);
        auto get = VirtualFunction<PFN_Get>(system, layout::kCvarGet);
        cvars.reserve(std::max(count, 0));
        for (int i = 0; i < count; ++i)
        {
            const void* cvar = get(system, i);
            if (!cvar)
                continue;
            console::Cvar c;
            c.name = ReadString(cvar, layout::kCvarName);
            c.value = ReadString(cvar, layout::kCvarValue);
            c.defaultValue = ReadString(cvar, layout::kCvarReset);
            c.description = ReadString(cvar, layout::kCvarDescription);
            c.changed = !SameValue(c.value, c.defaultValue);
            if (!c.name.empty())
                cvars.push_back(std::move(c));
        }
        std::ranges::sort(cvars, {}, &console::Cvar::name);
        return cvars;
    }

    std::vector<console::Command> SnapshotCommands()
    {
        std::vector<console::Command> commands;
        uintptr_t system = game::Address(addr::kCmdSystem);
        if (!system)
            return commands;
        const void* node = ReadPointerField(reinterpret_cast<const void*>(system), layout::kCommandList);
        for (size_t i = 0; node && i < kMaxCommands; ++i)
        {
            commands.push_back({ ReadString(node, layout::kCommandName), ReadString(node, layout::kCommandDescription) });
            node = ReadPointerField(node, layout::kCommandNext);
        }
        std::ranges::sort(commands, {}, &console::Command::name);
        return commands;
    }

    void RunCommand(const std::string& text)
    {
        uintptr_t system = game::Address(addr::kCmdSystem);
        if (!system)
            return;
        auto* self = reinterpret_cast<void*>(system);
        using PFN_ExecuteText = void(__fastcall*)(void*, const char*);
        VirtualFunction<PFN_ExecuteText>(self, layout::kCmdExecuteText)(self, text.c_str());
    }

    // ---- Output -------------------------------------------------------------------------------------------------

    // Engine text uses "^N" color codes.
    std::string StripColorCodes(std::string_view text)
    {
        std::string out;
        out.reserve(text.size());
        for (size_t i = 0; i < text.size(); ++i)
        {
            if (text[i] == '^' && i + 1 < text.size() && text[i + 1] >= '0' && text[i + 1] <= '9')
                ++i;
            else
                out += text[i];
        }
        return out;
    }

    void AppendLine(std::string text, bool isInput)
    {
        g_output.push_back({ std::move(text), isInput });
        while (g_output.size() > kMaxLines)
            g_output.pop_front();
    }

    // Output may arrive in pieces ("%-32s " ... "value\n"); complete lines are kept.
    void AppendOutput(std::string_view text)
    {
        std::lock_guard lock(g_outputMutex);
        g_partialLine += StripColorCodes(text);
        size_t start = 0;
        for (size_t end; (end = g_partialLine.find('\n', start)) != std::string::npos; start = end + 1)
            AppendLine(g_partialLine.substr(start, end - start), false);
        g_partialLine.erase(0, start);
    }

    void __cdecl Hook_Printf(const char* format, ...)
    {
        if (!g_capture || !format)
            return;
        char buffer[2048];
        va_list args;
        va_start(args, format);
        int written = vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        if (written > 0)
            AppendOutput(buffer);
    }
}

void console::Install()
{
    static bool installed = false;
    uintptr_t printf = game::Address(addr::kPrintf);
    if (!printf || std::exchange(installed, true))
        return;
    void* original = nullptr;  // an empty stub, never called
    hooks::InstallInline(reinterpret_cast<void*>(printf), reinterpret_cast<void*>(&Hook_Printf), &original,
                         "console: Printf capture");
}

namespace
{
    // Main thread: the enforced overrides and the values they replaced.
    std::vector<std::pair<std::string, std::string>> t_originals;  // {name, value before the first override}
    console::Overrides                               t_applied;
    int                                              t_framesSinceEnforce = 0;

    const void* FindCvar(std::string_view name)
    {
        void* system = CvarSystem();
        if (!system)
            return nullptr;
        using PFN_Count = int(__fastcall*)(void*);
        using PFN_Get = void*(__fastcall*)(void*, int);
        int count = VirtualFunction<PFN_Count>(system, layout::kCvarCount)(system);
        auto get = VirtualFunction<PFN_Get>(system, layout::kCvarGet);
        for (int i = 0; i < count; ++i)
        {
            const void* cvar = get(system, i);
            if (cvar && _stricmp(ReadString(cvar, layout::kCvarName).c_str(), std::string(name).c_str()) == 0)
                return cvar;
        }
        return nullptr;
    }

    void EnforceOverrides(const console::Overrides& wanted)
    {
        // Dropped overrides get their original value back.
        for (auto it = t_originals.begin(); it != t_originals.end();)
        {
            bool stillWanted = std::ranges::any_of(wanted, [&](const auto& w) { return w.first == it->first; });
            if (stillWanted)
            {
                ++it;
                continue;
            }
            RunCommand(it->first + " " + it->second);
            Log("console: {} restored to {}", it->first, it->second);
            it = t_originals.erase(it);
        }
        for (const auto& [name, value] : wanted)
        {
            const void* cvar = FindCvar(name);
            if (!cvar)
            {
                LOG_LIMITED(8, "console: override target {} not found", name);
                continue;
            }
            std::string current = ReadString(cvar, layout::kCvarValue);
            if (std::ranges::none_of(t_originals, [&](const auto& o) { return o.first == name; }))
                t_originals.emplace_back(name, current);
            if (!SameValue(current, value))
            {
                RunCommand(name + " " + value);
                Log("console: {} = {} (was {})", name, value, current);
            }
        }
        t_applied = wanted;
    }
}

void console::OnFrame()
{
    std::vector<std::string> queue;
    bool refresh = false;
    {
        std::lock_guard lock(g_mutex);
        queue.swap(g_queue);
        refresh = std::exchange(g_listsRequested, false);
    }
    if (!game::IsSupported())
        return;

    console::Overrides overrides;
    bool overridesChanged = false;
    {
        std::lock_guard lock(g_mutex);
        overridesChanged = std::exchange(g_overridesChanged, false);
        if (overridesChanged || ++t_framesSinceEnforce >= 60)
            overrides = g_overrides;
    }
    if (overridesChanged || t_framesSinceEnforce >= 60)
    {
        t_framesSinceEnforce = 0;
        EnforceOverrides(overrides);
    }

    for (const std::string& command : queue)
    {
        {
            std::lock_guard lock(g_outputMutex);
            AppendLine("] " + command, true);
        }
        RunCommand(command);
        refresh = true;  // values may have changed
    }
    if (refresh)
        g_lists.store(std::make_shared<const Lists>(Lists{ SnapshotCvars(), SnapshotCommands() }));
}

void console::Execute(std::string command)
{
    std::lock_guard lock(g_mutex);
    g_queue.push_back(std::move(command));
}

void console::SetOverrides(Overrides overrides)
{
    std::lock_guard lock(g_mutex);
    if (overrides == g_overrides)
        return;
    g_overrides = std::move(overrides);
    g_overridesChanged = true;
}

void console::RequestLists()
{
    std::lock_guard lock(g_mutex);
    g_listsRequested = true;
}

std::shared_ptr<const console::Lists> console::GetLists()
{
    return g_lists.load();
}

void console::SetCapture(bool capture)
{
    g_capture = capture;
}

std::vector<console::Line> console::Output()
{
    std::lock_guard lock(g_outputMutex);
    return { g_output.begin(), g_output.end() };
}

void console::ClearOutput()
{
    std::lock_guard lock(g_outputMutex);
    g_output.clear();
}
