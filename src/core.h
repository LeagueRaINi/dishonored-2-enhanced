// Shared foundation: Windows/D3D11 headers, logging, paths and small helpers used by every module.
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

// ---- Names --------------------------------------------------------------------------------------------------------

constexpr const char*    kModName     = "Dishonored 2 Enhanced";
constexpr const wchar_t* kModFolder   = L"d2enhanced";      // next to the exe: dxvk, logs, captures, sources
constexpr const wchar_t* kIniFileName = L"d2enhanced.ini";  // next to the exe
constexpr const wchar_t* kLogFileName = L"d2enhanced.log";  // in <kModFolder>\logs

// Image base the engine addresses in the sources are written against (the exe's preferred base).
constexpr uintptr_t kIdbImageBase = 0x140000000;

// ---- Logging ------------------------------------------------------------------------------------------------------
// <kModFolder>\logs\<kLogFileName>, opened with shared read access so it can be inspected while the game runs.

void LogInit();
void LogLine(std::string_view line, bool recent = true);  // recent: also kept for the overlay's log tab

// The newest log lines (up to kLogKeep) for the overlay's log tab. LogLineCount() changes with every new line, so
// the copy is only refreshed when something was logged.
constexpr size_t kLogKeep = 4000;
uint64_t LogLineCount();
void     CopyRecentLog(std::vector<std::string>& out);

template <typename... Args>
void Log(std::format_string<Args...> fmt, Args&&... args)
{
    LogLine(std::format(fmt, std::forward<Args>(args)...));
}

// Bulk detail (F8 captures): the log file only, so it doesn't flood the overlay's log tab.
template <typename... Args>
void LogFileOnly(std::format_string<Args...> fmt, Args&&... args)
{
    LogLine(std::format(fmt, std::forward<Args>(args)...), false);
}

// Logs only the first `limit` occurrences of a recurring problem.
#define LOG_LIMITED(limit, ...)                    \
    do                                             \
    {                                              \
        static int logLimitedCount_ = 0;           \
        if (logLimitedCount_ < (limit))            \
        {                                          \
            ++logLimitedCount_;                    \
            Log(__VA_ARGS__);                      \
        }                                          \
    } while (0)

// ---- Formatting helpers -------------------------------------------------------------------------------------------

std::string HrText(HRESULT hr);                 // "0x887a0005"
std::string Narrow(std::wstring_view text);     // UTF-16 -> UTF-8
std::wstring Widen(std::string_view text);      // UTF-8 -> UTF-16
std::string DescribeAddress(const void* addr);  // "d3d11.dll+0x45f60", exe addresses also as IDB VA

inline const void* Ptr(const void* p) { return p; }  // std::format only takes void pointers

// ---- Paths --------------------------------------------------------------------------------------------------------

std::filesystem::path ExeDir();
inline std::filesystem::path ExeFile(std::wstring_view name) { return ExeDir() / name; }
// Mod output stays out of the game folder: <kModFolder>\<folder>\<name> (the folder is created on demand).
std::filesystem::path ModFile(std::wstring_view folder, std::wstring_view name);

// ---- Misc ---------------------------------------------------------------------------------------------------------

// Set around code that reads memory it can't validate and handles access violations itself (SEH): the crash logger
// in proxy.cpp ignores faults on this thread meanwhile.
inline thread_local bool t_faultsExpected = false;

uint64_t Fnv1a64(std::span<const std::byte> data);

using Float4x4 = std::array<float, 16>;  // engine matrices, stored as the engine's constant buffers hold them

// Edge-triggered key polling (GetAsyncKeyState): true once per press.
class KeyEdge
{
public:
    bool Pressed(int virtualKey)
    {
        bool down = (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
        bool pressed = down && !m_down;
        m_down = down;
        return pressed;
    }

private:
    bool m_down = false;
};

// Fixed-size array of COM pointers filled by D3D11 Get* calls (which AddRef), released on destruction.
template <typename T, size_t N>
struct ComArray
{
    T* items[N] = {};

    ComArray() = default;
    ComArray(const ComArray&) = delete;
    ComArray& operator=(const ComArray&) = delete;
    ~ComArray()
    {
        for (T* item : items)
            if (item)
                item->Release();
    }

    void Reset()
    {
        for (T*& item : items)
            if (item)
                std::exchange(item, nullptr)->Release();
    }

    T**             data() { return items; }
    T* const*       data() const { return items; }
    T*              operator[](size_t i) const { return items[i]; }
    static constexpr UINT size() { return static_cast<UINT>(N); }
};

// Byte offset helpers for engine structures known only by layout.
template <typename T>
T& FieldAt(void* base, size_t offset)
{
    return *reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <typename T>
const T& FieldAt(const void* base, size_t offset)
{
    return *reinterpret_cast<const T*>(static_cast<const std::byte*>(base) + offset);
}
