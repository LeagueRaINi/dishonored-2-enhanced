#include "core.h"

#include <atomic>
#include <cstdio>
#include <deque>
#include <mutex>
#include <share.h>

namespace
{
    FILE*                   g_log = nullptr;
    std::mutex              g_logMutex;  // guards the file and the recent lines
    std::deque<std::string> g_recent;
    std::atomic<uint64_t>   g_lineCount{ 0 };
}

void LogInit()
{
    g_log = _wfsopen(ModFile(L"logs", kLogFileName).c_str(), L"w", _SH_DENYWR);
}

void LogLine(std::string_view line, bool recent)
{
    std::lock_guard lock(g_logMutex);
    if (g_log)
    {
        fwrite(line.data(), 1, line.size(), g_log);
        fputc('\n', g_log);
        fflush(g_log);
    }
    if (!recent)
        return;
    g_recent.emplace_back(line);
    if (g_recent.size() > kLogKeep)
        g_recent.pop_front();
    ++g_lineCount;
}

uint64_t LogLineCount()
{
    return g_lineCount;
}

void CopyRecentLog(std::vector<std::string>& out)
{
    std::lock_guard lock(g_logMutex);
    out.assign(g_recent.begin(), g_recent.end());
}

std::string HrText(HRESULT hr)
{
    return std::format("{:#010x}", static_cast<uint32_t>(hr));
}

std::string Narrow(std::wstring_view text)
{
    if (text.empty())
        return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    std::string out(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), size, nullptr, nullptr);
    return out;
}

std::wstring Widen(std::string_view text)
{
    if (text.empty())
        return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), size);
    return out;
}

std::string DescribeAddress(const void* addr)
{
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(addr), &module) || !module)
        return std::format("{}", addr);

    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(module, path, MAX_PATH);
    std::string name = Narrow(std::filesystem::path(path).filename().native());
    uintptr_t offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(module);
    if (module == GetModuleHandleW(nullptr))
        return std::format("{}+{:#x} (IDB {:#x})", name, offset, kIdbImageBase + offset);
    return std::format("{}+{:#x}", name, offset);
}

std::filesystem::path ExeDir()
{
    static const std::filesystem::path dir = [] {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, path, MAX_PATH);
        return std::filesystem::path(path).parent_path();
    }();
    return dir;
}

std::filesystem::path ModFile(std::wstring_view folder, std::wstring_view name)
{
    std::filesystem::path dir = ExeDir() / kModFolder / folder;
    std::error_code error;
    std::filesystem::create_directories(dir, error);
    return dir / name;
}

uint64_t Fnv1a64(std::span<const std::byte> data)
{
    uint64_t hash = 1469598103934665603ull;
    for (std::byte b : data)
        hash = (hash ^ static_cast<uint8_t>(b)) * 1099511628211ull;
    return hash;
}
