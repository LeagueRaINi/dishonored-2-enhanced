#include "config.h"

#include <algorithm>
#include <atomic>
#include <mutex>

namespace
{
    std::mutex                                 g_draftMutex;
    Config                                     g_draft;
    bool                                       g_draftDirty = false;
    std::atomic<std::shared_ptr<const Config>> g_current{ std::make_shared<const Config>() };

    // Typed access to one ini file.
    class IniFile
    {
    public:
        IniFile() : m_path(ExeFile(kIniFileName).wstring()) {}

        void Read(const wchar_t* section, const wchar_t* key, int& value) const
        {
            value = static_cast<int>(GetPrivateProfileIntW(section, key, value, m_path.c_str()));
        }

        void Read(const wchar_t* section, const wchar_t* key, bool& value) const
        {
            int number = value ? 1 : 0;
            Read(section, key, number);
            value = number != 0;
        }

        void Read(const wchar_t* section, const wchar_t* key, float& value) const
        {
            std::wstring text = String(section, key);
            if (!text.empty())
                value = static_cast<float>(_wtof(text.c_str()));
        }

        void Read(const wchar_t* section, const wchar_t* key, std::string& value) const
        {
            value = Narrow(String(section, key, Widen(value).c_str()));
        }

        void Write(const wchar_t* section, const wchar_t* key, std::wstring_view value) const
        {
            WritePrivateProfileStringW(section, key, std::wstring(value).c_str(), m_path.c_str());
        }
        void Write(const wchar_t* section, const wchar_t* key, int value) const { Write(section, key, std::to_wstring(value)); }
        void Write(const wchar_t* section, const wchar_t* key, bool value) const { Write(section, key, value ? 1 : 0); }
        void Write(const wchar_t* section, const wchar_t* key, float value) const { Write(section, key, std::format(L"{:.3f}", value)); }
        void Write(const wchar_t* section, const wchar_t* key, const std::string& value) const { Write(section, key, Widen(value)); }

    private:
        std::wstring String(const wchar_t* section, const wchar_t* key, const wchar_t* fallback = L"") const
        {
            wchar_t buffer[128] = {};
            GetPrivateProfileStringW(section, key, fallback, buffer, static_cast<DWORD>(std::size(buffer)), m_path.c_str());
            return buffer;
        }

        std::wstring m_path;
    };

    // Saved by config::Save (the overlay edits it) or read from the ini only.
    enum class Persist : bool { IniOnly, Saved };

    // Every ini setting once, for loading and saving: section, key, member, persistence and, for numbers, the valid
    // range values are clamped to. C is Config (loading) or const Config (saving).
    template <typename C, typename Visit>
    void ForEachSetting(C& c, Visit&& visit)
    {
        using enum Persist;
        visit(L"General", L"SkipIntro",                c.skipIntro,          IniOnly);
        visit(L"General", L"FpsLimit",                 c.fpsLimit,           Saved);
        visit(L"General", L"FpsCounter",               c.fpsCounter,         Saved, 0, 5);
        visit(L"General", L"MaxQueuedFrames",          c.maxQueuedFrames,    Saved, 0, 3);
        visit(L"General", L"ForceScale",               c.forceScale,         IniOnly);
        visit(L"General", L"RenderScale",              c.renderScale,        Saved, 0.33f, 1.0f);
        visit(L"General", L"UnlockCvars",              c.unlockCvars,        Saved);
        visit(L"General", L"UltraPlus",                c.ultraPlus,          Saved);
        visit(L"General", L"UltrawideHud",             c.ultrawideHud,       Saved);
        visit(L"Debug",   L"DumpShaders",              c.dumpShaders,        IniOnly);

        visit(L"Keys", L"Capture",      c.keyCapture,     IniOnly);
        visit(L"Keys", L"ToggleScale",  c.keyToggleScale, IniOnly);
        visit(L"Keys", L"ToggleFSR",    c.keyToggleFsr,   IniOnly);
        visit(L"Keys", L"FSRDebugView", c.keyDebugView,   IniOnly);
        visit(L"Keys", L"Overlay",      c.keyOverlay,     IniOnly);
        visit(L"Keys", L"Benchmark",    c.keyBenchmark,   IniOnly);

        visit(L"FSR", L"Enabled",                  c.fsrEnabled,         Saved);
        visit(L"FSR", L"Sharpness",                c.fsrSharpness,       Saved, 0.0f, 1.0f);
        visit(L"FSR", L"MotionJitterCancellation", c.jitterCancellation, Saved);
        visit(L"FSR", L"HaltonJitter",             c.haltonJitter,       Saved);
        visit(L"FSR", L"DebugChecking",            c.fsrDebugChecking,   IniOnly);
        visit(L"FSR", L"Version",                  c.fsrVersion,         Saved);
        visit(L"FSR", L"MipBias",                  c.fsrMipBias,         Saved);

        visit(L"HDR", L"Enabled",        c.hdrEnabled,        Saved);
        visit(L"HDR", L"PaperWhite",     c.hdrPaperWhite,     Saved, 80.0f, 500.0f);
        visit(L"HDR", L"PeakNits",       c.hdrPeakNits,       Saved, 0.0f, 10000.0f);
        visit(L"HDR", L"HighlightBoost", c.hdrHighlightBoost, Saved, 1.0f, 2.0f);

        visit(L"SSGI", L"Enabled",            c.ssgiEnabled,   Saved);
        visit(L"SSGI", L"Strength",           c.ssgiStrength,  Saved, 0.0f, 8.0f);
        visit(L"SSGI", L"Radius",             c.ssgiRadius,    Saved, 0.1f, 20.0f);
        visit(L"SSGI", L"Thickness",          c.ssgiThickness, Saved, 0.01f, 5.0f);
        visit(L"SSGI", L"Slices",             c.ssgiSlices,    Saved, 1, 8);
        visit(L"SSGI", L"Steps",              c.ssgiSteps,     Saved, 2, 32);
        visit(L"SSGI", L"AmbientOcclusion",   c.ssgiAo,        Saved, 0.0f, 1.0f);
        visit(L"SSGI", L"SurfaceReflectance", c.ssgiAlbedo,    Saved, 0.1f, 0.9f);

        visit(L"SSR", L"Enabled",           c.ssrEnabled,        Saved);
        visit(L"SSR", L"Strength",          c.ssrStrength,       Saved, 0.0f, 2.0f);
        visit(L"SSR", L"MaxRoughness",      c.ssrMaxRoughness,   Saved, 0.05f, 1.0f);
        visit(L"SSR", L"Thickness",         c.ssrThickness,      Saved, 0.005f, 0.5f);
        visit(L"SSR", L"Steps",             c.ssrSteps,          Saved, 8, 128);
        visit(L"SSR", L"MaxDistance",       c.ssrMaxDistance,    Saved, 1.0f, 500.0f);
        visit(L"SSR", L"SpecularOcclusion", c.specularOcclusion, Saved, 0.0f, 1.0f);

        visit(L"Materials", L"Capture", c.materialCapture, Saved);
        visit(L"Materials", L"Decals",  c.decalCapture,    Saved);

        visit(L"Grading", L"Enabled",     c.gradeEnabled,     Saved);
        visit(L"Grading", L"Exposure",    c.gradeExposure,    Saved, -3.0f, 3.0f);
        visit(L"Grading", L"Contrast",    c.gradeContrast,    Saved, 0.5f, 2.0f);
        visit(L"Grading", L"Saturation",  c.gradeSaturation,  Saved, 0.0f, 2.0f);
        visit(L"Grading", L"Vibrance",    c.gradeVibrance,    Saved, -1.0f, 1.0f);
        visit(L"Grading", L"Temperature", c.gradeTemperature, Saved, -1.0f, 1.0f);
        visit(L"Grading", L"Tint",        c.gradeTint,        Saved, -1.0f, 1.0f);
        visit(L"Grading", L"GameLook",    c.gradeGameLook,    Saved, 0.0f, 1.0f);
    }

    // [FSR] Mode: "upscale" or "native".
    constexpr const wchar_t* kModeSection = L"FSR";
    constexpr const wchar_t* kModeKey     = L"Mode";
}

void config::Load()
{
    const IniFile ini;
    Config c;
    ForEachSetting(c, [&ini](const wchar_t* section, const wchar_t* key, auto& value, Persist, auto... range) {
        ini.Read(section, key, value);
        if constexpr (sizeof...(range) == 2)
            value = std::clamp(value, range...);
    });
    std::string mode = "upscale";
    ini.Read(kModeSection, kModeKey, mode);
    c.fsrUpscale = _stricmp(mode.c_str(), "native") != 0;
    if (c.fsrEnabled && c.fsrUpscale)
        c.forceScale = true;  // upscaling needs the engine to render at the reduced resolution

    {
        std::lock_guard lock(g_draftMutex);
        g_draft = c;
        g_draftDirty = false;
    }
    g_current.store(std::make_shared<const Config>(c));
    Log("config: FSR {} ({}), scale {} {:.3f}, sharpness {:.2f}, fps limit {}, version '{}'",
        c.fsrEnabled ? "on" : "off", c.fsrUpscale ? "upscale" : "native", c.forceScale ? "forced" : "native",
        c.renderScale, c.fsrSharpness, c.fpsLimit, c.fsrVersion);
}

void config::Save()
{
    const Config c = Draft();
    const IniFile ini;
    ForEachSetting(c, [&ini](const wchar_t* section, const wchar_t* key, const auto& value, Persist persist, auto...) {
        if (persist == Persist::Saved)
            ini.Write(section, key, value);
    });
    ini.Write(kModeSection, kModeKey, std::wstring_view(c.fsrUpscale ? L"upscale" : L"native"));
    Log("config: saved to {}", Narrow(kIniFileName));
}

std::shared_ptr<const Config> config::Current()
{
    return g_current.load();
}

Config config::Draft()
{
    std::lock_guard lock(g_draftMutex);
    return g_draft;
}

void config::SetDraft(const Config& draft)
{
    std::lock_guard lock(g_draftMutex);
    g_draft = draft;
    g_draftDirty = true;
}

bool config::Publish()
{
    std::shared_ptr<const Config> next;
    {
        std::lock_guard lock(g_draftMutex);
        if (!g_draftDirty)
            return false;
        g_draftDirty = false;
        next = std::make_shared<const Config>(g_draft);
    }
    g_current.store(std::move(next));
    return true;
}
