#include "ui.h"

#include "config.h"
#include "console.h"
#include "counter.h"
#include "fsr.h"
#include "game.h"
#include "hdr.h"
#include "hooks.h"

#include <algorithm>
#include <array>
#include <cctype>

#include <shellapi.h>

#include "imgui.h"

namespace
{
    // ---- Theme --------------------------------------------------------------------------------------------------

    constexpr ImVec4 kAccent      = { 0.93f, 0.42f, 0.18f, 1.00f };  // FSR orange
    constexpr ImVec4 kAccentHover = { 1.00f, 0.52f, 0.28f, 1.00f };
    constexpr ImVec4 kAccentDim   = { 0.93f, 0.42f, 0.18f, 0.45f };
    constexpr ImVec4 kGood        = { 0.45f, 0.85f, 0.45f, 1.00f };
    constexpr ImVec4 kWarn        = { 1.00f, 0.75f, 0.30f, 1.00f };
    constexpr ImVec4 kBad         = { 1.00f, 0.40f, 0.32f, 1.00f };
    constexpr ImVec4 kMuted       = { 0.60f, 0.62f, 0.66f, 1.00f };

    ImVec4 Rgb(int r, int g, int b, float a = 1.0f)
    {
        return { r / 255.0f, g / 255.0f, b / 255.0f, a };
    }

    void HelpMarker(const char* text)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        ImGui::SetItemTooltip("%s", text);
    }


    // ---- Settings tabs ------------------------------------------------------------------------------------------

    void DrawStatusHeader(const fsr::Status& status, const Config& cfg)
    {
        const float fps = counter::AverageFps();
        if (ImGui::BeginTable("status", 3, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_PadOuterX))
        {
            ImGui::TableNextColumn();
            ImGui::TextColored(kMuted, "FRAME RATE");
            ImGui::Text("%.0f fps  /  %.2f ms", fps, fps > 0 ? 1000.0f / fps : 0.0f);
            ImGui::TableNextColumn();
            ImGui::TextColored(kMuted, "UPSCALER");
            if (status.failed)
                ImGui::TextColored(kBad, "Failed - see Log");
            else if (!cfg.fsrEnabled)
                ImGui::TextUnformatted("Engine TAA");
            else if (status.active)
                ImGui::Text("FSR %s / %.2f ms", status.provider.c_str(), status.gpuMs);
            else
                ImGui::TextColored(kWarn, "Starting...");
            ImGui::TableNextColumn();
            ImGui::TextColored(kMuted, "RESOLUTION");
            if (status.active && cfg.fsrEnabled)
                ImGui::Text("%ux%u > %ux%u", status.renderWidth, status.renderHeight,
                            status.outputWidth, status.outputHeight);
            else
                ImGui::TextDisabled("Waiting for render data");
            ImGui::EndTable();
        }
    }

    void DrawUpscalingTab(Config& c, const fsr::Status& status)
    {
        ImGui::SeparatorText("FidelityFX Super Resolution");
        ImGui::Checkbox("Enabled", &c.fsrEnabled);
        HelpMarker("Replaces the game's TAA. The in-game anti-aliasing setting must be TAA. Hotkey: F6.");

        int mode = c.fsrUpscale ? 0 : 1;
        if (ImGui::Combo("Mode", &mode, "Upscale\0Native AA\0"))
        {
            c.fsrUpscale = mode == 0;
            c.forceScale = c.fsrUpscale;
        }
        HelpMarker("Upscale: the scene renders at the render scale; FSR upscales it before post-processing, which\n"
                   "then runs at display resolution (AMD's recommended order).\n"
                   "Native AA: FSR only anti-aliases, at the render resolution.");

        int version = 0;
        std::string items = std::string("Newest") + '\0';
        for (size_t i = 0; i < status.versions.size(); ++i)
        {
            items += status.versions[i] + '\0';
            if (status.versions[i] == c.fsrVersion)
                version = static_cast<int>(i) + 1;
        }
        if (ImGui::Combo("Version", &version, items.c_str()))
            c.fsrVersion = version == 0 ? std::string() : status.versions[version - 1];

        ImGui::SliderFloat("Sharpness", &c.fsrSharpness, 0.0f, 1.0f, c.fsrSharpness > 0 ? "%.2f" : "off");
        ImGui::Checkbox("32-phase jitter", &c.haltonJitter);
        HelpMarker("Halton pattern spanning a full render pixel at any scale (recommended).\n"
                   "Off = the game's 8-position pattern.");
        ImGui::Checkbox("Texture detail for upscaling", &c.fsrMipBias);
        HelpMarker("Textures are sampled as sharply as at the output resolution (mip bias log2(scale) - 1, AMD's\n"
                   "recommendation for upscalers). Textures loaded before the change may need a level reload.");

        ImGui::SeparatorText("Render resolution");
        ImGui::Checkbox("Reduced render resolution", &c.forceScale);
        HelpMarker("Hotkey: F7. Applied a few seconds after launch.");
        ImGui::BeginDisabled(!c.forceScale);
        ImGui::SliderFloat("Render scale", &c.renderScale, 0.33f, 1.0f, "%.3f");
        struct Preset
        {
            const char* name;
            float       scale;
        };
        static constexpr Preset kPresets[] = {
            { "Native", 1.0f }, { "Ultra Quality", 0.77f }, { "Quality", 0.667f }, { "Balanced", 0.588f }, { "Performance", 0.5f },
        };
        for (const Preset& preset : kPresets)
        {
            if (&preset != kPresets && ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x +
                ImGui::CalcTextSize(preset.name).x + 2 * ImGui::GetStyle().FramePadding.x <
                ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x)
                ImGui::SameLine();
            bool selected = std::abs(c.renderScale - preset.scale) < 0.002f;
            if (selected)
                ImGui::PushStyleColor(ImGuiCol_Button, kAccentDim);
            if (ImGui::Button(preset.name))
                c.renderScale = preset.scale;
            if (selected)
                ImGui::PopStyleColor();
        }
        if (status.renderWidth)
            ImGui::TextColored(kMuted, "Current: %ux%u (sizes snap to multiples of 8)", status.renderWidth, status.renderHeight);
        ImGui::EndDisabled();
    }

    void DrawDisplayTab(Config& c)
    {
        ImGui::SeparatorText("Engine quality");
        ImGui::Checkbox("Ultra+", &c.ultraPlus);
        HelpMarker("Beyond the menu's Ultra: no shadow resolution dividers, model shadows drawn to 60 units\n"
                   "(instead of 25), 16x anisotropic filtering, full-quality skin shading. Turning it off restores\n"
                   "the values from before.");

        ImGui::SeparatorText("Ultrawide");
        ImGui::Checkbox("HUD at the screen edges", &c.ultrawideHud);
        HelpMarker("The game keeps its HUD inside a centred 16:9 area. This moves the elements on its sides\n"
                   "(health and mana) out to the edges of wider screens. Markers, the crosshair, the hotbar and\n"
                   "menus stay as they are. No effect at 16:9.");

        ImGui::SeparatorText("HDR");
        hdr::Status hdrStatus = hdr::GetStatus();
        if (hdrStatus.active)
            ImGui::TextColored(kGood, "HDR10 on  -  %s", hdrStatus.note.c_str());
        else
            ImGui::TextColored(kMuted, "HDR off  -  %s", hdrStatus.note.empty() ? "no swap chain yet" : hdrStatus.note.c_str());
        ImGui::Checkbox("Use HDR when Windows HDR is on", &c.hdrEnabled);
        HelpMarker("Takes effect at the next game start. The game only switches to HDR10 when Windows HDR is\n"
                   "already on for its display (Win+Alt+B toggles it); otherwise it stays SDR, untouched.");
        ImGui::BeginDisabled(!hdrStatus.active);
        ImGui::SliderFloat("Paper white", &c.hdrPaperWhite, 80.0f, 500.0f, "%.0f nits");
        HelpMarker("Brightness of SDR white: the HUD, menus and the scene's mid tones. 200 is a common choice;\n"
                   "match it to how bright the Windows desktop looks to you.");
        bool customPeak = c.hdrPeakNits > 0.0f;
        if (ImGui::Checkbox("Override peak brightness", &customPeak))
            c.hdrPeakNits = customPeak ? (hdrStatus.displayPeakNits > 0 ? hdrStatus.displayPeakNits : 1000.0f) : 0.0f;
        if (customPeak)
            ImGui::SliderFloat("Peak", &c.hdrPeakNits, 400.0f, 4000.0f, "%.0f nits");
        else
            ImGui::TextColored(kMuted, "Peak: %.0f nits (%s)", hdrStatus.displayPeakNits > 0 ? hdrStatus.displayPeakNits : 1000.0f,
                               hdrStatus.displayPeakNits > 0 ? "reported by the display" : "default, the display reports none");
        ImGui::SliderFloat("Highlight boost", &c.hdrHighlightBoost, 1.0f, 2.0f, "%.2f");
        HelpMarker("Contrast above mid grey. 1.00 keeps the scene's own brightness relations (only the very\n"
                   "brightest spots, like lamp filaments, go past paper white); higher values spread lamp glows,\n"
                   "windows and skies into the HDR range. Shadows and mid tones are never changed.");
        ImGui::Checkbox("Brightness heatmap", &c.hdrHeatmap);
        HelpMarker("Debug view of the scene's HDR output: grey up to paper white, then blue > green > yellow > red\n"
                   "as highlights approach the peak. The HUD stays normal.");
        ImGui::EndDisabled();

    }

    void DrawPerformanceTab(Config& c)
    {
        ImGui::SeparatorText("Frame rate");
        constexpr int kDefaultCap = game::kEngineTickRate;
        int limiter = c.fpsLimit < 0 ? 0 : (c.fpsLimit == 0 ? 1 : 2);
        if (ImGui::Combo("Frame limiter", &limiter, "Game setting\0Unlocked\0Custom\0"))
            c.fpsLimit = limiter == 0 ? -1 : (limiter == 1 ? 0 : kDefaultCap);
        if (c.fpsLimit > 0)
            ImGui::SliderInt("Cap", &c.fpsLimit, 30, 300, "%d fps");
        HelpMarker("The game simulates at 120 updates per second. Above 120 fps some frames get no update and the\n"
                   "camera stutters (uneven mouse look), which is why the in-game menu stops at 120.\n"
                   "A cap above 120 raises the game's update rate to match (experimental).");
        if (c.fpsLimit == 0)
            ImGui::TextColored(kWarn, "Unlocked: above 120 fps mouse look stutters. Set a cap instead.");
        else if (c.fpsLimit > game::kEngineTickRate)
            ImGui::TextColored(kWarn, "Experimental: the game updates at %d Hz. Watch for physics/animation oddities.",
                               c.fpsLimit);

        int queued = c.maxQueuedFrames;
        if (ImGui::Combo("Max queued frames", &queued, "Driver default\0" "1 (lowest latency)\0" "2\0" "3\0"))
            c.maxQueuedFrames = queued;
        HelpMarker("How many frames the CPU may run ahead of the GPU (DXGI maximum frame latency).\n"
                   "Lower = less input latency, possibly lower or less even frame rates. Compare 1 and 2.");

        ImGui::SeparatorText("Counter");
        ImGui::Combo("Frame counter", &c.fpsCounter, counter::kModeNames);
        HelpMarker("Top right corner.\n"
                   "Minimal: fps and frame time.  Frame time: 1% low, worst frame and a frame time graph.\n"
                   "FSR: provider, mode, render > output resolution, preset, GPU cost.  Full: both.\n"
                   "Benchmark: records a session (F11 starts/stops) and logs the result.");
        if (ImGui::Button("Start / stop benchmark"))
        {
            c.fpsCounter = 5;
            counter::ToggleBenchmark();
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Present intervals; values refresh four times per second.");
        ImGui::TextWrapped("1%% low uses the slowest 1%% of frames in the rolling history (up to 10 seconds). "
                           "The graph marks your configured cap, or display refresh when no cap is set. "
                           "FSR GPU time measures the upscaler only, not the whole frame.");
    }

    // ---- Console tab --------------------------------------------------------------------------------------------

    struct ConsoleState
    {
        char                     input[512] = {};
        char                     filter[128] = {};
        std::vector<std::string> history;
        int                      historyPos = -1;
        bool                     scrollToBottom = true;
        bool                     focusInput = false;
    } g_console;

    bool ContainsNoCase(std::string_view text, std::string_view part)
    {
        if (part.empty())
            return true;
        auto it = std::search(text.begin(), text.end(), part.begin(), part.end(),
                              [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) ==
                                                          std::tolower(static_cast<unsigned char>(b)); });
        return it != text.end();
    }

    int HistoryCallback(ImGuiInputTextCallbackData* data)
    {
        auto& state = *static_cast<ConsoleState*>(data->UserData);
        if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory || state.history.empty())
            return 0;
        int last = static_cast<int>(state.history.size()) - 1;
        if (data->EventKey == ImGuiKey_UpArrow)
            state.historyPos = state.historyPos < 0 ? last : std::max(0, state.historyPos - 1);
        else if (data->EventKey == ImGuiKey_DownArrow)
            state.historyPos = state.historyPos < 0 ? -1 : (state.historyPos >= last ? -1 : state.historyPos + 1);
        data->DeleteChars(0, data->BufTextLen);
        if (state.historyPos >= 0)
            data->InsertChars(0, state.history[state.historyPos].c_str());
        return 0;
    }

    void Submit(std::string command)
    {
        if (command.empty())
            return;
        g_console.history.push_back(command);
        g_console.historyPos = -1;
        g_console.scrollToBottom = true;
        console::Execute(std::move(command));
    }

    void DrawConsoleOutput(float height)
    {
        if (ImGui::BeginChild("output", ImVec2(0, height), ImGuiChildFlags_Borders))
        {
            for (const console::Line& line : console::Output())
            {
                if (line.isInput)
                    ImGui::TextColored(kAccent, "%s", line.text.c_str());
                else
                    ImGui::TextUnformatted(line.text.c_str());
            }
            if (g_console.scrollToBottom || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                ImGui::SetScrollHereY(1.0f);
            g_console.scrollToBottom = false;
        }
        ImGui::EndChild();

        ImGui::SetNextItemWidth(-ImGui::GetFrameHeight() * 3.5f);
        if (g_console.focusInput)
        {
            ImGui::SetKeyboardFocusHere();
            g_console.focusInput = false;
        }
        constexpr ImGuiInputTextFlags kInputFlags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory;
        if (ImGui::InputTextWithHint("##input", "command or cvar value  (try listCmds, listCvars -help)", g_console.input,
                                     sizeof(g_console.input), kInputFlags, HistoryCallback, &g_console))
        {
            Submit(g_console.input);
            g_console.input[0] = 0;
            g_console.focusInput = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear"))
            console::ClearOutput();
    }

    void DrawCvarTable(const console::Lists& lists)
    {
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
        std::vector<const console::Cvar*> shown;
        for (const console::Cvar& cvar : lists.cvars)
            if (ContainsNoCase(cvar.name, g_console.filter) || ContainsNoCase(cvar.description, g_console.filter))
                shown.push_back(&cvar);

        ImGui::TextColored(kMuted, "%zu of %zu cvars  -  double-click a row to edit, highlighted = changed", shown.size(),
                           lists.cvars.size());
        if (!ImGui::BeginTable("cvars", 3, kFlags, ImVec2(0, ImGui::GetContentRegionAvail().y)))
            return;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_None, 1.2f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_None, 0.6f);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_None, 2.0f);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(shown.size()));
        while (clipper.Step())
        {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row)
            {
                const console::Cvar& cvar = *shown[row];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                bool changed = cvar.changed;
                if (changed)
                    ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
                ImGui::Selectable(cvar.name.c_str(), false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
                if (changed)
                    ImGui::PopStyleColor();
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                {
                    std::string edit = cvar.name + " " + cvar.value;
                    strncpy_s(g_console.input, edit.c_str(), _TRUNCATE);
                    g_console.focusInput = true;
                }
                if (ImGui::IsItemHovered() && !cvar.defaultValue.empty())
                    ImGui::SetItemTooltip("default: %s", cvar.defaultValue.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(cvar.value.c_str());
                ImGui::TableNextColumn();
                ImGui::TextColored(kMuted, "%s", cvar.description.c_str());
            }
        }
        ImGui::EndTable();
    }

    void DrawCommandTable(const console::Lists& lists)
    {
        constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV;
        if (!ImGui::BeginTable("commands", 2, kFlags, ImVec2(0, ImGui::GetContentRegionAvail().y)))
            return;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Command", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 14);
        ImGui::TableSetupColumn("Description");
        ImGui::TableHeadersRow();
        for (const console::Command& command : lists.commands)
        {
            if (!ContainsNoCase(command.name, g_console.filter) && !ContainsNoCase(command.description, g_console.filter))
                continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable(command.name.c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
            {
                strncpy_s(g_console.input, (command.name + " ").c_str(), _TRUNCATE);
                g_console.focusInput = true;
            }
            ImGui::TableNextColumn();
            ImGui::TextColored(kMuted, "%s", command.description.c_str());
        }
        ImGui::EndTable();
    }

    void DrawConsoleTab(Config& c)
    {
        // Fresh lists whenever the tab is (re)opened.
        static int lastFrame = -10;
        if (ImGui::GetFrameCount() - lastFrame > 1)
            console::RequestLists();
        lastFrame = ImGui::GetFrameCount();
        std::shared_ptr<const console::Lists> lists = console::GetLists();

        ImGui::Checkbox("Allow all cvars", &c.unlockCvars);
        HelpMarker("Lifts the retail build's lock (\"CANNOT be set in retail\"). Read-only and command-line-only\n"
                   "cvars stay protected. Some debug cvars can break rendering or saves; listCvars -changed shows\n"
                   "what you changed, and restarting the game restores everything not saved to a config.");
        DrawConsoleOutput(ImGui::GetContentRegionAvail().y * 0.38f);
        ImGui::Spacing();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
        ImGui::InputTextWithHint("##filter", "filter (name or description)", g_console.filter, sizeof(g_console.filter));
        ImGui::SameLine();
        if (ImGui::Button("Refresh"))
            console::RequestLists();

        if (ImGui::BeginTabBar("browser"))
        {
            if (ImGui::BeginTabItem("Cvars"))
            {
                DrawCvarTable(*lists);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Commands"))
            {
                DrawCommandTable(*lists);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }

    // ---- Colour tab ---------------------------------------------------------------------------------------------

    void DrawColourTab(Config& c)
    {
        ImGui::SeparatorText("Colour grading");
        ImGui::Checkbox("Enabled##grade", &c.gradeEnabled);
        HelpMarker("Applied inside the game's tonemap pass, on the linear scene before its filmic curve, so highlights\n"
                   "keep rolling off naturally. Works in SDR and HDR. At the default values it matches the game.");
        ImGui::BeginDisabled(!c.gradeEnabled);
        ImGui::SliderFloat("Game look", &c.gradeGameLook, 0.0f, 1.0f, "%.2f");
        HelpMarker("Strength of the game's own colour grading (a lookup table that mutes and cools the image).\n"
                   "0 = neutral colours, 1 = Arkane's grade.");
        ImGui::SliderFloat("Saturation", &c.gradeSaturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vibrance", &c.gradeVibrance, -1.0f, 1.0f, "%.2f");
        HelpMarker("Saturation that mostly lifts muted colours and leaves already-strong ones alone.");
        ImGui::SliderFloat("Contrast", &c.gradeContrast, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Exposure", &c.gradeExposure, -3.0f, 3.0f, "%+.2f EV");
        ImGui::SliderFloat("Temperature", &c.gradeTemperature, -1.0f, 1.0f, "%+.2f");
        HelpMarker("White balance: negative = cooler / bluer, positive = warmer.");
        ImGui::SliderFloat("Tint", &c.gradeTint, -1.0f, 1.0f, "%+.2f");
        HelpMarker("White balance: negative = greener, positive = more magenta.");
        if (ImGui::Button("Reset to the game's look"))
        {
            Config defaults;
            c.gradeExposure = defaults.gradeExposure;
            c.gradeContrast = defaults.gradeContrast;
            c.gradeSaturation = defaults.gradeSaturation;
            c.gradeVibrance = defaults.gradeVibrance;
            c.gradeTemperature = defaults.gradeTemperature;
            c.gradeTint = defaults.gradeTint;
            c.gradeGameLook = defaults.gradeGameLook;
        }
        ImGui::EndDisabled();
    }

    // ---- Lighting tab -------------------------------------------------------------------------------------------

    void DrawLightingTab(Config& c, const fsr::Status& status)
    {
        if (!c.fsrEnabled)
            ImGui::TextColored(kWarn, "Enable FSR to use screen-space lighting.");
        if (!ImGui::BeginTabBar("lighting_sections"))
            return;
        if (ImGui::BeginTabItem("Bounce light"))
        {
            ImGui::SeparatorText("Global illumination (SSGI)");
            ImGui::Checkbox("Enabled", &c.ssgiEnabled);
            HelpMarker("Screen-space bounce light: lit surfaces brighten and tint what faces them, and corners get\n"
                       "contact shadowing. Traced at half render resolution, accumulated over frames, then fed to FSR.");
            ImGui::BeginDisabled(!c.ssgiEnabled);
            ImGui::SliderFloat("Strength", &c.ssgiStrength, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Radius", &c.ssgiRadius, 0.25f, 10.0f, "%.2f units");
            HelpMarker("How far bounce light travels, in world units. Larger radii reach further and\n"
                       "show more screen-space limits (light disappearing at the screen edges).");
            ImGui::SliderFloat("Ambient occlusion", &c.ssgiAo, 0.0f, 1.0f, "%.2f");
            HelpMarker("Extra contact shadowing from the GI trace, on top of the game's own ambient occlusion.");
            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Trace quality & material fallback"))
            {
                ImGui::SliderFloat("Thickness", &c.ssgiThickness, 0.05f, 2.0f, "%.2f units");
                HelpMarker("Assumed depth of objects seen only from the front. Small: thin objects (poles, railings) don't\n"
                           "shadow what's behind them. Large: everything behaves like a solid wall.");
                ImGui::SliderFloat("Surface reflectance", &c.ssgiAlbedo, 0.1f, 0.9f, "%.2f");
                HelpMarker("Fallback reflectance for surfaces whose material data could not be captured.\n"
                           "Captured materials use their own base colour.");
                ImGui::SliderInt("Slices", &c.ssgiSlices, 1, 8);
                ImGui::SliderInt("Steps", &c.ssgiSteps, 2, 32);
                HelpMarker("Quality: directions per pixel and samples per direction. Cost grows with slices x steps.");
            }
            ImGui::EndDisabled();

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Reflections"))
        {
            ImGui::SeparatorText("Reflections (SSR)");
            ImGui::Checkbox("Enabled##ssr", &c.ssrEnabled);
            HelpMarker("Screen-space reflections: where a reflected ray finds what's on screen, it replaces the game's\n"
                       "pre-baked cubemap reflection (which stays as the fallback). Needs material data.");
            ImGui::BeginDisabled(!c.ssrEnabled);
            ImGui::SliderFloat("Strength##ssr", &c.ssrStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Max roughness", &c.ssrMaxRoughness, 0.05f, 1.0f, "%.2f");
            HelpMarker("Rougher surfaces than this keep the cubemap only. Low: only polished floors, metal and water.");
            ImGui::Spacing();
            if (ImGui::CollapsingHeader("Ray tracing quality"))
            {
                ImGui::SliderFloat("Thickness##ssr", &c.ssrThickness, 0.005f, 0.5f, "%.3f");
                HelpMarker("How far behind a surface a ray still counts as hitting it (fraction of its distance).\n"
                           "Too low: holes in reflections. Too high: objects reflect things behind them.");
                ImGui::SliderInt("Steps##ssr", &c.ssrSteps, 8, 128);
                ImGui::SliderFloat("Max distance", &c.ssrMaxDistance, 1.0f, 200.0f, "%.0f units");
            }
            ImGui::EndDisabled();
            ImGui::SliderFloat("Specular occlusion", &c.specularOcclusion, 0.0f, 1.0f, "%.2f");
            HelpMarker("Darkens the game's cubemap reflection where the GI trace finds the surroundings closed in\n"
                       "(corners, under objects): pre-baked cubemaps don't know what's next to a surface and otherwise\n"
                       "glow there. Needs GI enabled; works with or without SSR.");

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Materials & debug"))
        {
            ImGui::SeparatorText("Material capture");
            ImGui::Checkbox("Capture material data", &c.materialCapture);
            HelpMarker("Patches the game's lighting shaders to also output each material's base colour, metalness and\n"
                       "roughness, plus the game's own reflection. GI uses the colours; SSR needs all of it.\n"
                       "Takes effect at the next game start.");
            ImGui::TextColored(kMuted, "Material capture changes require a game restart.");
            ImGui::BeginDisabled(!c.materialCapture);
            ImGui::Checkbox("Include decals", &c.decalCapture);
            HelpMarker("Decals (blood, puddles, dirt) blend their material data in by their opacity, so wet decals get\n"
                       "their own roughness and SSR reflects in them.");
            ImGui::EndDisabled();
            ImGui::SeparatorText("Visual diagnostics");
            ImGui::Combo("Debug view", &c.ssgiDebug,
                         "Off\0Added bounce light\0Visibility\0Surface colour\0Reflections\0Roughness\0Normals\0"
                         "SSR ray length\0SSR raw hits\0GI raw trace\0GI history length\0");
            HelpMarker("Normals: reconstructed from depth (what GI and SSR see).\n"
                       "SSR ray length: blue short .. red long, black = no hit.\n"
                       "SSR raw hits: this frame's rays before any filtering (radiance x confidence).\n"
                       "GI raw trace: this frame's bounce light before accumulation and denoising.\n"
                       "GI history length: black = just reset (disocclusion, moving edges) .. white = 32 frames.");
            if (status.renderWidth)
                ImGui::TextColored(kMuted, "Traced at %ux%u", (status.renderWidth + 1) / 2, (status.renderHeight + 1) / 2);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // ---- Log tab ------------------------------------------------------------------------------------------------

    struct LogView
    {
        std::vector<std::string> lines;
        uint64_t                 seen = ~0ull;  // LogLineCount() of the copy
        char                     filter[128] = {};
        bool                     follow = true;
    } g_log;

    ImVec4 LogLineColor(std::string_view line)
    {
        if (ContainsNoCase(line, "fail") || ContainsNoCase(line, "error") || ContainsNoCase(line, "crash"))
            return kBad;
        if (ContainsNoCase(line, "unexpected") || ContainsNoCase(line, "warning") || ContainsNoCase(line, "experimental"))
            return kWarn;
        if (line.starts_with("==="))
            return kAccent;
        return ImGui::GetStyleColorVec4(ImGuiCol_Text);
    }

    void DrawLogTab()
    {
        if (uint64_t count = LogLineCount(); count != g_log.seen)
        {
            CopyRecentLog(g_log.lines);
            g_log.seen = count;
        }
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 18);
        ImGui::InputTextWithHint("##logfilter", "filter", g_log.filter, sizeof(g_log.filter));
        ImGui::SameLine();
        ImGui::Checkbox("Follow", &g_log.follow);
        ImGui::SameLine();
        if (ImGui::Button("Open folder"))
            ShellExecuteW(nullptr, L"open", ModFile(L"logs", L"").c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        ImGui::TextDisabled("%zu recent lines / Full log: d2enhanced\\logs\\d2enhanced.log", g_log.lines.size());

        std::vector<const std::string*> shown;
        shown.reserve(g_log.lines.size());
        for (const std::string& line : g_log.lines)
            if (ContainsNoCase(line, g_log.filter))
                shown.push_back(&line);

        if (ImGui::BeginChild("log", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar))
        {
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(shown.size()));
            while (clipper.Step())
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
                    ImGui::TextColored(LogLineColor(*shown[i]), "%s", shown[i]->c_str());
            if (g_log.follow)
                ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
    }

    // ---- Diagnostics tab ----------------------------------------------------------------------------------------

    void DrawDiagnosticsTab(Config& c)
    {
        ImGui::SeparatorText("FSR");
        ImGui::Checkbox("Debug view", &c.fsrDebugView);
        HelpMarker("FSR's own visualization of its inputs (motion vectors, depth, reactivity). Hotkey: F10.");
        ImGui::Checkbox("Motion vector jitter cancellation", &c.jitterCancellation);
        HelpMarker("For engines whose motion vectors include the per-frame jitter (not Dishonored 2).\n"
                   "Recreates the FSR context.");
        ImGui::Checkbox("FSR auto-exposure", &c.fsrAutoExposure);
        HelpMarker("FSR estimates exposure itself instead of using the engine's auto-exposure value.\n"
                   "Test: if flickering highlights calm down, the exposure we pass is off. Recreates the context.");
        constexpr float kClamps[] = { 0.0f, 64.0f, 16.0f, 4.0f };
        int clamp = c.highlightClamp >= 64 ? 1 : c.highlightClamp >= 16 ? 2 : c.highlightClamp >= 4 ? 3 : 0;
        if (ImGui::Combo("Highlight clamp", &clamp, "Off\0" "64\0" "16\0" "4\0"))
            c.highlightClamp = kClamps[clamp];
        HelpMarker("Limits the exposed brightness of single pixels before FSR sees them (1 = white).\n"
                   "Test: if white flicker on small objects disappears, sub-pixel highlights cause it.");
        ImGui::Checkbox("FFX validation messages", &c.fsrDebugChecking);
        HelpMarker("FSR validates its inputs and logs warnings to the log. Costs performance.\n"
                   "Recreates the FSR context.");

        ImGui::SeparatorText("Capture");
        ImGui::TextWrapped("F8 logs every compute pass of one frame with its bindings to the log file and saves four "
                           "frames of upscaler inputs next to the game executable.");
    }
}

void ui::ApplyStyle(float scale)
{
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle(); // applying a new scale must not compound the previous scale
    ImGui::StyleColorsDark(&style);
    style.WindowPadding     = ImVec2(18, 16);
    style.FramePadding      = ImVec2(8, 5);
    style.ItemSpacing       = ImVec2(10, 9);
    style.ItemInnerSpacing  = ImVec2(6, 5);
    style.WindowRounding    = 8.0f;
    style.ChildRounding     = 6.0f;
    style.FrameRounding     = 5.0f;
    style.PopupRounding     = 6.0f;
    style.GrabRounding      = 4.0f;
    style.TabRounding       = 5.0f;
    style.ScrollbarRounding = 6.0f;
    style.WindowBorderSize  = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.SeparatorTextBorderSize = 2.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]             = Rgb(19, 23, 29, 0.98f);
    colors[ImGuiCol_ChildBg]              = Rgb(24, 29, 36, 0.75f);
    colors[ImGuiCol_PopupBg]              = Rgb(24, 26, 31, 0.98f);
    colors[ImGuiCol_Border]               = Rgb(60, 64, 72, 0.60f);
    colors[ImGuiCol_FrameBg]              = Rgb(36, 39, 46);
    colors[ImGuiCol_FrameBgHovered]       = Rgb(48, 52, 61);
    colors[ImGuiCol_FrameBgActive]        = Rgb(56, 61, 72);
    colors[ImGuiCol_TitleBg]              = Rgb(20, 22, 26);
    colors[ImGuiCol_TitleBgActive]        = Rgb(28, 30, 36);
    colors[ImGuiCol_Button]               = Rgb(44, 48, 56);
    colors[ImGuiCol_ButtonHovered]        = kAccentDim;
    colors[ImGuiCol_ButtonActive]         = kAccent;
    colors[ImGuiCol_Header]               = Rgb(44, 48, 56);
    colors[ImGuiCol_HeaderHovered]        = kAccentDim;
    colors[ImGuiCol_HeaderActive]         = kAccent;
    colors[ImGuiCol_CheckMark]            = kAccent;
    colors[ImGuiCol_SliderGrab]           = kAccent;
    colors[ImGuiCol_SliderGrabActive]     = kAccentHover;
    colors[ImGuiCol_Tab]                  = Rgb(32, 35, 41);
    colors[ImGuiCol_TabHovered]           = kAccentDim;
    colors[ImGuiCol_TabSelected]          = Rgb(52, 40, 34);
    colors[ImGuiCol_TabSelectedOverline]  = kAccent;
    colors[ImGuiCol_SeparatorHovered]     = kAccentDim;
    colors[ImGuiCol_SeparatorActive]      = kAccent;
    colors[ImGuiCol_ResizeGrip]           = kAccentDim;
    colors[ImGuiCol_ResizeGripHovered]    = kAccent;
    colors[ImGuiCol_PlotLines]            = kAccent;
    colors[ImGuiCol_TableHeaderBg]        = Rgb(32, 35, 41);
    colors[ImGuiCol_TableRowBgAlt]        = Rgb(255, 255, 255, 0.03f);
    colors[ImGuiCol_TextSelectedBg]       = kAccentDim;
    colors[ImGuiCol_NavCursor]            = kAccent;

    style.ScaleAllSizes(scale);
    style.FontScaleMain = scale;
}

namespace
{
    ui::Page g_page = ui::Page::Upscaling;
    constexpr const char* kPageNames[] = {
        "Upscaling", "Lighting", "Display & HDR", "Colour", "Performance", "Console", "Log", "Diagnostics"
    };
    constexpr const char* kPageDescriptions[] = {
        "Resolution, reconstruction and texture detail.",
        "Bounce light, reflections and material capture.",
        "Engine quality and HDR output.",
        "Shape the game's colour and tonal balance.",
        "Frame pacing, on-screen metrics and benchmarks.",
        "Browse engine variables and run commands.",
        "Recent messages from the renderer and mod.",
        "Inspect rendering inputs and capture a frame."
    };

    bool NavigationItem(const char* label, bool selected)
    {
        const float em = ImGui::GetFontSize();
        const ImVec2 start = ImGui::GetCursorScreenPos();
        const ImVec2 size(ImGui::GetContentRegionAvail().x, em * 1.9f);
        ImGui::PushID(label);
        const bool pressed = ImGui::InvisibleButton("##navigation", size, ImGuiButtonFlags_EnableNav);
        const bool hovered = ImGui::IsItemHovered();
        const bool focused = ImGui::IsItemFocused();
        auto* draw = ImGui::GetWindowDrawList();
        const ImVec2 end(start.x + size.x, start.y + size.y);
        if (selected || hovered)
        {
            draw->AddRectFilled(start, end, ImGui::GetColorU32(selected ? Rgb(40, 48, 59) : Rgb(31, 37, 46)), em * 0.3f);
            if (selected)
                draw->AddRect(start, end, ImGui::GetColorU32(Rgb(57, 66, 79)), em * 0.3f);
        }
        if (focused)
            draw->AddRect(start, end, ImGui::GetColorU32(kAccentDim), em * 0.3f);
        if (selected)
            draw->AddCircleFilled(ImVec2(start.x + em * 0.55f, start.y + size.y * 0.5f), em * 0.12f, ImGui::GetColorU32(kAccent));
        const ImVec4 textColor = selected || hovered ? Rgb(237, 240, 245) : Rgb(163, 173, 187);
        draw->AddText(ImVec2(start.x + em * 1.1f, start.y + (size.y - em) * 0.5f), ImGui::GetColorU32(textColor), label);
        ImGui::PopID();
        return pressed;
    }

}

void ui::SelectPage(Page page)
{
    if (page >= Page::Upscaling && page < Page::Count)
        g_page = page;
}

void ui::LoadFonts()
{
    ImGuiIO& io = ImGui::GetIO();
    const char* path = "C:\\Windows\\Fonts\\segoeui.ttf";
    if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES)
        io.FontDefault = io.Fonts->AddFontFromFileTTF(path, 16.0f);
    if (!io.FontDefault)
        io.FontDefault = io.Fonts->AddFontDefault();
}

void ui::DrawSettings()
{
    const Config before = config::Draft();
    Config c = before;
    const fsr::Status status = fsr::GetStatus();
    const float em = ImGui::GetFontSize();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    const float maxW = std::max(320.0f, display.x - 48.0f);
    const float maxH = std::max(280.0f, display.y - 48.0f);
    ImGui::SetNextWindowPos(ImVec2(24, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(std::min(em * 57, maxW), std::min(em * 38, maxH)), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(std::min(em * 43, maxW), std::min(em * 24, maxH)), ImVec2(maxW, maxH));
    if (!ImGui::Begin("Dishonored 2 Enhanced###settings", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar))
    {
        ImGui::End();
        return;
    }

    DrawStatusHeader(status, c);
    ImGui::Spacing();
    ImGui::Separator();
    const float footer = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
    const float bodyHeight = std::max(em * 6, ImGui::GetContentRegionAvail().y - footer);
    if (ImGui::BeginChild("navigation", ImVec2(em * 10.0f, bodyHeight), ImGuiChildFlags_Borders))
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, em * 0.2f));
        for (int i = 0; i < static_cast<int>(Page::Count); ++i)
        {
            if (i == 0 || i == 4 || i == 5)
            {
                if (i != 0) ImGui::Dummy(ImVec2(0, em * 0.65f));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + em * 0.55f);
                ImGui::PushFont(nullptr, 11.0f);
                ImGui::TextColored(Rgb(116, 130, 148), "%s", i == 0 ? "VISUALS" : i == 4 ? "MONITORING" : "TOOLS");
                ImGui::PopFont();
                ImGui::Dummy(ImVec2(0, em * 0.2f));
            }
            if (NavigationItem(kPageNames[i], static_cast<int>(g_page) == i))
                g_page = static_cast<Page>(i);
        }
        ImGui::PopStyleVar();
    }
    ImGui::EndChild();
    ImGui::SameLine();
    if (ImGui::BeginChild(std::format("page_{}", static_cast<int>(g_page)).c_str(), ImVec2(0, bodyHeight), ImGuiChildFlags_Borders))
    {
        const int index = static_cast<int>(g_page);
        ImGui::TextColored(kAccentHover, "%s", kPageNames[index]);
        ImGui::PushTextWrapPos(0);
        ImGui::TextColored(kMuted, "%s", kPageDescriptions[index]);
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.52f);
        ImGui::PushID(index);
        switch (g_page)
        {
        case Page::Upscaling: DrawUpscalingTab(c, status); break;
        case Page::Lighting: DrawLightingTab(c, status); break;
        case Page::Display: DrawDisplayTab(c); break;
        case Page::Colour: DrawColourTab(c); break;
        case Page::Performance: DrawPerformanceTab(c); break;
        case Page::Console: DrawConsoleTab(c); break;
        case Page::Log: DrawLogTab(); break;
        case Page::Diagnostics: DrawDiagnosticsTab(c); break;
        default: break;
        }
        ImGui::PopID();
        ImGui::PopItemWidth();
    }
    ImGui::EndChild();
    ImGui::Separator();
    if (ImGui::Button("Save settings"))
    {
        config::SetDraft(c);
        config::Save();
    }
    ImGui::SameLine();
    ImGui::TextColored(kMuted, "Live changes  /  Save keeps them after restart");
    ImGui::End();

    if (c != before)
        config::SetDraft(c);
}
