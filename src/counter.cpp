#include "counter.h"

#include "config.h"
#include "fsr.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <vector>

#include "imgui.h"

namespace
{
    // ---- Theme --------------------------------------------------------------------------------------------------
    // Quiet when fine: good values use the normal value color, only warnings and problems are colored.

    constexpr ImVec4 kBackground = { 0.055f, 0.067f, 0.086f, 0.90f };
    constexpr ImVec4 kBorder     = { 1.00f, 1.00f, 1.00f, 0.07f };
    constexpr ImVec4 kValue      = { 0.93f, 0.94f, 0.96f, 1.00f };
    constexpr ImVec4 kLabel      = { 0.64f, 0.67f, 0.72f, 1.00f };
    constexpr ImVec4 kDim        = { 0.64f, 0.67f, 0.72f, 0.50f };
    constexpr ImVec4 kWarn       = { 1.00f, 0.80f, 0.30f, 1.00f };
    constexpr ImVec4 kBad        = { 1.00f, 0.42f, 0.36f, 1.00f };
    constexpr ImVec4 kAccent     = { 0.93f, 0.42f, 0.18f, 1.00f };  // the FSR tag and FSR cost only

    constexpr float kBodySize     = 15.0f;  // px before the global UI scale
    constexpr float kHeadlineSize = 28.0f;
    constexpr int   kModeColumns[counter::kModes] = { 0, 20, 26, 26, 26, 26 };

    enum Mode { Off, Minimal, FrameTime, Fsr, Full, Benchmark };

    ImFont* g_font = nullptr;  // monospace; nullptr = ImGui default font
    int     g_refreshHz = 60;

    ImU32 U32(const ImVec4& c) { return ImGui::GetColorU32(c); }

    double Now()
    {
        static const double frequency = [] {
            LARGE_INTEGER f;
            QueryPerformanceFrequency(&f);
            return static_cast<double>(f.QuadPart);
        }();
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return static_cast<double>(t.QuadPart) / frequency;
    }

    // ---- Frame statistics ---------------------------------------------------------------------------------------

    // Present-to-present frame times, newest last (~12 s at 165 fps).
    class FrameHistory
    {
    public:
        static constexpr int kSize = 2048;

        void Add(float ms)
        {
            m_samples[m_next] = ms;
            m_next = (m_next + 1) % kSize;
            m_count = std::min(m_count + 1, kSize);
        }

        int   Count() const { return m_count; }
        float Back(int i) const { return m_samples[(m_next - 1 - i + kSize) % kSize]; }  // 0 = newest

        // Number of newest frames covering the given duration.
        int FramesFor(double seconds) const
        {
            double total = 0.0;
            int n = 0;
            while (n < m_count && total < seconds * 1000.0)
                total += Back(n++);
            return n;
        }

        std::vector<float> Newest(int n) const
        {
            std::vector<float> out(n);
            for (int i = 0; i < n; ++i)
                out[i] = Back(i);
            return out;
        }

    private:
        std::array<float, kSize> m_samples{};
        int                      m_next = 0;
        int                      m_count = 0;
    };

    float Average(const std::vector<float>& v)
    {
        double sum = 0.0;
        for (float x : v)
            sum += x;
        return v.empty() ? 0.0f : static_cast<float>(sum / v.size());
    }

    float Percentile(std::vector<float> v, float fraction)
    {
        if (v.empty())
            return 0.0f;
        size_t k = std::min(v.size() - 1, static_cast<size_t>(fraction * (v.size() - 1)));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    }

    // Average frame time of the slowest `fraction` of frames ("1% low" = 0.01).
    float SlowestAverage(std::vector<float> v, float fraction)
    {
        if (v.empty())
            return 0.0f;
        size_t n = std::max<size_t>(1, static_cast<size_t>(v.size() * fraction));
        std::nth_element(v.begin(), v.begin() + n - 1, v.end(), std::greater<>());
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum += v[i];
        return static_cast<float>(sum / n);
    }

    float Fps(float ms) { return ms > 0.0f ? 1000.0f / ms : 0.0f; }

    // ---- Benchmark ----------------------------------------------------------------------------------------------

    struct BenchmarkResult
    {
        double seconds = 0;
        int    frames = 0, hitches = 0;
        float  averageMs = 0, low1Ms = 0, low01Ms = 0, p50Ms = 0, p99Ms = 0;
        double fsrGpuSum = 0, fsrGpuMax = 0;
        int    fsrSamples = 0;
        int    resetsAtStart = 0, resets = 0;
    };

    struct BenchmarkState
    {
        bool               recording = false;
        double             start = 0;
        std::vector<float> frames;
        BenchmarkResult    result;
    };

    BenchmarkResult Evaluate(const std::vector<float>& frames, double seconds)
    {
        BenchmarkResult r;
        r.seconds = seconds;
        r.frames = static_cast<int>(frames.size());
        r.averageMs = Average(frames);
        r.low1Ms = SlowestAverage(frames, 0.01f);
        r.low01Ms = SlowestAverage(frames, 0.001f);
        r.p50Ms = Percentile(frames, 0.50f);
        r.p99Ms = Percentile(frames, 0.99f);
        r.hitches = static_cast<int>(std::ranges::count_if(frames, [&](float ms) { return ms > 2.5f * r.p50Ms; }));
        return r;
    }

    // ---- Snapshot (refreshed at 4 Hz) ---------------------------------------------------------------------------

    enum class Level { Good, Warn, Bad };

    ImVec4 LevelColor(Level level)
    {
        return level == Level::Bad ? kBad : level == Level::Warn ? kWarn : kValue;
    }

    struct Snapshot
    {
        float       ms = 0, lowMs = 0, maxMs = 0, medianMs = 0, targetMs = 16.7f;
        Level       fpsLevel = Level::Good, lowLevel = Level::Good, maxLevel = Level::Good;
        fsr::Status fsr;
        std::shared_ptr<const Config> cfg = std::make_shared<const Config>();
        double      time = -1.0;
    };

    FrameHistory      g_frames;
    BenchmarkState    g_benchmark;
    Snapshot          g_snap;
    double            g_lastPresent = 0.0;
    std::atomic<bool> g_benchmarkToggle{ false };
    float             g_graphTop = 16.7f;
    double            g_graphShrinkSince = 0.0;

    void RefreshSnapshot()
    {
        double now = Now();
        if (now - g_snap.time < 0.25)
            return;
        g_snap.time = now;
        g_snap.cfg = config::Current();
        g_snap.fsr = fsr::GetStatus();

        int targetHz = g_snap.cfg->fpsLimit > 0 ? g_snap.cfg->fpsLimit : g_refreshHz;
        g_snap.targetMs = 1000.0f / std::max(targetHz, 1);

        std::vector<float> quarter = g_frames.Newest(g_frames.FramesFor(0.25));
        std::vector<float> second = g_frames.Newest(g_frames.FramesFor(1.0));
        std::vector<float> tenSeconds = g_frames.Newest(g_frames.FramesFor(10.0));
        g_snap.ms = Average(quarter);
        g_snap.medianMs = Percentile(second, 0.5f);
        g_snap.maxMs = second.empty() ? 0.0f : *std::max_element(second.begin(), second.end());
        g_snap.lowMs = SlowestAverage(tenSeconds, 0.01f);

        float fps = Fps(g_snap.ms);
        g_snap.fpsLevel = fps >= 0.80f * targetHz ? Level::Good : fps >= 0.55f * targetHz ? Level::Warn : Level::Bad;
        float lowRatio = g_snap.lowMs > 0 ? g_snap.ms / g_snap.lowMs : 1.0f;
        g_snap.lowLevel = lowRatio >= 0.70f ? Level::Good : lowRatio >= 0.50f ? Level::Warn : Level::Bad;
        g_snap.maxLevel = g_snap.maxMs > 33.3f || g_snap.maxMs >= 3.0f * g_snap.medianMs ? Level::Bad
                          : g_snap.maxMs >= 2.0f * g_snap.medianMs                          ? Level::Warn
                                                                                            : Level::Good;

        if (g_benchmark.recording)
        {
            BenchmarkResult& r = g_benchmark.result;
            int resets = r.resetsAtStart;
            double gpuSum = r.fsrGpuSum, gpuMax = r.fsrGpuMax;
            int gpuSamples = r.fsrSamples;
            r = Evaluate(g_benchmark.frames, now - g_benchmark.start);
            r.resetsAtStart = resets;
            r.resets = g_snap.fsr.resetsTotal - resets;
            r.fsrGpuSum = gpuSum + g_snap.fsr.gpuMs;
            r.fsrGpuMax = std::max(gpuMax, g_snap.fsr.gpuMsPeak);
            r.fsrSamples = gpuSamples + 1;
        }
    }

    void ToggleBenchmarkNow()
    {
        BenchmarkState& b = g_benchmark;
        if (!b.recording)
        {
            b = {};
            b.recording = true;
            b.start = Now();
            b.frames.reserve(60 * 400);
            b.result.resetsAtStart = fsr::GetStatus().resetsTotal;
            Log("benchmark: recording");
            return;
        }
        b.recording = false;
        const BenchmarkResult previous = b.result;
        b.result = Evaluate(b.frames, Now() - b.start);
        b.result.resetsAtStart = previous.resetsAtStart;
        b.result.resets = fsr::GetStatus().resetsTotal - previous.resetsAtStart;
        b.result.fsrGpuSum = previous.fsrGpuSum;
        b.result.fsrGpuMax = previous.fsrGpuMax;
        b.result.fsrSamples = previous.fsrSamples;
        const BenchmarkResult& r = b.result;
        Log("benchmark: {:.1f} s, {} frames | avg {:.1f} fps ({:.2f} ms) | 1% low {:.1f} fps | 0.1% low {:.1f} fps | "
            "p50 {:.2f} ms p99 {:.2f} ms | {} hitches | FSR avg {:.2f} ms max {:.2f} ms, {} resets",
            r.seconds, r.frames, Fps(r.averageMs), r.averageMs, Fps(r.low1Ms), Fps(r.low01Ms), r.p50Ms, r.p99Ms, r.hitches,
            r.fsrSamples ? r.fsrGpuSum / r.fsrSamples : 0.0, r.fsrGpuMax, r.resets);
    }

    // ---- Layout helpers -----------------------------------------------------------------------------------------

    std::string_view PresetName(const Config& cfg)
    {
        if (!cfg.fsrUpscale)
            return "Native AA";
        if (!cfg.forceScale)
            return "Native";
        struct Preset
        {
            float            scale;
            std::string_view name;
        };
        static constexpr Preset kPresets[] = {
            { 1.0f, "Native" }, { 0.77f, "Ultra Quality" }, { 0.667f, "Quality" },
            { 0.588f, "Balanced" }, { 0.5f, "Performance" }, { 0.333f, "Ultra Perf" },
        };
        for (const Preset& p : kPresets)
            if (std::abs(cfg.renderScale - p.scale) < 0.01f)
                return p.name;
        return "Custom";
    }

    // One text row on a character grid. Cells are placed by column; mixed font sizes share a baseline.
    class Row
    {
    public:
        Row(float columnWidth, float height) : m_cw(columnWidth), m_height(height)
        {
            m_x0 = ImGui::GetCursorStartPos().x;
            m_y = ImGui::GetCursorPosY();
        }

        ~Row()
        {
            ImGui::SetCursorPosY(m_y + m_height + ImGui::GetStyle().ItemSpacing.y);
            ImGui::SetCursorPosX(m_x0);
            ImGui::Dummy(ImVec2(0, 0));
        }

        void Left(int column, const ImVec4& color, std::string_view text, float size = 0) { Put(m_x0 + column * m_cw, color, text, size); }

        void Right(int endColumn, const ImVec4& color, std::string_view text, float size = 0)
        {
            float width = Width(text, size);
            Put(m_x0 + endColumn * m_cw - width, color, text, size);
        }

    private:
        static float Width(std::string_view text, float size)
        {
            ImFont* font = ImGui::GetFont();
            float px = size > 0 ? size : ImGui::GetFontSize();
            return font->CalcTextSizeA(px, FLT_MAX, 0.0f, text.data(), text.data() + text.size()).x;
        }

        void Put(float x, const ImVec4& color, std::string_view text, float size)
        {
            float px = size > 0 ? size : ImGui::GetFontSize();
            ImVec2 window = ImGui::GetWindowPos();
            ImVec2 pos(window.x + x, window.y + m_y + (m_height - px) * 0.8f);
            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), px, pos, U32(color), text.data(), text.data() + text.size());
        }

        float m_cw, m_height, m_x0, m_y;
    };

    void StatusTag(Row& row, int endColumn)
    {
        const Config& cfg = *g_snap.cfg;
        if (cfg.fsrDebugView || cfg.fsrDebugChecking)
            row.Right(endColumn - 4, kBad, "DBG");
        if (g_snap.fsr.failed)
            row.Right(endColumn, kBad, "FSR!");
        else if (!cfg.fsrEnabled)
            row.Right(endColumn, kDim, "FSR");
        else if (!g_snap.fsr.active)
            row.Right(endColumn, kWarn, "FSR..");
        else
            row.Right(endColumn, kAccent, "FSR");
    }

    void HeadlineRow(float cw, int columns)
    {
        {
            Row row(cw, ImGui::GetFontSize());
            row.Left(0, kLabel, "PERFORMANCE");
            StatusTag(row, columns);
        }
        const float headline = kHeadlineSize * ImGui::GetStyle().FontScaleMain;
        Row row(cw, headline);
        row.Left(0, LevelColor(g_snap.fpsLevel), g_snap.ms > 0 ? std::format("{:.0f}", Fps(g_snap.ms)) : "--", headline);
        row.Left(8, kLabel, "FPS");
        row.Right(columns - 3, kValue, std::format("{:.2f}", g_snap.ms));
        row.Right(columns, kLabel, "ms");
    }

    void FrameRows(float cw)
    {
        {
            Row row(cw, ImGui::GetFontSize());
            row.Left(0, kLabel, "1% low");
            row.Right(12, LevelColor(g_snap.lowLevel), std::format("{:3.0f}", Fps(g_snap.lowMs)));
            row.Left(12, kLabel, " fps");
            row.Right(22, kValue, std::format("{:4.1f}", g_snap.lowMs));
            row.Left(22, kLabel, " ms");
        }
        Row row(cw, ImGui::GetFontSize());
        row.Left(0, kLabel, "max 1s");
        row.Right(22, LevelColor(g_snap.maxLevel), std::format("{:4.1f}", g_snap.maxMs));
        row.Left(22, kLabel, " ms");
    }

    // Frame-time graph: one sample per pixel (no spike falls between samples), fixed steps for the top so the line
    // doesn't breathe, a dashed line at the target frame time. The scale labels sit in a gutter left of the plot,
    // so the line never runs through them.
    void FrameGraph(float width, float height)
    {
        ImGui::Spacing();
        {
            const float cw = ImGui::CalcTextSize("0").x;
            Row caption(cw, ImGui::GetFontSize());
            caption.Left(0, kLabel, "FRAME TIME");
            caption.Right(26, kDim, std::format("{:.0f} Hz ref", Fps(g_snap.targetMs)));
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImFont* font = ImGui::GetFont();
        float labelSize = ImGui::GetFontSize() * 0.8f;
        float gutter = font->CalcTextSizeA(labelSize, FLT_MAX, 0.0f, "66.7").x + 4.0f;
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 p0(origin.x + gutter, origin.y);
        ImVec2 p1(origin.x + width, origin.y + height);
        ImGui::Dummy(ImVec2(width, height));

        int n = std::min(static_cast<int>(p1.x - p0.x), g_frames.Count());
        std::vector<float> visible = g_frames.Newest(n);
        float p99 = Percentile(visible, 0.99f);
        float target = g_snap.targetMs, median = std::max(g_snap.medianMs, 0.1f);

        float steps[] = { 2.0f * target, 16.7f, 33.3f, 66.7f };
        std::ranges::sort(steps);
        float wanted = steps[3];
        for (float s : steps)
            if (s >= 1.25f * p99)
            {
                wanted = s;
                break;
            }
        double now = Now();
        if (wanted > g_graphTop)
        {
            g_graphTop = wanted;  // grow immediately
            g_graphShrinkSince = now;
        }
        else if (wanted < g_graphTop && now - g_graphShrinkSince > 3.0)
        {
            g_graphTop = wanted;  // shrink after 3 s of headroom
            g_graphShrinkSince = now;
        }
        else if (wanted == g_graphTop)
        {
            g_graphShrinkSince = now;
        }
        float top = g_graphTop;

        auto y = [&](float ms) { return p1.y - std::min(ms, top) / top * height; };
        dl->AddRectFilled(p0, p1, IM_COL32(0, 0, 0, 64), 3.0f);
        float targetY = std::floor(y(target)) + 0.5f;
        for (float x = p0.x; x < p1.x; x += 6.0f)
            dl->AddLine(ImVec2(x, targetY), ImVec2(std::min(x + 3.0f, p1.x), targetY), IM_COL32(140, 224, 140, 140));

        for (int i = n - 1; i > 0; --i)  // oldest -> newest, left -> right
        {
            float a = visible[i], b = visible[i - 1];
            float x = p1.x - static_cast<float>(i - 1);
            ImU32 color = b > 2.5f * median ? U32(kBad) : b > 1.5f * median ? U32(kWarn) : IM_COL32(217, 222, 230, 230);
            dl->AddLine(ImVec2(x - 1.0f, y(a)), ImVec2(x, y(b)), color, 1.25f);
            if (b > top)
                dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p0.y + 3.0f), U32(kBad), 2.0f);  // clipped spike
        }

        // Gutter labels, right-aligned against the plot: the scale top, and the target next to its dashed line. The
        // target label is kept inside the graph's height and below the top label when the two would touch.
        auto label = [&](float centerY, ImU32 color, const std::string& text) {
            float w = font->CalcTextSizeA(labelSize, FLT_MAX, 0.0f, text.c_str()).x;
            float labelY = std::clamp(centerY - labelSize * 0.5f, p0.y, p1.y - labelSize);
            dl->AddText(font, labelSize, ImVec2(p0.x - w - 3.0f, labelY), color, text.c_str());
            return labelY;
        };
        float topLabelY = label(p0.y + labelSize * 0.5f, U32(kLabel), std::format("{:.1f}", top));
        float targetCenter = std::max(targetY, topLabelY + labelSize * 1.5f);
        label(targetCenter, IM_COL32(140, 224, 140, 200), std::format("{:.1f}", target));
    }

    // FSR cost as a share of the frame: a rough budget figure (FSR runs on its own D3D12 queue).
    void ShareBar(float cw, int column, int columns, float share)
    {
        float x0 = ImGui::GetWindowPos().x + ImGui::GetCursorStartPos().x + column * cw;
        float barHeight = ImGui::GetFontSize() * 0.45f;
        float yMid = ImGui::GetWindowPos().y + ImGui::GetCursorPosY() + ImGui::GetFontSize() * 0.55f;
        float width = columns * cw;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(x0, yMid - barHeight * 0.5f), ImVec2(x0 + width, yMid + barHeight * 0.5f),
                          IM_COL32(255, 255, 255, 30), barHeight * 0.5f);
        dl->AddRectFilled(ImVec2(x0, yMid - barHeight * 0.5f),
                          ImVec2(x0 + width * std::clamp(share, 0.0f, 1.0f), yMid + barHeight * 0.5f), U32(kAccent),
                          barHeight * 0.5f);
    }

    float FsrShare()
    {
        return g_snap.ms > 0 ? static_cast<float>(g_snap.fsr.gpuMs) / g_snap.ms : 0.0f;
    }

    Level ShareLevel(float share)
    {
        return share > 0.30f ? Level::Bad : share > 0.20f ? Level::Warn : Level::Good;
    }

    void FsrRows(float cw, bool compact)
    {
        const Config& cfg = *g_snap.cfg;
        const fsr::Status& st = g_snap.fsr;
        bool active = st.active && cfg.fsrEnabled && st.renderWidth;
        float fs = ImGui::GetFontSize();

        if (!compact)
        {
            Row row(cw, fs);
            row.Left(0, kAccent, "FSR");
            if (st.failed)
                row.Left(4, kBad, "failed - see log");
            else if (!cfg.fsrEnabled)
                row.Left(4, kLabel, "off - engine TAA");
            else
            {
                row.Left(4, kValue, st.provider.empty() ? "--" : st.provider);
                row.Right(26, active ? kValue : kWarn, active ? (st.upscaling ? "Upscale" : "Native AA") : "Starting");
            }
        }

        active = active && st.outputWidth > 0;
        float scale = active ? static_cast<float>(st.renderWidth) / st.outputWidth : 0.0f;
        if (compact)
        {
            Row row(cw, fs);
            row.Left(0, kValue, st.provider.empty() ? "--" : st.provider);
            row.Left(7, kLabel, PresetName(cfg));
            row.Right(26, kValue, active ? std::format("{:4.2f}x", 1.0f / scale) : "--");
        }
        {
            Row row(cw, fs);
            if (active)
            {
                row.Left(0, kValue, std::format("{}x{}", st.renderWidth, st.renderHeight));
                row.Left(10, kLabel, ">");
                row.Right(26, kValue, std::format("{}x{}", st.outputWidth, st.outputHeight));
            }
            else
                row.Left(0, kDim, "--");
        }
        if (!compact)
        {
            Row row(cw, fs);
            row.Left(0, kLabel, PresetName(cfg));
            row.Right(19, kValue, active ? std::format("{:5.1f}%", scale * 100.0f) : "--");
            row.Right(26, kValue, active ? std::format("{:4.2f}x", 1.0f / scale) : "--");
        }

        float share = FsrShare();
        {
            Row row(cw, fs);
            row.Left(0, kLabel, "FSR");
            row.Right(9, kValue, active ? std::format("{:5.2f}", st.gpuMs) : "--");
            row.Left(9, kLabel, " ms");
            if (compact)
            {
                ShareBar(cw, 13, 8, share);
            }
            else
            {
                row.Left(15, kLabel, "max");
                bool spiky = st.gpuMsPeak > 1.5 * st.gpuMs;
                row.Right(26, spiky ? kWarn : kValue, active ? std::format("{:5.2f}", st.gpuMsPeak) : "--");
            }
            if (compact)
                row.Right(26, LevelColor(ShareLevel(share)), active ? std::format("{:3.0f}%", share * 100.0f) : "--");
        }
        if (compact)
            return;
        {
            Row row(cw, fs);
            ShareBar(cw, 0, 21, share);
            row.Right(26, LevelColor(ShareLevel(share)), active ? std::format("{:3.0f}%", share * 100.0f) : "--");
        }
        Row row(cw, fs);
        row.Left(0, kLabel, "Sharp");
        row.Left(6, kDim, cfg.fsrSharpness > 0 ? std::format("{:4.2f}", cfg.fsrSharpness) : std::string("off"));
        row.Left(11, kLabel, "Jit");
        row.Left(15, kDim, cfg.haltonJitter ? "32" : " 8");
        row.Left(19, kLabel, "Rst");
        row.Right(26, kDim, std::format("{:3d}", st.resetsTotal));
    }

    void BenchmarkRows(float cw)
    {
        const BenchmarkResult& r = g_benchmark.result;
        float fs = ImGui::GetFontSize();
        {
            Row row(cw, kHeadlineSize * ImGui::GetStyle().FontScaleMain);
            if (g_benchmark.recording)
            {
                int minutes = static_cast<int>(r.seconds / 60.0);
                row.Left(0, kBad, "REC");
                row.Left(4, kValue, std::format("{:02d}:{:04.1f}", minutes, r.seconds - minutes * 60.0));
            }
            else
            {
                row.Left(0, kLabel, r.frames ? "DONE" : "IDLE");
                row.Left(5, kDim, "F11 = start/stop");
            }
            StatusTag(row, 26);
        }
        auto statRow = [&](std::string_view label, float ms) {
            Row row(cw, fs);
            row.Left(0, kLabel, label);
            row.Right(14, kValue, r.frames ? std::format("{:6.1f}", Fps(ms)) : "--");
            row.Left(14, kLabel, " fps");
            row.Right(26, kValue, r.frames ? std::format("{:5.2f}", ms) : "--");
        };
        statRow("avg", r.averageMs);
        statRow("1% low", r.low1Ms);
        statRow("0.1% low", r.low01Ms);
        {
            Row row(cw, fs);
            row.Left(0, kLabel, "p50/p99");
            row.Right(26, kValue, r.frames ? std::format("{:5.2f} / {:5.2f} ms", r.p50Ms, r.p99Ms) : "--");
        }
        {
            Row row(cw, fs);
            row.Left(0, kLabel, "frames");
            row.Right(13, kValue, std::format("{:6d}", r.frames));
            row.Left(15, kLabel, "hitch");
            row.Right(26, r.hitches ? kWarn : kValue, std::format("{:4d}", r.hitches));
        }
        {
            Row row(cw, fs);
            row.Left(0, kLabel, "FSR");
            row.Right(13, kValue, r.fsrSamples ? std::format("{:5.2f}", r.fsrGpuSum / r.fsrSamples) : "--");
            row.Left(13, kLabel, " avg");
            row.Right(26, kValue, r.fsrSamples ? std::format("{:5.2f} max", r.fsrGpuMax) : "--");
        }
        Row row(cw, fs);
        row.Left(0, kLabel, "Rst");
        row.Left(4, kValue, std::format("{}", r.resets));
        row.Left(9, kLabel, PresetName(*g_snap.cfg));
    }
}

void counter::LoadFont()
{
    constexpr const char* kFont = "C:\\Windows\\Fonts\\consolab.ttf";
    if (GetFileAttributesA(kFont) != INVALID_FILE_ATTRIBUTES)
        g_font = ImGui::GetIO().Fonts->AddFontFromFileTTF(kFont, kBodySize);
    Log("counter: font {}", g_font ? "Consolas Bold" : "default (consolab.ttf not found)");
}

void counter::SetRefreshRate(int hz)
{
    if (hz > 0)
        g_refreshHz = hz;
}

void counter::OnPresent(float sampleMs)
{
    double now = Now();
    if (g_lastPresent > 0.0)
    {
        float ms = sampleMs > 0 ? sampleMs : static_cast<float>((now - g_lastPresent) * 1000.0);
        RecordFrame(ms);
    }
    g_lastPresent = now;
    if (g_benchmarkToggle.exchange(false))
        ToggleBenchmarkNow();
}

void counter::RecordFrame(float ms)
{
    if (!std::isfinite(ms) || ms <= 0.0f)
        return;
    g_frames.Add(ms);
    if (g_benchmark.recording)
        g_benchmark.frames.push_back(ms);
}

void counter::ToggleBenchmark()
{
    g_benchmarkToggle = true;
}

float counter::AverageFps()
{
    RefreshSnapshot();
    return Fps(g_snap.ms);
}

void counter::Draw()
{
    int mode = std::clamp(config::Current()->fpsCounter, 0, kModes - 1);
    if (mode == Off)
        return;
    RefreshSnapshot();

    float scale = ImGui::GetStyle().FontScaleMain;
    ImGui::PushFont(g_font, kBodySize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * scale, 11 * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6 * scale, 2 * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6 * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBackground);
    ImGui::PushStyleColor(ImGuiCol_Border, kBorder);

    float cw = ImGui::CalcTextSize("0").x;
    float width = cw * kModeColumns[mode] + 2 * 14 * scale;
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 16 * scale, 16 * scale), ImGuiCond_Always, ImVec2(1, 0));
    ImGui::SetNextWindowSizeConstraints(ImVec2(width, 0), ImVec2(width, FLT_MAX));
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                                        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_AlwaysAutoResize;
    ImGui::Begin("##counter", nullptr, kFlags);

    int columns = kModeColumns[mode];
    float graphWidth = cw * columns;
    switch (mode)
    {
    case Minimal:
    {
        const float headline = 24.0f * scale;
        Row row(cw, headline);
        row.Left(0, LevelColor(g_snap.fpsLevel), g_snap.ms > 0 ? std::format("{:.0f}", Fps(g_snap.ms)) : "--", headline);
        row.Left(7, kLabel, "FPS");
        row.Right(columns - 3, kValue, std::format("{:.1f}", g_snap.ms));
        row.Right(columns, kLabel, "ms");
        break;
    }
    case FrameTime:
        HeadlineRow(cw, columns);
        FrameRows(cw);
        FrameGraph(graphWidth, ImGui::GetFontSize() * 3.0f);
        break;
    case Fsr:
        HeadlineRow(cw, columns);
        FsrRows(cw, false);
        break;
    case Full:
        HeadlineRow(cw, columns);
        FrameRows(cw);
        FrameGraph(graphWidth, ImGui::GetFontSize() * 2.4f);
        ImGui::Spacing();
        ImGui::Separator();
        FsrRows(cw, true);
        break;
    case Benchmark:
        BenchmarkRows(cw);
        break;
    }

    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(4);
    ImGui::PopFont();
}
