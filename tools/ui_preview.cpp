// Real menu/counter code, isolated from the game. See UI_PREVIEW.md.
#include "core.h"
#include "config.h"
#include "console.h"
#include "counter.h"
#include "fsr.h"
#include "hdr.h"
#include "ui.h"
#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#include <wincodec.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapchain;
    ComPtr<ID3D11RenderTargetView> target;
    int scenario = 0;
    bool simulatedHdr = true;
    std::vector<console::Line> output = {{"Preview console: commands are echoed only; no game is attached.", false}};

    void Check(HRESULT hr, const char* operation)
    {
        if (FAILED(hr)) throw std::runtime_error(std::format("{}: {}", operation, HrText(hr)));
    }

    void CreateTarget()
    {
        ComPtr<ID3D11Texture2D> buffer;
        Check(swapchain->GetBuffer(0, IID_PPV_ARGS(&buffer)), "GetBuffer");
        Check(device->CreateRenderTargetView(buffer.Get(), nullptr, &target), "CreateRenderTargetView");
    }

    LRESULT CALLBACK WndProc(HWND window, UINT message, WPARAM wp, LPARAM lp)
    {
        if (ImGui_ImplWin32_WndProcHandler(window, message, wp, lp)) return 1;
        if (message == WM_SIZE && swapchain && wp != SIZE_MINIMIZED)
        {
            context->OMSetRenderTargets(0, nullptr, nullptr);
            target.Reset();
            if (SUCCEEDED(swapchain->ResizeBuffers(0, LOWORD(lp), HIWORD(lp), DXGI_FORMAT_UNKNOWN, 0)))
            {
                ComPtr<ID3D11Texture2D> buffer;
                if (SUCCEEDED(swapchain->GetBuffer(0, IID_PPV_ARGS(&buffer))))
                    device->CreateRenderTargetView(buffer.Get(), nullptr, &target);
            }
            return 0;
        }
        if (message == WM_SYSCOMMAND && (wp & 0xfff0) == SC_KEYMENU) return 0;
        if (message == WM_DESTROY) { PostQuitMessage(0); return 0; }
        return DefWindowProcW(window, message, wp, lp);
    }

    // Direct GPU readback: captures the actual ImGui render even with a hidden/occluded window.
    void Capture(const std::filesystem::path& path)
    {
        ComPtr<ID3D11Texture2D> buffer, staging;
        Check(swapchain->GetBuffer(0, IID_PPV_ARGS(&buffer)), "capture buffer");
        D3D11_TEXTURE2D_DESC desc;
        buffer->GetDesc(&desc);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = desc.MiscFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Check(device->CreateTexture2D(&desc, nullptr, &staging), "capture staging");
        context->CopyResource(staging.Get(), buffer.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "capture map");
        std::vector<BYTE> pixels(static_cast<size_t>(desc.Width) * desc.Height * 4);
        for (UINT y = 0; y < desc.Height; ++y)
        {
            const auto* row = static_cast<const BYTE*>(mapped.pData) + y * mapped.RowPitch;
            BYTE* out = pixels.data() + static_cast<size_t>(y) * desc.Width * 4;
            for (UINT x = 0; x < desc.Width; ++x)
            {
                out[x * 4] = row[x * 4 + 2];
                out[x * 4 + 1] = row[x * 4 + 1];
                out[x * 4 + 2] = row[x * 4];
                out[x * 4 + 3] = 255;
            }
        }
        context->Unmap(staging.Get(), 0);
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        ComPtr<IWICImagingFactory> factory;
        ComPtr<IWICStream> stream;
        ComPtr<IWICBitmapEncoder> encoder;
        ComPtr<IWICBitmapFrameEncode> frame;
        Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "WIC");
        Check(factory->CreateStream(&stream), "PNG stream");
        Check(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), "PNG path");
        Check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "PNG encoder");
        Check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "PNG init");
        Check(encoder->CreateNewFrame(&frame, nullptr), "PNG frame");
        Check(frame->Initialize(nullptr), "frame init");
        Check(frame->SetSize(desc.Width, desc.Height), "PNG size");
        WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
        Check(frame->SetPixelFormat(&format), "PNG format");
        if (format != GUID_WICPixelFormat32bppBGRA) throw std::runtime_error("Unexpected PNG pixel format");
        Check(frame->WritePixels(desc.Height, desc.Width * 4, static_cast<UINT>(pixels.size()), pixels.data()), "PNG pixels");
        Check(frame->Commit(), "PNG frame commit");
        Check(encoder->Commit(), "PNG commit");
        std::printf("Captured %s (%ux%u)\n", path.string().c_str(), desc.Width, desc.Height);
    }

    float Sample(int index)
    {
        const float wave = 0.30f * std::sin(index * 0.13f) + 0.12f * std::sin(index * 0.71f);
        if (scenario == 1) return index % 137 == 0 ? 42.0f : 8.33f + wave;
        if (scenario == 2) return 24.0f + wave * 3.0f;
        return 8.33f + wave;
    }
}

// Only service boundaries are simulated. Config, UI, fonts, counter statistics and DX11 renderer are shared.
fsr::Status fsr::GetStatus()
{
    const auto cfg = config::Current();
    Status st;
    st.active = cfg->fsrEnabled && scenario != 3 && scenario != 4;
    st.failed = cfg->fsrEnabled && scenario == 3;
    st.provider = cfg->fsrVersion.empty() ? "4.1.1" : cfg->fsrVersion;
    st.versions = {"4.1.1", "3.1.5"};
    st.outputWidth = 2560; st.outputHeight = 1440;
    const float scale = cfg->forceScale ? cfg->renderScale : 1.0f;
    st.renderWidth = static_cast<uint32_t>(2560 * scale) / 8 * 8;
    st.renderHeight = static_cast<uint32_t>(1440 * scale) / 8 * 8;
    st.upscaling = cfg->fsrUpscale && scale < 1.0f;
    st.gpuMs = scenario == 2 ? 4.85 : 1.42;
    st.gpuMsPeak = scenario == 2 ? 6.12 : 1.76;
    st.resetsTotal = 2;
    return st;
}
hdr::Status hdr::GetStatus() { return {simulatedHdr, simulatedHdr, simulatedHdr ? 1000.0f : 0.0f, "Simulated display"}; }
void console::Execute(std::string command) { output.push_back({std::move(command), true}); output.push_back({"Preview only: command was not executed.", false}); }
void console::RequestLists() {}
std::shared_ptr<const console::Lists> console::GetLists()
{
    static const auto lists = [] {
        auto value = std::make_shared<Lists>();
        value->cvars = {{"r_antialiasing", "2", "2", "Anti-aliasing method (sample data)"},
                       {"r_lodZoomMultiplicator", "1.5", "1", "Model detail distance (sample data)", true},
                       {"r_shadowMaxResolution", "2048", "2048", "Shadow map resolution (sample data)"}};
        value->commands = {{"listCmds", "List console commands"}, {"listCvars", "List console variables"}};
        return value;
    }();
    return lists;
}
std::vector<console::Line> console::Output() { return output; }
void console::ClearOutput() { output.clear(); }

int main(int argc, char** argv)
{
    try
    {
        int width = 1600, height = 1000, mode = 4, frames = 8;
        float scale = 1.25f;
        bool hidden = false;
        int page = 0;
        std::filesystem::path capture;
        std::vector<ImVec2> clicks;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--hidden") { hidden = true; continue; }
            if (arg == "--sdr") { simulatedHdr = false; continue; }
            if (arg == "--help")
            {
                std::puts("ui_preview [--capture path.png] [--hidden] [--width 1600] [--height 1000] [--scale 1.25]\n"
                          "           [--page 0..7] [--mode 0..5] [--scenario 0..4] [--frames 8] [--sdr] [--click X,Y]\n"
                          "Pages: Upscaling, Lighting, Display, Colour, Performance, Console, Log, Diagnostics\n"
                          "Scenarios: steady, hitches, slow, failed FSR, starting FSR. Telemetry is simulated.");
                return 0;
            }
            if (++i >= argc) throw std::runtime_error("Missing option value");
            if (arg == "--capture") capture = std::filesystem::absolute(argv[i]);
            else if (arg == "--width") width = std::stoi(argv[i]);
            else if (arg == "--height") height = std::stoi(argv[i]);
            else if (arg == "--scale") scale = std::stof(argv[i]);
            else if (arg == "--page") page = std::stoi(argv[i]);
            else if (arg == "--mode") mode = std::stoi(argv[i]);
            else if (arg == "--scenario") scenario = std::stoi(argv[i]);
            else if (arg == "--frames") frames = std::stoi(argv[i]);
            else if (arg == "--click")
            {
                float x = 0, y = 0;
                if (sscanf_s(argv[i], "%f,%f", &x, &y) != 2 || !std::isfinite(x) || !std::isfinite(y))
                    throw std::runtime_error("--click expects X,Y in client pixels");
                clicks.emplace_back(x, y);
            }
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (width < 640 || width > 7680 || height < 480 || height > 4320 || !std::isfinite(scale) || scale < 0.75f || scale > 2.0f ||
            page < 0 || page >= static_cast<int>(ui::Page::Count) || mode < 0 || mode >= counter::kModes ||
            scenario < 0 || scenario > 4 || frames < 2 || frames > 10000 || (hidden && capture.empty()))
            throw std::runtime_error("Invalid options (use --help); --hidden requires --capture");
        frames = std::max(frames, 6 + static_cast<int>(clicks.size()) * 5);

        Check(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED), "COM init");
        ImGui_ImplWin32_EnableDpiAwareness();
        const WNDCLASSW wc = {CS_CLASSDC, WndProc, 0, 0, GetModuleHandleW(nullptr), nullptr,
                             LoadCursorW(nullptr, IDC_ARROW), nullptr, nullptr, L"D2EnhancedPreview"};
        if (!RegisterClassW(&wc)) throw std::runtime_error("RegisterClass failed");
        RECT rect = {0, 0, width, height};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
        HWND window = CreateWindowW(wc.lpszClassName, L"D2 Enhanced - UI preview (simulated telemetry)", WS_OVERLAPPEDWINDOW,
                                    CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
                                    nullptr, nullptr, wc.hInstance, nullptr);
        if (!window) throw std::runtime_error("CreateWindow failed");
        DXGI_SWAP_CHAIN_DESC desc{};
        desc.BufferCount = 2;
        desc.BufferDesc.Width = width; desc.BufferDesc.Height = height;
        desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.OutputWindow = window; desc.SampleDesc.Count = 1; desc.Windowed = TRUE;
        desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                                  D3D11_SDK_VERSION, &desc, &swapchain, &device, nullptr, &context);
        if (FAILED(hr))
            hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0,
                                             D3D11_SDK_VERSION, &desc, &swapchain, &device, nullptr, &context);
        Check(hr, "D3D11 device");
        CreateTarget();
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ui::ApplyStyle(scale); ui::LoadFonts(); counter::LoadFont(); counter::SetRefreshRate(120);
        if (!ImGui_ImplWin32_Init(window) || !ImGui_ImplDX11_Init(device.Get(), context.Get()))
            throw std::runtime_error("ImGui backend initialization failed");
        // No config::Load(): every preview starts from a repeatable fixture. Saving uses the EXE's isolated folder.
        config::Edit([&](Config& c) { c.fpsCounter = mode; c.forceScale = true; c.fpsLimit = 120; c.gradeEnabled = true; });
        config::Publish();
        ui::SelectPage(static_cast<ui::Page>(page));
        Log("UI preview: simulated telemetry; no game hooks installed");
        Log("FSR fixture: 1704x960 -> 2560x1440, 1.42 ms");
        Log("warning: preview settings save beside ui_preview.exe, not to the game configuration");
        for (int i = 0; i < 1200; ++i) counter::RecordFrame(Sample(i));
        if (!hidden) ShowWindow(window, SW_SHOWDEFAULT);
        bool done = false;
        bool settingsVisible = true;
        int index = 0;
        while (!done)
        {
            MSG message;
            while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&message); DispatchMessageW(&message);
                if (message.message == WM_QUIT) done = true;
            }
            if (done) break;
            if (!target) throw std::runtime_error("Preview render target unavailable after resize");
            if (IsIconic(window)) { Sleep(20); continue; }
            config::Publish();
            counter::OnPresent(Sample(1200 + index));
            ImGui_ImplDX11_NewFrame(); ImGui_ImplWin32_NewFrame();
            for (size_t i = 0; i < clicks.size(); ++i)
            {
                const int hoverFrame = 2 + static_cast<int>(i) * 5;
                if (index >= hoverFrame && (i + 1 == clicks.size() || index < hoverFrame + 5))
                {
                    ImGui::GetIO().AddFocusEvent(true);
                    ImGui::GetIO().AddMousePosEvent(clicks[i].x, clicks[i].y);
                    // Tabs allow overlapping items and require a prior hover frame before a press.
                    if (index == hoverFrame + 1 || index == hoverFrame + 2)
                        ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, index == hoverFrame + 1);
                }
            }
            ImGui::NewFrame();
            if (ImGui::IsKeyPressed(ImGuiKey_F2, false)) settingsVisible = !settingsVisible;
            if (ImGui::IsKeyPressed(ImGuiKey_F11, false))
            {
                config::Edit([](Config& c) { c.fpsCounter = 5; });
                mode = 5;
                counter::ToggleBenchmark();
            }
            const ImVec2 size = ImGui::GetIO().DisplaySize;
            auto* background = ImGui::GetBackgroundDrawList();
            background->AddRectFilledMultiColor(ImVec2(0, 0), size, IM_COL32(27, 37, 48, 255), IM_COL32(17, 24, 33, 255),
                                                IM_COL32(10, 16, 23, 255), IM_COL32(20, 30, 40, 255));
            background->AddText(ImVec2(26, size.y - 36), IM_COL32(150, 167, 185, 255),
                                "UI PREVIEW   /   Simulated game telemetry   /   Same menu and counter as the mod");
            if (settingsVisible) ui::DrawSettings();
            counter::Draw();
            bool saveImage = false;
            if (capture.empty())
            {
                ImGui::SetNextWindowPos(ImVec2(size.x - 20, size.y - 58), ImGuiCond_Always, ImVec2(1, 1));
                ImGui::SetNextWindowSize(ImVec2(310, 0));
                ImGui::Begin("Preview controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
                ImGui::TextDisabled("Simulated game services");
                if (ImGui::Combo("Scenario", &scenario, "Steady\0Hitches\0Slow\0FSR failed\0FSR starting\0"))
                    for (int i = 0; i < 2048; ++i) counter::RecordFrame(Sample(i));
                if (ImGui::Combo("Counter", &mode, counter::kModeNames)) config::Edit([&](Config& c) { c.fpsCounter = mode; });
                ImGui::Checkbox("Simulate HDR", &simulatedHdr);
                if (ImGui::SliderFloat("UI scale", &scale, 0.75f, 2.0f, "%.2f")) ui::ApplyStyle(scale);
                saveImage = ImGui::Button("Export preview.png");
                ImGui::End();
            }
            ImGui::Render();
            const float clear[] = {0.05f, 0.07f, 0.09f, 1.0f};
            context->OMSetRenderTargets(1, target.GetAddressOf(), nullptr);
            context->ClearRenderTargetView(target.Get(), clear);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            ++index;
            if (saveImage) Capture(ExeDir() / "preview.png");
            if (!capture.empty() && index >= frames) { Capture(capture); done = true; }
            else Check(swapchain->Present(hidden ? 0 : 1, 0), "Present");
        }
        ImGui_ImplDX11_Shutdown(); ImGui_ImplWin32_Shutdown(); ImGui::DestroyContext();
        context->ClearState(); target.Reset(); swapchain.Reset(); context.Reset(); device.Reset();
        DestroyWindow(window); UnregisterClassW(wc.lpszClassName, wc.hInstance); CoUninitialize();
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "UI preview: %s\n", error.what());
        return 1;
    }
}
