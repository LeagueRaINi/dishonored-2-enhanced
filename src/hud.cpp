// Each UI draw carries its own constants: rows 0-1 place the element on the stage (screen pixels), rows 5-7 map the
// stage to clip space. The HUD movie sits at scale = height / 1080 with a side margin of (width - 1920 * scale) / 2.
// A draw anchored in the movie's left or right column at element size (the ring and its counters are drawn at 0.75 to
// 1 of the movie scale, world markers at 0.35 to 0.5) gets a copy of its constants shifted by that margin, and its
// scissor rectangle (text boxes clip to their own) moves with it.
//
// Menus, loading screens and the menu cursor use the same shaders, so nothing moves unless the game logic state is
// "in game" (game.h): pause, power wheel, journal, conversations, cinematics, map transitions and the main menu are
// states of their own. Pre-transformed batches (identity transform at the stage origin) and anchors on or beyond the
// movie's edges (full-screen frames) never move.
#include "hud.h"

#include "cbuffers.h"
#include "config.h"
#include "game.h"
#include "shader_info.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>

namespace
{
    constexpr uint64_t kUiVertexShaders[] = { 0xef2892492518e297, 0x1e0bb1462b4e3ae4 };

    constexpr float kMovieWidth  = 1920.0f;
    constexpr float kMovieHeight = 1080.0f;
    constexpr float kSideColumn  = 0.21f;  // of the movie width: the ring ends at 0.13, the hotbar starts at 0.30
    constexpr float kMinScale    = 0.6f;   // element scale relative to the movie's, below it a draw is a world marker
    constexpr float kEdge        = 8.0f;   // movie units: anchors closer to the border belong to full-screen frames

    std::atomic<bool> g_gameplay{ false };  // the game logic state is "in game"

    std::mutex           g_mutex;   // creation of the buffer below
    ComPtr<ID3D11Buffer> g_shifted; // the shifted copy of a draw's constants

    thread_local ComPtr<ID3D11Buffer> t_saved;  // the game's cb0 while ours is bound
    thread_local D3D11_RECT           t_savedScissor = {};
    thread_local UINT                 t_savedScissors = 0;
    thread_local bool                 t_restore = false;

    struct Shift
    {
        float pixels = 0.0f;  // horizontal offset in stage pixels, 0 = leave the draw alone
        float toClip = 0.0f;  // stage pixels -> clip units
        float width = 0.0f;   // stage (screen) width in pixels
    };

    float Row(const cb::Snapshot& s, int row, int column)
    {
        return s.Get<float>(static_cast<size_t>(row * 4 + column) * 4);
    }

    // How far a draw with these constants moves.
    Shift ShiftFor(const cb::Snapshot& s)
    {
        // Stage -> clip without perspective: (2 / width, -2 / height) and (-1, 1).
        float toClipX = Row(s, 5, 0), toClipY = Row(s, 6, 1);
        if (toClipX <= 0.0f || toClipY >= 0.0f || Row(s, 5, 2) != 0.0f || Row(s, 6, 2) != 0.0f || Row(s, 7, 2) != 1.0f)
            return {};
        float width = 2.0f / toClipX, height = -2.0f / toClipY;
        float scale = height / kMovieHeight;
        float margin = (width - kMovieWidth * scale) * 0.5f;
        if (margin < 1.0f)
            return {};  // 16:9 or narrower

        // The element's anchor (its local origin) in movie units. Pre-transformed batches sit at the origin with an
        // identity transform; anchors outside the movie are world markers in the side margins.
        float anchorX = Row(s, 0, 3), anchorY = Row(s, 1, 3);
        float movieX = (anchorX - margin) / scale, movieY = anchorY / scale;
        float size = std::hypot(Row(s, 0, 0), Row(s, 0, 1)) / scale;
        if (!g_gameplay || (anchorX == 0.0f && anchorY == 0.0f) || size < kMinScale || movieX < kEdge || movieX > kMovieWidth - kEdge || movieY < kEdge ||
            movieY > kMovieHeight - kEdge)
            return {};
        float pixels = movieX < kSideColumn * kMovieWidth ? -margin : movieX > (1.0f - kSideColumn) * kMovieWidth ? margin : 0.0f;
        return { pixels, toClipX, width };
    }

    ID3D11Buffer* ShiftedBuffer(ID3D11DeviceContext* ctx)
    {
        std::lock_guard lock(g_mutex);
        if (!g_shifted)
        {
            ComPtr<ID3D11Device> device;
            ctx->GetDevice(&device);
            D3D11_BUFFER_DESC desc = {};
            desc.ByteWidth = cb::kUiConstantsSize;
            desc.Usage = D3D11_USAGE_DYNAMIC;
            desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(device->CreateBuffer(&desc, nullptr, &g_shifted)))
                LOG_LIMITED(1, "hud: constant buffer creation failed");
        }
        return g_shifted.Get();
    }
}

bool hud::IsUiVertexShader(uint64_t hash)
{
    return std::ranges::contains(kUiVertexShaders, hash);
}

void hud::OnDraw(ID3D11DeviceContext* ctx)
{
    if (!config::Current()->ultrawideHud)
        return;
    ComPtr<ID3D11VertexShader> vs;
    ctx->VSGetShader(&vs, nullptr, nullptr);
    ShaderInfo info;
    if (!vs || !ReadShaderInfo(vs.Get(), info) || !IsUiVertexShader(info.hash))
        return;
    ID3D11Buffer* shifted = ShiftedBuffer(ctx);
    ComPtr<ID3D11Buffer> constants;
    ctx->VSGetConstantBuffers(0, 1, &constants);
    cb::Snapshot s;
    if (!shifted || !constants || constants.Get() == shifted || !cb::Read(constants.Get(), s) ||
        s.size != cb::kUiConstantsSize)
        return;
    Shift shift = ShiftFor(s);
    if (shift.pixels == 0.0f)
        return;

    // clip.xy = (stage * rows 5-7).xy / z: adding d * z to x moves every vertex by d in clip space.
    float clip = shift.pixels * shift.toClip;
    for (int row = 5; row <= 7; ++row)
    {
        float value = Row(s, row, 0) + clip * Row(s, row, 2);
        memcpy(s.bytes.data() + static_cast<size_t>(row * 4) * 4, &value, sizeof(value));
    }
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(ctx->Map(shifted, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return;
    memcpy(mapped.pData, s.bytes.data(), cb::kUiConstantsSize);
    ctx->Unmap(shifted, 0);
    t_saved = constants;
    t_restore = true;
    ctx->VSSetConstantBuffers(0, 1, &shifted);

    t_savedScissors = 0;
    ctx->RSGetScissorRects(&t_savedScissors, nullptr);
    t_savedScissors = std::min(t_savedScissors, 1u);
    if (t_savedScissors)
        ctx->RSGetScissorRects(&t_savedScissors, &t_savedScissor);
    if (t_savedScissors)
    {
        LONG offset = std::lround(shift.pixels), width = std::lround(shift.width);
        D3D11_RECT moved = t_savedScissor;
        moved.left = std::clamp(moved.left + offset, 0L, width);
        moved.right = std::clamp(moved.right + offset, 0L, width);
        ctx->RSSetScissorRects(1, &moved);
    }
    LOG_LIMITED(1, "hud: moving side HUD elements to the screen edges");
}

void hud::OnFrame()
{
    static game::LogicState last = game::LogicState::Unknown;
    game::LogicState state = game::CurrentLogicState();
    if (state != last)
        LOG_LIMITED(200, "hud: game state {} ({})", static_cast<int>(state), game::LogicStateName(state));
    last = state;
    g_gameplay = state == game::LogicState::InGame;
}

bool hud::NeedsAfterDraw()
{
    return t_restore;
}

void hud::AfterDraw(ID3D11DeviceContext* ctx)
{
    ctx->VSSetConstantBuffers(0, 1, t_saved.GetAddressOf());
    if (t_savedScissors)
        ctx->RSSetScissorRects(1, &t_savedScissor);
    t_saved.Reset();
    t_restore = false;
}
