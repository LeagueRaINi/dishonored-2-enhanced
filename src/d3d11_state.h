// RAII snapshots of the D3D11 pipeline state the mod's own passes touch inside the engine's command stream.
// The engine caches its bindings, so everything we change must be put back exactly as it was.
#pragma once

#include "core.h"

// Compute shader, SRV slots [0, 16), UAV slots [0, 8) and constant buffer 0.
class ComputeStateGuard
{
public:
    explicit ComputeStateGuard(ID3D11DeviceContext* ctx) : m_ctx(ctx)
    {
        ctx->CSGetShader(&m_shader, nullptr, nullptr);
        ctx->CSGetShaderResources(0, m_srvs.size(), m_srvs.data());
        ctx->CSGetUnorderedAccessViews(0, m_uavs.size(), m_uavs.data());
        ctx->CSGetConstantBuffers(0, 1, &m_constants);
    }

    ~ComputeStateGuard()
    {
        m_ctx->CSSetShader(m_shader.Get(), nullptr, 0);
        m_ctx->CSSetShaderResources(0, m_srvs.size(), m_srvs.data());
        m_ctx->CSSetUnorderedAccessViews(0, m_uavs.size(), m_uavs.data(), nullptr);
        m_ctx->CSSetConstantBuffers(0, 1, m_constants.GetAddressOf());
    }

    ComputeStateGuard(const ComputeStateGuard&) = delete;
    ComputeStateGuard& operator=(const ComputeStateGuard&) = delete;

private:
    ID3D11DeviceContext*                   m_ctx;
    ComPtr<ID3D11ComputeShader>            m_shader;
    ComArray<ID3D11ShaderResourceView, 16> m_srvs;
    ComArray<ID3D11UnorderedAccessView, 8> m_uavs;
    ComPtr<ID3D11Buffer>                   m_constants;
};

// Everything a fullscreen draw changes: shaders, PS slot 0, input assembly, rasterizer, output merger. Binding a
// texture as a render target silently unbinds its SRVs from every stage, so the low SRV slots of all stages that
// could reference it are restored too (after the render targets, when rebinding is legal again).
class DrawStateGuard
{
public:
    explicit DrawStateGuard(ID3D11DeviceContext* ctx) : m_ctx(ctx)
    {
        ctx->VSGetShader(&m_vs, nullptr, nullptr);
        ctx->PSGetShader(&m_ps, nullptr, nullptr);
        ctx->GSGetShader(&m_gs, nullptr, nullptr);
        ctx->HSGetShader(&m_hs, nullptr, nullptr);
        ctx->DSGetShader(&m_ds, nullptr, nullptr);
        ctx->PSGetConstantBuffers(0, 1, &m_psConstants);
        ctx->VSGetShaderResources(0, m_vsSrvs.size(), m_vsSrvs.data());
        ctx->PSGetShaderResources(0, m_psSrvs.size(), m_psSrvs.data());
        ctx->CSGetShaderResources(0, m_csSrvs.size(), m_csSrvs.data());
        ctx->IAGetInputLayout(&m_inputLayout);
        ctx->IAGetPrimitiveTopology(&m_topology);
        ctx->RSGetState(&m_rasterizer);
        ctx->RSGetViewports(&m_viewportCount, m_viewports);
        ctx->RSGetScissorRects(&m_scissorCount, m_scissors);
        ctx->OMGetBlendState(&m_blend, m_blendFactor, &m_sampleMask);
        ctx->OMGetDepthStencilState(&m_depthStencil, &m_stencilRef);
        ctx->OMGetRenderTargets(m_rtvs.size(), m_rtvs.data(), &m_dsv);
    }

    ~DrawStateGuard()
    {
        m_ctx->OMSetRenderTargets(m_rtvs.size(), m_rtvs.data(), m_dsv.Get());
        m_ctx->VSSetShaderResources(0, m_vsSrvs.size(), m_vsSrvs.data());
        m_ctx->PSSetShaderResources(0, m_psSrvs.size(), m_psSrvs.data());
        m_ctx->CSSetShaderResources(0, m_csSrvs.size(), m_csSrvs.data());
        m_ctx->VSSetShader(m_vs.Get(), nullptr, 0);
        m_ctx->PSSetShader(m_ps.Get(), nullptr, 0);
        m_ctx->GSSetShader(m_gs.Get(), nullptr, 0);
        m_ctx->HSSetShader(m_hs.Get(), nullptr, 0);
        m_ctx->DSSetShader(m_ds.Get(), nullptr, 0);
        m_ctx->PSSetConstantBuffers(0, 1, m_psConstants.GetAddressOf());
        m_ctx->IASetInputLayout(m_inputLayout.Get());
        m_ctx->IASetPrimitiveTopology(m_topology);
        m_ctx->RSSetState(m_rasterizer.Get());
        m_ctx->RSSetViewports(m_viewportCount, m_viewports);
        m_ctx->RSSetScissorRects(m_scissorCount, m_scissors);
        m_ctx->OMSetBlendState(m_blend.Get(), m_blendFactor, m_sampleMask);
        m_ctx->OMSetDepthStencilState(m_depthStencil.Get(), m_stencilRef);
    }

    DrawStateGuard(const DrawStateGuard&) = delete;
    DrawStateGuard& operator=(const DrawStateGuard&) = delete;

private:
    static constexpr UINT kSrvSlots = 16;
    static constexpr UINT kMaxViewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;

    ID3D11DeviceContext*                                                    m_ctx;
    ComPtr<ID3D11VertexShader>                                              m_vs;
    ComPtr<ID3D11PixelShader>                                               m_ps;
    ComPtr<ID3D11GeometryShader>                                            m_gs;
    ComPtr<ID3D11HullShader>                                                m_hs;
    ComPtr<ID3D11DomainShader>                                              m_ds;
    ComPtr<ID3D11Buffer>                                                    m_psConstants;
    ComArray<ID3D11ShaderResourceView, kSrvSlots>                           m_vsSrvs;
    ComArray<ID3D11ShaderResourceView, kSrvSlots>                           m_psSrvs;
    ComArray<ID3D11ShaderResourceView, kSrvSlots>                           m_csSrvs;
    ComPtr<ID3D11InputLayout>                                               m_inputLayout;
    D3D11_PRIMITIVE_TOPOLOGY                                                m_topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ComPtr<ID3D11RasterizerState>                                           m_rasterizer;
    UINT                                                                    m_viewportCount = kMaxViewports;
    D3D11_VIEWPORT                                                          m_viewports[kMaxViewports] = {};
    UINT                                                                    m_scissorCount = kMaxViewports;
    D3D11_RECT                                                              m_scissors[kMaxViewports] = {};
    ComPtr<ID3D11BlendState>                                                m_blend;
    float                                                                   m_blendFactor[4] = {};
    UINT                                                                    m_sampleMask = 0;
    ComPtr<ID3D11DepthStencilState>                                         m_depthStencil;
    UINT                                                                    m_stencilRef = 0;
    ComArray<ID3D11RenderTargetView, D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT> m_rtvs;
    ComPtr<ID3D11DepthStencilView>                                          m_dsv;
};
