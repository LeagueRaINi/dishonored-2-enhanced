#include "cbuffers.h"

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

namespace
{
    constexpr UINT kSlots = 4;  // constant buffer slots searched for the PerViewCB

    // {7D3B6E2A-5C1F-4E0B-9A47-2F6D1C8E4B14}
    constexpr GUID kTrackerGuid = { 0x7d3b6e2a, 0x5c1f, 0x4e0b, { 0x9a, 0x47, 0x2f, 0x6d, 0x1c, 0x8e, 0x4b, 0x14 } };
    // {7D3B6E2A-5C1F-4E0B-9A47-2F6D1C8E4B15}
    constexpr GUID kTwinLinkGuid = { 0x7d3b6e2a, 0x5c1f, 0x4e0b, { 0x9a, 0x47, 0x2f, 0x6d, 0x1c, 0x8e, 0x4b, 0x15 } };

    // Minimal IUnknown for objects stored with SetPrivateDataInterface.
    class RefCounted : public IUnknown
    {
    public:
        virtual ~RefCounted() = default;

        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
        {
            if (!out)
                return E_POINTER;
            if (iid != __uuidof(IUnknown))
            {
                *out = nullptr;
                return E_NOINTERFACE;
            }
            AddRef();
            *out = static_cast<IUnknown*>(this);
            return S_OK;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++m_refs; }
        ULONG STDMETHODCALLTYPE Release() override
        {
            ULONG refs = --m_refs;
            if (refs == 0)
                delete this;
            return refs;
        }

    private:
        std::atomic<ULONG> m_refs{ 1 };
    };

    // Takes ownership of the initial reference of a freshly created RefCounted object.
    template <typename T>
    ComPtr<T> MakeRef()
    {
        ComPtr<T> object;
        object.Attach(new T());
        return object;
    }

    class SrwShared
    {
    public:
        explicit SrwShared(SRWLOCK& lock) : m_lock(lock) { AcquireSRWLockShared(&m_lock); }
        ~SrwShared() { ReleaseSRWLockShared(&m_lock); }

    private:
        SRWLOCK& m_lock;
    };

    class SrwExclusive
    {
    public:
        explicit SrwExclusive(SRWLOCK& lock) : m_lock(lock) { AcquireSRWLockExclusive(&m_lock); }
        ~SrwExclusive() { ReleaseSRWLockExclusive(&m_lock); }

    private:
        SRWLOCK& m_lock;
    };

    // Private data on a twin: weak link back to the engine buffer it mirrors (cleared when that buffer dies).
    class TwinLink : public RefCounted
    {
    public:
        SRWLOCK       lock = SRWLOCK_INIT;
        ID3D11Buffer* real = nullptr;
    };

    // Private data on a tracked engine buffer: its CPU shadow and (PerViewCB only) its twin.
    class Tracker : public RefCounted
    {
    public:
        ~Tracker() override
        {
            if (link)
            {
                SrwExclusive guard(link->lock);
                link->real = nullptr;
            }
        }

        SRWLOCK              lock = SRWLOCK_INIT;
        cb::Snapshot         shadow;
        ComPtr<ID3D11Buffer> twin;
        ComPtr<TwinLink>     link;
    };

    // DXVK's private data storage is unsynchronized; all of the mod's private data calls go through this lock.
    SRWLOCK g_privateDataLock = SRWLOCK_INIT;

    template <typename T>
    ComPtr<T> GetAttached(ID3D11DeviceChild* object, REFGUID guid)
    {
        IUnknown* attached = nullptr;
        UINT size = sizeof(attached);
        SrwShared guard(g_privateDataLock);
        if (FAILED(object->GetPrivateData(guid, &size, &attached)) || !attached)
            return nullptr;
        ComPtr<T> result;
        result.Attach(static_cast<T*>(attached));  // GetPrivateData returned an AddRef'd pointer
        return result;
    }

    // Attaches `fresh` unless another thread attached an object first; returns the attached one.
    template <typename T>
    ComPtr<T> Attach(ID3D11DeviceChild* object, REFGUID guid, ComPtr<T> fresh)
    {
        SrwExclusive guard(g_privateDataLock);
        IUnknown* existing = nullptr;
        UINT size = sizeof(existing);
        if (SUCCEEDED(object->GetPrivateData(guid, &size, &existing)) && existing)
        {
            ComPtr<T> result;
            result.Attach(static_cast<T*>(existing));
            return result;
        }
        object->SetPrivateDataInterface(guid, fresh.Get());
        return fresh;
    }

    UINT TrackedSize(ID3D11Resource* resource)
    {
        D3D11_RESOURCE_DIMENSION dimension;
        resource->GetType(&dimension);
        if (dimension != D3D11_RESOURCE_DIMENSION_BUFFER)
            return 0;
        D3D11_BUFFER_DESC desc;
        static_cast<ID3D11Buffer*>(resource)->GetDesc(&desc);
        if (!(desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER))
            return 0;
        bool tracked = desc.ByteWidth == cb::kPerViewSize || desc.ByteWidth == cb::kPerInstanceSize ||
                       desc.ByteWidth == cb::kUiConstantsSize;
        return tracked ? desc.ByteWidth : 0;
    }

    void StoreShadow(ID3D11Resource* resource, const void* data, UINT size)
    {
        ComPtr<Tracker> tracker = GetAttached<Tracker>(resource, kTrackerGuid);
        if (!tracker)
            tracker = Attach(resource, kTrackerGuid, MakeRef<Tracker>());
        SrwExclusive guard(tracker->lock);
        memcpy(tracker->shadow.bytes.data(), data, size);
        tracker->shadow.size = size;
        ++tracker->shadow.revision;
    }

    // A Map of a tracked buffer, completed at its Unmap.
    struct PendingMap
    {
        ID3D11DeviceContext* ctx = nullptr;
        ID3D11Resource*      resource = nullptr;
        const void*          data = nullptr;
        UINT                 size = 0;
    };
    thread_local PendingMap t_pendingMap;
    thread_local bool       t_internalMap = false;  // the mod's own twin writes are not engine updates

    // ---- Twins --------------------------------------------------------------------------------------------------

    struct WrittenTwin
    {
        ID3D11Buffer* twin;
        uint64_t      revision;
    };
    thread_local ID3D11DeviceContext*     t_displayCtx = nullptr;
    thread_local std::vector<WrittenTwin> t_written;  // twins mapped in t_displayCtx's current command list

    ComPtr<ID3D11Buffer> GetOrCreateTwin(ID3D11Buffer* real, Tracker& tracker)
    {
        SrwExclusive guard(tracker.lock);
        if (tracker.twin)
            return tracker.twin;

        D3D11_BUFFER_DESC desc;
        real->GetDesc(&desc);
        ComPtr<ID3D11Device> device;
        real->GetDevice(&device);
        ComPtr<ID3D11Buffer> twin;
        if (FAILED(device->CreateBuffer(&desc, nullptr, &twin)))
            return nullptr;

        ComPtr<TwinLink> link = MakeRef<TwinLink>();
        link->real = real;  // weak: the tracker (owned by `real`) clears it on destruction
        Attach(twin.Get(), kTwinLinkGuid, link);
        tracker.twin = twin;
        tracker.link = link;
        return twin;
    }

    // Makes sure the twin holds the real buffer's current contents with resolutionscale = 1.
    bool WriteTwin(ID3D11DeviceContext* ctx, ID3D11Buffer* twin, Tracker& tracker)
    {
        cb::Snapshot shadow;
        {
            SrwShared guard(tracker.lock);
            shadow = tracker.shadow;
        }
        if (shadow.size != cb::kPerViewSize)
            return false;

        auto written = std::ranges::find(t_written, twin, &WrittenTwin::twin);
        if (written != t_written.end() && written->revision == shadow.revision)
            return true;

        const float displayScale[2] = { 1.0f, 1.0f };
        memcpy(shadow.bytes.data() + cb::perview::kResolutionScale, displayScale, sizeof(displayScale));
        D3D11_MAPPED_SUBRESOURCE mapped;
        t_internalMap = true;
        HRESULT hr = ctx->Map(twin, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (SUCCEEDED(hr))
        {
            memcpy(mapped.pData, shadow.bytes.data(), cb::kPerViewSize);
            ctx->Unmap(twin, 0);
        }
        t_internalMap = false;
        if (FAILED(hr))
        {
            LOG_LIMITED(5, "cbuffers: twin Map failed {}", HrText(hr));
            return false;
        }

        if (written != t_written.end())
            written->revision = shadow.revision;
        else
            t_written.push_back({ twin, shadow.revision });
        return true;
    }

    void GetConstantBuffers(ID3D11DeviceContext* ctx, cb::Stage stage, ComArray<ID3D11Buffer, kSlots>& out)
    {
        switch (stage)
        {
        case cb::Stage::VS: ctx->VSGetConstantBuffers(0, kSlots, out.data()); break;
        case cb::Stage::PS: ctx->PSGetConstantBuffers(0, kSlots, out.data()); break;
        case cb::Stage::CS: ctx->CSGetConstantBuffers(0, kSlots, out.data()); break;
        }
    }

    void SetConstantBuffer(ID3D11DeviceContext* ctx, cb::Stage stage, UINT slot, ID3D11Buffer* buffer)
    {
        switch (stage)
        {
        case cb::Stage::VS: ctx->VSSetConstantBuffers(slot, 1, &buffer); break;
        case cb::Stage::PS: ctx->PSSetConstantBuffers(slot, 1, &buffer); break;
        case cb::Stage::CS: ctx->CSSetConstantBuffers(slot, 1, &buffer); break;
        }
    }

    bool IsPerViewSized(ID3D11Buffer* buffer)
    {
        D3D11_BUFFER_DESC desc;
        buffer->GetDesc(&desc);
        return desc.ByteWidth == cb::kPerViewSize && (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER);
    }
}

bool cb::Read(ID3D11Buffer* buffer, Snapshot& out)
{
    if (!buffer)
        return false;
    ComPtr<Tracker> tracker = GetAttached<Tracker>(buffer, kTrackerGuid);
    if (!tracker)
        return false;
    SrwShared guard(tracker->lock);
    out = tracker->shadow;
    return out.size != 0;
}

void cb::OnMap(ID3D11DeviceContext* ctx, ID3D11Resource* resource, D3D11_MAP type, const D3D11_MAPPED_SUBRESOURCE* mapped)
{
    if (t_internalMap || !mapped || (type != D3D11_MAP_WRITE_DISCARD && type != D3D11_MAP_WRITE_NO_OVERWRITE))
        return;
    if (UINT size = TrackedSize(resource))
        t_pendingMap = { ctx, resource, mapped->pData, size };
}

void cb::OnUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* resource)
{
    if (t_pendingMap.resource != resource || t_pendingMap.ctx != ctx || !t_pendingMap.data)
        return;
    StoreShadow(resource, t_pendingMap.data, t_pendingMap.size);
    t_pendingMap = {};
}

void cb::OnUpdateSubresource(ID3D11Resource* resource, const D3D11_BOX* box, const void* data)
{
    if (box || !data)
        return;
    if (UINT size = TrackedSize(resource))
        StoreShadow(resource, data, size);
}

void cb::OnFinishCommandList(ID3D11DeviceContext* ctx)
{
    if (ctx == t_displayCtx)
        t_written.clear();  // the next command list must map the twins again
}

void cb::BeginDisplayResolution(ID3D11DeviceContext* ctx)
{
    t_displayCtx = ctx;
    t_written.clear();
}

bool cb::IsDisplayResolution(ID3D11DeviceContext* ctx)
{
    return ctx && ctx == t_displayCtx;
}

void cb::BindTwins(ID3D11DeviceContext* ctx, std::initializer_list<Stage> stages)
{
    for (Stage stage : stages)
    {
        ComArray<ID3D11Buffer, kSlots> bound;
        GetConstantBuffers(ctx, stage, bound);
        for (UINT slot = 0; slot < kSlots; ++slot)
        {
            ID3D11Buffer* buffer = bound[slot];
            if (!buffer || !IsPerViewSized(buffer))
                continue;

            // Already a twin (bound by an earlier draw of this list): refresh it if the real buffer changed.
            if (ComPtr<TwinLink> link = GetAttached<TwinLink>(buffer, kTwinLinkGuid))
            {
                SrwShared guard(link->lock);
                if (link->real)
                    if (ComPtr<Tracker> tracker = GetAttached<Tracker>(link->real, kTrackerGuid))
                        WriteTwin(ctx, buffer, *tracker.Get());
                continue;
            }

            ComPtr<Tracker> tracker = GetAttached<Tracker>(buffer, kTrackerGuid);
            if (!tracker)
                continue;
            ComPtr<ID3D11Buffer> twin = GetOrCreateTwin(buffer, *tracker.Get());
            if (twin && WriteTwin(ctx, twin.Get(), *tracker.Get()))
                SetConstantBuffer(ctx, stage, slot, twin.Get());
        }
    }
}

void cb::EndDisplayResolution()
{
    ID3D11DeviceContext* ctx = std::exchange(t_displayCtx, nullptr);
    t_written.clear();
    if (!ctx)
        return;

    // The engine's state cache believes the real PerViewCB is bound.
    for (Stage stage : { Stage::VS, Stage::PS, Stage::CS })
    {
        ComArray<ID3D11Buffer, kSlots> bound;
        GetConstantBuffers(ctx, stage, bound);
        for (UINT slot = 0; slot < kSlots; ++slot)
        {
            if (!bound[slot])
                continue;
            if (ComPtr<TwinLink> link = GetAttached<TwinLink>(bound[slot], kTwinLinkGuid))
            {
                SrwShared guard(link->lock);  // keeps the real buffer's tracker from clearing the link meanwhile
                if (link->real)
                    SetConstantBuffer(ctx, stage, slot, link->real);
            }
        }
    }
}
