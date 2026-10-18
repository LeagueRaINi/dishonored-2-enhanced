//! Offline check of the photo composite (`examples/photo_offline.rs`): a stand-in "game"
//! D3D11 device with an engine-like `hdrBuffer` (RGBA16F with alpha, and R11G11B10) and
//! hardware-style depth, the D3D12 device on the same adapter (`rt::device::create`), our
//! image written by D3D12 into a composite slot (`SlotSet`, D3D11-created shared textures;
//! `CopyTextureRegion` from pitched buffers in COMMON -> COPY_DEST -> COMMON, as
//! `PhotoSession::copy_out` does), the CPU-observed fence, then the FOG-pass recording
//! (`Pipe::record_composite`) on a deferred context and the overlay preview on the
//! immediate one. Checks: matched pixels = ours x units (fitted and fixed), every other
//! pixel and the alpha channel untouched, the deferred context's state restored, the
//! preview drawn and letterboxed. Hybrid terms' depth capture (`Pipe::record_depth_export`
//! into a D3D11-created shared R32F texture with a D3D11 UAV, as `hybrid::DepthRequest`):
//! view depth = the test pattern within 0.1 % (0 for sky), state restored.
//!
//! Covers everything except engine memory, the FOG hook itself and the engine's views.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::composite::{FALLBACK_UNITS, Params, Pipe, SlotSet, TOLERANCE};
use crate::ao::{Saved, Slots as StateSlots};
use crate::rt::device::{self, Device, Queue};

fn err(what: &'static str) -> impl Fn(windows::core::Error) -> String {
    move |e| format!("{what}: {e}")
}

/// Image size of the check, region offset inside the engine targets.
const ORIGIN: [u32; 2] = [8, 4];
const PAD: [u32; 2] = [16, 12];
/// Projection depth terms (`ao::engine_projection` with near 0.01, far 7000) and our near.
const NEAR: f32 = 0.01;
const FAR: f32 = 7000.0;
/// Fitted units the test data is built for (game = ours x 2.5 x {2, 0.5} alternating).
const UNITS: f32 = 2.5;

fn depth_terms() -> (f32, f32) {
    // z = depth_mul / (d - depth_add) with the engine's reversed-Z projection (d = n f /
    // ((f - n) z) - n / (f - n)).
    let mul = NEAR * FAR / (FAR - NEAR);
    let add = -NEAR / (FAR - NEAR);
    (mul, add)
}

fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if v == 0.0 {
        return sign;
    }
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7C00;
    }
    sign | ((e as u16) << 10) | ((m >> 13) as u16)
}

fn f16_value(h: u16) -> f32 {
    let e = (h >> 10) & 0x1F;
    let m = f32::from(h & 0x3FF);
    let v = match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => 2f32.powi(i32::from(e) - 15) * (1.0 + m / 1024.0),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

fn small_float(bits: u32, mant: u32) -> f32 {
    let e = (bits >> mant) & 0x1F;
    let m = (bits & ((1 << mant) - 1)) as f32 / (1u32 << mant) as f32;
    match e {
        0 => m * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => 2f32.powi(e as i32 - 15) * (1.0 + m),
    }
}

fn r11g11b10(p: u32) -> [f32; 3] {
    [
        small_float(p & 0x7FF, 6),
        small_float((p >> 11) & 0x7FF, 6),
        small_float(p >> 22, 5),
    ]
}

/// Kind of a test pixel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Match,
    /// Our depth 10 % off.
    Mismatch,
    /// Our ray missed (depth 0).
    Miss,
    /// Game sky (depth 0).
    Sky,
}

fn kind(x: u32, y: u32, size: [u32; 2]) -> Kind {
    let (w, h) = (size[0], size[1]);
    if x < w / 8 {
        Kind::Sky
    } else if y < h / 8 {
        Kind::Miss
    } else if x >= w - w / 8 && y >= h / 2 {
        Kind::Mismatch
    } else {
        Kind::Match
    }
}

fn ours(x: u32, y: u32) -> [f32; 3] {
    [
        0.05 + (x % 7) as f32 * 0.03,
        0.04 + (y % 5) as f32 * 0.05,
        0.02 + ((x + y) % 3) as f32 * 0.04,
    ]
}

/// View depth of a test pixel (metres).
fn view_z(x: u32, y: u32) -> f32 {
    2.0 + (x as f32 * 0.37 + y as f32 * 0.21) % 40.0
}

/// Game colour: ours x UNITS x 2 or x 0.5 on alternating sampled rows (log mean = UNITS).
fn game(x: u32, y: u32) -> [f32; 3] {
    let k = if (y / 4).is_multiple_of(2) { 2.0 } else { 0.5 };
    ours(x, y).map(|v| v * UNITS * k)
}

struct Offline {
    d11: ID3D11Device,
    imm: ID3D11DeviceContext,
    dev: Arc<Device>,
    queue: Queue,
    warp: bool,
    name: String,
}

impl Offline {
    fn new(warp: bool) -> Result<Self, String> {
        let (mut d, mut imm) = (None, None);
        // SAFETY: device creation with valid arguments and out pointers.
        unsafe {
            D3D11CreateDevice(
                None,
                if warp {
                    D3D_DRIVER_TYPE_WARP
                } else {
                    D3D_DRIVER_TYPE_HARDWARE
                },
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
                D3D11_SDK_VERSION,
                Some(&mut d),
                None,
                Some(&mut imm),
            )
        }
        .map_err(err("D3D11CreateDevice"))?;
        let d11: ID3D11Device = d.ok_or("no device")?;
        let dev = Arc::new(device::create(&d11)?);
        let name = {
            let dxgi: windows::Win32::Graphics::Dxgi::IDXGIDevice =
                d11.cast().map_err(|e| e.to_string())?;
            // SAFETY: plain getters.
            let desc =
                unsafe { dxgi.GetAdapter().and_then(|a| a.GetDesc()) }.map_err(err("adapter"))?;
            String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_owned()
        };
        let queue = Queue::new(&dev, "photo offline")?;
        Ok(Self {
            d11,
            imm: imm.ok_or("no context")?,
            dev,
            queue,
            warp,
            name,
        })
    }

    fn texture(
        &self,
        size: [u32; 2],
        format: DXGI_FORMAT,
        bind: i32,
        data: Option<(&[u8], u32)>,
    ) -> Result<ID3D11Texture2D, String> {
        let d = D3D11_TEXTURE2D_DESC {
            Width: size[0],
            Height: size[1],
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind as u32,
            ..Default::default()
        };
        let init = data.map(|(b, pitch)| D3D11_SUBRESOURCE_DATA {
            pSysMem: b.as_ptr().cast(),
            SysMemPitch: pitch,
            SysMemSlicePitch: 0,
        });
        let mut t = None;
        // SAFETY: valid descriptor; the initial data covers the declared size.
        unsafe {
            self.d11
                .CreateTexture2D(&d, init.as_ref().map(|i| i as *const _), Some(&mut t))
        }
        .map_err(err("texture"))?;
        t.ok_or_else(|| "no texture".to_owned())
    }

    /// Raw bytes of a texture (staging copy), `bpp` bytes per texel, tightly packed.
    fn read(&self, t: &ID3D11Texture2D, bpp: usize) -> Result<Vec<u8>, String> {
        let mut d = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: plain getter.
        unsafe { t.GetDesc(&mut d) };
        let sd = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..d
        };
        let mut st = None;
        // SAFETY (whole block): staging creation, copy, map of a resource this thread owns.
        unsafe {
            self.d11
                .CreateTexture2D(&sd, None, Some(&mut st))
                .map_err(err("staging"))?;
            let st: ID3D11Texture2D = st.ok_or("no staging")?;
            self.imm.CopyResource(&st, t);
            let mut m = D3D11_MAPPED_SUBRESOURCE::default();
            self.imm
                .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
                .map_err(err("map"))?;
            let row = d.Width as usize * bpp;
            let mut out = vec![0u8; row * d.Height as usize];
            for y in 0..d.Height as usize {
                std::ptr::copy_nonoverlapping(
                    m.pData.cast::<u8>().add(y * m.RowPitch as usize),
                    out.as_mut_ptr().add(y * row),
                    row,
                );
            }
            self.imm.Unmap(&st, 0);
            Ok(out)
        }
    }

    /// Writes `data` (tightly packed rows of `bpp` bytes per texel) into the D3D12 side of
    /// a shared texture, as `PhotoSession::copy_out` does, and waits on the CPU.
    fn upload12(
        &self,
        tex: &ID3D12Resource,
        size: [u32; 2],
        format: DXGI_FORMAT,
        bpp: u32,
        data: &[u8],
    ) -> Result<(), String> {
        let pitch = (size[0] * bpp).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let bytes = u64::from(pitch) * u64::from(size[1]);
        let d12 = &self.dev.d12;
        // SAFETY (whole block): D3D12 object creation, mapping of an upload buffer of
        // `bytes`, recording and submission on our own queue; every resource outlives the
        // CPU wait at the end.
        unsafe {
            let mut buf: Option<ID3D12Resource> = None;
            d12.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_UPLOAD,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut buf,
            )
            .map_err(err("upload buffer"))?;
            let buf = buf.ok_or("no upload buffer")?;
            let mut p = std::ptr::null_mut();
            buf.Map(0, None, Some(&mut p)).map_err(err("map upload"))?;
            let row = (size[0] * bpp) as usize;
            for y in 0..size[1] as usize {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(y * row),
                    p.cast::<u8>().add(y * pitch as usize),
                    row,
                );
            }
            buf.Unmap(0, None);
            let alloc: ID3D12CommandAllocator = d12
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err("allocator"))?;
            let list: ID3D12GraphicsCommandList = d12
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(err("command list"))?;
            let barrier = |before, after| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(tex),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            };
            list.ResourceBarrier(&[barrier(
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buf),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: size[0],
                            Height: size[1],
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            list.Close().map_err(err("close"))?;
            let v = self
                .queue
                .submit(&[Some(list.cast().map_err(|e| e.to_string())?)])?;
            self.queue.wait(v, 10_000)?;
            // `transmute_copy` did not add references; nothing to release.
            let _ = (&src, &dst);
        }
        Ok(())
    }
}

/// Outcome of one composite variant.
struct Variant {
    name: &'static str,
    format: DXGI_FORMAT,
    units: f32,
}

/// Runs the check; returns the report (Err on the first failed expectation).
pub fn offline_check(warp: bool, size: [u32; 2]) -> Result<String, String> {
    let o = Offline::new(warp)?;
    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "photo composite offline check on {} ({}), image {}x{} at offset {:?} in {}x{} targets",
        o.name,
        if o.warp { "WARP" } else { "hardware" },
        size[0],
        size[1],
        ORIGIN,
        size[0] + PAD[0],
        size[1] + PAD[1]
    );
    let tex = [size[0] + PAD[0], size[1] + PAD[1]];
    let (mul, add) = depth_terms();
    // Our image and depth (D3D12 side of the slot).
    let mut ours_bytes = Vec::with_capacity((size[0] * size[1] * 16) as usize);
    let mut ours_depth = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    for y in 0..size[1] {
        for x in 0..size[0] {
            let c = ours(x, y);
            for v in [c[0], c[1], c[2], 1.0] {
                ours_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let z = view_z(x, y);
            let d = match kind(x, y, size) {
                Kind::Miss => 0.0,
                Kind::Mismatch => NEAR / (z * 1.1),
                Kind::Match | Kind::Sky => NEAR / (z * 1.005),
            };
            ours_depth.extend_from_slice(&d.to_le_bytes());
        }
    }
    let slots = SlotSet::new(&o.dev, size, NEAR)?;
    let t = Instant::now();
    o.upload12(
        &slots.slots[0].hdr.res,
        size,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        16,
        &ours_bytes,
    )?;
    o.upload12(
        &slots.slots[0].depth.res,
        size,
        DXGI_FORMAT_R32_FLOAT,
        4,
        &ours_depth,
    )?;
    slots.publish(0);
    let _ = writeln!(
        rep,
        "D3D12 -> shared slot copy + CPU fence wait: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    // Game depth (R32F hardware-style values) over the whole target.
    let mut depth = Vec::with_capacity((tex[0] * tex[1] * 4) as usize);
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            let inside = x >= ORIGIN[0]
                && y >= ORIGIN[1]
                && x < ORIGIN[0] + size[0]
                && y < ORIGIN[1] + size[1];
            let (ix, iy) = (x.wrapping_sub(ORIGIN[0]), y.wrapping_sub(ORIGIN[1]));
            let d = if !inside || kind(ix, iy, size) == Kind::Sky {
                0.0
            } else {
                mul / view_z(ix, iy) + add
            };
            depth.extend_from_slice(&f32::to_le_bytes(d));
        }
    }
    let zbuf = o.texture(
        tex,
        DXGI_FORMAT_R32_FLOAT,
        D3D11_BIND_SHADER_RESOURCE.0,
        Some((&depth, tex[0] * 4)),
    )?;
    let mut pipe = Pipe::new(&o.d11)?;
    let ours_srv = slots.slots[0].hdr.srv.clone().ok_or("no slot SRV")?;
    let ours_depth_srv = slots.slots[0].depth.srv.clone().ok_or("no slot SRV")?;
    let zres: ID3D11Resource = zbuf.cast().map_err(|e| e.to_string())?;
    let game_depth = pipe.srv(&zres, true)?;
    let mut counts = [0usize; 4];
    for y in 0..size[1] {
        for x in 0..size[0] {
            counts[kind(x, y, size) as usize] += 1;
        }
    }
    let _ = writeln!(
        rep,
        "pixels: match {}, ours 10 % deeper {}, our miss {}, game sky {}",
        counts[0], counts[1], counts[2], counts[3]
    );
    for v in [
        Variant {
            name: "RGBA16F, fitted units",
            format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            units: 0.0,
        },
        Variant {
            name: "RGBA16F, photo_units = 1.7",
            format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            units: 1.7,
        },
        Variant {
            name: "R11G11B10 (game format), fitted units",
            format: DXGI_FORMAT_R11G11B10_FLOAT,
            units: 0.0,
        },
    ] {
        let line = run_variant(
            &o,
            &mut pipe,
            &v,
            size,
            tex,
            &game_depth,
            &ours_srv,
            &ours_depth_srv,
        )?;
        let _ = writeln!(rep, "{line}");
    }
    // Preview (overlay display) into an RGBA8 "back buffer" of another aspect.
    let bb_size = [size[0] * 2, size[1] * 3];
    let bb = o.texture(
        bb_size,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D11_BIND_RENDER_TARGET.0,
        None,
    )?;
    let mut rtv = None;
    // SAFETY: default view of a render target texture.
    unsafe { o.d11.CreateRenderTargetView(&bb, None, Some(&mut rtv)) }.map_err(err("RTV"))?;
    let rtv = rtv.ok_or("no RTV")?;
    pipe.record_preview(&o.imm, &ours_srv, size, &rtv, bb_size)?;
    let px = o.read(&bb, 4)?;
    let at = |x: u32, y: u32| {
        let i = ((y * bb_size[0] + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    // Letterbox: width-limited (scale 2), image centred vertically.
    let top = (bb_size[1] - size[1] * 2) / 2;
    if at(bb_size[0] / 2, top / 2) != [0, 0, 0] || at(bb_size[0] / 2, bb_size[1] / 2) == [0, 0, 0] {
        return Err(format!(
            "preview: letterbox {:?} / centre {:?}",
            at(bb_size[0] / 2, top / 2),
            at(bb_size[0] / 2, bb_size[1] / 2)
        ));
    }
    let _ = writeln!(
        rep,
        "preview: tone-mapped image letterboxed into {}x{} (bars black, centre {:?})",
        bb_size[0],
        bb_size[1],
        at(bb_size[0] / 2, bb_size[1] / 2)
    );
    // Hybrid terms: the game's view depth of the region into a shared R32F texture.
    let line = depth_export(&o, &mut pipe, size, &game_depth)?;
    let _ = writeln!(rep, "{line}");
    // The slot protocol: the published slot is never writable, the other one only after
    // REUSE_FRAMES FOG passes (none ran here).
    if slots.writable().is_some() {
        return Err("a slot is writable right after publishing (no FOG pass ran)".into());
    }
    let _ = writeln!(
        rep,
        "slot protocol: published slot not writable, the other not before {} FOG passes",
        super::composite::REUSE_FRAMES
    );
    Ok(rep)
}

/// The hybrid depth capture: `record_depth_export` on a deferred context with some state,
/// into a D3D11-created shared R32F texture (UAV), executed, read back and compared.
fn depth_export(
    o: &Offline,
    pipe: &mut Pipe,
    size: [u32; 2],
    game_depth: &ID3D11ShaderResourceView,
) -> Result<String, String> {
    let out = device::shared_texture(&o.dev, DXGI_FORMAT_R32_FLOAT, size[0], size[1], true, false)?;
    let uav = out.uav.clone().ok_or("no UAV on the shared texture")?;
    // SAFETY: plain getter of the view's resource.
    let tex: ID3D11Texture2D = unsafe { uav.GetResource() }
        .map_err(err("UAV resource"))?
        .cast()
        .map_err(|e| e.to_string())?;
    let mut def = None;
    // SAFETY: valid out pointer.
    unsafe { o.d11.CreateDeferredContext(0, Some(&mut def)) }.map_err(err("deferred context"))?;
    let def: ID3D11DeviceContext = def.ok_or("no deferred context")?;
    // SAFETY: binds of live objects on our deferred context (the engine's own state).
    unsafe {
        def.CSSetShaderResources(2, Some(&[Some(game_depth.clone())]));
        def.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 1.0,
            TopLeftY: 2.0,
            Width: 3.0,
            Height: 4.0,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]));
    }
    let wide = StateSlots {
        srv: 16,
        cb: 8,
        sampler: 4,
        uav: 8,
    };
    // SAFETY: getters on our deferred context.
    let state0 = unsafe { Saved::capture(&def, wide) };
    let (mul, add) = depth_terms();
    let params = Params {
        origin: [ORIGIN[0] as i32, ORIGIN[1] as i32],
        size,
        depth_mul: mul,
        depth_add: add,
        ..Params::default()
    };
    pipe.record_depth_export(&def, &params, game_depth, &uav)?;
    // SAFETY: getters on our deferred context.
    let state1 = unsafe { Saved::capture(&def, wide) };
    let diff = state0.diff(&state1);
    if !diff.is_empty() {
        return Err(format!(
            "depth export: state not restored: {}",
            diff.join(", ")
        ));
    }
    // SAFETY: finishing our deferred context and executing it on the immediate one.
    unsafe {
        let mut list = None;
        def.FinishCommandList(false, Some(&mut list))
            .map_err(err("FinishCommandList"))?;
        o.imm
            .ExecuteCommandList(&list.ok_or("no command list")?, false);
    }
    let px = o.read(&tex, 4)?;
    let (mut surface, mut sky, mut worst) = (0usize, 0usize, 0.0f32);
    for y in 0..size[1] {
        for x in 0..size[0] {
            let i = ((y * size[0] + x) * 4) as usize;
            let z = f32::from_le_bytes([px[i], px[i + 1], px[i + 2], px[i + 3]]);
            if kind(x, y, size) == Kind::Sky {
                if z != 0.0 {
                    return Err(format!(
                        "depth export: sky at ({x}, {y}) is {z}, expected 0"
                    ));
                }
                sky += 1;
                continue;
            }
            let want = view_z(x, y);
            let rel = (z - want).abs() / want;
            worst = worst.max(rel);
            if rel > 1e-3 {
                return Err(format!("depth export: ({x}, {y}) is {z}, expected {want}"));
            }
            surface += 1;
        }
    }
    Ok(format!(
        "hybrid depth capture: {surface} view depths within {:.2e} relative, {sky} sky pixels 0, state restored",
        worst
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    o: &Offline,
    pipe: &mut Pipe,
    v: &Variant,
    size: [u32; 2],
    tex: [u32; 2],
    game_depth: &ID3D11ShaderResourceView,
    ours_srv: &ID3D11ShaderResourceView,
    ours_depth_srv: &ID3D11ShaderResourceView,
) -> Result<String, String> {
    let f16_target = v.format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    // Game colour: the test pattern inside the region, a marker outside; alpha 0.25.
    let outside = [0.75f32, 0.5, 0.25];
    let hdr = if f16_target {
        let mut b = Vec::with_capacity((tex[0] * tex[1] * 8) as usize);
        for y in 0..tex[1] {
            for x in 0..tex[0] {
                let c = if x >= ORIGIN[0]
                    && y >= ORIGIN[1]
                    && x < ORIGIN[0] + size[0]
                    && y < ORIGIN[1] + size[1]
                {
                    game(x - ORIGIN[0], y - ORIGIN[1])
                } else {
                    outside
                };
                for h in [f16(c[0]), f16(c[1]), f16(c[2]), f16(0.25)] {
                    b.extend_from_slice(&h.to_le_bytes());
                }
            }
        }
        o.texture(
            tex,
            v.format,
            D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0,
            Some((&b, tex[0] * 8)),
        )?
    } else {
        o.texture(
            tex,
            v.format,
            D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0,
            None,
        )?
    };
    let res: ID3D11Resource = hdr.cast().map_err(|e| e.to_string())?;
    if !f16_target {
        // R11G11B10 cannot be filled from the CPU here: one constant colour.
        let rtv = pipe.rtv(&res)?;
        // SAFETY: clear of our own target.
        unsafe {
            o.imm
                .ClearRenderTargetView(&rtv, &[outside[0], outside[1], outside[2], 1.0])
        };
    }
    let before = o.read(&hdr, if f16_target { 8 } else { 4 })?;
    let game_hdr = pipe.srv(&res, false)?;
    let target = pipe.rtv(&res)?;
    // A deferred context with some state of its own, as an engine pass would have.
    let mut def = None;
    // SAFETY: valid out pointer.
    unsafe { o.d11.CreateDeferredContext(0, Some(&mut def)) }.map_err(err("deferred context"))?;
    let def: ID3D11DeviceContext = def.ok_or("no deferred context")?;
    // SAFETY: binds of live objects on our deferred context.
    unsafe {
        def.PSSetShaderResources(2, Some(&[Some(game_depth.clone())]));
        def.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 1.0,
            TopLeftY: 2.0,
            Width: 3.0,
            Height: 4.0,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }]));
    }
    let wide = StateSlots {
        srv: 16,
        cb: 8,
        sampler: 4,
        uav: 8,
    };
    // SAFETY: getters on our deferred context.
    let state0 = unsafe { Saved::capture(&def, wide) };
    let (mul, add) = depth_terms();
    let params = Params {
        origin: [ORIGIN[0] as i32, ORIGIN[1] as i32],
        size,
        depth_mul: mul,
        depth_add: add,
        znear: NEAR,
        units: v.units,
        fallback: FALLBACK_UNITS,
        tolerance: TOLERANCE,
        dst_size: [0.0; 2],
    };
    let t = Instant::now();
    pipe.record_composite(
        &def,
        &params,
        ours_srv,
        ours_depth_srv,
        game_depth,
        &game_hdr,
        &target,
        [ORIGIN[0], ORIGIN[1], size[0], size[1]],
    )?;
    let record_ms = t.elapsed().as_secs_f64() * 1e3;
    // SAFETY: getters on our deferred context.
    let state1 = unsafe { Saved::capture(&def, wide) };
    let diff = state0.diff(&state1);
    if !diff.is_empty() {
        return Err(format!(
            "{}: state not restored: {}",
            v.name,
            diff.join(", ")
        ));
    }
    // SAFETY: finishing our deferred context and executing it on the immediate one.
    unsafe {
        let mut list = None;
        def.FinishCommandList(false, Some(&mut list))
            .map_err(err("FinishCommandList"))?;
        o.imm
            .ExecuteCommandList(&list.ok_or("no command list")?, false);
    }
    let after = o.read(&hdr, if f16_target { 8 } else { 4 })?;
    let px = |b: &[u8], x: u32, y: u32| -> ([f32; 3], Option<f32>) {
        if f16_target {
            let i = ((y * tex[0] + x) * 8) as usize;
            let h = |k: usize| f16_value(u16::from_le_bytes([b[i + 2 * k], b[i + 2 * k + 1]]));
            ([h(0), h(1), h(2)], Some(h(3)))
        } else {
            let i = ((y * tex[0] + x) * 4) as usize;
            (
                r11g11b10(u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])),
                None,
            )
        }
    };
    // Expected units: the fit over sampled matched pixels (exact for the RGBA16F pattern).
    let units = if v.units > 0.0 {
        v.units
    } else if f16_target {
        UNITS
    } else {
        // Constant game colour: geometric mean of game / ours luminance over the samples.
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let (mut s, mut n) = (0.0f64, 0u32);
        for y in (0..size[1]).step_by(4) {
            for x in (0..size[0]).step_by(4) {
                if kind(x, y, size) == Kind::Match {
                    let g = px(&before, x + ORIGIN[0], y + ORIGIN[1]).0;
                    s += f64::from((lum(g) / lum(ours(x, y))).log2());
                    n += 1;
                }
            }
        }
        2f64.powf(s / f64::from(n)) as f32
    };
    let (mut written, mut kept, mut worst) = (0usize, 0usize, 0.0f32);
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            let (a, alpha_a) = px(&after, x, y);
            let (b, alpha_b) = px(&before, x, y);
            if alpha_a != alpha_b {
                return Err(format!(
                    "{}: alpha changed at ({x}, {y}): {alpha_b:?} -> {alpha_a:?}",
                    v.name
                ));
            }
            let inside = x >= ORIGIN[0]
                && y >= ORIGIN[1]
                && x < ORIGIN[0] + size[0]
                && y < ORIGIN[1] + size[1];
            let k = inside.then(|| kind(x - ORIGIN[0], y - ORIGIN[1], size));
            if k == Some(Kind::Match) {
                let want = ours(x - ORIGIN[0], y - ORIGIN[1]).map(|c| c * units);
                for i in 0..3 {
                    let rel = (a[i] - want[i]).abs() / want[i].max(1e-3);
                    worst = worst.max(rel);
                    if rel > 0.03 {
                        return Err(format!(
                            "{}: ({x}, {y}) wrote {a:?}, expected {want:?} (units {units})",
                            v.name
                        ));
                    }
                }
                written += 1;
            } else {
                if a != b {
                    return Err(format!(
                        "{}: ({x}, {y}) ({k:?}) changed {b:?} -> {a:?}",
                        v.name
                    ));
                }
                kept += 1;
            }
        }
    }
    Ok(format!(
        "{}: {written} pixels composited (units {units:.3}, worst relative error {:.4}), {kept} kept exactly, alpha unchanged{}, state restored, recorded in {record_ms:.2} ms",
        v.name,
        worst,
        if f16_target { "" } else { " (no alpha)" }
    ))
}
