//! Offline check of the FFX upscaler bindings (`dinput8::ffx`), no game involved.
//!
//! Creates a D3D12 device on the default adapter, lists the upscaler providers, creates a
//! context with each one, then runs N frames of synthetic input (moving HDR pattern,
//! reversed-Z depth, matching motion vectors, 1x1 exposure = 1.0 or auto-exposure)
//! through the newest provider: once render -> display, once native display -> display.
//! Reports GPU time per dispatch (timestamp queries) and output statistics.
//!
//! ```text
//! cargo run -p void-inject --release --example fsr_check -- \
//!     [--ffx-dir <dir>] [--render WxH] [--display WxH] [--frames N] [--auto-exposure]
//!     [--debug-checker]
//! ```

use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::process::ExitCode;

use dinput8::ffx::{self, Loader, Provider, Res, UpscaleDispatch, Upscaler, UpscalerDesc};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::core::Interface;

struct Args {
    ffx_dir: PathBuf,
    render: [u32; 2],
    display: [u32; 2],
    frames: u32,
    auto_exposure: bool,
    debug_checker: bool,
}

fn parse_size(s: &str) -> Result<[u32; 2], String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("size {s:?}: expected WxH"))?;
    let p = |v: &str| {
        v.parse::<u32>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("size {s:?}: bad number {v:?}"))
    };
    Ok([p(w)?, p(h)?])
}

fn parse_args() -> Result<Args, String> {
    // void-rs/crates/void-inject -> ../../.. = the game directory.
    let mut a = Args {
        ffx_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        render: [1720, 720],
        display: [3440, 1440],
        frames: 16,
        auto_exposure: false,
        debug_checker: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--ffx-dir" => a.ffx_dir = PathBuf::from(value()?),
            "--render" => a.render = parse_size(&value()?)?,
            "--display" => a.display = parse_size(&value()?)?,
            "--frames" => {
                a.frames = value()?
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or("--frames: expected a positive integer")?
            }
            "--auto-exposure" => a.auto_exposure = true,
            "--debug-checker" => a.debug_checker = true,
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    if a.render[0] > a.display[0] || a.render[1] > a.display[1] {
        return Err("--render must not exceed --display".into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fsr_check: FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let gpu = Gpu::new()?;
    // `absolute` (GetFullPathNameW) resolves `..` without the `\\?\` prefix
    // `canonicalize` would add.
    let ffx_dir = std::path::absolute(&args.ffx_dir).map_err(|e| e.to_string())?;
    println!("ffx dir: {}", ffx_dir.display());
    let loader = Loader::load(&ffx_dir)?;

    let providers = ffx::upscaler_providers(&loader, &gpu.device)?;
    println!("upscaler providers ({}):", providers.len());
    for p in &providers {
        println!("  id {:#018x}  {}", p.id, p.version);
    }
    let newest = providers
        .iter()
        .max_by_key(|p| version_key(&p.version))
        .ok_or("no upscaler provider")?
        .clone();
    println!("newest: {} ({:#x})", newest.version, newest.id);

    let mut failed = false;
    println!(
        "context creation per provider ({}x{} -> {}x{}):",
        args.render[0], args.render[1], args.display[0], args.display[1]
    );
    for p in &providers {
        match Upscaler::new(&loader, &gpu.device, &desc(&args, Some(p.id))) {
            Ok(u) => println!(
                "  {:<12} ok, context reports {} ({:#x})",
                p.version,
                u.provider().version,
                u.provider().id
            ),
            Err(e) => {
                failed = true;
                println!("  {:<12} FAILED: {e}", p.version);
            }
        }
    }

    let mut cases = vec![(args.render, args.display)];
    if args.render != args.display {
        cases.push((args.display, args.display));
    }
    for (render, display) in cases {
        if let Err(e) = run_case(&gpu, &loader, &args, &newest, render, display) {
            failed = true;
            println!("  FAILED: {e}");
        }
    }
    if failed {
        Err("see above".into())
    } else {
        println!("fsr_check: OK");
        Ok(())
    }
}

/// Numeric key of a version name like "4.1.1" (non-digit separators ignored).
fn version_key(name: &str) -> Vec<u32> {
    name.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect()
}

/// d2enhanced's context choices (HDR linear colour, reversed infinite Z, dynamic
/// resolution with max render = display).
fn desc(args: &Args, provider: Option<u64>) -> UpscalerDesc {
    UpscalerDesc {
        max_render: args.display,
        max_upscale: args.display,
        hdr: true,
        depth_inverted: true,
        depth_infinite: true,
        auto_exposure: args.auto_exposure,
        dynamic_resolution: true,
        provider,
        debug_checker: args.debug_checker,
    }
}

// ---------------------------------------------------------------------------------------

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: std::cell::Cell<u64>,
}

impl Gpu {
    fn new() -> Result<Self, String> {
        // SAFETY (whole function): plain D3D12/DXGI object creation with valid descriptors
        // and out pointers.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| format!("CreateDXGIFactory2: {e}"))?;
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapters1(0)
                .map_err(|e| format!("EnumAdapters1(0): {e}"))?;
            let ad = adapter.GetDesc1().map_err(|e| e.to_string())?;
            let name_len = ad.Description.iter().position(|&c| c == 0).unwrap_or(128);
            println!(
                "adapter: {} (vendor {:#06x}, device {:#06x}, {} MiB VRAM)",
                String::from_utf16_lossy(&ad.Description[..name_len]),
                ad.VendorId,
                ad.DeviceId,
                ad.DedicatedVideoMemory >> 20
            );
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
                .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
            let device = device.ok_or("no device")?;
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(|e| format!("queue: {e}"))?;
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|e| format!("allocator: {e}"))?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(|e| format!("command list: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("fence: {e}"))?;
            Ok(Self {
                device,
                queue,
                allocator,
                list,
                fence,
                fence_value: 0.into(),
            })
        }
    }

    /// Closes and submits the list, waits for the GPU, reopens the list.
    fn submit(&self) -> Result<(), String> {
        let v = self.fence_value.get() + 1;
        self.fence_value.set(v);
        // SAFETY: the list is open and records only live resources; after the blocking
        // wait (null event) the GPU is idle, so resetting allocator and list is valid.
        unsafe {
            self.list.Close().map_err(|e| format!("Close: {e}"))?;
            let list: ID3D12CommandList = self.list.cast().map_err(|e| e.to_string())?;
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.queue
                .Signal(&self.fence, v)
                .map_err(|e| format!("Signal: {e}"))?;
            self.fence
                .SetEventOnCompletion(v, HANDLE::default())
                .map_err(|e| format!("wait: {e}"))?;
            self.device
                .GetDeviceRemovedReason()
                .map_err(|e| format!("device removed: {e}"))?;
            self.allocator.Reset().map_err(|e| e.to_string())?;
            self.list
                .Reset(&self.allocator, None)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn committed(
        &self,
        heap: D3D12_HEAP_TYPE,
        desc: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, String> {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: valid descriptors and out pointer.
        unsafe {
            self.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                desc,
                state,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("CreateCommittedResource {:?}: {e}", desc.Format))?;
        res.ok_or_else(|| "no resource".into())
    }

    fn texture(
        &self,
        format: DXGI_FORMAT,
        [w, h]: [u32; 2],
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, String> {
        self.committed(
            D3D12_HEAP_TYPE_DEFAULT,
            &tex_desc(format, [w, h], flags),
            state,
        )
    }

    fn buffer(&self, heap: D3D12_HEAP_TYPE, size: u64) -> Result<ID3D12Resource, String> {
        let state = if heap == D3D12_HEAP_TYPE_UPLOAD {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            D3D12_RESOURCE_STATE_COPY_DEST
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        self.committed(heap, &desc, state)
    }

    /// Placed footprint of a single-subresource texture at `offset`, plus its byte size.
    fn footprint(
        &self,
        tex: &ID3D12Resource,
        offset: u64,
    ) -> (D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64) {
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: one subresource, out pointers valid.
        unsafe {
            let desc = tex.GetDesc();
            self.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                offset,
                Some(&mut fp),
                None,
                None,
                Some(&mut total),
            );
        }
        (fp, total)
    }

    fn copy_buffer_to_texture(
        &self,
        buf: &ID3D12Resource,
        fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        tex: &ID3D12Resource,
    ) {
        let dst = loc_sub(tex);
        let src = loc_fp(buf, fp);
        // SAFETY: both resources outlive the submission; locations match their kinds.
        unsafe { self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }

    fn copy_texture_to_buffer(
        &self,
        tex: &ID3D12Resource,
        buf: &ID3D12Resource,
        fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) {
        let dst = loc_fp(buf, fp);
        let src = loc_sub(tex);
        // SAFETY: as above.
        unsafe { self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }

    fn barrier(
        &self,
        res: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        let b = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(res.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        // SAFETY: recording a transition on a live resource; the union holds the
        // transition built above, whose reference is released afterwards.
        unsafe {
            self.list.ResourceBarrier(std::slice::from_ref(&b));
            let t = ManuallyDrop::into_inner(b.Anonymous.Transition);
            drop(ManuallyDrop::into_inner(t.pResource));
        }
    }
}

fn tex_desc(
    format: DXGI_FORMAT,
    [w, h]: [u32; 2],
    flags: D3D12_RESOURCE_FLAGS,
) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(w),
        Height: h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    }
}

fn loc_sub(tex: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(tex.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    }
}

fn loc_fp(
    buf: &ID3D12Resource,
    fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(buf.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: fp,
        },
    }
}

/// Maps a CPU-visible buffer for the duration of `f`.
fn with_mapped<R>(
    buf: &ID3D12Resource,
    write: bool,
    f: impl FnOnce(*mut u8) -> R,
) -> Result<R, String> {
    let mut ptr = std::ptr::null_mut();
    let none = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: subresource 0 of a buffer on an upload/readback heap; `f` stays within the
    // buffer (callers index by footprints computed for it). An empty read range tells the
    // driver nothing is read back from upload heaps.
    unsafe {
        buf.Map(0, if write { Some(&none) } else { None }, Some(&mut ptr))
            .map_err(|e| format!("Map: {e}"))?;
        let r = f(ptr.cast());
        buf.Unmap(0, if write { None } else { Some(&none) });
        Ok(r)
    }
}

// ---------------------------------------------------------------------------------------

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x0080_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    // Round to nearest (carry into the exponent is the correct result).
    sign | (((e as u32) << 10) + ((man + 0x1000) >> 13)) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Scene radiance at scene-space point (x, y) in render pixels: smooth waves plus a grid
/// of bright HDR squares and thin lines (aliasing-prone detail for the upscaler).
fn scene(x: f32, y: f32) -> [f32; 3] {
    let base = 0.35 + 0.25 * (x * 0.071).sin() * (y * 0.053).cos();
    let cell = ((x / 48.0).floor() as i64 + (y / 48.0).floor() as i64) & 1;
    let hot = if cell == 0 && (x.rem_euclid(48.0) < 12.0) && (y.rem_euclid(48.0) < 12.0) {
        4.0
    } else {
        0.0
    };
    let line = if (x * 0.37 + y * 0.11).rem_euclid(9.0) < 0.6 {
        0.8
    } else {
        0.0
    };
    [base + hot + line, base * 0.8 + hot * 0.5, base * 0.6 + line]
}

fn run_case(
    gpu: &Gpu,
    loader: &Loader,
    args: &Args,
    provider: &Provider,
    render: [u32; 2],
    display: [u32; 2],
) -> Result<(), String> {
    println!(
        "case {}x{} -> {}x{}, {} frames, provider {}, exposure {}:",
        render[0],
        render[1],
        display[0],
        display[1],
        args.frames,
        provider.version,
        if args.auto_exposure {
            "auto"
        } else {
            "1x1 R32F = 1.0"
        }
    );
    let t0 = std::time::Instant::now();
    let mut up = Upscaler::new(loader, &gpu.device, &desc(args, Some(provider.id)))?;
    println!(
        "  context: {} ({:#x}), created in {:.1} ms",
        up.provider().version,
        up.provider().id,
        t0.elapsed().as_secs_f64() * 1e3
    );
    let phases = up.jitter_phase_count(render[0], display[0])?;
    println!("  jitter phase count: {phases}");

    let none = D3D12_RESOURCE_FLAG_NONE;
    let copy_dest = D3D12_RESOURCE_STATE_COPY_DEST;
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let color = gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, render, none, copy_dest)?;
    let depth = gpu.texture(DXGI_FORMAT_R32_FLOAT, render, none, copy_dest)?;
    let mv = gpu.texture(DXGI_FORMAT_R16G16_FLOAT, render, none, copy_dest)?;
    let exposure = gpu.texture(DXGI_FORMAT_R32_FLOAT, [1, 1], none, copy_dest)?;
    let output = gpu.texture(
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        display,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        uav,
    )?;

    // One upload buffer: depth | mv | exposure | colour frame 0..N (512-byte aligned
    // placements). All frames are prepared up front and recorded into one command list,
    // so the dispatches run back to back at full clocks (per-frame CPU stalls let the
    // GPU clock down and inflate the timestamps).
    let align = |v: u64| v.next_multiple_of(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT));
    let (fp_depth, sz_d) = gpu.footprint(&depth, 0);
    let (fp_mv, sz_m) = gpu.footprint(&mv, align(sz_d));
    let (fp_exp, sz_e) = gpu.footprint(&exposure, align(fp_mv.Offset + sz_m));
    let color_base = align(fp_exp.Offset + sz_e);
    let (fp_color0, sz_c) = gpu.footprint(&color, color_base);
    let color_stride = align(sz_c);
    let fp_color = |frame: u32| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        Offset: color_base + u64::from(frame) * color_stride,
        ..fp_color0
    };
    let upload = gpu.buffer(
        D3D12_HEAP_TYPE_UPLOAD,
        color_base + u64::from(args.frames) * color_stride,
    )?;

    // Content scrolls left by `speed` render pixels per frame, so the scene point now at
    // pixel x was at x + speed: motion vector (current -> previous) = +speed px, stored
    // in UV units and scaled back by render size.
    let speed = 1.5f32;
    let [rw, rh] = render.map(|v| v as usize);
    let near = 0.1f32;
    with_mapped(&upload, true, |p| {
        // SAFETY: every row write stays inside its footprint within the upload buffer.
        unsafe {
            for y in 0..rh {
                let drow = p
                    .add(fp_depth.Offset as usize + y * fp_depth.Footprint.RowPitch as usize)
                    .cast::<f32>();
                let mrow = p
                    .add(fp_mv.Offset as usize + y * fp_mv.Footprint.RowPitch as usize)
                    .cast::<u16>();
                for x in 0..rw {
                    // Reversed infinite Z: depth = near / view_z, view z 10..40 units.
                    let z = 10.0 + 30.0 * (y as f32 / rh as f32);
                    drow.add(x).write_unaligned(near / z);
                    mrow.add(2 * x)
                        .write_unaligned(f32_to_f16(speed / render[0] as f32));
                    mrow.add(2 * x + 1).write_unaligned(0);
                }
            }
            p.add(fp_exp.Offset as usize)
                .cast::<f32>()
                .write_unaligned(1.0);
        }
    })?;

    // Timestamps: 2 per frame.
    let n_ts = 2 * args.frames;
    let mut heap: Option<ID3D12QueryHeap> = None;
    // SAFETY: valid descriptor and out pointer.
    unsafe {
        gpu.device.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: n_ts,
                NodeMask: 0,
            },
            &mut heap,
        )
    }
    .map_err(|e| format!("query heap: {e}"))?;
    let heap = heap.ok_or("no query heap")?;
    let ts_buf = gpu.buffer(D3D12_HEAP_TYPE_READBACK, u64::from(n_ts) * 8)?;
    // SAFETY: plain query on a live queue.
    let ts_freq = unsafe { gpu.queue.GetTimestampFrequency() }.map_err(|e| e.to_string())?;

    let mut input_sum = 0f64;
    let mut input_count = 0u64;
    let mut jitters = Vec::new();
    for frame in 0..args.frames {
        let index = (frame as i32) % phases.max(1);
        let jitter = up.jitter_offset(index, phases)?;
        jitters.push(jitter);
        let shift = speed * frame as f32;
        let fp = fp_color(frame);
        with_mapped(&upload, true, |p| {
            // SAFETY: row writes stay inside this frame's colour footprint.
            unsafe {
                for y in 0..rh {
                    let row = p
                        .add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize)
                        .cast::<u16>();
                    for x in 0..rw {
                        // Jittered sample: the projection jitter moves the image by
                        // +jitter px, so the pixel centre sees scene point centre - jitter.
                        let sx = x as f32 + 0.5 - jitter[0] + shift;
                        let sy = y as f32 + 0.5 - jitter[1];
                        let c = scene(sx, sy);
                        if frame + 1 == args.frames {
                            input_sum += f64::from(c[0] + c[1] + c[2]);
                            input_count += 3;
                        }
                        for (i, v) in c.iter().chain(&[1.0]).enumerate() {
                            row.add(4 * x + i).write_unaligned(f32_to_f16(*v));
                        }
                    }
                }
            }
        })?;
    }
    println!(
        "  jitter[0..{}]: {}",
        jitters.len().min(4),
        jitters
            .iter()
            .take(4)
            .map(|j| format!("({:+.4}, {:+.4})", j[0], j[1]))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let mut cpu_record_ms = Vec::new();
    for (frame, &jitter) in (0..args.frames).zip(&jitters) {
        if frame == 0 {
            gpu.copy_buffer_to_texture(&upload, fp_depth, &depth);
            gpu.copy_buffer_to_texture(&upload, fp_mv, &mv);
            gpu.copy_buffer_to_texture(&upload, fp_exp, &exposure);
            for r in [&depth, &mv, &exposure] {
                gpu.barrier(r, copy_dest, read);
            }
        } else {
            gpu.barrier(&color, read, copy_dest);
        }
        gpu.copy_buffer_to_texture(&upload, fp_color(frame), &color);
        gpu.barrier(&color, copy_dest, read);

        // SAFETY: timestamp queries within the heap's range.
        unsafe {
            gpu.list
                .EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * frame)
        };
        let rec = std::time::Instant::now();
        up.dispatch(
            &gpu.list,
            &UpscaleDispatch {
                color: Res {
                    resource: &color,
                    state: read,
                },
                depth: Res {
                    resource: &depth,
                    state: read,
                },
                motion_vectors: Res {
                    resource: &mv,
                    state: read,
                },
                exposure: (!args.auto_exposure).then_some(Res {
                    resource: &exposure,
                    state: read,
                }),
                reactive: None,
                transparency_composition: None,
                output: Res {
                    resource: &output,
                    state: uav,
                },
                jitter,
                motion_vector_scale: [render[0] as f32, render[1] as f32],
                render_size: render,
                upscale_size: display,
                sharpening: None,
                frame_time_delta_ms: 16.7,
                pre_exposure: 1.0,
                reset: frame == 0,
                camera_near: near,
                camera_far: 1.0e6,
                camera_fov_y: 1.2,
                view_space_to_meters: 1.0,
                debug_view: false,
            },
        )?;
        // SAFETY: as above.
        unsafe {
            gpu.list
                .EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * frame + 1)
        };
        cpu_record_ms.push(rec.elapsed().as_secs_f64() * 1e3);
    }
    // SAFETY: resolving the written range into a readback buffer of n_ts u64s.
    unsafe {
        gpu.list
            .ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, n_ts, &ts_buf, 0)
    };
    gpu.submit()?;

    let ticks: Vec<u64> = with_mapped(&ts_buf, false, |p| {
        // SAFETY: the buffer holds n_ts u64 timestamps resolved above.
        (0..n_ts as usize)
            .map(|i| unsafe { p.cast::<u64>().add(i).read_unaligned() })
            .collect()
    })?;
    let ms: Vec<f64> = ticks
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[a, b]| b.wrapping_sub(*a) as f64 * 1e3 / ts_freq as f64)
        .collect();
    println!(
        "  GPU ms per dispatch: [{}]",
        ms.iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // The GPU starts the batch from idle clocks: the first frames run slower, so the
    // second half is reported separately as the steady-state figure.
    let summary = |from: usize| {
        let mut v: Vec<f64> = ms[from..].to_vec();
        v.sort_by(f64::total_cmp);
        format!(
            "frames {}..{}: p50 {:.3}, min {:.3}, max {:.3}",
            from + 1,
            ms.len(),
            v[v.len() / 2],
            v[0],
            v[v.len() - 1]
        )
    };
    println!(
        "  GPU ms: first {:.3}; {}",
        ms[0],
        summary(1.min(ms.len() - 1))
    );
    println!("  GPU ms steady: {}", summary(ms.len() / 2));
    let mut cpu = cpu_record_ms.clone();
    cpu.sort_by(f64::total_cmp);
    println!(
        "  CPU ms per ffxDispatch call p50 {:.3}",
        cpu[cpu.len() / 2]
    );

    // Read back the last output.
    let (fp_out, sz_out) = gpu.footprint(&output, 0);
    let rb = gpu.buffer(D3D12_HEAP_TYPE_READBACK, sz_out)?;
    gpu.barrier(&output, uav, D3D12_RESOURCE_STATE_COPY_SOURCE);
    gpu.copy_texture_to_buffer(&output, &rb, fp_out);
    gpu.barrier(&output, D3D12_RESOURCE_STATE_COPY_SOURCE, uav);
    gpu.submit()?;
    let [dw, dh] = display.map(|v| v as usize);
    let stats = with_mapped(&rb, false, |p| {
        let mut s = Stats::default();
        for y in 0..dh {
            // SAFETY: rows within the output footprint of the readback buffer.
            let row = unsafe { p.add(y * fp_out.Footprint.RowPitch as usize).cast::<u16>() };
            for x in 0..dw {
                for c in 0..3 {
                    // SAFETY: as above, 4 u16 per texel.
                    let v = f16_to_f32(unsafe { row.add(4 * x + c).read_unaligned() });
                    s.add(v);
                }
            }
        }
        s
    })?;
    println!(
        "  input (last frame) RGB mean {:.4}",
        input_sum / input_count.max(1) as f64
    );
    println!(
        "  output RGB: min {:.4}, max {:.4}, mean {:.4}, NaN {}, Inf {}, zero texels-channels {} of {}",
        stats.min,
        stats.max,
        stats.sum / stats.finite.max(1) as f64,
        stats.nan,
        stats.inf,
        stats.zero,
        stats.finite + stats.nan + stats.inf
    );
    if stats.nan > 0 || stats.inf > 0 {
        return Err(format!(
            "{} NaN / {} Inf in the output",
            stats.nan, stats.inf
        ));
    }
    if stats.max <= 0.0 {
        return Err("output is black".into());
    }
    Ok(())
}

#[derive(Default)]
struct Stats {
    min: f32,
    max: f32,
    sum: f64,
    finite: u64,
    nan: u64,
    inf: u64,
    zero: u64,
}

impl Stats {
    fn add(&mut self, v: f32) {
        if v.is_nan() {
            self.nan += 1;
        } else if v.is_infinite() {
            self.inf += 1;
        } else {
            if self.finite == 0 {
                self.min = v;
                self.max = v;
            }
            self.min = self.min.min(v);
            self.max = self.max.max(v);
            self.sum += f64::from(v);
            self.finite += 1;
            if v == 0.0 {
                self.zero += 1;
            }
        }
    }
}
