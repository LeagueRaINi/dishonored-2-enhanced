//! End-to-end check of the in-process API (`void_rt::ingame`, docs/raytracing.md 10.5) the
//! game's DLL uses: `Gpu::from_existing` on a device created here (as the host does),
//! `load_scene` on a worker thread, `upload`, `PhotoSession` with small tiles and the
//! non-blocking `step` (budget per call, polling), `reset_camera`, `copy_out` into
//! RGBA32F / R32F textures, `read_hdr` / `read_depth`, `write_png` / `write_hdr`, cut-off
//! TLAS build / rebuild / refit. The result is compared with a reference HDR written by the
//! `photo` example at the same camera, size and spp (same seeds: identical or within float
//! noise). Offline only; reads the archives and F3 captures, never touches the game.
//!
//! WARP first (`--smoke`: 96x40, 4 spp, 64-texel textures, 2,000 nearest instances), then
//! one small GPU run; both are logged in `local/rt/runs.log`:
//! cargo run --release -p void-rt --example photo_ingame_check -- --smoke --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 \
//!   --reference local/rt/ingame/baseline_smoke/photo_hdr_rgba32f.bin
//! cargo run --release -p void-rt --example photo_ingame_check -- --game .. \
//!   --snapshot local/inject/snap_1792183388_f1288 --size 860x360 --spp 64 \
//!   --reference local/rt/ingame/baseline_gpu/photo_hdr_rgba32f.bin

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use void_rt::ingame::{
    Camera, DAYLIGHT_STORED_SCALE, Gpu, PhotoSession, PhotoSettings, SceneOptions, SceneSource,
    load_scene, upload,
};
use void_rt::lights::Env;
use void_rt::run::{Report, parse_size, read_f32, run_log};
use void_rt::snapshot::{Snapshot, Sun};

#[derive(Parser)]
#[command(about = "End-to-end check of void_rt::ingame (photo mode in-process API)")]
struct Args {
    #[arg(long)]
    game: PathBuf,
    #[arg(
        long,
        default_value = "maps/campaign/edge_of_the_world/eotw_p_highchaos"
    )]
    map: String,
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, default_value = "860x360")]
    size: String,
    #[arg(long, default_value_t = 64)]
    spp: u32,
    #[arg(long, default_value_t = 512)]
    tex_size: u32,
    /// Pixels per tile; default: an eighth of the image (several tiles per sample).
    #[arg(long)]
    tile_pixels: Option<u32>,
    /// GPU budget per `step` call (ms).
    #[arg(long, default_value_t = 8.0)]
    budget_ms: f32,
    /// `photo` example HDR (`photo_hdr_rgba32f.bin`) of the same camera / size / spp.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Largest relative luminance RMSE accepted against the reference.
    #[arg(long, default_value_t = 1e-3)]
    tolerance: f64,
    /// Also time path-tracing tiles at this size (`WxH`, e.g. 3440x1440; GPU only).
    #[arg(long)]
    measure_size: Option<String>,
    #[arg(long)]
    warp: bool,
    /// WARP, 96x40, 4 spp, 64-texel textures, 2000 nearest instances.
    #[arg(long)]
    smoke: bool,
    #[arg(long, default_value = "local/rt/ingame/check")]
    out: PathBuf,
    #[arg(long, default_value = "local/rt/runs.log")]
    runs_log: PathBuf,
}

const MIB: f64 = 1024.0 * 1024.0;

fn main() {
    let mut args = Args::parse();
    if args.smoke {
        args.warp = true;
        args.size = "96x40".into();
        args.spp = args.spp.min(4);
        args.tex_size = args.tex_size.min(64);
        if args.out == Path::new("local/rt/ingame/check") {
            args.out = PathBuf::from("local/rt/ingame/check_smoke");
        }
    }
    let mode = if args.warp {
        "ingame-check warp"
    } else {
        "ingame-check gpu"
    };
    let cmd: Vec<String> = std::env::args().skip(1).collect();
    run_log(
        &args.runs_log,
        &format!(
            "start {mode} run (out {}): {}",
            args.out.display(),
            cmd.join(" ")
        ),
    );
    let r = check(&args);
    run_log(
        &args.runs_log,
        &match &r {
            Ok(()) => format!("end {mode} run: ok"),
            Err(e) => format!("end {mode} run: error: {}", e.lines().next().unwrap_or("")),
        },
    );
    if let Err(e) = r {
        eprintln!("photo_ingame_check: {e}");
        std::process::exit(1);
    }
}

/// The host side: a device on the high-performance adapter (or WARP) and a direct queue,
/// created outside `void_rt` as the DLL does.
fn host_device(warp: bool) -> Result<(ID3D12Device5, ID3D12CommandQueue, String), String> {
    // SAFETY: DXGI / D3D12 creation calls with valid arguments.
    unsafe {
        let factory: IDXGIFactory6 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
        let adapter: IDXGIAdapter1 = if warp {
            factory.EnumWarpAdapter().map_err(|e| e.to_string())?
        } else {
            factory
                .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
                .map_err(|e| e.to_string())?
        };
        let d = adapter.GetDesc1().map_err(|e| e.to_string())?;
        let n = d.Description.iter().position(|&c| c == 0).unwrap_or(128);
        let name = String::from_utf16_lossy(&d.Description[..n]);
        let mut device: Option<ID3D12Device5> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
            .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
        let device = device.ok_or("no device")?;
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(|e| e.to_string())?;
        Ok((device, queue, name))
    }
}

/// A 2D texture in COMMON, as the host's D3D11-shared textures arrive.
fn texture(gpu: &Gpu, w: u32, h: u32, format: DXGI_FORMAT) -> Result<ID3D12Resource, String> {
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid descriptors and out pointer.
    unsafe {
        gpu.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
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
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut res,
        )
    }
    .map_err(|e| format!("texture: {e}"))?;
    res.ok_or_else(|| "no texture".into())
}

/// Reads a texture back through a pitched readback buffer (`bpp` bytes per texel).
fn read_texture(
    gpu: &mut Gpu,
    tex: &ID3D12Resource,
    w: u32,
    h: u32,
    format: DXGI_FORMAT,
    bpp: u64,
) -> Result<Vec<u8>, String> {
    let pitch = (u64::from(w) * bpp).div_ceil(256) * 256;
    let rb = {
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: valid descriptors and out pointer.
        unsafe {
            gpu.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: pitch * u64::from(h),
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
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
        }
        .map_err(|e| format!("readback: {e}"))?;
        res.ok_or("no readback")?
    };
    let dst = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(rb.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format,
                    Width: w,
                    Height: h,
                    Depth: 1,
                    RowPitch: pitch as u32,
                },
            },
        },
    };
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(tex.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: live resources, footprint matches the texture.
    unsafe { gpu.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    gpu.transition_resource(
        tex,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
    drop(std::mem::ManuallyDrop::into_inner(src.pResource));
    gpu.submit()?;
    let mut out = vec![0u8; (u64::from(w) * bpp * u64::from(h)) as usize];
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the finished copy wrote `pitch * h` bytes.
    unsafe {
        rb.Map(0, None, Some(&mut ptr))
            .map_err(|e| format!("Map: {e}"))?;
        for y in 0..h as usize {
            std::ptr::copy_nonoverlapping(
                ptr.cast::<u8>().add(y * pitch as usize),
                out.as_mut_ptr().add(y * (u64::from(w) * bpp) as usize),
                (u64::from(w) * bpp) as usize,
            );
        }
        rb.Unmap(0, None);
    }
    Ok(out)
}

fn lum(c: &[f32; 4]) -> f64 {
    f64::from(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
}

/// Relative RMSE of luminance (RMSE / mean of the reference) and the largest absolute
/// channel difference.
fn compare(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f64, f64) {
    let mean = b.iter().map(lum).sum::<f64>() / b.len().max(1) as f64;
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (lum(x) - lum(y)).powi(2))
        .sum::<f64>()
        / a.len().max(1) as f64;
    let max = a
        .iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |k| f64::from((x[k] - y[k]).abs())))
        .fold(0.0, f64::max);
    (mse.sqrt() / mean.max(1e-12), max)
}

fn bad(v: &[[f32; 4]]) -> usize {
    v.iter()
        .filter(|c| c.iter().any(|x| !x.is_finite()))
        .count()
}

#[allow(clippy::too_many_lines)]
fn check(args: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let mut rep = Report::default();
    std::fs::create_dir_all(&args.out).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let (w, h) = parse_size(&args.size)?;
    let mut fails: Vec<String> = Vec::new();

    // ------------------------------------------------------------ host device
    let (device, queue, adapter) = host_device(args.warp)?;
    let mut gpu = Gpu::from_existing(device, queue)?;
    rep.line(format!(
        "device: {adapter} (host-created, Gpu::from_existing: {}, raytracing tier {}, SM {})",
        gpu.info.name,
        gpu.info.raytracing_tier_name(),
        gpu.info.shader_model_name()
    ));

    // ------------------------------------------------------------ scene on a worker thread
    let snap = Snapshot::load(&args.snapshot)?;
    let camera = snap.camera;
    let src =
        SceneSource {
            game_dir: args.game.clone(),
            map: args.map.clone(),
            entities: snap.entities.clone(),
            lights: snap.lights.clone(),
            sun: Some(snap.sun.unwrap_or_else(|| {
                Sun::from_angles(26.0, 100.0, [0.215_861, 0.171_441, 0.116_971])
            })),
            env: Some(snap.env.unwrap_or(Env {
                sky_zenith: [0.0; 3],
                sky_horizon: [37.7451, 52.411_766, 55.0],
                ambient: [1.0; 3],
            })),
        };
    // Same options as the `photo` example (its upload batch, smoke instance limit).
    let opts = SceneOptions {
        texture_size: args.tex_size,
        upload_batch_bytes: 256 << 20,
        spike_primitives: true,
        tlas_limit: args.smoke.then_some((2000, camera.org)),
        ..SceneOptions::default()
    };
    let t = Instant::now();
    let cpu = std::thread::spawn(move || load_scene(&src, &opts))
        .join()
        .map_err(|_| "load_scene thread panicked")??;
    rep.line(format!(
        "load_scene (worker thread): {:.1} s, {} meshes, {} instances, {} records, {} spike primitives, {} lights kept of {}",
        t.elapsed().as_secs_f64(),
        cpu.scene.meshes.len(),
        cpu.scene.instances.len(),
        cpu.scene.records.len(),
        cpu.scene.primitives.len(),
        cpu.lights.len(),
        cpu.light_rows
    ));
    let gs = Arc::new(upload(&mut gpu, cpu)?);
    let tl = &gs.full_tlas().stats;
    rep.line(format!(
        "upload: {:.1} s; generation {}; BLAS {} ({:.1} MiB, {} batches); full TLAS {} instances at {:#x}; textures {} slots {:.1} MiB; primitives {}; offsets {}",
        gs.upload_seconds(),
        gs.generation(),
        gs.blas_stats().count,
        gs.blas_stats().compacted_bytes as f64 / MIB,
        gs.blas_stats().batches,
        tl.instances,
        gs.full_tlas_va(),
        gs.textures().count,
        gs.textures().bytes as f64 / MIB,
        gs.primitives().map_or(0, |b| b.size),
        gs.primitive_offsets().map_or(0, |b| b.size),
    ));
    if gs.primitives().is_none() || gs.blas_va(0).is_none() {
        fails.push("spike primitives / BLAS address missing".into());
    }

    // ------------------------------------------------------------ cut-off TLAS
    let mut cut = gs.build_cutoff_tlas(&mut gpu, camera.org, 40.0)?;
    let n40 = cut.count();
    let moved = [camera.org[0] + 30.0, camera.org[1], camera.org[2]];
    let n_moved = gs.update_cutoff_tlas(&mut gpu, &mut cut, moved, 40.0)?;
    let descs = gs.cutoff_descs(moved, 40.0);
    cut.refit(&mut gpu, &descs)?;
    rep.line(format!(
        "cut-off TLAS (40 m): {n40} instances, rebuilt at +30 m x: {n_moved}, refit ok; capacity {}",
        cut.capacity()
    ));
    if cut
        .refit(&mut gpu, &descs[..descs.len().saturating_sub(1)])
        .is_ok()
        && !descs.is_empty()
    {
        fails.push("refit with a different instance count was accepted".into());
    }
    drop(cut);

    // ------------------------------------------------------------ photo session
    let settings = PhotoSettings {
        spp_target: args.spp,
        znear: snap.znear,
        tile_pixels: args.tile_pixels.unwrap_or((w * h / 8).max(w * 4)),
        ..PhotoSettings::default()
    };
    let mut s = PhotoSession::new(&mut gpu, gs.clone(), camera, w, h, settings)?;
    let (rows, tiles) = s.tiling();
    rep.line(format!(
        "session {w}x{h}, {} spp, {tiles} tiles of {rows} rows per sample, budget {} ms per step",
        args.spp, args.budget_ms
    ));

    // Camera reset: accumulate a few samples elsewhere, then reset to the snapshot camera
    // (the reference must not see them).
    let other = Camera {
        org: [camera.org[0], camera.org[1], camera.org[2] + 2.0],
        ..camera
    };
    s.reset_camera(&mut gpu, other)?;
    s.set_spp_target(2);
    while !s.step(&mut gpu, args.budget_ms)?.done {
        s.finish(&mut gpu)?;
    }
    s.reset_camera(&mut gpu, camera)?;
    s.set_spp_target(args.spp);
    if s.spp_done() != 0 {
        fails.push("reset_camera kept samples".into());
    }

    // Non-blocking stepping: count polls that returned busy (the host's frames).
    let t = Instant::now();
    let (mut calls, mut busy_calls, mut max_call_ms) = (0usize, 0usize, 0f64);
    let mut copy_fence = None;
    let hdr_tex = texture(&gpu, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT)?;
    let depth_tex = texture(&gpu, w, h, DXGI_FORMAT_R32_FLOAT)?;
    loop {
        let tc = Instant::now();
        let p = s.step(&mut gpu, args.budget_ms)?;
        max_call_ms = max_call_ms.max(tc.elapsed().as_secs_f64() * 1e3);
        calls += 1;
        busy_calls += usize::from(p.busy);
        // Once half the samples are in, push a preview to the textures (non-blocking).
        if copy_fence.is_none() && p.spp_done >= args.spp / 2 && p.depth_ready {
            copy_fence = Some(s.copy_out(&mut gpu, &hdr_tex, &depth_tex)?);
        }
        if p.done {
            break;
        }
        if calls > 10_000_000 {
            return Err("step did not finish".into());
        }
        std::thread::yield_now();
    }
    let wall = t.elapsed().as_secs_f64();
    let p = s.finish(&mut gpu)?;
    let sms = s.sample_ms().to_vec();
    let mut sorted = sms.clone();
    sorted.sort_by(f64::total_cmp);
    let (submits, max_submit) = s.submit_stats();
    rep.line(format!(
        "stepping: {calls} step calls ({busy_calls} returned busy), longest call {max_call_ms:.2} ms CPU, wall {wall:.2} s; {submits} submits, max {max_submit:.2} ms GPU per submit; ms per tile {:.3}, per sample median {:.3} (min {:.3}, max {:.3}); depth pass {:.3} ms",
        p.ms_per_tile,
        sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        sorted.first().copied().unwrap_or(0.0),
        sorted.last().copied().unwrap_or(0.0),
        s.depth_ms()
    ));
    if p.spp_done != args.spp || !p.depth_ready {
        fails.push(format!("progress {p:?}"));
    }

    // ------------------------------------------------------------ outputs
    let hdr = s.read_hdr(&mut gpu)?;
    let depth = s.read_depth(&mut gpu)?;
    let acc: Vec<[f32; 4]> = gpu.read(&s.accum().rad, (w * h) as usize)?;
    let acc_mean = void_rt::photo::mean(&acc, s.spp_done());
    let (d_rel, d_max) = compare(&hdr, &acc_mean);
    rep.line(format!(
        "read_hdr vs accumulator mean: rel RMSE {d_rel:.2e}, max abs {d_max:.2e}; non-finite pixels hdr {} acc {}",
        bad(&hdr),
        bad(&acc_mean)
    ));
    if d_max > 1e-4 || bad(&hdr) > 0 {
        fails.push("resolve differs from the accumulator mean".into());
    }
    let sky = depth.iter().filter(|&&d| d <= 0.0).count();
    rep.line(format!(
        "depth: {:.1} % hits, {:.1} % sky; range {:.2e}..{:.2e}",
        100.0 * (depth.len() - sky) as f64 / depth.len() as f64,
        100.0 * sky as f64 / depth.len() as f64,
        depth
            .iter()
            .copied()
            .filter(|&d| d > 0.0)
            .fold(f32::MAX, f32::min),
        depth.iter().copied().fold(0.0, f32::max)
    ));
    // Primary depth (pixel centre) vs the accumulated mean view depth of the jittered first
    // hits (AccNrm.w; silhouettes mix depths, sky samples count 0): the median ratio checks
    // the convention, the share within 1 % the coverage.
    let (_, nrm) = s.read_features(&mut gpu)?;
    let mut ratios: Vec<f32> = depth
        .iter()
        .zip(&nrm)
        .filter(|(d, n)| **d > 0.0 && n[3] > 0.0)
        .map(|(d, n)| (snap.znear / d) / n[3])
        .collect();
    let within = ratios.iter().filter(|r| (*r - 1.0).abs() < 0.01).count();
    ratios.sort_by(f32::total_cmp);
    let med = ratios.get(ratios.len() / 2).copied().unwrap_or(f32::NAN);
    let share = within as f64 / ratios.len().max(1) as f64;
    rep.line(format!(
        "depth vs accumulated first-hit view depth: median ratio {med:.5}, {:.1} % of {} hit pixels within 1 %",
        100.0 * share,
        ratios.len()
    ));
    let median_ok = (med - 1.0).abs() < 0.005; // false for NaN (no hits)
    if !median_ok || share < 0.6 {
        fails.push("depth disagrees with the traced first hits".into());
    }

    // copy_out: the preview fence, then a final copy compared texel by texel.
    if let Some(v) = copy_fence {
        gpu.wait_idle()?;
        if !gpu.completed(v)? {
            fails.push("copy_out fence not reached".into());
        }
    } else {
        fails.push("copy_out was not exercised during stepping".into());
    }
    let v = s.copy_out(&mut gpu, &hdr_tex, &depth_tex)?;
    gpu.wait_idle()?;
    let tex_hdr: Vec<[f32; 4]> =
        read_texture(&mut gpu, &hdr_tex, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, 16)?
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| {
                    f32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
                })
            })
            .collect();
    let tex_depth: Vec<f32> = read_texture(&mut gpu, &depth_tex, w, h, DXGI_FORMAT_R32_FLOAT, 4)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let hdr_same = tex_hdr == hdr;
    let depth_same = tex_depth == depth;
    rep.line(format!(
        "copy_out (fence {v}): texture == read_hdr {hdr_same}, texture == read_depth {depth_same}"
    ));
    if !hdr_same || !depth_same {
        fails.push("copy_out textures differ from the buffers".into());
    }

    let e = s.write_png(&mut gpu, &args.out.join("ingame.png"), None)?;
    s.write_hdr(
        &mut gpu,
        &args.out.join("ingame_hdr_rgba32f.bin"),
        "photo_ingame_check",
    )?;
    rep.line(format!(
        "saved ingame.png (x {DAYLIGHT_STORED_SCALE}, exposure {e:.4}) and ingame_hdr_rgba32f.bin"
    ));

    // ------------------------------------------------------------ reference
    if let Some(r) = &args.reference {
        let raw = read_f32(r)?;
        let reference: Vec<[f32; 4]> = raw.as_chunks::<4>().0.to_vec();
        if reference.len() != hdr.len() {
            fails.push(format!(
                "reference {} pixels, ours {}",
                reference.len(),
                hdr.len()
            ));
        } else {
            let (rel, max) = compare(&hdr, &reference);
            let identical = hdr
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a[..3] == b[..3])
                .count();
            rep.line(format!(
                "vs reference {}: rel luminance RMSE {rel:.3e}, max abs channel diff {max:.3e}, bit-identical pixels {identical} of {} (tolerance {:.1e}); non-finite: ours {}, reference {}",
                r.display(),
                hdr.len(),
                args.tolerance,
                bad(&hdr),
                bad(&reference)
            ));
            let within_tolerance = rel <= args.tolerance; // false for NaN
            if !within_tolerance {
                fails.push(format!("reference RMSE {rel:.3e} > {:.1e}", args.tolerance));
            }
        }
    }

    // ------------------------------------------------------------ tile timing at size
    if let Some(ms) = &args.measure_size {
        let (mw, mh) = parse_size(ms)?;
        let mut m = PhotoSession::new(
            &mut gpu,
            gs.clone(),
            camera,
            mw,
            mh,
            PhotoSettings {
                spp_target: 8,
                znear: snap.znear,
                ..PhotoSettings::default()
            },
        )?;
        while !m.step(&mut gpu, args.budget_ms)?.done {
            m.finish(&mut gpu)?;
        }
        let p = m.finish(&mut gpu)?;
        let (r, n) = m.tiling();
        let (sub, maxs) = m.submit_stats();
        let round = |v: &[f64]| {
            v.iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        };
        rep.line(format!(
            "timing at {mw}x{mh}: {n} tiles of {r} rows ({} px); per tile of the last sample {:?} ms (mean of the latest per tile {:.2}); per sample {:?} ms; depth pass {:.2} ms; {sub} submits (max {maxs:.2} ms GPU)",
            r * mw,
            round(m.last_sample_tile_ms()),
            p.ms_per_tile,
            round(m.sample_ms()),
            m.depth_ms()
        ));
    }

    rep.line(format!("total {:.1} s", t0.elapsed().as_secs_f64()));
    if fails.is_empty() {
        rep.line("RESULT: PASS");
    } else {
        for f in &fails {
            rep.line(format!("FAIL: {f}"));
        }
        rep.line("RESULT: FAIL");
    }
    std::fs::write(args.out.join("report.txt"), &rep.0).map_err(|e| e.to_string())?;
    if fails.is_empty() {
        Ok(())
    } else {
        Err(format!("{} check(s) failed", fails.len()))
    }
}
