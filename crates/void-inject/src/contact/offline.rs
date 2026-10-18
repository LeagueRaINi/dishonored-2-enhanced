//! Offline check of the contact shadows (`examples/contact_offline.rs`): a private D3D11
//! device and stand-in engine structures (pass with read/write lists, engine context, view
//! with projection, rotated view matrix, viewport, sun direction, shadow mode) laid out at
//! the offsets the hook reads, so `after_pass` runs exactly as after the engine's
//! SCREENSHADOWS callback. Scene: synthetic reversed-Z R24G8 depth of a floor, a lit wall, a
//! box and sky; the sun 15 degrees above the horizon behind-right of the box. The target is
//! R8G8B8A8_UNORM pre-filled like an engine screen-shadow image (one strip already in CSM
//! shadow). Checks: shadow on the floor in front of the box, none on open floor (near and
//! grazing), the lit wall or the box top (no acne), green = min(CSM, contact) (the strip
//! stays at its value), R/B/A and everything outside the viewport untouched, every D3D11
//! state of the deferred context unchanged, nothing written in constant-fill mode; GPU time
//! per quality at the full viewport and at render scale `scale`. External `.y` source (S1
//! entry point, `ao/external.rs`) with contact shadows on and off: replace and min give the
//! expected green inside the viewport (UNORM rounding: +-1), R/B/A and the outside equal the
//! run without it, nothing in constant-fill mode; with nothing registered (before and after a
//! registration) the output is bit-identical, and with contact shadows off as well the
//! target is untouched.

use super::*;
use crate::ao::ExternalMode;
use crate::ao::offline::{Gpu, Image, Probe, WIDE, external_pattern, external_source};

const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const NEAR: f32 = 0.01;
const FAR: f32 = 7000.0;
const JITTER: [f32; 2] = [0.0004, -0.0003];
/// Pre-filled target (R8G8B8A8_UNORM bytes): AO .x, CSM visibility .y, .z, .w.
const FILL: [u8; 4] = [64, 255, 191, 128];
/// Left strip already in CSM shadow (green 0.2).
const STRIP_GREEN: u8 = 51;
const STRIP_W: u32 = 64;
/// Camera yaw (engine view matrix), to exercise the matrix convention.
const YAW: f32 = 0.5;
const CAMERA: [f32; 3] = [5.0, 1.6, 2.0];

pub struct ContactReport {
    pub text: String,
    pub images: Vec<Image>,
    pub failures: Vec<String>,
}

/// Direction to the sun in the screen-consistent view frame: 15 degrees elevation,
/// horizontally toward +x and away from the camera.
fn sun_view() -> [f32; 3] {
    let e = 15f32.to_radians();
    let h = [0.6f32, 1.0];
    let n = (h[0] * h[0] + h[1] * h[1]).sqrt();
    [e.cos() * h[0] / n, e.sin(), e.cos() * h[1] / n]
}

/// Scene in the view frame (x right, y up, z forward, eye at the origin 1.6 m above the
/// floor): floor y = -1.6, back wall z = 8 and left wall x = -3 (z 2..8), both up to y = 1,
/// box x -0.5..0.5, y -1.6..-1.1, z 3.5..3.9; sky elsewhere.
fn trace(dir: [f32; 3]) -> Option<f32> {
    let mut best = f32::INFINITY;
    let mut hit = |t: f32, ok: bool| {
        if ok && t > 0.0 && t < best {
            best = t;
        }
    };
    if dir[1] < 0.0 {
        hit(-1.6 / dir[1], true);
    }
    let t = 8.0 / dir[2];
    hit(t, dir[1] * t <= 1.0);
    if dir[0] < 0.0 {
        let t = -3.0 / dir[0];
        hit(t, dir[1] * t <= 1.0 && (2.0..=8.0).contains(&(dir[2] * t)));
    }
    let (lo, hi) = ([-0.5f32, -1.6, 3.5], [0.5f32, -1.1, 3.9]);
    let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        let (u, v) = (lo[a] / dir[a], hi[a] / dir[a]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    }
    hit(t0, t0 <= t1);
    best.is_finite().then_some(best)
}

fn pixel(p: &Proj, size: [u32; 2], v: [f32; 3]) -> [u32; 2] {
    let ndc = [
        v[0] / v[2] * p.p00 + p.jitter[0],
        v[1] / v[2] * p.p11 + p.jitter[1],
    ];
    [
        ((ndc[0] + 1.0) * 0.5 * size[0] as f32) as u32,
        ((1.0 - ndc[1]) * 0.5 * size[1] as f32) as u32,
    ]
}

fn depth_texels(p: &Proj, tex: [u32; 2], region: [u32; 2]) -> Vec<u32> {
    let hw = |z: f32| {
        let d = (p.depth_add + p.depth_mul / z).clamp(0.0, 1.0);
        ((d * 16_777_215.0).round() as u32) | (5 << 24)
    };
    let mut out = Vec::with_capacity((tex[0] * tex[1]) as usize);
    for y in 0..tex[1] {
        for x in 0..tex[0] {
            out.push(if x < region[0] && y < region[1] {
                let ndc = [
                    (x as f32 + 0.5) / region[0] as f32 * 2.0 - 1.0,
                    1.0 - (y as f32 + 0.5) / region[1] as f32 * 2.0,
                ];
                let dir = [
                    (ndc[0] - p.jitter[0]) / p.p00,
                    (ndc[1] - p.jitter[1]) / p.p11,
                    1.0,
                ];
                trace(dir).map_or(0, hw)
            } else {
                hw(0.05)
            });
        }
    }
    out
}

/// Engine view matrix (row-major, column vectors) of a camera at `CAMERA` yawed by `YAW`,
/// and the world-space sun direction that maps to `sun_view()` through it.
fn camera() -> ([f32; 16], [f32; 3]) {
    let (s, c) = YAW.sin_cos();
    // Rotation about y (engine view <- world).
    let r = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
    let mut m = [0f32; 16];
    for (i, row) in r.iter().enumerate() {
        m[4 * i..4 * i + 3].copy_from_slice(row);
        m[4 * i + 3] = -(row[0] * CAMERA[0] + row[1] * CAMERA[1] + row[2] * CAMERA[2]);
    }
    m[15] = 1.0;
    // View frame (x, y, z forward) -> engine view (x, y, -z) -> world = R^T v.
    let l = sun_view();
    let v = [l[0], l[1], -l[2]];
    let w = std::array::from_fn(|k| (0..3).map(|i| r[i][k] * v[i]).sum());
    (m, w)
}

/// Stand-in engine memory for one SCREENSHADOWS pass (all buffers pinned while it lives).
struct Engine {
    pass: Vec<u64>,
    view: Vec<u64>,
    _ctx: Vec<u64>,
    _lists: Vec<Vec<u64>>,
    _images: Vec<Vec<u64>>,
    _image_views: Vec<Vec<u64>>,
}

impl Engine {
    fn new(
        ctx: &ID3D11DeviceContext,
        depth: &ID3D11Texture2D,
        target: &ID3D11Texture2D,
        proj: &[f32; 16],
        region: [u32; 2],
    ) -> Self {
        let images: Vec<Vec<u64>> = [depth.as_raw(), target.as_raw()]
            .iter()
            .map(|&r| {
                let mut img = vec![0u64; 0x140 / 8];
                img[0x130 / 8] = r as u64;
                img
            })
            .collect();
        let image_views: Vec<Vec<u64>> = images.iter().map(|i| vec![i.as_ptr() as u64]).collect();
        let lists = vec![
            vec![u64::from(HASH_DEPTH), image_views[0].as_ptr() as u64],
            vec![
                u64::from(HASH_SCREEN_SHADOWS),
                image_views[1].as_ptr() as u64,
            ],
        ];
        let engine_ctx = vec![ctx.as_raw() as u64, 0, 0, 0];
        let mut view = vec![0u64; (VIEW_SUN as usize + 64) / 8];
        let (vm, sun) = camera();
        {
            // SAFETY: `view` is a u64 buffer viewed as bytes for the writes below.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(view.as_mut_ptr().cast::<u8>(), view.len() * 8)
            };
            let put = |b: &mut [u8], off: u64, v: &[u8]| {
                b[off as usize..off as usize + v.len()].copy_from_slice(v);
            };
            let floats = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>();
            put(bytes, VIEW_SHADOW_MODE, &0u32.to_le_bytes());
            put(bytes, VIEW_PROJECTION, &floats(proj));
            put(bytes, VIEW_VIEW, &floats(&vm));
            let rect = [0i32, 0, region[0] as i32 - 1, region[1] as i32 - 1];
            put(
                bytes,
                VIEW_VIEWPORT,
                &rect
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            put(bytes, VIEW_SUN, &floats(&sun));
        }
        let mut pass = vec![0u64; 0x110 / 8];
        pass[(PASS_VIEW / 8) as usize] = view.as_ptr() as u64;
        pass[(PASS_ENGINE_CTX / 8) as usize] = engine_ctx.as_ptr() as u64;
        pass[(PASS_READS / 8) as usize] = lists[0].as_ptr() as u64;
        pass[(PASS_READS / 8) as usize + 1] = 1;
        pass[(PASS_WRITES / 8) as usize] = lists[1].as_ptr() as u64;
        pass[(PASS_WRITES / 8) as usize + 1] = 1;
        Self {
            pass,
            view,
            _ctx: engine_ctx,
            _lists: lists,
            _images: images,
            _image_views: image_views,
        }
    }

    fn set_mode(&mut self, mode: u32) {
        let i = (VIEW_SHADOW_MODE / 8) as usize;
        let shift = (VIEW_SHADOW_MODE % 8) * 8;
        self.view[i] = (self.view[i] & !(0xFFFF_FFFFu64 << shift)) | (u64::from(mode) << shift);
    }
}

/// R8G8B8A8 readback.
fn read8(g: &Gpu, tex: &ID3D11Texture2D, size: [u32; 2]) -> Result<Vec<[u8; 4]>, String> {
    let d = D3D11_TEXTURE2D_DESC {
        Width: size[0],
        Height: size[1],
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_STAGING,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        ..Default::default()
    };
    let mut st = None;
    // SAFETY (whole block): staging copy of an identical texture, mapped for reading within
    // the mapped pitch, unmapped afterwards.
    unsafe {
        g.dev
            .CreateTexture2D(&d, None, Some(&mut st))
            .map_err(create_err("staging"))?;
        let st = st.ok_or("no staging")?;
        g.imm.CopyResource(&st, tex);
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        g.imm
            .Map(&st, 0, D3D11_MAP_READ, 0, Some(&mut m))
            .map_err(create_err("Map staging"))?;
        let mut out = Vec::with_capacity((size[0] * size[1]) as usize);
        for y in 0..size[1] as usize {
            let row = m
                .pData
                .cast::<u8>()
                .add(y * m.RowPitch as usize)
                .cast::<[u8; 4]>();
            out.extend_from_slice(std::slice::from_raw_parts(row, size[0] as usize));
        }
        g.imm.Unmap(&st, 0);
        Ok(out)
    }
}

fn fill(size: [u32; 2]) -> Vec<u32> {
    let mut v = Vec::with_capacity((size[0] * size[1]) as usize);
    for _ in 0..size[1] {
        for x in 0..size[0] {
            let mut p = FILL;
            if x < STRIP_W {
                p[1] = STRIP_GREEN;
            }
            v.push(u32::from_le_bytes(p));
        }
    }
    v
}

/// Mean green (0..1) of the 5x5 block around `at`.
fn green(px: &[[u8; 4]], tex: [u32; 2], at: [u32; 2]) -> f32 {
    let mut sum = 0u32;
    for dy in 0..5 {
        for dx in 0..5 {
            let x = (at[0] + dx).saturating_sub(2).min(tex[0] - 1);
            let y = (at[1] + dy).saturating_sub(2).min(tex[1] - 1);
            sum += u32::from(px[(y * tex[0] + x) as usize][1]);
        }
    }
    sum as f32 / 25.0 / 255.0
}

struct Outcome {
    ours: bool,
    state_diff: Vec<String>,
    pixels: Vec<[u8; 4]>,
    gpu_ms: Option<f64>,
}

fn run(
    g: &Gpu,
    probe: &Probe,
    e: &Engine,
    target: &ID3D11Texture2D,
    size: [u32; 2],
    region: [u32; 2],
    time: bool,
) -> Result<Outcome, String> {
    probe.bind(&g.def, region);
    let before_ours = OURS.load(Ordering::Relaxed);
    let before_fail = FAILURES.load(Ordering::Relaxed);
    // SAFETY: capture on the deferred context recorded by this thread.
    let before = unsafe { Saved::capture(&g.def, WIDE) };
    after_pass(e.pass.as_ptr() as u64);
    // SAFETY: as above.
    let after = unsafe { Saved::capture(&g.def, WIDE) };
    if FAILURES.load(Ordering::Relaxed) != before_fail {
        return Err(lock(&ERROR).clone().unwrap_or_default());
    }
    let ours = OURS.load(Ordering::Relaxed) > before_ours;
    let state_diff = before.diff(&after);
    let mut list = None;
    // SAFETY: finishing the deferred context's list.
    unsafe { g.def.FinishCommandList(false, Some(&mut list)) }
        .map_err(create_err("FinishCommandList"))?;
    let list: ID3D11CommandList = list.ok_or("no command list")?;
    let data = fill(size);
    // SAFETY: full-texture update with data of the texture's size and pitch, then the list.
    unsafe {
        g.imm
            .UpdateSubresource(target, 0, None, data.as_ptr().cast(), size[0] * 4, 0);
        g.imm.ExecuteCommandList(&list, false);
    }
    let pixels = read8(g, target, size)?;
    let gpu_ms = if time { Some(g.time(&list)?) } else { None };
    Ok(Outcome {
        ours,
        state_diff,
        pixels,
        gpu_ms,
    })
}

/// See the module documentation.
pub fn offline_check(size: [u32; 2], scale: f32) -> Result<ContactReport, String> {
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let mut images = Vec::new();
    let g = Gpu::new()?;
    let code = bytecode()?;
    ao::external::compile_now()?;
    report.push(format!(
        "contact shaders compiled in {:.0} ms",
        code.compile_ms
    ));
    let probe = Probe::new(&g)?;
    let depth_bind = D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_DEPTH_STENCIL.0;
    let target_bind = D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0;
    let scaled = [
        crate::fsr_scale::snap_render(size[0], scale),
        crate::fsr_scale::snap_render(size[1], scale),
    ];
    let defaults = Settings::default();
    for (case, region) in [("full", size), ("scaled", scaled)] {
        let data = ao::engine_projection(
            region[0] as f32 / region[1] as f32,
            FOV_Y,
            NEAR,
            FAR,
            JITTER,
        );
        let proj = ao::projection(&data, 1)?;
        let texels = depth_texels(&proj, size, region);
        let depth = g.texture(size, DXGI_FORMAT_R24G8_TYPELESS, depth_bind, Some(&texels))?;
        let target = g.texture(size, DXGI_FORMAT_R8G8B8A8_TYPELESS, target_bind, None)?;
        let mut e = Engine::new(&g.def, &depth, &target, &data, region);
        let points = [
            (
                "contact (floor in front of the box)",
                [0.0, -1.6, 3.35],
                false,
            ),
            ("open floor", [2.0, -1.6, 3.0], true),
            ("open floor, grazing", [1.5, -1.6, 7.5], true),
            ("lit wall", [-3.0, -0.5, 6.0], true),
            ("box top", [0.0, -1.1, 3.7], true),
        ];
        let at: Vec<[u32; 2]> = points.iter().map(|p| pixel(&proj, region, p.1)).collect();
        report.push(format!(
            "\n{case}: texture {}x{}, viewport {}x{}; sun (view frame) {:?}; test pixels {at:?}",
            size[0],
            size[1],
            region[0],
            region[1],
            sun_view()
        ));
        for q in 0..QUALITIES.len() as u8 {
            settings::edit(|f| {
                f.contact_shadows = true;
                f.contact_quality = q;
            });
            let label = format!(
                "{} ({} steps)",
                quality_name(q),
                QUALITIES[usize::from(q)].1
            );
            let o = run(&g, &probe, &e, &target, size, region, true)?;
            let px = &o.pixels;
            let vals: Vec<f32> = at.iter().map(|&p| green(px, size, p)).collect();
            let (mut outside, mut other, mut strip, mut dark) = (0usize, 0usize, 0usize, 0usize);
            let mut n = 0usize;
            for y in 0..size[1] {
                for x in 0..size[0] {
                    let p = px[(y * size[0] + x) as usize];
                    let mut want = FILL;
                    if x < STRIP_W {
                        want[1] = STRIP_GREEN;
                    }
                    if x >= region[0] || y >= region[1] {
                        outside += usize::from(p != want);
                        continue;
                    }
                    other += usize::from(p[0] != want[0] || p[2] != want[2] || p[3] != want[3]);
                    if x < STRIP_W {
                        strip += usize::from(p[1] > STRIP_GREEN);
                    } else {
                        n += 1;
                        dark += usize::from(p[1] < 230);
                    }
                }
            }
            let dark_pct = 100.0 * dark as f64 / n.max(1) as f64;
            report.push(format!(
                "  {label:<17} GPU {:.3} ms | contact {:.3} | open {:.3} grazing {:.3} wall {:.3} box top {:.3} | darkened {dark_pct:.2} % | outside-written {outside} R/B/A-changed {other} strip-raised {strip} | state {}",
                o.gpu_ms.unwrap_or(f64::NAN),
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                if o.state_diff.is_empty() {
                    "restored".to_owned()
                } else {
                    format!("CHANGED: {}", o.state_diff.join(", "))
                }
            ));
            let mut fail = |ok: bool, what: String| {
                if !ok {
                    failures.push(format!("{case} {label}: {what}"));
                }
            };
            fail(o.ours, "not recorded".into());
            fail(
                o.state_diff.is_empty(),
                format!("state changed: {:?}", o.state_diff),
            );
            fail(
                outside == 0,
                format!("{outside} pixels outside the viewport written"),
            );
            fail(other == 0, format!("R/B/A changed on {other} pixels"));
            fail(
                strip == 0,
                format!("{strip} CSM-shadowed pixels raised (not min)"),
            );
            fail(
                vals[0] < 0.5,
                format!("contact shadow {:.3} not < 0.5", vals[0]),
            );
            for (i, v) in vals.iter().enumerate().skip(1) {
                fail(
                    *v > 0.95,
                    format!("{} {v:.3} not > 0.95 (acne)", points[i].0),
                );
            }
            fail(
                dark_pct < 3.0,
                format!("{dark_pct:.2} % of the viewport darkened"),
            );
            if q == defaults.contact_quality {
                images.push(Image {
                    name: format!("contact_{case}_{}", quality_name(q)),
                    width: region[0],
                    height: region[1],
                    gray: (0..region[1])
                        .flat_map(|y| (0..region[0]).map(move |x| (x, y)))
                        .map(|(x, y)| px[(y * size[0] + x) as usize][1])
                        .collect(),
                });
            }
        }
        // External `.y` source: after contact shadows (on) and alone (off).
        let ext_vals = external_pattern(region);
        let ext = external_source(&g, region, &ext_vals, ExternalMode::Replace)?;
        let unorm = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as i32;
        for contact in [true, false] {
            settings::edit(|f| {
                f.contact_shadows = contact;
                f.contact_quality = defaults.contact_quality;
            });
            set_external_y(None);
            let base = run(&g, &probe, &e, &target, size, region, false)?.pixels;
            for mode in [ExternalMode::Replace, ExternalMode::Min] {
                set_external_y(Some(ExternalSource {
                    mode,
                    ..ext.clone()
                }));
                let applied = ao::external::counters(Channel::Y).0;
                let o = run(&g, &probe, &e, &target, size, region, true)?;
                set_external_y(None);
                let applied = ao::external::counters(Channel::Y).0 > applied;
                let (mut wrong, mut off_by_one) = (0usize, 0usize);
                for y in 0..size[1] {
                    for x in 0..size[0] {
                        let i = (y * size[0] + x) as usize;
                        let (p, b) = (o.pixels[i], base[i]);
                        if x >= region[0] || y >= region[1] {
                            wrong += usize::from(p != b);
                            continue;
                        }
                        let v = unorm(ext_vals[(y * region[0] + x) as usize]);
                        let want = match mode {
                            ExternalMode::Replace => v,
                            ExternalMode::Min => v.min(i32::from(b[1])),
                        };
                        let d = (i32::from(p[1]) - want).abs();
                        off_by_one += usize::from(d == 1);
                        wrong += usize::from(d > 1 || p[0] != b[0] || p[2] != b[2] || p[3] != b[3]);
                    }
                }
                let name = format!(
                    "external .y {} (contact shadows {})",
                    mode.name(),
                    if contact { "on" } else { "off" }
                );
                report.push(format!(
                    "  {name:<38} GPU {:.3} ms | applied {applied} | wrong {wrong} | rounding +-1 {off_by_one} | state {}",
                    o.gpu_ms.unwrap_or(f64::NAN),
                    if o.state_diff.is_empty() {
                        "restored".to_owned()
                    } else {
                        format!("CHANGED: {}", o.state_diff.join(", "))
                    }
                ));
                if mode == ExternalMode::Min && contact {
                    images.push(Image {
                        name: format!("contact_{case}_external_min"),
                        width: region[0],
                        height: region[1],
                        gray: (0..region[1])
                            .flat_map(|y| (0..region[0]).map(move |x| (x, y)))
                            .map(|(x, y)| o.pixels[(y * size[0] + x) as usize][1])
                            .collect(),
                    });
                }
                let mut fail = |ok: bool, what: String| {
                    if !ok {
                        failures.push(format!("{case} {name}: {what}"));
                    }
                };
                fail(applied, "not applied".into());
                fail(
                    wrong == 0,
                    format!("{wrong} pixels differ from the expected"),
                );
                fail(
                    o.state_diff.is_empty(),
                    format!("state changed: {:?}", o.state_diff),
                );
            }
            let after = run(&g, &probe, &e, &target, size, region, false)?.pixels;
            let identical = after == base;
            let untouched = contact
                || after
                    == fill(size)
                        .iter()
                        .map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
            report.push(format!(
                "  nothing registered (contact shadows {}): bit-identical to the run before registering {identical}{}",
                if contact { "on" } else { "off" },
                if contact {
                    String::new()
                } else {
                    format!(", target untouched {untouched}")
                }
            ));
            if !identical || !untouched {
                failures.push(format!(
                    "{case}: output changed with nothing registered (contact shadows {contact})"
                ));
            }
        }
        settings::edit(|f| f.contact_shadows = true);
        // Constant-fill mode (no CSM): nothing recorded, target untouched, also with an
        // external source registered.
        e.set_mode(1);
        set_external_y(Some(ext.clone()));
        let skipped = SKIPPED.load(Ordering::Relaxed);
        let o = run(&g, &probe, &e, &target, size, region, false)?;
        set_external_y(None);
        let untouched = o.pixels
            == fill(size)
                .iter()
                .map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
        report.push(format!(
            "  constant-fill mode (external source registered): recorded {}, skipped {}, target untouched {untouched}",
            o.ours,
            SKIPPED.load(Ordering::Relaxed) - skipped
        ));
        if o.ours || !untouched {
            failures.push(format!("{case}: constant-fill mode was modified"));
        }
    }
    settings::edit(|f| *f = Settings::default());
    if failures.is_empty() {
        report.push("\nall checks passed".into());
    }
    Ok(ContactReport {
        text: report.join("\n"),
        images,
        failures,
    })
}
