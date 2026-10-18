//! The engine's own NVIDIA Ansel photo session, driven by us (docs/raytracing.md 11.1), and
//! the free camera through the `updateCamera` import slot.
//!
//! - startSession `sub_1408B0160` / stopSession `sub_1408B02C0` are the engine's Ansel
//!   callbacks (registered in `sub_14036F510`; never called on AMD). startSession refuses
//!   (returns 0) without a game, while paused, without a running map, in a cinematic, or for
//!   two conditions of unknown meaning; otherwise it saves and zeroes `rs_enable`, sets
//!   `r_skipIggyFullscreen 1` / `r_skipIggy3D 1`, buffers `Ansel_PlayerVisible 0` and sets
//!   the session flag `byte_1439A529C` (the game frame then takes its paused path; the
//!   renderer skips the first-person hands) and `byte_1439A529E` (initialise the camera
//!   block). stopSession restores all of it. Both read the scoped "current game"
//!   `qword_1439A5290`, so they are only called when it equals the stable game pointer.
//! - `idRenderView_Setup` (0x140375711) calls `updateCamera(&block)` through the import
//!   slot 0x141A7E058 every view while the flag is set, after filling the block from the
//!   view on the first frame; afterwards the block becomes the view (origin, axis from the
//!   quaternion via `sub_140111410`, fov). Our replacement moves the block (free camera).
//!
//! Quaternion convention (O, `sub_140110080` / `sub_140111410`): the block's quaternion
//! (x, y, z, w) is the rotation whose matrix columns are the view axes forward, left, up.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use hex_literal::hex;
use void_rt::camera::Camera;

use super::probe;
use crate::{ffi, log, mem};

const START_RVA: u64 = 0x8B_0160;
const START_HEAD: [u8; 16] = hex!("48 83 EC 28 48 8B 0D 25 51 0F 03 48 85 C9 0F 84");
const STOP_RVA: u64 = 0x8B_02C0;
const STOP_HEAD: [u8; 16] = hex!("48 83 EC 28 8B 15 DE 4F 0F 03 48 8D 0D EF BB 0C");
/// Import address table slot of `AnselSDK64.dll!updateCamera`.
const UPDATE_CAMERA_SLOT_RVA: u64 = 0x1A7_E058;

/// `StartSessionStatus(SessionConfiguration&, void* user)`; 1 = allowed. The callback
/// does not touch its arguments (O); a zeroed buffer stands in for the configuration.
type StartFn = unsafe extern "system" fn(*mut u8, *mut c_void) -> u32;
type StopFn = unsafe extern "system" fn(*mut c_void);

fn check_head(rva: u64, head: &[u8; 16]) -> Result<u64, String> {
    let base = mem::image_base().ok_or("no image base")?;
    let addr = base + rva;
    let mut b = [0u8; 16];
    if !mem::read_into(addr, &mut b) || b != *head {
        return Err(format!(
            "bytes {b:02x?} at {addr:#x} differ from the analysed build"
        ));
    }
    Ok(addr)
}

/// Calls the engine's startSession (main thread). Ok(true) = session running.
pub fn start_session(p: &probe::Probe) -> Result<bool, String> {
    if p.game == 0 || p.current_game != p.game {
        return Err(format!(
            "the scoped current game {:#x} is not the game {:#x}",
            p.current_game, p.game
        ));
    }
    let addr = check_head(START_RVA, &START_HEAD)?;
    let mut config = [0u8; 256];
    // SAFETY: verified start of `sub_1408B0160` (main thread, game running, current game
    // equals the game); it ignores its arguments.
    let r = unsafe {
        std::mem::transmute::<u64, StartFn>(addr)(config.as_mut_ptr(), std::ptr::null_mut())
    };
    let flag = probe::read().session_flag;
    log::line(&format!(
        "photo: startSession returned {r}, session flag {flag:?}"
    ));
    Ok(r & 0xFF == 1 && flag == Some(1))
}

/// Calls the engine's stopSession (main thread).
pub fn stop_session() -> Result<(), String> {
    let p = probe::read();
    if p.game == 0 || p.current_game != p.game {
        return Err(format!(
            "the scoped current game {:#x} is not the game {:#x}",
            p.current_game, p.game
        ));
    }
    let addr = check_head(STOP_RVA, &STOP_HEAD)?;
    // SAFETY: verified start of `sub_1408B02C0` (main thread; it ignores its argument).
    unsafe { std::mem::transmute::<u64, StopFn>(addr)(std::ptr::null_mut()) };
    log::line(&format!(
        "photo: stopSession, session flag {:?}",
        probe::read().session_flag
    ));
    Ok(())
}

// ------------------------------------------------------------------ free camera --

/// The export the slot must hold before we replace it, and our function.
static ORIGINAL: AtomicU64 = AtomicU64::new(0);
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// `AnselSDK64.dll!updateCamera`.
fn ansel_update_camera() -> Result<u64, String> {
    use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    use windows::core::{s, w};
    // SAFETY: module/export lookups by constant names; no pointers are kept besides the
    // address value.
    unsafe {
        let m =
            GetModuleHandleW(w!("AnselSDK64.dll")).map_err(|e| format!("AnselSDK64.dll: {e}"))?;
        GetProcAddress(m, s!("updateCamera"))
            .map(|f| f as usize as u64)
            .ok_or_else(|| "AnselSDK64.dll exports no updateCamera".to_owned())
    }
}

/// Replaces the `updateCamera` import slot with [`update_camera`] after checking that it
/// holds AnselSDK64's export.
pub fn install_camera() -> Result<(), String> {
    if INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }
    let base = mem::image_base().ok_or("no image base")?;
    let export = ansel_update_camera()?;
    let ours = update_camera as *const () as u64;
    {
        let mut c = cam();
        *c = FreeCam::default();
    }
    mem::swap_pointer(base + UPDATE_CAMERA_SLOT_RVA, export, ours)?;
    ORIGINAL.store(export, Ordering::Release);
    INSTALLED.store(true, Ordering::Release);
    log::line(&format!(
        "photo: updateCamera slot {:#x} redirected ({export:#x} -> {ours:#x})",
        base + UPDATE_CAMERA_SLOT_RVA
    ));
    Ok(())
}

/// Puts AnselSDK64's export back into the slot.
pub fn uninstall_camera() -> Result<(), String> {
    if !INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }
    let base = mem::image_base().ok_or("no image base")?;
    let ours = update_camera as *const () as u64;
    mem::swap_pointer(
        base + UPDATE_CAMERA_SLOT_RVA,
        ours,
        ORIGINAL.load(Ordering::Acquire),
    )?;
    INSTALLED.store(false, Ordering::Release);
    log::line("photo: updateCamera slot restored");
    Ok(())
}

/// Free-camera input (written by the main thread, read by [`update_camera`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
    /// Forward, left, up in [-1, 1].
    pub move_dir: [f32; 3],
    /// Speed multiplier (Shift / Ctrl).
    pub speed: f32,
    /// Mouse counts since the last frame.
    pub look: [f32; 2],
    pub reset: bool,
}

/// Free camera state.
#[derive(Clone, Copy, Debug, Default)]
struct FreeCam {
    /// Pose at the session start: position, yaw, pitch (degrees).
    start: Option<([f32; 3], f32, f32)>,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov_y: f32,
    pending: Input,
    last: Option<Instant>,
    /// Bumped whenever the pose changes.
    version: u64,
}

static CAM: Mutex<FreeCam> = Mutex::new(FreeCam {
    start: None,
    pos: [0.0; 3],
    yaw: 0.0,
    pitch: 0.0,
    fov_y: 0.0,
    pending: Input {
        move_dir: [0.0; 3],
        speed: 0.0,
        look: [0.0; 2],
        reset: false,
    },
    last: None,
    version: 0,
});

fn cam() -> std::sync::MutexGuard<'static, FreeCam> {
    CAM.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Movement speed (m/s) and mouse sensitivity (degrees per count).
const SPEED: f32 = 3.0;
const LOOK_DEG: f32 = 0.08;

/// Queues one frame of input for the next `updateCamera` call (main thread).
pub fn set_input(i: Input) {
    let mut c = cam();
    c.pending.move_dir = i.move_dir;
    c.pending.speed = i.speed;
    c.pending.look[0] += i.look[0];
    c.pending.look[1] += i.look[1];
    c.pending.reset |= i.reset;
}

/// Polls the free-camera keys (W/A/S/D, Q/E, Shift, Ctrl, R) and the captured mouse.
pub fn poll_input() -> Input {
    let k = |c: u8| ffi::key_down(i32::from(c));
    let axis = |pos: u8, neg: u8| f32::from(u8::from(k(pos))) - f32::from(u8::from(k(neg)));
    let (dx, dy) = crate::mouse::take_motion();
    Input {
        move_dir: [axis(b'W', b'S'), axis(b'A', b'D'), axis(b'E', b'Q')],
        speed: if k(0x10) {
            5.0
        } else if k(0x11) {
            0.2
        } else {
            1.0
        },
        look: [dx as f32, dy as f32],
        reset: k(b'R'),
    }
}

/// The free camera's current pose and its version (None before the first call).
pub fn current() -> Option<(Camera, u64)> {
    let c = cam();
    c.start?;
    Some((
        Camera {
            org: c.pos,
            axis: axes(c.yaw, c.pitch),
            fov_y: c.fov_y,
        },
        c.version,
    ))
}

/// View axes (forward, left, up) for yaw/pitch in degrees, pitch positive = down
/// (`void_rt::camera` convention).
pub fn axes(yaw: f32, pitch: f32) -> [[f32; 3]; 3] {
    let (sy, cy) = yaw.to_radians().sin_cos();
    let (sp, cp) = pitch.to_radians().sin_cos();
    [
        [cp * cy, cp * sy, -sp],
        [-sy, cy, 0.0],
        [sp * cy, sp * sy, cp],
    ]
}

/// The engine's quaternion -> 3x3 (row-major, `sub_140111410`); its columns are the axes.
pub fn quat_to_matrix(q: [f32; 4]) -> [f32; 9] {
    let [x, y, z, w] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Inverse of [`quat_to_matrix`] for a rotation matrix (row-major).
pub fn matrix_to_quat(m: [f32; 9]) -> [f32; 4] {
    let r = |i: usize, j: usize| m[3 * i + j];
    let t = r(0, 0) + r(1, 1) + r(2, 2);
    if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [
            (r(2, 1) - r(1, 2)) / s,
            (r(0, 2) - r(2, 0)) / s,
            (r(1, 0) - r(0, 1)) / s,
            0.25 * s,
        ]
    } else if r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2) {
        let s = (1.0 + r(0, 0) - r(1, 1) - r(2, 2)).sqrt() * 2.0;
        [
            0.25 * s,
            (r(0, 1) + r(1, 0)) / s,
            (r(0, 2) + r(2, 0)) / s,
            (r(2, 1) - r(1, 2)) / s,
        ]
    } else if r(1, 1) > r(2, 2) {
        let s = (1.0 + r(1, 1) - r(0, 0) - r(2, 2)).sqrt() * 2.0;
        [
            (r(0, 1) + r(1, 0)) / s,
            0.25 * s,
            (r(1, 2) + r(2, 1)) / s,
            (r(0, 2) - r(2, 0)) / s,
        ]
    } else {
        let s = (1.0 + r(2, 2) - r(0, 0) - r(1, 1)).sqrt() * 2.0;
        [
            (r(0, 2) + r(2, 0)) / s,
            (r(1, 2) + r(2, 1)) / s,
            0.25 * s,
            (r(1, 0) - r(0, 1)) / s,
        ]
    }
}

/// Quaternion of the view axes (columns forward, left, up).
pub fn axes_to_quat(a: [[f32; 3]; 3]) -> [f32; 4] {
    matrix_to_quat([
        a[0][0], a[1][0], a[2][0], a[0][1], a[1][1], a[2][1], a[0][2], a[1][2], a[2][2],
    ])
}

/// Yaw/pitch (degrees) of a forward vector.
fn yaw_pitch(fwd: [f32; 3]) -> (f32, f32) {
    (
        fwd[1].atan2(fwd[0]).to_degrees(),
        (-fwd[2]).clamp(-1.0, 1.0).asin().to_degrees(),
    )
}

/// Applies `input` over `dt` seconds.
fn advance(c: &mut FreeCam, input: Input, dt: f32) {
    if input.reset
        && let Some((p, y, pi)) = c.start
    {
        c.pos = p;
        c.yaw = y;
        c.pitch = pi;
        c.version += 1;
        return;
    }
    let before = (c.pos, c.yaw, c.pitch);
    c.yaw -= input.look[0] * LOOK_DEG;
    c.pitch = (c.pitch + input.look[1] * LOOK_DEG).clamp(-89.0, 89.0);
    let a = axes(c.yaw, c.pitch);
    let v = SPEED * input.speed * dt;
    for k in 0..3 {
        c.pos[k] += v * (input.move_dir[0] * a[0][k] + input.move_dir[1] * a[1][k]);
    }
    c.pos[2] += v * input.move_dir[2];
    if (c.pos, c.yaw, c.pitch) != before {
        c.version += 1;
    }
}

/// `updateCamera(ansel::Camera&)` replacement: position[3], quaternion[4], fov (degrees),
/// projection offset[2]. Called by `idRenderView_Setup` while the session flag is set.
unsafe extern "C" fn update_camera(block: *mut f32) {
    if block.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(|| {
        // SAFETY: the engine passes its camera block (0x14397BC60, 10 floats).
        let b = unsafe { std::slice::from_raw_parts_mut(block, 10) };
        let mut c = cam();
        let now = Instant::now();
        if c.start.is_none() {
            let m = quat_to_matrix([b[3], b[4], b[5], b[6]]);
            let (yaw, pitch) = yaw_pitch([m[0], m[3], m[6]]);
            c.pos = [b[0], b[1], b[2]];
            c.yaw = yaw;
            c.pitch = pitch;
            c.start = Some((c.pos, yaw, pitch));
            c.version += 1;
        }
        c.fov_y = b[7].to_radians();
        let dt = c.last.map_or(0.0, |l| (now - l).as_secs_f32()).min(0.1);
        c.last = Some(now);
        let input = std::mem::take(&mut c.pending);
        advance(&mut c, input, dt);
        let q = axes_to_quat(axes(c.yaw, c.pitch));
        b[0] = c.pos[0];
        b[1] = c.pos[1];
        b[2] = c.pos[2];
        b[3..7].copy_from_slice(&q);
        // No projection offset (the block keeps the engine's value otherwise).
        b[8] = 0.0;
        b[9] = 0.0;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn quaternion_round_trip_matches_engine_matrix() {
        for (yaw, pitch) in [
            (0.0, 0.0),
            (37.0, -20.0),
            (-120.0, 75.0),
            (179.0, 5.0),
            (90.0, -89.0),
        ] {
            let a = axes(yaw, pitch);
            let m = quat_to_matrix(axes_to_quat(a));
            // Columns of the engine matrix are the axes (rows of the view axis).
            for (k, axis) in a.iter().enumerate() {
                assert!(
                    close(&[m[k], m[3 + k], m[6 + k]], axis),
                    "{yaw} {pitch} axis {k}"
                );
            }
            let (y2, p2) = yaw_pitch([m[0], m[3], m[6]]);
            assert!((y2 - yaw).abs() < 1e-2 && (p2 - pitch).abs() < 1e-2);
        }
    }

    #[test]
    fn movement_follows_axes() {
        let mut c = FreeCam {
            start: Some(([1.0, 2.0, 3.0], 90.0, 0.0)),
            pos: [1.0, 2.0, 3.0],
            yaw: 90.0,
            ..FreeCam::default()
        };
        advance(
            &mut c,
            Input {
                move_dir: [1.0, 0.0, 0.0],
                speed: 1.0,
                ..Input::default()
            },
            1.0,
        );
        // Yaw 90: forward = +Y.
        assert!(close(&c.pos, &[1.0, 2.0 + SPEED, 3.0]));
        advance(
            &mut c,
            Input {
                reset: true,
                ..Input::default()
            },
            1.0,
        );
        assert!(close(&c.pos, &[1.0, 2.0, 3.0]));
    }
}
