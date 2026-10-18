//! DirectInput hooks: synthetic mouse-look input for the benchmark pan (`bench`), input
//! blocking while the overlay is open (`overlay`) and input capture for the photo mode's
//! free camera (`photo`).
//!
//! The game reads the mouse through DirectInput 8 in buffered mode (O, IDB:
//! `sub_1400E2800` creates the `GUID_SysMouse` device, sets the `c_dfDIMouse2` layout
//! (X at offset 0, relative axes) and a 256-entry buffer). `IDirectInputDevice8::
//! GetDeviceData` (vtable slot 10) of the system `dinput8.dll` is hooked; for the mouse
//! device, queued X counts are appended to the returned buffer as one X-axis record, so
//! the game turns the player exactly as for a real mouse movement. Nothing is injected
//! unless `add_x` queued counts.
//!
//! Blocking (`set_blocked`, d2enhanced `overlay.cpp` / `game.cpp`): the game's mouse and
//! keyboard devices are unacquired; `Acquire` (slot 7) then fails with
//! `DIERR_OTHERAPPHASPRIO`, `GetDeviceData` returns no records and `GetDeviceState`
//! (slot 9) a zeroed state for them. The engine's mouse poll (`idMouseWin32` vtable slot 8,
//! `sub_1400E2930`: `GetDeviceData(24, buf, &n)` on the device at `this+0x178`, on failure
//! `SetCooperativeLevel(hWnd, EXCLUSIVE|FOREGROUND)` + `Acquire`) additionally flushes the
//! device buffer first, as d2enhanced does (its notes: generic DirectInput hooks alone did
//! not stop mouse look there). Unblocking flushes the mouse buffers again, so no movement
//! collected meanwhile reaches the game; the game re-acquires by itself.
//!
//! Capture (`set_capture`, photo free camera): the engine keeps polling the mouse while the
//! game is paused (O, IDB: `idCommonLocal` frame `sub_1400450D0` -> `sub_14003A900` ->
//! `sub_14003A470` calls `idSysLocal` slot 23 `GetMouse` then the poll, every frame); the
//! records it reads are summed for [`take_motion`] and withheld. `lock_cursor` hides the
//! cursor and clips it to the window centre meanwhile.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use hex_literal::hex;
use windows::Win32::Foundation::HINSTANCE;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::core::{GUID, HRESULT};

use crate::{ffi, log, mem};

const IID_DIRECTINPUT8W: GUID = GUID::from_u128(0xBF798031_483A_4DA2_AA99_5D64ED369700);
const IID_DIRECTINPUT8A: GUID = GUID::from_u128(0xBF798030_483A_4DA2_AA99_5D64ED369700);
const GUID_SYSMOUSE: GUID = GUID::from_u128(0x6F1D2B60_D5A0_11CF_BFC7_444553540000);
const GUID_SYSKEYBOARD: GUID = GUID::from_u128(0x6F1D2B61_D5A0_11CF_BFC7_444553540000);
const DIRECTINPUT_VERSION: u32 = 0x0800;
/// `IDirectInput8::CreateDevice`; `IDirectInputDevice8` methods.
const SLOT_CREATE_DEVICE: usize = 3;
const SLOT_RELEASE: usize = 2;
const SLOT_GET_CAPABILITIES: usize = 3;
const SLOT_ACQUIRE: usize = 7;
const SLOT_UNACQUIRE: usize = 8;
const SLOT_GET_DEVICE_STATE: usize = 9;
const SLOT_GET_DEVICE_DATA: usize = 10;
const DIGDD_PEEK: u32 = 1;
const DI8DEVTYPE_MOUSE: u32 = 0x12;
const DI8DEVTYPE_KEYBOARD: u32 = 0x13;
/// `DIERR_OTHERAPPHASPRIO` (= E_ACCESSDENIED).
const DIERR_OTHERAPPHASPRIO: HRESULT = HRESULT(0x8007_0005_u32 as i32);
/// `DIDEVICEOBJECTDATA`: dwOfs, dwData, dwTimeStamp, dwSequence (+ uAppData).
const RECORD_MIN: usize = 16;
/// `sizeof(DIDEVICEOBJECTDATA)` on x64 (the engine's poll passes 24).
const RECORD_SIZE: u32 = 24;
/// Devices not read for this long are not unacquired (they may have been released).
const SEEN_RECENTLY: Duration = Duration::from_secs(2);

/// idMouseWin32 poll (`sub_1400E2930`) and its device field.
const MOUSE_POLL_RVA: u64 = 0xE_2930;
const MOUSE_POLL_HEAD: [u8; 16] = hex!("48 89 4C 24 08 55 41 56 48 8D AC 24 78 E8 FF FF");
const MOUSE_DEVICE: u64 = 0x178;

/// Counts queued for the next mouse read / delivered to the game so far.
static PENDING_X: AtomicI32 = AtomicI32::new(0);
static DELIVERED_X: AtomicI64 = AtomicI64::new(0);
/// Mouse/keyboard input withheld from the game (overlay open).
static BLOCKED: AtomicBool = AtomicBool::new(false);
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// Trampolines per method, one per distinct implementation (wide/ANSI, mouse/keyboard).
const IMPLS: usize = 4;
static DATA_ORIG: [AtomicUsize; IMPLS] = [const { AtomicUsize::new(0) }; IMPLS];
static STATE_ORIG: [AtomicUsize; IMPLS] = [const { AtomicUsize::new(0) }; IMPLS];
static ACQUIRE_ORIG: [AtomicUsize; IMPLS] = [const { AtomicUsize::new(0) }; IMPLS];
static POLL_ORIG: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Mouse,
    Keyboard,
    Other,
}

struct Device {
    ptr: usize,
    kind: Kind,
    seen: Instant,
}

/// Devices the game read from.
static DEVICES: Mutex<Vec<Device>> = Mutex::new(Vec::new());

/// Queues `dx` mouse counts (positive = right) for the game's next mouse read.
pub fn add_x(dx: i32) {
    PENDING_X.fetch_add(dx, Ordering::Relaxed);
}

/// Total counts the game has received from `add_x`.
pub fn delivered() -> i64 {
    DELIVERED_X.load(Ordering::Relaxed)
}

/// Drops counts that the game has not read yet.
pub fn clear_pending() {
    PENDING_X.store(0, Ordering::Relaxed);
}

/// Mouse motion read from the game's blocked mouse device (photo free camera).
static CAPTURE: AtomicBool = AtomicBool::new(false);
static MOTION_X: AtomicI32 = AtomicI32::new(0);
static MOTION_Y: AtomicI32 = AtomicI32::new(0);

/// Captures the game's mouse and keyboard (`true`) or gives them back: the devices stay
/// acquired, `GetDeviceData` records reach us instead of the game (mouse X/Y collected for
/// [`take_motion`]), `GetDeviceState` returns zeros. The overlay's blocking takes precedence
/// while it is open. Called every frame by the photo mode: the collected motion is reset
/// only when the state changes (in-game run 1 bug: a reset on every call dropped the counts
/// the engine's mouse poll had collected earlier in the frame, so mouse look never moved).
pub fn set_capture(on: bool) {
    if CAPTURE.swap(on, Ordering::Relaxed) != on {
        MOTION_X.store(0, Ordering::Relaxed);
        MOTION_Y.store(0, Ordering::Relaxed);
    }
}

/// Cursor lock of the photo free camera: `ShowCursor(FALSE)` calls made (undone on
/// release) and the clip rectangle before the lock (restored on release).
struct CursorLock {
    hidden: u32,
    clip_before: Option<windows::Win32::Foundation::RECT>,
}

static CURSOR_LOCK: Mutex<Option<CursorLock>> = Mutex::new(None);

/// Hides the cursor and holds it at the centre of the foreground (game) window (`true`,
/// main thread, every frame: re-applied when the game shows or frees it), or restores the
/// cursor and the previous clip rectangle (`false`). Mouse look reads DirectInput's
/// relative counts, so the cursor position is not needed.
pub fn lock_cursor(on: bool) {
    use windows::Win32::Foundation::{POINT, RECT};
    use windows::Win32::Graphics::Gdi::ClientToScreen;
    use windows::Win32::UI::WindowsAndMessaging::{
        CURSOR_SHOWING, CURSORINFO, ClipCursor, GetClientRect, GetClipCursor, GetCursorInfo,
        GetForegroundWindow, ShowCursor,
    };
    let mut lock = CURSOR_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !on {
        if let Some(l) = lock.take() {
            // SAFETY: plain user32 calls; the rectangle is a local copy.
            unsafe {
                for _ in 0..l.hidden {
                    ShowCursor(true);
                }
                let _ = ClipCursor(l.clip_before.as_ref().map(std::ptr::from_ref));
            }
        }
        return;
    }
    let l = lock.get_or_insert_with(|| {
        let mut r = RECT::default();
        // SAFETY: valid out pointer.
        let before = unsafe { GetClipCursor(&mut r) }.ok().map(|()| r);
        CursorLock {
            hidden: 0,
            clip_before: before,
        }
    });
    // SAFETY: plain user32 calls with local out pointers.
    unsafe {
        // Hide only while shown (the display counter cannot be read without changing it):
        // each `ShowCursor(FALSE)` lowers the counter by one until it is below zero.
        let mut info = CURSORINFO {
            cbSize: std::mem::size_of::<CURSORINFO>() as u32,
            ..Default::default()
        };
        if GetCursorInfo(&mut info).is_ok() && info.flags.0 & CURSOR_SHOWING.0 != 0 {
            for _ in 0..16 {
                l.hidden += 1;
                if ShowCursor(false) < 0 {
                    break;
                }
            }
        }
        let hwnd = GetForegroundWindow();
        let mut c = RECT::default();
        if GetClientRect(hwnd, &mut c).is_ok() {
            let mut p = POINT {
                x: (c.left + c.right) / 2,
                y: (c.top + c.bottom) / 2,
            };
            if ClientToScreen(hwnd, &mut p).as_bool() {
                let r = RECT {
                    left: p.x,
                    top: p.y,
                    right: p.x + 1,
                    bottom: p.y + 1,
                };
                let _ = ClipCursor(Some(&r));
            }
        }
    }
}

/// Mouse counts (x right, y down) collected since the last call.
pub fn take_motion() -> (i32, i32) {
    (
        MOTION_X.swap(0, Ordering::Relaxed),
        MOTION_Y.swap(0, Ordering::Relaxed),
    )
}

/// Whether the input hooks are installed.
pub fn installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// Adds the X/Y axis records of a buffer the original `GetDeviceData` filled.
///
/// # Safety
/// `data` holds `n` records of `cb >= RECORD_MIN` bytes.
unsafe fn collect_motion(data: *const u8, cb: usize, n: u32) {
    for i in 0..n as usize {
        // SAFETY: per the caller contract; `dwOfs` at +0 and `dwData` at +4.
        let (ofs, value) = unsafe {
            let rec = data.add(i * cb);
            (
                rec.cast::<u32>().read_unaligned(),
                rec.add(4).cast::<i32>().read_unaligned(),
            )
        };
        match ofs {
            0 => MOTION_X.fetch_add(value, Ordering::Relaxed),
            4 => MOTION_Y.fetch_add(value, Ordering::Relaxed),
            _ => 0,
        };
    }
}

type GetDeviceData = unsafe extern "system" fn(*mut c_void, u32, *mut u8, *mut u32, u32) -> HRESULT;
type GetDeviceState = unsafe extern "system" fn(*mut c_void, u32, *mut u8) -> HRESULT;
type Method0 = unsafe extern "system" fn(*mut c_void) -> HRESULT;
type GetCapabilities = unsafe extern "system" fn(*mut c_void, *mut u32) -> HRESULT;
type CreateDevice =
    unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void, *mut c_void) -> HRESULT;
type Release = unsafe extern "system" fn(*mut c_void) -> u32;
type MousePoll = unsafe extern "system" fn(u64) -> u64;

/// # Safety
/// `obj` must be a live COM interface pointer whose vtable has more than `slot` entries.
unsafe fn vtable_entry(obj: *mut c_void, slot: usize) -> usize {
    // SAFETY: per the caller contract; the first field of a COM object is its vtable.
    unsafe { *(*(obj as *const *const usize)).add(slot) }
}

/// Kind of a device the game called a method on (cached; refreshes its last-seen time).
fn kind(dev: *mut c_void) -> Kind {
    let mut devices = DEVICES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(d) = devices.iter_mut().find(|d| d.ptr == dev as usize) {
        d.seen = Instant::now();
        return d.kind;
    }
    // DIDEVCAPS: 11 DWORDs, dwDevType at index 2.
    let mut caps = [0u32; 11];
    caps[0] = 44;
    // SAFETY: `dev` is the device the game called a method on (a live
    // IDirectInputDevice8); `caps` is a DIDEVCAPS-sized buffer with dwSize set.
    let hr = unsafe {
        std::mem::transmute::<usize, GetCapabilities>(vtable_entry(dev, SLOT_GET_CAPABILITIES))(
            dev,
            caps.as_mut_ptr(),
        )
    };
    let kind = match caps[2] & 0xFF {
        _ if hr.is_err() => Kind::Other,
        DI8DEVTYPE_MOUSE => Kind::Mouse,
        DI8DEVTYPE_KEYBOARD => Kind::Keyboard,
        _ => Kind::Other,
    };
    devices.push(Device {
        ptr: dev as usize,
        kind,
        seen: Instant::now(),
    });
    kind
}

/// Mouse or keyboard input is currently withheld from `dev` (overlay open, devices
/// unacquired).
fn blocked_for(dev: *mut c_void) -> bool {
    BLOCKED.load(Ordering::Relaxed)
        && std::panic::catch_unwind(|| kind(dev) != Kind::Other).unwrap_or(false)
}

/// Input is captured from `dev` ([`set_capture`]): its records are read for us and
/// withheld from the game; the device stays acquired.
fn captured_kind(dev: *mut c_void) -> Option<Kind> {
    if !CAPTURE.load(Ordering::Relaxed) {
        return None;
    }
    std::panic::catch_unwind(|| kind(dev))
        .ok()
        .filter(|k| *k != Kind::Other)
}

unsafe extern "system" fn get_device_data<const I: usize>(
    dev: *mut c_void,
    cb: u32,
    data: *mut u8,
    inout: *mut u32,
    flags: u32,
) -> HRESULT {
    // SAFETY: MinHook trampoline with this signature.
    let f = unsafe {
        std::mem::transmute::<usize, GetDeviceData>(DATA_ORIG[I].load(Ordering::Relaxed))
    };
    // Capacity on input, record count on output.
    let cap = if inout.is_null() {
        0
    } else {
        // SAFETY: non-null pointer supplied by the game for this call.
        unsafe { *inout }
    };
    // SAFETY: arguments passed through unchanged.
    let hr = unsafe { f(dev, cb, data, inout, flags) };
    if blocked_for(dev) {
        if !inout.is_null() && !data.is_null() {
            // SAFETY: non-null count pointer of this call; the records read are dropped.
            unsafe { *inout = 0 };
        }
        return hr;
    }
    if let Some(k) = captured_kind(dev) {
        if !inout.is_null() && !data.is_null() {
            if k == Kind::Mouse
                && hr.is_ok()
                && flags & DIGDD_PEEK == 0
                && (cb as usize) >= RECORD_MIN
            {
                // SAFETY: the original call filled `*inout` (<= cap) records of `cb` bytes.
                unsafe { collect_motion(data, cb as usize, (*inout).min(cap)) };
            }
            // SAFETY: non-null count pointer of this call; the records are withheld.
            unsafe { *inout = 0 };
        }
        return hr;
    }
    if hr.is_err()
        || flags & DIGDD_PEEK != 0
        || data.is_null()
        || inout.is_null()
        || (cb as usize) < RECORD_MIN
        || PENDING_X.load(Ordering::Relaxed) == 0
    {
        return hr;
    }
    // SAFETY: non-null, filled by the original call.
    let n = unsafe { *inout };
    if n >= cap || !std::panic::catch_unwind(|| kind(dev) == Kind::Mouse).unwrap_or(false) {
        return hr;
    }
    let dx = PENDING_X.swap(0, Ordering::Relaxed);
    if dx == 0 {
        return hr;
    }
    let cb = cb as usize;
    // SAFETY: the game's buffer holds `cap` records of `cb` bytes and `n < cap`; the
    // previous record (if any) was written by the original call.
    unsafe {
        let rec = data.add(n as usize * cb);
        let (ts, seq) = if n > 0 {
            let prev = data.add((n as usize - 1) * cb);
            (
                prev.add(8).cast::<u32>().read_unaligned(),
                prev.add(12).cast::<u32>().read_unaligned().wrapping_add(1),
            )
        } else {
            (windows::Win32::System::SystemInformation::GetTickCount(), 0)
        };
        std::ptr::write_bytes(rec, 0, cb);
        rec.cast::<u32>().write_unaligned(0); // DIMOFS_X
        rec.add(4).cast::<u32>().write_unaligned(dx as u32);
        rec.add(8).cast::<u32>().write_unaligned(ts);
        rec.add(12).cast::<u32>().write_unaligned(seq);
        *inout = n + 1;
    }
    DELIVERED_X.fetch_add(i64::from(dx), Ordering::Relaxed);
    hr
}

unsafe extern "system" fn get_device_state<const I: usize>(
    dev: *mut c_void,
    cb: u32,
    data: *mut u8,
) -> HRESULT {
    // SAFETY: MinHook trampoline with this signature.
    let f = unsafe {
        std::mem::transmute::<usize, GetDeviceState>(STATE_ORIG[I].load(Ordering::Relaxed))
    };
    // SAFETY: arguments passed through unchanged.
    let hr = unsafe { f(dev, cb, data) };
    if !data.is_null() && (blocked_for(dev) || captured_kind(dev).is_some()) {
        // SAFETY: the game's state buffer of `cb` bytes: no keys, buttons or motion.
        unsafe { std::ptr::write_bytes(data, 0, cb as usize) };
    }
    hr
}

unsafe extern "system" fn acquire<const I: usize>(dev: *mut c_void) -> HRESULT {
    if blocked_for(dev) {
        return DIERR_OTHERAPPHASPRIO;
    }
    // SAFETY: MinHook trampoline with this signature; argument unchanged.
    unsafe { std::mem::transmute::<usize, Method0>(ACQUIRE_ORIG[I].load(Ordering::Relaxed))(dev) }
}

/// Discards the buffered events of `dev` (`GetDeviceData` with a null buffer and an
/// INFINITE count).
///
/// # Safety
/// `dev` must be a live IDirectInputDevice8.
unsafe fn flush(dev: *mut c_void) {
    let mut n = u32::MAX;
    // SAFETY: per the caller contract; the null buffer is the documented flush form.
    let _ = unsafe {
        std::mem::transmute::<usize, GetDeviceData>(vtable_entry(dev, SLOT_GET_DEVICE_DATA))(
            dev,
            RECORD_SIZE,
            std::ptr::null_mut(),
            &mut n,
            0,
        )
    };
}

unsafe extern "system" fn mouse_poll(this: u64) -> u64 {
    if BLOCKED.load(Ordering::Relaxed)
        && let Some(dev) = mem::u64_at(this + MOUSE_DEVICE).filter(|&d| d != 0)
    {
        // SAFETY: the engine's mouse device (idMouseWin32 +0x178), live while it polls.
        unsafe { flush(dev as *mut c_void) };
    }
    // SAFETY: MinHook trampoline of the poll; argument unchanged.
    unsafe { std::mem::transmute::<usize, MousePoll>(POLL_ORIG.load(Ordering::Relaxed))(this) }
}

/// Withholds mouse and keyboard input from the game (`true`) or gives it back.
pub fn set_blocked(on: bool) {
    if !INSTALLED.load(Ordering::Relaxed) || BLOCKED.swap(on, Ordering::Relaxed) == on {
        return;
    }
    let devices: Vec<(usize, Kind)> = DEVICES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|d| d.kind != Kind::Other && d.seen.elapsed() < SEEN_RECENTLY)
        .map(|d| (d.ptr, d.kind))
        .collect();
    for (ptr, kind) in devices {
        let dev = ptr as *mut c_void;
        // SAFETY: devices the game read from within the last seconds (its input devices
        // live for the whole session).
        unsafe {
            if on {
                let _ =
                    std::mem::transmute::<usize, Method0>(vtable_entry(dev, SLOT_UNACQUIRE))(dev);
            } else if kind == Kind::Mouse {
                flush(dev);
            }
        }
    }
}

/// Finds the system DirectInput device implementations through temporary mouse and
/// keyboard devices (wide and ANSI interfaces) and hooks `GetDeviceData`,
/// `GetDeviceState` and `Acquire`, plus the engine's mouse poll.
pub fn install() -> Result<(), String> {
    if INSTALLED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let create = ffi::real_create().ok_or("system dinput8.dll not loaded")?;
    // SAFETY: plain query for the executable's module handle.
    let hinst: HINSTANCE = unsafe { GetModuleHandleW(None) }
        .map_err(|e| format!("GetModuleHandleW: {e}"))?
        .into();
    // Distinct implementations per method: GetDeviceData, GetDeviceState, Acquire.
    let mut targets: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for iid in [IID_DIRECTINPUT8W, IID_DIRECTINPUT8A] {
        let mut di: *mut c_void = std::ptr::null_mut();
        // SAFETY: documented DirectInput8Create arguments with a valid out pointer.
        let hr = unsafe {
            create(
                hinst,
                DIRECTINPUT_VERSION,
                &iid,
                &mut di,
                std::ptr::null_mut(),
            )
        };
        if hr.is_err() || di.is_null() {
            return Err(format!("DirectInput8Create: {hr:?}"));
        }
        for guid in [GUID_SYSMOUSE, GUID_SYSKEYBOARD] {
            let mut dev: *mut c_void = std::ptr::null_mut();
            // SAFETY: `di` is a live IDirectInput8; CreateDevice is slot 3.
            let hr = unsafe {
                std::mem::transmute::<usize, CreateDevice>(vtable_entry(di, SLOT_CREATE_DEVICE))(
                    di,
                    &guid,
                    &mut dev,
                    std::ptr::null_mut(),
                )
            };
            if hr.is_err() || dev.is_null() {
                continue;
            }
            for (t, slot) in
                targets
                    .iter_mut()
                    .zip([SLOT_GET_DEVICE_DATA, SLOT_GET_DEVICE_STATE, SLOT_ACQUIRE])
            {
                // SAFETY: `dev` is a live IDirectInputDevice8.
                let f = unsafe { vtable_entry(dev, slot) };
                if !t.contains(&f) && t.len() < IMPLS {
                    t.push(f);
                }
            }
            // SAFETY: releasing the temporary device.
            unsafe { std::mem::transmute::<usize, Release>(vtable_entry(dev, SLOT_RELEASE))(dev) };
        }
        // SAFETY: releasing the temporary DirectInput object.
        unsafe { std::mem::transmute::<usize, Release>(vtable_entry(di, SLOT_RELEASE))(di) };
    }
    if targets[0].is_empty() {
        return Err("no mouse or keyboard device".into());
    }
    let data: [GetDeviceData; IMPLS] = [
        get_device_data::<0>,
        get_device_data::<1>,
        get_device_data::<2>,
        get_device_data::<3>,
    ];
    let state: [GetDeviceState; IMPLS] = [
        get_device_state::<0>,
        get_device_state::<1>,
        get_device_state::<2>,
        get_device_state::<3>,
    ];
    let acq: [Method0; IMPLS] = [acquire::<0>, acquire::<1>, acquire::<2>, acquire::<3>];
    let hooks: [(&str, Vec<*mut c_void>, &[AtomicUsize; IMPLS]); 3] = [
        (
            "GetDeviceData",
            data.iter().map(|f| *f as *mut c_void).collect(),
            &DATA_ORIG,
        ),
        (
            "GetDeviceState",
            state.iter().map(|f| *f as *mut c_void).collect(),
            &STATE_ORIG,
        ),
        (
            "Acquire",
            acq.iter().map(|f| *f as *mut c_void).collect(),
            &ACQUIRE_ORIG,
        ),
    ];
    for ((name, detours, orig), t) in hooks.iter().zip(&targets) {
        for (i, &target) in t.iter().enumerate() {
            // SAFETY: dinput8.dll implementation of this method; same signature.
            let tr = unsafe { minhook::MinHook::create_hook(target as *mut c_void, detours[i]) }
                .map_err(|e| format!("create_hook {name}: {e:?}"))?;
            orig[i].store(tr as usize, Ordering::Relaxed);
            // SAFETY: hook created above; the trampoline is stored.
            unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
                .map_err(|e| format!("enable_hook {name}: {e:?}"))?;
        }
    }
    INSTALLED.store(true, Ordering::Relaxed);
    log::line(&format!(
        "mouse: DirectInput GetDeviceData {:x?}, GetDeviceState {:x?}, Acquire {:x?} hooked",
        targets[0], targets[1], targets[2]
    ));
    match install_poll() {
        Ok(a) => log::line(&format!("mouse: engine mouse poll hooked at {a:#x}")),
        Err(e) => log::line(&format!("mouse: engine mouse poll not hooked: {e}")),
    }
    Ok(())
}

fn install_poll() -> Result<u64, String> {
    let target = mem::image_base().ok_or("no image base")? + MOUSE_POLL_RVA;
    let mut head = [0u8; 16];
    if !mem::read_into(target, &mut head) || head != MOUSE_POLL_HEAD {
        return Err(format!("bytes {head:02x?} at {target:#x} differ"));
    }
    // SAFETY: verified prologue of `sub_1400E2930(this)`; same signature.
    let tr =
        unsafe { minhook::MinHook::create_hook(target as *mut c_void, mouse_poll as *mut c_void) }
            .map_err(|e| format!("create_hook: {e:?}"))?;
    POLL_ORIG.store(tr as usize, Ordering::Relaxed);
    // SAFETY: hook created above; the trampoline is stored.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook: {e:?}"))?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_keeps_motion_across_frames() {
        set_capture(true);
        // The engine's poll collects counts; the photo tick re-asserts the capture, then
        // takes them.
        MOTION_X.fetch_add(12, Ordering::Relaxed);
        MOTION_Y.fetch_add(-3, Ordering::Relaxed);
        set_capture(true);
        assert_eq!(take_motion(), (12, -3));
        assert_eq!(take_motion(), (0, 0));
        MOTION_X.fetch_add(7, Ordering::Relaxed);
        set_capture(false);
        set_capture(true);
        assert_eq!(take_motion(), (0, 0));
        set_capture(false);
    }
}
