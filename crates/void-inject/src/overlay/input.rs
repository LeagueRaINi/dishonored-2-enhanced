//! Overlay input: the game window's WndProc is subclassed (on the first opening) and,
//! while the overlay is visible, mouse/keyboard window messages are translated into
//! egui events and not passed to the game. The cursor functions the game uses for mouse
//! look and confinement (`SetCursorPos`, `GetCursorPos`, `ClipCursor`, user32) are hooked
//! as in d2enhanced `overlay.cpp`: while visible, the game's own re-centring is skipped,
//! its `GetCursorPos` calls see the position it last set (zero look delta), and clipping
//! is lifted. Calls from other modules (the Steam overlay, Windows) are untouched: callers
//! are told apart by their return address in the executable's image.
//!
//! Positions are queued in client pixels; `overlay.rs` maps them to egui points.

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, AtomicUsize, Ordering};

use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetKeyState, VIRTUAL_KEY, VK_BACK, VK_CONTROL, VK_DELETE, VK_DOWN, VK_END, VK_ESCAPE, VK_HOME,
    VK_INSERT, VK_LEFT, VK_MENU, VK_NEXT, VK_PRIOR, VK_RETURN, VK_RIGHT, VK_SHIFT, VK_SPACE,
    VK_TAB, VK_UP,
};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{s, w};

use egui::{Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, TouchPhase, Vec2};

use crate::{log, mem};

/// Events since the last overlay frame (pointer positions in client pixels).
static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static GAME_WNDPROC: AtomicIsize = AtomicIsize::new(0);
static WINDOW: AtomicIsize = AtomicIsize::new(0);
static FOCUSED: AtomicBool = AtomicBool::new(true);

/// Queues an event (also used by the offline preview for synthetic input).
pub fn push(e: Event) {
    EVENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(e);
}

/// Takes the queued events.
pub fn take() -> Vec<Event> {
    std::mem::take(
        &mut *EVENTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

pub fn focused() -> bool {
    FOCUSED.load(Ordering::Relaxed)
}

fn key_held(vk: VIRTUAL_KEY) -> bool {
    // SAFETY: plain query.
    let state = unsafe { GetKeyState(i32::from(vk.0)) };
    state < 0
}

pub fn modifiers() -> Modifiers {
    let ctrl = key_held(VK_CONTROL);
    Modifiers {
        alt: key_held(VK_MENU),
        ctrl,
        shift: key_held(VK_SHIFT),
        mac_cmd: false,
        command: ctrl,
    }
}

/// Window virtual key -> egui key.
fn key(vk: u32) -> Option<Key> {
    let k = VIRTUAL_KEY(vk as u16);
    Some(match k {
        VK_LEFT => Key::ArrowLeft,
        VK_RIGHT => Key::ArrowRight,
        VK_UP => Key::ArrowUp,
        VK_DOWN => Key::ArrowDown,
        VK_ESCAPE => Key::Escape,
        VK_TAB => Key::Tab,
        VK_BACK => Key::Backspace,
        VK_RETURN => Key::Enter,
        VK_SPACE => Key::Space,
        VK_INSERT => Key::Insert,
        VK_DELETE => Key::Delete,
        VK_HOME => Key::Home,
        VK_END => Key::End,
        VK_PRIOR => Key::PageUp,
        VK_NEXT => Key::PageDown,
        _ => match vk {
            0x30..=0x39 | 0x41..=0x5A => {
                return Key::from_name(&char::from(vk as u8).to_string());
            }
            0x70..=0x87 => return Key::from_name(&format!("F{}", vk - 0x6F)),
            0xBD | 0x6D => Key::Minus,
            0xBB | 0x6B => Key::Plus,
            0xBE | 0x6E => Key::Period,
            0xBC => Key::Comma,
            _ => return None,
        },
    })
}

fn lparam_pos(l: LPARAM) -> Pos2 {
    let x = (l.0 & 0xFFFF) as u16 as i16;
    let y = ((l.0 >> 16) & 0xFFFF) as u16 as i16;
    Pos2::new(f32::from(x), f32::from(y))
}

fn is_input(msg: u32) -> bool {
    (WM_MOUSEFIRST..=WM_MOUSELAST).contains(&msg)
        || (WM_KEYFIRST..=WM_KEYLAST).contains(&msg)
        || msg == WM_INPUT
}

/// Translates one window message into egui events.
fn translate(msg: u32, wp: WPARAM, lp: LPARAM) {
    let m = modifiers();
    let button = |b, pressed| {
        push(Event::PointerButton {
            pos: lparam_pos(lp),
            button: b,
            pressed,
            modifiers: m,
        });
    };
    match msg {
        WM_MOUSEMOVE => push(Event::PointerMoved(lparam_pos(lp))),
        WM_LBUTTONDOWN | WM_LBUTTONDBLCLK => button(PointerButton::Primary, true),
        WM_LBUTTONUP => button(PointerButton::Primary, false),
        WM_RBUTTONDOWN | WM_RBUTTONDBLCLK => button(PointerButton::Secondary, true),
        WM_RBUTTONUP => button(PointerButton::Secondary, false),
        WM_MBUTTONDOWN | WM_MBUTTONDBLCLK => button(PointerButton::Middle, true),
        WM_MBUTTONUP => button(PointerButton::Middle, false),
        WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
            let d = f32::from((wp.0 >> 16) as u16 as i16) / 120.0;
            let delta = if msg == WM_MOUSEWHEEL {
                Vec2::new(0.0, d)
            } else {
                Vec2::new(-d, 0.0)
            };
            push(Event::MouseWheel {
                unit: MouseWheelUnit::Line,
                delta,
                phase: TouchPhase::Move,
                modifiers: m,
            });
        }
        WM_KEYDOWN | WM_SYSKEYDOWN | WM_KEYUP | WM_SYSKEYUP => {
            if let Some(k) = key(wp.0 as u32) {
                push(Event::Key {
                    key: k,
                    physical_key: None,
                    pressed: msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN,
                    repeat: (lp.0 >> 30) & 1 != 0 && (msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN),
                    modifiers: m,
                });
            }
        }
        WM_CHAR => {
            if let Some(c) = char::from_u32(wp.0 as u32).filter(|c| !c.is_control()) {
                push(Event::Text(c.to_string()));
            }
        }
        _ => {}
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_SETFOCUS => FOCUSED.store(true, Ordering::Relaxed),
        WM_KILLFOCUS => FOCUSED.store(false, Ordering::Relaxed),
        _ => {}
    }
    if super::visible() && is_input(msg) {
        let _ = std::panic::catch_unwind(|| translate(msg, wp, lp));
        return LRESULT(0);
    }
    let prev = GAME_WNDPROC.load(Ordering::Relaxed);
    // SAFETY: the game's window procedure as returned by SetWindowLongPtrW.
    unsafe {
        CallWindowProcW(
            std::mem::transmute::<isize, WNDPROC>(prev),
            hwnd,
            msg,
            wp,
            lp,
        )
    }
}

/// Subclasses the game window once.
pub fn subclass(hwnd: HWND) -> Result<(), String> {
    if WINDOW.load(Ordering::Relaxed) != 0 {
        return Ok(());
    }
    // SAFETY: a window of this process; the replacement procedure forwards to the
    // previous one for everything it does not consume.
    let prev = unsafe { SetWindowLongPtrW(hwnd, GWLP_WNDPROC, wndproc as *const () as isize) };
    if prev == 0 {
        return Err("SetWindowLongPtrW failed".into());
    }
    GAME_WNDPROC.store(prev, Ordering::Relaxed);
    WINDOW.store(hwnd.0 as isize, Ordering::Relaxed);
    log::line(&format!(
        "overlay: window {:#x} subclassed",
        hwnd.0 as usize
    ));
    Ok(())
}

/// Client-area size of the game window (pixels).
pub fn client_size(hwnd: HWND) -> Option<[f32; 2]> {
    let mut r = RECT::default();
    // SAFETY: valid out pointer.
    unsafe { GetClientRect(hwnd, &mut r) }.ok()?;
    let (w, h) = (r.right - r.left, r.bottom - r.top);
    (w > 0 && h > 0).then_some([w as f32, h as f32])
}

// ---------------------------------------------------------------- cursor hooks --

type SetCursorPosFn = unsafe extern "system" fn(i32, i32) -> i32;
type GetCursorPosFn = unsafe extern "system" fn(*mut POINT) -> i32;
type ClipCursorFn = unsafe extern "system" fn(*const RECT) -> i32;

static SET_ORIG: AtomicUsize = AtomicUsize::new(0);
static GET_ORIG: AtomicUsize = AtomicUsize::new(0);
static CLIP_ORIG: AtomicUsize = AtomicUsize::new(0);
/// The cursor position the game last set (mouse-look re-centring).
static GAME_X: AtomicI32 = AtomicI32::new(0);
static GAME_Y: AtomicI32 = AtomicI32::new(0);
static HAVE_GAME_POS: AtomicBool = AtomicBool::new(false);
static EXE: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// The caller of the hooked function is executable code (not a DLL). Must be called
/// directly from the detour (MinHook jumps to the detour, so the detour's return address
/// is the call site of the hooked API).
#[inline(never)]
fn called_from_game() -> bool {
    let mut frames = [std::ptr::null_mut::<c_void>(); 1];
    // Frame 0 is this function, frame 1 the detour; frame 2 is the API's caller.
    // SAFETY: valid output buffer.
    let n = unsafe {
        windows::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, &mut frames, None)
    };
    let (lo, hi) = *EXE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    n == 1 && (lo..hi).contains(&(frames[0] as u64))
}

unsafe extern "system" fn set_cursor_pos(x: i32, y: i32) -> i32 {
    if called_from_game() {
        GAME_X.store(x, Ordering::Relaxed);
        GAME_Y.store(y, Ordering::Relaxed);
        HAVE_GAME_POS.store(true, Ordering::Relaxed);
        if super::visible() {
            return 1;
        }
    }
    // SAFETY: trampoline of user32!SetCursorPos.
    unsafe { std::mem::transmute::<usize, SetCursorPosFn>(SET_ORIG.load(Ordering::Relaxed))(x, y) }
}

unsafe extern "system" fn get_cursor_pos(p: *mut POINT) -> i32 {
    if super::visible()
        && HAVE_GAME_POS.load(Ordering::Relaxed)
        && !p.is_null()
        && called_from_game()
    {
        // SAFETY: the caller's output pointer.
        unsafe {
            *p = POINT {
                x: GAME_X.load(Ordering::Relaxed),
                y: GAME_Y.load(Ordering::Relaxed),
            };
        }
        return 1;
    }
    // SAFETY: trampoline of user32!GetCursorPos.
    unsafe { std::mem::transmute::<usize, GetCursorPosFn>(GET_ORIG.load(Ordering::Relaxed))(p) }
}

unsafe extern "system" fn clip_cursor(r: *const RECT) -> i32 {
    let r = if super::visible() {
        std::ptr::null()
    } else {
        r
    };
    // SAFETY: trampoline of user32!ClipCursor.
    unsafe { std::mem::transmute::<usize, ClipCursorFn>(CLIP_ORIG.load(Ordering::Relaxed))(r) }
}

/// Hooks `SetCursorPos`, `GetCursorPos` and `ClipCursor` (user32).
pub fn install_cursor_hooks() -> Result<(), String> {
    let base = mem::image_base().ok_or("no image base")?;
    // PE: e_lfanew at +0x3C, SizeOfImage at NT headers +80.
    let size = mem::u32_at(base + 0x3C)
        .and_then(|l| mem::u32_at(base + u64::from(l) + 80))
        .ok_or("PE header unreadable")?;
    *EXE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = (base, base + u64::from(size));
    // SAFETY: loading a system DLL by name (already loaded in a GUI process).
    let user32 =
        unsafe { GetModuleHandleW(w!("user32.dll")).or_else(|_| LoadLibraryW(w!("user32.dll"))) }
            .map_err(|e| format!("user32: {e}"))?;
    let hooks: [(windows::core::PCSTR, *mut c_void, &AtomicUsize); 3] = [
        (s!("SetCursorPos"), set_cursor_pos as *mut c_void, &SET_ORIG),
        (s!("GetCursorPos"), get_cursor_pos as *mut c_void, &GET_ORIG),
        (s!("ClipCursor"), clip_cursor as *mut c_void, &CLIP_ORIG),
    ];
    for (name, detour, orig) in hooks {
        // SAFETY: valid module handle and NUL-terminated name.
        let target = unsafe { GetProcAddress(user32, name) }
            .ok_or_else(|| "user32 export missing".to_owned())? as *mut c_void;
        // SAFETY: the user32 export; the detour has its signature.
        let tr = unsafe { minhook::MinHook::create_hook(target, detour) }
            .map_err(|e| format!("create_hook cursor: {e:?}"))?;
        orig.store(tr as usize, Ordering::Relaxed);
        // SAFETY: hook created above; the trampoline is stored.
        unsafe { minhook::MinHook::enable_hook(target) }
            .map_err(|e| format!("enable_hook cursor: {e:?}"))?;
    }
    Ok(())
}

/// Overlay opened: lift the game's cursor confinement.
pub fn on_open() {
    let f = CLIP_ORIG.load(Ordering::Relaxed);
    if f != 0 {
        // SAFETY: trampoline of user32!ClipCursor; null = no confinement.
        unsafe { std::mem::transmute::<usize, ClipCursorFn>(f)(std::ptr::null()) };
    }
    take();
}

/// Overlay closed: put the cursor back where the game last placed it, so its next
/// `GetCursorPos` reads no movement (no mouse-look jump).
pub fn on_close() {
    let f = SET_ORIG.load(Ordering::Relaxed);
    if f != 0 && HAVE_GAME_POS.load(Ordering::Relaxed) {
        // SAFETY: trampoline of user32!SetCursorPos.
        unsafe {
            std::mem::transmute::<usize, SetCursorPosFn>(f)(
                GAME_X.load(Ordering::Relaxed),
                GAME_Y.load(Ordering::Relaxed),
            )
        };
    }
    take();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_positions() {
        assert_eq!(key(0x41), Some(Key::A));
        assert_eq!(key(0x30), Some(Key::Num0));
        assert_eq!(key(0x71), Some(Key::F2));
        assert_eq!(key(VK_RETURN.0.into()), Some(Key::Enter));
        assert_eq!(key(0x07), None);
        assert_eq!(lparam_pos(LPARAM(0x0010_0020)), Pos2::new(32.0, 16.0));
        assert_eq!(lparam_pos(LPARAM(0xFFFF_FFFF)), Pos2::new(-1.0, -1.0));
    }
}
