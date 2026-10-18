//! DllMain and the forwarded `DirectInput8Create` export.
//!
//! Invariants: DllMain does no allocation-heavy work and takes no locks besides
//! spawning the initialization thread; the system `dinput8.dll` is loaded lazily on
//! the first `DirectInput8Create` call (outside the loader lock).

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Foundation::{HINSTANCE, HMODULE};
use windows::Win32::System::LibraryLoader::{
    DisableThreadLibraryCalls, GetModuleFileNameW, GetProcAddress, LoadLibraryW,
};
use windows::Win32::System::SystemInformation::GetSystemDirectoryW;
use windows::Win32::System::SystemServices::DLL_PROCESS_ATTACH;
use windows::core::{GUID, HRESULT, PCWSTR, s};

pub(crate) type Create = unsafe extern "system" fn(
    hinst: HINSTANCE,
    version: u32,
    riid: *const GUID,
    out: *mut *mut c_void,
    outer: *mut c_void,
) -> HRESULT;

static REAL: OnceLock<Option<usize>> = OnceLock::new();

pub(crate) fn real_create() -> Option<Create> {
    let addr = (*REAL.get_or_init(|| {
        let mut buf = [0u16; 260];
        // SAFETY: buffer is valid for its length.
        let n = unsafe { GetSystemDirectoryW(Some(&mut buf)) } as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        let mut path: Vec<u16> = buf[..n].to_vec();
        path.extend("\\dinput8.dll\0".encode_utf16());
        // SAFETY: `path` is NUL-terminated and outlives the call.
        let module = unsafe { LoadLibraryW(PCWSTR(path.as_ptr())) }.ok()?;
        // SAFETY: valid module handle and NUL-terminated export name.
        let proc = unsafe { GetProcAddress(module, s!("DirectInput8Create")) }?;
        Some(proc as usize)
    }))?;
    // SAFETY: the address is the system export with the documented signature.
    Some(unsafe { std::mem::transmute::<usize, Create>(addr) })
}

/// Forwarded export used by the game for input.
///
/// # Safety
/// Called by the game with the arguments of the real `DirectInput8Create`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DirectInput8Create(
    hinst: HINSTANCE,
    version: u32,
    riid: *const GUID,
    out: *mut *mut c_void,
    outer: *mut c_void,
) -> HRESULT {
    match real_create() {
        // SAFETY: arguments are passed through unchanged.
        Some(f) => unsafe { f(hinst, version, riid, out, outer) },
        None => HRESULT(0x8007_007Eu32 as i32), // ERROR_MOD_NOT_FOUND
    }
}

/// True while the virtual key is held (global async key state).
pub fn key_down(vk: i32) -> bool {
    // SAFETY: plain query without pointers.
    let state = unsafe { windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState(vk) };
    state < 0
}

pub(crate) fn exe_path() -> Option<std::path::PathBuf> {
    let mut buf = vec![0u16; 1024];
    // SAFETY: buffer valid; `None` selects the process executable.
    let n = unsafe { GetModuleFileNameW(None::<HMODULE>, &mut buf) } as usize;
    (n > 0 && n < buf.len()).then(|| std::path::PathBuf::from(String::from_utf16_lossy(&buf[..n])))
}

/// # Safety
/// Called by the Windows loader.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllMain(
    module: HINSTANCE,
    reason: u32,
    _reserved: *mut c_void,
) -> i32 {
    if reason == DLL_PROCESS_ATTACH {
        // SAFETY: valid module handle from the loader.
        let _ = unsafe { DisableThreadLibraryCalls(HMODULE(module.0)) };
        // Before any game code runs (this DLL is an import of the executable): settings
        // and the boot-time code patches (patches.rs).
        crate::boot();
        // Texture mods: hooked before WinMain opens any resource (texmods.rs explains).
        crate::texmods::install_early();
        // Thread creation is allowed under the loader lock; the thread itself waits
        // for the lock to be released before running.
        let _ = std::thread::Builder::new()
            .name("void-inject-init".into())
            .spawn(|| {
                if let Some(exe) = exe_path() {
                    let _ = std::panic::catch_unwind(|| crate::initialize(exe));
                }
            });
    }
    1
}
