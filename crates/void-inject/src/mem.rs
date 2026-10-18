//! Checked reads of the game's memory.
//!
//! Reads go through `ReadProcessMemory` on the current process, which fails with an
//! error (instead of an access violation) when an address is not readable. Callers
//! therefore never dereference game pointers directly.

use std::ffi::c_void;

use iced_x86::BlockEncoderOptions;
use iced_x86::code_asm::{AsmRegister64, CodeAssembler, r9, rax};

use windows::Win32::System::Diagnostics::Debug::ReadProcessMemory;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::GetCurrentProcess;

/// Copies `out.len()` bytes from `addr`; false if any byte is unreadable.
pub fn read_into(addr: u64, out: &mut [u8]) -> bool {
    if addr == 0 {
        return false;
    }
    let mut done = 0usize;
    // SAFETY: `out` is valid for writes of its length; the source is validated by the
    // kernel and a failure is reported instead of faulting.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            addr as *const c_void,
            out.as_mut_ptr().cast(),
            out.len(),
            Some(&mut done),
        )
    };
    ok.is_ok() && done == out.len()
}

/// Writes `bytes` at `addr` (checked by the kernel like `read_into`); false on failure.
pub fn write_bytes(addr: u64, bytes: &[u8]) -> bool {
    use windows::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    if addr == 0 {
        return false;
    }
    let mut done = 0usize;
    // SAFETY: the source is a live local slice; the destination is validated by the kernel.
    let ok = unsafe {
        WriteProcessMemory(
            GetCurrentProcess(),
            addr as *const c_void,
            bytes.as_ptr().cast(),
            bytes.len(),
            Some(&mut done),
        )
    };
    ok.is_ok() && done == bytes.len()
}

pub fn write_u32(addr: u64, value: u32) -> bool {
    write_bytes(addr, &value.to_le_bytes())
}

/// Makes `len` bytes at `addr` (e.g. a constant table in `.rdata`) read-write for good.
pub fn make_writable(addr: u64, len: usize) -> Result<(), String> {
    use windows::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, PAGE_READWRITE, VirtualProtect};
    let mut probe = vec![0u8; len];
    if !read_into(addr, &mut probe) {
        return Err(format!("{addr:#x} not readable"));
    }
    let mut old = PAGE_PROTECTION_FLAGS::default();
    // SAFETY: the range was just read (mapped); only its protection changes.
    unsafe { VirtualProtect(addr as *const c_void, len, PAGE_READWRITE, &mut old) }
        .map_err(|e| format!("VirtualProtect: {e}"))
}

/// Replaces the code bytes `expected` at `addr` with `new` (same length) if, and only if,
/// they match exactly; restores the page protection afterwards.
///
/// Safe while the game runs (MinHook's technique for its own hooks): every other thread of
/// the process is suspended for the write, and the write only happens when no suspended
/// thread's instruction pointer lies strictly inside the replaced bytes (a thread at the
/// first byte executes the new instruction; the old and new code are whole instructions
/// of the same total length, so return addresses stay valid). On a conflict the threads
/// are resumed and the attempt repeated after 1 ms, up to [`PATCH_ATTEMPTS`] times. No heap
/// allocation happens while threads are suspended (one of them may hold the heap lock).
pub fn patch_code(addr: u64, expected: &[u8], new: &[u8]) -> Result<(), String> {
    if expected.len() != new.len() {
        return Err("length mismatch".into());
    }
    let mut cur = vec![0u8; expected.len()];
    if !read_into(addr, &mut cur) {
        return Err(format!("{addr:#x} not readable"));
    }
    if cur != expected {
        return Err(format!("unexpected bytes {cur:02x?} at {addr:#x}"));
    }
    let threads = Threads::open_others()?;
    for _ in 0..PATCH_ATTEMPTS {
        threads.suspend();
        let busy = threads.ip_inside(addr + 1, addr + new.len() as u64);
        let written = if busy {
            Ok(false)
        } else {
            write_code(addr, expected, new, &mut cur).map(|()| true)
        };
        threads.resume();
        match written {
            Ok(true) => return Ok(()),
            Ok(false) => std::thread::sleep(std::time::Duration::from_millis(1)),
            Err(WriteError::Changed) => {
                return Err(format!("bytes at {addr:#x} changed during the patch"));
            }
            Err(WriteError::Protect) => return Err(format!("VirtualProtect failed at {addr:#x}")),
        }
    }
    Err(format!(
        "a thread kept executing the bytes at {addr:#x} ({PATCH_ATTEMPTS} attempts)"
    ))
}

/// Attempts of [`patch_code`] before giving up on a thread inside the patched bytes.
const PATCH_ATTEMPTS: u32 = 200;

/// Failure of [`write_code`] (no `String`: it runs with threads suspended).
enum WriteError {
    Changed,
    Protect,
}

/// The write part of [`patch_code`] (threads suspended): re-checks the bytes into the
/// preallocated `scratch`, makes the range writable, copies, restores the protection and
/// flushes the instruction cache. Allocates nothing.
fn write_code(
    addr: u64,
    expected: &[u8],
    new: &[u8],
    scratch: &mut [u8],
) -> Result<(), WriteError> {
    use windows::Win32::System::Memory::{
        PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualProtect,
    };
    if !read_into(addr, scratch) || scratch != expected {
        return Err(WriteError::Changed);
    }
    let mut old = PAGE_PROTECTION_FLAGS::default();
    // SAFETY: the range was just read (mapped); protection is restored below; the bytes
    // were verified, so this only changes the intended instruction; no other thread runs.
    unsafe {
        if VirtualProtect(
            addr as *const c_void,
            new.len(),
            PAGE_EXECUTE_READWRITE,
            &mut old,
        )
        .is_err()
        {
            return Err(WriteError::Protect);
        }
        std::ptr::copy_nonoverlapping(new.as_ptr(), addr as *mut u8, new.len());
        let mut tmp = PAGE_PROTECTION_FLAGS::default();
        let _ = VirtualProtect(addr as *const c_void, new.len(), old, &mut tmp);
        let _ = windows::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            GetCurrentProcess(),
            Some(addr as *const c_void),
            new.len(),
        );
    }
    Ok(())
}

/// Handles of every other thread of this process (opened before any is suspended).
struct Threads {
    handles: Vec<windows::Win32::Foundation::HANDLE>,
    /// Which handles `suspend` actually suspended (preallocated, same length).
    suspended: std::cell::RefCell<Vec<bool>>,
}

/// `CONTEXT` must be 16-byte aligned for `GetThreadContext`.
#[repr(C, align(16))]
struct AlignedContext(windows::Win32::System::Diagnostics::Debug::CONTEXT);

impl Threads {
    fn open_others() -> Result<Self, String> {
        use windows::Win32::System::Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
        };
        use windows::Win32::System::Threading::{
            GetCurrentProcessId, GetCurrentThreadId, OpenThread, THREAD_GET_CONTEXT,
            THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
        };
        // SAFETY: plain Toolhelp enumeration with a correctly sized entry; the snapshot
        // and every opened thread handle are closed (snapshot here, threads in `Drop`).
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
                .map_err(|e| format!("thread snapshot: {e}"))?;
            let (pid, me) = (GetCurrentProcessId(), GetCurrentThreadId());
            let mut entry = THREADENTRY32 {
                dwSize: std::mem::size_of::<THREADENTRY32>() as u32,
                ..Default::default()
            };
            let mut handles = Vec::new();
            let mut more = Thread32First(snap, &mut entry).is_ok();
            while more {
                if entry.th32OwnerProcessID == pid
                    && entry.th32ThreadID != me
                    && let Ok(h) = OpenThread(
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                        false,
                        entry.th32ThreadID,
                    )
                {
                    handles.push(h);
                }
                more = Thread32Next(snap, &mut entry).is_ok();
            }
            let _ = windows::Win32::Foundation::CloseHandle(snap);
            let n = handles.len();
            Ok(Self {
                handles,
                suspended: std::cell::RefCell::new(vec![false; n]),
            })
        }
    }

    fn suspend(&self) {
        let mut s = self.suspended.borrow_mut();
        for (h, done) in self.handles.iter().zip(s.iter_mut()) {
            // SAFETY: our own open thread handle; failure (thread exited) returns u32::MAX.
            *done = unsafe { windows::Win32::System::Threading::SuspendThread(*h) } != u32::MAX;
        }
    }

    fn resume(&self) {
        let mut s = self.suspended.borrow_mut();
        for (h, done) in self.handles.iter().zip(s.iter_mut()) {
            if std::mem::take(done) {
                // SAFETY: resumes exactly the suspension made by `suspend`.
                unsafe { windows::Win32::System::Threading::ResumeThread(*h) };
            }
        }
    }

    /// True if a suspended thread's RIP is in `[lo, hi)`.
    fn ip_inside(&self, lo: u64, hi: u64) -> bool {
        use windows::Win32::System::Diagnostics::Debug::{CONTEXT_CONTROL_AMD64, GetThreadContext};
        let s = self.suspended.borrow();
        self.handles.iter().zip(s.iter()).any(|(h, &done)| {
            if !done {
                return false;
            }
            // SAFETY: zeroed CONTEXT is a valid bit pattern; `GetThreadContext` on a
            // suspended thread with an aligned, flagged context.
            unsafe {
                let mut c: AlignedContext = std::mem::zeroed();
                c.0.ContextFlags = CONTEXT_CONTROL_AMD64;
                GetThreadContext(*h, &mut c.0).is_ok() && (lo..hi).contains(&c.0.Rip)
            }
        })
    }
}

impl Drop for Threads {
    fn drop(&mut self) {
        self.resume();
        for h in &self.handles {
            // SAFETY: handles opened in `open_others`, closed once.
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(*h);
            }
        }
    }
}

/// Allocates `code` as executable memory within +-2 GiB of `near` (so a `call rel32` at
/// `near` can reach it) and returns its address. Searches downwards from `near` in 64 KiB
/// steps (allocation granularity), as MinHook does for its trampolines.
pub fn alloc_code_near(near: u64, code: &[u8]) -> Result<u64, String> {
    use windows::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS,
        VirtualAlloc, VirtualProtect,
    };
    const GRANULE: u64 = 0x1_0000;
    const REACH: u64 = 0x7FF0_0000;
    let start = (near & !(GRANULE - 1)).saturating_sub(GRANULE);
    let mut addr = start;
    while near - addr < REACH && addr >= GRANULE {
        // SAFETY: requesting fresh pages at a hint address; failure is reported as null.
        let p = unsafe {
            VirtualAlloc(
                Some(addr as *const c_void),
                code.len(),
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            )
        };
        if !p.is_null() {
            let p = p as u64;
            // SAFETY: `p` is a fresh writable allocation of at least `code.len()` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(code.as_ptr(), p as *mut u8, code.len());
                let mut old = PAGE_PROTECTION_FLAGS::default();
                let _ = VirtualProtect(p as *const c_void, code.len(), PAGE_EXECUTE_READ, &mut old);
                let _ = windows::Win32::System::Diagnostics::Debug::FlushInstructionCache(
                    GetCurrentProcess(),
                    Some(p as *const c_void),
                    code.len(),
                );
            }
            return Ok(p);
        }
        addr -= GRANULE;
    }
    Err(format!("no free memory within 2 GiB below {near:#x}"))
}

/// Redirects the 5-byte `call rel32` (`E8`) or 6-byte `call [rax+disp32]` (`FF 90`) at
/// `site` (whose exact bytes must equal `expected`) to a stub that copies `arg4` into
/// `r9` (the fourth argument, extra state for the target) and then jumps to `target`:
/// `mov r9, arg4; mov rax, target; jmp rax`. The stub is a tail jump, so `target` returns
/// straight to the instruction after the call and must follow the callee's convention.
pub fn redirect_call(
    site: u64,
    expected: &[u8],
    arg4: AsmRegister64,
    target: u64,
) -> Result<u64, String> {
    if !matches!(expected.len(), 5 | 6) || !matches!(expected.first(), Some(0xE8 | 0xFF)) {
        return Err("unsupported call instruction".into());
    }
    let at = alloc_code_near(site, &call_stub(arg4, target)?)?;
    let new = call_patch(site, at, expected.len()).map_err(|e| format!("stub at {at:#x}: {e}"))?;
    patch_code(site, expected, &new)?;
    Ok(at)
}

/// `mov r9, arg4; mov rax, target; jmp rax` (position independent).
fn call_stub(arg4: AsmRegister64, target: u64) -> Result<Vec<u8>, String> {
    let mut a = CodeAssembler::new(64).map_err(|e| e.to_string())?;
    a.mov(r9, arg4).map_err(|e| e.to_string())?;
    a.mov(rax, target).map_err(|e| e.to_string())?;
    a.jmp(rax).map_err(|e| e.to_string())?;
    a.assemble(0).map_err(|e| e.to_string())
}

/// `call stub` (rel32) at `site`, padded with `nop`s to `len` bytes; fails instead of
/// widening the call when `stub` is out of rel32 reach.
fn call_patch(site: u64, stub: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut a = CodeAssembler::new(64).map_err(|e| e.to_string())?;
    a.call(stub).map_err(|e| e.to_string())?;
    let mut code = a
        .assemble_options(site, BlockEncoderOptions::DONT_FIX_BRANCHES)
        .map_err(|e| e.to_string())?
        .inner
        .code_buffer;
    if code.len() > len {
        return Err(format!("call encodes to {} bytes", code.len()));
    }
    code.resize(len, 0x90);
    Ok(code)
}

pub fn u32_at(addr: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    read_into(addr, &mut b).then(|| u32::from_le_bytes(b))
}

pub fn u64_at(addr: u64) -> Option<u64> {
    let mut b = [0u8; 8];
    read_into(addr, &mut b).then(|| u64::from_le_bytes(b))
}

pub fn f32s<const N: usize>(addr: u64) -> Option<[f32; N]> {
    let mut b = vec![0u8; 4 * N];
    if !read_into(addr, &mut b) {
        return None;
    }
    let mut out = [0f32; N];
    for (o, c) in out.iter_mut().zip(b.as_chunks::<4>().0) {
        *o = f32::from_le_bytes(*c);
    }
    Some(out)
}

/// Base address of the executable image in this process.
pub fn image_base() -> Option<u64> {
    // SAFETY: `None` selects the process executable; no pointers are passed.
    unsafe { GetModuleHandleW(None) }.ok().map(|h| h.0 as u64)
}

/// Replaces the 8-byte, 8-aligned pointer at `addr` (e.g. an import address table slot)
/// with `new` if it holds `expected`, as one atomic compare-exchange so that concurrent
/// callers read either the old or the new pointer. The page protection is restored
/// afterwards.
pub fn swap_pointer(addr: u64, expected: u64, new: u64) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use windows::Win32::System::Memory::{PAGE_PROTECTION_FLAGS, PAGE_READWRITE, VirtualProtect};
    if !addr.is_multiple_of(8) {
        return Err(format!("{addr:#x} is not 8-byte aligned"));
    }
    match u64_at(addr) {
        Some(v) if v == expected => {}
        Some(v) => return Err(format!("{addr:#x} holds {v:#x}, expected {expected:#x}")),
        None => return Err(format!("{addr:#x} not readable")),
    }
    let mut old = PAGE_PROTECTION_FLAGS::default();
    // SAFETY: the 8 bytes were just read (mapped); only their page protection changes.
    unsafe { VirtualProtect(addr as *const c_void, 8, PAGE_READWRITE, &mut old) }
        .map_err(|e| format!("VirtualProtect: {e}"))?;
    // SAFETY: `addr` is mapped, writable now and 8-aligned; the slot is only ever accessed
    // as a whole pointer.
    let slot = unsafe { AtomicU64::from_ptr(addr as *mut u64) };
    let swapped = slot.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst);
    let mut ignored = PAGE_PROTECTION_FLAGS::default();
    // SAFETY: restores the protection read above on the same range.
    let _ = unsafe { VirtualProtect(addr as *const c_void, 8, old, &mut ignored) };
    swapped
        .map(|_| ())
        .map_err(|v| format!("{addr:#x} changed to {v:#x} during the swap"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::code_asm::{r14, rbx};
    use iced_x86::{Decoder, DecoderOptions};

    fn disassemble(code: &[u8], ip: u64) -> Vec<String> {
        Decoder::with_ip(64, code, ip, DecoderOptions::NONE)
            .into_iter()
            .map(|i| i.to_string())
            .collect()
    }

    #[test]
    fn call_stub_is_mov_mov_jmp() {
        let target = 0x7FF8_1234_5678_9ABC;
        for (reg, mov, old) in [
            (r14, "mov r9,r14", hex_literal::hex!("4D 89 F1")),
            (rbx, "mov r9,rbx", hex_literal::hex!("49 89 D9")),
        ] {
            let stub = call_stub(reg, target).unwrap();
            assert_eq!(
                disassemble(&stub, 0),
                [mov, "mov rax,7FF8123456789ABCh", "jmp rax"]
            );
            // Byte-identical to the former hand-written stub.
            let mut want = old.to_vec();
            want.extend(hex_literal::hex!("48 B8"));
            want.extend(target.to_le_bytes());
            want.extend(hex_literal::hex!("FF E0"));
            assert_eq!(stub, want);
        }
    }

    #[test]
    fn call_patch_is_rel32_call_plus_nops() {
        let site = 0x1_4032_7817;
        let stub = 0x1_4000_0000;
        let five = call_patch(site, stub, 5).unwrap();
        assert_eq!(disassemble(&five, site), ["call 0000000140000000h"]);
        let six = call_patch(site, stub, 6).unwrap();
        assert_eq!(disassemble(&six, site), ["call 0000000140000000h", "nop"]);
        assert!(call_patch(site, site + 0x1_0000_0000, 5).is_err());
        // Byte-identical to the former hand-written `E8 rel32` (+ `90`).
        let rel = i32::try_from(stub as i64 - (site as i64 + 5)).unwrap();
        let mut want = hex_literal::hex!("E8").to_vec();
        want.extend(rel.to_le_bytes());
        assert_eq!(five, want);
        want.extend(hex_literal::hex!("90"));
        assert_eq!(six, want);
    }

    /// `patch_code` retargets a `call rel32` while another thread keeps calling through it:
    /// the caller only ever sees one of the two targets (never torn code or a crash).
    #[test]
    fn patch_code_is_safe_while_another_thread_calls_through_the_site() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        // 0x00: sub rsp,28h; call A; add rsp,28h; ret   A (0x20): mov eax,1; ret
        //                                               B (0x30): mov eax,2; ret
        let mut code = vec![0xCCu8; 0x40];
        code[..4].copy_from_slice(&hex_literal::hex!("48 83 EC 28"));
        code[4] = 0xE8;
        code[9..14].copy_from_slice(&hex_literal::hex!("48 83 C4 28 C3"));
        code[0x20..0x26].copy_from_slice(&hex_literal::hex!("B8 01 00 00 00 C3"));
        code[0x30..0x36].copy_from_slice(&hex_literal::hex!("B8 02 00 00 00 C3"));
        let call = |to: i32| {
            let mut b = vec![0xE8];
            b.extend((to - 9).to_le_bytes());
            b
        };
        let (to_a, to_b) = (call(0x20), call(0x30));
        code[4..9].copy_from_slice(&to_a);
        let base = alloc_code_near(read_into as *const () as u64, &code).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let seen = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
        let caller = {
            let (stop, seen) = (stop.clone(), seen.clone());
            std::thread::spawn(move || {
                // SAFETY: `base` holds the function assembled above (Win64 ABI, no args).
                let f: extern "system" fn() -> u32 = unsafe { std::mem::transmute(base) };
                while !stop.load(Ordering::Relaxed) {
                    let r = f() as usize;
                    seen[r.min(2)].fetch_add(1, Ordering::Relaxed);
                }
            })
        };
        let site = base + 4;
        for i in 0..100 {
            let (from, to) = if i % 2 == 0 {
                (&to_a, &to_b)
            } else {
                (&to_b, &to_a)
            };
            patch_code(site, from, to).unwrap();
        }
        // Final state B; wait until the caller has observed it.
        patch_code(site, &to_a, &to_b).unwrap();
        let before = seen[2].load(Ordering::Relaxed);
        while seen[2].load(Ordering::Relaxed) == before {
            std::thread::yield_now();
        }
        stop.store(true, Ordering::Relaxed);
        caller.join().unwrap();
        assert_eq!(seen[0].load(Ordering::Relaxed), 0, "unexpected result");
        assert!(seen[1].load(Ordering::Relaxed) > 0 && seen[2].load(Ordering::Relaxed) > 0);
        let mut now = [0u8; 5];
        assert!(read_into(site, &mut now));
        assert_eq!(now.to_vec(), to_b);
    }
}
