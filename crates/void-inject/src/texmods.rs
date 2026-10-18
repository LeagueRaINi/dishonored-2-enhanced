//! Texture and resource mods for the original renderer (docs/texture-replacement.md §3).
//!
//! Detour of `idFileSystemLocal::OpenFileRead` (0x140161850, O), the single choke point
//! for image base files, streamed `_mipN` parts, decls and models. For a name with a
//! valid mod it returns an engine `idFile_Memory` holding the mod's bytes; every other
//! name goes to the original unchanged. Mods come from `void_formats::texmod` (layout,
//! validation and conversion shared with `void-tool texmod check`), so an invalid mod is
//! logged and never served.
//!
//! Installed from DllMain (`install_early`), not from `initialize`: that thread only
//! starts after the loader lock is released and first hashes the 41 MB executable,
//! while the main thread runs straight into WinMain, whose filesystem/game init opens
//! resources (callback windows, §2 of the doc). Nothing orders the two, so a later
//! install could miss game-init loads. DllMain therefore does only cheap work: a
//! directory listing of the mods dir, exact-byte checks of the four engine functions
//! used (instead of the executable hash, like the boot-video patch), and the MinHook
//! install. The archive index is opened lazily on the first request for a modded
//! name, and each mod is validated once on its first request.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Instant;

use hex_literal::hex;
use void_formats::texmod::{self, Checked, ModFile, ModSet, Originals};
use void_res::Installation;

use crate::{ffi, log, mem, settings};

const LINK_BASE: u64 = 0x1_4000_0000;

/// Engine function with the exact prologue bytes of the analysed build.
struct Site {
    va: u64,
    head: &'static [u8],
    name: &'static str,
}

/// `idFileSystemLocal::OpenFileRead(fs, name, useResourceCallback, flags)` (vtable +248).
const OPEN_FILE_READ: Site = Site {
    va: 0x1_4016_1850,
    head: &hex!("48 8B C4 57 41 56 41 57 48 83 EC 50"),
    name: "idFileSystemLocal_OpenFileRead",
};
/// Engine `operator new(size)` (`sub_1400F13F0`, TAG_NEW); the `idFile_Memory`
/// destructor frees the object with the matching `sub_1400F1520`.
const ENGINE_NEW: Site = Site {
    va: 0x1_400F_13F0,
    head: &hex!("40 53 48 83 EC 50 48 8B D9"),
    name: "operator new",
};
/// `idFile_Memory::idFile_Memory(name)` (`sub_14015AC60`), 360-byte object.
const MEMFILE_CTOR: Site = Site {
    va: 0x1_4015_AC60,
    head: &hex!("48 89 4C 24 08 57 48 83 EC 30"),
    name: "idFile_Memory_ctor",
};
/// `idFile_Memory::SetData(data, length)` (`sub_14015CA70`); clears the owns flag.
const MEMFILE_SET_DATA: Site = Site {
    va: 0x1_4015_CA70,
    head: &hex!("4C 89 81 38 01 00 00 4C 89 81 40 01 00 00"),
    name: "idFile_Memory_SetData",
};

/// `idFileSystemLocal` +0xC0: resource-open callback, set during game init and map load.
const FS_RESOURCE_CALLBACK: usize = 0xC0;
const MEMFILE_SIZE: u64 = 360;
/// `idFile_Memory` +296: free callback, called by the destructor when +352 is set.
const MEMFILE_FREE_FN: usize = 296;
/// `idFile_Memory` +352: byte "owns data".
const MEMFILE_OWNS: usize = 352;

type OpenFileRead = unsafe extern "system" fn(
    fs: *mut c_void,
    name: *const c_char,
    cb: u8,
    flags: u8,
) -> *mut c_void;
type EngineNew = unsafe extern "system" fn(size: u64) -> *mut c_void;
type MemFileCtor = unsafe extern "system" fn(this: *mut c_void, name: *const c_char) -> *mut c_void;
type MemFileSetData = unsafe extern "system" fn(this: *mut c_void, data: *mut u8, len: i64);
type ScalarDtor = unsafe extern "system" fn(this: *mut c_void, flags: u32) -> *mut c_void;

struct Entry {
    file: ModFile,
    verdict: OnceLock<Result<Checked, String>>,
    reported: AtomicBool,
}

struct State {
    set: ModSet,
    entries: HashMap<String, Entry>,
    game_dir: PathBuf,
    original: OpenFileRead,
    engine_new: EngineNew,
    ctor: MemFileCtor,
    set_data: MemFileSetData,
}

static STATE: OnceLock<State> = OnceLock::new();
/// Originals from the archives, opened on the first request for a modded name.
static ARCHIVE: OnceLock<Result<GameOriginals, String>> = OnceLock::new();
/// Buffers owned by served `idFile_Memory` objects, keyed by object address; released
/// by `free_buffer` when the engine destroys the file.
static BUFFERS: LazyLock<Mutex<HashMap<usize, Box<[u8]>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Messages before the log exists (DllMain, early engine loads): (log ready, pending).
static NOTES: Mutex<(bool, Vec<String>)> = Mutex::new((false, Vec::new()));

/// Logs now, or queues until `report` (the log is opened by `initialize`).
fn note(msg: String) {
    if let Ok(mut g) = NOTES.lock() {
        if g.0 {
            log::line(&format!("texmods: {msg}"));
        } else {
            g.1.push(msg);
        }
    }
}

/// Writes the queued messages; later ones go straight to the log. Called once the log
/// is open.
pub fn report() {
    if let Ok(mut g) = NOTES.lock() {
        for m in g.1.drain(..) {
            log::line(&format!("texmods: {m}"));
        }
        g.0 = true;
    }
}

/// Mods served / rejected so far, the time of the last change (ms since `T0`), and the
/// counts already announced (`take_summary`).
static SERVED: AtomicU32 = AtomicU32::new(0);
static REJECTED: AtomicU32 = AtomicU32::new(0);
static CHANGED_MS: AtomicU64 = AtomicU64::new(0);
static ANNOUNCED: Mutex<(u32, u32)> = Mutex::new((0, 0));
static T0: LazyLock<Instant> = LazyLock::new(Instant::now);
/// Quiet time after the last served/rejected mod before a summary is due (a load burst).
const SUMMARY_QUIET_MS: u64 = 3000;

fn count(n: &AtomicU32) {
    n.fetch_add(1, Ordering::Relaxed);
    CHANGED_MS.store(T0.elapsed().as_millis() as u64, Ordering::Relaxed);
}

/// A summary of the mods served / rejected since the last summary, once loading has been
/// quiet for a few seconds (overlay toast; polled every frame, two atomic loads when
/// nothing changed). Warn level when anything was rejected.
pub fn take_summary() -> Option<(crate::notify::Level, String)> {
    let (served, rejected) = (
        SERVED.load(Ordering::Relaxed),
        REJECTED.load(Ordering::Relaxed),
    );
    let mut a = ANNOUNCED.try_lock().ok()?;
    if (served, rejected) == *a
        || T0.elapsed().as_millis() as u64 <= CHANGED_MS.load(Ordering::Relaxed) + SUMMARY_QUIET_MS
    {
        return None;
    }
    let (ds, dr) = (served - a.0, rejected - a.1);
    *a = (served, rejected);
    let text = format!(
        "texture mods: {ds} served, {dr} rejected ({served} / {rejected} in total; Texture mods panel)"
    );
    Some(if dr > 0 {
        (crate::notify::Level::Warn, text)
    } else {
        (crate::notify::Level::Info, text)
    })
}

/// Texture-mod state for the overlay.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// The OpenFileRead hook is installed.
    pub installed: bool,
    pub dir: Option<PathBuf>,
    pub files: usize,
    /// Resource names served at least once.
    pub served: Vec<String>,
    /// Resource names rejected by validation, with the reason.
    pub rejected: Vec<(String, String)>,
}

pub fn status() -> Status {
    let Some(state) = STATE.get() else {
        return Status::default();
    };
    let mut st = Status {
        installed: true,
        dir: Some(state.set.dir.clone()),
        files: state.entries.len(),
        ..Status::default()
    };
    for (name, e) in &state.entries {
        match e.verdict.get() {
            Some(Ok(_)) if e.reported.load(Ordering::Relaxed) => st.served.push(name.clone()),
            Some(Err(why)) => st.rejected.push((name.clone(), why.clone())),
            _ => {}
        }
    }
    st.served.sort();
    st.rejected.sort();
    st
}

/// Installs the detour when `texture_mods = 1` and the mods directory holds files.
/// Called from DllMain; never panics into the loader.
pub fn install_early() {
    let r = std::panic::catch_unwind(install).unwrap_or_else(|_| Err("panic".into()));
    if let Err(e) = r {
        note(format!("not installed: {e}"));
    }
}

fn site_addr(base: u64, s: &Site) -> Result<u64, String> {
    let addr = s.va - LINK_BASE + base;
    let mut head = vec![0u8; s.head.len()];
    if !mem::read_into(addr, &mut head) || head != s.head {
        return Err(format!(
            "{} at {addr:#x}: bytes {head:02x?} differ from the analysed build",
            s.name
        ));
    }
    Ok(addr)
}

fn install() -> Result<(), String> {
    let s = settings::get();
    if !s.texture_mods {
        return Ok(());
    }
    let game_dir = ffi::exe_path()
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or("no executable path")?;
    let dir = {
        let p = Path::new(&s.mods_dir);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            game_dir.join(p)
        }
    };
    let set = ModSet::scan(&dir);
    for n in &set.notes {
        note(n.clone());
    }
    if set.files.is_empty() {
        note(format!("no mods in {}; hook not installed", dir.display()));
        return Ok(());
    }
    let base = mem::image_base().ok_or("no image base")?;
    let target = site_addr(base, &OPEN_FILE_READ)?;
    let new = site_addr(base, &ENGINE_NEW)?;
    let ctor = site_addr(base, &MEMFILE_CTOR)?;
    let set_data = site_addr(base, &MEMFILE_SET_DATA)?;
    let entries = set
        .files
        .values()
        .map(|f| {
            (
                f.resource.clone(),
                Entry {
                    file: f.clone(),
                    verdict: OnceLock::new(),
                    reported: AtomicBool::new(false),
                },
            )
        })
        .collect();
    let count = set.files.len();
    // SAFETY: `target` holds the verified prologue of OpenFileRead and `detour` has its
    // signature (x64 MSVC: rcx, rdx, r8b, r9b -> rax).
    let trampoline =
        unsafe { minhook::MinHook::create_hook(target as *mut c_void, detour as *mut c_void) }
            .map_err(|e| format!("create_hook: {e:?}"))?;
    // SAFETY: the addresses hold the verified functions with these signatures; the
    // trampoline calls the original OpenFileRead.
    let state = unsafe {
        State {
            set,
            entries,
            game_dir,
            original: std::mem::transmute::<*mut c_void, OpenFileRead>(trampoline),
            engine_new: std::mem::transmute::<u64, EngineNew>(new),
            ctor: std::mem::transmute::<u64, MemFileCtor>(ctor),
            set_data: std::mem::transmute::<u64, MemFileSetData>(set_data),
        }
    };
    if STATE.set(state).is_err() {
        return Err("installed twice".into());
    }
    // SAFETY: hook created above; the state it reads is complete.
    unsafe { minhook::MinHook::enable_hook(target as *mut c_void) }
        .map_err(|e| format!("enable_hook: {e:?}"))?;
    note(format!(
        "{count} mod files in {}; OpenFileRead hooked at {target:#x}",
        dir.display()
    ));
    Ok(())
}

unsafe extern "system" fn detour(
    fs: *mut c_void,
    name: *const c_char,
    cb: u8,
    flags: u8,
) -> *mut c_void {
    let Some(state) = STATE.get() else {
        return std::ptr::null_mut();
    };
    let original = state.original;
    if !name.is_null() {
        // SAFETY: the engine passes a NUL-terminated name.
        let raw = unsafe { CStr::from_ptr(name) };
        let key = texmod::normalize(&raw.to_string_lossy());
        if let Some(entry) = state.entries.get(&key) {
            let served = std::panic::catch_unwind(|| serve(state, entry, fs, name, cb, flags));
            if let Ok(Some(file)) = served {
                return file;
            }
        }
    }
    // SAFETY: unchanged arguments to the original.
    unsafe { original(fs, name, cb, flags) }
}

/// Returns the replacement file, or `None` to let the original run.
fn serve(
    state: &State,
    entry: &Entry,
    fs: *mut c_void,
    name: *const c_char,
    cb: u8,
    flags: u8,
) -> Option<*mut c_void> {
    let resource = &entry.file.resource;
    let verdict = entry.verdict.get_or_init(|| {
        let r = match ARCHIVE.get_or_init(|| GameOriginals::open(state)) {
            Ok(orig) => texmod::check(&state.set, resource, orig),
            Err(e) => Err(format!("archives unavailable: {e}")),
        };
        if let Err(e) = &r {
            note(format!("skipped (invalid, never served): {e}"));
            count(&REJECTED);
        }
        r
    });
    let checked = verdict.as_ref().ok()?;
    let bytes = match texmod::materialize(&entry.file, checked) {
        Ok(b) => b,
        Err(e) => {
            if !entry.reported.swap(true, Ordering::Relaxed) {
                note(format!("{resource}: cannot read mod, original used: {e}"));
                count(&REJECTED);
            }
            return None;
        }
    };
    let len = bytes.len();
    let file = new_memory_file(state, name, bytes.into_boxed_slice())?;
    // Inside a resource-callback window (game init, map load) the background loader
    // hands out queued entries in order; let it consume this one (and drop its file)
    // so its read-ahead buffer is released (doc §3.1 step 2, H). Elsewhere, and for
    // names absent from the index, skip the original read.
    // SAFETY: `fs` is the engine's idFileSystemLocal for this call.
    let callback = mem::u64_at(fs as u64 + FS_RESOURCE_CALLBACK as u64).unwrap_or(0);
    let call_through = cb != 0 && callback != 0 && checked.in_index;
    if call_through {
        // SAFETY: unchanged arguments to the original; a returned file is an engine
        // idFile whose vtable slot 0 is the scalar deleting destructor.
        unsafe {
            let f = (state.original)(fs, name, cb, flags);
            if !f.is_null() {
                let vt = *(f as *const *const usize);
                let dtor = std::mem::transmute::<usize, ScalarDtor>(*vt);
                dtor(f, 1);
            }
        }
    }
    if !entry.reported.swap(true, Ordering::Relaxed) {
        count(&SERVED);
        note(format!(
            "served {resource} ({}, {len} bytes, {}, thread {:?})",
            checked.summary,
            if call_through {
                "original consumed first"
            } else {
                "original not read"
            },
            std::thread::current().id()
        ));
    }
    Some(file)
}

/// Builds an engine `idFile_Memory` that owns `data` through `free_buffer`.
fn new_memory_file(state: &State, name: *const c_char, data: Box<[u8]>) -> Option<*mut c_void> {
    // SAFETY: verified engine functions; the object is 360 bytes from the engine heap,
    // constructed before use; the free callback and owns flag are plain fields set as
    // the engine's own loaders do (owns = 1 after SetData).
    unsafe {
        let obj = (state.engine_new)(MEMFILE_SIZE);
        if obj.is_null() {
            return None;
        }
        (state.ctor)(obj, name);
        let mut data = data;
        let ptr = data.as_mut_ptr();
        let len = data.len() as i64;
        if let Ok(mut g) = BUFFERS.lock() {
            g.insert(obj as usize, data);
        } else {
            // Keep the buffer alive (leak) rather than handing out a dangling pointer.
            std::mem::forget(data);
        }
        (state.set_data)(obj, ptr, len);
        *((obj as *mut u8).add(MEMFILE_FREE_FN) as *mut usize) = free_buffer as *const () as usize;
        *(obj as *mut u8).add(MEMFILE_OWNS) = 1;
        Some(obj)
    }
}

/// Called by the `idFile_Memory` destructor (owns flag set): releases our buffer.
unsafe extern "system" fn free_buffer(file: *mut c_void) {
    if let Ok(mut g) = BUFFERS.lock() {
        g.remove(&(file as usize));
    }
}

/// Original resources for validation: only the names the mods refer to are indexed.
struct GameOriginals {
    inst: Installation,
    images: HashMap<String, Vec<(usize, usize)>>,
    names: HashSet<String>,
}

impl GameOriginals {
    fn open(state: &State) -> Result<Self, String> {
        let inst = Installation::open(&state.game_dir).map_err(|e| e.to_string())?;
        let mut wanted: HashSet<&str> = HashSet::new();
        for k in state.entries.keys() {
            wanted.insert(k);
            if let Some((base, _)) = texmod::split_part(k) {
                wanted.insert(base);
            }
        }
        let mut images: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut names = HashSet::new();
        for (g, group) in inst.groups.iter().enumerate() {
            for (i, e) in group.index.entries.iter().enumerate() {
                if wanted.contains(e.name.as_str()) {
                    names.insert(e.name.clone());
                    if e.kind == "image" {
                        images.entry(e.name.clone()).or_default().push((g, i));
                    }
                }
            }
        }
        note(format!(
            "archive index opened ({} of {} modded names present)",
            names.len(),
            wanted.len()
        ));
        Ok(Self {
            inst,
            images,
            names,
        })
    }
}

impl Originals for GameOriginals {
    fn images(&self, name: &str) -> Result<Vec<Vec<u8>>, String> {
        let mut out: Vec<Vec<u8>> = Vec::new();
        for &(g, i) in self.images.get(name).into_iter().flatten() {
            let d = self
                .inst
                .read(g, &self.inst.groups[g].index.entries[i])
                .map_err(|e| e.to_string())?;
            if !out.contains(&d) {
                out.push(d);
            }
        }
        Ok(out)
    }

    fn exists(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}
