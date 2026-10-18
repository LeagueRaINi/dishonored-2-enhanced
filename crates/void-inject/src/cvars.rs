//! The engine's console variables: dump (written with every benchmark run), cached-value
//! writes, and settings-driven overrides (`cvar.<name> = <value>` in `void-inject.ini`).
//!
//! O (IDB, exe `b671f824...6180`): `idCVar` constructors (`sub_140101640`) push each static
//! cvar onto a singly linked list headed at `0x1431C04B0`, `next` at +120. Fields: value
//! `idStr` (data pointer at +0, length in the low 29 bits of +8), cached int +40, float +44,
//! name +48, reset (default) string +56, description +64, flags +72. Corroborated by
//! d2enhanced `src/console.cpp` (value +0x00, name +0x30, reset +0x38, description +0x40),
//! which reads the same objects through the `idCVarSystem` interface. Cvars created at
//! runtime (not static objects) are not on this list.
//!
//! Modified flag (O): flag bit 0x40000 marks a changed cvar; per-frame consumers poll and
//! clear it, e.g. `sub_1402EB9F0` (called every frame from `sub_1403619D0`) re-reads the
//! texture-filter cvars and `image_anisotropy` (float +44 -> image manager +100,
//! `sub_1402EB830`) only when one of them carries the bit. Overrides therefore write the
//! cached int and float and set the bit; the string value stays as loaded (console
//! display only).

use std::sync::Mutex;

use crate::{log, mem};

/// RVA of the static cvar list head (`qword_1431C04B0`).
const LIST_HEAD_RVA: u64 = 0x31C_04B0;
const NEXT: u64 = 120;
const INT: u64 = 40;
const FLOAT: u64 = 44;
const FLAGS: u64 = 72;
const MODIFIED: u32 = 0x4_0000;
const MAX_CVARS: usize = 20_000;
/// Overrides are re-checked every this many frames (menus or config loads may write the
/// cvar again).
const ENFORCE_EVERY: u64 = 30;

fn string_at(ptr_field: u64) -> String {
    let Some(p) = mem::u64_at(ptr_field) else {
        return String::new();
    };
    if p == 0 {
        return String::new();
    }
    let mut out = Vec::new();
    let mut chunk = [0u8; 64];
    let mut addr = p;
    while out.len() < 4096 {
        if !mem::read_into(addr, &mut chunk) {
            // Near the end of a page: fall back to single bytes.
            let mut b = [0u8; 1];
            if !mem::read_into(addr, &mut b) || b[0] == 0 {
                break;
            }
            out.push(b[0]);
            addr += 1;
            continue;
        }
        match chunk.iter().position(|&c| c == 0) {
            Some(n) => {
                out.extend_from_slice(&chunk[..n]);
                break;
            }
            None => out.extend_from_slice(&chunk),
        }
        addr += chunk.len() as u64;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// TSV: name, value, int, float, default, flags (hex), description; sorted by name.
pub fn dump_tsv() -> Result<(String, usize), String> {
    let base = mem::image_base().ok_or("no image base")?;
    let mut node = mem::u64_at(base + LIST_HEAD_RVA).ok_or("cvar list head unreadable")?;
    let mut rows = Vec::new();
    while node != 0 && rows.len() < MAX_CVARS {
        let name = string_at(node + 48);
        if !name.is_empty() {
            let int = mem::u32_at(node + 40).unwrap_or(0) as i32;
            let float = mem::f32s::<1>(node + 44).map_or(f32::NAN, |v| v[0]);
            let flags = mem::u32_at(node + 72).unwrap_or(0);
            rows.push(format!(
                "{}\t{}\t{int}\t{float}\t{}\t{flags:#x}\t{}",
                clean(&name),
                clean(&string_at(node)),
                clean(&string_at(node + 56)),
                clean(&string_at(node + 64)),
            ));
        }
        node = mem::u64_at(node + NEXT).unwrap_or(0);
    }
    rows.sort();
    let n = rows.len();
    Ok((
        format!(
            "name\tvalue\tint\tfloat\tdefault\tflags\tdescription\n{}\n",
            rows.join("\n")
        ),
        n,
    ))
}

/// Address of the static cvar object called `name` (cached after the first walk).
fn find(name: &str) -> Option<u64> {
    use std::collections::HashMap;
    static CACHE: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);
    let mut cache = CACHE.lock().ok()?;
    let map = cache.get_or_insert_with(HashMap::new);
    if let Some(n) = map.get(name) {
        return Some(*n);
    }
    let base = mem::image_base()?;
    let mut node = mem::u64_at(base + LIST_HEAD_RVA)?;
    let mut n = 0;
    while node != 0 && n < MAX_CVARS && string_at(node + 48) != name {
        node = mem::u64_at(node + NEXT).unwrap_or(0);
        n += 1;
    }
    if node == 0 || n >= MAX_CVARS {
        return None;
    }
    map.insert(name.to_owned(), node);
    Some(node)
}

/// Sets a static cvar's cached int (+40) and float (+44) values; returns the previous
/// int. The string value (+0) is left as is (console display only). O: the renderer
/// reads the int, e.g. `idRenderView_Setup` `cmp [r_skipStatic + 0x28], 0` at
/// 0x140374F1C (cvar object 0x14397A9D0 from its constructor 0x141378750).
pub fn set_int(name: &str, value: i32) -> Option<i32> {
    let node = find(name)?;
    let old = mem::u32_at(node + INT)? as i32;
    (mem::write_u32(node + INT, value as u32)
        && mem::write_u32(node + FLOAT, (value as f32).to_bits()))
    .then_some(old)
}

struct Override {
    name: String,
    value: f32,
    /// Times the value was (re)written; the first write and the first few re-writes
    /// are logged.
    writes: u32,
    /// Not a static cvar (logged once, then skipped).
    missing: bool,
}

static OVERRIDES: Mutex<Vec<Override>> = Mutex::new(Vec::new());

/// Registers the settings' cvar overrides (enforced by `enforce_overrides`); also called
/// by the overlay on every edit. Entries whose name and value are unchanged keep their
/// write count and missing flag. A removed override leaves the cvar at its last value
/// until the game or the console sets it again.
pub fn set_overrides(list: &[(String, f32)]) {
    let mut o = OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let new = list
        .iter()
        .map(|(name, value)| {
            let old = o
                .iter()
                .find(|x| x.name == *name && x.value.to_bits() == value.to_bits());
            Override {
                name: name.clone(),
                value: *value,
                writes: old.map_or(0, |x| x.writes),
                missing: old.is_some_and(|x| x.missing),
            }
        })
        .collect();
    *o = new;
}

/// Main thread, every frame: every `ENFORCE_EVERY` frames, writes each override whose
/// cached int/float differ from it (int = value truncated, as the engine caches it) and
/// sets the modified flag so per-frame consumers re-read it.
pub fn enforce_overrides(frame: u64) {
    if !frame.is_multiple_of(ENFORCE_EVERY) {
        return;
    }
    let mut list = OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for o in list.iter_mut().filter(|o| !o.missing) {
        let Some(node) = find(&o.name) else {
            // Static cvars exist from process start: a miss is permanent (the full list
            // walk is not repeated).
            o.missing = true;
            log::line(&format!("cvar override {}: no such static cvar", o.name));
            continue;
        };
        let (Some(int), Some(float)) = (mem::u32_at(node + INT), mem::f32s::<1>(node + FLOAT))
        else {
            continue;
        };
        let want_int = o.value as i32;
        if int as i32 == want_int && float[0].to_bits() == o.value.to_bits() {
            continue;
        }
        let flags = mem::u32_at(node + FLAGS).unwrap_or(0);
        let ok = mem::write_u32(node + INT, want_int as u32)
            && mem::write_u32(node + FLOAT, o.value.to_bits())
            && mem::write_u32(node + FLAGS, flags | MODIFIED);
        o.writes += 1;
        if o.writes <= 3 {
            log::line(&format!(
                "cvar override {} = {} (was int {} float {}; frame {frame}; write {}{})",
                o.name,
                o.value,
                int as i32,
                float[0],
                o.writes,
                if ok { "" } else { "; FAILED" }
            ));
        }
    }
}

/// A static cvar's current state (overlay lookup).
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub name: String,
    /// String value (as loaded / last set through the console).
    pub value: String,
    pub int: i32,
    pub float: f32,
    pub default: String,
    pub flags: u32,
    pub description: String,
}

/// Looks up the static cvar `name` (one list walk on the first lookup of a name).
pub fn info(name: &str) -> Option<Info> {
    let node = find(name)?;
    Some(Info {
        name: string_at(node + 48),
        value: string_at(node),
        int: mem::u32_at(node + INT)? as i32,
        float: mem::f32s::<1>(node + FLOAT)?[0],
        default: string_at(node + 56),
        flags: mem::u32_at(node + FLAGS).unwrap_or(0),
        description: string_at(node + 64),
    })
}

/// State of one registered override: name, value, writes so far, not a static cvar.
pub fn overrides() -> Vec<(String, f32, u32, bool)> {
    OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|o| (o.name.clone(), o.value, o.writes, o.missing))
        .collect()
}
