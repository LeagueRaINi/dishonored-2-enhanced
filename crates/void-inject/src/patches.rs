//! Game code patches: one registry ([`PATCHES`]) of byte edits to the analysed executable,
//! each switchable with `patch.<id> = 0|1` in the settings file and in the overlay.
//!
//! A patch is applied only if every edit's original bytes are present (all or nothing,
//! through `mem::patch_code`). `Apply::Live` patches can also be reverted at run time
//! (the patched bytes are checked before the original bytes are written back); the
//! others are applied once, at DllMain (before the game's own initialisation) or at
//! `initialize` (after the executable check), and a change takes effect at the next start.
//!
//! Adding a patch = one [`Patch`] entry with its evidence; nothing else changes (the
//! settings file, the overlay panel and the log pick it up from the registry).

use std::sync::Mutex;

use hex_literal::hex;

use crate::settings::Settings;
use crate::{log, mem, notify, settings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Apply {
    /// From DllMain: must be in place before the game's own initialisation runs.
    DllMain,
    /// From `initialize`, after the executable hash check.
    #[expect(
        dead_code,
        reason = "no Init patch registered yet; remove when adding one"
    )]
    Init,
    /// Applied and reverted at run time (overlay toggle).
    Live,
}

impl Apply {
    pub fn label(self) -> &'static str {
        match self {
            Apply::DllMain => "applied at process start",
            Apply::Init => "applied at initialisation",
            Apply::Live => "live",
        }
    }
}

/// One byte edit at `rva` (image base 0x140000000 build).
pub struct Edit {
    pub rva: u64,
    pub original: &'static [u8],
    pub patched: &'static [u8],
}

pub struct Patch {
    /// Settings key suffix (`patch.<id>`).
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// Evidence (ledger rows, notes, IDB functions).
    pub source: &'static str,
    pub when: Apply,
    pub default_on: bool,
    pub edits: &'static [Edit],
}

pub const PATCHES: &[Patch] = &[
    Patch {
        id: "skip_intro",
        name: "Skip boot videos",
        description: "skips the Legal, warning, ArkBeth and Void videos: in sub_140064E00 the \
                      `jnz` after `test sil, sil` (0x140064F08) becomes an unconditional jmp \
                      (deleting the .bk2 files instead hangs the game)",
        source: "E046, d2enhanced re/NOTES.md Boot videos",
        when: Apply::DllMain,
        default_on: true,
        edits: &[Edit {
            rva: 0x6_4F08,
            original: &hex!("0F 85 BC 00 00 00"),
            patched: &hex!("E9 BD 00 00 00 90"),
        }],
    },
    Patch {
        id: "shadows_submit_early",
        name: "Submit shadow maps before the depth pre-pass",
        description: "the frame-graph builder sub_140351780 makes SHADOWS_WORLD and \
                      SHADOWS_DYNAMIC submit-depend (RenderPass_AddDependency mode 2) on \
                      DEPTHPASS, so their recorded lists wait in the serial submission until \
                      the depth pre-pass is recorded and replayed; this makes them depend on \
                      CLEAR (group slot 0) instead (`mov rdx, rdi` -> `mov rdx, r13` before \
                      both calls). Record dependencies and every consumer's dependency on the \
                      shadow passes stay. The graph is rebuilt every frame, so the switch is live. \
                      Measured -0.47 ms per CPU-bound frame, image unchanged (user check)",
        source: "E061, E062, docs/performance-baseline.md \"patch.shadows_submit_early A/B\"",
        when: Apply::Live,
        default_on: true,
        edits: &[
            Edit {
                rva: 0x35_46A0,
                original: &hex!("41 B8 02 00 00 00 48 8B D7 48 8B CE E8"),
                patched: &hex!("41 B8 02 00 00 00 49 8B D5 48 8B CE E8"),
            },
            Edit {
                rva: 0x35_46E3,
                original: &hex!("41 B8 02 00 00 00 48 8B D7 49 8D 8D E8 02 00 00"),
                patched: &hex!("41 B8 02 00 00 00 49 8B D5 49 8D 8D E8 02 00 00"),
            },
        ],
    },
];

/// Outcome per patch (index into `PATCHES`).
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Not attempted yet (its apply time has not come, or no matching executable).
    Pending,
    /// Disabled in the settings; the code is untouched.
    Off,
    Applied,
    /// The original bytes are not present (other build, or patched by someone else).
    Mismatch(String),
    Failed(String),
}

impl State {
    pub fn text(&self) -> String {
        match self {
            State::Pending => "not applied yet".into(),
            State::Off => "off".into(),
            State::Applied => "applied".into(),
            State::Mismatch(e) => format!("original bytes mismatch: {e}"),
            State::Failed(e) => format!("failed: {e}"),
        }
    }
}

static STATES: Mutex<Vec<State>> = Mutex::new(Vec::new());
/// Set once `report` ran (the log is open): later changes are logged directly.
static REPORTED: Mutex<bool> = Mutex::new(false);

fn set_state(i: usize, s: State) {
    let mut g = STATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if g.len() < PATCHES.len() {
        g.resize(PATCHES.len(), State::Pending);
    }
    let log_now = *REPORTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if log_now && g[i] != s {
        log::line(&format!("patch {}: {}", PATCHES[i].id, s.text()));
    }
    g[i] = s;
}

/// Whether `p` is switched on in `s` (default when the file does not mention it).
pub fn enabled(s: &Settings, p: &Patch) -> bool {
    s.patches.get(p.id).copied().unwrap_or(p.default_on)
}

/// Writes `to` over `from` for every edit, all or nothing.
fn write_edits(p: &Patch, to_patched: bool) -> Result<(), State> {
    let base = mem::image_base().ok_or_else(|| State::Failed("no image base".into()))?;
    let pick = |e: &Edit| {
        if to_patched {
            (e.original, e.patched)
        } else {
            (e.patched, e.original)
        }
    };
    for e in p.edits {
        let (from, _) = pick(e);
        let mut cur = vec![0u8; from.len()];
        if !mem::read_into(base + e.rva, &mut cur) {
            return Err(State::Failed(format!("{:#x} not readable", base + e.rva)));
        }
        if cur != from {
            return Err(State::Mismatch(format!(
                "{:02x?} at {:#x}",
                cur,
                0x1_4000_0000 + e.rva
            )));
        }
    }
    for (n, e) in p.edits.iter().enumerate() {
        let (from, to) = pick(e);
        if let Err(err) = mem::patch_code(base + e.rva, from, to) {
            // Undo the edits already written.
            for e in &p.edits[..n] {
                let (from, to) = pick(e);
                let _ = mem::patch_code(base + e.rva, to, from);
            }
            return Err(State::Failed(err));
        }
    }
    Ok(())
}

fn apply_where(s: &Settings, when: impl Fn(Apply) -> bool) {
    for (i, p) in PATCHES.iter().enumerate().filter(|(_, p)| when(p.when)) {
        let state = if !enabled(s, p) {
            State::Off
        } else {
            match write_edits(p, true) {
                Ok(()) => State::Applied,
                Err(e) => e,
            }
        };
        set_state(i, state);
    }
}

/// DllMain: applies the enabled `Apply::DllMain` patches (no logging; see `report`).
pub fn apply_boot(s: &Settings) {
    apply_where(s, |w| w == Apply::DllMain);
}

/// `initialize` (executable verified): applies the enabled `Init` and `Live` patches.
pub fn apply_init(s: &Settings) {
    apply_where(s, |w| matches!(w, Apply::Init | Apply::Live));
}

/// Logs every patch's state once (the log is open from `initialize` on) and posts the
/// startup notifications: one for the applied patches, one per mismatch or failure.
pub fn report() {
    let mut applied = Vec::new();
    for (p, s) in status() {
        log::line(&format!("patch {}: {}", p.id, s.text()));
        match s {
            State::Applied => applied.push(p.name),
            State::Mismatch(_) => notify::warn(format!("patch {} not applied: {}", p.id, s.text())),
            State::Failed(_) => notify::error(format!("patch {}: {}", p.id, s.text())),
            State::Pending | State::Off => {}
        }
    }
    if !applied.is_empty() {
        notify::info(format!("patches applied: {}", applied.join(", ")));
    }
    *REPORTED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
}

/// Every patch with its state.
pub fn status() -> Vec<(&'static Patch, State)> {
    let g = STATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    PATCHES
        .iter()
        .enumerate()
        .map(|(i, p)| (p, g.get(i).cloned().unwrap_or(State::Pending)))
        .collect()
}

/// Overlay toggle: records the setting (saved with the settings file) and, for a live
/// patch, applies or reverts it now. Returns the error for a failed live change.
pub fn set_enabled(id: &str, on: bool) -> Result<(), String> {
    let (i, p) = PATCHES
        .iter()
        .enumerate()
        .find(|(_, p)| p.id == id)
        .ok_or_else(|| format!("no patch {id}"))?;
    settings::edit(|s| {
        s.patches.insert(id.to_owned(), on);
    });
    if p.when != Apply::Live {
        return Ok(());
    }
    let applied = status()[i].1 == State::Applied;
    if on == applied {
        return Ok(());
    }
    match write_edits(p, on) {
        Ok(()) => {
            set_state(i, if on { State::Applied } else { State::Off });
            Ok(())
        }
        Err(e) => {
            let text = e.text();
            set_state(i, e);
            notify::error(format!("patch {id}: {text}"));
            Err(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_invariants() {
        let mut ids = std::collections::HashSet::new();
        for p in PATCHES {
            assert!(ids.insert(p.id), "duplicate id {}", p.id);
            assert!(
                !p.id.is_empty() && p.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
                "{}",
                p.id
            );
            assert!(!p.name.is_empty() && !p.description.is_empty() && !p.source.is_empty());
            assert!(!p.edits.is_empty(), "{}", p.id);
            for e in p.edits {
                assert!(!e.original.is_empty(), "{}", p.id);
                assert_eq!(e.original.len(), e.patched.len(), "{}", p.id);
                assert_ne!(e.original, e.patched, "{}", p.id);
            }
            // Edits of one patch must not overlap.
            for (a, ea) in p.edits.iter().enumerate() {
                for eb in &p.edits[a + 1..] {
                    let (a0, a1) = (ea.rva, ea.rva + ea.original.len() as u64);
                    let (b0, b1) = (eb.rva, eb.rva + eb.original.len() as u64);
                    assert!(a1 <= b0 || b1 <= a0, "{}: overlapping edits", p.id);
                }
            }
        }
    }

    #[test]
    fn defaults_and_settings() {
        let mut s = Settings::default();
        let p = &PATCHES[0];
        assert_eq!(enabled(&s, p), p.default_on);
        s.patches.insert(p.id.to_owned(), !p.default_on);
        assert_eq!(enabled(&s, p), !p.default_on);
        s.patches.clear();
        assert_eq!(enabled(&s, p), p.default_on);
    }
}
