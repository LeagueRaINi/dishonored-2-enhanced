//! Minimal append-only file log (no allocation-free guarantees; not for use in
//! DllMain) plus an in-memory ring of the last lines (shown by the overlay, also filled
//! when no log file exists). Failures to log are ignored: logging must never affect the
//! game.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static LOG: Mutex<Option<File>> = Mutex::new(None);
/// Last `RING_LINES` lines.
static RING: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
const RING_LINES: usize = 300;

pub fn init(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("void-inject.log"))
        .ok();
    if let Ok(mut g) = LOG.lock() {
        *g = file;
    }
}

pub fn line(msg: &str) {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    if let Ok(mut r) = RING.lock() {
        if r.len() >= RING_LINES {
            r.pop_front();
        }
        r.push_back(format!("{t:.3} {msg}"));
    }
    if let Ok(mut g) = LOG.lock()
        && let Some(f) = g.as_mut()
    {
        let _ = writeln!(f, "{t:.3} {msg}");
        let _ = f.flush();
    } else {
        // No log file (offline tools): stderr.
        eprintln!("{msg}");
    }
}

/// The last lines (oldest first).
pub fn recent() -> Vec<String> {
    RING.lock()
        .map(|r| r.iter().cloned().collect())
        .unwrap_or_default()
}
