//! In-game notifications (shown as toasts by the overlay, also written to the log).
//!
//! Producers (benchmark, FSR, patches, texture mods, ...) call [`info`], [`warn`] or
//! [`error`] from any thread; the overlay drains the queue with [`take`] once per frame
//! and keeps each toast on screen for its duration.
//!
//! [`status`] is a live line (photo-mode progress): a note with a key replaces the toast
//! of the same key in place instead of stacking, is not logged (the producer logs its
//! milestones), and fades out when not refreshed within its duration; [`clear_status`]
//! removes it at once.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::log;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub struct Note {
    pub level: Level,
    pub text: String,
    pub at: Instant,
    /// How long the toast stays visible.
    pub ttl: Duration,
    /// Live status line: replaces the toast with the same key (empty text removes it).
    pub key: Option<&'static str>,
}

/// Unread notes; bounded so a silent overlay (disabled, or not drawing) cannot grow it.
static QUEUE: Mutex<Vec<Note>> = Mutex::new(Vec::new());
const MAX_QUEUED: usize = 64;

fn enqueue(note: Note) {
    let mut q = QUEUE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A newer status line supersedes a queued one of the same key.
    if let Some(k) = note.key {
        q.retain(|n| n.key != Some(k));
    }
    if q.len() >= MAX_QUEUED {
        q.remove(0);
    }
    q.push(note);
}

fn push(level: Level, text: String, ttl: Duration) {
    log::line(&format!("notify: {text}"));
    enqueue(Note {
        level,
        text,
        at: Instant::now(),
        ttl,
        key: None,
    });
}

pub fn info(text: impl Into<String>) {
    push(Level::Info, text.into(), Duration::from_secs(4));
}

pub fn warn(text: impl Into<String>) {
    push(Level::Warn, text.into(), Duration::from_secs(8));
}

pub fn error(text: impl Into<String>) {
    push(Level::Error, text.into(), Duration::from_secs(12));
}

/// Shows or updates the live status line `key` (info level, not logged) for `ttl`.
pub fn status(key: &'static str, text: impl Into<String>, ttl: Duration) {
    enqueue(Note {
        level: Level::Info,
        text: text.into(),
        at: Instant::now(),
        ttl,
        key: Some(key),
    });
}

/// Removes the live status line `key`.
pub fn clear_status(key: &'static str) {
    status(key, String::new(), Duration::ZERO);
}

/// Notes queued since the last call (overlay, once per frame).
pub fn take() -> Vec<Note> {
    std::mem::take(
        &mut *QUEUE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}
