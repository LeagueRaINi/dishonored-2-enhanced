use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    BadMagic {
        expected: [u8; 4],
        found: [u8; 4],
    },
    Truncated {
        field: &'static str,
        offset: u64,
        needed: u64,
        available: u64,
    },
    InvalidUtf8 {
        field: &'static str,
        offset: u64,
    },
    SizeMismatch {
        field: &'static str,
        declared: u64,
        actual: u64,
    },
    TrailingData {
        offset: u64,
        len: u64,
    },
    /// The entry's location word refers to a group/slot the master index
    /// does not define, or uses bits whose meaning is unknown.
    UnresolvedLocation {
        entry: String,
        location: u16,
        reason: &'static str,
    },
    EntryOutOfRange {
        entry: String,
        offset: u64,
        packed_size: u64,
        file_len: u64,
    },
    Decompress {
        entry: String,
        detail: String,
    },
    UnsafePath {
        name: String,
        reason: &'static str,
    },
    UnknownGroup(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::BadMagic { expected, found } => {
                write!(f, "bad magic: expected {expected:02x?}, found {found:02x?}")
            }
            Error::Truncated {
                field,
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated reading {field} at offset {offset:#x}: need {needed} bytes, {available} available"
            ),
            Error::InvalidUtf8 { field, offset } => {
                write!(f, "{field} at offset {offset:#x} is not valid UTF-8")
            }
            Error::SizeMismatch {
                field,
                declared,
                actual,
            } => write!(f, "{field}: declared {declared}, actual {actual}"),
            Error::TrailingData { offset, len } => {
                write!(f, "{len} unparsed trailing bytes at offset {offset:#x}")
            }
            Error::UnresolvedLocation {
                entry,
                location,
                reason,
            } => write!(f, "{entry}: location {location:#06x}: {reason}"),
            Error::EntryOutOfRange {
                entry,
                offset,
                packed_size,
                file_len,
            } => write!(
                f,
                "{entry}: range {offset:#x}+{packed_size:#x} exceeds file length {file_len:#x}"
            ),
            Error::Decompress { entry, detail } => write!(f, "{entry}: zlib: {detail}"),
            Error::UnsafePath { name, reason } => {
                write!(f, "unsafe output path {name:?}: {reason}")
            }
            Error::UnknownGroup(g) => write!(f, "unknown group {g}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Error {
    let path = path.into();
    move |source| Error::Io { path, source }
}
