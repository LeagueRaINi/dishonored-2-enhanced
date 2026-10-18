use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    BadMagic {
        expected: [u8; 4],
        found: [u8; 4],
    },
    UnsupportedVersion(u32),
    Truncated {
        field: &'static str,
        offset: usize,
        needed: u64,
        available: usize,
    },
    InvalidUtf8 {
        field: &'static str,
        offset: usize,
    },
    /// A value the original reader rejects, or that breaks a structural invariant.
    Invalid {
        field: &'static str,
        detail: String,
    },
    TrailingData {
        offset: usize,
        len: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic { expected, found } => {
                write!(f, "bad magic: expected {expected:02x?}, found {found:02x?}")
            }
            Error::UnsupportedVersion(v) => write!(f, "unsupported version {v:#x}"),
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
            Error::Invalid { field, detail } => write!(f, "invalid {field}: {detail}"),
            Error::TrailingData { offset, len } => {
                write!(f, "{len} unparsed trailing bytes at offset {offset:#x}")
            }
        }
    }
}

impl std::error::Error for Error {}
