//! Bounds-checked cursor over an in-memory byte slice.
//!
//! Every read either succeeds completely or returns [`Error::Truncated`] naming
//! the field and offset, so malformed input never panics or over-allocates.

use crate::Error;

pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn bytes(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], Error> {
        if len > self.remaining() {
            return Err(Error::Truncated {
                field,
                offset: self.pos as u64,
                needed: len as u64,
                available: self.remaining() as u64,
            });
        }
        let out = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }

    pub fn array<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N], Error> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.bytes(N, field)?);
        Ok(out)
    }

    pub fn u16_be(&mut self, field: &'static str) -> Result<u16, Error> {
        self.array(field).map(u16::from_be_bytes)
    }

    pub fn u32_be(&mut self, field: &'static str) -> Result<u32, Error> {
        self.array(field).map(u32::from_be_bytes)
    }

    pub fn u32_le(&mut self, field: &'static str) -> Result<u32, Error> {
        self.array(field).map(u32::from_le_bytes)
    }

    pub fn u64_be(&mut self, field: &'static str) -> Result<u64, Error> {
        self.array(field).map(u64::from_be_bytes)
    }

    /// Little-endian u32 length followed by that many bytes of UTF-8.
    pub fn string_le32(&mut self, field: &'static str) -> Result<String, Error> {
        let start = self.pos as u64;
        let len = self.u32_le(field)? as usize;
        let bytes = self.bytes(len, field)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::InvalidUtf8 {
            field,
            offset: start,
        })
    }
}
