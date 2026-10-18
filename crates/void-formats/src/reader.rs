//! Bounds-checked little-endian cursor.

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
                offset: self.pos,
                needed: len as u64,
                available: self.remaining(),
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

    pub fn u8(&mut self, field: &'static str) -> Result<u8, Error> {
        Ok(self.array::<1>(field)?[0])
    }

    pub fn u16(&mut self, field: &'static str) -> Result<u16, Error> {
        self.array(field).map(u16::from_le_bytes)
    }

    pub fn u32(&mut self, field: &'static str) -> Result<u32, Error> {
        self.array(field).map(u32::from_le_bytes)
    }

    pub fn f32(&mut self, field: &'static str) -> Result<f32, Error> {
        self.array(field).map(f32::from_le_bytes)
    }

    /// u32 byte length followed by that many bytes.
    pub fn sized_blob(&mut self, field: &'static str) -> Result<&'a [u8], Error> {
        let len = self.u32(field)? as usize;
        self.bytes(len, field)
    }

    /// u32 length followed by UTF-8 bytes.
    pub fn string(&mut self, field: &'static str) -> Result<String, Error> {
        let offset = self.pos;
        let bytes = self.sized_blob(field)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::InvalidUtf8 { field, offset })
    }

    /// Reads an element count and rejects counts that cannot fit in the
    /// remaining input at `min_size` bytes per element.
    pub fn count(&mut self, min_size: usize, field: &'static str) -> Result<usize, Error> {
        let offset = self.pos;
        let n = self.u32(field)? as usize;
        let needed = n as u64 * min_size as u64;
        if needed > self.remaining() as u64 {
            return Err(Error::Truncated {
                field,
                offset,
                needed,
                available: self.remaining(),
            });
        }
        Ok(n)
    }

    pub fn u32_vec(&mut self, field: &'static str) -> Result<Vec<u32>, Error> {
        let n = self.count(4, field)?;
        (0..n).map(|_| self.u32(field)).collect()
    }
}
