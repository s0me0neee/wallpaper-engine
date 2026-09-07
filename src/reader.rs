//! Little-endian primitives shared by the `.pkg` and `.tex` parsers.

use anyhow::{Result, bail};
use std::io::{Read, Seek, SeekFrom};

/// A counting reader: every read advances `pos`, which both formats need in
/// order to locate data blobs and to verify they consumed a file exactly.
pub struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    pub fn pos(&self) -> u64 {
        self.pos
    }

    pub fn into_inner(self) -> R {
        self.inner
    }

    pub fn i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        self.pos += 4;
        Ok(i32::from_le_bytes(buf))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn bytes(&mut self, count: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; count];
        self.inner.read_exact(&mut buf)?;
        self.pos += count as u64;
        Ok(buf)
    }

    /// A NUL-terminated magic string, as used by every `.tex` chunk header.
    pub fn magic(&mut self) -> Result<String> {
        let mut buf = Vec::new();
        loop {
            let mut byte = [0u8; 1];
            self.inner.read_exact(&mut byte)?;
            self.pos += 1;
            if byte[0] == 0 {
                break;
            }
            buf.push(byte[0]);
            if buf.len() > 64 {
                bail!("unterminated magic string");
            }
        }
        Ok(String::from_utf8(buf)?)
    }

    /// An `int32` length followed by that many UTF-8 bytes, as used by `.pkg`.
    pub fn string(&mut self) -> Result<String> {
        let length = self.i32()?;
        if !(0..=4096).contains(&length) {
            bail!("implausible string length {length} at offset {}", self.pos - 4);
        }
        Ok(String::from_utf8(self.bytes(length as usize)?)?)
    }
}

impl<R: Read + Seek> Reader<R> {
    pub fn seek_to(&mut self, pos: u64) -> Result<()> {
        self.inner.seek(SeekFrom::Start(pos))?;
        self.pos = pos;
        Ok(())
    }
}
