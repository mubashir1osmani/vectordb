use std::io::{Read, Write};

use crate::error::{Error, Result};

pub fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(Error::Corrupt(
                    "truncated record while reading storage".into(),
                ))
            }
            Ok(n) => filled += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(true)
}

pub fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    if !read_exact(r, &mut buf)? {
        return Err(Error::Corrupt("unexpected end of file".into()));
    }
    Ok(buf[0])
}

pub fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    if !read_exact(r, &mut buf)? {
        return Err(Error::Corrupt("unexpected end of file".into()));
    }
    Ok(u32::from_le_bytes(buf))
}

pub fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    if !read_exact(r, &mut buf)? {
        return Err(Error::Corrupt("unexpected end of file".into()));
    }
    Ok(u64::from_le_bytes(buf))
}

pub fn read_f32s<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    if n > 0 && !read_exact(r, &mut bytes)? {
        return Err(Error::Corrupt("unexpected end of file".into()));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn write_u8<W: Write>(w: &mut W, v: u8) -> Result<()> {
    w.write_all(&[v])?;
    Ok(())
}

pub fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

pub fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

pub fn write_f32s<W: Write>(w: &mut W, values: &[f32]) -> Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// IEEE CRC-32. Used to tell a torn tail from a finished record.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // ISO-HDLC check value for "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
