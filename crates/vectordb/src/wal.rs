use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::binio::{self, crc32};
use crate::error::{Error, Result};

const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;
/// A single record must stay under this so a corrupt length cannot allocate
/// an unbounded buffer. 768-d float32 is 3 KiB; this cap is far above that.
const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum WalRecord {
    Upsert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// `fdatasync` after every append. Durable, slow.
    Always,
    /// Leave durability to the OS. Used by benchmarks and tests.
    Never,
}

pub struct Wal {
    file: File,
    sync: SyncPolicy,
    /// Byte offset of the last record that passed CRC.
    good_len: u64,
}

impl Wal {
    /// Opens the log and replays every intact record.
    ///
    /// A torn tail from a crash is truncated. The returned log is positioned
    /// for further appends.
    pub fn open(path: &Path, sync: SyncPolicy) -> Result<(Self, Vec<WalRecord>)> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let mut wal = Self {
            file,
            sync,
            good_len: 0,
        };
        let records = wal.replay()?;
        Ok((wal, records))
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let payload = encode(record);
        let crc = crc32(&payload);
        self.file.seek(SeekFrom::Start(self.good_len))?;
        self.file.set_len(self.good_len)?;
        binio::write_u32(&mut self.file, payload.len() as u32)?;
        self.file.write_all(&payload)?;
        binio::write_u32(&mut self.file, crc)?;
        self.good_len += 4 + payload.len() as u64 + 4;
        if self.sync == SyncPolicy::Always {
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Reads every intact record. A short tail is a crash remnant and is
    /// truncated. A complete record with a bad checksum is an error.
    pub fn replay(&mut self) -> Result<Vec<WalRecord>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut records = Vec::new();
        let mut good = 0u64;
        loop {
            let mut len_buf = [0u8; 4];
            match read_some(&mut self.file, &mut len_buf)? {
                ReadSome::Eof => break,
                ReadSome::Partial => {
                    self.truncate_tail(good)?;
                    break;
                }
                ReadSome::Full => {}
            }
            let len = u32::from_le_bytes(len_buf);
            if len == 0 || len > MAX_PAYLOAD {
                return Err(Error::Corrupt(format!(
                    "wal record length {len} is outside 1..={MAX_PAYLOAD}"
                )));
            }
            let mut payload = vec![0u8; len as usize];
            match read_some(&mut self.file, &mut payload)? {
                ReadSome::Full => {}
                ReadSome::Eof | ReadSome::Partial => {
                    self.truncate_tail(good)?;
                    break;
                }
            }
            let mut crc_buf = [0u8; 4];
            match read_some(&mut self.file, &mut crc_buf)? {
                ReadSome::Full => {}
                ReadSome::Eof | ReadSome::Partial => {
                    self.truncate_tail(good)?;
                    break;
                }
            }
            let crc = u32::from_le_bytes(crc_buf);
            if crc32(&payload) != crc {
                return Err(Error::Corrupt(
                    "wal checksum mismatch on a complete record".into(),
                ));
            }
            records.push(decode(&payload)?);
            good += 4 + len as u64 + 4;
        }
        self.good_len = good;
        Ok(records)
    }

    pub fn truncate(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.good_len = 0;
        if self.sync == SyncPolicy::Always {
            self.file.sync_data()?;
        }
        Ok(())
    }

    fn truncate_tail(&mut self, good: u64) -> Result<()> {
        self.file.set_len(good)?;
        self.good_len = good;
        Ok(())
    }
}

enum ReadSome {
    Eof,
    Partial,
    Full,
}

fn read_some(file: &mut File, buf: &mut [u8]) -> Result<ReadSome> {
    let mut filled = 0;
    while filled < buf.len() {
        match std::io::Read::read(file, &mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(ReadSome::Eof),
            Ok(0) => return Ok(ReadSome::Partial),
            Ok(n) => filled += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err.into()),
        }
    }
    Ok(ReadSome::Full)
}

fn encode(record: &WalRecord) -> Vec<u8> {
    let mut payload = Vec::new();
    match record {
        WalRecord::Upsert { id, vector } => {
            binio::write_u8(&mut payload, OP_UPSERT).expect("vec write");
            binio::write_u64(&mut payload, *id).expect("vec write");
            binio::write_u32(&mut payload, vector.len() as u32).expect("vec write");
            binio::write_f32s(&mut payload, vector).expect("vec write");
        }
        WalRecord::Delete { id } => {
            binio::write_u8(&mut payload, OP_DELETE).expect("vec write");
            binio::write_u64(&mut payload, *id).expect("vec write");
        }
    }
    payload
}

fn decode(payload: &[u8]) -> Result<WalRecord> {
    let mut cursor = std::io::Cursor::new(payload);
    let op = binio::read_u8(&mut cursor)?;
    let id = binio::read_u64(&mut cursor)?;
    match op {
        OP_UPSERT => {
            let dim = binio::read_u32(&mut cursor)? as usize;
            let vector = binio::read_f32s(&mut cursor, dim)?;
            Ok(WalRecord::Upsert { id, vector })
        }
        OP_DELETE => Ok(WalRecord::Delete { id }),
        other => Err(Error::Corrupt(format!("unknown wal op {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_wal(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vdb-wal-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("wal.bin")
    }

    #[test]
    fn replay_roundtrip_and_torn_tail() {
        let path = temp_wal("roundtrip");
        let (mut wal, records) = Wal::open(&path, SyncPolicy::Never).unwrap();
        assert!(records.is_empty());
        wal.append(&WalRecord::Upsert {
            id: 7,
            vector: vec![1.0, 2.0],
        })
        .unwrap();
        wal.append(&WalRecord::Delete { id: 7 }).unwrap();
        drop(wal);

        let (_wal, records) = Wal::open(&path, SyncPolicy::Never).unwrap();
        assert_eq!(records.len(), 2);

        // A partial length prefix must be discarded, not applied.
        {
            use std::fs::OpenOptions;
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&[0x20, 0x00]).unwrap();
        }
        let (_wal, records) = Wal::open(&path, SyncPolicy::Never).unwrap();
        assert_eq!(records.len(), 2);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
