use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::arena::Arena;
use crate::binio;
use crate::error::{Error, Result};
use crate::hnsw::HnswParams;
use crate::metric::Metric;

const META_MAGIC: u32 = 0x5644_424D;
const SEGMENT_MAGIC: u32 = 0x5644_4253;
const VERSION: u32 = 1;

const KIND_FLAT: u8 = 1;
const KIND_HNSW: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexParams {
    Flat,
    Hnsw(HnswParams),
}

impl IndexParams {
    pub fn name(&self) -> &'static str {
        match self {
            IndexParams::Flat => "flat",
            IndexParams::Hnsw(_) => "hnsw",
        }
    }

    fn kind_code(&self) -> u8 {
        match self {
            IndexParams::Flat => KIND_FLAT,
            IndexParams::Hnsw(_) => KIND_HNSW,
        }
    }

    fn hnsw_fields(&self) -> (u32, u32, u32, u64) {
        match self {
            IndexParams::Flat => (0, 0, 0, 0),
            IndexParams::Hnsw(p) => (
                p.m as u32,
                p.ef_construction as u32,
                p.ef_search as u32,
                p.seed,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredConfig {
    pub dim: usize,
    pub metric: Metric,
    pub index: IndexParams,
}

pub fn meta_path(dir: &Path) -> std::path::PathBuf {
    dir.join("meta.bin")
}

pub fn segment_path(dir: &Path) -> std::path::PathBuf {
    dir.join("segment.bin")
}

pub fn wal_path(dir: &Path) -> std::path::PathBuf {
    dir.join("wal.bin")
}

pub fn write_meta(dir: &Path, config: &StoredConfig) -> Result<()> {
    fs::create_dir_all(dir)?;
    let tmp = dir.join("meta.bin.tmp");
    {
        let file = File::create(&tmp)?;
        let mut w = BufWriter::new(file);
        binio::write_u32(&mut w, META_MAGIC)?;
        binio::write_u32(&mut w, VERSION)?;
        binio::write_u32(&mut w, config.dim as u32)?;
        binio::write_u8(&mut w, config.metric.code())?;
        binio::write_u8(&mut w, config.index.kind_code())?;
        let (m, ef_c, ef_s, seed) = config.index.hnsw_fields();
        binio::write_u32(&mut w, m)?;
        binio::write_u32(&mut w, ef_c)?;
        binio::write_u32(&mut w, ef_s)?;
        binio::write_u64(&mut w, seed)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(tmp, meta_path(dir))?;
    Ok(())
}

pub fn read_meta(dir: &Path) -> Result<Option<StoredConfig>> {
    let path = meta_path(dir);
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let magic = binio::read_u32(&mut r)?;
    if magic != META_MAGIC {
        return Err(Error::Corrupt("meta.bin magic mismatch".into()));
    }
    let version = binio::read_u32(&mut r)?;
    if version != VERSION {
        return Err(Error::Corrupt(format!(
            "unsupported meta version {version}"
        )));
    }
    let dim = binio::read_u32(&mut r)? as usize;
    let metric = Metric::from_code(binio::read_u8(&mut r)?)
        .ok_or_else(|| Error::Corrupt("unknown metric code".into()))?;
    let kind = binio::read_u8(&mut r)?;
    let m = binio::read_u32(&mut r)? as usize;
    let ef_construction = binio::read_u32(&mut r)? as usize;
    let ef_search = binio::read_u32(&mut r)? as usize;
    let seed = binio::read_u64(&mut r)?;
    let index = match kind {
        KIND_FLAT => IndexParams::Flat,
        KIND_HNSW => IndexParams::Hnsw(HnswParams {
            m,
            ef_construction,
            ef_search,
            seed,
        }),
        other => return Err(Error::Corrupt(format!("unknown index kind {other}"))),
    };
    Ok(Some(StoredConfig { dim, metric, index }))
}

pub fn write_segment(dir: &Path, arena: &Arena) -> Result<()> {
    let tmp = dir.join("segment.bin.tmp");
    {
        let file = File::create(&tmp)?;
        let mut w = BufWriter::new(file);
        binio::write_u32(&mut w, SEGMENT_MAGIC)?;
        binio::write_u32(&mut w, VERSION)?;
        binio::write_u32(&mut w, arena.dim() as u32)?;
        binio::write_u64(&mut w, arena.alive_count() as u64)?;
        for slot in 0..arena.slot_count() {
            let slot = slot as u32;
            if !arena.is_alive(slot) {
                continue;
            }
            binio::write_u64(&mut w, arena.id(slot))?;
            binio::write_f32s(&mut w, arena.vector(slot))?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(tmp, segment_path(dir))?;
    Ok(())
}

pub fn read_segment(dir: &Path, dim: usize) -> Result<Arena> {
    let path = segment_path(dir);
    let mut arena = Arena::new(dim);
    if !path.exists() {
        return Ok(arena);
    }
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let magic = binio::read_u32(&mut r)?;
    if magic != SEGMENT_MAGIC {
        return Err(Error::Corrupt("segment.bin magic mismatch".into()));
    }
    let version = binio::read_u32(&mut r)?;
    if version != VERSION {
        return Err(Error::Corrupt(format!(
            "unsupported segment version {version}"
        )));
    }
    let stored_dim = binio::read_u32(&mut r)? as usize;
    if stored_dim != dim {
        return Err(Error::Corrupt(format!(
            "segment dim {stored_dim} does not match meta dim {dim}"
        )));
    }
    let count = binio::read_u64(&mut r)? as usize;
    for _ in 0..count {
        let id = binio::read_u64(&mut r)?;
        let vector = binio::read_f32s(&mut r, dim)?;
        arena.append(id, &vector);
    }
    Ok(arena)
}
