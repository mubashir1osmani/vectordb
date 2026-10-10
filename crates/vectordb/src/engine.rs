use std::path::PathBuf;

use crate::arena::Arena;
use crate::distance::prepare_vector;
use crate::error::{Error, Result};
use crate::flat::FlatIndex;
use crate::hnsw::{HnswIndex, HnswParams};
use crate::metric::Metric;
use crate::segment::{self, IndexParams, StoredConfig};
use crate::wal::{SyncPolicy, Wal, WalRecord};

/// One neighbor from [`Engine::search`].
///
/// `distance` is smaller-is-nearer for every metric. L2 reports squared
/// Euclidean distance. Cosine and inner product report the negative dot
/// product of the (normalized, for cosine) vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub distance: f32,
}

/// How to build the in-memory index when the engine opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub dim: usize,
    pub metric: Metric,
    pub index: IndexParams,
    pub sync: SyncPolicy,
}

impl EngineConfig {
    pub fn flat(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            index: IndexParams::Flat,
            sync: SyncPolicy::Always,
        }
    }

    pub fn hnsw(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self {
            dim,
            metric,
            index: IndexParams::Hnsw(params),
            sync: SyncPolicy::Always,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(Error::EmptyVector);
        }
        if let IndexParams::Hnsw(params) = self.index {
            params.validate().map_err(Error::ConfigMismatch)?;
        }
        Ok(())
    }
}

enum SearchIndex {
    Flat(FlatIndex),
    Hnsw(Box<HnswIndex>),
}

enum Backend {
    Memory,
    Disk { path: PathBuf, wal: Wal },
}

/// Single-node vector engine.
///
/// Vectors are the source of truth. The HNSW graph is rebuilt from them on
/// open and after [`Engine::flush`] compacts deletes. The graph is not
/// persisted: at a billion vectors that graph lives on SSD, which this
/// process does not do yet.
///
/// The engine is single-threaded. Search borrows interior scratch inside HNSW.
pub struct Engine {
    dim: usize,
    metric: Metric,
    index_params: IndexParams,
    arena: Arena,
    index: SearchIndex,
    backend: Backend,
}

impl Engine {
    /// In-memory engine. [`Engine::flush`] compacts deletes and rebuilds the
    /// graph, and writes nothing.
    pub fn open_memory(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let mut engine = Self {
            dim: config.dim,
            metric: config.metric,
            index_params: config.index,
            arena: Arena::new(config.dim),
            index: SearchIndex::Flat(FlatIndex::new(config.metric)),
            backend: Backend::Memory,
        };
        engine.rebuild_index();
        Ok(engine)
    }

    /// Opens a directory, creating it when `meta.bin` is absent.
    ///
    /// Dim and metric must match an existing database. Index parameters come
    /// from `config` and the graph is rebuilt to match them.
    pub fn open(path: impl Into<PathBuf>, config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let path = path.into();
        std::fs::create_dir_all(&path)?;
        if let Some(stored) = segment::read_meta(&path)? {
            if stored.dim != config.dim {
                return Err(Error::ConfigMismatch(format!(
                    "stored dim {} != {}",
                    stored.dim, config.dim
                )));
            }
            if stored.metric != config.metric {
                return Err(Error::ConfigMismatch(format!(
                    "stored metric {} != {}",
                    stored.metric.name(),
                    config.metric.name()
                )));
            }
        }
        let stored = StoredConfig {
            dim: config.dim,
            metric: config.metric,
            index: config.index.clone(),
        };
        segment::write_meta(&path, &stored)?;

        let mut arena = segment::read_segment(&path, config.dim)?;
        let (mut wal, records) = Wal::open(&segment::wal_path(&path), config.sync)?;
        for record in records {
            apply_record(&mut arena, config.dim, record)?;
        }
        // Drop a torn tail's extra bytes by leaving the wal where replay stopped.
        let _ = &mut wal;

        let mut engine = Self {
            dim: config.dim,
            metric: config.metric,
            index_params: config.index,
            arena,
            index: SearchIndex::Flat(FlatIndex::new(config.metric)),
            backend: Backend::Disk { path, wal },
        };
        engine.rebuild_index();
        Ok(engine)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.arena.alive_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn index_name(&self) -> &'static str {
        self.index_params.name()
    }

    /// Raises or lowers the HNSW search beam. Ignored by the flat index.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        let ef_search = ef_search.max(1);
        if let IndexParams::Hnsw(params) = &mut self.index_params {
            params.ef_search = ef_search;
        }
        if let SearchIndex::Hnsw(index) = &mut self.index {
            index.set_ef_search(ef_search);
        }
    }

    /// Inserts a vector or replaces the vector for an existing id.
    ///
    /// Replacing a vector under HNSW updates the stored bytes immediately.
    /// Edges are left in place until the next [`Engine::flush`], which
    /// rebuilds the graph. New ids are linked incrementally.
    pub fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        self.check_vector(vector)?;
        let stored = prepare_vector(self.metric, vector);
        self.apply_upsert(id, &stored)?;
        self.log(WalRecord::Upsert { id, vector: stored })?;
        Ok(())
    }

    pub fn delete(&mut self, id: u64) -> Result<()> {
        if self.arena.kill(id).is_none() {
            return Err(Error::NotFound(id));
        }
        self.log(WalRecord::Delete { id })?;
        Ok(())
    }

    /// Top-k nearest alive vectors. Results are sorted nearest-first.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.check_vector(query)?;
        let prepared = prepare_vector(self.metric, query);
        let hits = match &self.index {
            SearchIndex::Flat(index) => index.search(&self.arena, &prepared, k),
            SearchIndex::Hnsw(index) => index.search(&self.arena, &prepared, k),
        };
        Ok(hits
            .into_iter()
            .map(|hit| Hit {
                id: self.arena.id(hit.slot),
                distance: hit.distance,
            })
            .collect())
    }

    /// Compacts dead slots, rebuilds the graph, and checkpoints the directory.
    ///
    /// On disk the checkpoint is a segment file plus an empty WAL. A crash
    /// before the WAL is truncated replays the same idempotent records.
    pub fn flush(&mut self) -> Result<()> {
        self.arena.compact();
        self.rebuild_index();
        if let Backend::Disk { path, wal } = &mut self.backend {
            let stored = StoredConfig {
                dim: self.dim,
                metric: self.metric,
                index: self.index_params.clone(),
            };
            // Segment first, then drop the log. Replaying a checkpointed log
            // on top of the new segment is idempotent.
            segment::write_segment(path, &self.arena)?;
            segment::write_meta(path, &stored)?;
            wal.truncate()?;
        }
        Ok(())
    }

    fn check_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if vector.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(())
    }

    fn apply_upsert(&mut self, id: u64, stored: &[f32]) -> Result<()> {
        if let Some(slot) = self.arena.slot_of(id) {
            self.arena.set(slot, stored);
            return Ok(());
        }
        let slot = self.arena.append(id, stored);
        match &mut self.index {
            SearchIndex::Flat(_) => {}
            SearchIndex::Hnsw(index) => index.insert(&self.arena, slot),
        }
        Ok(())
    }

    fn log(&mut self, record: WalRecord) -> Result<()> {
        if let Backend::Disk { wal, .. } = &mut self.backend {
            wal.append(&record)?;
        }
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.index = match &self.index_params {
            IndexParams::Flat => SearchIndex::Flat(FlatIndex::new(self.metric)),
            IndexParams::Hnsw(params) => {
                SearchIndex::Hnsw(Box::new(HnswIndex::new(self.metric, *params)))
            }
        };
        let slots = self.arena.slot_count();
        for slot in 0..slots {
            let slot = slot as u32;
            if !self.arena.is_alive(slot) {
                continue;
            }
            if let SearchIndex::Hnsw(index) = &mut self.index {
                index.insert(&self.arena, slot);
            }
        }
    }
}

fn apply_record(arena: &mut Arena, dim: usize, record: WalRecord) -> Result<()> {
    match record {
        WalRecord::Upsert { id, vector } => {
            if vector.len() != dim {
                return Err(Error::Corrupt(format!(
                    "wal vector dim {} != database dim {dim}",
                    vector.len()
                )));
            }
            if let Some(slot) = arena.slot_of(id) {
                arena.set(slot, &vector);
            } else {
                arena.append(id, &vector);
            }
        }
        WalRecord::Delete { id } => {
            arena.kill(id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::recall_at_k;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(name: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("vdb-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn flat_search_orders_l2() {
        let mut engine = Engine::open_memory(EngineConfig::flat(2, Metric::L2)).unwrap();
        engine.upsert(1, &[0.0, 0.0]).unwrap();
        engine.upsert(2, &[5.0, 0.0]).unwrap();
        engine.upsert(3, &[1.0, 0.0]).unwrap();
        let hits = engine.search(&[0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 3);
    }

    #[test]
    fn cosine_finds_same_direction() {
        let mut engine = Engine::open_memory(EngineConfig::flat(2, Metric::Cosine)).unwrap();
        engine.upsert(1, &[1.0, 0.0]).unwrap();
        engine.upsert(2, &[0.0, 2.0]).unwrap();
        let hits = engine.search(&[5.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn delete_hides_a_vector() {
        let mut engine = Engine::open_memory(EngineConfig::flat(2, Metric::L2)).unwrap();
        engine.upsert(1, &[0.0, 0.0]).unwrap();
        engine.upsert(2, &[1.0, 0.0]).unwrap();
        engine.delete(1).unwrap();
        let hits = engine.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 2);
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn wal_survives_restart_without_flush() {
        let dir = temp_dir("wal");
        let config = EngineConfig {
            sync: SyncPolicy::Never,
            ..EngineConfig::flat(2, Metric::L2)
        };
        {
            let mut engine = Engine::open(&dir, config.clone()).unwrap();
            engine.upsert(4, &[0.0, 1.0]).unwrap();
            engine.upsert(5, &[1.0, 0.0]).unwrap();
        }
        let engine = Engine::open(&dir, config).unwrap();
        assert_eq!(engine.len(), 2);
        let hits = engine.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flush_clears_deleted_ids() {
        let dir = temp_dir("flush");
        let config = EngineConfig {
            sync: SyncPolicy::Never,
            ..EngineConfig::hnsw(4, Metric::L2, HnswParams::default())
        };
        {
            let mut engine = Engine::open(&dir, config.clone()).unwrap();
            engine.upsert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
            engine.upsert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
            engine.delete(1).unwrap();
            engine.flush().unwrap();
        }
        let engine = Engine::open(&dir, config).unwrap();
        assert_eq!(engine.len(), 1);
        let hits = engine.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hnsw_recall_against_flat() {
        let dim = 16;
        let n = 300;
        let k = 10;
        let mut rng = StdRng::seed_from_u64(11);
        let mut flat = Engine::open_memory(EngineConfig::flat(dim, Metric::L2)).unwrap();
        let mut hnsw = Engine::open_memory(EngineConfig::hnsw(
            dim,
            Metric::L2,
            HnswParams {
                m: 8,
                ef_construction: 64,
                ef_search: 64,
                seed: 11,
            },
        ))
        .unwrap();
        for id in 0..n {
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            flat.upsert(id, &vector).unwrap();
            hnsw.upsert(id, &vector).unwrap();
        }
        let mut recall = 0.0;
        let queries = 15;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let truth: Vec<u64> = flat
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            let approx: Vec<u64> = hnsw
                .search(&query, k)
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect();
            recall += recall_at_k(&truth, &approx);
        }
        recall /= queries as f32;
        assert!(recall >= 0.95, "recall {recall:.3} below 0.95");
    }

    #[test]
    fn hnsw_delete_is_hidden_before_flush() {
        let mut engine = Engine::open_memory(EngineConfig::hnsw(
            2,
            Metric::L2,
            HnswParams {
                m: 4,
                ef_construction: 16,
                ef_search: 16,
                seed: 1,
            },
        ))
        .unwrap();
        engine.upsert(1, &[0.0, 0.0]).unwrap();
        engine.upsert(2, &[4.0, 0.0]).unwrap();
        engine.delete(1).unwrap();
        let hits = engine.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 2);
    }

    #[test]
    fn rejects_bad_dim() {
        let mut engine = Engine::open_memory(EngineConfig::flat(4, Metric::L2)).unwrap();
        let err = engine.upsert(1, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            Error::DimMismatch {
                expected: 4,
                got: 2
            }
        ));
    }
}
