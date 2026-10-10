//! How a caller uses the engine.
//!
//! One directory holds the database. Upserts and deletes go to the write-ahead
//! log immediately. Search sees them in this process. Dropping the engine is a
//! crash: the next open replays the log. `flush` writes a segment and empties
//! the log.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vectordb::{Engine, EngineConfig, HnswParams, Metric, SyncPolicy};

fn workspace() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("vdb-e2e-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

fn config() -> EngineConfig {
    EngineConfig {
        sync: SyncPolicy::Always,
        ..EngineConfig::hnsw(
            4,
            Metric::Cosine,
            HnswParams {
                m: 8,
                ef_construction: 32,
                ef_search: 32,
                seed: 7,
            },
        )
    }
}

fn ids_of(engine: &Engine, query: &[f32], k: usize) -> Vec<u64> {
    engine
        .search(query, k)
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

#[test]
fn document_search_survives_restart_and_checkpoint() {
    let dir = workspace();
    let rust = [1.0, 0.0, 0.0, 0.0];
    let pasta = [0.0, 0.0, 1.0, 0.0];

    // Open a new database and insert five documents.
    let mut engine = Engine::open(&dir, config()).unwrap();
    assert!(engine.is_empty());
    engine.upsert(1, &rust).unwrap();
    engine.upsert(2, &[1.0, 1.0, 0.0, 0.0]).unwrap();
    engine.upsert(3, &pasta).unwrap();
    engine.upsert(4, &[0.0, 0.0, 1.0, 1.0]).unwrap();
    engine.upsert(5, &[0.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(engine.len(), 5);

    // A rust-shaped query returns the rust documents, nearest first.
    assert_eq!(ids_of(&engine, &rust, 2), vec![1, 2]);
    let hits = engine.search(&rust, 2).unwrap();
    assert!(hits[0].distance < hits[1].distance);

    // Replace document 2 with a pasta vector. The new bytes are searchable
    // immediately. Document 3 is an exact copy, so the top two are {2, 3}.
    engine.upsert(2, &pasta).unwrap();
    let mut pasta_hits = ids_of(&engine, &pasta, 2);
    pasta_hits.sort_unstable();
    assert_eq!(pasta_hits, vec![2, 3]);
    assert_eq!(ids_of(&engine, &rust, 1), vec![1]);

    // Delete the other pasta document. It must disappear from search.
    engine.delete(3).unwrap();
    assert_eq!(engine.len(), 4);
    assert_eq!(ids_of(&engine, &pasta, 1), vec![2]);
    assert!(engine.delete(3).is_err());

    // Process exit before flush. The log still has every operation.
    drop(engine);
    assert!(dir.join("wal.bin").metadata().unwrap().len() > 0);
    assert!(!dir.join("segment.bin").exists());

    // Reopen replays the log. Updates and the delete are still there.
    let mut engine = Engine::open(&dir, config()).unwrap();
    assert_eq!(engine.len(), 4);
    assert_eq!(ids_of(&engine, &pasta, 1), vec![2]);
    assert_eq!(ids_of(&engine, &rust, 1), vec![1]);
    assert!(!ids_of(&engine, &pasta, 4).contains(&3));

    // Checkpoint compacts the live set into a segment and clears the log.
    engine.flush().unwrap();
    drop(engine);
    assert!(dir.join("segment.bin").metadata().unwrap().len() > 0);
    assert_eq!(dir.join("wal.bin").metadata().unwrap().len(), 0);
    assert!(dir.join("meta.bin").exists());

    // A later open reads the segment. The same queries return the same ids.
    let engine = Engine::open(&dir, config()).unwrap();
    assert_eq!(engine.index_name(), "hnsw");
    assert_eq!(engine.dim(), 4);
    assert_eq!(engine.metric(), Metric::Cosine);
    assert_eq!(ids_of(&engine, &pasta, 1), vec![2]);
    assert_eq!(ids_of(&engine, &rust, 1), vec![1]);

    // The stored dim and metric are fixed. A different metric is rejected.
    let wrong = EngineConfig::flat(4, Metric::L2);
    match Engine::open(&dir, wrong) {
        Err(err) => {
            let message = err.to_string();
            assert!(message.contains("metric"), "{message}");
        }
        Ok(_) => panic!("opening with a different metric should fail"),
    }

    // The same bytes can be opened with an exact index. On this collection
    // HNSW and the exact scan agree.
    let flat = EngineConfig {
        sync: SyncPolicy::Always,
        ..EngineConfig::flat(4, Metric::Cosine)
    };
    let flat = Engine::open(&dir, flat).unwrap();
    assert_eq!(flat.index_name(), "flat");
    assert_eq!(ids_of(&flat, &pasta, 4), ids_of(&engine, &pasta, 4));
    assert_eq!(ids_of(&flat, &rust, 4), ids_of(&engine, &rust, 4));

    let _ = fs::remove_dir_all(&dir);
}
