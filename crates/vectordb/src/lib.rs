//! Single-node vector database.
//!
//! This crate is the first slice of the 1B / 100ms design in `SCOPE.md`:
//! an exact flat index, an in-memory HNSW index, and a crash-safe WAL.
//! Disk-graph search (Vamana) and product quantization are later phases.
//! The flat index is the recall oracle for every approximate index.
//!
//! Distances are **smaller-is-nearer**. Cosine vectors are normalized on
//! write and on query.

mod arena;
mod binio;
mod distance;
mod engine;
mod error;
mod eval;
mod flat;
mod hnsw;
mod metric;
mod segment;
mod wal;

pub use engine::{Engine, EngineConfig, Hit};
pub use error::{Error, Result};
pub use eval::{percentile, recall_at_k};
pub use hnsw::HnswParams;
pub use metric::Metric;
pub use segment::IndexParams;
pub use wal::SyncPolicy;
