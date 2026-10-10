use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::arena::Arena;
use crate::distance::distance;
use crate::metric::Metric;

/// One neighbor. `distance` is smaller-is-nearer for every metric.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub slot: u32,
    pub distance: f32,
}

#[derive(Clone, Copy)]
struct Ranked {
    distance: f32,
    slot: u32,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.slot.cmp(&other.slot))
    }
}

/// Exact scan. This is the recall oracle for the approximate indexes.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    metric: Metric,
}

impl FlatIndex {
    pub fn new(metric: Metric) -> Self {
        Self { metric }
    }

    pub fn search(&self, arena: &Arena, query: &[f32], k: usize) -> Vec<Hit> {
        if k == 0 || arena.alive_count() == 0 {
            return Vec::new();
        }
        // Max-heap of the current k nearest. The worst of those sits on top.
        let mut best: BinaryHeap<Ranked> = BinaryHeap::new();
        for slot in 0..arena.slot_count() {
            let slot = slot as u32;
            if !arena.is_alive(slot) {
                continue;
            }
            let distance = distance(self.metric, query, arena.vector(slot));
            if best.len() < k {
                best.push(Ranked { distance, slot });
                continue;
            }
            if let Some(worst) = best.peek() {
                if distance < worst.distance {
                    best.pop();
                    best.push(Ranked { distance, slot });
                }
            }
        }
        let mut hits: Vec<Hit> = best
            .into_iter()
            .map(|r| Hit {
                slot: r.slot,
                distance: r.distance,
            })
            .collect();
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.slot.cmp(&b.slot)));
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_identical_point() {
        let mut arena = Arena::new(2);
        arena.append(1, &[0.0, 0.0]);
        arena.append(2, &[10.0, 0.0]);
        arena.append(3, &[0.0, 10.0]);
        let index = FlatIndex::new(Metric::L2);
        let hits = index.search(&arena, &[0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(arena.id(hits[0].slot), 1);
        assert!(hits[0].distance < 1e-6);
    }

    #[test]
    fn skips_deleted_slots() {
        let mut arena = Arena::new(2);
        arena.append(1, &[0.0, 0.0]);
        arena.append(2, &[1.0, 0.0]);
        arena.kill(1);
        let index = FlatIndex::new(Metric::L2);
        let hits = index.search(&arena, &[0.0, 0.0], 1);
        assert_eq!(arena.id(hits[0].slot), 2);
    }
}
