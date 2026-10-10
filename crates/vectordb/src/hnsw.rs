use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Reverse;

use crate::arena::Arena;
use crate::distance::distance;
use crate::flat::Hit;
use crate::metric::Metric;

/// Construction and search parameters for the HNSW graph.
///
/// `m` is the number of neighbors per node on upper layers. Layer 0 keeps
/// `2 * m` neighbors, matching the usual hnswlib layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 128,
            seed: 0x5644_4201,
        }
    }
}

impl HnswParams {
    pub fn validate(self) -> Result<(), String> {
        if self.m < 2 {
            return Err("hnsw m must be at least 2".into());
        }
        if self.ef_construction < self.m {
            return Err("hnsw ef_construction must be at least m".into());
        }
        if self.ef_search == 0 {
            return Err("hnsw ef_search must be at least 1".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Cand {
    distance: f32,
    slot: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.slot.cmp(&other.slot))
    }
}

struct Node {
    /// `neighbors[layer]` holds outgoing links. Empty means a slot placeholder.
    neighbors: Vec<Vec<u32>>,
}

impl Node {
    fn placeholder() -> Self {
        Self {
            neighbors: Vec::new(),
        }
    }

    fn with_level(level: usize) -> Self {
        Self {
            neighbors: (0..=level).map(|_| Vec::new()).collect(),
        }
    }

    fn level(&self) -> Option<usize> {
        self.neighbors.len().checked_sub(1)
    }
}

struct Scratch {
    stamps: Vec<u32>,
    stamp: u32,
}

impl Scratch {
    fn new() -> Self {
        Self {
            stamps: Vec::new(),
            stamp: 0,
        }
    }

    fn begin(&mut self, slots: usize) {
        if self.stamps.len() < slots {
            self.stamps.resize(slots, 0);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamps.fill(0);
            self.stamp = 1;
        }
    }

    /// Returns true the first time `slot` is seen during this search.
    fn mark(&mut self, slot: u32) -> bool {
        let idx = slot as usize;
        if self.stamps[idx] == self.stamp {
            return false;
        }
        self.stamps[idx] = self.stamp;
        true
    }
}

/// In-memory Hierarchical Navigable Small World graph.
///
/// The graph stores slot ids. Vectors live in the [`Arena`]. Single-threaded:
/// search scratch uses a `RefCell`.
pub struct HnswIndex {
    metric: Metric,
    params: HnswParams,
    m_max0: usize,
    level_mult: f32,
    rng: StdRng,
    entry: Option<u32>,
    max_level: usize,
    nodes: Vec<Node>,
    scratch: RefCell<Scratch>,
}

impl HnswIndex {
    pub fn new(metric: Metric, params: HnswParams) -> Self {
        let m = params.m.max(2);
        Self {
            metric,
            params,
            m_max0: m * 2,
            level_mult: 1.0 / (m as f32).ln(),
            rng: StdRng::seed_from_u64(params.seed),
            entry: None,
            max_level: 0,
            nodes: Vec::new(),
            scratch: RefCell::new(Scratch::new()),
        }
    }

    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.params.ef_search = ef_search.max(1);
    }

    pub fn insert(&mut self, arena: &Arena, slot: u32) {
        let level = self.sample_level();
        while (self.nodes.len() as u32) < slot {
            self.nodes.push(Node::placeholder());
        }
        debug_assert_eq!(self.nodes.len() as u32, slot);
        self.nodes.push(Node::with_level(level));

        let Some(entry) = self.entry else {
            self.entry = Some(slot);
            self.max_level = level;
            return;
        };

        let query = arena.vector(slot);
        let mut ep = entry;
        // Greedy descent from the top until the new node's level.
        let top = self.max_level;
        if level < top {
            for layer in ((level + 1)..=top).rev() {
                let nearest = self.search_layer(arena, query, ep, 1, layer);
                if let Some(best) = nearest.first() {
                    ep = best.slot;
                }
            }
        }

        let link_until = level.min(top);
        for layer in (0..=link_until).rev() {
            let ef = self.params.ef_construction;
            let candidates = self.search_layer(arena, query, ep, ef, layer);
            if let Some(best) = candidates.first() {
                ep = best.slot;
            }
            let max_links = self.max_links(layer);
            let selected = self.select_neighbors(arena, slot, &candidates, max_links);
            self.nodes[slot as usize].neighbors[layer] = selected.clone();
            for neighbor in selected {
                self.link_back(arena, neighbor, slot, layer, max_links);
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(slot);
        }
    }

    pub fn search(&self, arena: &Arena, query: &[f32], k: usize) -> Vec<Hit> {
        if k == 0 {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut ep = entry;
        if self.max_level > 0 {
            for layer in (1..=self.max_level).rev() {
                let nearest = self.search_layer(arena, query, ep, 1, layer);
                if let Some(best) = nearest.first() {
                    ep = best.slot;
                }
            }
        }
        let ef = self.params.ef_search.max(k);
        let mut hits: Vec<Hit> = self
            .search_layer(arena, query, ep, ef, 0)
            .into_iter()
            .filter(|c| arena.is_alive(c.slot) && self.nodes[c.slot as usize].level().is_some())
            .map(|c| Hit {
                slot: c.slot,
                distance: c.distance,
            })
            .collect();
        hits.truncate(k);
        hits
    }

    fn max_links(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.params.m
        }
    }

    fn sample_level(&mut self) -> usize {
        // hnswlib: floor(-ln(U) * (1 / ln(M))). Clamp so a bad draw cannot
        // create a tower dozens of layers tall on a small index.
        let unit = self.rng.gen_range(f32::EPSILON..1.0);
        let level = (-unit.ln() * self.level_mult).floor() as usize;
        level.min(16)
    }

    fn search_layer(
        &self,
        arena: &Arena,
        query: &[f32],
        entry: u32,
        ef: usize,
        layer: usize,
    ) -> Vec<Cand> {
        let mut scratch = self.scratch.borrow_mut();
        scratch.begin(self.nodes.len());
        scratch.mark(entry);

        let entry_dist = distance(self.metric, query, arena.vector(entry));
        // Closest unexplored candidate. `Reverse` turns the max-heap into a min-heap.
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // Farthest member of the current result window.
        let mut window: BinaryHeap<Cand> = BinaryHeap::new();
        let entry_cand = Cand {
            distance: entry_dist,
            slot: entry,
        };
        candidates.push(Reverse(entry_cand));
        window.push(entry_cand);

        while let Some(Reverse(current)) = candidates.pop() {
            if window.len() >= ef {
                if let Some(worst) = window.peek() {
                    if current.distance > worst.distance {
                        break;
                    }
                }
            }
            let neighbors = match self.nodes.get(current.slot as usize) {
                Some(node) if layer < node.neighbors.len() => node.neighbors[layer].clone(),
                _ => continue,
            };
            for neighbor in neighbors {
                if !scratch.mark(neighbor) {
                    continue;
                }
                let dist = distance(self.metric, query, arena.vector(neighbor));
                let worse_than_window = window.peek().map(|w| dist >= w.distance).unwrap_or(false);
                if window.len() >= ef && worse_than_window {
                    continue;
                }
                let cand = Cand {
                    distance: dist,
                    slot: neighbor,
                };
                candidates.push(Reverse(cand));
                window.push(cand);
                if window.len() > ef {
                    window.pop();
                }
            }
        }

        let mut found: Vec<Cand> = window.into_iter().collect();
        found.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.slot.cmp(&b.slot)));
        found
    }

    /// Keep a diverse set of neighbors, then fill any free slots with the closest
    /// candidates the diversity rule rejected.
    ///
    /// The diversity rule alone drops degree toward 1 on high-dimensional
    /// random vectors, and search recall falls with it. Filling back up to
    /// `max_links` is the usual keep-pruned-connections step.
    fn select_neighbors(
        &self,
        arena: &Arena,
        query_slot: u32,
        candidates: &[Cand],
        max_links: usize,
    ) -> Vec<u32> {
        let mut selected = Vec::with_capacity(max_links.min(candidates.len()));
        let mut discarded = Vec::new();
        for cand in candidates {
            if cand.slot == query_slot {
                continue;
            }
            if selected.len() >= max_links {
                break;
            }
            let cand_vec = arena.vector(cand.slot);
            let mut dominated = false;
            for &kept in &selected {
                let dist_to_kept = distance(self.metric, cand_vec, arena.vector(kept));
                if dist_to_kept < cand.distance {
                    dominated = true;
                    break;
                }
            }
            if dominated {
                discarded.push(cand.slot);
            } else {
                selected.push(cand.slot);
            }
        }
        for slot in discarded {
            if selected.len() >= max_links {
                break;
            }
            selected.push(slot);
        }
        selected
    }

    fn link_back(
        &mut self,
        arena: &Arena,
        neighbor: u32,
        slot: u32,
        layer: usize,
        max_links: usize,
    ) {
        let overflow = {
            let links = &mut self.nodes[neighbor as usize].neighbors[layer];
            if links.contains(&slot) {
                return;
            }
            links.push(slot);
            links.len() > max_links
        };
        if !overflow {
            return;
        }
        let existing = self.nodes[neighbor as usize].neighbors[layer].clone();
        let origin = arena.vector(neighbor);
        let mut candidates = Vec::with_capacity(existing.len());
        for other in existing {
            candidates.push(Cand {
                distance: distance(self.metric, origin, arena.vector(other)),
                slot: other,
            });
        }
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.slot.cmp(&b.slot)));
        let pruned = self.select_neighbors(arena, neighbor, &candidates, max_links);
        self.nodes[neighbor as usize].neighbors[layer] = pruned;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use rand::Rng;

    #[test]
    fn recall_matches_exact_on_a_small_set() {
        let dim = 16;
        let n = 400;
        let queries = 20;
        let k = 10;
        let mut rng = StdRng::seed_from_u64(7);
        let mut arena = Arena::new(dim);
        for id in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            arena.append(id, &v);
        }
        let mut hnsw = HnswIndex::new(
            Metric::L2,
            HnswParams {
                m: 8,
                ef_construction: 64,
                ef_search: 64,
                seed: 7,
            },
        );
        for slot in 0..n {
            hnsw.insert(&arena, slot as u32);
        }
        let flat = FlatIndex::new(Metric::L2);
        let mut recall_sum = 0.0;
        for _ in 0..queries {
            let q: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let truth = flat.search(&arena, &q, k);
            let approx = hnsw.search(&arena, &q, k);
            let truth_ids: std::collections::HashSet<u64> =
                truth.iter().map(|h| arena.id(h.slot)).collect();
            let hits = approx
                .iter()
                .filter(|h| truth_ids.contains(&arena.id(h.slot)))
                .count();
            recall_sum += hits as f32 / k as f32;
        }
        let recall = recall_sum / queries as f32;
        assert!(
            recall >= 0.95,
            "expected recall@10 >= 0.95, got {recall:.3}"
        );
    }
}
