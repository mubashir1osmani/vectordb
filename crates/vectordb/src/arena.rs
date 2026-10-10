use std::collections::HashMap;

/// Contiguous vector storage. Slot ids are stable until compaction.
#[derive(Clone, Debug)]
pub struct Arena {
    dim: usize,
    data: Vec<f32>,
    ids: Vec<u64>,
    alive: Vec<bool>,
    id_to_slot: HashMap<u64, u32>,
}

impl Arena {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            ids: Vec::new(),
            alive: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn slot_count(&self) -> usize {
        self.ids.len()
    }

    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|a| **a).count()
    }

    pub fn slot_of(&self, id: u64) -> Option<u32> {
        self.id_to_slot.get(&id).copied()
    }

    pub fn is_alive(&self, slot: u32) -> bool {
        self.alive.get(slot as usize).copied().unwrap_or(false)
    }

    pub fn id(&self, slot: u32) -> u64 {
        self.ids[slot as usize]
    }

    pub fn vector(&self, slot: u32) -> &[f32] {
        let start = slot as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    pub fn append(&mut self, id: u64, vector: &[f32]) -> u32 {
        debug_assert_eq!(vector.len(), self.dim);
        debug_assert!(!self.id_to_slot.contains_key(&id));
        let slot = self.ids.len() as u32;
        self.data.extend_from_slice(vector);
        self.ids.push(id);
        self.alive.push(true);
        self.id_to_slot.insert(id, slot);
        slot
    }

    pub fn set(&mut self, slot: u32, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        let start = slot as usize * self.dim;
        self.data[start..start + self.dim].copy_from_slice(vector);
    }

    /// Marks the id dead. The slot stays so graph edges remain addressable.
    pub fn kill(&mut self, id: u64) -> Option<u32> {
        let slot = self.id_to_slot.remove(&id)?;
        self.alive[slot as usize] = false;
        Some(slot)
    }

    /// Drops dead slots and renumbers survivors from zero.
    ///
    /// Graph indexes store slot ids, so the caller rebuilds them after this.
    pub fn compact(&mut self) {
        let live = self.alive_count();
        let mut data = Vec::with_capacity(live * self.dim);
        let mut ids = Vec::with_capacity(live);
        let mut alive = Vec::with_capacity(live);
        let mut id_to_slot = HashMap::with_capacity(live);
        for slot in 0..self.ids.len() {
            if !self.alive[slot] {
                continue;
            }
            let new_slot = ids.len() as u32;
            let id = self.ids[slot];
            data.extend_from_slice(self.vector(slot as u32));
            ids.push(id);
            alive.push(true);
            id_to_slot.insert(id, new_slot);
        }
        self.data = data;
        self.ids = ids;
        self.alive = alive;
        self.id_to_slot = id_to_slot;
    }
}
