use std::collections::HashSet;

/// Fraction of `truth` ids that also appear in `approx`.
///
/// `truth` is the exact top-k. The score is `|intersection| / |truth|`.
pub fn recall_at_k(truth: &[u64], approx: &[u64]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_set: HashSet<u64> = truth.iter().copied().collect();
    let hits = approx.iter().filter(|id| truth_set.contains(id)).count();
    hits as f32 / truth.len() as f32
}

/// `p` is in `0.0..=100.0`. `samples` is sorted ascending.
pub fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = ((sorted.len() - 1) as f64) * (p / 100.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        ((sorted[lo] as f64) * (1.0 - w) + (sorted[hi] as f64) * w).round() as u128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_half() {
        let score = recall_at_k(&[1, 2, 3, 4], &[1, 2, 9, 8]);
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn percentile_median() {
        assert_eq!(percentile(&[10, 20, 30, 40], 50.0), 25);
    }
}
