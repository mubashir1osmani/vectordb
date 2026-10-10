use crate::metric::Metric;

/// Dot product. The compiler autovectorizes the unrolled body.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    let mut i = 0;
    let n = a.len().min(b.len());
    while i + 8 <= n {
        acc += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3]
            + a[i + 4] * b[i + 4]
            + a[i + 5] * b[i + 5]
            + a[i + 6] * b[i + 6]
            + a[i + 7] * b[i + 7];
        i += 8;
    }
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// Squared Euclidean distance. Ordering matches L2, without a sqrt.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    let mut i = 0;
    let n = a.len().min(b.len());
    while i + 8 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        let d4 = a[i + 4] - b[i + 4];
        let d5 = a[i + 5] - b[i + 5];
        let d6 = a[i + 6] - b[i + 6];
        let d7 = a[i + 7] - b[i + 7];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        i += 8;
    }
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc
}

/// Smaller means nearer for every metric.
///
/// Cosine and inner product are returned as negative dot products so one
/// ordering works for exact search and for the graph.
#[inline]
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2_sq(a, b),
        Metric::Cosine | Metric::InnerProduct => -dot(a, b),
    }
}

/// In-place L2 normalization. A zero vector is left unchanged.
pub fn normalize(v: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for x in v.iter() {
        norm_sq += x * x;
    }
    if norm_sq == 0.0 {
        return;
    }
    let inv = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub(crate) fn prepare_vector(metric: Metric, src: &[f32]) -> Vec<f32> {
    let mut out = src.to_vec();
    if metric == Metric::Cosine {
        normalize(&mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_of_three_four_five() {
        let d = l2_sq(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 25.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_normalizes_to_unit() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let n = dot(&v, &v);
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inner_product_order_is_negative_dot() {
        let a = [1.0, 0.0];
        let near = [0.9, 0.1];
        let far = [0.0, 1.0];
        let d_near = distance(Metric::InnerProduct, &a, &near);
        let d_far = distance(Metric::InnerProduct, &a, &far);
        assert!(d_near < d_far);
    }
}
