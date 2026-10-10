/// Distance used at insert and query time.
///
/// Cosine vectors are L2-normalized on write and on query, then scored as
/// inner product. Stored bytes are the normalized values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance. Smaller is nearer.
    L2,
    /// Cosine similarity, stored as normalized inner product.
    Cosine,
    /// Raw inner product. Larger dot product is nearer.
    InnerProduct,
}

impl Metric {
    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Metric::L2),
            2 => Some(Metric::Cosine),
            3 => Some(Metric::InnerProduct),
            _ => None,
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            Metric::L2 => 1,
            Metric::Cosine => 2,
            Metric::InnerProduct => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Cosine => "cosine",
            Metric::InnerProduct => "ip",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "l2" | "euclidean" => Some(Metric::L2),
            "cosine" => Some(Metric::Cosine),
            "ip" | "dot" | "inner" => Some(Metric::InnerProduct),
            _ => None,
        }
    }
}
