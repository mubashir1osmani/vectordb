use std::fmt;
use std::io;

/// Recoverable failure from the engine, index, or storage layer.
#[derive(Debug)]
pub enum Error {
    DimMismatch { expected: usize, got: usize },
    NonFinite,
    EmptyVector,
    NotFound(u64),
    ConfigMismatch(String),
    Corrupt(String),
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimMismatch { expected, got } => {
                write!(f, "vector dim {got} does not match index dim {expected}")
            }
            Error::NonFinite => write!(f, "vector contains NaN or infinity"),
            Error::EmptyVector => write!(f, "vector dimension must be greater than zero"),
            Error::NotFound(id) => write!(f, "id {id} is not in the index"),
            Error::ConfigMismatch(msg) => write!(f, "config does not match stored database: {msg}"),
            Error::Corrupt(msg) => write!(f, "corrupt storage: {msg}"),
            Error::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
