use thiserror::Error;

#[derive(Error, Debug)]
pub enum XimError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Shape mismatch: expected {expected}, found {found}")]
    ShapeMismatch { expected: usize, found: usize },

    #[error("Other error: {0}")]
    Other(String),
}

impl From<bincode::Error> for XimError {
    fn from(err: bincode::Error) -> Self {
        XimError::Serialization(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, XimError>;
