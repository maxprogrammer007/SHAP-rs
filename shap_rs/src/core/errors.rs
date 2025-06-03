// src/core/errors.rs
use std::fmt;
use ndarray; // Required for ndarray::ShapeError

#[derive(Debug)]
pub enum ShapError {
    InvalidInput(String),
    IncompatibleDimensions(String),
    MissingValue(String),
    ModelPredictionError(String),
    InternalError(String),
    NdarrayError(String), // New variant for ndarray errors
    // Add more specific error types as needed
}

impl fmt::Display for ShapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapError::InvalidInput(msg) => write!(f, "Invalid Input: {}", msg),
            ShapError::IncompatibleDimensions(msg) => write!(f, "Incompatible Dimensions: {}", msg),
            ShapError::MissingValue(msg) => write!(f, "Missing Value: {}", msg),
            ShapError::ModelPredictionError(msg) => write!(f, "Model Prediction Error: {}", msg),
            ShapError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
            ShapError::NdarrayError(msg) => write!(f, "Ndarray Error: {}", msg), // Display for new variant
        }
    }
}

impl std::error::Error for ShapError {} // Allow ? operator with this error type

// Implement From<ndarray::ShapeError> for ShapError
impl From<ndarray::ShapeError> for ShapError {
    fn from(err: ndarray::ShapeError) -> Self {
        ShapError::NdarrayError(format!("ndarray ShapeError: {}", err.to_string()))
    }
}

// Convenience type alias for Result
pub type Result<T> = std::result::Result<T, ShapError>;
