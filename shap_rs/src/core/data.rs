// src/core/data.rs
use ndarray::{Array1, Array2};
use std::fmt;

/// Represents a single data instance (a row of features).
/// We use `f64` for flexibility with various model outputs and calculations.
pub type Instance = Array1<f64>;

/// Represents a dataset (multiple instances, e.g., background data).
pub type Dataset = Array2<f64>;

/// Represents the output of a SHAP explanation for a single instance.
#[derive(Debug, Clone)] // PartialEq will be useful for testing
pub struct Explanation {
    /// SHAP values, one for each feature.
    pub shap_values: Array1<f64>,
    /// The base value, E[f(x)], or the average model output over the background data.
    pub expected_value: f64,
    /// The actual prediction for the instance being explained.
    pub actual_prediction: f64,
    /// Optional: The instance that was explained.
    /// Could be useful for plotting or context.
    pub instance: Option<Instance>,
    // Potentially feature names in the future: pub feature_names: Option<Vec<String>>,
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Explanation:")?;
        writeln!(f, "  Expected Value (Base): {:.4}", self.expected_value)?;
        writeln!(f, "  Actual Prediction:     {:.4}", self.actual_prediction)?;
        writeln!(f, "  SHAP Values:")?;
        for (i, val) in self.shap_values.iter().enumerate() {
            writeln!(f, "    Feature {}: {:.4}", i, val)?;
        }
        if let Some(inst) = &self.instance {
            writeln!(f, "  Instance Values (first 10):")?;
            for (i, val) in inst.iter().take(10).enumerate() {
                 writeln!(f, "    Feature {}: {:.4}", i, val)?;
            }
            if inst.len() > 10 {
                 writeln!(f, "    ...")?;
            }
        }
        Ok(())
    }
}

// We might add more complex types or traits for data handling later,
// especially if we want to support sparse data or different data types.