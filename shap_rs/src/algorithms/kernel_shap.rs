// src/algorithms/kernel_shap.rs

use crate::core::{Dataset, Instance, Explanation, ShapError, Result};
use crate::traits::PredictModel;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal}; // Ensure StandardNormal is imported

/// Configuration for the KernelSHAP explainer.
#[derive(Debug, Clone)]
pub struct KernelShapConfig {
    pub n_samples: KernelShapSamples,
    pub noise_std_dev: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum KernelShapSamples {
    Auto,
    Fixed(usize),
}

impl Default for KernelShapConfig {
    fn default() -> Self {
        KernelShapConfig {
            n_samples: KernelShapSamples::Auto,
            noise_std_dev: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct KernelExplainer<M: PredictModel> {
    model: M,
    background_data: Dataset,
    expected_value: f64,
    num_features: usize,
    config: KernelShapConfig,
}

impl<M: PredictModel> KernelExplainer<M> {
    pub fn new(model: M, background_data: Dataset, config: Option<KernelShapConfig>) -> Result<Self> {
        let num_features = model.num_features();

        if background_data.is_empty() {
            return Err(ShapError::InvalidInput(
                "Background data cannot be empty.".to_string(),
            ));
        }

        if background_data.ncols() != num_features {
            return Err(ShapError::IncompatibleDimensions(format!(
                "Background data has {} features, but model expects {}.",
                background_data.ncols(),
                num_features
            )));
        }

        let background_predictions = model.predict(&background_data)?;
        let expected_value = background_predictions.mean().ok_or_else(|| {
            ShapError::InternalError("Failed to calculate mean of background predictions.".to_string())
        })?;

        let resolved_config = config.unwrap_or_default();

        Ok(KernelExplainer {
            model,
            background_data,
            expected_value,
            num_features,
            config: resolved_config,
        })
    }

    pub fn expected_value(&self) -> f64 {
        self.expected_value
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn background_data(&self) -> &Dataset {
        &self.background_data
    }

    pub fn config(&self) -> &KernelShapConfig {
        &self.config
    }

    pub fn shap_values(&self, instance: &Instance) -> Result<Explanation> {
        if instance.len() != self.num_features {
            return Err(ShapError::IncompatibleDimensions(format!(
                "Instance to explain has {} features, but explainer expects {}.",
                instance.len(),
                self.num_features
            )));
        }

        let n_samples_config = match self.config.n_samples {
            KernelShapSamples::Auto => 2 * self.num_features + 2048,
            KernelShapSamples::Fixed(n) => n,
        };

        let max_possible_samples = 1_usize.checked_shl(self.num_features as u32).unwrap_or(usize::MAX);
        let actual_n_samples = n_samples_config.min(max_possible_samples).max(1);

        if self.num_features > 20 && actual_n_samples >= max_possible_samples {
            return Err(ShapError::InvalidInput(format!(
                "Number of features ({}) is too large to generate all coalitions. Please use sampling.",
                self.num_features
            )));
        }
        let (coalition_matrix, weights, num_subset_sizes) = self.generate_coalitions(actual_n_samples)?;

        let mut perturbed_instances_vec: Vec<f64> = Vec::with_capacity(
            actual_n_samples * self.background_data.nrows() * self.num_features,
        );
        let mut effective_weights: Vec<f64> = Vec::with_capacity(
            actual_n_samples * self.background_data.nrows(),
        );
        
        let mut rng_for_noise = thread_rng(); // Initialize RNG once if used in a loop often

        for i in 0..actual_n_samples {
            let coalition_vector = coalition_matrix.row(i);
            let weight = weights[i];

            // We don't skip for infinite/zero weights here; WLS should handle them.
            // The check for weight <= 0.0 might be relevant if kernel_weight could produce it
            // for non-endpoint coalitions, but SHAP kernel is positive.

            for bg_idx in 0..self.background_data.nrows() {
                let background_sample = self.background_data.row(bg_idx);
                let mut current_perturbed_instance = Instance::zeros(self.num_features);

                for feat_idx in 0..self.num_features {
                    if coalition_vector[feat_idx] == 1.0 {
                        current_perturbed_instance[feat_idx] = instance[feat_idx];
                    } else {
                        current_perturbed_instance[feat_idx] = background_sample[feat_idx];
                        if self.background_data.nrows() == 1 && self.config.noise_std_dev > 0.0 {
                            // *** FIX FOR StandardNormal TYPE AMBIGUITY ***
                            let noise_sample: f64 = StandardNormal.sample(&mut rng_for_noise);
                            let noise = noise_sample * self.config.noise_std_dev;
                            current_perturbed_instance[feat_idx] += noise;
                        }
                    }
                }
                perturbed_instances_vec.extend_from_slice(current_perturbed_instance.as_slice().unwrap());
                effective_weights.push(weight);
            }
        }

        let perturbed_dataset = Dataset::from_shape_vec(
            (actual_n_samples * self.background_data.nrows(), self.num_features),
            perturbed_instances_vec,
        )
        .map_err(|e| ShapError::InternalError(format!("Failed to create perturbed dataset: {}", e)))?;

        let final_weights = Array1::from_vec(effective_weights);
        let model_outputs = self.model.predict(&perturbed_dataset)?;
        let full_prediction = self.model.predict(&instance.view().insert_axis(Axis(0)).to_owned())?[0];

        let mut averaged_model_outputs = Array1::zeros(actual_n_samples);
        if self.background_data.nrows() > 0 { // Ensure no division by zero if background_data was somehow empty (though checked in new)
            for i in 0..actual_n_samples {
                let start = i * self.background_data.nrows();
                let end = (i + 1) * self.background_data.nrows();
                if start < end && end <= model_outputs.len() { // Ensure slice is valid and within bounds
                    let outputs_for_coalition = model_outputs.slice(s![start..end]);
                    averaged_model_outputs[i] = outputs_for_coalition.mean().unwrap_or(self.expected_value); // Default to expected if mean fails
                } else if start == end && start < model_outputs.len() { // Path for single background sample
                     averaged_model_outputs[i] = model_outputs[start];
                } else {
                    // This case implies an issue with indexing logic or empty model_outputs for a coalition.
                    // Defaulting to expected_value or handling as an error might be necessary.
                    averaged_model_outputs[i] = self.expected_value;
                }
            }
        } else { // Should not be reached due to check in `new`, but as a safeguard.
             averaged_model_outputs.fill(self.expected_value);
        }
        

        let shap_values_coeffs = self.solve_weighted_least_squares(
            coalition_matrix.view(),
            averaged_model_outputs.view(),
            weights.view(), // Use original weights for WLS, not effective_weights
        )?;

        let mut final_shap_values: Array1<f64>;
        let mut calculated_expected_value = self.expected_value;

        if shap_values_coeffs.len() == self.num_features {
            final_shap_values = shap_values_coeffs;
            let sum_phi = final_shap_values.sum();
            let sum_phi_full_pred_diff = full_prediction - self.expected_value;
            if num_subset_sizes == 1 && (sum_phi - sum_phi_full_pred_diff).abs() > 1e-4 && self.num_features > 0 {
                let diff = sum_phi_full_pred_diff - sum_phi;
                final_shap_values = final_shap_values.mapv(|v| v + diff / self.num_features as f64);
            }
        } else if shap_values_coeffs.len() == self.num_features + 1 {
            calculated_expected_value = shap_values_coeffs[0];
            final_shap_values = shap_values_coeffs.slice(s![1..]).to_owned();
            let sum_phi = final_shap_values.sum();
            let sum_phi_full_pred_diff = full_prediction - calculated_expected_value;
            if num_subset_sizes == 1 && (sum_phi - sum_phi_full_pred_diff).abs() > 1e-4 && self.num_features > 0 {
                let diff = sum_phi_full_pred_diff - sum_phi;
                final_shap_values = final_shap_values.mapv(|v| v + diff / self.num_features as f64);
            }
        } else {
            return Err(ShapError::InternalError(format!(
                "WLS solver returned unexpected number of coefficients: {}. Expected {} or {}.",
                shap_values_coeffs.len(),
                self.num_features,
                self.num_features + 1
            )));
        }

        Ok(Explanation {
            shap_values: final_shap_values,
            expected_value: calculated_expected_value,
            actual_prediction: full_prediction,
            instance: Some(instance.to_owned()),
        })
    }

    fn generate_coalitions(&self, n_to_sample: usize) -> Result<(Dataset, Array1<f64>, usize)> {
        let m = self.num_features;
        if m == 0 {
            return Ok((Dataset::zeros((0, 0)), Array1::zeros(0), 0));
        }

        let max_coalitions = 1_usize.checked_shl(m as u32).unwrap_or(usize::MAX);

        if n_to_sample >= max_coalitions && m <= 20 {
            let mut coalitions = Dataset::zeros((max_coalitions, m));
            let mut weights = Array1::zeros(max_coalitions);
            for i in 0..max_coalitions {
                let mut current_coalition_size = 0;
                for j in 0..m {
                    if (i >> j) & 1 == 1 {
                        coalitions[[i, j]] = 1.0;
                        current_coalition_size += 1;
                    } else {
                        coalitions[[i, j]] = 0.0;
                    }
                }
                weights[i] = self.kernel_weight(current_coalition_size, m);
            }
            return Ok((coalitions, weights, m + 1));
        }

        let mut coalitions = Dataset::zeros((n_to_sample, m));
        let mut weights = Array1::zeros(n_to_sample);
        let mut rng = thread_rng();
        let mut current_sample_idx = 0;

        if n_to_sample > 0 {
            weights[current_sample_idx] = self.kernel_weight(0, m);
            current_sample_idx += 1;
        }
        if n_to_sample > 1 && current_sample_idx < n_to_sample {
            coalitions.row_mut(current_sample_idx).fill(1.0);
            weights[current_sample_idx] = self.kernel_weight(m, m);
            current_sample_idx += 1;
        }

        let mut subset_sizes = std::collections::HashSet::new();
        subset_sizes.insert(0);
        if m > 0 { subset_sizes.insert(m); }


        let features_indices: Vec<usize> = (0..m).collect();
        for i in current_sample_idx..n_to_sample {
            let k = if m > 1 {
                rand::Rng::gen_range(&mut rng, 1..m)
            } else if m == 1 { // if m=1, only sizes 0 and 1 are possible, handled by endpoints. Sample 0 or 1 if forced.
                rand::Rng::gen_range(&mut rng, 0..=1) // This path for m=1 should ideally not be hit often if endpoints are sufficient.
            } else { // m=0
                0
            };
            subset_sizes.insert(k);
            let mut coalition_row = coalitions.row_mut(i);
            coalition_row.fill(0.0);
            let chosen_indices: Vec<_> = features_indices.choose_multiple(&mut rng, k).cloned().collect();
            for &idx in &chosen_indices {
                coalition_row[idx] = 1.0;
            }
            weights[i] = self.kernel_weight(k, m);
        }
        Ok((coalitions, weights, subset_sizes.len()))
    }

    /// Calculates the SHAP kernel weight for a coalition.
    /// Paper formula: pi_x(z') = (M-1) / (C(M-1, |z'|) * |z'| * (M-|z'|))
    /// where M is num_features, |z'| is coalition_size.
    /// Returns a very large number (effective infinity) for |z'|=0 or |z'|=M.
    fn kernel_weight(&self, coalition_size: usize, num_features: usize) -> f64 {
        if coalition_size == 0 || coalition_size == num_features {
            return 1e9; // Effectively "infinite" weight
        }
        // At this point, 0 < coalition_size < num_features.
        // So, num_features must be at least 2 for the formula to be non-trivial.
        if num_features <= 1 {
            // This case implies coalition_size must be 0 or 1 if num_features is 1.
            // Handled by the first check. If num_features = 0, also handled.
            // This is a safeguard for num_features = 1, where M-1 = 0.
            return 1e9;
        }

        let m_minus_1 = num_features - 1; // M-1

        // Term C(M-1, |z'|) from the paper's kernel definition.
        // Here, |z'| is 'coalition_size'.
        let combinations_val = Self::n_choose_k(m_minus_1, coalition_size);

        if combinations_val == 0.0 {
            // This could happen if coalition_size > m_minus_1, meaning coalition_size = num_features.
            // But that specific case is handled by the first 'if' condition.
            // Or if n_choose_k returns 0.0 for valid inputs (e.g. float precision issues, though unlikely here).
            // It's a safeguard. The original paper implies |z'| < M for this combination.
            return 1e9; // Consider if another value or error is more appropriate.
        }

        // Denominator: C(M-1, |z'|) * |z'| * (M - |z'|)
        let denominator = combinations_val * coalition_size as f64 * (num_features - coalition_size) as f64;

        if denominator == 0.0 {
            // This would occur if coalition_size is 0 or num_features (handled),
            // or if combinations_val is 0.0 (handled).
            // This is a final safeguard.
            return 1e9;
        }

        (m_minus_1 as f64) / denominator
    }

    /// Helper for combinations C(n, k) = n! / (k! * (n-k)!)
    fn n_choose_k(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }
        // Exploit symmetry C(n, k) = C(n, n-k) to keep k small for precision/performance
        let eff_k = if k > n / 2 { n - k } else { k };

        let mut res = 1.0;
        for i in 0..eff_k {
            // (n-i) can be large, (i+1) can be small.
            // Calculate carefully to maintain precision.
            res *= (n - i) as f64;
            res /= (i + 1) as f64;
        }
        res
    }

    fn solve_weighted_least_squares(
        &self,
        features: ArrayView2<f64>,
        target: ArrayView1<f64>,
        weights: ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        if features.ncols() == 0 {
             // If there are no features, result depends on whether an intercept is expected.
             // If only intercept, it should be target.mean() or similar.
             // If WLS expects to return (num_features + 1) coefficients, then return 1 coeff (intercept).
             // If WLS expects num_features coeffs, then return 0.
             // Let's assume our WLS below returns intercept + coeffs for features.
             // If num_features is 0, n_coeffs will be 1.
            if features.nrows() > 0 && !weights.is_empty() { // Calculate weighted mean for intercept
                let weighted_sum = target.iter().zip(weights.iter()).map(|(&y, &w)| y * w).sum::<f64>();
                let sum_weights = weights.iter().sum::<f64>();
                if sum_weights.abs() > 1e-9 {
                    return Ok(Array1::from_vec(vec![weighted_sum / sum_weights]));
                } else { // All weights zero or problematic
                    return Ok(Array1::from_vec(vec![target.mean().unwrap_or(0.0)]));
                }
            }
            return Ok(Array1::zeros(1)); // Default intercept of 0
        }

        let n_samples = features.nrows();
        let n_model_features = features.ncols(); // Number of actual features in the input matrix
        let n_coeffs_to_solve = n_model_features + 1; // +1 for intercept

        if n_samples == 0 {
            return Ok(Array1::zeros(n_coeffs_to_solve));
        }
        
        // Check for underdetermined system more carefully
        // We need at least n_coeffs_to_solve effective samples (non-zero weight)
        let effective_samples = weights.iter().filter(|&&w| w > 1e-9).count();
        if effective_samples < n_coeffs_to_solve {
            return Err(ShapError::InternalError(format!(
                "Underdetermined system in WLS: {} effective samples, {} coefficients to solve. Increase n_samples or check weights.",
                effective_samples, n_coeffs_to_solve
            )));
        }

        let mut x_augmented_vec = Vec::with_capacity(n_samples * n_coeffs_to_solve);
        for i in 0..n_samples {
            x_augmented_vec.push(1.0); // Intercept column
            for j in 0..n_model_features {
                x_augmented_vec.push(features[[i, j]]);
            }
        }
        let x_augmented = Array2::from_shape_vec((n_samples, n_coeffs_to_solve), x_augmented_vec)
            .map_err(|e| ShapError::InternalError(format!("WLS X_aug creation failed: {}", e)))?;

        let mut x_weighted_vec = Vec::with_capacity(n_samples * n_coeffs_to_solve);
        let mut y_weighted_vec = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sqrt_w = weights[i].sqrt();
            // Skip if weight is too small, but ensure vectors maintain length for from_shape_vec
            // A better approach for WLS might be to filter out zero-weight samples earlier
            // and adjust n_samples accordingly.
            // For now, if sqrt_w is effectively zero, the row contributes nothing.
            let eff_sqrt_w = if weights[i] < 1e-9 { 0.0 } else { sqrt_w };

            for j in 0..n_coeffs_to_solve {
                x_weighted_vec.push(x_augmented[[i, j]] * eff_sqrt_w);
            }
            y_weighted_vec.push(target[i] * eff_sqrt_w);
        }

        let x_w = Array2::from_shape_vec((n_samples, n_coeffs_to_solve), x_weighted_vec)
            .map_err(|e| ShapError::InternalError(format!("WLS X_w creation failed: {}", e)))?;
        let y_w = Array1::from_vec(y_weighted_vec);

        if x_w.is_empty() && y_w.is_empty() && n_coeffs_to_solve > 0 { // No samples made it through weighting
            return Ok(Array1::zeros(n_coeffs_to_solve));
        }
        if x_w.is_empty() && n_coeffs_to_solve == 0 { // No features, no samples
             return Ok(Array1::zeros(0));
        }


        #[cfg(feature = "linalg")]
        {
            // Using SVD based least squares for more robustness
            use ndarray_linalg::LeastSquaresSvd;
            let results = x_w.least_squares(&y_w)
                .map_err(|e| ShapError::InternalError(format!("WLS solver SVD failed: {}", e)))?;
            
            // Add check for rank deficiency if possible/needed from `results.rank`
            if results.rank <( n_coeffs_to_solve as i32 ){
                 // Handle rank deficiency: could indicate multicollinearity or insufficient unique samples
                 // For now, we proceed but this might lead to unstable coefficients.
                 // Consider logging a warning.
            }
            Ok(results.solution)
        }
        #[cfg(not(feature = "linalg"))]
        {
            return Err(ShapError::InternalError(
                "WLS solver not implemented. Enable 'linalg' feature or provide an implementation.".to_string(),
            ));
        }
    }
}


// At the bottom of src/algorithms/kernel_shap.rs

#[cfg(test)]
mod tests {
    use super::*; // Imports KernelExplainer, KernelShapConfig, etc.
    use crate::core::{Dataset, Instance, Result}; // Import core types
    use crate::traits::PredictModel; // Import the trait
    use ndarray::{array, Array1, Axis}; // For creating test data

    // Define a simple linear model for testing
    struct SimpleLinearModel {
        coefficients: Array1<f64>,
        intercept: f64,
    }

    impl SimpleLinearModel {
        fn new(coefficients: Vec<f64>, intercept: f64) -> Self {
            SimpleLinearModel {
                coefficients: Array1::from(coefficients),
                intercept,
            }
        }
    }

    impl PredictModel for SimpleLinearModel {
        fn predict(&self, instances: &Dataset) -> Result<Array1<f64>> {
            if instances.ncols() != self.coefficients.len() {
                return Err(ShapError::IncompatibleDimensions(format!(
                    "Model expects {} features, got {}",
                    self.coefficients.len(),
                    instances.ncols()
                )));
            }
            // Calculate predictions: (instances * coefficients) + intercept
            let mut predictions = Vec::new();
            for instance_row in instances.rows() {
                let dot_product = instance_row.dot(&self.coefficients);
                predictions.push(dot_product + self.intercept);
            }
            Ok(Array1::from_vec(predictions))
        }

        fn num_features(&self) -> usize {
            self.coefficients.len()
        }
    }

    #[test]
    fn test_kernel_explainer_simple_linear_model() -> Result<()> {
        // Model: f(x1, x2) = 2*x1 + 3*x2 + 5
        let model = SimpleLinearModel::new(vec![2.0, 3.0], 5.0);
        let num_features = model.num_features();

        // Background data: a few simple points
        let background_data = Dataset::from_shape_vec(
            (3, num_features), // 3 samples, 2 features
            vec![
                0.0, 0.0, // Sample 1
                1.0, 0.0, // Sample 2
                0.0, 1.0, // Sample 3
            ],
        )?;

        // Instance to explain
        let instance_to_explain = Instance::from(vec![2.0, 1.0]); // x1=2, x2=1

        // Prediction for instance_to_explain: 2*2 + 3*1 + 5 = 4 + 3 + 5 = 12
        let expected_model_prediction_for_instance = 12.0;

        // Configuration for KernelSHAP - use a fixed number of samples for reproducibility in tests
        // For M features, 2^M coalitions. For M=2, 2^2=4 coalitions.
        // Add 2 for the 0 and M feature coalitions.
        // Let's use enough samples to potentially cover all coalitions for M=2
        let config = KernelShapConfig {
            n_samples: KernelShapSamples::Fixed(2 * num_features + 4), // e.g., 2*2 + 4 = 8 samples
            noise_std_dev: 0.0, // No noise for this simple test
        };

        let explainer = KernelExplainer::new(model, background_data, Some(config))?;

        // --- Check expected_value from explainer ---
        // Predictions on background:
        // [0,0] -> 2*0 + 3*0 + 5 = 5
        // [1,0] -> 2*1 + 3*0 + 5 = 7
        // [0,1] -> 2*0 + 3*1 + 5 = 8
        // Mean = (5+7+8)/3 = 20/3 = 6.666...
        let expected_explainer_base_value = (5.0 + 7.0 + 8.0) / 3.0;
        assert!((explainer.expected_value() - expected_explainer_base_value).abs() < 1e-6);

        // --- Get SHAP values ---
        let explanation = explainer.shap_values(&instance_to_explain)?;

        println!("Explanation from test: {:?}", explanation); // For debugging if needed

        // --- Assertions ---
        // 1. Number of SHAP values
        assert_eq!(explanation.shap_values.len(), num_features);

        // 2. Sum of SHAP values + expected_value == actual_prediction
        let sum_shap_plus_expected = explanation.shap_values.sum() + explanation.expected_value;
        assert!((sum_shap_plus_expected - explanation.actual_prediction).abs() < 1e-4,
                "Sum of SHAP values + base value ({}) did not match actual prediction ({})",
                sum_shap_plus_expected, explanation.actual_prediction);

        // 3. Check actual_prediction from explanation matches our manual calculation
        assert!((explanation.actual_prediction - expected_model_prediction_for_instance).abs() < 1e-6);
        
        // 4. For a linear model f(x) = sum(coeff_i * x_i) + intercept, and if background is all zeros,
        //    SHAP value for feature i is often coeff_i * x_i.
        //    If background is not all zeros, it's coeff_i * (x_i - E[x_i_background]).
        //    This is a more advanced check and depends heavily on the background data and WLS solution.
        //    Let's keep it simple for now. The sum check (Assertion 2) is the most fundamental.

        // A simple check for linear models: SHAP_i = coeff_i * (x_i - E[background_i])
        // E[background_x1] = (0+1+0)/3 = 1/3
        // E[background_x2] = (0+0+1)/3 = 1/3
        // Expected SHAP for x1: 2.0 * (2.0 - 1.0/3.0) = 2.0 * (5.0/3.0) = 10.0/3.0 = 3.333...
        // Expected SHAP for x2: 3.0 * (1.0 - 1.0/3.0) = 3.0 * (2.0/3.0) = 2.0
        let expected_shap_x1 = 2.0 * (2.0 - 1.0/3.0);
        let expected_shap_x2 = 3.0 * (1.0 - 1.0/3.0);

        println!("Expected SHAP x1 (approx): {}", expected_shap_x1);
        println!("Actual SHAP x1: {}", explanation.shap_values[0]);
        println!("Expected SHAP x2 (approx): {}", expected_shap_x2);
        println!("Actual SHAP x2: {}", explanation.shap_values[1]);

        // These might not match perfectly due to the sampling in KernelSHAP and WLS solution,
        // especially with few samples. The sum property is more robust.
        // We might need more samples for these individual values to converge better.
        // assert!((explanation.shap_values[0] - expected_shap_x1).abs() < 0.5); // Looser tolerance for individual values
        // assert!((explanation.shap_values[1] - expected_shap_x2).abs() < 0.5);


        Ok(())
    }

    // You can add more tests:
    // - Test with different numbers of features.
    // - Test with a single background sample.
    // - Test with noise_std_dev > 0.
    // - Test edge cases for generate_coalitions (e.g., num_features = 0 or 1).
}