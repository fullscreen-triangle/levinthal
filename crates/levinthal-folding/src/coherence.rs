//! Coherence Metrics
//!
//! Measures of phase synchronization and ordering in oscillator networks.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Order parameter measurements.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrderParameter {
    /// Magnitude r ∈ [0, 1]
    pub r: f64,
    /// Mean phase ψ ∈ [-π, π]
    pub psi: f64,
}

impl OrderParameter {
    /// Create from (r, ψ) values.
    pub fn new(r: f64, psi: f64) -> Self {
        Self { r, psi }
    }

    /// Calculate from phase array.
    pub fn from_phases(phases: &[f64]) -> Self {
        let n = phases.len() as f64;
        if n == 0.0 {
            return Self::new(0.0, 0.0);
        }

        let sum_cos: f64 = phases.iter().map(|&p| p.cos()).sum();
        let sum_sin: f64 = phases.iter().map(|&p| p.sin()).sum();

        let r = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
        let psi = (sum_sin / n).atan2(sum_cos / n);

        Self::new(r, psi)
    }

    /// Check if system is coherent (above threshold).
    pub fn is_coherent(&self, threshold: f64) -> bool {
        self.r >= threshold
    }

    /// Check if system is fully synchronized.
    pub fn is_synchronized(&self, epsilon: f64) -> bool {
        self.r >= 1.0 - epsilon
    }
}

impl std::fmt::Display for OrderParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r={:.4}, ψ={:.4}", self.r, self.psi)
    }
}

/// Collection of coherence metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    /// Time series of order parameter magnitude.
    r_values: Vec<f64>,
    /// Time series of mean phase.
    psi_values: Vec<f64>,
}

impl CoherenceMetrics {
    /// Create new metrics collector.
    pub fn new() -> Self {
        Self {
            r_values: Vec::new(),
            psi_values: Vec::new(),
        }
    }

    /// Record a measurement.
    pub fn record(&mut self, order_param: OrderParameter) {
        self.r_values.push(order_param.r);
        self.psi_values.push(order_param.psi);
    }

    /// Record from phases.
    pub fn record_phases(&mut self, phases: &[f64]) {
        self.record(OrderParameter::from_phases(phases));
    }

    /// Get number of measurements.
    pub fn len(&self) -> usize {
        self.r_values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.r_values.is_empty()
    }

    /// Get r time series.
    pub fn r_values(&self) -> &[f64] {
        &self.r_values
    }

    /// Get psi time series.
    pub fn psi_values(&self) -> &[f64] {
        &self.psi_values
    }

    /// Calculate mean coherence.
    pub fn mean_coherence(&self) -> f64 {
        if self.r_values.is_empty() {
            return 0.0;
        }
        self.r_values.iter().sum::<f64>() / self.r_values.len() as f64
    }

    /// Calculate maximum coherence.
    pub fn max_coherence(&self) -> f64 {
        self.r_values.iter().cloned().fold(0.0, f64::max)
    }

    /// Calculate minimum coherence.
    pub fn min_coherence(&self) -> f64 {
        self.r_values.iter().cloned().fold(1.0, f64::min)
    }

    /// Calculate coherence variance.
    pub fn coherence_variance(&self) -> f64 {
        if self.r_values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_coherence();
        let sum_sq: f64 = self.r_values.iter().map(|&r| (r - mean).powi(2)).sum();
        sum_sq / (self.r_values.len() - 1) as f64
    }

    /// Calculate coherence standard deviation.
    pub fn coherence_std(&self) -> f64 {
        self.coherence_variance().sqrt()
    }

    /// Find time to first coherence above threshold.
    pub fn time_to_coherence(&self, threshold: f64) -> Option<usize> {
        self.r_values.iter().position(|&r| r >= threshold)
    }

    /// Calculate phase velocity (rate of mean phase change).
    pub fn phase_velocity(&self) -> f64 {
        if self.psi_values.len() < 2 {
            return 0.0;
        }

        // Handle phase wrapping
        let mut total_change = 0.0;
        for i in 1..self.psi_values.len() {
            let mut delta = self.psi_values[i] - self.psi_values[i - 1];
            // Unwrap phase
            while delta > PI {
                delta -= 2.0 * PI;
            }
            while delta < -PI {
                delta += 2.0 * PI;
            }
            total_change += delta;
        }

        total_change / (self.psi_values.len() - 1) as f64
    }

    /// Check if system achieved sustained coherence.
    ///
    /// Returns true if coherence was above threshold for at least `min_duration` samples.
    pub fn achieved_sustained_coherence(&self, threshold: f64, min_duration: usize) -> bool {
        let mut consecutive = 0;
        for &r in &self.r_values {
            if r >= threshold {
                consecutive += 1;
                if consecutive >= min_duration {
                    return true;
                }
            } else {
                consecutive = 0;
            }
        }
        false
    }
}

impl Default for CoherenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate pairwise phase coherence between two oscillators.
pub fn pairwise_coherence(phase_i: f64, phase_j: f64) -> f64 {
    (phase_j - phase_i).cos()
}

/// Calculate global pairwise coherence matrix.
pub fn coherence_matrix(phases: &[f64]) -> Vec<Vec<f64>> {
    let n = phases.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = pairwise_coherence(phases[i], phases[j]);
        }
    }

    matrix
}

/// Calculate phase locking value (PLV) between two phase time series.
///
/// PLV = |⟨e^{i(θ₁(t) - θ₂(t))}⟩|
pub fn phase_locking_value(phases1: &[f64], phases2: &[f64]) -> f64 {
    if phases1.len() != phases2.len() || phases1.is_empty() {
        return 0.0;
    }

    let n = phases1.len() as f64;
    let (sum_cos, sum_sin): (f64, f64) = phases1
        .iter()
        .zip(phases2.iter())
        .map(|(&p1, &p2)| {
            let diff = p1 - p2;
            (diff.cos(), diff.sin())
        })
        .fold((0.0, 0.0), |(sc, ss), (c, s)| (sc + c, ss + s));

    ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_order_parameter_synchronized() {
        let phases = vec![0.0, 0.0, 0.0, 0.0];
        let op = OrderParameter::from_phases(&phases);
        assert_relative_eq!(op.r, 1.0);
    }

    #[test]
    fn test_order_parameter_anti_phase() {
        let phases = vec![0.0, PI, 0.0, PI];
        let op = OrderParameter::from_phases(&phases);
        assert_relative_eq!(op.r, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coherence_metrics() {
        let mut metrics = CoherenceMetrics::new();
        metrics.record(OrderParameter::new(0.5, 0.0));
        metrics.record(OrderParameter::new(0.7, 0.1));
        metrics.record(OrderParameter::new(0.9, 0.2));

        assert_eq!(metrics.len(), 3);
        assert_relative_eq!(metrics.mean_coherence(), 0.7, epsilon = 1e-10);
        assert_relative_eq!(metrics.max_coherence(), 0.9);
    }

    #[test]
    fn test_plv_synchronized() {
        let phases1 = vec![0.0, 0.1, 0.2, 0.3];
        let phases2 = vec![0.0, 0.1, 0.2, 0.3];
        let plv = phase_locking_value(&phases1, &phases2);
        assert_relative_eq!(plv, 1.0, epsilon = 1e-10);
    }
}
