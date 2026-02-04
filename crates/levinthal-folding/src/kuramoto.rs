//! Kuramoto Oscillator Model
//!
//! Implements coupled phase oscillators for modeling hydrogen bond
//! network synchronization in protein folding.
//!
//! ## Kuramoto Model
//!
//! The classic Kuramoto model:
//! ```text
//! dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
//! ```
//!
//! Where:
//! - θᵢ is the phase of oscillator i
//! - ωᵢ is the natural frequency
//! - K is the coupling strength
//! - N is the number of oscillators
//!
//! ## In Protein Folding
//!
//! - Each oscillator represents an H-bond
//! - Natural frequencies come from local environment
//! - Coupling reflects backbone connectivity
//! - Synchronization = native state formation

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use rand::Rng;

/// A single Kuramoto oscillator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KuramotoOscillator {
    /// Current phase [0, 2π)
    phase: f64,
    /// Natural frequency (rad/s)
    natural_frequency: f64,
}

impl KuramotoOscillator {
    /// Create a new oscillator.
    pub fn new(phase: f64, natural_frequency: f64) -> Self {
        Self {
            phase: phase % (2.0 * PI),
            natural_frequency,
        }
    }

    /// Create with random phase.
    pub fn random<R: Rng>(rng: &mut R, natural_frequency: f64) -> Self {
        Self::new(rng.gen::<f64>() * 2.0 * PI, natural_frequency)
    }

    /// Get current phase.
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Get natural frequency.
    pub fn natural_frequency(&self) -> f64 {
        self.natural_frequency
    }

    /// Set phase.
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase % (2.0 * PI);
        if self.phase < 0.0 {
            self.phase += 2.0 * PI;
        }
    }

    /// Update phase given a rate of change.
    pub fn advance(&mut self, d_phase: f64) {
        self.set_phase(self.phase + d_phase);
    }
}

/// A network of coupled Kuramoto oscillators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KuramotoNetwork {
    /// The oscillators
    oscillators: Vec<KuramotoOscillator>,
    /// Global coupling strength K
    coupling_strength: f64,
    /// Adjacency matrix (connectivity)
    /// If None, all-to-all coupling
    adjacency: Option<Vec<Vec<f64>>>,
}

impl KuramotoNetwork {
    /// Create a new network with uniform natural frequencies.
    pub fn new_uniform(n: usize, natural_frequency: f64, coupling_strength: f64) -> Self {
        let mut rng = rand::thread_rng();
        let oscillators = (0..n)
            .map(|_| KuramotoOscillator::random(&mut rng, natural_frequency))
            .collect();

        Self {
            oscillators,
            coupling_strength,
            adjacency: None,
        }
    }

    /// Create with frequency distribution.
    pub fn new_distributed<R: Rng>(
        rng: &mut R,
        n: usize,
        mean_frequency: f64,
        frequency_spread: f64,
        coupling_strength: f64,
    ) -> Self {
        let oscillators = (0..n)
            .map(|_| {
                let freq = mean_frequency + (rng.gen::<f64>() - 0.5) * frequency_spread;
                KuramotoOscillator::random(rng, freq)
            })
            .collect();

        Self {
            oscillators,
            coupling_strength,
            adjacency: None,
        }
    }

    /// Create with specific adjacency matrix.
    pub fn with_adjacency(mut self, adjacency: Vec<Vec<f64>>) -> Self {
        assert_eq!(adjacency.len(), self.oscillators.len());
        for row in &adjacency {
            assert_eq!(row.len(), self.oscillators.len());
        }
        self.adjacency = Some(adjacency);
        self
    }

    /// Create a chain (nearest-neighbor) coupling.
    pub fn chain_coupling(n: usize, natural_frequency: f64, coupling_strength: f64) -> Self {
        let mut network = Self::new_uniform(n, natural_frequency, coupling_strength);
        let mut adjacency = vec![vec![0.0; n]; n];

        for i in 0..n {
            if i > 0 {
                adjacency[i][i - 1] = 1.0;
            }
            if i < n - 1 {
                adjacency[i][i + 1] = 1.0;
            }
        }

        network.adjacency = Some(adjacency);
        network
    }

    /// Number of oscillators.
    pub fn len(&self) -> usize {
        self.oscillators.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.oscillators.is_empty()
    }

    /// Get oscillators.
    pub fn oscillators(&self) -> &[KuramotoOscillator] {
        &self.oscillators
    }

    /// Get coupling strength.
    pub fn coupling_strength(&self) -> f64 {
        self.coupling_strength
    }

    /// Set coupling strength.
    pub fn set_coupling_strength(&mut self, k: f64) {
        self.coupling_strength = k;
    }

    /// Get phases as vector.
    pub fn phases(&self) -> Vec<f64> {
        self.oscillators.iter().map(|o| o.phase()).collect()
    }

    /// Calculate the order parameter (complex).
    ///
    /// r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
    ///
    /// Returns (r, ψ) where:
    /// - r ∈ [0, 1] is the coherence
    /// - ψ is the mean phase
    pub fn order_parameter(&self) -> (f64, f64) {
        let n = self.oscillators.len() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }

        let (sum_cos, sum_sin): (f64, f64) = self
            .oscillators
            .iter()
            .map(|o| (o.phase().cos(), o.phase().sin()))
            .fold((0.0, 0.0), |(sc, ss), (c, s)| (sc + c, ss + s));

        let r = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
        let psi = (sum_sin / n).atan2(sum_cos / n);

        (r, psi)
    }

    /// Calculate coherence (magnitude of order parameter).
    pub fn coherence(&self) -> f64 {
        self.order_parameter().0
    }

    /// Calculate mean phase.
    pub fn mean_phase(&self) -> f64 {
        self.order_parameter().1
    }

    /// Evolve the network by one time step using Euler integration.
    pub fn step(&mut self, dt: f64) {
        let n = self.oscillators.len();
        if n == 0 {
            return;
        }

        let phases: Vec<f64> = self.phases();
        let n_f64 = n as f64;

        // Calculate phase derivatives
        let mut d_phases = vec![0.0; n];

        for i in 0..n {
            // Natural frequency contribution
            d_phases[i] = self.oscillators[i].natural_frequency();

            // Coupling contribution
            let coupling_sum: f64 = match &self.adjacency {
                Some(adj) => {
                    (0..n)
                        .map(|j| adj[i][j] * (phases[j] - phases[i]).sin())
                        .sum()
                }
                None => {
                    // All-to-all coupling
                    (0..n)
                        .map(|j| (phases[j] - phases[i]).sin())
                        .sum()
                }
            };

            let effective_n = match &self.adjacency {
                Some(adj) => adj[i].iter().sum::<f64>().max(1.0),
                None => n_f64,
            };

            d_phases[i] += (self.coupling_strength / effective_n) * coupling_sum;
        }

        // Update phases
        for (i, osc) in self.oscillators.iter_mut().enumerate() {
            osc.advance(d_phases[i] * dt);
        }
    }

    /// Evolve for multiple steps.
    pub fn evolve(&mut self, dt: f64, steps: usize) {
        for _ in 0..steps {
            self.step(dt);
        }
    }

    /// Evolve until coherence exceeds threshold or max steps reached.
    ///
    /// Returns (final_coherence, steps_taken).
    pub fn evolve_until_coherent(
        &mut self,
        dt: f64,
        coherence_threshold: f64,
        max_steps: usize,
    ) -> (f64, usize) {
        for step in 0..max_steps {
            let coherence = self.coherence();
            if coherence >= coherence_threshold {
                return (coherence, step);
            }
            self.step(dt);
        }
        (self.coherence(), max_steps)
    }

    /// Record coherence time series.
    pub fn coherence_trajectory(&mut self, dt: f64, steps: usize) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity(steps);
        for _ in 0..steps {
            trajectory.push(self.coherence());
            self.step(dt);
        }
        trajectory
    }

    /// Reset to random phases.
    pub fn randomize_phases<R: Rng>(&mut self, rng: &mut R) {
        for osc in &mut self.oscillators {
            osc.set_phase(rng.gen::<f64>() * 2.0 * PI);
        }
    }

    /// Set all oscillators to the same phase (synchronized state).
    pub fn synchronize(&mut self, phase: f64) {
        for osc in &mut self.oscillators {
            osc.set_phase(phase);
        }
    }
}

/// Calculate critical coupling strength for Kuramoto model.
///
/// For a Lorentzian frequency distribution with half-width γ:
/// Kc = 2γ
pub fn critical_coupling(frequency_spread: f64) -> f64 {
    2.0 * frequency_spread
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_oscillator_phase_wrap() {
        let mut osc = KuramotoOscillator::new(0.0, 1.0);
        osc.set_phase(3.0 * PI);
        assert!(osc.phase() >= 0.0 && osc.phase() < 2.0 * PI);
    }

    #[test]
    fn test_synchronized_order_parameter() {
        let mut network = KuramotoNetwork::new_uniform(10, 1.0, 1.0);
        network.synchronize(0.0);

        let (r, _) = network.order_parameter();
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_network_evolution() {
        let mut network = KuramotoNetwork::new_uniform(5, 1.0, 10.0);

        // Strong coupling should increase coherence
        let initial = network.coherence();
        network.evolve(0.01, 1000);
        let final_coherence = network.coherence();

        assert!(final_coherence >= initial - 0.1); // Should not decrease much
    }

    #[test]
    fn test_chain_coupling() {
        let network = KuramotoNetwork::chain_coupling(5, 1.0, 1.0);
        assert_eq!(network.len(), 5);
    }
}
