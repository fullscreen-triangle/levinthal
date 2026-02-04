//! GroEL Resonance Chamber Simulation
//!
//! Models the GroEL chaperonin as a resonance chamber that facilitates
//! protein folding through ATP-driven frequency modulation.
//!
//! ## Mechanism
//!
//! 1. **Cavity Geometry**: The GroEL barrel creates a confined space
//!    that selects for coherent states
//!
//! 2. **ATP Hydrolysis**: 7 ATP molecules hydrolyze in coordinated cycles,
//!    creating frequency modulation ±15% around 13.2 THz
//!
//! 3. **Phase Selection**: Incoherent states are destabilized; only
//!    coherent states persist through multiple cycles
//!
//! 4. **Trajectory Completion**: The chamber doesn't "compute" the answer;
//!    it creates conditions where only the correct trajectory is stable

use serde::{Deserialize, Serialize};
use crate::kuramoto::KuramotoNetwork;
use crate::coherence::{OrderParameter, CoherenceMetrics};

/// GroEL resonance frequency (THz).
pub const GROEL_FREQUENCY_THZ: f64 = 13.2;

/// ATP-induced frequency modulation (±15%).
pub const ATP_MODULATION_FRACTION: f64 = 0.15;

/// Number of ATP molecules in GroEL ring.
pub const ATP_COUNT: usize = 7;

/// Coherence threshold for native state.
pub const NATIVE_COHERENCE_THRESHOLD: f64 = 0.8;

/// GroEL chamber simulation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroELChamber {
    /// Oscillator network representing the protein.
    network: KuramotoNetwork,
    /// Base frequency (rad/s).
    base_frequency: f64,
    /// Current ATP cycle phase.
    atp_phase: f64,
    /// Number of completed ATP cycles.
    cycles_completed: usize,
    /// Coherence threshold for folding.
    coherence_threshold: f64,
}

impl GroELChamber {
    /// Create a new GroEL chamber simulation.
    ///
    /// # Arguments
    ///
    /// * `n_residues` - Number of residues (oscillators) in the protein
    /// * `coupling` - Coupling strength between residues
    pub fn new(n_residues: usize, coupling: f64) -> Self {
        // Convert THz to rad/s: ω = 2π × f
        let base_frequency = 2.0 * std::f64::consts::PI * GROEL_FREQUENCY_THZ * 1e12;

        // Create network with distributed frequencies around base
        let mut rng = rand::thread_rng();
        let network = KuramotoNetwork::new_distributed(
            &mut rng,
            n_residues,
            base_frequency,
            base_frequency * 0.1, // 10% initial spread
            coupling,
        );

        Self {
            network,
            base_frequency,
            atp_phase: 0.0,
            cycles_completed: 0,
            coherence_threshold: NATIVE_COHERENCE_THRESHOLD,
        }
    }

    /// Create with chain coupling (backbone connectivity).
    pub fn with_backbone_coupling(n_residues: usize, coupling: f64) -> Self {
        let base_frequency = 2.0 * std::f64::consts::PI * GROEL_FREQUENCY_THZ * 1e12;

        let network = KuramotoNetwork::chain_coupling(n_residues, base_frequency, coupling);

        Self {
            network,
            base_frequency,
            atp_phase: 0.0,
            cycles_completed: 0,
            coherence_threshold: NATIVE_COHERENCE_THRESHOLD,
        }
    }

    /// Set coherence threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.coherence_threshold = threshold;
        self
    }

    /// Get current coherence.
    pub fn coherence(&self) -> f64 {
        self.network.coherence()
    }

    /// Get order parameter.
    pub fn order_parameter(&self) -> OrderParameter {
        let (r, psi) = self.network.order_parameter();
        OrderParameter::new(r, psi)
    }

    /// Check if protein is in native state.
    pub fn is_folded(&self) -> bool {
        self.coherence() >= self.coherence_threshold
    }

    /// Get number of completed ATP cycles.
    pub fn cycles(&self) -> usize {
        self.cycles_completed
    }

    /// Calculate current ATP-modulated coupling.
    ///
    /// Coupling varies with ATP phase: K(t) = K₀(1 + ε·sin(ω_ATP·t))
    fn modulated_coupling(&self, base_coupling: f64) -> f64 {
        base_coupling * (1.0 + ATP_MODULATION_FRACTION * self.atp_phase.sin())
    }

    /// Advance simulation by one time step.
    ///
    /// This includes ATP-driven frequency modulation.
    pub fn step(&mut self, dt: f64) {
        // Update ATP phase (7 cycles per full GroEL cycle)
        let atp_frequency = self.base_frequency / 1e3; // Slower ATP cycle
        self.atp_phase += atp_frequency * dt * (ATP_COUNT as f64);

        // Track ATP cycles
        let new_cycles = (self.atp_phase / (2.0 * std::f64::consts::PI)) as usize;
        if new_cycles > self.cycles_completed {
            self.cycles_completed = new_cycles;
        }

        // Modulate coupling based on ATP phase
        let base_coupling = self.network.coupling_strength();
        let modulated = self.modulated_coupling(base_coupling);
        self.network.set_coupling_strength(modulated);

        // Evolve network
        self.network.step(dt);

        // Restore base coupling
        self.network.set_coupling_strength(base_coupling);
    }

    /// Run simulation until folded or max cycles reached.
    ///
    /// Returns (folded, coherence, cycles).
    pub fn fold(
        &mut self,
        dt: f64,
        steps_per_cycle: usize,
        max_cycles: usize,
    ) -> (bool, f64, usize) {
        for _ in 0..max_cycles {
            for _ in 0..steps_per_cycle {
                self.step(dt);
            }

            if self.is_folded() {
                return (true, self.coherence(), self.cycles_completed);
            }
        }

        (false, self.coherence(), self.cycles_completed)
    }

    /// Run simulation and record metrics.
    pub fn fold_with_metrics(
        &mut self,
        dt: f64,
        total_steps: usize,
    ) -> (CoherenceMetrics, bool) {
        let mut metrics = CoherenceMetrics::new();

        for _ in 0..total_steps {
            metrics.record(self.order_parameter());
            self.step(dt);

            if self.is_folded() {
                // Record final state
                metrics.record(self.order_parameter());
                return (metrics, true);
            }
        }

        (metrics, false)
    }

    /// Get underlying network.
    pub fn network(&self) -> &KuramotoNetwork {
        &self.network
    }

    /// Get mutable network.
    pub fn network_mut(&mut self) -> &mut KuramotoNetwork {
        &mut self.network
    }

    /// Reset simulation to random state.
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        self.network.randomize_phases(&mut rng);
        self.atp_phase = 0.0;
        self.cycles_completed = 0;
    }
}

/// Simulate folding success rate.
///
/// Returns (success_rate, mean_cycles_to_fold).
pub fn folding_success_rate(
    n_residues: usize,
    coupling: f64,
    n_trials: usize,
    dt: f64,
    max_steps: usize,
) -> (f64, f64) {
    let mut successes = 0;
    let mut total_cycles = 0;

    for _ in 0..n_trials {
        let mut chamber = GroELChamber::new(n_residues, coupling);
        let (folded, _, cycles) = chamber.fold(dt, 100, max_steps);

        if folded {
            successes += 1;
            total_cycles += cycles;
        }
    }

    let success_rate = successes as f64 / n_trials as f64;
    let mean_cycles = if successes > 0 {
        total_cycles as f64 / successes as f64
    } else {
        f64::INFINITY
    };

    (success_rate, mean_cycles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chamber_creation() {
        let chamber = GroELChamber::new(10, 1.0);
        assert!(!chamber.is_folded()); // Random initial state
    }

    #[test]
    fn test_atp_modulation() {
        let mut chamber = GroELChamber::new(5, 1.0);

        // Coupling should vary with ATP phase
        let c1 = chamber.modulated_coupling(1.0);
        chamber.atp_phase = std::f64::consts::PI / 2.0;
        let c2 = chamber.modulated_coupling(1.0);

        assert!(c1 != c2);
    }

    #[test]
    fn test_folding_simulation() {
        let mut chamber = GroELChamber::new(5, 10.0); // Strong coupling

        // High coupling should lead to synchronization
        let (metrics, _folded) = chamber.fold_with_metrics(0.001, 1000);

        assert!(!metrics.is_empty());
        // Coherence should increase over time with strong coupling
        let final_coherence = *metrics.r_values().last().unwrap();
        assert!(final_coherence > 0.1);
    }
}
