"""
Kuramoto Coupled Oscillator Network (Equation V).

dφ_i/dt = ω_i + Σ K_ij sin(φ_j - φ_i)
Order parameter: r·e^(iψ) = (1/N) Σ e^(iφ_j)

Transliterated from crates/levinthal-folding/src/kuramoto.rs
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class KuramotoNetwork:
    """Coupled phase oscillator network for H-bond synchronization."""

    def __init__(self, n_oscillators: int, frequencies: np.ndarray,
                 coupling_matrix: Optional[np.ndarray] = None,
                 coupling_strength: float = 2.0):
        self.n = n_oscillators
        self.frequencies = frequencies.copy()
        self.coupling_strength = coupling_strength
        self.phases = np.random.uniform(0, 2 * np.pi, n_oscillators)

        if coupling_matrix is not None:
            self.coupling_matrix = coupling_matrix.copy()
        else:
            self.coupling_matrix = np.ones((n_oscillators, n_oscillators))
            np.fill_diagonal(self.coupling_matrix, 0)

    @classmethod
    def from_hbond_network(cls, n_hbonds: int, base_freq: float = 13.2e12,
                           freq_spread: float = 0.10,
                           coupling_K0: float = 3.0,
                           coupling_r0: float = 12.0,
                           seed: int = 42) -> 'KuramotoNetwork':
        """Create network from H-bond parameters with distance-based coupling.

        Frequencies are normalized to dimensionless units (relative to base_freq)
        so that coupling_K0 ~ O(1) is meaningful for synchronization.
        """
        rng = np.random.RandomState(seed)

        # Normalized frequencies: centered around 1.0 with spread
        frequencies = 1.0 + freq_spread * (rng.rand(n_hbonds) - 0.5)

        # Generate random 3D positions (H-bond donor-acceptor midpoints)
        positions = rng.randn(n_hbonds, 3) * 10.0  # ~10 Å spread

        # Distance-based coupling: K_ij = K0 * exp(-r_ij / r0)
        coupling_matrix = np.zeros((n_hbonds, n_hbonds))
        for i in range(n_hbonds):
            for j in range(n_hbonds):
                if i != j:
                    r_ij = np.linalg.norm(positions[i] - positions[j])
                    coupling_matrix[i, j] = coupling_K0 * np.exp(-r_ij / coupling_r0)

        network = cls(n_hbonds, frequencies, coupling_matrix, coupling_K0)
        network.phases = rng.uniform(0, 2 * np.pi, n_hbonds)
        network._positions = positions
        network._base_freq = base_freq  # Store for reference
        return network

    def order_parameter(self) -> Tuple[float, float]:
        """Compute order parameter r·e^(iψ) = (1/N) Σ e^(iφ_j)."""
        mean_cos = np.mean(np.cos(self.phases))
        mean_sin = np.mean(np.sin(self.phases))
        r = np.sqrt(mean_cos**2 + mean_sin**2)
        psi = np.arctan2(mean_sin, mean_cos)
        return r, psi

    def coherence(self) -> float:
        """Return magnitude of order parameter."""
        return self.order_parameter()[0]

    def _derivatives(self, phases: np.ndarray) -> np.ndarray:
        """Compute dφ_i/dt = ω_i + Σ K_ij sin(φ_j - φ_i). Vectorized."""
        dpdt = self.frequencies.copy()
        # phase_diffs[i,j] = phases[j] - phases[i]
        phase_diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
        sin_diffs = np.sin(phase_diffs)
        # coupling_sums[i] = Σ_j K_ij sin(φ_j - φ_i)
        coupling_sums = np.sum(self.coupling_matrix * sin_diffs, axis=1)
        neighbor_counts = np.maximum(1.0, np.sum(self.coupling_matrix > 0, axis=1).astype(float))
        dpdt += (self.coupling_strength / neighbor_counts) * coupling_sums
        return dpdt

    def step_rk4(self, dt: float):
        """Fourth-order Runge-Kutta integration step."""
        k1 = self._derivatives(self.phases)
        k2 = self._derivatives(self.phases + 0.5 * dt * k1)
        k3 = self._derivatives(self.phases + 0.5 * dt * k2)
        k4 = self._derivatives(self.phases + dt * k3)
        self.phases += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.phases = self.phases % (2 * np.pi)

    def evolve(self, dt: float, n_steps: int,
               record_every: int = 1) -> Dict:
        """Evolve network and record time series."""
        r_series = []
        psi_series = []
        phase_snapshots = []

        for step in range(n_steps):
            if step % record_every == 0:
                r, psi = self.order_parameter()
                r_series.append(r)
                psi_series.append(psi)
                phase_snapshots.append(self.phases.copy())
            self.step_rk4(dt)

        # Record final state
        r, psi = self.order_parameter()
        r_series.append(r)
        psi_series.append(psi)
        phase_snapshots.append(self.phases.copy())

        return {
            'r_series': np.array(r_series),
            'psi_series': np.array(psi_series),
            'phase_snapshots': phase_snapshots,
            'n_steps': n_steps,
            'dt': dt,
            'final_r': float(r),
            'final_psi': float(psi),
        }

    def set_coupling_decay(self, K0: float, r0: float = 5.0):
        """Set distance-dependent coupling K_ij = K0·exp(-r_ij/r0)."""
        if hasattr(self, '_positions'):
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        r_ij = np.linalg.norm(
                            self._positions[i] - self._positions[j]
                        )
                        self.coupling_matrix[i, j] = K0 * np.exp(-r_ij / r0)
        self.coupling_strength = K0

    def perturb_coupling(self, region_indices: List[int],
                         factor: float):
        """Reduce coupling for oscillators in region (mutation effect)."""
        for i in region_indices:
            self.coupling_matrix[i, :] *= factor
            self.coupling_matrix[:, i] *= factor

    def synchronize(self, phase: float = 0.0):
        """Set all oscillators to same phase."""
        self.phases[:] = phase

    def randomize(self, seed: int = None):
        """Reset to random phases."""
        rng = np.random.RandomState(seed)
        self.phases = rng.uniform(0, 2 * np.pi, self.n)


def simulate_phase_lock(n_hbonds: int = 165, n_steps: int = 3000,
                        dt: float = 0.01, seed: int = 42) -> Dict:
    """
    Simulate H-bond phase-locking during protein folding.
    Returns trajectory from random phases to coherent native state.
    """
    network = KuramotoNetwork.from_hbond_network(n_hbonds, seed=seed)

    # Record evolution
    result = network.evolve(dt, n_steps, record_every=10)

    # Find time to coherence threshold
    r_series = result['r_series']
    t_coherent = None
    for i, r in enumerate(r_series):
        if r > 0.8:
            t_coherent = i * 10 * dt
            break

    result['n_hbonds'] = n_hbonds
    result['time_to_coherence'] = t_coherent
    result['coherence_achieved'] = float(r_series[-1]) > 0.8

    return result


def run_phaselock_validation() -> Dict:
    """Run complete phase-lock validation for the paper."""
    np.random.seed(42)

    # Main simulation
    main_result = simulate_phase_lock(165, 5000, 1e-15, seed=42)

    # Coupling decay data for figure panel
    r0_values = [2.0, 5.0, 10.0, 20.0]
    decay_data = {}
    for r0 in r0_values:
        distances = np.linspace(0, 30, 100)
        coupling = 2.0 * np.exp(-distances / r0)
        decay_data[f'r0_{r0}'] = {
            'distances': distances.tolist(),
            'coupling': coupling.tolist(),
        }

    return {
        'phase_lock': {
            'r_series': main_result['r_series'].tolist(),
            'final_r': main_result['final_r'],
            'coherence_achieved': main_result['coherence_achieved'],
            'n_hbonds': 165,
        },
        'coupling_decay': decay_data,
        'n_steps': 5000,
    }
