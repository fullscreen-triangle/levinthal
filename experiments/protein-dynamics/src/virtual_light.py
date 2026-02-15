#!/usr/bin/env python3
"""
Virtual Light: Ternary State Dynamics for Protein Atoms

Implements the virtual light concept from the self-observation framework:
- Atoms in ground (0), natural (1), or excited (2) states
- Absorption beam: 0 → 1, 2 (ground absorbs energy)
- Emission beam: 2, 1 → 0 (excited emits energy)

The protein atoms probe their environment through state transitions,
generating "virtual light" without physical photon exchange.

Author: Kundai Farai Sachikonye
Date: February 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

try:
    from .pdb_loader import ProteinStructure, Atom
except ImportError:
    from pdb_loader import ProteinStructure, Atom


# =============================================================================
# Constants
# =============================================================================

# Physical constants
KB = 1.381e-23      # Boltzmann constant (J/K)
EPSILON_0 = 8.854e-12  # Vacuum permittivity (F/m)
E_CHARGE = 1.602e-19   # Elementary charge (C)

# Threshold fields for state transitions (V/m)
E_GROUND_THRESHOLD = 1e5   # Below this → ground state
E_EXCITED_THRESHOLD = 5e5  # Above this → excited state

# Transition probabilities per timestep
P_GROUND_TO_NATURAL = 0.10   # 0 → 1
P_NATURAL_TO_GROUND = 0.05   # 1 → 0
P_NATURAL_TO_EXCITED = 0.05  # 1 → 2
P_EXCITED_TO_NATURAL = 0.10  # 2 → 1


# =============================================================================
# Virtual Light Data Classes
# =============================================================================

@dataclass
class AtomSpectrometer:
    """
    An atom functioning as a ternary spectrometer.

    Probes local environment through state transitions:
    - 0 (ground): Low energy, buried/shielded
    - 1 (natural): Equilibrium state
    - 2 (excited): High energy, exposed
    """
    atom: Atom
    state: int = 1  # 0, 1, or 2

    # S-entropy coordinates
    s_k: float = 1.0  # Knowledge entropy
    s_t: float = 0.5  # Temporal entropy
    s_e: float = 0.0  # Evolution entropy

    # Local field (cached)
    local_field: float = 0.0

    # Transition history
    absorption_count: int = 0
    emission_count: int = 0

    @property
    def is_absorbing(self) -> bool:
        """Atom in ground state can absorb."""
        return self.state == 0

    @property
    def is_emitting(self) -> bool:
        """Atom in excited state can emit."""
        return self.state == 2

    @property
    def is_anomalous(self) -> bool:
        """Not in natural state = anomalous."""
        return self.state != 1


@dataclass
class VirtualBeam:
    """
    Represents a virtual light beam from atomic state transitions.
    """
    beam_type: str  # "absorption" or "emission"
    atoms: List[AtomSpectrometer]
    intensity: float
    centroid: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class StateDistribution:
    """Distribution of ternary states across atoms."""
    ground: int = 0    # State 0
    natural: int = 0   # State 1
    excited: int = 0   # State 2

    @property
    def total(self) -> int:
        return self.ground + self.natural + self.excited

    @property
    def fractions(self) -> Tuple[float, float, float]:
        t = self.total
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.ground / t, self.natural / t, self.excited / t)

    def chi_squared(self, expected: Tuple[float, float, float]) -> float:
        """Compute chi-squared against expected fractions."""
        obs = [self.ground, self.natural, self.excited]
        exp = [e * self.total for e in expected]
        chi2 = 0.0
        for o, e in zip(obs, exp):
            if e > 0:
                chi2 += (o - e) ** 2 / e
        return chi2


# =============================================================================
# Electric Field Computation
# =============================================================================

def compute_local_field(position: np.ndarray,
                        protein: ProteinStructure,
                        epsilon_r: float = 4.0) -> float:
    """
    Compute local electric field at a position from nearby charged atoms.

    Args:
        position: 3D position in Angstroms
        protein: Protein structure
        epsilon_r: Relative permittivity (protein interior ~4)

    Returns:
        Electric field magnitude in V/m
    """
    E_total = np.zeros(3)

    # Charged residue types
    charged_residues = {
        'ASP': -1, 'GLU': -1,  # Negative
        'LYS': +1, 'ARG': +1,  # Positive
        'HIS': +0.5  # Partial positive at neutral pH
    }

    for atom in protein.atoms:
        # Check if charged
        charge = charged_residues.get(atom.residue_name, 0)
        if charge == 0:
            continue

        # Only count charge-carrying atoms
        if atom.name.strip() not in ['CG', 'CD', 'NZ', 'NH1', 'NH2', 'ND1', 'NE2']:
            continue

        # Distance vector (convert to meters)
        r_vec = (position - atom.position) * 1e-10  # Å to m
        r = np.linalg.norm(r_vec)

        if r < 1e-12:  # Avoid singularity
            continue

        # Coulomb field
        k = 8.99e9  # N·m²/C²
        E_mag = k * charge * E_CHARGE / (epsilon_r * r ** 2)
        E_total += E_mag * r_vec / r

    return np.linalg.norm(E_total)


def compute_field_from_binding(position: np.ndarray,
                               binding_center: np.ndarray,
                               binding_charge: float = -1.0) -> float:
    """
    Compute field from a binding event (e.g., ligand).

    Args:
        position: Atom position
        binding_center: Center of binding site/ligand
        binding_charge: Effective charge of ligand

    Returns:
        Additional field magnitude
    """
    r_vec = (position - binding_center) * 1e-10
    r = np.linalg.norm(r_vec)

    if r < 1e-12:
        return 0.0

    k = 8.99e9
    return abs(k * binding_charge * E_CHARGE / (4.0 * r ** 2))


# =============================================================================
# Virtual Light Engine
# =============================================================================

class VirtualLightEngine:
    """
    Engine for simulating virtual light from protein atoms.

    Implements the atoms-as-spectrometers paradigm where:
    - Each atom has ternary state (0, 1, 2)
    - State transitions generate virtual absorption/emission
    - Counting anomalies identify active regions
    """

    def __init__(self, protein: ProteinStructure):
        self.protein = protein
        self.spectrometers: Dict[int, AtomSpectrometer] = {}

        # Initialize spectrometers for all atoms
        for atom in protein.atoms:
            self.spectrometers[atom.serial] = AtomSpectrometer(atom=atom)

        # Expected equilibrium distribution
        self.expected_distribution = (0.1, 0.8, 0.1)  # ground, natural, excited

        # History
        self.state_history: List[StateDistribution] = []
        self.beam_history: List[Tuple[VirtualBeam, VirtualBeam]] = []

    def initialize_states(self, temperature: float = 300.0):
        """
        Initialize atomic states based on local environment.

        Uses protein structure to compute local fields and assign states.
        """
        print(f"  Initializing {len(self.spectrometers)} atom spectrometers...")

        for serial, spec in self.spectrometers.items():
            # Compute local field
            spec.local_field = compute_local_field(
                spec.atom.position, self.protein
            )

            # Assign initial state based on field
            spec.state = self._field_to_state(spec.local_field, temperature)

        # Report initial distribution
        dist = self.get_state_distribution()
        print(f"  Initial distribution: ground={dist.ground}, "
              f"natural={dist.natural}, excited={dist.excited}")

    def _field_to_state(self, field: float, temperature: float) -> int:
        """Convert local field to ternary state."""
        # Thermal energy scale
        kT = KB * temperature

        # Field thresholds (with thermal fluctuations)
        noise = np.random.normal(0, 0.1)

        if field < E_GROUND_THRESHOLD * (1 + noise):
            return 0  # Ground (shielded)
        elif field > E_EXCITED_THRESHOLD * (1 + noise):
            return 2  # Excited (exposed)
        else:
            return 1  # Natural

    def update_states(self,
                      perturbation: Optional[Callable[[np.ndarray], float]] = None):
        """
        Update atomic states based on transitions and perturbations.

        Args:
            perturbation: Optional function giving additional field at each position
        """
        for spec in self.spectrometers.values():
            # Apply perturbation if provided
            if perturbation is not None:
                additional_field = perturbation(spec.atom.position)
                total_field = spec.local_field + additional_field
            else:
                total_field = spec.local_field

            # State transitions (stochastic)
            old_state = spec.state

            if spec.state == 0:  # Ground
                if np.random.random() < P_GROUND_TO_NATURAL:
                    spec.state = 1
                    spec.absorption_count += 1

            elif spec.state == 1:  # Natural
                # High field → excited
                if total_field > E_EXCITED_THRESHOLD:
                    if np.random.random() < P_NATURAL_TO_EXCITED * 2:
                        spec.state = 2
                # Low field → ground
                elif total_field < E_GROUND_THRESHOLD:
                    if np.random.random() < P_NATURAL_TO_GROUND * 2:
                        spec.state = 0
                # Random transitions
                else:
                    r = np.random.random()
                    if r < P_NATURAL_TO_GROUND:
                        spec.state = 0
                    elif r < P_NATURAL_TO_GROUND + P_NATURAL_TO_EXCITED:
                        spec.state = 2

            elif spec.state == 2:  # Excited
                if np.random.random() < P_EXCITED_TO_NATURAL:
                    spec.state = 1
                    spec.emission_count += 1

    def get_state_distribution(self) -> StateDistribution:
        """Get current distribution of states."""
        dist = StateDistribution()
        for spec in self.spectrometers.values():
            if spec.state == 0:
                dist.ground += 1
            elif spec.state == 1:
                dist.natural += 1
            else:
                dist.excited += 1
        return dist

    def generate_virtual_beams(self) -> Tuple[VirtualBeam, VirtualBeam]:
        """
        Generate absorption and emission beams from current states.

        Returns:
            (absorption_beam, emission_beam)
        """
        absorbing = [s for s in self.spectrometers.values() if s.is_absorbing]
        emitting = [s for s in self.spectrometers.values() if s.is_emitting]

        total = len(self.spectrometers)

        # Absorption beam
        if absorbing:
            abs_centroid = np.mean([s.atom.position for s in absorbing], axis=0)
        else:
            abs_centroid = self.protein.center_of_mass()

        absorption_beam = VirtualBeam(
            beam_type="absorption",
            atoms=absorbing,
            intensity=len(absorbing) / total if total > 0 else 0,
            centroid=abs_centroid
        )

        # Emission beam
        if emitting:
            emi_centroid = np.mean([s.atom.position for s in emitting], axis=0)
        else:
            emi_centroid = self.protein.center_of_mass()

        emission_beam = VirtualBeam(
            beam_type="emission",
            atoms=emitting,
            intensity=len(emitting) / total if total > 0 else 0,
            centroid=emi_centroid
        )

        return absorption_beam, emission_beam

    def get_anomalous_atoms(self) -> List[AtomSpectrometer]:
        """Get atoms not in natural state (anomalies)."""
        return [s for s in self.spectrometers.values() if s.is_anomalous]

    def compute_trit(self) -> int:
        """
        Compute ternary digit from state distribution.

        trit = 0: More ground than excited (absorption dominant)
        trit = 1: Balanced (natural dominant)
        trit = 2: More excited than ground (emission dominant)
        """
        dist = self.get_state_distribution()

        if dist.ground > dist.excited:
            return 0
        elif dist.excited > dist.ground:
            return 2
        else:
            return 1

    def run_dynamics(self,
                     n_steps: int,
                     perturbation: Optional[Callable[[np.ndarray], float]] = None,
                     record_interval: int = 10) -> List[StateDistribution]:
        """
        Run state dynamics simulation.

        Args:
            n_steps: Number of timesteps
            perturbation: Optional perturbation field
            record_interval: How often to record state

        Returns:
            List of state distributions over time
        """
        history = []

        for step in range(n_steps):
            self.update_states(perturbation)

            if step % record_interval == 0:
                dist = self.get_state_distribution()
                history.append(dist)

                abs_beam, emi_beam = self.generate_virtual_beams()
                self.beam_history.append((abs_beam, emi_beam))

        self.state_history = history
        return history

    def get_active_region_centroid(self) -> np.ndarray:
        """Get centroid of anomalous (active) atoms."""
        anomalous = self.get_anomalous_atoms()
        if not anomalous:
            return self.protein.center_of_mass()
        return np.mean([s.atom.position for s in anomalous], axis=0)

    def identify_binding_site(self, threshold_chi2: float = 10.0) -> List[Atom]:
        """
        Identify potential binding site from counting anomalies.

        Atoms with high local chi-squared are likely in active region.
        """
        # Group atoms by residue and compute local chi-squared
        residue_atoms: Dict[str, List[AtomSpectrometer]] = {}

        for spec in self.spectrometers.values():
            res_id = spec.atom.residue_id
            if res_id not in residue_atoms:
                residue_atoms[res_id] = []
            residue_atoms[res_id].append(spec)

        # Find residues with anomalous distribution
        active_residues = []
        for res_id, specs in residue_atoms.items():
            if len(specs) < 3:
                continue

            # Local distribution
            local_dist = StateDistribution()
            for s in specs:
                if s.state == 0:
                    local_dist.ground += 1
                elif s.state == 1:
                    local_dist.natural += 1
                else:
                    local_dist.excited += 1

            chi2 = local_dist.chi_squared(self.expected_distribution)
            if chi2 > threshold_chi2:
                active_residues.extend([s.atom for s in specs])

        return active_residues


# =============================================================================
# Convenience Functions
# =============================================================================

def create_binding_perturbation(center: np.ndarray,
                                 strength: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Create a perturbation function simulating ligand binding.

    The perturbation creates a localized field enhancement near the binding site.
    """
    def perturbation(position: np.ndarray) -> float:
        r = np.linalg.norm(position - center)
        # Gaussian perturbation
        sigma = 5.0  # Angstroms
        return strength * E_EXCITED_THRESHOLD * np.exp(-r**2 / (2 * sigma**2))

    return perturbation


def create_helix_motion_perturbation(helix_atoms: List[Atom],
                                      direction: np.ndarray,
                                      amplitude: float = 2.0) -> Callable[[np.ndarray], float]:
    """
    Create perturbation from helix motion.

    Moving helix creates transient field changes along its path.
    """
    helix_center = np.mean([a.position for a in helix_atoms], axis=0)

    def perturbation(position: np.ndarray) -> float:
        # Distance to helix axis
        to_helix = position - helix_center
        # Component along motion direction
        along_motion = np.dot(to_helix, direction)

        # Field enhancement near moving helix
        r_perp = np.linalg.norm(to_helix - along_motion * direction)
        sigma = 3.0
        return amplitude * E_GROUND_THRESHOLD * np.exp(-r_perp**2 / (2 * sigma**2))

    return perturbation


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from .pdb_loader import load_protein

    # Load lysozyme
    print("Loading protein...")
    protein = load_protein("1LYZ", cache_dir=Path("data/pdb"))
    print(protein.summary())

    # Create virtual light engine
    print("\nInitializing virtual light engine...")
    engine = VirtualLightEngine(protein)
    engine.initialize_states()

    # Run dynamics
    print("\nRunning state dynamics...")
    history = engine.run_dynamics(n_steps=100)

    # Report
    print("\nFinal state distribution:")
    final = engine.get_state_distribution()
    print(f"  Ground: {final.ground} ({100*final.fractions[0]:.1f}%)")
    print(f"  Natural: {final.natural} ({100*final.fractions[1]:.1f}%)")
    print(f"  Excited: {final.excited} ({100*final.fractions[2]:.1f}%)")

    # Virtual beams
    abs_beam, emi_beam = engine.generate_virtual_beams()
    print(f"\nVirtual beams:")
    print(f"  Absorption: {abs_beam.n_atoms} atoms, intensity={abs_beam.intensity:.3f}")
    print(f"  Emission: {emi_beam.n_atoms} atoms, intensity={emi_beam.intensity:.3f}")
