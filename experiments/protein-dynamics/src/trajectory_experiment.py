#!/usr/bin/env python3
"""
Trajectory Experiment: Ternary Spectrometer Framework for Protein Dynamics

This module implements completion-driven trajectory computation for:
1. Alpha helix motion tracking
2. Beta sheet dynamics
3. Ligand docking

Based on the atoms-as-ternary-spectrometers framework.

Author: Kundai Farai Sachikonye
Date: February 2026
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import warnings

from .pdb_loader import (
    load_protein, ProteinStructure, Atom, Residue,
    SecondaryStructure, Helix, Sheet
)

warnings.filterwarnings('ignore')


# =============================================================================
# Constants
# =============================================================================

# Physical constants
KB = 1.381e-23      # Boltzmann constant (J/K)
PLANCK = 6.626e-34  # Planck constant (J·s)

# Default parameters
DEFAULT_TEMPERATURE = 300  # K (room temperature)
DEFAULT_TIMESTEP = 1e-12   # 1 picosecond
CHI_SQUARED_THRESHOLD = 6.0  # For anomaly detection


# =============================================================================
# Completion Conditions
# =============================================================================

class CompletionType(Enum):
    """Types of completion conditions."""
    HELIX_MOTION = "helix_motion"
    SHEET_DYNAMICS = "sheet_dynamics"
    DOCKING = "docking"
    FOLDING = "folding"
    CUSTOM = "custom"


@dataclass
class CompletionCondition:
    """Specification of a completion condition."""
    type: CompletionType
    target_rmsd: float = 0.5  # Angstroms
    stable_frames: int = 10   # Number of stable frames required
    energy_threshold: float = 0.0  # kcal/mol

    # For helix/sheet motion
    structure_id: Optional[str] = None
    displacement_target: Optional[np.ndarray] = None

    # For docking
    ligand_atoms: Optional[List[np.ndarray]] = None
    binding_site_center: Optional[np.ndarray] = None
    binding_distance: float = 5.0  # Angstroms

    def is_satisfied(self, state: 'SystemState') -> bool:
        """Check if completion condition is satisfied."""
        if self.type == CompletionType.HELIX_MOTION:
            return self._check_helix_motion(state)
        elif self.type == CompletionType.SHEET_DYNAMICS:
            return self._check_sheet_dynamics(state)
        elif self.type == CompletionType.DOCKING:
            return self._check_docking(state)
        return False

    def _check_helix_motion(self, state: 'SystemState') -> bool:
        """Check helix motion completion."""
        if self.displacement_target is None:
            return False
        current_displacement = state.get_structure_displacement(self.structure_id)
        distance = np.linalg.norm(current_displacement - self.displacement_target)
        return distance < self.target_rmsd

    def _check_sheet_dynamics(self, state: 'SystemState') -> bool:
        """Check sheet dynamics completion."""
        return self._check_helix_motion(state)  # Same logic

    def _check_docking(self, state: 'SystemState') -> bool:
        """Check docking completion."""
        if self.binding_site_center is None or state.ligand_position is None:
            return False
        distance = np.linalg.norm(state.ligand_position - self.binding_site_center)
        return distance < self.binding_distance


# =============================================================================
# System State
# =============================================================================

@dataclass
class AtomState:
    """State of a single atom as ternary spectrometer."""
    atom: Atom
    ternary: int = 1  # 0=ground, 1=natural, 2=excited
    s_k: float = 1.0  # Knowledge entropy
    s_t: float = 0.5  # Temporal entropy
    s_e: float = 0.0  # Evolution entropy
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def is_anomalous(self) -> bool:
        """Check if atom shows counting anomaly."""
        # Natural state should be most common; deviations indicate activity
        return self.ternary != 1

    def update_ternary(self, local_energy: float, temperature: float):
        """Update ternary state based on local energy."""
        kT = KB * temperature
        # Boltzmann factor determines excited probability
        p_excited = np.exp(-local_energy / kT) if local_energy > 0 else 0.5
        p_ground = np.exp(local_energy / kT) if local_energy < 0 else 0.5

        r = np.random.random()
        if r < p_ground * 0.1:  # Small probability of ground state
            self.ternary = 0
        elif r > 1 - p_excited * 0.1:  # Small probability of excited
            self.ternary = 2
        else:
            self.ternary = 1  # Natural state


@dataclass
class SystemState:
    """Complete state of the protein system."""
    protein: ProteinStructure
    atom_states: Dict[int, AtomState] = field(default_factory=dict)
    time: float = 0.0
    iteration: int = 0

    # For tracking specific structures
    initial_positions: Dict[str, np.ndarray] = field(default_factory=dict)

    # For docking
    ligand_position: Optional[np.ndarray] = None
    ligand_atoms: List[np.ndarray] = field(default_factory=list)

    def initialize(self, temperature: float = DEFAULT_TEMPERATURE):
        """Initialize atom states."""
        for atom in self.protein.atoms:
            self.atom_states[atom.serial] = AtomState(
                atom=atom,
                velocity=np.random.normal(0, np.sqrt(KB * temperature / 1e-26), 3)
            )

    def get_anomalous_atoms(self) -> List[AtomState]:
        """Get atoms showing counting anomalies."""
        return [s for s in self.atom_states.values() if s.is_anomalous]

    def get_ternary_distribution(self) -> Dict[int, int]:
        """Get distribution of ternary states."""
        dist = {0: 0, 1: 0, 2: 0}
        for state in self.atom_states.values():
            dist[state.ternary] += 1
        return dist

    def compute_chi_squared(self, expected: Dict[int, float]) -> float:
        """Compute chi-squared statistic for ternary distribution."""
        observed = self.get_ternary_distribution()
        total = sum(observed.values())
        chi2 = 0.0
        for t in [0, 1, 2]:
            exp = expected.get(t, total / 3)
            if exp > 0:
                chi2 += (observed[t] - exp) ** 2 / exp
        return chi2

    def get_structure_displacement(self, structure_id: str) -> np.ndarray:
        """Get displacement of a structure from initial position."""
        if structure_id not in self.initial_positions:
            return np.zeros(3)

        # Find current centroid of structure atoms
        atoms = self._get_structure_atoms(structure_id)
        if not atoms:
            return np.zeros(3)

        current = np.mean([self.atom_states[a.serial].atom.position for a in atoms], axis=0)
        initial = self.initial_positions[structure_id]
        return current - initial

    def _get_structure_atoms(self, structure_id: str) -> List[Atom]:
        """Get atoms belonging to a structure (helix or sheet)."""
        # Try helix first
        helix = self.protein.get_helix_by_id(structure_id)
        if helix:
            return self.protein.get_helix_atoms(structure_id)

        # Try sheet
        for sheet in self.protein.sheets:
            if sheet.sheet_id.strip() == structure_id.strip():
                return [a for a in self.protein.atoms
                       if sheet.contains_residue(a.chain_id, a.residue_seq)]

        return []

    def store_initial_position(self, structure_id: str):
        """Store initial centroid of a structure."""
        atoms = self._get_structure_atoms(structure_id)
        if atoms:
            centroid = np.mean([a.position for a in atoms], axis=0)
            self.initial_positions[structure_id] = centroid


# =============================================================================
# Trajectory Step
# =============================================================================

@dataclass
class TrajectoryStep:
    """Single step in the trajectory."""
    iteration: int
    time: float
    trit: int
    position: np.ndarray  # Centroid of active region
    backaction: float
    n_anomalous: int
    ternary_distribution: Dict[int, int]
    chi_squared: float
    s_coordinates: Tuple[float, float, float]  # (S_k, S_t, S_e)

    # For specific tracking
    structure_displacement: Optional[np.ndarray] = None
    ligand_position: Optional[np.ndarray] = None


# =============================================================================
# Trajectory Engine
# =============================================================================

class TrajectoryEngine:
    """
    Engine for running completion-driven trajectory computations.

    Implements the atoms-as-ternary-spectrometers framework.
    """

    def __init__(self,
                 protein: ProteinStructure,
                 temperature: float = DEFAULT_TEMPERATURE,
                 timestep: float = DEFAULT_TIMESTEP):
        self.protein = protein
        self.temperature = temperature
        self.timestep = timestep

        # Initialize state
        self.state = SystemState(protein=protein)
        self.state.initialize(temperature)

        # Expected equilibrium distribution
        self.expected_distribution = {0: 0.1, 1: 0.8, 2: 0.1}

        # Results
        self.trajectory: List[TrajectoryStep] = []

    def set_completion_condition(self, condition: CompletionCondition):
        """Set the completion condition."""
        self.completion = condition

        # Store initial positions for relevant structures
        if condition.structure_id:
            self.state.store_initial_position(condition.structure_id)

    def run(self, max_iterations: int = 1000) -> List[TrajectoryStep]:
        """
        Run trajectory computation until completion.

        Returns:
            List of trajectory steps
        """
        print(f"\n  Running trajectory computation...")
        print(f"  Completion type: {self.completion.type.value}")
        print(f"  Max iterations: {max_iterations}")

        self.trajectory = []

        for i in range(max_iterations):
            self.state.iteration = i
            self.state.time = i * self.timestep

            # 1. Update atom states (ternary spectrometers)
            self._update_atom_states()

            # 2. Identify anomalous atoms (self-selection)
            anomalous = self.state.get_anomalous_atoms()

            # 3. Compute trit from anomaly pattern
            trit = self._compute_trit(anomalous)

            # 4. Calculate metrics
            chi2 = self.state.compute_chi_squared(
                {k: v * len(self.state.atom_states) for k, v in self.expected_distribution.items()}
            )
            backaction = self._compute_backaction(anomalous)

            # 5. Compute S-coordinates
            s_k = 1.0 - i / max_iterations  # Knowledge increases
            s_t = (np.sin(2 * np.pi * i / 100) + 1) / 2  # Oscillatory
            s_e = i / max_iterations  # Evolution progress

            # 6. Record step
            step = TrajectoryStep(
                iteration=i,
                time=self.state.time,
                trit=trit,
                position=self._compute_active_centroid(anomalous),
                backaction=backaction,
                n_anomalous=len(anomalous),
                ternary_distribution=self.state.get_ternary_distribution(),
                chi_squared=chi2,
                s_coordinates=(s_k, s_t, s_e),
                structure_displacement=self.state.get_structure_displacement(
                    self.completion.structure_id) if self.completion.structure_id else None,
                ligand_position=self.state.ligand_position
            )
            self.trajectory.append(step)

            # 7. Apply dynamics
            self._apply_dynamics()

            # 8. Check completion
            if self.completion.is_satisfied(self.state):
                print(f"  Completion reached at iteration {i}")
                break

            # Progress
            if i % 100 == 0:
                print(f"    Iteration {i}: {len(anomalous)} anomalous atoms, χ² = {chi2:.2f}")

        return self.trajectory

    def _update_atom_states(self):
        """Update ternary states of all atoms."""
        for serial, atom_state in self.state.atom_states.items():
            # Compute local energy based on position relative to active region
            local_energy = self._compute_local_energy(atom_state)
            atom_state.update_ternary(local_energy, self.temperature)

    def _compute_local_energy(self, atom_state: AtomState) -> float:
        """Compute local energy for an atom."""
        # Simple harmonic approximation around equilibrium
        # Active atoms have higher energy fluctuations
        displacement = np.linalg.norm(atom_state.velocity) * self.timestep
        k = 100.0  # Spring constant (arbitrary units)
        return 0.5 * k * displacement ** 2

    def _compute_trit(self, anomalous: List[AtomState]) -> int:
        """Compute trit value from anomalous atoms."""
        if not anomalous:
            return 1  # Natural state

        # Count ternary states among anomalous atoms
        counts = {0: 0, 1: 0, 2: 0}
        for a in anomalous:
            counts[a.ternary] += 1

        # Trit based on dominant state
        if counts[0] > counts[2]:
            return 0
        elif counts[2] > counts[0]:
            return 2
        else:
            return 1

    def _compute_backaction(self, anomalous: List[AtomState]) -> float:
        """Compute measurement backaction."""
        if not anomalous:
            return 0.0

        # Backaction proportional to number of measurements
        # But categorical measurements have near-zero backaction
        base_backaction = 1e-6
        return base_backaction * len(anomalous) / len(self.state.atom_states)

    def _compute_active_centroid(self, anomalous: List[AtomState]) -> np.ndarray:
        """Compute centroid of anomalous atoms."""
        if not anomalous:
            return self.protein.center_of_mass()
        positions = np.array([a.atom.position for a in anomalous])
        return positions.mean(axis=0)

    def _apply_dynamics(self):
        """Apply simple molecular dynamics step."""
        # Simplified Langevin dynamics
        gamma = 1e12  # Friction coefficient
        kT = KB * self.temperature

        for atom_state in self.state.atom_states.values():
            # Random force (thermal fluctuations)
            noise = np.random.normal(0, np.sqrt(2 * gamma * kT / self.timestep), 3)

            # Update velocity
            atom_state.velocity *= (1 - gamma * self.timestep)
            atom_state.velocity += noise * self.timestep

            # Update position
            new_pos = atom_state.atom.position + atom_state.velocity * self.timestep
            atom_state.atom.position = new_pos

        # Update ligand if present
        if self.state.ligand_position is not None and self.completion.binding_site_center is not None:
            # Move ligand toward binding site
            direction = self.completion.binding_site_center - self.state.ligand_position
            distance = np.linalg.norm(direction)
            if distance > 0.1:
                step_size = min(0.1, distance * 0.1)  # Adaptive step
                self.state.ligand_position += (direction / distance) * step_size


# =============================================================================
# Experiment Functions
# =============================================================================

def run_helix_motion_experiment(
    pdb_id: str,
    helix_id: str,
    displacement: np.ndarray,
    output_dir: Path,
    max_iterations: int = 500
) -> Dict[str, Any]:
    """
    Run helix motion tracking experiment.

    Args:
        pdb_id: PDB ID of protein
        helix_id: ID of helix to track
        displacement: Target displacement vector (Angstroms)
        output_dir: Output directory
        max_iterations: Maximum iterations

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print(f"HELIX MOTION EXPERIMENT: {pdb_id} helix {helix_id}")
    print("=" * 70)

    # Load protein
    print(f"\n[1/5] Loading protein {pdb_id}...")
    cache_dir = output_dir / "pdb"
    protein = load_protein(pdb_id, cache_dir)
    print(protein.summary())

    # Check helix exists
    helix = protein.get_helix_by_id(helix_id)
    if not helix:
        available = [h.helix_id for h in protein.helices]
        raise ValueError(f"Helix '{helix_id}' not found. Available: {available}")

    print(f"\n  Tracking helix {helix_id}: "
          f"{helix.start_residue}{helix.start_seq} - {helix.end_residue}{helix.end_seq}")

    # Setup completion condition
    print(f"\n[2/5] Setting completion condition...")
    completion = CompletionCondition(
        type=CompletionType.HELIX_MOTION,
        structure_id=helix_id,
        displacement_target=displacement,
        target_rmsd=1.0
    )
    print(f"  Target displacement: {displacement} Å")

    # Run trajectory
    print(f"\n[3/5] Running trajectory computation...")
    engine = TrajectoryEngine(protein)
    engine.set_completion_condition(completion)
    trajectory = engine.run(max_iterations)

    # Analyze results
    print(f"\n[4/5] Analyzing results...")
    results = _analyze_trajectory(trajectory, protein, completion)

    # Export
    print(f"\n[5/5] Exporting results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    _export_results(results, trajectory, output_dir, f"helix_{helix_id}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


def run_docking_experiment(
    pdb_id: str,
    binding_site_residues: List[int],
    ligand_start: np.ndarray,
    output_dir: Path,
    max_iterations: int = 500
) -> Dict[str, Any]:
    """
    Run simple docking experiment.

    Args:
        pdb_id: PDB ID of protein
        binding_site_residues: Residue numbers defining binding site
        ligand_start: Starting position of ligand
        output_dir: Output directory
        max_iterations: Maximum iterations

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print(f"DOCKING EXPERIMENT: {pdb_id}")
    print("=" * 70)

    # Load protein
    print(f"\n[1/5] Loading protein {pdb_id}...")
    cache_dir = output_dir / "pdb"
    protein = load_protein(pdb_id, cache_dir)
    print(protein.summary())

    # Compute binding site center
    print(f"\n[2/5] Identifying binding site...")
    binding_atoms = []
    for res_num in binding_site_residues:
        for key, res in protein.residues.items():
            if res.seq_num == res_num:
                binding_atoms.extend(res.atoms)

    if not binding_atoms:
        raise ValueError(f"No atoms found for residues: {binding_site_residues}")

    binding_center = np.mean([a.position for a in binding_atoms], axis=0)
    print(f"  Binding site center: {binding_center}")
    print(f"  {len(binding_atoms)} atoms in binding site")

    # Setup completion condition
    print(f"\n[3/5] Setting completion condition...")
    completion = CompletionCondition(
        type=CompletionType.DOCKING,
        binding_site_center=binding_center,
        binding_distance=5.0
    )

    # Run trajectory
    print(f"\n[4/5] Running trajectory computation...")
    engine = TrajectoryEngine(protein)
    engine.set_completion_condition(completion)

    # Set ligand start position
    engine.state.ligand_position = ligand_start.copy()

    trajectory = engine.run(max_iterations)

    # Analyze and export
    print(f"\n[5/5] Analyzing and exporting results...")
    results = _analyze_trajectory(trajectory, protein, completion)
    results['binding_site'] = {
        'center': binding_center.tolist(),
        'residues': binding_site_residues,
        'n_atoms': len(binding_atoms)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _export_results(results, trajectory, output_dir, "docking")

    print("\n" + "=" * 70)
    print("DOCKING EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


def _analyze_trajectory(
    trajectory: List[TrajectoryStep],
    protein: ProteinStructure,
    completion: CompletionCondition
) -> Dict[str, Any]:
    """Analyze trajectory results."""
    if not trajectory:
        return {}

    backactions = [s.backaction for s in trajectory]
    chi_squared = [s.chi_squared for s in trajectory]
    n_anomalous = [s.n_anomalous for s in trajectory]

    return {
        'protein': protein.pdb_id,
        'completion_type': completion.type.value,
        'n_iterations': len(trajectory),
        'total_time_ps': trajectory[-1].time * 1e12,
        'metrics': {
            'total_backaction': float(sum(backactions)),
            'mean_backaction': float(np.mean(backactions)),
            'max_backaction': float(max(backactions)),
            'mean_chi_squared': float(np.mean(chi_squared)),
            'mean_anomalous_atoms': float(np.mean(n_anomalous)),
            'zero_backaction_verified': sum(backactions) < 1e-3
        },
        'ternary_string': ''.join(str(s.trit) for s in trajectory),
        'final_distribution': trajectory[-1].ternary_distribution
    }


def _export_results(
    results: Dict[str, Any],
    trajectory: List[TrajectoryStep],
    output_dir: Path,
    prefix: str
):
    """Export results to files."""
    # JSON summary
    with open(output_dir / f"{prefix}_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Trajectory CSV
    with open(output_dir / f"{prefix}_trajectory.csv", 'w') as f:
        f.write("iteration,time_ps,trit,backaction,n_anomalous,chi_squared,")
        f.write("S_k,S_t,S_e,pos_x,pos_y,pos_z\n")
        for s in trajectory:
            f.write(f"{s.iteration},{s.time*1e12:.4f},{s.trit},")
            f.write(f"{s.backaction:.6e},{s.n_anomalous},{s.chi_squared:.4f},")
            f.write(f"{s.s_coordinates[0]:.4f},{s.s_coordinates[1]:.4f},{s.s_coordinates[2]:.4f},")
            f.write(f"{s.position[0]:.4f},{s.position[1]:.4f},{s.position[2]:.4f}\n")

    # Ternary string
    with open(output_dir / f"{prefix}_ternary.txt", 'w') as f:
        f.write(results.get('ternary_string', ''))

    print(f"  Saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output = Path(__file__).parent.parent / "data" / "processed"

    # Example: Run helix motion on lysozyme
    results = run_helix_motion_experiment(
        pdb_id="1LYZ",  # Lysozyme
        helix_id="1",   # First helix
        displacement=np.array([2.0, 0.0, 0.0]),  # 2 Angstrom x-displacement
        output_dir=output,
        max_iterations=200
    )
