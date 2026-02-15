"""
Protein Dynamics Experiment Package

Implements the atoms-as-ternary-spectrometers framework for:
1. Alpha helix motion tracking
2. Beta sheet dynamics
3. Ligand docking
4. General protein dynamics

Modules:
    pdb_loader: Download and parse PDB structures
    trajectory_experiment: Completion-driven trajectory computation
    virtual_light: Ternary state dynamics and virtual beam generation

Author: Kundai Farai Sachikonye
Date: February 2026
"""

from .pdb_loader import (
    load_protein,
    load_local_pdb,
    ProteinStructure,
    Atom,
    Residue,
    Helix,
    Sheet,
    SecondaryStructure,
)

from .trajectory_experiment import (
    TrajectoryEngine,
    CompletionCondition,
    CompletionType,
    SystemState,
    TrajectoryStep,
    run_helix_motion_experiment,
    run_docking_experiment,
)

from .virtual_light import (
    VirtualLightEngine,
    AtomSpectrometer,
    VirtualBeam,
    StateDistribution,
    create_binding_perturbation,
    create_helix_motion_perturbation,
)

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
