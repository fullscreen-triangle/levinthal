"""
Principia Proteomica: Configuration and Constants
All physical constants, experiment parameters, and validation data.
"""
import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================
K_B = 1.380649e-23       # Boltzmann constant (J/K)
HBAR = 1.054571817e-34   # Reduced Planck constant (J·s)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
C_LIGHT = 2.99792458e8   # Speed of light (m/s)
ELECTRON_MASS = 9.1093837015e-31  # Electron mass (kg)
BOHR_MAGNETON = 9.2740100783e-24  # Bohr magneton (J/T)

# =============================================================================
# Azurin Electron Transfer Parameters
# =============================================================================
AZURIN = {
    'pdb_id': '4AZU',
    'residues': 128,
    'mass_da': 14000,
    'transfer_time_s': 850e-15,
    'transfer_distance_m': 12.5e-10,
    'reorganization_energy_eV': 0.7,
    'electronic_coupling_eV': 0.1,
    'pathway_residues': ['His46', 'Cys112', 'His117', 'Met121'],
    'cu_positions_angstrom': {
        'Cu1': np.array([10.0, 15.0, 20.0]),
        'Cu2': np.array([22.5, 15.0, 20.0]),
    },
    'temperature_K': 4,
    'n_trisection_iterations': 17,
    'timestep_s': 10e-15,
}

# =============================================================================
# SOD1 Parameters
# =============================================================================
SOD1 = {
    'residues': 153,
    'n_hbonds': 165,
    'folding_time_ms': 100,
    'loop_residues': (121, 142),
    'loop_n_hbonds': 8,
}

# =============================================================================
# Enzyme Catalysis Data (8-enzyme validation table)
# =============================================================================
ENZYME_TABLE = [
    {'name': 'SOD1',                  'ec': '1.15.1.1', 'dC': 1, 'log_kcat_Km_obs': 9.85},
    {'name': 'Carbonic anhydrase',    'ec': '4.2.1.1',  'dC': 1, 'log_kcat_Km_obs': 8.0},
    {'name': 'Catalase',              'ec': '1.11.1.6', 'dC': 1, 'log_kcat_Km_obs': 7.6},
    {'name': 'Acetylcholinesterase',  'ec': '3.1.1.7',  'dC': 1, 'log_kcat_Km_obs': 8.3},
    {'name': 'Fumarase',              'ec': '4.2.1.2',  'dC': 2, 'log_kcat_Km_obs': 8.9},
    {'name': 'β-Amylase',             'ec': '3.2.1.2',  'dC': 2, 'log_kcat_Km_obs': 7.6},
    {'name': 'Lysozyme',              'ec': '3.2.1.17', 'dC': 3, 'log_kcat_Km_obs': 6.5},
    {'name': 'Chymotrypsin',          'ec': '3.4.21.1', 'dC': 4, 'log_kcat_Km_obs': 4.0},
]

# =============================================================================
# Disease / ALS SOD1 Variant Data
# =============================================================================
SOD1_VARIANTS = [
    {'name': 'Wild-type', 'mutation': None,   'target_r': 0.87, 'severity': 'none',     'survival_years': None},
    {'name': 'D90A',      'mutation': 'D90A', 'target_r': 0.62, 'severity': 'mild',     'survival_years': 12.0},
    {'name': 'G93A',      'mutation': 'G93A', 'target_r': 0.51, 'severity': 'moderate', 'survival_years': 3.0},
    {'name': 'A4V',       'mutation': 'A4V',  'target_r': 0.43, 'severity': 'severe',   'survival_years': 1.0},
    {'name': 'H46R',      'mutation': 'H46R', 'target_r': 0.38, 'severity': 'severe',   'survival_years': 1.0},
]

# Mutation perturbation types
MUTATION_PERTURBATIONS = {
    'D90A': {'type': 'coupling_reduction', 'region': 'beta_barrel', 'factor': 0.7},
    'G93A': {'type': 'rigidity_increase', 'region': 'beta_barrel', 'factor': 0.6},
    'A4V':  {'type': 'interface_disruption', 'region': 'dimer_interface', 'factor': 0.4},
    'H46R': {'type': 'metal_loss', 'region': 'cu_ligand', 'factor': 0.3},
}

# =============================================================================
# Atomic Shell Data (for partition capacity validation)
# =============================================================================
PERIODIC_TABLE_SHELLS = {
    1: {'capacity': 2,  'cumulative': 2,   'elements': 'H, He'},
    2: {'capacity': 8,  'cumulative': 10,  'elements': 'Li-Ne'},
    3: {'capacity': 18, 'cumulative': 28,  'elements': 'Na-Ar + 3d'},
    4: {'capacity': 32, 'cumulative': 60,  'elements': 'K-Kr + 4d + 4f'},
}

SUBSHELL_LABELS = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
SUBSHELL_CAPACITIES = {0: 2, 1: 6, 2: 10, 3: 14}

# =============================================================================
# Proteomics Validation Targets
# =============================================================================
PROTEOMICS = {
    'ptm_localization_accuracy': 0.887,
    'zero_shot_platform_transfer': 0.893,
    'speed_improvement': 23,
    'partial_sequence_reconstruction': 0.42,
    'scale_free_exponent': 2.3,
    'scale_free_exponent_std': 0.4,
    'cross_platform_cv': 0.021,
}

# =============================================================================
# Kuramoto / Phase-Lock Parameters
# =============================================================================
KURAMOTO = {
    'base_frequency_Hz': 13.2e12,    # ~13 THz (H-bond stretching)
    'frequency_spread': 0.10,         # 10% natural frequency variation
    'coupling_r0_angstrom': 5.0,      # Characteristic interaction range
    'coupling_K0': 2.0,               # Base coupling strength
    'dt_s': 1e-15,                    # 1 fs timestep
    'coherence_threshold': 0.8,       # Native state threshold
    'misfolding_threshold': 0.5,      # Misfolding criterion
}
