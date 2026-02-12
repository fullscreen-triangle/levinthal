"""
Azurin Electron Transfer Validation Experiment

This package implements zero-backaction ternary trisection for
visualizing electron transfer in azurin protein (PDB: 4AZU).

Modules:
    validation_experiment: Main experiment implementation
    visualization_panels: 5 visualization panel generators

Author: Kundai Farai Sachikonye
Date: February 2026
"""

from .validation_experiment import (
    run_validation_experiment,
    create_azurin_structure,
    run_ternary_trisection,
    reconstruct_wavefunction,
)

from .visualization_panels import (
    generate_all_panels,
    plot_panel_1_trajectory,
    plot_panel_2_backaction,
    plot_panel_3_categorical,
    plot_panel_4_probability,
    plot_panel_5_sentropy,
    plot_panel_6_protein_structure,
    plot_panel_7_wavefunction,
    plot_panel_8_electron_cloud,
    plot_panel_9_perturbation_fields,
    plot_panel_10_ternary_trisection,
)

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
