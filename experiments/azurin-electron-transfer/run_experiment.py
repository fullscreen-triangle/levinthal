#!/usr/bin/env python3
"""
Run Complete Azurin Electron Transfer Validation Experiment

This script runs the full validation experiment:
1. Simulates electron transfer in azurin (Cu(I) -> Cu(II))
2. Performs zero-backaction ternary trisection localization
3. Reconstructs wavefunction from categorical trajectory
4. Generates all 5 visualization panels
5. Exports results and validation metrics

Author: Kundai Farai Sachikonye
Date: February 2026

Usage:
    python run_experiment.py
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from validation_experiment import run_validation_experiment
from visualization_panels import generate_all_panels


def main():
    """Run the complete validation experiment."""
    print("+" + "=" * 78 + "+")
    print("|" + " AZURIN ELECTRON TRANSFER VALIDATION EXPERIMENT ".center(78) + "|")
    print("|" + " Zero-Backaction Ternary Trisection for Cu(I) -> Cu(II) ".center(78) + "|")
    print("+" + "=" * 78 + "+")

    # Define directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'processed'
    viz_dir = base_dir / 'visualizations'

    # Step 1: Run validation experiment
    print("\n" + "-" * 80)
    print("PHASE 1: VALIDATION EXPERIMENT")
    print("-" * 80)

    results = run_validation_experiment(data_dir)

    # Step 2: Generate visualizations
    print("\n" + "-" * 80)
    print("PHASE 2: VISUALIZATION GENERATION")
    print("-" * 80)

    generate_all_panels(data_dir, viz_dir)

    # Summary
    print("\n" + "+" + "=" * 78 + "+")
    print("|" + " EXPERIMENT COMPLETE ".center(78) + "|")
    print("+" + "=" * 78 + "+")

    # Key results
    total_backaction = results['metrics']['total_backaction']
    threshold = results['validation']['threshold']
    verified = results['validation']['zero_backaction_verified']

    status = "[VERIFIED]" if verified else "[FAILED]"

    print(f"|  {'Zero-Backaction:':<30} {status:<45}  |")
    print(f"|  {'Total Backaction:':<30} Dp/p = {total_backaction:.2e}{'':<27}  |")
    print(f"|  {'Threshold:':<30} Dp/p < {threshold:.0e}{'':<29}  |")
    print(f"|  {'Ternary Iterations:':<30} {results['metrics']['iterations']:<45}  |")
    print(f"|  {'Speedup vs Binary:':<30} {results['metrics']['speedup_vs_binary']:.3f}x{'':<40}  |")

    print("+" + "-" * 78 + "+")
    print(f"|  {'Data saved to:':<30} {str(data_dir):<44}  |")
    print(f"|  {'Visualizations saved to:':<30} {str(viz_dir):<44}  |")
    print("+" + "=" * 78 + "+")

    return results


if __name__ == "__main__":
    results = main()
