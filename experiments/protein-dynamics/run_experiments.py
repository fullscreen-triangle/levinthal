#!/usr/bin/env python3
"""
Run Protein Dynamics Experiments

This script runs validation experiments using real PDB structures:
1. Helix motion tracking in lysozyme
2. Sheet dynamics in immunoglobulin
3. Simple docking simulation

Usage:
    python run_experiments.py [experiment_name]

    experiment_name: helix, sheet, docking, all (default: all)

Author: Kundai Farai Sachikonye
Date: February 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import with absolute paths (not relative)
from pdb_loader import load_protein, ProteinStructure
from virtual_light import (
    VirtualLightEngine, create_binding_perturbation,
    create_helix_motion_perturbation
)


# =============================================================================
# Output Directory
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
PDB_CACHE = Path(__file__).parent / "data" / "pdb"


# =============================================================================
# Experiment 1: Helix Motion in Lysozyme
# =============================================================================

def run_helix_experiment():
    """
    Track alpha helix motion in hen egg-white lysozyme (1LYZ).

    Lysozyme has well-characterized helices that we can track.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: HELIX MOTION IN LYSOZYME (1LYZ)")
    print("=" * 70)

    # Load protein
    print("\n[1/6] Loading lysozyme from PDB...")
    protein = load_protein("1LYZ", cache_dir=PDB_CACHE)
    print(protein.summary())

    # List available helices
    print("\n[2/6] Available helices:")
    for helix in protein.helices:
        atoms = protein.get_helix_atoms(helix.helix_id)
        print(f"  Helix {helix.helix_id}: {helix.start_residue}{helix.start_seq} - "
              f"{helix.end_residue}{helix.end_seq} ({len(atoms)} atoms)")

    # Select first helix
    if not protein.helices:
        print("  No helices found!")
        return None

    target_helix = protein.helices[0]
    helix_id = target_helix.helix_id.strip()
    print(f"\n  Tracking helix: {helix_id}")

    # Initialize virtual light engine
    print("\n[3/6] Initializing virtual light engine...")
    engine = VirtualLightEngine(protein)
    engine.initialize_states(temperature=300.0)

    # Get helix atoms for perturbation
    helix_atoms = protein.get_helix_atoms(helix_id)
    if not helix_atoms:
        print("  No helix atoms found!")
        return None

    # Create helix motion perturbation (simulate 2Å displacement along x)
    print("\n[4/6] Applying helix motion perturbation...")
    displacement = np.array([2.0, 0.0, 0.0])  # 2 Angstrom x-displacement
    perturbation = create_helix_motion_perturbation(
        helix_atoms, direction=displacement / np.linalg.norm(displacement),
        amplitude=2.0
    )

    # Run dynamics
    print("\n[5/6] Running state dynamics (200 steps)...")
    history = engine.run_dynamics(n_steps=200, perturbation=perturbation, record_interval=5)

    # Collect results
    print("\n[6/6] Analyzing results...")

    # Compute ternary string
    ternary_string = ""
    for i, dist in enumerate(history):
        if dist.ground > dist.excited:
            ternary_string += "0"
        elif dist.excited > dist.ground:
            ternary_string += "2"
        else:
            ternary_string += "1"

    # Get final beams
    abs_beam, emi_beam = engine.generate_virtual_beams()

    # Identify active atoms
    anomalous = engine.get_anomalous_atoms()
    active_residues = set(s.atom.residue_id for s in anomalous)

    results = {
        "experiment": "helix_motion",
        "protein": "1LYZ",
        "helix_id": helix_id,
        "helix_residues": f"{target_helix.start_seq}-{target_helix.end_seq}",
        "displacement_target": displacement.tolist(),
        "n_iterations": len(history),
        "n_atoms": len(protein.atoms),
        "n_helix_atoms": len(helix_atoms),
        "final_distribution": {
            "ground": history[-1].ground,
            "natural": history[-1].natural,
            "excited": history[-1].excited
        },
        "ternary_string": ternary_string,
        "virtual_beams": {
            "absorption_intensity": abs_beam.intensity,
            "emission_intensity": emi_beam.intensity,
            "absorption_atoms": abs_beam.n_atoms,
            "emission_atoms": emi_beam.n_atoms
        },
        "anomalous_atoms": len(anomalous),
        "active_residues": list(active_residues)[:20],  # First 20
        "chi_squared": history[-1].chi_squared((0.1, 0.8, 0.1)),
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_file = OUTPUT_DIR / "helix_motion_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save ternary string
    with open(OUTPUT_DIR / "helix_motion_ternary.txt", 'w') as f:
        f.write(ternary_string)

    # Print summary
    print("\n" + "-" * 50)
    print("HELIX MOTION RESULTS")
    print("-" * 50)
    print(f"Protein: {results['protein']}")
    print(f"Helix: {results['helix_id']} (residues {results['helix_residues']})")
    print(f"Atoms tracked: {results['n_helix_atoms']} helix / {results['n_atoms']} total")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Ternary string: {ternary_string[:50]}...")
    print(f"Final distribution: G={results['final_distribution']['ground']}, "
          f"N={results['final_distribution']['natural']}, "
          f"E={results['final_distribution']['excited']}")
    print(f"Anomalous atoms: {results['anomalous_atoms']} (self-selected)")
    print(f"Chi-squared: {results['chi_squared']:.2f}")
    print(f"Saved to: {output_file}")

    return results


# =============================================================================
# Experiment 2: Sheet Dynamics in Immunoglobulin
# =============================================================================

def run_sheet_experiment():
    """
    Track beta sheet dynamics in immunoglobulin Fab fragment (1IGT).

    Immunoglobulins have characteristic beta sandwich structures.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SHEET DYNAMICS IN IMMUNOGLOBULIN (1IGT)")
    print("=" * 70)

    # Load protein
    print("\n[1/6] Loading immunoglobulin Fab from PDB...")
    try:
        protein = load_protein("1IGT", cache_dir=PDB_CACHE)
    except Exception as e:
        print(f"  Failed to load 1IGT: {e}")
        print("  Trying alternative: 1HZH (antibody Fab)")
        protein = load_protein("1HZH", cache_dir=PDB_CACHE)

    print(protein.summary())

    # List available sheets
    print("\n[2/6] Available sheets:")
    if not protein.sheets:
        print("  No sheets found in structure!")
        return None

    for sheet in protein.sheets[:5]:  # First 5
        print(f"  Sheet {sheet.sheet_id} strand {sheet.strand}: "
              f"{sheet.start_residue}{sheet.start_seq} - "
              f"{sheet.end_residue}{sheet.end_seq}")

    # Select first sheet
    target_sheet = protein.sheets[0]
    sheet_id = target_sheet.sheet_id.strip()
    print(f"\n  Tracking sheet: {sheet_id}")

    # Get sheet atoms
    sheet_atoms = protein.get_sheets_atoms()
    print(f"  Sheet atoms: {len(sheet_atoms)}")

    # Initialize virtual light engine
    print("\n[3/6] Initializing virtual light engine...")
    engine = VirtualLightEngine(protein)
    engine.initialize_states(temperature=300.0)

    # Create sheet motion perturbation
    print("\n[4/6] Applying sheet dynamics perturbation...")
    if sheet_atoms:
        sheet_center = np.mean([a.position for a in sheet_atoms], axis=0)
        perturbation = create_binding_perturbation(sheet_center, strength=0.5)
    else:
        perturbation = None

    # Run dynamics
    print("\n[5/6] Running state dynamics (150 steps)...")
    history = engine.run_dynamics(n_steps=150, perturbation=perturbation, record_interval=5)

    # Collect results
    print("\n[6/6] Analyzing results...")

    ternary_string = ""
    for dist in history:
        if dist.ground > dist.excited:
            ternary_string += "0"
        elif dist.excited > dist.ground:
            ternary_string += "2"
        else:
            ternary_string += "1"

    abs_beam, emi_beam = engine.generate_virtual_beams()
    anomalous = engine.get_anomalous_atoms()

    results = {
        "experiment": "sheet_dynamics",
        "protein": protein.pdb_id,
        "sheet_id": sheet_id,
        "n_iterations": len(history),
        "n_atoms": len(protein.atoms),
        "n_sheet_atoms": len(sheet_atoms),
        "final_distribution": {
            "ground": history[-1].ground,
            "natural": history[-1].natural,
            "excited": history[-1].excited
        },
        "ternary_string": ternary_string,
        "virtual_beams": {
            "absorption_intensity": abs_beam.intensity,
            "emission_intensity": emi_beam.intensity
        },
        "anomalous_atoms": len(anomalous),
        "chi_squared": history[-1].chi_squared((0.1, 0.8, 0.1)),
        "timestamp": datetime.now().isoformat()
    }

    # Save
    output_file = OUTPUT_DIR / "sheet_dynamics_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "sheet_dynamics_ternary.txt", 'w') as f:
        f.write(ternary_string)

    # Print summary
    print("\n" + "-" * 50)
    print("SHEET DYNAMICS RESULTS")
    print("-" * 50)
    print(f"Protein: {results['protein']}")
    print(f"Sheet: {results['sheet_id']}")
    print(f"Sheet atoms: {results['n_sheet_atoms']}")
    print(f"Ternary string: {ternary_string[:50]}...")
    print(f"Anomalous atoms: {results['anomalous_atoms']}")
    print(f"Chi-squared: {results['chi_squared']:.2f}")
    print(f"Saved to: {output_file}")

    return results


# =============================================================================
# Experiment 3: Docking Simulation
# =============================================================================

def run_docking_experiment_main():
    """
    Simple docking simulation using azurin (4AZU).

    Simulates a ligand approaching the copper binding site.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: DOCKING SIMULATION IN AZURIN (4AZU)")
    print("=" * 70)

    # Load protein
    print("\n[1/6] Loading azurin from PDB...")
    protein = load_protein("4AZU", cache_dir=PDB_CACHE)
    print(protein.summary())

    # Find copper binding site (Cu + coordinating residues)
    print("\n[2/6] Identifying copper binding site...")
    cu_atoms = [a for a in protein.atoms if a.element == "CU"]

    if cu_atoms:
        binding_center = cu_atoms[0].position
        print(f"  Copper position: {binding_center}")
    else:
        # Use center of protein if no Cu found
        binding_center = protein.center_of_mass()
        print(f"  Using center of mass: {binding_center}")

    # Coordinating residues (His46, Cys112, His117, Met121)
    coord_residues = [46, 112, 117, 121]
    binding_atoms = []
    for atom in protein.atoms:
        if atom.residue_seq in coord_residues:
            binding_atoms.append(atom)
    print(f"  Binding site atoms: {len(binding_atoms)}")

    # Initialize virtual light engine
    print("\n[3/6] Initializing virtual light engine...")
    engine = VirtualLightEngine(protein)
    engine.initialize_states(temperature=300.0)

    # Simulate ligand approaching binding site
    print("\n[4/6] Simulating ligand approach...")

    # Start ligand 20 Angstroms away
    ligand_start = binding_center + np.array([20.0, 0.0, 0.0])
    ligand_position = ligand_start.copy()

    trajectory_data = []
    ternary_string = ""

    n_steps = 100
    for step in range(n_steps):
        # Move ligand toward binding site
        direction = binding_center - ligand_position
        distance = np.linalg.norm(direction)

        if distance > 1.0:  # Stop when close
            step_size = min(0.5, distance * 0.1)
            ligand_position += (direction / distance) * step_size

        # Create perturbation from ligand
        perturbation = create_binding_perturbation(ligand_position, strength=1.5)

        # Update states
        engine.update_states(perturbation)

        # Record
        dist = engine.get_state_distribution()
        trit = engine.compute_trit()
        ternary_string += str(trit)

        trajectory_data.append({
            "step": step,
            "ligand_distance": float(distance),
            "ligand_position": ligand_position.tolist(),
            "distribution": [dist.ground, dist.natural, dist.excited],
            "trit": trit
        })

        if step % 20 == 0:
            print(f"    Step {step}: ligand distance = {distance:.2f} Å, trit = {trit}")

    # Final analysis
    print("\n[5/6] Analyzing results...")
    anomalous = engine.get_anomalous_atoms()
    abs_beam, emi_beam = engine.generate_virtual_beams()

    # Identify which residues responded (self-selected binding site)
    active_residues = set()
    for spec in anomalous:
        active_residues.add(spec.atom.residue_seq)

    # Check if coordinating residues were detected
    coord_detected = active_residues.intersection(set(coord_residues))

    results = {
        "experiment": "docking",
        "protein": "4AZU",
        "binding_site_center": binding_center.tolist(),
        "coordinating_residues": coord_residues,
        "n_iterations": n_steps,
        "n_atoms": len(protein.atoms),
        "ligand_start": ligand_start.tolist(),
        "ligand_final": ligand_position.tolist(),
        "final_distance": float(np.linalg.norm(binding_center - ligand_position)),
        "final_distribution": {
            "ground": engine.get_state_distribution().ground,
            "natural": engine.get_state_distribution().natural,
            "excited": engine.get_state_distribution().excited
        },
        "ternary_string": ternary_string,
        "virtual_beams": {
            "absorption_intensity": abs_beam.intensity,
            "emission_intensity": emi_beam.intensity
        },
        "anomalous_atoms": len(anomalous),
        "active_residues_detected": list(active_residues)[:20],
        "coordinating_residues_detected": list(coord_detected),
        "binding_site_accuracy": len(coord_detected) / len(coord_residues),
        "timestamp": datetime.now().isoformat()
    }

    # Save
    print("\n[6/6] Saving results...")
    output_file = OUTPUT_DIR / "docking_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "docking_ternary.txt", 'w') as f:
        f.write(ternary_string)

    with open(OUTPUT_DIR / "docking_trajectory.json", 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    # Print summary
    print("\n" + "-" * 50)
    print("DOCKING RESULTS")
    print("-" * 50)
    print(f"Protein: {results['protein']}")
    print(f"Ligand approach: {results['ligand_start'][0]:.1f}A -> {results['final_distance']:.1f}A")
    print(f"Ternary string: {ternary_string}")
    print(f"Anomalous atoms: {results['anomalous_atoms']} (self-selected)")
    print(f"Active residues: {len(active_residues)}")
    print(f"Coordinating residues detected: {coord_detected}")
    print(f"Binding site accuracy: {results['binding_site_accuracy']*100:.0f}%")
    print(f"Saved to: {output_file}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    """Run experiments based on command line argument."""
    print("=" * 70)
    print("PROTEIN DYNAMICS EXPERIMENTS")
    print("Atoms as Ternary Spectrometers Framework")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"PDB cache: {PDB_CACHE}")

    # Parse argument
    experiment = "all"
    if len(sys.argv) > 1:
        experiment = sys.argv[1].lower()

    all_results = {}

    if experiment in ["helix", "all"]:
        try:
            result = run_helix_experiment()
            if result:
                all_results["helix_motion"] = result
        except Exception as e:
            print(f"\nHelix experiment failed: {e}")

    if experiment in ["sheet", "all"]:
        try:
            result = run_sheet_experiment()
            if result:
                all_results["sheet_dynamics"] = result
        except Exception as e:
            print(f"\nSheet experiment failed: {e}")

    if experiment in ["docking", "all"]:
        try:
            result = run_docking_experiment_main()
            if result:
                all_results["docking"] = result
        except Exception as e:
            print(f"\nDocking experiment failed: {e}")

    # Save combined results
    if all_results:
        combined_file = OUTPUT_DIR / "all_experiments.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")

    # Summary table
    print("\nSUMMARY:")
    print("-" * 50)
    for name, result in all_results.items():
        print(f"  {name}:")
        print(f"    Protein: {result.get('protein', 'N/A')}")
        print(f"    Iterations: {result.get('n_iterations', 'N/A')}")
        print(f"    Anomalous atoms: {result.get('anomalous_atoms', 'N/A')}")
        ternary = result.get('ternary_string', '')
        if len(ternary) > 30:
            ternary = ternary[:30] + "..."
        print(f"    Ternary: {ternary}")


if __name__ == "__main__":
    main()
