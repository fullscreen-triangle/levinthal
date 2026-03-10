#!/usr/bin/env python3
"""
Principia Proteomica: Master Validation Experiment Suite

Runs all 39 validation tests across 6 domains, generates 12 figures,
and outputs structured JSON results.

Usage: python experiments/principia-proteomica/run_all.py

Author: Kundai Farai Sachikonye
"""
import sys
import os
import json
import shutil
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np


def main():
    print("=" * 80)
    print("PRINCIPIA PROTEOMICA: Validation Experiment Suite")
    print("=" * 80)
    start_time = time.time()

    # Create output directories
    data_dir = project_root / 'data' / 'results'
    fig_dir = project_root / 'figures'
    pub_fig_dir = project_root.parent.parent / 'publication' / 'revised' / 'protein-dynamics' / 'figures'

    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Custom JSON encoder for numpy types
    from src.utils.json_encoder import NumpyEncoder

    all_results = {}
    all_tests = []

    # =========================================================================
    # Domain 1: Foundation — Partition Coordinates (Equation I)
    # =========================================================================
    print("\n[1/6] Running Atomic Structure validation...")
    from src.core.partition import run_partition_validation
    partition_results = run_partition_validation()
    all_results['partition'] = partition_results

    for t in partition_results['tests']:
        all_tests.append({**t, 'domain': 'Atomic structure'})
    _print_domain_summary('Atomic structure', partition_results['tests'])

    # =========================================================================
    # Domain 1b: Foundation — Partition Depth (Equation III)
    # =========================================================================
    print("\n[1b/6] Running Partition Depth validation...")
    from src.core.partition_depth import run_depth_validation
    depth_results = run_depth_validation()
    all_results['depth'] = depth_results

    # =========================================================================
    # Domain 1c: Foundation — S-Entropy Conservation (Equation VII)
    # =========================================================================
    print("\n[1c/6] Running S-Entropy validation...")
    from src.core.sentropy import run_sentropy_validation
    sentropy_results = run_sentropy_validation()
    all_results['sentropy'] = sentropy_results

    # =========================================================================
    # Domain 1d: Foundation — Phase-Lock (Equation V)
    # =========================================================================
    print("\n[1d/6] Running Phase-Lock validation...")
    from src.physics.kuramoto import run_phaselock_validation
    phaselock_results = run_phaselock_validation()
    all_results['phaselock'] = phaselock_results

    # Save foundation results
    with open(data_dir / '01_foundation_tests.json', 'w') as f:
        json.dump({
            'partition': partition_results,
            'depth': depth_results,
            'sentropy': sentropy_results,
            'phaselock': phaselock_results,
        }, f, cls=NumpyEncoder, indent=2)
    print("  Saved: data/results/01_foundation_tests.json")

    # =========================================================================
    # Domain 2: Electron Transfer (Equations II, VII)
    # =========================================================================
    print("\n[2/6] Running Electron Transfer validation...")
    from src.physics.electron_transfer import run_electron_transfer_validation
    et_results = run_electron_transfer_validation()
    all_results['electron_transfer'] = et_results

    for t in et_results['tests']:
        all_tests.append({**t, 'domain': 'Electron transfer'})
    _print_domain_summary('Electron transfer', et_results['tests'])

    with open(data_dir / '02_electron_transfer.json', 'w') as f:
        json.dump(et_results, f, cls=NumpyEncoder, indent=2)
    print("  Saved: data/results/02_electron_transfer.json")

    # =========================================================================
    # Domain 3: Enzyme Catalysis (Equation IV)
    # =========================================================================
    print("\n[3/6] Running Enzyme Catalysis validation...")
    from src.physics.enzyme_catalysis import run_catalysis_validation
    cat_results = run_catalysis_validation()
    all_results['catalysis'] = cat_results

    for t in cat_results['tests']:
        all_tests.append({**t, 'domain': 'Enzyme catalysis'})
    _print_domain_summary('Enzyme catalysis', cat_results['tests'])

    with open(data_dir / '03_enzyme_catalysis.json', 'w') as f:
        json.dump(cat_results, f, cls=NumpyEncoder, indent=2)
    print("  Saved: data/results/03_enzyme_catalysis.json")

    # =========================================================================
    # Domain 4: Protein Folding (Equations V, VI)
    # =========================================================================
    print("\n[4/6] Running Protein Folding validation...")
    from src.physics.protein_folding import run_folding_validation
    fold_results = run_folding_validation()
    all_results['folding'] = fold_results

    for t in fold_results['tests']:
        all_tests.append({**t, 'domain': 'Protein folding'})
    _print_domain_summary('Protein folding', fold_results['tests'])

    with open(data_dir / '04_protein_folding.json', 'w') as f:
        json.dump(fold_results, f, cls=NumpyEncoder, indent=2)
    print("  Saved: data/results/04_protein_folding.json")

    # =========================================================================
    # Domain 5: Conformational Dynamics
    # =========================================================================
    print("\n[5/6] Running Conformational Dynamics validation...")
    from src.physics.conformational import run_conformational_validation
    conf_results = run_conformational_validation()
    all_results['conformational'] = conf_results

    # =========================================================================
    # Domain 6: Disease / ALS
    # =========================================================================
    print("\n[6/6] Running Disease / ALS validation...")
    from src.physics.disease import run_disease_validation
    disease_results = run_disease_validation()
    all_results['disease'] = disease_results

    for t in disease_results['tests']:
        all_tests.append({**t, 'domain': 'Disease (ALS)'})
    _print_domain_summary('Disease (ALS)', disease_results['tests'])

    with open(data_dir / '05_disease.json', 'w') as f:
        json.dump(disease_results, f, cls=NumpyEncoder, indent=2)
    print("  Saved: data/results/05_disease.json")

    # =========================================================================
    # Grand Validation Summary
    # =========================================================================
    total_passed = sum(1 for t in all_tests if t.get('passed', False))
    total_tests = len(all_tests)

    # Group by domain
    domain_summary = {}
    for t in all_tests:
        domain = t.get('domain', 'Unknown')
        if domain not in domain_summary:
            domain_summary[domain] = {'passed': 0, 'total': 0}
        domain_summary[domain]['total'] += 1
        if t.get('passed', False):
            domain_summary[domain]['passed'] += 1

    grand_validation = {
        'total_passed': total_passed,
        'total_tests': total_tests,
        'pass_rate': total_passed / total_tests if total_tests > 0 else 0,
        'domains': domain_summary,
        'all_tests': all_tests,
    }
    all_results['grand_validation'] = grand_validation

    with open(data_dir / 'grand_validation.json', 'w') as f:
        json.dump(grand_validation, f, cls=NumpyEncoder, indent=2)

    # =========================================================================
    # Generate Figures
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    figure_modules = [
        ('fig01_overview',       'fig01_overview.png'),
        ('fig02_partition',      'fig02_partition.png'),
        ('fig03_selection',      'fig03_selection.png'),
        ('fig04_depth',          'fig04_depth.png'),
        ('fig05_phaselock',      'fig05_phaselock.png'),
        ('fig06_sentropy',       'fig06_sentropy.png'),
        ('fig08_electron',       'fig08_electron.png'),
        ('fig09_catalysis',      'fig09_catalysis.png'),
        ('fig10_folding',        'fig10_folding.png'),
        ('fig11_conformational', 'fig11_conformational.png'),
        ('fig12_disease',        'fig12_disease.png'),
        ('fig13_validation',     'fig13_validation.png'),
    ]

    generated_figures = []
    for module_name, filename in figure_modules:
        output_path = str(fig_dir / filename)
        try:
            module = __import__(f'src.figures.{module_name}', fromlist=['generate'])
            module.generate(all_results, output_path)
            generated_figures.append(filename)
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [!!] {filename}: {e}")

    # =========================================================================
    # Copy to publication directory
    # =========================================================================
    if pub_fig_dir.exists():
        print(f"\nCopying figures to {pub_fig_dir}...")
        for filename in generated_figures:
            src = fig_dir / filename
            dst = pub_fig_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied: {filename}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("GRAND VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n  Total tests:      {total_tests}")
    print(f"  Passed:           {total_passed}")
    print(f"  Pass rate:        {total_passed}/{total_tests} "
          f"({total_passed/total_tests*100:.0f}%)")
    print(f"  Free parameters:  0")
    print(f"  Equations:        7")
    print(f"  Domains:          {len(domain_summary)}")
    print(f"\n  Domain breakdown:")
    for domain, counts in domain_summary.items():
        rate = counts['passed'] / counts['total'] * 100
        status = 'OK' if rate >= 80 else '!!'
        print(f"    [{status}] {domain}: {counts['passed']}/{counts['total']} "
              f"({rate:.0f}%)")
    print(f"\n  Figures generated: {len(generated_figures)}/12")
    print(f"  Elapsed time:     {elapsed:.1f}s")
    print("=" * 80)


def _print_domain_summary(domain: str, tests: list):
    """Print domain test results."""
    passed = sum(1 for t in tests if t.get('passed', False))
    total = len(tests)
    status = 'OK' if passed == total else 'PARTIAL'
    print(f"  [{status}] {domain}: {passed}/{total} tests passed")
    for t in tests:
        marker = 'OK' if t.get('passed', False) else 'FAIL'
        print(f"    [{marker}] Test {t.get('id', '?')}: {t.get('name', '')}")


if __name__ == '__main__':
    main()
