"""
Validation 2: Secondary Structure S-Entropy Encoding
=====================================================
Computes S-entropy coordinates for secondary structure types from their
characteristic amide band frequencies, demonstrating that alpha-helix,
beta-sheet, and coil occupy distinct regions of S-entropy space.

Outputs:
  results/secondary_structure_coordinates.csv
  results/secondary_structure_separation.json
  figures/panel_2_secondary_structure.png
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'validation', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Characteristic amide band frequencies (cm^-1) for secondary structures
# Sources: Barth 2007, Krimm & Bandekar 1986, Barth & Zscherp 2002
# ============================================================================
# Each structure type has characteristic amide I, II, III, and A bands
SECONDARY_STRUCTURES = {
    "alpha_helix": {
        "label": r"$\alpha$-helix",
        "amide_I": 1650,
        "amide_II": 1545,
        "amide_III": 1300,
        "amide_A": 3290,
        "color": "#F44336",
    },
    "beta_sheet_parallel": {
        "label": r"$\beta$-sheet (parallel)",
        "amide_I": 1630,
        "amide_II": 1530,
        "amide_III": 1235,
        "amide_A": 3280,
        "color": "#2196F3",
    },
    "beta_sheet_antiparallel": {
        "label": r"$\beta$-sheet (antiparallel)",
        "amide_I": 1680,
        "amide_II": 1530,
        "amide_III": 1230,
        "amide_A": 3270,
        "color": "#1565C0",
    },
    "310_helix": {
        "label": r"$3_{10}$-helix",
        "amide_I": 1660,
        "amide_II": 1550,
        "amide_III": 1310,
        "amide_A": 3300,
        "color": "#FF5722",
    },
    "random_coil": {
        "label": "Random coil",
        "amide_I": 1645,
        "amide_II": 1535,
        "amide_III": 1260,
        "amide_A": 3300,
        "color": "#9E9E9E",
    },
    "beta_turn_type_I": {
        "label": r"$\beta$-turn (type I)",
        "amide_I": 1670,
        "amide_II": 1525,
        "amide_III": 1280,
        "amide_A": 3310,
        "color": "#4CAF50",
    },
    "beta_turn_type_II": {
        "label": r"$\beta$-turn (type II)",
        "amide_I": 1685,
        "amide_II": 1520,
        "amide_III": 1275,
        "amide_A": 3305,
        "color": "#8BC34A",
    },
    "polyproline_II": {
        "label": "PPII helix",
        "amide_I": 1645,
        "amide_II": 1540,
        "amide_III": 1245,
        "amide_A": 3310,
        "color": "#FF9800",
    },
}

# Reference bounds for normalization (from all known protein amide bands)
OMEGA_REF_MAX = 4401  # H2 stretch (highest molecular vibration)
OMEGA_REF_MIN = 218   # CCl4 lowest (from compound database)


def compute_sentropy_from_frequencies(freqs):
    """
    Compute S-entropy coordinates from a set of vibrational frequencies.

    Sk: Normalized Shannon entropy of frequency distribution
    St: Log ratio of frequency range (temporal span)
    Se: Harmonic proximity density (evolution entropy)
    """
    freqs = np.array(sorted(freqs))
    N = len(freqs)

    # Sk: knowledge entropy (spectral distribution)
    p = freqs / freqs.sum()
    H = -np.sum(p * np.log2(p + 1e-15))
    Sk = H / np.log2(N) if N > 1 else freqs[0] / OMEGA_REF_MAX

    # St: temporal entropy (timescale span)
    if N >= 2:
        St = np.log(freqs.max() / freqs.min()) / np.log(OMEGA_REF_MAX / OMEGA_REF_MIN)
    else:
        St = np.log(freqs[0] / 0.39) / np.log(OMEGA_REF_MAX / 0.39)

    # Se: evolution entropy (harmonic network density)
    if N < 2:
        Se = 0.0
    else:
        n_pairs = N * (N - 1) // 2
        n_harmonic = 0
        delta = 0.05
        for i in range(N):
            for j in range(i + 1, N):
                ratio = max(freqs[i], freqs[j]) / min(freqs[i], freqs[j])
                # Check against rationals p/q with p,q <= 8
                is_harmonic = False
                for p in range(1, 9):
                    for q in range(1, p + 1):
                        if abs(ratio - p / q) < delta:
                            is_harmonic = True
                            break
                    if is_harmonic:
                        break
                if is_harmonic:
                    n_harmonic += 1
        Se = n_harmonic / max(n_pairs, 1)

    return np.clip(Sk, 0, 1), np.clip(St, 0, 1), np.clip(Se, 0, 1)


def ternary_encode(Sk, St, Se, depth=18):
    """Generate interleaved ternary address."""
    remainders = [Sk, St, Se]
    trits = []
    for j in range(depth):
        dim = j % 3
        t = int(3 * remainders[dim])
        t = min(t, 2)
        remainders[dim] = 3 * remainders[dim] - t
        trits.append(t)
    return ''.join(str(t) for t in trits)


def main():
    print("=" * 70)
    print("VALIDATION 2: Secondary Structure S-Entropy Encoding")
    print("=" * 70)

    # ---- Step 1: Compute S-entropy for each secondary structure type ----
    records = []
    for key, ss in SECONDARY_STRUCTURES.items():
        freqs = [ss["amide_III"], ss["amide_II"], ss["amide_I"], ss["amide_A"]]
        Sk, St, Se = compute_sentropy_from_frequencies(freqs)
        addr = ternary_encode(Sk, St, Se, depth=18)
        records.append({
            "structure": key,
            "label": ss["label"],
            "amide_I": ss["amide_I"],
            "amide_II": ss["amide_II"],
            "amide_III": ss["amide_III"],
            "amide_A": ss["amide_A"],
            "Sk": round(Sk, 4),
            "St": round(St, 4),
            "Se": round(Se, 4),
            "ternary_18": addr,
            "ternary_6": addr[:6],
            "ternary_3": addr[:3],
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "secondary_structure_coordinates.csv"), index=False)
    print(f"\n[1] Computed S-entropy coordinates for {len(df)} secondary structure types:")
    print(df[["structure", "Sk", "St", "Se", "ternary_6"]].to_string(index=False))

    # ---- Step 2: Pairwise separation ----
    separation = {}
    names = [r["structure"] for r in records]
    coords = [(r["Sk"], r["St"], r["Se"]) for r in records]

    for (i, j) in combinations(range(len(records)), 2):
        d = np.sqrt(sum((a - b)**2 for a, b in zip(coords[i], coords[j])))
        spd = 0
        t1, t2 = records[i]["ternary_18"], records[j]["ternary_18"]
        for k in range(min(len(t1), len(t2))):
            if t1[k] != t2[k]:
                break
            spd = k + 1
        else:
            spd = min(len(t1), len(t2))
        pair_key = f"{names[i]}_vs_{names[j]}"
        separation[pair_key] = {
            "structure_1": names[i],
            "structure_2": names[j],
            "euclidean_distance": round(float(d), 6),
            "shared_prefix_depth": spd,
            "distinct_at_depth_3": records[i]["ternary_3"] != records[j]["ternary_3"],
            "distinct_at_depth_6": records[i]["ternary_6"] != records[j]["ternary_6"],
        }

    with open(os.path.join(RESULTS_DIR, "secondary_structure_separation.json"), 'w') as f:
        json.dump(separation, f, indent=2)

    # Key separations
    print(f"\n[2] Key pairwise separations:")
    major_types = ["alpha_helix", "beta_sheet_parallel", "beta_sheet_antiparallel", "random_coil"]
    for (i, j) in combinations(range(len(records)), 2):
        if names[i] in major_types and names[j] in major_types:
            d = np.sqrt(sum((a - b)**2 for a, b in zip(coords[i], coords[j])))
            print(f"    {names[i]:30s} vs {names[j]:30s}: d = {d:.4f}")

    # Count distinct at various depths
    for depth in [3, 6, 9, 12, 15, 18]:
        addrs = [ternary_encode(r["Sk"], r["St"], r["Se"], depth=depth) for r in records]
        unique = len(set(addrs))
        print(f"    Unique addresses at depth {depth:2d}: {unique}/{len(records)}")

    # ---- Step 3: Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: 3D scatter
    ax1 = fig.add_subplot(131, projection='3d')
    for r in records:
        ax1.scatter(r["Sk"], r["St"], r["Se"],
                    c=SECONDARY_STRUCTURES[r["structure"]]["color"],
                    s=120, edgecolors='k', linewidths=0.5, zorder=5)
        ax1.text(r["Sk"], r["St"], r["Se"],
                 f'  {r["structure"][:6]}', fontsize=6, alpha=0.7)
    ax1.set_xlabel('$S_k$', fontsize=10)
    ax1.set_ylabel('$S_t$', fontsize=10)
    ax1.set_zlabel('$S_e$', fontsize=10)
    ax1.set_title('A. Secondary Structures in S-Space', fontsize=10, fontweight='bold')

    # Panel B: Sk vs St projection
    ax2 = axes[1]
    for r in records:
        ax2.scatter(r["Sk"], r["St"],
                    c=SECONDARY_STRUCTURES[r["structure"]]["color"],
                    s=120, edgecolors='k', linewidths=0.5, zorder=5)
        ax2.annotate(r["structure"].replace("_", "\n")[:12],
                     (r["Sk"], r["St"]), fontsize=6,
                     textcoords="offset points", xytext=(5, 5))
    ax2.set_xlabel('$S_k$ (spectral distribution)', fontsize=10)
    ax2.set_ylabel('$S_t$ (timescale span)', fontsize=10)
    ax2.set_title('B. $S_k$ vs $S_t$ Projection', fontsize=10, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel C: Pairwise distance matrix
    ax3 = axes[2]
    n = len(records)
    dist_matrix = np.zeros((n, n))
    for (i, j) in combinations(range(n), 2):
        d = np.sqrt(sum((a - b)**2 for a, b in zip(coords[i], coords[j])))
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d
    im = ax3.imshow(dist_matrix, cmap='viridis_r', aspect='auto')
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    short_names = [r["structure"][:8] for r in records]
    ax3.set_xticklabels(short_names, fontsize=6, rotation=45, ha='right')
    ax3.set_yticklabels(short_names, fontsize=6)
    ax3.set_title('C. Pairwise Distance Matrix', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Euclidean Distance', shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "panel_2_secondary_structure.png"), dpi=200, bbox_inches='tight')
    print(f"\n[3] Saved figure: figures/panel_2_secondary_structure.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    dists = [v["euclidean_distance"] for v in separation.values()]
    print(f"  Structures encoded:          {len(records)}")
    print(f"  Min pairwise distance:       {min(dists):.4f}")
    print(f"  Max pairwise distance:       {max(dists):.4f}")
    depth3_addrs = set(r["ternary_3"] for r in records)
    depth6_addrs = set(r["ternary_6"] for r in records)
    print(f"  Distinct at depth 3:         {len(depth3_addrs)}/{len(records)}")
    print(f"  Distinct at depth 6:         {len(depth6_addrs)}/{len(records)}")


if __name__ == "__main__":
    main()
