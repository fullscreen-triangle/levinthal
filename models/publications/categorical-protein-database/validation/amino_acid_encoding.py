"""
Validation 1: Amino Acid S-Entropy Encoding and Ternary Addressing
==================================================================
Computes S-entropy coordinates (Sk, St, Se) for all 20 standard amino acids
from physicochemical properties, generates ternary addresses at multiple depths,
and validates resolution, chemical family clustering, and pairwise distances.

Outputs:
  results/amino_acid_coordinates.csv
  results/amino_acid_ternary_addresses.csv
  results/pairwise_distances.csv
  results/resolution_by_depth.csv
  results/chemical_family_clustering.json
  figures/panel_1_sentropy_space.png
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
# Raw physicochemical data for 20 standard amino acids
# h: Kyte-Doolittle hydrophobicity index
# v: van der Waals volume (Å³) from Creighton 1993
# e: Electrostatic index (charge + polarity, normalized to [0,1])
# ============================================================================
AMINO_ACIDS = [
    {"name": "Isoleucine",     "code3": "Ile", "code1": "I", "h":  4.5, "v": 166.7, "e": 0.00, "class": "nonpolar"},
    {"name": "Valine",         "code3": "Val", "code1": "V", "h":  4.2, "v": 140.0, "e": 0.00, "class": "nonpolar"},
    {"name": "Leucine",        "code3": "Leu", "code1": "L", "h":  3.8, "v": 166.7, "e": 0.00, "class": "nonpolar"},
    {"name": "Phenylalanine",  "code3": "Phe", "code1": "F", "h":  2.8, "v": 189.9, "e": 0.00, "class": "aromatic"},
    {"name": "Cysteine",       "code3": "Cys", "code1": "C", "h":  2.5, "v": 108.5, "e": 0.10, "class": "polar"},
    {"name": "Methionine",     "code3": "Met", "code1": "M", "h":  1.9, "v": 162.9, "e": 0.00, "class": "nonpolar"},
    {"name": "Alanine",        "code3": "Ala", "code1": "A", "h":  1.8, "v":  88.6, "e": 0.00, "class": "nonpolar"},
    {"name": "Glycine",        "code3": "Gly", "code1": "G", "h": -0.4, "v":  60.1, "e": 0.00, "class": "nonpolar"},
    {"name": "Threonine",      "code3": "Thr", "code1": "T", "h": -0.7, "v": 116.1, "e": 0.30, "class": "polar"},
    {"name": "Serine",         "code3": "Ser", "code1": "S", "h": -0.8, "v":  89.0, "e": 0.30, "class": "polar"},
    {"name": "Tryptophan",     "code3": "Trp", "code1": "W", "h": -0.9, "v": 227.8, "e": 0.10, "class": "aromatic"},
    {"name": "Tyrosine",       "code3": "Tyr", "code1": "Y", "h": -1.3, "v": 193.6, "e": 0.20, "class": "aromatic"},
    {"name": "Proline",        "code3": "Pro", "code1": "P", "h": -1.6, "v": 122.7, "e": 0.00, "class": "nonpolar"},
    {"name": "Histidine",      "code3": "His", "code1": "H", "h": -3.2, "v": 153.2, "e": 0.60, "class": "positive"},
    {"name": "Asparagine",     "code3": "Asn", "code1": "N", "h": -3.5, "v": 114.1, "e": 0.50, "class": "polar"},
    {"name": "Glutamine",      "code3": "Gln", "code1": "Q", "h": -3.5, "v": 143.8, "e": 0.50, "class": "polar"},
    {"name": "Aspartate",      "code3": "Asp", "code1": "D", "h": -3.5, "v": 111.1, "e": 1.00, "class": "negative"},
    {"name": "Glutamate",      "code3": "Glu", "code1": "E", "h": -3.5, "v": 138.4, "e": 1.00, "class": "negative"},
    {"name": "Lysine",         "code3": "Lys", "code1": "K", "h": -3.9, "v": 168.6, "e": 1.00, "class": "positive"},
    {"name": "Arginine",       "code3": "Arg", "code1": "R", "h": -4.5, "v": 173.4, "e": 1.00, "class": "positive"},
]

# Normalization bounds
H_MIN, H_MAX = -4.5, 4.5
V_MIN, V_MAX = 60.1, 227.8
E_MIN, E_MAX = 0.0, 1.0


def compute_sentropy(aa):
    """Compute S-entropy coordinates from physicochemical properties."""
    Sk = (aa["h"] - H_MIN) / (H_MAX - H_MIN)
    St = (aa["v"] - V_MIN) / (V_MAX - V_MIN)
    Se = (aa["e"] - E_MIN) / (E_MAX - E_MIN)
    return np.clip(Sk, 0, 1), np.clip(St, 0, 1), np.clip(Se, 0, 1)


def ternary_encode(Sk, St, Se, depth=18):
    """Generate interleaved ternary address from S-entropy coordinates."""
    remainders = [Sk, St, Se]
    trits = []
    for j in range(depth):
        dim = j % 3
        t = int(3 * remainders[dim])
        t = min(t, 2)  # clamp
        remainders[dim] = 3 * remainders[dim] - t
        trits.append(t)
    return trits


def trit_string(trits):
    """Convert trit list to string."""
    return ''.join(str(t) for t in trits)


def shared_prefix_depth(t1, t2):
    """Compute length of longest common prefix between two trit strings."""
    for i in range(min(len(t1), len(t2))):
        if t1[i] != t2[i]:
            return i
    return min(len(t1), len(t2))


def euclidean_distance(c1, c2):
    """Euclidean distance in S-entropy space."""
    return np.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))


def main():
    print("=" * 70)
    print("VALIDATION 1: Amino Acid S-Entropy Encoding")
    print("=" * 70)

    # ---- Step 1: Compute S-entropy coordinates ----
    records = []
    for aa in AMINO_ACIDS:
        Sk, St, Se = compute_sentropy(aa)
        trits = ternary_encode(Sk, St, Se, depth=18)
        records.append({
            "name": aa["name"],
            "code3": aa["code3"],
            "code1": aa["code1"],
            "h": aa["h"],
            "v": aa["v"],
            "e": aa["e"],
            "class": aa["class"],
            "Sk": round(Sk, 4),
            "St": round(St, 4),
            "Se": round(Se, 4),
            "ternary_18": trit_string(trits),
            "ternary_12": trit_string(trits[:12]),
            "ternary_9": trit_string(trits[:9]),
            "ternary_6": trit_string(trits[:6]),
            "ternary_3": trit_string(trits[:3]),
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "amino_acid_coordinates.csv"), index=False)
    print(f"\n[1] Computed S-entropy coordinates for {len(df)} amino acids")
    print(df[["code3", "Sk", "St", "Se", "class", "ternary_9"]].to_string(index=False))

    # ---- Step 2: Pairwise distances ----
    names = [r["code3"] for r in records]
    coords = [(r["Sk"], r["St"], r["Se"]) for r in records]
    trits_18 = [r["ternary_18"] for r in records]

    pair_records = []
    for (i, j) in combinations(range(len(records)), 2):
        d = euclidean_distance(coords[i], coords[j])
        spd = shared_prefix_depth(trits_18[i], trits_18[j])
        pair_records.append({
            "aa1": names[i],
            "aa2": names[j],
            "distance": round(d, 6),
            "shared_prefix_depth": spd,
        })

    df_pairs = pd.DataFrame(pair_records)
    df_pairs.to_csv(os.path.join(RESULTS_DIR, "pairwise_distances.csv"), index=False)
    print(f"\n[2] Computed {len(df_pairs)} pairwise distances")
    print(f"    Min distance: {df_pairs['distance'].min():.4f} ({df_pairs.loc[df_pairs['distance'].idxmin(), 'aa1']}-{df_pairs.loc[df_pairs['distance'].idxmin(), 'aa2']})")
    print(f"    Max distance: {df_pairs['distance'].max():.4f}")
    print(f"    Mean distance: {df_pairs['distance'].mean():.4f}")

    # ---- Step 3: Resolution by trit depth ----
    resolution_records = []
    for depth in [3, 6, 9, 12, 15, 18]:
        addrs = [trit_string(ternary_encode(*coords[i], depth=depth)) for i in range(len(records))]
        unique_addrs = len(set(addrs))
        pairs_resolved = 0
        total_pairs = 0
        for (i, j) in combinations(range(len(records)), 2):
            total_pairs += 1
            if addrs[i] != addrs[j]:
                pairs_resolved += 1
        cell_width = 3 ** (-(depth // 3))
        resolution_records.append({
            "depth": depth,
            "unique_addresses": unique_addrs,
            "pairs_resolved": pairs_resolved,
            "total_pairs": total_pairs,
            "resolution_pct": round(100 * pairs_resolved / total_pairs, 2),
            "cell_width": round(cell_width, 6),
        })

    df_res = pd.DataFrame(resolution_records)
    df_res.to_csv(os.path.join(RESULTS_DIR, "resolution_by_depth.csv"), index=False)
    print(f"\n[3] Resolution by trit depth:")
    print(df_res.to_string(index=False))

    # ---- Step 4: Chemical family clustering at depth 3 ----
    classes = sorted(set(r["class"] for r in records))
    depth3_addrs = {r["code3"]: r["ternary_3"] for r in records}
    class_map = {r["code3"]: r["class"] for r in records}

    # Compute intra-class vs inter-class mean shared prefix depth
    clustering = {}
    for cls in classes:
        members = [k for k, v in class_map.items() if v == cls]
        if len(members) < 2:
            continue
        intra_spd = []
        inter_spd = []
        for (a, b) in combinations(names, 2):
            spd = shared_prefix_depth(
                ternary_encode(*coords[names.index(a)], depth=18),
                ternary_encode(*coords[names.index(b)], depth=18)
            )
            if class_map[a] == cls and class_map[b] == cls:
                intra_spd.append(spd)
            elif class_map[a] == cls or class_map[b] == cls:
                inter_spd.append(spd)

        mean_intra = np.mean(intra_spd) if intra_spd else 0
        mean_inter = np.mean(inter_spd) if inter_spd else 0
        R = mean_intra / mean_inter if mean_inter > 0 else float('inf')
        clustering[cls] = {
            "members": members,
            "n_members": len(members),
            "mean_intra_spd": round(float(mean_intra), 3),
            "mean_inter_spd": round(float(mean_inter), 3),
            "cohesion_ratio_R": round(float(R), 3),
            "cohesive": bool(R > 1.0),
        }

    with open(os.path.join(RESULTS_DIR, "chemical_family_clustering.json"), 'w') as f:
        json.dump(clustering, f, indent=2)

    print(f"\n[4] Chemical family clustering (cohesion ratio R):")
    cohesive_count = 0
    for cls, data in sorted(clustering.items()):
        status = "PASS" if data["cohesive"] else "FAIL"
        if data["cohesive"]:
            cohesive_count += 1
        print(f"    {cls:12s}: R = {data['cohesion_ratio_R']:.3f}  intra={data['mean_intra_spd']:.2f}  inter={data['mean_inter_spd']:.2f}  [{status}]")
    print(f"    Cohesive families: {cohesive_count}/{len(clustering)}")

    # ---- Step 5: Generate 4-panel figure ----
    fig = plt.figure(figsize=(16, 12))

    # Color map for classes
    class_colors = {
        "nonpolar": "#2196F3",
        "polar": "#4CAF50",
        "positive": "#F44336",
        "negative": "#FF9800",
        "aromatic": "#9C27B0",
    }
    colors = [class_colors[r["class"]] for r in records]

    # Panel A: 3D S-entropy space
    ax1 = fig.add_subplot(221, projection='3d')
    Sks = [r["Sk"] for r in records]
    Sts = [r["St"] for r in records]
    Ses = [r["Se"] for r in records]
    ax1.scatter(Sks, Sts, Ses, c=colors, s=80, edgecolors='k', linewidths=0.5, alpha=0.9)
    for i, r in enumerate(records):
        ax1.text(r["Sk"], r["St"], r["Se"], f' {r["code1"]}', fontsize=6, alpha=0.7)
    ax1.set_xlabel('$S_k$ (hydrophobicity)', fontsize=9)
    ax1.set_ylabel('$S_t$ (volume)', fontsize=9)
    ax1.set_zlabel('$S_e$ (electrostatic)', fontsize=9)
    ax1.set_title('A. Amino Acids in S-Entropy Space', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)

    # Panel B: Resolution curve
    ax2 = fig.add_subplot(222)
    ax2.plot(df_res["depth"], df_res["resolution_pct"], 'o-', color='#1565C0', linewidth=2, markersize=8)
    ax2.axhline(y=100, color='grey', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trit Depth k', fontsize=10)
    ax2.set_ylabel('Pairs Resolved (%)', fontsize=10)
    ax2.set_title('B. Resolution vs Trit Depth', fontsize=10, fontweight='bold')
    ax2.set_ylim(70, 102)
    ax2.grid(True, alpha=0.3)
    for _, row in df_res.iterrows():
        ax2.annotate(f"{row['unique_addresses']}/20",
                     (row['depth'], row['resolution_pct']),
                     textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

    # Panel C: Pairwise similarity matrix
    ax3 = fig.add_subplot(223)
    n = len(records)
    sim_matrix = np.zeros((n, n))
    for _, row in df_pairs.iterrows():
        i = names.index(row["aa1"])
        j = names.index(row["aa2"])
        sim_matrix[i, j] = row["shared_prefix_depth"]
        sim_matrix[j, i] = row["shared_prefix_depth"]
    np.fill_diagonal(sim_matrix, 18)
    im = ax3.imshow(sim_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels([r["code1"] for r in records], fontsize=7)
    ax3.set_yticklabels([r["code1"] for r in records], fontsize=7)
    ax3.set_title('C. Pairwise Ternary Similarity', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Shared Prefix Depth', shrink=0.8)

    # Panel D: Chemical family cohesion
    ax4 = fig.add_subplot(224)
    cls_names = sorted(clustering.keys())
    x = np.arange(len(cls_names))
    intra_vals = [clustering[c]["mean_intra_spd"] for c in cls_names]
    inter_vals = [clustering[c]["mean_inter_spd"] for c in cls_names]
    w = 0.35
    bars1 = ax4.bar(x - w/2, intra_vals, w, label='Intra-class', color='#1565C0', alpha=0.8)
    bars2 = ax4.bar(x + w/2, inter_vals, w, label='Inter-class', color='#BDBDBD', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cls_names, fontsize=8, rotation=30, ha='right')
    ax4.set_ylabel('Mean Shared Prefix Depth', fontsize=10)
    ax4.set_title('D. Chemical Family Cohesion', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    for i, c in enumerate(cls_names):
        R = clustering[c]["cohesion_ratio_R"]
        ax4.text(i, max(intra_vals[i], inter_vals[i]) + 0.2, f'R={R:.1f}', ha='center', fontsize=7)
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "panel_1_sentropy_space.png"), dpi=200, bbox_inches='tight')
    print(f"\n[5] Saved figure: figures/panel_1_sentropy_space.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Amino acids encoded:         {len(records)}")
    print(f"  Full resolution at depth:    {df_res[df_res['resolution_pct'] == 100]['depth'].min()}")
    print(f"  Min pairwise distance:       {df_pairs['distance'].min():.4f}")
    print(f"  Cohesive families:           {cohesive_count}/{len(clustering)}")
    print(f"  Results saved to:            validation/results/")
    print(f"  Figure saved to:             figures/")


if __name__ == "__main__":
    main()
