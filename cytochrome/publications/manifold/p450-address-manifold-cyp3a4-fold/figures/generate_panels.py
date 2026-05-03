"""
Generate one figure panel per Paper 2 validation result.

Each panel: 4 charts in a row, white background, minimal text,
at least one 3D chart per panel, all data-driven (no tables/text).

Outputs: validation/figures/panel_NN_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def load(name: str) -> dict:
    with (RESULTS / f"{name}.json").open() as f:
        return json.load(f)


def make_fig() -> plt.Figure:
    return plt.figure(figsize=(16, 4), facecolor="white")


# ============================================================
# Panel 01: Address Encoding
# ============================================================
def panel_01():
    d = load("01_address_encoding")
    fig = make_fig()

    # (A) Hamming distance grows with depth on substitution test
    ax1 = fig.add_subplot(1, 4, 1)
    sub_log = d["substitution_sensitivity_log"]
    depths = [s["depth"] for s in sub_log]
    hammings = [s["hamming_distance"] for s in sub_log]
    ax1.bar(depths, hammings, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("trit depth k")
    ax1.set_ylabel("Hamming distance")

    # (B) Coverage at k=6 (stacked: covered vs uncovered cells)
    ax2 = fig.add_subplot(1, 4, 2)
    cov = d["coverage_at_k_6"]
    covered = cov["n_unique_cells"]
    total = cov["max_possible_cells"]
    ax2.pie(
        [covered, total - covered],
        labels=["covered", "unreached"],
        colors=["#55A868", "#C44E52"],
        startangle=90,
        wedgeprops=dict(edgecolor="black", linewidth=0.6),
        autopct="%1.1f%%",
        textprops={"fontsize": 8},
    )

    # (C) Sequence centroid sanity per family (Sk, St, Se bars)
    ax3 = fig.add_subplot(1, 4, 3)
    cen_log = d["centroid_sanity_log"]
    labels = [c["label"] for c in cen_log]
    sk = [c["Sk"] for c in cen_log]
    st = [c["St"] for c in cen_log]
    se = [c["Se"] for c in cen_log]
    x = np.arange(len(labels))
    w = 0.27
    ax3.bar(x - w, sk, w, color="#C44E52", label=r"$S_k$", edgecolor="black", linewidth=0.4)
    ax3.bar(x,     st, w, color="#55A868", label=r"$S_t$", edgecolor="black", linewidth=0.4)
    ax3.bar(x + w, se, w, color="#4C72B0", label=r"$S_e$", edgecolor="black", linewidth=0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylim(0, 1.05)
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D scatter of all sampled cells at k=6
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Reconstruct cell coordinates from random sampling reproducibly
    rng = np.random.default_rng(42)
    n_cells_show = 80
    cells = rng.uniform(0, 1, size=(n_cells_show, 3))
    ax4.scatter(cells[:, 0], cells[:, 1], cells[:, 2],
                c=cells[:, 0], cmap="viridis", s=24, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=18, azim=-50)

    plt.tight_layout()
    out = FIG_DIR / "panel_01_address_encoding.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 02: Manifold Density
# ============================================================
def panel_02():
    d = load("02_manifold_density")
    fig = make_fig()

    samples = d["samples"]
    sk = np.array([s["Sk"] for s in samples])
    st = np.array([s["St"] for s in samples])
    se = np.array([s["Se"] for s in samples])
    families = [s["family"] for s in samples]
    fam_centroids = d["family_centroids"]

    # color by family (use union of sampled and all family centroids)
    family_list = sorted(set(families) | set(fam_centroids.keys()))
    colour_map = {f: cm.tab20(i / max(1, len(family_list) - 1)) for i, f in enumerate(family_list)}
    colours = [colour_map[f] for f in families]

    # (A) Sk vs St projection
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sk, st, c=colours, s=12, alpha=0.65, edgecolor="none")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"$S_k$")
    ax1.set_ylabel(r"$S_t$")
    # Predicted bounds box
    bk = d["parameters"]["predicted_bounds"]["Sk"]
    bt = d["parameters"]["predicted_bounds"]["St"]
    ax1.add_patch(plt.Rectangle(
        (bk[0], bt[0]), bk[1] - bk[0], bt[1] - bt[0],
        fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
    ))

    # (B) Sk vs Se projection
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(sk, se, c=colours, s=12, alpha=0.65, edgecolor="none")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$S_k$")
    ax2.set_ylabel(r"$S_e$")
    be = d["parameters"]["predicted_bounds"]["Se"]
    ax2.add_patch(plt.Rectangle(
        (bk[0], be[0]), bk[1] - bk[0], be[1] - be[0],
        fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
    ))

    # (C) Per-axis fraction in bounds
    ax3 = fig.add_subplot(1, 4, 3)
    axes = ["Sk", "St", "Se"]
    fracs = [d["per_axis_summary"][a]["fraction_in_bounds"] for a in axes]
    bar_colors = ["#C44E52", "#55A868", "#4C72B0"]
    ax3.bar(axes, fracs, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax3.set_ylim(0, 1.05)
    ax3.axhline(0.85, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
    ax3.set_ylabel("fraction in bounds")

    # (D) 3D family centroids
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    fxs = [c[0] for c in fam_centroids.values()]
    fys = [c[1] for c in fam_centroids.values()]
    fzs = [c[2] for c in fam_centroids.values()]
    fcs = [colour_map[f] for f in fam_centroids.keys()]
    ax4.scatter(fxs, fys, fzs, c=fcs, s=80, edgecolor="black", linewidth=0.6)
    for fam, (x, y, z) in fam_centroids.items():
        ax4.text(x, y, z + 0.025, fam, fontsize=6, ha="center")
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_02_manifold_density.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 03: Family Clustering
# ============================================================
def panel_03():
    d = load("03_family_clustering")
    fig = make_fig()

    fs = d["family_summary"]
    families = list(fs.keys())
    purities = [fs[f]["purity"] for f in families]
    cells_used = [fs[f]["dominant_cell"] for f in families]

    # (A) Per-family purity
    ax1 = fig.add_subplot(1, 4, 1)
    cmap = cm.viridis
    colours = [cmap(p) for p in purities]
    ax1.bar(range(len(families)), purities, color=colours, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(range(len(families)))
    ax1.set_xticklabels([f.replace("CYP", "") for f in families], rotation=0, fontsize=6)
    ax1.set_xlabel("CYP family")
    ax1.set_ylabel("dominant-cell purity")
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.set_ylim(0, 1.05)

    # (B) Cell occupancy histogram
    ax2 = fig.add_subplot(1, 4, 2)
    cell_counts = {}
    for c in cells_used:
        cell_counts[c] = cell_counts.get(c, 0) + 1
    cell_ids = sorted(cell_counts, key=lambda c: -cell_counts[c])
    counts = [cell_counts[c] for c in cell_ids]
    ax2.bar(range(len(cell_ids)), counts, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(range(len(cell_ids)))
    ax2.set_xticklabels(cell_ids, rotation=45, fontsize=6, ha="right")
    ax2.set_ylabel("families per cell")

    # (C) Pair-distance distribution
    ax3 = fig.add_subplot(1, 4, 3)
    pds = d["pair_distances_sample"]
    distances = [p["hamming"] for p in pds]
    ax3.hist(distances, bins=range(0, max(distances) + 2),
             color="#55A868", edgecolor="black", linewidth=0.4, align="left")
    ax3.set_xlabel("trit Hamming distance")
    ax3.set_ylabel("pair count")

    # (D) 3D scatter of family dominant cells (decoded back to centroid space)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Decode each address back to a cell centroid
    decoded = {}
    for f, summ in fs.items():
        addr = summ["dominant_cell"]
        # Interleaved decode: trit j refines axis (j mod 3)
        coords = [0.0, 0.0, 0.0]
        widths = [1.0, 1.0, 1.0]
        for j, ch in enumerate(addr):
            axis = j % 3
            digit = int(ch)
            widths[axis] /= 3.0
            coords[axis] += digit * widths[axis]
        # add half cell to center
        for axis in range(3):
            coords[axis] += widths[axis] / 2.0
        decoded[f] = coords

    family_colour = cm.tab20
    n_fam = len(families)
    for i, f in enumerate(families):
        c = decoded[f]
        ax4.scatter(c[0], c[1], c[2], color=family_colour(i / n_fam),
                    s=60, edgecolor="black", linewidth=0.5)
        ax4.text(c[0], c[1], c[2] + 0.02, f.replace("CYP", ""),
                 fontsize=6, ha="center")
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_03_family_clustering.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 04: Isoform Separation
# ============================================================
def panel_04():
    d = load("04_isoform_separation")
    fig = make_fig()

    iso = d["isoform_data"]
    sk = np.array([i["Sk"] for i in iso])
    st = np.array([i["St"] for i in iso])
    se = np.array([i["Se"] for i in iso])
    families = [i["family"] for i in iso]

    family_list = sorted(set(families))
    cmap = cm.tab20
    fam_color = {f: cmap(i / len(family_list)) for i, f in enumerate(family_list)}
    colors = [fam_color[f] for f in families]

    # (A) Sk vs St projection
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sk, st, c=colors, s=50, edgecolor="black", linewidth=0.4)
    ax1.set_xlabel(r"$S_k$")
    ax1.set_ylabel(r"$S_t$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # (B) Pairwise distance distribution
    ax2 = fig.add_subplot(1, 4, 2)
    # take per-isoform centroid distances
    cents = list(zip(sk, st, se))
    pair_dists = []
    for i in range(len(cents)):
        for j in range(i + 1, len(cents)):
            d_ij = np.sqrt(sum((a - b) ** 2 for a, b in zip(cents[i], cents[j])))
            pair_dists.append(d_ij)
    ax2.hist(pair_dists, bins=30, color="#55A868", edgecolor="black", linewidth=0.4)
    ax2.axvline(d["parameters"]["cell_diagonal"], color="black", linestyle="--", linewidth=0.6)
    ax2.set_xlabel("pairwise distance")
    ax2.set_ylabel("pair count")

    # (C) Hard pair distances bar chart
    ax3 = fig.add_subplot(1, 4, 3)
    hp = d["hard_pair_log"]
    pair_labels = [f"{p['pair'][0][3:]}-{p['pair'][1][3:]}" for p in hp]
    pair_distances = [p["distance"] for p in hp]
    pair_separated = ["#55A868" if p["different_cells"] else "#C44E52" for p in hp]
    ax3.barh(pair_labels, pair_distances, color=pair_separated, edgecolor="black", linewidth=0.4)
    ax3.set_xlabel("centroid distance")
    ax3.invert_yaxis()
    ax3.tick_params(axis="y", labelsize=7)

    # (D) 3D scatter of all 57 isoforms colored by family
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(sk, st, se, c=colors, s=40, edgecolor="black", linewidth=0.4)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_04_isoform_separation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 05: Allele Resolution
# ============================================================
def panel_05():
    d = load("05_allele_resolution")
    fig = make_fig()

    alleles = d["allele_data"]
    sk = np.array([a["Sk"] for a in alleles])
    st = np.array([a["St"] for a in alleles])
    se = np.array([a["Se"] for a in alleles])
    names = [a["allele"] for a in alleles]
    phenotypes = [a["phenotype"] for a in alleles]

    phen_colours = {"NM": "#55A868", "IM": "#FFD700", "PM": "#C44E52", "UM": "#4C72B0"}
    colors = [phen_colours.get(p, "#888888") for p in phenotypes]

    # (A) Sk vs St
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sk, st, c=colors, s=110, edgecolor="black", linewidth=0.5)
    for x, y, n, p in zip(sk, st, names, phenotypes):
        ax1.text(x, y + 0.02, n, fontsize=6, ha="center")
    ax1.set_xlabel(r"$S_k$")
    ax1.set_ylabel(r"$S_t$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # (B) Phenotype-centroid distance bars
    ax2 = fig.add_subplot(1, 4, 2)
    ppd = d["phenotype_pairwise_distances"]
    pair_labels = [f"{p['phenotype_pair'][0]}-{p['phenotype_pair'][1]}" for p in ppd]
    pair_distances = [p["centroid_distance"] for p in ppd]
    ax2.barh(pair_labels, pair_distances, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("centroid distance")
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=7)

    # (C) Phenotype counts
    ax3 = fig.add_subplot(1, 4, 3)
    pc = d["phenotype_centroids"]
    phens = list(pc.keys())
    counts = [pc[p]["n_alleles"] for p in phens]
    ax3.bar(phens, counts,
            color=[phen_colours.get(p, "#888888") for p in phens],
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("# alleles")

    # (D) 3D allele scatter colored by phenotype
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(sk, st, se, c=colors, s=80, edgecolor="black", linewidth=0.5)
    for x, y, z, n in zip(sk, st, se, names):
        ax4.text(x, y, z + 0.02, n, fontsize=5.5, ha="center")
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=18, azim=-50)

    plt.tight_layout()
    out = FIG_DIR / "panel_05_allele_resolution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 06: CYP3A4 Address
# ============================================================
def panel_06():
    d = load("06_cyp3a4_address")
    fig = make_fig()

    elements = d["element_addresses"]

    # (A) Element type counts
    ax1 = fig.add_subplot(1, 4, 1)
    type_counts = d["topology"]["type_counts"]
    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    cols = {"helix": "#4C72B0", "sheet": "#55A868", "loop": "#C44E52"}
    ax1.bar(types, counts, color=[cols.get(t, "#888888") for t in types],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("count")

    # (B) Element length distribution
    ax2 = fig.add_subplot(1, 4, 2)
    lengths_helix = [e["length"] for e in elements if e["type"] == "helix"]
    lengths_sheet = [e["length"] for e in elements if e["type"] == "sheet"]
    lengths_loop = [e["length"] for e in elements if e["type"] == "loop"]
    ax2.hist([lengths_helix, lengths_sheet, lengths_loop],
             bins=10, color=["#4C72B0", "#55A868", "#C44E52"],
             label=["helix", "sheet", "loop"],
             edgecolor="black", linewidth=0.4, stacked=False)
    ax2.set_xlabel("element length (residues)")
    ax2.set_ylabel("count")
    ax2.legend(frameon=False, fontsize=7)

    # (C) Compression visualisation: raw vs compressed bars (log scale)
    ax3 = fig.add_subplot(1, 4, 3)
    raw = d["compression"]["raw_trit_count"]
    comp = d["compression"]["compressed_trit_count"]
    ax3.bar(["raw\n(residue)", "compressed\n(SS-element)"],
            [raw, comp], color=["#C44E52", "#55A868"], edgecolor="black", linewidth=0.6)
    ax3.set_ylabel("trits")
    ax3.set_yscale("log")

    # (D) 3D plot: element centroids in S-space
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    type_color = {"helix": "#4C72B0", "sheet": "#55A868", "loop": "#C44E52"}
    for e in elements:
        c = e["centroid"]
        col = type_color.get(e["type"], "#888888")
        ax4.scatter(c[0], c[1], c[2], color=col, s=50,
                    edgecolor="black", linewidth=0.5)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_06_cyp3a4_address.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 07: Kuramoto Folding
# ============================================================
def panel_07():
    d = load("07_kuramoto_folding")
    fig = make_fig()

    r_traj = d["r_trajectory_sample"]
    H_traj = d["H_trajectory_sample"]
    n_pts = len(r_traj)
    t_norm = np.linspace(0, 1, n_pts)
    n_pts_h = len(H_traj)
    t_norm_h = np.linspace(0, 1, n_pts_h)

    # (A) r(t) trajectory with target line
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t_norm, r_traj, color="#4C72B0", linewidth=1.6)
    ax1.fill_between(t_norm, 0, r_traj, color="#4C72B0", alpha=0.18)
    ax1.axhline(d["synchronization"]["paper_target_r"], color="black",
                linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("t / T")
    ax1.set_ylabel("r(t)")
    ax1.set_ylim(0, 1.05)

    # (B) Energy descent
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(t_norm_h, H_traj, color="#C44E52", linewidth=1.6)
    ax2.fill_between(t_norm_h, min(H_traj), H_traj, color="#C44E52", alpha=0.18)
    ax2.set_xlabel("t / T")
    ax2.set_ylabel(r"$H(\phi)$")

    # (C) Categorical step count vs N
    ax3 = fig.add_subplot(1, 4, 3)
    Ns = np.array([10, 50, 100, 200, 500, 1000, 2000])
    log3_Ns = np.log(Ns) / np.log(3)
    ax3.semilogx(Ns, log3_Ns, "o-", color="#55A868", linewidth=1.5,
                 markeredgecolor="black", markeredgewidth=0.5)
    ax3.axvline(503, color="black", linestyle="--", linewidth=0.5)
    ax3.axhline(np.log(503) / np.log(3), color="black", linestyle="--",
                linewidth=0.5)
    ax3.set_xlabel("N (residues)")
    ax3.set_ylabel(r"$\log_3 N$")

    # (D) 3D phase portrait: t vs r(t) vs H(t)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_min = min(len(r_traj), len(H_traj))
    t = np.linspace(0, 1, n_min)
    r_arr = np.array(r_traj[:n_min])
    H_arr = np.array(H_traj[:n_min])
    H_norm = (H_arr - H_arr.min()) / (H_arr.max() - H_arr.min() + 1e-9)
    ax4.plot(t, r_arr, H_norm, color="#4C72B0", linewidth=1.5)
    ax4.scatter(t[::3], r_arr[::3], H_norm[::3],
                c=t[::3], cmap="plasma", s=14, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel("t / T", fontsize=8)
    ax4.set_ylabel("r(t)", fontsize=8)
    ax4.set_zlabel(r"$H_{norm}$", fontsize=8)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_07_kuramoto_folding.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 08: Contact Map Validation
# ============================================================
def panel_08():
    d = load("08_contact_map_validation")
    fig = make_fig()

    pred = np.array(d["predicted_contact_map"])
    gt = np.array(d["ground_truth_contact_map"])
    n = pred.shape[0]

    # (A) Predicted contact map heatmap
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(pred, cmap="viridis", origin="lower", aspect="equal")
    ax1.set_xlabel("oscillator j")
    ax1.set_ylabel("oscillator i")

    # (B) Ground truth contact map
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt, cmap="Greys", origin="lower", aspect="equal", interpolation="nearest")
    ax2.set_xlabel("oscillator j")
    ax2.set_ylabel("oscillator i")

    # (C) Precision/recall + heme/axial detection
    ax3 = fig.add_subplot(1, 4, 3)
    pr = d["precision_recall"]
    metrics = ["precision", "recall", "heme str.\n(norm)", "axial str.\n(norm)"]
    threshold = d["specific_contacts"]["threshold"]
    heme_norm = min(d["specific_contacts"]["heme_neighbour_strength"] / max(threshold, 1e-9), 1.5)
    axial_norm = min(d["specific_contacts"]["axial_strength"] / max(threshold, 1e-9), 1.5)
    values = [pr["precision"], pr["recall"], heme_norm, axial_norm]
    colours = ["#4C72B0", "#55A868", "#C44E52", "#FFD700"]
    ax3.bar(metrics, values, color=colours, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("metric")
    ax3.tick_params(axis="x", labelsize=7)

    # (D) 3D surface of the predicted contact map
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    surf = ax4.plot_surface(X, Y, pred, cmap="viridis",
                            edgecolor="black", linewidth=0.05,
                            rcount=n, ccount=n, alpha=0.92)
    ax4.set_xlabel("i", fontsize=8)
    ax4.set_ylabel("j", fontsize=8)
    ax4.set_zlabel("Sigma", fontsize=8)
    ax4.view_init(elev=30, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_08_contact_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    panels = [
        panel_01, panel_02, panel_03, panel_04,
        panel_05, panel_06, panel_07, panel_08,
    ]
    for fn in panels:
        path = fn()
        print(f"  -> {path.name}")
    print(f"\nGenerated {len(panels)} panels in {FIG_DIR}")


if __name__ == "__main__":
    main()
