"""
Generate Paper 2.5 panels: 4 charts each (>=1 3D), white background.

Each panel reads its source JSON from
    ../validation/results/<validation_id>.json
and writes
    panel_NN_<short_name>.png
into this directory.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# We also need to load real GLB data for some panels.
# Add cytochrome/glb to sys.path so we can import levinthal_glb.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                # paper directory
RESULTS = ROOT / "validation" / "results"
GLB_DIR = HERE.parents[3] / "glb"                  # cytochrome/glb/
sys.path.insert(0, str(GLB_DIR))


plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load(name: str) -> dict:
    return json.load((RESULTS / f"{name}.json").open())


def make_fig():
    return plt.figure(figsize=(16, 4), facecolor="white")


# Common colours
C_GLB = {
    "atomistic": "#4C72B0",
    "ribbon1":   "#FFA500",
    "ribbon2":   "#C44E52",
    "highlight": "#55A868",
}

ELEMENT_COLORS = {
    "H":  "#FFFFFF", "C":  "#909090", "N":  "#3050F8", "O":  "#FF0D0D",
    "S":  "#FFFF30", "P":  "#FF8000", "Fe": "#E06633", "X":  "#9370DB",
    "?":  "#888888",
}


# ============================================================
# Panel 01 — Parser smoke test
# ============================================================
def panel_01():
    d = load("01_glb_parser_smoke")
    fig = make_fig()
    per = d["per_glb"]
    names = list(per.keys())
    short = ["Ribbon 1\n(haem highlit)",
             "Atomistic\n(P450/O2/drug)",
             "Ribbon 2\n(cytochrome C)"]
    counts = [per[n]["n_positioned_objects"] for n in names]

    # (A) Positioned objects per GLB
    ax1 = fig.add_subplot(1, 4, 1)
    bars = ax1.bar(short, counts, color=[C_GLB["ribbon1"], C_GLB["atomistic"],
                                          C_GLB["ribbon2"]],
                   edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 4, str(c), ha="center", fontsize=8)
    ax1.set_ylabel("# positioned objects")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.tick_params(axis="x", labelsize=7)

    # (B) Metadata coverage as stacked dots
    ax2 = fig.add_subplot(1, 4, 2)
    keys = ["author", "license", "source"]
    for i, name in enumerate(names):
        for j, k in enumerate(keys):
            ok = per[name][f"has_{k}"]
            ax2.scatter([j], [i], color="#55A868" if ok else "#CCCCCC",
                        s=200, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, fontsize=8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(["Rib 1", "Atom", "Rib 2"], fontsize=7)
    ax2.set_xlim(-0.5, len(keys) - 0.5)
    ax2.set_ylim(-0.5, len(names) - 0.5)
    ax2.set_title("metadata fields present", fontsize=9)
    ax2.invert_yaxis()

    # (C) checks pass/fail
    ax3 = fig.add_subplot(1, 4, 3)
    checks = d["checks"]
    cnames = list(checks.keys())
    cvals  = [1 if v else 0 for v in checks.values()]
    short_c = [c.replace("_", "\n") for c in cnames]
    ax3.barh(range(len(cnames)),  cvals,
             color=["#55A868" if v else "#C44E52" for v in cvals],
             edgecolor="black", linewidth=0.5)
    ax3.set_yticks(range(len(cnames)))
    ax3.set_yticklabels(short_c, fontsize=6)
    ax3.set_xlim(0, 1.2)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["fail", "pass"], fontsize=8)

    # (D) 3D bar chart: object count per GLB across three "label" columns
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    xs = np.arange(len(names))
    ys = np.zeros(len(names))
    zs = np.zeros(len(names))
    dx = 0.6
    dy = 0.6
    for i, (n, c) in enumerate(zip(names,
                                    [C_GLB["ribbon1"], C_GLB["atomistic"], C_GLB["ribbon2"]])):
        ax4.bar3d(i, 0, 0, dx, dy, per[n]["n_positioned_objects"],
                  color=c, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(["Rib 1", "Atom", "Rib 2"], fontsize=7)
    ax4.set_yticks([])
    ax4.set_zlabel("# positioned objects", fontsize=8)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_01_glb_parser_smoke.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 02 — CPK colour decoder
# ============================================================
def panel_02():
    d = load("02_cpk_color_decoder")
    fig = make_fig()

    # Reconstruct a CPK reference table for plotting
    elements = list(ELEMENT_COLORS.keys())
    rgbs = []
    decoded_correct = []
    for e in elements:
        c = ELEMENT_COLORS[e].lstrip("#")
        rgbs.append(tuple(int(c[i:i+2], 16) for i in (0, 2, 4)))
        # Was this element correctly decoded in the test?
        decoded_correct.append(any(p["element"] == e and p["match"]
                                   for p in []))   # filled below

    # (A) Reference colour swatches (visual table)
    ax1 = fig.add_subplot(1, 4, 1)
    n = len(elements)
    for i, (e, c) in enumerate(zip(elements, [ELEMENT_COLORS[e] for e in elements])):
        ax1.add_patch(plt.Rectangle((0, n - 1 - i), 1, 0.85,
                                    facecolor=c, edgecolor="black"))
        ax1.text(1.15, n - 1 - i + 0.35, e, fontsize=10, va="center")
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, n)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("CPK reference swatches", fontsize=9)
    ax1.spines[:].set_visible(False)

    # (B) Exact / alias / perturbed / out-of-tolerance summary
    ax2 = fig.add_subplot(1, 4, 2)
    cats = ["Reference\n(exact)", "Aliases", "Perturbed\n(±15)", "Out-of-tol"]
    vals = [
        d["exact_match_summary"]["correct"],
        sum(1 for a in d["alias_results"] if a["match"]),
        sum(1 for a in d["perturbation_results"] if a["match"]),
        sum(1 for a in d["out_of_tolerance_results"] if a["decoded"] == "?"),
    ]
    totals = [
        d["exact_match_summary"]["total"],
        len(d["alias_results"]),
        len(d["perturbation_results"]),
        len(d["out_of_tolerance_results"]),
    ]
    rates = [v / t if t > 0 else 0 for v, t in zip(vals, totals)]
    ax2.bar(cats, rates, color=["#55A868"] * 4, edgecolor="black", linewidth=0.5)
    for i, (v, t) in enumerate(zip(vals, totals)):
        ax2.text(i, rates[i] + 0.02, f"{v}/{t}", ha="center", fontsize=8)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("decode success rate")
    ax2.tick_params(axis="x", labelsize=7)

    # (C) vdW radius bar
    ax3 = fig.add_subplot(1, 4, 3)
    radii = d["vdw_radii_table"]
    elems_sorted = sorted(radii.keys(), key=lambda k: radii[k])
    rs = [radii[e] for e in elems_sorted]
    ax3.barh(elems_sorted, rs,
             color=[ELEMENT_COLORS.get(e, "#888888") for e in elems_sorted],
             edgecolor="black", linewidth=0.4)
    ax3.set_xlabel("vdW radius (Å)")
    ax3.tick_params(axis="y", labelsize=6)

    # (D) 3D scatter of all 25 reference colours in RGB cube
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    from levinthal_glb.cpk import CPK_COLORS  # noqa: E402
    for elem, rgb in CPK_COLORS.items():
        rgb_norm = [c / 255 for c in rgb]
        ax4.scatter(rgb[0], rgb[1], rgb[2], color=rgb_norm,
                    s=80, edgecolor="black", linewidth=0.4)
        if elem in ("Fe", "S", "O", "N", "C", "H", "P"):
            ax4.text(rgb[0], rgb[1], rgb[2] + 8, elem,
                     fontsize=7, ha="center")
    ax4.set_xlabel("R", fontsize=8)
    ax4.set_ylabel("G", fontsize=8)
    ax4.set_zlabel("B", fontsize=8)
    ax4.set_xlim(0, 255); ax4.set_ylim(0, 255); ax4.set_zlim(0, 255)
    ax4.view_init(elev=22, azim=-55)
    ax4.set_title("RGB cube placements", fontsize=9)

    plt.tight_layout()
    out = HERE / "panel_02_cpk_color_decoder.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 03 — Artifact filtering
# ============================================================
def panel_03():
    d = load("03_artifact_filtering")
    fig = make_fig()
    per = d["per_glb"]
    names = list(per.keys())
    short = ["Rib 1", "Atom", "Rib 2"]

    # (A) raw vs filtered counts per GLB
    ax1 = fig.add_subplot(1, 4, 1)
    raw    = [per[n]["raw"]["n_atoms"]      for n in names]
    filt   = [per[n]["filtered"]["n_atoms"] for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax1.bar(x - w/2, raw,  w, label="raw",      color="#CCCCCC",
            edgecolor="black", linewidth=0.4)
    ax1.bar(x + w/2, filt, w, label="filtered", color=C_GLB["atomistic"],
            edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(short, fontsize=8)
    ax1.set_ylabel("atom count")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.legend(frameon=False, fontsize=7)

    # (B) Sphere-size distribution before/after for atomistic GLB
    ax2 = fig.add_subplot(1, 4, 2)
    atom = per["model_of_cytochrome_p450__oxygen__drug_complex.glb"]
    ax2.bar(["raw\nmax", "filt\nmax"],
            [atom["raw"]["max_sphere_size"], atom["filtered"]["max_sphere_size"]],
            color=["#CCCCCC", C_GLB["atomistic"]],
            edgecolor="black", linewidth=0.5)
    ax2.axhline(5.0, color="red", linestyle="--", linewidth=0.8,
                label="filter threshold")
    ax2.set_ylabel("max sphere size (Å)")
    ax2.legend(frameon=False, fontsize=7)
    ax2.tick_params(axis="x", labelsize=7)

    # (C) Idempotence check (filt vs filt-of-filt)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.bar(["filtered", "filtered ×2"],
            [atom["filtered"]["n_atoms"], atom["twice"]["n_atoms"]],
            color=[C_GLB["atomistic"], C_GLB["highlight"]],
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("atom count")
    ax3.set_title("idempotence", fontsize=9)

    # (D) 3D scatter of real atomistic positions, filtered
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    from levinthal_glb import parse_glb  # noqa: E402
    s = parse_glb(GLB_DIR / "model_of_cytochrome_p450__oxygen__drug_complex.glb")
    s_filt = s.filter_oversized(max_size=5.0)
    P = np.array([a.position for a in s_filt.atoms
                  if not (a.position[0] == 0 and a.position[1] == 0
                          and a.position[2] == 0)])
    elems = [a.element for a in s_filt.atoms
             if not (a.position[0] == 0 and a.position[1] == 0
                     and a.position[2] == 0)]
    cols = [ELEMENT_COLORS.get(e, "#888888") for e in elems]
    ax4.scatter(P[:, 0], P[:, 1], P[:, 2], c=cols,
                s=15, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel("x (Å)", fontsize=8)
    ax4.set_ylabel("y (Å)", fontsize=8)
    ax4.set_zlabel("z (Å)", fontsize=8)
    ax4.set_title("filtered atoms ({} total)".format(len(P)), fontsize=9)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_03_artifact_filtering.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 04 — Iron coordination shell
# ============================================================
def panel_04():
    d = load("04_iron_coordination_shell")
    fig = make_fig()
    nbrs = d["first_shell_neighbours"]
    ranges = d["ranges_used"]

    # (A) Distance histogram per element (first shell)
    ax1 = fig.add_subplot(1, 4, 1)
    for n in nbrs:
        c = ELEMENT_COLORS.get(n["element"], "#888888")
        ax1.scatter(n["distance_A"], n["element"], color=c, s=60,
                    edgecolor="black", linewidth=0.4)
    # range bars
    ax1.axvspan(ranges["Fe_O_oxy_complex_A"][0], ranges["Fe_O_oxy_complex_A"][1],
                color="#FF0D0D", alpha=0.15, label="Fe-O range")
    ax1.axvspan(ranges["Fe_N_porphyrin_A"][0], ranges["Fe_N_porphyrin_A"][1],
                color="#3050F8", alpha=0.15, label="Fe-N range")
    ax1.axvspan(ranges["Fe_S_thiolate_A"][0], ranges["Fe_S_thiolate_A"][1],
                color="#FFFF30", alpha=0.20, label="Fe-S range")
    ax1.set_xlabel("distance (Å)")
    ax1.set_ylabel("element")
    ax1.legend(frameon=False, fontsize=6, loc="lower right")
    ax1.set_xlim(1.5, 3.0)

    # (B) summary counts vs minimums
    ax2 = fig.add_subplot(1, 4, 2)
    counts = d["summary_counts"]
    bars_x = ["≥4 N\n(porph)", "≥1 S\n(Cys)", "≥1 O\n(axial)"]
    have   = [counts["n_porphyrin_N"],
              counts["n_thiolate_S"],
              counts["n_axial_oxycomplex_O"]]
    need   = [4, 1, 1]
    x = np.arange(len(bars_x))
    w = 0.36
    ax2.bar(x - w/2, need, w, label="required",
            color="#CCCCCC", edgecolor="black", linewidth=0.4)
    ax2.bar(x + w/2, have, w, label="observed",
            color="#55A868", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(bars_x, fontsize=7)
    ax2.set_ylabel("count in first shell")
    ax2.legend(frameon=False, fontsize=7)

    # (C) closest distance with ranges marked
    ax3 = fig.add_subplot(1, 4, 3)
    closest_o = d["closest_O"]["distance_A"] if d["closest_O"] else float("nan")
    refs = {
        "Cpd I (1.65)":          1.65,
        "oxy-complex (1.80)":    1.80,
        "hydroperoxo (1.85)":    1.85,
        "axial OH (2.10)":       2.10,
    }
    rl, rv = list(refs.keys()), list(refs.values())
    ax3.bar(rl, rv, color="#CCCCCC", edgecolor="black", linewidth=0.4)
    ax3.axhline(closest_o, color="red", linestyle="--", linewidth=1.5,
                label=f"observed = {closest_o:.3f} Å")
    ax3.set_ylabel("Fe-O distance (Å)")
    ax3.tick_params(axis="x", labelsize=6, rotation=20)
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D scatter of Fe + first-shell atoms
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    from levinthal_glb import parse_glb  # noqa: E402
    from levinthal_glb.structure import find_iron, neighbours_of  # noqa: E402
    s = parse_glb(GLB_DIR / "model_of_cytochrome_p450__oxygen__drug_complex.glb")
    s = s.filter_oversized(5.0)
    s.atoms[:] = [a for a in s.atoms
                  if not (a.position[0] == 0 and a.position[1] == 0
                          and a.position[2] == 0)]
    fe_idx = find_iron(s)
    fe_pos = s.atoms[fe_idx].position
    nbr_idxs = neighbours_of(s, fe_idx, cutoff_A=3.0)
    # Fe first
    ax4.scatter(*fe_pos, color=ELEMENT_COLORS["Fe"], s=200,
                edgecolor="black", linewidth=0.7)
    ax4.text(fe_pos[0], fe_pos[1], fe_pos[2] + 0.15, "Fe",
             fontsize=9, ha="center")
    for j in nbr_idxs:
        a = s.atoms[j]
        ax4.scatter(*a.position, color=ELEMENT_COLORS.get(a.element, "#888"),
                    s=70, edgecolor="black", linewidth=0.4)
        ax4.plot([fe_pos[0], a.position[0]],
                 [fe_pos[1], a.position[1]],
                 [fe_pos[2], a.position[2]],
                 "k-", linewidth=0.4, alpha=0.5)
    ax4.set_xlabel("x (Å)", fontsize=8)
    ax4.set_ylabel("y (Å)", fontsize=8)
    ax4.set_zlabel("z (Å)", fontsize=8)
    ax4.set_title("Fe + first shell", fontsize=9)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_04_iron_coordination.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 05 — State 4 oxy-complex identification
# ============================================================
def panel_05():
    d = load("05_state4_oxy_complex")
    fig = make_fig()

    refs = d["reference_distances_A"]
    deltas = d["delta_to_each_state_A"]
    closest = d["closest_Fe_O_A"]

    # (A) closest Fe-O vs all reference distances
    ax1 = fig.add_subplot(1, 4, 1)
    states = list(refs.keys())
    vals = list(refs.values())
    cols = ["#C44E52", C_GLB["highlight"], "#FFA500", "#4C72B0"]
    ax1.bar(states, vals, color=cols, edgecolor="black", linewidth=0.4)
    ax1.axhline(closest, color="red", linestyle="--", linewidth=1.5,
                label=f"obs = {closest:.3f}")
    ax1.set_ylabel("reference Fe-O (Å)")
    ax1.tick_params(axis="x", labelsize=6, rotation=20)
    ax1.legend(frameon=False, fontsize=7)

    # (B) Distance to each canonical state
    ax2 = fig.add_subplot(1, 4, 2)
    short = [s.replace("_", "\n") for s in states]
    ax2.bar(short, list(deltas.values()),
            color=[C_GLB["highlight"] if s == d["nearest_canonical_state"] else "#CCCCCC"
                   for s in states],
            edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("|Δ| to reference (Å)")
    ax2.tick_params(axis="x", labelsize=6)

    # (C) all oxygen distances near Fe
    ax3 = fig.add_subplot(1, 4, 3)
    ods = d["all_oxygen_distances_A"]
    ax3.scatter(range(1, len(ods) + 1), ods,
                color="#FF0D0D", s=70, edgecolor="black", linewidth=0.4)
    ax3.axhspan(1.75, 1.90, color="#55A868", alpha=0.18,
                label="oxy-complex window")
    ax3.axhline(1.65, color="black", linestyle=":", linewidth=0.8,
                label="Cpd I (1.65)")
    ax3.set_xlabel("oxygen index (sorted)")
    ax3.set_ylabel("Fe-O (Å)")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D state trajectory: Cpd I, oxy-complex, peroxo, water; observed marker
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Use a simple 3D arrangement: x = state index, y = bond length, z = oxidation
    state_xyz = {
        "compound_I_ferryl":   (5, 1.65, 4),
        "oxy_complex_Fe_O2":   (3, 1.80, 2),
        "hydroperoxo_Fe_OOH":  (4, 1.85, 3),
        "water_axial_FeIII_OH": (1, 2.10, 3),
    }
    for s, (x, y, z) in state_xyz.items():
        c = "#55A868" if s == d["nearest_canonical_state"] else "#888888"
        ax4.scatter(x, y, z, color=c, s=120, edgecolor="black", linewidth=0.5)
        ax4.text(x, y, z + 0.18, s.split("_")[0], fontsize=7, ha="center")
    # observed
    nearest_xy = state_xyz[d["nearest_canonical_state"]]
    ax4.scatter(nearest_xy[0], closest, nearest_xy[2], color="red",
                marker="*", s=240, edgecolor="black", linewidth=0.5,
                label=f"obs = {closest:.3f}")
    ax4.set_xlabel("state #", fontsize=8)
    ax4.set_ylabel("Fe-O (Å)", fontsize=8)
    ax4.set_zlabel("Fe ox. state", fontsize=8)
    ax4.legend(frameon=False, fontsize=7, loc="upper left")
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_05_state4_oxy_complex.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 06 — Morphism chain
# ============================================================
def panel_06():
    d = load("06_morphism_chain")
    fig = make_fig()
    md = d["matrix_diagnostics"]

    # (A) Max-entry across stages
    ax1 = fig.add_subplot(1, 4, 1)
    stages = ["observe", "catalyze", "fuse"]
    maxes  = [md[s]["max_entry"] for s in stages]
    ax1.bar(stages, maxes, color=[C_GLB["atomistic"], "#FFA500", C_GLB["highlight"]],
            edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("max σ entry")

    # (B) Symmetry / property checks per stage
    ax2 = fig.add_subplot(1, 4, 2)
    stages_full = ["observe", "catalyze", "fuse", "access"]
    props = ["symmetric", "diag_zero", "monotonic", "within_min_max", "binary"]
    grid = []
    for s in stages_full:
        row = []
        for p in props:
            row.append(1 if md[s].get(p, None) is True else
                       (0 if md[s].get(p, None) is False else 0.5))
        grid.append(row)
    grid = np.array(grid)
    ax2.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(props)))
    ax2.set_xticklabels(props, rotation=30, fontsize=6)
    ax2.set_yticks(range(len(stages_full)))
    ax2.set_yticklabels(stages_full, fontsize=8)
    ax2.set_title("matrix property by stage", fontsize=9)

    # (C) Number of contacts vs. partition outputs
    ax3 = fig.add_subplot(1, 4, 3)
    n = d["n_atoms"]
    nc = d["n_contacts"]
    mx = d["max_possible_contacts"]
    ax3.bar(["actual", "max possible"], [nc, mx],
            color=[C_GLB["atomistic"], "#CCCCCC"],
            edgecolor="black", linewidth=0.4)
    ax3.set_yscale("log")
    ax3.set_ylabel("# contact pairs")
    ax3.set_title(f"density = {d['contact_density']:.4f}", fontsize=9)

    # (D) 3D bar: per-stage matrix max-entry over receiver chain
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    stages_x = stages_full[:3]
    xs = np.arange(len(stages_x))
    for i, s in enumerate(stages_x):
        ax4.bar3d(i, 0, 0, 0.6, 0.6, md[s]["max_entry"],
                  color=[C_GLB["atomistic"], "#FFA500", C_GLB["highlight"]][i],
                  edgecolor="black", linewidth=0.3, alpha=0.85)
        ax4.text(i + 0.3, 0.3, md[s]["max_entry"] + 0.02, s,
                 fontsize=7, ha="center")
    ax4.set_xticks(xs); ax4.set_xticklabels(stages_x, fontsize=7)
    ax4.set_yticks([])
    ax4.set_zlabel("max σ", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_06_morphism_chain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 07 — S-entropy address
# ============================================================
def panel_07():
    d = load("07_s_entropy_address")
    fig = make_fig()
    table = d["element_table"]

    # (A) Each element's |S| (norm)
    ax1 = fig.add_subplot(1, 4, 1)
    elems = list(table.keys())
    norms = [table[e]["norm"] for e in elems]
    cols  = [ELEMENT_COLORS.get(e, "#888888") for e in elems]
    ax1.bar(elems, norms, color=cols, edgecolor="black", linewidth=0.4)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=0.8,
                label="unit sphere")
    ax1.set_ylabel("‖S‖")
    ax1.legend(frameon=False, fontsize=7)
    ax1.tick_params(axis="x", labelsize=7)

    # (B) Centroid and F_CB result
    ax2 = fig.add_subplot(1, 4, 2)
    cs = d["structure_centroid_S"]
    ax2.bar(["S_k", "S_t", "S_e"], cs,
            color=["#C44E52", "#55A868", "#4C72B0"],
            edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("centroid coord")
    ax2.set_title(
        f"M = {d['F_CB_on_centroid']['M']:.3f}, "
        f"(n,l) = ({d['F_CB_on_centroid']['n']},"
        f"{d['F_CB_on_centroid']['l']})", fontsize=9)

    # (C) Trit address — visualise as ternary digit grid
    ax3 = fig.add_subplot(1, 4, 3)
    addr = d["structure_centroid_address"]
    for j, t in enumerate(addr):
        c = ["#FFFFFF", "#888888", "#222222"][int(t)]
        ax3.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=c,
                                    edgecolor="black", linewidth=0.4))
        ax3.text(j + 0.5, 0.5, t, ha="center", va="center",
                 fontsize=12, color="red" if t == "0" else "white")
    ax3.set_xlim(0, 9); ax3.set_ylim(0, 1)
    ax3.set_xticks([j + 0.5 for j in range(9)])
    ax3.set_xticklabels([str(j) for j in range(9)], fontsize=7)
    ax3.set_yticks([])
    ax3.set_title(f"centroid trit address (depth 9)", fontsize=9)
    ax3.set_xlabel("position j")
    ax3.spines[:].set_visible(False)

    # (D) 3D scatter of every covered element in S-cube
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for e, info in table.items():
        s = info["S"]
        c = ELEMENT_COLORS.get(e, "#888888")
        ax4.scatter(s[0], s[1], s[2], color=c, s=80,
                    edgecolor="black", linewidth=0.4)
        ax4.text(s[0], s[1], s[2] + 0.03, e, fontsize=7, ha="center")
    # centroid star
    ax4.scatter(cs[0], cs[1], cs[2], color="red", marker="*", s=200,
                edgecolor="black", linewidth=0.5, label="centroid")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.set_zlim(0, 1)
    ax4.set_xlabel("$S_k$", fontsize=8)
    ax4.set_ylabel("$S_t$", fontsize=8)
    ax4.set_zlabel("$S_e$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)
    ax4.legend(frameon=False, fontsize=7)

    plt.tight_layout()
    out = HERE / "panel_07_s_entropy_address.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
# Panel 08 — Five GLB roles taxonomy
# ============================================================
def panel_08():
    d = load("08_five_roles_taxonomy")
    fig = make_fig()
    per = d["per_glb"]
    names = list(per.keys())
    labels = ["Rib 1", "Atom", "Rib 2"]

    role_keys = ["role_1_calibration", "role_2_initial_conditions",
                 "role_3_validation_target", "role_4_interactive_probe",
                 "role_5_trajectory_waypoint"]
    role_short = ["1\ncalib", "2\ninit", "3\nvalid", "4\nprobe", "5\ntraject"]

    # (A) roles satisfied per GLB
    ax1 = fig.add_subplot(1, 4, 1)
    counts = [per[n]["n_roles_satisfied"] for n in names]
    ax1.bar(labels, counts, color=[C_GLB["ribbon1"], C_GLB["atomistic"], C_GLB["ribbon2"]],
            edgecolor="black", linewidth=0.4)
    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("# roles satisfied (of 5)")

    # (B) role × GLB heatmap
    ax2 = fig.add_subplot(1, 4, 2)
    grid = []
    for n in names:
        row = []
        for r in role_keys:
            row.append(1 if per[n]["roles"][r] else 0)
        grid.append(row)
    grid = np.array(grid)
    ax2.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(role_short, fontsize=7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(labels, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax2.text(j, i, "+" if grid[i, j] else "-",
                     ha="center", va="center",
                     color="white" if grid[i, j] == 0 else "black",
                     fontsize=11)
    ax2.set_title("role × GLB", fontsize=9)

    # (C) atom counts (drives roles 1-3)
    ax3 = fig.add_subplot(1, 4, 3)
    n_atoms = [per[n]["n_atoms"] for n in names]
    ax3.bar(labels, n_atoms,
            color=[C_GLB["ribbon1"], C_GLB["atomistic"], C_GLB["ribbon2"]],
            edgecolor="black", linewidth=0.4)
    ax3.axhline(50, color="black", linestyle="--", linewidth=0.7,
                label="role-2 threshold")
    ax3.axhline(10, color="red", linestyle=":", linewidth=0.7,
                label="role-3 threshold")
    ax3.set_yscale("symlog", linthresh=10)
    ax3.set_ylabel("# atoms (filtered)")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D bar: roles satisfied vs. GLB
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for i, n in enumerate(names):
        for j, r in enumerate(role_keys):
            v = 1 if per[n]["roles"][r] else 0
            ax4.bar3d(i, j, 0, 0.6, 0.6, v,
                      color="#55A868" if v else "#CCCCCC",
                      edgecolor="black", linewidth=0.3, alpha=0.9)
    ax4.set_xticks(range(len(names))); ax4.set_xticklabels(labels, fontsize=7)
    ax4.set_yticks(range(5)); ax4.set_yticklabels(role_short, fontsize=6)
    ax4.set_zlabel("satisfied", fontsize=8)
    ax4.set_zlim(0, 1.2)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = HERE / "panel_08_five_roles.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def main():
    panels = [panel_01, panel_02, panel_03, panel_04,
              panel_05, panel_06, panel_07, panel_08]
    print("Generating Paper 2.5 panels...")
    for p in panels:
        out = p()
        print(f"  + {out.name}")
    print(f"All panels written to: {HERE}")


if __name__ == "__main__":
    main()
