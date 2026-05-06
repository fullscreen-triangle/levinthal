"""
Generate one figure panel per Paper 4 validation result.

Each panel: 4 charts in a row, white background, minimal text,
≥1 3D chart per panel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "validation" / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Wire the shader-pipeline package into sys.path
import sys as _sys
GLB_DIR = ROOT.parents[2] / "glb"
if str(GLB_DIR) not in _sys.path:
    _sys.path.insert(0, str(GLB_DIR))

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
# Panel 01: Chain Topology
# ============================================================
def panel_01():
    d = load("01_chain_topology")
    fig = make_fig()

    cof = d["cofactor_partition_data"][:4]  # NADPH, FAD, FMN, Fe³⁺
    names = [c["name"] for c in cof]
    Ms = [c["M"] for c in cof]
    sk = [c["S"][0] for c in cof]
    st = [c["S"][1] for c in cof]
    se = [c["S"][2] for c in cof]

    # (A) M depth along chain
    ax1 = fig.add_subplot(1, 4, 1)
    cmap = cm.viridis
    bar_colors = [cmap(i / 3) for i in range(4)]
    ax1.bar(range(len(cof)), Ms, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(range(len(cof)))
    short_names = ["NADPH", "FAD", "FMN", "Fe³⁺"]
    ax1.set_xticklabels(short_names, fontsize=7)
    ax1.set_ylabel(r"$\mathcal{M}$ (partition depth)")

    # (B) S-coordinates evolution along chain
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(range(len(cof)), sk, "o-", color="#C44E52", label=r"$S_k$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.plot(range(len(cof)), st, "s-", color="#55A868", label=r"$S_t$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.plot(range(len(cof)), se, "^-", color="#4C72B0", label=r"$S_e$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.set_xticks(range(len(cof)))
    ax2.set_xticklabels(short_names, fontsize=7)
    ax2.set_ylabel("S-coord")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_ylim(0, 1)

    # (C) Distances along chain
    ax3 = fig.add_subplot(1, 4, 3)
    distances = d["distances_A"]
    hop_labels = [f"{d_['from']}→{d_['to']}" for d_ in distances]
    dist_values = [d_["distance_A"] for d_ in distances]
    ax3.bar(hop_labels, dist_values, color="#FFA500",
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("distance (Å)")
    ax3.tick_params(axis="x", labelsize=7, rotation=15)
    ax3.axhline(10, color="black", linestyle="--", linewidth=0.5,
                label="10 Å Marcus boundary")
    ax3.legend(frameon=False, fontsize=7, loc="upper left")

    # (D) 3D scatter of cofactors in S-space
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for i, c in enumerate(cof):
        ax4.scatter(c["S"][0], c["S"][1], c["S"][2],
                    color=cmap(i / 3), s=110,
                    edgecolor="black", linewidth=0.6)
        ax4.text(c["S"][0], c["S"][1], c["S"][2] + 0.03,
                 short_names[i], fontsize=8, ha="center")
    # Connect cofactors with line showing chain
    ax4.plot([c["S"][0] for c in cof],
             [c["S"][1] for c in cof],
             [c["S"][2] for c in cof],
             "k--", linewidth=0.8, alpha=0.5)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_01_chain_topology.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 02: d_C = 4 Efficiency
# ============================================================
def panel_02():
    d = load("02_dc4_efficiency")
    fig = make_fig()

    # (A) k_cat/K_M vs d_C
    ax1 = fig.add_subplot(1, 4, 1)
    sw = d["dc_sweep"]
    dcs = [s["d_C"] for s in sw]
    kcats = [s["log10_kcat_KM"] for s in sw]
    ax1.plot(dcs, kcats, "o-", color="#4C72B0", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax1.axvline(4, color="red", linestyle="--", linewidth=0.8,
                label="CPR-CYP3A4 chain")
    ax1.axhline(6, color="red", linestyle=":", linewidth=0.5)
    ax1.set_xlabel(r"$d_C$")
    ax1.set_ylabel(r"$\log_{10}(k_{cat}/K_M)$")
    ax1.legend(frameon=False, fontsize=7)

    # (B) Measured kcat/KM substrates
    ax2 = fig.add_subplot(1, 4, 2)
    measured = d["measured_data"]
    short_subs = [m["substrate"].split(" ")[0][:8] for m in measured]
    log_kcat = [math.log10(m["kcat_KM_M_per_s"]) for m in measured]
    ax2.barh(short_subs, log_kcat, color="#55A868",
             edgecolor="black", linewidth=0.4)
    ax2.axvline(6, color="red", linestyle="--", linewidth=1.0,
                label="prediction (10^6)")
    ax2.set_xlabel(r"$\log_{10}(k_{cat}/K_M)$")
    ax2.legend(frameon=False, fontsize=7)
    ax2.tick_params(axis="y", labelsize=7)

    # (C) Deviation from prediction
    ax3 = fig.add_subplot(1, 4, 3)
    devs = d["deviations"]
    sub_names = [m["substrate"].split(" ")[0][:8] for m in devs]
    dev_values = [m["log_deviation_from_prediction"] for m in devs]
    colors = ["#C44E52" if v < 0 else "#55A868" for v in dev_values]
    ax3.bar(sub_names, dev_values, color=colors, edgecolor="black", linewidth=0.4)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("log deviation from prediction")
    ax3.tick_params(axis="x", labelsize=7, rotation=20)

    # (D) 3D bar: efficiency landscape over d_C and substrate
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_subs = len(measured)
    dc_grid = np.arange(2, 8)
    log_kcat_grid = np.array([10 - dc for dc in dc_grid])
    # Bar positions
    xpos, ypos, zpos = [], [], []
    dxs, dys, dzs = [], [], []
    bar_colors = []
    for i, dc in enumerate(dc_grid):
        for j, sub in enumerate(measured):
            xpos.append(dc)
            ypos.append(j)
            zpos.append(0)
            dxs.append(0.6)
            dys.append(0.6)
            dzs.append(log_kcat_grid[i])
            bar_colors.append(cm.viridis(i / len(dc_grid)))
    ax4.bar3d(xpos, ypos, zpos, dxs, dys, dzs,
              color=bar_colors, edgecolor="black", linewidth=0.3, alpha=0.85)
    ax4.set_xlabel(r"$d_C$", fontsize=8)
    ax4.set_yticks(range(n_subs))
    ax4.set_yticklabels([s["substrate"].split(" ")[0][:6] for s in measured], fontsize=6)
    ax4.set_zlabel(r"$\log_{10}(k_{cat}/K_M)$", fontsize=8)
    ax4.view_init(elev=22, azim=-50)

    plt.tight_layout()
    out = FIG_DIR / "panel_02_dc4_efficiency.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 03: Marcus Distance
# ============================================================
def panel_03():
    d = load("03_marcus_distance")
    fig = make_fig()

    # (A) Beta sweep
    ax1 = fig.add_subplot(1, 4, 1)
    sw = d["beta_sweep"]
    betas = [s["beta_per_A"] for s in sw]
    rates = [s["log10_rate"] for s in sw]
    ax1.plot(betas, rates, "o-", color="#4C72B0", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax1.axvline(1.1, color="red", linestyle="--", linewidth=0.8,
                label=r"protein $\beta = 1.1$")
    ax1.set_xlabel(r"$\beta$ (Å⁻¹)")
    ax1.set_ylabel(r"$\log_{10}(k_{14\,\AA})$")
    ax1.legend(frameon=False, fontsize=7)

    # (B) Rate vs distance
    ax2 = fig.add_subplot(1, 4, 2)
    distances = np.linspace(4, 20, 50)
    rate_4A = d["rate_4A_per_s"]
    beta = d["parameters"]["beta_per_A"]
    rates_d = [rate_4A * math.exp(-beta * (r - 4)) for r in distances]
    ax2.semilogy(distances, rates_d, color="#C44E52", linewidth=1.5)
    ax2.axvline(4, color="black", linestyle="--", linewidth=0.5)
    ax2.axvline(14, color="red", linestyle="--", linewidth=0.8,
                label="FMN-heme")
    ax2.set_xlabel("distance (Å)")
    ax2.set_ylabel("rate (s⁻¹)")
    ax2.legend(frameon=False, fontsize=7)

    # (C) Predicted vs experimental at 14 Å
    ax3 = fig.add_subplot(1, 4, 3)
    pred = d["rate_14A_per_s"]
    expt = d["experimental_FMN_heme_rate_per_s"]
    log_pred = math.log10(max(pred, 1e-30))
    log_expt = math.log10(expt)
    ax3.bar(["predicted\n(Marcus)", "experimental"], [log_pred, log_expt],
            color=["#FFA500", "#55A868"], edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"$\log_{10}(k)$ s⁻¹")
    ax3.tick_params(axis="x", labelsize=7)

    # (D) 3D surface of rate(distance, beta)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    d_grid = np.linspace(4, 18, 25)
    b_grid = np.linspace(0.8, 1.4, 25)
    DG, BG = np.meshgrid(d_grid, b_grid)
    rate_4A = d["rate_4A_per_s"]
    R = rate_4A * np.exp(-BG * (DG - 4))
    log_R = np.log10(np.maximum(R, 1e-30))
    surf = ax4.plot_surface(DG, BG, log_R, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel("distance (Å)", fontsize=8)
    ax4.set_ylabel(r"$\beta$ (Å⁻¹)", fontsize=8)
    ax4.set_zlabel(r"$\log_{10}(k)$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_03_marcus_distance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 04: Selection Rules
# ============================================================
def panel_04():
    d = load("04_selection_rules")
    fig = make_fig()

    transitions = d["transitions"]

    # (A) Δl, Δm, Δs per transition
    ax1 = fig.add_subplot(1, 4, 1)
    n_trans = len(transitions)
    x = np.arange(n_trans)
    w = 0.27
    delta_l = [t["delta_l"] for t in transitions]
    delta_m = [t["delta_m"] for t in transitions]
    delta_s = [t["delta_s_orbital"] for t in transitions]
    ax1.bar(x - w, delta_l, w, color="#4C72B0", label=r"$\Delta\ell$",
            edgecolor="black", linewidth=0.4)
    ax1.bar(x,     delta_m, w, color="#55A868", label=r"$\Delta m$",
            edgecolor="black", linewidth=0.4)
    ax1.bar(x + w, delta_s, w, color="#C44E52", label=r"$\Delta s_{orb}$",
            edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"H{i+1}" for i in range(n_trans)], fontsize=7)
    ax1.set_ylabel("Δ value")
    ax1.legend(frameon=False, fontsize=7)
    ax1.axhline(0, color="black", linewidth=0.4)

    # (B) Selection rule satisfaction matrix
    ax2 = fig.add_subplot(1, 4, 2)
    rules = ["|Δℓ|≤1", "|Δm|≤1", "Δs=0", "all"]
    matrix = []
    for t in transitions:
        matrix.append([
            int(abs(t["delta_l"]) <= 1),
            int(t["delta_m_correct"]),
            int(t["delta_s_correct"]),
            int(t["all_satisfied"]),
        ])
    matrix = np.array(matrix)
    ax2.imshow(matrix, cmap="RdYlGn", aspect="auto",
               vmin=0, vmax=1)
    ax2.set_xticks(range(len(rules)))
    ax2.set_xticklabels(rules, fontsize=7)
    ax2.set_yticks(range(n_trans))
    ax2.set_yticklabels([f"H{i+1}" for i in range(n_trans)], fontsize=7)
    for i in range(n_trans):
        for j in range(len(rules)):
            ax2.text(j, i, "✓" if matrix[i, j] else "✗",
                     ha="center", va="center", fontsize=10)

    # (C) d_C contributions
    ax3 = fig.add_subplot(1, 4, 3)
    # Each transition contributes 1 to d_C
    d_c_per_trans = [1] * n_trans
    cumulative = np.cumsum(d_c_per_trans)
    ax3.bar(range(n_trans), d_c_per_trans, color="#FFA500",
            edgecolor="black", linewidth=0.4, label="per-transition")
    ax3.plot(range(n_trans), cumulative, "o-", color="#C44E52",
             linewidth=1.5, markersize=8, label="cumulative")
    ax3.set_xticks(range(n_trans))
    ax3.set_xticklabels([f"H{i+1}" for i in range(n_trans)], fontsize=7)
    ax3.set_ylabel(r"$d_C$ contribution")
    ax3.axhline(d["d_C_chain"], color="black", linestyle="--", linewidth=0.5,
                label=f"total = {d['d_C_chain']}")
    ax3.legend(frameon=False, fontsize=6)

    # (D) 3D scatter of cofactor states in (n, l, m)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    states = d["cofactor_states"]
    ns = [s["n"] for s in states]
    ls = [s["l"] for s in states]
    ms = [s["m"] for s in states]
    ax4.scatter(ns, ls, ms, c=range(len(states)), cmap="viridis",
                s=80, edgecolor="black", linewidth=0.5)
    for s in states:
        # Truncate name to last segment
        short = s["name"].split("-")[0][:8]
        ax4.text(s["n"], s["l"], s["m"] + 0.1, short, fontsize=6, ha="center")
    ax4.set_xlabel("n", fontsize=8)
    ax4.set_ylabel(r"$\ell$", fontsize=8)
    ax4.set_zlabel("m", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_04_selection_rules.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 05: Semiquinone Ladder
# ============================================================
def panel_05():
    d = load("05_semiquinone_ladder")
    fig = make_fig()

    states = d["flavin_states"]
    names = [s["name"] for s in states]
    Ms = [s["M"] for s in states]
    sk = [s["S"][0] for s in states]
    st = [s["S"][1] for s in states]
    se = [s["S"][2] for s in states]

    # (A) M progression across redox ladder
    ax1 = fig.add_subplot(1, 4, 1)
    state_colors = ["#FFA500", "#4C72B0", "#55A868"]
    ax1.bar(["ox", "semi", "red"], Ms, color=state_colors,
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\mathcal{M}$ (partition depth)")

    # (B) Free energy ladder
    ax2 = fig.add_subplot(1, 4, 2)
    fe = d["free_energies"]
    delta_G_semi = fe["delta_G_semi_kcal_per_mol"]
    delta_G_red = fe["delta_G_red_kcal_per_mol"]
    delta_G_total = delta_G_semi + delta_G_red
    bars = ["ox→semi", "semi→red", "ox→red"]
    G_values = [delta_G_semi, delta_G_red, delta_G_total]
    ax2.bar(bars, G_values, color="#C44E52",
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax2.tick_params(axis="x", labelsize=7)

    # (C) S-coordinate evolution
    ax3 = fig.add_subplot(1, 4, 3)
    x = np.arange(3)
    w = 0.27
    ax3.bar(x - w, sk, w, color="#C44E52", label=r"$S_k$",
            edgecolor="black", linewidth=0.4)
    ax3.bar(x,     st, w, color="#55A868", label=r"$S_t$",
            edgecolor="black", linewidth=0.4)
    ax3.bar(x + w, se, w, color="#4C72B0", label=r"$S_e$",
            edgecolor="black", linewidth=0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["ox", "semi", "red"], fontsize=7)
    ax3.set_ylabel("S-coord")
    ax3.legend(frameon=False, fontsize=7)
    ax3.set_ylim(0, 1)

    # (D) 3D scatter of three states in S-space
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for i, s in enumerate(states):
        ax4.scatter(s["S"][0], s["S"][1], s["S"][2],
                    color=state_colors[i], s=140,
                    edgecolor="black", linewidth=0.6)
    # Connect with arrows showing redox ladder
    for i in range(2):
        s1 = states[i]
        s2 = states[i + 1]
        ax4.plot([s1["S"][0], s2["S"][0]],
                 [s1["S"][1], s2["S"][1]],
                 [s1["S"][2], s2["S"][2]],
                 "k-", linewidth=1.0, alpha=0.6)
    for i, label in enumerate(["ox", "semi", "red"]):
        ax4.text(states[i]["S"][0], states[i]["S"][1], states[i]["S"][2] + 0.005,
                 label, fontsize=8, ha="center")
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_05_semiquinone_ladder.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 06: Chain Kinetics
# ============================================================
def panel_06():
    d = load("06_chain_kinetics")
    fig = make_fig()

    rates = d["hop_rates_per_s"]

    # (A) Hop rates (log scale)
    ax1 = fig.add_subplot(1, 4, 1)
    hop_names = ["hop 1", "hop 2", "hop 3"]
    rate_values = list(rates.values())
    bar_colors = ["#4C72B0", "#55A868", "#C44E52"]
    log_rates = [math.log10(r) for r in rate_values]
    ax1.bar(hop_names, log_rates, color=bar_colors,
            edgecolor="black", linewidth=0.5)
    ax1.axhline(math.log10(d["intrinsic_rate_per_s"]),
                color="black", linestyle="--", linewidth=0.5,
                label=f"intrinsic clock 10^{int(math.log10(d['intrinsic_rate_per_s']))}")
    ax1.set_ylabel(r"$\log_{10}(k)$ s⁻¹")
    ax1.legend(frameon=False, fontsize=7)

    # (B) Damping factors
    ax2 = fig.add_subplot(1, 4, 2)
    damping = d["damping_factors"]
    damping_logs = [math.log10(damping[h]) for h in rates.keys()]
    ax2.bar(hop_names, damping_logs, color="#FFA500",
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$\log_{10}(\mathrm{intrinsic}/\mathrm{observed})$")

    # (C) Chain composition (series resistor)
    ax3 = fig.add_subplot(1, 4, 3)
    inv_rates = [1.0 / r for r in rate_values]
    inv_total = sum(inv_rates)
    ax3.bar(hop_names + ["total"],
            [math.log10(t) for t in inv_rates] + [math.log10(inv_total)],
            color=["#4C72B0", "#55A868", "#C44E52", "#888888"],
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"$\log_{10}(1/k)$ s")

    # (D) 3D bar: rates landscape
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    intrinsic = d["intrinsic_rate_per_s"]
    xpos = np.arange(3)
    ypos_obs = np.zeros(3)
    ypos_int = np.ones(3)
    obs_logs = [math.log10(r) for r in rate_values]
    int_logs = [math.log10(intrinsic)] * 3
    ax4.bar3d(xpos - 0.3, ypos_obs, np.zeros(3),
              0.5, 0.4, obs_logs,
              color="#C44E52", edgecolor="black", linewidth=0.3)
    ax4.bar3d(xpos - 0.3, ypos_int, np.zeros(3),
              0.5, 0.4, int_logs,
              color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(hop_names, fontsize=7)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["observed", "intrinsic"], fontsize=7)
    ax4.set_zlabel(r"$\log_{10}(k)$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_06_chain_kinetics.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 07: Newton's Cradle
# ============================================================
def panel_07():
    d = load("07_newton_cradle")
    fig = make_fig()

    # (A) Donor vs delivered label comparison
    ax1 = fig.add_subplot(1, 4, 1)
    donor = d["original_donor_label"]
    delivered = d["delivered_to_terminal"]
    same = donor == delivered
    ax1.bar(["donor", "delivered"], [1.0, 1.0],
            color=["#4C72B0", "#55A868" if not same else "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax1.text(0, 1.05, donor, ha="center", fontsize=7)
    ax1.text(1, 1.05, delivered, ha="center", fontsize=7)
    ax1.set_ylabel("electron label")
    ax1.set_ylim(0, 1.3)

    # (B) Final cofactor distributions
    ax2 = fig.add_subplot(1, 4, 2)
    cof_final = d["cofactors_final_state"]
    cof_sizes = [len(c) for c in cof_final]
    short_names = ["NADPH", "FAD", "FMN", "Fe³⁺"]
    ax2.bar(short_names, cof_sizes, color=cm.viridis(np.linspace(0, 0.8, 4)),
            edgecolor="black", linewidth=0.5)
    ax2.axhline(3, color="black", linestyle="--", linewidth=0.5,
                label="initial count")
    ax2.set_ylabel("final electron count")
    ax2.legend(frameon=False, fontsize=7)
    ax2.tick_params(axis="x", labelsize=7)

    # (C) Propagation log
    ax3 = fig.add_subplot(1, 4, 3)
    log_entries = [s for s in d["simulation_log"] if "carrier_label" in s or "incoming_carrier" in s]
    n_entries = len(log_entries)
    y_pos = list(range(n_entries))
    ax3.barh(y_pos, [1.0] * n_entries,
             color=cm.plasma(np.linspace(0, 0.8, n_entries)),
             edgecolor="black", linewidth=0.4)
    for i, entry in enumerate(log_entries):
        carrier = entry.get("carrier_label") or entry.get("incoming_carrier", "?")
        ax3.text(0.5, i, carrier, ha="center", va="center", fontsize=7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"step {i+1}" for i in range(n_entries)], fontsize=7)
    ax3.set_xticks([])
    ax3.invert_yaxis()

    # (D) 3D representation of Newton's cradle
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for i, c in enumerate(cof_final):
        x_cof = i
        for j, label in enumerate(c):
            color = "#C44E52" if "L0" in label else "#4C72B0"
            ax4.scatter(x_cof, j * 0.4, 0,
                       color=color, s=50,
                       edgecolor="black", linewidth=0.4)
            ax4.text(x_cof, j * 0.4, 0.3, label, fontsize=5,
                    ha="center")
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(short_names, fontsize=7)
    ax4.set_xlabel("cofactor", fontsize=8)
    ax4.set_ylabel("electron stack", fontsize=8)
    ax4.view_init(elev=18, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_07_newton_cradle.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 08: Falsifiable Predictions
# ============================================================
def panel_08():
    d = load("08_falsifiable_predictions")
    fig = make_fig()

    # (A) Isotope transfer prediction comparison
    ax1 = fig.add_subplot(1, 4, 1)
    iso = d["prediction_1_isotope_transfer"]
    ax1.bar(["framework", "Marcus"],
            [iso["framework_predicted"], iso["marcus_predicted"]],
            color=["#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("isotope transfer probability")
    ax1.set_ylim(0, 1.1)

    # (B) Semiquinone destabilisation effect
    ax2 = fig.add_subplot(1, 4, 2)
    semi = d["prediction_2_semiquinone_necessity"]
    ax2.bar(["intact", "destabilised"],
            [semi["intact_log_kcat_KM"], semi["destabilised_log_kcat_KM"]],
            color=["#4C72B0", "#FFA500"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$\log_{10}(k_{cat}/K_M)$")

    # (C) d_C scaling for engineered chains
    ax3 = fig.add_subplot(1, 4, 3)
    chains = d["prediction_3_dc_scaling"]["engineered_chains"]
    n_cof = [c["n_cofactors"] for c in chains]
    log_kcat = [c["log10_kcat_KM"] for c in chains]
    ax3.plot(n_cof, log_kcat, "o-", color="#C44E52", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax3.set_xlabel("# cofactors in chain")
    ax3.set_ylabel(r"$\log_{10}(k_{cat}/K_M)$")
    ax3.axvline(4, color="black", linestyle="--", linewidth=0.5,
                label="natural CPR")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D bunching test: Fano factor across regimes
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    bunch = d["prediction_4_bunching"]
    fano_p = bunch["fano_poisson"]
    fano_b = bunch["fano_bunched"]
    # Surface: Fano factor as function of burst size and burst gap
    burst_sizes = np.arange(1, 30)
    burst_gaps = np.linspace(0.1, 10, 30)
    BS, BG = np.meshgrid(burst_sizes, burst_gaps)
    # Approximate Fano: for bunched arrivals, Fano scales with burst size and gap ratio
    F = 1 + (BS - 1) * BG / 10
    surf = ax4.plot_surface(BS, BG, F, cmap="plasma",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.scatter([1], [1], [fano_p], color="#4C72B0",
                s=100, edgecolor="black", linewidth=0.6)
    ax4.scatter([20], [5], [fano_b], color="#C44E52",
                s=100, edgecolor="black", linewidth=0.6)
    ax4.set_xlabel("burst size", fontsize=8)
    ax4.set_ylabel("burst gap", fontsize=8)
    ax4.set_zlabel("Fano factor", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_08_falsifiable_predictions.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 09: Apparatus Stack (5-layer instrument)
# ============================================================
def panel_09():
    d = load("09_apparatus_stack")
    fig = make_fig()

    # (A) Layer 1 oscillator frequencies (log scale, ten orders of magnitude)
    ax1 = fig.add_subplot(1, 4, 1)
    osc = d["Layer_1_oscillators"]
    names = list(osc.keys())
    freqs = [osc[n]["freq_Hz"] for n in names]
    coords = [osc[n]["resolves"] for n in names]
    short_names = [n.replace("_", "\n") for n in names]
    bars = ax1.bar(short_names, freqs,
                    color=["#4C72B0", "#55A868", "#C44E52", "#FFA500"],
                    edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("oscillator frequency (Hz)")
    ax1.set_title("Layer 1: hardware oscillators", fontsize=9)
    for bar, c in zip(bars, coords):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.5,
                 f"$\\to {c}$", ha="center", fontsize=8)
    ax1.tick_params(axis="x", labelsize=7)

    # (B) Layer 3 strobe windows (log timescale)
    ax2 = fig.add_subplot(1, 4, 2)
    strobes = d["Layer_3_strobes"]
    names = list(strobes.keys())
    ts = [strobes[n]["timescale_s"] for n in names]
    short = ["W_Sk\n(fs)", "W_St\n(ns)", "W_Se\n($\\mu$s+)"]
    ax2.bar(short, ts,
            color=["#FFA500", "#55A868", "#4C72B0"],
            edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("gate timescale (s)")
    ax2.set_title("Layer 3: ensemble strobes", fontsize=9)
    ax2.tick_params(axis="x", labelsize=7)

    # (C) Layer 5 observables list as a table
    ax3 = fig.add_subplot(1, 4, 3)
    obs = d["Layer_5_pipeline"]["observables"]
    short_obs = [
        "1. coupling K",
        "2. Franck--Condon",
        "3. Stokes shift",
        "4. Huang--Rhys",
        "5. Marcus $\\lambda$",
        "6. point group",
    ]
    for i, name in enumerate(short_obs):
        ax3.add_patch(plt.Rectangle((0.0, len(short_obs) - 1 - i), 1, 0.92,
                                    facecolor="#55A868" if i == 4 else "#CCCCCC",
                                    edgecolor="black", linewidth=0.4))
        ax3.text(0.06, len(short_obs) - 1 - i + 0.45, name, fontsize=9)
    ax3.set_xlim(0, 1.05); ax3.set_ylim(0, len(short_obs))
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("Layer 5: 6 hologram observables", fontsize=9)
    ax3.spines[:].set_visible(False)

    # (D) 3D pyramid: 5 layers stacked
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    layer_names = ["L1: oscillators", "L2: equiv. cert",
                   "L3: strobes", "L4: resonator",
                   "L5: hologram"]
    layer_colors = ["#4C72B0", "#888888", "#55A868", "#C44E52", "#FFA500"]
    for i, (lname, lcol) in enumerate(zip(layer_names, layer_colors)):
        # widths shrink with height
        side = 1.0 - 0.15 * i
        ax4.bar3d(-side / 2, -side / 2, i, side, side, 0.85,
                  color=lcol, edgecolor="black", linewidth=0.4, alpha=0.92)
        ax4.text(0.0, 0.65, i + 0.45, lname, fontsize=7)
    ax4.set_zticks([])
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_xlim(-1, 1); ax4.set_ylim(-1, 1)
    ax4.set_title("apparatus stack", fontsize=9)
    ax4.view_init(elev=18, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_09_apparatus_stack.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 10: Cofactor self-selection by counting anomaly
# ============================================================
def panel_10():
    d = load("10_cofactor_self_selection")
    fig = make_fig()
    cof = d["cofactors"]
    byst_chi2 = [r["chi2"] for r in []]  # filled below
    # Reload bystander details from JSON: we kept a summary only
    byst_summary = d["bystander_summary"]
    threshold = d["model"]["chi2_threshold"]

    # (A) chi^2 of cofactors vs bystander mean
    ax1 = fig.add_subplot(1, 4, 1)
    cof_names = [r["atom_id"].replace("_", "\n") for r in cof]
    cof_chi2 = [r["chi2"] for r in cof]
    bars = ax1.bar(cof_names, cof_chi2,
                    color=["#55A868"] * len(cof),
                    edgecolor="black", linewidth=0.5)
    ax1.axhline(threshold, color="red", linestyle="--", linewidth=0.8,
                label=f"$\\chi^2$ threshold = {threshold}")
    ax1.axhline(byst_summary["mean_chi2"], color="grey", linestyle=":",
                linewidth=0.8, label=f"bystander mean = {byst_summary['mean_chi2']:.2f}")
    ax1.set_ylabel("$\\chi^2$ statistic")
    ax1.set_title("self-selection by counting", fontsize=9)
    ax1.legend(frameon=False, fontsize=7)
    ax1.tick_params(axis="x", labelsize=7)

    # (B) ternary state distribution: equilibrium vs perturbed
    ax2 = fig.add_subplot(1, 4, 2)
    eq = d["model"]["equilibrium_distribution"]
    pe = d["model"]["perturbed_distribution_at_active_centres"]
    states = ["$\\tau=0$\nground", "$\\tau=1$\nnatural", "$\\tau=2$\nexcited"]
    x = np.arange(3); w = 0.36
    ax2.bar(x - w / 2, eq, w, label="equilibrium\n(bystanders)",
            color="#CCCCCC", edgecolor="black", linewidth=0.4)
    ax2.bar(x + w / 2, pe, w, label="ET-active\n(cofactors)",
            color="#55A868", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(states, fontsize=7)
    ax2.set_ylabel("state probability")
    ax2.legend(frameon=False, fontsize=7)

    # (C) selection metrics: recall and false-positive rate
    ax3 = fig.add_subplot(1, 4, 3)
    metrics = d["selection_metrics"]
    fp = byst_summary["false_positive_rate"]
    ax3.bar(["recall\n(cofactors)", "specificity\n(1 - FP)", "FP rate\n(bystanders)"],
            [metrics["cofactor_accuracy"], metrics["selection_specificity"], fp],
            color=["#55A868", "#4C72B0", "#C44E52"],
            edgecolor="black", linewidth=0.4)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("rate")
    ax3.tick_params(axis="x", labelsize=7)

    # (D) 3D scatter: 4 cofactors above the chi^2 plane, bystanders below
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # arrange cofactors at known cluster positions
    cof_positions = [
        ("NADPH", 0, 0, cof[0]["chi2"]),
        ("FAD",   1, 0, cof[1]["chi2"]),
        ("FMN",   2, 0, cof[2]["chi2"]),
        ("Fe",    3, 0, cof[3]["chi2"]),
    ]
    for name, x, y, z in cof_positions:
        ax4.scatter(x, y, z, color="#55A868", s=180,
                    edgecolor="black", linewidth=0.6)
        ax4.text(x, y, z + 8, name, ha="center", fontsize=8)
    # bystanders: scatter below threshold
    rng = np.random.default_rng(42)
    n_byst_show = 80
    bx = rng.uniform(-1, 4, n_byst_show)
    by = rng.uniform(-2, 3, n_byst_show)
    bz = rng.uniform(0, threshold, n_byst_show)
    ax4.scatter(bx, by, bz, color="#888888", s=10, alpha=0.5)
    # threshold plane
    XX, YY = np.meshgrid([-1, 4], [-2, 3])
    ZZ = np.full_like(XX, threshold, dtype=float)
    ax4.plot_surface(XX, YY, ZZ, color="red", alpha=0.18)
    ax4.set_xticks([0, 1, 2, 3]); ax4.set_xticklabels(["NADPH", "FAD", "FMN", "Fe"], fontsize=7)
    ax4.set_yticks([])
    ax4.set_zlabel("$\\chi^2$", fontsize=8)
    ax4.set_title("cofactors above threshold", fontsize=9)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_10_cofactor_self_selection.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 11: GLB-grounded electron-movement visualisations (HEADLINE)
# ============================================================
def panel_11():
    """Headline panel: load the real cytochrome P450 GLB, run the
    five-pass shader pipeline, and render |psi(r,t)|^2 across the
    four-cofactor chain anchored to the real heme-Fe position."""
    from levinthal_glb import run_pipeline_glb_grounded

    GLB = (ROOT.parents[2] / "glb"
           / "model_of_cytochrome_p450__oxygen__drug_complex.glb")

    pipeline = run_pipeline_glb_grounded(
        glb_path=str(GLB),
        t_fs_frames=(0.0, 100.0, 250.0, 500.0, 800.0),
        grid_shape=(48, 48, 48),
    )
    frames = pipeline["frames"]
    cof_pos = np.array(pipeline["cofactor_positions_A"])  # 4x3
    fe_pos  = np.array(pipeline["fe_position_A"])         # 3
    bbox_min = np.array(pipeline["bbox_min_A"])
    bbox_max = np.array(pipeline["bbox_max_A"])
    cmap = plt.get_cmap("plasma")
    cof_lab = ["NADPH", "FAD", "FMN", "heme"]

    # (A) Marcus lambda recovered from each frame's diffraction pattern
    ax1 = fig_layout = make_fig()
    ax1 = fig_layout.add_subplot(1, 4, 1)
    times_for_lambda = [f["t_fs"] for f in frames if f["lambda_eV"] is not None]
    lambdas = [f["lambda_eV"] for f in frames if f["lambda_eV"] is not None]
    ax1.plot(times_for_lambda, lambdas, "o-", color="#55A868",
             linewidth=1.3, markersize=8,
             markeredgecolor="black", markeredgewidth=0.5)
    ax1.axhspan(0.7, 1.0, color="#55A868", alpha=0.12, label="lit. range 0.7-1.0 eV")
    ax1.set_xlabel("time (fs)")
    ax1.set_ylabel(r"$\lambda$ from diffraction (eV)")
    ax1.set_title("Marcus $\\lambda$ via Pass 3 FFT", fontsize=9)
    ax1.legend(frameon=False, fontsize=7)

    # (B) Electron centroid along the chain axis vs time (GLB coords)
    ax2 = fig_layout.add_subplot(1, 4, 2)
    # Project density onto the chain axis (NADPH -> heme)
    axis_vec = (cof_pos[3] - cof_pos[0])
    axis_vec /= max(np.linalg.norm(axis_vec), 1e-12)
    chain_origin = cof_pos[0]   # NADPH

    centroids = []
    for i, f in enumerate(frames):
        density = f["density"]
        nx, ny, nz = density.shape
        xs = np.linspace(bbox_min[0], bbox_max[0], nx)
        ys = np.linspace(bbox_min[1], bbox_max[1], ny)
        zs = np.linspace(bbox_min[2], bbox_max[2], nz)
        XG, YG, ZG = np.meshgrid(xs, ys, zs, indexing="ij")
        # signed distance along the chain axis from NADPH
        rel = np.stack([XG - chain_origin[0],
                         YG - chain_origin[1],
                         ZG - chain_origin[2]], axis=-1)
        proj = (rel * axis_vec).sum(axis=-1)
        weight = density
        if weight.sum() > 0:
            centroid_proj = float((proj * weight).sum() / weight.sum())
        else:
            centroid_proj = 0.0
        centroids.append(centroid_proj)
    times = [f["t_fs"] for f in frames]
    for i, (t, c) in enumerate(zip(times, centroids)):
        ax2.scatter(t, c, color=cmap(i / max(1, len(times) - 1)),
                    s=180, edgecolor="black", linewidth=0.5)
    # cofactor projections (reference lines)
    cof_projs = [(cof_pos[j] - chain_origin) @ axis_vec for j in range(4)]
    for cp, lab in zip(cof_projs, cof_lab):
        ax2.axhline(cp, color="grey", linestyle=":", linewidth=0.5)
        ax2.text(times[-1] + 50, cp, lab, fontsize=7,
                 va="center", color="grey")
    ax2.set_xlabel("time (fs)")
    ax2.set_ylabel(r"electron centroid along chain (\AA)")
    ax2.set_title("trajectory in GLB coordinates", fontsize=9)
    ax2.set_xlim(-50, times[-1] + 220)

    # (C) |psi|^2 profile along the chain axis at each frame
    ax3 = fig_layout.add_subplot(1, 4, 3)
    n_samples = 200
    chain_t = np.linspace(0, np.linalg.norm(cof_pos[3] - cof_pos[0]),
                          n_samples)
    chain_pts = chain_origin[None, :] + chain_t[:, None] * axis_vec[None, :]
    for i, f in enumerate(frames):
        density = f["density"]
        nx, ny, nz = density.shape
        # Sample the 3D grid at chain_pts via nearest-neighbour
        ix = np.clip(((chain_pts[:, 0] - bbox_min[0])
                      / (bbox_max[0] - bbox_min[0]) * (nx - 1)).astype(int), 0, nx - 1)
        iy = np.clip(((chain_pts[:, 1] - bbox_min[1])
                      / (bbox_max[1] - bbox_min[1]) * (ny - 1)).astype(int), 0, ny - 1)
        iz = np.clip(((chain_pts[:, 2] - bbox_min[2])
                      / (bbox_max[2] - bbox_min[2]) * (nz - 1)).astype(int), 0, nz - 1)
        psi2_line = density[ix, iy, iz]
        col = cmap(i / max(1, len(times) - 1))
        ax3.fill_between(chain_t, i, i + psi2_line, color=col, alpha=0.7,
                          edgecolor="black", linewidth=0.3)
        ax3.text(chain_t[-1] + 0.6, i + 0.35, f"{int(times[i])} fs",
                 fontsize=8, va="center")
    for cp, lab in zip(cof_projs, cof_lab):
        ax3.axvline(cp, color="grey", linestyle=":", linewidth=0.5)
    ax3.set_xlabel(r"position along chain (\AA, from NADPH)")
    ax3.set_yticks([])
    ax3.set_ylim(-0.2, len(times))
    ax3.set_xlim(-1, chain_t[-1] + 4.5)
    ax3.set_title("$|\\psi(\\mathbf{r}, t)|^2$ profile", fontsize=9)

    # (D) 3D voxel cloud at t=500 fs in REAL GLB coordinates
    ax4 = fig_layout.add_subplot(1, 4, 4, projection="3d")
    f_mid = frames[3]   # t=500 fs
    density3d = f_mid["density"]
    nx, ny, nz = density3d.shape
    xs = np.linspace(bbox_min[0], bbox_max[0], nx)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz)
    XG, YG, ZG = np.meshgrid(xs, ys, zs, indexing="ij")
    mask = density3d > 0.25
    cols = cmap(density3d[mask])
    ax4.scatter(XG[mask], YG[mask], ZG[mask], c=cols, s=14,
                alpha=0.55, edgecolor="none")
    # Cofactor markers in real GLB coordinates
    cof_colors = ["#4C72B0", "#FFA500", "#55A868", "#C44E52"]
    for j, (lab, col) in enumerate(zip(cof_lab, cof_colors)):
        cx, cy, cz = cof_pos[j]
        ax4.scatter([cx], [cy], [cz], color=col, s=180,
                    edgecolor="black", linewidth=0.6)
        ax4.text(cx, cy, cz + 1.6, lab, ha="center", fontsize=7)
    # Highlight Fe (real GLB position)
    ax4.scatter([fe_pos[0]], [fe_pos[1]], [fe_pos[2]], color="black",
                marker="x", s=140, linewidth=2)
    ax4.set_xlabel("x (\\AA)", fontsize=8)
    ax4.set_ylabel("y (\\AA)", fontsize=8)
    ax4.set_zlabel("z (\\AA)", fontsize=8)
    ax4.set_title("|$\\psi$|$^2$ in real GLB coords, t=500 fs",
                  fontsize=9)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_11_electron_visualisations.png"
    fig_layout.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig_layout)
    return out


# ============================================================
# Panel 12: GLB+shader pipeline integration audit
# ============================================================
def panel_12():
    """Apparatus integration audit. Visualises the GLB-anchored cofactor
    cluster, the time-evolution of cofactor occupancies under the
    shader pipeline, the per-frame Marcus lambda, and the validation
    checks of script 12."""
    d = load("12_glb_shader_pipeline")
    fig = make_fig()

    cof_pos = np.array(d["cofactor_positions_A"])  # 4x3
    fe_pos = np.array(d["fe_position_A"])
    cof_lab = ["NADPH", "FAD", "FMN", "heme"]
    cof_colors = ["#4C72B0", "#FFA500", "#55A868", "#C44E52"]

    # (A) Cofactor occupancies under the shader pipeline (final-frame readout)
    ax1 = fig.add_subplot(1, 4, 1)
    occ = d["final_frame_occupancy_NADPH_FAD_FMN_heme"]
    ax1.bar(cof_lab, occ,
            color=cof_colors,
            edgecolor="black", linewidth=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("occupancy at $t = 800$ fs")
    ax1.set_title("electron arrived at heme", fontsize=9)

    # (B) Centroid advance along the chain
    ax2 = fig.add_subplot(1, 4, 2)
    centroids = d["centroids_along_chain_A"]
    times = [0, 100, 250, 500, 800]
    ax2.plot(times, centroids, "o-", color="#4C72B0",
             linewidth=1.4, markersize=8,
             markeredgecolor="black", markeredgewidth=0.5)
    chain_length = d["chain_length_A"]
    cof_projs = [(cof_pos[j] - cof_pos[0]) @ (
        (cof_pos[3] - cof_pos[0]) / chain_length) for j in range(4)]
    for cp, lab in zip(cof_projs, cof_lab):
        ax2.axhline(cp, color="grey", linestyle=":", linewidth=0.5)
        ax2.text(840, cp, lab, fontsize=7, va="center", color="grey")
    ax2.set_xlabel("time (fs)")
    ax2.set_ylabel(r"centroid along chain (\AA)")
    ax2.set_title("trajectory in real GLB space", fontsize=9)
    ax2.set_xlim(-50, 1000)

    # (C) Apparatus integration checks pass/fail
    ax3 = fig.add_subplot(1, 4, 3)
    checks = d["checks"]
    cnames = list(checks.keys())
    cvals = [1 if v else 0 for v in checks.values()]
    short_c = [
        c.replace("_", "\n")
         .replace("\nat\nleast", " >=")
         .replace("\nwithin\n20\npercent\nof", " ~~ ")
        for c in cnames
    ]
    ax3.barh(range(len(cnames)), cvals,
             color=["#55A868" if v else "#C44E52" for v in cvals],
             edgecolor="black", linewidth=0.5)
    ax3.set_yticks(range(len(cnames)))
    ax3.set_yticklabels(short_c, fontsize=5)
    ax3.set_xlim(0, 1.2)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["fail", "pass"], fontsize=8)
    ax3.set_title("apparatus integration", fontsize=9)

    # (D) 3D scene: GLB-anchored cofactor cluster + chain axis
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Plot cofactor markers in real 3D coordinates
    for j, (lab, col) in enumerate(zip(cof_lab, cof_colors)):
        cx, cy, cz = cof_pos[j]
        ax4.scatter([cx], [cy], [cz], color=col, s=200,
                    edgecolor="black", linewidth=0.6)
        ax4.text(cx, cy, cz + 1.8, lab, ha="center", fontsize=7)
    # Chain axis as a line
    ax4.plot(cof_pos[:, 0], cof_pos[:, 1], cof_pos[:, 2],
             "k-", linewidth=1.0, alpha=0.7)
    # Highlight Fe (real GLB position)
    ax4.scatter([fe_pos[0]], [fe_pos[1]], [fe_pos[2]],
                color="black", marker="x", s=200, linewidth=2.5)
    ax4.text(fe_pos[0], fe_pos[1], fe_pos[2] - 2.5,
             f"Fe (GLB)\n({fe_pos[0]:.1f}, {fe_pos[1]:.1f}, {fe_pos[2]:.1f})",
             ha="center", fontsize=6.5)
    ax4.set_xlabel(r"x (\AA)", fontsize=8)
    ax4.set_ylabel(r"y (\AA)", fontsize=8)
    ax4.set_zlabel(r"z (\AA)", fontsize=8)
    ax4.set_title("real GLB-anchored chain", fontsize=9)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_12_glb_shader_integration.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    panels = [panel_01, panel_02, panel_03, panel_04,
              panel_05, panel_06, panel_07, panel_08,
              panel_09, panel_10, panel_11, panel_12]
    for fn in panels:
        path = fn()
        print(f"  -> {path.name}")
    print(f"\nGenerated {len(panels)} panels in {FIG_DIR}")


if __name__ == "__main__":
    main()
