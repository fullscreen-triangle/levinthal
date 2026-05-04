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


def main():
    panels = [panel_01, panel_02, panel_03, panel_04,
              panel_05, panel_06, panel_07, panel_08]
    for fn in panels:
        path = fn()
        print(f"  -> {path.name}")
    print(f"\nGenerated {len(panels)} panels in {FIG_DIR}")


if __name__ == "__main__":
    main()
