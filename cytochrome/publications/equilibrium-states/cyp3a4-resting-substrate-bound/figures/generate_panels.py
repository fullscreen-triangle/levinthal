"""
Generate one figure panel per Paper 3 validation result.

Each panel: 4 charts in a row, white background, minimal text,
at least one 3D chart per panel.
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
# Panel 01: Closed-Form Functors
# ============================================================
def panel_01():
    d = load("01_closed_form_functors")
    fig = make_fig()

    # (A) F_OC sweep showing S in unit cube
    ax1 = fig.add_subplot(1, 4, 1)
    sk_vals = [r["S"][0] for r in d["F_OC_log"]]
    st_vals = [r["S"][1] for r in d["F_OC_log"]]
    se_vals = [r["S"][2] for r in d["F_OC_log"]]
    ax1.scatter(sk_vals, st_vals, c=se_vals, cmap="viridis",
                s=14, edgecolor="none", alpha=0.85)
    ax1.set_xlabel(r"$S_k$")
    ax1.set_ylabel(r"$S_t$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # (B) F_CB results: M_LS vs M_HS as bars
    ax2 = fig.add_subplot(1, 4, 2)
    M_ls = d["F_CB_Fe_LS"]["M"]
    M_hs = d["F_CB_Fe_HS"]["M"]
    ax2.bar(["Fe LS", "Fe HS"], [M_ls, M_hs],
            color=["#4C72B0", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"partition depth $\mathcal{M}$")

    # (C) Cycle closure samples
    ax3 = fig.add_subplot(1, 4, 3)
    cycles = d["cycle_closure_samples"]
    omegas_in = [c["input"]["omega"] for c in cycles]
    omegas_out = [c["output"]["omega"] for c in cycles]
    Ms = [c["part"]["M"] for c in cycles]
    ax3.scatter(omegas_in, omegas_out, c=Ms, cmap="plasma",
                s=120, edgecolor="black", linewidth=0.6)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\omega_{in}$ (Hz)")
    ax3.set_ylabel(r"$\omega_{out}$ (Hz)")

    # (D) 3D scatter F_OC outputs colored by ||S||
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    norms = [math.sqrt(s ** 2 + t ** 2 + e ** 2)
             for s, t, e in zip(sk_vals, st_vals, se_vals)]
    sc = ax4.scatter(sk_vals, st_vals, se_vals, c=norms, cmap="viridis",
                     s=18, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_01_closed_form_functors.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 02: Resting State — Coherent Regime
# ============================================================
def panel_02():
    d = load("02_resting_state_regime")
    fig = make_fig()

    r_traj = d["r_trajectory_sample"]
    n = len(r_traj)
    t_norm = np.linspace(0, 1, n)

    # (A) r(t) trajectory
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t_norm, r_traj, color="#4C72B0", linewidth=1.6)
    ax1.fill_between(t_norm, 0, r_traj, color="#4C72B0", alpha=0.18)
    ax1.axhline(0.95, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax1.set_xlabel("t / T")
    ax1.set_ylabel("r(t)")
    ax1.set_ylim(0, 1.05)

    # (B) Regime classification visualisation
    ax2 = fig.add_subplot(1, 4, 2)
    regimes = ["coherent", "locked", "aperture", "hierarchical", "turbulent"]
    thresholds = [0.95, 0.80, 0.50, 0.20, 0.0]
    colours = ["#4C72B0", "#55A868", "#FFD700", "#FFA500", "#C44E52"]
    for i, (regime, threshold, colour) in enumerate(zip(regimes, thresholds, colours)):
        ax2.barh(regime, 1.0 - threshold if i == 0 else thresholds[i - 1] - threshold,
                 left=threshold, color=colour, edgecolor="black", linewidth=0.3)
    r_final = d["synchronization"]["r_final_mean_last_quarter"]
    ax2.axvline(r_final, color="red", linewidth=2.0, linestyle="-")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"$\langle r\rangle$")
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=7)

    # (C) Phase variance bar
    ax3 = fig.add_subplot(1, 4, 3)
    phase_var = d["synchronization"]["phase_variance_rad2"]
    ax3.bar(["resting"], [phase_var], color="#4C72B0",
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"phase variance $\sigma^2(\phi)$")

    # (D) 3D phase trajectory: t vs r(t) vs derivative
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    r_arr = np.array(r_traj)
    dr = np.gradient(r_arr)
    ax4.plot(t_norm, r_arr, dr, color="#4C72B0", linewidth=1.6)
    ax4.scatter(t_norm[::2], r_arr[::2], dr[::2],
                c=t_norm[::2], cmap="plasma",
                s=14, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel("t / T", fontsize=8)
    ax4.set_ylabel("r(t)", fontsize=8)
    ax4.set_zlabel("dr/dt", fontsize=8)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_02_resting_state.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 03: Heme-Pocket Capacitor
# ============================================================
def panel_03():
    d = load("03_heme_capacitor")
    fig = make_fig()

    # (A) Canonical values
    ax1 = fig.add_subplot(1, 4, 1)
    cv = d["canonical_values"]
    metrics = ["C (aF)", "U (eV)", r"$\tau_{RC}$ (ps)"]
    values = [cv["C_heme_aF"], cv["U_heme_eV"], cv["tau_RC_ps"]]
    paper = [d["paper_predictions"]["C_F"] * 1e18,
             d["paper_predictions"]["U_eV"],
             d["paper_predictions"]["tau_RC_ps"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w/2, values, w, color="#55A868", edgecolor="black", linewidth=0.4, label="computed")
    ax1.bar(x + w/2, paper, w, color="#C44E52", edgecolor="black", linewidth=0.4, label="paper")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.set_yscale("log")
    ax1.legend(frameon=False, fontsize=7)

    # (B) Epsilon_r sweep: capacitance scaling
    ax2 = fig.add_subplot(1, 4, 2)
    eps_log = d["epsilon_r_sweep"]
    eps_r_vals = [r["epsilon_r"] for r in eps_log]
    C_vals = [r["C_F"] * 1e18 for r in eps_log]
    U_vals = [r["U_eV"] for r in eps_log]
    ax2.plot(eps_r_vals, C_vals, "o-", color="#4C72B0", linewidth=1.5,
             markeredgecolor="black", markeredgewidth=0.5, label="C (aF)")
    ax2_b = ax2.twinx()
    ax2_b.plot(eps_r_vals, U_vals, "s-", color="#C44E52", linewidth=1.5,
               markeredgecolor="black", markeredgewidth=0.5, label="U (eV)")
    ax2.set_xlabel(r"$\varepsilon_r$")
    ax2.set_ylabel("C (aF)")
    ax2_b.set_ylabel("U (eV)")

    # (C) Separation sweep
    ax3 = fig.add_subplot(1, 4, 3)
    sep_log = d["separation_sweep"]
    sep_vals = [r["sep_A"] for r in sep_log]
    tau_vals = [r["tau_RC_s"] * 1e12 for r in sep_log]
    ax3.plot(sep_vals, tau_vals, "o-", color="#55A868", linewidth=1.5,
             markeredgecolor="black", markeredgewidth=0.5)
    ax3.set_xlabel(r"separation (\AA)")
    ax3.set_ylabel(r"$\tau_{RC}$ (ps)")

    # (D) 3D surface of capacitance C(eps_r, sep)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    eps_grid = np.linspace(3, 12, 20)
    sep_grid = np.linspace(4, 8, 20)
    EG, SG = np.meshgrid(eps_grid, sep_grid)
    EPS0_local = 8.854e-12
    AREA = (8e-10) ** 2
    C = EPS0_local * EG * AREA / (SG * 1e-10) * 1e18  # in aF
    surf = ax4.plot_surface(EG, SG, C, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel(r"$\varepsilon_r$", fontsize=8)
    ax4.set_ylabel(r"sep (\AA)", fontsize=8)
    ax4.set_zlabel("C (aF)", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_03_heme_capacitor.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 04: Variance Free Energy
# ============================================================
def panel_04():
    d = load("04_water_variance_free_energy")
    fig = make_fig()

    # (A) Resting vs bound variance
    ax1 = fig.add_subplot(1, 4, 1)
    rest_var = d["resting_state"]["mean_variance_rad2"]
    bound_var = d["bound_state"]["mean_variance_rad2"]
    ax1.bar(["resting", "bound"], [rest_var, bound_var],
            color=["#4C72B0", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\sigma^2(\phi)$ (rad$^2$)")

    # (B) Free energy comparison
    ax2 = fig.add_subplot(1, 4, 2)
    rest_F = d["resting_state"]["F_kcal_per_mol"]
    bound_F = d["bound_state"]["F_kcal_per_mol"]
    delta_F = d["substrate_binding"]["delta_F_bind_kcal_per_mol"]
    ax2.bar(["F_rest", "F_bound", r"$\Delta F_{bind}$"],
            [rest_F, bound_F, delta_F],
            color=["#4C72B0", "#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("F (kcal/mol)")
    ax2.axhline(0, color="black", linewidth=0.4)

    # (C) Variance to F sweep
    ax3 = fig.add_subplot(1, 4, 3)
    sw = d["variance_to_F_sweep"]
    var_vals = [s["variance_rad2"] for s in sw]
    F_vals = [s["F_kcal_per_mol"] for s in sw]
    ax3.plot(var_vals, F_vals, "o-", color="#4C72B0", linewidth=1.5,
             markeredgecolor="black", markeredgewidth=0.5)
    ax3.fill_between(var_vals, 0, F_vals, color="#4C72B0", alpha=0.18)
    ax3.set_xlabel(r"$\sigma^2(\phi)$ (rad$^2$)")
    ax3.set_ylabel("F (kcal/mol)")

    # (D) 3D ribbon: F as function of variance and N_eff
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    var_grid = np.linspace(0.01, 0.30, 25)
    n_grid = np.linspace(10, 200, 25)
    VG, NG = np.meshgrid(var_grid, n_grid)
    KB_T_KCAL = 4.27e-21 * 6.022e23 / 4184.0
    F = KB_T_KCAL * VG * NG
    surf = ax4.plot_surface(VG, NG, F, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.9)
    ax4.set_xlabel(r"$\sigma^2(\phi)$", fontsize=8)
    ax4.set_ylabel(r"$N_{eff}$", fontsize=8)
    ax4.set_zlabel("F (kcal/mol)", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_04_variance_free_energy.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 05: Spin-Crossover ΔM
# ============================================================
def panel_05():
    d = load("05_spin_crossover")
    fig = make_fig()

    # (A) M_LS vs M_HS
    ax1 = fig.add_subplot(1, 4, 1)
    M_ls = d["F_CB_results"]["Fe_LS"]["M"]
    M_hs = d["F_CB_results"]["Fe_HS"]["M"]
    ax1.bar(["Fe LS", "Fe HS"], [M_ls, M_hs],
            color=["#4C72B0", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\mathcal{M}$ (partition depth)")
    # Annotate ΔM
    ax1.text(0.5, max(M_ls, M_hs) * 1.05,
             rf"$\Delta\mathcal{{M}} = {d['delta_M']:.3f}$",
             ha="center", fontsize=10, fontweight="bold")

    # (B) Activation energy
    ax2 = fig.add_subplot(1, 4, 2)
    Ea_kcal = d["activation_energy"]["E_a_kcal_per_mol"]
    paper_Ea = d["paper_predictions"]["E_a_kcal"]
    ax2.bar(["computed", "paper"], [Ea_kcal, paper_Ea],
            color=["#55A868", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$E_a$ (kcal/mol)")

    # (C) Sensitivity sweep
    ax3 = fig.add_subplot(1, 4, 3)
    sw = d["sensitivity_sweep"]
    offsets = [s["hs_offset"] for s in sw]
    dMs = [s["delta_M"] for s in sw]
    Eas = [s["E_a_kcal"] for s in sw]
    ax3.plot(offsets, dMs, "o-", color="#4C72B0", linewidth=1.5,
             markeredgecolor="black", markeredgewidth=0.5, label=r"$\Delta\mathcal{M}$")
    ax3_b = ax3.twinx()
    ax3_b.plot(offsets, Eas, "s-", color="#C44E52", linewidth=1.5,
               markeredgecolor="black", markeredgewidth=0.5, label=r"$E_a$ (kcal/mol)")
    ax3.set_xlabel("HS coordinate offset")
    ax3.set_ylabel(r"$\Delta\mathcal{M}$")
    ax3_b.set_ylabel(r"$E_a$ (kcal/mol)")

    # (D) 3D: F_CB landscape
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sk_grid = np.linspace(0.5, 0.9, 25)
    se_grid = np.linspace(0.3, 0.7, 25)
    SK, SE = np.meshgrid(sk_grid, se_grid)
    St_fixed = 0.51
    norm = np.sqrt(SK ** 2 + St_fixed ** 2 + SE ** 2)
    norm_safe = np.minimum(norm, 1 - 1e-4)
    M = -np.log(1 - norm_safe)
    surf = ax4.plot_surface(SK, SE, M, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    # Mark Fe LS and HS
    ax4.scatter(d["F_CB_results"]["Fe_LS"]["S"][0],
                d["F_CB_results"]["Fe_LS"]["S"][2],
                d["F_CB_results"]["Fe_LS"]["M"],
                color="#4C72B0", s=80, edgecolor="black", linewidth=0.5)
    ax4.scatter(d["F_CB_results"]["Fe_HS"]["S"][0],
                d["F_CB_results"]["Fe_HS"]["S"][2],
                d["F_CB_results"]["Fe_HS"]["M"],
                color="#C44E52", s=80, edgecolor="black", linewidth=0.5)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_e$", fontsize=8)
    ax4.set_zlabel(r"$\mathcal{M}$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_05_spin_crossover.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 06: Substrate-Bound Locked Regime
# ============================================================
def panel_06():
    d = load("06_substrate_bound_regime")
    fig = make_fig()

    r_traj = d["r_trajectory_sample"]
    n = len(r_traj)
    t_norm = np.linspace(0, 1, n)

    # (A) r(t) compared to resting baseline (r=0.99 reference)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(t_norm, r_traj, color="#C44E52", linewidth=1.6, label="bound")
    ax1.axhline(0.99, color="#4C72B0", linestyle="--", linewidth=1.0,
                label="resting baseline")
    ax1.axhline(0.95, color="black", linestyle=":", linewidth=0.6, alpha=0.6)
    ax1.axhline(0.80, color="black", linestyle=":", linewidth=0.6, alpha=0.6)
    ax1.set_xlabel("t / T")
    ax1.set_ylabel("r(t)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=7, loc="lower right")

    # (B) r drop visualisation
    ax2 = fig.add_subplot(1, 4, 2)
    r_total = d["synchronization"]["r_final_mean_total"]
    r_protein = d["synchronization"]["r_final_mean_protein_only"]
    r_resting = 0.99  # paper value
    ax2.bar(["resting", "bound (total)", "bound (protein)"],
            [r_resting, r_total, r_protein],
            color=["#4C72B0", "#C44E52", "#FFA500"],
            edgecolor="black", linewidth=0.5)
    ax2.axhline(0.95, color="black", linestyle="--", linewidth=0.5)
    ax2.axhline(0.80, color="black", linestyle="--", linewidth=0.5)
    ax2.set_ylabel(r"$\langle r\rangle$")
    ax2.set_ylim(0, 1.05)

    # (C) Regime classification
    ax3 = fig.add_subplot(1, 4, 3)
    regimes = ["coherent", "locked", "aperture", "hierar.", "turbu."]
    thresholds = [0.95, 0.80, 0.50, 0.20, 0.0]
    colours = ["#4C72B0", "#55A868", "#FFD700", "#FFA500", "#C44E52"]
    bands = [(thresholds[i], thresholds[i - 1] if i > 0 else 1.0)
             for i in range(len(regimes))]
    for i, (regime, (lo, hi), colour) in enumerate(zip(regimes, bands, colours)):
        ax3.barh(regime, hi - lo, left=lo,
                 color=colour, edgecolor="black", linewidth=0.3)
    ax3.axvline(r_total, color="red", linewidth=2.0)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel(r"$\langle r\rangle$")
    ax3.invert_yaxis()
    ax3.tick_params(axis="y", labelsize=7)

    # (D) 3D r trajectory
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    r_arr = np.array(r_traj)
    dr = np.gradient(r_arr)
    ax4.plot(t_norm, r_arr, dr, color="#C44E52", linewidth=1.6)
    ax4.scatter(t_norm[::2], r_arr[::2], dr[::2],
                c=t_norm[::2], cmap="plasma",
                s=14, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel("t / T", fontsize=8)
    ax4.set_ylabel("r(t)", fontsize=8)
    ax4.set_zlabel("dr/dt", fontsize=8)
    ax4.view_init(elev=20, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_06_substrate_bound.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 07: Chamber Confinement
# ============================================================
def panel_07():
    d = load("07_chamber_confinement")
    fig = make_fig()

    # (A) Canonical values comparison
    ax1 = fig.add_subplot(1, 4, 1)
    cv = d["canonical_values"]
    metrics = [r"$\Delta\phi$ (V)", r"$|e\Delta\phi|/k_BT$"]
    values = [cv["delta_phi_V"], cv["confinement_kT_units"]]
    paper = [d["paper_predictions"]["delta_phi_V"],
             d["paper_predictions"]["confinement_kT_units"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w/2, values, w, color="#55A868",
            edgecolor="black", linewidth=0.4, label="computed")
    ax1.bar(x + w/2, paper, w, color="#C44E52",
            edgecolor="black", linewidth=0.4, label="paper")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.legend(frameon=False, fontsize=7)

    # (B) Sigma sweep
    ax2 = fig.add_subplot(1, 4, 2)
    sw = d["sigma_sweep"]
    sigmas = [s["delta_sigma_C_per_m2"] for s in sw]
    confs = [s["confinement_kT_units"] for s in sw]
    ax2.semilogx(sigmas, confs, "o-", color="#4C72B0", linewidth=1.5,
                 markeredgecolor="black", markeredgewidth=0.5)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.5,
                label=r"$|e\Delta\phi|/k_BT = 1$")
    ax2.set_xlabel(r"$|\delta\sigma|$ (C/m$^2$)")
    ax2.set_ylabel(r"$|e\Delta\phi|/k_BT$")
    ax2.legend(frameon=False, fontsize=7)

    # (C) Mutational predictions
    ax3 = fig.add_subplot(1, 4, 3)
    ml = d["mutational_predictions"]
    n_mut = [m["n_mutated_arg"] for m in ml]
    rel_rate = [m["relative_binding_rate"] for m in ml]
    ax3.plot(n_mut, rel_rate, "o-", color="#C44E52", linewidth=1.5,
             markersize=10, markeredgecolor="black", markeredgewidth=0.5)
    ax3.set_xlabel("# mutated chamber arginines")
    ax3.set_ylabel("relative binding rate")
    ax3.set_xticks(n_mut)

    # (D) 3D: confinement(sigma, eps_r)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    sigma_grid = np.linspace(0.005, 0.15, 25)
    eps_grid = np.linspace(3, 10, 25)
    SG, EG = np.meshgrid(sigma_grid, eps_grid)
    a = 6e-10
    EPS0_local = 8.854e-12
    KB_T_local = 4.27e-21
    e_local = 1.6e-19
    DPHI = SG * a / (2 * EPS0_local * EG)
    CONF = e_local * DPHI / KB_T_local
    surf = ax4.plot_surface(SG, EG, CONF, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel(r"$|\delta\sigma|$", fontsize=8)
    ax4.set_ylabel(r"$\varepsilon_r$", fontsize=8)
    ax4.set_zlabel(r"$|e\Delta\phi|/k_BT$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_07_chamber.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 08: Redox Shift
# ============================================================
def panel_08():
    d = load("08_redox_shift")
    fig = make_fig()

    # (A) Single-electron vs full d-shell shift
    ax1 = fig.add_subplot(1, 4, 1)
    se = d["single_electron_shift"]["mV"]
    fs = d["full_shell_shift"]["mV"]
    paper = d["paper_predictions"]["delta_E_mV"]
    ax1.bar(["single-e^-", "full d-shell\n(n_eff=5)", "paper"],
            [se, fs, paper],
            color=["#FFA500", "#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\Delta E_{1/2}$ (mV)")
    ax1.tick_params(axis="x", labelsize=7)

    # (B) n_eff sweep
    ax2 = fig.add_subplot(1, 4, 2)
    sw = d["n_eff_sweep"]
    n_vals = [s["n_eff"] for s in sw]
    shifts = [s["shift_mV"] for s in sw]
    ax2.plot(n_vals, shifts, "o-", color="#4C72B0", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.axvline(5, color="black", linestyle="--", linewidth=0.5)
    ax2.axhline(120, color="red", linestyle="--", linewidth=0.5,
                label="paper 120 mV")
    ax2.set_xlabel(r"$n_{eff}$")
    ax2.set_ylabel(r"$\Delta E_{1/2}$ (mV)")
    ax2.legend(frameon=False, fontsize=7)

    # (C) Delta_M sweep
    ax3 = fig.add_subplot(1, 4, 3)
    sw = d["delta_M_sweep"]
    dMs = [s["delta_M"] for s in sw]
    shifts = [s["shift_mV"] for s in sw]
    ax3.plot(dMs, shifts, "o-", color="#55A868", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax3.axvline(0.92, color="black", linestyle="--", linewidth=0.5)
    ax3.set_xlabel(r"$\Delta\mathcal{M}$")
    ax3.set_ylabel(r"$\Delta E_{1/2}$ (mV)")

    # (D) 3D: redox shift surface
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_grid = np.linspace(1, 6, 25)
    dM_grid = np.linspace(0.3, 1.5, 25)
    NG, DMG = np.meshgrid(n_grid, dM_grid)
    KB_T_local = 4.27e-21
    e_local = 1.6e-19
    DE_mV = (KB_T_local / e_local) * NG * DMG * 1000  # mV
    surf = ax4.plot_surface(NG, DMG, DE_mV, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    # Mark CYP3A4 operating point
    ax4.scatter(5, 0.92, 120,
                color="red", s=100, edgecolor="black", linewidth=0.6)
    ax4.set_xlabel(r"$n_{eff}$", fontsize=8)
    ax4.set_ylabel(r"$\Delta\mathcal{M}$", fontsize=8)
    ax4.set_zlabel(r"$\Delta E_{1/2}$ (mV)", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_08_redox_shift.png"
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
