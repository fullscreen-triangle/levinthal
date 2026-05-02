"""
Generate one figure panel per validation result.

Each panel has 4 charts in a row, white background, minimal text,
at least one 3D chart per panel. All charts are data-driven (no
tables, no conceptual diagrams).

Outputs: validation/figures/panel_NN.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

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


def make_panel_figure() -> tuple[plt.Figure, list]:
    """Standard 4-charts-in-a-row layout, ~16 in wide, ~4 in tall."""
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    return fig


# ============================================================
# Panel 01: Floor Theorem
# ============================================================
def panel_01():
    d = load("01_floor_theorem")
    fig = make_panel_figure()

    # (A) Floor decomposition bar chart (log scale)
    ax1 = fig.add_subplot(1, 4, 1)
    components = ["disc", "Q-osc", "conv"]
    values = [
        d["floor_components"]["floor_disc"],
        d["floor_components"]["floor_Q"]["quadrature_sum"],
        d["floor_components"]["floor_conv"],
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax1.bar(components, values, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"floor contribution")
    ax1.axhline(d["floor_total"], color="black", linestyle="--", linewidth=0.8)
    ax1.text(2.4, d["floor_total"] * 1.3, f"total = {d['floor_total']:.2e}",
             fontsize=8, ha="right")

    # (B) Per-oscillator Q-floor (4 hardware oscillators)
    ax2 = fig.add_subplot(1, 4, 2)
    osc_data = d["floor_components"]["floor_Q"]["per_oscillator"]
    names = [o["oscillator"] for o in osc_data]
    sigmas = [o["sigma"] for o in osc_data]
    ax2.barh(names, sigmas, color="#55A868", edgecolor="black", linewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\sigma_i$")
    ax2.invert_yaxis()

    # (C) Depth sensitivity
    ax3 = fig.add_subplot(1, 4, 3)
    sweep = d["depth_sensitivity"]
    depths = [s["depth"] for s in sweep]
    f_total = [s["floor_total"] for s in sweep]
    f_disc = [s["floor_disc"] for s in sweep]
    f_conv = [s["floor_conv"] for s in sweep]
    ax3.semilogy(depths, f_total, "o-", color="black", label="total", linewidth=1.5)
    ax3.semilogy(depths, f_disc, "s--", color="#4C72B0", label="disc", linewidth=1.0)
    ax3.semilogy(depths, f_conv, "^--", color="#C44E52", label="conv", linewidth=1.0)
    ax3.set_xlabel("recursion depth d")
    ax3.set_ylabel("floor")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D: floor components vs depth (3D bar)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    xpos = []
    ypos = []
    zpos = []
    dx = []
    dy = []
    dz = []
    color_map = {"disc": "#4C72B0", "Q": "#55A868", "conv": "#C44E52"}
    bar_colors = []
    Q_value = d["floor_components"]["floor_Q"]["quadrature_sum"]
    for i, s in enumerate(sweep):
        for j, (label, val) in enumerate(zip(
            ["disc", "Q", "conv"],
            [s["floor_disc"], Q_value, s["floor_conv"]],
        )):
            xpos.append(i)
            ypos.append(j)
            zpos.append(0)
            dx.append(0.6)
            dy.append(0.6)
            dz.append(np.log10(val) - (-7))  # offset for log-style display
            bar_colors.append(color_map[label])
    ax4.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax4.set_xticks(range(len(sweep)))
    ax4.set_xticklabels([s["depth"] for s in sweep], fontsize=7)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["disc", "Q", "conv"], fontsize=7)
    ax4.set_xlabel("depth", fontsize=8)
    ax4.set_zlabel(r"$\log_{10}(\mathrm{floor})+7$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_01_floor_theorem.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 02: Capacity & Selection Rules
# ============================================================
def panel_02():
    d = load("02_capacity_selection")
    fig = make_panel_figure()

    # (A) C(n) = 2n^2
    ax1 = fig.add_subplot(1, 4, 1)
    rows = d["capacity_table"]
    ns = [r["n"] for r in rows]
    cs = [r["C_formula_2n2"] for r in rows]
    ax1.bar(ns, cs, color="#4C72B0", edgecolor="black", linewidth=0.6)
    n_smooth = np.linspace(0.5, max(ns) + 0.5, 200)
    ax1.plot(n_smooth, 2 * n_smooth ** 2, "k--", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("n")
    ax1.set_ylabel("C(n)")
    ax1.set_xticks(ns)

    # (B) Cumulative capacity
    ax2 = fig.add_subplot(1, 4, 2)
    cum = np.cumsum(cs)
    ax2.plot(ns, cum, "o-", color="#55A868", linewidth=1.5, markersize=6)
    ax2.fill_between(ns, 0, cum, color="#55A868", alpha=0.2)
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$\sum_{k=1}^n C(k)$")

    # (C) Selection rule allowed/forbidden bars
    ax3 = fig.add_subplot(1, 4, 3)
    audit = d["transition_audit"]
    counts = [audit["allowed_count"], audit["forbidden_count"]]
    ax3.pie(
        counts,
        labels=["allowed", "forbidden"],
        colors=["#55A868", "#C44E52"],
        startangle=90,
        wedgeprops=dict(edgecolor="black", linewidth=0.6),
        autopct="%1.1f%%",
        textprops={"fontsize": 8},
    )

    # (D) 3D: scatter of all (n, l, m) states for n=1..5
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    xs, ys, zs, sizes = [], [], [], []
    for n in range(1, 6):
        for l in range(0, n):
            for m in range(-l, l + 1):
                xs.append(n)
                ys.append(l)
                zs.append(m)
                sizes.append(20 + 2 * n)
    sc = ax4.scatter(xs, ys, zs, c=zs, s=sizes, cmap="viridis",
                     edgecolor="black", linewidth=0.3, alpha=0.85)
    ax4.set_xlabel("n", fontsize=8)
    ax4.set_ylabel(r"$\ell$", fontsize=8)
    ax4.set_zlabel("m", fontsize=8)
    ax4.set_xticks(range(1, 6))
    ax4.view_init(elev=18, azim=-65)

    plt.tight_layout()
    out = FIG_DIR / "panel_02_capacity_selection.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 03: Cycle Closure
# ============================================================
def panel_03():
    d = load("03_cycle_closure")
    fig = make_panel_figure()

    # (A) Octave error fraction within bound
    ax1 = fig.add_subplot(1, 4, 1)
    stats = d["round_trip_statistics"]
    metrics = ["mean", "max", "frac<=1"]
    values = [
        stats["octave_error_mean"],
        stats["octave_error_max"],
        stats["fraction_within_one_octave"],
    ]
    colors = ["#4C72B0", "#C44E52", "#55A868"]
    ax1.bar(metrics, values, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("octave error")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)

    # (B) Composition log: input n vs output n
    ax2 = fig.add_subplot(1, 4, 2)
    log_data = d["composition_log_sample"]
    ins = [c["input"][0] for c in log_data]
    outs = [c["output"][0] for c in log_data]
    ax2.plot([0, 5], [0, 5], "k--", linewidth=0.6, alpha=0.4)
    ax2.scatter(ins, outs, s=120, c="#4C72B0", edgecolor="black", linewidth=0.6, zorder=3)
    ax2.set_xlabel(r"$n_{in}$")
    ax2.set_ylabel(r"$n_{out}$")
    ax2.set_xticks(range(0, 6))
    ax2.set_yticks(range(0, 6))
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_aspect("equal")

    # (C) Cell invariance rate as ring chart
    ax3 = fig.add_subplot(1, 4, 3)
    rate = d["cell_invariance"]["rate"]
    ax3.pie(
        [rate, 1 - rate],
        labels=["preserved", "drift"],
        colors=["#55A868", "#C44E52"],
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="black", linewidth=0.6),
        textprops={"fontsize": 8},
    )
    ax3.text(0, 0, f"{rate:.0%}", ha="center", va="center", fontsize=14, fontweight="bold")

    # (D) 3D: synthetic round-trip surface in (omega_in, omega_out, error)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Reproduce the round-trip mapping over a grid for visualization
    omega_ref = d["parameters"]["omega_ref_hz"]
    log_omega = np.linspace(4, 15, 40)
    omegas = 10 ** log_omega
    err = []
    out_omegas = []
    for w in omegas:
        # mimic the floor-log mapping
        wn = max(w / omega_ref, 1e-12)
        n_c = int(np.floor(np.log2(wn))) + 1
        if n_c < 1:
            n_c = 1
        # reconstructed lower-octave-edge omega
        w_back = omega_ref * (2 ** (n_c - 1))
        out_omegas.append(w_back)
        err.append(abs(np.log2(wn + 1e-30) - np.log2(w_back / omega_ref + 1e-30)))
    ax4.plot(log_omega, np.log10(np.asarray(out_omegas)), err,
             color="#4C72B0", linewidth=1.5, alpha=0.85)
    ax4.scatter(log_omega, np.log10(np.asarray(out_omegas)), err,
                c=err, cmap="plasma", s=15, edgecolor="black", linewidth=0.2)
    ax4.set_xlabel(r"$\log_{10} \omega_{in}$", fontsize=8)
    ax4.set_ylabel(r"$\log_{10} \omega_{out}$", fontsize=8)
    ax4.set_zlabel("octave err", fontsize=8)
    ax4.view_init(elev=20, azim=-60)

    plt.tight_layout()
    out = FIG_DIR / "panel_03_cycle_closure.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 04: Amino Acid S-Coords
# ============================================================
def panel_04():
    d = load("04_amino_acid_coords")
    fig = make_panel_figure()

    coord_table = d["coordinates"]
    aas = [c["aa"] for c in coord_table]
    sk = np.array([c["Sk"] for c in coord_table])
    st = np.array([c["St"] for c in coord_table])
    se = np.array([c["Se"] for c in coord_table])

    # Family colours
    hydrophobic = {"V", "I", "L", "F", "M"}
    charged = {"D", "E", "K", "R"}
    small = {"G", "A", "S"}

    def colour(aa: str) -> str:
        if aa in hydrophobic:
            return "#C44E52"
        if aa in charged:
            return "#4C72B0"
        if aa in small:
            return "#55A868"
        return "#888888"

    colors = [colour(aa) for aa in aas]

    # (A) Sk vs St
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sk, st, c=colors, s=80, edgecolor="black", linewidth=0.6)
    for x, y, lbl in zip(sk, st, aas):
        ax1.text(x, y + 0.025, lbl, fontsize=7, ha="center")
    ax1.set_xlabel(r"$S_k$ (hydrophobic)")
    ax1.set_ylabel(r"$S_t$ (volume)")
    ax1.set_xlim(-0.05, 1.1)
    ax1.set_ylim(-0.05, 1.15)

    # (B) Sk vs Se
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(sk, se, c=colors, s=80, edgecolor="black", linewidth=0.6)
    for x, y, lbl in zip(sk, se, aas):
        ax2.text(x, y + 0.025, lbl, fontsize=7, ha="center")
    ax2.set_xlabel(r"$S_k$")
    ax2.set_ylabel(r"$S_e$ (charge)")
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.15)

    # (C) Family means bar
    ax3 = fig.add_subplot(1, 4, 3)
    means = d["family_means"]
    ax3.bar(
        [r"hydro$\langle S_k\rangle$", r"charged$\langle S_e\rangle$", r"small$\langle S_t\rangle$"],
        [means["hydrophobic_Sk_mean"], means["charged_Se_mean"], means["small_St_mean"]],
        color=["#C44E52", "#4C72B0", "#55A868"],
        edgecolor="black", linewidth=0.6,
    )
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("mean coordinate")
    ax3.tick_params(axis="x", labelsize=7)

    # (D) 3D scatter of all 20 amino acids
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(sk, st, se, c=colors, s=70, edgecolor="black", linewidth=0.6)
    for x, y, z, lbl in zip(sk, st, se, aas):
        ax4.text(x, y, z + 0.04, lbl, fontsize=6.5, ha="center")
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=18, azim=-50)

    plt.tight_layout()
    out = FIG_DIR / "panel_04_amino_acid_coords.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 05: Tau-Assignment
# ============================================================
def panel_05():
    d = load("05_tau_assignment")
    fig = make_panel_figure()

    sweep = d["perturbation_sweep"]
    strengths = [s["perturbation_strength"] for s in sweep]
    fracs = np.array([s["tau_fractions"] for s in sweep])

    # (A) Stacked bar: tau fractions vs perturbation
    ax1 = fig.add_subplot(1, 4, 1)
    bottoms = np.zeros(len(strengths))
    colors_t = ["#4C72B0", "#55A868", "#C44E52"]
    labels_t = [r"$\tau$=0", r"$\tau$=1", r"$\tau$=2"]
    for j in range(3):
        ax1.bar(strengths, fracs[:, j], bottom=bottoms,
                width=0.08, color=colors_t[j], label=labels_t[j],
                edgecolor="black", linewidth=0.4)
        bottoms += fracs[:, j]
    ax1.set_xlabel("perturbation")
    ax1.set_ylabel(r"$\tau$ fraction")
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False, fontsize=7, loc="upper right")

    # (B) Chi^2 vs perturbation
    ax2 = fig.add_subplot(1, 4, 2)
    chi2 = [s["chi_squared_vs_eq_baseline"] for s in sweep]
    ax2.plot(strengths, chi2, "o-", color="#C44E52", linewidth=1.5, markersize=7,
             markeredgecolor="black", markeredgewidth=0.6)
    ax2.fill_between(strengths, 0, chi2, color="#C44E52", alpha=0.18)
    ax2.set_xlabel("perturbation")
    ax2.set_ylabel(r"$\chi^2$ vs eq")

    # (C) Delta_pi std vs perturbation
    ax3 = fig.add_subplot(1, 4, 3)
    stds = [s["delta_pi_std"] for s in sweep]
    means = [s["delta_pi_mean"] for s in sweep]
    ax3.plot(strengths, stds, "s-", color="#4C72B0", label=r"std $\Delta\Pi$",
             linewidth=1.5, markersize=6, markeredgecolor="black", markeredgewidth=0.6)
    ax3.plot(strengths, [abs(m) for m in means], "^-", color="#55A868", label=r"|mean $\Delta\Pi$|",
             linewidth=1.5, markersize=6, markeredgecolor="black", markeredgewidth=0.6)
    ax3.set_xlabel("perturbation")
    ax3.set_ylabel(r"$\Delta\Pi$")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D bars: tau counts (ground/natural/excited) vs perturbation
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    counts = np.array([s["tau_counts"] for s in sweep])
    xs, ys, zs = [], [], []
    dxs, dys, dzs = [], [], []
    cols = []
    for i, p in enumerate(strengths):
        for j in range(3):
            xs.append(p)
            ys.append(j)
            zs.append(0)
            dxs.append(0.07)
            dys.append(0.6)
            dzs.append(counts[i, j])
            cols.append(colors_t[j])
    ax4.bar3d(xs, ys, zs, dxs, dys, dzs, color=cols, edgecolor="black", linewidth=0.3)
    ax4.set_xlabel("perturbation", fontsize=8)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels([r"$\tau$=0", r"$\tau$=1", r"$\tau$=2"], fontsize=7)
    ax4.set_zlabel("count", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_05_tau_assignment.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 06: Spin-Crossover (P450 catalytic cycle)
# ============================================================
def panel_06():
    d = load("06_spin_crossover")
    fig = make_panel_figure()

    states = d["catalytic_states"]
    idx = [s["index"] for s in states]
    S_tot = [s["S_total"] for s in states]
    s_orb = [s["s_orbital_fe"] for s in states]
    fe_ox = [int(s["fe_oxidation"]) for s in states]
    fe_d = [s["fe_d_count"] for s in states]

    # (A) Total spin S along cycle
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(idx, S_tot, "o-", color="#C44E52", linewidth=1.6, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6)
    ax1.fill_between(idx, 0, S_tot, color="#C44E52", alpha=0.18)
    ax1.set_xlabel("catalytic state #")
    ax1.set_ylabel(r"$S_{tot}$")
    ax1.set_xticks(idx)

    # (B) s_orbital invariance (constant 0.5)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(idx, s_orb, "o-", color="#4C72B0", linewidth=1.6, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(idx)
    ax2.set_xlabel("catalytic state #")
    ax2.set_ylabel(r"$s_{orbital}$")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    # (C) Fe oxidation and d-count side by side
    ax3 = fig.add_subplot(1, 4, 3)
    width = 0.4
    x = np.array(idx)
    ax3.bar(x - width / 2, fe_ox, width, color="#55A868", label="Fe ox", edgecolor="black", linewidth=0.4)
    ax3.bar(x + width / 2, fe_d, width, color="#4C72B0", label=r"d-count", edgecolor="black", linewidth=0.4)
    ax3.set_xlabel("catalytic state #")
    ax3.set_ylabel("count / charge")
    ax3.set_xticks(idx)
    ax3.legend(frameon=False, fontsize=7, loc="upper right")

    # (D) 3D path through (state, S_total, oxidation)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    cycle_idx = idx + [idx[0]]
    cycle_S = S_tot + [S_tot[0]]
    cycle_ox = fe_ox + [fe_ox[0]]
    ax4.plot(cycle_idx, cycle_S, cycle_ox, "-", color="#888888", linewidth=1.0, alpha=0.7)
    ax4.scatter(idx, S_tot, fe_ox, c=fe_d, cmap="plasma", s=120,
                edgecolor="black", linewidth=0.6)
    for i, S, ox in zip(idx, S_tot, fe_ox):
        ax4.text(i, S, ox + 0.15, str(i), fontsize=8, ha="center")
    ax4.set_xlabel("state #", fontsize=8)
    ax4.set_ylabel(r"$S_{tot}$", fontsize=8)
    ax4.set_zlabel("Fe ox", fontsize=8)
    ax4.view_init(elev=18, azim=-58)

    plt.tight_layout()
    out = FIG_DIR / "panel_06_spin_crossover.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 07: Kuramoto Synchronization
# ============================================================
def panel_07():
    d = load("07_kuramoto_sync")
    fig = make_panel_figure()

    sweep = d["sweep_results"]
    K0_vals = [r["K0"] for r in sweep]

    # (A) r(t) trajectories for each K0
    ax1 = fig.add_subplot(1, 4, 1)
    cmap = cm.get_cmap("viridis")
    for i, run in enumerate(sweep):
        traj = run["r_trajectory_sample"]
        t_norm = np.linspace(0, 1, len(traj))
        ax1.plot(t_norm, traj, color=cmap(i / max(1, len(sweep) - 1)),
                 linewidth=1.4, label=f"K={run['K0']:.1f}")
    ax1.axhline(0.8, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("t / T")
    ax1.set_ylabel("r(t)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=6, loc="lower right")

    # (B) <r> vs K0 (synchronization curve)
    ax2 = fig.add_subplot(1, 4, 2)
    r_mean = [r["r_mean_last_quarter"] for r in sweep]
    ax2.semilogx(K0_vals, r_mean, "o-", color="#C44E52", linewidth=1.6, markersize=8,
                 markeredgecolor="black", markeredgewidth=0.6)
    ax2.fill_between(K0_vals, 0, r_mean, color="#C44E52", alpha=0.18)
    ax2.axhline(0.8, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(r"$K_0$")
    ax2.set_ylabel(r"$\langle r\rangle$")
    ax2.set_ylim(0, 1.05)

    # (C) Energy descent: H_initial vs H_final
    ax3 = fig.add_subplot(1, 4, 3)
    h_i = [r["H_initial"] for r in sweep]
    h_f = [r["H_final"] for r in sweep]
    width = 0.35
    x = np.arange(len(K0_vals))
    ax3.bar(x - width / 2, h_i, width, color="#4C72B0", label="initial", edgecolor="black", linewidth=0.4)
    ax3.bar(x + width / 2, h_f, width, color="#55A868", label="final", edgecolor="black", linewidth=0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{k:.1f}" for k in K0_vals], fontsize=7)
    ax3.set_xlabel(r"$K_0$")
    ax3.set_ylabel("H")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D surface: r(t) vs K0 vs t
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_t = min(len(r["r_trajectory_sample"]) for r in sweep)
    R = np.array([r["r_trajectory_sample"][:n_t] for r in sweep])
    K = np.array(K0_vals)
    T = np.linspace(0, 1, n_t)
    Tg, Kg = np.meshgrid(T, K)
    surf = ax4.plot_surface(Tg, Kg, R, cmap="viridis", edgecolor="black",
                            linewidth=0.15, alpha=0.92, rcount=12, ccount=12)
    ax4.set_xlabel("t / T", fontsize=8)
    ax4.set_ylabel(r"$K_0$", fontsize=8)
    ax4.set_zlabel("r(t)", fontsize=8)
    ax4.set_zlim(0, 1)
    ax4.view_init(elev=22, azim=-60)

    plt.tight_layout()
    out = FIG_DIR / "panel_07_kuramoto_sync.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Panel 08: Morphism Chain
# ============================================================
def panel_08():
    d = load("08_morphism_chain")
    fig = make_panel_figure()

    contact_map = np.array(d["contact_map"]["matrix"])
    n = contact_map.shape[0]

    # (A) Entropy through chain
    ax1 = fig.add_subplot(1, 4, 1)
    et = d["entropy_trace"]
    steps = ["observe", "catalyze*", "fuse"]
    s_vals = [et["S_after_observe"], et["S_after_catalyze"], et["S_after_fuse"]]
    ax1.plot(steps, s_vals, "o-", color="#4C72B0", linewidth=1.6, markersize=10,
             markeredgecolor="black", markeredgewidth=0.6)
    ax1.set_ylabel("S (Sigma)")
    ax1.tick_params(axis="x", labelsize=8)

    # (B) Contact map heatmap
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(contact_map, cmap="Greys", origin="lower", interpolation="nearest")
    ax2.set_xlabel("residue j")
    ax2.set_ylabel("residue i")
    ax2.set_xticks(range(0, n))
    ax2.set_yticks(range(0, n))
    ax2.tick_params(labelsize=6)

    # (C) Entropy bound check (actual vs theoretical)
    ax3 = fig.add_subplot(1, 4, 3)
    actual = et["actual_change"]
    bound = et["theoretical_bound"]
    ax3.bar(["actual", "bound"], [actual, bound],
            color=["#55A868", "#C44E52"], edgecolor="black", linewidth=0.6)
    ax3.set_ylabel(r"$|\Delta S|$")

    # (D) 3D surface of fused signature (reconstructed by re-running sigma)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    # Use the contact map as the surface; render as 3D bar
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    # Build a smooth surface: contact map values at integer grid
    Z = contact_map.astype(float)
    surf = ax4.plot_surface(X, Y, Z, cmap="Greys",
                            edgecolor="black", linewidth=0.15, rcount=n, ccount=n)
    ax4.set_xlabel("i", fontsize=8)
    ax4.set_ylabel("j", fontsize=8)
    ax4.set_zlabel("contact", fontsize=8)
    ax4.set_zlim(0, 1.2)
    ax4.view_init(elev=30, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_08_morphism_chain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    panels = [
        panel_01, panel_02, panel_03, panel_04,
        panel_05, panel_06, panel_07, panel_08,
    ]
    paths = []
    for fn in panels:
        path = fn()
        paths.append(path)
        print(f"  -> {path.name}")
    print(f"\nGenerated {len(paths)} panels in {FIG_DIR}")


if __name__ == "__main__":
    main()
