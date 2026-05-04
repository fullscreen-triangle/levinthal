"""Generate Paper 5 panels: 4 charts in a row, 3D in each, white background."""

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
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load(name): return json.load((RESULTS / f"{name}.json").open())


def make_fig(): return plt.figure(figsize=(16, 4), facecolor="white")


# ============================================================
def panel_01():
    d = load("01_peroxo_state")
    fig = make_fig()

    states = d["states"]

    # (A) M values across 3 states
    ax1 = fig.add_subplot(1, 4, 1)
    names = ["Fe³⁺ HS\n(state 2)", "Cpd 0\n(state 5)", "Cpd I\n(state 6)"]
    Ms = [states["Fe_HS_state2"]["M"], states["Cpd0_peroxo_state5"]["M"],
          states["CpdI_state6"]["M"]]
    colors = ["#4C72B0", "#FFA500", "#C44E52"]
    ax1.bar(names, Ms, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\mathcal{M}$ (partition depth)")
    ax1.tick_params(axis="x", labelsize=7)

    # (B) S-coordinates across 3 states
    ax2 = fig.add_subplot(1, 4, 2)
    coords = [states[k]["S"] for k in ["Fe_HS_state2", "Cpd0_peroxo_state5", "CpdI_state6"]]
    sk_v = [c[0] for c in coords]
    st_v = [c[1] for c in coords]
    se_v = [c[2] for c in coords]
    x = range(3)
    ax2.plot(x, sk_v, "o-", color="#C44E52", label=r"$S_k$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.plot(x, st_v, "s-", color="#55A868", label=r"$S_t$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.plot(x, se_v, "^-", color="#4C72B0", label=r"$S_e$",
             linewidth=1.5, markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Fe³⁺", "Cpd 0", "Cpd I"], fontsize=7)
    ax2.set_ylabel("S-coord")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_ylim(0, 1)

    # (C) Pairwise S-distances
    ax3 = fig.add_subplot(1, 4, 3)
    distances = d["distances"]
    pairs = list(distances.keys())
    dists = list(distances.values())
    ax3.bar([p.replace("_to_", "→") for p in pairs], dists,
            color="#FFA500", edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("S-distance")
    ax3.tick_params(axis="x", labelsize=7)

    # (D) 3D scatter of three states
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    state_names = ["Fe HS", "Cpd 0", "Cpd I"]
    for i, (s, name, c) in enumerate(zip(coords, state_names, colors)):
        ax4.scatter(s[0], s[1], s[2], color=c, s=120,
                    edgecolor="black", linewidth=0.6)
        ax4.text(s[0], s[1], s[2] + 0.025, name, fontsize=8, ha="center")
    # Connect with chain
    ax4.plot([c[0] for c in coords], [c[1] for c in coords],
             [c[2] for c in coords], "k--", linewidth=0.8, alpha=0.5)
    ax4.set_xlabel(r"$S_k$", fontsize=8)
    ax4.set_ylabel(r"$S_t$", fontsize=8)
    ax4.set_zlabel(r"$S_e$", fontsize=8)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.set_zlim(0, 1)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_01_peroxo_state.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_02():
    d = load("02_bond_order_coordinate")
    fig = make_fig()

    # (A) Binary states
    ax1 = fig.add_subplot(1, 4, 1)
    states = d["bond_states"]
    labels = [s["label"] for s in states]
    contributions = [s["partition_contribution"] for s in states]
    ax1.bar(labels, contributions, color=["#C44E52", "#55A868"],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("partition contribution")
    ax1.set_xlabel(r"bond-order coord $\beta$")

    # (B) Delta_M comparison
    ax2 = fig.add_subplot(1, 4, 2)
    schemes = list(d["schemes_comparison"].keys())
    schemes_short = [s.replace("_", "\n") for s in schemes]
    delta_Ms = []
    for s in schemes:
        v = d["schemes_comparison"][s]["delta_M"]
        delta_Ms.append(v if isinstance(v, (int, float)) else 0)
    ax2.bar(schemes_short, delta_Ms, color="#4C72B0",
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$\Delta\mathcal{M}$")
    ax2.tick_params(axis="x", labelsize=6)

    # (C) Cleavage time
    ax3 = fig.add_subplot(1, 4, 3)
    tau_p = d["tau_p_fs"]
    tau_cleave = d["tau_cleave_fs"]
    ax3.bar([r"$\tau_p$", r"$\tau_{\mathrm{cleave}}$"], [tau_p, tau_cleave],
            color=["#FFA500", "#C44E52"], edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("time (fs)")

    # (D) 3D landscape: M vs N states
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_grid = np.arange(2, 10)
    n_axis = np.arange(2, 10)
    NG, NA = np.meshgrid(n_grid, n_axis)
    M_landscape = np.log(NG)
    surf = ax4.plot_surface(NG, NA, M_landscape, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    # Mark binary
    ax4.scatter([2], [2], [math.log(2)], color="red", s=100,
                edgecolor="black", linewidth=0.5)
    ax4.set_xlabel("# states", fontsize=8)
    ax4.set_ylabel("repetition", fontsize=8)
    ax4.set_zlabel(r"$\Delta\mathcal{M}$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_02_bond_order.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_03():
    d = load("03_oo_heterolysis_aperture")
    fig = make_fig()

    # (A) Selection rules
    ax1 = fig.add_subplot(1, 4, 1)
    sr = d["selection_rules"]
    rule_names = [r"$\Delta\beta$", r"$\Delta s_{orb}$", r"$|\Delta m|_{max}$"]
    rule_values = [sr["delta_beta"], sr["delta_s_orbital"], sr["delta_m_max"]]
    ax1.bar(rule_names, rule_values, color=["#4C72B0", "#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.4)
    ax1.set_ylabel("Δ value")

    # (B) E_a comparison
    ax2 = fig.add_subplot(1, 4, 2)
    E_a_pred = d["activation_energy_kcal_per_mol"]
    E_range = d["experimental_range_kcal_per_mol"]
    ax2.bar(["predicted", "exp min", "exp max"],
            [E_a_pred, E_range[0], E_range[1]],
            color=["#FFA500", "#4C72B0", "#4C72B0"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$E_a$ (kcal/mol)")

    # (C) k_intrinsic vs d_C
    ax3 = fig.add_subplot(1, 4, 3)
    dCs = list(range(1, 7))
    kcat_pred = [10 ** (10 - dc) for dc in dCs]
    log_kcat = [math.log10(k) for k in kcat_pred]
    ax3.plot(dCs, log_kcat, "o-", color="#4C72B0", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax3.axvline(1, color="red", linestyle="--", linewidth=0.8,
                label="Cpd I formation")
    ax3.set_xlabel(r"$d_C$")
    ax3.set_ylabel(r"$\log_{10}(k_{intrinsic})$")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D landscape: k_intrinsic vs (d_C, T)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    dC_grid = np.arange(1, 7)
    T_grid = np.linspace(193, 350, 30)
    DG, TG = np.meshgrid(dC_grid, T_grid)
    KB_local = 1.38e-23
    HBAR_local = 1.05e-34
    log_k = np.log10(KB_local * TG / HBAR_local) - DG
    surf = ax4.plot_surface(DG, TG, log_k, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel(r"$d_C$", fontsize=8)
    ax4.set_ylabel("T (K)", fontsize=8)
    ax4.set_zlabel(r"$\log_{10}(k)$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_03_aperture.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_04():
    d = load("04_pcet_concerted")
    fig = make_fig()

    # (A) Concerted vs sequential rates
    ax1 = fig.add_subplot(1, 4, 1)
    conc = d["concerted"]
    seq = d["sequential"]
    ax1.bar(["concerted\n(d_C=1)", "sequential\n(d_C=2)"],
            [math.log10(conc["predicted_intrinsic_rate_per_s"]),
             math.log10(seq["predicted_intrinsic_rate_per_s"])],
            color=["#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\log_{10}(k_{intrinsic})$ s⁻¹")

    # (B) KIE comparison
    ax2 = fig.add_subplot(1, 4, 2)
    kies = [conc["predicted_KIE"], seq["predicted_KIE"],
            d["discrimination"]["experimental_KIE"]]
    ax2.bar(["concerted\npred", "sequential\npred", "experimental"],
            kies, color=["#55A868", "#C44E52", "#4C72B0"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("KIE")

    # (C) Rate-ratio prediction
    ax3 = fig.add_subplot(1, 4, 3)
    ratio = d["discrimination"]["rate_ratio_concerted_to_sequential"]
    ax3.bar(["framework"], [ratio], color="#FFA500",
            edgecolor="black", linewidth=0.5)
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=0.5,
                label="Marcus baseline")
    ax3.set_ylabel("rate ratio (concerted/sequential)")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D: rate(d_C, KIE)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    dC_grid = np.arange(1, 5)
    kie_grid = np.linspace(1, 8, 30)
    DG, KG = np.meshgrid(dC_grid, kie_grid)
    log_k = (10 - DG) - 0.1 * (KG - 2)  # KIE damping
    surf = ax4.plot_surface(DG, KG, log_k, cmap="plasma",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.scatter([1], [conc["predicted_KIE"]], [math.log10(conc["predicted_intrinsic_rate_per_s"])],
                color="green", s=100, edgecolor="black", linewidth=0.5)
    ax4.scatter([2], [seq["predicted_KIE"]], [math.log10(seq["predicted_intrinsic_rate_per_s"])],
                color="red", s=100, edgecolor="black", linewidth=0.5)
    ax4.set_xlabel(r"$d_C$", fontsize=8)
    ax4.set_ylabel("KIE", fontsize=8)
    ax4.set_zlabel(r"$\log_{10}(k)$", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_04_pcet.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_05():
    d = load("05_anharmonic_recurrence")
    fig = make_fig()

    # (A) Morse potential vs harmonic
    ax1 = fig.add_subplot(1, 4, 1)
    r_axis = np.linspace(1.0, 3.0, 100)
    morse_param = d["morse_parameters"]
    De = morse_param["De_kcal_per_mol"]
    r0 = morse_param["r_eq_A"]
    alpha = morse_param["alpha_per_A"]
    morse_V = De * (1 - np.exp(-alpha * (r_axis - r0))) ** 2
    harmonic_V = 0.5 * 2 * De * alpha ** 2 * (r_axis - r0) ** 2
    ax1.plot(r_axis, morse_V, color="#C44E52", linewidth=2.0,
             label="Morse")
    ax1.plot(r_axis, harmonic_V, color="#4C72B0", linewidth=1.5,
             linestyle="--", label="harmonic")
    ax1.axvline(r0, color="black", linestyle=":", linewidth=0.5)
    ax1.set_xlabel("r (Å)")
    ax1.set_ylabel("V (kcal/mol)")
    ax1.set_ylim(0, 80)
    ax1.legend(frameon=False, fontsize=7)

    # (B) Asymmetry quantification
    ax2 = fig.add_subplot(1, 4, 2)
    anh = d["anharmonicity"]
    ax2.bar(["asymmetry", "freq amp\ndependence"],
            [anh["potential_asymmetry"],
             anh["frequency_amplitude_dependence"]],
            color=["#FFA500", "#55A868"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("anharmonicity metric")

    # (C) Trajectory samples
    ax3 = fig.add_subplot(1, 4, 3)
    samples = d["trajectory_samples"]
    min_returns = [s["min_return_distance"] for s in samples]
    ax3.bar(range(len(samples)), min_returns, color="#4C72B0",
            edgecolor="black", linewidth=0.4)
    ax3.axhline(1e-12, color="red", linestyle="--", linewidth=0.5,
                label="exact-recurrence floor")
    ax3.set_xlabel("trajectory #")
    ax3.set_ylabel("min return distance (Å)")
    ax3.set_yscale("log")
    ax3.legend(frameon=False, fontsize=7)

    # (D) 3D Morse landscape
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    r_grid = np.linspace(1.0, 3.0, 40)
    De_grid = np.linspace(40, 80, 40)
    RG, DG = np.meshgrid(r_grid, De_grid)
    V = DG * (1 - np.exp(-alpha * (RG - r0))) ** 2
    surf = ax4.plot_surface(RG, DG, V, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel("r (Å)", fontsize=8)
    ax4.set_ylabel(r"$D_e$ (kcal/mol)", fontsize=8)
    ax4.set_zlabel("V (kcal/mol)", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_05_anharmonic.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_06():
    d = load("06_cpdI_lifetime")
    fig = make_fig()

    # (A) Lifetime vs temperature
    ax1 = fig.add_subplot(1, 4, 1)
    lt = d["lifetimes_per_temperature"]
    Ts = [l["T_K"] for l in lt]
    taus = [l["tau_ms"] for l in lt]
    ax1.semilogy(Ts, taus, "o-", color="#4C72B0", linewidth=1.5,
                 markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax1.axvline(193, color="red", linestyle="--", linewidth=0.5,
                label="Rittle-Green 193 K")
    ax1.axvline(310, color="green", linestyle="--", linewidth=0.5,
                label="physiological 310 K")
    ax1.set_xlabel("T (K)")
    ax1.set_ylabel(r"$\tau_{Cpd\,I}$ (ms)")
    ax1.legend(frameon=False, fontsize=7)

    # (B) Predicted vs experimental at two T
    ax2 = fig.add_subplot(1, 4, 2)
    pred = d["specific_predictions"]
    ax2.bar(["193K\npred", "193K\nexp", "310K\npred", "310K\nexp"],
            [pred["tau_193K_ms_predicted"], pred["tau_193K_ms_experimental"],
             pred["tau_310K_ms_predicted"], pred["tau_310K_ms_experimental"]],
            color=["#FFA500", "#4C72B0", "#FFA500", "#4C72B0"],
            edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\tau$ (ms)")
    ax2.tick_params(axis="x", labelsize=7)

    # (C) Activation energy
    ax3 = fig.add_subplot(1, 4, 3)
    ae = d["activation_energy"]
    ax3.bar(["from ΔM", "paper"],
            [ae["E_a_from_DeltaM_kcal"], ae["E_a_paper_kcal"]],
            color=["#55A868", "#C44E52"],
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"$E_a$ (kcal/mol)")

    # (D) 3D: tau(T, E_a)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    T_grid = np.linspace(150, 350, 40)
    Ea_grid = np.linspace(8, 16, 40)
    TG, EG = np.meshgrid(T_grid, Ea_grid)
    KB_local = 1.38e-23
    HBAR_local = 1.05e-34
    kbT_kcal = (KB_local * TG / 4184.0) * 6.022e23
    tau_p = HBAR_local / (KB_local * TG)
    log_tau_ms = np.log10(tau_p * np.exp(EG / kbT_kcal) * 1000)
    surf = ax4.plot_surface(TG, EG, log_tau_ms, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.set_xlabel("T (K)", fontsize=8)
    ax4.set_ylabel(r"$E_a$ (kcal/mol)", fontsize=8)
    ax4.set_zlabel(r"$\log_{10}(\tau)$ ms", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_06_lifetime.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_07():
    d = load("07_oxidation_potential")
    fig = make_fig()

    # (A) Decomposed contributions
    ax1 = fig.add_subplot(1, 4, 1)
    decomp = d["decomposed_contributions_V"]
    names = list(decomp.keys())
    short = [n.replace("_", "\n") for n in names]
    Es = [decomp[n]["E_contribution_V"] for n in names]
    ax1.bar(short, Es, color=cm.viridis(np.linspace(0, 0.85, len(names))),
            edgecolor="black", linewidth=0.4)
    ax1.set_ylabel("E contribution (V)")
    ax1.tick_params(axis="x", labelsize=6)

    # (B) Total predicted vs experimental
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(["predicted", "experimental"],
            [d["E_predicted_V"], d["E_experimental_V"]],
            color=["#FFA500", "#4C72B0"],
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$E^\circ$ (V vs NHE)")

    # (C) n_eff sweep
    ax3 = fig.add_subplot(1, 4, 3)
    sw = d["n_eff_sweep"]
    ax3.plot([s["n_eff"] for s in sw], [s["E_V"] for s in sw],
             "o-", color="#C44E52", linewidth=1.5,
             markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax3.axvline(d["n_eff"], color="black", linestyle="--", linewidth=0.5)
    ax3.set_xlabel(r"$n_{eff}$")
    ax3.set_ylabel(r"$E^\circ$ (V)")

    # (D) 3D: E(n_eff, ΔM_cumulative)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_grid = np.arange(1, 7)
    dM_grid = np.linspace(2, 6, 30)
    NG, DM = np.meshgrid(n_grid, dM_grid)
    KB_T_J = 4.27e-21
    e = 1.6e-19
    E_surface = (KB_T_J / e) * NG * DM
    surf = ax4.plot_surface(NG, DM, E_surface, cmap="viridis",
                            edgecolor="black", linewidth=0.1, alpha=0.92)
    ax4.scatter([d["n_eff"]], [d["Delta_M_cumulative"]], [d["E_predicted_V"]],
                color="red", s=100, edgecolor="black", linewidth=0.5)
    ax4.set_xlabel(r"$n_{eff}$", fontsize=8)
    ax4.set_ylabel(r"$\Delta\mathcal{M}_{cumulative}$", fontsize=8)
    ax4.set_zlabel(r"$E^\circ$ (V)", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_07_potential.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return out


# ============================================================
def panel_08():
    d = load("08_spectroscopic_observables")
    fig = make_fig()

    cmp = d["observable_comparison"]

    # (A) Predicted vs experimental scatter
    ax1 = fig.add_subplot(1, 4, 1)
    pred_v = [c["predicted"] for c in cmp]
    exp_v = [c["experimental"] for c in cmp]
    obs_names = [c["observable"][:15] for c in cmp]
    ax1.scatter(pred_v, exp_v, s=80, c=range(len(cmp)),
                cmap="viridis", edgecolor="black", linewidth=0.5)
    # diagonal
    max_v = max(max(pred_v), max(exp_v))
    ax1.plot([0, max_v * 1.1], [0, max_v * 1.1], "k--", linewidth=0.5)
    ax1.set_xlabel("predicted")
    ax1.set_ylabel("experimental")

    # (B) Per-observable rel error
    ax2 = fig.add_subplot(1, 4, 2)
    errors = [c["relative_error"] for c in cmp]
    ax2.barh([c["observable"][:18] for c in cmp], errors,
             color=cm.RdYlGn_r(np.array(errors)),
             edgecolor="black", linewidth=0.4)
    ax2.axvline(0.20, color="black", linestyle="--", linewidth=0.5,
                label="20% threshold")
    ax2.set_xlabel("relative error")
    ax2.tick_params(axis="y", labelsize=6)
    ax2.legend(frameon=False, fontsize=7)

    # (C) Agreement count
    ax3 = fig.add_subplot(1, 4, 3)
    n_agree = d["n_agreement_within_20pct"]
    n_total = d["n_total_observables"]
    ax3.pie([n_agree, n_total - n_agree],
            labels=["agree", "deviate"],
            colors=["#55A868", "#C44E52"],
            startangle=90, autopct="%d/%d" % (n_agree, n_total),
            textprops={"fontsize": 9},
            wedgeprops=dict(edgecolor="black", linewidth=0.5))

    # (D) 3D bar landscape
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n = len(cmp)
    xpos = np.arange(n)
    pred = np.array([c["predicted"] for c in cmp])
    exp_arr = np.array([c["experimental"] for c in cmp])
    ax4.bar3d(xpos - 0.3, np.zeros(n), np.zeros(n),
              0.5, 0.4, pred, color="#FFA500",
              edgecolor="black", linewidth=0.3)
    ax4.bar3d(xpos - 0.3, np.ones(n), np.zeros(n),
              0.5, 0.4, exp_arr, color="#4C72B0",
              edgecolor="black", linewidth=0.3)
    ax4.set_xticks(xpos)
    ax4.set_xticklabels([c["observable"][:8] for c in cmp],
                        fontsize=5, rotation=30)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["pred", "exp"], fontsize=7)
    ax4.set_zlabel("value", fontsize=8)
    ax4.view_init(elev=22, azim=-55)

    plt.tight_layout()
    out = FIG_DIR / "panel_08_spectroscopy.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
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
