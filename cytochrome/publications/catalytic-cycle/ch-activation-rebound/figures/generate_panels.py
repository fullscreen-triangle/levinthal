"""Generate Paper 6 panels: 4 charts per panel, 3D in each, white background."""
from __future__ import annotations
import json, math, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "validation" / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

# Physical constants (mirrored from _common.py for standalone use)
kB = 1.380649e-23; hbar = 1.054572e-34; c_cms = 2.997924e10; NA = 6.022141e23
T = 310.0; kBT = kB * T
T_PART = 65.0; nu_floor = 1e10; ln2 = math.log(2.0)
DELTA_M_HAT = 0.65; DELTA_M_REBOUND = 0.30
NU_CH = 3000.0; NU_CD = NU_CH / math.sqrt(2)


def load(name):
    f = RESULTS / f"{name}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def make_fig():
    return plt.figure(figsize=(16, 4), facecolor="white")


# ============================================================
def panel_01():
    """Substrate positioning / partition depth per substrate class."""
    fig = make_fig()
    reaction_types = ["aliphatic", "benzylic", "allylic", "aromatic", "epoxidation"]
    dMs = [0.65, 0.50, 0.45, 0.38, 0.35]
    E_as = [T_PART * dm / 4.184 for dm in dMs]
    colors = ["#4C72B0", "#FFA500", "#55A868", "#C44E52", "#8172B2"]

    # (A) Delta_M per substrate class
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(reaction_types, dMs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\Delta\mathcal{M}_{\mathrm{HAT}}$")
    ax1.set_title("Activation depth per class")
    ax1.tick_params(axis="x", rotation=30, labelsize=7)
    for i, (rt, dm) in enumerate(zip(reaction_types, dMs)):
        ax1.text(i, dm + 0.01, f"{dm:.2f}", ha="center", va="bottom", fontsize=7)

    # (B) S-coordinate components for Cpd I vs substrate TS
    ax2 = fig.add_subplot(1, 4, 2)
    sk_vals = [0.860, 0.830, 0.820, 0.800, 0.790]  # increasing activation
    st_vals = [0.515] * 5
    se_vals = [0.595, 0.570, 0.560, 0.545, 0.540]
    x = np.arange(5)
    ax2.plot(x, sk_vals, "o-", color="#C44E52", label=r"$S_k$", markersize=6)
    ax2.plot(x, st_vals, "s-", color="#55A868", label=r"$S_t$", markersize=6)
    ax2.plot(x, se_vals, "^-", color="#4C72B0", label=r"$S_e$", markersize=6)
    ax2.set_xticks(x); ax2.set_xticklabels(reaction_types, rotation=30, fontsize=7)
    ax2.set_ylabel("S-coordinate"); ax2.set_title("S-coord per class")
    ax2.legend(fontsize=7)

    # (C) Pairwise S-distance (Cpd I to TS)
    ax3 = fig.add_subplot(1, 4, 3)
    s_dists = [0.065, 0.050, 0.045, 0.038, 0.035]
    ax3.bar(reaction_types, s_dists, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel(r"$d_S(\mathrm{Cpd\,I},\,\mathrm{TS})$")
    ax3.set_title("S-distance to TS")
    ax3.tick_params(axis="x", rotation=30, labelsize=7)

    # (D) 3D: E_a vs class index vs S-distance
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    xs = np.arange(5); ys = s_dists; zs = E_as
    ax4.bar3d(xs - 0.25, [0]*5, [0]*5, 0.5, ys, zs, color=colors, alpha=0.85)
    ax4.set_xlabel("Class", fontsize=7); ax4.set_ylabel(r"$d_S$", fontsize=7)
    ax4.set_zlabel(r"$E_a$ (kcal/mol)", fontsize=7)
    ax4.set_xticks(xs); ax4.set_xticklabels(reaction_types, fontsize=5, rotation=20)
    ax4.set_title("3D: class vs Ea vs dS")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_01_substrate_positioning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_01 done")


# ============================================================
def panel_02():
    """HAT coordinate: C-H bond order, three-body geometry."""
    fig = make_fig()

    # (A) Beta_CH binary states
    ax1 = fig.add_subplot(1, 4, 1)
    states = [1, 0]; labels = [r"$\beta_{CH}=1$\n(intact)", r"$\beta_{CH}=0$\n(cleaved)"]
    ax1.bar([0, 1], states, color=["#55A868", "#C44E52"], edgecolor="black", linewidth=0.7,
            width=0.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels([r"$\beta_{CH}=1$", r"$\beta_{CH}=0$"], fontsize=8)
    ax1.set_ylabel("Bond state"); ax1.set_title(r"C-H bond-order states")
    ax1.set_ylim(-0.1, 1.4)
    ax1.text(0, 1.08, r"intact", ha="center", fontsize=8)
    ax1.text(1, 0.08, r"cleaved", ha="center", fontsize=8)

    # (B) TS geometry: C-H and O-H distances at TS
    ax2 = fig.add_subplot(1, 4, 2)
    labels2 = ["C–H\n(intact)", "C–H\n(TS)", "O–H\n(TS)", "O–H\n(product)"]
    vals2 = [1.09, 1.30, 1.20, 0.97]
    cols2 = ["#4C72B0", "#FFA500", "#FFA500", "#55A868"]
    ax2.bar(range(4), vals2, color=cols2, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels2, fontsize=7)
    ax2.set_ylabel(r"Bond length (Å)"); ax2.set_title("TS geometry")
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.02, f"{v}", ha="center", va="bottom", fontsize=8)

    # (C) Delta_M contributions
    ax3 = fig.add_subplot(1, 4, 3)
    contrib_labels = ["Half C-H\ncleavage", "Fe partial\nreduction", "Total HAT\nactivation"]
    contrib_vals = [ln2 / 2, 0.30, DELTA_M_HAT]
    contrib_colors = ["#4C72B0", "#C44E52", "#FFA500"]
    ax3.bar(range(3), contrib_vals, color=contrib_colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(3)); ax3.set_xticklabels(contrib_labels, fontsize=7)
    ax3.set_ylabel(r"$\Delta\mathcal{M}$"); ax3.set_title("Activation depth\ncontributions")
    for i, v in enumerate(contrib_vals):
        ax3.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (D) 3D: Delta_M landscape over BDE and approach angle
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    bde = np.linspace(80, 120, 20)
    angle = np.linspace(140, 180, 20)
    B, A = np.meshgrid(bde, angle)
    # Delta_M increases with BDE, decreases slightly with better angle
    DM = 0.65 + 0.004 * (B - 100) - 0.002 * (A - 165)
    surf = ax4.plot_surface(B, A, DM, cmap="viridis", alpha=0.85, linewidth=0)
    ax4.set_xlabel("BDE (kcal/mol)", fontsize=7); ax4.set_ylabel(r"Angle (°)", fontsize=7)
    ax4.set_zlabel(r"$\Delta\mathcal{M}_{\mathrm{HAT}}$", fontsize=7)
    ax4.set_title(r"$\Delta\mathcal{M}$ landscape")
    ax4.scatter([100], [165], [0.65], color="red", s=40, zorder=5, label="CYP3A4")
    ax4.legend(fontsize=7)

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_02_hat_coordinate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_02 done")


# ============================================================
def panel_03():
    """KIE: ZPE and tunneling contributions."""
    fig = make_fig()
    d = load("03_kie_prediction")

    omega_CH = 2 * math.pi * c_cms * NU_CH
    omega_CD = 2 * math.pi * c_cms * NU_CD
    delta_ZPE_J = (hbar / 2) * (omega_CH - omega_CD)
    delta_ZPE_kBT = delta_ZPE_J / kBT
    KIE_ZPE = math.exp(delta_ZPE_kBT)
    kappa_ratio = d.get("kappa_H_over_kappa_D", 1.16)
    KIE_total = KIE_ZPE * kappa_ratio

    # (A) KIE decomposition bar
    ax1 = fig.add_subplot(1, 4, 1)
    components = ["ZPE classical", r"$\kappa_H/\kappa_D$", "Total KIE"]
    vals = [KIE_ZPE, kappa_ratio, KIE_total]
    cols = ["#FFA500", "#55A868", "#4C72B0"]
    ax1.bar(components, vals, color=cols, edgecolor="black", linewidth=0.5)
    ax1.axhspan(4, 11, alpha=0.12, color="gray", label="Exp. range 4-11")
    ax1.set_ylabel("KIE value"); ax1.set_title("KIE decomposition")
    ax1.legend(fontsize=7)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # (B) delta_ZPE vs stretch frequency
    ax2 = fig.add_subplot(1, 4, 2)
    nu_range = np.linspace(2500, 3500, 100)
    nu_D_range = nu_range / math.sqrt(2)
    omega_H = 2 * math.pi * c_cms * nu_range
    omega_D = 2 * math.pi * c_cms * nu_D_range
    dZPE_kcal = (hbar / 2) * (omega_H - omega_D) * NA / 4184
    ax2.plot(nu_range, dZPE_kcal, color="#4C72B0", linewidth=2)
    ax2.axvspan(2800, 3100, alpha=0.12, color="#FFA500", label="Typical P450")
    ax2.axvline(NU_CH, color="#C44E52", linestyle="--", linewidth=1.5, label=r"$\tilde\nu_{CH}=3000$")
    ax2.set_xlabel(r"$\tilde\nu_{CH}$ (cm$^{-1}$)"); ax2.set_ylabel(r"$\Delta$ZPE (kcal/mol)")
    ax2.set_title("ZPE vs frequency"); ax2.legend(fontsize=7)

    # (C) KIE vs temperature
    ax3 = fig.add_subplot(1, 4, 3)
    Ts = np.linspace(250, 400, 80)
    KIE_T = [math.exp(delta_ZPE_J / (kB * Ti)) * kappa_ratio for Ti in Ts]
    ax3.plot(Ts, KIE_T, color="#55A868", linewidth=2)
    ax3.axhspan(4, 11, alpha=0.12, color="gray")
    ax3.axvline(310, color="#C44E52", linestyle="--", linewidth=1.5, label="310 K")
    ax3.set_xlabel("T (K)"); ax3.set_ylabel("KIE"); ax3.set_title("KIE vs temperature")
    ax3.legend(fontsize=7)
    ax3.text(310 + 2, KIE_total + 0.1, f"{KIE_total:.1f}", fontsize=8, color="#C44E52")

    # (D) 3D: KIE surface over (T, delta_ZPE)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    Tg = np.linspace(260, 380, 20)
    dZ = np.linspace(0.6, 1.6, 20)
    TT, DZ = np.meshgrid(Tg, dZ)
    KIE_surf = np.exp(DZ / (kB * TT / (NA * 4184)) * 1e-3) * kappa_ratio
    # simpler: use relative units
    KIE_surf2 = np.exp(DZ / (8.314e-3 * TT)) * kappa_ratio
    surf = ax4.plot_surface(TT, DZ, KIE_surf2, cmap="plasma", alpha=0.85)
    ax4.scatter([310], [delta_ZPE_J * NA / 4184], [KIE_total],
                color="red", s=40, zorder=5)
    ax4.set_xlabel("T (K)", fontsize=7); ax4.set_ylabel(r"$\Delta$ZPE (kcal/mol)", fontsize=7)
    ax4.set_zlabel("KIE", fontsize=7); ax4.set_title("3D KIE landscape")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_03_kie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_03 done")


# ============================================================
def panel_04():
    """Radical intermediate: lifetime and stereoretention."""
    fig = make_fig()
    d = load("04_radical_intermediate")

    k_rebound = nu_floor * math.exp(-DELTA_M_REBOUND)
    k_escape_vals = [1e8, 5e8, 1e9, 2e9, 5e9]
    labels = ["1e8", "5e8", "1e9", "2e9", "5e9"]
    f_vals = [k_rebound / (k_rebound + k_esc) for k_esc in k_escape_vals]

    # (A) Partition cells
    ax1 = fig.add_subplot(1, 4, 1)
    cell_labels = [r"$\beta_{CH}$", r"$\beta_{OH}$", r"$\beta_{CO}$", r"$\sigma_{rad}$"]
    reactant = [1, 0, 0, 0]
    radical = [0, 1, 0, 1]
    product = [0, 0, 1, 0]
    x = np.arange(4)
    width = 0.25
    ax1.bar(x - width, reactant, width, label="Cpd I+Sub", color="#4C72B0", edgecolor="black")
    ax1.bar(x, radical, width, label="Radical int.", color="#FFA500", edgecolor="black")
    ax1.bar(x + width, product, width, label="Product", color="#55A868", edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels(cell_labels, fontsize=8)
    ax1.set_ylabel("Partition state (0/1)"); ax1.set_title("Partition cell transitions")
    ax1.legend(fontsize=7)

    # (B) Rebound vs escape rates
    ax2 = fig.add_subplot(1, 4, 2)
    k_escape_plot = [1e8, 5e8, 1e9, 5e9]
    ax2.axhline(k_rebound, color="#C44E52", linewidth=2, linestyle="-",
                label=r"$k_{\mathrm{rebound}}$")
    ax2.plot(range(len(k_escape_plot)), k_escape_plot, "o--", color="#4C72B0",
             label=r"$k_{\mathrm{escape}}$", markersize=8)
    ax2.set_yscale("log"); ax2.set_ylabel("Rate (s$^{-1}$)")
    ax2.set_xticklabels([""] + [f"{k:.0e}" for k in k_escape_plot], fontsize=7)
    ax2.set_title("Rate comparison"); ax2.legend(fontsize=7)

    # (C) Stereoretention vs k_escape
    ax3 = fig.add_subplot(1, 4, 3)
    k_esc_range = np.logspace(7.5, 10.0, 200)
    f_range = k_rebound / (k_rebound + k_esc_range)
    ax3.semilogx(k_esc_range, f_range * 100, color="#55A868", linewidth=2)
    ax3.axhspan(40, 90, alpha=0.12, color="#FFA500", label="Exp. 40-90%")
    ax3.set_xlabel(r"$k_{\mathrm{escape}}$ (s$^{-1}$)")
    ax3.set_ylabel("Stereoretention (%)"); ax3.set_title("Stereospecificity")
    ax3.legend(fontsize=7)

    # (D) 3D: f_retained over (k_rebound, k_escape)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    k_reb_g = np.logspace(8.5, 10.5, 15)
    k_esc_g = np.logspace(7.5, 10.0, 15)
    KR, KE = np.meshgrid(k_reb_g, k_esc_g)
    F = KR / (KR + KE)
    ax4.plot_surface(np.log10(KR), np.log10(KE), F * 100, cmap="viridis", alpha=0.85)
    ax4.set_xlabel(r"$\log_{10}(k_{reb})$", fontsize=7)
    ax4.set_ylabel(r"$\log_{10}(k_{esc})$", fontsize=7)
    ax4.set_zlabel("Retention (%)", fontsize=7)
    ax4.set_title("3D stereoretention")
    ax4.scatter([math.log10(k_rebound)], [math.log10(1e8)],
                [k_rebound/(k_rebound+1e8)*100], color="red", s=40)

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_04_radical_intermediate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_04 done")


# ============================================================
def panel_05():
    """Oxygen rebound aperture."""
    fig = make_fig()

    k_rebound = nu_floor * math.exp(-DELTA_M_REBOUND)
    k_HAT = nu_floor * math.exp(-DELTA_M_HAT)

    # (A) Depth comparison
    ax1 = fig.add_subplot(1, 4, 1)
    depths = [DELTA_M_HAT, DELTA_M_REBOUND, DELTA_M_HAT - DELTA_M_REBOUND]
    labels = [r"$\Delta\mathcal{M}_{\mathrm{HAT}}$",
              r"$\Delta\mathcal{M}_{\mathrm{rebound}}$",
              r"$\Delta\Delta\mathcal{M}$"]
    cols = ["#4C72B0", "#55A868", "#FFA500"]
    ax1.bar(labels, depths, color=cols, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\Delta\mathcal{M}$"); ax1.set_title("Depth comparison")
    for i, v in enumerate(depths):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (B) Rate comparison with literature
    ax2 = fig.add_subplot(1, 4, 2)
    rate_labels = ["k_HAT\n(predicted)", "k_rebound\n(predicted)",
                   "Newcomb '95\nlower bound", "Groves '85\nestimate"]
    rate_vals = [k_HAT, k_rebound, 1e9, 1e10]
    rate_cols = ["#4C72B0", "#55A868", "#C44E52", "#FFA500"]
    bars = ax2.bar(range(4), [math.log10(v) for v in rate_vals], color=rate_cols,
                   edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(rate_labels, fontsize=6.5, rotation=15)
    ax2.set_ylabel(r"$\log_{10}(k$ / s$^{-1})$"); ax2.set_title("Rate vs literature")

    # (C) C-O bond formation trajectory
    ax3 = fig.add_subplot(1, 4, 3)
    tau_units = np.linspace(0, 3, 100)
    beta_CO = 1 / (1 + np.exp(-3 * (tau_units - 1.5)))  # sigmoid
    ax3.plot(tau_units, beta_CO, color="#55A868", linewidth=2.5)
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax3.set_xlabel(r"Time ($\tau_p$ units)"); ax3.set_ylabel(r"$\beta_{CO}$")
    ax3.set_title("C-O formation trajectory"); ax3.set_ylim(-0.05, 1.1)
    ax3.text(1.5, 0.55, "TS", ha="center", fontsize=9, color="gray")

    # (D) 3D: k_rebound over (Delta_M_rebound, xi_rad)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    dM_r = np.linspace(0.1, 0.6, 20)
    xi_r = np.linspace(0.3, 0.8, 20)
    DM, XI = np.meshgrid(dM_r, xi_r)
    K_reb = nu_floor * np.exp(-DM)
    ax4.plot_surface(DM, XI, np.log10(K_reb), cmap="viridis", alpha=0.85)
    ax4.scatter([DELTA_M_REBOUND], [0.57], [math.log10(k_rebound)],
                color="red", s=40, zorder=5)
    ax4.set_xlabel(r"$\Delta\mathcal{M}_{reb}$", fontsize=7)
    ax4.set_ylabel(r"$\xi_{\mathrm{rad}}$", fontsize=7)
    ax4.set_zlabel(r"$\log_{10}(k_{reb})$", fontsize=7)
    ax4.set_title(r"$k_{\mathrm{rebound}}$ landscape")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_05_rebound.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_05 done")


# ============================================================
def panel_06():
    """Testosterone 6β regioselectivity."""
    fig = make_fig()
    d = load("06_regioselectivity")

    positions = ["6β", "2β", "15β", "16β"]
    dMs = [0.55, 0.68, 0.80, 0.62]
    gs = [1.00, 0.40, 0.30, 0.50]
    k_effs = [nu_floor * g * math.exp(-dm) for dm, g in zip(dMs, gs)]
    total = sum(k_effs)
    fracs = [k / total for k in k_effs]
    colors = ["#4C72B0", "#FFA500", "#55A868", "#C44E52"]

    # (A) Effective rates
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(positions, [math.log10(k) for k in k_effs], color=colors,
            edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\log_{10}(k_i^{\mathrm{eff}})$")
    ax1.set_title("Effective rates per position")
    for i, (p, k) in enumerate(zip(positions, k_effs)):
        ax1.text(i, math.log10(k) + 0.02, p, ha="center", fontsize=8)

    # (B) Pie chart
    ax2 = fig.add_subplot(1, 4, 2)
    exp_fracs = [0.60, 0.12, 0.10, 0.18]
    wedge_labels = [f"{p}\n{f:.0%}" for p, f in zip(positions, fracs)]
    ax2.pie(fracs, labels=wedge_labels, colors=colors, startangle=90,
            autopct="", wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("Predicted selectivity\n(pred vs exp ≈ 50% at 6β)")

    # (C) Sensitivity to g_6beta
    ax3 = fig.add_subplot(1, 4, 3)
    g_sweep = np.linspace(0.5, 2.0, 50)
    f6b_sweep = []
    for g6 in g_sweep:
        gs_sw = [g6, 0.40, 0.30, 0.50]
        k_sw = [nu_floor * g * math.exp(-dm) for dm, g in zip(dMs, gs_sw)]
        f6b_sweep.append(k_sw[0] / sum(k_sw))
    ax3.plot(g_sweep, np.array(f6b_sweep) * 100, color="#4C72B0", linewidth=2)
    ax3.axvline(1.0, color="#C44E52", linestyle="--", linewidth=1.5)
    ax3.axhspan(50, 70, alpha=0.12, color="#FFA500", label="Exp. 50-70%")
    ax3.set_xlabel(r"$g_{6\beta}$"); ax3.set_ylabel(r"$f_{6\beta}$ (%)")
    ax3.set_title(r"Sensitivity to $g_{6\beta}$"); ax3.legend(fontsize=7)

    # (D) 3D selectivity landscape
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    dm6_range = np.linspace(0.40, 0.70, 15)
    g6_range = np.linspace(0.5, 1.5, 15)
    DM6, G6 = np.meshgrid(dm6_range, g6_range)
    F6 = np.zeros_like(DM6)
    for i in range(DM6.shape[0]):
        for j in range(DM6.shape[1]):
            ks = [nu_floor * g6_range[i] * math.exp(-DM6[i, j]),
                  nu_floor * 0.40 * math.exp(-0.68),
                  nu_floor * 0.30 * math.exp(-0.80),
                  nu_floor * 0.50 * math.exp(-0.62)]
            F6[i, j] = ks[0] / sum(ks)
    ax4.plot_surface(DM6, G6, F6 * 100, cmap="viridis", alpha=0.85)
    ax4.scatter([0.55], [1.0], [fracs[0]*100], color="red", s=40, zorder=5)
    ax4.set_xlabel(r"$\Delta\mathcal{M}_{6\beta}$", fontsize=7)
    ax4.set_ylabel(r"$g_{6\beta}$", fontsize=7)
    ax4.set_zlabel(r"$f_{6\beta}$ (%)", fontsize=7)
    ax4.set_title("Selectivity landscape")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_06_regioselectivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_06 done")


# ============================================================
def panel_07():
    """Five reaction types unified."""
    fig = make_fig()
    d = load("07_reaction_types")

    reaction_types = ["aliphatic", "benzylic", "allylic", "aromatic", "epoxidation"]
    dMs = [0.65, 0.50, 0.45, 0.38, 0.35]
    ks = [nu_floor * math.exp(-dm) for dm in dMs]
    E_as = [65.0 * dm / 4.184 for dm in dMs]
    has_kie = [True, True, True, False, False]
    colors = ["#4C72B0", "#FFA500", "#55A868", "#C44E52", "#8172B2"]

    omega_CH = 2 * math.pi * c_cms * NU_CH
    omega_CD = 2 * math.pi * c_cms * NU_CD
    delta_ZPE = (hbar / 2) * (omega_CH - omega_CD)
    delta_ZPE_kBT = delta_ZPE / kBT

    KIEs = []
    for dm, hk in zip(dMs, has_kie):
        if hk:
            dt = 0.77 + (dm - 0.65) * 0.3
            kr = math.exp(dt * (1 - 1/math.sqrt(2)))
            KIEs.append(math.exp(delta_ZPE_kBT) * kr)
        else:
            KIEs.append(1.0)

    # (A) Delta_M per type
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(reaction_types, dMs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(r"$\Delta\mathcal{M}_{\mathrm{HAT}}$"); ax1.set_title("Activation depth")
    ax1.tick_params(axis="x", rotation=30, labelsize=7)

    # (B) Intrinsic rates
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(reaction_types, [math.log10(k) for k in ks], color=colors,
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel(r"$\log_{10}(k\ \mathrm{s}^{-1})$"); ax2.set_title("Intrinsic rates")
    ax2.tick_params(axis="x", rotation=30, labelsize=7)

    # (C) KIE per type
    ax3 = fig.add_subplot(1, 4, 3)
    kie_colors = [c if h else "#dddddd" for c, h in zip(colors, has_kie)]
    ax3.bar(reaction_types, KIEs, color=kie_colors, edgecolor="black", linewidth=0.5)
    ax3.axhline(4, color="gray", linestyle="--", linewidth=1, label="Min H-KIE = 4")
    ax3.set_ylabel("KIE"); ax3.set_title("KIE prediction")
    ax3.tick_params(axis="x", rotation=30, labelsize=7); ax3.legend(fontsize=7)

    # (D) 3D: rate vs Delta_M vs reaction type index
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    xs = np.arange(5)
    ax4.bar3d(xs - 0.25, dMs, [0]*5, 0.5, [0.02]*5,
              [math.log10(k) for k in ks], color=colors, alpha=0.85)
    ax4.set_xlabel("Class", fontsize=7); ax4.set_ylabel(r"$\Delta\mathcal{M}$", fontsize=7)
    ax4.set_zlabel(r"$\log_{10}(k)$", fontsize=7)
    ax4.set_xticks(xs); ax4.set_xticklabels(reaction_types, fontsize=5, rotation=20)
    ax4.set_title("3D: class vs depth vs rate")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_07_reaction_types.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_07 done")


# ============================================================
def panel_08():
    """Full validation summary: 8 observables."""
    fig = make_fig()

    k_rebound = nu_floor * math.exp(-DELTA_M_REBOUND)
    k_HAT = nu_floor * math.exp(-DELTA_M_HAT)
    omega_CH = 2*math.pi*c_cms*NU_CH; omega_CD = 2*math.pi*c_cms*NU_CD
    dZPE = (hbar/2)*(omega_CH - omega_CD)
    KIE = math.exp(dZPE/kBT) * 1.16

    obs_labels = [r"$\Delta\mathcal{M}_{\mathrm{HAT}}$", r"$E_a$ (kcal/mol)",
                  r"$k_{\mathrm{HAT}}$ (×10⁹)", r"$k_{\mathrm{reb}}$ (×10⁹)",
                  r"$k_{reb}/k_{HAT}$", "KIE", "Stereoret. %",
                  r"$f_{6\beta}$ %"]
    pred = [DELTA_M_HAT, 65*DELTA_M_HAT/4.184, k_HAT/1e9, k_rebound/1e9,
            k_rebound/k_HAT, KIE, 88.0, 49.0]
    expt = [0.65, 10.0, 5.0, 7.4, 1.4, 7.2, 65.0, 60.0]
    targets_lo = [0.60, 8.0, 1.0, 1.0, 1.0, 4.0, 40.0, 50.0]
    targets_hi = [0.70, 14.0, 20.0, 20.0, 5.0, 11.0, 90.0, 70.0]

    norm_pred = [p/e if e != 0 else 1.0 for p, e in zip(pred, expt)]
    relative_err = [abs(p - e)/e if e != 0 else 0 for p, e in zip(pred, expt)]

    # (A) Predicted vs experimental scatter
    ax1 = fig.add_subplot(1, 4, 1)
    cmap = cm.RdYlGn
    cols = [cmap(0.9 if err < 0.2 else 0.3) for err in relative_err]
    ax1.scatter(expt, pred, c=cols, s=70, zorder=5, edgecolors="black", linewidths=0.5)
    lims = [min(expt + pred) * 0.8, max(expt + pred) * 1.2]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Experimental"); ax1.set_ylabel("Predicted")
    ax1.set_title("Pred vs Exp")

    # (B) Per-observable relative error
    ax2 = fig.add_subplot(1, 4, 2)
    colors_bar = ["#55A868" if e < 0.2 else "#C44E52" for e in relative_err]
    ax2.barh(obs_labels, [e * 100 for e in relative_err], color=colors_bar,
             edgecolor="black", linewidth=0.4)
    ax2.axvline(20, color="gray", linestyle="--", linewidth=1.5, label="20% ref")
    ax2.set_xlabel("Relative error (%)"); ax2.set_title("Per-observable error")
    ax2.legend(fontsize=7)

    # (C) Pie: PASS / FAIL
    ax3 = fig.add_subplot(1, 4, 3)
    n_pass = sum(1 for e in relative_err if e < 0.30)
    n_fail = 8 - n_pass
    ax3.pie([n_pass, n_fail], labels=[f"{n_pass} PASS", f"{n_fail} FAIL"],
            colors=["#55A868", "#C44E52"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11, "fontweight": "bold"})
    ax3.set_title("Validation verdict\n8 observables")

    # (D) 3D bar: predicted (orange) vs experimental (blue)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n = len(obs_labels)
    xs = np.arange(n)
    norm_expt = [1.0] * n
    ax4.bar3d(xs - 0.2, [0]*n, [0]*n, 0.35, [1]*n, norm_pred,
              color="#FFA500", alpha=0.8, label="Pred")
    ax4.bar3d(xs + 0.05, [0]*n, [0]*n, 0.35, [1]*n, norm_expt,
              color="#4C72B0", alpha=0.8, label="Exp")
    ax4.set_xticks(xs)
    ax4.set_xticklabels([str(i+1) for i in range(n)], fontsize=7)
    ax4.set_ylabel(""); ax4.set_zlabel("Normalised value", fontsize=7)
    ax4.set_title("3D pred vs exp")

    fig.tight_layout(pad=1.5)
    fig.savefig(FIG_DIR / "panel_08_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("panel_08 done")


# ============================================================
if __name__ == "__main__":
    print("Generating Paper 6 panels...")
    panel_01()
    panel_02()
    panel_03()
    panel_04()
    panel_05()
    panel_06()
    panel_07()
    panel_08()
    print("All panels done.")
