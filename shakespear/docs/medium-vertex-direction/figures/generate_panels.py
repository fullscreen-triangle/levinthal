"""
Panels for "The Medium Vertex and the Direction of a Protein Process".

Five panels, four charts each, at least one 3-D per panel. Every number
plotted is computed by the kernel at draw time --- nothing is transcribed
from the paper or from the results JSON, so a panel cannot drift out of
agreement with the theory it illustrates.

White background, minimal text, no conceptual diagrams, no tables.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "validation"
sys.path.insert(0, str(VAL / "kernel"))
sys.path.insert(0, str(VAL / "experiments"))

from medium import (  # noqa: E402
    BETA_DEFAULT,
    BULK,
    SOL,
    STRUCTURAL,
    Chain,
    ContactGraph,
    Leaf,
    Medium,
    direction,
    medium_bias,
    robustness_family,
    weight_log,
)
from exp02_direction import ALL_IDENTITIES, chain_for  # noqa: E402

B = BETA_DEFAULT
OUT = HERE

# ---- house style ----------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 160,
})

C_STRUCT = "#1d4ed8"   # structural / forward
C_BULK = "#d97706"     # bulk / reverse
C_REFUSE = "#b91c1c"   # refusal
C_NEUTRAL = "#6b7280"
C_FLOOR = "#111827"
FAMILY_COLORS = {
    "log (eq. 1)": "#1d4ed8",
    "sqrt": "#0d9488",
    "rational": "#b45309",
    "linear-cap": "#7c3aed",
}


def _finish(fig, name: str) -> None:
    fig.tight_layout(pad=1.1)
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


def _panel(n_3d_index: int):
    """A 1x4 panel; the axis at n_3d_index is 3-D."""
    fig = plt.figure(figsize=(13.2, 3.1))
    axes = []
    for i in range(4):
        if i == n_3d_index:
            axes.append(fig.add_subplot(1, 4, i + 1, projection="3d"))
        else:
            axes.append(fig.add_subplot(1, 4, i + 1))
    return fig, axes


# =====================================================================
#  Panel 1 --- the medium weight and the two boundaries
# =====================================================================
def panel_01() -> None:
    fig, ax = _panel(2)

    # (a) w(l,m) vs occupancy, all four weight families
    mu = np.logspace(-8, 3, 400)
    for fname, fn in robustness_family().items():
        w = [fn(m, 1e-3, B) / B for m in mu]
        ax[0].semilogx(mu, w, color=FAMILY_COLORS[fname], label=fname)
    ax[0].axhline(1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[0].set_xlabel(r"ambient occupancy $\mu$")
    ax[0].set_ylabel(r"$w(\ell,\mathsf{m})\,/\,\beta$")
    ax[0].set_ylim(0.8, 6)
    ax[0].legend(frameon=False, loc="upper right")

    # (b) the two competing boundaries vs occupancy
    mu2 = np.logspace(-8, 3, 300)
    wm = np.array([weight_log(m, 1e-3, B) / B for m in mu2])
    for k, lab in [(4.0, None), (2.0, None), (1.05, None)]:
        ax[1].semilogx(mu2, np.full_like(mu2, k), color=C_STRUCT, lw=1.2,
                       alpha=0.35 + 0.2 * (k / 4))
    ax[1].semilogx(mu2, wm, color=C_BULK, lw=2.0)
    ax[1].fill_between(mu2, wm, 6, color=C_STRUCT, alpha=0.07)
    ax[1].fill_between(mu2, 0.8, wm, color=C_BULK, alpha=0.07)
    ax[1].set_xlabel(r"ambient occupancy $\mu$")
    ax[1].set_ylabel(r"boundary $/\,\beta$")
    ax[1].set_ylim(0.8, 6)
    ax[1].legend(handles=[
        Line2D([], [], color=C_STRUCT, lw=1.5),
        Line2D([], [], color=C_BULK, lw=2.0),
    ], labels=[r"$\rho_{\rm str}$", r"$w(\ell,\mathsf{m})$"],
        frameon=False, loc="upper right")

    # (c) 3-D role surface over (occupancy, system contact)
    m_ax = np.logspace(-8, 2, 70)
    c_ax = np.linspace(1.0, 5.0, 70)
    MU, CC = np.meshgrid(m_ax, c_ax)
    W = np.vectorize(lambda m: weight_log(m, 1e-3, B) / B)(MU)
    Z = CC - W  # >0 structural, <0 bulk
    ax[2].plot_surface(np.log10(MU), CC, Z, cmap="coolwarm_r",
                       linewidth=0, antialiased=True, alpha=0.93,
                       rstride=1, cstride=1)
    ax[2].contour(np.log10(MU), CC, Z, levels=[0], colors=[C_FLOOR],
                  linewidths=2.0, offset=Z.min())
    ax[2].set_xlabel(r"$\log_{10}\mu$", labelpad=-2)
    ax[2].set_ylabel(r"$\rho_{\rm str}/\beta$", labelpad=-2)
    ax[2].set_zlabel(r"$\rho_{\rm str}-w$", labelpad=-4)
    ax[2].view_init(elev=22, azim=-131)
    ax[2].tick_params(pad=-1)

    # (d) scale invariance: three decades of tau collapse onto one curve,
    # because only the ratio mu/tau is observable (Def. 2.1). x is mu/tau,
    # so mu = tau * ratio.
    ratio = np.logspace(-4, 4, 300)
    for i, tau in enumerate([1e-5, 1e-3, 1e-1]):
        w = [weight_log(tau * r, tau, B) / B for r in ratio]
        ax[3].semilogx(ratio, w, lw=3.4 - 1.0 * i,
                       color=["#1d4ed8", "#60a5fa", "#bfdbfe"][i],
                       alpha=0.95)
    ax[3].axhline(1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[3].set_xlabel(r"$\mu/\tau$")
    ax[3].set_ylabel(r"$w(\ell,\mathsf{m})\,/\,\beta$")
    ax[3].set_ylim(0.8, 6)

    _finish(fig, "panel_01_medium_weight")


# =====================================================================
#  Panel 2 --- solvent role is derivable
# =====================================================================
def panel_02() -> None:
    fig, ax = _panel(3)

    def graph(mu_w: float, contact: float | None) -> ContactGraph:
        g = ContactGraph(Medium(mu={"H2O": mu_w}, tau=1e-3), beta=B)
        g.add_leaf(Leaf("w", SOL, "H2O"))
        g.add_leaf(Leaf("p", "res", "ALA"))
        if contact is not None:
            g.add_contact("w", "p", contact * B)
        return g

    # (a) the two boundaries for axial vs bulk water (real kernel values)
    g = graph(55.5, 4.0)
    axial = g.role_report("w")
    gb = graph(55.5, None)
    bulk = gb.role_report("w")
    x = np.arange(2)
    ax[0].bar(x - 0.19, [axial["structural_residue"] / B,
                         bulk["structural_residue"] / B],
              0.38, color=C_STRUCT, label=r"$\rho_{\rm str}$")
    ax[0].bar(x + 0.19, [axial["medium_weight"] / B,
                         bulk["medium_weight"] / B],
              0.38, color=C_BULK, label=r"$w(\ell,\mathsf{m})$")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["axial", "bulk"])
    ax[0].set_ylabel(r"boundary $/\,\beta$")
    ax[0].legend(frameon=False)

    # (b) role phase boundary in (mu, contact) for all weight families
    mu = np.logspace(-8, 3, 260)
    for fname, fn in robustness_family().items():
        ax[1].semilogx(mu, [fn(m, 1e-3, B) / B for m in mu],
                       color=FAMILY_COLORS[fname], label=fname)
    ax[1].set_xlabel(r"ambient occupancy $\mu$")
    ax[1].set_ylabel(r"critical $\rho_{\rm str}/\beta$")
    ax[1].set_ylim(0.9, 5)
    ax[1].legend(frameon=False, loc="upper right")

    # (c) monotonicity: role flip point across the sweep, per family
    sweep = np.logspace(-8, 3, 160)
    for j, (fname, fn) in enumerate(robustness_family().items()):
        roles = [1 if graph(m, 1.05).role("w", fn) == STRUCTURAL else 0
                 for m in sweep]
        ax[2].semilogx(sweep, np.array(roles) + j * 0.045 - 0.07,
                       color=FAMILY_COLORS[fname], lw=2.0, drawstyle="steps-post")
    ax[2].set_xlabel(r"ambient occupancy $\mu$")
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(["bulk", "struct"])
    ax[2].set_ylim(-0.25, 1.25)

    # (d) 3-D: role region as a function of (mu, contact, tau)
    m_ax = np.logspace(-7, 2, 46)
    c_ax = np.linspace(1.0, 4.0, 46)
    MU, CC = np.meshgrid(m_ax, c_ax)
    for tau, cmapname, off in [(1e-4, "Blues", 0), (1e-2, "Oranges", 0)]:
        W = np.vectorize(lambda m, t=tau: weight_log(m, t, B) / B)(MU)
        ax[3].contour(np.log10(MU), CC, CC - W, levels=[0],
                      colors=[C_STRUCT if tau == 1e-4 else C_BULK],
                      linewidths=2.2, zdir="z", offset=0)
        ax[3].plot_surface(np.log10(MU), CC, CC - W, alpha=0.45,
                           cmap=cmapname, linewidth=0, rstride=2, cstride=2)
    ax[3].set_xlabel(r"$\log_{10}\mu$", labelpad=-2)
    ax[3].set_ylabel(r"$\rho_{\rm str}/\beta$", labelpad=-2)
    ax[3].set_zlabel("margin", labelpad=-4)
    ax[3].view_init(elev=20, azim=-127)
    ax[3].tick_params(pad=-1)

    _finish(fig, "panel_02_solvent_role")


# =====================================================================
#  Panel 3 --- reversal invariance and the medium bias
# =====================================================================
def panel_03() -> None:
    fig, ax = _panel(1)

    c = chain_for("2.6.1.2")
    cr = c.reversed_chain()

    # (a) Lemma 4.1: the per-cut residues reverse, but every accumulated
    # INVARIANT is identical. Plot the invariants (which coincide exactly)
    # rather than the running sums (which differ until the last step and
    # would read as a refutation of the lemma).
    inv_labels = ["total", "max", "min", "mean", "range"]
    fwd_inv = [sum(c.residues) / B, max(c.residues) / B, min(c.residues) / B,
               (sum(c.residues) / len(c.residues)) / B,
               (max(c.residues) - min(c.residues)) / B]
    rev_inv = [sum(cr.residues) / B, max(cr.residues) / B,
               min(cr.residues) / B,
               (sum(cr.residues) / len(cr.residues)) / B,
               (max(cr.residues) - min(cr.residues)) / B]
    xi = np.arange(len(inv_labels))
    ax[0].bar(xi - 0.19, fwd_inv, 0.38, color=C_STRUCT)
    ax[0].bar(xi + 0.19, rev_inv, 0.38, color=C_BULK,
              hatch="///", edgecolor="white", linewidth=0.0)
    ax[0].set_xticks(xi)
    ax[0].set_xticklabels(inv_labels, rotation=30, ha="right")
    ax[0].set_ylabel(r"invariant $/\,\beta$")

    # (b) 3-D: bias surface over (product depletion, reactant depletion)
    pe = np.linspace(-9, 3, 60)
    re_ = np.linspace(-9, 3, 60)
    PE, RE = np.meshgrid(pe, re_)
    Z = np.zeros_like(PE)
    for i in range(PE.shape[0]):
        for j in range(PE.shape[1]):
            mu = {x: 1e-3 for x in ALL_IDENTITIES}
            mu["L-glutamate"] = 10.0 ** PE[i, j]
            mu["2-oxoglutarate"] = 10.0 ** RE[i, j]
            Z[i, j] = medium_bias(c, Medium(mu=mu, tau=1e-3), B) / B
    Z = np.clip(Z, -12, 12)
    ax[1].plot_surface(PE, RE, Z, cmap="RdBu_r", linewidth=0,
                       antialiased=True, rstride=1, cstride=1, alpha=0.95,
                       vmin=-12, vmax=12)
    for lv, col in [(1, C_STRUCT), (-1, C_BULK)]:
        ax[1].contour(PE, RE, Z, levels=[lv], colors=[col], linewidths=1.8,
                      offset=-12)
    ax[1].set_xlabel(r"$\log_{10}\mu_{\rm prod}$", labelpad=-2)
    ax[1].set_ylabel(r"$\log_{10}\mu_{\rm reac}$", labelpad=-2)
    ax[1].set_zlabel(r"$\delta/\beta$", labelpad=-4)
    ax[1].view_init(elev=24, azim=-124)
    ax[1].tick_params(pad=-1)

    # (c) the two-sided sweep: all three cases of Thm 4.4
    exps = list(range(-9, 10))
    deltas, cols = [], []
    for e in exps:
        mu = {x: 1e-3 for x in ALL_IDENTITIES}
        if e < 0:
            mu["L-glutamate"] = 1e-3 * (10.0 ** e)
        elif e > 0:
            mu["2-oxoglutarate"] = 1e-3 * (10.0 ** -e)
        d = direction(c, Medium(mu=mu, tau=1e-3), B)
        deltas.append(d["delta_over_floor"])
        cols.append({"forward": C_STRUCT, "reverse": C_BULK,
                     "undirected": C_NEUTRAL}[d["direction"]])
    ax[2].bar(exps, deltas, color=cols, width=0.78)
    ax[2].axhline(1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[2].axhline(-1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[2].axhspan(-1, 1, color=C_NEUTRAL, alpha=0.12)
    ax[2].set_xlabel("depletion exponent")
    ax[2].set_ylabel(r"$\delta/\beta$")

    # (d) antisymmetry across weight families. Symmetric log scale: the
    # sqrt family reaches ~1e4 while linear-cap saturates near 1, so a
    # linear axis would flatten three of the four curves into the axis.
    exps2 = np.arange(-8, 9)
    for fname, fn in robustness_family().items():
        fwd, rev = [], []
        for e in exps2:
            mu = {x: 1e-3 for x in ALL_IDENTITIES}
            mu["L-glutamate"] = 1e-3 * (10.0 ** float(e))
            med = Medium(mu=mu, tau=1e-3)
            fwd.append(medium_bias(c, med, B, fn) / B)
            rev.append(medium_bias(cr, med, B, fn) / B)
        ax[3].plot(exps2, fwd, color=FAMILY_COLORS[fname], label=fname)
        ax[3].plot(exps2, rev, color=FAMILY_COLORS[fname], ls=":", lw=1.3)
    ax[3].set_yscale("symlog", linthresh=1.0)
    ax[3].axhline(0.0, color=C_FLOOR, lw=0.8)
    ax[3].axhspan(-1, 1, color=C_NEUTRAL, alpha=0.14)
    ax[3].set_xlabel("depletion exponent")
    ax[3].set_ylabel(r"$\delta/\beta$")
    ax[3].legend(frameon=False, loc="upper right", ncol=2)

    _finish(fig, "panel_03_direction")


# =====================================================================
#  Panel 4 --- the refusals
# =====================================================================
def panel_04() -> None:
    fig, ax = _panel(2)

    # (a) refusal fraction over the (mu, contact) plane
    mu = np.logspace(-8, 3, 90)
    contacts = np.linspace(0.0, 4.0, 90)
    frac = []
    for cc in contacts:
        row = 0
        for m in mu:
            g = ContactGraph(Medium(mu={"H2O": m}, tau=1e-3), beta=B)
            g.add_leaf(Leaf("w", SOL, "H2O"))
            g.add_leaf(Leaf("p", "res", "ALA"))
            if cc > 0:
                g.add_contact("w", "p", max(cc, 1.0) * B)
            if g.role("w") == BULK:
                row += 1
        frac.append(row / len(mu))
    ax[0].plot(contacts, frac, color=C_REFUSE, lw=2.0)
    ax[0].fill_between(contacts, 0, frac, color=C_REFUSE, alpha=0.12)
    ax[0].set_xlabel(r"system contact $/\,\beta$")
    ax[0].set_ylabel("fraction refused")
    ax[0].set_ylim(-0.03, 1.03)

    # (b) orientation refusal band vs depletion
    exps = np.linspace(-6, 6, 300)
    dd = []
    for e in exps:
        mu = {x: 1e-3 for x in ALL_IDENTITIES}
        if e < 0:
            mu["L-glutamate"] = 1e-3 * (10.0 ** e)
        else:
            mu["2-oxoglutarate"] = 1e-3 * (10.0 ** -e)
        dd.append(medium_bias(chain_for("2.6.1.2"),
                              Medium(mu=mu, tau=1e-3), B) / B)
    dd = np.array(dd)
    ax[1].plot(exps, dd, color=C_STRUCT, lw=2.0)
    ax[1].axhspan(-1, 1, color=C_REFUSE, alpha=0.16)
    ax[1].axhline(1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[1].axhline(-1.0, color=C_FLOOR, ls="--", lw=1.0)
    ax[1].set_xlabel("depletion exponent")
    ax[1].set_ylabel(r"$\delta/\beta$")

    # (c) 3-D: the refusal volume --- |delta| <= beta over two depletions
    pe = np.linspace(-6, 2, 55)
    re_ = np.linspace(-6, 2, 55)
    PE, RE = np.meshgrid(pe, re_)
    Z = np.zeros_like(PE)
    for i in range(PE.shape[0]):
        for j in range(PE.shape[1]):
            mu = {x: 1e-3 for x in ALL_IDENTITIES}
            mu["L-glutamate"] = 10.0 ** PE[i, j]
            mu["2-oxoglutarate"] = 10.0 ** RE[i, j]
            Z[i, j] = abs(medium_bias(chain_for("2.6.1.2"),
                                      Medium(mu=mu, tau=1e-3), B) / B)
    Z = np.clip(Z, 0, 10)
    ax[2].plot_surface(PE, RE, Z, cmap="RdYlBu", linewidth=0,
                       rstride=1, cstride=1, alpha=0.92)
    ax[2].contourf(PE, RE, (Z <= 1.0).astype(float), levels=[0.5, 1.5],
                   colors=[C_REFUSE], alpha=0.5, offset=0)
    ax[2].set_xlabel(r"$\log_{10}\mu_{\rm prod}$", labelpad=-2)
    ax[2].set_ylabel(r"$\log_{10}\mu_{\rm reac}$", labelpad=-2)
    ax[2].set_zlabel(r"$|\delta|/\beta$", labelpad=-4)
    ax[2].view_init(elev=26, azim=-120)
    ax[2].tick_params(pad=-1)

    # (d) the log-2 ceiling: flooding one product cannot reverse
    sat = np.logspace(-3, 30, 300)
    vals = []
    for s in sat:
        mu = {x: 1e-3 for x in ALL_IDENTITIES}
        mu["L-glutamate"] = s
        vals.append(medium_bias(chain_for("2.6.1.2"),
                                Medium(mu=mu, tau=1e-3), B) / B)
    ax[3].semilogx(sat, vals, color=C_STRUCT, lw=2.0)
    ax[3].axhline(-math.log(2), color=C_REFUSE, ls="--", lw=1.4)
    ax[3].axhline(-1.0, color=C_FLOOR, ls=":", lw=1.2)
    ax[3].axhspan(-1, 1, color=C_REFUSE, alpha=0.12)
    ax[3].set_xlabel(r"product occupancy $\mu$")
    ax[3].set_ylabel(r"$\delta/\beta$")
    ax[3].set_ylim(-1.4, 3)

    _finish(fig, "panel_04_refusal")


# =====================================================================
#  Panel 5 --- the representational partition
# =====================================================================
def panel_05() -> None:
    from exp03_partition import QUESTIONS

    fig, ax = _panel(1)

    sig = np.array([q["signature"] for q in QUESTIONS], float)
    chn = np.array([q["chain"] for q in QUESTIONS], float)
    grp = np.array([q["group"] for q in QUESTIONS])
    idx = np.arange(len(QUESTIONS))

    # (a) per-question capability, the two views back to back
    ax[0].barh(idx, -sig, color=C_STRUCT, height=0.72)
    ax[0].barh(idx, chn, color=C_BULK, height=0.72)
    ax[0].axvline(0, color=C_FLOOR, lw=0.9)
    for g, y in [(1, 1.5), (2, 7.0), (3, 11.0)]:
        ax[0].axhline(y + 0.5, color=C_NEUTRAL, lw=0.7, ls=":")
    ax[0].set_yticks([])
    ax[0].set_xticks([-2, -1, 0, 1, 2])
    ax[0].set_xticklabels(["2", "1", "0", "1", "2"])
    ax[0].set_xlabel("signature   |   chain")
    ax[0].invert_yaxis()

    # (b) 3-D: capability cube by group
    for g, col in [(1, C_STRUCT), (2, C_BULK), (3, "#0d9488")]:
        m = grp == g
        ax[1].scatter(sig[m] + np.random.default_rng(g).normal(0, .045, m.sum()),
                      chn[m] + np.random.default_rng(g + 9).normal(0, .045, m.sum()),
                      np.full(m.sum(), g), s=64, color=col,
                      depthshade=False, edgecolor="white", linewidth=0.6)
    ax[1].set_xlabel("signature", labelpad=-3)
    ax[1].set_ylabel("chain", labelpad=-3)
    ax[1].set_zlabel("group", labelpad=-4)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_zticks([1, 2, 3])
    ax[1].view_init(elev=17, azim=-128)
    ax[1].tick_params(pad=-1)

    # (c) group-level capability totals
    gs = [1, 2, 3]
    tot_s = [sig[grp == g].sum() for g in gs]
    tot_c = [chn[grp == g].sum() for g in gs]
    x = np.arange(3)
    ax[2].bar(x - 0.19, tot_s, 0.38, color=C_STRUCT)
    ax[2].bar(x + 0.19, tot_c, 0.38, color=C_BULK)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["1", "2", "3"])
    ax[2].set_xlabel("question group")
    ax[2].set_ylabel("capability")

    # (d) overlap: how many questions each view answers, and jointly
    only_s = int(((sig == 2) & (chn < 2)).sum())
    only_c = int(((chn == 2) & (sig < 2)).sum())
    both = int(((sig == 2) & (chn == 2)).sum())
    none = int(((sig == 0) & (chn == 0)).sum())
    partial = len(QUESTIONS) - only_s - only_c - both - none
    ax[3].bar([0, 1, 2, 3, 4], [only_s, only_c, both, partial, none],
              color=[C_STRUCT, C_BULK, "#7c3aed", C_NEUTRAL, C_REFUSE],
              width=0.68)
    ax[3].set_xticks([0, 1, 2, 3, 4])
    ax[3].set_xticklabels(["sig", "chain", "both", "part", "none"])
    ax[3].set_ylabel("questions")

    _finish(fig, "panel_05_partition")


if __name__ == "__main__":
    print("generating panels...")
    panel_01()
    panel_02()
    panel_03()
    panel_04()
    panel_05()
    print("done.")
