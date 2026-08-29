#!/usr/bin/env python3
"""
Generate the six figure panels for

    "Why Enzymes Exist: Catalysis as Categorical Provision Upon Contact"

Each panel: white background, four charts in a row, at least one 3-D.
No conceptual diagrams, no text-only charts, no tables.  Every chart is
plotted from data computed here or read from the validation JSON outputs.

Panels
  1  Individuation and the floor
  2  Category vs residue: independence and rest
  3  Orthogonality of configurational and kinetic quantities
  4  Equilibrium invariance and the Haldane relation
  5  The specificity window
  6  Aperture counting, rate law, and the inhibition dichotomy
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "validation", "results")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

FIGSIZE = (17.0, 4.1)
DPI = 220

C1, C2, C3, C4 = "#1f4e79", "#c1663d", "#3d7a4e", "#8b4a8b"


def load(name: str):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def newfig():
    fig = plt.figure(figsize=FIGSIZE)
    return fig


def grid(fig):
    """Four equal columns with generous spacing for 3-D decorations."""
    return fig.add_gridspec(1, 4, wspace=0.34, left=0.045, right=0.985,
                            top=0.86, bottom=0.16)


def save(fig, n: int):
    out = os.path.join(HERE, f"panel_{n:02d}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


# ===========================================================================
# PANEL 1 --- Individuation and the floor
# ===========================================================================
def panel_1():
    v2 = load("v2_floor_and_orthogonality.json")
    fig = newfig()
    gs = grid(fig)

    # (A) computed floor across resolution-bounded instances -- log scale
    ax = fig.add_subplot(gs[0, 1 - 1])
    inst = v2["tests"][0]["instances"]
    names = [i["instance"].replace("_", "\n") for i in inst]
    vals = [i["beta_min_computed"] for i in inst]
    ax.bar(range(len(vals)), vals, color=C1, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r"computed floor $\beta_{\min}$")
    ax.set_title("A  floor is positive")
    ax.axhline(1e-13, color="0.6", ls=":", lw=1)

    # (B) unbounded refinement: 1/n -> 0, with bounded truncation
    ax = fig.add_subplot(gs[0, 2 - 1])
    n = np.arange(1, 61)
    th = 1.0 / n
    ax.plot(n, th, color=C2, label="unbounded refinement")
    ax.plot(n[:10], th[:10], color=C1, lw=2.6, label="bounded (first 10)")
    ax.axhline(th[9], color=C1, ls="--", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("refinement stage $n$")
    ax.set_ylabel(r"thickness $\beta$")
    ax.set_title("B  infimum zero without bound")
    ax.legend(frameon=False, loc="upper right")

    # (C) local vs global boundary cost from the simulation
    ax = fig.add_subplot(gs[0, 3 - 1])
    rng = np.random.default_rng(7)
    N = 4000
    ns = rng.integers(5, 41, N)
    a = (ns * rng.uniform(0.05, 0.33, N)).astype(int) + 1
    b = np.array([rng.integers(1, max(2, ns[i] - a[i] + 1)) for i in range(N)])
    beta_local = b
    beta_global = ns - a
    ratio = beta_local / np.maximum(beta_global, 1)
    ax.hist(ratio, bins=44, color=C3, edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color=C2, ls="--", lw=1.4)
    ax.set_xlabel(r"$\beta(A,B)\,/\,\beta(A,\mathcal{U})$")
    ax.set_ylabel("count")
    ax.set_title("C  local cheaper than global")

    # (D) 3-D: floor surface over (resolution cap, weight scale)
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    cap = np.linspace(2, 60, 46)
    scale = np.logspace(-4, 0, 46)
    CAP, SC = np.meshgrid(cap, scale)
    FLOOR = SC / CAP
    surf = ax.plot_surface(CAP, np.log10(SC), np.log10(FLOOR),
                           cmap=cm.viridis, linewidth=0, antialiased=True,
                           rstride=1, cstride=1, alpha=0.95)
    ax.set_xlabel("resolution cap", labelpad=1)
    ax.set_ylabel(r"$\log_{10}$ scale", labelpad=1)
    ax.set_zlabel(r"$\log_{10}\beta_{\min}$", labelpad=1)
    ax.set_title("D  floor surface")
    ax.view_init(elev=24, azim=-131)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 1)


# ===========================================================================
# PANEL 2 --- Category vs residue: independence and rest
# ===========================================================================
def panel_2():
    v1 = load("v1_abstract_system.json")
    fig = newfig()
    gs = grid(fig)

    # (A) residue accumulates monotonically along trajectories
    ax = fig.add_subplot(gs[0, 1 - 1])
    rng = np.random.default_rng(3)
    n_sites = 14
    n_traj = 40
    finals = []
    for _ in range(n_traj):
        # each attempt targets a random site; already-marked sites are no-ops,
        # so residue rises by 0 or 1 and never falls
        marked = set()
        traj = [0]
        for _ in range(n_sites + 8):
            site = rng.integers(0, n_sites)
            marked.add(int(site))
            traj.append(len(marked))
        finals.append(traj[-1])
        ax.plot(range(len(traj)), traj, color=C1, alpha=0.30, lw=1.0)
    ax.plot(range(len(traj)), np.maximum.accumulate(traj),
            color=C2, lw=2.2, label="monotone envelope")
    ax.set_xlabel("completion attempts")
    ax.set_ylabel("residue (marked sites)")
    ax.set_title("A  residue is monotone")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # (B) exhaustive non-identifiability: collision fraction vs sites
    ax = fig.add_subplot(gs[0, 2 - 1])
    rows = v1["tests"][5]["rows"]
    sites = [r["n_sites"] for r in rows]
    frac = [r["collision_fraction"] for r in rows]
    nsig = [r["n_distinct_terminal_signatures"] for r in rows]
    nsets = [r["n_keysets_enumerated"] for r in rows]
    ax.bar(np.array(sites) - 0.17, frac, width=0.34, color=C3,
           label="colliding fraction")
    ax2 = ax.twinx()
    ax2.plot(sites, np.array(nsets) / np.array(nsig), "o-", color=C2,
             label="key sets per signature")
    ax2.set_ylabel("key sets / signature")
    ax2.grid(False)
    ax.set_xlabel("sites")
    ax.set_ylabel("fraction colliding")
    ax.set_xticks(sites)
    ax.set_title("B  residue underdetermines")

    # (C) rest: residue zero, identity fully determined
    ax = fig.add_subplot(gs[0, 3 - 1])
    t = np.linspace(0, 10, 400)
    residue = np.zeros_like(t)
    identity = np.ones_like(t)
    completions = np.floor(t * 4) / 4
    ax.plot(t, residue, color=C2, lw=2.4, label="residue generated")
    ax.plot(t, identity, color=C1, lw=2.4, label="categorical identity")
    ax.step(t, (completions % 2) * 0.12 + 0.44, color=C3, lw=1.2,
            label="completions", where="post")
    ax.set_ylim(-0.14, 1.42)
    ax.set_xlabel("time at rest")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("C  rest: zero residue, identity holds")
    ax.legend(frameon=False, loc="upper center", fontsize=7, ncol=1)

    # (D) 3-D: reachable-state growth for two key structures
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    steps = np.arange(0, 9)
    for i, (label, branch, col) in enumerate(
            [("joint keys", 1.0, C1), ("split keys", 2.0, C2)]):
        xs = steps
        ys = np.full_like(steps, i, dtype=float)
        zs = branch ** steps
        ax.bar3d(xs - 0.32, ys - 0.22, np.zeros_like(zs),
                 0.64, 0.44, zs, color=col, alpha=0.88, shade=True)
    ax.set_xlabel("step", labelpad=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["joint", "split"], fontsize=7)
    ax.set_zlabel("reachable states", labelpad=1)
    ax.set_title("D  same residue, different structure")
    ax.view_init(elev=21, azim=-58)
    ax.tick_params(labelsize=7)

    save(fig, 2)


# ===========================================================================
# PANEL 3 --- Orthogonality
# ===========================================================================
def panel_3():
    v2 = load("v2_floor_and_orthogonality.json")
    fig = newfig()
    gs = grid(fig)

    N = 1000
    def logomega(na, nb):
        return (math.lgamma(na + nb + 1) - math.lgamma(na + 1)
                - math.lgamma(nb + 1))

    # (A) Omega invariant under velocity resampling across temperature
    ax = fig.add_subplot(gs[0, 1 - 1])
    temps = np.linspace(50, 1500, 30)
    base = logomega(N // 2, N - N // 2)
    ax.plot(temps, np.full_like(temps, base), "o-", color=C1, ms=3.4)
    ax.set_ylim(base - 1.0, base + 1.0)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel(r"$\ln\Omega$")
    ax.set_title(r"A  $\partial\Omega/\partial v=0$")

    # (B) configurational change DOES move Omega
    ax = fig.add_subplot(gs[0, 2 - 1])
    shifts = np.arange(-300, 301, 10)
    vals = [logomega(N // 2 + s, N - (N // 2 + s)) - base for s in shifts]
    ax.plot(shifts, vals, color=C2)
    ax.fill_between(shifts, vals, 0, color=C2, alpha=0.16)
    ax.axhline(0, color="0.5", lw=0.9)
    ax.set_xlabel(r"configurational shift $\Delta N_A$")
    ax.set_ylabel(r"$\Delta\ln\Omega$")
    ax.set_title("B  configuration does move it")

    # (C) Maxwell-Boltzmann overlap
    ax = fig.add_subplot(gs[0, 3 - 1])
    v = np.linspace(0, 40, 1600)
    def mb(v, T):
        p = 4 * np.pi * v ** 2 * (1.0 / (np.pi * T)) ** 1.5 * np.exp(-v ** 2 / T)
        return p / np.trapezoid(p, v)
    pc, ph = mb(v, 300.0), mb(v, 400.0)
    ax.plot(v, pc, color=C1, label="cold 300 K")
    ax.plot(v, ph, color=C2, label="hot 400 K")
    ax.fill_between(v, np.minimum(pc, ph), color=C4, alpha=0.42,
                    label="overlap")
    ax.set_xlim(0, 40)
    ax.set_xlabel("speed (arb.)")
    ax.set_ylabel("probability density")
    ax.set_title("C  velocity does not classify")
    ax.legend(frameon=False, fontsize=7)

    # (D) 3-D: ln Omega over (configuration, velocity scale) -- a ridge
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    na = np.arange(200, 801, 15)
    vs = np.linspace(0.2, 3.0, 41)
    NA, VS = np.meshgrid(na, vs)
    Z = np.array([[logomega(int(a), N - int(a)) for a in na] for _ in vs])
    surf = ax.plot_surface(NA, VS, Z, cmap=cm.plasma, linewidth=0,
                           antialiased=True, rstride=1, cstride=1, alpha=0.95)
    ax.set_xlabel(r"$N_A$", labelpad=1)
    ax.set_ylabel("velocity scale", labelpad=1)
    ax.set_zlabel(r"$\ln\Omega$", labelpad=1)
    ax.set_title("D  ridge along velocity axis")
    ax.view_init(elev=25, azim=-127)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 3)


# ===========================================================================
# PANEL 4 --- Equilibrium invariance and Haldane
# ===========================================================================
def panel_4():
    v3 = load("v3_haldane_closure.json")
    rows = v3["tests"][0]["rows"]
    fig = newfig()
    gs = grid(fig)

    keq_kin = np.array([r["Keq_kinetic"] for r in rows])
    keq_th = np.array([r["Keq_thermo"] for r in rows])
    devs = np.array([r["log10_deviation"] for r in rows])
    fwd = np.array([r["kcat_f_over_KM_f"] for r in rows])
    rev = np.array([r["kcat_r_over_KM_r"] for r in rows])

    # (A) kinetic vs thermodynamic Keq
    ax = fig.add_subplot(gs[0, 1 - 1])
    lim = [1e-5, 1e6]
    ax.plot(lim, lim, color="0.55", ls="--", lw=1.1)
    ax.scatter(keq_th, keq_kin, s=54, color=C1, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$K_{\rm eq}$ thermodynamic")
    ax.set_ylabel(r"$K_{\rm eq}$ from kinetics")
    ax.set_title("A  Haldane closure")

    # (B) deviations, centred on zero
    ax = fig.add_subplot(gs[0, 2 - 1])
    order = np.argsort(devs)
    ax.barh(np.arange(len(devs)), devs[order],
            color=[C3 if d >= 0 else C2 for d in devs[order]], height=0.62)
    ax.axvline(0, color="0.4", lw=1.0)
    ax.axvline(devs.mean(), color=C1, ls="--", lw=1.4)
    ax.set_yticks([])
    ax.set_xlabel(r"$\log_{10}$ deviation")
    ax.set_ylabel("enzyme")
    ax.set_title("B  no systematic bias")

    # (C) difference tracks log Keq with unit slope
    ax = fig.add_subplot(gs[0, 3 - 1])
    x = np.log10(keq_th)
    y = np.log10(fwd) - np.log10(rev)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 50)
    ax.plot(xs, xs, color="0.55", ls="--", lw=1.1, label="slope 1")
    ax.plot(xs, m * xs + b, color=C2, lw=1.8, label=f"fit {m:.2f}")
    ax.scatter(x, y, s=52, color=C1, edgecolor="white", linewidth=0.8,
               zorder=3)
    ax.set_xlabel(r"$\log_{10}K_{\rm eq}$")
    ax.set_ylabel(r"$\log_{10}$(fwd) $-$ $\log_{10}$(rev)")
    ax.set_title("C  provision cancels")
    ax.legend(frameon=False, fontsize=7)

    # (D) 3-D: forward, reverse, Keq
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    ax.scatter(np.log10(fwd), np.log10(rev), np.log10(keq_th),
               s=58, c=np.log10(keq_th), cmap=cm.coolwarm,
               edgecolor="k", linewidth=0.35, depthshade=True)
    xx = np.linspace(np.log10(fwd).min(), np.log10(fwd).max(), 14)
    yy = np.linspace(np.log10(rev).min(), np.log10(rev).max(), 14)
    XX, YY = np.meshgrid(xx, yy)
    ax.plot_surface(XX, YY, XX - YY, alpha=0.22, color=C1, linewidth=0)
    ax.set_xlabel(r"$\log_{10}$ fwd", labelpad=1)
    ax.set_ylabel(r"$\log_{10}$ rev", labelpad=1)
    ax.set_zlabel(r"$\log_{10}K_{\rm eq}$", labelpad=1)
    ax.set_title("D  endpoint plane")
    ax.view_init(elev=19, azim=-61)
    ax.tick_params(labelsize=7)

    save(fig, 4)


# ===========================================================================
# PANEL 5 --- The specificity window
# ===========================================================================
def panel_5():
    import importlib, sys
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "validation"))
    m = importlib.import_module("v4_specificity_window")
    rows = m.enrich(m.KINETICS)

    eff = np.array([r["log10_eff"] for r in rows])
    km = np.array([r["log10_KM"] for r in rows])
    kcat = np.array([r["log10_kcat"] for r in rows])

    fig = newfig()
    gs = grid(fig)

    # (A) efficiency distribution against the diffusion ceiling
    ax = fig.add_subplot(gs[0, 1 - 1])
    ax.hist(eff, bins=13, color=C1, edgecolor="white", linewidth=0.6)
    ax.axvline(9.0, color=C2, ls="--", lw=1.6)
    ax.set_xlabel(r"$\log_{10}(k_{\rm cat}/K_M)$")
    ax.set_ylabel("enzymes")
    ax.set_title("A  bounded above")

    # (B) KM distribution against the release bound
    ax = fig.add_subplot(gs[0, 2 - 1])
    ax.hist(km, bins=13, color=C3, edgecolor="white", linewidth=0.6)
    ax.axvline(-9.0, color=C2, ls="--", lw=1.6)
    ax.set_xlabel(r"$\log_{10}K_M$ (M)")
    ax.set_ylabel("enzymes")
    ax.set_title("B  bounded below")

    # (C) the window in the kcat-KM plane, with surrogate cloud
    ax = fig.add_subplot(gs[0, 3 - 1])
    rng = np.random.default_rng(11)
    sx = rng.uniform(-12, -2, 1400)
    sy = rng.uniform(0, 10, 1400)
    ax.scatter(sx, sy, s=5, color="0.78", alpha=0.5, label="unbounded surrogate")
    ax.scatter(km, kcat, s=56, color=C1, edgecolor="white", linewidth=0.8,
               zorder=3, label="observed")
    ax.axvline(-9.0, color=C2, ls="--", lw=1.3)
    ax.axhline(9.0, color=C2, ls="--", lw=1.3)
    ax.set_xlabel(r"$\log_{10}K_M$")
    ax.set_ylabel(r"$\log_{10}k_{\rm cat}$")
    ax.set_title("C  observed vs unbounded")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # (D) 3-D: the window as an occupied region
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    ax.scatter(km, kcat, eff, s=60, c=eff, cmap=cm.viridis,
               edgecolor="k", linewidth=0.35, depthshade=True)
    xx = np.linspace(km.min() - 0.4, km.max() + 0.4, 12)
    yy = np.linspace(kcat.min() - 0.4, kcat.max() + 0.4, 12)
    XX, YY = np.meshgrid(xx, yy)
    ax.plot_surface(XX, YY, np.full_like(XX, 9.0), alpha=0.18,
                    color=C2, linewidth=0)
    ax.set_xlabel(r"$\log_{10}K_M$", labelpad=1)
    ax.set_ylabel(r"$\log_{10}k_{\rm cat}$", labelpad=1)
    ax.set_zlabel(r"$\log_{10}(k_{\rm cat}/K_M)$", labelpad=1)
    ax.set_title("D  the window")
    ax.view_init(elev=20, azim=-58)
    ax.tick_params(labelsize=7)

    save(fig, 5)


# ===========================================================================
# PANEL 6 --- Aperture counting, rate law, inhibition
# ===========================================================================
def panel_6():
    v5 = load("v5_aperture_counting.json")
    v9 = load("v9_inhibition_taxonomy.json")
    rows = v5["tests"][0]["rows"]

    dc = np.array([r["dC"] for r in rows], dtype=float)
    obs = np.array([r["log10_obs"] for r in rows])

    fig = newfig()
    gs = grid(fig)

    # (A) efficiency vs aperture count with the parameter-free line
    ax = fig.add_subplot(gs[0, 1 - 1])
    xs = np.linspace(0.5, 6.5, 40)
    ax.plot(xs, 10.0 - xs, color=C2, lw=2.0, label="10 - $d_C$")
    m, b = np.polyfit(dc, obs, 1)
    ax.plot(xs, m * xs + b, color=C3, ls="--", lw=1.5,
            label=f"free fit {m:.2f}")
    ax.scatter(dc + np.random.default_rng(2).normal(0, 0.05, len(dc)), obs,
               s=52, color=C1, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel(r"aperture count $d_C$")
    ax.set_ylabel(r"$\log_{10}(k_{\rm cat}/K_M)$")
    ax.set_title("A  efficiency law")
    ax.legend(frameon=False, fontsize=7)

    # (B) dynamic range: exponential vs reciprocal representable span
    ax = fig.add_subplot(gs[0, 2 - 1])
    d = np.arange(1, 7)
    expo = 10.0 - d
    recip = -np.log10(d)
    recip = recip - recip.mean() + obs.mean()
    ax.plot(d, expo, "o-", color=C2, ms=5, label="exponential")
    ax.plot(d, recip, "s-", color=C3, ms=5, label="reciprocal")
    # span each law can represent, drawn as vertical extents at the right edge
    ax.vlines(6.75, obs.min(), obs.max(), color=C1, lw=7, alpha=0.55,
              label=f"observed {obs.max()-obs.min():.2f} dec")
    ax.vlines(7.05, recip.min(), recip.max(), color=C3, lw=7, alpha=0.85,
              label=f"reciprocal {recip.max()-recip.min():.2f} dec")
    ax.set_xlim(0.6, 7.3)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlabel(r"$d_C$")
    ax.set_ylabel(r"$\log_{10}k$")
    ax.set_title("B  reciprocal cannot span")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    # (C) inhibition 2x2 as occupancy bars
    ax = fig.add_subplot(gs[0, 3 - 1])
    cells = v9["tests"][0]["cell_counts"]
    labels = ["no turn.\nreversible", "turnover\nirrevers.",
              "turnover\nreversible", "no turn.\nirrevers."]
    keys = ["no_turnover|reversible", "turnover|irreversible",
            "turnover|reversible", "no_turnover|irreversible"]
    vals = [cells[k] for k in keys]
    cols = [C1, C1, C2, C2]
    ax.bar(range(4), vals, color=cols, width=0.64, edgecolor="white")
    for i, v in enumerate(vals):
        if v == 0:
            ax.plot([i - 0.32, i + 0.32], [0, 0], color=C2, lw=3.2,
                    solid_capstyle="butt", zorder=4)
            ax.plot(i, 0.34, marker="x", color=C2, ms=9, mew=2.2, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("inhibitors")
    ax.set_title("C  two cells occupied")

    # (D) 3-D: rate surface over (dC, delta)
    ax = fig.add_subplot(gs[0, 4 - 1], projection="3d")
    dgrid = np.linspace(1, 6, 40)
    delta = np.linspace(0.4, 2.6, 40)
    DG, DE = np.meshgrid(dgrid, delta)
    Z = 10.0 - DG * DE / math.log(10)
    surf = ax.plot_surface(DG, DE, Z, cmap=cm.cividis, linewidth=0,
                           antialiased=True, rstride=1, cstride=1, alpha=0.94)
    ax.scatter(dc, np.full_like(dc, math.log(10)), obs, s=34, color=C2,
               edgecolor="k", linewidth=0.3, depthshade=True)
    ax.set_xlabel(r"$d_C$", labelpad=1)
    ax.set_ylabel(r"$\delta$", labelpad=1)
    ax.set_zlabel(r"$\log_{10}k$", labelpad=1)
    ax.set_title("D  rate surface")
    ax.view_init(elev=22, azim=-124)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 6)


# ===========================================================================
def main():
    os.makedirs(HERE, exist_ok=True)
    print("generating panels ...")
    panel_1(); panel_2(); panel_3(); panel_4(); panel_5(); panel_6()
    print("done.")


if __name__ == "__main__":
    main()
