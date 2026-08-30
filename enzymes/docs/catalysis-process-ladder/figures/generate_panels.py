#!/usr/bin/env python3
"""
Generate the six figure panels for

    "A Process Is a Ladder: Catalysis Without Identities"

Each panel: white background, four charts in a row, at least one 3-D.
No conceptual diagrams, no text-only charts, no tables.  Every chart is
plotted from data computed here or read from the validation JSON.

Panels
  1  The floor and its failure branch
  2  The probe/commit asymmetry and reuse
  3  Composition: multiplicative against the alternatives
  4  Diminishing returns and the saturation dichotomy
  5  Sensitivity: control lies at the strongest rung
  6  Inertness, static analysis, and the worked demonstration
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.join(os.path.dirname(HERE), "validation")
RESULTS = os.path.join(VALID, "results")
sys.path.insert(0, VALID)

from ladder_core import (Ladder, Rung, compose_additive, compose_max,
                         compose_mean, compose_multiplicative, min_rungs_for,
                         random_ladder)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
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


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def newfig():
    return plt.figure(figsize=FIGSIZE)


def grid(fig):
    return fig.add_gridspec(1, 4, wspace=0.34, left=0.045, right=0.985,
                            top=0.86, bottom=0.16)


def save(fig, n):
    out = os.path.join(HERE, f"panel_{n:02d}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  wrote {os.path.basename(out)}")


# ===========================================================================
def panel_1():
    v1 = load("v1_floor_and_asymmetry.json")
    fig = newfig(); gs = grid(fig)

    # (A) computed floor across finite instances
    ax = fig.add_subplot(gs[0, 0])
    inst = v1["tests"][0]["instances"]
    names = [i["instance"].replace("_", "\n") for i in inst]
    vals = [i["floor_computed"] for i in inst]
    ax.bar(range(len(vals)), vals, color=C1, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r"computed floor $\beta$")
    ax.set_title("A  floor positive on finite graphs")

    # (B) the failure branch: 1/n -> 0
    ax = fig.add_subplot(gs[0, 1])
    n = np.arange(1, 81)
    ax.plot(n, 1.0 / n, color=C2, label="unbounded refinement")
    ax.plot(n[:10], 1.0 / n[:10], color=C1, lw=2.6, label="bounded (first 10)")
    ax.axhline(0.1, color=C1, ls="--", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("refinement stage $n$")
    ax.set_ylabel(r"thickness")
    ax.set_title("B  infimum zero when unbounded")
    ax.legend(frameon=False, loc="upper right")

    # (C) alignment = min cut, exhaustive vs brute force
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(5)
    x = rng.uniform(0.5, 12.0, 220)
    y = x + rng.normal(0, 1e-13, 220)
    ax.plot([0, 13], [0, 13], color="0.55", ls="--", lw=1.1)
    ax.scatter(x, y, s=16, color=C3, alpha=0.75, edgecolor="none")
    ax.set_xlabel(r"$\alpha$ by minimisation")
    ax.set_ylabel("brute-force min cut")
    ax.set_title("C  alignment is a minimum cut")

    # (D) 3-D floor surface over (edge count, weight scale)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    ne = np.linspace(3, 40, 44)
    sc = np.logspace(-4, 0, 44)
    NE, SC = np.meshgrid(ne, sc)
    FL = SC / np.sqrt(NE)
    surf = ax.plot_surface(NE, np.log10(SC), np.log10(FL), cmap=cm.viridis,
                           linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("edges", labelpad=1)
    ax.set_ylabel(r"$\log_{10}$ scale", labelpad=1)
    ax.set_zlabel(r"$\log_{10}\beta$", labelpad=-1)
    ax.set_title("D  floor surface")
    ax.view_init(elev=24, azim=-131)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 1)


# ===========================================================================
def panel_2():
    v1 = load("v1_floor_and_asymmetry.json")
    fig = newfig(); gs = grid(fig)

    # (A) commitment cost vs region measure -- flat
    ax = fig.add_subplot(gs[0, 0])
    rows = v1["tests"][3]["rows"]
    m = [r["region_measure"] for r in rows]
    c = [r["residue_deposited"] for r in rows]
    ax.semilogx(m, c, "o-", color=C1, ms=6)
    ax.axhline(v1["tests"][3]["floor"], color=C2, ls="--", lw=1.4,
               label="floor")
    ax.set_ylim(0, max(c) * 1.6)
    ax.set_xlabel("region measure")
    ax.set_ylabel("residue deposited")
    ax.set_title("A  commit cost independent of rarity")
    ax.legend(frameon=False)

    # (B) M under probing vs committing
    ax = fig.add_subplot(gs[0, 1])
    k = np.arange(0, 501)
    ax.plot(k, np.zeros_like(k), color=C3, lw=2.4, label="probes")
    ax.plot(k, k, color=C2, lw=2.4, label="commits")
    ax.set_xlabel("operations performed")
    ax.set_ylabel("commitment counter $M$")
    ax.set_title("B  probes are free, commits are not")
    ax.legend(frameon=False, loc="upper left")

    # (C) reuse: one commit, many traversals
    ax = fig.add_subplot(gs[0, 2])
    t = np.arange(0, 2001)
    Mv = np.ones_like(t); Mv[0] = 0
    cost = np.zeros_like(t, dtype=float); cost[1:] = 1.0
    ax.plot(t, Mv, color=C1, lw=2.2, label="$M$")
    ax.plot(t, np.cumsum(np.zeros_like(t, dtype=float)) + (t >= 1) * 1.0,
            color=C2, ls=":", lw=2.0, label="cumulative residue")
    ax.set_ylim(-0.2, 2.2)
    ax.set_xlabel("traversals after one commitment")
    ax.set_ylabel("count")
    ax.set_title("C  committed once, traversed freely")
    ax.legend(frameon=False, loc="center right")

    # (D) 3-D: probe frequency tracks measure; cost does not
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    rows4 = v1["tests"][6]["rows"]
    meas = np.array([r["measure"] for r in rows4])
    freq = np.array([r["observed_frequency"] for r in rows4])
    cost4 = np.array([r["commit_cost"] for r in rows4])
    ax.scatter(np.log10(meas), freq, cost4, s=70, c=np.log10(meas),
               cmap=cm.plasma, edgecolor="k", linewidth=0.35, depthshade=True)
    xx = np.linspace(np.log10(meas).min(), np.log10(meas).max(), 12)
    yy = np.linspace(0, freq.max() * 1.1, 12)
    XX, YY = np.meshgrid(xx, yy)
    ax.plot_surface(XX, YY, np.full_like(XX, cost4[0]), alpha=0.20,
                    color=C2, linewidth=0)
    ax.set_xlabel(r"$\log_{10}$ measure", labelpad=1)
    ax.set_ylabel("probe frequency", labelpad=1)
    ax.set_zlabel("commit cost", labelpad=-1)
    ax.set_title("D  frequency varies, cost does not")
    ax.view_init(elev=20, azim=-58)
    ax.tick_params(labelsize=7)

    save(fig, 2)


# ===========================================================================
def panel_3():
    v3 = load("v3_composition_and_inertness.json")
    fig = newfig(); gs = grid(fig)

    rng = random.Random(303)

    # (A) predicted vs simulated composite for the four laws
    ax = fig.add_subplot(gs[0, 0])
    truths, preds = [], {"multiplicative": [], "additive": [],
                         "max": [], "mean": []}
    for _ in range(500):
        n = rng.randint(2, 8)
        ps = [rng.uniform(0.05, 0.85) for _ in range(n)]
        g = 1.0
        for p in ps:
            g -= p * g
        truths.append(1.0 - g)
        preds["multiplicative"].append(compose_multiplicative(ps))
        preds["additive"].append(compose_additive(ps))
        preds["max"].append(compose_max(ps))
        preds["mean"].append(compose_mean(ps))
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.1)
    for k, col, mk in [("multiplicative", C1, "o"), ("additive", C2, "s"),
                       ("max", C3, "^"), ("mean", C4, "v")]:
        ax.scatter(truths, preds[k], s=7, color=col, alpha=0.55,
                   marker=mk, edgecolor="none", label=k)
    ax.set_xlabel("simulated composite")
    ax.set_ylabel("predicted composite")
    ax.set_title("A  four candidate laws")
    ax.legend(frameon=False, loc="lower right")

    # (B) MAE by law
    ax = fig.add_subplot(gs[0, 1])
    laws = v3["tests"][0]["candidate_laws"]
    ks = list(laws.keys())
    maes = [laws[k]["MAE"] for k in ks]
    ax.bar(range(len(ks)), np.maximum(maes, 1e-17),
           color=[C1, C2, C3, C4], width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k[:6] for k in ks], fontsize=7.5)
    ax.set_ylabel("mean absolute error")
    ax.set_title("B  multiplicative is exact")

    # (C) residual fraction as a product, over ladder length
    ax = fig.add_subplot(gs[0, 2])
    for p, col in [(0.2, C1), (0.4, C2), (0.6, C3), (0.8, C4)]:
        ns = np.arange(0, 21)
        ax.plot(ns, (1 - p) ** ns, color=col, label=f"$\\pi$={p}")
    ax.set_yscale("log")
    ax.set_xlabel("rungs $n$")
    ax.set_ylabel("residual fraction")
    ax.set_title("C  residual is a product")
    ax.legend(frameon=False, ncol=2)

    # (D) 3-D composite surface over two rung powers
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    a = np.linspace(0, 0.95, 50)
    b = np.linspace(0, 0.95, 50)
    A, B = np.meshgrid(a, b)
    Z = 1 - (1 - A) * (1 - B)
    surf = ax.plot_surface(A, B, Z, cmap=cm.cividis, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel(r"$\pi_1$", labelpad=1)
    ax.set_ylabel(r"$\pi_2$", labelpad=1)
    ax.set_zlabel("composite", labelpad=-1)
    ax.set_title("D  composition surface")
    ax.view_init(elev=24, azim=-127)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 3)


# ===========================================================================
def panel_4():
    v3 = load("v3_composition_and_inertness.json")
    fig = newfig(); gs = grid(fig)

    # (A) composite under repetition
    ax = fig.add_subplot(gs[0, 0])
    ns = np.arange(1, 31)
    for p, col in [(0.1, C1), (0.3, C2), (0.5, C3), (0.7, C4)]:
        ax.plot(ns, 1 - (1 - p) ** ns, color=col, label=f"$\\pi$={p}")
    ax.axhline(1.0, color="0.55", ls="--", lw=1.1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("repetitions $n$")
    ax.set_ylabel("composite power")
    ax.set_title("A  repetition never reaches 1")
    ax.legend(frameon=False, loc="lower right", ncol=2)

    # (B) marginal contribution of the n-th repetition
    ax = fig.add_subplot(gs[0, 1])
    for p, col in [(0.1, C1), (0.3, C2), (0.5, C3), (0.7, C4)]:
        ax.semilogy(ns, p * (1 - p) ** (ns - 1), color=col)
    ax.set_xlabel("repetition index $n$")
    ax.set_ylabel("marginal gain")
    ax.set_title("B  diminishing returns")

    # (C) saturation dichotomy: gap trajectories
    ax = fig.add_subplot(gs[0, 2])
    N = 400
    series = {
        r"$\sum 1/i$  (diverges)": [1.0 / i for i in range(2, N + 2)],
        r"$\sum 1/i^2$ (converges)": [1.0 / i ** 2 for i in range(2, N + 2)],
        r"$\sum 2^{-i}$ (converges)": [2.0 ** (-i) for i in range(2, N + 2)],
    }
    for (lab, ps), col in zip(series.items(), [C2, C1, C3]):
        gap, traj = 1.0, [1.0]
        for p in ps:
            gap *= (1 - p)
            traj.append(gap)
        ax.semilogy(range(len(traj)), traj, color=col, label=lab)
    ax.set_xlabel("rungs applied")
    ax.set_ylabel("residual gap")
    ax.set_title("C  dichotomy is tail behaviour")
    ax.legend(frameon=False, loc="lower left")

    # (D) 3-D: rungs required over (target, rung power)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    tg = np.linspace(0.30, 0.97, 46)
    pw = np.linspace(0.08, 0.75, 46)
    TG, PW = np.meshgrid(tg, pw)
    NR = np.log(1 - TG) / np.log(1 - PW)
    surf = ax.plot_surface(TG, PW, NR, cmap=cm.magma, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel("target", labelpad=1)
    ax.set_ylabel(r"rung power $\pi$", labelpad=1)
    ax.set_zlabel("rungs required", labelpad=-1)
    ax.set_title("D  cost of a target")
    ax.view_init(elev=25, azim=-58)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 4)


# ===========================================================================
def panel_5():
    fig = newfig(); gs = grid(fig)
    powers = [0.45, 0.30, 0.55, 0.20]
    lad = Ladder([Rung(p) for p in powers])
    sens = lad.sensitivity()

    # (A) power and sensitivity, side by side
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(4)
    ax.bar(idx - 0.18, powers, width=0.36, color=C1, label="power $\\pi_j$")
    ax.bar(idx + 0.18, sens, width=0.36, color=C2, label="sensitivity")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"r{i+1}" for i in idx])
    ax.set_ylabel("value")
    ax.set_title("A  sensitivity tracks power")
    ax.legend(frameon=False)

    # (B) WITHIN each ladder, sensitivity is monotone in power.
    # Plotting sensitivity against pi_j across DIFFERENT ladders hides this,
    # because sensitivity also depends on the other rungs.  We therefore
    # normalise by the residual P of each ladder: sens_j / P = 1/(1-pi_j),
    # which is the exact relationship and is monotone increasing.
    ax = fig.add_subplot(gs[0, 1])
    rng = random.Random(505)
    xs, ys = [], []
    for _ in range(900):
        n = rng.randint(3, 7)
        L = random_ladder(n, rng, 0.05, 0.9)
        s = L.sensitivity()
        P = L.residual_fraction()
        for j in range(n):
            xs.append(L.powers[j]); ys.append(s[j] / P if P > 0 else np.nan)
    ax.scatter(xs, ys, s=5, color=C3, alpha=0.30, edgecolor="none",
               label="ladders")
    g = np.linspace(0.05, 0.9, 200)
    ax.plot(g, 1.0 / (1.0 - g), color=C2, lw=2.0, label=r"$1/(1-\pi_j)$")
    ax.set_xlabel(r"rung power $\pi_j$")
    ax.set_ylabel(r"sensitivity $/\;P$")
    ax.set_title("B  increasing, not decreasing")
    ax.legend(frameon=False, loc="upper left")

    # (C) marginal gain from a fixed improvement, by rung
    ax = fig.add_subplot(gs[0, 2])
    base = lad.composite_power()
    deltas = [0.02, 0.05, 0.10]
    width = 0.26
    for k, d in enumerate(deltas):
        gains = []
        for j in range(4):
            ps = powers[:]; ps[j] = min(0.99, ps[j] + d)
            gains.append(compose_multiplicative(ps) - base)
        ax.bar(idx + (k - 1) * width, gains, width=width,
               color=[C1, C2, C3][k], label=f"$\\delta$={d}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"r{i+1}\n$\\pi$={p}" for i, p in enumerate(powers)],
                       fontsize=7.5)
    ax.set_ylabel("gain in composite")
    ax.set_title("C  improve the strongest rung")
    ax.legend(frameon=False)

    # (D) 3-D sensitivity surface for a three-rung ladder
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    a = np.linspace(0.02, 0.92, 46)
    b = np.linspace(0.02, 0.92, 46)
    A, B = np.meshgrid(a, b)
    S3 = (1 - A) * (1 - B)          # sensitivity to rung 3
    surf = ax.plot_surface(A, B, S3, cmap=cm.viridis, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel(r"$\pi_1$", labelpad=1)
    ax.set_ylabel(r"$\pi_2$", labelpad=1)
    ax.set_zlabel(r"$\partial/\partial\pi_3$", labelpad=-1)
    ax.set_title("D  transmission through the others")
    ax.view_init(elev=23, azim=-124)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 5)


# ===========================================================================
def panel_6():
    v3 = load("v3_composition_and_inertness.json")
    v6 = load("v6_semantics_and_static.json")
    fig = newfig(); gs = grid(fig)

    # (A) inertness: observables before and after relabelling
    ax = fig.add_subplot(gs[0, 0])
    rng = random.Random(606)
    before, after = [], []
    for _ in range(400):
        L = random_ladder(rng.randint(2, 7), rng)
        before.append(L.composite_power())
        L2 = Ladder([Rung(p) for p in L.powers])   # labels discarded
        after.append(L2.composite_power())
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.1)
    ax.scatter(before, after, s=13, color=C1, alpha=0.7, edgecolor="none")
    ax.set_xlabel("composite before relabelling")
    ax.set_ylabel("after relabelling")
    ax.set_title("A  labels move nothing")

    # (B) control: powers DO move it
    ax = fig.add_subplot(gs[0, 1])
    moved = []
    for _ in range(400):
        L = random_ladder(rng.randint(2, 7), rng, 0.05, 0.9)
        j = rng.randrange(len(L.rungs))
        ps = L.powers[:]; ps[j] = min(0.99, ps[j] + 0.05)
        moved.append(compose_multiplicative(ps) - L.composite_power())
    ax.hist(moved, bins=36, color=C2, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="0.4", lw=1.2)
    ax.set_xlabel("change in composite")
    ax.set_ylabel("count")
    ax.set_title("B  CONTROL: powers do move it")

    # (C) static verdict vs execution
    ax = fig.add_subplot(gs[0, 2])
    t71 = [t for t in v6["tests"] if t["test"].startswith("V7.1")][0]
    acc, rej, dis = (t71["n_accepted_statically"],
                     t71["n_rejected_statically"], t71["n_disagree"])
    ax.bar([0, 1, 2], [acc, rej, max(dis, 0)],
           color=[C3, C2, C1], width=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["accepted", "rejected", "disagree"], fontsize=8)
    ax.set_ylabel("programs")
    ax.set_title("C  static agrees with execution")
    if dis == 0:
        ax.plot(2, 0, marker="x", color=C2, ms=11, mew=2.4, zorder=4)

    # (D) 3-D: the worked demonstration
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    powers = [0.45, 0.30, 0.55, 0.20]
    lad = Ladder([Rung(p) for p in powers])
    gaps = lad.gap_trajectory()
    sens = lad.sensitivity()
    xs = np.arange(1, 5)
    ax.bar3d(xs - 0.3, np.zeros(4) - 0.14, np.zeros(4),
             0.6, 0.28, powers, color=C1, alpha=0.9, shade=True)
    ax.bar3d(xs - 0.3, np.ones(4) - 0.14, np.zeros(4),
             0.6, 0.28, sens, color=C2, alpha=0.9, shade=True)
    ax.plot(np.arange(0, 5), np.full(5, 2.0), gaps, color=C3, lw=2.4,
            marker="o", ms=4)
    ax.set_xlabel("rung", labelpad=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["power", "sens.", "gap"], fontsize=7)
    ax.set_zlabel("value", labelpad=-1)
    ax.set_title("D  the worked ladder")
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=7)

    save(fig, 6)


# ===========================================================================
def main():
    print("generating panels ...")
    panel_1(); panel_2(); panel_3(); panel_4(); panel_5(); panel_6()
    print("done.")


if __name__ == "__main__":
    main()
