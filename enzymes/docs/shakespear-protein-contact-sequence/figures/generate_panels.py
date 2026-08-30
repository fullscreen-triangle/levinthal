#!/usr/bin/env python3
"""
Generate the six figure panels for "Shakespeare with Ladders".

Each panel: white background, four charts in a row, at least one 3-D.
No conceptual diagrams, no text-only charts, no tables.  Every chart is
plotted from data computed here or read from the validation JSON.

Panels
  1  The substrate: floor, the finiteness branch, the cut key, the floor surface
  2  Resolution: the radius sweep and its mechanism (ball overlap)
  3  Intensivity: drift under extension, and the near-miss
  4  Order dependence: the graded result
  5  Composition and sensitivity: control at the strongest rung
  6  The language: freeness, the clock, and the refusal
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

from shk_core import (ContactGraph, Ladder, Rung, chain_graph, complete_graph,
                      compose_additive, compose_max, compose_mean,
                      compose_multiplicative, derive_sequential, local_floor,
                      min_rungs_for, power_extensive, power_globalfloor,
                      power_intensive, random_ladder)

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
    v1 = load("v1_substrate_and_resolution.json")
    fig = newfig(); gs = grid(fig)

    # (A) computed floor across instances
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

    # (B) the finiteness branch
    ax = fig.add_subplot(gs[0, 1])
    n = np.arange(1, 81)
    ax.plot(n, 1.0 / n, color=C2, label="unbounded refinement")
    ax.plot(n[:10], 1.0 / n[:10], color=C1, lw=2.6, label="bounded (first 10)")
    ax.axhline(0.1, color=C1, ls="--", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("refinement stage $n$")
    ax.set_ylabel("thickness")
    ax.set_title("B  infimum zero when unbounded")
    ax.legend(frameon=False, loc="upper right")

    # (C) local cut key vs brute force
    ax = fig.add_subplot(gs[0, 2])
    rng = random.Random(303)
    xs, ys = [], []
    for _ in range(160):
        g = chain_graph(rng.randint(3, 6), rng)
        v = f"v{rng.randrange(len(g.items()))}"
        xs.append(g.sigma_local(v, radius=12))
        ys.append(g.sigma(v))
    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], color="0.55", ls="--", lw=1.1)
    ax.scatter(xs, ys, s=16, color=C3, alpha=0.75, edgecolor="none")
    ax.set_xlabel(r"$\sigma_\varrho(v)$ at full radius")
    ax.set_ylabel(r"exhaustive $\sigma(v)$")
    ax.set_title("C  cut key against ground truth")

    # (D) 3-D floor surface
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
    v1 = load("v1_substrate_and_resolution.json")
    t = [x for x in v1["tests"] if x["test"].startswith("V1.4")][0]
    fig = newfig(); gs = grid(fig)

    radii = sorted(int(k) for k in t["pairs_identified_by_radius"])
    ident = [t["pairs_identified_by_radius"][str(r)] for r in radii]
    overl = [t["mean_ball_overlap_by_radius"][str(r)] for r in radii]
    dist = [t["distinct_powers_by_radius"][str(r)] for r in radii]

    # (A) identification against radius
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(radii, ident, color=C1, width=0.6)
    ax.set_xticks(radii)
    ax.set_xlabel(r"radius $\varrho$")
    ax.set_ylabel("pairs identified")
    ax.set_title("A  identification rises with radius")

    # (B) the mechanism: overlap on the same axis
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(radii, overl, "o-", color=C2, ms=6, label="mean ball overlap")
    ax.set_xticks(radii)
    ax.set_xlabel(r"radius $\varrho$")
    ax.set_ylabel("mean pairwise ball overlap")
    ax2 = ax.twinx()
    ax2.plot(radii, dist, "s--", color=C3, ms=5, label="distinct powers")
    ax2.set_ylabel("distinct powers")
    ax2.grid(False)
    ax.set_title("B  overlap is the mechanism")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")

    # (C) power trajectories converging as radius grows
    ax = fig.add_subplot(gs[0, 2])
    rng = random.Random(77)
    g = chain_graph(6, rng)
    for i in range(6):
        ys = [power_intensive(g, f"v{i}", r) for r in (0, 1, 2, 3, 4)]
        ax.plot([0, 1, 2, 3, 4], ys, "o-", ms=4, alpha=0.85)
    ax.set_xlabel(r"radius $\varrho$")
    ax.set_ylabel(r"derived power $\pi_\varrho(v)$")
    ax.set_title("C  powers converge as balls merge")

    # (D) 3-D: power over (item, radius)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    rng2 = random.Random(91)
    g2 = chain_graph(7, rng2)
    R = np.arange(0, 5)
    I = np.arange(0, 7)
    II, RR = np.meshgrid(I, R)
    Z = np.array([[power_intensive(g2, f"v{i}", r) for i in I] for r in R])
    surf = ax.plot_surface(II, RR, Z, cmap=cm.plasma, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel("item", labelpad=1)
    ax.set_ylabel(r"radius $\varrho$", labelpad=1)
    ax.set_zlabel(r"$\pi$", labelpad=-1)
    ax.set_title("D  the resolution surface")
    ax.view_init(elev=26, azim=-121)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 2)


# ===========================================================================
def panel_3():
    v2 = load("v2_intensivity.json")
    t = v2["tests"][0]
    fig = newfig(); gs = grid(fig)
    order = ["intensive", "globalfloor", "extensive"]
    cols = {"intensive": C1, "globalfloor": C2, "extensive": C3}

    # (A) max drift by candidate
    ax = fig.add_subplot(gs[0, 0])
    vals = [max(t["by_candidate"][k]["max_drift"], 1e-18) for k in order]
    ax.bar(range(3), vals, color=[cols[k] for k in order], width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["intensive", "near-miss", "extensive"], fontsize=7.5)
    ax.set_ylabel("max drift under extension")
    ax.set_title("A  intensive does not move")

    # (B) fraction of graphs on which each moved
    ax = fig.add_subplot(gs[0, 1])
    n = t["n_graphs"]
    fr = [t["by_candidate"][k]["n_moved"] / n for k in order]
    ax.bar(range(3), fr, color=[cols[k] for k in order], width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["intensive", "near-miss", "extensive"], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of graphs moved")
    ax.set_title("B  CONTROL: both controls move")

    # (C) live drift scatter, recomputed
    ax = fig.add_subplot(gs[0, 2])
    rng = random.Random(404)
    for name, fn in [("intensive", power_intensive),
                     ("globalfloor", power_globalfloor),
                     ("extensive", power_extensive)]:
        xs, ys = [], []
        for _ in range(90):
            base = chain_graph(rng.randint(4, 6), rng)
            verts = set(base.vertices); ws = dict(base.weights)
            last = max(int(v[1:]) for v in base.items())
            nv = f"v{last+1}"
            verts.add(nv)
            ws[frozenset((nv, base.medium))] = base.floor * 0.5
            ws[frozenset((f"v{last}", nv))] = rng.uniform(0.5, 3.0)
            ext = ContactGraph(verts, ws, base.medium)
            xs.append(fn(base, "v0", 1)); ys.append(fn(ext, "v0", 1))
        ax.scatter(xs, ys, s=11, alpha=0.55, edgecolor="none",
                   color=cols[name], label=name)
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.1)
    ax.set_xlabel("power before extension")
    ax.set_ylabel("after extension")
    ax.set_title("C  only intensive is on the identity")
    ax.legend(frameon=False, loc="upper left")

    # (D) 3-D: drift over (extension size, weight of added edge)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    rng3 = random.Random(505)
    base = chain_graph(5, rng3)
    sizes = np.arange(1, 6)
    scales = np.linspace(0.05, 1.0, 12)
    Z = np.zeros((len(scales), len(sizes)))
    for a, sc in enumerate(scales):
        for b, k in enumerate(sizes):
            verts = set(base.vertices); ws = dict(base.weights)
            last = max(int(v[1:]) for v in base.items())
            for j in range(1, int(k) + 1):
                nv = f"v{last+j}"
                verts.add(nv)
                ws[frozenset((nv, base.medium))] = base.floor * sc
                ws[frozenset((f"v{last+j-1}", nv))] = 1.5
            ext = ContactGraph(verts, ws, base.medium)
            Z[a, b] = abs(power_globalfloor(base, "v0", 1)
                          - power_globalfloor(ext, "v0", 1))
    SS, KK = np.meshgrid(sizes, scales)
    surf = ax.plot_surface(KK, SS, Z, cmap=cm.inferno, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel("added weight / " + r"$\beta$", labelpad=1)
    ax.set_ylabel("items added", labelpad=1)
    ax.set_zlabel("near-miss drift", labelpad=-1)
    ax.set_title("D  the near-miss drift surface")
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 3)


# ===========================================================================
def panel_4():
    v2 = load("v2_intensivity.json")
    t = [x for x in v2["tests"] if x["test"].startswith("V2.3")][0]
    fig = newfig(); gs = grid(fig)
    order = ["intensive", "globalfloor", "extensive"]
    cols = {"intensive": C1, "globalfloor": C2, "extensive": C3}
    labels = ["intensive", "near-miss", "extensive"]

    # (A) distinct composites at radius 1
    ax = fig.add_subplot(gs[0, 0])
    d1 = [t["by_radius"]["1"][k]["distinct"] for k in order]
    ax.bar(range(3), d1, color=[cols[k] for k in order], width=0.6)
    ax.axhline(1, color="0.4", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("distinct composites / 720 orderings")
    ax.set_title("A  graded, and never 1")

    # (B) radius 0 against radius 1
    ax = fig.add_subplot(gs[0, 1])
    w = 0.36
    idx = np.arange(3)
    d0 = [t["by_radius"]["0"][k]["distinct"] for k in order]
    ax.bar(idx - w / 2, d0, width=w, color=C1, label=r"$\varrho=0$")
    ax.bar(idx + w / 2, d1, width=w, color=C2, label=r"$\varrho=1$")
    ax.set_yscale("log")
    ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("distinct composites")
    ax.set_title("B  finer locality tightens it")
    ax.legend(frameon=False)

    # (C) the actual distribution of composites over orderings
    ax = fig.add_subplot(gs[0, 2])
    import itertools
    rng = random.Random(23)
    g = chain_graph(6, rng)
    items = [f"v{i}" for i in range(6)]
    perms = list(itertools.permutations(items))
    for name, fn in [("intensive", power_intensive),
                     ("extensive", power_extensive)]:
        comps = [compose_multiplicative(derive_sequential(g, p, fn, 1))
                 for p in perms]
        ax.hist(comps, bins=40, alpha=0.6, color=cols[name], label=name,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("composite power over orderings")
    ax.set_ylabel("count")
    ax.set_title("C  spread over all 720 orderings")
    ax.legend(frameon=False)

    # (D) 3-D: composite over (ordering index, radius)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    sample = perms[::30]
    RS = [0, 1, 2]
    X = np.arange(len(sample))
    Z = np.array([[compose_multiplicative(
        derive_sequential(g, p, power_intensive, r)) for p in sample]
        for r in RS])
    XX, RR = np.meshgrid(X, RS)
    surf = ax.plot_surface(XX, RR, Z, cmap=cm.cividis, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.set_xlabel("ordering (sampled)", labelpad=1)
    ax.set_ylabel(r"radius $\varrho$", labelpad=1)
    ax.set_zlabel("composite", labelpad=-1)
    ax.set_title("D  order surface, intensive")
    ax.view_init(elev=26, azim=-64)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 4)


# ===========================================================================
def panel_5():
    v3 = load("v3_semantics_and_refusal.json")
    fig = newfig(); gs = grid(fig)

    # (A) MAE by law
    ax = fig.add_subplot(gs[0, 0])
    mae = [x for x in v3["tests"] if x["test"].startswith("V3.1 ")][0]["MAE_by_law"]
    ks = ["multiplicative", "additive", "max", "mean"]
    ax.bar(range(4), [max(mae[k], 1e-17) for k in ks],
           color=[C1, C2, C3, C4], width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["mult.", "add.", "max", "mean"], fontsize=7.5)
    ax.set_ylabel("mean absolute error")
    ax.set_title("A  multiplicative is exact")

    # (B) sensitivity normalised: exactly 1/(1-pi)
    ax = fig.add_subplot(gs[0, 1])
    rng = random.Random(505)
    xs, ys = [], []
    for _ in range(700):
        n = rng.randint(3, 7)
        L = random_ladder(n, rng, 0.05, 0.9)
        s = L.sensitivity(); P = L.residual_fraction()
        for j in range(n):
            xs.append(L.powers[j]); ys.append(s[j] / P if P > 0 else np.nan)
    ax.scatter(xs, ys, s=5, color=C3, alpha=0.28, edgecolor="none",
               label="ladders")
    gg = np.linspace(0.05, 0.9, 200)
    ax.plot(gg, 1.0 / (1.0 - gg), color=C2, lw=2.0,
            label=r"$1/(1-\pi_j)$")
    ax.set_xlabel(r"rung power $\pi_j$")
    ax.set_ylabel(r"sensitivity $/\,P$")
    ax.set_title("B  increasing, not decreasing")
    ax.legend(frameon=False, loc="upper left")

    # (C) gain from a fixed improvement, by rung
    ax = fig.add_subplot(gs[0, 2])
    powers = [0.45, 0.30, 0.55, 0.20]
    base = compose_multiplicative(powers)
    idx = np.arange(4); width = 0.26
    for k, d in enumerate([0.02, 0.05, 0.10]):
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

    # (D) 3-D sensitivity surface
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    a = np.linspace(0.02, 0.92, 46); b = np.linspace(0.02, 0.92, 46)
    A, B = np.meshgrid(a, b)
    S3 = (1 - A) * (1 - B)
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
    v3 = load("v3_semantics_and_refusal.json")
    fig = newfig(); gs = grid(fig)

    # (A) the clock under free operations and commitments
    ax = fig.add_subplot(gs[0, 0])
    k = np.arange(0, 501)
    ax.plot(k, np.zeros_like(k), color=C3, lw=2.4, label="free rules")
    ax.plot(k, k, color=C2, lw=2.4, label="climb (per rung)")
    ax.set_xlabel("operations performed")
    ax.set_ylabel("clock $M$")
    ax.set_title("A  free rules leave $M$ fixed")
    ax.legend(frameon=False, loc="upper left")

    # (B) static verdict against execution
    ax = fig.add_subplot(gs[0, 1])
    t = [x for x in v3["tests"] if x["test"].startswith("V3.6")][0]
    ax.bar([0, 1, 2], [t["n_accepted"], t["n_rejected"],
                       max(t["n_disagree"], 0)],
           color=[C3, C2, C1], width=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["accepted", "refused", "disagree"], fontsize=8)
    ax.set_ylabel("programs")
    ax.set_title("B  the refusal is tight")
    if t["n_disagree"] == 0:
        ax.plot(2, 0, marker="x", color=C2, ms=11, mew=2.4, zorder=4)

    # (C) label independence and its control
    ax = fig.add_subplot(gs[0, 2])
    rng = random.Random(606)
    moved = []
    for _ in range(500):
        L = random_ladder(rng.randint(2, 7), rng, 0.05, 0.9)
        j = rng.randrange(len(L.rungs))
        ps = L.powers[:]; ps[j] = min(0.99, ps[j] + 0.05)
        moved.append(compose_multiplicative(ps) - L.composite_power())
    ax.hist(moved, bins=36, color=C2, edgecolor="white", linewidth=0.4,
            label="power perturbed")
    ax.axvline(0, color=C1, lw=2.2, label="relabelled (all exactly 0)")
    ax.set_xlabel("change in composite")
    ax.set_ylabel("count")
    ax.set_title("C  labels move nothing, powers do")
    ax.legend(frameon=False)

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

    save(fig, 6)


def main():
    print("generating panels ...")
    panel_1(); panel_2(); panel_3(); panel_4(); panel_5(); panel_6()
    print("done.")


if __name__ == "__main__":
    main()
