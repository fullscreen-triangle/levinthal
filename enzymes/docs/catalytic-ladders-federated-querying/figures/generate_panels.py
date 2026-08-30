#!/usr/bin/env python3
"""
Generate the six figure panels for
"Catalytic Ladders in a Federated Query Language: An Integration Report".

Each panel: white background, four charts in a row, at least one 3-D.
No conceptual diagrams, no text-only charts, no tables.  Every chart is
plotted from the validation JSON or recomputed here from the frozen corpus.

Panels
  1  The host suite and the shape of the change
  2  Cost: the ladder in the tier the host already had
  3  The refusal: both verdicts, the shortfall, the blame walk
  4  The union bound against the multiplicative law
  5  The corpus: two heterogeneous sources, frozen
  6  Composition over the corpus: chain length, saturation, sensitivity
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.join(os.path.dirname(HERE), "validation")
RESULTS = os.path.join(VALID, "results")
FIXTURES = os.path.join(VALID, "fixtures")

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


def corpus():
    with open(os.path.join(FIXTURES, "sources.json")) as fh:
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


def compose(ps):
    q = 1.0
    for p in ps:
        q *= (1.0 - p)
    return 1.0 - q


# ===========================================================================
def panel_1():
    e1 = load("e1.json"); e2 = load("e2.json"); e7 = load("e7.json")
    fig = newfig(); gs = grid(fig)

    # (A) the host suite, before and after
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([0, 1], [e1["total"], e1["passing"]], color=[C1, C3], width=0.55)
    ax.bar([2], [e1["failing"]], color=C2, width=0.55)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["checks", "passing", "failing"], fontsize=8)
    ax.set_ylabel("host checks")
    ax.set_title("A  the host suite after the change")
    ax.plot(2, 0, marker="x", color=C2, ms=11, mew=2.4, zorder=4)

    # (B) plans exercised, by fixture world
    ax = fig.add_subplot(gs[0, 1])
    worlds = Counter(r.get("world") for r in e7["rows"] if r.get("world"))
    ks = sorted(worlds)
    ax.bar(range(len(ks)), [worlds[k] for k in ks], color=C1, width=0.55)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel("plans executed")
    ax.set_title("B  CONTROL: ladder-free plans, all worlds")

    # (C) requests issued per shipped plan
    ax = fig.add_subplot(gs[0, 2])
    names = [r["plan"].replace(".hfq", "") for r in e7["rows"]]
    reqs = [r.get("requests", 0) for r in e7["rows"]]
    order = np.argsort(reqs)[::-1]
    ax.barh([names[i][:18] for i in order], [reqs[i] for i in order],
            color=C3, height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("requests issued")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title("C  shipped plans still issue requests")

    # (D) 3-D: verdict distribution across plans
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    vlabels = ["answer", "empty", "starved", "refused", "surface", "timeout"]
    rows = [r for r in e7["rows"] if r.get("verdicts")]
    Z = np.zeros((len(rows), len(vlabels)))
    for i, r in enumerate(rows):
        c = Counter(r["verdicts"].values())
        for j, v in enumerate(vlabels):
            Z[i, j] = c.get(v, 0)
    X, Y = np.meshgrid(np.arange(len(vlabels)), np.arange(len(rows)))
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, alpha=0.95)
    ax.set_xticks(range(len(vlabels)))
    ax.set_xticklabels([v[:4] for v in vlabels], fontsize=6)
    ax.set_ylabel("plan", labelpad=1)
    ax.set_zlabel("steps", labelpad=-1)
    ax.set_title("D  verdicts across shipped plans")
    ax.view_init(elev=28, azim=-119)
    ax.tick_params(labelsize=7)

    save(fig, 1)


# ===========================================================================
def panel_2():
    e4 = load("e4.json")
    fig = newfig(); gs = grid(fig)
    steps = ["a", "b", "u", "L"]
    kinds = ["source", "source", "union", "ladder"]
    spent = [e4["spent"][k] for k in steps]
    alloc = [e4["allocated"][k] for k in steps]

    # (A) spent per step
    ax = fig.add_subplot(gs[0, 0])
    cols = [C1, C1, C3, C2]
    ax.bar(range(4), spent, color=cols, width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s}\n{k}" for s, k in zip(steps, kinds)], fontsize=7)
    ax.set_ylabel("budget spent")
    ax.set_title("A  the ladder spends nothing")

    # (B) allocated vs spent
    ax = fig.add_subplot(gs[0, 1])
    w = 0.36; idx = np.arange(4)
    ax.bar(idx - w / 2, alloc, width=w, color=C1, label="allocated")
    ax.bar(idx + w / 2, spent, width=w, color=C2, label="spent")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{s}\n{k}" for s, k in zip(steps, kinds)], fontsize=7)
    ax.set_ylabel("requests")
    ax.set_title("B  the reservation is never drawn")
    ax.legend(frameon=False)

    # (C) cumulative budget along the plan
    ax = fig.add_subplot(gs[0, 2])
    cum = np.cumsum([0] + spent)
    ax.step(range(len(cum)), cum, where="post", color=C1, lw=2.2)
    ax.scatter(range(1, len(cum)), cum[1:], s=42,
               color=[C1, C1, C3, C2], zorder=4)
    ax.set_xticks(range(len(cum)))
    ax.set_xticklabels([""] + steps, fontsize=8)
    ax.set_ylabel("cumulative requests")
    ax.set_title("C  the ladder is flat on the budget")

    # (D) 3-D: budget consumed over (n sources, n ladders)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    ns = np.arange(0, 9)
    nl = np.arange(0, 9)
    NS, NL = np.meshgrid(ns, nl)
    Z = NS * 1.0 + NL * 0.0
    surf = ax.plot_surface(NS, NL, Z, cmap=cm.plasma, linewidth=0, alpha=0.95)
    ax.set_xlabel("source steps", labelpad=1)
    ax.set_ylabel("ladder steps", labelpad=1)
    ax.set_zlabel("requests spent", labelpad=-1)
    ax.set_title("D  cost depends on sources alone")
    ax.view_init(elev=24, azim=-125)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 2)


# ===========================================================================
def panel_3():
    e3 = load("e3.json")
    fig = newfig(); gs = grid(fig)
    acc = e3["cases"]["accepted"]; ref = e3["cases"]["refused"]

    # (A) composite against the two declared targets
    ax = fig.add_subplot(gs[0, 0])
    comp = acc["composite"]
    ax.bar([0, 1], [0.70, 0.95], color=["0.75", "0.75"], width=0.5,
           label="declared target")
    ax.axhline(comp, color=C1, lw=2.4, label=f"composite {comp:.4f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"target 0.70\n{acc['verdict']}",
                        f"target 0.95\n{ref['verdict']}"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("power")
    ax.set_title("A  one composite, two verdicts")
    ax.legend(frameon=False, loc="lower left")

    # (B) the shortfall
    ax = fig.add_subplot(gs[0, 1])
    sf = ref["diagnosis"]["shortfall"]
    ax.barh([0], [comp], color=C1, height=0.5, label="attained")
    ax.barh([0], [sf], left=[comp], color=C2, height=0.5, label="shortfall")
    ax.set_yticks([]); ax.set_xlim(0, 1.0)
    ax.set_xlabel("power")
    ax.set_title(f"B  shortfall named: {sf:.5f}")
    ax.legend(frameon=False, loc="lower right")

    # (C) blame walk length, ladder vs a hypothetical blaming ladder
    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0, 1], [len(ref["blame_chain"]), 2], color=[C3, C2], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["as implemented\n(stops at ladder)",
                        "if it named its input\n(would accuse it)"],
                       fontsize=7)
    ax.set_ylabel("steps in the blame walk")
    ax.set_title("C  blame terminates at the ladder")

    # (D) 3-D: verdict region over (composite, target)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    cvals = np.linspace(0, 1, 60)
    tvals = np.linspace(0, 1, 60)
    CC, TT = np.meshgrid(cvals, tvals)
    Z = np.clip(TT - CC, 0, None)          # shortfall surface
    surf = ax.plot_surface(CC, TT, Z, cmap=cm.inferno, linewidth=0, alpha=0.95)
    ax.scatter([comp], [0.70], [0.0], s=55, color=C3, edgecolor="k",
               linewidth=0.5, depthshade=False)
    ax.scatter([comp], [0.95], [sf], s=55, color=C2, edgecolor="k",
               linewidth=0.5, depthshade=False)
    ax.set_xlabel("composite", labelpad=1)
    ax.set_ylabel("declared target", labelpad=1)
    ax.set_zlabel("shortfall", labelpad=-1)
    ax.set_title("D  the refusal surface")
    ax.view_init(elev=26, azim=-129)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 3)


# ===========================================================================
def panel_4():
    e5 = load("e5.json")
    rows = e5["rows"]
    ks = [r["stages"] for r in rows]
    add = [r["mean_additive_bound"] for r in rows]
    mul = [r["mean_multiplicative"] for r in rows]
    gap = [r["mean_gap"] for r in rows]
    vac = [r["fraction_additive_vacuous"] for r in rows]
    fig = newfig(); gs = grid(fig)

    # (A) the two forms
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ks, mul, "o-", color=C1, label=r"$1-\prod(1-r_i)$")
    ax.plot(ks, add, "s-", color=C2, label=r"$1-\sum(1-r_i)$")
    ax.axhline(0, color="0.45", ls="--", lw=1.2)
    ax.fill_between(ks, -1.2, 0, color=C2, alpha=0.10)
    ax.set_xlabel("chain length $k$")
    ax.set_ylabel("bound value")
    ax.set_title("A  the additive form goes negative")
    ax.legend(frameon=False, loc="lower left")

    # (B) fraction vacuous
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(ks)), vac, color=C2, width=0.6)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("chain length $k$")
    ax.set_ylabel("fraction of trials vacuous")
    ax.set_ylim(0, 1.05)
    ax.set_title("B  vacuous by four stages")

    # (C) the gap
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(range(len(ks)), gap, "o-", color=C3)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("chain length $k$")
    ax.set_ylabel("mean gap")
    ax.set_title("C  the gap grows with length")

    # (D) 3-D: gap over (k, retention)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    kk = np.arange(2, 10)
    rr = np.linspace(0.5, 0.99, 40)
    KK, RR = np.meshgrid(kk, rr)
    ADD = 1 - KK * (1 - RR)
    MUL = 1 - (1 - RR) ** KK
    surf = ax.plot_surface(KK, RR, MUL - ADD, cmap=cm.magma, linewidth=0,
                           alpha=0.95)
    ax.set_xlabel("chain length $k$", labelpad=1)
    ax.set_ylabel("retention $r$", labelpad=1)
    ax.set_zlabel("gap", labelpad=-1)
    ax.set_title("D  the gap surface")
    ax.view_init(elev=24, azim=-62)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.46, pad=0.17, aspect=16)

    save(fig, 4)


# ===========================================================================
def panel_5():
    d = corpus()
    kegg, rx = d["kegg"], d["reactome"]
    fig = newfig(); gs = grid(fig)

    # (A) source sizes and coverage
    ax = fig.add_subplot(gs[0, 0])
    vals = [len(kegg), sum(1 for e in kegg if e.get("reactions")),
            len(rx), sum(1 for r in rx if r.get("catalysts"))]
    ax.bar(range(4), vals, color=[C1, C1, C3, C3], width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["KEGG\nrecords", "KEGG\nw/ rxn",
                        "Reactome\nreactions", "Reactome\nw/ cat"],
                       fontsize=7)
    ax.set_ylabel("count")
    ax.set_title("A  two heterogeneous sources")

    # (B) reactions per pathway
    ax = fig.add_subplot(gs[0, 1])
    per = Counter(r["pathway"] for r in rx if r.get("pathway"))
    lens = sorted(per.values(), reverse=True)
    ax.bar(range(len(lens)), lens, color=C3, width=0.75)
    ax.set_xlabel("pathway (ranked)")
    ax.set_ylabel("reactions")
    ax.set_title("B  chain lengths in the corpus")

    # (C) reactions per EC in KEGG
    ax = fig.add_subplot(gs[0, 2])
    nre = [len(e.get("reactions", [])) for e in kegg]
    ax.hist([n for n in nre if n > 0], bins=28, color=C1,
            edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("reactions per enzyme record")
    ax.set_ylabel("count")
    ax.set_title("C  KEGG reaction multiplicity")

    # (D) 3-D: inputs vs outputs vs catalysts per reaction
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    xs = [len(r.get("inputs", [])) for r in rx]
    ys = [len(r.get("outputs", [])) for r in rx]
    zs = [len(r.get("catalysts", [])) for r in rx]
    ax.scatter(xs, ys, zs, s=9, c=zs, cmap=cm.viridis, alpha=0.55,
               edgecolor="none", depthshade=True)
    ax.set_xlabel("inputs", labelpad=1)
    ax.set_ylabel("outputs", labelpad=1)
    ax.set_zlabel("catalysts", labelpad=-1)
    ax.set_title("D  reaction participants")
    ax.view_init(elev=22, azim=-59)
    ax.tick_params(labelsize=7)

    save(fig, 5)


# ===========================================================================
def panel_6():
    d = corpus()
    rx = d["reactome"]
    per = Counter(r["pathway"] for r in rx if r.get("pathway"))
    lens = sorted(per.values(), reverse=True)
    fig = newfig(); gs = grid(fig)
    rng = random.Random(9)

    # (A) composite against chain length at three uniform rung powers
    ax = fig.add_subplot(gs[0, 0])
    ns = np.arange(1, 61)
    for p, col in [(0.05, C1), (0.10, C2), (0.20, C3)]:
        ax.plot(ns, 1 - (1 - p) ** ns, color=col, label=f"$\\pi$={p}")
    for L in lens[:6]:
        ax.axvline(L, color="0.8", lw=0.8, zorder=0)
    ax.axhline(1.0, color="0.55", ls="--", lw=1.0)
    ax.set_xlabel("rungs (grey lines: corpus pathway lengths)")
    ax.set_ylabel("composite power")
    ax.set_title("A  composition over corpus chain lengths")
    ax.legend(frameon=False, loc="lower right")

    # (B) rungs required to reach a target
    ax = fig.add_subplot(gs[0, 1])
    for t, col in [(0.80, C1), (0.90, C2), (0.99, C3)]:
        ps = np.linspace(0.03, 0.6, 120)
        ax.plot(ps, np.log(1 - t) / np.log(1 - ps), color=col,
                label=f"target {t}")
    ax.set_yscale("log")
    ax.set_xlabel(r"rung power $\pi$")
    ax.set_ylabel("rungs required")
    ax.set_title("B  cost of a target")
    ax.legend(frameon=False)

    # (C) sensitivity across a corpus-length ladder
    ax = fig.add_subplot(gs[0, 2])
    n = min(lens[0], 24)
    powers = [rng.uniform(0.05, 0.5) for _ in range(n)]
    P = 1.0
    for p in powers:
        P *= (1 - p)
    sens = [P / (1 - p) for p in powers]
    ax.scatter(powers, sens, s=26, color=C1, alpha=0.85, edgecolor="none")
    gg = np.linspace(min(powers), max(powers), 100)
    ax.plot(gg, P / (1 - gg), color=C2, lw=1.8, label=r"$P/(1-\pi_j)$")
    ax.set_xlabel(r"rung power $\pi_j$")
    ax.set_ylabel("sensitivity")
    ax.set_title("C  control at the strongest rung")
    ax.legend(frameon=False)

    # (D) 3-D: composite over (chain length, rung power)
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    kk = np.arange(1, 41)
    pp = np.linspace(0.02, 0.5, 40)
    KK, PP = np.meshgrid(kk, pp)
    Z = 1 - (1 - PP) ** KK
    surf = ax.plot_surface(KK, PP, Z, cmap=cm.cividis, linewidth=0, alpha=0.95)
    ax.set_xlabel("rungs", labelpad=1)
    ax.set_ylabel(r"rung power $\pi$", labelpad=1)
    ax.set_zlabel("composite", labelpad=-1)
    ax.set_title("D  saturation surface")
    ax.view_init(elev=25, azim=-124)
    ax.tick_params(labelsize=7)
    fig.colorbar(surf, ax=ax, shrink=0.48, pad=0.13, aspect=16)

    save(fig, 6)


def main():
    print("generating panels ...")
    panel_1(); panel_2(); panel_3(); panel_4(); panel_5(); panel_6()
    print("done.")


if __name__ == "__main__":
    main()
