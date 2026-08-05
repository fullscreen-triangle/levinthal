r"""
Panels for "Causal Knowledge Graphs for Proteins".

Five panels, four charts each, at least one 3-D per panel. Every number
plotted is COMPUTED HERE at draw time -- the residue coordinates of
tab:aa are the only transcribed input, and every distance, address,
weight, cut and gap is derived from them. Nothing is read back from the
results JSON, so a panel cannot drift out of agreement with the theory
it illustrates.

White background, minimal text, no conceptual diagrams, no tables.

Panel 5 plots the pre-correction contact ratio (sqrt3/d_min = 22.9)
alongside the realised one (d_max/d_min = 18.7). That requires the
superseded reading, which is computed explicitly here rather than
imported, since the manuscript no longer states it as a value. The
distinction is the subject of rem:ratio-bound.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

HERE = Path(__file__).resolve().parent
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

C_HOLD = "#1d4ed8"      # theorem holds / factoring
C_FAIL = "#b91c1c"      # control / violation / bound exceeded
C_BOUND = "#d97706"     # bounds
C_NEUTRAL = "#6b7280"
C_MARK = "#111827"
C_ALT = "#0d9488"

SQRT3 = math.sqrt(3.0)
MED = "MED"

# tab:aa -- the only transcribed input
AA = {
    "I": (1.000, 0.636, 0.000), "V": (0.967, 0.482, 0.000),
    "L": (0.922, 0.636, 0.000), "F": (0.811, 0.774, 0.000),
    "C": (0.778, 0.319, 0.150), "M": (0.711, 0.629, 0.000),
    "A": (0.700, 0.164, 0.000), "G": (0.456, 0.000, 0.000),
    "T": (0.422, 0.372, 0.100), "S": (0.411, 0.166, 0.100),
    "W": (0.400, 1.000, 0.000), "Y": (0.356, 0.836, 0.150),
    "P": (0.322, 0.399, 0.000), "H": (0.144, 0.653, 0.500),
    "E": (0.111, 0.545, 1.000), "Q": (0.111, 0.558, 0.200),
    "D": (0.111, 0.383, 1.000), "N": (0.111, 0.381, 0.200),
    "K": (0.067, 0.641, 1.000), "R": (0.000, 0.676, 1.000),
}


# ================================================================ kernel

def delta(k: int) -> float:
    return SQRT3 * 3.0 ** (-(k // 3))


def address(pt, k: int):
    lo, hi = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    out = []
    for i in range(k):
        c = i % 3
        span = (hi[c] - lo[c]) / 3.0
        d = min(2, int((pt[c] - lo[c]) / span)) if span > 0 else 0
        out.append(d)
        lo[c] = lo[c] + d * span
        hi[c] = lo[c] + span
    return tuple(out)


def pair_distances():
    return sorted(
        (math.dist(a, b), x, y)
        for (x, a), (y, b) in itertools.combinations(AA.items(), 2)
    )


def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cut_weight(W, S):
    return sum(w for (u, v), w in W.items() if (u in S) != (v in S))


def build_trie(k, b, f, med_w):
    items = list(itertools.product(range(b), repeat=k))
    W = {}
    for i, u in enumerate(items):
        for v in items[i + 1:]:
            W[(u, v)] = f(lcp(u, v))
        W[(u, MED)] = med_w(u)
    return items, W


def exhaustive_min(items, W, v):
    """min over admissible S, excluding the degenerate whole-set cut."""
    others = [x for x in items if x != v]
    n = len(items)
    best = float("inf")
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            S = frozenset((v,) + sub)
            if len(S) == n:
                continue
            best = min(best, cut_weight(W, S))
    return best


def chain_min(items, W, v, k):
    n = len(items)
    best, arg = float("inf"), None
    for d in range(k + 1):
        S = frozenset(x for x in items if x[:d] == v[:d])
        if len(S) == n:
            continue
        c = cut_weight(W, S)
        if c < best - 1e-12:
            best, arg = c, d
    return best, arg


def worst_gap(items, W, k):
    return max(chain_min(items, W, v, k)[0] - exhaustive_min(items, W, v)
               for v in items)


def finish(fig, name):
    # 3-D subplots carry their z-label outside the axes box, which
    # collides with the next chart's y-label at the default spacing.
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ================================================================ panel 1
# The encoding: coordinates, separations, resolution, cell shrinkage.

def panel1():
    fig = plt.figure(figsize=(13.6, 3.3))
    ds = pair_distances()
    d_min, d_max = ds[0][0], ds[-1][0]

    # (A) 3-D: the twenty residues in S-entropy space
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    xs = [p[0] for p in AA.values()]
    ys = [p[1] for p in AA.values()]
    zs = [p[2] for p in AA.values()]
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=34,
               edgecolor=C_MARK, linewidth=0.4, depthshade=False)
    # d_max is a segment; d_min (0.0756) is sub-pixel at this scale, so it
    # is ringed rather than drawn -- a line there would be invisible.
    a, b = AA["I"], AA["R"]
    ax.plot(*zip(a, b), color=C_ALT, lw=2.0)
    kr = np.array([AA["K"], AA["R"]]).mean(axis=0)
    ax.scatter([kr[0]], [kr[1]], [kr[2]], s=190, facecolor="none",
               edgecolor=C_FAIL, linewidth=1.6, depthshade=False)
    ax.set_xlabel(r"$S_{\rm k}$", labelpad=-8)
    ax.set_ylabel(r"$S_{\rm t}$", labelpad=-8)
    ax.set_zlabel(r"$S_{\rm e}$", labelpad=-8)
    ax.tick_params(pad=-2)
    ax.set_title("Residues in $\\mathcal{S}$")
    ax.view_init(22, 38)

    # (B) sorted pairwise separations
    ax = fig.add_subplot(1, 4, 2)
    vals = [d for d, _, _ in ds]
    ax.plot(range(1, len(vals) + 1), vals, color=C_HOLD)
    ax.axhline(d_min, color=C_FAIL, lw=1.0, ls="--")
    ax.axhline(SQRT3, color=C_NEUTRAL, lw=1.0, ls=":")
    ax.scatter([1], [d_min], color=C_FAIL, s=26, zorder=5)
    ax.scatter([len(vals)], [d_max], color=C_ALT, s=26, zorder=5)
    ax.set_xlabel("pair rank")
    ax.set_ylabel("separation")
    ax.set_ylim(0, SQRT3 * 1.06)
    ax.set_title("Pairwise separations")

    # (C) resolution: cell diagonal against d_min
    ax = fig.add_subplot(1, 4, 3)
    ks = list(range(1, 13))
    ax.semilogy(ks, [delta(k) for k in ks], color=C_HOLD, marker="o", ms=3)
    ax.axhline(d_min, color=C_FAIL, lw=1.0, ls="--")
    k_guar = next(k for k in ks if delta(k) < d_min)
    ax.axvline(k_guar, color=C_BOUND, lw=1.0, ls="-.")
    ax.set_xlabel("depth $k$")
    ax.set_ylabel(r"$\delta(k)$")
    ax.set_xticks(ks[1::2])
    ax.set_title("Resolution limit")

    # (D) addresses actually distinct, by depth
    ax = fig.add_subplot(1, 4, 4)
    n_dist = [len({address(p, k) for p in AA.values()}) for k in ks]
    ax.plot(ks, n_dist, color=C_HOLD, marker="o", ms=3)
    ax.axhline(len(AA), color=C_NEUTRAL, lw=1.0, ls=":")
    k_act = next(k for k, n in zip(ks, n_dist) if n == len(AA))
    ax.axvline(k_act, color=C_ALT, lw=1.0, ls="-.")
    ax.axvline(k_guar, color=C_BOUND, lw=1.0, ls="-.")
    ax.fill_betweenx([0, len(AA) + 1], k_act, k_guar,
                     color=C_BOUND, alpha=0.12, lw=0)
    ax.set_xlabel("depth $k$")
    ax.set_ylabel("distinct addresses")
    ax.set_ylim(0, len(AA) + 1)
    ax.set_xticks(ks[1::2])
    ax.set_title("Actual vs guaranteed")

    finish(fig, "panel_01_encoding.png")


# ================================================================ panel 2
# The derived weighting: reciprocal profile, inversion, Lambda-cancellation.

def panel2():
    fig = plt.figure(figsize=(13.6, 3.3))
    ds = pair_distances()
    d_min, d_max = ds[0][0], ds[-1][0]
    dists = np.array([d for d, _, _ in ds])

    # (A) the derived profile w = Lambda/||.||
    ax = fig.add_subplot(1, 4, 1)
    s = np.linspace(d_min * 0.85, SQRT3, 400)
    ax.plot(s, 1.0 / s, color=C_HOLD)
    ax.scatter(dists, 1.0 / dists, s=9, color=C_MARK, alpha=0.5, zorder=4)
    ax.axvline(d_min, color=C_FAIL, lw=1.0, ls="--")
    ax.axvline(d_max, color=C_ALT, lw=1.0, ls="--")
    ax.set_xlabel("S-entropy separation")
    ax.set_ylabel(r"$w\;/\;\Lambda$")
    ax.set_title("Derived weighting")

    # (B) the inversion, MEASURED: rank each pair independently by
    # distance and by w = Lambda/d, then plot one against the other.
    # Hardcoding the reversal would draw the same line for a wrong
    # weighting, so the ranks are computed separately and joined by pair.
    ax = fig.add_subplot(1, 4, 2)
    n = len(dists)
    pairs = [(d, x, y) for d, x, y in ds]
    r_dist = {(x, y): i for i, (_, x, y) in
              enumerate(sorted(pairs, key=lambda t: t[0]), 1)}
    # both rankings ASCENDING in their own quantity -- ranking weight
    # descending would reproduce the distance order and hide the very
    # inversion this chart exists to show.
    r_wt = {(x, y): i for i, (_, x, y) in
            enumerate(sorted(pairs, key=lambda t: 1.0 / t[0]), 1)}
    X = [r_dist[(x, y)] for _, x, y in pairs]
    Y = [r_wt[(x, y)] for _, x, y in pairs]
    ax.scatter(X, Y, s=9, color=C_HOLD, alpha=0.75)
    tau = np.corrcoef(X, Y)[0, 1]
    ax.scatter([r_dist[("K", "R")]], [r_wt[("K", "R")]],
               color=C_FAIL, s=30, zorder=5)
    ax.scatter([r_dist[("I", "R")]], [r_wt[("I", "R")]],
               color=C_ALT, s=30, zorder=5)
    ax.set_xlabel("rank by distance")
    ax.set_ylabel("rank by weight")
    ax.set_title(f"Cost inverts distance ($r={tau:.0f}$)")

    # (C) 3-D: weight surface over a coordinate slice
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    g = np.linspace(0.02, 1.0, 60)
    X, Y = np.meshgrid(g, g)
    ref = np.array(AA["K"])
    Z = 1.0 / np.sqrt((X - ref[0]) ** 2 + (Y - ref[1]) ** 2 + 1e-3)
    ax.plot_surface(X, Y, np.minimum(Z, 20), cmap="magma",
                    linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel(r"$S_{\rm k}$", labelpad=-8)
    ax.set_ylabel(r"$S_{\rm t}$", labelpad=-8)
    ax.set_zlabel(r"$w/\Lambda$", labelpad=-8)
    ax.tick_params(pad=-2)
    ax.set_title("Cost surface about Lys")
    ax.view_init(26, 132)

    # (D) Lambda cancels. Raw weights sweep nine decades with Lambda;
    # the ratio does not move. Both are drawn on one log axis so the
    # cancellation is visible rather than asserted by a flat line alone.
    ax = fig.add_subplot(1, 4, 4)
    lams = np.logspace(-3, 6, 40)
    ax.loglog(lams, [lam / d_min for lam in lams], color=C_NEUTRAL,
              lw=1.4, ls="--")
    ax.loglog(lams, [lam / d_max for lam in lams], color=C_NEUTRAL,
              lw=1.4, ls=":")
    ax.loglog(lams, [(lam / d_min) / (lam / d_max) for lam in lams],
              color=C_HOLD, lw=2.0)
    ax.axhline(SQRT3 / d_min, color=C_BOUND, lw=1.0, ls="--")
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$w/\Lambda$   and   max/min")
    ax.set_title("Scale cancels")

    finish(fig, "panel_02_weighting.png")


# ================================================================ panel 3
# thm:subtrie-cut: the gap, the depth profile, the control.

def panel3():
    fig = plt.figure(figsize=(13.6, 3.3))
    rng = random.Random(4242)
    k, b = 3, 2
    n = b ** k
    scale = 5.0 * n / 4.0

    def med_factory(lo, hi):
        mw = {}

        def m(u):
            if u not in mw:
                mw[u] = rng.uniform(lo, hi) * scale
            return mw[u]
        return m

    profs = {
        "$3^{\\ell}$": lambda l: 3.0 ** l,
        "$1+3\\ell$": lambda l: 1.0 + 3.0 * l,
        "$3^{-\\ell}$": lambda l: 3.0 ** (-l),
        "$[1,9,1,4]$": lambda l: [1.0, 9.0, 1.0, 4.0][min(l, 3)],
        "const": lambda l: 1.0,
    }

    # (A) the chain: cut weight against block depth
    ax = fig.add_subplot(1, 4, 1)
    for (nm, f), col in zip(profs.items(),
                            [C_HOLD, C_ALT, C_FAIL, C_BOUND, C_NEUTRAL]):
        items, W = build_trie(k, b, f, med_factory(0.05, 0.4))
        v = items[0]
        ys = []
        for d in range(k + 1):
            S = frozenset(x for x in items if x[:d] == v[:d])
            ys.append(np.nan if len(S) == len(items) else cut_weight(W, S))
        ax.plot(range(k + 1), ys, marker="o", ms=3, color=col, label=nm)
    ax.set_xlabel("block depth $d$")
    ax.set_ylabel("cut weight")
    ax.set_xticks(range(k + 1))
    ax.set_title("Chain of nested cuts")
    ax.legend(frameon=False, loc="upper center", ncol=2)

    # (B) gap: factoring vs control, over trials
    ax = fig.add_subplot(1, 4, 2)
    fac, ctl = [], []
    for _ in range(28):
        f = lambda l: 3.0 ** l  # noqa: E731
        items, W = build_trie(k, b, f, med_factory(0.05, 0.4))
        fac.append(worst_gap(items, W, k))
        items2 = list(itertools.product(range(b), repeat=k))
        W2 = {}
        for i, u in enumerate(items2):
            for v2 in items2[i + 1:]:
                W2[(u, v2)] = rng.uniform(0.1, 10.0)
            W2[(u, MED)] = rng.uniform(0.05, 0.4) * scale
        ctl.append(worst_gap(items2, W2, k))
    ax.plot(fac, color=C_HOLD, marker="o", ms=3, label="factoring")
    ax.plot(ctl, color=C_FAIL, marker="s", ms=3, label="control")
    ax.axhline(0, color=C_NEUTRAL, lw=0.8, ls=":")
    ax.set_xlabel("trial")
    ax.set_ylabel("worst gap")
    ax.set_title("Gap vanishes iff factoring")
    ax.legend(frameon=False)

    # (C) 3-D: gap surface over medium coupling x profile
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    meds = np.linspace(0.02, 1.6, 9)
    pnames = list(profs)
    Zc = np.zeros((len(pnames), len(meds)))
    Zf = np.zeros((len(pnames), len(meds)))
    for i, nm in enumerate(pnames):
        for j, mv in enumerate(meds):
            items, W = build_trie(k, b, profs[nm],
                                  med_factory(mv * 0.6, mv * 1.4))
            Zf[i, j] = worst_gap(items, W, k)
            items2 = list(itertools.product(range(b), repeat=k))
            W2 = {}
            for a_, u in enumerate(items2):
                for v2 in items2[a_ + 1:]:
                    W2[(u, v2)] = rng.uniform(0.1, 10.0)
                W2[(u, MED)] = rng.uniform(mv * 0.6, mv * 1.4) * scale
            Zc[i, j] = worst_gap(items2, W2, k)
    Xg, Yg = np.meshgrid(meds, np.arange(len(pnames)))
    ax.plot_surface(Xg, Yg, Zc, color=C_FAIL, alpha=0.55, linewidth=0)
    ax.plot_surface(Xg, Yg, Zf, color=C_HOLD, alpha=0.95, linewidth=0)
    ax.set_xlabel("medium", labelpad=-8)
    ax.set_ylabel("profile", labelpad=-8)
    ax.set_zlabel("gap", labelpad=-8)
    ax.set_yticks(range(len(pnames)))
    ax.set_yticklabels([""] * len(pnames))
    ax.tick_params(pad=-2)
    ax.set_title("Gap surface")
    ax.view_init(24, -128)

    # (D) where the optimum sits
    ax = fig.add_subplot(1, 4, 4)
    counts = np.zeros(k + 1)
    for nm, f in profs.items():
        for lo, hi in [(0.005, 0.05), (0.05, 0.4), (0.2, 1.5),
                       (0.8, 4.0), (3.0, 20.0)]:
            items, W = build_trie(k, b, f, med_factory(lo, hi))
            for v in items:
                counts[chain_min(items, W, v, k)[1]] += 1
    cols = [C_NEUTRAL if d in (0, k) else C_HOLD for d in range(k + 1)]
    ax.bar(range(k + 1), counts, color=cols, width=0.62)
    ax.set_xlabel("optimal depth $d^{*}$")
    ax.set_ylabel("items")
    ax.set_xticks(range(k + 1))
    ax.set_title("Interior optima")

    finish(fig, "panel_03_subtrie_cut.png")


# ================================================================ panel 4
# Degradation off-hypothesis.

def panel4():
    fig = plt.figure(figsize=(13.6, 3.3))
    rng = random.Random(99)
    k, b = 3, 2
    n = b ** k
    scale = 5.0 * n / 4.0
    f = lambda l: 3.0 ** l  # noqa: E731

    def instance(m, lo, hi):
        mw = {}

        def med(u):
            if u not in mw:
                mw[u] = rng.uniform(0.05, 0.4) * scale
            return mw[u]
        items, W = build_trie(k, b, f, med)
        pairs = [(u, v) for i, u in enumerate(items) for v in items[i + 1:]]
        P = 0.0
        for p in rng.sample(pairs, m) if m else []:
            dw = rng.uniform(lo, hi)
            W[p] = W[p] + dw
            P += dw
        gaps = [chain_min(items, W, v, k)[0] - exhaustive_min(items, W, v)
                for v in items]
        return P, max(gaps), sum(1 for g in gaps if g <= 1e-9) / len(gaps)

    mags = [(0.5, 6.0), (5.0, 40.0), (30.0, 200.0)]
    ms = [1, 2, 3, 5]

    # (A) gap against the bound P
    ax = fig.add_subplot(1, 4, 1)
    Ps, Gs = [], []
    for lo, hi in mags:
        for m in ms:
            for _ in range(7):
                P, g, _ = instance(m, lo, hi)
                Ps.append(P)
                Gs.append(g)
    hi_p = max(Ps) * 1.05
    ax.plot([0, hi_p], [0, hi_p], color=C_FAIL, lw=1.2, ls="--")
    ax.scatter(Ps, Gs, s=13, color=C_HOLD, alpha=0.75)
    ax.set_xlabel("non-factoring weight $P$")
    ax.set_ylabel("worst gap")
    ax.set_xlim(0, hi_p)
    ax.set_ylim(0, hi_p)
    ax.set_title("Gap stays under bound")

    # (B) exactness retained
    ax = fig.add_subplot(1, 4, 2)
    for (lo, hi), col in zip(mags, [C_HOLD, C_BOUND, C_FAIL]):
        ys = [np.mean([instance(m, lo, hi)[2] for _ in range(7)]) for m in ms]
        ax.plot(ms, ys, marker="o", ms=3.5, color=col)
    ax.set_xlabel("non-factoring edges $m$")
    ax.set_ylabel("fraction exact")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(ms)
    ax.set_title("Graceful degradation")

    # (C) 3-D: exactness over (m, magnitude). Six magnitudes and 14
    # replicates -- a coarser grid was dominated by sampling noise and
    # showed a spurious upturn at the far corner, contradicting (B).
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    smags = [(0.5, 3.0), (2.0, 10.0), (6.0, 30.0),
             (18.0, 80.0), (40.0, 160.0), (90.0, 320.0)]
    mag_mid = [np.sqrt(lo * hi) for lo, hi in smags]
    Z = np.zeros((len(smags), len(ms)))
    for i, (lo, hi) in enumerate(smags):
        for j, m in enumerate(ms):
            Z[i, j] = np.mean([instance(m, lo, hi)[2] for _ in range(14)])
    Xg, Yg = np.meshgrid(ms, np.log10(mag_mid))
    ax.plot_surface(Xg, Yg, Z, cmap="coolwarm_r", linewidth=0.2,
                    edgecolor="white", alpha=0.96)
    ax.set_xlabel("$m$", labelpad=-8)
    ax.set_ylabel(r"$\log_{10}P$", labelpad=-8)
    ax.set_zlabel("exact", labelpad=-8)
    ax.set_zlim(0, 1)
    ax.tick_params(pad=-2)
    ax.set_title("Exactness surface")
    ax.view_init(22, 42)

    # (D) cut evaluations: chain vs exhaustive
    ax = fig.add_subplot(1, 4, 4)
    sizes = list(range(3, 15))
    ax.semilogy(sizes, [2 ** (s - 1) - 1 for s in sizes],
                color=C_FAIL, marker="s", ms=3)
    ax.semilogy(sizes, [max(1, int(math.log(s, 2)) + 1) for s in sizes],
                color=C_HOLD, marker="o", ms=3)
    ax.set_xlabel("items $|V|$")
    ax.set_ylabel("cut evaluations")
    ax.set_xticks(sizes[::2])
    ax.set_title("Chain vs exhaustive")

    finish(fig, "panel_04_degradation.png")


# ================================================================ panel 5
# The corrected ratio, and what the alphabet occupies.

def panel5():
    fig = plt.figure(figsize=(13.6, 3.3))
    ds = pair_distances()
    d_min, d_max = ds[0][0], ds[-1][0]
    realised, bound = d_max / d_min, SQRT3 / d_min

    # (A) 3-D: convex hull extent against the unit cube
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    P = np.array(list(AA.values()))
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=20, color=C_HOLD,
               edgecolor=C_MARK, linewidth=0.3, depthshade=False)
    for s_, e_ in itertools.combinations(np.array(list(itertools.product(
            [0, 1], repeat=3)), dtype=float), 2):
        if np.sum(np.abs(s_ - e_)) == 1:
            ax.plot(*zip(s_, e_), color=C_NEUTRAL, lw=0.5, alpha=0.5)
    ax.plot(*zip((0., 0., 0.), (1., 1., 1.)), color=C_BOUND, lw=1.6,
            ls="--", zorder=1)
    a, b_ = AA["I"], AA["R"]
    ax.plot(*zip(a, b_), color=C_ALT, lw=2.4, zorder=6)
    ax.set_xlabel(r"$S_{\rm k}$", labelpad=-8)
    ax.set_ylabel(r"$S_{\rm t}$", labelpad=-8)
    ax.set_zlabel(r"$S_{\rm e}$", labelpad=-8)
    ax.tick_params(pad=-2)
    ax.set_title("Realised vs cube diameter")
    ax.view_init(20, 44)

    # (B) how the bound and the realised value separate as the alphabet
    # grows: random sub-alphabets of size s, realised ratio against the
    # bound the cube diameter would give for the same d_min.
    ax = fig.add_subplot(1, 4, 2)
    rr = random.Random(11)
    keys = list(AA)
    sizes = list(range(3, len(keys) + 1))
    med_r, med_b = [], []
    for s in sizes:
        rs, bs = [], []
        for _ in range(60):
            sub = rr.sample(keys, s)
            dd = [math.dist(AA[x], AA[y])
                  for x, y in itertools.combinations(sub, 2)]
            rs.append(max(dd) / min(dd))
            bs.append(SQRT3 / min(dd))
        med_r.append(np.median(rs))
        med_b.append(np.median(bs))
    ax.plot(sizes, med_b, color=C_BOUND, lw=1.8, ls="--")
    ax.plot(sizes, med_r, color=C_HOLD, lw=1.8)
    ax.fill_between(sizes, med_r, med_b, color=C_BOUND, alpha=0.13, lw=0)
    ax.scatter([len(keys)], [realised], color=C_HOLD, s=30, zorder=5)
    ax.scatter([len(keys)], [bound], color=C_BOUND, s=30, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("alphabet size")
    ax.set_ylabel("max/min contact")
    ax.set_title("Value below bound")

    # (C) occupancy: distance distribution against the cube's
    ax = fig.add_subplot(1, 4, 3)
    vals = np.array([d for d, _, _ in ds])
    rng = np.random.default_rng(7)
    unif = rng.random((4000, 3))
    ref = rng.random((4000, 3))
    cube_d = np.linalg.norm(unif - ref, axis=1)
    bins = np.linspace(0, SQRT3, 34)
    ax.hist(cube_d, bins=bins, density=True, color=C_NEUTRAL,
            alpha=0.45, lw=0)
    ax.hist(vals, bins=bins, density=True, color=C_HOLD, alpha=0.8, lw=0)
    ax.axvline(d_max, color=C_ALT, lw=1.2, ls="--")
    ax.axvline(SQRT3, color=C_BOUND, lw=1.2, ls="--")
    ax.set_xlabel("separation")
    ax.set_ylabel("density")
    ax.set_title("Alphabet occupancy")

    # (D) sensitivity: ratio if the extreme pair were removed
    ax = fig.add_subplot(1, 4, 4)
    labels, ratios = [], []
    for drop in [None, "I", "R", "K", "W"]:
        sub = {k_: v for k_, v in AA.items() if k_ != drop}
        dd = [math.dist(a2, b2)
              for (_, a2), (_, b2) in itertools.combinations(sub.items(), 2)]
        labels.append("all" if drop is None else f"-{drop}")
        ratios.append(max(dd) / min(dd))
    ax.plot(range(len(ratios)), ratios, marker="o", ms=4, color=C_HOLD)
    ax.axhline(bound, color=C_BOUND, lw=1.0, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("max/min contact")
    ax.set_ylim(0, bound * 1.2)
    ax.set_title("Ratio sensitivity")

    finish(fig, "panel_05_ratio.png")


def main():
    print("generating panels...")
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print("done.")


if __name__ == "__main__":
    main()
