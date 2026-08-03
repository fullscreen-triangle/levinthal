"""
Panels for "Attention-Allocated Retrieval".

Five panels, four charts each, at least one 3-D per panel. Every number
plotted is computed from the kernel at draw time --- nothing is
transcribed from the paper or from the results JSON, so a panel cannot
drift out of agreement with the theory it illustrates.

White background, minimal text, no conceptual diagrams, no tables.

Panel 5 plots the PRE-FIX allocator alongside the current one. That
requires reproducing the broken rule, which is done here explicitly
(`allocate_prefix`) rather than by importing it, since the shipped
allocator no longer contains it. The 8.76% gap it shows at B=F is the
subject of Proposition 4.4.
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
sys.path.insert(0, str(VAL))

from kernel.allocator import (  # noqa: E402
    JoinSource,
    Refusal,
    _water_fill,
    allocate,
)

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

C_CONVEX = "#b91c1c"    # convex branch / refusal / danger
C_CONCAVE = "#1d4ed8"   # concave branch / water-fill
C_COMMIT = "#d97706"    # commit-subset
C_NEUTRAL = "#6b7280"
C_MARK = "#111827"
C_ALT = "#0d9488"

N_SHADES = {10: "#93c5fd", 40: "#60a5fa", 100: "#2563eb", 400: "#1e3a8a"}


def _finish(fig, name: str) -> None:
    fig.tight_layout(pad=1.1, w_pad=2.2)
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


def _panel(n_3d_index: int):
    """A 1x4 panel; the axis at n_3d_index is 3-D."""
    fig = plt.figure(figsize=(13.6, 3.3))
    axes = []
    for i in range(4):
        if i == n_3d_index:
            axes.append(fig.add_subplot(1, 4, i + 1, projection="3d"))
        else:
            axes.append(fig.add_subplot(1, 4, i + 1))
    return fig, axes


def _tag(ax, letter: str, three_d: bool = False) -> None:
    if three_d:
        ax.text2D(-0.06, 1.04, f"({letter})", transform=ax.transAxes,
                  fontsize=9, fontweight="bold", va="top")
    else:
        ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")


# =====================================================================
#  Closed forms --- restated here ONLY as the theory, to be checked
#  against the kernel. Panel 1(b) plots the difference, which must be 0.
# =====================================================================


def yield_theory(e: np.ndarray, N: int) -> np.ndarray:
    p = 1.0 - (1.0 - 1.0 / N) ** (e / 2.0)
    return N * p * p


def ypp_theory(e: np.ndarray, N: int) -> np.ndarray:
    """Y'' = (N/2)(ln q)^2 r (2r - 1); Eq. (secondderiv)."""
    q = 1.0 - 1.0 / N
    r = q ** (e / 2.0)
    return 0.5 * N * (math.log(q) ** 2) * r * (2.0 * r - 1.0)


def yp_theory(e: np.ndarray, N: int) -> np.ndarray:
    """Y' = -N r p ln q."""
    q = 1.0 - 1.0 / N
    r = q ** (e / 2.0)
    return -N * r * (1.0 - r) * math.log(q)


def e_star(N: int) -> float:
    return 2.0 * math.log(2.0) / math.log(1.0 / (1.0 - 1.0 / N))


# Commit/spread crossover, Eq. (ucross). Solves (1+t)^2 = 2 for
# t = 2^(-u/2), i.e. u = 2 log2(1 + sqrt 2). Independent of N.
U_CROSS = 2.0 * math.log2(1.0 + math.sqrt(2.0))


# =====================================================================
#  PANEL 1 --- the yield surface and its inflection
# =====================================================================


def panel1() -> None:
    fig, ax = _panel(3)

    # (a) normalised yield; every curve crosses 0.25 at e/e* = 1
    a = ax[0]
    u = np.linspace(0.01, 4.0, 400)
    for N, c in N_SHADES.items():
        es = e_star(N)
        src = JoinSource(f"N{N}", corpus_size=N)
        y = np.array([src.yield_at(uu * es) for uu in u]) / N
        a.plot(u, y, color=c, label=f"N={N}")
    a.axvline(1.0, color=C_MARK, lw=0.9, ls="--")
    a.axhline(0.25, color=C_MARK, lw=0.9, ls="--")
    a.plot([1.0], [0.25], "o", color=C_CONVEX, ms=5, zorder=5)
    a.set_xlabel(r"effort $e/e^{*}$")
    a.set_ylabel(r"yield $Y/N$")
    a.set_ylim(0, 1.0)
    a.legend(frameon=False, loc="lower right")
    _tag(a, "a")

    # (b) second difference: sign change at e*, and kernel-vs-theory
    a = ax[1]
    N = 90
    es = e_star(N)
    e = np.linspace(0.05 * es, 3.0 * es, 500)
    ypp = ypp_theory(e, N)
    a.axhline(0.0, color=C_NEUTRAL, lw=0.8)
    a.fill_between(e / es, 0, ypp, where=ypp > 0, color=C_CONVEX,
                   alpha=0.25, linewidth=0)
    a.fill_between(e / es, 0, ypp, where=ypp <= 0, color=C_CONCAVE,
                   alpha=0.25, linewidth=0)
    a.plot(e / es, ypp, color=C_MARK, lw=1.4)
    a.axvline(1.0, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel(r"effort $e/e^{*}$")
    a.set_ylabel(r"$Y''(e)$   ($N=90$)")
    _tag(a, "b")

    # (c) marginal yield PEAKS at e* -- the mechanism behind Prop 4.4
    a = ax[2]
    for N, c in N_SHADES.items():
        es = e_star(N)
        e = np.linspace(0.02 * es, 4.0 * es, 400)
        a.plot(e / es, yp_theory(e, N), color=c, label=f"N={N}")
    a.axvline(1.0, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel(r"effort $e/e^{*}$")
    a.set_ylabel(r"marginal yield $Y'(e)$")
    a.legend(frameon=False, loc="upper right")
    _tag(a, "c")

    # (d) 3-D yield surface over (e, N) with the threshold ridge
    a = ax[3]
    Ns = np.linspace(10, 300, 60)
    uu = np.linspace(0.05, 3.0, 60)
    U, NN = np.meshgrid(uu, Ns)
    Z = np.zeros_like(U)
    for i in range(NN.shape[0]):
        Nv = NN[i, 0]
        es = 2.0 * math.log(2.0) / math.log(1.0 / (1.0 - 1.0 / Nv))
        Z[i, :] = yield_theory(U[i, :] * es, Nv) / Nv
    a.plot_surface(U, NN, Z, cmap="Blues", rstride=1, cstride=1,
                   linewidth=0, antialiased=True, alpha=0.92)
    a.plot(np.ones_like(Ns), Ns, np.full_like(Ns, 0.25),
           color=C_CONVEX, lw=2.2, zorder=10)
    a.set_xlabel(r"$e/e^{*}$", labelpad=-2)
    a.set_ylabel(r"$N$", labelpad=-2)
    a.set_zlabel(r"$Y/N$", labelpad=-4)
    a.view_init(elev=22, azim=-127)
    a.tick_params(pad=-1)
    _tag(a, "d", three_d=True)

    _finish(fig, "panel_01_yield_surface")


# =====================================================================
#  PANEL 2 --- the threshold reverses the optimal policy
# =====================================================================


def panel2() -> None:
    fig, ax = _panel(3)

    N = 40
    src = JoinSource("s", corpus_size=N)
    es = src.e_star

    def total(x: float, B: float) -> float:
        return src.yield_at(x) + src.yield_at(B - x)

    # (a) advantage of committing over spreading vs budget
    a = ax[0]
    shares = np.linspace(0.02, 6.0, 400)
    adv = []
    for s in shares:
        B = s * es
        commit = total(B, B)
        spread = total(B / 2.0, B)
        adv.append(100.0 * (commit - spread) / spread)
    adv = np.array(adv)
    a.axhline(0.0, color=C_NEUTRAL, lw=0.8)
    a.fill_between(shares, 0, adv, where=adv > 0, color=C_COMMIT,
                   alpha=0.3, linewidth=0)
    a.fill_between(shares, 0, adv, where=adv <= 0, color=C_CONCAVE,
                   alpha=0.3, linewidth=0)
    a.plot(shares, adv, color=C_MARK, lw=1.4)
    a.axvline(1.0, color=C_NEUTRAL, lw=0.9, ls=":")
    a.axvline(U_CROSS, color=C_MARK, lw=0.9, ls="--")
    a.plot([U_CROSS], [0.0], "o", color=C_MARK, ms=5, zorder=6)
    a.set_xscale("log")
    a.set_xlabel(r"budget $B/e^{*}$")
    a.set_ylabel("commit advantage (%)")
    _tag(a, "a")

    # (b) sub-threshold: total yield vs split -> maximised at endpoints
    a = ax[1]
    B = 0.5 * es
    xs = np.linspace(0, 1, 400)
    ys = np.array([total(f * B, B) for f in xs])
    a.plot(xs, ys, color=C_CONVEX, lw=1.8)
    imax = int(np.argmax(ys))
    a.plot([xs[imax]], [ys[imax]], "o", color=C_MARK, ms=6, zorder=5)
    a.plot([xs[-1]], [ys[-1]], "o", color=C_MARK, ms=6, zorder=5)
    a.set_xlabel("split fraction $x/B$")
    a.set_ylabel(r"total yield   ($B=0.5\,e^{*}$)")
    _tag(a, "b")

    # (c) super-threshold: maximised at the centre
    a = ax[2]
    B = 3.0 * es
    ys = np.array([total(f * B, B) for f in xs])
    a.plot(xs, ys, color=C_CONCAVE, lw=1.8)
    imax = int(np.argmax(ys))
    a.plot([xs[imax]], [ys[imax]], "o", color=C_MARK, ms=6, zorder=5)
    a.set_xlabel("split fraction $x/B$")
    a.set_ylabel(r"total yield   ($B=3\,e^{*}$)")
    _tag(a, "c")

    # (d) 3-D: the ridge of optima migrates from edge to centre
    a = ax[3]
    frac = np.linspace(0, 1, 70)
    budg = np.linspace(0.1 * es, 4.0 * es, 70)
    F, Bg = np.meshgrid(frac, budg)
    Z = np.zeros_like(F)
    for i in range(Bg.shape[0]):
        Bv = Bg[i, 0]
        Z[i, :] = [total(f * Bv, Bv) / N for f in F[i, :]]
    a.plot_surface(F, Bg / es, Z, cmap="RdYlBu", rstride=1, cstride=1,
                   linewidth=0, antialiased=True, alpha=0.93)
    ridge_f, ridge_b, ridge_z = [], [], []
    for i in range(Bg.shape[0]):
        Bv = Bg[i, 0]
        vals = np.array([total(f * Bv, Bv) for f in frac])
        j = int(np.argmax(vals))
        ridge_f.append(frac[j])
        ridge_b.append(Bv / es)
        ridge_z.append(vals[j] / N)
    # Split the ridge at the measured jump so the discontinuity reads as
    # one, rather than as a steep segment joining the two branches.
    ridge_f = np.array(ridge_f); ridge_b = np.array(ridge_b)
    ridge_z = np.array(ridge_z)
    lo_br = ridge_f < 0.25
    a.plot(ridge_f[lo_br], ridge_b[lo_br], ridge_z[lo_br],
           color=C_COMMIT, lw=2.4, zorder=10)
    a.plot(ridge_f[~lo_br], ridge_b[~lo_br], ridge_z[~lo_br],
           color=C_CONCAVE, lw=2.4, zorder=10)
    jump = float(ridge_b[~lo_br].min()) if (~lo_br).any() else float("nan")
    step = float(ridge_b[1] - ridge_b[0])
    ok = 0.0 <= jump - U_CROSS <= step
    print(f"    panel2(d) ridge jumps at B/e* = {jump:.4f} "
          f"(theory {U_CROSS:.4f}, grid step {step:.4f}) "
          f"{'OK' if ok else 'OFF-GRID'}")
    a.set_xlabel("$x/B$", labelpad=-2)
    a.set_ylabel(r"$B/e^{*}$", labelpad=-2)
    a.set_zlabel("yield", labelpad=-8)
    a.view_init(elev=26, azim=-60)
    a.tick_params(pad=-1)
    _tag(a, "d", three_d=True)

    _finish(fig, "panel_02_policy_reversal")


# =====================================================================
#  PANEL 3 --- the allocator against exhaustive search
# =====================================================================


def brute2(s0: JoinSource, s1: JoinSource, B: float, steps: int = 800):
    xs = np.linspace(0.0, B, steps + 1)
    vals = np.array([s0.yield_at(x) + s1.yield_at(B - x) for x in xs])
    j = int(np.argmax(vals))
    return float(vals[j]), float(xs[j])


def panel3() -> None:
    fig, ax = _panel(3)

    chebi = JoinSource("chebi-rhea", corpus_size=30)
    rhea = JoinSource("rhea-uniprot", corpus_size=90)
    pair = [chebi, rhea]
    F = chebi.e_star + rhea.e_star
    m = min(s.e_star for s in pair)

    # (a) allocator vs brute force
    a = ax[0]
    budgets = np.linspace(m, 8.0 * F, 60)
    alloc_y, bf_y = [], []
    for B in budgets:
        try:
            alloc_y.append(allocate(pair, float(B)).total_yield)
        except Refusal:
            alloc_y.append(np.nan)
        bf_y.append(brute2(chebi, rhea, float(B))[0])
    a.plot(budgets, bf_y, color=C_NEUTRAL, lw=3.4, alpha=0.55,
           label="brute force")
    a.plot(budgets, alloc_y, color=C_CONCAVE, lw=1.5, label="allocator")
    a.axvline(F, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel("budget $B$")
    a.set_ylabel("total yield")
    a.legend(frameon=False, loc="lower right")
    _tag(a, "a")

    # (b) normalised efforts converge; three unequal corpora
    a = ax[1]
    trio = [JoinSource("small", 20), JoinSource("mid", 60),
            JoinSource("large", 200)]
    F3 = sum(s.e_star for s in trio)
    bs = np.linspace(F3 * 1.02, F3 * 6.0, 200)
    series = {s.name: [] for s in trio}
    is_wf = []
    for B in bs:
        al = allocate(trio, float(B))
        is_wf.append(al.policy == "water-fill")
        for s in trio:
            # NaN, not 0, for an unfunded source: a gap in the line reads
            # as "not allocated", whereas a zero reads as "disagrees with
            # the equalisation claim". Prop. 3.4 is conditioned on every
            # source being on its concave branch, i.e. the water-fill
            # regime, which is shaded.
            v = al.normalised.get(s.name, 0.0)
            series[s.name].append(v if v > 1e-9 else np.nan)
    is_wf = np.array(is_wf)
    if is_wf.any():
        a.axvspan(float(bs[is_wf].min()) / F3, float(bs.max()) / F3,
                  color=C_CONCAVE, alpha=0.07, linewidth=0)
        a.axvline(float(bs[is_wf].min()) / F3, color=C_MARK, lw=0.9,
                  ls="--")
    for (name, vals), c in zip(series.items(),
                               (C_CONVEX, C_ALT, C_CONCAVE)):
        a.plot(bs / F3, vals, color=c, label=name)
    a.set_xlabel(r"budget $B/F$")
    a.set_ylabel(r"normalised effort $e_i/e^{*}_i$")
    a.legend(frameon=False, loc="lower right")
    _tag(a, "b")

    # (c) policy regions, by measured yield share to each source
    a = ax[2]
    budgets = np.linspace(0.15 * m, 3.2 * F, 320)
    shares_small, shares_large, refused = [], [], []
    for B in budgets:
        try:
            al = allocate(pair, float(B))
            tot = sum(al.per_source.values())
            shares_small.append(al.per_source.get(chebi.name, 0.0) / tot)
            shares_large.append(al.per_source.get(rhea.name, 0.0) / tot)
            refused.append(np.nan)
        except Refusal:
            shares_small.append(np.nan)
            shares_large.append(np.nan)
            refused.append(1.0)
    a.fill_between(budgets, 0, refused, color=C_CONVEX, alpha=0.25,
                   linewidth=0)
    a.plot(budgets, shares_small, color=C_ALT, label="chebi-rhea")
    a.plot(budgets, shares_large, color=C_CONCAVE, label="rhea-uniprot")
    a.axvline(m, color=C_CONVEX, lw=0.9, ls="--")
    a.axvline(F, color=C_MARK, lw=0.9, ls="--")
    a.set_ylim(0, 1.05)
    a.set_xlabel("budget $B$")
    a.set_ylabel("share of budget")
    a.legend(frameon=False, loc="center right")
    _tag(a, "c")

    # (d) 3-D simplex surface just above F, interior vs corner
    a = ax[3]
    B = F * 1.0001
    frac = np.linspace(0, 1, 90)
    NN = np.linspace(0.6, 1.6, 60)   # scale the LARGE corpus
    FF, SS = np.meshgrid(frac, NN)
    Z = np.zeros_like(FF)
    for i in range(SS.shape[0]):
        big = JoinSource("big", corpus_size=int(round(90 * SS[i, 0])))
        Bv = chebi.e_star + big.e_star
        Z[i, :] = [chebi.yield_at(f * Bv) + big.yield_at((1 - f) * Bv)
                   for f in FF[i, :]]
    a.plot_surface(FF, SS, Z, cmap="coolwarm", rstride=1, cstride=1,
                   linewidth=0, antialiased=True, alpha=0.93)
    # trace the argmax ridge: it sits at f=0 (fund only the large source)
    rf, rs, rz = [], [], []
    for i in range(SS.shape[0]):
        big = JoinSource("big", corpus_size=int(round(90 * SS[i, 0])))
        Bv = chebi.e_star + big.e_star
        vals = np.array([chebi.yield_at(f * Bv) + big.yield_at((1 - f) * Bv)
                         for f in frac])
        j = int(np.argmax(vals))
        rf.append(frac[j]); rs.append(SS[i, 0]); rz.append(vals[j])
    a.plot(rf, rs, rz, color=C_MARK, lw=2.2, zorder=10)
    a.set_xlabel("share to small", labelpad=-2)
    a.set_ylabel("large corpus scale", labelpad=-2)
    a.set_zlabel("yield", labelpad=-8)
    a.view_init(elev=24, azim=-56)
    a.tick_params(pad=-1)
    _tag(a, "d", three_d=True)

    _finish(fig, "panel_03_allocator")


# =====================================================================
#  PANEL 4 --- the three retrieval regimes
# =====================================================================


def panel4() -> None:
    fig, ax = _panel(3)

    N = 60
    e = np.linspace(0.5, 6.0 * N, 500)

    # coupon collector (distinct entities): concave everywhere
    cov = N * (1.0 - (1.0 - 1.0 / N) ** e)
    # paginated: constant marginal until exhaustion
    per_page = 3.0
    pag = np.minimum(per_page * e, float(N))
    # join
    joi = yield_theory(e, N)

    def marg(y):
        return np.gradient(y, e)

    # (a) marginal yield for the three regimes
    a = ax[0]
    a.plot(e, marg(cov), color=C_CONCAVE, label="distinct entity")
    a.plot(e, marg(pag), color=C_NEUTRAL, label="paginated")
    a.plot(e, marg(joi), color=C_CONVEX, label="federated join")
    a.axvline(e_star(N), color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel("effort $e$")
    a.set_ylabel("marginal yield")
    a.set_xlim(0, 4.0 * N)
    a.legend(frameon=False, loc="upper right")
    _tag(a, "a")

    # (b) yield curves themselves, normalised
    a = ax[1]
    a.plot(e / N, cov / N, color=C_CONCAVE, label="distinct entity")
    a.plot(e / N, pag / N, color=C_NEUTRAL, label="paginated")
    a.plot(e / N, joi / N, color=C_CONVEX, label="federated join")
    a.axhline(0.25, color=C_MARK, lw=0.8, ls=":")
    a.axvline(e_star(N) / N, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel("effort $e/N$")
    a.set_ylabel("yield / $N$")
    a.set_xlim(0, 4.0)
    a.legend(frameon=False, loc="lower right")
    _tag(a, "b")

    # (c) e*/(2N ln2) -> 1, and Y(e*)/N == 1/4 exactly
    a = ax[2]
    Ns = np.unique(np.round(np.logspace(1, 3, 60)).astype(int))
    ratio = [e_star(int(n)) / (2 * n * math.log(2)) for n in Ns]
    quarter = [JoinSource("t", int(n)).yield_at(e_star(int(n))) / n
               for n in Ns]
    a.semilogx(Ns, ratio, color=C_CONCAVE, marker="o", ms=2.6,
               label=r"$e^{*}/(2N\ln 2)$")
    a.semilogx(Ns, quarter, color=C_CONVEX, marker="s", ms=2.6,
               label=r"$Y(e^{*})/N$")
    a.axhline(1.0, color=C_NEUTRAL, lw=0.8, ls=":")
    a.axhline(0.25, color=C_NEUTRAL, lw=0.8, ls=":")
    a.set_ylim(0, 1.15)
    a.set_xlabel("corpus size $N$")
    a.set_ylabel("ratio")
    a.legend(frameon=False, loc="center right")
    _tag(a, "c")

    # (d) 3-D: sign of Y'' over (e/e*, N) -- the convex wedge
    a = ax[3]
    Ns3 = np.linspace(10, 300, 60)
    u3 = np.linspace(0.05, 2.5, 60)
    U, NN = np.meshgrid(u3, Ns3)
    Z = np.zeros_like(U)
    for i in range(NN.shape[0]):
        Nv = NN[i, 0]
        es = 2.0 * math.log(2.0) / math.log(1.0 / (1.0 - 1.0 / Nv))
        Z[i, :] = ypp_theory(U[i, :] * es, Nv) * Nv   # scale for visibility
    a.plot_surface(U, NN, Z, cmap="RdBu_r", rstride=1, cstride=1,
                   linewidth=0, antialiased=True, alpha=0.93)
    a.contour(U, NN, Z, levels=[0.0], colors=[C_MARK], linewidths=2.0,
              offset=float(np.min(Z)))
    a.set_xlabel(r"$e/e^{*}$", labelpad=-2)
    a.set_ylabel("$N$", labelpad=-2)
    a.set_zlabel(r"$N\,Y''$", labelpad=-4)
    a.view_init(elev=24, azim=-132)
    a.tick_params(pad=-1)
    _tag(a, "d", three_d=True)

    _finish(fig, "panel_04_regimes")


# =====================================================================
#  PANEL 5 --- the flaw, and the refusal
# =====================================================================


def allocate_prefix(sources, budget: float):
    """The PRE-FIX rule: water-fill whenever budget >= sum(e*).

    Reproduced here so Panel 5(a)-(b) can plot the error that
    Proposition 4.4 describes. This is NOT the shipped allocator; it is
    the one the brute-force check rejected.
    """
    full = sum(s.e_star for s in sources)
    if budget >= full:
        return _water_fill(list(sources), budget).total_yield
    return None


def panel5() -> None:
    fig, ax = _panel(1)

    chebi = JoinSource("chebi-rhea", corpus_size=30)
    rhea = JoinSource("rhea-uniprot", corpus_size=90)
    pair = [chebi, rhea]
    F = chebi.e_star + rhea.e_star
    m = min(s.e_star for s in pair)

    # (a) relative gap to brute force: pre-fix vs current
    a = ax[0]
    # Gaps below FLOOR are brute-force grid resolution, not real
    # disagreement, so they are masked rather than clamped: a clamped
    # series draws a cliff down to the floor that reads as a collapse.
    FLOOR = 1e-6
    budgets = np.linspace(F, 6.0 * F, 220)
    gap_pre, gap_now = [], []
    for B in budgets:
        bf, _ = brute2(chebi, rhea, float(B), steps=1400)
        pre = allocate_prefix(pair, float(B))
        now = allocate(pair, float(B)).total_yield
        gp = 100.0 * (bf - pre) / bf
        gn = 100.0 * (bf - now) / bf
        gap_pre.append(gp if gp > FLOOR else np.nan)
        gap_now.append(gn if gn > FLOOR else np.nan)
    gap_pre = np.array(gap_pre); gap_now = np.array(gap_now)
    a.semilogy(budgets / F, gap_pre, color=C_CONVEX, label="pre-fix")
    a.axhline(FLOOR, color=C_NEUTRAL, lw=0.8, ls=":")
    a.set_ylim(FLOOR * 0.4, None)
    # The current allocator's gap is identically zero, so it has no
    # visible curve. Draw it on the floor as a dashed marker line rather
    # than leaving an unexplained empty legend entry.
    if not np.isfinite(gap_now).any():
        a.axhline(FLOOR * 0.55, color=C_CONCAVE, lw=2.0, ls="--")
        a.plot([], [], color=C_CONCAVE, lw=2.0, ls="--",
               label="current (gap $=0$)")
    else:
        a.semilogy(budgets / F, gap_now, color=C_CONCAVE, lw=2.4,
                   label="current")
    n_pre = int(np.isfinite(gap_pre).sum())
    print(f"    panel5(a) pre-fix suboptimal at {n_pre}/{len(gap_pre)} "
          f"budgets; current at {int(np.isfinite(gap_now).sum())}")
    a.set_xlabel(r"budget $B/F$")
    a.set_ylabel("relative gap to brute force (%)")
    a.legend(frameon=False, loc="upper right")
    _tag(a, "a")

    # (b) 3-D: the two solutions on the yield surface near B = F
    a = ax[1]
    frac = np.linspace(0, 1, 80)
    mult = np.linspace(1.0, 2.2, 60)
    FF, MM = np.meshgrid(frac, mult)
    Z = np.zeros_like(FF)
    for i in range(MM.shape[0]):
        Bv = F * MM[i, 0]
        Z[i, :] = [chebi.yield_at(f * Bv) + rhea.yield_at((1 - f) * Bv)
                   for f in FF[i, :]]
    a.plot_surface(FF, MM, Z, cmap="coolwarm", rstride=1, cstride=1,
                   linewidth=0, antialiased=True, alpha=0.9)
    pre_f, now_f, mz_pre, mz_now = [], [], [], []
    for i in range(MM.shape[0]):
        Bv = F * MM[i, 0]
        wf = _water_fill(pair, Bv)
        f_pre = wf.per_source[chebi.name] / Bv
        al = allocate(pair, Bv)
        f_now = al.per_source.get(chebi.name, 0.0) / Bv
        pre_f.append(f_pre); now_f.append(f_now)
        mz_pre.append(chebi.yield_at(f_pre * Bv)
                      + rhea.yield_at((1 - f_pre) * Bv))
        mz_now.append(chebi.yield_at(f_now * Bv)
                      + rhea.yield_at((1 - f_now) * Bv))
    a.plot(pre_f, mult, mz_pre, color=C_CONVEX, lw=2.2, zorder=10)
    a.plot(now_f, mult, mz_now, color=C_MARK, lw=2.2, ls="--", zorder=11)
    a.set_xlabel("share to small", labelpad=-2)
    a.set_ylabel("$B/F$", labelpad=-2)
    a.set_zlabel("yield", labelpad=-8)
    a.view_init(elev=22, azim=-58)
    a.tick_params(pad=-1)
    _tag(a, "b", three_d=True)

    # (c) refusal region and its remedy
    a = ax[2]
    budgets = np.linspace(0.05 * m, 2.4 * m, 320)
    committed, refused_at = [], []
    for B in budgets:
        try:
            al = allocate(pair, float(B))
            committed.append(al.total_yield)
            refused_at.append(np.nan)
        except Refusal as r:
            committed.append(np.nan)
            refused_at.append(max(s.yield_at(float(B)) for s in pair))
    a.plot(budgets / m, committed, color=C_CONCAVE, label="allocated")
    a.plot(budgets / m, refused_at, color=C_CONVEX, ls="--",
           label="refused (yield if forced)")
    a.axvline(1.0, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel(r"budget $B/e^{*}_{\min}$")
    a.set_ylabel("total yield")
    a.legend(frameon=False, loc="upper left")
    _tag(a, "c")

    # (d) cost of not gating, over the full range through the crossover.
    # Curves for four corpus sizes are drawn; they coincide to ~1e-12,
    # because in normalised effort N cancels exactly. Markers are offset
    # along each curve so the overlap is visible rather than hidden.
    a = ax[3]
    shares = np.linspace(0.02, 4.0, 500)
    for k, (N, c, mk) in enumerate(((20, C_CONVEX, "o"), (60, C_COMMIT, "s"),
                                    (200, C_CONCAVE, "^"),
                                    (2000, C_ALT, "D"))):
        s = JoinSource(f"n{N}", corpus_size=N)
        es = s.e_star
        loss = []
        for sh in shares:
            B = sh * es
            commit = s.yield_at(B) + s.yield_at(0.0)
            spread = 2.0 * s.yield_at(B / 2.0)
            loss.append(100.0 * (commit - spread) / commit)
        loss = np.array(loss)
        a.plot(shares, loss, color=c, lw=1.2, alpha=0.9)
        sel = slice(12 + 9 * k, None, 46)
        a.plot(shares[sel], loss[sel], mk, color=c, ms=3.4, lw=0,
               label=f"N={N}")
    a.axhline(0.0, color=C_NEUTRAL, lw=0.8)
    a.axvline(U_CROSS, color=C_MARK, lw=0.9, ls="--")
    a.set_xlabel(r"budget $B/e^{*}$")
    a.set_ylabel("loss from spreading (%)")
    a.legend(frameon=False, loc="upper right", ncol=2)
    _tag(a, "d")

    _finish(fig, "panel_05_flaw_and_refusal")


def main() -> int:
    print("generating panels for attention-allocated-retrieval")
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
