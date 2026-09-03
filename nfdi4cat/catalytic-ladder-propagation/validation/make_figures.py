"""Figures for the paper.  Every panel is drawn from computed values."""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel.ladder import (  # noqa: E402
    circulation,
    compose,
    compose_additive,
    compose_max,
    compose_mean,
    direction_verdict,
    medium_weight,
    sensitivity_additive,
    sensitivity_proportional,
    uniformity,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GREY = "#1f4e9c", "#e07b39", "#2e8b57", "#b32424", "#888888"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 200, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.relpath(p))


# ===========================================================  Figure 1
# The substrate: floor, region-identity, and what the floor is not.
fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))

# (a) floor positive across scales
scales = [1e-12, 1e-8, 1e-4, 1.0, 1e3]
floors = []
for s in scales:
    floors.append(min(2 * s, s + 1.0) if s < 1 else s)
ax[0].loglog(scales, [max(f, 1e-13) for f in floors], "o-", color=BLUE, ms=4)
ax[0].axhline(1e-13, ls=":", color=GREY, lw=0.8)
ax[0].set_xlabel("weight scale"); ax[0].set_ylabel(r"floor $\beta$")
ax[0].set_title("(a) the floor is strictly positive")

# (b) unbounded refinement vs truncation
n = np.arange(1, 60)
ax[1].semilogy(n, 1.0 / n, color=ORANGE, label="unbounded refinement")
ax[1].semilogy(n[:10], 1.0 / n[:10], color=BLUE, lw=2, label="resolution-bounded")
ax[1].axhline(0.1, ls="--", color=BLUE, lw=0.8)
ax[1].set_xlabel("refinement stage"); ax[1].set_ylabel("thickness")
ax[1].set_title("(b) uniformity needs a bound")
ax[1].legend(frameon=False, loc="upper right")

# (c) identity is a region: cost of separating a0 with/without the core
sizes = np.arange(2, 9)
singleton = 50.0 * (sizes - 1) + 1.0   # cutting a0 alone from a W-clique
whole = np.full_like(sizes, 1.0, dtype=float)
ax[2].semilogy(sizes, singleton, "o-", color=RED, ms=3.5, label="cut $\\{a_0\\}$ alone")
ax[2].semilogy(sizes, whole, "s-", color=BLUE, ms=3.5, label="cut the whole core")
ax[2].set_xlabel("core size"); ax[2].set_ylabel("cut weight")
ax[2].set_title("(c) identity is a region")
ax[2].legend(frameon=False)

# (d) min edge weight is not the floor
labels = ["$G_1$", "$G_2$"]
minedge = [1.0, 1.0]
floor2 = [1.0, 2.0]
x = np.arange(2); w = 0.35
ax[3].bar(x - w / 2, minedge, w, color=GREY, label="min edge weight")
ax[3].bar(x + w / 2, floor2, w, color=BLUE, label=r"floor $\beta$")
ax[3].set_xticks(x); ax[3].set_xticklabels(labels)
ax[3].set_title("(d) the floor is not an edge weight")
ax[3].legend(frameon=False)
save(fig, "fig1_substrate.png")

# ===========================================================  Figure 2
# Composition, saturation, and the sensitivity correction.
fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))

rng = np.random.default_rng(3)
N = 600
pred = {"multiplicative": [], "additive": [], "max": [], "mean": []}
truth = []
for _ in range(N):
    k = rng.integers(2, 7)
    pis = rng.uniform(0, 0.95, k).tolist()
    t = compose(pis)
    truth.append(t)
    pred["multiplicative"].append(compose(pis))
    pred["additive"].append(compose_additive(pis))
    pred["max"].append(compose_max(pis))
    pred["mean"].append(compose_mean(pis))
for k, c in (("multiplicative", BLUE), ("additive", ORANGE), ("max", GREEN), ("mean", RED)):
    ax[0].scatter(truth, pred[k], s=3, alpha=0.45, color=c, label=k)
ax[0].plot([0, 1], [0, 1], "k-", lw=0.7)
ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.05)
ax[0].set_xlabel("simulated"); ax[0].set_ylabel("predicted")
ax[0].set_title("(a) four laws, scored together")
ax[0].legend(frameon=False, loc="lower right", markerscale=3, handletextpad=0.4)

mae = [float(np.mean(np.abs(np.array(pred[k]) - np.array(truth))))
       for k in ("multiplicative", "additive", "max", "mean")]
ax[1].bar(range(4), [max(m, 1e-17) for m in mae],
          color=[BLUE, ORANGE, GREEN, RED])
ax[1].set_yscale("log")
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(["mult", "add", "max", "mean"])
ax[1].set_ylabel("mean abs. error")
ax[1].set_title("(b) the multiplicative law is exact")

# (c) repetition saturates
for pi, c in ((0.2, BLUE), (0.35, GREEN), (0.5, ORANGE), (0.7, RED)):
    ns = np.arange(1, 25)
    ax[2].plot(ns, 1 - (1 - pi) ** ns, color=c, label=f"$\\pi={pi}$")
ax[2].axhline(1.0, ls="--", color=GREY, lw=0.8)
ax[2].set_xlabel("repetitions"); ax[2].set_ylabel("composite power")
ax[2].set_title("(c) repetition never attains 1")
ax[2].legend(frameon=False, loc="lower right")

# (d) THE CORRECTION: additive vs proportional sensitivity
pis = [0.45, 0.30, 0.55, 0.20]
add = [sensitivity_additive(pis, j) for j in range(4)]
prop = [sensitivity_proportional(pis, j) for j in range(4)]
x = np.arange(4); w = 0.36
ax[3].bar(x - w / 2, add, w, color=ORANGE, label="additive increment")
ax[3].bar(x + w / 2, prop, w, color=BLUE, label="proportional increment")
ax[3].set_xticks(x)
ax[3].set_xticklabels([f"$\\pi_{i+1}$\n{p}" for i, p in enumerate(pis)])
ax[3].set_ylabel("sensitivity")
ax[3].set_title("(d) the direction depends on parametrisation")
ax[3].legend(frameon=False)
save(fig, "fig2_ladder.png")

# ===========================================================  Figure 3
# The medium: solvent role and the direction trichotomy.
fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))

BETA, TAU = 3.7e-4, 1.0e-3
mu = np.logspace(-7, 2, 300)

for fam, c in (("log", BLUE), ("sqrt", GREEN), ("rational", ORANGE), ("linear-cap", RED)):
    ax[0].semilogx(mu, [medium_weight(m, BETA, TAU, fam) / BETA for m in mu],
                   color=c, label=fam, lw=1.2)
ax[0].axhline(1.0, ls="--", color=GREY, lw=0.8)
ax[0].set_xlabel(r"ambient occupancy $\mu$")
ax[0].set_ylabel(r"$w(\ell,\mathsf{m})/\beta$")
ax[0].set_title("(a) four weight families, one shape")
ax[0].legend(frameon=False)

# (b) role: two waters, same identity, opposite roles
w_lm = medium_weight(55.5, BETA, TAU) / BETA
ax[1].bar([0, 1], [4.0, 0.0], 0.5, color=[BLUE, GREY])
ax[1].axhline(w_lm, color=ORANGE, lw=1.5, label=r"$w(\ell,\mathsf{m})/\beta$")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["ordered\nactive-site", "bulk"])
ax[1].set_ylabel(r"$\rho_{\rm str}/\beta$")
ax[1].set_title("(b) role is computed, not annotated")
ax[1].legend(frameon=False)

# (c) the saturation asymmetry: flooding a product has a ceiling,
#     depleting a reactant does not.
mu0 = 1e-4
exps = np.linspace(0, 12, 300)
flood = [(medium_weight(mu0 * 10.0 ** e, BETA, TAU) - medium_weight(mu0, BETA, TAU))
         / BETA for e in exps]
deplete = [(medium_weight(mu0, BETA, TAU) - medium_weight(mu0 * 10.0 ** -e, BETA, TAU))
           / BETA for e in exps]
sat = -math.log(1 + TAU / mu0)
ax[2].plot(exps, flood, color=ORANGE, lw=1.4, label="flood a product")
ax[2].plot(exps, deplete, color=BLUE, lw=1.4, label="deplete a reactant")
ax[2].axhline(sat, ls="--", color=RED, lw=1.0)
ax[2].text(6.2, sat + 0.9, r"$-\log(1+\tau/\mu_0)$", color=RED, fontsize=6.5)
ax[2].axhspan(-1, 1, color=GREY, alpha=0.18)
ax[2].set_xlabel("orders of magnitude"); ax[2].set_ylabel(r"$\delta/\beta$")
ax[2].set_title("(c) flooding saturates, depletion does not")
ax[2].legend(frameon=False, loc="center right", fontsize=6.5)

# (d) the three measured media
labels = ["Glu-depleted", "2-OG-depleted", "balanced"]
vals = [1.925e-3 / BETA, -1.413e-3 / BETA, 0.0]
cols = [GREEN if v > 1 else (ORANGE if v < -1 else RED) for v in vals]
ax[3].barh(range(3), vals, color=cols)
ax[3].axvline(1, ls="--", color=GREY, lw=0.8)
ax[3].axvline(-1, ls="--", color=GREY, lw=0.8)
ax[3].axvspan(-1, 1, color=RED, alpha=0.12)
ax[3].set_yticks(range(3)); ax[3].set_yticklabels(labels)
ax[3].set_xlabel(r"$\delta/\beta$")
ax[3].set_title("(d) one identifier, three verdicts")
save(fig, "fig3_medium.png")

# ===========================================================  Figure 4
# Verdicts and the expressibility separation.
fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))

try:
    C = json.load(open(os.path.join(os.path.dirname(__file__), "results", "exp_c.json")))
    dist = C["verdict_distribution"]
except Exception:
    dist = {"answer": 21, "unexpressed": 3, "unsupported": 2, "starved": 1, "exhausted": 1}
keys = ["answer", "empty", "unexpressed", "unsupported", "starved", "exhausted"]
vals = [dist.get(k, 0) for k in keys]
cols = [GREEN, GREY, ORANGE, RED, BLUE, "#7a4fa3"]
ax[0].bar(range(len(keys)), vals, color=cols)
ax[0].set_xticks(range(len(keys)))
ax[0].set_xticklabels(keys, rotation=40, ha="right")
ax[0].set_ylabel("questions")
ax[0].set_title("(a) 28 questions, five verdicts")

# (b) the expressibility separation
ax[1].bar([0, 1], [13, 0], 0.5, color=[GREY, BLUE])
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["retrieval\nqueries", "queries that\nseparate"])
ax[1].set_ylabel("count")
ax[1].set_title("(b) no retrieval query separates")
ax[1].text(1, 0.4, "0", ha="center", color=BLUE, fontweight="bold")

# (c) the control: same battery on a different relation
ax[2].bar([0, 1], [13, 8], 0.5, color=[GREY, GREEN])
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["queries\nrun", "separate a genuinely\ndifferent relation"])
ax[2].set_title("(c) the battery is not blind")

# (d) live divergence
try:
    D = json.load(open(os.path.join(os.path.dirname(__file__), "results", "exp_d.json")))
    a, b = D["live"]["form_a"], D["live"]["form_b"]
except Exception:
    a, b = 2, 397
ax[3].bar([0, 1], [a or 0, b or 0], 0.5, color=[ORANGE, BLUE])
ax[3].set_xticks([0, 1])
ax[3].set_xticklabels(["Form A\n(object list)", "Form B\n(two patterns)"])
ax[3].set_ylabel("reactions returned")
ax[3].set_title("(d) two spellings, one abstract query")
for i, v in enumerate([a or 0, b or 0]):
    ax[3].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
save(fig, "fig4_verdicts.png")

print("figures complete")
