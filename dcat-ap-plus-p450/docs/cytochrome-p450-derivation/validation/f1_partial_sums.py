"""F1. Partial sums of counted and measured costs, and the dichotomy.

Checks, for the foundation manuscript:
  (a) every partial sum of integer costs satisfies S_n >= n (no exceptions);
  (b) positive real costs admit a convergent sum (geometric) and a running
      minimum that tends to zero, so positivity alone gives no floor;
  (c) in the dichotomy the distance series converges while the number of
      committed midpoints needed to reach within eps of the goal diverges
      as eps -> 0, and is finite for every eps > 0.
"""
from __future__ import annotations

import math

import numpy as np

from common import (BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, FIG_FOUNDATION,
                    panel_label, record, save_figure, plt)

rng = np.random.default_rng(20261011)
N = 10_000
TRIALS = 1_000

# (a) integer costs: uniform on {1,...,5}, and the minimal sequence c_i = 1.
int_costs = rng.integers(1, 6, size=(TRIALS, N))
int_sums = np.cumsum(int_costs, axis=1)
n_idx = np.arange(1, N + 1)
violations = int((int_sums < n_idx).sum())
strict_increase = bool((np.diff(int_sums, axis=1) >= 1).all())

# (b) positive real costs.
geom = np.cumsum(0.5 ** n_idx)                     # converges to 1
harm = np.cumsum(1.0 / n_idx)                      # diverges, no floor
real_costs = rng.uniform(0, 1, size=(TRIALS, N)) / n_idx   # positive, inf -> 0
running_min_real = np.minimum.accumulate(real_costs, axis=1)
running_min_int = np.minimum.accumulate(int_costs, axis=1)

# (c) the dichotomy: distance D, midpoints at D(1 - 2^-k).
D = 1.0
eps = np.logspace(-1, -15, 200)
k_needed = np.ceil(np.log2(D / eps)).astype(int)
dist_after_k = D * (1 - 0.5 ** np.arange(0, 61))
commit_budget = np.arange(0, 61)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

ax = axes[0]
med = np.median(int_sums, axis=0)
lo, hi = np.percentile(int_sums, [2.5, 97.5], axis=0)
ax.fill_between(n_idx, lo, hi, color=BLUE, alpha=0.18, lw=0)
ax.plot(n_idx, med, color=BLUE, label="integer costs, $c_i\\in\\{1..5\\}$")
ax.plot(n_idx, n_idx, color=INK2, ls="--", lw=1.1, label="$S_n=n$ (all $c_i=1$)")
ax.plot(n_idx, harm, color=ORANGE, label="real, $c_i=1/i$")
ax.plot(n_idx, geom, color=AQUA, label="real, $c_i=2^{-i}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of commitments $n$"); ax.set_ylabel("partial sum $S_n$")
ax.set_title("Counted costs never stall")
ax.legend(loc="upper left")
panel_label(ax, "A")

ax = axes[1]
q = np.percentile(running_min_real, [2.5, 50, 97.5], axis=0)
ax.fill_between(n_idx, q[0], q[2], color=ORANGE, alpha=0.18, lw=0)
ax.plot(n_idx, q[1], color=ORANGE, label="real, $c_i=U_i/i$")
ax.plot(n_idx, np.median(running_min_int, axis=0), color=BLUE, label="integer")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of terms $n$"); ax.set_ylabel("running minimum of $c_j$, $j\\leq n$")
ax.set_title("Positivity is not a floor")
ax.legend(loc="lower left")
panel_label(ax, "B")

ax = axes[2]
ax.plot(eps, k_needed, color=BLUE, label="committed midpoints needed")
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks([1e-2, 1e-6, 1e-10, 1e-14])
ax.set_xlabel("distance left to the goal, $\\varepsilon$")
ax.set_ylabel("commitments $k(\\varepsilon)=\\lceil\\log_2 D/\\varepsilon\\rceil$")
ax2 = ax.inset_axes([0.14, 0.55, 0.38, 0.33])
ax2.plot(commit_budget, dist_after_k, color=AQUA, lw=1.3)
ax2.set_xlabel("commitments", fontsize=6.5); ax2.set_ylabel("distance", fontsize=6.5)
ax2.tick_params(labelsize=6); ax2.set_ylim(0, 1.05)
ax.set_title("Distance converges; count diverges")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_FOUNDATION, "f1_partial_sums")

record("f1_partial_sums", {
    "trials": TRIALS, "terms": N,
    "integer_partial_sum_violations_of_S_n_ge_n": violations,
    "integer_partial_sums_strictly_increasing": strict_increase,
    "integer_running_min_final": int(running_min_int[:, -1].min()),
    "real_running_min_final_median": float(np.median(running_min_real[:, -1])),
    "real_running_min_final_max": float(running_min_real[:, -1].max()),
    "geometric_sum_at_N": float(geom[-1]),
    "harmonic_sum_at_N": float(harm[-1]),
    "dichotomy_commitments_for_eps": {f"{e:.0e}": int(math.ceil(math.log2(D / e)))
                                      for e in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15)},
})
print("violations", violations, "| real running min median",
      float(np.median(running_min_real[:, -1])))
