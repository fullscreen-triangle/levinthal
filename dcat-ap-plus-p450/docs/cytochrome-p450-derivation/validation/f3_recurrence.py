"""F3. Configurations recur; the record does not.

(a) Kac ring: N sites on a ring, a ball of colour 0/1 at each site, a fixed
    set of marked edges. Each step every ball moves one site clockwise and
    flips colour when it crosses a marked edge. The dynamics is reversible
    and has period at most 2N. We follow the coarse-grained entropy of the
    colour fraction, the Hamming distance to the initial configuration, and
    a register that receives one entry per step.
(b) Reversal: run forward T steps, then run the inverse map T steps. The
    configuration returns exactly to the initial one; the register length
    is 2T.
(c) Arnold cat map on an N x N lattice: every orbit is periodic, with a
    period bounded by 3N; we compute the period for N = 2..400.
"""
from __future__ import annotations

import numpy as np

from common import BLUE, ORANGE, AQUA, INK2, FIG_FOUNDATION, panel_label, record, save_figure, plt

rng = np.random.default_rng(3)
N, MARKED = 400, 41
marked = np.zeros(N, dtype=bool)
marked[rng.choice(N, size=MARKED, replace=False)] = True  # edge i joins site i -> i+1


def forward(c):
    flipped = c ^ marked            # ball at site i crosses edge i
    return np.roll(flipped, 1)


def backward(c):
    c = np.roll(c, -1)
    return c ^ marked


def coarse_entropy(c):
    p = c.mean()
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


c0 = np.zeros(N, dtype=bool)
c = c0.copy()
S, H, register = [coarse_entropy(c)], [0], [0]
recurrence = None
for t in range(1, 2 * N + 1):
    c = forward(c)
    S.append(coarse_entropy(c)); H.append(int((c != c0).sum())); register.append(t)
    if recurrence is None and np.array_equal(c, c0):
        recurrence = t

T = 150
c = c0.copy(); Sr, Hr, reg_r = [coarse_entropy(c)], [0], [0]
for t in range(1, 2 * T + 1):
    c = forward(c) if t <= T else backward(c)
    Sr.append(coarse_entropy(c)); Hr.append(int((c != c0).sum())); reg_r.append(t)
returned = bool(np.array_equal(c, c0))


def cat_period(n):
    # period of the matrix [[1,1],[1,2]] modulo n: smallest k with A^k = I mod n
    a = np.array([[1, 1], [1, 2]], dtype=np.int64)
    m = np.eye(2, dtype=np.int64)
    for k in range(1, 3 * n + 2):
        m = (m @ a) % n
        if (m == np.eye(2, dtype=np.int64) % n).all():
            return k
    return None


ns = np.arange(2, 401)
periods = np.array([cat_period(int(n)) for n in ns])

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

ax = axes[0]
ax.plot(range(len(S)), np.array(S) / np.log(2), color=BLUE, lw=1.1, label="coarse entropy / ln 2")
ax.plot(range(len(H)), np.array(H) / N, color=ORANGE, lw=0.9, alpha=0.8, label="Hamming distance / N")
ax.axvline(recurrence, color=INK2, ls="--", lw=1)
ax.set_xlabel("step"); ax.set_ylabel("fraction")
ax.set_title(f"Kac ring, N={N}: recurrence at step {recurrence}")
ax.set_ylim(-0.03, 1.08)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
panel_label(ax, "A")

ax = axes[1]
ax.plot(range(len(Sr)), np.array(Sr) / np.log(2), color=BLUE, lw=1.2, label="coarse entropy / ln 2")
ax.plot(range(len(Hr)), np.array(Hr) / N, color=ORANGE, lw=1.0, label="Hamming distance / N")
ax.plot(range(len(reg_r)), np.array(reg_r) / (2 * T), color=AQUA, lw=1.4, label="register length / 2T")
ax.axvline(T, color=INK2, ls=":", lw=1)
ax.set_xlabel("step"); ax.set_ylabel("fraction"); ax.set_ylim(-0.03, 1.08)
ax.set_title("Reversed at step %d: configuration returns" % T)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
panel_label(ax, "B")

ax = axes[2]
ax.scatter(ns, periods, s=4, color=BLUE, lw=0, label="period $P(N)$")
ax.plot(ns, 3 * ns, color=INK2, ls="--", lw=1, label="$3N$")
ax.set_xlabel("lattice side $N$"); ax.set_ylabel("period of every orbit")
ax.set_title("Cat map: finite cells, finite period")
ax.legend(loc="upper left")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_FOUNDATION, "f3_recurrence")

record("f3_recurrence", {
    "kac_ring": {"N": N, "marked_edges": MARKED, "recurrence_step": recurrence,
                 "max_coarse_entropy_over_ln2": float(max(S) / np.log(2))},
    "reversal": {"T": T, "configuration_returned": returned,
                 "register_length_at_return": reg_r[-1],
                 "register_strictly_increasing": bool(np.all(np.diff(reg_r) == 1))},
    "cat_map": {"N_range": [2, 400], "all_periods_finite": bool(all(p is not None for p in periods)),
                "max_period_over_N": float(max(p / n for p, n in zip(periods, ns))),
                "periods_sample": {int(n): int(p) for n, p in zip(ns, periods) if n in (5, 10, 50, 100, 250, 400)}},
})
print("recurrence", recurrence, "returned", returned, "max P/N", max(p / n for p, n in zip(periods, ns)))
