"""F4. Ladders: the gap closes multiplicatively, its logarithm grows additively.

A ladder of rung powers p_1..p_n leaves the gap g_n = prod (1 - p_i). We
compare ladders with a floor on p (p_i >= p_min > 0), a ladder whose powers
are positive but summable (no floor), and a harmonic ladder. With exact
rational arithmetic we check that no finite ladder with p_i < 1 closes the
gap. We also exhibit two ladders with equal composite and different
trajectories.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from common import BLUE, ORANGE, AQUA, YELLOW, INK2, FIG_FOUNDATION, panel_label, record, save_figure, plt

rng = np.random.default_rng(11)
n = np.arange(1, 501)
ladders = {
    "constant, $p=0.2$": np.full(n.size, 0.2),
    "random, $p\\in[0.05,0.5]$": rng.uniform(0.05, 0.5, n.size),
    "harmonic, $p_i=0.5/i$": 0.5 / n,
    "summable, $p_i=0.5/i^2$": 0.5 / n ** 2,
}
colors = [BLUE, AQUA, ORANGE, YELLOW]

log_gap = {k: np.cumsum(-np.log1p(-p)) for k, p in ladders.items()}

# exact check: rational powers strictly below one never close the gap
exact_closed = 0
for trial in range(200):
    g = Fraction(1)
    for _ in range(200):
        num = int(rng.integers(1, 99))
        g *= 1 - Fraction(num, 100)
        if g == 0:
            exact_closed += 1
            break

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

ax = axes[0]
for (k, lg), c in zip(log_gap.items(), colors):
    ax.plot(n, np.exp(-lg), color=c, label=k)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("rungs climbed $n$"); ax.set_ylabel("remaining gap $g_n/g_0$")
ax.set_title("The gap shrinks, never to zero")
ax.legend(loc="lower left")
panel_label(ax, "A")

ax = axes[1]
for (k, lg), c in zip(log_gap.items(), colors):
    ax.plot(n, lg, color=c)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("rungs climbed $n$"); ax.set_ylabel("$\\sum_i -\\ln(1-p_i)$")
ax.set_title("Log gap: a sum of positive terms")
residual = float(np.exp(-log_gap["summable, $p_i=0.5/i^2$"][-1]))
ax.text(0.97, 0.05, f"summable ladder stalls:\nresidual gap {residual:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
panel_label(ax, "B")

ax = axes[2]
a = [1, 0.5, 0.25]
b = [1, 0.25]
ax.step([0, 1, 2], a, where="post", color=BLUE, label="powers (0.5, 0.5)")
ax.step([0, 1], b, where="post", color=ORANGE, label="power (0.75)")
ax.plot([0, 2], [0.25, 0.25], color=INK2, ls=":", lw=1)
ax.set_xticks([0, 1, 2]); ax.set_xlim(-0.1, 2.3); ax.set_ylim(0, 1.08)
ax.set_xlabel("rung"); ax.set_ylabel("remaining gap")
ax.set_title("Same composite, different trajectory")
ax.legend(loc="upper right")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_FOUNDATION, "f4_ladders")

record("f4_ladders", {
    "rungs": int(n.size),
    "final_gap": {k: float(np.exp(-lg[-1])) for k, lg in log_gap.items()},
    "summable_ladder_residual": residual,
    "summable_ladder_residual_closed_form": "prod(1-0.5/i^2) -> sin(pi/sqrt2)/(pi/sqrt2)",
    "summable_closed_form_value": float(np.sin(np.pi / np.sqrt(2)) / (np.pi / np.sqrt(2))),
    "exact_rational_ladders_closed": exact_closed,
    "exact_rational_ladders_tested": 200,
})
print("residual", residual, "closed form", np.sin(np.pi / np.sqrt(2)) / (np.pi / np.sqrt(2)), "closed", exact_closed)
