"""F2. Separation cost from the medium under expansion of a contact graph.

A contact graph has a medium node m in contact with every other node. The
separation cost sigma(v) of a node is the weight of a minimum cut separating
v from m (a maximum flow from v to m). We grow graphs one node at a time and
check, for counted (integer) and measured (real) weights:
  (a) sigma(v) of a fixed node never decreases under expansion;
  (b) the smallest edge incident to v, and the global smallest edge, stay at
      1 for integer weights but drift towards 0 for real weights;
  (c) in the final graphs, the fraction of nodes whose singleton cut {v} is
      strictly heavier than the minimum cut, so that every minimiser is a
      region containing more than v.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from common import BLUE, ORANGE, INK2, FIG_FOUNDATION, panel_label, record, save_figure, plt

rng = np.random.default_rng(7)
SEQS, STEPS, K = 40, 160, 3
MEDIUM, V = "m", 0


def weight(kind):
    if kind == "integer":
        return int(rng.integers(1, 6))
    return float(rng.exponential(1.5))          # positive reals, no floor


def grow(kind):
    g = nx.Graph()
    for u in range(4):
        g.add_edge(u, MEDIUM, capacity=weight(kind))
    for u in range(4):
        for w in range(u + 1, 4):
            if rng.random() < 0.6:
                g.add_edge(u, w, capacity=weight(kind))
    sig, inc_min, glob_min = [], [], []
    nxt = 4
    for _ in range(STEPS):
        others = [u for u in g.nodes if u != MEDIUM]
        targets = rng.choice(others, size=min(K, len(others)), replace=False)
        g.add_edge(nxt, MEDIUM, capacity=weight(kind))
        for t in targets:
            g.add_edge(nxt, int(t), capacity=weight(kind))
        # with probability 0.15 the fixed node v gains a new neighbour
        if rng.random() < 0.15 and not g.has_edge(V, nxt):
            g.add_edge(V, nxt, capacity=weight(kind))
        nxt += 1
        sig.append(nx.maximum_flow_value(g, V, MEDIUM))
        inc_min.append(min(d["capacity"] for _, _, d in g.edges(V, data=True)))
        glob_min.append(min(d["capacity"] for _, _, d in g.edges(data=True)))
    return g, np.array(sig), np.array(inc_min), np.array(glob_min)


def region_fraction(g):
    nodes = [u for u in g.nodes if u != MEDIUM]
    region, sigmas = 0, []
    for u in nodes:
        s = nx.maximum_flow_value(g, u, MEDIUM)
        singleton = sum(d["capacity"] for _, _, d in g.edges(u, data=True))
        sigmas.append(s)
        if singleton > s + 1e-9:
            region += 1
    return region / len(nodes), np.array(sigmas)


out = {}
for kind in ("integer", "real"):
    runs = [grow(kind) for _ in range(SEQS)]
    sig = np.array([r[1] for r in runs])
    inc = np.array([r[2] for r in runs])
    glob = np.array([r[3] for r in runs])
    decreases = int((np.diff(sig, axis=1) < -1e-9).sum())
    frac, sigmas = zip(*(region_fraction(r[0]) for r in runs[:10]))
    all_sig = np.concatenate(sigmas)
    out[kind] = dict(sig=sig, inc=inc, glob=glob, decreases=decreases,
                     region_fraction=np.array(frac), all_sigma=all_sig)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
steps = np.arange(1, STEPS + 1)

ax = axes[0]
for i in range(8):
    ax.plot(steps, out["integer"]["sig"][i], color=BLUE, lw=1.0, alpha=0.75)
ax.set_xlabel("nodes added"); ax.set_ylabel("separation cost $\\sigma(v)$")
ax.set_title("$\\sigma(v)$ never decreases")
ax.text(0.97, 0.05, f"decreases observed: {out['integer']['decreases']} (integer), "
        f"{out['real']['decreases']} (real)\nover {SEQS}x{STEPS} steps each",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2)
panel_label(ax, "A")

ax = axes[1]
for kind, c in (("integer", BLUE), ("real", ORANGE)):
    q = np.percentile(out[kind]["glob"], [10, 50, 90], axis=0)
    ax.fill_between(steps, q[0], q[2], color=c, alpha=0.18, lw=0)
    ax.plot(steps, q[1], color=c, label=f"{kind} weights")
ax.set_yscale("log")
ax.set_xlabel("nodes added"); ax.set_ylabel("smallest edge weight in graph")
ax.set_title("Only counted weights keep a floor")
ax.legend(loc="lower left")
panel_label(ax, "B")

ax = axes[2]
bins = np.logspace(-2.5, 2, 40)
ax.hist(out["real"]["all_sigma"], bins=bins, color=ORANGE, alpha=0.7, label="real weights")
ax.hist(out["integer"]["all_sigma"], bins=bins, color=BLUE, alpha=0.7, label="integer weights")
ax.axvline(1, color=INK2, ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("$\\sigma(u)$ over all nodes of final graphs"); ax.set_ylabel("nodes")
ax.set_title("Individuation of every node")
ax.legend(loc="upper left")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_FOUNDATION, "f2_individuation")

payload = {"sequences": SEQS, "steps": STEPS, "edges_per_new_node": K + 1}
for kind in ("integer", "real"):
    o = out[kind]
    payload[kind] = {
        "sigma_v_decreases": o["decreases"],
        "sigma_v_final_median": float(np.median(o["sig"][:, -1])),
        "global_min_edge_final_median": float(np.median(o["glob"][:, -1])),
        "global_min_edge_final_min": float(o["glob"][:, -1].min()),
        "incident_min_of_v_final_median": float(np.median(o["inc"][:, -1])),
        "min_sigma_over_all_nodes": float(o["all_sigma"].min()),
        "fraction_nodes_identity_is_region_mean": float(o["region_fraction"].mean()),
    }
record("f2_individuation", payload)
print({k: payload[k] for k in ("integer", "real")})
