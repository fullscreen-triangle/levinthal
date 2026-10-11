"""G5. Compaction, burial of the heme, and separation cost in the contact graph.

(a) Radius of gyration of the C-alpha atoms of the heme-bearing chain,
    against chain length, for P450s and other heme proteins; compared with
    the empirical scalings for folded chains, Rg = 2.2 N^0.38 (Skolnick et
    al. 1997), and for chemically denatured chains, Rg = 1.927 N^0.598
    (Kohn et al. 2004).
(b) Solvent exposure of the heme: its accessible surface in the complex
    divided by its accessible surface in isolation (Shrake-Rupley, probe
    1.4 Å, 192 points per atom). Waters and non-heme ligands are removed.
(c) A residue contact graph with integer weights: the weight between two
    residues (or a residue and the heme) is the number of heavy-atom pairs
    closer than 4.5 Å; the medium is joined to each node by the number of
    its atoms with accessible surface above 1 Å^2. The separation cost of
    a node is the maximum flow from it to the medium.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from common import BLUE, ORANGE, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt
from structures import HEME_NAMES, OTHER_HEME_ENTRIES, P450_ENTRIES, THREE_TO_ONE, heme_sites, load

RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "FE": 1.40, "SE": 1.90}
PROBE = 1.4


def sphere(n=192):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


SPH = sphere()


def sasa(xyz, radii):
    r = radii + PROBE
    tree = cKDTree(xyz)
    out = np.zeros(len(xyz))
    rmax = r.max()
    for i in range(len(xyz)):
        pts = xyz[i] + r[i] * SPH
        nb = [j for j in tree.query_ball_point(xyz[i], r[i] + rmax) if j != i]
        if not nb:
            out[i] = 4 * np.pi * r[i] ** 2
            continue
        d = np.linalg.norm(pts[:, None, :] - xyz[nb][None, :, :], axis=2)
        free = (d >= r[nb][None, :]).all(axis=1)
        out[i] = 4 * np.pi * r[i] ** 2 * free.mean()
    return out


def chain_system(pid):
    s = load(pid)
    site = heme_sites(s)[0]
    ch, hseq = site["chain"], site["heme_resseq"]
    prot = [a for a in s.atoms if a.chain == ch and not a.hetero and a.resname in THREE_TO_ONE and a.element != "H"]
    heme = [a for a in s.atoms if a.chain == ch and a.resname in HEME_NAMES and a.resseq == hseq and a.element != "H"]
    return s, prot, heme


def analyse(pid, contact_graph=False):
    s, prot, heme = chain_system(pid)
    atoms = prot + heme
    xyz = np.array([a.xyz for a in atoms])
    rad = np.array([RADII.get(a.element, 1.8) for a in atoms])
    acc = sasa(xyz, rad)
    hxyz = np.array([a.xyz for a in heme])
    h_alone = sasa(hxyz, np.array([RADII.get(a.element, 1.8) for a in heme])).sum()
    h_in = acc[len(prot):].sum()
    ca = np.array([a.xyz for a in prot if a.name == "CA"])
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    out = {"N": int(len(ca)), "rg": rg, "heme_exposure": float(h_in / h_alone),
           "buried_atom_fraction": float((acc[:len(prot)] < 0.5).mean()),
           "resolution": s.resolution}
    if contact_graph:
        node = [("HEME",) if a in heme else (a.resseq, a.icode) for a in atoms]
        g = nx.Graph()
        tree = cKDTree(xyz)
        for i, j in tree.query_pairs(4.5):
            u, v = node[i], node[j]
            if u == v:
                continue
            w = g.get_edge_data(u, v, {"capacity": 0})["capacity"]
            g.add_edge(u, v, capacity=w + 1)
        exposed = {}
        for n_, a_ in zip(node, acc):
            exposed[n_] = exposed.get(n_, 0) + int(a_ > 1.0)
        for n_, e_ in exposed.items():
            if e_ > 0:
                g.add_edge(n_, "MEDIUM", capacity=e_)
        sig = {n_: nx.maximum_flow_value(g, n_, "MEDIUM") for n_ in g.nodes if n_ != "MEDIUM"}
        singleton = {n_: sum(d["capacity"] for _, _, d in g.edges(n_, data=True)) for n_ in sig}
        out["sigma"] = sig
        out["singleton"] = singleton
        out["sigma_heme"] = sig[("HEME",)]
    return out


results = {}
for pid in list(P450_ENTRIES) + list(OTHER_HEME_ENTRIES):
    results[pid] = analyse(pid, contact_graph=pid in ("2CPP", "1TQN"))
    print(pid, {k: v for k, v in results[pid].items() if k not in ("sigma", "singleton")})

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

ax = axes[0]
nn = np.logspace(np.log10(80), np.log10(700), 50)
ax.plot(nn, 2.2 * nn ** 0.38, color=INK2, ls="--", lw=1, label="folded, $2.2N^{0.38}$")
ax.plot(nn, 1.927 * nn ** 0.598, color=MUTED, ls=":", lw=1.2, label="denatured, $1.927N^{0.598}$")
for pid, r in results.items():
    is_p = pid in P450_ENTRIES
    ax.scatter(r["N"], r["rg"], s=22, color=BLUE if is_p else ORANGE, edgecolor="white", lw=0.6, zorder=3)
ax.scatter([], [], s=22, color=BLUE, label="P450s"); ax.scatter([], [], s=22, color=ORANGE, label="other heme proteins")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([100, 200, 300, 500], ["100", "200", "300", "500"]); ax.minorticks_off()
ax.set_yticks([10, 20, 40, 80], ["10", "20", "40", "80"])
ax.set_xlabel("residues in chain $N$"); ax.set_ylabel("$R_g$ of C$\\alpha$ (Å)")
ax.set_title("Chains are compact")
ax.legend(loc="upper left")
panel_label(ax, "A")

ax = axes[1]
ids = sorted(results, key=lambda p: results[p]["heme_exposure"])
vals = [results[p]["heme_exposure"] * 100 for p in ids]
ax.barh(range(len(ids)), vals, color=[BLUE if p in P450_ENTRIES else ORANGE for p in ids], height=0.7)
ax.set_yticks(range(len(ids)), ids, fontsize=6.3)
ax.set_xlabel("heme accessible surface, % of isolated heme")
ax.set_title("The heme is buried")
panel_label(ax, "B")

ax = axes[2]
r = results["2CPP"]
sig = np.array([v for k, v in r["sigma"].items() if k != ("HEME",)])
ax.hist(sig, bins=40, color=MUTED)
ax.axvline(r["sigma_heme"], color=ORANGE, lw=1.6)
ax.annotate(f"heme, $\\sigma$={r['sigma_heme']:.0f}", (r["sigma_heme"], ax.get_ylim()[1] * 0.8),
            xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=7, color=INK2)
ax.set_xlabel("separation cost from the medium $\\sigma$ (contacts)")
ax.set_ylabel("residues")
ax.set_title("2CPP contact graph, integer weights")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_P450, "g5_fold")

payload = {}
for pid, r in results.items():
    payload[pid] = {k: v for k, v in r.items() if k not in ("sigma", "singleton")}
for pid in ("2CPP", "1TQN"):
    r = results[pid]
    sig = np.array([v for k, v in r["sigma"].items() if k != ("HEME",)])
    region = [k for k in r["sigma"] if r["singleton"][k] > r["sigma"][k] + 1e-9]
    payload[pid]["sigma_residue_quantiles"] = np.percentile(sig, [0, 25, 50, 75, 100]).tolist()
    payload[pid]["heme_sigma_percentile_among_residues"] = float((sig < r["sigma_heme"]).mean())
    payload[pid]["heme_singleton_cut"] = r["singleton"][("HEME",)]
    payload[pid]["nodes_with_region_identity"] = len(region)
    payload[pid]["min_sigma"] = float(sig.min())
    payload[pid]["sigma_values"] = sorted(float(v) for v in sig)
record("g5_fold", {"probe": PROBE, "points_per_atom": len(SPH), "entries": payload})
