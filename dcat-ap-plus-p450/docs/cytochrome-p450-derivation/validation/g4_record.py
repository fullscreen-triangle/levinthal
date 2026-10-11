"""G4. The record: word length, redundancy and substitution tolerance.

(a) The shortest word over a four-letter alphabet that can name twenty
    residues and a stop: 4^L >= 21 gives L = 3.
(b) The standard genetic code (NCBI table 1): codons per residue.
(c) Substitution tolerance. For every single-letter substitution between
    sense codons we take the squared change in Kyte-Doolittle hydropathy of
    the encoded residue (zero when synonymous). The mean over all such
    substitutions is the code's cost. We compare the standard code with
    200,000 codes in which the twenty residue labels are permuted among the
    twenty synonymous blocks, keeping the block structure and the stops
    (the construction of Haig & Hurst 1991).
"""
from __future__ import annotations

import itertools

import numpy as np

from common import BLUE, ORANGE, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt
from scales import KD

BASES = "TCAG"
AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
codons = ["".join(c) for c in itertools.product(BASES, repeat=3)]
code = dict(zip(codons, AAS))
residues = sorted(set(AAS) - {"*"})
sense = [c for c in codons if code[c] != "*"]

# all single-letter substitutions between sense codons
pairs = []
for c in sense:
    for i in range(3):
        for b in BASES:
            if b != c[i]:
                d = c[:i] + b + c[i + 1:]
                if code[d] != "*":
                    pairs.append((c, d))
src_idx = np.array([residues.index(code[a]) for a, _ in pairs])
dst_idx = np.array([residues.index(code[b]) for _, b in pairs])
kd = np.array([KD[r] for r in residues])


def cost(perm):
    v = kd[perm]
    return float(np.mean((v[src_idx] - v[dst_idx]) ** 2))


identity = np.arange(20)
real = cost(identity)
rng = np.random.default_rng(1991)
TRIALS = 200_000
random_costs = np.array([cost(rng.permutation(20)) for _ in range(TRIALS)])
better = int((random_costs <= real).sum())
synonymous = float(np.mean(src_idx == dst_idx))
per_res = {r: AAS.count(r) for r in residues}

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [0.7, 1.1, 1.2]})

ax = axes[0]
L = np.arange(1, 5)
ax.bar(L, 4 ** L, color=[MUTED, MUTED, BLUE, MUTED], width=0.6)
ax.axhline(21, color=ORANGE, ls="--", lw=1.1)
ax.text(0.6, 25, "20 residues + stop", color=INK2, fontsize=6.8)
ax.set_yscale("log"); ax.set_xticks(L)
ax.set_xlabel("word length $L$"); ax.set_ylabel("words $4^L$")
ax.set_title("Shortest sufficient word")
panel_label(ax, "A")

ax = axes[1]
srt = sorted(residues, key=lambda r: (-per_res[r], r))
ax.bar(range(20), [per_res[r] for r in srt], color=BLUE, width=0.7)
ax.set_xticks(range(20), srt, fontsize=7)
ax.set_ylabel("codons"); ax.set_xlabel("residue")
ax.set_title("61 sense words name 20 residues")
panel_label(ax, "B")

ax = axes[2]
ax.hist(random_costs, bins=120, color=MUTED)
ax.axvline(real, color=ORANGE, lw=1.6)
ax.annotate(f"standard code\n{better} of {TRIALS:,} permuted\ncodes do as well",
            (real, ax.get_ylim()[1] * 0.85), xytext=(8, 0), textcoords="offset points",
            fontsize=6.8, color=INK2, va="top")
ax.set_xlabel("mean squared hydropathy change per substitution")
ax.set_ylabel("permuted codes")
ax.set_title("Substitution tolerance of the code")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_P450, "g4_record")

hist_counts, hist_edges = np.histogram(random_costs, bins=80)
record("g4_record", {
    "raw": {"cost_hist_edges": np.round(hist_edges, 4).tolist(), "cost_hist_counts": hist_counts.tolist()},
    "min_word_length": int(min(l for l in range(1, 6) if 4 ** l >= 21)),
    "sense_codons": len(sense), "stop_codons": 64 - len(sense),
    "codons_per_residue": per_res,
    "single_letter_substitutions_between_sense_codons": len(pairs),
    "synonymous_fraction": synonymous,
    "standard_code_cost": real,
    "permuted_codes": TRIALS,
    "permuted_codes_at_or_below_standard": better,
    "fraction_at_or_below": better / TRIALS,
    "permuted_cost_mean_sd": [float(random_costs.mean()), float(random_costs.std())],
})
print("cost", real, "better", better, "of", TRIALS, "syn", synonymous)
