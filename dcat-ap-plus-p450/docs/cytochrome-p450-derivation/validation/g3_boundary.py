"""G3. The N-terminal segment of P450s and the bilayer.

Hypothesis tested: P450s whose annotated location is the endoplasmic
reticulum membrane carry an N-terminal segment hydrophobic enough to span a
bilayer, and soluble bacterial P450s do not.

Measure: the largest mean Kyte-Doolittle hydropathy over 19-residue windows
lying within the first 60 residues. A value above 1.6 is the conventional
indicator of a membrane-spanning segment (Kyte & Doolittle 1982).

Data: all reviewed UniProtKB entries in the cytochrome P450 family, with
their taxonomic lineage and annotated subcellular location; and eleven named
P450s for per-residue profiles.
"""
from __future__ import annotations

import numpy as np

from common import BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt
from scales import KD
from structures import uniprot_fasta, uniprot_tsv

WIN, SPAN, THRESH = 19, 60, 1.6


def profile(seq):
    v = np.array([KD.get(c, 0.0) for c in seq])
    return np.convolve(v, np.ones(WIN) / WIN, mode="valid")


def nterm_max(seq):
    p = profile(seq[:SPAN])
    return float(p.max()), int(p.argmax()) + 1


rows = uniprot_tsv('family:"cytochrome P450 family" AND reviewed:true',
                   "accession,protein_name,organism_name,lineage,cc_subcellular_location,sequence",
                   "p450_reviewed")


def classify(r):
    lin = r.get("Taxonomic lineage", "")
    loc = r.get("Subcellular location [CC]", "")
    if "Bacteria (domain)" in lin:
        return "bacterial"
    if "Eukaryota" not in lin:
        return None
    if "Mitochondrion" in loc:
        return "mitochondrial"
    if "Endoplasmic reticulum membrane" in loc or "Microsome membrane" in loc:
        return "ER membrane"
    return None


groups = {"ER membrane": [], "mitochondrial": [], "bacterial": []}
for r in rows:
    g = classify(r)
    seq = r.get("Sequence", "")
    if g and len(seq) >= SPAN:
        groups[g].append(nterm_max(seq)[0])

named = {
    "P08684": "CYP3A4", "P11712": "CYP2C9", "P10635": "CYP2D6", "P05177": "CYP1A2",
    "P05181": "CYP2E1", "Q16850": "CYP51A1", "P11511": "CYP19A1", "P05093": "CYP17A1",
    "P05108": "CYP11A1", "P00183": "CYP101A1", "P14779": "CYP102A1",
}
named_out = {}
for acc, lab in named.items():
    head, seq = uniprot_fasta(acc)
    m, at = nterm_max(seq)
    named_out[lab] = {"accession": acc, "header": head[:120], "length": len(seq),
                      "nterm_max_hydropathy": m, "window_start": at}

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]
x = np.arange(1, SPAN - WIN + 2) + WIN // 2
for lab, c in (("CYP3A4", BLUE), ("CYP2C9", AQUA), ("CYP11A1", YELLOW), ("CYP101A1", ORANGE), ("CYP102A1", INK2)):
    acc = next(a for a, l in named.items() if l == lab)
    _, seq = uniprot_fasta(acc)
    ax.plot(x, profile(seq[:SPAN]), color=c, label=lab)
ax.axhline(THRESH, color=MUTED, ls="--", lw=1)
ax.text(SPAN - 9, THRESH + 0.1, "1.6", color=INK2, fontsize=6.8)
ax.set_xlabel("residue (window centre)"); ax.set_ylabel("mean hydropathy, 19-residue window")
ax.set_title("N-terminal hydropathy profiles")
ax.legend(loc="lower left", ncol=2)
panel_label(ax, "A")

ax = axes[1]
order = ["ER membrane", "mitochondrial", "bacterial"]
cols = [BLUE, YELLOW, ORANGE]
data = [groups[g] for g in order]
parts = ax.violinplot(data, showextrema=False, widths=0.8)
for b, c in zip(parts["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor("none")
for i, d in enumerate(data, start=1):
    ax.plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color=INK2, lw=1.4)
ax.axhline(THRESH, color=MUTED, ls="--", lw=1)
ax.set_xticks([1, 2, 3], [f"{g}\n(n={len(groups[g])})" for g in order])
ax.set_ylabel("largest window mean, residues 1–60")
ax.set_title("Reviewed P450s by location")
panel_label(ax, "B")

fig.tight_layout()
save_figure(fig, FIG_P450, "g3_boundary")

summary = {}
for g in order:
    d = np.array(groups[g])
    summary[g] = {"n": int(d.size), "median": float(np.median(d)),
                  "fraction_above_1.6": float((d > THRESH).mean())}
profiles = {}
for lab in ("CYP3A4", "CYP2C9", "CYP11A1", "CYP101A1", "CYP102A1"):
    acc = next(a for a, l in named.items() if l == lab)
    _, seq = uniprot_fasta(acc)
    profiles[lab] = [round(float(v), 3) for v in profile(seq[:SPAN])]
record("g3_boundary", {"window": WIN, "span": SPAN, "threshold": THRESH,
                       "reviewed_p450_entries": len(rows), "groups": summary,
                       "named": named_out,
                       "raw": {"values": {g: [round(v, 3) for v in groups[g]] for g in order},
                               "profiles": profiles, "profile_x0": int(x[0])}})
print(summary)
for k, v in named_out.items():
    print(k, round(v["nterm_max_hydropathy"], 2), v["window_start"])
