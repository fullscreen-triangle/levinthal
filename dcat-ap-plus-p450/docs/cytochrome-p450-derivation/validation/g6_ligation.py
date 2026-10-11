"""G6. The proximal ligand of the heme iron, and the sequence that places it.

(a) For every heme iron in the twenty entries: the nearest protein heavy
    atom, its residue type and its distance.
(b) The PROSITE signature PS00086 (pattern read from the PROSITE entry):
    fraction of reviewed UniProtKB P450-family sequences that match it,
    against reviewed heme-binding (keyword KW-0349) sequences outside the
    P450 family; and where in the sequence the matched cysteine lies.
(c) In each P450 structure: does the cysteine bound to the iron lie at the
    cysteine position of a signature match in the structure's own chain?
(d) The sixth (distal) ligand: water within 2.9 Å of the iron, and the
    iron's displacement from the pyrrole-nitrogen plane.
"""
from __future__ import annotations

import re

import numpy as np

from common import BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt
from structures import (OTHER_HEME_ENTRIES, P450_ENTRIES, THREE_TO_ONE, heme_sites, load,
                        prosite_pattern, uniprot_tsv)

pattern, rx = prosite_pattern("PS00086")
motif = re.compile(rx)
CYS_OFFSET = 7          # position of C in the 10-element pattern

# (a) and (c), (d)
sites, struct_check = [], {}
for pid, (lab, _) in {**P450_ENTRIES, **OTHER_HEME_ENTRIES}.items():
    s = load(pid)
    for st in heme_sites(s):
        sites.append({"pdb": pid, "label": lab, "p450": pid in P450_ENTRIES, **{k: v for k, v in st.items() if k != "fe_xyz"}})
    if pid in P450_ENTRIES:
        st = heme_sites(s)[0]
        res = s.residues(st["chain"])
        seq = "".join(THREE_TO_ONE[r[2]] for r in res)
        hits = [res[m.start() + CYS_OFFSET][0] for m in motif.finditer(seq)]
        struct_check[pid] = {"ligand_resseq": st["proximal_resseq"], "motif_cys_resseq": hits,
                             "ligand_in_motif": st["proximal_resseq"] in hits}

# (b)
pos = uniprot_tsv('family:"cytochrome P450 family" AND reviewed:true',
                  "accession,protein_name,organism_name,lineage,cc_subcellular_location,sequence",
                  "p450_reviewed")
neg = uniprot_tsv('keyword:KW-0349 AND reviewed:true AND NOT family:"cytochrome P450 family"',
                  "accession,protein_name,sequence", "heme_non_p450_reviewed")


def hit_positions(seq):
    return [m.start() + CYS_OFFSET for m in motif.finditer(seq)]


pos_hits = [hit_positions(r["Sequence"]) for r in pos]
neg_hits = [hit_positions(r["Sequence"]) for r in neg]
pos_frac = float(np.mean([bool(h) for h in pos_hits]))
neg_frac = float(np.mean([bool(h) for h in neg_hits]))
from_c_end = [len(r["Sequence"]) - h[-1] for r, h in zip(pos, pos_hits) if h]
rel_pos = [h[-1] / len(r["Sequence"]) for r, h in zip(pos, pos_hits) if h]
# a relaxed core, G-x-[RHKPT]-x-C-x-[GA], searched only in the last 160 residues
LOOSE = re.compile("G.[RHKPT].C.[GA]")
TAIL = 160
pos_loose = float(np.mean([bool(h) or bool(LOOSE.search(r["Sequence"][-TAIL:])) for r, h in zip(pos, pos_hits)]))
neg_loose = float(np.mean([bool(h) or bool(LOOSE.search(r["Sequence"][-TAIL:])) for r, h in zip(neg, neg_hits)]))
neg_examples = [r["Protein names"][:70] for r, h in zip(neg, neg_hits) if h][:15]

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.2, 1, 1]})

ax = axes[0]
groups = {}
for s_ in sites:
    key = f"{s_['proximal_resname']} {s_['proximal_atom']}"
    groups.setdefault(key, []).append(s_)
keys = sorted(groups, key=lambda k: -len(groups[k]))
rng = np.random.default_rng(0)
for i, k in enumerate(keys):
    for s_ in groups[k]:
        ax.scatter(i + rng.uniform(-0.18, 0.18), s_["proximal_distance"], s=16,
                   color=BLUE if s_["p450"] else ORANGE, edgecolor="white", lw=0.5, zorder=3)
ax.scatter([], [], s=16, color=BLUE, label="P450"); ax.scatter([], [], s=16, color=ORANGE, label="other heme proteins")
ax.set_xticks(range(len(keys)), [f"{k}\n(n={len(groups[k])})" for k in keys])
ax.set_ylabel("Fe to nearest protein atom (Å)")
ax.set_title("Proximal ligand of every heme iron")
ax.legend(loc="upper right")
panel_label(ax, "A")

ax = axes[1]
xb = np.array([0, 1])
ax.bar(xb - 0.19, [pos_frac * 100, neg_frac * 100], color=BLUE, width=0.36, label="PS00086")
ax.bar(xb + 0.19, [pos_loose * 100, neg_loose * 100], color=AQUA, width=0.36,
       label="PS00086 or relaxed core\nin last 160 residues")
ax.set_xticks(xb, [f"P450 family\n(n={len(pos)})", f"other heme\nproteins (n={len(neg)})"])
ax.set_ylabel("sequences matching (%)")
for x_, v in ((-0.19, pos_frac), (0.19, pos_loose), (0.81, neg_frac), (1.19, neg_loose)):
    ax.text(x_, v * 100 + 2, f"{v*100:.1f}", ha="center", fontsize=6.5, color=INK2)
ax.set_ylim(0, 125)
ax.legend(loc="upper right")
ax.set_title("Cysteine-loop signature")
panel_label(ax, "B")

ax = axes[2]
ax.hist(from_c_end, bins=np.arange(0, 160, 4), color=BLUE)
ax.set_xlabel("residues from matched Cys to the C-terminus")
ax.set_ylabel("P450 sequences")
ax.set_title("The ligand sits near the C-terminus")
ax.text(0.97, 0.95, f"median {np.median(from_c_end):.0f} residues\n"
        f"relative position {np.median(rel_pos):.2f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.8, color=INK2)
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_P450, "g6_ligation")

p450_sites = [s_ for s_ in sites if s_["p450"]]
record("g6_ligation", {
    "prosite_pattern": pattern, "regex": rx,
    "sites": sites,
    "ligand_summary": {k: {"n": len(v), "mean": float(np.mean([s_["proximal_distance"] for s_ in v])),
                           "min": float(min(s_["proximal_distance"] for s_ in v)),
                           "max": float(max(s_["proximal_distance"] for s_ in v)),
                           "entries": sorted({s_["pdb"] for s_ in v})} for k, v in groups.items()},
    "p450_all_cys_sg": all(s_["proximal_resname"] == "CYS" and s_["proximal_atom"] == "SG" for s_ in p450_sites),
    "structure_motif_check": struct_check,
    "uniprot": {"p450_reviewed": len(pos), "p450_match_fraction": pos_frac,
                "p450_without_match": [r["Entry"] + " " + r["Protein names"][:60]
                                       for r, h in zip(pos, pos_hits) if not h][:40],
                "p450_without_match_count": int(sum(1 for h in pos_hits if not h)),
                "relaxed_core": "G.[RHKPT].C.[GA] in last %d residues" % TAIL,
                "p450_strict_or_relaxed_fraction": pos_loose,
                "heme_non_p450_strict_or_relaxed_fraction": neg_loose,
                "heme_non_p450_reviewed": len(neg), "heme_non_p450_match_fraction": neg_frac,
                "heme_non_p450_match_examples": neg_examples,
                "cys_from_c_terminus_median": float(np.median(from_c_end)),
                "cys_from_c_terminus_iqr": np.percentile(from_c_end, [25, 75]).tolist(),
                "cys_relative_position_median": float(np.median(rel_pos))},
    "raw": {"cys_from_c_terminus_counts": np.histogram(from_c_end, bins=np.arange(0, 161, 4))[0].tolist(),
            "cys_from_c_terminus_bins": np.arange(0, 161, 4).tolist()},
    "distal": {s_["pdb"]: {"water_distance": s_["distal_water_distance"],
                           "fe_out_of_plane": s_["fe_out_of_plane"],
                           "fe_pyrrole_n_mean": s_["fe_pyrrole_n_mean"]}
               for s_ in p450_sites},
})
print("pattern", pattern, "| P450", pos_frac, pos_loose, len(pos), "| other", neg_frac, neg_loose, len(neg))
print(struct_check)
print(neg_examples[:8])
