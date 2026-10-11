"""G2. Bond geometry of the peptide carrier in twelve P450 structures.

For the heme-bearing chain of each P450 entry: the peptide C-N bond against
the single N-CA and CA-C bonds; the tetrahedral N-CA-CB angle; the
peptide dihedral omega; and the backbone dihedrals phi, psi.

Bond lengths and angles are restrained during crystallographic refinement,
so their agreement with ideal values is not independent evidence; they are
shown to fix the geometry the later sections use. omega is restrained only
weakly and phi, psi are not restrained.
"""
from __future__ import annotations

import numpy as np

from common import BLUE, ORANGE, AQUA, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt
from structures import P450_ENTRIES, THREE_TO_ONE, heme_sites, load


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))


omega, phi, psi, cn, nca, cac, ncacb = [], [], [], [], [], [], []
per_entry = {}
for pid in P450_ENTRIES:
    s = load(pid)
    chain = heme_sites(s)[0]["chain"]
    res = {}
    for a in s.atoms:
        if a.chain == chain and not a.hetero and a.resname in THREE_TO_ONE:
            res.setdefault((a.resseq, a.icode), {})[a.name] = a.xyz
    keys = sorted(res)
    n_om = 0
    for i, k in enumerate(keys):
        r = res[k]
        if not {"N", "CA", "C"} <= r.keys():
            continue
        nca.append(np.linalg.norm(r["N"] - r["CA"])); cac.append(np.linalg.norm(r["CA"] - r["C"]))
        if "CB" in r:
            ncacb.append(angle(r["N"], r["CA"], r["CB"]))
        if i + 1 < len(keys):
            q = res[keys[i + 1]]
            if {"N", "CA", "C"} <= q.keys() and np.linalg.norm(r["C"] - q["N"]) < 1.6:
                cn.append(np.linalg.norm(r["C"] - q["N"]))
                omega.append(dihedral(r["CA"], r["C"], q["N"], q["CA"])); n_om += 1
                psi.append(dihedral(r["N"], r["CA"], r["C"], q["N"]))
                phi.append(dihedral(r["C"], q["N"], q["CA"], q["C"]))
    per_entry[pid] = {"chain": chain, "peptide_bonds": n_om}

omega = np.array(omega); phi = np.array(phi); psi = np.array(psi)
trans = np.abs(np.abs(omega) - 180) <= 20
cis = np.abs(omega) <= 30

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

ax = axes[0]
bins = np.linspace(1.20, 1.62, 70)
ax.hist(cn, bins=bins, color=BLUE, alpha=0.85, label="peptide C$-$N")
ax.hist(nca, bins=bins, color=ORANGE, alpha=0.7, label="N$-$C$\\alpha$")
ax.hist(cac, bins=bins, color=AQUA, alpha=0.7, label="C$\\alpha-$C")
ax.set_xlabel("bond length (Å)"); ax.set_ylabel("bonds")
ax.set_title("The peptide bond is shorter")
ax.legend(loc="upper right")
panel_label(ax, "A")

ax = axes[1]
w = np.where(omega < 0, omega + 360, omega)
ax.hist(w, bins=np.arange(0, 361, 4), color=BLUE)
ax.set_yscale("log")
ax.set_xlabel("peptide dihedral $\\omega$ (°, 0–360)"); ax.set_ylabel("peptide bonds")
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_title("Planar: trans or cis, nothing between")
ax.text(0.03, 0.95, f"{trans.mean()*100:.1f}% within 20° of trans\n{int(cis.sum())} cis of {omega.size}",
        transform=ax.transAxes, va="top", fontsize=6.8, color=INK2)
panel_label(ax, "B")

ax = axes[2]
h = ax.hist2d(phi, psi, bins=72, range=[[-180, 180], [-180, 180]], cmap="Blues", cmin=1)
ax.set_xlabel("$\\phi$ (°)"); ax.set_ylabel("$\\psi$ (°)")
ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
ax.grid(False)
ax.set_title("Two backbone dihedrals carry the fold")
panel_label(ax, "C")

fig.tight_layout()
save_figure(fig, FIG_P450, "g2_backbone")

helix = (phi > -100) & (phi < -30) & (psi > -80) & (psi < -10)
sheet = (phi > -180) & (phi < -45) & ((psi > 90) | (psi < -150))
record("g2_backbone", {
    "entries": per_entry,
    "peptide_bonds": int(omega.size),
    "peptide_CN_mean_sd": [float(np.mean(cn)), float(np.std(cn))],
    "N_CA_mean_sd": [float(np.mean(nca)), float(np.std(nca))],
    "CA_C_mean_sd": [float(np.mean(cac)), float(np.std(cac))],
    "N_CA_CB_angle_mean_sd": [float(np.mean(ncacb)), float(np.std(ncacb))],
    "tetrahedral_angle": 109.47,
    "omega_trans_within_20deg_fraction": float(trans.mean()),
    "omega_cis_count": int(cis.sum()),
    "omega_intermediate_count": int((~trans & ~cis).sum()),
    "phi_psi_helical_fraction": float(helix.mean()),
    "phi_psi_extended_fraction": float(sheet.mean()),
})
print(omega.size, trans.mean(), int(cis.sum()), np.mean(cn), np.mean(nca), np.mean(ncacb))
