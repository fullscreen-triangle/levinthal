"""G1. Shell capacities, closures and the d shell of iron.

(a) From the label counting 2(2l+1) per subshell and the (n+l, n) filling
    order, predict the period lengths and the closed-shell atomic numbers.
(b) Against measured first ionization energies (NIST values carried by the
    mendeleev package): are the six largest drops IE(Z) -> IE(Z+1) located
    immediately after the predicted closures?
(c) The d shell: number of distinguishable occupations C(10, k), unpaired
    electrons in the high- and low-spin arrangements, with iron's oxidation
    states marked. Iron's ground configuration is read from the same data.

The filling order is an empirical rule (Madelung); it is used here as an
input and labelled as such in the manuscript.
"""
from __future__ import annotations

import sqlite3
from math import comb
from pathlib import Path

import mendeleev
import numpy as np

from common import BLUE, ORANGE, AQUA, INK2, MUTED, FIG_P450, panel_label, record, save_figure, plt

db = next(Path(mendeleev.__file__).parent.rglob("elements.db"))
con = sqlite3.connect(db)
ie = dict(con.execute("select atomic_number, ionization_energy from ionizationenergies "
                      "where ion_charge = 0 and ionization_energy is not null"))
fe_config = con.execute("select ground_configuration from ionizationenergies "
                        "where atomic_number = 26 and ion_charge = 0").fetchone()[0]
# ion_charge q is the species Fe^{q+} before ionization
fe2_config = con.execute("select ground_configuration from ionizationenergies "
                         "where atomic_number = 26 and ion_charge = 2").fetchone()[0]
fe3_config = con.execute("select ground_configuration from ionizationenergies "
                         "where atomic_number = 26 and ion_charge = 3").fetchone()[0]
period_of = dict(con.execute("select atomic_number, period from elements"))

# (a) subshells ordered by (n + l, n); each holds 2(2l + 1)
subshells = sorted(((n, l) for n in range(1, 8) for l in range(0, n)), key=lambda s: (s[0] + s[1], s[0]))
z, closures, periods, current = 0, [], [], 0
for n, l in subshells:
    if l == 0 and z > 0:            # a new s subshell opens a new period
        closures.append(z); periods.append(current); current = 0
    cap = 2 * (2 * l + 1)
    z += cap; current += cap
    if z >= 118:
        break
periods.append(current)
closures = [c for c in closures if c <= 118]
observed_periods = [sum(1 for zz, p in period_of.items() if p == k and zz <= 118) for k in range(1, 8)]

# (b) largest drops in first ionization energy
zs = np.array(sorted(ie))
e = np.array([ie[k] for k in zs])
drops = {int(zs[i]): float(e[i] - e[i + 1]) for i in range(len(zs) - 1) if zs[i + 1] == zs[i] + 1}
top6 = sorted(sorted(drops, key=drops.get, reverse=True)[:6])

# (c) d-shell counting
k = np.arange(0, 11)
occupations = np.array([comb(10, int(x)) for x in k])
high_spin = np.array([x if x <= 5 else 10 - x for x in k])
# low-spin octahedral: fill t2g (6) first, then eg (4), Hund's rule within each
def _unpaired(m, size):
    return m if m <= size // 2 else size - m
low_spin = np.array([_unpaired(min(x, 6), 6) + _unpaired(max(x - 6, 0), 4) for x in k])

fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.0), gridspec_kw={"width_ratios": [1.5, 0.8, 0.9, 0.8]})

ax = axes[0]
ax.plot(zs, e, color=BLUE, lw=1.0, marker="o", ms=2.2)
for c in closures:
    ax.axvline(c, color=MUTED, lw=0.7, ls="--")
    if c in ie:
        ax.annotate(str(c), (c, ie[c]), xytext=(2, 3), textcoords="offset points", fontsize=6.5, color=INK2)
ax.axvline(26, color=ORANGE, lw=1.0)
ax.annotate("Fe", (26, ie[26]), xytext=(3, -12), textcoords="offset points", fontsize=7, color=ORANGE)
ax.set_xlabel("atomic number $Z$"); ax.set_ylabel("first ionization energy (eV)")
ax.set_title("Counted closures (dashed), measured energies")
panel_label(ax, "A")

ax = axes[1]
xs = np.arange(1, 8)
ax.bar(xs - 0.18, periods, width=0.34, color=BLUE, label="counted")
ax.bar(xs + 0.18, observed_periods, width=0.34, color=AQUA, label="observed")
ax.set_xticks(xs); ax.set_xlabel("period"); ax.set_ylabel("elements")
ax.set_title("Period lengths")
ax.legend(loc="upper left")
panel_label(ax, "B")

ax = axes[2]
ax.bar(k, occupations, color=MUTED, width=0.7, label="occupations $\\binom{10}{k}$")
ax.set_xlabel("d electrons $k$"); ax.set_ylabel("distinguishable occupations")
for kk, lab in ((5, "Fe(III)"), (6, "Fe(II)")):
    ax.bar(kk, occupations[kk], color=ORANGE, width=0.7)
    ax.annotate(lab, (kk, occupations[kk]), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=6.5, color=INK2)
ax.set_ylim(0, 290)
ax.set_title("Occupations of the d shell")
panel_label(ax, "C")

ax = axes[3]
ax.plot(k, high_spin, color=ORANGE, marker="o", ms=3, label="high spin")
ax.plot(k, low_spin, color=BLUE, marker="o", ms=3, label="low spin (octahedral)")
for kk in (5, 6):
    ax.axvline(kk, color=MUTED, lw=0.7, ls=":")
ax.set_xlabel("d electrons $k$"); ax.set_ylabel("unpaired electrons")
ax.set_ylim(-0.2, 6.2)
ax.set_title("Two spin ladders")
ax.legend(loc="upper left")
panel_label(ax, "D")

fig.tight_layout()
save_figure(fig, FIG_P450, "g1_atoms")

record("g1_atoms", {
    "data_source": "mendeleev %s (ionization energies from NIST ASD)" % mendeleev.__version__,
    "predicted_closures": closures,
    "predicted_period_lengths": periods,
    "observed_period_lengths": observed_periods,
    "six_largest_ie_drops_after_Z": top6,
    "six_largest_ie_drops_eV": {z_: drops[z_] for z_ in top6},
    "noble_gas_closures_matched": sorted(set(top6) & set(closures)),
    "fe_ground_configuration": fe_config,
    "fe2_ion_ground_configuration": fe2_config,
    "fe3_ion_ground_configuration": fe3_config,
    "d_occupations": dict(zip(k.tolist(), occupations.tolist())),
    "unpaired_high_spin": dict(zip(k.tolist(), high_spin.tolist())),
    "unpaired_low_spin_octahedral": dict(zip(k.tolist(), low_spin.tolist())),
})
print("closures", closures, "periods", periods, observed_periods, "top6", top6, fe_config, fe2_config, fe3_config, low_spin.tolist())
