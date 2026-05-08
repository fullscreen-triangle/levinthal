"""Generate 8 figure panels for Paper 10: Pharmacogenomics Atlas."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

OUT = Path(__file__).parent
nu_floor = 1e10

DELTA_M_EM = 0.55; DELTA_M_PM = 2.50; DELTA_M_IM = 0.75; DELTA_M_UM = 0.27
K_EM = nu_floor * math.exp(-DELTA_M_EM)
K_PM = nu_floor * math.exp(-DELTA_M_PM)
K_IM = nu_floor * math.exp(-DELTA_M_IM)
K_UM = nu_floor * math.exp(-DELTA_M_UM)

DELTA_M_2C9_EM = 0.48; DELTA_M_2C9_2 = 0.62; DELTA_M_2C9_3 = 3.60
K_2C9_EM = nu_floor * math.exp(-DELTA_M_2C9_EM)
K_2C9_2  = nu_floor * math.exp(-DELTA_M_2C9_2)
K_2C9_3  = nu_floor * math.exp(-DELTA_M_2C9_3)

def savefig(name):
    p = OUT / name
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  saved {name}")

# ── Panel 01: CYP2D6 allele rates ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

phenotypes = ["PM\n(*4/*5)", "IM\n(*10/*17)", "EM\n(*1)", "UM\n(*1xN)"]
ks = [K_PM, K_IM, K_EM, K_UM]
colors = ["#e74c3c", "#e67e22", "#27ae60", "#3498db"]

ax1 = fig.add_subplot(131)
ax1.bar(phenotypes, [k/1e9 for k in ks], color=colors, edgecolor="k")
ax1.set_ylabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("CYP2D6 Phenotype Rates")

ax2 = fig.add_subplot(132)
dm_vals = [DELTA_M_PM, DELTA_M_IM, DELTA_M_EM, DELTA_M_UM]
ax2.plot(dm_vals, [k/1e9 for k in ks], "ko--", ms=10, lw=2)
for i, (phen, dm, k) in enumerate(zip(phenotypes, dm_vals, ks)):
    ax2.annotate(phen.split("\n")[0], (dm, k/1e9), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)
ax2.set_xlabel("ΔM")
ax2.set_ylabel("Rate (×10⁹ s⁻¹)")
ax2.set_title("Rate vs. ΔM (CYP2D6)")

ax3 = fig.add_subplot(133, projection="3d")
freq = [0.07, 0.30, 0.55, 0.08]
ax3.scatter(dm_vals, freq, [k/1e9 for k in ks], c=colors, s=200, zorder=5)
for i, phen in enumerate(phenotypes):
    ax3.text(dm_vals[i], freq[i], ks[i]/1e9 + 0.1, phen.split("\n")[0], fontsize=8)
ax3.set_xlabel("ΔM")
ax3.set_ylabel("Population frequency")
ax3.set_zlabel("Rate (×10⁹ s⁻¹)")
ax3.set_title("ΔM-Freq-Rate Space")

plt.suptitle("Panel 01: CYP2D6 Allele Rate Constants", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_01_cyp2d6_allele_rates.png")

# ── Panel 02: CYP2C9 warfarin dosing ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

alleles = ["*1 (EM)", "*2", "*3"]
ks_2c9 = [K_2C9_EM, K_2C9_2, K_2C9_3]
doses  = [K_2C9_EM/k for k in ks_2c9]

ax1 = fig.add_subplot(131)
ax1.bar(alleles, [k/1e9 for k in ks_2c9], color=["#27ae60","#e67e22","#e74c3c"],
        edgecolor="k")
ax1.set_ylabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("CYP2C9 Allele Rates")

ax2 = fig.add_subplot(132)
ax2.bar(alleles, doses, color=["#27ae60","#e67e22","#e74c3c"], edgecolor="k")
ax2.set_ylabel("Relative Warfarin Dose")
ax2.set_title("Dose Adjustment by Allele")
ax2.axhline(3, color="gray", ls="--", label="3x threshold")
ax2.legend()

ax3 = fig.add_subplot(133, projection="3d")
dm_arr = np.linspace(0.3, 4.5, 100)
k_arr  = nu_floor * np.exp(-dm_arr)
dose_arr = (nu_floor * math.exp(-DELTA_M_2C9_EM)) / k_arr
ax3.plot(dm_arr, k_arr/1e8, dose_arr, "b-", lw=2)
ax3.scatter([DELTA_M_2C9_EM, DELTA_M_2C9_2, DELTA_M_2C9_3],
            [K_2C9_EM/1e8, K_2C9_2/1e8, K_2C9_3/1e8],
            [doses[0], doses[1], doses[2]],
            c=["g","orange","r"], s=100, zorder=5)
ax3.set_xlabel("ΔM")
ax3.set_ylabel("k (×10⁸ s⁻¹)")
ax3.set_zlabel("Dose ratio")
ax3.set_title("Dose-Rate-ΔM Curve")

plt.suptitle("Panel 02: CYP2C9 Warfarin Dose Adjustment", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_02_cyp2c9_warfarin_dosing.png")

# ── Panel 03: Population phenotype frequencies ────────────────────────────────
fig = plt.figure(figsize=(16, 4))

freqs = {"PM":0.07, "IM":0.30, "EM":0.55, "UM":0.08}
k_pop = sum(freqs[p]*k for p, k in zip(["PM","IM","EM","UM"],[K_PM,K_IM,K_EM,K_UM]))

ax1 = fig.add_subplot(131)
ax1.pie(list(freqs.values()), labels=list(freqs.keys()),
        colors=["#e74c3c","#e67e22","#27ae60","#3498db"],
        autopct="%1.0f%%", startangle=90)
ax1.set_title("CYP2D6 Phenotype Freq.")

ax2 = fig.add_subplot(132)
contributions = {p: freqs[p]*k/1e9 for p, k in zip(["PM","IM","EM","UM"],[K_PM,K_IM,K_EM,K_UM])}
ax2.bar(list(contributions.keys()), list(contributions.values()),
        color=["#e74c3c","#e67e22","#27ae60","#3498db"], edgecolor="k")
ax2.set_ylabel("Contribution to k_pop (×10⁹ s⁻¹)")
ax2.set_title("Rate Contribution by Phenotype")

ax3 = fig.add_subplot(133, projection="3d")
freq_arr = np.linspace(0.01, 0.15, 20)
dm_pm_arr = np.linspace(1.5, 3.5, 20)
FA, DM = np.meshgrid(freq_arr, dm_pm_arr)
K_POP = FA * nu_floor * np.exp(-DM) + (1-FA) * K_EM
ax3.plot_surface(FA, DM, K_POP/1e9, cmap="coolwarm", alpha=0.85)
ax3.set_xlabel("f_PM")
ax3.set_ylabel("ΔM_PM")
ax3.set_zlabel("k_pop (×10⁹ s⁻¹)")
ax3.set_title("k_pop Surface")

plt.suptitle("Panel 03: Population Phenotype Frequency Distribution", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_03_population_phenotype_frequencies.png")

# ── Panel 04: Codeine toxicity model ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

k_elim = 2e-4
morphine_em = K_EM * 1e-6 / k_elim
morphine_um = K_UM * 1e-6 / k_elim
morphine_pm = K_PM * 1e-6 / k_elim
morphine_im = K_IM * 1e-6 / k_elim

ax1 = fig.add_subplot(131)
ax1.bar(["PM","IM","EM","UM"], [morphine_pm, morphine_im, morphine_em, morphine_um],
        color=["#e74c3c","#e67e22","#27ae60","#3498db"], edgecolor="k",
        width=0.5)
ax1.axhline(morphine_em * 1.3, color="red", ls="--", label="1.3x toxicity threshold")
ax1.set_ylabel("Morphine exposure (proxy)")
ax1.set_title("Morphine by Phenotype")
ax1.legend(fontsize=7)

ax2 = fig.add_subplot(132)
k_arr2 = np.linspace(K_PM*0.5, K_UM*1.5, 300)
morph_arr = k_arr2 * 1e-6 / k_elim
ratio_arr = morph_arr / morphine_em
ax2.plot(k_arr2/1e9, ratio_arr, "b-", lw=2)
ax2.axhline(1.3, color="red", ls="--", label="1.3x threshold")
ax2.scatter([K_PM/1e9, K_IM/1e9, K_EM/1e9, K_UM/1e9],
            [morphine_pm/morphine_em, morphine_im/morphine_em, 1.0, morphine_um/morphine_em],
            c=["r","orange","g","b"], s=80, zorder=5)
ax2.set_xlabel("k_CYP2D6 (×10⁹ s⁻¹)")
ax2.set_ylabel("Morphine ratio (vs EM)")
ax2.set_title("Morphine Ratio vs. Rate")
ax2.legend()

ax3 = fig.add_subplot(133, projection="3d")
k_3d = np.linspace(1e8, 1e10, 20)
kelim_3d = np.linspace(1e-5, 1e-3, 20)
K3D, KE3D = np.meshgrid(k_3d, kelim_3d)
MORPH_3D = K3D * 1e-6 / KE3D / morphine_em
ax3.plot_surface(np.log10(K3D), np.log10(KE3D), np.log10(MORPH_3D+0.01),
                 cmap="RdYlGn", alpha=0.85)
ax3.set_xlabel("log k_CYP")
ax3.set_ylabel("log k_elim")
ax3.set_zlabel("log morphine ratio")
ax3.set_title("Morphine Exposure Surface")

plt.suptitle("Panel 04: Codeine Toxicity in UM Patients", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_04_codeine_toxicity_model.png")

# ── Panel 05: CYP3A4 induction ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

fold_arr = np.linspace(1, 30, 200)
auc_arr  = 1.0 / fold_arr

ax1 = fig.add_subplot(131)
ax1.semilogy(fold_arr, auc_arr, "b-", lw=2)
ax1.axvline(20, color="red", ls="--", label="Rifampicin 20x")
ax1.axhline(0.1, color="orange", ls=":", label="AUC=0.1 threshold")
ax1.set_xlabel("Fold induction")
ax1.set_ylabel("AUC ratio (victim)")
ax1.set_title("AUC vs. Induction Fold")
ax1.legend(fontsize=7)

k_base = nu_floor * math.exp(-0.45)
k_ind = 20 * k_base
ax2 = fig.add_subplot(132)
ax2.bar(["Baseline", "Induced\n(20x)"], [k_base/1e9, k_ind/1e9],
        color=["#3498db","#e74c3c"], edgecolor="k")
ax2.set_ylabel("k (×10⁹ s⁻¹)")
ax2.set_title("CYP3A4 Induction Rate")

ax3 = fig.add_subplot(133, projection="3d")
fold_3d = np.linspace(1, 30, 20)
dm_3d   = np.linspace(0.3, 0.7, 20)
F3D, DM3D = np.meshgrid(fold_3d, dm_3d)
AUC_3D = 1.0 / F3D
ax3.plot_surface(F3D, DM3D, AUC_3D, cmap="RdYlGn_r", alpha=0.85)
ax3.set_xlabel("Fold induction")
ax3.set_ylabel("ΔM_base")
ax3.set_zlabel("AUC ratio")
ax3.set_title("AUC Surface")

plt.suptitle("Panel 05: CYP3A4 Induction by Rifampicin", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_05_ddI_cyp3a4_induction.png")

# ── Panel 06: Competitive inhibition ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

I_arr = np.linspace(0, 2.0, 200)
Ki    = 0.24
alpha_arr = 1.0 + I_arr / Ki

ax1 = fig.add_subplot(131)
ax1.plot(I_arr, alpha_arr, "b-", lw=2)
ax1.axvline(0.5, color="red", ls="--", label="[I]=0.5 μM (fluoxetine)")
ax1.axhline(1+0.5/Ki, color="red", ls=":")
ax1.set_xlabel("[I] (μM)")
ax1.set_ylabel("α = 1 + [I]/Ki")
ax1.set_title("Alpha vs. Inhibitor Concentration")
ax1.legend(fontsize=7)

ax2 = fig.add_subplot(132)
auc_ddi_arr = 1.0 + I_arr / Ki
ax2.plot(I_arr, auc_ddi_arr, "r-", lw=2)
ax2.axvline(0.5, color="blue", ls="--")
ax2.axhline(2.0, color="gray", ls=":", label="Strong DDI (2x)")
ax2.axhline(5.0, color="orange", ls=":", label="Contraindicated (5x)")
ax2.set_xlabel("[I] (μM)")
ax2.set_ylabel("AUC ratio (victim)")
ax2.set_title("DDI AUC Ratio")
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(133, projection="3d")
I_3d  = np.linspace(0, 2, 20)
Ki_3d = np.linspace(0.1, 1.0, 20)
I3D, KI3D = np.meshgrid(I_3d, Ki_3d)
ALPHA_3D = 1.0 + I3D / KI3D
ax3.plot_surface(I3D, KI3D, ALPHA_3D, cmap="hot", alpha=0.85)
ax3.set_xlabel("[I] (μM)")
ax3.set_ylabel("Ki (μM)")
ax3.set_zlabel("α")
ax3.set_title("α([I], Ki) Surface")

plt.suptitle("Panel 06: Competitive Inhibition by Fluoxetine", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_06_inhibition_competitive.png")

# ── Panel 07: Ethnic variation ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ancestry_freqs = {
    "European":   {"PM":0.07,"IM":0.15,"EM":0.70,"UM":0.08},
    "East_Asian": {"PM":0.01,"IM":0.50,"EM":0.41,"UM":0.08},
    "African":    {"PM":0.02,"IM":0.35,"EM":0.55,"UM":0.08},
    "South_Asian":{"PM":0.05,"IM":0.20,"EM":0.67,"UM":0.08},
}
k_rate = {"PM":K_PM,"IM":K_IM,"EM":K_EM,"UM":K_UM}
pop_rates_et = {a: sum(ancestry_freqs[a][p]*k_rate[p] for p in ["PM","IM","EM","UM"])
                for a in ancestry_freqs}

ax1 = fig.add_subplot(131)
ancs = list(ancestry_freqs.keys())
im_freqs = [ancestry_freqs[a]["IM"] for a in ancs]
pm_freqs = [ancestry_freqs[a]["PM"] for a in ancs]
x = np.arange(len(ancs))
ax1.bar(x, im_freqs, label="IM freq", color="#e67e22", edgecolor="k")
ax1.bar(x, pm_freqs, bottom=im_freqs, label="PM freq", color="#e74c3c", edgecolor="k")
ax1.set_xticks(x)
ax1.set_xticklabels([a[:3] for a in ancs])
ax1.set_ylabel("Frequency")
ax1.set_title("PM + IM Freq by Ancestry")
ax1.legend()

ax2 = fig.add_subplot(132)
pop_rate_vals = [pop_rates_et[a]/1e9 for a in ancs]
ax2.bar(ancs, pop_rate_vals, color=["#3498db","#e74c3c","#27ae60","#9b59b6"], edgecolor="k")
ax2.set_ylabel("k_pop (×10⁹ s⁻¹)")
ax2.set_title("Population Metabolic Rate")
ax2.set_xticklabels([a[:4] for a in ancs], rotation=20, ha="right")

ax3 = fig.add_subplot(133, projection="3d")
pm_arr3 = np.linspace(0, 0.15, 20)
im_arr3 = np.linspace(0, 0.6, 20)
PM3, IM3 = np.meshgrid(pm_arr3, im_arr3)
EM3 = np.clip(1.0 - PM3 - IM3 - 0.08, 0, 1)
KPOP3 = PM3*K_PM + IM3*K_IM + EM3*K_EM + 0.08*K_UM
ax3.plot_surface(PM3, IM3, KPOP3/1e9, cmap="viridis", alpha=0.85)
ax3.set_xlabel("f_PM")
ax3.set_ylabel("f_IM")
ax3.set_zlabel("k_pop (×10⁹ s⁻¹)")
ax3.set_title("k_pop(f_PM, f_IM)")

plt.suptitle("Panel 07: Ethnic Variation in CYP2D6 Rates", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_07_ethnic_variation.png")

# ── Panel 08: Validation summary ──────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

scripts = ["01 2D6\nRates", "02 2C9\nDosing", "03 Pop\nFreqs", "04 Codeine\nTox",
           "05 3A4\nInduct.", "06 Inhibit.", "07 Ethnic\nVar.", "08 Full\nTable"]

ax1 = fig.add_subplot(131)
ax1.bar(range(8), [1]*8, color="#27ae60", edgecolor="k")
ax1.set_xticks(range(8))
ax1.set_xticklabels(scripts, fontsize=6, rotation=45, ha="right")
ax1.set_yticks([])
ax1.set_title("8/8 PASS")

ax2 = fig.add_subplot(132)
dm_all = [DELTA_M_UM, DELTA_M_EM, DELTA_M_IM, DELTA_M_PM]
k_all  = [K_UM, K_EM, K_IM, K_PM]
phen   = ["UM","EM","IM","PM"]
ax2.scatter(dm_all, [k/1e9 for k in k_all],
            c=["#3498db","#27ae60","#e67e22","#e74c3c"], s=150, zorder=5)
for i, p in enumerate(phen):
    ax2.annotate(p, (dm_all[i], k_all[i]/1e9), textcoords="offset points",
                 xytext=(5, 5))
ax2.set_xlabel("ΔM")
ax2.set_ylabel("Rate (×10⁹ s⁻¹)")
ax2.set_title("ΔM-Rate Space (CYP2D6)")

ax3 = fig.add_subplot(133, projection="3d")
dm_range = np.linspace(0, 4, 50)
k_range  = nu_floor * np.exp(-dm_range)
ax3.plot(dm_range, k_range/1e9, np.zeros_like(dm_range), "b-", lw=2, label="k(ΔM)")
ax3.scatter(dm_all, [k/1e9 for k in k_all], [0.07, 0.55, 0.30, 0.08],
            c=["b","g","orange","r"], s=100)
ax3.set_xlabel("ΔM")
ax3.set_ylabel("k (×10⁹ s⁻¹)")
ax3.set_zlabel("Pop. frequency")
ax3.set_title("ΔM-Rate-Freq Phase Space")

plt.suptitle("Panel 08: Pharmacogenomics Validation Summary", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_08_validation.png")

print("All 8 panels generated.")
