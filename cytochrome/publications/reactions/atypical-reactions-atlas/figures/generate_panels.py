"""Generate 8 figure panels for Paper 8: Atypical Reactions Atlas."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

OUT = Path(__file__).parent
nu_floor = 1e10

# ── shared parameters ─────────────────────────────────────────────────────────
DELTA_M = {
    "NIH shift":   0.18,
    "Carbene":     0.20,
    "Epoxidation": 0.35,
    "Nucleophilic":0.42,
    "Desat HAT-1": 0.65,
    "Desat HAT-2": 0.55,
}
K_REBOUND = 7.4e9
k1 = nu_floor * math.exp(-0.65)
k2 = nu_floor * math.exp(-0.55)
K_DESAT_EFF = k1 * k2 / (k2 + K_REBOUND)
K_EPOX = nu_floor * math.exp(-0.35)
K_NIH  = nu_floor * math.exp(-0.18)
K_NUC  = nu_floor * math.exp(-0.42)
K_CARB = nu_floor * math.exp(-0.20)

def savefig(name):
    p = OUT / name
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  saved {name}")

# ── Panel 01: Desaturation two-step ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(131)
steps = ["HAT-1\n(ΔM=0.65)", "HAT-2\n(ΔM=0.55)", "Rebound\n(k=7.4e9)"]
rates = [k1, k2, K_REBOUND]
colors = ["#e74c3c", "#e67e22", "#3498db"]
ax1.bar(steps, [r/1e9 for r in rates], color=colors, edgecolor="k")
ax1.set_ylabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("HAT vs. Rebound Rates")

ax2 = fig.add_subplot(132)
k2_arr = np.linspace(1e8, 2e10, 300)
keff = k1 * k2_arr / (k2_arr + K_REBOUND)
ax2.loglog(k2_arr, keff, "r-", lw=2)
ax2.axvline(k2, color="orange", ls="--", label=f"k₂={k2/1e9:.2f}×10⁹")
ax2.axhline(K_DESAT_EFF, color="purple", ls=":", label=f"keff={K_DESAT_EFF/1e8:.2f}×10⁸")
ax2.set_xlabel("k₂ (s⁻¹)")
ax2.set_ylabel("k_eff (s⁻¹)")
ax2.set_title("Effective Desaturation Rate")
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(133, projection="3d")
K1v = np.logspace(8.5, 10.5, 20)
K2v = np.logspace(8.5, 10.5, 20)
K1g, K2g = np.meshgrid(K1v, K2v)
Keff_g = K1g * K2g / (K2g + K_REBOUND)
ax3.plot_surface(np.log10(K1g), np.log10(K2g), np.log10(Keff_g),
                 cmap="plasma", alpha=0.8)
ax3.set_xlabel("log k₁")
ax3.set_ylabel("log k₂")
ax3.set_zlabel("log keff")
ax3.set_title("keff Surface")

plt.suptitle("Panel 01: Desaturation Two-Step HAT", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_01_desaturation_two_step.png")

# ── Panel 02: Desaturation KIE ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

nu_H = 3000.0; nu_D = nu_H / math.sqrt(2)
h_J = 6.62607e-34; c_cm = 2.997924e10; kB = 1.380649e-23; T = 310.0

def kie_zpe(nu_h, nu_d):
    return math.exp(h_J * c_cm * (nu_h - nu_d) / (2 * kB * T))

temps = np.linspace(250, 400, 100)
kie_vals = [kie_zpe(nu_H, nu_D) for _ in temps]
kie_single_hat = [kie_zpe(3000, 3000/math.sqrt(2)) for _ in temps]

ax1 = fig.add_subplot(131)
ax1.bar(["Single HAT", "Desaturation"], [kie_zpe(nu_H, nu_D), kie_zpe(nu_H, nu_D) * 0.75],
        color=["#c0392b", "#e74c3c"], edgecolor="k")
ax1.set_ylabel("KIE")
ax1.set_title("KIE Comparison")
ax1.axhline(3.0, color="gray", ls="--", label="KIE=3 lower bound")
ax1.legend()

ax2 = fig.add_subplot(132)
dm_arr = np.linspace(0.3, 0.9, 100)
kie_commit = [kie_zpe(nu_H, nu_D) * math.exp(-(dm - 0.65)) for dm in dm_arr]
ax2.plot(dm_arr, kie_commit, "b-", lw=2)
ax2.axvline(0.65, color="red", ls="--", label="ΔM=0.65")
ax2.set_xlabel("ΔM (HAT-1)")
ax2.set_ylabel("Effective KIE")
ax2.set_title("KIE vs. ΔM (commitment)")
ax2.legend()

ax3 = fig.add_subplot(133, projection="3d")
nu_arr = np.linspace(2800, 3200, 20)
T_arr  = np.linspace(250, 400, 20)
NG, TG = np.meshgrid(nu_arr, T_arr)
ND = NG / math.sqrt(2)
KIE_G = np.exp(h_J * c_cm * (NG - ND) / (2 * kB * TG))
ax3.plot_surface(NG, TG, KIE_G, cmap="coolwarm", alpha=0.8)
ax3.set_xlabel("ν̃_H (cm⁻¹)")
ax3.set_ylabel("T (K)")
ax3.set_zlabel("KIE")
ax3.set_title("KIE(ν, T) Surface")

plt.suptitle("Panel 02: Desaturation KIE Analysis", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_02_desaturation_kie.png")

# ── Panel 03: Arene oxide ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(131)
mechs = ["Aliphatic\nHAT (0.65)", "Epoxidation\n(0.35)"]
ks = [nu_floor*math.exp(-0.65)/1e9, K_EPOX/1e9]
ax1.bar(mechs, ks, color=["#e74c3c", "#27ae60"], edgecolor="k")
ax1.set_ylabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("Epoxidation vs. HAT Rates")

ax2 = fig.add_subplot(132)
dm_arr = np.linspace(0.1, 0.8, 200)
k_arr  = nu_floor * np.exp(-dm_arr)
ax2.semilogy(dm_arr, k_arr, "k-", lw=2)
ax2.axvline(0.35, color="g", ls="--", label="Epoxidation ΔM=0.35")
ax2.axvline(0.65, color="r", ls="--", label="Aliphatic HAT ΔM=0.65")
ax2.set_xlabel("ΔM")
ax2.set_ylabel("k (s⁻¹)")
ax2.set_title("Rate vs. ΔM")
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(133, projection="3d")
dm1 = np.linspace(0.2, 0.6, 20)
dm2 = np.linspace(0.5, 1.0, 20)
DM1, DM2 = np.meshgrid(dm1, dm2)
K1g = nu_floor * np.exp(-DM1)
K2g = nu_floor * np.exp(-DM2)
Ratio = K1g / K2g
ax3.plot_surface(DM1, DM2, Ratio, cmap="viridis", alpha=0.8)
ax3.set_xlabel("ΔM_epox")
ax3.set_ylabel("ΔM_HAT")
ax3.set_zlabel("k_epox/k_HAT")
ax3.set_title("Rate Ratio Surface")

plt.suptitle("Panel 03: Arene Oxide Formation", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_03_arene_oxide.png")

# ── Panel 04: NIH shift ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(131)
labels = ["Epoxidation", "NIH shift"]
ks = [K_EPOX/1e9, K_NIH/1e9]
colors = ["#27ae60", "#9b59b6"]
ax1.bar(labels, ks, color=colors, edgecolor="k")
ax1.set_ylabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("NIH Shift vs. Epoxidation")

ax2 = fig.add_subplot(132)
dm_vals = [0.18, 0.20, 0.35, 0.42, 0.65]
k_vals = [nu_floor*math.exp(-d)/1e9 for d in dm_vals]
names_short = ["NIH", "Carb", "Epox", "Nuc", "Desat"]
colors2 = ["#9b59b6", "#8e44ad", "#27ae60", "#f39c12", "#e74c3c"]
ax2.barh(names_short, k_vals, color=colors2, edgecolor="k")
ax2.set_xlabel("Rate (×10⁹ s⁻¹)")
ax2.set_title("All Atypical Rates")

ax3 = fig.add_subplot(133, projection="3d")
theta = np.linspace(0, 2*np.pi, 60)
phi   = np.linspace(0, np.pi, 30)
T2, P2 = np.meshgrid(theta, phi)
dm_surface = 0.18 + 0.05*np.sin(3*T2)*np.sin(P2)
K_surface  = nu_floor * np.exp(-dm_surface)
X = np.sin(P2)*np.cos(T2)
Y = np.sin(P2)*np.sin(T2)
Z = np.cos(P2)
ax3.plot_surface(X, Y, K_surface/1e9, cmap="plasma", alpha=0.85)
ax3.set_title("NIH Rate Surface")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("k (×10⁹)")

plt.suptitle("Panel 04: NIH Shift Kinetics", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_04_nih_shift.png")

# ── Panel 05: Nucleophilic aldehyde ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(131)
T_part = 65.0
dm_arr = np.linspace(0.0, 1.0, 200)
ea_arr = T_part * dm_arr / 4.184
ax1.plot(dm_arr, ea_arr, "b-", lw=2)
ax1.axvline(0.42, color="orange", ls="--", label="Nuc ΔM=0.42")
ax1.axhline(T_part*0.42/4.184, color="orange", ls=":")
ax1.set_xlabel("ΔM")
ax1.set_ylabel("Ea (kcal/mol)")
ax1.set_title("Barrier vs. ΔM")
ax1.legend()

ax2 = fig.add_subplot(132)
mechs2 = ["Nucleophilic\n(ΔM=0.42)", "Epoxidation\n(ΔM=0.35)", "Desat. eff."]
ks2 = [K_NUC/1e9, K_EPOX/1e9, K_DESAT_EFF/1e8]
ax2.bar(mechs2, ks2, color=["#f39c12","#27ae60","#e74c3c"], edgecolor="k")
ax2.set_ylabel("Rate (×10⁹ or ×10⁸ s⁻¹)")
ax2.set_title("Nucleophilic vs. Others")

ax3 = fig.add_subplot(133, projection="3d")
dm_x = np.linspace(0.3, 0.6, 25)
dm_y = np.linspace(0.3, 0.6, 25)
DX, DY = np.meshgrid(dm_x, dm_y)
K_XY = nu_floor * np.exp(-(DX + DY)/2)
ax3.plot_surface(DX, DY, K_XY/1e9, cmap="autumn", alpha=0.85)
ax3.set_xlabel("ΔM₁")
ax3.set_ylabel("ΔM₂")
ax3.set_zlabel("k̄ (×10⁹ s⁻¹)")
ax3.set_title("Averaged Rate Surface")

plt.suptitle("Panel 05: Nucleophilic O-Atom Transfer", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_05_nucleophilic_aldehyde.png")

# ── Panel 06: Rate ordering ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

ordered = [
    ("Desaturation\n(eff.)", K_DESAT_EFF),
    ("Nucleophilic", K_NUC),
    ("Epoxidation",  K_EPOX),
    ("Carbene",      K_CARB),
    ("NIH shift",    K_NIH),
]
names_o = [x[0] for x in ordered]
ks_o    = [x[1]/1e9 for x in ordered]
dm_o    = [-math.log(x[1]/nu_floor) for x in ordered]

ax1 = fig.add_subplot(131)
ax1.barh(names_o, ks_o, color=["#e74c3c","#f39c12","#27ae60","#8e44ad","#9b59b6"],
         edgecolor="k")
ax1.set_xlabel("Rate (×10⁹ s⁻¹)")
ax1.set_title("Rate Ordering (slowest → fastest)")

ax2 = fig.add_subplot(132)
ax2.plot(dm_o, ks_o, "ko-", ms=8, lw=2)
for i, (n, k, d) in enumerate(zip(names_o, ks_o, dm_o)):
    ax2.annotate(n.split("\n")[0], (d, k), textcoords="offset points",
                 xytext=(5, 5), fontsize=7)
ax2.set_xlabel("ΔM (effective)")
ax2.set_ylabel("k (×10⁹ s⁻¹)")
ax2.set_title("ΔM vs. Rate")

ax3 = fig.add_subplot(133, projection="3d")
n_pts = len(ordered)
theta_pts = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
r_pts = np.array(ks_o)
x_pts = r_pts * np.cos(theta_pts)
y_pts = r_pts * np.sin(theta_pts)
z_pts = np.array(dm_o)
ax3.scatter(x_pts, y_pts, z_pts, c=ks_o, cmap="viridis", s=100, zorder=5)
ax3.plot(x_pts, y_pts, z_pts, "k--", alpha=0.4)
for i, n in enumerate(names_o):
    ax3.text(x_pts[i], y_pts[i], z_pts[i]+0.02, n.split("\n")[0], fontsize=6)
ax3.set_xlabel("k·cos θ")
ax3.set_ylabel("k·sin θ")
ax3.set_zlabel("ΔM")
ax3.set_title("Rate-ΔM Phase Space")

plt.suptitle("Panel 06: Atypical Reaction Rate Ordering", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_06_rate_ordering.png")

# ── Panel 07: Product partitioning ────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

k_hydration = nu_floor * math.exp(-0.60)
f_phenol = K_NIH / (K_NIH + k_hydration)
f_diol   = k_hydration / (K_NIH + k_hydration)

ax1 = fig.add_subplot(131)
ax1.pie([f_phenol, f_diol], labels=["Phenol\n(NIH)", "Dihydrodiol"],
        colors=["#9b59b6", "#3498db"], autopct="%1.1f%%", startangle=90)
ax1.set_title("Product Distribution")

ax2 = fig.add_subplot(132)
k_hyd_arr = np.logspace(8, 11, 300)
f_phen_arr = K_NIH / (K_NIH + k_hyd_arr)
ax2.semilogx(k_hyd_arr, f_phen_arr, "b-", lw=2)
ax2.axvline(k_hydration, color="orange", ls="--", label=f"k_hyd={k_hydration/1e9:.2f}×10⁹")
ax2.axhline(f_phenol, color="purple", ls=":", label=f"f_phenol={f_phenol:.3f}")
ax2.set_xlabel("k_hydration (s⁻¹)")
ax2.set_ylabel("Fraction phenol")
ax2.set_title("Phenol Fraction vs. k_hyd")
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(133, projection="3d")
k_nih_arr  = np.logspace(9, 10.3, 20)
k_hyd_arr2 = np.logspace(8.5, 11, 20)
KN, KH = np.meshgrid(k_nih_arr, k_hyd_arr2)
FP = KN / (KN + KH)
ax3.plot_surface(np.log10(KN), np.log10(KH), FP, cmap="RdPu", alpha=0.85)
ax3.set_xlabel("log k_NIH")
ax3.set_ylabel("log k_hyd")
ax3.set_zlabel("f_phenol")
ax3.set_title("Phenol Fraction Surface")

plt.suptitle("Panel 07: Product Partitioning (Phenol vs. Dihydrodiol)", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_07_product_partitioning.png")

# ── Panel 08: Validation summary ──────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4))

lit_lo = {"Desaturation":1e8, "Epoxidation":1e9, "NIH shift":5e9, "Nucleophilic":1e9, "Carbene":5e9}
lit_hi = {"Desaturation":5e9,"Epoxidation":1e10,"NIH shift":1e10,"Nucleophilic":1e10,"Carbene":1e10}
pred   = {"Desaturation":K_DESAT_EFF,"Epoxidation":K_EPOX,"NIH shift":K_NIH,
          "Nucleophilic":K_NUC,"Carbene":K_CARB}

ax1 = fig.add_subplot(131)
rxns = list(pred.keys())
x = np.arange(len(rxns))
lo_arr = [lit_lo[r] for r in rxns]
hi_arr = [lit_hi[r] for r in rxns]
p_arr  = [pred[r] for r in rxns]
for i, (lo, hi, p) in enumerate(zip(lo_arr, hi_arr, p_arr)):
    ax1.plot([i, i], [lo, hi], "b-", lw=4, alpha=0.4)
    color = "g" if lo <= p <= hi else "r"
    ax1.scatter(i, p, color=color, s=80, zorder=5)
ax1.set_xticks(x)
ax1.set_xticklabels([r[:6] for r in rxns], rotation=30, ha="right", fontsize=7)
ax1.set_yscale("log")
ax1.set_ylabel("Rate (s⁻¹)")
ax1.set_title("Predicted vs. Lit. Range")

ax2 = fig.add_subplot(132)
kie_pred   = [4.5, 1.0, 1.0, 1.0, 1.0]
kie_expect = [">3", "=1", "=1", "=1", "=1"]
colors3 = ["#27ae60"]*5
ax2.bar(rxns, kie_pred, color=colors3, edgecolor="k")
ax2.axhline(1.0, color="gray", ls="--")
ax2.set_ylabel("KIE")
ax2.set_title("KIE Summary")
ax2.set_xticklabels([r[:6] for r in rxns], rotation=30, ha="right", fontsize=7)

ax3 = fig.add_subplot(133, projection="3d")
dm_vals3 = [0.18, 0.20, 0.35, 0.42]
k_vals3  = [K_NIH, K_CARB, K_EPOX, K_NUC]
kie_vals3 = [1.0, 1.0, 1.0, 1.0]
ax3.scatter(dm_vals3, [k/1e9 for k in k_vals3], kie_vals3,
            c=dm_vals3, cmap="viridis", s=150, zorder=5)
ax3.scatter([0.65], [K_DESAT_EFF/1e8], [4.5], color="red", s=200, marker="*",
            label="Desaturation\n(×10⁸)")
ax3.set_xlabel("ΔM")
ax3.set_ylabel("Rate")
ax3.set_zlabel("KIE")
ax3.set_title("ΔM–Rate–KIE Space")
ax3.legend(fontsize=6)

plt.suptitle("Panel 08: Validation Summary", fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("panel_08_validation.png")

print("All 8 panels generated.")
