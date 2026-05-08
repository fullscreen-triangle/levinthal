"""Shared constants for Paper 13: Spectroscopic Atlas of P450 States."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Spectroscopic parameters for the seven P450 states (Paper 12)
# Soret band (nm) and extinction coefficient (mM^-1 cm^-1)
# Q-bands (nm) and alpha/beta peak positions
# EPR signals (g-values): high-spin (HS) g = 7.7, 3.5, 1.8; low-spin (LS) g = 2.42, 2.25, 1.92

# UV-Vis Soret peak wavelengths (nm) per state
SORET_NM = {
    "resting_FeIII_LS":   417,   # substrate-free, water-bound
    "substrate_bound_HS": 392,   # HS after substrate binding (blue shift)
    "ferrous_FeII":       408,   # reduced iron, CO-bound at 450 nm
    "oxy_complex":        418,   # FeII-O2 complex
    "peroxo":             440,   # peroxo anion (broad)
    "compound0":          367,   # compound 0 (hydroperoxo), estimated
    "compound_I":         370,   # compound I (porphyrin cation radical), estimated
}

# EPR g-values (low-spin resting state)
EPR_G_LS = (2.42, 2.25, 1.92)
EPR_G_HS = (7.70, 3.50, 1.80)

# Resonance Raman: Fe=O stretch (Compound I/II) ~ 795 cm^-1 (shifted by ^18O to ~758)
RAMAN_FEO_CM1    = 795.0
RAMAN_FEO_18O    = 795.0 * math.sqrt(16.0 / 18.0)   # 18O isotope shift

# Activation partition depth per spectroscopic transition
# ΔM_spec correlates with transition energy (hν/T_part)
T_PART = 65.0  # kJ/mol
nu_floor = 1e10  # s^-1

def soret_to_dm(lam_nm: float) -> float:
    """Convert Soret peak wavelength to ΔM proxy."""
    hc_kJ = 1.196e5 / lam_nm   # kJ/mol for photon at lambda nm
    return hc_kJ / T_PART

SORET_DM = {k: soret_to_dm(v) for k, v in SORET_NM.items()}


def write_result(name: str, data: dict, checks: dict) -> dict:
    passed = all(checks.values())
    out = {
        "script": name,
        "verdict": "PASS" if passed else "FAIL",
        "checks": checks,
        **data,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(out, indent=2))
    verdict = "PASS" if passed else "FAIL"
    print(f"{name}: {verdict}")
    for k, v in checks.items():
        mark = "OK" if v else "XX"
        print(f"  {mark} {k}")
    return out
