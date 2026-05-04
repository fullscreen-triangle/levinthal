"""
Validation 03: Heme-pocket capacitor.

Verifies Section 5 of Paper 3:
  - C_heme = epsilon_0 * epsilon_r * A / d ≈ 5.7e-20 F
  - U_heme = Q^2 / (2 C) ≈ 1.4 eV
  - tau_RC = R * C ≈ 60 ps
  - Sensitivity to dielectric constant epsilon_r in [4, 10]

Outputs: results/03_heme_capacitor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    EPS0,
    HEME_AREA_M2,
    HEME_DIELECTRIC,
    HEME_INNER_CHARGE,
    HEME_SEPARATION_M,
    JOULE_TO_EV,
    PROTEIN_RESISTANCE_OHM,
)


def capacitance(eps_r: float, area: float, sep: float) -> float:
    """C = eps_0 * eps_r * A / d (parallel plate)."""
    return EPS0 * eps_r * area / sep


def stored_energy(Q: float, C: float) -> float:
    """U = Q^2 / (2C) in joules."""
    return Q * Q / (2.0 * C)


def rc_time(R: float, C: float) -> float:
    return R * C


def main() -> dict:
    # Canonical calculation
    C_heme = capacitance(HEME_DIELECTRIC, HEME_AREA_M2, HEME_SEPARATION_M)
    U_heme_J = stored_energy(HEME_INNER_CHARGE, C_heme)
    U_heme_eV = U_heme_J * JOULE_TO_EV
    tau_RC_s = rc_time(PROTEIN_RESISTANCE_OHM, C_heme)

    # Sensitivity to epsilon_r
    eps_r_sweep = []
    for eps_r in [3.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
        C = capacitance(eps_r, HEME_AREA_M2, HEME_SEPARATION_M)
        U_J = stored_energy(HEME_INNER_CHARGE, C)
        eps_r_sweep.append({
            "epsilon_r": eps_r,
            "C_F": C,
            "U_J": U_J,
            "U_eV": U_J * JOULE_TO_EV,
            "tau_RC_s": rc_time(PROTEIN_RESISTANCE_OHM, C),
        })

    # Sensitivity to separation
    sep_sweep = []
    for sep_A in [4.0, 5.0, 6.0, 7.0, 8.0]:
        sep_m = sep_A * 1e-10
        C = capacitance(HEME_DIELECTRIC, HEME_AREA_M2, sep_m)
        U_J = stored_energy(HEME_INNER_CHARGE, C)
        sep_sweep.append({
            "sep_A": sep_A,
            "C_F": C,
            "U_eV": U_J * JOULE_TO_EV,
            "tau_RC_s": rc_time(PROTEIN_RESISTANCE_OHM, C),
        })

    # Compare to other capacitors
    comparison = {
        "membrane_patch_clamp_pF": 1e-12,        # 1 pF reference
        "DNA_chromatin_pF": 300e-12,              # 300 pF (cellular charge paper)
        "heme_pocket_F": C_heme,
        "heme_pocket_aF": C_heme * 1e18,          # attofarads
    }

    # Comparison with paper-quoted values
    paper_C = 5.7e-20
    paper_U_eV = 1.4
    paper_tau_RC = 60e-12

    checks = {
        "C_heme_within_factor_2_of_paper": bool(0.5 * paper_C <= C_heme <= 2.0 * paper_C),
        "C_heme_in_attofarad_range": bool(1e-21 <= C_heme <= 1e-18),
        "U_heme_within_factor_2_of_paper_eV": bool(0.5 * paper_U_eV <= U_heme_eV <= 2.0 * paper_U_eV),
        "U_heme_above_redox_potential_range": bool(U_heme_eV > 0.5),
        "tau_RC_within_factor_2_of_paper_ps": bool(0.5 * paper_tau_RC <= tau_RC_s <= 2.0 * paper_tau_RC),
        "tau_RC_below_ms_turnover": bool(tau_RC_s < 1e-3),
    }

    return {
        "validation_id": "03_heme_capacitor",
        "paper_reference": "Paper 3, Section 5",
        "parameters": {
            "epsilon_r": HEME_DIELECTRIC,
            "area_m2": HEME_AREA_M2,
            "separation_m": HEME_SEPARATION_M,
            "Q_inner_C": HEME_INNER_CHARGE,
            "R_protein_ohm": PROTEIN_RESISTANCE_OHM,
        },
        "canonical_values": {
            "C_heme_F": C_heme,
            "C_heme_aF": C_heme * 1e18,
            "U_heme_J": U_heme_J,
            "U_heme_eV": U_heme_eV,
            "tau_RC_s": tau_RC_s,
            "tau_RC_ps": tau_RC_s * 1e12,
        },
        "paper_predictions": {
            "C_F": paper_C,
            "U_eV": paper_U_eV,
            "tau_RC_ps": paper_tau_RC * 1e12,
        },
        "epsilon_r_sweep": eps_r_sweep,
        "separation_sweep": sep_sweep,
        "comparison": comparison,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "03_heme_capacitor.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    cv = out["canonical_values"]
    print(f"[{out['verdict']}] heme-pocket capacitor")
    print(f"  C_heme    = {cv['C_heme_aF']:.2f} aF (paper 57 aF)")
    print(f"  U_heme    = {cv['U_heme_eV']:.3f} eV (paper 1.4 eV)")
    print(f"  tau_RC    = {cv['tau_RC_ps']:.1f} ps (paper 60 ps)")
    print(f"  -> wrote {out_path}")
