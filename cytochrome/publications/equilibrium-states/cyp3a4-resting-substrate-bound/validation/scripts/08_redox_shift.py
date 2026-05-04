"""
Validation 08: Redox-potential shift from partition-depth change.

Verifies Theorem 14.1 of Paper 3 (Equation redox_shift):
    Delta_E_{1/2} = (k_B T / e) * n_eff * Delta_M * ln(b)

For CYP3A4 with Delta_M = 0.92, n_eff = 5, T = 310 K, b = e:
    Delta_E_{1/2} = (4.27e-21 / 1.6e-19) * 5 * 0.92 * 1
                  = 0.027 V * 5 = 123 mV

Paper-quoted value: +120 mV (matches within 3%).

Outputs: results/08_redox_shift.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    F_CB,
    KB_T,
    LN_BASE,
    N_EFF_DSHELL,
    S_FE_HS,
    S_FE_LS,
)

# Charge of an electron in coulombs (we use ELEM_CHARGE from _common)
import math

from _common import ELEM_CHARGE


def redox_shift_V(delta_M: float, n_eff: int, T_kBT: float = KB_T) -> float:
    """Delta_E = (k_B T / e) * n_eff * Delta_M * ln(b), with b = e (so ln b = 1)."""
    return (T_kBT / ELEM_CHARGE) * n_eff * delta_M * LN_BASE


def main() -> dict:
    # 1. Compute Delta_M from F_CB
    fe_ls = F_CB(S_FE_LS)
    fe_hs = F_CB(S_FE_HS)
    delta_M = fe_hs["M"] - fe_ls["M"]

    # 2. Single-electron contribution
    single_electron_shift_V = redox_shift_V(delta_M, n_eff=1)

    # 3. Full d-shell contribution
    full_shell_shift_V = redox_shift_V(delta_M, n_eff=N_EFF_DSHELL)

    # 4. Sweep over n_eff to show linear scaling
    sweep_n_eff = []
    for n in range(1, 7):
        shift_V = redox_shift_V(delta_M, n)
        sweep_n_eff.append({
            "n_eff": n,
            "shift_V": shift_V,
            "shift_mV": shift_V * 1000.0,
        })

    # 5. Sweep over Delta_M to show linear scaling
    sweep_dM = []
    for dM in [0.5, 0.7, 0.92, 1.1, 1.3]:
        shift_V = redox_shift_V(dM, N_EFF_DSHELL)
        sweep_dM.append({
            "delta_M": dM,
            "shift_V": shift_V,
            "shift_mV": shift_V * 1000.0,
        })

    # 6. Compare to experimental measurement
    paper_shift_mV = 120.0  # Daff 1997 measurement

    # 7. Comparison: resting potential to substrate-bound potential
    E_rest_mV = -300.0  # vs NHE
    E_bound_mV = E_rest_mV + (full_shell_shift_V * 1000.0)
    E_bound_paper_mV = -180.0

    checks = {
        "delta_M_consistent": bool(abs(delta_M - 0.92) < 0.15),
        "single_electron_shift_in_kT_range": bool(
            0.01 <= single_electron_shift_V <= 0.05
        ),
        "full_shell_shift_within_factor_2_of_paper_mV": bool(
            0.5 * paper_shift_mV <= full_shell_shift_V * 1000.0 <= 2.0 * paper_shift_mV
        ),
        "full_shell_within_30pct_of_paper": bool(
            abs(full_shell_shift_V * 1000.0 - paper_shift_mV) / paper_shift_mV < 0.30
        ),
        "shift_linear_in_dM": bool(
            abs(sweep_dM[3]["shift_mV"] / sweep_dM[1]["shift_mV"] - 1.1 / 0.7) < 0.05
        ),
        "E_bound_within_30mV_of_paper": bool(abs(E_bound_mV - E_bound_paper_mV) < 30.0),
    }

    return {
        "validation_id": "08_redox_shift",
        "paper_reference": "Paper 3, Theorem 14.1",
        "parameters": {
            "S_Fe_LS": list(S_FE_LS),
            "S_Fe_HS": list(S_FE_HS),
            "n_eff_dshell": N_EFF_DSHELL,
            "T_K": 310.0,
            "kB_T_J": KB_T,
        },
        "delta_M": delta_M,
        "single_electron_shift": {
            "V": single_electron_shift_V,
            "mV": single_electron_shift_V * 1000.0,
        },
        "full_shell_shift": {
            "V": full_shell_shift_V,
            "mV": full_shell_shift_V * 1000.0,
        },
        "redox_potentials": {
            "E_resting_mV_vs_NHE": E_rest_mV,
            "E_bound_predicted_mV_vs_NHE": E_bound_mV,
            "E_bound_experimental_mV_vs_NHE": E_bound_paper_mV,
        },
        "paper_predictions": {
            "delta_E_mV": paper_shift_mV,
        },
        "n_eff_sweep": sweep_n_eff,
        "delta_M_sweep": sweep_dM,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "08_redox_shift.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] redox shift")
    print(f"  ΔM             = {out['delta_M']:.3f}")
    print(f"  ΔE single-e^-  = {out['single_electron_shift']['mV']:.1f} mV")
    print(f"  ΔE full d^5    = {out['full_shell_shift']['mV']:.1f} mV (paper 120 mV)")
    print(f"  E_bound        = {out['redox_potentials']['E_bound_predicted_mV_vs_NHE']:.0f} mV "
          f"(paper {out['redox_potentials']['E_bound_experimental_mV_vs_NHE']} mV)")
    print(f"  -> wrote {out_path}")
