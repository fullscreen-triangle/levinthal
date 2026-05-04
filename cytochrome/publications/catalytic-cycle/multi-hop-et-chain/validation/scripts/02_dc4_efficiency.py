"""
Validation 02: d_C = 4 catalytic efficiency prediction.

Verifies the headline result of Paper 4 (Section 13):
  log_10(k_cat/K_M) ≈ 10 - d_C  with d_C = 4
  Predicts k_cat/K_M ~ 10^6 M^-1 s^-1 for CPR-CYP3A4

Compares against measured values for representative substrates from
Shou 1999, Galetin 2005.

Outputs: results/02_dc4_efficiency.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import CYP3A4_KCAT_KM_DATA, DC_CHAIN, KCAT_KM_PREDICTED  # noqa: E402


def main() -> dict:
    # Predicted value
    predicted_log = 10 - DC_CHAIN
    predicted_kcat_KM = 10 ** predicted_log

    # Measured values
    measured_logs = [math.log10(d["kcat_KM_M_per_s"]) for d in CYP3A4_KCAT_KM_DATA]
    geometric_mean_log = sum(measured_logs) / len(measured_logs)
    geometric_mean = 10 ** geometric_mean_log

    # Sweep over d_C
    dc_sweep = []
    for dc in range(1, 9):
        sweep_kcat_km = 10 ** (10 - dc)
        dc_sweep.append({
            "d_C": dc,
            "log10_kcat_KM": 10 - dc,
            "kcat_KM_M_per_s": sweep_kcat_km,
        })

    # Distance from predicted: log scale
    deviations = []
    for d in CYP3A4_KCAT_KM_DATA:
        log_obs = math.log10(d["kcat_KM_M_per_s"])
        log_dev = log_obs - predicted_log
        deviations.append({
            "substrate": d["substrate"],
            "kcat_KM_M_per_s": d["kcat_KM_M_per_s"],
            "log10_kcat_KM": log_obs,
            "log_deviation_from_prediction": log_dev,
        })

    mean_log_dev = sum(d["log_deviation_from_prediction"] for d in deviations) / len(deviations)
    max_log_dev = max(abs(d["log_deviation_from_prediction"]) for d in deviations)

    checks = {
        "d_C_eq_4": bool(DC_CHAIN == 4),
        "predicted_kcat_KM_eq_1e6": bool(predicted_kcat_KM == 1e6),
        "geometric_mean_within_1_log": bool(abs(geometric_mean_log - predicted_log) < 1.0),
        "max_substrate_deviation_within_1_log": bool(max_log_dev < 1.0),
        "all_substrates_above_1e5": bool(all(d["kcat_KM_M_per_s"] > 1e5 for d in CYP3A4_KCAT_KM_DATA)),
        "monotonic_decreasing_in_dc": bool(all(
            dc_sweep[i]["kcat_KM_M_per_s"] > dc_sweep[i + 1]["kcat_KM_M_per_s"]
            for i in range(len(dc_sweep) - 1)
        )),
    }

    return {
        "validation_id": "02_dc4_efficiency",
        "paper_reference": "Paper 4, Section 13 (Equation efficiency)",
        "categorical_distance": DC_CHAIN,
        "predicted_kcat_KM_M_per_s": predicted_kcat_KM,
        "predicted_log10": predicted_log,
        "measured_data": CYP3A4_KCAT_KM_DATA,
        "measured_log10": measured_logs,
        "measured_geometric_mean_log10": geometric_mean_log,
        "measured_geometric_mean_M_per_s": geometric_mean,
        "deviations": deviations,
        "mean_log_deviation": mean_log_dev,
        "max_log_deviation": max_log_dev,
        "dc_sweep": dc_sweep,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "02_dc4_efficiency.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] d_C = 4 efficiency")
    print(f"  predicted: 10^{10 - out['categorical_distance']} = {out['predicted_kcat_KM_M_per_s']:.0e}")
    print(f"  geom mean: 10^{out['measured_geometric_mean_log10']:.2f} = {out['measured_geometric_mean_M_per_s']:.2e}")
    print(f"  max log deviation: {out['max_log_deviation']:.3f}")
    print(f"  -> wrote {out_path}")
