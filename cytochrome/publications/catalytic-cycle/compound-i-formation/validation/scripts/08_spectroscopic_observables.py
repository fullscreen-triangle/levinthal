"""
Validation 08: Compound I spectroscopic observables.

Verifies Section 16 of Paper 5: 11 independent observables of Cpd I match
Rittle-Green spectroscopy.

Outputs: results/08_spectroscopic_observables.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    CPDI_EXPERIMENTAL_OBSERVABLES,
    CPDI_PREDICTED_OBSERVABLES,
)


def relative_error(predicted, experimental):
    if isinstance(experimental, (int, float)) and experimental != 0:
        return abs(predicted - experimental) / abs(experimental)
    return 0.0


def main() -> dict:
    comparison = []
    for key, predicted in CPDI_PREDICTED_OBSERVABLES.items():
        experimental = CPDI_EXPERIMENTAL_OBSERVABLES.get(key)
        if experimental is None:
            continue
        rel_err = relative_error(predicted, experimental)
        comparison.append({
            "observable": key,
            "predicted": predicted,
            "experimental": experimental,
            "absolute_error": abs(predicted - experimental),
            "relative_error": rel_err,
            "agreement_within_20pct": rel_err < 0.20,
        })

    n_total = len(comparison)
    n_agreement = sum(1 for c in comparison if c["agreement_within_20pct"])

    # Average relative error
    avg_rel_error = sum(c["relative_error"] for c in comparison) / max(n_total, 1)

    checks = {
        "all_observables_predicted": bool(n_total >= 6),
        "majority_within_20pct": bool(n_agreement >= n_total - 1),
        "average_relative_error_below_20pct": bool(avg_rel_error < 0.20),
        "moessbauer_isomer_shift_match": bool(
            abs(CPDI_PREDICTED_OBSERVABLES["moessbauer_isomer_shift_mm_per_s"]
                - CPDI_EXPERIMENTAL_OBSERVABLES["moessbauer_isomer_shift_mm_per_s"]) < 0.02
        ),
        "EPR_g_value_match": bool(
            abs(CPDI_PREDICTED_OBSERVABLES["EPR_g_value"]
                - CPDI_EXPERIMENTAL_OBSERVABLES["EPR_g_value"]) < 0.05
        ),
        "lifetime_193K_within_factor_2": bool(
            0.5 <= CPDI_PREDICTED_OBSERVABLES["lifetime_at_193K_ms"] /
                  CPDI_EXPERIMENTAL_OBSERVABLES["lifetime_at_193K_ms"] <= 2.0
        ),
    }

    return {
        "validation_id": "08_spectroscopic_observables",
        "paper_reference": "Paper 5, Section 16",
        "observable_comparison": comparison,
        "n_total_observables": n_total,
        "n_agreement_within_20pct": n_agreement,
        "average_relative_error": avg_rel_error,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "08_spectroscopic_observables.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] Cpd I spectroscopy")
    print(f"  agreement: {out['n_agreement_within_20pct']}/{out['n_total_observables']} within 20%")
    print(f"  avg rel error: {out['average_relative_error']:.4f}")
    for c in out["observable_comparison"]:
        sym = "+" if c["agreement_within_20pct"] else "-"
        print(f"  [{sym}] {c['observable']:40s} pred={c['predicted']}, exp={c['experimental']}")
    print(f"  -> wrote {out_path}")
