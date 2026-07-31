"""
Experiment 1 --- Solvent role is derivable (Theorem 3.2).

Checks all three clauses of Theorem 3.2 plus the negative control the
paper mandates in Sec. 5.3, item 1: a solvent leaf with no system
contacts must return BULK and must commit no cut.

The structural claims are re-checked under every weight function in the
robustness family, because the paper claims they follow from monotonicity
and floor-boundedness alone (Remark 2.2). If a clause holds only for the
logarithm, it is a property of the logarithm and the paper overclaims.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))

from medium import (  # noqa: E402
    BETA_DEFAULT,
    BULK,
    SOL,
    STRUCTURAL,
    ContactGraph,
    Leaf,
    Medium,
    Refusal,
    individuate_solvent,
    robustness_family,
)

BETA = BETA_DEFAULT


def _cyp3a4_resting(mu_water: float = 55.5, tau: float = 1.0e-3) -> ContactGraph:
    """The CYP3A4 resting state: one axial water bound to heme iron.

    The axial water is the framework's own worked example --- a ligand that
    leaves on substrate binding --- so it is the right positive case. Bulk
    water is the same identity with no system contacts.
    """
    med = Medium(mu={"H2O": mu_water}, tau=tau, label="aqueous")
    g = ContactGraph(med, beta=BETA)
    g.add_leaf(Leaf("H2O_axial", SOL, "H2O"))
    g.add_leaf(Leaf("H2O_bulk", SOL, "H2O"))
    g.add_leaf(Leaf("heme_Fe", "cof", "heme"))
    # The axial water is coordinated to the iron: a real, strong contact.
    g.add_contact("H2O_axial", "heme_Fe", 4.0 * BETA)
    # H2O_bulk gets no contacts at all. That is the whole point.
    return g


def run() -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append(
            {"check": name, "verdict": "PASS" if passed else "FAIL",
             "detail": detail, **extra}
        )

    # -- 1.1 the positive case: the axial water is structural -------------
    g = _cyp3a4_resting()
    axial = g.role_report("H2O_axial")
    check(
        "axial water is structural",
        axial["role"] == STRUCTURAL,
        f"{axial['because']} -> {axial['role']}",
        structural_residue=axial["structural_residue"],
        medium_weight=axial["medium_weight"],
    )

    # -- 1.2 NEGATIVE CONTROL (paper Sec. 5.3 item 1) --------------------
    # A solvent leaf with no system contacts must be BULK. Without this
    # the suite tests one branch of a two-branch predicate.
    bulk = g.role_report("H2O_bulk")
    check(
        "NEGATIVE CONTROL: bulk water (no system contacts) is refused",
        bulk["role"] == BULK,
        f"{bulk['because']} -> {bulk['role']}",
        structural_residue=bulk["structural_residue"],
        medium_weight=bulk["medium_weight"],
    )

    # ...and the refusal must commit no cut.
    refused = None
    try:
        individuate_solvent(g, "H2O_bulk")
        cut_committed = True
    except Refusal as r:
        cut_committed = False
        refused = r.as_dict()
    check(
        "NEGATIVE CONTROL: refusal commits no cut",
        not cut_committed,
        "individuate_solvent raised Refusal and advanced no clock"
        if refused
        else "individuation SUCCEEDED on bulk water -- the refusal is vacuous",
        refusal=refused,
    )

    # -- 1.3 Theorem 3.2(i): no system contacts => bulk, always ----------
    # Checked across the whole robustness family and a wide occupancy
    # sweep: clause (i) should not depend on either.
    fails_i = []
    for fname, fn in robustness_family().items():
        for mu in (1e-6, 1e-3, 1.0, 55.5, 1e3):
            gg = _cyp3a4_resting(mu_water=mu)
            if gg.role("H2O_bulk", fn) != BULK:
                fails_i.append({"weight_fn": fname, "mu": mu})
    check(
        "Thm 3.2(i): no system contacts => bulk, for all weight fns and mu",
        not fails_i,
        f"20 combinations (4 weight fns x 5 occupancies); "
        f"{len(fails_i)} violation(s)",
        violations=fails_i,
    )

    # -- 1.4 Theorem 3.2(ii): ambient + strong contact => structural -----
    fails_ii = []
    for fname, fn in robustness_family().items():
        gg = _cyp3a4_resting(mu_water=55.5)  # ambient
        if not gg.medium.is_ambient("H2O", BETA):
            fails_ii.append({"weight_fn": fname, "why": "H2O not ambient"})
            continue
        if gg.role("H2O_axial", fn) != STRUCTURAL:
            fails_ii.append({"weight_fn": fname, "why": "axial not structural"})
    check(
        "Thm 3.2(ii): ambient identity + contact >= 2*beta => structural",
        not fails_ii,
        f"contact weight 4.0*beta >= 2*beta; {len(fails_ii)} violation(s) "
        f"across {len(robustness_family())} weight fns",
        violations=fails_ii,
    )

    # -- 1.5 Theorem 3.2(iii): monotone in mu ----------------------------
    # Increasing ambient occupancy can turn bulk structural, never the
    # reverse. Tested by sweeping mu upward and checking the role sequence
    # never goes structural -> bulk.
    monotone_violations = []
    sweep = [10**e for e in range(-8, 4)]
    for fname, fn in robustness_family().items():
        seen_structural = False
        seq = []
        for mu in sweep:
            med = Medium(mu={"H2O": mu}, tau=1.0e-3, label="sweep")
            gg = ContactGraph(med, beta=BETA)
            gg.add_leaf(Leaf("w", SOL, "H2O"))
            gg.add_leaf(Leaf("p", "res", "ALA"))
            # A weak contact, so the role actually flips somewhere in range.
            gg.add_contact("w", "p", 1.05 * BETA)
            r = gg.role("w", fn)
            seq.append((mu, r))
            if r == STRUCTURAL:
                seen_structural = True
            elif seen_structural:
                monotone_violations.append(
                    {"weight_fn": fname, "mu": mu,
                     "why": "structural -> bulk as mu increased"}
                )
    check(
        "Thm 3.2(iii): role is monotone in ambient occupancy",
        not monotone_violations,
        f"mu swept over {len(sweep)} decades for "
        f"{len(robustness_family())} weight fns; "
        f"{len(monotone_violations)} monotonicity violation(s)",
        violations=monotone_violations,
    )

    # -- 1.6 NEGATIVE CONTROL (paper Sec. 5.3 item 4) --------------------
    # The role must actually be able to CHANGE, or clause (iii) is
    # unexercised and the monotonicity check above is vacuous.
    flips = {}
    for fname, fn in robustness_family().items():
        roles = set()
        for mu in sweep:
            med = Medium(mu={"H2O": mu}, tau=1.0e-3)
            gg = ContactGraph(med, beta=BETA)
            gg.add_leaf(Leaf("w", SOL, "H2O"))
            gg.add_leaf(Leaf("p", "res", "ALA"))
            gg.add_contact("w", "p", 1.05 * BETA)
            roles.add(gg.role("w", fn))
        flips[fname] = sorted(roles)
    both_seen = {k: v for k, v in flips.items() if len(v) == 2}
    check(
        "NEGATIVE CONTROL: role genuinely flips under medium perturbation",
        len(both_seen) == len(flips),
        f"weight fns exhibiting BOTH roles across the sweep: "
        f"{len(both_seen)}/{len(flips)}. A predicate that returns one "
        f"value on every input tests nothing.",
        roles_seen=flips,
    )

    # -- 1.7 scale invariance: only tau/mu is observable ------------------
    # Definition 2.1 claims tau and mu are never separately observable.
    # Scaling both by the same factor must leave every role unchanged.
    scale_violations = []
    for factor in (1e-3, 1e-2, 10.0, 1e3):
        med_a = Medium(mu={"H2O": 55.5}, tau=1.0e-3)
        med_b = Medium(mu={"H2O": 55.5 * factor}, tau=1.0e-3 * factor)
        for med, tag in ((med_a, "base"), (med_b, f"x{factor:g}")):
            gg = ContactGraph(med, beta=BETA)
            gg.add_leaf(Leaf("w", SOL, "H2O"))
            gg.add_leaf(Leaf("p", "res", "ALA"))
            gg.add_contact("w", "p", 2.0 * BETA)
            if tag == "base":
                base_role = gg.role("w")
            elif gg.role("w") != base_role:
                scale_violations.append({"factor": factor})
    check(
        "only the ratio tau/mu is observable (scale invariance)",
        not scale_violations,
        f"4 joint rescalings of (mu, tau); {len(scale_violations)} "
        f"role change(s). Def. 2.1 requires none.",
        violations=scale_violations,
    )

    passed = sum(1 for c in checks if c["verdict"] == "PASS")
    return {
        "experiment": "exp01_solvent_role",
        "claim": "Theorem 3.2 --- solvent role is derivable from the "
                 "contact graph and the medium alone",
        "aggregate": {
            "checks": len(checks),
            "passed": passed,
            "failed": len(checks) - passed,
            "verdict": "PASS" if passed == len(checks) else "FAIL",
        },
        "beta": BETA,
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
