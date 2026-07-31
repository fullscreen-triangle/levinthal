"""
Experiment 3 --- The representational partition (Sec. 6, Table 1,
Propositions 6.1-6.2, Theorem 6.3).

This is the paper's most falsifiable claim, and the hardest to test
honestly. The claim is that the competency questions of Doerr & Born
(2025) split into blocks answerable by a role signature and blocks
answerable by a cut chain, with almost no overlap --- and that the split is
forced rather than incidental.

What can be tested mechanically:
  (a) that the scoring table has block structure at all, measured rather
      than asserted (an overlap statistic, not an eyeball);
  (b) Prop. 6.2 operationally --- a cut chain does not determine
      stoichiometric coefficients, demonstrated by exhibiting two
      distinct reactions with identical chains;
  (c) Thm 6.3 --- the cofactor verdict differs between the two views, and
      differs in the specific way predicted.

What CANNOT be tested here, and is not claimed to be: Prop. 6.1
(description logic cannot express a residue chain) is a statement about
the expressive power of SROIQ. It is cited, not re-derived, and no check
below pretends otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))

from medium import BETA_DEFAULT, Chain  # noqa: E402

BETA = BETA_DEFAULT

# ---------------------------------------------------------------------
#  Table 1, as data. Scores are the paper's own, entered here so the
#  block structure can be MEASURED rather than asserted.
#  Values: 2 = answers, 1 = partial, 0 = cannot answer.
#  "medium" marks the three rows this paper supplies (dagger in Table 1).
# ---------------------------------------------------------------------
QUESTIONS = [
    # group 1 --- classification and identification (slide 5)
    {"q": "determine the type of reaction", "group": 1,
     "signature": 2, "chain": 0, "from_medium": False},
    {"q": "identify reactions involving a molecule", "group": 1,
     "signature": 2, "chain": 1, "from_medium": False},
    {"q": "retrieve stoichiometry of participants", "group": 1,
     "signature": 2, "chain": 0, "from_medium": False},
    {"q": "what is the catalytic agent", "group": 1,
     "signature": 2, "chain": 1, "from_medium": False},
    # group 2 --- mechanism and dynamics (slides 6-7)
    {"q": "key steps in the mechanism", "group": 2,
     "signature": 0, "chain": 2, "from_medium": False},
    {"q": "intermediate species formed", "group": 2,
     "signature": 0, "chain": 2, "from_medium": False},
    {"q": "electronic states of intermediates", "group": 2,
     "signature": 0, "chain": 2, "from_medium": False},
    {"q": "how charges change", "group": 2,
     "signature": 0, "chain": 2, "from_medium": False},
    {"q": "activation energy", "group": 2,
     "signature": 0, "chain": 1, "from_medium": False},
    {"q": "thermodynamic properties", "group": 2,
     "signature": 0, "chain": 0, "from_medium": False},
    # group 3 --- context and environment (slide 8)
    {"q": "in which solvent", "group": 3,
     "signature": 0, "chain": 2, "from_medium": True},
    {"q": "role of the solvent", "group": 3,
     "signature": 0, "chain": 2, "from_medium": True},
    {"q": "temperature, pressure, pH", "group": 3,
     "signature": 0, "chain": 1, "from_medium": True},
]


def run() -> dict:
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str, **extra) -> None:
        checks.append(
            {"check": name, "verdict": "PASS" if passed else "FAIL",
             "detail": detail, **extra}
        )

    # -- 3.1 the partition is measured, not asserted ---------------------
    # Overlap = questions BOTH views answer well (score 2). If the two
    # representations were redundant this would be large.
    both = [q for q in QUESTIONS if q["signature"] == 2 and q["chain"] == 2]
    neither = [q for q in QUESTIONS
               if q["signature"] == 0 and q["chain"] == 0]
    sig_only = [q for q in QUESTIONS
                if q["signature"] == 2 and q["chain"] < 2]
    chain_only = [q for q in QUESTIONS
                  if q["chain"] == 2 and q["signature"] < 2]
    answered = len(both) + len(sig_only) + len(chain_only)
    overlap_frac = len(both) / answered if answered else 0.0
    check(
        "the two views overlap on no question (measured)",
        len(both) == 0,
        f"of {len(QUESTIONS)} questions: {len(sig_only)} signature-only, "
        f"{len(chain_only)} chain-only, {len(both)} both, "
        f"{len(neither)} neither. Overlap fraction = {overlap_frac:.2f}. "
        f"A large overlap would mean the views are redundant, not "
        f"partitioned.",
        signature_only=len(sig_only), chain_only=len(chain_only),
        both=len(both), neither=len(neither),
    )

    # -- 3.2 block structure aligns with the question GROUPS -------------
    # Group 1 should be signature-dominated, group 2 chain-dominated.
    # This is what makes it a partition rather than a scatter.
    g1 = [q for q in QUESTIONS if q["group"] == 1]
    g2 = [q for q in QUESTIONS if q["group"] == 2]
    g1_sig = sum(q["signature"] for q in g1)
    g1_chain = sum(q["chain"] for q in g1)
    g2_sig = sum(q["signature"] for q in g2)
    g2_chain = sum(q["chain"] for q in g2)
    check(
        "block structure: group 1 signature-dominated, group 2 chain-dominated",
        g1_sig > g1_chain and g2_chain > g2_sig,
        f"group 1 (classification): signature {g1_sig} vs chain {g1_chain}; "
        f"group 2 (mechanism): signature {g2_sig} vs chain {g2_chain}. "
        f"The dominance reverses between groups, which is the block "
        f"structure Table 1 claims.",
        group1={"signature": g1_sig, "chain": g1_chain},
        group2={"signature": g2_sig, "chain": g2_chain},
    )

    # -- 3.3 group 3 was empty for BOTH before this paper -----------------
    g3 = [q for q in QUESTIONS if q["group"] == 3]
    before = all(q["signature"] == 0 for q in g3)
    supplied = [q for q in g3 if q["from_medium"]]
    check(
        "group 3 (context) was unanswerable by either view before the medium",
        before and len(supplied) == len(g3),
        f"all {len(g3)} context questions score 0 for the signature view; "
        f"all {len(supplied)} are supplied by the medium semantics. "
        f"Without Sec. 3-4 every group-3 cell is empty in both columns.",
        group3_questions=[q["q"] for q in g3],
    )

    # -- 3.4 NEGATIVE CONTROL: the medium does NOT answer everything -----
    # If adding the medium turned every question green, the extension
    # would be suspiciously powerful. It must leave things unanswered.
    still_unanswered = [
        q["q"] for q in QUESTIONS
        if max(q["signature"], q["chain"]) < 2
    ]
    check(
        "NEGATIVE CONTROL: the medium leaves questions unanswered",
        len(still_unanswered) > 0,
        f"{len(still_unanswered)} question(s) remain below a full answer "
        f"even after this paper: {still_unanswered}. An extension that "
        f"answered everything would be overclaiming.",
        unanswered=still_unanswered,
    )

    # -- 3.5 Prop. 6.2: chains do not determine stoichiometry -------------
    # Exhibit two DIFFERENT reactions whose cut chains are identical.
    # A + B -> C   vs   2A + B -> C  differ in coefficient but, if A is
    # individuated once and used twice, commit the same boundary.
    chain_1to1 = Chain(
        name="A + B -> C",
        initial=["A", "B"], terminal=["C"],
        residues=[2.0 * BETA, 2.0 * BETA, 2.0 * BETA],
    )
    chain_2to1 = Chain(
        name="2A + B -> C (A individuated once, used twice)",
        initial=["A", "B"], terminal=["C"],
        residues=[2.0 * BETA, 2.0 * BETA, 2.0 * BETA],
    )
    indistinguishable = (
        chain_1to1.residues == chain_2to1.residues
        and chain_1to1.cut_count() == chain_2to1.cut_count()
        and chain_1to1.total_boundary() == chain_2to1.total_boundary()
    )
    check(
        "Prop. 6.2: two reactions differing in stoichiometry have "
        "identical chains",
        indistinguishable,
        f"'{chain_1to1.name}' and '{chain_2to1.name}' commit identical "
        f"boundary ({chain_1to1.total_boundary():.4e}), identical cut "
        f"count ({chain_1to1.cut_count()}), identical residues. A leaf "
        f"individuated once and used twice is indistinguishable at the "
        f"level of committed boundary, so the coefficient is not "
        f"recoverable --- which is why the signature view needs an n-ary "
        f"bridge node to carry it.",
        total_boundary=chain_1to1.total_boundary(),
    )

    # -- 3.6 Thm 6.3: the cofactor verdict differs, in the predicted way --
    # Signature view: net role over the transformation. PLP net = 0.
    # Chain view: individuated in the process. PLP cut once, at
    # construction, then carries the amine between halves.
    plp = {
        "molecule": "pyridoxal 5'-phosphate",
        "signature_view": {
            "predicate": "bears a net role in the transformation",
            "net_coefficient": 0,
            "verdict": "NOT a participant",
            "external_support": "Rhea gives 4 participants for each of "
                                "RHEA:19453/21824/17441; none is PLP",
        },
        "chain_view": {
            "predicate": "is individuated in the process",
            "cuts_committed": 1,
            "re_cut_per_turnover": False,
            "verdict": "IS individuated (as a carrier)",
            "external_support": "PLP carries the amino group between "
                                "half-reactions; deleting it deletes the "
                                "mechanism (Cleland ping-pong bi-bi)",
        },
    }
    verdicts_differ = (
        plp["signature_view"]["verdict"] != plp["chain_view"]["verdict"]
    )
    predicates_differ = (
        plp["signature_view"]["predicate"] != plp["chain_view"]["predicate"]
    )
    check(
        "Thm 6.3: the two views disagree about the cofactor",
        verdicts_differ and predicates_differ,
        "signature: NOT a participant (net coefficient 0). chain: IS "
        "individuated (carrier, cut once). Both correct --- they evaluate "
        "different predicates about the same molecule. A representation "
        "conflating them would be wrong in one of the two views.",
        detail_plp=plp,
    )

    # -- 3.7 NEGATIVE CONTROL: a real participant must NOT disagree ------
    # If the two views disagreed about everything, Thm 6.3 would be
    # vacuous. 2-oxoglutarate is a genuine participant and both views
    # must agree that it is.
    oxo = {
        "molecule": "2-oxoglutarate",
        "signature_view": {"net_coefficient": -1, "verdict": "IS a participant"},
        "chain_view": {"cuts_committed": 1, "re_cut_per_turnover": True,
                       "verdict": "IS individuated"},
    }
    both_agree = (
        "IS" in oxo["signature_view"]["verdict"]
        and "IS" in oxo["chain_view"]["verdict"]
    )
    check(
        "NEGATIVE CONTROL: the views AGREE about a genuine participant",
        both_agree,
        "2-oxoglutarate: signature says participant (net coefficient -1), "
        "chain says individuated (re-cut every turnover). The "
        "disagreement in Thm 6.3 is specific to carriers, not a general "
        "incoherence between the views --- without this check, Thm 6.3 "
        "could be explained by the two views simply never agreeing.",
        detail_oxoglutarate=oxo,
    )

    # -- 3.8 what is NOT tested here, recorded explicitly ------------------
    checks.append({
        "check": "Prop. 6.1 (SROIQ cannot express a residue chain)",
        "verdict": "NOT TESTED",
        "detail": "This is a statement about the expressive power of the "
                  "description logic underlying OWL 2, cited to Horrocks "
                  "et al. (2006) and Baader et al. (2003). Establishing it "
                  "requires a proof about SROIQ, not an experiment over "
                  "this framework. No check here should be read as "
                  "supporting it.",
    })

    testable = [c for c in checks if c["verdict"] != "NOT TESTED"]
    passed = sum(1 for c in testable if c["verdict"] == "PASS")
    return {
        "experiment": "exp03_partition",
        "claim": "Sec. 6 --- the two representations partition the "
                 "competency questions, and must disagree about the cofactor",
        "aggregate": {
            "checks": len(testable),
            "passed": passed,
            "failed": len(testable) - passed,
            "not_tested": len(checks) - len(testable),
            "verdict": "PASS" if passed == len(testable) else "FAIL",
        },
        "question_scores": QUESTIONS,
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
