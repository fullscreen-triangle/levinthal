#!/usr/bin/env python3
"""
02: Answering queries with no entry for the queried object.

The claim under test is the Empty Dictionary theorem instantiated on P450:
Identify, Similar, and Predict are answered for objects that were never
stored, by evaluating the coordinate map rather than by retrieval.

Two separable assertions are involved, and they need different controls:

  (i)  ANSWERABILITY -- the scheme returns an answer for an unstored object.
       Control A (a conventional exact-match dictionary over a disjoint
       training corpus) must fail here, or answerability is trivial.

  (ii) CONTENT -- the answer reflects the physics, not merely the arithmetic.
       Control B replaces the twenty-row table with a shuffled one: same
       size, same address algebra, wrong assignment. If a shuffled table
       scored as well, the coordinate map would be carrying no information.

A NOTE ON THE TASK, because the first version of this script failed here.
Nearest-neighbour over closely related P450 fragments (~94% identity) is
decided by string identity alone: any injective residue->address map ranks
the near-identical pair first, and the shuffled control scored a perfect
1.00, exactly matching the real table. That was a defective task, not a
confirmed claim -- it could not discriminate physics from bookkeeping. It is
retained below as `task_identity` precisely to document that it does not
discriminate.

The discriminating task is graded physicochemical proximity: address
distance must track hydrophobicity/volume/charge difference across all 190
residue pairs. Sequence identity cannot carry that; a shuffled table must
destroy it.
"""

import hashlib
import json
import os
import random
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, HERE)

from importlib import import_module
_m = import_module("01_storage_scaling")
AA_COORDS = _m.AA_COORDS

AA1 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}

# Independent reference scales -- NOT part of the resident state, and not
# used by the scheme. They exist only to score it from outside.
KD_HYDRO = {  # Kyte-Doolittle
    "Ala": 1.8, "Arg": -4.5, "Asn": -3.5, "Asp": -3.5, "Cys": 2.5,
    "Gln": -3.5, "Glu": -3.5, "Gly": -0.4, "His": -3.2, "Ile": 4.5,
    "Leu": 3.8, "Lys": -3.9, "Met": 1.9, "Phe": 2.8, "Pro": -1.6,
    "Ser": -0.8, "Thr": -0.7, "Trp": -0.9, "Tyr": -1.3, "Val": 4.2,
}
VOLUME = {  # side-chain volume, A^3
    "Ala": 88.6, "Arg": 173.4, "Asn": 114.1, "Asp": 111.1, "Cys": 108.5,
    "Gln": 143.8, "Glu": 138.4, "Gly": 60.1, "His": 153.2, "Ile": 166.7,
    "Leu": 166.7, "Lys": 168.6, "Met": 162.9, "Phe": 189.9, "Pro": 112.7,
    "Ser": 89.0, "Thr": 116.1, "Trp": 227.8, "Tyr": 193.6, "Val": 140.0,
}
CHARGE = {
    "Arg": 1.0, "Lys": 1.0, "His": 0.5, "Asp": -1.0, "Glu": -1.0,
}

K_TRITS = 9  # 3 trits per axis, interleaved -- Paper 2's k = 9


def trit_digits(x, k):
    out, v = [], min(max(x, 0.0), 1.0 - 1e-12)
    for _ in range(k):
        v *= 3.0
        d = int(v)
        out.append(d)
        v -= d
    return out


def address(res3, coords, k=K_TRITS):
    """Interleaved ternary address of one residue, recomputed on demand.
    No table of addresses is ever materialised."""
    sk, st, se = coords[res3]
    per = k // 3
    a, b, c = trit_digits(sk, per), trit_digits(st, per), trit_digits(se, per)
    return tuple(d for triple in zip(a, b, c) for d in triple)


def seq_address(seq, coords):
    return [address(AA1[ch], coords) for ch in seq if ch in AA1]


def lcp(u, v):
    n = 0
    for a, b in zip(u, v):
        if a != b:
            break
        n += 1
    return n


def similarity(s1, s2, coords):
    a1, a2 = seq_address(s1, coords), seq_address(s2, coords)
    n = min(len(a1), len(a2))
    if n == 0:
        return 0.0
    return sum(lcp(a1[i], a2[i]) for i in range(n)) / (n * K_TRITS)


def spearman(xs, ys):
    """Rank correlation, no SciPy dependency."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def table_hash(coords):
    return hashlib.sha256(json.dumps(coords, sort_keys=True).encode()).hexdigest()


def phys_distance(a, b):
    """Reference physicochemical distance, normalised per axis."""
    dh = abs(KD_HYDRO[a] - KD_HYDRO[b]) / 9.0
    dv = abs(VOLUME[a] - VOLUME[b]) / 167.7
    dc = abs(CHARGE.get(a, 0.0) - CHARGE.get(b, 0.0)) / 2.0
    return (dh * dh + dv * dv + dc * dc) ** 0.5


def addr_distance(a, b, coords):
    """Address separation: deeper shared prefix = closer."""
    return 1.0 - lcp(address(a, coords), address(b, coords)) / K_TRITS


TRAIN = {
    "CYP3A4_SRS1":  "LSLGGLLQPGDVLQPGAR",
    "CYP2D6_SRS1":  "LSPTVQRLAQRFGDVFLV",
    "CYP1A2_SRS4":  "GTETTSTTLRYGLLLLLK",
    "CYP2C9_SRS1":  "LSLPTLTLLLLLLFLLLK",
}
QUERY = {
    "CYP3A5_SRS1":  "LSLGGLLQPGDVLQPGVR",
    "CYP2D7_SRS1":  "LSPTVQRLAQRFGDVFLL",
    "CYP1A1_SRS4":  "GTETTSTTLRYGLLILLK",
    "CYP2C19_SRS1": "LSLPTLTLLLLLLFLLLR",
    "CYP17A1_frag": "WQRQRRLAQARALPAFHT",
    "CYP51A1_frag": "MHTQVAKETLARQQLQTL",
}
NEAREST_TRAIN = {
    "CYP3A5_SRS1": "CYP3A4_SRS1", "CYP2D7_SRS1": "CYP2D6_SRS1",
    "CYP1A1_SRS4": "CYP1A2_SRS4", "CYP2C19_SRS1": "CYP2C9_SRS1",
}


def run_identity_task(coords):
    """Non-discriminating task, retained to document that it is not."""
    correct = 0
    for qname, truth in NEAREST_TRAIN.items():
        scores = {t: similarity(QUERY[qname], s, coords) for t, s in TRAIN.items()}
        correct += (max(scores, key=scores.get) == truth)
    return correct / len(NEAREST_TRAIN)


def run_graded_task(coords):
    """Discriminating task: address distance vs physicochemical distance
    over all C(20,2) = 190 residue pairs."""
    pairs = list(combinations(sorted(AA_COORDS), 2))
    ad = [addr_distance(a, b, coords) for a, b in pairs]
    pd = [phys_distance(a, b) for a, b in pairs]
    return spearman(ad, pd), len(pairs)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    coords = dict(AA_COORDS)
    h_before = table_hash(coords)

    # --- (i) ANSWERABILITY ----------------------------------------------
    empty_answers = []
    for qname, qseq in QUERY.items():
        scores = {t: similarity(qseq, s, coords) for t, s in TRAIN.items()}
        best = max(scores, key=scores.get)
        empty_answers.append({
            "query": qname, "answered": True, "nearest": best,
            "score": scores[best], "truth": NEAREST_TRAIN.get(qname),
            "correct": best == NEAREST_TRAIN.get(qname)
                       if qname in NEAREST_TRAIN else None,
        })

    h_after = table_hash(coords)
    state_unchanged = (h_before == h_after)

    # CONTROL A: conventional exact-match dictionary over TRAIN only.
    lookup = {v: k for k, v in TRAIN.items()}
    ctrl_a = [{"query": q, "answered": lookup.get(s) is not None,
               "hit": lookup.get(s)} for q, s in QUERY.items()]
    a_answered = sum(r["answered"] for r in ctrl_a)

    # --- (ii) CONTENT ---------------------------------------------------
    rng = random.Random(20260812)
    real_rho, n_pairs = run_graded_task(coords)
    real_ident = run_identity_task(coords)

    # CONTROL B: many shuffles, so the comparison is distributional.
    keys = list(AA_COORDS)
    shuf_rho, shuf_ident = [], []
    N_SHUF = 200
    for _ in range(N_SHUF):
        vals = [AA_COORDS[k] for k in keys]
        rng.shuffle(vals)
        sh = dict(zip(keys, vals))
        shuf_rho.append(run_graded_task(sh)[0])
        shuf_ident.append(run_identity_task(sh))

    mean_shuf_rho = sum(shuf_rho) / N_SHUF
    mean_shuf_ident = sum(shuf_ident) / N_SHUF
    # One-sided empirical p: how often does a shuffle match the real table?
    p_rho = sum(1 for r in shuf_rho if r >= real_rho) / N_SHUF
    p_ident = sum(1 for r in shuf_ident if r >= real_ident) / N_SHUF

    verdict = "PASS" if (
        len(empty_answers) == len(QUERY)
        and state_unchanged
        and a_answered == 0            # control A cannot answer at all
        and real_rho > mean_shuf_rho   # graded task: real beats shuffled
        and p_rho < 0.05               # and does so significantly
    ) else "FAIL"

    out = {
        "experiment": "02_query_without_entries",
        "claim": ("queries are answered for objects with no stored entry, by "
                  "evaluation rather than retrieval, and the answers carry "
                  "physicochemical content"),
        "n_train": len(TRAIN), "n_query": len(QUERY),
        "resident_state_hash_before": h_before,
        "resident_state_hash_after": h_after,
        "resident_state_unchanged_by_queries": state_unchanged,
        "answerability": {
            "empty_scheme_answered": len(empty_answers), "of": len(QUERY),
            "control_a_dictionary_answered": a_answered, "of_": len(QUERY),
            "answers": empty_answers, "control_a": ctrl_a,
        },
        "task_identity_NON_DISCRIMINATING": {
            "note": ("nearest-neighbour at ~94% identity is decided by string "
                     "identity; a shuffled table scores the same. Reported to "
                     "document that this task does not test the claim."),
            "real_table_accuracy": real_ident,
            "shuffled_mean_accuracy": mean_shuf_ident,
            "empirical_p": p_ident,
            "discriminates": p_ident < 0.05,
        },
        "task_graded_DISCRIMINATING": {
            "note": ("Spearman rho between address distance and independent "
                     "physicochemical distance over all residue pairs"),
            "n_pairs": n_pairs,
            "real_table_rho": real_rho,
            "shuffled_mean_rho": mean_shuf_rho,
            "shuffled_max_rho": max(shuf_rho),
            "n_shuffles": N_SHUF,
            "empirical_p": p_rho,
            "discriminates": p_rho < 0.05,
        },
        "verdict": verdict,
    }

    with open(os.path.join(RESULTS, "02_query_without_entries.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"answerability : empty {len(empty_answers)}/{len(QUERY)}, "
          f"CONTROL A dictionary {a_answered}/{len(QUERY)}")
    print(f"state unchanged by queries: {state_unchanged}")
    print(f"identity task  (non-discriminating): real {real_ident:.2f} vs "
          f"shuffled {mean_shuf_ident:.2f}  p={p_ident:.3f}")
    print(f"graded task    (discriminating)    : real rho {real_rho:.3f} vs "
          f"shuffled {mean_shuf_rho:.3f} (max {max(shuf_rho):.3f})  "
          f"p={p_rho:.3f}  over {n_pairs} pairs")
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
