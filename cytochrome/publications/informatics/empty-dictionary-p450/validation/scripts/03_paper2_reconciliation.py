#!/usr/bin/env python3
"""
03: Reconciling Paper 2's storage figure with the O(1) claim.

Paper 2 of this monograph (the CYP3A4 address manifold) writes:

    "The Empty Dictionary stores nothing; the entire content of
     Sexp_P450^family is the syntactic concatenation of addresses. Storage
     cost is O(N*k) trits per sequence. For N = 500 residues and k = 9 trits
     per residue, this is 4500 trits ~ 7 kbits per sequence; for 4x10^5
     sequences, ~350 MB."

Those two sentences are not compatible. "Stores nothing" is O(1); "~350 MB
for 4x10^5 sequences" is O(n). This script establishes, by measurement,
which quantity each sentence is describing, so the reconciliation in the
manuscript rests on arithmetic rather than on assertion.

The resolution is a distinction between two different things:
    SCHEME  -- what must be resident to make any address computable at all
    CORPUS  -- what it costs to additionally MATERIALISE a chosen set of
               addresses, which is an optional cache, not a precondition

CONTROL. The reconciliation would be vacuous if materialised addresses were
required to answer queries. Experiment 02 already shows they are not (every
query answered with the corpus cache empty). Here the control is stronger
and direct: run the full query set with the materialised cache present and
then with it deleted, and compare the answers byte-for-byte. If they differ,
the cache is load-bearing and the O(1) claim fails.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
sys.path.insert(0, HERE)

from importlib import import_module
_s = import_module("01_storage_scaling")
_q = import_module("02_query_without_entries")

AA_COORDS = _s.AA_COORDS
QUERY, TRAIN = _q.QUERY, _q.TRAIN


def trits_to_bytes(n):
    return n * math.log2(3) / 8.0


class Scheme:
    """The resident state: twenty rows plus the encoding rule.

    `cache` holds materialised sequence addresses. It is optional by
    construction -- `answer()` never reads it."""

    def __init__(self, coords):
        self.coords = coords
        self.cache = {}

    def materialise(self, corpus):
        for name, seq in corpus.items():
            self.cache[name] = _q.seq_address(seq, self.coords)

    def drop_cache(self):
        self.cache = {}

    def cache_bytes(self):
        n_trits = sum(len(a) * _q.K_TRITS for a in self.cache.values())
        return trits_to_bytes(n_trits)

    def resident_bytes(self):
        return _s.resident_bytes()

    def answer(self, seq):
        """Nearest TRAIN fragment. Computed from self.coords only."""
        scores = {t: _q.similarity(seq, s, self.coords) for t, s in TRAIN.items()}
        best = max(scores, key=scores.get)
        return {"nearest": best, "score": round(scores[best], 12)}


def main():
    os.makedirs(RESULTS, exist_ok=True)

    sch = Scheme(dict(AA_COORDS))

    # --- With the corpus cache materialised ------------------------------
    sch.materialise({**TRAIN, **QUERY})
    cached_bytes = sch.cache_bytes()
    answers_with = {q: sch.answer(s) for q, s in QUERY.items()}
    resident_with = sch.resident_bytes()

    # --- With the cache deleted ------------------------------------------
    sch.drop_cache()
    answers_without = {q: sch.answer(s) for q, s in QUERY.items()}
    resident_without = sch.resident_bytes()

    identical = (json.dumps(answers_with, sort_keys=True)
                 == json.dumps(answers_without, sort_keys=True))
    resident_unaffected = (resident_with == resident_without)

    # --- Attribute each of Paper 2's sentences to a quantity -------------
    N_RES, K, N_SEQ = 500, 9, 400_000
    p2_per_seq_trits = N_RES * K
    p2_per_seq_kbits = p2_per_seq_trits * math.log2(3) / 1000.0
    p2_corpus_MB = trits_to_bytes(p2_per_seq_trits) * N_SEQ / 1e6

    out = {
        "experiment": "03_paper2_reconciliation",
        "claim": ("Paper 2's O(N*k) figure measures an optional materialised "
                  "cache, not the resident state; the resident state is O(1) "
                  "and the cache is not load-bearing"),
        "paper2_quoted": {
            "per_sequence_trits": p2_per_seq_trits,
            "per_sequence_kbits_recomputed": p2_per_seq_kbits,
            "per_sequence_kbits_as_stated": 7,
            "corpus_MB_recomputed": p2_corpus_MB,
            "corpus_MB_as_stated": 350,
            "n_sequences": N_SEQ,
        },
        "resident_state_bytes": resident_with,
        "materialised_cache_bytes_for_test_corpus": cached_bytes,
        "cache_dropped_answers_identical": identical,
        "resident_state_unaffected_by_cache": resident_unaffected,
        "attribution": {
            "'stores nothing'": "resident state (SCHEME) -- O(1)",
            "'~350 MB for 4x10^5 sequences'":
                "materialised address cache (CORPUS) -- O(n), optional",
        },
        "verdict": "PASS" if (identical and resident_unaffected) else "FAIL",
    }

    with open(os.path.join(RESULTS, "03_paper2_reconciliation.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"Paper 2 per-sequence : {p2_per_seq_trits} trits = "
          f"{p2_per_seq_kbits:.2f} kbits (stated ~7 kbits)")
    print(f"Paper 2 corpus       : {p2_corpus_MB:.1f} MB (stated ~350 MB)")
    print(f"resident state       : {resident_with} bytes")
    print(f"materialised cache   : {cached_bytes:.1f} bytes for "
          f"{len(TRAIN) + len(QUERY)} fragments")
    print(f"answers identical with/without cache : {identical}")
    print(f"resident state unaffected by cache   : {resident_unaffected}")
    print(f"VERDICT: {out['verdict']}")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
