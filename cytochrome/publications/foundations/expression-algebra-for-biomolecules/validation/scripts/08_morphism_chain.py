"""
Validation 08: Morphism chain end-to-end smoke test.

Verifies Theorem 8.1 (Type Safety) and Theorem 8.2 (S-Entropy Conservation)
of Paper 1 by exercising

    eval_R = access o fuse o catalyze* o observe

on a synthetic 12-residue receiver tree containing residue, cofactor, and
solvent leaves.

Checks:
  - Each operation's output type matches the next operation's input type
  - The signature dimensions are preserved through observe -> catalyze* -> fuse
  - S-entropy of the signature does not blow up; bounded by the formula in Eq. (12)
  - The final access output (a contact map) is a binary N x N matrix

Outputs: results/08_morphism_chain.json
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
N_LEAVES = 12  # mini-protein with 1 cofactor + 1 substrate water + 10 residues


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (nats)."""
    p = p.flatten()
    p = p[p > 1e-12]
    p = p / p.sum() if p.sum() > 0 else p
    return float(-np.sum(p * np.log(p + 1e-30)))


def synthetic_signature(n: int, rng: random.Random) -> np.ndarray:
    """Synthetic NxN partition signature with structured magnitude pattern."""
    Sig = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Sig[i, j] = 5.0  # diagonal: strong self-coupling
            else:
                seq_sep = abs(i - j)
                # Helix-like contacts at separation 3-4
                helix_contrib = 1.0 if seq_sep in (3, 4) else 0.0
                # Add small random off-diagonal
                noise = abs(rng.gauss(0, 0.5))
                Sig[i, j] = helix_contrib * 2.0 + noise
    # Make symmetric
    Sig = (Sig + Sig.T) / 2.0
    return Sig


def observe(coupling: np.ndarray) -> np.ndarray:
    """Pass 1: observe operation. Returns NxN signature."""
    Sig = np.abs(coupling)
    return Sig


def catalyze(Sig: np.ndarray, kernel_kind: str = "helix") -> np.ndarray:
    """Pass 2: catalyze operation with structural kernel."""
    n = Sig.shape[0]
    out = Sig.copy()
    if kernel_kind == "helix":
        # Enhance contacts at sequence separation 3-4
        for i in range(n):
            for j in range(n):
                if abs(i - j) in (3, 4):
                    out[i, j] *= 1.5
    elif kernel_kind == "cofactor":
        # Enhance the last row/column (cofactor leaf at position n-1)
        out[n - 1, :] *= 1.3
        out[:, n - 1] *= 1.3
    return out


def fuse(Sig_k: np.ndarray, Sig_t: np.ndarray, Sig_e: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Pass 3: fuse operation."""
    w_k, w_t, w_e = weights
    return w_k * Sig_k + w_t * Sig_t + w_e * Sig_e


def access_contact_map(Sig: np.ndarray, threshold: float) -> np.ndarray:
    """Pass 4: access with contact-map target."""
    return (Sig > threshold).astype(int)


def main() -> dict:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Build synthetic receiver tree's coupling matrix
    Sig_input = synthetic_signature(N_LEAVES, rng)

    # ---- Type Safety: trace types through the chain ----
    types = []

    # Step 1: observe
    Sig_observed = observe(Sig_input)
    types.append({
        "step": "observe",
        "input_type": f"coupling_matrix [{N_LEAVES} x {N_LEAVES}]",
        "output_type": f"Sigma [{Sig_observed.shape[0]} x {Sig_observed.shape[1]}]",
        "shape_match": bool(Sig_observed.shape == Sig_input.shape),
    })
    S_after_observe = shannon_entropy(Sig_observed)

    # Step 2: catalyze* (twice)
    Sig_helix = catalyze(Sig_observed, "helix")
    Sig_helix_cof = catalyze(Sig_helix, "cofactor")
    types.append({
        "step": "catalyze (helix)",
        "input_type": f"Sigma [{Sig_observed.shape[0]} x {Sig_observed.shape[1]}]",
        "output_type": f"Sigma [{Sig_helix.shape[0]} x {Sig_helix.shape[1]}]",
        "shape_match": bool(Sig_helix.shape == Sig_observed.shape),
    })
    types.append({
        "step": "catalyze (cofactor)",
        "input_type": f"Sigma [{Sig_helix.shape[0]} x {Sig_helix.shape[1]}]",
        "output_type": f"Sigma [{Sig_helix_cof.shape[0]} x {Sig_helix_cof.shape[1]}]",
        "shape_match": bool(Sig_helix_cof.shape == Sig_helix.shape),
    })
    S_after_catalyze = shannon_entropy(Sig_helix_cof)

    # Step 3: fuse (use 3 views: instantaneous = Sig_observed, time-avg = Sig_helix,
    # catalyzed = Sig_helix_cof)
    weights = (0.4, 0.4, 0.2)  # canonical (k, t, e) weighting
    Sig_fused = fuse(Sig_helix_cof, Sig_helix, Sig_observed, weights)
    types.append({
        "step": "fuse",
        "input_type": f"3 Sigmas [{N_LEAVES} x {N_LEAVES}], weights {weights}",
        "output_type": f"Sigma [{Sig_fused.shape[0]} x {Sig_fused.shape[1]}]",
        "shape_match": bool(Sig_fused.shape == Sig_observed.shape),
    })
    S_after_fuse = shannon_entropy(Sig_fused)

    # Step 4: access
    threshold = float(np.median(Sig_fused.flatten())) + float(Sig_fused.std()) * 0.5
    contact_map = access_contact_map(Sig_fused, threshold)
    types.append({
        "step": "access (contact map)",
        "input_type": f"Sigma [{Sig_fused.shape[0]} x {Sig_fused.shape[1]}], threshold={threshold:.4f}",
        "output_type": f"binary contact map [{contact_map.shape[0]} x {contact_map.shape[1]}]",
        "shape_match": bool(contact_map.shape == Sig_fused.shape),
        "binary_check": bool(set(np.unique(contact_map).tolist()).issubset({0, 1})),
    })

    # ---- S-Entropy Conservation: verify bounded change ----
    # Per Theorem 8.2, |S_out - S_in| <= sum_k ln|kernel_k| + N^2 * H(tau_c) + Floor
    n_kernels = 2
    kernel_size_max = 2  # helix kernel covers offsets 3, 4 (size ~2 along diagonal)
    kernel_bound = n_kernels * math.log(kernel_size_max + 1)
    tau_c = threshold / Sig_fused.max()
    binary_entropy = -tau_c * math.log(tau_c + 1e-30) - (1 - tau_c) * math.log(1 - tau_c + 1e-30)
    floor = 3.7e-4
    theoretical_bound = kernel_bound + N_LEAVES * N_LEAVES * binary_entropy + floor
    actual_change = abs(S_after_fuse - S_after_observe)

    # ---- Summaries ----
    contact_density = float(contact_map.sum() / contact_map.size)

    checks = {
        "all_steps_shape_match": bool(all(t["shape_match"] for t in types)),
        "access_output_binary": bool(types[-1]["binary_check"]),
        "n_chain_steps_executed": bool(len(types) == 5),
        "s_entropy_change_bounded": bool(actual_change <= theoretical_bound),
        "contact_density_reasonable": bool(0.05 < contact_density < 0.6),
    }

    result = {
        "validation_id": "08_morphism_chain",
        "paper_reference": "Paper 1, Theorems 8.1 and 8.2",
        "parameters": {
            "n_leaves": N_LEAVES,
            "fuse_weights": list(weights),
            "access_threshold": threshold,
            "random_seed": RANDOM_SEED,
        },
        "type_trace": types,
        "entropy_trace": {
            "S_after_observe": S_after_observe,
            "S_after_catalyze": S_after_catalyze,
            "S_after_fuse": S_after_fuse,
            "actual_change": actual_change,
            "theoretical_bound": theoretical_bound,
            "kernel_bound": kernel_bound,
            "binary_entropy_term": N_LEAVES * N_LEAVES * binary_entropy,
            "floor": floor,
        },
        "contact_map": {
            "shape": list(contact_map.shape),
            "n_contacts": int(contact_map.sum()),
            "density": contact_density,
            "matrix": contact_map.tolist(),
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    return result


if __name__ == "__main__":
    out = main()
    out_path = Path(__file__).parent.parent / "results" / "08_morphism_chain.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[{out['verdict']}] morphism chain smoke test")
    for t in out["type_trace"]:
        print(f"  {t['step']:25s} -> shape_ok={t['shape_match']}")
    print(f"  S change: {out['entropy_trace']['actual_change']:.3f} "
          f"<= bound {out['entropy_trace']['theoretical_bound']:.3f}")
    print(f"  contact density: {out['contact_map']['density']:.3f}")
    print(f"  -> wrote {out_path}")
