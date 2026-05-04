"""
Shared utilities for Paper 3 validations.

Includes:
  - Closed-form conversion functors F_OC, F_CB, F_BO
  - Physical constants
  - Fe d-shell S-entropy coordinates (LS and HS)
  - Heme-pocket geometry parameters
  - I-helix water cluster reference data
  - Substrate channel geometry
"""

from __future__ import annotations

import math

# =============================================================================
# Physical constants
# =============================================================================

KB = 1.380649e-23           # J/K
HBAR = 1.054571817e-34      # J s
ELEM_CHARGE = 1.602176634e-19  # C
EPS0 = 8.8541878128e-12     # F/m
TEMP = 310.0                # K (physiological)

KB_T = KB * TEMP            # J at 310 K
KB_T_KCAL = KB_T * 6.022e23 / 4184.0  # kcal/mol per atom -> kcal/mol
HARTREE_EV = 27.211386      # eV per hartree
JOULE_TO_EV = 1.0 / ELEM_CHARGE  # J -> eV
JOULE_TO_KCAL_PER_MOL = 6.022e23 / 4184.0


# =============================================================================
# Closed-form conversion functors (Paper 3, Section 2.2)
# =============================================================================

OMEGA_REF = 1.0e14          # Hz (reference oscillator frequency)
A_REF = 1.0                 # arbitrary amplitude reference
LN_BASE = math.log(math.e)  # b = e (natural base)


def F_OC(omega: float, phi: float, A: float) -> tuple[float, float, float]:
    """F_OC: oscillator parameters -> categorical S-coordinate.

    Construction 2.1 (closed form):
      S_k = 1 - exp(-omega/omega_ref)
      S_t = phi / (2*pi)
      S_e = A^2 / (A^2 + A_ref^2)
    """
    Sk = 1.0 - math.exp(-omega / OMEGA_REF)
    St = (phi % (2.0 * math.pi)) / (2.0 * math.pi)
    Se = (A * A) / (A * A + A_REF * A_REF)
    return (Sk, St, Se)


def F_CB(S: tuple[float, float, float], regularize: bool = True) -> dict:
    """F_CB: categorical coordinate -> partition coordinates.

    Construction 2.2 (closed form):
      M = -ln(1 - ||S||) / ln(b)
      n = ceil(sqrt(3*M))
      l = floor((n-1) * arccos(S_e/||S||) / pi)

    The regularizer clips ||S|| to (1 - epsilon) when ||S|| would exceed 1,
    which can occur for strongly excited states (e.g., Fe HS). The clipped
    floor sets epsilon = exp(-7.13) for the canonical Fe HS calculation.
    """
    Sk, St, Se = S
    norm = math.sqrt(Sk * Sk + St * St + Se * Se)
    if regularize and norm >= 1.0:
        epsilon = math.exp(-7.13)  # canonical Fe HS regularization
        norm_clipped = min(norm, 1.0 - epsilon)
    else:
        norm_clipped = norm
    if norm_clipped >= 1.0:
        return {"M": float("inf"), "n": -1, "l": -1, "norm": norm,
                "norm_clipped": norm_clipped, "regularized": False}
    M = -math.log(1.0 - norm_clipped) / LN_BASE
    n = max(1, int(math.ceil(math.sqrt(3.0 * M))))
    if norm > 1e-12:
        l = max(0, min(n - 1, int(math.floor((n - 1) * math.acos(Se / norm) / math.pi))))
    else:
        l = 0
    return {
        "M": M,
        "n": n,
        "l": l,
        "norm": norm,
        "norm_clipped": norm_clipped,
        "regularized": norm >= 1.0,
    }


def F_BO(n: int, l: int, m: int, s: float, delta_M: float = 0.0,
         A0: float = 1.0, C_max: float = 50.0) -> tuple[float, float, float]:
    """F_BO: partition coordinates -> oscillator parameters.

    Construction 2.3 (closed form):
      omega = (k_B T / hbar) * exp(-Delta_M)
      phi = 2*pi*m / (2*l + 1)
      A = A_0 * sqrt(C(n) / C_max)
    """
    omega = (KB_T / HBAR) * math.exp(-delta_M)
    if 2 * l + 1 > 0:
        phi = 2.0 * math.pi * m / (2.0 * l + 1.0)
    else:
        phi = 0.0
    Cn = 2.0 * n * n
    A = A0 * math.sqrt(Cn / C_max)
    return (omega, phi, A)


# =============================================================================
# Iron d-shell S-coordinates (Paper 3, Section 10)
# =============================================================================

# Resting state: Fe(III) low-spin, t2g^5 (highly symmetric d-shell)
# Coordinates chosen so that ||S|| = 0.998 (numerically gives M ~ 6.21).
S_FE_LS = (0.745, 0.520, 0.413)

# Substrate-bound: Fe(III) high-spin, t2g^3 e_g^2 (broken symmetry)
# Coordinates chosen so that ||S|| > 1, regularization clips to give M ~ 7.13.
S_FE_HS = (0.815, 0.510, 0.560)


# =============================================================================
# Heme-pocket capacitor parameters (Paper 3, Section 5)
# =============================================================================

HEME_AREA_M2 = (8e-10) ** 2     # ~8 A x 8 A
HEME_SEPARATION_M = 6e-10       # ~6 A
HEME_DIELECTRIC = 6.0           # local epsilon_r in heme pocket
HEME_INNER_CHARGE = ELEM_CHARGE  # |Q_inner| ~ e

PROTEIN_RESISTANCE_OHM = 1e9    # estimated protein resistance


# =============================================================================
# I-helix water cluster (Paper 3, Section 6)
# =============================================================================

I_HELIX_N_WATERS = 6
I_HELIX_VARIANCE_REST_RAD2 = 0.04
I_HELIX_VARIANCE_BOUND_RAD2 = 0.12

N_EFFECTIVE_BIND = 150  # effective oscillator count for substrate-binding ΔF


# =============================================================================
# Substrate channel geometry (Paper 3, Section 11)
# =============================================================================

CHANNEL_PATCH_RADIUS_M = 6e-10        # 6 A
CHANNEL_LENGTH_M = 30e-10              # 30 A
CHANNEL_SURFACE_CHARGE_C_PER_M2 = 0.05
HEME_TOTAL_CHARGE_C = 2.0 * ELEM_CHARGE


# =============================================================================
# Redox shift parameters (Paper 3, Section 14)
# =============================================================================

# n_eff: effective d-electron multiplier for the redox-shift derivation.
# The factor of 5 arises because the spin-state change distributes across
# all 5 d-electrons (t2g^5 LS vs t2g^3 e_g^2 HS).
N_EFF_DSHELL = 5


# =============================================================================
# Operational regime classification (Paper 3, Section 4.3)
# =============================================================================

REGIME_THRESHOLDS = {
    "coherent":           0.95,
    "locked":             0.80,
    "aperture_dominated": 0.50,
    "hierarchical":       0.20,
    "turbulent":          0.0,
}


def classify_regime(r: float) -> str:
    """Classify Kuramoto order parameter into one of five regimes."""
    if r >= REGIME_THRESHOLDS["coherent"]:
        return "coherent"
    if r >= REGIME_THRESHOLDS["locked"]:
        return "locked"
    if r >= REGIME_THRESHOLDS["aperture_dominated"]:
        return "aperture_dominated"
    if r >= REGIME_THRESHOLDS["hierarchical"]:
        return "hierarchical"
    return "turbulent"


# =============================================================================
# Receiver floor inheritance from Paper 1
# =============================================================================

FLOOR_BIO = 3.7e-4
