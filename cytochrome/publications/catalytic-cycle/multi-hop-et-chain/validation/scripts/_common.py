"""
Shared utilities for Paper 4 validations: multi-hop electron transfer chain.
"""

from __future__ import annotations

import math

# Physical constants
KB = 1.380649e-23
HBAR = 1.054571817e-34
ELEM_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
TEMP = 310.0
KB_T = KB * TEMP
KB_T_KCAL = KB_T * 6.022e23 / 4184.0
JOULE_TO_KCAL = 6.022e23 / 4184.0
JOULE_TO_EV = 1.0 / ELEM_CHARGE


# =============================================================================
# Cofactor S-coordinates (Paper 4, Sections 4-7)
# =============================================================================

# NADPH nicotinamide ring centroid
S_NADPH = (0.625, 0.510, 0.500)
OMEGA_NADPH_CM_INV = 1340.0
PARTITION_NADPH = (2, 0, 0, 0)  # hydride sp^3, paired

# FAD isoalloxazine in CPR diaphorase domain
S_FAD = (0.605, 0.500, 0.475)
OMEGA_FAD_CM_INV = 1580.0  # isoalloxazine pi-mode

# FMN isoalloxazine in CPR flavodoxin domain
S_FMN = (0.580, 0.495, 0.470)
OMEGA_FMN_CM_INV = 1580.0

# Heme Fe(III) high-spin (substrate-bound, paper 3)
S_FE_HS = (0.815, 0.510, 0.560)

# Heme Fe(II) high-spin (one-electron-reduced, state 3)
S_FE_HS_RED = (0.802, 0.508, 0.552)


# =============================================================================
# Cofactor distances (Paper 4, Section 8.2)
# =============================================================================

DISTANCE_NADPH_FAD_A = 4.0
DISTANCE_FAD_FMN_A = 4.0
DISTANCE_FMN_HEME_A = 14.0


# =============================================================================
# Marcus theory parameters (canonical)
# =============================================================================

BETA_PROTEIN_PER_A = 1.1   # 1/Å, through-protein decay
LAMBDA_REORG_EV = 0.85     # eV, reorganization energy estimate


# =============================================================================
# Categorical efficiency relation
# =============================================================================

DC_CHAIN = 4
KCAT_KM_PREDICTED = 10 ** (10 - DC_CHAIN)  # 10^6 M^-1 s^-1


# =============================================================================
# Closed-form F_CB
# =============================================================================

def F_CB(S: tuple[float, float, float], regularize: bool = True) -> dict:
    Sk, St, Se = S
    norm = math.sqrt(Sk * Sk + St * St + Se * Se)
    if regularize and norm >= 1.0:
        epsilon = math.exp(-7.13)
        norm_clipped = min(norm, 1.0 - epsilon)
    else:
        norm_clipped = norm
    if norm_clipped >= 1.0:
        return {"M": float("inf"), "n": -1, "norm": norm}
    M = -math.log(1.0 - norm_clipped)
    n = max(1, int(math.ceil(math.sqrt(3.0 * M))))
    if norm > 1e-12:
        try:
            l = max(0, min(n - 1, int(math.floor((n - 1) * math.acos(min(1.0, max(-1.0, Se / norm))) / math.pi))))
        except ValueError:
            l = 0
    else:
        l = 0
    return {"M": M, "n": n, "l": l, "norm": norm}


# =============================================================================
# Marcus rate
# =============================================================================

def marcus_rate(distance_A: float, lambda_eV: float, dG_eV: float = 0.0,
                H_DA_eV: float = 0.001) -> float:
    """Marcus rate: k = (2pi/hbar) * |H_DA|^2 / sqrt(4 pi lambda kBT) * exp(-(dG+lambda)^2 / 4 lambda kBT)"""
    H_DA_J = H_DA_eV * ELEM_CHARGE
    lambda_J = lambda_eV * ELEM_CHARGE
    dG_J = dG_eV * ELEM_CHARGE
    pre = (2 * math.pi / HBAR) * (H_DA_J ** 2) / math.sqrt(4 * math.pi * lambda_J * KB_T)
    exponent = -((dG_J + lambda_J) ** 2) / (4 * lambda_J * KB_T)
    return pre * math.exp(exponent)


def distance_dependent_rate(rate_at_ref_A: float, distance_A: float,
                            ref_distance_A: float = 4.0,
                            beta: float = BETA_PROTEIN_PER_A) -> float:
    """k(r) = k0 * exp(-beta * (r - r_ref))"""
    return rate_at_ref_A * math.exp(-beta * (distance_A - ref_distance_A))


# =============================================================================
# CYP3A4 substrate kcat/KM data (Paper 4, Section 16)
# =============================================================================

CYP3A4_KCAT_KM_DATA = [
    {"substrate": "Testosterone (6β-OH)", "kcat_KM_M_per_s": 1.5e6, "ref": "Shou1999"},
    {"substrate": "Midazolam (1'-OH)", "kcat_KM_M_per_s": 4.0e6, "ref": "Galetin2005"},
    {"substrate": "Erythromycin", "kcat_KM_M_per_s": 7.0e5, "ref": "Shou1999"},
    {"substrate": "Nifedipine", "kcat_KM_M_per_s": 2.0e6, "ref": "Shou1999"},
]


# =============================================================================
# Hop rates
# =============================================================================

HOP_RATE_INTRINSIC = KB_T / HBAR  # categorical clock ~ 6.5e12 s^-1

HOP_RATES_EXPERIMENTAL = {
    "hop1_NADPH_FAD": 600.0,        # s^-1, hydride transfer (matrix-damped)
    "hop2_FAD_FMN": 1e7,             # s^-1, intramolecular
    "hop3_FMN_heme": 5e7,            # s^-1, interprotein, rate-limiting
}


# =============================================================================
# Receiver floor
# =============================================================================

FLOOR_BIO = 3.7e-4
