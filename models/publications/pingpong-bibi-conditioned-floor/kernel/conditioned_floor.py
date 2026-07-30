"""
Condition-dependent floor for R_bio.

The expression-algebra paper gives the floor as a three-term estimate

    beta = floor_disc(d) + floor_Q(oscillators, T_int) + floor_conv(d)

and every downstream artefact then treats it as a scalar constant
(RECEIVER_FLOOR = 3.7e-4 in the sandbox, `floor 3.7e-4` in the plays).
That is only correct at one set of assay conditions.

The middle term is an Allan-deviation-style oscillator floor,

    sigma_i = 1 / (Q_i * sqrt(T_int * f_i)),

and the Q-factor of a molecular oscillator is not a constant: it falls as
thermal occupation of competing modes rises, and it depends on the medium
the oscillator is damped by. So beta is already a function of temperature,
solvent viscosity and integration time -- the framework simply never wrote
the dependence down.

This module writes it down. Nothing here overrides the published
three-term estimate: at the reference conditions it reproduces it. What it
adds is the statement that two cuts taken at different conditions are cuts
at *different floors*, and therefore the `up to the floor` comparability
that Theorem `thm:target-equiv` relies on has to name *which* floor.

Consequence, and the reason this matters beyond bookkeeping: a
measurement reported without its conditions is not a measurement at a
known floor, so its comparability to any other measurement is undefined.
That is the same claim STRENDA makes for enzyme kinetics, arrived at from
the floor rather than from reporting practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- physical constants ---------------------------------------------------
K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K
HBAR_EV_S = 6.582119569e-16  # reduced Planck constant, eV*s

# --- reference conditions -------------------------------------------------
# The conditions at which the published floor estimate is stated. Chosen as
# standard biochemical assay conditions; T_REF is 25 C, the temperature most
# kinetic constants in the literature are reported at.
T_REF_K = 298.15
PH_REF = 7.4
VISCOSITY_REF_CP = 0.890  # water at 25 C, centipoise
T_INT_REF_S = 1.0e-3

RECURSION_DEPTH = 9  # categorical depth d, as in the address manifold
N_FUNCTORS = 6  # |S_3|, the conversion-functor cycle


@dataclass(frozen=True)
class Conditions:
    """An assay condition vector.

    These are exactly the fields a reporting standard asks for, which is
    the point: they are not metadata *about* the measurement, they are the
    specification of the floor the measurement was taken at.
    """

    temperature_K: float = T_REF_K
    pH: float = PH_REF
    viscosity_cP: float = VISCOSITY_REF_CP
    integration_time_s: float = T_INT_REF_S

    def __str__(self) -> str:
        return (
            f"T={self.temperature_K:.2f}K pH={self.pH:.2f} "
            f"eta={self.viscosity_cP:.3f}cP T_int={self.integration_time_s:.1e}s"
        )


# Biomolecular oscillators of R_bio. Frequencies are the dominant vibrational
# modes named in the leaf algebra (amide-I for residues, Fe-O for the heme
# centre, O-H libration for ordered water); Q_ref is the quality factor at the
# reference conditions.
OSCILLATORS = (
    {"name": "amide-I (residue)", "freq_hz": 5.1e13, "Q_ref": 1.0e3},
    {"name": "Fe-O (cofactor)", "freq_hz": 2.4e13, "Q_ref": 5.0e2},
    {"name": "O-H libration (solvent)", "freq_hz": 1.9e13, "Q_ref": 1.0e2},
    {"name": "C-N stretch (substrate)", "freq_hz": 3.3e13, "Q_ref": 8.0e2},
)


def floor_disc(d: int = RECURSION_DEPTH) -> float:
    """Discretisation floor: 1/(2*3^d). Condition-independent.

    This term is purely combinatorial -- it is set by the categorical depth
    the receiver resolves to, not by anything physical.
    """
    return 1.0 / (2.0 * 3**d)


def floor_conv(d: int = RECURSION_DEPTH, n_functors: int = N_FUNCTORS) -> float:
    """Conversion-functor floor: n_functors/3^d. Condition-independent."""
    return n_functors / 3.0**d


def q_factor(q_ref: float, freq_hz: float, cond: Conditions) -> float:
    """Q at given conditions, from Q at the reference conditions.

    Two damping channels, both of which lower Q as conditions depart from
    the reference:

      thermal -- an oscillator of energy h*f in a bath at kT is damped in
        proportion to its thermal occupation. Using the high-temperature
        form of the Bose factor, n ~ kT/(h*f), the ratio of occupations
        between T and T_ref is just T/T_ref. Q falls as 1/n.

      viscous -- collisional damping by the medium scales with viscosity,
        so Q falls as 1/eta.

    Both are first-order and neither is claimed to be exact; what matters
    for the floor argument is the *sign and existence* of the dependence,
    not its precise form. A better damping model changes the numbers and
    leaves the structure alone.
    """
    thermal_ratio = cond.temperature_K / T_REF_K
    viscous_ratio = cond.viscosity_cP / VISCOSITY_REF_CP
    return q_ref / (thermal_ratio * viscous_ratio)


def floor_Q(cond: Conditions) -> dict:
    """Allan-deviation oscillator floor at the given conditions.

    sigma_i = 1/(Q_i(cond) * sqrt(T_int * f_i)), summed in quadrature over
    the independent oscillator classes.
    """
    per = []
    for osc in OSCILLATORS:
        q = q_factor(osc["Q_ref"], osc["freq_hz"], cond)
        sigma = 1.0 / (q * math.sqrt(cond.integration_time_s * osc["freq_hz"]))
        per.append({"oscillator": osc["name"], "Q": q, "sigma": sigma})
    quad = math.sqrt(sum(p["sigma"] ** 2 for p in per))
    return {"per_oscillator": per, "quadrature_sum": quad}


def beta(cond: Conditions | None = None, d: int = RECURSION_DEPTH) -> float:
    """The floor at given conditions. beta(reference) is the published value."""
    cond = cond or Conditions()
    return floor_disc(d) + floor_Q(cond)["quadrature_sum"] + floor_conv(d)


def beta_breakdown(cond: Conditions | None = None, d: int = RECURSION_DEPTH) -> dict:
    """beta with its three terms itemised, for audit."""
    cond = cond or Conditions()
    fq = floor_Q(cond)
    disc, conv = floor_disc(d), floor_conv(d)
    total = disc + fq["quadrature_sum"] + conv
    return {
        "conditions": str(cond),
        "floor_disc": disc,
        "floor_Q": fq["quadrature_sum"],
        "floor_conv": conv,
        "beta": total,
        "per_oscillator": fq["per_oscillator"],
        "dominant_term": max(
            (("disc", disc), ("Q", fq["quadrature_sum"]), ("conv", conv)),
            key=lambda kv: kv[1],
        )[0],
    }


# --- comparability --------------------------------------------------------


def commensurable(value_a: float, cond_a: Conditions,
                  value_b: float, cond_b: Conditions) -> dict:
    """Are two measurements at different conditions comparable?

    This is the operation the framework could not previously state. Two
    cuts are commensurable when their difference exceeds the coarser of the
    two floors they were taken at:

        |v_a - v_b| > max(beta(cond_a), beta(cond_b))

    Below that, the values are the same measurement and any ranking
    between them is an artefact. Note the *coarser* floor governs -- a
    precise measurement compared against a sloppy one inherits the sloppy
    one's resolution, which is why reporting conditions is not optional.
    """
    b_a, b_b = beta(cond_a), beta(cond_b)
    governing = max(b_a, b_b)
    delta = abs(value_a - value_b)
    return {
        "beta_a": b_a,
        "beta_b": b_b,
        "governing_floor": governing,
        "delta": delta,
        "commensurable": delta > governing,
        "limited_by": "a" if b_a >= b_b else "b",
    }


def distinguishable_at(values: list[float], cond: Conditions) -> list[list[int]]:
    """Partition value indices into floor-indistinguishable groups.

    Single-linkage on the floor: values are grouped when consecutive sorted
    values differ by less than beta. Returned groups are index lists into
    the original `values`.

    This is `thm:no-zero-value` applied to a set of measurements. A group
    of size > 1 is a set of results that a screen cannot tell apart, and
    reporting them as ranked is reporting a distinction the receiver did
    not make.
    """
    b = beta(cond)
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    current: list[int] = []
    for pos, idx in enumerate(order):
        if not current:
            current = [idx]
            continue
        if abs(values[idx] - values[order[pos - 1]]) <= b:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    if current:
        groups.append(current)
    return groups
