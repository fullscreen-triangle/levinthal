# Ping-pong bi-bi and the conditioned floor

Two extensions to R_bio, prompted by a field report on current problems in
enzyme research (PLP-dependent transaminases, high-throughput screening,
and the reporting standards the catalysis community has not adopted).

```bash
cd validation && python run_all.py     # 34 pass · 1 xfail · 0 fail
```

---

## 1. What the framework could not express, and now can

### Ping-pong bi-bi

`catalyze s in E yield r` assumes **one substrate inside one enzyme**, and
the sandbox interpreter commits a fixed two cuts for it
([shakespeare.js:193](../../../cytochrome/src/helpers/shakespeare.js#L193)).
That shape fits the cytochrome cycle, where the substrate enters, turns
over, and leaves while the enzyme returns unchanged.

Ping-pong bi-bi is a different topology, and three things break the
single-substrate form:

| | Cytochrome cycle | Ping-pong bi-bi |
|---|---|---|
| Complex | ternary (E·S·O₂) | **none** — first product leaves before second substrate binds |
| Enzyme between steps | unchanged | **chemically modified** (E-PLP → E-PMP → E-PLP) |
| Cofactor | participant in the cycle | **bound carrier, participant in nothing** |

The third is the one with consequences for the framework. PLP never
leaves, so it appears in no reaction equation — Rhea gives four
participants for each transaminase, none of them PLP. That means
**`participant` is not the same predicate as `leaf present in the
complex`**, and R_bio had no way to say so. Here a carrier leaf is
committed once at construction and never re-cut per turnover, while
participants are cut on every pass.

Cut costs are derived from the topology (`residue = floor × boundaries`),
not tabulated — which matters, because the existing sandbox **replays baked
oracle numbers** rather than computing (its own header says so). This
module computes.

### The conditioned floor

The published floor is a three-term estimate, and every downstream artefact
treats it as a scalar (`RECEIVER_FLOOR = 3.7e-4`, `floor 3.7e-4` in the
plays). But the middle term is an Allan-deviation oscillator floor,
`σ = 1/(Q√(T_int·f))`, and **Q is not a constant** — it falls with thermal
occupation and with medium damping. So β was already a function of
temperature, viscosity and integration time; the framework simply never
wrote the dependence down.

Writing it down yields the operation R_bio was missing: **cross-condition
comparability.** Two cuts are commensurable when their difference exceeds
the *coarser* of the two floors. Pairing a precise measurement with a
sloppy one inherits the sloppy resolution — which is the same conclusion
reporting standards reach, arrived at from the floor instead of from
practice.

---

## 2. The finding: the conditioned floor does not matter at depth 9

**This is the result, and it refutes the extension's most attractive claim.**

At the categorical depth the address manifold uses (d = 9):

```
floor_disc = 1/(2·3⁹) = 2.54e-05
floor_Q                = 7.41e-08     ← the only condition-dependent term
floor_conv = 6/3⁹      = 3.05e-04     ← dominates by ~4100×
```

Varying temperature across the entire biochemical range (4 °C → 37 °C)
moves β by **0.0025%**. The dependence exists, has the right sign, and is
unmeasurable. A check asserting only `β(37°C) > β(4°C)` passes on
floating-point noise — the first version of this suite did exactly that,
and reported 28/28 green while two of its checks compared a number to
itself.

The conditioned floor only governs where the Q term dominates, which is
**short integration times**: at `T_int = 1e-15 s` (a femtosecond gate, the
regime of the ET-chain paper) the same temperature swing moves β by
**10.3%**.

So the honest scope is narrower than the idea: conditions govern the floor
in fast-gated measurement, and are irrelevant to steady-state assays at
depth 9. That check is marked `XFAIL` — it fails on purpose, and the suite
fails if it ever starts passing, because that would mean this finding has
gone stale.

---

## 3. Layout

```
kernel/
  conditioned_floor.py   β(conditions); commensurability; floor-grouping
  pingpong.py            two-half-reaction cut chain; carrier vs participant
validation/
  run_all.py             35 checks; writes results/_summary.json
```

## 4. Validation fixtures

The three transaminase reactions, with participant sets verified against
Rhea and ChEBI (source: `tacat-sources/nfdi4cat-sources/reference/
verified-identifiers.csv`, checked first-hand against live ChEBI):

| EC | Rhea | Reaction |
|---|---|---|
| 2.6.1.1 | RHEA:21824 | L-aspartate + 2-oxoglutarate = oxaloacetate + L-glutamate |
| 2.6.1.2 | RHEA:19453 | L-alanine + 2-oxoglutarate = pyruvate + L-glutamate |
| 2.6.1.3 | RHEA:17441 | L-cysteine + 2-oxoglutarate = 2-oxo-3-sulfanylpropanoate + L-glutamate |

The load-bearing external fact is negative: **PLP is a participant in none
of them.** Three checks assert it.

## 5. Screening consequence

`distinguishable_at` partitions a plate into floor-indistinguishable
groups. Seven variants whose activities include four clustered within one
β come back as **three groups, not seven** — reporting them as ranked
would report a distinction the receiver did not make. This is
`thm:no-zero-value` applied to a plate.

## 6. What this does not do

- **No kinetics.** The framework's entire kinetics surface is still
  `log10(kcat/KM) = 10 − dC`. Nothing here computes a rate, a Km, or a
  temperature-dependent kcat. Ping-pong *topology* is not ping-pong
  *kinetics* — the classic Cleland rate equation is absent.
- **No Shakespeare syntax.** The kernel is Python. No `.shk` verb, no
  grammar change, no type-checker rule. Wiring a `pingpong` verb into the
  interpreter is a separate piece of work.
- **Damping model is first-order.** The Q(T, η) form uses a
  high-temperature Bose factor and linear viscous damping. What is claimed
  is the *existence and sign* of the dependence, not its precise form; a
  better damping model changes the numbers and leaves the structure alone.
- **No experimental validation.** The topology is checked against Rhea
  participant sets. Nothing here has been compared to a measured
  transaminase progress curve.
