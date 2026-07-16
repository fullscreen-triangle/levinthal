# The Shakespeare Sandbox — Design & Lesson Ladder

*A VSCode-like IDE that teaches the Shakespeare receiver language by
performing the Cytochrome P450 monograph, one `.shk` play at a time.*

Status: design document. No runtime code committed yet — this is the
blueprint the sandbox is built against.

---

## 1. The core idea

The Cytochrome P450 monograph (`cytochrome/publications/`, ~19 papers across
12 themes, **156 Python validation scripts**) is not just documentation — it
is a **ready-made curriculum with a built-in answer key**:

- Each **paper** = one lesson's *why* (prose, figures, the derivation).
- Each paper's **validation scripts** already compute the exact numbers the
  paper claims, and emit them as `validation/results/*.json` with a
  `verdict: PASS`. These are the **oracles**.
- Each paper's **`figures/` panels** (`public/panels/paper-N/`) are the
  charts — we re-draw them live in **D3**.

So the sandbox does not invent tutorial content or expected outputs. It
**transcribes the monograph into runnable `.shk` plays** and checks each
play's output against the paper's already-validated JSON.

The monograph even names itself for this: its prefatory paper is titled
*"On the Condition in Which Shakespeare Has Access to a Macbook, the Result
Is Shakespeare."* The sandbox literally gives Shakespeare (the language) a
Macbook (the IDE). The thesis — *the protein is evaluating a receiver, not
searching* — is exactly what a `.shk` play performs.

---

## 2. Two commitments that shape everything

### 2.1 TypeScript-first — the sandbox *is* the WebGL/TS target

We do **not** build the Rust interpreter yet. The Shakespeare spec
(`models/sources/shakespeare-language.tex`) already defines **two targets**
from one shared front end and one back-end-agnostic Cut-IR:

- **Target R (Rust)** — exact max-flow min-cut. *Deferred.*
- **Target S (TypeScript / WebGL2)** — cuts as render passes, "the online
  Shakespeare tool." **This is the sandbox.**

By Theorem `thm:target-equiv` the two agree up to the floor, so building
only Target S now costs nothing in correctness — the Rust oracle can slot in
later behind the identical Cut-IR interface. The `shakespear/` directory
(Next.js + `@react-three/fiber` + `d3` already in `package.json`) is its
home.

The runtime for the tutorial is a **minimal in-browser interpreter** for the
lesson subset of the language: `receiver`, `floor`, the leaf verbs, `~`,
`close`/`complex`, `fold`, `track…until converge`, `catalyze`, `mutate`,
`complete`, `observe`. It parses `.shk` → Cut-IR → evaluates against an
in-memory contact graph. Where a computation is heavy (Kuramoto fold, the
full 7-state solve), the interpreter **loads the paper's oracle JSON** as the
authoritative result rather than recomputing — the oracle is ground truth, and
this keeps the tutorial exact and fast. The interpreter's own arithmetic
(addresses, cut counts, residues, coherence) is checked against those same
JSONs so "✓ matches the monograph" is a real assertion, not decoration.

### 2.2 Progressive complexity = one accumulating receiver

This is the important structural decision, and it is *forced by the language's
own semantics*, not chosen for pedagogy.

The operational semantics (`sec:opsem`) says:

- **The committed-cut count `M` is a monotone intrinsic clock**
  (Thm `thm:monotone`): it never resets; re-evaluation is a new cut at a
  higher count, never a cached recomputation.
- **Measurement is reshuffling that conserves the medium**: individuating a
  new part accrues boundary onto the *same* contact graph.

Therefore the sandbox is a **REPL over one growing receiver**, not a set of
isolated scripts. Lessons run **in order**, and the contact-graph / receiver
state from lesson *N* **persists into lesson *N+1***:

```
Lesson 1  cut the residue leaves of CYP3A4        → leaves live in the graph
Lesson 2  fold "1TQN"                              → uses those leaves; M grows
Lesson 3  complex CYP3A4(heme, CYS442, H2O)        → binds onto the fold
Lesson 4  track O2 in E until converge             → propagates through the complex
...
```

"No memory issues as the items from the previous experiment count in the new
one" (the user's phrase) is exactly this: nothing is re-cut from scratch, the
medium is conserved, and each lesson is cheaper because the graph is already
built. A **"reset receiver"** control exists for starting the play over, but
the default is accumulate — the intrinsic clock only moves forward.

A left-rail **State Inspector** shows the growing receiver: current leaf count
by class, committed-cut count `M`, floor `β`, and the live contact-graph size.
Watching `M` tick up across lessons *is* the tutorial's spine.

---

## 3. IDE layout (VSCode-resemblant)

```
┌─────────────┬──────────────────────────────┬───────────────────────┐
│  LESSONS     │   EDITOR  (Monaco, .shk)     │   MONOGRAPH PANE       │
│  (file tree) │   syntax-highlit Shakespeare │   the bound paper's    │
│              │                              │   prose + figure, the  │
│  00 hello    │   receiver bio               │   "why" for this play  │
│  01 address  │   floor 3.7e-4               │                        │
│  02 fold  ◀  │   C := fold "1TQN"           │                        │
│  03 resting  │   observe C                  │                        │
│  ...         │                              │                        │
│              │  [ ▶ Perform ]  [↺ reset]    │                        │
│ ── STATE ──  ├──────────────────────────────┴───────────────────────┤
│ leaves 503   │   CONSOLE (terminal)          │   CHARTS (D3)          │
│ M = 1 240    │   ▸ perform 02_fold           │   ┌─ contact map ──┐   │
│ β = 3.7e-4   │   fold "1TQN" : Fold @ 3.7e-4 │   │ NxN heatmap    │   │
│ graph 4061v  │   helices 13  sheets 5        │   │                │   │
│              │   coherence 0.83  M += 226    │   └────────────────┘   │
│              │   ✓ matches Paper 9 oracle    │   ┌─ S-entropy 3D ─┐   │
│              │     (13 helix / 5 sheet)      │   │ (react-three)  │   │
│              │                              │   └────────────────┘   │
└─────────────┴──────────────────────────────┴───────────────────────┘
```

- **Lessons rail** — the ladder (§5), ordered; a lesson unlocks when the prior
  one has been performed (progressive).
- **Editor** — Monaco with a Shakespeare language definition (reuse the
  keyword set from the `.tex` listing style: `receiver floor cut contact close
  complex fold track until yield when observe catalyze mutate complete …`).
- **Monograph pane** — renders the bound paper's abstract + the relevant
  figure. This is where the depth of the monograph pays off: the learner reads
  the real derivation beside the play that performs it.
- **Console** — the accountable values as terminal text (see §4).
- **Charts** — D3 (2D: contact map, order-parameter curve, orbit, bar charts;
  `@react-three/fiber` for the 3D S-entropy scatters, mirroring the panels).
- **State inspector** — the accumulating receiver (§2.2).

---

## 4. Execution output: console + D3 charts

Running a `.shk` (the **Perform** button) produces two synchronized outputs.

### 4.1 Console (terminal) — the accountable values

Every `observe`d value prints as its accountable form: type, floor, residue,
the cut-count delta, and a **✓ against the oracle**. Example for lesson 4
(seven-state cycle), whose oracle is
`synthesis/seven-state-closed-orbit/validation/results/01_seven_states.json`:

```
▸ perform 04_catalytic-cycle
track O2 in E until converge yield cycle
cycle : Path @ 3.7e-4
  states       7   (Resting → SubBound → Reduced → Oxy → Peroxo → Cpd0 → CpdI)
  transitions  8   ΔM sum = 4.963
  orbit        closed (7 → 1, product release)
  M += 8       (clock now 1 248)
  ✓ matches synthesis/seven-state-closed-orbit  [verdict PASS]
```

The console reads the oracle JSON's fields directly (`n_states`, `DM_sum`,
`orbit_closed`, `verdict`) — the numbers on screen are the monograph's own.

### 4.2 Charts — D3, mirroring the monograph panels

Each lesson declares which chart(s) it draws. All are plain D3 (2D) or
react-three (3D), fed by the interpreter's output / oracle JSON — **no PNGs**,
they render live and update as the learner edits the script:

| Chart | D3 form | Used by lessons |
|---|---|---|
| **Contact map** | `NxN` heatmap (`d3.scaleSequential`) | fold, resting, variant |
| **S-entropy scatter** | 3D point cloud in `[0,1]³` (react-three) | address, isoforms |
| **Order parameter** | line `r(t)` vs step, threshold at 0.8 | fold |
| **Seven-state orbit** | closed ring, nodes = states, edge width = ΔM | catalytic-cycle, compound-i |
| **ΔM / rate bars** | horizontal bars per transition | catalytic-cycle, rebound |
| **Depth ladder** | address trits at depth 3 / 6 / 9 | address, isoforms, variant |
| **Electron-transfer trace** | multi-hop path vs fs time | electron-chain |

These are exactly the `figures/*captions.tex` panels the monograph already
specifies, so the captions become the chart tooltips/legends for free.

---

## 5. The lesson ladder

Ordered by **categorical depth**, exactly as the monograph introduction orders
itself: *depth-3 taxonomy → depth-6 isoforms → depth-9 variants → 6-step fold
→ 7-state orbit → Compound I → 14 reactions → fs electron transfer.* Each
lesson = one `.shk` file + the paper it performs + the oracle it is checked
against. State accumulates down the ladder (§2.2).

| # | `.shk` lesson | New verb / idea | Monograph paper (dir) | Oracle (results/) |
|---|---|---|---|---|
| 0 | `00_hello-receiver` | `receiver bio`, `floor`, one leaf, `observe` | introduction | — (prints M, β) |
| 1 | `01_sequence-address` | `residue`; address at depth 3/6/9 | foundations/p450-sequence-space · manifold | `06_cyp3a4_address` → `110112212`, 503→4527 trits |
| 2 | `02_fold-cyp3a4` | `fold "1TQN"`; 6-step fold, contact map | manifold/…-cyp3a4-fold | manifold JSONs (13 helix / 5 sheet) |
| 3 | `03_resting-state` | `complex`, cofactor/solvent leaves, receiver tree | equilibrium-states/cyp3a4-resting | resting-bound JSONs |
| 4 | `04_catalytic-cycle` | `track … until converge`; the 7 states | synthesis/seven-state-closed-orbit | `01_seven_states` (ΔM sum 4.963, closed) |
| 5 | `05_compound-i` | Miracle Principle (transient intermediate licensed by closure) | catalytic-cycle/compound-i-formation | compound-i JSONs |
| 6 | `06_ch-rebound` | `catalyze` (reaction = measurement) | catalytic-cycle/ch-activation-rebound | rebound JSONs |
| 7 | `07_electron-chain` | electronic leaves; fs ET trajectory | catalytic-cycle/multi-hop-et-chain | multi-hop JSONs |
| 8 | `08_isoform-diversity` | depth-6 family enumeration | diversity/57-human-isoforms · synthesis/57-isoform-taxonomy | isoform-taxonomy JSONs |
| 9 | `09_variant-effect` | `mutate`; PharmVar variants at depth 9 | diversity/polymorphisms-ddi · pharmacology | pharmacogenomics JSONs |
| 10 | `10_db-recovery` | `complete` (trajectory completion / empty-dictionary) | informatics/database-recovery | database-recovery JSONs |

Optional later rungs (the monograph has the material): `spectroscopy/`
(`observe` as rendering-measurement, 2D-IR chart), `construction/membrane-cpr`
(membrane + CPR cofactor leaves), `reactions/` (14-reaction atlas).

---

## 6. The binding: how a `.shk` lesson finds its paper and oracle

Each lesson is a small directory (or a front-matter header in the `.shk`)
declaring three links, so the IDE can wire editor ↔ prose ↔ oracle:

```
shakespear/lessons/04_catalytic-cycle/
  play.shk          -- the Shakespeare program
  lesson.json       -- { title, paper, oracle, charts, requires: "03" }
```

`lesson.json` for lesson 4:

```json
{
  "id": "04_catalytic-cycle",
  "title": "The seven-state closed orbit",
  "paper": "cytochrome/publications/synthesis/seven-state-closed-orbit/closed-orbit.tex",
  "oracle": "cytochrome/publications/synthesis/seven-state-closed-orbit/validation/results/01_seven_states.json",
  "charts": ["seven-state-orbit", "dm-bars"],
  "requires": "03_resting-state",
  "verbs_introduced": ["track", "until", "converge", "yield"]
}
```

- `paper` → rendered in the Monograph pane.
- `oracle` → the JSON the console diffs against (`verdict`, named fields).
- `charts` → which D3 components mount in the Charts pane.
- `requires` → enforces the accumulate-forward order; the receiver from the
  required lesson must already be in state.

The build step can generate `lessons/*/lesson.json` automatically by walking
`cytochrome/publications/*/validation/results/` — the ladder stays in sync
with the monograph with no hand maintenance.

---

## 7. Why this is the fast path

- **Curriculum:** free — it's the monograph's own narrative arc.
- **Answer key:** free — 156 validation scripts already emit
  `verdict: PASS` JSON with the exact claimed numbers.
- **Charts:** free — every paper already specifies its panels in
  `figures/*captions.tex`; we redraw them in D3.
- **Runtime:** one target (TS/WebGL), and heavy computations replay the oracle
  rather than re-deriving, so the language runtime can be minimal and still
  exact.
- **Correctness:** guaranteed against the monograph, and (later) against the
  Rust oracle behind the same Cut-IR by Thm `thm:target-equiv`.

The sandbox is, in one line: **a REPL over a single accumulating P450 receiver,
whose lessons are the monograph performed as Shakespeare plays, checked live
against the monograph's own validated numbers, with console output and D3
charts.**

---

## 8. Build order (when we implement)

1. **Lesson content first** — author the 11 `play.shk` + `lesson.json` files
   (they exist independent of any IDE; the interpreter can be stubbed with
   oracle-replay).
2. **Monaco + Shakespeare language def** — highlighting, the lesson rail, the
   monograph pane (render the bound `.tex` abstract/figure).
3. **Minimal TS interpreter** — parse → Cut-IR → evaluate the lesson subset
   over one persistent contact graph; oracle-replay for heavy ops; ✓-diff.
4. **D3 chart components** — contact-map heatmap, order-parameter line,
   seven-state orbit, ΔM bars, depth ladder; react-three S-entropy scatter.
5. **State inspector + accumulate-forward wiring** (`requires`, monotone `M`).
6. (Later) **Rust Target R** behind the same Cut-IR as the exact oracle.
