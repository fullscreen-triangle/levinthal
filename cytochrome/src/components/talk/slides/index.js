import { S01Title, S02Counted, S03Zeno, S04Kac } from "./derivation1";
import { S05Individuation, S06Ladders, S07Atoms } from "./derivation2";
import { S08Bonds, S09Order, S10Heme } from "./derivation3";
import { S11Structure, S12Isoforms, S13Equilibrium, S14Delivery, S15CompoundI } from "./applications1";
import { S16CH, S17Atypical, S18Cycle, S19Pharmacology, S20Informatics } from "./applications2";

// steps = number of extra builds after the slide first appears.
const SLIDES = [
  {
    id: "title", part: "I", short: "Title and the chain of conditions", Component: S01Title, steps: 1,
    notes: "Thank the group. The talk has two halves: ten slides derive a P450 from one assumption, ten summarise the application papers built on it. The chain on screen is the order of the derivation; each link is labelled derived, supplied or checked in the manuscript. Second build: what exists — two manuscripts, ten validation experiments on public data, seventeen application papers.",
  },
  {
    id: "counted", part: "I", short: "Counted vs measured sums", Component: S02Counted, steps: 2,
    notes: "The base statement. Positive is not enough: 1/2 + 1/4 + … is finite. Integers carry a floor of one. Theorem: positive terms have inf > 0 exactly when every infinite selection diverges. The only physical assumption of the whole talk is that a record costs at least θ > 0 — the Landauer bound is its physical counterpart, not something we derive.",
  },
  {
    id: "zeno", part: "I", short: "Occupation vs commitment; Zeno; direction", Component: S03Zeno, steps: 2,
    notes: "Occupation — being somewhere — is free in this accounting; a record is not. Zeno's dichotomy uses both sums: distances converge (calculus answer), the commitments needed to distinguish every midpoint diverge. The runner arrives; an observer committing midpoints never finishes, but needs only ⌈log₂ D/ε⌉ commitments for any resolution ε. Third build: the register; its count orders events — that is the direction.",
  },
  {
    id: "kac", part: "I", short: "Kac ring: configurations recur, records do not", Component: S04Kac, steps: 2,
    notes: "Kac ring: reversible, finite, periodic. Entropy of the colour fraction rises, then the configuration recurs exactly. Second build: reverse the dynamics — the configuration and its entropy return, while the register keeps counting. Count, time and register entropy are proportional: M = f t = S/(k_B ln b). Loschmidt's reversal acts on configurations, not on M. This does not replace the statistical account of the second law.",
  },
  {
    id: "individuation", part: "I", short: "Individuation as a min-cut against the medium", Component: S05Individuation, steps: 2,
    notes: "Separation cost: the cheapest cut between an item and its medium — a max-flow, computed live here. With counted weights it is at least one and never decreases as contacts are added. Second build: real weights — monotonicity still holds, but the floor drifts towards zero; an earlier lemma of mine assumed positivity was enough; the integer version is what holds. Third: identity is a region, and measuring adds contacts, so it cannot lower σ.",
  },
  {
    id: "ladders", part: "I", short: "Ladders: counted sums in log coordinates", Component: S06Ladders, steps: 2,
    notes: "Stepwise refinement: each rung closes a fraction of what remains. In logs the gap is a sum of positive terms. With a floor the gap goes to zero but never closes in finitely many rungs; without one it can stall (closed form sin(π/√2)/(π/√2)). A composite does not identify its ladder — for P450 the composite is the coupling efficiency and the rungs are the intermediates.",
  },
  {
    id: "atoms", part: "I", short: "Shells, closures and the iron d shell", Component: S07Atoms, steps: 2,
    notes: "Supplied inputs are named: spin, exclusion, the empirical Madelung filling order. With them, counted capacities 2(2ℓ+1) give closures at 2, 10, 18, 36, 54, 86 — exactly where NIST first ionization energies drop most — and all seven period lengths. Iron: Fe(III) d5 and Fe(II) d6 are where high- and low-spin arrangements differ by the most unpaired electrons; we do not claim this selects iron (Mn(II), Co(III) share it).",
  },
  {
    id: "bonds", part: "I", short: "Bonds and the peptide plane", Component: S08Bonds, steps: 2,
    notes: "A bond as a contact that makes the pair cheaper to cut as one: w(a,b) > ½ max(cut a, cut b). Tetrahedral 109.47° from maximal separation; measured N–Cα–Cβ is refinement-restrained, so geometry only. ω is weakly restrained: 99.2% trans, 7 cis, 35 between of 5,406 peptide bonds in twelve P450s. φ/ψ are unrestrained.",
  },
  {
    id: "order", part: "I", short: "Boundary → record → catalyst", Component: S09Order, steps: 2,
    notes: "A dependency order, explicitly not a history. Boundary needs one amphiphile; record needs pairing and a boundary to persist; a 20-letter catalyst cannot be copied by pairing and needs a record. One test each: 93% of 532 ER-membrane P450s carry a bilayer-spanning N-terminal window, 0.7% of 145 bacterial; CYP17A1 is an exception at this threshold. Codons: 4² < 21 ≤ 4³; the standard code is in the best 0.78% of 200,000 permuted codes for hydropathy change.",
  },
  {
    id: "heme", part: "I", short: "Compact fold, buried heme, thiolate ligand", Component: S10Heme, steps: 3,
    notes: "P450 chains have folded, not denatured, radii; the heme keeps 1.5–6.4% of its accessible surface. In the integer contact graph of P450cam the heme's separation cost (306) exceeds every residue's. Every P450 iron in 20 structures is bound by Cys Sγ; the ligand Cys is the PROSITE PS00086 cysteine in 12/12 structures; the signature matches 82.6% of reviewed P450s and 0.07% of other heme proteins (the few hits carry a P450 domain). Last build: substrate displaces the distal water — the spin switch — and P420 as a refused rung.",
  },
  {
    id: "structure", part: "II", short: "GLB input and ternary addressing", Component: S11Structure, steps: 2,
    notes: "Part II: application papers. Each slide marks values as checked here, literature, or model values. The GLB-derived iron geometry falls inside the crystal-structure ranges measured in part I. Ternary addresses: capacity 3^k against 18 families and 57 isoforms.",
  },
  {
    id: "isoforms", part: "II", short: "57 human isoforms, 18 families", Component: S12Isoforms, steps: 2,
    notes: "The human complement as a packed hierarchy. Filled: the main drug-metabolising isoforms. The papers report family recall 0.94 at depth 3 and isoform distinctness 0.97 at depth 6 — model results, to be read as such.",
  },
  {
    id: "equilibrium", part: "II", short: "Spin switch and the electron gate", Component: S13Equilibrium, steps: 2,
    notes: "Connects directly to the derivation: the crystal structures show water displaced and iron moving out of plane. The paper models the CYP3A4 step and its reduction-potential shift, reported near +120 mV.",
  },
  {
    id: "delivery", part: "II", short: "Membrane anchors and electron delivery", Component: S14Delivery, steps: 2,
    notes: "Anchor from part I; CPR and P450 on one membrane; electron NADPH → FAD → FMN → heme. Moser–Dutton gives a standard distance check. Be open that the two papers use different FMN–heme distances (14 vs 18 Å), which matter by over two orders of magnitude in rate.",
  },
  {
    id: "compound-i", part: "II", short: "Compound I formation", Component: S15CompoundI, steps: 2,
    notes: "Second protonation and heterolysis. The thiolate from part I is why this is heterolytic. Observables compared with Rittle and Green 2010. The Raman value is as cited in the spectroscopic atlas — check the primary source before quoting it.",
  },
  {
    id: "ch", part: "II", short: "C–H activation, rebound, heteroatoms", Component: S16CH, steps: 2,
    notes: "HAT then rebound (Groves). KIE 7.2 within the reported 4–11. Heteroatom oxidation ordered by the same parameter: direct O-transfer below H-abstraction-based dealkylation.",
  },
  {
    id: "atypical", part: "II", short: "Atypical reactions", Component: S17Atypical, steps: 1,
    notes: "Five atypical classes on one scale; only desaturation carries a primary KIE in the model. The rate order is compared class by class with the literature.",
  },
  {
    id: "cycle", part: "II", short: "The cycle and its spectra", Component: S18Cycle, steps: 1,
    notes: "Seven states, each distinguishable; Soret positions from the atlas, with literature values for the resting, substrate-bound and Compound I states. Read as a ladder: coupling efficiency is the composite. I leave out absolute time scales here.",
  },
  {
    id: "pharmacology", part: "II", short: "Alleles, inhibitors, inactivators", Component: S19Pharmacology, steps: 3,
    notes: "Standard pharmacology expressed in one parameter: phenotypes, α = 1 + [I]/Ki for ketoconazole, quinidine and fluoxetine, Kitz–Wilson for clarithromycin. Compared with reported clinical magnitudes.",
  },
  {
    id: "informatics", part: "II", short: "Addressing, recovery, and what this offers", Component: S20Informatics, steps: 1,
    notes: "Constant resident state; capacity at k = 6 above the 57-isoform entropy. Then the offer — framed as things to test together, not claims: pre-screens, separation-cost ranking for mutagenesis checked against their variant data, substitution accounting for libraries, coupling as a ladder for engineered P450s and BVMOs.",
  },
];

export default SLIDES;
