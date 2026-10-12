import dynamic from "next/dynamic";
import StoryPage, { Scene, Interlude } from "@/components/scrolly/Story";
import { CycleRing, ThiolateStrip, MorseChart, ReturnChart, RateRule, LifetimeChart, PotentialChart, ObservableTable } from "@/components/scrolly/charts/compoundI";
import { HEME_MARKER } from "@/data/glbMarkers";

const GLBViewer = dynamic(() => import("@/components/GLBViewer"), { ssr: false });

const LEDGER = [
  ["Fe–S distance in P450s", "2.32 Å mean, 1.90–2.52 Å, 23 sites in 12 structures", "—", "C"],
  ["Compound I Soret band", "367 nm", "Rittle & Green 2010", "L"],
  ["Compound I Mössbauer δ, ΔE_Q", "0.11, 0.90 mm/s", "Rittle & Green 2010", "L"],
  ["O–O Morse parameters", "D_e 60 kcal/mol, r_e 1.45 Å, a 2.0 Å⁻¹", "set in the script", "M"],
  ["bonded / cleaved as one distinction", "ln 2 per recorded change", "counting of the foundation paper", "D"],
  ["concerted ÷ sequential rate", "10×", "follows from the rate rule", "M"],
  ["isotope effect, concerted / sequential", "2.0 / 6.0 (typed in)", "1.5–2.0, citation to be checked", "M"],
  ["lifetime at 193 K", "113 ms", "210 ms, CYP119", "M"],
  ["lifetime at 310 K", "1.4 µs", "no measurement; 1 ms is an estimate", "M"],
  ["oxidation potential", "0.54 V", "0.9 V, Green 2009 as cited", "M"],
];

export default function CompoundI() {
  return (
    <StoryPage
      meta={{ title: "Compound I", description: "Compound I formation by O–O heterolysis in cytochrome P450: what is measured, what the model computes, and where they differ." }}
      kicker="Part III · Paper 5 · Downstream chemistry"
      title="Compound I: the O–O bond breaks"
      lead={<>
        <p>
          After the second electron and two protons, the O–O bond of the iron-bound hydroperoxide breaks. One oxygen
          leaves as water; the other stays on the iron as Compound I, an iron(IV)-oxo porphyrin radical cation and the
          oxidant of the cycle.
        </p>
        <p>
          This page shows what the Compound I paper models, which of its numbers are computed and which are entered,
          and where the structure measured in the derivation enters.
        </p>
      </>}
      ledger={LEDGER}
      sources={<>Model values: <code>publications/catalytic-cycle/compound-i-formation</code> (tex, <code>validation/scripts</code>, <code>validation/results</code>). Ligand distances: the twenty structures of the P450 derivation manuscript.</>}
      prev={{ href: "/transfer", label: "Electron transfer through the reductase" }}
      next={{ href: "/ch-activation", label: "C–H activation and rebound" }}
    >
      <Scene n={1} title="Where Compound I sits in the cycle" chart={CycleRing} steps={[
        { tags: ["L"], body: <p>The cycle as usually drawn: resting low-spin iron(III) with a water ligand; substrate binding and the spin switch; two one-electron reductions with O₂ binding in between; two protonations; and Compound I, which oxidises the substrate.</p> },
        { tags: ["L", "M"], title: "The step this paper is about", body: <p>The peroxo-iron species takes a proton to give Compound 0, Fe(III)–OOH. A second proton on the distal oxygen and O–O cleavage release water and leave Compound I. The paper treats cleavage and proton arrival as one event rather than a sequence.</p> },
        { tags: ["L", "M"], title: "Each state has its own Soret band", body: <p>Resting (417 nm), substrate-bound (about 390 nm) and Compound I (367 nm, Rittle &amp; Green 2010) are measured values. The positions of the other states are the spectroscopic-atlas paper&apos;s assignments.</p> },
      ]} />

      <Scene n={2} title="The proximal ligand: cysteine thiolate" chart={ThiolateStrip} steps={[
        { tags: ["C"], body: <p>In the derivation we measured the iron&apos;s proximal ligand in twenty heme-protein structures. P450s hold the iron by cysteine sulphur; globins, peroxidases and cytochrome c by histidine nitrogen; catalase by tyrosine oxygen.</p> },
        { tags: ["C"], title: "Every P450 site", body: <p>All 23 heme sites in the 12 P450 structures: Fe–S 2.32 Å on average, from 1.90 to 2.52 Å. In every structure the bound cysteine is the one in the conserved cysteine-loop signature.</p> },
        { tags: ["L"], title: "Shared, not unique", body: <p>Chloroperoxidase and nitric-oxide synthase also bind the iron through a thiolate, and chloroperoxidase also forms Compound I. Electron donation from the thiolate is the usual explanation for heterolytic rather than homolytic O–O cleavage (Green 2009). The Compound I paper cites this work but does not model the effect.</p> },
      ]} />

      <Interlude
        title="The heme and its iron"
        note="The site's model of the P450 heme, with the iron marked. Drag to rotate. The axial thiolate lies below the porphyrin plane; Compound I forms on the open face above it."
      >
        <GLBViewer height={420} badge="heme · Fe" preset="heme" markers={HEME_MARKER} />
      </Interlude>

      <Scene n={3} title="The O–O bond" chart={MorseChart} steps={[
        { tags: ["M"], body: <p>The paper models O–O as a Morse oscillator, with D_e = 60 kcal/mol, r_e = 1.45 Å and a = 2.0 Å⁻¹ set in its script. The dashed curve is the harmonic approximation with the same curvature at the minimum.</p> },
        { tags: ["M"], title: "Anharmonic", body: <p>The potential rises faster on compression than on stretching. Every real bond behaves this way; nothing here is specific to P450.</p> },
        { tags: ["D", "M"], title: "Bonded or cleaved", body: <p>Whether the bond is intact is a two-valued question. Recording the answer is one binary commitment, ln 2 in the units of the foundation paper. That is counting applied to a bond: it says how much is recorded, not how fast the bond breaks.</p> },
      ]} />

      <Scene n={4} title="Does the bond come back?" chart={ReturnChart} steps={[
        { tags: ["M"], body: <p>Ten trajectories on the Morse potential, integrated for 1000 steps. For each, the closest approach to its starting point.</p> },
        { tags: ["D", "M"], title: "None returns exactly", body: <p>The foundation paper separates two cases. The configurations of a finite system recur exactly; a continuous anharmonic motion need not; the record of either does not recur. Ten runs illustrate the second case. They do not establish the paper&apos;s stronger statement that bond breaking is guaranteed.</p> },
      ]} />

      <Scene n={5} title="Rates from one rule" chart={RateRule} steps={[
        { tags: ["M"], body: <p>The paper gives every step a number d_C of simultaneous changes and the rate 10^(10 − d_C) s⁻¹. The rule is set in the model; it is not derived from the protein.</p> },
        { tags: ["M"], title: "Concerted or sequential", body: <p>Concerted proton-coupled electron transfer counts as one change and sequential as two. The predicted tenfold difference is the rule applied once.</p> },
        { tags: ["M", "L"], title: "Isotope effects", body: <p>The values 2.0 (concerted) and 6.0 (sequential) are typed into the script. The paper compares them with 1.5–2.0, attributed to Vatsis et al. 2002. That article is on ipso-substitution; check it for a Compound I formation isotope effect before quoting the comparison.</p> },
      ]} />

      <Scene n={6} title="How long Compound I lives" chart={LifetimeChart} steps={[
        { tags: ["M"], body: <p>An Eyring-type expression with one activation energy, E_a = 11 kcal/mol, set in the script.</p> },
        { tags: ["M", "L"], title: "At 193 K", body: <p>The model gives 113 ms. The measured value it is compared with is 210 ms for CYP119 Compound I (Rittle &amp; Green 2010): agreement within a factor of two.</p> },
        { tags: ["M"], title: "At body temperature", body: <p>The model gives 1.4 µs. The 1 ms in the paper&apos;s table is an estimate from turnover numbers, not a measurement, so at 310 K there is nothing direct to compare with.</p> },
      ]} />

      <Scene n={7} title="How strongly Compound I oxidises" chart={PotentialChart} steps={[
        { tags: ["M"], body: <p>The paper adds four contributions: spin state, oxidation state, the Fe=O bond and the porphyrin radical. Each is a ΔM value chosen in the script; the sum is scaled by k_BT/e and n_eff = 5.</p> },
        { tags: ["M", "L"], title: "Against the cited value", body: <p>The result is 0.54 V, against the 0.9 V it is compared with (Green 2009). The abstract calls this agreement; the gap is 0.36 V.</p> },
        { tags: ["M"], title: "What the gap depends on", body: <p>n_eff ≈ 8.4 would close it. Neither n_eff nor the four ΔM values is derived, so the agreement or disagreement here does not test the framework.</p> },
      ]} />

      <Scene n={8} title="What has been measured on Compound I" chart={ObservableTable} steps={[
        { tags: ["L"], body: <p>The paper&apos;s comparison table lists eight Compound I observables. The measured column (Soret 367 nm, Mössbauer δ 0.11 and ΔE_Q 0.90 mm/s, g ≈ 1.99, S = ½) is the Rittle–Green characterisation.</p> },
        { tags: ["M"], title: "Listed against computed", body: <p>Where the scripts compute a value, it is not the one in the table&apos;s &quot;predicted&quot; column: 0.54 V rather than 0.9 V, 113 ms rather than 200 ms, 1.4 µs rather than 1 ms.</p> },
        { tags: ["L"], title: "The other five", body: <p>These are not computed anywhere in the code; they are the literature values. On this page they appear as measurements of Compound I, not as model results. The resonance Raman and ENDOR values listed on the earlier version of this page are not in the paper.</p> },
      ]} />
    </StoryPage>
  );
}
