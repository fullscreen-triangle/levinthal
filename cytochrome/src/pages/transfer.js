import dynamic from "next/dynamic";
import StoryPage, { Scene, Interlude, Tag } from "@/components/scrolly/Story";
import { ChainDiagram, DistanceRate, HopRates, EfficiencyChart, SemiquinoneChart, OccupancyChart } from "@/components/scrolly/charts/transfer";
import { C } from "@/components/scrolly/core";

const ElectronTransferViewer = dynamic(() => import("@/components/ElectronTransferViewer"), {
  ssr: false,
  loading: () => (
    <div className="flex w-full items-center justify-center rounded-xl text-xs uppercase tracking-widest" style={{ height: 520, background: C.bg, color: C.muted }}>
      Loading 3-D model…
    </div>
  ),
});

const LEDGER = [
  ["cofactor spacings NADPH–FAD–FMN–heme", "4, 4, 14 Å (18 Å in one equation)", "crystal structures, as cited", "M"],
  ["Moser–Dutton rate at 14 Å / 18 Å", "5.8×10⁶ / 2.3×10⁴ s⁻¹", "empirical tunnelling ruler", "S"],
  ["the paper's Marcus rate, FMN → heme at 14 Å", "610 s⁻¹", "5×10⁷ s⁻¹, Denisov 2007 as cited", "M"],
  ["hop rates used", "600 / 10⁷ / 5×10⁷ s⁻¹", "Murataliev 1999; none; Denisov 2007", "L"],
  ["chain in series", "600 s⁻¹, set by the NADPH hop", "text names the FMN → heme hop", "M"],
  ["k_cat/K_M from four sequential steps", "10⁶ M⁻¹s⁻¹", "0.7–4 ×10⁶, four CYP3A4 substrates", "M"],
  ["flavin semiquinone formation", "+2.8 kcal/mol", "−3 kcal/mol as quoted, no citation", "M"],
  ["selection rules satisfied", "2 of 4 transitions", "text reports 4 of 4", "M"],
  ["reorganisation energy λ", "0.85 eV, set in the code", "0.7–1.0 eV as quoted", "M"],
  ["hop rates in the 3-D animation", "2–6 ×10¹² s⁻¹", "chosen for display", "M"],
  ["N-terminal membrane anchor", "93 % of 532 ER P450s, 0.7 % of 145 bacterial", "derivation manuscript", "C"],
];

export default function Transfer() {
  return (
    <StoryPage
      meta={{ title: "Transfer", description: "Electron transfer from NADPH through cytochrome P450 reductase to the P450 heme: distances, rates and how the paper's numbers compare with a standard tunnelling ruler." }}
      kicker="Part III · Paper 4 · Electron delivery"
      title="Two electrons, four cofactors, one long hop"
      lead={<>
        <p>
          Every P450 turnover needs two electrons, delivered one at a time from NADPH by cytochrome P450 reductase.
          The electron passes FAD and FMN and then crosses to the heme of a second protein on the same membrane.
        </p>
        <p>
          The transfer paper follows that path. This page sets its distances and rates beside the standard tunnelling
          ruler and the measured rates it uses, and marks each number as measured, supplied, modelled or set.
        </p>
      </>}
      ledger={LEDGER}
      sources={<>Model values: <code>publications/catalytic-cycle/multi-hop-et-chain</code> (tex, <code>validation/scripts/_common.py</code>, <code>validation/results</code>) and <code>glb/levinthal_glb/shader_pipeline.py</code>. Ruler: Moser et al., Nature 1992; Page et al., Nature 1999.</>}
      prev={{ href: "/equilibrium", label: "Resting and substrate-bound CYP3A4" }}
      next={{ href: "/compound-i", label: "Compound I formation" }}
    >
      <Scene n={1} title="Four cofactors, two proteins" chart={ChainDiagram} scrub steps={[
        { tags: ["L", "M"], body: <p>From NADPH, electrons pass through the reductase&apos;s two flavins, FAD then FMN, to the P450 heme. The spacings drawn (4, 4 and 14 Å) are those in the paper&apos;s code. Scroll to move the electron along.</p> },
        { tags: ["M"], title: "One distance, two values", body: <p>The FMN–heme distance is 14 Å in the code and most of the text, but 18 Å in the paper&apos;s crystallographic equation and one figure caption. On the tunnelling ruler in the next scene the two values differ by 2.4 decades in rate, so the paper needs to choose one.</p> },
        { tags: ["C", "L"], title: "One membrane", body: <p>Both proteins are anchored in the ER membrane by an N-terminal helix. In the derivation, 93 % of 532 reviewed ER-membrane P450s carry a segment hydrophobic enough to span the bilayer, against 0.7 % of 145 bacterial P450s; bacterial systems use soluble redox partners instead.</p> },
      ]} />

      <Scene n={2} title="Distance sets the rate" chart={DistanceRate} steps={[
        { tags: ["S"], body: <p>Tunnelling rates in proteins fall about tenfold per 1.7 Å. The Moser–Dutton ruler, log₁₀k = 13 − 0.6(R − 3.6), gives the rate at optimal driving force. It is the usual first check of a proposed electron path.</p> },
        { tags: ["M", "L"], title: "The paper's own curve", body: <p>The paper&apos;s Marcus expression (λ 0.85 eV, ΔG −0.1 eV, coupling 1 meV, β 1.1 Å⁻¹) gives 610 s⁻¹ at 14 Å. The FMN → heme rate it cites is 5×10⁷ s⁻¹, five decades higher. The text nevertheless lists the cited rate as the prediction.</p> },
        { tags: ["S"], title: "Why the distance matters", body: <p>On the ruler, 14 Å allows up to 6×10⁶ s⁻¹ and 18 Å up to 2×10⁴ s⁻¹. Which distance holds in the complex decides whether this hop can be fast.</p> },
      ]} />

      <Scene n={3} title="Three hops in series" chart={HopRates} steps={[
        { tags: ["L"], body: <p>The paper uses three hop rates: 600 s⁻¹ for NADPH → FAD (Murataliev 1999), 10⁷ s⁻¹ for FAD → FMN (no citation), and 5×10⁷ s⁻¹ for FMN → heme (Denisov 2007). These are literature inputs, not results.</p> },
        { tags: ["M"], title: "The slowest step", body: <p>For steps in series the overall rate is 1/Σ(1/k): 600 s⁻¹, set almost entirely by hydride transfer from NADPH. The text names the FMN → heme hop as rate-limiting; its own numbers point to the first hop.</p> },
        { tags: ["M"], title: "About the animation below", body: <p>The 3-D animation on this site uses 6, 4 and 2 ×10¹² s⁻¹ so that the electron crosses in under a picosecond. It shows the path, not the kinetics.</p> },
      ]} />

      <Interlude
        title="The path on the structure"
        note="The site's P450 model with the four cofactor centres. The white centroid follows the occupancies of the shader pipeline; press play or scrub the slider. The time axis is in femtoseconds because the display rates are ten decades faster than the measured ones."
      >
        <ElectronTransferViewer height={520} autoplay />
      </Interlude>

      <Scene n={4} title="Efficiency and the number of steps" chart={EfficiencyChart} steps={[
        { tags: ["M"], body: <p>The paper&apos;s efficiency rule gives one decade per sequential step. With four steps it gives 10⁶ M⁻¹s⁻¹.</p> },
        { tags: ["L"], title: "Four CYP3A4 substrates", body: <p>The values the paper cites for testosterone, midazolam, erythromycin and nifedipine lie between 7×10⁵ and 4×10⁶ M⁻¹s⁻¹.</p> },
        { tags: ["M"], title: "What agreement in the mean shows", body: <p>The rule gives one number for every substrate, while the measured values spread over 0.76 decades. k_cat/K_M also contains binding and the chemical step, which differ between substrates. So agreement in the mean does not isolate electron delivery.</p> },
      ]} />

      <Scene n={5} title="The flavin semiquinone" chart={SemiquinoneChart} steps={[
        { tags: ["M"], body: <p>FAD and FMN pass electrons one at a time through a semiquinone. The model&apos;s free energies are 0 for oxidised, +2.8 for semiquinone and +7.4 kcal/mol for hydroquinone.</p> },
        { tags: ["M", "L"], title: "The sign", body: <p>The paper quotes −3 kcal/mol for semiquinone formation, without a citation. The model has the opposite sign: it places the one-electron intermediate above the oxidised state, which would make the stepwise path unfavourable.</p> },
      ]} />

      <Scene n={6} title="What the shader draws" chart={OccupancyChart} scrub steps={[
        { tags: ["M"], body: <p>The paper&apos;s shader pipeline turns three assumed hop rates into an occupancy for each cofactor over 800 fs.</p> },
        { tags: ["M"], title: "Density along the chain", body: <p>Each cofactor is drawn as a Gaussian weighted by its occupancy; their sum is the density along the chain. Scroll to move the cursor through time.</p> },
        { tags: ["M"], title: "What it does not measure", body: <p>The pipeline also reports a reorganisation energy of 0.85 eV. The code computes λ = (0.85/σ)·σ, so the value is set rather than read out. See the apparatus page for the rest of the stack.</p> },
      ]} />

      <Interlude title="What the paper proposes to test">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-1">
          {[
            ["Engineered chains", "Adding a cofactor to the path should lower log₁₀(k_cat/K_M) by one, a slope of −1 per step."],
            ["Semiquinone mutants", "Destabilising the flavin semiquinone should cut k_cat/K_M by orders of magnitude, not just slow one hop."],
            ["Single-molecule statistics", "The rate-limiting hop should show a Fano factor well above 1, where a single-rate Marcus step gives about 1."],
            ["Isotope tracking", "A labelled NADPH should deliver charge to the heme without its own electron arriving there; this one is specific to the paper's framework."],
          ].map(([t, b]) => (
            <div key={t} className="rounded-xl border p-5" style={{ borderColor: C.line, background: C.surface }}>
              <div className="mb-2 flex items-center gap-3"><span className="text-[17px] font-semibold" style={{ color: C.ink }}>{t}</span><Tag k="M">prediction</Tag></div>
              <p className="text-[15px] leading-relaxed" style={{ color: C.ink2 }}>{b}</p>
            </div>
          ))}
        </div>
      </Interlude>
    </StoryPage>
  );
}
