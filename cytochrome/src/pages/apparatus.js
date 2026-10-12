import dynamic from "next/dynamic";
import StoryPage, { Scene, Interlude } from "@/components/scrolly/Story";
import { StackChart, HydrogenChart, CrosstalkChart, CycleRankChart, ChiChart, LambdaChart } from "@/components/scrolly/charts/apparatus";
import { CHAIN_MARKERS } from "@/data/glbMarkers";

const GLBViewer = dynamic(() => import("@/components/GLBViewer"), { ssr: false });

const LEDGER = [
  ["Layer 1 oscillator frequencies", "10⁴, 10⁸, 10⁹, 10¹⁴ Hz", "set; the source paper gives other values", "M"],
  ["admissibility floor (Layer 2)", "3.7×10⁻⁴", "set; 3.43×10⁻⁴ elsewhere in the paper", "M"],
  ["hydrogen lines against NIST", "26 lines, mean 8.7×10⁻⁵", "text: 24 lines, 1.6×10⁻⁴", "C"],
  ["21 cm hyperfine line", "1420.40575 MHz, 1.3×10⁻⁹", "NIST", "S"],
  ["strobe cross-talk η", "3.7×10⁻³", "set, from η = 0.37·Δt/τ_em", "M"],
  ["cycle rank for the reductase chain", "≥ 6", "set; the graph is not built", "M"],
  ["χ² of the four cofactors", "19–59", "synthetic counts", "M"],
  ["reorganisation energy λ", "0.85 eV", "set in the code", "M"],
  ["Fe position and atom count", "(17.26, 11.53, 24.76) Å, 146 atoms", "read from the GLB file", "C"],
];

export default function Apparatus() {
  return (
    <StoryPage
      meta={{ title: "Apparatus", description: "The five-layer stack of the transfer paper, layer by layer: which numbers are measured, computed from data, computed from assumed inputs, or set in the code." }}
      kicker="Part III · The instrument stack of Paper 4"
      title="The five-layer stack, number by number"
      lead={<>
        <p>
          The transfer paper presents its numbers as readings from a five-layer instrument, with hardware oscillators at
          the bottom and a GPU shader pipeline at the top. It calls this an apparatus rather than a simulation.
        </p>
        <p>
          This page goes through the layers. For each number it asks whether it is measured, computed from data,
          computed from assumed inputs, or set in the code. Two numbers come from data: the iron position and the atom
          count, read from the GLB structure. The rest are computed or set, so the stack is better read as the
          paper&apos;s computational pipeline.
        </p>
      </>}
      ledger={LEDGER}
      sources={<>Code: <code>publications/catalytic-cycle/multi-hop-et-chain/validation/scripts/09–12</code>, <code>glb/levinthal_glb/shader_pipeline.py</code>. Source papers: <code>sources/hyperfine-transitions.tex</code>, <code>ensemble-strobes.tex</code>, <code>harmonic-molecular-resonator.tex</code>, <code>superimposed-multi-modal-holograms.tex</code>. The hydrogen differences were recomputed here from the tables in the first.</>}
      prev={{ href: "/transfer", label: "Electron transfer through the reductase" }}
      next={{ href: "/glb-input", label: "GLB structural input" }}
    >
      <Scene n={1} title="Five layers and their time scales" chart={StackChart} steps={[
        { tags: ["M"], body: <p>The stack as the paper defines it. Layer 1 reads hardware oscillators; Layer 2 is the argument for reading them as atomic coordinates; Layer 3 gates counts in time; Layer 4 counts loops of commensurate modes; Layer 5 is a GPU shader pipeline that draws densities and reports six observables.</p> },
        { tags: ["M"], title: "Layer 1", body: <p>Four hardware oscillators are read as the quantum numbers n, ℓ, m and s. The validation script uses round frequencies; the source paper gives 3 GHz, 800 MHz, 4.6×10¹⁴ Hz and 64 kHz. No reading of these oscillators is recorded.</p> },
        { tags: ["M"], title: "Layer 3", body: <p>Three gates, from femtoseconds to microseconds, are defined by the analyte&apos;s own absorption and emission times. In the validation they are constants.</p> },
      ]} />

      <Scene n={2} title="Calibration on hydrogen" chart={HydrogenChart} steps={[
        { tags: ["S", "C"], body: <p>Layer 1 is calibrated on hydrogen. From the tables in the source paper we recomputed each line&apos;s fractional difference from NIST: 26 lines, from exact agreement up to 5×10⁻⁴.</p> },
        { tags: ["C"], title: "Summary against table", body: <p>The paper reports 1.6×10⁻⁴ over 24 lines. Its own table has 26 lines with a mean of 8.7×10⁻⁵; the largest difference, 5.3×10⁻⁴, is the ionisation energy.</p> },
        { tags: ["S"], title: "What the agreement shows", body: <p>The paper&apos;s computed column coincides with the Rydberg formula. The agreement shows that standard atomic physics is reproduced; it does not show that the hardware oscillators measure anything.</p> },
      ]} />

      <Scene n={3} title="Strobe cross-talk" chart={CrosstalkChart} steps={[
        { tags: ["M"], body: <p>The fraction of one gate&apos;s signal that leaks into the next: η = 0.37·Δt/τ_em, up to 0.37.</p> },
        { tags: ["M"], title: "The operating point", body: <p>For the reductase the paper takes an emission time of about 1 ns and gates of about 10 ps, which gives η = 3.7×10⁻³.</p> },
        { tags: ["M"], title: "What is checked", body: <p>The validation compares that number, entered as a constant, with a threshold of 10⁻². No counts are gated.</p> },
      ]} />

      <Scene n={4} title="Loops of commensurate modes" chart={CycleRankChart} steps={[
        { tags: ["M"], body: <p>Layer 4 builds a graph whose edges join vibrational modes with commensurate frequencies. Its cycle rank counts the independent loops, which the paper treats as independent checks.</p> },
        { tags: ["C"], title: "Recounting", body: <p>The paper&apos;s formula C = |E| − |V| + 1, applied to its own table, gives 3 for H₂O and CO₂, where the table lists 1.</p> },
        { tags: ["M"], title: "For the reductase", body: <p>For the reductase chain the graph is never built. The required rank, C ≥ 6, is a constant in the script.</p> },
      ]} />

      <Interlude
        title="What is read from data"
        note="The site's P450 model with the four cofactor centres marked. The iron position, (17.26, 11.53, 24.76) Å, and the atom count, 146, are read from this file. They are the only numbers in the stack taken from data. Drag to rotate."
      >
        <GLBViewer height={420} badge="GLB · cofactor centres" preset="chain" markers={CHAIN_MARKERS} />
      </Interlude>

      <Scene n={5} title="Which atoms stand out" chart={ChiChart} steps={[
        { tags: ["M"], body: <p>In the selection step each atom gets a three-state count histogram. A χ² test against the equilibrium distribution flags atoms whose counts deviate. Of 196 bystander atoms, 11 cross the threshold by chance.</p> },
        { tags: ["M"], title: "The four cofactors", body: <p>NADPH C4, FAD N5, FMN N5 and the heme iron are flagged, with χ² between 19 and 59.</p> },
        { tags: ["M"], title: "Where the counts come from", body: <p>The counts are synthetic. The script draws the four cofactors from a shifted distribution it is given and the bystanders from the equilibrium one. This shows that the χ² test works on such data. It does not show that the cofactors select themselves; that would need measured counts from the protein.</p> },
      ]} />

      <Scene n={6} title="The reorganisation energy" chart={LambdaChart} steps={[
        { tags: ["M"], body: <p>Script 11 builds a width from a chosen λ, adds 5 % noise and inverts it: 0.85 eV in, 0.79 eV out.</p> },
        { tags: ["M"], title: "The shader's readout", body: <p>In the shader pipeline the readout is λ = (0.85/σ)·σ, which is 0.85 eV for any width.</p> },
        { tags: ["M", "L"], title: "Where 0.85 eV comes from", body: <p>The value lies in the quoted 0.7–1.0 eV range because it was chosen from that range. A reorganisation energy for the FMN → heme step has to come from measurement, for example from how the rate depends on temperature or driving force.</p> },
      ]} />
    </StoryPage>
  );
}
