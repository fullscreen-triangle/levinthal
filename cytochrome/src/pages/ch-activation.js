import dynamic from "next/dynamic";
import StoryPage, { Scene, Interlude } from "@/components/scrolly/Story";
import { HatGeometry, ClassDepths, KieChart, ReboundChart, TestosteroneChart, ScoreChart } from "@/components/scrolly/charts/ch";
import { HEME_MARKER } from "@/data/glbMarkers";

const GLBViewer = dynamic(() => import("@/components/GLBViewer"), { ssr: false });

const LEDGER = [
  ["C–H and O–H at the transition state", "1.30 Å, 1.20 Å", "QM/MM, Shaik et al. 2010, as cited", "L"],
  ["activation depth per substrate class", "0.35–0.65", "set per class", "M"],
  ["rate per class, k = 10¹⁰·e^(−ΔM)", "5.2–7.0 ×10⁹ s⁻¹", "—", "M"],
  ["isotope effect, aliphatic C–H, 310 K", "9.63 (script), 7.2 (text)", "4–11 for P450s, as cited", "M"],
  ["k_rebound ÷ k_HAT", "1.42", "follows from the two set ΔM values", "M"],
  ["stereoretention at k_esc 10⁸–10⁹ s⁻¹", "88–99 %", "40–90 %, as cited", "M"],
  ["testosterone 6β fraction, CYP3A4", "48.8 % (27.9 % without g)", "50–70 %, Guengerich 1998, Rendic 2002", "M"],
  ["primary isotope effect only when H moves", "aliphatic, benzylic, allylic", "established mechanism", "L"],
];

export default function CHActivation() {
  return (
    <StoryPage
      meta={{ title: "C–H Activation", description: "Hydrogen-atom transfer and oxygen rebound by P450 Compound I: the mechanism, the paper's model values, and how they compare with measurements." }}
      kicker="Part III · Paper 6 · Catalytic chemistry"
      title="C–H activation: abstraction, then rebound"
      lead={<>
        <p>
          Compound I takes a hydrogen atom from an unactivated C–H bond and returns a hydroxyl to the carbon radical it
          leaves behind. This is the hydrogen-abstraction and oxygen-rebound mechanism of Groves.
        </p>
        <p>
          The paper treats the abstraction as one event involving three bodies: carbon, hydrogen and the ferryl oxygen.
          This page goes through its numbers: the isotope effect, the competition between rebound and escape, and the
          selectivity on testosterone. Each is compared with the measurement it is checked against.
        </p>
      </>}
      ledger={LEDGER}
      sources={<>Model values: <code>publications/catalytic-cycle/ch-activation-rebound</code> (tex, <code>validation/scripts</code>, <code>figures/generate_panels.py</code>). Where the text and the code differ, the page shows what the code returns.</>}
      prev={{ href: "/compound-i", label: "Compound I formation" }}
      next={{ href: "/heteroatom", label: "Heteroatom oxidation and dealkylation" }}
    >
      <Scene n={1} title="Three bodies: carbon, hydrogen, oxygen" chart={HatGeometry} scrub steps={[
        { tags: ["L"], body: <p>The substrate C–H bond points at the oxygen of Compound I. Scroll to move along the reaction coordinate.</p> },
        { tags: ["L"], title: "The transition state", body: <p>The hydrogen is shared: C–H stretched to about 1.30 Å, O–H forming at about 1.20 Å. These lengths come from the QM/MM studies the paper cites (Shaik et al. 2010). The paper requires the C–H bond to break and the O–H bond to form in one event, which it calls a single aperture.</p> },
        { tags: ["L"], title: "Radical and Fe–OH", body: <p>After the transfer, the carbon is a radical and the iron carries a hydroxyl. What happens next decides whether the configuration at that carbon is kept.</p> },
      ]} />

      <Scene n={2} title="Five substrate classes on one scale" chart={ClassDepths} steps={[
        { tags: ["M"], body: <p>The paper gives each class one number, an activation depth ΔM: aliphatic 0.65, benzylic 0.50, allylic 0.45, aromatic 0.38, double-bond epoxidation 0.35. The values are assigned per class and not computed from bond energies. The activation energy is 65 kJ/mol × ΔM.</p> },
        { tags: ["M"], title: "As rates", body: <p>Through k = 10¹⁰·e^(−ΔM) the five classes span only a factor of 1.35. The depth scale orders the classes but compresses their rates into a narrow band.</p> },
        { tags: ["L"], title: "What does separate them", body: <p>Only where the hydrogen moves in the rate-limiting step is there a primary isotope effect. That is aliphatic, benzylic and allylic C–H, not aromatic hydroxylation or epoxidation. This is the established mechanistic picture, and the paper adopts it.</p> },
      ]} />

      <Scene n={3} title="The isotope effect" chart={KieChart} steps={[
        { tags: ["M"], body: <p>Replacing H by D lowers the C–H stretch from 3000 to 3000/√2 cm⁻¹. The zero-point energy difference, 1.26 kcal/mol, alone gives k_H/k_D = 7.7 at 310 K.</p> },
        { tags: ["M", "L"], title: "With tunnelling", body: <p>The script multiplies by a tunnelling factor of 1.253 and returns 9.63. That lies inside 4–11, the range the paper cites for P450 hydroxylations.</p> },
        { tags: ["M"], title: "Text and code differ", body: <p>The paper&apos;s text, and the earlier version of this page, give 7.2 using a factor of 1.16; that factor gives 8.91 instead. Reaching 7.2 would need a factor below one. The temperature dependence is the testable part: the model&apos;s isotope effect falls by about a third between 290 and 350 K.</p> },
      ]} />

      <Interlude
        title="The active site"
        note="The heme with its iron marked. Substrate binds on the open face above the iron, where Compound I's oxygen sits. Drag to rotate."
      >
        <GLBViewer height={420} badge="heme · Fe" preset="heme" markers={HEME_MARKER} />
      </Interlude>

      <Scene n={4} title="Rebound against escape" chart={ReboundChart} steps={[
        { tags: ["M"], body: <p>The carbon radical either recombines with the Fe–OH (rebound) or moves before it does. The fraction retaining its configuration is k_rebound / (k_rebound + k_esc). With the paper&apos;s two depths, rebound is 1.42 times faster than abstraction. That ratio follows from the two numbers chosen, 0.30 and 0.65.</p> },
        { tags: ["M", "L"], title: "With the proposed escape rates", body: <p>The text proposes escape rates of 10⁸–10⁹ s⁻¹. These give 88–99 % retention, not the 40–90 % range of measured stereoretention the paper sets out to explain.</p> },
        { tags: ["M"], title: "Reaching the low end", body: <p>Retention of 40 % needs an escape rate of 1.1 ×10¹⁰ s⁻¹. The script sets that value for one substrate class, so the low end of the range is an input rather than a result.</p> },
      ]} />

      <Scene n={5} title="Testosterone: where the 6β fraction comes from" chart={TestosteroneChart} steps={[
        { tags: ["M"], body: <p>CYP3A4 hydroxylates testosterone mainly at 6β. The paper gives four positions a depth ΔM and an accessibility factor g, and predicts 48.8 % at 6β. Reported values are 50–70 % (Guengerich 1998; Rendic 2002).</p> },
        { tags: ["M"], title: "Without the accessibility factors", body: <p>With equal g the depths alone give 6β 27.9 %. The depths separate the positions only weakly.</p> },
        { tags: ["M", "L"], title: "The factor does the work", body: <p>The match comes from g₆β = 1.0 against 0.3–0.5 for the other positions. Reaching 50 % takes g₆β ≈ 1.05, and 70 % about 2.5. An independent estimate of g, for example from docking or crystal poses, would turn this into a test.</p> },
      ]} />

      <Scene n={6} title="Reading the validation table" chart={ScoreChart} steps={[
        { tags: ["M"], body: <p>The paper reports eight checks, all passed. With the script&apos;s own 30 % threshold the count is seven of eight; at 20 % it is six.</p> },
        { tags: ["M"], title: "Four checks compare the model with itself", body: <p>For ΔM, k_HAT, k_rebound and their ratio, the reference values are the model&apos;s own outputs or targets. Agreement there is not evidence.</p> },
        { tags: ["L"], title: "Four compare with literature", body: <p>Activation energy, isotope effect, stereoretention and the 6β fraction are compared with literature, using a value inside each reported range as the reference. Three predictions fall inside their ranges. The 6β fraction, 48.8 %, is just below 50 %, and it depends on the chosen accessibility factor. With ranges as wide as 4–11 or 40–90 %, falling inside is a weak test.</p> },
      ]} />
    </StoryPage>
  );
}
