import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";
import { CHAIN_MARKERS } from "@/data/glbMarkers";

export default function Foundations() {
  const paper = PAPER_BY_SLUG["foundations"];
  return (
    <PaperPage paper={paper} glb={{ badge: "R_bio · receiver", preset: "full", markers: CHAIN_MARKERS }}>
      <h2>What this paper does</h2>
      <p>
        Establishes the categorical mechanics machinery used by every
        subsequent paper in the monograph. Defines the
        S-expression{" "}
        <code>ξ<sub>protein</sub></code> for biomolecules and the receiver{" "}
        <code>R<sub>bio</sub></code> as a sextuple
        <code> (𝓢, 𝓛, β<sub>floor</sub>, ε<sub>floor</sub>, T, Σ)</code>,
        with morphism chain
        <code> eval<sub>R<sub>bio</sub></sub> = access ∘ fuse ∘ catalyze ∘ observe</code>.
      </p>

      <h2>Closed-form conversion functors</h2>
      <p>
        Three conversion functors mediate between oscillatory, categorical,
        and partition representations:
      </p>
      <ul>
        <li><code>F<sub>OC</sub></code> – oscillatory ↔ categorical</li>
        <li><code>F<sub>CB</sub></code> – categorical ↔ partition (closed-form
          via the triple-isomorphism architecture)</li>
        <li><code>F<sub>BO</sub></code> – partition ↔ oscillatory</li>
      </ul>

      <h2>Resolutions</h2>
      <p>
        The paper resolves the two open questions left by the framework
        sources: the τ-assignment ambiguity (which folds into the receiver
        specification) and the Δs=0 / spin-crossover question (resolved
        via the two-tier chirality coordinate s = (s<sub>orbital</sub>,
        s<sub>state</sub>)).
      </p>
    </PaperPage>
  );
}
