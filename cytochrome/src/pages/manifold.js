import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";
import { CHAIN_MARKERS } from "@/data/glbMarkers";

export default function Manifold() {
  const paper = PAPER_BY_SLUG["manifold"];
  return (
    <PaperPage paper={paper} glb={{ badge: "address manifold · CYP3A4", preset: "full", markers: CHAIN_MARKERS }}>
      <h2>One evaluation, three depths</h2>
      <p>
        The four claims of this paper — family clustering at k=3, isoform
        separation at k=6, allelic variation at k=9, and the CYP3A4 native
        fold — are truncations of the same evaluation
        <code> eval<sub>R</sub>(ξ<sub>P450</sub><sup>family</sup>)</code>
        at different recursion depths. The taxonomy is not learned from
        data; it is a direct consequence of partition-coordinate density
        in the address manifold.
      </p>

      <h2>CYP3A4 fold from sequence</h2>
      <p>
        The same evaluation, run at full residue-and-structural depth on
        UniProt P08684 (503 residues), folds the protein against the
        unliganded crystal structure PDB&nbsp;1TQN. Folding completes in
        O(log<sub>3</sub>&nbsp;N) ≈ 6 categorical steps with order
        parameter r → 0.87.
      </p>

      <h2>Comparisons</h2>
      <p>
        Compared against AlphaFold2 (deep learning), MD simulation
        (force-field integration), and BLAST/HMMER classification (sequence
        statistics). The categorical address-manifold approach matches all
        three on their own metrics while requiring no training data.
      </p>
    </PaperPage>
  );
}
