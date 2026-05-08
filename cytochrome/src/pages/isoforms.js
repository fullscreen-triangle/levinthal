import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Isoforms() {
  const paper = PAPER_BY_SLUG["isoforms"];
  return (
    <PaperPage paper={paper}>
      <section>
        <h2>Ternary Taxonomy of 57 Human CYPs</h2>
        <p>
          The 57 human CYP isoforms distributed across 18 Nelson families emerge
          naturally from ternary (base-3) address encoding. Depth k=3 (3³=27)
          separates all 18 families with recall 0.94; depth k=6 (3⁶=729) achieves
          distinctness 0.97 for all 57 isoforms; depth k=9 resolves allelic
          variants catalogued in PharmVar.
        </p>
      </section>
      <section>
        <h2>CYP3A4 Fold Depth</h2>
        <p>
          CYP3A4 has 503 residues (UniProt P08684). The categorical encoding
          depth is log₃(503) ≈ 5.69 steps — consistent with structural resolution
          at RMSD &lt; 2.5 Å versus PDB 1TQN within 6 depth steps.
        </p>
      </section>
      <section>
        <h2>Substrate Selectivity Windows</h2>
        <p>
          Each CYP family carries a characteristic ΔM range. CYP3A4 has the
          widest window (ΔM ∈ [0.40, 0.70]) matching its broadest substrate
          scope. CYP3A4, CYP2D6, and CYP2C9 together metabolize ≥80% of
          FDA-approved drugs.
        </p>
      </section>
    </PaperPage>
  );
}
