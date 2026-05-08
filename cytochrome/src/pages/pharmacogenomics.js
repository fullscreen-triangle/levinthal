import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Pharmacogenomics() {
  const paper = PAPER_BY_SLUG["pharmacogenomics"];
  return (
    <PaperPage paper={paper}>
      <section>
        <h2>Allele-Specific ΔM Parameters</h2>
        <p>
          CYP2D6 pharmacogenomic variation is encoded as shifts in the activation
          partition depth: UM (ΔM=0.27, gene duplication), EM (ΔM=0.55, wild-type),
          IM (ΔM=0.75, reduced function), PM (ΔM=2.50, null allele). This 9-fold
          rate span from PM to UM directly predicts clinical dose adjustments.
        </p>
      </section>
      <section>
        <h2>Warfarin and Codeine Predictions</h2>
        <p>
          CYP2C9*3 (I359L) raises ΔM to 3.60, reducing S-warfarin hydroxylation
          to &lt;5% of wild-type and predicting a ~23x dose reduction for *3/*3
          homozygotes. For codeine, CYP2D6 UM patients produce 32% excess morphine
          (k_UM/k_EM = e⁰·²⁸ ≈ 1.32), consistent with the FDA black-box warning.
        </p>
      </section>
      <section>
        <h2>Drug-Drug Interactions</h2>
        <p>
          Competitive inhibition is modelled as α = 1 + [I]/Kᵢ, shifting the
          apparent ΔM by ln(α). Fluoxetine (Kᵢ ≈ 0.24 μM) at clinical
          concentrations (0.5 μM) gives α ≈ 3.1, predicting a strong DDI.
          CYP3A4 induction by rifampicin (20×) reduces victim drug AUC to ~5%
          of baseline.
        </p>
      </section>
    </PaperPage>
  );
}
