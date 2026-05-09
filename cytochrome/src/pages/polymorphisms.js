import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Polymorphisms() {
  const paper = PAPER_BY_SLUG["polymorphisms"];
  return (
    <PaperPage paper={paper}>
      <section>
        <h2>Allele ΔM Shifts: Genotype to Rate</h2>
        <p>
          CYP2D6 phenotypes span a 7-fold rate range encoded as ΔM shifts:
          UM (ΔM=0.27, gene duplication), EM (ΔM=0.55, wild-type), IM
          (ΔM=0.75, reduced function *10/*17), and PM (ΔM=2.50, null *4/*5).
          CYP2C9*3 (I359L, ΔM=3.60) reduces S-warfarin hydroxylation to
          &lt;5% of wild-type, while CYP3A4*22 reduces expression ~50%
          (ΔΔM = ln 2 ≈ 0.69). All allele effects are additive in ΔM.
        </p>
      </section>
      <section>
        <h2>Inhibition: Competitive and Mechanism-Based</h2>
        <p>
          Competitive inhibitors impose ΔΔM = ln(α) where α = 1 + [I]/Ki.
          Ketoconazole (Ki=0.037 μM) at 0.2 μM gives α=6.4, reducing CYP3A4
          to &lt;20% activity. Quinidine (Ki=0.027 μM) at 0.5 μM drives
          CYP2D6-EM below the natural PM rate — a pharmacodynamic phenocopy.
          Mechanism-based inactivators follow Kitz-Wilson kinetics;
          clarithromycin (kinact/KI = 0.011 min⁻¹ μM⁻¹) is the most potent
          of the three macrolide/non-macrolide inactivators modelled.
        </p>
      </section>
      <section>
        <h2>Induction and Compound Phenotype Prediction</h2>
        <p>
          CYP3A4 induction by rifampicin (20×) reduces victim drug AUC to 5%
          of baseline (ΔΔM = −ln 20 = −3.0). The compound phenotype formula
          ΔM_eff = ΔM_allele + ΣΔΔMj predicts a single patient's effective
          rate from genotype plus co-medication list. A CYP2C9*3/*3 patient
          receiving fluconazole (ΔΔM ≈ 1.17) has essentially abolished
          S-warfarin metabolism, directly from the additive ΔM model.
        </p>
      </section>
    </PaperPage>
  );
}
