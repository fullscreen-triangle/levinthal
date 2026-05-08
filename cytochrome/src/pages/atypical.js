import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Atypical() {
  const paper = PAPER_BY_SLUG["atypical"];
  return (
    <PaperPage paper={paper}>
      <section>
        <h2>Beyond Canonical Hydroxylation</h2>
        <p>
          P450 Compound I engages substrates through five mechanistically distinct
          atypical pathways: desaturation (two-step HAT), arene epoxidation,
          NIH shift, nucleophilic O-atom transfer, and carbene insertion in
          engineered enzymes. Each is characterized by a single activation
          partition depth ΔM within the categorical mechanics framework.
        </p>
      </section>
      <section>
        <h2>Rate Hierarchy</h2>
        <p>
          NIH shift (ΔM = 0.18, k ≈ 8.4×10⁹ s⁻¹) is fastest due to its
          charge-delocalized hydride migration. Desaturation is slowest
          (k_eff ≈ 1.3×10⁸ s⁻¹) because rebound competition at the radical
          intermediate dramatically attenuates the effective rate. All five
          predicted rates fall within published experimental ranges.
        </p>
      </section>
      <section>
        <h2>Kinetic Isotope Effects</h2>
        <p>
          Only desaturation carries a primary KIE (≈ 4–6), because the first
          HAT step breaks a C–H bond. Epoxidation, NIH shift, nucleophilic
          O-atom transfer, and carbene insertion all have KIE ≈ 1 since no
          hydrogen is transferred in the rate-limiting step.
        </p>
      </section>
    </PaperPage>
  );
}
