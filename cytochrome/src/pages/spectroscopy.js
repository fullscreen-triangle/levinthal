import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Spectroscopy() {
  const paper = PAPER_BY_SLUG["spectroscopy"];
  return (
    <PaperPage paper={paper}>
      <section>
        <h2>Soret Band and Spin-State Diagnostics</h2>
        <p>
          All seven P450 catalytic states carry distinct Soret signatures.
          The resting Fe³⁺ low-spin state absorbs at 417 nm; substrate binding
          shifts this to 392 nm (25 nm blue shift) as the iron moves to high-spin.
          EPR directly confirms spin state: LS shows a rhombic S=1/2 spectrum
          at g = 2.42, 2.25, 1.92; HS shows an axial S=5/2 signal at g = 7.70,
          3.50, 1.80. UV-Vis alone cannot distinguish all seven states — states
          near 367–370 nm (Compound 0 and Compound I) require Raman confirmation.
        </p>
      </section>
      <section>
        <h2>Compound I: Resonance Raman Fe=O</h2>
        <p>
          Compound I (Fe⁴⁺=O porphyrin radical cation) carries a unique
          resonance Raman Fe=O stretch at 795 cm⁻¹. The ¹⁸O isotope shift
          to ~758 cm⁻¹ (Δν = 37 cm⁻¹) matches the reduced-mass prediction
          exactly: ν(¹⁸O) = 795 × √(μ₁₆/μ₁₈) ≈ 758 cm⁻¹. This isotope
          shift is the definitive spectroscopic fingerprint for Fe=O bond
          character and uniquely identifies Compound I among all seven states.
        </p>
      </section>
      <section>
        <h2>ΔM_spec Correlation with Soret Energy</h2>
        <p>
          The spectroscopic activation partition depth ΔM_spec = hcν̃/T_part
          correlates linearly with Soret photon energy across all seven states
          (Pearson r &gt; 0.9). This confirms that the categorical mechanics
          partition depth encodes real physical spectroscopic information —
          not merely kinetic barriers. The substrate-bound high-spin state has
          the highest Soret energy (392 nm = 25,510 cm⁻¹) and the highest
          ΔM_spec, consistent with its role as the activated entry point for
          the electron transfer cascade.
        </p>
      </section>
    </PaperPage>
  );
}
