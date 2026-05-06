import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Equilibrium() {
  const paper = PAPER_BY_SLUG["equilibrium"];
  return (
    <PaperPage paper={paper}>
      <h2>State 1 → State 2 as one categorical aperture</h2>
      <p>
        Water displacement, substrate insertion, and spin-crossover are
        not three separate events but three facets of one partition
        reorganisation. The transition has categorical distance
        <code> d<sub>C</sub> = 1</code> with explicit partition-depth
        change <code>ΔM = 0.918</code>.
      </p>

      <h2>The +120 mV redox shift</h2>
      <p>
        The redox shift gating CPR-mediated electron acceptance follows
        directly from ΔM, with no additional fitted parameters:
      </p>
      <pre className="text-xs">
        ΔE<sub>1/2</sub> = (k<sub>B</sub>T / e) · n<sub>eff</sub> · ΔM · ln b
      </pre>
      <p>
        Computed value: <strong>122 mV</strong>; measured value (Daff 1997):
        120 mV. Deviation: 1.7 %.
      </p>

      <h2>Heme-pocket capacitor</h2>
      <ul>
        <li>Capacitance C ≈ 5.7 × 10⁻²⁰ F (56.7 aF)</li>
        <li>Stored energy U ≈ 1.4 eV</li>
        <li>RC discharge time τ<sub>RC</sub> ≈ 60 ps</li>
      </ul>
      <p>
        These are the parameters that govern the femtosecond-to-picosecond
        kinetics of the electron-transfer events observed in Paper 4.
      </p>

      <h2>Imports</h2>
      <ul>
        <li>Closed-form functors F<sub>OC</sub>, F<sub>CB</sub>, F<sub>BO</sub>{" "}
            from the triple-isomorphism architecture</li>
        <li>Variance–free-energy identity{" "}
            F = k<sub>B</sub>T · σ²(φ)</li>
        <li>Electrostatic chamber confinement{" "}
            |eΔφ| / k<sub>B</sub>T ≈ 7</li>
      </ul>
    </PaperPage>
  );
}
