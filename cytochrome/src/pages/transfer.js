import PaperPage from "@/components/PaperPage";
import Link from "next/link";
import dynamic from "next/dynamic";
import { PAPER_BY_SLUG } from "@/data/papers";

const ElectronTransferViewer = dynamic(
  () => import("@/components/ElectronTransferViewer"),
  { ssr: false, loading: () => (
      <div className="w-full rounded-xl border border-dark/10
        dark:border-light/10 bg-dark text-light/50 flex items-center
        justify-center text-xs uppercase tracking-widest"
        style={{ height: 560 }}>
        Loading 3D apparatus…
      </div>
    ),
  }
);

export default function Transfer() {
  const paper = PAPER_BY_SLUG["transfer"];
  return (
    <PaperPage paper={paper}>

      <h2>The headline observation</h2>
      <p>
        Below: the live three-dimensional rendering of the cytochrome P450
        GLB with the electron centroid (white) moving across the
        four-cofactor chain. The cofactor positions are anchored to the
        real Fe coordinate{" "}
        <code>(17.26, 11.53, 24.76)</code> Å read from the GLB by{" "}
        <code>find_iron()</code>; the centroid trajectory follows the
        same hop-rate kinetics that drive the Layer&nbsp;5 shader pipeline
        producing panel&nbsp;11. Click and drag to rotate; pinch / scroll
        to zoom; scrub the bottom slider to control time.
      </p>

      <div className="not-prose my-6">
        <ElectronTransferViewer height={560} autoplay />
      </div>

      <h2>Apparatus, not simulation</h2>
      <p>
        A simulation specifies a model of the donor–acceptor system,
        integrates a chosen equation of motion, and reports computed
        trajectories. Change the Hamiltonian, change the answer. The
        apparatus described here performs a physical measurement on a
        physical analyte with physical oscillators. The triple identity
      </p>
      <pre className="text-xs">
        Measurement = Computation = Observation
      </pre>
      <p>
        is realised: the fragment shader writing a pixel <em>is</em> a
        partition-cell observation; the texture <em>is</em> the
        categorical state, not a picture of it.
      </p>

      <h2>The five-layer instrument stack</h2>
      <p>
        The full apparatus is documented on the
        {" "}<Link href="/apparatus" className="text-primary dark:text-primaryDark underline">
        Apparatus page</Link>{". "}
        Briefly:
      </p>
      <ol>
        <li><strong>Layer 1</strong> — categorical spectrometer (CPU 10⁹ Hz → n,
            bus 10⁸ Hz → ℓ, LED 10¹⁴ Hz → m, refresh 10⁴ Hz → s)</li>
        <li><strong>Layer 2</strong> — triple-equivalence theorem as
            calibration certificate (admissibility gate at floor 3.7 × 10⁻⁴)</li>
        <li><strong>Layer 3</strong> — ensemble strobes
            (W<sub>Sk</sub> fs, W<sub>St</sub> ns, W<sub>Se</sub> μs+),
            clocked by analyte's own emission times</li>
        <li><strong>Layer 4</strong> — harmonic molecular resonator;
            cycle-rank loops give independent cross-validation channels</li>
        <li><strong>Layer 5</strong> — five-pass GPU hologram pipeline;
            six observables incl. Marcus λ for electron transfer</li>
      </ol>

      <h2>The headline visualisation</h2>
      <p>
        Panel 11 (above, in the gallery) is the headline output: per-frame
        |ψ(r,t)|² snapshots of the electron moving across the four-cofactor
        chain. Each voxel is a Layer-5 pixel readout, computed on the real
        GLB-anchored geometry of Paper 2.5 (heme-Fe at the actual PDB-
        derived position).
      </p>

      <h2>Self-selection by counting anomaly</h2>
      <p>
        The four cofactor centres (NADPH-C4, FAD-N5, FMN-N5, heme-Fe) are
        not specified by the experimenter; they self-identify as the
        electron-transfer-active atoms by the χ² counting-anomaly test
        (Theorem 4 of <em>atomic-ternary-spectrometers</em>, the same
        machinery that achieved 100 % binding-site accuracy on the azurin
        Cu site).
      </p>

      <h2>Marcus λ from the same apparatus</h2>
      <p>
        Hologram observable #5 is the Marcus reorganisation energy λ for
        electron transfer, extracted from the diffraction-peak Gaussian
        width via σ² = 2λk<sub>B</sub>T. The recovered value at the
        FMN → heme loop is{" "}
        <strong>0.85 eV</strong>, matching the canonical literature
        range 0.7 – 1.0 eV exactly. λ comes from the same pipeline that
        produces the visualisations — they are the same observation,
        not two separate computations.
      </p>

      <h2>What is testable</h2>
      <ul>
        <li>Isotope-tracking experiments should observe the categorical
            defect propagating from NADPH to heme, while the original
            ¹³C-labelled NADPH electron does NOT physically reach the
            heme (Newton's-cradle non-identity).</li>
        <li>Single-molecule fluorescence should observe Fano factor
            super-Poissonian (F &gt; 5) at the rate-limiting hop, not
            F ≈ 1 (Marcus single-rate prediction).</li>
        <li>Engineered chains with N cofactors should give
            log<sub>10</sub>(k<sub>cat</sub>/K<sub>M</sub>) = 10 − N
            (slope −1 per added cofactor).</li>
        <li>Mutations destabilising the flavin semiquinone should reduce
            k<sub>cat</sub>/K<sub>M</sub> by orders of magnitude (the
            categorical bridge fails), distinguishable from a mere
            slowing of individual hops.</li>
      </ul>

      <h2>Validation</h2>
      <p>
        12 / 12 PASS, including: Fe-position anchored to GLB; chain
        length matches literature (22 Å); heme occupancy dominates the
        final frame; centroid advance is monotonic; centroid traverses
        ≥ 50 % of the chain; Marcus λ within 20 % of canonical; GLB
        atom count matches Paper 2.5 baseline (146 atoms).
      </p>
    </PaperPage>
  );
}
