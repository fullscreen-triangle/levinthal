import PaperPage from "@/components/PaperPage";
import dynamic from "next/dynamic";
import { PAPER_BY_SLUG } from "@/data/papers";

const ElectronTransferViewer = dynamic(
  () => import("@/components/ElectronTransferViewer"),
  { ssr: false, loading: () => (
      <div className="w-full rounded-xl border border-dark/10
        dark:border-light/10 bg-dark text-light/50 flex items-center
        justify-center text-xs uppercase tracking-widest"
        style={{ height: 700 }}>
        Loading 3D structure…
      </div>
    ),
  }
);

export default function GLBInput() {
  const paper = PAPER_BY_SLUG["glb-input"];
  return (
    <PaperPage paper={paper}>

      <h2>The productive GLB, live</h2>
      <p>
        The very GLB the parser reads. Heme-Fe at the rust marker is the
        coordinate read directly from the scene graph by{" "}
        <code>find_iron()</code>; the chain extension up to NADPH is
        anchored to that Fe. The white sphere is the electron centroid
        animating across the chain at Paper 4&apos;s hop-rate kinetics.
      </p>

      <div className="not-prose my-6">
        <ElectronTransferViewer height={700} autoplay />
      </div>

      <h2>Why this paper exists</h2>
      <p>
        Papers 1–5 are validated against synthetic CYP3A4-statistical
        sequences and target observables drawn from the literature. This
        deferral is honest — the synthetic suite tests the mathematical
        machinery — but it leaves a gap: when the framework predicts a
        contact map matching PDB 1TQN, the matching is asserted, not
        measured against actual coordinates. This methods paper closes
        that gap by bridging GLB-distributed PDB structures to
        R<sub>bio</sub>.
      </p>

      <h2>What the parser recovers</h2>
      <p>
        For atomistic ball-and-stick GLBs (CPK-style space-filling), the
        parser walks the scene graph, extracts per-atom 3D positions from
        node transforms, identifies elements via PBR baseColorFactor → CPK
        lookup, and infers bonds from element-aware vdW thresholds. On
        the productive cytochrome P450 oxy-complex GLB:
      </p>
      <ul>
        <li>146 atoms after artifact filtering (171 raw)</li>
        <li>Composition: 80 C, 22 H, 13 O, 12 X (custom ligand), 12 N,
            4 P, 2 S, 1 Fe</li>
        <li>Fe coordination: 4 N (porphyrin) at 2.01–2.04 Å,
            S (Cys thiolate) at 2.228 Å, axial O at 1.814 Å</li>
      </ul>

      <h2>Five GLB roles</h2>
      <ol>
        <li><strong>Calibration references</strong> — ground-truth atomic
            geometry to verify framework-predicted distances.</li>
        <li><strong>Initial conditions</strong> — real Cα positions seed
            Kuramoto folding simulations.</li>
        <li><strong>Validation targets</strong> — real top-L contact
            precision/recall computed against actual coordinates.</li>
        <li><strong>Interactive probes</strong> — web-frontend integration
            for user-driven exploration (this site is one).</li>
        <li><strong>Trajectory waypoints</strong> — GLB anchors at specific
            catalytic states (the present GLB anchors state 4 of the
            seven-state cycle).</li>
      </ol>

      <h2>Headline use in Paper 4</h2>
      <p>
        The productive GLB's heme-Fe position{" "}
        <code>(17.26, 11.53, 24.76) Å</code> is used directly as the
        anchor for the cofactor cluster in the headline observation
        paper. The shader pipeline of Paper 4 reads atomic positions
        from this GLB; the Marcus λ recovered at the FMN→heme loop
        comes from a diffraction pattern computed on the real
        GLB-anchored geometry.
      </p>
    </PaperPage>
  );
}
