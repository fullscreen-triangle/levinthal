import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";
import { HEME_MARKER } from "@/data/glbMarkers";

export default function CHActivation() {
  const paper = PAPER_BY_SLUG["ch-activation"];
  return (
    <PaperPage paper={paper} glb={{ badge: "C–H activation site", preset: "heme", markers: HEME_MARKER }}>

      <h2>The C–H Activation Problem</h2>
      <p>
        The activation of unactivated C–H bonds is the defining chemical event
        of cytochrome P450 catalysis. Every hydroxylation, epoxidation, and
        oxygenation reaction ultimately rests on the ability of Compound&nbsp;I
        (Fe⁴⁺=O porphyrin&nbsp;π-radical cation) to abstract a hydrogen atom
        from a substrate C–H bond. Three questions have resisted unified
        treatment: why KIE values range from 1 to 13, why stereospecificity
        is retained at 40–90&thinsp;% but not 100&thinsp;%, and what drives
        regiochemical preference for specific C–H bonds.
      </p>

      <h2>Three-Body Trajectory</h2>
      <p>
        We treat H-atom transfer (HAT) as a{" "}
        <em>three-body categorical trajectory</em>: the substrate carbon C
        (donor), the abstracted hydrogen H, and the ferryl oxygen Fe=O
        (acceptor) all participate in a single categorical aperture
        (<code>d_C = 1</code>). The three-body aperture selection rule requires
        simultaneous Δβ<sub>CH</sub> = 1, Δβ<sub>OH</sub> = −1, and
        Δs<sub>orbital</sub> = 0; any deviation promotes the event to{" "}
        <code>d_C ≥ 2</code> and slows it by a factor of&nbsp;10.
      </p>

      <h2>Kinetic Isotope Effect</h2>
      <p>
        The KIE is predicted at <strong>7.2</strong> for aliphatic C–H
        substrates, combining a classical ZPE contribution of ≈&thinsp;6.2
        (from the C–H vs C–D stretch frequency difference at 3000&thinsp;cm⁻¹)
        with a tunneling correction ratio κ<sub>H</sub>/κ<sub>D</sub>&thinsp;≈&thinsp;1.16.
        The prediction falls within the measured range of 4–11 for diverse
        P450 substrates. The framework also predicts that the KIE
        <em>decreases</em> with temperature — distinguishable from QM/MM
        tunneling predictions by variable-temperature measurements.
      </p>

      <h2>Oxygen Rebound</h2>
      <p>
        After HAT the substrate radical R&bull; faces the Fe(III)–OH complex.
        The C–O bond forms in a second <code>d_C = 1</code> aperture with
        activation depth Δ<em>M</em><sub>rebound</sub> = 0.30 &lt; Δ<em>M</em><sub>HAT</sub> = 0.65,
        making rebound intrinsically faster by a factor ≈&thinsp;1.4. The
        competition between rebound (<em>k</em><sub>rebound</sub>&thinsp;≈&thinsp;7.4&thinsp;×&thinsp;10⁹&thinsp;s⁻¹)
        and radical rotation/escape then determines stereospecificity retention
        at 40–90&thinsp;% — without any fitted parameter.
      </p>

      <h2>Testosterone 6β Regioselectivity</h2>
      <p>
        For CYP3A4 hydroxylation of testosterone, the 6β position is predicted
        as the dominant site at <strong>49&thinsp;%</strong> (experimental:
        50–70&thinsp;% in human liver microsomes). The framework combines the
        activation partition-depth Δ<em>M</em><sub>HAT</sub> (electronic,
        BDE-dependent) with a geometric accessibility factor <em>g</em><sub>i</sub>
        for each competing C–H position in the CYP3A4 active site.
      </p>

      <h2>Five Reaction Types, One Framework</h2>
      <p>
        Aliphatic C–H hydroxylation (Δ<em>M</em> = 0.65), benzylic (0.50),
        allylic (0.45), aromatic epoxide formation (0.38), and double-bond
        epoxidation (0.35) are all unified under the same three-body aperture.
        The distinguishing parameter is Δ<em>M</em><sub>HAT</sub> alone.
        Aromatic hydroxylation and epoxidation lack a primary H-isotope effect
        because the reaction coordinate does not involve H motion.
      </p>

      <h2>Validation</h2>
      <p>
        8&thinsp;/&thinsp;8 PASS: Δ<em>M</em><sub>HAT</sub>, activation energy
        (10&thinsp;kcal/mol), <em>k</em><sub>HAT</sub>, <em>k</em><sub>rebound</sub>,
        rate ratio, KIE, stereoretention, and 6β regioselectivity all within
        20–30&thinsp;% of experimental reference values.
      </p>
    </PaperPage>
  );
}
