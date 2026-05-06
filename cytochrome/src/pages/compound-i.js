import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function CompoundI() {
  const paper = PAPER_BY_SLUG["compound-i"];
  return (
    <PaperPage paper={paper}>
      <h2>What happens after the electrons arrive</h2>
      <p>
        Paper 4 (Transfer) is the headline: the apparatus stack observes
        the electron arriving at heme. This paper picks up the chemistry
        that follows. It is not the monograph's centre of gravity — it
        is a demonstration that the framework, having observed the
        transfer, also predicts the downstream catalytic chemistry
        without additional machinery.
      </p>

      <h2>Compound I as a single d_C = 1 aperture</h2>
      <p>
        Compound I (Fe⁴⁺=O porphyrin•⁺) is the most controversial
        intermediate in all of biocatalysis. The framework treats its
        formation as a partition transition involving simultaneously:
      </p>
      <ul>
        <li>O–O bond-order trajectory from 1 (peroxo) to 0 (cleaved)</li>
        <li>Proton arrival from the I-helix Asp251/Thr252 water network</li>
        <li>Fe(III) → Fe(IV) redox transition</li>
        <li>Porphyrin radical localisation</li>
      </ul>
      <p>
        These are facets of one aperture, not four sequential events.
        The bond-order partition coordinate β ∈ &#123;0, 1&#125; gives
        ΔM = ln 2 ≈ 0.693 for the cleavage.
      </p>

      <h2>Anharmonic non-recurrence</h2>
      <p>
        Bond-breaking is structurally guaranteed by the anharmonicity of
        the Morse potential, not a rare event awaiting a thermal
        fluctuation. By the generative-novelty corollary of the
        triple-isomorphism architecture, exact recurrence has Lebesgue
        measure zero — every catalytic turnover ends in a slightly
        different conformation.
      </p>

      <h2>PCET concerted vs sequential</h2>
      <p>
        The framework predicts a <strong>10× rate ratio</strong> between
        concerted PCET (d_C = 1) and sequential PCET (d_C = 2), and a
        KIE of ~2 for concerted (water-network mediated) vs ~6 for
        sequential (direct proton transfer). The experimental KIE for
        Compound I formation is ~1.7 (Vatsis 2002), supporting the
        concerted mechanism.
      </p>

      <h2>Spectroscopic match</h2>
      <p>
        Eight Rittle–Green spectroscopic observables (UV-Vis Soret peak,
        EPR g-tensor, Mössbauer δ and ΔE<sub>Q</sub>, Resonance Raman
        ν₄/ν₂/ν₃, ENDOR hyperfine) all match within 20 % using the same
        S-coordinate (0.860, 0.515, 0.595) for Compound I. No
        per-observable fitting; the partition coordinate is one input
        evaluated under R<sub>bio</sub> in eight different modalities.
      </p>
    </PaperPage>
  );
}
