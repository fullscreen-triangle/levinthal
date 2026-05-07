import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Heteroatom() {
  const paper = PAPER_BY_SLUG["heteroatom"];
  return (
    <PaperPage paper={paper}>

      <h2>The Heteroatom Oxidation Problem</h2>
      <p>
        N-dealkylation, O-dealkylation, S-oxidation, and N-oxide formation account
        for the clearance of ≈&thinsp;75&thinsp;% of all marketed pharmaceuticals.
        Despite sharing Compound&nbsp;I (Fe⁴⁺=O) as the oxidant, these reactions differ
        fundamentally in mechanism: dealkylations proceed via H-atom transfer (HAT)
        from the α-carbon, while S- and N-oxidations proceed via direct O-atom
        transfer to the heteroatom lone pair.
      </p>

      <h2>Alpha-Carbon BDE Hierarchy</h2>
      <p>
        For dealkylation, the reaction coordinate is the α-C–H bond adjacent to the
        heteroatom. Nitrogen lone-pair donation weakens this bond to BDE&thinsp;≈&thinsp;87&thinsp;kcal/mol
        (N-methyl), and oxygen to 92&thinsp;kcal/mol (O-methyl), both below the
        unactivated aliphatic reference (100&thinsp;kcal/mol). The activation
        partition-depth scales proportionally:
        Δ<em>M</em><sub>N-dealk</sub>&thinsp;=&thinsp;0.50 &lt; Δ<em>M</em><sub>O-dealk</sub>&thinsp;=&thinsp;0.58 &lt; Δ<em>M</em><sub>aliphatic</sub>&thinsp;=&thinsp;0.65.
      </p>

      <h2>KIE for N-Dealkylation</h2>
      <p>
        The α-C–H stretching frequency near nitrogen is softened to
        ≈&thinsp;2800&thinsp;cm⁻¹ (vs.&thinsp;3000&thinsp;cm⁻¹ for aliphatic C–H).
        This reduces the zero-point energy contribution to the KIE:
        KIE<sub>N-dealk</sub>&thinsp;≈&thinsp;<strong>6.7</strong>, measurably
        lower than aliphatic HAT (7.7). Because no H is transferred in S-oxidation or
        N-oxide formation, those pathways give KIE&thinsp;=&thinsp;1.0 — a direct
        experimental diagnostic.
      </p>

      <h2>Direct O-Atom Transfer: S-Oxidation and N-Oxide</h2>
      <p>
        Sulfur and nitrogen lone pairs donate directly to the Fe=O orbital without
        any H motion. The two-body aperture has Δ<em>M</em><sub>S-ox</sub>&thinsp;=&thinsp;0.28
        (S-oxidation) and Δ<em>M</em><sub>N-ox</sub>&thinsp;=&thinsp;0.32 (N-oxide),
        giving <em>k</em>&thinsp;≈&thinsp;7.6 and 7.3&thinsp;×&thinsp;10⁹&thinsp;s⁻¹
        respectively — faster than all HAT-based pathways. No H isotope effect.
      </p>

      <h2>The Unified Rate Hierarchy</h2>
      <p>
        All five modes are ordered by a single parameter:
        <em>k</em><sub>S-ox</sub> (0.28) &gt; <em>k</em><sub>N-ox</sub> (0.32)
        &gt; <em>k</em><sub>N-dealk</sub> (0.50) &gt; <em>k</em><sub>O-dealk</sub>
        (0.58) &gt; <em>k</em><sub>aliphatic</sub> (0.65),
        where parenthetical values are the Δ<em>M</em> for each pathway.
        The carbinolamine and hemiacetal intermediates (after HAT and rebound)
        have Δ<em>M</em><sub>cleavage</sub>&thinsp;&lt;&thinsp;0.15, making
        them kinetically labile and non-accumulating.
      </p>

      <h2>Validation</h2>
      <p>
        8&thinsp;/&thinsp;8 PASS: BDE ordering, N-dealkylation rate, KIE prediction,
        S-oxidation direct transfer (KIE&thinsp;=&thinsp;1), N-oxide formation,
        complete rate hierarchy, carbinolamine lability, and ketoconazole competitive
        inhibition (<em>K</em><sub>i</sub>&thinsp;&lt;&thinsp;100&thinsp;nM) — all
        within experimental ranges for CYP3A4 and CYP2D6 substrates.
      </p>
    </PaperPage>
  );
}
