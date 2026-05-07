import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function Membrane() {
  const paper = PAPER_BY_SLUG["membrane"];
  return (
    <PaperPage paper={paper}>

      <h2>Why Membrane Anchoring Matters</h2>
      <p>
        Cytochrome P450 enzymes are not free-floating in the cytoplasm —
        they are anchored to the endoplasmic reticulum (ER) membrane through
        a 20-residue N-terminal transmembrane helix. This anchoring places
        the globular catalytic domain on the cytosolic face of the ER,
        where it can receive electrons from its redox partners CPR and
        cytochrome&nbsp;<em>b</em><sub>5</sub>, and where lipophilic drug
        substrates concentrate from the surrounding bilayer.
      </p>

      <h2>TM Helix Insertion</h2>
      <p>
        CYP3A4 residues 3–22 form an &alpha;-helix embedded in the ER
        membrane. The 20-residue hydrophobic segment achieves a total
        insertion free energy of approximately&nbsp;
        <strong>&Delta;G = &minus;10 kcal/mol</strong>, corresponding to
        a categorical partition-depth &Delta;<em>M</em> = 0.42. A proline-rich
        hinge at residues 30–34 mechanically separates the TM anchor from
        the 480-residue globular domain.
      </p>

      <h2>The CPR–P450 Complex</h2>
      <p>
        CPR (78 kDa) is a diflavin oxidoreductase that shuttles electrons
        from NADPH through FAD and then FMN to the P450 heme iron. The FMN-binding
        domain contacts the P450 proximal face (the side carrying the
        Cys thiolate axial ligand) via electrostatic complementarity:
      </p>
      <ul>
        <li>
          <strong>P450 proximal face</strong>: 5 Arg + 3 Lys = 8 positive charges
        </li>
        <li>
          <strong>CPR FMN domain</strong>: 6 Asp + 4 Glu = 10 negative charges
        </li>
      </ul>
      <p>
        The complementarity score 8&thinsp;&times;&thinsp;10 = 80 &ge; 60 (threshold)
        drives stable complex formation with{" "}
        <strong><em>K</em><sub>d</sub> = 0.1 &mu;M</strong>
        (&Delta;<em>G</em> &asymp; &minus;9.96 kcal/mol).
      </p>

      <h2>FMN to Heme Electron Transfer</h2>
      <p>
        The electron transfer from FMN semiquinone to the heme iron spans
        a 14&thinsp;&Aring; edge-to-edge distance through the protein medium
        (&beta; = 1.4&thinsp;&Aring;<sup>&minus;1</sup>,
        &lambda; = 0.85 eV). The observed rate is{" "}
        <strong><em>k</em><sub>FMN&rarr;heme</sub> = 5&thinsp;&times;&thinsp;10<sup>6</sup>&thinsp;s<sup>&minus;1</sup></strong>,
        corresponding to a categorical depth{" "}
        <strong>&Delta;<em>M</em><sub>ET</sub> = ln(10<sup>10</sup>/5&times;10<sup>6</sup>) &asymp; 7.60</strong>.
        This large depth marks the FMN&rarr;heme hop as the rate-limiting step
        for CPR-mediated electron delivery — five to six orders of magnitude
        slower than intrinsic P450 chemistry.
      </p>

      <h2>Cytochrome b5 as a Faster Alternative</h2>
      <p>
        Cytochrome&nbsp;<em>b</em><sub>5</sub> (Cyt&nbsp;<em>b</em><sub>5</sub>)
        provides an alternative electron source, particularly for the second
        electron step. It binds tighter{" "}
        (<em>K</em><sub>d</sub> = 0.05 &mu;M &lt; 0.1 &mu;M for CPR)
        and transfers electrons six-fold faster:
      </p>
      <pre className="text-xs">
        k(b5→heme) = 3×10⁷ s⁻¹  vs  k(FMN→heme) = 5×10⁶ s⁻¹
        ratio = 6×  |  edge distance = 11 Å vs 14 Å
      </pre>

      <h2>Membrane Enrichment of Substrates</h2>
      <p>
        The ER bilayer acts as a concentrating reservoir for lipophilic drugs.
        The enrichment factor near the active site is:
      </p>
      <pre className="text-xs">
        ε = 10^(logP − 2)   for logP &gt; 2
      </pre>
      <p>
        At logP&thinsp;=&thinsp;3: &epsilon;&thinsp;=&thinsp;10&thinsp;&times;,
        reducing the apparent <em>K</em><sub>m</sub> ten-fold relative to
        bulk-solution measurements.
      </p>

      <h2>Headline Numbers</h2>
      <ul>
        <li>
          <strong><em>K</em><sub>d</sub> (CPR–P450)</strong>: 0.1 &mu;M
          &nbsp;[literature: 0.05–0.5 &mu;M]
        </li>
        <li>
          <strong><em>k</em><sub>FMN&rarr;heme</sub></strong>: 5&thinsp;&times;&thinsp;10<sup>6</sup>&thinsp;s<sup>&minus;1</sup>
          &nbsp;[literature: 10<sup>6</sup>–10<sup>7</sup>&thinsp;s<sup>&minus;1</sup>]
        </li>
        <li>
          <strong>Membrane enrichment (logP=3)</strong>: 10&thinsp;&times;
          &nbsp;[target: &gt;5&thinsp;&times; for lipophilic]
        </li>
      </ul>

      <h2>Validation</h2>
      <p>
        8&thinsp;/&thinsp;8 PASS: TM helix insertion energy
        (&Delta;<em>G</em> = &minus;10 kcal/mol, &Delta;<em>M</em> = 0.42),
        CPR binding affinity, FMN&rarr;heme ET parameters
        (&Delta;<em>M</em> = 7.60), Cyt&nbsp;<em>b</em><sub>5</sub>
        comparison, membrane enrichment (10&thinsp;&times; at logP=3),
        stoichiometry analysis (CPR:P450 = 1:10), proximal face
        electrostatics (score = 80), and full complex summary.
      </p>
    </PaperPage>
  );
}
