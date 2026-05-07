import PaperPage from "@/components/PaperPage";
import { PAPER_BY_SLUG } from "@/data/papers";

export default function ClosedOrbit() {
  const paper = PAPER_BY_SLUG["closed-orbit"];
  return (
    <PaperPage paper={paper}>

      <h2>The Orbit Concept</h2>
      <p>
        The cytochrome P450 catalytic cycle is not merely a kinetic scheme —
        it is a <em>closed categorical orbit</em> in the receiver address
        space (n, &ell;, m, s). Each of the seven distinct chemical states
        occupies a unique lattice point in this four-dimensional space,
        and the cycle returns to its starting point after exactly eight
        categorical transitions. This is the synthesis result of
        Papers&nbsp;1–11.
      </p>

      <h2>The Seven States</h2>
      <ul>
        <li><strong>State 1</strong>: Resting Fe³⁺-H₂O, low spin — address (3,2,0,1)</li>
        <li><strong>State 2</strong>: Substrate-bound Fe³⁺, high spin — address (3,2,1,1)</li>
        <li><strong>State 3</strong>: Reduced Fe²⁺ — address (4,2,0,0)</li>
        <li><strong>State 4</strong>: Oxy-Fe²⁺-O₂ — address (4,2,1,0)</li>
        <li><strong>State 5</strong>: Peroxo-Fe³⁺-OO²⁻ — address (3,2,2,0)</li>
        <li><strong>State 6</strong>: Compound 0 (Fe³⁺-OOH) — address (3,2,3,0)</li>
        <li><strong>State 7</strong>: Compound I (Fe⁴⁺=O, π-radical) — address (4,3,0,1)</li>
      </ul>
      <p>
        Product release from Compound&nbsp;I returns the system to State&nbsp;1.
      </p>

      <h2>Newton's Cradle Non-Identity Theorem</h2>
      <p>
        A central result is that no two states share the same receiver address.
        The minimum pairwise Hamming distance across all 21 state pairs
        is&nbsp;1 (States 1 and 2 differ only in the&nbsp;<em>m</em>
        coordinate). This guarantees that the catalytic cycle involves seven
        genuinely distinct physical configurations — the system cannot shortcut
        from State&nbsp;1 to State&nbsp;7 without traversing all intermediate
        states.
      </p>

      <h2>Orbit Sum and Transition Depths</h2>
      <table className="text-xs w-full border-collapse my-4">
        <thead>
          <tr className="border-b border-gray-300">
            <th className="text-left p-1">Transition</th>
            <th className="text-right p-1">&Delta;M</th>
            <th className="text-right p-1">k (s⁻¹)</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["1→2 Substrate binding", "0.92", "4.0×10⁹"],
            ["2→3 First electron", "0.68", "5.1×10⁹"],
            ["3→4 O₂ binding", "0.55", "5.8×10⁹"],
            ["4→5 Second electron", "0.72", "4.9×10⁹"],
            ["5→Cpd0 Protonation", "0.45", "6.4×10⁹"],
            ["Cpd0→CpdI O-O heterolysis", "0.693", "5.0×10⁹"],
            ["CpdI C-H activation", "0.65", "5.2×10⁹"],
            ["Product release", "0.30", "7.4×10⁹"],
          ].map(([step, dm, k]) => (
            <tr key={step} className="border-b border-gray-100">
              <td className="p-1">{step}</td>
              <td className="text-right p-1">{dm}</td>
              <td className="text-right p-1">{k}</td>
            </tr>
          ))}
          <tr className="font-bold border-t border-gray-400">
            <td className="p-1">Sum</td>
            <td className="text-right p-1">4.963</td>
            <td className="text-right p-1">—</td>
          </tr>
        </tbody>
      </table>
      <p>
        The orbit sum &Sigma;&Delta;<em>M</em> = 4.963 lies in the
        predicted range [4.5, 6.0] for a seven-state closed orbit.
      </p>

      <h2>Poincaré Return Time</h2>
      <p>
        The intrinsic Poincaré return time is the sum of dwell times:
        <em>T</em><sub>return</sub> = &Sigma;(1/<em>k<sub>i</sub></em>) &asymp; 1.4 ps.
        The corresponding intrinsic rate is
        <em>k</em><sub>cat,intrinsic</sub> &asymp; 7&thinsp;&times;&thinsp;10¹¹&thinsp;s⁻¹.
        This is five orders of magnitude faster than the FMN→heme tunneling
        bottleneck from Paper&nbsp;11 (<em>k</em><sub>ET</sub> = 5&thinsp;&times;&thinsp;10⁶&thinsp;s⁻¹),
        confirming that <strong>chemistry is not rate-limiting</strong> in
        cytochrome P450 catalysis.
      </p>

      <h2>Anharmonic Closure</h2>
      <p>
        Every intrinsic transition satisfies &Delta;<em>M</em><sub>i</sub> &lt; ln(10) &asymp; 2.30
        (the classical sink threshold). The maximum intrinsic depth is 0.92
        (substrate binding), well below 2.30. No intermediate is a kinetic trap;
        the orbit is ergodic over the catalytic timescale.
      </p>

      <h2>Chemistry vs Electron Transfer</h2>
      <p>
        The ratio of the slowest chemical rate to the FMN→heme tunneling rate is:
      </p>
      <pre className="text-xs">
        k_chem,min / k_ET(Paper 11) = 4.0×10⁹ / 5×10⁶ ≈ 800
        k_HAT / k_ET = 5.2×10⁹ / 5×10⁶ ≈ 1040
      </pre>
      <p>
        All chemical steps are at least 800-fold faster than the
        electron-tunneling bottleneck. The rate hierarchy is clear:
        substrate diffusion &lt; CPR electron tunneling &lt;&lt; intrinsic chemistry.
      </p>

      <h2>Headline Numbers</h2>
      <ul>
        <li><strong>States in closed orbit</strong>: 7 (non-degenerate)</li>
        <li><strong>Poincaré return time</strong>: &asymp; 1.4 ps (ET-limited in vivo: &asymp; 400 ns)</li>
        <li><strong>k_chem / k_ET ratio</strong>: &asymp; 1000&thinsp;&times; (chemistry not rate-limiting)</li>
        <li><strong>Orbit sum &Sigma;&Delta;M</strong>: 4.963 &isin; [4.5, 6.0]</li>
      </ul>

      <h2>Validation</h2>
      <p>
        8&thinsp;/&thinsp;8 PASS: seven non-degenerate states, rate analysis
        (&Delta;<em>M</em><sub>max</sub> = 0.92 &gt; 0.5), orbit closure
        (&Sigma;&Delta;<em>M</em> = 4.963), Newton's Cradle non-identity
        (Hamming &ge; 1 for all pairs), Poincaré return time, anharmonic closure
        (all &Delta;<em>M</em> &lt; ln(10)), rate hierarchy
        (ratio &ge; 100), and full cycle summary.
      </p>
    </PaperPage>
  );
}
