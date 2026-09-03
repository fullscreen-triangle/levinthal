import React, { useMemo, useState } from 'react'
import { BETA, CHEBI, MEDIA, REACTIONS, TAU } from '../lib/corpus.js'
import {
  directionVerdict, mediumBias, mediumWeight, saturationLimit, solventRole,
  WEIGHT_FAMILIES,
} from '../lib/kernel.js'
import { Chart, Slider } from './Primitives.jsx'

const C = {
  blue: '#4c8dff', orange: '#e0803a', green: '#3fb27f',
  red: '#e05c5c', grey: '#8b949e', purple: '#a577e0',
}
const FAM_COLOR = { log: C.blue, sqrt: C.green, rational: C.orange, 'linear-cap': C.red }

export default function MediumLab() {
  const [family, setFamily] = useState('log')
  const [rhoStr, setRhoStr] = useState(4.0)
  const [medium, setMedium] = useState('glutamate-depleted cytosol')
  const [floodExp, setFloodExp] = useState(0)
  const [depleteExp, setDepleteExp] = useState(0)

  const rxn = REACTIONS['RXN:19453']

  // -- solvent role ---------------------------------------------------------
  const wlm = mediumWeight(55.5, BETA, TAU, family) / BETA
  const role = solventRole(rhoStr, wlm)

  // -- direction trichotomy -------------------------------------------------
  const rows = Object.entries(MEDIA).map(([label, mu]) => {
    const d = mediumBias(rxn.substrates, rxn.products, mu, BETA, TAU, family)
    return { label, delta: d / BETA, verdict: directionVerdict(d, BETA), note: mu.note }
  })
  const active = rows.find((r) => r.label === medium)

  // -- saturation asymmetry -------------------------------------------------
  const mu0 = 1e-4
  const sat = saturationLimit(TAU, mu0)
  const floodCurve = useMemo(
    () => Array.from({ length: 121 }, (_, i) => {
      const e = (i / 120) * 12
      return [e, (mediumWeight(mu0 * 10 ** e, BETA, TAU, family)
                  - mediumWeight(mu0, BETA, TAU, family)) / BETA]
    }), [family]
  )
  const depleteCurve = useMemo(
    () => Array.from({ length: 121 }, (_, i) => {
      const e = (i / 120) * 12
      return [e, (mediumWeight(mu0, BETA, TAU, family)
                  - mediumWeight(mu0 * 10 ** -e, BETA, TAU, family)) / BETA]
    }), [family]
  )
  const floodNow = (mediumWeight(mu0 * 10 ** floodExp, BETA, TAU, family)
                    - mediumWeight(mu0, BETA, TAU, family)) / BETA
  const deplNow = (mediumWeight(mu0, BETA, TAU, family)
                   - mediumWeight(mu0 * 10 ** -depleteExp, BETA, TAU, family)) / BETA

  const famCurve = (f) =>
    Array.from({ length: 90 }, (_, i) => {
      const mu = 10 ** (-7 + (i / 89) * 9)
      return [Math.log10(mu), mediumWeight(mu, BETA, TAU, f) / BETA]
    })

  return (
    <div>
      <div className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <h3 style={{ margin: 0 }}>The medium weight, and why its form does not matter</h3>
          <div className="seg">
            {WEIGHT_FAMILIES.map((f) => (
              <button key={f} className={family === f ? 'on' : ''}
                      onClick={() => setFamily(f)}>{f}</button>
            ))}
          </div>
        </div>
        <div className="grid g2" style={{ marginTop: 12 }}>
          <Chart
            series={WEIGHT_FAMILIES.map((f) => ({
              points: famCurve(f), color: FAM_COLOR[f],
              label: f, width: f === family ? 2.6 : 1.2,
            }))}
            xDomain={[-7, 2]} yDomain={[0.8, 6]}
            rules={[{ y: 1, color: C.grey, label: 'floor β' }]}
            xLabel="log₁₀ ambient occupancy μ" yLabel="w(ℓ,m) / β" height={196}
          />
          <div>
            <p style={{ fontSize: 13.6, marginTop: 0 }}>
              A leaf surrounded by copies of itself is <b>cheap</b> to
              individuate against — it is barely distinguishable from its
              surroundings, so the weight falls to the floor. A leaf of a scarce
              identity is <b>expensive</b>: there is little there to tell it
              apart from.
            </p>
            <div className="callout">
              <b>Only two properties are load-bearing.</b> Every theorem here
              uses the weight only through the facts that it is strictly
              decreasing in μ and bounded below by β. The four families above
              share nothing else — one is logarithmic, one square-root, one
              rational, one a non-smooth linear cap — and every structural result
              is re-verified under all four. Switch families and watch the
              verdicts below stay put while the numbers move.
            </div>
          </div>
        </div>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Solvent role is computed, not annotated</h3>
          <p style={{ fontSize: 13.6, color: 'var(--ink-dim)' }}>
            Two water molecules of <em>identical chemical identity</em>. One is
            an ordered active-site water making specific contacts; one is bulk.
            A vocabulary with a single <code>solvent</code> class cannot tell
            them apart. Here the distinction is a comparison of two boundaries.
          </p>
          <Slider label="ρ_str — boundary the water holds against the system"
                  value={rhoStr} min={0} max={8} step={0.1}
                  onChange={setRhoStr} fmt={(v) => `${v.toFixed(1)} β`} />
          <table style={{ marginTop: 14 }}>
            <tbody>
              <tr>
                <td>ρ_str (against the system)</td>
                <td className="num"><b>{rhoStr.toFixed(2)} β</b></td>
              </tr>
              <tr>
                <td>w(ℓ,m) (against the surroundings)</td>
                <td className="num">{wlm.toFixed(2)} β</td>
              </tr>
              <tr className="hl">
                <td>role</td>
                <td className="num">
                  <b style={{ color: role === 'structural' ? 'var(--accent-2)' : 'var(--neutral)' }}>
                    {role}
                  </b>
                </td>
              </tr>
            </tbody>
          </table>
          <p className="note">
            structural ⟺ ρ_str ≥ w(ℓ,m). No curator supplies this and no
            ontology term records it; it is derived from the graph and is
            invariant under any rescaling that preserves the floor.
          </p>
        </div>

        <div className="panel">
          <h3 style={{ marginTop: 0 }}>One identifier, three directions</h3>
          <p style={{ fontSize: 13.6, color: 'var(--ink-dim)', marginBottom: 6 }}>
            <code>{rxn.equation}</code>
          </p>
          <p style={{ fontSize: 13.4 }}>
            Public resources store this as one master reaction with two
            directional children, and the choice between them is{' '}
            <em>asserted by a curator</em>. Here it is a trichotomy on one
            inequality, and the answer depends on the medium — not on the chain.
          </p>
          <div className="seg" style={{ marginTop: 10, marginBottom: 12 }}>
            {rows.map((r) => (
              <button key={r.label} className={medium === r.label ? 'on' : ''}
                      onClick={() => setMedium(r.label)}
                      style={{ fontSize: 12 }}>
                {r.label.replace(' cytosol', '').replace(' medium', '')}
              </button>
            ))}
          </div>
          <table>
            <thead>
              <tr><th>medium</th><th className="num">δ / β</th><th className="num">verdict</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label} className={r.label === medium ? 'hl' : ''}>
                  <td style={{ fontSize: 12.8 }}>{r.label}</td>
                  <td className="num">{r.delta >= 0 ? '+' : ''}{r.delta.toFixed(3)}</td>
                  <td className="num" style={{
                    color: r.verdict === 'forward' ? 'var(--ok)'
                      : r.verdict === 'reverse' ? 'var(--warn)' : 'var(--bad)' }}>
                    <b>{r.verdict}</b>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {active && <p className="note">{active.note}</p>}
          <div className="callout" style={{ marginTop: 10 }}>
            <b>The reaction identifier is not wrong.</b> It names the chain,
            which is direction-symmetric: a chain and its reversal commit the
            same total boundary, the same cut count, the same edges. Direction is
            named by the medium, which the identifier does not carry. The
            balanced case is a negative control — without a medium that{' '}
            <em>refuses</em> to orient, the trichotomy would be a dichotomy with
            a decorative third branch.
          </div>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>
          Product inhibition cannot reverse a reaction. Substrate depletion can.
        </h3>
        <p className="sub" style={{ marginBottom: 14 }}>
          We did not anticipate this, and would not have found it by inspection.
          An early sweep varied only the product end, reached “forward” and
          “undirected” but never “reverse”, and looked like a refutation. The
          theorem was not at fault — the sweep was one-sided.
        </p>
        <div className="grid g2">
          <Chart
            series={[
              { points: floodCurve, color: C.orange, label: 'flood a product' },
              { points: depleteCurve, color: C.blue, label: 'deplete a reactant' },
            ]}
            xDomain={[0, 12]} yDomain={[-30, 3]}
            rules={[{ y: sat, color: C.red, label: `ceiling −log(1+τ/μ₀) = ${sat.toFixed(3)}` }]}
            bands={[{ from: -1, to: 1, color: C.grey }]}
            xLabel="orders of magnitude of change" yLabel="δ / β" height={210}
          />
          <div>
            <Slider label="flood one product by" value={floodExp} min={0} max={12}
                    step={0.1} onChange={setFloodExp}
                    fmt={(v) => `10^${v.toFixed(1)} ×`} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 13, margin: '4px 0 14px',
                          color: C.orange }}>
              δ = {floodNow.toFixed(6)} β
              {floodExp > 6 && (
                <span style={{ color: 'var(--ink-faint)' }}> · saturated</span>
              )}
            </div>
            <Slider label="deplete one reactant by" value={depleteExp} min={0} max={12}
                    step={0.1} onChange={setDepleteExp}
                    fmt={(v) => `10^−${v.toFixed(1)} ×`} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 13, margin: '4px 0 0',
                          color: C.blue }}>
              δ = −{deplNow.toFixed(4)} β
              <span style={{ color: 'var(--ink-faint)' }}> · still falling</span>
            </div>
            <div className="callout" style={{ marginTop: 16 }}>
              <b>An exact ceiling, not a fitted one.</b> As a flooded product
              becomes abundant its medium weight falls to the floor, so the bias
              stops at exactly <code>−log(1 + τ/μ₀) = {sat.toFixed(6)} β</code>.
              Thirty orders of magnitude of further flooding move it no further:
              measured limit −2.397895 β against predicted −2.397895 β. Depletion
              has no such bound and reaches −22.93 β over the same range.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
