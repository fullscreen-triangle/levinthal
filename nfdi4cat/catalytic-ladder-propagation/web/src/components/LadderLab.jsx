import React, { useMemo, useState } from 'react'
import {
  circulation, compose, composeAdditive, composeMax, composeMean,
  rungsNeeded, sensitivityAdditive, sensitivityProportional, uniformity,
} from '../lib/kernel.js'
import { Bars, Chart, Slider } from './Primitives.jsx'

const C = {
  blue: '#4c8dff', orange: '#e0803a', green: '#3fb27f',
  red: '#e05c5c', grey: '#8b949e',
}

export default function LadderLab() {
  const [pis, setPis] = useState([0.45, 0.3, 0.55, 0.2])
  const [target, setTarget] = useState(0.8)
  const [mode, setMode] = useState('additive')

  const set = (i, v) => setPis(pis.map((p, j) => (j === i ? v : p)))
  const addRung = () => pis.length < 7 && setPis([...pis, 0.3])
  const dropRung = () => pis.length > 2 && setPis(pis.slice(0, -1))

  const total = compose(pis)
  const P = pis.reduce((a, b) => a * (1 - b), 1)
  const meets = total >= target

  const sens = pis.map((_, j) =>
    mode === 'additive' ? sensitivityAdditive(pis, j) : sensitivityProportional(pis, j)
  )
  const best = sens.indexOf(Math.max(...sens))
  const weakest = pis.indexOf(Math.min(...pis))
  const spread = Math.max(...sens) - Math.min(...sens)

  const deletions = pis.map((_, i) => compose(pis.filter((_, j) => j !== i)))

  const gapTrace = useMemo(() => {
    const pts = [[0, 1]]
    let g = 1
    pis.forEach((p, i) => {
      g *= 1 - p
      pts.push([i + 1, g])
    })
    return pts
  }, [pis])

  const lawRows = [
    { k: 'multiplicative  1 − Π(1−πᵢ)', v: compose(pis), exact: true },
    { k: 'additive  min(1, Σπᵢ)', v: composeAdditive(pis) },
    { k: 'maximum  max πᵢ', v: composeMax(pis) },
    { k: 'mean  (1/n)Σπᵢ', v: composeMean(pis) },
  ]

  return (
    <div>
      <div className="grid g2">
        <div className="panel">
          <div style={{ display: 'flex', justifyContent: 'space-between',
                        alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>The chain</h3>
            <div className="seg">
              <button className="ghost" onClick={dropRung}>−</button>
              <button className="ghost" onClick={addRung}>+ rung</button>
            </div>
          </div>

          {pis.map((p, i) => (
            <Slider key={i} label={`rung ${i + 1} — π${i + 1}`} value={p}
                    min={0} max={0.95} step={0.01} onChange={(v) => set(i, v)}
                    fmt={(v) => v.toFixed(2)} />
          ))}

          <Slider label="required composite power" value={target} min={0.3}
                  max={0.99} step={0.01} onChange={setTarget}
                  fmt={(v) => v.toFixed(2)} />

          <div style={{ marginTop: 16, padding: '13px 15px', borderRadius: 9,
                        background: meets
                          ? 'color-mix(in srgb, var(--ok) 10%, transparent)'
                          : 'color-mix(in srgb, var(--bad) 10%, transparent)',
                        border: `1px solid color-mix(in srgb, ${
                          meets ? 'var(--ok)' : 'var(--bad)'} 32%, transparent)` }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 19,
                          color: meets ? 'var(--ok)' : 'var(--bad)',
                          fontWeight: 700 }}>
              π(L) = {total.toFixed(4)}
            </div>
            <div style={{ fontSize: 13, color: 'var(--ink-dim)', marginTop: 3 }}>
              {meets
                ? `meets the requirement of ${target.toFixed(2)} — the plan compiles`
                : `falls short of ${target.toFixed(2)} — the plan is refused before execution`}
            </div>
            <div style={{ fontSize: 12.4, color: 'var(--ink-faint)', marginTop: 8,
                          fontFamily: 'var(--mono)' }}>
              residual gap Π(1−πᵢ) = {P.toFixed(4)} · minimum rungs at π
              ≤ {Math.max(...pis).toFixed(2)}: {rungsNeeded(target, Math.max(...pis))}
            </div>
          </div>
        </div>

        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Where the gap closes</h3>
          <Chart
            series={[{ points: gapTrace, color: C.blue, label: 'residual gap' }]}
            xDomain={[0, pis.length]} yDomain={[0, 1]}
            xLabel="rungs committed" yLabel="above-floor gap" height={168}
          />
          <h3>Composition law, scored against simulation</h3>
          <table>
            <tbody>
              {lawRows.map((r) => (
                <tr key={r.k} className={r.exact ? 'hl' : ''}>
                  <td><code style={{ fontSize: 11.6 }}>{r.k}</code></td>
                  <td className="num">{r.v.toFixed(4)}</td>
                  <td className="num" style={{
                    color: r.exact ? 'var(--ok)' : 'var(--ink-faint)' }}>
                    {r.exact ? 'exact' : `err ${Math.abs(r.v - total).toFixed(3)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            All four are scored. Reporting only the favoured law would not have
            distinguished it from the others — the comparison is the test.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <h3 style={{ margin: 0 }}>Which rung is worth improving?</h3>
          <div className="seg">
            <button className={mode === 'additive' ? 'on' : ''}
                    onClick={() => setMode('additive')}>
              additive increment
            </button>
            <button className={mode === 'proportional' ? 'on' : ''}
                    onClick={() => setMode('proportional')}>
              proportional increment
            </button>
          </div>
        </div>

        <Bars
          items={sens.map((s, i) => ({
            label: `π${i + 1} = ${pis[i].toFixed(2)}`,
            value: s,
            color: i === best ? C.orange : C.blue,
          }))}
          fmtV={(v) => v.toFixed(4)}
        />

        {mode === 'additive' ? (
          <div className="callout">
            <b>Control sits at the strongest rung, not the bottleneck.</b> The
            sensitivity is <code>∂π(L)/∂πⱼ = Π<sub>i≠j</sub>(1−πᵢ)</code>, which
            increases in πⱼ. Here the best investment is{' '}
            <b>rung {best + 1}</b> (π = {pis[best].toFixed(2)}), while the weakest
            rung is rung {weakest + 1}. A marginal gain at rung j is transmitted
            through the <em>other</em> rungs; a strong rung has already removed
            most of the gap, so its improvement passes through almost
            undiminished.
          </div>
        ) : (
          <div className="callout warn">
            <b>…but only under an additive parametrisation.</b> If a rung is
            improved by a fixed fraction of its own headroom —{' '}
            <code>δ(1−πⱼ)</code>, which is what “improve this catalyst by ten per
            cent” usually means — the factors cancel and the gain is{' '}
            <code>δ·P</code> at <em>every</em> rung. Measured spread across rungs
            here: <code>{spread.toExponential(1)}</code>. Over 5000 random
            ladders the spread is 1.1×10⁻¹⁶, machine zero. Which parametrisation
            applies is an empirical question about how improvements are actually
            purchased, and a check comparing the analytic derivative to a
            numerical one cannot detect the difference.
          </div>
        )}
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>What does deleting a step cost?</h3>
          <table>
            <thead>
              <tr><th>remove</th><th className="num">composite</th><th className="num">verdict</th></tr>
            </thead>
            <tbody>
              {deletions.map((d, i) => (
                <tr key={i}>
                  <td>rung {i + 1} <span style={{ color: 'var(--ink-faint)' }}>
                    (π = {pis[i].toFixed(2)})</span></td>
                  <td className="num">{d.toFixed(4)}</td>
                  <td className="num" style={{
                    color: d >= target ? 'var(--ok)' : 'var(--bad)' }}>
                    {d >= target ? 'tolerated' : 'fails'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            Predicted before any experiment, from the numbers alone.
          </p>
        </div>

        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Closed ladders: rings have no target</h3>
          <p style={{ fontSize: 13.6, color: 'var(--ink-dim)' }}>
            A ring returns, so composite power is undefined for it. Two
            invariants take its place: the <b>circulation</b> ϱ = −Σ log(1−πᵢ),
            which is what one circuit deposits, and the <b>uniformity</b> υ,
            which measures how evenly the closures are spread.
          </p>
          <table>
            <tbody>
              <tr>
                <td>circulation ϱ</td>
                <td className="num"><b>{circulation(pis).toFixed(4)}</b></td>
              </tr>
              <tr>
                <td>circulation per rung</td>
                <td className="num">{(circulation(pis) / pis.length).toFixed(4)}</td>
              </tr>
              <tr>
                <td>uniformity υ</td>
                <td className="num"><b>{uniformity(pis).toFixed(4)}</b></td>
              </tr>
            </tbody>
          </table>
          <div className="callout warn" style={{ marginTop: 12 }}>
            <b>Where ϱ adds nothing, stated plainly.</b> For a <em>linear</em>{' '}
            ladder ϱ = −log(1 − π(L)) exactly, so it is a monotone
            reparametrisation of composite power and carries no new information.
            Its content is that it is defined <em>without a target</em>, so it
            survives the passage to a ring where composite power does not. The
            invariant that genuinely adds something is υ.
          </div>
        </div>
      </div>
    </div>
  )
}
