import React, { useMemo, useState } from 'react'
import { SOURCES } from '../lib/corpus.js'
import { Pill } from './Primitives.jsx'

const FEATURES = [
  'reaction', 'participant', 'ec', 'equation', 'sequence', 'organism',
  'lineage', 'catalyses', 'pathway', 'buffer', 'ph', 'temperature',
  'device', 'device-settings', 'operator', 'date', 'dataset', 'compound',
]

export default function PlanBuilder() {
  const [src, setSrc] = useState('RXN')
  const [want, setWant] = useState(['reaction', 'participant'])
  const [budget, setBudget] = useState(4)
  const [nSteps, setNSteps] = useState(3)

  const toggle = (f) =>
    setWant(want.includes(f) ? want.filter((x) => x !== f) : [...want, f])

  const caps = SOURCES[src].capabilities
  const missing = want.filter((f) => !caps.includes(f))
  const ok = missing.length === 0

  // which sources COULD serve the request
  const alternatives = Object.entries(SOURCES)
    .filter(([k, v]) => k !== src && want.every((f) => v.capabilities.includes(f)))
    .map(([k]) => k)

  const exhausted = Math.max(0, nSteps - budget)

  return (
    <div className="grid g2">
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>Build a step</h3>

        <label className="ctl">source</label>
        <div className="seg">
          {Object.entries(SOURCES).map(([k, v]) => (
            <button key={k} className={src === k ? 'on' : ''} onClick={() => setSrc(k)}>
              {k}
            </button>
          ))}
        </div>
        <p className="note" style={{ marginTop: 8 }}>
          {SOURCES[src].label} · {SOURCES[src].shape}
        </p>

        <label className="ctl" style={{ marginTop: 16 }}>
          features this step requires
        </label>
        <div className="seg">
          {FEATURES.map((f) => (
            <button key={f} className={want.includes(f) ? 'on' : ''}
                    onClick={() => toggle(f)}
                    style={{ fontSize: 11.5, padding: '4px 9px' }}>
              {f}
            </button>
          ))}
        </div>

        <label className="ctl" style={{ marginTop: 18 }}>
          plan length <b>{nSteps}</b> steps · retrieval budget <b>{budget}</b>
        </label>
        <div style={{ display: 'flex', gap: 12 }}>
          <input type="range" min="1" max="8" value={nSteps}
                 onChange={(e) => setNSteps(+e.target.value)} />
          <input type="range" min="0" max="8" value={budget}
                 onChange={(e) => setBudget(+e.target.value)} />
        </div>
      </div>

      <div className="panel">
        <h3 style={{ marginTop: 0 }}>Static verdict — before any request is issued</h3>

        <div style={{ margin: '14px 0' }}>
          <Pill verdict={ok ? (exhausted ? 'exhausted' : 'answer') : 'unexpressed'} />
        </div>

        {!ok && (
          <>
            <div className="blocker">
              <b>blocker.</b> <code>{src}</code> cannot state:{' '}
              <b>{missing.join(', ')}</b>. The plan is refused at compile time,
              naming the offending step and feature.
            </div>
            <div className="unblock">
              <b>what would unblock it.</b>{' '}
              {alternatives.length
                ? <>route this step to <b>{alternatives.join(' or ')}</b>, which
                   declare every required feature</>
                : <>no single registered source declares all of these — split the
                   step, or compute the missing feature from a retrieved
                   attribute</>}
            </div>
          </>
        )}

        {ok && exhausted > 0 && (
          <>
            <div className="blocker">
              <b>blocker.</b> {exhausted} of {nSteps} steps will not run: the
              budget is spent before they are reached.
            </div>
            <div className="unblock">
              <b>what would unblock it.</b> raise the budget to {nSteps}.
            </div>
            <p className="note">
              Note this is a <em>different</em> verdict from an empty answer. A
              rows-only interface would return the same empty table for both,
              and the caller could not tell “nothing matched” from “we ran out of
              budget before looking”.
            </p>
          </>
        )}

        {ok && exhausted === 0 && (
          <p style={{ fontSize: 13.6 }}>
            Every required feature is contained in the source’s declared
            capability set, and the budget covers the plan. Lowering is total on
            this step, so it compiles.
          </p>
        )}

        <h3>Capability sets</h3>
        <table>
          <tbody>
            {Object.entries(SOURCES).map(([k, v]) => (
              <tr key={k} className={k === src ? 'hl' : ''}>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>{k}</td>
                <td style={{ fontSize: 11.6, color: 'var(--ink-dim)' }}>
                  {v.capabilities.map((c) => (
                    <code key={c} style={{
                      fontSize: 10.5, marginRight: 4, display: 'inline-block',
                      marginBottom: 3,
                      color: want.includes(c) ? 'var(--accent)' : undefined,
                      borderColor: want.includes(c) ? 'var(--accent)' : undefined,
                    }}>{c}</code>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          A capability claim describes a <em>compiler</em>, not a formalism. The
          two come apart in both directions, and only the compiler-level reading
          admits a decidable containment check.
        </p>
      </div>
    </div>
  )
}
