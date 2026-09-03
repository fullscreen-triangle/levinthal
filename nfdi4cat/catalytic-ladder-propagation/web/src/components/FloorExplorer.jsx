import React, { useMemo, useState } from 'react'
import {
  addEdge, cutWeight, edgeList, floorOf, makeGraph, minEdgeWeight,
  separationCost,
} from '../lib/kernel.js'
import { Slider } from './Primitives.jsx'

/**
 * Build a "buried residue" graph: a tightly packed core joined to a solvent-
 * exposed shell by one link, with the medium adjacent to the shell only.
 */
function buildGraph(coreW, linkW, shellW) {
  const g = makeGraph()
  const core = ['a0', 'a1', 'a2', 'a3']
  const shell = ['b0', 'b1', 'b2']
  for (let i = 0; i < core.length; i++)
    for (let j = i + 1; j < core.length; j++) addEdge(g, core[i], core[j], coreW)
  addEdge(g, 'a0', 'b0', linkW)
  shell.forEach((b) => addEdge(g, b, 'm', shellW))
  for (let i = 0; i < shell.length - 1; i++) addEdge(g, shell[i], shell[i + 1], shellW)
  return { g, core, shell }
}

const POS = {
  a0: [148, 118], a1: [96, 76], a2: [96, 160], a3: [58, 118],
  b0: [244, 118], b1: [300, 74], b2: [300, 162],
  m: [392, 118],
}

export default function FloorExplorer() {
  const [coreW, setCoreW] = useState(50)
  const [linkW, setLinkW] = useState(1)
  const [shellW, setShellW] = useState(1)
  const [probe, setProbe] = useState('a0')

  const { g, core } = useMemo(
    () => buildGraph(coreW, linkW, shellW), [coreW, linkW, shellW]
  )
  const sep = useMemo(() => separationCost(g, probe), [g, probe])
  const { beta, witness } = useMemo(() => floorOf(g), [g])
  const minEdge = minEdgeWeight(g)
  const region = new Set(sep.region)
  const cutEdges = edgeList(g).filter(
    ({ u, v }) => region.has(u) !== region.has(v)
  )

  const isRegion = sep.region.length > 1

  return (
    <div className="panel">
      <div className="grid g2">
        <div>
          <svg viewBox="0 0 440 236" style={{ width: '100%', height: 'auto' }}>
            {/* the minimising region */}
            {isRegion && (
              <ellipse
                cx={sep.region.every((v) => core.includes(v)) ? 104 : 190}
                cy={118}
                rx={sep.region.every((v) => core.includes(v)) ? 76 : 150}
                ry={72}
                fill="var(--accent)" opacity="0.11"
                stroke="var(--accent)" strokeDasharray="5 4" strokeWidth="1.3"
              />
            )}
            {edgeList(g).map(({ u, v, weight }, i) => {
              const inCut = cutEdges.some(
                (e) => (e.u === u && e.v === v) || (e.u === v && e.v === u)
              )
              const [x1, y1] = POS[u]
              const [x2, y2] = POS[v]
              return (
                <g key={i}>
                  <line
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={inCut ? 'var(--bad)' : 'var(--line)'}
                    strokeWidth={inCut ? 2.6 : 1.4}
                    strokeDasharray={inCut ? '6 3' : undefined}
                  />
                  <text
                    x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 4}
                    fontSize="9" textAnchor="middle" fontFamily="var(--mono)"
                    fill={inCut ? 'var(--bad)' : 'var(--ink-faint)'}
                  >
                    {weight}
                  </text>
                </g>
              )
            })}
            {Object.entries(POS).map(([v, [x, y]]) => {
              const isMed = v === 'm'
              const isProbe = v === probe
              return (
                <g key={v} onClick={() => !isMed && setProbe(v)}
                   style={{ cursor: isMed ? 'default' : 'pointer' }}>
                  <circle
                    cx={x} cy={y} r={isMed ? 17 : 14}
                    fill={isMed ? 'var(--panel-2)' : isProbe ? 'var(--accent)' : 'var(--panel-2)'}
                    stroke={isProbe ? 'var(--accent)' : 'var(--line)'}
                    strokeWidth={isProbe ? 2.4 : 1.4}
                  />
                  <text x={x} y={y + 4} textAnchor="middle" fontSize="11"
                        fontFamily="var(--mono)" fontWeight="600"
                        fill={isProbe ? '#fff' : 'var(--ink)'}>
                    {v}
                  </text>
                </g>
              )
            })}
            <text x={392} y={158} textAnchor="middle" fontSize="9.5"
                  fill="var(--ink-faint)">medium</text>
            <text x={104} y={214} textAnchor="middle" fontSize="9.5"
                  fill="var(--ink-faint)">buried core</text>
            <text x={272} y={214} textAnchor="middle" fontSize="9.5"
                  fill="var(--ink-faint)">exposed shell</text>
          </svg>
          <p className="note">
            Click any vertex to probe it. Red dashed edges are the minimum cut;
            the shaded region is the minimiser.
          </p>
        </div>

        <div>
          <Slider label="core packing weight W" value={coreW} min={1} max={80}
                  step={1} onChange={setCoreW} />
          <Slider label="core → shell link" value={linkW} min={0.1} max={40}
                  step={0.1} onChange={setLinkW} fmt={(v) => v.toFixed(1)} />
          <Slider label="shell ↔ medium" value={shellW} min={0.1} max={40}
                  step={0.1} onChange={setShellW} fmt={(v) => v.toFixed(1)} />

          <table style={{ marginTop: 18 }}>
            <tbody>
              <tr>
                <td>separation cost σ({probe})</td>
                <td className="num"><b>{sep.cost.toFixed(2)}</b></td>
              </tr>
              <tr>
                <td>minimising region</td>
                <td className="num">{'{' + sep.region.join(', ') + '}'}</td>
              </tr>
              <tr className={minEdge !== beta ? 'hl' : ''}>
                <td>floor β (min over all vertices)</td>
                <td className="num"><b>{beta.toFixed(2)}</b> <span
                  style={{ color: 'var(--ink-faint)' }}>at {witness}</span></td>
              </tr>
              <tr className={minEdge !== beta ? 'hl' : ''}>
                <td>smallest edge weight</td>
                <td className="num">{minEdge.toFixed(2)}</td>
              </tr>
            </tbody>
          </table>

          {minEdge !== beta && (
            <div className="callout warn" style={{ marginTop: 14 }}>
              <b>The floor is not an edge weight.</b> Here the smallest edge is{' '}
              <code>{minEdge.toFixed(2)}</code> while β = <code>{beta.toFixed(2)}</code>.
              A cheap edge sitting inside an expensive cut does not make that cut
              cheap. This is the distinction the whole architecture turns on —
              and a query over stored weights sees the edge, not the floor.
            </div>
          )}

          {isRegion && (
            <div className="callout" style={{ marginTop: 12 }}>
              <b>Identity is a region, not a point.</b> The cheapest way to
              individuate <code>{probe}</code> is to cut out{' '}
              <code>{'{' + sep.region.join(', ') + '}'}</code> — the whole packed
              core — not <code>{probe}</code> alone. Its identity is a property
              of the core. Any scheme assigning identity residue-by-residue is
              assigning something else.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
