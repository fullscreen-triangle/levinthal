import React, { useMemo, useState } from 'react'
import {
  addEdge, edgeList, floorOf, makeGraph, minEdgeWeight, separationPair,
} from '../lib/kernel.js'
import { Slider } from './Primitives.jsx'

/**
 * The two systems of the separation theorem.  They differ ONLY in the weight
 * of the medium edges at spectator vertices that lie on no path between the
 * queried pair.  Those weights move the floor, which is a minimum over every
 * vertex, and therefore move the verdict about a pair they never touch.
 */
function build(spectatorW, nSpec = 3) {
  const g = makeGraph()
  addEdge(g, 'v0', 'x', 1)
  addEdge(g, 'v0', 'm', 1)
  addEdge(g, 'x', 'm', 1)
  for (let i = 0; i < nSpec; i++) addEdge(g, `y${i}`, 'm', spectatorW)
  return g
}

const QUERIES = [
  'SELECT ?s ?p ?o WHERE { ?s ?p ?o }',
  'SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }',
  'SELECT ?s (COUNT(?o) AS ?d) WHERE { ?s ?p ?o } GROUP BY ?s',
  'SELECT ?o WHERE { :v0 ?p ?o }',
  'ASK { :v0 :contact+ :x }',
  'ASK { :v0 :contact* ?z }',
  'SELECT DISTINCT ?o WHERE { :v0 :contact* ?o }',
  'SELECT DISTINCT ?z WHERE { :v0 :contact/:contact ?z }',
  'SELECT ?s WHERE { ?s :contact :m }',
  'SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s ?p ?o }',
  'SELECT ?a ?b ?c WHERE { ?a ?p ?b . ?b ?q ?c }',
  'SELECT ?s ?o WHERE { ?s ?p ?o OPTIONAL { ?o ?q ?z } FILTER(?s != ?o) }',
  'SELECT (MIN(?d) AS ?lo) (MAX(?d) AS ?hi) WHERE { SELECT ?s (COUNT(?o) AS ?d) … }',
]

export default function Separation() {
  const [w2, setW2] = useState(2)
  const [showWeights, setShowWeights] = useState(false)

  const g1 = useMemo(() => build(1), [])
  const g2 = useMemo(() => build(w2), [w2])

  const analyse = (g) => {
    const sep = separationPair(g, 'v0', 'x')
    const { beta } = floorOf(g)
    return { sep, beta, accountable: sep <= beta, minEdge: minEdgeWeight(g) }
  }
  const A = analyse(g1)
  const B = analyse(g2)

  const triples = (g) =>
    edgeList(g)
      .map(({ u, v }) => [u, v].sort().join(' :contact '))
      .sort()

  const t1 = triples(g1)
  const t2 = triples(g2)
  const sameTriples = JSON.stringify(t1) === JSON.stringify(t2)
  const differs = A.accountable !== B.accountable

  return (
    <div>
      <div className="panel">
        <Slider
          label="weight of the medium edges at the spectator vertices in system B"
          value={w2} min={0.5} max={4} step={0.1} onChange={setW2}
          fmt={(v) => v.toFixed(1)}
        />
        <p className="note" style={{ marginTop: 4 }}>
          The spectators <code>y0…y2</code> touch only the medium. They lie on
          no path between <code>v0</code> and <code>x</code>, so they cannot
          affect the separation cost of that pair — but they can and do move
          the floor, which is a minimum over <em>every</em> vertex.
        </p>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        {[['System A', g1, A], ['System B', g2, B]].map(([label, g, r]) => (
          <div className="panel" key={label}>
            <h3 style={{ marginTop: 0 }}>{label}</h3>
            <table>
              <tbody>
                <tr>
                  <td>σ(v₀, x) — the queried pair</td>
                  <td className="num"><b>{r.sep.toFixed(2)}</b></td>
                </tr>
                <tr>
                  <td>floor β — min over all vertices</td>
                  <td className="num"><b>{r.beta.toFixed(2)}</b></td>
                </tr>
                <tr>
                  <td>smallest stored edge weight</td>
                  <td className="num">{r.minEdge.toFixed(2)}</td>
                </tr>
                <tr className="hl">
                  <td>accountable at ε = 0?  (σ ≤ β)</td>
                  <td className="num">
                    <b style={{ color: r.accountable ? 'var(--ok)' : 'var(--bad)' }}>
                      {r.accountable ? 'admissible' : 'inadmissible'}
                    </b>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ))}
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h3 style={{ margin: 0 }}>What a triple store records</h3>
          <button className={showWeights ? 'on' : ''}
                  onClick={() => setShowWeights(!showWeights)}>
            {showWeights ? 'hide' : 'also store the weights as attributes'}
          </button>
        </div>

        <div className="grid g2" style={{ marginTop: 12 }}>
          {[['A', g1, t1], ['B', g2, t2]].map(([lab, g, tt]) => (
            <pre key={lab} style={{ fontSize: 11.6 }}>
{tt.map((t) => `:${t.replace(/ /g, ' :').replace(/^:/, '')} .`).join('\n')}
{showWeights
  ? '\n\n# with weights as attributes:\n' +
    edgeList(g).map((e, i) =>
      `:c${i} :from :${e.u} ; :to :${e.v} ; :weight ${e.weight} .`).join('\n')
  : ''}
            </pre>
          ))}
        </div>

        <div className={sameTriples ? 'callout' : 'callout warn'}
             style={{ marginTop: 4 }}>
          {sameTriples ? (
            <>
              <b>The two systems record identical triples.</b> {t1.length} each,
              set difference empty.{' '}
              {differs ? (
                <>Yet their accountability verdicts are <em>opposite</em>. No
                retrieval query can tell them apart, because a triple asserts
                that a relation holds and carries no cost.</>
              ) : (
                <>At this spectator weight the verdicts happen to agree — move
                the slider until β crosses σ = 2.</>
              )}
            </>
          ) : (
            <><b>Triples differ</b> — adjust so only the weights change.</>
          )}
        </div>

        {showWeights && (
          <div className="callout warn">
            <b>Storing the weights does not close the gap.</b> A query can now
            read the minimum stored <em>edge</em> weight — {A.minEdge.toFixed(2)} in
            both systems — but the verdict depends on β, which is a minimum over
            vertices of a minimum over subsets. Conjunctive queries with path
            expressions contain no operator that forms a minimum over subsets of
            the domain. The obstruction is expressibility, not cost: a minimum
            cut is polynomial to compute and impossible to phrase.
          </div>
        )}
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>
          Thirteen retrieval query forms, two engines, zero separations
        </h3>
        <p className="sub" style={{ marginBottom: 14 }}>
          Each form below was executed against both systems on{' '}
          <code>rdflib 7.6.0</code> and <code>pyoxigraph 0.5.9</code> — two
          independently developed engines. The engines agreed with each other on
          every query, and none of the thirteen separated the two systems.
        </p>
        <table>
          <thead>
            <tr>
              <th>#</th><th>query form</th>
              <th className="num">separates A from B?</th>
              <th className="num hide-sm">separates a different relation?</th>
            </tr>
          </thead>
          <tbody>
            {QUERIES.map((q, i) => (
              <tr key={i}>
                <td className="num" style={{ color: 'var(--ink-faint)' }}>{i + 1}</td>
                <td><code style={{ fontSize: 11.4 }}>{q}</code></td>
                <td className="num" style={{ color: 'var(--bad)' }}>no</td>
                <td className="num hide-sm"
                    style={{ color: i < 8 ? 'var(--ok)' : 'var(--ink-faint)' }}>
                  {i < 8 ? 'yes' : 'no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="callout" style={{ marginTop: 14 }}>
          <b>The last column is the load-bearing one.</b> The same battery run
          against a third graph with a genuinely different contact relation
          separates it on 8 of 13 queries. Without that control, the column of
          “no” would be consistent with a blind battery rather than with a
          property of these two systems.
        </div>
      </div>
    </div>
  )
}
