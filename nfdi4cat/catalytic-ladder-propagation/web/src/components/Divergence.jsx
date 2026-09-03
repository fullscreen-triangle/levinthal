import React, { useState } from 'react'
import { Stat } from './Primitives.jsx'

const FORM_A = `SELECT (COUNT(DISTINCT ?r) AS ?n) WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a , ?o .
  VALUES ?aa { chebi:35238 chebi:37022 }
  VALUES ?ox { chebi:35179 chebi:36147 chebi:133294 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}`

const FORM_B = `SELECT (COUNT(DISTINCT ?r) AS ?n) WHERE {
  ?r rdfs:subClassOf rh:Reaction ; rh:status rh:Approved .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?a .
  ?r rh:side/rh:contains/rh:compound/rh:chebi ?o .
  VALUES ?aa { chebi:35238 chebi:37022 }
  VALUES ?ox { chebi:35179 chebi:36147 chebi:133294 }
  ?a rdfs:subClassOf* ?aa .
  ?o rdfs:subClassOf* ?ox .
}`

export default function Divergence() {
  const [show, setShow] = useState('both')

  return (
    <div>
      <div className="grid g4" style={{ marginBottom: 18 }}>
        <div className="panel"><Stat value="2" label="Form A returned" color="var(--warn)" /></div>
        <div className="panel"><Stat value="397" label="Form B returned" color="var(--accent)" /></div>
        <div className="panel"><Stat value="2 / 2" label="local engines agree" color="var(--ok)" /></div>
        <div className="panel"><Stat value="217 544" label="classes at the endpoint" /></div>
      </div>

      <div className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <h3 style={{ margin: 0 }}>
            Two spellings the specification defines as one abstract query
          </h3>
          <div className="seg">
            <button className={show === 'both' ? 'on' : ''} onClick={() => setShow('both')}>
              side by side
            </button>
            <button className={show === 'diff' ? 'on' : ''} onClick={() => setShow('diff')}>
              just the difference
            </button>
          </div>
        </div>

        {show === 'diff' ? (
          <pre style={{ marginTop: 14 }}>
{`  Form A   ?r  rh:side/rh:contains/rh:compound/rh:chebi  ?a , ?o .

  Form B   ?r  rh:side/rh:contains/rh:compound/rh:chebi  ?a .
           ?r  rh:side/rh:contains/rh:compound/rh:chebi  ?o .`}
          </pre>
        ) : (
          <div className="grid g2" style={{ marginTop: 14 }}>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12,
                            color: 'var(--warn)', marginBottom: 6 }}>
                Form A — object list · returned 2 · 2.7 s
              </div>
              <pre style={{ fontSize: 11.4 }}>{FORM_A}</pre>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12,
                            color: 'var(--accent)', marginBottom: 6 }}>
                Form B — two triple patterns · returned 397 · 37.3 s
              </div>
              <pre style={{ fontSize: 11.4 }}>{FORM_B}</pre>
            </div>
          </div>
        )}

        <div className="callout" style={{ marginTop: 16 }}>
          <b>They must denote the same query.</b> SPARQL 1.1 §4.2.2 defines an
          object list as an <em>abbreviation</em>: <code>?s ?p ?o1 , ?o2</code>{' '}
          expands to <code>?s ?p ?o1 . ?s ?p ?o2 .</code>, and the expansion
          happens during translation to the abstract query — before any
          evaluation semantics is reached. Nothing makes it conditional on the
          predicate being a bare IRI.
        </div>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>The local control</h3>
          <p style={{ fontSize: 13.6 }}>
            Both spellings were run unmodified against a hand-checkable miniature
            of the same data shape — two reactions, one qualifying — on two
            independently developed engines.
          </p>
          <table>
            <thead>
              <tr><th>engine</th><th>Form A</th><th>Form B</th><th className="num">agree?</th></tr>
            </thead>
            <tbody>
              <tr><td>rdflib 7.6.0</td><td><code>[r1]</code></td><td><code>[r1]</code></td>
                  <td className="num" style={{ color: 'var(--ok)' }}>yes</td></tr>
              <tr><td>pyoxigraph 0.5.9</td><td><code>[r1]</code></td><td><code>[r1]</code></td>
                  <td className="num" style={{ color: 'var(--ok)' }}>yes</td></tr>
              <tr className="hl"><td>hand computation</td><td colSpan="2"><code>[r1]</code></td>
                  <td className="num" style={{ color: 'var(--ok)' }}>yes</td></tr>
            </tbody>
          </table>
          <p className="note">
            A further control confirms the miniature returns nothing when asked
            for an oxidant root present in no record — so the agreement above is
            not an artefact of a query matching everything.
          </p>
        </div>

        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Status of the finding</h3>
          <div className="callout">
            <b>Divergence: established.</b> Two spellings the specification
            declares equivalent, verified equivalent on two independent engines
            against a hand-checked miniature, return 2 and 397 against the same
            endpoint on the same date (3 September 2026).
          </div>
          <div className="callout warn">
            <b>Cause: not established.</b> Query planning, an internal traversal
            cap, and evaluation-order effects on the shared property path all
            remain live candidates. We did not investigate further, because doing
            so would mean probing a third party’s service.
          </div>
          <h3>A confound, reported against our own interest</h3>
          <p style={{ fontSize: 13.4 }}>
            The endpoint’s loaded ontology carried <b>217 544</b> classes on the
            date of measurement. Any count comparison between the endpoint and a
            separately published download is therefore confounded — the two are
            different snapshots. An investigator trying to adjudicate (2, 397) by
            reproducing the query against a download is not running a control;
            they are running a third route.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <h3 style={{ marginTop: 0 }}>Reading divergence as coverage</h3>
        <p style={{ fontSize: 13.8 }}>
          The natural conclusion is that something is broken. A different reading
          is available. If two routes share endpoints and converge to the same
          fixed point, neither can return an item the other excludes: their
          answers are nested, and the numerical difference measures how much of
          the underlying network each route resolved. On that reading (2, 397) is
          a <em>measurement of resolved extent</em>, and reporting it as “one
          query is broken” discards the quantity it carries.
        </p>
        <p style={{ fontSize: 13.8 }}>
          We are explicit that this reclassification is <b>conditional</b>: it
          requires both routes to converge to the same fixed point, which for
          this endpoint we could not verify from outside. Without that check the
          ordinary defect reading stands.
        </p>
        <h3>What follows, and costs nothing to adopt</h3>
        <ol style={{ fontSize: 13.6, lineHeight: 1.75, paddingLeft: 20 }}>
          <li>Record both answers. Keeping both costs one integer.</li>
          <li>Report the difference as coverage, not error. “Routes A and B
              differ by 395” is more informative than “route A failed”.</li>
          <li>Check the convergence hypothesis before reclassifying, or a genuine
              fault will be silently absorbed as a coverage difference.</li>
          <li>Keep at least one route strict enough that its disagreement is
              meaningful.</li>
          <li><b>Never compare counts across snapshots.</b> This one is
              unconditional.</li>
        </ol>
      </div>
    </div>
  )
}
