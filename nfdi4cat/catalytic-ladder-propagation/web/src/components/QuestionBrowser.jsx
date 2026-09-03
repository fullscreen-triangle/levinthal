import React, { useMemo, useState } from 'react'
import { GROUPS, QUESTIONS } from '../lib/questions.js'
import { SOURCES } from '../lib/corpus.js'
import { Pill, Stat } from './Primitives.jsx'

function QCard({ q, open, onToggle }) {
  const r = useMemo(() => q.run(), [q])
  const refused = r.verdict !== 'answer'

  return (
    <div className="qcard">
      <div className="qhead" onClick={onToggle}>
        <div className="qid">{q.id}</div>
        <div className="qtext">
          {q.text}
          <div style={{ marginTop: 7, display: 'flex', gap: 7, alignItems: 'center',
                        flexWrap: 'wrap' }}>
            <Pill verdict={r.verdict} />
            {q.sources.map((s) => (
              <span key={s} className="badge" style={{ fontSize: 10.5, padding: '2px 7px' }}>
                {s}
              </span>
            ))}
            {q.origin === 'practitioner' && (
              <span style={{ fontSize: 11, color: 'var(--ink-faint)' }}>
                supplied by a practitioner
              </span>
            )}
            {q.highlight && (
              <span style={{ fontSize: 11, color: 'var(--accent-2)' }}>
                ★ not modellable — computed
              </span>
            )}
          </div>
        </div>
        <div style={{ color: 'var(--ink-faint)', fontSize: 17, lineHeight: 1 }}>
          {open ? '−' : '+'}
        </div>
      </div>

      {open && (
        <div className="qbody">
          {q.why && <div className="qwhy">{q.why}</div>}

          <div className="steps">
            <div style={{ fontSize: 11.5, textTransform: 'uppercase',
                          letterSpacing: '.07em', color: 'var(--ink-faint)',
                          marginBottom: 6 }}>
              plan steps
            </div>
            {r.steps.map((s, i) => (
              <div className="step" key={i}>
                <span className="src">{s.src}</span>
                <span className="op">{s.op}</span>
                <span className="n">{s.n}</span>
              </div>
            ))}
          </div>

          {!refused && (
            <>
              <div style={{ fontSize: 11.5, textTransform: 'uppercase',
                            letterSpacing: '.07em', color: 'var(--ink-faint)' }}>
                result
              </div>
              <ul className="payload">
                {r.payload.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </>
          )}

          {refused && (
            <>
              <div className="blocker">
                <b>blocker.</b> {r.blocker}
                {r.blame && (
                  <> The blame walk terminates at <code>{r.blame}</code>, not at a
                  step that answered correctly.</>
                )}
              </div>
              <div className="unblock">
                <b>what would unblock it.</b> {r.unblock}
              </div>
              <p className="note">
                Note that the payload is empty and <em>structurally</em> so: only{' '}
                <code>answer</code> may carry a result. Constructing a refusal
                that carries one throws. That is what stops five different
                situations from producing one indistinguishable empty table.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default function QuestionBrowser() {
  const [filter, setFilter] = useState('all')
  const [open, setOpen] = useState({ Q1: true })

  const results = useMemo(
    () => QUESTIONS.map((q) => ({ q, r: q.run() })), []
  )
  const dist = useMemo(() => {
    const d = {}
    results.forEach(({ r }) => { d[r.verdict] = (d[r.verdict] || 0) + 1 })
    return d
  }, [results])

  const shown = results.filter(({ q, r }) => {
    if (filter === 'all') return true
    if (filter === 'refused') return r.verdict !== 'answer'
    if (filter === 'practitioner') return q.origin === 'practitioner'
    return q.group === filter
  })

  const answered = dist.answer || 0
  const refused = QUESTIONS.length - answered

  return (
    <div>
      <div className="grid g4" style={{ marginBottom: 18 }}>
        <div className="panel"><Stat value={QUESTIONS.length} label="questions resolved" /></div>
        <div className="panel"><Stat value={answered} label="answered" color="var(--ok)" /></div>
        <div className="panel"><Stat value={refused} label="refused, with reasons" color="var(--warn)" /></div>
        <div className="panel"><Stat value={Object.keys(dist).length} label="distinct verdicts" /></div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="seg">
          <button className={filter === 'all' ? 'on' : ''} onClick={() => setFilter('all')}>
            all {QUESTIONS.length}
          </button>
          <button className={filter === 'practitioner' ? 'on' : ''}
                  onClick={() => setFilter('practitioner')}>
            practitioner-supplied 12
          </button>
          <button className={filter === 'refused' ? 'on' : ''}
                  onClick={() => setFilter('refused')}>
            refused {refused}
          </button>
          {GROUPS.map((g) => (
            <button key={g} className={filter === g ? 'on' : ''}
                    onClick={() => setFilter(g)} style={{ fontSize: 12 }}>
              {g}
            </button>
          ))}
        </div>
        <p className="note" style={{ marginTop: 12 }}>
          Every question below is executed in your browser against the fixture
          corpus when the card is opened — the results are computed, not stored.
          A system that answered all {QUESTIONS.length} would have an empty
          defined class: it would not classify, it would merely accept.
        </p>
      </div>

      {shown.map(({ q }) => (
        <QCard key={q.id} q={q} open={!!open[q.id]}
               onToggle={() => setOpen({ ...open, [q.id]: !open[q.id] })} />
      ))}
    </div>
  )
}
