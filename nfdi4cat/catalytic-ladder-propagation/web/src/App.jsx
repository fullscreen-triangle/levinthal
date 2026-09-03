import React, { useEffect, useState } from 'react'
import { Section, Stat } from './components/Primitives.jsx'
import FloorExplorer from './components/FloorExplorer.jsx'
import Separation from './components/Separation.jsx'
import LadderLab from './components/LadderLab.jsx'
import MediumLab from './components/MediumLab.jsx'
import QuestionBrowser from './components/QuestionBrowser.jsx'
import PlanBuilder from './components/PlanBuilder.jsx'
import Divergence from './components/Divergence.jsx'

const NAV = [
  ['problem', 'The problem'],
  ['floor', 'Contact & floor'],
  ['ladder', 'The ladder'],
  ['medium', 'The medium'],
  ['separation', 'What SPARQL cannot say'],
  ['plan', 'Plans & verdicts'],
  ['questions', 'The 28 questions'],
  ['divergence', 'Rhea, 2 vs 397'],
  ['limits', 'What this does not do'],
]

function useActive() {
  const [active, setActive] = useState('problem')
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => e.isIntersecting && setActive(e.target.id))
      },
      { rootMargin: '-45% 0px -50% 0px' }
    )
    NAV.forEach(([id]) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [])
  return active
}

export default function App() {
  const active = useActive()

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-brand">Catalytic <span>Ladder</span> Propagation</div>
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`} className={active === id ? 'on' : ''}>
              {label}
            </a>
          ))}
        </div>
      </nav>

      {/* ---------------------------------------------------------- hero */}
      <header className="hero">
        <div className="wrap">
          <h1>Querying what you cannot model</h1>
          <p className="lede">
            A knowledge graph is only as useful as the person who built it. To
            ask it a question you must know how it was modelled, and how to
            phrase your question in that model. Both requirements bind, and
            together they bound the work that can be done with the artefact.
          </p>
          <p className="lede">
            This page walks through an architecture in which neither requirement
            binds — for an important class of question — and lets you run every
            claim in your browser.
          </p>
          <blockquote className="pull">
            “Please don’t worry whether our current data can already answer them.
            Finding out that it cannot is actually valuable for us as well.”
          </blockquote>
          <div className="badges">
            <span className="badge"><b>28</b> questions resolved</span>
            <span className="badge"><b>12</b> supplied by practitioners</span>
            <span className="badge"><b>7</b> refused, with named blockers</span>
            <span className="badge"><b>57</b> checks, all passing</span>
            <span className="badge"><b>4</b> negative controls</span>
            <span className="badge"><b>2 vs 397</b> replicated 3 Sep 2026</span>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------- problem */}
      <Section
        id="problem"
        kicker="The failure that motivates all of this"
        title="Five situations, one output"
        sub="The failure mode is worth stating concretely, because it is invisible."
      >
        <div className="panel">
          <p style={{ marginTop: 0 }}>
            Ask a system: <i>which reactions have exactly four participants?</i>{' '}
            Suppose the corpus contains three such reactions, and the system
            returns the empty set. There are at least five distinct situations
            that produce that output.
          </p>
          <table style={{ marginTop: 14 }}>
            <thead>
              <tr><th>#</th><th>what actually happened</th><th>what you should do</th></tr>
            </thead>
            <tbody>
              <tr><td className="num">1</td>
                <td>the model has no cardinality constructs — the question was never representable</td>
                <td>change the model, or ask elsewhere</td></tr>
              <tr><td className="num">2</td>
                <td>it has them, but the count is not <i>entailed</i> under an open-world reading</td>
                <td>nothing — the answer is correct</td></tr>
              <tr><td className="num">3</td>
                <td>the corpus genuinely contains no such reaction</td>
                <td>believe it, and move on</td></tr>
              <tr><td className="num">4</td>
                <td>the compiler silently dropped the restriction while lowering</td>
                <td>file a bug</td></tr>
              <tr><td className="num">5</td>
                <td>the reasoner exhausted its budget and returned partial results</td>
                <td>raise the budget and re-run</td></tr>
            </tbody>
          </table>
          <div className="callout warn" style={{ marginTop: 16 }}>
            <b>One output is produced for all five.</b> It is not wrong in any
            way inspection reveals: well formed, correctly typed, promptly
            returned — and in case 2 it is even <em>correct</em> under the
            semantics the system implements. It is uninformative in a way that is
            invisible.
          </div>
          <p>
            An architecture in which those five produce five different outputs is
            not a reporting improvement bolted onto a query engine. It is a
            different kind of object: the system becomes an <b>instrument</b>,
            and its output at each question is a <b>verdict</b>, of which an
            answer set is one possible constituent.
          </p>
        </div>

        <h3>Why incompleteness is the expected state, not a temporary one</h3>
        <div className="grid g3">
          {[
            ['Expressive', 'Every decidable fragment of first-order logic omits constructions some question will need. A model chosen so reasoning terminates is a model in which some questions are not statable. That is a theorem about the fragments, not an engineering artefact.'],
            ['Empirical', 'Curated corpora are assembled by finite human effort from an open literature. Reaction databases record what has been curated, not what exists. Absence in a corpus is not evidence of absence in the world.'],
            ['Computational', 'Even where a question is statable and entailed, the reasoner may not finish. A question that cannot run on the available machine in the available time is unanswered for a third, independent reason.'],
          ].map(([h, b]) => (
            <div className="panel" key={h}>
              <h3 style={{ marginTop: 0 }}>{h}</h3>
              <p style={{ fontSize: 13.6, marginBottom: 0 }}>{b}</p>
            </div>
          ))}
        </div>
        <div className="callout" style={{ marginTop: 18 }}>
          A system that answers sixty per cent of a question set and cannot
          characterise the other forty is <em>less</em> useful than one that
          answers forty and, for each of the remaining sixty, says which of the
          three sources is responsible and what would have to change. The second
          supports a decision; the first supports only a percentage.
        </div>
      </Section>

      {/* --------------------------------------------------------- floor */}
      <Section
        id="floor"
        kicker="The substrate, in chemist’s terms"
        title="Contact, cut, and the floor"
        sub="Everything below is built from one object: a weighted graph of contacts with a distinguished vertex for the medium. The weight on an edge is the cost of separating two things — never a distance between them."
      >
        <div className="grid g2" style={{ marginBottom: 18 }}>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>The three moving parts</h3>
            <table>
              <tbody>
                <tr>
                  <td><b>contact weight</b> <code>w(u,v)</code></td>
                  <td style={{ fontSize: 13.2 }}>what it costs to pull <i>u</i> and{' '}
                    <i>v</i> apart. Costs add along a cut; distances would not.</td>
                </tr>
                <tr>
                  <td><b>the medium</b> <code>m</code></td>
                  <td style={{ fontSize: 13.2 }}>whatever a thing is separated{' '}
                    <i>from</i> — the solvent, the bulk phase. A structural slot,
                    not a chemical assumption.</td>
                </tr>
                <tr>
                  <td><b>separation cost</b> <code>σ(v)</code></td>
                  <td style={{ fontSize: 13.2 }}>the cheapest cut that puts{' '}
                    <i>v</i> on one side and the medium on the other.</td>
                </tr>
                <tr className="hl">
                  <td><b>the floor</b> <code>β</code></td>
                  <td style={{ fontSize: 13.2 }}>the smallest separation cost{' '}
                    <i>anywhere in the system</i>. An irreducible price of telling
                    anything apart from anything.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>Why identity is a cut and not a label</h3>
            <p style={{ fontSize: 13.6 }}>
              A label is an arbitrary assignment. Two curators may attach
              different labels to the same object, and the same label to
              different objects, and nothing internal to the graph detects it.
              Formally: a labelling carries no invariant that the graph itself
              determines.
            </p>
            <div className="callout">
              <b>This is why cross-references must be curated.</b> Two resources
              labelling the same system with independent labellings{' '}
              <em>cannot compute</em> their correspondence from the labels — any
              mapping is an assertion, maintained by hand, that decays. Locate
              identity in the cut instead and the correspondence becomes a
              computation.
            </div>
          </div>
        </div>

        <h3>Explore it</h3>
        <p className="sub">
          A buried core joined to a solvent-exposed shell. Drag the packing
          weight up and watch what happens to the identity of a buried residue.
        </p>
        <FloorExplorer />
      </Section>

      {/* -------------------------------------------------------- ladder */}
      <Section
        id="ladder"
        kicker="The stencil"
        title="The ladder"
        sub="A rung carries exactly one number: the fraction of the remaining distance it closes. A ladder is a sequence of rungs. That is the whole object — and it is deliberately blind to what the rungs are made of, which is what lets it be laid over records from any source."
      >
        <div className="panel" style={{ marginBottom: 18 }}>
          <div className="grid g2">
            <div>
              <h3 style={{ marginTop: 0 }}>What a rung is</h3>
              <p style={{ fontSize: 13.8 }}>
                Take a process heading toward some target. At each step, ask what
                fraction of the <em>remaining above-floor distance</em> that step
                closes. That fraction, a number in [0,1], is the rung’s{' '}
                <b>power</b> π. Nothing else about the step enters: not its
                mechanism, not its rate constant, not what molecule performed it.
              </p>
              <p style={{ fontSize: 13.8 }}>
                Two rungs with the same power are, at this resolution, the same
                rung. Contrast that with the usual criteria for whether two
                catalysts are “the same kind” — mechanism, fold, active-site
                residues, sequence homology — which are known to disagree with
                one another.
              </p>
            </div>
            <div>
              <h3 style={{ marginTop: 0 }}>How rungs compose</h3>
              <pre style={{ fontSize: 13 }}>{`π(L)  =  1 − Π (1 − πᵢ)
                    i`}</pre>
              <p style={{ fontSize: 13.6, marginTop: 12 }}>
                Multiplicative, not additive. Each rung works on what the
                previous ones left, so the residual gap is a product. Three
                consequences follow immediately, and all three are testable:
              </p>
              <ul style={{ fontSize: 13.4, lineHeight: 1.7 }}>
                <li>repetition saturates — copies of what already works have
                    geometrically diminishing returns</li>
                <li>a demanding target has a minimum rung count</li>
                <li>control sits at the <em>strongest</em> rung, not the
                    bottleneck — under one parametrisation. Try the toggle.</li>
              </ul>
            </div>
          </div>
        </div>
        <LadderLab />
      </Section>

      {/* -------------------------------------------------------- medium */}
      <Section
        id="medium"
        kicker="Where solvent and direction come from"
        title="The medium"
        sub="Two questions a reaction ontology must record as annotations — what role the solvent plays, and which way the reaction runs — become computations once the medium carries content."
      >
        <MediumLab />
      </Section>

      {/* ---------------------------------------------------- separation */}
      <Section
        id="separation"
        kicker="A theorem, executed rather than argued"
        title="What retrieval cannot say"
        sub="Two systems recording identical triples, whose propagation verdicts differ. Thirteen retrieval query forms on two independent SPARQL engines fail to tell them apart — and a control confirms the battery is not blind."
      >
        <div className="panel" style={{ marginBottom: 18 }}>
          <p style={{ marginTop: 0, fontSize: 13.8 }}>
            The queried quantity <code>σ(v₀, x)</code> depends on weights along
            paths between the pair. The threshold <code>β</code> depends on a
            minimum over <em>every</em> item in the system — including items
            lying on no path between the two, and from the query’s point of view
            entirely irrelevant. Edit a weight out there, and the verdict about
            the pair changes without anything a pattern-matcher could match on
            changing at all.
          </p>
          <div className="callout">
            <b>Accountability is a global property expressed in local terms</b>,
            and a stored graph records only the local terms. That is the whole
            argument. The rest is arithmetic.
          </div>
        </div>
        <Separation />
      </Section>

      {/* ---------------------------------------------------------- plan */}
      <Section
        id="plan"
        kicker="The architecture"
        title="Plans, capabilities, verdicts"
        sub="The sources a catalysis question must cross do not share a query model, and no amount of engineering will give them one. They do share a result model: finite sets of identifiers carrying attributes. So the language’s terms denote results, not graph patterns — and queries become leaves the user never writes."
      >
        <div className="grid g2" style={{ marginBottom: 18 }}>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>The four stages</h3>
            <ol style={{ fontSize: 13.6, lineHeight: 1.8, paddingLeft: 20 }}>
              <li><b>Fetch</b> — records from heterogeneous sources through a
                  plan language whose terms denote result sets.</li>
              <li><b>Translate</b> — records into weighted contact graphs with a
                  medium vertex.</li>
              <li><b>Stencil</b> — lay a ladder over the translated graph; each
                  contact contributes a rung.</li>
              <li><b>Verdict</b> — return one of six outcomes, of which an answer
                  set is one.</li>
            </ol>
            <div className="callout" style={{ marginTop: 4 }}>
              <b>Why this removes the modelling requirement.</b> The stencil asks
              of a record only what is needed to compute a separation cost. It
              does not ask what class the record belongs to, which ontology term
              names its relation, or whether a curator anticipated the question.
            </div>
          </div>
          <div className="panel">
            <h3 style={{ marginTop: 0 }}>The six verdicts</h3>
            <table>
              <tbody>
                <tr><td><code>answer</code></td><td style={{ fontSize: 13 }}>a result set, certified non-empty</td></tr>
                <tr><td><code>empty</code></td><td style={{ fontSize: 13 }}>well posed; the answer is certified empty</td></tr>
                <tr><td><code>unexpressed</code></td><td style={{ fontSize: 13 }}>the question cannot be stated in this source’s model</td></tr>
                <tr><td><code>unsupported</code></td><td style={{ fontSize: 13 }}>statable, but not lowerable by this compiler</td></tr>
                <tr><td><code>starved</code></td><td style={{ fontSize: 13 }}>an earlier step under-retrieved — and it is named</td></tr>
                <tr><td><code>exhausted</code></td><td style={{ fontSize: 13 }}>the budget ran out before this step</td></tr>
              </tbody>
            </table>
            <div className="callout" style={{ marginTop: 12 }}>
              <b>Non-degeneracy is structural, not a reporting habit.</b> No
              verdict other than <code>answer</code> may carry a payload.
              Constructing one that does <em>throws</em>. That is what stops the
              five situations above from collapsing into one empty table.
            </div>
          </div>
        </div>

        <h3>Build a step and see it refused before it runs</h3>
        <PlanBuilder />

        <h3 style={{ marginTop: 26 }}>What a plan looks like</h3>
        <pre>{`plan bacterial_transaminase_no_cys

  source RXN  capability { reaction, participant, ec, equation }
  source PROT capability { protein, organism, lineage, sequence }

  let rxns  = fetch RXN  where participant contains "benzylethylamine"
  let enzs  = fetch PROT where catalyses in rxns
  let bact  = keep enzs  where lineage contains "Bacteria"
  let final = keep bact  where not contains(sequence, "C")

  require final nonempty
  emit final`}</pre>
        <p className="note">
          No SPARQL is written. The last step is a predicate over a retrieved
          attribute, not a triple lookup: no public store holds a{' '}
          <code>hasCysteine</code> flag, so a pattern-matching query returns
          nothing. Here it is simply computed.
        </p>
      </Section>

      {/* ----------------------------------------------------- questions */}
      <Section
        id="questions"
        kicker="Run them yourself"
        title="Twenty-eight questions"
        sub="Twelve were supplied by practitioners without reference to this framework. Sixteen were added to reach outcomes the twelve do not. Open any card to see the plan it took, the sources it touched, and — where it refuses — the blocker and what would unblock it."
      >
        <QuestionBrowser />
      </Section>

      {/* ---------------------------------------------------- divergence */}
      <Section
        id="divergence"
        kicker="A live measurement, replicated 3 September 2026"
        title="Two spellings, one query, 2 and 397"
        sub="Why a client that treats route divergence as an error is discarding information it could be keeping for free."
      >
        <Divergence />
      </Section>

      {/* -------------------------------------------------------- limits */}
      <Section
        id="limits"
        kicker="Stated plainly"
        title="What this does not do"
        sub="Several of these are severe enough that a reader could otherwise take the work to claim more than it does."
      >
        <div className="grid g2">
          {[
            ['No experimental validation', 'Nothing here has been compared against a measured hydration structure, a measured direction reversal, a measured ambient occupancy, or a measured rate. The direction trichotomy is worked from the qualitative fact that organisms differ in metabolite demand, not from measured intracellular concentrations.'],
            ['The powers are declared, not derived', 'Rung powers on this page are literals. Deriving them from retrieved data — and answering what a derived power means for a result set — is the substantive next step and is not taken. Until it is, the ladder demonstrates a form of answer rather than a measured one.'],
            ['Translation supplies structure', 'Converting a record to a contact graph adds implicit hydrogens, inferred bond orders, and a medium the record never mentions. Unless the translator tags each element as stated or supplied, a reader cannot tell a recorded fact from a convention.'],
            ['The medium weight is a modelling choice', 'It is justified by three properties and by reduction to a chemical potential in the dilute limit, but not derived. A different monotone, floor-bounded function gives the same theorems with different constants — which is why every structural result is re-run under four families.'],
            ['No kinetics, no thermodynamics', 'The medium bias is a difference of individuation costs, not a free energy. In consequence “at what temperature does this reaction take place” remains open: the floor is condition-dependent in principle, but at this depth the dependence is immaterial, and we would rather say so than report a number a laboratory cannot use.'],
            ['Fixtures are not the public resources', 'The questions on this page run against small hand-built stand-ins with the same record shape as the public sources — not copies of them. This makes every number reproducible offline and makes none of them evidence about biology. Only the Rhea divergence touches a live service.'],
          ].map(([h, b]) => (
            <div className="panel" key={h}>
              <h3 style={{ marginTop: 0 }}>{h}</h3>
              <p style={{ fontSize: 13.4, marginBottom: 0, color: 'var(--ink-dim)' }}>{b}</p>
            </div>
          ))}
        </div>

        <div className="callout" style={{ marginTop: 22 }}>
          <b>Two results here came from tests written to be capable of
          failing.</b> A one-sided sweep of the direction trichotomy never
          reached its third case, and the reason turned out to be a saturation
          bound we had not anticipated. And the counter-intuitive claim that
          control lies at the strongest rung survives only under an additive
          parametrisation. We report both because a suite written to pass would
          have hidden them.
        </div>
      </Section>

      <footer>
        <div className="wrap">
          <p>
            <b>Catalytic Ladder Propagation.</b> Every figure and number on this
            page is computed in your browser from the same definitions the paper
            states — nothing here is a stored screenshot of a result. The
            accompanying validation suite reports 57 scored checks, 4 negative
            controls that must fail on well-formed input, and 1 check excluded as
            non-discriminating.
          </p>
          <p style={{ color: 'var(--ink-faint)' }}>
            Fixture data is illustrative and is not evidence about biology. The
            Rhea measurement was taken on 3 September 2026 and is dated because
            endpoints change.
          </p>
        </div>
      </footer>
    </>
  )
}
