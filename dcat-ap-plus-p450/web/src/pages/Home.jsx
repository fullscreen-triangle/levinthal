import { useEffect } from 'react';
import GraphExplorer from '../components/GraphExplorer.jsx';
import PhaseTimeline from '../components/PhaseTimeline.jsx';
import LayerCards from '../components/LayerCards.jsx';
import ComponentTable from '../components/ComponentTable.jsx';
import { edges, layers, nodes, phases } from '../lib/spec.js';

// The library's four layers (LIBRARY-PLAN.md), each built from the ones before it.
const LIBRARY_LAYERS = ['papers', 'records', 'queries', 'model'];

export default function Home() {
  useEffect(() => {
    document.title = 'dcat-ap-plus-p450 — component graph';
  }, []);

  return (
    <>
      <header className="intro">
        <p className="eyebrow">Cytochrome P450 knowledge library</p>
        <h1>dcat-ap-plus-p450</h1>
        <p className="purpose">
          A DCAT-AP-PLUS metadata extension for cytochrome P450, a federated query layer (HFQ) over its records, and a
          domain model that writes plans and explains verdicts — hosted in Buhera OS.
        </p>
        <p className="library-layers" aria-label="The four layers of the library">
          {LIBRARY_LAYERS.map((l, i) => (
            <span key={l}>
              <span className="library-layer">{l}</span>
              {i < LIBRARY_LAYERS.length - 1 ? <span className="arrow"> → </span> : null}
            </span>
          ))}
          <span className="muted"> — each built from the ones before it; the model is never a source of values.</span>
        </p>
        <dl className="headline-stats">
          <div>
            <dt>components</dt>
            <dd className="mono">{nodes.length}</dd>
          </div>
          <div>
            <dt>connections</dt>
            <dd className="mono">{edges.length}</dd>
          </div>
          <div>
            <dt>layers</dt>
            <dd className="mono">{layers.length}</dd>
          </div>
          <div>
            <dt>phases</dt>
            <dd className="mono">{phases.length}</dd>
          </div>
        </dl>
      </header>

      <GraphExplorer />
      <PhaseTimeline />
      <LayerCards />
      <ComponentTable />
    </>
  );
}
