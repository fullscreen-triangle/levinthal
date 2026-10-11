import { Link } from 'react-router-dom';
import { componentPath, groupByKind, layerById, nodeById, nodesByLayer } from '../lib/spec.js';
import { STATUS_ENCODING, edgeColour } from '../lib/encoding.js';
import { StatusGlyph, Swatch } from '../components/Glyphs.jsx';
import Neighbourhood from '../components/Neighbourhood.jsx';

function EdgeGroups({ title, edges, direction }) {
  const groups = groupByKind(edges);
  return (
    <section className="edge-section" aria-labelledby={`edges-${direction}`}>
      <h2 id={`edges-${direction}`}>
        {title} <span className="muted mono">{edges.length}</span>
      </h2>
      {groups.length === 0 ? (
        <p className="muted">None.</p>
      ) : (
        groups.map(({ kind, edges: list }) => (
          <div key={kind.id} className="edge-group">
            <h3>
              <Swatch colour={edgeColour(kind.id)} shape="line" />
              {kind.label} <span className="muted mono">{list.length}</span>
            </h3>
            <p className="edge-kind-desc muted">{kind.description}</p>
            <ul className="neighbour-list">
              {list.map((e, i) => {
                const otherId = direction === 'in' ? e.source : e.target;
                const other = nodeById.get(otherId);
                const layer = layerById.get(other?.layer);
                return (
                  <li key={`${otherId}-${i}`}>
                    <Swatch colour={layer?.color} shape="dot" title={layer?.name} />
                    <Link to={componentPath(otherId)}>{other?.label ?? otherId}</Link>
                    <span className="mono muted"> {otherId}</span>
                    {e.label ? <span className="edge-label">“{e.label}”</span> : null}
                  </li>
                );
              })}
            </ul>
          </div>
        ))
      )}
    </section>
  );
}

export default function GenericComponentPage({ node, layer, phase, status, incoming, outgoing }) {
  const siblings = (nodesByLayer.get(node.layer) ?? []).filter((n) => n.id !== node.id);
  return (
    <article className="component-page" style={{ '--layer': layer?.color }}>
      <nav className="crumbs" aria-label="Breadcrumb">
        <Link to="/">Component graph</Link> <span aria-hidden="true">/</span> <span>{layer?.name}</span>
      </nav>

      <header className="component-head">
        <p className="eyebrow">
          <Swatch colour={layer?.color} shape="dot" /> {layer?.name}
        </p>
        <h1>{node.label}</h1>
        <p className="component-id mono">{node.id}</p>
      </header>

      <p className="component-summary">{node.summary}</p>

      <dl className="facts">
        <div>
          <dt>Layer</dt>
          <dd>
            <Swatch colour={layer?.color} shape="dot" /> {layer?.name} <span className="muted mono">({layer?.id})</span>
            <div className="fact-note muted">{layer?.description}</div>
          </dd>
        </div>
        <div>
          <dt>Phase</dt>
          <dd>
            <span className="mono">{phase?.id}</span> · {phase?.name}
            <div className="fact-note muted">{phase?.description}</div>
          </dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>
            <StatusGlyph status={node.status} size={14} /> {status?.label}
            <span className="muted"> — drawn as {STATUS_ENCODING[node.status]?.shape?.toLowerCase() ?? 'a cube'}</span>
            <div className="fact-note muted">{status?.description}</div>
          </dd>
        </div>
        <div>
          <dt>Repository</dt>
          <dd className="mono">{node.repo || '—'}</dd>
        </div>
        <div>
          <dt>Path</dt>
          <dd className="mono path">{node.path || '— (none given)'}</dd>
        </div>
        <div>
          <dt>Connections</dt>
          <dd className="mono">
            {incoming.length} in · {outgoing.length} out
          </dd>
        </div>
        <div>
          <dt>Specification</dt>
          <dd>
            components-graph.md § <span className="mono">{node.id}</span>
          </dd>
        </div>
      </dl>

      {incoming.length + outgoing.length > 0 ? (
        <Neighbourhood node={node} incoming={incoming} outgoing={outgoing} />
      ) : null}

      <div className="edge-columns">
        <EdgeGroups title="Incoming" edges={incoming} direction="in" />
        <EdgeGroups title="Outgoing" edges={outgoing} direction="out" />
      </div>

      {siblings.length > 0 ? (
        <section className="siblings" aria-labelledby="siblings-heading">
          <h2 id="siblings-heading">Elsewhere in {layer?.name}</h2>
          <ul className="chip-list">
            {siblings.map((n) => (
              <li key={n.id}>
                <Link to={componentPath(n.id)} className="chip">
                  <StatusGlyph status={n.status} size={11} />
                  {n.label}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <p className="back-link">
        <Link to="/">← Back to the component graph</Link>
      </p>
    </article>
  );
}
