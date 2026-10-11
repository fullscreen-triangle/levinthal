import { Link } from 'react-router-dom';
import { componentPath, countBy, layers, nodesByLayer, statuses, statusIds } from '../lib/spec.js';
import { StatusGlyph } from './Glyphs.jsx';

export default function LayerCards() {
  return (
    <section aria-labelledby="layers-heading">
      <div className="section-head">
        <h2 id="layers-heading">Layers</h2>
        <p className="lede">Top to bottom, as in the graph: what is argued, down to where it runs.</p>
      </div>
      <div className="layer-grid">
        {[...layers].reverse().map((l) => {
          const members = nodesByLayer.get(l.id) ?? [];
          const byStatus = countBy(members, 'status', statusIds);
          return (
            <article key={l.id} className="layer-card" style={{ '--layer': l.color }}>
              <header>
                <h3>{l.name}</h3>
                <span className="mono muted">
                  {l.id} · order {l.order}
                </span>
              </header>
              <p>{l.description}</p>
              <div className="layer-stats">
                <span className="layer-count mono">
                  {members.length} component{members.length === 1 ? '' : 's'}
                </span>
                <div
                  className="status-bar"
                  role="img"
                  aria-label={statuses.map((s) => `${byStatus.get(s.id)} ${s.label}`).join(', ')}
                >
                  {statuses.map((s) =>
                    byStatus.get(s.id) ? (
                      <span
                        key={s.id}
                        className={`status-seg status-seg-${s.id}`}
                        style={{ flexGrow: byStatus.get(s.id) }}
                        title={`${byStatus.get(s.id)} ${s.label}`}
                      />
                    ) : null,
                  )}
                </div>
                <div className="status-breakdown mono">
                  {statuses.map((s) => (
                    <span key={s.id} className={byStatus.get(s.id) ? '' : 'muted'}>
                      <StatusGlyph status={s.id} size={11} /> {byStatus.get(s.id)} {s.label.toLowerCase()}
                    </span>
                  ))}
                </div>
              </div>
              <ul className="chip-list">
                {members.map((n) => (
                  <li key={n.id}>
                    <Link to={componentPath(n.id)} className="chip">
                      {n.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </article>
          );
        })}
      </div>
    </section>
  );
}
