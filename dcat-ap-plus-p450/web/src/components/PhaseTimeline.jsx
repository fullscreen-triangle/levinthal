import { Link } from 'react-router-dom';
import { componentPath, countBy, layerById, nodes, phases, statuses, statusIds } from '../lib/spec.js';
import { StatusGlyph, Swatch } from './Glyphs.jsx';

export default function PhaseTimeline() {
  let cumulative = 0;
  return (
    <section aria-labelledby="phases-heading">
      <div className="section-head">
        <h2 id="phases-heading">Construction phases</h2>
        <p className="lede">
          Every later phase builds on the ones before it. The slider above replays this order in the graph.
        </p>
      </div>
      <ol className="timeline">
        {phases.map((p) => {
          const added = nodes.filter((n) => n.phase === p.id);
          cumulative += added.length;
          const byStatus = countBy(added, 'status', statusIds);
          return (
            <li key={p.id} className="timeline-item">
              <div className="timeline-marker mono" aria-hidden="true">
                {p.id}
              </div>
              <div className="timeline-body">
                <h3>
                  Phase {p.id} · {p.name}
                </h3>
                <p>{p.description}</p>
                <p className="timeline-counts mono">
                  +{added.length} components · {cumulative}/{nodes.length} cumulative
                  {statuses.map((s) =>
                    byStatus.get(s.id) ? (
                      <span key={s.id} className="inline-stat" title={s.description}>
                        <StatusGlyph status={s.id} size={11} /> {byStatus.get(s.id)} {s.label.toLowerCase()}
                      </span>
                    ) : null,
                  )}
                </p>
                <details>
                  <summary>Components added in phase {p.id}</summary>
                  <ul className="chip-list">
                    {added.map((n) => (
                      <li key={n.id}>
                        <Link to={componentPath(n.id)} className="chip">
                          <Swatch colour={layerById.get(n.layer)?.color} shape="dot" />
                          {n.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
