import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { componentPath, degree, layerById, nodes, phaseById, statusById, statusIds } from '../lib/spec.js';
import { StatusTag, Swatch } from './Glyphs.jsx';

const statusRank = new Map(statusIds.map((s, i) => [s, i]));

const COLUMNS = [
  { key: 'label', label: 'Component', value: (n) => n.label.toLowerCase() },
  { key: 'id', label: 'Id', value: (n) => n.id },
  { key: 'layer', label: 'Layer', value: (n) => layerById.get(n.layer)?.order ?? -1 },
  { key: 'phase', label: 'Phase', value: (n) => n.phase },
  { key: 'status', label: 'Status', value: (n) => statusRank.get(n.status) ?? 99 },
  { key: 'degree', label: 'In / out', value: (n) => degree.get(n.id).total },
];

export default function ComponentTable() {
  const [sort, setSort] = useState({ key: 'layer', dir: 'desc' });
  const [query, setQuery] = useState('');

  const rows = useMemo(() => {
    const col = COLUMNS.find((c) => c.key === sort.key);
    const q = query.trim().toLowerCase();
    const filtered = q
      ? nodes.filter((n) =>
          [n.id, n.label, n.summary, n.repo, n.path, layerById.get(n.layer)?.name].some((f) =>
            String(f ?? '').toLowerCase().includes(q),
          ),
        )
      : nodes;
    const sign = sort.dir === 'asc' ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const va = col.value(a);
      const vb = col.value(b);
      if (va < vb) return -sign;
      if (va > vb) return sign;
      return a.label.localeCompare(b.label);
    });
  }, [sort, query]);

  const toggleSort = (key) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' }));

  return (
    <section id="components" aria-labelledby="components-heading">
      <div className="section-head">
        <h2 id="components-heading">All components</h2>
        <p className="lede">
          The same graph as a table: sortable by any column, and the way in for keyboards, screen readers and small
          screens.
        </p>
      </div>
      <div className="table-tools">
        <label htmlFor="component-filter" className="control-label">
          Filter
        </label>
        <input
          id="component-filter"
          type="search"
          value={query}
          placeholder="id, name, repo, path, summary…"
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="muted mono" aria-live="polite">
          {rows.length} of {nodes.length}
        </span>
      </div>
      <div className="table-scroll">
        <table className="component-table">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={sort.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button type="button" className="sort-btn" onClick={() => toggleSort(c.key)}>
                    {c.label}
                    <span className="sort-mark" aria-hidden="true">
                      {sort.key === c.key ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((n) => {
              const layer = layerById.get(n.layer);
              const d = degree.get(n.id);
              return (
                <tr key={n.id}>
                  <th scope="row">
                    <Link to={componentPath(n.id)}>{n.label}</Link>
                  </th>
                  <td className="mono">{n.id}</td>
                  <td>
                    <span className="layer-cell">
                      <Swatch colour={layer?.color} shape="dot" />
                      {layer?.name}
                    </span>
                  </td>
                  <td title={phaseById.get(n.phase)?.name}>
                    <span className="mono">{n.phase}</span> <span className="muted">{phaseById.get(n.phase)?.name}</span>
                  </td>
                  <td title={statusById.get(n.status)?.description}>
                    <StatusTag status={n.status} />
                  </td>
                  <td className="mono num">
                    {d.in} / {d.out}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
