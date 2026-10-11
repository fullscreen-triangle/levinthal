// The one place the site reads the component graph. The file lives outside
// web/ and is imported in place: it is never copied, and every number the site
// shows is derived from it here at runtime.
import spec from '../../../docs/specifications/components.json';

export { spec };

const byId = (list) => new Map(list.map((item) => [item.id, item]));

/** Layers, bottom (compute) to top (papers). */
export const layers = [...spec.layers].sort((a, b) => a.order - b.order);
export const phases = [...spec.phases].sort((a, b) => a.id - b.id);
export const statuses = spec.statuses;
export const edgeKinds = spec.edgeKinds;
export const nodes = spec.nodes;
export const edges = spec.edges;

export const layerById = byId(layers);
export const phaseById = byId(phases);
export const statusById = byId(statuses);
export const edgeKindById = byId(edgeKinds);
export const nodeById = byId(nodes);

export const minPhase = phases[0].id;
export const maxPhase = phases[phases.length - 1].id;
export const maxLayerOrder = layers[layers.length - 1].order;

/** id -> { in: Edge[], out: Edge[] } */
export const adjacency = new Map(nodes.map((n) => [n.id, { in: [], out: [] }]));
for (const e of edges) {
  adjacency.get(e.source)?.out.push(e);
  adjacency.get(e.target)?.in.push(e);
}

/** id -> { in, out, total } edge counts over the whole graph. */
export const degree = new Map(
  [...adjacency].map(([id, a]) => [id, { in: a.in.length, out: a.out.length, total: a.in.length + a.out.length }]),
);

export const maxDegree = Math.max(...[...degree.values()].map((d) => d.total));

/** Count items by key, returning a Map seeded with every allowed value (so zeros show). */
export function countBy(items, key, allowed) {
  const counts = new Map((allowed ?? []).map((v) => [v, 0]));
  for (const item of items) {
    const v = typeof key === 'function' ? key(item) : item[key];
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  return counts;
}

export const statusIds = statuses.map((s) => s.id);
export const nodesByLayer = new Map(layers.map((l) => [l.id, nodes.filter((n) => n.layer === l.id)]));
export const edgeCountByKind = countBy(edges, 'kind', edgeKinds.map((k) => k.id));

/** Group edges by kind in the order edgeKinds[] declares. */
export function groupByKind(list) {
  return edgeKinds
    .map((kind) => ({ kind, edges: list.filter((e) => e.kind === kind.id) }))
    .filter((g) => g.edges.length > 0);
}

export const componentPath = (id) => `/component/${encodeURIComponent(id)}`;
