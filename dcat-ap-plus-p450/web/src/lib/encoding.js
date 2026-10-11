// Visual encodings that the JSON does not carry: edge-kind colours, status
// shapes, node size. Layer colours come from the JSON itself.
import { edgeKinds, layerById, phaseById, statusById, degree } from './spec.js';

// Edge colour by kind. Fixed per kind id, never by rank, so a filter never
// repaints the survivors. The eight most frequent kinds take the eight slots
// of a categorical palette validated for dark surfaces; the four rarest take
// a neutral and three light tints. Twelve kinds exceed what hue alone can
// separate, so identity is also carried by the hover label, the legend that
// isolates a kind, and the grouped lists on each component page.
const EDGE_KIND_COLOURS = {
  feeds: '#3987e5',
  reuses: '#d95926',
  emits: '#199e70',
  documents: '#c98500',
  snapshots: '#d55181',
  dispatches: '#2fa82f',
  queries: '#9085e9',
  analyses: '#e66767',
  'runs-on': '#8b8a83',
  trains: '#f2a7c6',
  deploys: '#7fd1e8',
  conforms: '#e6e3d6',
};
const FALLBACK_EDGE_COLOUR = '#a7a59c';

export const edgeColour = (kindId) => EDGE_KIND_COLOURS[kindId] ?? FALLBACK_EDGE_COLOUR;

// Kinds the JSON adds later still render (in the fallback colour) — but say so.
export const unpalettedKinds = edgeKinds.filter((k) => !(k.id in EDGE_KIND_COLOURS)).map((k) => k.id);

export const layerColour = (layerId) => layerById.get(layerId)?.color ?? '#888888';

// Status as shape/material, so it never competes with layer colour.
export const STATUS_ENCODING = {
  exists: { shape: 'Solid sphere', note: 'fully opaque, lit' },
  adapt: { shape: 'Translucent sphere with a ring', note: 'reduced opacity and glow' },
  new: { shape: 'Wireframe sphere', note: 'outline only: not built yet' },
  external: { shape: 'Octahedron', note: 'outside the user’s repositories' },
};

// Node area ∝ degree: radius ∝ √degree. nodeVal = r³ with nodeRelSize 1, so
// the force graph places arrowheads on the surface of each shape.
export const NODE_SCALE = 1.9;
export const nodeRadius = (id) => NODE_SCALE * Math.sqrt(Math.max(1, degree.get(id)?.total ?? 1));

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESCAPES[c]);

export function nodeTooltipHtml(node) {
  const layer = layerById.get(node.layer);
  const phase = phaseById.get(node.phase);
  const status = statusById.get(node.status);
  const d = degree.get(node.id);
  return `<div class="gtip">
    <div class="gtip-title">${escapeHtml(node.label)}</div>
    <div class="gtip-id">${escapeHtml(node.id)}</div>
    <div class="gtip-meta"><span class="gtip-swatch" style="background:${escapeHtml(layer?.color)}"></span>${escapeHtml(layer?.name)}
      · phase ${escapeHtml(node.phase)} ${escapeHtml(phase?.name)} · ${escapeHtml(status?.label)} · ${d.in} in / ${d.out} out</div>
    <p>${escapeHtml(node.summary)}</p>
    <div class="gtip-hint">Click to open the component page</div>
  </div>`;
}
