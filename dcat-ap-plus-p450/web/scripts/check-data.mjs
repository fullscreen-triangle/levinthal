// Integrity check for the single source of truth the site renders.
// Usage: npm run check   (exits 1 on any failure)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const file = path.resolve(here, '../../docs/specifications/components.json');
const spec = JSON.parse(readFileSync(file, 'utf8'));

const failures = [];
let warnings = [];
const fail = (msg) => failures.push(msg);

function uniqueIds(list, name, key = 'id') {
  const seen = new Set();
  for (const item of list) {
    if (seen.has(item[key])) fail(`duplicate ${name} id: ${item[key]}`);
    seen.add(item[key]);
  }
  return seen;
}

for (const key of ['statuses', 'phases', 'layers', 'edgeKinds', 'nodes', 'edges']) {
  if (!Array.isArray(spec[key])) fail(`missing array: ${key}`);
}
if (failures.length) report();

const statuses = uniqueIds(spec.statuses, 'status');
const phases = uniqueIds(spec.phases, 'phase');
const layers = uniqueIds(spec.layers, 'layer');
const kinds = uniqueIds(spec.edgeKinds, 'edge kind');
const nodes = uniqueIds(spec.nodes, 'node');

const orders = new Set();
for (const l of spec.layers) {
  if (!Number.isInteger(l.order)) fail(`layer ${l.id}: order is not an integer`);
  if (orders.has(l.order)) fail(`layer ${l.id}: duplicate order ${l.order}`);
  orders.add(l.order);
  if (!/^#[0-9a-fA-F]{6}$/.test(l.color ?? '')) fail(`layer ${l.id}: color is not #rrggbb`);
}

for (const n of spec.nodes) {
  for (const field of ['id', 'label', 'summary']) {
    if (typeof n[field] !== 'string' || n[field] === '') fail(`node ${n.id}: empty ${field}`);
  }
  if (!layers.has(n.layer)) fail(`node ${n.id}: unknown layer "${n.layer}"`);
  if (!phases.has(n.phase)) fail(`node ${n.id}: unknown phase ${JSON.stringify(n.phase)}`);
  if (!statuses.has(n.status)) fail(`node ${n.id}: unknown status "${n.status}"`);
}

const pairs = new Set();
const degree = new Map(spec.nodes.map((n) => [n.id, 0]));
spec.edges.forEach((e, i) => {
  const where = `edge #${i} ${e.source} -> ${e.target}`;
  if (!nodes.has(e.source)) fail(`${where}: source does not resolve`);
  if (!nodes.has(e.target)) fail(`${where}: target does not resolve`);
  if (!kinds.has(e.kind)) fail(`${where}: unknown kind "${e.kind}"`);
  if (e.source === e.target) fail(`${where}: self loop`);
  const key = `${e.source}|${e.target}|${e.kind}`;
  if (pairs.has(key)) fail(`${where}: duplicate edge of kind ${e.kind}`);
  pairs.add(key);
  if (degree.has(e.source)) degree.set(e.source, degree.get(e.source) + 1);
  if (degree.has(e.target)) degree.set(e.target, degree.get(e.target) + 1);
});

const isolated = [...degree].filter(([, d]) => d === 0).map(([id]) => id);
warnings = isolated.length ? [`isolated nodes (degree 0): ${isolated.join(', ')}`] : [];

report();

function report() {
  const count = (key) => (Array.isArray(spec[key]) ? spec[key].length : '?');
  console.log(`components.json v${spec.version} (${spec.updated})`);
  console.log(
    `  ${count('nodes')} nodes, ${count('edges')} edges, ${count('layers')} layers, ` +
      `${count('phases')} phases, ${count('statuses')} statuses, ${count('edgeKinds')} edge kinds`,
  );
  for (const w of warnings) console.log(`  warning: ${w}`);
  if (failures.length) {
    for (const f of failures) console.error(`  FAIL ${f}`);
    console.error(`${failures.length} failure(s).`);
    process.exit(1);
  }
  console.log('  OK: every edge endpoint resolves; every layer, phase, status and edge kind exists.');
}
