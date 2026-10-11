// Small numeric helpers used by the talk's live figures.

export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Maximum flow (Edmonds-Karp) on an undirected weighted graph.
// edges: [{a, b, w}] with node ids; returns {value, cut: Set of edge indices}.
export function maxFlow(nodes, edges, s, t) {
  const id = new Map(nodes.map((n, i) => [n, i]));
  const n = nodes.length;
  const cap = Array.from({ length: n }, () => new Map());
  edges.forEach(({ a, b, w }) => {
    const i = id.get(a), j = id.get(b);
    cap[i].set(j, (cap[i].get(j) || 0) + w);
    cap[j].set(i, (cap[j].get(i) || 0) + w);
  });
  const S = id.get(s), T = id.get(t);
  let value = 0;
  for (;;) {
    const prev = new Array(n).fill(-1);
    prev[S] = S;
    const q = [S];
    while (q.length && prev[T] === -1) {
      const u = q.shift();
      for (const [v, c] of cap[u]) {
        if (c > 1e-12 && prev[v] === -1) { prev[v] = u; q.push(v); }
      }
    }
    if (prev[T] === -1) break;
    let f = Infinity;
    for (let v = T; v !== S; v = prev[v]) f = Math.min(f, cap[prev[v]].get(v));
    for (let v = T; v !== S; v = prev[v]) {
      const u = prev[v];
      cap[u].set(v, cap[u].get(v) - f);
      cap[v].set(u, (cap[v].get(u) || 0) + f);
    }
    value += f;
  }
  // source side of the minimum cut: nodes reachable in the residual graph
  const seen = new Array(n).fill(false);
  const q = [S];
  seen[S] = true;
  while (q.length) {
    const u = q.shift();
    for (const [v, c] of cap[u]) if (c > 1e-12 && !seen[v]) { seen[v] = true; q.push(v); }
  }
  const side = new Set(nodes.filter((_, i) => seen[i]));
  const cut = new Set();
  edges.forEach((e, k) => { if (side.has(e.a) !== side.has(e.b)) cut.add(k); });
  return { value, cut, side };
}

// Moser-Dutton tunnelling estimate, optimal driving force (dG = -lambda):
// log10 k = 13 - 0.6 (R - 3.6), R edge-to-edge distance in Angstrom.
export const moserDutton = (R) => 10 ** (13 - 0.6 * (R - 3.6));
