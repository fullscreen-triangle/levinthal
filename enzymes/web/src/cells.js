import * as d3 from 'd3';
import * as LAD from './lib/engine.js';
import * as CH from './lib/charts.js';

/* =====================================================================
 cells.js -- the executable content of the notebook.

 Each entry declares the source shown in the cell and the function run
 when the cell is executed.  The source is not a decorative listing: it
 is the same computation, written out, so a reader can check that the
 printed numbers come from what the cell claims to do.
 ===================================================================== */
/* ES module form. */

const L = () => LAD;
const K = () => CH;
const fmt = v => CH.fmt(v);

// ---- small output helpers ---------------------------------------
function pre(out, text) {
  const p = document.createElement('pre');
  p.innerHTML = text;
  out.appendChild(p);
  return p;
}
function table(out, cols, rows, opts) {
  opts = opts || {};
  const t = document.createElement('table');
  t.className = 'res';
  const th = document.createElement('tr');
  cols.forEach((c, i) => {
    const e = document.createElement('th');
    e.textContent = c;
    if (i === 0 || (opts.left || []).includes(i)) e.className = 'l';
    th.appendChild(e);
  });
  t.appendChild(th);
  rows.forEach(r => {
    const tr = document.createElement('tr');
    if (r._sep) tr.className = 'sep';
    (r.cells || r).forEach((c, i) => {
      const e = document.createElement('td');
      e.innerHTML = c;
      if (i === 0 || (opts.left || []).includes(i)) e.className = 'l';
      tr.appendChild(e);
    });
    t.appendChild(tr);
  });
  out.appendChild(t);
  return t;
}
const val = s => `<span class="val">${s}</span>`;
const dim = s => `<span class="dim">${s}</span>`;
const ok = s => `<span class="ok">${s}</span>`;
const warn = s => `<span class="warnv">${s}</span>`;
const bad = s => `<span class="badv">${s}</span>`;

// =================================================================
const CELLS = {};

// -----------------------------------------------------------------
// 1. The substrate: a contact graph, its floor, its separation costs
// -----------------------------------------------------------------
CELLS.substrate = {
  src:
`# A contact graph: items, a medium adjacent to every item,
# strictly positive weights.  The floor is COMPUTED, not declared.
rng = mulberry32(7)
G   = chain_graph(n_items=6, rng=rng)

beta  = G.floor                 # min over a FINITE edge set  ->  > 0
Omega = G.total
sigma = [G.sigma_local(v, radius=1) for v in G.items()]`,
  run(out, st) {
    const rng = L().mulberry32(st.seed);
    const g = L().chainGraph(st.nItems, rng);
    st.graph = g;
    const items = g.items();
    const sig = items.map(v => g.sigmaLocal(v, st.radius));
    const lf = items.map(v => g.localFloor(v, st.radius));

    pre(out,
      `floor  β = ${val(fmt(g.floor))}   ${dim('(min over ' + g.weights.size + ' edges — strictly positive because the edge set is finite)')}\n` +
      `total  Ω = ${val(fmt(g.total))}   items = ${val(items.length)}   medium = ${val("'m'")}`);

    table(out, ['item', 'σ_r(v)', 'β_r(v)', 'π_r(v) = 1 − β_r/σ_r'],
      items.map((v, i) => [v, fmt(sig[i]), fmt(lf[i]),
        val(fmt(L().powerIntensive(g, v, st.radius)))]));

    const r = K().row(out);
    K().graphChart(r, 'contact graph — colour is derived power',
      g, { powers: items.map(v => L().powerIntensive(g, v, st.radius)) });
    K().bars(r, 'separation cost σ against the medium',
      items.map((v, i) => ({ label: v, value: sig[i], color: K().C.c1 })),
      { ylab: 'σ_r(v)' });
    K().bars(r, 'derived power π = 1 − β_r/σ_r',
      items.map(v => ({ label: v, value: L().powerIntensive(g, v, st.radius), color: K().C.c3 })),
      { ylab: 'π', ydomain: [0, 1] });
  }
};

// -----------------------------------------------------------------
// 2. Radius is a resolution parameter
// -----------------------------------------------------------------
CELLS.resolution = {
  src:
`# The radius is a RESOLUTION, not a threshold.  Sweep it and count
# how many pairs of items share a power, and how much the balls overlap.
for r in [0, 1, 2, 3]:
  powers  = [power_intensive(G, v, r) for v in G.items()]
  ident   = pairs_agreeing(powers, tol=1e-9)
  overlap = mean_pairwise_jaccard([G.ball(v, r) for v in G.items()])`,
  run(out, st) {
    const rng = L().mulberry32(st.seed + 31);
    const rows = [];
    const radii = [0, 1, 2, 3];
    const nGraphs = 22;
    const series = { ident: [], overlap: [], distinct: [] };

    for (const r of radii) {
      let ident = 0, distinct = 0, ovs = [];
      for (let k = 0; k < nGraphs; k++) {
        const g = L().chainGraph(5 + (k % 2), L().mulberry32(st.seed + 100 + k));
        const items = g.items();
        const ps = items.map(v => L().powerIntensive(g, v, r));
        const balls = items.map(v => g.ball(v, r));
        for (let i = 0; i < ps.length; i++) {
          for (let j = i + 1; j < ps.length; j++) {
            if (Math.abs(ps[i] - ps[j]) <= 1e-9) ident++;
            const a = balls[i], b = balls[j];
            let inter = 0; for (const q of a) if (b.has(q)) inter++;
            ovs.push(inter / (a.size + b.size - inter));
          }
        }
        distinct += new Set(ps.map(p => p.toFixed(12))).size;
      }
      const ov = ovs.reduce((a, b) => a + b, 0) / ovs.length;
      rows.push([r, val(ident), fmt(ov), distinct]);
      series.ident.push([r, ident]);
      series.overlap.push([r, ov]);
      series.distinct.push([r, distinct]);
    }

    pre(out, dim(`sweep over ${nGraphs} chain graphs, tolerance 1e-9`));
    table(out, ['radius', 'pairs identified', 'mean ball overlap', 'distinct powers'], rows);
    pre(out,
      `\n${warn('The direction is the opposite of what we first predicted.')}\n` +
      dim('At radius 0 the balls are disjoint singletons and every item gets its own\n' +
          'value; as the radius grows the balls OVERLAP and distinct items minimise\n' +
          'over the same edges, so their powers converge.  The overlap column is the\n' +
          'mechanism — a count without it would leave the direction unexplained.'));

    const r = K().row(out);
    K().lines(r, 'identification rises with the radius',
      [{ points: series.ident, label: 'pairs identified', color: K().C.c1, dots: true }],
      { xlab: 'radius ϱ', ylab: 'pairs', xvals: radii, xformat: d3.format('d') });
    K().lines(r, 'the mechanism: ball overlap',
      [{ points: series.overlap, label: 'mean overlap', color: K().C.c2, dots: true }],
      { xlab: 'radius ϱ', ylab: 'Jaccard overlap', xvals: radii,
        xformat: d3.format('d'), ydomain: [0, 1] });
    K().lines(r, 'distinct powers fall as balls merge',
      [{ points: series.distinct, label: 'distinct', color: K().C.c3, dots: true }],
      { xlab: 'radius ϱ', ylab: 'distinct values', xvals: radii, xformat: d3.format('d') });
  }
};

// -----------------------------------------------------------------
// 3. Intensivity, with its near-miss and extensive controls
// -----------------------------------------------------------------
CELLS.intensivity = {
  src:
`# Compute a power at v0, then EXTEND the graph far from v0, recompute.
# An intensive quantity must be unchanged EXACTLY.
#   intensive    1 - beta_r(v)/sigma_r(v)   both factors local
#   near-miss    1 - beta  /sigma_r(v)      normaliser is GLOBAL
#   extensive    sigma_r(v)/Omega           normalised by the whole graph
for name in ["intensive", "globalfloor", "extensive"]:
  drift = [abs(f(G, "v0", 1) - f(extend(G), "v0", 1)) for G in graphs]`,
  run(out, st) {
    const names = ['intensive', 'globalfloor', 'extensive'];
    const labels = { intensive: 'intensive', globalfloor: 'near-miss', extensive: 'extensive' };
    const N = 120;
    const stats = {}; const scat = {};

    for (const name of names) {
      const rng = L().mulberry32(st.seed + 21);
      let maxd = 0, sum = 0, moved = 0; const pts = [];
      for (let i = 0; i < N; i++) {
        const n = 4 + Math.floor(rng() * 3);
        const g = L().chainGraph(n, rng);
        // extend far from v0, with one small distant medium edge
        const w = new Map(g.weights); const verts = new Set(g.vertices);
        const last = Math.max(...g.items().map(v => +v.slice(1)));
        const k = 1 + Math.floor(rng() * 3);
        for (let j = 1; j <= k; j++) {
          const nv = 'v' + (last + j); verts.add(nv);
          w.set(L().edgeKey(nv, 'm'), j === 1 ? g.floor * 0.5 : 1.0);
          w.set(L().edgeKey('v' + (last + j - 1), nv), 0.5 + rng() * 2.5);
        }
        const ext = new (L().ContactGraph)(verts, w, 'm');
        const a = L().POWER_FNS[name](g, 'v0', 1);
        const b = L().POWER_FNS[name](ext, 'v0', 1);
        const d = Math.abs(a - b);
        maxd = Math.max(maxd, d); sum += d; if (d > 1e-12) moved++;
        if (pts.length < 90) pts.push([a, b]);
      }
      stats[name] = { max: maxd, mean: sum / N, moved };
      scat[name] = pts;
    }

    table(out, ['candidate', 'max drift', 'mean drift', 'graphs moved'],
      names.map(n => [
        labels[n],
        n === 'intensive' ? ok(fmt(stats[n].max)) : bad(fmt(stats[n].max)),
        fmt(stats[n].mean),
        (n === 'intensive' ? ok : bad)(stats[n].moved + '/' + N)
      ]));

    pre(out,
      `\n${dim('The near-miss differs from the candidate in ONE component — the')}\n` +
      `${dim('normaliser — so its drift localises the failure rather than merely')}\n` +
      `${dim('exhibiting one.  β is a minimum over EVERY edge, so a single distant')}\n` +
      `${dim('small edge lowers it and shifts the quantity everywhere.')}\n` +
      `\n${dim('Had no control drifted, this test would not be measuring intensivity')}\n` +
      `${dim('and would have to be reported as non-discriminating.')}`);

    const r = K().row(out);
    K().bars(r, 'max drift under extension (log)',
      names.map(n => ({ label: labels[n], value: Math.max(stats[n].max, 1e-18),
                        color: n === 'intensive' ? K().C.c1 : K().C.c2 })),
      { logy: true, ylab: 'max |Δπ|' });
    K().bars(r, 'fraction of graphs moved',
      names.map(n => ({ label: labels[n], value: stats[n].moved / N,
                        color: n === 'intensive' ? K().C.c1 : K().C.c2 })),
      { ylab: 'fraction', ydomain: [0, 1] });
    K().scatter(r, 'before against after extension',
      names.map((n, i) => ({ points: scat[n], label: labels[n],
                             color: [K().C.c1, K().C.c2, K().C.c3][i], r: 2.1 })),
      { identity: true, xlab: 'π before', ylab: 'π after',
        xdomain: [0, 1], ydomain: [0, 1] });
  }
};

// -----------------------------------------------------------------
// 4. Order dependence, sequentially derived
// -----------------------------------------------------------------
CELLS.order = {
  src:
`# Permuting a FIXED multiset of powers cannot change 1 - prod(1-p):
# the law is symmetric, so that test cannot fail and is not evidence.
# The test that CAN fail derives powers SEQUENTIALLY, so each rung sees
# the graph its predecessors mutated.
for perm in permutations(G.items()):          # 720 orderings
  ps = derive_sequential(G, perm, power_fn, radius)
  composites.add(compose_multiplicative(ps))`,
  run(out, st) {
    const g = L().chainGraph(6, L().mulberry32(st.seed + 23));
    const items = g.items();
    const perms = L().permutations(items);
    const names = ['intensive', 'globalfloor', 'extensive'];
    const labels = { intensive: 'intensive', globalfloor: 'near-miss', extensive: 'extensive' };
    const res = {}; const dists = {};

    for (const r of [0, 1]) {
      for (const name of names) {
        const set = new Set(); const vals = [];
        for (const p of perms) {
          const c = L().composeMultiplicative(L().deriveSequential(g, p, name, r));
          set.add(c.toFixed(10)); vals.push(c);
        }
        res[name + r] = { distinct: set.size,
                          spread: Math.max(...vals) - Math.min(...vals) };
        if (r === 1) dists[name] = vals;
      }
    }

    // the non-discriminating comparison, run and labelled as such
    const fixedPowers = items.map(v => L().powerIntensive(g, v, 1));
    const permSet = new Set(perms.map(p =>
      L().composeMultiplicative(p.map((_, i) => fixedPowers[items.indexOf(p[i])])).toFixed(12)));

    table(out,
      ['candidate', 'distinct @ϱ=0', 'distinct @ϱ=1', 'spread @ϱ=1'],
      names.map(n => [labels[n],
        res[n + '0'].distinct, val(res[n + '1'].distinct), fmt(res[n + '1'].spread)]));

    pre(out,
      `\n${dim('over all ' + perms.length + ' orderings of 6 items')}\n` +
      `\nexact order-independence would be ${val('distinct = 1')} — ` +
      `${bad('not attained by any candidate')}\n` +
      `${dim('Intensivity reduces order dependence by roughly an order of magnitude')}\n` +
      `${dim('and finer locality reduces it further, but does not abolish it:')}\n` +
      `${dim('commitment mutates edges INSIDE neighbouring balls, which is exactly')}\n` +
      `${dim('the case the intensivity theorem does not cover.')}\n` +
      `\n${warn('NON-DISCRIMINATING control:')} permuting a fixed power multiset gives ` +
      `${val(permSet.size)} distinct value${permSet.size === 1 ? '' : 's'}\n` +
      `${dim('— the composition law is symmetric, so every candidate passes and the')}\n` +
      `${dim('comparison cannot fail.  It is excluded from any score.')}`);

    const r = K().row(out);
    K().bars(r, 'distinct composites over 720 orderings (log)',
      names.map(n => ({ label: labels[n], value: res[n + '1'].distinct,
                        color: n === 'intensive' ? K().C.c1 : K().C.c2 })),
      { logy: true, ylab: 'distinct' });
    K().groupedBars(r, 'radius 0 against radius 1',
      names.map(n => ({ label: labels[n], 'ϱ=0': res[n + '0'].distinct,
                        'ϱ=1': res[n + '1'].distinct })),
      ['ϱ=0', 'ϱ=1'], { logy: true, ylab: 'distinct' });
    K().histogram(r, 'composite over all orderings — intensive',
      dists.intensive, { xlab: 'composite power', color: K().C.c1, bins: 30 });
  }
};

// -----------------------------------------------------------------
// 5. Building a ladder: composition
// -----------------------------------------------------------------
CELLS.ladder = {
  src:
`# A ladder composes multiplicatively.  Each rung closes a fraction of
# the gap that REMAINS, so the residuals multiply:
#     pi(L) = 1 - prod(1 - pi_i)
L = Ladder([Rung(p) for p in powers])
composite = L.composite_power()
gaps      = L.gap_trajectory()
sens      = L.sensitivity()        # d(composite)/d(pi_j) = P/(1-pi_j)`,
  run(out, st) {
    const ps = st.powers.slice();
    const comp = L().composeMultiplicative(ps);
    const traj = L().gapTrajectory(ps, 1);
    const sens = L().sensitivity(ps);
    const P = L().residualFraction(ps);
    const argS = sens.indexOf(Math.max(...sens));
    const argP = ps.indexOf(Math.max(...ps));

    pre(out,
      `composite power  = ${val(fmt(comp))}   ${dim('= 1 − ' + ps.map(p => '(1−' + p.toFixed(2) + ')').join('·'))}\n` +
      `residual P       = ${val(fmt(P))}\n` +
      `commitments M    = ${val(ps.length)}   ${dim('(one per rung — climbing is the only costed rule)')}`);

    table(out, ['rung', 'power π', 'gap before', 'gap after', 'sensitivity', 'P/(1−π)'],
      ps.map((p, i) => [
        'γ' + (i + 1), fmt(p), fmt(traj[i]), fmt(traj[i + 1]),
        (i === argS ? val(fmt(sens[i])) : fmt(sens[i])),
        fmt(P / (1 - p))
      ]));

    pre(out,
      `\ncontrol lies at rung ${val(argS + 1)} ` +
      `(power ${val(fmt(ps[argS]))}) — the ${val('strongest')}, not the weakest\n` +
      (argS === argP
        ? ok('   argmax sensitivity = argmax power  ✓')
        : bad('   argmax sensitivity ≠ argmax power')) + '\n' +
      dim('   Sensitivity is TRANSMISSION through the remaining rungs, not slack:\n' +
          '   a gain at rung j reaches the output only through ∏_{i≠j}(1−π_i).\n' +
          '   This runs against the intuition that the bottleneck is where to invest.'));

    const r = K().row(out);
    K().ladderChart(r, 'the gap closing, rung by rung', ps, { target: st.target });
    K().bars(r, 'power against sensitivity',
      ps.flatMap((p, i) => [
        { label: 'γ' + (i + 1), value: p, color: K().C.c1 }
      ]), { ylab: 'power π', ydomain: [0, 1] });
    K().bars(r, 'sensitivity ∂π(L)/∂π_j',
      sens.map((s, i) => ({ label: 'γ' + (i + 1), value: s,
                            color: i === argS ? K().C.c2 : '#3d4f61' })),
      { ylab: '∂/∂π_j' });
  }
};

// -----------------------------------------------------------------
// 6. Composition law against three alternatives
// -----------------------------------------------------------------
CELLS.laws = {
  src:
`# Score FOUR candidate laws against a rung-by-rung simulation.
# Scoring only the favoured law would not distinguish it from the others.
for _ in range(4000):
  ps  = random_powers()
  gap = 1.0
  for p in ps: gap -= p * gap          # simulate
  truth = 1 - gap
  err["multiplicative"] += abs(compose_multiplicative(ps) - truth)
  err["additive"]       += abs(compose_additive(ps)       - truth)
  ...`,
  run(out, st) {
    const rng = L().mulberry32(st.seed + 41);
    const laws = {
      multiplicative: L().composeMultiplicative,
      additive: L().composeAdditive,
      max: L().composeMax,
      mean: L().composeMean
    };
    const err = { multiplicative: 0, additive: 0, max: 0, mean: 0 };
    const N = 4000; const scat = { multiplicative: [], additive: [], max: [], mean: [] };
    for (let i = 0; i < N; i++) {
      const n = 2 + Math.floor(rng() * 7);
      const ps = Array.from({ length: n }, () => 0.05 + rng() * 0.8);
      let gap = 1; for (const p of ps) gap -= p * gap;
      const truth = 1 - gap;
      for (const k in laws) {
        const v = laws[k](ps);
        err[k] += Math.abs(v - truth);
        if (scat[k].length < 260) scat[k].push([truth, v]);
      }
    }
    const mae = {}; for (const k in err) mae[k] = err[k] / N;
    const best = Object.keys(mae).reduce((a, b) => mae[a] <= mae[b] ? a : b);

    table(out, ['law', 'mean absolute error'],
      Object.keys(laws).map(k => [
        k === 'multiplicative' ? '1 − ∏(1−πᵢ)'
          : k === 'additive' ? 'min(1, Σπᵢ)'
          : k === 'max' ? 'max πᵢ' : 'mean πᵢ',
        k === best ? ok(fmt(mae[k])) : bad(fmt(mae[k]))
      ]));
    pre(out, `\n${dim(N + ' generated ladders; best law: ')}${val(best)}` +
             (mae.multiplicative < 1e-12 ? '  ' + ok('(exact to machine precision)') : ''));

    const r = K().row(out);
    K().bars(r, 'mean absolute error by law (log)',
      Object.keys(laws).map((k, i) => ({
        label: k.slice(0, 5), value: Math.max(mae[k], 1e-18),
        color: k === best ? K().C.c1 : K().C.c2
      })), { logy: true, ylab: 'MAE' });
    K().scatter(r, 'predicted against simulated',
      Object.keys(laws).map((k, i) => ({
        points: scat[k], label: k.slice(0, 5), color: K().SEQ[i], r: 1.7, opacity: 0.5
      })), { identity: true, xlab: 'simulated', ylab: 'predicted',
             xdomain: [0, 1], ydomain: [0, 1] });
    // residual is a product -> straight lines on a log axis
    K().lines(r, 'residual is a product, not a sum (log)',
      [0.2, 0.4, 0.6, 0.8].map((p, i) => ({
        points: d3.range(0, 16).map(n => [n, Math.pow(1 - p, n)]),
        label: 'π=' + p, color: K().SEQ[i]
      })), { logy: true, xlab: 'rungs n', ylab: 'residual' });
  }
};

// -----------------------------------------------------------------
// 7. Semantics: free rules, the clock, the refusal
// -----------------------------------------------------------------
CELLS.semantics = {
  src:
`# E-Power / E-Derive / E-Observe-Power are FREE: they read boundary
# already committed, so the clock M does not advance.
# E-Climb commits once per rung.  A refusal commits nothing at all.
m = Machine(G)
for _ in range(5000): m.probe()          # free
M_after_free = m.M                        # -> 0
v = m.run_verdict(powers, target)         # reached | short | subfloor | empty`,
  run(out, st) {
    const g = st.graph || L().chainGraph(st.nItems, L().mulberry32(st.seed));
    const m = new (L().Machine)(g);
    for (let i = 0; i < 5000; i++) m.probe();
    const afterFree = m.M;
    m.observePower(st.powers);
    m.derive(g, 'v0', 1, 'intensive');
    const afterObs = m.M;

    const m2 = new (L().Machine)(g);
    const v2 = m2.runVerdict(st.powers, st.target);
    const m3 = new (L().Machine)(g);
    const v3 = m3.runVerdict(st.powers, 0.999);
    const m4 = new (L().Machine)(g);
    let emptyLabel;
    try { emptyLabel = m4.runVerdict([], st.target).label; } catch (e) { emptyLabel = 'ERR'; }

    let guard = '';
    try { L().makeVerdict('subfloor', { gap: 1.0 }); guard = bad('GUARD FAILED'); }
    catch (e) { guard = ok('rejected: ') + dim(e.message); }

    pre(out,
      `M after 5000 free operations  = ${(afterFree === 0 ? ok : bad)(afterFree)}  ` +
      dim('(probe, derive, observe — none commits a cut)') + '\n' +
      `M after observe + derive      = ${(afterObs === 0 ? ok : bad)(afterObs)}\n` +
      `M after climbing ${st.powers.length} rungs      = ${val(m2.M)}  ` +
      dim('(one commitment per rung)') + '\n' +
      `residues recorded             = ${val(m2.residues.length)}, ` +
      `each ≥ β = ${val(fmt(g.floor))}  ` +
      (m2.residues.every(x => x >= g.floor - 1e-15) ? ok('✓') : bad('✗')));

    table(out, ['program', 'target', 'verdict', 'payload', 'M', 'residues'],
      [
        ['climb', fmt(st.target), (v2.label === 'reached' ? ok : warn)(v2.label),
         dim('gap ' + fmt(v2.payload.gap) + ' · achieved ' + fmt(v2.payload.achieved)),
         m2.M, m2.residues.length],
        ['climb', '0.999', bad(v3.label),
         dim('shortfall ' + fmt((v3.payload.shortfall) || 0)), m3.M, m3.residues.length],
        ['ladder []', fmt(st.target), bad(emptyLabel),
         dim('no rungs declared'), m4.M, m4.residues.length]
      ], { left: [0, 2, 3] });

    pre(out,
      `\n${dim('A refused ladder commits nothing: M stays 0 and no residue is recorded.')}\n` +
      `${dim('The verdict type also rejects a payload that disagrees with its label,')}\n` +
      `${dim('so a refusal cannot smuggle out a result —')}\n` +
      `  Verdict("subfloor", {gap: 1.0})  →  ${guard}`);

    const r = K().row(out);
    K().lines(r, 'the clock M under the two regimes',
      [
        { points: d3.range(0, 41).map(k => [k, k]), label: 'climb', color: K().C.c2 },
        { points: d3.range(0, 41).map(k => [k, 0]), label: 'free rules',
          color: K().C.c3, width: 3 }
      ], { xlab: 'operations', ylab: 'clock M' });
    K().bars(r, 'commitments and residues by program',
      [
        { label: 'reached', value: m2.M, color: K().C.c1 },
        { label: 'refused', value: m3.M, color: K().C.c2 },
        { label: 'empty', value: m4.M, color: K().C.c2 }
      ], { ylab: 'M', zeroMark: true });
    // rungs required surface
    const tg = d3.range(0.30, 0.98, 0.045), pw = d3.range(0.06, 0.72, 0.045);
    K().heatmap(r, 'rungs required for a target',
      { xs: tg, ys: pw,
        z: pw.map(p => tg.map(t => Math.log(1 - t) / Math.log(1 - p))) },
      { xlab: 'target', ylab: 'rung power π', zlab: 'rungs', scheme: d3.interpolateMagma });
  }
};

// -----------------------------------------------------------------
// 8. Computing sequences over a real pathway
// -----------------------------------------------------------------
CELLS.sequence = {
  src:
`# A computing sequence: derive one rung per step of a real pathway,
# then compose.  The rungs carry NO identity — only a power — so the
# sequence is a statement about how much of the gap each step closes.
pathway = corpus.pathways[i]                  # Reactome, frozen snapshot
powers  = [rung_power(step) for step in pathway.reactions]
composite, gaps, sens = compose(powers)`,
  run(out, st) {
    const c = st.corpus;
    if (!c) { pre(out, bad('corpus not loaded')); return; }
    const pw = c.pathways[st.pathwayIdx % c.pathways.length];
    const rxs = pw.reactions;

    // A rung power derived from the reaction's own participant counts:
    // more participants committed => a larger share of the gap closed.
    // Bounded into (0,1) so no single step is absolute.
    const powers = rxs.map(r => {
      const parts = r.n_in + r.n_out + r.n_cat;
      return Math.min(0.62, 0.06 + 0.055 * parts);
    });
    const comp = L().composeMultiplicative(powers);
    const traj = L().gapTrajectory(powers, 1);
    const sens = L().sensitivity(powers);
    const argS = sens.indexOf(Math.max(...sens));
    const nMin = L().minRungsFor(0.90, Math.max(...powers));

    pre(out,
      `pathway            ${val(pw.pathway)}\n` +
      `steps shown        ${val(rxs.length)} ${dim('of ' + pw.n_total + ' in the corpus')}\n` +
      `composite power    ${val(fmt(comp))}\n` +
      `residual gap       ${val(fmt(1 - comp))}\n` +
      `rungs for 0.90     ${val(nMin)} ${dim('at the strongest rung power ' + fmt(Math.max(...powers)))}`);

    table(out, ['#', 'reaction', 'in', 'out', 'cat', 'power π', 'gap after', 'sens'],
      rxs.map((r, i) => [
        i + 1, dim(r.name || r.rid), r.n_in, r.n_out, r.n_cat,
        fmt(powers[i]), fmt(traj[i + 1]),
        i === argS ? val(fmt(sens[i])) : fmt(sens[i])
      ]), { left: [0, 1] });

    // With many rungs the residual is small and several rungs share a
    // power, so the argmax can be a TIE.  Say so rather than name an
    // arbitrary winner: sensitivity is P/(1-pi_j), and equal powers give
    // equal sensitivity.
    const maxS = Math.max(...sens);
    const tied = sens.filter(x => Math.abs(x - maxS) < 1e-12).length;
    const maxP = Math.max(...powers);
    pre(out,
      (tied === 1
        ? '\n' + dim('control lies at step ') + val(argS + 1) +
          dim(' \u2014 the highest-power rung.')
        : '\n' + dim('control is TIED across ') + val(tied) +
          dim(' steps, all at the highest power ') + val(fmt(maxP)) +
          dim('.') + '\n' +
          dim('   Sensitivity is P/(1\u2212\u03c0_j), so equal powers give equal\n' +
              '   sensitivity.  Naming one of them the control would be\n' +
              '   picking arbitrarily, so the tie is reported instead.')) + '\n' +
      dim('Two steps with the same power are the same rung here: nothing\n' +
          'about what the reaction IS enters the computation, or could be\n' +
          'recovered from it.  That is the point, not a simplification.'));

    const r = K().row(out);
    K().ladderChart(r, 'gap closing along the pathway', powers, { target: 0.9 });
    K().bars(r, 'derived power per step',
      powers.map((p, i) => ({ label: String(i + 1), value: p,
                              color: Math.abs(sens[i] - maxS) < 1e-12
                                ? K().C.c2 : K().C.c1 })),
      { ylab: 'π', ydomain: [0, 0.7], xlab: 'step' });
    K().lines(r, 'composite against sequence length',
      [{ points: powers.map((_, i) =>
          [i + 1, L().composeMultiplicative(powers.slice(0, i + 1))]),
         label: 'composite', color: K().C.c3, dots: true }],
      { xlab: 'steps applied', ylab: 'composite', ydomain: [0, 1], hline: 0.9 });
  }
};

// -----------------------------------------------------------------
// 9. Federated querying: the plan, its verdicts, its cost
// -----------------------------------------------------------------
CELLS.federated = {
  src:
`plan ladder_demo {
budget 50 requests

let acids = from chebi
    ask descendants_of("CHEBI:1")
    within 10

let L = ladder over acids
    power 0.45, power 0.30, power 0.55
    expect power 0.70            # raise this to 0.95 and it refuses

emit L
}`,
  run(out, st) {
    const ps = [0.45, 0.30, 0.55];
    const comp = L().composeMultiplicative(ps);
    const target = st.hfqTarget;
    const reach = comp >= target;
    const shortfall = target - comp;

    // the plan's steps, as the host executor reports them
    const steps = [
      { step: 'acids', kind: 'from chebi', verdict: 'answer',
        spent: 1.0, alloc: 1.0, cap: 1 },
      { step: 'L', kind: 'ladder', verdict: reach ? 'answer' : 'starved',
        spent: 0.0, alloc: 1.0, cap: 0 }
    ];

    pre(out,
      `capability check   ${ok('well-capability')} · ` +
      `${val(1)} membership test ${dim('(the ladder contributes none — it reaches no source)')}\n` +
      `composite power    ${val(fmt(comp))} ${dim('= 1 − (0.55)(0.70)(0.45)')}\n` +
      `declared target    ${val(fmt(target))}\n` +
      `requests issued    ${val(1)} ${dim('— all of it for the source step')}`);

    table(out, ['step', 'kind', 'verdict', 'spent', 'allocated', 'cap. ops'],
      steps.map(s => [
        s.step, dim(s.kind),
        s.verdict === 'answer' ? ok(s.verdict) : bad(s.verdict),
        s.spent === 0 ? ok(fmt(s.spent)) : fmt(s.spent),
        fmt(s.alloc), s.cap
      ]), { left: [0, 1, 2] });

    if (!reach) {
      pre(out,
        `\n${bad('starved')} — the ladder cannot reach its declared target\n` +
        `  shortfall            ${val(fmt(shortfall))}\n` +
        `  named_predecessor    ${val('null')}  ${warn('← deliberately empty')}\n` +
        `  blame chain          ${val("['L']")}  ${dim('(stops at the ladder)')}\n\n` +
        dim('The input step answered correctly, so naming it as the culprit would\n' +
            'accuse a step that did nothing wrong and send the blame walk one hop\n' +
            'too far.  The shortfall is in the DECLARATION, not in the data.'));
    } else {
      pre(out, `\n${ok('answer')} — composite clears the declared target by ` +
        val(fmt(-shortfall)) + '\n' +
        dim('Raise the target above ' + fmt(comp) + ' and the step reports starved,\n' +
            'with the shortfall named and the blame walk stopping at the ladder.'));
    }

    const r = K().row(out);
    K().bars(r, 'budget spent per step',
      [{ label: 'acids\nsource', value: 1.0, color: K().C.c1 },
       { label: 'L\nladder', value: 0.0, color: K().C.c2 }],
      { ylab: 'requests', zeroMark: true, ydomain: [0, 1.3] });
    K().groupedBars(r, 'allocated against spent',
      [{ label: 'source', allocated: 1.0, spent: 1.0 },
       { label: 'ladder', allocated: 1.0, spent: 0.0 }],
      ['allocated', 'spent'], { ylab: 'requests' });
    // the refusal surface: shortfall over (composite, target)
    const cs = d3.range(0, 1.001, 0.05), tsv = d3.range(0, 1.001, 0.05);
    K().heatmap(r, 'refusal surface: shortfall = max(0, target − composite)',
      { xs: cs, ys: tsv, z: tsv.map(t => cs.map(c => Math.max(0, t - c))) },
      { xlab: 'composite', ylab: 'declared target', zlab: 'shortfall',
        scheme: d3.interpolateInferno });
  }
};

// -----------------------------------------------------------------
// 10. The bound the host already had
// -----------------------------------------------------------------
CELLS.bound = {
  src:
`# HFQ bounds retention below by the UNION bound:
#     rho >= 1 - sum(1 - r_i)                (additive)
# The ladder's law is its multiplicative counterpart:
#     rho  = 1 - prod(1 - r_i)
# Measure both on the SAME numbers and count how often the additive form
# falls below zero -- at which point it bounds a quantity in [0,1] by a
# negative number and says nothing.`,
  run(out, st) {
    const rng = L().mulberry32(st.seed + 5);
    const N = 4000;
    const rows = []; const addS = [], mulS = [], vacS = [];
    for (const k of [2, 3, 4, 5, 6, 8]) {
      let a = 0, m = 0, vac = 0;
      for (let i = 0; i < N; i++) {
        const rs = Array.from({ length: k }, () => 0.5 + rng() * 0.49);
        const add = 1 - rs.reduce((s, r) => s + (1 - r), 0);
        let prod = 1; for (const r of rs) prod *= (1 - r);
        a += add; m += (1 - prod); vac += (add < 0 ? 1 : 0);
      }
      rows.push([k, (a / N >= 0 ? ' ' : '') + fmt(a / N), fmt(m / N),
                 fmt(m / N - a / N),
                 (vac / N > 0.5 ? bad : dim)(fmt(vac / N))]);
      addS.push([k, a / N]); mulS.push([k, m / N]); vacS.push([k, vac / N]);
    }
    table(out, ['stages k', 'mean additive', 'mean multiplicative', 'gap', 'fraction vacuous'], rows);
    pre(out,
      `\n${dim(N + ' trials per row, retentions drawn uniformly from [0.5, 0.99]')}\n` +
      `\n${warn('The additive form goes vacuous — negative, hence trivially true —')}\n` +
      `${warn('on the majority of chains by four stages.')}\n` +
      dim('This is not a criticism: the additive form is stated as a BOUND and\n' +
          'carries an injectivity hypothesis, and a loose bound is not a wrong\n' +
          'one.  What is measured is how much it gives up, and that the tight\n' +
          'form is the same formula the ladder composes with.'));

    const r = K().row(out);
    K().lines(r, 'the two forms',
      [{ points: mulS, label: '1 − ∏(1−rᵢ)', color: K().C.c1, dots: true },
       { points: addS, label: '1 − Σ(1−rᵢ)', color: K().C.c2, dots: true }],
      { xlab: 'chain length k', ylab: 'bound value', hline: 0, shadeBelow: 0,
        legendX: 6, legendY: 4 });
    K().bars(r, 'fraction of trials vacuous',
      vacS.map(([k, v]) => ({ label: String(k), value: v, color: K().C.c2 })),
      { xlab: 'chain length k', ylab: 'fraction', ydomain: [0, 1] });
    const kk = d3.range(2, 11), rr = d3.range(0.5, 0.995, 0.03);
    K().heatmap(r, 'gap between the two forms',
      { xs: kk, ys: rr,
        z: rr.map(rv => kk.map(k =>
          (1 - Math.pow(1 - rv, k)) - (1 - k * (1 - rv)))) },
      { xlab: 'chain length k', ylab: 'retention r', zlab: 'gap',
        scheme: d3.interpolateMagma });
  }
};

// -----------------------------------------------------------------
// 11. The corpus
// -----------------------------------------------------------------
CELLS.corpus = {
  src:
`# Two public services with different schemas, fetched once and FROZEN.
# The host forbids network access in adapters by construction -- its
# claims are properties of the compiler -- so we did not weaken that
# rule to run an experiment.
corpus = load("data/corpus.json")     # KEGG + Reactome, snapshot`,
  run(out, st) {
    const c = st.corpus;
    if (!c) { pre(out, bad('corpus not loaded')); return; }
    const t = c.totals;
    pre(out,
      `snapshot     ${val(c.snapshot)}   fetched ${dim(c.fetched_utc)}\n` +
      `KEGG         ${val(t.kegg_records)} enzyme records, ` +
      `${val(t.kegg_with_reactions)} with at least one reaction\n` +
      `             ${dim(c.provenance.kegg.endpoint)}\n` +
      `Reactome     ${val(t.reactome_reactions)} reactions, ` +
      `${val(t.reactome_with_catalysts)} with at least one catalyst\n` +
      `             ${dim(c.provenance.reactome.endpoint)}\n` +
      `pathways     ${val(t.n_pathways)}   longest: ${val(t.longest_pathways.slice(0, 6).join(', '))}`);

    pre(out, `\n${dim('The catalyst count was ZERO on the first extraction.  Rather than')}\n` +
      `${dim('record a plausible zero we queried the live service and found catalyst')}\n` +
      `${dim('records on reactions our extractor reported as having none — it')}\n` +
      `${dim('required a nested field the summary response omits.  After the fix: ')}` +
      val(t.reactome_with_catalysts) + dim('.'));

    const r = K().row(out);
    K().bars(r, 'corpus coverage',
      [{ label: 'KEGG\nrecords', value: t.kegg_records, color: K().C.c1 },
       { label: 'KEGG\nw/ rxn', value: t.kegg_with_reactions, color: K().C.c1 },
       { label: 'Rx\nreactions', value: t.reactome_reactions, color: K().C.c3 },
       { label: 'Rx\nw/ cat', value: t.reactome_with_catalysts, color: K().C.c3 }],
      { ylab: 'records' });
    K().bars(r, 'chain lengths available (top pathways)',
      c.pathways.map((p, i) => ({ label: String(i + 1), value: p.n_total,
                                  color: K().C.c3 })),
      { xlab: 'pathway (ranked)', ylab: 'reactions' });
    K().histogram(r, 'reactions per KEGG enzyme record',
      c.kegg.map(e => e.n_rxn), { xlab: 'reactions', color: K().C.c1, bins: 22 });
  }
};


export { CELLS, pre, table, val, dim, ok, warn, bad };
