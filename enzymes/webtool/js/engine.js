/* =====================================================================
   engine.js -- the Shakespeare ladder kernel, ported from
   docs/shakespear-protein-contact-sequence/validation/shk_core.py

   This is the machine, not a mock of it.  Every number the notebook
   reports is computed here at run time in the browser.

   A rung carries exactly one datum: its power.  There is no identity
   field.  That is the formalism, not an omission -- a demonstration of
   label-independence that compared objects carrying names would be
   demonstrating the names.
   ===================================================================== */
(function (global) {
  'use strict';

  // ------------------------------------------------------------------
  // Contact graph
  // ------------------------------------------------------------------
  function edgeKey(a, b) { return a < b ? a + '|' + b : b + '|' + a; }

  class ContactGraph {
    constructor(vertices, weights, medium) {
      this.vertices = new Set(vertices);
      this.weights = new Map(weights);      // "u|v" -> w > 0
      this.medium = medium || 'm';
      for (const [k, w] of this.weights) {
        if (!(w > 0)) throw new Error('weight must be strictly positive: ' + k);
      }
    }
    clone() {
      return new ContactGraph(this.vertices, this.weights, this.medium);
    }
    get floor() {                            // beta = min over a FINITE edge set
      if (this.weights.size === 0) throw new Error('floor undefined on empty edge set');
      let m = Infinity;
      for (const w of this.weights.values()) if (w < m) m = w;
      return m;
    }
    get total() {
      let s = 0; for (const w of this.weights.values()) s += w; return s;
    }
    items() {
      const out = []; for (const v of this.vertices) if (v !== this.medium) out.push(v);
      return out.sort();
    }
    endpoints(k) { return k.split('|'); }
    neighbours(v) {
      const out = new Set();
      for (const k of this.weights.keys()) {
        const [a, b] = this.endpoints(k);
        if (a === v) out.add(b); else if (b === v) out.add(a);
      }
      return out;
    }
    /* Ball of radius r, NOT traversing the medium.  The medium is adjacent
       to everything, so a path through it joins any two items in two steps
       and every ball of radius 2 would be the whole graph -- the resolution
       parameter would then have two settings and not be a resolution. */
    ball(v, radius) {
      const seen = new Set([v]);
      let frontier = new Set([v]);
      for (let i = 0; i < radius; i++) {
        const next = new Set();
        for (const u of frontier) {
          for (const n of this.neighbours(u)) {
            if (n !== this.medium && !seen.has(n)) next.add(n);
          }
        }
        if (next.size === 0) break;
        for (const n of next) seen.add(n);
        frontier = next;
      }
      return seen;
    }
    cutWeight(S) {                           // S a Set of vertices
      let s = 0;
      for (const [k, w] of this.weights) {
        const [a, b] = this.endpoints(k);
        if (S.has(a) !== S.has(b)) s += w;
      }
      return s;
    }
    /* sigma_r(v): minimum cut weight over subsets of the ball that contain v
       and omit the medium.  Exhaustive, hence correct by construction; the
       ball keeps it small. */
    sigmaLocal(v, radius) {
      const ball = this.ball(v, radius);
      const others = [...ball].filter(x => x !== v).sort();
      let best = Infinity;
      const n = others.length;
      for (let mask = 0; mask < (1 << n); mask++) {
        const S = new Set([v]);
        for (let i = 0; i < n; i++) if (mask & (1 << i)) S.add(others[i]);
        const r = this.cutWeight(S);
        if (r < best) best = r;
      }
      return best;
    }
    /* beta_r(v): least weight among edges internal to the ball together with
       its medium edges.  LOCAL -- this is what the global floor is not. */
    localFloor(v, radius) {
      const ball = this.ball(v, radius);
      let m = Infinity;
      for (const [k, w] of this.weights) {
        const [a, b] = this.endpoints(k);
        const ok = (ball.has(a) || a === this.medium) &&
                   (ball.has(b) || b === this.medium);
        if (ok && w < m) m = w;
      }
      return m === Infinity ? this.floor : m;
    }
  }

  function chainGraph(nItems, rng, mediumWeight, lo, hi) {
    mediumWeight = mediumWeight === undefined ? 1.0 : mediumWeight;
    lo = lo === undefined ? 0.5 : lo;
    hi = hi === undefined ? 3.0 : hi;
    const verts = ['m']; const w = new Map();
    for (let i = 0; i < nItems; i++) verts.push('v' + i);
    for (let i = 0; i < nItems; i++) w.set(edgeKey('v' + i, 'm'), mediumWeight);
    for (let i = 0; i < nItems - 1; i++) {
      w.set(edgeKey('v' + i, 'v' + (i + 1)), lo + rng() * (hi - lo));
    }
    return new ContactGraph(verts, w, 'm');
  }

  // ------------------------------------------------------------------
  // Derived power: the candidate and its two controls
  // ------------------------------------------------------------------
  const clamp01 = x => Math.max(0, Math.min(1, x));

  /* INTENSIVE.  Both the cut key and its normaliser are local, so a
     modification outside the ball cannot move it. */
  function powerIntensive(g, v, radius) {
    const s = g.sigmaLocal(v, radius);
    if (!(s > 0)) return 0;
    return clamp01(1 - g.localFloor(v, radius) / s);
  }

  /* NEAR-MISS.  Local cut key, GLOBAL normaliser.  This is the version we
     wrote first and it is not intensive: beta is a minimum over every edge,
     so one distant small edge lowers it and shifts this everywhere. */
  function powerGlobalFloor(g, v, radius) {
    const s = g.sigmaLocal(v, radius);
    if (!(s > 0)) return 0;
    return clamp01(1 - g.floor / s);
  }

  /* EXTENSIVE control.  Normalised by the total edge weight of the WHOLE
     graph, so any edge added anywhere moves it. */
  function powerExtensive(g, v, radius) {
    const t = g.total;
    if (!(t > 0)) return 0;
    return clamp01(g.sigmaLocal(v, radius) / t);
  }

  const POWER_FNS = {
    intensive: powerIntensive,
    globalfloor: powerGlobalFloor,
    extensive: powerExtensive
  };

  // ------------------------------------------------------------------
  // Composition
  // ------------------------------------------------------------------
  function composeMultiplicative(ps) {
    let q = 1; for (const p of ps) q *= (1 - p); return 1 - q;
  }
  function composeAdditive(ps) {
    return Math.min(1, ps.reduce((a, b) => a + b, 0));
  }
  function composeMax(ps) { return ps.length ? Math.max(...ps) : 0; }
  function composeMean(ps) {
    return ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : 0;
  }

  function residualFraction(ps) {
    let q = 1; for (const p of ps) q *= (1 - p); return q;
  }
  function gapTrajectory(ps, gap0) {
    let g = gap0 === undefined ? 1 : gap0;
    const out = [g];
    for (const p of ps) { g *= (1 - p); out.push(g); }
    return out;
  }
  /* d(composite)/d(pi_j) = prod_{i != j} (1 - pi_i) = P/(1-pi_j).
     Increasing in pi_j: control lies at the STRONGEST rung. */
  function sensitivity(ps) {
    return ps.map((_, j) => {
      let q = 1;
      ps.forEach((p, i) => { if (i !== j) q *= (1 - p); });
      return q;
    });
  }
  function minRungsFor(target, p) {
    if (target >= 1 || p <= 0) return null;
    return Math.ceil(Math.log(1 - target) / Math.log(1 - p));
  }
  function staticReachable(ps, target) {
    return composeMultiplicative(ps) >= target;
  }
  function saturationDiagnostic(n, pMax, target) {
    return (1 - Math.pow(1 - pMax, n)) < target;
  }

  // ------------------------------------------------------------------
  // Sequential derivation -- the setting in which order CAN matter.
  // Each rung sees the graph its predecessors left; committing consumes the
  // item's weakest item-item contact.
  // ------------------------------------------------------------------
  function deriveSequential(g, order, fnName, radius) {
    const fn = POWER_FNS[fnName];
    const gg = g.clone();
    const out = [];
    for (const v of order) {
      out.push(fn(gg, v, radius));
      let bestK = null, bestW = Infinity;
      for (const [k, w] of gg.weights) {
        const [a, b] = k.split('|');
        if ((a === v || b === v) && a !== gg.medium && b !== gg.medium) {
          if (w < bestW) { bestW = w; bestK = k; }
        }
      }
      if (bestK) gg.weights.delete(bestK);
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Verdicts.  Only gap-carrying labels may carry a gap; the constructor
  // rejects a payload that disagrees with its label, so a refusal cannot
  // smuggle out a result.
  // ------------------------------------------------------------------
  const VERDICTS = ['reached', 'short', 'subfloor', 'refused', 'empty'];

  function makeVerdict(label, payload) {
    if (VERDICTS.indexOf(label) < 0) throw new Error('unknown verdict: ' + label);
    payload = payload || {};
    const carries = (label === 'reached' || label === 'short');
    const has = Object.prototype.hasOwnProperty.call(payload, 'gap');
    if (carries !== has) {
      throw new Error("label '" + label + "' and payload disagree on carrying a gap");
    }
    return { label: label, payload: payload };
  }

  // ------------------------------------------------------------------
  // The machine.  E-Power / E-Derive / E-Observe-Power are FREE: they read
  // boundary already committed, so the clock M does not advance.  E-Climb
  // commits once per rung.
  // ------------------------------------------------------------------
  class Machine {
    constructor(graph, epsilon) {
      this.graph = graph;
      this.epsilon = epsilon === undefined ? 1e-9 : epsilon;
      this.residues = [];
      this.M = 0;
      this.trace = [];
    }
    probe() { this.trace.push('E-Probe'); return this.M; }          // free
    observePower(ps) { this.trace.push('E-Observe-Power');           // free
                       return composeMultiplicative(ps); }
    derive(g, v, radius, fnName) { this.trace.push('E-Derive');      // free
                                   return POWER_FNS[fnName || 'intensive'](g, v, radius); }
    commit(rung, state) {
      const residue = this.graph.floor;
      this.residues.push(residue);
      this.M += 1;
      state.gap *= (1 - rung);
      this.trace.push('E-Commit');
      return state;
    }
    climb(ps, gap0) {
      const state = { gap: gap0 === undefined ? 1 : gap0 };
      for (const p of ps) {
        if (state.gap <= this.epsilon) break;
        this.trace.push('E-Climb');
        this.commit(p, state);
      }
      this.trace.push('E-Halt');
      return state;
    }
    runVerdict(ps, target, gap0) {
      if (!ps.length) return makeVerdict('empty', { reason: 'ladder declares no rungs' });
      if (!staticReachable(ps, target)) {
        return makeVerdict('subfloor', {
          reason: 'declared rungs cannot reach declared target',
          best_possible: composeMultiplicative(ps),
          target: target,
          shortfall: target - composeMultiplicative(ps)
        });
      }
      const st = this.climb(ps, gap0);
      const achieved = 1 - st.gap;
      return makeVerdict(achieved >= target - 1e-12 ? 'reached' : 'short',
                         { gap: st.gap, achieved: achieved, M: this.M });
    }
  }

  // ------------------------------------------------------------------
  // Deterministic RNG so every run of the notebook is reproducible
  // ------------------------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function permutations(arr) {
    if (arr.length <= 1) return [arr.slice()];
    const out = [];
    for (let i = 0; i < arr.length; i++) {
      const rest = arr.slice(0, i).concat(arr.slice(i + 1));
      for (const p of permutations(rest)) out.push([arr[i]].concat(p));
    }
    return out;
  }

  global.LADDER = {
    ContactGraph, chainGraph, edgeKey,
    powerIntensive, powerGlobalFloor, powerExtensive, POWER_FNS,
    composeMultiplicative, composeAdditive, composeMax, composeMean,
    residualFraction, gapTrajectory, sensitivity,
    minRungsFor, staticReachable, saturationDiagnostic,
    deriveSequential, makeVerdict, VERDICTS, Machine,
    mulberry32, permutations
  };
})(window);
