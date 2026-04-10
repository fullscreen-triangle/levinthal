/**
 * Operation Executor (Symbolic Layer)
 * ====================================
 * Sits between the compiled probe (model) and the shader engine.
 * Translates operation sequences into GPU shader passes.
 *
 * This is the deterministic "CPU" between the neural compiler and the GPU apparatus.
 * When the Purpose model is trained, it produces operation sequences.
 * For now, we parse natural language into operations via pattern matching.
 *
 * Operations map 1:1 to Partition Calculus:
 *   probe(seq)       → Pass 1 (S-entropy)
 *   couple(N)        → Pass 3 (coupling matrix)
 *   observe(N,t)     → Pass 6 (spectrum)
 *   diagnose(N)      → Pass 7 (coherence/contacts)
 *   detect_cavities  → Pass C (harmonic network)
 *   mutate(pos,aa)   → modify sequence, re-run pipeline
 *   compare(A,B)     → run pipeline on both, compute deviation
 *   complete         → run all passes, extract fingerprint
 */

import { extractFingerprint, activityDescriptor, fingerprintDistance } from './CavityDatabase';

/** All recognized operations. */
const OPERATIONS = {
  probe:     { passes: [1],       desc: 'Compute S-entropy field from sequence' },
  couple:    { passes: [1,3],     desc: 'Build coupling matrix K_ij' },
  observe:   { passes: [1,3,6],   desc: 'Generate coupling spectrum (2D-IR)' },
  diagnose:  { passes: [1,3,6,7], desc: 'Measure coherence and predict contacts' },
  cavities:  { passes: [1,3,'C'], desc: 'Detect virtual resonant cavities' },
  complete:  { passes: [1,3,6,7,'C'], desc: 'Full pipeline + fingerprint extraction' },
  mutate:    { passes: ['mutate',1,3,6,7], desc: 'Introduce mutation and measure effect' },
  compare:   { passes: ['compare'], desc: 'Compare two sequences' },
  predict:   { passes: [1,3,6,7,'C','sar'], desc: 'Predict activity from cavity fingerprint' },
  search:    { passes: [1,3,6,7,'C','search'], desc: 'Search database by cavity fingerprint' },
};

/** Parse natural language query into an operation sequence. */
export function parseQuery(text) {
  const lower = text.toLowerCase().trim();

  // Direct operation commands (for power users)
  if (lower.startsWith('op:')) {
    const opName = lower.slice(3).trim().split(/\s+/)[0];
    const args = lower.slice(3).trim().split(/\s+/).slice(1);
    if (OPERATIONS[opName]) {
      return { operation: opName, args, raw: text };
    }
  }

  // Natural language patterns → operations
  const patterns = [
    // Folding
    { match: /will.*(fold|stable|structure)/i,    op: 'diagnose', view: 'contacts' },
    { match: /fold/i,                              op: 'diagnose', view: 'contacts' },
    { match: /structure/i,                         op: 'complete', view: 'contacts' },

    // Binding
    { match: /bind|dock|interact|affinity/i,       op: 'cavities', view: 'cavity' },
    { match: /binding site|pocket|active site/i,   op: 'cavities', view: 'cavity' },

    // Disease / Mutation
    { match: /mutate?\s+(\w)(\d+)(\w)/i,           op: 'mutate',   view: 'contacts',
      extract: (m) => ({ pos: parseInt(m[2]) - 1, from: m[1], to: m[3] }) },
    { match: /pathogen|disease|harmful|damag/i,     op: 'diagnose', view: 'contacts' },
    { match: /mutation|variant|snp/i,               op: 'diagnose', view: 'contacts' },

    // Spectrum / Observation
    { match: /spectrum|2d.?ir|spectro/i,            op: 'observe',  view: 'spectrum' },
    { match: /coupling|resonat/i,                   op: 'couple',   view: 'coupling' },
    { match: /coherence|health|order param/i,       op: 'diagnose', view: 'contacts' },

    // Cavity
    { match: /cavity|cavities|loop|harmonic/i,      op: 'cavities', view: 'cavity' },
    { match: /virtual.*resonan/i,                   op: 'cavities', view: 'cavity' },

    // SAR / Activity
    { match: /activity|sar|ic50|potency|predict/i,  op: 'predict',  view: 'cavity' },
    { match: /search|find|similar|database/i,       op: 'search',   view: 'cavity' },

    // S-entropy
    { match: /entropy|s.?entropy|composition/i,     op: 'probe',    view: 'sentropy' },

    // Generic analysis
    { match: /analyz|observ|measur|comput/i,        op: 'complete', view: 'spectrum' },
  ];

  for (const p of patterns) {
    const m = lower.match(p.match);
    if (m) {
      const result = { operation: p.op, view: p.view, raw: text };
      if (p.extract) result.args = p.extract(m);
      return result;
    }
  }

  // Default: full pipeline
  return { operation: 'complete', view: 'spectrum', raw: text };
}

/**
 * Execute an operation sequence on the shader engine.
 * Returns the result including which view to display and any computed values.
 */
export function executeOperation(parsed, engine, sequence, db, sar) {
  const result = {
    operation: parsed.operation,
    view: parsed.view || 'spectrum',
    message: '',
    metrics: null,
    fingerprint: null,
    searchResults: null,
    prediction: null,
    mutationDelta: null,
  };

  const opDef = OPERATIONS[parsed.operation];
  if (!opDef) {
    result.message = `Unknown operation: ${parsed.operation}`;
    return result;
  }

  // Handle mutation: modify sequence first
  if (parsed.operation === 'mutate' && parsed.args) {
    const { pos, to } = parsed.args;
    if (pos >= 0 && pos < sequence.length) {
      const before = sequence[pos];
      const mutSeq = sequence.substring(0, pos) + to.toUpperCase() + sequence.substring(pos + 1);

      // Run pipeline on original
      engine.setSequence(sequence);
      engine.observe(0.01);
      const metricsBefore = engine.readCoherence();

      // Run pipeline on mutant
      engine.setSequence(mutSeq);
      engine.observe(0.01);
      const metricsAfter = engine.readCoherence();

      const deltaEta = metricsAfter.eta - metricsBefore.eta;
      const pathogenic = Math.abs(deltaEta) > 0.05;

      result.message = `Mutation ${before}${pos + 1}${to.toUpperCase()}: ` +
        `Δη = ${deltaEta > 0 ? '+' : ''}${deltaEta.toFixed(3)} → ` +
        `${pathogenic ? 'POTENTIALLY PATHOGENIC' : 'LIKELY BENIGN'}`;
      result.mutationDelta = { before, after: to, pos, deltaEta, pathogenic };
      result.metrics = metricsAfter;

      // Restore mutant sequence for display
      return { ...result, newSequence: mutSeq };
    }
  }

  // Standard pipeline execution
  result.message = opDef.desc;

  // Run engine (it already runs in the animation loop, but we ensure a fresh observation)
  if (engine && engine.N >= 2) {
    engine.observe(0.01);
    result.metrics = engine.readCoherence();

    // Fingerprint extraction for SAR/search operations
    if (['complete', 'predict', 'search', 'cavities'].includes(parsed.operation)) {
      try {
        const gl = engine.gl;
        const N = engine.N;
        const sData = new Float32Array(N * 1 * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, engine.framebuffers.sentropy.fbo);
        gl.readPixels(0, 0, N, 1, gl.RGBA, gl.FLOAT, sData);
        const cData = new Float32Array(N * N * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, engine.framebuffers.coupling.fbo);
        gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, cData);
        const hData = new Float32Array(N * N * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, engine.framebuffers.coherence.fbo);
        gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, hData);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        result.fingerprint = extractFingerprint(sData, cData, hData, N);
      } catch (e) { /* ignore */ }
    }

    // Database search
    if (parsed.operation === 'search' && db && result.fingerprint) {
      result.searchResults = db.search(result.fingerprint, 5);
      result.message = `Found ${result.searchResults.length} matches in cavity database`;
    }

    // SAR prediction
    if (parsed.operation === 'predict' && sar && result.fingerprint) {
      const pred = sar.predictFromFingerprint(result.fingerprint);
      result.prediction = {
        logActivity: Math.round(pred * 100) / 100,
        ic50_nm: Math.round(Math.pow(10, -pred) * 1e9),
      };
      result.message = `Predicted pIC50 = ${result.prediction.logActivity}`;
    }
  }

  return result;
}

/** Format operation result for display. */
export function formatResult(result) {
  const lines = [];
  lines.push(`> ${result.operation}: ${result.message}`);

  if (result.metrics) {
    const eta = result.metrics.eta;
    const status = eta > 0.8 ? 'COHERENT' : eta > 0.5 ? 'STRESSED' : 'DECOHERENT';
    lines.push(`  η = ${eta.toFixed(3)} [${status}]`);
    lines.push(`  contacts = ${result.metrics.contacts}`);
  }

  if (result.fingerprint) {
    const fp = result.fingerprint;
    lines.push(`  cavities = ${fp.nCavities}, <Q> = ${Math.round(fp.meanQ)}, edges = ${fp.nEdges}`);
  }

  if (result.mutationDelta) {
    const md = result.mutationDelta;
    lines.push(`  ${md.before}${md.pos + 1}${md.after}: Δη = ${md.deltaEta > 0 ? '+' : ''}${md.deltaEta.toFixed(3)}`);
    lines.push(`  verdict: ${md.pathogenic ? '⚠ PATHOGENIC' : '✓ BENIGN'}`);
  }

  if (result.searchResults) {
    lines.push(`  matches:`);
    result.searchResults.forEach((r, i) =>
      lines.push(`    ${i + 1}. ${r.name} (d=${r.distance.toFixed(3)})`)
    );
  }

  if (result.prediction) {
    lines.push(`  pIC50 = ${result.prediction.logActivity}`);
  }

  return lines.join('\n');
}
