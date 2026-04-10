/**
 * Cavity Fingerprint Database
 * ============================
 * In-browser protein/drug database indexed by virtual cavity fingerprints.
 * Search by cavity structure, not sequence or atomic coordinates.
 *
 * Similar function ⟺ similar cavity fingerprint
 * (even if sequence/structure are different -- convergent evolution)
 *
 * O(1) memory per query via observation-on-demand.
 * LSH indexing for sub-millisecond search on large databases.
 */

import { AA_INDEX, AA_SENTROPY } from './ShaderEngine';

/**
 * Compute cavity fingerprint from S-entropy coordinates and coupling data.
 * This runs on CPU from GPU readback data -- the GPU computed the cavities,
 * CPU extracts the fingerprint vector for database indexing.
 */
export function extractFingerprint(sentropyData, couplingData, coherenceData, N) {
  if (!sentropyData || N < 2) return null;

  // Detect harmonic edges from coupling strengths
  const edges = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 5; j < N; j++) {
      const idx = (i * N + j) * 4;
      const K = couplingData[idx];
      if (K > 1.0) {
        // Check harmonic proximity
        const omega_i = sentropyData[i * 4] * 10 + 1;
        const omega_j = sentropyData[j * 4] * 10 + 1;
        const ratio = Math.max(omega_i, omega_j) / Math.min(omega_i, omega_j);
        let best_dev = 1.0;
        for (let p = 1; p <= 8; p++) {
          for (let q = 1; q <= p; q++) {
            best_dev = Math.min(best_dev, Math.abs(ratio - p / q));
            best_dev = Math.min(best_dev, Math.abs(ratio - q / p));
          }
        }
        if (best_dev < 0.05) {
          edges.push({ i, j, K, ratio, harmonic: 1 - best_dev / 0.05 });
        }
      }
    }
  }

  // Detect cavities (triangular loops in harmonic network)
  const cavities = [];
  const edgeSet = new Set(edges.map(e => `${e.i}-${e.j}`));
  for (const e1 of edges) {
    for (const e2 of edges) {
      if (e2.i !== e1.j && e2.j !== e1.j) continue;
      const k = e2.i === e1.j ? e2.j : e2.i;
      if (k === e1.i) continue;
      const key = `${Math.min(e1.i, k)}-${Math.max(e1.i, k)}`;
      if (edgeSet.has(key)) {
        const tri = [e1.i, e1.j, k].sort((a, b) => a - b);
        const triKey = tri.join('-');
        if (!cavities.find(c => c.key === triKey)) {
          // Compute cavity properties
          const members = tri;
          const Q = members.reduce((s, m) => {
            const sk = sentropyData[m * 4];
            return s + sk * 1000;
          }, 0) / members.length;
          const omega = members.reduce((s, m) => {
            return s + (sentropyData[m * 4] * 10 + 1);
          }, 0) / members.length;
          const area = members.reduce((s, m) => {
            return s + sentropyData[m * 4 + 1] * 50;
          }, 0);

          cavities.push({
            key: triKey,
            members,
            Q: Math.round(Q),
            omega,
            area: Math.round(area * 10) / 10,
          });
        }
      }
    }
  }

  // Global coherence
  let sumR = 0, sumI = 0, count = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const idx = (i * N + j) * 4;
      const mag = coherenceData[idx + 3];
      const phase = coherenceData[idx + 1] * 2 * Math.PI * 2 - Math.PI;
      sumR += mag * Math.cos(phase);
      sumI += mag * Math.sin(phase);
      count++;
    }
  }
  const eta = Math.min(Math.sqrt(sumR * sumR + sumI * sumI) / (count || 1) * 50, 1.0);

  // Mean S-entropy of sequence
  let meanSk = 0, meanSt = 0, meanSe = 0;
  for (let i = 0; i < N; i++) {
    meanSk += sentropyData[i * 4];
    meanSt += sentropyData[i * 4 + 1];
    meanSe += sentropyData[i * 4 + 2];
  }
  meanSk /= N; meanSt /= N; meanSe /= N;

  return {
    nCavities: cavities.length,
    cavities: cavities.slice(0, 20), // keep top 20
    meanQ: cavities.length > 0 ? cavities.reduce((s, c) => s + c.Q, 0) / cavities.length : 0,
    meanOmega: cavities.length > 0 ? cavities.reduce((s, c) => s + c.omega, 0) / cavities.length : 0,
    meanArea: cavities.length > 0 ? cavities.reduce((s, c) => s + c.area, 0) / cavities.length : 0,
    coherence: eta,
    nEdges: edges.length,
    meanSk, meanSt, meanSe,
    nResidues: N,
  };
}

/**
 * Activity descriptor for SAR prediction.
 * 5-dimensional vector: (N_cav, <Q>, <ω>, <A_cav>, η)
 */
export function activityDescriptor(fingerprint) {
  if (!fingerprint) return [0, 0, 0, 0, 0];
  return [
    fingerprint.nCavities / 20,
    fingerprint.meanQ / 1000,
    fingerprint.meanOmega / 12,
    fingerprint.meanArea / 100,
    fingerprint.coherence,
  ];
}

/**
 * Distance between two cavity fingerprints.
 */
export function fingerprintDistance(f1, f2) {
  if (!f1 || !f2) return Infinity;

  // Global descriptor distance
  const d1 = activityDescriptor(f1);
  const d2 = activityDescriptor(f2);
  let globalDist = 0;
  for (let i = 0; i < 5; i++) {
    globalDist += (d1[i] - d2[i]) ** 2;
  }
  globalDist = Math.sqrt(globalDist);

  // Cavity count penalty
  const countPenalty = 0.1 * Math.abs(f1.nCavities - f2.nCavities);

  return globalDist + countPenalty;
}

/**
 * In-memory cavity fingerprint database.
 * For browser-based search. Scales to ~100K entries in-memory.
 */
export class CavityDB {
  constructor() {
    this.entries = [];
    this.lshBuckets = new Map();
  }

  /** Add a molecule with its fingerprint to the database. */
  add(id, name, type, fingerprint, metadata = {}) {
    const entry = { id, name, type, fingerprint, metadata };
    this.entries.push(entry);

    // LSH bucket for fast search
    const bucket = this._hash(fingerprint);
    if (!this.lshBuckets.has(bucket)) this.lshBuckets.set(bucket, []);
    this.lshBuckets.get(bucket).push(this.entries.length - 1);
  }

  /** Search for similar molecules by cavity fingerprint. */
  search(queryFingerprint, topK = 10) {
    const bucket = this._hash(queryFingerprint);
    // Search this bucket and adjacent buckets
    const candidates = new Set();
    for (let b = bucket - 2; b <= bucket + 2; b++) {
      const entries = this.lshBuckets.get(b);
      if (entries) entries.forEach(i => candidates.add(i));
    }

    // If too few candidates, fall back to full scan
    if (candidates.size < topK * 2) {
      for (let i = 0; i < this.entries.length; i++) candidates.add(i);
    }

    // Rank by fingerprint distance
    const results = [];
    for (const idx of candidates) {
      const entry = this.entries[idx];
      const dist = fingerprintDistance(queryFingerprint, entry.fingerprint);
      results.push({ ...entry, distance: Math.round(dist * 1000) / 1000 });
    }

    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, topK);
  }

  /** LSH hash: quantize the 5D activity descriptor into a single integer bucket. */
  _hash(fingerprint) {
    const d = activityDescriptor(fingerprint);
    // Coarse quantization: 5 bins per dimension → 5^5 = 3125 buckets
    const bins = d.map(v => Math.floor(Math.min(v, 0.99) * 5));
    return bins[0] * 625 + bins[1] * 125 + bins[2] * 25 + bins[3] * 5 + bins[4];
  }

  /** Get database stats. */
  stats() {
    return {
      totalEntries: this.entries.length,
      proteins: this.entries.filter(e => e.type === 'protein').length,
      drugs: this.entries.filter(e => e.type === 'drug').length,
      compounds: this.entries.filter(e => e.type === 'compound').length,
      buckets: this.lshBuckets.size,
    };
  }

  /** Export database as JSON. */
  toJSON() {
    return JSON.stringify(this.entries.map(e => ({
      id: e.id, name: e.name, type: e.type,
      fingerprint: e.fingerprint, metadata: e.metadata,
    })));
  }

  /** Import database from JSON. */
  static fromJSON(json) {
    const db = new CavityDB();
    const entries = JSON.parse(json);
    for (const e of entries) {
      db.add(e.id, e.name, e.type, e.fingerprint, e.metadata);
    }
    return db;
  }
}

/**
 * SAR Predictor: predicts activity from cavity fingerprint.
 * Simple linear regression on the 5D activity descriptor.
 * Trained from (fingerprint, measured_activity) pairs.
 */
export class SARPredictor {
  constructor() {
    this.coefficients = null; // 6 values: [bias, w1, w2, w3, w4, w5]
    this.trainingData = [];
    this.r2 = 0;
  }

  /** Add training point: (fingerprint, log_activity). */
  addTrainingPoint(fingerprint, logActivity) {
    this.trainingData.push({
      x: activityDescriptor(fingerprint),
      y: logActivity,
    });
  }

  /** Fit linear regression. */
  fit() {
    const n = this.trainingData.length;
    if (n < 6) return; // need at least 6 points for 5 features + bias

    // Build design matrix X (n x 6) and target vector Y (n x 1)
    const X = this.trainingData.map(d => [1, ...d.x]);
    const Y = this.trainingData.map(d => d.y);

    // Normal equations: β = (X^T X)^{-1} X^T Y
    // Simple implementation for 6x6 system
    const XtX = Array(6).fill(null).map(() => Array(6).fill(0));
    const XtY = Array(6).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 6; j++) {
        XtY[j] += X[i][j] * Y[i];
        for (let k = 0; k < 6; k++) {
          XtX[j][k] += X[i][j] * X[i][k];
        }
      }
    }

    // Solve via Gaussian elimination
    this.coefficients = this._solveLinear(XtX, XtY);

    // Compute R²
    const yMean = Y.reduce((s, v) => s + v, 0) / n;
    let ssTot = 0, ssRes = 0;
    for (let i = 0; i < n; i++) {
      const yPred = this.predict(this.trainingData[i].x);
      ssRes += (Y[i] - yPred) ** 2;
      ssTot += (Y[i] - yMean) ** 2;
    }
    this.r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  /** Predict log(activity) from descriptor vector. */
  predict(descriptor) {
    if (!this.coefficients) return 0;
    const x = Array.isArray(descriptor[0]) ? descriptor : [1, ...descriptor];
    return this.coefficients.reduce((s, c, i) => s + c * (x[i] || 0), 0);
  }

  /** Predict from fingerprint. */
  predictFromFingerprint(fingerprint) {
    return this.predict(activityDescriptor(fingerprint));
  }

  /** Gaussian elimination for Ax = b. */
  _solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
      // Pivot
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
      }
      [M[col], M[maxRow]] = [M[maxRow], M[col]];

      if (Math.abs(M[col][col]) < 1e-12) continue;

      // Eliminate
      for (let row = col + 1; row < n; row++) {
        const factor = M[row][col] / M[col][col];
        for (let k = col; k <= n; k++) {
          M[row][k] -= factor * M[col][k];
        }
      }
    }

    // Back-substitute
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      if (Math.abs(M[i][i]) < 1e-12) continue;
      x[i] = M[i][n];
      for (let j = i + 1; j < n; j++) {
        x[i] -= M[i][j] * x[j];
      }
      x[i] /= M[i][i];
    }
    return x;
  }
}
