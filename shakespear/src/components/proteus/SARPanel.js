/**
 * SAR Panel + Database Search Panel
 * ==================================
 * Structure-Activity Relationships and cavity fingerprint database search.
 * All computation runs in the browser via GPU shaders + JS.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { extractFingerprint, activityDescriptor, fingerprintDistance,
         CavityDB, SARPredictor } from './CavityDatabase';

/** SAR comparison between two molecules. */
function SARComparison({ fpA, fpB, nameA, nameB }) {
  if (!fpA || !fpB) return null;

  const descA = activityDescriptor(fpA);
  const descB = activityDescriptor(fpB);
  const dist = fingerprintDistance(fpA, fpB);

  const labels = ['N_cav', '<Q>', '<ω>', '<A>', 'η'];
  const diffs = descA.map((a, i) => descB[i] - a);
  const maxDiff = Math.max(...diffs.map(Math.abs), 0.001);

  return (
    <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
      <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-3 font-mono">
        SAR Comparison
      </div>
      <div className="flex justify-between text-xs text-dark/60 dark:text-light/50 mb-2 font-mono">
        <span>{nameA}</span>
        <span className="text-primaryDark">d = {dist.toFixed(3)}</span>
        <span>{nameB}</span>
      </div>
      {labels.map((label, i) => {
        const pct = ((diffs[i] / maxDiff) * 50 + 50);
        const color = diffs[i] > 0.01 ? '#58E6D9' : diffs[i] < -0.01 ? '#E53935' : '#666';
        return (
          <div key={label} className="flex items-center gap-2 mb-1">
            <span className="w-10 text-[10px] text-dark/40 dark:text-light/30 font-mono">{label}</span>
            <span className="w-12 text-[10px] text-right text-dark/50 dark:text-light/40 font-mono">
              {descA[i].toFixed(2)}
            </span>
            <div className="flex-1 h-2 bg-dark/5 dark:bg-dark/40 rounded-full overflow-hidden relative">
              <div className="absolute top-0 left-1/2 w-px h-full bg-dark/20 dark:bg-light/20" />
              <div className="h-full rounded-full transition-all duration-300"
                   style={{
                     width: `${Math.abs(pct - 50)}%`,
                     marginLeft: pct < 50 ? `${pct}%` : '50%',
                     backgroundColor: color,
                   }} />
            </div>
            <span className="w-12 text-[10px] text-dark/50 dark:text-light/40 font-mono">
              {descB[i].toFixed(2)}
            </span>
            <span className="w-8 text-[10px] font-mono" style={{ color }}>
              {diffs[i] > 0 ? '+' : ''}{diffs[i].toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Database search results. */
function SearchResults({ results, onSelect }) {
  if (!results || results.length === 0) return null;

  return (
    <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
      <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-3 font-mono">
        Search Results ({results.length})
      </div>
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {results.map((r, i) => (
          <button key={r.id}
            onClick={() => onSelect && onSelect(r)}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded
                       hover:bg-primaryDark/10 transition-colors text-left">
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-dark/30 dark:text-light/25 font-mono w-4">{i + 1}</span>
              <span className="text-xs text-dark/70 dark:text-light/60 font-mono">{r.name}</span>
              <span className="text-[9px] text-dark/30 dark:text-light/25">{r.type}</span>
            </div>
            <span className="text-[10px] text-primaryDark font-mono">{r.distance.toFixed(3)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/** Full SAR + Search Panel. */
export default function SARPanel({ engine, sequence, fingerprint }) {
  const dbRef = useRef(null);
  const sarRef = useRef(null);
  const [searchResults, setSearchResults] = useState(null);
  const [compareFingerprint, setCompareFingerprint] = useState(null);
  const [compareName, setCompareName] = useState('');
  const [dbStats, setDbStats] = useState({ totalEntries: 0 });
  const [prediction, setPrediction] = useState(null);

  // Initialize database with built-in examples
  useEffect(() => {
    const db = new CavityDB();

    // Pre-populate with example proteins (fingerprints from known properties)
    const examples = [
      { id: 'crambin', name: 'Crambin', type: 'protein',
        fp: { nCavities: 3, meanQ: 520, meanOmega: 6.2, meanArea: 35, coherence: 0.72,
              nEdges: 8, meanSk: 0.54, meanSt: 0.37, meanSe: 0.19, nResidues: 46, cavities: [] }},
      { id: 'lysozyme', name: 'Lysozyme', type: 'protein',
        fp: { nCavities: 7, meanQ: 680, meanOmega: 5.8, meanArea: 48, coherence: 0.81,
              nEdges: 22, meanSk: 0.45, meanSt: 0.42, meanSe: 0.33, nResidues: 129, cavities: [] }},
      { id: 'myoglobin', name: 'Myoglobin', type: 'protein',
        fp: { nCavities: 5, meanQ: 720, meanOmega: 5.5, meanArea: 52, coherence: 0.85,
              nEdges: 18, meanSk: 0.46, meanSt: 0.47, meanSe: 0.38, nResidues: 153, cavities: [] }},
      { id: 'sod1', name: 'SOD1', type: 'protein',
        fp: { nCavities: 6, meanQ: 590, meanOmega: 5.9, meanArea: 41, coherence: 0.76,
              nEdges: 15, meanSk: 0.46, meanSt: 0.39, meanSe: 0.33, nResidues: 153, cavities: [] }},
      { id: 'trypsin', name: 'Trypsin', type: 'protein',
        fp: { nCavities: 8, meanQ: 750, meanOmega: 5.6, meanArea: 55, coherence: 0.83,
              nEdges: 28, meanSk: 0.44, meanSt: 0.45, meanSe: 0.35, nResidues: 223, cavities: [] },
        metadata: { function: 'Serine protease' }},
      { id: 'chymotrypsin', name: 'Chymotrypsin', type: 'protein',
        fp: { nCavities: 8, meanQ: 730, meanOmega: 5.7, meanArea: 53, coherence: 0.82,
              nEdges: 26, meanSk: 0.45, meanSt: 0.44, meanSe: 0.34, nResidues: 241, cavities: [] },
        metadata: { function: 'Serine protease' }},
      { id: 'ca2', name: 'Carbonic Anhydrase II', type: 'protein',
        fp: { nCavities: 9, meanQ: 810, meanOmega: 5.4, meanArea: 58, coherence: 0.87,
              nEdges: 35, meanSk: 0.43, meanSt: 0.45, meanSe: 0.36, nResidues: 259, cavities: [] },
        metadata: { function: 'Zinc metalloenzyme' }},
      { id: 'insulin', name: 'Insulin', type: 'protein',
        fp: { nCavities: 2, meanQ: 450, meanOmega: 6.5, meanArea: 28, coherence: 0.69,
              nEdges: 5, meanSk: 0.52, meanSt: 0.47, meanSe: 0.26, nResidues: 51, cavities: [] },
        metadata: { function: 'Hormone' }},
      { id: 'groel', name: 'GroEL subunit', type: 'protein',
        fp: { nCavities: 12, meanQ: 880, meanOmega: 5.2, meanArea: 65, coherence: 0.89,
              nEdges: 48, meanSk: 0.42, meanSt: 0.46, meanSe: 0.37, nResidues: 547, cavities: [] },
        metadata: { function: 'Chaperonin' }},
      { id: 'hemoglobin', name: 'Hemoglobin', type: 'protein',
        fp: { nCavities: 10, meanQ: 780, meanOmega: 5.5, meanArea: 55, coherence: 0.86,
              nEdges: 38, meanSk: 0.44, meanSt: 0.46, meanSe: 0.36, nResidues: 574, cavities: [] },
        metadata: { function: 'Oxygen transport' }},
    ];

    for (const ex of examples) {
      db.add(ex.id, ex.name, ex.type, ex.fp, ex.metadata || {});
    }

    dbRef.current = db;
    setDbStats(db.stats());

    // Initialize SAR predictor with mock training data
    const sar = new SARPredictor();
    examples.forEach((ex, i) => {
      // Mock activity: larger, more cavities, higher coherence → more active
      const mockActivity = -Math.log10(
        (20 - ex.fp.nCavities) * 0.1 + (1 - ex.fp.coherence) * 5 + Math.random() * 0.5
      );
      sar.addTrainingPoint(ex.fp, mockActivity);
    });
    sar.fit();
    sarRef.current = sar;
  }, []);

  // Search when fingerprint changes
  useEffect(() => {
    if (!fingerprint || !dbRef.current) return;
    const results = dbRef.current.search(fingerprint, 5);
    setSearchResults(results);

    // Predict activity
    if (sarRef.current && sarRef.current.coefficients) {
      const pred = sarRef.current.predictFromFingerprint(fingerprint);
      setPrediction({
        logActivity: Math.round(pred * 100) / 100,
        ic50_nm: Math.round(Math.pow(10, -pred) * 1e9),
        r2: Math.round(sarRef.current.r2 * 100) / 100,
      });
    }
  }, [fingerprint]);

  // Add current molecule to database
  const handleAddToDB = useCallback(() => {
    if (!fingerprint || !dbRef.current) return;
    const id = `user_${Date.now()}`;
    const name = sequence.substring(0, 8) + '...';
    dbRef.current.add(id, name, 'protein', fingerprint, {});
    setDbStats(dbRef.current.stats());
  }, [fingerprint, sequence]);

  // Export database
  const handleExport = useCallback(() => {
    if (!dbRef.current) return;
    const json = dbRef.current.toJSON();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'shakespear_cavity_db.json';
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Import database
  const handleImport = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        dbRef.current = CavityDB.fromJSON(ev.target.result);
        setDbStats(dbRef.current.stats());
      } catch (err) {
        console.warn('Import failed:', err);
      }
    };
    reader.readAsText(file);
  }, []);

  return (
    <div className="mt-4 space-y-3">
      {/* Fingerprint Summary */}
      {fingerprint && (
        <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
          <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-2 font-mono">
            Cavity Fingerprint
          </div>
          <div className="grid grid-cols-5 gap-2 text-center">
            {[
              ['Cavities', fingerprint.nCavities],
              ['<Q>', Math.round(fingerprint.meanQ)],
              ['Edges', fingerprint.nEdges],
              ['η', fingerprint.coherence.toFixed(2)],
              ['<A>', fingerprint.meanArea.toFixed(0)],
            ].map(([label, val]) => (
              <div key={label}>
                <div className="text-[9px] text-dark/30 dark:text-light/25 font-mono">{label}</div>
                <div className="text-sm font-bold text-primary dark:text-primaryDark font-mono">{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* SAR Prediction */}
      {prediction && (
        <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
          <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-2 font-mono">
            Activity Prediction (SAR)
          </div>
          <div className="flex items-baseline gap-4">
            <div>
              <span className="text-xs text-dark/40 dark:text-light/30 font-mono">Predicted pIC50: </span>
              <span className="text-lg font-bold text-primary dark:text-primaryDark font-mono">
                {prediction.logActivity}
              </span>
            </div>
            <div className="text-[10px] text-dark/30 dark:text-light/25 font-mono">
              R&sup2; = {prediction.r2}
            </div>
          </div>
        </div>
      )}

      {/* Database Search */}
      <SearchResults results={searchResults}
        onSelect={(r) => { setCompareFingerprint(r.fingerprint); setCompareName(r.name); }} />

      {/* SAR Comparison */}
      {compareFingerprint && (
        <SARComparison fpA={fingerprint} fpB={compareFingerprint}
          nameA="Query" nameB={compareName} />
      )}

      {/* Database Controls */}
      <div className="flex flex-wrap gap-2">
        <button onClick={handleAddToDB}
          className="px-3 py-1 text-[10px] rounded border border-dark/20 dark:border-dark/30
                     text-dark/50 dark:text-light/50 hover:border-primaryDark/50 transition-colors font-mono">
          Add to DB
        </button>
        <button onClick={handleExport}
          className="px-3 py-1 text-[10px] rounded border border-dark/20 dark:border-dark/30
                     text-dark/50 dark:text-light/50 hover:border-primaryDark/50 transition-colors font-mono">
          Export DB
        </button>
        <label className="px-3 py-1 text-[10px] rounded border border-dark/20 dark:border-dark/30
                          text-dark/50 dark:text-light/50 hover:border-primaryDark/50 cursor-pointer transition-colors font-mono">
          Import DB
          <input type="file" accept=".json" onChange={handleImport} className="hidden" />
        </label>
        <span className="text-[10px] text-dark/30 dark:text-light/20 self-center font-mono">
          {dbStats.totalEntries} entries
        </span>
      </div>
    </div>
  );
}
