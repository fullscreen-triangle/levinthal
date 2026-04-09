/**
 * PROTEUS Instrument Component
 * ============================
 * React component that wraps the ShaderEngine and provides the UI for
 * the five protein observation instruments.
 *
 * No Python backend. The GPU IS the computation.
 * Input: protein sequence (typed or pasted)
 * Output: real-time observation of protein harmonic structure
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import { ShaderEngine, AA_SENTROPY, AA_INDEX } from './ShaderEngine';

// Color maps for visualization
const VIRIDIS = (t) => {
  const r = Math.max(0, Math.min(1, -0.7 + 4.0 * t - 3.3 * t * t));
  const g = Math.max(0, Math.min(1, -0.1 + 1.6 * t - 0.6 * t * t));
  const b = Math.max(0, Math.min(1, 0.3 + 1.5 * t - 2.8 * t * t + 1.0 * t * t * t));
  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
};

const INFERNO = (t) => {
  const r = Math.max(0, Math.min(1, t * 3.0 - 0.5));
  const g = Math.max(0, Math.min(1, t * 2.0 - 0.7));
  const b = Math.max(0, Math.min(1, 0.5 - t * 0.5 + t * t));
  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
};

const COOLWARM = (t) => {
  const r = Math.max(0, Math.min(1, 0.2 + t * 0.8));
  const g = Math.max(0, Math.min(1, 0.3 + 0.4 * (1 - Math.abs(t - 0.5) * 2)));
  const b = Math.max(0, Math.min(1, 1.0 - t * 0.8));
  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
};

const EXAMPLE_SEQUENCES = {
  'Crambin (46 aa)': 'TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN',
  'Insulin B (30 aa)': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
  'Custom': '',
};

export default function ProteusInstrument() {
  const glCanvasRef = useRef(null);
  const displayCanvasRef = useRef(null);
  const engineRef = useRef(null);
  const animFrameRef = useRef(null);

  const [sequence, setSequence] = useState(EXAMPLE_SEQUENCES['Crambin (46 aa)']);
  const [activeView, setActiveView] = useState('spectrum'); // spectrum | coupling | contacts | sentropy
  const [metrics, setMetrics] = useState({ eta: 0, contacts: 0, N: 0 });
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [sentropyData, setSentropyData] = useState([]);

  // Initialize engine
  useEffect(() => {
    const canvas = glCanvasRef.current;
    if (!canvas) return;
    canvas.width = 512;
    canvas.height = 512;

    let engine;
    (async () => {
      try {
        engine = new ShaderEngine(canvas);
        await engine.init();
        engineRef.current = engine;
        setReady(true);
      } catch (e) {
        setError(e.message);
      }
    })();

    return () => {
      if (engine) engine.destroy();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Set sequence and start observation
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !ready) return;

    const cleanSeq = sequence.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
    if (cleanSeq.length < 2) return;

    engine.setSequence(cleanSeq);
    setMetrics(prev => ({ ...prev, N: cleanSeq.length }));

    // Read S-entropy after first pass
    engine.observe(0.001);
    const se = engine.readSEntropy();
    setSentropyData(se);
  }, [sequence, ready]);

  // Animation loop: continuous observation
  useEffect(() => {
    if (!ready) return;
    const engine = engineRef.current;
    const displayCanvas = displayCanvasRef.current;
    if (!engine || !displayCanvas) return;

    const ctx = displayCanvas.getContext('2d');
    const N = engine.N;
    if (N < 2) return;

    const SIZE = 512;
    displayCanvas.width = SIZE;
    displayCanvas.height = SIZE;

    let lastTime = performance.now();

    const loop = (now) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      // Run the shader pipeline (THIS IS THE OBSERVATION)
      engine.observe(dt);

      // Read back the active view texture and render to display canvas
      const gl = engine.gl;
      const fb = engine.framebuffers[activeView === 'contacts' ? 'coherence' :
                                     activeView === 'sentropy' ? 'sentropy' :
                                     activeView];
      if (!fb) { animFrameRef.current = requestAnimationFrame(loop); return; }

      const w = fb.width, h = fb.height;
      const pixels = new Float32Array(w * h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb.fbo);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Map to image data
      const imgData = ctx.createImageData(SIZE, SIZE);
      const scaleX = w / SIZE, scaleY = h / SIZE;

      for (let py = 0; py < SIZE; py++) {
        for (let px = 0; px < SIZE; px++) {
          const sx = Math.min(Math.floor(px * scaleX), w - 1);
          const sy = Math.min(Math.floor(py * scaleY), h - 1);
          const srcIdx = (sy * w + sx) * 4;
          const dstIdx = (py * SIZE + px) * 4;

          let r, g, b;
          if (activeView === 'spectrum') {
            const mag = Math.min(pixels[srcIdx], 1.0);
            [r, g, b] = INFERNO(mag);
          } else if (activeView === 'coupling') {
            const K = Math.min(pixels[srcIdx], 1.0);
            [r, g, b] = INFERNO(K / 5.0);
          } else if (activeView === 'contacts') {
            const contact = pixels[srcIdx];
            const helix = pixels[srcIdx + 1];
            const sheet = pixels[srcIdx + 2];
            r = Math.floor(Math.min(1, helix * 3) * 255);
            g = Math.floor(Math.min(1, sheet * 3) * 100);
            b = Math.floor(Math.min(1, contact) * 255);
          } else { // sentropy
            r = Math.floor(pixels[srcIdx] * 255);
            g = Math.floor(pixels[srcIdx + 1] * 255);
            b = Math.floor(pixels[srcIdx + 2] * 255);
          }

          imgData.data[dstIdx] = r;
          imgData.data[dstIdx + 1] = g;
          imgData.data[dstIdx + 2] = b;
          imgData.data[dstIdx + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);

      // Update metrics every ~10 frames
      if (Math.random() < 0.1) {
        const m = engine.readCoherence();
        setMetrics(prev => ({ ...prev, ...m }));
      }

      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [ready, activeView, sequence]);

  const healthStatus = metrics.eta > 0.8 ? 'Healthy' :
                       metrics.eta > 0.5 ? 'Stressed' : 'Pathological';
  const healthColor = metrics.eta > 0.8 ? '#43A047' :
                      metrics.eta > 0.5 ? '#FB8C00' : '#E53935';

  return (
    <div style={{ background: '#0a0a0a', color: '#e0e0e0', padding: '24px',
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace", minHeight: '100vh' }}>

      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ margin: 0, fontSize: '24px', letterSpacing: '2px', color: '#fff' }}>
          PROTEUS
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: '11px', color: '#888', letterSpacing: '1px' }}>
          Protein Resonance Observation &amp; Trajectory Exploration via Universal Shaders
        </p>
      </div>

      {/* Sequence Input */}
      <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
        {Object.entries(EXAMPLE_SEQUENCES).map(([name, seq]) => (
          <button key={name} onClick={() => seq && setSequence(seq)}
            style={{ padding: '4px 10px', fontSize: '11px', cursor: 'pointer',
                     background: sequence === seq ? '#1565C0' : '#222',
                     color: '#e0e0e0', border: '1px solid #333', borderRadius: '3px' }}>
            {name}
          </button>
        ))}
      </div>
      <textarea value={sequence}
        onChange={e => setSequence(e.target.value)}
        style={{ width: '100%', height: '48px', background: '#111', color: '#4CAF50',
                 border: '1px solid #333', fontFamily: 'inherit', fontSize: '13px',
                 padding: '8px', resize: 'vertical', borderRadius: '3px' }}
        placeholder="Enter protein sequence (one-letter codes)..."
      />

      {error && <div style={{ color: '#E53935', marginTop: '8px' }}>{error}</div>}

      {/* View Selector */}
      <div style={{ margin: '12px 0', display: 'flex', gap: '6px' }}>
        {['spectrum', 'coupling', 'contacts', 'sentropy'].map(v => (
          <button key={v} onClick={() => setActiveView(v)}
            style={{ padding: '6px 14px', fontSize: '11px', cursor: 'pointer',
                     background: activeView === v ? '#1565C0' : '#1a1a1a',
                     color: '#e0e0e0', border: '1px solid #333', borderRadius: '3px',
                     textTransform: 'uppercase', letterSpacing: '1px' }}>
            {v === 'spectrum' ? 'Spectrometer' :
             v === 'coupling' ? 'Resonator' :
             v === 'contacts' ? 'Diagnostician' : 'S-Entropy'}
          </button>
        ))}
      </div>

      {/* Main Display */}
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>

        {/* Observation Canvas */}
        <div style={{ position: 'relative' }}>
          <canvas ref={displayCanvasRef}
            style={{ width: '512px', height: '512px', border: '1px solid #333',
                     borderRadius: '3px', imageRendering: 'pixelated' }} />
          <div style={{ position: 'absolute', top: '8px', left: '8px',
                        background: 'rgba(0,0,0,0.7)', padding: '4px 8px',
                        borderRadius: '2px', fontSize: '10px', color: '#888' }}>
            {activeView.toUpperCase()} | {metrics.N} residues | GPU observation
          </div>
        </div>

        {/* Metrics Panel */}
        <div style={{ minWidth: '200px', flex: 1 }}>

          {/* Coherence Meter */}
          <div style={{ background: '#111', padding: '16px', borderRadius: '3px',
                        border: '1px solid #222', marginBottom: '12px' }}>
            <div style={{ fontSize: '10px', color: '#888', textTransform: 'uppercase',
                          letterSpacing: '1px', marginBottom: '8px' }}>
              Coherence Order Parameter
            </div>
            <div style={{ fontSize: '36px', fontWeight: 'bold', color: healthColor }}>
              {metrics.eta.toFixed(3)}
            </div>
            <div style={{ fontSize: '12px', color: healthColor, marginTop: '4px' }}>
              {healthStatus}
            </div>
            <div style={{ marginTop: '8px', height: '4px', background: '#222',
                          borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ width: `${metrics.eta * 100}%`, height: '100%',
                            background: healthColor, transition: 'width 0.3s' }} />
            </div>
          </div>

          {/* Contact Count */}
          <div style={{ background: '#111', padding: '16px', borderRadius: '3px',
                        border: '1px solid #222', marginBottom: '12px' }}>
            <div style={{ fontSize: '10px', color: '#888', textTransform: 'uppercase',
                          letterSpacing: '1px', marginBottom: '8px' }}>
              Predicted Long-Range Contacts
            </div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1E88E5' }}>
              {metrics.contacts}
            </div>
          </div>

          {/* Sequence Composition */}
          <div style={{ background: '#111', padding: '16px', borderRadius: '3px',
                        border: '1px solid #222', marginBottom: '12px' }}>
            <div style={{ fontSize: '10px', color: '#888', textTransform: 'uppercase',
                          letterSpacing: '1px', marginBottom: '8px' }}>
              S-Entropy Composition
            </div>
            {sentropyData.length > 0 && (() => {
              const avgSk = sentropyData.reduce((s, d) => s + d.Sk, 0) / sentropyData.length;
              const avgSt = sentropyData.reduce((s, d) => s + d.St, 0) / sentropyData.length;
              const avgSe = sentropyData.reduce((s, d) => s + d.Se, 0) / sentropyData.length;
              return (
                <div style={{ fontSize: '12px', lineHeight: '1.8' }}>
                  <div>S<sub>k</sub> = <span style={{ color: '#E53935' }}>{avgSk.toFixed(3)}</span></div>
                  <div>S<sub>t</sub> = <span style={{ color: '#43A047' }}>{avgSt.toFixed(3)}</span></div>
                  <div>S<sub>e</sub> = <span style={{ color: '#1E88E5' }}>{avgSe.toFixed(3)}</span></div>
                </div>
              );
            })()}
          </div>

          {/* Pipeline Info */}
          <div style={{ background: '#111', padding: '16px', borderRadius: '3px',
                        border: '1px solid #222', fontSize: '10px', color: '#666' }}>
            <div>Pipeline: Pass 1 → 3 → 6 → 7</div>
            <div>GPU Memory: ~{Math.ceil((metrics.N * metrics.N * 4 * 4 * 3 + metrics.N * 16) / 1024)} KB</div>
            <div>Backend: None (pure WebGL2)</div>
            <div>Observation = Computation</div>
          </div>
        </div>
      </div>

      {/* Hidden WebGL canvas (computation only) */}
      <canvas ref={glCanvasRef} style={{ display: 'none' }} />
    </div>
  );
}
