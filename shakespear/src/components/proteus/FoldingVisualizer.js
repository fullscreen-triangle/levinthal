/**
 * Protein Folding Visualizer (D3 Force Graph + Kuramoto Dynamics)
 * ================================================================
 * Each residue is a Kuramoto oscillator. Coupling is from S-entropy distance.
 * Phase-locked pairs form bonds. The force graph IS the harmonic network.
 * You watch the protein fold as edges appear.
 *
 * Runs alongside the shader pipeline -- shares the same S-entropy coordinates.
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import { AA_SENTROPY, AA_INDEX } from './ShaderEngine';

// ============================================================================
// Kuramoto Protein Simulator
// ============================================================================
class KuramotoSimulator {
  constructor(sequence) {
    this.sequence = sequence;
    this.N = sequence.length;
    this.residues = this._initResidues();
    this.K0 = 3.0;
    this.bonds = [];
    this.coherence = 0;
    this.time = 0;
    this.dt = 0.005;
    this.stepCount = 0;
  }

  _initResidues() {
    return this.sequence.split('').map((aa, i) => {
      const idx = AA_INDEX[aa] || 0;
      const [Sk, St, Se] = AA_SENTROPY[idx];
      return {
        id: i, aa, Sk, St, Se,
        omega: Sk * 10 + 1, // natural frequency from hydrophobicity
        phi: Math.random() * 2 * Math.PI,
        x: 0, y: 0, vx: 0, vy: 0, // set by D3
      };
    });
  }

  sDistance(i, j) {
    const a = this.residues[i], b = this.residues[j];
    const dk = a.Sk-b.Sk, dt = a.St-b.St, de = a.Se-b.Se;
    return Math.sqrt(dk*dk + dt*dt + de*de);
  }

  coupling(i, j) {
    const dS = this.sDistance(i, j);
    const sep = Math.abs(i - j);
    const backbone = Math.exp(-sep / 2);
    const d4 = sep-4, d3 = sep-3;
    const helix = 0.8 * Math.exp(-d4*d4) + 0.4 * Math.exp(-d3*d3);
    const tertiary = Math.exp(-dS / 0.2) * (1 - Math.exp(-sep / 8));
    return this.K0 * (0.3*backbone + 0.35*helix + 0.25*tertiary);
  }

  step(stepsPerFrame = 20) {
    const N = this.N;
    for (let s = 0; s < stepsPerFrame; s++) {
      const dPhi = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        dPhi[i] = this.residues[i].omega;
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const K = this.coupling(i, j);
          dPhi[i] += K * Math.sin(this.residues[j].phi - this.residues[i].phi) / N;
        }
      }
      for (let i = 0; i < N; i++) {
        this.residues[i].phi = (this.residues[i].phi + dPhi[i] * this.dt) % (2 * Math.PI);
      }
      this.time += this.dt;
      this.stepCount++;
    }

    // Detect phase-locked bonds
    this.bonds = [];
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const K = this.coupling(i, j);
        if (K < 0.3) continue;
        const dp = Math.abs(this.residues[i].phi - this.residues[j].phi);
        const dpNorm = Math.min(dp, 2 * Math.PI - dp);
        if (dpNorm < 0.3) {
          this.bonds.push({ source: i, target: j, strength: K, phaseDiff: dpNorm });
        }
      }
    }

    // Coherence (Kuramoto order parameter)
    let sr = 0, si = 0;
    for (let i = 0; i < N; i++) {
      sr += Math.cos(this.residues[i].phi);
      si += Math.sin(this.residues[i].phi);
    }
    this.coherence = Math.sqrt(sr*sr + si*si) / N;
  }

  reset() {
    this.residues.forEach(r => { r.phi = Math.random() * 2 * Math.PI; });
    this.bonds = [];
    this.coherence = 0;
    this.time = 0;
    this.stepCount = 0;
  }
}

// ============================================================================
// Color utilities
// ============================================================================
function residueColor(r) {
  // S-entropy → RGB
  const R = Math.round(r.Sk * 255);
  const G = Math.round(r.St * 200);
  const B = Math.round(r.Se * 255);
  return `rgb(${R},${G},${B})`;
}

function phaseColor(phi) {
  // Phase → hue wheel
  const hue = (phi / (2 * Math.PI)) * 360;
  return `hsl(${hue}, 80%, 55%)`;
}

function bondColor(bond) {
  // Strength → cyan to white
  const t = Math.min(bond.strength / 3, 1);
  const r = Math.round(88 + t * 167);
  const g = Math.round(230 + t * 25);
  const b = Math.round(217 + t * 38);
  return `rgb(${r},${g},${b})`;
}

// ============================================================================
// React Component
// ============================================================================
export default function FoldingVisualizer({ sequence, width = 500, height = 400 }) {
  const canvasRef = useRef(null);
  const simRef = useRef(null);
  const animRef = useRef(null);
  const nodesRef = useRef([]);

  const [running, setRunning] = useState(false);
  const [coherence, setCoherence] = useState(0);
  const [bondCount, setBondCount] = useState(0);
  const [colorMode, setColorMode] = useState('sentropy'); // sentropy | phase | type
  const [speed, setSpeed] = useState(20);

  // Initialize simulator when sequence changes
  useEffect(() => {
    if (!sequence || sequence.length < 2) return;
    const sim = new KuramotoSimulator(sequence);
    simRef.current = sim;

    // Initialize positions in a circle
    const cx = width / 2, cy = height / 2;
    const radius = Math.min(width, height) * 0.35;
    sim.residues.forEach((r, i) => {
      const angle = (i / sim.N) * 2 * Math.PI - Math.PI / 2;
      r.x = cx + radius * Math.cos(angle);
      r.y = cy + radius * Math.sin(angle);
      r.vx = 0; r.vy = 0;
    });
    nodesRef.current = sim.residues;

    setCoherence(0);
    setBondCount(0);
    setRunning(false);
  }, [sequence, width, height]);

  // Force simulation (simple spring model, no D3 dependency)
  const applyForces = useCallback(() => {
    const sim = simRef.current;
    if (!sim) return;
    const nodes = sim.residues;
    const N = nodes.length;
    const cx = width / 2, cy = height / 2;

    // Center gravity
    for (let i = 0; i < N; i++) {
      nodes[i].vx += (cx - nodes[i].x) * 0.0005;
      nodes[i].vy += (cy - nodes[i].y) * 0.0005;
    }

    // Repulsion between all nodes
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx*dx + dy*dy) || 1;
        const force = -200 / (dist * dist);
        const fx = force * dx / dist;
        const fy = force * dy / dist;
        nodes[i].vx += fx; nodes[i].vy += fy;
        nodes[j].vx -= fx; nodes[j].vy -= fy;
      }
    }

    // Backbone springs (sequential neighbors)
    for (let i = 0; i < N - 1; i++) {
      const dx = nodes[i+1].x - nodes[i].x;
      const dy = nodes[i+1].y - nodes[i].y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const target = 20;
      const force = (dist - target) * 0.05;
      const fx = force * dx / dist;
      const fy = force * dy / dist;
      nodes[i].vx += fx; nodes[i].vy += fy;
      nodes[i+1].vx -= fx; nodes[i+1].vy -= fy;
    }

    // Bond springs (phase-locked pairs attract)
    for (const bond of sim.bonds) {
      const a = nodes[bond.source], b = nodes[bond.target];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 1;
      const target = 25;
      const force = (dist - target) * 0.03 * bond.strength;
      const fx = force * dx / dist;
      const fy = force * dy / dist;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }

    // Integrate
    const damping = 0.92;
    for (let i = 0; i < N; i++) {
      nodes[i].vx *= damping;
      nodes[i].vy *= damping;
      nodes[i].x += nodes[i].vx;
      nodes[i].y += nodes[i].vy;
      // Bounds
      nodes[i].x = Math.max(10, Math.min(width - 10, nodes[i].x));
      nodes[i].y = Math.max(10, Math.min(height - 10, nodes[i].y));
    }
  }, [width, height]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let active = true;
    const draw = () => {
      if (!active) return;
      const sim = simRef.current;
      if (!sim) { animRef.current = requestAnimationFrame(draw); return; }

      // Step Kuramoto if running
      if (running) {
        sim.step(speed);
        setCoherence(sim.coherence);
        setBondCount(sim.bonds.length);
      }

      // Apply forces
      applyForces();

      const nodes = sim.residues;
      const N = nodes.length;

      // Clear
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, width, height);

      // Draw backbone chain
      ctx.strokeStyle = 'rgba(88, 230, 217, 0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < N - 1; i++) {
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[i+1].x, nodes[i+1].y);
      }
      ctx.stroke();

      // Draw phase-locked bonds
      for (const bond of sim.bonds) {
        const a = nodes[bond.source], b = nodes[bond.target];
        ctx.strokeStyle = bondColor(bond);
        ctx.lineWidth = 0.5 + bond.strength * 0.8;
        ctx.globalAlpha = 0.4 + bond.strength * 0.2;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Draw residue nodes
      for (let i = 0; i < N; i++) {
        const r = nodes[i];
        const radius = 4 + r.St * 4; // size from volume

        // Color based on mode
        let color;
        if (colorMode === 'phase') {
          color = phaseColor(r.phi);
        } else if (colorMode === 'type') {
          if (r.Se > 0.5) color = '#E53935';       // charged → red
          else if (r.Sk > 0.6) color = '#1E88E5';  // hydrophobic → blue
          else color = '#43A047';                    // polar → green
        } else {
          color = residueColor(r);
        }

        // Glow for phase-locked residues
        const inBond = sim.bonds.some(b => b.source === i || b.target === i);
        if (inBond) {
          ctx.shadowColor = '#58E6D9';
          ctx.shadowBlur = 8;
        }

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(r.x, r.y, radius, 0, 2 * Math.PI);
        ctx.fill();

        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        // Border
        ctx.strokeStyle = inBond ? '#58E6D9' : 'rgba(255,255,255,0.15)';
        ctx.lineWidth = inBond ? 1.5 : 0.5;
        ctx.stroke();

        // Label (only for small proteins)
        if (N <= 60) {
          ctx.fillStyle = 'rgba(255,255,255,0.6)';
          ctx.font = '7px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(r.aa, r.x, r.y + 2.5);
        }
      }

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => { active = false; if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [running, colorMode, speed, width, height, applyForces]);

  const handleReset = () => {
    if (simRef.current) {
      simRef.current.reset();
      // Re-layout in circle
      const cx = width / 2, cy = height / 2;
      const radius = Math.min(width, height) * 0.35;
      simRef.current.residues.forEach((r, i) => {
        const angle = (i / simRef.current.N) * 2 * Math.PI - Math.PI / 2;
        r.x = cx + radius * Math.cos(angle);
        r.y = cy + radius * Math.sin(angle);
        r.vx = 0; r.vy = 0;
      });
      setCoherence(0);
      setBondCount(0);
    }
  };

  const healthColor = coherence > 0.6 ? '#58E6D9' : coherence > 0.3 ? '#FB8C00' : '#E53935';

  return (
    <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg overflow-hidden">
      {/* Controls */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-dark/10 dark:border-dark/20">
        <button onClick={() => setRunning(!running)}
          className={`px-3 py-1 text-[10px] rounded font-mono uppercase tracking-wider
            ${running ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : 'bg-primaryDark/20 text-primaryDark border border-primaryDark/30'}`}>
          {running ? 'Pause' : 'Fold'}
        </button>
        <button onClick={handleReset}
          className="px-3 py-1 text-[10px] rounded font-mono uppercase tracking-wider
                     bg-dark/5 dark:bg-dark/40 text-dark/50 dark:text-light/40 border border-dark/10 dark:border-dark/20">
          Reset
        </button>

        <span className="mx-1 text-dark/10 dark:text-light/10">|</span>

        {['sentropy', 'phase', 'type'].map(m => (
          <button key={m} onClick={() => setColorMode(m)}
            className={`px-2 py-0.5 text-[9px] rounded font-mono
              ${colorMode === m ? 'text-primaryDark' : 'text-dark/30 dark:text-light/25'}`}>
            {m}
          </button>
        ))}

        <span className="mx-1 text-dark/10 dark:text-light/10">|</span>

        <input type="range" min="1" max="100" value={speed}
          onChange={e => setSpeed(parseInt(e.target.value))}
          className="w-16 h-1 accent-primaryDark" />
        <span className="text-[9px] text-dark/30 dark:text-light/25 font-mono">{speed}x</span>

        <div className="flex-1" />

        {/* Coherence readout */}
        <span className="text-[10px] font-mono" style={{ color: healthColor }}>
          r = {coherence.toFixed(3)}
        </span>
        <span className="text-[10px] font-mono text-dark/30 dark:text-light/25">
          {bondCount} bonds
        </span>
      </div>

      {/* Canvas */}
      <canvas ref={canvasRef} width={width} height={height}
        style={{ width, height, display: 'block' }} />
    </div>
  );
}
