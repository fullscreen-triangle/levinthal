/**
 * FoldingSimulation -- Full-page protein folding visualization
 * =============================================================
 * Kuramoto oscillator network + DFT coupling spectrum + contact map + secondary structure.
 * Self-contained: no dependency on ShaderEngine or ProteusViewer.
 *
 * Four panels:
 *   1. Force graph: residues as oscillators, bonds form via phase-locking
 *   2. Coupling spectrum: DFT of coupling history (synthetic 2D-IR)
 *   3. Contact map: predicted from spectrum thresholding
 *   4. Metrics: coherence, bonds, secondary structure, partition state
 */

import { useRef, useState, useEffect, useCallback } from 'react'

// Amino acid S-entropy coordinates (same as ShaderEngine)
const AA = {
  A:[0.700,0.170,0.000], C:[0.778,0.289,0.100], D:[0.111,0.304,1.000],
  E:[0.111,0.467,1.000], F:[0.811,0.774,0.000], G:[0.456,0.000,0.000],
  H:[0.144,0.555,0.600], I:[1.000,0.636,0.000], K:[0.067,0.647,1.000],
  L:[0.922,0.636,0.000], M:[0.711,0.613,0.000], N:[0.111,0.322,0.500],
  P:[0.322,0.373,0.000], Q:[0.111,0.499,0.500], R:[0.000,0.676,1.000],
  S:[0.411,0.172,0.300], T:[0.422,0.334,0.300], W:[0.400,1.000,0.100],
  Y:[0.356,0.796,0.200], V:[0.967,0.476,0.000],
}

const PROTEINS = {
  'Villin (35 aa)':  'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
  'Insulin B (30 aa)': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
  'Crambin (46 aa)': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
  'BPTI (58 aa)':    'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
}

// ---- Kuramoto Simulator ----
class Simulator {
  constructor(seq) {
    this.seq = seq
    this.N = seq.length
    this.K0 = 3.0
    this.dt = 0.005
    this.time = 0
    this.coherence = 0
    this.bonds = []
    this.couplingHistory = []
    this.maxHistory = 128

    this.res = seq.split('').map((aa, i) => {
      const s = AA[aa] || [0.5, 0.5, 0.5]
      return {
        id: i, aa, Sk: s[0], St: s[1], Se: s[2],
        omega: s[0] * 10 + 1,
        phi: Math.random() * Math.PI * 2,
        x: 0, y: 0, vx: 0, vy: 0,
      }
    })
  }

  coupling(i, j) {
    const a = this.res[i], b = this.res[j]
    const dk = a.Sk-b.Sk, dt = a.St-b.St, de = a.Se-b.Se
    const dS = Math.sqrt(dk*dk + dt*dt + de*de)
    const sep = Math.abs(i - j)
    const d4 = sep-4, d3 = sep-3
    const backbone = Math.exp(-sep / 2)
    const helix = 0.8 * Math.exp(-d4*d4) + 0.4 * Math.exp(-d3*d3)
    const tertiary = Math.exp(-dS / 0.2) * (1 - Math.exp(-sep / 8))
    return this.K0 * (0.3*backbone + 0.35*helix + 0.25*tertiary)
  }

  step(n) {
    const N = this.N
    for (let s = 0; s < n; s++) {
      // Build coupling matrix and store
      const K = new Float32Array(N * N)
      for (let i = 0; i < N; i++)
        for (let j = i+1; j < N; j++) {
          const k = this.coupling(i, j)
          K[i*N+j] = k; K[j*N+i] = k
        }

      if (this.couplingHistory.length >= this.maxHistory) this.couplingHistory.shift()
      this.couplingHistory.push(K)

      // Kuramoto ODE
      const dPhi = new Float64Array(N)
      for (let i = 0; i < N; i++) {
        dPhi[i] = this.res[i].omega
        for (let j = 0; j < N; j++) {
          if (i === j) continue
          dPhi[i] += K[i*N+j] * Math.sin(this.res[j].phi - this.res[i].phi) / N
        }
      }
      for (let i = 0; i < N; i++)
        this.res[i].phi = (this.res[i].phi + dPhi[i] * this.dt) % (Math.PI * 2)

      this.time += this.dt
    }

    // Detect bonds
    this.bonds = []
    const K = this.couplingHistory[this.couplingHistory.length - 1]
    for (let i = 0; i < N; i++)
      for (let j = i+1; j < N; j++) {
        const k = K[i*N+j]
        if (k < 0.3) continue
        const dp = Math.abs(this.res[i].phi - this.res[j].phi)
        const dpN = Math.min(dp, Math.PI*2 - dp)
        if (dpN < 0.3) this.bonds.push({ s: i, t: j, k, dp: dpN })
      }

    // Coherence
    let sr = 0, si = 0
    for (let i = 0; i < N; i++) {
      sr += Math.cos(this.res[i].phi)
      si += Math.sin(this.res[i].phi)
    }
    this.coherence = Math.sqrt(sr*sr + si*si) / N
  }

  // DFT of coupling history
  computeSpectrum() {
    const N = this.N, T = this.couplingHistory.length
    if (T < 4) return null
    const spec = new Float32Array(N * N)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        let re = 0, im = 0
        for (let t = 0; t < T; t++) {
          const v = this.couplingHistory[t][i*N+j]
          const angle = 2 * Math.PI * t / T
          re += v * Math.cos(angle)
          im += v * Math.sin(angle)
        }
        spec[i*N+j] = Math.sqrt(re*re + im*im) / T
      }
    return spec
  }

  // Secondary structure from coupling
  secondaryStructure() {
    const ss = new Array(this.N).fill('C')
    const K = this.couplingHistory.length > 0 ? this.couplingHistory[this.couplingHistory.length-1] : null
    if (!K) return ss
    for (let i = 0; i < this.N - 4; i++)
      if (K[i*this.N + i+4] > 1.5)
        for (let k = 0; k < 5 && i+k < this.N; k++) ss[i+k] = 'H'
    for (let i = 0; i < this.N; i++)
      for (let j = i+8; j < this.N; j++)
        if (K[i*this.N+j] > 2.0) { ss[i] = 'E'; ss[j] = 'E' }
    return ss
  }

  reset() {
    this.res.forEach(r => { r.phi = Math.random() * Math.PI * 2 })
    this.bonds = []; this.coherence = 0; this.time = 0
    this.couplingHistory = []
  }
}

// ---- Inferno colormap ----
function inferno(t) {
  t = Math.max(0, Math.min(1, t))
  const r = Math.floor(Math.min(1, Math.max(0, t*3 - 0.5)) * 255)
  const g = Math.floor(Math.min(1, Math.max(0, t*2 - 0.7)) * 255)
  const b = Math.floor(Math.min(1, Math.max(0, 0.5 - t*0.3)) * 255)
  return [r, g, b]
}

// ---- Main Component ----
export default function FoldingSimulation() {
  const netRef = useRef(null)    // force graph canvas
  const specRef = useRef(null)   // spectrum canvas
  const contRef = useRef(null)   // contact map canvas
  const simRef = useRef(null)
  const animRef = useRef(null)

  const [seq, setSeq] = useState(PROTEINS['Crambin (46 aa)'])
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(20)
  const [coupling, setCoupling] = useState(3.0)
  const [coherence, setCoherence] = useState(0)
  const [bondCount, setBondCount] = useState(0)
  const [colorMode, setColorMode] = useState('sentropy')
  const [ss, setSs] = useState([])
  const [specComputed, setSpecComputed] = useState(false)

  const W_NET = 480, H_NET = 400
  const W_SPEC = 220, H_SPEC = 220
  const W_CONT = 220, H_CONT = 220

  // Init simulator
  useEffect(() => {
    if (!seq || seq.length < 2) return
    const sim = new Simulator(seq)
    simRef.current = sim
    // Circle layout
    const cx = W_NET/2, cy = H_NET/2, rad = Math.min(W_NET, H_NET) * 0.35
    sim.res.forEach((r, i) => {
      const a = (i / sim.N) * Math.PI * 2 - Math.PI/2
      r.x = cx + rad * Math.cos(a)
      r.y = cy + rad * Math.sin(a)
    })
    setCoherence(0); setBondCount(0); setSs([]); setSpecComputed(false)
    setRunning(false)
  }, [seq])

  // Update coupling strength
  useEffect(() => { if (simRef.current) simRef.current.K0 = coupling }, [coupling])

  // Force layout (no D3 dependency)
  const applyForces = useCallback(() => {
    const sim = simRef.current
    if (!sim) return
    const nodes = sim.res, N = nodes.length
    const cx = W_NET/2, cy = H_NET/2

    for (let i = 0; i < N; i++) {
      nodes[i].vx += (cx - nodes[i].x) * 0.0005
      nodes[i].vy += (cy - nodes[i].y) * 0.0005
    }
    for (let i = 0; i < N; i++)
      for (let j = i+1; j < N; j++) {
        const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y
        const dist = Math.sqrt(dx*dx + dy*dy) || 1
        const f = -200 / (dist * dist)
        nodes[i].vx += f*dx/dist; nodes[i].vy += f*dy/dist
        nodes[j].vx -= f*dx/dist; nodes[j].vy -= f*dy/dist
      }
    for (let i = 0; i < N-1; i++) {
      const dx = nodes[i+1].x - nodes[i].x, dy = nodes[i+1].y - nodes[i].y
      const dist = Math.sqrt(dx*dx + dy*dy) || 1
      const f = (dist - 20) * 0.05
      nodes[i].vx += f*dx/dist; nodes[i].vy += f*dy/dist
      nodes[i+1].vx -= f*dx/dist; nodes[i+1].vy -= f*dy/dist
    }
    for (const b of sim.bonds) {
      const a = nodes[b.s], c = nodes[b.t]
      const dx = c.x - a.x, dy = c.y - a.y
      const dist = Math.sqrt(dx*dx + dy*dy) || 1
      const f = (dist - 25) * 0.03 * b.k
      a.vx += f*dx/dist; a.vy += f*dy/dist
      c.vx -= f*dx/dist; c.vy -= f*dy/dist
    }
    for (let i = 0; i < N; i++) {
      nodes[i].vx *= 0.92; nodes[i].vy *= 0.92
      nodes[i].x = Math.max(8, Math.min(W_NET-8, nodes[i].x + nodes[i].vx))
      nodes[i].y = Math.max(8, Math.min(H_NET-8, nodes[i].y + nodes[i].vy))
    }
  }, [])

  // Render loop
  useEffect(() => {
    const netCvs = netRef.current
    if (!netCvs) return
    const ctx = netCvs.getContext('2d')
    let active = true

    const draw = () => {
      if (!active) return
      const sim = simRef.current
      if (!sim) { animRef.current = requestAnimationFrame(draw); return }

      if (running) {
        sim.step(speed)
        setCoherence(sim.coherence)
        setBondCount(sim.bonds.length)
      }
      applyForces()

      const nodes = sim.res, N = nodes.length

      // Clear
      ctx.fillStyle = '#0a0a0a'
      ctx.fillRect(0, 0, W_NET, H_NET)

      // Backbone
      ctx.strokeStyle = 'rgba(88,230,217,0.12)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let i = 0; i < N-1; i++) {
        ctx.moveTo(nodes[i].x, nodes[i].y)
        ctx.lineTo(nodes[i+1].x, nodes[i+1].y)
      }
      ctx.stroke()

      // Bonds
      for (const b of sim.bonds) {
        const a = nodes[b.s], c = nodes[b.t]
        const t = Math.min(b.k / 3, 1)
        ctx.strokeStyle = `rgba(${88+t*167},${230+t*25},${217+t*38},${0.3+t*0.4})`
        ctx.lineWidth = 0.5 + b.k * 0.6
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(c.x, c.y); ctx.stroke()
      }

      // Nodes
      const bondSet = new Set()
      sim.bonds.forEach(b => { bondSet.add(b.s); bondSet.add(b.t) })

      for (let i = 0; i < N; i++) {
        const r = nodes[i]
        const rad = 3 + r.St * 4
        const inBond = bondSet.has(i)

        let col
        if (colorMode === 'phase') {
          const hue = (r.phi / (Math.PI*2)) * 360
          col = `hsl(${hue},80%,55%)`
        } else if (colorMode === 'type') {
          col = r.Se > 0.5 ? '#E53935' : r.Sk > 0.6 ? '#1E88E5' : '#43A047'
        } else {
          col = `rgb(${Math.round(r.Sk*255)},${Math.round(r.St*200)},${Math.round(r.Se*255)})`
        }

        if (inBond) { ctx.shadowColor = '#58E6D9'; ctx.shadowBlur = 8 }
        ctx.fillStyle = col
        ctx.beginPath(); ctx.arc(r.x, r.y, rad, 0, Math.PI*2); ctx.fill()
        ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0
        ctx.strokeStyle = inBond ? '#58E6D9' : 'rgba(255,255,255,0.12)'
        ctx.lineWidth = inBond ? 1.5 : 0.5
        ctx.stroke()

        if (N <= 60) {
          ctx.fillStyle = 'rgba(255,255,255,0.5)'
          ctx.font = '7px monospace'
          ctx.textAlign = 'center'
          ctx.fillText(r.aa, r.x, r.y + 2.5)
        }
      }

      animRef.current = requestAnimationFrame(draw)
    }
    animRef.current = requestAnimationFrame(draw)
    return () => { active = false; if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [running, speed, colorMode, applyForces])

  // Compute spectrum
  const computeSpectrum = useCallback(() => {
    const sim = simRef.current
    if (!sim) return
    const spec = sim.computeSpectrum()
    if (!spec) return

    const N = sim.N
    const maxVal = Math.max(...spec) || 1

    // Draw spectrum
    const cvs = specRef.current
    if (cvs) {
      cvs.width = W_SPEC; cvs.height = H_SPEC
      const ctx = cvs.getContext('2d')
      const img = ctx.createImageData(W_SPEC, H_SPEC)
      for (let py = 0; py < H_SPEC; py++)
        for (let px = 0; px < W_SPEC; px++) {
          const si = Math.floor(px * N / W_SPEC)
          const sj = Math.floor(py * N / H_SPEC)
          const v = spec[si * N + sj] / maxVal
          const [r, g, b] = inferno(v)
          const idx = (py * W_SPEC + px) * 4
          img.data[idx] = r; img.data[idx+1] = g; img.data[idx+2] = b; img.data[idx+3] = 255
        }
      ctx.putImageData(img, 0, 0)
    }

    // Draw contact map
    const ccvs = contRef.current
    if (ccvs) {
      ccvs.width = W_CONT; ccvs.height = H_CONT
      const ctx = ccvs.getContext('2d')
      const img = ctx.createImageData(W_CONT, H_CONT)
      const threshold = maxVal * 0.25
      for (let py = 0; py < H_CONT; py++)
        for (let px = 0; px < W_CONT; px++) {
          const si = Math.floor(px * N / W_CONT)
          const sj = Math.floor(py * N / H_CONT)
          const v = spec[si * N + sj]
          const isContact = v > threshold && Math.abs(si - sj) > 4
          const idx = (py * W_CONT + px) * 4
          if (isContact) {
            img.data[idx] = 88; img.data[idx+1] = 230; img.data[idx+2] = 217
          } else {
            img.data[idx] = 10; img.data[idx+1] = 10; img.data[idx+2] = 10
          }
          img.data[idx+3] = 255
        }
      ctx.putImageData(img, 0, 0)
    }

    // Secondary structure
    setSs(sim.secondaryStructure())
    setSpecComputed(true)
  }, [])

  const handleReset = () => {
    if (simRef.current) {
      simRef.current.reset()
      const cx = W_NET/2, cy = H_NET/2, rad = Math.min(W_NET, H_NET) * 0.35
      simRef.current.res.forEach((r, i) => {
        const a = (i / simRef.current.N) * Math.PI * 2 - Math.PI/2
        r.x = cx + rad * Math.cos(a)
        r.y = cy + rad * Math.sin(a)
        r.vx = 0; r.vy = 0
      })
      setCoherence(0); setBondCount(0); setSs([]); setSpecComputed(false)
    }
  }

  const etaColor = coherence > 0.6 ? '#58E6D9' : coherence > 0.3 ? '#FB8C00' : '#E53935'
  const helixPct = ss.length > 0 ? (ss.filter(s => s === 'H').length / ss.length * 100).toFixed(0) : '—'
  const sheetPct = ss.length > 0 ? (ss.filter(s => s === 'E').length / ss.length * 100).toFixed(0) : '—'

  return (
    <div className="w-full">
      {/* Protein selector */}
      <div className="mb-4 flex flex-wrap gap-2 items-center">
        {Object.entries(PROTEINS).map(([name, s]) => (
          <button key={name} onClick={() => setSeq(s)}
            className={`px-3 py-1 text-xs rounded border transition-colors
              ${seq === s ? 'bg-primaryDark/20 border-primaryDark text-primaryDark'
                         : 'bg-dark/5 dark:bg-dark/50 border-dark/20 dark:border-dark/30 text-dark/60 dark:text-light/60 hover:border-primaryDark/50'}`}>
            {name}
          </button>
        ))}
      </div>

      <textarea value={seq} onChange={e => setSeq(e.target.value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, ''))}
        className="w-full h-12 bg-light dark:bg-dark/80 text-primary dark:text-primaryDark border border-dark/10
                   dark:border-dark/30 rounded font-mono text-sm p-2 resize-none focus:outline-none focus:border-primaryDark"
        placeholder="Enter protein sequence..." />

      {/* Controls */}
      <div className="mt-3 mb-4 flex flex-wrap gap-2 items-center">
        <button onClick={() => setRunning(!running)}
          className={`px-4 py-1.5 text-xs rounded font-mono uppercase tracking-wider border
            ${running ? 'bg-red-500/20 text-red-400 border-red-500/30'
                      : 'bg-primaryDark/20 text-primaryDark border-primaryDark/30'}`}>
          {running ? 'Pause' : 'Fold'}
        </button>
        <button onClick={handleReset}
          className="px-4 py-1.5 text-xs rounded font-mono uppercase tracking-wider
                     bg-dark/5 dark:bg-dark/40 text-dark/50 dark:text-light/40 border border-dark/10 dark:border-dark/20">
          Reset
        </button>
        <button onClick={computeSpectrum}
          className="px-4 py-1.5 text-xs rounded font-mono uppercase tracking-wider
                     bg-primary/10 dark:bg-primaryDark/10 text-primary dark:text-primaryDark border border-primary/30 dark:border-primaryDark/30">
          Compute Spectrum
        </button>

        <span className="text-dark/10 dark:text-light/10 mx-1">|</span>

        {['sentropy', 'phase', 'type'].map(m => (
          <button key={m} onClick={() => setColorMode(m)}
            className={`px-2 py-0.5 text-[10px] rounded font-mono
              ${colorMode === m ? 'text-primaryDark' : 'text-dark/30 dark:text-light/25'}`}>
            {m}
          </button>
        ))}

        <span className="text-dark/10 dark:text-light/10 mx-1">|</span>

        <label className="text-[10px] text-dark/40 dark:text-light/30 font-mono flex items-center gap-1">
          Speed
          <input type="range" min="1" max="100" value={speed}
            onChange={e => setSpeed(parseInt(e.target.value))}
            className="w-16 h-1 accent-primaryDark" />
          <span className="w-6">{speed}x</span>
        </label>

        <label className="text-[10px] text-dark/40 dark:text-light/30 font-mono flex items-center gap-1">
          K
          <input type="range" min="0.5" max="10" step="0.5" value={coupling}
            onChange={e => setCoupling(parseFloat(e.target.value))}
            className="w-16 h-1 accent-primaryDark" />
          <span className="w-6">{coupling.toFixed(1)}</span>
        </label>
      </div>

      {/* Main panels */}
      <div className="flex gap-4 flex-wrap xl:flex-nowrap">

        {/* Force graph */}
        <div className="relative rounded-lg overflow-hidden border border-dark/10 dark:border-dark/20 flex-shrink-0"
             style={{ width: W_NET, height: H_NET }}>
          <canvas ref={netRef} width={W_NET} height={H_NET}
            style={{ width: W_NET, height: H_NET, display: 'block' }} />
          <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-[10px] text-white/60 font-mono">
            Kuramoto Network | {seq.length} residues
          </div>
          <div className="absolute top-2 right-2 bg-black/50 px-2 py-1 rounded text-[10px] font-mono"
               style={{ color: etaColor }}>
            r = {coherence.toFixed(3)} | {bondCount} bonds
          </div>
        </div>

        {/* Right column: spectrum + contacts + metrics */}
        <div className="flex flex-col gap-3" style={{ minWidth: 230 }}>

          {/* Spectrum */}
          <div className="relative rounded-lg overflow-hidden border border-dark/10 dark:border-dark/20"
               style={{ width: W_SPEC, height: H_SPEC }}>
            <canvas ref={specRef} width={W_SPEC} height={H_SPEC}
              style={{ width: W_SPEC, height: H_SPEC, display: 'block', imageRendering: 'pixelated',
                       background: '#0a0a0a' }} />
            <div className="absolute top-1 left-1 bg-black/50 px-1.5 py-0.5 rounded text-[9px] text-white/50 font-mono">
              {specComputed ? '2D-IR Spectrum' : 'Click Compute'}
            </div>
          </div>

          {/* Contact map */}
          <div className="relative rounded-lg overflow-hidden border border-dark/10 dark:border-dark/20"
               style={{ width: W_CONT, height: H_CONT }}>
            <canvas ref={contRef} width={W_CONT} height={H_CONT}
              style={{ width: W_CONT, height: H_CONT, display: 'block', imageRendering: 'pixelated',
                       background: '#0a0a0a' }} />
            <div className="absolute top-1 left-1 bg-black/50 px-1.5 py-0.5 rounded text-[9px] text-white/50 font-mono">
              {specComputed ? 'Contact Map' : '—'}
            </div>
          </div>
        </div>

        {/* Metrics column */}
        <div className="flex flex-col gap-3 flex-grow" style={{ minWidth: 180 }}>

          {/* Coherence */}
          <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
            <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-2 font-mono">
              Phase Coherence
            </div>
            <div className="text-4xl font-bold font-mono" style={{ color: etaColor }}>
              {coherence.toFixed(3)}
            </div>
            <div className="mt-2 h-1.5 bg-dark/10 dark:bg-dark/50 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-300"
                   style={{ width: `${coherence * 100}%`, backgroundColor: etaColor }} />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-2">
            {[
              ['Bonds', bondCount],
              ['Residues', seq.length],
              ['Helix', `${helixPct}%`],
              ['Sheet', `${sheetPct}%`],
            ].map(([label, val]) => (
              <div key={label}
                className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-3 text-center">
                <div className="text-[9px] text-dark/30 dark:text-light/30 uppercase tracking-widest font-mono">{label}</div>
                <div className="text-sm font-bold text-primary dark:text-primaryDark mt-1 font-mono">{val}</div>
              </div>
            ))}
          </div>

          {/* Secondary structure bar */}
          {ss.length > 0 && (
            <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-3">
              <div className="text-[9px] text-dark/30 dark:text-light/30 uppercase tracking-widest font-mono mb-2">
                Secondary Structure
              </div>
              <div className="flex h-3 rounded-full overflow-hidden">
                {ss.map((s, i) => (
                  <div key={i} style={{
                    flex: 1,
                    backgroundColor: s === 'H' ? '#E53935' : s === 'E' ? '#1E88E5' : '#333',
                  }} />
                ))}
              </div>
              <div className="flex gap-3 mt-2 text-[9px] font-mono text-dark/40 dark:text-light/30">
                <span><span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1" />Helix</span>
                <span><span className="inline-block w-2 h-2 rounded-full bg-blue-500 mr-1" />Sheet</span>
                <span><span className="inline-block w-2 h-2 rounded-full bg-neutral-700 mr-1" />Coil</span>
              </div>
            </div>
          )}

          {/* Info */}
          <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-3
                          text-[10px] text-dark/30 dark:text-light/20 font-mono leading-relaxed">
            <div>Oscillator: Kuramoto (N-body)</div>
            <div>Coupling: S-entropy distance</div>
            <div>Spectrum: DFT of K[t][i][j]</div>
            <div>Contacts: spectrum thresholding</div>
            <div>Backend: None (pure JS)</div>
          </div>
        </div>
      </div>
    </div>
  )
}
