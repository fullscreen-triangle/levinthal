/**
 * shakespear Viewer Component
 * ===========================
 * The unified protein observation instrument.
 * GLB mesh + PDB coordinates + GPU shader pipeline.
 *
 * Three.js: 3D scene, GLB loading, orbit controls, backbone visualization
 * WebGL2 shaders: S-entropy, coupling, spectrum, coherence, cavity detection
 * Display shader: GPU-side colormap rendering (no CPU readback for display)
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { parsePDB, computeBounds } from './PDBParser';
import { ShaderEngine, AA_SENTROPY, AA_INDEX } from './ShaderEngine';
import { extractFingerprint, CavityDB, SARPredictor } from './CavityDatabase';
import SARPanel from './SARPanel';
import QueryInterface from './QueryInterface';
import PROTEINS from './ProteinRegistry';
import FoldingVisualizer from './FoldingVisualizer';

const VIEWS = [
  ['spectrum',  'Spectrometer'],
  ['coupling',  'Resonator'],
  ['contacts',  'Diagnostician'],
  ['cavity',    'Cavities'],
  ['sentropy',  'S-Entropy'],
];

/** Build backbone + side-chain visualization from PDB data */
function buildProteinMesh(scene, pdb, groupRef) {
  // Remove old protein group
  if (groupRef.current) scene.remove(groupRef.current);
  const group = new THREE.Group();
  groupRef.current = group;

  const bounds = computeBounds(pdb.atoms);
  const scale = 40 / bounds.size;
  const cx = bounds.center[0], cy = bounds.center[1], cz = bounds.center[2];

  const toLocal = (pos) => new THREE.Vector3(
    (pos[0] - cx) * scale,
    (pos[1] - cy) * scale,
    (pos[2] - cz) * scale
  );

  // CA spheres colored by S-entropy
  const sphereGeo = new THREE.SphereGeometry(0.5, 12, 12);
  const positions = [];

  pdb.caPositions.forEach((pos, i) => {
    const aa = pdb.sequence[i];
    const idx = AA_INDEX[aa] || 0;
    const [sk, st, se] = AA_SENTROPY[idx];

    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(sk, st * 0.8, se),
      metalness: se > 0.5 ? 0.7 : 0.15,
      roughness: 0.25,
      emissive: new THREE.Color(sk * 0.15, st * 0.1, se * 0.2),
      clearcoat: 0.3,
    });
    const mesh = new THREE.Mesh(sphereGeo, mat);
    const p = toLocal(pos);
    mesh.position.copy(p);
    group.add(mesh);
    positions.push(p);
  });

  // Backbone bonds (CA-CA lines)
  if (positions.length > 1) {
    const linePoints = [];
    for (let i = 0; i < positions.length - 1; i++) {
      linePoints.push(positions[i], positions[i + 1]);
    }
    const lineGeo = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x58E6D9,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });
    group.add(new THREE.LineSegments(lineGeo, lineMat));
  }

  // Harmonic edges: connect residues with similar S-entropy at long range
  const edgePoints = [];
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 5; j < positions.length; j++) {
      const aa_i = AA_INDEX[pdb.sequence[i]] || 0;
      const aa_j = AA_INDEX[pdb.sequence[j]] || 0;
      const [sk_i, st_i, se_i] = AA_SENTROPY[aa_i];
      const [sk_j, st_j, se_j] = AA_SENTROPY[aa_j];
      const dist = Math.sqrt(
        (sk_i - sk_j) ** 2 + (st_i - st_j) ** 2 + (se_i - se_j) ** 2
      );
      // Strong harmonic coupling: close in S-entropy AND close in 3D space
      const spatial = positions[i].distanceTo(positions[j]);
      if (dist < 0.15 && spatial < 15) {
        edgePoints.push(positions[i], positions[j]);
      }
    }
  }
  if (edgePoints.length > 0) {
    const edgeGeo = new THREE.BufferGeometry().setFromPoints(edgePoints);
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0xB63E96,
      transparent: true,
      opacity: 0.2,
      linewidth: 1,
    });
    group.add(new THREE.LineSegments(edgeGeo, edgeMat));
  }

  scene.add(group);
  return { positions, nResidues: pdb.nResidues, nEdges: edgePoints.length / 2 };
}

export default function ProteusViewer() {
  const mountRef = useRef(null);
  const obsCanvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const proteinGroupRef = useRef(null);
  const glbModelRef = useRef(null);
  const animRef = useRef(null);

  const [sequence, setSequence] = useState('');
  const [view, setView] = useState('spectrum');
  const [metrics, setMetrics] = useState({ eta: 0, contacts: 0, N: 0, edges: 0 });
  const [pdbData, setPdbData] = useState(null);
  const [glbLoaded, setGlbLoaded] = useState(false);
  const [ready, setReady] = useState(false);
  const [showGlb, setShowGlb] = useState(true);
  const [fingerprint, setFingerprint] = useState(null);
  const [activeProtein, setActiveProtein] = useState(null);
  const [loading, setLoading] = useState(false);
  const dbRef = useRef(null);
  const sarRef = useRef(null);

  // ---- Initialize Three.js scene + shader engine ----
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080808);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 80);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(560, 560);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;
    controlsRef.current = controls;

    // Lighting
    scene.add(new THREE.AmbientLight(0x404040, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(30, 30, 30);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x58E6D9, 0.4);
    fill.position.set(-20, 10, -20);
    scene.add(fill);
    const rim = new THREE.PointLight(0xB63E96, 0.6, 200);
    rim.position.set(0, -30, 30);
    scene.add(rim);

    // Load default protein (first in registry)
    // Deferred to after engine init via loadProtein()

    // Shader engine initialized in separate effect (needs obsCanvasRef)

    // Render loop
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // ---- Initialize shader engine after canvas is mounted ----
  const [canvasMounted, setCanvasMounted] = useState(false);
  const obsCanvasCallback = useCallback((node) => {
    if (node !== null) {
      obsCanvasRef.current = node;
      setCanvasMounted(true);
    }
  }, []);

  useEffect(() => {
    if (!canvasMounted || engineRef.current) return;
    const obsCanvas = obsCanvasRef.current;
    if (!obsCanvas) return;

    obsCanvas.width = 280;
    obsCanvas.height = 280;

    (async () => {
      try {
        const engine = new ShaderEngine(obsCanvas);
        await engine.init();
        engineRef.current = engine;
        setReady(true);
      } catch (e) {
        console.warn('ShaderEngine:', e.message);
        setReady(true);
      }
    })();
  }, [canvasMounted]);

  // ---- Initialize cavity DB + SAR predictor ----
  useEffect(() => {
    const db = new CavityDB();
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
              nEdges: 28, meanSk: 0.44, meanSt: 0.45, meanSe: 0.35, nResidues: 223, cavities: [] }},
      { id: 'insulin', name: 'Insulin', type: 'protein',
        fp: { nCavities: 2, meanQ: 450, meanOmega: 6.5, meanArea: 28, coherence: 0.69,
              nEdges: 5, meanSk: 0.52, meanSt: 0.47, meanSe: 0.26, nResidues: 51, cavities: [] }},
      { id: 'hemoglobin', name: 'Hemoglobin', type: 'protein',
        fp: { nCavities: 10, meanQ: 780, meanOmega: 5.5, meanArea: 55, coherence: 0.86,
              nEdges: 38, meanSk: 0.44, meanSt: 0.46, meanSe: 0.36, nResidues: 574, cavities: [] }},
    ];
    for (const ex of examples) db.add(ex.id, ex.name, ex.type, ex.fp);
    dbRef.current = db;

    const sar = new SARPredictor();
    examples.forEach(ex => {
      const mockActivity = -Math.log10((20 - ex.fp.nCavities) * 0.1 + (1 - ex.fp.coherence) * 5 + 0.5);
      sar.addTrainingPoint(ex.fp, mockActivity);
    });
    sar.fit();
    sarRef.current = sar;
  }, []);

  // ---- Load a paired GLB+CIF protein ----
  const loadProtein = useCallback(async (protein) => {
    const scene = sceneRef.current;
    if (!scene) return;
    setLoading(true);

    // Remove old protein geometry
    if (glbModelRef.current) { scene.remove(glbModelRef.current); glbModelRef.current = null; }
    if (proteinGroupRef.current) { scene.remove(proteinGroupRef.current); proteinGroupRef.current = null; }

    // Load GLB model
    try {
      const gltf = await new Promise((resolve, reject) => {
        new GLTFLoader().load(protein.glb, resolve, undefined, reject);
      });
      const model = gltf.scene;
      model.traverse((child) => {
        if (child.isMesh) {
          child.material = new THREE.MeshPhysicalMaterial({
            color: 0x58E6D9, metalness: 0.25, roughness: 0.35,
            clearcoat: 0.4, transparent: true, opacity: 0.65,
          });
        }
      });
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      model.position.sub(center);
      model.scale.multiplyScalar(40 / size);
      model.name = 'glb_model';
      scene.add(model);
      glbModelRef.current = model;
      setGlbLoaded(true);
    } catch (e) {
      console.warn('GLB load failed:', e.message);
    }

    // Load CIF structure
    try {
      const res = await fetch(protein.cif);
      const text = await res.text();
      const pdb = parsePDB(text);
      if (pdb.sequence.length >= 2) {
        setSequence(pdb.sequence);
        setPdbData(pdb);
        const info = buildProteinMesh(scene, pdb, proteinGroupRef);
        setMetrics(prev => ({ ...prev, edges: info.nEdges, N: pdb.nResidues }));
      }
    } catch (e) {
      console.warn('CIF load failed:', e.message);
    }

    setActiveProtein(protein);
    setLoading(false);
  }, []);

  // Load default protein on mount
  useEffect(() => {
    if (ready && sceneRef.current && !activeProtein) {
      loadProtein(PROTEINS[0]);
    }
  }, [ready, loadProtein, activeProtein]);

  // ---- GLB visibility toggle ----
  useEffect(() => {
    if (glbModelRef.current) glbModelRef.current.visible = showGlb;
  }, [showGlb]);

  // ---- Set sequence on engine ----
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !ready) return;
    const clean = sequence.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
    if (clean.length < 2) return;
    engine.setSequence(clean);
    setMetrics(prev => ({ ...prev, N: clean.length }));
  }, [sequence, ready]);

  // ---- Observation loop: GPU shader pipeline + GPU display ----
  useEffect(() => {
    if (!ready) return;
    const engine = engineRef.current;
    if (!engine || !engine.initialized || engine.N < 2) return;

    let running = true;
    let lastT = performance.now();
    let frameCount = 0;

    const loop = () => {
      if (!running) return;
      const now = performance.now();
      const dt = (now - lastT) / 1000;
      lastT = now;

      // Run the full shader pipeline (THE OBSERVATION)
      engine.observe(dt);

      // Render to the visible observation canvas using GPU display shader
      engine.renderToCanvas(view);

      // Update metrics + fingerprint every 30 frames
      frameCount++;
      if (frameCount % 30 === 0) {
        const m = engine.readCoherence();
        setMetrics(prev => ({ ...prev, ...m }));

        // Extract fingerprint from GPU data
        try {
          const N = engine.N;
          const sData = engine._readFBO(engine.framebuffers.sentropy.fbo, N, 1);
          const cData = engine._readFBO(engine.framebuffers.coupling.fbo, N, N);
          const hData = engine._readFBO(engine.framebuffers.coherence.fbo, N, N);
          const fp = extractFingerprint(sData, cData, hData, N);
          if (fp) setFingerprint(fp);
        } catch (e) { /* ignore readback errors */ }
      }

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    return () => { running = false; };
  }, [ready, view, sequence]);

  // ---- File handlers ----
  const handlePDB = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const pdb = parsePDB(ev.target.result);
      if (pdb.sequence.length < 2) return;
      setSequence(pdb.sequence);
      setPdbData(pdb);

      const scene = sceneRef.current;
      if (!scene) return;
      const info = buildProteinMesh(scene, pdb, proteinGroupRef);
      setMetrics(prev => ({ ...prev, edges: info.nEdges }));
    };
    reader.readAsText(file);
  }, []);

  const handleGLB = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const scene = sceneRef.current;
      if (!scene) return;
      // Remove old GLB
      if (glbModelRef.current) scene.remove(glbModelRef.current);

      const loader = new GLTFLoader();
      loader.parse(ev.target.result, '', (gltf) => {
        const model = gltf.scene;
        model.traverse((child) => {
          if (child.isMesh) {
            child.material = new THREE.MeshPhysicalMaterial({
              color: 0x58E6D9,
              metalness: 0.25,
              roughness: 0.35,
              clearcoat: 0.4,
              transparent: true,
              opacity: 0.7,
            });
          }
        });
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        model.position.sub(center);
        model.scale.multiplyScalar(40 / size);
        model.name = 'glb_model';
        scene.add(model);
        glbModelRef.current = model;
        setGlbLoaded(true);
        setShowGlb(true);
      });
    };
    reader.readAsArrayBuffer(file);
  }, []);

  const healthColor = metrics.eta > 0.8 ? '#58E6D9' :
                      metrics.eta > 0.5 ? '#FB8C00' : '#E53935';
  const healthText = metrics.eta > 0.8 ? 'Coherent' :
                     metrics.eta > 0.5 ? 'Stressed' : 'Decoherent';

  return (
    <div className="w-full">
      {/* Protein Selector */}
      <div className="mb-4 flex flex-wrap gap-2 items-center">
        {PROTEINS.map(p => (
          <button key={p.id} onClick={() => loadProtein(p)}
            className={`px-3 py-1 text-xs rounded border transition-colors
              ${activeProtein?.id === p.id
                ? 'bg-primaryDark/20 border-primaryDark text-primaryDark'
                : 'bg-dark/5 dark:bg-dark/50 border-dark/20 dark:border-dark/30 text-dark/60 dark:text-light/60 hover:border-primaryDark/50'}`}>
            {p.name}
          </button>
        ))}
        <label className="px-3 py-1 text-xs rounded border border-dark/20 dark:border-dark/30 text-dark/60 dark:text-light/60
                          hover:border-primaryDark/50 cursor-pointer transition-colors">
          Upload PDB/CIF
          <input type="file" accept=".pdb,.cif,.ent" onChange={handlePDB} className="hidden" />
        </label>
        <label className="px-3 py-1 text-xs rounded border border-dark/20 dark:border-dark/30 text-dark/60 dark:text-light/60
                          hover:border-primary/50 cursor-pointer transition-colors">
          Upload GLB
          <input type="file" accept=".glb,.gltf" onChange={handleGLB} className="hidden" />
        </label>
        {glbLoaded && (
          <button onClick={() => setShowGlb(!showGlb)}
            className={`px-3 py-1 text-xs rounded border transition-colors
              ${showGlb ? 'border-primaryDark/50 text-primaryDark' : 'border-dark/30 text-dark/40 dark:text-light/40'}`}>
            {showGlb ? 'Hide' : 'Show'} Mesh
          </button>
        )}
        {loading && <span className="text-[10px] text-primaryDark animate-pulse font-mono">Loading...</span>}
      </div>

      {/* Sequence */}
      <textarea value={sequence} onChange={e => setSequence(e.target.value)}
        className="w-full h-12 bg-light dark:bg-dark/80 text-primary dark:text-primaryDark
                   border border-dark/10 dark:border-dark/30 rounded font-mono text-sm p-2
                   resize-none focus:outline-none focus:border-primaryDark"
        placeholder="Enter protein sequence..." />

      {/* Query Interface (model-driven) */}
      <QueryInterface
        engine={engineRef.current}
        sequence={sequence}
        setSequence={setSequence}
        setView={setView}
        db={dbRef.current}
        sar={sarRef.current}
        fingerprint={fingerprint}
      />

      {/* Manual view tabs (secondary) */}
      <div className="mt-3 mb-4 flex gap-1 flex-wrap">
        {VIEWS.map(([key, label]) => (
          <button key={key} onClick={() => setView(key)}
            className={`px-2 py-1 text-[10px] uppercase tracking-wider rounded transition-colors
              ${view === key
                ? 'bg-primaryDark/20 text-primaryDark border border-primaryDark/30'
                : 'bg-dark/5 dark:bg-dark/50 text-dark/30 dark:text-light/30 border border-transparent hover:text-dark/50 dark:hover:text-light/50'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Main Display */}
      <div className="flex gap-4 flex-wrap xl:flex-nowrap">

        {/* 3D Scene */}
        <div className="relative rounded-lg overflow-hidden border border-dark/10 dark:border-dark/20 flex-shrink-0"
             style={{ width: 560, height: 560 }}>
          <div ref={mountRef} className="w-full h-full" />
          <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-[10px] text-white/60 font-mono">
            {activeProtein ? activeProtein.name : 'shakespear'} {activeProtein ? `| ${activeProtein.pdbId}` : ''}
            {glbLoaded && showGlb ? ' | GLB' : ''}
          </div>
          {pdbData && (
            <div className="absolute bottom-2 left-2 bg-black/50 px-2 py-1 rounded text-[10px] text-white/40 font-mono">
              {pdbData.nResidues} residues | {pdbData.nAtoms} atoms | {pdbData.format.toUpperCase()} | {metrics.edges} edges
            </div>
          )}
        </div>

        {/* Right Panel: Observation + Metrics */}
        <div className="flex flex-col gap-3 flex-grow" style={{ minWidth: 280 }}>

          {/* 2D Observation (GPU-rendered via display shader) */}
          <div className="relative rounded-lg overflow-hidden border border-dark/10 dark:border-dark/20"
               style={{ width: 280, height: 280 }}>
            <canvas ref={obsCanvasCallback}
              width={280} height={280}
              style={{ width: 280, height: 280, imageRendering: 'pixelated' }} />
            <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-[10px] text-white/60 font-mono">
              {view.toUpperCase()} | {engineRef.current?.initialized ? 'GPU' : 'INIT...'} {engineRef.current?.hasFloat ? 'F32' : 'U8'}
              {engineRef.current?.error && <span className="text-red-400 ml-1">{engineRef.current.error}</span>}
            </div>
          </div>

          {/* Coherence */}
          <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20 rounded-lg p-4">
            <div className="text-[10px] text-dark/40 dark:text-light/40 uppercase tracking-widest mb-2 font-mono">
              Coherence Order Parameter
            </div>
            <div className="text-4xl font-bold font-mono" style={{ color: healthColor }}>
              {metrics.eta.toFixed(3)}
            </div>
            <div className="text-xs mt-1 font-mono" style={{ color: healthColor }}>
              {healthText}
            </div>
            <div className="mt-3 h-1.5 bg-dark/10 dark:bg-dark/50 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-500"
                   style={{ width: `${metrics.eta * 100}%`, backgroundColor: healthColor }} />
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-2">
            {[
              ['Residues', metrics.N],
              ['Contacts', metrics.contacts],
              ['GPU Mem', `${Math.ceil((metrics.N * metrics.N * 4 * 4 * 4 + metrics.N * 16) / 1024)} KB`],
              ['Backend', 'None'],
            ].map(([label, value]) => (
              <div key={label}
                className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20
                           rounded-lg p-3 text-center">
                <div className="text-[9px] text-dark/30 dark:text-light/30 uppercase tracking-widest font-mono">
                  {label}
                </div>
                <div className="text-sm font-bold text-primary dark:text-primaryDark mt-1 font-mono">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {/* Pipeline Info */}
          <div className="bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/20
                          rounded-lg p-3 text-[10px] text-dark/30 dark:text-light/25 font-mono leading-relaxed">
            <div>Pass 1: Sequence &rarr; S-Entropy</div>
            <div>Pass 3: S-Entropy &rarr; Coupling K<sub>ij</sub></div>
            <div>Pass 6: Coupling &rarr; Spectrum (2D-IR)</div>
            <div>Pass 7: Spectrum &rarr; Contacts + Coherence</div>
            <div>Pass C: Harmonic Network &rarr; Virtual Cavities</div>
            <div>Display: GPU colormap (no CPU readback)</div>
          </div>
        </div>
      </div>
      {/* Folding Visualizer */}
      {sequence.length >= 2 && sequence.length <= 200 && (
        <div className="mt-4">
          <div className="text-[10px] text-dark/40 dark:text-light/30 uppercase tracking-widest font-mono mb-2">
            Kuramoto Folding Dynamics
          </div>
          <FoldingVisualizer sequence={sequence} width={560} height={400} />
        </div>
      )}

      {/* SAR + Database Search */}
      <SARPanel engine={engineRef.current} sequence={sequence} fingerprint={fingerprint} />
    </div>
  );
}
