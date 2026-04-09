/**
 * PROTEUS Viewer Component
 * ========================
 * The unified protein observation instrument.
 * Combines GLB mesh + PDB coordinates + GPU shader pipeline.
 *
 * Three.js handles: GLB loading, 3D scene, orbit controls
 * WebGL2 shaders handle: S-entropy, coupling, spectrum, coherence (the physics)
 * The 3D scene IS the observation apparatus.
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { parsePDB, computeBounds } from './PDBParser';
import { ShaderEngine, AA_SENTROPY, AA_INDEX } from './ShaderEngine';

const EXAMPLE_SEQUENCES = {
  'Crambin (46 aa)': 'TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN',
  'Insulin B (30 aa)': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
  'BPTI (58 aa)': 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
};

export default function ProteusViewer() {
  const mountRef = useRef(null);
  const canvasRef = useRef(null);
  const glCanvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animRef = useRef(null);

  const [sequence, setSequence] = useState(EXAMPLE_SEQUENCES['Crambin (46 aa)']);
  const [view, setView] = useState('spectrum');
  const [metrics, setMetrics] = useState({ eta: 0, contacts: 0, N: 0 });
  const [pdbLoaded, setPdbLoaded] = useState(false);
  const [glbLoaded, setGlbLoaded] = useState(false);
  const [ready, setReady] = useState(false);

  // ---- Initialize Three.js scene + WebGL2 shader engine ----
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 80);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(560, 560);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Lighting
    scene.add(new THREE.AmbientLight(0x404040, 0.8));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(30, 30, 30);
    scene.add(dirLight);
    const rimLight = new THREE.PointLight(0x58E6D9, 1.0, 200);
    rimLight.position.set(-30, 20, -30);
    scene.add(rimLight);

    // Try loading the GLB model
    const gltfLoader = new GLTFLoader();
    gltfLoader.load('/models/conformational_transition_of_troponin.glb',
      (gltf) => {
        const model = gltf.scene;
        model.traverse((child) => {
          if (child.isMesh) {
            child.material = new THREE.MeshPhysicalMaterial({
              color: 0x58E6D9,
              metalness: 0.3,
              roughness: 0.4,
              clearcoat: 0.5,
              transparent: true,
              opacity: 0.85,
            });
          }
        });
        // Center and scale
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        model.position.sub(center);
        model.scale.multiplyScalar(40 / size);
        scene.add(model);
        setGlbLoaded(true);
      },
      undefined,
      () => { /* GLB not found -- ok, we work without it */ }
    );

    // Shader engine (hidden canvas for GPU computation)
    const glCanvas = document.createElement('canvas');
    glCanvas.width = 512;
    glCanvas.height = 512;
    glCanvasRef.current = glCanvas;

    (async () => {
      try {
        const engine = new ShaderEngine(glCanvas);
        await engine.init();
        engineRef.current = engine;
        setReady(true);
      } catch (e) {
        console.warn('ShaderEngine init failed:', e.message);
        setReady(true); // still allow 3D view
      }
    })();

    // Render loop
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // ---- Set sequence on engine ----
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !ready) return;
    const cleanSeq = sequence.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
    if (cleanSeq.length < 2) return;
    engine.setSequence(cleanSeq);
    setMetrics(prev => ({ ...prev, N: cleanSeq.length }));
  }, [sequence, ready]);

  // ---- Observation loop: run shader pipeline, render 2D observation ----
  useEffect(() => {
    if (!ready) return;
    const engine = engineRef.current;
    const canvas = canvasRef.current;
    if (!engine || !canvas || engine.N < 2) return;

    const ctx = canvas.getContext('2d');
    const SIZE = 280;
    canvas.width = SIZE;
    canvas.height = SIZE;

    let running = true;
    let lastT = performance.now();

    const loop = () => {
      if (!running) return;
      const now = performance.now();
      const dt = (now - lastT) / 1000;
      lastT = now;

      engine.observe(dt);

      // Read active texture
      const gl = engine.gl;
      const fbKey = view === 'contacts' ? 'coherence' : view;
      const fb = engine.framebuffers[fbKey];
      if (!fb) { requestAnimationFrame(loop); return; }

      const w = fb.width, h = fb.height;
      const pixels = new Float32Array(w * h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb.fbo);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const imgData = ctx.createImageData(SIZE, SIZE);
      for (let py = 0; py < SIZE; py++) {
        for (let px = 0; px < SIZE; px++) {
          const sx = Math.min(Math.floor(px * w / SIZE), w - 1);
          const sy = Math.min(Math.floor(py * h / SIZE), h - 1);
          const si = (sy * w + sx) * 4;
          const di = (py * SIZE + px) * 4;

          let r, g, b;
          if (view === 'spectrum') {
            const m = Math.min(pixels[si], 1);
            r = Math.floor(Math.min(1, m * 3 - 0.5) * 255);
            g = Math.floor(Math.min(1, m * 2 - 0.7) * 255);
            b = Math.floor(Math.min(1, 0.5 - m * 0.3) * 255);
          } else if (view === 'coupling') {
            const k = Math.min(pixels[si] / 5, 1);
            r = Math.floor(k * 255);
            g = Math.floor(k * 0.6 * 255);
            b = Math.floor((1 - k) * 0.3 * 255);
          } else if (view === 'contacts') {
            r = Math.floor(Math.min(1, pixels[si + 1] * 4) * 255);
            g = Math.floor(Math.min(1, pixels[si + 2] * 4) * 100);
            b = Math.floor(Math.min(1, pixels[si]) * 255);
          } else {
            r = Math.floor(pixels[si] * 255);
            g = Math.floor(pixels[si + 1] * 255);
            b = Math.floor(pixels[si + 2] * 255);
          }
          imgData.data[di] = r;
          imgData.data[di + 1] = g;
          imgData.data[di + 2] = b;
          imgData.data[di + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);

      if (Math.random() < 0.08) {
        setMetrics(prev => ({ ...prev, ...engine.readCoherence() }));
      }

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    return () => { running = false; };
  }, [ready, view, sequence]);

  // ---- PDB file handler ----
  const handlePDB = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const pdb = parsePDB(ev.target.result);
      setSequence(pdb.sequence);
      setPdbLoaded(true);

      // Add CA spheres to 3D scene
      const scene = sceneRef.current;
      if (!scene) return;
      const bounds = computeBounds(pdb.atoms);
      const scale = 40 / bounds.size;
      const geo = new THREE.SphereGeometry(0.4, 8, 8);

      pdb.caPositions.forEach((pos, i) => {
        const aa = pdb.sequence[i];
        const idx = AA_INDEX[aa] || 0;
        const [sk, st, se] = AA_SENTROPY[idx];
        const mat = new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(sk, st, se),
          metalness: se > 0.5 ? 0.8 : 0.1,
          roughness: 0.3,
          emissive: new THREE.Color(sk * 0.2, st * 0.2, se * 0.2),
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(
          (pos[0] - bounds.center[0]) * scale,
          (pos[1] - bounds.center[1]) * scale,
          (pos[2] - bounds.center[2]) * scale
        );
        scene.add(mesh);
      });
    };
    reader.readAsText(file);
  }, []);

  const healthColor = metrics.eta > 0.8 ? '#58E6D9' :
                      metrics.eta > 0.5 ? '#FB8C00' : '#E53935';
  const healthText = metrics.eta > 0.8 ? 'Coherent' :
                     metrics.eta > 0.5 ? 'Stressed' : 'Decoherent';

  return (
    <div className="w-full">
      {/* Sequence Input */}
      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(EXAMPLE_SEQUENCES).map(([name, seq]) => (
          <button key={name} onClick={() => setSequence(seq)}
            className={`px-3 py-1 text-xs rounded border transition-colors
              ${sequence === seq
                ? 'bg-primaryDark/20 border-primaryDark text-primaryDark'
                : 'bg-dark/50 border-dark/30 text-light/60 hover:border-primaryDark/50'}`}>
            {name}
          </button>
        ))}
        <label className="px-3 py-1 text-xs rounded border border-dark/30 text-light/60
                          hover:border-primaryDark/50 cursor-pointer transition-colors">
          Upload PDB
          <input type="file" accept=".pdb" onChange={handlePDB} className="hidden" />
        </label>
      </div>

      <textarea value={sequence} onChange={e => setSequence(e.target.value)}
        className="w-full h-12 bg-dark/80 text-primaryDark border border-dark/30 rounded
                   font-mono text-sm p-2 resize-none focus:outline-none focus:border-primaryDark"
        placeholder="Enter protein sequence..." />

      {/* View Tabs */}
      <div className="mt-3 mb-4 flex gap-1">
        {[
          ['spectrum', 'Spectrometer'],
          ['coupling', 'Resonator'],
          ['contacts', 'Diagnostician'],
          ['sentropy', 'S-Entropy'],
        ].map(([key, label]) => (
          <button key={key} onClick={() => setView(key)}
            className={`px-3 py-1.5 text-xs uppercase tracking-wider rounded transition-colors
              ${view === key
                ? 'bg-primaryDark text-dark font-bold'
                : 'bg-dark/50 text-light/50 hover:text-light/80'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Main Display: 3D + 2D side by side */}
      <div className="flex gap-4 flex-wrap lg:flex-nowrap">

        {/* 3D GLB/PDB View */}
        <div className="relative rounded overflow-hidden border border-dark/20"
             style={{ width: 560, height: 560 }}>
          <div ref={mountRef} className="w-full h-full" />
          <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-[10px] text-light/50">
            shakespear {glbLoaded && '| GLB'} {pdbLoaded && '| PDB'}
          </div>
        </div>

        {/* 2D Observation + Metrics */}
        <div className="flex flex-col gap-3" style={{ minWidth: 280 }}>

          {/* 2D Observation Canvas */}
          <div className="relative rounded overflow-hidden border border-dark/20">
            <canvas ref={canvasRef}
              style={{ width: 280, height: 280, imageRendering: 'pixelated' }} />
            <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-[10px] text-light/50">
              {view.toUpperCase()} | GPU
            </div>
          </div>

          {/* Coherence Meter */}
          <div className="bg-dark/80 border border-dark/20 rounded p-4">
            <div className="text-[10px] text-light/40 uppercase tracking-widest mb-2">
              Coherence
            </div>
            <div className="text-3xl font-bold" style={{ color: healthColor }}>
              {metrics.eta.toFixed(3)}
            </div>
            <div className="text-xs mt-1" style={{ color: healthColor }}>
              {healthText}
            </div>
            <div className="mt-2 h-1 bg-dark/50 rounded overflow-hidden">
              <div className="h-full rounded transition-all duration-300"
                   style={{ width: `${metrics.eta * 100}%`, backgroundColor: healthColor }} />
            </div>
          </div>

          {/* Stats */}
          <div className="bg-dark/80 border border-dark/20 rounded p-4 text-xs text-light/50">
            <div className="flex justify-between mb-1">
              <span>Residues</span><span className="text-primaryDark">{metrics.N}</span>
            </div>
            <div className="flex justify-between mb-1">
              <span>Contacts</span><span className="text-primaryDark">{metrics.contacts}</span>
            </div>
            <div className="flex justify-between mb-1">
              <span>GPU Memory</span>
              <span className="text-primaryDark">
                {Math.ceil((metrics.N * metrics.N * 4 * 4 * 3 + metrics.N * 16) / 1024)} KB
              </span>
            </div>
            <div className="flex justify-between">
              <span>Backend</span><span className="text-primaryDark">None</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
