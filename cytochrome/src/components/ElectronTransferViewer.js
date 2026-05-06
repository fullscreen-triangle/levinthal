// Live electron-transfer viewer.
//
// Loads the productive cytochrome P450 GLB
// (public/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb), places the
// four-cofactor chain (NADPH → FAD → FMN → heme) anchored to the real Fe
// position read by levinthal_glb.find_iron(), and animates the electron
// probability cloud across the chain at the same hop-rate kinetics as the
// shader pipeline of Paper 4.
//
// Cofactor positions and hop rates are pre-computed from
// run_pipeline_glb_grounded() and shipped here as constants so the viewer
// runs in the browser without re-parsing the GLB for coordinates.

import { useRef, useState, useMemo, useEffect, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Html, AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";
import { motion } from "framer-motion";

// =====================================================================
// Constants — match cytochrome/glb/levinthal_glb/shader_pipeline.py
// =====================================================================

const GLB_URL = "/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb";

// Fe position from levinthal_glb.find_iron() on the GLB
const FE_POS = [17.2587, 11.5273, 24.7629];

// Cofactor positions from CofactorPlacement, anchored to FE_POS.
// (These are the values printed by run_pipeline_glb_grounded.)
const COFACTORS = [
  { name: "NADPH", pos: [32.5480, 22.2499, 36.3932], color: "#4C72B0",
    sigma: 1.4, role: "hydride donor" },
  { name: "FAD",   pos: [29.7681, 20.3003, 34.2786], color: "#FFA500",
    sigma: 1.5, role: "first flavin" },
  { name: "FMN",   pos: [26.9883, 18.3507, 32.1640], color: "#55A868",
    sigma: 1.7, role: "heme-facing flavin" },
  { name: "heme",  pos: [17.2587, 11.5273, 24.7629], color: "#C44E52",
    sigma: 2.6, role: "Fe (real GLB position)" },
];

const HOP_RATES_INV_S = [6e12, 4e12, 2e12];   // s^-1 — at the categorical clock floor
const T_MAX_FS = 800;                          // animation horizon
const T_LOOP_SECONDS = 8;                      // wall-time loop length

// =====================================================================
// Pure JS port of hop_occupancies() from shader_pipeline.py
// =====================================================================

function hopOccupancies(t_s, k = HOP_RATES_INV_S) {
  const [k1, k2, k3] = k;
  const e1 = Math.exp(-k1 * t_s);
  const e2 = Math.exp(-k2 * t_s);
  const e3 = Math.exp(-k3 * t_s);
  return [
    e1,
    (1 - e1) * e2,
    (1 - e1) * (1 - e2) * e3,
    (1 - e1) * (1 - e2) * (1 - e3),
  ];
}

// Centroid of the cofactor probability cloud at time t_fs
function densityCentroid(t_fs) {
  const occ = hopOccupancies(t_fs * 1e-15);
  const total = occ.reduce((a, b) => a + b, 0) || 1;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < 4; i++) {
    const w = occ[i] / total;
    cx += COFACTORS[i].pos[0] * w;
    cy += COFACTORS[i].pos[1] * w;
    cz += COFACTORS[i].pos[2] * w;
  }
  return [cx, cy, cz, occ];
}

// =====================================================================
// 3D scene components
// =====================================================================

function GLBStructure({ wireframe = false }) {
  const { scene } = useGLTF(GLB_URL);

  // Make every material translucent so the electron cloud is visible
  // through the structure. Done once on mount.
  const cloned = useMemo(() => {
    const c = scene.clone(true);
    c.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        const m = obj.material.clone();
        m.transparent = true;
        m.opacity = wireframe ? 0.45 : 0.32;
        m.depthWrite = false;
        m.side = THREE.DoubleSide;
        if (wireframe) m.wireframe = true;
        obj.material = m;
      }
    });
    return c;
  }, [scene, wireframe]);

  return <primitive object={cloned} />;
}

function CofactorMarkers({ occupancies = [0, 0, 0, 0] }) {
  return (
    <group>
      {COFACTORS.map((c, i) => {
        const radius = 0.7 + occupancies[i] * 1.6;
        const emissive = occupancies[i];
        return (
          <group key={c.name} position={c.pos}>
            <mesh>
              <sphereGeometry args={[radius, 24, 24]} />
              <meshStandardMaterial
                color={c.color}
                emissive={c.color}
                emissiveIntensity={0.4 + emissive * 1.6}
                roughness={0.3}
                metalness={0.05}
              />
            </mesh>
            <Html
              distanceFactor={20}
              position={[0, radius + 1.6, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <div className="text-[10px] font-mono uppercase tracking-widest
                bg-dark/80 text-light px-2 py-0.5 rounded whitespace-nowrap"
                style={{ color: c.color, borderColor: c.color }}>
                {c.name}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

function ChainAxis() {
  // Draw a thin line from NADPH → heme as the spatial backbone of the chain
  const points = useMemo(() => COFACTORS.map((c) => new THREE.Vector3(...c.pos)), []);
  const geom = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);
  return (
    <line geometry={geom}>
      <lineBasicMaterial color="#888888" transparent opacity={0.4} />
    </line>
  );
}

function ElectronCloud({ tFsRef }) {
  // A rich electron cloud built from N transparent spheres each centred on
  // a cofactor with size weighted by occupancy. The centroid of the cloud
  // sweeps from NADPH → heme as time advances.
  const groups = useRef([]);

  useFrame(() => {
    const t = tFsRef.current;
    const occ = hopOccupancies(t * 1e-15);
    for (let i = 0; i < 4; i++) {
      const g = groups.current[i];
      if (!g) continue;
      const o = occ[i];
      g.scale.setScalar(0.4 + o * 4.0);
      g.children[0].material.opacity = Math.min(0.9, 0.05 + o * 1.2);
    }
  });

  return (
    <group>
      {COFACTORS.map((c, i) => (
        <group
          key={c.name}
          position={c.pos}
          ref={(el) => { groups.current[i] = el; }}
        >
          <mesh>
            <sphereGeometry args={[1.0, 28, 28]} />
            <meshBasicMaterial
              color="#FF55FF"
              transparent
              opacity={0.05}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function TravelingElectron({ tFsRef }) {
  // A bright "centroid" sphere that traces the trajectory across the chain.
  const meshRef = useRef();

  useFrame(() => {
    const t = tFsRef.current;
    const [cx, cy, cz] = densityCentroid(t);
    if (meshRef.current) {
      meshRef.current.position.set(cx, cy, cz);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.55, 24, 24]} />
      <meshStandardMaterial
        color="#FFFFFF"
        emissive="#FF55FF"
        emissiveIntensity={3.0}
        toneMapped={false}
      />
    </mesh>
  );
}

function CameraRig() {
  const { camera } = useThree();
  useEffect(() => {
    // Centre on the midpoint of the chain
    const midpoint = [
      (COFACTORS[0].pos[0] + COFACTORS[3].pos[0]) / 2,
      (COFACTORS[0].pos[1] + COFACTORS[3].pos[1]) / 2,
      (COFACTORS[0].pos[2] + COFACTORS[3].pos[2]) / 2,
    ];
    camera.position.set(
      midpoint[0] + 35,
      midpoint[1] + 25,
      midpoint[2] + 35
    );
    camera.lookAt(...midpoint);
  }, [camera]);
  return null;
}

// =====================================================================
// Top-level viewer component (ssr-safe via dynamic import in pages)
// =====================================================================

export default function ElectronTransferViewer({
  height = 520,
  autoplay = true,
  showWireframe = false,
}) {
  const [playing, setPlaying] = useState(autoplay);
  const [tFs, setTFs] = useState(0);
  const tFsRef = useRef(0);

  // Animate t with a wall-clock loop; tFsRef is the canonical clock for the
  // scene, tFs is the React mirror used by the HUD.
  useEffect(() => {
    if (!playing) return;
    let frameId;
    const start = performance.now();
    const startT = tFsRef.current;
    const tick = (now) => {
      const elapsed = (now - start) / 1000;     // seconds
      const cycleProgress = ((startT / T_MAX_FS) + elapsed / T_LOOP_SECONDS) % 1;
      const t = cycleProgress * T_MAX_FS;
      tFsRef.current = t;
      setTFs(t);
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [playing]);

  // When the user scrubs, sync the ref so the next frame picks up the new t
  const onScrub = (e) => {
    const v = parseFloat(e.target.value);
    tFsRef.current = v;
    setTFs(v);
    setPlaying(false);
  };

  const occ = hopOccupancies(tFs * 1e-15);

  return (
    <div className="relative w-full rounded-xl overflow-hidden border
      border-dark/10 dark:border-light/10 bg-dark"
      style={{ height }}>

      <Canvas
        camera={{ fov: 45, near: 0.1, far: 500, position: [60, 35, 60] }}
        dpr={[1, 1.5]}
      >
        <color attach="background" args={["#0a0a12"]} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[40, 50, 40]} intensity={0.9} />
        <pointLight position={[-30, 20, -30]} intensity={0.45} color="#88aaff" />

        <Suspense fallback={null}>
          <GLBStructure wireframe={showWireframe} />
        </Suspense>

        <ChainAxis />
        <CofactorMarkers occupancies={occ} />
        <ElectronCloud tFsRef={tFsRef} />
        <TravelingElectron tFsRef={tFsRef} />

        <CameraRig />
        <OrbitControls
          target={[24.5, 16.5, 30.5]}
          enablePan
          enableDamping
          dampingFactor={0.08}
          maxDistance={120}
          minDistance={6}
        />
        <AdaptiveDpr pixelated />
      </Canvas>

      {/* HUD — top-left: time and cofactor occupancies */}
      <div className="absolute top-3 left-3 px-3 py-2 rounded-md
        bg-dark/70 backdrop-blur text-[11px] text-light/85 font-mono
        space-y-0.5 pointer-events-none">
        <div className="flex items-center gap-2">
          <span className="text-primaryDark">t =</span>
          <span className="font-bold tabular-nums">{tFs.toFixed(1)} fs</span>
        </div>
        {COFACTORS.map((c, i) => (
          <div key={c.name} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full"
              style={{ background: c.color }}></span>
            <span className="w-12 inline-block">{c.name}</span>
            <span className="tabular-nums">{(occ[i] * 100).toFixed(1)}%</span>
            <div className="flex-1 h-1 bg-light/15 rounded-full overflow-hidden ml-1
              w-16">
              <div className="h-full transition-all"
                style={{ width: `${Math.min(100, occ[i] * 100)}%`,
                         background: c.color }}></div>
            </div>
          </div>
        ))}
      </div>

      {/* HUD — top-right: badge */}
      <div className="absolute top-3 right-3 px-3 py-1.5 rounded-md
        bg-primaryDark/15 border border-primaryDark/40
        text-[10px] uppercase tracking-widest text-primaryDark
        pointer-events-none">
        live · GLB-grounded · L5 readout
      </div>

      {/* HUD — bottom: timeline + play/pause */}
      <div className="absolute bottom-0 left-0 right-0 px-4 py-3
        bg-gradient-to-t from-dark/85 to-transparent flex items-center gap-3">
        <button
          onClick={() => setPlaying(!playing)}
          className="w-9 h-9 rounded-full bg-primaryDark text-dark flex items-center
            justify-center text-base hover:scale-105 transition-transform shrink-0"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <input
          type="range"
          min={0}
          max={T_MAX_FS}
          step={1}
          value={tFs}
          onChange={onScrub}
          className="flex-1 accent-primaryDark cursor-pointer"
        />
        <div className="text-[10px] text-light/65 font-mono tabular-nums shrink-0
          min-w-[3.5rem] text-right">
          {Math.round(tFs)} / {T_MAX_FS} fs
        </div>
      </div>
    </div>
  );
}

// Pre-load the GLB so the first render isn't blocked by network latency
useGLTF.preload(GLB_URL);
