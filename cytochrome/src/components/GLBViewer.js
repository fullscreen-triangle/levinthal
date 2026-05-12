// Lightweight static GLB viewer — auto-rotating structure with optional
// cofactor markers. No animation timeline; see ElectronTransferViewer.js
// for the full animated transfer viewer used on Transfer / GLB-input.

import { useMemo, useEffect, Suspense } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Html, AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";

const GLB_URL = "/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb";

const PRESETS = {
  full:  { cam: [60, 35, 60],  tgt: [24.5, 16.5, 30.5] },
  heme:  { cam: [32, 22, 36],  tgt: [17.3, 11.5, 24.8] },
  chain: { cam: [48, 30, 50],  tgt: [24.5, 16.5, 30.5] },
};

function GLBStructure() {
  const { scene } = useGLTF(GLB_URL);
  const cloned = useMemo(() => {
    const c = scene.clone(true);
    c.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        const m = obj.material.clone();
        m.transparent = true;
        m.opacity = 0.26;
        m.depthWrite = false;
        m.side = THREE.DoubleSide;
        obj.material = m;
      }
    });
    return c;
  }, [scene]);
  return <primitive object={cloned} />;
}

function Markers({ markers = [] }) {
  return (
    <group>
      {markers.map((m) => {
        const r = m.radius ?? 0.9;
        return (
          <group key={m.name} position={m.pos}>
            <mesh>
              <sphereGeometry args={[r, 20, 20]} />
              <meshStandardMaterial
                color={m.color}
                emissive={m.color}
                emissiveIntensity={m.glow ?? 0.7}
                roughness={0.3}
                metalness={0.05}
              />
            </mesh>
            <Html
              distanceFactor={20}
              position={[0, r + 1.3, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <div
                className="text-[10px] font-mono uppercase tracking-widest
                  bg-dark/80 px-2 py-0.5 rounded whitespace-nowrap"
                style={{ color: m.color }}
              >
                {m.name}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

function CameraRig({ cam, tgt }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(...cam);
    camera.lookAt(...tgt);
  }, []);
  return null;
}

export default function GLBViewer({
  height = 320,
  badge = "GLB · CYP3A4",
  preset = "full",
  markers = [],
}) {
  const { cam, tgt } = PRESETS[preset] ?? PRESETS.full;

  return (
    <div
      className="relative w-full rounded-xl overflow-hidden border
        border-dark/10 dark:border-light/10 bg-dark"
      style={{ height }}
    >
      <Canvas
        camera={{ fov: 45, near: 0.1, far: 500, position: cam }}
        dpr={[1, 1.5]}
      >
        <color attach="background" args={["#0a0a12"]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[40, 50, 40]} intensity={0.8} />
        <pointLight position={[-30, 20, -30]} intensity={0.4} color="#88aaff" />

        <Suspense fallback={null}>
          <GLBStructure />
        </Suspense>

        <Markers markers={markers} />
        <CameraRig cam={cam} tgt={tgt} />

        <OrbitControls
          target={tgt}
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          maxDistance={120}
          minDistance={6}
          autoRotate
          autoRotateSpeed={0.4}
        />
        <AdaptiveDpr pixelated />
      </Canvas>

      <div
        className="absolute top-3 right-3 px-3 py-1.5 rounded-md
          bg-primaryDark/15 border border-primaryDark/40
          text-[10px] uppercase tracking-widest text-primaryDark
          pointer-events-none"
      >
        {badge}
      </div>
    </div>
  );
}

useGLTF.preload(GLB_URL);
