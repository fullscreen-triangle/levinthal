// =====================================================================
//  StructureOutput — a folded/complexed protein IS legitimate script
//  output. When a play produces a Fold, Complex, or Trajectory, the
//  receiver's structure is rendered here as the live 3D GLB model, with
//  the cofactor/heme markers the play committed.
// =====================================================================
import { useMemo, Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Html, AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";

// which GLB + camera per structure kind emitted by the interpreter
export const STRUCTURE_PRESETS = {
  fold: {
    url: "/glb/cytochrome_p450_with_haem_highlighted.glb",
    cam: [58, 34, 58],
    tgt: [24.5, 16.5, 30.5],
    label: "CYP3A4 fold · heme highlighted",
  },
  resting: {
    url: "/glb/cytochrome_p450_with_haem_highlighted.glb",
    cam: [34, 22, 38],
    tgt: [17.3, 11.5, 24.8],
    label: "resting complex · Fe³⁺ low-spin",
  },
  cycle: {
    url: "/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb",
    cam: [60, 35, 60],
    tgt: [24.5, 16.5, 30.5],
    label: "catalytic complex · O₂ + drug",
  },
  chain: {
    url: "/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb",
    cam: [48, 30, 50],
    tgt: [24.5, 16.5, 30.5],
    label: "electron-transfer chain",
  },
};

function Structure({ url, opacity = 0.24 }) {
  const { scene } = useGLTF(url);
  const cloned = useMemo(() => {
    const c = scene.clone(true);
    c.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        const m = obj.material.clone();
        m.transparent = true;
        m.opacity = opacity;
        m.depthWrite = false;
        m.side = THREE.DoubleSide;
        obj.material = m;
      }
    });
    return c;
  }, [scene, opacity]);
  return <primitive object={cloned} />;
}

function Markers({ markers = [] }) {
  return (
    <group>
      {markers.map((mk) => {
        const r = mk.radius ?? 1.0;
        return (
          <group key={mk.name} position={mk.pos}>
            <mesh>
              <sphereGeometry args={[r, 18, 18]} />
              <meshStandardMaterial
                color={mk.color}
                emissive={mk.color}
                emissiveIntensity={mk.glow ?? 0.8}
                roughness={0.3}
                metalness={0.05}
              />
            </mesh>
            <Html distanceFactor={22} position={[0, r + 1.2, 0]} center style={{ pointerEvents: "none" }}>
              <div style={{ fontSize: 10, color: "#e5e7eb", whiteSpace: "nowrap", textShadow: "0 0 4px #000" }}>
                {mk.name}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

export default function StructureOutput({ data }) {
  // data: { kind, markers }
  const preset = STRUCTURE_PRESETS[data?.kind] ?? STRUCTURE_PRESETS.fold;
  return (
    <div className="rounded-md border border-neutral-700 bg-[#0c0c0c]">
      <div className="flex items-center justify-between px-2 py-1 text-[11px] uppercase tracking-wider text-neutral-400">
        <span>Structure output</span>
        <span className="text-[#58E6D9] normal-case tracking-normal">{preset.label}</span>
      </div>
      <div style={{ height: 300 }}>
        <Canvas camera={{ position: preset.cam, fov: 42, near: 0.1, far: 2000 }} dpr={[1, 1.6]} gl={{ antialias: true }}>
          <color attach="background" args={["#0c0c0c"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[40, 60, 40]} intensity={0.9} />
          <directionalLight position={[-30, -20, -40]} intensity={0.3} />
          <Suspense
            fallback={
              <Html center>
                <div style={{ color: "#6b7280", fontSize: 11 }}>loading structure…</div>
              </Html>
            }
          >
            <group position={[-preset.tgt[0], -preset.tgt[1], -preset.tgt[2]]}>
              <Structure url={preset.url} />
              <Markers markers={data?.markers ?? []} />
            </group>
          </Suspense>
          <OrbitControls target={[0, 0, 0]} enablePan={false} autoRotate autoRotateSpeed={0.6} minDistance={12} maxDistance={140} />
          <AdaptiveDpr pixelated />
        </Canvas>
      </div>
    </div>
  );
}

// preload the two structures used most
useGLTF.preload?.("/glb/cytochrome_p450_with_haem_highlighted.glb");
useGLTF.preload?.("/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb");
