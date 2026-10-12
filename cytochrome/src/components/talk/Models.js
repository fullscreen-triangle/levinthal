// 3-D models for the talk, reusing the site's GLB files. Client-only: import
// through next/dynamic with ssr: false.

import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Bounds, Center, OrbitControls, useAnimations, useGLTF, AdaptiveDpr } from "@react-three/drei";
import * as THREE from "three";

export const GLB = {
  heme: "/glb/cytochrome_p450_with_haem_highlighted.glb",
  oxyDrug: "/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb",
};

// The heme-highlighted model carries its own animation clip; play it looped.
function AnimatedHemeModel({ speed = 1 }) {
  const group = useRef();
  const { scene, animations } = useGLTF(GLB.heme);
  const cloned = useMemo(() => scene.clone(true), [scene]);
  const { actions, mixer } = useAnimations(animations, group);
  useEffect(() => {
    Object.values(actions).forEach((a) => {
      if (!a) return;
      a.reset().setLoop(THREE.LoopRepeat, Infinity).play();
    });
    mixer.timeScale = speed;
    return () => Object.values(actions).forEach((a) => a && a.stop());
  }, [actions, mixer, speed]);
  return (
    <group ref={group}>
      <primitive object={cloned} />
    </group>
  );
}

export function AnimatedHeme({ height = 320, autoRotate = true, speed = 1 }) {
  return (
    <div style={{ height, width: "100%" }}>
      <Canvas camera={{ fov: 40, near: 0.1, far: 2000, position: [0, 0, 60] }} dpr={[1, 1.5]} gl={{ alpha: true }}>
        <hemisphereLight args={["#dfe9ff", "#26221c", 1.4]} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[40, 50, 40]} intensity={2.4} />
        <directionalLight position={[-40, -20, 30]} intensity={1.2} color="#9fc2ff" />
        <pointLight position={[-30, 20, -30]} intensity={1.0} color="#88aaff" />
        <Suspense fallback={null}>
          <Bounds fit clip observe margin={1.15}>
            <Center>
              <AnimatedHemeModel speed={speed} />
            </Center>
          </Bounds>
        </Suspense>
        <OrbitControls makeDefault enablePan={false} enableZoom={false} autoRotate={autoRotate} autoRotateSpeed={0.7} />
        <AdaptiveDpr pixelated />
      </Canvas>
    </div>
  );
}

// Load every model into drei's cache so later slides render without a pause.
export function preloadModels() {
  Object.values(GLB).forEach((u) => useGLTF.preload(u));
}
