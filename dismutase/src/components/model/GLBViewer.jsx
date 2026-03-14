'use client'
import { Environment, OrbitControls, useAnimations, useGLTF } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

function Model({ url, autoRotate }) {
    const group = useRef();
    const { scene, animations } = useGLTF(url);
    const { actions, names } = useAnimations(animations, group);

    useEffect(() => {
        if (names.length > 0) {
            console.log("Available animations:", names);
            names.forEach((name) => {
                actions[name]?.play();
            });
        }
    }, [actions, names]);

    useFrame((state, delta) => {
        if (autoRotate && group.current) {
            group.current.rotation.y += delta * 0.5;
        }
    });

    useEffect(() => {
        if (scene) {
            const box = new THREE.Box3().setFromObject(scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            // Center the model
            scene.position.sub(center);

            // Adjust this scale value to make model smaller/larger
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim; // Changed from 3 to 2 for smaller model
            scene.scale.setScalar(scale);

            console.log("Model size:", size);
            console.log("Model scale:", scale);
        }
    }, [scene]);

    return <primitive ref={group} object={scene} />;
}

export default function GLBViewer({
    modelPath = "/glb/conformational_transition_of_troponin.glb",
    autoRotate = false,
    showControls = true,
    backgroundColor = "#1a1a1a"
}) {
    const [error, setError] = useState(null);

    return (
        <div style={{ width: "100%", height: "100vh", background: backgroundColor }}>
            {error && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    color: '#ff4444',
                    fontSize: '1.2rem',
                    zIndex: 100,
                    textAlign: 'center',
                    padding: '20px'
                }}>
                    Error loading model: {error}
                </div>
            )}

            <Canvas
                shadows
                camera={{ 
                    position: [0, 0, 10],  // Move camera further back (was 5)
                    fov: 45,               // Narrower FOV (was 50) - less distortion
                    near: 0.1,             // Near clipping plane
                    far: 1000              // Far clipping plane
                }}
                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: "high-performance"
                }}
                onCreated={({ gl }) => {
                    gl.setClearColor(backgroundColor);
                }}
            >
                {/* Lighting Setup */}
                <ambientLight intensity={0.6} />
                <directionalLight
                    position={[10, 10, 5]}
                    intensity={1}
                    castShadow
                    shadow-mapSize-width={1024}
                    shadow-mapSize-height={1024}
                />
                <pointLight position={[-10, -10, -5]} intensity={0.3} />
                <spotLight
                    position={[0, 10, 0]}
                    angle={0.3}
                    penumbra={1}
                    intensity={0.5}
                    castShadow
                />

                {/* Environment for realistic reflections */}
                <Environment preset="city" />

                {/* The 3D Model */}
                <Model url={modelPath} autoRotate={autoRotate} />

                {/* Add OrbitControls if you want user interaction */}
                {showControls && (
                    <OrbitControls
                        enableZoom={true}
                        enablePan={true}
                        minDistance={5}    // Minimum zoom distance
                        maxDistance={20}   // Maximum zoom distance
                        target={[0, 0, 0]} // Look at center
                    />
                )}
            </Canvas>
        </div>
    );
}
