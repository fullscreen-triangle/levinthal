import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, MeshDistortMaterial, MeshWobbleMaterial, OrbitControls, Environment } from '@react-three/drei';
import { Suspense } from 'react';
import * as THREE from 'three';

const twistMaterial = (material, callback) => {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = { value: 0 };
    shader.uniforms.uIntensity = { value: 0.3 };
    shader.uniforms.uFrequency = { value: 0.5 };
    
    shader.vertexShader = shader.vertexShader.replace(
      "#include <common>",
      `
        #include <common>
        uniform float uTime;
        uniform float uIntensity;
        uniform float uFrequency;
        
        mat2 get2dRotateMatrix(float _angle) {
          return mat2(cos(_angle), -sin(_angle), sin(_angle), cos(_angle));
        }
      `
    );
    
    shader.vertexShader = shader.vertexShader.replace(
      "#include <beginnormal_vertex>",
      `
        #include <beginnormal_vertex>
        float angle = (position.y + uTime * uFrequency) * uIntensity;
        mat2 rotateMatrix = get2dRotateMatrix(angle);
        objectNormal.xz = rotateMatrix * objectNormal.xz;
      `
    );
    
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `
        #include <begin_vertex>
        transformed.xz = rotateMatrix * transformed.xz;
      `
    );

    if (callback) callback(shader);
  };
};

function BookModel(props) {
  const { nodes, materials } = useGLTF('/models/the_works_of_william_shakespeare.glb');
  const groupRef = useRef();
  const materialRef = useRef();
  const shaderRef = useRef();
  const [animationPhase, setAnimationPhase] = useState(0);

  // Cycle through different animation phases
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationPhase(prev => (prev + 1) % 4);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Apply twist material effect
  useEffect(() => {
    if (materialRef.current && animationPhase === 1) {
      twistMaterial(materialRef.current, (shader) => {
        shaderRef.current = shader;
      });
    }
  }, [animationPhase]);

  useFrame((state, delta) => {
    if (groupRef.current) {
      // Gentle rotation
      groupRef.current.rotation.y += delta * 0.2;
      
      // Floating animation
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }

    // Update twist shader
    if (shaderRef.current) {
      shaderRef.current.uniforms.uTime.value += delta;
    }
  });

  const renderMaterial = () => {
    switch (animationPhase) {
      case 0: // Normal
        return <meshStandardMaterial color="#8B4513" roughness={0.8} metalness={0.1} />;
      case 1: // Twist
        return <meshStandardMaterial ref={materialRef} color="#8B4513" roughness={0.8} metalness={0.1} />;
      case 2: // Distort
        return (
          <MeshDistortMaterial
            color="#8B4513"
            roughness={0.8}
            metalness={0.1}
            speed={2}
            distort={0.3}
          />
        );
      case 3: // Wobble
        return (
          <MeshWobbleMaterial
            color="#8B4513"
            roughness={0.8}
            metalness={0.1}
            speed={1}
            factor={0.2}
          />
        );
      default:
        return <meshStandardMaterial color="#8B4513" roughness={0.8} metalness={0.1} />;
    }
  };

  return (
    <group ref={groupRef} {...props} dispose={null}>
      <group rotation={[-Math.PI / 2, 0, 0]} scale={[2, 2, 2]}>
        {/* Main book meshes */}
        <mesh castShadow receiveShadow geometry={nodes.Object_2.geometry}>
          {renderMaterial()}
        </mesh>
        <mesh castShadow receiveShadow geometry={nodes.Object_3.geometry}>
          {renderMaterial()}
        </mesh>
        <mesh castShadow receiveShadow geometry={nodes.Object_4.geometry}>
          {renderMaterial()}
        </mesh>
        <mesh castShadow receiveShadow geometry={nodes.Object_5.geometry}>
          {renderMaterial()}
        </mesh>
        <mesh castShadow receiveShadow geometry={nodes.Object_6.geometry}>
          {renderMaterial()}
        </mesh>
        <mesh castShadow receiveShadow geometry={nodes.Object_7.geometry}>
          {renderMaterial()}
        </mesh>
      </group>
      
      {/* Floating particles around the book */}
      {[...Array(20)].map((_, i) => (
        <mesh
          key={i}
          position={[
            Math.sin(i * 0.5) * 3,
            Math.cos(i * 0.3) * 2,
            Math.sin(i * 0.7) * 2
          ]}
        >
          <sphereGeometry args={[0.02, 8, 8]} />
          <meshStandardMaterial 
            color={`hsl(${200 + i * 10}, 70%, 60%)`}
            emissive={`hsl(${200 + i * 10}, 70%, 20%)`}
          />
        </mesh>
      ))}
    </group>
  );
}

export function AnimatedShakespeareBook({ className = "" }) {
  return (
    <div className={`h-96 w-full ${className}`}>
      <Canvas
        camera={{ 
          position: [0, 2, 8], 
          fov: 45,
          near: 0.1,
          far: 1000
        }}
        shadows
        gl={{ 
          antialias: true,
          alpha: true,
          powerPreference: "high-performance"
        }}
      >
        <Suspense fallback={null}>
          {/* Lighting Setup */}
          <ambientLight intensity={0.3} />
          <directionalLight 
            position={[10, 10, 5]} 
            intensity={1}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <pointLight 
            position={[-10, -10, -5]} 
            intensity={0.5}
            color="#4A90E2"
          />
          <spotLight
            position={[0, 10, 0]}
            angle={0.3}
            penumbra={1}
            intensity={0.5}
            castShadow
            color="#FFD700"
          />
          
          {/* The Book Model */}
          <BookModel />
          
          {/* Controls */}
          <OrbitControls 
            enableZoom={false}
            enablePan={false}
            autoRotate={false}
            maxPolarAngle={Math.PI / 2}
            minPolarAngle={Math.PI / 4}
          />
          
          {/* Environment */}
          <Environment preset="studio" />
          
          {/* Ground plane for shadows */}
          <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
            <planeGeometry args={[20, 20]} />
            <shadowMaterial opacity={0.1} />
          </mesh>
        </Suspense>
      </Canvas>
    </div>
  );
}

useGLTF.preload('/models/the_works_of_william_shakespeare.glb');
