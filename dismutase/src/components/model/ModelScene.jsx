'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, PerspectiveCamera } from '@react-three/drei';
import { ConformationalChange } from './ConformationalChange';
import { Suspense } from 'react';

export default function ModelScene({ modelPath }) {
  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 5]} />
        
        {/* Beleuchtung */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <pointLight position={[-10, -10, -5]} intensity={0.5} />
        
        {/* Suspense für Lazy Loading - nutzt dein App-Loading */}
        <Suspense fallback={null}>
          <ConformationalChange 
            modelPath={modelPath} 
            scale={1} 
            position={[0, 0, 0]} 
          />
          
          {/* Environment für bessere Reflexionen */}
          <Environment preset="sunset" />
        </Suspense>
        
        {/* Kamera-Steuerung */}
        <OrbitControls 
          enableZoom={true}
          enablePan={true}
          enableRotate={true}
        />
      </Canvas>
    </div>
  );
}
