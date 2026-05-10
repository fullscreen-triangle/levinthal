'use client'
import React, { useRef } from 'react'
import { useGLTF } from '@react-three/drei'

export function Model(props) {
  const { nodes, materials } = useGLTF('/glb/practice_molecules_cytochrome_c.glb')
  return (
    <group {...props} dispose={null}>
      <group rotation={[-Math.PI / 2, 0, 0]} scale={0.013}>
        <group rotation={[Math.PI / 2, 0, 0]}>
          <group
            position={[-4.028, 20.067, 18.802]}
            rotation={[Math.PI / 2, 0, -Math.PI]}
            scale={0.525}>
            <mesh
              castShadow
              receiveShadow
              geometry={nodes.Object_4.geometry}
              material={materials['Material.002']}
            />
            <mesh
              castShadow
              receiveShadow
              geometry={nodes.Object_5.geometry}
              material={materials['Material.002']}
            />
            <mesh
              castShadow
              receiveShadow
              geometry={nodes.Object_6.geometry}
              material={materials['Material.003']}
            />
            <mesh
              castShadow
              receiveShadow
              geometry={nodes.Object_7.geometry}
              material={materials['Material.001']}
            />
          </group>
        </group>
      </group>
    </group>
  )
}

useGLTF.preload('/glb/practice_molecules_cytochrome_c.glb')