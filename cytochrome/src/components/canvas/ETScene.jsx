/**
 * ETScene — Electron Transfer Chain 3D scene
 * Holographic P450 GLB with scroll-driven explosion and model swap.
 *
 * Props:
 *   sectionsRef  {Object}  { heroProgress, explodeProgress, chainProgress }
 *                          scalar 0→1 values updated by GSAP ScrollTrigger
 */

import { useRef, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import * as THREE from 'three'

// ─── shared scroll state (written by GSAP, read by useFrame) ─────────────────
export const etScrollState = {
  heroProgress: 0,
  explodeProgress: 0,
  chainProgress: 0,
  time: 0,
}

// ─── holographic ShaderMaterial (inline GLSL, no file imports needed) ────────
function makeHoloMaterial(color = '#00c8ff') {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime:  { value: 0 },
      uColor: { value: new THREE.Color(color) },
      uAlpha: { value: 1.0 },
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      varying vec3 vNormalW;
      varying vec3 vPositionW;
      void main() {
        vUv = uv;
        vNormalW  = normalize(mat3(modelMatrix) * normal);
        vPositionW = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uTime;
      uniform vec3  uColor;
      uniform float uAlpha;
      varying vec2  vUv;
      varying vec3  vNormalW;
      varying vec3  vPositionW;

      float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
      }

      void main() {
        // Fresnel rim
        vec3 viewDir = normalize(cameraPosition - vPositionW);
        float fresnel = clamp(1.0 - dot(viewDir, vNormalW), 0.0, 1.0);
        fresnel = pow(fresnel, 2.0);

        // Scanlines
        float scan = sin(vUv.y * 80.0 + uTime * 2.5) * 0.5 + 0.5;
        scan      *= smoothstep(0.3, 0.7, sin(vUv.y * 40.0 - uTime * 1.2));

        // RGB noise per scanline
        float r = rand(vec2(vUv.y * 20.0, uTime));
        float noise = r * scan * 0.12;

        // Compose
        vec3 col = uColor * (0.3 + 0.7 * fresnel) + uColor * scan * 0.4 + noise;
        float alpha = (0.18 + 0.5 * fresnel + 0.18 * scan) * uAlpha;

        gl_FragColor = vec4(col, alpha);
      }
    `,
    transparent:  true,
    blending:     THREE.AdditiveBlending,
    depthWrite:   false,
    side:         THREE.DoubleSide,
  })
}

// ─── cofactor accent colors ──────────────────────────────────────────────────
const COFACTOR_COLORS = {
  nadph: '#854794', // purple
  fad:   '#00A8DE', // blue
  fmn:   '#54AE37', // green
  heme:  '#E84750', // red
}

// ─── main scene ──────────────────────────────────────────────────────────────
export default function ETScene() {
  const { scene: p450Scene } = useGLTF(
    '/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb'
  )
  const { scene: cytoCScene } = useGLTF(
    '/glb/practice_molecules_cytochrome_c.glb'
  )
  const { scene: hemeScene } = useGLTF(
    '/glb/cytochrome_p450_with_haem_highlighted.glb'
  )

  const p450Group   = useRef()
  const hemeGroup   = useRef()
  const cytoCGroup  = useRef()
  const holoMat     = useRef()
  const meshDataRef = useRef([]) // { mesh, original, target }

  // ── build holographic material once ───────────────────────────────────────
  useEffect(() => {
    holoMat.current = makeHoloMaterial('#00c8ff')
  }, [])

  // ── clone scenes so we don't mutate the GLTF cache ────────────────────────
  const p450Clone  = useMemo(() => p450Scene.clone(true),  [p450Scene])
  const hemeClone  = useMemo(() => hemeScene.clone(true),  [hemeScene])
  const cytoCClone = useMemo(() => cytoCScene.clone(true), [cytoCScene])

  // ── apply holographic material + compute explosion vectors ─────────────────
  useEffect(() => {
    if (!holoMat.current) return

    const explosionCenter = new THREE.Vector3(17.26, 11.53, 24.76)
    const explosionFactor = 1.1
    meshDataRef.current = []

    p450Clone.traverse((child) => {
      if (!child.isMesh) return

      // assign holographic shader
      child.material = holoMat.current.clone()
      child.material.uniforms.uAlpha.value = 0.92

      // compute explosion target
      const orig   = child.position.clone()
      const dir    = orig.clone().sub(explosionCenter).normalize()
      const dist   = orig.distanceTo(explosionCenter)
      const target = orig.clone().add(dir.multiplyScalar(dist * explosionFactor))

      meshDataRef.current.push({ mesh: child, original: orig, target })
    })

    // heme model: wire-only teal
    hemeClone.traverse((child) => {
      if (!child.isMesh) return
      child.material = makeHoloMaterial('#7fffd4')
    })

    // cytochrome c: gold accent
    cytoCClone.traverse((child) => {
      if (!child.isMesh) return
      child.material = makeHoloMaterial('#ffd700')
    })
  }, [p450Clone, hemeClone, cytoCClone])

  // ── frame loop ────────────────────────────────────────────────────────────
  useFrame(({ clock }) => {
    const t    = clock.getElapsedTime()
    const hero = etScrollState.heroProgress
    const expl = etScrollState.explodeProgress
    const chn  = etScrollState.chainProgress

    // update time uniform on all holographic materials
    meshDataRef.current.forEach(({ mesh }) => {
      if (mesh.material?.uniforms?.uTime) {
        mesh.material.uniforms.uTime.value = t
      }
    })

    // ── p450 group ──────────────────────────────────────────────────────────
    if (p450Group.current) {
      // slow base rotation
      p450Group.current.rotation.y = t * 0.12 + hero * Math.PI * 0.5
      // zoom in slightly during hero
      p450Group.current.scale.setScalar(0.035 + hero * 0.005)
      // fade out as chain section comes in
      p450Group.current.visible = chn < 0.8
    }

    // ── mesh explosion ───────────────────────────────────────────────────────
    meshDataRef.current.forEach(({ mesh, original, target }) => {
      mesh.position.lerpVectors(original, target, expl)
      // glow brighter when exploded
      if (mesh.material?.uniforms?.uAlpha) {
        mesh.material.uniforms.uAlpha.value = 0.5 + expl * 0.5
      }
    })

    // ── heme group (hero background, fades out) ──────────────────────────────
    if (hemeGroup.current) {
      hemeGroup.current.rotation.y = -t * 0.07
      hemeGroup.current.rotation.x = Math.sin(t * 0.3) * 0.08
      hemeGroup.current.visible = hero < 0.5
    }

    // ── cytochrome c (chain / Marcus section) ──────────────────────────────
    if (cytoCGroup.current) {
      cytoCGroup.current.rotation.y = t * 0.2
      cytoCGroup.current.opacity    = chn
      cytoCGroup.current.visible    = chn > 0.05
    }
  })

  return (
    <>
      {/* ambient fill */}
      <ambientLight intensity={0.15} />
      <pointLight position={[40, 30, 20]} intensity={1.2} color="#3bf" />
      <pointLight position={[-30, -20, 10]} intensity={0.6} color="#f3b" />

      {/* heme-highlighted P450 — hero background */}
      <group ref={hemeGroup} position={[0, 0, 0]} scale={0.04}>
        <primitive object={hemeClone} />
      </group>

      {/* oxygen-drug-complex P450 — main explodable model */}
      <group ref={p450Group} position={[0, 0, 0]} scale={0.035}>
        <primitive object={p450Clone} />
      </group>

      {/* cytochrome c — chain / Marcus section */}
      <group ref={cytoCGroup} position={[3, 0, 0]} scale={0.3} visible={false}>
        <primitive object={cytoCClone} />
      </group>
    </>
  )
}

// preload all three models
useGLTF.preload('/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb')
useGLTF.preload('/glb/practice_molecules_cytochrome_c.glb')
useGLTF.preload('/glb/cytochrome_p450_with_haem_highlighted.glb')
