import React, { useRef } from 'react'
import { useGLTF } from '@react-three/drei'

export function Model(props) {
  const { nodes, materials } = useGLTF('/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb')
  return (
    <group {...props} dispose={null}>
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <group rotation={[Math.PI / 2, 0, 0]}>
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color.geometry}
            material={materials.material_1}
            position={[20.018, 7.514, 26.442]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_1.geometry}
            material={materials.material_1}
            position={[18.772, 8.587, 23.857]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_2.geometry}
            material={materials.material_2}
            position={[17.125, 9.408, 24.089]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_3.geometry}
            material={materials.material_3}
            position={[20.039, 11.305, 26.75]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_4.geometry}
            material={materials.material_3}
            position={[19.261, 12.376, 22.091]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_5.geometry}
            material={materials.material_3}
            position={[14.565, 11.89, 22.857]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_6.geometry}
            material={materials.material_3}
            position={[15.254, 11.216, 27.561]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_7.geometry}
            material={materials.material_3}
            position={[20.243, 11.656, 25.431]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_8.geometry}
            material={materials.material_3}
            position={[21.531, 11.88, 24.792]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_9.geometry}
            material={materials.material_3}
            position={[21.288, 12.204, 23.526]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_10.geometry}
            material={materials.material_3}
            position={[19.849, 12.181, 23.322]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_11.geometry}
            material={materials.material_3}
            position={[22.274, 12.55, 22.39]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_12.geometry}
            material={materials.material_3}
            position={[22.877, 11.83, 25.568]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_13.geometry}
            material={materials.material_3}
            position={[23.468, 13.153, 26.021]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_14.geometry}
            material={materials.material_3}
            position={[24.853, 13.059, 26.722]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_15.geometry}
            material={materials.light_0}
            position={[25.537, 12.031, 26.561]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_16.geometry}
            material={materials.light_0}
            position={[25.156, 14.064, 27.418]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_17.geometry}
            material={materials.material_3}
            position={[17.911, 12.306, 21.869]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_18.geometry}
            material={materials.material_3}
            position={[17.26, 12.475, 20.586]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_19.geometry}
            material={materials.material_3}
            position={[15.949, 12.325, 20.793]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_20.geometry}
            material={materials.material_3}
            position={[15.751, 12.086, 22.206]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_21.geometry}
            material={materials.material_3}
            position={[18.021, 12.757, 19.291]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_22.geometry}
            material={materials.material_3}
            position={[14.815, 12.443, 19.769]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_23.geometry}
            material={materials.material_3}
            position={[15.082, 12.283, 18.444]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_24.geometry}
            material={materials.material_3}
            position={[14.311, 11.708, 24.174]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_25.geometry}
            material={materials.material_3}
            position={[13.004, 11.603, 24.8]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_26.geometry}
            material={materials.material_3}
            position={[13.202, 11.434, 26.112]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_27.geometry}
            material={materials.material_3}
            position={[14.63, 11.459, 26.349]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_28.geometry}
            material={materials.material_3}
            position={[11.688, 11.664, 23.988]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_29.geometry}
            material={materials.material_3}
            position={[12.126, 11.246, 27.196]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_30.geometry}
            material={materials.material_3}
            position={[11.964, 12.151, 28.159]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_31.geometry}
            material={materials.material_3}
            position={[16.632, 11.097, 27.718]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_32.geometry}
            material={materials.material_3}
            position={[17.301, 10.802, 28.974]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_33.geometry}
            material={materials.material_3}
            position={[18.611, 10.897, 28.771]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_34.geometry}
            material={materials.material_3}
            position={[18.817, 11.123, 27.352]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_35.geometry}
            material={materials.material_3}
            position={[16.516, 10.602, 30.295]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_36.geometry}
            material={materials.material_3}
            position={[19.746, 10.706, 29.796]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_37.geometry}
            material={materials.material_3}
            position={[20.497, 9.377, 29.825]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_38.geometry}
            material={materials.material_3}
            position={[21.703, 9.481, 30.833]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_39.geometry}
            material={materials.light_0}
            position={[22.068, 8.361, 31.202]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_40.geometry}
            material={materials.light_0}
            position={[22.108, 10.617, 31.077]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_41.geometry}
            material={materials.material_5}
            position={[19.237, 11.785, 24.494]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_42.geometry}
            material={materials.material_5}
            position={[16.967, 12.083, 22.846]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_43.geometry}
            material={materials.material_5}
            position={[15.259, 11.634, 25.151]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_44.geometry}
            material={materials.material_5}
            position={[17.577, 11.277, 26.737]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_45.geometry}
            material={materials.material_6}
            position={[17.259, 11.527, 24.763]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.63}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_46.geometry}
            material={materials.light_0}
            position={[16.996, 13.281, 25.146]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.355}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_47.geometry}
            material={materials.light_0}
            position={[16.135, 14.189, 25.077]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.355}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_48.geometry}
            material={materials.material_7}
            position={[19.314, 17.039, 26.85]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_49.geometry}
            material={materials.material_7}
            position={[20.182, 16.998, 28.138]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_50.geometry}
            material={materials.light_0}
            position={[20.289, 17.893, 28.964]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_51.geometry}
            material={materials.material_7}
            position={[20.63, 15.557, 27.99]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_52.geometry}
            material={materials.material_7}
            position={[20.097, 14.967, 26.683]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_53.geometry}
            material={materials.material_7}
            position={[18.677, 14.811, 27.263]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_54.geometry}
            material={materials.material_7}
            position={[18.086, 16.186, 27.169]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_55.geometry}
            material={materials.material_7}
            position={[20.017, 16.204, 25.726]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_56.geometry}
            material={materials.material_7}
            position={[21.387, 16.725, 25.339]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_57.geometry}
            material={materials.material_7}
            position={[19.095, 15.883, 24.538]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_58.geometry}
            material={materials.material_7}
            position={[19.116, 18.51, 26.461]}
            rotation={[-0.329, 1.073, -2.473]}
            scale={0.2}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_59.geometry}
            material={materials.material_3}
            position={[20.141, 11.48, 26.09]}
            rotation={[-1.294, 0.113, -0.15]}
            scale={[0.2, 0.69, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_60.geometry}
            material={materials.material_3}
            position={[19.428, 11.214, 27.051]}
            rotation={[0.513, 0.466, 1.471]}
            scale={[0.2, 0.687, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_61.geometry}
            material={materials.material_3}
            position={[19.555, 12.279, 22.706]}
            rotation={[1.492, -0.46, -0.497]}
            scale={[0.2, 0.689, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_62.geometry}
            material={materials.material_3}
            position={[18.586, 12.341, 21.98]}
            rotation={[-0.165, -0.169, 1.595]}
            scale={[0.2, 0.685, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_63.geometry}
            material={materials.material_3}
            position={[15.158, 11.988, 22.532]}
            rotation={[-0.536, 0.37, -1.196]}
            scale={[0.2, 0.684, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_64.geometry}
            material={materials.material_3}
            position={[14.438, 11.799, 23.515]}
            rotation={[1.667, 0.213, 0.193]}
            scale={[0.2, 0.676, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_65.geometry}
            material={materials.material_3}
            position={[14.942, 11.338, 26.955]}
            rotation={[-1.192, -0.342, 0.499]}
            scale={[0.2, 0.693, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_66.geometry}
            material={materials.material_3}
            position={[15.943, 11.157, 27.64]}
            rotation={[0.114, -0.123, -1.643]}
            scale={[0.2, 0.696, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_67.geometry}
            material={materials.material_3}
            position={[20.887, 11.768, 25.111]}
            rotation={[-0.486, 0.344, -1.222]}
            scale={[0.2, 0.728, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_68.geometry}
            material={materials.material_3}
            position={[19.994, 11.688, 25.199]}
            rotation={[-0.864, -0.469, 0.956]}
            scale={[0.2, 0.342, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_69.geometry}
            material={materials.material_5}
            position={[19.491, 11.753, 24.73]}
            rotation={[0.943, -0.577, -1.054]}
            scale={[0.2, 0.349, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_70.geometry}
            material={materials.material_3}
            position={[21.41, 12.042, 24.159]}
            rotation={[-1.294, -0.141, 0.186]}
            scale={[0.2, 0.665, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_71.geometry}
            material={materials.material_3}
            position={[22.204, 11.855, 25.18]}
            rotation={[0.592, -0.463, -1.316]}
            scale={[0.2, 0.777, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_72.geometry}
            material={materials.material_3}
            position={[20.569, 12.192, 23.424]}
            rotation={[-0.142, -0.142, 1.566]}
            scale={[0.2, 0.727, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_73.geometry}
            material={materials.material_3}
            position={[21.781, 12.377, 22.958]}
            rotation={[-0.922, 0.394, -0.764]}
            scale={[0.2, 0.772, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_74.geometry}
            material={materials.material_3}
            position={[19.697, 12.083, 23.612]}
            rotation={[1.584, 0.556, 0.549]}
            scale={[0.2, 0.342, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_75.geometry}
            material={materials.material_5}
            position={[19.391, 11.885, 24.198]}
            rotation={[-1.093, 0.297, -0.482]}
            scale={[0.2, 0.348, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_76.geometry}
            material={materials.material_3}
            position={[23.172, 12.491, 25.795]}
            rotation={[0.304, -0.062, -0.401]}
            scale={[0.2, 0.759, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_77.geometry}
            material={materials.material_3}
            position={[24.161, 13.106, 26.372]}
            rotation={[0.521, -0.441, -1.398]}
            scale={[0.2, 0.777, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_78.geometry}
            material={materials.material_3}
            position={[25.048, 12.766, 26.676]}
            rotation={[-0.142, 0.42, -2.497]}
            scale={[0.2, 0.355, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_79.geometry}
            material={materials.light_0}
            position={[25.39, 12.252, 26.595]}
            rotation={[0.13, 0.039, 0.582]}
            scale={[0.2, 0.268, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_80.geometry}
            material={materials.material_3}
            position={[24.94, 13.345, 26.92]}
            rotation={[0.588, -0.074, -0.244]}
            scale={[0.2, 0.359, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_81.geometry}
            material={materials.light_0}
            position={[25.091, 13.847, 27.268]}
            rotation={[-2.317, -0.718, 0.325]}
            scale={[0.2, 0.271, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_82.geometry}
            material={materials.material_3}
            position={[17.585, 12.39, 21.228]}
            rotation={[-1.247, -0.364, 0.501]}
            scale={[0.2, 0.725, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_83.geometry}
            material={materials.material_3}
            position={[17.677, 12.251, 22.111]}
            rotation={[1.058, 0.619, 1.001]}
            scale={[0.2, 0.341, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_84.geometry}
            material={materials.material_5}
            position={[17.205, 12.139, 22.6]}
            rotation={[-0.897, 0.432, -0.856]}
            scale={[0.2, 0.347, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_85.geometry}
            material={materials.material_3}
            position={[16.605, 12.4, 20.69]}
            rotation={[0.158, 0.173, 1.657]}
            scale={[0.2, 0.668, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_86.geometry}
            material={materials.material_3}
            position={[17.641, 12.616, 19.939]}
            rotation={[-1.135, 0.364, -0.562]}
            scale={[0.2, 0.764, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_87.geometry}
            material={materials.material_3}
            position={[15.85, 12.205, 21.5]}
            rotation={[1.716, 0.161, 0.139]}
            scale={[0.2, 0.723, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_88.geometry}
            material={materials.material_3}
            position={[15.382, 12.384, 20.281]}
            rotation={[-0.851, -0.477, 0.984]}
            scale={[0.2, 0.766, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_89.geometry}
            material={materials.material_3}
            position={[16.052, 12.085, 22.365]}
            rotation={[0.537, -0.426, -1.332]}
            scale={[0.2, 0.34, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_90.geometry}
            material={materials.material_5}
            position={[16.66, 12.084, 22.685]}
            rotation={[-0.537, -0.424, 1.329]}
            scale={[0.2, 0.347, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_91.geometry}
            material={materials.material_3}
            position={[14.948, 12.363, 19.107]}
            rotation={[-1.647, 0.218, -0.203]}
            scale={[0.2, 0.681, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_92.geometry}
            material={materials.material_3}
            position={[13.657, 11.656, 24.487]}
            rotation={[0.495, 0.431, 1.429]}
            scale={[0.2, 0.727, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_93.geometry}
            material={materials.material_3}
            position={[14.546, 11.69, 24.416]}
            rotation={[1.004, -0.555, -0.958]}
            scale={[0.2, 0.338, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_94.geometry}
            material={materials.material_5}
            position={[15.02, 11.653, 24.904]}
            rotation={[-0.95, -0.492, 0.909]}
            scale={[0.2, 0.344, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_95.geometry}
            material={materials.material_3}
            position={[13.103, 11.519, 25.456]}
            rotation={[1.673, -0.167, -0.151]}
            scale={[0.2, 0.668, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_96.geometry}
            material={materials.material_3}
            position={[12.346, 11.633, 24.394]}
            rotation={[-0.62, -0.444, 1.227]}
            scale={[0.2, 0.774, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_97.geometry}
            material={materials.material_3}
            position={[13.916, 11.447, 26.23]}
            rotation={[0.167, -0.16, -1.527]}
            scale={[0.2, 0.724, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_98.geometry}
            material={materials.material_3}
            position={[12.664, 11.34, 26.654]}
            rotation={[1.017, 0.596, 1.007]}
            scale={[0.2, 0.77, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_99.geometry}
            material={materials.material_3}
            position={[14.786, 11.502, 26.052]}
            rotation={[-1.224, 0.367, -0.516]}
            scale={[0.2, 0.338, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_100.geometry}
            material={materials.material_5}
            position={[15.1, 11.59, 25.453]}
            rotation={[1.44, 0.483, 0.547]}
            scale={[0.2, 0.344, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_101.geometry}
            material={materials.material_3}
            position={[12.045, 11.698, 27.677]}
            rotation={[0.81, 0.052, 0.122]}
            scale={[0.2, 0.666, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_102.geometry}
            material={materials.material_3}
            position={[16.966, 10.95, 28.346]}
            rotation={[1.498, -0.523, -0.561]}
            scale={[0.2, 0.727, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_103.geometry}
            material={materials.material_3}
            position={[16.866, 11.142, 27.475]}
            rotation={[-0.915, 0.449, -0.869]}
            scale={[0.2, 0.341, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_104.geometry}
            material={materials.material_5}
            position={[17.339, 11.232, 26.985]}
            rotation={[1.046, 0.601, 0.986]}
            scale={[0.2, 0.347, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_105.geometry}
            material={materials.material_3}
            position={[17.956, 10.849, 28.872]}
            rotation={[-0.155, 0.141, -1.477]}
            scale={[0.2, 0.665, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_106.geometry}
            material={materials.material_3}
            position={[16.909, 10.702, 29.634]}
            rotation={[1.379, 0.519, 0.623]}
            scale={[0.2, 0.774, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_107.geometry}
            material={materials.material_3}
            position={[18.714, 11.01, 28.062]}
            rotation={[-1.396, 0.12, -0.143]}
            scale={[0.2, 0.726, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_108.geometry}
            material={materials.material_3}
            position={[19.179, 10.802, 29.284]}
            rotation={[0.931, -0.593, -1.093]}
            scale={[0.2, 0.771, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_109.geometry}
            material={materials.material_3}
            position={[18.51, 11.161, 27.2]}
            rotation={[-0.492, -0.362, 1.259]}
            scale={[0.2, 0.345, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_110.geometry}
            material={materials.material_5}
            position={[17.89, 11.238, 26.892]}
            rotation={[0.515, -0.458, -1.449]}
            scale={[0.2, 0.351, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_111.geometry}
            material={materials.material_3}
            position={[20.122, 10.042, 29.81]}
            rotation={[0.019, -0.071, -2.626]}
            scale={[0.2, 0.763, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_112.geometry}
            material={materials.material_3}
            position={[21.1, 9.429, 30.329]}
            rotation={[0.805, -0.478, -1.039]}
            scale={[0.2, 0.788, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_113.geometry}
            material={materials.material_3}
            position={[21.807, 9.162, 30.938]}
            rotation={[1.457, -1.265, -1.376]}
            scale={[0.2, 0.351, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_114.geometry}
            material={materials.light_0}
            position={[21.989, 8.602, 31.123]}
            rotation={[-0.304, -0.046, 0.3]}
            scale={[0.2, 0.265, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_115.geometry}
            material={materials.material_3}
            position={[21.819, 9.805, 30.903]}
            rotation={[0.199, -0.034, -0.336]}
            scale={[0.2, 0.35, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_116.geometry}
            material={materials.light_0}
            position={[22.021, 10.372, 31.024]}
            rotation={[-0.384, -1.015, 2.468]}
            scale={[0.2, 0.265, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_117.geometry}
            material={materials.light_0}
            position={[16.565, 13.735, 25.112]}
            rotation={[-0.055, -0.022, 0.757]}
            scale={[0.2, 0.333, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_118.geometry}
            material={materials.material_7}
            position={[19.748, 17.019, 27.494]}
            rotation={[1.23, -0.496, -0.688]}
            scale={[0.2, 0.777, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_119.geometry}
            material={materials.material_7}
            position={[18.7, 16.613, 27.01]}
            rotation={[0.227, 0.389, 2.091]}
            scale={[0.2, 0.765, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_120.geometry}
            material={materials.material_7}
            position={[19.666, 16.622, 26.288]}
            rotation={[-1.712, 0.76, -0.667]}
            scale={[0.2, 0.783, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_121.geometry}
            material={materials.material_7}
            position={[19.215, 17.775, 26.656]}
            rotation={[-0.257, -0.017, 0.13]}
            scale={[0.2, 0.767, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_122.geometry}
            material={materials.material_7}
            position={[20.21, 17.236, 28.358]}
            rotation={[0.743, -0.034, -0.088]}
            scale={[0.2, 0.325, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_123.geometry}
            material={materials.light_0}
            position={[20.264, 17.683, 28.771]}
            rotation={[-2.376, -0.223, 0.09]}
            scale={[0.2, 0.286, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_124.geometry}
            material={materials.material_7}
            position={[20.406, 16.278, 28.064]}
            rotation={[-0.121, 0.622, -2.77]}
            scale={[0.2, 0.758, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_125.geometry}
            material={materials.material_7}
            position={[20.363, 15.262, 27.337]}
            rotation={[-1.788, -0.506, 0.41]}
            scale={[0.2, 0.765, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_126.geometry}
            material={materials.material_7}
            position={[19.387, 14.889, 26.973]}
            rotation={[0.42, 0.396, 1.51]}
            scale={[0.2, 0.771, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_127.geometry}
            material={materials.material_7}
            position={[20.057, 15.586, 26.205]}
            rotation={[-0.658, -0.017, 0.051]}
            scale={[0.2, 0.783, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_128.geometry}
            material={materials.material_7}
            position={[18.382, 15.498, 27.216]}
            rotation={[-0.063, -0.013, 0.406]}
            scale={[0.2, 0.75, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_129.geometry}
            material={materials.material_7}
            position={[20.702, 16.464, 25.532]}
            rotation={[-0.262, 0.173, -1.162]}
            scale={[0.2, 0.758, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_130.geometry}
            material={materials.material_7}
            position={[19.556, 16.043, 25.132]}
            rotation={[-1.263, -0.626, 0.833]}
            scale={[0.2, 0.769, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_131.geometry}
            material={materials.material_1}
            position={[19.395, 8.05, 25.149]}
            rotation={[-1.06, -0.257, 0.434]}
            scale={[0.2, 1.531, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_132.geometry}
            material={materials.material_1}
            position={[18.348, 8.798, 23.917]}
            rotation={[0.126, 0.077, 1.099]}
            scale={[0.2, 0.478, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_133.geometry}
            material={materials.material_2}
            position={[17.525, 9.209, 24.033]}
            rotation={[-0.128, 0.201, -2.007]}
            scale={[0.2, 0.45, 0.2]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_0.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_1.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_2.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_3.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_4.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_5.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_6.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_7.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_8.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_9.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_10.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_11.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_12.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_13.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_14.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_15.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_16.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_17.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_18.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_19.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_20.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_21.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_22.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_134.geometry}
            material={materials.light_1}
            position={[18.808, 11.729, 24.552]}
            rotation={[0.136, 0.151, 1.68]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_135.geometry}
            material={materials.light_1}
            position={[18.11, 11.638, 24.647]}
            rotation={[0.136, 0.151, 1.68]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_136.geometry}
            material={materials.light_1}
            position={[17.03, 11.963, 23.262]}
            rotation={[1.825, -0.191, -0.148]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_137.geometry}
            material={materials.light_1}
            position={[17.133, 11.766, 23.939]}
            rotation={[1.825, -0.191, -0.148]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_138.geometry}
            material={materials.light_1}
            position={[15.692, 11.611, 25.067]}
            rotation={[-0.195, 0.198, -1.585]}
            scale={[0.048, 0.242, 0.048]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_139.geometry}
            material={materials.light_1}
            position={[16.404, 11.573, 24.929]}
            rotation={[-0.195, 0.198, -1.585]}
            scale={[0.048, 0.242, 0.048]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_140.geometry}
            material={materials.light_1}
            position={[17.508, 11.332, 26.309]}
            rotation={[-1.423, -0.138, 0.16]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_141.geometry}
            material={materials.light_1}
            position={[17.396, 11.42, 25.612]}
            rotation={[-1.423, -0.138, 0.16]}
            scale={[0.047, 0.237, 0.047]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_142.geometry}
            material={materials.light_1}
            position={[17.071, 12.777, 25.036]}
            rotation={[-2.581, 1.162, -0.374]}
            scale={[0.033, 0.166, 0.033]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_143.geometry}
            material={materials.light_1}
            position={[17.143, 12.297, 24.931]}
            rotation={[-2.581, 1.162, -0.374]}
            scale={[0.033, 0.166, 0.033]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_144.geometry}
            material={materials.light_1}
            position={[17.154, 9.864, 24.234]}
            rotation={[0.307, -0.009, -0.06]}
            scale={[0.056, 0.28, 0.056]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.primitive_color_145.geometry}
            material={materials.light_1}
            position={[17.204, 10.662, 24.488]}
            rotation={[0.307, -0.009, -0.06]}
            scale={[0.056, 0.28, 0.056]}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_23.geometry}
            material={materials.material_0}
          />
          <mesh
            castShadow
            receiveShadow
            geometry={nodes.matsym_24.geometry}
            material={materials.material_0}
          />
        </group>
      </group>
    </group>
  )
}

useGLTF.preload('/glb/model_of_cytochrome_p450__oxygen__drug_complex.glb')
