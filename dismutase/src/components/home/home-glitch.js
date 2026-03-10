import dynamic from "next/dynamic";
import Link from "next/link";
import React from "react";
import { RotateTextAnimation } from "../AnimationText";

// Dynamischer Import des 3D-Modells
const ModelScene = dynamic(() => import('../model/ModelScene'), {
  ssr: false,
});

export default function HomeGlitch({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* HOME */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section animated rollIn"
            : "cavani_tm_section animated hidden rollOut"
        }
        id="home_"
        style={{ position: 'relative' }}
      >
        {/* 3D Model als Hintergrund */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 0,
          pointerEvents: 'none' // Damit Klicks durchgehen
        }}>
          <ModelScene modelPath="/glb/conformational_transition_of_troponin.glb" />
        </div>

        {/* Content im Vordergrund */}
        <div className="cavani_tm_home" style={{ position: 'relative', zIndex: 1 }}>
          <div className="content">
            <h3 className="name">Dismutase</h3>
            <span className="line" />
            <h3 className="job">
              <RotateTextAnimation />
            </h3>
            <div className="cavani_tm_button transition_link">
              <Link href="#about">
                <a onClick={() => handleOnClick(1)}>Explore Framework</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* /HOME */}
    </>
  );
}
