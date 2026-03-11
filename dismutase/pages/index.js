import dynamic from "next/dynamic";
import React, { useState, useEffect } from "react";
import Landing from "../src/components/landing/landing";
import ContactDefault from "../src/components/contact/contact";
import News from "../src/components/News";
import Header from "../src/layout/header";
import Layout from "../src/layout/layout";
import LeftRightBar from "../src/layout/left-right-bar";
import Mobilemenu from "../src/layout/mobilemenu";
import Modalbox from "../src/layout/modalbox";
import TopBar from "../src/layout/top-bar";

const GLBViewer = dynamic(
  () => import("../src/components/model/GLBViewer"),
  { ssr: false }
);

const Dynamics = dynamic(
  () => import("../src/components/dynamics/dynamics"),
  { ssr: false }
);
const Folding = dynamic(
  () => import("../src/components/folding/folding"),
  { ssr: false }
);
const Catalysis = dynamic(
  () => import("../src/components/catalysis/catalysis"),
  { ssr: false }
);

export default function Home() {
  const [ActiveIndex, setActiveIndex] = useState(0);
  const handleOnClick = (index) => {
    setActiveIndex(index);
  };

  useEffect(() => {
    document.body.classList.add('dark');
    return () => document.body.classList.remove('dark');
  }, []);

  const [isToggled, setToggled] = useState(false);
  const toggleTrueFalse = () => setToggled(!isToggled);

  return (
    <>
      <Layout>
        <div
          className="cavani_tm_all_wrap"
          data-magic-cursor="show"
          data-enter="rollIn"
          data-exit="rollOut"
        >
          <Modalbox />
          <Header handleOnClick={handleOnClick} ActiveIndex={ActiveIndex} />
          <LeftRightBar />
          <TopBar toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} />
          <Mobilemenu toggleTrueFalse={toggleTrueFalse} isToggled={isToggled} handleOnClick={handleOnClick} />

          {/* MAINPART */}
          <div className="cavani_tm_mainpart">
              <GLBViewer className="author_image" />
            <div className="main_content">
              <Landing ActiveIndex={ActiveIndex} handleOnClick={handleOnClick} />
              <Dynamics ActiveIndex={ActiveIndex} />
              <Folding ActiveIndex={ActiveIndex} />
              <Catalysis ActiveIndex={ActiveIndex} />
              <News animation={"rollIn"} ActiveIndex={ActiveIndex} />
              <ContactDefault ActiveIndex={ActiveIndex} />
            </div>
          </div>
          {/* /MAINPART */}

        </div>
      </Layout>
    </>
  );
}
