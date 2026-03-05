import React from 'react'
import Image from 'next/image'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const circleProgressData = [
    { language: 'Electron Transfer', progress: 98 },
    { language: 'Enzyme Catalysis', progress: 95 },
    { language: 'Protein Folding', progress: 92 },
    { language: 'Disease Prediction', progress: 88 },
    { language: 'Mass Spectrometry', progress: 85 },
];

const progressBarData = [
    { bgcolor: "#f9d77e", completed: 99, title: 'Backaction \u03B4 ~ 10\u207B\u2074' },
    { bgcolor: "#f9d77e", completed: 100, title: 'Categorical Distance d_C = 1' },
    { bgcolor: "#f9d77e", completed: 95, title: 'Coherence \u27E8r\u27E9 Preservation' },
];

const keyResults = [
    {
        desc: "The categorical distance d_C = 1 for SOD1 Cu/Zn superoxide dismutase confirms that enzyme catalysis follows an exact mathematical trajectory through partition space.",
        img: "img/research/catalysis.png",
        info1: "Catalysis Validation",
        info2: "SOD1 d_C = 1"
    },
    {
        desc: "Protein folding complexity reduces from exponential to O(log\u2083 N) through ternary partitioning of conformational space, resolving Levinthal's paradox.",
        img: "img/research/folding.png",
        info1: "Folding Complexity",
        info2: "O(log\u2083 N)"
    },
    {
        desc: "ALS disease prediction through coherence loss metrics provides a quantitative, measurement-backaction-derived framework for neurodegeneration tracking.",
        img: "img/research/disease.png",
        info1: "Disease Prediction",
        info2: "ALS Coherence Loss"
    },
]

export default function AboutGlitch({ ActiveIndex }) {
    return (
        <>
            {/* <!-- ABOUT --> */}
            <div className={ActiveIndex === 1 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"} id="about_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>About Dismutase</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p><strong>Dismutase</strong> develops a categorical mechanics framework for bounded quantum systems, providing exact mathematical descriptions of biological processes from electron transfer to disease prediction.</p>
                                    <p>The framework derives from a single axiom — bounded phase space — producing partition coordinates (n, &#8467;, m, s) with capacity C(n) = 2n&sup2; that describe biological systems as reflexive measurement devices operating through ternary encoding.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Researcher:</span><span className="second">Kundai Farai Sachikonye</span></li>
                                        <li><span className="first">Focus:</span><span className="second">Categorical Mechanics</span></li>
                                        <li><span className="first">Framework:</span><span className="second">Bounded Quantum Systems</span></li>
                                        <li><span className="first">Domains:</span><span className="second">5 Validated</span></li>
                                        <li><span className="first">Mail:</span><span className="second"><a href="mailto:kundai.sachikonye@bitspark.com">kundai.sachikonye@bitspark.com</a></span></li>
                                        <li><span className="first">GitHub:</span><span className="second"><a href="https://github.com/fullscreen-triangle/levinthal" target="_blank" rel="noopener noreferrer">levinthal</a></span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Research Capabilities</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Categorical Partition Theory</li>
                                            <li>Reflexive Ternary Encoding</li>
                                            <li>Measurement Backaction Analysis</li>
                                            <li>Phase-Lock Dynamics</li>
                                            <li>S-Entropy Quantification</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Application Domains</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Electron Transfer Kinetics</li>
                                            <li>Enzyme Catalysis Mechanisms</li>
                                            <li>Protein Folding Trajectories</li>
                                            <li>Neurodegenerative Disease</li>
                                            <li>Database-Free Proteomics</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Key Metrics</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {progressBarData.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Domain Coverage</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {circleProgressData.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            stroke='#f9d77e'
                                                            Language={item.language}
                                                            className={"list_inner"}
                                                        />
                                                        <div className="title"><span>{item.language}</span></div>
                                                    </div>
                                                </div>
                                            ))}

                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="resume">
                            <div className="wrapper">
                                <div className="education">
                                    <div className="cavani_tm_title">
                                        <span>Key Publications</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Categorical Mechanics of Enzyme Catalysis</h3>
                                                            <span>SOD1 validation, d_C = 1</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Protein Folding Trajectory</h3>
                                                            <span>O(log&#8323; N) complexity resolution</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Biological Partition Landscape</h3>
                                                            <span>Unified partition theory</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Validation Milestones</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Azurin</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Electron Transfer</h3>
                                                            <span>Cu(I) to Cu(II), backaction &#948; ~ 10&#8315;&#8308;</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>SOD1</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Enzyme Catalysis</h3>
                                                            <span>Categorical distance d_C = 1 (exact)</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>ALS</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Disease Prediction</h3>
                                                            <span>Coherence loss metric for neurodegeneration</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="testimonials">
                            <div className="cavani_tm_title">
                                <span>Key Results</span>
                            </div>
                            <div className="list">
                                <ul className="">
                                    <li>
                                        <Swiper
                                            slidesPerView={1}
                                            spaceBetween={30}
                                            loop={true}
                                            className="custom-class"
                                            breakpoints={{
                                                768: {
                                                    slidesPerView: 2,
                                                }
                                            }}
                                        >
                                            {keyResults.map((item, i) => (
                                                <SwiperSlide key={i}>
                                                    <div className="list_inner">
                                                        <div className="text">
                                                            <i className="icon-quote-left" />
                                                            <p>{item.desc}</p>
                                                        </div>
                                                        <div className="details">
                                                            <div className="image">
                                                                <div className="main" data-img-url={item.img} />
                                                            </div>
                                                            <div className="info">
                                                                <h3>{item.info1}</h3>
                                                                <span>{item.info2}</span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                </SwiperSlide>
                                            ))}
                                        </Swiper>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- ABOUT --> */}
        </>
    )
}
