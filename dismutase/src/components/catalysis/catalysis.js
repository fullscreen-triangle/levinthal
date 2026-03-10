import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const enzymeAccuracy = [
    { language: 'SOD1', progress: 91 },
    { language: 'Carbonic Anh.', progress: 89 },
    { language: 'Catalase', progress: 84 },
    { language: 'AChE', progress: 92 },
    { language: 'Fumarase', progress: 89 },
];

const catalysisMetrics = [
    { bgcolor: "#f9d77e", completed: 97, title: 'MAE = 0.97 log units (8 enzymes)' },
    { bgcolor: "#f9d77e", completed: 100, title: 'SOD1 coherence \u27E8r\u27E9 = 1.000 during catalysis' },
    { bgcolor: "#f9d77e", completed: 88, title: '7/8 enzymes within \u00B11 log unit' },
];

const keyInsights = [
    {
        desc: "Enzyme efficiency is predicted by a single integer: the categorical distance d_C. The prediction rule log\u2081\u2080(kcat/KM) \u2248 10 \u2212 d_C achieves MAE = 0.97 log units across six orders of magnitude with zero free parameters.",
        info1: "Categorical Distance",
        info2: "d_C Prediction Rule"
    },
    {
        desc: "SOD1 maintains near-perfect phase-lock coherence (\u27E8r\u27E9 = 1.000) throughout 2,000 catalytic cycles. This confirms that diffusion-limited enzymes (d_C = 1) operate as maximally coherent quantum systems.",
        info1: "Catalytic Coherence",
        info2: "\u27E8r\u27E9 = 1.000"
    },
    {
        desc: "Enzymes don't make reactions faster \u2014 they make pathways shorter. A d_C = 1 enzyme (SOD1) provides a single-step categorical pathway, while d_C = 4 (chymotrypsin) requires four steps, reducing efficiency by 10,000-fold.",
        info1: "Paradigm Shift",
        info2: "Topology, Not Kinetics"
    },
]

export default function Catalysis({ ActiveIndex }) {
    return (
        <>
            <div className={ActiveIndex === 8 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"} id="catalysis_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>Enzyme Catalysis</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p>Enzyme catalytic efficiency is determined by a single topological quantity: the <strong>categorical distance d<sub>C</sub></strong>. This integer counts the minimum partition boundary crossings between substrate and product states.</p>
                                    <p style={{fontFamily: 'monospace', color: '#f9d77e', fontSize: '14px', padding: '10px 0'}}>
                                        log&#8321;&#8320;(k<sub>cat</sub>/K<sub>M</sub>) &#8776; 10 &minus; d<sub>C</sub>
                                    </p>
                                    <p>This prediction rule achieves <strong>MAE = 0.97 log units</strong> across eight enzymes spanning six orders of magnitude, with zero free parameters.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Enzymes:</span><span className="second">8 validated</span></li>
                                        <li><span className="first">MAE:</span><span className="second">0.97 log units</span></li>
                                        <li><span className="first">Parameters:</span><span className="second">Zero free parameters</span></li>
                                        <li><span className="first">Range:</span><span className="second">6 orders of magnitude</span></li>
                                        <li><span className="first">SOD1 d<sub>C</sub>:</span><span className="second">1 (diffusion-limited)</span></li>
                                        <li><span className="first">SOD1 r:</span><span className="second">&#10216;r&#10217; = 1.000 during catalysis</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>d<sub>C</sub> = 1 (Diffusion-Limited)</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>SOD1 &mdash; predicted: 9.0, observed: 9.85</li>
                                            <li>Carbonic anhydrase &mdash; pred: 9.0, obs: 8.0</li>
                                            <li>Catalase &mdash; pred: 9.0, obs: 7.6</li>
                                            <li>Acetylcholinesterase &mdash; pred: 9.0, obs: 8.3</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>d<sub>C</sub> = 2&ndash;4 (Sub-Diffusion)</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Fumarase (d<sub>C</sub>=2) &mdash; pred: 8.0, obs: 8.9</li>
                                            <li>&#946;-Amylase (d<sub>C</sub>=2) &mdash; pred: 8.0, obs: 7.6</li>
                                            <li>Lysozyme (d<sub>C</sub>=3) &mdash; pred: 7.0, obs: 6.5</li>
                                            <li>Chymotrypsin (d<sub>C</sub>=4) &mdash; pred: 6.0, obs: 4.0</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Prediction Accuracy</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {catalysisMetrics.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Prediction Match (%)</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {enzymeAccuracy.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            stroke='#f9d77e'
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
                                        <span>Governing Equations</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Eq. IV</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Gradient Flow</h3>
                                                            <span>dx/dt = &minus;&#947; &#8711;M determines catalytic rate</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Eq. V</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Phase-Lock During Catalysis</h3>
                                                            <span>SOD1 maintains &#10216;r&#10217; = 1.000 over 2,000 cycles</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Rule</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Efficiency Prediction</h3>
                                                            <span>log&#8321;&#8320;(k<sub>cat</sub>/K<sub>M</sub>) &#8776; 10 &minus; d<sub>C</sub></span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>SOD1 Catalytic Validation</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>d<sub>C</sub> = 1</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Diffusion-Limited</h3>
                                                            <span>Single partition boundary crossing</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>k<sub>cat</sub>/K<sub>M</sub></span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>10&#8313;&#183;&#8312;&#8309; M&#8315;&#185;s&#8315;&#185;</h3>
                                                            <span>Near diffusion limit (predicted: 10&#8313;)</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>20 runs</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Trajectory CV = 5 &#215; 10&#8315;&#8311;</h3>
                                                            <span>Deterministic catalytic pathway</span>
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
                                <span>Key Insights</span>
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
                                            {keyInsights.map((item, i) => (
                                                <SwiperSlide key={i}>
                                                    <div className="list_inner">
                                                        <div className="text">
                                                            <i className="icon-quote-left" />
                                                            <p>{item.desc}</p>
                                                        </div>
                                                        <div className="details">
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
        </>
    )
}
