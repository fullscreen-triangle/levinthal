import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const phaseData = [
    { language: 'Phase 1', progress: 68 },
    { language: 'Phase 2', progress: 100 },
    { language: 'Phase 3', progress: 100 },
    { language: 'Phase 4', progress: 100 },
    { language: 'Phase 5', progress: 100 },
];

const foldingMetrics = [
    { bgcolor: "#f9d77e", completed: 100, title: 'Final \u27E8r\u27E9 = 1.000 (native achieved)' },
    { bgcolor: "#f9d77e", completed: 100, title: 'Variance \u03C3\u00B2 = 1.08 \u00D7 10\u207B\u2079 (30 runs)' },
    { bgcolor: "#f9d77e", completed: 100, title: '5 categorical steps from \u2308log\u2083(165)\u2309' },
];

const keyInsights = [
    {
        desc: "Levinthal's paradox asks how proteins find the native state among 10\u2077\u00B3 conformations. The answer: they don't search. The categorical framework reduces the problem to \u2308log\u2083 N\u2309 = 5 deterministic steps through partition space.",
        info1: "Levinthal's Paradox Resolved",
        info2: "10\u2077\u00B3 \u2192 5 steps"
    },
    {
        desc: "Folding is not a funnel \u2014 it is a highway. Across 30 independent trajectories, the variance is 1.08 \u00D7 10\u207B\u2079. Every molecule follows the same path through categorical space despite thermal fluctuations.",
        info1: "Deterministic Trajectories",
        info2: "\u03C3\u00B2 = 1.08 \u00D7 10\u207B\u2079"
    },
    {
        desc: "The protein does not 'search' for the native state. Instead, the native state (maximum \u27E8r\u27E9) derives the folding pathway backward through selection rules. The pathway is a proof, not a search.",
        info1: "Epistemological Inversion",
        info2: "Backward Derivation"
    },
]

export default function Folding({ ActiveIndex }) {
    return (
        <>
            <div className={ActiveIndex === 6 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"} id="folding_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>Protein Folding</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p><strong>Levinthal&apos;s paradox</strong> states that a protein with 165 hydrogen bonds has ~10<sup>73</sup> possible conformations, yet folds in milliseconds. The categorical framework resolves this: folding proceeds through exactly <strong>&#8968;log&#8323; N&#8969; = 5</strong> deterministic steps.</p>
                                    <p>Each step corresponds to a ternary trisection of conformational space, reducing complexity from exponential to logarithmic. The order parameter &#10216;r&#10217; rises monotonically through five distinct phases from random (r &#8776; 0.11) to native (r = 1.000).</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Protein:</span><span className="second">SOD1 (153 residues, 165 H-bonds)</span></li>
                                        <li><span className="first">Conformations:</span><span className="second">10&#8311;&#179; possible states</span></li>
                                        <li><span className="first">Steps:</span><span className="second">&#8968;log&#8323;(165)&#8969; = 5</span></li>
                                        <li><span className="first">Final r:</span><span className="second">&#10216;r&#10217; = 1.000</span></li>
                                        <li><span className="first">Variance:</span><span className="second">&#963;&#178; = 1.08 &#215; 10&#8315;&#8313;</span></li>
                                        <li><span className="first">Trajectories:</span><span className="second">30 independent runs</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Five Folding Phases</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Phase 1: Hydrophobic collapse (r: 0 &#8594; 0.68)</li>
                                            <li>Phase 2: &#946;-sheet nucleation (r: 0.68 &#8594; 1.00)</li>
                                            <li>Phase 3: &#946;-barrel assembly (r: 1.00)</li>
                                            <li>Phase 4: Metal binding Cu/Zn (r: 1.00)</li>
                                            <li>Phase 5: Loop ordering (r: 1.00)</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Framework Predictions</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Exact step count from &#8968;log&#8323; N<sub>HB</sub>&#8969;</li>
                                            <li>Monotonic &#10216;r&#10217; increase guaranteed</li>
                                            <li>Deterministic: &#963;&#178; &lt; 10&#8315;&#8309;</li>
                                            <li>Native state = arg max &#10216;r&#10217;</li>
                                            <li>72 orders of magnitude reduction</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Folding Metrics</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {foldingMetrics.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Phase Completion &#10216;r&#10217;</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {phaseData.map((item, idx) => (
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
                                        <span>Folding Trajectory</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>0&ndash;5 ms</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Hydrophobic Collapse</h3>
                                                            <span>Chain compaction, r: 0 &#8594; 0.68</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>5&ndash;20 ms</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>&#946;-Sheet Nucleation</h3>
                                                            <span>Secondary structure, r: 0.68 &#8594; 1.00</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>20&ndash;100 ms</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Assembly &amp; Metal Binding</h3>
                                                            <span>&#946;-barrel + Cu/Zn coordination, r = 1.00</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Comparison with Existing Theories</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Brute force</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>10&#8311;&#179; Conformations</h3>
                                                            <span>Exhaustive search: age of universe</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Funnel</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Energy Landscape Theory</h3>
                                                            <span>Descriptive, not predictive</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Categorical</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>5 Deterministic Steps</h3>
                                                            <span>Derived from bounded phase space axiom</span>
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
