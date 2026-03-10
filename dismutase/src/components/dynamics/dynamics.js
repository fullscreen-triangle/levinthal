import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const coherenceMetrics = [
    { language: 'Closed Loop', progress: 34 },
    { language: 'Open Loop', progress: 25 },
    { language: 'Re-closed', progress: 32 },
];

const dynamicsMetrics = [
    { bgcolor: "#f9d77e", completed: 100, title: 'Phase-lock final r = 1.000' },
    { bgcolor: "#f9d77e", completed: 94, title: '94% validation pass rate (34/36)' },
    { bgcolor: "#f9d77e", completed: 100, title: 'S-entropy conservation = 1.000 \u00B1 0.000' },
];

const keyInsights = [
    {
        desc: "Protein structure is maintained by Kuramoto phase-locking of hydrogen bond oscillators. The order parameter \u27E8r\u27E9 quantifies collective synchronization \u2014 when oscillators lock in phase, the protein achieves its native fold.",
        info1: "Phase-Lock Mechanism",
        info2: "Kuramoto Synchronization"
    },
    {
        desc: "The SOD1 electrostatic loop (residues 121\u2013142) gates active site access through topology changes. Closed state r = 0.34, open state r = 0.25 \u2014 a \u0394r = 0.09 transition through 4 intermediates.",
        info1: "Conformational Gating",
        info2: "SOD1 Loop Dynamics"
    },
    {
        desc: "Loop topology is temperature-independent: the same conformational states appear at T = 200\u2013400 K. Temperature affects kinetic rates but not the pathway itself \u2014 topology is invariant under thermal perturbation.",
        info1: "Kinetic Independence",
        info2: "Temperature Invariant"
    },
]

export default function Dynamics({ ActiveIndex }) {
    return (
        <>
            <div className={ActiveIndex === 5 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"} id="dynamics_">
                <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>Protein Dynamics</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p>Protein dynamics emerges from <strong>Kuramoto phase-locking</strong> of hydrogen bond oscillators coupled through an exponentially decaying network. The governing equation is:</p>
                                    <p style={{fontFamily: 'monospace', color: '#f9d77e', fontSize: '14px', padding: '10px 0'}}>
                                        d&#966;<sub>i</sub>/dt = &#969;<sub>i</sub> + &#931;<sub>j</sub> K<sub>ij</sub> sin(&#966;<sub>j</sub> &minus; &#966;<sub>i</sub>)
                                    </p>
                                    <p>where K<sub>ij</sub> = K<sub>0</sub> exp(&minus;r<sub>ij</sub>/r<sub>0</sub>) couples spatially proximal hydrogen bonds. This produces the <strong>order parameter</strong> &#10216;r&#10217; that quantifies structural coherence.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">System:</span><span className="second">SOD1 Cu/Zn Superoxide Dismutase</span></li>
                                        <li><span className="first">H-bonds:</span><span className="second">165 coupled oscillators</span></li>
                                        <li><span className="first">Coupling:</span><span className="second">K&#8320; exp(&minus;r/r&#8320;), r&#8320; = 5.0 &#8491;</span></li>
                                        <li><span className="first">Native r:</span><span className="second">&#10216;r&#10217; = 1.000</span></li>
                                        <li><span className="first">Loop gating:</span><span className="second">&#916;r = 0.09, d<sub>C</sub> = 4</span></li>
                                        <li><span className="first">Temp range:</span><span className="second">200&ndash;400 K (invariant topology)</span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Phase-Lock Network</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>165 hydrogen bond oscillators in SOD1</li>
                                            <li>Exponential coupling decay K&#8320;e<sup>&minus;r/r&#8320;</sup></li>
                                            <li>Random initial phases &#8594; synchronized native state</li>
                                            <li>Order parameter &#10216;r&#10217; = |N<sup>&minus;1</sup> &#931; e<sup>i&#966;j</sup>|</li>
                                            <li>Native state = arg max &#10216;r&#10217; (Equation VI)</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Conformational Transitions</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Loop gating: closed (0.34) &#8594; open (0.25)</li>
                                            <li>4 intermediate states (d<sub>C</sub> = 4)</li>
                                            <li>Each step = one H-bond disruption</li>
                                            <li>Re-closure: open &#8594; r = 0.32</li>
                                            <li>Temperature-independent topology</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Validation Metrics</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {dynamicsMetrics.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Loop State Coherence &#10216;r&#10217;</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {coherenceMetrics.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`0.${item.progress}`}
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
                                                            <span>Eq. V</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Phase-Lock Dynamics</h3>
                                                            <span>d&#966;<sub>i</sub>/dt = &#969;<sub>i</sub> + &#931; K<sub>ij</sub> sin(&#966;<sub>j</sub> &minus; &#966;<sub>i</sub>)</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Eq. VI</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Native Structure</h3>
                                                            <span>Native = arg max &#10216;r&#10217;</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>Eq. IV</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Gradient Flow</h3>
                                                            <span>dx/dt = &minus;&#947; &#8711;M (partition depth descent)</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Temperature Independence</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>200 K</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>r = 0.499</h3>
                                                            <span>Same topology, slower kinetics</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>300 K</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>r = 0.353</h3>
                                                            <span>Physiological temperature</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>400 K</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>r = 0.189</h3>
                                                            <span>Same topology, faster kinetics</span>
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
