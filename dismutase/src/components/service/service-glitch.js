import React, { useState } from 'react'
import { dataImage } from '../../plugin/plugin'
import Modal from 'react-modal';
import { SVG_Custom1, SVG_Custom2, SVG_Custom3, SVG_Custom4, SVG_Custom5 } from '../../plugin/svg';
export default function Service({ ActiveIndex }) {

    const [isOpen7, setIsOpen7] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen7(!isOpen7);
    }
    const service = [
        {
            img: "img/research/electron.png",
            svg: <SVG_Custom1 />,
            text: "Azurin Cu(I) to Cu(II) electron transfer tracked through categorical partition mechanics with measurement backaction \u03B4 ~ 10\u207B\u2074.",
            title: "Electron Transfer",
            text1: "The azurin copper protein serves as the primary validation system for the categorical mechanics framework. Electron transfer between Cu(I) and Cu(II) states is described as a bounded quantum measurement where the protein scaffold acts as the measurement apparatus.",
            text2: "Key result: The measurement backaction parameter \u03B4 ~ 10\u207B\u2074 demonstrates that the protein environment creates a nearly ideal bounded quantum system, where electron tunneling follows exact categorical trajectories through partition space.",
            text3: "This validates the core prediction that biological electron transfer is not merely quantum tunneling through a static barrier, but a categorically partitioned process where the protein dynamically selects the transfer pathway through 17 ternary trisection iterations."
        },
        {
            img: "img/research/catalysis.png",
            svg: <SVG_Custom2 />,
            text: "SOD1 Cu/Zn superoxide dismutase achieves categorical distance d_C = 1, confirming exact catalytic trajectory.",
            title: "Enzyme Catalysis",
            text1: "Superoxide dismutase (SOD1) catalyzes the disproportionation of superoxide radicals with near diffusion-limited efficiency. The categorical mechanics framework reveals why: the enzyme achieves d_C = 1, meaning its catalytic cycle follows an exact categorical trajectory.",
            text2: "The partition analysis shows that SOD1's active site creates a bounded quantum system where substrate binding, electron transfer, and product release form a single coherent categorical operation.",
            text3: "This result provides the first mathematical explanation for why certain enzymes achieve catalytic perfection \u2014 they operate at categorical distance unity, the minimal possible distance in partition space."
        },
        {
            img: "img/research/folding.png",
            svg: <SVG_Custom3 />,
            text: "Protein folding complexity reduced from exponential to O(log\u2083 N) through ternary partitioning, resolving Levinthal's paradox.",
            title: "Protein Folding",
            text1: "Levinthal's paradox asks how proteins fold to their native state in microseconds when random search through conformational space would take longer than the age of the universe. The categorical mechanics framework resolves this through ternary partitioning.",
            text2: "Each amino acid residue contributes to a reflexive ternary partition of conformational space, reducing the search complexity from O(3^N) to O(log\u2083 N). The protein folds by successively partitioning its conformational space, not by searching through it.",
            text3: "This O(log\u2083 N) scaling has been validated against experimental folding rates across protein families, demonstrating that folding time scales logarithmically with chain length rather than exponentially."
        },
        {
            img: "img/research/disease.png",
            svg: <SVG_Custom4 />,
            text: "ALS neurodegeneration tracked through coherence loss in the categorical mechanics framework, enabling quantitative disease prediction.",
            title: "Disease Prediction",
            text1: "Neurodegenerative diseases like ALS involve the progressive loss of protein function. In the categorical mechanics framework, this corresponds to coherence loss \u2014 the degradation of exact categorical trajectories into approximate ones.",
            text2: "The coherence parameter \u27E8r\u27E9 quantifies the degree to which a protein system maintains its categorical structure. In healthy SOD1, \u27E8r\u27E9 is near unity. In ALS-associated mutants, \u27E8r\u27E9 decreases systematically, providing a quantitative predictor of disease severity.",
            text3: "This framework enables prediction of which mutations will cause disease and estimation of disease progression rates from first principles, without requiring empirical fitting to clinical data."
        },
        {
            img: "img/research/validation.png",
            svg: <SVG_Custom5 />,
            text: "Database-free peptide identification through categorical partition analysis of mass spectrometry fragmentation patterns.",
            title: "Mass Spectrometry",
            text1: "Traditional proteomics relies on matching observed mass spectra against databases of known peptide sequences. The categorical mechanics framework enables database-free identification by analyzing fragmentation patterns as categorical partitions.",
            text2: "Each peptide bond cleavage creates a partition of the molecular space, and the resulting fragment ions form a categorical trajectory. By analyzing the categorical structure of the fragmentation pattern, peptide sequences can be determined without reference to a database.",
            text3: "This approach achieves 88.7% PTM accuracy with CV < 2.1% cross-platform reproducibility and 89.3% zero-shot transfer, making it particularly valuable for identifying novel peptides and post-translational modifications missed by database-dependent methods."
        }
    ]
    return (
        <>
            {/* <!-- APPLICATIONS --> */}
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section hidden animated rollOut"} id="news_">
            <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Application Domains</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {service.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={toggleModalFour}>
                                            {item.svg}
                                            <h3 className="title" onClick={toggleModalFour}>{item.title}</h3>
                                            <p className="text">{item.text}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={() => setModalContent(item)} />
                                            <img className="popup_service_image" src={item.img} alt="" />
                                            <div className="service_hidden_details">
                                                <div className="service_popup_informations">
                                                    <div className="descriptions">
                                                        <p>{item.text1}</p>
                                                        <p>{item.text2}</p>
                                                        <p>{item.text3}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

            </div>
            {/* <!-- /APPLICATIONS --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen7}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" data-img-url="img/news/1.jpg" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="details">
                                        <span>{modalContent.tag}</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
                                        <p>{modalContent.text1}</p>
                                        <p>{modalContent.text2}</p>
                                        <p>{modalContent.text3}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    )
}
