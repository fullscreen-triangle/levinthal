import { useState, useEffect, useRef } from 'react'
import Isotope from 'isotope-layout'
import { dataImage } from '../../plugin/plugin'
import { LightgalleryProvider } from 'react-lightgallery';
import { PhotoItem } from '../../plugin/lightGalleryOptions'
import Modal from 'react-modal';

export default function PortfolioDefault({ ActiveIndex, Animation }) {

    const [isOpen4, setIsOpen4] = useState(false);
    const [modalContent, setModalContent] = useState(null);

    function toggleModalFour(item) {
        if (item && item.title) {
            setModalContent(item);
        }
        setIsOpen4(!isOpen4);
    }

    const portfolioItems = [
        {
            filterClass: "electron",
            img: "img/research/electron.png",
            title: "Azurin Cu Transfer",
            category: "Electron Transfer",
            desc1: "Zero-backaction measurement of electron dynamics in azurin copper protein, tracking Cu(I) to Cu(II) transfer with measurement backaction delta ~ 10^-4.",
            desc2: "The protein scaffold acts as a bounded quantum measurement apparatus, enabling categorical partition analysis of the electron transfer pathway at 73pm spatial resolution."
        },
        {
            filterClass: "catalysis",
            img: "img/research/catalysis.png",
            title: "SOD1 Catalysis",
            category: "Enzyme Catalysis",
            desc1: "Cu/Zn superoxide dismutase achieves categorical distance d_C = 1, confirming that its catalytic cycle follows an exact mathematical trajectory through partition space.",
            desc2: "This provides the first rigorous explanation for catalytic perfection in enzymes operating at the diffusion limit."
        },
        {
            filterClass: "folding",
            img: "img/research/folding.png",
            title: "Protein Folding",
            category: "Protein Folding",
            desc1: "Levinthal's paradox resolved through ternary partitioning of conformational space, reducing folding complexity from exponential O(3^N) to logarithmic O(log_3 N).",
            desc2: "Validated against experimental folding rates across protein families, demonstrating logarithmic scaling of folding time with chain length."
        },
        {
            filterClass: "disease",
            img: "img/research/disease.png",
            title: "ALS Prediction",
            category: "Disease",
            desc1: "Neurodegenerative disease tracked through coherence loss in the categorical mechanics framework, providing quantitative prediction of mutation pathogenicity.",
            desc2: "The coherence parameter r quantifies degradation of categorical structure in ALS-associated SOD1 mutants."
        },
        {
            filterClass: "spectrometry",
            img: "img/research/validation.png",
            title: "Peptide Identification",
            category: "Mass Spectrometry",
            desc1: "Database-free peptide identification through categorical partition analysis of mass spectrometry fragmentation patterns.",
            desc2: "Achieves 88.7% PTM accuracy with CV < 2.1% cross-platform reproducibility and 89.3% zero-shot transfer."
        },
        {
            filterClass: "electron",
            img: "img/research/trajectory.png",
            title: "Electron Trajectory",
            category: "Electron Transfer",
            desc1: "Complete electron trajectory from 0.02 Angstrom to 1.99 Angstrom over 160fs, tracked through 17 ternary trisection iterations.",
            desc2: "Ternary string [1,1,1,1,1,1,1,1,1,2,1,1,2,1,2,2,1] encodes the full transfer pathway with mean backaction 1.73 x 10^-6."
        },
    ];

    const imagesCollection = [
        ["img/research/phaselock.png"],
        ["img/research/sentropy.png"],
        ["img/research/conformational.png"],
    ];

    // init one ref to store the future isotope object
    const isotope = useRef()
    // store the filter keyword in a state
    const [filterKey, setFilterKey] = useState('*')

    // initialize an Isotope object with configs
    useEffect(() => {
        setTimeout(() => {
            isotope.current = new Isotope(".filter-container", {
                itemSelector: ".filter-item",
                   layoutMode: "fitRows",
            });
        }, 500);
        return () => isotope.current.destroy();
    }, []);

    // handling filter key change
    useEffect(() => {
        if (isotope.current) {
            filterKey === '*'
                ? isotope.current.arrange({ filter: '*' })
                : isotope.current.arrange({ filter: `.${filterKey}` })
        }
    }, [filterKey])

    const handleFilterKeyChange = key => () => setFilterKey(key)

    return (
        <>
            {/* <!-- PORTFOLIO --> */}

            <div className={ActiveIndex === 2 ? `cavani_tm_section active animated ${Animation ? Animation: "fadeInUp"}` : "cavani_tm_section hidden animated"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Research Results</span>
                        </div>

                        <div className="portfolio_filter">
                            <ul>
                                <li><a href='#' onClick={handleFilterKeyChange('*')} className="current">All</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('electron')}>Electron Transfer</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('catalysis')}>Catalysis</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('folding')}>Folding</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('disease')}>Disease</a></li>
                                <li><a href='#' onClick={handleFilterKeyChange('spectrometry')}>Mass Spec</a></li>
                            </ul>
                        </div>
                        <div className="portfolio_list">

                            <div className="filter-container">
                                {portfolioItems.map((item, i) => (
                                    <div key={i} className={`filter-item ${item.filterClass}`}>
                                        <div className="list_inner">
                                            <div className="image">
                                                <img src="img/thumbs/1-1.jpg" alt="" />
                                                <div className="main" data-img-url={item.img} onClick={() => toggleModalFour(item)}></div>
                                                <div className="details">
                                                    <h3>{item.title}</h3>
                                                    <span>{item.category}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}

                                {/* Photo Gallery */}
                                <LightgalleryProvider>
                                    {imagesCollection.map((p, idx) => (
                                        <PhotoItem key={idx} image={p[0]} thumb={p[1]} />
                                    ))}
                                </LightgalleryProvider>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /PORTFOLIO --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen4}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour}>
                                <a href="#">
                                    <i className="icon-cancel" />
                                </a>
                            </div>
                            <div className="description_wrap">
                                <div className="popup_details">
                                    <div className="top_image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="portfolio_main_title">
                                        <h3>{modalContent.title}</h3>
                                        <span>{modalContent.category}</span>
                                    </div>
                                    <div className="main_details">
                                        <div className="textbox">
                                            <p>{modalContent.desc1}</p>
                                            <p>{modalContent.desc2}</p>
                                        </div>
                                        <div className="detailbox">
                                            <ul>
                                                <li>
                                                    <span className="first">Domain</span>
                                                    <span>{modalContent.category}</span>
                                                </li>
                                                <li>
                                                    <span className="first">Year</span>
                                                    <span>2025</span>
                                                </li>
                                                <li>
                                                    <span className="first">Repository</span>
                                                    <span><a href="https://github.com/fullscreen-triangle/levinthal" target="_blank" rel="noopener noreferrer">levinthal</a></span>
                                                </li>
                                            </ul>
                                        </div>
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
