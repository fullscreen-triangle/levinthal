import { Fragment, useEffect, useState } from "react";
import Modal from "react-modal";
import { CloseButton } from "../plugin/svg";
const News = ({ ActiveIndex, animation }) => {
  const [isOpen4, setIsOpen4] = useState(false);
  const [modalContent, setModalContent] = useState({});

  useEffect(() => {
    var lists = document.querySelectorAll(".news_list > ul > li");
    let box = document.querySelector(".cavani_fn_moving_box");
    if (!box) {
      let body = document.querySelector("body");
      let div = document.createElement("div");
      div.classList.add("cavani_fn_moving_box");
      body.appendChild(div);
    }

    lists.forEach((list) => {
      list.addEventListener("mouseenter", (event) => {
        box.classList.add("opened");
        var imgURL = list.getAttribute("data-img");
        box.style.backgroundImage = `url(${imgURL})`;
        box.style.top = event.clientY - 50 + "px";
        if (imgURL === "") {
          box.classList.remove("opened");
          return false;
        }
      });
      list.addEventListener("mouseleave", () => {
        box.classList.remove("opened");
      });
    });
  }, []);

  function toggleModalFour(value) {
    setIsOpen4(!isOpen4);
    setModalContent(value);
  }
  const newsData = [
    {
      img: "img/research/catalysis.png",
      tag: "Catalysis",
      date: "2025",
      comments: "SOD1",
      title: "Categorical Mechanics of Enzyme Catalysis: SOD1 Validation",
      text1:
        "This paper establishes the categorical mechanics framework for enzyme catalysis, using Cu/Zn superoxide dismutase (SOD1) as the primary validation system.",
      text2:
        "The key finding is that SOD1 achieves categorical distance d_C = 1, meaning its catalytic cycle follows an exact mathematical trajectory through partition space. This provides the first rigorous explanation for catalytic perfection.",
      text3:
        "The framework introduces partition coordinates, phase-lock dynamics, and S-entropy as the fundamental variables governing enzyme function, replacing traditional transition state theory with a bounded quantum measurement description.",
    },
    {
      img: "img/research/folding.png",
      tag: "Folding",
      date: "2025",
      comments: "Levinthal",
      title: "Protein Folding Trajectory: Resolving Levinthal's Paradox",
      text1:
        "This paper applies the categorical mechanics framework to the protein folding problem, demonstrating that folding complexity scales as O(log\u2083 N) rather than exponentially with chain length.",
      text2:
        "The resolution of Levinthal's paradox comes from recognizing that proteins do not search conformational space \u2014 they partition it. Each residue contributes to a reflexive ternary encoding that successively narrows the conformational landscape.",
      text3:
        "Validation against experimental folding rates across multiple protein families confirms the logarithmic scaling prediction and provides folding trajectory descriptions at single-residue resolution.",
    },
    {
      img: "img/research/partition.png",
      tag: "Theory",
      date: "2025",
      comments: "Framework",
      title: "The Biological Partition Landscape",
      text1:
        "This foundational paper introduces the partition landscape concept, showing how biological systems from enzymes to cells organize their function through categorical partitioning of phase space.",
      text2:
        "The partition landscape unifies descriptions of metabolism, signaling, and gene regulation under a single mathematical framework based on bounded quantum measurement theory.",
      text3:
        "Key predictions include the universality of ternary encoding in biological systems and the relationship between partition depth and biological complexity.",
    },
    {
      img: "img/research/electron.png",
      tag: "Electron Transfer",
      date: "2025",
      comments: "Azurin",
      title: "Zero-Backaction Electron Trajectory in Azurin",
      text1:
        "Experimental validation of the categorical mechanics framework using azurin copper protein electron transfer between Cu(I) and Cu(II) states.",
      text2:
        "The measurement backaction parameter \u03B4 ~ 10\u207B\u2074 confirms that the protein environment creates a nearly ideal bounded quantum system for electron tunneling.",
      text3:
        "Results demonstrate that categorical partition coordinates provide a complete description of the electron transfer pathway, with predictions matching experimental Marcus theory parameters.",
    },
    {
      img: "img/research/sentropy.png",
      tag: "Proteomics",
      date: "2025",
      comments: "Mass Spec",
      title: "Database-Free Peptide Identification via Categorical Partitions",
      text1:
        "Application of categorical mechanics to mass spectrometry-based proteomics, enabling peptide identification without database matching.",
      text2:
        "Fragment ion patterns are analyzed as categorical partitions of molecular space, with the partition structure encoding sequence information directly.",
      text3:
        "This approach identifies novel peptides and modifications missed by traditional database search methods, achieving 88.7% PTM accuracy with CV < 2.1% cross-platform reproducibility.",
    },
    {
      img: "img/research/disease.png",
      tag: "Disease",
      date: "2025",
      comments: "ALS",
      title: "Coherence Loss as a Quantitative Disease Predictor",
      text1:
        "The categorical mechanics framework is applied to neurodegenerative disease, specifically ALS, through the lens of coherence loss in protein systems.",
      text2:
        "The coherence parameter \u27E8r\u27E9 tracks the degradation of categorical structure in disease-associated SOD1 mutants, providing a quantitative measure of disease severity.",
      text3:
        "Predictions of mutation pathogenicity and disease progression rates from first principles are validated against clinical data, demonstrating the framework's translational potential.",
    },
  ];
  return (
    <Fragment>
      <div
        className={
          ActiveIndex === 3
            ? `cavani_tm_section active animated ${animation ? animation : "fadeInUp"
            }`
            : "cavani_tm_section hidden animated"
        }
        id="news__"
      >
        <div className="section_inner">
          <div className="cavani_tm_news">
            <div className="cavani_tm_title">
              <span>Publications</span>
            </div>
            <div className="news_list">
              <ul>
                {newsData.map((news, i) => (
                  <li data-img={news.img} key={i}>
                    <div className="list_inner">
                      <span className="number">{`${i <= 9 ? 0 : ""}${i + 1
                        }`}</span>
                      <div className="details">
                        <div className="extra_metas">
                          <ul>
                            <li>
                              <span>{news.date}</span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.tag}
                                </a>
                              </span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.comments}
                                </a>
                              </span>
                            </li>
                          </ul>
                        </div>
                        <div className="post_title">
                          <h3>
                            <a href="#" onClick={() => toggleModalFour(news)}>
                              {news.title}
                            </a>
                          </h3>
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
                  <i className="icon-cancel"></i>
                </a>
              </div>
              <div className="description_wrap">
                <div className="news_popup_informations">
                  <div className="image">
                    <img src="img/thumbs/4-2.jpg" alt="" />
                    <div
                      className="main"
                      data-img-url="img/news/1.jpg"
                      style={{ backgroundImage: `url(${modalContent.img})` }}
                    />
                  </div>
                  <div className="details">
                    <div className="meta">
                      <ul>
                        <li><span>{modalContent.date}</span></li>
                        <li><span><a href="#">{modalContent.tag}</a></span></li>
                        <li><span><a href="#">{modalContent.comments}</a></span></li>
                      </ul>
                    </div>
                    <div className="title">
                      <h3>{modalContent.title}</h3>
                    </div>
                  </div>
                  <div className="text">
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
    </Fragment>
  );
};
export default News;
