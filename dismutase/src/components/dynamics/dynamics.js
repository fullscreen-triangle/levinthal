import React, { useState, useCallback } from 'react'
import ScrollArticle from '../ScrollArticle/ScrollArticle'
import DynamicsChart from './DynamicsChart'

const sections = [
  {
    id: 'docking-trajectory',
    content: (
      <>
        <h2>Protein–Ligand Docking as Categorical Trajectory</h2>
        <p>
          Protein dynamics is governed by the motion of atoms through <strong>categorical states</strong> — discrete
          quantum-like configurations defined by partition coordinates (n, l, m, s). When a ligand approaches a
          protein binding site, it does not simply diffuse randomly; it follows a deterministic trajectory through
          these categorical states.
        </p>
        <div className="equation">
          dx/dt = −γ ∇M(x) — gradient descent through the partition landscape
        </div>
        <p>
          The chart shows azurin (PDB: 4AZU) docking a copper ligand across 100 iterations. The ligand starts
          20 Å from the binding site and converges to within 0.93 Å — achieving <strong>100% binding site
          accuracy</strong> by detecting all four coordinating residues (His46, His112, Cys117, Met121).
        </p>
        <div className="insight-box">
          <h4>Key Result</h4>
          <p>
            The docking trajectory is not a simulation of molecular dynamics — it is a <strong>categorical
            computation</strong>. Each step reclassifies atoms into ground, natural, or excited states based
            on their partition coordinates, and the ligand moves along the gradient of the partition operator.
          </p>
        </div>
      </>
    )
  },
  {
    id: 'ternary-distribution',
    content: (
      <>
        <h2>Ternary State Classification</h2>
        <p>
          At each step of the docking trajectory, every atom in the protein is classified into one of
          three categorical states:
        </p>
        <ul>
          <li><strong>Ground (0)</strong> — atoms at their equilibrium partition coordinate</li>
          <li><strong>Natural (1)</strong> — atoms displaced but within the natural bandwidth</li>
          <li><strong>Excited (2)</strong> — atoms perturbed beyond their natural configuration</li>
        </ul>
        <p>
          This ternary classification is fundamental to the framework. It maps the continuous configuration
          space of a protein onto a discrete, finite alphabet — making protein dynamics computable in the
          information-theoretic sense.
        </p>
        <div className="equation">
          {"State(atom) ∈ {0, 1, 2} — a trit (ternary digit)"}
        </div>
        <p>
          As the ligand approaches, the distribution shifts: excited-state atoms increase as the binding
          site reorganizes to accommodate the ligand. The <strong>natural → excited</strong> transition
          at the binding site is the categorical signature of molecular recognition.
        </p>
      </>
    )
  },
  {
    id: 'ternary-encoding',
    content: (
      <>
        <h2>Base-3 Trajectory Encoding</h2>
        <p>
          The entire docking trajectory can be encoded as a <strong>ternary string</strong> — a sequence of
          trits (0, 1, 2) where each position represents the dominant categorical state at that docking step.
        </p>
        <div className="equation">
          {"T = t₁t₂t₃...t_N where t_i ∈ {0, 1, 2}"}
        </div>
        <p>
          For the azurin docking, the ternary string is a sequence of 2s (all excited), reflecting that the
          protein is constantly reorganizing around the approaching ligand. This uniform excitation is
          characteristic of <strong>active binding</strong> — the protein is not passive; it actively
          restructures its partition landscape to capture the ligand.
        </p>
        <div className="insight-box">
          <h4>Why Ternary?</h4>
          <p>
            Binary encoding (folded/unfolded) loses the intermediate states that drive dynamics. The ternary
            basis captures the full categorical structure: <span className="metric">3^N</span> possible
            states for N atoms, encoding position, transition, and trajectory in a single string.
          </p>
        </div>
        <p>
          Each ternary digit carries <span className="metric">log₂(3) ≈ 1.585 bits</span> of information.
          A protein with 4,228 atoms encodes <span className="metric">~6,700 bits</span> of structural
          information per step — sufficient to specify the complete categorical state.
        </p>
      </>
    )
  },
  {
    id: 'convergence',
    content: (
      <>
        <h2>Convergence and Binding</h2>
        <p>
          The dual-axis view reveals the relationship between geometric convergence (ligand distance)
          and categorical reorganization (excited state count). As the ligand approaches:
        </p>
        <ul>
          <li>Ligand distance decreases monotonically from 20.0 Å to 0.93 Å</li>
          <li>Excited state count rises then plateaus — the protein has fully reorganized</li>
          <li>Final distribution: <span className="metric">2,145 natural</span> / <span className="metric">2,083 excited</span></li>
        </ul>
        <p>
          The near-equal split between natural and excited states at convergence is significant: it means
          the binding event engages approximately <strong>half the protein</strong>. This is not a local
          perturbation — molecular recognition is a <strong>global categorical transition</strong>.
        </p>
        <div className="equation">
          Binding accuracy = 1.000 — all 4 coordinating residues detected
        </div>
        <div className="insight-box">
          <h4>Collaboration Opportunity</h4>
          <p>
            This framework predicts binding sites from first principles, without training data or homology.
            It could transform drug discovery by computing protein–ligand interactions as categorical
            trajectories rather than expensive molecular dynamics simulations.
          </p>
        </div>
      </>
    )
  }
]

export default function Dynamics({ ActiveIndex }) {
  const [activeStep, setActiveStep] = useState(0)

  const handleStepChange = useCallback((step) => {
    setActiveStep(step)
  }, [])

  return (
    <div
      className={ActiveIndex === 1 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"}
      id="dynamics_"
    >
      <div className="section_inner" style={{ padding: 0, maxWidth: 'none' }}>
        <ScrollArticle
          chartComponent={<DynamicsChart activeStep={activeStep} />}
          sections={sections}
          onStepChange={handleStepChange}
          activeStep={activeStep}
        />
      </div>
    </div>
  )
}
