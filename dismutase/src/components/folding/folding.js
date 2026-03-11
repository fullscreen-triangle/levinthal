import React, { useState, useCallback } from 'react'
import ScrollArticle from '../ScrollArticle/ScrollArticle'
import FoldingChart from './FoldingChart'

const sections = [
  {
    id: 'partition-capacity',
    content: (
      <>
        <h2>The Partition Landscape</h2>
        <p>
          Levinthal's paradox asks: how does a protein find its native fold among 10^300 possible
          conformations? Our answer begins with <strong>partition coordinates</strong> — a set of
          quantum numbers (n, l, m, s) that partition the bounded phase space of a protein into
          discrete, navigable shells.
        </p>
        <div className="equation">
          C(n) = Σ(2l + 1) × 2 = 2n² — shell capacity
        </div>
        <p>
          This is not an analogy. The partition capacity formula <strong>C(n) = 2n²</strong> is
          derived from first principles by counting the number of distinguishable categorical states
          within a bounded spherical phase space. It reproduces the electron shell capacities
          (K=2, L=8, M=18, N=32...) with <strong>zero residual error</strong> across all seven shells.
        </p>
        <div className="insight-box">
          <h4>Why This Matters</h4>
          <p>
            If the same mathematical structure that organizes electrons in atoms also organizes
            residues in proteins, then protein folding is not a search problem — it is a
            <strong> partitioning problem</strong>. The protein does not search conformational space;
            it descends a partition landscape.
          </p>
        </div>
      </>
    )
  },
  {
    id: 'sentropy-trajectories',
    content: (
      <>
        <h2>S-Entropy Coordinates</h2>
        <p>
          Every biological process traces a trajectory through <strong>S-entropy space</strong> — a
          three-dimensional coordinate system S = [0,1]³ with axes:
        </p>
        <ul>
          <li><strong>Sₖ (Kinetic entropy)</strong> — derived from molecular weight and atomic number</li>
          <li><strong>Sₜ (Thermal entropy)</strong> — derived from hydropathy and charge</li>
          <li><strong>Sₑ (Electronic entropy)</strong> — derived from electron count and orbital occupancy</li>
        </ul>
        <p>
          The chart shows two trajectories: <strong>ATP synthesis</strong> and <strong>protein folding</strong>.
          Both are smooth, continuous curves in S-entropy space — not random walks. Each point on the
          trajectory is determined by the amino acid sequence and the partition operator.
        </p>
        <div className="equation">
          S(residue) = (Sₖ, Sₜ, Sₑ) ∈ [0,1]³
        </div>
        <p>
          The trajectories converge to fixed points — the <strong>native states</strong> of each process.
          This convergence is guaranteed by the gradient flow of the partition operator, not by
          thermodynamic equilibrium.
        </p>
      </>
    )
  },
  {
    id: 'coherence-equation',
    content: (
      <>
        <h2>The Universal Coherence Equation</h2>
        <p>
          Every oscillatory system in biology — from enzymes to cells to organisms — can be characterized
          by a single dimensionless number: the <strong>coherence η</strong>.
        </p>
        <div className="equation">
          η = (Π_obs − Π_deg) / (Π_opt − Π_deg)
        </div>
        <p>
          Where Π_obs is the observed performance, Π_opt is the optimal (healthy) performance, and
          Π_deg is the degenerate (non-functional) baseline. This maps any biological observable onto
          a universal scale from 0 (dead) to 1 (optimal).
        </p>
        <p>
          For enzymes, η maps catalytic rate to health: <strong>Carbonic anhydrase</strong> (η = 1.00)
          is perfectly coherent, while <strong>RuBisCO</strong> (η ≈ 0) operates near the degenerate
          limit — which explains its notorious inefficiency as an evolutionary relic.
        </p>
        <div className="insight-box">
          <h4>Collaboration Opportunity</h4>
          <p>
            The coherence equation provides a <strong>universal diagnostic</strong> for any biological
            system. Partner labs could validate η across their specific enzyme families, cell lines,
            or disease models — contributing to a comprehensive coherence atlas.
          </p>
        </div>
      </>
    )
  },
  {
    id: 'folding-diagnostics',
    content: (
      <>
        <h2>Protein Folding as Cellular Diagnostic</h2>
        <p>
          The most striking prediction: protein folding cycles are not just a means to an end — they are
          a <strong>readout of cellular health</strong>. The number of folding cycles a cell requires
          encodes its coherence state.
        </p>
        <ul>
          <li><strong>Healthy (η ≈ 0.88)</strong> — minimum cycles, efficient folding, k ≈ 12.5</li>
          <li><strong>Stressed (η ≈ 0.50)</strong> — increased cycles, partial dysfunction, k ≈ 14.0</li>
          <li><strong>Diseased (η ≈ 0.13)</strong> — many cycles, misfolding risk, k ≈ 15.5</li>
          <li><strong>Critical (η &lt; 0)</strong> — folding fails, aggregation, cell death, k &gt; 16</li>
        </ul>
        <p>
          This explains why protein misfolding diseases (Alzheimer's, Parkinson's, ALS) correlate with
          cellular stress: they are not the <em>cause</em> of disease but a <strong>symptom of lost
          coherence</strong>. The cell can no longer maintain the phase-lock network that ensures
          efficient folding.
        </p>
        <div className="equation">
          Diagnostic power: AUC &gt; 0.84 for disease state detection
        </div>
        <div className="insight-box">
          <h4>Funding Opportunity</h4>
          <p>
            A folding-cycle assay could serve as an early diagnostic for neurodegenerative disease —
            detecting loss of cellular coherence before clinical symptoms appear. This has direct
            translational potential for pharmaceutical and diagnostic companies.
          </p>
        </div>
      </>
    )
  }
]

export default function Folding({ ActiveIndex }) {
  const [activeStep, setActiveStep] = useState(0)

  const handleStepChange = useCallback((step) => {
    setActiveStep(step)
  }, [])

  return (
    <div
      className={ActiveIndex === 2 ? "cavani_tm_section active animated rollIn" : "cavani_tm_section active hidden animated rollOut"}
      id="folding_"
    >
      <div className="section_inner" style={{ padding: 0, maxWidth: 'none' }}>
        <ScrollArticle
          chartComponent={<FoldingChart activeStep={activeStep} />}
          sections={sections}
          onStepChange={handleStepChange}
          activeStep={activeStep}
        />
      </div>
    </div>
  )
}
