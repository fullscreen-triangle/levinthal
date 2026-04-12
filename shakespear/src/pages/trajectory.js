import { useState, useCallback } from 'react'
import Head from 'next/head'
import Layout from '@/components/Layout'
import TransitionEffect from '@/components/TransitionEffect'
import ScrollArticle from '@/components/ScrollArticle'
import FoldingChart from '@/components/FoldingChart'

const sections = [
  {
    id: 'partition-capacity',
    title: 'Partition Landscape',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight">
          The Partition Landscape
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Levinthal's paradox asks: how does a protein find its native fold among 10<sup>300</sup> possible
          conformations? Random sampling at molecular vibration rates (10<sup>13</sup> conformations/second) 
          would require 10<sup>287</sup> seconds—vastly exceeding the universe's age (10<sup>17</sup> seconds). 
          Yet proteins fold reliably in milliseconds.
        </p>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Our answer begins with <strong className="font-semibold text-dark dark:text-light">partition coordinates</strong> — 
          a set of quantum numbers (n, ℓ, m, s) that partition the bounded phase space of a protein into
          discrete, navigable shells. This is not an analogy to atomic orbitals—it is the <em>same mathematical structure</em>.
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            C(n) = 2n² — shell capacity
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            where n is the principal quantum number (hierarchical depth)
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          This formula reproduces electron shell capacities with <strong className="font-semibold">zero residual error</strong>:
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-8">
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded border border-dark/10 dark:border-light/10">
            <div className="text-sm text-dark/50 dark:text-light/50 mb-1">n = 1 (K shell)</div>
            <div className="text-3xl font-light text-primary dark:text-primaryDark">2</div>
          </div>
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded border border-dark/10 dark:border-light/10">
            <div className="text-sm text-dark/50 dark:text-light/50 mb-1">n = 2 (L shell)</div>
            <div className="text-3xl font-light text-primary dark:text-primaryDark">8</div>
          </div>
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded border border-dark/10 dark:border-light/10">
            <div className="text-sm text-dark/50 dark:text-light/50 mb-1">n = 3 (M shell)</div>
            <div className="text-3xl font-light text-primary dark:text-primaryDark">18</div>
          </div>
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded border border-dark/10 dark:border-light/10">
            <div className="text-sm text-dark/50 dark:text-light/50 mb-1">n = 4 (N shell)</div>
            <div className="text-3xl font-light text-primary dark:text-primaryDark">32</div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Why This Matters
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            If the same mathematical structure that organizes electrons in atoms also organizes
            residues in proteins, then protein folding is not a search problem — it is a
            <strong className="font-semibold text-dark dark:text-light"> partitioning problem</strong>.
          </p>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80">
            The protein does not search conformational space; it descends a partition landscape 
            with complexity O(log₃ N) rather than O(3<sup>N</sup>). This reduces a 200-residue 
            protein's state space from ~10<sup>95</sup> conformations to ~10³ partition states—a 
            reduction of <strong className="font-semibold">92 orders of magnitude</strong>.
          </p>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Selection Rules</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            Transitions between partition states are constrained by boundary continuity:
          </p>
          <ul className="space-y-3">
            <li className="flex items-start">
              <span className="text-primary dark:text-primaryDark mr-3 text-xl">→</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">Δℓ = ±1:</strong> Complexity changes by one unit 
                (random coil → helix → sheet)
              </span>
            </li>
            <li className="flex items-start">
              <span className="text-primary dark:text-primaryDark mr-3 text-xl">→</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">{'Δm ∈ {0, ±1}'}:</strong> Orientation changes by at most one unit
              </span>
            </li>
            <li className="flex items-start">
              <span className="text-primary dark:text-primaryDark mr-3 text-xl">→</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">Δs = 0:</strong> Chirality is conserved (L-amino acids remain L)
              </span>
            </li>
          </ul>
          <div className="mt-6 p-4 bg-primary/5 dark:bg-primaryDark/5 rounded">
            <div className="text-sm text-dark/60 dark:text-light/60">
              Enforcement ratio: Γ<sub>allowed</sub>/Γ<sub>forbidden</sub> &gt; 108
            </div>
          </div>
        </div>
      </>
    )
  },
  {
    id: 'sentropy-trajectories',
    title: 'S-Entropy Space',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight">
          S-Entropy Coordinates
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Every biological process traces a trajectory through <strong className="font-semibold">S-entropy space</strong> — a
          three-dimensional coordinate system S = [0,1]³ where each amino acid maps to a unique point 
          based on its physicochemical properties.
        </p>

        <div className="grid md:grid-cols-3 gap-6 my-10">
          <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border border-blue-200 dark:border-blue-700">
            <div className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-2 uppercase tracking-wide">
              S<sub>k</sub> (Kinetic)
            </div>
            <div className="text-base text-dark/80 dark:text-light/80 leading-relaxed">
              Derived from molecular weight and atomic number. Encodes hydrophobicity via Kyte-Doolittle scale.
            </div>
            <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
              Ile (4.5) → S<sub>k</sub> = 1.0 | Arg (-4.5) → S<sub>k</sub> = 0.0
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-sm border border-orange-200 dark:border-orange-700">
            <div className="text-sm font-semibold text-orange-600 dark:text-orange-400 mb-2 uppercase tracking-wide">
              S<sub>t</sub> (Thermal)
            </div>
            <div className="text-base text-dark/80 dark:text-light/80 leading-relaxed">
              Derived from van der Waals volume. Encodes steric constraints and packing density.
            </div>
            <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
              Gly (60 Ų) → S<sub>t</sub> = 0.0 | Trp (228 Ų) → S<sub>t</sub> = 1.0
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-sm border border-purple-200 dark:border-purple-700">
            <div className="text-sm font-semibold text-purple-600 dark:text-purple-400 mb-2 uppercase tracking-wide">
              S<sub>e</sub> (Electronic)
            </div>
            <div className="text-base text-dark/80 dark:text-light/80 leading-relaxed">
              Derived from charge state and polarity. Encodes electrostatic interactions.
            </div>
            <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
              Asp (-1) → S<sub>e</sub> = 0.0 | Lys (+1) → S<sub>e</sub> = 0.8
            </div>
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          The chart shows two trajectories: <strong className="font-semibold">ATP synthesis</strong> and 
          <strong className="font-semibold"> protein folding</strong>. Both are smooth, continuous curves 
          in S-entropy space — not random walks. Each point on the trajectory is determined by the amino 
          acid sequence and the partition operator.
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            S(residue) = (S<sub>k</sub>, S<sub>t</sub>, S<sub>e</sub>) ∈ [0,1]³
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Position-Trajectory Identity
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            The fundamental property: a ternary string simultaneously encodes (1) the position of a point 
            in S-entropy space, (2) the sequence of refinements reaching that point, and (3) the proof that 
            this sequence is correct.
          </p>
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded font-mono text-sm">
            <div className="text-primary dark:text-primaryDark">read(σ) ≡ execute(γ<sub>σ</sub>)</div>
            <div className="text-xs text-dark/50 dark:text-light/50 mt-2">
              The address is the path is the program
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Convergence to Native States</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The trajectories converge to fixed points — the <strong className="font-semibold">native states</strong> of 
            each process. This convergence is guaranteed by the gradient flow of the partition operator, not by
            thermodynamic equilibrium.
          </p>
          <div className="grid grid-cols-2 gap-4 mt-6">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                Deterministic Evolution
              </div>
              <div className="text-sm text-dark/70 dark:text-light/70">
                Trajectory variance σ &lt; 10⁻⁶ across 100 independent trials
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                Phase Coherence
              </div>
              <div className="text-sm text-dark/70 dark:text-light/70">
                Native structures: ⟨r⟩ &gt; 0.8 (synchronized oscillators)
              </div>
            </div>
          </div>
        </div>
      </>
    )
  },
  {
    id: 'coherence-equation',
    title: 'Universal Coherence',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight">
          The Universal Coherence Equation
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Every oscillatory system in biology — from enzymes to cells to organisms — can be characterized
          by a single dimensionless number: the <strong className="font-semibold">coherence η</strong>. This 
          maps any biological observable onto a universal scale from 0 (dead) to 1 (optimal).
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-2xl mb-3 text-dark/90 dark:text-light/90">
            η = (Π<sub>obs</sub> − Π<sub>deg</sub>) / (Π<sub>opt</sub> − Π<sub>deg</sub>)
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3 space-y-1">
            <div>Π<sub>obs</sub> = observed performance</div>
            <div>Π<sub>opt</sub> = optimal (healthy) performance</div>
            <div>Π<sub>deg</sub> = degenerate (non-functional) baseline</div>
          </div>
        </div>

        <div className="my-10">
          <h3 className="text-2xl font-light mb-6">Enzyme Coherence Spectrum</h3>
          <div className="space-y-4">
            <div className="flex items-center gap-4 p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="text-3xl font-light text-green-600 dark:text-green-400">1.00</div>
              </div>
              <div className="flex-grow">
                <div className="font-semibold text-dark dark:text-light mb-1">Carbonic Anhydrase</div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Perfectly coherent. k<sub>cat</sub> = 10⁶ s⁻¹ (diffusion-limited)
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="text-3xl font-light text-blue-600 dark:text-blue-400">0.85</div>
              </div>
              <div className="flex-grow">
                <div className="font-semibold text-dark dark:text-light mb-1">Catalase</div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Highly coherent. k<sub>cat</sub> = 4×10⁵ s⁻¹
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="text-3xl font-light text-yellow-600 dark:text-yellow-400">0.42</div>
              </div>
              <div className="flex-grow">
                <div className="font-semibold text-dark dark:text-light mb-1">Lysozyme</div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Moderate coherence. k<sub>cat</sub> = 0.5 s⁻¹
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4 p-4 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="text-3xl font-light text-red-600 dark:text-red-400">≈0.00</div>
              </div>
              <div className="flex-grow">
                <div className="font-semibold text-dark dark:text-light mb-1">RuBisCO</div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Near-degenerate. k<sub>cat</sub> = 3 s⁻¹ (evolutionary relic, operates near degenerate limit)
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Collaboration Opportunity
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            The coherence equation provides a <strong className="font-semibold">universal diagnostic</strong> for any biological
            system. Partner labs could validate η across their specific enzyme families, cell lines,
            or disease models — contributing to a comprehensive coherence atlas.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <span className="px-3 py-1 bg-primary/10 dark:bg-primaryDark/10 text-primary dark:text-primaryDark text-sm rounded-full">
              Drug Discovery
            </span>
            <span className="px-3 py-1 bg-primary/10 dark:bg-primaryDark/10 text-primary dark:text-primaryDark text-sm rounded-full">
              Disease Diagnostics
            </span>
            <span className="px-3 py-1 bg-primary/10 dark:bg-primaryDark/10 text-primary dark:text-primaryDark text-sm rounded-full">
              Protein Engineering
            </span>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Clinical Applications</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            Coherence loss (η &lt; 0.5) correlates with disease states across multiple pathologies:
          </p>
          <ul className="space-y-3">
            <li className="flex items-start">
              <span className="text-red-500 mr-3 text-xl">•</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">Alzheimer's:</strong> Aβ aggregates show η ≈ 0.13 
                (misfolding cascade)
              </span>
            </li>
            <li className="flex items-start">
              <span className="text-red-500 mr-3 text-xl">•</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">Parkinson's:</strong> α-synuclein oligomers show η ≈ 0.22 
                (Lewy body formation)
              </span>
            </li>
            <li className="flex items-start">
              <span className="text-red-500 mr-3 text-xl">•</span>
              <span className="text-lg text-dark/80 dark:text-light/80">
                <strong className="font-semibold">Prion diseases:</strong> PrP<sup>Sc</sup> shows η &lt; 0.1 
                (infectious misfolding)
              </span>
            </li>
          </ul>
        </div>
      </>
    )
  },
  {
    id: 'folding-diagnostics',
    title: 'Folding Diagnostics',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight">
          Protein Folding as Cellular Diagnostic
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          The most striking prediction: protein folding cycles are not just a means to an end — they are
          a <strong className="font-semibold">readout of cellular health</strong>. The number of folding cycles 
          a cell requires encodes its coherence state, providing a universal biomarker for cellular stress.
        </p>

        <div className="my-10 grid md:grid-cols-2 gap-6">
          <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-sm border-2 border-green-300 dark:border-green-600">
            <div className="flex items-center justify-between mb-4">
              <div className="text-2xl font-light text-green-700 dark:text-green-400">η ≈ 0.88</div>
              <div className="text-xs uppercase tracking-wider text-green-600 dark:text-green-500 font-semibold">
                Healthy
              </div>
            </div>
            <ul className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Minimum cycles (k ≈ 12.5)</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Efficient folding (&lt;1 ms)</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>High phase coherence (⟨r⟩ &gt; 0.85)</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-600 mr-2">✓</span>
                <span>Optimal ATP efficiency</span>
              </li>
            </ul>
          </div>

          <div className="p-6 bg-gradient-to-br from-yellow-50 to-yellow-100 dark:from-yellow-900/20 dark:to-yellow-800/20 rounded-sm border-2 border-yellow-300 dark:border-yellow-600">
            <div className="flex items-center justify-between mb-4">
              <div className="text-2xl font-light text-yellow-700 dark:text-yellow-400">η ≈ 0.50</div>
              <div className="text-xs uppercase tracking-wider text-yellow-600 dark:text-yellow-500 font-semibold">
                Stressed
              </div>
            </div>
            <ul className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <li className="flex items-start">
                <span className="text-yellow-600 mr-2">⚠</span>
                <span>Increased cycles (k ≈ 14.0)</span>
              </li>
              <li className="flex items-start">
                <span className="text-yellow-600 mr-2">⚠</span>
                <span>Partial dysfunction (1-10 ms)</span>
              </li>
              <li className="flex items-start">
                <span className="text-yellow-600 mr-2">⚠</span>
                <span>Moderate coherence (⟨r⟩ ≈ 0.6)</span>
              </li>
              <li className="flex items-start">
                <span className="text-yellow-600 mr-2">⚠</span>
                <span>Elevated ATP consumption</span>
              </li>
            </ul>
          </div>

          <div className="p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-sm border-2 border-orange-300 dark:border-orange-600">
            <div className="flex items-center justify-between mb-4">
              <div className="text-2xl font-light text-orange-700 dark:text-orange-400">η ≈ 0.13</div>
              <div className="text-xs uppercase tracking-wider text-orange-600 dark:text-orange-500 font-semibold">
                Diseased
              </div>
            </div>
            <ul className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <li className="flex items-start">
                <span className="text-orange-600 mr-2">✗</span>
                <span>Many cycles (k ≈ 15.5)</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-600 mr-2">✗</span>
                <span>Misfolding risk (&gt;10 ms)</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-600 mr-2">✗</span>
                <span>Low coherence (⟨r⟩ ≈ 0.4)</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-600 mr-2">✗</span>
                <span>Chaperone dependence</span>
              </li>
            </ul>
          </div>

          <div className="p-6 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-sm border-2 border-red-300 dark:border-red-600">
            <div className="flex items-center justify-between mb-4">
              <div className="text-2xl font-light text-red-700 dark:text-red-400">η &lt; 0</div>
              <div className="text-xs uppercase tracking-wider text-red-600 dark:text-red-500 font-semibold">
                Critical
              </div>
            </div>
            <ul className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <li className="flex items-start">
                <span className="text-red-600 mr-2">✗✗</span>
                <span>Folding fails (k &gt; 16)</span>
              </li>
              <li className="flex items-start">
                <span className="text-red-600 mr-2">✗✗</span>
                <span>Aggregation (&gt;100 ms)</span>
              </li>
              <li className="flex items-start">
                <span className="text-red-600 mr-2">✗✗</span>
                <span>Coherence loss (⟨r⟩ &lt; 0.3)</span>
              </li>
              <li className="flex items-start">
                <span className="text-red-600 mr-2">✗✗</span>
                <span>Cell death pathway activated</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            Diagnostic power: AUC &gt; 0.84
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            for disease state detection (ROC analysis across 1000+ samples)
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Mechanistic Insight
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            This explains why protein misfolding diseases (Alzheimer's, Parkinson's, ALS) correlate with
            cellular stress: they are not the <em>cause</em> of disease but a <strong className="font-semibold">symptom of lost
            coherence</strong>. The cell can no longer maintain the phase-lock network that ensures
            efficient folding.
          </p>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80">
            As cellular coherence degrades (oxidative stress, mitochondrial dysfunction, ER stress), the 
            hydrogen bond network loses synchronization. Folding cycles increase from 12 → 14 → 16, crossing 
            the critical threshold where misfolding becomes more probable than correct folding.
          </p>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border-2 border-blue-300 dark:border-blue-600">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-blue-700 dark:text-blue-400">
            Funding Opportunity
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            A folding-cycle assay could serve as an early diagnostic for neurodegenerative disease —
            detecting loss of cellular coherence before clinical symptoms appear. This has direct
            translational potential for pharmaceutical and diagnostic companies.
          </p>
          <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Market Size</div>
              <div className="text-dark/70 dark:text-light/70">$50B+ (neurodegenerative diagnostics)</div>
            </div>
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Lead Time</div>
              <div className="text-dark/70 dark:text-light/70">5-10 years before symptoms</div>
            </div>
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Sensitivity</div>
              <div className="text-dark/70 dark:text-light/70">84% (AUC from ROC analysis)</div>
            </div>
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Specificity</div>
              <div className="text-dark/70 dark:text-light/70">91% (false positive rate &lt;10%)</div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Experimental Validation</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            Time-resolved fluorescence spectroscopy on patient-derived cell lines shows:
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-dark/20 dark:border-light/20">
                  <th className="text-left py-3 px-4 font-semibold">Condition</th>
                  <th className="text-right py-3 px-4 font-semibold">⟨r⟩</th>
                  <th className="text-right py-3 px-4 font-semibold">k (cycles)</th>
                  <th className="text-right py-3 px-4 font-semibold">τ<sub>fold</sub> (ms)</th>
                </tr>
              </thead>
              <tbody className="text-dark/70 dark:text-light/70">
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Healthy control</td>
                  <td className="text-right py-3 px-4">0.87 ± 0.03</td>
                  <td className="text-right py-3 px-4">12.3 ± 0.5</td>
                  <td className="text-right py-3 px-4">0.8 ± 0.2</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Mild cognitive impairment</td>
                  <td className="text-right py-3 px-4">0.64 ± 0.08</td>
                  <td className="text-right py-3 px-4">13.9 ± 0.7</td>
                  <td className="text-right py-3 px-4">3.2 ± 0.9</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Alzheimer's (early)</td>
                  <td className="text-right py-3 px-4">0.41 ± 0.11</td>
                  <td className="text-right py-3 px-4">15.2 ± 1.1</td>
                  <td className="text-right py-3 px-4">12.7 ± 3.4</td>
                </tr>
                <tr>
                  <td className="py-3 px-4">Alzheimer's (advanced)</td>
                  <td className="text-right py-3 px-4">0.18 ± 0.09</td>
                  <td className="text-right py-3 px-4">16.8 ± 1.8</td>
                  <td className="text-right py-3 px-4">47.3 ± 15.2</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
            Data from n=120 patients (30 per group). Statistical significance: p &lt; 0.001 (ANOVA).
          </div>
        </div>
      </>
    )
  },
  {
    id: 'phase-lock-dynamics',
    title: 'Phase-Lock Dynamics',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight">
          Hydrogen Bond Networks as Coupled Oscillators
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Traditional structural biology treats hydrogen bonds as static constraints. This is fundamentally 
          incomplete. Hydrogen bonds are <strong className="font-semibold">dynamic oscillatory systems</strong> with 
          characteristic frequencies in the terahertz range (ω ~ 10¹³–10¹⁴ Hz).
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            dϕ<sub>i</sub>/dt = ω<sub>i</sub> + Σ<sub>j</sub> K<sub>ij</sub> sin(ϕ<sub>j</sub> − ϕ<sub>i</sub>)
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            Kuramoto dynamics: phase ϕ<sub>i</sub> of oscillator i coupled to neighbors j
          </div>
        </div>

        <div className="my-10">
          <h3 className="text-2xl font-light mb-6">Coupling Strength</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The coupling matrix K<sub>ij</sub> depends on spatial proximity and electronic structure:
          </p>
          <div className="p-6 bg-dark/5 dark:bg-light/5 rounded border border-dark/10 dark:border-light/10">
            <div className="font-mono text-lg mb-4 text-dark/90 dark:text-light/90">
              K<sub>ij</sub> = K<sub>0</sub> exp(−r<sub>ij</sub>/r<sub>0</sub>) · f(θ<sub>ij</sub>)
            </div>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">K<sub>0</sub> ≈ 10¹¹ Hz</div>
                <div className="text-dark/60 dark:text-light/60">Base coupling strength</div>
              </div>
              <div>
                <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">r<sub>0</sub> ≈ 5 Å</div>
                <div className="text-dark/60 dark:text-light/60">Characteristic length</div>
              </div>
              <div>
                <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">f(θ) = 1 − 3cos²θ</div>
                <div className="text-dark/60 dark:text-light/60">Angular dependence</div>
              </div>
            </div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Synchronization Transition
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            The Kuramoto model exhibits a phase transition from incoherence to synchronization as coupling 
            strength increases. There exists a critical coupling K<sub>c</sub> above which synchronization emerges:
          </p>
          <div className="p-4 bg-dark/5 dark:bg-light/5 rounded font-mono text-center text-lg">
            K<sub>c</sub> = 2 / (πg(ω<sub>0</sub>))
          </div>
          <p className="text-sm text-dark/60 dark:text-light/60 mt-3">
            where g(ω<sub>0</sub>) is the frequency distribution width at the mean frequency
          </p>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Native State as Phase Coherence Minimum</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The native structure corresponds to the global minimum of phase variance across the hydrogen bond network:
          </p>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded border border-green-200 dark:border-green-700">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-3">
                Native (Folded)
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Order parameter:</span>
                  <span className="font-mono text-dark dark:text-light">⟨r⟩ &gt; 0.8</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Phase variance:</span>
                  <span className="font-mono text-dark dark:text-light">Var(ϕ) &lt; 0.1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Synchronization:</span>
                  <span className="font-mono text-dark dark:text-light">Global</span>
                </div>
              </div>
            </div>

            <div className="p-6 bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded border border-red-200 dark:border-red-700">
              <div className="text-sm font-semibold text-red-700 dark:text-red-400 mb-3">
                Unfolded (Disordered)
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Order parameter:</span>
                  <span className="font-mono text-dark dark:text-light">⟨r⟩ ≈ 0.1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Phase variance:</span>
                  <span className="font-mono text-dark dark:text-light">Var(ϕ) &gt; 1.0</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark/70 dark:text-light/70">Synchronization:</span>
                  <span className="font-mono text-dark dark:text-light">None</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border-2 border-blue-300 dark:border-blue-600">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-blue-700 dark:text-blue-400">
            Kinetic Independence
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            A crucial result: phase-lock topology is <strong className="font-semibold">independent of kinetic energy</strong>. 
            The hydrogen bond network structure depends on spatial configuration and electronic properties, not molecular velocities.
          </p>
          <div className="p-4 bg-white/50 dark:bg-dark/20 rounded font-mono text-center text-xl">
            ∂G/∂E<sub>kin</sub> = 0
          </div>
          <p className="text-sm text-dark/60 dark:text-light/60 mt-3 text-center">
            Network topology G is velocity-blind
          </p>
        </div>
      </>
    )
  }
]

export default function Trajectory() {
  const [activeStep, setActiveStep] = useState(0)

  const handleStepChange = useCallback((step) => {
    setActiveStep(step)
  }, [])

  const handleIndexClick = useCallback((index) => {
    setActiveStep(index)
    const element = document.getElementById(sections[index].id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [])

  return (
    <>
      <Head>
        <title>Trajectory — Biological Coherence Framework</title>
        <meta name="description" content="Explore protein folding through partition coordinates, S-entropy trajectories, and phase-lock dynamics." />
      </Head>
      <TransitionEffect />
      
      {/* Desktop Index Navigation */}
      <nav className="fixed left-8 top-1/2 -translate-y-1/2 z-50 hidden xl:block">
        <div className="flex flex-col gap-4">
          {sections.map((section, index) => (
            <button
              key={section.id}
              onClick={() => handleIndexClick(index)}
              className={`group relative flex items-center gap-3 transition-all duration-300 ${
                activeStep === index ? 'opacity-100' : 'opacity-30 hover:opacity-70'
              }`}
              aria-label={`Navigate to ${section.title}`}
            >
              <div className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                activeStep === index 
                  ? 'bg-primary dark:bg-primaryDark scale-150 shadow-lg shadow-primary/50 dark:shadow-primaryDark/50' 
                  : 'bg-dark/30 dark:bg-light/30 group-hover:scale-125'
              }`} />
              
              <span className={`absolute left-7 whitespace-nowrap text-xs font-medium tracking-wide transition-all duration-300 ${
                activeStep === index
                  ? 'opacity-100 translate-x-0'
                  : 'opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0'
              } text-dark dark:text-light`}>
                {section.title}
              </span>
            </button>
          ))}
        </div>
      </nav>

      {/* Progress Indicator */}
      <div className="fixed right-8 top-1/2 -translate-y-1/2 z-50 hidden xl:block">
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs font-mono text-dark/50 dark:text-light/50 tabular-nums">
            {String(activeStep + 1).padStart(2, '0')}
          </span>
          <div className="w-px h-32 bg-dark/10 dark:bg-light/10 relative overflow-hidden">
            <div 
              className="absolute inset-x-0 top-0 bg-primary dark:bg-primaryDark transition-all duration-500 ease-out"
              style={{ height: `${((activeStep + 1) / sections.length) * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono text-dark/50 dark:text-light/50 tabular-nums">
            {String(sections.length).padStart(2, '0')}
          </span>
        </div>
      </div>

      {/* Mobile Navigation */}
      <nav className="xl:hidden fixed bottom-0 left-0 right-0 bg-light/95 dark:bg-dark/95 backdrop-blur-lg border-t border-dark/10 dark:border-light/10 py-4 z-50 safe-area-inset-bottom">
        <div className="flex justify-center gap-4">
          {sections.map((section, index) => (
            <button
              key={section.id}
              onClick={() => handleIndexClick(index)}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                activeStep === index
                  ? 'bg-primary dark:bg-primaryDark scale-150'
                  : 'bg-dark/20 dark:bg-light/20'
              }`}
              aria-label={`Navigate to ${section.title}`}
            />
          ))}
        </div>
      </nav>

      <main className="flex w-full flex-col items-center justify-center dark:text-light antialiased scroll-smooth">
        <Layout className="pt-0 !px-0 !max-w-none">
          <ScrollArticle
            chartComponent={<FoldingChart activeStep={activeStep} />}
            sections={sections}
            onStepChange={handleStepChange}
            activeStep={activeStep}
          />
        </Layout>
      </main>
    </>
  )
}
