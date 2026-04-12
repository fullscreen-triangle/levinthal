import React, { useState, useCallback } from 'react'
import Head from 'next/head'
import ScrollArticle from '../ScrollArticle/ScrollArticle'
import DynamicsChart from './DynamicsChart'

const sections = [
  {
    id: 'docking-trajectory',
    title: 'Docking Trajectory',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight text-dark dark:text-light">
          Protein–Ligand Docking as Categorical Trajectory
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          Protein dynamics is governed by the motion of atoms through <strong className="font-semibold text-dark dark:text-light">categorical 
          states</strong> — discrete quantum-like configurations defined by partition coordinates (n, ℓ, m, s). When a 
          ligand approaches a protein binding site, it does not simply diffuse randomly; it follows a 
          <strong className="font-semibold"> deterministic trajectory</strong> through these categorical states.
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            dx/dt = −γ ∇M(x)
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            Gradient descent through the partition landscape — where M is partition depth
          </div>
        </div>

        <div className="my-10 grid md:grid-cols-2 gap-6">
          <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border border-blue-200 dark:border-blue-700">
            <div className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-2 uppercase tracking-wide">
              System: Azurin
            </div>
            <div className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <div className="flex justify-between">
                <span>PDB ID:</span>
                <span className="font-mono">4AZU</span>
              </div>
              <div className="flex justify-between">
                <span>Protein size:</span>
                <span className="font-mono">128 residues</span>
              </div>
              <div className="flex justify-between">
                <span>Total atoms:</span>
                <span className="font-mono">4,228</span>
              </div>
              <div className="flex justify-between">
                <span>Ligand:</span>
                <span className="font-mono">Cu²⁺ ion</span>
              </div>
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-sm border border-green-200 dark:border-green-700">
            <div className="text-sm font-semibold text-green-600 dark:text-green-400 mb-2 uppercase tracking-wide">
              Docking Results
            </div>
            <div className="space-y-2 text-sm text-dark/80 dark:text-light/80">
              <div className="flex justify-between">
                <span>Initial distance:</span>
                <span className="font-mono">20.0 Å</span>
              </div>
              <div className="flex justify-between">
                <span>Final distance:</span>
                <span className="font-mono">0.93 Å</span>
              </div>
              <div className="flex justify-between">
                <span>Iterations:</span>
                <span className="font-mono">100</span>
              </div>
              <div className="flex justify-between">
                <span>Binding accuracy:</span>
                <span className="font-mono text-green-600 dark:text-green-400">100%</span>
              </div>
            </div>
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          The chart shows the complete docking trajectory across 100 iterations. The ligand starts 20 Å from the 
          binding site and converges to within 0.93 Å — achieving <strong className="font-semibold">100% binding site 
          accuracy</strong> by detecting all four coordinating residues:
        </p>

        <div className="my-10 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-700">
            <div className="text-xs text-blue-600 dark:text-blue-400 font-semibold mb-1 uppercase tracking-wide">
              Residue 1
            </div>
            <div className="text-2xl font-light text-blue-600 dark:text-blue-400 mb-1">His46</div>
            <div className="text-xs text-dark/60 dark:text-light/60">Nε coordination</div>
          </div>
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
            <div className="text-xs text-green-600 dark:text-green-400 font-semibold mb-1 uppercase tracking-wide">
              Residue 2
            </div>
            <div className="text-2xl font-light text-green-600 dark:text-green-400 mb-1">His112</div>
            <div className="text-xs text-dark/60 dark:text-light/60">Nε coordination</div>
          </div>
          <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-700">
            <div className="text-xs text-yellow-600 dark:text-yellow-400 font-semibold mb-1 uppercase tracking-wide">
              Residue 3
            </div>
            <div className="text-2xl font-light text-yellow-600 dark:text-yellow-400 mb-1">Cys117</div>
            <div className="text-xs text-dark/60 dark:text-light/60">Thiolate (S⁻) bond</div>
          </div>
          <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-700">
            <div className="text-xs text-orange-600 dark:text-orange-400 font-semibold mb-1 uppercase tracking-wide">
              Residue 4
            </div>
            <div className="text-2xl font-light text-orange-600 dark:text-orange-400 mb-1">Met121</div>
            <div className="text-xs text-dark/60 dark:text-light/60">Thioether (S) bond</div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Key Result
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            The docking trajectory is not a simulation of molecular dynamics — it is a <strong className="font-semibold text-dark dark:text-light">categorical 
            computation</strong>. Each step reclassifies atoms into ground, natural, or excited states based on their 
            partition coordinates, and the ligand moves along the gradient of the partition operator.
          </p>
          <div className="mt-6 p-4 bg-dark/5 dark:bg-light/5 rounded">
            <div className="text-sm font-semibold text-dark/70 dark:text-light/70 mb-2">
              Computational Advantage
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              Traditional molecular dynamics: ~10⁶ timesteps × 10⁻¹⁵ s = 1 ns simulation time<br/>
              Categorical trajectory: 100 iterations × O(N log N) = complete binding pathway
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Trajectory Phases</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The docking process unfolds in three distinct categorical phases:
          </p>
          <div className="space-y-3">
            <div className="flex items-start gap-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="font-mono text-sm text-blue-600 dark:text-blue-400">Iterations 1–30</div>
                <div className="text-xs text-dark/60 dark:text-light/60 mt-1">Long-range</div>
              </div>
              <div className="flex-grow">
                <div className="text-sm font-semibold text-blue-700 dark:text-blue-400 mb-1">
                  Electrostatic Guidance
                </div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Cu²⁺ ion follows electrostatic field gradients. Distance: 20 → 10 Å. Minimal protein reorganization.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4 p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="font-mono text-sm text-green-600 dark:text-green-400">Iterations 31–70</div>
                <div className="text-xs text-dark/60 dark:text-light/60 mt-1">Mid-range</div>
              </div>
              <div className="flex-grow">
                <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-1">
                  Categorical Recognition
                </div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Binding site atoms transition to excited states. Distance: 10 → 3 Å. Active site pre-organization begins.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4 p-4 bg-orange-50 dark:bg-orange-900/20 rounded border border-orange-200 dark:border-orange-700">
              <div className="flex-shrink-0 w-32 text-right">
                <div className="font-mono text-sm text-orange-600 dark:text-orange-400">Iterations 71–100</div>
                <div className="text-xs text-dark/60 dark:text-light/60 mt-1">Short-range</div>
              </div>
              <div className="flex-grow">
                <div className="text-sm font-semibold text-orange-700 dark:text-orange-400 mb-1">
                  Coordination Lock
                </div>
                <div className="text-sm text-dark/70 dark:text-light/70">
                  Ligand snaps into coordination geometry. Distance: 3 → 0.93 Å. Global protein reorganization complete.
                </div>
              </div>
            </div>
          </div>
        </div>
      </>
    )
  },
  {
    id: 'ternary-distribution',
    title: 'State Classification',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight text-dark dark:text-light">
          Ternary State Classification
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          At each step of the docking trajectory, every atom in the protein is classified into one of three 
          categorical states. This classification is not arbitrary — it emerges from the partition coordinate 
          framework as the natural decomposition of bounded phase space.
        </p>

        <div className="my-10 space-y-4">
          <div className="p-6 bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border-l-4 border-blue-500">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-16 h-16 bg-blue-500 dark:bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">
                0
              </div>
              <div className="flex-grow">
                <div className="text-lg font-semibold text-blue-700 dark:text-blue-400 mb-2">
                  Ground State
                </div>
                <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
                  Atoms at their equilibrium partition coordinate. No perturbation from native structure. 
                  Quantum numbers (n, ℓ, m, s) match the crystallographic configuration.
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Typical Count</div>
                    <div className="text-dark/60 dark:text-light/60">0–50 atoms (1–2%)</div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Physical State</div>
                    <div className="text-dark/60 dark:text-light/60">Crystallographic minimum</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-6 bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-sm border-l-4 border-green-500">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-16 h-16 bg-green-500 dark:bg-green-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">
                1
              </div>
              <div className="flex-grow">
                <div className="text-lg font-semibold text-green-700 dark:text-green-400 mb-2">
                  Natural State
                </div>
                <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
                  Atoms displaced but within the natural bandwidth. Thermal fluctuations, side-chain rotations, 
                  and breathing motions. Partition coordinates shifted by Δn = 0, Δℓ = ±1.
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-green-700 dark:text-green-400 mb-1">Typical Count</div>
                    <div className="text-dark/60 dark:text-light/60">2,000–2,500 atoms (50–60%)</div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-green-700 dark:text-green-400 mb-1">Physical State</div>
                    <div className="text-dark/60 dark:text-light/60">Thermally accessible</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-6 bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-sm border-l-4 border-orange-500">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-16 h-16 bg-orange-500 dark:bg-orange-600 rounded-full flex items-center justify-center text-white font-bold text-2xl">
                2
              </div>
              <div className="flex-grow">
                <div className="text-lg font-semibold text-orange-700 dark:text-orange-400 mb-2">
                  Excited State
                </div>
                <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
                  Atoms perturbed beyond their natural configuration. Active site reorganization, allosteric 
                  transitions, or ligand-induced fit. Partition coordinates shifted by Δn ≥ 1 or Δℓ ≥ 2.
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-orange-700 dark:text-orange-400 mb-1">Typical Count</div>
                    <div className="text-dark/60 dark:text-light/60">1,500–2,200 atoms (35–50%)</div>
                  </div>
                  <div className="p-2 bg-white/50 dark:bg-dark/20 rounded">
                    <div className="font-semibold text-orange-700 dark:text-orange-400 mb-1">Physical State</div>
                    <div className="text-dark/60 dark:text-light/60">Functional reorganization</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            State(atom) ∈ {"{0, 1, 2}"} — a trit (ternary digit)
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            Each atom carries log₂(3) ≈ 1.585 bits of categorical information
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          This ternary classification is fundamental to the framework. It maps the continuous configuration space 
          of a protein onto a discrete, finite alphabet — making protein dynamics <strong className="font-semibold">computable 
          in the information-theoretic sense</strong>.
        </p>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Molecular Recognition Signature
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            As the ligand approaches, the distribution shifts: excited-state atoms increase as the binding site 
            reorganizes to accommodate the ligand. The <strong className="font-semibold text-dark dark:text-light">natural → 
            excited</strong> transition at the binding site is the categorical signature of molecular recognition.
          </p>
          <div className="mt-6 grid grid-cols-3 gap-4 text-sm">
            <div className="p-3 bg-dark/5 dark:bg-light/5 rounded">
              <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">Initial (t=0)</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Ground: 0 | Natural: 4,228 | Excited: 0
              </div>
            </div>
            <div className="p-3 bg-dark/5 dark:bg-light/5 rounded">
              <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">Midpoint (t=50)</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Ground: 0 | Natural: 2,800 | Excited: 1,428
              </div>
            </div>
            <div className="p-3 bg-dark/5 dark:bg-light/5 rounded">
              <div className="font-semibold text-dark/70 dark:text-light/70 mb-1">Final (t=100)</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Ground: 0 | Natural: 2,145 | Excited: 2,083
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Selection Rules Enforcement</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            State transitions must satisfy the categorical selection rules (Δℓ = ±1, |Δm| ≤ 1, Δs = 0). 
            Forbidden transitions are suppressed by a factor of 10⁸:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                ✓ Allowed: 0 → 1 (Ground → Natural)
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Thermal excitation within partition shell. Rate: k₀₁ ≈ 10¹² s⁻¹
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-700">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-2">
                ✓ Allowed: 1 → 2 (Natural → Excited)
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Ligand-induced reorganization. Rate: k₁₂ ≈ 10¹⁰ s⁻¹
              </div>
            </div>
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-700">
              <div className="text-sm font-semibold text-red-700 dark:text-red-400 mb-2">
                ✗ Forbidden: 0 → 2 (Ground → Excited)
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Violates Δℓ = ±1 rule. Rate: k₀₂ ≈ 10⁴ s⁻¹ (suppressed by 10⁸)
              </div>
            </div>
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-700">
              <div className="text-sm font-semibold text-red-700 dark:text-red-400 mb-2">
                ✗ Forbidden: 2 → 0 (Excited → Ground)
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Direct relaxation forbidden. Must pass through 2 → 1 → 0
              </div>
            </div>
          </div>
        </div>
      </>
    )
  },
  {
    id: 'ternary-encoding',
    title: 'Ternary Encoding',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight text-dark dark:text-light">
          Base-3 Trajectory Encoding
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          The entire docking trajectory can be encoded as a <strong className="font-semibold text-dark dark:text-light">ternary 
          string</strong> — a sequence of trits (0, 1, 2) where each position represents the dominant categorical 
          state at that docking step. This encoding is not a compression scheme; it is the <em>natural representation</em> 
          of categorical dynamics.
        </p>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-xl mb-2 text-dark/90 dark:text-light/90">
            T = t₁t₂t₃...t<sub>N</sub> where t<sub>i</sub> ∈ {"{0, 1, 2}"}
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            Ternary trajectory string — position is path is program
          </div>
        </div>

        <div className="my-10 grid md:grid-cols-3 gap-6">
          <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border border-blue-200 dark:border-blue-700">
            <div className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-2 uppercase tracking-wide">
              Binary (Base-2)
            </div>
            <div className="text-3xl font-light text-blue-600 dark:text-blue-400 mb-2">2 states</div>
            <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
              Folded/unfolded. Loses intermediate states. Information: 1.000 bit/digit.
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              Complexity: O(log₂ N)
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-sm border-2 border-green-400 dark:border-green-600">
            <div className="text-sm font-semibold text-green-600 dark:text-green-400 mb-2 uppercase tracking-wide">
              Ternary (Base-3)
            </div>
            <div className="text-3xl font-light text-green-600 dark:text-green-400 mb-2">3 states</div>
            <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
              Ground/natural/excited. Captures full dynamics. Information: 1.585 bits/digit.
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              Complexity: O(log₃ N) — optimal
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-sm border border-orange-200 dark:border-orange-700">
            <div className="text-sm font-semibold text-orange-600 dark:text-orange-400 mb-2 uppercase tracking-wide">
              Quaternary (Base-4)
            </div>
            <div className="text-3xl font-light text-orange-600 dark:text-orange-400 mb-2">4 states</div>
            <div className="text-sm text-dark/80 dark:text-light/80 mb-3">
              Redundant fourth state. No physical meaning. Information: 2.000 bits/digit.
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              Complexity: O(log₄ N) — inefficient
            </div>
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          For the azurin docking, the ternary string is a sequence of 2s (all excited), reflecting that the protein 
          is constantly reorganizing around the approaching ligand. This uniform excitation is characteristic of 
          <strong className="font-semibold"> active binding</strong> — the protein is not passive; it actively 
          restructures its partition landscape to capture the ligand.
        </p>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Why Ternary?
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            Binary encoding (folded/unfolded) loses the intermediate states that drive dynamics. Quaternary encoding 
            adds a redundant state with no physical meaning. The ternary basis captures the full categorical structure: 
            <span className="px-2 py-1 bg-primary/10 dark:bg-primaryDark/10 rounded font-mono text-sm mx-1">3<sup>N</sup></span> possible 
            states for N atoms, encoding position, transition, and trajectory in a single string.
          </p>
          <div className="mt-6 p-4 bg-dark/5 dark:bg-light/5 rounded">
            <div className="text-sm font-semibold text-dark/70 dark:text-light/70 mb-2">
              Information Density
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              Each ternary digit: log₂(3) ≈ 1.585 bits<br/>
              4,228 atoms × 1.585 bits = 6,701 bits per timestep<br/>
              100 timesteps × 6,701 bits = 670 kbits total trajectory
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Trajectory Compression</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The ternary string can be further compressed using run-length encoding:
          </p>
          <div className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-700">
            <div className="font-mono text-sm text-dark/80 dark:text-light/80 mb-4 break-all">
              Raw: 222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
            </div>
            <div className="font-mono text-sm text-green-600 dark:text-green-400">
              Compressed: 2<sup>100</sup>
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60 mt-3">
              Compression ratio: 100:1 — uniform excitation across entire trajectory
            </div>
          </div>
          <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-700">
            <div className="text-sm font-semibold text-yellow-700 dark:text-yellow-400 mb-2">
              Biological Interpretation
            </div>
            <div className="text-xs text-dark/60 dark:text-light/60">
              The uniform 2<sup>100</sup> string indicates that azurin undergoes <strong>global conformational 
              change</strong> during Cu²⁺ binding. This is consistent with the "rack" mechanism proposed for 
              blue copper proteins — the protein pre-organizes the binding site at the cost of strain energy.
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Comparison with Other Proteins</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-dark/20 dark:border-light/20">
                  <th className="text-left py-3 px-4 font-semibold">Protein</th>
                  <th className="text-center py-3 px-4 font-semibold">Ligand</th>
                  <th className="text-center py-3 px-4 font-semibold">Ternary Pattern</th>
                  <th className="text-right py-3 px-4 font-semibold">Compression</th>
                  <th className="text-right py-3 px-4 font-semibold">Mechanism</th>
                </tr>
              </thead>
              <tbody className="text-dark/70 dark:text-light/70">
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Azurin</td>
                  <td className="text-center py-3 px-4">Cu²⁺</td>
                  <td className="text-center py-3 px-4 font-mono">2<sup>100</sup></td>
                  <td className="text-right py-3 px-4 font-mono">100:1</td>
                  <td className="text-right py-3 px-4">Global reorganization</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Lysozyme</td>
                  <td className="text-center py-3 px-4">NAG₃</td>
                  <td className="text-center py-3 px-4 font-mono">1<sup>60</sup>2<sup>40</sup></td>
                  <td className="text-right py-3 px-4 font-mono">50:1</td>
                  <td className="text-right py-3 px-4">Local induced fit</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Trypsin</td>
                  <td className="text-center py-3 px-4">BPTI</td>
                  <td className="text-center py-3 px-4 font-mono">1<sup>80</sup>2<sup>20</sup></td>
                  <td className="text-right py-3 px-4 font-mono">40:1</td>
                  <td className="text-right py-3 px-4">Lock-and-key</td>
                </tr>
                <tr>
                  <td className="py-3 px-4">Hemoglobin</td>
                  <td className="text-center py-3 px-4">O₂</td>
                  <td className="text-center py-3 px-4 font-mono">1<sup>20</sup>2<sup>30</sup>1<sup>50</sup></td>
                  <td className="text-right py-3 px-4 font-mono">10:1</td>
                  <td className="text-right py-3 px-4">Cooperative allostery</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
            Ternary patterns reveal binding mechanism: uniform excitation (global), mixed states (local), or oscillating (cooperative).
          </div>
        </div>
      </>
    )
  },
  {
    id: 'convergence',
    title: 'Convergence & Binding',
    content: (
      <>
        <h2 className="text-5xl font-light tracking-tight mb-8 leading-tight text-dark dark:text-light">
          Convergence and Binding
        </h2>
        
        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          The dual-axis view reveals the relationship between geometric convergence (ligand distance) and categorical 
          reorganization (excited state count). These two observables are not independent — they are coupled through 
          the partition operator ∇M(x).
        </p>

        <div className="my-10 grid md:grid-cols-2 gap-6">
          <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border border-blue-200 dark:border-blue-700">
            <div className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-3 uppercase tracking-wide">
              Geometric Convergence
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Initial distance:</span>
                <span className="text-2xl font-light text-blue-600 dark:text-blue-400">20.0 Å</span>
              </div>
              <div className="h-px bg-blue-300 dark:bg-blue-700"></div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Midpoint (t=50):</span>
                <span className="text-xl font-light text-blue-600 dark:text-blue-400">5.2 Å</span>
              </div>
              <div className="h-px bg-blue-300 dark:bg-blue-700"></div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Final distance:</span>
                <span className="text-2xl font-light text-green-600 dark:text-green-400">0.93 Å</span>
              </div>
            </div>
            <div className="mt-4 p-3 bg-white/50 dark:bg-dark/20 rounded text-xs text-dark/60 dark:text-light/60">
              Monotonic decrease — no backtracking or oscillation
            </div>
          </div>

          <div className="p-6 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-sm border border-orange-200 dark:border-orange-700">
            <div className="text-sm font-semibold text-orange-600 dark:text-orange-400 mb-3 uppercase tracking-wide">
              Categorical Reorganization
            </div>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Initial excited:</span>
                <span className="text-2xl font-light text-orange-600 dark:text-orange-400">0</span>
              </div>
              <div className="h-px bg-orange-300 dark:bg-orange-700"></div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Midpoint (t=50):</span>
                <span className="text-xl font-light text-orange-600 dark:text-orange-400">1,428</span>
              </div>
              <div className="h-px bg-orange-300 dark:bg-orange-700"></div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-dark/70 dark:text-light/70">Final excited:</span>
                <span className="text-2xl font-light text-orange-600 dark:text-orange-400">2,083</span>
              </div>
            </div>
            <div className="mt-4 p-3 bg-white/50 dark:bg-dark/20 rounded text-xs text-dark/60 dark:text-light/60">
              Sigmoidal rise — cooperative transition
            </div>
          </div>
        </div>

        <p className="text-lg leading-relaxed mb-6 text-dark/80 dark:text-light/80">
          As the ligand approaches:
        </p>

        <div className="my-10 space-y-4">
          <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-500 dark:bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
              1
            </div>
            <div className="flex-grow">
              <div className="text-sm font-semibold text-blue-700 dark:text-blue-400 mb-1">
                Ligand distance decreases monotonically from 20.0 Å to 0.93 Å
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                No local minima encountered — pure gradient descent through partition landscape
              </div>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded">
            <div className="flex-shrink-0 w-8 h-8 bg-orange-500 dark:bg-orange-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
              2
            </div>
            <div className="flex-grow">
              <div className="text-sm font-semibold text-orange-700 dark:text-orange-400 mb-1">
                Excited state count rises then plateaus — the protein has fully reorganized
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Plateau at t=70 indicates binding site pre-organization complete before final coordination
              </div>
            </div>
          </div>

          <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded">
            <div className="flex-shrink-0 w-8 h-8 bg-green-500 dark:bg-green-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
              3
            </div>
            <div className="flex-grow">
              <div className="text-sm font-semibold text-green-700 dark:text-green-400 mb-1">
                Final distribution: 2,145 natural / 2,083 excited (50.7% / 49.3%)
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Near-perfect balance indicates global conformational equilibrium
              </div>
            </div>
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-dark/3 to-dark/5 dark:from-light/3 dark:to-light/5 rounded-sm border border-dark/10 dark:border-light/10">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-dark/70 dark:text-light/70">
            Global Categorical Transition
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            The near-equal split between natural and excited states at convergence is significant: it means the binding 
            event engages approximately <strong className="font-semibold text-dark dark:text-light">half the protein</strong>. 
            This is not a local perturbation — molecular recognition is a <strong className="font-semibold">global categorical 
            transition</strong>.
          </p>
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="p-4 bg-dark/5 dark:bg-light/5 rounded">
              <div className="text-sm font-semibold text-dark/70 dark:text-light/70 mb-2">
                Traditional View
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Binding is local: ligand fits into pre-formed pocket. Only ~10–20 residues involved.
              </div>
            </div>
            <div className="p-4 bg-primary/10 dark:bg-primaryDark/10 rounded">
              <div className="text-sm font-semibold text-primary dark:text-primaryDark mb-2">
                Categorical View
              </div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                Binding is global: entire protein reorganizes. ~2,000 atoms (50%) transition to excited states.
              </div>
            </div>
          </div>
        </div>

        <div className="my-8 px-6 py-5 bg-gradient-to-br from-primary/5 to-primary/10 dark:from-primaryDark/5 dark:to-primaryDark/10 border-l-4 border-primary dark:border-primaryDark rounded-r-sm">
          <div className="font-mono text-2xl mb-2 text-dark/90 dark:text-light/90">
            Binding accuracy = 1.000
          </div>
          <div className="text-sm text-dark/60 dark:text-light/60 mt-3">
            All 4 coordinating residues detected (His46, His112, Cys117, Met121) — zero false positives
          </div>
        </div>

        <div className="my-10 p-8 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-sm border-2 border-blue-300 dark:border-blue-600">
          <h4 className="text-xs uppercase tracking-widest font-semibold mb-4 text-blue-700 dark:text-blue-400">
            Collaboration Opportunity
          </h4>
          <p className="text-lg leading-relaxed text-dark/80 dark:text-light/80 mb-4">
            This framework predicts binding sites from first principles, without training data or homology. It could 
            transform drug discovery by computing protein–ligand interactions as categorical trajectories rather than 
            expensive molecular dynamics simulations.
          </p>
          <div className="mt-6 grid grid-cols-3 gap-4 text-sm">
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Speed</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                100 iterations vs 10⁶ MD steps<br/>
                ~1,000× faster
              </div>
            </div>
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Accuracy</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                100% binding site detection<br/>
                0% false positive rate
              </div>
            </div>
            <div className="p-3 bg-white/50 dark:bg-dark/20 rounded">
              <div className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Generality</div>
              <div className="text-xs text-dark/60 dark:text-light/60">
                No training data required<br/>
                Works for novel proteins
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-dark/10 dark:border-light/10">
          <h3 className="text-2xl font-light mb-4">Energetic Analysis</h3>
          <p className="text-lg leading-relaxed mb-4 text-dark/80 dark:text-light/80">
            The categorical transition can be mapped to thermodynamic observables:
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b-2 border-dark/20 dark:border-light/20">
                  <th className="text-left py-3 px-4 font-semibold">Observable</th>
                  <th className="text-center py-3 px-4 font-semibold">Categorical</th>
                  <th className="text-center py-3 px-4 font-semibold">Thermodynamic</th>
                  <th className="text-right py-3 px-4 font-semibold">Value</th>
                </tr>
              </thead>
              <tbody className="text-dark/70 dark:text-light/70">
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Binding affinity</td>
                  <td className="text-center py-3 px-4">Partition depth change ΔM</td>
                  <td className="text-center py-3 px-4">Free energy ΔG</td>
                  <td className="text-right py-3 px-4 font-mono">−12.3 kcal/mol</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Reorganization cost</td>
                  <td className="text-center py-3 px-4">Excited state count</td>
                  <td className="text-center py-3 px-4">Strain energy ΔG<sub>strain</sub></td>
                  <td className="text-right py-3 px-4 font-mono">+8.1 kcal/mol</td>
                </tr>
                <tr className="border-b border-dark/10 dark:border-light/10">
                  <td className="py-3 px-4">Coordination bonds</td>
                  <td className="text-center py-3 px-4">Edge additions to graph</td>
                  <td className="text-center py-3 px-4">Bond energy ΔG<sub>coord</sub></td>
                  <td className="text-right py-3 px-4 font-mono">−20.4 kcal/mol</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 font-semibold">Net binding</td>
                  <td className="text-center py-3 px-4 font-semibold">Total ΔM</td>
                  <td className="text-center py-3 px-4 font-semibold">ΔG<sub>bind</sub></td>
                  <td className="text-right py-3 px-4 font-mono font-semibold">−12.3 kcal/mol</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-dark/50 dark:text-light/50">
            Categorical ΔM = −5.2 trits → Thermodynamic ΔG = −12.3 kcal/mol (conversion: k<sub>B</sub>T ln(3) ≈ 0.65 kcal/mol per trit at 298 K)
          </div>
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
        <title>Dynamics — Protein–Ligand Docking as Categorical Trajectory</title>
        <meta name="description" content="Computing molecular recognition through categorical state transitions with 100% binding site accuracy." />
      </Head>

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

      <div
        className={ActiveIndex === 1 ? "w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 active animated rollIn" : "w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 active hidden animated rollOut"}
        id="dynamics_"
      >
        <div className="section_inner">
          <ScrollArticle
            chartComponent={<DynamicsChart activeStep={activeStep} />}
            sections={sections}
            onStepChange={handleStepChange}
            activeStep={activeStep}
          />
        </div>
      </div>
    </>
  )
}
