import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import { CHAIN_MARKERS } from "@/data/glbMarkers";

const GLBViewer = dynamic(
  () => import("@/components/GLBViewer"),
  {
    ssr: false,
    loading: () => (
      <div
        className="w-full rounded-xl bg-dark/40 border border-dark/10
          dark:border-light/10 flex items-center justify-center
          text-[11px] uppercase tracking-widest text-light/25"
        style={{ height: 320 }}
      >
        Loading structure…
      </div>
    ),
  }
);

const LAYERS = [
  {
    n: 5,
    name: "Hologram pipeline",
    role: "Electron-movement visualisations + Marcus λ readout",
    source: "superimposed-multi-modal-holograms.tex",
    output: "Six observables: K, Franck-Condon, Stokes, Huang-Rhys, λ, point group",
    detail: "Five-pass GPU shader: temporal separation → coherent superposition H(ω,t) → 2D FFT → categorical synthesis (n,ℓ,m,s) → (S_k,S_t,S_e) → trajectory completion. 1024×1024 texture, < 1 ms / draw cycle. The shader source lives at cytochrome/glb/levinthal_glb/shaders/electron_trajectory.frag; the numpy CPU prototype at shader_pipeline.py.",
    color: "#FFA500",
  },
  {
    n: 4,
    name: "Harmonic molecular resonator",
    role: "Cycle-rank cross-validation loops",
    source: "harmonic-molecular-resonator.tex",
    output: "Cycle rank C = |E| − |V| + 1 independent loops",
    detail: "Promotes the per-atom spectrometer of L1+2+3 into a per-molecule spectrometer. The cofactor cluster is treated as a graph G = (V, E) where edges are pairs of atomic modes whose frequencies stand in a low-order rational ratio ω_i/ω_j ≈ p/q with η = max(p,q) ≤ 10. Closed loops are virtual resonant cavities; each is an independent cross-validation channel for the same partition coordinate.",
    color: "#C44E52",
  },
  {
    n: 3,
    name: "Ensemble strobes",
    role: "Three temporal gates clocked by analyte emission",
    source: "ensemble-strobes.tex",
    output: "W_Sk (fs) | W_St (ns) | W_Se (μs+); cross-talk η < 3.7 × 10⁻³",
    detail: "Three temporal gates defined by the analyte's own absorption (τ_abs) and emission (τ_em) events. For the CPR cofactor cluster: τ_em ~ 1 ns, gate Δt ~ 10 ps. Recursive ternary depth d gives 3^d orthogonal sub-projections. Three internal self-validation channels (V_me, Ritz, triple-convertibility round-trip).",
    color: "#55A868",
  },
  {
    n: 2,
    name: "Triple-equivalence theorem",
    role: "Calibration certificate",
    source: "spectroscopic-derivation-of-elements.tex",
    output: "Admissibility gate at floor 𝔖_floor(R_bio) ≈ 3.7 × 10⁻⁴",
    detail: "The theorem 𝔒 ≡ 𝔆 ≡ 𝔓 (oscillatory ≡ categorical ≡ partition) licenses Layer 1: a bounded oscillator at partition depth n is the same mathematical object as an atomic mode at the same depth. Without this layer, the hardware oscillators of Layer 1 would just be hardware; with it, they are legitimate atomic-coordinate resolvers.",
    color: "#888888",
  },
  {
    n: 1,
    name: "Categorical spectrometer",
    role: "Hardware-oscillator readout of (n, ℓ, m, s)",
    source: "hyperfine-transitions.tex",
    output: "Four oscillator subsystems spanning 10 orders of magnitude in frequency",
    detail: "CPU clock 10⁹ Hz → n; memory bus 10⁸ Hz → ℓ; LED emission 10¹⁴ Hz → m; memory refresh 10⁴ Hz → s. Calibrated against NIST Atomic Spectra Database (mean fractional error 1.6 × 10⁻⁴ across 24 hydrogen lines; 21 cm hyperfine reproduced to 9 decimal places at 1420.40575 MHz).",
    color: "#4C72B0",
  },
];

export default function Apparatus() {
  return (
    <>
      <Head>
        <title>Apparatus · cytochrome</title>
        <meta
          name="description"
          content="The five-layer instrument stack used to observe electron transfer through CPR–CYP3A4. Hardware oscillators at the bottom; GPU hologram pipeline at the top. Apparatus, not simulation."
        />
      </Head>
      <TransitionEffect />
      <Layout className="!pt-12 md:!pt-12 sm:!pt-8">

        <div className="mb-12">
          <span className="text-[11px] uppercase tracking-widest
            text-dark/40 dark:text-light/40">
            The instrument
          </span>
          <AnimatedText
            text="Five layers of apparatus."
            className="!text-4xl xl:!text-3xl md:!text-2xl !text-left mt-2 mb-4"
          />
          <p className="text-base text-dark/75 dark:text-light/65 max-w-3xl">
            Every numerical result reported in this monograph is a physical
            observable from the stack below — not a computation against a
            chosen model Hamiltonian. The protein under observation
            (Papers 1–3) enters the apparatus at Layer 3; its absorption
            and emission times define the strobe windows.
          </p>
        </div>

        <div className="space-y-3 mb-12">
          {LAYERS.map((L) => (
            <div
              key={L.n}
              className="rounded-2xl border border-dark/10 dark:border-light/10
                bg-light/30 dark:bg-light/5 p-6"
            >
              <div className="flex items-baseline gap-4 flex-wrap">
                <div
                  className="text-3xl font-bold w-14 text-center"
                  style={{ color: L.color }}
                >
                  L{L.n}
                </div>
                <div className="flex-1 min-w-[200px]">
                  <h3 className="text-lg font-bold text-dark dark:text-light">
                    {L.name}
                  </h3>
                  <p className="text-sm text-dark/65 dark:text-light/55">
                    {L.role}
                  </p>
                </div>
                <code className="text-[11px] text-dark/45 dark:text-light/40
                  font-mono">
                  {L.source}
                </code>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-4 lg:grid-cols-1">
                <div className="lg:col-span-1">
                  <div className="text-[10px] uppercase tracking-widest
                    text-dark/40 dark:text-light/40 mb-1">
                    Output
                  </div>
                  <div className="text-xs font-mono text-primaryDark">
                    {L.output}
                  </div>
                </div>
                <div className="col-span-2 lg:col-span-1">
                  <div className="text-[10px] uppercase tracking-widest
                    text-dark/40 dark:text-light/40 mb-1">
                    Detail
                  </div>
                  <p className="text-xs text-dark/70 dark:text-light/60
                    leading-relaxed">
                    {L.detail}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* GLB analyte structure */}
        <div className="mb-12">
          <h2 className="text-xs uppercase tracking-widest
            text-dark/40 dark:text-light/40 mb-4">
            The analyte
          </h2>
          <GLBViewer
            height={320}
            badge="instrument analyte · CYP3A4"
            preset="chain"
            markers={CHAIN_MARKERS}
          />
          <p className="mt-3 text-[11px] text-dark/55 dark:text-light/50">
            The productive cytochrome P450 GLB. The four coloured markers
            are the cofactor centres (NADPH, FAD, FMN, heme) that the five-layer
            stack observes. The GLB is parsed by Paper&nbsp;2.5; the instrument
            reads its own analyte.
          </p>
        </div>

        {/* Apparatus stack visualisation */}
        <div className="mb-12">
          <h2 className="text-xs uppercase tracking-widest
            text-dark/40 dark:text-light/40 mb-4">
            The stack rendered
          </h2>
          <div className="rounded-xl overflow-hidden border border-dark/10
            dark:border-light/10 bg-white">
            <div className="relative aspect-[4/1]">
              <Image
                src="/panels/paper-4/panel_09_apparatus_stack.png"
                alt="Five-layer instrument stack"
                fill
                sizes="100vw"
                className="object-contain"
              />
            </div>
          </div>
          <p className="mt-3 text-[11px] text-dark/55 dark:text-light/50">
            Panel 9 of Paper 4: layer frequencies on log axis, strobe-window
            timescales, six-observable list, and 3D pyramidal stack view.
          </p>
        </div>

        {/* Why apparatus, not simulation */}
        <div className="mb-12 rounded-2xl bg-primaryDark/10 dark:bg-primaryDark/5
          border border-primaryDark/30 p-6">
          <h2 className="text-sm font-bold mb-3 text-primaryDark uppercase tracking-wider">
            The triple identity
          </h2>
          <p className="text-base text-dark/75 dark:text-light/70 mb-3">
            <strong>Measurement = Computation = Observation.</strong>
          </p>
          <p className="text-sm text-dark/70 dark:text-light/60 max-w-3xl">
            The fragment shader writing a pixel <em>is</em> performing a
            physical observation of a partition cell. The texture <em>is</em>
            the categorical state, not a picture of it. The protein is not
            a passive sample but an active participant: sample, instrument,
            computer, and result unified in completion-driven navigation
            through categorical state space. The apparatus stack realises
            this identity with O(1) GPU memory regardless of analyte size
            (the empty-dictionary principle).
          </p>
        </div>

        <div className="flex items-center justify-between
          pt-6 border-t border-dark/10 dark:border-light/10
          text-sm sm:flex-col sm:items-start sm:gap-3">
          <Link
            href="/transfer"
            className="text-primary dark:text-primaryDark font-bold uppercase
              tracking-wider hover:underline"
          >
            ← see the apparatus in action: Paper 4 (HEADLINE)
          </Link>
          <Link
            href="/glb-input"
            className="text-dark/60 dark:text-light/55 uppercase
              tracking-wider hover:underline"
          >
            analyte input via GLB · Paper 2.5 →
          </Link>
        </div>

      </Layout>
    </>
  );
}
