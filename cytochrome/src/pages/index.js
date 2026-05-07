import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import PaperCard from "@/components/PaperCard";
import { PAPERS } from "@/data/papers";

const ElectronTransferViewer = dynamic(
  () => import("@/components/ElectronTransferViewer"),
  { ssr: false, loading: () => (
      <div className="w-full rounded-xl border border-dark/10
        dark:border-light/10 bg-dark text-light/50 flex items-center
        justify-center text-xs uppercase tracking-widest"
        style={{ height: 520 }}>
        Loading 3D apparatus…
      </div>
    ),
  }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>cytochrome · Categorical Mechanics of P450</title>
        <meta
          name="description"
          content="The cytochrome P450 monograph: protein construction (Papers 1–3), GLB-grounded structural input (Paper 2.5), and headline observation of the NADPH→FAD→FMN→heme electron transfer chain (Paper 4) via a five-layer instrument stack."
        />
      </Head>
      <TransitionEffect />

      <article className="text-dark dark:text-light">

        {/* Hero */}
        <Layout className="!pt-8 md:!pt-12 sm:!pt-8">

          <div className="flex items-start gap-12 lg:flex-col">

            {/* Left: text */}
            <div className="flex-1 min-w-0">
              <span className="inline-block text-[11px] uppercase tracking-widest
                text-dark/45 dark:text-light/45 mb-3">
                A monograph · {PAPERS.length} papers complete
              </span>

              <AnimatedText
                text="Categorical Mechanics of Cytochrome P450."
                className="!text-5xl xl:!text-4xl lg:!text-4xl md:!text-3xl sm:!text-2xl
                  !text-left mb-4"
              />

              <p className="text-base text-dark/75 dark:text-light/70 max-w-2xl mb-3">
                Papers 1–3 construct the protein from first principles (the
                receiver R<sub>bio</sub>, the P450 sequence manifold, the
                CYP3A4 fold, the resting and substrate-bound states).
                Paper 2.5 grounds the construction in real PDB coordinates
                via GLB-based structural input.
              </p>
              <p className="text-base text-dark/75 dark:text-light/70 max-w-2xl mb-6">
                Paper 4 — the headline — <em>observes</em> the
                NADPH→FAD→FMN→heme electron transfer chain through the
                resulting protein, using a five-layer instrument stack
                that is hardware all the way down. Paper 5 covers the
                downstream chemistry (Compound I formation).
              </p>

              <div className="flex items-center gap-4 mb-8 sm:flex-col sm:items-start">
                <Link
                  href="/transfer"
                  className="rounded-lg bg-primaryDark px-5 py-2.5
                    text-sm font-bold uppercase tracking-wider text-dark
                    hover:bg-primary hover:text-light transition-colors"
                >
                  read the headline · paper 4 →
                </Link>
                <Link
                  href="/apparatus"
                  className="text-sm uppercase tracking-wider text-dark
                    dark:text-light underline underline-offset-4 hover:text-primaryDark"
                >
                  the instrument stack
                </Link>
              </div>

              <div className="grid grid-cols-3 gap-4 max-w-xl sm:grid-cols-1">
                <Stat label="Papers complete" value="6 / 14" />
                <Stat label="Validations" value="60 / 60" sub="all PASS" />
                <Stat label="Marcus λ" value="0.85 eV" sub="Layer 5 readout" />
              </div>
            </div>

            {/* Right: live 3D electron transfer viewer */}
            <div className="w-1/2 lg:w-full lg:max-w-3xl flex flex-col">
              <ElectronTransferViewer height={520} autoplay />
              <p className="mt-3 text-[11px] text-dark/55 dark:text-light/50
                leading-relaxed">
                <span className="text-primaryDark font-bold">LIVE</span> · the
                productive cytochrome P450 GLB rendered in real PDB coordinates;
                the white sphere is the electron centroid moving NADPH → FAD →
                FMN → heme under the same hop-rate kinetics that the Layer 5
                shader pipeline uses to produce the headline visualisations.
                Click and drag to rotate; scrub the timeline to control t.
              </p>
            </div>
          </div>
        </Layout>

        {/* Paper cards */}
        <Layout className="!pt-0">
          <div className="mb-6 flex items-baseline justify-between
            border-b border-dark/10 dark:border-light/10 pb-2">
            <h2 className="text-xs uppercase tracking-widest
              text-dark/45 dark:text-light/45">
              The monograph
            </h2>
            <Link href="/gallery" className="text-[11px] uppercase
              tracking-wider text-primaryDark hover:underline">
              all panels →
            </Link>
          </div>

          <div className="grid grid-cols-3 lg:grid-cols-2 sm:grid-cols-1 gap-4">
            {PAPERS.map((p) => (
              <PaperCard key={p.id} paper={p} />
            ))}
          </div>

          <div className="mt-12 rounded-2xl border border-dark/10
            dark:border-light/10 p-6 bg-light/30 dark:bg-light/5">
            <h3 className="text-sm font-bold mb-2">
              Roadmap
            </h3>
            <p className="text-sm text-dark/70 dark:text-light/65">
              Papers 6–14 cover C–H activation and rebound, heteroatom
              oxidation and dealkylation, the atypical reactions atlas,
              the 57 human isoforms as variants, polymorphisms and
              drug–drug interactions, membrane anchoring and partner
              coupling, the seven-state closed orbit, the complete
              spectroscopic atlas, and the database-wide validation
              benchmark. Status: planned.
            </p>
          </div>
        </Layout>
      </article>
    </>
  );
}

const Stat = ({ label, value, sub }) => (
  <div className="border-l-2 border-primaryDark pl-3">
    <div className="text-[10px] uppercase tracking-widest
      text-dark/45 dark:text-light/45 mb-1">
      {label}
    </div>
    <div className="text-xl font-bold text-dark dark:text-light">
      {value}
    </div>
    {sub && (
      <div className="text-[10px] text-dark/45 dark:text-light/45 italic">
        {sub}
      </div>
    )}
  </div>
);
