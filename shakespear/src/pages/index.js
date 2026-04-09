import Head from "next/head";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

const ProteusViewer = dynamic(
  () => import("@/components/proteus/ProteusViewer"),
  { ssr: false }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>shakespear | Protein Observation Instrument</title>
        <meta
          name="description"
          content="Real-time protein observation through GPU fragment shaders. GLB models + PDB coordinates + partition calculus. No backend. The shader IS the instrument."
        />
      </Head>
      <TransitionEffect />
      <main className="flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="!pt-8">
          <div className="mb-6">
            <h1 className="text-5xl font-bold text-dark dark:text-light tracking-tight md:text-4xl sm:text-3xl">
              shakespear
            </h1>
            <p className="text-sm text-dark/50 dark:text-light/40 mt-1 tracking-wider">
              Spectroscopic Harmonic Analysis of Kinetic Entropy in
              Structural Protein Evaluation And Resonance
            </p>
          </div>
          <ProteusViewer />
        </Layout>
      </main>
    </>
  );
}
