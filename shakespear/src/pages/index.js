import Head from "next/head";
import dynamic from "next/dynamic";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import Link from "next/link";

const AnimatedShakespeareBook = dynamic(
  () => import("@/components/AnimatedShakespeareBook").then(mod => mod.AnimatedShakespeareBook),
  { ssr: false }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>shakespear | Protein Observation Instrument</title>
        <meta
          name="description"
          content="Spectroscopic Harmonic Analysis of Kinetic Entropy in Structural Protein Evaluation And Resonance. Real-time protein observation through GPU fragment shaders."
        />
      </Head>
      <TransitionEffect />
      <article className="flex min-h-screen items-center text-dark dark:text-light sm:items-start">
        <Layout className="!pt-0 md:!pt-16 sm:!pt-16">
          <div className="flex w-full items-start justify-between md:flex-col">

            {/* Left: 3D Book */}
            <div className="w-1/2 lg:w-full">
              <AnimatedShakespeareBook className="h-[500px] md:h-80" />
            </div>

            {/* Right: Text */}
            <div className="flex w-1/2 flex-col items-center self-center lg:w-full lg:text-center">
              <AnimatedText
                text="shakespear"
                className="!text-left !text-7xl xl:!text-6xl lg:!text-center lg:!text-6xl md:!text-5xl sm:!text-3xl"
              />
              <p className="my-2 text-sm font-medium text-dark/50 dark:text-light/40 tracking-wider md:text-xs">
                Spectroscopic Harmonic Analysis of Kinetic Entropy in
                Structural Protein Evaluation And Resonance
              </p>
              <p className="my-4 text-base font-medium md:text-sm sm:!text-xs">
                A real-time protein observation instrument powered by GPU fragment shaders.
                Load a protein, observe its harmonic structure, detect virtual cavities,
                and predict activity — all in your browser, with no backend.
                The shader IS the instrument.
              </p>
              <div className="mt-2 flex items-center self-start gap-4 lg:self-center">
                <Link
                  href="/instrument"
                  className="flex items-center rounded-lg border-2 border-solid bg-dark p-2.5 px-6
                    text-lg font-semibold capitalize text-light hover:border-dark hover:bg-transparent
                    hover:text-dark dark:bg-light dark:text-dark dark:hover:border-light
                    dark:hover:bg-dark dark:hover:text-light md:p-2 md:px-4 md:text-base"
                >
                  Launch Instrument
                </Link>
                <Link
                  href="https://github.com/fullscreen-triangle/shakespear"
                  target="_blank"
                  className="text-lg font-medium capitalize text-dark underline
                    dark:text-light md:text-base"
                >
                  GitHub
                </Link>
              </div>
            </div>
          </div>
        </Layout>
      </article>
    </>
  );
}
