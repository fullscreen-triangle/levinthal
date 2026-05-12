import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import Layout from "./Layout";
import TransitionEffect from "./TransitionEffect";
import PanelGrid from "./PanelGrid";
import { PAPERS } from "@/data/papers";
import AnimatedText from "./AnimatedText";

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

const PaperPage = ({ paper, children, glb }) => {
  const isHeadline = !!paper.headline_paper;

  // Find prev/next paper
  const idx = PAPERS.findIndex((p) => p.id === paper.id);
  const prev = idx > 0 ? PAPERS[idx - 1] : null;
  const next = idx < PAPERS.length - 1 ? PAPERS[idx + 1] : null;

  return (
    <>
      <Head>
        <title>{paper.short} · cytochrome</title>
        <meta name="description" content={paper.title} />
      </Head>
      <TransitionEffect />
      <Layout className="!pt-12 md:!pt-12 sm:!pt-8">

        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-[11px] uppercase tracking-widest
              text-dark/40 dark:text-light/40">
              {paper.part}
            </span>
            <span className="text-[11px] text-dark/40 dark:text-light/40">·</span>
            <span className="text-[11px] uppercase tracking-widest
              text-dark/40 dark:text-light/40">
              Paper {paper.id}
            </span>
            {isHeadline && (
              <span className="rounded-full bg-primaryDark px-2 py-0.5
                text-[9px] font-bold uppercase tracking-widest text-dark">
                headline
              </span>
            )}
          </div>

          <AnimatedText
            text={paper.title}
            className="!text-4xl xl:!text-3xl lg:!text-3xl md:!text-2xl sm:!text-xl !text-left mb-3"
          />

          <p className="text-base text-dark/70 dark:text-light/65
            max-w-4xl whitespace-pre-line">
            {paper.abstract}
          </p>
        </div>

        {/* Headline numbers */}
        <div className="mb-12">
          <h2 className="text-xs uppercase tracking-widest text-dark/40 dark:text-light/40 mb-4">
            Headline numbers
          </h2>
          <div className="grid grid-cols-3 md:grid-cols-1 gap-4">
            {paper.headline.map((h, i) => (
              <div key={i} className="rounded-xl border border-dark/10
                dark:border-light/10 p-4 bg-light/30 dark:bg-light/5">
                <div className="text-xs text-dark/55 dark:text-light/50 mb-1">
                  {h.label}
                </div>
                <div className="text-xl font-bold text-primaryDark mb-1">
                  {h.computed}
                </div>
                <div className="text-[11px] text-dark/40 dark:text-light/40 font-mono">
                  vs {h.target}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-[11px] text-dark/45 dark:text-light/40 italic">
            {paper.status}
          </div>
        </div>

        {/* GLB structure viewer */}
        {glb && (
          <div className="mb-12 not-prose">
            <h2 className="text-xs uppercase tracking-widest
              text-dark/40 dark:text-light/40 mb-4">
              Structure
            </h2>
            <GLBViewer {...glb} />
          </div>
        )}

        {/* Page-specific body */}
        {children && (
          <div className="mb-12 max-w-3xl text-dark dark:text-light paper-body">
            {children}
          </div>
        )}

        {/* Panels */}
        <div className="mb-12">
          <h2 className="text-xs uppercase tracking-widest text-dark/40 dark:text-light/40 mb-4">
            Panels ({paper.panels.length})
          </h2>
          <PanelGrid panels={paper.panels} panelDir={paper.panelDir} columns={2} />
        </div>

        {/* Prev / Next */}
        <div className="flex items-stretch justify-between gap-4 pt-8 border-t
          border-dark/10 dark:border-light/10 md:flex-col">
          {prev ? (
            <Link href={prev.href} className="flex-1 rounded-xl border
              border-dark/10 dark:border-light/10 p-4
              hover:border-primary dark:hover:border-primaryDark transition-colors">
              <div className="text-[10px] uppercase tracking-widest
                text-dark/40 dark:text-light/40 mb-1">
                ← prev · Paper {prev.id}
              </div>
              <div className="text-sm font-semibold">{prev.short}</div>
            </Link>
          ) : <div className="flex-1" />}
          {next ? (
            <Link href={next.href} className="flex-1 rounded-xl border
              border-dark/10 dark:border-light/10 p-4 text-right
              hover:border-primary dark:hover:border-primaryDark transition-colors">
              <div className="text-[10px] uppercase tracking-widest
                text-dark/40 dark:text-light/40 mb-1">
                next · Paper {next.id} →
              </div>
              <div className="text-sm font-semibold">{next.short}</div>
            </Link>
          ) : <div className="flex-1" />}
        </div>

      </Layout>
    </>
  );
};

export default PaperPage;
