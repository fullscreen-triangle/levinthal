import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Layout from "@/components/Layout";
import AnimatedText from "@/components/AnimatedText";
import TransitionEffect from "@/components/TransitionEffect";
import { PAPERS, ALL_PANELS } from "@/data/papers";

export default function Gallery() {
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState(null);

  const filtered = filter === "all"
    ? ALL_PANELS
    : ALL_PANELS.filter((p) => p.paperId === filter);

  return (
    <>
      <Head>
        <title>Gallery · cytochrome</title>
        <meta
          name="description"
          content="All 52 figure panels from the cytochrome P450 monograph, browsable by paper."
        />
      </Head>
      <TransitionEffect />
      <Layout className="!pt-12 md:!pt-12 sm:!pt-8">

        <div className="mb-8">
          <span className="text-[11px] uppercase tracking-widest
            text-dark/40 dark:text-light/40">
            The gallery
          </span>
          <AnimatedText
            text={`${ALL_PANELS.length} panels across the monograph.`}
            className="!text-4xl xl:!text-3xl md:!text-2xl !text-left mt-2 mb-4"
          />
          <p className="text-sm text-dark/70 dark:text-light/60 max-w-3xl">
            Each paper has 8 panels (Paper 4: 12). All visualisations are
            generated from the validation suites — every panel sources its
            data from a passing validation script.
          </p>
        </div>

        {/* Filter chips */}
        <div className="mb-8 flex flex-wrap gap-2">
          <FilterChip
            active={filter === "all"}
            onClick={() => setFilter("all")}
            label={`All · ${ALL_PANELS.length}`}
          />
          {PAPERS.map((p) => (
            <FilterChip
              key={p.id}
              active={filter === p.id}
              onClick={() => setFilter(p.id)}
              label={`P${p.id} · ${p.short} (${p.panels.length})`}
              accent={p.headline_paper}
            />
          ))}
        </div>

        {/* Grid */}
        <div className="grid grid-cols-3 lg:grid-cols-2 sm:grid-cols-1 gap-4 mb-8">
          {filtered.map((panel, i) => (
            <motion.figure
              key={panel.src}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.4) }}
              whileHover={{ y: -2 }}
              className="cursor-pointer rounded-xl overflow-hidden border
                border-dark/10 dark:border-light/10 bg-white"
              onClick={() => setOpen(panel)}
            >
              <div className="relative aspect-[4/1.05] bg-white">
                <Image
                  src={panel.src}
                  alt={panel.caption}
                  fill
                  sizes="(max-width: 768px) 100vw, 33vw"
                  className="object-contain"
                />
              </div>
              <figcaption className="px-3 py-2 text-[11px]">
                <div className="flex items-center justify-between mb-0.5">
                  <Link
                    href={panel.paperHref}
                    className="text-[10px] uppercase tracking-widest
                      text-primaryDark hover:underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    P{panel.paperId} · {panel.paper}
                  </Link>
                  <span className="text-[10px] text-dark/40 dark:text-light/40 font-mono">
                    {String(panel.panelIndex).padStart(2, "0")}
                  </span>
                </div>
                <div className="text-dark/65 dark:text-light/55 line-clamp-2">
                  {panel.caption}
                </div>
              </figcaption>
            </motion.figure>
          ))}
        </div>

        {/* Lightbox */}
        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center
                bg-dark/80 backdrop-blur-sm p-8"
              onClick={() => setOpen(null)}
            >
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.9 }}
                className="relative max-w-7xl w-full"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="relative aspect-[4/1] bg-white rounded-lg overflow-hidden">
                  <Image
                    src={open.src}
                    alt={open.caption}
                    fill
                    sizes="100vw"
                    className="object-contain"
                  />
                </div>
                <div className="mt-3 text-light text-sm">
                  <Link href={open.paperHref}
                    className="text-primaryDark hover:underline mr-3">
                    Paper {open.paperId} · {open.paper}
                  </Link>
                  <span className="font-mono text-primaryDark mr-2">
                    Panel {String(open.panelIndex).padStart(2, "0")}
                  </span>
                  <span>{open.caption}</span>
                </div>
                <button
                  className="absolute -top-2 -right-2 w-8 h-8 rounded-full
                    bg-light text-dark text-lg font-bold"
                  onClick={() => setOpen(null)}
                  aria-label="Close"
                >
                  ×
                </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </Layout>
    </>
  );
}

const FilterChip = ({ active, onClick, label, accent = false }) => (
  <button
    onClick={onClick}
    className={`px-3 py-1.5 rounded-full text-[11px] uppercase tracking-wider
      transition-colors
      ${active
        ? "bg-primaryDark text-dark"
        : accent
        ? "border border-primaryDark text-primaryDark hover:bg-primaryDark/10"
        : "border border-dark/15 dark:border-light/15 text-dark/65 dark:text-light/55 hover:border-dark dark:hover:border-light"}`}
  >
    {label}
  </button>
);
