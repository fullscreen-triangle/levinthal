import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { C, STAGE } from "./theme";
import SLIDES from "./slides";

// Keyboard:  → / space / PageDown  next build or slide
//            ← / PageUp            previous
//            F  full screen   N  speaker notes   O  overview   Home / End
// URL hash #7 opens slide 7; ?all shows every build (for printing / review).

function parseHash() {
  if (typeof window === "undefined") return 0;
  const n = parseInt(window.location.hash.replace("#", ""), 10);
  return Number.isFinite(n) && n >= 1 && n <= SLIDES.length ? n - 1 : 0;
}

export default function Deck() {
  // client-only component, so the hash can be read before the first render
  const [index, setIndex] = useState(parseHash);
  const [step, setStep] = useState(0);
  const [notes, setNotes] = useState(false);
  const [overview, setOverview] = useState(false);
  const [scale, setScale] = useState(1);
  const [chrome, setChrome] = useState(true);
  const [showAll, setShowAll] = useState(
    () => typeof window !== "undefined" && new URLSearchParams(window.location.search).has("all"));
  const rootRef = useRef(null);
  const idleRef = useRef(null);

  const slide = SLIDES[index];
  const lastStep = slide.steps ?? 0;

  useEffect(() => {
    const onHash = () => { setIndex(parseHash()); setStep(0); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, []);

  useEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / STAGE.w, window.innerHeight / STAGE.h));
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#${index + 1}`);
  }, [index]);

  const go = useCallback((i, s = 0) => {
    const j = Math.max(0, Math.min(SLIDES.length - 1, i));
    setIndex(j);
    setStep(s === "end" ? SLIDES[j].steps ?? 0 : s);
  }, []);

  const next = useCallback(() => {
    if (step < lastStep) setStep(step + 1);
    else if (index < SLIDES.length - 1) go(index + 1, 0);
  }, [step, lastStep, index, go]);

  const prev = useCallback(() => {
    if (step > 0) setStep(step - 1);
    else if (index > 0) go(index - 1, "end");
  }, [step, index, go]);

  const toggleFull = useCallback(() => {
    if (!document.fullscreenElement) rootRef.current?.requestFullscreen?.();
    else document.exitFullscreen?.();
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target && ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
      const k = e.key;
      if (["ArrowRight", "PageDown", " "].includes(k)) { e.preventDefault(); next(); }
      else if (["ArrowLeft", "PageUp"].includes(k)) { e.preventDefault(); prev(); }
      else if (k === "Home") go(0);
      else if (k === "End") go(SLIDES.length - 1, "end");
      else if (k === "f" || k === "F") toggleFull();
      else if (k === "n" || k === "N") setNotes((v) => !v);
      else if (k === "o" || k === "O") setOverview((v) => !v);
      else if (k === "Escape") setOverview(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, go, toggleFull]);

  // hide the controls after a few seconds without pointer movement
  useEffect(() => {
    const wake = () => {
      setChrome(true);
      clearTimeout(idleRef.current);
      idleRef.current = setTimeout(() => setChrome(false), 2600);
    };
    wake();
    window.addEventListener("mousemove", wake);
    return () => { window.removeEventListener("mousemove", wake); clearTimeout(idleRef.current); };
  }, []);

  const Slide = slide.Component;
  const effectiveStep = showAll ? lastStep : step;
  const partColor = slide.part === "I" ? C.blue : C.orange;

  return (
    <div ref={rootRef} className="fixed inset-0 z-[200] overflow-hidden select-none" style={{ background: C.bg, fontFamily: "var(--font-mont), system-ui, sans-serif" }}>
      <div
        className="absolute left-1/2 top-1/2"
        style={{ width: STAGE.w, height: STAGE.h, transform: `translate(-50%, -50%) scale(${scale})`, transformOrigin: "center" }}
      >
        <div className="absolute inset-0 rounded-sm overflow-hidden" style={{ background: C.surface }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={slide.id}
              className="absolute inset-0"
              initial={showAll ? false : { opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -24 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
            >
              <Slide step={effectiveStep} />
            </motion.div>
          </AnimatePresence>

          {/* footer inside the stage: part, title, progress */}
          <div className="absolute left-0 right-0 bottom-0 h-[64px] px-[72px] flex items-center gap-6" style={{ color: C.muted }}>
            <span className="text-[14px] uppercase tracking-[0.16em]" style={{ color: partColor }}>
              {slide.part === "I" ? "I · Derivation" : "II · Applications"}
            </span>
            <div className="flex-1 flex gap-[3px]">
              {SLIDES.map((s, i) => (
                <button
                  key={s.id}
                  aria-label={`slide ${i + 1}`}
                  onClick={() => go(i)}
                  className="h-[5px] flex-1 rounded-full"
                  style={{ background: i <= index ? (s.part === "I" ? C.blue : C.orange) : C.line, opacity: i === index ? 1 : 0.7 }}
                />
              ))}
            </div>
            <span className="text-[14px] tabular-nums">
              {String(index + 1).padStart(2, "0")} / {SLIDES.length}
              {lastStep > 0 && (
                <span className="ml-3">
                  {Array.from({ length: lastStep + 1 }).map((_, i) => (
                    <span key={i} className="inline-block w-[6px] h-[6px] rounded-full ml-[3px]" style={{ background: i <= effectiveStep ? C.ink2 : C.line }} />
                  ))}
                </span>
              )}
            </span>
          </div>
        </div>
      </div>

      {/* controls outside the stage */}
      <div className={`absolute top-3 right-3 flex gap-2 transition-opacity duration-500 ${chrome ? "opacity-100" : "opacity-0"}`}>
        {[
          ["Overview (O)", () => setOverview((v) => !v), "▦"],
          ["Notes (N)", () => setNotes((v) => !v), "✎"],
          ["Full screen (F)", toggleFull, "⛶"],
        ].map(([t, f, g]) => (
          <button key={t} title={t} onClick={f} className="w-9 h-9 rounded-md text-[16px]" style={{ background: C.panel, color: C.ink2, border: `1px solid ${C.line}` }}>
            {g}
          </button>
        ))}
        <Link href="/" title="Leave the talk" className="w-9 h-9 rounded-md text-[16px] flex items-center justify-center" style={{ background: C.panel, color: C.ink2, border: `1px solid ${C.line}` }}>
          ✕
        </Link>
      </div>

      <div className={`absolute top-1/2 left-0 right-0 flex justify-between px-2 pointer-events-none transition-opacity duration-500 ${chrome ? "opacity-100" : "opacity-0"}`}>
        <button onClick={prev} aria-label="previous" className="pointer-events-auto w-10 h-16 rounded-md text-[22px]" style={{ background: "rgba(32,32,30,0.6)", color: C.ink2 }}>‹</button>
        <button onClick={next} aria-label="next" className="pointer-events-auto w-10 h-16 rounded-md text-[22px]" style={{ background: "rgba(32,32,30,0.6)", color: C.ink2 }}>›</button>
      </div>

      <AnimatePresence>
        {notes && (
          <motion.div
            initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 40, opacity: 0 }}
            className="absolute left-1/2 -translate-x-1/2 bottom-4 w-[min(1100px,92vw)] max-h-[38vh] overflow-auto rounded-lg px-6 py-4 text-[17px] leading-relaxed"
            style={{ background: "rgba(20,20,19,0.96)", border: `1px solid ${C.line}`, color: C.ink2 }}
          >
            <div className="text-[13px] uppercase tracking-[0.16em] mb-2" style={{ color: C.muted }}>
              Notes · slide {index + 1} · build {effectiveStep + 1}/{lastStep + 1}
            </div>
            {slide.notes}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {overview && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="absolute inset-0 overflow-auto p-10"
            style={{ background: "rgba(18,18,17,0.97)" }}
          >
            <div className="grid grid-cols-5 gap-4 max-w-[1500px] mx-auto">
              {SLIDES.map((s, i) => (
                <button
                  key={s.id}
                  onClick={() => { go(i); setOverview(false); }}
                  className="text-left rounded-lg p-4 h-[150px] flex flex-col"
                  style={{ background: C.panel, border: `1px solid ${i === index ? C.ink2 : C.line}` }}
                >
                  <span className="text-[12px] uppercase tracking-[0.14em]" style={{ color: s.part === "I" ? C.blue : C.orange }}>
                    {String(i + 1).padStart(2, "0")} · {s.part === "I" ? "Derivation" : "Applications"}
                  </span>
                  <span className="mt-2 text-[15px] leading-snug" style={{ color: C.ink }}>{s.short}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
