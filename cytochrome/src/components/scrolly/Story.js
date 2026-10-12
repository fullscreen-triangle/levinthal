// Scroll-told paper page. A page is a hero, a sequence of scenes and a ledger.
// In a scene the chart is pinned (CSS sticky) while its steps scroll past; the
// step nearest the reading line becomes active and the chart re-highlights.

import Head from "next/head";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import TransitionEffect from "@/components/TransitionEffect";
import { C, LEDGER, gsap, ScrollTrigger, useSmoothScroll } from "./core";

export function Tag({ k, children }) {
  const t = LEDGER[k];
  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-[1px] text-[11px] align-middle"
      style={{ borderColor: t.color, color: C.ink2 }}
      title={t.label}
    >
      <span className="inline-block h-[7px] w-[7px] rounded-full" style={{ background: t.color }} />
      {children || t.label}
    </span>
  );
}

export function Legend({ keys = ["D", "S", "C", "M", "L"] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {keys.map((k) => <Tag key={k} k={k} />)}
    </div>
  );
}

// Reading line: where a step must reach to become active.
const readingLine = () => (window.innerWidth < 1024 ? "top 78%" : "top 58%");

export function Scene({ n, title, chart: Chart, steps, scrub = false, chartProps = {} }) {
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);
  const root = useRef(null);
  const stepRefs = useRef([]);

  useEffect(() => {
    const triggers = stepRefs.current.filter(Boolean).map((el, i) =>
      ScrollTrigger.create({
        trigger: el,
        start: readingLine,
        end: () => (window.innerWidth < 1024 ? "bottom 78%" : "bottom 58%"),
        onToggle: (self) => { if (self.isActive) setActive(i); },
      })
    );
    if (scrub) {
      triggers.push(ScrollTrigger.create({
        trigger: root.current, start: "top 40%", end: "bottom 60%",
        onUpdate: (self) => setProgress(Math.round(self.progress * 400) / 400),
      }));
    }
    const fades = stepRefs.current.filter(Boolean).map((el) =>
      gsap.from(el.querySelector(".step-fade"), {
        y: 40, opacity: 0, duration: 0.8, ease: "power2.out",
        scrollTrigger: { trigger: el, start: "top 88%", toggleActions: "play none none reverse" },
      })
    );
    return () => {
      triggers.forEach((t) => t.kill());
      fades.forEach((f) => { f.scrollTrigger && f.scrollTrigger.kill(); f.kill(); });
    };
  }, [steps.length, scrub]);

  return (
    <section ref={root} className="relative mx-auto max-w-[1500px] px-16 lg:px-6">
      <div className="pt-24 pb-6 lg:pt-16">
        <div className="text-[12px] uppercase tracking-[0.2em]" style={{ color: C.muted }}>
          {String(n).padStart(2, "0")}
        </div>
        <h2 className="mt-2 text-[34px] font-semibold leading-tight tracking-tight lg:text-[26px]" style={{ color: C.ink }}>
          {title}
        </h2>
      </div>
      <div className="flex gap-14 lg:flex-col-reverse lg:gap-0">
        <div className="w-[38%] shrink-0 lg:w-full">
          {steps.map((s, i) => (
            <div key={i} ref={(el) => (stepRefs.current[i] = el)} className="flex min-h-[78vh] items-center py-8 lg:min-h-[64vh] lg:items-start">
              <div className="step-fade w-full">
              <article
                className="rounded-xl border p-6 transition-[opacity,border-color] duration-500 lg:bg-[#1b1b1b]/95"
                style={{ borderColor: i === active ? C.ink2 + "55" : C.line, opacity: i === active ? 1 : 0.42, background: C.surface }}
              >
                {s.tags && <div className="mb-3 flex flex-wrap gap-2">{s.tags.map((t) => <Tag key={t} k={t} />)}</div>}
                {s.title && <h3 className="mb-2 text-[20px] font-semibold leading-snug" style={{ color: C.ink }}>{s.title}</h3>}
                <div className="space-y-3 text-[16px] leading-relaxed" style={{ color: C.ink2 }}>{s.body}</div>
              </article>
              </div>
            </div>
          ))}
        </div>
        <div className="min-w-0 flex-1 lg:sticky lg:top-0 lg:z-10 lg:h-[50vh] lg:pt-2 lg:bg-[#1b1b1b]">
          <div className="sticky top-[12vh] h-[76vh] lg:static lg:h-full">
            <div className="flex h-full w-full items-center justify-center rounded-2xl border p-4" style={{ borderColor: C.line, background: C.bg }}>
              <Chart step={active} progress={progress} {...chartProps} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// A full-width pause between scenes, for a 3-D model or a short statement.
export function Interlude({ title, children, note }) {
  const root = useRef(null);
  useEffect(() => {
    const tw = gsap.from(root.current, {
      opacity: 0, y: 60, duration: 1, ease: "power2.out",
      scrollTrigger: { trigger: root.current, start: "top 85%", toggleActions: "play none none reverse" },
    });
    return () => { tw.scrollTrigger && tw.scrollTrigger.kill(); tw.kill(); };
  }, []);
  return (
    <section ref={root} className="mx-auto max-w-[1500px] px-16 pt-28 lg:px-6">
      {title && <h2 className="mb-5 text-[28px] font-semibold tracking-tight" style={{ color: C.ink }}>{title}</h2>}
      {children}
      {note && <p className="mt-4 max-w-[980px] text-[15px] leading-relaxed" style={{ color: C.ink2 }}>{note}</p>}
    </section>
  );
}

// The whole page: hero, scenes (children), ledger, neighbours.
export default function StoryPage({ meta, kicker, title, lead, children, ledger, prev, next, sources }) {
  useSmoothScroll();
  const bar = useRef(null);

  useEffect(() => {
    const st = gsap.to(bar.current, {
      scaleX: 1, ease: "none",
      scrollTrigger: { trigger: document.documentElement, start: "top top", end: "bottom bottom", scrub: 0.3 },
    });
    const hero = gsap.from(".story-hero > *", { y: 30, opacity: 0, duration: 0.9, stagger: 0.12, ease: "power2.out", delay: 0.5 });
    return () => { st.scrollTrigger && st.scrollTrigger.kill(); st.kill(); hero.kill(); };
  }, []);

  return (
    <>
      <Head>
        <title>{meta.title} · cytochrome</title>
        <meta name="description" content={meta.description} />
      </Head>
      <TransitionEffect />
      <div ref={bar} className="fixed left-0 right-0 top-0 z-40 h-[3px] origin-left scale-x-0" style={{ background: C.aqua }} />
      <div className="w-full" style={{ background: "#1b1b1b" }}>
        <header className="story-hero mx-auto max-w-[1500px] px-16 pt-20 pb-10 lg:px-6 lg:pt-10">
          <div className="text-[12px] uppercase tracking-[0.2em]" style={{ color: C.muted }}>{kicker}</div>
          <h1 className="mt-3 max-w-[1100px] text-[56px] font-semibold leading-[1.05] tracking-tight xl:text-[46px] md:text-[34px]" style={{ color: C.ink }}>
            {title}
          </h1>
          <div className="mt-6 max-w-[920px] space-y-4 text-[19px] leading-relaxed md:text-[17px]" style={{ color: C.ink2 }}>{lead}</div>
          <div className="mt-8 flex flex-wrap items-center gap-3 text-[13px]" style={{ color: C.muted }}>
            <span>Every number on this page carries its status:</span>
            <Legend />
          </div>
          <div className="mt-14 text-[12px] uppercase tracking-[0.2em]" style={{ color: C.muted }}>scroll ↓</div>
        </header>

        {children}

        {ledger && (
          <section className="mx-auto max-w-[1500px] px-16 pt-28 pb-12 lg:px-6">
            <h2 className="text-[30px] font-semibold tracking-tight" style={{ color: C.ink }}>What each number is</h2>
            <p className="mt-2 max-w-[900px] text-[16px]" style={{ color: C.ink2 }}>
              A model value agreeing with a literature value is a consistency check of the model, not a measurement.
              Values checked in this work come from public structures and sequences; see the derivation manuscript.
            </p>
            <div className="mt-6 overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse text-[15px]">
                <thead>
                  <tr style={{ color: C.muted }} className="text-left text-[12px] uppercase tracking-[0.12em]">
                    <th className="py-2 pr-4 font-normal">quantity</th>
                    <th className="py-2 pr-4 font-normal">value</th>
                    <th className="py-2 pr-4 font-normal">compared with</th>
                    <th className="py-2 font-normal">status</th>
                  </tr>
                </thead>
                <tbody>
                  {ledger.map((r, i) => (
                    <tr key={i} className="border-t" style={{ borderColor: C.line }}>
                      <td className="py-3 pr-4" style={{ color: C.ink }}>{r[0]}</td>
                      <td className="py-3 pr-4 tabular-nums" style={{ color: C.ink2 }}>{r[1]}</td>
                      <td className="py-3 pr-4" style={{ color: C.ink2 }}>{r[2]}</td>
                      <td className="py-3"><Tag k={r[3]} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {sources && <div className="mt-6 text-[13px] leading-relaxed" style={{ color: C.muted }}>{sources}</div>}
          </section>
        )}

        <nav className="mx-auto flex max-w-[1500px] items-stretch justify-between gap-4 px-16 pt-10 pb-24 lg:px-6 md:flex-col">
          {prev ? (
            <Link href={prev.href} className="flex-1 rounded-xl border p-4 transition-colors hover:border-[#58E6D9]" style={{ borderColor: C.line }}>
              <div className="mb-1 text-[11px] uppercase tracking-widest" style={{ color: C.muted }}>← previous</div>
              <div className="text-[15px] font-semibold" style={{ color: C.ink }}>{prev.label}</div>
            </Link>
          ) : <div className="flex-1" />}
          {next ? (
            <Link href={next.href} className="flex-1 rounded-xl border p-4 text-right transition-colors hover:border-[#58E6D9]" style={{ borderColor: C.line }}>
              <div className="mb-1 text-[11px] uppercase tracking-widest" style={{ color: C.muted }}>next →</div>
              <div className="text-[15px] font-semibold" style={{ color: C.ink }}>{next.label}</div>
            </Link>
          ) : <div className="flex-1" />}
        </nav>
      </div>
    </>
  );
}
