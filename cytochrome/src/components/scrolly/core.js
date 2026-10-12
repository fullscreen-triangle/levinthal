// Shared machinery for the scroll-told paper pages: Lenis smooth scrolling
// driving GSAP ScrollTrigger (the pattern of pages/highlight.js), and a small
// hook for d3 charts that are drawn once and then re-highlighted per step.

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";
import Lenis from "@studio-freight/lenis";

if (typeof window !== "undefined") gsap.registerPlugin(ScrollTrigger);

export { gsap, ScrollTrigger };

// Colours: the talk's dark palette, so the site and the talk read alike.
export { C, LEDGER } from "@/components/talk/theme";

// One Lenis instance per page. Skipped for visitors who ask for reduced motion.
export function useSmoothScroll() {
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let lenis = null;
    const tick = (time) => lenis && lenis.raf(time * 1000);
    if (!reduce) {
      lenis = new Lenis({ duration: 1.15, easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), smoothWheel: true });
      lenis.on("scroll", ScrollTrigger.update);
      gsap.ticker.add(tick);
      gsap.ticker.lagSmoothing(0);
    }
    // the page-transition overlay delays layout; measure again once it has run
    const t1 = setTimeout(() => ScrollTrigger.refresh(), 900);
    const onLoad = () => ScrollTrigger.refresh();
    window.addEventListener("load", onLoad);
    return () => {
      clearTimeout(t1);
      window.removeEventListener("load", onLoad);
      gsap.ticker.remove(tick);
      if (lenis) lenis.destroy();
    };
  }, []);
}

// A chart drawn into a fixed viewBox. `build(svg)` runs once and returns a
// handle; `update(handle, step, progress)` runs on every step or progress
// change and should only move, fade or recolour what build() made.
export function useChart(build, update, step, progress = 0) {
  const ref = useRef(null);
  const handle = useRef(null);
  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    handle.current = build(svg) || {};
    update(handle.current, step, progress, true);
    return () => {
      svg.selectAll("*").interrupt();
      if (handle.current && handle.current.stop) handle.current.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (handle.current) update(handle.current, step, progress, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, progress]);
  return ref;
}

export const T = 650; // default transition length, ms

// Recessive axes for charts drawn at viewBox scale.
export function axis(g, { grid = 0, orient = "x", size = 13 } = {}) {
  g.selectAll("path.domain").attr("stroke", "#5a5955");
  g.selectAll(".tick line").attr("stroke", "#5a5955");
  g.selectAll(".tick text").attr("fill", "#c3c2b7").style("font-size", `${size}px`);
  if (grid) {
    g.selectAll(".tick line").clone().attr("stroke", "#2a2a27")
      .attr(orient === "x" ? "y2" : "x2", orient === "x" ? -grid : grid);
  }
  return g;
}

export function label(svg, text, { x, y, rotate = false, anchor = "middle", size = 13, color = "#c3c2b7" }) {
  return svg.append("text").attr("x", x).attr("y", y)
    .attr("transform", rotate ? `rotate(-90, ${x}, ${y})` : null)
    .attr("text-anchor", anchor).attr("fill", color).style("font-size", `${size}px`).text(text);
}

// Fade everything in `sel` to `dim` except elements for which `keep(d, i)` holds.
export function focus(sel, keep, { dim = 0.18, dur = T } = {}) {
  sel.transition().duration(dur).attr("opacity", (d, i) => (keep(d, i) ? 1 : dim));
}

// Superscript exponent formatting for log axes: 1e+6 → 10⁶.
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
export const pow10 = (v) => {
  const e = Math.round(Math.log10(v));
  return `10${String(e).split("").map((c) => SUP[c] || c).join("")}`;
};
