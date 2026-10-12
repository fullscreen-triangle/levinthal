// Charts for the C–H activation page. Numbers from
// publications/catalytic-cycle/ch-activation-rebound/validation and
// figures/generate_panels.py; see the ledger on the page for their status.

import * as d3 from "d3";
import { C, T, axis, label, useChart } from "../core";
import { Svg, W, H, caption, fmtSci, logFmt } from "./base";

// ── 1. Three bodies: C, H and the ferryl oxygen ───────────────────────────
// Bond lengths at the transition state from the QM/MM work the paper cites.
const GEOM = { CH0: 1.09, CHts: 1.3, OHts: 1.2, OH1: 0.97 };

export function HatGeometry({ step, progress }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const s = 150; // px per Å
    const yb = 300, xC = 170;
    // C and O fixed; C···O distance at the TS = C–H + O–H (collinear, 180° for drawing)
    const xO = xC + (GEOM.CHts + GEOM.OHts) * s;
    const g = svg.append("g");
    g.append("line").attr("x1", xO).attr("x2", xO + 110).attr("y1", yb).attr("y2", yb).attr("stroke", C.ink2).attr("stroke-width", 6);
    g.append("text").attr("x", xO + 130).attr("y", yb + 6).attr("fill", C.ink).style("font-size", "20px").text("Fe");
    g.append("text").attr("x", xO + 62).attr("y", yb - 18).attr("fill", C.muted).style("font-size", "13px").text("ferryl");
    const ch = g.append("line").attr("y1", yb).attr("y2", yb).attr("stroke", C.ink).attr("stroke-width", 5);
    const oh = g.append("line").attr("y1", yb).attr("y2", yb).attr("stroke", C.aqua).attr("stroke-width", 5);
    g.append("circle").attr("cx", xC).attr("cy", yb).attr("r", 30).attr("fill", "#55534c");
    g.append("text").attr("x", xC).attr("y", yb + 7).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "20px").text("C");
    g.append("circle").attr("cx", xO).attr("cy", yb).attr("r", 26).attr("fill", C.red);
    g.append("text").attr("x", xO).attr("y", yb + 7).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "19px").style("font-weight", 600).text("O");
    const hAt = g.append("circle").attr("cy", yb).attr("r", 15).attr("fill", C.ink2);
    const hT = g.append("text").attr("y", yb + 5).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "15px").style("font-weight", 600).text("H");
    const lCH = g.append("text").attr("y", yb + 62).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "15px");
    const lOH = g.append("text").attr("y", yb + 62).attr("text-anchor", "middle").attr("fill", C.aqua).style("font-size", "15px");
    const rad = g.append("text").attr("x", xC - 20).attr("y", yb - 44).attr("fill", C.magenta).style("font-size", "17px").text("R•");
    // bond-order bar
    const bx = d3.scaleLinear().domain([0, 1]).range([170, 590]);
    const by = 470;
    svg.append("text").attr("x", 170).attr("y", by - 16).attr("fill", C.muted).style("font-size", "13px").text("bond order:  C–H  ↔  O–H");
    svg.append("rect").attr("x", bx(0)).attr("y", by).attr("width", bx(1) - bx(0)).attr("height", 12).attr("rx", 6).attr("fill", C.panel);
    const bfill = svg.append("rect").attr("x", bx(0)).attr("y", by).attr("height", 12).attr("rx", 6).attr("fill", C.aqua);
    return { set, s, xC, xO, ch, oh, hAt, hT, lCH, lOH, rad, bx, bfill };
  }, (h, step, p) => {
    // reaction coordinate: scroll progress, clamped to the active step's window
    const win = [[0, 0.2], [0.35, 0.6], [0.8, 1]][step];
    const q = Math.min(win[1], Math.max(win[0], p));
    // H position: C–H grows 1.09 → 1.30 at q = 0.5, then O–H shrinks 1.20 → 0.97
    const dCO = (h.xO - h.xC) / h.s;
    const xH = q <= 0.5
      ? h.xC + (GEOM.CH0 + (GEOM.CHts - GEOM.CH0) * (q / 0.5)) * h.s
      : h.xO - (GEOM.OHts - (GEOM.OHts - GEOM.OH1) * ((q - 0.5) / 0.5)) * h.s;
    const dCH = (xH - h.xC) / h.s, dOH = (h.xO - xH) / h.s;
    h.hAt.attr("cx", xH); h.hT.attr("x", xH);
    h.ch.attr("x1", h.xC).attr("x2", xH).attr("opacity", q < 0.75 ? 1 - Math.max(0, q - 0.4) * 1.6 : 0.08);
    h.oh.attr("x1", xH).attr("x2", h.xO).attr("opacity", q > 0.3 ? Math.min(1, (q - 0.3) * 2.5) : 0);
    h.lCH.attr("x", (h.xC + xH) / 2).text(`C–H ${dCH.toFixed(2)} Å`).attr("opacity", q < 0.85 ? 1 : 0.35);
    h.lOH.attr("x", (xH + h.xO) / 2).attr("y", 300 - 46).text(`O–H ${dOH.toFixed(2)} Å`).attr("opacity", q > 0.3 ? 1 : 0);
    h.rad.attr("opacity", q > 0.7 ? 1 : 0);
    h.bfill.attr("width", h.bx(q) - h.bx(0));
    void dCO;
    h.set(...(step === 0 ? ["Before: an intact C–H bond", "Compound I's oxygen faces the substrate carbon"]
      : step === 1 ? ["Transition state: H shared by C and O", "C–H 1.30 Å, O–H 1.20 Å, from the QM/MM work the paper cites"]
      : ["After: a carbon radical and Fe–OH", "the O–H bond is formed; the radical waits for rebound"]));
  }, step, progress);
  return <Svg svgRef={ref} />;
}

// ── 2. Five substrate classes on one depth scale ──────────────────────────
const CLASSES = [
  { name: "aliphatic", dm: 0.65, primary: true },
  { name: "benzylic", dm: 0.5, primary: true },
  { name: "allylic", dm: 0.45, primary: true },
  { name: "aromatic", dm: 0.38, primary: false },
  { name: "epoxidation", dm: 0.35, primary: false },
].map((c) => ({ ...c, k: 1e10 * Math.exp(-c.dm), ea: (65 * c.dm) / 4.184 }));

export function ClassDepths({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 140, r: 60, t: 100, b: 70 };
    const set = caption(svg);
    const y = d3.scaleBand().domain(CLASSES.map((c) => c.name)).range([m.t, H - m.b]).padding(0.3);
    const xDm = d3.scaleLinear().domain([0, 0.7]).range([m.l, W - m.r]);
    const xK = d3.scaleLog().domain([1e9, 1e10]).range([m.l, W - m.r]);
    const gx = svg.append("g").attr("transform", `translate(0,${H - m.b})`);
    const xl = label(svg, "", { x: (m.l + W - m.r) / 2, y: H - 22 });
    svg.append("g").selectAll("text").data(CLASSES).join("text").attr("x", m.l - 14).attr("y", (c) => y(c.name) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "15px").text((c) => c.name);
    const bars = svg.append("g").selectAll("rect").data(CLASSES).join("rect")
      .attr("x", m.l).attr("y", (c) => y(c.name)).attr("height", y.bandwidth()).attr("fill", C.yellow).attr("rx", 3);
    const vals = svg.append("g").selectAll("text").data(CLASSES).join("text")
      .attr("y", (c) => y(c.name) + y.bandwidth() / 2 + 5).attr("fill", C.ink).style("font-size", "13px");
    const kie = svg.append("g").selectAll("text").data(CLASSES).join("text").attr("x", W - m.r + 6)
      .attr("y", (c) => y(c.name) + y.bandwidth() / 2 + 5).attr("fill", (c) => (c.primary ? C.aqua : C.muted)).style("font-size", "12px")
      .attr("text-anchor", "end").attr("opacity", 0).text((c) => (c.primary ? "H moves: primary KIE" : "no H in the coordinate"));
    return { set, bars, vals, kie, gx, xl, xDm, xK, m };
  }, (h, step) => {
    const useK = step >= 1;
    const x = useK ? h.xK : h.xDm;
    h.gx.transition().duration(T).call(useK ? d3.axisBottom(h.xK).tickValues([1e9, 2e9, 5e9, 1e10]).tickFormat((v) => fmtSci(v, 0)) : d3.axisBottom(h.xDm).ticks(7)).on("end", function () { axis(d3.select(this)); });
    axis(h.gx);
    h.xl.text(useK ? "rate k = 10¹⁰·e^(−ΔM) s⁻¹ (log scale)" : "activation depth ΔM (set per class)");
    h.bars.transition().duration(T).attr("width", (c) => x(useK ? c.k : c.dm) - h.m.l)
      .attr("fill", (c) => (step === 2 ? (c.primary ? C.aqua : C.panel) : C.yellow));
    h.vals.transition().duration(T).attr("x", (c) => x(useK ? c.k : c.dm) + 8)
      .text((c) => (useK ? `${fmtSci(c.k, 2)} s⁻¹` : `ΔM ${c.dm.toFixed(2)} · E_a ${c.ea.toFixed(1)} kcal/mol`));
    h.kie.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.vals.transition().duration(T).attr("opacity", step === 2 ? 0 : 1);
    h.set(...(step === 0 ? ["One number per substrate class", "ΔM is assigned per class; E_a = 65 kJ/mol × ΔM"]
      : step === 1 ? ["On a rate scale the classes nearly coincide", "5.2 to 7.0 ×10⁹ s⁻¹: a factor of 1.35 across all five"]
      : ["What does separate them: whether H moves", "a primary isotope effect only where H is in the reaction coordinate"]));
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 3. The kinetic isotope effect ──────────────────────────────────────────
const KB = 1.380649e-23, HC = 6.62607015e-34 * 2.99792458e10;
const DZPE = (HC * 3000 * (1 - 1 / Math.SQRT2)) / 2; // J
const KAPPA = 1.253;
const kie = (T, kappa = 1) => Math.exp(DZPE / (KB * T)) * kappa;

export function KieChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 80, r: 40, t: 90, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLinear().domain([250, 400]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 17]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(8)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "temperature (K)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "k_H / k_D", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const band = svg.append("g").attr("opacity", 0);
    band.append("rect").attr("x", m.l).attr("width", W - m.r - m.l).attr("y", y(11)).attr("height", y(4) - y(11)).attr("fill", C.violet).attr("opacity", 0.18);
    band.append("text").attr("x", m.l + 8).attr("y", y(4) - 8).attr("fill", C.violet).style("font-size", "13px").text("4–11, the range the paper cites for P450s");
    const Ts = d3.range(250, 400.1, 2);
    const zpe = svg.append("path").attr("d", d3.line()(Ts.map((t) => [x(t), y(kie(t))]))).attr("fill", "none").attr("stroke", C.blue).attr("stroke-width", 2.5).attr("stroke-dasharray", "6 5");
    const tot = svg.append("path").attr("d", d3.line()(Ts.map((t) => [x(t), y(kie(t, KAPPA))]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3.5).attr("opacity", 0);
    const v310 = svg.append("line").attr("x1", x(310)).attr("x2", x(310)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.muted).attr("stroke-dasharray", "2 4");
    svg.append("text").attr("x", x(310) + 6).attr("y", H - m.b - 8).attr("fill", C.muted).style("font-size", "12px").text("310 K");
    const pZ = svg.append("g");
    pZ.append("circle").attr("cx", x(310)).attr("cy", y(kie(310))).attr("r", 6).attr("fill", C.blue);
    const lx = W - m.r - 8, ly = m.t + 14;
    pZ.append("text").attr("x", lx).attr("y", ly).attr("text-anchor", "end").attr("fill", C.blue).style("font-size", "13px").text(`- - zero-point energy only: ${kie(310).toFixed(2)} at 310 K`);
    const pT = svg.append("g").attr("opacity", 0);
    pT.append("circle").attr("cx", x(310)).attr("cy", y(kie(310, KAPPA))).attr("r", 7).attr("fill", C.yellow);
    pT.append("text").attr("x", lx).attr("y", ly + 22).attr("text-anchor", "end").attr("fill", C.yellow).style("font-size", "13px").text(`— with tunnelling κ = ${KAPPA}: ${kie(310, KAPPA).toFixed(2)} at 310 K`);
    const pS = svg.append("g").attr("opacity", 0);
    const t72 = d3.range(300, 400, 0.5).find((t) => kie(t, KAPPA) <= 7.2);
    pS.append("circle").attr("cx", x(310)).attr("cy", y(7.2)).attr("r", 7).attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 2);
    pS.append("text").attr("x", lx).attr("y", ly + 44).attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "13px").text("○ 7.2, stated in the text, at 310 K");
    pS.append("circle").attr("cx", x(t72)).attr("cy", y(7.2)).attr("r", 5).attr("fill", C.yellow);
    pS.append("text").attr("x", x(t72)).attr("y", y(7.2) + 26).attr("text-anchor", "middle").attr("fill", C.yellow).style("font-size", "13px").text(`model reaches 7.2 at ${Math.round(t72)} K`);
    return { set, band, zpe, tot, pZ, pT, pS, v310 };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Isotope effect from the zero-point energy", "C–H at 3000 cm⁻¹, C–D at 3000/√2: ΔZPE = 1.26 kcal/mol"]
      : step === 1 ? ["Add the tunnelling factor and compare", "κ_H/κ_D = 1.253 in the script; the result falls inside 4–11"]
      : ["The quoted 7.2 is not what the code returns", "at 310 K the script gives 9.63; the text's κ = 1.16 gives 8.91"]));
    h.band.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.tot.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.pT.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.pS.transition().duration(T).attr("opacity", step >= 2 ? 1 : 0);
    h.pZ.transition().duration(T).attr("opacity", step === 1 ? 0.35 : 1);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 4. Rebound against radical escape ─────────────────────────────────────
const K_HAT = 1e10 * Math.exp(-0.65), K_REB = 1e10 * Math.exp(-0.3);
const retained = (kesc) => K_REB / (K_REB + kesc);

export function ReboundChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 80, r: 40, t: 90, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLog().domain([10 ** 7.5, 10 ** 10.5]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 100]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickFormat(logFmt)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "rate at which the radical escapes or rotates, k_esc (s⁻¹)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "configuration retained (%)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const band = svg.append("g");
    band.append("rect").attr("x", m.l).attr("width", W - m.r - m.l).attr("y", y(90)).attr("height", y(40) - y(90)).attr("fill", C.violet).attr("opacity", 0.16);
    band.append("text").attr("x", m.l + 8).attr("y", y(40) - 8).attr("fill", C.violet).style("font-size", "13px").text("40–90 % retention, the range the paper cites");
    const ks = d3.range(7.5, 10.51, 0.02).map((e) => 10 ** e);
    svg.append("path").attr("d", d3.line()(ks.map((k) => [x(k), y(100 * retained(k))]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3.5);
    const rates = svg.append("g");
    rates.append("text").attr("x", W - m.r).attr("y", m.t - 4).attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "13px")
      .text(`k_HAT = ${fmtSci(K_HAT, 2)} s⁻¹   k_rebound = ${fmtSci(K_REB, 2)} s⁻¹`);
    const win = svg.append("g").attr("opacity", 0);
    win.append("rect").attr("x", x(1e8)).attr("width", x(1e9) - x(1e8)).attr("y", m.t).attr("height", H - m.b - m.t).attr("fill", C.aqua).attr("opacity", 0.12);
    win.append("text").attr("x", x(3.2e8)).attr("y", y(70)).attr("text-anchor", "middle").attr("fill", C.aqua).style("font-size", "13px").text("k_esc 10⁸–10⁹ (paper's text)");
    win.append("text").attr("x", x(3.2e8)).attr("y", y(70) + 18).attr("text-anchor", "middle").attr("fill", C.aqua).style("font-size", "13px")
      .text(`→ ${(100 * retained(1e9)).toFixed(0)}–${(100 * retained(1e8)).toFixed(0)} % retained`);
    const k40 = (K_REB * 0.6) / 0.4;
    const p40 = svg.append("g").attr("opacity", 0);
    p40.append("circle").attr("cx", x(k40)).attr("cy", y(40)).attr("r", 7).attr("fill", C.orange);
    p40.append("text").attr("x", x(k40) - 10).attr("y", y(40) + 24).attr("text-anchor", "end").attr("fill", C.orange).style("font-size", "13px")
      .text(`40 % needs k_esc = ${fmtSci(k40, 1)} s⁻¹`);
    return { set, win, p40 };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Rebound competes with the radical moving away", "retained = k_rebound / (k_rebound + k_esc)"]
      : step === 1 ? ["With the escape rates the text proposes", "retention stays near 90 % or above, not across 40–90 %"]
      : ["Reaching the low end takes a faster escape", "the script sets k_esc = 1.1×10¹⁰ s⁻¹ for one class to get there"]));
    h.win.transition().duration(T).attr("opacity", step === 1 ? 1 : 0.15);
    h.p40.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 5. Testosterone: where the 6β fraction comes from ─────────────────────
const SITES = [
  { pos: "6β", dm: 0.55, g: 1.0 },
  { pos: "16β", dm: 0.62, g: 0.5 },
  { pos: "2β", dm: 0.68, g: 0.4 },
  { pos: "15β", dm: 0.8, g: 0.3 },
];
const frac = (gs) => {
  const k = SITES.map((s, i) => gs[i] * Math.exp(-s.dm));
  const t = d3.sum(k);
  return k.map((v) => v / t);
};
const F_SET = frac(SITES.map((s) => s.g));
const F_EQ = frac(SITES.map(() => 1));
const f6 = (g) => frac([g, ...SITES.slice(1).map((s) => s.g)])[0];

export function TestosteroneChart({ step }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const m = { l: 90, r: 40 };
    const top = 100, bh = 200;
    const x = d3.scaleBand().domain(SITES.map((s) => s.pos)).range([m.l, W - m.r]).padding(0.35);
    const y = d3.scaleLinear().domain([0, 0.75]).range([top + bh, top]);
    axis(svg.append("g").attr("transform", `translate(0,${top + bh})`).call(d3.axisBottom(x)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3, "%")), { grid: W - m.r - m.l, orient: "y" });
    const lit = svg.append("rect").attr("x", x("6β") - 8).attr("width", x.bandwidth() + 16).attr("y", y(0.7)).attr("height", y(0.5) - y(0.7))
      .attr("fill", C.violet).attr("opacity", 0.22);
    svg.append("text").attr("x", x("6β") + x.bandwidth() + 14).attr("y", y(0.6) + 4).attr("fill", C.violet).style("font-size", "12px").text("50–70 % reported (6β)");
    const bars = svg.append("g").selectAll("rect").data(SITES).join("rect").attr("x", (s) => x(s.pos)).attr("width", x.bandwidth()).attr("rx", 3);
    const vals = svg.append("g").selectAll("text").data(SITES).join("text").attr("x", (s) => x(s.pos) + x.bandwidth() / 2).attr("text-anchor", "middle")
      .attr("fill", C.ink).style("font-size", "13px");
    const gl = svg.append("g").selectAll("text").data(SITES).join("text").attr("x", (s) => x(s.pos) + x.bandwidth() / 2).attr("y", top + bh + 42)
      .attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "12px");
    // sweep
    const g2 = svg.append("g").attr("opacity", 0);
    const t2 = 400, b2 = 510;
    const sx = d3.scaleLinear().domain([0.5, 2.5]).range([m.l, W - m.r]);
    const sy = d3.scaleLinear().domain([0.25, 0.75]).range([b2, t2]);
    axis(g2.append("g").attr("transform", `translate(0,${b2})`).call(d3.axisBottom(sx).ticks(8)));
    axis(g2.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(sy).ticks(3, "%")));
    label(g2, "accessibility factor chosen for 6β, g₆β", { x: (m.l + W - m.r) / 2, y: b2 + 38 });
    g2.append("rect").attr("x", m.l).attr("width", W - m.r - m.l).attr("y", sy(0.7)).attr("height", sy(0.5) - sy(0.7)).attr("fill", C.violet).attr("opacity", 0.18);
    g2.append("path").attr("d", d3.line()(d3.range(0.5, 2.51, 0.02).map((g) => [sx(g), sy(f6(g))]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3);
    const g50 = d3.range(0.5, 3, 0.005).find((g) => f6(g) >= 0.5), g70 = d3.range(0.5, 3, 0.005).find((g) => f6(g) >= 0.7);
    g2.append("text").attr("x", sx(g50)).attr("y", sy(0.5) + 18).attr("fill", C.violet).style("font-size", "12px")
      .text(`50 % needs g = ${g50.toFixed(2)}; 70 % needs ${g70 ? g70.toFixed(2) : ">2.5"}`);
    g2.append("circle").attr("cx", sx(1)).attr("cy", sy(f6(1))).attr("r", 5).attr("fill", C.yellow);
    return { set, bars, vals, gl, y, top, bh, g2, lit };
  }, (h, step) => {
    const f = step === 1 ? F_EQ : F_SET;
    h.bars.transition().duration(T).attr("y", (s, i) => h.y(f[i])).attr("height", (s, i) => h.top + h.bh - h.y(f[i]))
      .attr("fill", (s) => (s.pos === "6β" ? C.yellow : "#7a6a3a"));
    h.vals.transition().duration(T).attr("y", (s, i) => h.y(f[i]) - 8).text((s, i) => `${(100 * f[i]).toFixed(1)} %`);
    h.gl.text((s) => (step === 1 ? "g = 1" : `ΔM ${s.dm} · g ${s.g}`));
    h.g2.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.set(...(step === 0 ? ["CYP3A4 on testosterone: four competing positions", "k = 10¹⁰ · g · e^(−ΔM), with an accessibility factor g per position"]
      : step === 1 ? ["Without the accessibility factors", "the depths alone give 6β 27.9 %"]
      : ["The match comes from the factor chosen for 6β", "the 49 % follows from g₆β = 1.0 against 0.3–0.5 for the others"]));
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 6. The paper's validation table, read for what it compares ────────────
const SCORE = [
  { q: "ΔM_HAT", rel: 0.0, kind: "own" },
  { q: "k_HAT", rel: 0.044, kind: "own" },
  { q: "k_rebound", rel: 0.001, kind: "own" },
  { q: "k_rebound / k_HAT", rel: 0.014, kind: "own" },
  { q: "activation energy", rel: 0.01, kind: "lit" },
  { q: "isotope effect", rel: 0.238, kind: "lit" },
  { q: "6β fraction", rel: 0.183, kind: "lit" },
  { q: "stereoretention", rel: 0.354, kind: "lit" },
];

export function ScoreChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 190, r: 50, t: 100, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLinear().domain([0, 0.4]).range([m.l, W - m.r]);
    const y = d3.scaleBand().domain(SCORE.map((s) => s.q)).range([m.t, H - m.b]).padding(0.3);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4, "%")), { grid: H - m.b - m.t });
    label(svg, "relative difference from the paper's reference value", { x: (m.l + W - m.r) / 2, y: H - 22 });
    [[0.2, "20 %"], [0.3, "30 %"]].forEach(([v, t]) => {
      svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", m.t - 10).attr("y2", H - m.b).attr("stroke", C.red).attr("stroke-dasharray", "4 4");
      svg.append("text").attr("x", x(v)).attr("y", m.t - 16).attr("text-anchor", "middle").attr("fill", C.red).style("font-size", "12px").text(t);
    });
    svg.append("g").selectAll("text").data(SCORE).join("text").attr("x", m.l - 14).attr("y", (s) => y(s.q) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "14px").text((s) => s.q);
    const bars = svg.append("g").selectAll("rect").data(SCORE).join("rect").attr("x", m.l).attr("y", (s) => y(s.q)).attr("height", y.bandwidth())
      .attr("width", (s) => Math.max(2, x(s.rel) - m.l)).attr("rx", 3);
    const vals = svg.append("g").selectAll("text").data(SCORE).join("text").attr("x", (s) => Math.max(x(s.rel), m.l + 2) + 8)
      .attr("y", (s) => y(s.q) + y.bandwidth() / 2 + 5).attr("fill", C.ink).style("font-size", "12px")
      .text((s) => `${(100 * s.rel).toFixed(1)} %${s.kind === "own" ? "  · against the model's own value" : ""}`);
    return { set, bars, vals };
  }, (h, step) => {
    h.bars.transition().duration(T)
      .attr("fill", (s) => (step === 0 ? C.yellow : step === 1 ? (s.kind === "own" ? C.muted : C.panel) : s.kind === "lit" ? (s.rel > 0.3 ? C.red : C.violet) : C.panel))
      .attr("opacity", (s) => (step === 1 && s.kind === "lit" ? 0.35 : step === 2 && s.kind === "own" ? 0.35 : 1));
    h.vals.transition().duration(T).attr("opacity", (s) => (step === 1 && s.kind === "lit" ? 0.35 : step === 2 && s.kind === "own" ? 0.35 : 1));
    h.set(...(step === 0 ? ["The paper's eight checks", "reported as 8/8; the script's own 30 % rule gives 7/8"]
      : step === 1 ? ["Four compare the model with itself", "their reference values are the model's outputs or targets"]
      : ["Four compare with literature", "the reference is a value inside each reported range"]));
  }, step);
  return <Svg svgRef={ref} />;
}
