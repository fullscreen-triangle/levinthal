// Charts for the electron-transfer page. Numbers from
// publications/catalytic-cycle/multi-hop-et-chain (validation/scripts/_common.py,
// results/0N_*.json, tex) and glb/levinthal_glb/shader_pipeline.py.

import * as d3 from "d3";
import { C, T, axis, label, useChart } from "../core";
import { Svg, W, H, caption, fmtSci, logFmt } from "./base";

const COF = [
  { id: "NADPH", x: 0, c: C.violet, sigma: 1.4 },
  { id: "FAD", x: 4, c: C.yellow, sigma: 1.5 },
  { id: "FMN", x: 8, c: C.aqua, sigma: 1.7 },
  { id: "heme", x: 22, c: C.red, sigma: 2.6 },
];

// ── 1. Four cofactors, two proteins, one membrane ─────────────────────────
export function ChainDiagram({ step, progress }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const x = d3.scaleLinear().domain([-2, 28]).range([60, W - 40]);
    const yc = 250;
    // proteins
    const prot = svg.append("g");
    prot.append("rect").attr("x", x(-1.6)).attr("y", yc - 70).attr("width", x(10) - x(-1.6)).attr("height", 140).attr("rx", 50).attr("fill", "#22303f").attr("opacity", 0.8);
    prot.append("rect").attr("x", x(17.5)).attr("y", yc - 80).attr("width", x(27.5) - x(17.5)).attr("height", 160).attr("rx", 60).attr("fill", "#3a2526").attr("opacity", 0.85);
    prot.append("text").attr("x", x(4)).attr("y", yc - 82).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "14px").text("CPR (reductase)");
    prot.append("text").attr("x", x(22.5)).attr("y", yc - 92).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "14px").text("P450");
    // links
    const links = svg.append("g").selectAll("g").data(COF.slice(0, 3)).join("g");
    links.append("line").attr("x1", (d) => x(d.x)).attr("x2", (d, i) => x(COF[i + 1].x)).attr("y1", yc).attr("y2", yc)
      .attr("stroke", C.ink2).attr("stroke-width", 2).attr("stroke-dasharray", "5 5");
    const dl = links.append("text").attr("x", (d, i) => (x(d.x) + x(COF[i + 1].x)) / 2).attr("y", yc + 48).attr("text-anchor", "middle")
      .attr("fill", C.ink).style("font-size", "15px").text((d, i) => `${COF[i + 1].x - d.x} Å`);
    const nodes = svg.append("g").selectAll("g").data(COF).join("g").attr("transform", (d) => `translate(${x(d.x)},${yc})`);
    nodes.append("circle").attr("r", 22).attr("fill", (d) => d.c);
    nodes.append("text").attr("y", -32).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "15px").text((d) => d.id);
    // the other distance
    const alt = svg.append("g").attr("opacity", 0);
    alt.append("circle").attr("cx", x(26)).attr("cy", yc).attr("r", 22).attr("fill", "none").attr("stroke", C.red).attr("stroke-dasharray", "4 4").attr("stroke-width", 2);
    alt.append("line").attr("x1", x(8)).attr("x2", x(26)).attr("y1", yc + 80).attr("y2", yc + 80).attr("stroke", C.orange).attr("stroke-width", 2);
    alt.append("text").attr("x", x(17)).attr("y", yc + 100).attr("text-anchor", "middle").attr("fill", C.orange).style("font-size", "14px")
      .text("18 Å in the paper's crystallographic equation and one figure");
    // membrane
    const mem = svg.append("g").attr("opacity", 0);
    const my = 430;
    mem.append("rect").attr("x", 40).attr("y", my).attr("width", W - 80).attr("height", 38).attr("fill", "#2b2a24");
    mem.append("text").attr("x", 52).attr("y", my + 24).attr("fill", C.muted).style("font-size", "13px").text("ER membrane");
    [x(4), x(22.5)].forEach((cx) => {
      mem.append("path").attr("d", `M${cx},${yc + 72} C${cx},${my - 30} ${cx + 18},${my - 20} ${cx + 18},${my + 30}`).attr("fill", "none").attr("stroke", C.aqua).attr("stroke-width", 4);
    });
    mem.append("text").attr("x", x(22.5) + 30).attr("y", my - 14).attr("fill", C.aqua).style("font-size", "13px").text("N-terminal anchor");
    // electron
    const e = svg.append("circle").attr("r", 9).attr("cy", yc).attr("fill", "#ffffff").attr("stroke", C.bg).attr("stroke-width", 2);
    return { set, x, e, alt, mem, dl };
  }, (h, step, p) => {
    const pos = d3.scaleLinear().domain([0, 0.33, 0.66, 1]).range([0, 4, 8, 22]).clamp(true)(p);
    h.e.attr("cx", h.x(pos));
    h.alt.transition().duration(T).attr("opacity", step === 1 ? 1 : 0);
    h.mem.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.set(...(step === 0 ? ["NADPH → FAD → FMN → heme", "two proteins; spacings as used in the paper's code"]
      : step === 1 ? ["The FMN–heme distance is not settled in the paper", "14 Å in the code and most of the text, 18 Å in one equation"]
      : ["Both partners sit on one membrane", "the anchor is the boundary condition from the derivation"]));
  }, step, progress);
  return <Svg svgRef={ref} />;
}

// ── 2. Distance and rate ──────────────────────────────────────────────────
const md = (r) => 10 ** (13 - 0.6 * (r - 3.6)); // Moser–Dutton, optimal driving force
const K4 = 3.6517e7, BETA = 1.1;
const paperK = (r) => K4 * Math.exp(-BETA * (r - 4));

export function DistanceRate({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 90, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLinear().domain([3, 22]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([1e1, 1e13]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(10)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7).tickFormat(logFmt)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "edge-to-edge distance R (Å)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "electron-transfer rate k (s⁻¹)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const rs = d3.range(3, 22.01, 0.1);
    svg.append("path").attr("d", d3.line()(rs.map((r) => [x(r), y(md(r))]))).attr("fill", "none").attr("stroke", C.ink2).attr("stroke-width", 2.5);
    svg.append("text").attr("x", x(4.2)).attr("y", y(md(4.2)) - 10).attr("fill", C.ink2).style("font-size", "13px").text("Moser–Dutton ruler, optimal ΔG");
    const paper = svg.append("g").attr("opacity", 0);
    paper.append("path").attr("d", d3.line()(rs.filter((r) => r >= 4 && paperK(r) >= 10).map((r) => [x(r), y(paperK(r))]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3);
    paper.append("circle").attr("cx", x(14)).attr("cy", y(paperK(14))).attr("r", 7).attr("fill", C.yellow);
    paper.append("text").attr("x", x(14) - 12).attr("y", y(paperK(14)) + 22).attr("text-anchor", "end").attr("fill", C.yellow).style("font-size", "13px").text(`paper's Marcus rate at 14 Å: ${paperK(14).toFixed(0)} s⁻¹`);
    paper.append("circle").attr("cx", x(14)).attr("cy", y(5e7)).attr("r", 7).attr("fill", C.violet);
    paper.append("text").attr("x", x(14) + 12).attr("y", y(5e7) + 5).attr("fill", C.violet).style("font-size", "13px").text("rate it cites: 5×10⁷ s⁻¹");
    paper.append("line").attr("x1", x(14)).attr("x2", x(14)).attr("y1", y(paperK(14)) - 8).attr("y2", y(5e7) + 8).attr("stroke", C.ink2).attr("stroke-dasharray", "3 4");
    const two = svg.append("g").attr("opacity", 0);
    [14, 18].forEach((r) => {
      two.append("path").attr("d", d3.symbol(d3.symbolDiamond, 140)()).attr("transform", `translate(${x(r)},${y(md(r))})`).attr("fill", r === 14 ? C.aqua : C.orange);
      two.append("text").attr("x", x(r) + 12).attr("y", y(md(r)) - 10).attr("fill", r === 14 ? C.aqua : C.orange).style("font-size", "13px")
        .text(`${r} Å: ${fmtSci(md(r), 1)} s⁻¹`);
    });
    return { set, paper, two };
  }, (h, step) => {
    h.set(...(step === 0 ? ["A standard ruler for tunnelling distance", "log₁₀k = 13 − 0.6(R − 3.6), Moser & Dutton"]
      : step === 1 ? ["The paper's own Marcus curve", "λ 0.85 eV, ΔG −0.1 eV, H_DA 1 meV, β 1.1 Å⁻¹: five decades below the rate it cites"]
      : ["Why the distance matters", "14 Å against 18 Å is 2.4 decades on the ruler"]));
    h.paper.transition().duration(T).attr("opacity", step === 1 ? 1 : 0);
    h.two.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 3. Three hops in series ───────────────────────────────────────────────
const HOPS = [
  { hop: "NADPH → FAD", k: 600, src: "cited, Murataliev 1999" },
  { hop: "FAD → FMN", k: 1e7, src: "no citation" },
  { hop: "FMN → heme", k: 5e7, src: "cited, Denisov 2007" },
];
const SERIES = 1 / d3.sum(HOPS, (h) => 1 / h.k);

export function HopRates({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 150, r: 60, t: 100, b: 80 };
    const set = caption(svg);
    const x = d3.scaleLog().domain([1e1, 1e13]).range([m.l, W - m.r]);
    const rows = [...HOPS.map((h) => h.hop), "chain in series", "3-D animation"];
    const y = d3.scaleBand().domain(rows).range([m.t, H - m.b]).padding(0.3);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7).tickFormat(logFmt)), { grid: H - m.b - m.t });
    label(svg, "rate (s⁻¹)", { x: (m.l + W - m.r) / 2, y: H - 26 });
    const rl = svg.append("g").selectAll("text").data(rows).join("text").attr("x", m.l - 14).attr("y", (r) => y(r) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "14px").text((r) => r);
    const bars = svg.append("g").selectAll("rect").data(HOPS).join("rect").attr("x", m.l).attr("y", (h) => y(h.hop)).attr("height", y.bandwidth())
      .attr("width", (h) => x(h.k) - m.l).attr("fill", C.violet).attr("rx", 3);
    const bt = svg.append("g").selectAll("text").data(HOPS).join("text").attr("x", (h) => x(h.k) + 8).attr("y", (h) => y(h.hop) + y.bandwidth() / 2 + 5)
      .attr("fill", C.ink).style("font-size", "12px").text((h) => `${fmtSci(h.k, 1)} · ${h.src}`);
    const ser = svg.append("g").attr("opacity", 0);
    ser.append("rect").attr("x", m.l).attr("y", y("chain in series")).attr("height", y.bandwidth()).attr("width", x(SERIES) - m.l).attr("fill", C.yellow).attr("rx", 3);
    ser.append("text").attr("x", x(SERIES) + 8).attr("y", y("chain in series") + y.bandwidth() / 2 + 5).attr("fill", C.ink).style("font-size", "12px")
      .text(`1/Σ(1/k) = ${SERIES.toFixed(0)} s⁻¹: set by the first hop`);
    const anim = svg.append("g").attr("opacity", 0);
    anim.append("rect").attr("x", x(2e12)).attr("width", x(6e12) - x(2e12)).attr("y", y("3-D animation")).attr("height", y.bandwidth()).attr("fill", C.orange).attr("rx", 3);
    anim.append("text").attr("x", x(2e12) - 8).attr("y", y("3-D animation") + y.bandwidth() / 2 + 5).attr("text-anchor", "end").attr("fill", C.orange).style("font-size", "12px")
      .text("2–6 ×10¹² s⁻¹, for display");
    return { set, bars, ser, anim, rl };
  }, (h, step) => {
    h.set(...(step === 0 ? ["The three hop rates the paper uses", "two cited, one not"]
      : step === 1 ? ["In series the slowest hop sets the pace", "that is the NADPH hop; the paper's text names the FMN → heme hop"]
      : ["The animation on this site runs ten decades faster", "its rates are chosen so the electron moves within a picosecond"]));
    h.bars.transition().duration(T).attr("opacity", (d, i) => (step === 1 && i > 0 ? 0.35 : 1));
    h.ser.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.anim.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.rl.transition().duration(T).attr("opacity", (r) => ((r === "chain in series" && step < 1) || (r === "3-D animation" && step < 2) ? 0 : 1));
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 4. Catalytic efficiency against the number of steps ───────────────────
const SUBS = [
  { name: "testosterone 6β-OH", v: 1.5e6 },
  { name: "midazolam 1′-OH", v: 4.0e6 },
  { name: "erythromycin", v: 7.0e5 },
  { name: "nifedipine", v: 2.0e6 },
];

export function EfficiencyChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 90, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLinear().domain([0.5, 8.5]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([1, 10]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(9).tickFormat((v) => `10${"⁰¹²³⁴⁵⁶⁷⁸⁹"[v] ?? "¹⁰"}`)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "number of sequential steps d_C", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "k_cat/K_M (M⁻¹s⁻¹)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    svg.append("path").attr("d", d3.line()(d3.range(1, 8.01, 1).map((d) => [x(d), y(10 - d)]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3);
    svg.append("g").selectAll("circle").data(d3.range(1, 9)).join("circle").attr("cx", (d) => x(d)).attr("cy", (d) => y(10 - d)).attr("r", 5).attr("fill", C.yellow);
    const d4 = svg.append("g");
    d4.append("circle").attr("cx", x(4)).attr("cy", y(6)).attr("r", 10).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 2);
    d4.append("text").attr("x", x(4) + 16).attr("y", y(6) - 14).attr("fill", C.yellow).style("font-size", "13px").text("d_C = 4 → 10⁶");
    const subs = svg.append("g").attr("opacity", 0);
    subs.selectAll("circle").data(SUBS).join("circle").attr("cx", (s, i) => x(4) - 30 - i * 14).attr("cy", (s) => y(Math.log10(s.v))).attr("r", 6).attr("fill", C.violet);
    subs.selectAll("text").data(SUBS).join("text").attr("x", x(4) - 100).attr("y", (s) => y(Math.log10(s.v)) + 4).attr("text-anchor", "end")
      .attr("fill", C.violet).style("font-size", "12px").text((s) => `${s.name} ${fmtSci(s.v, 1)}`);
    const spread = svg.append("g").attr("opacity", 0);
    const lo = d3.min(SUBS, (s) => Math.log10(s.v)), hi = d3.max(SUBS, (s) => Math.log10(s.v));
    spread.append("rect").attr("x", x(3.3)).attr("width", x(4.7) - x(3.3)).attr("y", y(hi)).attr("height", y(lo) - y(hi)).attr("fill", C.violet).attr("opacity", 0.15);
    spread.append("text").attr("x", x(4.8)).attr("y", y((lo + hi) / 2) + 5).attr("fill", C.ink2).style("font-size", "13px")
      .text(`spread ${(hi - lo).toFixed(2)} decades; the rule gives one value`);
    return { set, subs, spread };
  }, (h, step) => {
    h.set(...(step === 0 ? ["One decade per step", "the paper's rule log₁₀(k_cat/K_M) = 10 − d_C, with d_C = 4 for the chain"]
      : step === 1 ? ["Four CYP3A4 substrates, as cited in the paper", "geometric mean 1.7×10⁶ M⁻¹s⁻¹"]
      : ["What the rule cannot express", "k_cat/K_M also contains binding and chemistry, which differ by substrate"]));
    h.subs.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.spread.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 5. The flavin semiquinone ─────────────────────────────────────────────
export function SemiquinoneChart({ step }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const y = d3.scaleLinear().domain([-5, 9]).range([H - 70, 100]);
    const lv = [
      { s: "oxidised", g: 0, x: 170 },
      { s: "semiquinone", g: 2.8, x: 380 },
      { s: "hydroquinone", g: 7.37, x: 590 },
    ];
    axis(svg.append("g").attr("transform", "translate(80,0)").call(d3.axisLeft(y).ticks(7)), { grid: W - 120, orient: "y" });
    label(svg, "free energy relative to oxidised (kcal/mol)", { x: 26, y: (100 + H - 70) / 2, rotate: true });
    const lvl = svg.append("g").selectAll("g").data(lv).join("g");
    lvl.append("line").attr("x1", (d) => d.x - 70).attr("x2", (d) => d.x + 70).attr("y1", (d) => y(d.g)).attr("y2", (d) => y(d.g)).attr("stroke", C.yellow).attr("stroke-width", 5);
    lvl.append("text").attr("x", (d) => d.x).attr("y", (d) => y(d.g) - 12).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "14px").text((d) => d.s);
    lvl.append("text").attr("x", (d) => d.x).attr("y", (d) => y(d.g) + 22).attr("text-anchor", "middle").attr("fill", C.yellow).style("font-size", "13px").text((d) => (d.g ? `+${d.g.toFixed(2)}` : "0"));
    const lit = svg.append("g").attr("opacity", 0);
    lit.append("line").attr("x1", 310).attr("x2", 450).attr("y1", y(-3)).attr("y2", y(-3)).attr("stroke", C.violet).attr("stroke-width", 5);
    lit.append("text").attr("x", 380).attr("y", y(-3) + 24).attr("text-anchor", "middle").attr("fill", C.violet).style("font-size", "13px").text("−3, the value the paper quotes (no citation)");
    lit.append("line").attr("x1", 380).attr("x2", 380).attr("y1", y(2.8) + 30).attr("y2", y(-3) - 8).attr("stroke", C.ink2).attr("stroke-dasharray", "3 4");
    return { set, lit };
  }, (h, step) => {
    h.set(...(step === 0 ? ["The flavin's three redox states", "the model's free energies, ΔG = 65 kJ/mol × ΔM"]
      : ["The sign is the opposite of the quoted value", "a stable semiquinone sits below the oxidised state; the model puts it above"]));
    h.lit.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 6. Occupancy and density along the chain, as the shader draws it ──────
const KS = [6e12, 4e12, 2e12];
function occ(tfs) {
  const t = tfs * 1e-15;
  const a = KS.map((k) => Math.exp(-k * t));
  return [a[0], (1 - a[0]) * a[1], (1 - a[0]) * (1 - a[1]) * a[2], (1 - a[0]) * (1 - a[1]) * (1 - a[2])];
}

export function OccupancyChart({ step, progress }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const m = { l: 80, r: 40 };
    // top: occupancy vs time
    const t1 = 100, b1 = 250;
    const x = d3.scaleLinear().domain([0, 800]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([b1, t1]);
    axis(svg.append("g").attr("transform", `translate(0,${b1})`).call(d3.axisBottom(x).ticks(8)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4)));
    label(svg, "time (fs)", { x: W - m.r, y: b1 + 34, anchor: "end" });
    label(svg, "occupancy", { x: 26, y: (t1 + b1) / 2, rotate: true });
    const ts = d3.range(0, 800.1, 4);
    COF.forEach((c, j) => {
      svg.append("path").attr("d", d3.line()(ts.map((t) => [x(t), y(occ(t)[j])]))).attr("fill", "none").attr("stroke", c.c).attr("stroke-width", 2.5);
      svg.append("text").attr("x", W - m.r - (3 - j) * 70).attr("y", t1 - 10).attr("fill", c.c).style("font-size", "13px").attr("text-anchor", "end").text(`— ${c.id}`);
    });
    const cursor = svg.append("line").attr("y1", t1).attr("y2", b1).attr("stroke", "#ffffff").attr("stroke-width", 1.5);
    // bottom: density along the chain
    const t2 = 350, b2 = 488;
    const xs = d3.scaleLinear().domain([-4, 27]).range([m.l, W - m.r]);
    const ys = d3.scaleLinear().domain([0, 1.05]).range([b2, t2]);
    axis(svg.append("g").attr("transform", `translate(0,${b2})`).call(d3.axisBottom(xs).ticks(8)));
    label(svg, "position along the chain (Å)", { x: (m.l + W - m.r) / 2, y: b2 + 34 });
    svg.append("g").selectAll("text").data(COF).join("text").attr("x", (c) => xs(c.x)).attr("y", t2 - 6).attr("text-anchor", "middle")
      .attr("fill", (c) => c.c).style("font-size", "12px").text((c) => c.id);
    const area = svg.append("path").attr("fill", "#ffffff").attr("fill-opacity", 0.18).attr("stroke", "#ffffff").attr("stroke-width", 2);
    const tl = svg.append("text").attr("x", m.l).attr("y", t2 - 28).attr("fill", C.ink).style("font-size", "13px");
    const lam = svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", b2 + 62).attr("text-anchor", "middle").attr("fill", C.orange).style("font-size", "13px").attr("opacity", 0)
      .text("λ readout: 0.85 eV at every frame, by construction");
    return { set, x, xs, ys, cursor, area, tl, lam };
  }, (h, step, p) => {
    const t = 800 * p;
    h.cursor.attr("x1", h.x(t)).attr("x2", h.x(t));
    const o = occ(t);
    const xs = d3.range(-4, 27.01, 0.2);
    const rho = xs.map((xx) => d3.sum(COF, (c, j) => o[j] * Math.exp(-((xx - c.x) ** 2) / (2 * c.sigma ** 2))));
    const mx = d3.max(rho) || 1;
    h.area.attr("d", d3.area().x((d, i) => h.xs(xs[i])).y0(h.ys(0)).y1((d) => h.ys(d / mx))(rho));
    h.tl.text(`t = ${t.toFixed(0)} fs · ${COF.map((c, j) => `${c.id} ${(100 * o[j]).toFixed(0)}%`).join(" · ")}`);
    h.lam.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.set(...(step === 0 ? ["What the shader pipeline draws", "occupancies from three assumed hop rates, 6, 4 and 2 ×10¹² s⁻¹"]
      : step === 1 ? ["Density along the chain at the cursor", "a Gaussian per cofactor, weighted by its occupancy; scroll moves time"]
      : ["What it does not measure", "the reorganisation energy it reports is fixed in the code"]));
  }, step, progress);
  return <Svg svgRef={ref} />;
}
