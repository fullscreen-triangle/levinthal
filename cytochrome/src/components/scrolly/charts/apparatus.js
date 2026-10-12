// Charts for the apparatus page. Numbers from
// publications/catalytic-cycle/multi-hop-et-chain/validation/scripts/09–12,
// glb/levinthal_glb/shader_pipeline.py and the source papers in sources/.

import * as d3 from "d3";
import { C, T, axis, label, useChart } from "../core";
import { Svg, W, H, caption, fmtSci, logFmt } from "./base";

export const LAYERS = [
  { n: 5, name: "hologram pipeline", what: "GPU shader passes → density and six observables", c: C.orange },
  { n: 4, name: "harmonic molecular resonator", what: "closed loops of commensurate modes, counted by cycle rank", c: C.red },
  { n: 3, name: "ensemble strobes", what: "three time gates set by absorption and emission times", c: C.aqua },
  { n: 2, name: "triple-equivalence theorem", what: "the argument that licenses reading hardware as atomic coordinates", c: C.muted },
  { n: 1, name: "categorical spectrometer", what: "four hardware oscillators read as (n, ℓ, m, s)", c: C.blue },
];

// ── 1. The stack and its time scales ──────────────────────────────────────
const OSC = [
  { name: "memory refresh → s", script: 1e4, source: 6.4e4 },
  { name: "memory bus → ℓ", script: 1e8, source: 8e8 },
  { name: "CPU clock → n", script: 1e9, source: 3e9 },
  { name: "LED emission → m", script: 1e14, source: 4.6e14 },
];
const GATES = [
  { name: "W_Sk", tau: 1e-15 },
  { name: "W_St", tau: 1e-9 },
  { name: "W_Se", tau: 1e-6 },
];

export function StackChart({ step }) {
  const ref = useChart((svg) => {
    const set = caption(svg);
    const g0 = svg.append("g");
    const bh = 70, top = 100;
    LAYERS.forEach((L, i) => {
      const y0 = top + i * (bh + 12);
      g0.append("rect").attr("x", 60 + i * 14).attr("y", y0).attr("width", W - 120 - i * 28).attr("height", bh).attr("rx", 10)
        .attr("fill", C.panel).attr("stroke", L.c).attr("stroke-width", 1.5);
      g0.append("text").attr("x", 84 + i * 14).attr("y", y0 + 30).attr("fill", L.c).style("font-size", "20px").style("font-weight", 700).text(`L${L.n}`);
      g0.append("text").attr("x", 134 + i * 14).attr("y", y0 + 30).attr("fill", C.ink).style("font-size", "16px").text(L.name);
      g0.append("text").attr("x", 134 + i * 14).attr("y", y0 + 52).attr("fill", C.muted).style("font-size", "13px").text(L.what);
    });
    const g1 = svg.append("g").attr("opacity", 0);
    const m = { l: 70, r: 40 };
    const x = d3.scaleLog().domain([1e3, 1e16]).range([m.l, W - m.r]);
    const ay = 300;
    axis(g1.append("g").attr("transform", `translate(0,${ay})`).call(d3.axisBottom(x).ticks(13).tickFormat(logFmt)));
    label(g1, "frequency (Hz)", { x: (m.l + W - m.r) / 2, y: ay + 40 });
    OSC.forEach((o, i) => {
      const yy = ay - 40 - i * 42;
      g1.append("line").attr("x1", x(o.script)).attr("x2", x(o.source)).attr("y1", yy).attr("y2", yy).attr("stroke", C.ink2).attr("stroke-dasharray", "2 3");
      g1.append("circle").attr("cx", x(o.script)).attr("cy", yy).attr("r", 7).attr("fill", C.blue);
      g1.append("circle").attr("cx", x(o.source)).attr("cy", yy).attr("r", 7).attr("fill", "none").attr("stroke", C.blue).attr("stroke-width", 2);
      const left = o.source > 1e12;
      g1.append("text").attr("x", left ? x(o.script) - 12 : x(o.source) + 12).attr("y", yy + 5).attr("text-anchor", left ? "end" : "start")
        .attr("fill", C.ink2).style("font-size", "13px").text(o.name);
    });
    g1.append("text").attr("x", m.l).attr("y", 104).attr("fill", C.muted).style("font-size", "12px").text("filled: value in the validation script · open: value in the source paper");
    const g2 = svg.append("g").attr("opacity", 0);
    const xt = d3.scaleLog().domain([1e-16, 1e-5]).range([m.l, W - m.r]);
    const ty = 470;
    axis(g2.append("g").attr("transform", `translate(0,${ty})`).call(d3.axisBottom(xt).ticks(11).tickFormat(logFmt)));
    label(g2, "gate time scale (s)", { x: (m.l + W - m.r) / 2, y: ty + 40 });
    GATES.forEach((gt) => {
      g2.append("rect").attr("x", xt(gt.tau) - 4).attr("width", 8).attr("y", ty - 60).attr("height", 60).attr("fill", C.aqua);
      g2.append("text").attr("x", xt(gt.tau)).attr("y", ty - 68).attr("text-anchor", "middle").attr("fill", C.aqua).style("font-size", "13px").text(gt.name);
    });
    return { set, g0, g1, g2 };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Five layers, as the paper defines them", "bottom: hardware oscillators; top: a GPU shader pipeline"]
      : step === 1 ? ["Layer 1: four oscillators, ten decades apart", "the script uses round values; the source paper's are different"]
      : ["Layer 3: three gates from femtoseconds to microseconds", "set as constants in the script"]));
    h.g0.transition().duration(T).attr("opacity", step === 0 ? 1 : 0);
    h.g1.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.g2.transition().duration(T).attr("opacity", step >= 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 2. Calibration against hydrogen ───────────────────────────────────────
// Fractional difference computed − NIST, recomputed from the tables in
// sources/hyperfine-transitions.tex. Zero entries agree to the digits printed.
const HLINES = [
  ["Ly-α", 0, "Lyman"], ["Ly-β", 1.95e-6, "Lyman"], ["Ly-γ", 3.08e-6, "Lyman"], ["Ly-δ", 3.16e-6, "Lyman"], ["Ly-ε", 7.46e-6, "Lyman"], ["Ly limit", 3.29e-6, "Lyman"],
  ["H-α", 4.6e-7, "Balmer"], ["H-β", 0, "Balmer"], ["H-γ", 4.6e-7, "Balmer"], ["H-δ", 4.9e-7, "Balmer"], ["H-ε", 1.26e-6, "Balmer"], ["Ba limit", 2.76e-4, "Balmer"],
  ["Pa-α", 5.3e-5, "Paschen"], ["Pa-β", 1.56e-4, "Paschen"], ["Pa-γ", 1.83e-4, "Paschen"], ["Pa-δ", 9.9e-5, "Paschen"],
  ["Br-α", 1.97e-4, "Brackett"], ["Br-β", 3.05e-4, "Brackett"], ["Br-γ", 2.31e-4, "Brackett"],
  ["Pf-α", 2.7e-5, "Pfund"], ["Pf-β", 4.3e-5, "Pfund"],
  ["21 cm", 1.25e-9, "other"], ["ionisation", 5.34e-4, "other"], ["Ly-α doublet", 0, "other"], ["2p fine", 1.5e-7, "other"], ["Lamb shift", 1.46e-4, "other"],
].map(([n, e, s]) => ({ n, e, s }));
const SERIES_C = { Lyman: C.violet, Balmer: C.blue, Paschen: C.aqua, Brackett: C.yellow, Pfund: C.orange, other: C.magenta };

export function HydrogenChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 80, r: 30, t: 100, b: 110 };
    const set = caption(svg);
    const x = d3.scaleBand().domain(HLINES.map((d) => d.n)).range([m.l, W - m.r]).padding(0.2);
    const y = d3.scaleLog().domain([1e-10, 1e-3]).range([H - m.b, m.t]).clamp(true);
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7).tickFormat(logFmt)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "fractional difference from NIST", { x: 24, y: (m.t + H - m.b) / 2, rotate: true });
    svg.append("g").selectAll("text").data(HLINES).join("text").attr("transform", (d) => `translate(${x(d.n) + x.bandwidth() / 2},${H - m.b + 10}) rotate(55)`)
      .attr("fill", C.ink2).style("font-size", "11px").text((d) => d.n);
    const bars = svg.append("g").selectAll("rect").data(HLINES).join("rect").attr("x", (d) => x(d.n)).attr("width", x.bandwidth())
      .attr("y", (d) => (d.e > 0 ? y(d.e) : H - m.b - 3)).attr("height", (d) => (d.e > 0 ? H - m.b - y(d.e) : 3))
      .attr("fill", (d) => SERIES_C[d.s]).attr("rx", 2);
    const mean = d3.mean(HLINES, (d) => d.e);
    const lines = svg.append("g").attr("opacity", 0);
    [[mean, `mean of these ${HLINES.length}: ${fmtSci(mean, 1)}`, C.ink], [1.6e-4, "stated in the paper: 1.6×10⁻⁴ over 24 lines", C.orange]].forEach(([v, t, c], i) => {
      lines.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", c).attr("stroke-dasharray", "5 4");
      lines.append("text").attr("x", m.l + 8).attr("y", y(v) + (i ? -8 : 16)).attr("fill", c).style("font-size", "12px").text(t);
    });
    return { set, bars, lines };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Hydrogen lines against NIST", "recomputed here from the tables in the source paper"]
      : step === 1 ? ["The summary does not match the table", "26 lines, mean 8.7×10⁻⁵; the text gives 24 lines and 1.6×10⁻⁴"]
      : ["What the agreement shows", "the 'spectrometer' column equals the Rydberg formula: standard physics, reproduced"]));
    h.lines.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.bars.transition().duration(T).attr("opacity", (d) => (step === 2 && d.s !== "other" && d.s !== "Lyman" && d.s !== "Balmer" ? 0.5 : 1));
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 3. Strobe cross-talk ──────────────────────────────────────────────────
const eta = (r) => 0.37 * Math.min(r, 1);

export function CrosstalkChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 100, b: 70 };
    const set = caption(svg);
    const x = d3.scaleLog().domain([1e-4, 10]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([1e-5, 1]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(logFmt)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(logFmt)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "gate width ÷ emission time, Δt/τ_em", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "cross-talk η", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const rs = d3.range(-4, 1.001, 0.05).map((e) => 10 ** e);
    svg.append("path").attr("d", d3.line()(rs.map((r) => [x(r), y(eta(r))]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3);
    const op = svg.append("g").attr("opacity", 0);
    op.append("circle").attr("cx", x(1e-2)).attr("cy", y(3.7e-3)).attr("r", 8).attr("fill", C.aqua);
    op.append("text").attr("x", x(1e-2) + 14).attr("y", y(3.7e-3) + 22).attr("fill", C.aqua).style("font-size", "13px").text("10 ps gate, 1 ns emission: η = 3.7×10⁻³");
    const chk = svg.append("g").attr("opacity", 0);
    chk.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(1e-2)).attr("y2", y(1e-2)).attr("stroke", C.red).attr("stroke-dasharray", "5 4");
    chk.append("text").attr("x", W - m.r).attr("y", y(1e-2) - 8).attr("text-anchor", "end").attr("fill", C.red).style("font-size", "12px").text("the script's pass line, compared with a constant 3.7×10⁻³");
    return { set, op, chk };
  }, (h, step) => {
    h.set(...(step === 0 ? ["How much one gate leaks into the next", "η = 0.37·Δt/τ_em, saturating at 0.37"]
      : step === 1 ? ["The operating point the paper chooses", "for the reductase: emission ~1 ns, gate ~10 ps"]
      : ["What the validation checks", "a typed constant against a threshold; no counts are gated"]));
    h.op.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.chk.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 4. Cycle rank of the resonator graphs ─────────────────────────────────
const MOL = [
  { m: "H₂", N: 2, E: 2, Cs: 1 },
  { m: "CO", N: 7, E: 9, Cs: 3 },
  { m: "H₂O", N: 3, E: 5, Cs: 1 },
  { m: "CO₂", N: 3, E: 5, Cs: 1 },
  { m: "CH₄", N: 4, E: 9, Cs: 6 },
  { m: "C₆H₆", N: 5, E: 10, Cs: 6 },
].map((d) => ({ ...d, Cf: d.E - d.N + 1 }));

export function CycleRankChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 80, r: 40, t: 100, b: 70 };
    const set = caption(svg);
    const x0 = d3.scaleBand().domain([...MOL.map((d) => d.m), "CPR chain"]).range([m.l, W - m.r]).padding(0.25);
    const x1 = d3.scaleBand().domain(["Cs", "Cf"]).range([0, x0.bandwidth()]).padding(0.1);
    const y = d3.scaleLinear().domain([0, 7]).range([H - m.b, m.t]);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x0)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(7)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "cycle rank C", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const g = svg.append("g").selectAll("g").data(MOL).join("g").attr("transform", (d) => `translate(${x0(d.m)},0)`);
    g.append("rect").attr("x", x1("Cs")).attr("width", x1.bandwidth()).attr("y", (d) => y(d.Cs)).attr("height", (d) => H - m.b - y(d.Cs)).attr("fill", C.red).attr("rx", 2);
    const cf = g.append("rect").attr("x", x1("Cf")).attr("width", x1.bandwidth()).attr("y", (d) => y(d.Cf)).attr("height", (d) => H - m.b - y(d.Cf))
      .attr("fill", C.aqua).attr("rx", 2).attr("opacity", 0);
    const bad = g.filter((d) => d.Cs !== d.Cf).append("text").attr("x", x0.bandwidth() / 2).attr("y", (d) => y(d.Cf) - 10).attr("text-anchor", "middle")
      .attr("fill", C.orange).style("font-size", "13px").attr("opacity", 0).text("≠");
    const cpr = svg.append("g").attr("opacity", 0);
    cpr.append("rect").attr("x", x0("CPR chain")).attr("width", x0.bandwidth()).attr("y", y(6)).attr("height", H - m.b - y(6)).attr("fill", "none")
      .attr("stroke", C.muted).attr("stroke-dasharray", "5 4");
    cpr.append("text").attr("x", x0("CPR chain") + x0.bandwidth() / 2).attr("y", y(6) - 26).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "12px").text("C ≥ 6 typed in;");
    cpr.append("text").attr("x", x0("CPR chain") + x0.bandwidth() / 2).attr("y", y(6) - 10).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "12px").text("no graph built");
    const leg = svg.append("g");
    leg.append("rect").attr("x", W - 300).attr("y", 76).attr("width", 12).attr("height", 12).attr("fill", C.red);
    leg.append("text").attr("x", W - 282).attr("y", 87).attr("fill", C.ink2).style("font-size", "12px").text("C in the paper's table");
    const l2 = leg.append("g").attr("opacity", 0);
    l2.append("rect").attr("x", W - 300).attr("y", 94).attr("width", 12).attr("height", 12).attr("fill", C.aqua);
    l2.append("text").attr("x", W - 282).attr("y", 105).attr("fill", C.ink2).style("font-size", "12px").text("|E| − |V| + 1 from the same table");
    return { set, cf, bad, cpr, l2 };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Independent loops in a graph of commensurate modes", "the source paper's table for six small molecules"]
      : step === 1 ? ["Recounting with the paper's own formula", "H₂O and CO₂ disagree with the table"]
      : ["For the reductase chain the graph was never built", "the required rank is a constant in the script"]));
    h.cf.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.bad.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.l2.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.cpr.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 5. Cofactor self-selection by χ² ──────────────────────────────────────
const CHI = [
  { a: "FMN N5", v: 59.163 },
  { a: "NADPH C4", v: 45.041 },
  { a: "heme Fe", v: 28.832 },
  { a: "FAD N5", v: 18.824 },
];
const BY = { n: 196, mean: 1.907, max: 13.026, above: 11, thr: 5.99 };

export function ChiChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 175, r: 40, t: 100, b: 80 };
    const set = caption(svg);
    const x = d3.scaleLinear().domain([0, 65]).range([m.l, W - m.r]);
    const rows = ["196 bystander atoms", ...CHI.map((c) => c.a)];
    const y = d3.scaleBand().domain(rows).range([m.t, 360]).padding(0.3);
    axis(svg.append("g").attr("transform", `translate(0,360)`).call(d3.axisBottom(x).ticks(7)), { grid: 360 - m.t });
    label(svg, "χ² of the three-state count histogram", { x: (m.l + W - m.r) / 2, y: 398 });
    svg.append("g").selectAll("text").data(rows).join("text").attr("x", m.l - 12).attr("y", (r) => y(r) + y.bandwidth() / 2 + 5).attr("text-anchor", "end")
      .attr("fill", C.ink2).style("font-size", "13px").text((r) => r);
    svg.append("line").attr("x1", x(BY.thr)).attr("x2", x(BY.thr)).attr("y1", m.t - 8).attr("y2", 360).attr("stroke", C.red).attr("stroke-dasharray", "4 4");
    svg.append("text").attr("x", x(BY.thr) + 4).attr("y", m.t - 12).attr("fill", C.red).style("font-size", "12px").text("threshold 5.99");
    const yb = y("196 bystander atoms");
    svg.append("rect").attr("x", x(0)).attr("width", x(BY.max) - x(0)).attr("y", yb).attr("height", y.bandwidth()).attr("fill", C.muted).attr("opacity", 0.35).attr("rx", 3);
    svg.append("line").attr("x1", x(BY.mean)).attr("x2", x(BY.mean)).attr("y1", yb).attr("y2", yb + y.bandwidth()).attr("stroke", C.ink).attr("stroke-width", 2);
    svg.append("text").attr("x", x(BY.max) + 8).attr("y", yb + y.bandwidth() / 2 + 5).attr("fill", C.ink2).style("font-size", "12px")
      .text(`mean ${BY.mean}, max ${BY.max}; ${BY.above} of ${BY.n} above threshold`);
    const cof = svg.append("g").attr("opacity", 0);
    CHI.forEach((c) => {
      cof.append("rect").attr("x", x(0)).attr("width", x(c.v) - x(0)).attr("y", y(c.a)).attr("height", y.bandwidth()).attr("fill", C.yellow).attr("rx", 3);
      cof.append("text").attr("x", x(c.v) + 8).attr("y", y(c.a) + y.bandwidth() / 2 + 5).attr("fill", C.ink).style("font-size", "12px").text(c.v.toFixed(1));
    });
    // the two sampling distributions
    const dist = svg.append("g").attr("opacity", 0);
    const dx = d3.scaleBand().domain(["0", "1", "2"]).range([m.l, m.l + 220]).padding(0.2);
    const dy = d3.scaleLinear().domain([0, 0.7]).range([520, 440]);
    [["equilibrium (bystanders)", [0.261, 0.499, 0.24], C.muted, 0], ["perturbed (cofactors)", [0.1, 0.3, 0.6], C.yellow, 300]].forEach(([n, p, c, off]) => {
      dist.append("g").selectAll("rect").data(p).join("rect").attr("x", (v, i) => off + dx(String(i))).attr("width", dx.bandwidth())
        .attr("y", (v) => dy(v)).attr("height", (v) => 520 - dy(v)).attr("fill", c);
      dist.append("text").attr("x", off + m.l).attr("y", 540).attr("fill", c).style("font-size", "12px").text(`${n}: ${p.join(" / ")}`);
    });
    dist.append("text").attr("x", m.l).attr("y", 428).attr("fill", C.ink2).style("font-size", "12px").text("the script draws each group's counts from a distribution it is given:");
    return { set, cof, dist };
  }, (h, step) => {
    h.set(...(step === 0 ? ["A χ² test on per-atom count histograms", "196 bystander atoms; 11 cross the threshold by chance"]
      : step === 1 ? ["The four cofactors stand out", "each well above threshold"]
      : ["They stand out because they are sampled to", "the counts are synthetic; the test of the protein has not been run"]));
    h.cof.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.dist.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 6. The reorganisation energy readout ──────────────────────────────────
const S11 = [[0.7, 0.674], [0.85, 0.789], [1.0, 0.997]];

export function LambdaChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 100, b: 70 };
    const set = caption(svg);
    const g1 = svg.append("g");
    const x = d3.scaleLinear().domain([0.6, 1.1]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0.6, 1.1]).range([H - m.b, m.t]);
    axis(g1.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5)));
    axis(g1.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: W - m.r - m.l, orient: "y" });
    label(g1, "λ put in to build the synthetic width (eV)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(g1, "λ read back (eV)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    g1.append("line").attr("x1", x(0.6)).attr("x2", x(1.1)).attr("y1", y(0.6)).attr("y2", y(1.1)).attr("stroke", C.muted).attr("stroke-dasharray", "4 4");
    g1.selectAll("circle").data(S11).join("circle").attr("cx", (d) => x(d[0])).attr("cy", (d) => y(d[1])).attr("r", 8).attr("fill", C.yellow);
    g1.selectAll("text.v").data(S11).join("text").attr("class", "v").attr("x", (d) => x(d[0]) + 12).attr("y", (d) => y(d[1]) + 18).attr("fill", C.yellow)
      .style("font-size", "12px").text((d) => `${d[0]} → ${d[1]}`);
    const g2 = svg.append("g").attr("opacity", 0);
    const sx = d3.scaleLinear().domain([0.5, 3.5]).range([m.l, W - m.r]);
    axis(g2.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(sx).ticks(6)));
    axis(g2.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: W - m.r - m.l, orient: "y" });
    label(g2, "density width σ fed to the shader (Å)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    g2.append("line").attr("x1", sx(0.5)).attr("x2", sx(3.5)).attr("y1", y(0.85)).attr("y2", y(0.85)).attr("stroke", C.orange).attr("stroke-width", 3.5);
    g2.append("text").attr("x", sx(2)).attr("y", y(0.85) - 14).attr("text-anchor", "middle").attr("fill", C.orange).style("font-size", "14px")
      .text("λ = (0.85/σ)·σ = 0.85 for every σ");
    const band = svg.append("g").attr("opacity", 0);
    band.append("rect").attr("x", m.l).attr("width", W - m.r - m.l).attr("y", y(1.0)).attr("height", y(0.7) - y(1.0)).attr("fill", C.violet).attr("opacity", 0.15);
    band.append("text").attr("x", W - m.r - 8).attr("y", y(1.0) + 18).attr("text-anchor", "end").attr("fill", C.violet).style("font-size", "12px")
      .text("0.7–1.0 eV, the range the paper quotes");
    return { set, g1, g2, band };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Reading λ back from a width", "script 11: a synthetic width with 5 % noise is built from λ and inverted"]
      : step === 1 ? ["The shader's readout", "the value is fixed in the code, whatever the width"]
      : ["So the 0.85 eV is an input", "it sits in the quoted range because it was chosen from it"]));
    h.g1.transition().duration(T).attr("opacity", step === 0 ? 1 : 0);
    h.g2.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.band.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}
