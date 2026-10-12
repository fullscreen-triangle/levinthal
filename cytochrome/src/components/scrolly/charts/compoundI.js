import * as d3 from "d3";
import { C, T, axis, label, useChart } from "../core";
import { Svg, W, H, arrows, caption, fmtSci, logFmt } from "./base";
import { STATES } from "@/data/talk/applications";
import derivation from "@/data/talk/derivation.json";
import { MORSE, RETURN_MIN, DC_RULE, KIE, LIFETIME_EA, LIFETIME_PTS, POTENTIAL, OBSERVABLES } from "@/data/story/compoundI";

// ── 1. Where Compound I sits in the cycle ─────────────────────────────────
export function CycleRing({ step }) {
  const ref = useChart((svg) => {
    const cx = 380, cy = 300, R = 175, n = STATES.length;
    const arrow = arrows(svg);
    const pos = STATES.map((s, i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
      return { ...s, a, x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
    });
    const arcPath = (i) => {
      const s = pos[i], e = pos[(i + 1) % n], gap = 0.2;
      const a0 = s.a + gap, a1 = e.a - gap + (i === n - 1 ? 2 * Math.PI : 0);
      return `M${cx + R * Math.cos(a0)},${cy + R * Math.sin(a0)} A${R},${R} 0 0 1 ${cx + R * Math.cos(a1)},${cy + R * Math.sin(a1)}`;
    };
    const arcs = svg.append("g").selectAll("path").data(pos).join("path")
      .attr("d", (d, i) => arcPath(i)).attr("fill", "none").attr("stroke", C.line).attr("stroke-width", 3)
      .attr("marker-end", arrow(C.line));
    const nodes = svg.append("g").selectAll("g").data(pos).join("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
    nodes.append("circle").attr("r", 26).attr("fill", C.panel).attr("stroke", C.ink2).attr("stroke-width", 1.5);
    nodes.append("text").attr("text-anchor", "middle").attr("dy", 6).attr("fill", C.ink).style("font-size", "17px").style("font-weight", 600).text((d) => d.id);
    const lab = svg.append("g").selectAll("g").data(pos).join("g").attr("transform", (d) => {
      const r = R + 50;
      return `translate(${cx + r * Math.cos(d.a)},${cy + r * Math.sin(d.a)})`;
    });
    const anchor = (d) => (Math.cos(d.a) > 0.3 ? "start" : Math.cos(d.a) < -0.3 ? "end" : "middle");
    const dy0 = (d) => (Math.sin(d.a) < -0.8 ? -26 : Math.sin(d.a) > 0.8 ? 4 : -8);
    lab.append("text").attr("text-anchor", anchor).attr("dy", dy0).attr("fill", C.ink).style("font-size", "15px").text((d) => d.name);
    lab.append("text").attr("text-anchor", anchor).attr("dy", (d) => dy0(d) + 17).attr("fill", C.muted).style("font-size", "13px").text((d) => d.label);
    const sor = lab.append("text").attr("text-anchor", anchor).attr("dy", (d) => dy0(d) + 34).attr("fill", C.aqua).style("font-size", "13px")
      .attr("opacity", 0).text((d) => `Soret ${d.soret} nm${d.lit ? " · lit." : ""}`);
    const c1 = svg.append("text").attr("x", cx).attr("y", cy - 4).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "17px");
    const c2 = svg.append("text").attr("x", cx).attr("y", cy + 18).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "13px");
    return { arcs, nodes, lab, sor, c1, c2, arrow };
  }, (h, step) => {
    const hot = step === 1 ? [5, 6, 7] : step === 2 ? [1, 7] : STATES.map((s) => s.id);
    h.nodes.transition().duration(T).attr("opacity", (d) => (hot.includes(d.id) ? 1 : 0.25));
    h.nodes.select("circle").transition().duration(T).attr("fill", (d) => (step === 1 && hot.includes(d.id) ? C.orange : C.panel));
    h.lab.transition().duration(T).attr("opacity", (d) => (hot.includes(d.id) ? 1 : 0.25));
    h.arcs.transition().duration(T)
      .attr("stroke", (d) => (step === 1 && [5, 6].includes(d.id) ? C.orange : C.line))
      .attr("marker-end", (d) => h.arrow(step === 1 && [5, 6].includes(d.id) ? C.orange : C.line))
      .attr("stroke-width", (d) => (step === 1 && [5, 6].includes(d.id) ? 5 : 3));
    h.sor.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    const [a, b] = step === 0 ? ["seven states, one cycle", "substrate in → product out"]
      : step === 1 ? ["two protons and an O–O cleavage", "peroxo → Compound 0 → Compound I"]
      : ["417 nm → 367 nm", "the Soret band follows the iron"];
    h.c1.text(a); h.c2.text(b);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 2. The thiolate ligand, measured in public structures ─────────────────
const SITES = derivation.ligation.sites;
const GROUPS = [
  { key: "p450", name: "P450 · Cys Sγ", test: (s) => s.p450 },
  { key: "thiol", name: "other thiolate heme · Cys Sγ", test: (s) => !s.p450 && s.ligand === "CYS" },
  { key: "his", name: "globins, peroxidases, cyt c · His Nε", test: (s) => s.ligand === "HIS" },
  { key: "tyr", name: "catalase · Tyr O", test: (s) => s.ligand === "TYR" },
];
const groupOf = (s) => GROUPS.find((g) => g.test(s)).key;

export function ThiolateStrip({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 250, r: 40, t: 90, b: 70 };
    const x = d3.scaleLinear().domain([1.78, 2.6]).range([m.l, W - m.r]);
    const y = d3.scaleBand().domain(GROUPS.map((g) => g.key)).range([m.t, H - m.b]).padding(0.3);
    const set = caption(svg);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8)), { grid: H - m.b - m.t });
    label(svg, "distance from iron to the proximal ligand atom (Å)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    svg.append("g").selectAll("text").data(GROUPS).join("text").attr("x", m.l - 14).attr("y", (g) => y(g.key) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "14px").text((g) => g.name);
    const color = { p450: C.aqua, thiol: C.yellow, his: C.blue, tyr: C.violet };
    const rnd = d3.randomLcg(7);
    const dots = svg.append("g").selectAll("circle").data(SITES).join("circle")
      .attr("cx", (s) => x(s.d)).attr("cy", (s) => y(groupOf(s)) + y.bandwidth() * (0.2 + 0.6 * rnd()))
      .attr("r", 7).attr("fill", (s) => color[groupOf(s)]).attr("fill-opacity", 0.85).attr("stroke", C.bg);
    const p = SITES.filter((s) => s.p450).map((s) => s.d);
    const mean = d3.mean(p);
    const mline = svg.append("g").attr("opacity", 0);
    mline.append("line").attr("x1", x(mean)).attr("x2", x(mean)).attr("y1", m.t - 10).attr("y2", H - m.b).attr("stroke", C.aqua).attr("stroke-dasharray", "4 4");
    mline.append("text").attr("x", x(mean) - 6).attr("y", m.t - 14).attr("text-anchor", "end").attr("fill", C.aqua).style("font-size", "13px")
      .text(`P450 mean ${mean.toFixed(2)} Å · ${p.length} sites · ${d3.min(p).toFixed(2)}–${d3.max(p).toFixed(2)} Å`);
    return { dots, mline, set };
  }, (h, step) => {
    h.set(...(step === 0 ? ["Proximal ligands in 20 heme protein structures", "each dot is one heme site; distances measured in this work"]
      : step === 1 ? ["Every P450 iron is held by cysteine sulphur", "12 P450 structures, every chain"]
      : ["The thiolate is shared, not unique", "chloroperoxidase and NO synthase also use it"]));
    h.dots.transition().duration(T).attr("opacity", (s) =>
      step === 0 ? 1 : step === 1 ? (s.p450 ? 1 : 0.15) : (groupOf(s) === "thiol" || s.p450 ? 1 : 0.15));
    h.mline.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 3. The O–O bond on a Morse potential ──────────────────────────────────
const morse = (r) => MORSE.De * (1 - Math.exp(-MORSE.a * (r - MORSE.re))) ** 2;
const harm = (r) => MORSE.De * MORSE.a ** 2 * (r - MORSE.re) ** 2;

export function MorseChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 80, r: 40, t: 80, b: 70 };
    const x = d3.scaleLinear().domain([1.0, 3.0]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 80]).range([H - m.b, m.t]);
    const set = caption(svg);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(8)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "O–O distance r (Å)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "energy (kcal/mol)", { x: 24, y: (m.t + H - m.b) / 2, rotate: true });
    const zone = svg.append("g").attr("opacity", 0);
    zone.append("rect").attr("x", x(1.0)).attr("width", x(1.95) - x(1.0)).attr("y", m.t).attr("height", H - m.b - m.t).attr("fill", C.aqua).attr("opacity", 0.08);
    zone.append("rect").attr("x", x(2.4)).attr("width", x(3.0) - x(2.4)).attr("y", m.t).attr("height", H - m.b - m.t).attr("fill", C.orange).attr("opacity", 0.08);
    zone.append("text").attr("x", x(1.48)).attr("y", m.t + 20).attr("text-anchor", "middle").attr("fill", C.aqua).style("font-size", "14px").text("β = 1, bonded");
    zone.append("text").attr("x", x(2.7)).attr("y", m.t + 20).attr("text-anchor", "middle").attr("fill", C.orange).style("font-size", "14px").text("β = 0, cleaved");
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(MORSE.De)).attr("y2", y(MORSE.De)).attr("stroke", C.muted).attr("stroke-dasharray", "3 5");
    svg.append("text").attr("x", W - m.r).attr("y", y(MORSE.De) - 8).attr("text-anchor", "end").attr("fill", C.muted).style("font-size", "13px").text(`D_e = ${MORSE.De} kcal/mol`);
    const rs = d3.range(1.0, 3.001, 0.01);
    const pts = (f) => rs.map((r) => [x(r), y(Math.min(f(r), 85))]).filter((p) => p[1] >= m.t);
    const pathH = svg.append("path").attr("d", d3.line()(pts(harm))).attr("fill", "none").attr("stroke", C.blue).attr("stroke-width", 2.5).attr("stroke-dasharray", "6 5");
    const pathM = svg.append("path").attr("d", d3.line()(pts(morse))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3.5);
    svg.append("text").attr("x", x(1.12)).attr("y", y(70)).attr("fill", C.blue).style("font-size", "14px").text("harmonic");
    svg.append("text").attr("x", x(2.05)).attr("y", y(45)).attr("fill", C.yellow).style("font-size", "14px").text("Morse");
    const asym = svg.append("g").attr("opacity", 0);
    [MORSE.re - 0.1, MORSE.re + 0.1].forEach((r) => {
      asym.append("circle").attr("cx", x(r)).attr("cy", y(morse(r))).attr("r", 6).attr("fill", C.yellow);
      asym.append("text").attr("x", x(r)).attr("y", y(morse(r)) - 14).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "13px")
        .text(`${morse(r).toFixed(1)}`);
    });
    asym.append("text").attr("x", x(1.62)).attr("y", y(22)).attr("fill", C.ink2).style("font-size", "13px")
      .text(`±0.1 Å from r_e = ${MORSE.re} Å: ${morse(MORSE.re - 0.1).toFixed(1)} vs ${morse(MORSE.re + 0.1).toFixed(1)} kcal/mol`);
    return { set, zone, asym, pathM, pathH };
  }, (h, step) => {
    h.set(...(step === 0 ? ["O–O as an anharmonic bond", "Morse parameters set in the paper's script: D_e 60 kcal/mol, a 2.0 Å⁻¹"]
      : step === 1 ? ["Stretching and compressing are not symmetric", "the restoring force weakens as the bond lengthens"]
      : ["Two states, one distinction", "bonded or cleaved: a binary coordinate, ln 2 per recorded change"]));
    h.asym.transition().duration(T).attr("opacity", step === 1 ? 1 : 0);
    h.zone.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 4. No exact return in ten trajectories ────────────────────────────────
export function ReturnChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 90, b: 70 };
    const x = d3.scaleBand().domain(d3.range(RETURN_MIN.length)).range([m.l, W - m.r]).padding(0.3);
    const y = d3.scaleLog().domain([1e-8, 1e-3]).range([H - m.b, m.t]);
    const set = caption(svg);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickFormat((i) => i + 1)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(logFmt)), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "trajectory", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "closest return to the start (Å)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const bars = svg.append("g").selectAll("rect").data(RETURN_MIN).join("rect")
      .attr("x", (d, i) => x(i)).attr("width", x.bandwidth()).attr("y", H - m.b).attr("height", 0).attr("fill", C.yellow).attr("rx", 3);
    const minI = d3.minIndex(RETURN_MIN);
    const note = svg.append("text").attr("x", x(minI) + x.bandwidth() / 2).attr("y", y(RETURN_MIN[minI]) - 12).attr("text-anchor", "middle")
      .attr("fill", C.ink).style("font-size", "13px").attr("opacity", 0).text(`${fmtSci(RETURN_MIN[minI])} Å`);
    return { set, bars, y, m, note, minI };
  }, (h, step, _p, first) => {
    h.bars.transition().duration(first ? 0 : 900).delay((d, i) => i * 60)
      .attr("y", (d) => h.y(d)).attr("height", (d) => H - h.m.b - h.y(d))
      .attr("opacity", (d, i) => (step === 1 && i !== h.minI ? 0.3 : 1));
    h.note.transition().duration(T).attr("opacity", step === 1 ? 1 : 0);
    h.set(...(step === 0 ? ["Ten trajectories on the Morse potential", "closest approach to the starting point in 1000 integration steps"]
      : ["None returns exactly", "the closest comes within 2×10⁻⁷ Å; exact return is not expected for this motion"]));
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 5. Rates from the d_C rule, and the isotope effect ────────────────────
export function RateRule({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 90, b: 70 };
    const set = caption(svg);
    const g1 = svg.append("g");
    const x = d3.scaleBand().domain(DC_RULE.map((d) => d.d)).range([m.l, W - m.r]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 10]).range([H - m.b, m.t]);
    axis(g1.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)));
    axis(g1.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((v) => `10${"⁰¹²³⁴⁵⁶⁷⁸⁹"[v] ?? "¹⁰"}`)), { grid: W - m.r - m.l, orient: "y" });
    label(g1, "number of simultaneous changes d_C", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(g1, "rate k (s⁻¹)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const bars = g1.append("g").selectAll("rect").data(DC_RULE).join("rect")
      .attr("x", (d) => x(d.d)).attr("width", x.bandwidth()).attr("y", (d) => y(d.logk)).attr("height", (d) => H - m.b - y(d.logk))
      .attr("fill", C.yellow).attr("rx", 3);
    const tags = g1.append("g").selectAll("text").data(DC_RULE.slice(0, 2)).join("text")
      .attr("x", (d) => x(d.d) + x.bandwidth() / 2).attr("y", (d) => y(d.logk) - 10).attr("text-anchor", "middle")
      .attr("fill", C.ink).style("font-size", "14px").attr("opacity", 0).text((d) => (d.d === 1 ? "concerted" : "sequential"));
    // KIE panel
    const g2 = svg.append("g").attr("opacity", 0);
    const kx = d3.scaleBand().domain(["concerted", "sequential"]).range([m.l + 40, W - m.r - 160]).padding(0.35);
    const ky = d3.scaleLinear().domain([0, 7]).range([H - m.b, m.t]);
    axis(g2.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(kx)));
    axis(g2.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(ky).ticks(7)), { grid: W - m.r - m.l, orient: "y" });
    label(g2, "kinetic isotope effect k_H/k_D", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    g2.append("rect").attr("x", m.l).attr("width", W - m.r - m.l).attr("y", ky(KIE.lit[1])).attr("height", ky(KIE.lit[0]) - ky(KIE.lit[1]))
      .attr("fill", C.violet).attr("opacity", 0.25);
    g2.append("text").attr("x", W - m.r - 6).attr("y", ky(KIE.lit[1]) - 8).attr("text-anchor", "end").attr("fill", C.violet).style("font-size", "13px")
      .text(`cited range ${KIE.lit[0]}–${KIE.lit[1]}`);
    g2.selectAll("rect.k").data([["concerted", KIE.concerted], ["sequential", KIE.sequential]]).join("rect").attr("class", "k")
      .attr("x", (d) => kx(d[0])).attr("width", kx.bandwidth()).attr("y", (d) => ky(d[1])).attr("height", (d) => H - m.b - ky(d[1]))
      .attr("fill", C.yellow).attr("rx", 3);
    g2.selectAll("text.k").data([["concerted", KIE.concerted], ["sequential", KIE.sequential]]).join("text").attr("class", "k")
      .attr("x", (d) => kx(d[0]) + kx.bandwidth() / 2).attr("y", (d) => ky(d[1]) - 10).attr("text-anchor", "middle")
      .attr("fill", C.ink).style("font-size", "15px").text((d) => d[1].toFixed(1));
    return { set, g1, g2, bars, tags };
  }, (h, step) => {
    h.set(...(step === 0 ? ["One rule for every step: k = 10^(10 − d_C) s⁻¹", "the paper's rate law, a set rule"]
      : step === 1 ? ["Concerted against sequential proton-coupled transfer", "one change at a time costs one decade: the 10× is the rule itself"]
      : ["The isotope effect separates the two", "values typed into the script, against the cited measurement"]));
    h.g1.transition().duration(T).attr("opacity", step === 2 ? 0 : 1);
    h.g2.transition().duration(T).attr("opacity", step === 2 ? 1 : 0);
    h.bars.transition().duration(T).attr("opacity", (d) => (step === 1 && d.d > 2 ? 0.2 : 1));
    h.tags.transition().duration(T).attr("opacity", step === 1 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 6. Lifetime against temperature ───────────────────────────────────────
const HBAR = 1.054571817e-34, KB = 1.380649e-23, R = 8.314462618;
const tau = (T) => (HBAR / (KB * T)) * Math.exp((LIFETIME_EA * 4184) / (R * T)) * 1000; // ms

export function LifetimeChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 90, r: 40, t: 90, b: 70 };
    const x = d3.scaleLinear().domain([180, 360]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([1e-4, 1e4]).range([H - m.b, m.t]);
    const set = caption(svg);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(9)));
    axis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(8, "~g")), { grid: W - m.r - m.l, orient: "y" });
    label(svg, "temperature (K)", { x: (m.l + W - m.r) / 2, y: H - 22 });
    label(svg, "Compound I lifetime (ms)", { x: 26, y: (m.t + H - m.b) / 2, rotate: true });
    const pts = d3.range(180, 360.1, 2).map((t) => [x(t), y(Math.max(1e-4, Math.min(1e4, tau(t))))]);
    const path = svg.append("path").attr("d", d3.line()(pts)).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 3);
    svg.append("text").attr("x", x(262)).attr("y", y(tau(262)) - 14).attr("fill", C.yellow).style("font-size", "13px").text(`E_a = ${LIFETIME_EA} kcal/mol`);
    const marks = LIFETIME_PTS.map((p) => {
      const g = svg.append("g").attr("opacity", 0);
      g.append("line").attr("x1", x(p.T)).attr("x2", x(p.T)).attr("y1", y(p.model)).attr("y2", y(p.lit)).attr("stroke", C.ink2).attr("stroke-dasharray", "3 4");
      g.append("circle").attr("cx", x(p.T)).attr("cy", y(p.model)).attr("r", 7).attr("fill", C.yellow);
      g.append("circle").attr("cx", x(p.T)).attr("cy", y(p.lit)).attr("r", 7).attr("fill", p.estimate ? "none" : C.violet).attr("stroke", C.violet).attr("stroke-width", 2.5);
      const right = p.T < 300;
      g.append("text").attr("x", x(p.T) + (right ? 14 : -14)).attr("y", y(p.lit) - 12).attr("text-anchor", right ? "start" : "end").attr("fill", C.violet).style("font-size", "13px")
        .text(`${p.lit} ms — ${p.litNote}`);
      g.append("text").attr("x", x(p.T) + (right ? 14 : -14)).attr("y", y(p.model) + 20).attr("text-anchor", right ? "start" : "end").attr("fill", C.yellow).style("font-size", "13px")
        .text(`model ${p.model >= 1 ? p.model.toFixed(0) + " ms" : (p.model * 1000).toFixed(1) + " µs"}`);
      return g;
    });
    return { set, marks, path };
  }, (h, step) => {
    h.set(...(step === 0 ? ["An Eyring-type lifetime with one activation energy", "τ = (ħ/k_BT)·exp(E_a/RT)"]
      : step === 1 ? ["At 193 K: within a factor of two", "the model's 113 ms against 210 ms for CYP119"]
      : ["At body temperature there is no measurement to compare", "the paper's 1 ms is an estimate; the model gives 1.4 µs"]));
    h.marks[0].transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.marks[1].transition().duration(T).attr("opacity", step >= 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 7. Oxidation potential ────────────────────────────────────────────────
export function PotentialChart({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 60, r: 40, t: 100 };
    const x = d3.scaleLinear().domain([0, 1.0]).range([m.l, W - m.r]);
    const set = caption(svg);
    const by = 150, bh = 60;
    axis(svg.append("g").attr("transform", `translate(0,${by + bh + 10})`).call(d3.axisBottom(x).ticks(10)));
    label(svg, "potential (V vs NHE)", { x: (m.l + W - m.r) / 2, y: by + bh + 50 });
    const cols = [C.blue, C.aqua, C.yellow, C.magenta];
    let acc = 0;
    const parts = POTENTIAL.parts.map(([name, dm, v], i) => { const s = acc; acc += v; return { name, dm, v, s, c: cols[i] }; });
    const segs = svg.append("g").selectAll("rect").data(parts).join("rect")
      .attr("x", (d) => x(d.s)).attr("y", by).attr("height", bh).attr("width", 0).attr("fill", (d) => d.c).attr("stroke", C.bg);
    const segL = svg.append("g").selectAll("text").data(parts).join("text")
      .attr("x", (d) => x(d.s + d.v / 2)).attr("y", (d, i) => by - 12 - (i % 2) * 18).attr("text-anchor", "middle")
      .attr("fill", (d) => d.c).style("font-size", "12px").attr("opacity", 0).text((d) => `${d.name} ${d.v.toFixed(2)}`);
    const tot = svg.append("text").attr("x", x(POTENTIAL.total) + 8).attr("y", by + bh / 2 + 5).attr("fill", C.ink).style("font-size", "15px")
      .attr("opacity", 0).text(`${POTENTIAL.total.toFixed(2)} V`);
    const lit = svg.append("g").attr("opacity", 0);
    lit.append("line").attr("x1", x(POTENTIAL.lit)).attr("x2", x(POTENTIAL.lit)).attr("y1", by - 50).attr("y2", by + bh + 10).attr("stroke", C.violet).attr("stroke-width", 3);
    lit.append("text").attr("x", x(POTENTIAL.lit)).attr("y", by - 56).attr("text-anchor", "middle").attr("fill", C.violet).style("font-size", "13px").text(`${POTENTIAL.lit} V, as cited`);
    // n_eff sweep
    const g2 = svg.append("g").attr("opacity", 0);
    const top = 330, bot = 500;
    const nx = d3.scaleLinear().domain([1, 10]).range([m.l + 30, W - m.r]);
    const ny = d3.scaleLinear().domain([0, 1.1]).range([bot, top]);
    axis(g2.append("g").attr("transform", `translate(0,${bot})`).call(d3.axisBottom(nx).ticks(10)));
    axis(g2.append("g").attr("transform", `translate(${m.l + 30},0)`).call(d3.axisLeft(ny).ticks(4)));
    label(g2, "n_eff, the multiplier set in the script", { x: (m.l + W - m.r) / 2, y: bot + 40 });
    g2.append("path").attr("d", d3.line()(d3.range(1, 10.01, 0.1).map((n) => [nx(n), ny(n * POTENTIAL.perNeff)]))).attr("fill", "none").attr("stroke", C.yellow).attr("stroke-width", 2.5);
    g2.append("line").attr("x1", nx(1)).attr("x2", nx(10)).attr("y1", ny(POTENTIAL.lit)).attr("y2", ny(POTENTIAL.lit)).attr("stroke", C.violet).attr("stroke-dasharray", "4 4");
    const need = POTENTIAL.lit / POTENTIAL.perNeff;
    g2.append("circle").attr("cx", nx(POTENTIAL.neff)).attr("cy", ny(POTENTIAL.total)).attr("r", 6).attr("fill", C.yellow);
    g2.append("text").attr("x", nx(POTENTIAL.neff) + 10).attr("y", ny(POTENTIAL.total) + 18).attr("fill", C.yellow).style("font-size", "13px").text("n_eff = 5 (set)");
    g2.append("circle").attr("cx", nx(need)).attr("cy", ny(POTENTIAL.lit)).attr("r", 6).attr("fill", "none").attr("stroke", C.violet).attr("stroke-width", 2);
    g2.append("text").attr("x", nx(need) - 10).attr("y", ny(POTENTIAL.lit) - 12).attr("text-anchor", "end").attr("fill", C.violet).style("font-size", "13px")
      .text(`0.9 V would need n_eff ≈ ${need.toFixed(1)}`);
    return { set, segs, segL, tot, lit, g2, x };
  }, (h, step, _p, first) => {
    h.set(...(step === 0 ? ["Four contributions, each a set ΔM", "E = (k_BT/e)·n_eff·ΣΔM at 310 K"]
      : step === 1 ? ["0.54 V computed against 0.9 V cited", "the paper's text states agreement; its own proof says qualitative"]
      : ["What the gap depends on", "n_eff is not derived; it scales the whole bar"]));
    h.segs.transition().duration(first ? 0 : 900).delay((d, i) => i * 120).attr("width", (d) => h.x(d.s + d.v) - h.x(d.s));
    h.segL.transition().duration(T).attr("opacity", 1);
    h.tot.transition().delay(500).duration(T).attr("opacity", 1);
    h.lit.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.g2.transition().duration(T).attr("opacity", step >= 2 ? 1 : 0);
  }, step);
  return <Svg svgRef={ref} />;
}

// ── 8. What the observable table contains ─────────────────────────────────
export function ObservableTable({ step }) {
  const ref = useChart((svg) => {
    const m = { l: 200, r: 50, t: 100, b: 70 };
    const x = d3.scaleLog().domain([5e-4, 4]).range([m.l, W - m.r]);
    const y = d3.scaleBand().domain(OBSERVABLES.map((o) => o.name)).range([m.t, H - m.b]).padding(0.35);
    const set = caption(svg);
    axis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([0.001, 0.01, 0.1, 0.5, 1, 2]).tickFormat(d3.format("~g"))), { grid: H - m.b - m.t });
    label(svg, "value ÷ literature value", { x: (m.l + W - m.r) / 2, y: H - 22 });
    svg.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", m.t - 10).attr("y2", H - m.b).attr("stroke", C.violet).attr("stroke-width", 2);
    svg.append("text").attr("x", x(1)).attr("y", m.t - 16).attr("text-anchor", "middle").attr("fill", C.violet).style("font-size", "13px").text("literature");
    svg.append("g").selectAll("text").data(OBSERVABLES).join("text").attr("x", m.l - 14).attr("y", (o) => y(o.name) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "14px").text((o) => o.name);
    const cy = (o) => y(o.name) + y.bandwidth() / 2;
    const listed = svg.append("g").selectAll("circle").data(OBSERVABLES).join("circle")
      .attr("cx", (o) => x(o.listed / o.lit)).attr("cy", cy).attr("r", 8).attr("fill", "none").attr("stroke", C.ink2).attr("stroke-width", 2.5);
    const comp = OBSERVABLES.filter((o) => o.computed != null);
    const links = svg.append("g").selectAll("line").data(comp).join("line")
      .attr("x1", (o) => x(o.listed / o.lit)).attr("x2", (o) => x(o.listed / o.lit)).attr("y1", cy).attr("y2", cy)
      .attr("stroke", C.yellow).attr("stroke-width", 2);
    const cdots = svg.append("g").selectAll("circle").data(comp).join("circle")
      .attr("cx", (o) => x(o.listed / o.lit)).attr("cy", cy).attr("r", 7).attr("fill", C.yellow).attr("opacity", 0);
    const ctext = svg.append("g").selectAll("text").data(comp).join("text")
      .attr("x", (o) => x(o.computed / o.lit) + (o.computed / o.lit > 0.1 ? -14 : 12)).attr("y", (o) => cy(o) + 5)
      .attr("text-anchor", (o) => (o.computed / o.lit > 0.1 ? "end" : "start")).attr("fill", C.yellow).style("font-size", "12px").attr("opacity", 0)
      .text((o) => `computed ${o.computed} ${o.unit}`);
    return { set, listed, links, cdots, ctext, x };
  }, (h, step) => {
    h.set(...(step === 0 ? ["The paper's table of eight observables", "open rings: the values in its 'predicted' column"]
      : step === 1 ? ["Where the scripts compute a value, it is not the listed one", "filled: what the model actually returns"]
      : ["Five rows are not computed at all", "they are literature values entered as predictions"]));
    h.links.transition().duration(900).attr("x2", (o) => (step >= 1 ? h.x(o.computed / o.lit) : h.x(o.listed / o.lit)));
    h.cdots.transition().duration(900).attr("cx", (o) => (step >= 1 ? h.x(o.computed / o.lit) : h.x(o.listed / o.lit))).attr("opacity", step >= 1 ? 1 : 0);
    h.ctext.transition().duration(T).attr("opacity", step >= 1 ? 1 : 0);
    h.listed.transition().duration(T).attr("stroke", (o) => (step === 2 && o.computed == null ? C.red : C.ink2))
      .attr("opacity", (o) => (step === 2 && o.computed != null ? 0.3 : 1));
  }, step);
  return <Svg svgRef={ref} />;
}
