// =====================================================================
//  ShakespeareChart — one D3 renderer, switched by chart `name`.
//  Charts mirror the monograph's figure panels, drawn live.
// =====================================================================
import { useEffect, useRef } from "react";
import * as d3 from "d3";

const PRIMARY = "#58E6D9";
const ACCENT = "#B63E96";
const DIM = "#6b7280";
const TEXT = "#cbd5e1";

export default function ShakespeareChart({ name, data, width = 360, height = 300 }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !data) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");

    const draw = DRAW[name];
    if (draw) draw(svg, data, width, height);
  }, [name, data, width, height]);

  return (
    <div className="rounded-md border border-neutral-700 bg-[#151515] p-2">
      <div className="mb-1 px-1 text-[11px] uppercase tracking-wider text-neutral-400">
        {TITLES[name] ?? name}
      </div>
      <svg ref={ref} />
    </div>
  );
}

const TITLES = {
  "contact-map": "Contact map",
  "order-parameter": "Order parameter r(t)",
  "seven-state-orbit": "Seven-state closed orbit",
  "dm-bars": "Partition-depth ΔM per transition",
  "depth-ladder": "Categorical address by depth",
  "s-entropy": "S-entropy space [0,1]³",
  "et-trace": "Electron-transfer trace",
  "spin-orbit": "Spin: s_orbital conserved · S_total varies",
  rebound: "Bond-order coordinate (Fe=O → C–O)",
};

const DRAW = {
  // -------------------------------------------------- contact map
  "contact-map": (svg, { N, cells }, W, H) => {
    const m = 24;
    const size = Math.min(W, H) - m * 2;
    const s = size / N;
    const g = svg.append("g").attr("transform", `translate(${m},${m})`);
    const color = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    g.selectAll("rect")
      .data(cells)
      .join("rect")
      .attr("x", (d) => d.j * s)
      .attr("y", (d) => d.i * s)
      .attr("width", s + 0.5)
      .attr("height", s + 0.5)
      .attr("fill", (d) => color(d.v));
    axisLabels(svg, W, H, "residue i", "residue j");
  },

  // -------------------------------------------------- order parameter
  "order-parameter": (svg, { pts, threshold, finalR }, W, H) => {
    const m = { l: 40, r: 12, t: 14, b: 30 };
    const x = d3.scaleLinear([0, 1], [m.l, W - m.r]);
    const y = d3.scaleLinear([0, 1], [H - m.b, m.t]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5)).call(styleAxis);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).call(styleAxis);
    // threshold
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(threshold)).attr("y2", y(threshold))
      .attr("stroke", ACCENT).attr("stroke-dasharray", "4 3").attr("opacity", 0.7);
    svg.append("text").attr("x", W - m.r).attr("y", y(threshold) - 4).attr("text-anchor", "end")
      .attr("fill", ACCENT).attr("font-size", 10).text("r_c = 0.8");
    const line = d3.line().x((d) => x(d.t)).y((d) => y(d.r)).curve(d3.curveMonotoneX);
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", PRIMARY).attr("stroke-width", 2).attr("d", line);
    svg.append("text").attr("x", W - m.r).attr("y", y(finalR) - 6).attr("text-anchor", "end")
      .attr("fill", PRIMARY).attr("font-size", 10).text(`⟨r⟩ = ${finalR}`);
  },

  // -------------------------------------------------- seven-state orbit
  "seven-state-orbit": (svg, { nodes, edges, DM_sum }, W, H) => {
    const cx = W / 2, cy = H / 2 + 4, R = Math.min(W, H) / 2 - 46;
    const pos = (a) => [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    const wScale = d3.scaleLinear(d3.extent(edges, (e) => e.dm), [1, 6]);
    edges.forEach((e) => {
      const [x1, y1] = pos(nodes[e.from].angle);
      const [x2, y2] = pos(nodes[e.to].angle);
      svg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", e.to === 0 ? ACCENT : PRIMARY).attr("stroke-width", wScale(e.dm)).attr("opacity", 0.75);
    });
    nodes.forEach((n) => {
      const [x, y] = pos(n.angle);
      svg.append("circle").attr("cx", x).attr("cy", y).attr("r", 13)
        .attr("fill", n.i === 6 ? ACCENT : "#0f766e").attr("stroke", PRIMARY).attr("stroke-width", 1.5);
      svg.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("fill", "#fff").attr("font-size", 11).attr("font-weight", 700).text(n.i + 1);
      const lx = cx + (R + 26) * Math.cos(n.angle);
      const ly = cy + (R + 26) * Math.sin(n.angle);
      svg.append("text").attr("x", lx).attr("y", ly).attr("text-anchor", "middle")
        .attr("fill", TEXT).attr("font-size", 8)
        .text(n.label.split(" ")[0]);
    });
    svg.append("text").attr("x", cx).attr("y", cy).attr("text-anchor", "middle")
      .attr("fill", DIM).attr("font-size", 10).text(`ΣΔM = ${DM_sum}`);
  },

  // -------------------------------------------------- dm bars
  "dm-bars": (svg, { bars, sum, critical, domainMax, ylabel }, W, H) => {
    const m = { l: 118, r: 40, t: 10, b: 24 };
    const x = d3.scaleLinear([0, domainMax ?? d3.max(bars, (b) => b.v) * 1.15], [m.l, W - m.r]);
    const y = d3.scaleBand(bars.map((b) => b.label), [m.t, H - m.b]).padding(0.25);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y)).call(styleAxis)
      .selectAll("text").attr("font-size", 8);
    svg.selectAll("rect").data(bars).join("rect")
      .attr("x", m.l).attr("y", (b) => y(b.label)).attr("height", y.bandwidth())
      .attr("width", (b) => x(b.v) - m.l).attr("fill", (b) => (b.highlight ? ACCENT : PRIMARY)).attr("opacity", 0.85);
    svg.selectAll(".val").data(bars).join("text").attr("class", "val")
      .attr("x", (b) => x(b.v) + 4).attr("y", (b) => y(b.label) + y.bandwidth() / 2 + 3)
      .attr("fill", TEXT).attr("font-size", 9).text((b) => b.v);
    if (critical != null) {
      svg.append("line").attr("x1", x(critical)).attr("x2", x(critical)).attr("y1", m.t).attr("y2", H - m.b)
        .attr("stroke", "#f59e0b").attr("stroke-dasharray", "3 2").attr("opacity", 0.6);
    }
    if (sum != null) {
      svg.append("text").attr("x", W - m.r).attr("y", H - 6).attr("text-anchor", "end")
        .attr("fill", DIM).attr("font-size", 9).text(`Σ = ${sum}`);
    }
  },

  // -------------------------------------------------- depth ladder
  "depth-ladder": (svg, { trits, marks }, W, H) => {
    const m = { l: 30, r: 16, t: 20, b: 26 };
    const x = d3.scaleBand(trits.map((t) => t.i), [m.l, W - m.r]).padding(0.2);
    const color = d3.scaleOrdinal([0, 1, 2], [PRIMARY, ACCENT, "#f59e0b"]);
    svg.selectAll("rect").data(trits).join("rect")
      .attr("x", (d) => x(d.i)).attr("y", (d) => m.t + (2 - d.t) * ((H - m.t - m.b) / 3))
      .attr("width", x.bandwidth()).attr("height", (H - m.t - m.b) / 3 - 2)
      .attr("fill", (d) => color(d.t)).attr("rx", 1);
    svg.selectAll(".tl").data(trits).join("text").attr("class", "tl")
      .attr("x", (d) => x(d.i) + x.bandwidth() / 2).attr("y", H - m.b + 12).attr("text-anchor", "middle")
      .attr("fill", TEXT).attr("font-size", 9).text((d) => d.t);
    marks.forEach((mk) => {
      const gx = x(mk.depth - 1);
      if (gx == null) return;
      svg.append("line").attr("x1", gx + x.bandwidth() + x.step() * 0.1).attr("x2", gx + x.bandwidth() + x.step() * 0.1)
        .attr("y1", m.t - 6).attr("y2", H - m.b).attr("stroke", DIM).attr("stroke-dasharray", "2 2").attr("opacity", 0.5);
      svg.append("text").attr("x", gx).attr("y", m.t - 8).attr("fill", DIM).attr("font-size", 8).text(`d${mk.depth} ${mk.label}`);
    });
  },

  // -------------------------------------------------- s-entropy (2D proj)
  "s-entropy": (svg, { pts }, W, H) => {
    const m = 34;
    const x = d3.scaleLinear([0, 1], [m, W - m]);
    const y = d3.scaleLinear([0, 1], [H - m, m]);
    svg.append("g").attr("transform", `translate(0,${H - m})`).call(d3.axisBottom(x).ticks(4)).call(styleAxis);
    svg.append("g").attr("transform", `translate(${m},0)`).call(d3.axisLeft(y).ticks(4)).call(styleAxis);
    const z = d3.scaleSequential(d3.interpolatePlasma).domain([0, 1]);
    svg.selectAll("circle").data(pts).join("circle")
      .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", (d) => (d.big ? 8 : 5))
      .attr("fill", (d) => z(d.z)).attr("stroke", (d) => (d.big ? "#fff" : "none")).attr("stroke-width", 1.5);
    svg.selectAll(".lb").data(pts).join("text").attr("class", "lb")
      .attr("x", (d) => x(d.x) + 8).attr("y", (d) => y(d.y) + 3).attr("fill", TEXT).attr("font-size", 8).text((d) => d.label);
    axisLabels(svg, W, H, "S_k (hydrophobicity)", "S_t (volume)");
  },

  // -------------------------------------------------- spin-orbit
  "spin-orbit": (svg, { orbital, total }, W, H) => {
    const m = { l: 40, r: 14, t: 14, b: 28 };
    const x = d3.scalePoint(total.map((d) => d.state), [m.l, W - m.r]).padding(0.5);
    const y = d3.scaleLinear([0, 2.6], [H - m.b, m.t]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)).call(styleAxis);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).call(styleAxis);
    const lineT = d3.line().x((d) => x(d.state)).y((d) => y(d.v));
    const lineO = d3.line().x((d) => x(d.state)).y((d) => y(d.v));
    svg.append("path").datum(total).attr("fill", "none").attr("stroke", ACCENT).attr("stroke-width", 2).attr("d", lineT);
    svg.selectAll(".t").data(total).join("circle").attr("class", "t").attr("cx", (d) => x(d.state)).attr("cy", (d) => y(d.v)).attr("r", 4).attr("fill", ACCENT);
    svg.append("path").datum(orbital).attr("fill", "none").attr("stroke", PRIMARY).attr("stroke-width", 2).attr("stroke-dasharray", "4 3").attr("d", lineO);
    svg.append("text").attr("x", W - m.r).attr("y", y(2.5)).attr("text-anchor", "end").attr("fill", ACCENT).attr("font-size", 9).text("S_total");
    svg.append("text").attr("x", W - m.r).attr("y", y(0.5) - 4).attr("text-anchor", "end").attr("fill", PRIMARY).attr("font-size", 9).text("s_orbital = ½ (conserved)");
    svg.append("text").attr("x", W / 2).attr("y", H - 4).attr("text-anchor", "middle").attr("fill", DIM).attr("font-size", 9).text("catalytic state");
  },

  // -------------------------------------------------- rebound
  rebound: (svg, { pts }, W, H) => {
    const m = { l: 40, r: 14, t: 14, b: 28 };
    const x = d3.scaleLinear([0, 1], [m.l, W - m.r]);
    const y = d3.scaleLinear([0, 1], [H - m.b, m.t]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4)).call(styleAxis);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4)).call(styleAxis);
    const l1 = d3.line().x((d) => x(d.x)).y((d) => y(d.feO)).curve(d3.curveMonotoneX);
    const l2 = d3.line().x((d) => x(d.x)).y((d) => y(d.cO)).curve(d3.curveMonotoneX);
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", "#f59e0b").attr("stroke-width", 2).attr("d", l1);
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", PRIMARY).attr("stroke-width", 2).attr("d", l2);
    svg.append("text").attr("x", m.l + 6).attr("y", y(0.9)).attr("fill", "#f59e0b").attr("font-size", 9).text("Fe=O");
    svg.append("text").attr("x", W - m.r - 4).attr("y", y(0.9)).attr("text-anchor", "end").attr("fill", PRIMARY).attr("font-size", 9).text("C–O");
    svg.append("text").attr("x", W / 2).attr("y", H - 4).attr("text-anchor", "middle").attr("fill", DIM).attr("font-size", 9).text("reaction coordinate");
  },

  // -------------------------------------------------- et trace
  "et-trace": (svg, { pts, timescale }, W, H) => {
    const m = { l: 60, r: 20, t: 20, b: 30 };
    const x = d3.scaleLinear(d3.extent(pts, (p) => p.t), [m.l, W - m.r]);
    const y = d3.scalePoint(pts.map((p) => p.label), [m.t, H - m.b]).padding(0.5);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4)).call(styleAxis);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y)).call(styleAxis);
    const line = d3.line().x((p) => x(p.t)).y((p) => y(p.label)).curve(d3.curveStepAfter);
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", PRIMARY).attr("stroke-width", 2).attr("d", line);
    svg.selectAll("circle").data(pts).join("circle").attr("cx", (p) => x(p.t)).attr("cy", (p) => y(p.label))
      .attr("r", 5).attr("fill", ACCENT).attr("stroke", "#fff").attr("stroke-width", 1);
    svg.append("text").attr("x", (W) / 2).attr("y", H - 4).attr("text-anchor", "middle").attr("fill", DIM)
      .attr("font-size", 9).text(`time (fs) · ${timescale}`);
  },
};

function styleAxis(g) {
  g.selectAll("text").attr("fill", "#94a3b8").attr("font-size", 9);
  g.selectAll("line").attr("stroke", "#374151");
  g.selectAll("path").attr("stroke", "#374151");
}

function axisLabels(svg, W, H, xl, yl) {
  svg.append("text").attr("x", W / 2).attr("y", H - 4).attr("text-anchor", "middle").attr("fill", DIM).attr("font-size", 9).text(xl);
  svg.append("text").attr("transform", `translate(10,${H / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("fill", DIM).attr("font-size", 9).text(yl);
}
