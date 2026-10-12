import { C } from "../core";

export const W = 760;
export const H = 560;

// The chart surface: a fixed viewBox scaled into whatever box the scene gives.
export function Svg({ svgRef, w = W, h = H }) {
  return <svg ref={svgRef} viewBox={`0 0 ${w} ${h}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet" style={{ maxHeight: "100%" }} />;
}

// Arrowhead markers, one per colour, defined once per chart.
export function arrows(svg, colors = [C.line, C.ink2, C.orange, C.aqua, C.yellow, C.violet]) {
  const defs = svg.append("defs");
  colors.forEach((c) => {
    defs.append("marker").attr("id", `ar-${c.slice(1)}`).attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", c);
  });
  return (c) => `url(#ar-${c.slice(1)})`;
}

// A short caption line at the top of a chart, rewritten per step.
export function caption(svg, x = 24, y = 34) {
  const t = svg.append("text").attr("x", x).attr("y", y).attr("fill", C.ink).style("font-size", "17px").style("font-weight", 600);
  const s = svg.append("text").attr("x", x).attr("y", y + 22).attr("fill", C.muted).style("font-size", "13px");
  return (title, sub = "") => { t.text(title); s.text(sub); };
}

// Log-axis labels: 10ⁿ at powers of ten, nothing at the minor ticks.
export const logFmt = (v) => {
  const e = Math.log10(v);
  if (Math.abs(e - Math.round(e)) > 1e-9) return "";
  return `10${String(Math.round(e)).replace(/-/g, "⁻").replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c])}`;
};

export const fmtSci = (v, d = 1) => {
  if (!Number.isFinite(v)) return "—";
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  const sup = String(e).replace(/-/g, "⁻").replace(/\d/g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]);
  return `${m.toFixed(d)}×10${sup}`;
};
