import { useEffect, useRef } from "react";
import * as d3 from "d3";

// Runs `draw(svg)` on a cleared <svg> whenever deps change. `draw` may return
// a cleanup function (stop timers, cancel transitions).
export default function useD3(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return undefined;
    const svg = d3.select(ref.current);
    svg.selectAll("*").interrupt().remove();
    const cleanup = draw(svg);
    return () => {
      svg.selectAll("*").interrupt();
      if (typeof cleanup === "function") cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

// Recessive axes in the talk's ink.
export function styleAxis(g, { grid = false, width = 0, height = 0, orient = "x" } = {}) {
  g.selectAll("path.domain").attr("stroke", "#5a5955");
  g.selectAll(".tick line").attr("stroke", "#5a5955");
  g.selectAll(".tick text").attr("fill", "#c3c2b7").style("font-size", "15px");
  if (grid) {
    g.selectAll(".tick line")
      .clone()
      .attr("stroke", "#2a2a27")
      .attr(orient === "x" ? "y2" : "x2", orient === "x" ? -height : width);
  }
  return g;
}

export function axisLabel(svg, text, { x, y, rotate = false, anchor = "middle" }) {
  return svg
    .append("text")
    .attr("x", x)
    .attr("y", y)
    .attr("transform", rotate ? `rotate(-90, ${x}, ${y})` : null)
    .attr("text-anchor", anchor)
    .attr("fill", "#c3c2b7")
    .style("font-size", "16px")
    .text(text);
}
