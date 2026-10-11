import * as d3 from "d3";
import { C, fmt } from "../theme";
import useD3, { styleAxis, axisLabel } from "../useD3";
import { SlideFrame, Reveal, Statement, Legend } from "../ui";
import { rng, maxFlow } from "../util";
import DATA from "@/data/talk/derivation.json";

// ------------------------------------------------------ 5 · individuation
function GrowingGraph({ real }) {
  const W = 1456, H = 580;
  const ref = useD3((svg) => {
    const GW = 800, GH = 560;
    const r = rng(real ? 99 : 7);
    const weight = () => (real ? -1.5 * Math.log(1 - r()) : 1 + Math.floor(r() * 5));
    const nodes = [{ id: "m", fx: GW / 2, fy: GH - 40 }, { id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }];
    const edges = [];
    const addEdge = (a, b) => edges.push({ a, b, w: weight(), source: a, target: b });
    [0, 1, 2, 3].forEach((i) => addEdge(i, "m"));
    addEdge(0, 1); addEdge(1, 2); addEdge(2, 3); addEdge(0, 3);
    let next = 4;

    const gLinks = svg.append("g"), gNodes = svg.append("g");
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d) => d.id).distance((e) => (e.b === "m" || e.a === "m" ? 210 : 70)).strength(0.25))
      .force("charge", d3.forceManyBody().strength(-170))
      .force("x", d3.forceX(GW / 2).strength(0.04))
      .force("y", d3.forceY(GH / 2 - 40).strength(0.04))
      .force("collide", d3.forceCollide(16));

    // right-hand charts
    const ox = GW + 40, cw = W - ox - 10, chH = 130, y2 = 250;
    const hist = [];
    const xs = d3.scaleLinear().domain([0, 60]).range([ox + 50, ox + cw]);
    const ys1 = d3.scaleLinear().domain([0, 80]).range([28 + chH, 28]);
    const ys2 = d3.scaleLog().domain(real ? [1e-4, 10] : [0.3, 10]).range([y2 + chH, y2]);
    const ax1 = styleAxis(svg.append("g").attr("transform", `translate(${ox + 50},0)`).call(d3.axisLeft(ys1).ticks(4)));
    styleAxis(svg.append("g").attr("transform", `translate(0,${28 + chH})`).call(d3.axisBottom(xs).ticks(4)));
    styleAxis(svg.append("g").attr("transform", `translate(${ox + 50},0)`).call(d3.axisLeft(ys2).ticks(4, "~g")));
    styleAxis(svg.append("g").attr("transform", `translate(0,${y2 + chH})`).call(d3.axisBottom(xs).ticks(4)));
    svg.append("text").attr("x", ox + 56).attr("y", 18).attr("fill", C.ink).style("font-size", "17px").text("separation cost σ(v) of the marked node");
    svg.append("text").attr("x", ox + 56).attr("y", y2 - 10).attr("fill", C.ink).style("font-size", "17px").text("smallest contact weight in the graph");
    axisLabel(svg, "nodes added", { x: ox + cw / 2 + 25, y: y2 + chH + 40 });
    const p1 = svg.append("path").attr("fill", "none").attr("stroke", C.blue).attr("stroke-width", 2.6);
    const p2 = svg.append("path").attr("fill", "none").attr("stroke", real ? C.orange : C.blue).attr("stroke-width", 2.6);
    const status = svg.append("text").attr("x", ox + 56).attr("y", 28 + chH + 42).attr("fill", C.muted).style("font-size", "15px");
    let decreases = 0;

    const render = () => {
      const ids = nodes.map((n) => n.id);
      const { value, cut, side } = maxFlow(ids, edges, 0, "m");
      const prevSigma = hist.length ? hist[hist.length - 1].s : 0;
      if (value < prevSigma - 1e-9) decreases += 1;
      const minW = d3.min(edges, (e) => e.w);
      hist.push({ k: hist.length, s: value, w: minW });
      if (value > ys1.domain()[1]) { ys1.domain([0, value * 1.2]); ax1.call(d3.axisLeft(ys1).ticks(4)); styleAxis(ax1); }
      p1.attr("d", d3.line().curve(d3.curveStepAfter).x((d) => xs(d.k)).y((d) => ys1(d.s))(hist));
      p2.attr("d", d3.line().curve(d3.curveStepAfter).x((d) => xs(d.k)).y((d) => ys2(Math.max(d.w, ys2.domain()[0])))(hist));
      status.text(`σ(v) = ${fmt(value, real ? 2 : 0)} · decreases so far: ${decreases}`);
      gLinks.selectAll("line").data(edges).join("line")
        .attr("stroke", (e, k) => (cut.has(k) ? C.orange : C.line))
        .attr("stroke-opacity", (e) => (e.a === "m" || e.b === "m" ? 0.55 : 0.95))
        .attr("stroke-width", (e) => 1 + Math.min(e.w, 6) * 0.9);
      gNodes.selectAll("circle").data(nodes, (d) => d.id).join("circle")
        .attr("r", (d) => (d.id === "m" ? 26 : d.id === 0 ? 13 : 8))
        .attr("fill", (d) => (d.id === "m" ? "#3a3a36" : d.id === 0 ? C.blue : side.has(d.id) ? "#6aa3ea" : "#9d9c95"))
        .attr("stroke", C.surface).attr("stroke-width", 2);
      sim.nodes(nodes);
      sim.force("link").links(edges);
      sim.alpha(0.6).restart();
    };
    sim.on("tick", () => {
      gLinks.selectAll("line").attr("x1", (e) => e.source.x).attr("y1", (e) => e.source.y).attr("x2", (e) => e.target.x).attr("y2", (e) => e.target.y);
      gNodes.selectAll("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });
    svg.append("text").attr("x", GW / 2).attr("y", GH + 8).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "15px").text("medium");
    render();
    const it = d3.interval(() => {
      if (nodes.length > 62) { it.stop(); return; }
      const id = next++;
      const others = nodes.filter((n) => n.id !== "m").map((n) => n.id);
      nodes.push({ id, x: GW / 2 + (r() - 0.5) * 300, y: GH / 2 + (r() - 0.5) * 200 });
      addEdge(id, "m");
      for (let j = 0; j < 2; j++) addEdge(id, others[Math.floor(r() * others.length)]);
      if (r() < 0.22) addEdge(id, 0);
      render();
    }, 1000);
    return () => { it.stop(); sim.stop(); };
  }, [real]);
  return <svg ref={ref} width={W} height={H} />;
}

export function S05Individuation({ step }) {
  const f2 = DATA.foundation.individuation;
  return (
    <SlideFrame
      kicker="Derivation · individuation"
      title="An item is what it costs to cut it from its medium"
      lead="Separation cost σ(v): the weight of the cheapest cut between v and the medium — a maximum flow. Orange edges are the cut being computed live as the graph grows."
    >
      <GrowingGraph real={step >= 1} />
      <div className="absolute right-0 bottom-0 w-[600px]">
        {step === 0 && <Statement>Counted weights: σ(v) ≥ 1 (Menger), and σ(v) never decreases when contacts are added — 0 decreases in {(2 * 40 * 160).toLocaleString()} growth steps.</Statement>}
        {step === 1 && <Statement accent={C.orange}>Real weights: σ(v) still never decreases, but the floor drifts — the smallest weight fell to a median of {f2.real.global_min_edge_final_median.toExponential(1)}. No bound survives growth.</Statement>}
        {step >= 2 && <Statement accent={C.violet}>Identity is a region: with w(v,u)=10, w(v,m)=3, w(u,m)=1, cutting {"{v}"} costs 13 but {"{v,u}"} costs 4. Measuring adds contacts, so it cannot lower σ.</Statement>}
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 6 · ladders
function LadderChart({ step }) {
  const W = 900, H = 470, m = { l: 80, r: 24, t: 20, b: 56 };
  const ref = useD3((svg) => {
    const n = 70;
    const r = rng(11);
    const ladders = [
      { label: "constant p = 0.2", color: C.blue, p: () => 0.2 },
      { label: "random p ∈ [0.05, 0.5]", color: C.aqua, p: () => 0.05 + 0.45 * r() },
      ...(step >= 1 ? [
        { label: "harmonic pᵢ = 0.5/i (no floor)", color: C.orange, p: (i) => 0.5 / i },
        { label: "summable pᵢ = 0.5/i² (no floor)", color: C.yellow, p: (i) => 0.5 / (i * i) },
      ] : []),
    ];
    const x = d3.scaleLinear().domain([0, n]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([1e-14, 1.3]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)), { grid: true, height: H - m.t - m.b });
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, "~e")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "rungs climbed n", { x: (W + m.l) / 2, y: H - 12 });
    axisLabel(svg, "remaining gap gₙ / g₀", { x: 20, y: H / 2, rotate: true });
    const resid = Math.sin(Math.PI / Math.SQRT2) / (Math.PI / Math.SQRT2);
    if (step >= 1) {
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(resid)).attr("y2", y(resid)).attr("stroke", C.yellow).attr("stroke-dasharray", "5,5").attr("opacity", 0.7);
      svg.append("text").attr("x", W - m.r - 4).attr("y", y(resid) - 8).attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "15px")
        .text(`residual sin(π/√2)/(π/√2) = ${resid.toFixed(4)}`);
    }
    const data = ladders.map((L) => { let g = 1; const pts = [[0, 1]]; for (let i = 1; i <= n; i++) { g *= 1 - L.p(i); pts.push([i, g]); } return pts; });
    const line = d3.line().x((d) => x(d[0])).y((d) => y(Math.max(d[1], 1e-14)));
    ladders.forEach((L, i) => {
      const p = svg.append("path").attr("d", line(data[i])).attr("fill", "none").attr("stroke", L.color).attr("stroke-width", 2.6);
      const len = p.node().getTotalLength();
      p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().duration(2600).ease(d3.easeCubicOut).attr("stroke-dashoffset", 0);
    });
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

function Screen() {
  const W = 470, H = 170;
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([0, 2.3]).range([50, W - 10]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([H - 34, 12]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - 34})`).call(d3.axisBottom(x).ticks(3)));
    styleAxis(svg.append("g").attr("transform", "translate(50,0)").call(d3.axisLeft(y).ticks(3)));
    const st = d3.line().curve(d3.curveStepAfter).x((d) => x(d[0])).y((d) => y(d[1]));
    svg.append("path").attr("d", st([[0, 1], [1, 0.5], [2, 0.25], [2.3, 0.25]])).attr("stroke", C.blue).attr("stroke-width", 3).attr("fill", "none");
    svg.append("path").attr("d", st([[0, 1], [1, 0.25], [2.3, 0.25]])).attr("stroke", C.orange).attr("stroke-width", 3).attr("fill", "none").attr("stroke-dasharray", "7,5");
    svg.append("text").attr("x", x(0.1)).attr("y", y(0.6)).attr("fill", C.ink2).style("font-size", "15px").text("(0.5, 0.5) and (0.75): composite 0.75 both");
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S06Ladders({ step }) {
  const f4 = DATA.foundation.ladders;
  return (
    <SlideFrame
      kicker="Derivation · stepwise refinement"
      title="A ladder is a counted sum written as a product"
      lead="Each rung closes a fraction pᵢ of what remains: gₙ = g₀ ∏(1 − pᵢ). In logarithms, −ln(gₙ/g₀) = Σ −ln(1 − pᵢ) — a sum of positive terms, so the counting result applies."
    >
      <div className="absolute inset-0 flex gap-8">
        <div>
          <LadderChart step={step} />
          <div className="ml-[80px]">
            <Legend items={[["constant p = 0.2", C.blue], ["random p ∈ [0.05, 0.5]", C.aqua],
              ...(step >= 1 ? [["harmonic 0.5/i", C.orange], ["summable 0.5/i²", C.yellow]] : [])]} />
          </div>
        </div>
        <div className="flex-1 flex flex-col gap-4">
          <Statement>With a floor pᵢ ≥ p<sub>min</sub> &gt; 0 the gap goes to zero, but no finite ladder closes it: {f4.exact_rational_ladders_closed} of 200 exact rational ladders did.</Statement>
          <Reveal step={step} at={1}><Statement accent={C.yellow}>Without a floor a ladder can climb forever and stall: pᵢ = 0.5/i² leaves {f4.summable_ladder_residual.toFixed(3)} (closed form {f4.summable_closed_form_value.toFixed(4)}).</Statement></Reveal>
          <Reveal step={step} at={2}>
            <Statement accent={C.violet}>A composite does not identify its ladder. For P450 the composite is the coupling efficiency; the rungs are the intermediates.</Statement>
            <Screen />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 7 · atoms and iron
function IonizationChart({ step }) {
  const W = 860, H = 430, m = { l: 70, r: 16, t: 16, b: 52 };
  const a = DATA.atoms;
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([0, 110]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 26]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(11)), { grid: false });
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "atomic number Z", { x: (W + m.l) / 2, y: H - 10 });
    axisLabel(svg, "first ionization energy (eV)", { x: 18, y: H / 2, rotate: true });
    a.closures.forEach((c, i) => {
      svg.append("line").attr("x1", x(c + 0.5)).attr("x2", x(c + 0.5)).attr("y1", m.t).attr("y2", H - m.b)
        .attr("stroke", C.muted).attr("stroke-dasharray", "4,5").attr("opacity", 0)
        .transition().delay(900 + i * 220).duration(400).attr("opacity", 1);
      svg.append("text").attr("x", x(c) + 4).attr("y", m.t + 14).attr("fill", C.ink2).style("font-size", "14px").text(c)
        .attr("opacity", 0).transition().delay(900 + i * 220).attr("opacity", 1);
    });
    svg.append("line").attr("x1", x(26)).attr("x2", x(26)).attr("y1", m.t + 20).attr("y2", H - m.b).attr("stroke", C.orange).attr("stroke-width", 2);
    svg.append("text").attr("x", x(26) + 5).attr("y", y(3)).attr("fill", C.orange).style("font-size", "16px").text("Fe");
    const p = svg.append("path").attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(a.ie)).attr("fill", "none").attr("stroke", C.blue).attr("stroke-width", 2.2);
    const len = p.node().getTotalLength();
    p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().duration(1600).ease(d3.easeLinear).attr("stroke-dashoffset", 0);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Periods() {
  const W = 560, H = 190;
  const a = DATA.atoms;
  const ref = useD3((svg) => {
    const x = d3.scaleBand().domain(d3.range(1, 8)).range([40, W - 10]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 34]).range([H - 30, 10]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - 30})`).call(d3.axisBottom(x)));
    styleAxis(svg.append("g").attr("transform", "translate(40,0)").call(d3.axisLeft(y).ticks(4)));
    const bw = x.bandwidth() / 2 - 1;
    [["periodsCounted", C.blue, 0], ["periodsObserved", C.aqua, bw + 2]].forEach(([k, col, off]) => {
      svg.selectAll(null).data(a[k]).join("rect").attr("x", (_, i) => x(i + 1) + off).attr("width", bw)
        .attr("y", H - 30).attr("height", 0).attr("rx", 2).attr("fill", col)
        .transition().duration(700).delay((_, i) => i * 80).attr("y", (d) => y(d)).attr("height", (d) => H - 30 - y(d));
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function DShell() {
  const W = 560, H = 270;
  const ref = useD3((svg) => {
    const hsY = 30, lsHigh = 112, lsLow = 162, bw = 48, gap = 12;
    svg.append("text").attr("x", 0).attr("y", hsY - 10).attr("fill", C.ink2).style("font-size", "15px").text("high spin: spread, then pair");
    svg.append("text").attr("x", 0).attr("y", lsHigh - 10).attr("fill", C.ink2).style("font-size", "15px").text("low spin: lower three first");
    const hs = d3.range(5).map((i) => ({ x: i * (bw + gap), y: hsY }));
    const ls = [...d3.range(3).map((i) => ({ x: i * (bw + gap), y: lsLow })), ...d3.range(2).map((i) => ({ x: (i + 0.5) * (bw + gap), y: lsHigh }))];
    const box = (arr) => svg.append("g").selectAll("g").data(arr).join("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
    const H1 = box(hs), L1 = box(ls);
    [H1, L1].forEach((g) => g.append("rect").attr("width", bw).attr("height", 38).attr("rx", 4).attr("fill", C.panel).attr("stroke", C.line));
    const ktext = svg.append("text").attr("x", 330).attr("y", 62).attr("fill", C.ink).style("font-size", "30px").style("font-weight", 600);
    const ktext2 = svg.append("text").attr("x", 0).attr("y", 236).attr("fill", C.ink2).style("font-size", "16px");
    const ktext3 = svg.append("text").attr("x", 0).attr("y", 260).attr("fill", C.ink2).style("font-size", "16px");
    const fill = (k) => {
      const hsOcc = d3.range(5).map((i) => (k > i ? 1 : 0) + (k > 5 + i ? 1 : 0));
      const t = Math.min(k, 6), e = Math.max(k - 6, 0);
      const lsOcc = [...d3.range(3).map((i) => (t > i ? 1 : 0) + (t > 3 + i ? 1 : 0)), ...d3.range(2).map((i) => (e > i ? 1 : 0) + (e > 2 + i ? 1 : 0))];
      const arrows = (o) => (o === 2 ? "↑↓" : o === 1 ? "↑" : "");
      H1.selectAll("text").data((_, i) => [hsOcc[i]]).join("text").attr("x", bw / 2).attr("y", 27).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "20px").text(arrows);
      L1.selectAll("text").data((_, i) => [lsOcc[i]]).join("text").attr("x", bw / 2).attr("y", 27).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "20px").text(arrows);
      const un = (a) => d3.sum(a, (o) => (o === 1 ? 1 : 0));
      const fe = k === 5 ? "  Fe(III)" : k === 6 ? "  Fe(II)" : "";
      ktext.text(`d${k}${fe}`).attr("fill", k === 5 || k === 6 ? C.orange : C.ink);
      ktext2.text(`unpaired electrons: high spin ${un(hsOcc)}, low spin ${un(lsOcc)}`);
      ktext3.text(`difference ${un(hsOcc) - un(lsOcc)}${k === 5 || k === 6 ? " — the largest possible" : ""}`);
    };
    let k = 0;
    fill(0);
    const it = d3.interval(() => { k = (k + 1) % 11; fill(k); }, 1100);
    return () => it.stop();
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

// "3d6.4s2" -> "3d⁶ 4s²"
const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const config = (c) => c.split(".").map((t) => t.replace(/([spdf])(\d+)$/, (_, l, n) => l + [...n].map((d) => SUP[d]).join(""))).join(" ");

export function S07Atoms({ step }) {
  const a = DATA.atoms;
  return (
    <SlideFrame
      kicker="Derivation · atoms and the iron d shell"
      title="Counted cells give shells; iron sits where two spin ladders differ most"
      lead="Rotation labels give 2ℓ + 1 cells, spin doubles them, one occupant per cell (exclusion, supplied). With the empirical (n + ℓ, n) filling order (supplied), closures and periods follow."
    >
      <div className="absolute inset-0 flex gap-8">
        <div className="flex flex-col">
          <IonizationChart />
          <div className="text-[16px] ml-[70px]" style={{ color: C.ink2 }}>
            NIST ionization energies. The six largest drops fall after Z = {a.closures.join(", ")} — the counted closures.
          </div>
        </div>
        <div className="flex-1 flex flex-col gap-3">
          <Reveal step={step} at={1}>
            <div className="text-[17px] mb-1" style={{ color: C.ink }}>period lengths: counted vs observed</div>
            <Periods />
          </Reveal>
          <Reveal step={step} at={2}>
            <div className="text-[17px] mb-1" style={{ color: C.ink }}>Fe [Ar] {config(a.fe[0])} · Fe²⁺ {config(a.fe[1])} · Fe³⁺ {config(a.fe[2])}</div>
            <DShell />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}
