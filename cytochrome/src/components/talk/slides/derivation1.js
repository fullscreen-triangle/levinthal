import * as d3 from "d3";
import dynamic from "next/dynamic";
import { C, fmt } from "../theme";
import useD3, { styleAxis, axisLabel } from "../useD3";
import { SlideFrame, Reveal, Statement, Stat, Box, Legend, Badge } from "../ui";
import { rng } from "../util";
import DATA from "@/data/talk/derivation.json";

const AnimatedHeme = dynamic(() => import("../Models").then((m) => m.AnimatedHeme), { ssr: false });

// ---------------------------------------------------------------- 1 · title
const CHAIN = [
  "counted commitment", "direction · time · entropy", "individuation σ(v) ≥ 1", "finite parts, recurrence",
  "shells and closures", "iron d shell", "bonds, peptide plane", "order of the polymers",
  "membrane anchor", "triplet code", "compact fold", "heme–thiolate", "cycle as a ladder",
];

function ChainFigure() {
  const W = 1456, H = 290;
  const ref = useD3((svg) => {
    const cols = 5, bw = 248, bh = 56, gx = (W - cols * bw) / (cols - 1), gy = 40;
    const pos = CHAIN.map((_, i) => {
      const row = Math.floor(i / cols);
      let col = i % cols;
      if (row % 2 === 1) col = cols - 1 - col;
      return { x: col * (bw + gx), y: row * (bh + gy) + 8 };
    });
    const g = svg.append("g");
    for (let i = 0; i < CHAIN.length - 1; i++) {
      const a = pos[i], b = pos[i + 1];
      const sameRow = Math.abs(a.y - b.y) < 1;
      const p = d3.path();
      if (sameRow) {
        const dir = b.x > a.x ? 1 : -1;
        p.moveTo(a.x + (dir > 0 ? bw : 0), a.y + bh / 2);
        p.lineTo(b.x + (dir > 0 ? 0 : bw), b.y + bh / 2);
      } else {
        p.moveTo(a.x + bw / 2, a.y + bh);
        p.lineTo(b.x + bw / 2, b.y);
      }
      g.append("path").attr("class", `lk lk${i}`).attr("d", p.toString())
        .attr("stroke", C.line).attr("stroke-width", 2).attr("fill", "none");
    }
    const nodes = g.selectAll("g.n").data(CHAIN).join("g").attr("class", "n")
      .attr("transform", (_, i) => `translate(${pos[i].x},${pos[i].y})`);
    nodes.append("rect").attr("width", bw).attr("height", bh).attr("rx", 8)
      .attr("fill", C.panel).attr("stroke", C.line).attr("stroke-width", 1.5);
    nodes.append("text").attr("x", bw / 2).attr("y", bh / 2 + 6).attr("text-anchor", "middle")
      .attr("fill", C.ink2).style("font-size", "17px").text((d) => d);
    const counter = svg.append("text").attr("x", W).attr("y", H - 4).attr("text-anchor", "end")
      .attr("fill", C.muted).style("font-size", "16px");
    let k = 0, M = 0;
    const tick = () => {
      const i = k % (CHAIN.length + 3);
      nodes.select("rect").transition().duration(300)
        .attr("stroke", (_, j) => (j < i ? (j < 4 ? C.blue : j < 7 ? C.violet : C.orange) : C.line))
        .attr("fill", (_, j) => (j === i - 1 ? "#2a2a28" : C.panel));
      nodes.select("text").transition().duration(300).attr("fill", (_, j) => (j < i ? C.ink : C.ink2));
      g.selectAll("path.lk").transition().duration(300)
        .attr("stroke", (_, j) => (j < i - 1 ? C.ink2 : C.line));
      if (i > 0 && i <= CHAIN.length) M += 1;
      counter.text(`commitments so far: M = ${M}`);
      k += 1;
    };
    tick();
    const timer = d3.interval(tick, 650);
    return () => timer.stop();
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S01Title({ step }) {
  return (
    <div className="absolute inset-0 flex flex-col px-[72px] pt-[70px] pb-[88px]" style={{ color: C.ink }}>
      <div className="absolute right-[48px] top-[36px] w-[360px]">
        <AnimatedHeme height={360} />
        <div className="text-center text-[13px] -mt-2" style={{ color: C.muted }}>P450 with its heme highlighted · GLB, animated</div>
      </div>
      <div className="uppercase tracking-[0.2em] text-[16px]" style={{ color: C.muted }}>
        Biotechnology &amp; Enzyme Catalysis · Institute of Biochemistry · Universität Greifswald
      </div>
      <h1 className="mt-6 text-[68px] leading-[1.02] font-semibold tracking-tight max-w-[1080px]">
        Cytochrome P450 from counted commitment
      </h1>
      <p className="mt-4 text-[28px] max-w-[1060px]" style={{ color: C.ink2 }}>
        A derivation from individuation to the thiolate-ligated heme — and what it is used for
      </p>
      <div className="mt-6 text-[20px]" style={{ color: C.ink2 }}>
        Kundai Farai Sachikonye
        <span style={{ color: C.muted }}> · AIMe Registry for Artificial Intelligence · Experimental Bioinformatics, Freising</span>
      </div>
      <div className="mt-6 flex-1 relative">
        <ChainFigure />
      </div>
      <Reveal step={step} at={1} className="flex gap-4">
        <Stat value="2" label="manuscripts: foundation + P450 derivation" accent={C.blue} />
        <Stat value="10" label="validation experiments on public data" accent={C.aqua} />
        <Stat value="17" label="application papers, summarised in part II" accent={C.orange} />
        <Stat value="D · S · C" label="every link labelled derived, supplied or checked" accent={C.violet} />
      </Reveal>
    </div>
  );
}

// ------------------------------------------------------ 2 · counted vs measured
function PartialSums({ step }) {
  const W = 860, H = 520, m = { l: 80, r: 20, t: 20, b: 60 };
  const ref = useD3((svg) => {
    const N = 10000;
    const r = rng(20261011);
    const series = [
      { key: "int", color: C.blue, f: (() => { let s = 0; return () => (s += 1 + Math.floor(r() * 5)); })() },
      { key: "one", color: C.ink2, dash: "6,5", f: (() => { let s = 0; return () => (s += 1); })() },
      { key: "harm", color: C.orange, f: (() => { let s = 0, i = 0; return () => (s += 1 / ++i); })() },
      { key: "geom", color: C.aqua, f: (() => { let s = 0, i = 0; return () => (s += 0.5 ** ++i); })() },
    ].filter((s) => step >= 1 || s.key === "int" || s.key === "one");
    const x = d3.scaleLog().domain([1, N]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([0.4, 4e4]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5, "~s")), { grid: true, height: H - m.t - m.b });
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, "~s")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "number of commitments n", { x: (W + m.l) / 2, y: H - 14 });
    axisLabel(svg, "partial sum Sₙ", { x: 22, y: H / 2, rotate: true });
    const pts = series.map(() => []);
    const line = d3.line().x((d) => x(d[0])).y((d) => y(Math.max(d[1], 0.4)));
    const paths = series.map((s) => svg.append("path").attr("fill", "none").attr("stroke", s.color)
      .attr("stroke-width", 2.5).attr("stroke-dasharray", s.dash || null));
    let n = 0;
    const t = d3.timer((el) => {
      const target = Math.min(N, Math.floor(10 ** (el / 700)));
      while (n < target) {
        n += 1;
        series.forEach((s, i) => { const v = s.f(); if (n < 50 || n % Math.ceil(n / 60) === 0 || n === target) pts[i].push([n, v]); });
      }
      paths.forEach((p, i) => p.attr("d", line(pts[i])));
      if (n >= N) t.stop();
    });
    return () => t.stop();
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

export function S02Counted({ step }) {
  const f1 = DATA.foundation.partialSums;
  return (
    <SlideFrame
      kicker="Derivation · the one statement"
      title="A sum of infinitely many positive integers grows without bound — because of the word integer"
      lead="Positive terms alone can add up to a finite total. Integers carry a floor of one, and the floor is what makes every selection of terms diverge."
    >
      <div className="absolute inset-0 flex gap-10">
        <div>
          <PartialSums step={step} />
          <div className="ml-[80px] mt-1">
            <Legend items={[
              ["counted costs cᵢ ∈ {1…5}", C.blue], ["Sₙ = n", C.ink2, true],
              ...(step >= 1 ? [["real cᵢ = 1/i", C.orange], ["real cᵢ = 2⁻ⁱ (stalls at 1)", C.aqua]] : []),
            ]} />
          </div>
        </div>
        <div className="flex-1 flex flex-col gap-4 pt-2">
          <Reveal step={step} at={0}>
            <Statement>
              Counted: every cᵢ ≥ 1, so S<sub>n+1</sub> − S<sub>n</sub> ≥ 1 and S<sub>n</sub> ≥ n.
              <div className="mt-2 text-[16px]" style={{ color: C.muted }}>
                {f1.integer_partial_sum_violations_of_S_n_ge_n} violations in 10⁷ partial sums (10³ sequences × 10⁴ terms)
              </div>
            </Statement>
          </Reveal>
          <Reveal step={step} at={1}>
            <Statement accent={C.orange}>
              Measured: cᵢ &gt; 0 only. ½ + ¼ + ⅛ + … = 1, and the running minimum of cᵢ = Uᵢ/i falls to{" "}
              {f1.real_running_min_final_median.toExponential(1)} after 10⁴ terms.
            </Statement>
          </Reveal>
          <Reveal step={step} at={2}>
            <Statement accent={C.violet}>
              <b>Theorem.</b> For positive terms, inf cᵢ &gt; 0 <b>if and only if</b> every infinite selection of the terms
              has a divergent sum.
              <div className="mt-2 text-[16px]" style={{ color: C.muted }}>
                Proof: if inf cᵢ = 0, pick terms below 2⁻ᵏ; they sum to less than 1.
              </div>
            </Statement>
          </Reveal>
          <Reveal step={step} at={2}>
            <div className="text-[18px] leading-relaxed" style={{ color: C.ink2 }}>
              One assumption carries the physics: <b style={{ color: C.ink }}>a record costs at least θ &gt; 0</b>.
              Its physical counterpart is the Landauer bound k<sub>B</sub>T ln 2 per erased bit.
            </div>
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 3 · Zeno, direction
function Dichotomy({ step }) {
  const W = 1456, H = 300;
  const ref = useD3((svg) => {
    const x0 = 40, x1 = W - 60, y = 92;
    const x = d3.scaleLinear().domain([0, 1]).range([x0, x1]);
    svg.append("line").attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y).attr("stroke", C.line).attr("stroke-width", 6).attr("stroke-linecap", "round");
    svg.append("text").attr("x", x0).attr("y", y + 40).attr("fill", C.muted).style("font-size", "16px").text("start");
    svg.append("text").attr("x", x1).attr("y", y + 40).attr("text-anchor", "end").attr("fill", C.muted).style("font-size", "16px").text("goal D");
    const runner = svg.append("circle").attr("cx", x0).attr("cy", y).attr("r", 13).attr("fill", C.aqua);
    const occ = svg.append("text").attr("x", x0).attr("y", y - 26).attr("fill", C.aqua).style("font-size", "16px").text("occupation");
    const ticks = svg.append("g");
    const info = svg.append("g").attr("transform", `translate(${x0},${y + 92})`);
    const t1 = info.append("text").attr("fill", C.ink).style("font-size", "22px");
    const t2 = info.append("text").attr("y", 34).attr("fill", C.ink2).style("font-size", "18px");
    const timers = [];
    // occupation: the runner simply moves and arrives, again and again
    const run = () => {
      runner.attr("cx", x0).transition().duration(2600).ease(d3.easeLinear).attr("cx", x1)
        .on("end", () => timers.push(d3.timeout(run, 600)));
      occ.attr("x", x0).transition().duration(2600).ease(d3.easeLinear).attr("x", x1 - 80);
    };
    run();
    if (step >= 1) {
      let k = 0;
      const commit = () => {
        k += 1;
        const p = 1 - 0.5 ** k;
        ticks.append("line").attr("x1", x(p)).attr("x2", x(p)).attr("y1", y - 26).attr("y2", y - 26)
          .attr("stroke", C.orange).attr("stroke-width", 2.5)
          .transition().duration(300).attr("y2", y + 26);
        t1.text(`committed midpoints k = ${k}   ·   cost ≥ ${k} θ`);
        t2.text(`distance still unresolved D·2⁻ᵏ = ${(0.5 ** k).toExponential(2)} D   ·   distance covered → D, commitments → ∞`);
        if (k < 40) timers.push(d3.timeout(commit, Math.max(160, 700 - k * 40)));
      };
      timers.push(d3.timeout(commit, 400));
    } else {
      t1.text("A runner occupies every point of the course and arrives.");
      t2.text("Occupation is not a record: it costs nothing in this accounting.");
    }
    return () => timers.forEach((t) => t.stop());
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

function Register() {
  const W = 1456, H = 90;
  const ref = useD3((svg) => {
    const cell = 26;
    const g = svg.append("g").attr("transform", "translate(40,20)");
    const label = svg.append("text").attr("x", W - 60).attr("y", 46).attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "22px");
    let M = 0;
    const r = rng(7);
    const add = () => {
      if (M * (cell + 4) > W - 360) { g.selectAll("rect").remove(); M = 0; }
      g.append("rect").attr("x", M * (cell + 4)).attr("y", 0).attr("width", cell).attr("height", cell + 14).attr("rx", 4)
        .attr("fill", r() < 0.5 ? C.blue : C.violet).attr("opacity", 0).transition().duration(250).attr("opacity", 0.9);
      M += 1;
      label.selectAll("tspan").remove();
      label.append("tspan").text(`register: M = ${M}   ·   t = M/f   ·   S = k`);
      label.append("tspan").attr("baseline-shift", "sub").style("font-size", "15px").text("B");
      label.append("tspan").text(" M ln b");
    };
    const it = d3.interval(add, 420);
    return () => it.stop();
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S03Zeno({ step }) {
  return (
    <SlideFrame
      kicker="Derivation · direction"
      title="Occupation is free; a record is not"
      lead="Zeno's dichotomy uses two sums at once: the distances between midpoints converge, while the commitments needed to distinguish the midpoints do not."
    >
      <Dichotomy step={step} />
      <Reveal step={step} at={2}>
        <Register />
        <div className="grid grid-cols-3 gap-4 mt-2">
          <Statement>Commitments are strictly ordered by their count M. No commitment lowers M — that order is the <b>direction</b>.</Statement>
          <Statement accent={C.aqua}>Undoing a record is itself a record: an erasure raises M.</Statement>
          <Statement accent={C.violet}>A finite budget B makes at most ⌊B/θ⌋ records. To resolve arrival to ε takes ⌈log₂ D/ε⌉ of them: 50 for ε = 10⁻¹⁵ D.</Statement>
        </div>
      </Reveal>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 4 · Kac ring
function KacRing({ step }) {
  const W = 1456, H = 420;
  const ref = useD3((svg) => {
    const N = 120, R = 165, cx = 220, cy = 205;
    const r = rng(3);
    const marked = d3.range(N).map(() => false);
    let placed = 0;
    while (placed < 13) { const i = Math.floor(r() * N); if (!marked[i]) { marked[i] = true; placed += 1; } }
    let c = d3.range(N).map(() => 0);
    const c0 = c.slice();
    const ang = (i) => (i / N) * 2 * Math.PI - Math.PI / 2;
    const g = svg.append("g");
    g.selectAll("line.mk").data(d3.range(N).filter((i) => marked[i])).join("line")
      .attr("x1", (i) => cx + (R - 26) * Math.cos(ang(i + 0.5))).attr("y1", (i) => cy + (R - 26) * Math.sin(ang(i + 0.5)))
      .attr("x2", (i) => cx + (R + 26) * Math.cos(ang(i + 0.5))).attr("y2", (i) => cy + (R + 26) * Math.sin(ang(i + 0.5)))
      .attr("stroke", C.orange).attr("stroke-width", 2.5);
    const balls = g.selectAll("circle").data(d3.range(N)).join("circle")
      .attr("cx", (i) => cx + R * Math.cos(ang(i))).attr("cy", (i) => cy + R * Math.sin(ang(i))).attr("r", 6.5);
    const center = g.append("text").attr("x", cx).attr("y", cy - 6).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "26px");
    const center2 = g.append("text").attr("x", cx).attr("y", cy + 24).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "16px");
    const paint = () => balls.attr("fill", (i) => (c[i] ? C.blue : "#e9e7df"));
    paint();

    // chart of coarse entropy, Hamming distance and register
    const cx0 = 500, cw = W - cx0 - 20, ch = 340, top = 20;
    const T = step >= 1 ? 120 : 2 * N;
    const xs = d3.scaleLinear().domain([0, T]).range([cx0 + 50, cx0 + cw]);
    const ys = d3.scaleLinear().domain([0, 1.25]).range([top + ch, top]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${top + ch})`).call(d3.axisBottom(xs).ticks(6)), { grid: true, height: ch });
    styleAxis(svg.append("g").attr("transform", `translate(${cx0 + 50},0)`).call(d3.axisLeft(ys).tickValues([0, 0.25, 0.5, 0.75, 1])), { grid: true, width: cw - 50, orient: "y" });
    axisLabel(svg, "step", { x: cx0 + cw / 2 + 25, y: top + ch + 44 });
    const series = [
      { color: C.blue, pts: [], label: "coarse entropy / ln 2" },
      { color: C.orange, pts: [], label: "Hamming distance to start / N" },
      { color: C.aqua, pts: [], label: step >= 1 ? "register M / 120" : "register M / 240" },
    ];
    const paths = series.map((s) => svg.append("path").attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2.4));
    const line = d3.line().x((d) => xs(d[0])).y((d) => ys(d[1]));
    if (step >= 1) {
      svg.append("line").attr("x1", xs(60)).attr("x2", xs(60)).attr("y1", top).attr("y2", top + ch)
        .attr("stroke", C.muted).attr("stroke-dasharray", "4,4");
      svg.append("text").attr("x", xs(60) + 8).attr("y", ys(0.06)).attr("fill", C.muted).style("font-size", "15px").text("reverse the dynamics");
    }
    const ent = () => {
      const p = d3.mean(c);
      return p === 0 || p === 1 ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p)) / Math.log(2);
    };
    const fwd = () => { const f = c.map((v, i) => v ^ (marked[i] ? 1 : 0)); c = f.map((_, i) => f[(i - 1 + N) % N]); };
    const bwd = () => { const s = c.map((_, i) => c[(i + 1) % N]); c = s.map((v, i) => v ^ (marked[i] ? 1 : 0)); };
    let t = 0;
    const push = () => {
      const h = d3.sum(c, (v, i) => (v !== c0[i] ? 1 : 0)) / N;
      series[0].pts.push([t, ent()]); series[1].pts.push([t, h]); series[2].pts.push([t, t / T]);
      paths.forEach((p, i) => p.attr("d", line(series[i].pts)));
      center.text(`M = ${t}`);
      center2.text(t > 0 && c.every((v, i) => v === c0[i]) ? "configuration returned" : "steps committed");
    };
    push();
    const it = d3.interval(() => {
      if (t >= T) { it.stop(); return; }
      if (step >= 1 && t >= 60) bwd(); else fwd();
      t += 1; paint(); push();
    }, step >= 1 ? 70 : 45);
    // legend in the headroom above 1, one row
    const lg = svg.append("g").attr("transform", `translate(${cx0 + 70},${ys(1.15)})`);
    series.forEach((s, i) => {
      lg.append("line").attr("x1", i * 300).attr("x2", i * 300 + 22).attr("y1", 0).attr("y2", 0).attr("stroke", s.color).attr("stroke-width", 3);
      lg.append("text").attr("x", i * 300 + 30).attr("y", 5).attr("fill", C.ink2).style("font-size", "15px").text(s.label);
    });
    return () => it.stop();
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

export function S04Kac({ step }) {
  const rec = DATA.foundation.recurrence;
  return (
    <SlideFrame
      kicker="Derivation · time, entropy, Loschmidt"
      title="Configurations return; records do not"
      lead="Kac ring: balls step round a circle and change colour at the marked edges. The dynamics is reversible and periodic. Each step is one commitment."
    >
      <KacRing step={step} />
      <div className="grid grid-cols-3 gap-4 -mt-1">
        <Reveal step={step} at={0}><Statement>Forward: the coarse-grained entropy rises, then the configuration recurs exactly — here at step {2 * 120}, and at step {rec.kac_ring.recurrence_step} for N = {rec.kac_ring.N}.</Statement></Reveal>
        <Reveal step={step} at={1}><Statement accent={C.orange}>Reversed: the configuration returns to the start and its entropy falls back to zero. The register still grows: {rec.reversal.register_length_at_return} commitments for T = {rec.reversal.T}.</Statement></Reveal>
        <Reveal step={step} at={2}><Statement accent={C.violet}>Count, time and register entropy are one number: <b>M = f t = S/(k<sub>B</sub> ln b)</b>. Loschmidt's reversal acts on configurations, never on M.</Statement></Reveal>
      </div>
    </SlideFrame>
  );
}
