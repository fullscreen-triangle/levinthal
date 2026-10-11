import * as d3 from "d3";
import { C, fmt, pct } from "../theme";
import useD3, { styleAxis, axisLabel } from "../useD3";
import { SlideFrame, Reveal, Statement, Legend, Badge } from "../ui";
import { rng } from "../util";
import DATA from "@/data/talk/derivation.json";

// ------------------------------------------------------ 8 · bonds, peptide
function BondDemo() {
  const W = 620, H = 360;
  const ref = useD3((svg) => {
    const am = 6, bm = 8;
    const A = { x: 150, y: 90 }, B = { x: 330, y: 90 }, M = { x: 240, y: 270 };
    const e = (p, q, col, w) => svg.append("line").attr("x1", p.x).attr("y1", p.y).attr("x2", q.x).attr("y2", q.y).attr("stroke", col).attr("stroke-width", w);
    e(A, M, C.line, 4); e(B, M, C.line, 5);
    const ab = e(A, B, C.blue, 2);
    [[A, "a"], [B, "b"], [M, "medium"]].forEach(([p, t]) => {
      svg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", t === "medium" ? 26 : 20).attr("fill", t === "medium" ? "#3a3a36" : C.panel).attr("stroke", C.ink2);
      svg.append("text").attr("x", p.x).attr("y", p.y + 6).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "16px").text(t === "medium" ? "m" : t);
    });
    svg.append("text").attr("x", (A.x + M.x) / 2 - 40).attr("y", 190).attr("fill", C.muted).style("font-size", "15px").text(`w = ${am}`);
    svg.append("text").attr("x", (B.x + M.x) / 2 + 12).attr("y", 190).attr("fill", C.muted).style("font-size", "15px").text(`w = ${bm}`);
    const wt = svg.append("text").attr("x", 240).attr("y", 72).attr("text-anchor", "middle").attr("fill", C.blue).style("font-size", "16px");
    const bars = svg.append("g").attr("transform", "translate(420,40)");
    const x = d3.scaleLinear().domain([0, 30]).range([0, 150]);
    const rows = ["cut {a}", "cut {b}", "cut {a,b}"];
    const rects = bars.selectAll("rect").data(rows).join("rect").attr("y", (_, i) => i * 46).attr("height", 26).attr("rx", 3);
    bars.selectAll("text.l").data(rows).join("text").attr("class", "l").attr("y", (_, i) => i * 46 - 4).attr("fill", C.ink2).style("font-size", "14px").text((d) => d);
    const vals = bars.selectAll("text.v").data(rows).join("text").attr("class", "v").attr("y", (_, i) => i * 46 + 19).attr("fill", C.ink).style("font-size", "14px");
    const verdict = svg.append("text").attr("x", 20).attr("y", 326).attr("fill", C.ink).style("font-size", "17px");
    const verdict2 = svg.append("text").attr("x", 20).attr("y", 350).attr("fill", C.ink2).style("font-size", "15px");
    let w = 0;
    const draw = () => {
      const v = [am + w, bm + w, am + bm];
      rects.attr("width", (_, i) => x(v[i])).attr("fill", (_, i) => (i === 2 ? (v[2] < Math.min(v[0], v[1]) ? C.aqua : C.muted) : C.line));
      vals.attr("x", (_, i) => x(v[i]) + 6).text((_, i) => v[i].toFixed(0));
      ab.attr("stroke-width", 1 + w * 0.8);
      wt.text(`w(a,b) = ${w}`);
      const bond = w > Math.max(am + w, bm + w) / 2;
      verdict.text(bond ? "bond: the pair is cut as one item" : "no bond yet").attr("fill", bond ? C.aqua : C.ink);
      verdict2.text(`condition w(a,b) > ½ max(cut{a}, cut{b}) = ${(Math.max(am + w, bm + w) / 2).toFixed(1)}`);
    };
    draw();
    const it = d3.interval(() => { w = (w + 1) % 15; draw(); }, 650);
    return () => it.stop();
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function OmegaChart() {
  const W = 760, H = 300, m = { l: 66, r: 14, t: 14, b: 48 };
  const b = DATA.backbone;
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([0, 360]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([0.8, 3000]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([0, 90, 180, 270, 360])));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4, "~s")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "peptide dihedral ω (°)", { x: (W + m.l) / 2, y: H - 8 });
    axisLabel(svg, "bonds", { x: 16, y: H / 2, rotate: true });
    const bw = x(4) - x(0);
    svg.selectAll("rect").data(b.omega_counts.map((c, i) => [b.omega_bins_deg[i], c]).filter((d) => d[1] > 0)).join("rect")
      .attr("x", (d) => x(d[0]) + 0.5).attr("width", Math.max(bw - 1, 1)).attr("y", H - m.b).attr("height", 0).attr("fill", C.blue)
      .transition().duration(900).attr("y", (d) => y(Math.max(d[1], 0.8))).attr("height", (d) => H - m.b - y(Math.max(d[1], 0.8)));
    svg.append("text").attr("x", m.l + 12).attr("y", m.t + 18).attr("fill", C.ink2).style("font-size", "15px")
      .text(`${pct(b.transFraction)} within 20° of trans · ${b.cis} cis · ${b.intermediate} between · n = ${b.nBonds.toLocaleString()}`);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Rama() {
  const W = 330, H = 300;
  const b = DATA.backbone;
  const ref = useD3((svg) => {
    const s = 7.2, ox = 50, oy = 10;
    const max = d3.max(b.rama_counts_phi_by_psi.flat());
    const col = d3.scaleSequentialLog(d3.interpolateRgb("#1f3350", "#a9cdfb")).domain([1, max]);
    b.rama_counts_phi_by_psi.forEach((row, i) => row.forEach((v, j) => {
      if (v > 0) svg.append("rect").attr("x", ox + i * s).attr("y", oy + (35 - j) * s).attr("width", s).attr("height", s).attr("fill", col(v)).attr("opacity", 0).transition().delay((i + j) * 8).attr("opacity", 1);
    }));
    const x = d3.scaleLinear().domain([-180, 180]).range([ox, ox + 36 * s]);
    const y = d3.scaleLinear().domain([-180, 180]).range([oy + 36 * s, oy]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${oy + 36 * s})`).call(d3.axisBottom(x).tickValues([-180, 0, 180])));
    styleAxis(svg.append("g").attr("transform", `translate(${ox},0)`).call(d3.axisLeft(y).tickValues([-180, 0, 180])));
    axisLabel(svg, "φ", { x: ox + 18 * s, y: H - 2 });
    axisLabel(svg, "ψ", { x: 14, y: oy + 18 * s, rotate: true });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S08Bonds({ step }) {
  const b = DATA.backbone;
  return (
    <SlideFrame
      kicker="Derivation · bonds and the peptide carrier"
      title="A bond is a contact that makes a pair cheaper to cut as one"
      lead={`cut{a,b} = cut{a} + cut{b} − 2w(a,b). Geometry by maximal pair separation gives the tetrahedral 109.47°; in twelve P450 structures N–Cα–Cβ is ${fmt(b.NCACB[0], 1)} ± ${fmt(b.NCACB[1], 1)}° (refinement-restrained, shown for geometry only).`}
    >
      <div className="absolute inset-0 grid grid-cols-[620px_1fr] gap-8">
        <div className="flex flex-col gap-3">
          <BondDemo />
          <Reveal step={step} at={1}>
            <Statement accent={C.aqua}>
              Shared electrons over three atoms shorten the peptide C–N to <b>{fmt(b.CN[0], 3)} Å</b> against {fmt(b.NCA[0], 3)} Å for N–Cα, and lock the group in a plane.
            </Statement>
          </Reveal>
        </div>
        <div className="flex flex-col gap-2">
          <Reveal step={step} at={1}><OmegaChart /></Reveal>
          <Reveal step={step} at={2}>
            <div className="flex gap-6 items-start">
              <Rama />
              <div className="text-[18px] leading-relaxed pt-4" style={{ color: C.ink2 }}>
                Two free rotations per residue remain, φ and ψ — unrestrained in refinement.
                <div className="mt-3"><b style={{ color: C.ink }}>{pct(b.helical, 0)}</b> helical, <b style={{ color: C.ink }}>{pct(b.extended, 0)}</b> extended.</div>
                <div className="mt-3 text-[15px]" style={{ color: C.muted }}>5,406 peptide bonds from 2CPP, 1PHC, 1TQN, 1OG2, 2F9Q, 2HI4, 3E6I, 1BU7, 3LD6, 3EQM, 3RUK, 3N9Y.</div>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 9 · boundary → record → catalyst
function Dependency() {
  const W = 1456, H = 170;
  const ref = useD3((svg) => {
    const items = [
      { t: "boundary", s: "one amphiphile species · self-assembles · no template, no catalyst", c: C.aqua },
      { t: "record", s: "two complementary classes · copied by pairing · needs a boundary to persist", c: C.violet },
      { t: "catalyst", s: "20 units in a specified order · cannot be copied by pairing · needs a record", c: C.orange },
    ];
    const bw = 420, gap = (W - 3 * bw) / 2;
    const g = svg.selectAll("g").data(items).join("g").attr("transform", (_, i) => `translate(${i * (bw + gap)},20)`).attr("opacity", 0);
    g.append("rect").attr("width", bw).attr("height", 110).attr("rx", 10).attr("fill", C.panel).attr("stroke", (d) => d.c).attr("stroke-width", 2);
    g.append("text").attr("x", 20).attr("y", 40).attr("fill", C.ink).style("font-size", "26px").style("font-weight", 600).text((d) => d.t);
    g.append("foreignObject").attr("x", 20).attr("y", 52).attr("width", bw - 40).attr("height", 56)
      .append("xhtml:div").style("color", C.ink2).style("font-size", "15px").style("line-height", "1.35").text((d) => d.s);
    g.transition().delay((_, i) => i * 450).duration(500).attr("opacity", 1);
    [0, 1].forEach((i) => {
      const x1 = (i + 1) * bw + i * gap + 8, x2 = (i + 1) * (bw + gap) - 10;
      svg.append("line").attr("x1", x1).attr("x2", x1).attr("y1", 75).attr("y2", 75).attr("stroke", C.ink2).attr("stroke-width", 3)
        .transition().delay(300 + i * 450).duration(400).attr("x2", x2);
      svg.append("path").attr("d", `M${x2},75 l-12,-8 l0,16 z`).attr("fill", C.ink2).attr("opacity", 0).transition().delay(700 + i * 450).attr("opacity", 1);
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Hydropathy() {
  const W = 700, H = 300, m = { l: 60, r: 10, t: 14, b: 40 };
  const B = DATA.boundary;
  const ref = useD3((svg) => {
    const groups = [["ER membrane", C.blue], ["mitochondrial", C.yellow], ["bacterial", C.orange]];
    const x = d3.scaleBand().domain(groups.map((g) => g[0])).range([m.l, W - m.r]).padding(0.25);
    const y = d3.scaleLinear().domain([-1.2, 3.8]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "N-terminal hydropathy", { x: 16, y: H / 2 - 10, rotate: true });
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(B.threshold)).attr("y2", y(B.threshold)).attr("stroke", C.ink2).attr("stroke-dasharray", "6,5");
    svg.append("text").attr("x", W - m.r).attr("y", y(B.threshold) - 6).attr("text-anchor", "end").attr("fill", C.ink2).style("font-size", "14px").text("1.6: spans a bilayer");
    const r = rng(5);
    groups.forEach(([g, col]) => {
      const vals = B.values[g];
      svg.append("g").selectAll("circle").data(vals).join("circle")
        .attr("cx", () => x(g) + x.bandwidth() * (0.15 + 0.7 * r())).attr("cy", H - m.b).attr("r", 2.6)
        .attr("fill", col).attr("opacity", 0.75)
        .transition().duration(900).delay((_, i) => i * 1.5).attr("cy", (d) => y(d));
      const grp = B.groups[g];
      svg.append("text").attr("x", x(g) + x.bandwidth() / 2).attr("y", m.t + 12).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "16px")
        .text(`${pct(grp.fraction_above_1_6 ?? grp["fraction_above_1.6"], 1)} of ${grp.n}`);
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function CodeChart() {
  const W = 700, H = 300, m = { l: 60, r: 14, t: 14, b: 50 };
  const R = DATA.record;
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([R.edges[0] - 0.5, R.edges[R.edges.length - 1]]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(R.counts)]).range([H - m.b, m.t + 30]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4, "~s")));
    axisLabel(svg, "mean squared hydropathy change per point substitution", { x: (W + m.l) / 2, y: H - 10 });
    svg.selectAll("rect").data(R.counts).join("rect")
      .attr("x", (_, i) => x(R.edges[i])).attr("width", (_, i) => Math.max(x(R.edges[i + 1]) - x(R.edges[i]) - 1, 1))
      .attr("y", H - m.b).attr("height", 0).attr("fill", C.muted)
      .transition().duration(800).delay((_, i) => i * 6).attr("y", (d) => y(d)).attr("height", (d) => H - m.b - y(d));
    svg.append("line").attr("x1", x(R.standardCost)).attr("x2", x(R.standardCost)).attr("y1", m.t + 20).attr("y2", H - m.b).attr("stroke", C.orange).attr("stroke-width", 3);
    svg.append("text").attr("x", x(R.standardCost) + 8).attr("y", m.t + 34).attr("fill", C.ink).style("font-size", "16px")
      .text(`standard code: ${pct(R.atOrBelow / R.permuted, 2)} of permuted codes as good`);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S09Order({ step }) {
  return (
    <SlideFrame
      kicker="Derivation · the order of the polymers"
      title="Boundary, then record, then catalyst — a dependency order, tested once for each"
      lead="What each structure presupposes, not when it arose. P450 needs all three: an ER anchor, a gene, and a fold that holds an iron porphyrin."
    >
      <Dependency />
      <div className="grid grid-cols-2 gap-8 mt-2">
        <Reveal step={step} at={1}>
          <div className="text-[18px] mb-1" style={{ color: C.ink }}>Boundary: an α-helix needs ~20 hydrophobic residues to cross 3 nm of bilayer. 1,919 reviewed P450s:</div>
          <Hydropathy />
        </Reveal>
        <Reveal step={step} at={2}>
          <div className="text-[18px] mb-1" style={{ color: C.ink }}>Record: 4² = 16 &lt; 21 ≤ 4³, so codons are three letters. The code against 200,000 permuted codes:</div>
          <CodeChart />
        </Reveal>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 10 · fold and heme–thiolate
function RgChart() {
  const W = 470, H = 260, m = { l: 56, r: 10, t: 10, b: 44 };
  const F = DATA.fold.entries;
  const ref = useD3((svg) => {
    const x = d3.scaleLog().domain([90, 650]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([10, 100]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([100, 200, 300, 500]).tickFormat(d3.format("d"))));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([10, 20, 40, 80]).tickFormat(d3.format("d"))));
    axisLabel(svg, "residues N", { x: (W + m.l) / 2, y: H - 6 });
    axisLabel(svg, "Rg (Å)", { x: 14, y: H / 2, rotate: true });
    const nn = d3.range(90, 651, 10);
    svg.append("path").attr("d", d3.line().x((n) => x(n)).y((n) => y(2.2 * n ** 0.38))(nn)).attr("stroke", C.ink2).attr("stroke-dasharray", "6,4").attr("fill", "none");
    svg.append("path").attr("d", d3.line().x((n) => x(n)).y((n) => y(1.927 * n ** 0.598))(nn)).attr("stroke", C.muted).attr("stroke-dasharray", "2,4").attr("fill", "none");
    svg.append("text").attr("x", x(120)).attr("y", y(1.927 * 120 ** 0.598) - 8).attr("fill", C.muted).style("font-size", "13px").text("denatured");
    svg.append("text").attr("x", x(400)).attr("y", y(2.2 * 400 ** 0.38) + 22).attr("fill", C.ink2).style("font-size", "13px").text("folded");
    svg.selectAll("circle").data(F).join("circle").attr("cx", (d) => x(d.n)).attr("cy", (d) => y(d.rg)).attr("r", 0)
      .attr("fill", (d) => (d.p450 ? C.blue : C.orange)).attr("stroke", C.surface)
      .transition().delay((_, i) => i * 60).attr("r", 6);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function SigmaChart() {
  const W = 470, H = 260, m = { l: 50, r: 14, t: 10, b: 44 };
  const S = DATA.fold.sigma["2CPP"];
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([0, 330]).range([m.l, W - m.r]);
    const bins = d3.bin().domain(x.domain()).thresholds(40)(S.values);
    const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length)]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4)));
    axisLabel(svg, "separation cost σ from solvent (contacts)", { x: (W + m.l) / 2, y: H - 6 });
    svg.selectAll("rect").data(bins).join("rect").attr("x", (b) => x(b.x0) + 0.5).attr("width", (b) => Math.max(x(b.x1) - x(b.x0) - 1, 1))
      .attr("y", (b) => y(b.length)).attr("height", (b) => H - m.b - y(b.length)).attr("fill", C.muted);
    svg.append("line").attr("x1", x(S.heme)).attr("x2", x(S.heme)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", C.orange).attr("stroke-width", 3);
    svg.append("text").attr("x", x(S.heme) - 6).attr("y", m.t + 18).attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "15px").text(`heme ${S.heme}`);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function LigandChart() {
  const W = 470, H = 260, m = { l: 56, r: 10, t: 10, b: 44 };
  const sites = DATA.ligation.sites;
  const ref = useD3((svg) => {
    const groups = ["CYS SG", "HIS NE2", "TYR OH"];
    const x = d3.scaleBand().domain(groups).range([m.l, W - m.r]).padding(0.3);
    const y = d3.scaleLinear().domain([1.75, 2.6]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "Fe to nearest protein atom (Å)", { x: 14, y: H / 2, rotate: true });
    const r = rng(2);
    svg.selectAll("circle").data(sites).join("circle")
      .attr("cx", (d) => x(`${d.ligand} ${d.atom}`) + x.bandwidth() * (0.2 + 0.6 * r())).attr("cy", (d) => y(d.d)).attr("r", 0)
      .attr("fill", (d) => (d.p450 ? C.blue : C.orange)).attr("stroke", C.surface)
      .transition().delay((_, i) => i * 40).attr("r", 6);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function SpinSwitch() {
  const W = 470, H = 230;
  const dist = DATA.ligation.distal;
  const ref = useD3((svg) => {
    const panel = (ox, title, oop, water) => {
      const g = svg.append("g").attr("transform", `translate(${ox},0)`);
      g.append("text").attr("x", 105).attr("y", 18).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "16px").text(title);
      if (water) {
        g.append("circle").attr("cx", 105).attr("cy", 58).attr("r", 11).attr("fill", C.blue);
        g.append("text").attr("x", 122).attr("y", 62).attr("fill", C.ink2).style("font-size", "13px").text(`H₂O ${water.toFixed(2)} Å`);
      } else {
        g.append("text").attr("x", 105).attr("y", 62).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "13px").text("camphor · no water");
      }
      g.append("line").attr("x1", 20).attr("x2", 190).attr("y1", 110).attr("y2", 110).attr("stroke", C.ink2).attr("stroke-width", 4);
      g.append("text").attr("x", 194).attr("y", 114).attr("fill", C.muted).style("font-size", "12px").text("N₄");
      const fe = g.append("g").attr("transform", "translate(105,110)");
      fe.append("circle").attr("r", 13).attr("fill", C.orange);
      fe.append("text").attr("y", 4).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "11px").style("font-weight", 600).text("Fe");
      fe.transition().duration(1200).attr("transform", `translate(105,${110 + oop * 60})`);
      g.append("line").attr("x1", 105).attr("x2", 105).attr("y1", 140 + oop * 60).attr("y2", 185).attr("stroke", C.yellow).attr("stroke-width", 4);
      g.append("text").attr("x", 112).attr("y", 186).attr("fill", C.yellow).style("font-size", "13px").text("Cys S");
      g.append("text").attr("x", 105).attr("y", 218).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "14px").text(`Fe out of plane ${oop.toFixed(2)} Å`);
    };
    panel(0, "1PHC substrate-free", dist["1PHC"].fe_out_of_plane, dist["1PHC"].water_distance);
    panel(240, "2CPP camphor-bound", dist["2CPP"].fe_out_of_plane, null);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

const r1 = (v) => (Math.round(v * 10) / 10).toFixed(1);

export function S10Heme({ step }) {
  const L = DATA.ligation;
  const sig = L.signature;
  const p450 = DATA.fold.entries.filter((e) => e.p450);
  return (
    <SlideFrame
      kicker="Derivation · the catalyst and its heme"
      title="A compact fold buries the heme, and a cysteine binds its iron"
      lead="Contacts can only raise separation cost, so a fold individuates what it encloses. The proximal ligand then sorts heme proteins by chemistry."
    >
      <div className="absolute inset-0 grid grid-cols-3 gap-x-6 gap-y-2">
        <div>
          <RgChart />
          <div className="text-[15px] ml-[56px]" style={{ color: C.ink2 }}>
            P450 Rg {r1(d3.min(p450, (e) => e.rg))}–{r1(d3.max(p450, (e) => e.rg))} Å; heme {r1(d3.min(p450, (e) => e.heme))}–{r1(d3.max(p450, (e) => e.heme))}% exposed (globins 18–20%).
          </div>
        </div>
        <Reveal step={step} at={1}>
          <SigmaChart />
          <div className="text-[15px] ml-[50px]" style={{ color: C.ink2 }}>Integer contact graph of P450cam: the heme's σ exceeds every residue's (max {d3.max(DATA.fold.sigma["2CPP"].values)}).</div>
        </Reveal>
        <Reveal step={step} at={2}>
          <LigandChart />
          <div className="text-[15px] ml-[56px]" style={{ color: C.ink2 }}>
            All 26 thiolate irons are Cys Sγ (P450, CPO, NOS). PS00086 matches {pct(sig.p450_match_fraction)} of {sig.p450_reviewed.toLocaleString()} P450s, {pct(sig.heme_non_p450_match_fraction, 2)} of {sig.heme_non_p450_reviewed.toLocaleString()} other heme proteins; ligand Cys = motif Cys in 12/12 structures.
          </div>
        </Reveal>
        <Reveal step={step} at={3} className="col-span-2">
          <div className="flex gap-6 items-start">
            <SpinSwitch />
            <div className="flex-1 flex flex-col gap-3 pt-2">
              <Statement accent={C.orange}>Substrate displaces the distal water and the Fe(III) moves out of plane: low spin → high spin, the d⁵ switch between the two ladders. The reduction potential rises, gating the first electron.</Statement>
              <div className="flex gap-2 flex-wrap"><Badge k="D" /> <Badge k="S" /> <Badge k="C" /> <span className="text-[15px]" style={{ color: C.ink2 }}>every link in the manuscript carries one of these labels</span></div>
            </div>
          </div>
        </Reveal>
        <Reveal step={step} at={3}>
          <Statement accent={C.violet}>The 450-nm band of Fe(II)–CO needs the thiolate. P420, where ligation is lost, is a state the enzyme can occupy that closes nothing — a refused rung.</Statement>
        </Reveal>
      </div>
    </SlideFrame>
  );
}
