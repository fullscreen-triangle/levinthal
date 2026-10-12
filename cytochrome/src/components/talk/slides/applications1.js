import * as d3 from "d3";
import dynamic from "next/dynamic";
import { C, fmt } from "../theme";
import useD3, { styleAxis, axisLabel } from "../useD3";
import { SlideFrame, Reveal, Statement, Compare, Legend } from "../ui";
import { moserDutton } from "../util";
import { PAPERS, HUMAN_CYPS, GLB_FE, DEPTHS } from "@/data/talk/applications";
import DATA from "@/data/talk/derivation.json";
import { HEME_MARKER } from "@/data/glbMarkers";

// the site's own 3-D viewers, client-only
const GLBViewer = dynamic(() => import("@/components/GLBViewer"), { ssr: false });
const ElectronTransferViewer = dynamic(() => import("@/components/ElectronTransferViewer"), { ssr: false });

// ------------------------------------------------------ 11 · structure input, address manifold
function FeDistances() {
  const W = 520, H = 290, m = { l: 190, r: 16, t: 20, b: 46 };
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([1.7, 2.6]).range([m.l, W - m.r]);
    const y = d3.scaleBand().domain(GLB_FE.map((d) => d.bond)).range([m.t, H - m.b]).padding(0.35);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)), { grid: true, height: H - m.t - m.b });
    axisLabel(svg, "distance from Fe (Å)", { x: (W + m.l) / 2, y: H - 6 });
    GLB_FE.forEach((d) => {
      const yy = y(d.bond) + y.bandwidth() / 2;
      svg.append("text").attr("x", m.l - 12).attr("y", yy + 5).attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "16px").text(d.bond);
      const range = d.pdb || d.lit;
      svg.append("rect").attr("x", x(range[0])).attr("width", 0).attr("y", yy - 9).attr("height", 18).attr("rx", 9)
        .attr("fill", d.pdb ? C.blue : C.violet).attr("opacity", 0.55)
        .transition().duration(800).attr("width", x(range[1]) - x(range[0]));
      if (d.pdbMean) svg.append("line").attr("x1", x(d.pdbMean)).attr("x2", x(d.pdbMean)).attr("y1", yy - 12).attr("y2", yy + 12).attr("stroke", C.ink).attr("stroke-width", 2);
      d3.range(d.glb[0], d.glb[1] + 1e-9, Math.max(d.glb[1] - d.glb[0], 1)).forEach((v) =>
        svg.append("path").attr("d", d3.symbol(d3.symbolDiamond, 140)()).attr("transform", `translate(${x(v)},${yy})`).attr("fill", C.orange)
          .attr("opacity", 0).transition().delay(700).attr("opacity", 1));
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Depths() {
  const W = 470, H = 290, m = { l: 60, r: 10, t: 20, b: 46 };
  const ref = useD3((svg) => {
    const x = d3.scaleBand().domain(d3.range(1, 10)).range([m.l, W - m.r]).padding(0.2);
    const y = d3.scaleLog().domain([1, 30000]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, "~s")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "address depth k (trits)", { x: (W + m.l) / 2, y: H - 6 });
    axisLabel(svg, "addresses 3ᵏ", { x: 14, y: H / 2, rotate: true });
    svg.selectAll("rect").data(d3.range(1, 10)).join("rect").attr("x", (k) => x(k)).attr("width", x.bandwidth())
      .attr("y", H - m.b).attr("height", 0).attr("fill", (k) => (DEPTHS.some((d) => d.k === k) ? C.blue : C.line))
      .transition().duration(700).delay((k) => k * 70).attr("y", (k) => y(3 ** k)).attr("height", (k) => H - m.b - y(3 ** k));
    [[18, "18 families"], [57, "57 human isoforms"]].forEach(([v, t]) => {
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(v)).attr("y2", y(v)).attr("stroke", C.orange).attr("stroke-dasharray", "6,4");
      svg.append("text").attr("x", m.l + 6).attr("y", y(v) - 6).attr("fill", C.ink2).style("font-size", "14px").text(t);
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S11Structure({ step }) {
  return (
    <SlideFrame
      kicker="Applications · structure and addressing"
      papers={[PAPERS.glb, PAPERS.manifold]}
      title="From a 3-D model to an address: iron geometry and ternary depth"
      lead="A GLB model of CYP3A4 yields per-atom positions; its iron coordination is compared with the twelve crystal structures measured in the derivation. Residue coordinates are then addressed in base three."
    >
      <div className="grid grid-cols-[400px_520px_1fr] gap-6 items-start">
        <GLBViewer height={290} preset="heme" markers={HEME_MARKER} badge="GLB · Fe coordination" />
        <div>
          <FeDistances />
          <Legend items={[["PDB, 12 P450s", C.blue], ["GLB model", C.orange], ["literature", C.violet]]} />
        </div>
        <Reveal step={step} at={1}><Depths /></Reveal>
      </div>
      <Reveal step={step} at={2} className="mt-3">
        <Compare rows={[
          ["Fe–S (Cys)", "2.228 Å", "1.90–2.52 Å, mean 2.32 (PDB)", "C"],
          ["Fe–N (pyrrole)", "2.01–2.04 Å", "1.98–2.10 Å (PDB)", "C"],
          ["family separation at k = 3", "recall 0.94", "Nelson families", "M"],
          ["isoform separation at k = 6", "0.97", "57 human CYPs", "M"],
          ["CYP3A4 fold, log₃ 503", "5.69 steps", "address refinements", "M"],
        ]} />
      </Reveal>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 12 · isoforms
const DRUG = new Set(["3A4", "2D6", "2C9", "2C19", "1A2", "2B6", "2E1", "3A5", "2C8"]);

function IsoformPack() {
  const W = 760, H = 560;
  const ref = useD3((svg) => {
    const root = d3.hierarchy({
      name: "human CYP",
      children: Object.entries(HUMAN_CYPS).map(([fam, genes]) => ({ name: fam, children: genes.map((g) => ({ name: g, value: 1 })) })),
    }).sum((d) => d.value || 0).sort((a, b) => b.value - a.value);
    d3.pack().size([W, H]).padding(5)(root);
    const famColor = d3.scaleOrdinal().domain(Object.keys(HUMAN_CYPS)).range([C.blue, C.aqua, C.orange, C.violet, C.magenta, C.yellow]);
    const fams = svg.selectAll("g.f").data(root.children).join("g").attr("class", "f");
    fams.append("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y).attr("r", 0).attr("fill", "none").attr("stroke", (d) => famColor(d.data.name)).attr("stroke-width", 1.6)
      .transition().duration(700).delay((_, i) => i * 50).attr("r", (d) => d.r);
    fams.filter((d) => d.r > 34).append("text").attr("x", (d) => d.x).attr("y", (d) => d.y - d.r + 18).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "14px").text((d) => d.data.name);
    const leaves = svg.selectAll("g.l").data(root.leaves()).join("g").attr("class", "l");
    leaves.append("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y).attr("r", 0)
      .attr("fill", (d) => (DRUG.has(d.data.name) ? famColor(d.parent.data.name) : C.panel))
      .attr("stroke", (d) => famColor(d.parent.data.name)).attr("stroke-width", 1.2)
      .transition().duration(600).delay((_, i) => 400 + i * 15).attr("r", (d) => d.r);
    leaves.append("text").attr("x", (d) => d.x).attr("y", (d) => d.y + 4).attr("text-anchor", "middle")
      .attr("fill", (d) => (DRUG.has(d.data.name) ? C.bg : C.ink2)).style("font-size", (d) => `${Math.min(13, d.r * 0.75)}px`)
      .text((d) => d.data.name).attr("opacity", 0).transition().delay(1200).attr("opacity", 1);
    const singles = root.children.filter((d) => d.r <= 34);
    svg.append("text").attr("x", W - 4).attr("y", H - 6).attr("text-anchor", "end").attr("fill", C.muted).style("font-size", "13px")
      .text(`${singles.length} small families unlabelled: ${singles.map((d) => d.data.name.replace("CYP", "")).join(", ")}`);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S12Isoforms({ step }) {
  return (
    <SlideFrame
      kicker="Applications · the human P450 complement"
      papers={[PAPERS.isoforms, PAPERS.taxonomy]}
      title="57 human P450s in 18 families, separated by address depth"
      lead="Filled circles: the isoforms that carry most drug oxidation. The papers ask at which ternary depth the family and isoform partitions of this tree are recovered."
    >
      <div className="absolute inset-0 grid grid-cols-[760px_1fr] gap-10">
        <IsoformPack />
        <div className="flex flex-col gap-4">
          <Statement>3³ = 27 ≥ 18 families: depth k = 3 suffices in capacity; the paper reports family recall 0.94 against the Nelson nomenclature.</Statement>
          <Reveal step={step} at={1}><Statement accent={C.orange}>3⁶ = 729 ≥ 57 isoforms: depth k = 6, distinctness 0.97. Depth k = 9 (19,683) leaves room for catalogued alleles.</Statement></Reveal>
          <Reveal step={step} at={1}><Statement accent={C.violet}>Family-level differences in substrate breadth are expressed as ranges of the model parameter ΔM; CYP3A4's is the widest in the paper.</Statement></Reveal>
          <Reveal step={step} at={2}>
            <Compare rows={[
              ["families / genes", "18 / 57", "Nelson nomenclature", "L"],
              ["family recall, k = 3", "0.94", "—", "M"],
              ["isoform distinctness, k = 6", "0.97", "—", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 13 · resting ⇄ substrate-bound
function SpinGate({ step }) {
  const W = 760, H = 470;
  const d = DATA.ligation.distal;
  const ref = useD3((svg) => {
    const cx = 260, py = 250;
    svg.append("line").attr("x1", cx - 170).attr("x2", cx + 170).attr("y1", py).attr("y2", py).attr("stroke", C.ink2).attr("stroke-width", 6);
    svg.append("text").attr("x", cx + 178).attr("y", py + 5).attr("fill", C.muted).style("font-size", "15px").text("porphyrin");
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", py + 70).attr("y2", py + 150).attr("stroke", C.yellow).attr("stroke-width", 6);
    svg.append("text").attr("x", cx + 10).attr("y", py + 150).attr("fill", C.yellow).style("font-size", "16px").text("Cys S⁻");
    const fe = svg.append("circle").attr("cx", cx).attr("cy", py + d["1PHC"].fe_out_of_plane * 80).attr("r", 22).attr("fill", C.orange);
    const feT = svg.append("text").attr("x", cx).attr("y", py + 6).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "15px").style("font-weight", 600).text("Fe³⁺");
    const water = svg.append("circle").attr("cx", cx).attr("cy", py - 90).attr("r", 16).attr("fill", C.blue);
    const wT = svg.append("text").attr("x", cx + 24).attr("y", py - 86).attr("fill", C.ink2).style("font-size", "15px").text("H₂O");
    const sub = svg.append("g").attr("transform", `translate(${cx - 260},${py - 190})`);
    sub.append("ellipse").attr("rx", 42).attr("ry", 26).attr("fill", C.aqua).attr("opacity", 0.85);
    sub.append("text").attr("y", 5).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "14px").text("substrate");
    const spin = svg.append("text").attr("x", 470).attr("y", 60).attr("fill", C.ink).style("font-size", "24px");
    const coord = svg.append("text").attr("x", 470).attr("y", 92).attr("fill", C.ink2).style("font-size", "16px");
    const soret = svg.append("text").attr("x", 470).attr("y", 120).attr("fill", C.ink2).style("font-size", "16px");
    // reduction potential bar
    const ex = d3.scaleLinear().domain([-350, -100]).range([470, 720]);
    svg.append("text").attr("x", 470).attr("y", 200).attr("fill", C.ink2).style("font-size", "15px").text("Fe³⁺/Fe²⁺ potential (mV)");
    svg.append("g").attr("transform", "translate(0,250)").call(d3.axisBottom(ex).ticks(3)).call((g) => styleAxis(g));
    const bar = svg.append("rect").attr("x", ex(-350)).attr("y", 215).attr("height", 30).attr("rx", 4).attr("fill", C.violet).attr("width", ex(-300) - ex(-350));
    const gate = svg.append("text").attr("x", 470).attr("y", 300).attr("fill", C.ink).style("font-size", "17px");
    const state = (bound) => {
      spin.text(bound ? "high spin, S = 5/2" : "low spin, S = 1/2");
      coord.text(bound ? "five-coordinate, Fe below the plane" : "six-coordinate, distal water");
      soret.text(`Soret ≈ ${bound ? 392 : 417} nm`);
      gate.text(bound ? "first electron from CPR accepted" : "first electron not favourable").attr("fill", bound ? C.aqua : C.ink2);
    };
    state(false);
    if (step >= 1) {
      sub.transition().duration(1400).attr("transform", `translate(${cx},${py - 150})`);
      water.transition().delay(900).duration(900).attr("cx", cx + 230).attr("cy", py - 200).attr("opacity", 0);
      wT.transition().delay(900).duration(600).attr("opacity", 0);
      fe.transition().delay(1500).duration(900).attr("cy", py + d["2CPP"].fe_out_of_plane * 80);
      feT.transition().delay(1500).duration(900).attr("y", py + d["2CPP"].fe_out_of_plane * 80 + 6);
      bar.transition().delay(1800).duration(900).attr("width", ex(-180) - ex(-350));
      const t = d3.timeout(() => state(true), 1700);
      return () => t.stop();
    }
    return undefined;
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

export function S13Equilibrium({ step }) {
  const d = DATA.ligation.distal;
  return (
    <SlideFrame
      kicker="Applications · the first rung of the cycle"
      papers={[PAPERS.equilibrium]}
      title="Substrate binding switches the spin ladder and opens the electron gate"
      lead="The derivation measured the switch in P450cam crystal structures. The equilibrium paper models the same step for CYP3A4 and reproduces the reported shift in reduction potential."
    >
      <div className="absolute inset-0 grid grid-cols-[760px_1fr] gap-10">
        <SpinGate step={step} />
        <div className="flex flex-col gap-4">
          <Statement accent={C.aqua}>Measured (this work): distal water {d["1PHC"].water_distance.toFixed(2)} Å → absent; Fe out of plane {d["1PHC"].fe_out_of_plane.toFixed(2)} → {d["2CPP"].fe_out_of_plane.toFixed(2)} Å (1PHC → 2CPP).</Statement>
          <Reveal step={step} at={1}><Statement>Rung 1 → 2 in the paper: one aperture, ΔM = 0.92; the potential shift follows as ΔE = (k<sub>B</sub>T/e)·n<sub>eff</sub>·ΔM·ln b.</Statement></Reveal>
          <Reveal step={step} at={2}>
            <Compare rows={[
              ["water displaced, Fe moves out", "0.30 → 0.43 Å", "1PHC vs 2CPP", "C"],
              ["Soret, resting → bound", "417 → 392 nm", "≈ 417–420 / ≈ 390 nm", "L"],
              ["redox shift on binding", "+122 mV", "+120 mV (reported)", "M"],
              ["ΔM, LS → HS", "0.918", "—", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 14 · electron delivery
function MoserDutton() {
  const W = 700, H = 330, m = { l: 70, r: 16, t: 14, b: 48 };
  const ref = useD3((svg) => {
    const x = d3.scaleLinear().domain([3.6, 22]).range([m.l, W - m.r]);
    const y = d3.scaleLog().domain([1e2, 1e13]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5, "~e")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "edge-to-edge distance R (Å)", { x: (W + m.l) / 2, y: H - 8 });
    axisLabel(svg, "k (s⁻¹), optimal ΔG", { x: 14, y: H / 2, rotate: true });
    const pts = d3.range(3.6, 22.01, 0.2).map((R) => [R, moserDutton(R)]);
    const p = svg.append("path").attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(pts)).attr("stroke", C.ink2).attr("stroke-width", 2.4).attr("fill", "none");
    const len = p.node().getTotalLength();
    p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().duration(1400).attr("stroke-dashoffset", 0);
    [[4, "flavin hops"], [14, "14 Å (membrane paper)"], [18, "18 Å (ET-chain paper)"]].forEach(([R, t], i) => {
      svg.append("circle").attr("cx", x(R)).attr("cy", y(moserDutton(R))).attr("r", 0).attr("fill", i === 0 ? C.yellow : C.orange)
        .transition().delay(1400 + i * 300).attr("r", 7);
      svg.append("text").attr("x", i === 0 ? x(R) + 12 : x(R) - 12).attr("y", y(moserDutton(R)) + (i === 0 ? 5 : -12))
        .attr("text-anchor", i === 0 ? "start" : "end").attr("fill", C.ink2).style("font-size", "14px")
        .text(`${t}: ${moserDutton(R).toExponential(1)} s⁻¹`).attr("opacity", 0).transition().delay(1500 + i * 300).attr("opacity", 1);
    });
    svg.append("path").attr("d", d3.symbol(d3.symbolDiamond, 120)()).attr("transform", `translate(${x(14)},${y(5e6)})`).attr("fill", C.blue);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S14Delivery({ step }) {
  return (
    <SlideFrame
      kicker="Applications · delivering electrons"
      papers={[PAPERS.membrane, PAPERS.et]}
      title="Two membrane anchors, four cofactors, one long hop"
      lead="The derivation found the N-terminal anchor in 93% of ER-membrane P450s. These papers place CPR and the P450 on the same membrane and follow an electron NADPH → FAD → FMN → heme — shown left on the GLB structure, with occupancy per cofactor."
    >
      <div className="absolute inset-0 grid grid-cols-[680px_1fr] gap-10">
        <div className="flex flex-col gap-2">
          <ElectronTransferViewer height={370} autoplay />
          <Reveal step={step} at={2}>
            <Compare rows={[
              ["N-terminal anchor", "≈ 20 residues", "93% of 532 ER P450s", "C"],
              ["Kd, CPR–P450", "0.1 µM", "0.05–0.5 µM", "L"],
              ["k, FMN → heme", "5×10⁶ s⁻¹", "Moser–Dutton at 14 Å: 6×10⁶", "M"],
              ["lipophilic enrichment, logP 3", "≈ 10×", "—", "M"],
            ]} />
          </Reveal>
        </div>
        <Reveal step={step} at={1}>
          <MoserDutton />
          <div className="text-[16px] leading-relaxed" style={{ color: C.ink2 }}>
            Moser–Dutton ruler, log₁₀ k = 13 − 0.6 (R − 3.6), at optimal driving force. Diamond: the membrane paper's
            k<sub>ET</sub> = 5×10⁶ s⁻¹. The two papers' FMN–heme distances differ by 4 Å — about 2.4 orders of magnitude in rate.
          </div>
        </Reveal>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 15 · Compound I
function OOCleavage({ step }) {
  const W = 760, H = 470;
  const ref = useD3((svg) => {
    const cx = 300, py = 300;
    svg.append("line").attr("x1", cx - 180).attr("x2", cx + 180).attr("y1", py).attr("y2", py).attr("stroke", C.ink2).attr("stroke-width", 6);
    svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", py + 30).attr("y2", py + 110).attr("stroke", C.yellow).attr("stroke-width", 6);
    svg.append("text").attr("x", cx + 10).attr("y", py + 110).attr("fill", C.yellow).style("font-size", "16px").text("Cys S⁻ (push)");
    svg.append("circle").attr("cx", cx).attr("cy", py).attr("r", 22).attr("fill", C.orange);
    const feT = svg.append("text").attr("x", cx).attr("y", py + 6).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "14px").style("font-weight", 600).text("Fe³⁺");
    const o1 = svg.append("circle").attr("cx", cx).attr("cy", py - 70).attr("r", 18).attr("fill", C.red);
    const o2 = svg.append("circle").attr("cx", cx).attr("cy", py - 140).attr("r", 18).attr("fill", C.red);
    const oo = svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", py - 88).attr("y2", py - 122).attr("stroke", C.ink).attr("stroke-width", 5);
    const h = svg.append("circle").attr("cx", cx + 34).attr("cy", py - 160).attr("r", 10).attr("fill", C.ink2);
    const h2 = svg.append("circle").attr("cx", cx - 200).attr("cy", py - 200).attr("r", 10).attr("fill", C.ink2).attr("opacity", 0);
    const label = svg.append("text").attr("x", 40).attr("y", 40).attr("fill", C.ink).style("font-size", "22px").text("Compound 0: Fe(III)–OOH");
    const rad = svg.append("text").attr("x", cx - 240).attr("y", py + 6).attr("fill", C.magenta).style("font-size", "18px").attr("opacity", 0).text("π•⁺");
    // bond-order trajectory
    const x = d3.scaleLinear().domain([0, 1]).range([520, 740]);
    const y = d3.scaleLinear().domain([0, 1]).range([300, 140]);
    styleAxis(svg.append("g").attr("transform", "translate(0,300)").call(d3.axisBottom(x).tickValues([0.5, 1])));
    styleAxis(svg.append("g").attr("transform", "translate(520,0)").call(d3.axisLeft(y).ticks(2)));
    svg.append("text").attr("x", 520).attr("y", 125).attr("fill", C.ink2).style("font-size", "15px").text("O–O bond order");
    svg.append("text").attr("x", 630).attr("y", 336).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "14px").text("progress");
    const traj = svg.append("path").attr("fill", "none").attr("stroke", C.red).attr("stroke-width", 3);
    if (step >= 1) {
      h2.transition().duration(900).attr("opacity", 1).attr("cx", cx + 30).attr("cy", py - 150);
      oo.transition().delay(900).duration(1200).attr("stroke-width", 0.5).attr("opacity", 0.2);
      o2.transition().delay(1500).duration(1100).attr("cx", cx - 170).attr("cy", py - 190);
      h.transition().delay(1500).duration(1100).attr("cx", cx - 140).attr("cy", py - 210);
      h2.transition().delay(1500).duration(1100).attr("cx", cx - 200).attr("cy", py - 210);
      o1.transition().delay(1500).duration(700).attr("cy", py - 52);
      feT.transition().delay(2000).text("Fe⁴⁺");
      rad.transition().delay(2200).attr("opacity", 1);
      const t = d3.timeout(() => label.text("Compound I: Fe(IV)=O π•⁺ + H₂O"), 2200);
      const pts = d3.range(0, 1.001, 0.02).map((p) => [p, 1 / (1 + Math.exp((p - 0.5) * 14))]);
      traj.attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(pts));
      const len = traj.node().getTotalLength();
      traj.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().delay(900).duration(1800).attr("stroke-dashoffset", 0);
      return () => t.stop();
    }
    return undefined;
  }, [step >= 1]);
  return <svg ref={ref} width={W} height={H} />;
}

export function S15CompoundI({ step }) {
  return (
    <SlideFrame
      kicker="Applications · Compound I"
      papers={[PAPERS.cpdI]}
      title="Second protonation, O–O heterolysis, Compound I"
      lead="The paper treats the O–O cleavage as a single step whose bond order runs from 1 to 0 (ΔM = ln 2), and compares the predicted observables of Compound I with the characterisation by Rittle and Green (2010)."
    >
      <div className="absolute inset-0 grid grid-cols-[760px_1fr] gap-8">
        <OOCleavage step={step} />
        <div className="flex flex-col gap-3">
          <Statement accent={C.yellow}>The thiolate ligand from the derivation is the reason this step is heterolytic in P450 rather than reversible O₂ binding as in the globins.</Statement>
          <Reveal step={step} at={2}>
            <Compare rows={[
              ["Compound I lifetime, 193 K", "200 ms", "200 ms", "L"],
              ["Mössbauer δ", "0.10 mm/s", "0.11 mm/s", "L"],
              ["total spin, EPR g", "½, 1.99", "½, 1.99", "L"],
              ["Soret shift", "397 → 367 nm", "367 nm", "L"],
              ["Fe=O stretch (¹⁸O)", "795 (758) cm⁻¹", "as cited in the atlas", "L"],
              ["concerted / sequential PCET", "10×", "—", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}
