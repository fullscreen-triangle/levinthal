import * as d3 from "d3";
import { C, fmt, pct } from "../theme";
import useD3, { styleAxis, axisLabel } from "../useD3";
import { SlideFrame, Reveal, Statement, Compare, Legend, Box } from "../ui";
import { PAPERS, STATES, ORBIT_DM, HETERO_DM, ATYPICAL, CYP2D6, INHIBITORS, CLARITHROMYCIN } from "@/data/talk/applications";

// ------------------------------------------------------ 16 · C–H activation, heteroatoms
function Rebound() {
  const W = 700, H = 300;
  const ref = useD3((svg) => {
    const fe = { x: 120, y: 200 }, O = { x: 220, y: 200 };
    svg.append("circle").attr("cx", fe.x).attr("cy", fe.y).attr("r", 24).attr("fill", C.orange);
    svg.append("text").attr("x", fe.x).attr("y", fe.y + 6).attr("text-anchor", "middle").attr("fill", C.bg).style("font-size", "14px").text("Fe⁴⁺");
    svg.append("line").attr("x1", fe.x + 24).attr("x2", O.x - 18).attr("y1", fe.y).attr("y2", O.y).attr("stroke", C.ink).attr("stroke-width", 5);
    const o = svg.append("circle").attr("cx", O.x).attr("cy", O.y).attr("r", 18).attr("fill", C.red);
    const c = svg.append("circle").attr("cx", 480).attr("cy", 200).attr("r", 20).attr("fill", C.muted);
    svg.append("text").attr("x", 480).attr("y", 244).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "15px").text("substrate C");
    const hAtom = svg.append("circle").attr("r", 10).attr("fill", C.ink);
    const lab = svg.append("text").attr("x", 20).attr("y", 40).attr("fill", C.ink).style("font-size", "20px");
    const lab2 = svg.append("text").attr("x", 20).attr("y", 68).attr("fill", C.ink2).style("font-size", "15px");
    let alive = true;
    const loop = () => {
      if (!alive) return;
      hAtom.attr("cx", 450).attr("cy", 180).attr("opacity", 1);
      c.attr("cx", 480).attr("fill", C.muted);
      o.attr("cx", O.x).attr("cy", O.y);
      lab.text("1 · hydrogen-atom transfer"); lab2.text("C–H bond order 1 → 0, O–H 0 → 1 · rate-limiting · primary KIE");
      hAtom.transition().delay(500).duration(1300).attr("cx", O.x + 14).attr("cy", O.y - 22)
        .on("end", () => {
          c.attr("fill", C.magenta);
          lab.text("2 · oxygen rebound"); lab2.text("the radical recombines with Fe–OH · faster than abstraction in the model");
          o.transition().duration(1100).attr("cx", 440).attr("cy", 196);
          hAtom.transition().duration(1100).attr("cx", 432).attr("cy", 172)
            .on("end", () => { lab.text("3 · alcohol product"); lab2.text("C–OH formed, Fe returns to Fe(III)"); d3.timeout(loop, 1600); });
        });
    };
    loop();
    return () => { alive = false; };
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function HeteroBars() {
  const W = 700, H = 210, m = { l: 150, r: 30, t: 6, b: 40 };
  const ref = useD3((svg) => {
    const y = d3.scaleBand().domain(HETERO_DM.map((d) => d[0])).range([m.t, H - m.b]).padding(0.3);
    const x = d3.scaleLinear().domain([0, 0.7]).range([m.l, W - m.r]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(7)));
    axisLabel(svg, "assigned depth ΔM (model)", { x: (W + m.l) / 2, y: H - 4 });
    svg.selectAll("text.l").data(HETERO_DM).join("text").attr("class", "l").attr("x", m.l - 10).attr("y", (d) => y(d[0]) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "15px").text((d) => d[0]);
    svg.selectAll("rect").data(HETERO_DM).join("rect").attr("x", m.l).attr("y", (d) => y(d[0])).attr("height", y.bandwidth()).attr("rx", 3)
      .attr("fill", (d) => (d[2].startsWith("direct") ? C.aqua : C.blue)).attr("width", 0)
      .transition().duration(700).delay((_, i) => i * 120).attr("width", (d) => x(d[1]) - m.l);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S16CH({ step }) {
  return (
    <SlideFrame
      kicker="Applications · C–H activation and heteroatoms"
      papers={[PAPERS.ch, PAPERS.hetero]}
      title="Abstraction, then rebound; direct transfer is faster at heteroatoms"
      lead="One three-body step (C, H, ferryl O), followed by rebound. Heteroatom reactions are placed on the same activation scale, with depths assigned per reaction class."
    >
      <div className="absolute inset-0 grid grid-cols-2 gap-8">
        <div className="flex flex-col gap-2">
          <Rebound />
          <Reveal step={step} at={1}>
            <div className="text-[17px] mb-1" style={{ color: C.ink }}>Heteroatom oxidation: direct O-transfer (aqua) vs via H abstraction (blue)</div>
            <HeteroBars />
          </Reveal>
        </div>
        <div className="flex flex-col gap-3">
          <Statement>A primary isotope effect appears only where hydrogen moves in the rate-limiting step: the established mechanistic picture, which the papers adopt. Direct O-transfer comes out faster than dealkylation because its depths are set lower.</Statement>
          <Reveal step={step} at={2}>
            <Compare rows={[
              ["KIE, aliphatic C–H, 310 K", "9.6", "4–11 reported", "M"],
              ["testosterone 6β share (CYP3A4)", "49% (28% without g)", "50–70% reported", "M"],
              ["k rebound / k HAT", "1.42", "set by the two depths", "M"],
              ["KIE, N-dealkylation", "6.7", "not compared with experiment", "M"],
              ["S-ox / N-dealk rate", "1.25×", "set by the assigned depths", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 17 · atypical reactions
function Atypical() {
  const W = 820, H = 400, m = { l: 220, r: 40, t: 20, b: 50 };
  const ref = useD3((svg) => {
    const y = d3.scaleBand().domain(ATYPICAL.map((d) => d.name)).range([m.t, H - m.b]).padding(0.3);
    const x = d3.scaleLinear().domain([0, 0.5]).range([m.l, W - m.r]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5)), { grid: true, height: H - m.t - m.b });
    axisLabel(svg, "assigned depth ΔM (model)", { x: (W + m.l) / 2, y: H - 8 });
    ATYPICAL.forEach((d, i) => {
      const yy = y(d.name);
      svg.append("text").attr("x", m.l - 12).attr("y", yy + y.bandwidth() / 2 + 5).attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "16px").text(d.name);
      if (d.dm != null) {
        svg.append("rect").attr("x", m.l).attr("y", yy).attr("height", y.bandwidth()).attr("rx", 3).attr("fill", C.blue).attr("width", 0)
          .transition().duration(700).delay(i * 140).attr("width", x(d.dm) - m.l);
        svg.append("text").attr("x", x(d.dm) + 8).attr("y", yy + y.bandwidth() / 2 + 5).attr("fill", C.ink2).style("font-size", "14px").text(`KIE ${d.kie}`)
          .attr("opacity", 0).transition().delay(800 + i * 140).attr("opacity", 1);
      } else {
        svg.append("text").attr("x", m.l + 8).attr("y", yy + y.bandwidth() / 2 + 5).attr("fill", C.orange).style("font-size", "15px")
          .text(`two sequential HATs · effective k ≈ 2.3×10⁹ s⁻¹ · primary KIE ${d.kie}`);
      }
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S17Atypical({ step }) {
  return (
    <SlideFrame
      kicker="Applications · beyond hydroxylation"
      papers={[PAPERS.atypical]}
      title="Five atypical reactions on one scale"
      lead="NIH shift, carbene insertion, epoxidation, nucleophilic O-atom transfer and desaturation, each given an assigned activation depth. The rate order follows from those assignments; the isotope-effect pattern is the mechanistic expectation."
    >
      <div className="absolute inset-0 grid grid-cols-[820px_1fr] gap-8">
        <Atypical />
        <div className="flex flex-col gap-3">
          <Statement accent={C.orange}>Desaturation is the only class with a primary KIE in the model (two hydrogen abstractions); the others are assigned KIE ≈ 1 because no hydrogen moves in their rate step.</Statement>
          <Reveal step={step} at={1}>
            <Compare rows={[
              ["NIH shift rate", "8.4×10⁹ s⁻¹", "no cited measurement", "M"],
              ["desaturation, effective k", "2.3×10⁹ s⁻¹", "two HATs against rebound", "M"],
              ["KIE, desaturation", "≈ 5", "not yet compared with experiment", "M"],
              ["rate order", "NIH > carbene > epox > nucl > desat", "follows from the assigned depths", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 18 · cycle and spectra
const nmColor = d3.scaleLinear().domain([350, 380, 410, 440, 470]).range(["#5b4a86", "#7b3fd0", "#5a46e6", "#3a63f2", "#2d86e8"]);

function CycleWheel({ step }) {
  const W = 760, H = 560;
  const ref = useD3((svg) => {
    const cx = 300, cy = 290, R = 210;
    const ang = (i) => (i / STATES.length) * 2 * Math.PI - Math.PI / 2;
    const pos = STATES.map((_, i) => ({ x: cx + R * Math.cos(ang(i)), y: cy + R * Math.sin(ang(i)) }));
    const arcW = d3.scaleLinear().domain([0.3, 0.92]).range([2, 10]);
    STATES.forEach((_, i) => {
      const a = pos[i], b = pos[(i + 1) % STATES.length];
      const dm = ORBIT_DM[i][2];
      svg.append("path").attr("d", `M${a.x},${a.y} Q${cx + (a.x + b.x - 2 * cx) * 0.62},${cy + (a.y + b.y - 2 * cy) * 0.62} ${b.x},${b.y}`)
        .attr("fill", "none").attr("stroke", C.line).attr("stroke-width", arcW(dm));
    });
    const nodes = svg.selectAll("g.s").data(STATES).join("g").attr("class", "s").attr("transform", (_, i) => `translate(${pos[i].x},${pos[i].y})`);
    nodes.append("circle").attr("r", 40).attr("fill", C.panel).attr("stroke", (d) => nmColor(d.soret)).attr("stroke-width", 3);
    nodes.append("text").attr("y", -4).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "14px").text((d) => d.id);
    nodes.append("text").attr("y", 14).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "11px").text((d) => d.label);
    const title = svg.append("text").attr("x", cx).attr("y", cy - 10).attr("text-anchor", "middle").attr("fill", C.ink).style("font-size", "22px");
    const sub = svg.append("text").attr("x", cx).attr("y", cy + 18).attr("text-anchor", "middle").attr("fill", C.ink2).style("font-size", "15px");
    // spectrum axis
    const sx = d3.scaleLinear().domain([350, 470]).range([560, 740]);
    const grad = svg.append("defs").append("linearGradient").attr("id", "spec");
    d3.range(350, 471, 10).forEach((nm) => grad.append("stop").attr("offset", `${((nm - 350) / 120) * 100}%`).attr("stop-color", nmColor(nm)));
    svg.append("rect").attr("x", 560).attr("y", 60).attr("width", 180).attr("height", 22).attr("fill", "url(#spec)").attr("rx", 3);
    styleAxis(svg.append("g").attr("transform", "translate(0,84)").call(d3.axisBottom(sx).tickValues([360, 400, 440])));
    svg.append("text").attr("x", 560).attr("y", 48).attr("fill", C.ink2).style("font-size", "15px").text("Soret (nm)");
    svg.append("line").attr("x1", sx(450)).attr("x2", sx(450)).attr("y1", 54).attr("y2", 88).attr("stroke", C.ink).attr("stroke-dasharray", "3,3");
    svg.append("text").attr("x", sx(450)).attr("y", 130).attr("text-anchor", "middle").attr("fill", C.muted).style("font-size", "13px").text("Fe(II)–CO 450");
    const marker = svg.append("path").attr("d", d3.symbol(d3.symbolTriangle, 160)()).attr("fill", C.ink);
    let i = 0;
    const show = () => {
      const s = STATES[i];
      nodes.select("circle").transition().duration(300).attr("fill", (_, j) => (j === i ? "#2c2c29" : C.panel)).attr("r", (_, j) => (j === i ? 46 : 40));
      title.text(s.name);
      sub.text(`${s.label}${s.spin ? " · " + s.spin : ""} · Soret ${s.soret} nm`);
      marker.transition().duration(500).attr("transform", `translate(${sx(s.soret)},48) rotate(180)`);
      i = (i + 1) % STATES.length;
    };
    show();
    const it = d3.interval(show, 1800);
    return () => it.stop();
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S18Cycle({ step }) {
  return (
    <SlideFrame
      kicker="Applications · the cycle and its spectra"
      papers={[PAPERS.orbit, PAPERS.spectra]}
      title="Seven distinguishable states, each with its own spectrum"
      lead="The closed-cycle paper gives each state a distinct label; the spectroscopic atlas tabulates each state's Soret band. Resting, substrate-bound and Compound I are measured values; the others are estimates. Arc width: the activation depth assigned to each transition."
    >
      <div className="absolute inset-0 grid grid-cols-[760px_1fr] gap-8">
        <CycleWheel />
        <div className="flex flex-col gap-3">
          <Statement>Read as a ladder (derivation, slide 6): eight rungs, one commitment each, and the coupling efficiency as the composite. Recording each intermediate commits once per rung.</Statement>
          <Reveal step={step} at={1}>
            <Compare rows={[
              ["resting LS Soret", "417 nm", "literature", "L"],
              ["substrate-bound HS Soret", "≈ 390 nm", "literature", "L"],
              ["EPR g, LS / HS", "2.42, 2.25, 1.92 / ≈ 7.7, 3.5, 1.8", "P450cam, literature", "L"],
              ["Compound I Soret", "367 nm", "Rittle & Green 2010", "L"],
              ["ferrous, oxy, peroxo, Cpd 0 Soret", "408, 418, 440, 367 nm", "estimates in the atlas", "M"],
              ["sum of ΔM over the cycle", "4.96", "sum of assigned depths", "M"],
            ]} />
          </Reveal>
        </div>
      </div>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 19 · pharmacology
function Phenotypes() {
  const W = 460, H = 300, m = { l: 150, r: 20, t: 10, b: 44 };
  const ref = useD3((svg) => {
    const y = d3.scaleBand().domain(CYP2D6.map((d) => d.phenotype)).range([m.t, H - m.b]).padding(0.3);
    const x = d3.scaleLinear().domain([0, 1.4]).range([m.l, W - m.r]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4)));
    axisLabel(svg, "relative activity (EM = 1)", { x: (W + m.l) / 2, y: H - 6 });
    svg.selectAll("text.l").data(CYP2D6).join("text").attr("class", "l").attr("x", m.l - 10).attr("y", (d) => y(d.phenotype) + y.bandwidth() / 2 + 5)
      .attr("text-anchor", "end").attr("fill", C.ink).style("font-size", "15px").text((d) => d.phenotype);
    svg.selectAll("rect").data(CYP2D6).join("rect").attr("x", m.l).attr("y", (d) => y(d.phenotype)).attr("height", y.bandwidth()).attr("rx", 3)
      .attr("fill", C.blue).attr("width", 0).transition().duration(700).attr("width", (d) => x(d.rel) - m.l);
    svg.selectAll("text.f").data(CYP2D6).join("text").attr("class", "f").attr("x", (d) => x(d.rel) + 6).attr("y", (d) => y(d.phenotype) + y.bandwidth() / 2 + 5)
      .attr("fill", C.ink2).style("font-size", "13px").text((d) => d.alleles);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Inhibition() {
  const W = 460, H = 300, m = { l: 56, r: 16, t: 10, b: 44 };
  const ref = useD3((svg) => {
    const x = d3.scaleLog().domain([0.001, 10]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4, "~g")));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4, "%")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "inhibitor concentration (µM)", { x: (W + m.l) / 2, y: H - 6 });
    const cols = [C.blue, C.orange, C.aqua];
    INHIBITORS.forEach((inh, i) => {
      const pts = d3.range(-3, 1.01, 0.05).map((l) => [10 ** l, 1 / (1 + 10 ** l / inh.ki)]);
      const p = svg.append("path").attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(pts)).attr("fill", "none").attr("stroke", cols[i]).attr("stroke-width", 2.6);
      const len = p.node().getTotalLength();
      p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().delay(i * 300).duration(1200).attr("stroke-dashoffset", 0);
      const a = 1 + inh.conc / inh.ki;
      svg.append("circle").attr("cx", x(inh.conc)).attr("cy", y(1 / a)).attr("r", 6).attr("fill", cols[i]).attr("stroke", C.surface);
      svg.append("text").attr("x", x(inh.conc) + 9).attr("y", y(1 / a) - 6 - i * 2).attr("fill", C.ink2).style("font-size", "13px").text(`${inh.name}: α = ${a.toFixed(1)}`);
    });
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

function Inactivation() {
  const W = 460, H = 300, m = { l: 56, r: 16, t: 10, b: 44 };
  const ref = useD3((svg) => {
    const { KI, kinact, conc } = CLARITHROMYCIN;
    const kobs = (kinact * conc) / (KI + conc);
    const x = d3.scaleLinear().domain([0, 120]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4, "%")), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "minutes", { x: (W + m.l) / 2, y: H - 6 });
    const pts = d3.range(0, 121, 2).map((t) => [t, Math.exp(-kobs * t)]);
    const p = svg.append("path").attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1]))(pts)).attr("fill", "none").attr("stroke", C.violet).attr("stroke-width", 2.6);
    const len = p.node().getTotalLength();
    p.attr("stroke-dasharray", `${len} ${len}`).attr("stroke-dashoffset", len).transition().duration(1400).attr("stroke-dashoffset", 0);
    svg.append("circle").attr("cx", x(60)).attr("cy", y(Math.exp(-kobs * 60))).attr("r", 6).attr("fill", C.violet);
    svg.append("text").attr("x", x(60) + 10).attr("y", y(Math.exp(-kobs * 60)) - 8).attr("fill", C.ink2).style("font-size", "13px")
      .text(`60 min: ${pct(Math.exp(-kobs * 60), 0)} active`);
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S19Pharmacology({ step }) {
  return (
    <SlideFrame
      kicker="Applications · pharmacology"
      papers={[PAPERS.ddi, PAPERS.pgx]}
      title="Alleles, inhibitors and inactivators as shifts of one activation parameter"
      lead={<>Genotype moves ΔM; a competitive inhibitor adds ln α with α = 1 + [I]/K<sub>i</sub>; a mechanism-based inactivator removes enzyme at k<sub>obs</sub> = k<sub>inact</sub>[I]/(K<sub>I</sub> + [I]).</>}
    >
      <div className="grid grid-cols-3 gap-6">
        <div>
          <div className="text-[17px] mb-1" style={{ color: C.ink }}>CYP2D6 phenotypes in the model · assigned depths</div>
          <Phenotypes />
        </div>
        <Reveal step={step} at={1}>
          <div className="text-[17px] mb-1" style={{ color: C.ink }}>competitive inhibition, residual activity 1/α</div>
          <Inhibition />
        </Reveal>
        <Reveal step={step} at={2}>
          <div className="text-[17px] mb-1" style={{ color: C.ink }}>clarithromycin on CYP3A4, 4 µM (Kitz–Wilson)</div>
          <Inactivation />
        </Reveal>
      </div>
      <Reveal step={step} at={3} className="mt-3">
        <Compare rows={[
          ["CYP2D6 PM (*4/*5) activity", "14% of EM", "≈ 0 in null-allele carriers", "M"],
          ["CYP2C9*3 residual", "4.4%", "depth set to give ≈ 5%", "M"],
          ["rifampicin 20× induction → victim AUC", "0.05 = 1/20", "induction from Backman 1996", "L"],
          ["codeine in UM: morphine formed", "1.32× EM", "FDA boxed warning (no numeric threshold)", "M"],
        ]} />
      </Reveal>
    </SlideFrame>
  );
}

// ------------------------------------------------------ 20 · informatics and close
function Capacity() {
  const W = 620, H = 300, m = { l: 56, r: 16, t: 14, b: 44 };
  const ref = useD3((svg) => {
    const x = d3.scaleBand().domain(d3.range(1, 10)).range([m.l, W - m.r]).padding(0.2);
    const y = d3.scaleLinear().domain([0, 15]).range([H - m.b, m.t]);
    styleAxis(svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x)));
    styleAxis(svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)), { grid: true, width: W - m.l - m.r, orient: "y" });
    axisLabel(svg, "address depth k", { x: (W + m.l) / 2, y: H - 6 });
    axisLabel(svg, "bits", { x: 16, y: H / 2, rotate: true });
    svg.selectAll("rect").data(d3.range(1, 10)).join("rect").attr("x", (k) => x(k)).attr("width", x.bandwidth())
      .attr("y", H - m.b).attr("height", 0).attr("fill", (k) => (k === 6 ? C.blue : C.line))
      .transition().duration(700).delay((k) => k * 60).attr("y", (k) => y(k * Math.log2(3))).attr("height", (k) => H - m.b - y(k * Math.log2(3)));
    const e57 = Math.log2(57);
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(e57)).attr("y2", y(e57)).attr("stroke", C.orange).attr("stroke-dasharray", "6,4");
    svg.append("text").attr("x", m.l + 6).attr("y", y(e57) - 6).attr("fill", C.ink2).style("font-size", "14px").text(`log₂ 57 = ${e57.toFixed(2)} bits`);
    svg.append("line").attr("x1", x(6)).attr("x2", x(6) + x.bandwidth()).attr("y1", y(6.66)).attr("y2", y(6.66)).attr("stroke", C.aqua).attr("stroke-width", 3);
    svg.append("text").attr("x", x(6) + x.bandwidth() + 6).attr("y", y(6.66) + 4).attr("fill", C.ink2).style("font-size", "13px").text("70% of a k = 6 address: 6.66 bits");
  }, []);
  return <svg ref={ref} width={W} height={H} />;
}

export function S20Informatics({ step }) {
  return (
    <SlideFrame
      kicker="Applications · informatics, and what this offers"
      papers={[PAPERS.database, PAPERS.dictionary]}
      title="Addresses instead of entries — and checks a lab can run today"
      lead="The addressing scheme stores 20 residue rows and an encoding rule, and answers queries for sequences it has never seen. Capacity at depth 6 exceeds what 57 isoforms need, which is what would make recovery from partial addresses possible."
    >
      <div className="absolute inset-0 grid grid-cols-[620px_1fr] gap-8">
        <div className="flex flex-col gap-3">
          <Capacity />
          <Statement accent={C.aqua}>Resident state is constant: 20 rows + a rule. Address distance and residue similarity, scored on physicochemical scales held outside the table, are rank-correlated: ρ = 0.382; none of 200 shuffled tables reached it.</Statement>
        </div>
        <Reveal step={step} at={1}>
          <Box style={{ borderLeft: `3px solid ${C.orange}` }}>
            <div className="text-[24px] font-semibold mb-3">What I would like to test with you</div>
            <ul className="text-[19px] leading-[1.55] list-disc pl-6" style={{ color: C.ink2 }}>
              <li><b style={{ color: C.ink }}>Pre-screens from structure or sequence</b> — N-terminal anchor, cysteine-loop signature, heme burial, distal water — each computed in seconds from public data, as shown in part I.</li>
              <li><b style={{ color: C.ink }}>Separation cost per residue</b> in an integer contact graph as a candidate ranking for mutagenesis — to be tested against your variant data, not assumed.</li>
              <li><b style={{ color: C.ink }}>Substitution accounting for library design</b> — which single-nucleotide changes stay near the parent residue, codon by codon.</li>
              <li><b style={{ color: C.ink }}>Coupling as a ladder</b> — product per NADPH as the composite, intermediates as rungs, for engineered P450 or BVMO variants.</li>
            </ul>
            <div className="mt-4 text-[16px]" style={{ color: C.muted }}>
              Code, data and both manuscripts: validation suite reproduces every number in part I with one command.
            </div>
          </Box>
          <div className="mt-6 text-[34px] font-semibold">Thank you.</div>
        </Reveal>
      </div>
    </SlideFrame>
  );
}
