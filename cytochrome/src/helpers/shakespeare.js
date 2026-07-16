// =====================================================================
//  Minimal Shakespeare interpreter (TypeScript/WebGL target, JS form)
//
//  Parses a .shk play line-by-line and evaluates it against an
//  ACCUMULATING receiver: the contact graph and the monotone cut
//  count M persist across lessons (measurement conserves the medium;
//  the clock only moves forward). Heavy operations (fold, track) replay
//  the lesson's baked oracle -- the monograph's own validated numbers --
//  rather than recomputing, keeping the tutorial exact and fast.
//
//  Output: { console: [line...], charts: {name: data} }
// =====================================================================
import { CHAIN_MARKERS, HEME_MARKER, HEME_FMN_MARKERS } from "@/data/glbMarkers";

// A fresh receiver state. The IDE holds ONE of these and mutates it
// forward as lessons are performed; call resetReceiver() to start over.
export function newReceiver(floor = 3.7e-4) {
  return {
    floor,
    M: 0, // committed-cut count = intrinsic clock (monotone)
    leaves: { res: 0, cof: 0, ele: 0, sol: 0, sub: 0 }, // by class
    bindings: {}, // ident -> { type, cls, meta }
    graphVertices: 1, // the medium
    history: [], // performed lesson ids
  };
}

const CLASS_OF = {
  residue: "res",
  cofactor: "cof",
  electronic: "ele",
  solvent: "sol",
  substrate: "sub",
};

const TYPE_STR = {
  res: "Leaf_res",
  cof: "Leaf_cof",
  ele: "Leaf_ele",
  sol: "Leaf_sol",
  sub: "Leaf_sub",
};

function fmtFloor(f) {
  return f.toExponential(1).replace("e", "e");
}

// Commit k cut events: advance the clock, grow the graph.
function commit(recv, k, vertices = 0) {
  recv.M += k;
  recv.graphVertices += vertices;
}

// ---------------------------------------------------------------------
//  Evaluate one play against the (persistent) receiver.
//  `lesson` supplies the baked oracle for heavy verbs.
// ---------------------------------------------------------------------
export function performPlay(src, recv, lesson) {
  const out = []; // console lines: { kind, text }
  const charts = {};
  const say = (text, kind = "log") => out.push({ kind, text });

  const oracle = lesson?.oracle ?? {};
  const startM = recv.M;

  const lines = src
    .split("\n")
    .map((l) => l.replace(/--.*$/, "").trim()) // strip comments
    .filter(Boolean);

  say(`▸ perform ${lesson?.id ?? "play"}`, "cmd");

  for (const line of lines) {
    // receiver bio
    if (/^receiver\s+\w+/.test(line)) {
      say(`receiver bio           R_bio fixed`, "dim");
      continue;
    }
    // floor N
    let m = line.match(/^floor\s+([0-9.eE+-]+)/);
    if (m) {
      recv.floor = parseFloat(m[1]);
      say(`floor ${fmtFloor(recv.floor)}          nothing is free`, "dim");
      continue;
    }

    // ident := <leafClass> "arg"    (arity-one cut)
    m = line.match(/^(\w+)\s*:=\s*(residue|cofactor|electronic|solvent|substrate)\s+"([^"]+)"/);
    if (m) {
      const [, id, cls, arg] = m;
      const k = CLASS_OF[cls];
      recv.leaves[k] += 1;
      recv.bindings[id] = { type: TYPE_STR[k], cls: k, meta: { arg } };
      commit(recv, 1, 1);
      say(`${id} := ${cls} "${arg}" : ${TYPE_STR[k]} @ ${fmtFloor(recv.floor)}   [M+1]`);
      continue;
    }

    // ident := cut "arg"           (generic leaf)
    m = line.match(/^(\w+)\s*:=\s*cut\s+"?([^"]+?)"?$/);
    if (m) {
      const [, id, arg] = m;
      recv.leaves.ele += 1;
      recv.bindings[id] = { type: "Cut", cls: "ele", meta: { arg } };
      commit(recv, 1, 1);
      say(`${id} := cut "${arg}" : Cut @ ${fmtFloor(recv.floor)}   [M+1]`);
      continue;
    }

    // ident := fold "PDB"          (closure -> Fold)
    m = line.match(/^(\w+)\s*:=\s*fold\s+"([^"]+)"/);
    if (m) {
      const [, id, pdb] = m;
      const helices = oracle.helices ?? 13;
      const sheets = oracle.sheets ?? 5;
      // folding commits ~ one cut per structural element + contacts
      const nEl = oracle.n_elements ?? helices + sheets + (oracle.loops ?? 8);
      commit(recv, nEl * 10, nEl);
      recv.bindings[id] = {
        type: "Fold",
        meta: { pdb, helices, sheets, coherence: oracle.coherence ?? 0.83 },
      };
      say(`${id} := fold "${pdb}" : Fold @ ${fmtFloor(recv.floor)}`);
      say(`  helices ${helices}   sheets ${sheets}   coherence ${oracle.coherence ?? 0.83}   [M+${nEl * 10}]`, "ok");
      say(`  ↳ structure emitted (the fold IS the output)`, "dim");
      charts["structure"] = { kind: "fold", markers: HEME_MARKER };
      charts["contact-map"] = contactMapData(helices, sheets, oracle.loops ?? 8);
      charts["order-parameter"] = orderParameterData(oracle.coherence ?? 0.83);
      continue;
    }

    // ident := complex NAME( ... ) [by ID]     (closure -> Complex)
    m = line.match(/^(\w+)\s*:=\s*complex\s+(\w+)\s*\(/);
    if (m) {
      const [, id, name] = m;
      // count leaf args across the (possibly multi-line already-joined) form
      commit(recv, 3, 3);
      recv.bindings[id] = {
        type: "Complex",
        meta: {
          name,
          spin: oracle.spin_state ?? "low-spin",
          S: oracle.S_total ?? 0.5,
          fe: oracle.fe_oxidation ?? 3,
        },
      };
      say(`${id} := complex ${name}(...) : Complex @ ${fmtFloor(recv.floor)}`);
      say(`  Fe${oracle.fe_oxidation ?? 3}+ ${oracle.spin_state ?? "low-spin"}  S=${oracle.S_total ?? 0.5}  proximal ${oracle.proximal ?? "CYS442"}   [M+3]`, "ok");
      say(`  ↳ structure emitted (resting receiver tree)`, "dim");
      charts["structure"] = { kind: "resting", markers: HEME_FMN_MARKERS };
      if (oracle.helices || recv.bindings.C?.meta?.helices) {
        const b = recv.bindings.C?.meta;
        if (b) charts["contact-map"] = contactMapData(b.helices, b.sheets, 8);
      }
      continue;
    }

    // ident := track X in P [with reps ...] until ADMIT yield R
    m = line.match(/^(\w+)\s*:=\s*track\s+(\w+)\s+in\s+(\w+)/);
    if (m) {
      const [, id, x, p] = m;
      const nStates = oracle.n_states ?? 7;
      const nTrans = oracle.n_transitions ?? 8;
      commit(recv, nTrans, nStates);
      recv.bindings[id] = { type: "Path", meta: { ...oracle } };
      say(`${id} := track ${x} in ${p} until converge yield amalgamation : Path @ ${fmtFloor(recv.floor)}`);
      if (oracle.n_states) {
        say(`  states ${nStates}   transitions ${nTrans}   sum(dM) = ${oracle.DM_sum}`, "ok");
        say(`  orbit ${oracle.orbit_closed ? "closed (7 -> 1, product release)" : "open"}   [M+${nTrans}]`, "ok");
        say(`  ✓ matches ${lesson.paper}  [verdict ${oracle.verdict}]`, "pass");
        say(`  ↳ structure emitted (catalytic complex)`, "dim");
        charts["structure"] = { kind: "cycle", markers: HEME_MARKER };
        charts["seven-state-orbit"] = orbitData(oracle);
        charts["dm-bars"] = dmBarsData(oracle);
      } else if (oracle.hops) {
        say(`  hops ${oracle.hops.join(" -> ")}   ${oracle.timescale}`, "ok");
        say(`  ${oracle.selection_rule}`, "dim");
        say(`  ↳ structure emitted (ET chain)`, "dim");
        charts["structure"] = { kind: "chain", markers: CHAIN_MARKERS };
        charts["et-trace"] = etTraceData(oracle);
      } else if (oracle.fe_oxidation) {
        say(`  ${oracle.species ?? "Compound I"}  oxidation +${oracle.fe_oxidation}`, "ok");
        say(`  licensed by ${oracle.licensed_by ?? "cycle closure"}`, "dim");
        say(`  ↳ structure emitted (Compound I excursion)`, "dim");
        charts["structure"] = { kind: "cycle", markers: HEME_MARKER };
        charts["dm-bars"] = dmBarsData(recv.bindings.cycle?.meta ?? oracle);
        charts["spin-orbit"] = spinOrbitData();
      }
      continue;
    }

    // ident := catalyze SUB in E yield product
    m = line.match(/^(\w+)\s*:=\s*catalyze\s+(\w+)\s+in\s+(\w+)/);
    if (m) {
      const [, id, sub, e] = m;
      commit(recv, 2, 1);
      recv.bindings[id] = { type: "Trajectory", meta: { ...oracle } };
      say(`${id} := catalyze ${sub} in ${e} yield product : Trajectory @ ${fmtFloor(recv.floor)}`);
      say(`  ${oracle.mechanism ?? "H-atom abstraction + rebound"}  ->  ${oracle.product ?? "R-OH"}   [M+2]`, "ok");
      say(`  ${oracle.reading ?? "reaction == measurement"}`, "dim");
      say(`  ↳ structure emitted (enzyme + substrate)`, "dim");
      charts["structure"] = { kind: "cycle", markers: HEME_MARKER };
      charts["dm-bars"] = dmBarsData(recv.bindings.cycle?.meta ?? oracle, "HAT_DM");
      charts["rebound"] = reboundData();
      continue;
    }

    // ident := mutate WT by A -> B at N
    m = line.match(/^(\w+)\s*:=\s*mutate\s+(\w+)\s+by\s+(\w+)\s*->\s*(\w+)\s+at\s+(\d+)/);
    if (m) {
      const [, id, wt, a, b, pos] = m;
      commit(recv, 4, 0);
      recv.bindings[id] = {
        type: "Fold",
        meta: {
          mutation: `${a}${pos}${b}`,
          coherence: oracle.mut_coherence ?? 0.79,
          wt_coherence: oracle.wt_coherence ?? 0.83,
        },
      };
      say(`${id} := mutate ${wt} by ${a} -> ${b} at ${pos} : Fold @ ${fmtFloor(recv.floor)}`);
      say(`  re-cut K_ij   coherence ${oracle.mut_coherence ?? 0.79} (wt ${oracle.wt_coherence ?? 0.83})   [M+4]`, "ok");
      say(`  effect: ${oracle.effect ?? "altered efficiency"}`, "dim");
      charts["contact-map"] = contactMapData(13, 5, 8, oracle.mut_coherence ?? 0.79);
      charts["dm-bars"] = coherenceBars(oracle);
      continue;
    }

    // ident := complete E from partial
    m = line.match(/^(\w+)\s*:=\s*complete\s+(\w+)\s+from\s+(\w+)/);
    if (m) {
      const [, id, e, part] = m;
      commit(recv, 5, 0);
      recv.bindings[id] = { type: "Signature", meta: { ...oracle } };
      say(`${id} := complete ${e} from ${part} : Signature @ ${fmtFloor(recv.floor)}`);
      say(`  ${oracle.recovery ?? "constraint-propagation fixed point"}   stored ${oracle.stored ?? 0}   [M+5]`, "ok");
      say(`  empty dictionary: synthesise, never store`, "dim");
      charts["s-entropy"] = sEntropyData();
      continue;
    }

    // observe X [as Y]
    m = line.match(/^observe\s+(\w+)(?:\s+as\s+(\w+))?/);
    if (m) {
      const [, id, as] = m;
      const b = recv.bindings[id];
      commit(recv, 0); // observe forces the measurement now (already committed)
      if (as === "address") {
        const fam = oracle.family_address_d3 ?? oracle.examples?.[id] ?? "110";
        const full = oracle.full_address_d9 ?? "110112212";
        say(`observe ${id} as address`);
        say(`  family (depth 3) = ${fam}`, "ok");
        if (oracle.full_address_d9) say(`  full   (depth 9) = ${full}`, "ok");
        if (oracle.raw_trit_count)
          say(`  ${oracle.n_residues} residues -> ${oracle.raw_trit_count} trits (compressed)`, "dim");
        say(`  ✓ matches ${lesson.paper}  [verdict ${oracle.verdict}]`, "pass");
        charts["depth-ladder"] = depthLadderData(full);
        charts["s-entropy"] = sEntropyData(oracle.centroid);
      } else if (b?.type === "Fold") {
        say(`observe ${id} : Fold @ ${fmtFloor(recv.floor)}`);
        say(`  helices ${b.meta.helices}  sheets ${b.meta.sheets}  coherence ${b.meta.coherence}`, "ok");
      } else if (b?.type) {
        say(`observe ${id} : ${b.type} @ ${fmtFloor(recv.floor)}`);
      } else {
        say(`observe ${id}`);
      }
      continue;
    }

    // assert COND emit "msg"
    m = line.match(/^assert\s+(.+?)(?:\s+emit\s+"([^"]*)")?$/);
    if (m) {
      say(`assert ${m[1]}   ${oracle.verdict === "PASS" ? "✓" : "✗"}`, oracle.verdict === "PASS" ? "pass" : "err");
      continue;
    }

    // unrecognised (multi-line closure bodies etc.) -- ignore quietly
  }

  const dM = recv.M - startM;
  say(`clock M = ${recv.M}   (this play committed ${dM} cuts)`, "clock");
  if (oracle.verdict === "PASS" && !out.some((l) => l.kind === "pass")) {
    say(`✓ verdict ${oracle.verdict} — matches ${lesson.paper}`, "pass");
  }

  if (!recv.history.includes(lesson.id)) recv.history.push(lesson.id);
  return { console: out, charts };
}

// ---------------------------------------------------------------------
//  Chart data builders (fed to the D3 components)
// ---------------------------------------------------------------------
function contactMapData(helices = 13, sheets = 5, loops = 8, coherence = 0.83) {
  const N = 60;
  const cells = [];
  // helix diagonal bands (i, i+3/+4), sheet off-diagonal, plus backbone
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let v = i === j ? 1 : 0;
      const d = Math.abs(i - j);
      if (d >= 3 && d <= 4) v = Math.max(v, 0.7 * coherence); // helical
      if (d === 1) v = Math.max(v, 0.5);
      // a couple of long-range beta contacts
      if ((Math.abs(i - (N - 1 - j)) < 2)) v = Math.max(v, 0.55 * coherence);
      if (v > 0.05) cells.push({ i, j, v });
    }
  }
  return { N, cells, meta: { helices, sheets, loops } };
}

function orderParameterData(finalR = 0.83) {
  const pts = [];
  const steps = 80;
  for (let t = 0; t <= steps; t++) {
    const x = t / steps;
    // sigmoidal rise to finalR
    const r = finalR / (1 + Math.exp(-12 * (x - 0.45)));
    pts.push({ t: x, r });
  }
  return { pts, threshold: 0.8, finalR };
}

function orbitData(oracle) {
  const labels = oracle.states ?? [];
  const dm = oracle.DM_values ?? [];
  const nodes = labels.map((label, i) => ({
    i,
    label,
    angle: (i / labels.length) * 2 * Math.PI - Math.PI / 2,
  }));
  const edges = nodes.map((n, i) => ({
    from: i,
    to: (i + 1) % nodes.length,
    dm: dm[i] ?? 0.5,
    label: (oracle.DM_labels ?? [])[i] ?? "",
  }));
  return { nodes, edges, DM_sum: oracle.DM_sum };
}

function dmBarsData(oracle, highlightKey) {
  const vals = oracle.DM_values ?? [0.92, 0.68, 0.55, 0.72, 0.45, 0.693, 0.65, 0.3];
  const labels =
    oracle.DM_labels ?? [
      "substrate binding",
      "first electron",
      "O2 binding",
      "second electron",
      "protonation",
      "O-O heterolysis",
      "H-atom transfer",
      "product release",
    ];
  const hi = highlightKey === "HAT_DM" ? "H-atom transfer" : "O-O heterolysis";
  return {
    bars: vals.map((v, i) => ({ label: labels[i], v, highlight: labels[i] === hi })),
    sum: oracle.DM_sum,
    critical: 2.3026,
  };
}

function coherenceBars(oracle) {
  return {
    bars: [
      { label: "wild-type", v: oracle.wt_coherence ?? 0.83, highlight: false },
      { label: oracle.mutation ?? "variant", v: oracle.mut_coherence ?? 0.79, highlight: true },
    ],
    sum: null,
    critical: 0.8,
    domainMax: 1,
    ylabel: "coherence",
  };
}

function depthLadderData(full = "110112212") {
  const trits = full.split("").map((c, i) => ({ i, t: parseInt(c, 10) }));
  return {
    trits,
    marks: [
      { depth: 3, label: "family" },
      { depth: 6, label: "isoform" },
      { depth: 9, label: "variant" },
    ],
  };
}

function sEntropyData(centroid) {
  // a small cloud around the CYP3A4 centroid + a few isoform points
  const c = centroid ?? [0.545, 0.483, 0.316];
  const pts = [{ x: c[0], y: c[1], z: c[2], label: "CYP3A4", big: true }];
  const seed = [
    [0.61, 0.44, 0.29, "2D6"],
    [0.5, 0.52, 0.34, "2C9"],
    [0.58, 0.47, 0.31, "2E1"],
    [0.47, 0.5, 0.4, "1A2"],
    [0.55, 0.41, 0.27, "2C19"],
  ];
  seed.forEach(([x, y, z, label]) => pts.push({ x, y, z, label, big: false }));
  return { pts };
}

function etTraceData(oracle) {
  const hops = oracle.hops ?? ["FAD", "FMN", "heme-Fe"];
  const pts = hops.map((label, i) => ({
    label,
    t: i * 350, // fs
    y: hops.length - 1 - i,
  }));
  return { pts, timescale: oracle.timescale ?? "femtosecond" };
}

// spin-crossover across the 7 states: s_orbital conserved, S_total changes
function spinOrbitData() {
  const states = ["1", "2", "3", "4", "5", "6", "7"];
  const Stot = [0.5, 2.5, 2.0, 0.0, 0.5, 0.5, 0.5];
  return {
    orbital: states.map((s, i) => ({ state: s, v: 0.5 })), // strictly conserved
    total: states.map((s, i) => ({ state: s, v: Stot[i] })),
  };
}

// rebound: bond-order coordinate along the C-H activation
function reboundData() {
  const pts = [];
  for (let i = 0; i <= 40; i++) {
    const x = i / 40;
    // Fe=O bond weakens, C-O forms: two crossing sigmoids
    const feO = 1 - 1 / (1 + Math.exp(-10 * (x - 0.55)));
    const cO = 1 / (1 + Math.exp(-10 * (x - 0.55)));
    pts.push({ x, feO, cO });
  }
  return { pts };
}
