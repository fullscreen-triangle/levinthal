// =====================================================================
//  ShakespeareIDE — a full-page, VSCode-resemblant sandbox that
//  performs the Cytochrome P450 monograph as Shakespeare plays.
//
//  Layout:  [file tree] | [editor + monograph] | [console / charts tabs]
//  State:   ONE accumulating receiver; the cut count M only moves
//           forward across lessons (measurement conserves the medium).
// =====================================================================
import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { LESSONS, RECEIVER_FLOOR } from "@/data/lessons";
import dynamic from "next/dynamic";
import { newReceiver, performPlay } from "@/helpers/shakespeare";
import ShakespeareChart from "@/components/shakespeare/ShakespeareChart";

// 3D structure output is client-only (react-three-fiber Canvas)
const StructureOutput = dynamic(() => import("@/components/shakespeare/StructureOutput"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[320px] items-center justify-center rounded-md border border-neutral-700 bg-[#0c0c0c] text-xs text-neutral-600">
      loading structure…
    </div>
  ),
});

// ---- syntax highlighting for the editor overlay ---------------------
const KW1 = new Set([
  "receiver", "floor", "cut", "contact", "close", "complex", "fold", "track",
  "until", "yield", "when", "observe", "catalyze", "mutate", "complete",
  "in", "as", "let", "with", "by", "reps", "converge", "diverge", "assert", "emit", "at",
]);
const KW2 = new Set(["residue", "cofactor", "electronic", "solvent", "substrate", "address", "amalgamation", "trajectory", "product"]);

function highlight(src) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return src
    .split("\n")
    .map((line) => {
      const c = line.indexOf("--");
      let code = line, comment = "";
      if (c >= 0) { code = line.slice(0, c); comment = line.slice(c); }
      const html = esc(code).replace(/("[^"]*")|(\b[0-9][0-9.eE+-]*\b)|([A-Za-z_]\w*)/g, (mm, str, num, word) => {
        if (str) return `<span style="color:#f472b6">${str}</span>`;
        if (num) return `<span style="color:#fbbf24">${num}</span>`;
        if (word && KW1.has(word)) return `<span style="color:#58E6D9;font-weight:600">${word}</span>`;
        if (word && KW2.has(word)) return `<span style="color:#B63E96">${word}</span>`;
        return mm;
      });
      return html + (comment ? `<span style="color:#5b7a5b;font-style:italic">${esc(comment)}</span>` : "");
    })
    .join("\n");
}

const KIND_COLOR = {
  cmd: "text-cyan-300 font-semibold",
  ok: "text-emerald-300",
  pass: "text-emerald-400 font-semibold",
  err: "text-rose-400",
  dim: "text-neutral-500",
  clock: "text-amber-300",
  log: "text-neutral-200",
};

export default function ShakespeareIDE() {
  const [activeId, setActiveId] = useState(LESSONS[0].id);
  const [srcById, setSrcById] = useState(() => Object.fromEntries(LESSONS.map((l) => [l.id, l.src])));
  const [tab, setTab] = useState("console"); // console | charts
  const [consoleOut, setConsoleOut] = useState([]);
  const [charts, setCharts] = useState({});
  const recvRef = useRef(newReceiver(RECEIVER_FLOOR));
  const [recvView, setRecvView] = useState(() => snapshot(recvRef.current));

  const lesson = useMemo(() => LESSONS.find((l) => l.id === activeId), [activeId]);
  const src = srcById[activeId];

  function run() {
    const { console: c, charts: ch } = performPlay(src, recvRef.current, lesson);
    setConsoleOut((prev) => [...prev, ...c]);
    setCharts(ch);
    setRecvView(snapshot(recvRef.current));
    if (Object.keys(ch).length) setTab("charts");
    else setTab("console");
  }
  function runAll() {
    // perform every lesson in order into the same receiver — the ladder
    recvRef.current = newReceiver(RECEIVER_FLOOR);
    const all = [];
    let lastCharts = {};
    for (const l of LESSONS) {
      const { console: c, charts: ch } = performPlay(srcById[l.id], recvRef.current, l);
      all.push(...c, { kind: "dim", text: "" });
      if (Object.keys(ch).length) lastCharts = ch;
    }
    setConsoleOut(all);
    setCharts(lastCharts);
    setRecvView(snapshot(recvRef.current));
    setTab("console");
  }
  function resetReceiver() {
    recvRef.current = newReceiver(RECEIVER_FLOOR);
    setConsoleOut([]);
    setCharts({});
    setRecvView(snapshot(recvRef.current));
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-[#1b1b1b] font-mono text-sm text-neutral-200">
      {/* title bar */}
      <div className="flex items-center justify-between border-b border-neutral-800 bg-[#111] px-4 py-1.5 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-[#58E6D9]">◆</span>
          <span className="font-semibold tracking-wide">Shakespeare</span>
          <span className="text-neutral-500">— receiver sandbox · Cytochrome P450 monograph</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={runAll} className="rounded border border-neutral-700 px-2 py-0.5 text-neutral-300 hover:bg-neutral-800">
            ⏩ Perform all
          </button>
          <button onClick={resetReceiver} className="rounded border border-neutral-700 px-2 py-0.5 text-neutral-400 hover:bg-neutral-800">
            ↺ Reset receiver
          </button>
          <Link href="/" className="text-neutral-500 hover:text-neutral-300">exit</Link>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* -------- file tree / lessons rail -------- */}
        <div className="flex w-64 flex-col border-r border-neutral-800 bg-[#181818]">
          <div className="px-3 py-2 text-[10px] uppercase tracking-widest text-neutral-500">Lessons · lessons/</div>
          <div className="flex-1 overflow-y-auto">
            {LESSONS.map((l) => {
              const done = recvView.history.includes(l.id);
              const active = l.id === activeId;
              return (
                <button
                  key={l.id}
                  onClick={() => setActiveId(l.id)}
                  className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-neutral-800 ${
                    active ? "bg-neutral-800 text-white" : "text-neutral-400"
                  }`}
                >
                  <span className={done ? "text-emerald-400" : "text-neutral-600"}>{done ? "●" : "○"}</span>
                  <span className="text-neutral-600">{String(l.n).padStart(2, "0")}</span>
                  <span className="truncate">{l.id.replace(/^\d+_/, "")}.shk</span>
                </button>
              );
            })}
          </div>
          {/* state inspector */}
          <div className="border-t border-neutral-800 px-3 py-2 text-[11px] leading-5">
            <div className="mb-1 text-[10px] uppercase tracking-widest text-neutral-500">Receiver state</div>
            <Row k="clock M" v={recvView.M} accent />
            <Row k="floor β" v={recvView.floor.toExponential(1)} />
            <Row k="graph |V|" v={recvView.graphVertices} />
            <Row k="residue" v={recvView.leaves.res} />
            <Row k="cofactor" v={recvView.leaves.cof} />
            <Row k="electronic" v={recvView.leaves.ele} />
            <Row k="solvent" v={recvView.leaves.sol} />
            <Row k="substrate" v={recvView.leaves.sub} />
          </div>
        </div>

        {/* -------- editor + monograph -------- */}
        <div className="flex min-w-0 flex-1 flex-col">
          {/* tab strip */}
          <div className="flex items-center border-b border-neutral-800 bg-[#141414] text-xs">
            <div className="border-r border-neutral-800 bg-[#1b1b1b] px-4 py-1.5 text-neutral-200">
              {activeId.replace(/^\d+_/, "")}.shk
            </div>
            <div className="px-3 text-neutral-500">{lesson.title} — <span className="italic">{lesson.subtitle}</span></div>
            <div className="ml-auto pr-3">
              <button onClick={run} className="rounded bg-[#0f766e] px-3 py-0.5 font-semibold text-white hover:bg-[#0d9488]">
                ▶ Perform
              </button>
            </div>
          </div>

          {/* editor */}
          <div className="relative min-h-0 flex-1 overflow-auto bg-[#1b1b1b]">
            <CodeEditor value={src} onChange={(v) => setSrcById((s) => ({ ...s, [activeId]: v }))} />
          </div>

          {/* monograph strip */}
          <div className="border-t border-neutral-800 bg-[#141414] px-4 py-2 text-[11px] text-neutral-400">
            <span className="text-neutral-500">performs</span>{" "}
            <span className="text-[#58E6D9]">{lesson.paper}</span>
            <span className="mx-2 text-neutral-700">·</span>
            <span className="text-neutral-500">verbs</span>{" "}
            {lesson.verbs.map((v) => (
              <span key={v} className="mr-1 rounded bg-neutral-800 px-1.5 py-0.5 text-[#B63E96]">{v}</span>
            ))}
            {lesson.requires && (
              <>
                <span className="mx-2 text-neutral-700">·</span>
                <span className="text-neutral-500">requires</span>{" "}
                <span className="text-neutral-300">{lesson.requires.replace(/^\d+_/, "")}</span>
              </>
            )}
          </div>
        </div>

        {/* -------- output column: console / charts tabs -------- */}
        <div className="flex w-[420px] flex-col border-l border-neutral-800 bg-[#151515]">
          <div className="flex border-b border-neutral-800 text-xs">
            {["console", "charts"].map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 capitalize ${tab === t ? "border-b-2 border-[#58E6D9] text-white" : "text-neutral-500 hover:text-neutral-300"}`}
              >
                {t}
                {t === "charts" && Object.keys(charts).length > 0 && (
                  <span className="ml-1 rounded-full bg-[#0f766e] px-1.5 text-[10px]">{Object.keys(charts).length}</span>
                )}
              </button>
            ))}
          </div>

          {tab === "console" ? (
            <div className="flex-1 overflow-y-auto p-3 text-[12px] leading-5">
              {consoleOut.length === 0 && (
                <div className="text-neutral-600">Press <span className="text-[#58E6D9]">▶ Perform</span> to run this play. Output accrues into the receiver — the clock M only moves forward.</div>
              )}
              {consoleOut.map((l, i) => (
                <div key={i} className={`whitespace-pre-wrap ${KIND_COLOR[l.kind] ?? "text-neutral-200"}`}>{l.text}</div>
              ))}
            </div>
          ) : (
            <div className="flex-1 space-y-3 overflow-y-auto p-3">
              {Object.keys(charts).length === 0 && (
                <div className="text-neutral-600 text-xs">This play draws no charts. Perform a lesson with a fold, an address, or a cycle.</div>
              )}
              {/* structure first — the folded protein IS the primary output */}
              {charts.structure && <StructureOutput data={charts.structure} />}
              {Object.entries(charts)
                .filter(([name]) => name !== "structure")
                .map(([name, data]) => (
                  <ShakespeareChart key={name} name={name} data={data} />
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// immutable view of the receiver so React re-renders the inspector
function snapshot(r) {
  return {
    M: r.M,
    floor: r.floor,
    graphVertices: r.graphVertices,
    leaves: { ...r.leaves },
    history: [...r.history],
  };
}

function Row({ k, v, accent }) {
  return (
    <div className="flex justify-between">
      <span className="text-neutral-500">{k}</span>
      <span className={accent ? "text-amber-300 font-semibold" : "text-neutral-300"}>{v}</span>
    </div>
  );
}

// syntax-highlighted textarea (overlay technique, dependency-free)
function CodeEditor({ value, onChange }) {
  const taRef = useRef(null);
  const preRef = useRef(null);
  const sync = () => {
    if (preRef.current && taRef.current) {
      preRef.current.scrollTop = taRef.current.scrollTop;
      preRef.current.scrollLeft = taRef.current.scrollLeft;
    }
  };
  return (
    <div className="relative min-h-full">
      <pre
        ref={preRef}
        aria-hidden
        className="pointer-events-none absolute inset-0 m-0 overflow-auto whitespace-pre p-4 text-[13px] leading-6"
        dangerouslySetInnerHTML={{ __html: highlight(value) + "\n" }}
      />
      <textarea
        ref={taRef}
        value={value}
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
        onScroll={sync}
        className="absolute inset-0 h-full w-full resize-none bg-transparent p-4 text-[13px] leading-6 text-transparent caret-white outline-none"
      />
    </div>
  );
}
