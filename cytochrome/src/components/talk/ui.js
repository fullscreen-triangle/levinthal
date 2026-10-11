import { motion } from "framer-motion";
import { C, LEDGER } from "./theme";

// One slide: kicker, title, lead, and a content area that fills the rest.
export function SlideFrame({ kicker, title, lead, papers, children }) {
  return (
    <div className="absolute inset-0 flex flex-col px-[72px] pt-[56px] pb-[88px]" style={{ color: C.ink }}>
      <div className="flex items-baseline gap-4 mb-2">
        <span className="uppercase tracking-[0.18em] text-[15px]" style={{ color: C.muted }}>{kicker}</span>
        {papers && papers.length > 0 && (
          <span className="flex gap-2 flex-wrap">
            {papers.map((p) => (
              <span key={p} className="text-[13px] px-2 py-[2px] rounded border" style={{ borderColor: C.line, color: C.ink2 }}>
                {p}
              </span>
            ))}
          </span>
        )}
      </div>
      <h1 className="text-[46px] leading-[1.08] font-semibold tracking-tight max-w-[1380px]">{title}</h1>
      {lead && (
        <p className="mt-4 text-[22px] leading-[1.45] max-w-[1300px]" style={{ color: C.ink2 }}>
          {lead}
        </p>
      )}
      <div className="relative flex-1 mt-6 min-h-0">{children}</div>
    </div>
  );
}

// A build: shown once `step` reaches `at`.
export function Reveal({ step, at = 0, children, className = "", y = 14 }) {
  const on = step >= at;
  return (
    <motion.div
      className={className}
      initial={false}
      animate={{ opacity: on ? 1 : 0, y: on ? 0 : y }}
      transition={{ duration: 0.45, ease: "easeOut" }}
      style={{ pointerEvents: on ? "auto" : "none" }}
    >
      {children}
    </motion.div>
  );
}

export function Badge({ k, children }) {
  const l = LEDGER[k];
  return (
    <span
      className="inline-flex items-center gap-1 text-[13px] px-2 py-[1px] rounded-full border align-middle whitespace-nowrap"
      style={{ borderColor: l.color, color: C.ink2 }}
    >
      <span className="w-2 h-2 rounded-full" style={{ background: l.color }} />
      {children || l.label}
    </span>
  );
}

export function Stat({ value, label, sub, accent = C.blue }) {
  return (
    <div className="rounded-lg px-5 py-4" style={{ background: C.panel, borderLeft: `3px solid ${accent}` }}>
      <div className="text-[34px] font-semibold leading-none tabular-nums">{value}</div>
      <div className="mt-2 text-[16px]" style={{ color: C.ink2 }}>{label}</div>
      {sub && <div className="mt-1 text-[13px]" style={{ color: C.muted }}>{sub}</div>}
    </div>
  );
}

export function Box({ children, className = "", style }) {
  return (
    <div className={`rounded-lg px-6 py-5 ${className}`} style={{ background: C.panel, ...style }}>
      {children}
    </div>
  );
}

export function Statement({ children, accent = C.blue }) {
  return (
    <div className="rounded-lg px-6 py-4 text-[21px] leading-[1.45]" style={{ background: C.panel, borderLeft: `3px solid ${accent}` }}>
      {children}
    </div>
  );
}

// Compared-with strip for application slides.
export function Compare({ rows }) {
  return (
    <table className="w-full text-[15px] leading-snug" style={{ color: C.ink2 }}>
      <thead>
        <tr style={{ color: C.muted }}>
          <th className="text-left font-normal pb-2">quantity</th>
          <th className="text-right font-normal pb-2">paper</th>
          <th className="text-right font-normal pb-2">compared with</th>
          <th className="text-right font-normal pb-2 pl-3">status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([q, v, ref, k]) => (
          <tr key={q} className="border-t" style={{ borderColor: C.line }}>
            <td className="py-[7px] pr-3" style={{ color: C.ink }}>{q}</td>
            <td className="py-[7px] text-right tabular-nums">{v}</td>
            <td className="py-[7px] text-right">{ref}</td>
            <td className="py-[7px] text-right pl-3 w-[118px]"><Badge k={k} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Legend({ items }) {
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1 text-[15px]" style={{ color: C.ink2 }}>
      {items.map(([label, color, dashed]) => (
        <span key={label} className="inline-flex items-center gap-2">
          <span
            className="inline-block w-5"
            style={{ borderTop: `3px ${dashed ? "dashed" : "solid"} ${color}`, height: 0 }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}
