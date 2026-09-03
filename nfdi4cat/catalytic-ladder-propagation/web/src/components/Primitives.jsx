import React from 'react'
import { VERDICTS } from '../lib/kernel.js'

export function Pill({ verdict }) {
  const v = VERDICTS[verdict] || { label: verdict, tone: 'neutral' }
  return <span className={`pill ${v.tone}`} title={v.gloss}>{v.label}</span>
}

export function Stat({ value, label, color }) {
  return (
    <div className="stat">
      <div className="v" style={color ? { color } : undefined}>{value}</div>
      <div className="k">{label}</div>
    </div>
  )
}

export function Section({ id, kicker, title, sub, children }) {
  return (
    <section id={id}>
      <div className="wrap">
        {kicker && <p className="kicker">{kicker}</p>}
        <h2>{title}</h2>
        {sub && <p className="sub">{sub}</p>}
        {children}
      </div>
    </section>
  )
}

export function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <>
      <label className="ctl">
        {label} <b>{fmt ? fmt(value) : value}</b>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </>
  )
}

/** A tiny dependency-free line chart. */
export function Chart({
  series, width = 520, height = 190, xLabel, yLabel,
  xDomain, yDomain, bands = [], rules = [], padL = 46, padB = 30,
}) {
  const padT = 10
  const padR = 12
  const all = series.flatMap((s) => s.points)
  const xs = all.map((p) => p[0])
  const ys = all.map((p) => p[1])
  const x0 = xDomain ? xDomain[0] : Math.min(...xs)
  const x1 = xDomain ? xDomain[1] : Math.max(...xs)
  const y0 = yDomain ? yDomain[0] : Math.min(...ys)
  const y1 = yDomain ? yDomain[1] : Math.max(...ys)
  const sx = (x) => padL + ((x - x0) / (x1 - x0 || 1)) * (width - padL - padR)
  const sy = (y) => height - padB - ((y - y0) / (y1 - y0 || 1)) * (height - padT - padB)

  const ticks = (a, b, n = 4) =>
    Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n)

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }}>
      {bands.map((b, i) => (
        <rect
          key={i}
          x={padL}
          y={sy(b.to)}
          width={width - padL - padR}
          height={Math.abs(sy(b.from) - sy(b.to))}
          fill={b.color}
          opacity={b.opacity ?? 0.14}
        />
      ))}
      {ticks(y0, y1).map((t, i) => (
        <g key={i}>
          <line x1={padL} x2={width - padR} y1={sy(t)} y2={sy(t)}
                stroke="var(--line)" strokeWidth="1" />
          <text x={padL - 7} y={sy(t) + 3.5} textAnchor="end"
                fontSize="9.5" fill="var(--ink-faint)" fontFamily="var(--mono)">
            {Math.abs(t) >= 1000 || (Math.abs(t) < 0.01 && t !== 0)
              ? t.toExponential(0) : (+t.toFixed(2)).toString()}
          </text>
        </g>
      ))}
      {ticks(x0, x1).map((t, i) => (
        <text key={i} x={sx(t)} y={height - padB + 14} textAnchor="middle"
              fontSize="9.5" fill="var(--ink-faint)" fontFamily="var(--mono)">
          {Math.abs(t) >= 10000 ? t.toExponential(0) : (+t.toFixed(2)).toString()}
        </text>
      ))}
      {rules.map((r, i) => (
        <g key={i}>
          <line x1={padL} x2={width - padR} y1={sy(r.y)} y2={sy(r.y)}
                stroke={r.color} strokeWidth="1.4" strokeDasharray="5 4" />
          {r.label && (
            <text x={width - padR - 4} y={sy(r.y) - 5} textAnchor="end"
                  fontSize="9.5" fill={r.color} fontFamily="var(--mono)">
              {r.label}
            </text>
          )}
        </g>
      ))}
      {series.map((s, i) => (
        <polyline
          key={i}
          points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')}
          fill="none" stroke={s.color} strokeWidth={s.width ?? 1.9}
          strokeDasharray={s.dash || undefined}
          strokeLinejoin="round"
        />
      ))}
      {series.filter((s) => s.label).map((s, i) => (
        <g key={`l${i}`} transform={`translate(${padL + 8}, ${padT + 12 + i * 14})`}>
          <line x1="0" x2="16" y1="-3.5" y2="-3.5" stroke={s.color} strokeWidth="2.4"
                strokeDasharray={s.dash || undefined} />
          <text x="21" y="0" fontSize="10" fill="var(--ink-dim)" fontFamily="var(--mono)">
            {s.label}
          </text>
        </g>
      ))}
      <text x={(width + padL) / 2} y={height - 2} textAnchor="middle"
            fontSize="10" fill="var(--ink-faint)">{xLabel}</text>
      <text x={11} y={height / 2} textAnchor="middle" fontSize="10"
            fill="var(--ink-faint)" transform={`rotate(-90 11 ${height / 2})`}>
        {yLabel}
      </text>
    </svg>
  )
}

export function Bars({ items, height = 180, fmtV = (v) => v.toFixed(2), unit = '' }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9)
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', height,
                  padding: '8px 0' }}>
      {items.map((it, i) => (
        <div key={i} style={{ flex: 1, textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11,
                        color: 'var(--ink)', marginBottom: 4 }}>
            {fmtV(it.value)}{unit}
          </div>
          <div
            style={{
              height: `${(Math.abs(it.value) / max) * (height - 58)}px`,
              background: it.color, borderRadius: '4px 4px 0 0',
              transition: 'height .18s ease',
            }}
          />
          <div style={{ fontSize: 11, color: 'var(--ink-dim)', marginTop: 6,
                        lineHeight: 1.25 }}>
            {it.label}
          </div>
        </div>
      ))}
    </div>
  )
}
