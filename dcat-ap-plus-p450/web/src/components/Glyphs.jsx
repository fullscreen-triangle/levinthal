// 2D stand-ins for the 3D encodings, used in legends, tables and pages so a
// shape means the same thing everywhere on the site.
import { statusById } from '../lib/spec.js';

const GLYPH_INK = '#d8d6ce';

export function StatusGlyph({ status, size = 14, colour = GLYPH_INK }) {
  const c = size / 2;
  const r = size * 0.36;
  let shape;
  switch (status) {
    case 'exists':
      shape = <circle cx={c} cy={c} r={r} fill={colour} />;
      break;
    case 'adapt':
      shape = (
        <>
          <circle cx={c} cy={c} r={r * 0.78} fill={colour} fillOpacity="0.45" />
          <ellipse cx={c} cy={c} rx={r * 1.25} ry={r * 0.42} fill="none" stroke={colour} strokeWidth="1.3" />
        </>
      );
      break;
    case 'new':
      shape = (
        <>
          <circle cx={c} cy={c} r={r} fill="none" stroke={colour} strokeWidth="1.2" />
          <ellipse cx={c} cy={c} rx={r} ry={r * 0.4} fill="none" stroke={colour} strokeWidth="0.8" />
          <line x1={c} y1={c - r} x2={c} y2={c + r} stroke={colour} strokeWidth="0.8" />
        </>
      );
      break;
    case 'external': {
      const d = r * 1.2;
      shape = (
        <polygon
          points={`${c},${c - d} ${c + d},${c} ${c},${c + d} ${c - d},${c}`}
          fill={colour}
          stroke={colour}
          strokeWidth="0.5"
        />
      );
      break;
    }
    default:
      shape = <rect x={c - r} y={c - r} width={2 * r} height={2 * r} fill="none" stroke={colour} />;
  }
  return (
    <svg className="glyph" width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      {shape}
    </svg>
  );
}

export function StatusTag({ status }) {
  return (
    <span className={`status-tag status-${status}`}>
      <StatusGlyph status={status} size={12} />
      {statusById.get(status)?.label ?? status}
    </span>
  );
}

export function Swatch({ colour, shape = 'square', title }) {
  return (
    <span
      className={`swatch swatch-${shape}`}
      style={{ background: colour }}
      title={title}
      aria-hidden={title ? undefined : 'true'}
    />
  );
}
