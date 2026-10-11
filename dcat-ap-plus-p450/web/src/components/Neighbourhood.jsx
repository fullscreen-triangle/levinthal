// A flat ego view of one component: incoming connections on the left,
// outgoing on the right, one row per edge, coloured like the 3D graph.
import { useNavigate } from 'react-router-dom';
import { componentPath, edgeKindById, layerById, nodeById } from '../lib/spec.js';
import { edgeColour } from '../lib/encoding.js';

const W = 760;
const ROW = 26;
const PAD = 22;
const X_IN = 230;
const X_MID = 380;
const X_OUT = 530;

export default function Neighbourhood({ node, incoming, outgoing }) {
  const navigate = useNavigate();
  const rows = Math.max(incoming.length, outgoing.length, 1);
  const H = PAD * 2 + rows * ROW;
  const yMid = H / 2;
  const yRow = (i, n) => yMid + (i - (n - 1) / 2) * ROW;
  const kinds = [...new Set([...incoming, ...outgoing].map((e) => e.kind))];

  const go = (id) => (evt) => {
    evt.preventDefault();
    navigate(componentPath(id));
  };

  const neighbour = (e, i, n, side) => {
    const otherId = side === 'in' ? e.source : e.target;
    const other = nodeById.get(otherId);
    const x = side === 'in' ? X_IN : X_OUT;
    const y = yRow(i, n);
    const colour = edgeColour(e.kind);
    const [x1, y1, x2, y2] = side === 'in' ? [x + 6, y, X_MID - 11, yMid] : [X_MID + 8, yMid, x - 9, y];
    const cx = (x1 + x2) / 2;
    const kind = edgeKindById.get(e.kind)?.label ?? e.kind;
    return (
      <g key={`${side}-${i}`}>
        <path
          d={`M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}`}
          fill="none"
          stroke={colour}
          strokeWidth="1.4"
          strokeOpacity="0.85"
          markerEnd={`url(#arrow-${e.kind})`}
        >
          <title>{`${nodeById.get(e.source)?.label} ${kind} ${nodeById.get(e.target)?.label}${e.label ? ` (${e.label})` : ''}`}</title>
        </path>
        <a href={componentPath(otherId)} onClick={go(otherId)} className="ego-link">
          <circle cx={x} cy={y} r="5.5" fill={layerById.get(other?.layer)?.color ?? '#888'} />
          <text
            x={side === 'in' ? x - 11 : x + 11}
            y={y}
            dy="0.35em"
            textAnchor={side === 'in' ? 'end' : 'start'}
            className="ego-label"
          >
            {other?.label ?? otherId}
          </text>
        </a>
      </g>
    );
  };

  return (
    <figure className="ego">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-labelledby="ego-title">
        <title id="ego-title">
          {`${node.label}: ${incoming.length} incoming and ${outgoing.length} outgoing connections`}
        </title>
        <defs>
          {kinds.map((k) => (
            <marker
              key={k}
              id={`arrow-${k}`}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0,1 L9,5 L0,9 z" fill={edgeColour(k)} />
            </marker>
          ))}
        </defs>
        <text x={X_IN} y={14} textAnchor="middle" className="ego-heading">
          incoming
        </text>
        <text x={X_OUT} y={14} textAnchor="middle" className="ego-heading">
          outgoing
        </text>
        {incoming.map((e, i) => neighbour(e, i, incoming.length, 'in'))}
        {outgoing.map((e, i) => neighbour(e, i, outgoing.length, 'out'))}
        <circle cx={X_MID} cy={yMid} r="9" fill={layerById.get(node.layer)?.color} stroke="#0b0d10" strokeWidth="2" />
      </svg>
      <figcaption className="muted">
        Neighbourhood of {node.label}. Edge colour is the connection kind, dot colour the neighbour’s layer.
      </figcaption>
    </figure>
  );
}
