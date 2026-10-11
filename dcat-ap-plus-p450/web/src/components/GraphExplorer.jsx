import { Component, Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  componentPath,
  edgeKinds,
  edges,
  layers,
  maxDegree,
  maxPhase,
  minPhase,
  nodes,
  phaseById,
  phases,
  statuses,
} from '../lib/spec.js';
import { STATUS_ENCODING, edgeColour } from '../lib/encoding.js';
import { hasWebGL } from '../lib/webgl.js';
import { StatusGlyph, Swatch } from './Glyphs.jsx';

const ComponentGraph3D = lazy(() => import('./ComponentGraph3D.jsx'));

const NARROW = '(max-width: 720px)';
const PLAY_INTERVAL_MS = 1800;
// Legend dot diameter in px for a given degree: same √degree law as the 3D radius.
const legendDotPx = (d) => 2 * 2.2 * Math.sqrt(d);

class GraphErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) return this.props.fallback(this.state.error);
    return this.props.children;
  }
}

function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

const toggleIn = (set, value) => {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
};

export default function GraphExplorer() {
  const navigate = useNavigate();
  const webgl = useMemo(() => hasWebGL(), []);
  const narrow = useMemo(() => window.matchMedia?.(NARROW).matches ?? false, []);
  const [show3D, setShow3D] = useState(webgl && !narrow);

  const [phase, setPhase] = useState(maxPhase);
  const [playing, setPlaying] = useState(false);
  const [hiddenLayers, setHiddenLayers] = useState(() => new Set());
  const [hiddenKinds, setHiddenKinds] = useState(() => new Set());
  const [showLabels, setShowLabels] = useState(false);

  const stageRef = useRef(null);
  const { width, height } = useElementSize(stageRef);

  // Construction playback: step the phase slider from the first phase to the last.
  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => {
      setPhase((p) => {
        if (p >= maxPhase) {
          setPlaying(false);
          return p;
        }
        return p + 1;
      });
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [playing]);

  const play = () => {
    setPhase(minPhase);
    setPlaying(true);
  };

  const visibleNodeIds = useMemo(
    () => new Set(nodes.filter((n) => n.phase <= phase && !hiddenLayers.has(n.layer)).map((n) => n.id)),
    [phase, hiddenLayers],
  );
  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target) && !hiddenKinds.has(e.kind),
      ),
    [visibleNodeIds, hiddenKinds],
  );

  const phaseNodes = useMemo(() => nodes.filter((n) => n.phase <= phase), [phase]);
  const addedThisPhase = nodes.filter((n) => n.phase === phase).length;
  const current = phaseById.get(phase);

  const countIn = (list, key, value) => list.filter((x) => x[key] === value).length;
  const edgesUpToPhase = useMemo(() => {
    const ids = new Set(phaseNodes.map((n) => n.id));
    return edges.filter((e) => ids.has(e.source) && ids.has(e.target));
  }, [phaseNodes]);

  const filtersActive = hiddenLayers.size > 0 || hiddenKinds.size > 0;
  const openNode = (id) => navigate(componentPath(id));

  const fallbackNote = (reason) => (
    <div className="graph-fallback" role="note">
      <p>
        <strong>The 3D graph is not shown.</strong> {reason} Every component and connection is in the{' '}
        <a href="#components">component list</a> below, and each component page lists its neighbours.
      </p>
    </div>
  );

  let stage;
  if (!webgl) {
    stage = fallbackNote('This browser could not create a WebGL context, which the 3D view needs.');
  } else if (!show3D) {
    stage = (
      <div className="graph-fallback" role="note">
        <p>
          On a narrow screen the 3D view is off by default: dragging it rotates the graph instead of scrolling the
          page. The <a href="#components">component list</a> below holds the same information.
        </p>
        <button type="button" className="btn" onClick={() => setShow3D(true)}>
          Show the 3D graph
        </button>
      </div>
    );
  } else {
    stage = (
      <GraphErrorBoundary fallback={(err) => fallbackNote(`The 3D view failed to start (${err.message}).`)}>
        <Suspense fallback={<div className="graph-loading">Loading the 3D view…</div>}>
          {width > 0 && height > 0 && (
            <ComponentGraph3D
              visibleNodeIds={visibleNodeIds}
              hiddenKinds={hiddenKinds}
              hiddenLayers={hiddenLayers}
              showLabels={showLabels}
              width={width}
              height={height}
              onOpenNode={openNode}
            />
          )}
        </Suspense>
      </GraphErrorBoundary>
    );
  }

  const degreeSamples = [1, Math.round(maxDegree / 3), maxDegree].filter((v, i, a) => v > 0 && a.indexOf(v) === i);

  return (
    <section className="explorer" aria-labelledby="graph-heading">
      <div className="section-head">
        <h2 id="graph-heading">How the components connect</h2>
        <p className="lede">
          Each sphere is a component, stacked by layer from university compute at the bottom to papers at the top.
          Hover for a summary; click to open its page. Drag to rotate, scroll to zoom.
        </p>
      </div>

      <div className="phase-bar">
        <div className="phase-control">
          <label htmlFor="phase-slider" className="control-label">
            Construction phase
          </label>
          <div className="phase-row">
            <input
              id="phase-slider"
              type="range"
              min={minPhase}
              max={maxPhase}
              step="1"
              value={phase}
              list="phase-ticks"
              onChange={(e) => {
                setPlaying(false);
                setPhase(Number(e.target.value));
              }}
              aria-valuetext={`Phase ${phase}: ${current?.name}`}
            />
            <datalist id="phase-ticks">
              {phases.map((p) => (
                <option key={p.id} value={p.id} label={String(p.id)} />
              ))}
            </datalist>
            <button type="button" className="btn btn-small" onClick={playing ? () => setPlaying(false) : play}>
              {playing ? 'Pause' : 'Play construction'}
            </button>
          </div>
          <ol className="phase-ticks" aria-hidden="true">
            {phases.map((p) => (
              <li key={p.id} className={p.id <= phase ? 'on' : ''}>
                {p.id}
              </li>
            ))}
          </ol>
        </div>
        <div className="phase-readout" aria-live="polite">
          <div className="phase-name">
            Phase {phase} · {current?.name}
            {phase < maxPhase ? <span className="muted"> (and every earlier phase)</span> : null}
          </div>
          <p className="phase-desc">{current?.description}</p>
          <p className="phase-counts mono">
            {phaseNodes.length}/{nodes.length} components · {edgesUpToPhase.length}/{edges.length} connections built ·{' '}
            +{addedThisPhase} added in phase {phase}
            {filtersActive ? (
              <>
                {' '}
                · {visibleNodeIds.size} components, {visibleEdges.length} connections after filters
              </>
            ) : null}
          </p>
        </div>
      </div>

      <div className="explorer-body">
        <div className={show3D ? 'graph-stage' : 'graph-stage graph-stage-idle'} ref={stageRef}>
          {stage}
        </div>

        <aside className="explorer-controls" aria-label="Graph filters and legend">
          <fieldset className="control-group">
            <legend>Layers · colour</legend>
            <ul className="toggle-list">
              {[...layers].reverse().map((l) => {
                const total = nodes.filter((n) => n.layer === l.id).length;
                const shown = countIn(phaseNodes, 'layer', l.id);
                return (
                  <li key={l.id}>
                    <label className={hiddenLayers.has(l.id) ? 'toggle off' : 'toggle'}>
                      <input
                        type="checkbox"
                        checked={!hiddenLayers.has(l.id)}
                        onChange={() => setHiddenLayers((s) => toggleIn(s, l.id))}
                      />
                      <Swatch colour={l.color} shape="dot" />
                      <span className="toggle-name">{l.name}</span>
                      <span className="toggle-count mono">
                        {shown}/{total}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
            <div className="control-actions">
              <button type="button" className="link-btn" onClick={() => setHiddenLayers(new Set())}>
                all
              </button>
              <button
                type="button"
                className="link-btn"
                onClick={() => setHiddenLayers(new Set(layers.map((l) => l.id)))}
              >
                none
              </button>
            </div>
          </fieldset>

          <fieldset className="control-group">
            <legend>Connections · colour by kind</legend>
            <ul className="toggle-list">
              {edgeKinds.map((k) => {
                const total = countIn(edges, 'kind', k.id);
                const shown = countIn(edgesUpToPhase, 'kind', k.id);
                return (
                  <li key={k.id}>
                    <label className={hiddenKinds.has(k.id) ? 'toggle off' : 'toggle'} title={k.description}>
                      <input
                        type="checkbox"
                        checked={!hiddenKinds.has(k.id)}
                        onChange={() => setHiddenKinds((s) => toggleIn(s, k.id))}
                      />
                      <Swatch colour={edgeColour(k.id)} shape="line" />
                      <span className="toggle-name">{k.label}</span>
                      <span className="toggle-count mono">
                        {shown}/{total}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
            <div className="control-actions">
              <button type="button" className="link-btn" onClick={() => setHiddenKinds(new Set())}>
                all
              </button>
              <button
                type="button"
                className="link-btn"
                onClick={() => setHiddenKinds(new Set(edgeKinds.map((k) => k.id)))}
              >
                none
              </button>
            </div>
          </fieldset>

          <fieldset className="control-group">
            <legend>Status · shape</legend>
            <ul className="legend-list">
              {statuses.map((s) => (
                <li key={s.id} title={s.description}>
                  <StatusGlyph status={s.id} size={16} />
                  <span>
                    <strong>{s.label}</strong> — {STATUS_ENCODING[s.id]?.shape ?? 'cube outline'}
                    <span className="muted"> · {countIn(phaseNodes, 'status', s.id)}</span>
                  </span>
                </li>
              ))}
            </ul>
            <p className="legend-note">
              Size: area ∝ degree (
              {degreeSamples.map((d, i) => (
                <span key={d} className="size-sample">
                  <span
                    className="size-dot"
                    style={{ width: legendDotPx(d), height: legendDotPx(d) }}
                    aria-hidden="true"
                  />
                  {d}
                  {i < degreeSamples.length - 1 ? ', ' : ''}
                </span>
              ))}
              ). Arrows and moving particles point from source to target.
            </p>
            <label className="toggle toggle-plain">
              <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
              <span className="toggle-name">Show labels in the graph</span>
            </label>
          </fieldset>
        </aside>
      </div>

      <p className="sr-only">
        The 3D graph is a visual summary only. The same {nodes.length} components and their connections are listed in
        the table below; each component page lists its incoming and outgoing connections.
      </p>
    </section>
  );
}
