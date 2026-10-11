// The 3D force graph. Loaded lazily (three.js is most of the bundle) and only
// when the browser has WebGL.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import SpriteText from 'three-spritetext';
import { forceX, forceY, forceZ } from 'd3-force-3d';
import { edgeKindById, layerById, layers, nodeById, nodes, edges, maxLayerOrder } from '../lib/spec.js';
import { edgeColour, escapeHtml, layerColour, nodeRadius, nodeTooltipHtml } from '../lib/encoding.js';

const BACKGROUND = '#0b0d10';
const LAYER_GAP = 30; // world units between adjacent layer planes
const LABEL_INK = '#e6e4dd';

/** Vertical position of a layer: compute at the bottom, papers at the top. */
const layerY = (layerId) => ((layerById.get(layerId)?.order ?? 0) - maxLayerOrder / 2) * LAYER_GAP;

// Persistent simulation objects. d3 writes x/y/z/vx… onto these and replaces
// link.source/target with node references, so they are created once and the
// visible subset is filtered from them: a node keeps its place when the phase
// slider hides and re-shows it, and when the user comes back from a page.
const simNodes = nodes.map((n) => ({ ...n, __r: nodeRadius(n.id) }));
const simLinks = edges.map((e, i) => ({
  ...e,
  __key: i,
  __sourceId: e.source,
  __targetId: e.target,
}));

function buildNodeObject(node, showLabels) {
  const colour = new THREE.Color(layerColour(node.layer));
  const r = node.__r;
  const group = new THREE.Group();

  if (node.status === 'exists') {
    group.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(r, 24, 16),
        new THREE.MeshLambertMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.35 }),
      ),
    );
  } else if (node.status === 'adapt') {
    group.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(r * 0.85, 24, 16),
        new THREE.MeshLambertMaterial({
          color: colour,
          emissive: colour,
          emissiveIntensity: 0.08,
          transparent: true,
          opacity: 0.45,
          depthWrite: false,
        }),
      ),
    );
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(r * 1.25, Math.max(0.18, r * 0.07), 8, 48),
      new THREE.MeshLambertMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.25 }),
    );
    ring.rotation.x = Math.PI / 2;
    group.add(ring);
  } else if (node.status === 'new') {
    group.add(
      new THREE.Mesh(
        new THREE.IcosahedronGeometry(r, 1),
        new THREE.MeshBasicMaterial({ color: colour, wireframe: true }),
      ),
    );
  } else {
    // external (and any status the JSON adds later)
    group.add(
      new THREE.Mesh(
        new THREE.OctahedronGeometry(r * 1.2),
        new THREE.MeshLambertMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.2, flatShading: true }),
      ),
    );
  }

  if (showLabels) {
    const label = new SpriteText(node.label, 2.6, LABEL_INK);
    label.fontFace = 'system-ui, -apple-system, "Segoe UI", sans-serif';
    label.position.set(0, r + 3, 0);
    label.material.depthWrite = false;
    group.add(label);
  }
  return group;
}

function linkTooltipHtml(link) {
  const s = nodeById.get(link.__sourceId);
  const t = nodeById.get(link.__targetId);
  const kind = edgeKindById.get(link.kind);
  return `<div class="gtip gtip-link">
    <div class="gtip-meta"><span class="gtip-swatch gtip-line" style="background:${escapeHtml(edgeColour(link.kind))}"></span>${escapeHtml(kind?.label ?? link.kind)}</div>
    <div>${escapeHtml(s?.label)} → ${escapeHtml(t?.label)}</div>
    ${link.label ? `<div class="gtip-id">${escapeHtml(link.label)}</div>` : ''}
  </div>`;
}

export default function ComponentGraph3D({
  visibleNodeIds,
  hiddenKinds,
  hiddenLayers,
  showLabels,
  width,
  height,
  onOpenNode,
}) {
  const fgRef = useRef(null);
  const railRef = useRef(new Map());
  const fittedRef = useRef(false);
  const [hoverId, setHoverId] = useState(null);

  const graphData = useMemo(
    () => ({
      nodes: simNodes.filter((n) => visibleNodeIds.has(n.id)),
      links: simLinks.filter(
        (l) => visibleNodeIds.has(l.__sourceId) && visibleNodeIds.has(l.__targetId) && !hiddenKinds.has(l.kind),
      ),
    }),
    [visibleNodeIds, hiddenKinds],
  );

  // Forces: pull every node to its layer's height so the stack reads bottom
  // (compute, sources) to top (papers); keep x/z loosely centred.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force('center', null);
    fg.d3Force('layerY', forceY((n) => layerY(n.layer)).strength(0.9));
    fg.d3Force('x', forceX(0).strength(0.012));
    fg.d3Force('z', forceZ(0).strength(0.02));
    fg.d3Force('charge')?.strength(-110);
    fg.d3Force('link')?.distance(26).strength(0.25);
    fg.d3ReheatSimulation();

    // Layer rail: dim labels at each layer's height, left of the graph.
    const scene = fg.scene();
    const rail = railRef.current;
    for (const layer of layers) {
      const text = new SpriteText(layer.name.toUpperCase(), 6, layer.color);
      text.fontFace = 'system-ui, -apple-system, "Segoe UI", sans-serif';
      text.fontWeight = '600';
      text.material.opacity = 0.95;
      text.material.depthWrite = false;
      text.position.set(-120, layerY(layer.id), 0);
      scene.add(text);
      rail.set(layer.id, text);
    }
    return () => {
      for (const text of rail.values()) {
        scene.remove(text);
        text.material.map?.dispose();
        text.material.dispose();
      }
      rail.clear();
    };
  }, []);

  useEffect(() => {
    for (const [layerId, text] of railRef.current) text.visible = !hiddenLayers.has(layerId);
  }, [hiddenLayers]);

  // Fit the camera even if the simulation never reports that it stopped
  // (slow machines, background tabs): once shortly after layout starts,
  // again when the visible set changes. onEngineStop refits when it does fire.
  useEffect(() => {
    const timer = setTimeout(() => fgRef.current?.zoomToFit(600, 10), 2500);
    return () => clearTimeout(timer);
  }, [graphData]);

  const nodeThreeObject = useCallback((node) => buildNodeObject(node, showLabels), [showLabels]);

  const isAdjacent = useCallback(
    (link) => hoverId !== null && (link.__sourceId === hoverId || link.__targetId === hoverId),
    [hoverId],
  );

  const handleHover = useCallback((node) => {
    setHoverId(node ? node.id : null);
    const el = fgRef.current?.renderer()?.domElement;
    if (el) el.style.cursor = node ? 'pointer' : 'grab';
  }, []);

  const handleEngineStop = useCallback(() => {
    // Keep the layer rail just left of whatever is currently laid out.
    const xs = graphData.nodes.map((n) => n.x).filter(Number.isFinite);
    if (xs.length) {
      const railX = Math.min(...xs) - 40;
      for (const text of railRef.current.values()) text.position.x = railX;
    }
    if (fittedRef.current) return;
    fittedRef.current = true;
    fgRef.current?.zoomToFit(700, 10);
  }, [graphData]);

  return (
    <ForceGraph3D
      ref={fgRef}
      graphData={graphData}
      width={width}
      height={height}
      backgroundColor={BACKGROUND}
      showNavInfo={false}
      controlType="orbit"
      nodeId="id"
      nodeRelSize={1}
      nodeVal={(n) => n.__r ** 3}
      nodeThreeObject={nodeThreeObject}
      nodeLabel={nodeTooltipHtml}
      onNodeHover={handleHover}
      onNodeClick={(n) => onOpenNode(n.id)}
      enableNodeDrag={false}
      linkSource="source"
      linkTarget="target"
      linkColor={(l) => edgeColour(l.kind)}
      linkOpacity={0.5}
      linkWidth={(l) => (isAdjacent(l) ? 0.9 : 0)}
      linkLabel={linkTooltipHtml}
      linkDirectionalArrowLength={3.2}
      linkDirectionalArrowRelPos={1}
      linkDirectionalArrowColor={(l) => edgeColour(l.kind)}
      linkDirectionalParticles={(l) => (isAdjacent(l) ? 3 : 0)}
      linkDirectionalParticleWidth={1.4}
      linkDirectionalParticleSpeed={0.008}
      linkDirectionalParticleColor={(l) => edgeColour(l.kind)}
      cooldownTicks={220}
      onEngineStop={handleEngineStop}
    />
  );
}

