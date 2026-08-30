import { gapTrajectory } from './engine.js';
/* =====================================================================
 charts.js -- D3 ports of the figure panels from the two papers.
 Dark, minimal, no chartjunk.  Every chart is fed data computed by
 engine.js at run time; none is a picture of a stored result.
 ===================================================================== */
import * as d3 from 'd3';

/* ES module form; the body is unchanged. */

const C = {
  c1: '#6d9fd0', c2: '#d0876a', c3: '#7fae8a', c4: '#a98ac0',
  dim: '#4a4a4a', grid: '#141414', axis: '#262626', fg: '#c2c2c2'
};
const SEQ = [C.c1, C.c2, C.c3, C.c4];

let tipEl = null;
function tip() {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tip';
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
function showTip(ev, html) {
  const t = tip();
  t.innerHTML = html;
  t.style.left = (ev.pageX + 12) + 'px';
  t.style.top = (ev.pageY - 10) + 'px';
  t.style.opacity = 1;
}
function hideTip() { if (tipEl) tipEl.style.opacity = 0; }

/* Build a chart frame inside `host` (an .out element). */
function frame(host, title, opts) {
  opts = opts || {};
  const W = opts.width || 300, H = opts.height || 190;
  const m = Object.assign({ t: 10, r: 12, b: 32, l: 42 }, opts.margin || {});
  const box = document.createElement('div');
  box.className = 'chart';
  if (title) {
    const h = document.createElement('p');
    h.className = 'ct'; h.textContent = title;
    box.appendChild(h);
  }
  host.appendChild(box);
  const svg = d3.select(box).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
  return { svg, g, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b, box };
}

function row(host) {
  const r = document.createElement('div');
  r.className = 'chart-row';
  host.appendChild(r);
  return r;
}

function axes(f, x, y, xlab, ylab, opts) {
  opts = opts || {};
  const xa = d3.axisBottom(x).ticks(opts.xticks || 5).tickSizeOuter(0);
  const ya = d3.axisLeft(y).ticks(opts.yticks || 4).tickSizeOuter(0);
  if (opts.xformat) xa.tickFormat(opts.xformat);
  if (opts.yformat) ya.tickFormat(opts.yformat);
  if (opts.xvals) xa.tickValues(opts.xvals);

  f.g.append('g').attr('class', 'axis')
    .attr('transform', `translate(0,${f.ih})`).call(xa);
  f.g.append('g').attr('class', 'axis').call(ya);

  // horizontal gridlines
  f.g.insert('g', ':first-child').selectAll('line')
    .data(y.ticks(opts.yticks || 4)).join('line')
    .attr('class', 'gridline')
    .attr('x1', 0).attr('x2', f.iw)
    .attr('y1', d => y(d)).attr('y2', d => y(d));

  if (xlab) f.g.append('text').attr('class', 'axis-label')
    .attr('x', f.iw / 2).attr('y', f.ih + 27)
    .attr('text-anchor', 'middle').text(xlab);
  if (ylab) f.g.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -f.ih / 2).attr('y', -32)
    .attr('text-anchor', 'middle').text(ylab);
}

function legend(f, items, x0, y0) {
  const g = f.g.append('g').attr('class', 'legend')
    .attr('transform', `translate(${x0},${y0})`);
  items.forEach((it, i) => {
    const r = g.append('g').attr('transform', `translate(0,${i * 12})`);
    r.append('line').attr('x1', 0).attr('x2', 12).attr('y1', 0).attr('y2', 0)
      .attr('stroke', it.color).attr('stroke-width', 2)
      .attr('stroke-dasharray', it.dash || null);
    r.append('text').attr('x', 17).attr('y', 3).text(it.label);
  });
}

// ==================================================================
// 1. Multi-series line chart
// ==================================================================
function lines(host, title, series, opts) {
  opts = opts || {};
  const f = frame(host, title, opts);
  const all = series.flatMap(s => s.points);
  const x = (opts.logx ? d3.scaleLog() : d3.scaleLinear())
    .domain(opts.xdomain || d3.extent(all, p => p[0])).range([0, f.iw]).nice();
  const y = (opts.logy ? d3.scaleLog() : d3.scaleLinear())
    .domain(opts.ydomain || d3.extent(all, p => p[1])).range([f.ih, 0]).nice();
  axes(f, x, y, opts.xlab, opts.ylab, opts);

  if (opts.hline !== undefined) {
    f.g.append('line').attr('x1', 0).attr('x2', f.iw)
      .attr('y1', y(opts.hline)).attr('y2', y(opts.hline))
      .attr('stroke', C.dim).attr('stroke-dasharray', '3,3');
  }
  if (opts.shadeBelow !== undefined) {
    const y0 = y(opts.shadeBelow), y1 = f.ih;
    if (y1 > y0) {
      f.g.insert('rect', ':first-child')
        .attr('x', 0).attr('y', y0).attr('width', f.iw).attr('height', y1 - y0)
        .attr('fill', C.c2).attr('opacity', 0.07);
    }
  }
  const line = d3.line().x(p => x(p[0])).y(p => y(p[1]));
  series.forEach((s, i) => {
    const col = s.color || SEQ[i % SEQ.length];
    f.g.append('path').datum(s.points).attr('fill', 'none')
      .attr('stroke', col).attr('stroke-width', s.width || 1.7)
      .attr('stroke-dasharray', s.dash || null).attr('d', line);
    if (s.dots) {
      f.g.selectAll('.d' + i).data(s.points).join('circle')
        .attr('cx', p => x(p[0])).attr('cy', p => y(p[1])).attr('r', 2.6)
        .attr('fill', col)
        .on('mousemove', (ev, p) => showTip(ev,
          `${s.label || ''}<br>x ${fmt(p[0])}<br>y ${fmt(p[1])}`))
        .on('mouseleave', hideTip);
    }
  });
  if (opts.legend !== false && series.some(s => s.label)) {
    legend(f, series.filter(s => s.label)
      .map((s, i) => ({ label: s.label, color: s.color || SEQ[i % SEQ.length], dash: s.dash })),
      opts.legendX !== undefined ? opts.legendX : f.iw - 84,
      opts.legendY !== undefined ? opts.legendY : 4);
  }
  return f;
}

// ==================================================================
// 2. Bars
// ==================================================================
function bars(host, title, data, opts) {
  opts = opts || {};
  const f = frame(host, title, opts);
  const x = d3.scaleBand().domain(data.map(d => d.label))
    .range([0, f.iw]).padding(0.34);
  const vals = data.map(d => d.value);
  const y = (opts.logy ? d3.scaleLog() : d3.scaleLinear())
    .domain(opts.ydomain ||
            [opts.logy ? d3.min(vals.filter(v => v > 0)) * 0.5 : Math.min(0, d3.min(vals)),
             d3.max(vals) * 1.1 || 1])
    .range([f.ih, 0]).nice();
  axes(f, x, y, opts.xlab, opts.ylab, opts);

  f.g.selectAll('rect').data(data).join('rect')
    .attr('x', d => x(d.label)).attr('width', x.bandwidth())
    .attr('y', d => y(Math.max(d.value, y.domain()[0])))
    .attr('height', d => Math.max(0, y(y.domain()[0]) - y(Math.max(d.value, y.domain()[0]))))
    .attr('fill', (d, i) => d.color || SEQ[i % SEQ.length])
    .on('mousemove', (ev, d) => showTip(ev, `${d.label}<br>${fmt(d.value)}`))
    .on('mouseleave', hideTip);

  if (opts.zeroMark) {
    data.forEach(d => {
      if (d.value === 0) {
        f.g.append('text').attr('x', x(d.label) + x.bandwidth() / 2)
          .attr('y', y(y.domain()[0]) - 5).attr('text-anchor', 'middle')
          .attr('fill', C.c2).attr('font-family', 'monospace')
          .attr('font-size', 11).text('✕');
      }
    });
  }
  return f;
}

// grouped bars
function groupedBars(host, title, groups, keys, opts) {
  opts = opts || {};
  const f = frame(host, title, opts);
  const x0 = d3.scaleBand().domain(groups.map(g => g.label))
    .range([0, f.iw]).padding(0.26);
  const x1 = d3.scaleBand().domain(keys).range([0, x0.bandwidth()]).padding(0.1);
  const maxv = d3.max(groups, g => d3.max(keys, k => g[k]));
  const minv = d3.min(groups, g => d3.min(keys, k => g[k]));
  const y = (opts.logy ? d3.scaleLog() : d3.scaleLinear())
    .domain(opts.ydomain || [opts.logy ? Math.max(1e-18, minv) : Math.min(0, minv), maxv * 1.1])
    .range([f.ih, 0]).nice();
  axes(f, x0, y, opts.xlab, opts.ylab, opts);

  const gs = f.g.selectAll('.grp').data(groups).join('g')
    .attr('transform', d => `translate(${x0(d.label)},0)`);
  keys.forEach((k, i) => {
    gs.append('rect')
      .attr('x', x1(k)).attr('width', x1.bandwidth())
      .attr('y', d => y(Math.max(d[k], y.domain()[0])))
      .attr('height', d => Math.max(0, y(y.domain()[0]) - y(Math.max(d[k], y.domain()[0]))))
      .attr('fill', SEQ[i % SEQ.length])
      .on('mousemove', (ev, d) => showTip(ev, `${d.label} · ${k}<br>${fmt(d[k])}`))
      .on('mouseleave', hideTip);
  });
  legend(f, keys.map((k, i) => ({ label: k, color: SEQ[i % SEQ.length] })),
         opts.legendX !== undefined ? opts.legendX : f.iw - 74,
         opts.legendY !== undefined ? opts.legendY : 4);
  return f;
}

// ==================================================================
// 3. Scatter, with optional identity line
// ==================================================================
function scatter(host, title, series, opts) {
  opts = opts || {};
  const f = frame(host, title, opts);
  const all = series.flatMap(s => s.points);
  const x = d3.scaleLinear().domain(opts.xdomain || d3.extent(all, p => p[0]))
    .range([0, f.iw]).nice();
  const y = d3.scaleLinear().domain(opts.ydomain || d3.extent(all, p => p[1]))
    .range([f.ih, 0]).nice();
  axes(f, x, y, opts.xlab, opts.ylab, opts);

  if (opts.identity) {
    const lo = Math.max(x.domain()[0], y.domain()[0]);
    const hi = Math.min(x.domain()[1], y.domain()[1]);
    f.g.append('line')
      .attr('x1', x(lo)).attr('y1', y(lo)).attr('x2', x(hi)).attr('y2', y(hi))
      .attr('stroke', C.dim).attr('stroke-dasharray', '3,3');
  }
  if (opts.curve) {
    const line = d3.line().x(p => x(p[0])).y(p => y(p[1]));
    f.g.append('path').datum(opts.curve.points).attr('fill', 'none')
      .attr('stroke', opts.curve.color || C.c2).attr('stroke-width', 1.8)
      .attr('d', line);
  }
  series.forEach((s, i) => {
    f.g.selectAll('.s' + i).data(s.points).join('circle')
      .attr('cx', p => x(p[0])).attr('cy', p => y(p[1]))
      .attr('r', s.r || 2.2)
      .attr('fill', s.color || SEQ[i % SEQ.length])
      .attr('opacity', s.opacity === undefined ? 0.62 : s.opacity)
      .on('mousemove', (ev, p) => showTip(ev,
        `${s.label || ''}<br>${fmt(p[0])} → ${fmt(p[1])}`))
      .on('mouseleave', hideTip);
  });
  if (series.some(s => s.label)) {
    legend(f, series.filter(s => s.label).map((s, i) =>
      ({ label: s.label, color: s.color || SEQ[i % SEQ.length] })),
      opts.legendX !== undefined ? opts.legendX : 6,
      opts.legendY !== undefined ? opts.legendY : 4);
  }
  return f;
}

// ==================================================================
// 4. Heatmap -- the 2-D stand-in for the papers' 3-D surfaces
// ==================================================================
function heatmap(host, title, grid, opts) {
  // grid: {xs:[], ys:[], z:[[...]]}  z[yi][xi]
  opts = opts || {};
  const f = frame(host, title, Object.assign(
    { margin: { t: 10, r: 40, b: 32, l: 42 } }, opts));
  const xs = grid.xs, ys = grid.ys;
  const x = d3.scaleLinear().domain([d3.min(xs), d3.max(xs)]).range([0, f.iw]);
  const y = d3.scaleLinear().domain([d3.min(ys), d3.max(ys)]).range([f.ih, 0]);
  const flat = grid.z.flat();
  const col = d3.scaleSequential(opts.scheme || d3.interpolateViridis)
    .domain(opts.zdomain || [d3.min(flat), d3.max(flat)]);
  const cw = f.iw / (xs.length - 1 || 1), ch = f.ih / (ys.length - 1 || 1);

  for (let j = 0; j < ys.length; j++) {
    for (let i = 0; i < xs.length; i++) {
      f.g.append('rect')
        .attr('x', x(xs[i]) - cw / 2).attr('y', y(ys[j]) - ch / 2)
        .attr('width', cw + 0.6).attr('height', ch + 0.6)
        .attr('fill', col(grid.z[j][i]))
        .on('mousemove', ev => showTip(ev,
          `${opts.xlab || 'x'} ${fmt(xs[i])}<br>` +
          `${opts.ylab || 'y'} ${fmt(ys[j])}<br>` +
          `${opts.zlab || 'z'} ${fmt(grid.z[j][i])}`))
        .on('mouseleave', hideTip);
    }
  }
  // axes drawn over the cells
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`)
    .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
  f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
  if (opts.xlab) f.g.append('text').attr('class', 'axis-label')
    .attr('x', f.iw / 2).attr('y', f.ih + 27).attr('text-anchor', 'middle').text(opts.xlab);
  if (opts.ylab) f.g.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)').attr('x', -f.ih / 2).attr('y', -32)
    .attr('text-anchor', 'middle').text(opts.ylab);

  // colour ramp
  const rampId = 'r' + Math.random().toString(36).slice(2, 8);
  const defs = f.svg.append('defs');
  const lg = defs.append('linearGradient').attr('id', rampId)
    .attr('x1', '0').attr('y1', '1').attr('x2', '0').attr('y2', '0');
  d3.range(0, 1.01, 0.1).forEach(t => {
    lg.append('stop').attr('offset', (t * 100) + '%')
      .attr('stop-color', col(col.domain()[0] + t * (col.domain()[1] - col.domain()[0])));
  });
  f.g.append('rect').attr('x', f.iw + 8).attr('y', 0)
    .attr('width', 8).attr('height', f.ih).attr('fill', `url(#${rampId})`);
  f.g.append('text').attr('class', 'axis-label')
    .attr('x', f.iw + 8).attr('y', -2).attr('font-size', 8.5)
    .text(fmt(col.domain()[1]));
  f.g.append('text').attr('class', 'axis-label')
    .attr('x', f.iw + 8).attr('y', f.ih + 9).attr('font-size', 8.5)
    .text(fmt(col.domain()[0]));
  return f;
}

// ==================================================================
// 5. Ladder diagram -- gap closing rung by rung
// ==================================================================
function ladderChart(host, title, powers, opts) {
  opts = opts || {};
  const f = frame(host, title, Object.assign({ height: 200 }, opts));
  const traj = gapTrajectory(powers, 1);
  const n = powers.length;
  const x = d3.scaleLinear().domain([0, n]).range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
  axes(f, x, y, 'rungs applied', 'gap remaining',
       { xticks: Math.min(n, 8), xformat: d3.format('d') });

  // each rung as a band showing the slice of gap it closes
  for (let i = 0; i < n; i++) {
    f.g.append('rect')
      .attr('x', x(i)).attr('width', Math.max(1, x(i + 1) - x(i) - 2))
      .attr('y', y(traj[i])).attr('height', Math.max(0, y(traj[i + 1]) - y(traj[i])))
      .attr('fill', SEQ[i % SEQ.length]).attr('opacity', 0.55)
      .on('mousemove', ev => showTip(ev,
        `rung ${i + 1}<br>power ${fmt(powers[i])}<br>` +
        `gap ${fmt(traj[i])} → ${fmt(traj[i + 1])}`))
      .on('mouseleave', hideTip);
  }
  const line = d3.line().x((d, i) => x(i)).y(d => y(d)).curve(d3.curveStepAfter);
  f.g.append('path').datum(traj).attr('fill', 'none')
    .attr('stroke', '#e2e2e2').attr('stroke-width', 1.8).attr('d', line);
  f.g.selectAll('.pt').data(traj).join('circle')
    .attr('cx', (d, i) => x(i)).attr('cy', d => y(d)).attr('r', 2.6)
    .attr('fill', '#e2e2e2');

  if (opts.target !== undefined) {
    const gapTarget = 1 - opts.target;
    f.g.append('line').attr('x1', 0).attr('x2', f.iw)
      .attr('y1', y(gapTarget)).attr('y2', y(gapTarget))
      .attr('stroke', C.c2).attr('stroke-dasharray', '4,3');
    f.g.append('text').attr('class', 'axis-label')
      .attr('x', f.iw - 2).attr('y', y(gapTarget) - 4)
      .attr('text-anchor', 'end').attr('fill', C.c2)
      .text('target ' + fmt(opts.target));
  }
  return f;
}

// ==================================================================
// 6. Contact graph -- force layout, medium at the centre
// ==================================================================
function graphChart(host, title, g, opts) {
  opts = opts || {};
  const f = frame(host, title, Object.assign(
    { height: 210, margin: { t: 8, r: 8, b: 8, l: 8 } }, opts));
  const nodes = [...g.vertices].map(id => ({ id, medium: id === g.medium }));
  const links = [...g.weights.entries()].map(([k, w]) => {
    const [a, b] = k.split('|');
    return { source: a, target: b, w };
  });
  const wex = d3.extent(links, l => l.w);
  const sw = d3.scaleLinear().domain(wex).range([0.5, 2.6]);
  const powers = {};
  if (opts.powers) g.items().forEach((v, i) => { powers[v] = opts.powers[i]; });
  const pcol = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

  /* Deterministic layout: the medium at the centre, items on a ring.
     A force simulation would place them too, but it depends on d3's
     timer and gives a different picture on every reload; a fixed layout
     is reproducible, and for a graph this small it reads at least as
     well. */
  const cx = f.iw / 2, cy = f.ih / 2;
  const R = Math.min(f.iw, f.ih) / 2 - 14;
  const items = nodes.filter(n => !n.medium);
  const med = nodes.find(n => n.medium);
  if (med) { med.x = cx; med.y = cy; }
  items.forEach((n, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / items.length;
    n.x = cx + R * Math.cos(a);
    n.y = cy + R * Math.sin(a);
  });
  const byId = new Map(nodes.map(n => [n.id, n]));
  links.forEach(l => {
    if (typeof l.source === 'string') l.source = byId.get(l.source);
    if (typeof l.target === 'string') l.target = byId.get(l.target);
  });

  f.g.selectAll('line.e').data(links).join('line')
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
    .attr('stroke', d => (d.source.medium || d.target.medium) ? '#1c1c1c' : '#3a3a3a')
    .attr('stroke-width', d => sw(d.w))
    .on('mousemove', (ev, d) => showTip(ev,
      `${d.source.id}–${d.target.id}<br>w ${fmt(d.w)}`))
    .on('mouseleave', hideTip);

  const nsel = f.g.selectAll('g.n').data(nodes).join('g')
    .attr('transform', d => `translate(${d.x},${d.y})`);
  nsel.append('circle')
    .attr('r', d => d.medium ? 9 : 7)
    .attr('fill', d => d.medium ? '#111'
      : (opts.powers ? pcol(powers[d.id] || 0) : '#2c3f52'))
    .attr('stroke', d => d.medium ? '#3a3a3a' : '#0a0a0a')
    .attr('stroke-width', d => d.medium ? 1.2 : 1)
    .attr('stroke-dasharray', d => d.medium ? '2,2' : null)
    .on('mousemove', (ev, d) => showTip(ev, d.medium
      ? 'medium — adjacent to every item, never individuated'
      : `${d.id}${opts.powers ? '<br>power ' + fmt(powers[d.id]) : ''}`))
    .on('mouseleave', hideTip);
  nsel.append('text')
    .attr('text-anchor', 'middle').attr('dy', 3)
    .attr('font-family', 'monospace').attr('font-size', 7.5)
    .attr('fill', d => d.medium ? '#777' : '#0a0a0a')
    .attr('pointer-events', 'none')
    .text(d => d.medium ? 'm' : d.id.replace('v', ''));
  return f;
}

// ==================================================================
// 7. Histogram
// ==================================================================
function histogram(host, title, values, opts) {
  opts = opts || {};
  const f = frame(host, title, opts);
  const x = d3.scaleLinear().domain(opts.xdomain || d3.extent(values))
    .range([0, f.iw]).nice();
  const bins = d3.bin().domain(x.domain()).thresholds(opts.bins || 26)(values);
  const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length)])
    .range([f.ih, 0]).nice();
  axes(f, x, y, opts.xlab, opts.ylab || 'count', opts);
  f.g.selectAll('rect').data(bins).join('rect')
    .attr('x', b => x(b.x0) + 0.5)
    .attr('width', b => Math.max(0.5, x(b.x1) - x(b.x0) - 1))
    .attr('y', b => y(b.length))
    .attr('height', b => f.ih - y(b.length))
    .attr('fill', opts.color || C.c2)
    .on('mousemove', (ev, b) => showTip(ev,
      `[${fmt(b.x0)}, ${fmt(b.x1)})<br>n ${b.length}`))
    .on('mouseleave', hideTip);
  if (opts.vline !== undefined) {
    f.g.append('line').attr('x1', x(opts.vline)).attr('x2', x(opts.vline))
      .attr('y1', 0).attr('y2', f.ih)
      .attr('stroke', opts.vlineColor || C.c1).attr('stroke-width', 2);
  }
  return f;
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-4) return d3.format('.3e')(v);
  if (Number.isInteger(v)) return String(v);
  return d3.format(a < 1 ? '.4f' : '.3f')(v);
}


export {
  row, lines, bars, groupedBars, scatter, heatmap,
  ladderChart, graphChart, histogram, fmt, C, SEQ
};
