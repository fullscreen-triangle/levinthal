/* =====================================================================
   main.js -- the runtime: the stage, the cells, the controls, the rail.
   ===================================================================== */

import './styles/main.css';
import { CELLS } from './cells.js';
import { SHK_CELLS, playOptions } from './shkcells.js';
import { clock, resetReceiver, LESSONS } from './lib/shk.js';

const ALL = { ...CELLS, ...SHK_CELLS };

const ST = {
  seed: 7, nItems: 6, radius: 1,
  powers: [0.45, 0.30, 0.55, 0.20],
  target: 0.80,
  hfqTarget: 0.70,
  reach: 0.70,
  playIdx: 0,
  pathwayIdx: 0,
  corpus: null,
  graph: null,
};

// ---------------------------------------------------------------------
// Syntax highlighting: one pass, escaping as it emits.  Three separate
// regex replacements over already-escaped text would match the class
// attributes injected by the earlier passes and leak markup.
// ---------------------------------------------------------------------
const KW = new Set([
  'for', 'in', 'if', 'else', 'return', 'def', 'class', 'import', 'from',
  'while', 'try', 'except', 'with', 'as', 'and', 'or', 'not', 'None',
  'True', 'False', 'lambda',
  // shakespeare
  'receiver', 'floor', 'medium', 'ambient', 'depleted', 'exchange',
  'cut', 'contact', 'close', 'complex', 'fold', 'track', 'until', 'yield',
  'when', 'observe', 'catalyze', 'mutate', 'complete', 'let', 'converge',
  'diverge', 'by', 'reps', 'assert', 'emit', 'scene', 'perform',
  'residue', 'cofactor', 'electronic', 'solvent', 'substrate',
  'ladder', 'power', 'climb', 'reach', 'derived', 'radius', 'role',
  // hfq
  'plan', 'budget', 'requests', 'ask', 'within', 'over', 'expect', 'partial',
  'map', 'via', 'union', 'intersect', 'join', 'filter', 'where',
]);

function highlight(src) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const span = (cls, txt) => `<span class="${cls}">${esc(txt)}</span>`;

  return src.split('\n').map(line => {
    let out = '', i = 0;
    while (i < line.length) {
      const c = line[i];
      if (c === '#') { out += span('c', line.slice(i)); break; }
      if (c === '-' && line[i + 1] === '-') { out += span('c', line.slice(i)); break; }
      if (c === '"') {
        let j = i + 1;
        while (j < line.length && line[j] !== '"') j++;
        out += span('s', line.slice(i, Math.min(j + 1, line.length)));
        i = j + 1; continue;
      }
      if (/[0-9]/.test(c) && (i === 0 || !/[A-Za-z0-9_]/.test(line[i - 1]))) {
        const m = /^[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?/.exec(line.slice(i));
        if (m) { out += span('n', m[0]); i += m[0].length; continue; }
      }
      if (/[A-Za-z_]/.test(c)) {
        const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(line.slice(i));
        out += KW.has(m[0]) ? span('k', m[0]) : esc(m[0]);
        i += m[0].length; continue;
      }
      out += esc(c); i++;
    }
    return out;
  }).join('\n');
}

// ---------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------
const REGISTRY = [];
let counter = 0;

function buildCell(host) {
  const name = host.dataset.cell;
  const spec = ALL[name];
  if (!spec) { host.textContent = 'missing cell: ' + name; return; }
  const n = ++counter;

  const inWrap = document.createElement('div');
  inWrap.className = 'cell-in';
  const g1 = document.createElement('div');
  g1.className = 'gutter';
  g1.textContent = `In [${n}]`;

  const src = document.createElement('div');
  src.className = 'src' + (spec.lang ? ' shk' : '');
  if (spec.lang) {
    const lg = document.createElement('div');
    lg.className = 'lang';
    lg.textContent = spec.lang.toUpperCase();
    src.appendChild(lg);
  }
  const pre = document.createElement('pre');
  src.appendChild(pre);

  const bar = document.createElement('div');
  bar.className = 'runbar';
  const btn = document.createElement('button');
  btn.className = 'run';
  btn.textContent = '▶ run';
  const hint = document.createElement('span');
  hint.className = 'hint';
  bar.appendChild(btn); bar.appendChild(hint);
  src.appendChild(bar);
  inWrap.appendChild(g1); inWrap.appendChild(src);

  const outWrap = document.createElement('div');
  outWrap.className = 'cell-out';
  const g2 = document.createElement('div');
  g2.className = 'gutter';
  const out = document.createElement('div');
  out.className = 'out';
  outWrap.appendChild(g2); outWrap.appendChild(out);

  host.appendChild(inWrap); host.appendChild(outWrap);

  const entry = { name, spec, out, btn, hint, g1, g2, n, pre };
  paintSource(entry);
  REGISTRY.push(entry);
  btn.addEventListener('click', () => runCell(entry));
  return entry;
}

function paintSource(e) {
  const text = e.spec.srcFor ? e.spec.srcFor(ST) : e.spec.src;
  e.pre.innerHTML = highlight(text);
}

function runCell(e) {
  e.out.innerHTML = '';
  e.g1.className = 'gutter busy';
  e.g1.textContent = 'In [*]';
  e.btn.disabled = true;
  e.hint.textContent = 'running…';
  return new Promise(resolve => setTimeout(() => {
    const t0 = performance.now();
    try {
      e.spec.run(e.out, ST);
      e.hint.textContent = (performance.now() - t0).toFixed(0) + ' ms';
      e.g1.className = 'gutter done';
      e.g1.textContent = `In [${e.n}]`;
      e.g2.textContent = `Out[${e.n}]`;
    } catch (err) {
      const p = document.createElement('pre');
      p.className = 'err';
      p.textContent = err && err.stack ? err.stack : String(err);
      e.out.appendChild(p);
      e.hint.textContent = 'error';
      e.g1.className = 'gutter';
      e.g1.textContent = `In [${e.n}]`;
    }
    e.btn.disabled = false;
    paintClock();
    resolve();
  }, 12));
}

function rerun(names) {
  REGISTRY.filter(e => names.includes(e.name)).forEach(e => {
    paintSource(e);
    if (e.out.innerHTML !== '') runCell(e);
  });
}

function paintClock() {
  const el = document.getElementById('clockval');
  if (el) el.textContent = clock();
}

// ---------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------
function slider(id, vid, onChange, fmt) {
  const el = document.getElementById(id);
  const lab = document.getElementById(vid);
  if (!el) return;
  const paint = () => { lab.textContent = fmt ? fmt(parseFloat(el.value)) : el.value; };
  paint();
  let t = null;
  el.addEventListener('input', () => {
    paint();
    clearTimeout(t);
    t = setTimeout(() => onChange(parseFloat(el.value)), 130);
  });
}

function wireControls() {
  const f2 = v => v.toFixed(2);
  slider('c-items', 'v-items', v => { ST.nItems = v; rerun(['substrate', 'semantics']); });
  slider('c-radius', 'v-radius', v => { ST.radius = v; rerun(['substrate']); });
  slider('c-seed', 'v-seed', v => {
    ST.seed = v;
    rerun(['substrate', 'resolution', 'intensivity', 'order', 'laws', 'semantics', 'bound']);
  });
  [0, 1, 2, 3].forEach(i => slider('c-p' + i, 'v-p' + i, v => {
    ST.powers[i] = v; rerun(['ladder', 'semantics']);
  }, f2));
  slider('c-target', 'v-target', v => { ST.target = v; rerun(['ladder', 'semantics']); }, f2);
  slider('c-hfq', 'v-hfq', v => { ST.hfqTarget = v; rerun(['federated']); }, f2);
  slider('c-reach', 'v-reach', v => { ST.reach = v; rerun(['shkladder']); }, f2);

  const play = document.getElementById('c-play');
  if (play) {
    playOptions().forEach(o => {
      const el = document.createElement('option');
      el.value = o.value; el.textContent = o.label;
      play.appendChild(el);
    });
    play.addEventListener('change', () => {
      ST.playIdx = parseInt(play.value, 10);
      rerun(['shkplay']);
    });
  }

  const pw = document.getElementById('c-pathway');
  if (pw) pw.addEventListener('change', () => {
    ST.pathwayIdx = parseInt(pw.value, 10);
    rerun(['sequence']);
  });

  const rst = document.getElementById('reset-recv');
  if (rst) rst.addEventListener('click', () => {
    resetReceiver();
    paintClock();
    rerun(['shkplay', 'shkladder']);
  });
}

function fillPathways() {
  const sel = document.getElementById('c-pathway');
  if (!sel || !ST.corpus) return;
  sel.innerHTML = '';
  ST.corpus.pathways.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = i;
    o.textContent = `${p.pathway.slice(0, 40)}  (${p.n_total} reactions)`;
    sel.appendChild(o);
  });
}

// ---------------------------------------------------------------------
// The stage: one link open at a time
// ---------------------------------------------------------------------
function wireStage() {
  const links = [...document.querySelectorAll('#chain .link')];
  links.forEach((l, i) => {
    l.addEventListener('click', ev => {
      const wasOn = l.classList.contains('on');
      links.forEach(x => x.classList.remove('on'));
      if (!wasOn) l.classList.add('on');
      // a second click on an open link jumps to its section
      if (wasOn && l.dataset.goto) {
        const t = document.getElementById(l.dataset.goto);
        if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      ev.preventDefault();
    });
  });
  if (links[0]) links[0].classList.add('on');
}

function buildTOC() {
  const box = document.getElementById('toc-links');
  const secs = [...document.querySelectorAll('section.md[id]')];
  secs.forEach(s => {
    const h = s.querySelector('h2');
    if (!h) return;
    const a = document.createElement('a');
    a.href = '#' + s.id;
    a.textContent = h.textContent;
    box.appendChild(a);
  });
  const links = [...box.querySelectorAll('a')];
  const obs = new IntersectionObserver(es => {
    es.forEach(en => {
      if (en.isIntersecting) {
        links.forEach(l => l.classList.toggle('on',
          l.getAttribute('href') === '#' + en.target.id));
      }
    });
  }, { rootMargin: '-10% 0px -80% 0px' });
  secs.forEach(s => obs.observe(s));
}

// ---------------------------------------------------------------------
async function boot() {
  document.querySelectorAll('.cell[data-cell]').forEach(buildCell);
  buildTOC();
  wireStage();
  wireControls();
  paintClock();

  const kc = document.getElementById('kcorpus');
  try {
    const r = await fetch('/data/corpus.json');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    ST.corpus = await r.json();
    kc.textContent = `${ST.corpus.snapshot} · ${ST.corpus.totals.kegg_records} KEGG · ` +
                     `${ST.corpus.totals.reactome_reactions} Reactome`;
    fillPathways();
  } catch {
    kc.textContent = 'unavailable';
    kc.style.color = 'var(--warn)';
  }

  document.getElementById('kstate').textContent =
    `${LESSONS.length} plays loaded`;

  document.getElementById('runall').addEventListener('click', async ev => {
    ev.preventDefault();
    const st = document.getElementById('kstate');
    st.textContent = 'running';
    for (const e of REGISTRY) await runCell(e);
    st.textContent = 'idle';
  });

  for (const name of ['substrate', 'ladder', 'shkladder']) {
    const e = REGISTRY.find(x => x.name === name);
    if (e) await runCell(e);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else { boot(); }
