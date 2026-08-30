/* =====================================================================
   notebook.js -- the runtime: renders cells, wires the controls, runs
   the computations, builds the table of contents.
   ===================================================================== */
(function () {
  'use strict';

  // shared state every cell reads
  const ST = {
    seed: 7,
    nItems: 6,
    radius: 1,
    powers: [0.45, 0.30, 0.55, 0.20],
    target: 0.80,
    hfqTarget: 0.70,
    pathwayIdx: 0,
    corpus: null,
    graph: null
  };

  // ------------------------------------------------------------------
  // Minimal syntax highlighting for the python-ish source listings
  // ------------------------------------------------------------------
  const KW = new Set(['for', 'in', 'if', 'else', 'return', 'def', 'class',
    'import', 'from', 'while', 'try', 'except', 'with', 'as', 'and', 'or',
    'not', 'None', 'True', 'False', 'lambda', 'plan', 'let', 'ladder', 'power',
    'expect', 'emit', 'budget', 'requests', 'ask', 'within', 'over']);

  /* Tokenise in ONE pass and escape as we emit.  The previous version ran
     three independent regex replacements over already-escaped text, so the
     class attributes injected by the first pass were themselves matched by
     the later ones and leaked into the listing as visible markup. */
  function highlight(src) {
    const esc = s => s.replace(/&/g, '&amp;')
                      .replace(/</g, '&lt;')
                      .replace(/>/g, '&gt;');
    const span = (cls, txt) => '<span class="' + cls + '">' + esc(txt) + '</span>';

    return src.split('\n').map(line => {
      let out = '';
      let i = 0;
      while (i < line.length) {
        const c = line[i];

        // comment to end of line
        if (c === '#') { out += span('c', line.slice(i)); break; }

        // double-quoted string
        if (c === '"') {
          let j = i + 1;
          while (j < line.length && line[j] !== '"') j++;
          out += span('s', line.slice(i, Math.min(j + 1, line.length)));
          i = j + 1;
          continue;
        }

        // number (not when it is part of an identifier such as mulberry32)
        if (/[0-9]/.test(c) && (i === 0 || !/[A-Za-z0-9_]/.test(line[i - 1]))) {
          const m = /^[0-9]+\.?[0-9]*(?:e-?[0-9]+)?/.exec(line.slice(i));
          if (m) { out += span('n', m[0]); i += m[0].length; continue; }
        }

        // identifier or keyword
        if (/[A-Za-z_]/.test(c)) {
          const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(line.slice(i));
          out += KW.has(m[0]) ? span('k', m[0]) : esc(m[0]);
          i += m[0].length;
          continue;
        }

        out += esc(c);
        i++;
      }
      return out;
    }).join('\n');
  }

  // ------------------------------------------------------------------
  // Cell construction
  // ------------------------------------------------------------------
  const REGISTRY = [];
  let counter = 0;

  function buildCell(host) {
    const name = host.dataset.cell;
    const spec = window.CELLS[name];
    if (!spec) { host.textContent = 'missing cell: ' + name; return; }
    const n = ++counter;

    const inWrap = document.createElement('div');
    inWrap.className = 'cell-in';
    const g1 = document.createElement('div');
    g1.className = 'gutter';
    g1.textContent = 'In [' + n + ']';
    const src = document.createElement('div');
    src.className = 'src';
    const pre = document.createElement('pre');
    pre.innerHTML = highlight(spec.src);
    const bar = document.createElement('div');
    bar.className = 'runbar';
    const btn = document.createElement('button');
    btn.className = 'run';
    btn.textContent = '▶ run';
    const hint = document.createElement('span');
    hint.className = 'hint';
    bar.appendChild(btn); bar.appendChild(hint);
    src.appendChild(pre); src.appendChild(bar);
    inWrap.appendChild(g1); inWrap.appendChild(src);

    const outWrap = document.createElement('div');
    outWrap.className = 'cell-out';
    const g2 = document.createElement('div');
    g2.className = 'gutter';
    const out = document.createElement('div');
    out.className = 'out';
    outWrap.appendChild(g2); outWrap.appendChild(out);

    host.appendChild(inWrap); host.appendChild(outWrap);

    const entry = { name, spec, out, btn, hint, g1, g2, n };
    REGISTRY.push(entry);
    btn.addEventListener('click', () => runCell(entry));
    return entry;
  }

  function runCell(e) {
    e.out.innerHTML = '';
    e.g1.className = 'gutter busy';
    e.g1.textContent = 'In [*]';
    e.btn.disabled = true;
    e.hint.textContent = 'running…';
    // yield so the busy state paints before a synchronous computation
    return new Promise(resolve => setTimeout(() => {
      const t0 = performance.now();
      try {
        e.spec.run(e.out, ST);
        const ms = performance.now() - t0;
        e.hint.textContent = ms.toFixed(0) + ' ms';
        e.g1.className = 'gutter done';
        e.g1.textContent = 'In [' + e.n + ']';
        e.g2.textContent = 'Out[' + e.n + ']';
      } catch (err) {
        const p = document.createElement('pre');
        p.className = 'err';
        p.textContent = (err && err.stack ? err.stack : String(err));
        e.out.appendChild(p);
        e.hint.textContent = 'error';
        e.g1.className = 'gutter';
        e.g1.textContent = 'In [' + e.n + ']';
      }
      e.btn.disabled = false;
      resolve();
    }, 12));
  }

  function rerun(names) {
    REGISTRY.filter(e => names.includes(e.name) && e.out.innerHTML !== '')
      .forEach(e => runCell(e));
  }

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------
  function slider(id, vid, onChange, fmt) {
    const el = document.getElementById(id);
    const lab = document.getElementById(vid);
    if (!el) return;
    const paint = () => {
      const v = parseFloat(el.value);
      lab.textContent = fmt ? fmt(v) : v;
    };
    paint();
    let t = null;
    el.addEventListener('input', () => {
      paint();
      clearTimeout(t);
      t = setTimeout(() => onChange(parseFloat(el.value)), 130);
    });
  }

  function wireControls() {
    slider('c-items', 'v-items', v => { ST.nItems = v; rerun(['substrate', 'semantics']); });
    slider('c-radius', 'v-radius', v => { ST.radius = v; rerun(['substrate']); });
    slider('c-seed', 'v-seed', v => { ST.seed = v; rerun(['substrate', 'resolution', 'intensivity', 'order', 'laws', 'semantics', 'bound']); });

    [0, 1, 2, 3].forEach(i => {
      slider('c-p' + i, 'v-p' + i, v => {
        ST.powers[i] = v;
        rerun(['ladder', 'semantics']);
      }, v => v.toFixed(2));
    });
    slider('c-target', 'v-target', v => { ST.target = v; rerun(['ladder', 'semantics']); },
      v => v.toFixed(2));
    slider('c-hfq', 'v-hfq', v => { ST.hfqTarget = v; rerun(['federated']); },
      v => v.toFixed(2));

    const sel = document.getElementById('c-pathway');
    if (sel) {
      sel.addEventListener('change', () => {
        ST.pathwayIdx = parseInt(sel.value, 10);
        rerun(['sequence']);
      });
    }
  }

  function fillPathways() {
    const sel = document.getElementById('c-pathway');
    if (!sel || !ST.corpus) return;
    sel.innerHTML = '';
    ST.corpus.pathways.forEach((p, i) => {
      const o = document.createElement('option');
      o.value = i;
      o.textContent = p.pathway.slice(0, 42) + '  (' + p.n_total + ' reactions)';
      sel.appendChild(o);
    });
  }

  // ------------------------------------------------------------------
  // Table of contents
  // ------------------------------------------------------------------
  function buildTOC() {
    const toc = document.getElementById('toc');
    const secs = [...document.querySelectorAll('section.md[id]')];
    secs.forEach(s => {
      const h = s.querySelector('h2');
      if (!h) return;
      const a = document.createElement('a');
      a.href = '#' + s.id;
      a.textContent = h.textContent;
      toc.appendChild(a);
    });
    const links = [...toc.querySelectorAll('a')];
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

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  async function boot() {
    document.querySelectorAll('.cell[data-cell]').forEach(buildCell);
    buildTOC();
    wireControls();

    const kc = document.getElementById('kcorpus');
    try {
      const r = await fetch('data/corpus.json');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      ST.corpus = await r.json();
      kc.textContent = ST.corpus.snapshot + ' · ' +
        ST.corpus.totals.kegg_records + ' KEGG · ' +
        ST.corpus.totals.reactome_reactions + ' Reactome';
      fillPathways();
    } catch (err) {
      kc.textContent = 'unavailable (serve over http to load)';
      kc.style.color = 'var(--warn)';
    }

    document.getElementById('kstate').textContent = 'ready';

    document.getElementById('runall').addEventListener('click', async ev => {
      ev.preventDefault();
      const dot = document.getElementById('kdot');
      const state = document.getElementById('kstate');
      dot.style.background = 'var(--warn)';
      state.textContent = 'running';
      for (const e of REGISTRY) await runCell(e);
      dot.style.background = 'var(--ok)';
      state.textContent = 'idle';
    });

    // run the first cells so the page is not empty on arrival
    const boot_all = new URLSearchParams(location.search).has('all');
    const initial = boot_all ? REGISTRY.map(x => x.name) : ['substrate', 'ladder'];
    for (const name of initial) {
      const e = REGISTRY.find(x => x.name === name);
      if (e) await runCell(e);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else { boot(); }
})();
