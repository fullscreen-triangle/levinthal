/* =====================================================================
   shkcells.js -- cells that run real .shk plays.

   These do not simulate the language. They call `perform` from
   src/lib/shk.js, which delegates to the interpreter vendored unmodified
   from cytochrome/src/helpers/shakespeare.js. The receiver accumulates
   across the whole page, so the clock a cell reports is the clock the
   page is actually at.
   ===================================================================== */

import * as CH from './lib/charts.js';
import * as LAD from './lib/engine.js';
import { perform, receiver, clock, LESSONS } from './lib/shk.js';

const fmt = v => CH.fmt(v);

function consoleBlock(out, lines) {
  const box = document.createElement('div');
  box.className = 'shk-console';
  for (const l of lines) {
    const d = document.createElement('div');
    d.className = 'k-' + (l.kind || 'log');
    d.textContent = l.text;
    box.appendChild(d);
  }
  out.appendChild(box);
  return box;
}

function note(out, html) {
  const p = document.createElement('pre');
  p.innerHTML = html;
  out.appendChild(p);
}

const dim = s => `<span class="dim">${s}</span>`;
const val = s => `<span class="val">${s}</span>`;
const okv = s => `<span class="ok">${s}</span>`;
const badv = s => `<span class="badv">${s}</span>`;
const warnv = s => `<span class="warnv">${s}</span>`;

// ---------------------------------------------------------------------
// Draw the ladder charts a play emitted
// ---------------------------------------------------------------------
function ladderCharts(out, data) {
  if (!data) return;
  const { powers, target, composite, traj, sens } = data;
  const r = CH.row(out);
  CH.ladderChart(r, 'the gap closing, rung by rung', powers,
    target != null ? { target } : {});
  const s = sens || LAD.sensitivity(powers);
  const maxS = Math.max(...s);
  CH.bars(r, 'power per rung',
    powers.map((p, i) => ({
      label: 'γ' + (i + 1), value: p,
      color: Math.abs(s[i] - maxS) < 1e-12 ? CH.C.c2 : CH.C.c1
    })), { ylab: 'π', ydomain: [0, 1] });
  CH.lines(r, 'composite against rungs applied',
    [{
      points: powers.map((_, i) =>
        [i + 1, LAD.composeMultiplicative(powers.slice(0, i + 1))]),
      label: 'composite', color: CH.C.c3, dots: true
    }],
    {
      xlab: 'rungs applied', ylab: 'composite', ydomain: [0, 1],
      hline: target != null ? target : undefined,
      xformat: d => String(d)
    });
}

export const SHK_CELLS = {};

// ---------------------------------------------------------------------
// A monograph play, run through the vendored interpreter
// ---------------------------------------------------------------------
SHK_CELLS.shkplay = {
  lang: 'shakespeare · .shk',
  srcFor(st) {
    const L = LESSONS[st.playIdx % LESSONS.length];
    return L.src.trim();
  },
  run(out, st) {
    const L = LESSONS[st.playIdx % LESSONS.length];
    const before = clock();
    const res = perform(L.src, L);
    const after = clock();

    note(out,
      `${dim('play    ')} ${val(L.id)}\n` +
      `${dim('performs')} ${val(L.paper || '—')}\n` +
      `${dim('verbs   ')} ${dim((L.verbs || []).join(', '))}\n` +
      `${dim('clock   ')} M ${val(before)} → ${val(after)} ` +
      `${dim('(this play committed ' + (after - before) + ' cuts)')}`);

    consoleBlock(out, res.console);

    note(out,
      `\n${dim('This is the interpreter behind the cytochrome IDE, vendored')}\n` +
      `${dim('unmodified. Heavy verbs replay the monograph oracle rather than')}\n` +
      `${dim('recomputing, so a ✓ here is a real comparison against a')}\n` +
      `${dim('validated number, not decoration.')}\n` +
      `\n${warnv('Re-run this cell and M advances again.')} ` +
      `${dim('Re-evaluating a play is a')}\n${dim('NEW measurement at a higher clock, never a cached retrieval.')}`);

    // any charts the play emitted, rendered as data summaries
    const names = Object.keys(res.charts || {});
    if (names.length) {
      note(out, `\n${dim('charts emitted by this play: ')}${val(names.join(', '))}`);
      const cm = res.charts['contact-map'];
      if (Array.isArray(cm) && cm.length) {
        const r = CH.row(out);
        CH.scatter(r, 'contact map (residue i against j)',
          [{ points: cm.map(d => [d.i ?? d[0], d.j ?? d[1]]), color: CH.C.c1, r: 1.5, opacity: 0.5 }],
          { xlab: 'residue i', ylab: 'residue j' });
      }
      const dl = res.charts['depth-ladder'];
      if (Array.isArray(dl) && dl.length && typeof dl[0] === 'object') {
        const r2 = cm ? out.querySelector('.chart-row:last-of-type') : CH.row(out);
        CH.bars(r2 || CH.row(out), 'address trits by depth',
          dl.map((d, i) => ({
            label: String(d.depth ?? i + 1),
            value: Number(d.value ?? d.trit ?? 0), color: CH.C.c3
          })), { xlab: 'depth', ylab: 'trit' });
      }
    }
  }
};

// ---------------------------------------------------------------------
// The ladder as a play: free rules, the costed climb, and the refusal
// ---------------------------------------------------------------------
SHK_CELLS.shkladder = {
  lang: 'shakespeare · .shk',
  srcFor(st) {
    return `receiver bio
floor 3.7e-4

-- individuate two leaves; each is a cut, so each moves the clock
heme := cofactor "FE-heme"
sub  := substrate "camphor"

-- declaring a ladder commits NO cut: it is a number, not a boundary
L := ladder [power 0.45, power 0.30, power 0.55]

-- reading a composite is free for the same reason
observe L as power

-- climbing is the one costed rule: one commitment per rung
climb heme with L reach ${st.reach.toFixed(2)}`;
  },
  run(out, st) {
    const before = clock();
    const src = SHK_CELLS.shkladder.srcFor(st);
    const res = perform(src, { id: 'ladder-play', oracle: {} });
    const after = clock();

    consoleBlock(out, res.console);

    const refused = !!res.charts['ladder-refused'];
    const data = res.charts['ladder'] || res.charts['ladder-refused'];
    const composite = data ? data.composite : null;

    note(out,
      `\n${dim('clock   ')} M ${val(before)} → ${val(after)} ` +
      (refused
        ? badv('  (refused: nothing committed)')
        : dim('  (two leaf cuts + one per rung)')) + '\n' +
      `${dim('composite')} ${val(fmt(composite))}   ${dim('target ')}${val(st.reach.toFixed(2))}`);

    if (refused) {
      note(out,
        `\n${badv('The play was refused before evaluation.')}\n` +
        dim('The refusal is TIGHT: by the multiplicative law the composite is\n' +
            'exactly what executing every rung would attain, so no ladder that\n' +
            'could have succeeded is rejected here.  And it commits nothing —\n' +
            'the clock did not move for a program the language declined to run.'));
    } else {
      note(out,
        `\n${okv('Reached.')} ` +
        dim('Raise the target above ' + fmt(composite) + ' and the same play is\n') +
        dim('refused, with the shortfall named and the clock left alone.'));
    }

    ladderCharts(out, data && data.traj ? data : {
      powers: data ? data.powers : [], target: st.reach,
      composite, traj: LAD.gapTrajectory(data ? data.powers : [], 1),
      sens: LAD.sensitivity(data ? data.powers : [])
    });
  }
};

export function playOptions() {
  return LESSONS.map((L, i) => ({
    value: i,
    label: `${L.id.replace(/^\d+_/, '')}  ·  ${(L.paper || '').split('·').pop().trim()}`
  }));
}

export { receiver, clock };
