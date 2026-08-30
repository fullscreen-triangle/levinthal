/* =====================================================================
   shk.js -- the bridge to the real Shakespeare interpreter.

   The notebook does not reimplement the language.  It vendors
   cytochrome/src/helpers/shakespeare.js unmodified and drives it, so a
   play performed here and the same play performed in the cytochrome IDE
   move the same clock and produce the same verdict.

   Two things this module adds, and nothing else:

   1. A SESSION: one accumulating receiver for the whole page, matching
      the language's own commitment that M is a monotone intrinsic clock
      which never resets.  Re-performing a play is a NEW measurement at a
      higher M, never a cached retrieval -- so the notebook must not
      silently re-run plays behind the reader's back, and does not.

   2. LADDER VERBS.  The vendored interpreter predates the ladder
      construct, so `ladder` / `climb` / `observe ... as power` are
      handled here and everything else is delegated untouched.  The
      extension is additive in exactly the sense the paper claims: no
      existing line of the interpreter is altered, and a play with no
      ladder keyword takes no new code path.
   ===================================================================== */

import { newReceiver, performPlay } from './shakespeare.js';
import { LESSONS, RECEIVER_FLOOR } from './lessons.js';
import {
  composeMultiplicative, gapTrajectory, sensitivity, residualFraction,
  minRungsFor, staticReachable,
} from './engine.js';

export { LESSONS, RECEIVER_FLOOR };

// ---------------------------------------------------------------------
// The session receiver
// ---------------------------------------------------------------------
let RECV = newReceiver(RECEIVER_FLOOR);

export function receiver() { return RECV; }
export function resetReceiver() { RECV = newReceiver(RECEIVER_FLOOR); return RECV; }
export function clock() { return RECV.M; }

// ---------------------------------------------------------------------
// Ladder verbs, layered over the vendored interpreter
// ---------------------------------------------------------------------
const RE_LADDER = /^(\w+)\s*:=\s*ladder\s*\[([^\]]*)\]/;
const RE_CLIMB = /^climb\s+(\w+)\s+with\s+(\w+)(?:\s+reach\s+([0-9.]+))?/;
const RE_OBS_POWER = /^observe\s+(\w+)\s+as\s+power\s*$/;
const RE_DERIVED = /^(\w+)\s*:=\s*derived\s+(\w+)(?:\s+at\s+radius\s+(\d+))?/;

function parsePowers(inner) {
  return [...inner.matchAll(/power\s+([0-9]*\.?[0-9]+)/g)].map(m => parseFloat(m[1]));
}

function fmtFloor(f) { return f.toExponential(1); }

/**
 * Perform a play. Ladder lines are evaluated here; every other line is
 * passed to the vendored interpreter exactly as written.
 *
 * Returns the interpreter's own shape: { console: [{kind,text}], charts }.
 */
export function perform(src, lesson) {
  const ladders = {};          // ident -> powers[]
  const out = [];
  const charts = {};
  const say = (text, kind = 'log') => out.push({ kind, text });

  // Split the source into (a) lines the vendored interpreter should see
  // and (b) ladder lines this module owns. Order is preserved so the
  // console reads in program order.
  const lines = src.split('\n');
  const passthrough = [];
  const plan = [];             // [{own:boolean, line, idx}]

  for (const raw of lines) {
    const line = raw.replace(/--.*$/, '').trim();
    if (!line) { plan.push({ own: null, line: raw }); continue; }
    if (RE_LADDER.test(line) || RE_CLIMB.test(line) ||
        RE_OBS_POWER.test(line) || RE_DERIVED.test(line)) {
      plan.push({ own: true, line });
    } else {
      plan.push({ own: false, line });
      passthrough.push(raw);
    }
  }

  // Delegate first, so the receiver is in the state the ladder lines expect.
  let base = { console: [], charts: {} };
  if (passthrough.some(l => l.replace(/--.*$/, '').trim())) {
    base = performPlay(passthrough.join('\n'), RECV, lesson ?? { id: 'play', oracle: {} });
  }
  out.push(...base.console);
  Object.assign(charts, base.charts);

  // Then the ladder lines.
  for (const step of plan) {
    if (step.own !== true) continue;
    const line = step.line;
    let m;

    if ((m = line.match(RE_LADDER))) {
      const [, id, inner] = m;
      const ps = parsePowers(inner);
      if (!ps.length) {
        say(`${id} := ladder []   refused: a ladder declares at least one rung`, 'err');
        continue;
      }
      const bad = ps.find(p => p < 0 || p > 1);
      if (bad !== undefined) {
        say(`${id} := ladder [...]   refused: rung power ${bad} outside [0,1]`, 'err');
        continue;
      }
      ladders[id] = ps;
      RECV.bindings[id] = { type: 'Ladder', cls: 'lad', meta: { powers: ps } };
      // Forming a rung commits no cut: it declares a number, it does not
      // individuate anything. M must not move.
      say(`${id} := ladder [${ps.map(p => 'power ' + p).join(', ')}]` +
          ` : Ladder @ ${fmtFloor(RECV.floor)}   [M+0, free]`);
      continue;
    }

    if ((m = line.match(RE_OBS_POWER))) {
      const [, id] = m;
      const ps = ladders[id] ?? RECV.bindings[id]?.meta?.powers;
      if (!ps) { say(`observe ${id} as power   unbound`, 'err'); continue; }
      const c = composeMultiplicative(ps);
      say(`observe ${id} as power : Scalar @ ${fmtFloor(RECV.floor)}   [M+0, free]`);
      say(`  composite = ${c.toFixed(6)}  = 1 - ${ps.map(p => `(1-${p})`).join('')}`, 'ok');
      say(`  reads boundary already committed, so the clock does not advance`, 'dim');
      continue;
    }

    if ((m = line.match(RE_CLIMB))) {
      const [, region, lid, reach] = m;
      const ps = ladders[lid] ?? RECV.bindings[lid]?.meta?.powers;
      if (!ps) { say(`climb ${region} with ${lid}   unbound ladder`, 'err'); continue; }
      const target = reach !== undefined ? parseFloat(reach) : null;
      const composite = composeMultiplicative(ps);

      if (target !== null && !staticReachable(ps, target)) {
        // The refusal is decided statically and commits nothing.
        say(`climb ${region} with ${lid} reach ${target}`, 'cmd');
        say(`  REFUSED (subfloor): declared rungs reach ${composite.toFixed(6)}` +
            `, target ${target}`, 'err');
        say(`  shortfall ${(target - composite).toFixed(6)}   [M+0, nothing committed]`, 'err');
        say(`  the refusal is tight: executing this ladder would also fall short`, 'dim');
        charts['ladder-refused'] = { powers: ps, target, composite };
        continue;
      }

      const before = RECV.M;
      RECV.M += ps.length;                    // one commitment per rung
      const traj = gapTrajectory(ps, 1);
      say(`climb ${region} with ${lid}${target !== null ? ` reach ${target}` : ''}`, 'cmd');
      ps.forEach((p, i) => {
        say(`  rung ${i + 1}  power ${p.toFixed(4)}   gap ${traj[i].toFixed(4)}` +
            ` -> ${traj[i + 1].toFixed(4)}   [M+1]`);
      });
      say(`  composite ${composite.toFixed(6)}   residual ${residualFraction(ps).toFixed(6)}`,
          target !== null ? 'pass' : 'ok');
      say(`  M ${before} -> ${RECV.M}   (one commitment per rung; climbing is the` +
          ` only costed rule)`, 'clock');
      charts['ladder'] = { powers: ps, target, composite, traj, sens: sensitivity(ps) };
      continue;
    }

    if ((m = line.match(RE_DERIVED))) {
      const [, id, v, rad] = m;
      say(`${id} := derived ${v} at radius ${rad ?? 1}   [M+0, free]`, 'dim');
      say(`  the derivation reads edge weights already present; it commits no cut`, 'dim');
      continue;
    }
  }

  if (plan.some(s => s.own === true)) {
    say(`clock M = ${RECV.M}`, 'clock');
  }
  return { console: out, charts };
}

/** Run a play without touching the session receiver (for previews). */
export function performIsolated(src, floor = RECEIVER_FLOOR) {
  const saved = RECV;
  RECV = newReceiver(floor);
  try { return perform(src); } finally { RECV = saved; }
}

export const LADDER_HELP = {
  ladder: 'ident := ladder [power P, power P, ...]',
  climb: 'climb REGION with LADDER [reach TARGET]',
  observe: 'observe LADDER as power',
};
