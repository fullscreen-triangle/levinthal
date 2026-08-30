# Catalytic Ladders — runnable notebook

A Vite site that presents the whole chain in one page: from *why do enzymes
exist?* through the process ladder to Shakespeare, then out to federated
querying. Every number is computed in the browser at run time; nothing is a
stored result or an image of one.

## Run it

```bash
cd enzymes/web
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/
npm run preview    # serve the production build
```

## Deploy to Vercel

`vercel.json` is committed and the project is a stock Vite build, so:

```bash
npx vercel          # preview deploy
npx vercel --prod   # production
```

Or point Vercel at this directory in the dashboard — it detects Vite, runs
`npm run build`, and serves `dist/`. No environment variables, no server
functions, no runtime dependencies: the corpus is a static JSON in `public/`.

## The Shakespeare interpreter is the real one

`src/lib/shakespeare.js`, `src/lib/lessons.js` and `src/lib/glbMarkers.js` are
**vendored unmodified** from `cytochrome/src/` — only the import paths changed,
and each file carries a banner saying so. This is the interpreter behind the
cytochrome IDE at `/ide`, not a reimplementation, so a play performed here and
the same play performed there move the same clock and produce the same verdict.

`src/lib/shk.js` is the only new code in that path. It adds exactly two things:

1. **A session receiver.** One accumulating receiver for the whole page, because
   the language's clock `M` is monotone and never resets. Re-running a play is a
   new measurement at a higher clock, never a cached retrieval — so the page
   never silently re-runs plays behind the reader.

2. **Ladder verbs.** `ladder` / `climb` / `observe … as power`, layered on top.
   Everything else is delegated to the vendored interpreter untouched. The
   extension is additive in the sense the paper claims: no existing line is
   altered, and a play with no `ladder` keyword takes no new code path.

Verified live: forming a ladder and observing its power leave `M` alone;
climbing advances it once per rung; a ladder that cannot reach its declared
target is refused before evaluation and commits nothing.

```
L := ladder [power 0.45, power 0.30, power 0.55]   [M+0, free]
observe L as power    composite = 0.826750          [M+0, free]
climb heme with L reach 0.70                        M 2 -> 5
climb heme with L reach 0.95    REFUSED (subfloor), shortfall 0.123250, M unchanged
```

## Layout

| Path | What it is |
|---|---|
| `index.html` | the stage (five-link chain) and the notebook prose |
| `src/main.js` | runtime: cells, controls, rail, highlighting |
| `src/cells.js` | the eleven computed cells |
| `src/shkcells.js` | the two cells that run real `.shk` plays |
| `src/lib/shk.js` | session receiver + ladder verbs |
| `src/lib/shakespeare.js` | **vendored** interpreter |
| `src/lib/lessons.js` | **vendored** 11 monograph plays with baked oracles |
| `src/lib/engine.js` | ladder kernel, a port of `shk_core.py` |
| `src/lib/charts.js` | D3 ports of the paper panels |
| `public/data/corpus.json` | frozen KEGG + Reactome slice |

## Checked against the papers

The kernel reproduces the published numbers exactly: the worked ladder composes
to `0.861400`, the federated demo plan to `0.82675`, sensitivity is maximised at
the highest-power rung, and the refusal names a shortfall of `0.123250`.

## What this does not claim

The powers derived here come from a contact graph, which is itself a model.
Whether they correspond to rate constants measured in a laboratory is an
empirical question no cell bears on. The corpus is real data used so the plans
have something real to run on, not evidence about metabolism.

## The older static version

`enzymes/webtool/` is the pre-Vite version — plain files, no build step, opened
over `python -m http.server`. It has no Shakespeare interpreter. This directory
supersedes it.
