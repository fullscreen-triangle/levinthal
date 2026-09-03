# Catalytic Ladder Propagation — interactive companion

An explorable explanation of the framework, written for chemists rather than
for logicians. Every number on the page is **computed in the browser** from the
same definitions the paper states; nothing is a stored screenshot of a result.

## Run locally

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
npm run preview
```

## Deploy to Vercel

The repository root for the project is this `web/` directory.

```bash
npx vercel        # preview
npx vercel --prod # production
```

`vercel.json` already declares the Vite framework preset, the build command and
`dist` as the output directory, so no dashboard configuration is needed. If you
import the repository through the Vercel UI instead, set **Root Directory** to
`nfdi4cat/catalytic-ladder-propagation/web`.

## What is on the page

| Section | What you can do |
|---|---|
| The problem | The five-situations table: one empty answer, five different causes |
| Contact & floor | Drag core packing weight; watch identity become a *region*, and the floor detach from the smallest edge |
| The ladder | Build a chain, move rung powers, toggle additive vs proportional sensitivity, test deletions |
| The medium | Switch weight families; compute solvent role; reach all three direction verdicts; slide the saturation asymmetry |
| What SPARQL cannot say | Two systems, identical triples, opposite verdicts — with the 13-query battery and its control |
| Plans & verdicts | Build a step and watch capability containment refuse it before any request is issued |
| The 28 questions | Run each one; open a card for its plan, sources, and blocker/unblock |
| Rhea, 2 vs 397 | The live divergence replicated 3 September 2026, with the local control |
| What this does not do | The limitations, stated plainly |

## Structure

```
src/
  lib/kernel.js      port of validation/kernel/ladder.py — cuts, floor,
                     ladder algebra, medium, verdicts (non-degeneracy enforced)
  lib/corpus.js      the fixture corpus, same record shape as the sources
  lib/questions.js   the 28 questions, each with an executable run()
  components/        one component per interactive figure
  App.jsx            the narrative
```

## A note on the data

The fixture corpus is a set of small hand-checkable stand-ins with the same
record *shape* as the public resources — not copies of them. This makes every
result reproducible offline and makes none of them evidence about biology. The
only live measurement quoted is the Rhea divergence, which is dated because
endpoints change.
