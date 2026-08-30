# Catalytic Ladders — a runnable notebook

A demonstration web tool for the two ladder papers in `../docs/`. It looks like
a notebook, and it is one: every cell runs a real computation in the browser
and prints numbers and charts from it. Nothing on the page is a stored result
or an image of one.

## Running it

The page fetches `data/corpus.json`, so it needs to be served over HTTP rather
than opened from the filesystem:

```bash
cd enzymes/webtool
python -m http.server 8000
# then open http://localhost:8000/
```

Add `?all=1` to run every cell on load instead of just the first two.

## What is here

| File | What it is |
|---|---|
| `index.html` | the notebook: prose, cells, controls |
| `js/engine.js` | the kernel — a port of `docs/shakespear-protein-contact-sequence/validation/shk_core.py` |
| `js/charts.js` | D3 ports of the figure panels from both papers |
| `js/cells.js` | the executable content of each cell |
| `js/notebook.js` | the runtime: rendering, execution, controls, TOC |
| `css/notebook.css` | styling |
| `data/corpus.json` | a slice of the frozen KEGG + Reactome corpus |

## The engine is the real one

`engine.js` is a port, not a mock. It carries the same commitments as the
Python kernel:

- a rung has exactly **one** field, its power — no identity field, because a
  demonstration that identities do not matter, run on objects carrying names,
  would be a demonstration about the names;
- the floor is **computed** as the minimum over a finite edge set, never
  declared;
- the separation cost is an exhaustive minimum over subsets of a ball, so it is
  correct by construction rather than approximated;
- free rules (forming a rung, deriving one, reading a composite) leave the
  clock `M` untouched; climbing advances it once per rung;
- the verdict type rejects a payload that disagrees with its label, so a
  refusal cannot smuggle out a result.

Checked against the Python results: the worked ladder composes to `0.861400`
and the federated demo plan to `0.82675`, both matching the papers to the last
digit, and sensitivity is maximised at the highest-power rung.

## The corpus

`data/corpus.json` is a slice of the corpus described in the second paper: 400
KEGG enzyme records and 904 Reactome reactions, fetched once from
`rest.kegg.jp` and `reactome.org/ContentService` and then frozen. The full
snapshot lives in
`../docs/catalytic-ladders-federated-querying/validation/fixtures/sources.json`.

The data is frozen rather than fetched live because the host system forbids
network access in its adapters by construction — its claims are properties of
the compiler, and a live service can neither confirm nor refute them. The
notebook keeps that discipline.

## What the notebook does not claim

The powers derived here come from a contact graph, which is itself a model.
Whether they correspond to rate constants measured in a laboratory is an
empirical question no cell bears on. The corpus is real data used so the plans
have something real to run on, not evidence about metabolism.

## Testing

`js/engine.js` runs under Node without a DOM, so the kernel can be checked
directly. A DOM-level harness that drives every cell and reports what each
produced is in the session scratchpad (`nbtest.js`); it needs `jsdom` and `d3`
installed locally. Note that `d3.forceSimulation` overflows the stack under
jsdom, which is why the contact-graph layout is deterministic rather than
force-directed — that also makes the picture reproducible across reloads.
