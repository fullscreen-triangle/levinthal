# levinthal

## Symbol lookup: use `purpose` before Grep

This repo has a prebuilt `purpose` index (**8267 symbols**, `.purpose/index.json`,
gitignored). Build/refresh it with `purpose index` — that takes **~2m20s here**, so
re-index deliberately, not reflexively.

```bash
purpose ask "<symbol-stem>"    # file:line [kind] name + snippet, capped at 20
```

### Reach for it when

**"Where is `<X>` defined?" and the answer may span languages or subtrees.**

This repo is polyglot and heavily duplicated — Rust crates (`crates/`), Python
(`cytochrome/glb/`, `experiments/`, per-paper `validation/scripts/`), and LaTeX prose
(`cytochrome/publications/`, `models/sources/`). `purpose` indexes definitions *and*
`.tex`/`.md` headings in one pass, so it unifies slices a single Grep would miss.
Measured here:

| Query | purpose | Grep |
|---|---|---|
| `PartitionState` | **2390 B**, found 4 impls (1 Rust + 3 Python) | 6652 B, `--include=*.rs` missed the Python ones |
| `contact floor` | one slice spanning `.tex` prose + `rbio.py`/`structure.py` defs | needs 2+ passes |

That LaTeX-plus-code unification is the main reason it pays off in this repo
specifically — concepts live in prose and implementation at once.

### Do not reach for it when

Output is **unranked and capped at 20**, so a loose query returns a fixed wall of noise
and a targeted Grep is cheaper. Skip it for known-syntax lookups, call sites, imports,
config values, and string literals — none of those are indexed.

### Query shape

Case-insensitive **substring filter over symbol names**, OR-ed across whitespace terms.
Not semantic search.

- Query **one stem**, never a question — extra words *widen* the set and push the answer
  past the cap.
- Query the stem, not the concept noun; on a miss, **shorten** before giving up.
- **Filenames and paths are not indexed** — use Glob for those.

**A miss proves nothing.** Never report that something does not exist based on
`purpose ask` alone; fall back to Grep/Glob first.
