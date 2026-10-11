# levinthal

## Symbol lookup: use `purpose` before Grep

This repo has a prebuilt `purpose` index (**9509 symbols** as of 2026-10-08,
`.purpose/index.json`, gitignored). Refresh it with `purpose index` — **~20 s** here
since the indexer fix, so re-index whenever recent papers are missing from results.

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

## Analysis tools

Six local tools are installed and were tested against this repo on 2026-10-08. Their
usage docs are in [tools/](tools/); okgg's are at
`../../greifswald/conspirator/conspirator/okgg{,-reader}/README.md`. Read-only
subcommands are pre-allowed in [.claude/settings.json](.claude/settings.json);
anything that serves a port or reaches remote machines prompts.

### Which tool answers which question

| Question | Tool |
|---|---|
| Where is `X` defined? | `purpose ask "X"` (above) |
| Which passage *argues* this, in words I don't know? | `spraypaint ask "<phrase>" --scenes …` |
| Which papers/modules is this goal about; which are load-bearing? | `purpose ckg ask "<goal>"` |
| What distinguishes these files/sections from each other? | `okgg run <dir or .tex> --unit section` |
| Do these Python/Rust units coordinate, or just pass local tests? | `wt static <dir>` |

`smith` (agent-script DSL) and `srn-fleet` (SSH compute market) work but have no use
in this repo's research; don't reach for them unless asked.

### spraypaint — passage search over the LaTeX

Index: `spraypaint index` (~17 s, 1257 docs, ~110 MB, gitignored). Version is 0.2.0, so
`verify` exits 2 on N/A checks and `ask` prints a coverage verdict (`partial` names the
terms no returned passage holds).

⚠️ **Every committed `ask` must pass `--scenes`, and never `.claude` or `(root)`.** The
index includes `.claude/settings.local.json`, which holds a gateway credential, and
`ask` prints passage text. A PreToolUse hook
([.claude/hooks/guard_tools.py](.claude/hooks/guard_tools.py)) denies unscoped asks.
`--dry-run` needs no scenes and commits nothing — use it first. Scenes are top-level
dirs: `cytochrome enzymes models docs publication nfdi4cat experiments crates
shakespear dismutase`. For theory, `--scenes models,cytochrome,enzymes,docs` is the
usual choice; `models/sources/*.tex` holds the canonical definitions.

### purpose ckg — load-bearing papers for a goal

Lens is committed at [.purpose/lens.toml](.purpose/lens.toml); **read its header
before editing** — it records what was tried and why. One module = one paper
directory. Build after any lens edit: `purpose ckg build` (~10 s).

- `ckg ask "<goal>"` takes **~2 min** — run it in the background. It works: "contact
  floor" → 19 necessary of 117 reachable, matching the foundational-paper chain.
- Judge by goal saturation (< 50%), never by components (one giant component is a
  property of this single-vocabulary corpus) or β* (1.00 throughout).
- Common words like `transfer`, `trajectory`, `state`, `partition` are stopwords, so
  phrase goals by their rarer word. A DECLINED verdict means the goal's words are not
  in τ — check `purpose ckg lens --goal "<goal>"`.
- `ckg why <paper-dir>` degenerates to "every edge to the medium"; it's only useful
  for code dirs under `crates/` and `experiments/`.

### okgg — what tells files apart

- `--generator lexical` is instant but its facets are single words (`state`,
  `package`). `--generator ollama --model llama3.2 --limit 30` names real facets
  (`protein-folding` vs `mass-spectrum`) but takes **~110 s per step**; run it in the
  background with a small `--budget`.
- `--unit section` on a single `.tex` makes each section an entity — useful for
  checking whether a paper's sections actually draw distinct distinctions.
- Write output outside the repo (`--out <scratch>`) for exploratory runs; in-tree
  `.okgg/` is gitignored but clutters the tree.
- `okgg-reader` (`node <okgg-reader>/dist/cli.js <dir>/.okgg --receiver <id>`) is an
  interactive tutor that asks a *person* to sort files. It is for the user to run,
  not for Claude.

### wt — coordination of the code

`wt static crates` (Rust) and `wt static experiments` (Python) both run in < 1 s. On
2026-10-08 both read **Aperture-dominated** (R_est 0.36 / 0.44): the code units share
interfaces but no common purpose, which is expected for a collection of independent
validation scripts. It writes `.wt/` into the analysed dir (gitignored).
