# mtg-combos-subgraph

Find the Commander deck whose 50 combo-piece slots contain the most unique
two-card infinite combos.

Step 1 (this commit): build the graph. Step 2 (next): search it.

## Data source

Combo data comes from [Commander Spellbook](https://commanderspellbook.com/).
Their backend is MIT licensed ([SpaceCowMedia/commander-spellbook-backend](https://github.com/SpaceCowMedia/commander-spellbook-backend))
and its API docs ask tools **not** to page through the HTTP API. Instead they
publish the whole dataset as one JSON document, refreshed periodically:

| File | Notes |
|------|-------|
| <https://json.commanderspellbook.com/variants.json.gz> | gzipped, the one they ask you to use |
| <https://json.commanderspellbook.com/variants.json> | uncompressed |

The document is `{"timestamp", "version", "variants": [...], "aliases": [...]}`
where each variant has the same shape as a `/variants/` API response.
Please credit Commander Spellbook in anything derived from this.

```
python -m spellbook_graph download          # one conditional GET, cached in data/
# or by hand:
curl -L -A "mtg-combos-subgraph/0.1" -o data/variants.json.gz https://json.commanderspellbook.com/variants.json.gz
```

## Install and build

```
pip install -e ".[dev]"
python -m pytest
python -m spellbook_graph build             # reads data/variants.json.gz, writes data/graph/
python -m spellbook_graph stats
```

The bulk file is parsed as a stream (`ijson`), so memory stays flat. A
synthetic 300k-variant file (about 1.4 GB uncompressed) builds in 25 s on one
core.

## What counts as an edge

A variant becomes an edge between its two cards when all of these hold
(each is a CLI flag on `build`):

| Rule | Default | Flag |
|------|---------|------|
| `status` is `OK` (`E` = example, text hidden; `NR`/`D` are not public) | OK only | `--include-examples` |
| `legalities.commander` is true | required | `--ignore-legality` |
| not a spoiler (unreleased card) | excluded | `--include-spoilers` |
| no template requirement (`requires` empty) | required | none |
| exactly two distinct cards, quantity 1 each | required | `--card-count N` |
| produces a feature named `Infinite ...` | required | `--allow-near-infinite`, `--require-standalone` |

"Infinite" is matched on the produced feature's name (`^infinite\b`,
case-insensitive), the same convention the site's `result:` search uses.
`Near-infinite ...` results are excluded by default because the user asked
for infinite combos. Feature status is recorded; the backend's own bracket
estimator calls a combo "relevant" only if it produces a standalone (`S`)
feature, which `--require-standalone` replicates.

Multiple variants of the same pair (different results, different generator
combos) collapse into one edge. The objective is unique card pairs, so the
edge count is the deck score; the variant count is kept as `n_variants`.

### Commander-only edges

Some combos require one of the two cards to be your commander (`mustBeCommander`,
or the card's only starting zone is the command zone). Those edges carry
`unconditional: false` and `commander_required: [oracle ids]`. An edge is
unconditional if at least one of its variants has no such requirement. The
optimiser should count a conditional edge only when the deck's commander is
one of the listed cards. The default deck score should use unconditional
edges plus whatever a chosen commander unlocks.

### Color identity

The export carries identity per variant, not per card. Each node stores
`identity_upper_bound`, the intersection of the identities of every variant
it appears in. With a five-colour commander there is no identity constraint,
which is the natural setting for "most combos in 50 slots". If a narrower
commander is wanted later, exact per-card identity can be joined in from
Scryfall's oracle-cards bulk file by `oracle_id`.

## Output format (`data/graph/`)

- `graph.json`: `nodes` (index, oracle_id, name, spellbook_id, type_line,
  identity_upper_bound, n_variants) and `edges` (u, v, n_variants,
  variant_ids, unconditional, commander_required, identity,
  min_mana_value_needed, bracket_tags, features, max_popularity), plus the
  filter config, rejection counts and the source `timestamp`/`version`.
- `edges.tsv`: flat edge list with card names, for eyeballing.
- `graph.npz`: `u`, `v`, `unconditional`, `n_variants`, `n_nodes` arrays for the optimiser.

## Step 2: the search

Because every combo here has exactly two cards, this is a plain graph, not a
hypergraph: choose 50 vertices maximising induced edges — densest-k-subgraph
(NP-hard, no PTAS known). Three backends share one objective:

```
python -m spellbook_graph search --backend cpu --seconds 120           # per-core tabu search (NumPy)
python -m spellbook_graph search --backend numpy --population 256      # population search, CPU reference
python -m spellbook_graph search --backend tt --devices 4 --population 65536 --seconds 1200
```

Common flags: `--commander "Kenrith, the Returned King"` (default; use `none`
for no commander), `-k 50`, `--out data/deck`. The result is written to
`deck.json` (cards, per-card combo counts, the combo list, search history) and
`deck.txt` (a plain decklist) every time the incumbent improves, so an
interrupted run still leaves its best deck behind.

### Commander handling

The commander sits in the command zone, not in the 50 slots, so its combos are
free: each card carries a `bonus` = number of combos it has with the commander,
and the objective is induced edges + bonus of the chosen cards. Combos that
need a *different* card to be the commander are dropped. Kenrith is five
colours, so no identity filter is applied.

### CPU backend (`search.py`)

GRASP construction + best-swap tabu search + kicks, restarted for the time
budget, one process per core. O(k·n) per iteration.

### Population backend (`population.py`, `tt_search.py`)

A population of P decks takes one swap per *generation*, with the whole
neighbourhood evaluated by matmul: `C = X @ A` gives every deck's combo count
for every card; the best add is chosen with exact `max` reductions and random
tie-breaking, the best drop given that add uses a second one-hot matmul for
the row `A[b*, :]`. Per generation O(P·n²) flops in two matmuls plus O(P·n)
elementwise. `population.py` is the NumPy reference; `tt_search.py` runs the
identical step with ttnn on one Blackhole or a 1×N mesh and is checked
bit-for-bit against the reference. Progress is printed once per epoch.

Running the TT backend needs a tt-metal build with `ttnn` and torch:

```
export TT_METAL_HOME=/home/ttuser/sjameel/tt-metal      # read-only use of an existing build
export PYTHONPATH=$PWD:$TT_METAL_HOME/ttnn:$TT_METAL_HOME
$TT_METAL_HOME/python_env/bin/python -m spellbook_graph search --backend tt ...
```

Measured on the quietbox (4 × Blackhole): ~25 ms per generation up to
P ≈ 16k per device, 2.3M swaps/s at P = 65,536 on four cards, ~1.5–2M
end to end including host bookkeeping. Details, numerical-exactness findings
and the bugs met along the way are in `development-log.md`.

### Result

All three backends converge on **243 unique two-card infinite combos** among
50 cards (none of them with Kenrith himself) from independent random starts,
and a MILP (HiGHS via `scipy.optimize.milp`, `scratchpad/milp_bound.py` in
the dev log) proves 243 optimal in about 5 seconds — the instance is sparse
enough that the LP relaxation is tight. The optimum is not unique (two decks
differing in one card both score 243). Decks: `data/deck_tt/deck.txt`,
`data/deck_milp/deck.txt`.
