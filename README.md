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

## Next step: the search

Because every combo here has exactly two cards, this is a plain graph, not a
hypergraph: choose 50 vertices maximising induced edges. That is the
densest-k-subgraph problem (NP-hard, no PTAS known), but the instance is small
enough for strong heuristics: greedy peeling from the top-degree cards,
then local search / simulated annealing over swap moves with incremental
score updates, restarted many times. That runs in minutes on CPU; the
quietbox is more than enough. Since the graph is dominated by a few hub cards
(untappers, token doublers, damage/drain enablers) the interesting question is
which hubs to *drop* to make room for their partners, which is exactly what
swap-based local search explores.
