# Handoff — combos of any size (hypergraph search), 2026-09-04

Goal set by the user: a program that searches for the maximum density of MTG
infinite combos extremely efficiently on the tt-quietbox (4 × Blackhole).
Current instance: **unrestricted pool** (any infinite result, any
prerequisites), **50 cards**, Kenrith, the Returned King as commander, combos
of **any size** counted once per unique card set.

## State of play

| item | status |
|---|---|
| hypergraph build (`build --card-count 0` → `data/hypergraph/`) | done: 93,934 unique combos, 6,515 cards, sizes 2–10 (mostly 3 and 4), 335,789 incidences |
| CPU tabu (`spellbook_graph/hyper.py`) | done, exact swap deltas verified; 28 workers × 300 s → **1,745** (`data/deck_any_cpu/`), other attractor 1,692 (`data/deck_any_cpu_1692/`) |
| plain-text lists | `results/any-size/cpu-tabu-1745.txt`, `cpu-tabu-1692.txt` (committed locally, **not pushed** — user will add GitHub auth) |
| artifact with both lists | https://claude.ai/code/artifact/fc02b150-4fcd-480a-ac98-9afd7b750862 |
| MILP bound | **deliberately not run** (user: skip if it won't work; hubs with 5,713 combos make the LP hopeless) |
| NumPy reference of the sparse population step (`hyperpop.py`) | done, matches brute force on a 20-card test; incremental score/cnt verified |
| TT backend (`tt_hyper.py`) | done and **bit-for-bit exact** vs the reference (host noise), eager mode, P=64…4096, 1 device |
| trace capture/replay | done: replay == eager exactly; 14.2 ms/generation at P=256 on one card (18k swaps/s) |
| mesh (4 devices) | code path written (shard on dim 1, salt per device) but **not yet tested** |
| driver / CLI for the hypergraph search | **not written** (`driver.run` and `cli.py search` still only know the two-card `Instance`) |
| README / dev log | dev log has today's entry below; README not yet updated for any-size combos |

## Numbers worth remembering

- Pool after pruning cards with < 8 combos (`min_degree=8`): 3,494 cards, 85,506 combos, 361,888 padded slots; `min_degree=16`: 2,723 cards, 321,376 slots. The 1,745 deck's weakest card has 64 in-deck combos and only 65 in the pool, the 1,692 deck uses a card with 44 → keep `min_degree ≤ 32`.
- Eager TT step, one device: 72 ms/gen at P=1024 (14.1k swaps/s), 237 ms at P=4096 (17.3k/s). About 12 passes over the (I × P) slot tensor at ~3.5–4.7 ms each per 1024 replicas, plus ~45 small ops at 0.5–0.8 ms dispatch each. The reduction matmul is not the bottleneck (LoFi/bfp4 gives the same time).
- Trace replay removes the dispatch cost: 14.2 ms/gen at P=256. Not yet measured at P=1024–4096 (expect ~50 ms and ~200 ms, i.e. ~20k swaps/s per device, ~80k/s on four).
- Kicked CPU tabu converges to 1,692 or 1,745 within seconds and never leaves those basins (kick = drop 6 weakest + greedy refill is too gentle for this landscape).

## Device findings (add to the dev log's "bugs found" list)

- `ttnn.embedding(..., layout=TILE_LAYOUT)` returns wrong rows once a weight row exceeds 512 KB (exact at width 262,144 bf16, wrong at 361,888: first bad column 2049, 3.7 % of elements). Row-major output is exact at every width → gather row-major, then `to_layout`.
- `ttnn.sum` on bfloat16 input rounds the *output* to bf16 (1,023 ones → 1,024); float32 input is exact. Hence the two-level reduction (32-slot tile-row sum ≤ 32, exact; then 0/1 matmul with fp32 accumulation, float32 out).
- `ttnn.reshape` of a tiled (P, I) tensor to (P/32, chunks, 32, 32) is a *semantic* reshape with a 59 ms copy, not a tile view — no free way to drop the transposes.
- Row-major elementwise ops are ~3× slower than tiled (add 15.8 vs 4.65 ms) — keep everything tiled (user also asked for this).
- bfp8_b: add 3.1 ms / transpose 2.05 ms (faster than bf16) but `eq` 9.2 ms — not adopted.
- Trace capture records without executing (state unchanged after `end_trace_capture`); `ttnn.rand` inside a trace replays identical numbers → noise is `frac(rand + salt)` with a Weyl-sequence salt advanced in place each generation.
- `ttnn.subtract(scalar, tensor)` still not accepted (tensor-first only).

## What to do next, in order

1. **Driver + CLI.** Add `run_hyper` to `driver.py` (mirror `run`: epochs, progress line with swaps/s, incumbent written via `hyper.write_result`, independent rescoring of the incumbent with `HyperInstance.score` every time it improves) and make `cli.py search` dispatch on `hypergraph.json` (backends cpu / numpy / tt; `--min-degree`, `--gens-per-trace`, `--no-trace`). Epoch length must be a multiple of `gens_per_trace`.
2. **Mesh test.** Run `scratchpad/check_trace.py`-style verification with `devices=4` (P multiple of 128). Watch for: `copy_host_to_device_tensor` with a sharded host tensor, `ttnn.slice` of the salt on a mesh, `MeshTraceId`.
3. **Throughput at scale.** Measure trace replay at P=1024/2048/4096 per device; pick P that maximises swaps/s while leaving DRAM headroom (P=4096 uses ~12–15 GB of 32 GB). Then a 20-minute run for the record, e.g. `--population 16384 --devices 4 --epoch-gens 100 --min-degree 16`.
4. **Search quality.** The two CPU attractors suggest a rugged landscape: reseed stale replicas with a mix of random decks and *large* kicks (15–25 cards) of the incumbent, and consider seeding a fraction of replicas from the two CPU decks. Report the best TT result against 1,745.
5. **If more speed is needed** (in this order): (a) `min_degree` 16–24 to cut slots; (b) E-space state with two gathers (see dev log, ~1.5× fewer bytes); (c) bfp8_b for the slot tensors if a fast `eqz` exists; (d) profile one traced generation with `python -m tracy -r -p` from `$TT_METAL_HOME` (ENABLE_TRACY=ON in this build; `tt-perf-report` is installed in the project `.venv`) — user: only if stuck.
6. **Write-up.** Update README (any-size section: objective, hypergraph, incidence-space formulation, results table), dev log, and the artifact (add the TT deck as a third tab). Then push once GitHub auth is on the box.

## How to run what exists

```
# hypergraph
.venv/bin/python -m spellbook_graph build --card-count 0 --out data/hypergraph
# CPU tabu (28 workers, 300 s, full pool) — scratch script for now
.venv/bin/python <scratchpad>/cpu_run.py 300 28 data/deck_any_cpu [min_degree]
# TT checks (tt-metal interpreter, read-only checkout)
export TT_METAL_HOME=/home/ttuser/sjameel/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME/ttnn:$TT_METAL_HOME
$TT_METAL_HOME/python_env/bin/python <scratchpad>/check_tt_hyper.py P devices min_degree gens [fidelity] [s_dtype] [device_id]   # eager vs NumPy, bit-for-bit
$TT_METAL_HOME/python_env/bin/python <scratchpad>/check_trace.py P devices min_degree [device_id]                                  # trace vs eager, throughput
```
Scratch scripts live in `/tmp/claude-1000/-home-ttuser/06da2477-9484-4ec8-9307-e791f317b106/scratchpad/`
(`scan.py`, `cpu_run.py`, `check_tt_hyper.py`, `check_trace.py`, `debug_tt.py`, `bench_*.py`,
`build_lists_artifact.py`, `write_txt.py`). Copy anything worth keeping into the repo before that directory is lost.
