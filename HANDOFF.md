# Handoff — combos of any size (hypergraph search), updated 2026-09-05

Goal set by the user: a program that searches for the maximum density of MTG
infinite combos extremely efficiently on the tt-quietbox (4 × Blackhole).
Current instance: **unrestricted pool** (any infinite result, any
prerequisites), **50 cards**, Kenrith, the Returned King as commander, combos
of **any size** counted once per unique card set.

## State of play

| item | status |
|---|---|
| hypergraph build (`build --card-count 0` → `data/hypergraph/`) | done: 93,934 unique combos, 6,515 cards, sizes 2–10 |
| CPU tabu (`hyper.py`) | done; **1,745** (`results/any-size/cpu-tabu-1745.txt`), other attractor 1,692 |
| NumPy reference of the sparse population step (`hyperpop.py`) | done, tested (`tests/test_hyper.py`) |
| TT backend (`tt_hyper.py`) | done: bit-for-bit exact vs reference (fused tile gather on a 2048-aligned slot width), trace replay, device-side reseed/recount, 1 × 4 mesh |
| driver / CLI | done: `search --graph data/hypergraph --backend tt|cpu|numpy …` (`driver.run_hyper`) |
| mesh | done and measured: 97.7k swaps/s at P = 4,096 on 4 cards, 85k end to end |
| TT result | 1,745 from random starts in 44 generations (≈ 2 s device time); nothing above 1,745 in 8.6 M swaps |
| README / dev log | updated (2026-09-05 entry has the throughput and time-to-1,745 tables and the assessment) |
| `ttnn.rand` write-up for the devops team | `notes/ttnn-rand-for-scientific-compute.md` (+ `scripts/rand_probe.py`, `rand_analysis.py`) |
| pushing | pushed to `origin/claude/mtg-commander-combo-optimizer-cx4ukq` on 2026-09-05 (commit with `-c user.name=ttuser -c user.email=…`; no git identity is configured on the box) |

## Conclusions the user has been told

- For this instance the CPU is sufficient: one tabu descent reaches 1,692 or
  1,745 in < 1 s on the min-degree-16 pool. The device search finds 1,745 in
  ~2 s of device time but pays ~20 s of per-process setup (device open 5.5 s,
  constants upload 4.2 s → now ~2 s, first-execution program construction
  10 s, JSON load 3 s). The device move is weaker by design (best add, then
  best drop given the add) than the CPU's full pair scan; the top-t-adds
  variant would recover it at ~t× cost.
- **Speed verdict (user goal, 2026-09-05): the device search is not worthwhile for this
  application.** 44–48 generations to 1,745 in every run, but 14–26 s wall against the CPU's
  0.7–2.6 s; the losing part is per-process setup that a sub-second problem cannot amortise.
  Tracing is not the cause (eager is 1.8× slower). Details: dev log "Speed verdict".
- No proof of optimality for 1,745; MILP deliberately not attempted.
- The Kenrith artifact now has a "Combos of any size" entry with the 1,745 deck
  (`scripts/add_any_size_to_artifact.py`, `scripts/fetch_card_images.py`).

## If work continues, in order

1. **Search quality only if the user wants to look past 1,745**: top-t adds
   per replica on the device (t gathers + t drop evaluations), or the CPU
   tabu with larger kicks / path relinking between the 1,692 and 1,745 decks.
2. **Setup time**: cache the parsed `HyperInstance` (npz) and keep a resident
   process if many searches are run; the 10 s first-execution cost is
   tt-metal program construction and only disappears in a resident process.
3. **Throughput** (if ever needed): E-space state with two gathers (~1.5×
   fewer bytes per pass), bfp8_b slot tensors if an exact `eqz` exists, tracy
   profile of one traced generation.

## How to run what exists

```
.venv/bin/python -m pytest                                   # 24 tests
.venv/bin/python -m spellbook_graph search --graph data/hypergraph --backend cpu --seconds 300
export TT_METAL_HOME=/home/ttuser/sjameel/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME/ttnn:$TT_METAL_HOME
$TT_METAL_HOME/python_env/bin/python -m spellbook_graph search --graph data/hypergraph --backend tt \
    --devices 4 --population 4096 --min-degree 16 --epoch-gens 100 --seconds 600 --out data/deck_any_tt
$TT_METAL_HOME/python_env/bin/python scripts/check_tt_hyper.py 4096 1 32 3 hifi2 bfloat16 0   # eager vs NumPy, bit-for-bit
$TT_METAL_HOME/python_env/bin/python scripts/check_trace.py 512 4 16                          # mesh trace replay == eager
$TT_METAL_HOME/python_env/bin/python scripts/check_reseed.py 1024 1 16 0                      # device-side reseed vs host
.venv/bin/python scripts/cpu_time_to_best.py 240 8 16                                         # CPU time to first 1,745
```
