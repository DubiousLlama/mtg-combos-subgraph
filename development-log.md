# Development log

Working notes for the "most two-card infinite combos in 50 slots" search, kept
as the work happens so a later write-up on the Tenstorrent side has the record.
Newest entries at the bottom.

## 2026-09-04 — problem statement and graph (step 1)

Goal: a Commander deck whose 50 combo-piece slots contain the most *unique*
two-card infinite combos. Commander: **Kenrith, the Returned King** — five
colour identity (no colour filter needed) and any infinite coloured mana is a
win through his activated abilities, so every "Infinite ..." result counts.

Data: Commander Spellbook's bulk export (`https://json.commanderspellbook.com/variants.json.gz`,
MIT-licensed backend, they ask tools to use the bulk file instead of paging the
API). Graph builder is `spellbook_graph/` (see README for filter rules).

Fresh build today:

| | |
|---|---|
| variants scanned | 108,918 |
| accepted (2 cards, OK, commander-legal, no template, produces `Infinite …`) | 2,774 |
| rejected: 3+ cards or other counts | 97,179 |
| rejected: needs template card | 5,137 |
| rejected: not commander legal | 1,520 |
| rejected: no infinite result | 1,122 |
| rejected: spoiler | 1,186 |
| cards (nodes) | 1,630 |
| edges (unique card pairs) | 2,774 (2,766 unconditional, 8 require a specific commander) |

With Kenrith as commander: 1,623 candidate cards, 2,765 usable combos. Kenrith
himself has one combo in the data (with Composite Golem); it is a free +1 for
that card since the commander does not take a slot. The 8 commander-only
edges all need other commanders (Nevinyrral, Dargo, Estrid, Prossh, …) and are dropped.

Structure: one giant connected component (1,442 cards, 2,600 edges), then a
handful of tiny ones (largest 17 cards / 28 edges). Degree distribution is
heavy-tailed: 106 cards with degree ≥ 10, 790 with degree 1. Top hubs:
Mirror-Mad Phantasm 86, Kiki-Jiki 78, Intruder Alarm 76, Naru Meha 58,
Dualcaster Mage 58, Aggravated Assault 54. Sum of the top-50 degrees is 1,450,
so 725 is a trivial upper bound on the answer.

## 2026-09-04 — literature check (densest-k-subgraph)

Every combo here has exactly two cards, so the problem is exactly
**densest-k-subgraph (DkS)**: choose k = 50 vertices maximising induced edges.
NP-hard, no PTAS known. What the literature says about instances like ours:

- Exact methods (SDP/convex branch and bound) top out around 160 vertices —
  [Bombardieri et al., "On solving the densest k-subgraph problem on large graphs"](https://arxiv.org/pdf/1901.06344)
  (also [journal version](https://www.tandfonline.com/doi/full/10.1080/10556788.2019.1595620)).
  Our 1,623-vertex instance is out of reach for a proof of optimality; the user
  accepted a non-provable answer.
- For heuristics, Brimberg et al. compared greedy add/drop, VNS, tabu and
  multi-start local search and found VNS (kicked local search) best; Kincaid
  found tabu beats simulated annealing on DkS. Both are swap-neighbourhood
  methods: remove one vertex, add one, evaluated incrementally. That is what
  we implement: tabu search over swaps plus VNS-style kicks from the incumbent.
- Sparse-graph view: [Komusiewicz & Sorge, "Finding Dense Subgraphs of Sparse Graphs"](https://fpt.akt.tu-berlin.de/publications/Finding_Dense_Subgraphs_of_Sparse_Graphs-IPEC2012.pdf)
  and the [survey by Lanciano et al.](https://arxiv.org/pdf/2303.14467) for the wider family.

Swap-move algebra (used by every backend): with `cont[v]` = number of chosen
cards that combo with v and `bonus[v]` = combos with the commander,
`delta(drop a, add b) = (cont[b]+bonus[b]) - (cont[a]+bonus[a]) - A[a,b]`.

## 2026-09-04 — CPU baseline (`spellbook_graph/search.py`)

Multi-start GRASP + best-swap tabu search + kicks, NumPy, one process per core.
Per iteration it forms the full k × (n−k) delta matrix: O(k·n) ≈ 8·10^4
element-ops, about 0.3 ms in NumPy. 4 workers × 20 s reached **243 combos**;
all restarts converge to the same value, which is a hint (not proof) that 243
is the optimum or very close. The best deck is two overlapping clusters:
untappers + mana producers (Staff of Domination, Sword of the Paruns, Umbral
Mantle, Pemmin's Aura, Freed from the Real, Selvala, Bloom Tender, …) and the
spell-copy cluster (Twincast, Fork, Reverberate, Narset's Reversal + Kalamax,
Alania, Prismari, …).

## 2026-09-04 — why the Tenstorrent cards, and the tensor formulation

User direction: run the search on the quietbox's cards, not the CPU, because
the follow-up problem (3+ card combos → hypergraph, far more edges) will need
the throughput. So the search is reformulated as dense tensor ops over a
*population* of P decks at once:

```
X : P × n   0/1 matrix, one deck per row (n padded to 1664 = 52 tiles)
C = X @ A   C[p,v] = combos card v has with deck p          (one matmul)
V = C + bonus
best add  b* = argmax over v ∉ deck of V[p,v]  (random tie-break, tabu)
R = onehot(b*) @ A                              (row A[b*,:], second matmul)
best drop a* = argmin over v ∈ deck of V[p,v] + R[p,v]
X ← X + onehot(b*) − onehot(a*)
```

Cost per generation: 4·P·n² flops for the two matmuls plus ~30 elementwise /
reduction passes over P × n. For the hypergraph follow-up the only change is
that `C` becomes `[X@H == r−1] @ Hᵀ` (gains) and `[X@H == r] @ Hᵀ` (losses)
with H the n × E incidence matrix — same op mix at much larger shapes.

`population.py` is the NumPy reference implementation of exactly this
generation step; `tt_search.py` is the ttnn port. The reference reaches 243 in
100 generations with P = 64 (0.4 s), matching the CPU tabu search.

### Hardware / software actually used

- 4 × Tenstorrent Blackhole cards (`/dev/tenstorrent/0-3`, PCIe 01/41/42/c1),
  KMD 2.6.0, host AMD EPYC 8124P (16C/32T), 503 GB RAM.
- tt-metal checkout `/home/ttuser/sjameel/tt-metal` (commit 8542d12dc2,
  2026-02-19) has a complete Release build and a `python_env` with torch
  2.7.1+cpu. Used **read-only** (`TT_METAL_HOME` + its interpreter); the
  qwen38 bring-up trees are not touched. `~/qwen38-src` has `_ttnn.so` but no
  runtime libs, and `~/.tenstorrent-venv` has no torch.
- Smoke test: device 0 opens in 4.3 s; 1024×1664×1664 bf16 matmul = 0.73 ms
  (~7.8 TFLOP/s).

### Numerical exactness findings (this build, Blackhole)

The search only works if counts are exact. Measured:

| op | finding |
|---|---|
| `matmul` bf16 × bf16 of 0/1 matrices | exact while outputs < 256 (ours ≤ 86 = max degree) |
| `typecast` bf16→f32 | exact |
| `sum` / `max` on float32 input | **inputs are rounded to bf16**, accumulation in fp32: exact for small integers, off by up to 0.5 for values like 90.37 |
| `argmax` on float32 TILE | exact, but single-core: **119 ms** for 4096×1664 (everything else in a generation totals ~15 ms) |
| `argmax(use_multicore=True)` | needs ROW_MAJOR input; 0.6 ms relayout + 17 ms |
| `topk` | bf16 input only |
| `typecast` on argmax output | rejects (P,1) row-major; must `to_layout(TILE)` first |
| `rand` f32 | 0.6 ms for 4096×1664, seedable |
| elementwise ops (add/mul/eq/where) on 4096×1664 | ~0.55 ms each, i.e. mostly dispatch-bound |
| `where` with column broadcast | 1.9 ms |

Consequence: selection is done with exact `max` reductions on bf16-representable
values instead of `argmax`:

1. `m = max(V_masked)`, `cand = (V_masked == m)` — V is small integers, exact.
2. random tie-break: `key = cand * (1 + q)` with `q ∈ {0, 1/128, …, 127/128}`
   (7 fractional bits, so `1+q` is exact in bf16); `cand2 = (key == max(key))`.
3. deterministic uniqueness: split the column index into `hi = idx // 32 + 1`
   and `lo = idx % 32 + 1` (both < 256, exact), take `max(cand2 * hi)` then
   `max(· * lo)`; the survivor is unique (largest index among ties).

10 ops, **7.7 ms** at P = 4096, verified exact against a torch reference,
and it yields the one-hot row directly (no iota compare, no typecast).
The NumPy reference implements the same rule so the two can be diffed.

## 2026-09-04 — ttnn port: results, bugs found, throughput

`tt_search.py` runs `population.generation` on one device or a 1×4 mesh
(`ttnn.open_mesh_device(MeshShape(1,4))`, population sharded on dim 0, graph
replicated). Every step is checked bit-for-bit against the NumPy reference with
identical host-supplied noise (8 generations, single device and mesh): X, score
and tabu all equal, every row stays a 50-subset.

Tests (`tests/test_search.py`, 9 new): on random 16–18 vertex graphs with
bonus vertices, both the CPU tabu search and the NumPy population search find
the brute-force optimum (all C(n,6) subsets enumerated); the tie-break rule and
the noise quantisation are unit-tested.

### Throughput (one generation = one swap per replica)

| population P | devices | ms / generation | swaps / s |
|---|---|---|---|
| 4,096 | 1 | 245 (with `argmax`) | 16.7k |
| 4,096 | 1 | 25.0 (max-based select) | 164k |
| 8,192 | 1 | 25.8 | 317k |
| 16,384 | 1 | 24.9 | 659k |
| 65,536 | 4 (mesh) | 28.1 | 2.34M |
| 131,072 | 4 (mesh) | 38.5 | 3.40M |

Time per generation is flat until P ≈ 16k per device: the ~45 ops are
dispatch-bound at ~0.5 ms each, so a bigger population is free up to the point
where the two matmuls (4·P·n² flops ≈ 90 GFLOP per matmul at P = 16k) start
to dominate. Big-O per generation: O(P·n²) matmul + O(P·n) elementwise; per
swap that is O(n²) — worse than the O(k·n) CPU swap — but it is dense, exact,
and runs at ~2–3M swaps/s on four cards versus ~3k/s per CPU core in NumPy.
The dense matmul wastes the 99.9 % sparsity of A; that only matters once the
hypergraph incidence matrix makes n × E large, and is the natural next
optimisation (sparse/gather formulation of `onehot @ A`).

### Bugs found on the way

- `ttnn.where(mask_f32, a_bf16, b_bf16)` with a (P,1) column-broadcast mask
  returns wrong rows **silently** (every row wrong in a 256-row test). With a
  bf16 mask, or with `b + mask*(a-b)`, it is exact. This corrupted the
  per-replica best-deck tracking: the "best deck" had 98 cards, kicked copies
  of it were reseeded into the population, and scores of 25,424 appeared.
  `_bookkeep` now asserts the incumbent is a k-subset.
- `ttnn.subtract(scalar, tensor)` is not accepted (tensor-first only).
- `ttnn.rand` on a mesh gives every device the same stream for a seed; noise is
  now `frac(u + salt_row)` with a per-device salt (sharded upload), verified
  different across devices and still exact multiples of 1/128.

### Host ↔ device traffic

First driver version uploaded/downloaded X, tabu and best-X every epoch:
4 × 218 MB at P = 65,536 → 7.6 s of transfers per 5.6 s of compute, 0.9M
swaps/s end to end. Now X, tabu, best score and best deck stay on device;
per epoch only the (P,1) best-score column comes back (256 KB). The full X
round trip happens only on reseed epochs, and the best deck is fetched only
when the global best improves. End-to-end rate went to 1.96M swaps/s at
P = 65,536 with 300 generations per epoch (10 s epochs).

### Search behaviour on the real instance

Half the population (≈31,900 of 65,536 replicas) sits at **243** after the
first 300 generations, and the mean per-replica best is 216. The CPU tabu
search, the NumPy population search and the TT search all stop at 243 from
independent random starts, which is strong (not conclusive) evidence that 243
is the optimum for this graph.

## 2026-09-04 — precision question (user: "could you use bf4b?")

Precision matters only in the sense that every count must stay integer-exact.
But both matmul operands are 0/1 matrices, and block-float formats share one
exponent per 16-value block: a block whose nonzero entries are all 1.0 is
lossless even with bfp4_b's 3-bit mantissa. So `X` and `A` can be bfp4_b and
the matmul can run at LoFi (only the top mantissa bits are multiplied, and
1.0 has none) with no loss. Outputs must stay bf16 (or bfp8_b, exact for
integers < 128 per block; bfp4_b would round counts to multiples of 16).
Elementwise state stays bf16 because those ops are dispatch-bound below
~16k replicas per device, so fewer bytes would not help; the win is in the
matmuls at large P. Benchmark script: `scratchpad/bench_bfp.py` (run when the
cards are free).

## 2026-09-04 — bound attempt

MILP (HiGHS via scipy): x_v binary, y_e ≤ x_u, y_e ≤ x_v, Σx = 50, plus the
vertex-cover cut Σ_{e∋v} y_e ≤ min(deg v, k−1)·x_v. Expected to be weak (the LP
relaxation of DkS is notoriously loose); run with a 10-minute limit to record
the gap. Script: `scratchpad/milp_bound.py`.
Decision: not benchmarked now (user: skip if confident). Below ~16k replicas
per device the generation is dispatch-bound (flat 25 ms from P = 4k to 16k),
and the end-to-end rate is bounded by host-side reseeding, so bfp4_b/LoFi
would not change time-to-answer on this graph. It becomes worthwhile at
≥ 32k replicas per device (where the 38 ms generation is matmul-dominated)
and for the hypergraph follow-up, where it is the first thing to try.

**Result: the bound closed.** HiGHS (scipy `milp`) proves the optimum in
**4.8 s**: objective 243, dual bound 243, gap 0. Even the plain formulation
without the vertex-cover cuts solves in 6.1 s. The instance is sparse enough
(2,765 edges, average degree 3.4, one giant component) that the LP relaxation
plus HiGHS's own cuts is tight — the "≤160 vertices for exact methods"
figure from the literature is for dense benchmark graphs and does not apply
here. The MILP deck is rescored independently at 243 and written to
`data/deck_milp/`. It differs from the TT deck in one card (Krosan Restorer
vs Bigger on the Inside): the optimum is not unique.

So for two-card combos the answer is settled: **243 unique two-card infinite
combos is the maximum for 50 cards with Kenrith as commander**, and the
heuristic backends (CPU tabu, NumPy population, TT population) all found it
from random starts within their first seconds. The value of the TT machinery
is for the hypergraph follow-up, where the MILP will have one variable per
combo (~100k) and the LP will no longer be tight.

## 2026-09-04 — runs for the record

- CPU tabu, 16 workers × 300 s: 243 (`data/deck_cpu/`).
- TT population, 4 cards, P = 65,536, 400 generations/epoch, 1,200 s: 243
  from epoch 1 on; ~1.2–1.5M swaps/s while the CPU run and the MILP shared
  the host (the driver's per-op dispatch is host-CPU-bound), 1.96M alone.
- MILP: 243 proven optimal in 4.8 s (`data/deck_milp/`).

The three optimal decks are not the same deck. TT and MILP share 49 of 50
cards (Bigger on the Inside ↔ Krosan Restorer). The CPU deck shares only 31
with them: it drops 19 cards of the untap/mana cluster for the spell-copy
cluster (Twincast, Fork, Reverberate, Reiterate, Increasing Vengeance,
Narset's Reversal, Wild Ricochet, Dual Strike, Insidious Will, Return the
Favor, Refuse // Cooperate, Expansion // Explosion, Flare of Duplication,
Kalamax, Alania, Prismari, Rootha, Adaptive Training Post, Pyromancer's
Goggles) — and still scores exactly 243. Staff of Domination has 36 in-deck
partners in one optimum and 18 in the other. So the 243 plateau is broad; a
player can choose the flavour. Published as an artifact:
https://claude.ai/code/artifact/f8496d2d-4d9b-4b54-ae59-ab2708237626

Final line of the 20-minute TT run: 63 epochs, 25,200 generations,
1.65 billion swap moves, 1.54M swaps/s end to end, best 243 throughout;
48,599 of 65,536 replicas sitting at 243 by the end.

## 2026-09-04 — variant: win-condition combos only, 60 cards

User request: only count combos whose result is infinite life loss, life gain,
damage, +1/+1 counters on creatures, turns, or creature tokens; deck of 60.
Implemented as `--result REGEX` (repeatable, any match counts) on the edge's
produced-feature names. Regexes used:

```
^Infinite lifeloss                      (incl. "for target opponent")
^Infinite lifegain$                     (not "lifegain triggers")
^Infinite (combat )?damage(?! to (most |some |all )?creatures)
^Infinite \+1/\+1 counters on           (a creature / creatures you control / certain / a token)
^Infinite turns
^Infinite (tapped )?creature tokens?(?! for)   (not tokens given to opponents)
```

Filtered instance: 824 cards, 1,111 combos (from 1,623 / 2,765); no commander
bonus (Kenrith's Composite Golem combo is a mana combo). Trivial bound
Σ top-60 degrees / 2 = 441.

Compute budget: the full instance converged inside the first epoch, so the
TT run was given 180 s (P = 65,536, 300 generations/epoch) — generous by an
order of magnitude. MILP on the filtered instance: **183, proven optimal in
0.7 s**.

## 2026-09-04 — prerequisites (user: "no additional notable prerequisites?")

The searches so far ignored Commander Spellbook's `notablePrerequisites` and
the cards' starting zones. Tally over the 2,774 accepted combos: 1,819 carry a
notable prerequisite ("you have a way to deal damage to Vrondiss", "you
control two other creatures with toughness 1", …); 141 have both cards in
hand, 53 need a card in exile, 33 start from graveyard/library/command zone.

Added per variant: `notable_prerequisites`, `easy_prerequisites`,
`battlefield_or_hand` (all zones ⊆ {B, H}), `clean` (= no notable
prerequisites and battlefield/hand). Per edge: `clean` if any of its variants
is clean, plus `clean_features` (results produced by clean variants) so the
win-condition filter under `--strict` only looks at clean variants.
`--strict` on `search` keeps clean edges only.

| rules | cards | combos | k | optimum (MILP) | time |
|---|---|---|---|---|---|
| all results, prerequisites allowed | 1,623 | 2,765 | 50 | 243 | 4.8 s |
| all results, strict | 740 | 881 | 50 | **128** | 0.1 s |
| win conditions, prerequisites allowed | 824 | 1,111 | 60 | 183 | 0.7 s |
| win conditions, strict | 273 | 256 | 60 | **82** | 0.0 s |

The strict win-condition instance has fewer combos (256) than the loose one
has cards in its deck; a 60-card list holds 82 of them. Strict decks are
certified by the MILP only (user: no TT run once a MILP optimum exists —
recorded as a working rule). Artifact now has a dropdown over the four
results, each with its own graph, table, partner panel and decklist:
https://claude.ai/code/artifact/f8496d2d-4d9b-4b54-ae59-ab2708237626
For the record, the two strict TT runs that were already in flight both hit
the MILP optimum in their first epoch (128 at 28 s, 82 at 26 s); they are not
used by the artifact.

## 2026-09-04 — infinite mana only, strict, 50 cards

Regex: `^Infinite ([\w-]+ )?mana\b(?!.*(for|if your) (target )?opponent)` —
coloured / colourless / single-colour / "mana X you control can produce",
restricted-use mana included (it still pays Kenrith's activated abilities;
"can only be spent to cast creature spells" is the one debatable case, 4
clean edges), mana handed to opponents excluded. Strict pool: 132 cards,
145 combos. MILP: **88, proven optimal in 0.5 s**
(`data/deck_mana_strict_milp/`). Untapper cluster again: Doc Samson and
Mona Lisa (12 partners each), Ringing Strike Mastery 11, Basalt Monolith 7,
Freed from the Real / Pemmin's Aura 6. Added as the fifth dropdown entry.

## 2026-09-04 — a middle ground on prerequisites (`--prereqs kenrith`)

User: "all infinite assumes too much, no prerequisites assumes too little";
power thresholds are fine (Kenrith adds counters), needing to gain life is
fine (Kenrith gains life), counting creatures is fine (the deck is mostly
creatures), counting other permanents is not; "you control two enchantments"
is fine when the two combo cards are the enchantments.

`spellbook_graph/prereqs.py` classifies each line of `notablePrerequisites`:

- allowed — Kenrith-satisfiable (summoning sickness / haste / trample,
  power & toughness thresholds, +1/+1 counters, gain life, life total, draw a
  card, creature card in graveyard, "cast your commander"), inherent to the
  two cards (aura/equipment attached, clone copy, name chosen, soulbond, mana
  to cast), trivial state (haven't activated/cast yet, deck size parity),
  any plain creature count (colour / no-summoning-sickness qualifiers ok),
  and any "you control N <type>" that the two cards plus Kenrith supply
  (types checked against the cards' type lines);
- denied — counting other permanents, mana-producing boards, anything
  opponent-dependent, "you have a way to …" (third card), keywords from
  elsewhere, city's blessing / max speed, extra cards in hand, low life,
  non-+1/+1 counters, and any unclassified line (329 of them remain; top:
  "You have a sorcery card in hand", "four colours among permanents").
  A line matching both allow and deny is denied ("no summoning sickness and
  cannot be blocked").

Per-variant prerequisite text and zones are now stored on each edge in
`graph.json` so policies can change without re-scanning the export.

| prerequisites | counting | pool | combos | k | optimum |
|---|---|---|---|---|---|
| allowed | all infinite | 1,623 | 2,765 | 50 | 243 |
| allowed | win conditions | 824 | 1,111 | 60 | 183 |
| Kenrith-provided | all infinite | 927 | 1,298 | 50 | **180** |
| Kenrith-provided | win conditions | 391 | 410 | 60 | **109** |
| Kenrith-provided | infinite mana | 208 | 289 | 50 | **153** |
| none notable | all infinite | 740 | 881 | 50 | 128 |
| none notable | win conditions | 273 | 256 | 60 | 82 |
| none notable | infinite mana | 132 | 145 | 50 | 88 |

All eight are MILP-certified in ≤ 0.8 s. The artifact dropdown is grouped by
prerequisite policy; the summary table on the page is generated from the same
instances.

## 2026-09-04 — combos of any size (hypergraph), unrestricted pool, 50 cards

User: "infinite combos of any size, unrestricted pool (any infinite, any
prerequisite), 50 cards. You'll probably need to go sparse." Goal stated
afterwards: a program that searches for the maximum density of combos
extremely efficiently on the tt-quietbox.

Data: 93,934 unique card sets (2: 2,774; 3: 42,058; 4: 41,502; 5: 7,565;
6–10: 35) over 6,515 cards; 335,789 incidences. Hubs: Ashnod's Altar 5,713
combos, Phyrexian Altar 4,794, Pitiless Plunderer 2,919. With Kenrith in the
command zone his 346 combos shrink by one card; 1,097 combos needing another
commander are dropped. Builder: `hypergraph.py` (`build --card-count 0`).

MILP: not run, on purpose (user: skip if it won't work). The LP relaxation
of "y_e ≤ x_v for all v in e" with hubs in thousands of combos is hopeless.

CPU tabu (`hyper.py`): sparse incremental swap deltas,
delta(drop a, add b) = gain[b] − loss[a] − corr[a, b], verified against full
rescoring; ~5.6 ms per best-swap iteration in NumPy (k × n neighbourhood plus
bincounts over the incidence list). 28 workers × 300 s: **1,745** (1,570
three-card, 174 four-card, 1 two-card; Sun Titan / Karmic Guide / Fiend
Hunter / Saffi cluster) with half the workers stuck at 1,692. Both basins are
reached within seconds and never left; the landscape is rugged.

Tensor formulation ("go sparse"): the dense `X @ H` (P × n × E) would be
~1.25 TFLOP per matmul at P = 1024 and wastes 99.95 % zeros. Instead
(`hyperpop.py`, `tt_hyper.py`): incidence space. Every combo owns r slots,
grouped by card into 32-aligned ranges; state cnt (I × P) = deck cards in the
combo minus (r − 1), so near-complete == 0 and complete == 1. Adding card b
gathers row b of the (n × I) incidence table with `ttnn.embedding`, tilizes,
transposes and adds; "combos completed by v" is a tile-row sum over 32 slots
(exact in bf16) followed by a 0/1 matmul (n × chunks) with fp32 accumulation.
Selection over values > 256 uses five exact max reductions (value // 256,
value % 256, noise, index hi, index lo). Everything tiled (row-major
elementwise is 3× slower). Bit-for-bit exact against the NumPy reference with
host-supplied noise at P = 64 … 4096. Eager: 72 ms/generation at P = 1024,
237 ms at P = 4096 on one Blackhole (14–17k swaps/s); ~12 passes over the
slot tensor plus ~45 dispatch-bound small ops.

Trace replay (user: "it should trace and replay where appropriate"): all state
updated in place via `output_tensor=`, a block of 4 generations captured with
`begin_trace_capture` and replayed; replay equals eager execution exactly, and
`ttnn.rand` inside the trace is de-correlated with an on-device Weyl salt.
14.2 ms/generation at P = 256 (18k swaps/s on one card) — the dispatch cost is
gone. Not yet run at scale or on the mesh; see HANDOFF.md.

Device findings: `embedding(layout=TILE)` wrong beyond 512 KB rows; `sum` on
bf16 rounds its output (1,023 → 1,024); `reshape` to a 4-D tile view is a real
copy; capture does not execute. Full list in HANDOFF.md.
