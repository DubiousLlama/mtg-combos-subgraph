# `ttnn.rand` in a scientific search loop: limitations met and workarounds used

Date: 2026-09-05. Author: Samuel Jett (with Claude). Host: tt-quietbox, 4 × Blackhole p150
(device 0 reports a 13 × 10 compute grid, 130 cores). tt-metal build at
`/home/ttuser/sjameel/tt-metal` (commit 8542d12dc2, 2026-02-19); the op source quoted below is
`ttnn/cpp/ttnn/operations/rand/` in that tree.

Context. The MTG combo optimiser (`spellbook_graph/tt_search.py`, `tt_hyper.py`) runs a population
tabu search: P replicas (1,024 to 16,384 per device) each make one swap per generation, and the
best card is chosen by *exact* `max` reductions over small integers, with ties broken by a uniform
noise tensor. That needs, per generation, a fresh (cards × replicas) uniform tensor that is

1. different on every device of a mesh,
2. different on every replay of a captured trace (a generation is replayed thousands of times),
3. reproducible when we want to check the device against a NumPy reference,
4. exactly representable in bfloat16 as a multiple of 1/128, so that `1 + noise` is exact.

`ttnn.rand` gives none of the first three out of the box. Everything below was measured on the
box with `scripts/rand_probe.py` and `scripts/rand_analysis.py` in this repo.

## 1. How the op actually behaves

**Generation.** `rand.cpp` always generates float32 tiles on the SFPU
(`rand_tile`: read the hardware PRNG register, force the exponent to 127 to get [1, 2), subtract
1, then `from + scale * x`), then runs a separate `typecast` for any other dtype and a separate
`to_layout` for row-major output. A bfloat16 request therefore costs two ops.

**Seeding** (`rand_program_factory.cpp`). Core *i* of the grid receives the runtime argument
`seed != 0 ? seed + i : get_random_seed()`, where `get_random_seed()` draws from a static
`std::mt19937 rng(std::time(nullptr))`. The default `seed` is 0. The same code is in
`ttnn.uniform` and `ttnn.bernoulli`. Consequences, all confirmed on hardware:

| behaviour | measurement |
|---|---|
| `seed=s` is repeatable | two calls with `seed=5` are identical |
| unseeded calls differ call to call | yes, but see the next row |
| unseeded streams are seeded from wall-clock **seconds** | two processes launched at the same time on devices 2 and 3 produced byte-identical "random" tensors (hash `2edf34f1…` on both) |
| `seed+1` is `seed` shifted by one core | with 64 tiles per core, seed 6 == seed 5 shifted by 64 tiles; 8,256 of 8,320 tiles of seed 6 occur in seed 5. Any two seeds closer than the core count (130 here) overlap. |
| the stream depends on the raw per-core seed only | seed 5 on core 0 == seed 4 on core 1; the same seed with a different tensor shape gives the same leading tiles |
| every device of a mesh gets the same numbers | 1 × 2 mesh, seeded and unseeded: device 0 == device 1 (the program is built once, the runtime args are identical) |
| a trace replays the same numbers, seeded **or unseeded** | the seed is a kernel runtime argument fixed at capture; replay 1 == replay 2 in both cases |

**Quality of the stream** (seed 5, 8.5 M values, 130 cores × 64 tiles):

| property | measurement |
|---|---|
| mean / variance | 0.4986 / 0.0834 (uniform: 0.5 / 0.0833) |
| range | [1.07e-6, 0.99996668); `high` is applied as `high − 1e-6` |
| mantissa bits | all 23 vary, P(1) ≈ 0.50 for bits 4–22; low bits biased (bit 0: 0.53, bit 3: 0.44) |
| distinct values | **94 per 1,024-value tile**; 2,110 per core stream of 65,536; 104,470 distinct mantissas in 8.5 M values |
| repetition structure | within a tile the value pattern of column 0 repeats with period 17 rows; the most frequent value in a core stream occurs 32 times |
| autocorrelation within a core's stream | +0.31 at lag 1, +0.22 at lag 2, +0.18 at lag 128, ≈ 0 elsewhere |
| streams of different raw seeds | 0 shared values between core 0 of seed 5 and core 0 of seed 135 |
| bfloat16 output | float32 rounded to nearest, so it **returns exactly 1.0** (15,187 of 8.5 M values) although the doc says [0, 1) |
| speed | 0.29 ms for 3,520 × 1,024 float32; 0.28 ms for 2,752 × 4,096 (160 GB/s written) |

The moments are fine, so a quick histogram would pass; the value repetition and the lag
correlation would fail any serious test battery. For a Monte Carlo estimate with N samples
the effective sample size is a small fraction of N.

**Missing.** No integer output (`randint`), no normal distribution, no permutation / choice,
no in-place `output_tensor=` (the tensor is always freshly allocated, which matters inside a
trace), no per-device seed on a mesh.

## 2. What the search does about it

All of this lives in `TTHyperPopulation._noise` / `_advance_salt` (`tt_hyper.py`) and
`TTPopulation._noise` (`tt_search.py`).

- **W1, mesh decorrelation by salt.** A per-device salt (uploaded *sharded*, so each device holds
  a different slice) is added to the device stream and the fraction taken:
  `u = frac(rand(seed) + salt)`. Same `rand` output on every device, different noise.
- **W2, trace decorrelation by a Weyl sequence.** The salt is a persistent device tensor advanced
  in place once per generation, `salt ← frac(salt + 0.6180339887)`, through `output_tensor=` so it
  is part of the captured trace. Each replay adds a different salt to the same replayed `rand`
  output. Verified: two replays of one trace give different populations and both rescore
  exactly on the host.
- **W3, exact bfloat16 tie-break keys.** The two-card search quantises on the device,
  `floor(u · 128) / 128`, so `1 + q` is exact in bfloat16 and the NumPy reference can reproduce
  it (`population.quantize_noise`). The hypergraph search typecasts `u` to bfloat16 directly and
  accepts that 1/256 of the values round up to 1.0 (harmless for a tie-break key).
- **W4, seed spacing.** The code advances the seed by 1 (two-card) or 2 (hypergraph) per call,
  so under the `seed + core` rule consecutive noise tensors are shifted copies of each other.
  This went unnoticed because the salt decorrelates them; it is a trap for anyone using
  consecutive seeds for independent replicas. Seeds should be spaced by more than the core count
  or hashed.
- **W5, everything integer happens on the host.** Random k-subsets for the initial population,
  kicked copies of the incumbent, and reseeding decks are drawn with NumPy and uploaded.
- **W6, verification with host noise.** Because the device stream cannot be reproduced on the
  host, the bit-for-bit check of the device step against the NumPy reference uploads
  host-generated noise (`scripts/check_tt_hyper.py`); the device-noise path is validated only
  indirectly, by rescoring every result from scratch on the host.

## 3. What would make the op usable for scientific work

1. **A counter-based generator** (Philox / Threefry style) keyed by `(seed, element index,
   call counter)`: reproducible, no shifted-copy artefact between seeds, independent per element,
   and trivially different per device by mixing the device id into the key.
2. **Seed derivation by hashing**, not `seed + core_index`; unseeded calls should draw from a
   `std::random_device`-seeded generator (not `time(nullptr)` seconds) and mix in the device id.
3. **Device-side RNG state for traces**: an optional state tensor that the op reads and
   advances in place, so a captured trace produces fresh numbers on every replay. This is exactly
   W1 + W2 done inside the op.
4. **Fix the per-tile repetition** (94 distinct values per tile, lag-1 correlation 0.31): the
   SFPU PRNG register is being read in a way that repeats values across lanes / rows.
5. **bfloat16 that honours [0, 1)**: truncate instead of round when casting, or generate 8
   mantissa bits directly.
6. `output_tensor=` for in-place generation inside traces; `randint`, `randn`, `randperm`.

## 4. Reproduction

```
export TT_METAL_HOME=/home/ttuser/sjameel/tt-metal
export PYTHONPATH=$PWD:$TT_METAL_HOME/ttnn:$TT_METAL_HOME
$TT_METAL_HOME/python_env/bin/python scripts/rand_probe.py 0        # seeding, shift, trace, bf16, timing
$TT_METAL_HOME/python_env/bin/python scripts/rand_probe.py --mesh   # 1x2 mesh: same numbers on both devices
$TT_METAL_HOME/python_env/bin/python scripts/rand_probe.py --hash 2 & \
$TT_METAL_HOME/python_env/bin/python scripts/rand_probe.py --hash 3   # same-second launches: identical unseeded tensors
$TT_METAL_HOME/python_env/bin/python scripts/rand_analysis.py 0     # mantissa bits, distinct values, autocorrelation
```
