# ttnn.embedding with TILE_LAYOUT output: known corruption bug and how to use the op safely

Date: 2026-09-05. Author: Samuel Jett (with Claude). Evidence host: f07cs04 (Blackhole p150),
tt-metal main 954f5cf7 (2026-08-25) and 263c7570 (2026-08-27); the op code is unchanged on
main 1aa1b457 (2026-09-04). Applies to Wormhole too (the defect is in arch-independent kernel code).

## 1. Rule for implementers (read this first)

`ttnn.embedding(ids, weights, layout=ttnn.TILE_LAYOUT)` produces WRONG ROWS, or hangs the board,
when ALL THREE hold:

1. hidden (weight row width, elements) > 8192
2. hidden % 2048 != 0
3. number of gathered rows (batch * seq) > 32 * number of compute cores
   (p150: 110 or 130 cores depending on harvesting, so > 3520 or > 4160 rows; n150: 64 cores)

There is NO row-size cap. 707 KB rows are exact when condition 2 or 3 is false. The 512 KB
boundary we first saw was a coincidence (262144 = 128 x 2048 is aligned, 361888 is not).

Condition 3 is the one that hides the bug in small tests: the first 32-row block on every core is
always correct. Decode-sized calls (a few rows) never trigger it; prefill-sized calls do.

ROW_MAJOR output is correct at every width and row count (it has its own 1 MB chunking that is fine).

### Safe call pattern (Python)

```python
def embedding_tiled(ids, weights, hidden):
    """ids: uint32 ROW_MAJOR [batch, seq] on device. weights: bf16 ROW_MAJOR [vocab, hidden] on device."""
    if hidden <= 8192 or hidden % 2048 == 0:
        return ttnn.embedding(ids, weights, layout=ttnn.TILE_LAYOUT)      # fused path is safe here
    out = ttnn.embedding(ids, weights, layout=ttnn.ROW_MAJOR_LAYOUT)      # always exact
    return ttnn.to_layout(out, ttnn.TILE_LAYOUT)                           # separate tilize op
```

Alternative when the table is reused (token embeddings): pad the table once at load time to the
next multiple of 2048 columns and keep the fused TILE path:

```python
pad = (-hidden) % 2048                      # only needed when hidden > 8192
w_padded = torch.nn.functional.pad(w, (0, pad))          # [vocab, hidden + pad], zeros on the right
weights = ttnn.from_torch(w_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = ttnn.embedding(ids, weights, layout=ttnn.TILE_LAYOUT)  # [batch, seq, hidden + pad]
out = ttnn.slice(out, [0, 0, 0], [batch, seq, hidden])       # drop pad columns if the consumer needs it
```

This is exactly what tt-mlir does in its workaround (tt-mlir PR #8247).

Do NOT rely on keeping rows <= 32 * cores per call. It depends on the device grid and harvesting.

### Op contract details that matter (from `ttnn/cpp/ttnn/operations/embedding/embedding.cpp` and
`device/embedding_device_operation.cpp`)

- weights must be bf16 and end up ROW_MAJOR interleaved. If you pass TILE weights the composite
  converts them with `to_layout` on EVERY call (hidden cost). Store them ROW_MAJOR on device.
- ids: uint32 (or bf16) ROW_MAJOR interleaved, shape [batch, seq] or [seq]. Sharded ids unsupported.
- If `layout` is omitted, the output layout defaults to the WEIGHT's layout. A TILE weight silently
  selects the buggy fused TILE path. Always pass `layout=` explicitly.
- The fused TILE path is used only when seq % 32 == 0 and hidden % 32 == 0. Otherwise the op runs
  ROW_MAJOR and tilizes afterwards (correct, two ops).
- C++: `ttnn::embedding(input, weight, pad_token, layout, embeddings_type, dtype, memory_config,
  optional_output)` follows the same rules.

### Validation checklist for any code that calls this op with TILE output

- Test with rows = 32 * (compute_cores + 2) and a width with hidden % 2048 != 0 and hidden > 8192,
  compare bitwise against `torch.nn.functional.embedding`. Small tests pass and prove nothing.
- Run the small case once with `TT_METAL_WATCHER=1`: the NoC sanitizer reports
  "NOC transaction overflows a circular buffer" deterministically if the bad path is hit.
- `compute_cores = device.compute_with_storage_grid_size(); cores = g.x * g.y`.

## 2. Root cause

When hidden > 8192 the fused TILE-output program factory
(`ttnn/cpp/ttnn/operations/embedding/device/embeddings_fused_program_factory.cpp`) splits every
32-row block into 64-tile chunks and sizes both circular buffers (input CB c_0, output CB c_2) at
2 x 64 = 128 tiles. If `hidden / 32` is not a multiple of 64 the last chunk is partial.
PR #43912 (June 2026) gave the partial chunk the right byte count, but a partial push leaves the CB
write pointer at a non-multiple of 64. On the same core's NEXT 32-row block the second 64-tile chunk
starts mid-ring and runs past the CB end. tt-metal CBs never wrap a multi-page push
(`cb_push_back`: "producer always writes into contiguous memory, it cannot wrap"), so:

- the reader (`kernels/dataflow/embeddings_tilize.cpp`) writes weight rows over the neighbouring CBs
  (the 128 B token scratch and the output CB),
- the compute (`kernels/compute/tilize_chunked.cpp`) and the tile writer read stale slots,
- the dataflow-side pointer never realigns (it wraps only on `== fifo_limit`), so with wide rows it
  drifts off the end of L1 and the board hangs.

Corruption is timing dependent (1.6% to 64% of elements in our runs); rows after the first block on
a core are wrong from some column onward.

## 3. Hardware confirmation (cs04, 110 compute cores, random tokens, bf16, vocab 64)

| hidden | KB/row | last-chunk tiles | blocks per core | result |
|---|---|---|---|---|
| 8224 | 16 | 1 | 1.00 | exact |
| 8224 | 16 | 1 | 1.02 (2 cores get a 2nd block) | wrong, bad blocks exactly {1, 3}, 1.65% of elements |
| 8224 | 16 | 1 | 2.00 | wrong, every odd block, 29% |
| 8224 | 16 | 1 | 4.00 | wrong, 64% |
| 10240 | 20 | 0 (aligned) | 4.00 | exact |
| 262144 | 512 | 0 | 1.02 | exact |
| 264192 | 516 | 0 | 1.02 | exact |
| 361888 | 707 | 45 | 1.00 | exact |
| 361888 | 707 | 45 | 1.02 | board hang; tt-smi reset needed |

Blocks {1, 3} are the second blocks of cores 0 and 1, exactly where `split_work_to_cores` puts the
two leftover blocks.

Watcher run (`TT_METAL_WATCHER=1`, hidden 8224, 2 blocks per core) stopped the device with:

    NCRISC using noc0 tried to unicast read 4096 bytes to local L1[0x05ac00] from DRAM ...
    (NOC transaction overflows a circular buffer)

while running `embeddings_tilize.cpp` / `tilize_chunked.cpp` on worker core (0,0).

## 4. Why it was not caught upstream

- The #43912 regression test (`test_embedding_chunked_partial_last_chunk`) uses 64 rows: one block
  per core.
- `test_embedding_oom` uses hidden 16384 = 8 x 64 tiles: aligned.
- Issue tenstorrent/tt-metal#44500 (Gemma 4 E2B, hidden 8960 = 280 tiles) is the same defect family
  and is still open.
- tt-mlir PR #8247 pads hidden to a multiple of 2048; its empirical rule matches ours.

## 5. Fix direction (upstream)

Every CB must see one push size only. Options:

1. Separate CB pair for the partial chunk (input and output) plus a small writer that drains the
   full-chunk CB and the partial-chunk CB in order per block. Keeps 64-tile chunks.
2. Pick `tiles_per_chunk` as a divisor of the tile count (the first version of #43912). Correct,
   but degenerates to 1-tile chunks for prime tile counts (e.g. 257).

## 6. Reproduction

`emb_tile_probe.py` (next to this note). Run with a built tt-metal:

    export TT_METAL_HOME=<tt-metal> TT_METAL_RUNTIME_ROOT=<tt-metal>
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME
    export TT_METAL_CACHE=$HOME/emb-probe-cache      # cs04: ~/.cache/tt-metal-cache is not writable
    TT_VISIBLE_DEVICES=<n> python emb_tile_probe.py 0                          # all cases
    TT_VISIBLE_DEVICES=<n> python emb_tile_probe.py 0 small-unaligned-2blk     # one case (safe with watcher)

The 361888 x (cores+2 blocks) case hangs the board. Recover with
`tt-smi --offline -r <BDF>` and wait about 60 s before reopening the device.
