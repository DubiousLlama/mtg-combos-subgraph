# Probe: does ttnn.embedding(layout=TILE) fail on row size (>512KB) or on
# chunk alignment (hidden % 2048 != 0 with >= 2 blocks per core)?
import sys, time, torch, ttnn

dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
only = sys.argv[2] if len(sys.argv) > 2 else None
device = ttnn.open_device(device_id=dev_id)
grid = device.compute_with_storage_grid_size()
ncores = grid.x * grid.y
print(f"device {dev_id} arch={device.arch()} grid={grid.x}x{grid.y} ncores={ncores}", flush=True)

VOCAB = 64

def run(name, hidden, blocks):
    rows = 32 * blocks
    torch.manual_seed(0)
    ids = torch.randint(0, VOCAB, (1, rows), dtype=torch.int32)
    w = torch.randn(VOCAB, hidden).bfloat16()
    ref = torch.nn.functional.embedding(ids.long(), w)  # [1, rows, hidden]
    t_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t0 = time.time()
    out = ttnn.embedding(t_ids, t_w, layout=ttnn.TILE_LAYOUT)
    got = ttnn.to_torch(out)
    dt = time.time() - t0
    bad = got.view(rows, hidden) != ref.view(rows, hidden)
    nbad = int(bad.sum())
    frac = nbad / bad.numel()
    bad_rows = torch.nonzero(bad.any(dim=1)).flatten().tolist()
    bad_blocks = sorted(set(r // 32 for r in bad_rows))
    first_col = int(torch.nonzero(bad.any(dim=0)).flatten()[0]) if nbad else -1
    ntiles = hidden // 32
    print(f"[{name}] hidden={hidden} ({hidden*2/1024:.0f} KB/row, {ntiles} tiles, "
          f"{ntiles % 64} tiles in last chunk) blocks={blocks} ({blocks/ncores:.2f}/core) -> "
          f"{'EXACT' if nbad == 0 else 'WRONG'} bad={nbad} ({100*frac:.2f}%) "
          f"bad_rows={len(bad_rows)} bad_blocks={bad_blocks[:12]}{'...' if len(bad_blocks) > 12 else ''} "
          f"first_bad_col={first_col} t={dt:.1f}s", flush=True)
    ttnn.deallocate(out); ttnn.deallocate(t_w); ttnn.deallocate(t_ids)

cases = [
    # name, hidden, blocks
    ("small-unaligned-1blk", 8224, ncores),            # 257 tiles, 16 KB rows, 1 block/core
    ("small-unaligned-2cores2blk", 8224, ncores + 2),  # only cores 0,1 get a 2nd block
    ("small-unaligned-2blk", 8224, 2 * ncores),        # 2 blocks/core
    ("small-unaligned-4blk", 8224, 4 * ncores),        # 4 blocks/core
    ("small-aligned-4blk", 10240, 4 * ncores),         # 320 tiles = 5*64, 20 KB rows
    ("512KB-aligned-2blk", 262144, ncores + 2),        # user's exact width
    ("516KB-aligned-2blk", 264192, ncores + 2),        # 129*2048: > 512 KB, aligned
    ("user-707KB-1blk", 361888, ncores),               # user's failing width, 1 block/core
    ("user-707KB-2cores2blk", 361888, ncores + 2),     # user's failing width, cores 0,1 get 2 blocks
]
for name, hidden, blocks in cases:
    if only and only not in name:
        continue
    try:
        run(name, hidden, blocks)
    except Exception as e:  # keep going; report
        print(f"[{name}] EXCEPTION {type(e).__name__}: {str(e)[:300]}", flush=True)
ttnn.close_device(device)
