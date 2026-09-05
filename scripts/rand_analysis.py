"""Bit-level look at one seeded ttnn.rand tensor. usage: rand_analysis.py [device_id]"""
import sys, numpy as np, ttnn
dev = ttnn.open_device(device_id=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
grid = dev.compute_with_storage_grid_size(); ncores = grid.x * grid.y
shape = (32 * 8, 32 * 8 * ncores)
a = ttnn.to_torch(ttnn.rand(shape, device=dev, dtype=ttnn.float32, seed=5)).float().numpy()
ttnn.close_device(dev)
R, C = a.shape
tiles = a.reshape(R // 32, 32, C // 32, 32).transpose(0, 2, 1, 3).reshape(-1, 32, 32)  # (ntiles, 32, 32)
u = (a.astype(np.float32).view(np.uint32))
mant = u & 0x7FFFFF
# which mantissa bits ever vary, and how many distinct mantissas exist
varying = [b for b in range(23) if ((mant >> b) & 1).any() and not ((mant >> b) & 1).all()]
print(f"values: {a.size:,}; distinct mantissa patterns {len(np.unique(mant)):,}; varying mantissa bits: {varying}", flush=True)
# entropy per bit
for b in range(22, -1, -1):
    p = ((mant >> b) & 1).mean()
    print(f"  bit {b:2d}: P(1) = {p:.4f}", end=";" if b % 4 else "\n")
print()
# per-lane structure: within a tile, 32x32 = 1024 values; the SFPU produces 8 stores of 128 lanes(?) per tile
t0 = tiles[0]
print(f"distinct values in tile 0: {len(np.unique(t0))} of 1024; in first 64 tiles (one core): {len(np.unique(tiles[:64])):,} of {64*1024:,}", flush=True)
# repeated values: how often does the same value recur within one core's stream, and at what lag?
core = tiles[:64].reshape(-1)
vals, idx, counts = np.unique(core, return_index=True, return_counts=True)
print(f"one core (65,536 values): distinct {len(vals):,}; most repeated value occurs {counts.max()} times", flush=True)
# lag autocorrelation within one core's stream
x = core - core.mean()
ac = [float(np.dot(x[:-l], x[l:]) / np.dot(x, x)) for l in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
print("autocorr lags 1,2,4,...,1024 within one core's tile stream:", " ".join(f"{v:+.3f}" for v in ac), flush=True)
# same position in consecutive tiles (lane-wise)
p = tiles[:64].reshape(64, -1)
print(f"corr of the same tile position across consecutive tiles (mean over 1024 positions): {np.mean([np.corrcoef(p[:-1, j], p[1:, j])[0, 1] for j in range(0, 1024, 7)]):+.3f}", flush=True)
# are per-core streams (seed+i) related beyond the shift? compare core 0 and core 1 lag structure: values shared
s0, s1 = set(tiles[:64].ravel().tolist()), set(tiles[64:128].ravel().tolist())
print(f"values shared between core 0 and core 1 streams: {len(s0 & s1):,} of {len(s0):,}", flush=True)
