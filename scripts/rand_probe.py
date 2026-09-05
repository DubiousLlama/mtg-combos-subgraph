"""Probe ttnn.rand behaviour on hardware (single device, trace, 1x2 mesh, two processes).
usage: rand_probe.py [device_id]   /   rand_probe.py --mesh   /   rand_probe.py --hash device_id (prints a checksum of one unseeded call)"""
import os, sys, time, hashlib
import numpy as np, torch
import ttnn

def to_np(t, **kw):
    return ttnn.to_torch(t, **kw).float().numpy()

def h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]

if "--hash" in sys.argv:
    dev = ttnn.open_device(device_id=int(sys.argv[-1]))
    a = to_np(ttnn.rand((32, 32 * 8), device=dev, dtype=ttnn.float32))
    print(f"pid {os.getpid()} device {sys.argv[-1]} unseeded rand hash {h(a)} first {a.flat[:3]}", flush=True)
    ttnn.close_device(dev); sys.exit(0)

if "--mesh" in sys.argv:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    for label, kw in (("seeded (seed=5)", {"seed": 5}), ("unseeded", {})):
        a = to_np(ttnn.rand((256, 256), device=mesh, dtype=ttnn.float32, **kw), mesh_composer=comp)
        d0, d1 = a[:256], a[256:]
        print(f"mesh 1x2 {label}: device0 == device1: {np.array_equal(d0, d1)}", flush=True)
    ttnn.close_mesh_device(mesh); sys.exit(0)

dev_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dev = ttnn.open_device(device_id=dev_id, trace_region_size=64 << 20)
grid = dev.compute_with_storage_grid_size(); ncores = grid.x * grid.y
print(f"device {dev_id} grid {grid.x}x{grid.y} = {ncores} cores", flush=True)
shape = (32 * 8, 32 * 8 * ncores)  # 64 tiles per core, one contiguous 64-tile run per core
tiles_per_core = (shape[0] // 32) * (shape[1] // 32) // ncores

def rand(seed=None, dtype=ttnn.float32, shp=shape):
    kw = {"seed": seed} if seed is not None else {}
    return to_np(ttnn.rand(shp, device=dev, dtype=dtype, **kw))

def tiles(a):  # -> (ntiles, 32, 32) in the row-major tile order the writer uses
    R, C = a.shape
    return a.reshape(R // 32, 32, C // 32, 32).transpose(0, 2, 1, 3).reshape(-1, 32, 32)

a5, a5b, a6, u1, u2 = rand(5), rand(5), rand(6), rand(), rand()
print(f"seed=5 twice identical: {np.array_equal(a5, a5b)}; unseeded twice identical: {np.array_equal(u1, u2)}", flush=True)
t5, t6 = tiles(a5), tiles(a6)
shift = np.array_equal(t6[: -tiles_per_core], t5[tiles_per_core:])
print(f"seed=6 is seed=5 shifted by one core's tiles ({tiles_per_core} tiles): {shift}", flush=True)
# how many tiles of seed=6 appear anywhere in seed=5
keys5 = {t.tobytes() for t in t5}
print(f"tiles of seed=6 that also occur in seed=5: {sum(t.tobytes() in keys5 for t in t6)} of {len(t6)}", flush=True)
print(f"duplicate tiles within one seeded call: {len(t5) - len(keys5)}", flush=True)
# distribution
x = a5.ravel()
q = x * 2**23
print(f"range [{x.min():.3e}, {x.max():.8f}); mean {x.mean():.5f} (0.5); var {x.var():.5f} (0.08333); values are multiples of 2^-23: {np.all(q == np.round(q))}; "
      f"distinct values {len(np.unique(x)):,} of {x.size:,}", flush=True)
# lane structure: correlation between consecutive tiles and between rows/cols of a tile
c = np.corrcoef(t5[0].ravel(), t5[1].ravel())[0, 1]
print(f"corr(tile0, tile1) on one core: {c:+.4f}; corr(row0, row1) within tile0: {np.corrcoef(t5[0][0], t5[0][1])[0,1]:+.4f}", flush=True)
# a different shape with the same seed: same numbers?
b5 = rand(5, shp=(32 * 4, 32 * 8 * ncores))
print(f"same seed, half the rows: first tile equal: {np.array_equal(tiles(b5)[0], t5[0])}", flush=True)
# bf16 output
bb = rand(5, dtype=ttnn.bfloat16)
print(f"bfloat16 output: max {bb.max():.6f}, count == 1.0: {(bb == 1.0).sum():,} of {bb.size:,} (doc says [0, 1)); distinct {len(np.unique(bb))}", flush=True)
# low/high
lh = to_np(ttnn.rand(shape, device=dev, dtype=ttnn.float32, low=-2.0, high=3.0, seed=5))
print(f"low=-2 high=3: range [{lh.min():.4f}, {lh.max():.4f}]", flush=True)
# trace: seeded and unseeded calls inside a trace
for label, kw in (("seeded", {"seed": 7}), ("unseeded", {})):
    out = ttnn.rand(shape, device=dev, dtype=ttnn.float32, **kw)  # compile
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    ttnn.rand(shape, device=dev, dtype=ttnn.float32, output_tensor=out, **kw) if False else None
    r = ttnn.rand(shape, device=dev, dtype=ttnn.float32, **kw)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); r1 = to_np(r)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); r2 = to_np(r)
    print(f"trace {label}: replay 1 == replay 2: {np.array_equal(r1, r2)}", flush=True)
    ttnn.release_trace(dev, tid)
# timing
for shp in ((3520, 1024), (2752, 4096)):
    ttnn.rand(shp, device=dev, dtype=ttnn.float32, seed=1); ttnn.synchronize_device(dev)
    t0 = time.time()
    for i in range(10): ttnn.rand(shp, device=dev, dtype=ttnn.float32, seed=1 + i)
    ttnn.synchronize_device(dev); dt = (time.time() - t0) / 10
    print(f"rand float32 {shp}: {dt*1e3:.2f} ms ({shp[0]*shp[1]*4/dt/1e9:.0f} GB/s written)", flush=True)
ttnn.close_device(dev)
