"""Tenstorrent backend for the hypergraph population search (algorithm and layout: hyperpop.py).

Per device, with P replicas as *columns*:
  cnt   (I x P)  bfloat16   per-slot counts minus (r-1); near == 0, full == 1
  X     (n x P)  bfloat16   deck membership;  tabu (n x P), best deck (n x P)
  score (1 x P)  float32    maintained incrementally from the selected gain / loss
Constants: Hinc (n x I) bfloat16 row-major gather table (row v = 1 on every slot of
every combo containing v), S (n x chunks) 0/1 chunk-ownership matrix, index
tables for exact selection.

The per-card reduction ("combos completed by adding v") is a tile-row sum over
32 slots (exact in bfloat16) followed by matmul(S, partial) with fp32
accumulation and float32 output, exact for the integer counts involved (up to
the maximum degree, 5,713). Selection over values that exceed 256 cannot use
a single bfloat16 max, so it maximises (value // 256, value % 256, noise,
index) lexicographically with five exact max reductions.

Trace replay: with ``trace=True`` a block of ``gens_per_trace`` generations is
captured once (all state updated in place through ``output_tensor``) and then
replayed with ``ttnn.execute_trace``, which removes the per-op host dispatch
cost that otherwise dominates the ~45 small card-space ops of a generation.
``ttnn.rand`` inside a trace replays the same numbers, so the tie-break noise
is ``frac(rand + salt)`` with a per-row salt that advances as a Weyl sequence
on the device every generation.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from .hyper import HyperInstance
from .hyperpop import BIGV, HyperPopulation

TT_METAL_HOME_DEFAULT = "/home/ttuser/sjameel/tt-metal"
WEYL = 0.6180339887


def _import_ttnn():
    home = os.environ.setdefault("TT_METAL_HOME", TT_METAL_HOME_DEFAULT)
    for p in (f"{home}/ttnn", home):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ttnn  # noqa: E402

    return ttnn


class TTHyperPopulation(HyperPopulation):
    def __init__(self, inst: HyperInstance, k: int, P: int, devices: int = 1, device_ids: list[int] | None = None,
                 fidelity: str = "hifi2", s_dtype: str = "bfloat16", trace: bool = True, gens_per_trace: int = 4,
                 trace_region_size: int = 256 << 20, fused_gather: bool = True, **kwargs):
        t_init = time.time()
        super().__init__(inst, k, P, **kwargs)
        t_layout = time.time() - t_init
        assert self.layout.I % 2048 == 0, "fused embedding gather needs a slot width that is a multiple of 2048"
        self.fused_gather = fused_gather
        self.ttnn = ttnn = _import_ttnn()
        self.torch = __import__("torch")
        self.n_devices = devices
        self.use_trace, self.G = trace, gens_per_trace
        assert P % (32 * devices) == 0, "population must be a multiple of 32 per device"
        if devices > 1:
            self.device = ttnn.open_mesh_device(ttnn.MeshShape(1, devices), trace_region_size=trace_region_size)
            self._shard = ttnn.ShardTensorToMesh(self.device, dim=1)
            self._repl = ttnn.ReplicateTensorToMesh(self.device)
            self._concat = ttnn.ConcatMeshToTensor(self.device, dim=1)
        else:
            self.device = ttnn.open_device(device_id=(device_ids or [0])[0], trace_region_size=trace_region_size)
            self._shard = self._repl = self._concat = None
        t_open = time.time() - t_init - t_layout
        self.tile, self.rm = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
        self.bf16, self.f32, self.u32 = ttnn.bfloat16, ttnn.float32, ttnn.uint32
        self.transfer_seconds = 0.0
        fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[fidelity]
        self.mm_config = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=True)
        L = self.layout
        t0 = time.time()
        self.dH = self._up(L.H, self.bf16, layout=self.rm, replicate=True)
        sdt = {"bfloat16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[s_dtype]
        self.dS = self._up(L.S, sdt, replicate=True)
        # tiled copy of Hinc plus the slot offsets and first-slot mask: reseeding recomputes cnt and score
        # on the device (cnt = Hinc^T X + offset) instead of uploading a rebuilt (I x P) tensor from the host
        self.dHt = ttnn.to_layout(self.dH, self.tile)  # tilized on the device: no second 1.5 GB upload
        self.doffset = self._up(L.offset[:, None], self.bf16, replicate=True)
        first = np.zeros((1, L.I), dtype=np.float32)
        first[0, L.edge_slots[L.edge_slots_ptr[:-1]]] = 1.0
        self.dfirst = self._up(first, self.bf16, replicate=True)
        n, Pd = self.n, P // devices
        self.Pd = Pd
        self.dvalid = self._up(np.repeat(L.valid[:, None].astype(np.float32), Pd, axis=1), self.bf16, replicate=True)
        idx = np.arange(n, dtype=np.float32)
        self.dihi = self._up(np.repeat((idx // 32 + 1)[:, None], Pd, axis=1), self.bf16, replicate=True)
        self.dilo = self._up(np.repeat((idx % 32 + 1)[:, None], Pd, axis=1), self.bf16, replicate=True)
        self._seed = kwargs.get("seed", 0) * 1000
        salt = np.random.default_rng(self._seed + 1).random((n, devices), dtype=np.float32)
        self.dsalt = ttnn.slice(self._up(np.repeat(salt, 32, axis=1), self.f32), [0, 0], [n, 1])  # (n,1) per device, persistent
        self._dstate = None
        self._trace_id = None
        print(f"[tt] setup: host layout + incidence table {t_layout:.1f}s, device open {t_open:.1f}s, constants uploaded {time.time() - t0:.1f}s "
              f"(Hinc {n}x{L.I} = {n * L.I * 2 / 1e9:.2f} GB, twice; S {n}x{L.chunks}); P/device {Pd}", flush=True)

    # -- transfers ---------------------------------------------------------
    def _host(self, arr: np.ndarray, dtype, layout=None, replicate: bool = False):
        t = self.torch.from_numpy(np.ascontiguousarray(arr))
        kw = {}
        if self._shard is not None:
            kw["mesh_mapper"] = self._repl if replicate else self._shard
        return self.ttnn.from_torch(t, dtype=dtype, layout=layout or self.tile, **kw)

    def _up(self, arr: np.ndarray, dtype, layout=None, replicate: bool = False):
        t0 = time.time()
        out = self.ttnn.to_device(self._host(arr, dtype, layout, replicate), self.device)
        self.transfer_seconds += time.time() - t0
        return out

    def _write(self, arr: np.ndarray, dtype, dst) -> None:
        """Overwrite a persistent device tensor in place (the buffer stays put, so trace replays remain valid).
        On a mesh the host-side tilize of a sharded tensor is slow (260 ms for 2400x4096 bf16), so the data
        goes up row-major and is tilized on the device, then copied into the persistent buffer."""
        nn = self.ttnn
        t0 = time.time()
        if self._shard is None or arr.shape[0] < 32:
            nn.copy_host_to_device_tensor(self._host(arr, dtype), dst)
        else:
            rm = nn.to_device(self._host(arr, dtype, layout=self.rm), self.device)
            tiled = nn.to_layout(rm, self.tile)
            nn.add(tiled, 0.0, output_tensor=dst)
            nn.deallocate(tiled)
            nn.deallocate(rm)
        self.transfer_seconds += time.time() - t0

    def _down(self, t) -> np.ndarray:
        """Device -> host. `to_torch` with a mesh composer takes ~2 s for a 2400x4096 bf16 tensor on the 1x4 mesh
        (17 ms for one device's shard), so the shards are read one device at a time and concatenated on the host."""
        nn = self.ttnn
        t0 = time.time()
        if self._concat is None:
            out = nn.to_torch(t).float().numpy()
        else:
            parts = [nn.to_torch(s) for s in nn.get_device_tensors(nn.from_device(t))]
            out = self.torch.cat(parts, dim=1).float().numpy()
        self.transfer_seconds += time.time() - t0
        return out

    def _noise(self, seed: int):
        """(n x P/device) bf16 in [0,1): frac(rand(seed) + salt). Inside a trace `rand` replays the same numbers;
        the salt (advanced per generation by `_advance_salt`) makes every replay differ. Only the top 7 fraction
        bits survive `1 + noise` in bfloat16, so the tie-break resolution is 1/128 as in population.py."""
        nn = self.ttnn
        u = nn.rand((self.n, self.Pd), device=self.device, dtype=self.f32, seed=seed)
        u = nn.add(u, self.dsalt)
        u = nn.subtract(u, nn.floor(u))
        return nn.typecast(u, self.bf16)

    def _advance_salt(self):
        nn = self.ttnn
        t = nn.add(self.dsalt, WEYL)
        nn.subtract(t, nn.floor(t), output_tensor=self.dsalt)

    # -- building blocks -----------------------------------------------------
    def _segsum(self, flag):
        """(I x P) bf16 0/1 -> (n x P) f32 per-card sums."""
        nn = self.ttnn
        part = nn.sum(nn.reshape(flag, (self.layout.chunks, 32, self.Pd)), dim=-2)  # (chunks, P) exact (<= 32)
        return nn.matmul(self.dS, part, dtype=self.f32, compute_kernel_config=self.mm_config)

    def _select(self, W, noise):
        """W (n x P) f32 non-negative integers (0 = not selectable). Returns (one-hot bf16 (n x P), max value f32 (1 x P))."""
        nn = self.ttnn
        hi = nn.floor(nn.multiply(W, 1.0 / 256))
        lo = nn.subtract(W, nn.multiply(hi, 256.0))
        hib = nn.typecast(hi, self.bf16)
        lob = nn.typecast(nn.add(lo, 1.0), self.bf16)
        m1 = nn.max(hib, dim=-2, keepdim=True)
        c = nn.eq(hib, m1)
        key = nn.multiply(c, lob)
        m2 = nn.max(key, dim=-2, keepdim=True)
        c = nn.eq(key, m2)
        key = nn.multiply(c, nn.add(noise, 1.0))
        c = nn.eq(key, nn.max(key, dim=-2, keepdim=True))
        key = nn.multiply(c, self.dihi)
        c = nn.eq(key, nn.max(key, dim=-2, keepdim=True))
        key = nn.multiply(c, self.dilo)
        hot = nn.eq(key, nn.max(key, dim=-2, keepdim=True))
        value = nn.add(nn.multiply(nn.typecast(m1, self.f32), 256.0), nn.subtract(nn.typecast(m2, self.f32), 1.0))
        return hot, value

    def _gather_rows(self, hot):
        """one-hot (n x P) -> Hinc rows of the selected cards, transposed to (I x P) bf16."""
        nn = self.ttnn
        hi = nn.sum(nn.multiply(hot, self.dihi), dim=-2, keepdim=True)  # (1, P) exact small ints
        lo = nn.sum(nn.multiply(hot, self.dilo), dim=-2, keepdim=True)
        idx = nn.add(nn.multiply(nn.typecast(hi, self.f32), 32.0), nn.subtract(nn.typecast(lo, self.f32), 33.0))
        idx = nn.to_layout(nn.typecast(idx, self.u32), self.rm)
        if self.fused_gather:
            # fused tile-output gather: safe only because SlotLayout pads I to a multiple of 2048
            # (notes/ttnn-embedding-tile-chunk-alignment-bug.md); otherwise rows past the first 32-row
            # block on a core come back wrong, or the board hangs.
            rows = nn.reshape(nn.embedding(idx, self.dH, layout=self.tile), (self.Pd, self.layout.I))
        else:
            rows = nn.embedding(idx, self.dH, layout=self.rm)  # (1, P, I) row-major: exact at every width
            rows = nn.to_layout(nn.reshape(rows, (self.Pd, self.layout.I)), self.tile)
        return nn.transpose(rows, -2, -1)

    def _device_generation(self, X, cnt, tabu, score, noise_out=None, noise_in=None, seed: int = 0):
        """Functional form (fresh output tensors); used by the exactness check with host-supplied noise."""
        nn = self.ttnn
        tabu = nn.clamp(nn.subtract(tabu, 1.0), 0.0, float(self.tenure))
        free = nn.eq(tabu, 0.0)
        gain = self._segsum(nn.eq(cnt, 0.0))
        out_ok = nn.typecast(nn.multiply(nn.multiply(nn.eq(X, 0.0), self.dvalid), free), self.f32)
        Ob, wb = self._select(nn.multiply(nn.add(gain, 1.0), out_ok), noise_out if noise_out is not None else self._noise(seed))
        cnt = nn.add(cnt, self._gather_rows(Ob))
        loss = self._segsum(nn.eq(cnt, 1.0))
        in_ok = nn.typecast(nn.multiply(nn.eq(X, 1.0), free), self.f32)
        Oa, wa = self._select(nn.multiply(nn.subtract(nn.multiply(loss, -1.0), -float(BIGV)), in_ok), noise_in if noise_in is not None else self._noise(seed + 1))
        cnt = nn.subtract(cnt, self._gather_rows(Oa))
        X = nn.subtract(nn.add(X, Ob), Oa)
        tabu = nn.add(tabu, nn.multiply(nn.add(Ob, Oa), float(self.tenure)))
        score = nn.add(score, nn.subtract(nn.add(wb, wa), float(BIGV + 1)))  # (wb-1) - (BIGV-wa)
        return X, cnt, tabu, score

    def _generation_inplace(self, seed: int):
        """Same step, but every state tensor is updated through output_tensor= so a trace can replay it."""
        nn = self.ttnn
        X, cnt, tabu, score, best, bestX = self._dX, self._dcnt, self._dtabu, self._dscore, self._dbest, self._dbestX
        tabu_d = nn.clamp(nn.subtract(tabu, 1.0), 0.0, float(self.tenure))
        free = nn.eq(tabu_d, 0.0)
        gain = self._segsum(nn.eq(cnt, 0.0))
        out_ok = nn.typecast(nn.multiply(nn.multiply(nn.eq(X, 0.0), self.dvalid), free), self.f32)
        Ob, wb = self._select(nn.multiply(nn.add(gain, 1.0), out_ok), self._noise(seed))
        nn.add(cnt, self._gather_rows(Ob), output_tensor=cnt)
        loss = self._segsum(nn.eq(cnt, 1.0))
        in_ok = nn.typecast(nn.multiply(nn.eq(X, 1.0), free), self.f32)
        Oa, wa = self._select(nn.multiply(nn.subtract(nn.multiply(loss, -1.0), -float(BIGV)), in_ok), self._noise(seed + 1))
        nn.subtract(cnt, self._gather_rows(Oa), output_tensor=cnt)
        nn.subtract(nn.add(X, Ob), Oa, output_tensor=X)
        nn.add(tabu_d, nn.multiply(nn.add(Ob, Oa), float(self.tenure)), output_tensor=tabu)
        nn.add(score, nn.subtract(nn.add(wb, wa), float(BIGV + 1)), output_tensor=score)
        improved = nn.gt(score, best)  # (1, P) f32 0/1
        nn.add(best, nn.multiply(improved, nn.subtract(score, best)), output_tensor=best)
        m = nn.typecast(improved, self.bf16)
        nn.add(bestX, nn.multiply(m, nn.subtract(X, bestX)), output_tensor=bestX)  # row-broadcast bf16 0/1 mask: exact
        self._advance_salt()

    # -- epochs --------------------------------------------------------------
    def _init_state(self) -> None:
        pass  # cnt and score are computed on the device by _recount once the state tensors exist

    def _ensure_device_state(self) -> None:
        if self._dstate is None:
            self._dstate = True
            nn = self.ttnn
            self._dX = self._up(self.X, self.bf16)
            self._dtabu = self._up(self.tabu, self.bf16)
            self._dbest = self._up(np.full((1, self.P), -1.0, dtype=np.float32), self.f32)
            self._dbestX = self._up(self.X, self.bf16)
            self._dscore = self._up(np.zeros((1, self.P), dtype=np.float32), self.f32)
            self._dcnt = nn.zeros((self.layout.I, self.Pd), device=self.device, dtype=self.bf16, layout=self.tile)
            self._recount()
            nn.synchronize_device(self.device)

    def _block(self):
        for g in range(self.G):
            self._generation_inplace(self._seed + 2 * g)

    def _ensure_trace(self) -> None:
        if self._trace_id is not None or not self.use_trace:
            return
        nn = self.ttnn
        t0 = time.time()
        self._block()  # compile pass (also a real block of generations)
        nn.synchronize_device(self.device)
        self.generations += self.G
        t_compile = time.time() - t0
        self._trace_id = nn.begin_trace_capture(self.device, cq_id=0)
        self._block()  # recorded, not executed (verified: state is unchanged after capture)
        nn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        nn.synchronize_device(self.device)
        print(f"[tt] first eager block of {self.G} generations (kernel compile) {t_compile:.1f}s, trace capture {time.time() - t0 - t_compile:.1f}s", flush=True)

    def run_epoch(self, gens: int) -> int:
        nn = self.ttnn
        self._ensure_device_state()
        if self.use_trace:
            self._ensure_trace()
            for _ in range(max(1, gens // self.G)):
                nn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
                self.generations += self.G
            nn.synchronize_device(self.device)
        else:
            for g in range(gens):
                self._generation_inplace(self._seed + 2 * g)
                self.generations += 1
            nn.synchronize_device(self.device)
        best_now = self._down(self._dbest)[0].astype(np.int64)
        self._bookkeep(best_now, lambda p: self._down(self._dbestX)[:, p])
        return self.best_score

    def _recount(self) -> None:
        """cnt = Hinc^T X + offset and score = #complete combos, recomputed on the device from X into the
        persistent buffers. Every temporary is freed again so the trace's intermediate buffers stay where
        they were captured."""
        nn = self.ttnn
        XT = nn.transpose(self._dX, -2, -1)  # (P, n)
        R = nn.matmul(XT, self.dHt, dtype=self.bf16, compute_kernel_config=self.mm_config)  # (P, I) counts <= 10: exact
        RT = nn.transpose(R, -2, -1)
        nn.add(RT, self.doffset, output_tensor=self._dcnt)
        full = nn.eq(self._dcnt, 1.0)
        score = nn.matmul(self.dfirst, full, dtype=self.f32, compute_kernel_config=self.mm_config)  # (1, P) fp32 accumulation: exact
        nn.add(score, 0.0, output_tensor=self._dscore)
        for t in (XT, R, RT, full, score):
            nn.deallocate(t)

    def _reseed(self, stale: np.ndarray, cols: np.ndarray) -> None:
        """Replace stale replicas: X comes back, the new columns go in, counts and scores are rebuilt on the device,
        and every persistent buffer is overwritten in place (no reallocation between trace replays)."""
        nn = self.ttnn
        self.X = self._down(self._dX)
        self.X[:, stale] = cols
        self._write(self.X, self.bf16, self._dX)
        self._recount()
        nn.multiply(self._dtabu, 0.0, output_tensor=self._dtabu)
        mask = np.zeros((1, self.P), dtype=np.float32)  # 1 on reseeded replicas: best <- -1, bestX <- new deck, on the device
        mask[0, stale] = 1.0
        dm = self._up(mask, self.f32)
        nn.add(nn.multiply(self._dbest, nn.subtract(nn.multiply(dm, -1.0), -1.0)), nn.multiply(dm, -1.0), output_tensor=self._dbest)
        mb = self.ttnn.typecast(dm, self.bf16)
        nn.add(self._dbestX, nn.multiply(mb, nn.subtract(self._dX, self._dbestX)), output_tensor=self._dbestX)
        nn.deallocate(mb)
        nn.deallocate(dm)

    def close(self) -> None:
        if self._trace_id is not None:
            self.ttnn.release_trace(self.device, self._trace_id)
        if self.n_devices > 1:
            self.ttnn.close_mesh_device(self.device)
        else:
            self.ttnn.close_device(self.device)
