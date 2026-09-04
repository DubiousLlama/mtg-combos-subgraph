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
                 trace_region_size: int = 256 << 20, **kwargs):
        super().__init__(inst, k, P, **kwargs)
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
        print(f"[tt] constants uploaded in {time.time() - t0:.1f}s: Hinc {n}x{L.I} ({n * L.I * 2 / 1e9:.2f} GB), S {n}x{L.chunks}, P/device {Pd}", flush=True)

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
        """Overwrite a persistent device tensor in place (no allocation; safe between trace replays)."""
        t0 = time.time()
        self.ttnn.copy_host_to_device_tensor(self._host(arr, dtype), dst)
        self.transfer_seconds += time.time() - t0

    def _down(self, t) -> np.ndarray:
        t0 = time.time()
        kw = {"mesh_composer": self._concat} if self._concat is not None else {}
        out = self.ttnn.to_torch(t, **kw).float().numpy()
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
        # NB: embedding(..., layout=TILE_LAYOUT) returns wrong rows once a row exceeds 512 KB (checked); gather row-major, then tilize.
        rows = nn.embedding(idx, self.dH)  # (1, P, I) row-major
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
    def _ensure_device_state(self) -> None:
        if self._dstate is None:
            self._dstate = True
            self._dX = self._up(self.X, self.bf16)
            self._dcnt = self._up(self.cnt, self.bf16)
            self._dtabu = self._up(self.tabu, self.bf16)
            self._dscore = self._up(self.score[None, :], self.f32)
            self._dbest = self._up(np.full((1, self.P), -1.0, dtype=np.float32), self.f32)
            self._dbestX = self._up(self.X, self.bf16)

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
        self._trace_id = nn.begin_trace_capture(self.device, cq_id=0)
        self._block()  # recorded, not executed (verified: state is unchanged after capture)
        nn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        nn.synchronize_device(self.device)
        print(f"[tt] trace of {self.G} generations captured in {time.time() - t0:.1f}s", flush=True)

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

    def _reseed(self, stale: np.ndarray, cols: np.ndarray) -> None:
        """Replace stale replicas: X comes back, counts/scores are rebuilt on the host, persistent buffers are overwritten in place."""
        self.X = self._down(self._dX)
        self.X[:, stale] = cols
        self.cnt = self.layout.counts(self.X)
        self.score = self.layout.scores(self.X).astype(np.float32)
        self._write(self.X, self.bf16, self._dX)
        self._write(self.cnt, self.bf16, self._dcnt)
        self._write(self.score[None, :], self.f32, self._dscore)
        self._write(np.zeros((self.n, self.P), dtype=np.float32), self.bf16, self._dtabu)
        best = self._down(self._dbest)
        best[0, stale] = -1.0
        self._write(best, self.f32, self._dbest)
        bestX = self._down(self._dbestX)
        bestX[:, stale] = cols
        self._write(bestX, self.bf16, self._dbestX)

    def close(self) -> None:
        if self._trace_id is not None:
            self.ttnn.release_trace(self.device, self._trace_id)
        if self.n_devices > 1:
            self.ttnn.close_mesh_device(self.device)
        else:
            self.ttnn.close_device(self.device)
