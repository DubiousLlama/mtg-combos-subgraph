"""Tenstorrent backend for the population search (see population.py for the algorithm).

Runs `spellbook_graph.population.generation` as ttnn ops on one Blackhole or on
a 1xN mesh of them, sharding the population across devices. Exactness notes,
measured on this build (tt-metal Feb 2026, Blackhole):

- matmul of 0/1 bfloat16 operands is exact as long as outputs are < 256, which
  holds because C[p, v] <= max degree (86 here) and k = 50.
- ttnn.sum/max round their *inputs* to bfloat16 but accumulate in fp32, so
  reductions of small integers are exact while reductions of int+noise are not.
  Selection therefore uses ttnn.argmax on float32 (verified exact) and the
  score uses ttnn.sum over integer-valued tensors.
- All 0/1 and small-integer state (X, tabu) lives in bfloat16; value tensors
  that carry tie-breaking noise are float32.

Design note for the 3+ card (hypergraph) follow-up: `_device_generation` only
needs two things from the graph, gain(X) = "combos completed by adding v" and
loss(X) = "combos broken by dropping v". For two-card combos both are X @ A.
For an incidence matrix H (n x E) with combo sizes r they become
[X @ H == r - 1] @ H^T and [X @ H == r] @ H^T, the same op mix at larger shape,
which is where the matmul throughput of the cards starts to matter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .population import BIG, NOISE_LEVELS, PopulationSearch

TT_METAL_HOME_DEFAULT = "/home/ttuser/sjameel/tt-metal"


def _import_ttnn():
    home = os.environ.setdefault("TT_METAL_HOME", TT_METAL_HOME_DEFAULT)
    for p in (f"{home}/ttnn", home):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ttnn  # noqa: E402

    return ttnn


class TTPopulation(PopulationSearch):
    def __init__(self, *args, devices: int = 1, device_ids: list[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ttnn = ttnn = _import_ttnn()
        self.torch = __import__("torch")
        self.n_devices = devices
        if devices > 1:
            self.device = ttnn.open_mesh_device(ttnn.MeshShape(1, devices))
            self._shard = ttnn.ShardTensorToMesh(self.device, dim=0)
            self._repl = ttnn.ReplicateTensorToMesh(self.device)
            self._concat = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.device = ttnn.open_device(device_id=(device_ids or [0])[0])
            self._shard = self._repl = self._concat = None
        assert self.P % (32 * devices) == 0, "population must be a multiple of 32 per device"
        self.tile = ttnn.TILE_LAYOUT
        self.transfer_seconds = 0.0
        self.bf16, self.f32 = ttnn.bfloat16, ttnn.float32
        # constants, replicated
        self.dA = self._up(self.A, self.bf16, replicate=True)
        self.dbonus = self._up(self.bonus[None, :], self.bf16, replicate=True)
        self.dvalid = self._up(self.valid[None, :].astype(np.float32), self.bf16, replicate=True)
        idx = np.arange(self.n, dtype=np.float32)
        self.dhi = self._up((idx // 32 + 1)[None, :], self.bf16, replicate=True)
        self.dlo = self._up((idx % 32 + 1)[None, :], self.bf16, replicate=True)
        self._seed = 0
        # ttnn.rand gives every device of a mesh the same stream for a given seed;
        # a per-device salt row (sharded, so different per device) decorrelates them.
        salt = np.random.default_rng(kwargs.get("seed", 0) + 1).random((self.n_devices, self.n), dtype=np.float32)
        block = self._up(np.repeat(salt, 32, axis=0), self.f32)  # (32*devices, n) sharded -> (32, n) per device
        self.dsalt_row = ttnn.slice(block, [0, 0], [1, self.n])  # (1, n) per device

    # -- transfers ----------------------------------------------------------
    def _up(self, arr: np.ndarray, dtype, replicate: bool = False):
        t0 = time.time()
        t = self.torch.from_numpy(np.ascontiguousarray(arr))
        kw = {}
        if self._shard is not None:
            kw["mesh_mapper"] = self._repl if replicate else self._shard
        out = self.ttnn.from_torch(t, dtype=dtype, layout=self.tile, device=self.device, **kw)
        self.transfer_seconds += time.time() - t0
        return out

    def _down(self, t) -> np.ndarray:
        t0 = time.time()
        kw = {"mesh_composer": self._concat} if self._concat is not None else {}
        out = self.ttnn.to_torch(t, **kw).float().numpy()
        self.transfer_seconds += time.time() - t0
        return out

    def _noise(self):
        """(P/devices, n) bf16 tensor of multiples of 1/NOISE_LEVELS in [0,1), fresh from the device RNG."""
        nn = self.ttnn
        self._seed += 1
        u = nn.rand((self.P // self.n_devices, self.n), device=self.device, dtype=self.f32, seed=self._seed)
        u = nn.add(u, self.dsalt_row)  # frac(u + salt): same u on every device, different salt
        u = nn.subtract(u, nn.floor(u))
        q = nn.multiply(nn.floor(nn.multiply(u, float(NOISE_LEVELS))), 1.0 / NOISE_LEVELS)
        return nn.typecast(q, self.bf16)

    def _select_max(self, M, noise):
        """One-hot of the per-row max of M (bf16, small integers); ties broken by noise then by index.

        Exact `max` reductions only: bf16 keys, no argmax (see module docstring).
        """
        nn = self.ttnn
        cand = nn.eq(M, nn.max(M, dim=-1, keepdim=True))
        key = nn.multiply(cand, nn.add(noise, 1.0))
        hot = nn.eq(key, nn.max(key, dim=-1, keepdim=True))
        kh = nn.multiply(hot, self.dhi)
        hot = nn.eq(kh, nn.max(kh, dim=-1, keepdim=True))
        kl = nn.multiply(hot, self.dlo)
        return nn.eq(kl, nn.max(kl, dim=-1, keepdim=True))

    # -- one generation on device -------------------------------------------
    def _device_generation(self, X, tabu, noise_out=None, noise_in=None):
        """Mirror of population.generation. Returns (X, score, tabu) as device tensors (all bf16)."""
        nn = self.ttnn
        C = nn.matmul(X, self.dA)  # exact small ints
        V = nn.add(C, self.dbonus)
        # X*(V+bonus) = 2*edges + 2*commander combos per row: small ints, summed exactly in fp32
        score = nn.multiply(nn.sum(nn.typecast(nn.multiply(X, nn.add(V, self.dbonus)), self.f32), dim=-1, keepdim=True), 0.5)
        tabu = nn.clamp(nn.subtract(tabu, 1.0), 0.0, float(self.tenure))
        free = nn.eq(tabu, 0.0)
        out_ok = nn.multiply(nn.multiply(nn.eq(X, 0.0), self.dvalid), free)
        Vout = nn.where(out_ok, V, -float(BIG))
        Ob = self._select_max(Vout, noise_out if noise_out is not None else self._noise())
        Rb = nn.matmul(Ob, self.dA)
        in_ok = nn.multiply(nn.eq(X, 1.0), free)
        Vin = nn.where(in_ok, nn.neg(nn.add(V, Rb)), -float(BIG))
        Oa = self._select_max(Vin, noise_in if noise_in is not None else self._noise())
        X = nn.subtract(nn.add(X, Ob), Oa)
        tabu = nn.add(tabu, nn.multiply(nn.add(Ob, Oa), float(self.tenure)))
        return X, score, tabu

    # -- epochs: X, tabu, per-replica best score and best deck stay on device --
    def _ensure_device_state(self) -> None:
        if getattr(self, "_dX", None) is None:
            self._dX = self._up(self.X, self.bf16)
            self._dtabu = self._up(self.tabu, self.bf16)
            self._dbest = self._up(np.full((self.P, 1), -1.0, dtype=np.float32), self.f32)
            self._dbest_X = self._dX

    def run_epoch(self, gens: int) -> int:
        nn = self.ttnn
        self._ensure_device_state()
        X, tabu, best, best_X = self._dX, self._dtabu, self._dbest, self._dbest_X
        for _ in range(gens):
            X_before = X
            X, score, tabu = self._device_generation(X, tabu)
            improved = nn.gt(score, best)  # (P,1) float32
            best = nn.where(improved, score, best)
            # NB: ttnn.where silently mis-selects with a float32 mask over bfloat16 payloads; cast the mask.
            best_X = nn.where(nn.typecast(improved, self.bf16), X_before, best_X)
        self._dX, self._dtabu, self._dbest, self._dbest_X = X, tabu, best, best_X
        self.generations += gens
        best_now = self._down(best)[:, 0].astype(np.int64)  # (P,) -- the only per-epoch transfer
        self._bookkeep(best_now, lambda i: self._down(self._dbest_X)[i])
        return self.best_score

    def _reseed(self, stale: np.ndarray, rows: np.ndarray) -> None:
        """Patch stale rows of the device-resident population (full X round trip, only on reseed epochs)."""
        nn = self.ttnn
        self.X = self._down(self._dX)
        self.X[stale] = rows
        self._dX = self._up(self.X, self.bf16)
        self._dtabu = nn.multiply(self._dtabu, 0.0)  # tabu reset for everyone is harmless
        mask = np.ones((self.P, 1), dtype=np.float32)
        mask[stale] = 0.0
        keep = self._up(mask, self.f32)
        self._dbest = nn.add(nn.multiply(self._dbest, keep), nn.subtract(keep, 1.0))  # kept: best, stale: -1
        self._dbest_X = nn.where(nn.typecast(keep, self.bf16), self._dbest_X, self._dX)

    def close(self) -> None:
        if self.n_devices > 1:
            self.ttnn.close_mesh_device(self.device)
        else:
            self.ttnn.close_device(self.device)
