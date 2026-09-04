"""Population-parallel tabu search, written as dense tensor ops.

This is the algorithm the Tenstorrent backend (`tt_search.py`) runs; this file
is the NumPy reference used by the tests and as a CPU fallback. Every replica
of the population takes one swap move per generation, and the whole
neighbourhood of every replica is evaluated with one matmul:

    C = X @ A          C[p, v] = number of cards in deck p that combo with v

For deck p, adding card b gains V[b] = C[b] + bonus[b]; dropping card a loses
V[a] + A[a, b]. Each generation picks the best add b* (random tie-break, tabu
respected), fetches the row A[b*, :] with a second one-hot matmul, then picks
the best drop given b*. That is exact for "best drop given best add" and within
1 of the true best swap, which is plenty for a population search.

Cost per generation: two (P x n) @ (n x n) matmuls = 4 P n^2 flops plus O(P n)
elementwise work. With n = 1664 (1623 cards padded to a tile multiple) and
P = 4096 replicas that is 45 GFLOP, a few ms on one Blackhole.

Restarts and intensification happen on the host every `epoch` generations:
replicas that have not improved for a while are re-seeded either randomly or
as a kicked copy of the best deck found so far.
"""

from __future__ import annotations

import numpy as np

BIG = 1 << 16


def pad_to_tiles(n: int, tile: int = 32) -> int:
    return ((n + tile - 1) // tile) * tile


def random_decks(rng: np.random.Generator, P: int, n_real: int, k: int, n_pad: int, weight: np.ndarray | None = None) -> np.ndarray:
    """P random k-subsets of range(n_real) as a (P, n_pad) 0/1 float32 matrix.

    Weighted sampling without replacement via the Efraimidis-Spirakis trick
    (key = u^(1/w), keep the k largest), vectorised over replicas.
    """
    X = np.zeros((P, n_pad), dtype=np.float32)
    if P == 0:
        return X
    keys = rng.random((P, n_real), dtype=np.float32)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float32) + 1e-3
        keys = keys ** (1.0 / w)[None, :]
    top = np.argpartition(-keys, k - 1, axis=1)[:, :k]
    np.put_along_axis(X, top, 1.0, axis=1)
    return X


def kick_decks(rng: np.random.Generator, base: np.ndarray, P: int, n_real: int, strength: int, n_pad: int) -> np.ndarray:
    """P copies of `base` with `strength` random members swapped for random outsiders."""
    X = np.repeat(base[None, :], P, axis=0).astype(np.float32)
    if P == 0:
        return X
    inside = np.flatnonzero(base[:n_real] > 0)
    outside = np.flatnonzero(base[:n_real] == 0)
    drop = inside[np.argpartition(rng.random((P, inside.size)), strength - 1, axis=1)[:, :strength]]
    add = outside[np.argpartition(rng.random((P, outside.size)), strength - 1, axis=1)[:, :strength]]
    np.put_along_axis(X, drop, 0.0, axis=1)
    np.put_along_axis(X, add, 1.0, axis=1)
    return X


def scores(X: np.ndarray, A: np.ndarray, bonus: np.ndarray) -> np.ndarray:
    C = X @ A
    return ((X * C).sum(axis=1) / 2 + X @ bonus).astype(np.int64)


NOISE_LEVELS = 128  # tie-break noise is a multiple of 1/128 so 1+noise is exact in bfloat16


def quantize_noise(u: np.ndarray) -> np.ndarray:
    """Uniform [0,1) floats -> multiples of 1/NOISE_LEVELS (what the device uses)."""
    return np.floor(u * NOISE_LEVELS) / NOISE_LEVELS


def select_max(M: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Per row: the column holding the max of M, ties broken by `noise` (higher wins),
    remaining ties by the larger index. Returns a one-hot matrix.

    Mirrors the device implementation, which cannot use argmax cheaply: three
    exact max-reductions over bfloat16-representable keys.
    """
    P, n = M.shape
    m = M.max(axis=1, keepdims=True)
    cand = (M == m).astype(np.float32)
    key = cand * (1.0 + noise)
    hot = key == key.max(axis=1, keepdims=True)
    # largest index among survivors == argmax of the reversed row
    col = (n - 1) - np.argmax(hot[:, ::-1], axis=1)
    out = np.zeros_like(M, dtype=np.float32)
    out[np.arange(P), col] = 1.0
    return out


def generation(
    X: np.ndarray,
    A: np.ndarray,
    bonus: np.ndarray,
    valid: np.ndarray,
    tabu: np.ndarray,
    tenure: int,
    noise_out: np.ndarray,
    noise_in: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One swap per replica. Returns (new X, scores before the move, b*, a*, new tabu).

    `tabu` counts down: a card is tabu while tabu[p, v] > 0, and a move sets
    the two cards involved to `tenure`. Small integers so it fits in bfloat16.
    `noise_*` are (P, n) arrays of multiples of 1/NOISE_LEVELS in [0, 1) used
    purely for random tie-breaking; they are parameters so the TT backend can
    be checked against this function with identical randomness.
    """
    C = X @ A
    V = C + bonus[None, :]
    score = 0.5 * (X * (V + bonus[None, :])).sum(axis=1)  # edges + commander combos
    tabu = np.maximum(tabu - 1, 0)
    free = tabu == 0
    Vout = np.where((X == 0) & valid[None, :] & free, V, -BIG)
    Ob = select_max(Vout, noise_out)
    Rb = Ob @ A
    Vin = np.where((X == 1) & free, -(V + Rb), -BIG)  # maximise the negative = drop the cheapest
    Oa = select_max(Vin, noise_in)
    X = X + Ob - Oa
    tabu = tabu + (Ob + Oa) * tenure
    b = Ob.argmax(axis=1)
    a = Oa.argmax(axis=1)
    return X, score.astype(np.int64), b, a, tabu


class PopulationSearch:
    """Host-side driver: owns the population, runs epochs, tracks the incumbent.

    Per-replica best score / best deck persist across epochs. After each epoch
    `_bookkeep` compares them with the previous epoch to find stale replicas and
    reseeds those (half as kicked copies of the incumbent, half at random) via
    `_reseed`, which backends override to patch device-resident state.
    """

    def __init__(self, A: np.ndarray, bonus: np.ndarray, k: int, P: int, seed: int = 0, tenure: int = 5, kick: int = 6, stale_epochs: int = 3):
        self.n_real = A.shape[0]
        self.n = pad_to_tiles(self.n_real)
        self.A = np.zeros((self.n, self.n), dtype=np.float32)
        self.A[: self.n_real, : self.n_real] = A
        self.bonus = np.zeros(self.n, dtype=np.float32)
        self.bonus[: self.n_real] = bonus
        self.valid = np.zeros(self.n, dtype=bool)
        self.valid[: self.n_real] = True
        self.k, self.P, self.tenure, self.kick, self.stale_epochs = k, P, tenure, kick, stale_epochs
        self.rng = np.random.default_rng(seed)
        self.degree = self.A.sum(axis=1)[: self.n_real] + self.bonus[: self.n_real]
        self.X = random_decks(self.rng, P, self.n_real, k, self.n, weight=self.degree)
        self.tabu = np.zeros((P, self.n), dtype=np.float32)
        self.best_score = -1
        self.best_deck: np.ndarray | None = None
        self.replica_best = np.full(P, -1, dtype=np.int64)  # as of the previous epoch
        self.replica_stale = np.zeros(P, dtype=np.int64)
        self.generations = 0
        self.last_epoch: dict = {}
        # numpy backend state: per-replica best so far
        self._best = np.full(P, -1, dtype=np.int64)
        self._best_X = self.X.copy()

    # -- one epoch = `gens` generations, then host bookkeeping
    def run_epoch(self, gens: int) -> int:
        for _ in range(gens):
            noise_out = quantize_noise(self.rng.random((self.P, self.n), dtype=np.float32))
            noise_in = quantize_noise(self.rng.random((self.P, self.n), dtype=np.float32))
            X_before = self.X
            self.X, score, _, _, self.tabu = generation(self.X, self.A, self.bonus, self.valid, self.tabu, self.tenure, noise_out, noise_in)
            improved = score > self._best
            self._best[improved] = score[improved]
            self._best_X[improved] = X_before[improved]
        self.generations += gens
        self._bookkeep(self._best.copy(), lambda i: self._best_X[i].copy())
        return self.best_score

    def _reseed(self, stale: np.ndarray, rows: np.ndarray) -> None:
        self.X[stale] = rows
        self.tabu[stale] = 0
        self._best[stale] = -1

    def _bookkeep(self, best_now: np.ndarray, fetch_best_row) -> None:
        """`best_now[p]`: best score replica p has seen since its last reseed; `fetch_best_row(p)` -> that deck."""
        improved = best_now > self.replica_best
        self.replica_best = best_now.copy()
        self.replica_stale = np.where(improved, 0, self.replica_stale + 1)
        top = int(best_now.argmax())
        if best_now[top] > self.best_score:
            self.best_score = int(best_now[top])
            self.best_deck = fetch_best_row(top)
            assert self.best_deck.sum() == self.k, "best deck is not a k-subset"
        stale = np.flatnonzero(self.replica_stale >= self.stale_epochs)
        self.last_epoch = {
            "epoch_best": int(best_now.max()),
            "replicas_at_best": int((best_now == self.best_score).sum()),
            "mean_best": float(best_now.mean()),
            "reseeded": int(stale.size),
        }
        if stale.size:
            half = stale.size // 2 if self.best_deck is not None else 0
            rows = np.concatenate([
                kick_decks(self.rng, self.best_deck, half, self.n_real, self.kick, self.n) if half else np.zeros((0, self.n), np.float32),
                random_decks(self.rng, stale.size - half, self.n_real, self.k, self.n, weight=self.degree),
            ])
            self._reseed(stale, rows)
            self.replica_best[stale] = -1
            self.replica_stale[stale] = 0
