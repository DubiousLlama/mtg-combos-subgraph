"""Population search on the combo hypergraph, written as the tensor ops the Tenstorrent backend runs.

Sparse ("incidence space") formulation. Every combo e with r_e cards owns r_e
consecutive *slots*; the slots of all combos containing card v form one
contiguous, 32-aligned range (cards are laid out one after another, each
padded to a multiple of 32 slots). Per replica p the state is

    cnt[i, p] = (number of deck cards in combo e_i) - (r_e - 1)

so a combo is one card short of complete ("near") where cnt == 0 and complete
("full") where cnt == 1. Adding card b adds Hinc[b] (1 on every slot of every
combo containing b) to the column; dropping subtracts it. Then

    gain[v, p] = #{e ∋ v : near}  = sum of near over v's slot range
    loss[v, p] = #{e ∋ v : full}  = sum of full over v's slot range

are segmented sums, done as a tile-row sum (32 slots, exact in bfloat16)
followed by a 0/1 matmul S (cards x chunks) with fp32 accumulation. The
swap itself is "best add, then best drop given that add", as in population.py.
Tensors are stored transposed (slots x replicas, cards x replicas) so that
these reductions run over the second-to-last axis of tile-aligned views.

This module is the NumPy reference: identical semantics, checked bit-for-bit
against tt_hyper.py with host-supplied tie-break noise.
"""

from __future__ import annotations

import numpy as np

from .hyper import HyperInstance
from .population import NOISE_LEVELS, pad_to_tiles, quantize_noise

BIGV = 65536  # drop selection maximises BIGV - loss; loss < BIGV
PAD_CNT = -9  # padding slots never match near (0) or full (1)
SLOT_ALIGN = 2048  # slot width multiple (see SlotLayout)


class SlotLayout:
    """Card-major slot layout of the hypergraph, plus the constant tables the device needs."""

    def __init__(self, inst: HyperInstance):
        self.inst = inst
        m, E = inst.m, inst.E
        r = inst.r
        # bonus (combos with the commander alone) become size-1 combos
        bonus_cards = np.repeat(np.arange(m), inst.bonus)
        e_cards = [inst.edge_cards[inst.edge_ptr[e]:inst.edge_ptr[e + 1]] for e in range(E)] + [np.array([c], dtype=np.int32) for c in bonus_cards]
        sizes = np.array([len(c) for c in e_cards], dtype=np.int64)
        self.E_all = len(e_cards)
        deg = np.zeros(m, dtype=np.int64)
        for cards in e_cards:
            deg[cards] += 1
        self.pad_deg = ((deg + 31) // 32) * 32
        self.pad_deg[deg == 0] = 0
        self.card_start = np.concatenate([[0], np.cumsum(self.pad_deg)])
        # slot width padded to a multiple of 2048: ttnn.embedding's fused tile-output path corrupts rows
        # when the row width is not a multiple of 2048 elements (notes/ttnn-embedding-tile-chunk-alignment-bug.md)
        self.I = -(-int(self.card_start[-1]) // SLOT_ALIGN) * SLOT_ALIGN
        self.n = pad_to_tiles(m)
        self.chunks = self.I // 32
        # slot -> (edge, card); slots ordered by card, then by edge id
        inc_e = np.concatenate([np.full(len(c), e) for e, c in enumerate(e_cards)])
        inc_v = np.concatenate(e_cards)
        order = np.lexsort((inc_e, inc_v))
        inc_e, inc_v = inc_e[order], inc_v[order]
        offsets = np.arange(len(inc_v)) - np.repeat(np.concatenate([[0], np.cumsum(deg)[:-1]]), deg)
        self.slot = self.card_start[inc_v] + offsets  # slot index of each (card, edge) incidence
        self.slot_edge = np.full(self.I, -1, dtype=np.int64)
        self.slot_edge[self.slot] = inc_e
        self.slot_card = np.full(self.I, -1, dtype=np.int64)
        self.slot_card[self.slot] = inc_v
        self.sizes = sizes
        real = self.slot_edge >= 0
        self.offset = np.where(real, -(sizes[np.where(real, self.slot_edge, 0)] - 1), PAD_CNT).astype(np.float32)
        # edges -> their slots (CSR) for the incidence table
        self.edge_slots_ptr = np.zeros(self.E_all + 1, dtype=np.int64)
        self.edge_slots_ptr[1:] = np.cumsum(sizes)
        self.edge_slots = self.slot[np.argsort(inc_e, kind="stable")]
        self.e_cards = e_cards
        self.valid = np.zeros(self.n, dtype=bool)
        self.valid[:m] = deg > 0

    def incidence_rows(self, dtype=np.float32) -> np.ndarray:
        """Hinc (n x I): row v has 1 on every slot of every combo containing v."""
        H = np.zeros((self.n, self.I), dtype=dtype)
        for e, cards in enumerate(self.e_cards):
            slots = self.edge_slots[self.edge_slots_ptr[e]:self.edge_slots_ptr[e + 1]]
            for v in cards:
                H[v, slots] = 1
        return H

    def chunk_matrix(self, dtype=np.float32) -> np.ndarray:
        """S (n x chunks): S[v, j] = 1 if 32-slot chunk j lies in card v's range."""
        S = np.zeros((self.n, self.chunks), dtype=dtype)
        for v in range(self.inst.m):
            S[v, self.card_start[v] // 32: self.card_start[v + 1] // 32] = 1
        return S

    def _incidence_csr(self):
        if getattr(self, "_inc", None) is None:
            from scipy.sparse import csr_matrix
            rows = np.concatenate([np.full(len(c), e) for e, c in enumerate(self.e_cards)])
            cols = np.concatenate(self.e_cards)
            self._inc = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.E_all, self.n))
        return self._inc

    def counts(self, X: np.ndarray) -> np.ndarray:
        """cnt (I x P) for decks X (n x P): counts minus (r-1), PAD_CNT on padding slots."""
        cnt = np.asarray(self._incidence_csr() @ X, dtype=np.float32)
        out = np.full((self.I, X.shape[1]), PAD_CNT, dtype=np.float32)
        real = self.slot_edge >= 0
        out[real] = cnt[self.slot_edge[real]] + self.offset[real][:, None]
        return out

    def scores(self, X: np.ndarray) -> np.ndarray:
        cnt = self.counts(X)
        full = cnt == 1
        # each full combo contributes r slots; count once via its first slot
        first = self.edge_slots[self.edge_slots_ptr[:-1]]
        return full[first].sum(axis=0).astype(np.int64)


def segsum(layout: SlotLayout, flag: np.ndarray) -> np.ndarray:
    """(I x P) 0/1 -> (n x P) sums over each card's slot range (the device does tile-row sum + matmul)."""
    part = flag.reshape(layout.chunks, 32, -1).sum(axis=1)
    return layout.S @ part


def select_max_cols(M: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: one-hot of the row holding the max of M, ties by noise (higher wins) then larger row index.
    Returns (one-hot (n x P), max values (P,)). Same rule as population.select_max, transposed."""
    m = M.max(axis=0, keepdims=True)
    cand = (M == m)
    key = cand * (1.0 + noise)
    hot = key == key.max(axis=0, keepdims=True)
    n = M.shape[0]
    row = (n - 1) - np.argmax(hot[::-1, :], axis=0)
    out = np.zeros_like(M, dtype=np.float32)
    out[row, np.arange(M.shape[1])] = 1.0
    return out, m[0]


def generation(layout: SlotLayout, X: np.ndarray, cnt: np.ndarray, tabu: np.ndarray, score: np.ndarray, tenure: int,
               noise_out: np.ndarray, noise_in: np.ndarray):
    """One swap per replica (column). All arrays are (rows x P). Returns (X, cnt, tabu, score, b, a)."""
    H = layout.H
    valid = layout.valid[:, None]
    tabu = np.maximum(tabu - 1, 0)
    free = tabu == 0
    near = (cnt == 0).astype(np.float32)
    gain = segsum(layout, near)
    Wout = (gain + 1) * ((X == 0) & valid & free)
    Ob, wb = select_max_cols(Wout, noise_out)
    b = Ob.argmax(axis=0)
    cnt = cnt + H[b].T
    full = (cnt == 1).astype(np.float32)
    loss = segsum(layout, full)
    Win = (BIGV - loss) * ((X == 1) & free)
    Oa, wa = select_max_cols(Win, noise_in)
    a = Oa.argmax(axis=0)
    cnt = cnt - H[a].T
    X = X + Ob - Oa
    tabu = tabu + (Ob + Oa) * tenure
    score = score + (wb - 1) - (BIGV - wa)
    return X, cnt, tabu, score, b, a


class HyperPopulation:
    """Host driver for the NumPy reference (mirrors population.PopulationSearch, transposed layout)."""

    def __init__(self, inst: HyperInstance, k: int, P: int, seed: int = 0, tenure: int = 5, kick: int = 6, stale_epochs: int = 3,
                 big_kick: int = 20, seed_decks: list[np.ndarray] | None = None, seed_fraction: float = 0.0):
        """`seed_decks`: card-index arrays; `seed_fraction` of the initial replicas start as kicked copies of them.
        Stale replicas are reseeded half from random decks and half from the incumbent, kicked by a random
        `kick` … `big_kick` cards (the landscape has distant basins; small kicks fall straight back)."""
        self.inst = inst
        self.layout = SlotLayout(inst)
        self.layout.H = self.layout.incidence_rows()
        self.layout.S = self.layout.chunk_matrix()
        self.k, self.P, self.tenure, self.kick, self.stale_epochs = k, P, tenure, kick, stale_epochs
        self.big_kick = max(kick, big_kick)
        self.rng = np.random.default_rng(seed)
        self.n, self.m = self.layout.n, inst.m
        self.weight = inst.degree.astype(np.float64)
        self.X = self.random_decks(P)
        seeded = [d for d in (seed_decks or []) if d.size == k]
        n_seed = min(P, int(round(P * seed_fraction))) if seeded else 0
        if n_seed:
            cols = np.array_split(np.arange(n_seed), len(seeded))
            for d, c in zip(seeded, cols):
                base = np.zeros(self.n, dtype=np.float32)
                base[d] = 1.0
                self.X[:, c] = self.kick_decks(base, c.size, strength=self.rng.integers(0, self.kick + 1, c.size))
        self.cnt = self.layout.counts(self.X)
        self.tabu = np.zeros((self.n, P), dtype=np.float32)
        self.score = self.layout.scores(self.X).astype(np.float32)
        self.best_score, self.best_deck = -1, None
        self.replica_best = np.full(P, -1, dtype=np.int64)
        self.replica_stale = np.zeros(P, dtype=np.int64)
        self._best = np.full(P, -1, dtype=np.float32)
        self._best_X = self.X.copy()
        self.generations = 0
        self.last_epoch: dict = {}

    def random_decks(self, P: int) -> np.ndarray:
        X = np.zeros((self.n, P), dtype=np.float32)
        keys = self.rng.random((self.m, P)) ** (1.0 / (self.weight + 1e-3))[:, None]
        top = np.argpartition(-keys, self.k - 1, axis=0)[: self.k]
        np.put_along_axis(X, top, 1.0, axis=0)
        return X

    def kick_decks(self, base: np.ndarray, P: int, strength=None) -> np.ndarray:
        """P copies of `base` with `strength` (scalar or per-copy) cards swapped for random cards from the pool."""
        X = np.repeat(base[:, None], P, axis=1).astype(np.float32)
        inside, outside = np.flatnonzero(base[: self.m] > 0), np.flatnonzero((base[: self.m] == 0) & self.layout.valid[: self.m])
        strength = np.broadcast_to(np.asarray(self.kick if strength is None else strength), (P,))
        for p in range(P):
            s = int(min(strength[p], inside.size, outside.size))
            X[self.rng.choice(inside, s, replace=False), p] = 0
            X[self.rng.choice(outside, s, replace=False), p] = 1
        return X

    def reseed_decks(self, P: int) -> np.ndarray:
        """Replacement decks for P stale replicas: half random, half kicked copies of the incumbent."""
        half = P // 2 if self.best_deck is not None else 0
        parts = [self.random_decks(P - half)]
        if half:
            parts.append(self.kick_decks(self.best_deck, half, strength=self.rng.integers(self.kick, self.big_kick + 1, half)))
        return np.concatenate(parts, axis=1)

    def run_epoch(self, gens: int) -> int:
        for _ in range(gens):
            no = quantize_noise(self.rng.random((self.n, self.P), dtype=np.float32))
            ni = quantize_noise(self.rng.random((self.n, self.P), dtype=np.float32))
            self.X, self.cnt, self.tabu, self.score, _, _ = generation(self.layout, self.X, self.cnt, self.tabu, self.score, self.tenure, no, ni)
            improved = self.score > self._best
            self._best[improved] = self.score[improved]
            self._best_X[:, improved] = self.X[:, improved]  # score is post-move, so is the deck
        self.generations += gens
        self._bookkeep(self._best.astype(np.int64), lambda p: self._best_X[:, p].copy())
        return self.best_score

    def _bookkeep(self, best_now: np.ndarray, fetch_best_col) -> None:
        improved = best_now > self.replica_best
        self.replica_best = best_now.copy()
        self.replica_stale = np.where(improved, 0, self.replica_stale + 1)
        top = int(best_now.argmax())
        if best_now[top] > self.best_score:
            self.best_score = int(best_now[top])
            self.best_deck = fetch_best_col(top)
            assert self.best_deck.sum() == self.k, "best deck is not a k-subset"
        stale = np.flatnonzero(self.replica_stale >= self.stale_epochs)
        self.last_epoch = {"epoch_best": int(best_now.max()), "replicas_at_best": int((best_now == self.best_score).sum()),
                           "mean_best": float(best_now.mean()), "reseeded": int(stale.size)}
        if stale.size:
            self._reseed(stale, self.reseed_decks(stale.size))
            self.replica_best[stale] = -1
            self.replica_stale[stale] = 0

    def _reseed(self, stale: np.ndarray, cols: np.ndarray) -> None:
        self.X[:, stale] = cols
        self.cnt = self.layout.counts(self.X)
        self.score = self.layout.scores(self.X).astype(np.float32)
        self.tabu[:] = 0
        self._best[stale] = -1
        self._best_X[:, stale] = cols
