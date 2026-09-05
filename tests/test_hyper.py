"""Hypergraph search: CPU tabu and the NumPy population reference against brute force on small random instances."""
import itertools
import random

import numpy as np
import pytest

from spellbook_graph.hyper import HyperInstance, SwapState, describe, search
from spellbook_graph.hyperpop import HyperPopulation


def random_instance(seed: int, m: int = 20, n_edges: int = 60) -> HyperInstance:
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < n_edges:
        r = int(rng.integers(2, 5))
        edges.add(tuple(sorted(rng.choice(m, r, replace=False).tolist())))
    edges = sorted(edges)
    ptr = np.concatenate([[0], np.cumsum([len(e) for e in edges])]).astype(np.int64)
    ec = np.array([c for e in edges for c in e], dtype=np.int32)
    inc_e = np.repeat(np.arange(len(edges)), np.diff(ptr))
    order = np.argsort(ec, kind="stable")
    cp = np.concatenate([[0], np.cumsum(np.bincount(ec, minlength=m))]).astype(np.int64)
    bonus = np.zeros(m, dtype=np.int32)
    bonus[3] = 1
    return HyperInstance(names=[f"c{i}" for i in range(m)], oracle_ids=[""] * m, type_lines=[""] * m, orig_index=np.arange(m),
                         edge_ptr=ptr, edge_cards=ec, card_ptr=cp, card_edges=inc_e[order].astype(np.int32), bonus=bonus)


def brute_force(inst: HyperInstance, k: int) -> int:
    return max(inst.score(np.array(c)) for c in itertools.combinations(range(inst.m), k))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_swap_deltas_are_exact(seed):
    inst = random_instance(seed)
    rng = random.Random(seed)
    members = np.array(sorted(rng.sample(range(inst.m), 6)))
    st = SwapState(inst, members)
    inside, outside, delta = st.neighbourhood()
    base = st.score()
    for i in range(inside.size):
        for j in range(outside.size):
            new = np.array(sorted(set(members.tolist()) - {int(inside[i])} | {int(outside[j])}))
            assert inst.score(new) - base == delta[i, j]


@pytest.mark.parametrize("seed", [0, 1])
def test_tabu_matches_brute_force(seed):
    inst = random_instance(seed)
    members, score, _ = search(inst, 6, seconds=2.0, seed=seed, tabu_iters=300)
    assert score == inst.score(members) == brute_force(inst, 6)
    assert describe(inst, members)["score"] == score


@pytest.mark.parametrize("seed", [0, 1])
def test_population_matches_brute_force(seed):
    inst = random_instance(seed)
    k = 6
    ps = HyperPopulation(inst, k=k, P=64, seed=seed)
    for _ in range(6):
        ps.run_epoch(15)
        for p in range(0, 64, 7):  # incremental scores and counts stay consistent with a full rescoring
            deck = np.flatnonzero(ps.X[:, p])
            assert len(deck) == k
            assert inst.score(deck) == int(ps.score[p])
        assert np.allclose(ps.cnt, ps.layout.counts(ps.X))
    assert ps.best_score == inst.score(np.flatnonzero(ps.best_deck)) == brute_force(inst, k)


def test_seeded_population_and_reseeding():
    inst = random_instance(5)
    k = 6
    seed_deck = np.arange(k)
    ps = HyperPopulation(inst, k=k, P=32, seed=1, seed_decks=[seed_deck], seed_fraction=0.5, kick=2, big_kick=4)
    assert np.all(ps.X.sum(axis=0) == k)
    assert (ps.X[:k, :16].sum(axis=0) >= k - 2).all()  # seeded half is at most `kick` cards away from the seed
    ps.run_epoch(5)
    cols = ps.reseed_decks(10)
    assert cols.shape == (ps.n, 10) and np.all(cols.sum(axis=0) == k)
