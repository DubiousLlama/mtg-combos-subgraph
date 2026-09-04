"""Search correctness on instances small enough to brute-force."""

from itertools import combinations

import numpy as np
import pytest

from spellbook_graph.population import PopulationSearch, quantize_noise, select_max
from spellbook_graph.search import Instance, search


def random_instance(rng: np.random.Generator, n: int, p: float, bonus_p: float = 0.2) -> Instance:
    A = (rng.random((n, n)) < p).astype(np.uint8)
    A = np.triu(A, 1)
    A = A + A.T
    edges = [(int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(A, 1)))]
    bonus = (rng.random(n) < bonus_p).astype(np.int32)
    names = [f"card{i}" for i in range(n)]
    return Instance(names=names, oracle_ids=names, type_lines=[""] * n, orig_index=np.arange(n), A=A, bonus=bonus, edges=edges)


def brute_force(inst: Instance, k: int) -> int:
    return max(inst.score(np.array(c)) for c in combinations(range(inst.m), k))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tabu_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    inst = random_instance(rng, n=16, p=0.25)
    k = 6
    members, score, _ = search(inst, k, seconds=1.0, seed=seed, tabu_iters=100)
    assert len(set(members.tolist())) == k
    assert inst.score(members) == score
    assert score == brute_force(inst, k)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_population_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    inst = random_instance(rng, n=18, p=0.25)
    k = 6
    ps = PopulationSearch(inst.A.astype(np.float32), inst.bonus.astype(np.float32), k=k, P=64, seed=seed)
    for _ in range(4):
        ps.run_epoch(20)
    deck = np.flatnonzero(ps.best_deck)
    assert len(deck) == k
    assert inst.score(deck) == ps.best_score
    assert ps.best_score == brute_force(inst, k)
    assert np.all(ps.X.sum(axis=1) == k), "every replica must stay a k-subset"


def test_select_max_rule():
    M = np.array([[3, 5, 5, 1, 5], [0, 0, 0, 0, 0]], dtype=np.float32)
    noise = np.array([[0, 0.5, 0.25, 0.9, 0.5], [0.1, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    hot = select_max(M, noise)
    # row 0: max 5 at cols 1,2,4; noise ties cols 1 and 4 at 0.5; larger index wins -> 4
    assert hot[0].tolist() == [0, 0, 0, 0, 1]
    # row 1: all tie on value and noise -> last index
    assert hot[1].tolist() == [0, 0, 0, 0, 1]
    assert np.all(hot.sum(axis=1) == 1)


def test_quantize_noise_is_bf16_exact():
    q = quantize_noise(np.random.default_rng(0).random(10000, dtype=np.float32))
    assert np.all(q * 128 == np.floor(q * 128))
    assert q.min() >= 0 and q.max() < 1


def _ttnn_available() -> bool:
    try:
        from spellbook_graph.tt_search import _import_ttnn

        _import_ttnn()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ttnn_available(), reason="ttnn not importable")
def test_tt_generation_matches_numpy():
    """Bit-exact agreement of the device step with the NumPy reference (needs one Tenstorrent device)."""
    from spellbook_graph.population import generation
    from spellbook_graph.tt_search import TTPopulation

    rng = np.random.default_rng(5)
    inst = random_instance(rng, n=100, p=0.05)
    tp = TTPopulation(inst.A.astype(np.float32), inst.bonus.astype(np.float32), k=10, P=64, seed=1, devices=1)
    try:
        X, tabu = tp.X.copy(), tp.tabu.copy()
        dX, dtabu = tp._up(X, tp.bf16), tp._up(tabu, tp.bf16)
        for _ in range(5):
            no = quantize_noise(rng.random((64, tp.n), dtype=np.float32))
            ni = quantize_noise(rng.random((64, tp.n), dtype=np.float32))
            X, score, _, _, tabu = generation(X, tp.A, tp.bonus, tp.valid, tabu, tp.tenure, no, ni)
            dX, dscore, dtabu = tp._device_generation(dX, dtabu, tp._up(no, tp.bf16), tp._up(ni, tp.bf16))
            assert np.array_equal(tp._down(dX), X)
            assert np.array_equal(tp._down(dscore)[:, 0], score)
            assert np.array_equal(tp._down(dtabu), tabu)
        tp.run_epoch(20)
        assert tp.best_deck.sum() == 10
        assert inst.score(np.flatnonzero(tp.best_deck[: inst.m])) == tp.best_score
    finally:
        tp.close()


def test_prereq_classifier():
    from spellbook_graph.prereqs import classify

    ok = [
        ("Sanctum Weaver does not have summoning sickness.", []),
        ("You have a way to gain life.", []),
        ("Walking Ballista has at least two +1/+1 counters on it", []),
        ("You control at least five creatures.", []),
        ("You control at least two enchantments.", ["Enchantment — Aura", "Enchantment Creature — Dryad"]),
        ("Pemmin's Aura attached to Sanctum Weaver.\nSanctum Weaver does not have summoning sickness.", []),
    ]
    bad = [
        ("You control at least two enchantments.", ["Creature — Elf", "Artifact"]),
        ("You have a way to deal damage to Boros Reckoner.", []),
        ("Creatures you control can tap to produce at least {2}.", []),
        ("An opponent cannot block creatures you control.", []),
        ("Myojin of Cryptic Dreams has at least two indestructible counters on it.", []),
        ("Something nobody wrote a rule for.", []),
        ("Sanctum Weaver does not have summoning sickness and cannot be blocked by an opponent.", []),
    ]
    for text, types in ok:
        assert classify(text, types).ok, text
    for text, types in bad:
        assert not classify(text, types).ok, text
