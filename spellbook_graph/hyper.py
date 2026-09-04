"""Search the combo hypergraph: k cards containing the most unique infinite combos of any size.

A combo (hyperedge) counts when *all* of its cards are among the k chosen ones,
or in the command zone: combos containing the commander lose that card
(they are free), and a combo reduced to a single card becomes that card's
``bonus``. Combos that need a different card to be the commander are dropped.

Swap-move algebra with cnt[e] = number of chosen cards in combo e, r[e] = |e|:
  gain[b]  = #{e ∋ b : cnt[e] = r[e]-1}          combos completed by adding b
  loss[a]  = #{e ∋ a : cnt[e] = r[e]}            combos broken by dropping a
  corr[a,b]= #{e ∋ a,b : cnt[e] = r[e]-1}        completed by b but broken by a
  delta(drop a, add b) = gain[b] - loss[a] - corr[a,b]
All three are sparse sums over the incidence list, so a full best-swap scan of
the k x (n-k) neighbourhood costs O(|incidences| + k n) per iteration.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .prereqs import classify


@dataclass
class HyperInstance:
    names: list[str]
    oracle_ids: list[str]
    type_lines: list[str]
    orig_index: np.ndarray
    edge_ptr: np.ndarray      # (E+1,) CSR over hyperedges -> edge_cards
    edge_cards: np.ndarray    # (I,) candidate indices, sorted within each edge
    card_ptr: np.ndarray      # (m+1,) CSR over cards -> card_edges
    card_edges: np.ndarray    # (I,) hyperedge ids incident to each card
    bonus: np.ndarray         # (m,) combos with the commander alone
    commander: str | None = None
    commander_partners: list[str] = field(default_factory=list)
    edge_names: list[list[str]] = field(default_factory=list)  # for reporting (original-order names)

    @property
    def m(self) -> int:
        return len(self.names)

    @property
    def E(self) -> int:
        return len(self.edge_ptr) - 1

    @property
    def r(self) -> np.ndarray:
        return np.diff(self.edge_ptr)

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.card_ptr) + self.bonus

    def incidence(self) -> tuple[np.ndarray, np.ndarray]:
        """(inc_e, inc_v): flat incidence list, edge id and card id per slot."""
        inc_e = np.repeat(np.arange(self.E), self.r)
        return inc_e, self.edge_cards

    def counts(self, members) -> np.ndarray:
        in_s = np.zeros(self.m, dtype=np.int32)
        in_s[np.asarray(members, dtype=np.int64)] = 1
        return np.add.reduceat(in_s[self.edge_cards], self.edge_ptr[:-1]) if self.E else np.zeros(0, np.int32)

    def full_edges(self, members) -> np.ndarray:
        return np.flatnonzero(self.counts(members) == self.r)

    def score(self, members) -> int:
        members = np.asarray(members)
        return int(self.full_edges(members).size + self.bonus[members].sum())


def load_hyper_instance(graph_dir: Path, commander: str | None = "Kenrith, the Returned King", exclude: list[str] | None = None,
                        results: list[str] | None = None, prereqs: str = "any", min_degree: int = 1, max_size: int = 0) -> HyperInstance:
    """`prereqs`: 'any' | 'none' (clean variants only) | 'kenrith'. `min_degree`: drop cards in fewer combos
    (a heuristic pool reduction; 1 = exact). `max_size`: drop combos with more cards than this (0 = keep all)."""
    data = json.loads((Path(graph_dir) / "hypergraph.json").read_text())
    nodes = data["nodes"]
    n = len(nodes)
    names = [node["name"] for node in nodes]
    result_res = [re.compile(r, re.I) for r in (results or [])]

    cmdr_idx = None
    cmdr_oracle = None
    if commander and commander.lower() not in {"none", ""}:
        matches = [i for i, name in enumerate(names) if name.lower() == commander.lower()]
        if not matches:
            matches = [i for i, name in enumerate(names) if commander.lower() in name.lower()]
        if len(matches) != 1:
            raise SystemExit(f"commander {commander!r} matched {len(matches)} cards in the hypergraph")
        cmdr_idx = matches[0]
        cmdr_oracle = nodes[cmdr_idx]["oracle_id"]

    usable: list[tuple[int, ...]] = []
    for edge in data["hyperedges"]:
        cards = tuple(edge["cards"])
        if max_size and len(cards) > max_size:
            continue
        if prereqs == "none":
            if not edge.get("clean", False):
                continue
            features = edge.get("clean_features", [])
        elif prereqs == "kenrith":
            ok = [v for v in edge["variants"] if v.get("battlefield_or_hand") and classify(v.get("notable_prerequisites", ""), [nodes[c].get("type_line", "") for c in cards]).ok]
            if not ok:
                continue
            features = sorted({f for v in ok for f in v["features"]})
        else:
            features = edge["features"]
        if result_res and not any(r.search(f) for r in result_res for f in features):
            continue
        if not edge["unconditional"]:
            if cmdr_idx is None or cmdr_oracle not in (edge.get("commander_required") or []):
                continue
        usable.append(cards)

    bonus_full = np.zeros(n, dtype=np.int32)
    partners: set[str] = set()
    rest: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for cards in usable:
        if cmdr_idx is not None and cmdr_idx in cards:
            cards = tuple(c for c in cards if c != cmdr_idx)
            partners.update(names[c] for c in cards)
            if len(cards) == 1:
                bonus_full[cards[0]] += 1
                continue
        if cards in seen:  # a combo with the commander that duplicates one without him
            continue
        seen.add(cards)
        rest.append(cards)

    degree = np.zeros(n, dtype=np.int64)
    for cards in rest:
        for c in cards:
            degree[c] += 1
    excluded = {name.lower() for name in (exclude or [])}
    keep = [i for i in range(n) if i != cmdr_idx and names[i].lower() not in excluded and ((degree[i] + bonus_full[i]) >= max(1, min_degree))]
    keep_set = set(keep)
    remap = {orig: new for new, orig in enumerate(keep)}
    m = len(keep)

    edges = sorted(tuple(sorted(remap[c] for c in cards)) for cards in rest if all(c in keep_set for c in cards))
    E = len(edges)
    edge_ptr = np.zeros(E + 1, dtype=np.int64)
    edge_ptr[1:] = np.cumsum([len(c) for c in edges])
    edge_cards = np.array([c for cards in edges for c in cards], dtype=np.int32)
    inc_e = np.repeat(np.arange(E), np.diff(edge_ptr))
    order = np.argsort(edge_cards, kind="stable")
    card_edges = inc_e[order].astype(np.int32)
    card_ptr = np.zeros(m + 1, dtype=np.int64)
    card_ptr[1:] = np.cumsum(np.bincount(edge_cards, minlength=m))

    return HyperInstance(
        names=[names[i] for i in keep],
        oracle_ids=[nodes[i]["oracle_id"] for i in keep],
        type_lines=[nodes[i].get("type_line", "") for i in keep],
        orig_index=np.array(keep, dtype=np.int64),
        edge_ptr=edge_ptr, edge_cards=edge_cards, card_ptr=card_ptr, card_edges=card_edges,
        bonus=bonus_full[keep],
        commander=names[cmdr_idx] if cmdr_idx is not None else None,
        commander_partners=sorted(partners),
    )


# --- incremental state -------------------------------------------------------


class SwapState:
    """Deck membership + per-combo counts, with O(deg) updates and vectorised neighbourhood evaluation."""

    def __init__(self, inst: HyperInstance, members):
        self.inst = inst
        self.m, self.E = inst.m, inst.E
        self.r = inst.r.astype(np.int32)
        self.inc_e, self.inc_v = inst.incidence()
        self.in_s = np.zeros(self.m, dtype=bool)
        self.in_s[np.asarray(members, dtype=np.int64)] = True
        self.cnt = inst.counts(np.flatnonzero(self.in_s)).astype(np.int32)
        self.k = int(self.in_s.sum())

    def score(self) -> int:
        return int((self.cnt == self.r).sum() + self.inst.bonus[self.in_s].sum())

    def move(self, a: int, b: int) -> None:
        inst = self.inst
        self.cnt[inst.card_edges[inst.card_ptr[a]:inst.card_ptr[a + 1]]] -= 1
        self.cnt[inst.card_edges[inst.card_ptr[b]:inst.card_ptr[b + 1]]] += 1
        self.in_s[a] = False
        self.in_s[b] = True

    def neighbourhood(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(inside, outside, delta) with delta[i, j] = score change of swapping inside[i] for outside[j]."""
        inst = self.inst
        near = self.cnt == self.r - 1
        full = self.cnt == self.r
        near_i = near[self.inc_e]
        gain = np.bincount(self.inc_v, weights=near_i, minlength=self.m) + inst.bonus
        loss = np.bincount(self.inc_v, weights=full[self.inc_e], minlength=self.m) + inst.bonus
        inside = np.flatnonzero(self.in_s)
        outside = np.flatnonzero(~self.in_s)
        pos = np.full(self.m, -1, dtype=np.int64)
        pos[inside] = np.arange(inside.size)
        # missing card of every near-complete combo, then corr[pos[a], b] for the a's inside it
        sel = np.flatnonzero(near_i)
        e_sel, v_sel = self.inc_e[sel], self.inc_v[sel]
        missing = np.zeros(self.E, dtype=np.int64)
        np.add.at(missing, e_sel, np.where(self.in_s[v_sel], 0, v_sel + 1))  # single-owner sum
        missing -= 1
        a_in = self.in_s[v_sel]
        flat = pos[v_sel[a_in]] * self.m + missing[e_sel[a_in]]
        corr = np.bincount(flat, minlength=inside.size * self.m).reshape(inside.size, self.m)
        delta = gain[outside][None, :] - loss[inside][:, None] - corr[:, outside]
        return inside, outside, delta.astype(np.int64)


# --- tabu search --------------------------------------------------------------


def greedy_start(inst: HyperInstance, k: int, rng: random.Random, rcl: int = 8, seed_cards: list[int] | None = None) -> np.ndarray:
    """GRASP: add cards one at a time, each time choosing among the top-`rcl` by combos completed
    (ties broken by combos brought within one card of completion, then by total degree)."""
    st = SwapState(inst, seed_cards or [])
    weights_deg = inst.degree.astype(np.float64)
    while st.k < k:
        near = st.cnt == st.r - 1
        near2 = st.cnt == st.r - 2
        gain = np.bincount(st.inc_v, weights=near[st.inc_e], minlength=inst.m) + inst.bonus
        gain2 = np.bincount(st.inc_v, weights=near2[st.inc_e], minlength=inst.m)
        key = gain * 1e6 + gain2 * 1e2 + weights_deg / (weights_deg.max() + 1)
        key[st.in_s] = -1
        width = min(rcl, inst.m - st.k)
        pool = np.argpartition(-key, width - 1)[:width]
        pick = int(pool[rng.randrange(len(pool))])
        st.cnt[inst.card_edges[inst.card_ptr[pick]:inst.card_ptr[pick + 1]]] += 1
        st.in_s[pick] = True
        st.k += 1
    return np.flatnonzero(st.in_s)


def tabu(inst: HyperInstance, members: np.ndarray, iters: int, rng: random.Random, tenure: int, deadline: float | None = None, stall: int = 150) -> tuple[np.ndarray, int]:
    """Best-swap tabu search from `members`; stops after `iters` moves or `stall` moves without a new best."""
    st = SwapState(inst, members)
    score = st.score()
    best_score, best_members = score, np.flatnonzero(st.in_s)
    tabu_until = np.zeros(inst.m, dtype=np.int64)
    last_improvement = 0
    for it in range(1, iters + 1):
        if it - last_improvement > stall:
            break
        inside, outside, delta = st.neighbourhood()
        forbidden = (tabu_until[inside][:, None] > it) | (tabu_until[outside][None, :] > it)
        allowed = np.where(forbidden & ((score + delta) <= best_score), -(1 << 40), delta)
        flat = int(np.argmax(allowed))
        i, j = divmod(flat, outside.size)
        if allowed[i, j] <= -(1 << 39):
            i, j = rng.randrange(inside.size), rng.randrange(outside.size)
        a, b = int(inside[i]), int(outside[j])
        score += int(delta[i, j])
        st.move(a, b)
        tabu_until[a] = it + tenure + rng.randrange(tenure + 1)
        tabu_until[b] = it + tenure + rng.randrange(tenure + 1)
        if score > best_score:
            best_score, best_members, last_improvement = score, np.flatnonzero(st.in_s), it
        if deadline is not None and it % 20 == 0 and time.time() > deadline:
            break
    return best_members, best_score


def perturb(inst: HyperInstance, members: np.ndarray, strength: int, rng: random.Random) -> np.ndarray:
    """Drop the `strength` least useful cards (by combos they complete in the deck), refill greedily with randomness."""
    st = SwapState(inst, members)
    full = st.cnt == st.r
    loss = np.bincount(st.inc_v, weights=full[st.inc_e], minlength=inst.m) + inst.bonus
    order = np.argsort(loss[members] + rng.random() * 0, kind="stable")
    drop_pool = [int(members[i]) for i in order[: strength * 2]]
    rng.shuffle(drop_pool)
    drop = set(drop_pool[:strength])
    kept = [int(v) for v in members if v not in drop]
    return greedy_start(inst, len(members), rng, rcl=6, seed_cards=kept)


def search(inst: HyperInstance, k: int, seconds: float, seed: int = 0, tabu_iters: int = 5000, rcl: int = 8, kick: int = 6, report=None) -> tuple[np.ndarray, int, dict]:
    rng = random.Random(seed)
    deadline = time.time() + seconds
    tenure = max(3, k // 8)
    best_members, best_score = None, -1
    restarts = kicks = 0
    while time.time() < deadline:
        restarts += 1
        members = greedy_start(inst, k, rng, rcl)
        members, score = tabu(inst, members, tabu_iters, rng, tenure, deadline)
        if score > best_score:
            best_score, best_members = score, members.copy()
            if report:
                report(best_score, restarts, kicks)
        stale = 0
        while stale < 30 and time.time() < deadline:
            kicks += 1
            cand, cand_score = tabu(inst, perturb(inst, members, kick, rng), tabu_iters, rng, tenure, deadline)
            if cand_score > score:
                members, score, stale = cand, cand_score, 0
            else:
                stale += 1
            if score > best_score:
                best_score, best_members = score, members.copy()
                if report:
                    report(best_score, restarts, kicks)
    assert best_members is not None
    return best_members, best_score, {"restarts": restarts, "kicks": kicks, "seed": seed, "score": int(best_score)}


_WORKER_INST: HyperInstance | None = None


def _init_worker(inst):
    global _WORKER_INST
    _WORKER_INST = inst


def _worker(args):
    k, seconds, seed, tabu_iters, rcl, kick = args
    members, score, stats = search(_WORKER_INST, k, seconds, seed=seed, tabu_iters=tabu_iters, rcl=rcl, kick=kick)
    return [int(v) for v in members], score, stats


def parallel_search(inst: HyperInstance, k: int, seconds: float, jobs: int, tabu_iters: int = 5000, rcl: int = 8, kick: int = 6, seed0: int = 0):
    if jobs <= 1:
        members, score, stats = search(inst, k, seconds, seed=seed0, tabu_iters=tabu_iters, rcl=rcl, kick=kick,
                                       report=lambda s, r, kk: print(f"  {s} combos (restart {r}, kick {kk})", flush=True))
        return members, score, [stats]
    payload = [(k, seconds, seed0 + s, tabu_iters, rcl, kick) for s in range(jobs)]
    with mp.Pool(jobs, initializer=_init_worker, initargs=(inst,)) as pool:
        results = pool.map(_worker, payload)
    best = max(results, key=lambda r: r[1])
    return np.array(best[0], dtype=np.int64), best[1], [r[2] for r in results]


# --- reporting -----------------------------------------------------------------


def describe(inst: HyperInstance, members) -> dict:
    members = np.array(sorted(int(v) for v in members), dtype=np.int64)
    full = inst.full_edges(members)
    r = inst.r
    per_card = np.zeros(inst.m, dtype=np.int64)
    for e in full:
        per_card[inst.edge_cards[inst.edge_ptr[e]:inst.edge_ptr[e + 1]]] += 1
    per_card += inst.bonus
    by_size = {}
    for e in full:
        by_size[int(r[e])] = by_size.get(int(r[e]), 0) + 1
    cards = [{
        "name": inst.names[v], "oracle_id": inst.oracle_ids[v], "type_line": inst.type_lines[v],
        "combos_in_deck": int(per_card[v]), "combos_with_commander": int(inst.bonus[v]),
        "total_combos_in_pool": int(inst.card_ptr[v + 1] - inst.card_ptr[v] + inst.bonus[v]),
    } for v in members]
    cards.sort(key=lambda c: (-c["combos_in_deck"], c["name"]))
    combos = sorted([inst.names[c] for c in inst.edge_cards[inst.edge_ptr[e]:inst.edge_ptr[e + 1]]] for e in full)
    return {
        "commander": inst.commander, "k": int(len(members)),
        "score": int(len(full) + inst.bonus[members].sum()),
        "combos_among_deck_cards": int(len(full)), "combos_with_commander": int(inst.bonus[members].sum()),
        "combos_by_size": dict(sorted(by_size.items())),
        "cards": cards, "combos": combos,
    }


def write_result(out: Path, inst: HyperInstance, members, extra: dict) -> dict:
    result = describe(inst, members)
    result["search"] = extra
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "deck.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# {result['score']} unique infinite combos (any size) among {result['k']} cards; by size: "
             + ", ".join(f"{s}-card {c}" for s, c in result["combos_by_size"].items())]
    if inst.commander:
        lines.append(f"# commander: {inst.commander}")
    lines += [f"1 {c['name']}" for c in sorted(result["cards"], key=lambda c: c["name"])]
    (out / "deck.txt").write_text("\n".join(lines) + "\n")
    return result
