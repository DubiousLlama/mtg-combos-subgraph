"""Search the combo graph for the k cards inducing the most unique two-card combos.

Every combo here uses exactly two cards, so this is densest-k-subgraph: pick k
vertices maximising induced edges. NP-hard, no PTAS, but n is small (~1.6k
vertices, ~2.8k edges) so multi-start tabu search over swap moves finds very
good decks in seconds and can be run to exhaustion in minutes.

A commander sits in the command zone rather than in the k slots, so its combos
are free: each candidate carries a `bonus` equal to the number of combos it
makes with the commander. Combos that require *another* card to be the
commander are dropped.
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
class Instance:
    """Compressed graph: only vertices that can ever contribute an edge."""

    names: list[str]
    oracle_ids: list[str]
    type_lines: list[str]
    orig_index: np.ndarray  # candidate -> index in graph.json
    A: np.ndarray  # dense adjacency, uint8, (m, m)
    bonus: np.ndarray  # free edges with the commander, int32, (m,)
    edges: list[tuple[int, int]]  # candidate-space edge list
    commander: str | None = None
    commander_partners: list[str] = field(default_factory=list)

    @property
    def m(self) -> int:
        return len(self.names)

    def score(self, members: np.ndarray) -> int:
        sub = self.A[np.ix_(members, members)]
        return int(sub.sum() // 2 + self.bonus[members].sum())


def load_instance(graph_dir: Path, commander: str | None = None, exclude: list[str] | None = None, results: list[str] | None = None, strict: bool = False, prereqs: str = "any", report: dict | None = None) -> Instance:
    """`exclude`: card names to leave out of the candidate pool (robustness checks: re-run without a hub).
    `results`: regexes over produced feature names; when given, a combo counts only if one of its features matches.
    `strict`: only combos with a variant that has no notable prerequisites and starts on the battlefield / in hand.
    `prereqs`: 'any' | 'none' (same as strict) | 'kenrith' (see prereqs.py). `report`, if given, collects denied-line tallies."""
    if strict:
        prereqs = "none"
    result_res = [re.compile(r, re.I) for r in (results or [])]
    data = json.loads((graph_dir / "graph.json").read_text())
    nodes = data["nodes"]
    n = len(nodes)
    names = [node["name"] for node in nodes]

    cmdr_idx = None
    if commander and commander.lower() not in {"none", ""}:
        matches = [i for i, name in enumerate(names) if name.lower() == commander.lower()]
        if not matches:
            matches = [i for i, name in enumerate(names) if commander.lower() in name.lower()]
        if len(matches) != 1:
            raise SystemExit(f"commander {commander!r} matched {len(matches)} cards in the graph")
        cmdr_idx = matches[0]
        cmdr_oracle = nodes[cmdr_idx]["oracle_id"]

    # Which edges are usable given who the commander is.
    usable: list[tuple[int, int]] = []
    for edge in data["edges"]:
        if prereqs == "none":
            if not edge.get("clean", False):
                continue
            features = edge.get("clean_features", [])
        elif prereqs == "kenrith":
            ok_variants = []
            for var in edge.get("variants", []):
                if not var.get("battlefield_or_hand", False):
                    continue
                verdict = classify(var.get("notable_prerequisites", ""), [nodes[edge["u"]].get("type_line", ""), nodes[edge["v"]].get("type_line", "")])
                if report is not None:
                    for line, why in verdict.denied:
                        report.setdefault("denied", {}).setdefault(why, {})
                        report["denied"][why][line] = report["denied"][why].get(line, 0) + 1
                    for line, why in verdict.allowed:
                        report.setdefault("allowed", {})
                        report["allowed"][why] = report["allowed"].get(why, 0) + 1
                if verdict.ok:
                    ok_variants.append(var)
            if not ok_variants:
                continue
            features = sorted({f for var in ok_variants for f in var["features"]})
        else:
            features = edge.get("features", [])
        if result_res and not any(r.search(f) for r in result_res for f in features):
            continue
        if not edge["unconditional"]:
            required = edge.get("commander_required") or []
            if cmdr_idx is None or cmdr_oracle not in required:
                continue
        usable.append((edge["u"], edge["v"]))

    bonus_full = np.zeros(n, dtype=np.int32)
    partners: list[str] = []
    rest: list[tuple[int, int]] = []
    for u, v in usable:
        if u == cmdr_idx or v == cmdr_idx:
            other = v if u == cmdr_idx else u
            bonus_full[other] += 1
            partners.append(names[other])
        else:
            rest.append((u, v))

    degree = np.zeros(n, dtype=np.int32)
    for u, v in rest:
        degree[u] += 1
        degree[v] += 1

    excluded = {name.lower() for name in (exclude or [])}
    keep = [i for i in range(n) if i != cmdr_idx and names[i].lower() not in excluded and (degree[i] > 0 or bonus_full[i] > 0)]
    remap = {orig: new for new, orig in enumerate(keep)}
    m = len(keep)

    A = np.zeros((m, m), dtype=np.uint8)
    edges = []
    for u, v in rest:
        a, b = remap[u], remap[v]
        A[a, b] = A[b, a] = 1
        edges.append((a, b))

    return Instance(
        names=[names[i] for i in keep],
        oracle_ids=[nodes[i]["oracle_id"] for i in keep],
        type_lines=[nodes[i].get("type_line", "") for i in keep],
        orig_index=np.array(keep, dtype=np.int64),
        A=A,
        bonus=bonus_full[keep],
        edges=edges,
        commander=names[cmdr_idx] if cmdr_idx is not None else None,
        commander_partners=sorted(partners),
    )


# --- search ---------------------------------------------------------------


def _greedy_start(inst: Instance, k: int, rng: random.Random, rcl: int) -> np.ndarray:
    """GRASP construction: repeatedly add a card from the top-`rcl` by marginal gain."""
    m = inst.m
    in_s = np.zeros(m, dtype=bool)
    cont = inst.bonus.astype(np.int32).copy()
    members = []
    for _ in range(k):
        cont_masked = np.where(in_s, -1, cont)
        width = min(rcl, m - len(members))
        pool = np.argpartition(-cont_masked, width - 1)[:width] if width > 1 else np.array([int(np.argmax(cont_masked))])
        pick = int(pool[rng.randrange(len(pool))])
        members.append(pick)
        in_s[pick] = True
        cont += inst.A[pick]
    return np.array(sorted(members), dtype=np.int64)


def _tabu(inst: Instance, members: np.ndarray, k: int, iters: int, rng: random.Random, tenure: int) -> tuple[np.ndarray, int]:
    """Best-swap tabu search with aspiration. Returns (best members, best score)."""
    m = inst.m
    A = inst.A
    bonus = inst.bonus.astype(np.int32)
    in_s = np.zeros(m, dtype=bool)
    in_s[members] = True
    cont = A[members].sum(axis=0, dtype=np.int32) + bonus
    score = int(A[np.ix_(members, members)].sum() // 2 + bonus[members].sum())
    best_score = score
    best_members = members.copy()
    tabu_until = np.zeros(m, dtype=np.int64)
    all_idx = np.arange(m)

    for it in range(1, iters + 1):
        inside = all_idx[in_s]
        outside = all_idx[~in_s]
        # delta of swapping out `a` (row) for `b` (column)
        delta = cont[outside][None, :] - cont[inside][:, None] - A[np.ix_(inside, outside)].astype(np.int32)
        forbidden = (tabu_until[inside][:, None] > it) | (tabu_until[outside][None, :] > it)
        allowed_best = np.where(forbidden & ((score + delta) <= best_score), -(1 << 20), delta)
        flat = int(np.argmax(allowed_best))
        i, j = divmod(flat, outside.size)
        gain = int(delta[i, j])
        if allowed_best[i, j] <= -(1 << 19):  # everything tabu; nudge randomly
            i, j = rng.randrange(inside.size), rng.randrange(outside.size)
            gain = int(delta[i, j])
        a, b = int(inside[i]), int(outside[j])

        in_s[a] = False
        in_s[b] = True
        cont -= A[a]
        cont += A[b]
        score += gain
        # Discourage undoing the move for a while.
        tabu_until[a] = it + tenure + rng.randrange(tenure + 1)
        tabu_until[b] = it + tenure + rng.randrange(tenure + 1)
        if score > best_score:
            best_score = score
            best_members = all_idx[in_s].copy()

    return best_members, best_score


def _perturb(inst: Instance, members: np.ndarray, strength: int, rng: random.Random) -> np.ndarray:
    """Kick: drop the `strength` least useful cards, refill with random neighbours of the deck."""
    m = inst.m
    in_s = np.zeros(m, dtype=bool)
    in_s[members] = True
    cont = inst.A[members].sum(axis=0, dtype=np.int32) + inst.bonus
    order = np.argsort(cont[members], kind="stable")
    drop = set(int(members[i]) for i in order[:strength])
    kept = [int(v) for v in members if v not in drop]
    in_s[:] = False
    in_s[kept] = True
    cont = inst.A[kept].sum(axis=0, dtype=np.int32) + inst.bonus
    for _ in range(strength):
        cand = np.flatnonzero((~in_s) & (cont > 0))
        pick = int(cand[rng.randrange(len(cand))]) if cand.size else rng.randrange(m)
        while in_s[pick]:
            pick = rng.randrange(m)
        kept.append(pick)
        in_s[pick] = True
        cont += inst.A[pick]
    return np.array(sorted(kept), dtype=np.int64)


def search(
    inst: Instance,
    k: int,
    seconds: float,
    seed: int = 0,
    tabu_iters: int = 600,
    rcl: int = 8,
    kick: int = 6,
    report: bool = False,
) -> tuple[np.ndarray, int, dict]:
    rng = random.Random(seed)
    deadline = time.time() + seconds
    tenure = max(3, k // 10)
    best_members: np.ndarray | None = None
    best_score = -1
    restarts = 0
    kicks = 0

    while time.time() < deadline:
        restarts += 1
        members = _greedy_start(inst, k, rng, rcl)
        members, score = _tabu(inst, members, k, tabu_iters, rng, tenure)
        if score > best_score:
            best_score, best_members = score, members.copy()
        # Intensify around this basin until it stops paying off.
        stale = 0
        while stale < 25 and time.time() < deadline:
            kicks += 1
            cand, cand_score = _tabu(inst, _perturb(inst, members, kick, rng), k, tabu_iters, rng, tenure)
            if cand_score > score:
                members, score, stale = cand, cand_score, 0
            else:
                stale += 1
            if score > best_score:
                best_score, best_members = score, members.copy()
                if report:
                    print(f"  seed {seed}: {best_score} edges (restart {restarts}, kick {kicks})", flush=True)

    assert best_members is not None
    return best_members, best_score, {"restarts": restarts, "kicks": kicks, "seed": seed}


def _worker(args) -> tuple[list[int], int, dict]:
    graph_dir, commander, k, seconds, seed, tabu_iters, rcl, kick = args
    inst = load_instance(Path(graph_dir), commander)
    members, score, stats = search(inst, k, seconds, seed=seed, tabu_iters=tabu_iters, rcl=rcl, kick=kick)
    return [int(v) for v in members], score, stats


def parallel_search(
    graph_dir: Path,
    inst: Instance,
    k: int,
    seconds: float,
    jobs: int,
    commander: str | None,
    tabu_iters: int = 600,
    rcl: int = 8,
    kick: int = 6,
) -> tuple[np.ndarray, int, list[dict]]:
    if jobs <= 1:
        members, score, stats = search(inst, k, seconds, seed=0, tabu_iters=tabu_iters, rcl=rcl, kick=kick, report=True)
        return members, score, [stats]
    payload = [(str(graph_dir), commander, k, seconds, seed, tabu_iters, rcl, kick) for seed in range(jobs)]
    with mp.Pool(jobs) as pool:
        results = pool.map(_worker, payload)
    best = max(results, key=lambda r: r[1])
    return np.array(best[0], dtype=np.int64), best[1], [r[2] | {"score": r[1]} for r in results]


# --- reporting ------------------------------------------------------------


def describe(inst: Instance, members: np.ndarray) -> dict:
    members = np.array(sorted(int(v) for v in members), dtype=np.int64)
    member_set = set(int(v) for v in members)
    pairs = [(u, v) for u, v in inst.edges if u in member_set and v in member_set]
    in_degree = {int(v): 0 for v in members}
    for u, v in pairs:
        in_degree[u] += 1
        in_degree[v] += 1
    for v in members:
        in_degree[int(v)] += int(inst.bonus[v])
    cards = [
        {
            "name": inst.names[v],
            "oracle_id": inst.oracle_ids[v],
            "type_line": inst.type_lines[v],
            "combos_in_deck": in_degree[int(v)],
            "combos_with_commander": int(inst.bonus[v]),
            "total_combos_in_graph": int(inst.A[v].sum() + inst.bonus[v]),
        }
        for v in members
    ]
    cards.sort(key=lambda c: (-c["combos_in_deck"], c["name"]))
    return {
        "commander": inst.commander,
        "k": len(members),
        "score": int(len(pairs) + inst.bonus[members].sum()),
        "combos_among_deck_cards": len(pairs),
        "combos_with_commander": int(inst.bonus[members].sum()),
        "cards": cards,
        "combos": sorted([inst.names[u], inst.names[v]] for u, v in pairs),
    }
