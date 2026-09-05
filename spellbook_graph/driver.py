"""Run the population search for a wall-clock budget with progress on the console.

Prints one line per epoch (elapsed, generations, swaps/s, best score, how many
replicas sit at the best, how many were reseeded) and writes the incumbent deck
to disk every time it improves, so an interrupted run still leaves a result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .population import PopulationSearch
from .search import Instance, describe


def _write_result(out: Path, inst: Instance, deck: np.ndarray, extra: dict) -> dict:
    members = np.flatnonzero(deck[: inst.m])
    result = describe(inst, members)
    result["search"] = extra
    out.mkdir(parents=True, exist_ok=True)
    (out / "deck.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [f"# {result['score']} unique two-card infinite combos among {result['k']} cards"]
    if inst.commander:
        lines.append(f"# commander: {inst.commander} ({result['combos_with_commander']} of them use the commander)")
    lines += [f"1 {c['name']}" for c in sorted(result["cards"], key=lambda c: c["name"])]
    (out / "deck.txt").write_text("\n".join(lines) + "\n")
    return result


def run(
    inst: Instance,
    k: int,
    seconds: float,
    out: Path,
    backend: str = "tt",
    population: int = 65536,
    devices: int = 4,
    epoch_gens: int = 300,
    seed: int = 0,
    tenure: int = 5,
    kick: int = 6,
    stale_epochs: int = 3,
) -> dict:
    A = inst.A.astype(np.float32)
    bonus = inst.bonus.astype(np.float32)
    t0 = time.time()
    if backend == "tt":
        from .tt_search import TTPopulation

        ps = TTPopulation(A, bonus, k=k, P=population, seed=seed, tenure=tenure, kick=kick, stale_epochs=stale_epochs, devices=devices)
        where = f"{devices} Tenstorrent device(s)"
    else:
        ps = PopulationSearch(A, bonus, k=k, P=population, seed=seed, tenure=tenure, kick=kick, stale_epochs=stale_epochs)
        where = "NumPy on CPU"
    print(f"[{time.time() - t0:6.1f}s] population {population:,} decks of {k} on {where}; {epoch_gens} generations per epoch; budget {seconds:.0f}s", flush=True)

    deadline = t0 + seconds
    epoch = 0
    last_best = -1
    history = []
    try:
        while time.time() < deadline:
            te = time.time()
            best = ps.run_epoch(epoch_gens)
            epoch += 1
            elapsed = time.time() - t0
            swaps = ps.generations * population
            stats = getattr(ps, "last_epoch", {})
            transfer = getattr(ps, "transfer_seconds", 0.0)
            line = (
                f"[{elapsed:6.1f}s] epoch {epoch:4d}  gens {ps.generations:7,d}  swaps {swaps / 1e6:8.1f}M  "
                f"{population * epoch_gens / (time.time() - te) / 1e6:5.2f}M swaps/s  best {best:4d}  "
                f"(epoch best {stats.get('epoch_best', best)}, mean {stats.get('mean_best', 0):.1f}, "
                f"{stats.get('replicas_at_best', 0)} replicas at best, {stats.get('reseeded', 0)} reseeded"
                + (f", {transfer:.1f}s in transfers" if transfer else "")
                + ")"
            )
            history.append({"elapsed": elapsed, "generations": ps.generations, "best": best, **stats})
            if best > last_best:
                last_best = best
                result = _write_result(out, inst, ps.best_deck, {"backend": backend, "population": population, "elapsed": elapsed, "generations": ps.generations})
                line += f"  ** new best, wrote {out}/deck.txt"
            print(line, flush=True)
    except KeyboardInterrupt:
        print("interrupted; keeping the incumbent", file=sys.stderr, flush=True)
    finally:
        if hasattr(ps, "close"):
            ps.close()
    result = _write_result(
        out,
        inst,
        ps.best_deck,
        {"backend": backend, "population": population, "devices": devices if backend == "tt" else 0, "seconds": time.time() - t0, "generations": ps.generations, "swaps": ps.generations * population, "history": history},
    )
    return result


def run_hyper(
    inst,
    k: int,
    seconds: float,
    out: Path,
    backend: str = "tt",
    population: int = 4096,
    devices: int = 4,
    epoch_gens: int = 100,
    seed: int = 0,
    tenure: int = 5,
    kick: int = 6,
    stale_epochs: int = 3,
    gens_per_trace: int = 4,
    trace: bool = True,
    device_ids: list[int] | None = None,
    seed_decks: list[np.ndarray] | None = None,
    seed_fraction: float = 0.0,
    big_kick: int = 20,
) -> dict:
    """Population search on a `HyperInstance` (combos of any size) for a wall-clock budget.

    Mirrors `run`: one progress line per epoch, the incumbent written with `hyper.write_result`
    whenever it improves, and every new incumbent rescored independently with `HyperInstance.score`
    (the device keeps scores incrementally; a mismatch means a device bug, not a better deck).
    """
    from .hyper import write_result

    t0 = time.time()
    kw = dict(seed=seed, tenure=tenure, kick=kick, stale_epochs=stale_epochs, big_kick=big_kick,
              seed_decks=seed_decks, seed_fraction=seed_fraction)
    if backend == "tt":
        from .tt_hyper import TTHyperPopulation

        if trace:
            epoch_gens = max(gens_per_trace, (epoch_gens // gens_per_trace) * gens_per_trace)
        ps = TTHyperPopulation(inst, k, population, devices=devices, device_ids=device_ids, trace=trace, gens_per_trace=gens_per_trace, **kw)
        where = f"{devices} Tenstorrent device(s)" + (f", trace of {gens_per_trace} generations" if trace else ", eager")
    else:
        from .hyperpop import HyperPopulation

        ps = HyperPopulation(inst, k, population, **kw)
        where = "NumPy on CPU"
    L = ps.layout
    print(f"[{time.time() - t0:6.1f}s] population {population:,} decks of {k} on {where}; {epoch_gens} generations per epoch; "
          f"budget {seconds:.0f}s; {inst.m:,} cards, {L.E_all:,} combos, {L.I:,} slots", flush=True)

    deadline = t0 + seconds
    epoch = 0
    last_best = -1
    history = []
    extra = {"backend": backend, "population": population, "devices": devices if backend == "tt" else 0, "trace": trace and backend == "tt",
             "gens_per_trace": gens_per_trace, "epoch_gens": epoch_gens, "seed": seed, "min_pool_degree": int(inst.degree.min())}
    try:
        while time.time() < deadline:
            te = time.time()
            best = ps.run_epoch(epoch_gens)
            epoch += 1
            elapsed = time.time() - t0
            stats = getattr(ps, "last_epoch", {})
            transfer = getattr(ps, "transfer_seconds", 0.0)
            rate = population * epoch_gens / (time.time() - te)
            line = (
                f"[{elapsed:6.1f}s] epoch {epoch:4d}  gens {ps.generations:7,d}  swaps {ps.generations * population / 1e6:7.2f}M  "
                f"{rate / 1e3:6.1f}k swaps/s  best {best:5d}  (epoch best {stats.get('epoch_best', best)}, mean of replica bests {stats.get('mean_best', 0):.1f}, "
                f"{stats.get('replicas_at_best', 0)} replicas at {best}, {stats.get('reseeded', 0)} reseeded"
                + (f", {transfer:.1f}s in transfers" if transfer else "") + ")"
            )
            history.append({"elapsed": round(elapsed, 1), "generations": ps.generations, "best": best, **stats})
            if best > last_best:
                members = np.flatnonzero(ps.best_deck[: inst.m])
                check = inst.score(members)
                if check != best or members.size != k:
                    raise RuntimeError(f"device score {best} but host rescoring gives {check} for a {members.size}-card deck")
                last_best = best
                write_result(out, inst, members, {**extra, "elapsed": round(elapsed, 1), "generations": ps.generations})
                line += f"  ** new best (rescored {check}), wrote {out}/deck.txt"
            print(line, flush=True)
    except KeyboardInterrupt:
        print("interrupted; keeping the incumbent", file=sys.stderr, flush=True)
    finally:
        if hasattr(ps, "close"):
            ps.close()
    members = np.flatnonzero(ps.best_deck[: inst.m])
    return write_result(out, inst, members, {**extra, "seconds": round(time.time() - t0, 1), "generations": ps.generations,
                                              "swaps": ps.generations * population, "history": history})
