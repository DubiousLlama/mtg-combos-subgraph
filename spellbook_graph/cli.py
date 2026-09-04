"""Command line entry point.

    python -m spellbook_graph download                 # fetch the bulk export once
    python -m spellbook_graph build  [--input ...]     # build data/graph/{graph.json,edges.tsv,graph.npz}
    python -m spellbook_graph stats  [--graph ...]     # print summary of a built graph
    python -m spellbook_graph features [--input ...]   # tally produced feature names (validate the "Infinite" filter)
    python -m spellbook_graph search [--commander ...] # find the k cards with the most combos between them
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .download import BULK_VARIANTS_GZIP_URL, DEFAULT_USER_AGENT, download_bulk
from .filters import FilterConfig
from .graph import ComboGraph, build_graph, iter_variants


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("variant filters")
    group.add_argument("--include-examples", action="store_true", help="also accept status=E (example) variants, not only status=OK")
    group.add_argument("--include-spoilers", action="store_true", help="keep variants flagged as spoilers (unreleased cards)")
    group.add_argument("--ignore-legality", action="store_true", help="do not require legalities.commander")
    group.add_argument("--allow-near-infinite", action="store_true", help="count 'Near-infinite ...' results as infinite")
    group.add_argument("--require-standalone", action="store_true", help="only count infinite features with status S (standalone)")
    group.add_argument("--card-count", type=int, default=2, help="cards per combo (default 2; 0 = any size, builds a hypergraph)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spellbook-graph", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="download the Commander Spellbook bulk export")
    p_download.add_argument("--out", type=Path, default=Path("data/variants.json.gz"))
    p_download.add_argument("--url", default=BULK_VARIANTS_GZIP_URL)
    p_download.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    p_download.add_argument("--force", action="store_true")

    p_build = sub.add_parser("build", help="build the two-card infinite combo graph")
    p_build.add_argument("--input", type=Path, default=Path("data/variants.json.gz"))
    p_build.add_argument("--out", type=Path, default=Path("data/graph"))
    p_build.add_argument("--top", type=int, default=25, help="how many top-degree cards to print")
    _add_filter_args(p_build)

    p_search = sub.add_parser("search", help="find the k cards inducing the most two-card combos")
    p_search.add_argument("--graph", type=Path, default=Path("data/graph"))
    p_search.add_argument("--out", type=Path, default=Path("data/deck"))
    p_search.add_argument("-k", "--slots", type=int, default=50, help="combo-piece slots in the deck (default 50)")
    p_search.add_argument("--commander", default="Kenrith, the Returned King", help="commander name, or 'none'")
    p_search.add_argument("--seconds", type=float, default=120.0, help="wall-clock budget per worker")
    p_search.add_argument("--jobs", type=int, default=0, help="parallel workers (default: all cores)")
    p_search.add_argument("--tabu-iters", type=int, default=600)
    p_search.add_argument("--rcl", type=int, default=8, help="greedy candidate list width")
    p_search.add_argument("--kick", type=int, default=6, help="cards replaced per perturbation")
    p_search.add_argument("--backend", choices=["cpu", "tt", "numpy"], default="cpu", help="cpu: per-core tabu search; tt: population search on Tenstorrent devices; numpy: the same population search on CPU")
    p_search.add_argument("--population", type=int, default=65536, help="decks per generation for tt/numpy backends (multiple of 32 x devices)")
    p_search.add_argument("--devices", type=int, default=4, help="Tenstorrent devices to use as a 1xN mesh")
    p_search.add_argument("--epoch-gens", type=int, default=300, help="generations between host-side reseeding / progress lines")
    p_search.add_argument("--seed", type=int, default=0)
    p_search.add_argument("--exclude", action="append", default=[], help="card name to leave out (repeatable)")
    p_search.add_argument("--strict", action="store_true", help="only combos with no notable prerequisites whose cards start on the battlefield or in hand")
    p_search.add_argument("--prereqs", choices=["any", "none", "kenrith"], default="any", help="prerequisite policy (kenrith: allow what the commander or the two cards provide)")
    p_search.add_argument("--result", action="append", default=[], help="regex a combo's produced feature must match, e.g. '^Infinite turns' (repeatable, any match counts)")

    p_stats = sub.add_parser("stats", help="summarise a built graph")
    p_stats.add_argument("--graph", type=Path, default=Path("data/graph"))
    p_stats.add_argument("--top", type=int, default=25)

    p_features = sub.add_parser("features", help="tally produced feature names among two-card, commander-legal, OK variants")
    p_features.add_argument("--input", type=Path, default=Path("data/variants.json.gz"))
    p_features.add_argument("--top", type=int, default=60)
    p_features.add_argument("--all-variants", action="store_true", help="tally every public variant, not only two-card ones")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_bulk(args.out, url=args.url, user_agent=args.user_agent, force=args.force)
        return 0
    if args.command == "build":
        if not args.input.exists():
            print(f"{args.input} not found. Run `python -m spellbook_graph download` first, or download\n{BULK_VARIANTS_GZIP_URL} yourself and pass --input.", file=sys.stderr)
            return 2
        config = FilterConfig.from_args(args)
        print(f"building graph from {args.input} with {config}", file=sys.stderr)
        if config.card_count == 0:
            from .hypergraph import build_hypergraph
            hyper = build_hypergraph(args.input, config)
            hyper.save(args.out)
            print(hyper.summary(top=args.top))
            print(f"wrote {args.out}/hypergraph.json, hypergraph.npz", file=sys.stderr)
            return 0
        graph = build_graph(args.input, config)
        graph.save(args.out)
        print(graph.summary(top=args.top))
        print(f"wrote {args.out}/graph.json, edges.tsv, graph.npz", file=sys.stderr)
        return 0
    if args.command == "features":
        from collections import Counter
        from .filters import INFINITE_RE, NEAR_INFINITE_RE
        counter: Counter[tuple[str, str]] = Counter()
        for variant in iter_variants(args.input):
            if variant.get("status") not in ("OK", "E") or not (variant.get("legalities") or {}).get("commander"):
                continue
            if not args.all_variants and (len(variant.get("uses") or []) != 2 or variant.get("requires")):
                continue
            for produced in variant.get("produces") or []:
                feature = produced.get("feature") or {}
                counter[(feature.get("name", ""), feature.get("status", ""))] += 1
        infinite = sum(c for (name, _), c in counter.items() if INFINITE_RE.match(name) and not NEAR_INFINITE_RE.match(name))
        near = sum(c for (name, _), c in counter.items() if NEAR_INFINITE_RE.match(name))
        print(f"distinct feature names: {len(counter):,}; 'Infinite...' mentions: {infinite:,}; 'Near-infinite...' mentions: {near:,}")
        print(f"{'count':>8}  status  name")
        for (name, status), count in counter.most_common(args.top):
            print(f"{count:8,}  {status:6}  {name}")
        return 0
    if args.command == "search":
        import os
        from .search import load_instance, parallel_search, describe
        inst = load_instance(args.graph, args.commander, exclude=args.exclude, results=args.result, strict=args.strict, prereqs=args.prereqs)
        jobs = args.jobs or min(32, os.cpu_count() or 1)
        print(f"{inst.m:,} candidate cards, {len(inst.edges):,} usable combos between them"
              + (f"; commander {inst.commander} adds {int(inst.bonus.sum())} free combos with {len(inst.commander_partners)} cards" if inst.commander else ""),
              file=sys.stderr)
        print(f"searching for {args.slots} cards: {jobs} workers x {args.seconds:.0f}s", file=sys.stderr)
        if args.backend in ("tt", "numpy"):
            from .driver import run
            result = run(inst, args.slots, args.seconds, args.out, backend=args.backend, population=args.population,
                         devices=args.devices, epoch_gens=args.epoch_gens, seed=args.seed, kick=args.kick)
            print(f"best: {result['score']} combos ({result['combos_among_deck_cards']} in-deck + {result['combos_with_commander']} with commander)")
            for card in result["cards"]:
                print(f"{card['combos_in_deck']:7d}  {card['name']}")
            return 0
        members, score, stats = parallel_search(args.graph, inst, args.slots, args.seconds, jobs, args.commander,
                                                tabu_iters=args.tabu_iters, rcl=args.rcl, kick=args.kick)
        result = describe(inst, members)
        result["search"] = {"seconds": args.seconds, "jobs": jobs, "workers": stats}
        result["source"] = ComboGraph.load(args.graph).source if (args.graph / "graph.json").exists() else {}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "deck.json").write_text(json.dumps(result, indent=2) + "\n")
        lines = [f"# {result['score']} unique two-card infinite combos among {result['k']} cards"]
        if inst.commander:
            lines.append(f"# commander: {inst.commander} ({result['combos_with_commander']} of them use the commander)")
        lines += [f"1 {c['name']}" for c in sorted(result["cards"], key=lambda c: c["name"])]
        (args.out / "deck.txt").write_text("\n".join(lines) + "\n")
        print(f"best: {result['score']} combos ({result['combos_among_deck_cards']} in-deck + {result['combos_with_commander']} with commander)")
        print(f"{'combos':>7}  card")
        for card in result["cards"]:
            print(f"{card['combos_in_deck']:7d}  {card['name']}")
        print(f"wrote {args.out}/deck.json and {args.out}/deck.txt", file=sys.stderr)
        return 0
    if args.command == "stats":
        graph = ComboGraph.load(args.graph)
        print(graph.summary(top=args.top))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
