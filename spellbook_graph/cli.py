"""Command line entry point.

    python -m spellbook_graph download                 # fetch the bulk export once
    python -m spellbook_graph build  [--input ...]     # build data/graph/{graph.json,edges.tsv,graph.npz}
    python -m spellbook_graph stats  [--graph ...]     # print summary of a built graph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .download import BULK_VARIANTS_GZIP_URL, DEFAULT_USER_AGENT, download_bulk
from .filters import FilterConfig
from .graph import ComboGraph, build_graph


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("variant filters")
    group.add_argument("--include-examples", action="store_true", help="also accept status=E (example) variants, not only status=OK")
    group.add_argument("--include-spoilers", action="store_true", help="keep variants flagged as spoilers (unreleased cards)")
    group.add_argument("--ignore-legality", action="store_true", help="do not require legalities.commander")
    group.add_argument("--allow-near-infinite", action="store_true", help="count 'Near-infinite ...' results as infinite")
    group.add_argument("--require-standalone", action="store_true", help="only count infinite features with status S (standalone)")
    group.add_argument("--card-count", type=int, default=2, help="cards per combo (default 2)")


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

    p_stats = sub.add_parser("stats", help="summarise a built graph")
    p_stats.add_argument("--graph", type=Path, default=Path("data/graph"))
    p_stats.add_argument("--top", type=int, default=25)

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
        graph = build_graph(args.input, config)
        graph.save(args.out)
        print(graph.summary(top=args.top))
        print(f"wrote {args.out}/graph.json, edges.tsv, graph.npz", file=sys.stderr)
        return 0
    if args.command == "stats":
        graph = ComboGraph.load(args.graph)
        print(graph.summary(top=args.top))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
