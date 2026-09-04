"""Build the infinite-combo *hypergraph*: combos of any size.

Same filters as the two-card graph (``filters.classify_variant`` with
``card_count=0``), but a variant becomes a hyperedge over its full card set
instead of an edge between two cards. Variants that use the same card set
collapse into one hyperedge, because the objective counts unique combos.

Stored as ``hypergraph.json`` (nodes, hyperedges with per-variant prerequisite
text so search-time policies can change without rescanning the export) plus
``hypergraph.npz`` with flat CSR arrays for the optimisers.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .filters import AcceptedVariant, FilterConfig, classify_variant
from .graph import Node, _identity_set, _identity_str, _read_header, iter_variants


@dataclass
class HyperEdge:
    cards: tuple[int, ...]
    variant_ids: list[str] = field(default_factory=list)
    unconditional: bool = False
    commander_required: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    clean: bool = False
    clean_features: set[str] = field(default_factory=set)
    max_popularity: int | None = None
    variants: list[dict] = field(default_factory=list)

    def add(self, variant: AcceptedVariant) -> None:
        self.variant_ids.append(variant.variant_id)
        required = variant.commander_required
        if required:
            self.commander_required.update(required)
        else:
            self.unconditional = True
        self.features.update(variant.infinite_features)
        if variant.clean:
            self.clean = True
            self.clean_features.update(variant.infinite_features)
        if variant.popularity is not None:
            self.max_popularity = variant.popularity if self.max_popularity is None else max(self.max_popularity, variant.popularity)
        self.variants.append({
            "id": variant.variant_id,
            "notable_prerequisites": variant.notable_prerequisites,
            "battlefield_or_hand": variant.battlefield_or_hand,
            "commander_required": list(required),
            "features": list(variant.infinite_features),
        })

    def to_json(self) -> dict:
        return {
            "cards": list(self.cards),
            "n_variants": len(self.variant_ids),
            "variant_ids": self.variant_ids,
            "unconditional": self.unconditional,
            "commander_required": sorted(self.commander_required) if not self.unconditional else [],
            "features": sorted(self.features),
            "clean": self.clean,
            "clean_features": sorted(self.clean_features),
            "max_popularity": self.max_popularity,
            "variants": self.variants,
        }


@dataclass
class HyperGraph:
    nodes: list[Node]
    edges: dict[tuple[int, ...], HyperEdge]
    rejections: Counter
    config: FilterConfig
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, config: FilterConfig) -> "HyperGraph":
        return cls(nodes=[], edges={}, rejections=Counter(), config=config)

    def add_variants(self, variants: Iterable[dict], progress_every: int = 50_000) -> None:
        index_by_oracle = {n.oracle_id: n.index for n in self.nodes}
        started = time.time()
        for count, variant in enumerate(variants, 1):
            kind, accepted = classify_variant(variant, self.config)
            self.rejections[kind.value] += 1
            if accepted is None:
                continue
            idx = []
            for use in accepted.cards:
                key = use.oracle_id or f"spellbook:{use.spellbook_id}"
                i = index_by_oracle.get(key)
                if i is None:
                    i = len(self.nodes)
                    index_by_oracle[key] = i
                    self.nodes.append(Node(index=i, oracle_id=key, name=use.name, spellbook_id=use.spellbook_id, type_line=use.type_line))
                node = self.nodes[i]
                node.variant_ids.append(accepted.variant_id)
                node.identity_upper_bound = node.identity_upper_bound & _identity_set(accepted.identity)
                idx.append(i)
            cards = tuple(sorted(idx))
            edge = self.edges.get(cards)
            if edge is None:
                edge = self.edges[cards] = HyperEdge(cards=cards)
            edge.add(accepted)
            if progress_every and count % progress_every == 0:
                print(f"  {count:,} variants scanned, {len(self.edges):,} combos, {len(self.nodes):,} cards ({time.time() - started:.0f}s)", file=sys.stderr)

    def summary(self, top: int = 25) -> str:
        sizes = Counter(len(c) for c in self.edges)
        degree = Counter()
        for cards in self.edges:
            for c in cards:
                degree[c] += 1
        lines = [
            f"variants scanned: {sum(self.rejections.values()):,}",
            "rejections: " + ", ".join(f"{k}={v:,}" for k, v in sorted(self.rejections.items())),
            f"cards (nodes): {len(self.nodes):,}",
            f"infinite combos (unique card sets): {len(self.edges):,}",
            "  by size: " + ", ".join(f"{r}: {c:,}" for r, c in sorted(sizes.items())),
            f"  commander-only combos: {sum(1 for e in self.edges.values() if not e.unconditional):,}",
            f"  incidences (sum of sizes): {sum(len(c) for c in self.edges):,}",
            f"top {top} cards by number of combos:",
        ]
        for i, d in degree.most_common(top):
            lines.append(f"  {d:6d}  {self.nodes[i].name}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "format": "mtg-combos-subgraph/hypergraph/1",
            "source": self.source,
            "filter": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in self.config.__dict__.items()},
            "rejections": dict(self.rejections),
            "nodes": [n.to_json() for n in self.nodes],
            "hyperedges": [e.to_json() for _, e in sorted(self.edges.items())],
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "hypergraph.json").write_text(json.dumps(self.to_json()))
        edges = sorted(self.edges)
        ptr = np.zeros(len(edges) + 1, dtype=np.int64)
        ptr[1:] = np.cumsum([len(c) for c in edges])
        np.savez_compressed(
            directory / "hypergraph.npz",
            edge_ptr=ptr,
            edge_cards=np.array([c for cards in edges for c in cards], dtype=np.int32),
            unconditional=np.array([self.edges[c].unconditional for c in edges], dtype=np.bool_),
            n_nodes=np.array(len(self.nodes), dtype=np.int64),
        )

    @classmethod
    def load(cls, directory: Path) -> "HyperGraph":
        document = json.loads((Path(directory) / "hypergraph.json").read_text())
        raw_filter = document.get("filter", {})
        config = FilterConfig(**{k: (frozenset(v) if k == "statuses" else v) for k, v in raw_filter.items()})
        graph = cls.empty(config)
        graph.source = document.get("source", {})
        graph.rejections = Counter(document.get("rejections", {}))
        for n in document["nodes"]:
            graph.nodes.append(Node(index=n["index"], oracle_id=n["oracle_id"], name=n["name"], spellbook_id=n["spellbook_id"],
                                    type_line=n.get("type_line", ""), identity_upper_bound=_identity_set(n.get("identity_upper_bound", "WUBRG")),
                                    variant_ids=[""] * n.get("n_variants", 0)))
        for e in document["hyperedges"]:
            edge = HyperEdge(cards=tuple(e["cards"]), variant_ids=e["variant_ids"], unconditional=e["unconditional"],
                             commander_required=set(e["commander_required"]), features=set(e["features"]), clean=e["clean"],
                             clean_features=set(e["clean_features"]), max_popularity=e.get("max_popularity"), variants=e["variants"])
            graph.edges[edge.cards] = edge
        return graph


def build_hypergraph(path: Path, config: FilterConfig, progress_every: int = 50_000) -> HyperGraph:
    graph = HyperGraph.empty(config)
    graph.source = {"path": str(path), **_read_header(path)}
    graph.add_variants(iter_variants(path), progress_every=progress_every)
    return graph
