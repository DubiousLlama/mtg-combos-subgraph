"""Build the two-card infinite combo graph.

Nodes are oracle cards. An edge joins two cards when at least one accepted
variant (see ``filters``) uses exactly those two cards. Edge weight is the
number of distinct variants for the pair; the optimisation objective the user
asked for counts *unique two-card combos*, i.e. unique edges, so the default
objective weight is 1 per edge and the variant count is kept as metadata.

Commander requirement
---------------------
Some variants only work when one of the two cards is your commander (for
example, a card that reads "if this is your commander"). Such an edge is
tagged ``commander_required`` with the oracle ids that satisfy it. An edge is
``unconditional`` when at least one of its variants has no such requirement.
The optimiser can either ignore conditional edges (safe) or model a commander
slot and count the conditional edges the chosen commander unlocks.

Color identity
--------------
The bulk export carries color identity per variant, not per card, so a card's
identity is only known as the intersection of the identities of every variant
it appears in (``identity_upper_bound``). With a five-colour commander there is
no identity constraint at all, which is the natural setting for this problem.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

from .filters import AcceptedVariant, FilterConfig, VariantClass, classify_variant

COLOR_ORDER = "WUBRG"


def _open_maybe_gzip(path: Path) -> IO[bytes]:
    path = Path(path)
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return path.open("rb")


def iter_variants(path: Path) -> Iterator[dict]:
    """Stream the ``variants`` array of the bulk document.

    Uses ijson when available (constant memory, the bulk file is large).
    Falls back to ``json.load`` otherwise. Also accepts a bare JSON array or
    the paginated ``{"results": [...]}`` shape of the HTTP API for fixtures.
    """
    try:
        import ijson  # type: ignore
    except ImportError:  # pragma: no cover - exercised only without ijson installed
        ijson = None
    if ijson is not None:
        with _open_maybe_gzip(path) as f:
            head = f.read(64)
        stream_prefix = "variants.item"
        stripped = head.lstrip()
        if stripped.startswith(b"["):
            stream_prefix = "item"
        elif b'"results"' in head:
            stream_prefix = "results.item"
        with _open_maybe_gzip(path) as f:
            yield from ijson.items(f, stream_prefix, use_float=True)
        return
    with _open_maybe_gzip(path) as f:
        document = json.load(io.TextIOWrapper(f, encoding="utf8"))
    if isinstance(document, list):
        yield from document
    else:
        yield from document.get("variants") or document.get("results") or []


def _identity_set(identity: str) -> frozenset[str]:
    return frozenset(c for c in identity.upper() if c in COLOR_ORDER)


def _identity_str(colors: Iterable[str]) -> str:
    colors = set(colors)
    return "".join(c for c in COLOR_ORDER if c in colors) or "C"


@dataclass
class Node:
    index: int
    oracle_id: str
    name: str
    spellbook_id: int
    type_line: str
    identity_upper_bound: frozenset[str] = field(default_factory=lambda: frozenset(COLOR_ORDER))
    variant_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "oracle_id": self.oracle_id,
            "name": self.name,
            "spellbook_id": self.spellbook_id,
            "type_line": self.type_line,
            "identity_upper_bound": _identity_str(self.identity_upper_bound),
            "n_variants": len(self.variant_ids),
        }


@dataclass
class Edge:
    u: int
    v: int
    variant_ids: list[str] = field(default_factory=list)
    unconditional: bool = False
    commander_required: set[str] = field(default_factory=set)  # oracle ids that unlock the edge
    identity: frozenset[str] = field(default_factory=frozenset)
    min_mana_value_needed: int | None = None
    bracket_tags: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    max_popularity: int | None = None
    clean: bool = False  # some variant has no notable prerequisites and battlefield/hand starts only
    no_notable_prerequisites: bool = False
    battlefield_or_hand: bool = False
    clean_features: set[str] = field(default_factory=set)  # features produced by clean variants
    variants: list[dict] = field(default_factory=list)  # per-variant: id, prerequisites, zones, features

    def add(self, variant: AcceptedVariant) -> None:
        self.variant_ids.append(variant.variant_id)
        required = variant.commander_required
        if required:
            self.commander_required.update(required)
        else:
            self.unconditional = True
        self.identity = self.identity | _identity_set(variant.identity)
        if variant.mana_value_needed is not None:
            self.min_mana_value_needed = variant.mana_value_needed if self.min_mana_value_needed is None else min(self.min_mana_value_needed, variant.mana_value_needed)
        if variant.bracket_tag:
            self.bracket_tags.add(variant.bracket_tag)
        self.features.update(variant.infinite_features)
        self.no_notable_prerequisites |= not variant.notable_prerequisites
        self.battlefield_or_hand |= variant.battlefield_or_hand
        if variant.clean:
            self.clean = True
            self.clean_features.update(variant.infinite_features)
        self.variants.append({
            "id": variant.variant_id,
            "notable_prerequisites": variant.notable_prerequisites,
            "battlefield_or_hand": variant.battlefield_or_hand,
            "commander_required": list(required),
            "features": list(variant.infinite_features),
        })
        if variant.popularity is not None:
            self.max_popularity = variant.popularity if self.max_popularity is None else max(self.max_popularity, variant.popularity)

    def to_json(self) -> dict:
        return {
            "u": self.u,
            "v": self.v,
            "n_variants": len(self.variant_ids),
            "variant_ids": self.variant_ids,
            "unconditional": self.unconditional,
            "commander_required": sorted(self.commander_required) if not self.unconditional else [],
            "identity": _identity_str(self.identity),
            "min_mana_value_needed": self.min_mana_value_needed,
            "bracket_tags": sorted(self.bracket_tags),
            "features": sorted(self.features),
            "max_popularity": self.max_popularity,
            "clean": self.clean,
            "no_notable_prerequisites": self.no_notable_prerequisites,
            "battlefield_or_hand": self.battlefield_or_hand,
            "clean_features": sorted(self.clean_features),
            "variants": self.variants,
        }


@dataclass
class ComboGraph:
    nodes: list[Node]
    edges: dict[tuple[int, int], Edge]
    rejections: Counter[str]
    config: FilterConfig
    source: dict[str, Any] = field(default_factory=dict)

    # --- construction -----------------------------------------------------
    @classmethod
    def empty(cls, config: FilterConfig) -> "ComboGraph":
        return cls(nodes=[], edges={}, rejections=Counter(), config=config)

    def _node_for(self, use, index_by_oracle: dict[str, int]) -> Node:
        key = use.oracle_id or f"spellbook:{use.spellbook_id}"
        idx = index_by_oracle.get(key)
        if idx is None:
            idx = len(self.nodes)
            index_by_oracle[key] = idx
            self.nodes.append(Node(index=idx, oracle_id=key, name=use.name, spellbook_id=use.spellbook_id, type_line=use.type_line))
        return self.nodes[idx]

    def add_variants(self, variants: Iterable[dict], progress_every: int = 50_000) -> None:
        index_by_oracle = {n.oracle_id: n.index for n in self.nodes}
        started = time.time()
        for count, variant in enumerate(variants, 1):
            kind, accepted = classify_variant(variant, self.config)
            self.rejections[kind.value] += 1
            if accepted is None:
                continue
            nodes = [self._node_for(use, index_by_oracle) for use in accepted.cards]
            identity = _identity_set(accepted.identity)
            for node in nodes:
                node.variant_ids.append(accepted.variant_id)
                node.identity_upper_bound = node.identity_upper_bound & identity
            u, v = sorted(n.index for n in nodes)
            edge = self.edges.get((u, v))
            if edge is None:
                edge = self.edges[(u, v)] = Edge(u=u, v=v)
            edge.add(accepted)
            if progress_every and count % progress_every == 0:
                print(f"  {count:,} variants scanned, {len(self.edges):,} edges, {len(self.nodes):,} cards ({time.time() - started:.0f}s)", file=sys.stderr)

    # --- derived data -----------------------------------------------------
    @property
    def n_variants_scanned(self) -> int:
        return sum(self.rejections.values())

    def degrees(self, unconditional_only: bool = False) -> list[int]:
        degree = [0] * len(self.nodes)
        for edge in self.edges.values():
            if unconditional_only and not edge.unconditional:
                continue
            degree[edge.u] += 1
            degree[edge.v] += 1
        return degree

    def adjacency(self, unconditional_only: bool = False) -> list[set[int]]:
        adjacency: list[set[int]] = [set() for _ in self.nodes]
        for edge in self.edges.values():
            if unconditional_only and not edge.unconditional:
                continue
            adjacency[edge.u].add(edge.v)
            adjacency[edge.v].add(edge.u)
        return adjacency

    def summary(self, top: int = 25) -> str:
        lines = [
            f"variants scanned: {self.n_variants_scanned:,}",
            "rejections: " + ", ".join(f"{k}={v:,}" for k, v in sorted(self.rejections.items())),
            f"cards (nodes): {len(self.nodes):,}",
            f"two-card infinite combos (edges): {len(self.edges):,}",
            f"  unconditional edges: {sum(1 for e in self.edges.values() if e.unconditional):,}",
            f"  commander-only edges: {sum(1 for e in self.edges.values() if not e.unconditional):,}",
            f"  variants behind edges: {sum(len(e.variant_ids) for e in self.edges.values()):,}",
        ]
        degree = self.degrees()
        unconditional = self.degrees(unconditional_only=True)
        order = sorted(range(len(self.nodes)), key=lambda i: -degree[i])[:top]
        lines.append(f"top {top} cards by degree (all / unconditional):")
        for i in order:
            lines.append(f"  {degree[i]:5d} / {unconditional[i]:5d}  {self.nodes[i].name}")
        return "\n".join(lines)

    # --- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "format": "mtg-combos-subgraph/graph/1",
            "source": self.source,
            "filter": {**{k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in self.config.__dict__.items()}},
            "rejections": dict(self.rejections),
            "nodes": [n.to_json() for n in self.nodes],
            "edges": [e.to_json() for _, e in sorted(self.edges.items())],
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "graph.json").write_text(json.dumps(self.to_json(), indent=1))
        with (directory / "edges.tsv").open("w") as f:
            f.write("u\tv\tunconditional\tn_variants\tcard_u\tcard_v\n")
            for (u, v), edge in sorted(self.edges.items()):
                f.write(f"{u}\t{v}\t{int(edge.unconditional)}\t{len(edge.variant_ids)}\t{self.nodes[u].name}\t{self.nodes[v].name}\n")
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return
        edges = sorted(self.edges.items())
        np.savez_compressed(
            directory / "graph.npz",
            u=np.array([u for (u, _), _ in edges], dtype=np.int32),
            v=np.array([v for (_, v), _ in edges], dtype=np.int32),
            unconditional=np.array([e.unconditional for _, e in edges], dtype=np.bool_),
            n_variants=np.array([len(e.variant_ids) for _, e in edges], dtype=np.int32),
            n_nodes=np.array(len(self.nodes), dtype=np.int64),
        )

    @classmethod
    def load(cls, directory: Path) -> "ComboGraph":
        document = json.loads((Path(directory) / "graph.json").read_text())
        raw_filter = document.get("filter", {})
        config = FilterConfig(**{k: (frozenset(v) if k == "statuses" else v) for k, v in raw_filter.items()})
        graph = cls.empty(config)
        graph.source = document.get("source", {})
        graph.rejections = Counter(document.get("rejections", {}))
        for n in document["nodes"]:
            graph.nodes.append(Node(
                index=n["index"], oracle_id=n["oracle_id"], name=n["name"], spellbook_id=n["spellbook_id"],
                type_line=n.get("type_line", ""), identity_upper_bound=_identity_set(n.get("identity_upper_bound", "WUBRG")),
                variant_ids=[""] * n.get("n_variants", 0),
            ))
        for e in document["edges"]:
            edge = Edge(
                u=e["u"], v=e["v"], variant_ids=e["variant_ids"], unconditional=e["unconditional"],
                commander_required=set(e["commander_required"]), identity=_identity_set(e["identity"]),
                min_mana_value_needed=e.get("min_mana_value_needed"), bracket_tags=set(e.get("bracket_tags", [])),
                features=set(e.get("features", [])), max_popularity=e.get("max_popularity"),
                clean=bool(e.get("clean", False)), no_notable_prerequisites=bool(e.get("no_notable_prerequisites", False)),
                battlefield_or_hand=bool(e.get("battlefield_or_hand", False)), clean_features=set(e.get("clean_features", [])),
                variants=list(e.get("variants", [])),
            )
            graph.edges[(edge.u, edge.v)] = edge
        return graph


def build_graph(path: Path, config: FilterConfig = FilterConfig(), progress_every: int = 50_000) -> ComboGraph:
    graph = ComboGraph.empty(config)
    graph.source = {"path": str(path)}
    header = _read_header(path)
    graph.source.update(header)
    graph.add_variants(iter_variants(path), progress_every=progress_every)
    return graph


def _read_header(path: Path) -> dict:
    """Pull ``timestamp`` and ``version`` from the front of the bulk document without parsing it all."""
    with _open_maybe_gzip(Path(path)) as f:
        head = f.read(4096).decode("utf8", errors="replace")
    result = {}
    for key in ("timestamp", "version"):
        marker = f'"{key}": '
        pos = head.find(marker)
        if pos >= 0:
            try:
                result[key] = json.JSONDecoder().raw_decode(head, pos + len(marker))[0]
            except ValueError:
                pass
    return result
