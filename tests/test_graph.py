import json
import subprocess
import sys
from pathlib import Path

import pytest

from spellbook_graph import ComboGraph, FilterConfig, VariantClass, build_graph, classify_variant
from spellbook_graph.graph import iter_variants

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "mini_variants.json"
FIXTURE_GZ = FIXTURES / "mini_variants.json.gz"


def _by_name(graph):
    return {n.name: n.index for n in graph.nodes}


def test_iter_variants_streams_both_plain_and_gzip():
    plain = list(iter_variants(FIXTURE))
    gz = list(iter_variants(FIXTURE_GZ))
    assert len(plain) == 16
    assert [v["id"] for v in plain] == [v["id"] for v in gz]


def test_classification_reasons():
    variants = {v["id"]: v for v in iter_variants(FIXTURE)}
    expected = {
        "1-2": VariantClass.NOT_INFINITE,  # "Win the game" is not an infinite result
        "4-5": VariantClass.ACCEPTED,
        "4-6": VariantClass.ACCEPTED,       # hidden-utility infinite feature still counts by default
        "7-8": VariantClass.ACCEPTED,
        "10-8": VariantClass.ACCEPTED,
        "9-8": VariantClass.NOT_INFINITE,   # near-infinite
        "4-t": VariantClass.NEEDS_TEMPLATE,
        "4-5-6": VariantClass.WRONG_CARD_COUNT,
        "13-8": VariantClass.NOT_COMMANDER_LEGAL,
        "12-8": VariantClass.SPOILER,
        "7-9": VariantClass.BAD_STATUS,
        "5-6": VariantClass.BAD_STATUS,
        "9-9": VariantClass.DUPLICATE_CARD,
    }
    for vid, kind in expected.items():
        assert classify_variant(variants[vid])[0] == kind, vid


def test_filter_options():
    variants = {v["id"]: v for v in iter_variants(FIXTURE)}
    assert classify_variant(variants["9-8"], FilterConfig(allow_near_infinite=True))[0] == VariantClass.ACCEPTED
    assert classify_variant(variants["7-9"], FilterConfig(statuses=frozenset({"OK", "E"})))[0] == VariantClass.ACCEPTED
    assert classify_variant(variants["13-8"], FilterConfig(require_commander_legal=False))[0] == VariantClass.ACCEPTED
    assert classify_variant(variants["12-8"], FilterConfig(include_spoilers=True))[0] == VariantClass.ACCEPTED
    # 4-6 only produces a hidden-utility infinite feature
    assert classify_variant(variants["4-6"], FilterConfig(require_standalone_feature=True))[0] == VariantClass.NOT_INFINITE
    assert classify_variant(variants["4-5-6"], FilterConfig(card_count=3))[0] == VariantClass.ACCEPTED


def test_graph_edges_and_commander_requirement():
    graph = build_graph(FIXTURE, progress_every=0)
    idx = _by_name(graph)
    assert graph.source["version"] == "test"
    assert graph.n_variants_scanned == 16
    assert graph.rejections["accepted"] == 6
    # Six accepted variants collapse into five unique pairs (7-8 and 7-8b share an edge).
    assert len(graph.edges) == 5
    scepter_reversal = graph.edges[tuple(sorted((idx["Isochron Scepter"], idx["Dramatic Reversal"])))]
    assert sorted(scepter_reversal.variant_ids) == ["7-8", "7-8b"]
    assert scepter_reversal.unconditional
    assert scepter_reversal.min_mana_value_needed == 4
    assert scepter_reversal.max_popularity == 50
    assert "Infinite mana" in scepter_reversal.features

    najeela = graph.edges[tuple(sorted((idx["Najeela, the Blade-Blossom"], idx["Dramatic Reversal"])))]
    assert not najeela.unconditional
    assert najeela.commander_required == {"oracle-najeela"}
    derevi = graph.edges[tuple(sorted((idx["Derevi, Empyrial Tactician"], idx["Dramatic Reversal"])))]
    assert not derevi.unconditional, "zoneLocations == ['C'] implies the card must be the commander"

    degrees = graph.degrees()
    assert degrees[idx["Dramatic Reversal"]] == 3
    assert graph.degrees(unconditional_only=True)[idx["Dramatic Reversal"]] == 1
    assert degrees[idx["Kiki-Jiki, Mirror Breaker"]] == 2

    # Identity upper bound: Dramatic Reversal appears in U, WUBRG and GWU variants -> U.
    assert graph.nodes[idx["Dramatic Reversal"]].to_json()["identity_upper_bound"] == "U"
    assert graph.nodes[idx["Kiki-Jiki, Mirror Breaker"]].to_json()["identity_upper_bound"] == "R"


def test_save_and_load_roundtrip(tmp_path):
    graph = build_graph(FIXTURE_GZ, progress_every=0)
    graph.save(tmp_path)
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "edges.tsv").exists()
    loaded = ComboGraph.load(tmp_path)
    assert [n.name for n in loaded.nodes] == [n.name for n in graph.nodes]
    assert set(loaded.edges) == set(graph.edges)
    for key, edge in graph.edges.items():
        assert loaded.edges[key].to_json() == edge.to_json()
    assert loaded.config == graph.config
    np = pytest.importorskip("numpy")
    npz = np.load(tmp_path / "graph.npz")
    assert int(npz["n_nodes"]) == len(graph.nodes)
    assert len(npz["u"]) == len(graph.edges)
    assert npz["unconditional"].sum() == 3


def test_cli_build_and_stats(tmp_path):
    out = tmp_path / "graph"
    result = subprocess.run([sys.executable, "-m", "spellbook_graph", "build", "--input", str(FIXTURE), "--out", str(out), "--top", "3"], capture_output=True, text=True, check=True)
    assert "two-card infinite combos (edges): 5" in result.stdout
    assert "Dramatic Reversal" in result.stdout
    result = subprocess.run([sys.executable, "-m", "spellbook_graph", "stats", "--graph", str(out)], capture_output=True, text=True, check=True)
    assert "cards (nodes): 7" in result.stdout
    document = json.loads((out / "graph.json").read_text())
    assert document["format"] == "mtg-combos-subgraph/graph/1"
    assert document["filter"]["statuses"] == ["OK"]
