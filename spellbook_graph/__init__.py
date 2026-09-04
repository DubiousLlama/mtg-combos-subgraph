"""Two-card infinite combo graph built from the Commander Spellbook bulk export.

Data source: https://json.commanderspellbook.com/variants.json.gz, published by
Commander Spellbook (https://commanderspellbook.com/) precisely so that tools
do not page through the HTTP API. See ``spellbook_graph.download``.
"""

from .filters import FilterConfig, classify_variant, VariantClass
from .graph import ComboGraph, build_graph

__all__ = ["FilterConfig", "classify_variant", "VariantClass", "ComboGraph", "build_graph"]
