"""Decide which Commander Spellbook variants count as two-card infinite combos.

Variant shape (camelCase, as exported by the backend's ``export_variants`` task,
see ``backend/spellbook/serializers/variant_serializer.py`` in
SpaceCowMedia/commander-spellbook-backend)::

    {
      "id": "1234-5678",
      "status": "OK" | "E",                      # only OK and E(xample) are public
      "uses": [ {"card": {"id", "name", "oracleId", "spoiler", "typeLine", ...},
                 "zoneLocations": ["H","B","C","E","G","L"],
                 "mustBeCommander": bool, "quantity": int, ...}, ... ],
      "requires": [ {"template": {"id","name","scryfallQuery"}, ...}, ... ],
      "produces": [ {"feature": {"id","name","uncountable","status"}, "quantity"}, ... ],
      "identity": "WUBRG" | "C" | ...,
      "manaValueNeeded": int, "spoiler": bool, "bracketTag": "R"|"S"|"P"|...,
      "legalities": {"commander": bool, ...}, ...
    }

Feature statuses: HU hidden utility, PU public utility, H helper, C contextual,
S standalone. The backend's own bracket estimator treats a variant as
"relevant" when it produces at least one standalone (S) feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

INFINITE_RE = re.compile(r"^\s*infinite\b", re.IGNORECASE)
NEAR_INFINITE_RE = re.compile(r"^\s*near-?\s*infinite\b", re.IGNORECASE)


class VariantClass(str, Enum):
    ACCEPTED = "accepted"
    BAD_STATUS = "bad_status"
    NOT_COMMANDER_LEGAL = "not_commander_legal"
    SPOILER = "spoiler"
    NEEDS_TEMPLATE = "needs_template"
    WRONG_CARD_COUNT = "wrong_card_count"
    DUPLICATE_CARD = "duplicate_card"
    NOT_INFINITE = "not_infinite"


@dataclass(frozen=True)
class FilterConfig:
    statuses: frozenset[str] = frozenset({"OK"})
    require_commander_legal: bool = True
    include_spoilers: bool = False
    allow_near_infinite: bool = False
    require_standalone_feature: bool = False
    card_count: int = 2  # 0 = any number of cards (hypergraph build)

    @classmethod
    def from_args(cls, args: Any) -> "FilterConfig":
        statuses = {"OK"}
        if getattr(args, "include_examples", False):
            statuses.add("E")
        return cls(
            statuses=frozenset(statuses),
            require_commander_legal=not getattr(args, "ignore_legality", False),
            include_spoilers=getattr(args, "include_spoilers", False),
            allow_near_infinite=getattr(args, "allow_near_infinite", False),
            require_standalone_feature=getattr(args, "require_standalone", False),
            card_count=getattr(args, "card_count", 2),
        )


def _get(obj: dict, camel: str, snake: str | None = None, default: Any = None) -> Any:
    """Read a key in camelCase, falling back to snake_case (defensive)."""
    if camel in obj:
        return obj[camel]
    if snake is not None and snake in obj:
        return obj[snake]
    return default


def _is_infinite_feature(feature: dict, config: FilterConfig) -> bool:
    name = feature.get("name", "")
    if config.require_standalone_feature and feature.get("status") != "S":
        return False
    if NEAR_INFINITE_RE.match(name):
        return config.allow_near_infinite
    return bool(INFINITE_RE.match(name))


@dataclass
class CardUse:
    spellbook_id: int
    oracle_id: str
    name: str
    type_line: str
    spoiler: bool
    zone_locations: tuple[str, ...]
    must_be_commander: bool
    quantity: int

    @classmethod
    def from_json(cls, use: dict) -> "CardUse":
        card = use["card"]
        zones = tuple(_get(use, "zoneLocations", "zone_locations", default=()) or ())
        must = bool(_get(use, "mustBeCommander", "must_be_commander", default=False))
        # A card whose only allowed starting zone is the command zone must be the commander.
        if zones and set(zones) == {"C"}:
            must = True
        return cls(
            spellbook_id=int(card["id"]),
            oracle_id=str(_get(card, "oracleId", "oracle_id", default="")),
            name=card["name"],
            type_line=str(_get(card, "typeLine", "type_line", default="") or ""),
            spoiler=bool(card.get("spoiler", False)),
            zone_locations=zones,
            must_be_commander=must,
            quantity=int(use.get("quantity", 1) or 1),
        )


@dataclass
class AcceptedVariant:
    variant_id: str
    cards: list[CardUse]
    identity: str
    mana_value_needed: int | None
    bracket_tag: str | None
    status: str
    infinite_features: list[str]
    all_features: list[str] = field(default_factory=list)
    popularity: int | None = None
    notable_prerequisites: str = ""
    easy_prerequisites: str = ""

    @property
    def battlefield_or_hand(self) -> bool:
        """Every card starts on the battlefield or in hand (no graveyard / exile / library / command-zone setup)."""
        return all(set(c.zone_locations) <= {"B", "H"} for c in self.cards)

    @property
    def clean(self) -> bool:
        """No notable prerequisites and ordinary starting zones: the two cards are all you need."""
        return not self.notable_prerequisites and self.battlefield_or_hand

    @property
    def commander_required(self) -> tuple[str, ...]:
        """Oracle ids of the cards that must be this deck's commander for the combo to work."""
        return tuple(c.oracle_id for c in self.cards if c.must_be_commander)


def classify_variant(variant: dict, config: FilterConfig = FilterConfig()) -> tuple[VariantClass, AcceptedVariant | None]:
    status = variant.get("status", "")
    if status not in config.statuses:
        return VariantClass.BAD_STATUS, None
    legalities = variant.get("legalities") or {}
    if config.require_commander_legal and not legalities.get("commander", False):
        return VariantClass.NOT_COMMANDER_LEGAL, None
    if not config.include_spoilers and variant.get("spoiler", False):
        return VariantClass.SPOILER, None
    if variant.get("requires"):
        return VariantClass.NEEDS_TEMPLATE, None
    uses = variant.get("uses") or []
    if len(uses) != config.card_count and config.card_count > 0:
        return VariantClass.WRONG_CARD_COUNT, None
    if not uses:
        return VariantClass.WRONG_CARD_COUNT, None
    cards = [CardUse.from_json(u) for u in uses]
    if any(c.quantity != 1 for c in cards) or len({c.oracle_id for c in cards}) != len(cards):
        return VariantClass.DUPLICATE_CARD, None
    features = [p["feature"] for p in (variant.get("produces") or []) if p.get("feature")]
    infinite = [f["name"] for f in features if _is_infinite_feature(f, config)]
    if not infinite:
        return VariantClass.NOT_INFINITE, None
    mana_value = _get(variant, "manaValueNeeded", "mana_value_needed")
    accepted = AcceptedVariant(
        variant_id=str(variant["id"]),
        cards=cards,
        identity=str(variant.get("identity", "") or ""),
        mana_value_needed=int(mana_value) if mana_value is not None else None,
        bracket_tag=_get(variant, "bracketTag", "bracket_tag"),
        status=status,
        infinite_features=infinite,
        all_features=[f["name"] for f in features],
        popularity=variant.get("popularity"),
        notable_prerequisites=str(_get(variant, "notablePrerequisites", "notable_prerequisites", default="") or "").strip(),
        easy_prerequisites=str(_get(variant, "easyPrerequisites", "easy_prerequisites", default="") or "").strip(),
    )
    return VariantClass.ACCEPTED, accepted


def classify_many(variants: Iterable[dict], config: FilterConfig = FilterConfig()) -> Iterable[tuple[VariantClass, AcceptedVariant | None]]:
    for variant in variants:
        yield classify_variant(variant, config)
