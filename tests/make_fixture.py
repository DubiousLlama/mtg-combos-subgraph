"""Generate tests/fixtures/mini_variants.json in the exact shape of the bulk export.

Field names follow the camelCase output of the backend's export_variants task
(VariantSerializer + djangorestframework_camel_case.util.camelize).
"""

import json
from pathlib import Path

CARDS = {
    "Thassa's Oracle": ("thassas", "Creature — Merfolk Wizard"),
    "Demonic Consultation": ("consult", "Instant"),
    "Tainted Pact": ("pact", "Instant"),
    "Kiki-Jiki, Mirror Breaker": ("kiki", "Legendary Creature — Goblin Shaman"),
    "Zealous Conscripts": ("conscripts", "Creature — Human Warrior"),
    "Pestermite": ("pestermite", "Creature — Faerie Rogue"),
    "Isochron Scepter": ("scepter", "Artifact"),
    "Dramatic Reversal": ("reversal", "Instant"),
    "Sol Ring": ("solring", "Artifact"),
    "Najeela, the Blade-Blossom": ("najeela", "Legendary Creature — Human Warrior"),
    "Derevi, Empyrial Tactician": ("derevi", "Legendary Creature — Bird Wizard"),
    "Unreleased Card": ("unreleased", "Creature"),
    "Paradox Engine": ("paradox", "Legendary Artifact"),
}
_ids = {name: i + 1 for i, name in enumerate(CARDS)}


def card(name, spoiler=False):
    oracle, type_line = CARDS[name]
    return {
        "id": _ids[name], "name": name, "oracleId": f"oracle-{oracle}", "spoiler": spoiler,
        "faces": [], "typeLine": type_line,
        "imageUriFrontPng": None, "imageUriFrontLarge": None, "imageUriFrontNormal": None,
        "imageUriFrontSmall": None, "imageUriFrontArtCrop": None, "layoutRotationFront": None,
        "imageUriBackPng": None, "imageUriBackLarge": None, "imageUriBackNormal": None,
        "imageUriBackSmall": None, "imageUriBackArtCrop": None,
    }


def use(name, zones=("H",), must_be_commander=False, quantity=1, spoiler=False):
    return {
        "card": card(name, spoiler=spoiler), "zoneLocations": list(zones),
        "battlefieldCardState": "", "exileCardState": "", "libraryCardState": "", "graveyardCardState": "",
        "mustBeCommander": must_be_commander, "quantity": quantity, "usedFace": None,
    }


def feature(name, status="S"):
    return {"feature": {"id": abs(hash(name)) % 10000, "name": name, "uncountable": False, "status": status}, "quantity": 1}


def variant(vid, uses, produces, *, status="OK", requires=(), identity="C", mana_value_needed=2, spoiler=False, commander_legal=True, bracket="R", popularity=10):
    return {
        "id": vid, "status": status, "uses": uses, "requires": list(requires), "produces": produces,
        "of": [{"id": 1}], "includes": [{"id": 1}], "identity": identity,
        "manaNeeded": "{2}", "manaValueNeeded": mana_value_needed,
        "easyPrerequisites": "", "notablePrerequisites": "", "description": "...", "notes": "",
        "popularity": popularity, "spoiler": spoiler, "bracketTag": bracket,
        "legalities": {"commander": commander_legal, "pauperCommanderMain": False, "pauperCommander": False,
                       "oathbreaker": True, "predh": False, "standardBrawl": False, "brawl": False,
                       "competitiveBrawl": False, "alchemy": False, "vintage": True, "legacy": True,
                       "premodern": False, "modern": False, "pioneer": False, "standard": False, "pauper": False},
        "prices": {"tcgplayer": "1.00", "cardkingdom": "1.00", "cardmarket": "1.00"},
        "variantCount": 1,
    }


template = {"template": {"id": 7, "name": "A creature with power 4 or greater", "scryfallQuery": "t:creature pow>=4", "scryfallApi": None},
            "zoneLocations": ["B"], "battlefieldCardState": "", "exileCardState": "", "libraryCardState": "",
            "graveyardCardState": "", "mustBeCommander": False, "quantity": 1}

VARIANTS = [
    # accepted: plain two-card wins
    variant("1-2", [use("Thassa's Oracle"), use("Demonic Consultation")], [feature("Win the game")], identity="UB"),
    variant("1-3", [use("Thassa's Oracle"), use("Tainted Pact")], [feature("Win the game")], identity="UB"),
    # accepted: infinite results
    variant("4-5", [use("Kiki-Jiki, Mirror Breaker"), use("Zealous Conscripts")], [feature("Infinite ETB"), feature("Infinite hasty creature tokens")], identity="R"),
    variant("4-6", [use("Kiki-Jiki, Mirror Breaker"), use("Pestermite")], [feature("Infinite ETB", status="HU")], identity="UR"),
    variant("7-8", [use("Isochron Scepter"), use("Dramatic Reversal")], [feature("Infinite untap of nonland permanents"), feature("Infinite colorless mana")], identity="U", mana_value_needed=4),
    # second variant of the same pair -> same edge, two variants
    variant("7-8b", [use("Isochron Scepter"), use("Dramatic Reversal")], [feature("Infinite mana")], identity="U", mana_value_needed=6, popularity=50),
    # commander-only edge: Najeela must be the commander
    variant("10-8", [use("Najeela, the Blade-Blossom", zones=("C",), must_be_commander=True), use("Dramatic Reversal")], [feature("Infinite combat phases")], identity="WUBRG"),
    # commander-only via zone C only (mustBeCommander not set)
    variant("11-8", [use("Derevi, Empyrial Tactician", zones=("C",)), use("Dramatic Reversal")], [feature("Infinite untap")], identity="GWU"),
    # rejected: near-infinite only
    variant("9-8", [use("Sol Ring"), use("Dramatic Reversal")], [feature("Near-infinite colorless mana")], identity="U"),
    # rejected: needs a template
    variant("4-t", [use("Kiki-Jiki, Mirror Breaker"), use("Pestermite")], [feature("Infinite damage")], requires=[template], identity="UR"),
    # rejected: three cards
    variant("4-5-6", [use("Kiki-Jiki, Mirror Breaker"), use("Zealous Conscripts"), use("Pestermite")], [feature("Infinite damage")], identity="UR"),
    # rejected: not commander legal
    variant("13-8", [use("Paradox Engine"), use("Dramatic Reversal")], [feature("Infinite mana")], identity="U", commander_legal=False),
    # rejected: spoiler
    variant("12-8", [use("Unreleased Card", spoiler=True), use("Dramatic Reversal")], [feature("Infinite mana")], identity="U", spoiler=True),
    # rejected by default: example status
    variant("7-9", [use("Isochron Scepter"), use("Sol Ring")], [feature("Infinite mana")], status="E", identity="C"),
    # rejected: needs review
    variant("5-6", [use("Zealous Conscripts"), use("Pestermite")], [feature("Infinite ETB")], status="NR", identity="UR"),
    # rejected: quantity 2 of one card
    variant("9-9", [use("Sol Ring", quantity=2), use("Dramatic Reversal")], [feature("Infinite mana")], identity="U"),
]

DOCUMENT = {"timestamp": "2026-09-01T00:00:00+00:00", "version": "test", "variants": VARIANTS, "aliases": []}

if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures" / "mini_variants.json"
    out.write_text(json.dumps(DOCUMENT, indent=1))
    print(f"wrote {out} ({len(VARIANTS)} variants)")
