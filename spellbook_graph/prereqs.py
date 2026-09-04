"""Classify Commander Spellbook `notablePrerequisites` lines.

Three modes for the search:
- ``any``:     ignore prerequisites (the original 243 search)
- ``none``:    only combos with no notable prerequisites at all (``--strict``)
- ``kenrith``: allow a prerequisite when the commander (Kenrith, the Returned
  King) or the two combo cards themselves satisfy it, deny the rest.

Kenrith's abilities: {R} haste+trample to all creatures, {1}{G} +1/+1 counter,
{2}{W} 5 life, {3}{U} draw, {4}{B} reanimate. So "no summoning sickness",
power/toughness thresholds, "+1/+1 counter on it", "a way to gain life / draw
a card / put a +1/+1 counter" are fine. Counting permanents, mana-producing
boards, opponent-dependent conditions, and "a way to <do X>" that needs a third
card are not. Every line of a prerequisite must be allowed for the variant to
pass; a line no rule matches is denied and reported so the lists can be tuned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ALLOW = [
    # Kenrith-satisfiable
    (r"summoning sick", "haste (Kenrith {R})"),
    (r"\bhas (haste|trample)\b", "haste/trample (Kenrith {R})"),
    (r"power (\d+|or) ", "power threshold (Kenrith +1/+1 counters)"),
    (r"toughness (\d+ or greater|greater than|at least)", "toughness threshold (Kenrith +1/+1 counters)"),
    (r"\+1/\+1 counters? on (it|them|each|.*)", "+1/+1 counters (Kenrith {1}{G})"),
    (r"way to put a \+1/\+1 counter", "+1/+1 counters (Kenrith {1}{G})"),
    (r"way to gain life", "life gain (Kenrith {2}{W})"),
    (r"gained life this turn", "life gain (Kenrith {2}{W})"),
    (r"life total is at least", "life total (Kenrith {2}{W})"),
    (r"way to draw a card", "card draw (Kenrith {3}{U})"),
    (r"creature card in (your|a) graveyard", "reanimation target (Kenrith {4}{B})"),
    (r"cast your commander from (your|the) command zone", "commander cast (trivially true)"),
    (r"deck size is (even|odd)|library has at least", "library state"),
    # inherent to the two cards / trivially arranged
    (r"attached to", "aura/equipment attached (part of casting it)"),
    (r"is a copy of|copying|entered as a copy", "clone copying the partner"),
    (r"chosen with|named with|naming", "name chosen"),
    (r"paired with", "soulbond pairing"),
    (r"enough mana to cast|mana to cast|mana available", "mana to cast the pieces"),
    (r"have not (activated|cast|attacked)|has not (attacked|been)", "fresh-turn state"),
    (r"is your commander|as your commander", "commander choice"),
    (r"in your hand$|in hand$", "card in hand"),
    (r"^(it is|during) your (turn|main phase|upkeep|combat)", "timing"),
    (r"life total is at least [2-9]$", "trivial life"),
]
_DENY = [
    (r"control (at least (two|three|four|five|six|seven|eight|nine|ten|\d+)|two|three|four|five|six|seven|eight|nine|ten|\d+)", "counting permanents you control"),
    (r"control (at least one|an?|another) (?!(other |additional )?creature\b)", "counting permanents you control"),
    (r"can (collectively )?tap to produce", "mana-producing board"),
    (r"opponent", "opponent-dependent"),
    (r"way to (deal|give|create|cast|sacrifice|destroy|exile|copy|untap|return|put a -1/-1|make|tap|discard|mill|blink|flicker)", "needs a third card"),
    (r"has (vigilance|lifelink|indestructible|flying|deathtouch|first strike|double strike|infect|flash|reach|menace|hexproof)", "keyword from elsewhere"),
    (r"is indestructible|are indestructible", "indestructible from elsewhere"),
    (r"city's blessing|max speed|monarch|initiative|ascend", "game-state milestone"),
    (r"cards in hand|hand size|castable|instant or sorcery card", "extra cards in hand"),
    (r"or less life|life total is (\d+ or less|less)", "low life"),
    (r"cannot be blocked|unblockable", "evasion from elsewhere"),
    (r"(?<!\+1/\+1 )counters? on (it|them)", "other counters"),
    (r"devotion|storm count|energy|experience", "resource count"),
]
# checked before the deny list: counts the deck itself satisfies
_PRE_ALLOW = [
    (r"control (at least )?(one|two|three|four|five|six|seven|eight|nine|ten|\d+|an?|another) (other |additional )?((white|blue|black|red|green|nontoken|legendary) )?creatures?( that (do not|don't) have summoning sickness)?\s*[.,]?$", "creature count (the deck is mostly creatures)"),
]
PRE_ALLOW = [(re.compile(p, re.I), why) for p, why in _PRE_ALLOW]
ALLOW = [(re.compile(p, re.I), why) for p, why in _ALLOW]
DENY = [(re.compile(p, re.I), why) for p, why in _DENY]


def split_lines(text: str) -> list[str]:
    return [l.strip() for l in re.split(r"[\n]+|(?<=[a-z\)])\.\s+", text or "") if l.strip()]


@dataclass
class Verdict:
    ok: bool
    allowed: list[tuple[str, str]] = field(default_factory=list)  # (line, why)
    denied: list[tuple[str, str]] = field(default_factory=list)


KENRITH_TYPE_LINE = "Legendary Creature — Human Noble"
_COUNT_RX = re.compile(r"control (?:at least )?(one|two|three|four|five|six|seven|eight|nine|ten|\d+) (other |additional )?([a-z' -]+?)s?\s*[.,]?$", re.I)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _supplied_by_pieces(line: str, type_lines: list[str]) -> str | None:
    """'You control at least two enchantments' is met when the combo pieces (and Kenrith) are those permanents."""
    m = _COUNT_RX.search(line)
    if not m:
        return None
    need = _WORDS.get(m.group(1).lower()) or int(m.group(1))
    other = bool(m.group(2))
    kind = m.group(3).strip().lower()
    if any(w in kind for w in (" with ", " that ", " which ", "mana", "untapped", "tapped", "token")):
        return None
    words = kind.split()
    if len(words) != 1 and not (len(words) == 2 and words[0] in ("nonland", "noncreature", "legendary", "nontoken")):
        return None
    noun = words[-1]
    singular = {"elve": "elf", "elves": "elf", "dwarves": "dwarf", "wolves": "wolf"}.get(noun, noun)
    def has(tl: str) -> bool:
        t = tl.lower()
        if singular == "permanent":
            ok = "instant" not in t and "sorcery" not in t
        else:
            ok = singular in t
        if words[0] == "nonland":
            ok = ok and "land" not in t
        if words[0] == "noncreature":
            ok = ok and "creature" not in t
        if words[0] == "nontoken":
            ok = ok and True
        return ok
    have = sum(1 for tl in type_lines if has(tl))
    if other:
        have -= 1
    return f"{need} {kind}: supplied by the combo pieces and Kenrith" if have >= need else None


def classify(text: str, type_lines: list[str] | None = None) -> Verdict:
    """`type_lines`: type lines of the combo's cards; Kenrith's is added automatically."""
    v = Verdict(ok=True)
    types = list(type_lines or []) + [KENRITH_TYPE_LINE]
    for line in split_lines(text):
        supplied = _supplied_by_pieces(line, types) or next((why for rx, why in PRE_ALLOW if rx.search(line)), None)
        if supplied:
            v.allowed.append((line, supplied))
            continue
        why_deny = next((why for rx, why in DENY if rx.search(line)), None)
        why_allow = next((why for rx, why in ALLOW if rx.search(line)), None)
        # a line that trips a deny rule is denied even if it also matches an allow rule
        # ("does not have summoning sickness and cannot be blocked")
        if why_deny:
            v.denied.append((line, why_deny))
        elif why_allow:
            v.allowed.append((line, why_allow))
        else:
            v.denied.append((line, "unclassified"))
    v.ok = not v.denied
    return v
