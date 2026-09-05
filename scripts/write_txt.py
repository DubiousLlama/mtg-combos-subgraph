"""Write a deck + full combo list as plain text. usage: write_txt.py deck.json out.txt"""
import sys, json
d = json.load(open(sys.argv[1]))
lines = [f"# {d['score']} unique infinite combos of any size among {d['k']} cards; commander {d['commander']}",
         "# by size: " + ", ".join(f"{s}-card {c}" for s, c in d["combos_by_size"].items()),
         f"# source: Commander Spellbook bulk export; search: {d['search'].get('backend')}", "", "## Deck (combos in deck per card)"]
lines += [f"{c['combos_in_deck']:5d}  {c['name']}" for c in d["cards"]]
lines += ["", f"## Combos ({len(d['combos'])})"]
lines += [" + ".join(c) for c in d["combos"]]
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
print(sys.argv[2], len(lines), "lines")
