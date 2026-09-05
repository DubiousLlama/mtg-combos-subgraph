"""Fetch Scryfall images for every card in the given deck.json files. usage: fetch_card_images.py card_images.json deck.json [deck.json ...]"""
import json, time, urllib.request, urllib.parse, io, base64, sys, os
from PIL import Image
cache_path = sys.argv[1]  # card_images.json cache, then deck.json files
out = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
names = []
for f in sys.argv[2:]:
    for c in json.load(open(f))["cards"]:
        if c["name"] not in out and c["name"] not in names: names.append(c["name"])
hdr = {"User-Agent": "mtg-combos-subgraph/0.1 (deck visualisation; one-off fetch)", "Accept": "application/json"}
for name in names:
    url = "https://api.scryfall.com/cards/named?" + urllib.parse.urlencode({"exact": name})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30) as r: card = json.load(r)
        uris = card.get("image_uris") or card["card_faces"][0]["image_uris"]
        with urllib.request.urlopen(urllib.request.Request(uris["normal"], headers={"User-Agent": hdr["User-Agent"]}), timeout=60) as r: raw = r.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((320, 447))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=78, optimize=True)
        out[name] = {"img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "url": card["scryfall_uri"],
                     "mana": card.get("mana_cost") or (card.get("card_faces") or [{}])[0].get("mana_cost", "")}
    except Exception as e:
        print(f"FAIL {name}: {e}")
    time.sleep(0.12)
json.dump(out, open(cache_path, "w"))
print(f"fetched {len(names)} new; cache holds {len(out)} cards, {sum(len(v['img']) for v in out.values())//1024} KB")
