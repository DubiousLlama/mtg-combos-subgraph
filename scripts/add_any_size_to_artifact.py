"""Add the any-size (hypergraph) deck to the Kenrith combo-cores artifact HTML.
usage: add_any_size_to_artifact.py SOURCE.html card_images.json deck.json OUT.html"""
import html, json, re, sys
src, cache, deckp, outp = sys.argv[1:5]
page = open(src).read()
imgs = json.load(open(cache))
d = json.load(open(deckp))

def kind(t):
    return "artifact" if t.startswith("Artifact") and "Creature" not in t else ("enchantment" if t.startswith("Enchantment") and "Creature" not in t else "creature")

vid = "any-size"
entry = {
    "label": "All infinite results, any size · 50 cards",
    "note": "Combos of any number of cards — 2 to 10 in the data, most of them 3 or 4 — counted once per unique card set, prerequisites allowed. Best found, not a proven maximum",
    "score": d["score"], "k": d["k"],
    "headline": "unique infinite combos of any size — the best fifty cards found, with no proof it is the most",
    "pool": 6475, "pool_combos": 92836, "maxsize": max(len(c) for c in d["combos"]),
    "proof": "none · same value from 28 CPU tabu workers and a 4-card population search",
    "by_size": d["combos_by_size"],
    "cards": [{"n": c["name"], "t": c["type_line"], "k": c["combos_in_deck"], "g": c["total_combos_in_pool"], "c": kind(c["type_line"]),
               "mana": imgs.get(c["name"], {}).get("mana", ""), "url": imgs.get(c["name"], {}).get("url", "")} for c in d["cards"]],
    "combos": d["combos"],
}
missing = [c["name"] for c in d["cards"] if c["name"] not in imgs]
print("cards without image:", missing)

def sub(a, b, count=1):
    global page
    assert page.count(a) >= 1, a[:80]
    page = page.replace(a, b, count)

# data: DECKS entry and images
i = page.index("const DECKS = ") + len("const DECKS = "); j = page.index("\n", i)
decks = json.loads(page[i:j].rstrip(";")); decks[vid] = entry
page = page[:i] + json.dumps(decks) + ";" + page[j:]
i = page.index("const IMG = ") + len("const IMG = "); j = page.index("\n", i)
IMG = json.loads(page[i:j].rstrip(";"))
for c in d["cards"]:
    if c["name"] in imgs and c["name"] not in IMG:
        IMG[c["name"]] = imgs[c["name"]]["img"]
page = page[:i] + json.dumps(IMG) + ";" + page[j:]

# dropdown, lede, summary table, method
sub('<optgroup label="No notable prerequisites">', '<optgroup label="Combos of any size">'
    f'<option value="{vid}">{html.escape(entry["label"])} — {d["score"]:,}</option></optgroup><optgroup label="No notable prerequisites">')
sub("Each list below is the proven maximum for its rules (HiGHS mixed-integer program, gap 0), found first by a population search on four Tenstorrent Blackhole cards.",
    "Each two-card list below is the proven maximum for its rules (HiGHS mixed-integer program, gap 0), found first by a population search on four Tenstorrent Blackhole cards. The any-size list counts combos of three, four and more cards too; it is the best two independent searches found, without a proof.")
sub('<tr><td>no notable prerequisites</td><td>all infinite results</td>',
    f'<tr><td>prerequisites allowed</td><td>all infinite results, any size</td><td class="num">6,475</td><td class="num">92,836</td><td class="num">50</td><td class="num">{d["score"]:,}†</td></tr>'
    '<tr><td>no notable prerequisites</td><td>all infinite results</td>')
by = d["combos_by_size"]
sub("Optima are rarely unique; ties are shown as found by the cards.",
    "Optima are rarely unique; ties are shown as found by the cards. † The any-size list is different in kind: every public combo of any length is a hyperedge (93,934 unique card sets of 2 to 10 cards over 6,515 cards; Kenrith drops out of the 346 that contain him), a set counts when all of its cards are in the deck, and the objective is a densest-k-subhypergraph, "
    f"which the mixed-integer program cannot close. The {d['score']:,} figure ({by.get('2', 0)} two-card, {by.get('3', 0):,} three-card, {by.get('4', 0)} four-card) is the best of 28 CPU tabu workers and of a population search in incidence space on the four Blackhole cards; both stop at the same two values, {d['score']:,} and 1,692, from every random start.")
# column titles no longer two-card specific
sub('title="two-card combos with other cards in this deck"', 'title="combos with other cards in this deck"')
sub('title="two-card combos this card has in the whole filtered pool"', 'title="combos this card has in the whole filtered pool"')
# script: combos of any length (pairs stay pairs)
sub("partners = {}; DATA.cards.forEach(c => partners[c.n] = []);\n  DATA.combos.forEach(([a,b]) => { partners[a].push(b); partners[b].push(a); });",
    "shared = {}; DATA.cards.forEach(c => shared[c.n] = {});\n  DATA.combos.forEach(cs => cs.forEach(a => cs.forEach(b => { if (a !== b) shared[a][b] = (shared[a][b] || 0) + 1; })));\n  partners = Object.fromEntries(Object.entries(shared).map(([n, m]) => [n, Object.keys(m)]));")
sub("let DATA, partners, byName,", "let DATA, partners, shared, byName,")
sub("<li>proof <span>HiGHS, gap 0</span></li>`;", "<li>proof <span>${DATA.proof || 'HiGHS, gap 0'}</span></li>`;")
sub("document.getElementById('allcombos').innerHTML = DATA.combos.map(([a,b]) => `<div>${a}<em>+</em>${b}</div>`).join('');",
    "document.getElementById('allcombos').innerHTML = DATA.combos.map(cs => `<div>${cs.join('<em>+</em>')}</div>`).join('');")
sub("const ps = partners[name].slice().sort((x,y) => byName[y].k - byName[x].k || x.localeCompare(y));",
    "const many = DATA.maxsize > 2;\n  const ps = partners[name].slice().sort((x,y) => (many ? shared[name][y] - shared[name][x] : 0) || byName[y].k - byName[x].k || x.localeCompare(y));")
sub("document.getElementById('partners').innerHTML = ps.map(p => `<li data-card=\"${p.replace(/\"/g,'&quot;')}\">${p}<span>${byName[p].k}</span></li>`).join('');",
    "document.getElementById('partners').innerHTML = ps.map(p => `<li data-card=\"${p.replace(/\"/g,'&quot;')}\">${p}<span>${many ? shared[name][p] + ' together' : byName[p].k}</span></li>`).join('');")
sub("<h2>Partners</h2>", "<h2 id=\"partnerstitle\">Partners</h2>")
sub("document.getElementById('selname').textContent = name;",
    "document.getElementById('selname').textContent = name;\n  document.getElementById('partnerstitle').textContent = many ? 'Partners · combos shared' : 'Partners';")
sub("const links = DATA.combos.map(([a,b]) => ({ source: a, target: b }));",
    "const pair = {}; DATA.combos.forEach(cs => { for (let i = 0; i < cs.length; i++) for (let j = i + 1; j < cs.length; j++) { const key = cs[i] < cs[j] ? cs[i] + '\\u0000' + cs[j] : cs[j] + '\\u0000' + cs[i]; pair[key] = (pair[key] || 0) + 1; } });\n  const links = Object.entries(pair).map(([key, w]) => { const [a, b] = key.split('\\u0000'); return { source: a, target: b, w }; });\n  const many = DATA.maxsize > 2;")
sub(".force('link', d3.forceLink(links).id(d => d.id).distance(70).strength(0.35))",
    ".force('link', d3.forceLink(links).id(d => d.id).distance(many ? 90 : 70).strength(many ? d => Math.min(0.5, 0.05 + 0.03 * d.w) : 0.35))")
sub("edgeSel = gEdges.selectAll('line').data(links).join('line').attr('class', 'edge');",
    "edgeSel = gEdges.selectAll('line').data(links).join('line').attr('class', 'edge').style('stroke-width', d => many ? Math.min(6, 0.6 + Math.log2(d.w) * 0.9) + 'px' : null).style('stroke-opacity', d => many ? Math.min(0.9, 0.12 + 0.12 * Math.log2(d.w)) : null);")
sub("function highlight(name) {\n  if (!name) { edgeSel.attr('class', 'edge');",
    "function highlight(name) {\n  if (!name) { edgeSel.attr('class', 'edge').style('stroke-opacity', d => DATA.maxsize > 2 ? Math.min(0.9, 0.12 + 0.12 * Math.log2(d.w)) : null);")
sub("edgeSel.attr('class', d => 'edge ' + ((d.source.id === name || d.target.id === name) ? 'on' : 'off'));",
    "edgeSel.attr('class', d => 'edge ' + ((d.source.id === name || d.target.id === name) ? 'on' : 'off')).style('stroke-opacity', null);")
sub("document.getElementById('tipmeta').textContent = `${c.mana ? c.mana + ' · ' : ''}${c.k} combos in deck · ${c.g} in pool`;",
    "document.getElementById('tipmeta').textContent = `${c.mana ? c.mana + ' · ' : ''}${c.k} combos in deck · ${c.g} in pool`;")
open(outp, "w").write(page)
print("wrote", outp, len(page) // 1024, "KB")
