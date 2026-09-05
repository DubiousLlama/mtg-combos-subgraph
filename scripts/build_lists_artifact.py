import json
decks = []
for path, label in [("data/deck_any_cpu/deck.json", "1,745"), ("data/deck_any_cpu_1692/deck.json", "1,692")]:
    d = json.load(open(path))
    decks.append({"label": label, "score": d["score"], "by_size": d["combos_by_size"],
                  "cards": [[c["name"], c["combos_in_deck"], c["total_combos_in_pool"]] for c in d["cards"]],
                  "combos": d["combos"]})
data = json.dumps(decks, separators=(",", ":"))
html = r'''<title>Any-Size Combo Decks</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F3F5F7;--panel:#FFFFFF;--ink:#1B2430;--muted:#66717E;--rule:#D8DEE5;--accent:#0E7C7B;--accent-ink:#FFFFFF;--hl:#E3F2F1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141A21;--panel:#1C242D;--ink:#E6EBF0;--muted:#98A3AF;--rule:#2C3640;--accent:#3FB3B1;--accent-ink:#0F1A1A;--hl:#1F3634}}
:root[data-theme="dark"]{--bg:#141A21;--panel:#1C242D;--ink:#E6EBF0;--muted:#98A3AF;--rule:#2C3640;--accent:#3FB3B1;--accent-ink:#0F1A1A;--hl:#1F3634}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0;padding:32px 24px 48px}
h1{font-size:26px;font-weight:600;margin:0 0 4px;text-wrap:balance}
.sub{color:var(--muted);margin:0 0 20px;max-width:70ch}
.tabs{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}
.tab{border:1px solid var(--rule);background:var(--panel);color:var(--ink);padding:8px 14px;border-radius:6px;cursor:pointer;font:inherit;font-weight:500}
.tab[aria-selected="true"]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.tab:focus-visible,.filter:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.meta{display:flex;gap:24px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-bottom:16px}
.meta b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:minmax(280px,1fr) minmax(320px,2fr);gap:24px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px 18px;min-width:0}
.panel h2{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;text-align:left;vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--rule)}
td.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;color:var(--muted);white-space:nowrap}
td.n.k{color:var(--ink)}
tr.hit td{background:var(--hl)}
.filter{width:100%;font:inherit;padding:8px 10px;border:1px solid var(--rule);border-radius:6px;background:var(--bg);color:var(--ink);margin-bottom:10px}
.count{font-size:13px;color:var(--muted);margin-bottom:8px}
ol{margin:0;padding-left:0;list-style:none;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;line-height:1.6;max-height:70vh;overflow:auto}
ol li{padding:1px 0;border-bottom:1px dotted var(--rule)}
ol li span{color:var(--muted)}
.copy{margin-top:10px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);padding:6px 12px;border-radius:6px;font:inherit;font-size:13px;cursor:pointer}
</style>
<h1>Any-Size Combo Decks</h1>
<p class="sub">Fifty-card combo cores with Kenrith, the Returned King in the command zone, counting every unique infinite combo of any size from the Commander Spellbook export (any result, any prerequisites). Two attractors of the CPU tabu search; the Tenstorrent search is still running.</p>
<div class="tabs" role="tablist" id="tabs"></div>
<div class="meta" id="meta"></div>
<div class="grid">
  <section class="panel"><h2>Deck</h2><table><thead><tr><th>Card</th><th style="text-align:right">in deck</th><th style="text-align:right">in pool</th></tr></thead><tbody id="cards"></tbody></table>
  <button class="copy" id="copy">Copy decklist</button></section>
  <section class="panel"><h2>Combos</h2><input class="filter" id="filter" type="search" placeholder="Filter by card name" aria-label="Filter combos by card name"><div class="count" id="count"></div><ol id="combos"></ol></section>
</div>
<script>
const DECKS = __DATA__;
let cur = 0, q = "";
const $ = id => document.getElementById(id);
function render(){
  const d = DECKS[cur];
  $("tabs").innerHTML = DECKS.map((x,i)=>`<button class="tab" role="tab" aria-selected="${i===cur}" data-i="${i}">${x.label} combos</button>`).join("");
  $("meta").innerHTML = `<span><b>${d.score.toLocaleString()}</b> combos</span>` + Object.entries(d.by_size).map(([s,c])=>`<span><b>${c.toLocaleString()}</b> ${s}-card</span>`).join("") + `<span><b>${d.cards.length}</b> cards</span>`;
  const ql = q.trim().toLowerCase();
  const hit = n => ql && n.toLowerCase().includes(ql);
  $("cards").innerHTML = d.cards.map(([n,k,t])=>`<tr class="${hit(n)?"hit":""}"><td>${n}</td><td class="n k">${k}</td><td class="n">${t}</td></tr>`).join("");
  const rows = d.combos.filter(c => !ql || c.some(n => n.toLowerCase().includes(ql)));
  $("count").textContent = ql ? `${rows.length.toLocaleString()} of ${d.combos.length.toLocaleString()} combos mention "${q.trim()}"` : `${d.combos.length.toLocaleString()} combos, one per line`;
  $("combos").innerHTML = rows.map(c => `<li>${c.join(' <span>+</span> ')}</li>`).join("");
}
$("tabs").addEventListener("click", e => { const b = e.target.closest("[data-i]"); if (b) { cur = +b.dataset.i; render(); } });
$("filter").addEventListener("input", e => { q = e.target.value; render(); });
$("copy").addEventListener("click", () => {
  const txt = DECKS[cur].cards.map(c => "1 " + c[0]).sort().join("\n");
  navigator.clipboard.writeText(txt).then(()=>{ $("copy").textContent = "Copied"; setTimeout(()=>$("copy").textContent="Copy decklist",1500); });
});
render();
</script>
'''.replace("__DATA__", data)
open("/tmp/claude-1000/-home-ttuser/06da2477-9484-4ec8-9307-e791f317b106/scratchpad/any-size-decks.html", "w").write(html)
print(len(html))
