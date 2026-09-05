"""python cpu_run.py SECONDS JOBS OUT [min_degree]"""
import sys, time, json; sys.path.insert(0, "/home/ttuser/mtg-combos-subgraph")
from pathlib import Path
import numpy as np
from spellbook_graph.hyper import *
seconds, jobs, out = float(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
min_degree = int(sys.argv[4]) if len(sys.argv) > 4 else 1
inst = load_hyper_instance(Path("/home/ttuser/mtg-combos-subgraph/data/hypergraph"), min_degree=min_degree)
print(f"pool {inst.m} cards, {inst.E} combos, {len(inst.edge_cards)} incidences; {jobs} workers x {seconds:.0f}s", flush=True)
t0 = time.time()
members, score, stats = parallel_search(inst, 50, seconds, jobs)
print(f"best {score} in {time.time()-t0:.0f}s; per-worker: {sorted(s['score'] for s in stats)}", flush=True)
res = write_result(out, inst, members, {"backend": "cpu-tabu", "seconds": seconds, "jobs": jobs, "workers": stats, "min_degree": min_degree})
print(json.dumps(res["combos_by_size"]), "min pool degree of deck cards:", min(c["total_combos_in_pool"] for c in res["cards"]))
for c in res["cards"][:10]: print(f"{c['combos_in_deck']:6d}  {c['name']}")
