"""Device-side reseed: after trace replays, reseed some replicas and check cnt/score against the host layout,
then replay again and check host consistency. usage: check_reseed.py P devices min_degree [device_id]"""
import sys, time; sys.path.insert(0, "/home/ttuser/mtg-combos-subgraph")
import numpy as np
from pathlib import Path
from spellbook_graph.hyper import load_hyper_instance
from spellbook_graph.tt_hyper import TTHyperPopulation
P, devices, md = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = [int(sys.argv[4])] if len(sys.argv) > 4 else None
inst = load_hyper_instance(Path("/home/ttuser/mtg-combos-subgraph/data/hypergraph"), min_degree=md)
ps = TTHyperPopulation(inst, k=50, P=P, devices=devices, seed=3, trace=True, gens_per_trace=4, device_ids=dev, stale_epochs=1)
nn = ps.ttnn; L = ps.layout
best = ps.run_epoch(8)
print(f"epoch 1: best {best}, {ps.last_epoch}", flush=True)
# recount from the current device X must reproduce the device's incremental cnt/score exactly
X, cnt, score = ps._down(ps._dX), ps._down(ps._dcnt), ps._down(ps._dscore)
t0 = time.time(); ps._recount(); nn.synchronize_device(ps.device); dt = time.time() - t0
cnt2, score2 = ps._down(ps._dcnt), ps._down(ps._dscore)
print(f"recount in {dt*1e3:.0f} ms: cnt equal {np.array_equal(cnt, cnt2)}, score equal {np.array_equal(score, score2)}, host cnt equal {np.array_equal(cnt2, L.counts(X))}", flush=True)
# force a reseed of every replica (stale_epochs=1 and no improvement) through run_epoch's bookkeeping, then replay again
for ep in range(2, 6):
    best = ps.run_epoch(8)
    X, cnt, score = ps._down(ps._dX), ps._down(ps._dcnt), ps._down(ps._dscore)
    ok = np.array_equal(cnt, L.counts(X)) and np.array_equal(score[0], L.scores(X).astype(np.float32)) and np.all(X.sum(0) == 50)
    print(f"epoch {ep}: best {best}, reseeded {ps.last_epoch['reseeded']}, device state consistent with host: {ok}, transfers {ps.transfer_seconds:.1f}s", flush=True)
    assert ok
ps.close()
print("OK")
