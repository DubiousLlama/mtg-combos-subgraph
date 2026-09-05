"""Trace replay vs eager in-place execution from the same state, then throughput. usage: P devices min_degree [device_id]"""
import sys, time; sys.path.insert(0, "/home/ttuser/mtg-combos-subgraph")
import numpy as np
from pathlib import Path
from spellbook_graph.hyper import load_hyper_instance
from spellbook_graph.tt_hyper import TTHyperPopulation
P, devices, md = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = [int(sys.argv[4])] if len(sys.argv) > 4 else None
inst = load_hyper_instance(Path("/home/ttuser/mtg-combos-subgraph/data/hypergraph"), min_degree=md)
G = 4
ps = TTHyperPopulation(inst, k=50, P=P, devices=devices, seed=3, trace=True, gens_per_trace=G, device_ids=dev)
nn = ps.ttnn; L = ps.layout
ps._ensure_device_state()
X0, cnt0, score0, salt0 = ps._down(ps._dX), ps._down(ps._dcnt), ps._down(ps._dscore), ps._down(ps.dsalt)
# eager: one block in place
t0 = time.time(); ps._block(); nn.synchronize_device(ps.device); t_eager = time.time() - t0
Xe, cnte, scoree, beste, bestXe = ps._down(ps._dX), ps._down(ps._dcnt), ps._down(ps._dscore), ps._down(ps._dbest), ps._down(ps._dbestX)
# host consistency of eager results
for p in range(0, P, max(1, P // 8)):
    deck = np.flatnonzero(Xe[:, p]); assert len(deck) == 50, len(deck)
    assert inst.score(deck) == int(scoree[0, p]), (inst.score(deck), scoree[0, p])
    bd = np.flatnonzero(bestXe[:, p]); assert inst.score(bd) == int(beste[0, p])
assert np.array_equal(cnte, L.counts(Xe)), "cnt drifted"
print(f"eager block of {G} gens: {t_eager*1e3:.0f} ms; scores mean {scoree.mean():.1f} max {scoree.max():.0f}; host rescoring consistent", flush=True)
# reset state and replay the same block through a trace
ps._write(X0, ps.bf16, ps._dX); ps._write(cnt0, ps.bf16, ps._dcnt); ps._write(score0, ps.f32, ps._dscore)
ps._write(np.zeros((ps.n, P), np.float32), ps.bf16, ps._dtabu); ps._write(np.full((1, P), -1.0, np.float32), ps.f32, ps._dbest); ps._write(X0, ps.bf16, ps._dbestX)
ps._write(salt0, ps.f32, ps.dsalt)
# capture: the capture pass itself executes the block once (compile pass already happened above as the eager block)
tid = nn.begin_trace_capture(ps.device, cq_id=0); ps._block(); nn.end_trace_capture(ps.device, tid, cq_id=0); nn.synchronize_device(ps.device)
Xc = ps._down(ps._dX)
print("capture pass == eager:", np.array_equal(Xc, Xe), np.array_equal(ps._down(ps._dcnt), cnte), np.array_equal(ps._down(ps._dscore), scoree), flush=True)
# reset again and replay
ps._write(X0, ps.bf16, ps._dX); ps._write(cnt0, ps.bf16, ps._dcnt); ps._write(score0, ps.f32, ps._dscore)
ps._write(np.zeros((ps.n, P), np.float32), ps.bf16, ps._dtabu); ps._write(np.full((1, P), -1.0, np.float32), ps.f32, ps._dbest); ps._write(X0, ps.bf16, ps._dbestX)
ps._write(salt0, ps.f32, ps.dsalt)
nn.execute_trace(ps.device, tid, cq_id=0, blocking=True)
Xr, cntr, scorer, bestr, bestXr = ps._down(ps._dX), ps._down(ps._dcnt), ps._down(ps._dscore), ps._down(ps._dbest), ps._down(ps._dbestX)
print("replay == eager:", np.array_equal(Xr, Xe), np.array_equal(cntr, cnte), np.array_equal(scorer, scoree), np.array_equal(bestr, beste), np.array_equal(bestXr, bestXe), flush=True)
# second replay differs (salt advanced) but stays consistent
nn.execute_trace(ps.device, tid, cq_id=0, blocking=True)
X2, s2, c2 = ps._down(ps._dX), ps._down(ps._dscore), ps._down(ps._dcnt)
ok = all(inst.score(np.flatnonzero(X2[:, p])) == int(s2[0, p]) for p in range(0, P, max(1, P // 8))) and np.array_equal(c2, L.counts(X2))
print("second replay consistent:", ok, "changed vs first:", not np.array_equal(X2, Xr), flush=True)
reps = 20; t0 = time.time()
for _ in range(reps): nn.execute_trace(ps.device, tid, cq_id=0, blocking=False)
nn.synchronize_device(ps.device); dt = (time.time() - t0) / (reps * G)
print(f"trace replay: {dt*1e3:.1f} ms/generation, {P/dt:,.0f} swaps/s on {devices} device(s) at P={P}", flush=True)
nn.release_trace(ps.device, tid)
ps.close()
