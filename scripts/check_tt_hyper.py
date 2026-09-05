"""Bit-for-bit check of the TT hypergraph step against the NumPy reference, then timing.
usage: check_tt_hyper.py P devices min_degree gens fidelity s_dtype"""
import sys, time; sys.path.insert(0, "/home/ttuser/mtg-combos-subgraph")
import numpy as np
from pathlib import Path
from spellbook_graph.hyper import load_hyper_instance
from spellbook_graph.hyperpop import generation, quantize_noise
from spellbook_graph.tt_hyper import TTHyperPopulation
P, devices, min_degree, gens = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fid = sys.argv[5] if len(sys.argv) > 5 else "hifi2"; sdt = sys.argv[6] if len(sys.argv) > 6 else "bfloat16"
inst = load_hyper_instance(Path("/home/ttuser/mtg-combos-subgraph/data/hypergraph"), min_degree=min_degree)
t0 = time.time()
ps = TTHyperPopulation(inst, k=50, P=P, devices=devices, seed=3, fidelity=fid, s_dtype=sdt, device_ids=[int(sys.argv[7])] if len(sys.argv) > 7 else None)
L = ps.layout
print(f"pool {inst.m} cards, {L.E_all} combos, I={L.I} slots ({L.I/len(inst.edge_cards):.2f}x raw), chunks={L.chunks}, n={ps.n}; setup {time.time()-t0:.1f}s", flush=True)
ps._ensure_device_state()
nn = ps.ttnn
X, cnt, tabu, score = ps._dX, ps._dcnt, ps._dtabu, ps._dscore
hX, hcnt, htabu, hscore = ps.X.copy(), ps.cnt.copy(), ps.tabu.copy(), ps.score.copy()
rng = np.random.default_rng(7)
ok = True
for g in range(gens):
    no = quantize_noise(rng.random((ps.n, P), dtype=np.float32)); ni = quantize_noise(rng.random((ps.n, P), dtype=np.float32))
    dno, dni = ps._up(no, ps.bf16), ps._up(ni, ps.bf16)
    t1 = time.time()
    X, cnt, tabu, score = ps._device_generation(X, cnt, tabu, score, dno, dni)
    nn.synchronize_device(ps.device); dt = time.time() - t1
    hX, hcnt, htabu, hscore, b, a = generation(L, hX, hcnt, htabu, hscore, ps.tenure, no, ni)
    gX, gcnt, gtabu, gscore = ps._down(X), ps._down(cnt), ps._down(tabu), ps._down(score)[0]
    dx, dc, dtb, ds = np.abs(gX - hX).sum(), np.abs(gcnt - hcnt).sum(), np.abs(gtabu - htabu).sum(), np.abs(gscore - hscore).sum()
    print(f"gen {g}: {dt*1e3:.1f} ms  X diff {dx:.0f}  cnt diff {dc:.0f}  tabu diff {dtb:.0f}  score diff {ds:.0f}  host score mean {hscore.mean():.1f} max {hscore.max():.0f}  deck sizes ok {np.all(gX.sum(0)==50)}", flush=True)
    ok &= dx == 0 and dc == 0 and dtb == 0 and ds == 0
print("EXACT" if ok else "MISMATCH")
# timing of a full epoch with device noise
t1 = time.time(); n_t = 10
for _ in range(n_t):
    X, cnt, tabu, score = ps._device_generation(X, cnt, tabu, score)
nn.synchronize_device(ps.device); dt = (time.time() - t1) / n_t
print(f"steady state: {dt*1e3:.1f} ms/generation, {P/dt:,.0f} swaps/s on {devices} device(s)")
ps.close()
