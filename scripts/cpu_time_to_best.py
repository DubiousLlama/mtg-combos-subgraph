"""Time for independent CPU tabu workers to first reach a target score. usage: cpu_time_to_best.py SECONDS JOBS MIN_DEGREE [target]"""
import sys, time, multiprocessing as mp; sys.path.insert(0, "/home/ttuser/mtg-combos-subgraph")
from pathlib import Path
from spellbook_graph.hyper import load_hyper_instance, search
seconds, jobs, md = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
target = int(sys.argv[4]) if len(sys.argv) > 4 else 1745
INST = None
def init(i):
    global INST; INST = i
def worker(seed):
    t0 = time.time(); log = []
    def report(score, restarts, kicks):
        log.append((round(time.time() - t0, 1), score, restarts, kicks))
    members, score, stats = search(INST, 50, seconds, seed=seed, report=report)
    hit = next((t for t, s, *_ in log if s >= target), None)
    return seed, score, hit, log
if __name__ == "__main__":
    inst = load_hyper_instance(Path("/home/ttuser/mtg-combos-subgraph/data/hypergraph"), min_degree=md)
    print(f"pool {inst.m} cards, {inst.E} combos; {jobs} workers x {seconds:.0f}s; target {target}", flush=True)
    with mp.Pool(jobs, initializer=init, initargs=(inst,)) as pool:
        for seed, score, hit, log in pool.imap_unordered(worker, range(jobs)):
            print(f"seed {seed}: final {score}, reached {target} at {hit}s; trajectory {[(t, s) for t, s, *_ in log][-6:]}", flush=True)
