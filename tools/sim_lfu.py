#!/usr/bin/env python3
"""Frequency-gated hot set: a cold expert replaces the least used hot expert only if it is used more. CPU serves misses.
Replays a workload switch and reports miss rate and uploads per token."""
import heapq, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sim_cache as s


def run(seq, slots, ranked, margin, halve_every, budget):
    hot = set(ranked[:slots])
    freq = {k: 0.0 for k in hot}
    uses = miss = up = 0
    credit = 0.0
    n_rec = 0
    out = []
    for keys, tag in seq:
        n_rec += 1
        credit = min(credit + budget / 30.0, budget * 4)
        seg_m = 0
        for k in keys:
            uses += 1
            freq[k] = freq.get(k, 0) + 1
            if k not in hot:
                miss += 1
                seg_m += 1
        for k in keys:
            if k in hot or credit < 1:
                continue
            v = min(hot, key=lambda x: freq.get(x, 0)) 
            if freq[k] > freq.get(v, 0) + margin:
                hot.remove(v)
                hot.add(k)
                up += 1
                credit -= 1
        if n_rec % halve_every == 0:
            for k in freq:
                freq[k] *= 0.5
        out.append((tag, len(keys), seg_m))
    return miss, up, uses, out


prof = s.load(["results/gemma/profile.bin"])
ranked = s.rank(prof)
heldout = s.load(["results/gemma/heldout.bin"])
math = s.load(["results/gemma/broad/qtrain.bin"])
seq = [(k, "held") for k in s.keyed(heldout)[:30000]] + [(k, "math") for k in s.keyed(math)[:30000]] + [(k, "held2") for k in s.keyed(heldout)[30000:60000]]
slots = 1542
tok = len(seq) / 30
base = {}
for t in ("held", "math", "held2"):
    base[t] = [0, 0]
hot = set(ranked[:slots])
for k, t in seq:
    base[t][0] += len(k)
    base[t][1] += sum(x not in hot for x in k)
print("static", {t: round(v[1] / v[0], 3) for t, v in base.items()})
for margin, h, b in ((2, 3000, 8), (2, 3000, 16), (4, 3000, 8), (1, 1500, 8), (2, 6000, 4)):
    m, u, n, out = run(seq, slots, ranked, margin, h, b)
    a = {t: [0, 0] for t in base}
    for t, l, sm in out:
        a[t][0] += l
        a[t][1] += sm
    print(f"margin {margin} halve {h} budget {b}/tok:", {t: round(v[1] / v[0], 3) for t, v in a.items()}, f"uploads/tok {u/tok:.1f}")
