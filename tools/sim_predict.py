#!/usr/bin/env python3
"""Can earlier layers of a token predict later layers' cold experts?
Trains co-occurrence on the profile trace, then replays held-out tokens over a fixed hot set."""
import collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sim_cache as s


def tokens(recs):
    cur = {}
    for il, ids in recs:
        if len(ids) > 8:
            cur = {}
            continue
        if il in cur:
            yield cur
            cur = {}
        cur[il] = ids


prof = s.load(["results/gemma/profile.bin"])
test = s.load(["results/gemma/heldout.bin"])
math = s.load(["results/gemma/broad/qtrain.bin"])
ranked = s.rank(prof)
co = collections.defaultdict(collections.Counter)
prev = collections.defaultdict(collections.Counter)
for t in tokens(prof):
    ls = sorted(t)
    for a, b in zip(ls, ls[1:]):
        for x in t[a]:
            for y in t[b]:
                co[(a, x)][y] += 1


def run(name, recs, slots, m):
    hot = set(ranked[:slots])
    n = cold = hit = pre = 0
    last = {}
    for t in tokens(recs):
        ls = sorted(t)
        for a, b in zip(ls, ls[1:]):
            score = collections.Counter()
            for x in t[a]:
                score.update(co.get((a, x), {}))
            # same layer on the previous token is also a signal
            for e in last.get(b, []):
                score[e] += 3
            cand = [e for e, _ in score.most_common(40) if b * 1000 + e not in hot][:m]
            pre += len(cand)
            cs = set(cand)
            for e in t[b]:
                n += 1
                if b * 1000 + e not in hot:
                    cold += 1
                    hit += e in cs
        last = t
    print(f"{name:7s} slots {slots} prefetch {m}/layer: cold {cold/n:.3f}, predicted {hit/max(cold,1):.3f} of cold, precision {hit/max(pre,1):.3f}, cold after {(cold-hit)/n:.3f}")


for name, r in (("held", test), ("math", math)):
    for m in (1, 2, 4):
        run(name, r, 900, m)
