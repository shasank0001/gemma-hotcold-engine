#!/usr/bin/env python3
"""Replay LLAMA_MOE_TRACE files through expert cache policies.

Trace records are int32: il, n, ids[n]. One record per layer per ubatch.
Usage: sim_cache.py --profile a.bin [b.bin] --test c.bin --slots 800 1600
"""
import argparse, collections, struct, sys
from array import array


def load(paths):
    recs = []
    for p in paths:
        a = array("i")
        with open(p, "rb") as f:
            a.frombytes(f.read())
        i = 0
        while i < len(a):
            il, n = a[i], a[i + 1]
            recs.append((il, sorted(set(a[i + 2:i + 2 + n]))))
            i += 2 + n
    return recs


class SLRU:
    """Matches moe_cache_slru: hits go to protected, misses to probation, protected holds at most half."""

    def __init__(self, n, pinned=()):
        self.pinned = set(pinned)
        self.n = n - len(self.pinned)
        self.prob = collections.OrderedDict()
        self.prot = collections.OrderedDict()

    def access(self, keys):
        hits = [k for k in keys if k in self.pinned or k in self.prob or k in self.prot]
        for k in hits:
            if k in self.pinned:
                continue
            self.prob.pop(k, None)
            self.prot[k] = 1
            self.prot.move_to_end(k)
            if len(self.prot) > self.n // 2:
                old, _ = self.prot.popitem(last=False)
                self.prob[old] = 1
        miss = [k for k in keys if k not in hits]
        held = set(keys)
        for k in miss:
            self.insert(k, held)
        return len(miss)

    def insert(self, k, held=()):
        if k in self.pinned or k in self.prob or k in self.prot:
            return False
        if len(self.prob) + len(self.prot) >= self.n:
            for seg in (self.prob, self.prot):
                v = next((x for x in seg if x not in held), None)
                if v is not None:
                    del seg[v]
                    break
        self.prob[k] = 1
        return True


def keyed(recs):
    return [[il * 1000 + e for e in ids] for il, ids in recs]


def rank(recs):
    c = collections.Counter()
    for keys in keyed(recs):
        c.update(keys)
    return [k for k, _ in c.most_common()]


def run_slru(test, slots, pinned=()):
    c = SLRU(slots, pinned)
    return sum(c.access(k) for k in keyed(test))


def run_static(test, slots, hot):
    """Hot set never changes, every other use is a miss."""
    s = set(hot[:slots])
    return sum(1 for keys in keyed(test) for k in keys if k not in s)


def run_belady(test, slots):
    seq = keyed(test)
    nxt = collections.defaultdict(collections.deque)
    for t, keys in enumerate(seq):
        for k in keys:
            nxt[k].append(t)
    cache, miss = set(), 0
    for t, keys in enumerate(seq):
        for k in keys:
            nxt[k].popleft()
        held = set(keys)
        for k in keys:
            if k in cache:
                continue
            miss += 1
            if len(cache) >= slots:
                v = max((x for x in cache if x not in held), key=lambda x: nxt[x][0] if nxt[x] else 1 << 60)
                cache.remove(v)
            cache.add(k)
    return miss


def run_prefetch(test, slots, pinned=()):
    """SLRU plus upload of the experts this layer used on the previous token when they left the cache.
    Returns (demand misses, prefetch uploads)."""
    c = SLRU(slots, pinned)
    last, miss, pre = {}, 0, 0
    for (il, _), keys in zip(test, keyed(test)):
        for k in last.get(il, []):
            pre += int(c.insert(k))
        miss += c.access(keys)
        last[il] = keys
    return miss, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--slots", nargs="+", type=int, required=True)
    a = ap.parse_args()
    prof, test = load(a.profile), load(a.test)
    uses = sum(len(k) for _, k in test)
    ranked = rank(prof)
    print(f"# test uses={uses} distinct_profile={len(ranked)}")
    print("slots  policy                 misses  miss%  vs_slru")
    for n in a.slots:
        base = run_slru(test, n)
        rows = [("slru", base), ("static", run_static(test, n, ranked)), ("pin50+slru", run_slru(test, n, ranked[: n // 2])), ("pin75+slru", run_slru(test, n, ranked[: n * 3 // 4])), ("belady", run_belady(test, n))]
        m, p = run_prefetch(test, n)
        rows.append(("prefetch(demand)", m))
        rows.append(("prefetch(total)", m + p))
        for name, v in rows:
            print(f"{n:5d}  {name:18s} {v:9d}  {100*v/uses:5.1f}  {100*(v/base-1) if base else 0:+6.1f}%")


if __name__ == "__main__":
    sys.exit(main())
