#!/usr/bin/env python3
"""Rank experts by use in traces. Output lines: il expert (most used first)."""
import collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sim_cache as s

c = collections.Counter()
for il, ids in s.load(sys.argv[2:]):
    c.update((il, e) for e in ids)
with open(sys.argv[1], "w") as f:
    for (il, e), _ in c.most_common():
        f.write(f"{il} {e}\n")
print(len(c), "experts ranked")
