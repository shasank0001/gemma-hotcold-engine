#!/usr/bin/env python3
"""Summarise results/gemma/matrix into mean decode, prefill and peak VRAM per config."""
import collections, glob, json, os, re, statistics as st, sys
d = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*-r*.json"):
    d[re.sub(r"-r\d+\.json$", "", os.path.basename(f))].append(json.load(open(f)))
rows = []
for k, v in d.items():
    dec = [r["decode_tps"] for r in v]
    rows.append((max(r["peak_vram_mib"] for r in v), k, st.mean(dec), st.pstdev(dec), st.mean(r["prefill_tps_mean"] for r in v), len(v)))
for vram, k, m, s, p, n in sorted(rows):
    print(f"{vram:6d} MiB  {k:14s} decode {m:6.2f} +-{s:4.2f}  prefill {p:6.1f}  n={n}")
