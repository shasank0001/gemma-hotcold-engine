#!/usr/bin/env python3
"""Live demo: run the same prompts on stock and on this engine, one after the other, and print tok/s and VRAM.

Usage: demo.py [vram_mib]   default 3000 (about 5.6 GB in use)
Only one server runs at a time, because both need the GPU.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mb = sys.argv[1] if len(sys.argv) > 1 else "3000"
model = os.path.join(ROOT, "models/gemma-4-26B-A4B-it-Q4_0.gguf")
mtp = os.path.join(ROOT, "models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf")
base = ["-ngl", "99", "-t", "6", "-lm", "none", "--spec-type", "draft-mtp", "-md", mtp, "--spec-draft-n-max", "4", "-ngld", "99"]
runs = [
    ("stock layer split", "llama.cpp/build-stock/bin", {}, ["-ncmoe", "24"]),
    ("stock cache", "llama.cpp/build-stock/bin", {}, ["-cmoe", "--moe-cache-mib", mb]),
    ("this engine", "llama.cpp-engine/build-engine/bin", {"LLAMA_MOE_HOT": os.path.join(ROOT, "results/gemma/hot.txt"), "LLAMA_MOE_LFU": "1", "LLAMA_MOE_LFU_WEIGHT": "1", "LLAMA_MOE_OVERLAP": "1"}, ["-cmoe", "--moe-cache-mib", mb]),
]
res = []
for name, b, env, args in runs:
    out = os.path.join(ROOT, "results/gemma/demo-" + name.replace(" ", "_") + ".json")
    print(f"== {name}", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools/run_prompts.py"), "--bin", os.path.join(ROOT, b), "--model", model, "--split", "heldout", "--n", "96", "--out", out, "--"] + base + args, env={**os.environ, **env}, check=True)
    r = json.load(open(out))
    res.append((name, r["decode_tps"], r["peak_vram_mib"]))
print()
for name, tps, vram in res:
    print(f"{name:20s} {tps:6.1f} tok/s {vram:6d} MiB  " + "#" * int(tps))
