#!/usr/bin/env python3
"""Compare generated text and speed of result JSONs against a reference."""
import json, os, sys
a = json.load(open(sys.argv[1]))
print(os.path.basename(sys.argv[1]), f"decode {a['decode_tps']:.2f} prefill {a['prefill_tps_mean']:.1f} vram {a['peak_vram_mib']}")
for p in sys.argv[2:]:
    b = json.load(open(p))
    same = sum(x["text"] == y["text"] for x, y in zip(a["rows"], b["rows"]))
    first = next(((x["id"], next((i for i, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d), -1), len(x["text"])) for x, y in zip(a["rows"], b["rows"]) if x["text"] != y["text"]), None)
    print(os.path.basename(p), f"decode {b['decode_tps']:.2f} prefill {b['prefill_tps_mean']:.1f} vram {b['peak_vram_mib']} identical {same}/{len(a['rows'])} first_diff {first}")
    print(" ", *[s[17:] for s in b["cache_stats"] if "hybrid: h" in s])
