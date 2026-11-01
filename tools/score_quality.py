#!/usr/bin/env python3
"""Score quality runs and summarise long-prompt runs."""
import glob, json, re, statistics as st
Q = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open("prompts/quality.jsonl")}
for n in ("cmoe", "cache", "hybrid"):
    d = json.load(open(f"results/gemma/quality/q-{n}.json"))
    ok = miss = 0
    for x in d["rows"]:
        m = re.findall(r"ANSWER:\s*\**\s*(-?[\d,]+)", x["text"])
        if not m:
            miss += 1
        elif int(m[-1].replace(",", "")) == Q[x["id"]]:
            ok += 1
    print(n, "correct", ok, "/", len(d["rows"]), "no answer", miss, "decode", round(d["decode_tps"], 1))
for n in ("cache", "hybrid", "ncmoe20"):
    v = [json.load(open(f)) for f in glob.glob(f"results/gemma/quality/l-{n}-r*.json")]
    if v:
        print(n, "long prefill", round(st.mean(r["prefill_tps_mean"] for r in v), 1), "prompt tokens", v[0]["rows"][0]["n_prompt"], "decode", round(st.mean(r["decode_tps"] for r in v), 1))
