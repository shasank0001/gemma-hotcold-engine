#!/usr/bin/env python3
"""Top token log-probs at fixed contexts, saved as JSON. Same contexts for each engine, so runs can be compared.

Usage: logit_check.py --bin DIR --model M --out o.json [--tokens refs.json] -- <server args>
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


argv = sys.argv[1:]
i = argv.index("--")
ap = argparse.ArgumentParser()
for k in ("--bin", "--model", "--out", "--refs"):
    ap.add_argument(k)
ap.add_argument("--port", type=int, default=18081)
a = ap.parse_args(argv[:i])
cmd = [a.bin + "/llama-server", "-m", a.model, "--port", str(a.port), "-np", "1", "-c", "4096", "--no-webui", "--seed", "1"] + argv[i + 1:]
log = open(a.out + ".log", "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    refs = json.load(open(a.refs))["rows"]
    out = []
    for r in refs:
        base = r["id"]
        toks = post(a.port, "/tokenize", {"content": r["text"]})["tokens"]
        for n in (4, 16, 32, 64, 96):
            r2 = post(a.port, "/completion", {"prompt": toks[:n], "n_predict": 1, "n_probs": 10, "temperature": 0, "cache_prompt": False})
            top = r2["completion_probabilities"][0]["top_logprobs"]
            out.append({"id": f"{base}@{n}", "top": {str(t["id"]): t["logprob"] for t in top}})
    json.dump(out, open(a.out, "w"))
finally:
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()
