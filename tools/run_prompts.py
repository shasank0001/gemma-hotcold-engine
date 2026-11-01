#!/usr/bin/env python3
"""Start llama-server, run a prompt split, write per-prompt timings and cache stats as JSON.

Usage: run_prompts.py --bin llama.cpp/build-stock/bin --model M --split heldout --n 256 --out r.json -- <server args>
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "prompts/prompts.jsonl"))
    a = ap.parse_args(argv)

    env = dict(os.environ)
    if a.trace:
        env["LLAMA_MOE_TRACE"] = a.trace
    log_path = a.out + ".log"
    cmd = [os.path.join(a.bin, "llama-server"), "-m", a.model, "--port", str(a.port), "-np", "1", "-c", "4096", "--no-webui", "--cache-prompt", "--temp", "0", "--seed", "1"] + extra
    with open(log_path, "w") as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    prompts = [json.loads(l) for l in open(a.prompts)]
    prompts = [p for p in prompts if a.split == "all" or p["split"] == a.split or (a.split == "quality" and p["split"].startswith("q"))]
    rows, peak = [], 0
    try:
        for _ in range(300):
            if srv.poll() is not None:
                raise SystemExit(f"server exited {srv.returncode}, see {log_path}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server not ready")
        for p in prompts:
            r = post(f"http://127.0.0.1:{a.port}/v1/chat/completions", {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.n, "temperature": 0, "seed": 1, "cache_prompt": False, **({"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {})})
            t = r["timings"]
            mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            peak = max(peak, int(mem))
            rows.append({"id": p["id"], "decode_tps": t["predicted_per_second"], "prefill_tps": t["prompt_per_second"], "n": t["predicted_n"], "n_prompt": t["prompt_n"], "text": (r["choices"][0]["message"].get("reasoning_content") or "") + (r["choices"][0]["message"].get("content") or "")})
            print(p["id"], round(t["predicted_per_second"], 2), round(t["prompt_per_second"], 1), flush=True)
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    tot_n = sum(r["n"] for r in rows)
    tot_t = sum(r["n"] / r["decode_tps"] for r in rows)
    stats = [l.strip() for l in open(log_path, errors="replace") if "llama_moe_cache:" in l]
    out = {"cmd": cmd, "decode_tps": tot_n / tot_t if tot_t else 0, "prefill_tps_mean": sum(r["prefill_tps"] for r in rows) / len(rows), "peak_vram_mib": peak, "tokens": tot_n, "cache_stats": stats, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"DECODE {out['decode_tps']:.2f} tok/s PREFILL {out['prefill_tps_mean']:.1f} VRAM {peak}", *stats, sep="\n")


if __name__ == "__main__":
    main()
