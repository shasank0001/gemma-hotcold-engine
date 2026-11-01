#!/usr/bin/env python3
"""Stream the same prompts from each engine and record token times for the side-by-side replay."""
import json, os, signal, subprocess, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = ROOT + "/models/gemma-4-26B-A4B-it-Q4_0.gguf"; M = ROOT + "/models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf"
SP = ["--spec-type", "draft-mtp", "-md", M, "--spec-draft-n-max", "4", "-ngld", "99"]
BASE = ["-m", G, "--port", "18090", "-np", "1", "-c", "4096", "--no-webui", "--temp", "0", "--seed", "1", "-ngl", "99", "-t", "6", "-lm", "none"] + SP
PROMPTS = ["Write a Python function that merges overlapping intervals. Keep it short.", "What is 47 * 86? Show the steps briefly."]
RUNS = [("Stock llama.cpp", "llama.cpp/build-stock/bin", {}, ["-ncmoe", "24"]),
        ("Our engine", "llama.cpp-engine/build-engine/bin", {"LLAMA_MOE_HOT": ROOT + "/results/gemma/hot.txt", "LLAMA_MOE_LFU": "1", "LLAMA_MOE_LFU_WEIGHT": "1", "LLAMA_MOE_OVERLAP": "1"}, ["-cmoe", "--moe-cache-mib", "3000"])]
out = []
for name, b, env, args in RUNS:
    log = open(ROOT + "/logs/record.log", "w")
    srv = subprocess.Popen([ROOT + "/" + b + "/llama-server"] + BASE + args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:18090/health", timeout=2); break
        except Exception:
            time.sleep(1)
    runs = []
    for p in PROMPTS:
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 160, "temperature": 0, "seed": 1, "stream": True, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request("http://127.0.0.1:18090/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time(); ev = []
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line:
                    continue
                c = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if c:
                    ev.append([round(time.time() - t0, 3), c])
        runs.append({"prompt": p, "events": ev})
        print(name, len(ev), "chunks", ev[-1][0] if ev else 0, flush=True)
    srv.send_signal(signal.SIGINT); srv.wait(30)
    out.append({"name": name, "setting": " ".join(args), "runs": runs})
json.dump(out, open(ROOT + "/demo/recording.json", "w"))
print("RECORD DONE")
