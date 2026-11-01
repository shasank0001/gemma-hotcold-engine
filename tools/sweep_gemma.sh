#!/bin/bash
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf
run() { name=$1; shift; timeout 500 python3 tools/run_prompts.py --bin llama.cpp/build-stock/bin --model "$G" --split heldout --n 128 --out results/gemma/stock-$name.json -- "$@" | tail -1; }
for cfg in "cmoe|-ngl 99 -cmoe -t 6" "ncmoe16|-ngl 99 -ncmoe 16 -t 6" "ncmoe20|-ngl 99 -ncmoe 20 -t 6" "ncmoe24|-ngl 99 -ncmoe 24 -t 6" "cache5000-none|-ngl 99 -cmoe --moe-cache-mib 5000 -lm none" "cache6000-none|-ngl 99 -cmoe --moe-cache-mib 6000 -lm none"; do
    echo "== ${cfg%%|*}"; run ${cfg%%|*} ${cfg#*|}
done
E_="llama.cpp-engine/build-engine/bin"
timeout 1200 python3 tools/run_prompts.py --bin $E_ --model "$G" --split profile --n 256 --out results/gemma/profile-trace.json --trace results/gemma/profile.bin -- -ngl 99 -cmoe --moe-cache-mib 5000 | tail -5
LLAMA_MOE_TIMING=1 timeout 600 python3 tools/run_prompts.py --bin $E_ --model "$G" --split heldout --n 256 --out results/gemma/heldout-trace.json --trace results/gemma/heldout.bin -- -ngl 99 -cmoe --moe-cache-mib 5000 | tail -6
echo GEMMA DONE
