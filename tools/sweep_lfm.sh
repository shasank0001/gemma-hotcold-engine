#!/bin/bash
cd "$(dirname "$0")/.."
L=/home/shasank/shasank/Deep_learing/projects/custem-model/models/24B-Q4KM/LFM2.5-24B-A2B-Q4_K_M.gguf
run() { name=$1; shift; timeout 400 python3 tools/run_prompts.py --bin llama.cpp/build-stock/bin --model "$L" --split heldout --n 128 --out results/lfm/stock-$name.json -- "$@" | tail -1; }
for cfg in "cmoe-nocache|-ngl 99 -cmoe -t 6" "cmoe-nocache-t10|-ngl 99 -cmoe -t 10" "ncmoe24|-ngl 99 -ncmoe 24 -t 6" "ncmoe30|-ngl 99 -ncmoe 30 -t 6" "cache6500-none|-ngl 99 -cmoe --moe-cache-mib 6500 -lm none" "cache5000-none|-ngl 99 -cmoe --moe-cache-mib 5000 -lm none"; do
    echo "== ${cfg%%|*}"; run ${cfg%%|*} ${cfg#*|}
done
echo SWEEP DONE
