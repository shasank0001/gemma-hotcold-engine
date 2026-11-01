#!/bin/bash
# Try hard to beat the engine with stock: layer split plus cache, and thread counts.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; O=results/gemma/stockbest; mkdir -p $O
r() { n=$1; shift
  timeout 200 python3 tools/run_prompts.py --bin $S --model $G --split heldout --n 128 --out $O/held-$n-r1.json -- -ngl 99 "$@" | grep DECODE | sed "s/^/held $n /"
  timeout 250 python3 tools/run_prompts.py --bin $S --model $G --prompts prompts/quality.jsonl --split qtest --n 200 --no-think --out $O/math-$n-r1.json -- -ngl 99 "$@" | grep DECODE | sed "s/^/math $n /"; }
for c in "24 500" "26 1300" "28 2100" "22 1600" "20 800" "24 2500" "26 3300" "22 900"; do set -- $c
  r n$1-c$2 -ncmoe $1 --moe-cache-mib $2 -lm none -t 6
done
r t10-n24 -ncmoe 24 -lm none -t 10
r t16-n24 -ncmoe 24 -lm none -t 16
r t8-c5000 -cmoe --moe-cache-mib 5000 -lm none -t 8
r t12-n22 -ncmoe 22 -lm none -t 12
echo STOCKBEST DONE
