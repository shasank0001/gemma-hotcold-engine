#!/bin/bash
# Alternating stock/engine runs on heldout prompts. Writes results/gemma/matrix/<name>-r<rep>.json
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/matrix; mkdir -p $O
run() { name=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --split heldout --n 128 --out $O/$name-r$rep.json -- -ngl 99 -t 6 "$@" | grep DECODE | sed "s/^/$name r$rep /"; }
for rep in 1 2 3; do
  for mb in 1000 2000 3000 4000 5000; do
    run cache-$mb $S -cmoe --moe-cache-mib $mb -lm none
    LLAMA_MOE_HOT=$H run hybrid-$mb $B -cmoe --moe-cache-mib $mb -lm none
    LLAMA_MOE_HOT=$H LLAMA_MOE_DYNAMIC=1 run dyn-$mb $B -cmoe --moe-cache-mib $mb -lm none
  done
  for n in 20 22 24 26 28 30; do run ncmoe-$n $S -ncmoe $n -lm none; done
done
echo MATRIX DONE
