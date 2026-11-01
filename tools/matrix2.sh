#!/bin/bash
# Adaptive engine vs best stock. Alternating runs, two workloads, cold start each server.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/matrix2; mkdir -p $O
held() { name=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --split heldout --n 128 --out $O/held-$name-r$rep.json -- -ngl 99 -t 6 "$@" | grep DECODE | sed "s/^/held $name r$rep /"; }
math() { name=$1; bin=$2; shift 2; timeout 250 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/quality.jsonl --split qtest --n 200 --no-think --out $O/math-$name-r$rep.json -- -ngl 99 -t 6 "$@" | grep DECODE | sed "s/^/math $name r$rep /"; }
export LLAMA_MOE_LFU_X=1
for rep in 1 2 3; do
  for mb in 2000 3000 4000 5000; do
    held cache-$mb $S -cmoe --moe-cache-mib $mb -lm none
    LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 held lfu-$mb $B -cmoe --moe-cache-mib $mb -lm none
    math cache-$mb $S -cmoe --moe-cache-mib $mb -lm none
    LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 math lfu-$mb $B -cmoe --moe-cache-mib $mb -lm none
  done
  for n in 22 24 26; do held ncmoe-$n $S -ncmoe $n -lm none; math ncmoe-$n $S -ncmoe $n -lm none; done
done
echo MATRIX2 DONE
