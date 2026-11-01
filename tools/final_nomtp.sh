#!/bin/bash
# Frozen build without MTP on any engine. Alternating runs, 2 repetitions.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/final-nomtp/m; mkdir -p $O
held() { name=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --split heldout --n 128 --out $O/held-$name-r$rep.json -- -ngl 99 -t 6 -lm none "$@" | grep DECODE | sed "s/^/held $name r$rep /"; }
math() { name=$1; bin=$2; shift 2; timeout 250 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/quality.jsonl --split qtest --n 200 --no-think --out $O/math-$name-r$rep.json -- -ngl 99 -t 6 -lm none "$@" | grep DECODE | sed "s/^/math $name r$rep /"; }
ours() { export LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 LLAMA_MOE_LFU_WEIGHT=1 LLAMA_MOE_OVERLAP=1; "$@"; unset LLAMA_MOE_HOT LLAMA_MOE_LFU LLAMA_MOE_LFU_WEIGHT LLAMA_MOE_OVERLAP; }
for rep in 1 2; do
  for mb in 2000 3000 4000 5000; do
    ours held lfu-$mb $B -cmoe --moe-cache-mib $mb
    ours math lfu-$mb $B -cmoe --moe-cache-mib $mb
  done
  held cache-5000 $S -cmoe --moe-cache-mib 5000; math cache-5000 $S -cmoe --moe-cache-mib 5000
  for n in 20 22 24 26; do held ncmoe-$n $S -ncmoe $n; math ncmoe-$n $S -ncmoe $n; done
done
echo NOMTP DONE
