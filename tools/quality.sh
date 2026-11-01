#!/bin/bash
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/quality; mkdir -p $O
q() { name=$1; bin=$2; shift 2; timeout 900 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/quality.jsonl --split quality --n 200 --no-think --out $O/q-$name.json -- -ngl 99 -t 6 "$@" | grep DECODE | sed "s/^/q $name /"; }
l() { name=$1; bin=$2; rep=$3; shift 3; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/long.jsonl --split long --n 32 --out $O/l-$name-r$rep.json -- -ngl 99 -t 6 "$@" | grep DECODE | sed "s/^/long $name r$rep /"; }
q cmoe $S -cmoe
q cache $S -cmoe --moe-cache-mib 5000 -lm none
LLAMA_MOE_HOT=$H q hybrid $B -cmoe --moe-cache-mib 5000 -lm none
for rep in 1 2 3; do
  l cache $S $rep -cmoe --moe-cache-mib 5000 -lm none
  LLAMA_MOE_HOT=$H l hybrid $B $rep -cmoe --moe-cache-mib 5000 -lm none
  l ncmoe20 $S $rep -ncmoe 20 -lm none
done
echo QUALITY DONE
