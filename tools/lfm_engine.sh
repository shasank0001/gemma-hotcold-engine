#!/bin/bash
cd "$(dirname "$0")/.."
L=/home/shasank/shasank/Deep_learing/projects/custem-model/models/24B-Q4KM/LFM2.5-24B-A2B-Q4_K_M.gguf
S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/lfm/hot.txt; O=results/lfm/engine; mkdir -p $O
r() { n=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model "$L" --split heldout --n 128 --out $O/$n-r$rep.json -- -ngl 99 -t 6 -lm none "$@" | grep DECODE | sed "s/^/$n r$rep /"; }
for rep in 1 2 3; do
  for mb in 3000 4000 5000; do
    r cache-$mb $S -cmoe --moe-cache-mib $mb
    LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 r lfu-$mb $B -cmoe --moe-cache-mib $mb
  done
  for n in 24 26 28 30; do r ncmoe-$n $S -ncmoe $n; done
done
echo LFM DONE
