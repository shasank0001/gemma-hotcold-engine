#!/bin/bash
# Tune swap settings on profile prompts only.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; B=llama.cpp-engine/build-engine/bin; O=results/gemma/tune; mkdir -p $O
export LLAMA_MOE_HOT=$PWD/results/gemma/hot.txt LLAMA_MOE_LFU=1
for c in "2 8 3000" "1 8 1500" "1 16 1500" "1 4 3000" "4 8 6000" "0.5 16 1000"; do set -- $c
  for rep in 1 2; do LLAMA_MOE_LFU_MARGIN=$1 LLAMA_MOE_LFU_BUDGET=$2 LLAMA_MOE_LFU_HALVE=$3 timeout 300 python3 tools/run_prompts.py --bin $B --model $G --split profile --n 96 --out $O/m$1-b$2-h$3-r$rep.json -- -ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none | grep DECODE | sed "s/^/m$1 b$2 h$3 r$rep /"; done
done
python3 tools/demo.py 3000 | tail -4
echo TUNE DONE
