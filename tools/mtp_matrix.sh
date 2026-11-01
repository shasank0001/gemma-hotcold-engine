#!/bin/bash
# MTP on every engine. Draft length tuned on profile prompts only, then all compared on held-out and math.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; M=models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/mtp; mkdir -p $O/tune $O/m
for n in 1 2 3 4 6; do
  LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 timeout 300 python3 tools/run_prompts.py --bin $B --model $G --split profile --n 96 --out $O/tune/n$n.json -- -ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none --spec-type draft-mtp -md $M --spec-draft-n-max $n -ngld 99 | grep DECODE | sed "s/^/tune n$n /"
done
BEST=$(python3 -c "
import json
print(max((json.load(open(f'$O/tune/n{n}.json'))['decode_tps'],n) for n in (1,2,3,4,6))[1])")
echo "best draft $BEST"
SP="--spec-type draft-mtp -md $M --spec-draft-n-max $BEST -ngld 99"
held() { name=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --split heldout --n 128 --out $O/m/held-$name-r$rep.json -- -ngl 99 -t 6 -lm none $SP "$@" | grep DECODE | sed "s/^/held $name r$rep /"; }
math() { name=$1; bin=$2; shift 2; timeout 250 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/quality.jsonl --split qtest --n 200 --no-think --out $O/m/math-$name-r$rep.json -- -ngl 99 -t 6 -lm none $SP "$@" | grep DECODE | sed "s/^/math $name r$rep /"; }
for rep in 1 2 3; do
  for mb in 2000 3000 4000 4600; do
    held cache-$mb $S -cmoe --moe-cache-mib $mb
    LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 held lfu-$mb $B -cmoe --moe-cache-mib $mb
    math cache-$mb $S -cmoe --moe-cache-mib $mb
    LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 math lfu-$mb $B -cmoe --moe-cache-mib $mb
  done
  for n in 22 24 26; do held ncmoe-$n $S -ncmoe $n; math ncmoe-$n $S -ncmoe $n; done
done
echo MTP DONE
