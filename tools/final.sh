#!/bin/bash
# Frozen build. MTP on every engine, alternating runs, then quality and a short soak.
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; M=models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf; S=llama.cpp/build-stock/bin; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt
O=results/gemma/final; mkdir -p $O/m
SP="--spec-type draft-mtp -md $M --spec-draft-n-max 4 -ngld 99"
held() { name=$1; bin=$2; shift 2; timeout 200 python3 tools/run_prompts.py --bin $bin --model $G --split heldout --n 128 --out $O/m/held-$name-r$rep.json -- -ngl 99 -t 6 -lm none $SP "$@" | grep DECODE | sed "s/^/held $name r$rep /"; }
math() { name=$1; bin=$2; shift 2; timeout 250 python3 tools/run_prompts.py --bin $bin --model $G --prompts prompts/quality.jsonl --split qtest --n 200 --no-think --out $O/m/math-$name-r$rep.json -- -ngl 99 -t 6 -lm none $SP "$@" | grep DECODE | sed "s/^/math $name r$rep /"; }
E_="LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1 LLAMA_MOE_LFU_WEIGHT=1 LLAMA_MOE_OVERLAP=1"
for rep in 1 2 3; do
  for mb in 2000 3000 4000 4600; do
    env $E_ bash -c "$(declare -f held math); O=$O G=$G SP='$SP' rep=$rep; held lfu-$mb $B -cmoe --moe-cache-mib $mb; math lfu-$mb $B -cmoe --moe-cache-mib $mb"
  done
  held cache-4600 $S -cmoe --moe-cache-mib 4600; math cache-4600 $S -cmoe --moe-cache-mib 4600
  for n in 20 22 24 26; do held ncmoe-$n $S -ncmoe $n; math ncmoe-$n $S -ncmoe $n; done
  [ "$rep" = 1 ] && { env $E_ timeout 500 python3 tools/run_prompts.py --bin $B --model $G --prompts prompts/quality.jsonl --split quality --n 200 --no-think --out $O/q.json -- -ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none $SP | grep DECODE | sed 's/^/quality /'; }
done
env $E_ timeout 900 python3 tools/run_prompts.py --bin $B --model $G --split all --n 96 --out $O/soak.json -- -ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none $SP | grep -E 'DECODE' | sed 's/^/soak /'
echo FINAL DONE
