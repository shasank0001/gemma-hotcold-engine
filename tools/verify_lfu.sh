#!/bin/bash
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt; O=results/gemma/verify; mkdir -p $O
export LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1
A="-ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none"
timeout 900 python3 tools/run_prompts.py --bin $B --model $G --prompts prompts/quality.jsonl --split quality --n 200 --no-think --out $O/q-lfu.json -- $A | grep DECODE | sed 's/^/quality /'
for rep in 1 2 3; do timeout 200 python3 tools/run_prompts.py --bin $B --model $G --prompts prompts/long.jsonl --split long --n 32 --out $O/l-lfu-r$rep.json -- $A | grep DECODE | sed "s/^/long r$rep /"; done
# soak: every prompt, repeated in one server, 256 tokens
python3 - <<'P'
import json
L=[json.loads(l) for l in open('prompts/prompts.jsonl')]+[json.loads(l) for l in open('prompts/quality.jsonl')]
with open('/tmp/claude-1000/-home-shasank-shasank-Hackathons-hacktober/68d884cc-4918-464a-b610-f3f8d9d465f3/scratchpad/soak.jsonl','w') as f:
    for r in range(3):
        for o in L: f.write(json.dumps(dict(o,id=f"{o['id']}#{r}",split='soak'))+"\n")
P
timeout 3000 python3 tools/run_prompts.py --bin $B --model $G --prompts /tmp/claude-1000/-home-shasank-shasank-Hackathons-hacktober/68d884cc-4918-464a-b610-f3f8d9d465f3/scratchpad/soak.jsonl --split soak --n 96 --out $O/soak.json -- $A | grep -E 'DECODE|hybrid:|swaps' | sed 's/^/soak /'
echo "soak exit ${PIPESTATUS[0]}"
echo VERIFY DONE
