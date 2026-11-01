#!/bin/bash
cd "$(dirname "$0")/.."
G=models/gemma-4-26B-A4B-it-Q4_0.gguf; M=models/mtp-gemma-4-26B-A4B-it-Q4_0.gguf; B=llama.cpp-engine/build-engine/bin; H=$PWD/results/gemma/hot.txt; O=results/gemma/verify-mtp; mkdir -p $O
SCR=/tmp/claude-1000/-home-shasank-shasank-Hackathons-hacktober/68d884cc-4918-464a-b610-f3f8d9d465f3/scratchpad
export LLAMA_MOE_HOT=$H LLAMA_MOE_LFU=1
A="-ngl 99 -t 6 -cmoe --moe-cache-mib 3000 -lm none --spec-type draft-mtp -md $M --spec-draft-n-max 4 -ngld 99"
timeout 900 python3 tools/run_prompts.py --bin $B --model $G --prompts prompts/quality.jsonl --split quality --n 200 --no-think --out $O/q.json -- $A | grep DECODE | sed 's/^/quality /'
python3 - <<'P'
import json
SCR='/tmp/claude-1000/-home-shasank-shasank-Hackathons-hacktober/68d884cc-4918-464a-b610-f3f8d9d465f3/scratchpad'
L=[json.loads(l) for l in open('prompts/prompts.jsonl')]+[json.loads(l) for l in open('prompts/quality.jsonl')]
with open(SCR+'/soak2.jsonl','w') as f:
    for r in range(3):
        for o in L: f.write(json.dumps(dict(o,id=f"{o['id']}#{r}",split='soak'))+"\n")
P
timeout 3000 python3 tools/run_prompts.py --bin $B --model $G --prompts $SCR/soak2.jsonl --split soak --n 96 --out $O/soak.json -- $A | grep -E 'DECODE|hybrid:' | sed 's/^/soak /'
echo "soak exit ${PIPESTATUS[0]}"
python3 - <<'P'
import json,re
Q={json.loads(l)['id']:json.loads(l)['answer'] for l in open('prompts/quality.jsonl')}
d=json.load(open('results/gemma/verify-mtp/q.json'));ok=miss=bad=0
for x in d['rows']:
    m=re.findall(r'ANSWER:\s*\**\s*(-?[\d,]+)',x['text'])
    if not m: miss+=1
    elif int(m[-1].replace(',',''))==Q[x['id']]: ok+=1
    else: bad+=1
print('mtp quality correct',ok,'wrong',bad,'no answer',miss)
s=json.load(open('results/gemma/verify-mtp/soak.json'));r=s['rows']
print('soak prompts',len(r),'tokens',s['tokens'],'decode',round(s['decode_tps'],1),'empty',sum(not x['text'] for x in r),'garbled',sum('<unused' in x['text'] for x in r))
n=len(r)//3
for k in range(3):
    c=r[k*n:(k+1)*n]; print(' pass',k,round(sum(x['n'] for x in c)/sum(x['n']/x['decode_tps'] for x in c),1))
P
echo VMTP DONE
