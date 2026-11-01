# Overnight status

Started 2026-10-31 23:00 IST. Follows PLAN.md. Newest entries at the bottom.

## Resume rules
- Check the 5-hour limit before each long step. Pause near 90% and resume after the reset.
- Stock control: llama.cpp branch `stock`, commit 0ec4027, built in build-stock. Never modify it.
- Our work: branch `engine` in a separate git worktree at ../llama.cpp-engine, built there in build-engine. Never edit the stock tree.
- Model: models/gemma-4-26B-A4B-it-Q4_0.gguf (ggml-org).
- Never benchmark while compiling or downloading.

## Log
- 23:01 Clone from GitHub timed out; cloned local checkout and fetched upstream to 0ec4027. Cache code changes since 3f03b17 are a memory refactor only.
- 23:05 Model download and stock build started.
- Stock build OK (0ec4027). `llama-gguf` target does not exist; harmless.
- Engine worktree created. Trace (LLAMA_MOE_TRACE) and upload timing (LLAMA_MOE_TIMING) hooks added in copy(). Off unless the env var is set.
- Download about 2 MB/s through Xet, roughly 2 hours. du on models/ understates progress; Xet chunks land in ~/.cache/huggingface/xet (uses extra disk: check free space).
- 23:20 Session restart killed the download and engine build; both restarted detached (setsid). Check with pgrep; no completion notice comes from them directly.
- "/home/shasank/shasank/Deep_learing/gemma /models" (trailing space in folder name): NVFP4 file is unusable, 23 of 264 chunks only (1.5 GB of 17.7 GB, sparse). Header confirms Gemma 4 26B: 128 experts, top-8, 30 layers, embd 2816.
- Interim model: /home/shasank/shasank/Deep_learing/projects/custem-model/models/24B-Q4KM/LFM2.5-24B-A2B-Q4_K_M.gguf (13.5 GB, lfm2moe, 64 experts, top-4, 40 layers). Use it now to prove the pipeline. Results on it are labelled LFM, never Gemma. Do not modify that file.
- 00:05 LFM traces done (4608 profile, 1536 heldout tokens; -cmoe, cache 5000 MiB, about 925 slots, expert about 5.4 MiB). Decode 25.7 tok/s, hit rate 74.65%.
- sim_cache.py validated: predicts 26.6% misses, real about 26.9%.
- LFM result: pinning LOSES. static +64%, pin50 +11%, pin75 +22% misses vs SLRU. Belady -56%. Next-token prefetch useless. Re-test on Gemma, but do not expect pinning to win.
- BIG FINDING: LLAMA_MOE_TIMING shows uploads take 53.6 s of about 60 s decode (about 89%). `-lm none` (pinned host buffer) gives 29.3 tok/s vs 25.7.
- tools/bw: CUDA host to device caps at 6.5 GB/s pageable, 8.5 GB/s pinned, at every copy size. nvidia-smi reports Gen5 x8 but that is the measured limit. Better copy path alone cannot help further.
- New direction: decode speed is about 1 / uploaded bytes. One expert upload costs about 0.64 ms. Ideas: admission control (do not upload one-time experts, compute them on CPU), learned predictor toward Belady, CPU fallback for misses. Stock sweep running to measure CPU expert cost.
- 00:10 Stock LFM sweep (heldout, 128 tokens, stock build), decode tok/s / VRAM MiB: -cmoe no cache 35.1 / 1115; -ncmoe 24 53.0 / 6741; -ncmoe 30 44.7 / 4701; cache 6500 -lm none 42.3 / 7649; cache 5000 -lm none 30.3 / 6145. Layer split BEATS upstream's cache on this laptop. Baseline to beat is -ncmoe.
- Time model fits: token ms = 6.2 + 21.9 * miss fraction. CPU costs about 0.18 ms per expert use, upload 0.64 ms. So CPU beats upload about 3.5x on a one-time miss.
- DIRECTION: original per-expert idea is back. Hot experts resident on GPU, cold computed on CPU, no uploads in decode. Predicted on LFM at same VRAM: 53 -> 67 tok/s (+28%), before sync cost.
- 00:20 Gemma downloaded (needs tools/fetch_model.sh only if deleted). Gemma stock, decode / prefill / VRAM: cache 5000 -lm none 49.8 / 44 / 7601 (BEST DECODE, hit 90%, expert 3.2 MiB, about 1500 slots); ncmoe20 38.9 / 89 / 6653; ncmoe24 35.3 / 79 / 5021; cmoe 29.8 / 63 / 2571. cache 6000 and ncmoe16 run out of VRAM. Unlike LFM, the cache wins on Gemma decode.
- Gemma sim: pinning loses (+83% to +193% misses), Belady about -55%.
- Hybrid engine built (env LLAMA_MOE_HOT, LLAMA_MOE_DYNAMIC): hot on GPU, cold on CPU, 24 zero slots, per-token colouring because the GPU kernel needs distinct ids in a token, background upload thread. Bugs fixed: duplicate slot crash; -1 ids reading per-expert scales out of bounds (gave NaN garbage and a FALSE 68 tok/s, discarded).
- Valid first result, heldout 128 tok: stock 49.4 / 42.9; static hybrid 50.2 / 91.4, hit 70%; dynamic 48.2 / 68.3, hit 85%. Text identical only 1/6, checking if this is CPU vs GPU rounding.
- 00:45 Divergence is rounding: stock configs differ from each other too (cache vs cmoe 2/6, ncmoe20 vs cmoe 4/6, hybrid 3/6). Still need a logit-level check before any final claim.
- Tried and rejected: thread tuning (6 is the limit), CUDA spin (no gain), moving scales to GPU (no gain). Splits reduced 152 -> 92 (scale after sum, merged cold lookup): 51.3 tok/s, small gain.
- First clear result vs stock cache, decode: 1500 MiB 18.0 -> 35.0; 3000 MiB 30.3 -> 41.8; 5000 MiB 49.4 -> 51.3. Prefill about 2-3x stock cache. Must compare to BEST stock per VRAM, incl -ncmoe. Full matrix running: tools/matrix.sh, results/gemma/matrix/.
- patches/engine.patch holds the engine diff (git commits only when the user asks).
- 01:25 MATRIX (3 reps, heldout, 128 tok, tools/matrix_report.py). Hybrid vs best stock at or below same VRAM, decode: 3.6 GB 35.8 vs 34.8 (+3%); 4.6 GB 40.2 vs 36.6 (+10%); 5.6 GB 45.1 vs 38.5 (+17%); 6.6 GB 48.2 vs 40.9 (+18%); 7.6 GB 53.0 vs 49.6 (+7%). Prefill at 7.6 GB 97.7 vs cache 44.5 (ncmoe is similar at about 98 but lower decode). Dynamic hybrid loses to static everywhere: dropped.
- Logit check (30 contexts, tools/logit_check.py): plan's 1e-3 gate is impossible, stock configs differ from each other by dP(top) about 0.14. Hybrid vs cache 0.15. Same top token 24/30 for all GPU pairs. Hybrid is inside stock's cross-backend spread, not bit-equal; never claim identical.
- 02:15 Quality, 60 math prompts, thinking off, 200 tok, 7.6 GB: cmoe 57, cache 58, hybrid 57 correct; all misses are no parsable answer, no wrong answers. Decode on math: hybrid 43.6, cache 43.2 (tie: hot list from code/chat/reasoning does not cover math).
- Long prompt 1948 tokens, 3 reps: prefill all about 500 (cache 489, hybrid 507, ncmoe20 497). Decode after it: hybrid 47.9, ncmoe20 39.3 (noisy), cache 35.8. The 2x prefill gain applies ONLY to prompts <= 32 tokens. Do not claim it generally.
- 02:40 Hot list fit matters: on unseen math, original list 41.3, broad list 55.1, cache 42.9; broad list costs heldout (48.0 vs 53.0). tools/sim_lfu.py: frequency-gated swap halves misses over a workload switch (held 26->13%, math 49->17%) with 2.7 uploads per token.
- Built LLAMA_MOE_LFU (margin 2, budget 8 per token, halve every 3000 layers). First run: heldout 56.4 decode, 95.5 prefill, hit 80.6%; output sensible. Matrix running: tools/matrix2.sh, results/gemma/matrix2 (held and math workloads, cold start, 3 reps). Update README after it lands.
- prompts/prompts.jsonl: 24 prompts (code, chat, reasoning), 18 profile, 6 heldout.
- tools/sim_cache.py tested on synthetic iid data only. Finding: "prefetch last token's experts" does nothing because SLRU still holds them. Real prefetch must predict uncached experts (next idea: same-token earlier layers -> this layer). Test on real trace.

## Adaptive results (matrix2, cold start, 3 reps, std dev under 0.4)
Decode tok/s vs best stock at or below same VRAM. Held: 4.6 GB 36.4 -> 43.6; 5.6 GB 38.2 -> 48.6; 6.6 GB 40.2 -> 52.8; 7.6 GB 49.6 -> 56.3. Math (unseen, profile excluded it): 36.5 -> 46.2; 38.5 -> 52.0; 40.7 -> 56.7; 43.3 -> 60.5. Note ncmoe-20 (6669 MiB, 43.1 held in matrix 1) sits just above the 6.6 GB row.

## Verification of adaptive engine, 5.6 GB (tools/verify_lfu.sh)
Quality 56/60 correct, 0 wrong, 4 no parsable answer. Long prompt decode 43.6, prefill 436. Soak 252 prompts, 24192 tokens, no crash, 0 empty, 0 garbled, passes 51.5 / 51.3 / 50.9 tok/s. README updated. Not done: logit test on the adaptive build. Next: benchmark page, demo script, more speed (CPU/GPU overlap).

## 04:30
Logit test on adaptive: cache vs lfu dP 0.128 max 0.52, cmoe vs cache 0.139 max 0.57, same top 24/30. README and page updated. Benchmark page: https://claude.ai/artifact/76x4LFfUTB9YfhFWSZns2i (private until shared). tools/demo.py added.
Tuning rule: tune on profile prompts only, then re-measure once on held-out and math. Do not tune on held-out.

## 05:00
Tuning on profile prompts at 5.6 GB: 48.2-50.2 tok/s across all settings, keep defaults (margin 2, budget 8, halve 3000). tools/demo.py works: 48.5 vs 40.6 (ncmoe22, 5853 MiB) vs 30.8 (cache).
Stress test of baseline started: stock can combine -ncmoe with --moe-cache-mib, and may like more threads. tools/stock_best.sh.

## 06:35
Stock stress test: combined layer split + cache best 43.7 at 7.5 GB; threads 8 to 16 no better; -t 16 collapses to 19. Baseline holds. One run errored (math n22-c900, likely VRAM). README updated.
Next: engine on LFM2.5-24B to test it beyond Gemma. tools/lfm_engine.sh, results/lfm/engine.

## 07:05
LFM engine test (3 reps): lfu 55.3 @6155 vs ncmoe26 50.5 (+9%); lfu 48.3 @5155 vs ncmoe30 45.3 (+6%); lfu 42.0 @4151, cache 18.9. Prefill lower than ncmoe on LFM. README updated. Next: MTP sidecar test.

## 07:15
MTP sidecar (draft-mtp, n=3), single run at 5.9 GB: engine 59.4 tok/s, acceptance about 0.7; stock cache 31.3. Full matrix with MTP on every engine running: tools/mtp_matrix.sh, results/gemma/mtp/m. Do not claim until stock layer split + MTP is measured.

## 09:10 Final state, loop stopped at the user's request
MTP matrix (draft 4, 3 reps) and verification done: 58/60, 0 wrong; soak 252 prompts, 60.6 tok/s, no faults. Late in the run the whole laptop became about 20% slower, cause unknown. Paired runs in that state: 6.9 GB mixed 53.0 vs 36.9 (+44%), 7.5 GB math 87.0 vs ncmoe-20 48.8 (+78%), 7.5 GB mixed 57.9 vs cache 47.4 (+22%). ncmoe-20 runs from that state moved to results/gemma/mtp/slow-state. README and page updated. Nothing committed or pushed.

## Timing run (64 tokens, heldout, 7.6 GB)
ms per graph. Engine (hit 77%, 51.0 tok/s): into GPU 6.50, GPU launch 0.98, into CPU 4.52, CPU compute 9.17, total 21.2. Stock -cmoe: 1.00, 1.02, 5.61, 35.09, total 42.7. CPU compute is the largest engine phase; the earlier 3 ms estimate was wrong. Waits include GPU compute. Timers are in the engine tree (off unless GGML_SCHED_PHASE is set). The stopped command's edit had already reached the source file.

## Overlap (LLAMA_MOE_OVERLAP=1)
CPU cold split runs while GPU hot split runs: its inputs are copied before the GPU launch. Short check, 64 tokens, 7.6 GB: decode 50.0 -> 54.9, prefill 84.3 -> 109.6, output 6/6 identical to overlap off. Laptop may still be in its slower state; compare only paired runs.

## MTP cache fix (LLAMA_MOE_LFU_WEIGHT=1)
Counts each token that uses an expert, scales swap budget and decay by tokens, swaps busiest experts first. Short paired check, 5.9 GB, MTP, overlap on, 64 tokens, 2 reps: 46.1 -> 52.1 tok/s (+13%), expert uses on GPU 54.3% -> 63.0%, swaps 872 -> 2763. Same output as off on 4/6 prompts. Not run through soak or quality yet.
Combined with overlap against the old engine in the same short test: 41.8 -> 52.1 (+25%).

## 12:20 Final benchmark (frozen build, MTP on all, 3 reps, tools/final.sh)
Mixed: 39.1 -> 55.6, 43.3 -> 72.6, 45.6 -> 81.2, 51.3 -> 91.2. Math: 48.9 -> 72.3, 53.3 -> 91.1, 57.9 -> 107.0, 61.2 -> 118.2 (4.9, 5.9, 6.9, 7.5 GB). Quality 58/60, 0 wrong. Soak 24 prompts (split all covers prompts.jsonl only), 71.5 tok/s, no faults. README, PITCH, page updated. Long prompts and logits not rerun on the final build.

## 13:25 No-MTP on final build
2 rounds, stock figure is its faster round. Gains 17-43%. In README and PITCH.
