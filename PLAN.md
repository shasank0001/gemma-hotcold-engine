# Hackathon plan: Gemma 4 26B-A4B on an 8 GB laptop GPU, faster than stock llama.cpp

## Hardware / constraints
- RTX 5060 Laptop GPU, 8 GB VRAM, Blackwell sm_120, PCIe Gen5 x8 link, CUDA 13 already installed (see ~/Documents/ChatGPT/gemma4b for prior setup scripts).
- Intel Core Ultra 7 255H, 16 threads, mixed P/E cores, AVX2 + AVX-VNNI, no AVX-512.
- 22 GiB usable system RAM (about 15 GiB available at last check). 42 GB free disk. Linux.
- ~2 days total. Comparison must be fair: same GGUF file for stock llama.cpp and our engine.
- Open questions: do judges score raw speed or product/demo? Do the rules allow building on llama.cpp? This plan assumes llama.cpp is allowed and reserves Day 2 afternoon for the demo either way.

## Prior work (read-only reference)
- ~/Documents/ChatGPT/gemma4b: E2B/E4B fork, benchmark harness, benchmark page design in docs/benchmark/.
  - Dense E2B/E4B kernel work gave only +2-8% over llama.cpp.
  - Standalone native engine reached 71.55 tok/s after days of work; stock llama.cpp did 77.59 (not a paired run).
  - Lesson: don't rewrite kernels or the runtime. "Custom engine" here means a custom MoE scheduler on llama.cpp kernels.
- ~/shasank/Deep_learing/gemma/research_gemma4_consumer_gpu: findings_llamacpp.md and a clean llama.cpp checkout at 3f03b17 (no weights, no benchmarks).

## What upstream already does (read from src/llama-moe-cache.cpp at 3f03b17)
- `--moe-cache-mib`: GPU cache for host-resident experts, SLRU eviction, one SLRU per group of same-layout layers on a device.
- Batches of <= 32 tokens: experts run on GPU through a per-layer slot map; misses are uploaded from a copy callback, one layer at a time.
- Larger batches read cached experts, copy the rest, and evict nothing.
- Hit, miss and copy stats are already tracked.
- This is the baseline to beat, not only `--n-cpu-moe`.

## Core idea: profile-guided expert scheduling on upstream's cache
1. Pin: load a per-layer hot expert list from a sidecar profile file into the cache at startup, so there are no warm-up misses and hot experts are never evicted.
2. Prefetch: predict upcoming experts (from the previous token or earlier layers) and upload them ahead of time, to cut the per-layer upload stall.
3. Stretch only: compute misses on CPU instead of uploading. High risk: the graph is fixed before routing results are known, so this is not a small patch.

No GGUF repack and no router permutation: the slot map already provides the indirection, and the same-GGUF rule stays intact.

## Steps
### Tonight
- [ ] Clone current upstream llama.cpp into ./llama.cpp; check llama-moe-cache.cpp still matches the notes above. Build with CUDA, CMAKE_CUDA_ARCHITECTURES=120.
- [ ] Download a Q4 GGUF of gemma-4-26B-A4B-it (~15-17 GB) from a reputable HF repo (ggml-org / unsloth / bartowski). Run in background.
- [ ] Smoke test: generates sensible text. Read expert count and expert-used count from GGUF metadata (top-8 vs top-2 is unconfirmed). Record expert vs non-expert tensor sizes.
- [ ] Quick baseline: best `--n-cpu-moe` vs `--moe-cache-mib` at several sizes, with cache hit rates. Try P-core thread pinning.

### Day 1 morning
- [ ] Full baseline: decode and prefill tok/s for the best stock configs. Watch RAM; avoid swapping.
- [ ] Measure time per token spent in expert uploads vs everything else. This caps what pinning and prefetch can gain.
- [ ] Add a logging patch dumping selected expert ids per layer per token. Collect a trace on varied prompts (code, chat, reasoning). Split into profile and held-out prompts.
- [ ] Trace simulator, matching upstream's shared-slot groups, at the real VRAM budget: stock SLRU, static top-K, pinned + SLRU, prefetch, perfect-foresight upper bound. Report misses and uploaded bytes per token.

### Day 1 ~noon: GO / NO-GO
- [ ] Judge on held-out prompts against stock SLRU at the same VRAM budget. GO if pinning or prefetch cuts uploaded bytes per token by >= ~30% and uploads are a large enough share of token time for that to matter. Else fall back to the E2B multi-agent serving demo in ~/Documents/ChatGPT/gemma4b (already measured at +37% warm throughput).

### Day 1 afternoon/night
- [ ] Profile tool: trace to sidecar hot list.
- [ ] Pinning patch in moe_cache_slru.
- [ ] Prefetch patch, if the simulation supports it.
- [ ] Correctness: logits within atol=rtol=1e-3 of stock llama.cpp on fixed prompts, identical greedy tokens.

### Day 2 morning
- [ ] Benchmark vs best stock config (same GGUF, alternate runs, multiple reps, held-out prompts). Tune hot-set size.
- [ ] Time-boxed extras, reported separately: native NVFP4 (W4A4 on Blackwell) vs Q4_0; EAGLE-3 speculative decoding; CPU misses. Only if the main result is done.

### Day 2 afternoon
- [ ] Live side-by-side demo (tok/s + VRAM + cache hit rate), README, benchmark page, open-source repo (MIT, keep llama.cpp license notices).

## Rules
- Keep stock llama.cpp build untouched as the control; our changes in a separate branch/patch.
- Never benchmark while compiling. Record every result with the command used.
- Report decode and prefill separately; be honest where we don't win.
- Different quantizations (e.g. NVFP4) are separate results, never part of the same-GGUF comparison.
