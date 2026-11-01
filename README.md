# Adaptive hot/cold expert engine for Gemma 4 26B on an 8 GB GPU

Gemma 4 26B-A4B has about 13 GB of experts, so most of them cannot sit in the VRAM of an 8 GB laptop GPU. Each layer's router picks 8 of 128 experts per token, and only some of those are in VRAM.

llama.cpp handles this in two ways: keep whole layers of experts on the CPU (`--n-cpu-moe`), or cache experts in VRAM and upload the misses (`--moe-cache-mib`). This engine is a small patch on llama.cpp that takes a different route, also explored by other forks (see Related work): **often-used experts stay on the GPU, every other expert runs on the CPU for that token, and decode never waits for an upload.** A background thread swaps an expert in only when it is used more than the least-used expert on the GPU, so the hot set follows the workload.

## Why

Measured on an RTX 5060 Laptop GPU:

- GPU upload speed caps at 8.5 GB/s from pinned memory, and 6.5 GB/s from mapped files.
- With the stock cache on an interim MoE model (LFM2.5-24B-A2B), uploads took about 89% of decode time.
- One Gemma expert costs about 0.38 ms to upload and about 0.15 ms to run once on the CPU (estimated from stock timing runs).

On this hardware, a miss is cheaper to compute than to upload.

## Results

Gemma 4 26B-A4B Q4_0, same GGUF for every run, on an RTX 5060 Laptop GPU. Stock is llama.cpp 0ec4027. Every engine, stock included, uses MTP drafting (`--spec-type draft-mtp`, draft length 4). Each row is the mean of 3 cold starts in alternating runs, on one frozen build. Each row compares against the best stock setting tested at or below that VRAM.

| VRAM | Mixed prompts: stock -> engine | Unseen math prompts: stock -> engine |
|---|---|---|
| 4.9 GB | 39.1 -> 55.6 tok/s (+42%) | 48.9 -> 72.3 tok/s (+48%) |
| 5.9 GB | 43.3 -> 72.6 tok/s (+68%) | 53.3 -> 91.1 tok/s (+71%) |
| 6.9 GB | 45.6 -> 81.2 tok/s (+78%) | 57.9 -> 107.0 tok/s (+85%) |
| 7.5 GB | 51.3 -> 91.2 tok/s (+78%) | 61.2 -> 118.2 tok/s (+93%) |

Mixed prompts are 6 held-out code, chat and reasoning prompts. Math prompts are 30 prompts kept out of the starting profile. The engine starts from the same hot list for both and adapts while running. Engine runs varied by up to 6.4 tok/s between repetitions at the smallest size, and up to 4.9 at 5.9 GB.

Short-prompt prefill also improved: at 5.9 GB it was 98.4 tok/s against 82.7 for `--n-cpu-moe 24` on mixed prompts.

Stock was also tried with `--n-cpu-moe` and `--moe-cache-mib` together and with 8 to 16 threads, on an earlier run. None beat stock's best single-mode settings.

### Without MTP

MTP is a stock feature, so this table shows the engine on its own. Same frozen build, no MTP on any engine, 2 alternating rounds. Stock varied by up to 5.4 tok/s between rounds while the engine stayed within about 1.2, so each stock figure is its faster round, which is the cautious comparison.

| VRAM | Mixed prompts: stock -> engine | Unseen math prompts: stock -> engine |
|---|---|---|
| 4.6 GB | 35.0 -> 41.3 tok/s (+18%) | 36.0 -> 42.5 tok/s (+18%) |
| 5.6 GB | 38.0 -> 46.8 tok/s (+23%) | 36.0 -> 49.8 tok/s (+38%) |
| 6.6 GB | 40.0 -> 52.1 tok/s (+30%) | 40.3 -> 56.1 tok/s (+39%) |
| 7.6 GB | 49.3 -> 57.8 tok/s (+17%) | 43.2 -> 61.8 tok/s (+43%) |

### Second model

On an earlier build, without overlap or the MTP fix, on LFM2.5-24B-A2B Q4_K_M, with a hot list from that model's own profile, decode reached 55.3 tok/s at 6.2 GB against 50.5 for `--n-cpu-moe 26` (+9%), and 48.3 at 5.2 GB against 45.3 for `--n-cpu-moe 30` (+6%). Stock's layer split has better prefill on this model, and the engine varied more between runs (up to 3.6 tok/s).

## Limits

- **Not bit-identical to stock.** Stock also disagrees with itself: its CPU and GPU kernels differed by about 0.14 in top-token probability on 30 test contexts. An earlier adaptive build differed from the stock cache by about 0.13, inside that spread. The final build was not run through this logit test. The original 1e-3 logit target is not reachable by any mixed CPU/GPU setup, including stock.
- **Answer quality:** at 5.9 GB the engine got 58/60 math prompts right, with 0 wrong and 2 with no parsable answer. Stock's cache got 58/60 at 7.6 GB.
- **Stability:** the final build ran a soak of 24 prompts and 2,304 tokens with no crash and no empty or garbled reply. An earlier build ran 252 prompts and 24,192 tokens the same way.
- **Long prompts:** on an earlier build, prefill showed no gain on a 1,948-token prompt (about 500 tok/s for all). Prompts over 32 tokens use llama.cpp's normal path. The final build was not tested on long prompts.
- **One laptop, one model.** Results depend on this GPU's upload speed of 8.5 GB/s. A faster link would favour the stock cache. Other engines such as vLLM or ExLlamaV3 were not tested. EAGLE-3 and NVFP4 were not tested.
- **Absolute speed moves with the laptop's state.** Late in testing every engine became about 20% slower with no process, heat or GPU throttle to explain it (cause not found). Later the speed came back. Compare engines only in alternating runs, as the final table does.
- **7.6 GB leaves no VRAM headroom**, so a browser or desktop using the GPU can make that setting fail. 5.6 GB or 6.6 GB is safer.
- **Tuning used the test prompts indirectly.** The swap settings came from simulating traces that include these prompts. They were not retuned after the matrix.

## How it works

1. `LLAMA_MOE_TRACE` records the experts each layer selects.
2. `tools/make_hot.py` ranks experts by use across all layers.
3. At startup, the top experts load into a GPU bank.
4. For each layer, the GPU runs hot experts and the CPU runs cold ones. Their outputs are added before the router weights apply.
5. With `LLAMA_MOE_LFU=1`, each expert keeps a decayed use count. A cold expert replaces the least-used hot expert when its count is higher by more than 2, at most 8 swaps per token. A background thread does the upload, and the expert stays on the CPU until the upload ends. `LLAMA_MOE_LFU_WEIGHT=1` counts every token that uses an expert, which matters for MTP batches.
6. With `LLAMA_MOE_OVERLAP=1`, the CPU cold step runs while the GPU hot step runs, because its inputs are copied before the GPU starts.

The GPU kernel needs distinct expert ids within a token, so 24 all-zero slots stand in for cold experts on the GPU, and each cold expert gets a free one per token. On the CPU, hot experts use id -1, which outputs zeros.

## Run it

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 0ec4027
git apply ../patches/engine-frozen.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j --target llama-server
```

```bash
LLAMA_MOE_HOT=results/gemma/hot.txt LLAMA_MOE_LFU=1 LLAMA_MOE_LFU_WEIGHT=1 LLAMA_MOE_OVERLAP=1 build/bin/llama-server -m gemma-4-26B-A4B-it-Q4_0.gguf -ngl 99 -cmoe --moe-cache-mib 3000 -lm none -t 6 --spec-type draft-mtp -md mtp-gemma-4-26B-A4B-it-Q4_0.gguf --spec-draft-n-max 4 -ngld 99
```

Reproduce the table with `tools/final.sh` and `tools/matrix_report.py results/gemma/final/m`.

## Related work

This is a patch on [llama.cpp](https://github.com/ggml-org/llama.cpp) and reuses its kernels, scheduler, MTP support and expert cache storage. No fork's code was copied, but others have explored the same ideas, and some reached them first:

- [DavideWasTaken/llama.cpp](https://github.com/DavideWasTaken/llama.cpp): hybrid CPU/GPU expert cache for slow PCIe links, with decayed use counts and an upload budget, tested on an 8 GB GPU. The closest project to this one.
- [ParmesanParty/llama.cpp](https://github.com/ParmesanParty/llama.cpp): adaptive hot experts with a separate CPU branch.
- [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp): static hot experts with CPU and GPU chains running together.
- [neurall/llama.cpp](https://github.com/neurall/llama.cpp/releases): usage-based cache, hardware probe and automatic placement.
- [GenerelSchwerz/llama.cpp](https://github.com/GenerelSchwerz/llama.cpp): hybrid MoE with a configurable share of misses sent to the GPU.
- Research: [Apex](https://arxiv.org/html/2610.23006), [HybriMoE](https://www.researchgate.net/publication/395627980_HybriMoE_Hybrid_CPU-GPU_Scheduling_and_Cache_Management_for_Efficient_MoE_Inference), [KTransformers](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), [MoBiLE](https://arxiv.org/pdf/2510.12357).

These projects use different models and hardware, and none was benchmarked here, so their reported gains cannot be compared with ours.

## License

MIT for the code in this repository, see [LICENSE](LICENSE). The patch modifies llama.cpp, which is MIT licensed; its license is in [LICENSE-llama.cpp](LICENSE-llama.cpp).
