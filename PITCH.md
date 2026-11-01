# Pitch sheet

## 30-second pitch
Modern AI models are too big for the GPUs most people own. Gemma 4 26B needs about 14 GB, and most laptops have 8.

We measured where the time goes and found that on this laptop, moving part of the model to the GPU costs more than computing it on the CPU.

So we built a Gemma-tuned inference engine on top of llama.cpp. It learns which experts your work uses and keeps those on the GPU. The rest run on the CPU at the same time, and it never stops to wait for a transfer.

Same model file, same laptop, about 70% faster than stock llama.cpp at a safe 5.9 GB of VRAM, with matching math accuracy on our test. Up to 93% at the top of the range.

Our goal is an inference engine that fits itself to your hardware. This laptop is the proof.

## What we built
- Hot/cold split inside the Gemma expert graph: GPU and CPU each run part of a layer, then the results are added.
- Adaptive swapping: decayed use counts decide which experts stay on the GPU.
- Background uploads, so decode never waits.
- CPU and GPU overlap: a llama.cpp scheduler change so both run at once.
- MTP cache fix: swaps count every token that uses an expert. The GPU share of expert uses rose from 54% to 63% at 5.9 GB.

## Judge questions
**Did you build an engine from scratch?** No. It is a patch of about 500 lines on llama.cpp. We changed scheduling and expert handling, and reused its kernels.

**Is this new?** Not entirely. Several forks explored hot/cold experts and we credit them in the README. We did not benchmark against them. Our work is the measurements on this hardware, the tuning for Gemma, and the MTP cache fix.

**Why is the CPU better than the GPU for a miss?** This laptop's GPU link measured 8.5 GB/s. Uploading one 3.2 MiB expert takes about 0.38 ms, and using it once on the CPU takes about 0.15 ms.

**Does accuracy change?** Output is not bit-identical, but stock's own CPU and GPU paths also disagree by a similar amount. Math accuracy is 58/60 with 0 wrong answers, the same as stock's cache.

**Does it work on other models?** On LFM2.5-24B an earlier build, without the last two improvements, gained 6-9%, so the method carries over but the size of the gain depends on the model and hardware.

**What about other hardware?** A faster link may favour the stock cache. That is why our next step is an engine that measures the machine first.

**What did not work?** A fixed hot list lost to stock's cache, a next-layer expert predictor was too inaccurate, and upload-on-every-miss lost to frequency-based swaps.
