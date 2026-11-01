#!/bin/bash
# Resume until the file is complete. Plain HTTP resumes more reliably here than Xet.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1
until hf download ggml-org/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q4_0.gguf --local-dir models; do
    echo "retry $(date)"; sleep 20
done
echo "DOWNLOAD DONE $(date)"
