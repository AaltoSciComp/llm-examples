#!/bin/bash -l
#SBATCH --job-name=vllm-serve
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --gres=min-vram:80g,min-cuda-cc:80
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

module load model-huggingface/all scicomp-llm-env/2026.1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# FlashInfer JIT needs -lcuda; keep caches off $HOME
export XDG_CACHE_HOME="${WRKDIR:-/scratch/work/$USER}/.cache"
mkdir -p "$XDG_CACHE_HOME/cuda-stubs"
ln -sfn /usr/lib64/libcuda.so.1 "$XDG_CACHE_HOME/cuda-stubs/libcuda.so"
export LIBRARY_PATH="$XDG_CACHE_HOME/cuda-stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"

MODEL=${LLM_MODEL:-Qwen/Qwen3.8-27B}
PORT=$((8000 + SLURM_JOB_ID % 1000))

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" --kv-cache-dtype fp8 --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --max-num-seqs 1 &
trap "kill $! 2>/dev/null || true" EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do sleep 5; done

export LLM_API_URL="http://127.0.0.1:$PORT/v1" LLM_MODEL="$MODEL" LLM_API_KEY=local \
  MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-256}"
python call_the_server.py
