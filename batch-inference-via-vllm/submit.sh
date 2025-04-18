#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --output vllm-gpu.%J.out
#SBATCH --error vllm-gpu.%J.err

# Set HF_HOME to /scratch/shareddata/dldata/huggingface-hub-cache
module load model-huggingface/all

# Load Python environment
module load scicomp-llm-env

# Force transformer to load model(s) from local hub instead of download and load model(s) from remote hub.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -u your_script.py
