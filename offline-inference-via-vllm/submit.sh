#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gpus=1            
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-short 

#SBATCH --output vllm_%J.out
#SBATCH --error vllm_%J.err

# Set up environment to use locally stored Hugging Face models
module load model-huggingface/all

# Load Python environment
module load scicomp-llm-env

python -u your_script.py
