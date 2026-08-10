#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --gpus=1             
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-short 
#SBATCH --output hug_%J.out
#SBATCH --error hug_%J.err

module purge

# Set up environment to use locally stored Hugging Face models
module load model-huggingface/all

# Load Python environment
module load scicomp-llm-env

# Prefer conda's libstdc++ so pyarrow (via transformers) finds GLIBCXX_3.4.31+
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u your_script.py 
