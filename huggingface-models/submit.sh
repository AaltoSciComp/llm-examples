#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gpus=1             
#SBATCH --partition=gpu-v100-32g #specify gpu partitions 
#SBATCH --output hug.out
#SBATCH --error hug.err

# This will set HF_HOME to /scratch/shareddata/dldata/huggingface-hub-cache
module load model-huggingface/all

# Load Python environment
module load scicomp-llm-env

# Force transformer to load model(s) from local hub instead of download and load model(s) from remote hub.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python -u your_script.py 
