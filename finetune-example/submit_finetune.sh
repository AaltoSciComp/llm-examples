#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=82GB
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH -c 32
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-ellis
#SBATCH --output=llama3-3-test-%J.log

module purge
module load mamba
module load model-huggingface/all

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source activate fine-tune-env

module load gcc/12.3.0 cuda/12.2.1

export TRITON_CACHE_DIR=$PWD/triton-cache-dir

srun accelerate launch --num_processes 1 finetune.py ./data meta-llama/Llama-3.3-70B-Instruct --max_length 1024
