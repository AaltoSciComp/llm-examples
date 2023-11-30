#!/bin/bash
  #SBATCH --time=00:25:00
  #SBATCH --cpus_per_task=4
  #SBATCH --mem=20GB
  #SBATCH --gres=gpu:1
  #SBATCH --output=llama2inference-gpu.%J.out
  #SBATCH --error=llama2inference-gpu.%J.err

  # get the model weights
  module load model-llama2/7b
  echo $MODEL_ROOT
  # Expect output: /scratch/shareddata/dldata/llama-2/llama-2-7b
  echo $TOKENIZER_PATH
  # Expect output: /scratch/shareddata/dldata/llama-2/tokenizer.model
  
  # activate conda environment
  module load miniconda
  source activate llama2env

  # run batch inference
  torchrun --nproc_per_node 1 batch_inference.py \
    --prompts prompts.json \
    --ckpt_dir $MODEL_ROOT \
    --tokenizer_path $TOKENIZER_PATH \
    --max_seq_len 512 --max_batch_size 16
