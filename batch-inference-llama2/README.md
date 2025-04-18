# llama2-batch-inference

> **NOTE**: This example is no longer maintained and has been deprecated.

This example shows how to use llama2 for batch inference on triton.

First set up the environment, run:

```sh
module load mamba
mamba env create -f env.yml -p ./llama2env
```

This is the slurm [script](./batch_inference.sh) for sbatch run the inference.

```slurm
 #!/bin/bash
 #SBATCH --time=00:25:00
 #SBATCH --cpus_per_task=4
 #SBATCH --mem=20GB
 #SBATCH --gpus=1
 #SBATCH --output=llama2inference-gpu.%J.out
 #SBATCH --error=llama2inference-gpu.%J.err

  # get the model weights
 module load model-llama2/7b
 echo $MODEL_ROOT
  # Expect output: /scratch/shareddata/dldata/llama-2/llama-2-7b
 echo $TOKENIZER_PATH
  # Expect output: /scratch/shareddata/dldata/llama-2/tokenizer.model
  
  # activate conda environment
 module load mamba
 source activate llama2env/

  # run batch inference
 torchrun --nproc_per_node 1 batch_inference.py \
    --prompts prompts.json \
    --ckpt_dir $MODEL_ROOT \
    --tokenizer_path $TOKENIZER_PATH \
    --max_seq_len 512 --max_batch_size 16
```**Note:**
- The `--nproc_per_node` should be set to the [MP] value for the model you are using. Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |
- Adjust the `--max_seq_len` and `--max_batch_size` parameters according to the hardware.


