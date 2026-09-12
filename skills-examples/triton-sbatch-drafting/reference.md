# Triton sbatch reference

Source of truth: [Triton quick reference](https://scicomp.aalto.fi/triton/ref/).
Hardware changes — prefer live docs / `slurm features` over memorized lists.

## Common `#SBATCH` options

| Option | Meaning |
| --- | --- |
| `--time=HH:MM:SS` / `DD-HH` | Walltime |
| `--mem=N` / `--mem-per-cpu=N` | Memory (e.g. `4G`) |
| `--cpus-per-task=N` / `-c` | Shared-memory cores |
| `--ntasks=N` / `-n` | MPI / multi-process |
| `--nodes=N-M` / `-N` | Node count range |
| `--gpus=N` or `TYPE:N` | GPU count / type |
| `--gres=min-vram:NNg` / `min-cuda-cc:NN` | VRAM / compute capability (combine in one `--gres`) |
| `--partition=NAME` | Usually omit; auto from resources |
| `--job-name` / `--output` / `--error` | `%j` `%x` `%A` `%a` |
| `--array=…` | e.g. `0-9`, `1-10,15` |
| `--constraint=` / `--tmp=nnnG` / `--exclusive` | Feature pin / local `/tmp` / whole nodes |
| `--mail-type=` + `--mail-user=` | Aalto email only |

## GPU patterns

| Need | Example |
| --- | --- |
| Any single GPU | `--gpus=1` |
| Named type | `--gpus=h200:1` (`h100`/`a100`/`v100`/`b300`) |
| Min VRAM | `--gpus=1` + `--gres=min-vram:80g` |
| Min CUDA CC | `--gpus=1` + `--gres=min-cuda-cc:80` |
| VRAM + CC | `--gpus=1` + `--gres=min-vram:40g,min-cuda-cc:80` |
| Debug ≤30 min | `--partition=gpu-debug` + `--gpus=1` |
| AMD | `--gpus=1` + `-p gpu-amd` |

- One GPU unless the app supports multi-GPU; no CPU-only jobs on GPU nodes
- Prefer capability (`min-vram` / `min-cuda-cc`) over pinning a model name
- `scicomp-python-env` PyTorch → older GPUs (V100)
- H100/H200/B300 → `scicomp-pytorch-env/2026.1` + `--gres=min-cuda-cc:80`
- Grace-H200 (`--partition=gpu-grace-h200-141g`) is ARM — x86 binaries will not run
- VRAM (approx.): `b300` 288G, `h200` 141G, `h200_3g.71gb` 71G MIG, `h100`/`a100` 80G, `v100` 16/32G
- Live inventory: `slurm features` / [quick ref GPUs](https://scicomp.aalto.fi/triton/ref/)

## Examples

### Serial Python (central module)

```bash
#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --output=ScriptOutput.log

module load scicomp-python-env
srun python /path/to/script.py
```

### Serial Python (own conda/mamba env)

```bash
#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

module load mamba
source activate myenv
srun python /path/to/script.py
```

Use `source activate` / `source deactivate` on Triton — not `conda activate`.

### Shared-memory

```bash
#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

module load scicomp-python-env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python train.py
```

### GPU + VRAM floor (newer GPUs / PyTorch)

```bash
#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=min-vram:40g,min-cuda-cc:80
#SBATCH --output=logs/%x-%j.out

module load scicomp-pytorch-env/2026.1
srun python train_gpu.py
```

For V100-era PyTorch, drop `min-cuda-cc` and use `module load scicomp-python-env`.

### Array

```bash
#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --array=0-29%10
#SBATCH --output=logs/%A_%a.out

srun ./my_application -input input_data_${SLURM_ARRAY_TASK_ID}
```

Create `logs/` before `sbatch`. Rerun failures: `sbatch --array=2,5 script.sh`.

### CUDA binary

```bash
#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --mem=500M
#SBATCH --gpus=1
#SBATCH --output=pi-gpu.out

module load triton/2024.1-gcc cuda/12.2.1
srun ./pi-gpu 1000000
```

## Help

- Docs hub: https://scicomp.aalto.fi/
- Conda/mamba: https://scicomp.aalto.fi/triton/apps/conda/
- Monitoring (`seff`): https://scicomp.aalto.fi/triton/tut/monitoring/
- Garage: https://scicomp.aalto.fi/help/garage/
