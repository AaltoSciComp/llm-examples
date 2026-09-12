---
name: triton-sbatch-drafting
description: >-
  Drafts Slurm sbatch scripts for Aalto Triton HPC (time, mem, CPUs, GPUs,
  arrays, modules or conda/mamba envs). Use when the user asks for an sbatch
  script, Slurm job, batch job, GPU job, array job, or Triton resource requests.
---

# Triton sbatch drafting

Draft batch scripts for [Triton](https://scicomp.aalto.fi/triton/). Prefer
[quick reference](https://scicomp.aalto.fi/triton/ref/) over generic Slurm lore.
Do not run `sbatch` unless the user explicitly asks.

## Workflow

1. **Classify parallelism** (ask if unclear):
   - Serial → time + mem
   - Embarrassingly parallel → `--array` + `$SLURM_ARRAY_TASK_ID`
   - Shared memory → `--cpus-per-task=N`, one task
   - MPI → `--ntasks` (optionally `--nodes`)
   - GPU → `--gpus=…` plus optional `--gres=…`
2. Ask only for unknowns that change the script (walltime, mem, GPU/VRAM,
   software env, I/O paths). Prefer the user’s existing conda/mamba env when
   they have one; otherwise a central module.
3. Prefer conservative defaults for a first script; right-sizing after a
   test run.
4. Deliver a complete `.sh` + one-line submit command.

## Template

```bash
#!/bin/bash -l
#SBATCH --job-name=JOBNAME
#SBATCH --time=HH:MM:SS
#SBATCH --mem=Ng                 # OR --mem-per-cpu=Ng — never both
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Optional: --cpus-per-task=N | --gpus=1 | --gres=min-vram:40g
# Optional: --array=0-9%50 | --mail-type=END,FAIL | --mail-user=first.last@aalto.fi

set -euo pipefail

# Pick ONE software block:
# A) User conda/mamba env (common)
module load mamba
source activate ENVNAME
# B) Central module
# module load scicomp-python-env
# B') Newer-GPU PyTorch module (needs --gres=min-cuda-cc:80)
# module load scicomp-pytorch-env/2026.1

srun python my_script.py
```

Create the log dir **before** submit (`mkdir -p logs`) — Slurm opens
`--output`/`--error` before the script body runs.
Submit with `sbatch script.sh` — never `bash script.sh` for real jobs
(`#SBATCH` is ignored; work may hit the login node).

## Resource rules

| Goal | Directive |
| --- | --- |
| Walltime | `--time=HH:MM:SS` or `--time=DD-HH` |
| Memory | `--mem=4G` **or** `--mem-per-cpu=2G` (not both) |
| Multithreaded | `--cpus-per-task=N` |
| MPI | `--ntasks=N` |
| GPU | `--gpus=1` or `--gpus=h200:1` |
| Min VRAM / CUDA CC | one `--gres=min-vram:40g,min-cuda-cc:80` |
| Short GPU test ≤30 min | `--partition=gpu-debug` + `--gpus=1` |
| Array | `--array=0-99` or `0-99%50` (cap concurrency); logs `%A_%a` |
| Local storage | `--tmp=100G` → use `/tmp` (gone when job ends) |
| CPU arch pin | `--constraint=milan` only when needed |

**Vague defaults:** serial Python `--time=01:00:00 --mem=4G`; omit `--partition`
unless required; one GPU unless code needs more; Aalto mail only.

## Software & paths

Put activation **inside** the script after `#SBATCH`. Ask which they use:

| Choice | In the script |
| --- | --- |
| Own conda/mamba env | `module load mamba` then `source activate ENV` (not `conda activate`) |
| Central Python | `module load scicomp-python-env` |
| Central PyTorch on H100/H200/B300 | `module load scicomp-pytorch-env/2026.1` + `--gres=min-cuda-cc:80` |

Do not recreate or silently modify the user’s env unless they ask. Data under
`$WRKDIR` (`/scratch/work/$USER`) or `/scratch/DEPT/PROJECT/`, not `$HOME`.
Envs: [conda/mamba](https://scicomp.aalto.fi/triton/apps/conda/). Storage:
[data storage](https://scicomp.aalto.fi/triton/tut/storage/).

## Parallelism checks

- Array: map `$SLURM_ARRAY_TASK_ID`; throttle with `%N`; avoid thousands of
  tiny Lustre-thrashing jobs
- Shared memory: export `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` (or equivalent)
- GPU: always `--gpus`; CPU-only work on GPU nodes is not allowed; match
  PyTorch build / CUDA CC to the GPU generation

## Output format

1. Brief rationale (job type + key resources)
2. Full script
3. Submit / check: `sbatch …`, `slurm q` (after a run: `seff JOBID`)
4. One open question if a critical resource was guessed

## More detail

- Option tables, GPU names, examples: [reference.md](reference.md)
- Docs: [serial](https://scicomp.aalto.fi/triton/tut/serial/),
  [arrays](https://scicomp.aalto.fi/triton/tut/array/),
  [GPUs](https://scicomp.aalto.fi/triton/tut/gpu/),
  [parallel](https://scicomp.aalto.fi/triton/tut/parallel/),
  [monitoring](https://scicomp.aalto.fi/triton/tut/monitoring/),
  [AI agents on HPC](https://scicomp.aalto.fi/triton/usage/ai-agents/)
- Help: [SciComp garage](https://scicomp.aalto.fi/help/garage/)
