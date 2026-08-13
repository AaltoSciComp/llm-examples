---
name: triton-sbatch-drafting
description: >-
  Drafts Slurm sbatch scripts for Aalto Triton HPC (time, mem, CPUs, GPUs,
  arrays, modules). Use when the user asks for an sbatch script, Slurm job,
  batch job, GPU job, array job, or Triton resource requests.
---

# Triton sbatch script drafting

Example skill under `skills-examples/` for HPC users to **copy** into their agent’s skills/rules directory (Cursor, Claude Code, Codex, etc.). See the parent [skills-examples README](../README.md).

Draft Slurm batch scripts for [Aalto Triton](https://scicomp.aalto.fi/triton/) using current SciComp conventions. Prefer facts from [Triton quick reference](https://scicomp.aalto.fi/triton/ref/) over generic Slurm lore.

## Workflow

1. **Classify the parallel model** (ask if unclear):
   - Serial → time + mem only
   - Embarrassingly parallel → `--array` + `$SLURM_ARRAY_TASK_ID`
   - Shared memory (OpenMP / Python multiprocessing / Matlab pool) → `--cpus-per-task=N`, one task
   - MPI → `--ntasks` (and optionally `--nodes`), not many CPUs per task
   - GPU → `--gpus=…` plus optional `--gres=…`
2. **Ask for unknowns** only when they change the script: walltime, memory, GPU need/VRAM, software (module vs conda), input/output paths.
3. **Prefer conservative defaults**, then tell the user to refine with `seff JOBID` after a test run.
4. **Write a complete `.sh` script**, then a one-line submit command. Do not run `sbatch` unless the user explicitly asks.
5. **Before any submit**, re-check directives against [Triton docs](https://scicomp.aalto.fi/triton/ref/) (agents often invent wrong partitions/GRES).

## Script template

```bash
#!/bin/bash -l
#SBATCH --job-name=JOBNAME
#SBATCH --time=HH:MM:SS
#SBATCH --mem=Ng                 # OR --mem-per-cpu=Ng
#SBATCH --output=logs/%x-%j.out  # %x=job name, %j=job id
#SBATCH --error=logs/%x-%j.err
# Optional: #SBATCH --cpus-per-task=N
# Optional: #SBATCH --gpus=1
# Optional: #SBATCH --gres=min-vram:40g
# Optional: #SBATCH --array=0-9
# Optional: #SBATCH --mail-type=END,FAIL
# Optional: #SBATCH --mail-user=first.last@aalto.fi

set -euo pipefail
mkdir -p logs

module load scicomp-python-env   # adjust; see reference.md

srun python my_script.py
```

Submit with `sbatch script.sh` (never `bash script.sh` for real jobs — that ignores `#SBATCH` and can run on the login node).

## Resource rules (Triton)

| Goal | Directives |
| --- | --- |
| Walltime | `--time=HH:MM:SS` or `--time=DD-HH` |
| Memory (whole job/node) | `--mem=4G` |
| Memory per core | `--mem-per-cpu=2G` |
| Multithreaded | `--cpus-per-task=N` (one task) |
| MPI | `--ntasks=N` (optionally `--nodes`) |
| GPU | `--gpus=1` or `--gpus=h200:1` |
| Min VRAM / CUDA CC | `--gpus=1` and one `--gres=min-vram:40g` (combine with commas: `min-vram:40g,min-cuda-cc:80`) |
| Short GPU test (≤30 min) | `--partition=gpu-debug` + `--gpus=1` |
| Array | `--array=0-99` and use `$SLURM_ARRAY_TASK_ID`; outputs with `%A_%a` |
| Local node scratch | `--tmp=100G` → use `/tmp` |
| CPU arch pin | `--constraint=milan` (etc.; only when needed) |

**Defaults when the user is vague:**

- Serial Python: `--time=01:00:00`, `--mem=4G`
- Do **not** set `--partition` unless required (auto-selected from time/mem/GPU)
- Request **one GPU** unless the code is known to use more
- Prefer `--mem=` over huge `--mem-per-cpu` unless scaling cores
- Mail: Aalto addresses only (`--mail-user=first.last@aalto.fi`)

## Software loading

Put modules **inside** the script (after `#SBATCH`):

- Python (general): `module load scicomp-python-env`
- PyTorch on newer GPUs (e.g. B300): `module load scicomp-pytorch-env/2026.1`
- Own conda/mamba env: `module load mamba` then `source activate ENVNAME` (Triton-specific; not `conda activate`)
- CUDA compile/run: load a `triton/…` software stack + `cuda/…` as needed (`module spider` first)

Work from `$WRKDIR` (`/scratch/work/$USER`) or a project `/scratch/DEPT/PROJECT/` — not `$HOME` for data.

## Parallelism checklist

Confirm the code actually supports the model before requesting resources:

- Array jobs: identical scripts; map `$SLURM_ARRAY_TASK_ID` to files/params; avoid thousands of tiny jobs that thrash Lustre
- Shared memory: `--cpus-per-task`; ensure the app respects `OMP_NUM_THREADS` / equivalent
- GPU: CPU-only work on GPU nodes is **not** allowed; always include `--gpus`
- After runs: `seff JOBID` (and for GPUs `module load seff-gpu; seff JOBID`) — raise/lower mem/time/CPUs from efficiency, not guesses

## Agent / cluster hygiene

From [AI Agents on HPC](https://scicomp.aalto.fi/triton/usage/ai-agents/):

- Draft scripts for the user to review; do not spam `squeue`/`sacct` or submit huge arrays unprompted
- Prefer one array (or fewer larger jobs) over floods of single-task jobs
- Do not run heavy work on login nodes; connect coding agents to `code.triton.aalto.fi` when on Triton
- Scratch is not backed up — warn before destructive cleanup

## Output format

When delivering a script:

1. Brief rationale (job type + key resources)
2. Full script in a fenced `bash` block (or write the file if asked)
3. Submit / monitor commands: `sbatch …`, `slurm q`, `seff JOBID`
4. One open question if a critical resource is still guessed

## Additional resources

- Option tables, GPU names, and more examples: [reference.md](reference.md)
- Canonical docs: [Serial jobs](https://scicomp.aalto.fi/triton/tut/serial/), [Arrays](https://scicomp.aalto.fi/triton/tut/array/), [GPUs](https://scicomp.aalto.fi/triton/tut/gpu/), [Parallel models](https://scicomp.aalto.fi/triton/tut/parallel/), [Quickstart jobs](https://scicomp.aalto.fi/triton/quickstart/jobs/), [Monitoring](https://scicomp.aalto.fi/triton/tut/monitoring/), [scicomp.aalto.fi](https://scicomp.aalto.fi/)
