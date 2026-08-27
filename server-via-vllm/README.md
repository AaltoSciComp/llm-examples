# Server via vLLM (Triton)

Slurm job: start local vLLM → wait for `/v1/models` → run `call_the_server.py` → exit.

```bash
mkdir -p logs
sbatch submit.sh

tail -f logs/vllm-serve-<jobid>.out
```

Models must already be in the shared HF cache
([LLMs on Triton](https://scicomp.aalto.fi/triton/apps/llms/#huggingface-models)).
Default is `Qwen/Qwen3.8-27B`. Swap the client script or `QUESTIONS` in
`call_the_server.py` for your own workload; env vars are set in `submit.sh`.
