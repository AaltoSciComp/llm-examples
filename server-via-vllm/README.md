To start a openai-compatible server via vllm for a local model (a model that is predownloaded to Triton), run:

```bash
module load mamba
mamba env create -f env.yml
```
to create the environment.

Then start the server on a gpu node:
```bash
srun --time=02:00:00 --mem=80G --ntasks=1  --gres=gpu:1 --mail-type=BEGIN --mail-user=username@aalto.fi start-server.sh modelname
```

Note: the modelname is the model identifier on huggingface.

Then checkout the hostname via `squeue --me`

Modify the scritp `call_the_server.py` to use the correct hostname and model name.

Then run the script:

```bash
module load scicomp-llm-envs
python call_the_server.py
```
