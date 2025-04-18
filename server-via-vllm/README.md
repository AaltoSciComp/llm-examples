To start an openai-compatible server on a gpu node via vllm with a local model (a model that is predownloaded to Triton), run
```bash
srun --time=02:00:00 --mem=80G --ntasks=1  --gres=gpu:1 --mail-type=BEGIN --mail-user=username@aalto.fi start-server.sh modelname
```

Note: the modelname is the model identifier on huggingface.

Then checkout the hostname via `squeue --me`

Modify the scritp `call_the_server.py` to use the correct hostname and model name.

Then run the script:

```bash
module load scicomp-llm-env
python call_the_server.py
```
