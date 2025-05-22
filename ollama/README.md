# Ollama
> **NOTE**: This example is no longer maintained and has been deprecated.

This example uses Ollama to deploy a model: https://ollama.ai/


## Get model weights from Triton

In Triton:

```sh

module use /scratch/shareddata/modules/core-models

# Choose the model that we want to use
module load model-llama2/13b-chat

# Choose the llama.cpp model quantization we want to use
module load model-llama.cpp/q4_1-2023-08-28

# Get the path to model weights
echo $MODEL_WEIGHTS
# Example output: /scratch/shareddata/LLMs_tools/models/llama2-llama.cpp-2023-08-28/llama-2-13b-chat/ggml-model-q4_1.gguf
```

On whatever machine you're working on:

```sh
# Copy weights from Triton to your machine:
scp triton.aalto.fi:/scratch/shareddata/LLMs_tools/models/llama2-llama.cpp-2023-08-28/llama-2-13b-chat/ggml-model-q4_1.gguf llama-13b-chat-q4_1.gguf

# Get ollama
curl -L https://ollama.ai/download/ollama-linux-amd64 -o ollama
chmod +x ollama
```

Create a file called `Modelfile` with the following contents:

```dockerfile
FROM ./llama-13b-chat-q4_1.gguf
```

In one terminal, start up ollama:
```sh
./ollama serve
```

In another terminal, give the model to the ollama server:
```sh
./ollama create llama-13b-chat-q4_1 -f Modelfile
```

Run the model:
```sh
./ollama run llama-13b-chat-q4_1
```

For more information on Ollama installation, see [this page](https://github.com/jmorganca/ollama/blob/main/docs/linux.md#download-the-ollama-binary)
