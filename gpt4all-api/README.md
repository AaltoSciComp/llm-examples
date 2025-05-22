# GPT4All
> **NOTE**: This example is no longer maintained and has been deprecated.
## Pre-requisites

This example uses [GPT4All](https://github.com/nomic-ai/gpt4all)
for creating a OpenAI compatible API endpoint locally.

GPT4All needs a Docker to be installed.

On your machine:

```sh
# Clone GPT4All repository
git clone https://github.com/nomic-ai/gpt4all.git

# Copy customized docker-compose.yaml and Dockerfile.buildkit to the gpt4all/gpt4all-api directory
cp docker-compose.yaml gpt4all/gpt4all-api
cp Dockerfile.buildkit gpt4all/gpt4all-api/gpt4all_api

# Go to gpt4all-api subfolder
cd gpt4all/gpt4all-api
```

## Getting the API running

In Triton determine path to the weights:

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

Back on whatever machine you're working on:

```sh
# Copy weights from Triton to your machine:
scp triton.aalto.fi:/scratch/shareddata/LLMs_tools/models/llama2-llama.cpp-2023-08-28/llama-2-13b-chat/ggml-model-q4_1.gguf llama-13b-chat-q4_1.gguf

# Set our model weights as MODEL_BIN

export MODEL_NAME=llama-13b-chat-q4_1.gguf
```

Create the image:
```sh
DOCKER_BUILDKIT=1 docker build -t gpt4all_api --build-arg MODEL_BIN=./$MODEL_NAME --progress plain -f gpt4all_api/Dockerfile.buildkit .
```

Run the `docker-compose.yaml` in this directory.

```sh
docker compose up -d
```

## Testing it

You can test the API endpoint with the environement from the included `environment.yml`.

```sh
conda env create -f environment.yml
```

Test with openai:
```sh
source activate langchain
export MODEL_NAME=llama-13b-chat-q4_1.gguf
python openai_test.py
```

Test with langchain:
```sh
source activate langchain
export MODEL_NAME=llama-13b-chat-q4_1.gguf
python langchain_test.py
```
