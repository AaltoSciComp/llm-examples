#!/bin/bash

module load model-huggingface/all

module load scicomp-llm-env

export TRANSFORMERS_OFFLINE=1

python -m vllm.entrypoints.openai.api_server --model "$1"

