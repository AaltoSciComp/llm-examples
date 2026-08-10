#!/bin/bash

module load model-huggingface/all

module load scicomp-llm-env

python -m vllm.entrypoints.openai.api_server --model "$1"

