## Chat with your pdf document

This is a demo of using a llama.cpp model and langchain to interact with pdf document on triton.

### Set up the environment, run:

```sh
module load mamba
mamba env create -f env.yaml -p ./myenv
source activate ./myenv
```
NOTE: This environment is for CPU usage

### Get model weights 
```sh
# Choose the model that we want to use
module load model-llama2/13b-chat

# Choose the llama.cpp model quantization we want to use
module load model-llama.cpp/q4_1-2023-08-28

# Get the path to model weights
echo $MODEL_WEIGHTS
# Example output: /scratch/shareddata/LLMs_tools/models/llama2-llama.cpp-2023-08-28/llama-2-13b-chat/ggml-model-q4_1.gguf
```

Run the following command to start a chat with your document:
```sh
python chat_with_pdf.py
```

To close the chat interface, type the word 'exit'.
