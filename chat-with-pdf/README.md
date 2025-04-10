## Chat with your pdf document

This is a demonstration of using LangChain, LangGraph, and Hugging Face with a local LLM to create a PDF document interaction system on Triton - a small RAG (Retrieval-Augmented Generation) based chat interface.

**Set up the environment, run:**

```sh
srun --pty --time=00:30:00 --cpus-per-task=4 --mem=40GB --gpus=1 bash
module load model-huggingface/all
module load scicomp-llm-env
```

**Run the following command to start a chat with your document:**
```sh
python chat_with_pdf.py
```

To close the chat interface, type the word 'exit'.
