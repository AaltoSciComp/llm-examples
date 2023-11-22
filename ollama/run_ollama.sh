#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Load ollama
module load ollama

# Load ollama models
module load model-ollama

# Set ollama to use a random port
export OLLAMA_HOST=localhost:$(shuf -i 10000-20000 -n 1)

# Start serving the model and capture ollama server process id
ollama serve &> ollama-server-${SLURM_JOB_ID}.log &
OLLAMA_PID=$!

# Wait for server to start
sleep 30

# Stop ollama server when terminal exists
trap "kill $OLLAMA_PID ; exit" TERM EXIT

# Run a summary job
ollama run llama2:7b-chat "Summarize this file: $(cat run_ollama.sh)"
