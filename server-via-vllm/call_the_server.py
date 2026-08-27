"""OpenAI client for the in-job vLLM server. Env: LLM_API_URL, LLM_MODEL, LLM_API_KEY, MAX_OUTPUT_TOKENS"""

import os
from openai import OpenAI

QUESTIONS = [
    "In one short paragraph, what is Aalto Triton?",
    "What is the difference between a CPU and a GPU?",
    "Explain Slurm in two sentences.",
    "Write a one-line Python hello world.",
    "What is vLLM used for?",
]

client = OpenAI(base_url=os.environ["LLM_API_URL"], api_key=os.environ.get("LLM_API_KEY", "local"))
opts = dict(model=os.environ["LLM_MODEL"], max_tokens=int(os.environ.get("MAX_OUTPUT_TOKENS", "256")))

for q in QUESTIONS:
    r = client.chat.completions.create(messages=[{"role": "user", "content": q}], **opts)
    print(f"Q: {q}\nA: {r.choices[0].message.content}\n")
