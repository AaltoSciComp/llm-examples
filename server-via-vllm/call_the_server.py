"""OpenAI client for the in-job vLLM server. Env: LLM_API_URL, LLM_MODEL, LLM_API_KEY, MAX_OUTPUT_TOKENS

Requests go out together. vLLM continuous batching schedules whichever
ones are still active. That only happens if the server's --max-num-seqs is > 1.
"""

import asyncio
import os

from openai import AsyncOpenAI

QUESTIONS = [
    "In one short paragraph, what is Aalto Triton?",
    "What is the difference between a CPU and a GPU?",
    "Explain Slurm in two sentences.",
    "Write a one-line Python hello world.",
    "What is vLLM used for?",
]

client = AsyncOpenAI(base_url=os.environ["LLM_API_URL"], api_key=os.environ.get("LLM_API_KEY", "local"))
opts = dict(model=os.environ["LLM_MODEL"], max_tokens=int(os.environ.get("MAX_OUTPUT_TOKENS", "256")))


async def ask(q):
    r = await client.chat.completions.create(messages=[{"role": "user", "content": q}], **opts)
    return q, r.choices[0].message.content


async def main():
    for q, answer in await asyncio.gather(*(ask(q) for q in QUESTIONS)):
        print(f"Q: {q}\nA: {answer}\n")


asyncio.run(main())
