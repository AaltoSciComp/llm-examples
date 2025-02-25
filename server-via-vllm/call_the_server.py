from openai import OpenAI

client = OpenAI(
    base_url="http://gpu42:8000/v1", # replace with the real nodename
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about triton."}
    ],
    extra_body={
        "chat_template": "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    }
)

print(completion.choices[0].message)
