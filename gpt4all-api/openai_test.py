import os
import openai


model_name = os.environ["MODEL_NAME"]

openai.api_base = "http://localhost:4891/v1"

openai.api_key = "not needed for a local LLM"

prompt = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
response = openai.Completion.create(
    model=model_name,
    prompt=prompt,
    max_tokens=256,
    temperature=0.7,
    top_p=1,
    n=1,
    echo=True,
    stream=False
)

print(response['choices'][0]['text'])
