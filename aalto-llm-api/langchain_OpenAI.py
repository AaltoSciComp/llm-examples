## Aalto LLMs API doc: https://llm-gateway.k8s-test.cs.aalto.fi/docs#/
## Aalto LLMs API key: https://llm-gateway.k8s-test.cs.aalto.fi/keys/
## Available models: https://llm-gateway.k8s-test.cs.aalto.fi/v1/models

## the key can be stored in the .env file and load this way: 
import os
from dotenv import load_dotenv
load_dotenv()
my_key = os.getenv("MY_KEY")


import sys
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://llm-gateway.k8s-test.cs.aalto.fi/v1",
    openai_api_key=my_key,
    model_name="llama3-8b-q8-instruct",
)

response = llm.invoke("hello, cute llama!")
print(response)