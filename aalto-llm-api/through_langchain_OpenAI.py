## Aalto LLMs API doc: https://llm-gateway.k8s.aalto.fi/docs#/
## Aalto LLMs API key: https://llm-gateway.k8s.aalto.fi/keys/
## Available models: https://llm-gateway.aalto.fi/models

## the key can be stored in the .env file and load this way: 
import os
from dotenv import load_dotenv
load_dotenv()
my_key = os.getenv("MY_KEY")
from langchain.chat_models import ChatOpenAI

url = "https://llm-gateway.k8s.aalto.fi/api/v1"

llm = ChatOpenAI(
    openai_api_base=url,
    openai_api_key=my_key,
    model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic",
)

response = llm.invoke("hello, cute llama!")
print(response)
