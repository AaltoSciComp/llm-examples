## Aalto LLMs API doc: https://ai-gateway.k8s.aalto.fi/docs#/
## Aalto LLMs API key: https://ai-gateway.k8s.aalto.fi/keys/
## Available models: https://ai-gateway.k8s.aalto.fi/v1/models

## the key can be stored in the .env file and load this way: 
import os
from dotenv import load_dotenv
load_dotenv()
my_key = os.getenv("MY_KEY")
from langchain.chat_models import ChatOpenAI

url = "https://ai-gateway.k8s.aalto.fi/v1"

llm = ChatOpenAI(
    openai_api_base=url,
    openai_api_key=my_key,
    model_name="depseek-r1-distill-qwen-14b",
)

response = llm.invoke("hello, cute llama!")
print(response)
