## Aalto LLMs API doc: https://ai-gateway.k8s.aalto.fi/docs#/
## Aalto LLMs API key: https://ai-gateway.k8s.aalto.fi/keys/
## Available models: https://ai-gateway.k8s.aalto.fi/v1/models

## the key can be stored in the .env file and load this way: 
import os
from dotenv import load_dotenv
import requests
load_dotenv()
my_key = os.getenv("MY_KEY")

url = "https://ai-gateway.k8s.aalto.fi/v1/chat/completions"

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {my_key}'
    }
# Data to be sent to the server, modify the text as needed
data = {
    "model" : "depseek-r1-distill-qwen-14b",
    "messages" : [
        {"role" : "system", "content" : "Your are an AI Assistant"},
        {"role" : "user" , "content" : "Tell me about triton."}
    ]
}

# Sending a POST request to the FastAPI server and saving the response
response = requests.post(url=url,headers=headers,json=data)

# Checking if the request was successful
if response.status_code == 200:
    # Outputting the generated text
    print(response.json())
else:
    print(f"Failed to generate text, status code: {response.status_code}")
