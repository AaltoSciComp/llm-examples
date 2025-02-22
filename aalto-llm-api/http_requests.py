## Aalto LLMs API doc: https://llm-gateway.k8s-test.cs.aalto.fi/docs#/
## Aalto LLMs API key: https://llm-gateway.k8s-test.cs.aalto.fi/keys/
## Available models: https://llm-gateway.k8s-test.cs.aalto.fi/v1/models

## the key can be stored in a .env file and load this way: 
import os
from dotenv import load_dotenv
load_dotenv()
mykey = os.environ['MY_KEY']
import requests

# Endpoint where the FastAPI server is listening
url = 'https://llm-gateway.k8s-test.cs.aalto.fi/v1/chat/completions'

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {mykey}'
    }
# Data to be sent to the server, modify the text as needed
data = {
    "model" : "llama3-8b-q8-instruct",
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
