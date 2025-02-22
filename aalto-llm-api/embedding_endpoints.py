#This Python script is designed to evaluate the embedding endpoints of aalto llm API. Semantic similarity between pairs of sentences are calculated for sanity check.
import os
from dotenv import load_dotenv
load_dotenv()
mykey = os.environ['MY_KEY']
import requests
from scipy.spatial.distance import cosine
def get_embedding(sentence):
    url = 'https://llm-gateway.k8s-test.cs.aalto.fi/v1/embeddings'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {mykey}'
    }
    data = {
        "model": "llama2-7b",
        "input": sentence
    }
    
    try:
        print(f"Sending request for sentence: {sentence}")
        response = requests.post(url, headers=headers, json=data)
        print(f"Response status: {response.status_code}")
        
        # Print more detailed error information for 500 errors
        if response.status_code == 500:
            print("Server returned 500 error. Details:")
            print(f"Request headers: {headers}")
            print(f"Request data: {data}")
            print("Response text:", response.text[:8000] + "..." if len(response.text) > 8000 else response.text)
        
        return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# List of sentence pairs:
# These pairs are mixed; some are similar in meaning (1 and 3) while others are quite different (2, 4, and 5).
sentence_pairs = [
    ("The weather today is sunny and pleasant.", "Today's weather is sunny and enjoyable."),
    ("She is reading a novel in the garden.", "Last night was a peaceful night."),
    ("Technology is advancing rapidly in the 21st century.", "The 21st century is characterized by rapid technological advancements."),
    ("I love to paint landscapes in my free time.", "I went to Italy last summer, it was super hot there."),
    ("The cat is sleeping on the windowsill.", "I play badminton every thursday morning.")
]
# Compute and print the similarity for each pair of sentences to sanity check if the embeddings are meaningful.
for sent1, sent2 in sentence_pairs:
    embedding1 = get_embedding(sent1)
    embedding2 = get_embedding(sent2)
    
    similarity = 1 - cosine(embedding1, embedding2)
    print(f"Similarity between:\n'{sent1}'\nand\n'{sent2}'\nis: {similarity.item():.4f}")

