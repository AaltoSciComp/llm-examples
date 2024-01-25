#This Python script is designed to evaluate the embedding endpoints of aaltogput. Semantic similarity between pairs of sentences are calculated for sanity check.
import requests
from scipy.spatial.distance import cosine
def get_embedding(sentence):
    url = 'https://llm-gateway.k8s-test.cs.aalto.fi/v1/embeddings'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 321'
    }
    data = {
        "model": "llama2-7b",
        "input": sentence
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['data'][0]['embedding']

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

