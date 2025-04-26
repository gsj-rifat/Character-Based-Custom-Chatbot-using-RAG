# embeddings_utils.py

import numpy as np
import openai

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine"):
    if distance_metric == "cosine":
        return [cosine_similarity(query_embedding, e) for e in embeddings]
    elif distance_metric == "euclidean":
        return [np.linalg.norm(np.array(query_embedding) - np.array(e)) for e in embeddings]

def cosine_similarity(a, b):
    return np.dot(np.array(a), np.array(b)) / (np.linalg.norm(a) * np.linalg.norm(b))
