#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from similarity import get_sbert_embedding, compute_similarity
from sklearn.metrics.pairwise import cosine_similarity

def detect_paraphrase(text1, text2):
    """Detect if two sentences are paraphrases using SBERT."""
    embeddings = [get_sbert_embedding(text1), get_sbert_embedding(text2)]
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return "Paraphrase" if similarity_score > 0.8 else "Not a Paraphrase"

