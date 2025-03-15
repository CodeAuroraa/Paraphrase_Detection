from similarity import get_sbert_embedding
from sklearn.metrics.pairwise import cosine_similarity

def detect_paraphrase(text1, text2, threshold=0.8):
    """Detect if two sentences are paraphrases using SBERT."""
    embedding1 = get_sbert_embedding(text1)
    embedding2 = get_sbert_embedding(text2)
    
    similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
    
    return "Paraphrase" if similarity_score > threshold else "Not a Paraphrase"