#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
import string
import spacy
import torch
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Load NLP Models
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Preprocessing function
def preprocess_text(text):
    """Lowercase, remove punctuation, and filter stopwords."""
    stop_words = set(stopwords.words("english"))
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Function to generate embeddings
def get_use_embedding(text):
    """Generate embeddings using Universal Sentence Encoder (USE)."""
    return use_model([text])[0]

def get_sbert_embedding(text):
    """Generate embeddings using SBERT."""
    return sbert_model.encode(text)

# Compute similarity between two texts
def compute_similarity(text1, text2):
    emb1 = get_use_embedding(text1)
    emb2 = get_use_embedding(text2)
    cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
    return round(float(cosine_sim), 4)


# In[ ]:




