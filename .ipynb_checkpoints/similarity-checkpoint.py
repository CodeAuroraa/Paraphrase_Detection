import nltk
import string
import spacy
import torch
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import os

# Download NLTK data only if not already present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Initialize models with lazy loading
nlp = None
use_model = None
sbert_model = None

def load_models():
    global nlp, use_model, sbert_model
    if nlp is None:
        try:
            nlp = spacy.load('en_core_web_sm')
        except:
            os.system('python -m spacy download en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
    
    if use_model is None:
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    if sbert_model is None:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Preload on import
download_nltk_data()

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
    load_models()
    return use_model([text])[0]

def get_sbert_embedding(text):
    """Generate embeddings using SBERT."""
    load_models()
    return sbert_model.encode(text)

# Compute similarity between two texts
def compute_similarity(text1, text2):
    emb1 = get_sbert_embedding(text1)  # Changed to use SBERT for consistency
    emb2 = get_sbert_embedding(text2)
    cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
    return round(float(cosine_sim), 4)