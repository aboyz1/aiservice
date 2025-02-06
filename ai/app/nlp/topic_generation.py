import re
import logging
import nltk
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from typing import List

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "facebook/bart-large-cnn"
SENTENCE_MODEL_NAME = "all-mpnet-base-v2"
MAX_INPUT_LENGTH = 5000
STOPWORDS = set()

# Initialize NLP models
def initialize_models():
    """Initialize all required models and download necessary resources."""
    global summarizer, sentence_model, STOPWORDS

    # Load stopwords and tokenizers
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))

    # Load BART summarization model
    try:
        summarizer = pipeline("summarization", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        logging.critical(f"Failed to load BART model: {e}")
        raise

    # Load Sentence Transformer model
    try:
        sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    except Exception as e:
        logging.critical(f"Failed to load Sentence Transformer model: {e}")
        raise

# Text Preprocessing
def preprocess_text(text: str) -> str:
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(filtered_tokens)

# Text Expansion with BART
def expand_input_with_bart(input_text: str) -> str:
    """Summarize and expand input text using the BART model."""
    try:
        prompt = f"Summarize and expand on the key aspects of: {input_text}"
        expanded_text = summarizer(prompt, max_length=150, min_length=50, num_beams=4)[0]['summary_text']
        return expanded_text
    except Exception as e:
        logging.error(f"BART model error: {e}")
        raise

# Extract Topics using Sentence Embeddings and K-Means Clustering
def extract_topics(expanded_text: str, num_topics: int = 3) -> List[str]:
    """Extract topics from expanded text using clustering and keyword extraction."""
    try:
        sentences = nltk.sent_tokenize(expanded_text)
        embeddings = sentence_model.encode(sentences)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        topics = []
        for cluster_id in range(num_topics):
            cluster_sentences = [sentences[i] for i, label in enumerate(labels) if label == cluster_id]
            if cluster_sentences:
                topic_keywords = extract_keywords(cluster_sentences)
                topics.append(topic_keywords)
        return topics
    except Exception as e:
        logging.error(f"Topic extraction error: {e}")
        raise

# Keyword Extraction
def extract_keywords(texts: List[str], num_keywords: int = 3) -> str:
    """Extract the top keywords from a list of sentences."""
    try:
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
        X = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        sums = X.sum(axis=0).tolist()[0]
        ranked_terms = sorted(zip(terms, sums), key=lambda x: x[1], reverse=True)

        keywords = [term for term, _ in ranked_terms[:num_keywords]]
        return " & ".join(keywords) if keywords else "No keywords found"
    except Exception as e:
        logging.error(f"Keyword extraction error: {e}")
        raise

# Main Topic Generation Function
def generate_topics(user_input: str, num_topics: int = 3) -> List[str]:
    """Generate topics from user input."""
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty or whitespace.")
    if len(user_input) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters.")

    try:
        expanded_text = expand_input_with_bart(user_input)
        topics = extract_topics(expanded_text, num_topics=num_topics)
        return topics
    except Exception as e:
        logging.error(f"Unexpected error in topic generation: {e}")
        raise

try:
    initialize_models()
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    raise