import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_API_VERSION

openai_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_CHAT_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY
)

model = SentenceTransformer("microsoft/deberta-base")

def load_items(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]

def embed_text(items):
    return model.encode(items, show_progress_bar=False)

def compute_tfidf_similarity(clause_texts, chunk_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clause_texts + chunk_texts)
    clause_tfidf = tfidf_matrix[:len(clause_texts)]
    chunk_tfidf = tfidf_matrix[len(clause_texts):]
    similarity_matrix = clause_tfidf * chunk_tfidf.T
    return similarity_matrix.toarray()

def check_clauses(contract_chunks, required_clauses, alpha=0.7, beta=0.3):
    clause_texts = [c["clauseText"].strip().lower() for c in required_clauses]
    chunk_texts = [ch["text"].strip().lower() for ch in contract_chunks]

    chunk_embeddings = embed_text(chunk_texts)
    clause_embeddings = embed_text(clause_texts)
    tfidf_similarities = compute_tfidf_similarity(clause_texts, chunk_texts)

    clause_similarities = {}
    for i, clause_emb in enumerate(clause_embeddings):
        best_score = -1
        best_chunk_index = None
        for j, chunk_emb in enumerate(chunk_embeddings):
            sim = -tf.keras.losses.cosine_similarity(clause_emb, chunk_emb).numpy()
            norm_sim = (sim + 1) / 2
            tfidf_part = tfidf_similarities[i, j]
            combined_score = alpha * norm_sim + beta * tfidf_part
            if combined_score > best_score:
                best_score = combined_score
                best_chunk_index = j
        clause_name = required_clauses[i]["clauseName"]
        clause_similarities[clause_name] = {
            "score": best_score,
            "chunk_index": best_chunk_index,
            "chunk_text": chunk_texts[best_chunk_index]
        }
    return clause_similarities