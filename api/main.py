from fastapi import FastAPI
from similarity import load_items, check_clauses

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Document Similarity Checker API"}

@app.post("/check_similarity")
def similarity_endpoint():
    master_clauses = load_items("../data/master_contract_clause.jsonl")
    sample_chunks = load_items("../data/chunks.jsonl")
    results = check_clauses(sample_chunks, master_clauses)
    return results