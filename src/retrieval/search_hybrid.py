from pathlib import Path
import argparse
import json
import pickle
import re
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")

CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
EMBEDDINGS_DIR = DATA_ROOT / "processed" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_ALPHA = 0.7


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "how", "many", "much",
    "does", "do", "did", "with", "from", "by", "as", "at"
}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def load_chunks(path: Path) -> List[Dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to scale scores between 0 and 1.
    """
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def make_snippet(text: str, max_chars: int = 800) -> str:
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def search(query: str, top_k: int = 5, alpha: float = DEFAULT_ALPHA):
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    # BM25 scores
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_scores = np.array(bm25_scores)

    # Embedding scores
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    embedding_scores = embeddings @ query_embedding

    # Normalize both score types
    bm25_norm = normalize_scores(bm25_scores)
    embedding_norm = normalize_scores(embedding_scores)

    # Hybrid score
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * embedding_norm

    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    print(f"Query: {query}")
    print(f"Top {top_k} results from {len(chunks)} chunks")
    print(f"Alpha: {alpha}  (BM25 weight = {alpha}, embedding weight = {1 - alpha})")

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]

        print(f"\n--- Result {rank} ---")
        print(f"Hybrid score:   {hybrid_scores[idx]:.4f}")
        print(f"BM25 score:     {bm25_norm[idx]:.4f}")
        print(f"Embedding score:{embedding_norm[idx]:.4f}")
        print(f"File: {chunk['file_name']}")
        print(f"Collection: {chunk['collection']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("Text:")
        print(make_snippet(chunk["text"]))


def main():
    parser = argparse.ArgumentParser(description="Hybrid BM25 + embedding retrieval.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)

    args = parser.parse_args()
    search(args.query, args.top_k, args.alpha)


if __name__ == "__main__":
    main()
