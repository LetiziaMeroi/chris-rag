from pathlib import Path
import argparse
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")
EMBEDDINGS_DIR = DATA_ROOT / "processed" / "embeddings"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "document_chunks.pkl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)

    with CHUNKS_PATH.open("rb") as f:
        chunks = pickle.load(f)

    return embeddings, chunks


def make_snippet(text: str, max_chars: int = 800) -> str:
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def search(query: str, top_k: int = 5):
    embeddings, chunks = load_index()

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"Query: {query}")
    print(f"Top {top_k} results from {len(chunks)} chunks")

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]
        score = float(scores[idx])

        print(f"\n--- Result {rank} | score={score:.4f} ---")
        print(f"File: {chunk['file_name']}")
        print(f"Collection: {chunk['collection']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("Text:")
        print(make_snippet(chunk["text"]))


def main():
    parser = argparse.ArgumentParser(description="Semantic search over document chunks.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    search(args.query, args.top_k)


if __name__ == "__main__":
    main()
