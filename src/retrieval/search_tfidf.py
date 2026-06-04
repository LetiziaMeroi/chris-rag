from pathlib import Path
import json
import argparse
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_ROOT = Path("/storage/data/chris-rag")
CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"


def load_chunks(path: Path) -> List[Dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def format_result(chunk: Dict, score: float, rank: int) -> str:
    preview = chunk["text"].replace("\n", " ")
    if len(preview) > 700:
        preview = preview[:700] + "..."

    return (
        f"\n--- Result {rank} | score={score:.4f} ---\n"
        f"File: {chunk['file_name']}\n"
        f"Collection: {chunk['collection']}\n"
        f"Page: {chunk['page']}\n"
        f"Chunk ID: {chunk['chunk_id']}\n"
        f"Text:\n{preview}\n"
    )


def search(query: str, top_k: int = 5):
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    print(f"Query: {query}")
    print(f"Top {top_k} results from {len(chunks)} chunks")

    for rank, idx in enumerate(top_indices, start=1):
        print(format_result(chunks[idx], scores[idx], rank))


def main():
    parser = argparse.ArgumentParser(description="Simple TF-IDF search over PDF chunks.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()
    search(args.query, args.top_k)


if __name__ == "__main__":
    main()
