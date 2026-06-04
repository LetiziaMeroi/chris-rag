from pathlib import Path
import json
import argparse
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi


DATA_ROOT = Path("/storage/data/chris-rag")
CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"


def load_chunks(path: Path) -> List[Dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercase + alphanumeric tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return tokens

def make_snippet(text: str, query: str, window: int = 450) -> str:
    """
    Create a snippet centered around the first matching query token.
    """
    clean = text.replace("\n", " ")
    tokens = tokenize(query)

    positions = []
    lower = clean.lower()

    for token in tokens:
        if len(token) < 3:
            continue
        pos = lower.find(token.lower())
        if pos != -1:
            positions.append(pos)

    if positions:
        center = min(positions)
        start = max(center - window, 0)
        end = min(center + window, len(clean))
        snippet = clean[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(clean):
            snippet = snippet + "..."
        return snippet

    if len(clean) > 900:
        return clean[:900] + "..."

    return clean

def format_result(chunk: Dict, score: float, rank: int, query: str) -> str:
    preview = make_snippet(chunk["text"], query)

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

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    print(f"Query: {query}")
    print(f"Top {top_k} results from {len(chunks)} chunks")

    for rank, idx in enumerate(top_indices, start=1):
        print(format_result(chunks[idx], scores[idx], rank, query))


def main():
    parser = argparse.ArgumentParser(description="BM25 search over PDF chunks.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()
    search(args.query, args.top_k)


if __name__ == "__main__":
    main()
