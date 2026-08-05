from pathlib import Path
import argparse
import json
import re
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")

CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
EMBEDDINGS_PATH = DATA_ROOT / "processed" / "embeddings" / "document_embeddings.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
CANDIDATE_K = 100


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "how", "many", "much",
    "does", "do", "did", "with", "from", "by", "as", "at",
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


def make_snippet(text: str, max_chars: int = 800) -> str:
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def rank_indices(scores: np.ndarray, candidate_k: int) -> List[int]:
    return list(np.argsort(scores)[::-1][:candidate_k])


def build_rank_map(indices: List[int]) -> Dict[int, int]:
    """
    Convert ranked list of indices into:
    chunk_index -> rank
    """
    return {idx: rank for rank, idx in enumerate(indices, start=1)}


def reciprocal_rank_fusion(
    bm25_ranked: List[int],
    embedding_ranked: List[int],
    rrf_k: int = DEFAULT_RRF_K,
) -> Dict[int, float]:
    """
    Fuse BM25 and embedding rankings using Reciprocal Rank Fusion.
    """
    scores = {}

    bm25_ranks = build_rank_map(bm25_ranked)
    embedding_ranks = build_rank_map(embedding_ranked)

    all_indices = set(bm25_ranks.keys()) | set(embedding_ranks.keys())

    for idx in all_indices:
        score = 0.0

        if idx in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[idx])

        if idx in embedding_ranks:
            score += 1.0 / (rrf_k + embedding_ranks[idx])

        scores[idx] = score

    return scores


def search(query: str, top_k: int = DEFAULT_TOP_K, rrf_k: int = DEFAULT_RRF_K):
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    # BM25 ranking
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(tokenize(query)))
    bm25_ranked = rank_indices(bm25_scores, CANDIDATE_K)
    bm25_rank_map = build_rank_map(bm25_ranked)

    # Embedding ranking
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    embedding_scores = embeddings @ query_embedding
    embedding_ranked = rank_indices(embedding_scores, CANDIDATE_K)
    embedding_rank_map = build_rank_map(embedding_ranked)

    # RRF
    rrf_scores = reciprocal_rank_fusion(
        bm25_ranked=bm25_ranked,
        embedding_ranked=embedding_ranked,
        rrf_k=rrf_k,
    )

    top_indices = sorted(
        rrf_scores.keys(),
        key=lambda idx: rrf_scores[idx],
        reverse=True,
    )[:top_k]

    print(f"Query: {query}")
    print(f"Top {top_k} results from {len(chunks)} chunks")
    print(f"RRF k: {rrf_k}")
    print(f"Candidates per retriever: {CANDIDATE_K}")

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]

        bm25_rank = bm25_rank_map.get(idx, "")
        embedding_rank = embedding_rank_map.get(idx, "")

        print(f"\n--- Result {rank} ---")
        print(f"RRF score:      {rrf_scores[idx]:.6f}")
        print(f"BM25 rank:      {bm25_rank}")
        print(f"Embedding rank: {embedding_rank}")
        print(f"BM25 raw score: {bm25_scores[idx]:.4f}")
        print(f"Emb raw score:  {embedding_scores[idx]:.4f}")
        print(f"File: {chunk['file_name']}")
        print(f"Collection: {chunk['collection']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("Text:")
        print(make_snippet(chunk["text"]))


def main():
    parser = argparse.ArgumentParser(description="Hybrid RRF retrieval.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)

    args = parser.parse_args()
    search(args.query, args.top_k, args.rrf_k)


if __name__ == "__main__":
    main()
