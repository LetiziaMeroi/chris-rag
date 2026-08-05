from pathlib import Path
import csv
import json
import re
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")

CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
EMBEDDINGS_PATH = DATA_ROOT / "processed" / "embeddings_e5_base" / "document_embeddings.npy"

QUESTIONS_PATH = Path("src/evaluation/questions.csv")
OUTPUT_PATH = Path("src/evaluation/hybrid_rrf_e5_base_results.csv")

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 5
RRF_K = 60
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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks(path: Path) -> List[Dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def load_questions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def rank_indices(scores: np.ndarray, candidate_k: int) -> List[int]:
    return list(np.argsort(scores)[::-1][:candidate_k])


def build_rank_map(indices: List[int]) -> Dict[int, int]:
    return {idx: rank for rank, idx in enumerate(indices, start=1)}


def reciprocal_rank_fusion(
    bm25_ranked: List[int],
    embedding_ranked: List[int],
    rrf_k: int = RRF_K,
) -> Dict[int, float]:
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


def find_first_relevant_rank(results: List[Dict], expected_strings: str):
    expected_list = [
        normalize_text(s)
        for s in expected_strings.split("|")
        if s.strip()
    ]

    for rank, item in enumerate(results, start=1):
        text = normalize_text(item["chunk"]["text"])

        for expected in expected_list:
            if expected in text:
                return rank

    return None


def compute_metrics(first_rank):
    if first_rank is None:
        return {
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "mrr": 0.0,
        }

    return {
        "hit_at_1": int(first_rank <= 1),
        "hit_at_3": int(first_rank <= 3),
        "hit_at_5": int(first_rank <= 5),
        "mrr": 1.0 / first_rank,
    }


def main():
    chunks = load_chunks(CHUNKS_PATH)
    questions = load_questions(QUESTIONS_PATH)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded questions: {len(questions)}")
    print(f"RRF k: {RRF_K}")
    print(f"Candidates per retriever: {CANDIDATE_K}")

    # BM25 index
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Embedding index
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)

    rows = []

    for q in questions:
        question = q["question"]
        expected_strings = q["expected_strings"]

        bm25_scores = np.array(bm25.get_scores(tokenize(question)))
        bm25_ranked = rank_indices(bm25_scores, CANDIDATE_K)
        bm25_rank_map = build_rank_map(bm25_ranked)

        query_embedding = model.encode(
            ["query: " + question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        embedding_scores = embeddings @ query_embedding
        embedding_ranked = rank_indices(embedding_scores, CANDIDATE_K)
        embedding_rank_map = build_rank_map(embedding_ranked)

        rrf_scores = reciprocal_rank_fusion(
            bm25_ranked=bm25_ranked,
            embedding_ranked=embedding_ranked,
            rrf_k=RRF_K,
        )

        top_indices = sorted(
            rrf_scores.keys(),
            key=lambda idx: rrf_scores[idx],
            reverse=True,
        )[:TOP_K]

        results = [
            {
                "chunk": chunks[i],
                "rrf_score": float(rrf_scores[i]),
                "bm25_rank": bm25_rank_map.get(i, ""),
                "embedding_rank": embedding_rank_map.get(i, ""),
                "rank": rank,
            }
            for rank, i in enumerate(top_indices, start=1)
        ]

        first_rank = find_first_relevant_rank(results, expected_strings)
        metrics = compute_metrics(first_rank)

        top1 = results[0]["chunk"]

        rows.append({
            "question": question,
            "expected_strings": expected_strings,
            "first_relevant_rank": first_rank if first_rank is not None else "",
            "hit_at_1": metrics["hit_at_1"],
            "hit_at_3": metrics["hit_at_3"],
            "hit_at_5": metrics["hit_at_5"],
            "mrr": metrics["mrr"],
            "top1_file": top1["file_name"],
            "top1_page": top1["page"],
            "top1_chunk_id": top1["chunk_id"],
            "top1_rrf_score": results[0]["rrf_score"],
            "top1_bm25_rank": results[0]["bm25_rank"],
            "top1_embedding_rank": results[0]["embedding_rank"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "expected_strings",
            "first_relevant_rank",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "top1_file",
            "top1_page",
            "top1_chunk_id",
            "top1_rrf_score",
            "top1_bm25_rank",
            "top1_embedding_rank",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    avg_hit1 = sum(r["hit_at_1"] for r in rows) / n
    avg_hit3 = sum(r["hit_at_3"] for r in rows) / n
    avg_hit5 = sum(r["hit_at_5"] for r in rows) / n
    avg_mrr = sum(r["mrr"] for r in rows) / n

    print()
    print(f"Questions: {n}")
    print(f"Hit@1: {avg_hit1:.3f}")
    print(f"Hit@3: {avg_hit3:.3f}")
    print(f"Hit@5: {avg_hit5:.3f}")
    print(f"MRR:   {avg_mrr:.3f}")
    print(f"Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
