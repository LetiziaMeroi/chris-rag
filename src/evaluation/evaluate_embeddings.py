from pathlib import Path
import csv
import pickle
import re
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")
EMBEDDINGS_DIR = DATA_ROOT / "processed" / "embeddings"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "document_embeddings.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "document_chunks.pkl"

QUESTIONS_PATH = Path("src/evaluation/questions.csv")
OUTPUT_PATH = Path("src/evaluation/embedding_results.csv")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


def normalize_text(text: str) -> str:
    """
    Normalize text for simple string matching.
    This helps match variants like:
    13,393 / 13 393 / 13393
    """
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_questions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)

    with CHUNKS_PATH.open("rb") as f:
        chunks = pickle.load(f)

    return embeddings, chunks


def find_first_relevant_rank(results: List[Dict], expected_strings: str):
    """
    Find the rank of the first retrieved chunk containing
    at least one expected answer string.
    """
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
    embeddings, chunks = load_index()
    questions = load_questions(QUESTIONS_PATH)

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded chunks: {len(chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    rows = []

    for q in questions:
        question = q["question"]
        expected_strings = q["expected_strings"]

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Since embeddings are normalized, dot product = cosine similarity.
        scores = embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        results = [
            {
                "chunk": chunks[i],
                "score": float(scores[i]),
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
            "top1_score": float(results[0]["score"]),
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
            "top1_score",
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
