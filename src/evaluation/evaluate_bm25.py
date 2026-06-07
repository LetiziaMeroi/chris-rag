from pathlib import Path
import csv
import json
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi


DATA_ROOT = Path("/storage/data/chris-rag")
CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
QUESTIONS_PATH = Path("src/evaluation/questions.csv")
OUTPUT_PATH = Path("src/evaluation/bm25_results.csv")

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

def load_questions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

    
def find_first_relevant_rank(results: List[Dict], expected_string: str):
    expected = expected_string.lower()

    for rank, item in enumerate(results, start=1):
        text = item["chunk"]["text"].lower()
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

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    rows = []

    for q in questions:
        question = q["question"]
        expected_string = q["expected_string"]

        scores = bm25.get_scores(tokenize(question))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:5]

        results = [
            {
                "chunk": chunks[i],
                "score": float(scores[i]),
                "rank": rank,
            }
            for rank, i in enumerate(top_indices, start=1)
        ]

        first_rank = find_first_relevant_rank(results, expected_string)
        metrics = compute_metrics(first_rank)

        top1 = results[0]["chunk"]

        rows.append({
            "question": question,
            "expected_string": expected_string,
            "first_relevant_rank": first_rank if first_rank is not None else "",
            "hit_at_1": metrics["hit_at_1"],
            "hit_at_3": metrics["hit_at_3"],
            "hit_at_5": metrics["hit_at_5"],
            "mrr": metrics["mrr"],
            "top1_file": top1["file_name"],
            "top1_page": top1["page"],
            "top1_chunk_id": top1["chunk_id"],
            "top1_score": results[0]["score"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "expected_string",
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

    print(f"Questions: {n}")
    print(f"Hit@1: {avg_hit1:.3f}")
    print(f"Hit@3: {avg_hit3:.3f}")
    print(f"Hit@5: {avg_hit5:.3f}")
    print(f"MRR:   {avg_mrr:.3f}")
    print(f"Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()