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

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    rows = []

    for q in questions:
        question = q["question"]
        expected_strings = q["expected_strings"]

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

            "rank1_chunk_id": results[0]["chunk"]["chunk_id"],
            "rank1_file": results[0]["chunk"]["file_name"],
            "rank1_page": results[0]["chunk"]["page"],
            "rank1_score": results[0]["score"],

            "rank2_chunk_id": results[1]["chunk"]["chunk_id"],
            "rank2_file": results[1]["chunk"]["file_name"],
            "rank2_page": results[1]["chunk"]["page"],
            "rank2_score": results[1]["score"],

            "rank3_chunk_id": results[2]["chunk"]["chunk_id"],
            "rank3_file": results[2]["chunk"]["file_name"],
            "rank3_page": results[2]["chunk"]["page"],
            "rank3_score": results[2]["score"],

            "rank4_chunk_id": results[3]["chunk"]["chunk_id"],
            "rank4_file": results[3]["chunk"]["file_name"],
            "rank4_page": results[3]["chunk"]["page"],
            "rank4_score": results[3]["score"],

            "rank5_chunk_id": results[4]["chunk"]["chunk_id"],
            "rank5_file": results[4]["chunk"]["file_name"],
            "rank5_page": results[4]["chunk"]["page"],
            "rank5_score": results[4]["score"],
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

            "rank1_chunk_id", "rank1_file", "rank1_page", "rank1_score",
            "rank2_chunk_id", "rank2_file", "rank2_page", "rank2_score",
            "rank3_chunk_id", "rank3_file", "rank3_page", "rank3_score",
            "rank4_chunk_id", "rank4_file", "rank4_page", "rank4_score",
            "rank5_chunk_id", "rank5_file", "rank5_page", "rank5_score",
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