from pathlib import Path
import csv
import json
import re
import sys
from typing import List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rag.answer_with_context import retrieve, rank_answer_sentences, MODEL_NAME
from sentence_transformers import SentenceTransformer


QUESTIONS_PATH = Path("src/evaluation/questions.csv")
OUTPUT_PATH = Path("src/evaluation/answer_extraction_results.csv")

TOP_K_RETRIEVAL = 5
TOP_N_ANSWERS = 3


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_questions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_first_relevant_answer_rank(
    answer_candidates: List[Dict],
    expected_strings: str,
):
    expected_list = [
        normalize_text(s)
        for s in expected_strings.split("|")
        if s.strip()
    ]

    for rank, item in enumerate(answer_candidates, start=1):
        sentence = normalize_text(item["sentence"])

        for expected in expected_list:
            if expected in sentence:
                return rank

    return None


def compute_metrics(first_rank):
    if first_rank is None:
        return {
            "answer_hit_at_1": 0,
            "answer_hit_at_3": 0,
            "answer_mrr": 0.0,
        }

    return {
        "answer_hit_at_1": int(first_rank <= 1),
        "answer_hit_at_3": int(first_rank <= 3),
        "answer_mrr": 1.0 / first_rank,
    }


def main():
    questions = load_questions(QUESTIONS_PATH)

    print(f"Loaded questions: {len(questions)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    rows = []

    for q in questions:
        question = q["question"]
        expected_strings = q["expected_strings"]

        retrieved_results = retrieve(
            query=question,
            top_k=TOP_K_RETRIEVAL,
        )

        answer_candidates = rank_answer_sentences(
            query=question,
            results=retrieved_results,
            model=model,
            top_n=TOP_N_ANSWERS,
        )

        first_rank = find_first_relevant_answer_rank(
            answer_candidates=answer_candidates,
            expected_strings=expected_strings,
        )

        metrics = compute_metrics(first_rank)

        if answer_candidates:
            top1 = answer_candidates[0]
            source = top1["source"]

            top1_sentence = top1["sentence"]
            top1_score = top1["score"]
            top1_file = source["file_name"]
            top1_page = source["page"]
            top1_chunk_id = source["chunk_id"]
        else:
            top1_sentence = ""
            top1_score = ""
            top1_file = ""
            top1_page = ""
            top1_chunk_id = ""

        rows.append({
            "question": question,
            "expected_strings": expected_strings,
            "first_relevant_answer_rank": first_rank if first_rank is not None else "",
            "answer_hit_at_1": metrics["answer_hit_at_1"],
            "answer_hit_at_3": metrics["answer_hit_at_3"],
            "answer_mrr": metrics["answer_mrr"],
            "top1_answer_sentence": top1_sentence,
            "top1_answer_score": top1_score,
            "top1_file": top1_file,
            "top1_page": top1_page,
            "top1_chunk_id": top1_chunk_id,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "expected_strings",
            "first_relevant_answer_rank",
            "answer_hit_at_1",
            "answer_hit_at_3",
            "answer_mrr",
            "top1_answer_sentence",
            "top1_answer_score",
            "top1_file",
            "top1_page",
            "top1_chunk_id",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)

    avg_hit1 = sum(r["answer_hit_at_1"] for r in rows) / n
    avg_hit3 = sum(r["answer_hit_at_3"] for r in rows) / n
    avg_mrr = sum(r["answer_mrr"] for r in rows) / n

    print()
    print(f"Questions: {n}")
    print(f"Answer Hit@1: {avg_hit1:.3f}")
    print(f"Answer Hit@3: {avg_hit3:.3f}")
    print(f"Answer MRR:   {avg_mrr:.3f}")
    print(f"Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
