import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rag.answer_with_context import (
    MODEL_NAME,
    retrieve,
    rank_answer_sentences,
)

from src.rag.generate_answer_ollama import (
    format_answer_candidates,
    format_evidence,
    build_messages,
    generate_with_ollama,
)


QUESTIONS_PATH = Path("src/evaluation/questions.csv")
OUTPUT_PATH = Path("src/evaluation/ollama_answer_results.csv")

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TOP_K = 3


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_expected_strings(expected_strings: Optional[str]) -> List[str]:
    if expected_strings is None:
        return []

    return [
        normalize_text(s)
        for s in expected_strings.split("|")
        if s.strip()
    ]


def contains_expected_answer(answer: str, expected_strings: Optional[str]) -> bool:
    normalized_answer = normalize_text(answer)
    expected_list = parse_expected_strings(expected_strings)

    if not expected_list:
        return False

    for expected in expected_list:
        if expected in normalized_answer:
            return True

    return False

def candidate_contains_expected(candidate: Dict, expected_strings: Optional[str]) -> bool:
    sentence = candidate.get("sentence", "")
    return contains_expected_answer(sentence, expected_strings)


def compute_candidate_hits(
    answer_candidates: List[Dict],
    expected_strings: Optional[str],
) -> Dict:
    """
    Checks whether the expected answer appears in the extracted answer candidates.

    top_candidate_hit = 1 if A1 contains an expected string.
    candidate_hit_at_3 = 1 if any of A1, A2, A3 contains an expected string.
    """
    if not answer_candidates:
        return {
            "top_candidate_hit": 0,
            "candidate_hit_at_3": 0,
        }

    top_candidate_hit = int(
        candidate_contains_expected(answer_candidates[0], expected_strings)
    )

    candidate_hit_at_3 = int(
        any(
            candidate_contains_expected(candidate, expected_strings)
            for candidate in answer_candidates[:3]
        )
    )

    return {
        "top_candidate_hit": top_candidate_hit,
        "candidate_hit_at_3": candidate_hit_at_3,
    }


def answer_is_only_source(answer: str) -> bool:
    """
    Detects weak answers that mostly contain only file names, page numbers,
    chunk IDs, or citation labels instead of a natural-language answer.
    """
    normalized = normalize_text(answer)

    if not normalized:
        return False

    source_patterns = [
        ".pdf",
        "page ",
        "chunk ",
        "_p",
        "_c",
    ]

    has_source_pattern = any(pattern in normalized for pattern in source_patterns)

    # Remove citation labels such as [A1], [A2], [1]
    without_citations = re.sub(r"\[(A\d+|\d+)\]", "", answer).strip()

    word_count = len(re.findall(r"[A-Za-z]+", without_citations))

    return has_source_pattern and word_count < 12

def contains_citation(answer: str) -> bool:
    """
    Checks whether the final generated answer contains a citation label.
    Examples: [A1], [A2], [1], [2]
    """
    citation_pattern = r"\[(A\d+|\d+)\]"
    return re.search(citation_pattern, answer) is not None


def says_insufficient(answer: str) -> bool:
    normalized = normalize_text(answer)
    return "insufficient" in normalized or "not enough evidence" in normalized


def load_questions(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_columns = {"question", "expected_strings"}
    missing = required_columns - set(reader.fieldnames or [])

    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    clean_rows = []

    for i, row in enumerate(rows, start=2):
        question = row.get("question")
        expected_strings = row.get("expected_strings")

        if question is None or not question.strip():
            print(f"Skipping row {i}: missing question")
            continue

        if expected_strings is None or not expected_strings.strip():
            print(f"Skipping row {i}: missing expected_strings")
            continue

        clean_rows.append({
            "question": question.strip(),
            "expected_strings": expected_strings.strip(),
        })

    return clean_rows


def evaluate_question(
    question: str,
    expected_strings: str,
    sentence_model: SentenceTransformer,
    ollama_model: str,
    top_k: int,
) -> Dict:
    retrieved_results = retrieve(question, top_k=top_k)

    answer_candidates = rank_answer_sentences(
        query=question,
        results=retrieved_results,
        model=sentence_model,
        top_n=3,
    )

    answer_candidates_text = format_answer_candidates(answer_candidates)
    evidence_text = format_evidence(retrieved_results)

    messages = build_messages(
        question=question,
        answer_candidates_text=answer_candidates_text,
        evidence_text=evidence_text,
    )

    final_answer = generate_with_ollama(
        messages=messages,
        model_name=ollama_model,
    )

    candidate_hits = compute_candidate_hits(
        answer_candidates=answer_candidates,
        expected_strings=expected_strings,
    )

    expected_hit = contains_expected_answer(final_answer, expected_strings)
    citation_hit = contains_citation(final_answer)
    insufficient = says_insufficient(final_answer)
    non_empty = bool(final_answer.strip())

    top_candidate = answer_candidates[0] if answer_candidates else None

    if top_candidate is not None:
        top_candidate_sentence = top_candidate["sentence"]
        top_candidate_source = top_candidate["source"]
        top_candidate_chunk_id = top_candidate_source["chunk_id"]
        top_candidate_file = top_candidate_source["file_name"]
        top_candidate_page = top_candidate_source["page"]
    else:
        top_candidate_sentence = ""
        top_candidate_chunk_id = ""
        top_candidate_file = ""
        top_candidate_page = ""

    return {
        "question": question,
        "expected_strings": expected_strings,
        "final_answer": final_answer,
        "expected_hit": int(expected_hit),
        "citation_hit": int(citation_hit),
        "non_empty": int(non_empty),
        "says_insufficient": int(insufficient),
        "top_candidate_hit": candidate_hits["top_candidate_hit"],
        "candidate_hit_at_3": candidate_hits["candidate_hit_at_3"],
        "answer_is_only_source": int(answer_is_only_source(final_answer)),
        "manual_label": "",
        "notes": "",
        "top_candidate_sentence": top_candidate_sentence,
        "top_candidate_file": top_candidate_file,
        "top_candidate_page": top_candidate_page,
        "top_candidate_chunk_id": top_candidate_chunk_id,
    }


def compute_metrics(rows: List[Dict]) -> Dict:
    n = len(rows)

    if n == 0:
        return {
            "num_questions": 0,
            "answer_accuracy": 0.0,
            "citation_rate": 0.0,
            "non_empty_rate": 0.0,
            "insufficient_rate": 0.0,
            "top_candidate_accuracy": 0.0,
            "candidate_hit_at_3": 0.0,
            "only_source_rate": 0.0,
        }

    answer_accuracy = sum(row["expected_hit"] for row in rows) / n
    citation_rate = sum(row["citation_hit"] for row in rows) / n
    non_empty_rate = sum(row["non_empty"] for row in rows) / n
    insufficient_rate = sum(row["says_insufficient"] for row in rows) / n
    top_candidate_accuracy = sum(row["top_candidate_hit"] for row in rows) / n
    candidate_hit_at_3 = sum(row["candidate_hit_at_3"] for row in rows) / n
    only_source_rate = sum(row["answer_is_only_source"] for row in rows) / n

    return {
        "num_questions": n,
        "answer_accuracy": answer_accuracy,
        "citation_rate": citation_rate,
        "non_empty_rate": non_empty_rate,
        "insufficient_rate": insufficient_rate,
        "top_candidate_accuracy": top_candidate_accuracy,
        "candidate_hit_at_3": candidate_hit_at_3,
        "only_source_rate": only_source_rate,
    }


def save_results(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "expected_strings",
        "final_answer",
        "expected_hit",
        "citation_hit",
        "non_empty",
        "says_insufficient",
        "top_candidate_hit",
        "candidate_hit_at_3",
        "answer_is_only_source",
        "manual_label",
        "notes",
        "top_candidate_sentence",
        "top_candidate_file",
        "top_candidate_page",
        "top_candidate_chunk_id",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate local Ollama RAG final answers."
    )
    parser.add_argument("--questions", type=str, default=str(QUESTIONS_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)

    print(f"Loading questions from: {questions_path}")
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    print(f"Loading sentence model: {MODEL_NAME}")
    sentence_model = SentenceTransformer(MODEL_NAME)

    results = []

    for i, row in enumerate(questions, start=1):
        question = row["question"]
        expected_strings = row["expected_strings"]

        print("=" * 80)
        print(f"Question {i}/{len(questions)}")
        print(question)

        try:
            result = evaluate_question(
                question=question,
                expected_strings=expected_strings,
                sentence_model=sentence_model,
                ollama_model=args.model,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            result = {
                "question": question,
                "expected_strings": expected_strings,
                "final_answer": "",
                "expected_hit": 0,
                "citation_hit": 0,
                "non_empty": 0,
                "says_insufficient": 0,
                "top_candidate_hit": 0,
                "candidate_hit_at_3": 0,
                "answer_is_only_source": 0,
                "manual_label": "error",
                "notes": str(e),
                "top_candidate_sentence": "",
                "top_candidate_file": "",
                "top_candidate_page": "",
                "top_candidate_chunk_id": "",
            }

        print("Final answer:")
        print(result["final_answer"])
        print(f"Expected hit: {result['expected_hit']}")
        print(f"Citation hit: {result['citation_hit']}")

        results.append(result)

    save_results(results, output_path)

    metrics = compute_metrics(results)

    print("=" * 80)
    print("OLLAMA RAG ANSWER EVALUATION")
    print("=" * 80)
    print(f"Questions: {metrics['num_questions']}")
    print(f"Answer accuracy: {metrics['answer_accuracy']:.3f}")
    print(f"Citation rate: {metrics['citation_rate']:.3f}")
    print(f"Non-empty rate: {metrics['non_empty_rate']:.3f}")
    print(f"Insufficient rate: {metrics['insufficient_rate']:.3f}")
    print(f"Top candidate accuracy: {metrics['top_candidate_accuracy']:.3f}")
    print(f"Candidate Hit@3: {metrics['candidate_hit_at_3']:.3f}")
    print(f"Only-source answer rate: {metrics['only_source_rate']:.3f}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
