import csv
import json
import re
from pathlib import Path

from src.rag.document_answer_service import (
    DocumentAnswerService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "document_questions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "document_rag_results.jsonl"
)


def normalize(text: str) -> str:
    """
    Lightweight normalization for expected-fact matching.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def split_expected(value: str):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split("||")
        if x.strip()
    ]


def get_retrieved_sources(result):
    """
    Collect file names from both retrieval layers.
    """

    retrieval = result.get(
        "retrieval",
        {},
    )

    sources = set()

    for key in [
        "text_results",
        "table_results",
    ]:

        for item in retrieval.get(
            key,
            [],
        ):

            source = item.get(
                "source",
                {},
            )

            file_name = source.get(
                "file_name"
            )

            if file_name:
                sources.add(file_name)

    return sources


def evaluate_fact_hits(
    answer: str,
    expected_facts,
):
    """
    Returns how many expected facts are explicitly
    present in the generated answer.
    """

    normalized_answer = normalize(answer)

    hits = []

    for fact in expected_facts:

        found = (
            normalize(fact)
            in normalized_answer
        )

        hits.append({
            "fact": fact,
            "found": found,
        })

    if expected_facts:
        recall = (
            sum(x["found"] for x in hits)
            / len(expected_facts)
        )
    else:
        recall = 1.0

    return hits, recall


def evaluate_source_hits(
    retrieved_sources,
    expected_sources,
):

    hits = []

    for source in expected_sources:

        found = (
            source
            in retrieved_sources
        )

        hits.append({
            "source": source,
            "found": found,
        })

    if expected_sources:
        recall = (
            sum(x["found"] for x in hits)
            / len(expected_sources)
        )
    else:
        recall = 1.0

    return hits, recall


def has_citation(answer: str) -> bool:
    """
    Detect citations such as [T1], [B3].
    """

    return bool(
        re.search(
            r"\[(?:T|B)\d+(?:[^\]]*)?\]",
            answer,
        )
    )


def classify(
    fact_recall: float,
    source_recall: float,
):
    """
    Simple deterministic evaluation.

    correct:
        all expected facts found and at least one
        expected source retrieved

    partial:
        some expected information found

    incorrect:
        no expected facts found
    """

    if (
        fact_recall == 1.0
        and source_recall > 0
    ):
        return "correct"

    if (
        fact_recall > 0
        or source_recall > 0
    ):
        return "partial"

    return "incorrect"


def main():

    service = DocumentAnswerService()

    rows = []

    with open(
        QUESTIONS_PATH,
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "correct": 0,
        "partial": 0,
        "incorrect": 0,
    }

    total_fact_recall = 0.0
    total_source_recall = 0.0
    citation_count = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as out:

        for row in rows:

            question_id = row["id"]
            question = row["question"]

            expected_facts = split_expected(
                row["expected_facts"]
            )

            expected_sources = split_expected(
                row["expected_sources"]
            )

            print("\n" + "=" * 100)
            print(
                question_id,
                question,
            )
            print("=" * 100)

            result = service.answer(
                question
            )

            answer = result[
                "final_answer"
            ]

            retrieved_sources = (
                get_retrieved_sources(
                    result
                )
            )

            fact_hits, fact_recall = (
                evaluate_fact_hits(
                    answer,
                    expected_facts,
                )
            )

            source_hits, source_recall = (
                evaluate_source_hits(
                    retrieved_sources,
                    expected_sources,
                )
            )

            citation_present = (
                has_citation(answer)
            )

            verdict = classify(
                fact_recall,
                source_recall,
            )

            summary[verdict] += 1

            total_fact_recall += (
                fact_recall
            )

            total_source_recall += (
                source_recall
            )

            if citation_present:
                citation_count += 1

            record = {
                "id": question_id,
                "question": question,
                "language": row["language"],
                "answer": answer,
                "expected_facts": (
                    expected_facts
                ),
                "fact_hits": fact_hits,
                "fact_recall": fact_recall,
                "expected_sources": (
                    expected_sources
                ),
                "retrieved_sources": sorted(
                    retrieved_sources
                ),
                "source_hits": source_hits,
                "source_recall": source_recall,
                "citation_present": (
                    citation_present
                ),
                "verdict": verdict,
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print("\nANSWER")
            print(answer)

            print("\nEVALUATION")
            print(
                f"Fact recall:   "
                f"{fact_recall:.2f}"
            )
            print(
                f"Source recall: "
                f"{source_recall:.2f}"
            )
            print(
                f"Citation:      "
                f"{citation_present}"
            )
            print(
                f"Verdict:       "
                f"{verdict}"
            )

    n = len(rows)

    print("\n" + "=" * 100)
    print("DOCUMENT RAG SUMMARY")
    print("=" * 100)

    print(
        f"Questions: {n}"
    )

    print(
        f"Correct:   "
        f"{summary['correct']}"
    )

    print(
        f"Partial:   "
        f"{summary['partial']}"
    )

    print(
        f"Incorrect: "
        f"{summary['incorrect']}"
    )

    if n:

        print(
            f"Mean fact recall:   "
            f"{total_fact_recall / n:.3f}"
        )

        print(
            f"Mean source recall: "
            f"{total_source_recall / n:.3f}"
        )

        print(
            f"Citation rate:      "
            f"{citation_count / n:.3f}"
        )

    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
