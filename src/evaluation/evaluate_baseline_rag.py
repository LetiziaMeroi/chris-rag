import csv
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.rag.answer_with_context import (
    MODEL_NAME,
    retrieve,
    rank_answer_sentences,
)

from src.rag.generate_answer_ollama import (
    DEFAULT_OLLAMA_MODEL,
    format_answer_candidates,
    format_evidence,
    build_messages,
    generate_with_ollama,
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
    / "baseline_rag_results.jsonl"
)


def normalize(text: str) -> str:
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


def evaluate_fact_hits(
    answer: str,
    expected_facts,
):
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


def get_retrieved_sources(results):
    sources = set()

    for item in results:

        file_name = item.get(
            "file_name"
        )

        if file_name:
            sources.add(file_name)

    return sources


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
    Baseline citations can look like:

    [A1]
    [A2]
    [1]
    [2]
    [A1, A2]
    """

    return bool(
        re.search(
            r"\[(?:A\d+|\d+)(?:[^\]]*)?\]",
            answer,
        )
    )


def is_insufficient_answer(
    answer: str,
) -> bool:

    normalized = normalize(answer)

    patterns = [
        "retrieved evidence is insufficient",
        "evidence is insufficient",
        "insufficient to answer",
        "not explicitly mentioned in the provided evidence",
        "not explicitly stated in the provided evidence",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def classify(
    fact_recall: float,
    source_recall: float,
):

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

    # Load once for sentence-level ranking.
    #
    # This does not change the baseline algorithm;
    # it only avoids loading this model again here
    # for every evaluation question.
    print(
        f"Loading sentence model: "
        f"{MODEL_NAME}"
    )

    sentence_model = SentenceTransformer(
        MODEL_NAME
    )

    with open(
        QUESTIONS_PATH,
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

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

    answerable_total = 0

    abstention_total = 0
    abstention_correct = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as out:

        for row in rows:

            question_id = row["id"]
            question = row["question"]

            expected_facts = (
                split_expected(
                    row["expected_facts"]
                )
            )

            expected_sources = (
                split_expected(
                    row["expected_sources"]
                )
            )

            answerable = (
                row.get(
                    "answerable",
                    "yes",
                )
                .strip()
                .lower()
                == "yes"
            )

            print(
                "\n"
                + "=" * 100
            )

            print(
                question_id,
                question,
            )

            print(
                "=" * 100
            )

            # ==========================================
            # ORIGINAL BASELINE RETRIEVAL
            # ==========================================

            retrieved_results = retrieve(
                question,
                top_k=5,
            )

            # ==========================================
            # ORIGINAL SENTENCE RANKING
            # ==========================================

            answer_candidates = (
                rank_answer_sentences(
                    query=question,
                    results=retrieved_results,
                    model=sentence_model,
                    top_n=3,
                )
            )

            answer_candidates_text = (
                format_answer_candidates(
                    answer_candidates
                )
            )

            evidence_text = format_evidence(
                retrieved_results
            )

            # ==========================================
            # ORIGINAL BASELINE GENERATION
            # ==========================================

            messages = build_messages(
                question=question,
                answer_candidates_text=(
                    answer_candidates_text
                ),
                evidence_text=evidence_text,
            )

            final_answer = (
                generate_with_ollama(
                    messages=messages,
                    model_name=(
                        DEFAULT_OLLAMA_MODEL
                    ),
                )
            )

            answer = final_answer.strip()

            # ==========================================
            # EVALUATION
            # ==========================================

            retrieved_sources = (
                get_retrieved_sources(
                    retrieved_results
                )
            )

            citation_present = (
                has_citation(answer)
            )

            abstained = (
                is_insufficient_answer(
                    answer
                )
            )

            if answerable:

                answerable_total += 1

                (
                    fact_hits,
                    fact_recall,
                ) = evaluate_fact_hits(
                    answer,
                    expected_facts,
                )

                (
                    source_hits,
                    source_recall,
                ) = evaluate_source_hits(
                    retrieved_sources,
                    expected_sources,
                )

                verdict = classify(
                    fact_recall,
                    source_recall,
                )

                total_fact_recall += (
                    fact_recall
                )

                total_source_recall += (
                    source_recall
                )

                if citation_present:
                    citation_count += 1

            else:

                abstention_total += 1

                fact_hits = []
                source_hits = []

                fact_recall = None
                source_recall = None

                if abstained:
                    verdict = "correct"
                    abstention_correct += 1
                else:
                    verdict = "incorrect"

            summary[verdict] += 1

            record = {
                "id": question_id,
                "category": row.get(
                    "category"
                ),
                "question": question,
                "language": row.get(
                    "language"
                ),
                "answerable": answerable,
                "answer": answer,
                "abstained": abstained,
                "expected_facts": (
                    expected_facts
                ),
                "fact_hits": fact_hits,
                "fact_recall": (
                    fact_recall
                ),
                "expected_sources": (
                    expected_sources
                ),
                "retrieved_sources": sorted(
                    retrieved_sources
                ),
                "source_hits": source_hits,
                "source_recall": (
                    source_recall
                ),
                "citation_present": (
                    citation_present
                ),
                "verdict": verdict,
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

            print("\nANSWER")
            print(answer)

            print(
                "\nEVALUATION"
            )

            if answerable:

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

            else:

                print(
                    f"Abstained:     "
                    f"{abstained}"
                )

            print(
                f"Verdict:       "
                f"{verdict}"
            )

    # ==============================================
    # SUMMARY
    # ==============================================

    n = len(rows)

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BASELINE RAG SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Questions: {n}"
    )

    print(
        f"Answerable questions: "
        f"{answerable_total}"
    )

    print(
        f"Abstention questions: "
        f"{abstention_total}"
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

    if answerable_total:

        print(
            f"Mean fact recall:   "
            f"{total_fact_recall / answerable_total:.3f}"
        )

        print(
            f"Mean source recall: "
            f"{total_source_recall / answerable_total:.3f}"
        )

        print(
            f"Citation rate:      "
            f"{citation_count / answerable_total:.3f}"
        )

    if abstention_total:

        print(
            f"Abstention accuracy: "
            f"{abstention_correct / abstention_total:.3f}"
        )

    print(
        f"\nDetailed results saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
