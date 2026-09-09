import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "baseline_rag_results.jsonl"
)

DOCLING_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "document_rag_results.jsonl"
)


def load_jsonl(path):
    rows = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            rows[record["id"]] = record

    return rows


def summarize(records):

    answerable = [
        r
        for r in records.values()
        if r.get("answerable", True)
    ]

    unanswerable = [
        r
        for r in records.values()
        if not r.get("answerable", True)
    ]

    fact_values = [
        r["fact_recall"]
        for r in answerable
        if r.get("fact_recall") is not None
    ]

    source_values = [
        r["source_recall"]
        for r in answerable
        if r.get("source_recall") is not None
    ]

    citation_values = [
        1 if r.get("citation_present") else 0
        for r in answerable
    ]

    abstention_values = [
        1 if r.get("abstained") else 0
        for r in unanswerable
    ]

    return {
        "questions": len(records),

        "correct": sum(
            r.get("verdict") == "correct"
            for r in records.values()
        ),

        "partial": sum(
            r.get("verdict") == "partial"
            for r in records.values()
        ),

        "incorrect": sum(
            r.get("verdict") == "incorrect"
            for r in records.values()
        ),

        "fact_recall": (
            sum(fact_values)
            / len(fact_values)
            if fact_values
            else 0.0
        ),

        "source_recall": (
            sum(source_values)
            / len(source_values)
            if source_values
            else 0.0
        ),

        "citation_rate": (
            sum(citation_values)
            / len(citation_values)
            if citation_values
            else 0.0
        ),

        "abstention_accuracy": (
            sum(abstention_values)
            / len(abstention_values)
            if abstention_values
            else 0.0
        ),
    }


def print_metric(
    name,
    baseline,
    docling,
):
    delta = docling - baseline

    print(
        f"{name:22s} "
        f"{baseline:8.3f} "
        f"{docling:8.3f} "
        f"{delta:+8.3f}"
    )


def main():

    baseline = load_jsonl(
        BASELINE_PATH
    )

    docling = load_jsonl(
        DOCLING_PATH
    )

    baseline_summary = summarize(
        baseline
    )

    docling_summary = summarize(
        docling
    )

    print("\n" + "=" * 80)
    print("RAG SYSTEM COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':22s} "
        f"{'Baseline':>8s} "
        f"{'Docling':>8s} "
        f"{'Delta':>8s}"
    )

    print("-" * 80)

    print_metric(
        "Fact recall",
        baseline_summary["fact_recall"],
        docling_summary["fact_recall"],
    )

    print_metric(
        "Source recall",
        baseline_summary["source_recall"],
        docling_summary["source_recall"],
    )

    print_metric(
        "Citation rate",
        baseline_summary["citation_rate"],
        docling_summary["citation_rate"],
    )

    print_metric(
        "Abstention accuracy",
        baseline_summary["abstention_accuracy"],
        docling_summary["abstention_accuracy"],
    )

    print("\nVerdicts")
    print("-" * 80)

    for key in [
        "correct",
        "partial",
        "incorrect",
    ]:

        print(
            f"{key.capitalize():22s} "
            f"{baseline_summary[key]:8d} "
            f"{docling_summary[key]:8d} "
            f"{docling_summary[key] - baseline_summary[key]:+8d}"
        )

    print("\n" + "=" * 80)
    print("QUESTION-BY-QUESTION")
    print("=" * 80)

    all_ids = sorted(
        set(baseline.keys())
        | set(docling.keys())
    )

    for question_id in all_ids:

        old = baseline.get(
            question_id,
            {},
        )

        new = docling.get(
            question_id,
            {},
        )

        old_verdict = old.get(
            "verdict",
            "missing",
        )

        new_verdict = new.get(
            "verdict",
            "missing",
        )

        old_fact = old.get(
            "fact_recall"
        )

        new_fact = new.get(
            "fact_recall"
        )

        def fmt(value):
            if value is None:
                return " n/a"
            return f"{value:.2f}"

        marker = ""

        if (
            old_verdict != new_verdict
            or old_fact != new_fact
        ):
            marker = " <-- changed"

        print(
            f"{question_id}: "
            f"{old_verdict:9s} "
            f"({fmt(old_fact)})"
            f" -> "
            f"{new_verdict:9s} "
            f"({fmt(new_fact)})"
            f"{marker}"
        )


if __name__ == "__main__":
    main()
